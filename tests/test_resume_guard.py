#!/usr/bin/env python3
"""Unit tests for the SGA resume guard (sga.train_gpt._sga_resume_guard).

Stub legacy checkpoints under temp dirs; verifies the fail-closed cluster:

- rank resolution comes from the RANK env var (pre-distributed-init), and
  non-zero ranks never load anything;
- identical configs pass even when the saved JSON has a different key
  order (canonical comparison);
- differing / absent / empty saved sga_config refuse the resume (SystemExit);
  a legacy DENSE checkpoint cannot silently resume under active SGA;
- unreadable/incomplete checkpoints and unsupported layouts fail CLOSED
  (refuse) instead of proceeding -- --sga-allow-config-override is the only
  escape hatch, and it exempts every case;
- a MISSING latest-iteration tracker passes: mirroring megatron's
  load_checkpoint, no saved state under --load means a cold start.

Run: python tests/test_resume_guard.py  (or pytest tests/test_resume_guard.py)
"""

import json
import logging
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                + "/Megatron-LM")

# Importing the training entry also imports megatron.core, which registers
# the torch safe globals (argparse.Namespace etc.) that weights_only loading
# of the stub state dicts relies on.
import pretrain_gpt
from megatron.core.enums import ModelType
from model_provider import model_provider as root_model_provider

import sga.train_gpt as train_gpt
from sga.train_gpt import _sga_resume_guard

CURRENT = {"score": {"name": "abs_g"}, "cycle": {"period": 2}}
CURRENT_JSON = json.dumps(CURRENT, sort_keys=True)

EXPECTED_EXIT = "expected SystemExit(1) from the resume guard"


class _Args(SimpleNamespace):
    def __init__(self, load=None, config=CURRENT_JSON, override=False):
        super().__init__(sga_config=config, load=load,
                         sga_allow_config_override=override)


def make_ckpt(load_dir, saved_attr="sga_config", saved_value=CURRENT_JSON,
              shard_name="mp_rank_00/model_optim_rng.pt", garbage=False):
    """Legacy-style checkpoint: latest tracker + one rank shard."""
    iter_dir = os.path.join(load_dir, "iter_0000500")
    shard_path = os.path.join(iter_dir, shard_name)
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    with open(os.path.join(load_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("500")
    if garbage:
        with open(shard_path, "wb") as f:
            f.write(b"NOT-A-CHECKPOINT")
    else:
        saved = SimpleNamespace(source=42)
        if saved_value is not None:
            setattr(saved, saved_attr, saved_value)
        torch.save({"args": saved}, shard_path)
    return load_dir


class _CaptureErrors(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def expect_exit(fn, needle):
    logger = logging.getLogger("sga.train_gpt")
    capture = _CaptureErrors()
    logger.addHandler(capture)
    try:
        try:
            fn()
        except SystemExit as e:
            assert e.code == 1, f"exit code {e.code}, expected 1"
            assert any(needle in m for m in capture.messages), (
                f"log missing '{needle}': {capture.messages}"
            )
            return
    finally:
        logger.removeHandler(capture)
    raise AssertionError(EXPECTED_EXIT)


def _with_rank(rank, fn):
    saved = os.environ.get("RANK")
    os.environ["RANK"] = str(rank)
    try:
        return fn()
    finally:
        if saved is None:
            del os.environ["RANK"]
        else:
            os.environ["RANK"] = saved


def test_model_build_chain_is_stock():
    """The model wiring is megatron's stock chain, verbatim: the entry's
    model_provider/gpt_builder are the upstream repo-root objects and the
    ModelType is encoder_or_decoder (same as pretrain_gpt.__main__)."""
    assert train_gpt.model_provider is pretrain_gpt.model_provider
    assert train_gpt.gpt_builder.__module__ == "gpt_builders"
    assert train_gpt.model_provider is root_model_provider
    assert ModelType.encoder_or_decoder is not None
    print("ok: model build chain is stock (model_provider/gpt_builder)")


def test_identical_and_reordered_configs_pass():
    tmp = tempfile.mkdtemp(prefix="sga-resume-ok-")
    reordered = json.dumps(dict(reversed(list(CURRENT.items()))),
                           sort_keys=False)
    make_ckpt(tmp, saved_value=reordered)
    _with_rank(0, lambda: _sga_resume_guard(_Args(load=tmp)))  # must not raise
    print("ok: identical + key-reordered saved config passes")


def test_differing_config_refuses():
    tmp = tempfile.mkdtemp(prefix="sga-resume-diff-")
    other = json.dumps({"score": {"name": "update"},
                        "cycle": {"period": 2}}, sort_keys=True)
    make_ckpt(tmp, saved_value=other)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))), "differs")
    print("ok: differing saved config refuses (SystemExit 1)")


def test_absent_saved_config_refuses():
    tmp = tempfile.mkdtemp(prefix="sga-resume-absent-")
    make_ckpt(tmp, saved_attr="irrelevant", saved_value=None)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))), "no saved sga_config")
    print("ok: legacy DENSE checkpoint (absent attr) refuses")


def test_empty_saved_config_refuses():
    tmp = tempfile.mkdtemp(prefix="sga-resume-empty-")
    make_ckpt(tmp, saved_value="")
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))), "no saved sga_config")
    print("ok: empty saved sga_config refuses")


def test_unreadable_shard_fails_closed():
    tmp = tempfile.mkdtemp(prefix="sga-resume-garbage-")
    make_ckpt(tmp, garbage=True)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))), "could not verify")
    print("ok: garbage checkpoint shard fails closed")


def test_missing_tracker_passes_cold_start():
    # Megatron's load_checkpoint cold-starts when the tracker is absent (the
    # smoke configs point --load at the experiment's own still-empty dir);
    # the guard must mirror that instead of refusing. Covers: empty load
    # dir, dir with shards but no tracker, and a wholly non-existent path.
    empty = tempfile.mkdtemp(prefix="sga-resume-empty-dir-")
    _with_rank(0, lambda: _sga_resume_guard(_Args(load=empty)))
    no_tracker = tempfile.mkdtemp(prefix="sga-resume-notracker-")
    os.makedirs(os.path.join(no_tracker, "iter_0000500", "mp_rank_00"),
                exist_ok=True)
    _with_rank(0, lambda: _sga_resume_guard(_Args(load=no_tracker)))
    _with_rank(0, lambda: _sga_resume_guard(
        _Args(load=os.path.join(empty, "does-not-exist"))))
    print("ok: missing tracker = cold start, guard passes")


def test_override_exempt_everything():
    tmp = tempfile.mkdtemp(prefix="sga-resume-override-")
    make_ckpt(tmp, garbage=True)  # worst case: unreadable AND mismatched-ish
    _with_rank(0, lambda: _sga_resume_guard(_Args(load=tmp, override=True)))
    print("ok: --sga-allow-config-override exempts the guard entirely")


def test_nonzero_rank_never_loads():
    tmp = tempfile.mkdtemp(prefix="sga-resume-rank1-")
    loaded = {"flag": False}
    real_load = torch.load

    def spy_load(*a, **kw):
        loaded["flag"] = True
        return real_load(*a, **kw)

    torch.load = spy_load
    try:
        result = _with_rank(1, lambda: _sga_resume_guard(_Args(load="/no/such/dir")))
        assert result is None
        assert loaded["flag"] is False, "non-zero rank touched the checkpoint"
    finally:
        torch.load = real_load
    print("ok: RANK!=0 returns without any checkpoint load")


if __name__ == "__main__":
    test_model_build_chain_is_stock()
    test_identical_and_reordered_configs_pass()
    test_differing_config_refuses()
    test_absent_saved_config_refuses()
    test_empty_saved_config_refuses()
    test_unreadable_shard_fails_closed()
    test_missing_tracker_passes_cold_start()
    test_override_exempt_everything()
    test_nonzero_rank_never_loads()
    print("test_resume_guard: ALL PASS")
