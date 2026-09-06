#!/usr/bin/env python3
"""Unit tests for the SGA resume guard (sga.train_gpt._sga_resume_guard).

Stub legacy checkpoints under temp dirs; verifies the fail-closed cluster:

- rank resolution comes from the RANK env var (pre-distributed-init), and
  non-zero ranks never load anything;
- identical configs pass (RAW comparison, defaults-filled: an older
  checkpoint missing a newer key compares against its default — forward
  compatible);
- a DIFFERENT SPELLING of the same cycle anchor (percent vs absolute
  start_iter) refuses: the raw compare is deliberately spelling-faithful
  (deliberately), and a percent spelling anchors to its side's own
  train_iters, which rides the comparison;
- differing / absent saved sga arguments refuse the resume (SystemExit);
  a legacy DENSE checkpoint cannot silently resume under active SGA, a
  JSON-era checkpoint (retired --sga-config string) is refused with a
  pointed message, and so is a flatargs-era v1 namespace (the same
  string alongside the all-None sga_* attrs its default=None flags
  saved);
- a checkpoint saved before a parameter rename carries retired keys; the
  guard detects them directly and refuses with a pointed message instead
  of letting the defaults-fill swallow a config the checkpoint never ran
  with;
- unreadable/incomplete checkpoints and unsupported layouts fail CLOSED
  (refuse) instead of proceeding -- --sga-allow-config-override is the only
  escape hatch, and it exempts every case;
- a MISSING latest-iteration tracker passes: mirroring megatron's
  load_checkpoint, no saved state under --load means a cold start.

Run: python tests/test_resume_guard.py  (or pytest tests/test_resume_guard.py)
"""

import logging
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The megatron source: MEGATRON_SGA_DIR first (a dev worktree has no
# Megatron-LM sibling — env.sh's first-priority knob, same rule as
# test_config_flat), else the main checkout's sibling.
sys.path.insert(0, os.environ.get("MEGATRON_SGA_DIR")
                or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                + "/Megatron-LM")

# Importing the training entry also imports megatron.core, which registers
# the torch safe globals (argparse.Namespace etc.) that weights_only loading
# of the stub state dicts relies on.
import pretrain_gpt
from megatron.core.enums import ModelType
from model_provider import model_provider as root_model_provider

import sga.train_gpt as train_gpt
from sga.train_gpt import _sga_resume_guard

CURRENT = {
    "sga_enabled": True, "sga_cycle_start_iter": 20, "sga_cycle_period": 2,
    "sga_score": "abs_g", "sga_selection_method": "exact",
    "sga_selection_ratio_percent": 10.0, "sga_optimizer": "adamw",
    "sga_backend": "torch_pipelined_index",
}

# What a flatargs-era v1 run saved alongside its sga_config string: every
# flag registered with default=None, so the attrs exist but carry None.
V1_FLAT_DEFAULTS = {
    "sga_enabled": None, "sga_shadow": None, "sga_score": None,
    "sga_selection": None, "sga_selection_method": None,
    "sga_selection_ratio_percent": None, "sga_optimizer": None,
    "sga_backend": None, "sga_cycle_period": None,
}

EXPECTED_EXIT = "expected SystemExit(1) from the resume guard"


class _Args(SimpleNamespace):
    def __init__(self, load=None, config=None, override=False,
                 train_iters=5000):
        base = dict(CURRENT if config is None else config)
        super().__init__(load=load, sga_allow_config_override=override,
                         train_iters=train_iters, **base)


def make_ckpt(load_dir, saved_config=CURRENT, legacy_json=None,
              shard_name="mp_rank_00/model_optim_rng.pt", garbage=False,
              legacy_attrs=None, saved_train_iters=5000):
    """Flat-era checkpoint: latest tracker + one rank shard carrying the
    saved args namespace. ``legacy_json`` simulates a pre-flat checkpoint
    (only the retired sga_config string, no sga_* attributes);
    ``legacy_attrs`` adds the all-None sga_* attrs a flatargs-era v1 run
    saved alongside that string. ``saved_train_iters`` rides the saved
    namespace like megatron's full-args checkpointing does."""
    iter_dir = os.path.join(load_dir, "iter_0000500")
    shard_path = os.path.join(iter_dir, shard_name)
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    with open(os.path.join(load_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("500")
    if garbage:
        with open(shard_path, "wb") as f:
            f.write(b"NOT-A-CHECKPOINT")
    else:
        saved = SimpleNamespace(source=42, train_iters=saved_train_iters)
        if legacy_json is not None:
            saved.sga_config = legacy_json
            for key, value in (legacy_attrs or {}).items():
                setattr(saved, key, value)
        elif saved_config is not None:
            for key, value in saved_config.items():
                setattr(saved, key, value)
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


def test_identical_and_spelling_variant_configs_pass():
    tmp = tempfile.mkdtemp(prefix="sga-resume-ok-")
    make_ckpt(tmp)  # exact same flat attrs
    _with_rank(0, lambda: _sga_resume_guard(_Args(load=tmp)))  # must not raise
    # the same non-default spelling on BOTH sides still passes (raw
    # compare, both sides carry identical attrs): percent + int ratio
    variant = dict(CURRENT)
    del variant["sga_cycle_start_iter"]
    variant["sga_cycle_start_iter_percent"] = 0.4  # of 5000 -> 20
    variant["sga_selection_ratio_percent"] = 10    # int spelling
    tmp2 = tempfile.mkdtemp(prefix="sga-resume-ok2-")
    make_ckpt(tmp2, saved_config=variant)
    _with_rank(0, lambda: _sga_resume_guard(
        _Args(load=tmp2, config=variant)))
    print("ok: identical + same-spelling saved configs pass")


def test_cross_spelling_sides_refuse_and_train_iters_drift_refuses():
    """Raw compare: percent-vs-absolute across SIDES refuses (a
    different spelling is a different config — the percent side also
    carries its own train_iters anchor), and a percent spelling saved
    under train_iters=5000 refuses when the resume extends train_iters
    (the effective cycle anchor would move 20 -> 40; fail-closed, not
    re-anchored)."""
    abs_config = dict(CURRENT)  # sga_cycle_start_iter: 20
    pct_config = dict(CURRENT)
    del pct_config["sga_cycle_start_iter"]
    pct_config["sga_cycle_start_iter_percent"] = 0.4  # of 5000 -> 20
    # (a) saved absolute, resumed percent: different spelling -> refuse
    tmp = tempfile.mkdtemp(prefix="sga-resume-xabs-")
    make_ckpt(tmp, saved_config=abs_config)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(
            _Args(load=tmp, config=pct_config))), "differs")
    # (b) reverse direction: saved percent, resumed absolute -> refuse
    tmp2 = tempfile.mkdtemp(prefix="sga-resume-xpct-")
    make_ckpt(tmp2, saved_config=pct_config)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(
            _Args(load=tmp2, config=abs_config))), "differs")
    # (c) same percent spelling on both sides, but the saved namespace's
    # own train_iters=5000 vs the resume's 10000: refuse with "differs"
    tmp3 = tempfile.mkdtemp(prefix="sga-resume-tidrift-")
    make_ckpt(tmp3, saved_config=pct_config, saved_train_iters=5000)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(
            _Args(load=tmp3, config=pct_config, train_iters=10000))),
        "differs")
    print("ok: cross-spelling sides refuse; changed train_iters refuses")


def test_differing_config_refuses():
    tmp = tempfile.mkdtemp(prefix="sga-resume-diff-")
    other = dict(CURRENT, sga_score="update")
    make_ckpt(tmp, saved_config=other)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))), "differs")
    print("ok: differing saved config refuses (SystemExit 1)")


def test_retired_key_checkpoint_refuses_pointedly():
    """A checkpoint saved before a parameter rename carries keys no
    current flag declares (here the slice params' pre-rename spelling).
    The defaults-fill would only ever compare them as an extra-vs-missing
    difference; the guard detects the retired keys directly and refuses
    with the pointed message."""
    tmp = tempfile.mkdtemp(prefix="sga-resume-renamed-")
    make_ckpt(tmp, saved_config=dict(CURRENT, sga_backend_slices=3))
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))),
        "retired sga keys")
    print("ok: pre-rename checkpoint (retired key) refused pointedly")


def test_absent_saved_config_refuses():
    tmp = tempfile.mkdtemp(prefix="sga-resume-absent-")
    make_ckpt(tmp, saved_config=None)
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))),
        "no saved sga arguments")
    print("ok: legacy DENSE checkpoint (no sga attrs) refuses")


def test_json_era_checkpoint_refuses_pointedly():
    tmp = tempfile.mkdtemp(prefix="sga-resume-legacy-")
    make_ckpt(tmp, legacy_json='{"cycle": {"period": 2}}')
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))),
        "predates the flat sga arguments")
    print("ok: JSON-era checkpoint refused with the pointed message")


def test_v1_flatargs_era_checkpoint_refuses_pointedly():
    """A flatargs-era v1 checkpoint carries BOTH the retired sga_config
    string and all-None sga_* attrs (its flags registered default=None):
    the legacy string alone must fire the pointed refusal (a v2
    namespace never carries the string)."""
    tmp = tempfile.mkdtemp(prefix="sga-resume-legacy-v1flat-")
    make_ckpt(tmp, legacy_json='{"cycle": {"period": 2}}',
              legacy_attrs=dict(V1_FLAT_DEFAULTS))
    expect_exit(lambda: _with_rank(
        0, lambda: _sga_resume_guard(_Args(load=tmp))),
        "predates the flat sga arguments")
    print("ok: flatargs-era v1 checkpoint (sga_config + None attrs) "
          "refused with the pointed message")


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




def test_always_sync_layers_resume_legs():
    """The always-sync axis rides the raw comparison: a checkpoint saved
    WITHOUT the key refuses a current config that sets it (fail-closed),
    and the raw compare is spelling-faithful — a comma token vs a list
    (or a whitespace variant) are DIFFERENT values on the wire and
    refuse; the exact same spelling passes. (Production namespaces carry
    the argparse comma token on both sides; the list spelling exists only
    on hand-crafted or yaml-derived saved namespaces.)"""
    with tempfile.TemporaryDirectory() as tmp:
        make_ckpt(tmp)  # CURRENT: no sga_always_sync_layers attr
        current = dict(CURRENT, sga_always_sync_layers="embed,norm")
        expect_exit(lambda: _with_rank(
            0, lambda: _sga_resume_guard(_Args(load=tmp,
                                               config=current))), "differs")
    with tempfile.TemporaryDirectory() as tmp:
        saved = dict(CURRENT, sga_always_sync_layers="embed,norm")
        make_ckpt(tmp, saved_config=saved)
        _with_rank(0, lambda: _sga_resume_guard(
            _Args(load=tmp, config=saved)))  # same spelling: passes
        # a different pattern list still refuses
        expect_exit(lambda: _with_rank(
            0, lambda: _sga_resume_guard(
                _Args(load=tmp,
                      config=dict(CURRENT,
                                  sga_always_sync_layers="norm")))),
            "differs")
        # and the list spelling of the SAME value refuses too (raw wire
        # compare: "embed,norm" != ["embed", "norm"])
        expect_exit(lambda: _with_rank(
            0, lambda: _sga_resume_guard(
                _Args(load=tmp,
                      config=dict(CURRENT,
                                  sga_always_sync_layers=["embed", "norm"])))),
            "differs")
    with tempfile.TemporaryDirectory() as tmp:
        # identical whitespace spelling on both sides passes
        saved = dict(CURRENT, sga_always_sync_layers="embed, norm")
        make_ckpt(tmp, saved_config=saved)
        _with_rank(0, lambda: _sga_resume_guard(
            _Args(load=tmp,
                  config=dict(CURRENT, sga_always_sync_layers="embed, norm"))))
    print("ok: always-sync resume legs (absent key refuses, raw spellings)")


if __name__ == "__main__":
    test_model_build_chain_is_stock()
    test_always_sync_layers_resume_legs()
    test_identical_and_spelling_variant_configs_pass()
    test_cross_spelling_sides_refuse_and_train_iters_drift_refuses()
    test_differing_config_refuses()
    test_retired_key_checkpoint_refuses_pointedly()
    test_absent_saved_config_refuses()
    test_json_era_checkpoint_refuses_pointedly()
    test_v1_flatargs_era_checkpoint_refuses_pointedly()
    test_unreadable_shard_fails_closed()
    test_missing_tracker_passes_cold_start()
    test_override_exempt_everything()
    test_nonzero_rank_never_loads()
    print("test_resume_guard: ALL PASS")
