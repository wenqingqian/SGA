#!/usr/bin/env python3
"""Unit tests for sga.config validation (run: python tests/test_config.py)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config import validate_config, validate_sga_section
from sga.launcher.backend_megatron import _get_args_megatron

BASE = """
experiment:
  exp_name: t
  exp_dir: PLACEHOLDER_EXP_DIR
  task: {type: train, backend: megatron, entrypoint: ./sga/train_gpt.py}
  runner: {type: ssh}
sga:
  score: {enabled: false}
  selection: {enabled: false}
  error_feedback: {enabled: false}
  backend: {name: dense}
train:
  system: {tensor_model_parallel_size: 1}
  model: {num_layers: 2}
  data: {data_path: /tmp/x}
"""


def make_base():
    """A fresh BASE config with exp_dir in a private temp dir (auto-cleaned)."""
    tmp = tempfile.mkdtemp(prefix="sga-test-exp-")
    cfg = OmegaConf.create(BASE)
    cfg.experiment.exp_dir = os.path.join(tmp, "exp")
    return cfg, tmp


def expect_error(fn, needle):
    try:
        fn()
    except (ValueError, AssertionError) as e:
        assert needle in str(e), f"error missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


def test_all_off_dense_ok():
    cfg, _ = make_base()
    validate_config(cfg)
    print("ok: all-off dense config validates")


def test_enabled_field_named_in_error():
    cfg, _ = make_base()
    cfg.sga.selection.enabled = True
    expect_error(lambda: validate_sga_section(cfg), "selection")
    print("ok: enabled field is named in the error")


def test_missing_sga_section():
    cfg, _ = make_base()
    del cfg["sga"]
    expect_error(lambda: validate_sga_section(cfg), "sga")
    print("ok: missing sga section rejected")


def test_unknown_backend_lists_valid():
    cfg, _ = make_base()
    cfg.sga.backend.name = "quantum"
    expect_error(lambda: validate_sga_section(cfg), "dense")
    print("ok: unknown backend rejected with valid set")


def test_non_dict_subfield_rejected_cleanly():
    cfg, _ = make_base()
    cfg.sga.score = True
    expect_error(lambda: validate_sga_section(cfg), "score")
    print("ok: non-dict sga subfield raises curated ValueError")


def test_sga_never_reaches_megatron_cmdline():
    cfg, tmp = make_base()
    cfg.train.system.logging = {}
    from sga.launcher.backend_megatron import _update_config_train

    try:
        _update_config_train(cfg)
        args = _get_args_megatron(cfg)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    flat = " ".join(args)
    assert "--score" not in flat and "--backend" not in flat and "--selection" not in flat
    print("ok: sga section never reaches the megatron command line")


def test_flatten_bool_semantics():
    cfg, tmp = make_base()
    cfg.train.system.logging = {}
    from sga.launcher.backend_megatron import _update_config_train

    cfg.train.system.foo_off = False
    cfg.train.system.foo_on = True
    cfg.train.system.foo_val = 3
    try:
        _update_config_train(cfg)
        args = _get_args_megatron(cfg)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    assert "--foo-on" in args and "--foo-val" in args
    assert "--foo-off" not in args  # False silently dropped (upstream semantics)
    print("ok: flatten bool semantics preserved (False dropped)")


if __name__ == "__main__":
    test_all_off_dense_ok()
    test_enabled_field_named_in_error()
    test_missing_sga_section()
    test_unknown_backend_lists_valid()
    test_non_dict_subfield_rejected_cleanly()
    test_sga_never_reaches_megatron_cmdline()
    test_flatten_bool_semantics()
    print("test_config: ALL PASS")
