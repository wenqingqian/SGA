#!/usr/bin/env python3
"""Unit tests for sga.config validation (run: python tests/test_config.py)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config import (
    sga_section_to_config,
    validate_config,
    validate_sga_section,
)
from sga.launcher.backend_megatron import _get_args_megatron

BASE = """
experiment:
  exp_name: t
  exp_dir: PLACEHOLDER_EXP_DIR
  task: {type: train, backend: megatron, entrypoint: ./sga/train_gpt.py}
  runner: {type: ssh}
sga:
  enabled: false
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


def _enable_standard(cfg):
    """The canonical enabled combo: abs_m + topk(histogram) + adamw."""
    cfg.sga.enabled = True
    cfg.sga.cycle = {"start_iter": 0, "period": 8}
    cfg.sga.score = {"name": "abs_m"}
    cfg.sga.selection = {"name": "topk", "method": "histogram", "ratio_percent": 30.0}
    cfg.sga.optimizer = {"name": "adamw"}
    cfg.sga.backend = {"name": "torch_pipelined_index"}
    return cfg


def test_all_off_dense_ok():
    cfg, _ = make_base()
    validate_config(cfg)
    print("ok: all-off dense config validates")


def test_enabled_field_named_in_error():
    cfg, _ = make_base()
    cfg.sga.selection = {"name": "topk", "method": "histogram"}
    expect_error(lambda: validate_sga_section(cfg), "selection")
    print("ok: enabled field is named in the error")


def test_fourfield_combination_resolves():
    cfg, _ = make_base()
    _enable_standard(cfg)
    engine_cfg = sga_section_to_config(cfg)
    assert engine_cfg["selection"] == {"name": "topk", "method": "histogram",
                                       "ratio_percent": 30.0}
    assert engine_cfg["optimizer"] == {"name": "adamw"}
    print("ok: four-field combination resolves to the engine config")


def test_checks_section_validation():
    """sga.checks carries the consistency-machinery toggles (count_guard,
    default on); unknown keys are named in the error, and the section is
    rejected when the engine is off."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    engine_cfg = sga_section_to_config(cfg)
    assert "checks" not in engine_cfg  # absent -> engine defaults apply

    cfg.sga.checks = {"count_guard": False}
    engine_cfg = sga_section_to_config(cfg)
    assert engine_cfg["checks"] == {"count_guard": False}

    cfg.sga.checks = {"count_guard": "no"}
    expect_error(lambda: sga_section_to_config(cfg), "boolean")
    cfg.sga.checks = {"bogus_toggle": True}
    expect_error(lambda: sga_section_to_config(cfg), "bogus_toggle")

    cfg2, _ = make_base()
    cfg2.sga.checks = {"count_guard": False}
    expect_error(lambda: sga_section_to_config(cfg2), "enabled is false")
    print("ok: sga.checks validation (default absent / flag / unknown / off)")


def test_trace_section_validation():
    """sga.trace configures the engine-side trace collector: dir required,
    branch_points normalized from comma string or list, window default 100;
    unknown keys named in the error; section rejected when the engine is off."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    engine_cfg = sga_section_to_config(cfg)
    assert "trace" not in engine_cfg  # absent -> collector off

    cfg.sga.trace = {"dir": "/tmp/sga-trace-x", "branch_points": "50, 10,50"}
    engine_cfg = sga_section_to_config(cfg)
    assert engine_cfg["trace"]["dir"] == "/tmp/sga-trace-x"
    assert engine_cfg["trace"]["branch_points"] == [10, 50]  # sorted, deduped
    assert engine_cfg["trace"]["window"] == 100  # default

    cfg.sga.trace = {"dir": "/tmp/sga-trace-x", "branch_points": [200, 7],
                     "window": 5}
    engine_cfg = sga_section_to_config(cfg)
    assert engine_cfg["trace"]["branch_points"] == [7, 200]
    assert engine_cfg["trace"]["window"] == 5

    cfg.sga.trace = {"branch_points": [10]}
    expect_error(lambda: sga_section_to_config(cfg), "dir")
    cfg.sga.trace = {"dir": "/tmp/x", "bogus": 1}
    expect_error(lambda: sga_section_to_config(cfg), "bogus")
    cfg.sga.trace = {"dir": "/tmp/x", "branch_points": [-3]}
    expect_error(lambda: sga_section_to_config(cfg), ">= 0")
    cfg.sga.trace = {"dir": "/tmp/x", "window": 0}
    expect_error(lambda: sga_section_to_config(cfg), "window")

    cfg2, _ = make_base()
    cfg2.sga.trace = {"dir": "/tmp/x"}
    expect_error(lambda: sga_section_to_config(cfg2), "enabled is false")
    print("ok: sga.trace validation (absent / normalize / unknown / off)")


def test_start_iter_percent_resolves():
    """cycle.start_iter_percent resolves against train.model.train_iters in
    the loader, before the canonical --sga-config JSON is built."""
    cfg, _ = make_base()
    cfg.train.model.train_iters = 5000
    _enable_standard(cfg)
    cfg.sga.cycle = {"start_iter_percent": 10.0, "period": 4}
    engine_cfg = sga_section_to_config(cfg)
    assert engine_cfg["cycle"]["start_iter"] == 500
    assert "start_iter_percent" not in engine_cfg["cycle"]

    cfg.sga.cycle = {"start_iter": 4, "start_iter_percent": 10.0, "period": 4}
    expect_error(lambda: validate_sga_section(cfg), "not both")
    # an explicit null hits the intended range diagnostic, not float(None)
    cfg.sga.cycle = {"start_iter_percent": None, "period": 4}
    expect_error(lambda: validate_sga_section(cfg), "got null")
    cfg.sga.cycle = {"start_iter_percent": 10.0, "period": 4}
    cfg.train.model.train_iters = 0
    expect_error(lambda: validate_sga_section(cfg), "train_iters > 0")
    print("ok: start_iter_percent resolves in the loader")


def test_illegal_combination_rejected():
    # unknown score name is rejected with the registered list
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.score = {"name": "update_ratio"}  # the pre-rename name must fail
    expect_error(lambda: validate_sga_section(cfg), "Unknown SGA score")
    # cycle-calendar legality: an explicit full_sync_interval is illegal
    # without the decouple switch (including the exact legacy fsi=0
    # spelling); decoupled values must be strict divisors
    cfg.sga.score = {"name": "abs_m"}
    cfg.sga.cycle = {"start_iter": 0, "period": 8, "full_sync_interval": 4}
    expect_error(lambda: validate_sga_section(cfg), "full_sync")
    cfg.sga.cycle = {"start_iter": 0, "period": 8, "full_sync_interval": 0}
    expect_error(lambda: validate_sga_section(cfg),
                 "leave full_sync_interval unset")
    cfg.sga.cycle = {"start_iter": 0, "period": 8,
                     "decouple_full_sync_interval_and_period": True}
    expect_error(lambda: validate_sga_section(cfg), "full_sync")
    cfg.sga.cycle = {"start_iter": 0, "period": 8,
                     "decouple_full_sync_interval_and_period": True,
                     "full_sync_interval": 3}
    expect_error(lambda: validate_sga_section(cfg), "full_sync")
    # the legal decoupled spelling round-trips
    cfg.sga.cycle = {"start_iter": 0, "period": 8,
                     "decouple_full_sync_interval_and_period": True,
                     "full_sync_interval": 4}
    engine_cfg = sga_section_to_config(cfg)
    assert engine_cfg["cycle"]["full_sync_interval"] == 4
    print("ok: illegal combinations rejected (unknown score / cycle rules)")


def test_sparse_backend_required_when_enabled():
    cfg, _ = make_base()
    _enable_standard(cfg)
    # backend stays {name: dense} from BASE
    cfg.sga.backend = {"name": "dense"}
    expect_error(lambda: validate_sga_section(cfg), "dense")
    print("ok: enabled-with-dense-backend rejected")


def test_missing_sga_section():
    cfg, _ = make_base()
    del cfg["sga"]
    expect_error(lambda: validate_sga_section(cfg), "sga")
    print("ok: missing sga section rejected")


def test_unknown_backend_lists_valid():
    cfg, _ = make_base()
    cfg.sga.backend.name = "quantum"
    expect_error(lambda: validate_sga_section(cfg), "backend")
    print("ok: unknown backend rejected with curated error")


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


def test_enabled_emits_flat_flag_sequence():
    """Enabled path: the sga section reaches the megatron command line as
    VISIBLE flat --sga-* flags (no opaque --sga-config JSON token), exactly
    the sequence sga_config_to_flags derives from the resolved engine
    config — the spelling the training entry re-assembles into the
    canonical args.sga_config JSON."""
    cfg, tmp = make_base()
    cfg.sga.enabled = True
    cfg.sga.cycle = {"start_iter": 0, "period": 2}
    cfg.sga.score = {"name": "abs_m"}
    cfg.sga.selection = {"name": "topk", "method": "histogram", "ratio_percent": 20.0}
    cfg.sga.optimizer = {"name": "adamw", "residual_mean": True}
    cfg.sga.backend = {"name": "torch_pipelined_mask"}
    expected_engine_cfg = sga_section_to_config(cfg)

    from sga.config.arguments import sga_config_to_flags
    from sga.launcher.backend_megatron import _update_config_train

    try:
        _update_config_train(cfg)
        args = _get_args_megatron(cfg)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    expected_flags = sga_config_to_flags(expected_engine_cfg)
    assert args.count("--sga-config") == 0, args
    # appended verbatim at the tail of the megatron command line
    assert args[-len(expected_flags):] == expected_flags, args
    assert not [a for a in args[:-len(expected_flags)] if a.startswith("--sga")], args
    pairs = dict(zip(expected_flags[::2], expected_flags[1::2]))
    assert pairs["--sga-enabled"] == "true"
    assert pairs["--sga-cycle-period"] == "2"
    assert pairs["--sga-optimizer-residual-mean"] == "true"
    assert pairs["--sga-selection-ratio-percent"] == "20.0"
    print("ok: enabled path emits the flat --sga-* flag sequence")


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


def test_sga_top_level_allowlist():
    """The sga: section is a closed vocabulary. Unknown top-level keys are
    rejected mode-INDEPENDENTLY (before the enabled split): a `shadw: true`
    typo dropped silently would run live sparse where the user intended
    shadow's bitwise-dense invariant — the most dangerous silent failure on
    this config face. The legacy dense-yaml form (axis blocks carrying
    enabled: false instead of name, no top-level enabled key) still passes."""
    cfg, _ = make_base()
    cfg.sga.shadw = True
    expect_error(lambda: sga_section_to_config(cfg), "unknown keys ['shadw']")
    # unknown key beats the mode-legality diagnostics (membership first)
    cfg2, _ = make_base()
    cfg2.sga.selection = {"name": "topk", "method": "histogram"}
    cfg2.sga.slect = {"name": "topk", "method": "histogram"}
    expect_error(lambda: sga_section_to_config(cfg2), "unknown keys ['slect']")

    # every top-level key at once (enabled path) resolves
    cfg3, _ = make_base()
    _enable_standard(cfg3)
    cfg3.sga.checks = {"count_guard": True}
    cfg3.sga.trace = {"dir": "/tmp/sga-trace-x"}
    cfg3.sga.shadow = True
    cfg3.sga.shadow_score = True
    engine_cfg = sga_section_to_config(cfg3)
    for key in ("cycle", "score", "selection", "optimizer", "backend",
                "checks", "trace", "shadow", "shadow_score"):
        assert key in engine_cfg, key

    # legacy dense form (test_launcher_shape BASE shape) still validates
    cfg4, _ = make_base()
    del cfg4.sga.enabled
    for axis in ("score", "selection", "optimizer"):
        cfg4.sga[axis] = {"enabled": False}
    assert sga_section_to_config(cfg4) is None
    print("ok: sga top-level allowlist (typo / all-known / legacy dense form)")


def test_cycle_interior_allowlist():
    """cycle's interior is a fixed vocabulary: a `peroid` typo would
    otherwise silently fall back to period=1 — reselection every step
    instead of the configured cycle, with no diagnostic anywhere."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.cycle = {"start_iter": 0, "peroid": 8}
    expect_error(lambda: sga_section_to_config(cfg), "unknown keys ['peroid']")

    cfg.sga.cycle = 5
    expect_error(lambda: sga_section_to_config(cfg), "must be a mapping")
    print("ok: sga.cycle interior allowlist (peroid typo / non-mapping)")


def test_plugin_param_vocabulary():
    """Each plugin axis has a declared param vocabulary: unknown keys,
    wrong types, out-of-choices values, and missing required params are
    named in the error (the loader gate; constructors repeat it)."""
    cfg, _ = make_base()
    _enable_standard(cfg)

    cfg.sga.selection = {"name": "topk", "method": "histogram",
                         "ratio_percent": 30.0, "rtaio": 5}
    expect_error(lambda: sga_section_to_config(cfg), "unknown params ['rtaio']")

    cfg.sga.selection = {"name": "topk", "method": "histogram", "ratio_percent": "30"}
    expect_error(lambda: sga_section_to_config(cfg), "must be float")

    cfg.sga.selection = {"name": "topk", "method": "histogram",
                         "ratio_percent": 30.0, "histogram_mode": "turbo"}
    expect_error(lambda: sga_section_to_config(cfg), "must be one of")

    cfg.sga.selection = {"name": "topk", "method": "histogram"}  # ratio_percent required
    expect_error(lambda: sga_section_to_config(cfg), "required")

    cfg.sga.selection = {"name": "topk", "method": "histogram", "ratio_percent": 30.0}
    cfg.sga.optimizer = {"name": "adamw", "residual_mean": "yes"}
    expect_error(lambda: sga_section_to_config(cfg), "must be bool")

    cfg.sga.optimizer = {"name": "adamw"}
    cfg.sga.backend = {"name": "torch_pipelined_index", "anything": 1}
    expect_error(lambda: sga_section_to_config(cfg), "unknown params ['anything']")
    print("ok: plugin param vocabulary enforced (unknown/type/choices/required)")


if __name__ == "__main__":
    test_all_off_dense_ok()
    test_enabled_field_named_in_error()
    test_fourfield_combination_resolves()
    test_start_iter_percent_resolves()
    test_illegal_combination_rejected()
    test_sparse_backend_required_when_enabled()
    test_missing_sga_section()
    test_unknown_backend_lists_valid()
    test_non_dict_subfield_rejected_cleanly()
    test_sga_never_reaches_megatron_cmdline()
    test_enabled_emits_flat_flag_sequence()
    test_flatten_bool_semantics()
    test_checks_section_validation()
    test_trace_section_validation()
    test_sga_top_level_allowlist()
    test_cycle_interior_allowlist()
    test_plugin_param_vocabulary()
    print("test_config: ALL PASS")
