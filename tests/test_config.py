#!/usr/bin/env python3
"""Unit tests for the flat sga config validation (run: python tests/test_config.py).

The yaml ``sga:`` section speaks the SAME flat vocabulary as the --sga-*
flags (underscores instead of hyphens); validation lives in
sga/config/validation.py. These tests pin the section contract: disabled
semantics, axis requirements, engine sections
(cycle/checks/always_sync/trace), percent resolution, combination
rejection and the flag emission shape.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config import validate_run_config
from sga.config.validation import validate_sga_flat
from sga.launcher.backend_megatron import _get_args_megatron

BASE = """
experiment:
  exp_name: t
  exp_dir: PLACEHOLDER_EXP_DIR
  task: {type: train, backend: megatron, entrypoint: ./sga/train_gpt.py}
  runner: {type: ssh}
sga:
  sga_enabled: false
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


def _engine(cfg):
    """The nested read shape the engine used to get from
    validate_run_config; now assembled by the runtime."""
    from sga.runtime import materialize_engine_config
    canonical = validate_run_config(cfg)
    return None if canonical is None else materialize_engine_config(canonical)


def _enable_standard(cfg):
    """The canonical enabled combo: abs_m + topk(histogram) + adamw."""
    cfg.sga = {"sga_enabled": True, "sga_cycle_start_iter": 0,
               "sga_cycle_period": 8, "sga_score": "abs_m",
               "sga_selection_method": "histogram",
               "sga_selection_ratio_percent": 30.0,
               "sga_optimizer": "adamw",
               "sga_backend": "torch_pipelined_index"}
    return cfg


def test_all_off_dense_ok():
    cfg, _ = make_base()
    validate_run_config(cfg)
    print("ok: all-off dense config validates")


def test_enabled_field_named_in_error():
    cfg, _ = make_base()
    OmegaConf.update(cfg, "sga", {"sga_enabled": True, "sga_score": "abs_m",
                                  "sga_selection_method": "histogram",
                                  "sga_selection_ratio_percent": 30.0,
                                  "sga_optimizer": "adamw",
                                  "sga_backend": "torch_pipelined_index",
                                  "sga_bogus_key": 1})
    expect_error(lambda: validate_run_config(cfg), "sga_bogus_key")
    print("ok: unknown key is named in the error")


def test_fourfield_combination_resolves():
    cfg, _ = make_base()
    _enable_standard(cfg)
    engine_cfg = _engine(cfg)
    assert engine_cfg["selection"] == {"name": "topk", "method": "histogram",
                                       "ratio_percent": 30.0}
    assert engine_cfg["optimizer"] == {"name": "adamw"}
    # engine sections materialize complete (defaults applied)
    assert engine_cfg["cycle"]["period"] == 8
    assert engine_cfg["checks"] == {"count_guard": True}
    assert engine_cfg["shadow"] is False and engine_cfg["shadow_score"] is True
    assert "trace" not in engine_cfg
    print("ok: four-field combination resolves to the engine config")


def test_checks_validation():
    """sga_checks_count_guard carries the consistency-machinery toggle
    (default on); a non-bool value and an unknown checks key are named in
    the error, and any checks key is rejected when the engine is off."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    engine_cfg = _engine(cfg)
    assert engine_cfg["checks"] == {"count_guard": True}

    cfg.sga.sga_checks_count_guard = False
    engine_cfg = _engine(cfg)
    assert engine_cfg["checks"] == {"count_guard": False}

    cfg.sga.sga_checks_count_guard = "no"
    expect_error(lambda: validate_run_config(cfg), "bool")
    cfg2, _ = make_base()
    OmegaConf.update(cfg2, "sga.sga_checks_count_guard", False)
    expect_error(lambda: validate_run_config(cfg2), "sga_enabled is false")
    print("ok: checks validation (default / flag / bad type / off)")


def test_trace_validation():
    """sga_trace_dir configures the engine-side trace collector: dir
    required, branch_points normalized from comma string or list, window
    default 100; unknown trace keys named; rejected when the engine is
    off."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    engine_cfg = _engine(cfg)
    assert "trace" not in engine_cfg  # absent -> collector off

    cfg.sga.sga_trace_dir = "/tmp/sga-trace-x"
    cfg.sga.sga_trace_branch_points = "50, 10,50"
    engine_cfg = _engine(cfg)
    assert engine_cfg["trace"] == {"dir": "/tmp/sga-trace-x",
                                   "branch_points": [10, 50],  # sorted, deduped
                                   "window": 100}

    cfg.sga.sga_trace_branch_points = [7, "3"]
    engine_cfg = _engine(cfg)
    assert engine_cfg["trace"]["branch_points"] == [3, 7]

    cfg.sga.sga_trace_window = 0
    expect_error(lambda: validate_run_config(cfg), "window")
    cfg2, _ = make_base()
    OmegaConf.update(cfg2, "sga.sga_trace_dir", "/tmp/x")
    expect_error(lambda: validate_run_config(cfg2), "sga_enabled is false")

    cfg3, _ = make_base()
    _enable_standard(cfg3)
    OmegaConf.update(cfg3, "sga.sga_trace_bogus", 1)
    expect_error(lambda: validate_run_config(cfg3), "unknown keys")
    print("ok: trace validation (dir/branch/window/unknown/off)")


def test_start_iter_percent_resolves():
    cfg, _ = make_base()
    _enable_standard(cfg)
    del cfg.sga.sga_cycle_start_iter
    cfg.sga.sga_cycle_start_iter_percent = 1.6  # of train_iters 2 below
    OmegaConf.update(cfg, "train.model.train_iters", 5000)
    engine_cfg = _engine(cfg)
    assert engine_cfg["cycle"]["start_iter"] == 80  # round(0.016*5000)

    # both spellings at once is a conflict (explicit start_iter present)
    cfg.sga.sga_cycle_start_iter = 5
    expect_error(lambda: validate_run_config(cfg), "not both")
    # percent needs the train_iters context
    cfg4, _ = make_base()
    _enable_standard(cfg4)
    del cfg4.sga.sga_cycle_start_iter
    cfg4.sga.sga_cycle_start_iter_percent = 10.0
    expect_error(lambda: validate_run_config(cfg4), "train_iters")
    print("ok: start_iter_percent resolves / conflicts / needs context")


def test_illegal_combination_rejected():
    # unknown score name is rejected with the registered list
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_score = "update_ratio"  # the pre-rename name must fail
    expect_error(lambda: validate_run_config(cfg),
                 "must be one of ['abs_g', 'abs_m', 'coi', 'update']")
    # cycle-calendar legality: an explicit full_sync_interval is illegal
    # without the decouple switch (including the exact legacy fsi=0
    # spelling); decoupled values must be strict divisors
    cfg.sga.sga_score = "abs_m"
    cfg.sga.sga_cycle_full_sync_interval = 4
    expect_error(lambda: validate_run_config(cfg), "full_sync_interval")
    cfg.sga.sga_cycle_full_sync_interval = 0
    expect_error(lambda: validate_run_config(cfg),
                 "leave sga_cycle_full_sync_interval unset")
    cfg.sga.sga_cycle_full_sync_interval = None
    cfg.sga.sga_cycle_decouple_full_sync_interval_and_period = True
    expect_error(lambda: validate_run_config(cfg), "full_sync_interval")
    cfg.sga.sga_cycle_full_sync_interval = 3
    expect_error(lambda: validate_run_config(cfg), "full_sync_interval")
    # the legal decoupled spelling round-trips
    cfg.sga.sga_cycle_full_sync_interval = 4
    engine_cfg = _engine(cfg)
    assert engine_cfg["cycle"]["full_sync_interval"] == 4
    print("ok: illegal combinations rejected (unknown score / cycle rules)")


def test_sparse_backend_required_when_enabled():
    cfg, _ = make_base()
    OmegaConf.update(cfg, "sga", {"sga_enabled": True, "sga_score": "abs_m",
                                  "sga_selection_method": "histogram",
                                  "sga_selection_ratio_percent": 30.0,
                                  "sga_optimizer": "adamw",
                                  "sga_backend": "dense"})
    expect_error(lambda: validate_run_config(cfg), "dense")
    print("ok: backend dense rejected when enabled")


def test_missing_sga_section():
    cfg, _ = make_base()
    del cfg.sga
    expect_error(lambda: validate_run_config(cfg), "missing the 'sga:'")
    print("ok: missing sga section is loud")


def test_unknown_backend_lists_valid():
    flat = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "masked_pipelined"}
    expect_error(lambda: validate_sga_flat(flat),
                 "absorbed in the backend unification")
    print("ok: absorbed backend name lists the migration hint")


def test_disabled_rejects_any_other_explicit_key():
    for key, value in (("sga_shadow", True),
                       ("sga_cycle_period", 8),
                       ("sga_selection_method", "histogram"),
                       ("sga_trace_dir", "/tmp/x")):
        expect_error(lambda k=key, v=value: validate_sga_flat(
            {"sga_enabled": False, key: v}), "sga_enabled is false")
    # no exception anymore: an explicit sga_backend key is set config
    # while disabled, whatever its value (dense included)
    expect_error(lambda: validate_sga_flat(
        {"sga_enabled": False, "sga_backend": "dense"}),
        "sga_enabled is false")
    print("ok: disabled config rejects every other explicit key")


def test_sga_never_reaches_megatron_cmdline():
    """The sga: section must never leak into the train.* flatten: the
    launcher only forwards train.{system,model,data} as megatron args and
    the boundary assert inside validation backs it."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    args = _get_args_megatron(cfg)
    train_keys = ("log_dir", "details_dir", "scripts_dir", "pids_dir",
                  "no_shared_fs")
    assert not any(k in args for k in train_keys), args
    assert "sga" not in OmegaConf.to_container(cfg.train, resolve=True)
    print("ok: sga section never reaches the megatron command line")


def test_enabled_emits_flat_flag_sequence():
    """The sga: section becomes the visible --sga-* flags themselves
    (sorted, values spelled megatron-style)."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    args = _get_args_megatron(cfg)
    sga = [a for a in args if a.startswith("--sga-")]
    assert sga == ["--sga-backend", "--sga-cycle-period",
                   "--sga-cycle-start-iter", "--sga-enabled",
                   "--sga-optimizer", "--sga-score",
                   "--sga-selection-method",
                   "--sga-selection-ratio-percent"], sga
    flags = {args[i]: args[i + 1] for i in range(len(args) - 1)
             if args[i].startswith("--sga-")}
    assert flags["--sga-enabled"] == "true"
    assert flags["--sga-selection-ratio-percent"] == "30.0"
    assert flags["--sga-backend"] == "torch_pipelined_index"
    print("ok: enabled config emits the exact flat flag sequence")


def test_flatten_bool_semantics():
    """yaml booleans flatten to megatron-style true/false flag spellings."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_optimizer_residual_mean = True
    args = _get_args_megatron(cfg)
    flags = {args[i]: args[i + 1] for i in range(len(args) - 1)
             if args[i].startswith("--sga-")}
    assert flags["--sga-optimizer-residual-mean"] == "true"
    print("ok: booleans flatten to true/false spellings")


def test_plugin_param_vocabulary():
    """A key no flag declares is loud at validation; a declared key of
    ANOTHER plugin is the plugin's own business — validation (values
    only, no name attribution) passes it, the owning constructor
    rejects it at build time."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    expect_error(lambda: validate_sga_flat({**base, "sga_selection_bogus": 1}),
                 "unknown keys")
    expect_error(lambda: validate_sga_flat({**base, "sga_score_extra": 1}),
                 "unknown keys")
    canon = validate_sga_flat({**base, "sga_backend_mask_slices": 3})
    assert canon["sga_backend_mask_slices"] == 3  # validation is silent
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend

    expect_error(lambda: TorchPipelinedIndexBackend({"mask_slices": 3}),
                 "unknown params ['mask_slices']")
    print("ok: declared-flag typos loud at validation, plugin vocab at build")


def test_float_spelling_canonicalized():
    """ratio 10 vs 10.0 canonicalize identically (resume-safe)."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    canon = validate_sga_flat(dict(base))
    assert canon["sga_selection_ratio_percent"] == 10.0
    assert isinstance(canon["sga_selection_ratio_percent"], float)
    print("ok: float spellings canonicalize")


def test_wrong_types_fail_loud():
    """C1 pin: the yaml route feeds raw user values through _coerce
    (argparse typing is bypassed), so wrong-typed values must fail loud
    with the key named — a yaml true/false never becomes 1/0, a float
    never truncates for an int dest, numeric strings are not ints, and
    float dests reject non-finite values."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    expect_error(lambda: validate_sga_flat({**base, "sga_cycle_period": True}),
                 "sga_cycle_period must be a valid int (got True)")
    expect_error(lambda: validate_sga_flat({**base, "sga_backend_index_slices": 3.9}),
                 "sga_backend_index_slices must be a valid int (got 3.9)")
    expect_error(lambda: validate_sga_flat({**base, "sga_cycle_period": "8"}),
                 "sga_cycle_period must be a valid int (got '8')")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_selection_ratio_percent": True}),
        "sga_selection_ratio_percent must be a valid float (got True)")
    expect_error(lambda: validate_sga_flat({**base, "sga_trace_window": True}),
                 "sga_trace_window must be a valid int (got True)")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_selection_adaptive_warmup_syncs": True}),
        "sga_selection_adaptive_warmup_syncs must be a valid int_or_null "
        "(got True)")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_selection_ratio_percent": float("nan")}),
        "sga_selection_ratio_percent must be a finite float (got nan)")
    # integral floats are still rejected for int dests (8.0 is a float)
    expect_error(lambda: validate_sga_flat({**base, "sga_backend_index_slices": 3.0}),
                 "sga_backend_index_slices must be a valid int (got 3.0)")
    print("ok: wrong-typed yaml values fail loud (bool/float/str/nan)")


def test_shipped_examples_validate():
    """Every shipped example yaml passes validation as-is (a renamed or
    removed flag left behind in an example would otherwise surface only
    at a user's launch)."""
    import glob

    examples = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "llama3", "*.yaml")))
    assert examples, "no shipped examples found"
    for path in examples:
        cfg = OmegaConf.load(path)
        validate_run_config(cfg)  # must not raise (None = dense example)
    print(f"ok: all {len(examples)} shipped examples validate")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL CONFIG TESTS PASSED")




def test_list_kind_flag_emits_comma_token():
    """A list-valued sga key rides the command line as ONE comma-joined
    token (megatron list style) — the shape the str-typed flag parses."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_always_sync_layers = ["embed", "norm"]
    args = _get_args_megatron(cfg)
    flags = {args[i]: args[i + 1] for i in range(len(args) - 1)
             if args[i].startswith("--sga-")}
    assert flags["--sga-always-sync-layers"] == "embed,norm", flags
    print("ok: list-valued sga key emits one comma-joined token")


if __name__ == "__main__":
    _run_all()
