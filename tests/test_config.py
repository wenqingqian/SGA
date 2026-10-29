#!/usr/bin/env python3
"""Unit tests for the flat sga config validation (run: python tests/test_config.py).

The yaml ``sga:`` section speaks the SAME flat vocabulary as the --sga-*
flags (underscores instead of hyphens); validation lives in
sga/config/validation.py (``validate_sga_args``, megatron validate_args
style: value-level rules on an args namespace, write-backs onto it).
The launcher pre-validates by parsing the very flags it is about to emit
(``_prevalidate_sga``) — one argparse code path with the training entry.
These tests pin the section contract: disabled semantics (zero checks),
axis requirements, engine knobs (cycle/checks/always_sync/trace), percent
resolution, value rules and the flag emission shape.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config.validation import validate_sga_args
from sga.launcher.backend_megatron import _get_args_megatron, _prevalidate_sga

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
    """A fresh BASE config with exp_dir in a private temp dir."""
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


def _ns_of(cfg):
    """The launch-time namespace: parse the emitted flags with the real
    parser (exactly what _prevalidate_sga does), attach the two non-sga
    keys (through both train subtrees, flattening order — the same
    selects _prevalidate_sga runs), validate (write-backs land on the
    namespace)."""
    import argparse

    from sga.config.arguments import add_sga_args

    tokens, it = [], iter(_get_args_megatron(cfg))
    for tok in it:
        if tok.startswith("--sga-"):
            tokens.extend([tok, next(it)])
    parser = argparse.ArgumentParser(add_help=False)
    add_sga_args(parser)
    ns = parser.parse_args(tokens)
    sys_it = OmegaConf.select(cfg, "train.system.train_iters", default=None)
    ns.train_iters = OmegaConf.select(cfg, "train.model.train_iters",
                                      default=sys_it)
    sys_pp = OmegaConf.select(
        cfg, "train.system.pipeline_model_parallel_size", default=1)
    ns.pipeline_model_parallel_size = OmegaConf.select(
        cfg, "train.model.pipeline_model_parallel_size", default=sys_pp)
    validate_sga_args(ns)
    return ns


def _enable_standard(cfg):
    """The canonical enabled combo: abs_m + topk(histogram balanced —
    the only selection validation accepts) + adamw."""
    cfg.sga = {"sga_enabled": True, "sga_cycle_start_iter": 0,
               "sga_cycle_period": 8, "sga_score": "abs_m",
               "sga_selection_method": "histogram",
               "sga_selection_histogram_mode": "balanced",
               "sga_selection_ratio_percent": 30.0,
               "sga_optimizer": "adamw",
               "sga_backend": "torch_pipelined_index"}
    return cfg


def test_all_off_dense_ok():
    cfg, _ = make_base()
    _prevalidate_sga(cfg)
    print("ok: all-off dense config validates")


def test_enabled_field_named_in_error():
    cfg, _ = make_base()
    OmegaConf.update(cfg, "sga", {"sga_enabled": True, "sga_score": "abs_m",
                                  "sga_selection_method": "histogram",
                                  "sga_selection_ratio_percent": 30.0,
                                  "sga_optimizer": "adamw",
                                  "sga_backend": "torch_pipelined_index",
                                  "sga_bogus_key": 1})
    expect_error(lambda: _prevalidate_sga(cfg), "sga_bogus_key")
    print("ok: unknown key is named in the error")


def test_fourfield_combination_resolves():
    cfg, _ = make_base()
    _enable_standard(cfg)
    ns = _ns_of(cfg)
    assert ns.sga_selection == "topk"
    assert ns.sga_selection_method == "histogram"
    assert ns.sga_selection_ratio_percent == 30.0
    assert ns.sga_optimizer == "adamw"
    # the namespace is complete by construction (argparse defaults filled)
    assert ns.sga_cycle_period == 8
    assert ns.sga_checks_count_guard is True
    assert ns.sga_shadow is False and ns.sga_shadow_score is True
    assert ns.sga_trace_dir is None  # trace collector off
    print("ok: four-field combination resolves onto the namespace")


def test_checks_validation():
    """sga_checks_count_guard carries the consistency-machinery toggle
    (default on); a non-bool value is named in the parse error."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    ns = _ns_of(cfg)
    assert ns.sga_checks_count_guard is True

    cfg.sga.sga_checks_count_guard = False
    ns = _ns_of(cfg)
    assert ns.sga_checks_count_guard is False

    cfg.sga.sga_checks_count_guard = "no"
    expect_error(lambda: _prevalidate_sga(cfg), "true' or 'false")
    # any checks key under a disabled config is silently ignored
    # (megatron semantics: zero checks below sga_enabled=false)
    cfg2, _ = make_base()
    OmegaConf.update(cfg2, "sga.sga_checks_count_guard", False)
    _prevalidate_sga(cfg2)
    print("ok: checks validation (default / flag / bad type / off-ignored)")


def test_checks_master_switch():
    """sga_checks master (default true): false forces every consistency
    sub-flag off via the validation write-back — an explicitly-true sub
    under a false master is master-override semantics, not an error (bool
    defaults make 'explicitly true' indistinguishable from unset).
    Covers the namespace route (SimpleNamespace, the build_engine face)
    and the yaml route (_ns_of, the launcher face). sga_enabled=false
    returns before the forcing: sub-flags keep their own values."""
    from types import SimpleNamespace

    from sga.config.validation import sga_defaults

    subs = ("sga_checks_count_guard", "sga_checks_support_consistency",
            "sga_checks_norm_spread", "sga_checks_rng_bracket")

    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_histogram_mode": "balanced",
            "sga_selection_ratio_percent": 10.0,
            "sga_backend": "torch_pipelined_index",
            "sga_cycle_period": 8}

    def _ns(**over):
        ns = SimpleNamespace(**{**sga_defaults(), **base, **over})
        ns.train_iters = 5000
        ns.pipeline_model_parallel_size = 1
        validate_sga_args(ns)
        return ns

    # master false → all four subs written back False (explicit true sub
    # included — master-override, no error)
    ns = _ns(sga_checks=False, sga_checks_support_consistency=True)
    assert ns.sga_checks is False
    for dest in subs:
        assert getattr(ns, dest) is False, dest

    # default master → each sub honors its own spelling
    ns = _ns(sga_checks_norm_spread=False)
    assert ns.sga_checks is True
    assert ns.sga_checks_norm_spread is False
    for dest in subs:
        if dest != "sga_checks_norm_spread":
            assert getattr(ns, dest) is True, dest

    # yaml route (launcher face): master false lands on the parsed ns too
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_checks = False
    ns2 = _ns_of(cfg)
    for dest in subs:
        assert getattr(ns2, dest) is False, dest

    # feature off → early return BEFORE the forcing: subs untouched
    ns3 = SimpleNamespace(**{**sga_defaults(),
                             **{"sga_enabled": False, "sga_checks": False}})
    validate_sga_args(ns3)
    for dest in subs:
        assert getattr(ns3, dest) is True, dest
    print("ok: checks master switch (force-off both routes / sub autonomy "
          "/ feature-off early return)")


def test_trace_validation():
    """sga_trace_dir configures the engine-side trace collector: dir
    required, branch_points normalized from comma string or list, window
    default 100, an illegal dump dtype refused on the yaml route (the flag
    is named), unknown trace keys named."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    ns = _ns_of(cfg)
    assert ns.sga_trace_dir is None  # absent -> collector off

    cfg.sga.sga_trace_dir = "/tmp/sga-trace-x"
    cfg.sga.sga_trace_branch_points = "50, 10,50"
    ns = _ns_of(cfg)
    assert ns.sga_trace_dir == "/tmp/sga-trace-x"
    assert ns.sga_trace_branch_points == [10, 50]  # sorted, deduped
    assert ns.sga_trace_window == 100

    cfg.sga.sga_trace_branch_points = [7, "3"]
    ns = _ns_of(cfg)
    assert ns.sga_trace_branch_points == [3, 7]

    cfg.sga.sga_trace_window = 0
    expect_error(lambda: _prevalidate_sga(cfg), "window")

    # yaml dump dtype: the illegal value rides the emitted flags through
    # the real parser, so argparse choices rejects it (_LoudParser names
    # the flag); validation.py's membership check stays as defense-in-depth
    # for namespaces built without the parser
    cfg2, _ = make_base()
    _enable_standard(cfg2)
    cfg2.sga.sga_trace_dump_dtype = "fp16"
    expect_error(lambda: _prevalidate_sga(cfg2), "sga-trace-dump-dtype")

    cfg3, _ = make_base()
    _enable_standard(cfg3)
    OmegaConf.update(cfg3, "sga.sga_trace_bogus", 1)
    expect_error(lambda: _prevalidate_sga(cfg3), "sga_trace_bogus")
    print("ok: trace validation (dir/branch/window/dump-dtype/unknown)")


def test_start_iter_percent_resolves():
    cfg, _ = make_base()
    _enable_standard(cfg)
    del cfg.sga.sga_cycle_start_iter
    cfg.sga.sga_cycle_start_iter_percent = 1.6  # 1.6% of train_iters=5000 -> 80
    OmegaConf.update(cfg, "train.model.train_iters", 5000)
    ns = _ns_of(cfg)
    assert ns.sga_cycle_start_iter == 80  # round(0.016*5000)

    # both spellings at once is a conflict (explicit start_iter present)
    cfg.sga.sga_cycle_start_iter = 5
    expect_error(lambda: _prevalidate_sga(cfg), "not both")
    # percent needs the train_iters context
    cfg4, _ = make_base()
    _enable_standard(cfg4)
    del cfg4.sga.sga_cycle_start_iter
    cfg4.sga.sga_cycle_start_iter_percent = 10.0
    expect_error(lambda: _prevalidate_sga(cfg4), "train_iters")
    print("ok: start_iter_percent resolves / conflicts / needs context")


def test_illegal_combination_rejected():
    # unknown score name is rejected by the argparse choices at parse time
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_score = "update_ratio"  # the pre-rename name must fail
    expect_error(lambda: _prevalidate_sga(cfg), "invalid choice")
    # cycle-calendar legality: an explicit full_sync_interval is illegal
    # without the decouple switch (including the exact legacy fsi=0
    # spelling); decoupled values must be strict divisors
    cfg.sga.sga_score = "abs_m"
    cfg.sga.sga_cycle_full_sync_interval = 4
    expect_error(lambda: _prevalidate_sga(cfg), "full_sync_interval")
    cfg.sga.sga_cycle_full_sync_interval = 0
    expect_error(lambda: _prevalidate_sga(cfg),
                 "leave sga_cycle_full_sync_interval unset")
    cfg.sga.sga_cycle_full_sync_interval = None
    cfg.sga.sga_cycle_decouple_full_sync_interval_and_period = True
    expect_error(lambda: _prevalidate_sga(cfg), "full_sync_interval")
    cfg.sga.sga_cycle_full_sync_interval = 3
    expect_error(lambda: _prevalidate_sga(cfg), "full_sync_interval")
    # the legal decoupled spelling round-trips
    cfg.sga.sga_cycle_full_sync_interval = 4
    ns = _ns_of(cfg)
    assert ns.sga_cycle_full_sync_interval == 4
    print("ok: illegal combinations rejected (unknown score / cycle rules)")


def test_optimizer_dual_lr_scale_rules():
    """The boundary dual-lr knob's strict assertions: a finite
    float >= 0 is required (0.0 is legal — the freeze-the-tail ablation
    extreme; nan/inf are refused — argparse's float typing would accept
    their spellings), and a scale != 1.0 is refused under shadow (bitwise
    stock-replication contract) and under sga_cycle_period 1 (no sparse
    rounds — dead config, refused instead of silently ignored)."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_histogram_mode": "balanced",
            "sga_selection_ratio_percent": 10.0,
            "sga_backend": "torch_pipelined_index",
            "sga_cycle_period": 8}

    def _ns(**over):
        from types import SimpleNamespace

        from sga.config.validation import sga_defaults

        ns = SimpleNamespace(**{**sga_defaults(), **base, **over})
        ns.train_iters = 5000
        ns.pipeline_model_parallel_size = 1
        validate_sga_args(ns)
        return ns

    for spelling in (0.5, 0.0, 2):
        ns = _ns(sga_optimizer_full_sync_inactive_lr_scale=spelling)
        assert ns.sga_optimizer_full_sync_inactive_lr_scale == float(spelling)
    # the declared default rides along when unset
    assert _ns().sga_optimizer_full_sync_inactive_lr_scale == 1.0
    expect_error(lambda: _ns(
        sga_optimizer_full_sync_inactive_lr_scale=-0.5), ">= 0.0")
    expect_error(lambda: _ns(
        sga_optimizer_full_sync_inactive_lr_scale=float("nan")), "finite")
    expect_error(lambda: _ns(
        sga_optimizer_full_sync_inactive_lr_scale=float("inf")), "finite")
    expect_error(lambda: _ns(
        sga_optimizer_full_sync_inactive_lr_scale=0.5, sga_shadow=True),
        "sga_shadow")
    expect_error(lambda: _ns(
        sga_cycle_period=1,
        sga_optimizer_full_sync_inactive_lr_scale=0.5), "sga_cycle_period 1")
    # disabled: zero checks — the knob is silently ignored
    ns = _ns(sga_enabled=False, sga_optimizer_full_sync_inactive_lr_scale=0.5)
    assert ns.sga_optimizer_full_sync_inactive_lr_scale == 0.5
    print("ok: dual-lr scale rules (domain / shadow / period=1 / disabled)")


def test_sparse_backend_required_when_enabled():
    cfg, _ = make_base()
    OmegaConf.update(cfg, "sga", {"sga_enabled": True, "sga_score": "abs_m",
                                  "sga_selection_method": "histogram",
                                  "sga_selection_ratio_percent": 30.0,
                                  "sga_optimizer": "adamw",
                                  "sga_backend": "dense"})
    expect_error(lambda: _prevalidate_sga(cfg), "dense")
    print("ok: backend dense rejected when enabled")


def test_missing_sga_section():
    cfg, _ = make_base()
    del cfg.sga
    expect_error(lambda: _prevalidate_sga(cfg), "missing the 'sga:'")
    print("ok: missing sga section is loud")


def test_ill_typed_and_empty_sga_sections():
    """A scalar/list section would die inside OmegaConf.to_container with
    an opaque internal error, and an empty mapping would silently
    validate as dense — both must meet the missing-section loudness
    (named ValueError) at the launch door."""
    for bad in (True, "str", [1, 2]):
        cfg, _ = make_base()
        cfg.sga = bad
        expect_error(lambda c=cfg: _prevalidate_sga(c),
                     "sga section must be a mapping")
    cfg, _ = make_base()
    cfg.sga = {}
    expect_error(lambda: _prevalidate_sga(cfg), "empty 'sga:' section")
    print("ok: scalar/list/empty sga sections refuse loud, by name")


def test_shipped_layout_subtree_reads():
    """The two non-sga keys validation reads are selected through BOTH
    train subtrees in the flattening order (system first, model
    overrides): the SHIPPED layout carries pipeline_model_parallel_size
    under train.system (all llama3 examples), so reading only one subtree
    leaves the always-sync pp>1 rule dead at launch time for shipped
    configs (build_engine refuses only after spawn)."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_always_sync_layers = ["embed"]
    cfg.train.system.pipeline_model_parallel_size = 2
    expect_error(lambda: _prevalidate_sga(cfg), "pp == 1")
    # ...and a system-level train_iters feeds the percent resolution the
    # same way (the shipped examples carry it under train.model, which
    # the override leg of the same select keeps working)
    cfg2, _ = make_base()
    _enable_standard(cfg2)
    del cfg2.sga.sga_cycle_start_iter
    cfg2.sga.sga_cycle_start_iter_percent = 1.6
    cfg2.train.system.train_iters = 5000
    ns = _ns_of(cfg2)
    assert ns.sga_cycle_start_iter == 80  # round(0.016*5000)
    print("ok: pp/train_iters reach validation from train.system too")


def test_null_master_switch_refused():
    """A yaml null on the master switch is a typo, not an OFF spelling:
    it must refuse loud instead of rendering no --sga-enabled flag (a
    dense launch with the whole SGA section silently ignored — the
    null-as-unset mechanism stays for every non-master key)."""
    cfg, _ = make_base()
    _enable_standard(cfg)
    cfg.sga.sga_enabled = None
    expect_error(lambda: _prevalidate_sga(cfg),
                 "sga_enabled is explicitly null")
    # a null on a NON-master key keeps the unset spelling (no token is
    # rendered for it — the refusal is scoped to sga_enabled)
    from sga.launcher.backend_megatron import _sga_section_to_flags
    cfg2, _ = make_base()
    _enable_standard(cfg2)
    cfg2.sga.sga_optimizer_compact_search = None
    cfg2.sga.sga_backend_emission = None
    tokens = _sga_section_to_flags(cfg2)
    assert "--sga-enabled" in tokens
    assert "--sga-optimizer-compact-search" not in tokens
    assert "--sga-backend-emission" not in tokens
    print("ok: null master switch refuses loud; non-master nulls unset")


def test_unknown_backend_lists_valid():
    flat = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "masked_pipelined"}
    cfg, _ = make_base()
    cfg.sga = flat
    expect_error(lambda: _prevalidate_sga(cfg),
                 "Unknown SGA backend plugin 'masked_pipelined'")
    print("ok: absorbed backend name reads as unknown, registered list shown")


def test_disabled_ignores_other_keys():
    """Megatron semantics: a disabled feature's sub-args are
    never validated — every explicit key under sga_enabled=false is
    silently ignored."""
    cfg, _ = make_base()
    OmegaConf.update(cfg, "sga", {"sga_enabled": False, "sga_shadow": True,
                                  "sga_cycle_period": 8,
                                  "sga_selection_method": "histogram",
                                  "sga_trace_dir": "/tmp/x",
                                  "sga_backend": "dense"})
    _prevalidate_sga(cfg)  # zero checks, no exception
    print("ok: disabled config silently ignores every other key")


def test_sga_never_reaches_megatron_cmdline():
    """The sga: section must never leak into the train.* flatten: the
    launcher only forwards train.{system,model,data} as megatron args and
    the boundary assert inside pre-validation backs it."""
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
                   "--sga-selection-histogram-mode",
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
    cfg.sga.sga_selection_topk_histogram_add_global_allreduce = True
    args = _get_args_megatron(cfg)
    flags = {args[i]: args[i + 1] for i in range(len(args) - 1)
             if args[i].startswith("--sga-")}
    assert flags["--sga-optimizer-residual-mean"] == "true"
    assert flags["--sga-selection-topk-histogram-add-global-allreduce"] == "true"
    print("ok: booleans flatten to true/false spellings")


def test_plugin_param_dead_config_semantics():
    """Megatron dead-config semantics: a key no flag declares is
    loud at parse; a declared key of ANOTHER plugin (or under another
    selection method) is silently ignored — EXCEPT the curated exception
    (block_size under a non-histogram method refuses). Under the
    selection freeze a non-histogram method refuses at the freeze first,
    so the launch-route refusal names the freeze."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_histogram_mode": "balanced",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    cfg, _ = make_base()
    cfg.sga = {**base, "sga_selection_bogus": 1}
    expect_error(lambda: _prevalidate_sga(cfg), "sga_selection_bogus")
    cfg.sga = {**base, "sga_score_extra": 1}
    expect_error(lambda: _prevalidate_sga(cfg), "sga_score_extra")
    # a sibling backend plugin's param is dead config under this backend:
    # silently ignored, at a non-default value and at its default alike
    cfg.sga = {**base, "sga_backend_mask_slices": 3}
    _prevalidate_sga(cfg)
    cfg.sga = {**base, "sga_backend_mask_slices": 2}
    _prevalidate_sga(cfg)
    # the curated exception: block_size is histogram-only
    cfg.sga = {**base, "sga_selection_block_size": 4,
               "sga_backend_emission": "writeback"}
    _prevalidate_sga(cfg)  # histogram balanced + writeback pairing: legal
    cfg.sga = dict(base, sga_selection_method="exact",
                   sga_selection_block_size=4)
    expect_error(lambda: _prevalidate_sga(cfg),
                 "frozen to histogram balanced")
    print("ok: typos loud; sibling/method keys silently ignored; "
          "non-histogram methods meet the selection freeze")


def test_float_spelling_normalized():
    """ratio 10 vs 10.0 parse identically (argparse float typing)."""
    cfg, _ = make_base()
    cfg.sga = {"sga_enabled": True, "sga_score": "update",
               "sga_selection_method": "histogram",
               "sga_selection_histogram_mode": "balanced",
               "sga_selection_ratio_percent": 10,
               "sga_optimizer": "adamw",
               "sga_backend": "torch_pipelined_index"}
    ns = _ns_of(cfg)
    assert ns.sga_selection_ratio_percent == 10.0
    assert isinstance(ns.sga_selection_ratio_percent, float)
    print("ok: float spellings normalize through the parser")


def test_wrong_types_fail_loud():
    """The yaml route renders values into flag STRINGS and re-parses them
    with the real argparse typing, so wrong-typed values fail loud with
    the flag named (and nan/inf spellings pass float typing — validation's
    value rules refuse them)."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_histogram_mode": "balanced",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    cfg, _ = make_base()

    def _with(**over):
        c, _ = make_base()
        c.sga = {**base, **over}
        return c

    expect_error(lambda: _prevalidate_sga(_with(sga_cycle_period=True)),
                 "--sga-cycle-period")
    expect_error(lambda: _prevalidate_sga(_with(sga_backend_index_slices=3.9)),
                 "--sga-backend-index-slices")
    expect_error(lambda: _prevalidate_sga(_with(sga_selection_ratio_percent="abc")),
                 "--sga-selection-ratio-percent")
    expect_error(lambda: _prevalidate_sga(_with(sga_trace_window=True)),
                 "--sga-trace-window")
    expect_error(lambda: _prevalidate_sga(_with(sga_selection_adaptive_warmup_syncs=True)),
                 "--sga-selection-adaptive-warmup-syncs")
    # nan passes argparse float typing; the value rule refuses it
    expect_error(lambda: _prevalidate_sga(
        _with(sga_selection_ratio_percent=float("nan"))),
        "finite float in (0, 100)")
    print("ok: wrong-typed yaml values fail loud (bool/float/str/nan)")


def test_shipped_examples_validate():
    """Every shipped example yaml passes pre-validation as-is: any error
    (a renamed or removed flag left behind in an example, or a selection
    outside the frozen histogram-balanced surface) fails this test."""
    import glob

    examples = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "llama3", "*.yaml")))
    assert examples, "no shipped examples found"
    for path in examples:
        cfg = OmegaConf.load(path)
        _prevalidate_sga(cfg)  # must not raise
    print(f"ok: all {len(examples)} shipped examples validate")


def test_selection_freeze_and_escape():
    """The freeze refuses non-balanced method/mode pairings by default;
    the env escape exists to keep existing baseline scripts running
    and prints loudly every time it does. The escape is read at validation
    time, so dropping the var restores the refusal immediately."""
    base = {"sga_enabled": True, "sga_score": "abs_m",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}

    def _cfg(mode):
        c, _ = make_base()
        c.sga = {**base, "sga_selection_histogram_mode": mode}
        return c

    expect_error(lambda: _prevalidate_sga(_cfg("exact")),
                 "frozen to histogram balanced")
    saved = os.environ.pop("SGA_ALLOW_FROZEN_SELECTION", None)
    try:
        os.environ["SGA_ALLOW_FROZEN_SELECTION"] = "1"
        _prevalidate_sga(_cfg("exact"))  # must not raise
    finally:
        if saved is None:
            os.environ.pop("SGA_ALLOW_FROZEN_SELECTION", None)
        else:
            os.environ["SGA_ALLOW_FROZEN_SELECTION"] = saved
    expect_error(lambda: _prevalidate_sga(_cfg("exact")),
                 "frozen to histogram balanced")
    print("ok: selection freeze refuses by default; env escape opens it "
          "loudly and only while set")


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
