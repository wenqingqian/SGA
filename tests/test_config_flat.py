#!/usr/bin/env python3
"""The flat-config invariants (run: python tests/test_config_flat.py).

The declarations file (sga/config/arguments.py) is the single source of
the config surface; these tests make that STRUCTURAL, not disciplinary:

  INV-1 no plugin vocabulary in validation: the megatron-style validator
       owns no attribution table, no plugin param lists, no expand/guard
       helpers (the whole dead-config machinery is gone — value
       rules and the ONE curated exception are all that remain); the
       fork-side waist registers exactly one non-config sga_ key (its
       override switch) and no --sga-config transport exists anywhere.
  INV-2 flags are the transport: the yaml door and the training entry's
       parser both derive from the same declarations (unknown keys are
       named by the parser itself); the prefix/short-name MACHINES stay
       gone — pinned structurally — and the namespace the plugins read
       (the sga singleton, seeded by copy_from_megatron_args) carries
       every declared key and nothing else.
  INV-3 the declarations stay megatron-shaped: arguments.py contains
       only imports, the two type callables, add_argument_group and
       add_argument calls (AST-pinned — no validation logic grows in
       the file a developer reads for the knob list).
  INV-4 every knob is documented: help non-empty, kind known, default
       explicit; the historical surface is pinned as a golden table;
       the --sga-score choices stay synced with the score registry.
  INV-5 value rules, pinned behaviorally: topk's mode choices stay
       synced with the kernel impls; the selection axis is FROZEN to
       histogram balanced (any other method/mode refuses loud at
       validation); dead config is silently ignored (megatron
       semantics) EXCEPT the curated exception (block_size under a
       non-histogram method); the block comm-path cross-field rules
       and the emission pairing rule refuse loud; the zero-arg
       constructors keep their direct-construction backstops.
  INV-6 every declared dest has exactly one runtime consumer (AST scan
       of the singleton reads): with no central router, a declared dest
       no plugin reads would be silently dropped config, and a read of
       an undeclared key would be a typo riding the singleton.
"""

import ast
import os
import re
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.config import (
    NON_CONFIG_SGA_KEYS,
    copy_from_megatron_args,
    get_args,
    sga_config_attrs,
    set_args,
)
from sga.config import validation as V
from sga.config.arguments import add_sga_args, sga_bool, sga_int_or_null
from sga.config.validation import sga_actions, sga_defaults, validate_sga_args

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGA_DIR = os.path.join(REPO, "sga")


def expect_error(fn, needle):
    try:
        fn()
    except (ValueError, AssertionError) as e:
        assert needle in str(e), f"error missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


def _seed(over=None, base=True):
    """A validated-namespace candidate: declared defaults + the two
    non-sga anchor keys + overrides (base=True pre-sets the minimal
    enabled combination)."""
    ns = SimpleNamespace(**dict(sga_defaults(), train_iters=1000,
                                pipeline_model_parallel_size=1))
    if base:
        ns.sga_enabled = True
        ns.sga_score = "update"
        ns.sga_selection_method = "histogram"
        # the frozen selection combo: validation refuses any other
        # method/mode pairing, so the base config must spell balanced
        ns.sga_selection_histogram_mode = "balanced"
        ns.sga_selection_ratio_percent = 10.0
        ns.sga_backend = "torch_pipelined_index"
    for k, v in (over or {}).items():
        setattr(ns, k, v)
    return ns


def _kind(action) -> str:
    """The declared kind, derived from the argparse type callable."""
    if action.type is sga_bool:
        return "bool"
    if action.type is sga_int_or_null:
        return "int_or_null"
    return {int: "int", float: "float", str: "str"}[action.type]


def _fork_waist_path():
    root = os.environ.get("MEGATRON_SGA_DIR") or os.path.join(
        os.path.dirname(REPO), "Megatron-LM")
    return os.path.join(root, "megatron", "core", "sga_waist.py")


# ---------------------------------------------------------------- INV-1
def test_inv1_validation_owns_no_plugin_vocabulary():
    """No central plugin-param machinery exists anywhere: validation has
    neither the old attribution table and expand/guard helpers nor the
    ownership map, and topk carries no UNLOCKS table to stay synced
    with — a plugin's params are the plugin's own business (read
    straight off the singleton)."""
    for gone in ("_AXIS_PARAMS", "expand_plugin_params",
                 "guard_foreign_params", "_plugin_param_owners",
                 "explicit_from_flat", "validate_sga_flat",
                 "collect_sga_flat"):
        assert not hasattr(V, gone), gone
    import sga.plugins as plugins_mod
    import sga.plugins.selection.topk as topk_mod

    assert not hasattr(plugins_mod, "expand_plugin_params")
    assert not hasattr(topk_mod, "UNLOCKS")
    # the only plugins knowledge validation has is the name lookup —
    # imported INSIDE the function (module level would be a cycle)
    assert not hasattr(V, "plugin_class")
    assert "plugin_class" in V.validate_sga_args.__code__.co_names
    print("ok: validation owns no plugin vocabulary")


def test_inv1_fork_registers_exactly_the_override_switch():
    """The fork waist owns exactly ONE sga_ key — its override switch —
    and no --sga-config transport remains in the fork."""
    src = open(_fork_waist_path(), encoding="utf-8").read()
    dests = set(re.findall(r"dest=[\"']([a-z_]+)[\"']", src))
    assert dests == {"sga_allow_config_override"}, dests
    # no JSON transport: the flag itself is gone
    assert "--sga-config" not in src
    assert "add_argument" in src and "json" not in src
    print("ok: fork registers exactly the override switch")


# ---------------------------------------------------------------- INV-2
def test_inv2_unknown_key_named_by_the_parser():
    """The yaml door derives its accepted keys from the declarations —
    an unknown key is named by the same parser the training entry uses
    (there is no second key table anywhere in the pipeline)."""
    from omegaconf import OmegaConf

    from sga.launcher.backend_megatron import _prevalidate_sga

    cfg = OmegaConf.create({
        "experiment": {"exp_name": "t", "exp_dir": "/tmp/x",
                       "task": {"type": "train", "backend": "megatron",
                                "entrypoint": "./sga/train_gpt.py"},
                       "runner": {"type": "ssh"}},
        "sga": {"sga_enabled": "true", "sga_score": "update",
                "sga_nope": 1},
        "train": {"system": {"tensor_model_parallel_size": 1},
                  "model": {"num_layers": 2},
                  "data": {"data_path": "/tmp/x"}},
    })
    expect_error(lambda: _prevalidate_sga(cfg), "sga_nope")
    print("ok: unknown keys are named by the shared parser")


def test_inv2_prefix_machines_are_gone():
    """The prefix/short-name machines must stay absent — pinned ABSENT
    so they cannot creep back: no
    engine_section_defaults, no runtime materialization of a nested read
    shape, no expand_plugin_params with a dest_prefix, and no name
    assembly/stripping anywhere in the config chain's source (the
    sga_* COLLECTION — sga_config_attrs' startswith filter, the
    declarations themselves — is the deliberate exemption)."""
    import inspect

    import sga.backends.nccl_pipelined_index as npi
    import sga.backends.torch_pipelined_index as tpi
    import sga.backends.torch_pipelined_mask as tpm
    import sga.config as config_pkg
    import sga.engine as engine_mod
    import sga.launcher.backend_megatron as launcher_mod
    import sga.plugins.optimizer.adamw as adamw_mod
    import sga.plugins.scores as scores_mod
    import sga.plugins.selection.topk as topk_mod
    import sga.runtime as runtime_mod
    import sga.trace as trace_mod
    from sga import plugins as plugins_mod

    assert not hasattr(V, "engine_section_defaults")
    assert not hasattr(runtime_mod, "materialize_engine_config")
    assert not hasattr(plugins_mod, "_resolved")
    assert not hasattr(plugins_mod, "expand_plugin_params")
    for mod in (engine_mod, trace_mod):
        for leftover in ("_CYCLE_DEFAULTS", "_CHECKS_DEFAULTS",
                         "_ALWAYS_DEFAULTS", "_TOP_DEFAULTS",
                         "_TRACE_DEFAULTS"):
            assert not hasattr(mod, leftover), (mod.__name__, leftover)
    for mod in (V, runtime_mod, engine_mod, trace_mod, plugins_mod,
                topk_mod, adamw_mod, tpi, tpm, npi,
                launcher_mod, config_pkg, scores_mod):
        src = inspect.getsource(mod)
        assert 'f"sga_{' not in src and "f'sga_{" not in src, mod.__name__
        assert "len(prefix)" not in src, mod.__name__
        assert "dest_prefix" not in src, mod.__name__
        # no hardcoded strip spellings either (removeprefix / slice-off
        # of the 4-char "sga_" prefix)
        assert "removeprefix" not in src, mod.__name__
        assert "[4:]" not in src, mod.__name__
    print("ok: the prefix/short-name machines are gone (pinned absent)")


def test_inv2_singleton_carries_every_declared_key():
    """The sga singleton is the config carrier: seeded from a megatron
    namespace it carries every declared dest (the copy drops nothing),
    plus exactly the two anchor keys validation needs — and the
    NON_CONFIG_SGA_KEYS exclusion list never eats a real config key."""
    megatron_ns = SimpleNamespace(
        **dict(sga_defaults(), train_iters=123,
               pipeline_model_parallel_size=2,
               sga_allow_config_override=True, sga_config='{"v": 1}'))
    copy_from_megatron_args(megatron_ns)
    seeded = get_args()
    assert set(sga_config_attrs(seeded)) == set(sga_defaults())
    assert seeded.train_iters == 123
    assert seeded.pipeline_model_parallel_size == 2
    assert not hasattr(seeded, "sga_allow_config_override")
    assert not hasattr(seeded, "sga_config")
    # the exclusion list is a fixed two-entry contract with the waist
    # and the retired v1 transport — a real config key landing there
    # would vanish from every checkpoint comparison
    assert set(NON_CONFIG_SGA_KEYS) == {"sga_allow_config_override",
                                        "sga_config"}
    assert not set(NON_CONFIG_SGA_KEYS) & set(sga_defaults())
    print("ok: the singleton carries every declared key, nothing else")


# ---------------------------------------------------------------- INV-3
def test_inv3_arguments_file_is_pure_declarations():
    """arguments.py holds only declarations: allowed nodes are imports,
    the two argparse type callables, add_argument_group/add_argument
    calls and the add_sga_args dispatcher — validation logic must not
    creep into the developer-facing knob list."""
    tree = ast.parse(open(os.path.join(SGA_DIR, "config", "arguments.py"),
                          encoding="utf-8").read())
    def _declared_fn(name):
        return (name in ("add_sga_args", sga_bool.__name__,
                         sga_int_or_null.__name__)
                or (name.startswith("_add_sga_") and name.endswith("_args")))

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            continue
        if isinstance(node, ast.FunctionDef):
            assert _declared_fn(node.name), node.name
            continue
        if isinstance(node, ast.Assign):
            continue
        raise AssertionError(f"unexpected top-level node: {type(node).__name__}")
    # every METHOD call is a declaration call; no sga imports, no IO
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.ImportFrom):
                assert stmt.module == "argparse", stmt.module
            if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                assert stmt.func.attr in (
                    "add_argument", "add_argument_group", "strip", "lower",
                    "ArgumentTypeError",
                ), f"{node.name}: unexpected call .{stmt.func.attr}()"
            if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name):
                assert stmt.func.id in (
                    "int", "str", "bool", "float", "isinstance",
                    sga_bool.__name__, sga_int_or_null.__name__,
                    "ArgumentTypeError", "_add_sga_core_args",
                    "_add_sga_cycle_args", "_add_sga_checks_args",
                    "_add_sga_trace_args", "_add_sga_score_args",
                    "_add_sga_selection_args", "_add_sga_optimizer_args",
                    "_add_sga_backend_args",
                ), f"{node.name}: unexpected call {stmt.func.id}()"
    print("ok: arguments.py is pure declarations (AST-pinned)")


# ---------------------------------------------------------------- INV-4
def test_inv4_every_knob_documented_and_typed():
    for dest, action in sga_actions().items():
        assert isinstance(action.help, str) and action.help.strip(), dest
        assert action.default is not None or dest in (
            "sga_score", "sga_selection_method", "sga_selection_ratio_percent",
            "sga_backend", "sga_trace_dir", "sga_backend_timer",
            "sga_trace_branch_points", "sga_cycle_start_iter_percent",
            "sga_cycle_full_sync_interval", "sga_always_sync_layers"), dest
        assert _kind(action) in ("bool", "int", "int_or_null",
                                 "float", "str"), dest
    print("ok: every knob has help, explicit default and known kind")


def test_inv4_historical_surface_pinned():
    """Golden pin of the whole surface: changing a default/type/choices is
    a conscious act that updates this table."""
    pin = {
        "sga_enabled": (False, "bool", ()),
        "sga_shadow": (False, "bool", ()),
        "sga_shadow_score": (True, "bool", ()),
        "sga_cycle_start_iter": (0, "int", ()),
        "sga_cycle_start_iter_percent": (None, "float", ()),
        "sga_cycle_period": (1, "int", ()),
        "sga_cycle_decouple_full_sync_interval_and_period": (False, "bool", ()),
        "sga_cycle_full_sync_interval": (None, "int", ()),
        "sga_checks": (True, "bool", ()),
        "sga_checks_count_guard": (True, "bool", ()),
        "sga_checks_support_consistency": (True, "bool", ()),
        "sga_checks_norm_spread": (True, "bool", ()),
        "sga_checks_rng_bracket": (True, "bool", ()),
        "sga_always_sync_layers": (None, "str", ()),
        "sga_trace_dir": (None, "str", ()),
        "sga_trace_branch_points": (None, "str", ()),
        "sga_trace_window": (100, "int", ()),
        "sga_trace_dump_dtype": ("fp32", "str", ("fp32", "bf16")),
        "sga_trace_async": (False, "bool", ()),
        "sga_trace_async_backend": ("cpu_snapshot", "str",
                                    ("cpu_snapshot", "pinned_cuda")),
        "sga_trace_async_queue_depth": (2, "int", ()),
        "sga_trace_async_worker_count": (1, "int", ()),
        "sga_trace_async_max_workset_gib": (16.0, "float", ()),
        "sga_score": (None, "str", ("abs_g", "abs_m", "coi", "update")),
        "sga_selection": ("topk", "str", ()),
        "sga_selection_method": (None, "str", ("adaptive", "exact", "histogram")),
        "sga_selection_ratio_percent": (None, "float", ()),
        "sga_selection_histogram_mode": ("exact", "str", ("balanced", "exact", "fast")),
        "sga_selection_topk_histogram_add_global_allreduce": (False, "bool", ()),
        "sga_selection_adaptive_warmup_syncs": (5, "int_or_null", ()),
        "sga_selection_adaptive_warmup_mode": ("fast", "str", ("balanced", "exact", "fast")),
        "sga_selection_block_size": (1, "int", ()),
        "sga_selection_block_reduce": ("mean", "str", ("mean", "max")),
        "sga_optimizer": ("adamw", "str", ()),
        "sga_optimizer_residual_mean": (False, "bool", ()),
        "sga_optimizer_full_sync_inactive_lr_scale": (1.0, "float", ()),
        "sga_optimizer_compact_search": ("table", "str", ("table", "bisection")),
        "sga_backend": ("dense", "str", ()),
        "sga_backend_emission": ("auto", "str", ("auto", "compact", "writeback")),
        "sga_backend_layout": ("auto", "str", ("auto", "er", "eb", "bb")),
        "sga_backend_high_priority_group": (False, "bool", ()),
        "sga_backend_index_slices": (2, "int", ()),
        "sga_backend_mask_slices": (2, "int", ()),
        "sga_backend_timer": (None, "str", ()),
    }
    actions = sga_actions()
    assert set(pin) == set(actions), set(pin) ^ set(actions)
    for dest, (default, kind, choices) in pin.items():
        assert actions[dest].default == default, dest
        assert _kind(actions[dest]) == kind, dest
        assert tuple(getattr(actions[dest], "choices", None) or ()) == choices, dest
    print(f"ok: {len(pin)}-row golden pin holds")


def test_inv4_score_choices_synced_with_registry():
    """--sga-score choices are the registered score names (the registry
    is data; the declared choices must not drift from it)."""
    from sga.plugins.scores import score_names

    assert sga_actions()["sga_score"].choices == tuple(sorted(score_names()))
    print("ok: score choices synced with the registry")


# ---------------------------------------------------------------- INV-5
def test_inv5_topk_mode_choices_synced_with_kernel():
    """The method choices must stay sync'd with the impl classes topk
    dispatches to (a stale choice = a config-time lie): the declared
    choice set IS TopkSelection's method dispatch table, and every
    declared method resolves to its expected impl class off the
    singleton."""
    from sga.plugins.selection._adaptive import AdaptiveTau
    from sga.plugins.selection._exact import ExactKthvalue
    from sga.plugins.selection._histogram import HistogramSelect
    from sga.plugins.selection.topk import TopkSelection

    dispatch = {"adaptive": AdaptiveTau, "exact": ExactKthvalue,
                "histogram": HistogramSelect}
    methods = sga_actions()["sga_selection_method"].choices
    assert tuple(methods) == tuple(dispatch), (methods, tuple(dispatch))
    for method, impl_cls in dispatch.items():
        set_args(_seed({"sga_selection_method": method}))
        sel = TopkSelection()
        assert type(sel._impl) is impl_cls, (method, type(sel._impl))
    print("ok: topk method choices synced with kernel impls")


def test_inv5_dead_config_silently_ignored_except_curated():
    """Megatron dead-config semantics: a knob of another plugin/method
    is silently ignored at ANY value (no explicitness detection, no
    ownership machinery) — EXCEPT the curated exception list
    (_dead_config_exceptions), whose only entry is block_size under a
    non-histogram method (its silent death would mean a wrong
    experiment, not a no-op). The selection freeze refuses every
    non-balanced method/mode BEFORE the curated gate, so through
    validate_sga_args the freeze's refusal is the honest first error;
    the gate itself stays pinned behaviorally via a direct call."""
    # sibling knobs at loud non-default values: silently ignored (the
    # adaptive-family knobs are dead under the frozen histogram
    # selection)
    validate_sga_args(_seed({"sga_selection_adaptive_warmup_mode": "exact"}))
    validate_sga_args(_seed({"sga_selection_adaptive_warmup_syncs": None,
                             "sga_selection_adaptive_warmup_mode": "fast"}))
    validate_sga_args(_seed({"sga_backend": "torch_pipelined_mask",
                             "sga_backend_index_slices": 7,
                             "sga_backend_high_priority_group": True}))
    validate_sga_args(_seed({"sga_selection_block_size": 1,
                             "sga_selection_block_reduce": "max"}))
    # the curated gate, called directly (its block_size refusal names
    # the method — the comm cross-field rule stays out of the way)
    from types import SimpleNamespace
    expect_error(
        lambda: V._dead_config_exceptions(SimpleNamespace(
            **{**sga_defaults(), "sga_selection_method": "exact",
               "sga_selection_block_size": 4})),
        "only applies to the histogram method")
    # through validation the freeze preempts it — for a non-histogram
    # method at ANY block_size, default included (the freeze is
    # value-level: argparse cannot distinguish set-at-default from
    # unset, so 'exact' at defaults refuses too)
    expect_error(
        lambda: validate_sga_args(_seed({"sga_selection_method": "exact",
                                         "sga_backend": "torch_pipelined_mask",
                                         "sga_selection_block_size": 4})),
        "frozen to histogram balanced")
    expect_error(
        lambda: validate_sga_args(_seed({"sga_selection_method": "adaptive",
                                         "sga_backend": "torch_pipelined_mask",
                                         "sga_selection_block_size": 4})),
        "frozen to histogram balanced")
    expect_error(
        lambda: validate_sga_args(_seed({"sga_selection_method": "exact"})),
        "frozen to histogram balanced")
    # the curated list stays curated: exactly one gate lives there
    src = open(V.__file__, encoding="utf-8").read()
    body = src.split("def _dead_config_exceptions")[1].split("\ndef ")[0]
    gated = set(re.findall(r"sga_[a-z_0-9]+", body)) - {
        "sga_defaults", "sga_selection_method"}  # lookup + gate input
    assert gated == {"sga_selection_block_size"}, gated
    print("ok: dead config silently ignored; block_size is the one gate "
          "(shadowed by the selection freeze through validation)")


def test_inv5_block_size_rules():
    """Block-granularity value rules: power-of-two [1, 1024] range."""
    for bad in (0, 3, 2048, -16, True, 2.0):
        expect_error(
            lambda b=bad: validate_sga_args(
                _seed({"sga_selection_block_size": b})),
            "power of two")
    validate_sga_args(_seed({"sga_selection_block_size": 64,
                             "sga_backend": "torch_pipelined_mask"}))
    print("ok: block_size power-of-two range rule")


def test_inv5_block_backend_cross_field_rules():
    """Block-granularity comm-path cross-field rules under the layout
    taxonomy: the mask stays the canonical support form, and
    the layout routes which index form the transport consumes. bb rides
    the torch block path (writeback-only) or the nccl descriptor API
    (block_size bounded to a pow2 in [4, 64]); er/eb ride the element
    idx paths — so nccl+block 16 is a legitimate bb config, and the
    refusals are the dead/misaligned combinations instead."""
    base = {"sga_selection_block_size": 16}
    # tpi with the auto default: layout auto-derives bb and emission
    # resolves to compact here — the refusal must say so (the hint text
    # is the user's only signpost)
    expect_error(lambda: validate_sga_args(
        _seed({**base})), "resolves to 'compact'")
    expect_error(lambda: validate_sga_args(
        _seed({**base, "sga_backend_emission": "compact"})), "'writeback'")
    # nccl bb outside the descriptor x domain [4, 64]
    expect_error(lambda: validate_sga_args(
        _seed({**base, "sga_selection_block_size": 128,
               "sga_backend": "nccl_pipelined_index"})),
        "power of two in [4, 64]")
    # er names element selection — block_size > 1 contradicts it
    expect_error(lambda: validate_sga_args(
        _seed({**base, "sga_backend_layout": "er"})),
        "'er' (element selection + element index) refuses")
    # eb/bb name block selection — block_size 1 contradicts them
    for layout in ("eb", "bb"):
        expect_error(lambda l=layout: validate_sga_args(
            _seed({"sga_backend_layout": l})),
            "refuses sga_selection_block_size == 1")
    # an explicit layout on the mask backend is dead config
    expect_error(lambda: validate_sga_args(
        _seed({**base, "sga_backend_layout": "bb",
               "sga_backend": "torch_pipelined_mask"})),
        "refused with backend torch_pipelined_mask")
    # allowed: tpi+writeback (bb), tpi+eb+compact (element transport),
    # nccl+bb in domain, nccl+eb (element transport), the mask backend
    validate_sga_args(_seed(
        {**base, "sga_backend_emission": "writeback"}))
    validate_sga_args(_seed(
        {**base, "sga_backend_layout": "eb"}))
    validate_sga_args(_seed(
        {**base, "sga_backend": "nccl_pipelined_index"}))
    validate_sga_args(_seed(
        {**base, "sga_backend_layout": "eb",
         "sga_backend": "nccl_pipelined_index"}))
    validate_sga_args(_seed({**base, "sga_backend": "torch_pipelined_mask"}))
    print("ok: layout cross-field refusals (dead layouts, bb domains)")


def test_inv5_block_fast_mode_refusal():
    """fast is frozen outright with the rest of the non-balanced
    selections: the freeze refuses before the block rules, at block
    granularity and at element granularity alike (the fast+block
    rationale gate below the freeze stays as shadowed defense).
    balanced+block stays legal."""
    base = {"sga_selection_block_size": 16,
            "sga_backend": "torch_pipelined_mask"}
    expect_error(
        lambda: validate_sga_args(
            _seed({**base, "sga_selection_histogram_mode": "fast"})),
        "frozen to histogram balanced")
    validate_sga_args(_seed(
        {**base, "sga_selection_histogram_mode": "balanced"}))
    # fast at element granularity (block_size at its default): equally
    # frozen — the mode itself, not the block pairing, is what refuses
    expect_error(
        lambda: validate_sga_args(
            _seed({"sga_selection_histogram_mode": "fast"})),
        "frozen to histogram balanced")
    print("ok: fast frozen at any block_size; balanced+block legal")


def test_inv5_method_required_and_choices_on_real_routes():
    """A missing method surfaces as the required error on the namespace
    route; a bogus value is refused by the parser's choices on the two
    real routes (the launcher pre-validation parses the emitted flags —
    there is no second vocabulary check to drift)."""
    from omegaconf import OmegaConf

    from sga.launcher.backend_megatron import _prevalidate_sga

    expect_error(lambda: validate_sga_args(_seed(
        {"sga_selection_method": None})),
        "sga_selection_method is required by topk")

    def _cfg(**sga_over):
        return OmegaConf.create({
            "experiment": {"exp_name": "t", "exp_dir": "/tmp/x",
                           "task": {"type": "train", "backend": "megatron",
                                    "entrypoint": "./sga/train_gpt.py"},
                           "runner": {"type": "ssh"}},
            "sga": {"sga_enabled": "true", "sga_score": "update",
                    "sga_selection_method": "hystogram",
                    "sga_selection_ratio_percent": 10.0,
                    "sga_backend": "torch_pipelined_index", **sga_over},
            "train": {"system": {"tensor_model_parallel_size": 1},
                      "model": {"num_layers": 2, "train_iters": 10},
                      "data": {"data_path": "/tmp/x"}},
        })

    expect_error(lambda: _prevalidate_sga(_cfg()),
                 "invalid choice: 'hystogram'")
    print("ok: method required/choices enforced on the real routes")


def test_inv5_zero_arg_construction_backstops():
    """The zero-arg constructors keep their direct-construction
    backstops (the config route validates earlier): emission pairing,
    slice lower bounds and the ratio domain."""
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
    from sga.backends.torch_pipelined_mask import TorchPipelinedMaskBackend
    from sga.plugins import build_backend_plugin
    from sga.plugins.selection.topk import TopkSelection

    # compact on the writeback-inherent backend: backstop refusal
    set_args(_seed({"sga_backend": "torch_pipelined_mask",
                    "sga_backend_emission": "compact"}))
    expect_error(TorchPipelinedMaskBackend,
                 "only supported by torch_pipelined_index")
    expect_error(lambda: build_backend_plugin("torch_pipelined_mask"),
                 "only supported by torch_pipelined_index")
    # slice guards (direct construction bypasses argparse int typing: a
    # fractional slice count would truncate, not refuse)
    set_args(_seed({"sga_backend_index_slices": 0}))
    expect_error(TorchPipelinedIndexBackend, "index_slices must be an int >= 1")
    set_args(_seed({"sga_backend": "torch_pipelined_mask",
                    "sga_backend_mask_slices": 0}))
    expect_error(TorchPipelinedMaskBackend, "mask_slices must be an int >= 1")
    # ratio domain (argparse typing cannot see a direct namespace)
    set_args(_seed({"sga_selection_ratio_percent": 0.0}))
    expect_error(TopkSelection, "(0, 100)")
    # emission auto resolution: the flag default takes the backend's
    # inherent form; an explicit value survives to the instance
    set_args(_seed({"sga_backend": "torch_pipelined_index"}))
    assert build_backend_plugin("torch_pipelined_index").emission == "compact"
    set_args(_seed({"sga_backend": "torch_pipelined_index",
                    "sga_backend_emission": "writeback"}))
    assert build_backend_plugin(
        "torch_pipelined_index").emission == "writeback"
    set_args(_seed({"sga_backend": "torch_pipelined_mask"}))
    assert build_backend_plugin("torch_pipelined_mask").emission == "writeback"
    print("ok: zero-arg construction backstops hold")


def test_inv5_emission_pairing_rule():
    """'compact' is a switch only torch_pipelined_index implements —
    refused on the writeback-inherent backends at validation (the
    launcher catches it before any process is spawned); the retired
    merge_join_dir dest fails closed as an unknown key on the yaml
    route (the resume-side refusal is pinned in test_resume_guard)."""
    from omegaconf import OmegaConf

    from sga.launcher.backend_megatron import _prevalidate_sga

    for wb in ("torch_pipelined_mask", "nccl_pipelined_index"):
        expect_error(
            lambda w=wb: validate_sga_args(
                _seed({"sga_backend": w, "sga_backend_emission": "compact"})),
            "only supported by torch_pipelined_index")

    def _cfg(sga_over):
        return OmegaConf.create({
            "experiment": {"exp_name": "t", "exp_dir": "/tmp/x",
                           "task": {"type": "train", "backend": "megatron",
                                    "entrypoint": "./sga/train_gpt.py"},
                           "runner": {"type": "ssh"}},
            "sga": {"sga_enabled": "true", "sga_score": "update",
                    "sga_selection_method": "histogram",
                    "sga_selection_ratio_percent": 10.0,
                    "sga_backend": "torch_pipelined_index", **sga_over},
            "train": {"system": {"tensor_model_parallel_size": 1},
                      "model": {"num_layers": 2, "train_iters": 10},
                      "data": {"data_path": "/tmp/x"}},
        })

    expect_error(lambda: _prevalidate_sga(_cfg(
        {"sga_optimizer_merge_join_dir": "table"})), "unknown sga")
    print("ok: emission pairing + retired-key fail-closed")


def test_always_sync_layers_parser_edge_table():
    """Direct pin of parse_layer_patterns' documented edge cases (the
    cheapest defense layer of the always-sync knob): broken joins and
    whitespace-only elements refuse, non-string shapes refuse, the
    canonical form strips/dedupes/SORTS (order-insensitive, resume-safe),
    the unset spellings pass through as None, the launcher's null token
    refuses, and validation names the key on every refusal."""
    parse = V.parse_layer_patterns
    assert parse(None) is None
    assert parse("") is None            # hand-written empty CLI value
    assert parse("   ") is None         # the launcher's yaml-[] token
    assert parse([]) is None            # the yaml OFF spelling itself
    assert parse(("a", "b")) == ["a", "b"]   # tuple input accepted
    assert parse("b, a ,a") == ["a", "b"]    # strip + dedupe + sorted
    for bad in ("a,,b", "a, ,b", ["a", 1], [1, 2], 5, "null", ["null"]):
        try:
            parse(bad)
            raise AssertionError(f"parse_layer_patterns accepted {bad!r}")
        except ValueError:
            pass
    # validation prefixes every parser refusal with the key name
    expect_error(lambda: validate_sga_args(_seed(
        {"sga_always_sync_layers": "a,,b"})),
        "sga_always_sync_layers: patterns must be non-empty")
    expect_error(lambda: validate_sga_args(_seed(
        {"sga_always_sync_layers": "null"})),
        "an explicit null is NOT the unset spelling")
    print("ok: always-sync pattern parser edge table holds")


# ---------------------------------------------------------------- INV-6
_CONSUMER_FILES = {
    "engine": ("sga/engine.py", "sga/runtime.py", "sga/trace.py"),
    "route": ("sga/train_gpt.py",),
    "selection": ("sga/plugins/selection/topk.py",),
    "optimizer": ("sga/plugins/optimizer/adamw.py",),
    "backend": ("sga/backends/torch_pipelined_index.py",
                "sga/backends/torch_pipelined_mask.py",
                "sga/backends/nccl_pipelined_index.py"),
}
_AXIS_KEYS = {"sga_score", "sga_selection", "sga_optimizer", "sga_backend"}


def _singleton_reads(relpath):
    """The sga_* attributes a module READS off an args-like object
    (AST Attribute loads on a bare name — docstrings and comments
    cannot pollute the scan; sga.config itself is excluded: validation
    reads the whole surface to police it, consuming nothing)."""
    tree = ast.parse(open(os.path.join(REPO, relpath),
                          encoding="utf-8").read())
    out = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and node.attr.startswith("sga_")
                and isinstance(node.value, ast.Name)
                and node.value.id in ("args", "cfg", "ns", "megatron_args")):
            out.add(node.attr)
    return out


def test_inv6_every_declared_dest_has_one_consumer():
    """Completeness: plugins read the singleton directly, so a declared
    dest no runtime consumer reads would be silently DROPPED config,
    and a read of an undeclared key would be a typo (or a retirement
    survivor) riding the singleton. The AST read-scan must cover the
    declared surface exactly — every dest claimed once, the axis
    identity keys excepted (the optimizer legitimately re-reads
    sga_score; the backends share sga_backend_emission)."""
    groups = {name: set().union(*[_singleton_reads(p) for p in paths])
              for name, paths in _CONSUMER_FILES.items()}
    legal = set(NON_CONFIG_SGA_KEYS)
    for name, reads in groups.items():
        stale = reads - set(sga_defaults()) - legal
        assert not stale, f"{name} reads undeclared keys: {stale}"
    non_axis = [{k for k in reads if k not in _AXIS_KEYS}
                for reads in groups.values()]
    for i, ga in enumerate(non_axis):
        for gb in non_axis[i + 1:]:
            assert not ga & gb, f"double-claimed non-axis dests: {ga & gb}"
    covered = set().union(*groups.values()) - legal
    declared = set(sga_defaults())
    _NON_CONSUMERS = {"sga_cycle_start_iter_percent", "sga_checks"}
    assert not declared - covered - _NON_CONSUMERS, \
        f"unclaimed dests: {declared - covered - _NON_CONSUMERS}"
    # the percent spelling is resolved away by validation (write-back
    # onto sga_cycle_start_iter), and sga_checks is consumed by
    # validation itself (the master write-back — sga.config is excluded
    # from this scan by design) — the two deliberate non-consumers
    print("ok: every declared dest has exactly one consumer class")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL FLAT-CONFIG INVARIANT TESTS PASSED")
