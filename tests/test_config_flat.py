#!/usr/bin/env python3
"""The flat-config invariants (run: python tests/test_config_flat.py).

The declarations file (sga/config/arguments.py) is the single source of
the config surface; these tests make that STRUCTURAL, not disciplinary:

  INV-1 no central plugin table: validation owns no plugin vocabulary
       (the rewrite deleted _AXIS_PARAMS and the expand/guard helpers
       from it — each plugin's params are the plugin's own business);
       the fork-side waist registers exactly one non-config sga_ key
       (its override switch) and no --sga-config remains anywhere.
  INV-2 flags are exactly the declarations: the launcher's key
       check and the training entry's parser all derive
       from sga_actions(); the canonical flat dict carries every
       declared key; and the prefix/short-name MACHINES are gone for
       good — pinned structurally (no engine_section_defaults, no
       materialize_engine_config, no dest_prefix parameter, no
       f"sga_{...}" name assembly or prefix stripping in the config
       chain's source).
  INV-3 the declarations stay megatron-shaped: arguments.py contains
       only imports, the two type callables, add_argument_group and
       add_argument calls (AST-pinned — no validation logic grows in
       the file a developer reads for the knob list).
  INV-4 every knob is documented: help non-empty, kind known, default
       explicit; the historical surface is pinned as a golden table.
  INV-5 value rules, pinned behaviorally: topk's mode choices stay
       synced with the kernel impls; the dead-config (param of another
       method) refusals name the own vocabulary on BOTH layers
       (validation and the constructor); a missing/invalid method keeps
       the plain required/choices error; each plugin's constructor
       rejects foreign and missing params itself.
  INV-6 every declared dest has exactly one consumer class (completeness):
       with prefix routing gone, a declared-but-unclaimed dest would be
       silently dropped — the consumer map (engine read set, axis
       identity keys, plugin PARAMS vocabularies, explicit exemptions)
       must cover the declared surface exactly.
"""

import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.config import validation as V
from sga.config.arguments import add_sga_args, sga_bool, sga_int_or_null
from sga.config.validation import (
    _param_kind,
    declared_choices,
    explicit_from_flat,
    sga_actions,
    sga_defaults,
    validate_sga_flat,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGA_DIR = os.path.join(REPO, "sga")


def expect_error(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"error missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


def _fork_waist_path():
    root = os.environ.get("MEGATRON_SGA_DIR") or os.path.join(
        os.path.dirname(REPO), "Megatron-LM")
    return os.path.join(root, "megatron", "core", "sga_waist.py")


# ---------------------------------------------------------------- INV-1
def test_inv1_validation_owns_no_plugin_vocabulary():
    """No central plugin-param table exists: validation has neither the
    attribution table nor the expand/guard helpers (a plugin's params
    are the plugin's own business); the plugins package owns the shared
    expansion path."""
    assert not hasattr(V, "_AXIS_PARAMS")
    assert not hasattr(V, "expand_plugin_params")
    assert not hasattr(V, "guard_foreign_params")
    # the trace-activation set stays synced with the declared trace
    # surface (an explicit enumeration: a new sga_trace_* dest that
    # missed it would silently skip collector activation)
    assert set(V._TRACE_KEYS) == {
        d for d in sga_defaults() if d.startswith("sga_trace_")}
    print("ok: validation owns no plugin vocabulary")


def test_inv1_fork_registers_exactly_the_override_switch():
    """The fork waist owns exactly ONE sga_ key — its override switch —
    and no --sga-config transport remains in the fork."""
    src = open(_fork_waist_path(), encoding="utf-8").read()
    dests = set(re.findall(r"dest=[\"']([a-z_]+)[\"']", src))
    assert dests == {"sga_allow_config_override"}, dests
    # no JSON transport: the flag itself is gone (the retirement exclusion
    # for old checkpoint namespaces is the only tolerated mention)
    assert "--sga-config" not in src
    assert "add_argument" in src and "json" not in src
    print("ok: fork registers exactly the override switch")


# ---------------------------------------------------------------- INV-2
def test_inv2_launcher_keys_derive_from_declarations():
    """The launcher's accepted yaml keys ARE the parser's dests — there
    is no second key table anywhere in the pipeline."""
    expect_error(lambda: validate_sga_flat({"sga_nope": 1}), "sga_nope")
    print("ok: launcher/declaration vocabularies coincide")


def test_inv2_prefix_machines_are_gone():
    """The prefix/short-name machines this suite once pinned in place are
    deleted — and pinned ABSENT so they cannot creep back: no
    engine_section_defaults, no runtime materialization of a nested read
    shape, no dest_prefix in expand_plugin_params, no engine/trace
    fallback default dicts, and no name assembly/stripping anywhere in
    the config chain's source (the narrow-waist sga_* COLLECTION —
    sga_actions/collect_sga_flat/the fork waist — is the deliberate
    exemption, asserted at the yaml door by the prefix gate)."""
    import inspect

    import sga.engine as engine_mod
    import sga.runtime as runtime_mod
    import sga.trace as trace_mod
    from sga import plugins as plugins_mod

    assert not hasattr(V, "engine_section_defaults")
    assert not hasattr(runtime_mod, "materialize_engine_config")
    assert not hasattr(plugins_mod, "_resolved")
    for mod in (engine_mod, trace_mod):
        for leftover in ("_CYCLE_DEFAULTS", "_CHECKS_DEFAULTS",
                         "_ALWAYS_DEFAULTS", "_TOP_DEFAULTS",
                         "_TRACE_DEFAULTS"):
            assert not hasattr(mod, leftover), (mod.__name__, leftover)
    sig = inspect.signature(plugins_mod.expand_plugin_params)
    assert "dest_prefix" not in sig.parameters, sig
    # source-level pin: no f"sga_{...}" name assembly and no prefix
    # stripping in the config chain
    import sga.backends.nccl_pipelined_index as npi
    import sga.backends.torch_pipelined_index as tpi
    import sga.backends.torch_pipelined_mask as tpm
    import sga.config as config_pkg
    import sga.launcher.backend_megatron as launcher_mod
    import sga.plugins.optimizer.adamw as adamw_mod
    import sga.plugins.scores as scores_mod
    import sga.plugins.selection.topk as topk_mod

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


def test_inv2_canonical_carries_every_declared_key():
    """A minimal enabled config yields the complete canonical flat dict:
    every declared dest is present at its resolved value (percent
    resolved away by design), ready for direct full-name reads."""
    canon = validate_sga_flat({
        "sga_enabled": True, "sga_score": "update",
        "sga_selection_method": "exact", "sga_selection_ratio_percent": 5.0,
        "sga_optimizer": "adamw", "sga_backend": "torch_pipelined_index"})
    declared = set(sga_defaults())
    assert set(canon) == declared - {"sga_cycle_start_iter_percent"}
    print("ok: the canonical flat dict carries every declared key")


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
            "sga_backend", "sga_trace_dir",
            "sga_trace_branch_points", "sga_cycle_start_iter_percent",
            "sga_cycle_full_sync_interval", "sga_always_sync_layers"), dest
        assert _param_kind(dest) in ("bool", "int", "int_or_null",
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
        "sga_checks_count_guard": (True, "bool", ()),
        "sga_always_sync_layers": (None, "str", ()),
        "sga_trace_dir": (None, "str", ()),
        "sga_trace_branch_points": (None, "str", ()),
        "sga_trace_window": (100, "int", ()),
        "sga_score": (None, "str", ("abs_g", "abs_m", "coi", "update")),
        "sga_selection": ("topk", "str", ()),
        "sga_selection_method": (None, "str", ("adaptive", "exact", "histogram")),
        "sga_selection_ratio_percent": (None, "float", ()),
        "sga_selection_histogram_mode": ("exact", "str", ("balanced", "exact", "fast")),
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
        "sga_backend_high_priority_group": (False, "bool", ()),
        "sga_backend_index_slices": (2, "int", ()),
        "sga_backend_mask_slices": (2, "int", ()),
    }
    actions = sga_actions()
    assert set(pin) == set(actions), set(pin) ^ set(actions)
    for dest, (default, kind, choices) in pin.items():
        assert actions[dest].default == default, dest
        assert _param_kind(dest) == kind, dest
        assert tuple(getattr(actions[dest], "choices", None) or ()) == choices, dest
    print(f"ok: {len(pin)}-row golden pin holds")


def test_inv4_score_choices_synced_with_registry():
    """--sga-score choices are the registered score names (the registry
    is data; the declared choices must not drift from it)."""
    from sga.plugins.scores import score_names

    assert declared_choices("sga_score") == tuple(sorted(score_names()))
    print("ok: score choices synced with the registry")


# ---------------------------------------------------------------- INV-5
def test_inv5_topk_mode_choices_synced_with_kernel():
    """The method choices must stay sync'd with the impl classes topk
    dispatches to (a stale choice = a config-time lie): the declared
    choice set IS TopkSelection's method dispatch table, and every
    declared method resolves to its expected impl class."""
    from sga.plugins.selection._adaptive import AdaptiveTau
    from sga.plugins.selection._exact import ExactKthvalue
    from sga.plugins.selection._histogram import HistogramSelect
    from sga.plugins.selection.topk import TopkSelection

    # the method -> impl resolution TopkSelection.__init__ performs
    dispatch = {"adaptive": AdaptiveTau, "exact": ExactKthvalue,
                "histogram": HistogramSelect}
    methods = declared_choices("sga_selection_method")
    assert tuple(methods) == tuple(dispatch), (methods, tuple(dispatch))
    for method, impl_cls in dispatch.items():
        sel = TopkSelection({"sga_selection_method": method,
                             "sga_selection_ratio_percent": 10.0})
        assert type(sel._impl) is impl_cls, (method, type(sel._impl))
    print("ok: topk method choices synced with kernel impls")


def test_inv5_dead_config_refusal_names_own_vocabulary():
    """A param of another method is refused on BOTH layers, naming what
    the configured method does take; defaults expand from the
    declarations."""
    from sga.plugins.selection.topk import TopkSelection

    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    expect_error(
        lambda: validate_sga_flat({**base,
                                   "sga_selection_adaptive_warmup_mode": "exact"}),
        "do not belong to method 'histogram'")
    # constructor layer: same refusal, directly constructed
    expect_error(
        lambda: TopkSelection({"sga_selection_method": "exact",
                               "sga_selection_ratio_percent": 10.0,
                               "sga_selection_histogram_mode": "fast"}),
        "do not belong to method 'exact'")
    # constructor precedence: a missing or invalid method keeps the
    # shared validator's required/choices error even with a guarded
    # param present (the dead-config gate is method-gated)
    expect_error(
        lambda: TopkSelection({"sga_selection_ratio_percent": 10.0,
                               "sga_selection_histogram_mode": "fast"}),
        "param 'sga_selection_method' is required")
    expect_error(
        lambda: TopkSelection({"sga_selection_method": "hystogram",
                               "sga_selection_ratio_percent": 10.0,
                               "sga_selection_histogram_mode": "fast"}),
        "must be one of")
    # defaults expand from the declarations (single source)
    sel = TopkSelection({"sga_selection_method": "histogram",
                         "sga_selection_ratio_percent": 10.0})
    assert sel.params["sga_selection_histogram_mode"] == "exact"
    print("ok: dead-config refusal names the own vocabulary (both layers)")


def test_inv5_unlock_tables_in_sync():
    """Validation's dead-config refusal tracks the plugin's own UNLOCKS
    table for EVERY (method, param) pair (the table is duplicated on the
    two layers by design; a silent drift would move the yaml-route
    refusal to per-rank build time): under the owning method the param
    validates cleanly, under any other method it is refused."""
    from sga.plugins.selection.topk import UNLOCKS

    param_values = {"sga_selection_histogram_mode": "fast",
                    "sga_selection_adaptive_warmup_syncs": 3,
                    "sga_selection_adaptive_warmup_mode": "fast",
                    "sga_selection_block_size": 4,
                    "sga_selection_block_reduce": "max"}
    # params whose clean validation needs another knob off its default
    # (block_reduce is a dead knob under the block_size=1 default; any
    # block_size > 1 leg additionally needs a block-carrying comm
    # combination — tpi with explicit writeback — or the cross-field rule
    # fires first)
    param_extras = {"sga_selection_block_size":
                    {"sga_backend_emission": "writeback"},
                    "sga_selection_block_reduce":
                    {"sga_selection_block_size": 4,
                     "sga_backend_emission": "writeback"}}
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}
    for owner, params in UNLOCKS.items():
        for method in declared_choices("sga_selection_method"):
            for param in params:
                flat = {**base, "sga_selection_method": method,
                        param: param_values[param],
                        **param_extras.get(param, {})}
                if method == owner:
                    canon = validate_sga_flat(flat)
                    assert canon[param] == param_values[param]
                else:
                    expect_error(lambda f=flat: validate_sga_flat(f),
                                 "do not belong to method")
    print("ok: validation refusal tracks the plugin UNLOCKS table")


def test_inv5_block_size_rules():
    """Block-granularity value rules: power-of-two [1, 1024]
    range and the block_reduce dead-knob refusal under block_size 1."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index",
            "sga_backend_emission": "writeback"}  # the block-carrying comm
                                                  # combination (cross-field)
    canon = validate_sga_flat({**base, "sga_selection_block_size": 64,
                               "sga_selection_block_reduce": "max"})
    assert canon["sga_selection_block_size"] == 64
    assert canon["sga_selection_block_reduce"] == "max"
    for bad in (0, 3, 2048, -16):
        expect_error(lambda b=bad: validate_sga_flat(
            {**base, "sga_selection_block_size": b}), "power of two")
    # bool-as-int is refused one layer earlier, at type coercion
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_selection_block_size": True}), "valid int")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_selection_block_reduce": "max"}), "dead knob")
    print("ok: block_size power-of-two range + block_reduce dead-knob refusal")


def test_inv5_block_backend_cross_field_rules():
    """Block-granularity comm-path cross-field rules. The mask stays the
    canonical support form (correct on every backend), so these gates only
    refuse combinations whose evident intent — block communication — would
    silently degrade to element-level transport: nccl's fork kernel has no
    block-size argument, and tpi's compact consumers are
    element-granular. tpi+writeback and the mask backend carry blocks."""
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_selection_block_size": 16}
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_backend": "nccl_pipelined_index"}),
        "element-granular")
    # tpi with the auto default: auto resolves to compact here, and the
    # refusal must say so (the hint text is the user's only signpost)
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_backend": "torch_pipelined_index"}),
        "resolves to 'compact'")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_backend": "torch_pipelined_index",
         "sga_backend_emission": "compact"}), "'writeback'")
    # allowed: tpi+writeback and the mask backend (canonical round-trip)
    canon = validate_sga_flat(
        {**base, "sga_backend": "torch_pipelined_index",
         "sga_backend_emission": "writeback"})
    assert canon["sga_selection_block_size"] == 16
    assert canon["sga_backend_emission"] == "writeback"
    canon = validate_sga_flat(
        {**base, "sga_backend": "torch_pipelined_mask"})
    assert canon["sga_selection_block_size"] == 16
    print("ok: block comm-path cross-field refusals (nccl, tpi non-writeback)")


def test_inv5_topk_unknown_method_keeps_validator_error():
    """A missing or invalid method surfaces as the required/choices error,
    never as a dead-config refusal naming the raw value (0024 C1)."""
    expect_error(
        lambda: validate_sga_flat({
            "sga_enabled": True, "sga_score": "update",
            "sga_selection_ratio_percent": 10.0,
            "sga_selection_histogram_mode": "fast",
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}),
        "sga_selection_method is required")
    expect_error(
        lambda: validate_sga_flat({
            "sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "hystogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_selection_histogram_mode": "fast",
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}),
        "must be one of ['adaptive', 'exact', 'histogram']")
    print("ok: unknown method keeps the validator error")


def test_inv5_plugin_vocabularies_are_plugin_owned():
    """Each plugin's constructor enforces its OWN vocabulary: foreign
    params (declared flags of another plugin) and missing required ones
    fail loud at build time; on the config route validation rejects a
    foreign param first, naming the plugin that owns it (any
    explicitly-set key must belong to the selected plugin of its axis,
    whatever its value)."""
    from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
    from sga.plugins.selection.topk import TopkSelection

    expect_error(lambda: TorchPipelinedIndexBackend({"sga_backend_mask_slices": 3}),
                 "unknown params ['sga_backend_mask_slices']")
    expect_error(lambda: NcclPipelinedIndexBackend({"sga_backend_index_slices": 3}),
                 "unknown params ['sga_backend_index_slices']")
    expect_error(lambda: TopkSelection({"sga_selection_ratio_percent": 10.0}),
                 "param 'sga_selection_method' is required")
    # validation owns name attribution on the config route now: a
    # foreign-but-declared param is refused there too, naming the plugin
    # that owns it (regardless of value — the at-default spelling is
    # refused the same way, before the engine builder's default-diff
    # could absorb it)
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "exact", "sga_selection_ratio_percent": 5.0,
            "sga_optimizer": "adamw", "sga_backend": "torch_pipelined_index",
            "sga_backend_mask_slices": 3}
    expect_error(lambda: validate_sga_flat(base), "torch_pipelined_mask")
    print("ok: plugin vocabularies are plugin-owned (validation + build)")


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
    # the flat validator prefixes every parser refusal with the key name
    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "exact",
            "sga_selection_ratio_percent": 5.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index",
            "sga_always_sync_layers": "a,,b"}
    expect_error(lambda: validate_sga_flat(base),
                 "sga_always_sync_layers: patterns must be non-empty")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_always_sync_layers": "null"}),
        "sga_always_sync_layers: an explicit null is NOT the unset spelling")
    print("ok: always-sync pattern parser edge table holds")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL FLAT-CONFIG INVARIANT TESTS PASSED")


def test_inv2_emission_switch_flat_rules():
    """emission: 'auto' is the declared default (per-backend resolution is
    the plugin constructors' business, via BackendDecl.resolve_emission —
    the single home of the auto rule); explicit 'compact' is a
    torch_pipelined_index capability only; explicit values survive the
    canonical round-trip and the plugin-construction slicing; the retired
    merge_join_dir dest fails closed on the flat/yaml route (unknown
    key) and the resume route refuses retired keys outright
    (--sga-allow-config-override is the escape hatch; LEG-1 pins the
    kernel end states bitwise-identical)."""
    from sga.plugins import build_backend_plugin

    base = {"sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}

    def _built_emission(flat):
        canon = validate_sga_flat(flat)
        return build_backend_plugin(
            canon["sga_backend"], explicit_from_flat(canon)).emission

    # auto default: canonical carries it; nothing explicit flows to the
    # constructor, which resolves auto -> the backend's inherent form
    canon = validate_sga_flat(dict(base))
    assert canon["sga_backend_emission"] == "auto"
    assert _built_emission(dict(base)) == "compact"
    # explicit writeback on tpi: survives canonical AND flows through the
    # explicit-diff slicing into the constructor
    canon = validate_sga_flat({**base, "sga_backend_emission": "writeback"})
    assert canon["sga_backend_emission"] == "writeback"
    assert _built_emission(
        {**base, "sga_backend_emission": "writeback"}) == "writeback"
    # explicit compact on tpi itself: accepted
    assert validate_sga_flat({**base, "sga_backend_emission": "compact"})[
        "sga_backend_emission"] == "compact"
    # wrong-typed / bogus values refused by the str-kind coercion
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_backend_emission": None}), "must be a string")
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_backend_emission": "quadtree"}), "must be one of")
    # compact is tpi-only: refused flat-side on the writeback-inherent
    # backends (the launcher catches it before any process is spawned)
    for wb in ("torch_pipelined_mask", "nccl_pipelined_index"):
        expect_error(
            lambda wb=wb: validate_sga_flat(
                {**base, "sga_backend": wb,
                 "sga_backend_emission": "compact"}),
            "only supported by torch_pipelined_index")
    # the retired rename fails closed on both routes: unknown key flat,
    # and the resume guard refuses checkpoints carrying retired keys
    # (--sga-allow-config-override is the escape hatch)
    expect_error(lambda: validate_sga_flat(
        {**base, "sga_optimizer_merge_join_dir": "table"}), "unknown keys")


# ---------------------------------------------------------------- INV-6
def test_inv6_every_declared_dest_has_a_consumer():
    """Completeness: with prefix routing gone, a declared dest no consumer
    claims would be silently DROPPED on the way to the engine/plugins.
    Every declared sga_* dest must land in exactly one consumer class:
    the engine's read set (pinned literally), an axis-identity key
    (consumed by build_engine), a plugin's PARAMS vocabulary (imported
    from the plugin modules — the vocabularies stay plugin-owned), or
    the explicit exemptions (the master switch consumed by validation
    itself; the percent spelling resolved away at validation)."""
    from sga.backends.nccl_pipelined_index import PARAMS as NPI_PARAMS
    from sga.backends.torch_pipelined_index import PARAMS as TPI_PARAMS
    from sga.backends.torch_pipelined_mask import PARAMS as TPM_PARAMS
    from sga.plugins.optimizer.adamw import PARAMS as ADAMW_PARAMS
    from sga.plugins.selection.topk import PARAMS as TOPK_PARAMS

    engine_keys = {
        "sga_cycle_start_iter", "sga_cycle_period",
        "sga_cycle_decouple_full_sync_interval_and_period",
        "sga_cycle_full_sync_interval",
        "sga_checks_count_guard",
        "sga_always_sync_layers",
        "sga_trace_dir", "sga_trace_branch_points", "sga_trace_window",
        "sga_shadow", "sga_shadow_score",
    }
    axis_keys = {"sga_score", "sga_selection", "sga_optimizer", "sga_backend"}
    plugin_vocab = set().union(
        set(TOPK_PARAMS), set(ADAMW_PARAMS), set(TPI_PARAMS),
        set(TPM_PARAMS), set(NPI_PARAMS))
    exempt = {"sga_enabled", "sga_cycle_start_iter_percent"}

    # pairwise disjointness across ALL consumer groups: a dest
    # double-claimed across axes would be routed to BOTH by _axis_params'
    # independent per-axis slices. The backend vocabularies merge into
    # one group — within-union sharing of sga_backend_emission is
    # intended (the union slice relies on it), so the union absorbs it.
    groups = {"engine": engine_keys, "axis": axis_keys, "exempt": exempt,
              "selection": set(TOPK_PARAMS), "optimizer": set(ADAMW_PARAMS),
              "backend": set(TPI_PARAMS) | set(TPM_PARAMS) | set(NPI_PARAMS)}
    names = sorted(groups)
    for i, ga in enumerate(names):
        for gb in names[i + 1:]:
            overlap = groups[ga] & groups[gb]
            assert not overlap, f"double-claimed dests {ga}/{gb}: {overlap}"
    covered = set().union(*groups.values())
    declared = set(sga_defaults())
    assert not declared - covered, f"unclaimed dests: {declared - covered}"
    assert not covered - declared, f"stale consumer pins: {covered - declared}"
    # the validation-side ownership map stays tied to the modules' PARAMS
    assert set(V._plugin_param_owners()) == plugin_vocab
    print("ok: every declared dest has exactly one consumer class")


if __name__ == "__main__":
    _run_all()
