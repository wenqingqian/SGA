#!/usr/bin/env python3
"""The flat-config invariants (run: python tests/test_config_flat.py).

The declarations file (sga/config/arguments.py) is the single source of
the config surface; these tests make that STRUCTURAL, not disciplinary:

  INV-1 no orphans, either direction: every registered plugin has
       attribution rows in validation._AXIS_PARAMS and vice versa; the
       fork-side waist registers exactly one non-config sga_ key (its
       override switch) and no --sga-config remains anywhere.
  INV-2 flags are exactly the declarations: the schema dump, the
       launcher's key check and the training entry's parser all derive
       from sga_actions(); every sga_<axis>_* dest carries attribution
       (no dead plugin flag); engine-side default reads equal
       engine_section_defaults() for every section.
  INV-3 the declarations stay megatron-shaped: arguments.py contains
       only imports, the two type callables, add_argument_group and
       add_argument calls (AST-pinned — no validation logic grows in
       the file a developer reads for the knob list).
  INV-4 every knob is documented: help non-empty, kind known, default
       explicit; the historical surface is pinned as a golden table.
  INV-5 guards resolve: every attribution guard names a same-axis param
       whose declared choices contain the guarded value; topk's mode
       choices stay synced with the kernel impls; the foreign-key and
       unknown-method error paths are pinned byte-wise.
"""

import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.config import validation as V
from sga.config.arguments import add_sga_args, sga_bool, sga_int_or_null
from sga.config.validation import (
    _AXIS_PARAMS,
    _param_kind,
    axis_param_defaults,
    declared_choices,
    engine_section_defaults,
    expand_plugin_params,
    guard_foreign_params,
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
def test_inv1_attribution_matches_registries():
    """Registered plugin names == attribution keys, per axis, both ways."""
    from sga.plugins import register_default_plugins, registered_names

    register_default_plugins()
    names = registered_names()
    for axis in ("selection", "optimizer", "backend"):
        assert sorted(_AXIS_PARAMS[axis]) == names[axis], axis
    # the score axis is registry data with NO params: no dests may exist
    assert not [d for d in sga_actions() if d.startswith("sga_score_")]
    print("ok: attribution <-> registries, both directions")


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
def test_inv2_schema_and_launcher_derive_from_declarations():
    """The schema dump and the launcher's accepted keys ARE the parser's
    dests — there is no second key table anywhere in the pipeline."""
    import json

    schema = json.loads(json.dumps({}))  # import shim
    from sga.config.__main__ import _schema

    schema = _schema()
    assert set(schema["flags"]) == set(sga_actions())
    # the launcher validates yaml keys against sga_actions via
    # validate_run_config: an unknown key is rejected with the dest list
    expect_error(lambda: validate_sga_flat({"sga_nope": 1}), "sga_nope")
    print("ok: schema/launcher/declaration vocabularies coincide")


def test_inv2_every_plugin_dest_is_attributed():
    """Every sga_<axis>_* dest belongs to some plugin's attribution rows
    (a flag nobody attributes is a dead knob); and every attribution
    suffix has a declared dest (a row without a flag is undeclarable)."""
    for axis in ("selection", "optimizer", "backend"):
        prefix = f"sga_{axis}_"
        dest_suffixes = {d[len(prefix):] for d in sga_actions()
                         if d.startswith(prefix) and d != prefix[:-1]}
        row_suffixes = {s for rows in _AXIS_PARAMS[axis].values()
                        for s, _ in rows}
        assert dest_suffixes == row_suffixes, axis
    print("ok: plugin dests <-> attribution rows, both directions")


def test_inv2_engine_defaults_equal_declarations():
    """The engine-side .get() fallback dicts ARE the declared defaults for
    every engine section (single source, no restated numbers)."""
    import sga.engine as engine_mod
    import sga.trace as trace_mod

    assert engine_mod._CYCLE_DEFAULTS == {
        "start_iter": 0, "start_iter_percent": None, "period": 1,
        "decouple_full_sync_interval_and_period": False,
        "full_sync_interval": None,
    }
    assert engine_mod._CYCLE_DEFAULTS == engine_section_defaults("cycle")
    assert engine_mod._CHECKS_DEFAULTS == engine_section_defaults("checks")
    assert engine_mod._TOP_DEFAULTS == {
        "shadow": sga_defaults()["sga_shadow"],
        "shadow_score": sga_defaults()["sga_shadow_score"],
    }
    assert trace_mod._TRACE_DEFAULTS == engine_section_defaults("trace")
    print("ok: engine fallback defaults derive from the declarations")


def test_inv2_validation_materializes_every_section_key():
    """A minimal enabled config materializes complete engine sections:
    every engine-section dest appears in the canonical dict and the
    engine config sections carry exactly the declared keys."""
    eng, canon = validate_sga_flat({
        "sga_enabled": True, "sga_score": "update",
        "sga_selection_method": "exact", "sga_selection_ratio_percent": 5.0,
        "sga_optimizer": "adamw", "sga_backend": "torch_pipelined_index"})
    for section in ("cycle", "checks", "trace"):
        declared = engine_section_defaults(section)
        prefix = f"sga_{section}_"
        for key in declared:
            if prefix + key == "sga_cycle_start_iter_percent":
                continue  # resolved away by design (never in canonical)
            assert canon[prefix + key] is not None or declared[key] is None, key
    assert set(eng["cycle"]) == {"start_iter", "period",
                                 "decouple_full_sync_interval_and_period",
                                 "full_sync_interval"}
    assert set(eng["checks"]) == {"count_guard"}
    print("ok: validation materializes every declared section key")


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
            "sga_optimizer", "sga_backend", "sga_trace_dir",
            "sga_trace_branch_points", "sga_cycle_start_iter_percent",
            "sga_cycle_full_sync_interval"), dest
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
        "sga_trace_dir": (None, "str", ()),
        "sga_trace_branch_points": (None, "str", ()),
        "sga_trace_window": (100, "int", ()),
        "sga_score": (None, "str", ()),
        "sga_selection": ("topk", "str", ()),
        "sga_selection_method": (None, "str", ("adaptive", "exact", "histogram")),
        "sga_selection_ratio_percent": (None, "float", ()),
        "sga_selection_histogram_mode": ("exact", "str", ("balanced", "exact", "fast")),
        "sga_selection_adaptive_warmup_syncs": (5, "int_or_null", ()),
        "sga_selection_adaptive_warmup_mode": ("fast", "str", ("balanced", "exact", "fast")),
        "sga_optimizer": (None, "str", ()),
        "sga_optimizer_residual_mean": (False, "bool", ()),
        "sga_optimizer_merge_join_dir": ("table", "str", ("table", "binary")),
        "sga_backend": (None, "str", ()),
        "sga_backend_high_priority_group": (False, "bool", ()),
        "sga_backend_slices": (2, "int", ()),
        "sga_backend_num_slices": (0, "int", ()),
    }
    actions = sga_actions()
    assert set(pin) == set(actions), set(pin) ^ set(actions)
    for dest, (default, kind, choices) in pin.items():
        assert actions[dest].default == default, dest
        assert _param_kind(dest) == kind, dest
        assert tuple(getattr(actions[dest], "choices", None) or ()) == choices, dest
    print("ok: 26-row golden pin holds")


# ---------------------------------------------------------------- INV-5
def test_inv5_guards_resolve_inside_the_axis():
    for axis, plugins in _AXIS_PARAMS.items():
        all_rows = {s for rows in plugins.values() for s, _ in rows}
        for name, rows in plugins.items():
            for suffix, requires in rows:
                assert suffix in all_rows
                if requires is None:
                    continue
                other, value = requires
                dest = f"sga_{axis}_{other}"
                assert other in all_rows, (axis, other)
                assert value in declared_choices(dest), (axis, other, value)
    print("ok: every guard resolves to a sibling choice")


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
        sel = TopkSelection({"method": method, "ratio_percent": 10.0})
        assert type(sel._impl) is impl_cls, (method, type(sel._impl))
    print("ok: topk method choices synced with kernel impls")


def test_inv5_foreign_key_refusal_names_own_vocabulary():
    foreign, own = guard_foreign_params(
        "selection", "topk", {"method": "histogram",
                              "adaptive_warmup_mode": "exact"})
    assert foreign == ["adaptive_warmup_mode"]
    assert own == ["histogram_mode"]
    params = expand_plugin_params("selection", "topk",
                                  {"method": "histogram",
                                   "ratio_percent": 10.0})
    assert params["histogram_mode"] == "exact"  # default expanded
    expect_error(
        lambda: validate_sga_flat({
            "sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "histogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_selection_adaptive_warmup_mode": "exact",
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}),
        "do not belong to method 'histogram'")
    print("ok: foreign-key refusal names the own vocabulary")


def test_inv5_topk_unknown_method_keeps_validator_error():
    """A missing or invalid method surfaces as the required/choices error,
    never as a foreign-key refusal naming the raw value (0024 C1)."""
    expect_error(
        lambda: validate_sga_flat({
            "sga_enabled": True, "sga_score": "update",
            "sga_selection_ratio_percent": 10.0,
            "sga_selection_histogram_mode": "fast",
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}),
        "param 'method' is required")
    expect_error(
        lambda: validate_sga_flat({
            "sga_enabled": True, "sga_score": "update",
            "sga_selection_method": "hystogram",
            "sga_selection_ratio_percent": 10.0,
            "sga_optimizer": "adamw",
            "sga_backend": "torch_pipelined_index"}),
        "must be one of ['adaptive', 'exact', 'histogram']")
    print("ok: unknown method keeps the validator error")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL FLAT-CONFIG INVARIANT TESTS PASSED")


if __name__ == "__main__":
    _run_all()
