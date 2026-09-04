#!/usr/bin/env python3
"""The catalog invariants — sga/config/arguments.py as the single source.

Five structural pins , each turning a discipline rule into a
testable property of the CODE (not of anyone's good intentions):

  INV-1  bidirectional no-orphans  registered plugins <-> catalog rows
         (and the registration-time join fires loud without one)
  INV-2  completeness              every accepted config key and every
         flat flag traces back to a catalog row, and every flaggable
         catalog row has its flag
  INV-3  catalog purity            arguments.py is a pure data table: no
         sga imports, no functions, Param is the only class (AST-level)
  INV-4  documentation             every Param carries a non-empty doc,
         an explicit scalar type, required/default sanity
  INV-5  guard integrity           every equality guard resolves inside
         the catalog (sibling exists, value in sibling's choices)

Plus corner cases: the historical row metadata (default/required/nullable/
choices/yaml_only) is pinned to the pre-inversion vocabulary (a metadata
edit is a semantic change someone must own), the topk hist-mode choices
stay in sync with the kernel's mode table, the per-method foreign-key
refusal still names the method's own vocabulary, and a missing/unknown
topk method keeps the validator's required/choices error.

Run: python tests/test_config_catalog.py
"""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.config.arguments import (
    ENGINE_SECTIONS,
    PLUGIN_PARAMS,
    SCORE_NAMES,
    TOP,
)
from sga.config.flatargs import flat_flags
from sga.plugins import (
    SelectionDecl,
    guard_foreign_keys,
    plugin_params,
    register_default_plugins,
    registered_names,
)
from sga.plugins.selection._histogram import _HIST_MODES

CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sga", "config", "arguments.py")


def _all_params():
    """(where, Param) for every catalog row — engine top (""), engine
    sections ("cycle"/...), plugin axes ("selection/topk"/...)."""
    for p in TOP:
        yield ("", p)
    for section, decl in ENGINE_SECTIONS.items():
        for p in decl:
            yield (section, p)
    for axis, plugins in PLUGIN_PARAMS.items():
        for pname, decl in plugins.items():
            for p in decl:
                yield (f"{axis}/{pname}", p)


def expect_error(fn, needle):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert needle in str(e), f"error missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


# ---------------------------------------------------------------- INV-1

def test_inv1_no_orphans_either_direction():
    """Every registered plugin has a catalog row and vice versa; the score
    name list mirrors the score registry."""
    register_default_plugins()
    names = registered_names()
    for axis in ("selection", "optimizer", "backend"):
        assert set(names[axis]) == set(PLUGIN_PARAMS[axis]), axis
    assert list(SCORE_NAMES) == sorted(SCORE_NAMES)
    from sga.plugins import score_names

    assert list(SCORE_NAMES) == score_names()
    print("ok: INV-1 catalog rows and registry are orphan-free both ways")


def test_inv1_registration_join_fires_loud():
    """A plugin registered without a catalog row is refused AT REGISTRATION
    (import time), not at first config read."""
    from sga.plugins import _register

    registry = {}
    expect_error(
        lambda: _register("selection", registry, "ghost",
                          SelectionDecl(name="ghost")),
        "no param row",
    )
    assert registry == {}
    print("ok: INV-1 registration-time join refuses a catalog-less plugin")


def test_inv1_decls_structurally_carry_no_params():
    """The inversion is structural: DECLs have NO params field to fill —
    an inline declaration cannot even be written."""
    for decl_cls, kw in ((SelectionDecl, {"output_form": "mask"}),):
        expect_error(lambda decl_cls=decl_cls, kw=kw:
                     decl_cls(name="x", params=(), **kw),
                     "unexpected keyword argument")
    from sga.plugins import BackendDecl, OptimizerDecl

    expect_error(lambda: BackendDecl(name="x", params=()),
                 "unexpected keyword argument")
    expect_error(lambda: OptimizerDecl(name="x", params=()),
                 "unexpected keyword argument")
    print("ok: INV-1 DECL dataclasses have no params field at all")


# ---------------------------------------------------------------- INV-2

def test_inv2_flags_exactly_derive_from_catalog():
    """The flag table is EXACTLY the catalog's flaggable surface, in BOTH
    directions: an independent reconstruction of the vocabulary from the
    catalog rows set-equals the generated list — a row without a flag, a
    flag without a row, or a type/choices/nullable drift all fail (help
    rides verbatim from the same row's doc and is pinned nowhere).
    Duplicate flag names or dests are checked apart: the dict-keyed
    consumers would silently drop the colliding entries."""
    actual = flat_flags()
    assert len({f.flag for f in actual}) == len(actual), "duplicate flag name"
    assert len({f.dest for f in actual}) == len(actual), "duplicate dest"

    expected = set()

    def want(flag, section, key, typ, p=None, choices=()):
        expected.add((flag, flag[2:].replace("-", "_"), section, key, typ,
                      tuple(choices),
                      p.nullable if p is not None else False))

    for p in TOP:
        want(f"--sga-{p.name}".replace("_", "-"), "", p.name, p.type, p,
             p.choices or ())
    for section, decl in ENGINE_SECTIONS.items():
        for p in decl:
            if p.yaml_only:
                continue  # a yaml-only row must have no flag (set equality)
            want(f"--sga-{section}-{p.name}".replace("_", "-"),
                 section, p.name, p.type, p, p.choices or ())
    for axis in ("score", "selection", "optimizer", "backend"):
        names = SCORE_NAMES if axis == "score" else sorted(PLUGIN_PARAMS[axis])
        want(f"--sga-{axis}-name", axis, "name", str, choices=names)
        if axis == "score":
            continue
        first = {}  # same-name params across one axis: the first row wins
        for decl in PLUGIN_PARAMS[axis].values():
            for p in decl:
                first.setdefault(p.name, p)
        for name, p in sorted(first.items()):
            want(f"--sga-{axis}-{name}".replace("_", "-"), axis, name,
                 p.type, p, p.choices or ())

    got = {(f.flag, f.dest, f.section, f.key, f.type, f.choices, f.nullable)
           for f in actual}
    assert got == expected, \
        f"flag table vs catalog disagree: {sorted(got ^ expected)}"
    print("ok: INV-2 flag table is exactly the catalog's flaggable surface")


def test_inv2_accepted_keys_trace_to_catalog():
    """A full-surface config resolves through the loader with EXACTLY the
    catalog's keys — the loader cannot accept or invent anything else."""
    from sga.config.loader import engine_config_from_sga_dict

    sga = {
        "enabled": True,
        "shadow": True, "shadow_score": False,
        "cycle": {"start_iter": 4, "period": 8,
                  "decouple_full_sync_interval_and_period": True,
                  "full_sync_interval": 4},
        "checks": {"count_guard": False},
        "trace": {"dir": "/tmp/inv2", "branch_points": [4, 8], "window": 3},
        "score": {"name": "update"},
        "selection": {"name": "topk", "method": "adaptive",
                      "ratio_percent": 20.0,
                      "adaptive_warmup_syncs": None,
                      "adaptive_warmup_mode": "balanced"},
        "optimizer": {"name": "adamw", "residual_mean": True,
                      "merge_join_dir": "binary"},
        "backend": {"name": "torch_pipelined_index",
                    "high_priority_group": True, "slices": 3},
    }
    cfg = engine_config_from_sga_dict(sga)
    # the engine config carries no "enabled" key (it exists only when on);
    # the score axis is registry data and accepted without plugin rows
    assert set(cfg) == ({p.name for p in TOP if p.name != "enabled"}
                        | set(ENGINE_SECTIONS) | set(PLUGIN_PARAMS)
                        | {"score"})
    for section in ENGINE_SECTIONS:
        assert set(cfg[section]) <= {p.name for p in ENGINE_SECTIONS[section]
                                     if not p.yaml_only}, section
    # a plugin axis carries only its explicit keys; the catalog row is the
    # UNION across method variants (guards) — every accepted key must trace
    # to a row, never to anything else
    for axis in PLUGIN_PARAMS:
        name = cfg[axis].pop("name")
        catalog_names = {p.name for p in PLUGIN_PARAMS[axis][name]}
        assert set(cfg[axis]) <= catalog_names, axis
    print("ok: INV-2 accepted config keys are exactly the catalog's")


# ---------------------------------------------------------------- INV-3

def test_inv3_catalog_is_pure_data():
    """AST-level purity: no sga imports, no functions, Param is the only
    class, Param constructor calls are the only calls, module statements
    are data assignments only. A helper creeping into the catalog fails
    this before it fails a reader."""
    tree = ast.parse(open(CATALOG_PATH).read())
    for stmt in tree.body:
        assert isinstance(stmt, (ast.Expr, ast.ImportFrom, ast.ClassDef,
                                 ast.Assign, ast.AnnAssign)), \
            f"non-data statement at module level: {type(stmt).__name__}"
        if isinstance(stmt, ast.ImportFrom):
            assert stmt.module == "dataclasses", \
                f"only dataclasses may be imported, got {stmt.module}"
        if isinstance(stmt, ast.ClassDef):
            assert stmt.name == "Param", f"unexpected class {stmt.name}"
            for sub in stmt.body:  # fields (and the docstring) only
                assert isinstance(sub, (ast.Expr, ast.AnnAssign)), \
                    "Param grew a body member"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None)
            assert mod in (None, "dataclasses"), f"import of {mod}"
        assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.Lambda)), \
            f"function definition in catalog: {getattr(node, 'name', '?')}"
        if isinstance(node, ast.Call):
            assert (isinstance(node.func, ast.Name)
                    and node.func.id in ("Param", "dataclass")), \
                "only Param(...) rows (and the dataclass decorator) live " \
                "in the catalog"
    print("ok: INV-3 catalog is a pure data table (AST-checked)")


# ---------------------------------------------------------------- INV-4

def test_inv4_every_param_documented_and_typed():
    for where, p in _all_params():
        assert p.doc and p.doc.strip(), f"empty doc: {where}/{p.name}"
        assert p.type in (bool, int, float, str, list), \
            f"non-scalar type {p.type!r}: {where}/{p.name}"
        if p.required:
            assert p.default is None, f"required param has default: {p.name}"
            assert not p.nullable, f"required+nullable is contradictory: {p.name}"
        if p.type is str and p.choices:
            assert p.default is None or p.default in p.choices, \
                f"default not in choices: {where}/{p.name}"
    print("ok: INV-4 every catalog row has doc/type/required-default sanity")


def test_inv4_historical_metadata_pinned():
    """Every row's metadata is IDENTICAL to the pre-inversion declarations
    (engine.py/trace.py/loader literals + the old per-plugin PluginParam
    rows): (default, required, nullable, choices, yaml_only). Changing any
    of them is a semantic change: do it consciously and update this pin.
    nullable pins the flag vocabulary too — a list-typed flag can never
    spell null, so branch_points must stay non-nullable."""
    pin = {  # (where, name) -> (default, required, nullable, choices, yaml_only)
        ("", "enabled"): (None, True, False, (), False),
        ("", "shadow"): (False, False, False, (), False),
        ("", "shadow_score"): (True, False, False, (), False),
        ("cycle", "start_iter"): (0, False, False, (), False),
        ("cycle", "start_iter_percent"): (None, False, False, (), True),
        ("cycle", "period"): (1, False, False, (), False),
        ("cycle", "decouple_full_sync_interval_and_period"):
            (False, False, False, (), False),
        ("cycle", "full_sync_interval"): (None, False, False, (), False),
        ("checks", "count_guard"): (True, False, False, (), False),
        ("trace", "dir"): (None, True, False, (), False),
        ("trace", "branch_points"): (None, False, False, (), False),
        ("trace", "window"): (100, False, False, (), False),
        ("selection/topk", "method"):
            (None, True, False, ("adaptive", "exact", "histogram"), False),
        ("selection/topk", "ratio_percent"): (None, True, False, (), False),
        ("selection/topk", "histogram_mode"):
            ("exact", False, False, ("balanced", "exact", "fast"), False),
        ("selection/topk", "adaptive_warmup_syncs"):
            (5, False, True, (), False),
        ("selection/topk", "adaptive_warmup_mode"):
            ("fast", False, False, ("balanced", "exact", "fast"), False),
        ("optimizer/adamw", "residual_mean"): (False, False, False, (), False),
        ("optimizer/adamw", "merge_join_dir"):
            ("table", False, False, ("table", "binary"), False),
        ("backend/torch_pipelined_index", "high_priority_group"):
            (False, False, False, (), False),
        ("backend/torch_pipelined_index", "slices"):
            (2, False, False, (), False),
        ("backend/torch_pipelined_mask", "num_slices"):
            (0, False, False, (), False),
    }
    actual = {(where, p.name): (p.default, p.required, p.nullable,
                                p.choices, p.yaml_only)
              for where, p in _all_params()}
    for key, want in pin.items():
        assert key in actual, f"pinned key vanished from catalog: {key}"
        assert actual[key] == want, \
            f"{key}: metadata changed {actual[key]!r} != pinned {want!r}"
    print("ok: INV-4 catalog row metadata pinned to history")


# ---------------------------------------------------------------- INV-5

def test_inv5_guards_resolve_inside_catalog():
    """Every equality guard is well-formed: names a sibling param of the
    same plugin, its value is one of the sibling's choices, the sibling is
    itself unguarded (no chains), and each guarded param is unlocked by at
    least one sibling choice."""
    for axis, plugins in PLUGIN_PARAMS.items():
        for pname, decl in plugins.items():
            by_name = {p.name: p for p in decl}
            for p in decl:
                if not p.requires:
                    continue
                assert len(p.requires) == 2, (axis, pname, p.name)
                sib_name, sib_val = p.requires
                sib = by_name.get(sib_name)
                assert sib is not None, \
                    f"guard sibling missing: {axis}/{pname}.{p.name} -> {sib_name}"
                assert not sib.requires, \
                    f"chained guard: {axis}/{pname}.{sib_name}"
                assert sib.choices and sib_val in sib.choices, \
                    f"guard value {sib_val!r} not in {sib_name}'s choices"
    print("ok: INV-5 every guard resolves within its plugin's rows")


def test_inv5_topk_mode_choices_synced_with_kernel():
    """The catalog hardcodes the hist-mode tiers; the kernel's mode table
    is the behavioral truth — drift is caught here, not at 3am on the
    cluster."""
    want = tuple(sorted(_HIST_MODES))
    for p in plugin_params("selection", "topk"):
        if p.name in ("histogram_mode", "adaptive_warmup_mode"):
            assert p.choices == want, (p.name, p.choices, want)
    print("ok: INV-5 topk mode choices synced with _HIST_MODES")


def test_inv5_foreign_key_refusal_names_own_vocabulary():
    """The per-method refusal still lists the method's OWN legal extra
    params (and says 'no extra params' for exact) — a user fixing a
    wrong-method knob must be told what the right ones are."""
    foreign, own = guard_foreign_keys(
        "selection", "topk",
        {"method": "histogram", "ratio_percent": 10.0,
         "adaptive_warmup_syncs": 3, "adaptive_warmup_mode": "fast"})
    assert foreign == ["adaptive_warmup_mode", "adaptive_warmup_syncs"], foreign
    assert own == ["histogram_mode"], own
    foreign, own = guard_foreign_keys(
        "selection", "topk",
        {"method": "exact", "ratio_percent": 10.0, "histogram_mode": "fast"})
    assert foreign == ["histogram_mode"], foreign
    assert own == [], own
    # method absent entirely: every guarded key is foreign
    foreign, own = guard_foreign_keys(
        "selection", "topk", {"ratio_percent": 10.0, "histogram_mode": "fast"})
    assert foreign == ["histogram_mode"], foreign
    print("ok: INV-5 foreign-key refusal names the method's own vocabulary")


def test_inv5_topk_unknown_method_keeps_validator_error():
    """C1 regression: the constructor's foreign-key refusal is gated on a
    KNOWN method — a missing or invalid method must surface as the shared
    validator's required/choices error, never as a foreign-key refusal
    naming the raw value (the restored pre-refactor behavior; reachable on
    direct TopkSelection calls with unvalidated dicts)."""
    from sga.plugins.selection.topk import TopkSelection

    expect_error(
        lambda: TopkSelection({"ratio_percent": 10.0,
                               "histogram_mode": "fast"}),
        "param 'method' is required",
    )
    expect_error(
        lambda: TopkSelection({"method": "hist", "ratio_percent": 10.0,
                               "histogram_mode": "fast"}),
        "param 'method' must be one of ['adaptive', 'exact', 'histogram'] "
        "(got 'hist')",
    )
    print("ok: INV-5 missing/invalid method keeps the validator's error")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
