# Copyright (c) 2026, SGA framework contributors.
"""THE sga parameter catalog — the single declaration source.

This file is a PURE DATA TABLE: every sga config item — the engine sections
(top level / cycle / checks / trace) AND every plugin axis' params — is
declared HERE and nowhere else, with its type, default, choices and doc.
Read it top to bottom and you have seen the entire controllable surface of
SGA. It is the megatron ``arguments.py`` analog: the place you open to
answer "what can I configure?".

Everything else DERIVES from this table; nothing may restate a declaration:

  - the ``--sga-*`` flat-flag vocabulary, the nested-yaml -> flag
    translation and the flag -> canonical-JSON assembly
    (``sga/config/flatargs.py``)
  - the loader's key allowlists and the plugin-param checks
    (``sga/config/loader.py``, ``sga/plugins/__init__.py``)
  - engine-side ``.get(key, default)`` fallbacks (engine.py / trace.py /
    runtime.py) — the catalog default is the only default
  - the schema/resolve inspection CLI (``python -m sga.config``)

Adding a plugin or a param means adding a row here (the registry join in
``sga/plugins`` fails loud at registration when either side is missing).
The table must stay mechanically pure — no imports of sga code, no
functions, data assignments only — pinned by tests/test_config_catalog.py.

Guard metadata (``requires``) carries same-plugin conditional legality: a
param with ``requires=("method", "histogram")`` is only legal when the
sibling param ``method`` equals ``histogram`` (topk's per-method
vocabularies). Equality guards only — cross-axis conditions and numeric
relations stay in the code validators (``validate_combination``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    """One config-item declaration (the catalog's only type).

    ``type`` is the yaml-side scalar type (bool/int/float/str; ``list``
    rides as one comma-joined flat-flag token); ``choices`` (non-empty)
    restricts str params; ``nullable`` admits an explicit null (flat-flag
    spelling ``null``); required params carry no default; ``requires`` is
    an equality guard (sibling name, sibling value) under which alone the
    param is legal; ``yaml_only`` params get no flat flag (they need
    context only the launcher-side translation has).

    Input-semantics convention: ``required`` means required in the
    canonical enabled-form engine config (the loader may still accept the
    key absent — such a row's doc says so); ``default=None`` on an
    optional row marks an ABSENCE sentinel (absent == unset), not a null
    default, unless the row's doc says the null default is meaningful
    (branch_points).
    """

    name: str
    type: object
    required: bool = False
    default: object = None
    doc: str = ""
    choices: tuple = ()
    nullable: bool = False
    requires: tuple = ()
    yaml_only: bool = False


# ====================================================================
# engine top level (sga.enabled / sga.shadow / ...)
# ====================================================================

TOP = (
    Param(
        "enabled", bool, required=True,
        doc="master switch: true turns on the SGA engine; false (or a run "
            "with no sga config at all) is dense training, byte-identical "
            "to stock megatron — then the section may only name "
            "backend: dense. The loader accepts an absent enabled and "
            "treats it as false; required refers to the enabled form",
    ),
    Param(
        "shadow", bool, default=False,
        doc="shadow mode: replicate the stock optimizer step literally and "
            "only READ state at cycle boundaries (dense numerics "
            "invariant); the score is still computed read-only at "
            "boundaries for offline analysis",
    ),
    Param(
        "shadow_score", bool, default=True,
        doc="false strips even the read-only boundary scoring from a "
            "shadow run (the pure stock-replication probe); only "
            "meaningful with shadow: true",
    ),
)

# ====================================================================
# engine cycle (sga.cycle.*) — the reselection calendar
# ====================================================================

CYCLE = (
    Param(
        "start_iter", int, default=0,
        doc="engine step the cycle calendar anchors at; the first "
            "reselection happens at start_iter + period (before that the "
            "run is dense warmup)",
    ),
    Param(
        "start_iter_percent", float, yaml_only=True,
        doc="alternative spelling of start_iter as a percent of "
            "train.model.train_iters, resolved to the absolute step in "
            "the loader before anything is serialized (needs the "
            "train_iters context, hence yaml-only and absent from the "
            "canonical config); mutually exclusive with start_iter",
    ),
    Param(
        "period", int, default=1,
        doc="reselection period T (>= 1): the support is regenerated at "
            "full-sync boundaries every T steps and persists unchanged in "
            "between; residual accumulates on unselected coordinates",
    ),
    Param(
        "decouple_full_sync_interval_and_period", bool, default=False,
        doc="decouple the full-sync (residual fold + score emission) "
            "interval from the reselection period; true requires "
            "full_sync_interval to be set to a strict divisor of period",
    ),
    Param(
        "full_sync_interval", int,
        doc="full sync every N steps, N a strict divisor of period with "
            "N < period; only legal alongside "
            "decouple_full_sync_interval_and_period: true — absent means "
            "every cycle boundary is a full sync",
    ),
)

# ====================================================================
# engine checks (sga.checks.*) — consistency machinery toggles
# ====================================================================

CHECKS = (
    Param(
        "count_guard", bool, default=True,
        doc="layer-1 consistency guard: cross-rank element-count check of "
            "the selected support at each activation (runs once per cycle "
            "on cached counts); off is a research escape hatch, not a "
            "production setting",
    ),
)

# ====================================================================
# engine trace (sga.trace.*) — offline replay data collection
# ====================================================================

TRACE = (
    Param(
        "dir", str, required=True,
        doc="trace output directory; setting the sga.trace section at all "
            "activates the collector (an engine-side service — rejected "
            "when sga.enabled is false)",
    ),
    Param(
        "branch_points", list, default=None,
        doc="engine steps to dump full optimizer state at (comma string "
            "or int list; sorted, deduped, >= 0); none = no state dumps",
    ),
    Param(
        "window", int, default=100,
        doc="gradient-capture window length after each branch point "
            "(>= 1)",
    ),
)

ENGINE_SECTIONS = {"cycle": CYCLE, "checks": CHECKS, "trace": TRACE}

# ====================================================================
# score axis (sga.score.name) — registry DATA, no tunable params
# ====================================================================
# The score formulas themselves are registry data with their own docs in
# sga/plugins/scores.py (name -> formula + emission mode + doc); the name
# list below mirrors the registry and is kept in sync by the no-orphan
# invariant test. Any key besides `name` on this axis is a typo.

SCORE_NAMES = ("abs_g", "abs_m", "coi", "update")

# ====================================================================
# selection axis (sga.selection.*) — one registered subject: topk
# ====================================================================

SELECTION_PARAMS = {
    "topk": (
        Param(
            "method", str, required=True,
            choices=("adaptive", "exact", "histogram"),
            doc="selection algorithm: adaptive = persistent tau controller; "
                "exact = kthvalue + OR-merge; histogram = bit-pattern "
                "histogram select",
        ),
        Param(
            "ratio_percent", float, required=True,
            doc="topk mask ratio in percent, strictly inside (0, 100)",
        ),
        Param(
            "histogram_mode", str, default="exact",
            choices=("balanced", "exact", "fast"),
            requires=("method", "histogram"),
            doc="method=histogram: geometry tier exact / balanced / fast",
        ),
        Param(
            "adaptive_warmup_syncs", int, default=5, nullable=True,
            requires=("method", "adaptive"),
            doc="method=adaptive: gated warmup selects before the "
                "persistent tau takes over (null disables the warmup "
                "gate)",
        ),
        Param(
            "adaptive_warmup_mode", str, default="fast",
            choices=("balanced", "exact", "fast"),
            requires=("method", "adaptive"),
            doc="method=adaptive: histogram tier used to bootstrap/steer "
                "the adaptive tau",
        ),
    ),
}

# ====================================================================
# optimizer axis (sga.optimizer.*) — one registered template: adamw
# ====================================================================

OPTIMIZER_PARAMS = {
    "adamw": (
        Param(
            "residual_mean", bool, default=False,
            doc="pre-fold the accumulated residual by the fold window "
                "(the accumulation span between full syncs) at full-sync "
                "boundaries (residual-mean semantics)",
        ),
        Param(
            "merge_join_dir", str, default="table",
            choices=("table", "binary"),
            doc="compact-path merge-join bracket source: 'table' builds a "
                "1024-stride directory table of lower_bound(idx, j*1024) "
                "once per selection cycle (the bucket idx is frozen "
                "within a cycle, so the table cannot go stale) and "
                "brackets both lower_bounds to <=1024 candidates "
                "(~0.5ms build, amortized over the cycle's sparse steps); "
                "'binary' keeps the plain full-depth bisection (the "
                "historical form). Silently ignored on writeback "
                "backends and the torch fallback path (no compact "
                "merge-join there)",
        ),
    ),
}

# ====================================================================
# backend axis (sga.backend.*) — sparse allreduce transport selection
# ====================================================================

BACKEND_PARAMS = {
    "torch_pipelined_index": (
        Param(
            "high_priority_group", bool, default=False,
            doc="run sparse collectives on a twin high-priority NCCL "
                "group (true selects the legacy pipelined_idx_sorted "
                "TRANSPORT MODE only — reproducing that former "
                "configuration also needs slices: 3; false: legacy "
                "indexed_sparse on the training world's own group)",
        ),
        Param(
            "slices", int, default=2,
            doc="number of idx slices pipelined against the allreduces "
                "(enforced >= 1)",
        ),
    ),
    "torch_pipelined_mask": (
        Param(
            "num_slices", int, default=0,
            doc="mask slices pipelined against NCCL ops (0 = auto, i.e. 2)",
        ),
    ),
    "nccl_pipelined_index": (),
}

# No mutation after import: plugin_params()/the registration join read this
# dict live while flatargs caches its flag table with no invalidation — a
# post-import edit desynchronizes parser choices, the join and unknown-dest
# rejection (loudly, but confusingly). Test scaffolds must restore any row
# they add.
PLUGIN_PARAMS = {
    "selection": SELECTION_PARAMS,
    "optimizer": OPTIMIZER_PARAMS,
    "backend": BACKEND_PARAMS,
}
