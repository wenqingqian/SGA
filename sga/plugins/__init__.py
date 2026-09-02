# Copyright (c) 2026, SGA framework contributors.
"""Plugin declarations and registries for the four SGA config axes.

A scheme is a pure configuration combination:
``score / selection / optimizer / backend``, each with its own params.
Scores are DATA in the score registry (sga/plugins/scores.py: name ->
emission mode + formula + doc — no classes); selection/optimizer/backend are
class plugins that each declare their attributes. The combination validator
checks legality from those declarations — there is NO hardcoded list of
legal combinations and NO scheme names anywhere in the engine.

Every plugin also declares its PARAMETER VOCABULARY (``PluginParam`` entries
on the decl): the loader rejects unknown keys against it (a typo'd param
fails at launch instead of being silently swallowed by ``params.get``), and
the flat-flag vocabulary (sga/config/arguments.py) is generated from the
same declarations — one source of truth per axis. Constructors read their
defaults through ``resolve_plugin_params`` instead of restating them.

Declaration facts used by the validator (see ``validate_combination``):
  - Optimizer.emittable_scores: the score names this template's full-step
    kernel can materialize into the fp32 main-grad copy (the optimizer->score
    handoff). The configured score MUST be a member — this is the single
    cross-axis legality rule; registry entries outside every template's
    emittable set are offline-only (replay-evaluable, config-rejected).
  - Backend.consumes: 'mask' or 'idx' — the Support object provides both,
    so this routes the backend rather than restricting combinations.

Cycle-calendar legality is engine-level (not optimizer-declared): reselection
boundaries are structurally full syncs, so the residual fold contract holds
for any decoupled full-sync interval that divides the period.
"""

from dataclasses import dataclass
from typing import Any

from sga.plugins.scores import Score, ScoreSpec, get_score_spec, score_names

__all__ = [
    "Score",
    "ScoreSpec",
    "PluginParam",
    "get_score_spec",
    "score_names",
    "SelectionDecl",
    "OptimizerDecl",
    "BackendDecl",
    "register_selection",
    "register_optimizer",
    "register_backend_plugin",
    "get_selection_decl",
    "get_optimizer_decl",
    "get_backend_plugin_decl",
    "registered_names",
    "check_plugin_params",
    "resolve_plugin_params",
    "validate_combination",
    "build_selection",
    "build_optimizer",
    "build_backend_plugin",
    "register_default_plugins",
]


@dataclass(frozen=True)
class PluginParam:
    """One plugin-parameter declaration (the axis's config vocabulary).

    ``type`` is the yaml-side scalar type (bool/int/float/str); ``choices``
    (when non-empty) restricts str params; ``nullable`` admits an explicit
    null (e.g. warmup windows that can be turned off); required params have
    no default. The same entries drive the loader's unknown-key/type checks,
    the constructors' default resolution, and the flat flags.
    """

    name: str
    type: object
    required: bool = False
    default: Any = None
    doc: str = ""
    choices: tuple = ()
    nullable: bool = False


def check_plugin_params(axis: str, plugin_name: str, decl, params: dict) -> None:
    """Fail loud on unknown keys / missing required / wrong types.

    The single param-validation entry shared by the loader (config time) and
    ``resolve_plugin_params`` (build time): an unknown key here is a typo in
    a config that ``params.get`` would otherwise silently swallow.
    """
    known = {p.name for p in decl.params}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(
            f"sga.{axis} ({plugin_name}): unknown params {unknown} "
            f"(known: {sorted(known)})"
        )
    for p in decl.params:
        if p.name not in params:
            if p.required:
                raise ValueError(
                    f"sga.{axis} ({plugin_name}): param '{p.name}' is "
                    "required"
                )
            continue
        _check_value(axis, plugin_name, p, params[p.name])


def _check_value(axis: str, plugin_name: str, p: "PluginParam", val) -> None:
    """Validate one (param, value) pair: nullability, type, choices."""
    if val is None:
        if p.nullable:
            return
        raise ValueError(
            f"sga.{axis} ({plugin_name}): param '{p.name}' is not "
            "nullable (got null)"
        )
    if p.type is bool:
        ok = isinstance(val, bool)
    elif p.type is int:
        # bool is an int subclass; yaml true/false must not pass as 1/0
        ok = isinstance(val, int) and not isinstance(val, bool)
    elif p.type is float:
        ok = (isinstance(val, (int, float)) and not isinstance(val, bool))
    else:
        ok = isinstance(val, p.type)
    if not ok:
        raise ValueError(
            f"sga.{axis} ({plugin_name}): param '{p.name}' must be "
            f"{p.type.__name__} (got {val!r})"
        )
    if p.choices and val not in p.choices:
        raise ValueError(
            f"sga.{axis} ({plugin_name}): param '{p.name}' must be one "
            f"of {list(p.choices)} (got {val!r})"
        )


def resolve_plugin_params(axis: str, plugin_name: str, decl,
                          params: dict) -> dict:
    """Validate and expand a params dict to its full defaulted form.

    The constructors' single read path: defaults come from the DECLARATION,
    never from a restated ``params.get(name, default)``.
    """
    check_plugin_params(axis, plugin_name, decl, params)
    out = {p.name: p.default for p in decl.params if not p.required}
    out.update(params)
    return out


@dataclass(frozen=True)
class SelectionDecl:
    name: str
    output_form: str = "mask"  # 'mask' | 'idx' (Support canonicalizes either way)
    params: tuple = ()         # (PluginParam, ...)
    # Optional callable(params)->dict: vocabulary-aware default expansion for
    # inspection CLIs (e.g. topk's per-method view). None = plain resolve.
    effective: object = None


@dataclass(frozen=True)
class OptimizerDecl:
    name: str
    # Post-update buffer reset semantics the optimizer template owns.
    reset_mode: str = "selective"  # 'selective' | 'full'
    # Score names this template can materialize into main grads on full steps
    # (the optimizer->score handoff); the configured score must be a member.
    emittable_scores: frozenset = frozenset()
    params: tuple = ()             # (PluginParam, ...)
    # Same inspection hook as SelectionDecl.effective (unused today).
    effective: object = None

    @property
    def emits_score_view(self) -> bool:
        return bool(self.emittable_scores)


@dataclass(frozen=True)
class BackendDecl:
    name: str
    consumes: str = "mask"  # 'mask' | 'idx'
    params: tuple = ()      # (PluginParam, ...)
    # Same inspection hook as SelectionDecl.effective (unused today).
    effective: object = None


_SELECTIONS: dict = {}
_OPTIMIZERS: dict = {}
_BACKEND_PLUGINS: dict = {}

# Absorbed backend names -> the migration hint raised on lookup. The
# unification  merged indexed_sparse + pipelined_idx_sorted
# into torch_pipelined_index (its high_priority_group param selects the
# legacy behavior) and renamed masked_pipelined -> torch_pipelined_mask.
_RENAMED_BACKENDS = {
    "indexed_sparse": "torch_pipelined_index (high_priority_group: false, "
                      "the default, reproduces indexed_sparse)",
    "pipelined_idx_sorted": "torch_pipelined_index (high_priority_group: "
                            "true, slices: 3 reproduces pipelined_idx_sorted)",
    "masked_pipelined": "torch_pipelined_mask",
}


def register_selection(name, decl):
    _register("selection", _SELECTIONS, name, decl)


def register_optimizer(name, decl):
    _register("optimizer", _OPTIMIZERS, name, decl)


def register_backend_plugin(name, decl):
    _register("backend", _BACKEND_PLUGINS, name, decl)


def _register(axis, registry, name, decl):
    if name in registry:
        raise ValueError(f"Duplicate plugin registration for '{name}'")
    # A misdeclared default (wrong type/choices, or None on a non-nullable
    # param) would otherwise sail through every constructor gate.
    for p in decl.params:
        if not p.required:
            _check_value(axis, name, p, p.default)
    registry[name] = decl


def get_selection_decl(name):
    return _lookup(_SELECTIONS, name, "selection")


def get_optimizer_decl(name):
    return _lookup(_OPTIMIZERS, name, "optimizer")


def get_backend_plugin_decl(name):
    if name in _RENAMED_BACKENDS:
        raise ValueError(
            f"SGA backend '{name}' was absorbed in the backend unification; "
            f"use {_RENAMED_BACKENDS[name]}"
        )
    return _lookup(_BACKEND_PLUGINS, name, "backend")


def _lookup(registry, name, axis):
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown SGA {axis} plugin '{name}'. Registered: {sorted(registry)}"
        ) from None


def registered_names() -> dict:
    return {
        "score": score_names(),
        "selection": sorted(_SELECTIONS),
        "optimizer": sorted(_OPTIMIZERS),
        "backend": sorted(_BACKEND_PLUGINS),
    }


def validate_combination(score_name: str, selection: SelectionDecl,
                         optimizer: OptimizerDecl, backend, cycle: dict) -> None:
    """Declaration-based combination validation (config time, no scheme names).

    ``backend`` is a BackendDecl; ``cycle`` is the engine cycle config dict
    (start_iter / period / decouple_full_sync_interval_and_period /
    full_sync_interval). ``selection`` and ``backend`` are accepted for
    interface completeness only — the cross-axis legality rule lives on the
    optimizer decl (emittable scores); the cycle-calendar rules are
    engine-level (checked below).
    """
    problems = []
    get_score_spec(score_name)  # unknown name fails with the registered list
    if score_name not in optimizer.emittable_scores:
        problems.append(
            f"optimizer '{optimizer.name}' cannot materialize score "
            f"'{score_name}' into main grads on full steps (emittable: "
            f"{sorted(optimizer.emittable_scores) or 'nothing'}) — the score "
            "is offline-only under this optimizer"
        )
    period = int(cycle.get("period", 1))
    decouple = bool(cycle.get("decouple_full_sync_interval_and_period", False))
    interval = cycle.get("full_sync_interval", None)
    if not decouple:
        if interval is not None:
            problems.append(
                f"sga.cycle.full_sync_interval is set ({interval}) but "
                "decouple_full_sync_interval_and_period is false; every "
                "cycle boundary is already a full sync — leave "
                "full_sync_interval unset"
            )
    elif interval is None:
        problems.append(
            "decouple_full_sync_interval_and_period is true but "
            "sga.cycle.full_sync_interval is unset — set a divisor of the "
            f"period smaller than it (period={period})"
        )
    elif not (0 < int(interval) < period and period % int(interval) == 0):
        problems.append(
            f"decoupled full_sync_interval ({interval}) must satisfy "
            f"0 < full_sync_interval < period ({period}) and "
            f"period % full_sync_interval == 0"
        )
    if problems:
        raise ValueError("Illegal SGA plugin combination: " + "; ".join(problems))


def build_selection(name, params):
    from sga.plugins.selection.topk import TopkSelection

    register_default_plugins()
    params = _resolved("selection", name, params)  # curated unknown-name error
    return {"topk": TopkSelection}[name](params)


def build_optimizer(name, params, emit_score: str = None):
    from sga.plugins.optimizer.adamw import SgaAdamW

    register_default_plugins()
    params = _resolved("optimizer", name, params)  # curated unknown-name error
    return {"adamw": SgaAdamW}[name](params, emit_score=emit_score)


def build_backend_plugin(name, params):
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
    from sga.backends.torch_pipelined_mask import TorchPipelinedMaskBackend
    from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend

    register_default_plugins()
    params = _resolved("backend", name, params)  # curated unknown-name error
    return {
        "torch_pipelined_index": TorchPipelinedIndexBackend,
        "torch_pipelined_mask": TorchPipelinedMaskBackend,
        "nccl_pipelined_index": NcclPipelinedIndexBackend,
    }[name](params)


def _resolved(axis, name, params):
    """Curated unknown-plugin error + name strip; the CONSTRUCTOR stays the
    single resolver (defaults expand once, explicit-vs-default stays
    distinguishable for cross-param legality like topk's per-method
    foreign-key refusal)."""
    if axis == "backend":
        get_backend_plugin_decl(name)  # owns the _RENAMED_BACKENDS hints
    else:
        registry = {"selection": _SELECTIONS, "optimizer": _OPTIMIZERS}[axis]
        _lookup(registry, name, axis)
    params = dict(params)
    params.pop("name", None)
    return params


def register_default_plugins() -> None:
    """Idempotent registration of the built-in plugin declarations."""
    from sga.plugins.selection.topk import DECL as topk_decl
    from sga.plugins.optimizer.adamw import DECL as adamw_decl

    for name, decl in (
        ("topk", topk_decl),
    ):
        if name not in _SELECTIONS:
            register_selection(name, decl)
    for name, decl in (
        ("adamw", adamw_decl),
    ):
        if name not in _OPTIMIZERS:
            register_optimizer(name, decl)

    from sga.backends.torch_pipelined_index import DECL as tpi_decl
    from sga.backends.torch_pipelined_mask import DECL as tpm_decl
    from sga.backends.nccl_pipelined_index import DECL as npi_decl

    for name, decl in (
        ("torch_pipelined_index", tpi_decl),
        ("torch_pipelined_mask", tpm_decl),
        ("nccl_pipelined_index", npi_decl),
    ):
        if name not in _BACKEND_PLUGINS:
            register_backend_plugin(name, decl)
