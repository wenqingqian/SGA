# Copyright (c) 2026, SGA framework contributors.
"""Plugin declarations and registries for the four SGA config axes.

A scheme is a pure configuration combination (field model):
``score / selection / optimizer / backend``, each with its own params.
Scores are DATA in the score registry (sga/plugins/scores.py: name ->
emission mode + formula + doc — no classes); selection/optimizer/backend are
class plugins that each declare their attributes. The combination validator
checks legality from those declarations — there is NO hardcoded list of
legal combinations and NO scheme names anywhere in the engine.

Declaration facts used by the validator (see ``validate_combination``):
  - Optimizer.emittable_scores: the score names this template's full-step
    kernel can materialize into the fp32 main-grad copy (the optimizer->score
    handoff). The configured score MUST be a member — this is the single
    cross-axis legality rule; registry entries outside every template's
    emittable set are offline-only (replay-evaluable, config-rejected).
  - Optimizer.requires_full_sync_every_period: the residual contract only
    folds at full syncs, so cycle boundaries must be full.
  - Backend.consumes: 'mask' or 'idx' — the Support object provides both,
    so this routes the backend rather than restricting combinations.
"""

from dataclasses import dataclass

from sga.plugins.scores import Score, ScoreSpec, get_score_spec, score_names

__all__ = [
    "Score",
    "ScoreSpec",
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
    "validate_combination",
    "build_selection",
    "build_optimizer",
    "build_backend_plugin",
    "register_default_plugins",
]


@dataclass(frozen=True)
class SelectionDecl:
    name: str
    output_form: str = "mask"  # 'mask' | 'idx' (Support canonicalizes either way)


@dataclass(frozen=True)
class OptimizerDecl:
    name: str
    # Post-update buffer reset semantics the optimizer template owns.
    reset_mode: str = "selective"  # 'selective' | 'full'
    # Score names this template can materialize into main grads on full steps
    # (the optimizer->score handoff); the configured score must be a member.
    emittable_scores: frozenset = frozenset()
    # Residual contract folds only at full syncs => cycle boundaries must be full.
    requires_full_sync_every_period: bool = False

    @property
    def emits_score_view(self) -> bool:
        return bool(self.emittable_scores)


@dataclass(frozen=True)
class BackendDecl:
    name: str
    consumes: str = "mask"  # 'mask' | 'idx'


_SELECTIONS: dict = {}
_OPTIMIZERS: dict = {}
_BACKEND_PLUGINS: dict = {}


def register_selection(name, decl):
    _register(_SELECTIONS, name, decl)


def register_optimizer(name, decl):
    _register(_OPTIMIZERS, name, decl)


def register_backend_plugin(name, decl):
    _register(_BACKEND_PLUGINS, name, decl)


def _register(registry, name, decl):
    if name in registry:
        raise ValueError(f"Duplicate plugin registration for '{name}'")
    registry[name] = decl


def get_selection_decl(name):
    return _lookup(_SELECTIONS, name, "selection")


def get_optimizer_decl(name):
    return _lookup(_OPTIMIZERS, name, "optimizer")


def get_backend_plugin_decl(name):
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
    (start_iter / period / full_sync_interval). ``selection`` and ``backend``
    are accepted for interface completeness only — the cross-axis legality
    rules live on the optimizer decl (checked below).
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
    if optimizer.requires_full_sync_every_period:
        period = cycle.get("period", 1)
        if cycle.get("full_sync_interval", 0) not in (0, period):
            problems.append(
                f"optimizer '{optimizer.name}' folds residuals only at full "
                f"syncs: full_sync_interval must be 0 or equal to the cycle "
                f"period ({period})"
            )
    if problems:
        raise ValueError("Illegal SGA plugin combination: " + "; ".join(problems))


def build_selection(name, params):
    from sga.plugins.selection.circular_topk import CircularTopkSelection

    return {
        "circular_topk": CircularTopkSelection,
    }[name](params)


def build_optimizer(name, params, emit_score: str = None):
    from sga.plugins.optimizer.adamw import SgaAdamW

    return {
        "adamw": SgaAdamW,
    }[name](params, emit_score=emit_score)


def build_backend_plugin(name, params):
    from sga.backends.masked_pipelined import MaskedPipelinedBackend
    from sga.backends.indexed_sparse import IndexedSparseBackend
    from sga.backends.pipelined_idx_sorted import PipelinedIdxSortedBackend

    return {
        "masked_pipelined": MaskedPipelinedBackend,
        "indexed_sparse": IndexedSparseBackend,
        "pipelined_idx_sorted": PipelinedIdxSortedBackend,
    }[name](params)


def register_default_plugins() -> None:
    """Idempotent registration of the built-in plugin declarations."""
    from sga.plugins.selection.circular_topk import DECL as topk_decl
    from sga.plugins.optimizer.adamw import DECL as adamw_decl

    for name, decl in (
        ("circular_topk", topk_decl),
    ):
        if name not in _SELECTIONS:
            register_selection(name, decl)
    for name, decl in (
        ("adamw", adamw_decl),
    ):
        if name not in _OPTIMIZERS:
            register_optimizer(name, decl)

    from sga.backends.masked_pipelined import DECL as mp_decl
    from sga.backends.indexed_sparse import DECL as is_decl
    from sga.backends.pipelined_idx_sorted import DECL as pis_decl

    for name, decl in (
        ("masked_pipelined", mp_decl),
        ("indexed_sparse", is_decl),
        ("pipelined_idx_sorted", pis_decl),
    ):
        if name not in _BACKEND_PLUGINS:
            register_backend_plugin(name, decl)
