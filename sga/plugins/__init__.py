# Copyright (c) 2026, SGA framework contributors.
"""Plugin declarations and registries for the four SGA config axes.

A scheme is a pure configuration combination (field model):
``score / selection / error_feedback / backend``, each with its own params.
Every plugin declares its attributes; the combination validator checks
legality from those declarations — there is NO hardcoded list of legal
combinations and NO scheme names anywhere in the engine.

Declaration facts used by the validator (see ``validate_combination``):
  - Score.scoring_input: 'local_or_merged' (score from local accumulated
    grads, mask OR-merged across DP — selection may run every step) or
    'synced' (score needs the fully-synced gradient state of a full-sync
    step, e.g. an update materialized by the EF kernel).
  - Score.has_threshold_path: the score offers a fused ``threshold_mask``
    fast path (abs_g does; update_ratio does not).
  - Selection.scope: 'elementwise_local' vs 'global_ranked' (topk over the
    whole model needs a ranked view, e.g. the bit-pattern histogram).
  - ErrorFeedback.emittable_scores: the score plugins this EF can
    materialize into the fp32 main-grad copy on full steps (radius-style
    EF->score handoff); 'synced' scores must be paired with an EF that
    lists them.
  - ErrorFeedback.requires_full_sync_every_period: the EF's residual
    contract only folds at full syncs, so cycle boundaries must be full.
  - Backend.consumes: 'mask' or 'idx' — the Support object provides both,
    so this routes the backend rather than restricting combinations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScoreDecl:
    name: str
    # What gradient state the score needs.
    scoring_input: str = "local_or_merged"  # 'local_or_merged' | 'synced'
    # Fused ``threshold_mask`` fast path available (needed by threshold selection).
    has_threshold_path: bool = False


@dataclass(frozen=True)
class SelectionDecl:
    name: str
    output_form: str = "mask"  # 'mask' | 'idx' (Support canonicalizes either way)
    scope: str = "elementwise_local"  # 'elementwise_local' | 'global_ranked'
    # Requires the paired score to expose a fused threshold fast path.
    needs_threshold_path: bool = False


@dataclass(frozen=True)
class EFDecl:
    name: str
    # Post-update buffer reset semantics the EF owns.
    reset_mode: str = "selective"  # 'selective' | 'full'
    # Needs the engine-allocated uint8 accumulation-count buffers.
    needs_acc_count: bool = False
    # Score plugins this EF can materialize into main grads on full steps
    # (the EF->score handoff; 'synced' scores must be paired accordingly).
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


_SCORES: dict = {}
_SELECTIONS: dict = {}
_EFS: dict = {}
_BACKEND_PLUGINS: dict = {}


def register_score(name, decl):
    _register(_SCORES, name, decl)


def register_selection(name, decl):
    _register(_SELECTIONS, name, decl)


def register_ef(name, decl):
    _register(_EFS, name, decl)


def register_backend_plugin(name, decl):
    _register(_BACKEND_PLUGINS, name, decl)


def _register(registry, name, decl):
    if name in registry:
        raise ValueError(f"Duplicate plugin registration for '{name}'")
    registry[name] = decl


def get_score_decl(name):
    return _lookup(_SCORES, name, "score")


def get_selection_decl(name):
    return _lookup(_SELECTIONS, name, "selection")


def get_ef_decl(name):
    return _lookup(_EFS, name, "error_feedback")


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
        "score": sorted(_SCORES),
        "selection": sorted(_SELECTIONS),
        "error_feedback": sorted(_EFS),
        "backend": sorted(_BACKEND_PLUGINS),
    }


def validate_combination(score: ScoreDecl, selection: SelectionDecl,
                         ef: EFDecl, backend, cycle: dict) -> None:
    """Declaration-based combination validation (config time, no scheme names).

    ``backend`` is a BackendDecl; ``cycle`` is the engine cycle config dict
    (start_iter / period / full_sync_interval).
    """
    problems = []
    if selection.needs_threshold_path and not score.has_threshold_path:
        problems.append(
            f"selection '{selection.name}' needs a score with a fused threshold "
            f"path, but score '{score.name}' has none"
        )
    if score.scoring_input == "synced" and score.name not in ef.emittable_scores:
        problems.append(
            f"score '{score.name}' needs synced scoring input (a value the EF "
            f"materializes into main grads on full steps), but error_feedback "
            f"'{ef.name}' can only emit {sorted(ef.emittable_scores) or 'nothing'}"
        )
    if selection.scope == "global_ranked" and score.scoring_input != "synced":
        problems.append(
            f"selection '{selection.name}' is global-ranked and requires a "
            f"'synced' score input (got '{score.scoring_input}')"
        )
    if ef.requires_full_sync_every_period:
        period = cycle.get("period", 1)
        if cycle.get("full_sync_interval", 0) not in (0, period):
            problems.append(
                f"error_feedback '{ef.name}' folds residuals only at full "
                f"syncs: full_sync_interval must be 0 or equal to the cycle "
                f"period ({period})"
            )
    if problems:
        raise ValueError("Illegal SGA plugin combination: " + "; ".join(problems))


def build_score(name, params):
    from sga.plugins.scores.abs_g import AbsGradScore
    from sga.plugins.scores.abs_g_synced import AbsGSyncedScore
    from sga.plugins.scores.abs_m import AbsMScore
    from sga.plugins.scores.coi import CoiScore
    from sga.plugins.scores.update_ratio import UpdateRatioScore

    return {
        "abs_g": AbsGradScore,
        "abs_g_synced": AbsGSyncedScore,
        "abs_m": AbsMScore,
        "coi": CoiScore,
        "update_ratio": UpdateRatioScore,
    }[name](params)


def build_selection(name, params, score=None):
    from sga.plugins.selection.threshold import ThresholdSelection
    from sga.plugins.selection.circular_topk import CircularTopkSelection

    return {
        "threshold": ThresholdSelection,
        "circular_topk": CircularTopkSelection,
    }[name](params, score=score)


def build_ef(name, params, emit_score: Optional[str] = None):
    from sga.plugins.error_feedback.masked_adamw import MaskedAdamWEF
    from sga.plugins.error_feedback.radius_opt import RadiusOptEF

    return {
        "masked_adamw": MaskedAdamWEF,
        "radius_opt": RadiusOptEF,
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
    from sga.plugins.scores.abs_g import DECL as abs_g_decl
    from sga.plugins.scores.abs_g_synced import DECL as abs_g_synced_decl
    from sga.plugins.scores.abs_m import DECL as abs_m_decl
    from sga.plugins.scores.coi import DECL as coi_decl
    from sga.plugins.scores.update_ratio import DECL as update_ratio_decl
    from sga.plugins.selection.threshold import DECL as threshold_decl
    from sga.plugins.selection.circular_topk import DECL as topk_decl
    from sga.plugins.error_feedback.masked_adamw import DECL as masked_adamw_decl
    from sga.plugins.error_feedback.radius_opt import DECL as radius_opt_decl

    for name, decl in (
        ("abs_g", abs_g_decl),
        ("abs_g_synced", abs_g_synced_decl),
        ("abs_m", abs_m_decl),
        ("coi", coi_decl),
        ("update_ratio", update_ratio_decl),
    ):
        if name not in _SCORES:
            register_score(name, decl)
    for name, decl in (
        ("threshold", threshold_decl),
        ("circular_topk", topk_decl),
    ):
        if name not in _SELECTIONS:
            register_selection(name, decl)
    for name, decl in (
        ("masked_adamw", masked_adamw_decl),
        ("radius_opt", radius_opt_decl),
    ):
        if name not in _EFS:
            register_ef(name, decl)

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
