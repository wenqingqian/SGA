# Copyright (c) 2026, SGA framework contributors.
"""Plugin declarations and registries for the four SGA config axes.

A scheme is a pure configuration combination:
``score / selection / optimizer / backend``, each with its own params.
Scores are DATA in the score registry (sga/plugins/scores.py: name ->
emission mode + formula + doc — no classes); selection/optimizer/backend are
class plugins that each declare their attributes. The combination validator
checks legality from those declarations — there is NO hardcoded list of
legal combinations and NO scheme names anywhere in the engine.

Plugin PARAMS: the flat flags live in ``sga/config/arguments.py`` (plain
add_argument lines); which params belong to which plugin is EACH PLUGIN'S
OWN business — each plugin module declares its full flat dests in its
``PARAMS`` tuple and constructors expand their params through
``expand_plugin_params`` below (defaults come from the flag declarations,
never restated; unknown/missing params fail loud). There is no central
plugin-param table anywhere; config-side validation checks values plus one
name-attribution rule — an explicitly-set key owned by a NON-selected plugin
is rejected on sight (a sibling plugin's parameter is dead config; see
sga/config/validation.py). Names carry no structure: PARAMS entries are
complete dests, read directly — nothing is prefixed, stripped or derived.

Declaration facts used by the validator (see ``validate_combination``):
  - Optimizer.emittable_scores: the score names this template's full-step
    kernel can materialize into the fp32 main-grad copy (the optimizer->score
    handoff). The configured score MUST be a member — this is the single
    cross-axis legality rule; registry entries outside every template's
    emittable set are offline-only (replay-evaluable, config-rejected).
  - Backend.consumes: 'mask' or 'idx' — the Support object provides both,
    so this routes the backend rather than restricting combinations.
  - Backend.emission + Optimizer.consumes_compact: a compact-emission
    backend leaves the communicated values out of the grad buffer and
    hands (idx, values) back; the optimizer must declare that its sparse
    path consumes that handoff, or the combination is rejected.

Cycle-calendar legality is engine-level and lives in
sga.config.validation (flat keys).
"""

from dataclasses import dataclass
from typing import Literal, Optional

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
    "expand_plugin_params",
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
    # Whether the template's sparse path consumes the compact-emission
    # (idx, values) handoff. A compact backend requires True: the values
    # never enter the grad buffer, so an optimizer without the compact
    # commit would silently drop every communicated gradient on sparse
    # steps (enforced in validate_combination).
    consumes_compact: bool = False
    # Score names this template can materialize into main grads on full steps
    # (the optimizer->score handoff); the configured score must be a member.
    emittable_scores: frozenset = frozenset()

    @property
    def emits_score_view(self) -> bool:
        return bool(self.emittable_scores)


@dataclass(frozen=True)
class BackendDecl:
    name: str
    consumes: str = "mask"  # 'mask' | 'idx'
    # What allreduce_sparse LEAVES BEHIND: 'writeback' = the
    # communicated values are scattered back into the full grad buffer
    # (returns None); 'compact' = they STAY in the backend's compact buffer
    # and the return value is that values view (empty tensor when nothing
    # was selected) — the engine's clip/grad_correct then consume
    # (idx, values) directly. SGA-internal contract (the waist's three call
    # points and registries are unchanged — folded into the v2 flat-args
    # API rather than bumping it).
    emission: Literal["writeback", "compact"] = "writeback"

    def resolve_emission(self, value):
        """The effective emission for a flat ``sga_backend_emission``
        value: None/'auto' -> this backend's inherent emission; an
        explicit value passes through (legality is the config layer's).
        The ONE home of the auto-resolution rule — constructors and the
        combination validator both read it here."""
        return self.emission if value in (None, "auto") else value


_SELECTIONS: dict = {}
_OPTIMIZERS: dict = {}
_BACKEND_PLUGINS: dict = {}

# Absorbed backend names -> the migration hint raised on lookup. The
# unification merged indexed_sparse + pipelined_idx_sorted
# into torch_pipelined_index (its high_priority_group param selects the
# legacy behavior) and renamed masked_pipelined -> torch_pipelined_mask.
_RENAMED_BACKENDS = {
    "indexed_sparse": "torch_pipelined_index (high_priority_group: false, "
                      "the default, reproduces indexed_sparse)",
    "pipelined_idx_sorted": "torch_pipelined_index (high_priority_group: "
                            "true, index_slices: 3 reproduces pipelined_idx_sorted)",
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
    # The engine routes emission by EXACT equality, so a typo'd declaration
    # would silently behave as writeback while the backend skips the
    # writeback scatter. Only backend decls carry the field; the getattr
    # default keeps the check axis-generic.
    emission = getattr(decl, "emission", "writeback")
    if emission not in ("writeback", "compact"):
        raise ValueError(
            f"sga.{axis} ({name}): decl emission must be 'writeback' or "
            f"'compact' (got {emission!r})"
        )
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
                         optimizer: OptimizerDecl, backend,
                         backend_emission: Optional[str] = None) -> None:
    """Declaration-based CROSS-AXIS combination validation (config time,
    no scheme names).

    ``backend`` is a BackendDecl; ``backend_emission`` is the resolved
    flat value of sga_backend_emission (None/'auto' -> the decl's
    inherent emission — emission is a runtime switch on
    torch_pipelined_index, so the decl field alone no longer decides
    which handoff the optimizer must consume). ``selection`` is accepted
    for interface completeness only; the rules checked here are the ones
    the decls declare (emittable scores; compact emission requires an
    optimizer that consumes the compact handoff). The cycle-calendar
    rules are engine-level and live in sga.config.validation (flat keys).
    """
    problems = []
    get_score_spec(score_name)  # unknown score fails with the registered list
    if score_name not in optimizer.emittable_scores:
        problems.append(
            f"optimizer '{optimizer.name}' cannot materialize score "
            f"'{score_name}' into main grads on full steps (emittable: "
            f"{sorted(optimizer.emittable_scores) or 'nothing'}) — the score "
            "is offline-only under this optimizer"
        )
    effective_emission = backend.resolve_emission(backend_emission)
    if effective_emission == "compact" and not optimizer.consumes_compact:
        problems.append(
            f"backend '{backend.name}' resolves to emission='compact' but "
            f"optimizer '{optimizer.name}' does not consume the compact "
            "(idx, values) handoff (consumes_compact=false) — the "
            "communicated values would never reach the m/v commit and "
            "every sparse step would silently drop them"
        )
    if problems:
        raise ValueError("Illegal SGA plugin combination: " + "; ".join(problems))


def build_selection(name, params):
    from sga.plugins.selection.topk import PARAMS as TOPK_PARAMS, TopkSelection

    register_default_plugins()
    _lookup(_SELECTIONS, name, "selection")  # curated unknown-name error
    classes = {"topk": TopkSelection}
    return classes[name](_axis_params((TOPK_PARAMS,), params))


def build_optimizer(name, params, emit_score: str = None):
    from sga.plugins.optimizer.adamw import PARAMS as ADAMW_PARAMS, SgaAdamW

    register_default_plugins()
    _lookup(_OPTIMIZERS, name, "optimizer")  # curated unknown-name error
    classes = {"adamw": SgaAdamW}
    return classes[name](_axis_params((ADAMW_PARAMS,), params),
                         emit_score=emit_score)


def build_backend_plugin(name, params):
    from sga.backends.nccl_pipelined_index import (
        PARAMS as NPI_PARAMS,
        NcclPipelinedIndexBackend,
    )
    from sga.backends.torch_pipelined_index import (
        PARAMS as TPI_PARAMS,
        TorchPipelinedIndexBackend,
    )
    from sga.backends.torch_pipelined_mask import (
        PARAMS as TPM_PARAMS,
        TorchPipelinedMaskBackend,
    )

    register_default_plugins()
    get_backend_plugin_decl(name)  # owns the _RENAMED_BACKENDS hints
    classes = {
        "torch_pipelined_index": TorchPipelinedIndexBackend,
        "torch_pipelined_mask": TorchPipelinedMaskBackend,
        "nccl_pipelined_index": NcclPipelinedIndexBackend,
    }
    return classes[name](_axis_params(
        (TPI_PARAMS, TPM_PARAMS, NPI_PARAMS), params))


def _axis_params(param_vocabs, params):
    """Slice the explicit flat dict down to one axis: the union of the
    axis' plugin vocabularies (full flat dests, each declared in the
    plugin's own ``PARAMS``). Keys outside every vocabulary belong to
    another axis and are not routed here; a key owned by a SIBLING
    plugin of this axis is rejected at validation on the config route;
    for direct construction (replay/tests) it survives this slice and
    is rejected by the selected constructor's own unknown-param check
    (fail loud)."""
    vocab = set().union(*(set(v) for v in param_vocabs))
    return {dest: value for dest, value in params.items() if dest in vocab}


def expand_plugin_params(param_names, params, owner):
    """A plugin constructor's single read path: its explicit params (full
    flat dests) -> full defaults. Defaults and value legality
    (type/choices) come from the flag declarations
    (``sga.config.validation`` — single source, never restated); a param
    whose declared default is None is REQUIRED. Each plugin owns its own
    vocabulary — unknown params, missing required ones and wrong-valued
    ones fail loud here, at build time (an explicit-but-at-default
    foreign key may be diffed away before the constructor — the engine
    builder re-diffs the canonical down to explicit non-defaults — but
    the config route now rejects sibling-plugin keys at validation
    regardless of value, so that carve-out is reachable only via direct
    construction, whose params reach the unknown-param check undiffed
    and fail loud)."""
    from sga.config.validation import _coerce, sga_defaults

    names = tuple(param_names)
    unknown = sorted(set(params) - set(names))
    if unknown:
        raise ValueError(
            f"{owner}: unknown params {unknown} (known: {sorted(names)})")
    defaults = sga_defaults()
    full = {}
    for name in names:
        if name not in defaults:
            raise ValueError(
                f"{owner}: param {name!r} is not a declared flag (the "
                "plugin PARAMS vocabulary must name dests declared in "
                "sga/config/arguments.py)")
        full[name] = defaults[name]
    for name in sorted(names):
        if full[name] is None and name not in params:
            raise ValueError(f"{owner}: param '{name}' is required")
    full.update(params)
    # the config route already coerced every explicit value; this re-run
    # is idempotent there and guards direct construction (replay/tests)
    for name in params:
        full[name] = _coerce(name, full[name])
    return full


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
