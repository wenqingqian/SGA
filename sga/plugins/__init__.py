# Copyright (c) 2026, SGA framework contributors.
"""Class-plugin registries for the three class axes of SGA.

A scheme is a pure configuration combination ``score / selection /
optimizer / backend``. Scores are data in sga/plugins/scores.py (name ->
formula + doc, no classes); selection/optimizer/backend are CLASSES, one
instance each per engine, built zero-arg — every plugin reads its own
config straight off ``sga.config.get_args()`` in its constructor, so the
builder never routes params and there is no param vocabulary anywhere in
this file. Registration is the same idiom as the score registry: a global
``name -> class`` dict per axis, ``register_*`` with an in-function
duplicate check, ``register_default_plugins`` filling defaults idempotently
per name, and accessors lazily ensuring defaults before lookup.

Self-declared class facts (read by config validation and the engine):
  - ``consumes`` (backend): 'mask' or 'idx' — the handoff form its
    allreduce_sparse consumes; the Support object provides both, so this
    routes the backend rather than restricting combinations.
  - ``inherent_emission`` (backend): 'writeback' or 'compact' — what
    allreduce_sparse LEAVES BEHIND when ``sga_backend_emission='auto'``
    (the flag's explicit value wins otherwise; legality is the config
    layer's). Compact means the communicated values stay in the backend's
    compact buffer and are handed back as (idx, values) — only
    compact-capable optimizers may pair with them (config-checked).
  - ``block_form`` (idx backends): 'block_idx' or 'block_desc' — which
    bb-layout support form the backend transports, torch block heads
    (Triton gather/scatter) or nccl-packed descriptors (head | tail_len).
    Drives the engine's activation-time materialization and the
    span-alignment attach gate (descriptor packing needs x-aligned span
    starts); the default 'block_idx' keeps third-party idx backends
    working unchanged.

No hardcoded combination table lives here: value-level cross rules
(emission vs optimizer compact-consumption, score emittability) are
checked in sga/config/validation.py, the megatron-way single validation
pass.
"""

from sga.plugins.scores import ScoreSpec, get_score_spec, score_names

__all__ = [
    "ScoreSpec",
    "get_score_spec",
    "score_names",
    "resolve_emission",
    "resolve_layout",
    "register_selection",
    "register_optimizer",
    "register_backend_plugin",
    "register_default_plugins",
    "plugin_class",
    "build_selection",
    "build_optimizer",
    "build_backend_plugin",
]

_SELECTIONS: dict = {}
_OPTIMIZERS: dict = {}
_BACKENDS: dict = {}

_REGISTRIES = {"selection": _SELECTIONS, "optimizer": _OPTIMIZERS,
               "backend": _BACKENDS}


def resolve_emission(value, inherent):
    """The backend constructors' shared emission resolution: unset/'auto'
    takes the backend's inherent form; an explicit value wins (its
    legality is the config layer's, plus the constructors' own membership
    backstops — a bogus spelling must fail BEFORE this resolves it)."""
    return inherent if value in (None, "auto") else value


def resolve_layout(layout, block_size):
    """The single 'auto' derivation of ``sga_backend_layout``:
    element-granular selection names er, block-granular names bb. An
    EXPLICIT value always passes through untouched (its legality is the
    config layer's — eb under block_size 1 is refused there, not folded
    here). Config validation writes the resolved value back onto the
    args singleton, so production readers see er|eb|bb; this helper also
    serves direct construction paths (unit tests) whose args never ran
    validation — an 'auto' left on the namespace resolves identically
    here.

    The taxonomy's wire forms: er/eb transport the element idx (eb's is
    the block selection expanded to elements — same wire form, only the
    distribution differs); bb transports the backend's declared block
    form — torch block heads or nccl packed descriptors."""
    if layout not in (None, "auto"):
        return layout
    return "er" if block_size == 1 else "bb"


def register_selection(name, cls):
    if name in _SELECTIONS:
        raise ValueError(f"Duplicate selection registration for '{name}'")
    _SELECTIONS[name] = cls


def register_optimizer(name, cls):
    if name in _OPTIMIZERS:
        raise ValueError(f"Duplicate optimizer registration for '{name}'")
    _OPTIMIZERS[name] = cls


def register_backend_plugin(name, cls):
    if name in _BACKENDS:
        raise ValueError(f"Duplicate backend registration for '{name}'")
    _BACKENDS[name] = cls


def register_default_plugins() -> None:
    """Idempotent per-name fill of the built-in classes (setdefault, so a
    test that pre-registers a custom plugin under a default name first is
    never silently clobbered)."""
    from sga.plugins.selection.topk import TopkSelection
    from sga.plugins.optimizer.adamw import SgaAdamW
    from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
    from sga.backends.torch_pipelined_mask import TorchPipelinedMaskBackend

    _SELECTIONS.setdefault("topk", TopkSelection)
    _OPTIMIZERS.setdefault("adamw", SgaAdamW)
    _BACKENDS.setdefault("torch_pipelined_index", TorchPipelinedIndexBackend)
    _BACKENDS.setdefault("torch_pipelined_mask", TorchPipelinedMaskBackend)
    _BACKENDS.setdefault("nccl_pipelined_index", NcclPipelinedIndexBackend)


def plugin_class(axis: str, name: str):
    """Curated unknown-name failure: the error lists the registered names
    of that axis (an absorbed legacy name reads as plain unknown here)."""
    registry = _REGISTRIES[axis]
    register_default_plugins()
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown SGA {axis} plugin '{name}'. Registered: "
            f"{sorted(registry)}"
        ) from None


def build_selection(name):
    return plugin_class("selection", name)()


def build_optimizer(name):
    return plugin_class("optimizer", name)()


def build_backend_plugin(name):
    return plugin_class("backend", name)()
