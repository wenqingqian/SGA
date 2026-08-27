"""Config validation for the SGA run yaml.

Structure (one file, no defaults/inheritance/merging):

    experiment: {exp_name, exp_dir, task, runner, cmds, envs}
    action: run|dryrun|test|stop|query
    sga: {enabled, cycle, score, selection, error_feedback, backend}
    train: {system, model, data}                       # megatron args

The `sga:` section is validated here and never reaches the megatron command
line through the flatten (the launcher flattens only train.{system,model,data}
and instead injects the resolved section as the single opaque ``--sga-config``
canonical-JSON argument — notes/sga-waist-api-v1.md section 7).

the four-field model: a scheme is the four-field combination
score/selection/error_feedback/backend, each with its own params, plus the
engine cycle {start_iter, period, full_sync_interval}. Legality is checked
from plugin DECLARATIONS (sga.plugins.validate_combination) — no hardcoded
scheme list. ``enabled: false`` keeps the dense-off semantics: dense training,
backend name must be 'dense'.
"""

from omegaconf import DictConfig, OmegaConf

_AXES = ("score", "selection", "error_feedback", "backend")


def sga_section_to_config(config: DictConfig):
    """Normalize config.sga into the engine config dict, or None when disabled.

    Raises ValueError with the specific offending field on any structural
    problem; unknown plugin names fail against the registries.
    """
    sga = config.get("sga", None)
    if sga is None:
        raise ValueError(
            "Config is missing the 'sga:' section (declare it with enabled: "
            "false and backend: dense for pure dense training)."
        )

    enabled = bool(sga.get("enabled", False))

    if not enabled:
        # dense-off semantics: dense training, nothing else may be configured.
        for axis in _AXES:
            sub = sga.get(axis, None)
            if sub is None:
                continue
            if not OmegaConf.is_dict(sub):
                raise ValueError(
                    f"sga.{axis} must be a mapping (got: {sub!r})."
                )
            if sub.get("name", None) not in (None, "dense"):
                raise ValueError(
                    f"sga.{axis}.name is set but sga.enabled is false; either "
                    "enable SGA or remove the plugin configuration."
                )
        backend_cfg = sga.get("backend", None)
        name = None
        if backend_cfg is not None and OmegaConf.is_dict(backend_cfg):
            name = backend_cfg.get("name", None)
        if name not in (None, "dense"):
            raise ValueError(
                f"sga.enabled is false but backend.name is '{name}'; dense-only "
                "runs must use backend: dense (or omit the backend block)."
            )
        return None

    engine_config = {}
    for axis in _AXES:
        sub = sga.get(axis, None)
        if sub is None or not OmegaConf.is_dict(sub):
            raise ValueError(
                f"sga.{axis} is required when sga.enabled is true and must be "
                f"a mapping with a 'name' key (got: {sub!r})."
            )
        name = sub.get("name", None)
        if name is None:
            raise ValueError(f"sga.{axis}.name is required (registry-driven).")
        params = OmegaConf.to_container(sub, resolve=True)
        params.pop("name", None)
        engine_config[axis] = {"name": name, **params}

    if engine_config["backend"]["name"] == "dense":
        raise ValueError(
            "sga.enabled is true but backend is 'dense' — dense is the "
            "SGA-off representation; pick a sparse backend."
        )

    cycle = sga.get("cycle", None)
    engine_config["cycle"] = (
        OmegaConf.to_container(cycle, resolve=True) if cycle is not None else {}
    )

    _validate_against_registries(engine_config)
    return engine_config


def _validate_against_registries(engine_config: dict) -> None:
    """Registry membership + declaration-based combination check."""
    from sga.plugins import (
        get_backend_plugin_decl,
        get_ef_decl,
        get_score_decl,
        get_selection_decl,
        register_default_plugins,
        validate_combination,
    )

    register_default_plugins()
    score = get_score_decl(engine_config["score"]["name"])
    selection = get_selection_decl(engine_config["selection"]["name"])
    ef = get_ef_decl(engine_config["error_feedback"]["name"])
    backend = get_backend_plugin_decl(engine_config["backend"]["name"])
    cycle = engine_config.get("cycle", {})
    if int(cycle.get("period", 1)) < 1:
        raise ValueError("sga.cycle.period must be >= 1")
    validate_combination(score, selection, ef, backend, cycle)


def validate_sga_section(config: DictConfig) -> None:
    """Validate config.sga; fail with the specific offending field."""
    sga_section_to_config(config)


def validate_config(config: DictConfig) -> None:
    """Validate the whole run config before any runner is constructed."""
    validate_sga_section(config)
    # `sga:` must not leak into the megatron arg flattening; the launcher only
    # reads train.{system,model,data}, but assert the boundary here so a future
    # flatten change cannot silently start forwarding it.
    flat_train = OmegaConf.to_container(config.train, resolve=True)
    assert "sga" not in flat_train, "sga section leaked into train.*"
