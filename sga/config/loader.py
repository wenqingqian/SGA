"""Config validation for the single-layer SGA run yaml.

Structure (one file, no defaults/inheritance/merging):

    experiment: {exp_name, exp_dir, task, runner, cmds, envs}
    action: run|dryrun|test|stop|query
    sga: {score, selection, error_feedback, backend}   # algorithm section
    train: {system, model, data}                       # megatron args

The `sga:` section is validated here and never reaches the megatron command
line (the launcher flattens only train.{system,model,data}). The spec allows
exactly one combination: everything off + dense backend.
"""

from omegaconf import DictConfig, OmegaConf

_SGA_SUBFIELDS = ("score", "selection", "error_feedback")

_ENABLED_ERROR = (
    "SGA section '{field}.enabled' is true, but this build only supports "
    "dense training (all SGA subfields must be enabled: false)."
)


def validate_sga_section(config: DictConfig) -> None:
    """Validate config.sga; fail with the specific offending field."""
    sga = config.get("sga", None)
    if sga is None:
        raise ValueError(
            "Config is missing the 'sga:' section (declare it with score/"
            "selection/error_feedback disabled and backend: dense)."
        )
    for field in _SGA_SUBFIELDS:
        if bool(sga.get(field, {}).get("enabled", False)):
            raise ValueError(_ENABLED_ERROR.format(field=field))

    backend_name = sga.get("backend", {}).get("name", None)
    if backend_name is None:
        raise ValueError("sga.backend.name is required (registry-driven, e.g. 'dense').")
    # Late import: the registry is populated by explicit registration, and the
    # loader must not depend on plugin import order.
    from sga.backends import get_backend, register_default_backends

    register_default_backends()
    get_backend(backend_name)  # raises with the valid set on unknown names


def validate_config(config: DictConfig) -> None:
    """Validate the whole run config before any runner is constructed."""
    validate_sga_section(config)
    # `sga:` must not leak into the megatron arg flattening; the launcher only
    # reads train.{system,model,data}, but assert the boundary here so a future
    # flatten change cannot silently start forwarding it.
    flat_train = OmegaConf.to_container(config.train, resolve=True)
    assert "sga" not in flat_train, "sga section leaked into train.*"
