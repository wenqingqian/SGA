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

from sga.trace import parse_branch_points

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
        if sga.get("checks", None) is not None:
            raise ValueError(
                "sga.checks is set but sga.enabled is false; the consistency "
                "machinery only exists with an engine — remove the section."
            )
        if sga.get("trace", None) is not None:
            raise ValueError(
                "sga.trace is set but sga.enabled is false; trace collection "
                "is an engine-side service (dense replay-grade traces come "
                "from shadow runs) — remove the section."
            )
        for key in ("shadow", "shadow_score"):
            if sga.get(key, None) is not None:
                raise ValueError(
                    f"sga.{key} is set but sga.enabled is false; shadow mode "
                    "is an engine mode — remove the field."
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
    _resolve_cycle_start(engine_config["cycle"], config)

    # Engine consistency-machinery toggles (all default on; escape hatches
    # for research runs — e.g. count_guard=false restores the legacy
    # --x-skip-topk-count-check behavior).
    checks = sga.get("checks", None)
    if checks is not None:
        if not OmegaConf.is_dict(checks):
            raise ValueError(f"sga.checks must be a mapping (got: {checks!r}).")
        engine_config["checks"] = OmegaConf.to_container(checks, resolve=True)
        unknown = sorted(set(engine_config["checks"]) - {"count_guard"})
        if unknown:
            raise ValueError(
                f"sga.checks: unknown keys {unknown} (known: ['count_guard'])"
            )
        if not isinstance(engine_config["checks"].get("count_guard", True), bool):
            raise ValueError("sga.checks.count_guard must be a boolean.")

    # Engine-side trace collector (offline replay data source). Off unless
    # sga.trace.dir is set; validation/normalization happens HERE so the
    # collector sees a clean {dir, branch_points, window} dict.
    trace = sga.get("trace", None)
    if trace is not None:
        if not OmegaConf.is_dict(trace):
            raise ValueError(f"sga.trace must be a mapping (got: {trace!r}).")
        engine_config["trace"] = OmegaConf.to_container(trace, resolve=True)
        t = engine_config["trace"]
        unknown = sorted(set(t) - {"dir", "branch_points", "window"})
        if unknown:
            raise ValueError(
                f"sga.trace: unknown keys {unknown} "
                "(known: ['branch_points', 'dir', 'window'])"
            )
        if not isinstance(t.get("dir"), str) or not t["dir"]:
            raise ValueError(
                "sga.trace.dir is required and must be a non-empty string "
                "(the trace output directory)."
            )
        try:
            t["branch_points"] = parse_branch_points(t.get("branch_points"))
        except (TypeError, ValueError) as e:
            raise ValueError(f"sga.trace.branch_points: {e}") from None
        window = t.get("window", 100)
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(f"sga.trace.window must be an int >= 1 (got {window!r}).")
        t["window"] = window

    # Shadow mode: stock-step literal replication with read-only boundary
    # scoring (dense numerics invariant). shadow_score=false strips even the
    # read-only scoring — the stock-replication probe configuration.
    shadow = sga.get("shadow", None)
    if shadow is not None:
        if not isinstance(shadow, bool):
            raise ValueError(f"sga.shadow must be a boolean (got: {shadow!r}).")
        # Explicit false means plain live: no shadow key reaches the engine
        # config (consumers key off presence, absent == false).
        if shadow:
            engine_config["shadow"] = shadow
    shadow_score = sga.get("shadow_score", None)
    if shadow_score is not None:
        if engine_config.get("shadow") is not True:
            raise ValueError(
                "sga.shadow_score is set but sga.shadow is not true; it only "
                "modifies shadow runs."
            )
        if not isinstance(shadow_score, bool):
            raise ValueError(
                f"sga.shadow_score must be a boolean (got: {shadow_score!r})."
            )
        engine_config["shadow_score"] = shadow_score

    _validate_against_registries(engine_config)
    return engine_config


def _resolve_cycle_start(cycle: dict, config: DictConfig) -> None:
    """Resolve ``cycle.start_iter_percent`` (of train.model.train_iters) into
    the absolute ``start_iter`` the engine speaks — here, BEFORE the section
    is serialized into the canonical ``--sga-config`` JSON (which only ever
    carries start_iter). The two forms are mutually exclusive."""
    if "start_iter_percent" not in cycle:
        return
    if cycle.get("start_iter") is not None:
        raise ValueError(
            "sga.cycle: set either start_iter or start_iter_percent, not both."
        )
    train_iters = OmegaConf.select(config, "train.model.train_iters", default=None)
    if not train_iters or int(train_iters) <= 0:
        raise ValueError(
            "sga.cycle.start_iter_percent requires train.model.train_iters > 0."
        )
    pct_raw = cycle.pop("start_iter_percent")
    # An explicit null must fail with the range diagnostic below, not a
    # TypeError from float(None) inside the comparison.
    if pct_raw is None:
        raise ValueError(
            "sga.cycle.start_iter_percent must be a number in [0, 100] "
            "(got null)."
        )
    pct = float(pct_raw)
    if not 0.0 <= pct <= 100.0:
        raise ValueError(
            f"sga.cycle.start_iter_percent must be in [0, 100] (got {pct})."
        )
    cycle["start_iter"] = max(0, int(round(pct / 100.0 * int(train_iters))))


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
