"""Config validation for the SGA run yaml.

Structure (one file, no defaults/inheritance/merging):

    experiment: {exp_name, exp_dir, task, runner, cmds, envs}
    action: run|dryrun|test|stop|query
    sga: {enabled, cycle, score, selection, optimizer, backend}
    train: {system, model, data}                       # megatron args

The `sga:` section is validated here and never reaches the megatron command
line through the flatten. The launcher translates it into the FLAT flag
vocabulary (``--sga-*``; sga/config/arguments.py — the single parsing
pipeline: the flags are registered into megatron's argparse by the training
entry and re-assembled into the canonical ``args.sga_config`` JSON) —
notes/sga-waist-api-v1.md section 7 keeps the waist reading only that JSON.

the four-field model: a scheme is the four-field combination
score/selection/optimizer/backend, each with its own params, plus the engine
cycle {start_iter, period, full_sync_interval}. Legality is checked from
plugin DECLARATIONS (sga.plugins.validate_combination) — no hardcoded
scheme list — and each plugin's parameter vocabulary is declared on its
decl (unknown keys fail loud). ``enabled: false`` keeps the an earlier round
semantics: dense training, backend name must be 'dense'.
"""

from omegaconf import DictConfig, OmegaConf

from sga.trace import parse_branch_points

_AXES = ("score", "selection", "optimizer", "backend")

# Top-level sga: keys (single list, mode-independent: a typo like `shadw`
# must fail loud even in a disabled config, where it would otherwise be
# silently dropped and the run would proceed as live sparse instead of the
# intended shadow). Plugin params inside score/selection/optimizer/backend
# are validated against the plugins' declared vocabularies; cycle's own
# interior is a fixed vocabulary.
_SGA_KEYS = frozenset(
    (
        "enabled",
        "cycle",
        "score",
        "selection",
        "optimizer",
        "backend",
        "checks",
        "trace",
        "shadow",
        "shadow_score",
    )
)
_CYCLE_KEYS = frozenset(
    ("start_iter", "start_iter_percent", "period", "full_sync_interval")
)


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
    train_iters = OmegaConf.select(config, "train.model.train_iters",
                                   default=None)
    return engine_config_from_sga_dict(OmegaConf.to_container(sga, resolve=True),
                                       train_iters=train_iters)


def engine_config_from_sga_dict(sga: dict, train_iters=None):
    """The shared validation core, on a plain dict (both input forms meet
    here): the nested yaml section (via sga_section_to_config) and the flat
    flags re-nested by the assembler (sga/config/arguments.py). ``train_iters``
    is the percent-resolution context; the flat-flag path passes None
    (start_iter_percent has no flag and is rejected there).
    """
    if not isinstance(sga, dict):
        raise ValueError(f"sga section must be a mapping (got: {sga!r}).")

    unknown = sorted(set(sga.keys()) - _SGA_KEYS)
    if unknown:
        raise ValueError(
            f"sga: unknown keys {unknown} (known: {sorted(_SGA_KEYS)})"
        )

    enabled = bool(sga.get("enabled", False))

    if not enabled:
        # dense-off semantics: dense training, nothing else may be configured.
        for axis in _AXES:
            sub = sga.get(axis, None)
            if sub is None:
                continue
            if not isinstance(sub, dict):
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
        if backend_cfg is not None and isinstance(backend_cfg, dict):
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
        if sub is None or not isinstance(sub, dict):
            raise ValueError(
                f"sga.{axis} is required when sga.enabled is true and must be "
                f"a mapping with a 'name' key (got: {sub!r})."
            )
        name = sub.get("name", None)
        if name is None:
            raise ValueError(f"sga.{axis}.name is required (registry-driven).")
        params = dict(sub)
        params.pop("name", None)
        engine_config[axis] = {"name": name, **params}

    if engine_config["backend"]["name"] == "dense":
        raise ValueError(
            "sga.enabled is true but backend is 'dense' — dense is the "
            "SGA-off representation; pick a sparse backend."
        )

    cycle = sga.get("cycle", None)
    if cycle is not None:
        if not isinstance(cycle, dict):
            raise ValueError(f"sga.cycle must be a mapping (got: {cycle!r}).")
        engine_config["cycle"] = dict(cycle)
        unknown = sorted(set(engine_config["cycle"]) - _CYCLE_KEYS)
        if unknown:
            # A typo like `peroid` would otherwise silently fall back to the
            # period default of 1 — reselection every step instead of the
            # configured cycle, with no diagnostic anywhere.
            raise ValueError(
                f"sga.cycle: unknown keys {unknown} "
                f"(known: {sorted(_CYCLE_KEYS)})"
            )
    else:
        engine_config["cycle"] = {}
    _resolve_cycle_start(engine_config["cycle"], train_iters)

    # Engine consistency-machinery toggles (all default on; escape hatches
    # for research runs — e.g. count_guard=false restores the legacy
    # --x-skip-topk-count-check behavior).
    checks = sga.get("checks", None)
    if checks is not None:
        if not isinstance(checks, dict):
            raise ValueError(f"sga.checks must be a mapping (got: {checks!r}).")
        engine_config["checks"] = dict(checks)
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
        if not isinstance(trace, dict):
            raise ValueError(f"sga.trace must be a mapping (got: {trace!r}).")
        engine_config["trace"] = dict(trace)
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


def _resolve_cycle_start(cycle: dict, train_iters) -> None:
    """Resolve ``cycle.start_iter_percent`` (of train.model.train_iters) into
    the absolute ``start_iter`` the engine speaks — here, BEFORE the section
    is serialized into the canonical ``args.sga_config`` JSON (which only
    ever carries start_iter). The two forms are mutually exclusive."""
    if "start_iter_percent" not in cycle:
        return
    if cycle.get("start_iter") is not None:
        raise ValueError(
            "sga.cycle: set either start_iter or start_iter_percent, not both."
        )
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
    """Registry membership + declaration-based combination check + the
    plugins' declared parameter vocabularies (unknown keys fail loud here —
    a typo'd param would otherwise be silently swallowed downstream)."""
    from sga.plugins import (
        check_plugin_params,
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_score_spec,
        get_selection_decl,
        register_default_plugins,
        validate_combination,
    )

    register_default_plugins()
    score_name = engine_config["score"]["name"]
    selection = get_selection_decl(engine_config["selection"]["name"])
    optimizer = get_optimizer_decl(engine_config["optimizer"]["name"])
    backend = get_backend_plugin_decl(engine_config["backend"]["name"])
    cycle = engine_config.get("cycle", {})
    if int(cycle.get("period", 1)) < 1:
        raise ValueError("sga.cycle.period must be >= 1")
    validate_combination(score_name, selection, optimizer, backend, cycle)

    # Scores are registry DATA with no tunable params: any key besides the
    # name is a typo or a misunderstanding of the axis.
    extra_score_keys = sorted(set(engine_config["score"]) - {"name"})
    if extra_score_keys:
        raise ValueError(
            f"sga.score takes no params beyond name (got: {extra_score_keys})"
        )
    for axis, decl in (("selection", selection),
                       ("optimizer", optimizer),
                       ("backend", backend)):
        params = {k: v for k, v in engine_config[axis].items() if k != "name"}
        check_plugin_params(axis, engine_config[axis]["name"], decl, params)
        # Canonicalize float params to float regardless of the yaml spelling
        # (10 vs 10.0): the canonical --sga-config JSON must be
        # spelling-invariant, or the resume guard would refuse identical
        # configs over a byte difference.
        for p in decl.params:
            if p.type is float and isinstance(params.get(p.name), int):
                engine_config[axis][p.name] = float(engine_config[axis][p.name])


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
