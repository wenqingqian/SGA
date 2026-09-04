# Copyright (c) 2026, SGA framework contributors.
"""The flat SGA config validation core.

Everything a config may say lives in ``sga/config/arguments.py`` (plain
``add_argument`` declarations); THIS module derives the vocabulary from
that file by parser introspection — no key set, default or choice is
restated anywhere — and adds the context argparse cannot express:

  - unknown-key rejection (a typo'd ``sga_`` key fails loud, never
    silently dense),
  - the enabled/disabled contract (disabled = everything else default),
  - axis-identity requirements and plugin-name membership (registries),
  - per-plugin parameter attribution with equality guards ("belongs to
    method=histogram"), preserving the historical foreign-key errors,
  - ``sga_cycle_start_iter_percent`` resolution against train_iters,
  - trace normalization, shadow constraints, float canonicalization,
  - the cross-axis combination check (``sga.plugins.validate_combination``).

Inputs are EXPLICIT-key dicts (only what the user actually wrote): the
launcher passes the yaml ``sga:`` section verbatim; the training side
passes ``explicit_from_args(args)`` (the argparse-defaults diff — with
real defaults filled in by argparse, "flag absent" is indistinguishable
from "flag at default", and the two are semantically identical anyway).

Output: ``(engine_config, canonical)`` where ``engine_config`` is the
internal NESTED engine dict (section defaults materialized; plugin axes
carry explicit params only — constructors expand defaults through
``expand_plugin_params``) and ``canonical`` is the complete typed FLAT
dict (percent resolved away) used for the engine's config print, the
resume guard comparison and ``python -m sga.config resolve``.

Disabled-clean input returns ``(None, None)``.
"""

import argparse
import json
import math

from sga.config.arguments import add_sga_args, sga_bool, sga_int_or_null

# The only plugin-vocabulary data outside arguments.py: which flat
# suffixes belong to which plugin of an axis, and each guarded param's
# equality condition (sibling param == value). Types/defaults/choices
# are NOT here — they are introspected from the declarations. Convention:
# a param whose declared default is None is REQUIRED (required-ness is
# inferred from default-None); a future optional param needs a
# table/annotation update before it can exist.
_AXIS_PARAMS = {
    "selection": {
        "topk": (
            ("method", None),
            ("ratio_percent", None),
            ("histogram_mode", ("method", "histogram")),
            ("adaptive_warmup_syncs", ("method", "adaptive")),
            ("adaptive_warmup_mode", ("method", "adaptive")),
        ),
    },
    "optimizer": {
        "adamw": (
            ("residual_mean", None),
            ("merge_join_dir", None),
        ),
    },
    "backend": {
        "torch_pipelined_index": (
            ("high_priority_group", None),
            ("slices", None),
        ),
        "torch_pipelined_mask": (
            ("num_slices", None),
        ),
        "nccl_pipelined_index": (),
    },
}

_PARSER = None


def _parser() -> argparse.ArgumentParser:
    """A throwaway parser carrying the full SGA vocabulary."""
    global _PARSER
    if _PARSER is None:
        _PARSER = argparse.ArgumentParser(add_help=False)
        add_sga_args(_PARSER)
    return _PARSER


def sga_actions() -> dict:
    """dest -> argparse action (type/default/choices/help) for the SGA flags."""
    return {a.dest: a for a in _parser()._actions if a.dest.startswith("sga_")}


def sga_defaults() -> dict:
    """dest -> declared default (the single source; nothing restates it)."""
    return {dest: action.default for dest, action in sga_actions().items()}


def declared_choices(dest: str) -> tuple:
    """The argparse-declared choices of one flag (empty when open-ended)."""
    return tuple(getattr(sga_actions()[dest], "choices", None) or ())


def engine_section_defaults(section: str) -> dict:
    """{key: default} for one engine section (``cycle``/``checks``/``trace``,
    prefix stripped) — the fallback single source for engine-side reads."""
    prefix = f"sga_{section}_"
    # cycle picks up the vestigial start_iter_percent key (None default):
    # harmless parity — the canonical form resolves it away and no engine
    # read consumes it.
    return {dest[len(prefix):]: action.default
            for dest, action in sga_actions().items() if dest.startswith(prefix)}


def axis_param_defaults(axis: str, plugin_name: str) -> dict:
    """{suffix: default} for one plugin's params (guarded params included);
    the score axis (no params) yields {}."""
    defaults = sga_defaults()
    return {suffix: defaults[f"sga_{axis}_{suffix}"]
            for suffix, _ in _AXIS_PARAMS.get(axis, {}).get(plugin_name, ())}


def expand_plugin_params(axis: str, plugin_name: str, params: dict) -> dict:
    """Validate an explicit params dict and expand it to full defaults —
    the constructors' single read path (defaults come from the
    declarations, never from a restated ``params.get(name, default)``).
    """
    rows = _AXIS_PARAMS.get(axis, {}).get(plugin_name)
    if rows is None:
        # a registered plugin without attribution rows would otherwise die
        # as a bare KeyError far from the cause (INV-1 pins the sync, this
        # keeps the runtime failure actionable)
        raise ValueError(
            f"sga.{axis} plugin '{plugin_name}' lacks an attribution row in "
            "_AXIS_PARAMS (validation.py); update the table when registering "
            "plugins"
        )
    known = {suffix for suffix, _ in rows}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(
            f"sga.{axis} ({plugin_name}): unknown params {unknown} "
            f"(known: {sorted(known)})"
        )
    defaults = axis_param_defaults(axis, plugin_name)
    for suffix in sorted(known):
        if defaults[suffix] is None and suffix not in params:
            raise ValueError(
                f"sga.{axis} ({plugin_name}): param '{suffix}' is "
                "required"
            )
    out = dict(defaults)
    out.update(params)
    for suffix in params:
        _check_value(axis, plugin_name, f"sga_{axis}_{suffix}", suffix, out[suffix])
    return out


def guard_foreign_params(axis: str, plugin_name: str, params: dict) -> tuple:
    """(foreign, own) under the equality guards, on an EXPLICIT
    (pre-expansion) params dict: ``foreign`` are the present params whose
    guard the dict does not satisfy (knobs of another sibling value —
    silently dead where they are); ``own`` are the guarded params the
    sibling value does unlock (for the refusal message)."""
    guarded = [(suffix, requires) for suffix, requires
               in _AXIS_PARAMS[axis][plugin_name] if requires]
    foreign = sorted(suffix for suffix, (other, value) in guarded
                     if suffix in params and params.get(other) != value)
    own = sorted(suffix for suffix, (other, value) in guarded
                 if params.get(other) == value)
    return foreign, own


def _param_kind(dest: str):
    """'bool' | 'int' | 'int_or_null' | 'float' | 'str' for one dest."""
    action = sga_actions()[dest]
    if action.type is sga_bool:
        return "bool"
    if action.type is sga_int_or_null:
        return "int_or_null"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "str"


def _check_value(axis, plugin_name, dest, suffix, val) -> None:
    """Validate one (param, value) pair: nullability, type, choices."""
    kind = _param_kind(dest)
    actions = sga_actions()
    if val is None:
        if kind == "int_or_null":
            return
        raise ValueError(
            f"sga.{axis} ({plugin_name}): param '{suffix}' is not "
            "nullable (got null)"
        )
    if kind == "bool":
        ok = isinstance(val, bool)
        typ = "bool"
    elif kind in ("int", "int_or_null"):
        # bool is an int subclass; yaml true/false must not pass as 1/0
        ok = isinstance(val, int) and not isinstance(val, bool)
        typ = "int"
    elif kind == "float":
        ok = isinstance(val, (int, float)) and not isinstance(val, bool)
        typ = "float"
    else:
        ok = isinstance(val, str)
        typ = "str"
    if not ok:
        raise ValueError(
            f"sga.{axis} ({plugin_name}): param '{suffix}' must be "
            f"{typ} (got {val!r})"
        )
    choices = tuple(getattr(actions[dest], "choices", None) or ())
    if choices and val not in choices:
        raise ValueError(
            f"sga.{axis} ({plugin_name}): param '{suffix}' must be one "
            f"of {list(choices)} (got {val!r})"
        )


def _coerce(dest: str, val):
    """One explicit value -> typed (accepts yaml spellings AND the
    argparse-typed namespace values; canonicalizes floats; enforces
    argparse-declared choices). Wrong-typed values fail loud: only the
    yaml route feeds raw user values through here (the CLI route is
    typed by argparse first), so a bare ``true``/``8.0``/``"8"`` must
    never silently become 1/8/8 — bools are rejected explicitly (bool is
    an int subclass), the int kinds accept only ``type(val) is int`` and
    float dests must be finite."""
    kind = _param_kind(dest)
    if kind == "bool":
        if isinstance(val, bool):
            return val
        t = str(val).strip().lower()
        if t == "true":
            return True
        if t == "false":
            return False
        raise ValueError(f"{dest} must be a valid bool (got {val!r})")
    if kind == "int_or_null":
        if val is None:
            return None
        if isinstance(val, str) and val.strip().lower() == "null":
            return None
        if type(val) is not int:
            raise ValueError(
                f"{dest} must be a valid int_or_null (got {val!r})")
        return val
    if kind == "int":
        if type(val) is not int:
            raise ValueError(f"{dest} must be a valid int (got {val!r})")
        return val
    if kind == "float":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{dest} must be a valid float (got {val!r})")
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"{dest} must be a finite float (got {val!r})")
        return fval
    if isinstance(val, (list, tuple)):
        # the yaml spelling of the one list-kind flag (branch_points
        # rides the command line as a comma token; yaml may give a list)
        if dest == "sga_trace_branch_points":
            return list(val)
        raise ValueError(f"{dest} must be a string (got {val!r})")
    if not isinstance(val, str):
        # str-kind values arrive typed already; anything else is a shape bug
        raise ValueError(f"{dest} must be a string (got {val!r})")
    choices = tuple(getattr(sga_actions()[dest], "choices", None) or ())
    if choices and val not in choices:
        raise ValueError(
            f"{dest} must be one of {list(choices)} (got {val!r})")
    return val


def explicit_from_flat(flat: dict) -> dict:
    """A complete raw flat dict (e.g. the waist's namespace collection,
    argparse defaults included) -> the explicit diff. A value equal to
    its default AND of the same type is semantically identical to an
    absent flag; the exceptions survive as explicit: an explicit null on
    a param with a real default (sga_int_or_null), and a wrong-typed
    look-alike of a cross-type default (0 == False must not read as
    absent)."""
    defaults = sga_defaults()
    out = {}
    for dest, default in defaults.items():
        if dest not in flat:
            continue
        value = flat[dest]
        if (value == default and type(value) is type(default)
                and not (value is None and default is not None)):
            continue
        out[dest] = value
    return out


def explicit_from_args(args) -> dict:
    """The explicit SGA keys of a parsed namespace: the argparse-defaults
    diff (a parser without the SGA provider contributes nothing — stock
    entries stay dense-clean)."""
    present = {dest: getattr(args, dest)
               for dest in sga_defaults() if hasattr(args, dest)}
    return explicit_from_flat(present)


# sga_config is the retired v1 JSON-transport key: it still rides old
# checkpoints' saved namespaces and must never masquerade as config.
_NON_CONFIG_SGA_KEYS = ("sga_allow_config_override", "sga_config")


def collect_sga_flat(args) -> dict:
    """Every sga_* attribute of a namespace, RAW (fork-side collection
    twin — used SGA-side by the resume guard to re-read checkpointed
    namespaces)."""
    return {key: value for key, value in vars(args).items()
            if key.startswith("sga_") and key not in _NON_CONFIG_SGA_KEYS}


def canonical_flat_json(canonical: dict) -> str:
    """Canonical (sorted-key) serialization of a validated flat dict."""
    return json.dumps(canonical, sort_keys=True)


def validate_sga_flat(flat: dict, train_iters=None):
    """Validate an explicit-key flat dict -> (engine_config, canonical),
    or (None, None) when disabled-clean. Raises ValueError with the
    specific offending key on any problem."""
    actions = sga_actions()
    unknown = sorted(set(flat) - set(actions))
    if unknown:
        raise ValueError(
            f"sga: unknown keys {unknown} (known: {sorted(actions)})"
        )
    typed = {dest: _coerce(dest, val) for dest, val in flat.items()}
    defaults = sga_defaults()

    if typed.get("sga_enabled", defaults["sga_enabled"]) is not True:
        # Disabled semantics: dense training; nothing else may be
        # configured (an explicit sga_backend: dense is the documented
        # dense spelling and passes).
        for dest in sorted(typed):
            if dest == "sga_enabled":
                continue
            if dest == "sga_backend" and typed[dest] == "dense":
                continue
            raise ValueError(
                f"{dest} is set but sga_enabled is false; either "
                "enable SGA or remove the flag"
            )
        return None, None

    for dest in ("sga_score", "sga_optimizer", "sga_backend"):
        if typed.get(dest) is None:
            raise ValueError(
                f"{dest} is required when sga_enabled is true "
                "(registered names are the choices)"
            )
    if typed["sga_backend"] == "dense":
        raise ValueError(
            "sga_enabled is true but sga_backend is 'dense' — dense is "
            "the SGA-off representation; pick a sparse backend."
        )

    # --- cycle (incl. percent resolution and the decouple calendar) ---
    if typed.get("sga_cycle_start_iter_percent") is not None:
        if "sga_cycle_start_iter" in typed:
            raise ValueError(
                "sga_cycle: set either sga_cycle_start_iter or "
                "sga_cycle_start_iter_percent, not both."
            )
        if not train_iters or int(train_iters) <= 0:
            raise ValueError(
                "sga_cycle_start_iter_percent requires train_iters > 0."
            )
        pct = float(typed["sga_cycle_start_iter_percent"])
        if not 0.0 <= pct <= 100.0:
            raise ValueError(
                "sga_cycle_start_iter_percent must be in [0, 100] "
                f"(got {pct})."
            )
        start_iter = max(0, int(round(pct / 100.0 * int(train_iters))))
    else:
        start_iter = int(typed.get("sga_cycle_start_iter",
                                   defaults["sga_cycle_start_iter"]))
    period = int(typed.get("sga_cycle_period", defaults["sga_cycle_period"]))
    if period < 1:
        raise ValueError("sga_cycle_period must be >= 1")
    decouple = bool(typed.get(
        "sga_cycle_decouple_full_sync_interval_and_period",
        defaults["sga_cycle_decouple_full_sync_interval_and_period"]))
    interval = typed.get("sga_cycle_full_sync_interval",
                         defaults["sga_cycle_full_sync_interval"])
    if not decouple:
        if interval is not None:
            raise ValueError(
                f"sga_cycle_full_sync_interval is set ({interval}) but "
                "sga_cycle_decouple_full_sync_interval_and_period is "
                "false; every cycle boundary is already a full sync — "
                "leave sga_cycle_full_sync_interval unset"
            )
    elif interval is None:
        raise ValueError(
            "sga_cycle_decouple_full_sync_interval_and_period is true "
            "but sga_cycle_full_sync_interval is unset — set a divisor "
            f"of the period smaller than it (period={period})"
        )
    elif not (0 < int(interval) < period and period % int(interval) == 0):
        raise ValueError(
            f"decoupled sga_cycle_full_sync_interval ({interval}) must "
            f"satisfy 0 < full_sync_interval < period ({period}) and "
            "period % full_sync_interval == 0"
        )
    engine_cycle = {
        "start_iter": start_iter,
        "period": period,
        "decouple_full_sync_interval_and_period": decouple,
        "full_sync_interval": None if interval is None else int(interval),
    }

    # --- checks ---
    engine_checks = {"count_guard": bool(typed.get(
        "sga_checks_count_guard", defaults["sga_checks_count_guard"]))}

    # --- trace (setting any trace key activates the collector) ---
    trace_present = any(dest in typed for dest in actions
                        if dest.startswith("sga_trace_"))
    engine_trace = None
    if trace_present:
        tdir = typed.get("sga_trace_dir")
        if not isinstance(tdir, str) or not tdir:
            raise ValueError(
                "sga_trace_dir is required and must be a non-empty string "
                "(the trace output directory)."
            )
        # Lazy import: sga.trace -> sga.config (package __init__) -> this
        # module -> sga.trace is a cycle; sga.trace-first processes must
        # keep importing (only call site, so the deferral costs nothing).
        from sga.trace import parse_branch_points
        try:
            branch = parse_branch_points(
                typed.get("sga_trace_branch_points",
                          defaults["sga_trace_branch_points"]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"sga_trace_branch_points: {e}") from None
        window = typed.get("sga_trace_window", defaults["sga_trace_window"])
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(
                f"sga_trace_window must be an int >= 1 (got {window!r}).")
        engine_trace = {"dir": tdir, "branch_points": branch,
                        "window": int(window)}

    # --- shadow ---
    shadow = bool(typed.get("sga_shadow", defaults["sga_shadow"]))
    if "sga_shadow_score" in typed and not shadow:
        raise ValueError(
            "sga_shadow_score is set but sga_shadow is not true; it only "
            "modifies shadow runs."
        )
    shadow_score = bool(typed.get("sga_shadow_score",
                                  defaults["sga_shadow_score"]))

    # --- plugin axes: registry membership, attribution, guards, float
    # canonicalization ---
    from sga.plugins import (
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_score_spec,
        get_selection_decl,
        register_default_plugins,
        validate_combination,
    )

    register_default_plugins()
    axis_names = {
        "score": typed["sga_score"],
        "selection": typed.get("sga_selection", defaults["sga_selection"]),
        "optimizer": typed["sga_optimizer"],
        "backend": typed["sga_backend"],
    }
    decls = {}
    axes_cfg = {}
    for axis, name in axis_names.items():
        # Registry membership BEFORE the per-plugin vocab check: an unknown
        # plugin name must surface as the registry's curated error (rename
        # hints included), never as "unknown params (known: [])" against
        # the unknown plugin's empty vocabulary.
        if axis == "score":
            decls[axis] = get_score_spec(name)
        elif axis == "selection":
            decls[axis] = get_selection_decl(name)
        elif axis == "optimizer":
            decls[axis] = get_optimizer_decl(name)
        else:
            decls[axis] = get_backend_plugin_decl(name)
        prefix = f"sga_{axis}_"
        params = {dest[len(prefix):]: typed[dest]
                  for dest in typed if dest.startswith(prefix)}
        union = {suffix for plugin in _AXIS_PARAMS.get(axis, {}).values()
                 for suffix, _ in plugin}
        unknown = sorted(set(params) - union)
        if unknown:
            raise ValueError(
                f"sga.{axis}: unknown params {unknown} "
                f"(known: {sorted(union)})"
            )
        vocab = {suffix for suffix, _ in
                  _AXIS_PARAMS.get(axis, {}).get(name, ())}
        notours = sorted(set(params) - vocab)
        if notours:
            raise ValueError(
                f"sga.{axis} ({name}): unknown params {notours} "
                f"(known: {sorted(vocab)})"
            )
        # Required-before-guards (the 0024 C1 lesson): a missing sibling
        # must surface as the required error, never as a foreign-key
        # refusal naming the raw value.
        plugin_defaults = axis_param_defaults(axis, name)
        for suffix in sorted(vocab):
            if plugin_defaults[suffix] is None and suffix not in params:
                raise ValueError(
                    f"sga.{axis} ({name}): param '{suffix}' is required"
                )
        guarded = [(suffix, requires) for suffix, requires
                   in _AXIS_PARAMS.get(axis, {}).get(name, ()) if requires]
        if guarded:
            sibling = guarded[0][1][0]
            sib_val = params.get(
                sibling, typed.get(prefix + sibling,
                                   defaults.get(prefix + sibling)))
            foreign, own = guard_foreign_params(axis, name, params)
            if foreign:
                raise ValueError(
                    f"sga.{axis} ({name}): params {foreign} do not belong "
                    f"to {sibling} '{sib_val}' (it takes "
                    f"{own or 'no extra params'}) — under this {sibling} "
                    "they would be silently dead; drop them or switch "
                    f"the {sibling}"
                )
        # Canonicalize float params regardless of the yaml spelling
        # (10 vs 10.0): the canonical print/resume form must be
        # spelling-invariant.
        for suffix in params:
            if _param_kind(prefix + suffix) == "float":
                params[suffix] = float(params[suffix])
        axes_cfg[axis] = {"name": name, **params}

    # --- cross-axis combination ---
    validate_combination(axis_names["score"], decls["selection"],
                         decls["optimizer"], decls["backend"])

    engine_config = {
        "cycle": engine_cycle,
        "checks": engine_checks,
        "shadow": shadow,
        "shadow_score": shadow_score,
        **({"trace": engine_trace} if engine_trace is not None else {}),
        **axes_cfg,
    }

    # --- canonical flat dict: every dest at its resolved value ---
    canonical = dict(defaults)
    canonical.update(typed)
    canonical["sga_enabled"] = True
    canonical.pop("sga_cycle_start_iter_percent", None)  # resolved away
    canonical["sga_cycle_start_iter"] = start_iter
    if engine_trace is not None:
        canonical["sga_trace_branch_points"] = engine_trace["branch_points"]
    return engine_config, canonical


def validate_sga_args(args) -> dict:
    """Training-side fail-fast entry: validate the parsed namespace's SGA
    configuration (also when disabled — the disabled-cleanliness contract
    is enforced here). Returns the engine config or None when disabled."""
    return validate_sga_flat(explicit_from_args(args),
                             train_iters=getattr(args, "train_iters", None))[0]


def validate_run_config(config):
    """Launcher-side entry on the whole run config: the ``sga:`` section
    (flat keys, validated fail-fast before any process is spawned) plus
    the train-boundary assert. Returns (engine_config, canonical)."""
    sga = config.get("sga", None)
    if sga is None:
        raise ValueError(
            "Config is missing the 'sga:' section (declare it with "
            "sga_enabled: false and sga_backend: dense for pure dense "
            "training)."
        )
    from omegaconf import DictConfig, OmegaConf

    if isinstance(sga, DictConfig):
        sga = OmegaConf.to_container(sga, resolve=True)
    if not isinstance(sga, dict):
        raise ValueError(f"sga section must be a mapping (got: {sga!r}).")
    train_iters = OmegaConf.select(config, "train.model.train_iters",
                                   default=None)
    result = validate_sga_flat(sga, train_iters=train_iters)
    # `sga:` must not leak into the megatron arg flattening; the launcher
    # only reads train.{system,model,data}, but assert the boundary here
    # so a future flatten change cannot silently start forwarding it.
    assert "sga" not in OmegaConf.to_container(config.train, resolve=True), \
        "sga section leaked into train.*"
    return result
