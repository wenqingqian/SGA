# Copyright (c) 2026, SGA framework contributors.
"""The flat SGA config validation core.

Everything a config may say lives in ``sga/config/arguments.py`` (plain
``add_argument`` declarations); THIS module derives the flag vocabulary
from that file by parser introspection — no key set, default or choice is
restated anywhere — and adds the context argparse cannot express as PLAIN
CONDITIONAL RULES over the flat keys (megatron ``validate_args`` style:
``if args.xxx: assert args.yyy``).

Names carry no structure here. Which flags exist is a question for the
declarations; what combinations of VALUES are legal is argued below, one
rule per branch (the enabled/disabled contract, required flags, the
cycle/trace/shadow rules, registered plugin names, topk's per-method
rules). Per-plugin parameters are each plugin's own business: their
vocabularies and defaults live with the plugins
(``sga/plugins/__init__.py:expand_plugin_params``), never in a central
table — a declared-but-foreign param is rejected by the owning
constructor at build time (caveat: an explicit-but-at-default foreign
key may be diffed away before the constructor, since
``materialize_engine_config`` re-diffs the canonical down to explicit
non-defaults). Note the two notions of "explicit" on the yaml route,
intentional per the layer split: the topk dead-config guard fires on
key presence in the explicit input, while the constructor route
consumes the explicit-default diff of the canonical.

Inputs are EXPLICIT-key dicts (only what the user actually wrote): the
launcher passes the yaml ``sga:`` section verbatim; the training side
passes ``explicit_from_args(args)`` (the argparse-defaults diff — with
real defaults filled in by argparse, "flag absent" is indistinguishable
from "flag at default", and the two are semantically identical anyway).

Output: the canonical FLAT dict (every dest at its resolved, typed
value; percent resolved away) used for the engine's config print and the
resume guard comparison. The
engine's internal NESTED read shape is assembled from the canonical by
``sga.runtime.materialize_engine_config`` — validation itself is flat
in, flat out.

Disabled-clean input returns ``None``.
"""

import argparse
import json
import math

from sga.config.arguments import add_sga_args, sga_bool, sga_int_or_null

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
    """Validate an explicit-key flat dict -> the canonical flat dict, or
    None when disabled-clean. Raises ValueError naming the offending key
    on any problem."""
    actions = sga_actions()
    unknown = sorted(set(flat) - set(actions))
    if unknown:
        raise ValueError(
            f"sga: unknown keys {unknown} (known: {sorted(actions)})")
    typed = {dest: _coerce(dest, val) for dest, val in flat.items()}
    defaults = sga_defaults()

    # --- enabled/disabled contract ---
    if typed.get("sga_enabled", defaults["sga_enabled"]) is not True:
        # Disabled semantics: dense training; nothing else may be set
        # (no other key has meaning while disabled).
        for dest in sorted(typed):
            if dest != "sga_enabled":
                raise ValueError(
                    f"{dest} is set but sga_enabled is false; either "
                    "enable SGA or remove the flag")
        return None

    # --- required when enabled ---
    if typed.get("sga_score") is None:
        raise ValueError(
            "sga_score is required when sga_enabled is true "
            "(the choices are the registered scores)")
    if typed.get("sga_backend", defaults["sga_backend"]) == "dense":
        raise ValueError(
            "sga_enabled is true but sga_backend is still 'dense' (the "
            "SGA-off default) — name a sparse backend")

    # --- cycle (incl. percent resolution and the decouple calendar) ---
    if typed.get("sga_cycle_start_iter_percent") is not None:
        if "sga_cycle_start_iter" in typed:
            raise ValueError(
                "set either sga_cycle_start_iter or "
                "sga_cycle_start_iter_percent, not both")
        if not train_iters or int(train_iters) <= 0:
            raise ValueError(
                "sga_cycle_start_iter_percent requires train_iters > 0")
        pct = float(typed["sga_cycle_start_iter_percent"])
        if not 0.0 <= pct <= 100.0:
            raise ValueError(
                "sga_cycle_start_iter_percent must be in [0, 100] "
                f"(got {pct})")
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
                "leave sga_cycle_full_sync_interval unset")
    elif interval is None:
        raise ValueError(
            "sga_cycle_decouple_full_sync_interval_and_period is true "
            "but sga_cycle_full_sync_interval is unset — set a divisor "
            f"of the period smaller than it (period={period})")
    elif not (0 < int(interval) < period and period % int(interval) == 0):
        raise ValueError(
            f"decoupled sga_cycle_full_sync_interval ({interval}) must "
            f"satisfy 0 < full_sync_interval < period ({period}) and "
            "period % full_sync_interval == 0")

    # --- trace (setting any trace key activates the collector) ---
    trace_present = any(dest in typed for dest in actions
                        if dest.startswith("sga_trace_"))
    branch_points = None
    if trace_present:
        tdir = typed.get("sga_trace_dir")
        if not isinstance(tdir, str) or not tdir:
            raise ValueError(
                "sga_trace_dir is required and must be a non-empty string "
                "(the trace output directory)")
        # Lazy import: sga.trace -> sga.config (package __init__) -> this
        # module -> sga.trace is a cycle; sga.trace-first processes must
        # keep importing (only call site, so the deferral costs nothing).
        from sga.trace import parse_branch_points
        try:
            branch_points = parse_branch_points(
                typed.get("sga_trace_branch_points",
                          defaults["sga_trace_branch_points"]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"sga_trace_branch_points: {e}") from None
        window = typed.get("sga_trace_window", defaults["sga_trace_window"])
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(
                f"sga_trace_window must be an int >= 1 (got {window!r})")

    # --- shadow ---
    shadow = bool(typed.get("sga_shadow", defaults["sga_shadow"]))
    if "sga_shadow_score" in typed and not shadow:
        raise ValueError(
            "sga_shadow_score is set but sga_shadow is not true; it only "
            "modifies shadow runs")

    # --- plugin names must be registered (plain value checks: the
    # registries own the name lists, nothing here restates them) ---
    from sga.plugins import (
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_score_spec,
        get_selection_decl,
        register_default_plugins,
        validate_combination,
    )

    register_default_plugins()
    score = typed["sga_score"]
    selection = typed.get("sga_selection", defaults["sga_selection"])
    optimizer = typed.get("sga_optimizer", defaults["sga_optimizer"])
    backend = typed["sga_backend"]
    emission = typed.get("sga_backend_emission",
                         defaults["sga_backend_emission"])
    get_score_spec(score)
    selection_decl = get_selection_decl(selection)
    optimizer_decl = get_optimizer_decl(optimizer)
    backend_decl = get_backend_plugin_decl(backend)
    # The emission rule sits after the decl lookups: a retired backend
    # name must raise the absorbed-backend hint, not the emission refusal.
    if emission == "compact" and backend != "torch_pipelined_index":
        raise ValueError(
            f"sga_backend_emission 'compact' is only supported by "
            f"torch_pipelined_index (got backend '{backend}'); the "
            "writeback-inherent backends run writeback only")

    # --- selection (topk): per-method rules over values ---
    if selection == "topk":
        if typed.get("sga_selection_method") is None:
            raise ValueError(
                "sga_selection_method is required by topk "
                "(adaptive | exact | histogram)")
        if "sga_selection_ratio_percent" not in typed:
            raise ValueError(
                "sga_selection_ratio_percent is required by topk")
        method = typed["sga_selection_method"]
        # dead config: a param another method owns is silently ignored
        # where it is — refuse it instead
        unlocks = {
            "histogram": ("sga_selection_histogram_mode",),
            "adaptive": ("sga_selection_adaptive_warmup_syncs",
                         "sga_selection_adaptive_warmup_mode"),
        }
        method_params = unlocks.get(method, ())
        dead = sorted(dest for dests in unlocks.values() for dest in dests
                      if dest in typed and dest not in method_params)
        if dead:
            raise ValueError(
                f"sga_selection (topk): params {dead} do not belong to "
                f"method '{method}' (it takes "
                f"{sorted(method_params) or 'no extra params'}) — under "
                "this method they would be silently dead; drop them or "
                "switch the method")

    validate_combination(score, selection_decl, optimizer_decl,
                         backend_decl, backend_emission=emission)

    # --- canonical flat dict: every dest at its resolved value ---
    canonical = dict(defaults)
    canonical.update(typed)
    canonical["sga_enabled"] = True
    canonical.pop("sga_cycle_start_iter_percent", None)  # resolved away
    canonical["sga_cycle_start_iter"] = start_iter
    if trace_present:
        canonical["sga_trace_branch_points"] = branch_points
    return canonical


def validate_sga_args(args):
    """Training-side fail-fast entry: validate the parsed namespace's SGA
    configuration (also when disabled — the disabled-cleanliness contract
    is enforced here). Returns the canonical flat dict, or None when
    disabled."""
    return validate_sga_flat(explicit_from_args(args),
                             train_iters=getattr(args, "train_iters", None))


def validate_run_config(config):
    """Launcher-side entry on the whole run config: the ``sga:`` section
    (flat keys, validated fail-fast before any process is spawned) plus
    the train-boundary assert. Returns the canonical flat dict or None
    when disabled."""
    sga = config.get("sga", None)
    if sga is None:
        raise ValueError(
            "Config is missing the 'sga:' section (declare it with just "
            "sga_enabled: false for pure dense training)."
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
