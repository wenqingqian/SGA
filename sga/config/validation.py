# Copyright (c) 2026, SGA framework contributors.
"""The SGA config validation — megatron ``validate_args`` style.

Everything a config may say lives in ``sga/config/arguments.py`` (plain
``add_argument`` declarations); THIS module derives the flag vocabulary from
that file by parser introspection (no key set or default is restated
anywhere; the one deliberate choice restatement is the
sga_trace_dump_dtype membership check below — defense-in-depth for
namespaces built without the parser, since both production routes go
through the parser's ``choices``) and adds the context argparse cannot
express as PLAIN CONDITIONAL
RULES over the args (``if args.xxx: raise ...``), exactly the way megatron's
own validate_args works:

  - feature-gated: ``sga_enabled`` false means zero checks below it run —
    a disabled feature's sub-args are never validated (megatron semantics;
    "key set" vs "key at default" is deliberately indistinguishable, that
    is argparse's native semantics too);
  - value-level only: no explicit-diff, no plugin vocabularies, no
    ownership machinery — dead config keys are silently ignored EXCEPT the
    curated list in ``_dead_config_exceptions`` (a different, louder
    category is the set-but-inapplicable pairing refusals —
    ``sga_cycle_full_sync_interval`` without the decouple switch,
    ``sga_shadow_score`` without ``sga_shadow``: live keys whose enabler
    is off, not dead-key resurrections);
  - write-backs: derived values are resolved ONTO the args (megatron
    writes ``args.data_parallel_size = ...`` the same way) — the percent
    spelling resolves into ``sga_cycle_start_iter``, the comma-token list
    keys parse into sorted lists, a false ``sga_checks`` forces every
    ``sga_checks_*`` sub-flag False, and the layout 'auto' spelling
    resolves into er|eb|bb. Exactly five write-back sites exist; every
    other key passes through untouched.

Inputs: an args-like namespace (attribute access). ``args=None`` reads the
sga singleton (``sga.config.get_args`` — the copy build_engine seeded from
the megatron namespace). The launcher pre-validation builds a temporary
namespace by parsing the flags it is about to emit with the real parser, so
both routes see argparse-typed values and one code path.
"""

import argparse
import math
import os

from sga.config.arguments import add_sga_args

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


def parse_layer_patterns(raw):
    """Always-sync layer patterns -> sorted, deduped list of non-empty
    strings (None passes through as the unset spelling — and a
    whitespace-only token is that spelling too: the launcher renders a
    yaml ``[]`` as the single empty token, and an empty list must stay
    OFF, the natural merge spelling for clearing an inherited list).

    Accepts the two legal spellings: one comma-joined token (the CLI
    form; the launcher emits the same shape from a yaml list) or a list
    of strings. Empty or whitespace-only ELEMENTS are refused — a typo'd
    join ("a,,b") would match nothing and the knob would die silently
    (the attach-time zero-match guard is the second line of defense).
    The literal token ``null`` is refused as well: the launcher's
    explicit-null spelling must not become a pattern that matches a
    parameter name (the branch-points flag sets the same precedent).
    The canonical form is sorted — matching is a union, so a reordered
    list is the same config (resume-safe, like parse_branch_points).
    """
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError(
            f"must be a comma-joined string or a list of strings "
            f"(got {raw!r})")
    out = set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"patterns must be non-empty strings (got {item!r})")
        pat = item.strip()
        if pat == "null":
            raise ValueError(
                "an explicit null is NOT the unset spelling here (the "
                "null token would match a parameter name literally); "
                "leave the flag unset instead")
        out.add(pat)
    return sorted(out) or None


def _dead_config_exceptions(args) -> None:
    """Dead-config refusals that are worth their loudness (user-curated
    list): everything else that a config path does not read is silently
    ignored, megatron-style. New entries go HERE and nowhere else.

    block_size: the one knob whose silent death means a WRONG EXPERIMENT
    rather than a no-op — block-granular selection configured under a
    non-histogram method would silently run element-sparse.
    """
    if args.sga_selection_method != "histogram":
        if args.sga_selection_block_size != sga_defaults()["sga_selection_block_size"]:
            raise ValueError(
                "sga_selection_block_size only applies to the histogram "
                "method (got method "
                f"'{args.sga_selection_method}'): under it the run would "
                "silently degrade to element-granular selection — drop the "
                "flag or switch the method")


def validate_sga_args(args=None) -> None:
    """Validate the SGA configuration on an args namespace.

    ``args=None`` reads the sga singleton (build_engine route); the
    launcher pre-validation passes the temporary namespace it parsed from
    the flags it is about to emit. Raises ValueError naming the offending
    key on any problem; returns None (megatron ``validate_args`` carries
    no return value — derived values are written back onto ``args``).
    """
    if args is None:
        from sga.config import get_args
        args = get_args()
    if not args.sga_enabled:
        # Feature off → zero checks (megatron semantics: a disabled
        # feature's sub-args are never validated).
        return
    # The name checks below consult the plugin registries; populate them
    # first (idempotent; function-level import keeps the cycle closed —
    # plugins import sga.config, never the other way round at module level).
    from sga.plugins import plugin_class, register_default_plugins

    register_default_plugins()

    # --- checks master switch (derived write-back, megatron style) ---
    # False forces every consistency sub-flag off. A sub-flag explicitly
    # set true under a false master is master-override semantics, NOT a
    # dead-knob error: the sub-flags default true, so "explicitly true"
    # is indistinguishable from "unset" and cannot be refused. The
    # write-back lands on the CALLER's namespace: the launcher's
    # pre-validation namespace is discarded after validation (the flag
    # emission face regenerates from the yaml independently), while the
    # build_engine route writes the sga singleton after seeding and
    # before the engine constructor caches the resolved flags — the
    # megatron namespace keeps the user's original spelling, so the
    # resume guard comparison face is unaffected. Idempotent.
    if not args.sga_checks:
        for _dest in (d for d in sga_defaults()
                      if d.startswith("sga_checks_")):
            setattr(args, _dest, False)

    # --- required when enabled ---
    if args.sga_score is None:
        raise ValueError(
            "sga_score is required when sga_enabled is true "
            "(the choices are the registered scores)")
    if args.sga_backend == "dense":
        raise ValueError(
            "sga_enabled is true but sga_backend is still 'dense' (the "
            "SGA-off default) — name a sparse backend")

    # --- plugin names must be registered (plain lookups, curated errors;
    # the score name is already pinned by the argparse choices) ---
    plugin_class("selection", args.sga_selection)
    plugin_class("optimizer", args.sga_optimizer)
    plugin_class("backend", args.sga_backend)

    # --- cycle (incl. percent resolution and the decouple calendar) ---
    if args.sga_cycle_start_iter_percent is not None:
        if args.sga_cycle_start_iter != sga_defaults()["sga_cycle_start_iter"]:
            raise ValueError(
                "set either sga_cycle_start_iter or "
                "sga_cycle_start_iter_percent, not both")
        if not args.train_iters or int(args.train_iters) <= 0:
            raise ValueError(
                "sga_cycle_start_iter_percent requires train_iters > 0")
        pct = float(args.sga_cycle_start_iter_percent)
        if not 0.0 <= pct <= 100.0:
            raise ValueError(
                "sga_cycle_start_iter_percent must be in [0, 100] "
                f"(got {pct})")
        # write-back (megatron validate_args pattern): the percent
        # spelling resolves into the absolute step it names
        args.sga_cycle_start_iter = max(
            0, int(round(pct / 100.0 * int(args.train_iters))))
        args.sga_cycle_start_iter_percent = None
    period = int(args.sga_cycle_period)
    if period < 1:
        raise ValueError("sga_cycle_period must be >= 1")
    decouple = bool(args.sga_cycle_decouple_full_sync_interval_and_period)
    interval = args.sga_cycle_full_sync_interval
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

    # --- trace (a non-default trace value activates the collector) ---
    trace_present = (
        args.sga_trace_dir is not None
        or args.sga_trace_branch_points is not None
        or args.sga_trace_window
        != sga_defaults()["sga_trace_window"]
        or args.sga_trace_dump_dtype
        != sga_defaults()["sga_trace_dump_dtype"]
        or args.sga_trace_async
        != sga_defaults()["sga_trace_async"]
        or args.sga_trace_async_queue_depth
        != sga_defaults()["sga_trace_async_queue_depth"]
        or args.sga_trace_async_max_workset_gib
        != sga_defaults()["sga_trace_async_max_workset_gib"])
    if trace_present:
        tdir = args.sga_trace_dir
        if not isinstance(tdir, str) or not tdir:
            raise ValueError(
                "sga_trace_dir is required and must be a non-empty string "
                "(the trace output directory)")
        # Defense-in-depth restatement of the argparse choices: the CLI and
        # the launcher's yaml re-parse are both gated by the parser, but a
        # hand-built namespace reaches here unguarded — same backstop role
        # as TraceCollector's constructor check.
        dd = args.sga_trace_dump_dtype
        if dd not in ("fp32", "bf16"):
            raise ValueError(
                f"sga_trace_dump_dtype must be 'fp32' or 'bf16' (got {dd!r})")
        # Lazy import: sga.trace pulls in torch, so import it only when a
        # trace config is actually present.
        from sga.trace import parse_branch_points
        try:
            # write-back: the comma token parses into the sorted list
            args.sga_trace_branch_points = parse_branch_points(
                args.sga_trace_branch_points)
        except (TypeError, ValueError) as e:
            raise ValueError(f"sga_trace_branch_points: {e}") from None
        window = args.sga_trace_window
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(
                f"sga_trace_window must be an int >= 1 (got {window!r})")
        depth = args.sga_trace_async_queue_depth
        if (not isinstance(depth, int) or isinstance(depth, bool)
                or depth < 1):
            raise ValueError(
                f"sga_trace_async_queue_depth must be an int >= 1 "
                f"(got {depth!r})")
        gib = args.sga_trace_async_max_workset_gib
        if (not isinstance(gib, (int, float)) or isinstance(gib, bool)
                or not math.isfinite(gib) or gib <= 0.0):
            raise ValueError(
                f"sga_trace_async_max_workset_gib must be a finite float "
                f"> 0 (got {gib!r})")

    # --- shadow ---
    shadow = bool(args.sga_shadow)
    if args.sga_shadow_score is not True and not shadow:
        raise ValueError(
            "sga_shadow_score is set but sga_shadow is not true; it only "
            "modifies shadow runs")
    if shadow and args.sga_optimizer == "adams":
        # Shadow is the one run shape whose optimizer step is the STOCK
        # megatron optimizer (a literal MixedPrecisionOptimizer.step()
        # replication) — every live path (sparse, full-sync and the dense
        # bootstrap phases) runs the configured plugin template. Under
        # AdamS that step would silently train AdamW stock numerics while
        # the config claims AdamS, so the combination is refused outright:
        # AdamS runs own every step they take.
        raise ValueError(
            "sga_optimizer 'adams' is refused with sga_shadow true: shadow "
            "replicates the stock optimizer step literally (AdamW "
            "numerics), so an AdamS-configured shadow run would silently "
            "train stock AdamW instead of AdamS — shadow probes exist to "
            "benchmark stock replication, which AdamS is not")

    # --- always-sync layer patterns (the per-parameter dense axis) ---
    # write-back: the comma token parses into the sorted pattern list
    try:
        args.sga_always_sync_layers = parse_layer_patterns(
            args.sga_always_sync_layers)
    except ValueError as e:
        raise ValueError(f"sga_always_sync_layers: {e}") from None
    if (args.sga_always_sync_layers is not None
            and args.pipeline_model_parallel_size > 1):
        # (a config-legality rule, not an engine-internal one: the
        # launcher pre-validation must see it too)
        raise ValueError(
            "sga_always_sync_layers: always-sync layer matching is "
            "per-rank over the attached stage's params (with pipeline "
            "parallelism > 1 the endpoint and middle ranks see different "
            "param sets); v1 pins pipeline_model_parallel_size == 1 — "
            "drop the flag or run with pp == 1")

    # --- optimizer (adamw | adams): the boundary dual-lr knob ---
    # The knob's rules below are optimizer-symmetric: both templates
    # implement the same split (the applied step's lr scales on the
    # mask-inactive tail, weight-decay component included), so the same
    # gates fire for either name.
    lr_scale = args.sga_optimizer_full_sync_inactive_lr_scale
    if lr_scale != 1.0:
        # argparse's float typing ACCEPTS "nan"/"inf" spellings — the
        # domain rule refuses them here; 0.0 is legal (freeze the tail
        # outright — the split multiplies the whole update, weight decay
        # included), anything negative is not a step size.
        if not math.isfinite(lr_scale) or lr_scale < 0.0:
            raise ValueError(
                "sga_optimizer_full_sync_inactive_lr_scale must be a finite "
                f"float >= 0.0 (got {lr_scale!r})")
        if shadow:
            raise ValueError(
                "sga_optimizer_full_sync_inactive_lr_scale "
                f"({lr_scale!r}) is refused with sga_shadow true: shadow "
                "replicates stock numerics bitwise and the dual-lr "
                "boundary apply deviates by design")
        if period == 1:
            raise ValueError(
                "sga_optimizer_full_sync_inactive_lr_scale "
                f"({lr_scale!r}) is refused with sga_cycle_period 1: "
                "every step is a full sync (no sparse rounds, no "
                "mask-inactive tail), the knob would be dead config")

    # --- emission ↔ backend pairing (the one cross-axis combination rule
    # that survives the registry simplification: 'compact' is a switch only
    # torch_pipelined_index implements — the writeback-inherent backends
    # have no compact handoff to give) ---
    emission = args.sga_backend_emission
    if emission == "compact" and args.sga_backend != "torch_pipelined_index":
        raise ValueError(
            f"sga_backend_emission 'compact' is only supported by "
            f"torch_pipelined_index (got backend '{args.sga_backend}'); "
            "the writeback-inherent backends run writeback only")

    # --- selection (topk) ---
    if args.sga_selection == "topk":
        if args.sga_selection_method is None:
            raise ValueError(
                "sga_selection_method is required by topk "
                "(adaptive | exact | histogram)")
        if args.sga_selection_ratio_percent is None:
            raise ValueError(
                "sga_selection_ratio_percent is required by topk")
        ratio = args.sga_selection_ratio_percent
        if (not isinstance(ratio, float) or not math.isfinite(ratio)
                or not 0.0 < ratio < 100.0):
            # argparse's float typing accepts "nan"/"inf"; the selector's
            # constructor assert is the direct-construction backstop only
            raise ValueError(
                "sga_selection_ratio_percent must be a finite float in "
                f"(0, 100) (got {ratio!r})")
        # --- selection freeze: only histogram balanced is supported for
        # now — other method/mode combinations refuse here so different
        # selections cannot confound the test comparisons. This sits
        # ahead of every selection-conditional rule below (the curated
        # dead-config gate and the block comm-path rules stay as shadowed
        # defense); re-enabling another selection is a deliberate
        # gate-lift, not a config choice. SGA_ALLOW_FROZEN_SELECTION=1
        # exists only to keep legacy baseline scripts' existing
        # configurations running; any other use is a gate-lift in
        # disguise, and the escape prints loudly every time it opens.
        if (args.sga_selection_method != "histogram"
                or args.sga_selection_histogram_mode != "balanced"):
            if os.environ.get("SGA_ALLOW_FROZEN_SELECTION") == "1":
                print("[sga] WARNING: selection freeze lifted by "
                      "SGA_ALLOW_FROZEN_SELECTION=1 — running method "
                      f"'{args.sga_selection_method}' mode "
                      f"'{args.sga_selection_histogram_mode}' outside the "
                      "frozen histogram-balanced surface", flush=True)
            else:
                raise ValueError(
                    "selection is frozen to histogram balanced for now: "
                    "only sga_selection_method 'histogram' with "
                    "sga_selection_histogram_mode 'balanced' is accepted "
                    f"(got method '{args.sga_selection_method}', mode "
                    f"'{args.sga_selection_histogram_mode}') — other "
                    "selections are refused at config time so runs stay "
                    "comparable; lift this gate to re-enable them")
        # --- block granularity (histogram-unlocked; see
        # _dead_config_exceptions for the method gate) ---
        block_size = args.sga_selection_block_size
        if (not isinstance(block_size, int) or isinstance(block_size, bool)
                or block_size < 1 or block_size > 1024
                or block_size & (block_size - 1)):
            raise ValueError(
                f"sga_selection_block_size must be a power of two in "
                f"[1, 1024] (got {block_size!r}); 1 keeps element-granular "
                "selection")
        # the curated dead-config gate runs BEFORE the block comm-path
        # rules: for method=exact + block_size + nccl/tpi-auto the honest
        # first error is the dead block_size knob, not a backend-switch
        # pointer
        _dead_config_exceptions(args)
        # --- backend layout: 'auto' resolves via sga.plugins.resolve_layout
        # (single source; an explicit value passes through untouched, its
        # legality is the rules below).
        layout_explicit = args.sga_backend_layout != "auto"
        if not layout_explicit:
            from sga.plugins import resolve_layout
            args.sga_backend_layout = resolve_layout(
                args.sga_backend_layout, block_size)
        layout = args.sga_backend_layout
        # Block-granularity comm-path constraints. The mask stays the
        # canonical support form, so a block-structured mask is CORRECT on
        # every backend; the layout only routes WHICH index form the
        # transport consumes (taxonomy: sga.plugins.resolve_layout).
        # Dead-layout and cross-axis refusals:
        if block_size > 1:
            # fast places the threshold at the crossing bin's LOWER edge
            # (systematic over-selection) and a block payload amplifies
            # it. Selection-layer gate: holds for eb AND bb.
            # TODO(fast-block): sound once the threshold moves to the
            # crossing bin's midpoint/upper edge; lift this gate there.
            if args.sga_selection_histogram_mode == "fast":
                raise ValueError(
                    "sga_selection_block_size > 1 with "
                    "sga_selection_histogram_mode 'fast': fast places the "
                    "threshold at the crossing bin's lower edge and the "
                    "block payload amplifies the systematic "
                    "over-selection (measured realized ratio 0.253 vs "
                    "0.20 target) — use 'balanced' or 'exact', or keep "
                    "block_size 1")
            if layout == "er":
                raise ValueError(
                    "sga_backend_layout 'er' (element selection + element "
                    f"index) refuses sga_selection_block_size > 1 (got "
                    f"{block_size}): er names element-granular selection. "
                    "Use layout 'eb' (block selection transported as the "
                    "expanded element idx) or 'bb' (block index), or set "
                    "block_size 1")
        elif layout_explicit and layout in ("eb", "bb"):
            raise ValueError(
                f"sga_backend_layout '{layout}' (block selection) refuses "
                "sga_selection_block_size == 1: it names block-granular "
                "selection. Use layout 'er' or set block_size > 1")
        # The mask backend never transports an index form — an explicit
        # layout there is dead config with a routing connotation (same
        # refusal category the curated dead-config list carves out).
        if layout_explicit and args.sga_backend == "torch_pipelined_mask":
            raise ValueError(
                f"sga_backend_layout '{layout}' is refused with backend "
                "torch_pipelined_mask: the mask transport consumes the "
                "canonical mask directly and no index form is ever "
                "built — drop the layout flag")
        # bb on the idx backends: torch routes the block gather/scatter
        # path (writeback-only); nccl routes the packed descriptor API
        # (block_size bounded to a power of two in [4, 64]).
        if layout == "bb" and args.sga_backend == "torch_pipelined_index" \
                and emission != "writeback":
            raise ValueError(
                f"sga_backend_layout 'bb' with backend "
                f"torch_pipelined_index needs sga_backend_emission "
                f"'writeback' (got '{emission}'"
                + (", which resolves to 'compact' here"
                   if emission == "auto" else "")
                + "): the block path's payload is mask-guarded lanes, "
                "not an (idx, values) compact handoff; layout 'eb' keeps "
                "the element idx and allows compact")
        if (layout == "bb" and args.sga_backend == "nccl_pipelined_index"
                and not (4 <= block_size <= 64)):
            raise ValueError(
                "sga_backend_layout 'bb' with backend nccl_pipelined_index "
                f"bounds sga_selection_block_size to a power of two in "
                f"[4, 64] (got {block_size}): 4 is the descriptor kernel's "
                "ABI floor (blockShift >= 2 decode), 64 keeps the packed "
                "heads unconditionally representable under the spans' "
                "64-element padding — use layout 'eb' (element transport) "
                "for other block sizes")
