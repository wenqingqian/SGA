# Copyright (c) 2026, SGA framework contributors.
"""The SGA arguments — a plain megatron-style declaration file.

This is the ONE file a developer reads to see every SGA knob: each
``add_argument`` line carries the flag, the type, the default and the
semantics. There are no classes and no tables — declarations only; the
validation that needs cross-argument context lives in
``sga/config/validation.py`` and derives its vocabulary from THIS file
by parser introspection (never restated).

The same flat vocabulary is the whole user-facing config surface, 1:1:

    yaml ``sga:`` section key == flag with ``-`` spelled ``_`` == the
    ``sga_*`` attribute the megatron namespace (and every checkpoint)
    carries.  There is no second form, no nested section, no JSON blob.

Pipeline (flags ARE the transport — nothing is re-serialized on the
way in):

    yaml ``sga:`` keys --launcher--> ``--sga-*`` flags
                                        |
    hand-written flags ----------------> megatron argparse (this file)
                                        |
                             args namespace (``sga_*`` attrs)
                                        |
              fork waist collects ``sga_*`` attrs -> ``build_engine``
                                        |
              ``sga/config/validation.py`` -> engine config (internal)

Conventions (single-sourced here, consumed by introspection):
  - booleans use the ``sga_bool`` string type ("true"/"false", any
    case) so the namespace carries real bools; the launcher emits the
    same spelling.
  - ``sga_selection_adaptive_warmup_syncs`` additionally accepts the
    explicit ``null`` spelling (``sga_int_or_null``): "null" disables
    the warmup gate, which is distinct from the default 5.
  - ``sga_trace_branch_points`` and ``sga_always_sync_layers`` ride as
    one comma-joined token (megatron list style; yaml may also give a
    string list); ``sga_cycle_start_iter_percent`` is resolved
    against ``train_iters`` during validation.

Axis-identity flags (``--sga-score``/``--sga-selection``/...) take
registered plugin names; the registries are runtime data, so the value
check happens in validation against ``sga.plugins.registered_names()``.
The one exception is ``--sga-score``, whose closed vocabulary is also
declared as argparse ``choices`` and kept synced with the registry by
``test_inv4_score_choices_synced_with_registry``. "Required" below
means required when ``--sga-enabled true`` (a dense run passes no SGA
flags at all).
"""

import argparse


def sga_bool(text):
    """Argparse type for SGA booleans: "true"/"false" (any case) -> bool."""
    if isinstance(text, bool):
        return text
    t = str(text).strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"must be 'true' or 'false' (got {text!r})")


def sga_int_or_null(text):
    """Argparse type for a nullable int: "null" -> None, else int."""
    if text is None:
        return None
    if isinstance(text, str) and text.strip().lower() == "null":
        return None
    return int(text)


def add_sga_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every SGA flag (called from the training entry's
    ``extra_args_provider`` chain). Groups mirror the config axes."""
    _add_sga_core_args(parser)
    _add_sga_cycle_args(parser)
    _add_sga_checks_args(parser)
    _add_sga_trace_args(parser)
    _add_sga_score_args(parser)
    _add_sga_selection_args(parser)
    _add_sga_optimizer_args(parser)
    _add_sga_backend_args(parser)
    return parser


def _add_sga_core_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA core", "master switch and engine modes")
    group.add_argument(
        "--sga-enabled", type=sga_bool, default=False,
        help="master switch: true turns on the SGA engine; false (or a run "
             "with no --sga-* flags at all) is dense training, "
             "byte-identical to stock megatron. When false no other sga "
             "flag may be set (an at-default spelling errors in the yaml "
             "sga: section, is silently dropped on the command line).")
    group.add_argument(
        "--sga-shadow", type=sga_bool, default=False,
        help="shadow mode: replicate the stock optimizer step literally and "
             "only READ state at cycle boundaries (dense numerics "
             "invariant); the score is still computed read-only at "
             "boundaries for offline analysis.")
    group.add_argument(
        "--sga-shadow-score", type=sga_bool, default=True,
        help="false strips even the read-only boundary scoring from a "
             "shadow run (the pure stock-replication probe); only "
             "meaningful with --sga-shadow true.")
    group.add_argument(
        "--sga-always-sync-layers", type=str, default=None,
        help="substring patterns of parameter names whose gradients are "
             "always fully communicated (the per-parameter dense axis; "
             "complements the per-step cycle): matched params never "
             "compete in the topk — their coordinates leave the ratio "
             "denominator too — and are force-included in every support, "
             "so they are communicated and cleared each sparse step, "
             "never accumulate residual, and get the standard update "
             "every step. One comma-joined token (megatron list style) or "
             "a yaml string list; unset = plain sparse selection over "
             "all coordinates. Pipeline parallelism > 1 is refused while "
             "this flag is set (v1): matching is per-rank over the "
             "attached stage's params.")


def _add_sga_cycle_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA cycle", "the reselection calendar")
    group.add_argument(
        "--sga-cycle-start-iter", type=int, default=0,
        help="engine step the cycle calendar anchors at; the first "
             "reselection happens at start_iter + period (before that "
             "the run is dense warmup). Default 0. Mutually exclusive "
             "with --sga-cycle-start-iter-percent when both are "
             "explicitly set; on this flag route a value equal to the "
             "default 0 counts as unset.")
    group.add_argument(
        "--sga-cycle-start-iter-percent", type=float, default=None,
        help="alternative spelling of --sga-cycle-start-iter as a percent "
             "of train-iters, resolved to the absolute step during "
             "validation (range [0, 100]); mutually exclusive with an "
             "explicitly set --sga-cycle-start-iter (a flag value equal "
             "to the default 0 counts as unset; on the yaml route an "
             "explicit sga_cycle_start_iter: 0 key counts as set).")
    group.add_argument(
        "--sga-cycle-period", type=int, default=1,
        help="reselection period T (>= 1): the support is regenerated at "
             "full-sync boundaries every T steps and persists unchanged "
             "in between; residual accumulates on unselected "
             "coordinates. Default 1.")
    group.add_argument(
        "--sga-cycle-decouple-full-sync-interval-and-period",
        type=sga_bool, default=False,
        help="decouple the full-sync (residual fold + score emission) "
             "interval from the reselection period; true requires "
             "--sga-cycle-full-sync-interval to be a strict divisor of "
             "the period. Default false.")
    group.add_argument(
        "--sga-cycle-full-sync-interval", type=int, default=None,
        help="full sync every N steps, N a strict divisor of the period "
             "with N < period; only legal alongside --sga-cycle-decouple-"
             "full-sync-interval-and-period true — unset means every "
             "cycle boundary is a full sync.")


def _add_sga_checks_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA checks", "consistency-machinery toggles (research escape hatches)")
    group.add_argument(
        "--sga-checks-count-guard", type=sga_bool, default=True,
        help="layer-1 consistency guard: cross-rank element-count check of "
             "the selected support at each activation (runs once per "
             "cycle on cached counts); off is a research escape hatch, "
             "not a production setting. Default true.")


def _add_sga_trace_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA trace", "offline replay data collection (engine-side service)")
    group.add_argument(
        "--sga-trace-dir", type=str, default=None,
        help="trace output directory; setting it at all activates the "
             "collector (rejected when --sga-enabled false — dense "
             "replay-grade traces come from shadow runs).")
    group.add_argument(
        "--sga-trace-branch-points", type=str, default=None,
        help="engine steps to dump full optimizer state at, as one "
             "comma-joined token (or a yaml int list); sorted, deduped, "
             ">= 0. Leave the flag unset for no state dumps — an explicit "
             "null is NOT the unset spelling (it is rejected; only "
             "--sga-selection-adaptive-warmup-syncs accepts null).")
    group.add_argument(
        "--sga-trace-window", type=int, default=100,
        help="gradient-capture window length after each branch point "
             "(>= 1). Default 100.")


def _add_sga_score_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA score", "the boundary score (registry data, no params)")
    group.add_argument(
        "--sga-score", type=str, default=None,
        choices=("abs_g", "abs_m", "coi", "update"),
        help="score name (required with --sga-enabled true; the choices "
             "are the registered scores). There are no score params — "
             "any --sga-score-* flag is a typo.")


def _add_sga_selection_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA selection", "the support-selection subject (topk)")
    group.add_argument(
        "--sga-selection", type=str, default="topk",
        help="selection plugin name (registered names are the choices). "
             "Default topk — currently the only subject, so the flag is "
             "omittable.")
    group.add_argument(
        "--sga-selection-method", type=str, default=None,
        choices=("adaptive", "exact", "histogram"),
        help="selection algorithm (required with --sga-enabled true): "
             "adaptive = persistent tau controller; exact = kthvalue + "
             "OR-merge; histogram = bit-pattern histogram select.")
    group.add_argument(
        "--sga-selection-ratio-percent", type=float, default=None,
        help="topk mask ratio in percent, strictly inside (0, 100) "
             "(required with --sga-enabled true).")
    group.add_argument(
        "--sga-selection-histogram-mode", type=str, default="exact",
        choices=("balanced", "exact", "fast"),
        help="method=histogram only: geometry tier. Default exact.")
    group.add_argument(
        "--sga-selection-adaptive-warmup-syncs", type=sga_int_or_null,
        default=5,
        help="method=adaptive only: gated warmup selects before the "
             "persistent tau takes over; 'null' disables the warmup "
             "gate. Default 5.")
    group.add_argument(
        "--sga-selection-adaptive-warmup-mode", type=str, default="fast",
        choices=("balanced", "exact", "fast"),
        help="method=adaptive only: histogram tier used to bootstrap/steer "
             "the adaptive tau. Default fast.")
    group.add_argument(
        "--sga-selection-block-size", type=int, default=1,
        help="method=histogram only: select in contiguous blocks of this "
             "many gradient elements — each support index then denotes the "
             "START of a contiguous x-element run instead of one element. "
             "Must be a power of two in [1, 1024]; 1 (default) keeps "
             "element-granular selection. With block_size > 1 the backend "
             "must be torch_pipelined_mask, or torch_pipelined_index with "
             "sga_backend_emission=writeback (the compact consumers and "
             "the nccl fork kernel are element-granular).")
    group.add_argument(
        "--sga-selection-block-reduce", type=str, default="mean",
        choices=("mean", "max"),
        help="method=histogram only, block_size > 1: the per-block topk "
             "value — the mean of |score| over the block's real length "
             "(default; tail blocks are not discriminated against) or the "
             "max (keeps spiky blocks). A dead knob under block_size 1 — "
             "setting it there is refused.")


def _add_sga_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA optimizer", "the optimizer template (adamw)")
    group.add_argument(
        "--sga-optimizer", type=str, default="adamw",
        help="optimizer template name (adamw, the only template).")
    group.add_argument(
        "--sga-optimizer-residual-mean", type=sga_bool, default=False,
        help="pre-fold the accumulated residual by the fold window (the "
             "accumulation span between full syncs) at full-sync "
             "boundaries (residual-mean semantics). Default false.")
    group.add_argument(
        "--sga-optimizer-full-sync-inactive-lr-scale", type=float,
        default=1.0,
        help="boundary dual-lr: on full-sync steps the optimizer apply "
             "uses lr at mask-active coords (communicated during the "
             "window) and lr * scale at mask-inactive coords (the "
             "residual-carrying tail). The m/v commit, the materialized "
             "score and the bias corrections are all unchanged — only "
             "the applied step size splits, multiplying the tail's "
             "whole displacement (weight-decay component included) by "
             "scale. The mask is the outgoing window's, so the fresh "
             "gradient at inactive coords is scaled too (not only the "
             "accumulated residual). Buckets "
             "with no support yet (bootstrap) and degenerate mask-less "
             "buckets are fully communicated: plain lr everywhere. "
             "Backend-agnostic: the resident bucket mask serves both "
             "emission modes. Refused with sga_shadow true or "
             "sga_cycle_period 1 when != 1.0. Default 1.0 (bit-identical "
             "to the single-lr behavior).")
    group.add_argument(
        "--sga-optimizer-compact-search", type=str, default="table",
        choices=("table", "bisection"),
        help="compact-path idx alignment search: 'table' builds a "
             "1024-stride directory table of lower_bound(idx, j*1024) "
             "once per selection cycle (the bucket idx is frozen within "
             "a cycle, so the table cannot go stale) and brackets both "
             "lower_bounds to <=1024 candidates; 'bisection' keeps the "
             "plain full-depth bisection (the historical form). "
             "Silently ignored on writeback-emission runs and the torch "
             "fallback path. Default table.")


def _add_sga_backend_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "SGA backend", "the sparse allreduce transport")
    group.add_argument(
        "--sga-backend", type=str, default="dense",
        help="backend plugin name; the default 'dense' is the SGA-off "
             "representation — with --sga-enabled true a sparse backend "
             "must be named (registered names are the choices).")
    group.add_argument(
        "--sga-backend-emission", type=str, default="auto",
        choices=("auto", "compact", "writeback"),
        help="how the sparse allreduce hands its result to the optimizer: "
             "'compact' keeps the communicated values in the compact "
             "buffer (the optimizer's merge-join kernel consumes "
             "(idx, values) directly; deterministic compact grad-norm); "
             "'writeback' scatters them back into the dense grad buffer "
             "for the masked optimizer kernel (the historical "
             "choreography). 'auto' takes each backend's inherent form — "
             "compact for torch_pipelined_index, writeback for "
             "torch_pipelined_mask / nccl_pipelined_index. Only "
             "torch_pipelined_index switches behavior; an explicit "
             "'compact' is refused on the writeback-inherent backends. "
             "Default auto.")
    group.add_argument(
        "--sga-backend-high-priority-group", type=sga_bool, default=False,
        help="torch_pipelined_index only: run sparse collectives on a twin "
             "high-priority NCCL group (true selects the legacy "
             "pipelined_idx_sorted TRANSPORT MODE only — reproducing that "
             "former configuration also needs index_slices 3; false: legacy "
             "indexed_sparse on the training world's own group). "
             "Default false.")
    group.add_argument(
        "--sga-backend-index-slices", type=int, default=2,
        help="torch_pipelined_index only: number of idx slices pipelined "
             "against the allreduces (enforced >= 1). Default 2.")
    group.add_argument(
        "--sga-backend-mask-slices", type=int, default=0,
        help="torch_pipelined_mask only: mask slices pipelined against "
             "NCCL ops (0 = auto, i.e. 2). Default 0.")
