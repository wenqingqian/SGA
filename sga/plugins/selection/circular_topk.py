# Copyright (c) 2026, SGA framework contributors.
"""Circular topk selection.

Strict-ratio global topk over the optimizer-materialized score magnitudes:
``k = max(1, round(n_total * ratio%))``. Default selector is the bit-pattern
histogram in one of three geometry tiers (``hist_mode``; see
``sga/plugins/selection/_histogram.py`` — exact / balanced / fast):
per-rank histograms of the top bits of |score| fp32 bit patterns are
SUM-allreduced over the comm group; with replicated inputs the merged
histogram is exactly ``world`` times the local one and the select math is
scale-invariant, so every rank derives the identical bin and threshold BY
CONSTRUCTION (tied boundary values are all selected — masks
``|score| >= thresh``). The opt-in ``exact_kthvalue`` control path
(full-model cat + torch.kthvalue + mask OR-merge) is kept for A/B
comparison. The default path does NOT OR-merge the final masks: it leans on
bitwise-replicated inputs plus the engine's layer-2 abort on divergence
instead — merging there too would trade a crash for consistency at the cost
of an extra uint8 allreduce per reselection, which v1 declines deliberately.

The opt-in ``adaptive`` mode (the port of scheme_xp's v3.1
controller) replaces the fixed-threshold steady state with a persistent
scalar tau: selections run through a gated warmup using the
``adaptive_warmup_mode`` histogram, then through the persistent tau steered
by log-space dead-band feedback toward the configured ratio (see
``_adaptive_select``). ``hist_mode`` and ``exact_kthvalue`` are rejected
under it — their knobs have no effect there.

"circular": the support is regenerated once per cycle period T at the
full-sync boundary and persisted unchanged in between; non-topk coordinates
accumulate as residual (the optimizer's reset contract folds it at the next
boundary). Selection ALWAYS runs after the optimizer template applied the
update and while
the materialized score still sits in the main-grad copy.
"""

import torch

from sga.plugins import SelectionDecl
from sga.plugins.selection import _histogram
from sga.plugins.selection._histogram import (
    _HIST_BINS,
    _HIST_MODES,
    _abs_bit_top16,
    _bin_edge_float,
    _histogram_select_threshold,
    select_masks_histogram,
)

DECL = SelectionDecl(
    name="circular_topk",
    output_form="mask",
)

# Back-compat alias: the histogram path was inlined here before the
# three-tier split (only tests/test_plugins_unit.py imports the underscore
# names today — Megatron-LM's waist never touches them).
_select_masks_histogram = select_masks_histogram

# Adaptive-threshold path (``adaptive`` param, v3.1): a persistent scalar tau
# replaces the histogram select after a gated warmup. Dead band ±_ADAPT_BAND
# around the target ratio; out-of-band steps are log-space
# (measured/target)**_ADAPT_LOG_GAIN — self-scaling, gentle at the band edge,
# fast when far — clamped to [_ADAPT_MAX_DOWN, _ADAPT_MAX_UP] per step.
_ADAPT_BAND = 0.01
_ADAPT_LOG_GAIN = 0.5
_ADAPT_MAX_UP = 1.15
_ADAPT_MAX_DOWN = 0.9


def _adapt_threshold(thresh, measured_ratio, target_ratio,
                     gain=_ADAPT_LOG_GAIN, max_up=_ADAPT_MAX_UP,
                     max_down=_ADAPT_MAX_DOWN, band=_ADAPT_BAND):
    """One multiplicative threshold update toward the target mask ratio.

    Inside the dead band the threshold is unchanged; outside it the threshold
    scales by (measured/target)^gain — a log-space step proportional to the
    relative deviation. Raising the threshold shrinks the mask, lowering
    grows it.
    """
    if abs(measured_ratio - target_ratio) <= band:
        return thresh
    factor = (measured_ratio / target_ratio) ** gain
    return thresh * min(max(factor, max_down), max_up)


def _count_on(m: torch.Tensor) -> int:
    """Selected-count of a uint8 mask, chunked (a single count_nonzero on a
    full-bucket mask materializes an int64 copy on this torch build)."""
    n = 0
    for s in range(0, m.numel(), _histogram._HIST_CHUNK):
        n += int(torch.count_nonzero(
            m[s:s + _histogram._HIST_CHUNK]).item())
    return n


def _threshold_masks(grads, tau):
    """``|g| >= tau`` masks, chunked (same pass discipline as the histogram)."""
    masks = []
    for g in grads:
        m = torch.empty(g.numel(), dtype=torch.uint8, device=g.device)
        # Read through the module at call time: tests patch
        # _histogram._HIST_CHUNK to exercise the chunking.
        chunk = _histogram._HIST_CHUNK
        for s in range(0, g.numel(), chunk):
            gc = g[s:s + chunk]
            m[s:s + chunk] = gc.abs() >= tau
        masks.append(m)
    return masks


class CircularTopkSelection:

    decl = DECL

    def __init__(self, params: dict):
        self.params = params
        self.ratio_percent = float(params["ratio_percent"])
        assert 0.0 < self.ratio_percent < 100.0, (
            f"ratio_percent must be in (0, 100), got {self.ratio_percent}"
        )
        self.hist_mode = params.get("hist_mode", "exact")
        if self.hist_mode not in _HIST_MODES:
            raise ValueError(
                f"circular_topk: unknown hist_mode {self.hist_mode!r} "
                f"(expected one of {sorted(_HIST_MODES)})"
            )
        self.exact_kthvalue = bool(params.get("exact_kthvalue", False))
        self.adaptive = bool(params.get("adaptive", False))
        warmup_syncs = params.get("adaptive_warmup_syncs", 5)
        self.adaptive_warmup_syncs = (
            int(warmup_syncs) if warmup_syncs is not None else None
        )
        self.adaptive_warmup_mode = params.get("adaptive_warmup_mode", "fast")
        if self.adaptive:
            if self.adaptive_warmup_mode not in _HIST_MODES:
                raise ValueError(
                    f"circular_topk: unknown adaptive_warmup_mode "
                    f"{self.adaptive_warmup_mode!r} (expected one of "
                    f"{sorted(_HIST_MODES)})"
                )
            if self.exact_kthvalue:
                raise ValueError(
                    "circular_topk: adaptive and exact_kthvalue are mutually "
                    "exclusive (the control path exists for exact A/B)"
                )
            if "hist_mode" in params:
                raise ValueError(
                    "circular_topk: hist_mode is dead under adaptive=True "
                    "(the bootstrap uses adaptive_warmup_mode and steady "
                    "state uses the persistent tau); drop it, or use a "
                    "non-adaptive config — hydra overrides must add it with "
                    "'+sga.selection.hist_mode=...'"
                )
        # Adaptive state — identical on all ranks by construction: the
        # full-sync scores are replicated, so every rank measures the same
        # ratio and applies the same deterministic update.
        self._tau = None
        self._warmup_passed = 0
        self.last_threshold = None

    def k_for(self, n_total: int) -> int:
        return max(1, int(round(n_total * self.ratio_percent / 100.0)))

    def warmup_sync_target(self, start_iter: int, period: int):
        """Optional engine hook: gated warmup selects implied by THIS
        selection's warmup window (the v3.1 anchor), anchored at start_iter.

        The window spans ``adaptive_warmup_syncs * period`` iterations from
        the run start, so a selection starting at start_iter sees
        ``max(0, (window - 1 - start_iter) // period)`` of it. The ``-1``
        encodes the v3.1 boundary rule: a boundary landing EXACTLY at the
        window edge counts as post-warmup (start=0, period=4, N=5 ->
        iterations 4..20 are all inside the window but only 4 gated selects
        remain — at 4, 8, 12, 16). A late start (window already over when
        masking starts) yields 0 — a single bootstrap select. Returns None
        when ``adaptive_warmup_syncs`` is unset.
        """
        if self.adaptive_warmup_syncs is None:
            return None
        warmup_iters = self.adaptive_warmup_syncs * period
        return max(0, (warmup_iters - 1 - start_iter) // period)

    def _tau_floor(self) -> float:
        """Smallest strictly-positive bin edge of the warmup-mode geometry.

        Bin 0's edge is exactly 0.0 — a tau latched there selects everything
        and the multiplicative feedback below can never leave it — so a
        bootstrapped/re-bootstrapped tau is clamped up to this floor."""
        shift, _bins, _stride = _HIST_MODES[self.adaptive_warmup_mode]
        return _bin_edge_float(1, shift)

    def _clamp_tau(self, thresh: float) -> float:
        return thresh if thresh > 0.0 else self._tau_floor()

    def select(self, ctx):
        """Return per-bucket element-level uint8 masks from the global topk.

        ``ctx.grad_views`` are the optimizer-materialized score views;
        selection runs only at full-sync cycle boundaries.
        """
        k = self.k_for(sum(g.numel() for g in ctx.grad_views))
        if self.adaptive:
            thresh, masks = self._adaptive_select(
                ctx.grad_views, k, ctx.comm_group,
                getattr(ctx, "warmup_target", None),
            )
        elif self.exact_kthvalue:
            thresh, masks = _select_masks_exact_kthvalue(ctx.grad_views, k, ctx.comm_group)
        else:
            thresh, masks = select_masks_histogram(ctx.grad_views, k, ctx.comm_group,
                                                   mode=self.hist_mode)
        self.last_threshold = thresh
        return masks

    def _adaptive_select(self, grads, k, comm_group, warmup_target):
        """Persistent-threshold select (v3.1, ported from scheme_xp).

        Warmup: histogram select in ``adaptive_warmup_mode`` (default fast),
        gated on the measured ratio; a miss resets the consecutive count
        (warmup is EXTENDED, never escalated) and the failed select's mask
        still serves that cycle. The effective gate requirement is
        max(warmup_target, 1) — the engine anchors ``warmup_target`` via
        this selection's own ``warmup_sync_target`` hook at start_iter, and
        a late start (window already over) short-circuits to 0, leaving a
        single bootstrap select. After warmup the persistent tau takes over
        with the dead-band feedback rule.

        Collapse guard: a zero measured ratio — or a non-positive tau, which
        is absorbing under the multiplicative update — means feedback cannot
        recover; the selector re-bootstraps from the warmup-mode histogram
        immediately, clamped to the smallest strictly-positive bin edge so
        tau can never latch at exactly 0.0.
        """
        target_ratio = self.ratio_percent / 100.0
        required = max(warmup_target if warmup_target is not None
                       else self.adaptive_warmup_syncs, 1)

        if self._tau is None or self._warmup_passed < required:
            thresh, masks = select_masks_histogram(
                grads, k, comm_group, mode=self.adaptive_warmup_mode)
            self._tau = self._clamp_tau(thresh)
            measured = (sum(_count_on(m) for m in masks)
                        / sum(g.numel() for g in grads))
            if abs(measured - target_ratio) <= _ADAPT_BAND:
                self._warmup_passed += 1
            else:
                self._warmup_passed = 0
            return self._tau, masks

        masks = _threshold_masks(grads, self._tau)
        measured = (sum(_count_on(m) for m in masks)
                    / sum(g.numel() for g in grads))
        if measured == 0.0 or self._tau <= 0.0:
            thresh, masks = select_masks_histogram(
                grads, k, comm_group, mode=self.adaptive_warmup_mode)
            self._tau = self._clamp_tau(thresh)
            return self._tau, masks

        tau_used = self._tau
        self._tau = _adapt_threshold(tau_used, measured, target_ratio)
        return tau_used, masks


def _select_masks_exact_kthvalue(grads, k, comm_group):
    """Control path: cat all scores + torch.kthvalue + mask OR-merge.

    Same threshold as the histogram path but costs O(model) temporaries and a
    full-model uint8 mask allreduce; kept for A/B comparison.
    """
    all_vals = torch.cat(grads)
    abs_vals = all_vals.abs_()  # in-place; the signed views are consumed
    thresh = torch.kthvalue(abs_vals, abs_vals.numel() - k + 1).values.item()
    mask_u8 = (abs_vals >= thresh).to(torch.uint8)
    del all_vals, abs_vals

    if comm_group is not None and torch.distributed.is_initialized():
        # OR-merge so the mask is identical across ranks even under residual
        # numerical divergence (a mismatched collective would deadlock).
        # MAX of 0/1 is exactly OR and cannot wrap at any DP world size (a
        # uint8 SUM wraps mod 256 at DP >= 256, identically on all ranks —
        # invisible to every consistency layer); MAX also stays inside
        # NCCL's supported dtype/op set, unlike an int16 widening.
        torch.distributed.all_reduce(
            mask_u8, op=torch.distributed.ReduceOp.MAX, group=comm_group
        )

    masks = []
    offset = 0
    for g in grads:
        n = g.numel()
        masks.append(mask_u8[offset:offset + n])
        offset += n
    return thresh, masks
