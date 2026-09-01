# Copyright (c) 2026, SGA framework contributors.
"""method=adaptive: persistent-tau topk controller (port of
scheme_xp's v3.1).

A persistent scalar tau replaces the per-boundary histogram select after a
gated warmup. Dead band ±_ADAPT_BAND around the target ratio; out-of-band
steps are log-space (measured/target)**_ADAPT_LOG_GAIN — self-scaling, gentle
at the band edge, fast when far — clamped to [_ADAPT_MAX_DOWN, _ADAPT_MAX_UP]
per step. Warmup and collapse re-bootstrapping reuse the histogram entry
(select_masks_histogram) in the configured warmup-mode geometry.
"""

import torch

from sga.plugins.selection import _histogram
from sga.plugins.selection._histogram import (
    _HIST_MODES,
    _bin_edge_float,
    select_masks_histogram,
)

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


def _tau_floor(warmup_mode) -> float:
    """Smallest strictly-positive bin edge of the warmup-mode geometry.

    Bin 0's edge is exactly 0.0 — a tau latched there selects everything and
    the multiplicative feedback below can never leave it — so a
    bootstrapped/re-bootstrapped tau is clamped up to this floor."""
    shift, _bins, _stride = _HIST_MODES[warmup_mode]
    return _bin_edge_float(1, shift)


class AdaptiveTau:
    """Impl signature: ``select(grads, k, comm_group, warmup_target)``.

    State — identical on all ranks by construction: the full-sync scores are
    replicated, so every rank measures the same ratio and applies the same
    deterministic update.
    """

    def __init__(self, ratio_percent: float, warmup_syncs, warmup_mode: str):
        if warmup_mode not in _HIST_MODES:
            raise ValueError(
                f"unknown adaptive_warmup_mode: {warmup_mode!r} "
                f"(expected one of {sorted(_HIST_MODES)})"
            )
        self.target_ratio = ratio_percent / 100.0
        self.warmup_syncs = warmup_syncs  # None disables the warmup gate
        self.warmup_mode = warmup_mode
        self.tau = None
        self.warmup_passed = 0

    def warmup_sync_target(self, start_iter: int, period: int):
        """Gated warmup selects implied by THIS selection's warmup window
        (the v3.1 anchor), anchored at start_iter.

        The window spans ``warmup_syncs * period`` iterations from the run
        start, so a selection starting at start_iter sees ``max(0, (window -
        1 - start_iter) // period)`` of it. The ``-1`` encodes the v3.1
        boundary rule: a boundary landing EXACTLY at the window edge counts
        as post-warmup (start=0, period=4, N=5 -> iterations 4..20 are all
        inside the window but only 4 gated selects remain — at 4, 8, 12,
        16). A late start (window already over when masking starts) yields
        0 — a single bootstrap select. Returns None when ``warmup_syncs``
        is unset.
        """
        if self.warmup_syncs is None:
            return None
        warmup_iters = self.warmup_syncs * period
        return max(0, (warmup_iters - 1 - start_iter) // period)

    def _clamp(self, thresh: float) -> float:
        return thresh if thresh > 0.0 else _tau_floor(self.warmup_mode)

    def select(self, grads, k, comm_group, warmup_target):
        """Persistent-threshold select (v3.1, ported from scheme_xp).

        Warmup: histogram select in ``warmup_mode`` (default fast), gated on
        the measured ratio; a miss resets the consecutive count (warmup is
        EXTENDED, never escalated) and the failed select's mask still serves
        that cycle. The effective gate requirement is max(warmup_target, 1)
        — the engine anchors ``warmup_target`` via this impl's own
        ``warmup_sync_target`` hook at start_iter, and a late start (window
        already over) short-circuits to 0, leaving a single bootstrap
        select. After warmup the persistent tau takes over with the
        dead-band feedback rule.

        Collapse guard: a zero measured ratio — or a non-positive tau,
        which is absorbing under the multiplicative update — means feedback
        cannot recover; the selector re-bootstraps from the warmup-mode
        histogram immediately, clamped to the smallest strictly-positive bin
        edge so tau can never latch at exactly 0.0.
        """
        required = max(warmup_target if warmup_target is not None
                       else (self.warmup_syncs if self.warmup_syncs is not None
                             else 1), 1)

        if self.tau is None or self.warmup_passed < required:
            thresh, masks = select_masks_histogram(
                grads, k, comm_group, mode=self.warmup_mode)
            self.tau = self._clamp(thresh)
            measured = (sum(_count_on(m) for m in masks)
                        / sum(g.numel() for g in grads))
            if abs(measured - self.target_ratio) <= _ADAPT_BAND:
                self.warmup_passed += 1
            else:
                self.warmup_passed = 0
            return self.tau, masks

        masks = _threshold_masks(grads, self.tau)
        measured = (sum(_count_on(m) for m in masks)
                    / sum(g.numel() for g in grads))
        if measured == 0.0 or self.tau <= 0.0:
            thresh, masks = select_masks_histogram(
                grads, k, comm_group, mode=self.warmup_mode)
            self.tau = self._clamp(thresh)
            return self.tau, masks

        tau_used = self.tau
        self.tau = _adapt_threshold(tau_used, measured, self.target_ratio)
        return tau_used, masks
