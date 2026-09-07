# Copyright (c) 2026, SGA framework contributors.
"""Topk selection — the selection axis' single registration subject.

Strict-ratio global topk over the optimizer-materialized score magnitudes:
``k = max(1, round(n_total * ratio%))``. Under ``sga_always_sync_layers``
the pool is the engine-filtered eligible subset — forced params never
compete, and ``n_total`` (like every ratio statistic) is the eligible
count (see ``SelectionContext``). The selection ALGORITHM is the
``sga_selection_method`` flag, dispatched to a per-method impl file:

- ``method: histogram`` (+ ``sga_selection_histogram_mode``:
  exact|balanced|fast): bit-pattern histogram select (``_histogram.py``).
  Per-rank histograms of the top bits of |score| fp32 bit patterns are
  SUM-allreduced over the comm group only when
  ``sga_selection_topk_histogram_add_global_allreduce`` is on (default
  off: local histogram only — bit-identical under the engine's replicated
  full-sync scores; on = the merged scheme_xp form). With replicated
  inputs the merged histogram is exactly ``world`` times the local one
  and the select math is scale-invariant, so every rank derives the
  identical bin and threshold BY CONSTRUCTION either way (tied boundary
  values are all selected — masks ``|score| >= thresh``). The default
  (``exact`` mode) does NOT OR-merge the final masks: it leans on
  bitwise-replicated inputs plus the engine's layer-2 abort on divergence
  — merging there too would trade a crash for consistency at the cost of
  an extra uint8 allreduce per reselection, which v1 declines
  deliberately.
- ``method: exact`` (``_exact.py``): full-model cat + torch.kthvalue +
  mask OR-merge. One consistency tier above the histogram family: OR-merge
  TOLERATES residual cross-rank numerical divergence instead of aborting
  (costs O(model) temporaries + a full-model uint8 mask allreduce).
- ``method: adaptive`` (+ ``sga_selection_adaptive_warmup_syncs`` /
  ``sga_selection_adaptive_warmup_mode``, ``_adaptive.py``): persistent-tau
  controller — gated histogram warmup, then log-space dead-band feedback
  toward the target ratio.

Every knob is read straight off ``sga.config.get_args()`` (config
validation owns required-ness, choices and the method gates — the
constructor keeps only the ratio's domain guard and the dispatch's final
else, the two things a direct-construction namespace bypassing argparse
could still get wrong).

"Circular" (the support is regenerated once per cycle period T at the
full-sync boundary and persists unchanged in between; non-topk coordinates
accumulate as residual, folded by the optimizer's reset contract at the
next boundary) is ENGINE-CYCLE semantics, not a property of this selector —
hence the bare registration name ``topk`` (registered historically as
``circular_topk``).
"""

import torch

from sga.config import get_args
from sga.plugins.selection._adaptive import AdaptiveTau
from sga.plugins.selection._exact import ExactKthvalue
from sga.plugins.selection._histogram import HistogramSelect


class TopkSelection:

    def __init__(self):
        args = get_args()
        self.method = args.sga_selection_method
        self.ratio_percent = float(args.sga_selection_ratio_percent)
        assert 0.0 < self.ratio_percent < 100.0, (
            f"ratio_percent must be in (0, 100), got {self.ratio_percent}"
        )
        if self.method == "adaptive":
            warmup_syncs = args.sga_selection_adaptive_warmup_syncs
            self._impl = AdaptiveTau(
                self.ratio_percent,
                int(warmup_syncs) if warmup_syncs is not None else None,
                args.sga_selection_adaptive_warmup_mode,
                global_allreduce=bool(
                    args.sga_selection_topk_histogram_add_global_allreduce),
            )
        elif self.method == "histogram":
            self._impl = HistogramSelect(
                args.sga_selection_histogram_mode,
                block_size=int(args.sga_selection_block_size),
                block_reduce=args.sga_selection_block_reduce,
                global_allreduce=bool(
                    args.sga_selection_topk_histogram_add_global_allreduce))
        elif self.method == "exact":
            self._impl = ExactKthvalue()
        else:
            # direct-construction backstop (argparse choices own the flag
            # routes): a bogus method must not fall through to exact
            raise ValueError(
                "sga_selection_method must be adaptive|exact|histogram "
                f"(got {self.method!r})")
        self.last_threshold = None
        self._rank_gate_checked = False

    @property
    def block_size(self):
        """Selection granularity: 1 = element-level (every non-histogram
        method); > 1 = each selected coordinate heads a contiguous run of
        ``block_size`` elements. The engine reads this to decide whether a
        block-index form exists for the support."""
        return getattr(self._impl, "block_size", 1)

    @property
    def tau(self):
        """Adaptive impl's persistent threshold (None for other methods)."""
        return getattr(self._impl, "tau", None)

    @property
    def warmup_passed(self):
        """Adaptive impl's consecutive-gate count (None for other methods)."""
        return getattr(self._impl, "warmup_passed", None)

    def k_for(self, n_total: int) -> int:
        return max(1, int(round(n_total * self.ratio_percent / 100.0)))

    def warmup_sync_target(self, start_iter: int, period: int):
        """Engine hook (adaptive-only): gated warmup selects implied by this
        selection's warmup window, anchored at start_iter. None for every
        other method and for adaptive-with-warmup-disabled."""
        hook = getattr(self._impl, "warmup_sync_target", None)
        return hook(start_iter, period) if callable(hook) else None

    def _check_rank_gate(self, ctx):
        """One-time rank agreement on the histogram allreduce gate.

        A rank-split ``sga_selection_topk_histogram_add_global_allreduce``
        splits collective participation itself: one rank enters the
        histogram SUM while the other advances to the engine's checksum
        allreduces — a hang no downstream check can attribute. A packed
        MIN/MAX probe over the selection group (the layer-1 count guard's
        form) turns the split into a loud, symmetric abort before the
        first conditional collective. The gate value is read from the impl
        — ``__init__`` stays the args key's single reader; method=exact
        impls expose no gate (a dead knob there) and skip the probe.
        """
        gate = getattr(self._impl, "global_allreduce", None)
        if (gate is None or ctx.comm_group is None
                or not torch.distributed.is_initialized()):
            return True
        device = ctx.grad_views[0].device if ctx.grad_views else "cpu"
        probe = torch.tensor([int(gate)], dtype=torch.int64, device=device)
        packed = torch.stack([probe, -probe]).flatten()
        torch.distributed.all_reduce(
            packed, op=torch.distributed.ReduceOp.MAX, group=ctx.comm_group)
        lo, hi = -int(packed[1].item()), int(packed[0].item())
        if lo != hi:
            raise ValueError(
                "sga_selection_topk_histogram_add_global_allreduce differs "
                f"across the selection group (local={int(gate)}, "
                f"group=[{lo}, {hi}]): ranks would disagree on entering "
                "the histogram SUM allreduce — a collective-participation "
                "split hangs — align the flag on every rank")
        return True

    def select(self, ctx):
        """Return per-bucket element-level uint8 masks from the global topk.

        ``ctx.grad_views`` are the optimizer-materialized score views;
        selection runs only at full-sync cycle boundaries.
        """
        if not self._rank_gate_checked:
            self._rank_gate_checked = self._check_rank_gate(ctx)
        k = self.k_for(sum(g.numel() for g in ctx.grad_views))
        thresh, masks = self._impl.select(
            ctx.grad_views, k, ctx.comm_group,
            getattr(ctx, "warmup_target", None),
        )
        self.last_threshold = thresh
        return masks
