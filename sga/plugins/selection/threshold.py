# Copyright (c) 2026, SGA framework contributors.
"""Threshold selection (historical origin scheme).

``support = {|g| >= threshold}`` with the adaptive threshold controller that
drives the selected ratio into a target band (ported verbatim in math from
scheme_origin/grad_sync.py::_adapt_threshold, 2026-08-27 survey S2):

- ratio measured over mask positions EXCLUDING always-sync elements;
- optional warmup anneal of the target bounds from ``warmup_start_ratio``
  down to [target_low, target_high] over ``warmup_iters`` reselections;
- multiplicative adjustment (adapt_factor 0.1, single-step increase capped
  at 1.15, decrease floored at 0.9) toward the band midpoint;
- hard cap ``threshold_max``.

Consistency: local masks are OR-merged across the DP group with one MAX
allreduce per bucket directly on the uint8 masks — pointwise max of 0/1 is
exactly OR, and unlike a uint8 SUM it cannot wrap mod 256 at DP >= 256
(a wrap would be identical on all ranks, invisible to every consistency
layer); any rank's decision still wins the merge — reproducing the
historical global-OR mask. The score must expose the fused threshold path.
``grad_norm_bypass``: if the (DP-bitwise-consistent) grad norm of a full-sync
step exceeds the threshold, the NEXT step is forced full — decided from
globally-consistent state only (notes section 9).
"""

import torch

from sga.plugins import SelectionDecl


DECL = SelectionDecl(
    name="threshold",
    output_form="mask",
    scope="elementwise_local",
    needs_threshold_path=True,
)

_ADAPT_FACTOR = 0.1
_INCREASE_CAP = 1.15
_DECREASE_FLOOR = 0.9


class ThresholdSelection:

    decl = DECL

    def __init__(self, params: dict, score=None):
        self.params = params
        self.score = score
        self.threshold = float(params.get("threshold", 1.0e-4))
        self.target_low = float(params.get("target_ratio_low", 0.20))
        self.target_high = float(params.get("target_ratio_high", 0.30))
        self.warmup_iters = int(params.get("warmup_iters", 0))
        self.warmup_start_ratio = float(params.get("warmup_start_ratio", 1.0))
        self.threshold_max = float(params.get("threshold_max", 0.0))
        self.block_size = int(params.get("block_size", 1))
        self.grad_norm_bypass = float(params.get("grad_norm_bypass", 0.0))
        self._adapt_iter = 0
        self.last_ratio = None
        self._forced_counts = None    # always-sync masks are static; cache counts

        if self.score is not None and not getattr(self.score, "has_threshold_path", False):
            raise ValueError(
                "threshold selection requires a score with a fused threshold path"
            )

    # -- selection ---------------------------------------------------------

    def select(self, ctx):
        """Return per-bucket element-level uint8 masks, OR-merged across DP.

        ``ctx.grad_views`` are the local accumulated bucket grads (the
        'local_or_merged' scoring input).
        """
        masks = self.score.threshold_mask(ctx.grad_views, self.threshold, self.block_size)
        # v1 freezes element-level support; block masks are a perf option the
        # historical scheme exposed, kept out of the first cut.
        assert self.block_size == 1, "block_size > 1 support not wired in v1"

        # Always-sync layers are forced into every support.
        for m, forced in zip(masks, ctx.always_sync_masks):
            if forced is not None:
                m |= forced

        # Global OR-merge: one MAX allreduce per bucket, in place on the
        # uint8 masks. MAX of 0/1 is exactly OR and cannot wrap at ANY DP
        # world size (a uint8 SUM wraps mod 256 at DP >= 256, identically on
        # all ranks, so no consistency layer could ever catch it). MAX also
        # stays inside NCCL's supported dtype/op set — int16 widening is NOT
        # an option here, NCCL rejects Short outright.
        if ctx.comm_group is not None and torch.distributed.is_initialized():
            for m in masks:
                torch.distributed.all_reduce(
                    m, op=torch.distributed.ReduceOp.MAX, group=ctx.comm_group
                )
        return masks

    # -- adaptive controller (engine calls once per reselection) -----------

    def adapt(self, masks, always_sync_masks):
        """Update the threshold from the just-measured selected ratio.

        Port of _adapt_threshold: ratio excludes always-sync positions from
        both numerator and denominator so the sparse portion independently
        hits the target band. The selected counts are reduced device-side
        into ONE accumulator (a single host transfer per reselection), and
        the static always-sync counts are transferred once and cached.
        """
        if not masks:
            return
        if (self._forced_counts is None
                or len(self._forced_counts) != len(always_sync_masks)):
            self._forced_counts = [None] * len(always_sync_masks)

        masked_acc = torch.zeros((), dtype=torch.int64,
                                 device=masks[0].device)
        total_elems = 0
        always_elems = 0
        for i, (m, forced) in enumerate(zip(masks, always_sync_masks)):
            masked_acc += m.sum(dtype=torch.int64)
            n_total = m.numel()
            n_always = self._forced_counts[i]
            if n_always is None:
                n_always = (int(forced.sum(dtype=torch.int64).item())
                            if forced is not None else 0)
                n_always = min(n_always, n_total)
                self._forced_counts[i] = n_always
            always_elems += n_always
            total_elems += n_total - n_always

        if total_elems == 0:
            return
        avg_ratio = (int(masked_acc.item()) - always_elems) / total_elems
        self.last_ratio = avg_ratio
        self._adapt_iter += 1

        target_low, target_high = self.target_low, self.target_high
        if self.warmup_iters > 0:
            progress = min(1.0, self._adapt_iter / self.warmup_iters)
            s = self.warmup_start_ratio
            target_low = target_low + (s - target_low) * (1.0 - progress)
            target_high = target_high + (s - target_high) * (1.0 - progress)
        target_mid = (target_low + target_high) / 2.0

        if avg_ratio > target_high:
            factor = 1.0 + (avg_ratio - target_mid) * _ADAPT_FACTOR
            factor = min(factor, _INCREASE_CAP)
            self.threshold *= factor
        elif avg_ratio < target_low:
            factor = 1.0 - (target_mid - avg_ratio) * _ADAPT_FACTOR
            factor = max(factor, _DECREASE_FLOOR)
            self.threshold *= factor

        if self.threshold_max > 0.0 and self.threshold > self.threshold_max:
            self.threshold = self.threshold_max

    def wants_force_full(self, grad_norm) -> bool:
        """Grad-norm bypass: force the next step full (globally consistent input)."""
        if self.grad_norm_bypass <= 0.0 or grad_norm is None:
            return False
        return grad_norm > self.grad_norm_bypass
