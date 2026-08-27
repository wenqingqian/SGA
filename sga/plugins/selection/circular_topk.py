# Copyright (c) 2026, SGA framework contributors.
"""Circular topk selection (historical radius scheme).

Strict-ratio global topk over the EF-materialized update magnitudes:
``k = max(1, round(n_total * ratio%))``. Default selector is the bit-pattern
histogram (ported verbatim from scheme_radius/optimizer.py, survey S3):
per-rank histograms of the top 16 bits of |update| fp32 bit patterns are
SUM-allreduced over the DP-CP group (256KB instead of a full-model mask
OR-merge); with replicated inputs the merged histogram is exactly ``world``
times the local one and the select math is scale-invariant, so every rank
derives the identical bin and threshold BY CONSTRUCTION (tied boundary
values are all selected — masks ``|update| >= thresh``). The opt-in
``exact_kthvalue`` fallback (full-model cat + torch.kthvalue + mask
OR-merge) is kept for A/B comparison. The default path does NOT OR-merge
the final masks: it leans on bitwise-replicated inputs plus the engine's
layer-2 abort on divergence instead — merging there too would trade a
crash for consistency at the cost of an extra uint8 allreduce per
reselection, which v1 declines deliberately.

"circular": the support is regenerated once per cycle period T at the
full-sync boundary and persisted unchanged in between; non-topk coordinates
accumulate as residual (the EF's reset contract folds it at the next
boundary). Selection ALWAYS runs after the EF applied the update and while
the materialized update still sits in the main-grad copy.
"""

import torch

from sga.plugins import SelectionDecl

DECL = SelectionDecl(
    name="circular_topk",
    output_form="mask",
    scope="global_ranked",
)

# Top 16 bits of a non-negative fp32 bit pattern span [0, 32768] (+1 spare bin).
_HIST_BINS = 32769


def _abs_bit_top16(g: torch.Tensor) -> torch.Tensor:
    """Top 16 bits of the |g| bit pattern (int32, monotone in |g|)."""
    return (g.view(torch.int32) & 0x7FFFFFFF) >> 16


def _histogram_select_threshold(hist, k, world, grads):
    """Exact k-th-largest |grad| threshold from a bit-pattern histogram.

    ``hist`` bins the top 16 bits of the |grad| bit patterns; when merged over
    ``world`` replica ranks it is ``world`` times the local histogram and ``k``
    is in local units (the bin index is scale-invariant). Non-negative fp32
    values sort identically to their int32 bit patterns, so an exact
    ``torch.kthvalue`` on the candidate bin alone yields the bit-exact same
    threshold VALUE as over all elements.

    Returns the threshold as a Python float.
    """
    cum = torch.cumsum(hist, 0)
    total = int(cum[-1].item())
    r = total - k * world + 1
    b = int(torch.searchsorted(cum, r).item())
    n_above = (total - int(cum[b].item())) // world  # local elements in bins > b

    # Exact kthvalue within bin b's candidates only (replicas: the local
    # candidate set equals the global one).
    cands = []
    for g in grads:
        top16 = _abs_bit_top16(g)
        c = g[top16 == b]
        if c.numel() > 0:
            cands.append(c.abs_())
        del top16
    assert cands, "histogram select found no candidates (histogram/candidates mismatch)"
    cand = torch.cat(cands) if len(cands) > 1 else cands[0]
    kk = cand.numel() - (k - n_above) + 1
    assert 1 <= kk <= cand.numel(), (
        f"histogram select kk out of range: kk={kk}, cand={cand.numel()}, "
        f"k={k}, n_above={n_above}"
    )
    return torch.kthvalue(cand, kk).values.item()


def _select_masks_histogram(grads, k, comm_group):
    """Exact k-th-largest |update| masks via the bit-pattern histogram."""
    device = grads[0].device
    hist = torch.zeros(_HIST_BINS, dtype=torch.int64, device=device)
    for g in grads:
        top16 = _abs_bit_top16(g)
        hist += torch.bincount(top16, minlength=_HIST_BINS)
        del top16

    if comm_group is not None and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            hist, op=torch.distributed.ReduceOp.SUM, group=comm_group
        )
        world = torch.distributed.get_world_size(comm_group)
    else:
        world = 1

    thresh = _histogram_select_threshold(hist, k, world, grads)
    masks = [(g.abs() >= thresh).to(torch.uint8) for g in grads]
    return thresh, masks


def _select_masks_exact_kthvalue(grads, k, comm_group):
    """Control path: cat all updates + torch.kthvalue + mask OR-merge.

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


class CircularTopkSelection:

    decl = DECL

    def __init__(self, params: dict, score=None):
        self.params = params
        self.score = score
        self.ratio_percent = float(params["ratio_percent"])
        assert 0.0 < self.ratio_percent < 100.0, (
            f"ratio_percent must be in (0, 100), got {self.ratio_percent}"
        )
        self.exact_kthvalue = bool(params.get("exact_kthvalue", False))
        self.last_threshold = None

    def k_for(self, n_total: int) -> int:
        return max(1, int(round(n_total * self.ratio_percent / 100.0)))

    def select(self, ctx):
        """Return per-bucket element-level uint8 masks from the global topk.

        ``ctx.grad_views`` are the EF-materialized update views (the 'synced'
        scoring input; selection runs only at full-sync cycle boundaries).
        """
        k = self.k_for(sum(g.numel() for g in ctx.grad_views))
        if self.exact_kthvalue:
            thresh, masks = _select_masks_exact_kthvalue(ctx.grad_views, k, ctx.comm_group)
        else:
            thresh, masks = _select_masks_histogram(ctx.grad_views, k, ctx.comm_group)
        self.last_threshold = thresh
        return masks
