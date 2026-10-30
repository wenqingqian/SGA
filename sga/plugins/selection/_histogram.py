# Copyright (c) 2026, SGA framework contributors.
"""Bit-pattern histogram top-k select, in three geometry tiers.

Ported from an earlier internal scheme: per-rank histograms of the top N
bits of the |x| fp32 bit patterns may be SUM-allreduced over the comm group
— the ``global_allreduce`` gate, wired from the
``sga_selection_topk_histogram_add_global_allreduce`` flag (default false;
True turns the merge on). With replicated inputs the merged histogram is
exactly ``world`` times the local one and the select math is
scale-invariant, so every rank derives the identical bin/threshold BY
CONSTRUCTION with or without the merge (tied boundary values are all
selected) — the collective is skippable on the local histogram alone, and
the engine's mask consistency check aborts if the replication premise ever
breaks. Non-negative fp32 values sort identically to their int32 bit
patterns, so an exact ``torch.kthvalue`` on the crossing bin's candidates
alone yields the bit-exact same threshold as over all elements.

Tiers (``mode``):

- ``exact``    (shift 16, 32769 bins, stride 1): crossing-bin candidate
  refine with an exact kthvalue; bit-exact same threshold as the kthvalue
  control path.
- ``balanced`` (shift 16, 32769 bins, stride 8): fine bins but the
  histogram is built on a 1/8 stride subsample, and the crossing bin is
  used directly (mask = ``top >= bin``) — no candidate extraction.
- ``fast``     (shift 19, 4097 bins, stride 8): coarse 13-bit bins + 1/8
  subsample; smallest allreduce payload (32KB vs 256KB).

The approximate tiers carry an over-selection BIAS, not a strict superset
property: the whole crossing bin is included relative to the measured
(subsampled) histogram, but the mask is applied to the full set, so stride
subsampling fluctuation allows a small under-selection relative to exact.
The two-sided criterion (pinned by tests/test_xp_capability.py) is
|n_approx - n_exact| / n < 0.5pp (balanced) / 2pp (fast); over-selection
is a bias, not a strict-superset guarantee.

Every pass is chunked at ``_HIST_CHUNK`` elements: the int32 bit-pattern
view/copy of the largest buckets is GB-scale on top of the resident
sparse-comm buffers. The chunk size is a multiple of the sample
stride, so the chunked histogram/subsample is bitwise identical to the
unchunked one.
"""

import torch

# Histogram bin geometry: the top N bits of a non-negative fp32 bit pattern
# (sign stripped) span [0, 2**(N-1)] (+1 spare bin).
_HIST_SHIFT = 16
_HIST_BINS = 32769
# Coarse geometry (top 13 bits) for mode='fast'.
_HIST_SHIFT_COARSE = 19
_HIST_BINS_COARSE = 4097
# The approximate modes (balanced/fast) bin a deterministic stride subsample
# (every Nth element) instead of the full gradient.
_HIST_SAMPLE_STRIDE = 8
_HIST_CHUNK = 64 << 20
# Chunked-subsample must be bitwise identical to global-subsample: that holds
# only while the chunk is a stride multiple (patched to a non-multiple the
# histogram silently changes — pinned here at import).
assert _HIST_CHUNK % _HIST_SAMPLE_STRIDE == 0, (
    "_HIST_CHUNK must be a multiple of _HIST_SAMPLE_STRIDE "
    "(chunked-subsample == global-subsample bitwise identity)"
)

# mode -> (bit shift, bin count, subsample stride)
_HIST_MODES = {
    "exact": (_HIST_SHIFT, _HIST_BINS, 1),
    "balanced": (_HIST_SHIFT, _HIST_BINS, _HIST_SAMPLE_STRIDE),
    "fast": (_HIST_SHIFT_COARSE, _HIST_BINS_COARSE, _HIST_SAMPLE_STRIDE),
}


def _abs_bit_top(g: torch.Tensor, shift: int = _HIST_SHIFT) -> torch.Tensor:
    """Top bits of the |g| bit pattern (int32, monotone in |g|)."""
    return (g.view(torch.int32) & 0x7FFFFFFF) >> shift


def _abs_bit_top16(g: torch.Tensor) -> torch.Tensor:
    """Top 16 bits of the |g| bit pattern (compat/short name)."""
    return _abs_bit_top(g, _HIST_SHIFT)


def _bin_edge_float(b: int, shift: int) -> float:
    """Bin b's lower edge as a Python float (representative threshold)."""
    bits = (b << shift) & 0x7FFFFFFF
    return torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()


def _sampled_count(n, stride, phase=0):
    """Count indices ``phase + i*stride`` in one length-``n`` view."""
    assert stride > 0 and 0 <= phase < stride
    return len(range(phase, n, stride))


def _accumulate_histogram(hist, g, shift, stride, phase=0):
    """Add one view's phased, chunked subsample to ``hist`` in place."""
    assert _HIST_CHUNK % stride == 0
    for s in range(0, g.numel(), _HIST_CHUNK):
        top = _abs_bit_top(g[s:s + _HIST_CHUNK], shift)
        sampled = top[phase::stride] if stride > 1 else top
        hist += torch.bincount(
            sampled, minlength=hist.numel()).to(hist.device)
        del top


def _build_histogram(grads, shift: int, nbins: int, stride: int) -> torch.Tensor:
    """Chunked per-rank histogram of the top-``shift`` bits of |g|."""
    device = grads[0].device
    hist = torch.zeros(nbins, dtype=torch.int64, device=device)
    for g in grads:
        _accumulate_histogram(hist, g, shift, stride)
    return hist


def _balanced_crossing_bin(hist, n_total, n_sampled, k, *, replicas=1):
    """Find the approximate crossing bin in local catalog units.

    ``replicas`` is the number of identical online inputs merged into
    ``hist``. Disjoint replay shards reconstruct one catalog and therefore
    pass one, not their shard count.
    """
    r = max(1, round(n_sampled * (n_total - k + 1) / n_total)) * replicas
    return int(torch.searchsorted(torch.cumsum(hist, 0), r).item())


def _histogram_select_threshold(hist, k, world, grads, shift=_HIST_SHIFT) -> float:
    """Exact k-th-largest |grad| threshold from a bit-pattern histogram.

    ``hist`` bins the top ``shift`` bits of the |grad| bit patterns; when
    merged over ``world`` replica ranks it is ``world`` times the local
    histogram and ``k`` is in local units (the bin index is
    scale-invariant).

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
        for s in range(0, g.numel(), _HIST_CHUNK):
            gc = g[s:s + _HIST_CHUNK]
            c = gc[_abs_bit_top(gc, shift) == b]
            if c.numel() > 0:
                cands.append(c.abs_())
    assert cands, "histogram select found no candidates (histogram/candidates mismatch)"
    cand = torch.cat(cands) if len(cands) > 1 else cands[0]
    kk = cand.numel() - (k - n_above) + 1
    assert 1 <= kk <= cand.numel(), (
        f"histogram select kk out of range: kk={kk}, cand={cand.numel()}, "
        f"k={k}, n_above={n_above}"
    )
    return torch.kthvalue(cand, kk).values.item()


def select_masks_histogram(grads, k, comm_group, mode="exact",
                           global_allreduce=True):
    """Topk masks on the given per-element scores via a bit-pattern histogram.

    Returns (threshold, masks): per-grad uint8 masks; ``threshold`` is the
    exact value for ``exact`` mode, otherwise the crossing bin's lower edge
    (used for logging and as the adaptive path's tau bootstrap). With
    ``global_allreduce=False`` the per-rank histogram is used as-is (world=1
    semantics) — under the engine's replicated full-sync scores this derives
    the identical threshold the merged form would. The keyword defaults to
    True so direct callers keep the merged form; the flag's default False
    is applied by ``TopkSelection`` (topk.py).
    """
    try:
        shift, nbins, stride = _HIST_MODES[mode]
    except KeyError:
        raise ValueError(f"unknown histogram_mode: {mode!r}") from None

    hist = _build_histogram(grads, shift, nbins, stride)
    if (global_allreduce and comm_group is not None
            and torch.distributed.is_initialized()):
        torch.distributed.all_reduce(
            hist, op=torch.distributed.ReduceOp.SUM, group=comm_group
        )
        world = torch.distributed.get_world_size(comm_group)
    else:
        # No merge requested (global_allreduce=False, the flag default) or
        # no group/dist to merge over: select on the local histogram
        # (world=1) — same threshold as the merged form (module docstring).
        world = 1

    if mode == "exact":
        thresh = _histogram_select_threshold(hist, k, world, grads, shift)
        masks = []
        for g in grads:
            m = torch.empty(g.numel(), dtype=torch.uint8, device=g.device)
            for s in range(0, g.numel(), _HIST_CHUNK):
                gc = g[s:s + _HIST_CHUNK]
                m[s:s + _HIST_CHUNK] = gc.abs() >= thresh
            masks.append(m)
        return thresh, masks

    # Approximate: replicated online inputs scale the rank with the merged
    # histogram; disjoint replay shards use replicas=1 in the same helper.
    n_total = sum(g.numel() for g in grads)
    n_sampled = sum(_sampled_count(g.numel(), stride) for g in grads)
    b = _balanced_crossing_bin(
        hist, n_total, n_sampled, k, replicas=world)
    masks = []
    for g in grads:
        m = torch.empty(g.numel(), dtype=torch.uint8, device=g.device)
        for s in range(0, g.numel(), _HIST_CHUNK):
            gc = g[s:s + _HIST_CHUNK]
            m[s:s + _HIST_CHUNK] = _abs_bit_top(gc, shift) >= b
        masks.append(m)
    return _bin_edge_float(b, shift), masks


def _block_reduce_views(grads, x, reduce):
    """Per-grad block scores: |g| reduced over contiguous x-element blocks.

    The tail block of a grad view (numel % x != 0) reduces over its REAL
    length, so mean never dilutes and sum-bias never discriminates it.
    Chunked at x-aligned slices of ``_HIST_CHUNK``: |g| never materializes
    as a second full-size copy.
    """
    chunk = _HIST_CHUNK - (_HIST_CHUNK % x)
    out = []
    for g in grads:
        # An empty view enters no chunk (parts would stay empty and the
        # cat below would raise); emit the empty block-score view directly,
        # matching the element path's tolerance of zero-numel views.
        if g.numel() == 0:
            out.append(torch.empty(0, device=g.device))
            continue
        parts = []
        for s in range(0, g.numel(), chunk):
            c = g[s:s + chunk].abs()
            nb, rem = divmod(c.numel(), x)
            if nb:
                b = c[:nb * x].view(nb, x)
                parts.append(b.mean(dim=1) if reduce == "mean"
                             else b.amax(dim=1))
            if rem:
                # only the param's LAST chunk can have a partial block
                t = c[nb * x:]
                parts.append(
                    (t.mean() if reduce == "mean" else t.amax()).reshape(1))
            del c
        out.append(torch.cat(parts) if len(parts) > 1 else parts[0])
    return out


class HistogramSelect:
    """method=histogram impl: ``select(grads, k, comm_group, warmup_target)``.

    ``warmup_target`` is adaptive-only and ignored here. Threshold semantics
    by tier: exact kth value (``exact``) or the crossing bin's lower edge
    (``balanced``/``fast``; also the adaptive tau's bootstrap input).

    ``block_size`` > 1 selects in contiguous blocks: each grad view is
    block-reduced (``block_reduce`` mean|max of |score| over the block's
    real length), the same histogram machinery runs on the ~n/x block
    scores with ``k_blocks = round(k / block_size)``, and the selected
    block masks expand back to element masks (tail block clamped). The
    threshold is in block-score units then, not |score| units.
    """

    def __init__(self, mode: str = "exact", block_size: int = 1,
                 block_reduce: str = "mean", global_allreduce: bool = True):
        if mode not in _HIST_MODES:
            raise ValueError(
                f"unknown histogram_mode: {mode!r} "
                f"(expected one of {sorted(_HIST_MODES)})"
            )
        if (not isinstance(block_size, int) or isinstance(block_size, bool)
                or block_size < 1 or block_size > 1024
                or block_size & (block_size - 1)):
            raise ValueError(
                f"block_size must be a power of two in [1, 1024], got "
                f"{block_size!r}")
        if block_reduce not in ("mean", "max"):
            raise ValueError(
                f"block_reduce must be 'mean' or 'max', got "
                f"{block_reduce!r}")
        if not isinstance(global_allreduce, bool):
            raise ValueError(
                f"global_allreduce must be a bool, got {global_allreduce!r}")
        self.mode = mode
        self.block_size = block_size
        self.block_reduce = block_reduce
        self.global_allreduce = global_allreduce

    def select(self, grads, k, comm_group, warmup_target=None):
        if self.block_size == 1:
            return select_masks_histogram(grads, k, comm_group,
                                          mode=self.mode,
                                          global_allreduce=self.global_allreduce)
        x = self.block_size
        k_blocks = max(1, int(round(k / x)))
        bviews = _block_reduce_views(grads, x, self.block_reduce)
        thresh, bmasks = select_masks_histogram(bviews, k_blocks,
                                                comm_group, mode=self.mode,
                                                global_allreduce=self.global_allreduce)
        masks = [bm.repeat_interleave(x)[:g.numel()]
                 for bm, g in zip(bmasks, grads)]
        return thresh, masks
