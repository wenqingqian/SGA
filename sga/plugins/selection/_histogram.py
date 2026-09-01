# Copyright (c) 2026, SGA framework contributors.
"""Bit-pattern histogram top-k select, in three geometry tiers.

Ported verbatim from scheme_xp (survey S3; the approximate tiers' superset
ruling is measured lore): per-rank histograms of the top N bits of the
|x| fp32 bit patterns are SUM-allreduced over the comm group — with
replicated inputs the merged histogram is exactly ``world`` times the local
one and the select math is scale-invariant, so every rank derives the
identical bin/threshold BY CONSTRUCTION (tied boundary values are all
selected). Non-negative fp32 values sort identically to their int32 bit
patterns, so an exact ``torch.kthvalue`` on the crossing bin's candidates
alone yields the bit-exact same threshold as over all elements.

Tiers (``mode`` — the earlier geometry table):

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
The executable criterion (from the migration source's own production tests)
is two-sided: |n_approx - n_exact| / n < 0.5pp (balanced) / 2pp (fast); the
lore claim "over-select (never under-select)" in the legacy docstring is
idealized (exact only under a full-set histogram) and must not be read as a
hard invariant.

Every pass is chunked at ``_HIST_CHUNK`` elements: the int32 bit-pattern
view/copy of a full 1.2B-element bucket is GB-scale, and stacked on the
legacy scheme's (since-deleted) non-update score side buffers and the
persistent sparse-comm buffers it OOM'd the legacy scheme's second
full-sync select at 1B scale. The chunk size is a multiple of the sample
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


def _build_histogram(grads, shift: int, nbins: int, stride: int) -> torch.Tensor:
    """Chunked per-rank histogram of the top-``shift`` bits of |g|."""
    device = grads[0].device
    hist = torch.zeros(nbins, dtype=torch.int64, device=device)
    for g in grads:
        for s in range(0, g.numel(), _HIST_CHUNK):
            top = _abs_bit_top(g[s:s + _HIST_CHUNK], shift)
            hist += torch.bincount(top[::stride] if stride > 1 else top,
                                   minlength=nbins)
            del top
    return hist


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
    # candidate set equals the global one). Chunked extraction: a full-bucket
    # int32 bit-pattern copy is GB-scale for the largest buckets.
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


def select_masks_histogram(grads, k, comm_group, mode="exact"):
    """Topk masks on the given per-element scores via a bit-pattern histogram.

    Returns (threshold, masks): per-grad uint8 masks; ``threshold`` is the
    exact value for ``exact`` mode, otherwise the crossing bin's lower edge
    (used for logging and as the adaptive path's tau bootstrap).
    """
    try:
        shift, nbins, stride = _HIST_MODES[mode]
    except KeyError:
        raise ValueError(f"unknown hist_mode: {mode!r}") from None

    hist = _build_histogram(grads, shift, nbins, stride)
    if comm_group is not None and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            hist, op=torch.distributed.ReduceOp.SUM, group=comm_group
        )
        world = torch.distributed.get_world_size(comm_group)
    else:
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

    # Approximate: find the crossing bin in local rank units (the merged
    # histogram is world × the local one, so the bin is scale-invariant).
    n_total = sum(g.numel() for g in grads)
    n_sampled = sum((g.numel() + stride - 1) // stride for g in grads)
    r = max(1, round(n_sampled * (n_total - k + 1) / n_total)) * world
    b = int(torch.searchsorted(torch.cumsum(hist, 0), r).item())
    masks = []
    for g in grads:
        m = torch.empty(g.numel(), dtype=torch.uint8, device=g.device)
        for s in range(0, g.numel(), _HIST_CHUNK):
            gc = g[s:s + _HIST_CHUNK]
            m[s:s + _HIST_CHUNK] = _abs_bit_top(gc, shift) >= b
        masks.append(m)
    return _bin_edge_float(b, shift), masks
