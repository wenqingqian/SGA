# Copyright (c) 2026, SGA framework contributors.
"""The Support object: which gradient coordinates are communicated this cycle.

Representation is settled project lore (2026-08-27): idx and mask are two
forms of the same support; the STORED form is the mask (the selection's
direct output, written into the engine's ping-pong pool), while the
position-sorted index array is derived lazily from it by ``compact_mask_u8``
and pooled on demand. Backends declare which form they consume. The
consistency machinery (layer-1 counts, layer-2 dual checksums) computes
directly from the MASK — chunked count/checksum kernels, no index
materialization — so both what is verified and what is communicated reduce
to the same underlying mask (and under shadow the idx is never materialized
at all).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

# Mask-compaction granularity: mask bytes processed per nonzero chunk. A
# full-mask nonzero on a 1.2B-element bucket allocates a fresh GB-scale
# int64 output every cycle; under expandable_segments those alloc/free
# cycles hit cuMemMap/cuMemSetAccess ioctl storms (the legacy scheme
# observed multi-second stalls at the second mask rebuild). Chunking keeps
# the transient int64 outputs small enough for the caching allocator.
_MASK_COMPACT_CHUNK = 32 << 20
# Index/gather buffer growth granularity: 4M elements (16MB fp32).
_GATHER_BUF_CHUNK = 4 << 20


class IndexPool:
    """Grow-only int32 buffer shared across cycles for one bucket's idx.

    Reallocating the sorted-index array every cycle is the same allocator
    storm the compaction chunking avoids; the capacity only ever grows
    (rounded up to _GATHER_BUF_CHUNK) and consumers take buf[:k] views.

    Lifecycle constraint: growth REPLACES ``buf``, stranding prior
    ``buf[:k]`` views over freed storage — safe only because the engine
    drops a cycle's old Support before compaction can grow the pool, so
    nothing reads a dropped support's idx afterwards.
    """

    def __init__(self):
        self.buf: Optional[torch.Tensor] = None

    def get(self, k: int, device) -> torch.Tensor:
        if self.buf is None or self.buf.numel() < k or self.buf.device != device:
            cap = -(-k // _GATHER_BUF_CHUNK) * _GATHER_BUF_CHUNK
            if self.buf is not None and self.buf.device == device:
                cap = max(cap, self.buf.numel())  # never shrink
            self.buf = torch.empty(cap, dtype=torch.int32, device=device)
        return self.buf


def compact_mask_u8(mask_u8: torch.Tensor, pool: Optional[IndexPool] = None) -> torch.Tensor:
    """Sorted int32 indices of the set positions of ``mask_u8``.

    Bitwise equivalent to ``mask_u8.nonzero(as_tuple=True)[0].to(torch.int32)``
    (chunks are processed in order, so global ordering is preserved). Without
    a pool (small/unit-scale inputs) it is the plain one-shot nonzero; with
    one, counting is chunked (a whole-mask sum(dtype=int64) materializes a
    full int64 copy — 8 bytes/elem) and the result lives in the persistent
    grow-only buffer instead of a fresh allocation per cycle.

    Precondition: mask bytes must be exactly 0/1 (asserted against the
    count/fill agreement below).
    """
    n = mask_u8.numel()
    # int32 canonical indices: refuse oversized masks loudly instead of
    # wrapping silently. Guarded HERE too (eager activation-time compaction
    # runs earlier than any backend's own assert, so wrapped indices could
    # otherwise sit cached through the activate->sync window).
    assert n < 2 ** 31, f"int32 support indices assume < 2^31 elements, got {n}"
    if pool is None and n <= _MASK_COMPACT_CHUNK:
        return mask_u8.nonzero(as_tuple=True)[0].to(torch.int32)
    k = 0
    for s in range(0, n, _MASK_COMPACT_CHUNK):
        k += int(mask_u8[s:s + _MASK_COMPACT_CHUNK].sum().item())
    if pool is not None:
        buf = pool.get(k, mask_u8.device)
    else:
        buf = torch.empty(k, dtype=torch.int32, device=mask_u8.device)
    off = 0
    for s in range(0, n, _MASK_COMPACT_CHUNK):
        sub = mask_u8[s:s + _MASK_COMPACT_CHUNK].nonzero(as_tuple=True)[0]
        m = sub.numel()
        if m:
            buf[off:off + m].copy_(sub.add_(s))
            off += m
    # Count/fill disagreement is the 0/1 precondition failing: the count pass
    # SUMS bytes, so values > 1 would otherwise hand back uninitialized int32
    # slots as indices.
    assert off == k, (
        f"compact_mask_u8: mask bytes must be exactly 0/1 "
        f"(counted {k}, filled {off})"
    )
    return buf[:k]


@dataclass
class BucketSupport:
    """Support for one gradient bucket, built from a selection decision."""

    # Element-level uint8 mask at bucket granularity (the selection output).
    mask_u8: torch.Tensor
    # Persistent grow-only idx pool for this bucket (engine-owned); None in
    # unit tests / small-scale use falls back to one-shot nonzero.
    idx_pool: Optional[IndexPool] = field(default=None, repr=False)
    # Number of selected elements (host int; derived once and cached — the
    # mask never changes within a cycle, so per-step rescans are pure waste).
    n_selected: Optional[int] = field(default=None, repr=False)
    # Canonical sorted int32 indices of the selected elements (derived lazily).
    _idx: torch.Tensor = field(default=None, repr=False)
    # Midpoint split for two-slice pipelining (derived lazily, cached).
    _split: int = field(default=None, repr=False)

    @property
    def count(self) -> int:
        return int(self.mask_u8.numel())

    def num_selected(self) -> int:
        """Number of selected elements (cached host int after first call).

        Counted in chunks: above the chunk size a single count_nonzero
        materializes an int64 copy (8 bytes/elem — GiB-scale on real
        models) on this torch build; below it the kernel stays fused.
        """
        if self.n_selected is None:
            m = self.mask_u8
            n = 0
            chunk = 1 << 26  # 64M
            for lo in range(0, m.numel(), chunk):
                n += int(torch.count_nonzero(m[lo:lo + chunk]).item())
            self.n_selected = n
        return self.n_selected

    @property
    def idx(self) -> torch.Tensor:
        if self._idx is None:
            self._idx = compact_mask_u8(self.mask_u8, self.idx_pool)
        return self._idx

    @property
    def split(self) -> int:
        """Index into ``idx`` that splits it at the bucket midpoint."""
        if self._split is None:
            mid = self.mask_u8.numel() // 2
            self._split = int(torch.searchsorted(self.idx, mid).item())
        return self._split


@dataclass
class Support:
    """Per-bucket support snapshot, active for exactly one cycle."""

    buckets: list  # [BucketSupport] aligned with the engine's bucket order
    _total_selected: Optional[int] = field(default=None, repr=False)

    @property
    def total_selected(self) -> int:
        """Total selected elements across buckets (cached; each per-bucket
        count itself caches on its BucketSupport)."""
        if self._total_selected is None:
            self._total_selected = sum(b.num_selected() for b in self.buckets)
        return self._total_selected

    @property
    def total_elements(self) -> int:
        return sum(b.count for b in self.buckets)
