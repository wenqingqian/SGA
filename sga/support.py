# Copyright (c) 2026, SGA framework contributors.
"""The Support object: which gradient coordinates are communicated this cycle.

Representation is settled project lore (2026-08-27): idx and mask are two forms
of the same support, without preference; the canonical form is the
position-sorted index array (support comes from synchronized scoring state and
is stable within a cycle), and the mask form is derived and cached on demand.
Backends declare which form they consume; consistency checks always operate on
the canonical sorted idx so there is exactly one conversion relationship
between what is verified and what is communicated.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BucketSupport:
    """Support for one gradient bucket, built from a selection decision."""

    # Element-level uint8 mask at bucket granularity (the selection output).
    mask_u8: torch.Tensor
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
            self._idx = self.mask_u8.nonzero(as_tuple=True)[0].to(torch.int32)
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
