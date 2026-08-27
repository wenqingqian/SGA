# Copyright (c) 2026, SGA framework contributors.
"""Index-form sparse allreduce backend (radius-style path).

Ported from the historical fork's ``_indexed_sparse_allreduce``
(scheme_radius/grad_sync.py): gather via cached sorted int32 indices into a
PERSISTENT grow-only compact buffer, two pipelined async allreduces split at
the bucket midpoint, Triton ``scatter_selected`` writeback. The two slices
cover exactly the same element sets as the legacy bool-index pipeline, so
the NCCL reduction order is unchanged. Persistent buffers are the verified
engineering constraint (notes/sga-engine-structure.md section 7): reallocating
GB-scale gathers every cycle triggers allocator map/unmap storms.

Payload-only scaling as in the mask backend (frozen decision).
"""

import torch

from sga.kernels import scatter_selected
from sga.plugins import BackendDecl

DECL = BackendDecl(name="indexed_sparse", consumes="idx")


class IndexedSparseBackend:
    """Consumes the canonical sorted-idx form of a bucket's support."""

    def __init__(self, params: dict):
        self._gather_bufs: dict = {}  # id(bucket-state-owner) -> persistent buffer

    def _gather_buf(self, key: int, k: int, dtype, device) -> torch.Tensor:
        buf = self._gather_bufs.get(key)
        if buf is None or buf.numel() != k or buf.dtype != dtype or buf.device != device:
            # Grow-only reallocation: only when the selected count changed.
            buf = torch.empty(k, dtype=dtype, device=device)
            self._gather_bufs[key] = buf
        return buf

    def allreduce_sparse(self, grad_data: torch.Tensor, support, comm_group,
                         gradient_scaling_factor: float, reduce_op,
                         key: int) -> None:
        """Sparse allreduce of ``grad_data`` at ``support.idx`` positions.

        ``support`` is a ``sga.support.BucketSupport``; ``key`` identifies the
        owning bucket state for the persistent gather buffer.
        """
        # int32 canonical indices: refuse oversized buckets loudly instead
        # of wrapping silently (buckets ~40M elements today, 2^31 is far).
        assert grad_data.numel() < 2 ** 31, (
            f"int32 support indices assume < 2^31 bucket elements, "
            f"got {grad_data.numel()} (bucket key {key})"
        )
        indices = support.idx
        k = indices.numel()
        if k == 0:
            return

        gather_buf = self._gather_buf(key, k, grad_data.dtype, grad_data.device)
        n0 = support.split
        slices = ((0, n0), (n0, k))

        def _scale(payload: torch.Tensor) -> torch.Tensor:
            if (reduce_op == torch.distributed.ReduceOp.SUM
                    and gradient_scaling_factor != 1.0):
                payload *= gradient_scaling_factor
            return payload

        # Slice 0: gather and launch async allreduce.
        s0, e0 = slices[0]
        prev_idx = indices[s0:e0]
        prev_sel = gather_buf[s0:e0]
        prev_work = None
        if e0 > s0:
            torch.index_select(grad_data, 0, prev_idx, out=prev_sel)
            _scale(prev_sel)
            prev_work = torch.distributed.all_reduce(
                prev_sel, op=reduce_op, group=comm_group, async_op=True
            )

        # Slice 1: gather (overlaps slice-0 NCCL), then wait/scatter/launch.
        s1, e1 = slices[1]
        cur_idx = indices[s1:e1]
        cur_sel = gather_buf[s1:e1]
        cur_work = None
        if e1 > s1:
            torch.index_select(grad_data, 0, cur_idx, out=cur_sel)
            _scale(cur_sel)

        if prev_work is not None:
            prev_work.wait()
            scatter_selected(grad_data, prev_idx, prev_sel)
        if e1 > s1:
            cur_work = torch.distributed.all_reduce(
                cur_sel, op=reduce_op, group=comm_group, async_op=True
            )
        if cur_work is not None:
            cur_work.wait()
            scatter_selected(grad_data, cur_idx, cur_sel)
