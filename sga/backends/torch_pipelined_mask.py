# Copyright (c) 2026, SGA framework contributors.
"""Mask-form pipelined sparse allreduce backend.

Ported from the historical fork's ``masked_all_reduce_pipelined`` (common.py)
with one framework-level change — the payload-scaling decision. The dense
path pre-scales the WHOLE bucket buffer
by ``gradient_scaling_factor`` before its SUM allreduce, which would scale
the residual on sparse steps once per cycle into oblivion. Here ONLY the
extracted payload is scaled, so communicated coordinates stay mathematically
equal to their dense-path values while uncommunicated coordinates keep their
raw accumulation.

Pipelining insight: NCCL handles
empty tensors gracefully, so no ``.sum().item()`` checks ever force a
GPU-CPU sync. Slice i's extraction kernel overlaps slice i-1's NCCL op
(plain ``async_op=True``; the completion contract is call-flow
synchronous).
"""

import torch

from sga.plugins import BackendDecl, expand_plugin_params

DECL = BackendDecl(
    name="torch_pipelined_mask",
    consumes="mask",
)


class TorchPipelinedMaskBackend:
    """Consumes the mask form of a bucket's support."""

    decl = DECL

    def __init__(self, params: dict):
        params = expand_plugin_params(
            "sga_backend_", ("num_slices",), dict(params),
            "sga_backend (torch_pipelined_mask)")
        self.num_slices = int(params["num_slices"]) or 2

    def allreduce_sparse(self, grad_data: torch.Tensor, support,
                         comm_group, gradient_scaling_factor: float,
                         reduce_op, key: int = None) -> None:
        """Sparse allreduce of ``grad_data`` restricted to the support mask.

        ``support`` is a ``sga.support.BucketSupport`` (this backend consumes
        the mask form; ``key`` identifies the bucket for backends that keep
        per-bucket state, unused here). ``grad_data`` is a 1-D fp32 bucket
        view modified in place. The element payload extracted from each slice
        is scaled by ``gradient_scaling_factor`` (payload-only; never the
        whole buffer) before the SUM allreduce, replicating dense-path math
        on the communicated coordinates.
        """
        mask_u8 = support.mask_u8
        element_mask = mask_u8 > 0
        numel = grad_data.numel()
        num_slices = self.num_slices
        if num_slices <= 1 or numel == 0:
            selected = grad_data[element_mask]
            if selected.numel() > 0:
                if reduce_op == torch.distributed.ReduceOp.SUM and gradient_scaling_factor != 1.0:
                    selected *= gradient_scaling_factor
                torch.distributed.all_reduce(selected, op=reduce_op, group=comm_group)
                grad_data[element_mask] = selected
            return

        base = numel // num_slices
        boundaries = [i * base for i in range(num_slices)] + [numel]

        def _scale(payload: torch.Tensor) -> torch.Tensor:
            if (reduce_op == torch.distributed.ReduceOp.SUM
                    and gradient_scaling_factor != 1.0):
                payload *= gradient_scaling_factor
            return payload

        # Slice 0: extract and launch async allreduce.
        s0, e0 = boundaries[0], boundaries[1]
        prev_mask_slice = element_mask[s0:e0]
        prev_selected = _scale(grad_data[s0:e0][prev_mask_slice])
        prev_work = torch.distributed.all_reduce(
            prev_selected, op=reduce_op, group=comm_group, async_op=True
        )
        prev_s, prev_e = s0, e0

        for i in range(1, num_slices):
            si, ei = boundaries[i], boundaries[i + 1]
            cur_mask_slice = element_mask[si:ei]
            # Extract launches a CUDA kernel — overlaps with prev allreduce.
            cur_selected = _scale(grad_data[si:ei][cur_mask_slice])

            # Wait for previous allreduce and scatter back.
            prev_work.wait()
            grad_data[prev_s:prev_e][prev_mask_slice] = prev_selected

            prev_work = torch.distributed.all_reduce(
                cur_selected, op=reduce_op, group=comm_group, async_op=True
            )
            prev_selected = cur_selected
            prev_mask_slice = cur_mask_slice
            prev_s, prev_e = si, ei

        prev_work.wait()
        grad_data[prev_s:prev_e][prev_mask_slice] = prev_selected
