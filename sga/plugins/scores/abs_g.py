# Copyright (c) 2026, SGA framework contributors.
"""abs_g score: importance = |grad|.

Historical origin scheme's score. Two consumption modes:

- Fused threshold path (``threshold_mask``): ``mask = |grad| >= threshold``
  computed straight from the gradient tensors by a Triton kernel, without
  materializing the absolute values — this is the historical origin hot path
  (element-wise and block-wise OR variants ported verbatim from
  scheme_origin/kernels.py).
- Materialized ``values()`` for ranked selections (debug/comparisons; not
  used by the threshold selection in production).

scoring_input = 'local_or_merged': the threshold may be applied to the local
accumulated (partially synced) gradient state; global consistency comes from
the selection's mask OR-merge, exactly like the historical scheme. This is
what allows per-step reselection (T=1) without a full sync.
"""

from typing import Optional

import torch

from sga.plugins import ScoreDecl

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


DECL = ScoreDecl(name="abs_g", scoring_input="local_or_merged", has_threshold_path=True)


if _HAS_TRITON:

    @triton.jit
    def _mask_ge_kernel(
        grad_ptr,
        mask_ptr,
        threshold,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        STRIDE: tl.constexpr,
    ):
        """Element-wise mask: mask[i] = |grad[i]| >= threshold."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load grad as float32 for the comparison; supports bf16/fp16/fp32 inputs.
        vals = tl.load(grad_ptr + offsets * STRIDE, mask=mask, other=0.0)
        vals = tl.abs(vals.to(tl.float32))
        out = vals >= threshold
        tl.store(mask_ptr + offsets, out.to(tl.uint8), mask=mask)

    @triton.jit
    def _mask_ge_block_kernel(
        grad_ptr,
        mask_ptr,
        threshold,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        MASK_BLOCK_SIZE: tl.constexpr,
        STRIDE: tl.constexpr,
    ):
        """Block-wise mask: mask[j] = OR_{k}(|grad[j*B+k]| >= threshold)."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        grad_base = offsets * MASK_BLOCK_SIZE
        out = tl.zeros((BLOCK_SIZE,), dtype=tl.int1)
        for k in range(MASK_BLOCK_SIZE):
            grad_offsets = grad_base + k
            m = grad_offsets < n_elements
            vals = tl.load(grad_ptr + grad_offsets * STRIDE, mask=m, other=0.0)
            vals = tl.abs(vals.to(tl.float32))
            out = out | (vals >= threshold)

        n_out = (n_elements + MASK_BLOCK_SIZE - 1) // MASK_BLOCK_SIZE
        tl.store(mask_ptr + offsets, out.to(tl.uint8), mask=offsets < n_out)


def compute_mask(grad: torch.Tensor, threshold: float,
                 out: Optional[torch.Tensor] = None, block_size: int = 1) -> torch.Tensor:
    """Compute a local mask ``|grad| >= threshold`` (uint8 output).

    block_size=1: out numel == grad numel. block_size>1: one mask element per
    block (OR), out numel == ceil(n/block_size), fp32 grads only.
    """
    if grad.numel() == 0:
        if out is not None:
            return out
        return torch.empty(0, dtype=torch.uint8, device=grad.device)

    flat_grad = grad.view(-1)
    n = flat_grad.numel()

    if block_size == 1:
        if out is None:
            out = torch.empty(n, dtype=torch.uint8, device=grad.device)
        else:
            assert out.numel() == n and out.dtype == torch.uint8
        if _HAS_TRITON and flat_grad.is_cuda:
            BLOCK = 2048
            grid = (triton.cdiv(n, BLOCK),)
            _mask_ge_kernel[grid](
                flat_grad, out, threshold, n,
                BLOCK_SIZE=BLOCK, STRIDE=flat_grad.stride(0),
            )
        else:  # reference fallback (CPU tests)
            torch.abs(flat_grad.float(), out=None)
            out.copy_((flat_grad.float().abs() >= threshold).to(torch.uint8))
        return out

    assert block_size > 1
    if flat_grad.dtype != torch.float32:
        raise TypeError(f"block-wise mask only supports fp32 grad, got {flat_grad.dtype}")
    assert n % block_size == 0, (
        f"grad numel ({n}) must be divisible by block_size ({block_size})"
    )
    n_mask = n // block_size
    if out is None:
        out = torch.empty(n_mask, dtype=torch.uint8, device=grad.device)
    else:
        assert out.numel() == n_mask and out.dtype == torch.uint8
    if _HAS_TRITON and flat_grad.is_cuda:
        BLOCK = 1024
        grid = (triton.cdiv(n_mask, BLOCK),)
        _mask_ge_block_kernel[grid](
            flat_grad, out, threshold, n,
            BLOCK_SIZE=BLOCK, MASK_BLOCK_SIZE=block_size, STRIDE=flat_grad.stride(0),
        )
    else:  # reference fallback (CPU tests)
        blocked = flat_grad.view(n_mask, block_size).abs()
        out.copy_((blocked.max(dim=1).values >= threshold).to(torch.uint8))
    return out


class AbsGradScore:
    """Score plugin: |grad| with a fused threshold fast path."""

    decl = DECL

    has_threshold_path = True

    def __init__(self, params: dict):
        self.params = params

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        """Fused fast path: per-tensor uint8 masks ``|g| >= threshold``."""
        return [compute_mask(g, threshold, block_size=block_size) for g in grad_views]

    def values(self, grad_views):
        """Materialized score values (|g|); ranked-selection/debug path only."""
        return [g.abs() for g in grad_views]
