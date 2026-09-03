# Copyright (c) 2026, SGA framework contributors.
"""Pure-torch fallbacks for the scheme-agnostic masked ops.

Physically isolated from ``sga/kernels.py`` (Triton kernels + dispatch only)
so CPU-only callers never touch Triton import paths. Each fallback is the
elementwise-equivalent reference of its fused kernel (bitwise contracts are
documented per op and pinned by tests/test_kernels_gpu.py).
"""

import torch


def masked_norm_sq_contribution_torch(grad_flat: torch.Tensor,
                                      mask_flat: torch.Tensor,
                                      total_norm_sq: torch.Tensor) -> None:
    """``sum(grad[mask]**2)`` accumulated into the fp32 scalar."""
    masked = grad_flat[mask_flat.bool()]
    total_norm_sq += masked.float().pow(2).sum()


def scatter_selected_torch(dst: torch.Tensor, indices: torch.Tensor,
                           src: torch.Tensor) -> None:
    """``dst[idx[i]] = src[i]`` via advanced indexing; indices arrive as
    int32 from the comm protocol and torch advanced indexing requires
    int64, hence the ``.long()`` cast."""
    dst[indices.long()] = src


def pack_selected_torch(grad_flat: torch.Tensor, indices: torch.Tensor,
                        values: torch.Tensor, factor: float) -> None:
    """``values[i] = grad[idx[i]] * factor; grad[idx[i]] = 0`` (one pass,
    reference form). Bitwise-equal to the fused Triton pack: one fp32
    multiply per element either way, the zero store is a pure store."""
    sel = grad_flat[indices.long()]
    if factor != 1.0:
        sel = sel * factor
    values.copy_(sel)
    grad_flat[indices.long()] = 0.0


def compact_norm_sq_contribution_torch(values: torch.Tensor,
                                       total_norm_sq: torch.Tensor) -> None:
    """``sum(values**2)`` accumulated into the fp32 scalar (reference form).
    torch's fixed-shape reduction tree is deterministic run-to-run."""
    total_norm_sq += values.float().pow(2).sum()
