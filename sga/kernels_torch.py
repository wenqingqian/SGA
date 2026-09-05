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


# Block-fallback slot chunk: the (kb, x) position matrix costs 8 bytes per
# payload slot in int64; chunking keeps the reference path's transient
# small (the Triton path computes positions in registers).
_BLOCK_CHUNK = 1 << 16


def _block_positions(block_idx: torch.Tensor, x: int, lo: int, hi: int,
                     n: int):
    """(valid lane mask, clamped positions) for blocks [lo, hi): lane (j, i)
    maps to position block_idx[j] + i; lanes past n-1 are invalid (tail
    blocks read short)."""
    pos = (block_idx[lo:hi].long().unsqueeze(1)
           + torch.arange(x, device=block_idx.device))
    inb = pos < n
    return inb, pos.clamp_max(n - 1)


def gather_selected_blocks_torch(grad_flat: torch.Tensor,
                                 mask_u8: torch.Tensor,
                                 block_idx: torch.Tensor,
                                 values: torch.Tensor, x: int,
                                 factor: float) -> None:
    """Block gather (reference): values[j*x+i] = grad[block_idx[j]+i] *
    factor where the position is in-bounds AND mask-set, else 0.

    The mask guard (not the block structure) decides what moves: tail
    blocks and any structural drift transfer exactly the mask-set lanes.
    """
    n = grad_flat.numel()
    kb = block_idx.numel()
    for lo in range(0, kb, _BLOCK_CHUNK):
        hi = min(kb, lo + _BLOCK_CHUNK)
        inb, pos = _block_positions(block_idx, x, lo, hi, n)
        act = inb & (mask_u8[pos] != 0)
        v = grad_flat[pos]
        if factor != 1.0:
            v = v * factor
        values[lo * x:hi * x] = torch.where(
            act, v, torch.zeros((), dtype=v.dtype, device=v.device)
        ).reshape(-1)


def scatter_selected_blocks_torch(dst: torch.Tensor, mask_u8: torch.Tensor,
                                  block_idx: torch.Tensor,
                                  src: torch.Tensor, x: int) -> None:
    """Block scatter (reference): dst[block_idx[j]+i] = src[j*x+i] on
    in-bounds mask-set lanes only (the gather's exact mirror)."""
    n = dst.numel()
    kb = block_idx.numel()
    for lo in range(0, kb, _BLOCK_CHUNK):
        hi = min(kb, lo + _BLOCK_CHUNK)
        inb, pos = _block_positions(block_idx, x, lo, hi, n)
        act = inb & (mask_u8[pos] != 0)
        dst[pos[act]] = src[lo * x:hi * x].reshape(hi - lo, x)[act]

