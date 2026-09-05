# Copyright (c) 2026, SGA framework contributors.
"""Scheme-agnostic Triton kernels and dispatchers shared by SGA plugins.

Ported from the historical fork's ``megatron/core/sga/common.py``. The
Megatron-coupled clip plumbing lives in the engine's clip service instead;
only pure tensor kernels and the AdamW step-counter helper live here. The
pure-torch fallbacks are physically isolated in ``sga/kernels_torch.py`` —
this file is Triton + dispatch.
"""

import torch

from sga.kernels_torch import (
    compact_norm_sq_contribution_torch,
    gather_selected_blocks_torch,
    masked_norm_sq_contribution_torch,
    pack_selected_torch,
    scatter_selected_blocks_torch,
    scatter_selected_torch,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is present in all GPU envs
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _masked_norm_sq_kernel(
        grad_ptr, mask_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Partial sum of grad**2 over mask-active positions; atomically
        accumulated into ``out_ptr`` (fp32 scalar). No temporary allocations:
        replaces the boolean-index gather + torch norm per parameter."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements
        mask_val = tl.load(mask_ptr + offsets, mask=valid, other=0)
        active = valid & (mask_val > 0)
        g = tl.load(grad_ptr + offsets, mask=active, other=0.0)
        tl.atomic_add(out_ptr, tl.sum(g * g))

    @triton.jit
    def _scatter_selected_kernel(
        src_ptr, idx_ptr, dst_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """``dst[idx[i]] = src[i]`` with int32 indices (torch ``index_copy_``
        requires int64). No index cast / gather temporaries."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements
        idx = tl.load(idx_ptr + offsets, mask=valid, other=0)
        vals = tl.load(src_ptr + offsets, mask=valid, other=0.0)
        tl.store(dst_ptr + idx, vals, mask=valid)

    @triton.jit
    def _pack_kernel(grad_ptr, idx_ptr, val_ptr, factor, k,
                     BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr):
        """Fused sparse pack: ``values[i] = grad[idx[i]] * factor`` and
        ``grad[idx[i]] = 0`` in one pass. The zero store MOVES the selective
        clear here from grad_correct: the buffer state invariant
        (selected coords zero before the next backward's accumulation) is
        established at grad_sync time on compact-emission paths. Bitwise
        equal to index_select + payload scale + scatter-style clear: one
        fp32 multiply per element, pure stores."""
        pid = tl.program_id(0)
        t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = t < k
        i = tl.load(idx_ptr + t, mask=valid, other=0)
        v = tl.load(grad_ptr + i, mask=valid, other=0.0)
        if SCALE:
            v = v * factor
        tl.store(val_ptr + t, v, mask=valid)
        tl.store(grad_ptr + i, tl.zeros([BLOCK_SIZE], tl.float32), mask=valid)

    @triton.jit
    def _compact_norm_stage1(val_ptr, part_ptr, k, BLOCK_SIZE: tl.constexpr):
        """Block partial sums of val**2 into ``part[pid]`` — fixed reduction
        order per block, NO atomics: the compact norm is run-to-run
        deterministic (the masked atomic kernel it replaces was the
        run-to-run clip-coeff jitter source)."""
        pid = tl.program_id(0)
        t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        v = tl.load(val_ptr + t, mask=t < k, other=0.0)
        tl.store(part_ptr + pid, tl.sum(v * v, axis=0))

    @triton.jit
    def _gather_blocks_kernel(grad_ptr, mask_ptr, blk_ptr, val_ptr, factor,
                              k_blocks, n_elements,
                              X: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                              SCALE: tl.constexpr):
        """Block gather: values[t] = grad[blk[t//X] + t%X] * factor on
        in-bounds MASK-SET lanes, 0 elsewhere. X is constexpr (a power of
        two from config validation), so the div/mod compile to shifts. The
        mask guard — not the block structure — decides what moves: tail
        blocks and any structural drift transfer exactly the set lanes, and
        every rank holds the same verified mask, so masked-off payload
        lanes are identical zeros everywhere."""
        pid = tl.program_id(0)
        t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = t < k_blocks * X
        j = t // X
        i = t % X
        start = tl.load(blk_ptr + j, mask=valid, other=0)
        pos = start + i
        inb = valid & (pos < n_elements)
        m = tl.load(mask_ptr + pos, mask=inb, other=0)
        act = inb & (m != 0)
        v = tl.load(grad_ptr + pos, mask=act, other=0.0)
        if SCALE:
            v = v * factor
        tl.store(val_ptr + t, v, mask=valid)

    @triton.jit
    def _scatter_blocks_kernel(src_ptr, mask_ptr, blk_ptr, dst_ptr,
                               k_blocks, n_elements,
                               X: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        """Block scatter: dst[blk[t//X] + t%X] = src[t] on in-bounds
        mask-set lanes only — the exact mirror of the gather's lane mask,
        so payload zeros (masked-off lanes) never land in the grad."""
        pid = tl.program_id(0)
        t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = t < k_blocks * X
        j = t // X
        i = t % X
        start = tl.load(blk_ptr + j, mask=valid, other=0)
        pos = start + i
        inb = valid & (pos < n_elements)
        m = tl.load(mask_ptr + pos, mask=inb, other=0)
        act = inb & (m != 0)
        v = tl.load(src_ptr + t, mask=valid, other=0.0)
        tl.store(dst_ptr + pos, v, mask=act)


_CLIP_BLOCK_SIZE = 1024


def scatter_selected(dst: torch.Tensor, indices: torch.Tensor, src: torch.Tensor):
    """Scatter ``src`` into ``dst`` at ``indices`` (int32/int64), in place.

    The Triton path uses int32 offsets into ``dst``; buckets at or above
    2^31 elements are refused loudly rather than wrapping silently.
    """
    n = indices.numel()
    if n == 0:
        return
    assert dst.numel() < 2 ** 31, (
        f"scatter_selected assumes < 2^31 bucket elements, got {dst.numel()}"
    )
    if _HAS_TRITON and dst.is_cuda:
        grid = ((n + _CLIP_BLOCK_SIZE - 1) // _CLIP_BLOCK_SIZE,)
        _scatter_selected_kernel[grid](
            src, indices, dst, n, BLOCK_SIZE=_CLIP_BLOCK_SIZE
        )
    else:
        scatter_selected_torch(dst, indices, src)


def pack_selected(grad: torch.Tensor, indices: torch.Tensor,
                  values: torch.Tensor, factor: float) -> None:
    """Fused pack: gather ``grad[indices]`` into ``values`` (scaled by
    ``factor``) and ZERO those positions of ``grad`` in the same pass.

    ``factor == 1.0`` compiles the multiply out (constexpr skip). int32
    indices, < 2^31 bucket elements, same contract as scatter_selected.
    """
    k = indices.numel()
    if k == 0:
        return
    assert grad.numel() < 2 ** 31, (
        f"pack_selected assumes < 2^31 bucket elements, got {grad.numel()}"
    )
    if _HAS_TRITON and grad.is_cuda:
        grid = ((k + _CLIP_BLOCK_SIZE - 1) // _CLIP_BLOCK_SIZE,)
        _pack_kernel[grid](
            grad, indices, values, factor, k,
            BLOCK_SIZE=_CLIP_BLOCK_SIZE, SCALE=factor != 1.0,
        )
    else:
        pack_selected_torch(grad, indices, values, factor)


def gather_selected_blocks(grad: torch.Tensor, mask_u8: torch.Tensor,
                           block_idx: torch.Tensor, values: torch.Tensor,
                           x: int, factor: float) -> None:
    """Block gather: fill ``values[:k_blocks*x]`` from ``grad`` at the
    x-element runs headed by ``block_idx`` (int32 block-start addresses).

    Lane (j, i) reads position ``block_idx[j] + i``; the lane is active
    only when in-bounds AND mask-set, and inactive lanes write 0 (so the
    payload is allreduce-safe: every rank holds the same verified mask and
    contributes the same zeros). Unlike ``pack_selected`` this is a PURE
    READ — the writeback choreography clears via the optimizer's masked
    path post-commit, not at sync time. Same int32 / < 2^31 contract.
    """
    kb = block_idx.numel()
    if kb == 0:
        return
    assert grad.numel() < 2 ** 31, (
        f"gather_selected_blocks assumes < 2^31 bucket elements, "
        f"got {grad.numel()}"
    )
    assert values.numel() >= kb * x, (
        f"values buffer {values.numel()} < k_blocks*x {kb * x}")
    if _HAS_TRITON and grad.is_cuda:
        total = kb * x
        grid = ((total + _CLIP_BLOCK_SIZE - 1) // _CLIP_BLOCK_SIZE,)
        _gather_blocks_kernel[grid](
            grad, mask_u8, block_idx, values, factor, kb, grad.numel(),
            X=x, BLOCK_SIZE=_CLIP_BLOCK_SIZE, SCALE=factor != 1.0,
        )
    else:
        gather_selected_blocks_torch(grad, mask_u8, block_idx, values,
                                     x, factor)


def scatter_selected_blocks(dst: torch.Tensor, mask_u8: torch.Tensor,
                            block_idx: torch.Tensor, src: torch.Tensor,
                            x: int) -> None:
    """Block scatter: write ``src[:k_blocks*x]`` back into ``dst`` at the
    x-element runs headed by ``block_idx`` — the exact mirror of
    ``gather_selected_blocks``' lane mask, so payload zeros (masked-off
    lanes, e.g. tail-block padding) never land in the grad buffer.
    """
    kb = block_idx.numel()
    if kb == 0:
        return
    assert dst.numel() < 2 ** 31, (
        f"scatter_selected_blocks assumes < 2^31 bucket elements, "
        f"got {dst.numel()}"
    )
    assert src.numel() >= kb * x, (
        f"src buffer {src.numel()} < k_blocks*x {kb * x}")
    if _HAS_TRITON and dst.is_cuda:
        total = kb * x
        grid = ((total + _CLIP_BLOCK_SIZE - 1) // _CLIP_BLOCK_SIZE,)
        _scatter_blocks_kernel[grid](
            src, mask_u8, block_idx, dst, kb, dst.numel(),
            X=x, BLOCK_SIZE=_CLIP_BLOCK_SIZE,
        )
    else:
        scatter_selected_blocks_torch(dst, mask_u8, block_idx, src, x)


def compact_norm_sq_contribution(values: torch.Tensor,
                                 total_norm_sq: torch.Tensor) -> None:
    """Accumulate ``sum(values**2)`` into the fp32 scalar ``total_norm_sq``.

    Two-stage deterministic reduction (block partials + a final sum over the
    small partials buffer) — no atomics, fixed order per shape.
    """
    k = values.numel()
    if k == 0:
        return
    if _HAS_TRITON and values.is_cuda:
        nblk = (k + _CLIP_BLOCK_SIZE - 1) // _CLIP_BLOCK_SIZE
        parts = torch.empty(nblk, dtype=torch.float32, device=values.device)
        _compact_norm_stage1[(nblk,)](values, parts, k,
                                      BLOCK_SIZE=_CLIP_BLOCK_SIZE)
        total_norm_sq += parts.sum()
    else:
        compact_norm_sq_contribution_torch(values, total_norm_sq)


def use_fused_masked_op(grad: torch.Tensor, mask: torch.Tensor) -> bool:
    """The fused masked-op Triton path (masked norm-sq accumulation) needs
    CUDA fp32 grads and a 1-byte mask (uint8/bool), both contiguous."""
    return (
        _HAS_TRITON
        and grad.is_cuda
        and grad.dtype == torch.float32
        and grad.is_contiguous()
        and mask.is_contiguous()
        and mask.element_size() == 1
    )


def masked_norm_sq_contribution(grad: torch.Tensor, mask: torch.Tensor,
                                total_norm_sq: torch.Tensor) -> None:
    """Accumulate ``sum(grad[mask]**2)`` into the fp32 scalar ``total_norm_sq``.

    Fused Triton path when possible (atomic-add accumulation — see the clip
    service for why the DP-bitwise fix afterwards is mandatory), boolean-index
    fallback otherwise.
    """
    grad_flat = grad.view(-1)
    mask_flat = mask.view(-1)
    if use_fused_masked_op(grad_flat, mask_flat):
        grid = ((grad_flat.numel() + _CLIP_BLOCK_SIZE - 1) // _CLIP_BLOCK_SIZE,)
        _masked_norm_sq_kernel[grid](
            grad_flat, mask_flat, total_norm_sq, grad_flat.numel(),
            BLOCK_SIZE=_CLIP_BLOCK_SIZE,
        )
    else:
        masked_norm_sq_contribution_torch(grad_flat, mask_flat, total_norm_sq)


def sga_peek_step(wrapper, group):
    """Read-only variant of sga_advance_step: the step the NEXT optimizer
    commit will use, WITHOUT mutating any optimizer-visible counter.

    Shadow mode contract: the stock commit path owns its own counters —
    TE/apex FusedAdam advances ``group['step']`` INSIDE ``step()`` (creating
    it on first use), torch AdamW advances per-param ``state['step']`` — so
    the engine must not write group['step'] on shadow runs. Calling
    sga_advance_step there would make FusedAdam's bias corrections run with
    a double-advanced step (engine +1, stock +1 per iteration). This peek
    derives the upcoming-commit number from whichever counter style the
    optimizer maintains.
    """
    if 'step' in group:
        s = group['step']
        return (int(s.item()) if torch.is_tensor(s) else int(s)) + 1
    for p in group['params']:
        p_state = wrapper.optimizer.state.get(p)
        if p_state is not None and 'step' in p_state:
            s = p_state['step']
            return (int(s.item()) if torch.is_tensor(s) else int(s)) + 1
    return 1  # no counter anywhere: the upcoming commit is the first


def sga_advance_step(wrapper, group):
    """Advance the shared Adam step counter by one and return the NEW value
    as a plain python int (for beta**step math).

    The engine's record service calls this EXACTLY once per optimizer step
    (repeated visits within one step are stamped out), so the counter tracks
    the fork's iteration one-to-one — the invariant every bias correction
    relies on. Callers are optimizer-internal; nothing else may invoke it.

    TE/Apex FusedAdam maintains group['step'] itself -> shared counter,
    nothing to seed (its own counters never advance because the optimizer
    template never calls its step, so this branch serves it too). torch AdamW keeps
    per-param state['step'] instead: seed the group counter from it ONCE
    (uniform across params), then keep incrementing locally. Absent state
    => fresh moments (iter 0 or release-ckpt finetune) => seed 0. Tensor
    values are coerced to int.
    """
    if 'step' in group:
        s = group['step']
        s = int(s.item()) if torch.is_tensor(s) else int(s)
        group['step'] = s + 1
        return group['step']

    # First call, torch-AdamW (or similar) path: seed from per-param state.
    if not getattr(wrapper, '_sga_step_seeded', False):
        seed = 0
        for p in group['params']:
            p_state = wrapper.optimizer.state.get(p)
            if p_state is not None and 'step' in p_state:
                s = p_state['step']
                seed = int(s.item()) if torch.is_tensor(s) else int(s)
                break
        wrapper._sga_step_seeded = True
        wrapper._sga_step_seed_value = seed

    group['step'] = wrapper._sga_step_seed_value + 1
    return group['step']
