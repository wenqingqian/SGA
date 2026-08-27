# Copyright (c) 2026, SGA framework contributors.
"""Radius optimizer error feedback (historical radius scheme).

Kernels ported verbatim from scheme_radius/kernels.py (survey S3). The AdamW
update is split so the weight-update direction can be materialized:

    update = weight_decay * w + (m / bias1) / (sqrt(v / bias2) + eps)

- Sparse steps (WRITE_UPDATE=False): fused apply ``w -= lr * update``; the
  grad buffer is untouched so the residual at non-topk positions keeps
  accumulating. m/v update at EVERY coordinate (masked-out coords
  contribute a zero gradient; momentum keeps decaying).
- Full steps (WRITE_UPDATE=True): the signed update is written into the
  fp32 main grads. That buffer ALIASES the DDP bucket's fp32 grad storage
  (megatron: main_param.grad = model_param.main_grad.float() is an identity
  view for fp32 buckets), so the write lands in the bucket where the
  residual lives. Safety is purely ORDER-based, and the invariant holds:
  on a full step the residual was just fully communicated; the EF reads
  the views BEFORE the caller's selective reset zeroes them; only after
  that does the selection read the still-materialized updates (EF->score
  handoff).

Reset contract (selective with residual fold): sparse steps zero only the
communicated (topk) positions; when the NEXT step is a full sync and
``residual_mean`` is set, the residual is pre-divided by the cycle period T
(fused kernel multiplies by the fp32 host reciprocal — bitwise identical to
torch's scalar division). Full steps zero everything.
"""

import torch

from sga.plugins import EFDecl

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

DECL = EFDecl(
    name="radius_opt",
    reset_mode="selective",
    needs_acc_count=False,
    emits_score_view=True,
    requires_full_sync_every_period=True,
)

_BLOCK_SIZE = 1024


if _HAS_TRITON:

    @triton.jit
    def _radius_grad_correct_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, mask_ptr,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        WRITE_UPDATE: tl.constexpr,
    ):
        """AdamW m/v update + weight-update, fused-apply or materialize-update."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements

        if HAS_MASK:
            mask_val = tl.load(mask_ptr + offsets, mask=valid, other=0)
            active = valid & (mask_val > 0)
        else:
            active = valid

        param = tl.load(param_ptr + offsets, mask=valid, other=0.0)
        grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
        m = tl.load(m_ptr + offsets, mask=valid, other=0.0)
        v = tl.load(v_ptr + offsets, mask=valid, other=0.0)

        m_new = beta1 * m + (1.0 - beta1) * grad
        v_new = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2

        update = weight_decay * param + m_hat / (tl.sqrt(v_hat) + eps)

        tl.store(m_ptr + offsets, m_new, mask=valid)
        tl.store(v_ptr + offsets, v_new, mask=valid)
        if WRITE_UPDATE:
            # Full iteration: materialize the update for the global topk.
            tl.store(grad_ptr + offsets, update, mask=valid)
        else:
            # Masked iteration: fused apply; grad buffer keeps the residual.
            param = param - lr * update
            tl.store(param_ptr + offsets, param, mask=valid)

    @triton.jit
    def _radius_reset_kernel(
        grad_ptr, mask_ptr,
        inv_interval,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        FOLD: tl.constexpr,
    ):
        """Selective residual reset, fused into a single pass:
        - mask-active (topk, communicated) positions: store 0;
        - inactive positions with FOLD=True: store grad * inv_interval —
          bitwise identical to torch's ``/= interval`` (CUDA scalar division
          multiplies by the fp32 reciprocal computed on the host)."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements
        mask_val = tl.load(mask_ptr + offsets, mask=valid, other=0)
        active = valid & (mask_val > 0)

        tl.store(grad_ptr + offsets, tl.zeros([BLOCK_SIZE], tl.float32), mask=active)

        if FOLD:
            inactive = valid & (mask_val == 0)
            g = tl.load(grad_ptr + offsets, mask=inactive, other=0.0)
            tl.store(grad_ptr + offsets, g * inv_interval, mask=inactive)


def radius_grad_correct(
    param_buffer: torch.Tensor,
    grad_buffer: torch.Tensor,
    exp_avg_buffer: torch.Tensor,
    exp_avg_sq_buffer: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    mask: torch.Tensor = None,
    write_update: bool = False,
) -> None:
    """Run the grad-correct step on contiguous same-size buffers.

    Updates m/v in place. ``write_update=True`` writes the signed update into
    ``grad_buffer``; otherwise applies ``w -= lr * update`` and leaves the
    grad buffer untouched. With ``mask``, positions outside it contribute a
    zero gradient.
    """
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    n_elements = param_buffer.numel()
    if n_elements == 0:
        return

    if not (_HAS_TRITON and param_buffer.is_cuda):
        _radius_grad_correct_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2, mask, write_update,
        )
        return

    has_mask = mask is not None
    mask_u8 = mask.to(torch.uint8) if (has_mask and mask.dtype != torch.uint8) else mask
    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _radius_grad_correct_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        mask_u8 if has_mask else grad_buffer,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        HAS_MASK=has_mask,
        WRITE_UPDATE=write_update,
    )


def _radius_grad_correct_torch(
    param, grad, m, v, lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2, mask, write_update,
) -> None:
    """Pure-torch reference path (CPU unit tests); same math as the kernel."""
    if mask is not None:
        g = torch.where(mask > 0, grad, torch.zeros_like(grad))
    else:
        g = grad
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * g * g
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    update = weight_decay * param + m_hat / (v_hat.sqrt() + eps)
    m.copy_(m_new)
    v.copy_(v_new)
    if write_update:
        grad.copy_(update)
    else:
        param.sub_(lr * update)


def radius_reset(grad_buffer: torch.Tensor, mask: torch.Tensor,
                 interval: float, fold: bool) -> None:
    """Fused selective reset: zero communicated positions, optionally fold
    (divide by interval) the residual at inactive positions."""
    n_elements = grad_buffer.numel()
    if n_elements == 0:
        return
    mask_u8 = mask if mask.dtype == torch.uint8 else mask.to(torch.uint8)
    if _HAS_TRITON and grad_buffer.is_cuda and grad_buffer.dtype == torch.float32:
        # fp32 reciprocal, computed exactly as torch's CUDA scalar division.
        inv_interval = float((torch.ones(1, dtype=torch.float32) / interval).item())
        grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
        _radius_reset_kernel[grid](
            grad_buffer, mask_u8, inv_interval, n_elements,
            BLOCK_SIZE=_BLOCK_SIZE, FOLD=fold,
        )
        return
    # Element-wise on purpose (boolean indexing materializes full-bucket
    # int64 indices, ~GB-scale). The unconditional divide is exact: actives
    # were just set to 0.0, and 0.0 / interval == 0.0.
    active = mask_u8 > 0
    grad_buffer.masked_fill_(active, 0.0)
    if fold:
        grad_buffer.div_(interval)


class RadiusOptEF:
    """Error-feedback plugin: radius grad-correct + residual reset."""

    decl = DECL

    def __init__(self, params: dict):
        self.params = params
        self.residual_mean = bool(params.get("residual_mean", False))

    def ensure_state(self, bucket_states):
        pass  # no persistent per-bucket buffers beyond the Support caches

    def _records_pass(self, ctx, write_update, use_mask):
        for rec in ctx.records:
            state = rec._state
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
            radius_grad_correct(
                param_buffer=rec.main_param.data.view(-1),
                grad_buffer=rec.grad.data.view(-1),
                exp_avg_buffer=state["exp_avg"].view(-1),
                exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
                lr=rec.lr,
                beta1=rec.beta1,
                beta2=rec.beta2,
                eps=rec.eps,
                weight_decay=rec.weight_decay,
                step=rec.step,
                mask=rec.mask_u8() if use_mask else None,
                write_update=write_update,
            )

    @torch.no_grad()
    def full_step(self, ctx):
        """Full-sync step: materialize the update and apply it.

        The materialized views are consumed by the ENGINE at the reselection
        boundary (main_param.grad reads in meta order); EFContext carries no
        separate handoff payload — see the engine module docstring on the
        declaration-only score axis.
        """
        # Phase 1: grad-correct with the update materialized into main grads.
        self._records_pass(ctx, write_update=True, use_mask=False)
        # Phase 2: apply w -= lr * update (signed update still in main grads).
        for rec in ctx.records:
            rec.main_param.data.add_(rec.grad.data, alpha=-rec.lr)

    @torch.no_grad()
    def sparse_step(self, ctx):
        """Sparse step: fused grad-correct + apply with the persisted mask;
        the grad buffer (residual) is untouched."""
        self._records_pass(ctx, write_update=False, use_mask=True)

    def reset(self, ctx, was_full: bool):
        """Sparse steps: zero communicated positions, optionally pre-fold the
        residual by T when the next step is a full sync. Full steps: zero all.
        The engine keeps each bucket's ``active_mask_u8`` in sync with the
        support used by this step's sync."""
        fold_all = self.residual_mean and ctx.next_step_is_full
        for bs in ctx.bucket_states:
            grad_data = bs.grad_data
            mask = bs.active_mask_u8
            if was_full or mask is None:
                grad_data.zero_()
            else:
                radius_reset(grad_data, mask, float(ctx.cycle_period), fold_all)
            for grad in bs.extra_main_grads:
                grad.zero_()
