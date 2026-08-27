# Copyright (c) 2026, SGA framework contributors.
"""Masked AdamW error feedback (historical origin scheme).

Kernels ported verbatim from scheme_origin/masked_adamw.py (survey S2):
four Triton kernels (element/block x standard/acc-aware) plus the dispatch
wrapper. Semantics:

- Active (communicated) positions receive the full AdamW update.
- Acc-aware mode (``grad_acc_count: true``): the accumulated gradient is
  averaged by its accumulation count online (``grad / count``), moments use
  ``(1 - beta**count)`` weights, effective lr scales by ``count``.
- ``freeze_inactive``: masked-out positions keep m/v (decoupled weight decay
  still applies) vs the default zero-gradient momentum decay update.
- Decoupled weight decay applies to ALL positions.

Full steps run the same kernel with an all-ones mask. Count-aware math
applies everywhere, so a full step reduces to standard AdamW ONLY where the
acc counts are uniformly 1 (fresh buffers / right after a full clear); when
a full step follows sparse steps, uncommunicated coordinates carry counts
>= 2 and their stale residual enters that step's moment ONCE via grad/count.
"""

from typing import Optional

import torch

from sga.plugins import EFDecl

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

DECL = EFDecl(
    name="masked_adamw",
    reset_mode="selective",
    needs_acc_count=True,
    emits_score_view=False,
    requires_full_sync_every_period=False,
)

_BLOCK_SIZE = 1024


if _HAS_TRITON:

    @triton.jit
    def _masked_adamw_acc_aware_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, mask_ptr, acc_count_ptr,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        FREEZE_INACTIVE: tl.constexpr,
    ):
        """Accumulation-aware masked AdamW step with element-level mask.

        The moment updates use (1 - beta**count) weights where count is the
        per-element accumulation count; grads are online-averaged (grad/count)
        and the effective lr scales by count at active positions.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements

        mask_val = tl.load(mask_ptr + offsets, mask=valid, other=0)
        active = valid & (mask_val > 0)

        param = tl.load(param_ptr + offsets, mask=valid, other=0.0)
        m = tl.load(m_ptr + offsets, mask=valid, other=0.0)
        v = tl.load(v_ptr + offsets, mask=valid, other=0.0)

        count = tl.load(acc_count_ptr + offsets, mask=valid, other=1).to(tl.float32)
        count = tl.maximum(count, 1.0)

        # Decoupled weight decay to all valid positions first.
        param = param - lr * weight_decay * param

        # Load grad only for active positions.
        grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
        grad = grad / count

        beta1_pow = tl.exp(count * tl.log(beta1))
        beta2_pow = tl.exp(count * tl.log(beta2))
        one_minus_beta1 = 1.0 - beta1_pow
        one_minus_beta2 = 1.0 - beta2_pow
        m_new = beta1 * m + one_minus_beta1 * grad
        v_new = beta2 * v + one_minus_beta2 * grad * grad
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2

        lr_eff = tl.where(active, lr * count, lr)
        grad_update = lr_eff * m_hat / (tl.sqrt(v_hat) + eps)

        if FREEZE_INACTIVE:
            m = tl.where(active, m_new, m)
            v = tl.where(active, v_new, v)
            param = tl.where(active, param - grad_update, param)
        else:
            m = m_new
            v = v_new
            param = param - grad_update

        tl.store(param_ptr + offsets, param, mask=valid)
        tl.store(m_ptr + offsets, m, mask=valid)
        tl.store(v_ptr + offsets, v, mask=valid)

    @triton.jit
    def _masked_adamw_standard_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, mask_ptr,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        FREEZE_INACTIVE: tl.constexpr,
    ):
        """Standard masked AdamW step with element-level mask (no acc count)."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements

        mask_val = tl.load(mask_ptr + offsets, mask=valid, other=0)
        active = valid & (mask_val > 0)

        param = tl.load(param_ptr + offsets, mask=valid, other=0.0)
        m = tl.load(m_ptr + offsets, mask=valid, other=0.0)
        v = tl.load(v_ptr + offsets, mask=valid, other=0.0)

        param = param - lr * weight_decay * param

        grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)

        m_new = beta1 * m + (1.0 - beta1) * grad
        v_new = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2
        grad_update = lr * m_hat / (tl.sqrt(v_hat) + eps)

        if FREEZE_INACTIVE:
            m = tl.where(active, m_new, m)
            v = tl.where(active, v_new, v)
            param = tl.where(active, param - grad_update, param)
        else:
            m = m_new
            v = v_new
            param = param - grad_update

        tl.store(param_ptr + offsets, param, mask=valid)
        tl.store(m_ptr + offsets, m, mask=valid)
        tl.store(v_ptr + offsets, v, mask=valid)


def masked_adamw_step(
    param_buffer: torch.Tensor,
    grad_buffer: torch.Tensor,
    exp_avg_buffer: torch.Tensor,
    exp_avg_sq_buffer: torch.Tensor,
    mask: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    freeze_inactive: bool = False,
    acc_count: Optional[torch.Tensor] = None,
) -> None:
    """Run a single masked AdamW step on contiguous same-size fp32 buffers.

    ``mask`` is uint8/bool at element level; with ``acc_count`` the update is
    acc-aware (online averaging), otherwise standard masked AdamW. The
    historical block-mask kernel variants are intentionally not ported (v1
    freezes element-level support).
    """
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    n_elements = param_buffer.numel()
    if n_elements == 0:
        return

    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    mask_u8 = mask if mask.dtype == torch.uint8 else mask.to(torch.uint8)

    if not (_HAS_TRITON and param_buffer.is_cuda):
        _masked_adamw_step_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            mask_u8.bool(), lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2, freeze_inactive, acc_count,
        )
        return

    if acc_count is not None:
        _masked_adamw_acc_aware_kernel[grid](
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            mask_u8, acc_count,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
            FREEZE_INACTIVE=freeze_inactive,
        )
    else:
        _masked_adamw_standard_kernel[grid](
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            mask_u8,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
            FREEZE_INACTIVE=freeze_inactive,
        )


def _masked_adamw_step_torch(
    param, grad, m, v, mask_bool, lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2, freeze_inactive, acc_count,
) -> None:
    """Pure-torch reference path (CPU unit tests); same math as the kernels."""
    if acc_count is not None:
        count = acc_count.float().clamp(min=1.0)
    else:
        count = torch.ones_like(param)
    g = torch.where(mask_bool, grad / count, torch.zeros_like(grad))
    beta1_pow = torch.exp(count * float(torch.log(torch.tensor(beta1))))
    beta2_pow = torch.exp(count * float(torch.log(torch.tensor(beta2))))
    m_new = beta1 * m + (1.0 - beta1_pow) * g
    v_new = beta2 * v + (1.0 - beta2_pow) * g * g
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    lr_eff = torch.where(mask_bool, lr * count, lr)
    update = lr_eff * m_hat / (v_hat.sqrt() + eps)

    param.sub_(lr * weight_decay * param)
    if freeze_inactive:
        m.copy_(torch.where(mask_bool, m_new, m))
        v.copy_(torch.where(mask_bool, v_new, v))
        param.sub_(torch.where(mask_bool, update, torch.zeros_like(update)))
    else:
        m.copy_(m_new)
        v.copy_(v_new)
        param.sub_(update)


@torch.no_grad()
def _inline_dense_adamw(p, grad, state, lr, beta1, beta2, eps, weight_decay, step):
    """Dense AdamW fallback for params without a mask (historical code path:
    hand-rolled, NOT the configured torch optimizer — the numeric contract
    only holds for torch AdamW, asserted by the engine)."""
    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
    if weight_decay != 0:
        p.data.add_(p.data, alpha=-lr * weight_decay)
    p.data.addcdiv_(exp_avg, denom, value=-step_size)


class MaskedAdamWEF:
    """Error-feedback plugin: masked AdamW + selective reset (origin combo)."""

    decl = DECL

    def __init__(self, params: dict):
        self.params = params
        self.freeze_inactive = bool(params.get("freeze_inactive", False))
        self.use_acc_count = bool(params.get("grad_acc_count", True))

    def ensure_state(self, bucket_states):
        """Allocate (once, grow-only) the per-bucket uint8 acc-count buffers."""
        if not self.use_acc_count:
            return
        for bs in bucket_states:
            n = bs.grad_data.numel()
            if bs.grad_acc_count is None or bs.grad_acc_count.numel() != n:
                bs.grad_acc_count = torch.ones(
                    n, dtype=torch.uint8, device=bs.grad_data.device
                )

    def _step_record(self, rec, mask):
        state = rec._state
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(rec.main_param.data)
            state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
        masked_adamw_step(
            param_buffer=rec.main_param.data.view(-1),
            grad_buffer=rec.grad.data.view(-1),
            exp_avg_buffer=state["exp_avg"].view(-1),
            exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
            mask=mask,
            lr=rec.lr,
            beta1=rec.beta1,
            beta2=rec.beta2,
            eps=rec.eps,
            weight_decay=rec.weight_decay,
            step=rec.step,
            freeze_inactive=self.freeze_inactive,
            acc_count=rec.acc_count() if self.use_acc_count else None,
        )

    @torch.no_grad()
    def full_step(self, ctx):
        """Full-sync step: same kernel with the all-ones mask.

        Count-aware math applies throughout: coordinates left uncommunicated
        by earlier sparse steps carry acc counts >= 2 and their stale residual
        enters this moment once (grad/count, m/v weights 1-beta**count), so
        the update equals standard AdamW only where counts are uniformly 1.
        Records without a bucket state take the inline dense AdamW fallback,
        same contract as sparse_step.
        """
        for rec in ctx.records:
            bs = rec.bucket_state
            if bs is None:
                state = rec._state
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                    state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
                _inline_dense_adamw(
                    rec.main_param, rec.grad.data, state, rec.lr, rec.beta1,
                    rec.beta2, rec.eps, rec.weight_decay, rec.step,
                )
                continue
            if bs.ones_mask_u8 is None:
                bs.ones_mask_u8 = torch.ones(
                    bs.grad_data.numel(), dtype=torch.uint8, device=bs.grad_data.device
                )
            self._step_record(rec, bs.ones_mask_u8[rec.start:rec.end])

    @torch.no_grad()
    def sparse_step(self, ctx):
        """Sparse step: masked update with the active support."""
        for rec in ctx.records:
            mask = rec.mask_u8()
            if mask is None:
                # No support yet (bootstrap edge): dense-equivalent update.
                state = rec._state
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                    state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
                _inline_dense_adamw(
                    rec.main_param, rec.grad.data, state, rec.lr, rec.beta1,
                    rec.beta2, rec.eps, rec.weight_decay, rec.step,
                )
                continue
            self._step_record(rec, mask)

    def reset(self, ctx, was_full: bool):
        """Selective reset: clear only communicated positions; uncommunicated
        grads keep accumulating. Acc counts: communicated -> 1, else +1
        (uint8, saturating at the ceiling — see update_grad_acc_count).
        Full-sync steps clear everything. The engine keeps each bucket's
        ``active_mask_u8`` in sync with the support used by this step's sync.
        """
        for bs in ctx.bucket_states:
            grad_data = bs.grad_data
            mask = bs.active_mask_u8
            if was_full or mask is None:
                grad_data.zero_()
                if self.use_acc_count and bs.grad_acc_count is not None:
                    bs.grad_acc_count.fill_(1)
            else:
                if self.use_acc_count and bs.grad_acc_count is not None:
                    update_grad_acc_count(bs.grad_acc_count, mask)
                # masked_fill_ stays element-wise; boolean indexing would
                # materialize full-bucket int64 indices (~GB-scale).
                grad_data.masked_fill_(mask > 0, 0.0)
            for grad in bs.extra_main_grads:
                grad.zero_()


def update_grad_acc_count(count: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """In-place acc-count update: active -> 1, inactive -> +1 (uint8).

    Ported from scheme_origin/kernels.py (element granularity only in v1),
    with one deliberate deviation: inactive coordinates SATURATE at the
    uint8 ceiling (255) instead of wrapping to 0. A wrapped coordinate would
    read count 0 -> clamped 1 while still carrying its FULL stale residual,
    over-weighting that moment by ~256x; saturation only mildly under-weights
    very long streaks (the residual is averaged over at most 255).

    Formulated ELEMENT-WISE on purpose: boolean advanced indexing
    (count[mask] += 1) materializes full-bucket int64 index tensors
    (8 bytes/elem — GiB-scale on real models) and OOMs; these ops allocate
    only 1-byte temporaries.
    """
    active = active_mask > 0
    growable = (~active) & (count < 255)
    count += growable.to(torch.uint8)   # +1 only where count<255: cannot wrap
    count *= (~active).to(torch.uint8)  # active -> 0
    count += active.to(torch.uint8)     # active -> 1
    return count
