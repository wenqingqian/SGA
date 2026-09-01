# Copyright (c) 2026, SGA framework contributors.
"""The SGA AdamW optimizer template (kernels ported from the historical
radius scheme).

The AdamW update is split so the weight-update direction can be materialized:

    update = weight_decay * w + (m / bias1) / (sqrt(v / bias2) + eps)

- Sparse steps (WRITE_UPDATE=False): fused apply ``w -= lr * update``; the
  grad buffer is untouched so the residual at non-topk positions keeps
  accumulating. m/v update at EVERY coordinate (masked-out coords
  contribute a zero gradient; momentum keeps decaying).
- Full steps (WRITE_UPDATE=True): the score named by the config is written
  into the fp32 main grads (the reselection reads it from there). With the
  default 'update' score the SIGNED update takes the grad buffer directly;
  with a named score (abs_g/abs_m/coi) the two outputs swap targets — the
  score takes over the grad buffer and the signed update goes to per-record
  side buffers, applied and freed record-by-record BEFORE the reselection
  reads the score views. The grad buffer ALIASES the DDP bucket's fp32 grad
  storage (megatron: main_param.grad = model_param.main_grad.float() is an
  identity view for fp32 buckets), so the write lands in the bucket where
  the residual lives. Safety is purely ORDER-based, and the invariant holds:
  on a full step the residual was just fully communicated; the engine reads
  the score views BEFORE the caller's selective reset zeroes them; only after that
  does the selection read the still-materialized score (the optimizer->score
  handoff). Score math is defined by the score registry (sga/plugins/scores.py);
  the kernel branches here are its online materialization, pinned against the
  torch reference within GPU tolerance by tests/test_kernels_gpu.py.

Reset contract (selective with residual fold): sparse steps zero only the
communicated (topk) positions; when the NEXT step is a full sync and
``residual_mean`` is set, the residual is pre-divided by the cycle period T
(fused kernel multiplies by the fp32 host reciprocal — bitwise identical to
torch's scalar division). Full steps zero everything.
"""

from typing import Optional

import torch

from sga.plugins import OptimizerDecl

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


DECL = OptimizerDecl(
    name="adamw",
    reset_mode="selective",
    emittable_scores=frozenset({"update", "abs_g", "abs_m", "coi"}),
    requires_full_sync_every_period=True,
)

_BLOCK_SIZE = 1024

# Score emission modes for full steps (SCORE_MODE constexpr). Everything is
# computed from values only reachable in this kernel pass: abs_m reads the
# PRE-commit momentum that the same pass overwrites; coi reads both the
# pre-commit and the folded moments. Mode 0 (the 'update' score) is the
# historical default: the signed update itself is the materialized score.
_SCORE_MODES = {"abs_g": 1, "abs_m": 2, "coi": 3}


if _HAS_TRITON:

    @triton.jit
    def _grad_correct_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, mask_ptr, score_ptr,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        WRITE_UPDATE: tl.constexpr,
        SCORE_MODE: tl.constexpr,
    ):
        """AdamW m/v update + weight-update, fused-apply or materialize-score."""
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

        if SCORE_MODE == 1:
            # abs_g: |g| of the gradient being committed this step.
            score = tl.abs(grad)
        elif SCORE_MODE == 2:
            # abs_m: |m| of the PRE-commit momentum.
            score = tl.abs(m)
        elif SCORE_MODE == 3:
            # coi: |U(g) - U(0)| with the PRE-commit moments; the weight-decay
            # term cancels in the difference. m/v are read, not yet mutated.
            u_0 = (beta1 * m / bias_correction1) / (
                tl.sqrt(beta2 * v / bias_correction2) + eps
            )
            score = tl.abs(m_hat / (tl.sqrt(v_hat) + eps) - u_0)

        tl.store(m_ptr + offsets, m_new, mask=valid)
        tl.store(v_ptr + offsets, v_new, mask=valid)
        if WRITE_UPDATE:
            if SCORE_MODE > 0:
                # Full step with a named score: the score takes over the grad
                # buffer (the reselection reads it from there, exactly like
                # the default update score) and the signed update goes to the
                # side buffer; the caller applies from it and frees it before
                # the reselection, so the select-time peak carries no
                # model-sized side-buffer set.
                tl.store(grad_ptr + offsets, score, mask=valid)
                tl.store(score_ptr + offsets, update, mask=valid)
            else:
                # Default (update score): materialize the update for the topk.
                tl.store(grad_ptr + offsets, update, mask=valid)
        else:
            # Masked iteration: fused apply; grad buffer keeps the residual.
            param = param - lr * update
            tl.store(param_ptr + offsets, param, mask=valid)

    @triton.jit
    def _selective_reset_kernel(
        grad_ptr, mask_ptr,
        inv_interval,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        FOLD: tl.constexpr,
    ):
        """Selective residual reset, fused into a single pass:
        - mask-active (topk, communicated) positions: store 0;
        - inactive positions with FOLD=True: store grad * inv_interval —
          exact vs torch's ``/= interval`` on the CPU/torch path; on GPU the
          kernel multiplies by the host-computed fp32 reciprocal, matching
          CUDA scalar division (no GPU fused-reset A/B test exists)."""
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


def adamw_grad_correct(
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
    score_out: torch.Tensor = None,
    score_mode: str = None,
) -> None:
    """Run the grad-correct step on contiguous same-size buffers.

    Updates m/v in place. ``write_update=True`` writes the materialized score
    into ``grad_buffer``; otherwise applies ``w -= lr * update`` and leaves
    the grad buffer untouched. With ``mask``, positions outside it contribute
    a zero gradient.

    Full-step score emission (``score_out`` given): the named score (one of
    abs_g / abs_m / coi, all computed from pre-commit state reachable only
    inside this pass) is written into ``grad_buffer`` — the reselection reads
    it from there exactly like the default update score — and the signed
    update goes to ``score_out`` instead; the caller applies and frees it
    before the reselection. An unknown ``score_mode`` alongside ``score_out``
    is rejected before any work.
    """
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    has_score = score_out is not None
    if has_score and score_mode not in _SCORE_MODES:
        raise ValueError(f"unknown score_mode: {score_mode!r}")

    n_elements = param_buffer.numel()
    if n_elements == 0:
        return

    if not (_HAS_TRITON and param_buffer.is_cuda):
        _grad_correct_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            mask, write_update,
            score_out, score_mode if has_score else None,
        )
        return

    has_mask = mask is not None
    mask_u8 = mask.to(torch.uint8) if (has_mask and mask.dtype != torch.uint8) else mask
    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _grad_correct_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        mask_u8 if has_mask else grad_buffer,
        score_out if has_score else grad_buffer,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        HAS_MASK=has_mask,
        WRITE_UPDATE=write_update,
        SCORE_MODE=_SCORE_MODES[score_mode] if has_score else 0,
    )


def _grad_correct_torch(
    param, grad, m, v, lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    mask, write_update,
    score_out=None, score_mode=None,
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
    if score_mode == "abs_g":
        score = g.abs()
    elif score_mode == "abs_m":
        score = m.abs()
    elif score_mode == "coi":
        u_0 = (beta1 * m / bias_correction1) / (
            (beta2 * v / bias_correction2).sqrt() + eps
        )
        score = (m_hat / (v_hat.sqrt() + eps) - u_0).abs()
    m.copy_(m_new)
    v.copy_(v_new)
    if write_update:
        if score_out is not None:
            grad.copy_(score)
            score_out.copy_(update)
        else:
            grad.copy_(update)
    else:
        param.sub_(lr * update)


def selective_reset(grad_buffer: torch.Tensor, mask: torch.Tensor,
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
        _selective_reset_kernel[grid](
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


class SgaAdamW:
    """The SGA AdamW optimizer template: grad-correct + residual reset."""

    decl = DECL

    def __init__(self, params: dict, emit_score: Optional[str] = None):
        self.params = params
        self.residual_mean = bool(params.get("residual_mean", False))
        # Which registry score to materialize on full steps (build-time
        # injection). None/'update' keeps the historical default: the signed
        # AdamW update itself is the topk input.
        self.emit_score = emit_score or "update"
        if self.emit_score not in DECL.emittable_scores:
            raise ValueError(
                f"adamw cannot emit score '{self.emit_score}' "
                f"(emittable: {sorted(DECL.emittable_scores)})"
            )

    def ensure_state(self, bucket_states):
        pass  # no persistent per-bucket buffers beyond the Support caches

    def _records_pass(self, ctx, write_update, use_mask, score_bufs=None):
        for rec in ctx.records:
            state = rec._state
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
            adamw_grad_correct(
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
                score_out=score_bufs.get(id(rec)) if score_bufs else None,
                score_mode=self.emit_score,
            )

    @torch.no_grad()
    def full_step(self, ctx):
        """Full-sync step: materialize the configured score and apply the update.

        With the default 'update' score the signed update lands in the main
        grads (Phase 2 applies from there). With a named score
        (abs_g/abs_m/coi) the two outputs swap targets: the score takes over
        the grad buffer and the update goes to per-record side buffers,
        applied and freed record-by-record BEFORE the engine's reselection
        reads the score views — the select-time GPU peak then carries no
        model-sized side-buffer set (the legacy scheme's OOM fix). Caveat on
        the Phase-0 allocation: side buffers for ALL records are allocated
        up front, so a named-score full step transiently carries +1
        model-sized fp32 buffer set versus the default path — size GPU
        headroom accordingly at 1B+ scale.
        """
        # Phase 0: side buffers for the signed update (named scores only).
        # Keyed by id(): ParamRecord is a dataclass — unhashable by value.
        # Flat: the grad-correct buffers are the params' view(-1) spans.
        score_bufs = None
        if self.emit_score != "update":
            score_bufs = {
                id(rec): torch.empty_like(rec.main_param.data.view(-1))
                for rec in ctx.records
            }
        # Phase 1: grad-correct; the configured score materializes into main grads.
        self._records_pass(ctx, write_update=True, use_mask=False,
                           score_bufs=score_bufs)
        # Phase 2: apply w -= lr * update (from side buffers for named scores,
        # from main grads for the default), freeing each as it is consumed.
        for rec in ctx.records:
            upd = score_bufs.pop(id(rec), None) if score_bufs else None
            if upd is not None:
                rec.main_param.data.add_(upd.view_as(rec.main_param.data),
                                         alpha=-rec.lr)
            else:
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
                selective_reset(grad_data, mask, float(ctx.cycle_period), fold_all)
            for grad in bs.extra_main_grads:
                grad.zero_()
