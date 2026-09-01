# Copyright (c) 2026, SGA framework contributors.
"""The SGA AdamW optimizer template (kernels ported from the historical
radius scheme).

The AdamW update is split so the weight-update direction can be materialized:

    update = weight_decay * w + (m / bias1) / (sqrt(v / bias2) + eps)

One fused kernel pass per record on EVERY step — m/v update at every
coordinate (sparse steps: masked-out coords contribute a zero gradient,
momentum keeps decaying) and the apply ``w -= lr * update`` ALWAYS fused
(single arithmetic implementation, no Python-side second pass):

- Sparse steps (EMIT_MODE=-1): masked m/v update + fused apply; the grad
  buffer is untouched so the residual at non-topk positions keeps
  accumulating.
- Full steps (EMIT_MODE>=0, the Score enum value of the configured score):
  the same pass additionally materializes the score into the fp32 main grads
  (the reselection reads it from there). The grad buffer ALIASES the DDP
  bucket's fp32 grad storage (megatron: main_param.grad =
  model_param.main_grad.float() is an identity view for fp32 buckets), so
  the write lands in the bucket where the residual lives. Safety is purely
  ORDER-based, and the invariant holds: on a full step the residual was just
  fully communicated; the engine reads the score views BEFORE the caller's
  selective reset zeroes them; only after that does the selection read the
  still-materialized score (the optimizer->score handoff).

Score math is defined by the score registry (sga/plugins/scores.py); the
kernel branches here are its online materialization, spelled as bare ints
with name comments (Triton takes compile-time ints only) and pinned against
the torch reference by tests/test_kernels_gpu.py. The torch reference path
lives in adamw_torch.py (physical isolation: this file is Triton + dispatch
only); CPU unit tests keep working without Triton.

Reset contract (selective with residual fold): sparse steps zero only the
communicated (topk) positions; when the NEXT step is a full sync and
``residual_mean`` is set, the residual is pre-divided by the cycle period T
(fused kernel multiplies by the fp32 host reciprocal — bitwise identical to
torch's scalar division). Full steps zero everything.
"""

from typing import Optional

import torch

from sga.plugins import OptimizerDecl
from sga.plugins.optimizer.adamw_torch import grad_correct_torch, selective_reset_torch
from sga.plugins.scores import Score, emittable_score_names, get_score_spec

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


DECL = OptimizerDecl(
    name="adamw",
    reset_mode="selective",
    # Derived, not restated: this template implements every registry score's
    # kernel branch (a second template would declare its own branch subset).
    emittable_scores=emittable_score_names(),
    requires_full_sync_every_period=True,
)

# The bare-int EMIT_MODE branches _grad_correct_kernel implements below.
# Load-time tie between the DERIVED emittable set above and that hardcoded
# branch coverage: a score registered without a branch here would pass every
# CPU gate (the registry formula exists) and only crash inside Triton
# tracing on the first GPU full step — possibly hours into a cluster run.
_IMPLEMENTED_MODES = frozenset({0, 1, 2, 3})
_MISSING_BRANCHES = sorted(
    n for n in emittable_score_names()
    if int(get_score_spec(n).score) not in _IMPLEMENTED_MODES
)
if _MISSING_BRANCHES:
    raise ValueError(
        f"registry scores {_MISSING_BRANCHES} have no _grad_correct_kernel "
        f"branch (implemented EMIT_MODEs: {sorted(_IMPLEMENTED_MODES)}) — "
        "add the branch or mark the score OFFLINE"
    )

_BLOCK_SIZE = 1024


if _HAS_TRITON:

    @triton.jit
    def _grad_correct_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, mask_ptr,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        EMIT_MODE: tl.constexpr,
    ):
        """AdamW m/v update + fused apply, with full-step score emission.

        EMIT_MODE is the Score enum value as a compile-time int: -1 = sparse
        (masked commit, grad buffer untouched); >=0 = full step (score
        materialized into the grad buffer). Both paths apply
        ``w -= lr * update`` in this same pass.
        """
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

        if EMIT_MODE == 1:
            # abs_g: |g| of the gradient being committed this step.
            score = tl.abs(grad)
        elif EMIT_MODE == 2:
            # abs_m: |m| of the PRE-commit momentum.
            score = tl.abs(m)
        elif EMIT_MODE == 3:
            # coi: |U(g) - U(0)| with the PRE-commit moments; the weight-decay
            # term cancels in the difference. m/v are read, not yet mutated.
            u_0 = (beta1 * m / bias_correction1) / (
                tl.sqrt(beta2 * v / bias_correction2) + eps
            )
            score = tl.abs(m_hat / (tl.sqrt(v_hat) + eps) - u_0)

        tl.store(m_ptr + offsets, m_new, mask=valid)
        tl.store(v_ptr + offsets, v_new, mask=valid)
        if EMIT_MODE >= 0:
            if EMIT_MODE == 0:
                # update: the signed update itself is the materialized score.
                tl.store(grad_ptr + offsets, update, mask=valid)
            else:
                # Named score: it takes over the grad buffer; the reselection
                # reads it from there exactly like the default update score.
                tl.store(grad_ptr + offsets, score, mask=valid)
        # Fused apply on every path (sparse: masked-out coords loaded grad=0,
        # so their update is the pure weight-decay/momentum drift — applied
        # exactly like the historical kernel did).
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
    emit_mode: Optional[int] = None,
) -> None:
    """Run the grad-correct step on contiguous same-size buffers.

    Updates m/v in place and always fuses the apply ``w -= lr * update``.
    ``emit_mode=None`` is the sparse step: with ``mask``, positions outside
    it contribute a zero gradient; the grad buffer (residual) is untouched.
    An int mode (a ``Score`` value) is the full step: the mode's score — all
    computed from pre-commit state reachable only inside this pass — is
    materialized into ``grad_buffer`` in the same pass (the reselection reads
    it from there). An unknown int or the OFFLINE sentinel raises a
    ValueError (owned messages) before any work.
    """
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    if emit_mode is not None:
        emit_mode = int(emit_mode)
        try:
            mode = Score(emit_mode)
        except ValueError:
            # Owned text: the raw enum-cast error ("7 is not a valid Score")
            # doesn't say which modes ARE valid.
            raise ValueError(
                f"emit_mode {emit_mode} is not a valid Score — known "
                f"full-step modes: "
                f"{[int(m) for m in Score if m is not Score.OFFLINE]}"
            ) from None
        if mode is Score.OFFLINE:
            raise ValueError(
                f"emit_mode {emit_mode} is Score.OFFLINE — offline-only "
                "scores have no kernel branch to materialize"
            )

    n_elements = param_buffer.numel()
    if n_elements == 0:
        return

    if not (_HAS_TRITON and param_buffer.is_cuda):
        grad_correct_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            mask, emit_mode,
        )
        return

    has_mask = mask is not None
    mask_u8 = mask.to(torch.uint8) if (has_mask and mask.dtype != torch.uint8) else mask
    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _grad_correct_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        mask_u8 if has_mask else grad_buffer,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        HAS_MASK=has_mask,
        # -1 here is the kernel's sparse-step sentinel; Score.OFFLINE's -1
        # never reaches this point (rejected host-side above).
        EMIT_MODE=emit_mode if emit_mode is not None else -1,
    )


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
    selective_reset_torch(grad_buffer, mask_u8, interval, fold)


class SgaAdamW:
    """The SGA AdamW optimizer template: grad-correct + residual reset."""

    decl = DECL

    def __init__(self, params: dict, emit_score: Optional[str] = None):
        self.params = params
        self.residual_mean = bool(params.get("residual_mean", False))
        # Which registry score to materialize on full steps (build-time
        # injection). None/'update' keeps the historical default: the signed
        # AdamW update itself is the topk input. The emission mode int is
        # resolved ONCE here from the registry (the kernel branches on it).
        self.emit_score = emit_score or Score.update.name
        spec = get_score_spec(self.emit_score)
        if spec.score is Score.OFFLINE:
            raise ValueError(
                f"adamw cannot emit score '{self.emit_score}': it is "
                "offline-only (no kernel branch). Emittable: "
                f"{sorted(DECL.emittable_scores)}"
            )
        self._emit_mode = int(spec.score)

    def ensure_state(self, bucket_states):
        pass  # no persistent per-bucket buffers beyond the Support caches

    def _records_pass(self, ctx, emit_mode, use_mask):
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
                emit_mode=emit_mode,
            )

    @torch.no_grad()
    def full_step(self, ctx):
        """Full-sync step: one fused pass per record — m/v commit, the
        configured score materialized into the main grads (the reselection
        reads it from there), and the apply ``w -= lr * update``."""
        self._records_pass(ctx, emit_mode=self._emit_mode, use_mask=False)

    @torch.no_grad()
    def sparse_step(self, ctx):
        """Sparse step: fused grad-correct + apply with the persisted mask;
        the grad buffer (residual) is untouched."""
        self._records_pass(ctx, emit_mode=None, use_mask=True)

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
