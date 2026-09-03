# Copyright (c) 2026, SGA framework contributors.
"""The SGA AdamW optimizer template (kernels ported from the historical
radius scheme).

The AdamW update is split so the weight-update direction can be materialized:

    update = weight_decay * w + (m / bias1) / (sqrt(v / bias2) + eps)

One fused kernel pass per record on EVERY step — m/v update at every
coordinate (sparse steps: masked-out coords contribute a zero gradient,
momentum keeps decaying) and the apply ``w -= lr * update`` ALWAYS fused
(single arithmetic implementation, no Python-side second pass):

- Sparse steps (EMIT_MODE=-1): masked m/v update + fused apply, with the
  clip coeff multiplied in-register. Writeback-emission backends keep the
  selective buffer reset fused post-commit — communicated
  positions store 0, and on the pre-boundary step with ``residual_mean``
  the non-topk positions store their residual times the fp32 host
  reciprocal of the fold window; on all other sparse steps the grad
  buffer outside the mask is untouched so the residual keeps accumulating.
  Compact-emission backends never touch the grad buffer here:
  the pack kernel already zeroed the communicated positions at grad_sync
  time, the commit reads (idx, values) via merge-join, and the fold step
  alone re-reads the residual at inactive coords.
- Full steps (EMIT_MODE>=0, the Score enum value of the configured score):
  the same pass additionally materializes the score into the fp32 main grads
  (the reselection reads it from there). The grad buffer ALIASES the DDP
  bucket's fp32 grad storage (megatron: main_param.grad =
  model_param.main_grad.float() is an identity view for fp32 buckets), so
  the write lands in the bucket where the residual lives. Safety is purely
  ORDER-based, and the invariant holds: on a full step the residual was just
  fully communicated; the engine reads the score views BEFORE the engine-side
  reset zeroes them; only after that does the selection read the
  still-materialized score (the optimizer->score handoff).

Score math is defined by the score registry (sga/plugins/scores.py); the
kernel branches here are its online materialization, spelled as bare ints
with name comments (Triton takes compile-time ints only) and pinned against
the torch reference by tests/test_kernels_gpu.py. The torch reference path
lives in adamw_torch.py (physical isolation: this file is Triton + dispatch
only); CPU unit tests keep working without Triton.

Reset contract (SgaAdamW.reset): sparse steps do NOTHING to the grad buffer
— the fused kernel already cleared the communicated positions (and folded
the residual when configured); zeroing again would destroy the accumulating
residual. Degenerate mask-less buckets and full steps zero everything.
"""

from typing import Optional

import torch

from sga.plugins import OptimizerDecl, PluginParam, resolve_plugin_params
from sga.plugins.optimizer.adamw_torch import grad_correct_torch
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
    # The sparse path consumes the compact-emission (idx, values) handoff
    # directly (_records_pass_compact) — required by compact backends,
    # ignored by writeback ones.
    consumes_compact=True,
    # Derived, not restated: this template implements every registry score's
    # kernel branch (a second template would declare its own branch subset).
    emittable_scores=emittable_score_names(),
    params=(
        PluginParam(
            "residual_mean", bool, default=False,
            doc="pre-fold the accumulated residual by the fold window (the "
                "accumulation span between full syncs) at full-sync "
                "boundaries (residual-mean semantics)",
        ),
    ),
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
        coeff, inv_interval,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        EMIT_MODE: tl.constexpr,
        APPLY_COEFF: tl.constexpr,
        CLEAR: tl.constexpr,
        FOLD: tl.constexpr,
    ):
        """AdamW m/v update + fused apply, with full-step score emission.

        EMIT_MODE is the Score enum value as a compile-time int: -1 = sparse
        (masked commit), >=0 = full step (score materialized into the grad
        buffer). Both paths apply ``w -= lr * update`` in this same pass.

        Sparse-step fusion (CLEAR/APPLY_COEFF/FOLD are sparse-only,
        the dispatch refuses them on full steps):
        - APPLY_COEFF: the clip coeff multiplies the loaded grad in-register
          — bitwise the standalone masked-scale pass (round_fp32(x*coeff)
          whether stored-and-reloaded or kept in a register).
        - CLEAR: active positions store 0 after use — bitwise the standalone
          selective-reset pass; nothing reads the grad buffer between this
          kernel and the next backward's accumulation.
        - FOLD: inactive positions store grad*inv_interval (the host fp32
          reciprocal, exactly the standalone reset kernel's fold). Needs the
          raw grad at inactive positions, so the load widens to full-valid
          only on fold steps (once per period at most) — the m/v commit
          still sees zero there (``grad = where(active, raw, 0)``); the
          residual enters the optimizer only via the boundary full step.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements

        if HAS_MASK:
            mask_val = tl.load(mask_ptr + offsets, mask=valid, other=0)
            active = valid & (mask_val > 0)
        else:
            active = valid

        if FOLD:
            # Fold needs the raw residual at inactive coords for the store,
            # but the m/v commit must still see ZERO there (the old sequence
            # committed via the active-masked load and folded in a separate
            # kernel — the residual enters m only at the boundary full step).
            raw = tl.load(grad_ptr + offsets, mask=valid, other=0.0)
            grad = tl.where(active, raw, 0.0)
        else:
            grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
        if APPLY_COEFF:
            grad = tl.where(active, grad * coeff, grad)

        param = tl.load(param_ptr + offsets, mask=valid, other=0.0)
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
        else:
            if CLEAR:
                tl.store(
                    grad_ptr + offsets,
                    tl.zeros([BLOCK_SIZE], tl.float32), mask=active,
                )
            if FOLD:
                inactive = valid & (mask_val == 0)
                tl.store(
                    grad_ptr + offsets, raw * inv_interval, mask=inactive,
                )
        # Fused apply on every path (sparse: masked-out coords loaded grad=0,
        # so their update is the pure weight-decay/momentum drift).
        param = param - lr * update
        tl.store(param_ptr + offsets, param, mask=valid)


    @triton.jit
    def _grad_correct_compact_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, idx_ptr, val_ptr,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        coeff, inv_interval,
        rec_start, n_elements, k_total,
        BLOCK_SIZE: tl.constexpr, SEARCH_DEPTH: tl.constexpr,
        LANE_DEPTH: tl.constexpr,
        APPLY_COEFF: tl.constexpr, FOLD: tl.constexpr,
    ):
        """Compact-emission grad-correct: merge-join against the bucket-level
        sorted idx/values.

        The record's selected positions form a contiguous segment of the
        sorted bucket idx, which is globally sorted AND unique — uniqueness
        comes from idx's producer ``compact_mask_u8`` (sga/support.py: its
        per-chunk ``nonzero`` construction emits each set coordinate once,
        and its count/fill assert pins the mask bytes to exactly 0/1); the
        pack step only consumes idx. Per block (coords [c0, c0+BLOCK)): two
        lower_bounds on the full idx bracket the segment [p, q), and
        q <= min(k_total, p + BLOCK) follows from dedup plus window width
        alone — idx holds at most one entry per coordinate and the window
        [c0, c1) is exactly BLOCK wide, so at most BLOCK entries can lie
        inside it. The per-lane search therefore runs L1-resident and
        the second lower_bound is bounded to [p, min(k, p+BLOCK)] instead
        of [p, k] (the unbounded form walked ~log2(k) dependent
        scalar loads per block and was the measured merge-join hotspot:
        -7.3ms/step on the real top-k distribution). All bisection
        midpoints are overflow-safe (a + (b-a)//2): the naive (a+b)//2
        wraps int32 once the sum exceeds 2^31, i.e. k_total > 2^30.
        No CLEAR branch by design: the pack kernel zeroed the communicated
        positions at grad_sync time. FOLD keeps the full sequential grad
        read — the residual it folds lives at inactive coords only, which
        the compact buffer does not carry. The m/v/param arithmetic is
        statement-identical to the masked kernel; ulp-level (FMA
        contraction per compilation), never bitwise, vs it — same binary
        reruns are bitwise-stable.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < n_elements
        coords = rec_start + offs

        c0 = rec_start + pid * BLOCK_SIZE
        c1 = c0 + BLOCK_SIZE
        a = 0
        b = k_total
        for _ in tl.static_range(SEARCH_DEPTH):
            run = a < b
            mid = a + (b - a) // 2
            mv = tl.load(idx_ptr + mid, mask=run, other=2147483647)
            a = tl.where(run & (mv < c0), mid + 1, a)
            b = tl.where(run & (mv >= c0), mid, b)
        p = a
        a = p
        # Block-subset upper bound: dedup + the BLOCK-wide coordinate window
        # cap q - p at BLOCK, so ~log2(BLOCK) active rounds converge here.
        # Do NOT widen back to k_total — a pure performance regression
        # (cut ~28 active scalar rounds to <= 10) that stays
        # mathematically correct, so no test would catch it.
        b = tl.minimum(k_total, p + BLOCK_SIZE)
        for _ in tl.static_range(SEARCH_DEPTH):
            run = a < b
            mid = a + (b - a) // 2
            mv = tl.load(idx_ptr + mid, mask=run, other=2147483647)
            a = tl.where(run & (mv < c1), mid + 1, a)
            b = tl.where(run & (mv >= c1), mid, b)
        q = a

        la = tl.zeros([BLOCK_SIZE], tl.int32) + p
        lb = tl.zeros([BLOCK_SIZE], tl.int32) + q
        for _ in tl.static_range(LANE_DEPTH):
            run = la < lb
            mid = la + (lb - la) // 2
            mv = tl.load(idx_ptr + mid, mask=run, other=2147483647)
            take = run & (mv < coords)
            la = tl.where(take, mid + 1, la)
            lb = tl.where(run & (~take), mid, lb)
        inseg = la < q
        fidx = tl.load(idx_ptr + la, mask=inseg, other=-1)
        found = valid & inseg & (fidx == coords)
        val = tl.load(val_ptr + la, mask=found, other=0.0)

        grad = tl.where(found, val, 0.0)
        if APPLY_COEFF:
            grad = tl.where(found, grad * coeff, grad)

        if FOLD:
            raw = tl.load(grad_ptr + offs, mask=valid, other=0.0)

        param = tl.load(param_ptr + offs, mask=valid, other=0.0)
        m = tl.load(m_ptr + offs, mask=valid, other=0.0)
        v = tl.load(v_ptr + offs, mask=valid, other=0.0)

        m_new = beta1 * m + (1.0 - beta1) * grad
        v_new = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2
        update = weight_decay * param + m_hat / (tl.sqrt(v_hat) + eps)

        tl.store(m_ptr + offs, m_new, mask=valid)
        tl.store(v_ptr + offs, v_new, mask=valid)
        if FOLD:
            tl.store(grad_ptr + offs, raw * inv_interval,
                     mask=valid & (~found))
        param = param - lr * update
        tl.store(param_ptr + offs, param, mask=valid)


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
    coeff: float = 1.0,
    clear: bool = False,
    fold: bool = False,
    interval: float = 1.0,
) -> None:
    """Run the grad-correct step on contiguous same-size 1-D buffers.

    Callers pass ``.view(-1)`` flats (the engine flattens every buffer;
    a 2-D grad against the 1-D bucket mask would broadcast-error here).
    Updates m/v in place and always fuses the apply ``w -= lr * update``.
    ``emit_mode=None`` is the sparse step: with ``mask``, positions outside
    it contribute a zero gradient. An int mode (a ``Score`` value) is the
    full step: the mode's score — all computed from pre-commit state
    reachable only inside this pass — is materialized into ``grad_buffer``
    in the same pass (the reselection reads it from there). An unknown int
    or the OFFLINE sentinel raises a ValueError (owned messages) before any
    work.

    Sparse-step fusion knobs (refused on full steps and without a
    mask — they only make sense for the masked commit path):
    - ``coeff != 1.0`` multiplies the loaded grads in-register — bitwise
      equal to a standalone masked-scale pass before this call.
    - ``clear`` zeroes the mask-active positions after use — bitwise equal
      to a standalone selective-reset pass after this call.
    - ``fold`` multiplies mask-inactive positions by 1/``interval`` (the
      host fp32 reciprocal on the Triton path) — bitwise equal to the
      standalone reset kernel's fold branch.
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

    sparse_fusion = (coeff != 1.0) or clear or fold
    if sparse_fusion and (emit_mode is not None or mask is None):
        raise ValueError(
            f"grad-correct sparse fusion (coeff={coeff!r}, clear={clear!r}, "
            f"fold={fold!r}) requires a sparse step (emit_mode=None) with a "
            "mask — full steps scale in place engine-side and never clear "
            "(the reselection reads the materialized score from the grad "
            "buffer)"
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
            coeff=coeff, clear=clear, fold=fold, interval=interval,
        )
        return

    has_mask = mask is not None
    mask_u8 = mask.to(torch.uint8) if (has_mask and mask.dtype != torch.uint8) else mask
    # fp32 reciprocal, computed exactly as the standalone reset kernel did
    # (bitwise-matches CUDA scalar division on the multiply path).
    inv_interval = float(
        (torch.ones(1, dtype=torch.float32) / interval).item()
    ) if fold else 1.0
    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _grad_correct_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        mask_u8 if has_mask else grad_buffer,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        coeff, inv_interval,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        HAS_MASK=has_mask,
        # -1 here is the kernel's sparse-step sentinel; Score.OFFLINE's -1
        # never reaches this point (rejected host-side above).
        EMIT_MODE=emit_mode if emit_mode is not None else -1,
        APPLY_COEFF=coeff != 1.0,
        CLEAR=clear,
        FOLD=fold,
    )


def adamw_grad_correct_compact(
    param_buffer: torch.Tensor,
    grad_buffer: torch.Tensor,
    exp_avg_buffer: torch.Tensor,
    exp_avg_sq_buffer: torch.Tensor,
    idx: torch.Tensor,
    values: torch.Tensor,
    rec_start: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    coeff: float = 1.0,
    fold: bool = False,
    interval: float = 1.0,
) -> None:
    """Compact-emission grad-correct on contiguous same-size 1-D buffers.

    Sparse step only (no score emission — compact values never carry the
    materialized score path). ``idx``/``values`` are the BUCKET-level sorted
    int32 indices and the communicated payload; ``rec_start`` is this
    record's element offset inside the bucket. ``grad_buffer`` is read only
    under ``fold`` (the accumulated residual at inactive coords). The clip
    ``coeff`` multiplies the loaded values in-register; the communicated
    positions of ``grad_buffer`` are NOT touched here — the pack kernel
    zeroed them at grad_sync time (there is deliberately no clear knob).

    ulp-class contract: values-level math is single-rounding (bitwise the
    masked path's), the m/v/param tail differs from the masked kernel by
    per-compilation FMA contraction only; the same binary reruns
    bitwise-stable (pinned by tests/test_kernels_gpu.py).
    """
    if not idx.is_contiguous() or not values.is_contiguous():
        raise ValueError(
            "adamw_grad_correct_compact requires contiguous idx/values "
            "(the gather buffer slices are; a strided view would silently "
            "corrupt the merge-join)"
        )
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    n_elements = param_buffer.numel()
    if n_elements == 0:
        return
    k_total = idx.numel()
    # int32 offset contract, refused loudly instead of wrapping silently
    # (the compact_mask_u8 precedent): the kernel forms p + BLOCK_SIZE
    # (<= k_total + BLOCK) and c1 = c0 + BLOCK_SIZE (<= rec_start +
    # n_elements + BLOCK) as int32 — crossing 2^31 wraps either negative
    # and whole blocks drop out of the merge-join silently.
    assert k_total + _BLOCK_SIZE < 2 ** 31, (
        f"adamw_grad_correct_compact: k_total + BLOCK_SIZE = "
        f"{k_total + _BLOCK_SIZE} must stay < 2^31 (int32 idx offsets)"
    )
    assert rec_start + n_elements + _BLOCK_SIZE < 2 ** 31, (
        f"adamw_grad_correct_compact: rec_start + n_elements + BLOCK_SIZE = "
        f"{rec_start + n_elements + _BLOCK_SIZE} must stay < 2^31 (int32 "
        "coordinate offsets)"
    )
    if not (_HAS_TRITON and param_buffer.is_cuda):
        # Fallback: scatter this record's value segment back into the
        # (pack-cleared) grad buffer and run the plain masked reference —
        # end state identical to the compact path (values consumed, actives
        # re-zeroed by clear, inactives folded when configured).
        lo = 0 if k_total == 0 else int(
            torch.searchsorted(idx, rec_start).item())
        hi = 0 if k_total == 0 else int(
            torch.searchsorted(idx, rec_start + n_elements).item())
        mask = torch.zeros(n_elements, dtype=torch.uint8,
                           device=param_buffer.device)
        if hi > lo:
            local = idx[lo:hi].long() - rec_start
            mask[local] = 1
            grad_buffer[local] = values[lo:hi]
        grad_correct_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            mask, None,
            coeff=coeff, clear=True, fold=fold, interval=interval,
        )
        return

    # fp32 reciprocal, computed exactly as the masked dispatcher does.
    inv_interval = float(
        (torch.ones(1, dtype=torch.float32) / interval).item()
    ) if fold else 1.0
    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _grad_correct_compact_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        idx, values,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        coeff, inv_interval,
        rec_start, n_elements, k_total,
        BLOCK_SIZE=_BLOCK_SIZE,
        SEARCH_DEPTH=max(1, (k_total + 1).bit_length()),
        # Lane search depth: a lower bound over a <= BLOCK_SIZE segment has
        # BLOCK_SIZE + 1 candidate positions, i.e. ceil(log2(BLOCK + 1)) ==
        # bit_length(BLOCK) iterations. One short and the loop exits with
        # (la, lb) un-converged inside a fully-selected block: the equality
        # probe reads idx[la] at the wrong lane and silently drops that
        # coordinate (grad committed as 0).
        LANE_DEPTH=_BLOCK_SIZE.bit_length(),
        APPLY_COEFF=coeff != 1.0,
        FOLD=fold,
    )


class SgaAdamW:
    """The SGA AdamW optimizer template: grad-correct + residual reset."""

    decl = DECL

    def __init__(self, params: dict, emit_score: Optional[str] = None):
        params = resolve_plugin_params("optimizer", "adamw", DECL, dict(params))
        self.params = params
        self.residual_mean = bool(params["residual_mean"])
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

    def _records_pass(self, ctx, emit_mode, use_mask,
                      coeff=1.0, clear=False, fold=False, interval=1.0):
        for rec in ctx.records:
            state = rec._state
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
            m = rec.mask_u8() if use_mask else None
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
                mask=m,
                emit_mode=emit_mode,
                # Degenerate mask-less bucket on a sparse step: plain full
                # commit (old behavior) — the fusion knobs require a mask,
                # and the engine scaled such a record explicitly; reset's
                # mask-less branch zeroes it afterwards.
                coeff=coeff if m is not None else 1.0,
                clear=clear and m is not None,
                fold=fold and m is not None,
                interval=interval,
            )

    def _records_pass_compact(self, ctx, coeff=1.0, fold=False, interval=1.0):
        """Compact-emission variant: consumes the backend's (idx, values)
        directly per record; the grad buffer is only read on fold steps."""
        for rec in ctx.records:
            state = rec._state
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
            bs = rec.bucket_state
            values = ctx.compact_values.get(id(bs))
            if values is None:
                # Degenerate mask-less bucket (no compact values): plain
                # full-buffer commit, mirror of the masked path's contract
                # (the engine scaled such a record explicitly; reset's
                # mask-less branch zeroes it afterwards).
                adamw_grad_correct(
                    param_buffer=rec.main_param.data.view(-1),
                    grad_buffer=rec.grad.data.view(-1),
                    exp_avg_buffer=state["exp_avg"].view(-1),
                    exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
                    lr=rec.lr, beta1=rec.beta1, beta2=rec.beta2,
                    eps=rec.eps, weight_decay=rec.weight_decay,
                    step=rec.step, mask=None, emit_mode=None,
                )
                continue
            idx = ctx.support.buckets[bs.support_index].idx
            adamw_grad_correct_compact(
                param_buffer=rec.main_param.data.view(-1),
                grad_buffer=rec.grad.data.view(-1),
                exp_avg_buffer=state["exp_avg"].view(-1),
                exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
                idx=idx, values=values, rec_start=rec.start,
                lr=rec.lr, beta1=rec.beta1, beta2=rec.beta2,
                eps=rec.eps, weight_decay=rec.weight_decay,
                step=rec.step,
                coeff=coeff, fold=fold, interval=interval,
            )

    @torch.no_grad()
    def full_step(self, ctx):
        """Full-sync step: one fused pass per record — m/v commit, the
        configured score materialized into the main grads (the reselection
        reads it from there), and the apply ``w -= lr * update``."""
        self._records_pass(ctx, emit_mode=self._emit_mode, use_mask=False)

    @torch.no_grad()
    def sparse_step(self, ctx):
        """Sparse step: fused grad-correct + apply. Compact-emission backends
        (``ctx.compact_values`` set) consume (idx, values) directly — the
        communicated positions were zeroed by the pack kernel at grad_sync
        time, so there is no clear here; the clip coeff multiplies
        in-register and the residual fold (when ``residual_mean`` is set on
        the pre-boundary step) reads the grad buffer at inactive coords in
        the same pass. Writeback backends keep the masked kernel with the
        fused clear (semantics unchanged)."""
        fold = self.residual_mean and ctx.next_step_is_full
        if getattr(ctx, "compact_values", None):
            self._records_pass_compact(
                ctx, coeff=float(getattr(ctx, "apply_coeff", 1.0)),
                fold=fold, interval=float(ctx.fold_window),
            )
            return
        self._records_pass(
            ctx, emit_mode=None, use_mask=True,
            coeff=float(getattr(ctx, "apply_coeff", 1.0)),
            clear=True, fold=fold, interval=float(ctx.fold_window),
        )

    def reset(self, ctx, was_full: bool):
        """Sparse steps need no grad-buffer work anymore: writeback backends
        got the clear inside grad_correct's fused pass; compact backends got
        it from the pack kernel at grad_sync time. Zeroing here would
        destroy the accumulating residual either way. Full steps (and the
        degenerate mask-less bucket): zero all. ``extra_main_grads`` zeroing
        stays."""
        for bs in ctx.bucket_states:
            if was_full or bs.active_mask_u8 is None:
                bs.grad_data.zero_()
            for grad in bs.extra_main_grads:
                grad.zero_()
