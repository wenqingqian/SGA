# Copyright (c) 2026, SGA framework contributors.
"""The SGA AdamS optimizer template — AdamW with a first-moment-memory
second moment.

AdamS is AdamW feature-complete (every knob, every emission mode, every
guard of ``adamw.py`` is replicated) with exactly one mathematical
difference, the recursion of the second moment:

    m = beta1 * m_old + (1 - beta1) * g              (AdamW-identical)
    v = beta2 * m_old^2 + (1 - beta2) * g^2          (AdamS)

``v`` is rebuilt from the PRE-commit first moment ``m_old`` — the same
read point the ``abs_m`` score uses — and never reads the stored ``v``:
inside the fused kernel ``v`` is computed BEFORE ``m`` is updated, so
``m_old`` is the pre-commit value by construction. The stored v buffer is
write-only on this path (an inherited AdamW v0 is simply overwritten on
the first step). The weight decay stays decoupled exactly like AdamW's:

    update = weight_decay * w + m_new / (sqrt(v_new) + eps)

There is NO bias correction: neither in the kernel (no correction term
exists anywhere in this file) nor in the score formulas (the registry's
bias-correction channels are forced to 1.0 for AdamS — see
``sga/plugins/scores.py``). The wrapper therefore takes no ``step``
argument at all. Rationale: with v rebuilt from m, the corrected first
step would overshoot by sqrt(1/(1-beta2)) ~ 3.16x; uncorrected, the
first step is bounded by |m/sqrt(v)| <= 1/sqrt(1-beta2) and
self-corrects within a few steps.

Everything else is a statement-level port of the AdamW template (same
step split, same sparse-fusion knobs coeff/clear/fold, same compact
merge-join with the per-cycle directory table, same boundary dual-lr
whose split multiplies the WHOLE applied step, weight-decay component
included). Self-contained on every engine path: sparse steps, full-sync
steps and the dense bootstrap phases all run THIS template — the only
run shape that would step a Megatron optimizer instead is shadow mode,
which config validation refuses for AdamS.

The torch reference lives in ``adams_torch.py`` (physical isolation: this
file is Triton + dispatch only, the same split as the AdamW template's
``adamw_torch.py``); it calls the score REGISTRY formulas with
``optimizer="adams"`` — the single source of the score math — and mirrors
the kernel's m/v commit skeleton. The merge-join directory table is
optimizer-agnostic (pure idx searchsorted data), so the builder is shared
with adamw.py.
"""

import math
from typing import Optional

import torch

from sga.config import get_args
from sga.plugins.optimizer.adams_torch import adams_grad_correct_torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

# Directory-table builder for the compact merge-join (compact_search=table):
# the table is a pure function of the bucket idx (lower_bound(idx, j*1024)),
# so the AdamW file's builder IS the AdamS one — one kernel, two templates.
from sga.plugins.optimizer.adamw import build_dir_table  # noqa: F401


# Score name -> the kernel's compile-time EMIT_MODE branch (the AdamW map's
# twin, same names, same ints, same three duties: launch mapping, the loud
# "registered but no kernel branch" refusal, and the human-readable branch
# inventory kept next to the kernel's own if-chain).
_TRITON_SCORE_MAP = {
    "update": 0,   # wd*w + U(g), signed — the default topk input
    "abs_g": 1,    # |g| of the committed gradient
    "abs_m": 2,    # |m| of the pre-commit first moment
    "coi": 3,      # |U(g) - U(0)| (AdamS semantics: provisional)
}

_BLOCK_SIZE = 1024


if _HAS_TRITON:

    @triton.jit
    def _adams_grad_correct_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, mask_ptr,
        lr, beta1, beta2, eps, weight_decay,
        coeff, inv_interval, lr_inactive,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        EMIT_MODE: tl.constexpr,
        APPLY_COEFF: tl.constexpr,
        CLEAR: tl.constexpr,
        FOLD: tl.constexpr,
        LR_SPLIT: tl.constexpr,
    ):
        """AdamS m/v update + fused apply, with full-step score emission.

        Statement-identical to the AdamW masked/full kernel
        (``_grad_correct_kernel``) except the moment tail: the second
        moment is rebuilt from the PRE-commit first moment —
        ``v = beta2 * m_old^2 + (1 - beta2) * g^2`` — so v_new is computed
        from the loaded ``m_old`` BEFORE ``m`` is updated, the v buffer is
        never read, and no bias correction exists (the correction terms
        are structurally absent, not divided out).

        EMIT_MODE is the _TRITON_SCORE_MAP branch id as a compile-time
        int: -1 = sparse (masked commit), >=0 = full step (score
        materialized into the grad buffer). Both paths apply
        ``w -= lr * update`` in this same pass. Sparse-step fusion
        (CLEAR/APPLY_COEFF/FOLD) and the full-step dual-lr (LR_SPLIT)
        carry the AdamW contract verbatim — see the AdamW kernel's
        docstring for the knob-by-knob semantics; only the update
        direction itself differs.
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
            # but the m/v commit must still see ZERO there (the residual
            # enters m only at the boundary full step) — AdamW's contract.
            raw = tl.load(grad_ptr + offsets, mask=valid, other=0.0)
            grad = tl.where(active, raw, 0.0)
        elif LR_SPLIT:
            grad = tl.load(grad_ptr + offsets, mask=valid, other=0.0)
        else:
            grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
        if APPLY_COEFF:
            grad = tl.where(active, grad * coeff, grad)

        param = tl.load(param_ptr + offsets, mask=valid, other=0.0)
        # m_old: the PRE-commit first moment — the AdamS v recursion's
        # memory base AND the abs_m score's read point (one load, both
        # consumers). The v buffer is deliberately not loaded: v_new
        # never reads the stored v.
        m_old = tl.load(m_ptr + offsets, mask=valid, other=0.0)

        # v FIRST (from the pre-commit m_old), then m — the order is the
        # spec: v = beta2 * m_old^2 + (1 - beta2) * g^2.
        v_new = beta2 * (m_old * m_old) + (1.0 - beta2) * grad * grad
        m_new = beta1 * m_old + (1.0 - beta1) * grad
        # No bias correction: AdamS applies the raw moments.
        update = weight_decay * param + m_new / (tl.sqrt(v_new) + eps)

        if EMIT_MODE == 1:
            # abs_g: |g| of the gradient being committed this step.
            score = tl.abs(grad)
        elif EMIT_MODE == 2:
            # abs_m: |m| of the PRE-commit momentum (the v base's own read).
            score = tl.abs(m_old)
        elif EMIT_MODE == 3:
            # coi: |U(g) - U(0)| with the PRE-commit moments. U(0) — the
            # zero-gradient counterfactual of the SAME commit — rebuilds
            # its v from m_old too: v_0 = beta2 * m_old^2 + 0. AdamS coi
            # is provisional: the m-based-v counterfactual contrast has
            # no validated reading yet, it ships for the frozen score
            # axis's completeness.
            u_0 = (beta1 * m_old) / (
                tl.sqrt(beta2 * (m_old * m_old)) + eps
            )
            score = tl.abs(m_new / (tl.sqrt(v_new) + eps) - u_0)

        tl.store(m_ptr + offsets, m_new, mask=valid)
        tl.store(v_ptr + offsets, v_new, mask=valid)
        if EMIT_MODE >= 0:
            if EMIT_MODE == 0:
                # update: the signed update itself is the materialized score.
                tl.store(grad_ptr + offsets, update, mask=valid)
            else:
                # Named score: it takes over the grad buffer; the
                # reselection reads it from there exactly like the
                # default update score.
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
        # Fused apply on every path (AdamW's contract: the split
        # multiplies the WHOLE update, weight-decay term included).
        if LR_SPLIT:
            param = param - tl.where(active, lr, lr_inactive) * update
        else:
            param = param - lr * update
        tl.store(param_ptr + offsets, param, mask=valid)

    @triton.jit
    def _adams_grad_correct_compact_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, idx_ptr, val_ptr,
        lr, beta1, beta2, eps, weight_decay,
        coeff, inv_interval,
        rec_start, n_elements, k_total,
        BLOCK_SIZE: tl.constexpr, SEARCH_DEPTH: tl.constexpr,
        LANE_DEPTH: tl.constexpr,
        APPLY_COEFF: tl.constexpr, FOLD: tl.constexpr,
    ):
        """Compact-emission AdamS grad-correct: merge-join against the
        bucket-level sorted idx/values.

        The lookup machinery (bracket bounds, lane search, depth laws,
        overflow-safe midpoints) is the AdamW plain compact kernel's,
        statement for statement — the search phase has no optimizer
        arithmetic in it; see the AdamW kernel's docstring for the full
        geometry argument. No CLEAR branch by design (the pack kernel
        zeroed the communicated positions at grad_sync time); FOLD keeps
        the full sequential grad read. The m/v/param tail is the AdamS
        recursion above — v rebuilt from the pre-commit m, no bias
        correction; ulp-class contract vs the masked kernel unchanged
        (per-compilation FMA contraction, same binary reruns stable).
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
        # cap q - p at BLOCK — do NOT widen back to k_total (a pure
        # performance regression that stays mathematically correct, so no
        # test would catch it; see the AdamW kernel's note).
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
        m_old = tl.load(m_ptr + offs, mask=valid, other=0.0)

        v_new = beta2 * (m_old * m_old) + (1.0 - beta2) * grad * grad
        m_new = beta1 * m_old + (1.0 - beta1) * grad
        update = weight_decay * param + m_new / (tl.sqrt(v_new) + eps)

        tl.store(m_ptr + offs, m_new, mask=valid)
        tl.store(v_ptr + offs, v_new, mask=valid)
        if FOLD:
            tl.store(grad_ptr + offs, raw * inv_interval,
                     mask=valid & (~found))
        param = param - lr * update
        tl.store(param_ptr + offs, param, mask=valid)

    @triton.jit
    def _adams_grad_correct_compact_table_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr, idx_ptr, val_ptr, tab_ptr,
        lr, beta1, beta2, eps, weight_decay,
        coeff, inv_interval,
        rec_start, n_elements, k_total,
        BLOCK_SIZE: tl.constexpr, TABLE_DEPTH: tl.constexpr,
        LANE_DEPTH: tl.constexpr,
        APPLY_COEFF: tl.constexpr, FOLD: tl.constexpr,
    ):
        """Table-bracketed merge-join (compact_search=table), AdamS tail.

        Identical contract to _adams_grad_correct_compact_kernel — same
        records, same idx/values, same found set — with both scalar
        lower_bounds seeded from the directory table instead of walking
        [0, k_total] full-depth. The numerical-invariance chain (sorted
        duplicate-free idx => unique lower_bound fixed point; table ==
        searchsorted 'left'; tight brackets contain the true p/q; a
        containing bracket bisects to the same p/q) is optimizer-agnostic
        — the AdamW table kernel's docstring carries it in full. The
        table itself is a per-cycle lazy derived field of BucketSupport
        (the shared build_dir_table builder), frozen for the cycle's
        lifetime, so staleness is structurally impossible.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < n_elements
        coords = rec_start + offs

        c0 = rec_start + pid * BLOCK_SIZE
        c1 = c0 + BLOCK_SIZE
        # The table stride is hardcoded below (b = c0 >> 10) while
        # BLOCK_SIZE is a constexpr: pin the coupling so a BLOCK change
        # cannot silently desync the table lookup from the kernel's
        # blocks (the builder's _BLOCK_SIZE is the same constant).
        tl.static_assert(BLOCK_SIZE == 1024)
        b = c0 >> 10
        r = c0 & (BLOCK_SIZE - 1)

        t0 = tl.load(tab_ptr + b)
        a = t0
        bb = tl.minimum(k_total, t0 + r)
        for _ in tl.static_range(TABLE_DEPTH):
            run = a < bb
            mid = a + (bb - a) // 2
            mv = tl.load(idx_ptr + mid, mask=run, other=2147483647)
            a = tl.where(run & (mv < c0), mid + 1, a)
            bb = tl.where(run & (mv >= c0), mid, bb)
        p = a

        t1 = tl.load(tab_ptr + b + 1)
        a2 = t1
        b2 = tl.minimum(k_total, t1 + r)
        for _ in tl.static_range(TABLE_DEPTH):
            run = a2 < b2
            mid = a2 + (b2 - a2) // 2
            mv = tl.load(idx_ptr + mid, mask=run, other=2147483647)
            a2 = tl.where(run & (mv < c1), mid + 1, a2)
            b2 = tl.where(run & (mv >= c1), mid, b2)
        q = a2

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
        m_old = tl.load(m_ptr + offs, mask=valid, other=0.0)

        v_new = beta2 * (m_old * m_old) + (1.0 - beta2) * grad * grad
        m_new = beta1 * m_old + (1.0 - beta1) * grad
        update = weight_decay * param + m_new / (tl.sqrt(v_new) + eps)

        tl.store(m_ptr + offs, m_new, mask=valid)
        tl.store(v_ptr + offs, v_new, mask=valid)
        if FOLD:
            tl.store(grad_ptr + offs, raw * inv_interval,
                     mask=valid & (~found))
        param = param - lr * update
        tl.store(param_ptr + offs, param, mask=valid)


def adams_grad_correct(
    param_buffer: torch.Tensor,
    grad_buffer: torch.Tensor,
    exp_avg_buffer: torch.Tensor,
    exp_avg_sq_buffer: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    mask: torch.Tensor = None,
    emit_score: Optional[str] = None,
    coeff: float = 1.0,
    clear: bool = False,
    fold: bool = False,
    interval: float = 1.0,
    inactive_lr_scale: float = 1.0,
) -> None:
    """Run the AdamS grad-correct step on contiguous same-size 1-D buffers.

    Same contract as ``adamw_grad_correct`` (which see for the full knob
    semantics) with the AdamS differences: v is rebuilt from the
    pre-commit first moment and there are no bias corrections, so the
    function takes NO ``step`` argument. Callers pass ``.view(-1)``
    flats. Updates m/v in place and always fuses the apply
    ``w -= lr * update``. ``emit_score=None`` is the sparse step: with
    ``mask``, positions outside it contribute a zero gradient. A score
    NAME is the full step: the score's math — computed from pre-commit
    state — is materialized into ``grad_buffer`` in the same pass. A
    name without a kernel branch (``_TRITON_SCORE_MAP``) raises a
    ValueError (owned message) before any work.
    """
    if emit_score is not None and emit_score not in _TRITON_SCORE_MAP:
        # Owned text: covers both an unregistered name and a registered
        # score without a kernel branch — either way there is no branch
        # to materialize it, and silent fallback would train on wrong
        # numerics.
        raise ValueError(
            f"emit_score '{emit_score}' has no _adams_grad_correct_kernel "
            f"branch (implemented: {sorted(_TRITON_SCORE_MAP)})")

    sparse_fusion = (coeff != 1.0) or clear or fold
    if sparse_fusion and (emit_score is not None or mask is None):
        raise ValueError(
            f"grad-correct sparse fusion (coeff={coeff!r}, clear={clear!r}, "
            f"fold={fold!r}) requires a sparse step (emit_score=None) with a "
            "mask — full steps scale in place engine-side and never clear "
            "(the reselection reads the materialized score from the grad "
            "buffer)"
        )
    if inactive_lr_scale != 1.0:
        if not (math.isfinite(inactive_lr_scale)
                and inactive_lr_scale >= 0.0):
            raise ValueError(
                "grad-correct inactive_lr_scale must be a finite float "
                f">= 0.0 (got {inactive_lr_scale!r})"
            )
        if emit_score is None or mask is None:
            raise ValueError(
                f"grad-correct boundary dual-lr (inactive_lr_scale="
                f"{inactive_lr_scale!r}) requires a full step (emit_score "
                "set) with a mask — only the boundary update's lr splits; "
                "sparse steps and mask-less records take plain lr"
            )
    if (mask is not None and emit_score is not None
            and inactive_lr_scale == 1.0):
        raise ValueError(
            "grad-correct masked-commit full step (mask + emit_score with "
            "inactive_lr_scale=1.0) is refused: the shape predates the "
            "dual-lr knob and nothing in production produces it — pass "
            "inactive_lr_scale != 1.0 for the boundary split, or drop "
            "the mask for the mask-blind full step"
        )

    n_elements = param_buffer.numel()
    if n_elements == 0:
        return

    if not (_HAS_TRITON and param_buffer.is_cuda):
        adams_grad_correct_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            lr, beta1, beta2, eps, weight_decay,
            mask, emit_score,
            coeff=coeff, clear=clear, fold=fold, interval=interval,
            inactive_lr_scale=inactive_lr_scale,
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
    _adams_grad_correct_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        mask_u8 if has_mask else grad_buffer,
        lr, beta1, beta2, eps, weight_decay,
        coeff, inv_interval, lr * inactive_lr_scale,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        HAS_MASK=has_mask,
        # -1 is the kernel's sparse-step sentinel; a real branch id comes
        # from the map (an unbranched name already raised above).
        EMIT_MODE=(_TRITON_SCORE_MAP[emit_score]
                   if emit_score is not None else -1),
        APPLY_COEFF=coeff != 1.0,
        CLEAR=clear,
        FOLD=fold,
        LR_SPLIT=inactive_lr_scale != 1.0,
    )


def adams_grad_correct_compact(
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
    coeff: float = 1.0,
    fold: bool = False,
    interval: float = 1.0,
    dir_table: Optional[torch.Tensor] = None,
) -> None:
    """Compact-emission AdamS grad-correct on contiguous same-size 1-D
    buffers.

    Same contract as ``adamw_grad_correct_compact`` (which see) with the
    AdamS moment tail (v from the pre-commit m, no bias corrections, no
    ``step`` argument). Sparse step only; ``idx``/``values`` are the
    BUCKET-level sorted int32 indices and the communicated payload;
    ``rec_start`` is this record's element offset inside the bucket.
    ``grad_buffer`` is read only under ``fold``. The communicated
    positions of ``grad_buffer`` are NOT touched here — the pack kernel
    zeroed them at grad_sync time (no clear knob, by design).

    ``dir_table`` (compact_search=table) routes to the table-bracketed
    kernel; None keeps the plain full-depth bisection form. Both forms
    share this wrapper's contract; ignored on the torch fallback path.

    ulp-class contract: values-level math is single-rounding (bitwise
    the masked path's), the m/v/param tail differs from the masked
    kernel by per-compilation FMA contraction only; the same binary
    reruns bitwise-stable.
    """
    if not idx.is_contiguous() or not values.is_contiguous():
        raise ValueError(
            "adams_grad_correct_compact requires contiguous idx/values "
            "(the gather buffer slices are; a strided view would silently "
            "corrupt the merge-join)"
        )
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
        f"adams_grad_correct_compact: k_total + BLOCK_SIZE = "
        f"{k_total + _BLOCK_SIZE} must stay < 2^31 (int32 idx offsets)"
    )
    assert rec_start + n_elements + _BLOCK_SIZE < 2 ** 31, (
        f"adams_grad_correct_compact: rec_start + n_elements + BLOCK_SIZE = "
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
        adams_grad_correct_torch(
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            lr, beta1, beta2, eps, weight_decay,
            mask, None,
            coeff=coeff, clear=True, fold=fold, interval=interval,
        )
        return

    # fp32 reciprocal, computed exactly as the masked dispatcher does.
    inv_interval = float(
        (torch.ones(1, dtype=torch.float32) / interval).item()
    ) if fold else 1.0
    grid = ((n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    if dir_table is not None:
        # Host contract for the table argument (the idx/values checks'
        # failure class — refuse loudly instead of corrupting silently):
        # the kernel loads table[b] and table[b + 1] (b = c0 >> 10) as
        # raw int32 brackets, so a non-int32 or strided table feeds
        # garbage brackets and a table built for a smaller bucket reads
        # out of bounds — both silently drop gradients.
        assert dir_table.dtype == torch.int32 and dir_table.is_contiguous(), (
            f"adams_grad_correct_compact: dir_table must be contiguous "
            f"int32 (raw int32 bracket loads), got {dir_table.dtype}, "
            f"contiguous={dir_table.is_contiguous()}"
        )
        assert dir_table.numel() >= ((rec_start + n_elements - 1) >> 10) + 2, (
            f"adams_grad_correct_compact: dir_table has "
            f"{dir_table.numel()} entries but the record's last block "
            f"({(rec_start + n_elements - 1) >> 10}) needs its b+1 "
            f"lookup (>= {((rec_start + n_elements - 1) >> 10) + 2} "
            "entries) — table built for a smaller bucket reads out of "
            "bounds"
        )
        _adams_grad_correct_compact_table_kernel[grid](
            param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
            idx, values, dir_table,
            lr, beta1, beta2, eps, weight_decay,
            coeff, inv_interval,
            rec_start, n_elements, k_total,
            BLOCK_SIZE=_BLOCK_SIZE,
            # Both table brackets hold <= 1024 candidates (r = c0 mod
            # BLOCK <= 1023), which 10 rounds pin exactly. One round
            # short and the r=1023 geometry bites: closed width-1023
            # brackets with p/q at the far edge exit 9 rounds with
            # a = true - 1, and the q-side drop commits every
            # fully-selected block's last coordinate as 0.
            TABLE_DEPTH=10,
            LANE_DEPTH=_BLOCK_SIZE.bit_length(),
            APPLY_COEFF=coeff != 1.0,
            FOLD=fold,
        )
        return
    _adams_grad_correct_compact_kernel[grid](
        param_buffer, grad_buffer, exp_avg_buffer, exp_avg_sq_buffer,
        idx, values,
        lr, beta1, beta2, eps, weight_decay,
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


class SgaAdamS:
    """The SGA AdamS optimizer template: grad-correct + residual reset.

    Feature-identical to SgaAdamW (the knobs, the guards, the reset
    contract); the step math is AdamS — see the module docstring.
    """

    def __init__(self):
        args = get_args()
        self.residual_mean = bool(args.sga_optimizer_residual_mean)
        # Compact-path idx alignment search: 'table' seeds both
        # lower_bound brackets from the per-cycle directory table;
        # 'bisection' is the plain full-depth form. Dispatched per call
        # (host branch) — the kernels stay branch-free. No-op on
        # writeback-emission runs (they never reach
        # _records_pass_compact) and on the torch fallback (it ignores
        # the table).
        self.compact_search = args.sga_optimizer_compact_search
        # Boundary dual-lr: on full-sync steps the apply runs at
        # lr * scale on the mask-inactive (never-communicated) coords.
        # 1.0 keeps the single-lr behavior bit-identical; the shadow/period
        # combination rules live in config validation (shadow is refused
        # for AdamS outright) — on the flag routes its domain rule
        # refuses non-finite too (argparse float typing ACCEPTS
        # "nan"/"inf"); this guard covers direct construction
        # (tests/replay), where the negative and non-finite legs are both
        # reachable.
        self.full_sync_inactive_lr_scale = float(
            args.sga_optimizer_full_sync_inactive_lr_scale)
        # -0.0 normalizes to 0.0: the same tail freeze, without the
        # sign-of-zero flips on params that are exactly -0.0
        if self.full_sync_inactive_lr_scale == 0.0:
            self.full_sync_inactive_lr_scale = 0.0
        if not (math.isfinite(self.full_sync_inactive_lr_scale)
                and self.full_sync_inactive_lr_scale >= 0.0):
            raise ValueError(
                "sga_optimizer_full_sync_inactive_lr_scale must be a "
                "finite float >= 0.0 (got "
                f"{self.full_sync_inactive_lr_scale!r})")
        # Which registry score to materialize on full steps — read straight
        # off the config (validation already required it). The name without
        # a kernel branch fails loud HERE, at build time: the kernel would
        # otherwise silently skip the materialization and the reselection
        # would rank raw grads.
        self.emit_score = args.sga_score
        try:
            _TRITON_SCORE_MAP[self.emit_score]
        except KeyError:
            raise ValueError(
                f"adams kernel has no EMIT_MODE branch for score "
                f"'{self.emit_score}' (implemented: "
                f"{sorted(_TRITON_SCORE_MAP)})") from None

    def ensure_state(self, bucket_states):
        pass  # no persistent per-bucket buffers beyond the Support caches

    def _records_pass(self, ctx, emit_score, use_mask,
                      coeff=1.0, clear=False, fold=False, interval=1.0,
                      inactive_lr_scale=1.0):
        for rec in ctx.records:
            state = rec._state
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(rec.main_param.data)
                state["exp_avg_sq"] = torch.zeros_like(rec.main_param.data)
            m = rec.mask_u8() if use_mask else None
            adams_grad_correct(
                param_buffer=rec.main_param.data.view(-1),
                grad_buffer=rec.grad.data.view(-1),
                exp_avg_buffer=state["exp_avg"].view(-1),
                exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
                lr=rec.lr,
                beta1=rec.beta1,
                beta2=rec.beta2,
                eps=rec.eps,
                weight_decay=rec.weight_decay,
                mask=m,
                emit_score=emit_score,
                # Degenerate mask-less bucket on a sparse step: plain full
                # commit (old behavior) — the fusion knobs require a mask,
                # and the engine scaled such a record explicitly; reset's
                # mask-less branch zeroes it afterwards.
                coeff=coeff if m is not None else 1.0,
                clear=clear and m is not None,
                fold=fold and m is not None,
                interval=interval,
                # A mask-less record on a split-lr full step (bootstrap /
                # degenerate bucket) was fully communicated: plain lr, the
                # documented semantics — never a split, never an error.
                inactive_lr_scale=inactive_lr_scale if m is not None else 1.0,
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
                adams_grad_correct(
                    param_buffer=rec.main_param.data.view(-1),
                    grad_buffer=rec.grad.data.view(-1),
                    exp_avg_buffer=state["exp_avg"].view(-1),
                    exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
                    lr=rec.lr, beta1=rec.beta1, beta2=rec.beta2,
                    eps=rec.eps, weight_decay=rec.weight_decay,
                    mask=None, emit_score=None,
                )
                continue
            bucket = ctx.support.buckets[bs.support_index]
            idx = bucket.idx
            # compact_search dual dispatch: the table is the bucket's lazy
            # per-cycle derived field — first sparse step of the cycle pays
            # the build, the rest reuse it (idx frozen per cycle). Gated
            # on the REAL execution path, not just the configured mode: on
            # the torch fallback (CPU tensors / triton-less) there is no
            # table to build — asking for one would raise before the
            # fallback could engage (compact_search's dead-config
            # exception: silently ignored off its live path).
            dir_table = (bucket.dir_table
                         if (self.compact_search == "table"
                             and _HAS_TRITON and idx.is_cuda) else None)
            adams_grad_correct_compact(
                param_buffer=rec.main_param.data.view(-1),
                grad_buffer=rec.grad.data.view(-1),
                exp_avg_buffer=state["exp_avg"].view(-1),
                exp_avg_sq_buffer=state["exp_avg_sq"].view(-1),
                idx=idx, values=values, rec_start=rec.start,
                lr=rec.lr, beta1=rec.beta1, beta2=rec.beta2,
                eps=rec.eps, weight_decay=rec.weight_decay,
                coeff=coeff, fold=fold, interval=interval,
                dir_table=dir_table,
            )

    @torch.no_grad()
    def full_step(self, ctx):
        """Full-sync step: one fused pass per record — m/v commit, the
        configured score materialized into the main grads (the reselection
        reads it from there), and the apply ``w -= lr * update``. With
        ``full_sync_inactive_lr_scale != 1.0`` the pass turns mask-aware:
        the OUTGOING window's mask splits the apply's lr, active coords at
        lr and the never-communicated tail at ``lr * scale``; the update
        vector stays mask-blind — only the applied lr splits. Both
        emission modes land here identically — no compact variant exists
        or is needed on full steps."""
        split = self.full_sync_inactive_lr_scale != 1.0
        self._records_pass(
            ctx, emit_score=self.emit_score, use_mask=split,
            inactive_lr_scale=(self.full_sync_inactive_lr_scale
                               if split else 1.0),
        )

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
        if ctx.compact_values:
            self._records_pass_compact(
                ctx, coeff=float(ctx.apply_coeff),
                fold=fold, interval=float(ctx.fold_window),
            )
            return
        self._records_pass(
            ctx, emit_score=None, use_mask=True,
            coeff=float(ctx.apply_coeff),
            clear=True, fold=fold, interval=float(ctx.fold_window),
        )

    def reset(self, ctx, was_full: bool):
        """Sparse steps need no grad-buffer work: writeback backends got
        the clear inside grad_correct's fused pass; compact backends got
        it from the pack kernel at grad_sync time. Zeroing here would
        destroy the accumulating residual either way. Full steps (and the
        degenerate mask-less bucket): zero all. ``extra_main_grads``
        zeroing stays."""
        for bs in ctx.bucket_states:
            if was_full or bs.active_mask_u8 is None:
                bs.grad_data.zero_()
            for grad in bs.extra_main_grads:
                grad.zero_()
