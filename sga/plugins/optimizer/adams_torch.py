# Copyright (c) 2026, SGA framework contributors.
"""Pure-torch reference path for the AdamS optimizer template.

Physically isolated from ``adams.py`` (Triton kernels + dispatch only) so
the CPU unit tests and non-CUDA fallback never touch Triton import paths —
the same isolation as ``adamw_torch.py``. Score math is NOT restated here:
the score-registry formulas (``sga/plugins/scores.py`` — the single source
of score math) are called directly by NAME with ``optimizer="adams"``; only
the m/v commit skeleton lives in this file. Numerics: the same op sequence
as the Triton kernel, tolerance-pinned by tests/test_adams_gpu.py (the
kernel's fused-apply FMA makes bitwise equality across the two impossible
by design); the op sequence itself is pinned elementwise (torch.equal)
against the independent fp64 oracles by tests/test_adams_unit.py.
"""

import torch

from sga.plugins.scores import get_score_spec


def _adams_score(name, param, m, v, g, beta1, beta2, eps, weight_decay):
    """Materialized AdamS score from the PRE-commit state, via the
    registry (the single source of score math; the bias-correction
    channels read 1.0 for AdamS by construction)."""
    return get_score_spec(name).formula(
        param, m, v, g, beta1, beta2, eps, weight_decay,
        1.0, 1.0,
        update_includes_wd=True,
        optimizer="adams",
    )


def adams_grad_correct_torch(param, grad, m, v, lr, beta1, beta2, eps,
                             weight_decay, mask, emit_score,
                             coeff=1.0, clear=False, fold=False,
                             interval=1.0, inactive_lr_scale=1.0) -> None:
    """Torch reference of the AdamS grad-correct step; same contract as
    the kernel.

    ``emit_score=None`` is the sparse step: fused apply. A score NAME is
    the full step: the score materializes into ``grad`` and the apply is
    fused in the same pass. Sparse-fusion knobs and the full-step
    dual-lr mirror the kernel (and the AdamW torch reference) knob for
    knob; the only differences from ``adamw_torch.grad_correct_torch``
    are the moment tail — ``v_new = beta2*(m*m) + (1-beta2)*g*g`` from
    the pre-commit m — and the absence of bias corrections (there is no
    step input to derive them from). Scores come from the registry with
    ``optimizer="adams"`` so the reference can never drift from the
    formula source.
    """
    if inactive_lr_scale != 1.0:
        g = grad
        lr_apply = torch.where(mask > 0, lr, lr * inactive_lr_scale)
    elif mask is not None:
        g = torch.where(mask > 0, grad, torch.zeros_like(grad))
        lr_apply = lr
    else:
        g = grad
        lr_apply = lr
    if coeff != 1.0:
        g = torch.where(mask > 0, g * coeff, g)
    update = _adams_score("update", param, m, v, g, beta1, beta2, eps,
                          weight_decay)
    if emit_score is None:
        score = None
    elif emit_score == "update":
        score = update
    else:
        score = _adams_score(emit_score, param, m, v, g, beta1, beta2, eps,
                             weight_decay)
    # m_new/v_new are computed a SECOND time here (the registry update
    # chain above already built them internally): deliberate — single
    # source of score math beats micro-perf on this reference path.
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * (m * m) + (1.0 - beta2) * g * g
    m.copy_(m_new)
    v.copy_(v_new)
    if emit_score is not None:
        grad.copy_(score)
    else:
        if clear and mask is not None:
            grad.masked_fill_(mask > 0, 0.0)
        if fold:
            grad.div_(interval)
    param.sub_(lr_apply * update)
