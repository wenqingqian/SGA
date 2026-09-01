# Copyright (c) 2026, SGA framework contributors.
"""Pure-torch reference path for the AdamW optimizer template.

Physically isolated from ``adamw.py`` (Triton kernels + dispatch only) so the
CPU unit tests and non-CUDA fallback never touch Triton import paths. Score
math is NOT restated here: the score-registry formulas
(``sga/plugins/scores.py`` — the single source of score math) are called
directly; only the m/v commit skeleton lives in this file. Numerics: the
same op sequence as the Triton kernel, tolerance-pinned by
tests/test_kernels_gpu.py (the kernel's fused-apply FMA makes bitwise
equality across the two impossible by design); the op sequence IS bitwise
identical to the pre-refactor inline torch reference — the load-bearing
claim for the CPU tests, which pin it elementwise (torch.equal) against
the shadow path in tests/test_shadow_unit.py.
"""

import torch

from sga.plugins.scores import Score, get_score_spec


def _registry_score(mode: Score, param, m, v, g, beta1, beta2, eps,
                    weight_decay, bias_correction1, bias_correction2):
    """Materialized score from the PRE-commit state, via the registry."""
    return get_score_spec(mode.name).formula(
        param, m, v, g, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        update_includes_wd=True,
    )


def grad_correct_torch(param, grad, m, v, lr, beta1, beta2, eps, weight_decay,
                       bias_correction1, bias_correction2,
                       mask, emit_mode) -> None:
    """Torch reference of the grad-correct step; same contract as the kernel.

    ``emit_mode=None`` is the sparse step: fused apply, grad buffer untouched.
    An int mode (a ``Score`` value) is the full step: the mode's score
    materializes into ``grad`` and the apply is fused in the same pass.
    Wrapper-fed: ``adamw_grad_correct`` validates ``emit_mode`` first — a
    DIRECT call with ``-1`` bypasses that guard and dies in the registry
    lookup with the misleading "Unknown SGA score 'OFFLINE'" message.
    """
    if mask is not None:
        g = torch.where(mask > 0, grad, torch.zeros_like(grad))
    else:
        g = grad
    # The update doubles as the 'update' score; both come from the registry
    # so the reference path can never drift from the online math.
    update = _registry_score(Score.update, param, m, v, g, beta1, beta2, eps,
                             weight_decay, bias_correction1, bias_correction2)
    if emit_mode is None:
        score = None
    elif emit_mode == int(Score.update):
        score = update
    else:
        score = _registry_score(Score(emit_mode), param, m, v, g, beta1, beta2,
                                eps, weight_decay, bias_correction1,
                                bias_correction2)
    # m_new/v_new are computed a SECOND time here (the registry update chain
    # above already built them internally): deliberate — single source of
    # score math beats micro-perf on this reference path.
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * g * g
    m.copy_(m_new)
    v.copy_(v_new)
    if emit_mode is not None:
        grad.copy_(score)
    param.sub_(lr * update)


def selective_reset_torch(grad_buffer: torch.Tensor, mask_u8: torch.Tensor,
                          interval: float, fold: bool) -> None:
    """Torch fallback of the fused selective reset.

    Element-wise on purpose (boolean indexing materializes full-bucket
    int64 indices, ~GB-scale). The unconditional divide is exact: actives
    were just set to 0.0, and 0.0 / interval == 0.0.
    """
    active = mask_u8 > 0
    grad_buffer.masked_fill_(active, 0.0)
    if fold:
        grad_buffer.div_(interval)
