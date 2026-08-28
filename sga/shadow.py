# Copyright (c) 2026, SGA framework contributors.
"""Shadow-mode read-only score materialization.

Shadow mode replicates the stock optimizer step literally (dense numerics)
and only READS state at cycle boundaries to produce a shadow support. The
live path materializes scores inside the EF's fused commit kernel — which
mutates m/v, so it cannot serve shadow. This module computes the same
score formulas from the pre-commit state (post-clip gradient, pre-commit
w/m/v, host-side bias corrections) WITHOUT writing anything back: the
formula bodies mirror ``_radius_grad_correct_torch`` line-for-line, and
tests/test_shadow_unit.py pins the two implementations elementwise.

The shadow support never touches communication: grad sync stays stock and
the EF never runs. The reselection itself runs the same selection
collectives as live (circular_topk's histogram SUM allreduce, exact
kthvalue's mask MAX), plus the engine's scalar consistency checks over
shadow state (count/checksum/norm-spread allreduces, which do not involve
training tensors).
"""

import torch


def readonly_score(rec, emit_name: str) -> torch.Tensor:
    """Score tensor for one ParamRecord from its PRE-COMMIT state.

    ``emit_name`` is the EF-materialized score name the selection consumes
    (``update_ratio`` for the default signed-update score, or one of
    coi/abs_g_synced/abs_m/snr_m/abs_m_post). The gradient input is the
    post-unscale post-clip main grad (the stock step's view at this point);
    m/v are the pre-commit moments; bias corrections use ``rec.step``, the
    upcoming commit number. Returns a fresh tensor; nothing is mutated.
    """
    w = rec.main_param.data
    g = rec.grad.data
    # torch-AdamW state is lazy: before the first commit the moments are
    # zeros (the pre-first-commit semantics every other consumer assumes)
    st = rec._state or {}
    if "exp_avg" in st:
        if "exp_avg_sq" not in st:
            raise ValueError(
                "shadow: partial optimizer state (exp_avg without "
                "exp_avg_sq) — the score formulas need both moments"
            )
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
    else:
        m = torch.zeros_like(w)
        v = torch.zeros_like(w)
    beta1, beta2, eps = rec.beta1, rec.beta2, rec.eps
    step = rec.step
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    bias_correction1_pre = 1.0 - beta1 ** (step - 1) if step > 1 else 0.0
    bias_correction2_pre = 1.0 - beta2 ** (step - 1) if step > 1 else 0.0

    if emit_name == "abs_g_synced":
        return g.abs()
    if emit_name == "abs_m":
        return m.abs()
    if emit_name == "snr_m":
        if bias_correction1_pre > 0:
            return (m / bias_correction1_pre).abs() / (
                (v / bias_correction2_pre).sqrt() + eps)
        return torch.zeros_like(m)
    m_new = beta1 * m + (1.0 - beta1) * g
    if emit_name == "abs_m_post":
        return m_new.abs()
    v_new = beta2 * v + (1.0 - beta2) * g * g
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    if emit_name == "coi":
        u_0 = (beta1 * m / bias_correction1) / (
            (beta2 * v / bias_correction2).sqrt() + eps
        )
        return (m_hat / (v_hat.sqrt() + eps) - u_0).abs()
    if emit_name == "update_ratio":
        # default score: the SIGNED materialized update (the live path
        # writes it into the grad buffer and the selection consumes it as-is)
        return rec.weight_decay * w + m_hat / (v_hat.sqrt() + eps)
    raise ValueError(
        f"shadow: no read-only formula for emit name {emit_name!r}"
    )
