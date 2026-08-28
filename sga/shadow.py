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

# Emit names readonly_score has a formula for (the EF-emittable scores the
# shadow path can recompute from pre-commit state). Exported for build-time
# gating: a shadow run over a synced score outside this set must fail at
# engine construction, not at the first boundary.
SHADOW_SUPPORTED_EMITS = frozenset(
    (
        "update_ratio",
        "coi",
        "abs_g_synced",
        "abs_m",
        "snr_m",
        "abs_m_post",
    )
)

# The emits whose formulas route temps into the ``scratch`` pair;
# abs_g_synced/abs_m return before touching it, so callers may skip the
# allocation for them (no dead pair at the boundary).
EMITS_USING_SCRATCH = frozenset(("update_ratio", "coi", "snr_m", "abs_m_post"))


def readonly_score(rec, emit_name: str, scratch=None) -> torch.Tensor:
    """Score tensor for one ParamRecord from its PRE-COMMIT state.

    ``emit_name`` is the EF-materialized score name the selection consumes
    (must be in SHADOW_SUPPORTED_EMITS). The gradient input is the
    post-unscale post-clip main grad (the stock step's view at this point);
    m/v are the pre-commit moments; bias corrections use ``rec.step``, the
    upcoming commit number. Returns a fresh tensor; no training state
    (w/g/m/v) is mutated (with ``scratch``, the caller-provided buffers are
    written by design).

    ``scratch`` (optional) is a pair of >=-numel tensors of the params'
    dtype (fp32 in practice) on w's device
    that the formula chain's DYING temporaries are routed into via out= —
    op-for-op the same elementwise sequence, so results stay bitwise
    identical, while the per-call chained temps (several param-sized
    allocations on the largest param) collapse into two reusable buffers;
    per-emit extras (coi's den/inner, the returned score) stay fresh
    per-call. Returned tensors never alias ``scratch``: per-param outputs
    are held simultaneously by the caller's views list and must stay
    distinct.
    """
    w = rec.main_param.data
    if rec.grad is None:
        raise ValueError(
            f"shadow: record for shape {tuple(w.shape)} has no gradient at "
            "the boundary — records were refreshed against a different "
            "optimizer view or the buffer was cleared early"
        )
    g = rec.grad.data
    # torch-AdamW state is lazy: before the first commit the moments are
    # zeros (the pre-first-commit semantics every other consumer assumes)
    st = rec._state or {}
    n = w.numel()
    bufs = scratch if scratch is not None else (
        torch.empty(n, dtype=w.dtype, device=w.device),
        torch.empty(n, dtype=w.dtype, device=w.device),
    )
    for i, buf in enumerate(bufs):
        assert buf.numel() >= n and buf.dtype == w.dtype, (
            f"scratch[{i}] contract violated: need >= {n} elements of the "
            f"params' dtype ({w.dtype}), got {buf.numel()} elements of "
            f"{buf.dtype}"
        )
    sa = bufs[0][:n].view_as(w)
    sb = bufs[1][:n].view_as(w)
    if "exp_avg" in st:
        if "exp_avg_sq" not in st:
            raise ValueError(
                "shadow: partial optimizer state (exp_avg without "
                "exp_avg_sq) — the score formulas need both moments"
            )
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
    else:
        # fresh zeros, NOT routed into sa/sb: the chain below overwrites the
        # scratch pair before the v-terms are consumed, which would silently
        # replace the zero moments with gradient terms
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
            torch.div(m, bias_correction1_pre, out=sa)
            torch.abs(sa, out=sa)
            torch.div(v, bias_correction2_pre, out=sb)
            torch.sqrt(sb, out=sb)
            sb.add_(eps)
            return torch.div(sa, sb)
        return torch.zeros_like(m)
    # m_new = beta1 * m + (1.0 - beta1) * g
    torch.mul(m, beta1, out=sa)
    torch.mul(g, 1.0 - beta1, out=sb)
    m_new = torch.add(sa, sb, out=sa)
    if emit_name == "abs_m_post":
        return m_new.abs()
    # v_new = beta2 * v + (1.0 - beta2) * g * g   (result doubles as a temp)
    result = torch.empty_like(w)
    torch.mul(v, beta2, out=sb)
    torch.mul(g, 1.0 - beta2, out=result)
    torch.mul(result, g, out=result)
    torch.add(sb, result, out=sb)
    # m_hat = m_new / bc1; v_hat = v_new / bc2 (in place: m_new/v_new die here)
    torch.div(sa, bias_correction1, out=sa)
    torch.div(sb, bias_correction2, out=sb)
    if emit_name == "coi":
        torch.mul(m, beta1, out=result)
        torch.div(result, bias_correction1, out=result)
        den = torch.mul(v, beta2)
        torch.div(den, bias_correction2, out=den)
        torch.sqrt(den, out=den)
        den.add_(eps)
        torch.div(result, den, out=result)
        inner = torch.add(torch.sqrt(sb), eps)
        torch.div(sa, inner, out=inner)
        torch.sub(inner, result, out=result)
        return result.abs()
    if emit_name == "update_ratio":
        # default score: the SIGNED materialized update (the live path
        # writes it into the grad buffer and the selection consumes it as-is)
        torch.sqrt(sb, out=sb)
        sb.add_(eps)
        torch.div(sa, sb, out=sa)
        return rec.weight_decay * w + sa
    raise ValueError(
        f"shadow: no read-only formula for emit name {emit_name!r} "
        f"(supported: {sorted(SHADOW_SUPPORTED_EMITS)})"
    )
