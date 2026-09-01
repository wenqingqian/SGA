# Copyright (c) 2026, SGA framework contributors.
"""Shadow-mode read-only score materialization.

Shadow mode replicates the stock optimizer step literally (dense numerics)
and only READS state at cycle boundaries to produce a shadow support. The
live path materializes scores inside the optimizer template's fused commit
kernel — which mutates m/v, so it cannot serve shadow. This module evaluates
the SCORE REGISTRY formula (sga/plugins/scores.py — the single source of
score math) on the pre-commit state (post-clip gradient, pre-commit w/m/v,
host-side bias corrections) WITHOUT writing anything back: no training
tensor (w/g/m/v) is mutated (with ``scratch``, the caller-provided buffers
are written by design). tests/test_shadow_unit.py pins the formula against
the optimizer kernel's torch reference elementwise.

The shadow support never touches communication: grad sync stays stock and
the optimizer template never runs. The reselection itself runs the same
selection collectives as live (circular_topk's histogram SUM allreduce,
exact kthvalue's mask MAX), plus the engine's scalar consistency checks over
shadow state (count/checksum/norm-spread allreduces, which do not involve
training tensors).
"""

import torch

from sga.plugins.scores import get_score_spec, score_names


def readonly_score(rec, emit_name: str, scratch=None) -> torch.Tensor:
    """Score tensor for one ParamRecord from its PRE-COMMIT state.

    ``emit_name`` is the registry score name the selection consumes. The
    gradient input is the post-unscale post-clip main grad (the stock step's
    view at this point); m/v are the pre-commit moments; bias corrections use
    ``rec.step``, the upcoming commit number. Returns a fresh tensor (the
    registry formula's materialized form — 'update' is signed).

    ``scratch`` (optional) is a pair of >=-numel tensors of the params'
    dtype (fp32 in practice) on w's device that the formula chain's DYING
    temporaries are routed into via out= — see the score registry's module
    docstring; it exists so the boundary's allocator high-water stays flat.
    Callers may skip the allocation for scores whose spec does not declare
    ``scratch``.
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
    if "exp_avg" in st:
        if "exp_avg_sq" not in st:
            raise ValueError(
                "shadow: partial optimizer state (exp_avg without "
                "exp_avg_sq) — the score formulas need both moments"
            )
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
    else:
        # fresh zeros, NOT routed into scratch: the formulas overwrite the
        # scratch pair before the v-terms are consumed, which would silently
        # replace the zero moments with gradient terms
        m = torch.zeros_like(w)
        v = torch.zeros_like(w)
    if scratch is not None:
        n = w.numel()
        for i, buf in enumerate(scratch):
            assert (buf.numel() >= n and buf.dtype == w.dtype
                    and buf.device == w.device), (
                f"scratch[{i}] contract violated: need >= {n} elements of the "
                f"params' dtype ({w.dtype}) on w's device ({w.device}), got "
                f"{buf.numel()} elements of {buf.dtype} on {buf.device}"
            )
    try:
        spec = get_score_spec(emit_name)
    except ValueError:
        raise ValueError(
            f"shadow: no formula for score name {emit_name!r} "
            f"(registered: {score_names()})"
        ) from None
    step = rec.step
    return spec.formula(
        w, m, v, g,
        rec.beta1, rec.beta2, rec.eps, rec.weight_decay,
        1.0 - rec.beta1 ** step,
        1.0 - rec.beta2 ** step,
        update_includes_wd=True,
        scratch=scratch,
    )
