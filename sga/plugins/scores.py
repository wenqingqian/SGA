# Copyright (c) 2026, SGA framework contributors.
"""Score registry: the single source of truth for SGA score names and math.

A score is a named importance signal over optimizer state. The registry maps
each name to its pure-torch formula plus documentation — no classes, no
per-score files. The same names are spoken by the online config (``sga.score.name``),
the offline replay CLI (``--score-names``) and the analysis tooling: one
namespace, no translation layers.

Each formula computes the score from one coordinate's pre-commit state:

    (w, m, v, g) + AdamW hyperparameters + the bias corrections of the
    UPCOMING commit

and returns the MATERIALIZED form — exactly what the online optimizer's
full-step kernel writes into the fp32 main-grad copy: ``update`` is the
SIGNED weight update, the others are magnitudes. Consumers that need a
non-negative ranking input (the histogram/kthvalue selects) take the
absolute value at the call site; abs(abs(x)) is identity, so one contract
serves both.

Capability levels: online usability is NOT declared here — it belongs to the
optimizer template (``OptimizerDecl.emittable_scores``: the kernel branches
it can materialize). A registry entry whose name is in no optimizer's
emittable set is offline-only: replay can evaluate it, configs reject it.
Promoting an offline candidate online = adding the kernel branch (plus
listing the name), nothing else.

``scratch`` (optional) is a pair of >=-numel tensors of w's dtype on w's
device that a formula's dying temporaries are routed into via ``out=`` —
op-for-op the same elementwise sequence, so results stay bitwise identical.
The pair removes the temp chain's persistent spike (the per-param scratch
writes would otherwise each leave a simultaneously-live allocation behind
across shadow's cycle-boundary scoring of every param). Some temporaries
still allocate fresh per call — the sqrt result, coi's ``inner``, the
``weight_decay * w`` term — but they die within the call, so the caching
allocator bounds the high-water. Formulas that return before touching any
temp ignore the pair. Returned tensors never alias ``scratch``: per-param
outputs are held simultaneously by callers' views lists.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


@dataclass(frozen=True)
class ScoreSpec:
    """One registry entry: name + formula + doc (see module docstring)."""

    name: str
    # (w, m, v, g, beta1, beta2, eps, weight_decay, bias_correction1,
    #  bias_correction2, update_includes_wd, scratch) -> score tensor
    formula: Callable[..., torch.Tensor]
    doc: str


def _resolve_scratch(w: torch.Tensor, scratch: Optional[Tuple]):
    # Narrow to [:w.numel()].view_as(w): the contract permits numel >=
    # w.numel() (callers share ONE flat max-numel pair across params), and an
    # out= op on a size-mismatched tensor would RESIZE the shared buffer in
    # place (deprecated; shrinks it under the next, larger record). Main
    # params are contiguous, so the view is safe.
    sa = (scratch[0][:w.numel()].view_as(w) if scratch is not None
          else torch.empty_like(w))
    sb = (scratch[1][:w.numel()].view_as(w) if scratch is not None
          else torch.empty_like(w))
    return sa, sb


def _formula_abs_g(w, m, v, g, beta1, beta2, eps, weight_decay,
                   bias_correction1, bias_correction2,
                   update_includes_wd=True, scratch=None):
    """|g| of the gradient being committed (the fully-synced step's)."""
    return g.abs()


def _formula_abs_m(w, m, v, g, beta1, beta2, eps, weight_decay,
                   bias_correction1, bias_correction2,
                   update_includes_wd=True, scratch=None):
    """|m| of the pre-commit first moment."""
    return m.abs()


def _u_g(w, m, v, g, beta1, beta2, eps, bias_correction1, bias_correction2,
         scratch):
    """U(g) = m_hat / (sqrt(v_hat) + eps) into ``sa``; returns (sa, sb, result).

    m_hat/v_hat use the moments AFTER folding in this step's gradient, with
    the upcoming commit's bias corrections — the update direction the commit
    is about to apply. ``result`` is one fresh per-call tensor the caller's
    tail may reuse; sa/sb are the (possibly caller-provided) scratch pair.
    """
    sa, sb = scratch
    result = torch.empty_like(w)
    # m_new = beta1 * m + (1.0 - beta1) * g
    torch.mul(m, beta1, out=sa)
    torch.mul(g, 1.0 - beta1, out=sb)
    torch.add(sa, sb, out=sa)
    # v_new = beta2 * v + (1.0 - beta2) * g * g
    torch.mul(v, beta2, out=sb)
    torch.mul(g, 1.0 - beta2, out=result)
    torch.mul(result, g, out=result)
    torch.add(sb, result, out=sb)
    # m_hat = m_new / bc1; v_hat = v_new / bc2 (in place: m_new/v_new die here)
    torch.div(sa, bias_correction1, out=sa)
    torch.div(sb, bias_correction2, out=sb)
    return sa, sb, result


def _formula_update(w, m, v, g, beta1, beta2, eps, weight_decay,
                    bias_correction1, bias_correction2,
                    update_includes_wd=True, scratch=None):
    """The materialized AdamW weight update.

    update_includes_wd=True (the online semantics): the SIGNED
    ``weight_decay * w + U(g)`` the full-step commit materializes — the
    selection consumes it as-is and ranks by magnitude. False: the wd-free
    U(g) continuity reference (the offline zerofill-mode convention).
    """
    sa, sb = _resolve_scratch(w, scratch)
    sa, sb, _result = _u_g(w, m, v, g, beta1, beta2, eps,
                           bias_correction1, bias_correction2, (sa, sb))
    torch.sqrt(sb, out=sb)
    sb.add_(eps)
    torch.div(sa, sb, out=sa)
    if update_includes_wd:
        return weight_decay * w + sa
    # wd-free return must NOT alias the (possibly caller-provided) scratch
    # pair — outputs are held simultaneously by callers' views lists. _result
    # is _u_g's fresh per-call tensor, dead at this point: reuse it.
    _result.copy_(sa)
    return _result


def _formula_coi(w, m, v, g, beta1, beta2, eps, weight_decay,
                 bias_correction1, bias_correction2,
                 update_includes_wd=True, scratch=None):
    """|U(g) - U(0)|: counterfactual optimizer impact.

    U(0) is the update direction the SAME commit would apply with a zero
    gradient, from the same pre-commit moments — the weight-decay term is
    identical in both and cancels in the difference.
    """
    sa, sb = _resolve_scratch(w, scratch)
    sa, sb, result = _u_g(w, m, v, g, beta1, beta2, eps,
                          bias_correction1, bias_correction2, (sa, sb))
    # u_0 = (beta1 * m / bc1) / (sqrt(beta2 * v / bc2) + eps) -> result
    torch.mul(m, beta1, out=result)
    torch.div(result, bias_correction1, out=result)
    den = torch.mul(v, beta2)
    torch.div(den, bias_correction2, out=den)
    torch.sqrt(den, out=den)
    den.add_(eps)
    torch.div(result, den, out=result)
    # u_g = sa / (sqrt(sb) + eps) -> inner; (u_g - u_0).abs() -> result
    inner = torch.add(torch.sqrt(sb), eps)
    torch.div(sa, inner, out=inner)
    torch.sub(inner, result, out=result)
    return result.abs()


SCORES = {
    spec.name: spec
    for spec in (
        ScoreSpec(
            name="abs_g",
            formula=_formula_abs_g,
            doc="|g| of the committed, fully-synced gradient",
        ),
        ScoreSpec(
            name="abs_m",
            formula=_formula_abs_m,
            doc="|m| of the pre-commit first moment",
        ),
        ScoreSpec(
            name="update",
            formula=_formula_update,
            doc="the SIGNED AdamW weight update wd*w + U(g) (U(g) from the "
                "pre-commit moments and this step's gradient, upcoming-commit "
                "bias corrections); consumers rank by absolute value",
        ),
        ScoreSpec(
            name="coi",
            formula=_formula_coi,
            doc="counterfactual optimizer impact |U(g) - U(0)| (wd cancels)",
        ),
    )
}

# Formulas that route temporaries through the scratch pair (see module
# docstring); abs_g/abs_m return before touching it, so callers may skip the
# allocation for them.
SCRATCH_SCORES = frozenset(("update", "coi"))


def get_score_spec(name: str) -> ScoreSpec:
    try:
        return SCORES[name]
    except KeyError:
        raise ValueError(
            f"Unknown SGA score '{name}'. Registered: {score_names()}"
        ) from None


def score_names():
    return sorted(SCORES)
