# Copyright (c) 2026, SGA framework contributors.
"""Score registry: the single source of truth for SGA score names and math.

A score is a named importance signal over optimizer state. The registry maps
each name to its pure-torch formula and documentation — no classes, no
per-score files. The same names are spoken by the online config
(``--sga-score``), the offline replay CLI (``--score-names``) and the
analysis tooling: one namespace, no translation layers. The registry is
registered-data style (``register_score`` + ``register_default_scores``,
idempotent), the same idiom as the class-plugin registries in
``sga/plugins/__init__.py``.

Adding an ONLINE score touches exactly TWO places: one entry here, and one
branch in the optimizer kernel (``_TRITON_SCORE_MAP`` + the kernel's
``EMIT_MODE`` if-chain live together in ``sga/plugins/optimizer/adamw.py``;
a registered score without a map entry fails loud at optimizer
construction, never silently mis-materializes). A replay-only score needs
only the entry here — but then no config may select it usefully; keep
research candidates out of the registry until they have their branch.

Each formula computes the score from one coordinate's pre-commit state:

    (w, m, v, g) + AdamW hyperparameters + the bias corrections of the
    UPCOMING commit

and returns the MATERIALIZED form — exactly what the online optimizer's
full-step kernel writes into the fp32 main-grad copy: ``update`` is the
SIGNED weight update, the others are magnitudes. Consumers that need a
non-negative ranking input (the histogram/kthvalue selects) take the
absolute value at the call site; abs(abs(x)) is identity, so one contract
serves both.

``scratch=True`` marks formulas that route their dying temporaries through a
caller-provided scratch pair via ``out=`` — op-for-op the same elementwise
sequence, so results stay bitwise identical. The pair removes the temp
chain's persistent spike (the per-param scratch writes would otherwise each
leave a simultaneously-live allocation behind across shadow's cycle-boundary
scoring of every param). Some temporaries still allocate fresh per call —
the sqrt result, coi's ``inner``, the ``weight_decay * w`` term — but they
die within the call, so the caching allocator bounds the high-water.
Formulas that return before touching any temp ignore the pair (scratch=False:
abs_g/abs_m). Returned tensors never alias ``scratch``: per-param outputs
are held simultaneously by callers' views lists.
"""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ScoreSpec:
    """One registry entry: formula + doc + scratch flag. The name is NOT
    repeated here — it is the registration key (the config-facing score
    name and the replay CLI name)."""

    # (w, m, v, g, beta1, beta2, eps, weight_decay, bias_correction1,
    #  bias_correction2, update_includes_wd, scratch) -> score tensor
    formula: Callable[..., torch.Tensor]
    doc: str
    scratch: bool = False


_SCORES: dict = {}


def register_score(name: str, spec: ScoreSpec) -> None:
    """Register one score; duplicate names fail loud at registration."""
    if name in _SCORES:
        raise ValueError(f"duplicate score registration '{name}'")
    _SCORES[name] = spec


def register_default_scores() -> None:
    """Idempotent registration of the built-in scores (per-name, so a
    test that pre-registers a custom score does not block the defaults)."""
    _SCORES.setdefault("update", ScoreSpec(
        formula=_formula_update,
        doc="the SIGNED AdamW weight update wd*w + U(g) (U(g) from the "
            "pre-commit moments and this step's gradient, upcoming-commit "
            "bias corrections); consumers rank by absolute value",
        scratch=True,
    ))
    _SCORES.setdefault("abs_g", ScoreSpec(
        formula=_formula_abs_g,
        doc="|g| of the committed, fully-synced gradient",
    ))
    _SCORES.setdefault("abs_m", ScoreSpec(
        formula=_formula_abs_m,
        doc="|m| of the pre-commit first moment",
    ))
    _SCORES.setdefault("coi", ScoreSpec(
        formula=_formula_coi,
        doc="counterfactual optimizer impact |U(g) - U(0)| (wd cancels)",
        scratch=True,
    ))


def get_score_spec(name: str) -> ScoreSpec:
    register_default_scores()
    try:
        return _SCORES[name]
    except KeyError:
        raise ValueError(
            f"Unknown SGA score '{name}'. Registered: {score_names()}"
        ) from None


def score_names():
    register_default_scores()
    return sorted(_SCORES)


def _resolve_scratch(w: torch.Tensor, scratch):
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


# Populate the built-ins at import: a caller could otherwise
# pre-register a custom formula under a built-in name before the lazy fill
# ran, silently divorcing the registry from the optimizer's fixed
# _TRITON_SCORE_MAP branch. After this call, register_score on a built-in
# name fails loud through the duplicate check.
register_default_scores()
