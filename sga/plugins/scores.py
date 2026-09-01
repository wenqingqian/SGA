# Copyright (c) 2026, SGA framework contributors.
"""Score registry: the single source of truth for SGA score names and math.

A score is a named importance signal over optimizer state. The registry maps
each name to its emission mode (the ``Score`` enum), its pure-torch formula
and documentation — no classes, no per-score files. The same names are spoken
by the online config (``sga.score.name``), the offline replay CLI
(``--score-names``) and the analysis tooling: one namespace, no translation
layers. Adding a score touches exactly TWO places: one entry here, and (for
online use) one branch in the optimizer kernel.

``Score(IntEnum)`` is the namespace: member names ARE the score names and the
int value is the compile-time branch the kernel switches on (Triton takes
constexpr ints only, so the kernel spells them as bare ints with name
comments; tests/test_kernels_gpu.py pins the mapping by name). The emission
mode is an INTRINSIC property of the score — it says which math gets
materialized, independent of any host optimizer. This does NOT reverse the
rule that online usability is declared per optimizer template
(``OptimizerDecl.emittable_scores``): a template declares the kernel-branch
subset it implements, and today's single template implements every non-OFFLINE
branch, so its emittable set is derived here. When a second template appears,
its emittable set is the intersection of its own branch subset with the
registry's online modes. ``Score.OFFLINE`` (-1) is the sentinel for
offline-only candidates: replay can evaluate them, configs reject them;
promotion = assigning a real mode + adding the kernel branch, nothing else.

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
are held simultaneously by callers' views lists. ``SCRATCH_SCORES`` is the
derived name set of scratch=True entries.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, Tuple

import torch


class Score(IntEnum):
    """Full-step emission modes; member names ARE the online score names.

    The int is the kernel's compile-time branch (EMIT_MODE constexpr).
    OFFLINE is the sentinel for offline-only candidates (no kernel branch);
    its -1 deliberately reuses the kernel's sparse-step EMIT_MODE sentinel —
    safe because the optimizer wrapper rejects OFFLINE host-side before the
    sparse path maps emit_mode=None to -1 at launch (a direct kernel call
    with -1 would be a sparse step, not an error).
    """

    update = 0
    abs_g = 1
    abs_m = 2
    coi = 3
    OFFLINE = -1


@dataclass(frozen=True)
class ScoreSpec:
    """One registry entry: emission mode + formula + doc + scratch flag
    (see module docstring). The name is NOT repeated here — it is the
    registry key, tied to the enum member name by the load-time check below.
    """

    score: Score
    # (w, m, v, g, beta1, beta2, eps, weight_decay, bias_correction1,
    #  bias_correction2, update_includes_wd, scratch) -> score tensor
    formula: Callable[..., torch.Tensor]
    doc: str
    scratch: bool = False


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
    "update": ScoreSpec(
        score=Score.update,
        formula=_formula_update,
        doc="the SIGNED AdamW weight update wd*w + U(g) (U(g) from the "
            "pre-commit moments and this step's gradient, upcoming-commit "
            "bias corrections); consumers rank by absolute value",
        scratch=True,
    ),
    "abs_g": ScoreSpec(
        score=Score.abs_g,
        formula=_formula_abs_g,
        doc="|g| of the committed, fully-synced gradient",
    ),
    "abs_m": ScoreSpec(
        score=Score.abs_m,
        formula=_formula_abs_m,
        doc="|m| of the pre-commit first moment",
    ),
    "coi": ScoreSpec(
        score=Score.coi,
        formula=_formula_coi,
        doc="counterfactual optimizer impact |U(g) - U(0)| (wd cancels)",
        scratch=True,
    ),
}

# The enum is the namespace: an online entry's key IS its mode's member name
# (offline candidates key under their candidate name with score=OFFLINE).
# Explicit raise, not assert: this guard is the only thing tying registry
# keys to enum member names and must survive ``python -O``.
for _name, _spec in SCORES.items():
    if not (_spec.score is Score.OFFLINE or _spec.score.name == _name):
        raise ValueError(
            f"score '{_name}' keyed differently from its mode "
            f"Score.{_spec.score.name} — the enum member name IS the score name"
        )
if SCORES:  # symmetric guard: no loop variables exist when the registry is empty
    del _name, _spec

# Names whose formulas route temporaries through the scratch pair (derived;
# see the module docstring). Callers may skip the pair allocation for others.
SCRATCH_SCORES = frozenset(n for n, s in SCORES.items() if s.scratch)


def emittable_score_names() -> frozenset:
    """Scores with an online emission mode (every non-OFFLINE entry)."""
    return frozenset(n for n, s in SCORES.items() if s.score is not Score.OFFLINE)


def get_score_spec(name: str) -> ScoreSpec:
    try:
        return SCORES[name]
    except KeyError:
        raise ValueError(
            f"Unknown SGA score '{name}'. Registered: {score_names()}"
        ) from None


def score_names():
    return sorted(SCORES)
