# Copyright (c) 2026, SGA framework contributors.
"""abs_m_sticky score: |m| with a multiplicative loyalty bonus for the
coordinates in LAST cycle's support:

    score = |m| * (1 + sticky_lambda * 1[coord in current pre-switch support])

From the legacy offline oracle's extended score set (evidence:
wins at long prediction horizons — the reason this family is kept).

This is a POST-PROCESSING score: the EF kernel materializes plain abs_m
(``emit_name = "abs_m"`` — the kernel stays free of support logic), and this
plugin multiplies the bonus on at reselection time. The engine reads the
support BEFORE switching to the new one, so "current support" is exactly last
cycle's. At bootstrap (no support yet) the bonus factor is 1 and the score
degenerates to abs_m, matching the oracle's first-selection behavior.
Forced full steps never reselect, so the bonus only ever applies at true
cycle boundaries.

scoring_input = 'synced': exists only on full-sync steps; selection (and
the global topk over it) happens at cycle boundaries only.
"""

import math

from sga.plugins import ScoreDecl

DECL = ScoreDecl(
    name="abs_m_sticky",
    scoring_input="synced",
    has_threshold_path=False,
    needs_support=True,
    emit_name="abs_m",
)


class AbsMStickyScore:
    """abs_m views x loyalty bonus over the current (pre-switch) support."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params
        self.sticky_lambda = float(params.get("sticky_lambda", 0.5))
        if not math.isfinite(self.sticky_lambda) or self.sticky_lambda < 0.0:
            raise ValueError(
                "abs_m_sticky: sticky_lambda must be finite and >= 0, got "
                f"{params.get('sticky_lambda')!r}"
            )

    def values(self, materialized_views, support_slices=None):
        """|m| views with the loyalty bonus applied elementwise.

        ``support_slices`` are the per-param uint8 mask slices of the
        pre-switch support in the same view order; None at bootstrap, where
        the score degenerates to plain abs_m. Fresh tensors are returned —
        the materialized views alias live main grads and are never mutated.
        """
        if support_slices is None:
            return list(materialized_views)
        if len(materialized_views) != len(support_slices):
            raise ValueError(
                f"abs_m_sticky: got {len(materialized_views)} views but "
                f"{len(support_slices)} support slices"
            )
        lam = self.sticky_lambda
        out = []
        for i, (v, s) in enumerate(zip(materialized_views, support_slices)):
            if s.numel() != v.numel():
                raise ValueError(
                    f"abs_m_sticky: support slice {i} has {s.numel()} "
                    f"elements but its view has {v.numel()}"
                )
            out.append(v * (1.0 + lam * s.to(v.dtype)))
        return out

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "abs_m_sticky has no fused threshold path; use a ranked selection"
        )
