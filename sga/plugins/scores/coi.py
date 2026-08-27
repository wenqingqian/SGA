# Copyright (c) 2026, SGA framework contributors.
"""coi score: counterfactual optimizer impact |U(g) - U(0)|.

Historical xp scheme's score. U(g) is the AdamW update direction with this
step's gradient, U(0) the same with a zero gradient, both from the PRE-commit
moments — the weight-decay term cancels in the difference. Both quantities
are only reachable inside the EF kernel (the pre-commit m/v are overwritten
by the very update that computes them), so the kernel emits the score into
the fp32 main-grad copy on full-sync steps and this plugin is a pure
declaration over the materialized views.

scoring_input = 'synced': exists only on full-sync steps; selection (and
the global topk over it) happens at cycle boundaries only.
"""

from sga.plugins import ScoreDecl

DECL = ScoreDecl(name="coi", scoring_input="synced", has_threshold_path=False)


class CoiScore:
    """Score plugin over the EF-materialized coi views."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params

    def values(self, materialized_views):
        """Score values = the materialized coi tensors (non-negative)."""
        return list(materialized_views)

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "coi has no fused threshold path; use a ranked selection"
        )
