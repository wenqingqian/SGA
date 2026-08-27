# Copyright (c) 2026, SGA framework contributors.
"""abs_m score: importance = |m| of the pre-commit momentum.

Historical xp scheme's score, ranked first in the 14-run online matrix
(abs_m > abs_g > update ~= coi). The pre-commit |m| is only reachable
inside the EF kernel (the same pass that overwrites m computes the score),
so the kernel emits it into the fp32 main-grad copy on full-sync steps and
this plugin is a pure declaration over the materialized views.

scoring_input = 'synced': exists only on full-sync steps; selection (and
the global topk over it) happens at cycle boundaries only.
"""

from sga.plugins import ScoreDecl

DECL = ScoreDecl(name="abs_m", scoring_input="synced", has_threshold_path=False)


class AbsMScore:
    """Score plugin over the EF-materialized |m| views."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params

    def values(self, materialized_views):
        """Score values = the materialized |m| tensors (non-negative)."""
        return list(materialized_views)

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "abs_m has no fused threshold path; use a ranked selection"
        )
