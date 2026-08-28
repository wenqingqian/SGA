# Copyright (c) 2026, SGA framework contributors.
"""abs_m_post score: |m_new| = |beta1*m + (1-beta1)*g| of the momentum AFTER
a hypothetical commit of this step's synced gradient.

From the legacy offline oracle's extended score set (evidence: no
gain — kept as an algorithm-exploration tool). Online it reads the pre-commit
m and this step's gradient inside the EF kernel pass (m_new is computed there
anyway for the m/v update), so the kernel emits it into the fp32 main-grad
copy on full-sync steps and this plugin is a pure declaration over the
materialized views.

scoring_input = 'synced': exists only on full-sync steps; selection (and
the global topk over it) happens at cycle boundaries only.
"""

from sga.plugins import ScoreDecl

DECL = ScoreDecl(name="abs_m_post", scoring_input="synced", has_threshold_path=False)


class AbsMPostScore:
    """Score plugin over the EF-materialized |m_new| views."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params

    def values(self, materialized_views):
        """Score values = the materialized |m_new| tensors (non-negative)."""
        return list(materialized_views)

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "abs_m_post has no fused threshold path; use a ranked selection"
        )
