# Copyright (c) 2026, SGA framework contributors.
"""update_ratio score: importance = |AdamW weight update|.

Historical radius scheme's score. The update
``wd * w + (m/b1) / (sqrt(v/b2) + eps)`` is NOT independently computable —
it is materialized into the fp32 main-grad copy by the paired EF kernel on
full-sync steps (the EF->score handoff frozen by the the design grilling:
``emits_score_view``). The validator therefore requires this score to be
paired with an emitting error_feedback plugin.

scoring_input = 'synced': the materialized update exists only on full-sync
steps, so selection (and the global topk over it) happens at cycle
boundaries only.
"""

from sga.plugins import ScoreDecl

DECL = ScoreDecl(name="update_ratio", scoring_input="synced", has_threshold_path=False)


class UpdateRatioScore:
    """Score plugin over the EF-materialized update views."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params

    def values(self, materialized_views):
        """Score values = the materialized update tensors (signed; ranked
        selections take magnitudes themselves)."""
        return list(materialized_views)

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "update_ratio has no fused threshold path; use a ranked selection"
        )
