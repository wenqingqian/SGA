# Copyright (c) 2026, SGA framework contributors.
"""abs_g_synced score: |g| of the committed, fully-synced gradient.

The historical xp scheme's ``abs_g`` — renamed here because the framework
already has an ``abs_g`` with different semantics (the origin scheme's
local-or-merged |accumulated grad| with a fused threshold path). This
variant is the magnitude of the gradient being committed on a full-sync
step, materialized into the fp32 main-grad copy by the EF kernel.

Matrix mapping (notes/online-matrix-evidence.md): old xp ``abs_g`` runs
map to this plugin's name.

scoring_input = 'synced': exists only on full-sync steps; selection (and
the global topk over it) happens at cycle boundaries only.
"""

from sga.plugins import ScoreDecl

DECL = ScoreDecl(name="abs_g_synced", scoring_input="synced",
                 has_threshold_path=False)


class AbsGSyncedScore:
    """Score plugin over the EF-materialized synced-|g| views."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params

    def values(self, materialized_views):
        """Score values = the materialized |g| tensors (non-negative)."""
        return list(materialized_views)

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "abs_g_synced has no fused threshold path; use a ranked selection"
        )
