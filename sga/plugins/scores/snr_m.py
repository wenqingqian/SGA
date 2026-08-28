# Copyright (c) 2026, SGA framework contributors.
"""snr_m score: pre-commit signal-to-noise |m_hat| / (sqrt(v_hat) + eps).

From the legacy offline oracle's extended score set (evidence: no
gain over radius/update — kept as an algorithm-exploration tool). It is the
update score minus the weight-decay term, computed from the PRE-commit m/v
with STATE-MATCHED bias corrections: the correction exponent is the number
of commits already applied (upcoming commit - 1), so at the very first step
(no commit yet) the score is 0 by definition.

The pre-commit moments are only reachable inside the EF kernel pass that
overwrites them, so the kernel emits the score into the fp32 main-grad copy
on full-sync steps and this plugin is a pure declaration over the
materialized views.

scoring_input = 'synced': exists only on full-sync steps; selection (and
the global topk over it) happens at cycle boundaries only.
"""

from sga.plugins import ScoreDecl

DECL = ScoreDecl(name="snr_m", scoring_input="synced", has_threshold_path=False)


class SnrMScore:
    """Score plugin over the EF-materialized snr_m views."""

    decl = DECL

    has_threshold_path = False

    def __init__(self, params: dict):
        self.params = params

    def values(self, materialized_views):
        """Score values = the materialized snr_m tensors (non-negative)."""
        return list(materialized_views)

    def threshold_mask(self, grad_views, threshold: float, block_size: int = 1):
        raise NotImplementedError(
            "snr_m has no fused threshold path; use a ranked selection"
        )
