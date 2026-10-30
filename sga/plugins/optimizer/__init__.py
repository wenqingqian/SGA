# Copyright (c) 2026, SGA framework contributors.
"""Optimizer-axis plugins: the masked optimizer step + buffer reset contract.

An optimizer template owns the ENTIRE inner optimizer step for ALL params
(torch's ``optimizer.step()`` cannot step a subset of param_groups, and the
SGA kernels update m/v at every coordinate every step). It also owns the
post-update selective buffer reset semantics; the reset has exactly ONE
execution site — the end of ``engine.optimizer_step``.

Score materialization is part of the same template: on full-sync steps the
commit kernel writes the configured score (see the score registry) into the
fp32 main-grad copy, where the engine's reselection reads it (the
optimizer->score handoff). Which scores a template can emit is the
template's own kernel-branch inventory (``_TRITON_SCORE_MAP`` — one per
kernel file, ``adamw.py`` and ``adams.py``, kept in lockstep; a configured
score outside it fails loud at construction of that template).
"""

from dataclasses import dataclass, field
@dataclass
class ParamRecord:
    """One (model param, fp32 main param) pair with everything the optimizer needs."""

    model_param: object      # bf16 param living in the DDP bucket
    main_param: object       # fp32 twin actually being optimized
    bucket_state: object     # engine BucketState (mask_u8, spans)
    start: int               # element span of this param within the bucket
    end: int
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    step: int                # Adam step count (sga_advance_step semantics)

    @property
    def grad(self):
        """Main grad: post-prepare_grads this is the (clipped) synced gradient
        view; on full steps it becomes the materialized score after
        ``full_step``."""
        return self.main_param.grad

    @property
    def m(self):
        return self._state["exp_avg"]

    @property
    def v(self):
        return self._state["exp_avg_sq"]

    _state: dict = field(default=None, repr=False)

    def mask_u8(self):
        """Element-level uint8 support mask slice for this param (or None)."""
        if self.bucket_state is None or self.bucket_state.active_mask_u8 is None:
            return None
        return self.bucket_state.active_mask_u8[self.start:self.end]


@dataclass
class OptContext:
    """Everything the optimizer sees for one optimizer step.

    No score-handoff payload: the reselection reads the materialized
    ``main_param.grad`` views that the engine re-derives itself (see the
    engine module docstring on the declaration-only score axis).
    """

    records: list                       # [ParamRecord]
    phase: str                          # 'full' | 'sparse'
    support: object = None              # active Support on sparse steps
    bucket_states: list = None          # engine BucketStates (for reset)
    fold_window: int = 1                # residual fold divisor = accumulation
                                        # window between full syncs (period,
                                        # or full_sync_interval when decoupled)
    next_step_is_full: bool = False     # for residual fold decisions
    apply_coeff: float = 1.0            # sparse-step clip coeff, folded into
                                        # the grad-correct read (grad_correct
                                        # multiplies in-register — bitwise the
                                        # standalone masked-scale pass; full
                                        # steps keep 1.0, they scale in place)
    compact_values: dict = None         # compact-emission sparse steps only:
                                        # {id(bucket_state): values view} the
                                        # backend returned from allreduce_
                                        # sparse — None/empty on writeback
                                        # backends (masked path) and full
                                        # steps
