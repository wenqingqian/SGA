# Copyright (c) 2026, SGA framework contributors.
"""The SGAEngine: the single runtime object of the SGA framework.

Holds the globally-unique cycle manager, one generic BucketState per gradient
bucket, the three plugin objects plus the score NAME (scores are registry
data, not objects), the three-layer cross-rank consistency machinery, the
shared masked-clip service, and the unified per-step metrics log. Scheme-free
by construction: behavior is selected by catalog parameters
(sga/config/arguments.py) and implemented behind scheme-free plugin capability
declarations.

optimizer_step ordering contract (frozen):

    prepare_grads() -> engine masked clip -> count_zeros (BEFORE the
    optimizer template: its full step overwrites main grads with the
    materialized score) -> optimizer/score/selection ->
    _copy_main_params_to_model_params -> selective reset (old support first,
    then switch) -> return (success, grad_norm, num_zeros)

A failed step returns (False, None, None), fully clears the buffers and
consumes its cycle position without performing a reselection (this fork's
train loop advances the iteration past a failed step — it is skipped, not
retried). The post-update selective reset replaces next iteration's dense
zero_grad_buffer (the static guard at the fork's train_step site); the
engine replicates that call's other side effects — bucket-group flag resets
and the param ``grad_added_to_main_grad`` flags — itself.

Two deliberate deviations from stock ``optimizer.step()`` return semantics:
the grad norm is computed even when ``clip_grad == 0`` (stock returns 0.0),
and ``num_zeros`` is None whenever the fork's ``log_num_zeros_in_grad``
flag is off (stock returns 0).

Scores are NAMES in the registry (sga/plugins/scores.py). On a reselection
boundary the engine reads the materialized ``main_param.grad`` views the
optimizer template's full-step kernel emits and hands them to the selection
DIRECTLY — no score object sits on the data path, so there is nothing to
skip and nothing to keep in sync with the kernel's math.

Shadow mode (sga_shadow) bypasses the whole contract above: the step is a
literal replication of the stock MixedPrecisionOptimizer.step() (stock
clip included — the engine's masked clip is NOT bitwise stock), with only
insertions that are read-only with respect to model/optimizer training
tensors and RNG streams (trace dump slot, cycle-boundary shadow scoring
whose RNG-freedom is asserted), and a full end-of-step clear. See
``_shadow_step``.
"""

import json
import os

import torch

from sga import backend_timer
from sga.kernels import (
    compact_norm_sq_contribution,
    masked_norm_sq_contribution,
    scatter_selected,
    sga_advance_step,
    sga_peek_step,
)
from sga.plugins.optimizer import OptContext, ParamRecord
from sga.plugins.scores import get_score_spec
from sga.shadow import readonly_score
from sga.support import BucketSupport, IndexPool, Support
from sga.trace import TraceCollector


class SGAEngineError(RuntimeError):
    """Raised on cross-rank inconsistency or contract violations."""


class BucketState:
    """Engine-side per-bucket generic state (no scheme knowledge)."""

    def __init__(self, grad_data, dp_group, gradient_scaling_factor,
                 bucket_id):
        self.grad_data = grad_data                  # fp32 flat view (read-write)
        self.dp_group = dp_group
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
        self.support_index = None                   # index into Support.buckets
        # Element spans of the params living in this bucket.
        self.param_spans = []                       # [(model_param, start, end)]
        self.param_names = []
        # Extra main grads owned by the buffer this bucket lives in.
        self.extra_main_grads = []                  # (assigned at attach)
        self.active_mask_u8 = None                  # active support mask (set by engine)


class SelectionContext:
    """What a selection sees at a reselection boundary (CPU-testable shape).

    ``grad_views`` is the ELIGIBLE pool: when always-sync layers are
    active the engine excludes the forced params from the views, so
    ``k`` and every ratio statistic (the adaptive controller's
    achieved-ratio included) are computed over the competing coordinates
    only. A selector must not assume full-model coverage of the views.
    """

    def __init__(self, grad_views, comm_group, warmup_target=None):
        self.grad_views = grad_views
        self.comm_group = comm_group
        # Gated warmup selects implied by the selection's OWN warmup window,
        # supplied via its optional ``warmup_sync_target(start_iter, period)``
        # hook (None when the selection declares no window).
        self.warmup_target = warmup_target


class CycleManager:
    """Globally-unique cycle accounting (engine derives steps from the fork's
    iteration number — see SGAEngine.optimizer_step for the consistency assert).

    Steps are 1-based. A step is FULL (dense sync) when forced, when still in
    the bootstrap region (<= start_iter), or when no support exists yet.
    Cycle boundaries ((step - start_iter) % period == 0, step > start_iter)
    are STRUCTURALLY full: reselection must read dense-averaged,
    rank-identical scores, so mask regeneration without a full sync is not
    a representable schedule. With ``decouple`` on, an additional full-sync
    grid ``(step - start_iter) % full_sync_interval == 0`` interleaves EXTRA
    dense steps between boundaries (same origin, so boundary steps are a
    subset of it — validated at config time: 0 < interval < period and
    period % interval == 0). Reselection happens at cycle boundaries on
    non-forced steps only — forced full steps advance the cycle but never
    reselect.
    """

    def __init__(self, start_iter: int, period: int, decouple: bool,
                 full_sync_interval: int):
        assert period >= 1
        if decouple:
            assert (full_sync_interval is not None
                    and 0 < full_sync_interval < period
                    and period % full_sync_interval == 0), (
                f"decoupled full_sync_interval ({full_sync_interval}) must "
                f"satisfy 0 < full_sync_interval < period ({period}) and "
                "period % full_sync_interval == 0"
            )
        else:
            assert full_sync_interval is None, (
                "full_sync_interval is only settable with "
                "decouple_full_sync_interval_and_period=true; boundaries are "
                "full syncs by construction (leave it unset)"
            )
        self.start_iter = start_iter
        self.period = period
        self.decouple = decouple
        self.full_sync_interval = full_sync_interval
        self.force_next = False  # grad-norm bypass, consumed at next grad_sync

    @property
    def fold_window(self) -> int:
        """Residual accumulation window between full syncs — the
        ``residual_mean`` fold divisor (the period, or the tighter
        full_sync_interval when decoupled)."""
        return self.full_sync_interval if self.decouple else self.period

    def reselect_boundary(self, step: int) -> bool:
        return step > self.start_iter and (step - self.start_iter) % self.period == 0

    def extra_full(self, step: int) -> bool:
        """Decoupled mid-cycle full syncs (boundary steps are included by
        the divisibility invariant; is_full checks boundaries first)."""
        return (self.decouple
                and step > self.start_iter
                and (step - self.start_iter) % self.full_sync_interval == 0)

    def is_full(self, step: int, force: bool = False) -> bool:
        if force or step <= self.start_iter:
            return True
        if self.reselect_boundary(step):
            return True
        if self.extra_full(step):
            return True
        return False


class SGAEngine:
    """Engine over the plugin objects; built by sga.runtime.build_engine."""

    def __init__(self, score_name: str, selection, opt, backend):
        """Reads its knobs straight off the sga args singleton
        (``sga.config.get_args`` — the validated copy build_engine seeded
        from the megatron namespace; validation and percent/list
        write-backs already ran, so reads below are plain attribute
        lookups). ``score_name`` stays an explicit argument: it is the
        engine's identity (shadow scoring, resume identity), not just a
        config value — build_engine passes ``args.sga_score``, and the
        optimizer derives its materialized score from the same key, so
        the two cannot desync on the build_engine route; direct
        construction is guarded by the assert below."""
        from sga.config import get_args

        cfg = get_args()
        self.score_name = score_name
        self.selection = selection
        self.opt = opt
        self.backend = backend
        assert self.opt.emit_score == self.score_name, (
            f"engine score '{score_name}' != optimizer emit_score "
            f"'{self.opt.emit_score}' — shadow would score a different "
            "formula than the optimizer materializes")

        self._cycle = CycleManager(
            start_iter=cfg.sga_cycle_start_iter,
            period=cfg.sga_cycle_period,
            decouple=cfg.sga_cycle_decouple_full_sync_interval_and_period,
            full_sync_interval=cfg.sga_cycle_full_sync_interval,
        )
        # Backend layout: which index form the idx backends transport
        # (er/eb/bb — see sga.plugins.resolve_layout). Read off the BACKEND
        # object (the emission precedent — the config key stays the backend
        # group's single consumer); validation resolved 'auto' onto the
        # singleton before construction in production, and an unresolved
        # 'auto' (direct construction in unit tests) belts through the
        # selection's own block granularity. Drives the activation-time
        # support materialization and the attach-time span-alignment gate
        # (bb on descriptor-form backends).
        layout = getattr(self.backend, "layout", "auto")
        if layout == "auto":
            layout = ("er" if getattr(self.selection, "block_size", 1) == 1
                      else "bb")
        self._backend_layout = layout
        self._selection_block_size = getattr(self.selection, "block_size", 1)
        self._chunks = {}                 # id(chunk) -> chunk
        self._bucket_states = []          # global ordered list (Support alignment)
        self._bucket_state_of = {}        # id(bucket) -> BucketState
        self._param_locations = {}        # id(model_param) -> (bucket_state, start, end)
        self._selection_meta = None       # per-param (bs, start, end) view metadata
        self._active_support = None
        # Compact-emission step state: {id(BucketState): the
        # values view allreduce_sparse returned}. Filled at grad_sync only
        # by compact backends, consumed in optimizer_step (clip + dump
        # rebuild + grad_correct handoff), wiped at the first sync of every
        # step and on the failed-step path — a stale view must never
        # outlive the pack that wrote it.
        self._compact_values = {}
        self._compact_values_step = None
        self._mask_pool = None            # per-bucket [slot_a, slot_b] uint8 pair (lazy)
        self._mask_turn = 0               # active slot index (flips at activate)
        self._idx_pools = None            # per-bucket grow-only IndexPool list (lazy)
        self._records = None              # [ParamRecord], built lazily
        self._refresh_step = None         # engine step the records were last refreshed at
        self._main_to_record = {}
        self._iter = 0                    # fork iterations consumed (assert-checked)
        self._first_step = True           # seeding guard: first optimizer_step call
        self._step_full = None            # set at grad_sync, consumed at optimizer_step
        self._step_forced = False
        self._mp_opt = None
        self._log_num_zeros = None        # resolved from args at first step
        # Layer-1 count guard toggle (sga_checks_count_guard, default on).
        # The guard runs once per cycle at support activation — within a
        # cycle the support is immutable and its counts are cached, so
        # per-step re-checks re-verify identical numbers.
        self._count_guard = cfg.sga_checks_count_guard
        # Layer-2/3 + shadow-RNG-bracket toggles (sga_checks_*, default
        # on). These are the RESOLVED values: a false sga_checks master
        # forced them False in validate_sga_args before this constructor
        # ran, so the one-time WARN below states the checks posture of
        # the whole run unambiguously in the hostcat.
        self._support_consistency = cfg.sga_checks_support_consistency
        self._norm_spread = cfg.sga_checks_norm_spread
        self._rng_bracket = cfg.sga_checks_rng_bracket
        if not (self._count_guard and self._support_consistency
                and self._norm_spread and self._rng_bracket):
            # rank-once (ctor runs before self._rank is set below): the
            # posture line is for the merged hostcat; per-rank copies
            # would say nothing new. The consequence sentences follow the
            # actual off layers — the shadow-local RNG bracket is not a
            # cross-rank detector, so its sentence must not claim one.
            if (not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0):
                _msg = (f"[SGA] WARN sga_checks escape hatch active "
                        f"(L1_count={int(self._count_guard)} "
                        f"L2_support={int(self._support_consistency)} "
                        f"L3_norm_spread={int(self._norm_spread)} "
                        f"rng_bracket={int(self._rng_bracket)}): "
                        "benchmark measurement posture, not a production "
                        "setting")
                if not (self._count_guard and self._support_consistency
                        and self._norm_spread):
                    _msg += (" — cross-rank divergence detection is OFF "
                             "(diverged supports hang or silently fork "
                             "instead of aborting)")
                if not self._rng_bracket:
                    _msg += (" — shadow RNG-invariant bracket skipped "
                             "(shadow runs only)")
                print(_msg, flush=True)
        # Always-sync layers (the per-parameter dense axis): None = off.
        # Resolved against attached param names at attach
        # (_always_sync_attach_check); the forced ids drive the selection
        # interception in _run_selection.
        layers = cfg.sga_always_sync_layers
        self._always_sync_layers = list(layers) if layers is not None else None
        self._forced_param_ids = set()
        # Trace collector: None unless sga_trace_dir is set — the dump slot
        # below then costs one None-check per step (zero-overhead-when-off).
        self._trace = (TraceCollector(
            dir=cfg.sga_trace_dir,
            branch_points=cfg.sga_trace_branch_points,
            window=cfg.sga_trace_window,
            dump_dtype=cfg.sga_trace_dump_dtype,
            async_enabled=cfg.sga_trace_async,
            async_queue_depth=cfg.sga_trace_async_queue_depth,
            async_max_workset_gib=cfg.sga_trace_async_max_workset_gib)
            if cfg.sga_trace_dir is not None else None)
        # Shadow mode (sga_shadow): replicate the stock step() literally and
        # only READ state at cycle boundaries. shadow_score=false strips even
        # the read-only scoring (the stock-replication probe configuration:
        # bitwise risk lives in the replication machinery, not the reads).
        self._shadow = cfg.sga_shadow
        self._shadow_score = cfg.sga_shadow_score
        self._metrics_file = os.environ.get("SGA_METRICS_FILE")
        self._rank = 0
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()

    # ------------------------------------------------------------------ attach

    def attach(self, model_chunk) -> None:
        """Call point 1: walk the chunk's bucket groups, build BucketStates.

        Structural duck-typing (the only caller is DDP.__init__'s tail; a
        structural check keeps the engine CPU-testable without megatron).
        """
        for attr in ("bucket_groups", "expert_parallel_bucket_groups", "buffers",
                     "module", "ddp_config", "params_with_grad"):
            if not hasattr(model_chunk, attr):
                raise TypeError(
                    f"SGA v1 attaches to megatron DDP chunks (missing .{attr}); "
                    f"got {type(model_chunk).__name__}"
                )
        if id(model_chunk) in self._chunks:
            return
        self._chunks[id(model_chunk)] = model_chunk

        # bucket -> owning buffer (extra_main_grads live on the buffer).
        buffer_of = {}
        for buffer in model_chunk.buffers + model_chunk.expert_parallel_buffers:
            for bucket in buffer.buckets:
                buffer_of[id(bucket)] = buffer

        name_of = {id(p): n for n, p in model_chunk.module.named_parameters()}

        groups = model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
        for group in groups:
            dp_group = getattr(group, "data_parallel_group", None)
            for bucket in group.buckets:
                bs = BucketState(
                    grad_data=bucket.grad_data,
                    dp_group=dp_group,
                    gradient_scaling_factor=bucket.gradient_scaling_factor,
                    bucket_id=bucket.bucket_id,
                )
                bs.support_index = len(self._bucket_states)
                self._bucket_state_of[id(bucket)] = bs
                buffer = buffer_of.get(id(bucket))
                bs.extra_main_grads = list(getattr(buffer, "extra_main_grads", []))
                for param in bucket.params_list:
                    start, end = bucket.param_to_index[param]
                    bs.param_spans.append((param, start, end))
                    bs.param_names.append(name_of.get(id(param), f"param@{id(param)}"))
                    self._param_locations[id(param)] = (bs, start, end)
                self._bucket_states.append(bs)

        self._selection_meta = None  # rebuilt after a new chunk attaches
        self._always_sync_attach_check()
        self.opt.ensure_state(self._bucket_states)
        # bb-on-descriptor startup gate: the packed descriptor
        # form is only representable when every param span STARTS
        # x-aligned (a head's low log2(x) bits carry the tail length, so
        # an offset span start is unencodable). The bucket layout is
        # final here, and the gate deliberately runs BEFORE the backend
        # prepare loop below — a refusal must not leave half-built raw
        # comms (and their warmup allocations) behind — naming the
        # offending bucket/span/param at start of run instead of an
        # assert deep inside the first reselection's packing. The
        # torch bb path is exempt: its block tiling is span-relative and
        # needs no absolute alignment.
        if (self._backend_layout == "bb"
                and getattr(self.backend, "block_form", "block_idx")
                == "block_desc"):
            self._bb_alignment_check()
        # The comm groups just became known (DDP bucket construction is a
        # world-collective, so attach runs symmetrically on every rank):
        # give the backend its optional eager-setup point BEFORE the training
        # loop expands the caching allocator. The bucket grad-buffer device
        # rides along as the ground truth for where collectives will run.
        # Generic hook probe — backends without one simply skip it.
        prepare = getattr(self.backend, "prepare", None)
        if callable(prepare):
            seen_ids = set()
            for bs in self._bucket_states:  # deterministic global bucket order
                dp_group = bs.dp_group
                if dp_group is None or id(dp_group) in seen_ids:
                    continue
                seen_ids.add(id(dp_group))
                prepare(dp_group, device=bs.grad_data.device)

    def _bb_alignment_check(self) -> None:
        """Refuse non-x-aligned param span starts for descriptor bb."""
        x = self._selection_block_size
        for bs in self._bucket_states:
            for (param, start, end), name in zip(bs.param_spans,
                                                 bs.param_names):
                if start % x:
                    raise SGAEngineError(
                        f"sga_backend_layout bb (nccl descriptor packing) "
                        f"needs every param span start {x}-aligned; bucket "
                        f"{bs.bucket_id} param '{name}' spans "
                        f"[{start}, {end}) — the descriptor head's low "
                        f"log2({x}) bits carry the tail length and cannot "
                        "encode the offset. Pad the bucket layout to "
                        "alignment or use layout eb (element transport)")

    # -------------------------------------------------------------- grad sync

    def grad_sync(self, model_chunk, force_all_reduce: bool) -> bool:
        """Call point 2. True = engine handled every group of this chunk."""
        if id(model_chunk) not in self._chunks:
            return False
        if self._shadow:
            # Stock dense sync in the fork's own call context (the literal
            # replication covers collectives too; no phase flags to stamp).
            return False
        step = self._iter + 1
        # timer iter stamp: every timed region in this step (sparse
        # backends + the boundary finish_grad_sync delegation) reads it
        backend_timer.set_iter(step)
        forced = bool(force_all_reduce) or self._cycle.force_next
        if self._cycle.force_next:
            self._cycle.force_next = False
        full = forced or self._cycle.is_full(step) or self._active_support is None
        self._step_full = full
        self._step_forced = forced

        # Step-scoped compact state: wipe at the FIRST sync of a step (the
        # marker survives optimizer_step, which bumps _iter at its end), so
        # multi-chunk syncs accumulate instead of clobbering each other.
        if self._compact_values_step != step:
            self._compact_values = {}
            self._compact_values_step = step

        if full:
            # Delegate to the REAL dense path — bitwise parity exactly where
            # it matters (full-sync steps produce the scoring input).
            model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
            return True

        avg = getattr(model_chunk.ddp_config, "average_in_collective", False)
        reduce_op = torch.distributed.ReduceOp.AVG if avg else torch.distributed.ReduceOp.SUM
        # the backend instance's RESOLVED emission (inherent_emission is
        # only the inherent default; torch_pipelined_index switches)
        compact = self.backend.emission == "compact"
        for bucket_key, bs in self._chunk_buckets(model_chunk):
            support = self._active_support.buckets[bs.support_index]
            factor = 1.0 if avg else bs.gradient_scaling_factor
            out = self.backend.allreduce_sparse(
                bs.grad_data, support, bs.dp_group, factor, reduce_op, key=bucket_key,
            )
            if compact:
                # The declared contract: a compact backend ALWAYS returns a
                # Tensor (empty when nothing was selected — None is the
                # writeback backends' signature and must not alias it).
                if not isinstance(out, torch.Tensor):
                    raise SGAEngineError(
                        "compact-emission backend returned "
                        f"{type(out).__name__} instead of a values Tensor — "
                        "the clip/grad_correct compact paths would silently "
                        "consume a stale buffer"
                    )
                self._compact_values[id(bs)] = out
        return True

    def _chunk_buckets(self, model_chunk):
        """(persistent key, BucketState) for the chunk's buckets."""
        out = []
        for group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            for bucket in group.buckets:
                bs = self._bucket_state_of.get(id(bucket))
                if bs is not None:
                    out.append((id(bucket), bs))
        return out

    # --------------------------------------------------------- optimizer step

    def optimizer_step(self, optimizer, model_chunks, iteration: int):
        """Call point 3 (frozen ordering contract; see module docstring).

        Resume-relative accounting: the engine's counter is seeded from the
        fork's ABSOLUTE iteration on the first call (a checkpoint resumed at
        iteration K arrives here with K, not 1), then the consistency assert
        below holds from that point on. The cycle calendar cold-starts at
        the resume point; with no support yet the bootstrap guard forces
        full steps until the first reselection.
        """
        mp_opt = self._unwrap_optimizer(optimizer)
        if self._first_step:
            self._first_step = False
            self._iter = iteration - 1
        was_full = self._step_full
        if was_full is None:  # no grad_sync seen (defensive)
            was_full = True
        step = self._iter + 1
        assert iteration == step, (
            f"SGA engine step {step} != fork iteration {iteration}; "
            "eval/rerun interleaving broke the step accounting"
        )
        if self._shadow:
            return self._shadow_step(mp_opt, model_chunks, iteration, step)

        # 1. prepare (copy + unscale + inf check + scaler update). NOTE: the
        # megatron contract returns found_inf (True = BAD), not success.
        found_inf = mp_opt.prepare_grads()
        if found_inf:
            self._full_clear(model_chunks)
            self._iter = iteration
            # Symmetric with the success path: no stale phase flags can leak
            # into the next step's bookkeeping (grad_sync overwrites them,
            # but keep the state self-consistent at rest).
            self._step_full = None
            self._step_forced = False
            return (False, None, None)

        # 2. masked clip (single shared service; unconditional DP-bitwise fix).
        # Sparse steps get the coeff back for grad_correct's fused multiply
        # instead of an in-place masked-scale pass here.
        grad_norm, sparse_coeff = self._masked_clip_grad_norm(
            mp_opt, was_full, step)

        # 3. num_zeros BEFORE the optimizer template (its full step
        # overwrites main grads with the materialized score). Compact-
        # emission sparse steps report the PACKED view (zeroed
        # communicated coords read as zeros) — the writeback-view
        # correction scan is waived so the diagnostic costs the same on
        # every emission mode (see _count_zeros).
        num_zeros = self._count_zeros(mp_opt) if self._log_num_zeros_flag() else None

        # 3b. trace dump slot (frozen slot: AFTER count_zeros, BEFORE the
        #     optimizer rewrites main_param.grad — the post-unscale post-clip
        #     view the offline replay consumes; grad_norm is this step's own
        #     clip). Compact-emission sparse steps rebuild the full-layout
        #     view IN PLACE for the dump window (the clip service already
        #     scaled the values), then restore the buffer-state invariant
        #     (selected coords zero) before grad_correct — dump steps only.
        rebuilt = None
        if (self._trace is not None and not was_full and self._compact_values
                and step is not None and any(self._trace.wants(step))):
            rebuilt = self._rebuild_full_views()
        try:
            if self._trace is not None:
                self._trace.maybe_dump(self, mp_opt, step, grad_norm)
        finally:
            # Exception-safe: a dump I/O failure must not leave the
            # scattered values in the grad buffer (the selected-coords-
            # zero invariant is step-scoped state) — restore first, then
            # let the original error propagate.
            if rebuilt:
                self._restore_compact_invariant(rebuilt)

        # 4. The optimizer template owns the entire inner step.
        records = self._refresh_records(mp_opt)
        next_full = self._cycle.is_full(step + 1)
        opt_ctx = OptContext(
            records=records,
            phase="full" if was_full else "sparse",
            support=self._active_support,
            bucket_states=self._bucket_states,
            fold_window=self._cycle.fold_window,
            next_step_is_full=next_full,
            apply_coeff=1.0 if was_full else sparse_coeff,
            compact_values=(self._compact_values
                            if (not was_full and self._compact_values)
                            else None),
        )
        if was_full:
            self.opt.full_step(opt_ctx)
        else:
            self.opt.sparse_step(opt_ctx)

        # 5. copy main -> model.
        mp_opt._copy_main_params_to_model_params()

        # 6. Reselection at non-forced cycle boundaries (old support completes
        #    its residual handling BEFORE the switch).
        reselect = self._cycle.reselect_boundary(step) and not self._step_forced
        if reselect:
            meta = self._selection_context()
            # Materialized-score views, IN META ORDER: records follow the
            # optimizer's param order, bucket spans follow the DDP bucket
            # order — they differ in real megatron.
            views = [p.main_param.grad.data.view(-1)
                     for (_, _, _, p) in meta]
            if self._norm_spread:
                self._check_norm_spread(views)
            new_support = self._run_selection(views, meta)
        else:
            new_support = None

        # 7. Selective reset (optimizer-owned semantics, old support), then switch.
        self.opt.reset(opt_ctx, was_full=was_full)
        if new_support is not None:
            self._activate_support(new_support)
            # Layer-1 deadlock guard, once per cycle at activation: the counts
            # being verified are the activation-time caches, and the support
            # is immutable until the next reselection — checking here covers
            # every sparse sync of the coming cycle.
            if self._count_guard:
                self._check_counts()

        # Grad-norm bypass: force next step full (globally consistent input).
        if hasattr(self.selection, "wants_force_full"):
            if self.selection.wants_force_full(grad_norm):
                self._cycle.force_next = True

        # 8. Replace zero_grad_buffer's remaining side effects.
        self._reset_host_state(model_chunks)

        self._log_step(step, was_full, reselect, grad_norm)
        self._step_full = None
        self._step_forced = False
        self._iter = iteration
        return (True, grad_norm, num_zeros)

    # ------------------------------------------------------------- shadow mode

    def _shadow_step(self, mp_opt, model_chunks, iteration, step):
        """Literal replication of the stock MixedPrecisionOptimizer.step().

        prepare_grads → [found_inf: full clear + stock early return] →
        STOCK clip_grad_norm → STOCK count_zeros → [trace dump slot] →
        [cycle-boundary read-only scoring/selection, shadow support] →
        stock step_with_ready_grads (commit + copy-back) → full clear.

        The engine's masked clip service is deliberately NOT used (it is not
        bitwise the stock clip); the optimizer template never runs; the end-of-step full
        clear replaces next iteration's zero_grad_buffer, which the fork's
        static guard skipped because the engine is active. Trace dumps in
        shadow see the stock clipped dense gradient — the replay-grade view.
        """
        found_inf = mp_opt.prepare_grads()
        if found_inf:
            self._full_clear(model_chunks)
            self._iter = iteration
            self._step_full = None
            self._step_forced = False
            return (False, None, None)

        grad_norm = 0.0
        if getattr(mp_opt.config, "clip_grad", 0.0) > 0.0:
            grad_norm = mp_opt.clip_grad_norm(mp_opt.config.clip_grad)
        num_zeros = 0
        if getattr(mp_opt.config, "log_num_zeros_in_grad", False):
            num_zeros = mp_opt.count_zeros()

        reselect = self._cycle.reselect_boundary(step)
        reselected = reselect and self._shadow_score

        # Same frozen dump slot as the live path: after count_zeros, before
        # the commit (the optimizer slot it precedes in live mode does not exist
        # here — the commit itself is the writer being avoided). Records are
        # NOT pre-refreshed: the dump refreshes them lazily only when a dump
        # actually fires, so the probe configuration runs record-free.
        if self._trace is not None:
            self._trace.maybe_dump(self, mp_opt, step, grad_norm)

        if reselected:
            self._shadow_reselect(self._refresh_records(mp_opt))

        success = mp_opt.step_with_ready_grads()

        self._full_clear(model_chunks)
        self._log_step(step, True, reselected, grad_norm)
        self._step_full = None
        self._step_forced = False
        self._iter = iteration
        return (bool(success), grad_norm, num_zeros)

    def _shadow_reselect(self, records):
        """Read-only scoring + selection at a shadow cycle boundary.

        Reuses the live reselection machinery (view routing, selection,
        bucket assembly, consistency checks, activation) with the score
        tensors coming from sga.shadow.readonly_score — the registry formula
        evaluated on the pre-commit state — instead of the optimizer
        template's fused materialization.
        The RNG-state assert brackets the whole insertion: the boundary
        must be consumption-free on every RNG stream (selections are
        histogram/topk-based by design — a violation means a plugin broke
        that contract and would silently fork dense trajectories).
        The bracket itself is the sga_checks_rng_bracket escape hatch:
        off (benchmark posture) skips both the capture and the compare.
        """
        rng_before = torch.get_rng_state() if self._rng_bracket else None
        cuda_before = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() and self._rng_bracket
                       else None)

        rec_of = {id(rec.model_param): rec for rec in records}
        emit = self.score_name
        meta = self._selection_context()
        # Two largest-param-sized buffers carry the shared formula-chain
        # temps (bitwise-identical out= routing) — the boundary's
        # allocator high-water otherwise spikes several param-sized
        # allocations on the largest param.
        scratch = None
        if records and get_score_spec(emit).scratch:
            max_n = max(r.main_param.numel() for r in records)
            ref = records[0].main_param
            scratch = (
                torch.empty(max_n, dtype=ref.dtype, device=ref.device),
                torch.empty(max_n, dtype=ref.dtype, device=ref.device),
            )
        views = []
        for (_, _, _, p) in meta:
            rec = rec_of.get(id(p))
            if rec is None:
                raise TypeError(
                    "shadow reselection: meta param without a record "
                    f"(shape {tuple(p.shape)}); records were refreshed "
                    "from a different optimizer view"
                )
            views.append(readonly_score(rec, emit, scratch).view(-1))
        if self._norm_spread:
            self._check_norm_spread(views)
        self._activate_support(self._run_selection(views, meta))
        if self._count_guard:
            self._check_counts()

        if self._rng_bracket:
            if not torch.equal(torch.get_rng_state(), rng_before):
                raise RuntimeError(
                    "shadow: boundary scoring consumed CPU RNG — selections "
                    "must be deterministic (histogram/topk), this forks "
                    "dense trajectories and breaks the shadow invariant"
                )
            if cuda_before is not None:
                cuda_after = torch.cuda.get_rng_state_all()
                if len(cuda_after) != len(cuda_before) or any(
                    not torch.equal(a, b)
                    for a, b in zip(cuda_before, cuda_after)
                ):
                    raise RuntimeError(
                        "shadow: boundary scoring consumed CUDA RNG — "
                        "selections must be deterministic (histogram/topk), "
                        "this forks dense trajectories and breaks the "
                        "shadow invariant"
                    )

    def _always_sync_attach_check(self):
        """Resolve the always-sync patterns against every attached param
        name — at attach (model build), so a typo'd pattern dies here,
        not at the first reselection. Two-tier zero-match policy: the
        whole list matching nothing is refused (a silently-dead knob is
        worse than a loud refusal — this tier is also what catches a
        broken comma join), while a single unmatched pattern only WARNs
        (fallback spellings across model variants legitimately hit a
        subset). Matching EVERY parameter is refused too: the topk
        competition pool would be empty (``k_for(0)`` degenerates to
        selecting one coordinate out of nothing). The cumulative rescan
        across attaches keeps the ``n_eligible == 0`` tier honest; the
        whole-list ZERO-MATCH tier is exact only under the pinned
        one-chunk-per-rank topology (vpp > 1 is refused at build time,
        and pp > 1 refuses the feature outright in build_engine)."""
        if self._always_sync_layers is None:
            return
        patterns = self._always_sync_layers
        matched_patterns = set()
        matched_names = []
        n_forced = 0
        n_total = 0
        for bs in self._bucket_states:
            for (param, start, end), name in zip(bs.param_spans,
                                                 bs.param_names):
                n_total += end - start
                # every matching pattern counts as matched (overlapping
                # patterns are a union — the param is forced once)
                hits = [p for p in patterns if p in name]
                if hits:
                    matched_patterns.update(hits)
                    matched_names.append(name)
                    n_forced += end - start
                    self._forced_param_ids.add(id(param))
        if not matched_names:
            sample = ", ".join(name for bs in self._bucket_states[:2]
                               for name in bs.param_names[:4])
            raise ValueError(
                f"sga_always_sync_layers: none of the patterns {patterns} "
                "matches any parameter name (attached names include "
                f"{sample or '(none)'}); refusing a silently-dead knob — "
                "fix the patterns")
        n_eligible = n_total - n_forced
        if n_eligible == 0:
            raise ValueError(
                "sga_always_sync_layers matches EVERY parameter — the "
                "topk competition pool is empty; drop patterns or run "
                "dense training")
        if self._rank == 0:
            for p in patterns:
                if p not in matched_patterns:
                    print(f"[SGA] WARN sga_always_sync_layers pattern "
                          f"{p!r} matches no parameter name (kept: "
                          "fallback spellings across model variants may "
                          "hit a subset)", flush=True)
            k_for = getattr(self.selection, "k_for", None)
            k = k_for(n_eligible) if callable(k_for) else None
            shown = ", ".join(matched_names[:8])
            more = (f" (+{len(matched_names) - 8} more)"
                    if len(matched_names) > 8 else "")
            eff = ((k + n_forced) / n_total if k is not None
                   else n_forced / n_total)
            k_txt = f"topk k≈{k}, " if k is not None else ""
            print(f"[SGA] always-sync layers: patterns {patterns} matched "
                  f"{len(matched_names)} params ({shown}{more}); forced "
                  f"{n_forced}/{n_total} coords; {k_txt}effective "
                  f"communicated ratio ≈ {eff:.1%}", flush=True)

    def _forced_flags(self, meta):
        """Always-sync membership per selection-meta entry (only called
        with the feature ON — _run_selection branches the off path away;
        once per reselection, not per step)."""
        if self._always_sync_layers is None:
            return [False] * len(meta)
        forced = self._forced_param_ids
        return [id(m[3]) in forced for m in meta]

    def _run_selection(self, views, meta) -> "Support":
        """Shared selection tail for both reselection call sites: warmup
        probe → context → select → bucket assembly → consistency verify →
        Support construction. The CALLER owns activation timing and
        everything around it: live activates only after the optimizer's selective
        reset has consumed the old support (step 7), shadow activates
        immediately inside the RNG bracket — those orderings are load-bearing
        and deliberately stay at the call sites.

        With always-sync layers active the selection sees ONLY the
        eligible (unforced) views — ``k`` and every ratio statistic are
        computed over the competing pool by construction — and the
        forced params' all-ones masks are spliced back in before bucket
        assembly, so both support forms (the mask and the idx derived
        from it) inherit the forced spans."""
        # Feature-off fast path: no _forced_flags allocation, the original
        # views/masks objects pass through untouched.
        forced = (self._forced_flags(meta)
                  if self._always_sync_layers is not None else None)
        has_forced = any(forced) if forced else False
        sel_views = ([v for v, f in zip(views, forced) if not f]
                     if has_forced else views)
        warmup_hook = getattr(self.selection, "warmup_sync_target", None)
        sel_ctx = SelectionContext(
            grad_views=sel_views,
            comm_group=self._dp_cp_group(),
            warmup_target=(warmup_hook(self._cycle.start_iter,
                                       self._cycle.period)
                           if callable(warmup_hook) else None),
        )
        sel_masks = self.selection.select(sel_ctx)
        if has_forced:
            if len(sel_masks) != len(sel_views):
                name = type(self.selection).__name__
                raise SGAEngineError(
                    f"selection plugin '{name}' returned {len(sel_masks)} "
                    f"masks for {len(sel_views)} eligible views — the "
                    "per-param splice would misalign; a selector must "
                    "return exactly one mask per view")
            masks = []
            picked = iter(sel_masks)
            for f, (_, start, end, _), view in zip(forced, meta, views):
                if f:
                    masks.append(torch.ones(end - start, dtype=torch.uint8,
                                            device=view.device))
                else:
                    masks.append(next(picked))
        else:
            masks = sel_masks
        bucket_masks = self._assemble_bucket_masks(masks, meta)
        if self._support_consistency:
            self._verify_support_consistency(bucket_masks)
        # Block granularity rides along from the selection (1 = element).
        # The block-index form is NOT assembled here: it derives lazily
        # from the canonical mask + param spans (single-source doctrine),
        # materialized at activation for idx-consuming backends only. The
        # spans are sorted by start so the derived block_idx is ascending
        # (params_list order is an implementation detail of the DDP side —
        # block_split's midpoint searchsorted assumes ascending).
        block_size = getattr(self.selection, "block_size", 1)
        return Support([
            BucketSupport(
                m, idx_pool=self._bucket_idx_pool(i),
                block_size=block_size,
                block_spans=(sorted((start, end) for (_p, start, end)
                                    in self._bucket_states[i].param_spans)
                             if block_size > 1 else None))
            for i, m in enumerate(bucket_masks)
        ])

    def _selection_context(self):
        """Per-param view metadata for reselection.

        ``meta`` entries are (bucket_state, start, end, model_param); the
        param rides along so the callers can fetch the matching materialized
        score (main_param.grad) in the same order.
        """
        if self._selection_meta is None:
            meta = []
            for bs in self._bucket_states:
                for (param, start, end) in bs.param_spans:
                    meta.append((bs, start, end, param))
            self._selection_meta = meta
        return self._selection_meta

    def _assemble_bucket_masks(self, masks, meta):
        """Scatter per-param masks into per-bucket element-level uint8 masks.

        Double-buffered (ping-pong): the new selection is written into the
        inactive slot of a per-bucket mask pair, so the OLD support's mask
        stays intact for the selective reset that runs between assemble and
        activate — and GB-scale masks are not freshly allocated every cycle
        (allocator map/unmap storms at reselection cadence).
        """
        if self._mask_pool is None:
            self._mask_pool = [
                [torch.zeros(bs.grad_data.numel(), dtype=torch.uint8,
                             device=bs.grad_data.device) for _ in range(2)]
                for bs in self._bucket_states
            ]
        bucket_masks = [self._mask_pool[i][1 - self._mask_turn]
                        for i in range(len(self._bucket_states))]
        for buf in bucket_masks:
            buf.zero_()
        for m, (bs, start, end, _) in zip(masks, meta):
            bucket_masks[bs.support_index][start:end] = m
        return bucket_masks

    # ------------------------------------------------------------- internals

    def _bucket_idx_pool(self, i: int):
        """Grow-only idx pool for bucket ``i`` (shared across cycles)."""
        if self._idx_pools is None:
            self._idx_pools = [IndexPool() for _ in self._bucket_states]
        return self._idx_pools[i]

    def _unwrap_optimizer(self, optimizer):
        """Unwrap ChainedOptimizer-of-1 (get_megatron_optimizer ALWAYS chains,
        even for one optimizer) and verify the structural contract; the inner
        torch optimizer must be Adam/AdamW for the dense-fallback numerics."""
        inner = optimizer
        if hasattr(optimizer, "chained_optimizers"):
            assert len(optimizer.chained_optimizers) == 1, (
                "SGA v1 supports exactly one inner optimizer (no chains)"
            )
            inner = optimizer.chained_optimizers[0]
        for attr in ("float16_groups", "fp32_from_float16_groups", "optimizer",
                     "prepare_grads", "_copy_main_params_to_model_params", "config"):
            if not hasattr(inner, attr):
                raise TypeError(
                    f"SGA v1 supports Float16OptimizerWithFloat16Params-shaped "
                    f"optimizers (missing .{attr}); got {type(inner).__name__}"
                )
        torch_opt = inner.optimizer
        assert type(torch_opt).__name__ in ("Adam", "AdamW", "FusedAdam"), (
            "SGA v1's dense fallback numerics only hold for AdamW-style "
            "optimizers (torch Adam/AdamW or TE/Apex FusedAdam with "
            "adam_w_mode, enforced at build time), got "
            f"{type(torch_opt).__name__}"
        )
        self._mp_opt = inner
        return inner

    def _refresh_records(self, mp_opt):
        """Lazy ParamRecord build + once-per-step hyperparam refresh.

        Every caller lives inside a single optimizer_step (the clip service,
        the zero count, and the optimizer handoff below), so the SHARED Adam step
        counter must advance at most once per engine step: repeated visits
        within the same step are stamped out via ``_refresh_step``. Dense
        AdamW uses exactly ``iteration`` as its step; the SGA kernels read
        ``rec.step``, so the two must coincide or every bias correction is
        permanently off by one.
        """
        step_key = self._iter + 1
        if self._records is not None:
            if self._refresh_step == step_key:
                return self._records
        else:
            # Native-fp32 params land in fp32_from_fp32_groups with NO
            # record (records cover only the float16 pairing) and would
            # never receive an optimizer step: fail loudly instead of
            # stepping silently around them.
            fp32_groups = getattr(mp_opt, "fp32_from_fp32_groups", None)
            if fp32_groups:
                for group in fp32_groups:
                    for p in group:
                        if getattr(p, "requires_grad", False):
                            raise TypeError(
                                "SGA v1 found a native-fp32 param that "
                                "requires grad; such params get no optimizer "
                                f"step under the optimizer-owned step (shape "
                                f"{tuple(p.shape)}). v1 supports bf16 params "
                                "with fp32 twins only"
                            )
            self._records = []
            for model_group, main_group in zip(
                mp_opt.float16_groups, mp_opt.fp32_from_float16_groups
            ):
                for model_param, main_param in zip(model_group, main_group):
                    loc = self._param_locations.get(id(model_param))
                    bs, start, end = loc if loc else (None, 0, 0)
                    if bs is None:
                        # Without a bucket span the optimizer's residual contract
                        # silently skips this param forever.
                        raise TypeError(
                            "SGA records: model param without an attached "
                            "bucket (no DDP bucket contains it); refusing to "
                            f"build records mid-run (shape {tuple(model_param.shape)}). "
                            "Re-check attach() / chunk construction"
                        )
                    rec = ParamRecord(
                        model_param=model_param, main_param=main_param,
                        bucket_state=bs, start=start, end=end,
                        lr=0.0, beta1=0.9, beta2=0.95, eps=1e-8,
                        weight_decay=0.0, step=1,
                    )
                    # Shadow binds non-inserting (rec._state may be None —
                    # readonly_score/trace read it via ``_state or {}``): the
                    # stock commit owns state materialization, so the probe
                    # run leaves the state dict untouched. The live EFs write
                    # the lazy moments INTO this dict entry, so live keeps
                    # the inserting bind.
                    rec._state = (mp_opt.optimizer.state.get(main_param)
                                  if self._shadow
                                  else mp_opt.optimizer.state[main_param])
                    self._records.append(rec)
                    self._main_to_record[id(main_param)] = rec

        # Refresh hyperparams + shared step counter exactly once per step.
        # Shadow reads the upcoming commit PEEK-style (no group['step']
        # write): the stock commit owns its counters — TE/apex FusedAdam
        # advances group['step'] itself inside step(), and a second engine
        # advance would double-step every bias correction.
        for group in mp_opt.optimizer.param_groups:
            step = (sga_peek_step(mp_opt, group) if self._shadow
                    else sga_advance_step(mp_opt, group))
            for p in group["params"]:
                rec = self._main_to_record.get(id(p))
                if rec is None:
                    continue
                betas = group.get("betas", (0.9, 0.95))
                rec.lr = group["lr"]
                rec.beta1, rec.beta2 = betas
                rec.eps = group.get("eps", 1e-8)
                rec.weight_decay = group.get("weight_decay", 1e-2)
                rec.step = step
        self._refresh_step = step_key
        return self._records

    def _masked_clip_grad_norm(self, mp_opt, was_full, step=None):
        """Single shared masked clip service; returns ``(grad_norm,
        sparse_coeff)``.

        Managed params contribute the RAW accumulated grad at support-masked
        coords; unmanaged/full-step params contribute in full. One global
        scalar: MP-group SUM, then the unconditional DP-bitwise fix (SUM over
        DP-CP + divide by world) — without it the fused kernel's atomic-add
        ulp jitter diverges clip coefficients, then masks, then deadlocks the
        sparse allreduce.

        The coeff application routes three ways: full steps
        scale in place and return 1.0; sparse steps with a materializing
        trace dump this step scale explicitly (rank-consistently — the dump
        slot reads the post-clip view) and return 1.0; all other sparse
        steps return the coeff for grad_correct's fused in-register
        multiply. Nothing but num_zeros' zero-count (order-invariant) reads
        the grad buffer between here and that kernel.
        """
        clip_grad = getattr(mp_opt.config, "clip_grad", 0.0)
        records = self._records or self._refresh_records(mp_opt)

        # Compact-emission sparse steps: the communicated values ARE the
        # norm input (post-allreduce, pre-coeff) — a contiguous
        # deterministic reduction with no atomics (the masked kernel's
        # atomic accumulation was the run-to-run jitter source; the
        # DP-bitwise fix below stays an unconditional invariant).
        compact = (self._compact_values
                   if (not was_full and self._compact_values
                       and self.backend.emission == "compact")
                   else None)
        compact_seen = set() if compact is not None else None

        device = "cpu"
        for rec in records:
            if rec.grad is not None:
                device = rec.grad.device
                break
        total_norm_sq = torch.zeros(1, dtype=torch.float32, device=device)
        for rec in records:
            if rec.grad is None:
                continue
            if compact is not None:
                # The values tensor is BUCKET-level and covers every record
                # sharing the bucket, so it enters the reduction ONCE per
                # bucket. Each such record must also count locally: a
                # shared/tensor-parallel duplicate's coords are inside the
                # bucket values but the compact contract has no per-record
                # exclusion for them (v1 pins tp == 1, where the check is
                # trivially True — fail closed rather than miscount on a
                # future topology).
                if not self._param_counts_locally(rec):
                    raise SGAEngineError(
                        "compact clip: record does not count locally "
                        f"(shared/tensor-parallel duplicate, shape "
                        f"{tuple(rec.model_param.shape)}) — the "
                        "bucket-level values contract has no per-record "
                        "exclusion for it"
                    )
                if rec.mask_u8() is None:
                    # The pack zeroed this record's selected coords at
                    # grad_sync, so a "full contribution" would silently
                    # omit every communicated value and deflate the norm
                    # (unreachable in the pinned topology: a compact
                    # sparse step implies an active support with masks).
                    raise SGAEngineError(
                        "compact clip: record without an active mask — "
                        "its communicated coords are already zeroed in "
                        "the grad buffer and no mask-less compact "
                        "contract exists"
                    )
                vals = compact.get(id(rec.bucket_state))
                if vals is None:
                    raise SGAEngineError(
                        "compact clip: bucket state without a values entry "
                        "(grad_sync skipped a supported bucket?)"
                    )
                if id(rec.bucket_state) not in compact_seen:
                    compact_seen.add(id(rec.bucket_state))
                    if vals.numel() > 0:
                        compact_norm_sq_contribution(vals, total_norm_sq)
                # Zero selection contributes exactly zero, same as the
                # all-zero masked sum.
                continue
            if not self._param_counts_locally(rec):
                continue
            mask = None if was_full else rec.mask_u8()
            grad = rec.grad.data
            if mask is None:
                total_norm_sq += grad.float().pow(2).sum()
            else:
                masked_norm_sq_contribution(grad.float(), mask, total_norm_sq)

        stats_group = self._stats_group(mp_opt)
        # world == 1 (the mp self-group under the pinned tp == cp == pp == 1
        # world): the allreduce is an identity op, so elide it — megatron's
        # own per-step collectives on that group initialize its communicator
        # regardless.
        if (stats_group is not None
                and torch.distributed.get_world_size(stats_group) > 1):
            torch.distributed.all_reduce(
                total_norm_sq, op=torch.distributed.ReduceOp.SUM,
                group=stats_group
            )
        self._dp_bitwise_fix(total_norm_sq)
        grad_norm = total_norm_sq.sqrt().item()

        sparse_coeff = 1.0
        if clip_grad > 0.0:
            coeff = clip_grad / (grad_norm + 1.0e-6)
            if coeff < 1.0:
                # The dump slot reads a post-clip full-layout view, so on
                # steps that actually dump (wants() is config-derived —
                # identical on every rank, so no all-rank divergence) keep
                # the explicit scale and hand the kernel coeff=1.0; the
                # same fp32 multiply either way, bitwise identical to the
                # fused path. Non-dumping trace steps take the fast path.
                trace_dumps = (self._trace is not None and step is not None
                               and any(self._trace.wants(step)))
                if was_full:
                    for rec in records:
                        if rec.grad is None:
                            continue
                        rec.grad.data.mul_(coeff)
                elif compact is not None:
                    # Compact sparse step: the coeff rides the values —
                    # either in grad_correct's in-register multiply (fast
                    # path) or as an explicit values scale on dump steps
                    # (the rebuild around maybe_dump scatters the scaled
                    # values into the full layout, so the dump slot reads
                    # the same post-clip view the writeback path produced;
                    # grad_correct then runs coeff=1.0 — the same fp32
                    # multiply either way). Degenerate mask-less records
                    # keep the explicit full-buffer scale in both cases.
                    if trace_dumps:
                        for vals in compact.values():
                            if vals.numel() > 0:
                                vals.mul_(coeff)
                    for rec in records:
                        if rec.grad is None:
                            continue
                        if rec.mask_u8() is None:
                            rec.grad.data.mul_(coeff)
                    if not trace_dumps:
                        sparse_coeff = float(coeff)
                elif trace_dumps:
                    for rec in records:
                        if rec.grad is None:
                            continue
                        m = rec.mask_u8()
                        if m is None:
                            rec.grad.data.mul_(coeff)
                        else:
                            rec.grad.data.view(-1)[m > 0] *= coeff
                else:
                    # Sparse steps defer the coeff to grad_correct's fused
                    # in-register multiply; nothing reads the
                    # grad buffer between here and that kernel. Degenerate
                    # mask-less buckets take the explicit full scale (the
                    # kernel path refuses the fusion knobs without a mask).
                    for rec in records:
                        if rec.grad is None:
                            continue
                        if rec.mask_u8() is None:
                            rec.grad.data.mul_(coeff)
                    sparse_coeff = float(coeff)
        return grad_norm, sparse_coeff

    def _rebuild_full_views(self):
        """Dump-window helper (compact emission, dump steps only): scatter
        the clip-scaled values back into the full grad buffers so the dump
        slot reads the same post-clip full-layout view the writeback path
        produced. Returns the scatter list for the restore pass; the
        state invariant (selected coords zero) is re-established in
        :meth:`_restore_compact_invariant` before grad_correct runs."""
        rebuilt = []
        for bs in self._bucket_states:
            vals = self._compact_values.get(id(bs))
            if vals is None or vals.numel() == 0:
                continue
            idx = self._active_support.buckets[bs.support_index].idx
            scatter_selected(bs.grad_data, idx, vals)
            rebuilt.append((bs, idx))
        return rebuilt

    def _restore_compact_invariant(self, rebuilt):
        """Re-zero the scattered positions (dump steps only — ~the pack
        clear's write cost, once per dump)."""
        for bs, idx in rebuilt:
            scatter_selected(bs.grad_data, idx,
                             torch.zeros(idx.numel(), dtype=bs.grad_data.dtype,
                                         device=bs.grad_data.device))

    def _param_counts_locally(self, rec) -> bool:
        """Shared/tensor-parallel duplicate filters (trivially True at tp=1)."""
        from megatron.core import tensor_parallel
        from megatron.core.transformer.module import param_is_not_shared

        p = rec.model_param
        if not param_is_not_shared(p):
            return False
        tp_group = self._tp_group()
        if tp_group is not None and not tensor_parallel.param_is_not_tensor_parallel_duplicate(
            p, tp_group
        ):
            return False
        return True

    def _dp_bitwise_fix(self, total_norm_sq):
        group = self._dp_cp_group()
        if group is None:
            return
        world = torch.distributed.get_world_size(group)
        if world <= 1:
            return
        torch.distributed.all_reduce(
            total_norm_sq, op=torch.distributed.ReduceOp.SUM, group=group
        )
        total_norm_sq.div_(world)

    def _count_zeros(self, mp_opt):
        """Pre-optimizer zero count of the main grads (frozen ordering-
        contract measurement point; diagnostic only — no control flow
        consumes it). Per-record form stays the stock Megatron idiom
        (count_zeros_fp32 uses the same numel - count_nonzero per
        parameter) so the instrumentation cost is byte-identical on
        sparse and full/dense steps.

        Compact-emission sparse steps count the PACKED view: the pack
        zeroed the communicated coords at grad_sync, so they all read as
        zeros and the reported number is inflated by k - zeros_in_vals
        per bucket. Accuracy is deliberately waived here: restoring the
        writeback-view number would require a values-buffer scan that
        only compact steps pay, so the diagnostic log would price one
        emission mode above the other."""
        records = self._records or self._refresh_records(mp_opt)
        n = 0
        for rec in records:
            if rec.grad is None:
                continue
            buf = rec.grad.data
            zeros = buf.numel() - int(torch.count_nonzero(buf).item())
            n += zeros
        return n

    # -------------------------------------------------- consistency machinery

    def _check_counts(self):
        """Layer 1 (deadlock guard): support counts identical across ranks.

        Packed [k0, -k0, k1, -k1, ...] int64, one MAX allreduce; min != max
        means a diverged support — refuse the mismatched collective. Runs
        ONCE PER CYCLE, right after support activation: the per-bucket counts
        are the caches filled at activation and the support is immutable
        within the cycle, so the activation-time verdict covers every sparse
        sync of the cycle.
        """
        group = self._dp_cp_group()
        if group is None or self._active_support is None:
            return
        ks = [b.num_selected() for b in self._active_support.buckets]
        device = self._bucket_states[0].grad_data.device if self._bucket_states else "cpu"
        packed = torch.tensor(ks, dtype=torch.int64, device=device)
        packed = torch.stack([packed, -packed], dim=1).flatten()
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.MAX, group=group)
        maxs = packed[0::2].tolist()
        mins = [-v for v in packed[1::2].tolist()]
        if maxs != mins:
            raise SGAEngineError(
                f"SGA support count mismatch across ranks: local ks={ks}, "
                f"cluster min={mins}, max={maxs}. Refusing the mismatched "
                "collective (it would deadlock or corrupt)."
            )

    def _verify_support_consistency(self, masks):
        """Layer 2 (numeric guard): dual checksum over selected positions.

        Two independent per-bucket quantities — the position sum and a
        staggered position-mixed sum — each allreduced as MIN and MAX; a
        single sum silently collides on different sets, the dual values do
        not. Both are computed in CHUNK-position blocks
        from the mask directly: materializing the full nonzero() int64 index
        array once cost ~k*8 bytes (6+ GiB when a reselection lands on a high
        ratio).
        """
        group = self._dp_cp_group()
        if group is None:
            return
        device = masks[0].device
        sums = torch.zeros(2 * len(masks), dtype=torch.int64, device=device)
        chunk = 1 << 24  # 16M positions -> ~250 MB of int64 temporaries
        for i, m in enumerate(masks):
            n = m.numel()
            s_pos = 0
            s_stag = 0
            for lo in range(0, n, chunk):
                sl = m[lo:lo + chunk]
                if int(torch.count_nonzero(sl)) == 0:
                    continue
                pos = torch.arange(lo, lo + sl.numel(), device=device,
                                   dtype=torch.int64)
                sel = sl.to(torch.int64)
                s_pos += int((sel * pos).sum().item())
                # Staggered weight: bounded by 9973*1009*|selected| ~ 9e15,
                # far below int64 range even at full selection.
                s_stag += int((sel * (pos % 9973) * (pos % 1009)).sum().item())
            sums[2 * i] = s_pos
            sums[2 * i + 1] = s_stag
        lo_t, hi_t = sums.clone(), sums.clone()
        torch.distributed.all_reduce(lo_t, op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(hi_t, op=torch.distributed.ReduceOp.MAX, group=group)
        if not torch.equal(lo_t, hi_t):
            raise SGAEngineError(
                "SGA support checksum mismatch across ranks (dual-value "
                f"MIN {lo_t.tolist()} != MAX {hi_t.tolist()}); masks diverged."
            )

    def _check_norm_spread(self, tensors):
        """Layer 3 (prevention): scoring-input norm spread across ranks.

        The full allreduce should leave every rank bitwise-near-identical;
        (MAX-MIN)/MAX <= 1e-6 tolerates only floating-point reduction-order
        jitter. All ranks raise BEFORE scoring on violation.
        """
        group = self._dp_cp_group()
        if group is None or not tensors:
            return
        device = tensors[0].device
        norms = torch.stack([t.float().pow(2).sum() for t in tensors])
        lo, hi = norms.clone(), norms.clone()
        torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX, group=group)
        spread = ((hi - lo) / hi.clamp(min=torch.finfo(torch.float32).tiny)).max().item()
        if spread > 1.0e-6:
            raise SGAEngineError(
                f"SGA scoring-input norm spread across ranks {spread:.3e} > 1e-6; "
                "the upstream full sync is inconsistent — refusing to score."
            )

    # ------------------------------------------------------------ reset paths

    def _full_clear(self, model_chunks):
        """Failed step: clear everything, leave no residual."""
        for bs in self._bucket_states:
            bs.grad_data.zero_()
            for grad in bs.extra_main_grads:
                grad.zero_()
        self._compact_values = {}
        self._compact_values_step = None
        self._reset_host_state(model_chunks)

    def _reset_host_state(self, model_chunks):
        """Replicate zero_grad_buffer's non-buffer side effects (the static
        guard at the fork's train_step site skipped the whole call)."""
        for chunk in model_chunks:
            for group in chunk.bucket_groups + chunk.expert_parallel_bucket_groups:
                group.reset()
            for param in getattr(chunk, "params_with_grad", []):
                param.grad_added_to_main_grad = False

    def _activate_support(self, support: Support):
        self._active_support = support
        # The support is immutable within a cycle by design: derive the host
        # cardinalities ONCE here so the per-step guards and logging never
        # re-scan device masks (each would add GPU->host syncs on the hot path).
        for bucket_support in support.buckets:
            bucket_support.num_selected()
        support.total_selected
        # Dual representation, resident per the backend's declared input
        # form (the config names the backend, which IS the manual
        # override): an idx-consuming backend gets its transport form
        # compacted and cached here, outside the collective path, instead
        # of paying the chunked derivation inside the first sparse sync.
        # The form is layout-driven (er/eb vs bb — taxonomy:
        # sga.plugins.resolve_layout). Shadow gates this off: grad sync
        # never goes sparse there, so no consumer exists and the compacted
        # pool would be pure resident waste (~idx-sized per bucket).
        if self.backend.consumes == "idx" and not self._shadow:
            block_form = getattr(self.backend, "block_form", "block_idx")
            for bucket_support in support.buckets:
                if self._backend_layout != "bb":
                    bucket_support.idx
                elif block_form == "block_desc":
                    # BOTH block forms: the descriptor carries the wire
                    # form, but the pre-scale path (factor != 1) gathers
                    # by the plain heads — so both are materialized here,
                    # outside the first sparse sync.
                    bucket_support.block_desc
                    bucket_support.block_idx
                else:
                    bucket_support.block_idx
        for bs, bucket_support in zip(self._bucket_states, support.buckets):
            bs.active_mask_u8 = bucket_support.mask_u8
        # The freshly written pool slot becomes the active one; the next
        # reselection writes into the other slot (see _assemble_bucket_masks).
        self._mask_turn = 1 - self._mask_turn

    # --------------------------------------------------------------- logging

    def _log_step(self, step, was_full, reselected, grad_norm):
        ratio = -1.0
        if self._active_support is not None and self._active_support.total_elements:
            ratio = self._active_support.total_selected / self._active_support.total_elements
        extra = ""
        if hasattr(self.selection, "last_threshold") and self.selection.last_threshold:
            extra = f" sel_thresh={self.selection.last_threshold:.3e}"
        line = (
            f"[SGA] step={step} phase={'full' if was_full else 'sparse'} "
            f"ratio={ratio:.4f}{extra} grad_norm={grad_norm:.3f}"
        )
        if self._rank == 0:
            print(line, flush=True)
            if self._metrics_file:
                with open(self._metrics_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "phase": "full" if was_full else "sparse",
                        "ratio": ratio,
                        "grad_norm": grad_norm,
                        "reselected": reselected,
                    }) + "\n")

    # ------------------------------------------------------------- group lookups

    def _dp_cp_group(self):
        """DP-CP group, or None ONLY in supported degraded states (no
        distributed init; megatron genuinely absent). These groups gate the
        correctness machinery (deadlock guard, dual checksums, norm spread,
        the DP-bitwise fix) — a failing parallel_state getter means broken
        parallel state and must propagate, not silently disable the layers.
        """
        if not torch.distributed.is_initialized():
            return None
        try:
            from megatron.core import parallel_state
        except ImportError:
            return None  # megatron absent (CPU unit env), nothing to degrade
        return parallel_state.get_data_parallel_group(with_context_parallel=True)

    def _tp_group(self):
        if not torch.distributed.is_initialized():
            return None
        try:
            from megatron.core import parallel_state
        except ImportError:
            return None  # megatron absent (CPU unit env)
        return parallel_state.get_tensor_model_parallel_group()

    def _stats_group(self, mp_opt):
        getter = getattr(mp_opt, "get_grad_stats_parallel_group", None)
        if getter is not None:
            stats_group = getter()
            if isinstance(stats_group, (list, tuple)):
                assert len(stats_group) == 1
                stats_group = stats_group[0]
            return stats_group
        return self._tp_group()

    def _log_num_zeros_flag(self) -> bool:
        """The fork's log_num_zeros_in_grad flag (the pre-optimizer zero count is a
        frozen ordering-contract measurement point). False ONLY in supported
        non-running states — no distributed init (unit/CPU envs have no
        global args to consult) or megatron genuinely absent. Inside a live
        run a broken args lookup must propagate instead of silently turning
        the metric off."""
        if self._log_num_zeros is None:
            try:
                from megatron.training import get_args
            except ImportError:
                self._log_num_zeros = False
                return self._log_num_zeros
            if torch.distributed.is_initialized():
                self._log_num_zeros = bool(get_args().log_num_zeros_in_grad)
            else:
                self._log_num_zeros = False
        return self._log_num_zeros
