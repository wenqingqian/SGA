#!/usr/bin/env python3
"""Unit tests for the SGAEngine core (CPU, fakes, mocked collectives).

Covers (acceptance + expert Q7 additions):
- CycleManager phase semantics for the origin pattern (period 1 + fsi 10) and
  the radius pattern (bootstrap region + period-8 boundaries, boundaries
  imply full);
- support lifecycle: bootstrap guard, force steps never reselect, the
  grad-norm bypass forces the next step full;
- payload-scaling residual preservation (the regression test for the bug the
  old fork dodged via average_in_collective=True): with a SUM-replica mocked
  collective, non-selected coordinates are BITWISE unchanged across a sparse
  sync and selected ones get exactly sum*scale;
- negative tests for the three consistency layers (count mismatch refuses,
  dual-checksum divergence refuses, norm-spread violation raises pre-score);
- a fake end-to-end optimizer_step flow (origin and radius plugin combos)
  exercising the frozen ordering contract: prepare -> clip -> EF -> selection
  -> old-support reset -> switch.

Run: python tests/test_engine_unit.py  (or pytest tests/test_engine_unit.py)
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.backends.indexed_sparse import IndexedSparseBackend
from sga.backends.masked_pipelined import MaskedPipelinedBackend
from sga.engine import BucketState, CycleManager, SGAEngine, SGAEngineError
from sga.plugins import build_ef, build_score, build_selection
from sga.support import BucketSupport, Support

torch.manual_seed(0)


# ------------------------------------------------------------------- cycle

def test_cycle_origin_pattern():
    cm = CycleManager(start_iter=0, period=1, full_sync_interval=10,
                      boundary_implies_full=False)
    phases = ["full" if cm.is_full(s) else "sparse" for s in range(1, 16)]
    # Pure cycle semantics (the engine ORs in the bootstrap no-support guard):
    # origin EF does not imply full at boundaries; fsi=10 -> step 10 full.
    assert phases == ["sparse"] * 9 + ["full"] + ["sparse"] * 5, phases
    assert all(cm.reselect_boundary(s) for s in range(1, 16))  # T=1


def test_cycle_radius_pattern():
    cm = CycleManager(start_iter=20, period=8, full_sync_interval=0,
                      boundary_implies_full=True)
    fulls = [s for s in range(1, 61) if cm.is_full(s)]
    assert [s for s in fulls if s <= 20] == list(range(1, 21))  # bootstrap
    assert [s for s in fulls if s > 20] == [28, 36, 44, 52, 60]
    assert [s for s in range(1, 61) if cm.reselect_boundary(s)] == \
        [28, 36, 44, 52, 60]


# ------------------------------------------------- payload scaling residual

class _ReplicaWorld:
    """Mocked allreduce across `world` replica buffers (identity at 1).

    Equal replicas make every op predictable: SUM == world * local, and
    MAX/MIN == the local value unchanged. SUM on uint8 honors torch wrap
    semantics, so world=256 over all-ones wraps to 0 — exactly the mod-256
    hazard the mask OR-merges dodge by using MAX.
    """

    def __init__(self, world=2):
        self.world = world
        self.saved = torch.distributed.all_reduce

    def __enter__(self):
        torch.distributed.all_reduce = self._allreduce
        return self

    def __exit__(self, *a):
        torch.distributed.all_reduce = self.saved

    def _allreduce(self, tensor, op=None, group=None, async_op=False):
        if op is torch.distributed.ReduceOp.SUM:
            tensor.mul_(self.world)  # equal replicas: SUM == world * local
        # MAX/MIN over equal replicas: tensor already holds the result
        return SimpleNamespace(wait=lambda: None)


def test_payload_scaling_preserves_residual_bitwise():
    n = 4096
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    support = BucketSupport(mask)
    grad0 = torch.randn(n)
    grad = grad0.clone()
    factor = 0.125  # power of two -> scaling is exact, comparison bitwise

    backend = MaskedPipelinedBackend({"num_slices": 2})
    with _ReplicaWorld(world=4):
        backend.allreduce_sparse(grad, support, None, factor,
                                 torch.distributed.ReduceOp.SUM, key=0)
    active = mask > 0
    # residual coords: BITWISE unchanged (the whole point of payload scaling)
    assert torch.equal(grad[~active], grad0[~active])
    # selected coords: exactly sum-over-replicas * factor (2-pow exact)
    assert torch.equal(grad[active], grad0[active] * 4 * factor)


# ------------------------------------------------ consistency negative tests

class _MutatingAllReduce:
    """Simulates rank divergence: mutates the tensor during the collective so
    MIN != MAX (single-process stand-in for cross-rank disagreement)."""

    _SHIFT = {torch.distributed.ReduceOp.MIN: 1,
              torch.distributed.ReduceOp.MAX: 5}

    def __init__(self, mutate=None):
        self.mutate = mutate
        self.saved = torch.distributed.all_reduce

    def __enter__(self):
        torch.distributed.all_reduce = self._ar
        return self

    def __exit__(self, *a):
        torch.distributed.all_reduce = self.saved

    def _ar(self, tensor, op=None, group=None, async_op=False):
        if self.mutate is not None:
            self.mutate(tensor)
        else:
            tensor.add_(self._SHIFT.get(op, 1.0))
        return SimpleNamespace(wait=lambda: None)


def _engine_for_consistency():
    eng = object.__new__(SGAEngine)
    eng._dp_cp_group = lambda: "fake-group"
    eng._bucket_states = [object.__new__(BucketState)]
    eng._bucket_states[0].grad_data = torch.zeros(128)
    eng._active_support = Support(
        [BucketSupport((torch.rand(128) < 0.5).to(torch.uint8))]
    )
    return eng


def test_count_consistency_refuses_mismatch():
    eng = _engine_for_consistency()
    with _MutatingAllReduce():
        try:
            eng._check_counts()
            raise AssertionError("mismatched counts were not refused")
        except SGAEngineError:
            pass


def test_dual_checksum_refuses_divergence():
    eng = _engine_for_consistency()
    masks = [eng._active_support.buckets[0].mask_u8.clone()]
    with _MutatingAllReduce():
        try:
            eng._verify_support_consistency(masks)
            raise AssertionError("diverged masks were not refused")
        except SGAEngineError:
            pass


def test_norm_spread_refuses_violation():
    eng = _engine_for_consistency()
    with _MutatingAllReduce():
        try:
            eng._check_norm_spread([torch.randn(64)])
            raise AssertionError("norm spread violation did not raise")
        except SGAEngineError:
            pass


def test_norm_spread_passes_when_equal():
    eng = _engine_for_consistency()
    with _MutatingAllReduce(mutate=lambda t: None):  # all ranks identical
        eng._check_norm_spread([torch.randn(64)])  # must not raise


# ------------------------------------------------------ fake end-to-end flow

class _FakeBucket:
    def __init__(self, params, spans, numel, bucket_id=0):
        # Deliberately REVERSED vs the optimizer's param order: real megatron
        # buckets reorder params, and the engine must not assume the two
        # orders coincide (found by the first real radius run).
        self.params_list = list(reversed(params))
        self.param_to_index = {p: spans[p] for p in self.params_list}
        self.grad_data = torch.zeros(numel)
        self.gradient_scaling_factor = 1.0
        self.bucket_id = bucket_id


class _FakeGroup:
    def __init__(self, bucket):
        self.buckets = [bucket]
        self.data_parallel_group = None
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _FakeChunk:
    def __init__(self, n_params=2, numel_per=64):
        self.params = [torch.nn.Parameter(torch.randn(4, 16)) for _ in range(n_params)]
        spans, off = {}, 0
        for p in self.params:
            spans[p] = (off, off + p.numel())
            off += p.numel()
        self.bucket = _FakeBucket(self.params, spans, off)
        self.group = _FakeGroup(self.bucket)
        self.bucket_groups = [self.group]
        self.expert_parallel_bucket_groups = []
        self.buffers = [SimpleNamespace(buckets=[self.bucket], extra_main_grads=[])]
        self.expert_parallel_buffers = []
        self.module = SimpleNamespace(
            named_parameters=lambda: [(f"p{i}", p) for i, p in enumerate(self.params)]
        )
        self.ddp_config = SimpleNamespace(average_in_collective=False)
        self.params_with_grad = list(self.params)
        self.finish_grad_sync_calls = 0
        # main_grad aliases the bucket buffer (megatron fp32 main_grad world)
        for p, (start, end) in spans.items():
            p.main_grad = self.bucket.grad_data[start:end].view_as(p)

    def finish_grad_sync(self, force_all_reduce=False):
        self.finish_grad_sync_calls += 1


class _FakeMPOpt:
    """Float16OptimizerWithFloat16Params-shaped fake (CPU, torch Adam)."""

    def __init__(self, chunk):
        self.config = SimpleNamespace(clip_grad=1.0)
        self.float16_groups = [list(chunk.params)]
        self.main_params = [p.detach().clone().float() for p in chunk.params]
        for mp, p in zip(self.main_params, chunk.params):
            p.main_param = mp
        self.fp32_from_float16_groups = [self.main_params]
        self.optimizer = torch.optim.Adam(self.main_params, lr=1e-3,
                                          betas=(0.9, 0.95), eps=1e-8,
                                          weight_decay=1e-2)
        self.copy_calls = 0

    def prepare_grads(self):
        for mp, p in zip(self.main_params, self.float16_groups[0]):
            mp.grad = p.main_grad.float()  # fp32: alias of the bucket view
        return False  # megatron contract: True means found_inf (BAD)

    def _copy_main_params_to_model_params(self):
        self.copy_calls += 1
        for mp, p in zip(self.main_params, self.float16_groups[0]):
            with torch.no_grad():
                p.copy_(mp)


_ORIGIN = {
    "score": "abs_g",
    "selection": "threshold",
    "selection_params": {"threshold": 0.5, "target_ratio_low": 0.2,
                         "target_ratio_high": 0.9},
    "ef": "masked_adamw",
    "ef_params": {"grad_acc_count": True, "freeze_inactive": False},
    "cycle": {"start_iter": 0, "period": 1, "full_sync_interval": 10},
    "backend": MaskedPipelinedBackend({"num_slices": 2}),
}

_RADIUS = {
    "score": "update_ratio",
    "selection": "circular_topk",
    "selection_params": {"ratio_percent": 30.0},
    "ef": "radius_opt",
    "ef_params": {"residual_mean": False},
    "cycle": {"start_iter": 2, "period": 3, "full_sync_interval": 0},
    "backend": IndexedSparseBackend({}),
}


def _build_engine(combo):
    score = build_score(combo["score"], {"name": combo["score"]})
    selection = build_selection(combo["selection"], dict(combo["selection_params"]),
                                score=score)
    ef = build_ef(combo["ef"], dict(combo["ef_params"]))
    eng = SGAEngine({"cycle": combo["cycle"]}, score, selection, ef, combo["backend"])
    eng._dp_cp_group = lambda: None  # single-process: skip cross-rank layers
    return eng


def _run_flow(combo, steps):
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(combo)
    eng.attach(chunk)
    log = []
    with _ReplicaWorld(world=1):  # identity collectives for the backends
        for step in range(1, steps + 1):
            # emulate backward accumulation into the bucket buffer
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            was_full = eng._step_full
            ok, grad_norm, nz = eng.optimizer_step(mp, [chunk], step)
            assert ok is True and grad_norm is not None
            log.append((step, was_full, eng._active_support is not None))
    return eng, chunk, mp, log


def test_origin_flow_phases_and_support():
    eng, chunk, mp, log = _run_flow(_ORIGIN, steps=12)
    # Step 1 full (bootstrap no-support guard) selects support S1 at its end;
    # steps 2-9 sparse; step 10 full via full_sync_interval; 11-12 sparse.
    fulls = [s for s, was_full, _ in log if was_full]
    assert fulls == [1, 10], fulls
    assert all(has for _, _, has in log)  # support exists from step 1's end on
    # selective reset left a genuinely partial mask
    mask = eng._bucket_states[0].active_mask_u8
    assert 0 < int(mask.sum()) < mask.numel()
    # host-state replication ran (zero_grad_buffer's other side effects)
    assert chunk.group.reset_calls == 12
    # full steps DELEGATE to the real dense path (bitwise-parity decision):
    # exactly one finish_grad_sync call per full step, none per sparse step.
    assert chunk.finish_grad_sync_calls == 2


def test_radius_flow_boundaries_and_materialization():
    eng, chunk, mp, log = _run_flow(_RADIUS, steps=9)
    # start=2, T=3, boundaries imply full. Steps 1-2 are the start_iter
    # bootstrap; steps 3-4 have no support yet (first boundary is at 5), so
    # the engine's no-support guard keeps them dense; 5 and 8 are boundaries
    # (full + reselect); 6, 7, 9 are sparse with the persisted support.
    fulls = [s for s, was_full, _ in log if was_full]
    assert fulls == [1, 2, 3, 4, 5, 8], fulls
    for s, was_full, has_support in log:
        if not was_full:
            assert has_support, f"sparse step {s} without a support"
    # strict-ratio selection: measured ratio within rounding of the config
    support = eng._active_support
    ratio = support.total_selected / support.total_elements
    k = eng.selection.k_for(support.total_elements)
    assert ratio == k / support.total_elements
    assert 0 < ratio < 1.0


def test_forced_step_never_reselects():
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_ORIGIN)
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        # step 1: bootstrap full, boundary (T=1) -> reselect S1
        eng.grad_sync(chunk, force_all_reduce=False)
        eng.optimizer_step(mp, [chunk], 1)
        S1 = eng._active_support
        # step 2 forced (wgrad-saving style): dense, cycle advances, NO switch
        eng.grad_sync(chunk, force_all_reduce=True)
        eng.optimizer_step(mp, [chunk], 2)
        assert eng._active_support is S1


def test_failed_step_fully_clears_and_repeats_accounting():
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_ORIGIN)
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        # step 1: bootstrap full, reselect -> support S1
        eng.grad_sync(chunk, force_all_reduce=False)
        eng.optimizer_step(mp, [chunk], 1)
        S1 = eng._active_support
        assert S1 is not None
        # step 2 reports found_inf (prepare_grads True = BAD in megatron):
        # failed step -> (False, None, None), buffers fully cleared, support
        # untouched; _iter still tracks the fork iteration (assert contract).
        chunk.bucket.grad_data += torch.randn_like(chunk.bucket.grad_data)
        real_prepare = mp.prepare_grads
        mp.prepare_grads = lambda: True
        try:
            ok, gn, nz = eng.optimizer_step(mp, [chunk], 2)
        finally:
            mp.prepare_grads = real_prepare
        assert (ok, gn, nz) == (False, None, None)
        assert torch.equal(chunk.bucket.grad_data, torch.zeros_like(chunk.bucket.grad_data))
        assert eng._active_support is S1
        # the failed step consumed iteration 2 without reselecting (T=1 would
        # otherwise reselect): engine._iter advanced, support persisted.
        # step 3 (same cycle position as a fresh step 3) still works:
        chunk.bucket.grad_data += torch.randn_like(chunk.bucket.grad_data)
        eng.grad_sync(chunk, force_all_reduce=False)
        ok3, gn3, _ = eng.optimizer_step(mp, [chunk], 3)
        assert ok3 is True and gn3 is not None


def test_grad_norm_bypass_forces_next_full():
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    combo = dict(_ORIGIN, selection_params=dict(
        _ORIGIN["selection_params"], grad_norm_bypass=0.001))
    eng = _build_engine(combo)
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        chunk.bucket.grad_data += torch.randn_like(chunk.bucket.grad_data)
        eng.grad_sync(chunk, force_all_reduce=False)
        ok, grad_norm, _ = eng.optimizer_step(mp, [chunk], 1)
        assert grad_norm > 0.001  # random grads make the norm large
        assert eng._cycle.force_next is True
        eng.grad_sync(chunk, force_all_reduce=False)  # NOT force_all_reduce
        assert eng._step_full is True  # the bypass forced the dense path


# ------------------------------------------------- step-counter + resume C1/C2

def _accumulate_into_bucket(chunk):
    off = 0
    for p in chunk.params:
        n = p.numel()
        chunk.bucket.grad_data[off:off + n] += torch.randn(n)
        off += n


def test_rec_step_tracks_iteration_both_seed_branches():
    """rec.step must equal the fork iteration at EVERY step (the bias-
    correction contract), covering both sga_advance_step seeding branches:
    group['step'] already present (FusedAdam-style) and absent (torch-AdamW
    per-param state seeding)."""
    # Branch A: FusedAdam-style — the resumed checkpoint's group counter.
    chunk_a = _FakeChunk()
    mp_a = _FakeMPOpt(chunk_a)
    eng_a = _build_engine(_ORIGIN)
    eng_a.attach(chunk_a)
    mp_a.optimizer.param_groups[0]["step"] = 41
    with _ReplicaWorld(world=1):
        for it in range(42, 46):
            _accumulate_into_bucket(chunk_a)
            eng_a.grad_sync(chunk_a, force_all_reduce=False)
            ok, _, _ = eng_a.optimizer_step(mp_a, [chunk_a], it)
            assert ok
            assert {r.step for r in eng_a._records} == {it}

    # Branch B: torch AdamW — no group['step']; seeded from per-param state.
    chunk_b = _FakeChunk()
    mp_b = _FakeMPOpt(chunk_b)
    eng_b = _build_engine(_ORIGIN)
    eng_b.attach(chunk_b)
    with _ReplicaWorld(world=1):
        for it in range(1, 5):
            _accumulate_into_bucket(chunk_b)
            eng_b.grad_sync(chunk_b, force_all_reduce=False)
            ok, _, _ = eng_b.optimizer_step(mp_b, [chunk_b], it)
            assert ok
            assert {r.step for r in eng_b._records} == {it}


def test_resume_first_optimizer_step_seeds_iteration():
    """Fresh engine, first optimizer_step called with an ABSOLUTE resumed
    iteration N>1: seeds its accounting from the fork, succeeds, and rides
    the moments/step surviving the optimizer state_dict round trip."""
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_ORIGIN)
    eng.attach(chunk)
    K = 7  # fork's absolute iteration at resume
    state = {}
    for main_p in mp.main_params:
        state[main_p] = {
            "step": torch.tensor(float(K - 1)),          # round-tripped counter
            "exp_avg": torch.zeros_like(main_p),
            "exp_avg_sq": torch.zeros_like(main_p),
        }
    mp.optimizer.state.update(state)

    with _ReplicaWorld(world=1):
        _accumulate_into_bucket(chunk)
        assert eng.grad_sync(chunk, force_all_reduce=False) is True
        ok, grad_norm, _ = eng.optimizer_step(mp, [chunk], K)
    assert ok is True and grad_norm is not None
    # accounting now aligned with the fork: rec.step == K == internal step,
    # and the assert-contract holds on the following iteration too.
    assert {r.step for r in eng._records} == {K}
    assert eng._iter == K
    with _ReplicaWorld(world=1):
        _accumulate_into_bucket(chunk)
        eng.grad_sync(chunk, force_all_reduce=False)
        ok2, _, _ = eng.optimizer_step(mp, [chunk], K + 1)
    assert ok2 is True
    assert {r.step for r in eng._records} == {K + 1}


# ------------------------------------------------- num-zeros BEFORE-EF ordering

def test_num_zeros_logged_before_radius_overwrite():
    """With the num-zeros flag forced on, a radius full step must report the
    zero count of the PRE-EF gradient (radius then overwrites main grads
    with the materialized update)."""
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_RADIUS)
    eng.attach(chunk)
    eng._log_num_zeros = True  # fake the args-derived flag (unit env)
    with _ReplicaWorld(world=1):
        half = chunk.bucket.grad_data.numel() // 2
        chunk.bucket.grad_data += torch.randn_like(chunk.bucket.grad_data)
        chunk.bucket.grad_data[half:] = 0.0
        zeros_before = int((chunk.bucket.grad_data == 0).sum().item())
        assert zeros_before >= half
        eng.grad_sync(chunk, force_all_reduce=False)
        ok, _, nz = eng.optimizer_step(mp, [chunk], 1)
    assert ok is True
    assert nz == zeros_before


# ------------------------------------------------- dual-checksum collision case

def test_dual_checksum_catches_position_sum_collision():
    """Two DIFFERENT selected sets with EQUAL position sums: a single-sum
    checksum cannot tell them apart; the dual (staggered) checksum must."""
    eng = _engine_for_consistency()
    n = 4096

    def mk(positions):
        m = torch.zeros(n, dtype=torch.uint8)
        for p in positions:
            m[p] = 1
        return m

    A, B = mk([200, 300]), mk([150, 350])   # pos-sum 500 for BOTH;
    # staggered sums differ: 200^2+300^2=130000 vs 150^2+350^2=145000

    # Rank disagreement emulated: local rank computed B; another replica saw
    # A. The MIN reduction agrees everywhere, but MAX sees A's staggered sum.
    calls = {"n": 0}
    saved_ar = torch.distributed.all_reduce

    def _disagreeing_ar(tensor, op=None, group=None, async_op=False):
        calls["n"] += 1
        if calls["n"] % 2 == 0:  # the MAX reduction diverges at the stag slot
            tensor[1] = 130000.0
        return SimpleNamespace(wait=lambda: None)

    try:
        torch.distributed.all_reduce = _disagreeing_ar
        try:
            eng._verify_support_consistency([B])
            raise AssertionError("position-sum collision was not caught")
        except SGAEngineError as e:
            assert "checksum" in str(e), e
    finally:
        torch.distributed.all_reduce = saved_ar

    # Control: identical replicas pass cleanly.
    with _MutatingAllReduce(mutate=lambda t: None):
        eng._verify_support_consistency([B])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
