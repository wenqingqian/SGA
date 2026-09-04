#!/usr/bin/env python3
"""Unit tests for the SGAEngine core (CPU, fakes, mocked collectives).

Covers:
- CycleManager phase semantics: the boundary-full pattern (bootstrap region
  + period-8 boundaries — boundaries are structurally full), the decoupled
  full-sync grid (extra dense steps interleaved between boundaries, same
  origin, divisor-validated), the fold-window divisor and the calendar's
  constructor contracts;
- support lifecycle: bootstrap guard, force steps never reselect, the
  grad-norm bypass forces the next step full;
- payload-scaling residual preservation (the regression test for the bug the
  old fork dodged via average_in_collective=True): with a SUM-replica mocked
  collective, non-selected coordinates are BITWISE unchanged across a sparse
  sync and selected ones get exactly sum*scale;
- the decoupled fold-divisor pin: adamw's selective reset divides the
  surviving residual by ctx.fold_window (= the decoupled full_sync_interval,
  not the period) ahead of a mid-cycle full sync;
- negative tests for the three consistency layers (count mismatch refuses,
  dual-checksum divergence refuses, norm-spread violation raises pre-score);
- a fake end-to-end optimizer_step flow (the default 'update' and the named
  'abs_m' combos) exercising the frozen ordering contract: prepare -> clip ->
  optimizer template -> selection -> old-support reset -> switch.

Run: python tests/test_engine_unit.py  (or pytest tests/test_engine_unit.py)
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
from sga.backends.torch_pipelined_mask import TorchPipelinedMaskBackend
from sga.engine import BucketState, CycleManager, SGAEngine, SGAEngineError
from sga.plugins import build_optimizer, build_selection
from sga.plugins.optimizer import OptContext, ParamRecord
from sga.support import BucketSupport, Support

torch.manual_seed(0)


# ------------------------------------------------------------------- cycle

def test_cycle_boundary_full_pattern():
    # Boundaries are STRUCTURALLY full (reselection must read dense-averaged,
    # rank-identical scores) — no optimizer declaration involved anymore.
    cm = CycleManager(start_iter=20, period=8)
    fulls = [s for s in range(1, 61) if cm.is_full(s)]
    assert [s for s in fulls if s <= 20] == list(range(1, 21))  # bootstrap
    assert [s for s in fulls if s > 20] == [28, 36, 44, 52, 60]
    assert [s for s in range(1, 61) if cm.reselect_boundary(s)] == \
        [28, 36, 44, 52, 60]
    assert cm.fold_window == 8  # residual fold divisor = the period


def test_cycle_decoupled_grid():
    # period=8 interval=4 start_iter=20: mask rounds at 28, 36, ...; full
    # syncs at 24, 28, 32, ... — same origin, boundaries are a subset.
    cm = CycleManager(start_iter=20, period=8, decouple=True,
                      full_sync_interval=4)
    assert [s for s in range(21, 45) if cm.is_full(s)] == \
        [24, 28, 32, 36, 40, 44]
    # reselection ONLY at period boundaries — extra fulls never reselect
    assert [s for s in range(21, 45) if cm.reselect_boundary(s)] == \
        [28, 36, 44]
    assert cm.fold_window == 4  # divisor = the fold window, not the period
    assert cm.extra_full(24) and not cm.reselect_boundary(24)
    assert cm.extra_full(28) and cm.reselect_boundary(28)  # subset rule

    # NON-coincident origin: start_iter=22 gives 22 % 4 != 0 — the case that
    # distinguishes the ANCHORED grid (step - start_iter) % interval == 0
    # from the deleted unanchored step % interval == 0 (which would yield
    # [24, 28, 32, 36, 40, 44] here, and passes every coincident-origin
    # case above). Spacing stays uniformly 4 == fold_window.
    cm = CycleManager(start_iter=22, period=8, decouple=True,
                      full_sync_interval=4)
    assert [s for s in range(23, 46) if cm.is_full(s)] == \
        [26, 30, 34, 38, 42]
    # boundaries remain exactly the reselect subset of the dense grid
    assert [s for s in range(23, 46) if cm.reselect_boundary(s)] == \
        [30, 38]


def test_cycle_constructor_contracts():
    # decouple=True needs a strict divisor of the period (the calendar
    # guards are asserts; catching only AssertionError keeps a real
    # regression from being masked by this refusal loop)
    for bad_interval in (None, 0, 8, 16, 3):
        try:
            CycleManager(start_iter=0, period=8, decouple=True,
                         full_sync_interval=bad_interval)
        except AssertionError:
            continue  # expected: the calendar refuses this combination
        raise AssertionError(f"decoupled interval {bad_interval!r} accepted")
    # decouple=False: boundaries are structurally full, an explicit
    # interval has no meaning there and must fail loud
    try:
        CycleManager(start_iter=0, period=8, full_sync_interval=4)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-decoupled interval accepted")


# (the v3.1 warmup-anchor formula moved INTO the selection impl with the
# selection-owned warmup_sync_target hook; its pinning lives in
# tests/test_plugins_unit.py::test_warmup_sync_target_anchor)


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

    backend = TorchPipelinedMaskBackend({"num_slices": 2})
    with _ReplicaWorld(world=4):
        backend.allreduce_sparse(grad, support, None, factor,
                                 torch.distributed.ReduceOp.SUM, key=0)
    active = mask > 0
    # residual coords: BITWISE unchanged (the whole point of payload scaling)
    assert torch.equal(grad[~active], grad0[~active])
    # selected coords: exactly sum-over-replicas * factor (2-pow exact)
    assert torch.equal(grad[active], grad0[active] * 4 * factor)


# ------------------------------------------------- decoupled fold divisor

def test_adamw_reset_folds_residual_by_decoupled_fold_window():
    """CPU pin of the decoupled fold divisor: on the pre-boundary SPARSE step
    (next_step_is_full=True after a sparse sync, residual_mean set), the
    grad-correct fused pass folds the residual at the UNCOMMUNICATED
    positions by ctx.fold_window — the decoupled full_sync_interval (4), NOT
    the period (8) — and clears the communicated positions; reset()
    afterwards must leave the folded residual intact (moved both
    the clear and the fold INTO the kernel pass). A revert to the period
    divisor would leave every residual coordinate 2x too large on the
    decoupled calendar."""
    opt = build_optimizer("adamw", {"residual_mean": True})
    n = 256
    mask = torch.zeros(n, dtype=torch.uint8)
    mask[::4] = 1  # communicated this cycle
    orig = torch.randn(n)
    bs = BucketState(orig.clone(), None, 1.0, 0)
    bs.active_mask_u8 = mask
    w = torch.randn(n)
    w.grad = bs.grad_data          # the record's grad IS the bucket buffer
    rec = ParamRecord(model_param=w, main_param=w, bucket_state=bs,
                      start=0, end=n, lr=1e-3, beta1=0.9, beta2=0.95,
                      eps=1e-8, weight_decay=0.0, step=7, _state={})
    ctx = OptContext(records=[rec], phase="sparse", bucket_states=[bs],
                     fold_window=4, next_step_is_full=True)

    opt.sparse_step(ctx)
    active = mask > 0
    # communicated positions: zeroed in the fused pass (their residual was
    # consumed by the just-finished sparse sync)
    assert torch.equal(bs.grad_data[active],
                       torch.zeros(int(active.sum())))
    # residual positions: divided by exactly the fold window (4 — a power
    # of two, so the comparison is bitwise), not by the period (8)
    assert torch.equal(bs.grad_data[~active], orig[~active] / 4)
    assert not torch.equal(bs.grad_data[~active], orig[~active] / 8)

    # reset on the sparse step must NOT touch the bucket (the folded
    # residual is the accumulation base the boundary full sync consumes);
    # on a full step it zeroes everything
    folded = bs.grad_data.clone()
    opt.reset(ctx, was_full=False)
    assert torch.equal(bs.grad_data, folded)
    opt.reset(ctx, was_full=True)
    assert torch.all(bs.grad_data == 0)


def test_torch_pipelined_index_modes_and_mechanics():
    """The unified idx backend: both transport modes (plain default vs
    high-priority twin) land bitwise-identical gradients (same element
    sets, payload scaling on selected only), the default mode uses the
    legacy bucket-midpoint split, grow-only buffers hold, prepare() is
    twin-mode-gated, and validate_runtime only constrains twin mode."""
    plain = TorchPipelinedIndexBackend({})          # hiprio=False, S=2
    twin = TorchPipelinedIndexBackend({"high_priority_group": True,
                                       "slices": 3})
    assert plain.decl.consumes == "idx"  # engine activates by this attribute
    assert plain.decl.name == "torch_pipelined_index"

    for world, k_shape in ((1, None), (4, None), (4, 7)):  # incl. tiny uneven K
        torch.manual_seed(21 + world)
        n = 4096
        mask = (torch.rand(n) < 0.25).to(torch.uint8)
        if k_shape is not None:  # force exactly k selected (uneven 3-way split)
            mask = torch.zeros(n, dtype=torch.uint8)
            mask[torch.randperm(n)[:k_shape]] = 1
        support = BucketSupport(mask)
        factor = 0.125

        g_ref = torch.randn(n)
        g_new = g_ref.clone()
        with _ReplicaWorld(world=world):
            plain.allreduce_sparse(g_ref, support, None, factor,
                                   torch.distributed.ReduceOp.SUM, key=0)
            twin.allreduce_sparse(g_new, support, None, factor,
                                  torch.distributed.ReduceOp.SUM, key=0)
        assert torch.equal(g_new, g_ref), (world, k_shape)

    # default mode splits at the legacy bucket midpoint; other configs
    # split the selected count evenly
    support = BucketSupport((torch.rand(4096) < 0.3).to(torch.uint8))
    k = support.idx.numel()
    assert plain._bounds(support, k) == [(0, support.split), (support.split, k)]
    assert twin._bounds(support, k) == [(i * k // 3, (i + 1) * k // 3)
                                        for i in range(3)]
    assert TorchPipelinedIndexBackend({"slices": 3})._bounds(support, k) == \
        [(i * k // 3, (i + 1) * k // 3) for i in range(3)]

    # grow-only discipline: a smaller second cycle keeps the same capacity
    be = TorchPipelinedIndexBackend({})
    with _ReplicaWorld(world=1):
        be.allreduce_sparse(torch.randn(1000), BucketSupport(
            torch.ones(1000, dtype=torch.uint8)), None, 1.0,
            torch.distributed.ReduceOp.SUM, key=3)
        cap = be._gather_bufs[3].numel()
        be.allreduce_sparse(torch.randn(1000), BucketSupport(
            (torch.rand(1000) < 0.01).to(torch.uint8)), None, 1.0,
            torch.distributed.ReduceOp.SUM, key=3)
    assert be._gather_bufs[3].numel() == cap

    # prepare() is twin-mode-gated and, in twin mode, where the ONLY group
    # creation happens (never lazily inside allreduce_sparse); idempotent;
    # without a live NCCL world the plain group is kept.
    sentinel = object()
    plain.prepare(sentinel)                 # plain mode: no-op
    assert plain._twins == {}
    assert twin._twins == {}
    twin.prepare(None)                      # missing group: no-op
    twin.prepare(sentinel)                  # no NCCL init here: keep plain
    twin.prepare(sentinel)                  # idempotent
    assert len(twin._twins) == 1 and twin._twins[sentinel] is sentinel
    assert twin._group_for(sentinel) is sentinel
    assert twin._group_for(None) is None
    try:
        TorchPipelinedIndexBackend({"slices": 0})
        raise AssertionError("slices=0 accepted")
    except ValueError:
        pass

    # validate_runtime constrains TWIN mode only: single pipeline phase +
    # exactly one DP-CP group covering the world (multi-DP-CP / tensor /
    # context / expert layouts are refused at config time, not ordered at
    # new_group); the plain mode imposes nothing beyond the shared v1 set.
    ns = SimpleNamespace
    ok = ns(pipeline_model_parallel_size=1, tensor_model_parallel_size=1,
            context_parallel_size=1, expert_model_parallel_size=1)
    twin.validate_runtime(ok)
    for bad in (
        {"pipeline_model_parallel_size": 2},
        {"tensor_model_parallel_size": 2},
        {"context_parallel_size": 2},
        {"expert_model_parallel_size": 4},
    ):
        try:
            twin.validate_runtime(ns(**{**ok.__dict__, **bad}))
            raise AssertionError(f"bad topology accepted: {bad}")
        except ValueError:
            pass
    plain.validate_runtime(ns(**{**ok.__dict__, "pipeline_model_parallel_size": 8,
                                 "tensor_model_parallel_size": 4}))

    # (resume fail-closed coverage lives in its own test function below so
    # a failure names that concern, not this mechanics mega-test)
    print("ok: torch_pipelined_index modes (parity, midpoint/even bounds, "
          "grow-only, gated prepare/validate_runtime)")


def test_resume_fail_closed_refusal_and_hermetic_override():
    """Resume fail-closed conventions: a load dir WITH a checkpoint tracker
    is refused (twin-PG residency does not fit a resume run's steady
    state); cold-start conventions (no dir / dir without tracker) pass;
    the escape-hatch env overrides. The env var is scrubbed before the
    refuse assertion and restored afterwards, so the refusal case stays
    hermetic even when it is pre-exported in the runner's shell (its
    documented use — 80GB hardware — is exactly exporting it before
    running the suite)."""
    import tempfile

    import sga.backends.torch_pipelined_index as tpi
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend

    ns = SimpleNamespace
    base = dict(pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1)
    be = TorchPipelinedIndexBackend({"high_priority_group": True})
    be_plain = TorchPipelinedIndexBackend({})
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "checkpoints")
        os.makedirs(ckpt)
        # no dir / dir without tracker: cold start, pass
        be.validate_runtime(ns(**base, load=os.path.join(td, "missing")))
        be.validate_runtime(ns(**base, load=ckpt))
        # tracker present: resume, refuse (message names the alternative
        # AND the documented escape hatch)
        open(os.path.join(ckpt, "latest_checkpointed_iteration.txt"), "w").write("60")
        # plain mode is the resume path: no gate applies
        be_plain.validate_runtime(ns(**base, load=ckpt))
        # hermetic: scrub any pre-exported escape hatch so the refusal
        # below tests the default behavior; restore whatever was set.
        prev = os.environ.pop(tpi._RESUME_OVERRIDE_ENV, None)
        try:
            try:
                be.validate_runtime(ns(**base, load=ckpt))
                raise AssertionError("resume accepted")
            except ValueError as e:
                assert "high_priority_group" in str(e)
                assert tpi._RESUME_OVERRIDE_ENV in str(e)
            os.environ[tpi._RESUME_OVERRIDE_ENV] = "1"
            be.validate_runtime(ns(**base, load=ckpt))  # override passes
        finally:
            if prev is None:
                os.environ.pop(tpi._RESUME_OVERRIDE_ENV, None)
            else:
                os.environ[tpi._RESUME_OVERRIDE_ENV] = prev
    print("ok: resume fail-closed (cold start passes, tracker refused, "
          "override honored, env restored)")


def test_nccl_pipelined_index_decl_and_loud_gates():
    """Everything reachable without a live raw NCCL comm: the decl
    vocabulary, param gates, uniform bounds, and the loud refusals. The
    real comm path (prepare / uid exchange / ctypes launch) is covered by
    the 2-rank CUDA dist suite with SGA_NCCL_SAR_LIB set."""
    from sga.backends.nccl_pipelined_index import (
        DECL, NcclPipelinedIndexBackend)
    ns = SimpleNamespace
    base = dict(pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1)

    be = NcclPipelinedIndexBackend({})
    assert be.decl is DECL and DECL.name == "nccl_pipelined_index"
    assert be.decl.consumes == "idx"  # engine routes by this attribute
    # single launch per sync (each extra slice = one more full kernel
    # invocation): no slicing vocabulary remains (the catalog row is empty),
    # and a stale config carrying it fails loud
    from sga.config.validation import _AXIS_PARAMS

    assert _AXIS_PARAMS["backend"]["nccl_pipelined_index"] == ()
    try:
        NcclPipelinedIndexBackend({"slices": 3})
        raise AssertionError("removed `slices` param accepted")
    except ValueError:
        pass

    # _scale_buf growth discipline (CPU-exercisable: no dist, no lib):
    # grow-only on shrink with 4M rounding; a dtype-change realloc (the
    # same branch as a device change) preserves the grown capacity
    sb = NcclPipelinedIndexBackend({})
    sb._scale_buf(0, 1000, torch.float32, torch.device("cpu"))
    cap = sb._scale_bufs[0].numel()
    assert cap == 4 << 20
    sb._scale_buf(0, 10, torch.float32, torch.device("cpu"))
    assert sb._scale_bufs[0].numel() == cap  # shrink keeps capacity
    sb._scale_buf(0, 2 * cap, torch.float32, torch.device("cpu"))
    assert sb._scale_bufs[0].numel() == 2 * cap  # growth rounds to 4M
    sb._scale_buf(0, 10, torch.float64, torch.device("cpu"))
    assert sb._scale_bufs[0].numel() == 2 * cap  # realloc preserves it

    g = torch.randn(64)
    sup = BucketSupport((torch.rand(64) < 0.5).to(torch.uint8))
    # pinned dtype/op vocabulary refuses before anything else
    try:
        be.allreduce_sparse(g.double(), sup, None, 1.0,
                            torch.distributed.ReduceOp.SUM, key=0)
        raise AssertionError("fp64 accepted")
    except ValueError as e:
        assert "fp32" in str(e), e
    try:
        be.allreduce_sparse(g, sup, None, 1.0,
                            torch.distributed.ReduceOp.MAX, key=0)
        raise AssertionError("MAX accepted")
    except ValueError as e:
        assert "SUM" in str(e), e
    # empty support: clean no-op, no comm needed
    be.allreduce_sparse(g, BucketSupport(torch.zeros(64, dtype=torch.uint8)),
                        None, 1.0, torch.distributed.ReduceOp.SUM, key=0)
    # non-empty support without a prepared comm: loud, not a silent fallback
    try:
        be.allreduce_sparse(g, sup, None, 1.0,
                            torch.distributed.ReduceOp.SUM, key=0)
        raise AssertionError("missing comm accepted")
    except RuntimeError as e:
        assert "prepare()" in str(e), e

    # validate_runtime: topology gates mirror the twin-mode contract
    be.validate_runtime(ns(**base))
    for bad in ({"pipeline_model_parallel_size": 2},
                {"tensor_model_parallel_size": 2},
                {"context_parallel_size": 2},
                {"expert_model_parallel_size": 4}):
        try:
            be.validate_runtime(ns(**bad))
            raise AssertionError(f"bad topology accepted: {bad}")
        except ValueError:
            pass
    print("ok: nccl_pipelined_index decl/params/bounds/loud gates")


def test_nccl_pipelined_index_resume_fail_closed():
    """Same resume conventions as the twin mode: a load dir WITH a tracker
    is refused (raw-comm residency unmeasured for resume, conservative
    refusal), cold start passes, the shared escape hatch overrides; env
    scrubbed before the refusal assertion and restored after."""
    import tempfile

    import sga.backends.nccl_pipelined_index as npi
    from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend

    ns = SimpleNamespace
    base = dict(pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1)
    be = NcclPipelinedIndexBackend({})
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "checkpoints")
        os.makedirs(ckpt)
        be.validate_runtime(ns(**base, load=os.path.join(td, "missing")))
        be.validate_runtime(ns(**base, load=ckpt))
        open(os.path.join(ckpt, "latest_checkpointed_iteration.txt"), "w").write("60")
        prev = os.environ.pop(npi._RESUME_OVERRIDE_ENV, None)
        try:
            try:
                be.validate_runtime(ns(**base, load=ckpt))
                raise AssertionError("resume accepted")
            except ValueError as e:
                assert npi._RESUME_OVERRIDE_ENV in str(e), e
            os.environ[npi._RESUME_OVERRIDE_ENV] = "1"
            be.validate_runtime(ns(**base, load=ckpt))  # override passes
        finally:
            if prev is None:
                os.environ.pop(npi._RESUME_OVERRIDE_ENV, None)
            else:
                os.environ[npi._RESUME_OVERRIDE_ENV] = prev
    print("ok: nccl_pipelined_index resume fail-closed (cold start passes, "
          "tracker refused, override honored, env restored)")


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
        # orders coincide (found by the first real run).
        self.params_list = list(reversed(params))
        self.param_to_index = {p: spans[p] for p in self.params_list}
        self.grad_data = torch.zeros(numel)
        self.gradient_scaling_factor = 1.0
        self.bucket_id = bucket_id


class _FakeGroup:
    def __init__(self, bucket, data_parallel_group=None):
        self.buckets = [bucket]
        self.data_parallel_group = data_parallel_group
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


# The default-score combo: 'update' materializes straight into the main grads
# over a mask-consuming backend.
_UPDATE = {
    "score": "update",
    "selection": "topk",
    "selection_params": {"method": "histogram", "ratio_percent": 30.0},
    "opt": "adamw",
    "opt_params": {"residual_mean": False},
    "cycle": {"start_iter": 0, "period": 3},
    "backend": TorchPipelinedMaskBackend({"num_slices": 2}),
}

# The named-score combo: abs_m materializes into the grad buffer with the
# apply fused in the same pass, over an idx-consuming backend.
_ABS_M = {
    "score": "abs_m",
    "selection": "topk",
    "selection_params": {"method": "histogram", "ratio_percent": 30.0},
    "opt": "adamw",
    "opt_params": {"residual_mean": False},
    "cycle": {"start_iter": 2, "period": 3},
    "backend": TorchPipelinedIndexBackend({}),
}


def _build_engine(combo):
    selection = build_selection(combo["selection"], dict(combo["selection_params"]))
    opt = build_optimizer(combo["opt"], dict(combo["opt_params"]),
                          emit_score=combo["score"])
    eng = SGAEngine({"cycle": combo["cycle"]}, combo["score"], selection, opt,
                    combo["backend"])
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
            _accumulate_into_bucket(chunk)
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            was_full = eng._step_full
            ok, grad_norm, nz = eng.optimizer_step(mp, [chunk], step)
            assert ok is True and grad_norm is not None
            log.append((step, was_full, eng._active_support is not None))
    return eng, chunk, mp, log


def test_update_flow_phases_and_support():
    eng, chunk, mp, log = _run_flow(_UPDATE, steps=12)
    # start=0, T=3, boundaries imply full (the adamw residual contract).
    # Steps 1-2 are full via the no-support guard (no boundary yet); step 3
    # is the first boundary (full + reselect); 4-5 sparse; 6/9/12 boundaries.
    fulls = [s for s, was_full, _ in log if was_full]
    assert fulls == [1, 2, 3, 6, 9, 12], fulls
    assert all(has for s, _, has in log if s >= 3), log  # support from step 3 on
    assert not any(has for s, _, has in log if s < 3), log
    # selective reset left a genuinely partial mask
    mask = eng._bucket_states[0].active_mask_u8
    assert 0 < int(mask.sum()) < mask.numel()
    # host-state replication ran (zero_grad_buffer's other side effects)
    assert chunk.group.reset_calls == 12
    # full steps DELEGATE to the real dense path (bitwise-parity decision):
    # exactly one finish_grad_sync call per full step, none per sparse step.
    assert chunk.finish_grad_sync_calls == 6


def test_abs_m_flow_boundaries_and_materialization():
    eng, chunk, mp, log = _run_flow(_ABS_M, steps=9)
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


def test_compact_emission_state_invariants():
    """Compact emission: pin the state contract at BOTH
    observation points of a sparse step. A (grad_sync return): the pack
    cleared the selected coords in-place, the values view is captured
    under the bucket key, the residual is bitwise untouched. B
    (optimizer_step return): selected coords STAY zero and the residual
    survives bitwise (residual_mean=False), the capture lives until the
    next step's first sync wipes it. The sparse-step grad_norm is pinned
    against an independent oracle over the captured values — the
    bucket-level reduction must happen ONCE per bucket (this fake has 2
    records on 1 bucket, so a per-record reduction inflates the norm by
    sqrt(2) and silently over-clips) — and num_zeros is pinned to the
    writeback view (pack-cleared coords re-entered through the values
    buffer). Plus the fail-closed guards: a writeback backend never
    populates the capture, and a declared-compact backend returning None
    is refused."""
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_ABS_M)
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        # bootstrap to the first sparse step: 1-5 full, 6 sparse (see
        # test_abs_m_flow_boundaries_and_materialization)
        for step in range(1, 6):
            _accumulate_into_bucket(chunk)
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            assert eng._compact_values == {}  # full steps never capture
            ok, _, _ = eng.optimizer_step(mp, [chunk], step)
            assert ok is True
        bs = eng._bucket_states[0]
        idx = eng._active_support.buckets[bs.support_index].idx.long()
        rest = torch.ones(bs.grad_data.numel(), dtype=torch.bool)
        rest[idx] = False
        _accumulate_into_bucket(chunk)
        pre = bs.grad_data.clone()
        assert eng.grad_sync(chunk, force_all_reduce=False) is True
        # observation point A: the compact communication result
        assert set(eng._compact_values) == {id(bs)}
        vals = eng._compact_values[id(bs)].clone()
        assert isinstance(vals, torch.Tensor)
        assert vals.numel() == idx.numel() > 0
        # gather with factor==1.0: a pure memory move, bitwise
        assert torch.equal(vals, pre[idx])
        # the pack cleared the communicated coords in grad_data
        assert torch.count_nonzero(bs.grad_data[idx]).item() == 0
        # the residual at unselected coords is bitwise untouched
        assert torch.equal(bs.grad_data[rest], pre[rest])
        # num-zeros measurement point on a sparse compact step: the
        # writeback view — raw post-pack zeros minus the communicated
        # coords, re-counted through the values buffer
        eng._log_num_zeros = True
        ok, grad_norm, nz = eng.optimizer_step(mp, [chunk], 6)
        assert ok is True and grad_norm is not None
        # independent norm oracle over the captured bucket values: one
        # contribution per bucket, exact against the engine's own fp32
        # reduction (a per-record double count fails this by ~sqrt(2))
        oracle_norm = float(vals.pow(2).sum().sqrt())
        assert abs(grad_norm - oracle_norm) <= 1e-5 * oracle_norm, (
            grad_norm, oracle_norm
        )
        assert nz == (int((pre[rest] == 0).sum().item())
                      + int((vals == 0).sum().item())), (nz,)
        # observation point B: the post-consumption end state
        assert torch.count_nonzero(bs.grad_data[idx]).item() == 0
        assert torch.equal(bs.grad_data[rest], pre[rest])
        # the capture survives optimizer_step; the NEXT first-sync wipes it
        assert id(bs) in eng._compact_values
        _accumulate_into_bucket(chunk)
        assert eng.grad_sync(chunk, force_all_reduce=False) is True
        assert eng._compact_values_step == 7

    # writeback emission (mask backend) keeps its None contract: sparse
    # steps over it never populate the capture
    chunk2 = _FakeChunk()
    mp2 = _FakeMPOpt(chunk2)
    eng2 = _build_engine(_UPDATE)  # start=0, T=3: sparse at steps 4-5
    eng2.attach(chunk2)
    with _ReplicaWorld(world=1):
        for step in range(1, 4):
            _accumulate_into_bucket(chunk2)
            assert eng2.grad_sync(chunk2, force_all_reduce=False) is True
            assert eng2.optimizer_step(mp2, [chunk2], step)[0] is True
        _accumulate_into_bucket(chunk2)
        assert eng2.grad_sync(chunk2, force_all_reduce=False) is True
        assert eng2._compact_values == {}
        assert eng2.optimizer_step(mp2, [chunk2], 4)[0] is True

    # declared compact but returned None: fail closed (a stale buffer must
    # not silently feed clip/grad_correct)
    class _NoneBackend(TorchPipelinedIndexBackend):
        def allreduce_sparse(self, *a, **k):
            return None

    chunk3 = _FakeChunk()
    mp3 = _FakeMPOpt(chunk3)
    eng3 = _build_engine({**_ABS_M, "backend": _NoneBackend({})})
    eng3.attach(chunk3)
    with _ReplicaWorld(world=1):
        for step in range(1, 6):
            _accumulate_into_bucket(chunk3)
            assert eng3.grad_sync(chunk3, force_all_reduce=False) is True
            assert eng3.optimizer_step(mp3, [chunk3], step)[0] is True
        _accumulate_into_bucket(chunk3)
        try:
            eng3.grad_sync(chunk3, force_all_reduce=False)
            raise AssertionError("None from compact backend accepted")
        except SGAEngineError as e:
            assert "values Tensor" in str(e), e


def test_reselect_boundary_orders_reset_before_activation():
    """Instrumented pin of the load-bearing support-switch ordering at a
    successful reselection boundary: optimizer full step -> model copy ->
    selection (reads the materialized score views) -> reset (zeroes the
    consumed grad buffers — captured here while the OLD support is still
    the active one on the buckets; an earlier round reset no longer reads the
    masks itself) -> activate(new support) -> host reset. The stub
    selection's consecutive masks are DISJOINT, so 'reset ran under the old
    support' is falsifiable: a refactor that activated the new support
    before reset would swap the bucket masks early and break the capture."""
    class _DisjointSelection:
        """Selection no. n masks every n-th-position parity: consecutive
        selections produce disjoint per-param masks."""

        params = {}

        def __init__(self, events):
            self._events = events
            self._calls = 0

        def select(self, ctx):
            masks = [(torch.arange(g.numel()) % 2 == self._calls % 2)
                     .to(torch.uint8) for g in ctx.grad_views]
            self._events.append(("select", [m.clone() for m in masks]))
            self._calls += 1
            return masks

    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    opt = build_optimizer(_UPDATE["opt"], dict(_UPDATE["opt_params"]),
                          emit_score=_UPDATE["score"])
    events = []
    eng = SGAEngine({"cycle": dict(_UPDATE["cycle"])}, _UPDATE["score"],
                    _DisjointSelection(events), opt, _UPDATE["backend"])
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)

    real_full = opt.full_step
    opt.full_step = lambda ctx: (events.append(("full_step", None)),
                                 real_full(ctx))[1]
    real_reset = opt.reset
    opt.reset = lambda ctx, was_full: (
        events.append(("reset", [bs.active_mask_u8.clone()
                                 if bs.active_mask_u8 is not None else None
                                 for bs in ctx.bucket_states])),
        real_reset(ctx, was_full=was_full))[1]
    real_copy = mp._copy_main_params_to_model_params
    mp._copy_main_params_to_model_params = lambda: (
        events.append(("copy", None)), real_copy())[1]
    real_activate = eng._activate_support
    eng._activate_support = lambda s: (events.append(("activate", None)),
                                       real_activate(s))[1]
    real_group_reset = chunk.group.reset
    chunk.group.reset = lambda: (events.append(("host_reset", None)),
                                 real_group_reset())[1]

    def run(step):
        _accumulate_into_bucket(chunk)
        eng.grad_sync(chunk, force_all_reduce=False)
        eng.optimizer_step(mp, [chunk], step)

    with _ReplicaWorld(world=1):
        for step in range(1, 6):  # _UPDATE cycle: reselections at 3 and 6
            run(step)
        old_masks = [bs.active_mask_u8.clone()
                     for bs in eng._bucket_states]  # the step-3 support
        assert all(m is not None for m in old_masks)
        run(6)  # second reselection boundary

    # boundary 1 (step 3): same cadence, and reset ran with NO support yet
    # (nothing to consume) before the first activation
    sel_idx = [i for i, e in enumerate(events) if e[0] == "select"]
    assert len(sel_idx) == 2
    i1, i2 = sel_idx
    assert [e[0] for e in events[i1 - 2:i1 + 4]] == \
        ["full_step", "copy", "select", "reset", "activate", "host_reset"]
    assert all(m is None for m in events[i1 + 1][1])

    # boundary 2 (step 6): the optimizer's reset consumed the OLD masks and
    # ran BEFORE the new support was activated
    assert [e[0] for e in events[i2 - 2:i2 + 4]] == \
        ["full_step", "copy", "select", "reset", "activate", "host_reset"]
    reset_masks = events[i2 + 1][1]
    assert len(reset_masks) == len(old_masks)
    for seen, old in zip(reset_masks, old_masks):
        assert torch.equal(seen, old)

    # the precondition that makes the pin falsifiable: the two supports are
    # genuinely disjoint, so 'reset saw the old mask' cannot pass by accident
    new_masks = [bs.active_mask_u8.clone() for bs in eng._bucket_states]
    for old, new in zip(old_masks, new_masks):
        assert int((old & new).sum()) == 0
        assert not torch.equal(old, new)


def test_forced_step_never_reselects():
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_UPDATE)
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        # steps 1-2 bootstrap dense; step 3 is a boundary -> reselect S1;
        # steps 4-5 ride S1 sparse
        for step in range(1, 6):
            _accumulate_into_bucket(chunk)
            eng.grad_sync(chunk, force_all_reduce=False)
            eng.optimizer_step(mp, [chunk], step)
        S1 = eng._active_support
        assert S1 is not None
        # step 6 IS a boundary (would reselect), but arrives FORCED
        # (wgrad-saving style): dense, cycle advances, NO switch
        _accumulate_into_bucket(chunk)
        eng.grad_sync(chunk, force_all_reduce=True)
        eng.optimizer_step(mp, [chunk], 6)
        assert eng._active_support is S1


def test_failed_step_fully_clears_and_repeats_accounting():
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_UPDATE)
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        # steps 1-5 run (S1 selected at the step-3 boundary)
        for step in range(1, 6):
            _accumulate_into_bucket(chunk)
            eng.grad_sync(chunk, force_all_reduce=False)
            eng.optimizer_step(mp, [chunk], step)
        S1 = eng._active_support
        assert S1 is not None
        # step 6 IS a boundary (would reselect) but reports found_inf
        # (prepare_grads True = BAD in megatron): failed step ->
        # (False, None, None), buffers fully cleared, support untouched;
        # _iter still tracks the fork iteration (assert contract).
        _accumulate_into_bucket(chunk)
        real_prepare = mp.prepare_grads
        mp.prepare_grads = lambda: True
        try:
            ok, gn, nz = eng.optimizer_step(mp, [chunk], 6)
        finally:
            mp.prepare_grads = real_prepare
        assert (ok, gn, nz) == (False, None, None)
        assert torch.equal(chunk.bucket.grad_data, torch.zeros_like(chunk.bucket.grad_data))
        assert eng._active_support is S1
        # the failed step consumed iteration 6 without reselecting (it was a
        # boundary and would otherwise have switched); step 7 rides S1 fine:
        _accumulate_into_bucket(chunk)
        eng.grad_sync(chunk, force_all_reduce=False)
        ok7, gn7, _ = eng.optimizer_step(mp, [chunk], 7)
        assert ok7 is True and gn7 is not None
        assert eng._active_support is S1


def test_grad_norm_bypass_forces_next_full():
    class _BypassSelection:
        """Stub exposing the optional wants_force_full hook the engine probes
        generically (plain topk does not implement it; the routing itself
        is the unit under test)."""

        params = {}

        def __init__(self):
            self.seen = []

        def select(self, ctx):
            return [(torch.ones(g.numel(), dtype=torch.uint8))
                    for g in ctx.grad_views]

        def wants_force_full(self, grad_norm):
            self.seen.append(grad_norm)
            return grad_norm > 0.001

    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    opt = build_optimizer(_UPDATE["opt"], dict(_UPDATE["opt_params"]),
                          emit_score=_UPDATE["score"])
    eng = SGAEngine({"cycle": dict(_UPDATE["cycle"])}, _UPDATE["score"],
                    _BypassSelection(), opt, _UPDATE["backend"])
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)
    with _ReplicaWorld(world=1):
        _accumulate_into_bucket(chunk)
        eng.grad_sync(chunk, force_all_reduce=False)
        ok, grad_norm, _ = eng.optimizer_step(mp, [chunk], 1)
        assert grad_norm > 0.001  # random grads make the norm large
        assert eng._cycle.force_next is True
        eng.grad_sync(chunk, force_all_reduce=False)  # NOT force_all_reduce
        assert eng._step_full is True  # the bypass forced the dense path


# ------------------------------------------------- generic optional hooks

class _HookSelection:
    """Stub selection exposing BOTH optional hooks the engine probes
    generically (same getattr discipline as adapt()/wants_force_full()).

    Masks alternate per call (even indices on the first call, odd on the
    second, ...): a support activated at one reselection is then
    distinguishable from what the NEXT call would select — all-ones masks
    would make the two indistinguishable.
    """

    params = {}

    def __init__(self):
        self.anchor_calls = []
        self.ctx = None
        self._calls = 0

    def k_for(self, n_total):
        return max(1, n_total // 2)

    def warmup_sync_target(self, start_iter, period):
        self.anchor_calls.append((start_iter, period))
        return 42

    def select(self, ctx):
        self.ctx = ctx
        masks = [(torch.arange(g.numel()) % 2 == self._calls % 2).to(torch.uint8)
                 for g in ctx.grad_views]
        self._calls += 1
        return masks


class _PrepareBackend(TorchPipelinedIndexBackend):
    def __init__(self):
        super().__init__({})
        self.prepared = []

    def prepare(self, comm_group, device=None):
        self.prepared.append((comm_group, device))


def test_generic_optional_hooks_fire():
    """The engine probes plugins generically: a selection-owned
    warmup_sync_target(start_iter, period) supplies ctx.warmup_target (the
    v3.1 anchor policy lives IN the selection), and a backend's optional
    prepare(comm_group, device) runs once per distinct DP group at attach
    time — before any training step, with the bucket grad-buffer device."""
    grp = object()
    chunk = _FakeChunk()
    chunk.group.data_parallel_group = grp

    backend = _PrepareBackend()
    selection = _HookSelection()
    opt = build_optimizer("adamw", {}, emit_score="abs_g")
    eng = SGAEngine(
        {"cycle": {"start_iter": 0, "period": 1}},
        "abs_g", selection, opt, backend)
    eng._dp_cp_group = lambda: None  # single-process: skip cross-rank layers

    eng.attach(chunk)
    assert backend.prepared == [(grp, chunk.bucket.grad_data.device)]
    eng.attach(chunk)                  # duplicate attach is a no-op
    assert backend.prepared == [(grp, chunk.bucket.grad_data.device)]

    mp = _FakeMPOpt(chunk)
    with _ReplicaWorld(world=1):
        _accumulate_into_bucket(chunk)
        assert eng.grad_sync(chunk, force_all_reduce=False) is True
        ok, _, _ = eng.optimizer_step(mp, [chunk], 1)
    assert ok is True
    # called with (cycle.start_iter, cycle.period); result lands in ctx
    assert selection.anchor_calls == [(0, 1)]
    assert selection.ctx.grad_views is not None


def test_count_guard_once_per_cycle_and_flag_off():
    """Layer-1 count guard placement (2026-08-28 ruling): it fires ONCE per
    support activation — never on sparse grad_sync steps (the counts are the
    activation-time caches and the support is immutable within a cycle, so
    per-step checks re-verify identical numbers). checks.count_guard=false
    (the legacy skip flag's counterpart) disables it entirely."""
    def _run_and_count(config_extra):
        selection = build_selection("topk", {"method": "histogram", "ratio_percent": 30.0})
        opt = build_optimizer("adamw", {"residual_mean": False},
                              emit_score="update")
        eng = SGAEngine({"cycle": dict(_ABS_M["cycle"]), **config_extra},
                        "update", selection, opt, TorchPipelinedIndexBackend({}))
        eng._dp_cp_group = lambda: None
        calls = []
        orig = eng._check_counts
        def _spy():
            calls.append(1)
            orig()
        eng._check_counts = _spy
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng.attach(chunk)
        history = []
        with _ReplicaWorld(world=1):
            for step in range(1, 10):  # _ABS_M cycle: reselections at 5 and 8
                _accumulate_into_bucket(chunk)
                eng.grad_sync(chunk, force_all_reduce=False)
                eng.optimizer_step(mp, [chunk], step)
                history.append(len(calls))
        return history

    history = _run_and_count({})
    # guard ON: fires at the two activations (steps 5, 8), never on the
    # sparse grad_sync steps in between
    assert history == [0, 0, 0, 0, 1, 1, 1, 2, 2], history
    history = _run_and_count({"checks": {"count_guard": False}})
    assert history == [0] * 9, history
    print("ok: count guard once per cycle (activations only); flag disables")


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
    eng_a = _build_engine(_UPDATE)
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
    eng_b = _build_engine(_UPDATE)
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
    eng = _build_engine(_UPDATE)
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


# ------------------------------------------------- num-zeros BEFORE-optimizer ordering

def test_num_zeros_logged_before_optimizer_overwrite():
    """With the num-zeros flag forced on, a named-score full step must report
    the zero count of the PRE-optimizer gradient (the full step then
    overwrites main grads with the materialized score)."""
    chunk = _FakeChunk()
    mp = _FakeMPOpt(chunk)
    eng = _build_engine(_ABS_M)
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
