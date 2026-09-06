#!/usr/bin/env python3
"""Unit tests for shadow mode (run: python tests/test_shadow_unit.py).

Covers: the literal stock-replication call sequence (spy-verified: stock
clip, stock commit; the engine masked clip and the optimizer template never
run), numerical equivalence of the whole replication path vs a plain stock
step loop (with boundary scoring active every step), the found-inf
early-return + full clear, the RNG-consumption guard (CPU and CUDA halves),
read-only score formulas vs the optimizer kernel's torch reference
(elementwise), the peek VALUE (upcoming-commit tracking across both
optimizer counter styles), the probe configuration (record-free machinery,
no phantom reselect in the metrics), grad_sync delegation, and loader
validation.
"""
import json
import os
import sys
import tempfile
import warnings
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.engine import SGAEngine
from sga.launcher.backend_megatron import _prevalidate_sga
from sga.plugins import (
    build_backend_plugin,
    build_optimizer,
    build_selection,
)
from sga.plugins.optimizer import ParamRecord
from sga.plugins.optimizer.adamw import _TRITON_SCORE_MAP
from sga.plugins.optimizer.adamw_torch import grad_correct_torch
from sga.plugins.scores import score_names
from sga.shadow import readonly_score

from test_engine_unit import (
    _ABS_M,
    _FakeChunk,
    _FakeMPOpt,
    _ReplicaWorld,
    _UPDATE,
    _seed,
    _seed_combo,
)


class _ShadowCapableMPOpt(_FakeMPOpt):
    """FakeMPOpt plus the stock wrapper surface shadow mode calls."""

    def __init__(self, chunk):
        super().__init__(chunk)
        self.config = SimpleNamespace(clip_grad=1.0, log_num_zeros_in_grad=False)
        self.calls = []

    def clip_grad_norm(self, clip_grad):
        self.calls.append("clip")
        sq = torch.zeros(1)
        for mp in self.main_params:
            sq += mp.grad.float().pow(2).sum()
        norm = float(sq.sqrt())
        coeff = min(1.0, clip_grad / (norm + 1.0e-6))
        if coeff < 1.0:
            for mp in self.main_params:
                mp.grad.data.mul_(coeff)
        return norm

    def count_zeros(self):
        self.calls.append("count_zeros")
        n = 0
        for mp in self.main_params:
            n += mp.grad.numel() - int(torch.count_nonzero(mp.grad).item())
        return n

    def step_with_ready_grads(self):
        self.calls.append("commit")
        self.optimizer.step()
        self._copy_main_params_to_model_params()
        return True

    def prepare_grads(self):
        self.calls.append("prepare")
        return super().prepare_grads()


def _build_shadow_engine(combo, config_extra=None, shadow_score=True):
    _seed_combo(combo, {**(config_extra or {}),
                        "sga_shadow": True, "sga_shadow_score": shadow_score})
    selection = build_selection(combo["selection"])
    opt = build_optimizer(combo["opt"])
    eng = SGAEngine(combo["score"], selection, opt,
                    build_backend_plugin(combo["backend"]))
    eng._dp_cp_group = lambda: None
    return eng


def _push_grads(chunk, gen):
    off = 0
    for p in chunk.params:
        n = p.numel()
        chunk.bucket.grad_data[off:off + n] += torch.randn(n, generator=gen)
        off += n


def test_shadow_replicates_stock_call_sequence():
    """Stock clip/commit run in order; the engine masked clip and the
    optimizer template never fire; buffers are fully cleared at step end."""
    torch.manual_seed(0)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    eng = _build_shadow_engine(_UPDATE, shadow_score=False)
    eng.attach(chunk)

    for name in ("full_step", "sparse_step", "reset"):
        def refusing(_ctx, _n=name):
            raise AssertionError(f"optimizer template.{_n} must never run in shadow")
        setattr(eng.opt, name, refusing)
    def no_masked_clip(*a, **k):
        raise AssertionError("engine masked clip must never run in shadow")
    eng._masked_clip_grad_norm = no_masked_clip

    gen = torch.Generator().manual_seed(3)
    with _ReplicaWorld(world=1):
        for step in range(1, 4):
            _push_grads(chunk, gen)
            assert eng.grad_sync(chunk, force_all_reduce=False) is False
            chunk.finish_grad_sync(force_all_reduce=False)
            ok, _, _ = eng.optimizer_step(mp, [chunk], step)
            assert ok
    assert mp.calls == ["prepare", "clip", "commit"] * 3
    assert float(chunk.bucket.grad_data.abs().sum()) == 0.0, "end-of-step clear"
    print("PASS test_shadow_replicates_stock_call_sequence")


def test_shadow_numerics_equal_stock_loop():
    """Whole replication path (boundary scoring active EVERY step) leaves
    model w and optimizer m/v bitwise equal to a plain stock step loop."""
    chunk_sh, chunk_st = _FakeChunk(), _FakeChunk()
    with torch.no_grad():
        for p_sh, p_st in zip(chunk_sh.params, chunk_st.params):
            p_st.copy_(p_sh)
    mp_sh = _ShadowCapableMPOpt(chunk_sh)
    mp_st = _ShadowCapableMPOpt(chunk_st)
    eng = _build_shadow_engine(
        _ABS_M, config_extra={"sga_cycle_start_iter": 0,
                              "sga_cycle_period": 1},
        shadow_score=True)
    eng.attach(chunk_sh)

    gen_sh = torch.Generator().manual_seed(7)
    gen_st = torch.Generator().manual_seed(7)
    with _ReplicaWorld(world=1):
        for step in range(1, 5):
            _push_grads(chunk_sh, gen_sh)
            _push_grads(chunk_st, gen_st)
            ok, _, _ = eng.optimizer_step(mp_sh, [chunk_sh], step)
            assert ok
            # peek VALUE: rec.step is the UPCOMING commit number (an off-by-
            # one here corrupts every shadow score bitwise-invisibly)
            assert {r.step for r in eng._records} == {step}
            # stock loop: same wrapper primitives, no insertions
            mp_st.prepare_grads()
            mp_st.clip_grad_norm(1.0)
            mp_st.step_with_ready_grads()
            # stock world: next iteration head zeroes the buffer
            chunk_st.bucket.grad_data.zero_()
            for group in chunk_st.bucket_groups:
                group.reset()
    for p_sh, p_st in zip(chunk_sh.params, chunk_st.params):
        assert torch.equal(p_sh, p_st), "model params diverged"
    for mp_a, mp_b in zip(mp_sh.main_params, mp_st.main_params):
        assert torch.equal(mp_a, mp_b), "main params diverged"
        sa = mp_sh.optimizer.state[mp_a]
        sb = mp_st.optimizer.state[mp_b]
        assert torch.equal(sa["exp_avg"], sb["exp_avg"]), "m diverged"
        assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"]), "v diverged"
    for g in mp_sh.optimizer.param_groups:
        assert "step" not in g, "shadow must not create group['step']"
    print("PASS test_shadow_numerics_equal_stock_loop")


def test_shadow_found_inf_full_clear_and_skip():
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    def found_inf_prepare():
        mp.calls.append("prepare")
        return True  # stock contract: True = found_inf (BAD)
    mp.prepare_grads = found_inf_prepare
    eng = _build_shadow_engine(_UPDATE, shadow_score=True)
    eng.attach(chunk)
    chunk.bucket.grad_data.add_(1.0)
    with _ReplicaWorld(world=1):
        ok, norm, nz = eng.optimizer_step(mp, [chunk], 1)
    assert (ok, norm, nz) == (False, None, None)
    assert float(chunk.bucket.grad_data.abs().sum()) == 0.0
    assert mp.calls == ["prepare"], f"no stock work after found_inf: {mp.calls}"
    assert eng._iter == 1  # failed step consumed (skipped, not retried)
    print("PASS test_shadow_found_inf_full_clear_and_skip")


def test_shadow_rng_guard_raises():
    """RNG consumption inside the boundary insertion (here: the readonly
    score evaluation) fails the RNG assert — the shadow guard dog."""
    import sga.engine as engine_mod

    torch.manual_seed(3)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    _seed({"sga_selection_method": "histogram",
           "sga_selection_ratio_percent": 30.0, "sga_score": "abs_m",
           "sga_cycle_start_iter": 0, "sga_cycle_period": 1,
           "sga_shadow": True})
    selection = build_selection("topk")
    opt = build_optimizer("adamw")
    eng = SGAEngine("abs_m", selection, opt,
                    build_backend_plugin(_ABS_M["backend"]))
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)
    orig = engine_mod.readonly_score

    def rng_eating(rec, emit_name, scratch=None):
        torch.rand(1)  # forbidden consumption on the CPU stream
        return orig(rec, emit_name, scratch)

    engine_mod.readonly_score = rng_eating
    gen = torch.Generator().manual_seed(5)
    try:
        with _ReplicaWorld(world=1):
            _push_grads(chunk, gen)
            chunk.finish_grad_sync(force_all_reduce=False)
            try:
                eng.optimizer_step(mp, [chunk], 1)  # boundary at step 1
            except RuntimeError as e:
                assert "RNG" in str(e), f"wrong error: {e}"
            else:
                raise SystemExit("RNG consumption must fail the shadow assert")
    finally:
        engine_mod.readonly_score = orig
    print("PASS test_shadow_rng_guard_raises")


def test_readonly_score_matches_optimizer_torch_reference():
    """readonly_score vs grad_correct_torch's materialized output,
    elementwise, for every registry emit name (a plumbing-equivalence pin,
    not a math one: both sides call the same registry formulas, so the
    score-math comparison is tautological by design — what this actually
    pins is state selection, bias-correction/step resolution, the read-only
    contract and scratch routing); inputs stay untouched (read-only)."""
    torch.manual_seed(4)
    for emit in ("update", "coi", "abs_g", "abs_m"):
        n = 256
        w = torch.randn(n)
        g = torch.randn(n)
        m0 = torch.randn(n) * 0.1
        v0 = torch.rand(n) * 0.01 + 1e-4
        b1, b2, eps, wd, lr = 0.9, 0.95, 1e-8, 0.1, 1e-3
        step = 37
        # kernel reference over clones: the score lands in the grad buffer
        g_ref = g.clone()
        w_ref = w.clone()
        grad_correct_torch(
            w_ref, g_ref, m0.clone(), v0.clone(), lr, b1, b2, eps, wd,
            1.0 - b1 ** step, 1.0 - b2 ** step,
            None, emit,
        )
        # shadow: fresh record over the same pre-commit state
        main_p = torch.nn.Parameter(w.clone())
        main_p.grad = g.clone()
        rec = ParamRecord(
            model_param=torch.nn.Parameter(w.clone()), main_param=main_p,
            bucket_state=SimpleNamespace(active_mask_u8=None),
            start=0, end=n, lr=lr, beta1=b1, beta2=b2, eps=eps,
            weight_decay=wd, step=step,
        )
        rec._state = {"exp_avg": m0.clone(), "exp_avg_sq": v0.clone()}
        got = readonly_score(rec, emit)
        assert torch.equal(got, g_ref), f"{emit}: shadow score != kernel reference"
        assert torch.equal(main_p.data, w) and torch.equal(main_p.grad, g)
        assert torch.equal(rec._state["exp_avg"], m0)
        assert torch.equal(rec._state["exp_avg_sq"], v0)
        # scratch routing: same values, fresh (non-aliasing) results, and a
        # second call reusing the SAME buffers still matches (per-param
        # outputs are held simultaneously and must stay distinct)
        scratch = (torch.empty(2 * n), torch.empty(2 * n))
        got_s = readonly_score(rec, emit, scratch)
        assert torch.equal(got_s, g_ref), f"{emit}: scratch path drifted"
        assert got_s.data_ptr() not in (scratch[0].data_ptr(),
                                        scratch[1].data_ptr()), emit
        got_s2 = readonly_score(rec, emit, scratch)
        assert torch.equal(got_s2, g_ref) and got_s2.data_ptr() != got_s.data_ptr()
    # state-absent record (pre-first-commit): moments read as zeros
    n = 16
    main_p = torch.nn.Parameter(torch.randn(n))
    main_p.grad = torch.randn(n)
    rec = ParamRecord(
        model_param=main_p, main_param=main_p,
        bucket_state=SimpleNamespace(active_mask_u8=None),
        start=0, end=n, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
        weight_decay=0.1, step=1,
    )
    rec._state = {}
    assert torch.equal(readonly_score(rec, "abs_m"), torch.zeros(n))
    # zero-state edges vs the reference: the update-direction scores over
    # zero moments (the pre-first-commit boundary every real run hits)
    for emit in ("update", "coi"):
        w = torch.randn(n)
        g = torch.randn(n)
        main_p = torch.nn.Parameter(w.clone())
        main_p.grad = g.clone()
        rec = ParamRecord(
            model_param=torch.nn.Parameter(w.clone()), main_param=main_p,
            bucket_state=SimpleNamespace(active_mask_u8=None),
            start=0, end=n, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
            weight_decay=0.1, step=1,
        )
        rec._state = {}
        g_ref = g.clone()
        grad_correct_torch(
            w.clone(), g_ref, torch.zeros(n), torch.zeros(n), 1e-3, 0.9, 0.95,
            1e-8, 0.1, 1.0 - 0.9 ** 1, 1.0 - 0.95 ** 1,
            None, emit,
        )
        assert torch.equal(readonly_score(rec, emit), g_ref), \
            f"{emit} at zero state: shadow score != kernel reference"
    print("PASS test_readonly_score_matches_optimizer_torch_reference")


def test_readonly_score_mixed_numel_records_share_one_scratch_pair():
    """C1 regression: two records of DIFFERENT numel AND rank score through
    ONE shared scratch pair sized to the max numel — exactly what the engine
    allocates per shadow boundary (a single flat max_n pair reused for every
    param in meta order, smaller-before-larger in practice). The formulas'
    [:n].view_as(w) narrowing must keep the out= chain off the pair's shape:
    an out= resize (deprecated, emits a UserWarning) would shrink the shared
    buffers under the first, smaller record and the next, larger record would
    trip readonly_score's contract assert."""
    torch.manual_seed(9)
    max_n = 10 * 20
    shared = (torch.empty(max_n), torch.empty(max_n))
    for shape in ((50,), (10, 20)):
        n = shape[0] * (shape[1] if len(shape) > 1 else 1)
        w = torch.randn(shape)
        g = torch.randn(shape)
        main_p = torch.nn.Parameter(w.clone())
        main_p.grad = g.clone()
        rec = ParamRecord(
            model_param=torch.nn.Parameter(w.clone()), main_param=main_p,
            bucket_state=SimpleNamespace(active_mask_u8=None),
            start=0, end=n, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
            weight_decay=0.1, step=5,
        )
        rec._state = {"exp_avg": torch.randn(shape) * 0.1,
                      "exp_avg_sq": torch.rand(shape) + 1e-4}
        fresh = readonly_score(rec, "update")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            got = readonly_score(rec, "update", shared)
        assert torch.equal(got, fresh), shape
        assert got.data_ptr() not in (shared[0].data_ptr(),
                                      shared[1].data_ptr()), shape
        # the pair keeps its full capacity across records (never resized)
        assert shared[0].numel() == max_n and shared[1].numel() == max_n
    print("PASS test_readonly_score_mixed_numel_records_share_one_scratch_pair")


def test_shadow_peek_matches_stock_fusedadam_step():
    """Shadow must never write group['step'] (TE/apex FusedAdam semantics:
    the stock commit itself does group['step'] = get(step, 0) + 1 inside
    step(); an engine-side advance would double-step every bias
    correction)."""
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    used = []
    orig_step = mp.optimizer.step

    def fused_style_step(*a, **k):
        for g in mp.optimizer.param_groups:
            g["step"] = g.get("step", 0) + 1  # apex/TE FusedAdam behavior
            used.append(g["step"])
        return orig_step(*a, **k)

    mp.optimizer.step = fused_style_step
    eng = _build_shadow_engine(
        _ABS_M, config_extra={"sga_cycle_start_iter": 0,
                              "sga_cycle_period": 1},
        shadow_score=True)
    eng.attach(chunk)
    gen = torch.Generator().manual_seed(11)
    with _ReplicaWorld(world=1):
        for step in range(1, 4):
            _push_grads(chunk, gen)
            ok, _, _ = eng.optimizer_step(mp, [chunk], step)
            assert ok
            # peek VALUE on the group['step'] branch: rec.step must be the
            # upcoming commit number, not the last committed one
            assert {r.step for r in eng._records} == {step}
    assert used == [1, 2, 3], f"stock commit steps must be 1,2,3 (got {used})"
    print("PASS test_shadow_peek_matches_stock_fusedadam_step")


def test_shadow_peek_value_tracks_upcoming_commit_torch_adamw():
    """torch-AdamW-style counters (NO group['step']; per-param tensor
    state['step'] created only at the first commit): the peek VALUE is the
    upcoming commit number at every step — via the no-counter fallthrough at
    step 1 and the per-param state branch afterwards. A silent off-by-one
    here corrupts every shadow bias correction while dense numerics stay
    bitwise identical, so only this pin catches it."""
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    eng = _build_shadow_engine(
        _ABS_M, config_extra={"sga_cycle_start_iter": 0,
                              "sga_cycle_period": 1},
        shadow_score=True)
    eng.attach(chunk)
    gen = torch.Generator().manual_seed(19)
    with _ReplicaWorld(world=1):
        for step in range(1, 5):
            _push_grads(chunk, gen)
            ok, _, _ = eng.optimizer_step(mp, [chunk], step)
            assert ok
            assert {r.step for r in eng._records} == {step}
            # peek must not seed the group counter the advance path would
            assert all("step" not in g for g in mp.optimizer.param_groups)
    print("PASS test_shadow_peek_value_tracks_upcoming_commit_torch_adamw")


def test_shadow_grad_sync_returns_false():
    """Shadow grad_sync delegates to the fork's stock finish_grad_sync and
    stamps no phase flags."""
    chunk = _FakeChunk()
    eng = _build_shadow_engine(_UPDATE)
    eng.attach(chunk)
    assert eng.grad_sync(chunk, force_all_reduce=False) is False
    print("PASS test_shadow_grad_sync_returns_false")


def test_shadow_probe_config_never_builds_records_or_state():
    """shadow_score=false (the probe) strips the boundary machinery entirely:
    _refresh_records never runs, the optimizer state dict gains no entries
    from the engine (only the stock commit materializes them), and the
    metrics JSONL never records a reselection — the calendar boundary
    without shadow_score is not a reselection that ran."""
    with tempfile.TemporaryDirectory() as td:
        metrics_path = os.path.join(td, "metrics.jsonl")
        prev = os.environ.get("SGA_METRICS_FILE")
        os.environ["SGA_METRICS_FILE"] = metrics_path
        try:
            chunk = _FakeChunk()
            mp = _ShadowCapableMPOpt(chunk)
            eng = _build_shadow_engine(
                _ABS_M, config_extra={"sga_cycle_start_iter": 0,
                                      "sga_cycle_period": 2},
                shadow_score=False)
            eng.attach(chunk)
            refresh_calls = []
            real_refresh = eng._refresh_records

            def spy_refresh(mp_opt):
                refresh_calls.append(1)
                return real_refresh(mp_opt)

            eng._refresh_records = spy_refresh
            pre_commit_lens = []
            real_commit = mp.step_with_ready_grads

            def commit_spy():
                pre_commit_lens.append(len(mp.optimizer.state))
                return real_commit()

            mp.step_with_ready_grads = commit_spy
            gen = torch.Generator().manual_seed(13)
            with _ReplicaWorld(world=1):
                for step in range(1, 6):  # calendar boundaries at steps 2, 4
                    _push_grads(chunk, gen)
                    ok, _, _ = eng.optimizer_step(mp, [chunk], step)
                    assert ok
            with open(metrics_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        finally:
            if prev is None:
                os.environ.pop("SGA_METRICS_FILE", None)
            else:
                os.environ["SGA_METRICS_FILE"] = prev
    assert refresh_calls == [], "probe must never build/refresh records"
    # state size grows ONLY at the first stock commit (its own lazy init);
    # an engine-side record build would have materialized entries before it
    assert pre_commit_lens == [0] + [len(chunk.params)] * 4, pre_commit_lens
    assert [r["reselected"] for r in rows] == [False] * 5
    print("PASS test_shadow_probe_config_never_builds_records_or_state")


def test_shadow_metrics_reselected_tracks_boundary_reselection():
    """shadow_score=true: the metrics JSONL reselected flag is true at
    exactly the boundaries where the shadow reselection ran (steps 2, 4)."""
    with tempfile.TemporaryDirectory() as td:
        metrics_path = os.path.join(td, "metrics.jsonl")
        prev = os.environ.get("SGA_METRICS_FILE")
        os.environ["SGA_METRICS_FILE"] = metrics_path
        try:
            chunk = _FakeChunk()
            mp = _ShadowCapableMPOpt(chunk)
            eng = _build_shadow_engine(
                _ABS_M, config_extra={"sga_cycle_start_iter": 0,
                                      "sga_cycle_period": 2},
                shadow_score=True)
            eng.attach(chunk)
            gen = torch.Generator().manual_seed(17)
            with _ReplicaWorld(world=1):
                for step in range(1, 6):
                    _push_grads(chunk, gen)
                    ok, _, _ = eng.optimizer_step(mp, [chunk], step)
                    assert ok
            with open(metrics_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        finally:
            if prev is None:
                os.environ.pop("SGA_METRICS_FILE", None)
            else:
                os.environ["SGA_METRICS_FILE"] = prev
    assert [r["reselected"] for r in rows] == [False, True, False, True, False]
    print("PASS test_shadow_metrics_reselected_tracks_boundary_reselection")


_BASE = {
    "experiment": {"exp_name": "t", "exp_dir": "/tmp/x",
                   "task": {"type": "train", "backend": "megatron",
                            "entrypoint": "./sga/train_gpt.py"},
                   "runner": {"type": "ssh"}},
    "sga": {
        "sga_enabled": True,
        "sga_cycle_start_iter": 0,
        "sga_cycle_period": 8,
        "sga_score": "abs_m",
        "sga_selection_method": "histogram",
        "sga_selection_ratio_percent": 30.0,
        "sga_optimizer": "adamw",
        "sga_backend": "torch_pipelined_index",
    },
    "train": {"system": {"tensor_model_parallel_size": 1},
              "model": {"num_layers": 2},
              "data": {"data_path": "/tmp/x"}},
}


def _cfg_with(**sga_overrides):
    cfg = OmegaConf.create(_BASE)
    for k, v in sga_overrides.items():
        cfg.sga["sga_" + k] = v
    return cfg


def _sga_section_to_config(cfg):
    return _prevalidate_sga(cfg)


def _expect_error(cfg, needle):
    try:
        _sga_section_to_config(cfg)
    except (ValueError, AssertionError) as e:
        assert needle in str(e), f"missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


def test_loader_shadow_validation():
    # positive spellings all pass the launch-time pre-validation
    _sga_section_to_config(_cfg_with(shadow=True))
    _sga_section_to_config(_cfg_with(shadow=False))
    _sga_section_to_config(_cfg_with(shadow=True, shadow_score=False))

    _expect_error(_cfg_with(shadow="yes"), "must be 'true' or 'false'")
    _expect_error(_cfg_with(shadow_score=False), "sga_shadow is not true")
    _expect_error(_cfg_with(shadow=True, shadow_score="no"), "must be 'true' or 'false'")

    # disabled → zero checks (megatron semantics): a shadow key under a
    # disabled config is silently ignored, not refused
    cfg = OmegaConf.create(_BASE)
    cfg.sga = {"sga_enabled": False, "sga_backend": "dense",
               "sga_shadow": True}
    _sga_section_to_config(cfg)
    print("PASS test_loader_shadow_validation")


def test_shadow_rng_guard_cuda_half_raises():
    """The RNG guard's CUDA half: a boundary insertion that consumes a CUDA
    RNG stream fails the assert even when the CPU stream stays clean (the
    CPU-only test is structurally blind to this half on CPU-only test envs —
    exercised via monkeypatched torch.cuda state accessors)."""
    import sga.engine as engine_mod

    torch.manual_seed(3)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    _seed({"sga_selection_method": "histogram",
           "sga_selection_ratio_percent": 30.0, "sga_score": "abs_m",
           "sga_cycle_start_iter": 0, "sga_cycle_period": 1,
           "sga_shadow": True})
    selection = build_selection("topk")
    opt = build_optimizer("adamw")
    eng = SGAEngine("abs_m", selection, opt,
                    build_backend_plugin(_ABS_M["backend"]))
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)

    calls = {"n": 0}
    snap = [torch.arange(8, dtype=torch.int64)]

    def states_all():
        calls["n"] += 1
        if calls["n"] >= 2:  # the post-insertion read: stream state moved
            return [snap[0] + 1]
        return [snap[0].clone()]

    real_avail = torch.cuda.is_available
    real_states = torch.cuda.get_rng_state_all
    torch.cuda.is_available = lambda: True
    torch.cuda.get_rng_state_all = states_all
    orig = engine_mod.readonly_score
    engine_mod.readonly_score = (lambda rec, emit, scratch=None:
                                  orig(rec, emit, scratch))
    try:
        gen = torch.Generator().manual_seed(5)
        with _ReplicaWorld(world=1):
            _push_grads(chunk, gen)
            chunk.finish_grad_sync(force_all_reduce=False)
            try:
                eng.optimizer_step(mp, [chunk], 1)
            except RuntimeError as e:
                assert "CUDA RNG" in str(e), f"wrong error: {e}"
            else:
                raise SystemExit("CUDA RNG consumption must fail the assert")
        assert calls["n"] >= 2, "guard must read CUDA state before and after"
    finally:
        torch.cuda.is_available = real_avail
        torch.cuda.get_rng_state_all = real_states
        engine_mod.readonly_score = orig
    print("PASS test_shadow_rng_guard_cuda_half_raises")


def test_readonly_score_grad_none_guard():
    """A record without a gradient at the boundary fails with the named
    diagnostic instead of an opaque AttributeError deep in the formula."""
    n = 8
    main_p = torch.nn.Parameter(torch.randn(n))
    rec = ParamRecord(
        model_param=main_p, main_param=main_p,
        bucket_state=SimpleNamespace(active_mask_u8=None),
        start=0, end=n, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
        weight_decay=0.1, step=3,
    )
    rec._state = {}
    try:
        readonly_score(rec, "abs_m")
    except ValueError as e:
        assert "no gradient" in str(e), e
    else:
        raise SystemExit("grad-less record must be rejected")
    print("PASS test_readonly_score_grad_none_guard")


def test_registry_scores_shadow_evaluable_and_decl_alignment():
    """The registry is the single emit namespace: every registered score
    evaluates through readonly_score (existence pin; value drift is pinned by
    test_readonly_score_matches_optimizer_torch_reference), unknown names
    fail with the registered list, and the adamw kernel's branch inventory
    (_TRITON_SCORE_MAP) covers the whole registry with distinct branch ids —
    every registry entry is online-usable today; a replay-only score must
    consciously relax the equality pin (and the kernel will refuse it loudly
    at optimizer construction)."""
    torch.manual_seed(2)
    n = 16
    main_p = torch.nn.Parameter(torch.randn(n))
    main_p.grad = torch.randn(n)
    rec = ParamRecord(
        model_param=torch.nn.Parameter(torch.randn(n)), main_param=main_p,
        bucket_state=SimpleNamespace(active_mask_u8=None),
        start=0, end=n, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
        weight_decay=0.1, step=5,
    )
    rec._state = {"exp_avg": torch.randn(n) * 0.1,
                  "exp_avg_sq": torch.rand(n) * 0.01}
    for emit in score_names():
        out = readonly_score(rec, emit)
        assert out.shape == (n,) and out.dtype == torch.float32, emit
    try:
        readonly_score(rec, "bogus")
    except ValueError as e:
        assert "bogus" in str(e) and "registered" in str(e), e
    else:
        raise SystemExit("unknown emit must be rejected")

    assert set(_TRITON_SCORE_MAP) == set(score_names()), \
        ("every registry entry must carry a kernel branch id (a replay-only "
         "entry must consciously relax this pin)")
    assert sorted(_TRITON_SCORE_MAP.values()) == \
        list(range(len(_TRITON_SCORE_MAP))), \
        "branch ids are distinct dense ints (kernel if-chain inventory)"
    print("PASS test_registry_scores_shadow_evaluable_and_decl_alignment")


def test_shadow_activates_without_idx_materialization():
    """VRAM gate: an idx-consuming backend under shadow never compacts the
    support — grad sync stays stock so no consumer exists, and the compacted
    pool would be resident waste. Mask-derived counts still resolve (both
    consistency layers are mask-based)."""
    torch.manual_seed(11)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    assert build_backend_plugin(_ABS_M["backend"]).consumes == "idx"
    # 'update' score: nonzero already at step 1 (abs_m over the zero
    # pre-first-commit moments would tie at 0 everywhere and select all)
    combo = dict(_ABS_M, score="update")
    eng = _build_shadow_engine(
        combo, config_extra={"sga_cycle_start_iter": 0,
                             "sga_cycle_period": 1},
        shadow_score=True)
    eng.attach(chunk)
    gen = torch.Generator().manual_seed(5)
    with _ReplicaWorld(world=1):
        _push_grads(chunk, gen)
        ok, _, _ = eng.optimizer_step(mp, [chunk], 1)
        assert ok
    assert eng._active_support is not None
    assert all(b._idx is None for b in eng._active_support.buckets), \
        "shadow must not materialize idx for an idx-consuming backend"
    assert 0 < eng._active_support.total_selected \
        < eng._active_support.total_elements
    print("PASS test_shadow_activates_without_idx_materialization")




def test_shadow_always_sync_inherits_eligible_interception():
    """The shadow boundary rides the SAME shared selection tail —
    the forced spans land all-ones in the shadow support and the selector
    sees only the eligible pool (no live/shadow support divergence)."""
    from test_engine_unit import _ALWAYS, _AS_NAMES, _AS_PATTERNS

    torch.manual_seed(0)
    chunk = _FakeChunk(n_params=len(_AS_NAMES), names=_AS_NAMES)
    mp = _ShadowCapableMPOpt(chunk)
    _seed({**_ALWAYS["selection_params"], **_ALWAYS["opt_params"],
           **_ALWAYS["backend_params"],
           "sga_cycle_start_iter": 0, "sga_cycle_period": 1,
           "sga_always_sync_layers": _AS_PATTERNS,
           "sga_shadow": True, "sga_shadow_score": True,
           "sga_score": "abs_g"})
    selection = build_selection("topk")
    opt = build_optimizer("adamw")
    eng = SGAEngine(
        "abs_g", selection, opt, build_backend_plugin(_ALWAYS["backend"]))
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)
    gen = torch.Generator().manual_seed(5)
    with _ReplicaWorld(world=1):
        _push_grads(chunk, gen)
        chunk.finish_grad_sync(force_all_reduce=False)
        ok, _, _ = eng.optimizer_step(mp, [chunk], 1)  # period=1: boundary
        assert ok is True
    mask = eng._bucket_states[0].active_mask_u8
    n_per = chunk.params[0].numel()
    assert mask[:n_per].min() == 1 and mask[2 * n_per:].min() == 1
    assert int(mask[n_per:2 * n_per].sum()) == max(1, int(round(n_per * 0.30)))
    print("PASS test_shadow_always_sync_inherits_eligible_interception")


def main():
    test_shadow_always_sync_inherits_eligible_interception()
    test_shadow_replicates_stock_call_sequence()
    test_shadow_numerics_equal_stock_loop()
    test_shadow_found_inf_full_clear_and_skip()
    test_shadow_rng_guard_raises()
    test_readonly_score_matches_optimizer_torch_reference()
    test_readonly_score_mixed_numel_records_share_one_scratch_pair()
    test_shadow_peek_matches_stock_fusedadam_step()
    test_shadow_peek_value_tracks_upcoming_commit_torch_adamw()
    test_shadow_grad_sync_returns_false()
    test_shadow_probe_config_never_builds_records_or_state()
    test_shadow_metrics_reselected_tracks_boundary_reselection()
    test_loader_shadow_validation()
    test_shadow_rng_guard_cuda_half_raises()
    test_readonly_score_grad_none_guard()
    test_registry_scores_shadow_evaluable_and_decl_alignment()
    test_shadow_activates_without_idx_materialization()
    print("\nall shadow unit tests passed")


if __name__ == "__main__":
    main()
