#!/usr/bin/env python3
"""Unit tests for shadow mode (run: python tests/test_shadow_unit.py).

Covers: the literal stock-replication call sequence (spy-verified: stock
clip, stock commit; the engine masked clip and the EF never run), numerical
equivalence of the whole replication path vs a plain stock step loop (with
boundary scoring active every step), the found-inf early-return + full
clear, the RNG-consumption guard, read-only score formulas vs the EF torch
reference (elementwise), shadow support lineage (sticky gets the pre-switch
shadow support), the peek VALUE (upcoming-commit tracking across both
optimizer counter styles), the probe configuration (record-free machinery,
no phantom reselect in the metrics), grad_sync delegation, and loader
validation.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config import sga_section_to_config
from sga.engine import SGAEngine
from sga.plugins import build_ef, build_score, build_selection
from sga.plugins.error_feedback import ParamRecord
from sga.plugins.error_feedback.radius_opt import _radius_grad_correct_torch
from sga.shadow import readonly_score

from test_engine_unit import (
    _FakeChunk,
    _FakeMPOpt,
    _ORIGIN,
    _RADIUS,
    _ReplicaWorld,
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


def _build_shadow_engine(combo, cycle=None, shadow_score=True):
    score = build_score(combo["score"], {"name": combo["score"]})
    selection = build_selection(combo["selection"], dict(combo["selection_params"]),
                                score=score)
    ef = build_ef(combo["ef"], dict(combo["ef_params"]))
    eng = SGAEngine({"cycle": cycle or dict(combo["cycle"]),
                     "shadow": True, "shadow_score": shadow_score},
                    score, selection, ef, combo["backend"])
    eng._dp_cp_group = lambda: None
    return eng


def _push_grads(chunk, gen):
    off = 0
    for p in chunk.params:
        n = p.numel()
        chunk.bucket.grad_data[off:off + n] += torch.randn(n, generator=gen)
        off += n


def test_shadow_replicates_stock_call_sequence():
    """Stock clip/commit run in order; the engine masked clip and the EF
    never fire; buffers are fully cleared at step end."""
    torch.manual_seed(0)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    eng = _build_shadow_engine(_ORIGIN, shadow_score=False)
    eng.attach(chunk)

    for name in ("full_step", "sparse_step", "reset"):
        def refusing(_ctx, _n=name):
            raise AssertionError(f"EF.{_n} must never run in shadow")
        setattr(eng.ef, name, refusing)
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
        _RADIUS, cycle={"start_iter": 0, "period": 1, "full_sync_interval": 0},
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
    eng = _build_shadow_engine(_ORIGIN, shadow_score=True)
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
    """RNG consumption inside the boundary insertion (here: the score
    plugin's values()) fails the RNG assert — the shadow guard dog."""
    torch.manual_seed(3)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    # synced score: values() sits on the shadow data path
    score = build_score("abs_m", {"name": "abs_m"})
    selection = build_selection("circular_topk",
                                {"ratio_percent": 30.0}, score=score)
    ef = build_ef("radius_opt", {})
    eng = SGAEngine({"cycle": {"start_iter": 0, "period": 1,
                               "full_sync_interval": 0},
                     "shadow": True}, score, selection, ef,
                    _RADIUS["backend"])
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)
    orig_values = score.values

    def rng_eating(views, support_slices=None):
        torch.rand(1)  # forbidden consumption on the CPU stream
        if support_slices is None:
            return orig_values(views)
        return orig_values(views, support_slices)

    eng.score.values = rng_eating
    gen = torch.Generator().manual_seed(5)
    with _ReplicaWorld(world=1):
        _push_grads(chunk, gen)
        chunk.finish_grad_sync(force_all_reduce=False)
        try:
            eng.optimizer_step(mp, [chunk], 1)  # boundary at step 1
        except RuntimeError as e:
            assert "RNG" in str(e), f"wrong error: {e}"
        else:
            raise SystemExit("RNG consumption must fail the shadow assert")
    print("PASS test_shadow_rng_guard_raises")


def test_readonly_score_matches_ef_torch_reference():
    """readonly_score vs _radius_grad_correct_torch's materialized output,
    elementwise, for every emit name (the anti-drift pin); inputs stay
    untouched (read-only)."""
    torch.manual_seed(4)
    for emit in ("update_ratio", "coi", "abs_g_synced", "abs_m",
                 "snr_m", "abs_m_post"):
        n = 256
        w = torch.randn(n)
        g = torch.randn(n)
        m0 = torch.randn(n) * 0.1
        v0 = torch.rand(n) * 0.01 + 1e-4
        b1, b2, eps, wd, lr = 0.9, 0.95, 1e-8, 0.1, 1e-3
        step = 37
        # EF reference over clones: named scores land in the grad buffer,
        # the default signed update lands there when no side buffer is given
        g_ref = g.clone()
        _radius_grad_correct_torch(
            w.clone(), g_ref, m0.clone(), v0.clone(), lr, b1, b2, eps, wd,
            1.0 - b1 ** step, 1.0 - b2 ** step,
            1.0 - b1 ** (step - 1), 1.0 - b2 ** (step - 1),
            None, True,
            score_out=(torch.empty(n) if emit != "update_ratio" else None),
            score_mode=(emit if emit != "update_ratio" else None),
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
        assert torch.equal(got, g_ref), f"{emit}: shadow score != EF reference"
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
    # zero-state edges vs the reference: snr_m at step 1 (the
    # bias_correction1_pre == 0 collapse-to-zeros branch) and a formula emit
    # (update_ratio) over zero moments
    for emit in ("snr_m", "update_ratio"):
        w = torch.randn(n)
        g = torch.randn(n)
        main_p = torch.nn.Parameter(w.clone())
        main_p.grad = g.clone()
        rec = ParamRecord(
            model_param=torch.nn.Parameter(w.clone()), main_param=main_p,
            bucket_state=SimpleNamespace(active_mask_u8=None),
            start=0, end=n, lr=lr, beta1=b1, beta2=b2, eps=eps,
            weight_decay=wd, step=1,
        )
        rec._state = {}
        g_ref = g.clone()
        _radius_grad_correct_torch(
            w.clone(), g_ref, torch.zeros(n), torch.zeros(n), lr, b1, b2,
            eps, wd, 1.0 - b1 ** 1, 1.0 - b2 ** 1,
            1.0 - b1 ** 0, 1.0 - b2 ** 0,
            None, True,
            score_out=(torch.empty(n) if emit != "update_ratio" else None),
            score_mode=(emit if emit != "update_ratio" else None),
        )
        assert torch.equal(readonly_score(rec, emit), g_ref), \
            f"{emit} at zero state: shadow score != EF reference"
    print("PASS test_readonly_score_matches_ef_torch_reference")


def test_shadow_sticky_gets_pre_switch_shadow_support():
    """At the second boundary the sticky score receives the FIRST boundary's
    shadow support (lineage via _activate_support), None at the first."""
    torch.manual_seed(5)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    score = build_score("abs_m_sticky", {"name": "abs_m_sticky"})
    selection = build_selection("circular_topk",
                                {"ratio_percent": 30.0}, score=score)
    ef = build_ef("radius_opt", {})
    eng = SGAEngine({"cycle": {"start_iter": 0, "period": 2,
                               "full_sync_interval": 0},
                     "shadow": True}, score, selection, ef,
                    _ORIGIN["backend"])
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)
    seen = []
    orig_values = score.values

    def spy(views, support_slices=None):
        seen.append(None if support_slices is None else
                    [s.clone() for s in support_slices])
        return orig_values(views, support_slices)

    score.values = spy
    gen = torch.Generator().manual_seed(6)
    expected = None
    with _ReplicaWorld(world=1):
        for step in range(1, 5):
            _push_grads(chunk, gen)
            chunk.finish_grad_sync(force_all_reduce=False)
            ok, _, _ = eng.optimizer_step(mp, [chunk], step)
            assert ok
            if step == 2:
                # expected side of the lineage: the just-activated step-2
                # support in meta order (the spy clone covers the received
                # side only)
                expected = [s.clone() for s in
                            eng._support_slices(eng._selection_views()[2])]
    assert len(seen) == 2  # boundaries at steps 2 and 4
    assert seen[0] is None  # bootstrap: no shadow support yet
    assert seen[1] is not None and all(s is not None for s in seen[1])
    # activation runs after score.values in _shadow_reselect, so the
    # post-switch support does not exist yet — compare elementwise against
    # the step-2 support
    for got, want in zip(seen[1], expected):
        assert torch.equal(got, want)
    print("PASS test_shadow_sticky_gets_pre_switch_shadow_support")


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
        _RADIUS, cycle={"start_iter": 0, "period": 1, "full_sync_interval": 0},
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
        _RADIUS, cycle={"start_iter": 0, "period": 1, "full_sync_interval": 0},
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
    eng = _build_shadow_engine(_ORIGIN)
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
                _RADIUS, cycle={"start_iter": 0, "period": 2,
                                "full_sync_interval": 0},
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
                _RADIUS, cycle={"start_iter": 0, "period": 2,
                                "full_sync_interval": 0},
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
    assert [r["reselected"] for r in rows] == \
        [False, True, False, True, False]
    print("PASS test_shadow_metrics_reselected_tracks_boundary_reselection")


_BASE = {
    "experiment": {"exp_name": "t", "exp_dir": "/tmp/x",
                   "task": {"type": "train", "backend": "megatron",
                            "entrypoint": "./sga/train_gpt.py"},
                   "runner": {"type": "ssh"}},
    "sga": {
        "enabled": True,
        "cycle": {"start_iter": 0, "period": 8, "full_sync_interval": 0},
        "score": {"name": "abs_m"},
        "selection": {"name": "circular_topk", "ratio_percent": 30.0},
        "error_feedback": {"name": "radius_opt"},
        "backend": {"name": "indexed_sparse"},
    },
    "train": {"system": {"tensor_model_parallel_size": 1},
              "model": {"num_layers": 2},
              "data": {"data_path": "/tmp/x"}},
}


def _cfg_with(**sga_overrides):
    cfg = OmegaConf.create(_BASE)
    for k, v in sga_overrides.items():
        cfg.sga[k] = v
    return cfg


def _expect_error(cfg, needle):
    try:
        sga_section_to_config(cfg)
    except (ValueError, AssertionError) as e:
        assert needle in str(e), f"missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


def test_loader_shadow_validation():
    out = sga_section_to_config(_cfg_with(shadow=True))
    assert out["shadow"] is True and "shadow_score" not in out  # default on

    out = sga_section_to_config(_cfg_with(shadow=False))
    assert "shadow" not in out  # explicit false = plain live, no shadow key

    out = sga_section_to_config(_cfg_with(shadow=True, shadow_score=False))
    assert out["shadow_score"] is False

    _expect_error(_cfg_with(shadow="yes"), "boolean")
    _expect_error(_cfg_with(shadow_score=False), "sga.shadow is not true")
    _expect_error(_cfg_with(shadow=True, shadow_score="no"), "boolean")

    cfg = OmegaConf.create(_BASE)
    cfg.sga.enabled = False
    cfg.sga.score = None
    cfg.sga.selection = None
    cfg.sga.error_feedback = None
    cfg.sga.backend = {"name": "dense"}
    cfg.sga.shadow = True
    _expect_error(cfg, "enabled is false")
    print("PASS test_loader_shadow_validation")


def test_shadow_rng_guard_cuda_half_raises():
    """The RNG guard's CUDA half: a boundary insertion that consumes a CUDA
    RNG stream fails the assert even when the CPU stream stays clean (the
    CPU-only test is structurally blind to this half on CPU-only test envs —
    exercised via monkeypatched torch.cuda state accessors)."""
    torch.manual_seed(3)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    score = build_score("abs_m", {"name": "abs_m"})
    selection = build_selection("circular_topk",
                                {"ratio_percent": 30.0}, score=score)
    ef = build_ef("radius_opt", {})
    eng = SGAEngine({"cycle": {"start_iter": 0, "period": 1,
                               "full_sync_interval": 0},
                     "shadow": True}, score, selection, ef,
                    _RADIUS["backend"])
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
    print("PASS test_shadow_rng_guard_cuda_half_raises")


def test_shadow_local_score_consumes_raw_bucket_views():
    """Non-synced scores (scoring_input='local_or_merged') skip readonly_score
    entirely: the selection consumes the raw bucket gradient views — at the
    shadow call point those hold the dense-synced, stock-clipped gradient
    (the honest dense-replay input)."""
    import sga.engine as engine_mod

    torch.manual_seed(6)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    # abs_g is local_or_merged; threshold selection operates elementwise
    eng = _build_shadow_engine(
        _ORIGIN, cycle={"start_iter": 0, "period": 1, "full_sync_interval": 0},
        shadow_score=True)
    eng.attach(chunk)

    def no_readonly(rec, emit_name, scratch=None):
        raise AssertionError(
            f"readonly_score must not run for local_or_merged scores "
            f"(got emit {emit_name!r})"
        )

    seen_views = []
    real_select = eng.selection.select

    def select_spy(ctx):
        seen_views.append(list(ctx.grad_views))
        return real_select(ctx)

    eng.selection.select = select_spy
    engine_mod.readonly_score = no_readonly
    try:
        gen = torch.Generator().manual_seed(9)
        with _ReplicaWorld(world=1):
            _push_grads(chunk, gen)
            chunk.finish_grad_sync(force_all_reduce=False)
            ok, _, _ = eng.optimizer_step(mp, [chunk], 1)
            assert ok
    finally:
        from sga.shadow import readonly_score as fresh_readonly
        engine_mod.readonly_score = fresh_readonly
        eng.selection.select = real_select
    assert len(seen_views) == 1, "reselection must run at the step-1 boundary"
    # raw bucket views: same storage as the engine's own meta-ordered spans
    # (bucket params_list order, not chunk.params order)
    _, _, meta = eng._selection_views()
    assert len(meta) == len(seen_views[0])
    for (bs, start, end, _p), v in zip(meta, seen_views[0]):
        assert v.data_ptr() == bs.grad_data[start:end].data_ptr(), \
            "local score must see the raw bucket grad views"
    print("PASS test_shadow_local_score_consumes_raw_bucket_views")


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


def test_shadow_supported_emits_and_build_gate():
    """SHADOW_SUPPORTED_EMITS is the single source of truth: every advertised
    emit has a formula (existence pin; value drift is pinned by
    test_readonly_score_matches_ef_torch_reference), unknown names name
    the supported set, and the build-time gate rejects a shadow run over an
    unsupported synced emit at engine construction instead of mid-run."""
    from sga.plugins import get_ef_decl, get_score_decl, register_default_plugins
    from sga.runtime import _assert_shadow_emits_supported
    from sga.shadow import SHADOW_SUPPORTED_EMITS

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
    for emit in sorted(SHADOW_SUPPORTED_EMITS):
        out = readonly_score(rec, emit)
        assert out.shape == (n,) and out.dtype == torch.float32, emit
    try:
        readonly_score(rec, "bogus")
    except ValueError as e:
        assert "bogus" in str(e) and "supported" in str(e), e
    else:
        raise SystemExit("unknown emit must be rejected")

    register_default_plugins()
    # emit-set relation pin: the shadow set is independently maintained from
    # the radius EF's emittable_scores; the two must stay equal.
    assert get_ef_decl("radius_opt").emittable_scores == SHADOW_SUPPORTED_EMITS
    decl = get_score_decl("abs_m")
    _assert_shadow_emits_supported({"shadow": True}, decl, "abs_m")
    _assert_shadow_emits_supported({}, decl, "whatever")  # live: not gated
    _assert_shadow_emits_supported({"shadow": False}, decl, "whatever")
    try:
        _assert_shadow_emits_supported({"shadow": True}, decl, "bogus")
    except ValueError as e:
        assert "shadow mode" in str(e) and "bogus" in str(e), e
    else:
        raise SystemExit("shadow build gate must reject unsupported emits")
    print("PASS test_shadow_supported_emits_and_build_gate")


def test_shadow_activates_without_idx_materialization():
    """VRAM gate: an idx-consuming backend under shadow never compacts the
    support — grad sync stays stock so no consumer exists, and the compacted
    pool would be resident waste. Mask-derived counts still resolve (both
    consistency layers are mask-based)."""
    torch.manual_seed(11)
    chunk = _FakeChunk()
    mp = _ShadowCapableMPOpt(chunk)
    assert _RADIUS["backend"].decl.consumes == "idx"
    eng = _build_shadow_engine(
        _RADIUS, cycle={"start_iter": 0, "period": 1, "full_sync_interval": 0},
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


def main():
    test_shadow_replicates_stock_call_sequence()
    test_shadow_numerics_equal_stock_loop()
    test_shadow_found_inf_full_clear_and_skip()
    test_shadow_rng_guard_raises()
    test_readonly_score_matches_ef_torch_reference()
    test_shadow_sticky_gets_pre_switch_shadow_support()
    test_shadow_peek_matches_stock_fusedadam_step()
    test_shadow_peek_value_tracks_upcoming_commit_torch_adamw()
    test_shadow_grad_sync_returns_false()
    test_shadow_probe_config_never_builds_records_or_state()
    test_shadow_metrics_reselected_tracks_boundary_reselection()
    test_loader_shadow_validation()
    test_shadow_rng_guard_cuda_half_raises()
    test_shadow_local_score_consumes_raw_bucket_views()
    test_readonly_score_grad_none_guard()
    test_shadow_supported_emits_and_build_gate()
    test_shadow_activates_without_idx_materialization()
    print("\nall shadow unit tests passed")


if __name__ == "__main__":
    main()
