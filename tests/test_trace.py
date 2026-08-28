#!/usr/bin/env python3
"""Unit tests for the engine-side trace collector (run: python tests/test_trace.py).

Covers: branch-point parsing, canonical (module.-stripped) catalog keys, the
frozen dump slot (after count_zeros / clip, BEFORE the EF — pinned by a
call-order spy), absolute (resume-safe) step numbering, per-group
hyperparameter meta, state_absent semantics, window gating, the rank gate,
and the zero-overhead-when-off structural contract.
"""
import os
import sys
import tempfile

import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.engine import SGAEngine
from sga.plugins import build_ef, build_score, build_selection
from sga.trace import TraceCollector, _canonical_name, parse_branch_points

from test_engine_unit import (
    _FakeChunk,
    _FakeMPOpt,
    _ORIGIN,
    _ReplicaWorld,
)


def _build_traced_engine(trace_cfg, chunk):
    score = build_score(_ORIGIN["score"], {"name": _ORIGIN["score"]})
    selection = build_selection(_ORIGIN["selection"], dict(_ORIGIN["selection_params"]),
                                score=score)
    ef = build_ef(_ORIGIN["ef"], dict(_ORIGIN["ef_params"]))
    eng = SGAEngine({"cycle": _ORIGIN["cycle"], "trace": trace_cfg},
                    score, selection, ef, _ORIGIN["backend"])
    eng._dp_cp_group = lambda: None
    eng.attach(chunk)
    return eng


def _prefixed_chunk():
    """Fake chunk whose module names carry the legacy DDP/Float16 wrapping."""
    chunk = _FakeChunk()
    chunk.module = SimpleNamespace(
        named_parameters=lambda: [
            (f"module.module.p{i}", p) for i, p in enumerate(chunk.params)
        ]
    )
    return chunk


def _drive_steps(eng, chunk, mp, steps):
    """Emulate backward + grad_sync + optimizer_step for the given absolute
    iteration numbers (0-indexed, like the fork's train_step call point)."""
    with _ReplicaWorld(world=1):
        for step in steps:
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            ok, grad_norm, _nz = eng.optimizer_step(mp, [chunk], step)
            assert ok is True and grad_norm is not None


def test_parse_branch_points():
    assert parse_branch_points(None) == []
    assert parse_branch_points("") == []
    assert parse_branch_points("50, 10,50") == [10, 50]
    assert parse_branch_points([200, 7]) == [7, 200]
    assert parse_branch_points([10.0, 2]) == [2, 10]  # integral floats fine
    try:
        parse_branch_points([-3])
    except ValueError as e:
        assert ">= 0" in str(e)
    else:
        raise SystemExit("negative branch point must be rejected")
    # non-integral specs: int() would silently truncate and land the window
    # on the wrong step
    for bad in ([10.5], "10.5", [None]):
        try:
            parse_branch_points(bad)
        except ValueError as e:
            assert "must be an integer" in str(e), e
        else:
            raise SystemExit(f"non-integral branch point accepted: {bad!r}")
    print("PASS test_parse_branch_points")


def test_canonical_name_strips_module_prefixes():
    assert _canonical_name("module.module.x.y") == "x.y"
    assert _canonical_name("module.x") == "x"
    assert _canonical_name("x.y") == "x.y"
    print("PASS test_canonical_name_strips_module_prefixes")


def test_zero_overhead_when_off_structural():
    """No sga.trace config -> the engine holds _trace = None (the per-step
    slot cost is a single None-check)."""
    chunk = _FakeChunk()
    score = build_score(_ORIGIN["score"], {"name": _ORIGIN["score"]})
    selection = build_selection(_ORIGIN["selection"], dict(_ORIGIN["selection_params"]),
                                score=score)
    ef = build_ef(_ORIGIN["ef"], dict(_ORIGIN["ef_params"]))
    eng = SGAEngine({"cycle": _ORIGIN["cycle"]}, score, selection, ef,
                    _ORIGIN["backend"])
    assert eng._trace is None
    eng2 = _build_traced_engine(
        {"dir": "/tmp/sga-trace-never", "branch_points": [], "window": 100}, chunk)
    assert isinstance(eng2._trace, TraceCollector)
    print("PASS test_zero_overhead_when_off_structural")


def test_dump_slot_before_ef_and_post_clip():
    """The dump fires inside optimizer_step AFTER the masked clip (payload is
    the clipped gradient) and BEFORE the EF's inner step (call-order spy)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            {"dir": td, "branch_points": [0], "window": 1}, chunk)

        calls = {"ef": 0, "dump": None}
        orig_full = eng.ef.full_step
        orig_sparse = eng.ef.sparse_step

        def counting_full(ctx):
            calls["ef"] += 1
            return orig_full(ctx)

        def counting_sparse(ctx):
            calls["ef"] += 1
            return orig_sparse(ctx)

        eng.ef.full_step = counting_full
        eng.ef.sparse_step = counting_sparse
        orig_dump = eng._trace.maybe_dump

        def spy(engine, mp_opt, step, grad_norm):
            calls["dump"] = (step, calls["ef"], grad_norm)
            return orig_dump(engine, mp_opt, step, grad_norm)

        eng._trace.maybe_dump = spy
        # capture the post-alias PRE-clip grads via a prepare_grads spy (the
        # clip then scales the aliased bucket storage in place)
        orig_prepare = mp.prepare_grads
        captured = {}

        def prepare_spy():
            found_inf = orig_prepare()
            captured["grads"] = [m.grad.clone() for m in mp.main_params]
            return found_inf

        mp.prepare_grads = prepare_spy
        with _ReplicaWorld(world=1):
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            ok, grad_norm, _ = eng.optimizer_step(mp, [chunk], 0)
            assert ok

        # slot ordering: dump ran at step 0 with zero EF calls so far
        assert calls["dump"] is not None and calls["dump"][:2] == (0, 0)
        assert calls["ef"] == 1  # exactly one inner step happened after
        # the spy saw the engine's own clip result
        assert calls["dump"][2] == grad_norm

        # content: dumped g == the post-clip grad, elementwise-exact
        pre = captured["grads"]
        norm_sq = torch.zeros(1, dtype=torch.float32)
        for g in pre:
            norm_sq += g.float().pow(2).sum()
        coeff = 1.0 / (float(norm_sq.sqrt()) + 1.0e-6)
        assert coeff < 1.0  # random grads: clip must be active here
        dumped = torch.load(os.path.join(td, "step_000000_g.pt"),
                            weights_only=False)
        assert dumped["meta"]["step"] == 0
        assert dumped["meta"]["grad_norm"] == float(norm_sq.sqrt())
        for i, g in enumerate(pre):
            want = g * coeff
            got = dumped["payload"][f"p{i}"]
            assert torch.equal(got, want), f"p{i}: dump is not the clipped grad"
    print("PASS test_dump_slot_before_ef_and_post_clip")


def test_dump_contents_window_and_state():
    """Branch state dump (w/m/v/g + state_absent), window gating of the
    per-step g files, canonical catalog keys, per-group meta."""
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as td:
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        # two optimizer groups so the additive per-group meta is observable
        mp.optimizer = torch.optim.Adam(
            [{"params": [mp.main_params[0]], "lr": 1e-3, "weight_decay": 1e-2},
             {"params": [mp.main_params[1]], "lr": 2e-3, "weight_decay": 0.0}],
            betas=(0.9, 0.95), eps=1e-8)
        eng = _build_traced_engine(
            {"dir": td, "branch_points": [0, 2], "window": 2}, chunk)
        # capture the moments exactly as the dump slot sees them (post-run
        # rec._state has later commits applied and cannot be compared against)
        seen = {}
        orig_dump = eng._trace.maybe_dump

        def spy(engine, mp_opt, step, grad_norm):
            if step == 2:
                seen["m"] = [r._state["exp_avg"].clone() for r in engine._records]
                seen["v"] = [r._state["exp_avg_sq"].clone()
                             for r in engine._records]
            return orig_dump(engine, mp_opt, step, grad_norm)

        eng._trace.maybe_dump = spy
        _drive_steps(eng, chunk, mp, [0, 1, 2, 3])

        files = set(os.listdir(td))
        # branch at 0 covers g for steps 0,1; branch at 2 covers g for 2,3
        assert "catalog.pt" in files
        assert {"step_000000_g.pt", "step_000001_g.pt",
                "step_000002_g.pt", "step_000003_g.pt"} <= files
        assert {"step_000000_state.pt", "step_000002_state.pt"} <= files
        assert "step_000004_g.pt" not in files

        cat = torch.load(os.path.join(td, "catalog.pt"), weights_only=False)
        assert set(cat["catalog"]) == {"p0", "p1"}  # module.* stripped
        for name, entry in cat["catalog"].items():
            assert entry["shape"] == [4, 16]
            assert entry["dtype"] == "torch.float32"
            assert entry["param_group"] == (0 if name == "p0" else 1)

        st0 = torch.load(os.path.join(td, "step_000000_state.pt"),
                         weights_only=False)
        assert st0["meta"]["step"] == 0
        assert st0["meta"]["state_absent"] is True  # pre-first-commit
        assert st0["meta"]["lr"] == [1e-3, 2e-3]
        assert st0["meta"]["param_groups"][0]["weight_decay"] == 1e-2
        assert st0["meta"]["param_groups"][1]["weight_decay"] == 0.0
        assert st0["meta"]["param_groups"][1]["adam_beta2"] == 0.95
        for name in ("p0", "p1"):
            assert torch.equal(st0["payload"]["m"][name],
                               torch.zeros(4, 16))
            assert name in st0["payload"]["g"]

        st2 = torch.load(os.path.join(td, "step_000002_state.pt"),
                         weights_only=False)
        assert st2["meta"]["state_absent"] is False
        # moments after two commits, as captured at the dump slot
        for i, name in enumerate(("p0", "p1")):
            assert torch.equal(st2["payload"]["m"][name], seen["m"][i])
            assert torch.equal(st2["payload"]["v"][name], seen["v"][i])
            assert st2["payload"]["m"][name].abs().sum() > 0  # real moments
    print("PASS test_dump_contents_window_and_state")


def test_absolute_numbering_resume_safe():
    """A run resumed at iteration K dumps step_00000K_* (never step_000000_*)
    and never rewrites an existing catalog.pt — but a catalog that does not
    match this run (a foreign model/config) is rejected before any step file
    is written: a stale catalog combined with new payloads would silently
    poison replay."""
    torch.manual_seed(2)
    with tempfile.TemporaryDirectory() as td:
        # resume-compatible catalog: exactly what _prefixed_chunk()'s run
        # would write (canonical names, shapes, dtypes, param_group)
        sentinel = os.path.join(td, "catalog.pt")
        torch.save({"meta": {"origin": "first-process"},
                    "catalog": {f"p{i}": {"shape": [4, 16],
                                          "dtype": "torch.float32",
                                          "param_group": 0}
                                for i in range(2)}},
                   sentinel)
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            {"dir": td, "branch_points": [10], "window": 2}, chunk)
        # resume: the first optimizer_step this process sees is iteration 10
        _drive_steps(eng, chunk, mp, [10, 11])
        files = set(os.listdir(td))
        assert "step_000010_state.pt" in files
        assert "step_000010_g.pt" in files and "step_000011_g.pt" in files
        assert not any(f.startswith("step_000000") for f in files)
        cat = torch.load(sentinel, weights_only=False)
        assert cat["meta"]["origin"] == "first-process"  # untouched
    with tempfile.TemporaryDirectory() as td:
        torch.save({"meta": {}, "catalog": {
            "p0": {"shape": [4, 16], "dtype": "torch.float32",
                   "param_group": 0},
            "p1": {"shape": [8, 16], "dtype": "torch.float32",
                   "param_group": 0},  # wrong shape: foreign run
        }}, os.path.join(td, "catalog.pt"))
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            {"dir": td, "branch_points": [10], "window": 2}, chunk)
        try:
            _drive_steps(eng, chunk, mp, [10])
            raise AssertionError("mismatched catalog.pt accepted")
        except ValueError as e:
            assert "does not match this run" in str(e), e
            assert "shape" in str(e) and "p1" in str(e), e
        assert not any(f.startswith("step_") for f in os.listdir(td))
    print("PASS test_absolute_numbering_resume_safe")


def test_bind_param_group_contract():
    """_bind raises on param/group structural gaps (same discipline as the
    unnamed/duplicate-name raises): param_group=None would silently drop
    replay onto the legacy mean-lr fallback, and a main param in multiple
    groups has an ambiguous group index (plain dict assignment would keep
    the last one)."""
    class _Rec:
        def __init__(self, model_param, main_param):
            self.model_param = model_param
            self.main_param = main_param

    model_p = torch.zeros(4, 4)
    main_p = model_p.clone()
    chunk = SimpleNamespace(
        module=SimpleNamespace(named_parameters=lambda: [("p0", model_p)]))
    rec = _Rec(model_p, main_p)
    eng = SimpleNamespace(_chunks={"c": chunk}, _records=[rec])
    collector = TraceCollector(
        {"dir": "/tmp/sga-trace-never", "branch_points": [], "window": 1})

    # main param in NO optimizer group
    mp_opt = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[]))
    try:
        collector._bind(eng, mp_opt)
        raise AssertionError("main param missing from every group accepted")
    except ValueError as e:
        assert "not found in any optimizer param_group" in str(e), e
        assert "(4, 4)" in str(e), e

    # main param in TWO groups (would silently keep the last index)
    mp_opt = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[
        {"params": [main_p]}, {"params": [main_p]}]))
    try:
        collector._bind(eng, mp_opt)
        raise AssertionError("main param in multiple groups accepted")
    except ValueError as e:
        assert "multiple optimizer param_groups" in str(e), e

    # healthy bind records the group index
    mp_opt = SimpleNamespace(optimizer=SimpleNamespace(
        param_groups=[{"params": [main_p]}]))
    collector._bind(eng, mp_opt)
    assert collector._entries == [("p0", rec, 0)]
    print("PASS test_bind_param_group_contract")


def test_rank_gate():
    """Only world-rank-0 writes; other ranks never touch the directory."""
    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as td:
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            {"dir": td, "branch_points": [0], "window": 1}, chunk)
        eng._rank = 1
        _drive_steps(eng, chunk, mp, [0])
        assert os.listdir(td) == []
    print("PASS test_rank_gate")


def test_meta_carries_engine_phase():
    """meta['phase'] self-describes the producing engine mode: live_full /
    live_sparse on the live path (the bootstrap region runs full — no
    support exists until the first reselect at step > start_iter — then the
    post-reselect step runs sparse under the active support), 'shadow' in
    shadow mode. A mixed trace directory stops being an inference exercise."""
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            {"dir": td, "branch_points": [0], "window": 3}, chunk)
        _drive_steps(eng, chunk, mp, [0, 1, 2])
        phases = {}
        for s in range(3):
            m = torch.load(os.path.join(td, f"step_{s:06d}_g.pt"),
                           weights_only=False)["meta"]
            phases[s] = m["phase"]
        assert phases[0] == "live_full", phases  # bootstrap: no support yet
        assert phases[1] == "live_full", phases  # step 0 had no reselect
        assert phases[2] == "live_sparse", phases  # post-reselect support
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        from sga.plugins import build_ef, build_score, build_selection
        from test_engine_unit import _ORIGIN
        from test_shadow_unit import _ShadowCapableMPOpt
        score = build_score(_ORIGIN["score"], {"name": _ORIGIN["score"]})
        selection = build_selection(_ORIGIN["selection"],
                                    dict(_ORIGIN["selection_params"]),
                                    score=score)
        ef = build_ef(_ORIGIN["ef"], dict(_ORIGIN["ef_params"]))
        eng = SGAEngine({"cycle": _ORIGIN["cycle"], "shadow": True,
                         "trace": {"dir": td, "branch_points": [0],
                                   "window": 1}},
                        score, selection, ef, _ORIGIN["backend"])
        eng._dp_cp_group = lambda: None
        eng.attach(chunk)
        mp = _ShadowCapableMPOpt(chunk)
        with _ReplicaWorld(world=1):
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            assert eng.grad_sync(chunk, force_all_reduce=False) is False
            ok, _, _ = eng.optimizer_step(mp, [chunk], 0)
            assert ok
        m = torch.load(os.path.join(td, "step_000000_g.pt"),
                       weights_only=False)["meta"]
        assert m["phase"] == "shadow", m["phase"]
    print("PASS test_meta_carries_engine_phase")


def main():
    test_parse_branch_points()
    test_canonical_name_strips_module_prefixes()
    test_zero_overhead_when_off_structural()
    test_dump_slot_before_ef_and_post_clip()
    test_dump_contents_window_and_state()
    test_absolute_numbering_resume_safe()
    test_bind_param_group_contract()
    test_rank_gate()
    test_meta_carries_engine_phase()
    print("\nall trace tests passed")


if __name__ == "__main__":
    main()
