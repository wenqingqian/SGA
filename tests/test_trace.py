#!/usr/bin/env python3
"""Unit tests for the engine-side trace collector (run: python tests/test_trace.py).

Covers: branch-point parsing, canonical (module.-stripped) catalog keys, the
frozen dump slot (after count_zeros / clip, BEFORE the EF — pinned by a
call-order spy), absolute (resume-safe) step numbering, per-group
hyperparameter meta, state_absent semantics, window gating, the rank gate,
the zero-overhead-when-off structural contract, and the exception-safe
compact dump-window restore.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.engine import SGAEngine
from sga.plugins import build_backend_plugin, build_optimizer, build_selection
from sga.trace import (
    TraceCollector,
    _AsyncGradWriter,
    _canonical_name,
    _values_match,
    parse_branch_points,
)

from test_engine_unit import (
    _FakeChunk,
    _FakeMPOpt,
    _ReplicaWorld,
    _UPDATE,
    _seed_combo,
)


def _build_traced_engine(dir, branch_points, window, chunk,
                         config_extra=None, backend=None):
    _seed_combo(_UPDATE, {**(config_extra or {}),
                          "sga_trace_dir": dir,
                          "sga_trace_branch_points": branch_points,
                          "sga_trace_window": window})
    selection = build_selection(_UPDATE["selection"])
    opt = build_optimizer(_UPDATE["opt"])
    eng = SGAEngine(_UPDATE["score"], selection, opt,
                    build_backend_plugin(backend or _UPDATE["backend"]))
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
    _seed_combo(_UPDATE)
    selection = build_selection(_UPDATE["selection"])
    opt = build_optimizer(_UPDATE["opt"])
    eng = SGAEngine(_UPDATE["score"], selection, opt,
                    build_backend_plugin(_UPDATE["backend"]))
    assert eng._trace is None
    eng2 = _build_traced_engine("/tmp/sga-trace-never", [], 100, chunk)
    assert isinstance(eng2._trace, TraceCollector)
    print("PASS test_zero_overhead_when_off_structural")


def test_dump_slot_before_optimizer_and_post_clip():
    """The dump fires inside optimizer_step AFTER the masked clip (payload is
    the clipped gradient) and BEFORE the optimizer template's inner step
    (call-order spy)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 1, chunk)

        calls = {"opt": 0, "dump": None}
        orig_full = eng.opt.full_step
        orig_sparse = eng.opt.sparse_step

        def counting_full(ctx):
            calls["opt"] += 1
            return orig_full(ctx)

        def counting_sparse(ctx):
            calls["opt"] += 1
            return orig_sparse(ctx)

        eng.opt.full_step = counting_full
        eng.opt.sparse_step = counting_sparse
        orig_dump = eng._trace.maybe_dump

        def spy(engine, mp_opt, step, grad_norm):
            calls["dump"] = (step, calls["opt"], grad_norm)
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

        # slot ordering: dump ran at step 0 with zero optimizer calls so far
        assert calls["dump"] is not None and calls["dump"][:2] == (0, 0)
        assert calls["opt"] == 1  # exactly one inner step happened after
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
    print("PASS test_dump_slot_before_optimizer_and_post_clip")


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
        eng = _build_traced_engine(td, [0, 2], 2, chunk)
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
        eng = _build_traced_engine(td, [10], 2, chunk)
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
        eng = _build_traced_engine(td, [10], 2, chunk)
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
        dir="/tmp/sga-trace-never", branch_points=[], window=1)

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
        eng = _build_traced_engine(td, [0], 1, chunk)
        eng._rank = 1
        _drive_steps(eng, chunk, mp, [0])
        assert os.listdir(td) == []
    print("PASS test_rank_gate")


def test_meta_carries_engine_phase():
    """meta['phase'] self-describes the producing engine mode: live_full /
    live_sparse on the live path (the bootstrap region runs full — no
    support exists until the first reselect at step > start_iter — then the
    post-reselect mid-cycle step runs sparse under the active support),
    'shadow' in shadow mode. A mixed trace directory stops being an
    inference exercise."""
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        # period 2: reselect at every even boundary (steps 2, 4); odd steps
        # after the first reselect are sparse
        eng = _build_traced_engine(
            td, [0], 4, chunk,
            config_extra={"sga_cycle_start_iter": 0,
                          "sga_cycle_period": 2})
        _drive_steps(eng, chunk, mp, [0, 1, 2, 3])
        phases = {}
        for s in range(4):
            m = torch.load(os.path.join(td, f"step_{s:06d}_g.pt"),
                           weights_only=False)["meta"]
            phases[s] = m["phase"]
        assert phases[0] == "live_full", phases  # bootstrap: no support yet
        assert phases[1] == "live_full", phases  # still no support (no boundary)
        assert phases[2] == "live_full", phases  # boundary: full + reselect
        assert phases[3] == "live_sparse", phases  # post-reselect support
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        from test_shadow_unit import _ShadowCapableMPOpt
        _seed_combo(_UPDATE, {"sga_shadow": True, "sga_trace_dir": td,
                              "sga_trace_branch_points": [0],
                              "sga_trace_window": 1})
        selection = build_selection(_UPDATE["selection"])
        opt = build_optimizer(_UPDATE["opt"])
        eng = SGAEngine(
            _UPDATE["score"], selection, opt,
            build_backend_plugin(_UPDATE["backend"]))
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


def test_sparse_dump_dualpath_content():
    """Dual-path, content level: on a DUMPING sparse step (trace
    window active) the engine applies the clip explicitly engine-side —
    payload = where(mask, g*coeff, g), inactives raw — and hands the
    optimizer apply_coeff == 1.0 (no double scale). A regression that
    scales the full layout, scales nothing, or re-defers the coeff on dump
    steps fails here and would silently corrupt offline replay inputs."""
    torch.manual_seed(5)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        # period 2: steps 0/1 bootstrap full, 2 boundary full + reselect,
        # 3 the sparse step inside the [0, 4) dump window
        eng = _build_traced_engine(
            td, [0], 4, chunk,
            config_extra={"sga_cycle_start_iter": 0,
                          "sga_cycle_period": 2})

        seen = {}
        orig_sparse = eng.opt.sparse_step

        def spy_sparse(ctx):
            seen["apply_coeff"] = ctx.apply_coeff
            return orig_sparse(ctx)

        eng.opt.sparse_step = spy_sparse
        orig_prepare = mp.prepare_grads
        captured = {}
        arming = {"next": False}

        def prepare_spy():
            found_inf = orig_prepare()
            if arming["next"]:
                captured["grads"] = [m.grad.clone() for m in mp.main_params]
                arming["next"] = False
            return found_inf

        mp.prepare_grads = prepare_spy

        with _ReplicaWorld(world=1):
            for it in (0, 1, 2):
                off = 0
                for p in chunk.params:
                    n = p.numel()
                    chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                    off += n
                assert eng.grad_sync(chunk, force_all_reduce=False) is True
                ok, _, _ = eng.optimizer_step(mp, [chunk], it)
                assert ok
            mask = eng._bucket_states[0].active_mask_u8
            assert mask is not None          # reselect happened at step 2
            mask = mask.clone()
            arming["next"] = True
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            ok, gn, _ = eng.optimizer_step(mp, [chunk], 3)
            assert ok

        coeff = 1.0 / (gn + 1.0e-6)
        assert coeff < 1.0                  # random grads: clip is active
        assert seen["apply_coeff"] == 1.0, seen   # engine scaled, not kernel

        dumped = torch.load(os.path.join(td, "step_000003_g.pt"),
                            weights_only=False)["payload"]
        m = mask > 0
        off = 0
        for i, p in enumerate(chunk.params):
            n = p.numel()
            g = captured["grads"][i]
            want = torch.where(m[off:off + n], g.view(-1) * coeff,
                               g.view(-1)).view_as(g)
            assert torch.equal(dumped[f"p{i}"], want), (
                f"p{i}: sparse dump is not the masked-clipped grad"
            )
            off += n
    print("PASS test_sparse_dump_dualpath_content")


def test_sparse_dump_compact_rebuild_content():
    """Compact emission, dump-window content: on a DUMPING
    sparse step the engine scales the compact values in-place, scatters
    them into the full grad layout around the dump (the slot reads the
    same post-clip full view the writeback path produced), hands the
    optimizer apply_coeff == 1.0, and RESTORES the selected-coords-zero
    invariant before grad_correct (the pack's clear does not repeat —
    the restore pass is the only thing keeping the end state clean)."""
    torch.manual_seed(5)

    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        # same cadence as test_sparse_dump_dualpath_content: period 2,
        # steps 0/1/2 full, 3 the sparse step inside the [0, 4) window
        eng = _build_traced_engine(
            td, [0], 4, chunk,
            config_extra={"sga_cycle_start_iter": 0,
                          "sga_cycle_period": 2},
            backend="torch_pipelined_index")

        seen = {}
        orig_sparse = eng.opt.sparse_step

        def spy_sparse(ctx):
            seen["apply_coeff"] = ctx.apply_coeff
            seen["compact"] = ctx.compact_values is not None
            return orig_sparse(ctx)

        eng.opt.sparse_step = spy_sparse

        with _ReplicaWorld(world=1):
            for it in (0, 1, 2):
                off = 0
                for p in chunk.params:
                    n = p.numel()
                    chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                    off += n
                assert eng.grad_sync(chunk, force_all_reduce=False) is True
                assert eng.optimizer_step(mp, [chunk], it)[0]
            bs = eng._bucket_states[0]
            mask = bs.active_mask_u8
            assert mask is not None          # reselect happened at step 2
            mask = mask.clone()
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            pre = bs.grad_data.clone()       # pre-communication full layout
            assert eng.grad_sync(chunk, force_all_reduce=False) is True
            vals_pre = eng._compact_values[id(bs)].clone()
            ok, gn, _ = eng.optimizer_step(mp, [chunk], 3)
            assert ok

        coeff = 1.0 / (gn + 1.0e-6)
        assert coeff < 1.0                  # random grads: clip is active
        assert seen["apply_coeff"] == 1.0, seen   # engine scaled, not kernel
        assert seen["compact"] is True, seen      # compact context reached opt

        dumped = torch.load(os.path.join(td, "step_000003_g.pt"),
                            weights_only=False)["payload"]
        m = mask > 0
        # the full-layout post-clip view: selected = scaled values, the
        # residual raw (exactly what the writeback path dumps)
        want_full = torch.zeros_like(pre)
        want_full[m] = vals_pre * coeff
        want_full[~m] = pre[~m]
        off = 0
        for i, p in enumerate(chunk.params):
            n = p.numel()
            assert torch.equal(dumped[f"p{i}"],
                               want_full[off:off + n].view_as(p)), (
                f"p{i}: compact dump is not the rebuilt post-clip view"
            )
            off += n
        # the restore pass re-zeroed the scattered coords: the end state
        # keeps the pack's invariant (consumption leaves no stale values)
        assert torch.count_nonzero(bs.grad_data[m]).item() == 0
        assert torch.equal(bs.grad_data[~m], pre[~m])
    print("PASS test_sparse_dump_compact_rebuild_content")


def test_sparse_dump_compact_restore_on_dump_exception():
    """Compact dump-window exception safety: when maybe_dump raises
    inside the rebuild window, the exception must propagate AND the
    scattered values must be re-zeroed first — the restore runs in a
    finally, so a failed dump cannot leak step-scoped state (selected
    coords non-zero) into the next backward's accumulation."""
    torch.manual_seed(6)

    class _DumpFailure(RuntimeError):
        pass

    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        # same cadence as test_sparse_dump_compact_rebuild_content: period 2,
        # steps 0/1/2 full, 3 the sparse step inside the [0, 4) window
        eng = _build_traced_engine(
            td, [0], 4, chunk,
            config_extra={"sga_cycle_start_iter": 0,
                          "sga_cycle_period": 2},
            backend="torch_pipelined_index")

        with _ReplicaWorld(world=1):
            for it in (0, 1, 2):
                off = 0
                for p in chunk.params:
                    n = p.numel()
                    chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                    off += n
                assert eng.grad_sync(chunk, force_all_reduce=False) is True
                assert eng.optimizer_step(mp, [chunk], it)[0]
            bs = eng._bucket_states[0]
            mask = bs.active_mask_u8
            assert mask is not None          # reselect happened at step 2
            sel = mask > 0
            rest = ~sel
            off = 0
            for p in chunk.params:
                n = p.numel()
                chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                off += n
            pre = bs.grad_data.clone()       # pre-communication full layout
            assert eng.grad_sync(chunk, force_all_reduce=False) is True

            def raising_dump(*a, **k):
                raise _DumpFailure("dump I/O exploded")

            eng._trace.maybe_dump = raising_dump
            try:
                eng.optimizer_step(mp, [chunk], 3)
                raise AssertionError("dump exception was swallowed")
            except _DumpFailure:
                pass
            # the finally restored the invariant BEFORE the exception flew
            assert torch.count_nonzero(bs.grad_data[sel]).item() == 0
            assert torch.equal(bs.grad_data[rest], pre[rest])
    print("PASS test_sparse_dump_compact_restore_on_dump_exception")


def test_dump_dtype_bf16_ab_convert():
    """--sga-trace-dump-dtype bf16: the g payloads are the per-tensor
    round-to-nearest bf16 of the same fp32 dump — bit-identical to what
    sga.trace_convert produces from the fp32 trace (so a runtime-bf16 trace
    and a converted trace are interchangeable replay inputs); state files
    stay fp32; catalog self-describes via g_dtype; the resume dtype matrix
    is fully pinned (refusals name g_dtype; the explicit-present-and-equal
    bf16/bf16 cell — the feature's crash-resume scenario — is accepted)."""
    from sga.trace_convert import convert
    with tempfile.TemporaryDirectory() as td:
        dirs = {
            "fp32": os.path.join(td, "a_fp32"),
            "bf16": os.path.join(td, "b_bf16"),
        }
        for tag, extra in (("fp32", {}),
                           ("bf16", {"sga_trace_dump_dtype": "bf16"})):
            torch.manual_seed(7)
            chunk = _prefixed_chunk()
            mp = _FakeMPOpt(chunk)
            eng = _build_traced_engine(dirs[tag], [0], 2, chunk,
                                       config_extra=extra)
            _drive_steps(eng, chunk, mp, [0, 1])

        ga = torch.load(os.path.join(dirs["fp32"], "step_000000_g.pt"),
                        weights_only=False)
        gb = torch.load(os.path.join(dirs["bf16"], "step_000000_g.pt"),
                        weights_only=False)
        for name, v in ga["payload"].items():
            w = gb["payload"][name]
            assert v.dtype == torch.float32
            assert w.dtype == torch.bfloat16
            assert torch.equal(w, v.to(torch.bfloat16))
        assert ga["meta"] == gb["meta"]  # dtype is storage-only

        st = torch.load(os.path.join(dirs["bf16"], "step_000000_state.pt"),
                        weights_only=False)
        for sect in ("w", "m", "v", "g"):
            for name in st["payload"][sect]:
                assert st["payload"][sect][name].dtype == torch.float32

        cat = torch.load(os.path.join(dirs["bf16"], "catalog.pt"),
                         weights_only=False)
        assert cat["g_dtype"] == "bf16"
        cat_fp32 = torch.load(os.path.join(dirs["fp32"], "catalog.pt"),
                              weights_only=False)
        assert cat_fp32["g_dtype"] == "fp32"

        conv_dir = os.path.join(td, "c_conv")
        convert(dirs["fp32"], conv_dir)
        gc = torch.load(os.path.join(conv_dir, "step_000000_g.pt"),
                        weights_only=False)
        for name in gb["payload"]:
            assert torch.equal(gc["payload"][name], gb["payload"][name])

        torch.manual_seed(7)
        chunk2 = _prefixed_chunk()
        mp2 = _FakeMPOpt(chunk2)
        eng2 = _build_traced_engine(dirs["bf16"], [0], 2, chunk2)
        try:
            _drive_steps(eng2, chunk2, mp2, [0])
        except ValueError as e:
            assert "g_dtype" in str(e)
        else:
            raise SystemExit("resume with a different dump dtype must fail")

        # remaining refusal cells of the resume matrix (same catalog, only
        # the dtype disagrees — the error must name g_dtype, not a catalog
        # field):
        # fp32 dir (g_dtype present = fp32) resumed by a bf16 run
        torch.manual_seed(7)
        chunk3 = _prefixed_chunk()
        mp3 = _FakeMPOpt(chunk3)
        eng3 = _build_traced_engine(
            dirs["fp32"], [0], 2, chunk3,
            config_extra={"sga_trace_dump_dtype": "bf16"})
        try:
            _drive_steps(eng3, chunk3, mp3, [0])
        except ValueError as e:
            assert "g_dtype" in str(e)
        else:
            raise SystemExit("fp32 dir resumed as bf16 must fail")
        # legacy dir (pre-g_dtype collector: no g_dtype key) resumed by a
        # bf16 run — the "fp32" default direction must refuse bf16 too
        legacy_dir = os.path.join(td, "legacy")
        os.makedirs(legacy_dir)
        torch.save({"meta": {}, "catalog": {
            f"p{i}": {"shape": [4, 16], "dtype": "torch.float32",
                      "param_group": 0} for i in range(2)}},
                   os.path.join(legacy_dir, "catalog.pt"))
        torch.manual_seed(7)
        chunk4 = _prefixed_chunk()
        mp4 = _FakeMPOpt(chunk4)
        eng4 = _build_traced_engine(
            legacy_dir, [0], 2, chunk4,
            config_extra={"sga_trace_dump_dtype": "bf16"})
        try:
            _drive_steps(eng4, chunk4, mp4, [0])
        except ValueError as e:
            assert "g_dtype" in str(e)
        else:
            raise SystemExit("legacy fp32 dir resumed as bf16 must fail")

        # the accept cell: an explicitly-tagged bf16 dir resumed by a bf16
        # run — the core crash-resume scenario; a regression that raises
        # whenever g_dtype is present would fail exactly here
        torch.manual_seed(7)
        chunk5 = _prefixed_chunk()
        mp5 = _FakeMPOpt(chunk5)
        eng5 = _build_traced_engine(
            dirs["bf16"], [0], 2, chunk5,
            config_extra={"sga_trace_dump_dtype": "bf16"})
        _drive_steps(eng5, chunk5, mp5, [0])  # must not raise
    print("PASS test_dump_dtype_bf16_ab_convert")


def test_dump_dtype_constructor_guard():
    """Direct-construction backstop: dump_dtype outside {fp32, bf16} is
    refused at the collector (engine tests seed config via _seed, which
    skips validation — on that route this branch is the only guard)."""
    try:
        TraceCollector(dir="/tmp/sga-trace-never", branch_points=[],
                       window=1, dump_dtype="fp16")
        raise AssertionError("invalid dump_dtype accepted")
    except ValueError as e:
        assert "dump_dtype" in str(e) and "fp16" in str(e)
    assert TraceCollector(dir="/tmp/sga-trace-never", branch_points=[],
                          window=1).dump_dtype == "fp32"  # default fp32
    print("PASS test_dump_dtype_constructor_guard")


# ----------------------------------------------------- async writer

class _WriterBoom(RuntimeError):
    """Sentinel writer failure (injected through torch.save)."""


def _drive_one(eng, chunk, mp, step):
    """One grad_sync + optimizer_step at an absolute iteration (the async
    tests need per-step control around the drain/latch timing)."""
    with _ReplicaWorld(world=1):
        off = 0
        for p in chunk.params:
            n = p.numel()
            chunk.bucket.grad_data[off:off + n] += torch.randn(n)
            off += n
        assert eng.grad_sync(chunk, force_all_reduce=False) is True
        ok, grad_norm, _ = eng.optimizer_step(mp, [chunk], step)
        assert ok is True and grad_norm is not None


def _tmp_files(dir):
    return sorted(f for f in os.listdir(dir) if ".tmp." in f)


def test_async_default_off_is_sync():
    """Default flags keep dumps synchronous and publish each file before
    optimizer_step returns."""
    torch.manual_seed(8)
    with tempfile.TemporaryDirectory() as td:
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 1, chunk)
        assert eng._trace.async_enabled is False
        _drive_steps(eng, chunk, mp, [0])
        assert eng._trace._writer is None      # lazy writer never spawned
        assert eng._trace.diagnostics() == {}  # sync mode has no stats face
        assert os.path.exists(os.path.join(td, "step_000000_g.pt"))
    print("PASS test_async_default_off_is_sync")


def test_async_parity_with_sync():
    """Same seeded run, sync vs async directories: identical file sets,
    bitwise-equal g payloads, equal meta and catalog — async changes WHEN
    the bytes land, never WHICH bytes (the fp32 and bf16 arms both hold;
    bf16 state files stay fp32 and the catalog self-describes g_dtype)."""
    from sga.trace_convert import convert
    with tempfile.TemporaryDirectory() as td:
        runs = {}
        for tag, extra in (
                ("fp32_sync", {}),
                ("fp32_async", {"sga_trace_async": True}),
                ("bf16_sync", {"sga_trace_dump_dtype": "bf16"}),
                ("bf16_async", {"sga_trace_dump_dtype": "bf16",
                                "sga_trace_async": True})):
            d = os.path.join(td, tag)
            torch.manual_seed(9)
            chunk = _prefixed_chunk()
            mp = _FakeMPOpt(chunk)
            eng = _build_traced_engine(d, [0, 2], 2, chunk,
                                       config_extra=extra)
            _drive_steps(eng, chunk, mp, [0, 1, 2, 3])
            runs[tag] = d

        for dtype in ("fp32", "bf16"):
            a, b = runs[f"{dtype}_sync"], runs[f"{dtype}_async"]
            fa, fb = set(os.listdir(a)), set(os.listdir(b))
            assert fa == fb, (fa, fb)
            assert not _tmp_files(b), _tmp_files(b)
            want_dt = torch.float32 if dtype == "fp32" else torch.bfloat16
            for fname in sorted(f for f in fa if f.endswith("_g.pt")):
                da = torch.load(os.path.join(a, fname), weights_only=False)
                db = torch.load(os.path.join(b, fname), weights_only=False)
                assert da["meta"] == db["meta"], fname
                assert set(da["payload"]) == set(db["payload"]), fname
                for name in da["payload"]:
                    assert da["payload"][name].dtype == want_dt
                    assert torch.equal(da["payload"][name],
                                       db["payload"][name]), (fname, name)
            for fname in sorted(f for f in fa if f.endswith("_state.pt")):
                sa = torch.load(os.path.join(a, fname), weights_only=False)
                sb = torch.load(os.path.join(b, fname), weights_only=False)
                assert sa["meta"] == sb["meta"], fname
                for sect in ("w", "m", "v", "g"):
                    for name in sa["payload"][sect]:
                        assert sa["payload"][sect][name].dtype == torch.float32
                        assert torch.equal(sa["payload"][sect][name],
                                           sb["payload"][sect][name])
            cat_a = torch.load(os.path.join(a, "catalog.pt"),
                               weights_only=False)
            cat_b = torch.load(os.path.join(b, "catalog.pt"),
                               weights_only=False)
            assert cat_a == cat_b
            assert cat_a["g_dtype"] == dtype

        # async bf16 stays interchangeable with the converted fp32 trace
        conv = os.path.join(td, "conv")
        convert(runs["fp32_sync"], conv)
        g_conv = torch.load(os.path.join(conv, "step_000000_g.pt"),
                            weights_only=False)
        g_ba = torch.load(os.path.join(runs["bf16_async"],
                                       "step_000000_g.pt"),
                          weights_only=False)
        for name in g_ba["payload"]:
            assert torch.equal(g_conv["payload"][name],
                               g_ba["payload"][name])
    print("PASS test_async_parity_with_sync")


def test_async_drains_at_last_target_step():
    """The last target gradient step drains INLINE: when its optimizer_step
    returns, every file of the run is on disk, no tmp survives, the writer
    thread is gone, and the diagnostics face reports the run."""
    torch.manual_seed(10)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 3, chunk,
                                   config_extra={"sga_trace_async": True})
        for step in (0, 1):
            _drive_one(eng, chunk, mp, step)
            assert not eng._trace._drained  # mid-window: no fence yet
        assert eng._trace._writer is not None
        _drive_one(eng, chunk, mp, 2)      # the last target step
        assert eng._trace._drained
        files = set(os.listdir(td))
        assert {"catalog.pt", "step_000000_state.pt",
                "step_000000_g.pt", "step_000001_g.pt",
                "step_000002_g.pt"} <= files, files
        assert not _tmp_files(td)
        assert not any(t.is_alive() for t in eng._trace._writer._threads)
        diag = eng._trace.diagnostics()
        assert diag["enqueued"] == 3 and diag["written"] == 3
        assert diag["error"] is None
        assert diag["peak_reserved_bytes"] <= 16 * (1 << 30)
        # native timers: one write per item, one real drain wait, fp32 run
        assert diag["write_n"] == 3 and diag["drain_n"] == 1, diag
        assert diag["cast_n"] == 0, diag
        # a multi-window run drains at the LAST window's end, not earlier
        torch.manual_seed(10)
        td2 = tempfile.mkdtemp(dir=td)
        chunk2 = _FakeChunk()
        mp2 = _FakeMPOpt(chunk2)
        eng2 = _build_traced_engine(td2, [0, 5], 2, chunk2,
                                    config_extra={"sga_trace_async": True})
        _drive_steps(eng2, chunk2, mp2, [0, 1, 2, 3, 4])
        assert not eng2._trace._drained   # window [0,2) done; [5,7) remains
        _drive_steps(eng2, chunk2, mp2, [5, 6])
        assert eng2._trace._drained
        assert "step_000006_g.pt" in set(os.listdir(td2))
        eng2._trace.close()               # idempotent second close: quiet
    print("PASS test_async_drains_at_last_target_step")


def test_async_drain_fence_covers_found_inf_gap():
    """A found-inf skip of the LAST target step never enqueues it, so the
    inline drain cannot fire there — the first step PAST the last target
    hits the early-return drain fence and flushes the queue anyway."""
    torch.manual_seed(11)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 3, chunk,
                                   config_extra={"sga_trace_async": True})
        # emulate found_inf exactly at the last target step (step 2): the
        # engine skips the dump slot for it (no enqueue ever fires there)
        orig_prepare = mp.prepare_grads
        calls = {"n": 0}

        def inf_on_third():
            found = orig_prepare()
            calls["n"] += 1
            return True if calls["n"] == 3 else found

        mp.prepare_grads = inf_on_third
        with _ReplicaWorld(world=1):
            for step in (0, 1, 2, 3):
                off = 0
                for p in chunk.params:
                    n = p.numel()
                    chunk.bucket.grad_data[off:off + n] += torch.randn(n)
                    off += n
                assert eng.grad_sync(chunk, force_all_reduce=False) is True
                eng.optimizer_step(mp, [chunk], step)  # step 2 returns skip
        assert calls["n"] == 4
        assert eng._trace._drained          # fenced at step 3's slot
        assert "step_000001_g.pt" in set(os.listdir(td))
        assert "step_000002_g.pt" not in set(os.listdir(td))  # the skip
        assert not _tmp_files(td)
        assert not any(t.is_alive() for t in eng._trace._writer._threads)
    print("PASS test_async_drain_fence_covers_found_inf_gap")


def test_async_snapshot_independent_and_write_overlapped():
    """The async handoff is real overlap AND a real copy: the writer's save
    for step 0 is held until after optimizer_step(0) has rewritten the
    grads (the update score materializes into them) — the published file
    must still carry the dump-slot view, so the queued tensors cannot alias
    training storage; and the step returns BEFORE its own file exists."""
    torch.manual_seed(12)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 3, chunk,
                                   config_extra={"sga_trace_async": True})
        expected = {}

        orig_dump = eng._trace.maybe_dump

        def spy(engine, mp_opt, step, grad_norm):
            # capture AFTER the dump: _entries is bound by it, and the
            # optimizer's inner step (which rewrites the grads) has not
            # run yet — this is still the dump-slot view
            ret = orig_dump(engine, mp_opt, step, grad_norm)
            if engine._trace._entries is not None:
                expected[step] = {name: rec.main_param.grad.detach().clone()
                                  for name, rec, _ in engine._trace._entries}
            return ret

        eng._trace.maybe_dump = spy

        gate = threading.Event()
        real_save = torch.save

        def gated_save(obj, path, **kw):
            if ".tmp." in str(path):
                gate.wait(timeout=10)   # hold step 0's file mid-write
            return real_save(obj, path, **kw)

        with patch("torch.save", side_effect=gated_save):
            _drive_one(eng, chunk, mp, 0)
            # overlap: the step completed while its file was still pending
            assert not os.path.exists(os.path.join(td, "step_000000_g.pt"))
            # the optimizer rewrote the grads (update score materializes);
            # whatever the writer publishes must be the dump-slot view
            rewritten = any(
                not torch.equal(rec.main_param.grad, expected[0][name])
                for name, rec, _ in eng._trace._entries)
            assert rewritten, "score materialization did not touch grads"
            gate.set()                  # release the writer
            _drive_steps(eng, chunk, mp, [1, 2])  # continue to the drain

        got = torch.load(os.path.join(td, "step_000000_g.pt"),
                         weights_only=False)["payload"]
        for name, want in expected[0].items():
            assert torch.equal(got[name], want), (
                f"{name}: published payload is not the dump-slot snapshot "
                "(the queue aliased training storage)")
        assert not _tmp_files(td)
    print("PASS test_async_snapshot_independent_and_write_overlapped")


def test_async_depth_budget_blocks_at_one_item():
    """sga_trace_async_queue_depth bounds items in flight (queued + under
    the writer's pen): with depth 1 and the writer held mid-save, a second
    dump's reserve BLOCKS until the first write completes — the writer
    never holds a second payload, and the wait shows up in diagnostics."""
    torch.manual_seed(13)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            td, [0], 2, chunk,
            config_extra={"sga_trace_async": True,
                          "sga_trace_async_queue_depth": 1})
        gate = threading.Event()
        entered = threading.Event()
        real_save = torch.save

        def gated_save(obj, path, **kw):
            if ".tmp." in str(path):
                entered.set()
                gate.wait(timeout=10)
            return real_save(obj, path, **kw)

        with patch("torch.save", side_effect=gated_save):
            _drive_one(eng, chunk, mp, 0)       # submit; writer enters save
            assert entered.wait(timeout=10)
            col = eng._trace
            meta = col._meta(mp, 1, 0.5, "live_full")
            second = threading.Thread(
                target=lambda: col._dump_grad(1, meta))
            second.start()
            time.sleep(0.3)
            assert second.is_alive(), (
                "second dump ran while the depth-1 budget was full")
            gate.set()                          # let step 0's write finish
            second.join(timeout=10)
            assert not second.is_alive()
            col.close()                         # drain, release the patch

        diag = eng._trace.diagnostics()
        assert diag["peak_inflight"] == 1, diag
        assert diag["backpressure_waits"] >= 1, diag
        assert diag["enqueued"] == 2 and diag["written"] == 2, diag
        assert diag["snapshot_n"] == 2, diag  # both dumps' copies timed
        assert "step_000001_g.pt" in set(os.listdir(td))
        assert not _tmp_files(td)
    print("PASS test_async_depth_budget_blocks_at_one_item")


def test_async_byte_budget_writer_level():
    """The byte budget at the writer face (no engine): a reservation that
    does not fit next to the outstanding one blocks until the write frees
    it; peak reservation never crosses the cap; an item whose 2x
    reservation alone exceeds the cap refuses immediately — before any
    snapshot could be allocated (reserve is its only entry point)."""
    with tempfile.TemporaryDirectory() as td:
        payload = {"p0": torch.arange(64, dtype=torch.float32)}
        nbytes = payload["p0"].numel() * 4      # 256 -> reservation 512
        cap = 512
        w = _AsyncGradWriter(td, queue_depth=8, max_workset_bytes=cap)
        w.start()
        w.reserve(cap)                           # reserved == cap exactly
        blocked = threading.Thread(target=lambda: w.reserve(cap))
        blocked.start()
        time.sleep(0.2)
        assert blocked.is_alive(), "byte budget did not block the second item"
        w.submit(0, {"step": 0}, payload, cap)
        blocked.join(timeout=10)                 # write frees -> reserve runs
        assert not blocked.is_alive()
        stats = w.close()
        assert stats["peak_reserved_bytes"] == cap, stats
        assert stats["written"] == 1, stats
        assert os.path.exists(os.path.join(td, "step_000000_g.pt"))

        # oversized single item: fail-fast, no wait, no bypass
        w2 = _AsyncGradWriter(td, queue_depth=8, max_workset_bytes=cap)
        w2.start()
        try:
            w2.reserve(cap + 16)                 # reservation > 512
            raise AssertionError("oversized item accepted")
        except ValueError as e:
            assert "exceeds" in str(e) and "no oversized bypass" in str(e), e
        assert w2.stats["enqueued"] == 0
        assert not _tmp_files(td)
        w2.close()
    print("PASS test_async_byte_budget_writer_level")


def test_async_oversized_item_fails_before_snapshot():
    """Engine-level fail-fast: a cap smaller than one item's reservation
    refuses at the FIRST dump, before the snapshot allocation — no g file,
    no tmp, nothing enqueued, and the error names the cap knob."""
    torch.manual_seed(14)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        # payload = 2 params x 64 elements x 4 bytes = 512 -> reservation
        # 1024 bytes; cap 512 bytes refuses the item outright
        eng = _build_traced_engine(
            td, [0], 2, chunk,
            config_extra={"sga_trace_async": True,
                          "sga_trace_async_max_workset_gib": 512 / 2**30})
        col = eng._trace
        try:
            _drive_one(eng, chunk, mp, 0)
            raise AssertionError("oversized g item accepted")
        except ValueError as e:
            assert "exceeds" in str(e), e
            assert "sga_trace_async_max_workset_gib" in str(e), e
        # reserve failed before the snapshot: nothing reached the writer.
        # The sync parts of THIS step legitimately landed (catalog + the
        # branch-point state file dump before the g refusal) — only the g
        # file and any tmp must be absent.
        assert col.diagnostics()["enqueued"] == 0
        assert not any(f.endswith("_g.pt") for f in os.listdir(td))
        assert not _tmp_files(td)
        col.close()
    print("PASS test_async_oversized_item_fails_before_snapshot")


def test_async_writer_exception_latch_fail_closed():
    """A writer failure latches and re-raises SYNCHRONOUSLY on the training
    thread (next enqueue or the final drain — whichever comes first), the
    failed target never appears under its own name (atomic publication),
    the backlog is discarded rather than half-written, and close stays
    idempotent: the latched error raises exactly once."""
    torch.manual_seed(15)
    with tempfile.TemporaryDirectory() as td:
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 4, chunk,
                                   config_extra={"sga_trace_async": True})

        real_save = torch.save

        def booming_save(obj, path, **kw):
            if ".tmp." in str(path):
                raise _WriterBoom("disk exploded")
            return real_save(obj, path, **kw)

        raised = None
        with patch("torch.save", side_effect=booming_save):
            for step in range(4):
                try:
                    _drive_one(eng, chunk, mp, step)
                except _WriterBoom as e:   # next enqueue or final drain
                    raised = e
                    break
            assert raised is not None, "writer failure never surfaced"
        assert isinstance(raised, _WriterBoom)

        files = set(os.listdir(td))
        assert "step_000000_g.pt" not in files   # failed write: no target
        assert not _tmp_files(td)                # partial tmp cleaned up
        assert "catalog.pt" in files             # sync parts unaffected
        diag = eng._trace.diagnostics()
        assert diag["error"] is not None and "disk exploded" in diag["error"]
        assert diag["written"] == 0, diag
        # the FAILED write attempt still accounts its elapsed time (the
        # write timer counts attempts, not only successes)
        assert diag["write_n"] == 1 and diag["write_s"] > 0.0, diag
        # idempotent close: close raises the latched error at most once
        # (the enqueue-time raise does not consume close's raise — both
        # are contract points), every later close is quiet
        try:
            eng._trace.close()
        except _WriterBoom:
            pass
        eng._trace.close()
        eng._trace.close()
    print("PASS test_async_writer_exception_latch_fail_closed")


def test_async_multiworker_failure_cleanup():
    """With two active writers, one failure discards queued backlog while an
    already-active peer may finish; simultaneous failures still preserve the
    first-error latch. Every accepted item's slot callback runs exactly once,
    reservations drain to zero, and no partial temporary file survives."""
    real_save = torch.save

    def run_case(both_fail):
        with tempfile.TemporaryDirectory() as td:
            writer = _AsyncGradWriter(
                td, queue_depth=4, max_workset_bytes=1024,
                worker_count=2)
            released = []
            release_lock = threading.Lock()

            def release(step):
                with release_lock:
                    released.append(step)

            for step in range(4):
                writer.reserve(1)
                writer.submit(
                    step, {"step": step}, {"g": torch.tensor([float(step)])},
                    1, release_slot=lambda step=step: release(step))

            rendezvous = threading.Barrier(2)

            def racing_save(obj, path, **kw):
                step = obj["meta"]["step"]
                if step in (0, 1):
                    rendezvous.wait(timeout=5)
                    if step == 0 or both_fail:
                        raise _WriterBoom(f"disk exploded at {step}")
                return real_save(obj, path, **kw)

            with patch("torch.save", side_effect=racing_save):
                writer.start()
                try:
                    writer.close()
                except _WriterBoom as e:
                    assert "disk exploded" in str(e)
                else:
                    raise AssertionError("multi-worker failure was not raised")

            # The exception is raised once; all later closes are idempotent.
            writer.close()
            writer.close()
            assert sorted(released) == [0, 1, 2, 3], released
            assert len(released) == len(set(released))
            assert writer._reserved == 0 and writer._inflight == 0
            assert writer._active_workers == 0
            assert not any(t.is_alive() for t in writer._threads)
            assert not _tmp_files(td)
            assert writer.stats["error"] is not None
            assert writer.stats["error_step"] in (0, 1)
            targets = sorted(f for f in os.listdir(td)
                             if f.endswith("_g.pt"))
            assert writer.stats["written"] == len(targets)
            assert "step_000000_g.pt" not in targets
            if both_fail:
                assert targets == []
            else:
                # The peer may already own and finish additional items before
                # the first failure acquires the latch. Unclaimed backlog is
                # discarded; already-active work is allowed to publish.
                assert "step_000001_g.pt" in targets
                assert set(targets) <= {
                    "step_000001_g.pt", "step_000002_g.pt",
                    "step_000003_g.pt"}

    run_case(both_fail=False)
    run_case(both_fail=True)
    print("PASS test_async_multiworker_failure_cleanup")


def test_async_resume_skip_and_refuse_overwrite():
    """Re-running the same deterministic trace into its own directory skips
    every already-published target (bitwise match, nothing rewritten); a
    target that exists with DIFFERENT content refuses the overwrite — a
    rerun must name an empty directory."""
    def _build(d, seed=16):
        torch.manual_seed(seed)
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(d, [0], 2, chunk,
                                   config_extra={"sga_trace_async": True})
        return eng, chunk, mp

    with tempfile.TemporaryDirectory() as td:
        eng, chunk, mp = _build(td)
        _drive_steps(eng, chunk, mp, [0, 1])
        before = {f: os.stat(os.path.join(td, f)).st_mtime_ns
                  for f in os.listdir(td)}
        eng, chunk, mp = _build(td)               # same seed: same bytes
        _drive_steps(eng, chunk, mp, [0, 1])
        diag = eng._trace.diagnostics()
        assert diag["skipped_existing"] == 2, diag
        assert diag["written"] == 0 and diag["enqueued"] == 0, diag
        # the skip path still pays (and times) its blocking snapshots
        assert diag["snapshot_n"] == 2 and diag["write_n"] == 0, diag
        after = {f: os.stat(os.path.join(td, f)).st_mtime_ns
                 for f in os.listdir(td)}
        # the g targets are byte-frozen by the skip (the sync state file
        # legitimately re-dumps — legacy resume recomputation semantics)
        assert {f: v for f, v in before.items() if f.endswith("_g.pt")} == \
               {f: v for f, v in after.items() if f.endswith("_g.pt")}, \
            "resume skip rewrote published g files"

        # a foreign payload under a target name: refuse, name the file
        bad = torch.load(os.path.join(td, "step_000001_g.pt"),
                         weights_only=False)
        bad["payload"] = {k: v + 1.0 for k, v in bad["payload"].items()}
        torch.save(bad, os.path.join(td, "step_000001_g.pt"))
        eng, chunk, mp = _build(td)
        try:
            _drive_steps(eng, chunk, mp, [0, 1])
            raise AssertionError("overwrite of a differing target accepted")
        except ValueError as e:
            assert "step_000001_g.pt" in str(e), e
            assert "empty directory" in str(e), e
            # the refused dump's snapshot is timed too: skip + refuse = 2
            d = eng._trace.diagnostics()
            assert d["snapshot_n"] == 2 and d["skipped_existing"] == 1, d
        finally:
            eng._trace.close()  # refusal fires before submit: quiet drain
    print("PASS test_async_resume_skip_and_refuse_overwrite")


def test_async_stale_tmp_cleanup_scoped():
    """Startup hygiene removes ONLY this protocol's tmp files whose owner
    pid is dead: a dead writer's orphan goes, our own and any live
    owner's tmp stay (their run may still be mid-write), and files that
    merely look tmp-ish are not ours to touch."""
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as td:
        dead = subprocess.Popen(["true"])
        dead.wait()                               # a real, reaped pid
        orphan = f"step_000009_g.pt.tmp.{dead.pid}.3"
        own = f"step_000009_g.pt.tmp.{os.getpid()}.4"
        foreignish = "notes.tmp"
        for name in (orphan, own, foreignish):
            with open(os.path.join(td, name), "wb") as f:
                f.write(b"x")
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 1, chunk,
                                   config_extra={"sga_trace_async": True})
        _drive_steps(eng, chunk, mp, [0])
        remaining = set(os.listdir(td))
        assert orphan not in remaining            # dead owner: cleaned
        assert own in remaining                   # live owner (us): kept
        assert foreignish in remaining            # not our protocol: kept
        # our actual publication still landed and no protocol tmp leaks
        assert "step_000000_g.pt" in remaining
        assert sorted(_tmp_files(td)) == [own], _tmp_files(td)
        os.remove(os.path.join(td, own))          # drop the planted keep
    print("PASS test_async_stale_tmp_cleanup_scoped")


def test_async_rank_gate_no_writer():
    """Non-zero ranks never construct the writer or touch the directory
    (the async machinery is rank-0 exactly like the sync one)."""
    torch.manual_seed(18)
    with tempfile.TemporaryDirectory() as td:
        chunk = _prefixed_chunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0], 1, chunk,
                                   config_extra={"sga_trace_async": True})
        eng._rank = 1
        _drive_steps(eng, chunk, mp, [0])
        assert os.listdir(td) == []
        assert eng._trace._writer is None
        assert eng._trace.close() == {}
    print("PASS test_async_rank_gate_no_writer")


def test_async_phase_timer_keys_and_accounting():
    """Async diagnostics expose count, cumulative and maximum duration for
    snapshot, cast, write and drain. Sync mode exposes no timer face."""
    def _assert_shape(diag):
        for phase in ("snapshot", "cast", "write", "drain"):
            for suf in ("_n", "_s", "_max_s"):
                assert f"{phase}{suf}" in diag, (phase, suf, sorted(diag))
            assert diag[f"{phase}_n"] >= 0, phase
            assert diag[f"{phase}_s"] >= 0.0, phase
            assert diag[f"{phase}_max_s"] >= 0.0, phase
            assert diag[f"{phase}_s"] >= diag[f"{phase}_max_s"], phase

    with tempfile.TemporaryDirectory() as td:
        # fp32 arm: 4 g dumps (windows [0,2) and [2,4)), no cast, clean run
        torch.manual_seed(19)
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(td, [0, 2], 2, chunk,
                                   config_extra={"sga_trace_async": True})
        _drive_steps(eng, chunk, mp, [0, 1, 2, 3])
        diag = eng._trace.diagnostics()
        _assert_shape(diag)
        assert diag["snapshot_n"] == 4, diag          # one per g dump
        assert diag["snapshot_n"] == diag["enqueued"] + diag["skipped_existing"]
        assert diag["cast_n"] == 0 and diag["cast_s"] == 0.0, diag
        assert diag["cast_max_s"] == 0.0, diag        # fp32: never cast
        assert diag["write_n"] == diag["written"] == 4, diag
        assert diag["write_s"] > 0.0, diag            # real file I/O happened
        assert diag["snapshot_s"] > 0.0, diag         # real copies happened
        assert diag["snapshot_max_s"] > 0.0, diag
        assert diag["write_max_s"] > 0.0, diag
        # the inline drain waited exactly once; the idempotent no-op close
        # below must not add a second drain record
        assert diag["drain_n"] == 1, diag
        assert diag["drain_s"] >= diag["drain_max_s"], diag
        eng._trace.close()
        assert eng._trace.diagnostics()["drain_n"] == 1, diag

    with tempfile.TemporaryDirectory() as td:
        # bf16 arm: the cast phase mirrors the snapshot phase 1:1
        torch.manual_seed(19)
        chunk = _FakeChunk()
        mp = _FakeMPOpt(chunk)
        eng = _build_traced_engine(
            td, [0, 2], 2, chunk,
            config_extra={"sga_trace_async": True,
                          "sga_trace_dump_dtype": "bf16"})
        _drive_steps(eng, chunk, mp, [0, 1, 2, 3])
        diag = eng._trace.diagnostics()
        _assert_shape(diag)
        assert diag["cast_n"] == diag["snapshot_n"] == 4, diag
        assert diag["cast_s"] > 0.0 and diag["cast_max_s"] > 0.0, diag
        assert diag["write_n"] == 4, diag
    print("PASS test_async_phase_timer_keys_and_accounting")


def test_async_close_bounded_and_values_match_helper():
    """A timed-out close reports incomplete draining; a later idempotent
    close succeeds after the writer unblocks. Also pin NaN-aware metadata
    equality for resume checks."""
    with tempfile.TemporaryDirectory() as td:
        w = _AsyncGradWriter(td, queue_depth=2, max_workset_bytes=1 << 20)
        w.start()
        gate = threading.Event()
        real_save = torch.save

        def gated_save(obj, path, **kw):
            if ".tmp." in str(path):
                gate.wait(timeout=10)
            return real_save(obj, path, **kw)

        with patch("torch.save", side_effect=gated_save):
            payload = {"p0": torch.zeros(4)}
            w.reserve(16)
            w.submit(0, {"step": 0}, payload, 16)
            try:
                w.close(timeout=0.2)
                raise AssertionError("bounded close pretended success")
            except RuntimeError as e:
                assert "did not drain" in str(e), e
            gate.set()
            stats = w.close(timeout=10)
        assert stats["written"] == 1, stats
        assert not _tmp_files(td)
        # the drain timer: the timed-out close definitely waited its full
        # 0.2s (one drain record, the max); the post-gate close may join a
        # writer that is still finishing (a second, tiny record) or one
        # that already exited (none) — only the bounded wait is
        # deterministic, and a later no-op close never adds anything
        assert stats["drain_n"] >= 1, stats
        assert stats["drain_max_s"] >= 0.2, stats
        assert stats["drain_s"] >= stats["drain_max_s"] >= 0.0, stats

        nan_meta = {"step": 3, "grad_norm": float("nan"), "lr": [1e-3]}
        assert _values_match(nan_meta, dict(nan_meta))
        clone = dict(nan_meta, grad_norm=float("nan"))
        assert _values_match(nan_meta, clone)
        assert not _values_match(nan_meta, dict(nan_meta, step=4))
        assert not _values_match(nan_meta, dict(nan_meta, lr=[2e-3]))
        assert _values_match({"pg": [{"eps": 1e-8}]},
                             {"pg": [{"eps": 1e-8}]})
        assert not _values_match(float("nan"), 0.0)
    print("PASS test_async_close_bounded_and_values_match_helper")


def test_pinned_cuda_backend_guards_and_optional_parity():
    """pinned_cuda is explicit, CUDA-only, and preserves sync payloads.

    CPU always exercises the fail-loud device guard.  A CUDA host also runs a
    focused collector-level parity and pool-reuse check using real pinned
    memory and a dedicated copy stream synchronized before return.
    """
    col = TraceCollector(
        dir="/tmp/sga-trace-never", branch_points=[], window=1,
        async_enabled=True, async_backend="pinned_cuda")
    assert col.async_backend == "pinned_cuda"
    try:
        TraceCollector(dir="/tmp/sga-trace-never", branch_points=[], window=1,
                       async_backend="pinned_cuda")
        raise AssertionError("pinned backend accepted with async disabled")
    except ValueError as e:
        assert "requires" in str(e), e
    col._entries = [("g", SimpleNamespace(
        main_param=SimpleNamespace(grad=torch.arange(8.0))), 0)]
    try:
        col._dump_grad_pinned_cuda(0, {"step": 0})
        raise AssertionError("pinned backend accepted CPU gradients")
    except RuntimeError as e:
        assert "requires CUDA gradients" in str(e), e
    finally:
        col.close()

    if not torch.cuda.is_available():
        print("SKIP pinned CUDA parity (CUDA unavailable)")
        return
    for dump_dtype in ("fp32", "bf16"):
        with tempfile.TemporaryDirectory() as td:
            grad = torch.linspace(-3, 3, 257, device="cuda")
            rec = SimpleNamespace(main_param=SimpleNamespace(grad=grad))
            col = TraceCollector(
                dir=td, branch_points=[], window=1, dump_dtype=dump_dtype,
                async_enabled=True, async_backend="pinned_cuda",
                async_queue_depth=3, async_worker_count=2,
                async_max_workset_gib=0.01)
            col._entries = [("g", rec, 0)]
            before = grad.detach().cpu()
            expected = []
            for step in range(7):  # > slot_count proves blocking reuse
                grad.copy_(torch.linspace(-3, 3, 257, device="cuda") + step)
                expected.append(grad.detach().cpu())
                col._dump_grad(step, {"step": step})
                grad.add_(1000)  # optimizer-like mutation after D2H completes
            stats = col.close()
            for step, before in enumerate(expected):
                got = torch.load(
                    os.path.join(td, f"step_{step:06d}_g.pt"),
                    weights_only=False)["payload"]["g"]
                want = (before if dump_dtype == "fp32"
                        else before.to(torch.bfloat16))
                assert torch.equal(got, want)
            assert len(col._pinned_slots) == 3
            assert len(col._free_pinned_slots) == 3
            assert all(t.is_pinned() for slot in col._pinned_slots
                       for t in slot.payload.values())
            assert stats["written"] == 7
            assert stats["d2h_submit_n"] == 7
            assert stats["d2h_sync_n"] == 7
            assert stats["error"] is None
    print("PASS test_pinned_cuda_backend_guards_and_optional_parity")


def test_async_constructor_guards():
    """Direct-construction backstops for the async knobs (the engine-test
    route seeds config without validation; same discipline as dump_dtype)."""
    ok = TraceCollector(dir="/tmp/sga-trace-never", branch_points=[],
                        window=1, async_enabled=True,
                        async_backend="cpu_snapshot", async_queue_depth=1,
                        async_max_workset_gib=0.5)
    assert ok.async_enabled and ok.async_backend == "cpu_snapshot"
    assert ok.async_queue_depth == 1
    assert ok.async_max_workset_gib == 0.5
    for bad in ({"async_queue_depth": 0}, {"async_queue_depth": True},
                {"async_queue_depth": 1.5},
                {"async_worker_count": 0}, {"async_worker_count": True},
                {"async_queue_depth": 1, "async_worker_count": 2},
                {"async_max_workset_gib": 0.0},
                {"async_max_workset_gib": -1.0},
                {"async_max_workset_gib": float("nan")},
                {"async_max_workset_gib": float("inf")}):
        try:
            TraceCollector(dir="/tmp/sga-trace-never", branch_points=[],
                           window=1, **bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert ("queue_depth" in str(e) or "worker_count" in str(e)
                    or "workset_gib" in str(e)), e
    print("PASS test_async_constructor_guards")


def main():
    test_parse_branch_points()
    test_canonical_name_strips_module_prefixes()
    test_zero_overhead_when_off_structural()
    test_dump_slot_before_optimizer_and_post_clip()
    test_dump_contents_window_and_state()
    test_absolute_numbering_resume_safe()
    test_bind_param_group_contract()
    test_rank_gate()
    test_meta_carries_engine_phase()
    test_sparse_dump_dualpath_content()
    test_sparse_dump_compact_rebuild_content()
    test_sparse_dump_compact_restore_on_dump_exception()
    test_dump_dtype_bf16_ab_convert()
    test_dump_dtype_constructor_guard()
    test_async_default_off_is_sync()
    test_async_parity_with_sync()
    test_async_drains_at_last_target_step()
    test_async_drain_fence_covers_found_inf_gap()
    test_async_snapshot_independent_and_write_overlapped()
    test_async_depth_budget_blocks_at_one_item()
    test_async_byte_budget_writer_level()
    test_async_oversized_item_fails_before_snapshot()
    test_async_writer_exception_latch_fail_closed()
    test_async_multiworker_failure_cleanup()
    test_async_resume_skip_and_refuse_overwrite()
    test_async_stale_tmp_cleanup_scoped()
    test_async_rank_gate_no_writer()
    test_async_phase_timer_keys_and_accounting()
    test_async_close_bounded_and_values_match_helper()
    test_pinned_cuda_backend_guards_and_optional_parity()
    test_async_constructor_guards()
    print("\nall trace tests passed")


if __name__ == "__main__":
    main()
