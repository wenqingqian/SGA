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
import sys
import tempfile

import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.engine import SGAEngine
from sga.plugins import build_backend_plugin, build_optimizer, build_selection
from sga.trace import TraceCollector, _canonical_name, parse_branch_points

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
    print("\nall trace tests passed")


if __name__ == "__main__":
    main()
