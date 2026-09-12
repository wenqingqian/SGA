#!/usr/bin/env python3
"""Unit tests for the multi-node fused replay launcher (sga.replay_launch).

Run from anywhere (CPU only; the "multi-node" simulations run both
node-workers on the local host and rendezvous them over 127.0.0.1, so the
ssh transport itself stays out of the unit-test surface — the cross-node
transport is covered by tests/smoke_replay_multinode.sh):

    python SGA/tests/test_replay_launch.py

Contract under test:
1. The worker config rides the ssh command line as base64(json): the
   roundtrip is exact (floats/lists/bools included).
2. The node-worker preflight is a real bounded rendezvous: a world=2
   preflight over 127.0.0.1 succeeds, and a node that cannot meet the
   master endpoint fails loudly with the node named.
3. The full run_fused_branch_multinode path in "local" executor mode:
   cold dense cache populated by the node-0 parent, published, confirmed
   on the second node, two node-workers spawned (global ranks 0 and 1),
   shards sliced by GLOBAL rank, the fragment assembled by global rank 0,
   and the branch report inside the world=2 pure-reordering band against a
   world=1 run (wrong slicing or a wrong global-rank mapping would blow
   the band, not pass silently) — plus the composition case (world=4 over
   two nodes: node_rank>0 AND local_shards>1) and the single-node
   degenerate case.
4. Launch gates: shard/world divisibility by the node count, empty and
   duplicated and ill-formed node lists, shard count >= 1, timeout bounds,
   the --device cuda:N rejection, and the launcher-side CUDA capacity
   check (per-node share under --nodes, whole world single-node) all name
   their option; the node-worker entry names its required options and
   rejects inconsistent geometry.
5. Failure paths fail WITHOUT orphans: a mid-spawn failure, a run-phase
   early death and a KeyboardInterrupt all reap every started node-worker;
   the opt-in --run-deadline kills a worker hung outside a collective.
6. The out-of-band cache confirmation names every failure (rc!=0, wrong
   bytes, unparseable probe output), and the cluster ssh transport policy
   lives in exactly one place (probe + node-worker render share it).
"""
import argparse
import io
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sga import replay as soa
from sga import replay_launch
from test_replay_fused import _base_args, _build_fixture, _compare_world2


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return soa.main(argv)


def _ctx(trace_dir):
    _c, _n, _o, _N, _s, g_files = soa.discover_trace(
        os.path.abspath(trace_dir))
    return {"trace_dir": os.path.abspath(trace_dir), "g_files": g_files}


def _multinode_args(td, cache, t0, shards, device="cpu",
                    preflight_timeout=600.0, run_deadline=0.0):
    """The args namespace run_fused_branch_multinode consumes (the same
    fields sga.replay's argparse produces after its own gates)."""
    return argparse.Namespace(
        replay_mode="cycle", self_check=None, dense_cache=cache,
        score_names=["abs_g", "update"], fused_shards=shards,
        fused_max_configs=2, ratios=[20.0, 30.0], horizons=[1, 3, 6],
        window=6, cycle_len=2, replay_dtype="fp64", trace_dtype="fp32",
        norm_dtype="fp64", chunk_mb=1, device=device, seed=1234,
        verbose=False, preflight_timeout=preflight_timeout,
        run_deadline=run_deadline, launch_log_dir=None)


def _sleep_argv(seconds=120):
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _exit_argv(rc):
    return [sys.executable, "-c", f"import sys; sys.exit({rc})"]


def _record_spawns():
    """Patch _spawn_node to record every spawned proc. Returns (spawned,
    restore): restore() unpatches and force-reaps anything left — call it
    in a finally, AFTER the launcher-contract assertions (reaping a
    survivor there would mask the very bug the assertions look for)."""
    spawned = []
    real_spawn = replay_launch._spawn_node

    def fake_spawn(node, node_rank, phase, render, log_dir):
        proc, log_path = real_spawn(node, node_rank, phase, render, log_dir)
        spawned.append(proc)
        return proc, log_path

    def restore():
        replay_launch._spawn_node = real_spawn
        for p in spawned:
            if p.poll() is None:
                p.kill()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    replay_launch._spawn_node = fake_spawn
    return spawned, restore


def test_cfg_b64_roundtrip():
    """The ssh command line carries the worker config as base64(json);
    the node-worker must recover it exactly (the engine consumes json-safe
    scalars/lists only — a tensor here would mean the launcher started
    shipping state that cannot cross an argv)."""
    cfg = {"trace_dir": "/w/trace", "t0": 900, "window": 6,
           "horizons": [1, 3, 6], "ratios": [20.0, 30.0],
           "score_names": ["abs_g", "update"], "cycle_len": 2,
           "cycle_mode": True, "replay_dtype": "fp64",
           "trace_dtype": "bf16", "upcast_bf16": True, "norm_fp32": False,
           "chunk_mb": 1, "device": "cpu", "dense_cache": "/w/cache",
           "max_configs": 2, "seed": 1234, "verbose": False,
           "dense_cache_hit": False}
    blob = replay_launch._cfg_to_b64(cfg)
    assert all(c.isalnum() or c in "+/=" for c in blob), \
        "cfg blob must be argv-safe ascii"
    assert replay_launch._cfg_from_b64(blob) == cfg
    print("PASS test_cfg_b64_roundtrip")


def test_node_worker_preflight_world2():
    """A real bounded rendezvous: two preflight children meet at a local
    tcp endpoint under the env-driven global-rank mapping (0 and 1). The
    banner must carry the actual plumbing (master:endpoint, world,
    local_shards) so argv-plumbing regressions surface here, not only on
    real nodes."""
    port = replay_launch._free_port_host()
    buf = io.StringIO()
    with redirect_stdout(buf):
        replay_launch.main([
            "--node-worker", "--phase", "preflight",
            "--master", "127.0.0.1", "--preflight-port", str(port),
            "--world", "2", "--local-shards", "2", "--node-rank", "0",
            "--device", "cpu", "--preflight-timeout", "30"])
    banner = buf.getvalue()
    assert "preflight" in banner, banner
    assert f"master=127.0.0.1:{port}" in banner, banner
    assert "world=2" in banner and "local_shards=2" in banner, banner
    print("PASS test_node_worker_preflight_world2")


def test_node_worker_arg_gates():
    """The node-worker entry names every required option instead of
    failing deep (argparse's loud exit), including the geometry hardening:
    inconsistent world/local-shards/node-rank and a zero rendezvous port
    are rejected by name on stderr."""
    cases = (
        (["--node-worker"], "required"),
        (["--node-worker", "--phase", "run", "--master", "127.0.0.1",
          "--local-shards", "1", "--world", "1"], "cfg-b64"),
        ([], "--node-worker"),
        (["--node-worker", "--phase", "run", "--master", "127.0.0.1",
          "--port", "0", "--local-shards", "2", "--world", "4",
          "--cfg-b64", "x", "--frag-path", "/tmp/f"], "--port"),
        (["--node-worker", "--phase", "preflight", "--master", "127.0.0.1",
          "--preflight-port", "0", "--local-shards", "1", "--world", "1"],
         "--preflight-port"),
        (["--node-worker", "--phase", "run", "--master", "127.0.0.1",
          "--port", "1", "--local-shards", "4", "--world", "6",
          "--cfg-b64", "x", "--frag-path", "/tmp/f"], "not divisible"),
        (["--node-worker", "--phase", "run", "--master", "127.0.0.1",
          "--port", "1", "--local-shards", "2", "--world", "4",
          "--node-rank", "5", "--cfg-b64", "x", "--frag-path", "/tmp/f"],
         "--node-rank"),
    )
    for argv, needle in cases:
        buf = io.StringIO()
        with redirect_stderr(buf):
            try:
                replay_launch.main(argv)
            except SystemExit as e:
                assert e.code == 2, (argv, e.code)
            else:
                raise AssertionError(f"node-worker accepted: {argv}")
        # the named-diagnostics contract: the failing option is on stderr
        assert needle in buf.getvalue(), (needle, buf.getvalue())
    print("PASS test_node_worker_arg_gates")


def test_node_worker_preflight_cuda_gate():
    """A cuda-sized node-worker must hit the CUDA device gate in the
    PREFLIGHT phase (before the rendezvous is paid), not only in the run
    phase."""
    ns = argparse.Namespace(node_rank=0, local_shards=8, world=8,
                            device="cuda", master="127.0.0.1",
                            preflight_port=1, preflight_timeout=5.0)
    try:
        replay_launch._node_worker_preflight(ns)
    except ValueError as e:
        assert "CUDA_VISIBLE_DEVICES" in str(e), str(e)
    else:
        # only reachable on a node with >= 8 visible GPUs
        assert torch.cuda.device_count() >= 8
    print("PASS test_node_worker_preflight_cuda_gate")


def test_multinode_local_world2_band():
    """The full multi-node orchestration, single-machine simulated: two
    node-workers (127.0.0.1 x2) rendezvous over a local tcp master. Cold
    cache: the node-0 parent populates the dense cache, confirms it on the
    second node (local executor: same filesystem), then both shard sets
    run; global rank 0 assembles the fragment. The branch report must sit
    in the world=2 pure-reordering band against a world=1 run — this is
    the loud witness for GLOBAL-rank slicing (a local-rank slicing bug
    makes node 1 replay node 0's slices and blows the band) and for the
    global-rank-0 fragment gate."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        # world=1 reference arm (cold cache: it populates its own cache dir;
        # hit semantics are pinned by test_replay_fused)
        f1 = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        assert f1["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is False
        # multi-node arm on a FRESH cache: miss -> node-0 parent populate
        # -> publish confirmation -> two simulated nodes
        fresh_cache = os.path.join(td, "dense_cache_cold")
        out = replay_launch.run_fused_branch_multinode(
            _multinode_args(td, fresh_cache, t0, 2), _ctx(td), t0,
            ["127.0.0.1", "127.0.0.1"], executor="local")
        assert out["replay"]["dense_cache_hit"] is False
        assert out["t0"] == t0 and out["window_len"] == 6
        # same sanitize the CLI report path applies before comparing
        w2 = soa.sanitize({"branch_results": {str(t0): out}}, nd=8)
        _compare_world2(f1, w2, "local-multinode")
    print("PASS test_multinode_local_world2_band")


def test_multinode_local_world4_band():
    """The composition case: node_rank>0 AND local_shards>1 — world=4 over
    two simulated nodes puts TWO shards per node with global rank
    m*local_shards+L. Every other coverage point collapses _global_rank_of
    to a trivial sum (world=2 over 2 nodes: local_shards=1; or node_rank=0
    with local_shards>1), so an off-by-composition bug there would pass
    this whole suite while corrupting every real multi-shard-per-node run;
    here a wrong mapping mis-slices the window and blows the exact band."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        f1 = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        # cache-hit arm: the cold populate/confirm path is pinned by the
        # world=2 band test above; this arm pins the shard->node composition
        out = replay_launch.run_fused_branch_multinode(
            _multinode_args(td, cache, t0, 4), _ctx(td), t0,
            ["127.0.0.1", "127.0.0.1"], executor="local")
        w4 = soa.sanitize({"branch_results": {str(t0): out}}, nd=8)
        _compare_world2(f1, w4, "local-multinode-world4")
    print("PASS test_multinode_local_world4_band")


def test_multinode_single_node_degenerate():
    """--nodes with a single address: m_nodes=1 skips the remote cache
    confirmations entirely and all shards land on node 0 — the launcher
    must still key them by GLOBAL rank and assemble the fragment."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        f1 = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        out = replay_launch.run_fused_branch_multinode(
            _multinode_args(td, cache, t0, 2), _ctx(td), t0,
            ["127.0.0.1"], executor="local")
        w2 = soa.sanitize({"branch_results": {str(t0): out}}, nd=8)
        _compare_world2(f1, w2, "local-multinode-single-node")
    print("PASS test_multinode_single_node_degenerate")


def test_multinode_gates():
    """Every invalid launch shape is rejected with its option named, before
    any node is contacted (the duplicate-node gate guards the real ssh
    transport; the all-local simulation co-locates "nodes" by design)."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        cases = [
            (_multinode_args(td, cache, t0, 3), ["127.0.0.1", "127.0.0.1"],
             "not divisible", "local"),
            (_multinode_args(td, cache, t0, 0), ["127.0.0.1"],
             "--fused-shards must be >= 1", "local"),
            (_multinode_args(td, cache, t0, 2), [],
             "names no node", "local"),
            (_multinode_args(td, cache, t0, 2), ["hostA", "hostA"],
             "--nodes contains duplicates", "ssh"),
            (_multinode_args(td, cache, t0, 2), ["ok-host", "bad/host"],
             "outside [A-Za-z0-9._-]", "local"),
            (_multinode_args(td, cache, t0, 2, preflight_timeout=0.0),
             ["127.0.0.1", "127.0.0.1"], "--preflight-timeout must be > 0",
             "local"),
            (_multinode_args(td, cache, t0, 2, run_deadline=-1.0),
             ["127.0.0.1", "127.0.0.1"], "--run-deadline must be >= 0",
             "local"),
            (_multinode_args(td, cache, t0, 2, device="cuda:1"),
             ["127.0.0.1", "127.0.0.1"], "CUDA_VISIBLE_DEVICES", "local"),
        ]
        for ns, nodes, msg, executor in cases:
            try:
                replay_launch.run_fused_branch_multinode(
                    ns, _ctx(td), t0, nodes, executor=executor)
            except ValueError as e:
                assert msg in str(e), (msg, str(e))
            else:
                raise AssertionError(f"gate accepted: shards={ns.fused_shards}"
                                     f" device={ns.device} nodes={nodes}")
    print("PASS test_multinode_gates")


def test_multinode_cuda_share_gate():
    """Under --nodes the launcher-side CUDA capacity check is per-node:
    the launcher runs no shards, so world=16 over 2 nodes passes a node
    showing 8 devices (the authoritative per-node check is
    _cuda_shard_gate, exercised in test_node_worker_preflight_cuda_gate),
    while 32 over 2 (16/node) and the single-node world=16 spelling both
    fail — the latter with the unchanged single-node message (the CLI
    gate test in test_replay_fused matches it)."""
    from sga.replay_fused import _common_gates
    with tempfile.TemporaryDirectory() as td:
        ns = _multinode_args(td, os.path.join(td, "dc"), 1000, 16,
                             device="cuda")
        orig = torch.cuda.device_count
        torch.cuda.device_count = lambda: 8
        try:
            try:
                _common_gates(ns)
            except ValueError as e:
                assert "exceeds the visible CUDA devices" in str(e), str(e)
            else:
                raise AssertionError("single-node world16 gate did not fire")
            ns.nodes = ["127.0.0.1", "127.0.0.1"]
            assert set(_common_gates(ns)) == {"abs_g", "update"}
            ns.fused_shards = 32
            try:
                _common_gates(ns)
            except ValueError as e:
                assert "16 shards on each" in str(e), str(e)
            else:
                raise AssertionError("world32 over 2 nodes gate did not fire")
        finally:
            torch.cuda.device_count = orig
    print("PASS test_multinode_cuda_share_gate")


def test_multinode_preflight_failure_loud():
    """A node that cannot meet the master endpoint fails the LAUNCH (not
    the run): the preflight port is pre-occupied by a dummy listener, so
    rank 0's store cannot bind and rank 1's handshake never completes —
    every preflight child errors within the bound and the launcher raises
    with the node names, the rc codes and the per-node log paths (nobody
    hangs on the run-level 2h gloo timeout, no shard was ever started,
    and no node-worker child survives the failure)."""
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    occupied = blocker.getsockname()[1]
    real_pick, real_margin = replay_launch._pick_ports, \
        replay_launch._PREFLIGHT_MARGIN_S
    replay_launch._pick_ports = lambda: (occupied, occupied)
    replay_launch._PREFLIGHT_MARGIN_S = 3.0  # tighten the test, not prod
    spawned, restore = _record_spawns()
    try:
        with tempfile.TemporaryDirectory() as td:
            t0 = _build_fixture(td)[0]
            cache = os.path.join(td, "dense_cache")
            try:
                replay_launch.run_fused_branch_multinode(
                    _multinode_args(td, cache, t0, 2), _ctx(td), t0,
                    ["127.0.0.1", "127.0.0.1"], executor="local",
                    preflight_timeout=5.0)
            except RuntimeError as e:
                # either surface is a correct bounded loud failure: the
                # fast node's EADDRINUSE with the slow node killed by the
                # early-bad policy, or both reaped by the launcher
                # deadline; in every variant the message carries the
                # per-node rc/log tail diagnosis
                msg = str(e)
                assert "preflight" in msg, msg
                assert "127.0.0.1" in msg, msg
                assert "log=" in msg, msg
                assert "EADDRINUSE" in msg or "PREFLIGHT FAILED" in msg, msg
            else:
                raise AssertionError("broken preflight accepted")
            assert spawned and all(p.poll() is not None for p in spawned), \
                [p.poll() for p in spawned]
    finally:
        replay_launch._pick_ports = real_pick
        replay_launch._PREFLIGHT_MARGIN_S = real_margin
        blocker.close()
        restore()
    print("PASS test_multinode_preflight_failure_loud")


def test_multinode_run_phase_early_kill():
    """Run-phase supervision: a node-worker that dies early (here: node 0
    exits rc=3 right after preflight passed) kills the survivors and the
    launcher raises naming the dead rc and the survivors — nobody waits
    out the 2h gloo timeout on a dead peer, and no sleeper survives the
    launch."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        real_render = replay_launch._render_node_cmd

        def fake_render(node, node_rank, phase, **kw):
            if phase == "preflight":
                return real_render(node, node_rank, phase, **kw)
            if node_rank == 0:
                return {"argv": _exit_argv(3), "env": None}
            return {"argv": _sleep_argv(), "env": None}

        spawned, restore = _record_spawns()
        replay_launch._render_node_cmd = fake_render
        try:
            try:
                replay_launch.run_fused_branch_multinode(
                    _multinode_args(td, cache, t0, 2), _ctx(td), t0,
                    ["127.0.0.1", "127.0.0.1"], executor="local")
            except RuntimeError as e:
                msg = str(e)
                assert "rc=[3]" in msg, msg
                assert "survivors killed" in msg, msg
                assert "log=" in msg, msg
            else:
                raise AssertionError("early run death accepted")
            assert spawned and all(p.poll() is not None for p in spawned), \
                [p.poll() for p in spawned]
        finally:
            replay_launch._render_node_cmd = real_render
            restore()
    print("PASS test_multinode_run_phase_early_kill")


def test_multinode_spawn_failure_leaves_no_child():
    """A failure while STILL spawning (node 1's render raises after node
    0's worker already started) must not orphan the started worker: the
    launcher's cleanup kills and reaps everything it started (a bad node
    token chains exactly into this orphan path via the log open())."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        real_render = replay_launch._render_node_cmd

        def fake_render(node, node_rank, phase, **kw):
            if node_rank == 0:
                return {"argv": _sleep_argv(), "env": None}
            raise ValueError("render boom on the second node")

        spawned, restore = _record_spawns()
        replay_launch._render_node_cmd = fake_render
        try:
            try:
                replay_launch.run_fused_branch_multinode(
                    _multinode_args(td, cache, t0, 2), _ctx(td), t0,
                    ["127.0.0.1", "127.0.0.1"], executor="local")
            except ValueError as e:
                assert "render boom" in str(e), str(e)
            else:
                raise AssertionError("spawn failure swallowed")
            assert spawned and all(p.poll() is not None for p in spawned), \
                [p.poll() for p in spawned]
        finally:
            replay_launch._render_node_cmd = real_render
            restore()
    print("PASS test_multinode_spawn_failure_leaves_no_child")


def test_multinode_interrupt_reaps_children():
    """An interrupt during run-phase supervision (operator Ctrl-C; the
    workers were spawned start_new_session, so they do not receive the
    terminal signal) must still reap every node-worker through the
    launcher's cleanup."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        real_render = replay_launch._render_node_cmd
        real_supervise = replay_launch._supervise
        calls = {"n": 0}

        def fake_render(node, node_rank, phase, **kw):
            if phase == "preflight":
                return real_render(node, node_rank, phase, **kw)
            return {"argv": _sleep_argv(), "env": None}

        def fake_supervise(procs, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_supervise(procs, **kw)  # preflight passes
            raise KeyboardInterrupt("operator Ctrl-C during the run phase")

        spawned, restore = _record_spawns()
        replay_launch._render_node_cmd = fake_render
        replay_launch._supervise = fake_supervise
        try:
            try:
                replay_launch.run_fused_branch_multinode(
                    _multinode_args(td, cache, t0, 2), _ctx(td), t0,
                    ["127.0.0.1", "127.0.0.1"], executor="local")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("KeyboardInterrupt swallowed")
            assert spawned and all(p.poll() is not None for p in spawned), \
                [p.poll() for p in spawned]
        finally:
            replay_launch._render_node_cmd = real_render
            replay_launch._supervise = real_supervise
            restore()
    print("PASS test_multinode_interrupt_reaps_children")


def test_multinode_run_deadline_kills_sleepers():
    """The opt-in --run-deadline (the run phase is otherwise unbounded by
    design): a node-worker hung OUTSIDE a collective (simulated by a plain
    sleeper that never rendezvouses) is killed at the deadline and the
    launch fails loudly with the per-node log tails."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        real_render = replay_launch._render_node_cmd

        def fake_render(node, node_rank, phase, **kw):
            if phase == "preflight":
                return real_render(node, node_rank, phase, **kw)
            return {"argv": _sleep_argv(), "env": None}

        spawned, restore = _record_spawns()
        replay_launch._render_node_cmd = fake_render
        try:
            try:
                replay_launch.run_fused_branch_multinode(
                    _multinode_args(td, cache, t0, 2, run_deadline=5.0),
                    _ctx(td), t0, ["127.0.0.1", "127.0.0.1"],
                    executor="local", preflight_timeout=60.0)
            except RuntimeError as e:
                msg = str(e)
                assert "run-deadline" in msg, msg
                assert "wall clock" in msg, msg
                assert "killed" in msg, msg
                assert "127.0.0.1" in msg and "log=" in msg, msg
            else:
                raise AssertionError("run-deadline not enforced")
            assert spawned and all(p.poll() is not None for p in spawned), \
                [p.poll() for p in spawned]
        finally:
            replay_launch._render_node_cmd = real_render
            restore()
    print("PASS test_multinode_run_deadline_kills_sleepers")


def test_supervise_deadline_kills_hang():
    """The deadline primitive behind --run-deadline: a worker that never
    exits is SIGKILLed at the deadline (not waited on forever) and reaped
    — _supervise reports it as pending, with no return code."""
    sleeper = subprocess.Popen(_sleep_argv(), start_new_session=True)
    started = time.monotonic()
    try:
        rcs, pending = replay_launch._supervise([sleeper], deadline_s=2.0)
        elapsed = time.monotonic() - started
        assert pending == [0] and rcs == {}, (rcs, pending)
        assert sleeper.poll() is not None, "sleeper killed but not reaped"
        assert elapsed < 30, f"deadline not honored ({elapsed:.1f}s)"
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
        sleeper.wait()
    print("PASS test_supervise_deadline_kills_hang")


def test_confirm_cache_visible_branches():
    """The out-of-band publish confirmation names every failure: a local
    byte-size mismatch, an ssh rc!=0 with stderr, a probe timeout, and
    polluted ssh stdout — where the LAST bare integer wins (motd/
    sitecustomize noise tolerated) and unparseable output is the named
    diagnostic, never a bare ValueError."""
    with tempfile.TemporaryDirectory() as td:
        blob = os.path.join(td, "dense_x.pt")
        with open(blob, "wb") as f:
            f.write(b"x" * 1234)
        key = os.path.join(td, "cluster_ssh_key")
        with open(key, "w", encoding="utf-8") as f:
            f.write("dummy\n")
        replay_launch._confirm_cache_visible("n0", blob, 1234,
                                             executor="local")
        try:
            replay_launch._confirm_cache_visible("n0", blob, 1235,
                                                 executor="local")
        except RuntimeError as e:
            assert "mismatch" in str(e) and "1234" in str(e), str(e)
        else:
            raise AssertionError("wrong byte size accepted")

        real_run = subprocess.run

        def fake_run(stdout="", returncode=0, stderr="", exc=None):
            def _run(argv, **kw):
                if exc is not None:
                    raise exc
                return argparse.Namespace(returncode=returncode,
                                          stdout=stdout, stderr=stderr)
            return _run

        try:
            subprocess.run = fake_run(returncode=1, stderr="perm denied")
            try:
                replay_launch._confirm_cache_visible(
                    "node1", blob, 1234, ssh_key=key)
            except RuntimeError as e:
                assert "rc=1" in str(e), str(e)
                assert "perm denied" in str(e), str(e)
            else:
                raise AssertionError("rc!=0 accepted")
            subprocess.run = fake_run(
                exc=subprocess.TimeoutExpired(cmd="ssh", timeout=5))
            try:
                replay_launch._confirm_cache_visible(
                    "node1", blob, 1234, ssh_key=key)
            except RuntimeError as e:
                assert "timed out" in str(e), str(e)
                assert "Nothing was launched" in str(e), str(e)
            else:
                raise AssertionError("probe timeout accepted")
            subprocess.run = fake_run(stdout=" Warning: motd\n1234\n")
            replay_launch._confirm_cache_visible("node1", blob, 1234,
                                                 ssh_key=key)
            subprocess.run = fake_run(stdout="hello\nworld\n")
            try:
                replay_launch._confirm_cache_visible(
                    "node1", blob, 1234, ssh_key=key)
            except RuntimeError as e:
                assert "unparseable probe output" in str(e), str(e)
            else:
                raise AssertionError("unparseable probe output accepted")
        finally:
            subprocess.run = real_run
    print("PASS test_confirm_cache_visible_branches")


def test_ssh_transport_and_probe_argv():
    """The cluster transport policy lives in ONE place: the cache-confirm
    probe argv and the node-worker render argv share the same ssh base
    (explicit key, port 2222, root@node, the shared option block), and the
    remote one-liner carries the explicit PYTHONPATH prepend instead of a
    silenced `source env.sh`."""
    with tempfile.TemporaryDirectory() as td:
        key = os.path.join(td, "cluster_ssh_key")
        with open(key, "w", encoding="utf-8") as f:
            f.write("dummy\n")
        probe = replay_launch._cache_probe_argv("node1", "/w/cache/x.pt",
                                                ssh_key=key)
        assert probe[0] == "ssh" and "root@node1" in probe, probe
        assert probe[probe.index("-i") + 1] == key, probe
        assert probe[probe.index("-p") + 1] == "2222", probe
        for opt in ("StrictHostKeyChecking=no", "BatchMode=yes",
                    "ServerAliveInterval=60", "ServerAliveCountMax=3"):
            assert opt in probe, (opt, probe)
        p = "import os, sys; print(os.path.getsize(sys.argv[1]))"
        assert probe[-1] == (f"python3 -c {shlex.quote(p)} "
                             f"{shlex.quote('/w/cache/x.pt')}"), probe[-1]
        render = replay_launch._render_node_cmd(
            "node1", 1, "run", sga_root="/w/sga", master="node1", port=1,
            pf_port=2, local_shards=1, world=1, device="cpu", cfg_b64="x",
            frag_path="/tmp/f.pt", pf_timeout=30.0, executor="ssh",
            ssh_key=key)
        assert render["argv"][:-1] == probe[:-1], render["argv"]
        remote = render["argv"][-1]
        assert "source env.sh" not in remote, remote
        assert "conda activate flagscale" in remote, remote
        assert "PYTHONPATH=" in remote, remote
    print("PASS test_ssh_transport_and_probe_argv")


def test_cli_forwards_ssh_key_and_nodes():
    """The dispatch forwards --ssh-key into the launcher along with the
    normalized node list."""
    seen = {}
    real = replay_launch.run_fused_branch_multinode

    def fake(args, ctx, t0, nodes, **kw):
        seen["nodes"] = list(nodes)
        seen["ssh_key"] = kw.get("ssh_key")
        # minimal branch dict that survives sanitize + print_summary
        return {"t0": t0, "window_len": 6, "state_absent": [],
                "scores": {}, "per_ratio": {},
                "replay": {"dense_cache_hit": True, "dtype": "fp64"}}

    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        replay_launch.run_fused_branch_multinode = fake
        try:
            report = _run(_base_args(td, cache, t0,
                                     ["--nodes", "n1,n2",
                                      "--ssh-key", "/tmp/fake_key"]))
        finally:
            replay_launch.run_fused_branch_multinode = real
    assert seen["nodes"] == ["n1", "n2"], seen
    assert seen["ssh_key"] == "/tmp/fake_key", seen
    assert report["config"]["nodes"] == ["n1", "n2"], report["config"]
    print("PASS test_cli_forwards_ssh_key_and_nodes")


def test_cli_empty_nodes_rejected():
    """'--nodes ,' must fail loudly instead of silently degrading to a
    single-node run (an explicitly requested multi-node launch that names
    no node is a silent-error class, not a default)."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        cache = os.path.join(td, "dense_cache")
        try:
            _run(_base_args(td, cache, t0, ["--nodes", ","]))
        except ValueError as e:
            assert "names no node" in str(e), str(e)
        else:
            raise AssertionError("'--nodes ,' accepted")
    print("PASS test_cli_empty_nodes_rejected")


def test_relative_dense_cache_normalized():
    """A relative --dense-cache resolves ONCE, against the launcher's cwd
    — not once per consumer (parent cwd / ssh login $HOME / worker
    sga_root): invoked from a foreign cwd, the populate, the confirm probe
    and the shipped cfg must all see the same absolute path."""
    with tempfile.TemporaryDirectory() as td:
        t0 = _build_fixture(td)[0]
        foreign = os.path.join(td, "foreign_cwd")
        os.makedirs(foreign)
        seen = {}
        real_confirm = replay_launch._confirm_cache_visible

        def fake_confirm(node, cache_path, expected_bytes, **kw):
            seen["path"] = cache_path
            return real_confirm(node, cache_path, expected_bytes, **kw)

        replay_launch._confirm_cache_visible = fake_confirm
        prev = os.getcwd()
        os.chdir(foreign)
        try:
            out = replay_launch.run_fused_branch_multinode(
                _multinode_args(td, "rel_cache", t0, 2), _ctx(td), t0,
                ["127.0.0.1", "127.0.0.1"], executor="local")
        finally:
            os.chdir(prev)
            replay_launch._confirm_cache_visible = real_confirm
        assert out["replay"]["dense_cache_hit"] is False
        assert os.path.isabs(seen["path"]), seen
        assert seen["path"] == os.path.join(foreign, "rel_cache",
                                            os.path.basename(seen["path"])), \
            seen
        assert os.listdir(os.path.join(foreign, "rel_cache")), seen
    print("PASS test_relative_dense_cache_normalized")


def main():
    test_cfg_b64_roundtrip()
    test_node_worker_preflight_world2()
    test_node_worker_arg_gates()
    test_node_worker_preflight_cuda_gate()
    test_multinode_local_world2_band()
    test_multinode_local_world4_band()
    test_multinode_single_node_degenerate()
    test_multinode_gates()
    test_multinode_preflight_failure_loud()
    test_multinode_run_phase_early_kill()
    test_multinode_spawn_failure_leaves_no_child()
    test_multinode_interrupt_reaps_children()
    test_multinode_run_deadline_kills_sleepers()
    test_supervise_deadline_kills_hang()
    test_confirm_cache_visible_branches()
    test_ssh_transport_and_probe_argv()
    test_cli_forwards_ssh_key_and_nodes()
    test_cli_empty_nodes_rejected()
    test_relative_dense_cache_normalized()
    print("\nall replay-launch tests passed")


if __name__ == "__main__":
    main()
