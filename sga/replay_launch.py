# Copyright (c) 2026, SGA framework contributors.
"""Multi-node launcher for the fused single-pass streaming replay.

Topology: shard r of the fused engine lives on node ``r // (world / M)``
(at that node's ``cuda:(r % (world / M))`` device, or any cpu) where M is
the node count; the launcher process itself runs on node 0 (it must see the
same NFS tree under the same paths as every other node — /workspace view
inside the containers).

Per launch the node-0 parent:

1. builds the worker config and, on a dense-cache miss, populates the cache
   itself (the dense pass must never run inside the process group);
2. publishes the cache and confirms it out of band (below);
3. bounded preflight: every node's shard set performs a throwaway gloo
   rendezvous with a short deadline, so a node that cannot start fails in
   minutes with a loud, per-node diagnosis instead of hanging the run on
   the run-level 2h gloo timeout;
4. launches one node-worker per node — every node, node 0 included, goes
   through the same rendered command: ssh :2222 (node 0 reaches itself by
   ssh-to-self, so the fragment handoff relies on that ssh landing back in
   the launcher's filesystem view), or a plain local subprocess under the
   executor=local single-machine simulation — each worker spawning its
   ``world / M`` shards with SGA_MASTER_ADDR / SGA_MASTER_PORT /
   SGA_GLOBAL_RANK set and the local shard index left in ``rank``;
5. supervises all node-workers: any early nonzero exit kills the survivors
   and fails loudly (nobody waits out the 2h gloo timeout on a dead peer);
   on success the branch fragment written by global rank 0 (node 0's first
   shard) is loaded and returned.

Boundedness: the preflight is deadline-bounded by construction. The RUN
phase has no wall-clock bound by default — legitimate replays run for
hours and no ceiling can distinguish a slow-valid run from a hung-invalid
one — so a shard that hangs OUTSIDE a collective (a torch.load stuck on a
stale NFS client, a CUDA stall) would keep the launcher waiting forever.
``--run-deadline <seconds>`` is the opt-in bound for exactly that case:
on expiry every node-worker is killed and the launch fails loudly with
the per-node log tails.

Dense-cache publish protocol (why not NFS polling): the cache is written
once by node 0 through ``save_dense_cache`` (tmp file + atomic rename, so
no reader ever sees a partial blob). Remote visibility is then confirmed
with a single out-of-band ``ssh :2222`` stat per node — one shot, bounded
deadline, exact byte-size match — and each shard still re-validates the
cache ident on load (the bitwise authority). Nothing ever polls an NFS
path: create-then-poll across NFS clients is unreliable (negative-dentry
blind windows ~acdirmax, and cross-frontend ENOENT measured at 30+
minutes on this cluster). A failed confirmation is a loud stop naming the node, the path
and the sizes — the operator warms/verifies manually and reruns; the
launch has not started, so nothing needs cleanup.

ssh discipline: remote paths are always /workspace view (host-side mount
paths do not exist on the nodes); conda activation precedes python; the
cluster key is discovered via --ssh-key / $SGA_SSH_KEY / the nearest
workspace ancestor holding ``cluster_ssh_key``.
"""
import argparse
import base64
import datetime
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _free_port_host():
    """A tcp port free as seen from EVERY interface: the rendezvous store
    must be reachable from the other nodes, so a 127.0.0.1-only probe could
    hand out a port that is taken on the external address."""
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _pick_ports():
    """(preflight, run) rendezvous ports — deliberately distinct, so the
    throwaway preflight store never collides with the run store (stale
    state / TIME_WAIT on a reused endpoint)."""
    return _free_port_host(), _free_port_host()


def _cfg_to_b64(cfg):
    """Worker config as argv-safe ascii (json + base64): the config rides
    the ssh command line, so every shell metacharacter must be neutralized
    once, here, instead of per-hop quoting."""
    return base64.b64encode(json.dumps(cfg).encode()).decode("ascii")


def _cfg_from_b64(blob):
    return json.loads(base64.b64decode(blob.encode("ascii")))


def _sga_root():
    """This repo's root (the launcher file lives in <root>/sga/). On the
    nodes this path is the /workspace view — exactly what remote commands
    must use."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_ssh_key(explicit=None):
    """Cluster ssh key: argument > $SGA_SSH_KEY > nearest workspace ancestor
    of this repo holding ``cluster_ssh_key``."""
    cand = explicit or os.environ.get("SGA_SSH_KEY")
    if cand:
        if not os.path.isfile(cand):
            raise ValueError(f"ssh key not found: {cand}")
        return cand
    cur = _sga_root()
    for _ in range(5):
        cand = os.path.join(cur, "cluster_ssh_key")
        if os.path.isfile(cand):
            return cand
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise ValueError(
        "cluster ssh key not found: pass --ssh-key or set SGA_SSH_KEY "
        f"(searched upward from {_sga_root()})")


# Trust model: private cluster — root@<node> on a fixed port with host-key
# checking disabled is the established cluster convention (the smoke script's
# own ssh line does the same).
SSH_USER = "root"
SSH_PORT = "2222"

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=30",
    # hours-long blocking channels: probe the peer so a vanished node turns
    # into a bounded ssh failure (the watchdog then stops the whole launch)
    # instead of a silent half-open hang
    "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=3",
]


def _ssh_base_argv(node, ssh_key=None):
    """[ssh -i <key> -p <port> <opts> root@<node>] — the ONE place the
    cluster transport policy lives; the cache-confirm probe and the
    node-worker command render both build on it."""
    return ["ssh", "-i", ssh_key or _find_ssh_key(), "-p", SSH_PORT,
            *_SSH_OPTS, f"{SSH_USER}@{node}"]


# ----------------------------------------------------------------------------
# node-worker: one node's shard set (preflight or run phase)
# ----------------------------------------------------------------------------

def _global_rank_of(node_rank, local_rank, local_shards):
    """Shard r -> (node r // local_shards, local r % local_shards) inverted:
    a node's local index L on node m is global rank m * local_shards + L."""
    return node_rank * local_shards + local_rank


def _preflight_child(local_rank, world, master, pf_port, pf_timeout,
                     node_rank, local_shards):
    """One throwaway rendezvous participant: proves that this node can start
    python, import torch and meet every other rank at the master endpoint —
    all under a short, launch-bound deadline (the run-level process group
    keeps its generous 2h timeout; the launch gate must not)."""
    os.environ["SGA_GLOBAL_RANK"] = str(
        _global_rank_of(node_rank, local_rank, local_shards))
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://{master}:{pf_port}",
        rank=int(os.environ["SGA_GLOBAL_RANK"]), world_size=world,
        timeout=datetime.timedelta(seconds=pf_timeout))
    dist.barrier()
    dist.destroy_process_group()


def _run_child(local_rank, world, cfg, port, frag_path, node_rank,
               local_shards):
    """One production shard: replay_fused._worker with the local shard
    index as ``rank`` (device cuda:{rank} per node) and the global rank
    delivered through the env (Comms reads it for the process-group rank)."""
    os.environ["SGA_GLOBAL_RANK"] = str(
        _global_rank_of(node_rank, local_rank, local_shards))
    from sga.replay_fused import _worker
    _worker(local_rank, world, cfg, port, frag_path)


def _cuda_shard_gate(local_shards, device):
    """Node-side CUDA capacity gate, shared by both phases: a
    mis-sized cuda launch must fail here — before the preflight rendezvous
    is paid — and on every node (the node-0 parent cannot see a remote
    node's GPU count)."""
    if device == "cpu":
        return
    visible = torch.cuda.device_count()
    if local_shards > visible:
        raise ValueError(
            f"--local-shards {local_shards} exceeds the visible "
            f"CUDA devices on this node ({visible}); restrict the set "
            "with CUDA_VISIBLE_DEVICES before launching")


def _node_worker_preflight(args):
    _cuda_shard_gate(args.local_shards, args.device)
    print(f"[node-worker] preflight: node_rank={args.node_rank} "
          f"local_shards={args.local_shards} world={args.world} "
          f"master={args.master}:{args.preflight_port} "
          f"timeout={args.preflight_timeout:.0f}s", flush=True)
    try:
        mp.spawn(_preflight_child,
                 args=(args.world, args.master, args.preflight_port,
                       args.preflight_timeout, args.node_rank,
                       args.local_shards),
                 nprocs=args.local_shards, join=True)
    except BaseException as e:
        # loud, named, and nonzero: the launcher collects this per node
        print(f"[node-worker] PREFLIGHT FAILED on node_rank="
              f"{args.node_rank}: {e!r}", flush=True)
        raise


def _node_worker_run(args):
    cfg = _cfg_from_b64(args.cfg_b64)
    # cfg["device"] is the run phase's authority
    _cuda_shard_gate(args.local_shards, cfg["device"])
    # the rendezvous endpoint arrives by env (Comms' multi-node mode); the
    # node-worker is its single authority on this machine
    os.environ["SGA_MASTER_ADDR"] = args.master
    os.environ["SGA_MASTER_PORT"] = str(args.port)
    print(f"[node-worker] run: node_rank={args.node_rank} "
          f"local_shards={args.local_shards} world={args.world} "
          f"master={args.master}:{args.port} device={cfg['device']}",
          flush=True)
    mp.spawn(_run_child,
             args=(args.world, cfg, args.port, args.frag_path,
                   args.node_rank, args.local_shards),
             nprocs=args.local_shards, join=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="multi-node fused replay node-worker (the orchestrator "
                    "is sga.replay's main with --nodes)")
    ap.add_argument("--node-worker", action="store_true",
                    help="run one node's shard set (launched over ssh by "
                         "run_fused_branch_multinode — every node, node 0 "
                         "included, via ssh-to-self — or locally in the "
                         "executor=local single-machine simulation)")
    ap.add_argument("--phase", choices=["preflight", "run"])
    ap.add_argument("--master", help="rendezvous master address (node 0)")
    ap.add_argument("--port", type=int, default=0,
                    help="run-phase rendezvous port")
    ap.add_argument("--preflight-port", type=int, default=0,
                    help="preflight rendezvous port")
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--local-shards", type=int,
                    help="shards on this node (world / node count)")
    ap.add_argument("--world", type=int, help="global shard count")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="preflight phase only: the device the launch is "
                         "sized for (the run phase's authority is the "
                         "cfg shipped via --cfg-b64)")
    ap.add_argument("--preflight-timeout", type=float, default=600.0,
                    help="preflight rendezvous deadline in seconds")
    ap.add_argument("--cfg-b64", help="run phase: base64(json) worker cfg")
    ap.add_argument("--frag-path", help="run phase: fragment output path "
                                        "(written by global rank 0 only)")
    args = ap.parse_args(argv)
    if not args.node_worker:
        ap.error("--node-worker is required (launch a multi-node fused "
                 "replay via sga.replay --nodes)")
    missing = [n for n in ("phase", "master", "local_shards", "world")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"--node-worker missing required options: "
                 f"{[m.replace('_', '-') for m in missing]}")
    if args.phase == "run":
        for opt in ("cfg_b64", "frag_path"):
            if getattr(args, opt) is None:
                ap.error(f"--phase run requires --{opt.replace('_', '-')}")
    # hardening: the launcher always sends a consistent geometry and a
    # real rendezvous port; a malformed manual invocation must be rejected
    # by name instead of surfacing as a hang-then-timeout
    if args.phase == "run" and args.port <= 0:
        ap.error(f"--phase run requires a positive --port, got {args.port}")
    if args.phase == "preflight" and args.preflight_port <= 0:
        ap.error(f"--phase preflight requires a positive --preflight-port, "
                 f"got {args.preflight_port}")
    if args.world % args.local_shards:
        ap.error(f"--world {args.world} is not divisible by --local-shards "
                 f"{args.local_shards}")
    if args.node_rank >= args.world // args.local_shards:
        ap.error(f"--node-rank {args.node_rank} is not < node count "
                 f"{args.world // args.local_shards} (world {args.world} / "
                 f"local-shards {args.local_shards})")
    if args.phase == "preflight":
        _node_worker_preflight(args)
    else:
        _node_worker_run(args)


# ----------------------------------------------------------------------------
# process supervision (bounded, loud)
# ----------------------------------------------------------------------------

def _kill_tree(proc):
    """SIGKILL the subprocess's whole process group (start_new_session made
    it the group leader): local node-workers take their mp.spawn children
    down with them; an ssh client takes its remote command down with the
    channel. For ssh-executor nodes the remote worker's death therefore
    relies on sshd delivering SIGHUP when the channel closes (bounded by
    the ServerAlive watchdog); failure paths additionally sweep the nodes
    out of band (_sweep_node_workers)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


_REAP_GRACE_S = 10.0


def _kill_and_reap(procs):
    """SIGKILL every still-live proc and bounded-wait() it, so the launcher
    never leaves a zombie or an orphan behind: used by the supervisor's
    kill paths and by the orchestrator's outer finally (idempotent)."""
    for p in procs:
        if p.poll() is None:
            _kill_tree(p)
    deadline = time.monotonic() + _REAP_GRACE_S
    for p in procs:
        if p.poll() is None:
            try:
                p.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


def _log_tail(path, lines=40):
    try:
        with open(path, errors="replace") as f:
            tail = f.readlines()[-lines:]
        return "".join(tail) or "(empty log)"
    except OSError as e:
        return f"(log unreadable: {e!r})"


def _launch_error(msg, nodes, rcs, log_paths, tail=40):
    """Assemble the loud failure text: what died, where, and the last log
    lines of every node's worker (no post-mortem ssh needed)."""
    out = [msg]
    for m in sorted(log_paths):
        out.append(f"--- node {m} ({nodes[m]}) "
                   f"rc={rcs.get(m, 'killed')} log={log_paths[m]} ---")
        out.append(_log_tail(log_paths[m], tail))
    return "\n".join(out)


def _collect(procs):
    """Non-blocking reap; returns (rcs_done, pending) snapshots."""
    rcs, pending = {}, []
    for m, p in enumerate(procs):
        rc = p.poll()
        if rc is None:
            pending.append(m)
        else:
            rcs[m] = rc
    return rcs, pending


def _supervise(procs, *, deadline_s=None, kill_on_early_bad=False,
               poll_s=2.0):
    """Shared supervision loop for both phases: reap via _collect
    until every proc resolved, with two policies layered on top:

    - kill_on_early_bad: on the first nonzero rc, SIGKILL every pending
      proc and return immediately — the run phase's dead-peer policy, and the preflight's too: one node failing its rendezvous must not
      hold the launch gate open for the stragglers;
    - deadline_s: hard wall-clock bound on the whole phase — the preflight
      rendezvous bound, or (run phase) the opt-in --run-deadline:
      on expiry every pending proc is killed.

    Returns (rcs, pending): rcs maps finished indices to return codes;
    pending lists the indices killed at the deadline/early-bad point (they
    have been reaped; the diagnostics report them as rc 'killed')."""
    deadline = (time.monotonic() + deadline_s) if deadline_s else None
    rcs, pending = _collect(procs)
    while pending and (deadline is None or time.monotonic() < deadline):
        if kill_on_early_bad and any(rc != 0 for rc in rcs.values()):
            break
        time.sleep(poll_s)
        rcs, pending = _collect(procs)
    if pending:
        _kill_and_reap([procs[m] for m in pending])
    return rcs, pending


# Grace over the preflight rendezvous timeout before the launcher kills a
# node's worker (ssh connect + conda + import overhead); a constant so the
# bounded-failure tests can tighten it.
_PREFLIGHT_MARGIN_S = 90.0


# ----------------------------------------------------------------------------
# dense-cache publish confirmation (out of band, one-shot, never a poll)
# ----------------------------------------------------------------------------

def _cache_probe_argv(node, cache_path, ssh_key=None):
    """The one-shot remote stat argv (factored out so the unit tests can
    pin the transport policy exactly)."""
    probe = "import os, sys; print(os.path.getsize(sys.argv[1]))"
    return _ssh_base_argv(node, ssh_key) + [
        f"python3 -c {shlex.quote(probe)} {shlex.quote(cache_path)}"]


def _parse_probe_stdout(node, cache_path, stdout, stderr):
    """The probe's byte count from ssh stdout: the LAST non-empty
    line must be a bare integer — motd/sitecustomize noise elsewhere on the
    channel is tolerated; anything unparseable becomes the named
    diagnostic instead of a bare ValueError."""
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    if lines and re.fullmatch(r"\d+", lines[-1]):
        return int(lines[-1])
    raise RuntimeError(
        f"dense-cache publish confirmation failed on node {node} "
        f"for {cache_path}: unparseable probe output "
        f"stdout={stdout.strip()!r} stderr={stderr.strip()!r}")


def _confirm_cache_visible(node, cache_path, expected_bytes, *,
                           executor="ssh", ssh_key=None, timeout_s=120.0):
    """One-shot confirmation that the published dense cache is visible from
    ``node``'s OWN NFS client, with the exact byte size the node-0 parent
    wrote. This is rendezvous, not validation: each shard re-validates the
    cache ident on load (bitwise authority in load_dense_cache). Deliberately
    NOT a poll loop — cross-client visibility on this cluster can lag minutes
    to hours (negative dentry / cross-frontend ENOENT), so a
    failure stops the launch loudly BEFORE any shard exists."""
    if executor == "local":
        # single-machine simulation: one filesystem by construction
        got = os.path.getsize(cache_path)
    else:
        argv = _cache_probe_argv(node, cache_path, ssh_key)
        try:
            cp = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"dense-cache publish confirmation timed out after "
                f"{timeout_s:.0f}s on node {node} for {cache_path} "
                f"(wrote {expected_bytes} bytes): the node's NFS client has "
                "not caught up (cross-frontend propagation can take minutes "
                "to hours on this cluster). Nothing was launched; verify/"
                "warm the file on that node manually and rerun.") from None
        if cp.returncode != 0:
            raise RuntimeError(
                f"dense-cache publish confirmation failed on node {node} "
                f"for {cache_path}: rc={cp.returncode} "
                f"stderr={cp.stderr.strip()!r}")
        got = _parse_probe_stdout(node, cache_path, cp.stdout, cp.stderr)
    if got != expected_bytes:
        raise RuntimeError(
            f"dense-cache publish mismatch on node {node}: {cache_path} "
            f"reads {got} bytes but the node-0 parent wrote "
            f"{expected_bytes}; refusing to launch shards against a stale "
            "or partial cache")


def _sweep_node_workers(nodes, *, executor, ssh_key=None):
    """Best-effort out-of-band cleanup for failure/interrupt paths:
    the launcher's kill already took each ssh client down and the remote
    worker should die with the channel's HUP — this sweep pkills any
    leftover node-worker on every node anyway, so a missed HUP cannot leave
    GPUs pinned past a failed launch. Never raises; skipped for the local
    executor (one filesystem — the local kill is authoritative) and when no
    ssh key can be found (no node was ever contacted)."""
    if executor == "local":
        return
    try:
        key = ssh_key or _find_ssh_key()
    except ValueError:
        return
    for node in nodes:
        try:
            subprocess.run(
                _ssh_base_argv(node, key)
                + ["pkill -f 'replay_launch.py --node-worker' || true"],
                capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass


# ----------------------------------------------------------------------------
# node command rendering
# ----------------------------------------------------------------------------

def _node_argv_tail(phase, master, port, pf_port, node_rank, local_shards,
                    world, device, cfg_b64, frag_path, pf_timeout):
    tail = [
        "--node-worker", "--phase", phase,
        "--master", master,
        "--port", str(port),
        "--preflight-port", str(pf_port),
        "--node-rank", str(node_rank),
        "--local-shards", str(local_shards),
        "--world", str(world),
        # f"{x}" not str(int(x)): a fractional timeout must not truncate
        # to 0 at the child
        "--preflight-timeout", f"{pf_timeout}",
    ]
    if phase == "preflight":
        # preflight has no cfg, so its CUDA shard gate takes the device
        # here; the run phase's authority is cfg["device"] via --cfg-b64
        # (a --device on the run tail would be dead argv)
        tail += ["--device", device]
    if phase == "run":
        tail += ["--cfg-b64", cfg_b64, "--frag-path", frag_path]
    return tail


def _render_node_cmd(node, node_rank, phase, *, sga_root, master, port,
                     pf_port, local_shards, world, device, cfg_b64,
                     frag_path, pf_timeout, executor, ssh_key):
    """The argv that runs one node's node-worker.

    executor="ssh": [ssh ... root@node, remote one-liner] — conda first,
    then the repo root as the /workspace-view path (host paths do not exist
    on the nodes), and an explicit sga_root PYTHONPATH prepend so the worker
    import never depends on env.sh (its megatron assertion is the caller's
    business — e.g. the smoke sources it itself — not the node-worker's).
    executor="local": the same node-worker via a local subprocess (the
    single-machine simulation used by the unit tests).
    """
    tail = _node_argv_tail(phase, master, port, pf_port, node_rank,
                           local_shards, world, device, cfg_b64, frag_path,
                           pf_timeout)
    if executor == "local":
        env = dict(os.environ)
        env["PYTHONPATH"] = (sga_root + os.pathsep
                             + env.get("PYTHONPATH", "")).rstrip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        return {"argv": [sys.executable,
                         os.path.join(sga_root, "sga", "replay_launch.py"),
                         *tail],
                "env": env}
    if executor != "ssh":
        raise ValueError(f"unknown executor {executor!r}")
    quoted = " ".join(shlex.quote(t) for t in tail)
    remote = (
        "source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || "
        "source /root/miniconda3/etc/profile.d/conda.sh; "
        "conda activate flagscale; "
        f"cd {shlex.quote(sga_root)}; "
        f"PYTHONPATH={shlex.quote(sga_root + os.pathsep)}"
        "${PYTHONPATH:-} PYTHONUNBUFFERED=1 python "
        f"{shlex.quote(os.path.join('sga', 'replay_launch.py'))} {quoted}"
    )
    return {"argv": _ssh_base_argv(node, ssh_key) + [remote], "env": None}


def _spawn_node(node, node_rank, phase, render, log_dir):
    log_path = os.path.join(
        log_dir, f"{phase}_node{node_rank}_{node.replace('.', '_')}.log")
    log = open(log_path, "wb")
    try:
        proc = subprocess.Popen(render["argv"], stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True, env=render["env"])
    finally:
        log.close()
    return proc, log_path


# ----------------------------------------------------------------------------
# the orchestrator (called from sga.replay main with --nodes)
# ----------------------------------------------------------------------------

def _require_launcher_visible_frag(frag_path):
    """Guard the ssh-to-self assumption behind the frag_path handoff:
    node 0's worker runs on node 0 through the same ssh-rendered command as
    every other node, so the fragment is launcher-visible only if that
    ssh-to-self lands back in the launcher's mount namespace. A missing or
    empty fragment means it did not — name THAT here instead of dying in
    torch.load with an opaque EOF/unpickling error far from the cause."""
    size = os.path.getsize(frag_path) if os.path.isfile(frag_path) else -1
    if size <= 0:
        raise RuntimeError(
            f"node-0 worker did not write the launcher-visible fragment "
            f"({frag_path}): ssh-to-self must land in the launcher's "
            "filesystem view (same mount namespace) for the frag_path "
            "handoff — the worker either never reached global rank 0 or "
            "wrote its fragment into a different filesystem")


def run_fused_branch_multinode(args, ctx, t0, nodes, *, executor=None,
                               ssh_key=None, preflight_timeout=None,
                               log_dir=None):
    """One branch point through the fused engine across M nodes; returns the
    complete branch dict (same JSON shape as the single-node path). The
    calling process must run on node 0 and share the NFS view of the other
    nodes (paths verbatim, /workspace view inside containers)."""
    import sga.replay as R
    from sga.replay_fused import (_common_gates, _fused_cfg,
                                  _populate_dense_cache)

    executor = executor or os.environ.get("SGA_LAUNCH_EXECUTOR", "ssh")
    preflight_timeout = (preflight_timeout if preflight_timeout is not None
                         else float(getattr(args, "preflight_timeout", 600.0)))
    if preflight_timeout <= 0:
        raise ValueError(
            f"--preflight-timeout must be > 0, got {preflight_timeout}")
    run_deadline = float(getattr(args, "run_deadline", 0.0) or 0.0)
    if run_deadline < 0:
        raise ValueError(
            f"--run-deadline must be >= 0 (0 disables the bound), "
            f"got {run_deadline}")
    score_names = _common_gates(args)
    world = args.fused_shards
    if world < 1:
        raise ValueError("--fused-shards must be >= 1")
    # a relative --dense-cache must name ONE file for every consumer:
    # the node-0 parent would resolve it against ITS cwd, the ssh confirm
    # probe against the node's login $HOME (no cd) and the node-workers
    # against sga_root (cd) — normalize once, here, before any use or
    # serialization (the cfg ships it verbatim to every node)
    if args.dense_cache:
        args.dense_cache = os.path.abspath(args.dense_cache)
    nodes = [str(n).strip() for n in nodes if str(n).strip()]
    if not nodes:
        raise ValueError("--nodes names no node")
    for n in nodes:
        # each token lands in a log filename and an ssh argv
        if not re.fullmatch(r"[A-Za-z0-9._-]+", n):
            raise ValueError(
                f"--nodes token {n!r} contains characters outside "
                "[A-Za-z0-9._-] (each token lands in a log filename and "
                "an ssh argv)")
    if executor != "local" and len(set(nodes)) != len(nodes):
        # duplicates over the real transport co-locate two node_workers'
        # shard sets on one host: silent GPU oversubscription and a
        # doubly-sliced topology. (executor=local is exempt: the
        # single-machine simulation co-locates "nodes" on 127.0.0.1 by
        # design.)
        raise ValueError(f"--nodes contains duplicates: {nodes}")
    m_nodes = len(nodes)
    if world % m_nodes:
        raise ValueError(
            f"--fused-shards {world} is not divisible by the --nodes count "
            f"{m_nodes}: shard r lives on node r // (shards / nodes), so "
            "every node must carry the same shard count")
    local_shards = world // m_nodes
    sga_root = _sga_root()

    # 1. node-0 parent: config + dense-cache populate on miss (before any
    # shard exists — the dense pass must never run inside the process
    # group), then the publish is confirmed out of band, never polled.
    cfg = _fused_cfg(args, ctx, t0, score_names)
    _ws, _eff, ident = _populate_dense_cache(args, ctx, cfg, t0)
    cache_path = R._dense_cache_path(args.dense_cache, ident)
    cache_bytes = os.path.getsize(cache_path)
    for node_rank, node in enumerate(nodes[1:], start=1):
        _confirm_cache_visible(node, cache_path, cache_bytes,
                               executor=executor, ssh_key=ssh_key)
    print(f"[fused-multinode] t0={t0} dense cache confirmed on "
          f"{m_nodes - 1} remote node(s): {cache_path} ({cache_bytes} B)",
          flush=True)

    # 2. rendezvous endpoints + fragment + logs
    pf_port, port = _pick_ports()
    master = nodes[0]
    fd, frag_path = tempfile.mkstemp(suffix=".pt", prefix="fused_branch_")
    os.close(fd)
    cfg_b64 = _cfg_to_b64(cfg)
    if log_dir is None:
        log_dir = getattr(args, "launch_log_dir", None) or os.path.join(
            os.path.dirname(os.path.abspath(frag_path)),
            f"fused_launch_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[fused-multinode] t0={t0} nodes={nodes} world={world} "
          f"({local_shards}/node) master={master} rendezvous "
          f"preflight={pf_port} run={port} logs={log_dir} "
          f"executor={executor}", flush=True)

    def render(node, node_rank, phase):
        # the one render closure both spawn loops share: identical
        # kwarg block, only node/rank/phase vary
        return _render_node_cmd(
            node, node_rank, phase, sga_root=sga_root, master=master,
            port=port, pf_port=pf_port, local_shards=local_shards,
            world=world, device=cfg["device"], cfg_b64=cfg_b64,
            frag_path=frag_path, pf_timeout=preflight_timeout,
            executor=executor, ssh_key=ssh_key)

    # both collections exist before the try, so the outer finally can clean
    # up whichever phase the failure happened in
    pre_procs, pre_logs = [], {}
    run_procs, run_logs = [], {}
    try:
        # 3. bounded preflight: every node's shard set rendezvouses on a
        # throwaway endpoint under a launch-bound deadline; a node that
        # cannot start fails here in minutes, loudly, per node — and a node
        # failing FAST kills the stragglers instead of holding the gate
        # open for the full deadline
        for node_rank, node in enumerate(nodes):
            proc, log_path = _spawn_node(node, node_rank, "preflight",
                                         render(node, node_rank, "preflight"),
                                         log_dir)
            pre_procs.append(proc)
            pre_logs[node_rank] = log_path
        rcs, pre_pending = _supervise(
            pre_procs,
            deadline_s=preflight_timeout + _PREFLIGHT_MARGIN_S,
            kill_on_early_bad=True, poll_s=1.0)
        pre_bad = {m: rc for m, rc in rcs.items() if rc != 0}
        if pre_pending and not pre_bad:
            raise RuntimeError(_launch_error(
                f"fused multi-node launch: preflight rendezvous exceeded "
                f"its {preflight_timeout:.0f}s deadline on node(s) "
                f"{[nodes[m] for m in sorted(pre_pending)]} — refusing to "
                "start the run (a node that cannot meet a bounded launch "
                "gate would hang the run on the 2h gloo timeout)",
                nodes, rcs, pre_logs))
        if pre_bad:
            raise RuntimeError(_launch_error(
                f"fused multi-node launch: preflight rendezvous FAILED on "
                f"node(s) {[nodes[m] for m in sorted(pre_bad)]} "
                f"(rc={[pre_bad[m] for m in sorted(pre_bad)]}) — "
                "no shards were started", nodes, rcs, pre_logs))
        print(f"[fused-multinode] t0={t0} preflight rendezvous OK on all "
              f"{m_nodes} node(s)", flush=True)

        # 4. the run: one node-worker per node through the same rendered
        # command (node 0 by ssh-to-self in production), supervised: an
        # early death anywhere stops everything, and the opt-in
        # --run-deadline bounds a shard hung outside a collective
        # (unbounded by default — see the module docstring).
        for node_rank, node in enumerate(nodes):
            proc, log_path = _spawn_node(node, node_rank, "run",
                                         render(node, node_rank, "run"),
                                         log_dir)
            run_procs.append(proc)
            run_logs[node_rank] = log_path
        rcs, run_pending = _supervise(run_procs,
                                      deadline_s=run_deadline or None,
                                      kill_on_early_bad=True)
        run_bad = {m: rc for m, rc in rcs.items() if rc != 0}
        if run_pending and run_bad:
            raise RuntimeError(_launch_error(
                f"fused multi-node run: node(s) "
                f"{[nodes[m] for m in sorted(run_bad)]} exited "
                f"rc={[run_bad[m] for m in sorted(run_bad)]} while "
                f"{[nodes[m] for m in sorted(run_pending)]} were still "
                "running — survivors killed (a dead peer would only "
                "surface as the 2h gloo timeout otherwise)",
                nodes, rcs, run_logs))
        if run_pending:
            raise RuntimeError(_launch_error(
                f"fused multi-node run: exceeded its --run-deadline "
                f"{run_deadline:.0f}s wall clock on node(s) "
                f"{[nodes[m] for m in sorted(run_pending)]} — every "
                "node-worker was killed (a shard hung outside a collective "
                "surfaces here instead of hanging the launcher forever)",
                nodes, rcs, run_logs))
        if run_bad:
            raise RuntimeError(_launch_error(
                f"fused multi-node run FAILED: node(s) "
                f"{[nodes[m] for m in sorted(run_bad)]} exited "
                f"rc={[run_bad[m] for m in sorted(run_bad)]}",
                nodes, rcs, run_logs))

        # 5. global rank 0 (node 0's first shard) wrote the fragment locally
        _require_launcher_visible_frag(frag_path)
        out = torch.load(frag_path, map_location="cpu", weights_only=False)
    except BaseException:
        # failure/interrupt path: reap locally, then sweep the nodes out
        # of band (see _sweep_node_workers). Clean success skips the sweep.
        _kill_and_reap(pre_procs + run_procs)
        _sweep_node_workers(nodes, executor=executor, ssh_key=ssh_key)
        raise
    finally:
        _kill_and_reap(pre_procs + run_procs)
        if os.path.exists(frag_path):
            os.remove(frag_path)
    return out


if __name__ == "__main__":
    main()
