# Copyright (c) 2026, SGA framework contributors.
"""Fused single-pass streaming replay (config-grouped, param-sharded).

The removed classic path replayed the window once per arm (1 dense pass +
one cycle pass per ratio×score), so a sweep re-read the multi-TB window
~7× per run; this engine advances ALL arms of one config group over the
window in lockstep — one window read per group — sharded across processes:

- ``--fused-shards`` shard processes (one per GPU; gloo over localhost for
  the tiny control-plane exchanges), each owning a contiguous slice of the
  global flat index space (catalog order). Every shard mmaps the same g
  files and reads only its slice; the host page cache dedups the NFS pulls
  (each file crosses the machine once per pass). A prefetch thread faults
  in the next step's slices while the current step computes.
- Per step the shards exchange only per-config clip-norm sums (fp64
  scalars); at cycle boundaries they additionally exchange the 32769-bin
  integer score histograms and per-shard (above, tie) counts — the global
  top-k stays exact, with the tie layer filled in global flat-index order
  (= shard order, then in-shard order), bit-identical to
  ``select_topk_flat`` on the concatenated vector.
- The dense arm is never re-run inside a pass: den_at / snapshots / label /
  recon come from the dense cache (``--dense-cache``); on a miss the parent
  process computes the dense pass with the classic machinery and saves the
  cache once (before the shards start), then every shard slices the
  artifacts.
- Multi-node (``sga/replay_launch.py``): shard r lives on node
  r // (world / M) at cuda:(r % (world / M)); the rendezvous is env-driven
  (SGA_MASTER_ADDR / SGA_MASTER_PORT / SGA_GLOBAL_RANK) and every rank-0
  role below is keyed by the GLOBAL rank, so "rank 0" always means the
  launcher node's first shard.

Arithmetic contract (historical): per-config math was op-identical to the
removed ``sparse_replay_cycle`` — per-param per-chunk operations in the
same order — so single-shard runs reproduced the classic report bitwise
(pinned by the pre-deletion e2e tiny-trace comparison). The standing spec
anchor is ``tests/test_replay_cycle.reference_cycle``. At world > 1 the
only deviations are cross-shard fp64 scalar-sum reorderings (covered by
the frozen pure-reordering band), plus the optional ``--norm-dtype fp32``
quantization (gated by its own frozen band). The
score/label SUM statistics are shard-partial sums (same reordering class);
min/max/zero-fractions stay exact. Ranking metrics are exact: the selected
coordinates' (score, label, label-sel) values are gathered to shard 0 in
global index order and the same arithmetic as ``ranking_metrics`` runs on
the reconstructed vectors.
"""
import datetime
import math
import os
import socket
import threading
import time
import queue

import torch
import torch.distributed as dist

# opt-in S0 profile (plan-level speedup baseline): counts host syncs / gloo
# ops / H2D bytes per pass and per-pass wall time; zero effect when unset
_S0_PROFILE = os.environ.get("REPLAY_S0_PROFILE", "") == "1"


# ----------------------------------------------------------------------------
# small comms wrapper: exact reductions, identity when world == 1
# ----------------------------------------------------------------------------

class Comms:
    """gloo wrapper for the tiny control-plane exchanges.

    Rendezvous: single-machine by default (tcp://127.0.0.1:{port}, rank ==
    process-group rank). A multi-node launcher instead sets, per process:
    SGA_MASTER_ADDR + SGA_MASTER_PORT (tcp rendezvous on the master node)
    and SGA_GLOBAL_RANK (the global process-group rank; world is passed
    pre-computed). In that mode init_process_group uses the global rank,
    ``self.rank`` stays the local (intra-node) shard index and
    ``self.global_rank`` is the process-group rank — consumers that index
    gathered collective results must use ``global_rank``.
    """

    def __init__(self, rank, world, port):
        master = os.environ.get("SGA_MASTER_ADDR")
        global_rank = os.environ.get("SGA_GLOBAL_RANK")
        if master is None:
            if global_rank is not None:
                raise ValueError(
                    "SGA_GLOBAL_RANK is set but SGA_MASTER_ADDR is not; "
                    "the global rank only means something with the "
                    "multi-node master rendezvous")
            self.global_rank = rank
            init_method = f"tcp://127.0.0.1:{port}"
        else:
            if "SGA_MASTER_PORT" not in os.environ:
                raise ValueError(
                    "SGA_MASTER_ADDR is set but SGA_MASTER_PORT is not; "
                    "set both to the master node's tcp rendezvous endpoint")
            self.global_rank = (int(global_rank) if global_rank is not None
                                else rank)
            init_method = f"tcp://{master}:{os.environ['SGA_MASTER_PORT']}"
        self.rank, self.world = rank, world
        self.distributed = world > 1
        if self.distributed:
            # explicit generous timeout: the 30-min gloo default would kill
            # a healthy run whose ranks legitimately take long to reconvene
            dist.init_process_group(backend="gloo", rank=self.global_rank,
                                    world_size=world,
                                    init_method=init_method,
                                    timeout=datetime.timedelta(hours=2))

    def sum_f64(self, vals):
        """Sum one python float per rank (list in, list of sums out — every
        rank gets the same fp64 sums)."""
        t = torch.tensor(vals, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()

    def all_gather_pad(self, t):
        """all_gather variable-length CPU 1-D tensors -> per-rank tensors in
        rank order (padding never enters the results)."""
        if not self.distributed:
            return [t]
        n = torch.tensor([t.numel()], dtype=torch.int64)
        ns = [torch.zeros(1, dtype=torch.int64) for _ in range(self.world)]
        dist.all_gather(ns, n)
        m = max(int(x.item()) for x in ns)
        buf = torch.zeros(m, dtype=t.dtype)
        buf[:t.numel()] = t
        out = [torch.zeros(m, dtype=t.dtype) for _ in range(self.world)]
        dist.all_gather(out, buf)
        return [out[r][:int(ns[r].item())] for r in range(self.world)]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _fresh_state(t, dev):
    """Copy a cpu reference tensor onto dev; never alias the original (the
    shard's w0/m0/v0 are shared across every arm in a config group)."""
    out = t.to(dev)
    if out is t:
        out = out.clone()
    return out


# ----------------------------------------------------------------------------
# sharded param table
# ----------------------------------------------------------------------------

class ShardParams:
    """The params overlapping one shard's [lo, hi) slice of the global flat
    index space, with local offsets and per-param hyperparameters."""

    def __init__(self, names, catalog, meta, lo, hi):
        from sga.replay import _param_hp
        self.lo, self.hi = lo, hi
        self.entries = []
        off = 0
        for name in names:
            nelt = int(catalog[name]["numel"])
            a, b = max(off, lo), min(off + nelt, hi)
            if b > a:
                b1, b2, eps, wd, group = _param_hp(meta, catalog.get(name))
                self.entries.append({
                    "name": name, "p_off": a - off,
                    "a": a - lo, "n": b - a,
                    "b1": b1, "b2": b2, "eps": eps, "wd": wd, "group": group,
                })
            off += nelt
        self.n = hi - lo


def shard_bounds(N, world):
    """Contiguous near-equal split of the flat index space; shard r owns
    [lo_r, hi_r) — the order that reproduces the global flat-index tie-fill."""
    base, rem = divmod(N, world)
    out, lo = [], 0
    for r in range(world):
        hi = lo + base + (1 if r < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


# ----------------------------------------------------------------------------
# exact global top-k over sharded non-negative fp32 score slices
# ----------------------------------------------------------------------------

def shard_select(score, k, comms):
    """Exact global top-k of a non-negative fp32 score vector sharded across
    ranks (each rank holds its contiguous slice; ``score`` may live on cpu or
    the shard's cuda device). Returns (local bool mask on the same device,
    global threshold) with the tie layer filled in GLOBAL flat-index order —
    bit-identical to select_topk_flat on the concatenated vector."""
    if not bool(torch.isfinite(score).all().item()):
        # NaN/Inf sit in the histogram's top bit-pattern bins: they would be
        # selected FIRST, silently displacing real coordinates from the topk.
        raise ValueError(
            "shard_select requires finite scores (NaN/Inf would land in "
            "the top bins and silently pollute the topk)")
    top16 = ((score.view(torch.int32) & 0x7FFFFFFF) >> 16)
    hist = torch.bincount(top16, minlength=32769).cpu()
    if comms.distributed:
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
    cum = torch.cumsum(hist, 0)
    total = int(cum[-1].item())
    if k > total:
        raise ValueError(f"k={k} > N={total}")
    r = total - k + 1
    b = int(torch.searchsorted(cum, r).item())
    n_above = total - int(cum[b].item())

    # exact k-th value: kthvalue over the union of the shards' candidates in
    # bin b (the same multiset the classic hist_threshold refines over)
    cand = score[top16 == b].cpu().clone()
    parts = comms.all_gather_pad(cand)
    whole = torch.cat(parts)
    kk = whole.numel() - (k - n_above) + 1
    if whole.numel() == 0 or not (1 <= kk <= whole.numel()):
        raise RuntimeError(
            f"fused hist select math out of range: cand={whole.numel()} "
            f"kk={kk} r={r}")
    thresh = float(torch.kthvalue(whole, kk).values.item())

    # tie-fill: first `need` tied coords in global index order = shard
    # order, then in-shard order. counting runs chunked and the fill marks
    # the tie layer's nonzero indices: bool .sum()/.count_nonzero()/.cumsum()
    # all materialize a full-size int64 copy (9+ GB per call at 1.2B elements
    # on one shard), which OOMs the world=1 oracle mode
    gt = score > thresh
    eq = score == thresh

    def _count_bool(t):
        total = 0
        for a in range(0, t.numel(), 64 * 1024 * 1024):
            total += int(t[a:a + 64 * 1024 * 1024].sum().item())
        return total

    counts = torch.tensor([_count_bool(gt), _count_bool(eq)],
                          dtype=torch.int64)
    allc = [torch.zeros(2, dtype=torch.int64) for _ in range(comms.world)]
    if comms.distributed:
        dist.all_gather(allc, counts)
    else:
        allc[0] = counts
    need = k - sum(int(c[0].item()) for c in allc)
    take = need
    # allc is in process-group rank order, so the shards before this one
    # are counted by global_rank (== rank on the single-machine default)
    for rr in range(comms.global_rank):
        take -= int(allc[rr][1].item())
    take = max(0, min(take, int(counts[1].item())))
    if take > 0:
        tie_idx = torch.nonzero(eq, as_tuple=False).squeeze(1)
        gt[tie_idx[:take]] = True
    return gt, thresh


# ----------------------------------------------------------------------------
# ranking metrics over gathered selections (exact; mirrors ranking_metrics)
# ----------------------------------------------------------------------------

def _device_ok_for_sort(numel, dev):
    """Conservative VRAM guard for computing one permutation on ``dev``
    (cub radix SortPairs double-buffers keys+indices ≈ 20-24 B/elem; this
    books 32). Runtime free-mem based — static accounting lies on shared
    nodes. Falls back (loudly) to the CPU path, which is bit-identical."""
    if dev is None:
        return False
    # mem_get_info is device-wide and therefore counts THIS process's own
    # cached-but-free allocator blocks (e.g. a previous sort's cub temp) as
    # used — release them before measuring or big sorts get refused despite
    # physically fitting. No sort tensor is live between calls (the .cpu()
    # fetch syncs), so only genuinely freeable blocks go back.
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(dev)
    need = numel * 32
    ok = need < free * 0.85
    if not ok:
        print(f"[fused] ranking permutation on {dev} skipped: "
              f"need ~{need / 2 ** 30:.1f} GiB, free {free / 2 ** 30:.1f} "
              f"GiB — CPU fallback (bit-identical)", flush=True)
    return ok


def _stable_order_desc(keys, dev):
    """Descending stable argsort as a CPU int64 tensor. With ``dev`` set
    (and enough VRAM) the permutation is computed on the device and
    fetched back; ``stable=True`` guarantees the same permutation on both
    backends. FINITENESS of the callers' keys is enforced upstream (the
    shard_select isfinite gate for scores, the one-time label assertion in
    _worker); non-negativity is only structural — scores are written via
    .abs(), labels are sums of squares — and NOT separately checked. A
    finite NEGATIVE key would still sort identically across backends (the
    only cross-backend key-class split is ±0.0, which can affect
    permutations only if it appears in the ranking KEYS; a -0.0 label
    touches sorted VALUES only, which is report-safe). Reductions never
    run here — they stay on the callers' original CPU expression path, so
    results are bit-identical with or without a device."""
    if _device_ok_for_sort(keys.numel(), dev):
        try:
            return torch.argsort(keys.to(dev), descending=True,
                                 stable=True).cpu()
        except torch.cuda.OutOfMemoryError:
            # TOCTOU: free VRAM can shrink between the guard's mem_get_info
            # and the cub allocation on a shared node — complete the
            # loud-CPU-fallback contract instead of aborting the run
            print(f"[fused] ranking permutation on {dev} hit CUDA OOM — "
                  "CPU fallback (bit-identical)", flush=True)
    return torch.argsort(keys, descending=True, stable=True)


def _fused_ranking(sc_all, lb_all, lsel_parts, k, total_score, total_label,
                   idcg, dev=None):
    """The classic ranking_metrics arithmetic on the gathered selected
    vectors (global index order), with recall from the gathered label-sel
    overlap counts. ``dev`` only accelerates the permutation (see
    _stable_order_desc); every sum/conversion below is untouched."""
    pos = torch.arange(1, k + 1, dtype=torch.float64)
    wts = 1.0 / torch.log2(pos + 1.0)
    order = _stable_order_desc(sc_all, dev)
    dcg = float((lb_all[order].to(torch.float64) * wts).sum())
    if idcg > 0.0:
        ndcg = dcg / idcg
    else:
        ndcg = 1.0 if dcg == 0.0 else 0.0
    recall = sum(int(p.sum().item()) for p in lsel_parts) / float(k)
    sel_score_sum = float(sc_all.sum())
    sel_label_sum = float(lb_all.sum())
    if total_label > 0:
        missed_label = 1.0 - sel_label_sum / total_label
    else:
        missed_label = 0.0 if sel_label_sum == 0 else 1.0
    if total_score > 0:
        missed_score = 1.0 - sel_score_sum / total_score
    else:
        missed_score = 0.0 if sel_score_sum == 0 else 1.0
    return {
        "ndcg_at_k": ndcg,
        "recall_at_k": recall,
        "missed_label_mass": missed_label,
        "missed_score_mass": missed_score,
    }


def _idcg_of_parts(parts, dev=None):
    """idcg over gathered label-sel values (already in global index order;
    idcg's sort is order-stable, so the value matches the classic path).
    ``dev`` only accelerates the sort — the sorted values come back to
    CPU before the fp64 conversion/weighting (bit-identical either way)."""
    whole = torch.cat(parts)
    rel2 = None
    if _device_ok_for_sort(whole.numel(), dev):
        try:
            rel2 = torch.sort(whole.to(dev), descending=True,
                              stable=True).values.cpu()
        except torch.cuda.OutOfMemoryError:
            # same guard-allocation TOCTOU window as _stable_order_desc —
            # loud CPU fallback, bit-identical by construction
            print(f"[fused] idcg sort on {dev} hit CUDA OOM — "
                  "CPU fallback (bit-identical)", flush=True)
    if rel2 is None:
        rel2 = torch.sort(whole, descending=True, stable=True).values
    pos = torch.arange(1, whole.numel() + 1, dtype=torch.float64)
    wts = 1.0 / torch.log2(pos + 1.0)
    return float((rel2.to(torch.float64) * wts).sum())


# ----------------------------------------------------------------------------
# g-slice prefetch (fault the NFS pages of upcoming steps off the hot path)
# ----------------------------------------------------------------------------

class GPrefetcher:
    """Background thread materializing the shard's per-step g slices as
    contiguous fp32 CPU tensors (page-cache faults + upcast overlap the
    compute of the current step); the compute side only pays the H2D."""

    def __init__(self, g_files, window_steps, sp, upcast_bf16, depth=2,
                 pin=False):
        self.g_files, self.window_steps, self.sp = g_files, window_steps, sp
        self.upcast_bf16 = upcast_bf16
        # pinned host buffers make the consumer's H2D non_blocking-able;
        # requires CUDA (callers gate this on the device)
        self.pin = pin
        self.q = queue.Queue(maxsize=depth)
        self.exc = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        from sga.replay import _load
        try:
            for gstep in self.window_steps:
                d = _load(self.g_files[gstep])
                pl = dict(d["payload"])
                bf16_keys = [k for k, v in pl.items()
                             if torch.is_tensor(v) and v.dtype == torch.bfloat16]
                if bf16_keys and not self.upcast_bf16:
                    raise ValueError(
                        f"{self.g_files[gstep]}: bf16 payload(s) {bf16_keys} "
                        "but --trace-dtype is fp32; rerun with --trace-dtype "
                        "bf16 (payloads are upcast to fp32 at load)")
                # no full-payload upcast: each entry slices its window of
                # the mmap'd payload FIRST and upcasts just that slice
                # (slicing and the elementwise upcast commute, so the
                # values are identical — but only the slice's pages fault
                # and only the slice is converted)
                local = {}
                for e in self.sp.entries:
                    t = pl.get(e["name"])
                    if t is None:
                        continue
                    src = t.reshape(-1)[e["p_off"]:e["p_off"] + e["n"]]
                    out_dtype = (torch.float32
                                 if src.dtype == torch.bfloat16
                                 else src.dtype)
                    if self.pin:
                        buf = torch.empty(src.shape, dtype=out_dtype,
                                          pin_memory=True)
                        buf.copy_(src)
                        local[e["name"]] = buf
                    elif out_dtype is not src.dtype:
                        local[e["name"]] = src.float()
                    else:
                        local[e["name"]] = src.clone()
                self.q.put((gstep, d["meta"], local))
        except BaseException as e:
            # die loudly in the consumer instead of leaving it blocked on
            # q.get() forever
            self.exc = e
            self.q.put(None)

    def __iter__(self):
        for _ in self.window_steps:
            item = self.q.get()
            if item is None:
                raise RuntimeError("g prefetch failed") from self.exc
            yield item


def _window_and_horizons(g_files, t0, window, horizons):
    """Window g steps + effective horizons — derived identically in the
    parent (dense-cache populate) and in every worker (streaming pass); both
    feed the dense-cache ident."""
    import sga.replay as R
    window_steps = R.window_steps_for(g_files, t0, window)
    if not window_steps:
        raise ValueError(f"fused t0={t0}: no g files in window")
    W = len(window_steps)
    eff_horizons = sorted({max(1, min(h, W)) for h in horizons})
    return window_steps, W, eff_horizons


# ----------------------------------------------------------------------------
# the shard worker
# ----------------------------------------------------------------------------

def _worker(rank, world, cfg, port, frag_path):
    import sga.replay as R

    torch.manual_seed(cfg["seed"])
    dev = "cpu" if cfg["device"] == "cpu" else f"cuda:{rank}"
    # sort_dev feeds the ranking/idcg sorts (device computes the
    # permutation; the guard inside falls back to CPU — see
    # _stable_order_desc)
    sort_dev = None if dev == "cpu" else dev
    # ops kill-switch (A/B arms and emergencies): force the bit-identical
    # CPU sort path — the S0FRONT print echoes the arm as front_sort=cpu.
    # Unknown values must fail loudly: a typo ("CPU", "1") would otherwise
    # silently leave the emergency knob a no-op.
    front_sort = os.environ.get("REPLAY_FUSED_FRONT_SORT", "auto")
    if front_sort not in ("auto", "cpu"):
        raise ValueError(
            f"REPLAY_FUSED_FRONT_SORT={front_sort!r} is not one of the "
            "allowed values ('auto', 'cpu')")
    if front_sort == "cpu":
        sort_dev = None
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    comms = Comms(rank, world, port)

    t0 = cfg["t0"]
    replay_dtype = (torch.float64 if cfg["replay_dtype"] == "fp64"
                    else torch.float32)
    catalog, names, offsets, N, state_files, g_files = R.discover_trace(
        cfg["trace_dir"])
    window_steps, W, eff_horizons = _window_and_horizons(
        g_files, t0, cfg["window"], cfg["horizons"])

    # the shard's slice of the flat index space is keyed by the GLOBAL rank
    # (multi-node: the launcher hands node m the local ranks m*g..m*g+g-1, so
    # a local-rank key would give every node the same slices); identical to
    # the local rank on the single-machine default path
    lo, hi = shard_bounds(N, world)[comms.global_rank]
    st = R._load(state_files[t0])
    meta = st["meta"]
    sp = ShardParams(names, catalog, meta, lo, hi)
    # local branch-state slices (cloned into replay_dtype; only local pages
    # of the mmap'd state file are ever faulted)
    for e in sp.entries:
        start = e["p_off"]
        pw = st["payload"]["w"].get(e["name"])
        if pw is None:
            # named, before any collective runs (classic build_branch_state
            # raises the same error)
            raise ValueError(
                f"state file {state_files[t0]} missing w for {e['name']}")
        e["w0"] = pw.reshape(-1)[start:start + e["n"]].to(replay_dtype) \
            .clone()
        for k3 in ("m", "v"):
            pt = st["payload"].get(k3, {}).get(e["name"])
            if pt is None:
                pt = torch.zeros_like(pw)
            e[k3 + "0"] = pt.reshape(-1)[start:start + e["n"]] \
                .to(replay_dtype).clone()
        pg = st["payload"].get("g", {}).get(e["name"])
        if pg is None:
            pg = torch.zeros_like(pw)
        e["g0"] = pg.reshape(-1)[start:start + e["n"]].float().clone()
    del st

    # ---- branch scores (per score_name, elementwise fp64 math -> fp32) ----
    _tf0 = time.perf_counter() if _S0_PROFILE else 0.0
    score_names = cfg["score_names"]
    specs = {s: R.get_score_spec(s) for s in score_names}
    t0step = int(meta["step"])
    chunk_n = R._chunk_elems(cfg["chunk_mb"], 8)
    score_loc = {s: torch.empty(sp.n, dtype=torch.float32)
                 for s in score_names}
    for e in sp.entries:
        b1, b2, eps, wd = e["b1"], e["b2"], e["eps"], e["wd"]
        bc1p = 1.0 - b1 ** (t0step + 1)
        bc2p = 1.0 - b2 ** (t0step + 1)
        for a in range(0, e["n"], chunk_n):
            b = min(e["n"], a + chunk_n)
            w64 = e["w0"][a:b].to(dtype=torch.float64, device=dev)
            m64 = e["m0"][a:b].to(dtype=torch.float64, device=dev)
            v64 = e["v0"][a:b].to(dtype=torch.float64, device=dev)
            g64 = e["g0"][a:b].to(dtype=torch.float64, device=dev)
            for s in score_names:
                sc = specs[s].formula(w64, m64, v64, g64, b1, b2, eps, wd,
                                      bc1p, bc2p,
                                      update_includes_wd=cfg["cycle_mode"])
                score_loc[s][e["a"] + a:e["a"] + b] = sc.abs()

    # ---- dense artifacts: the parent populated the cache before spawn -----
    # (the cache roundtrip is bitwise — pinned by test_dense_cache_roundtrip)
    if _S0_PROFILE:
        _tf_scores = time.perf_counter() - _tf0
        _tf0 = time.perf_counter()
    else:
        _tf_scores = 0.0
    ident = R._dense_cache_ident(cfg["trace_dir"], t0, window_steps,
                                 eff_horizons, replay_dtype,
                                 cfg["trace_dtype"],
                                 "cpu" if cfg["device"] == "cpu" else "dev")
    cached = R.load_dense_cache(cfg["dense_cache"], ident)
    if cached is None:
        raise RuntimeError("dense cache still missing after parent populate")
    label_loc = cached["label_flat"][lo:hi].clone()
    # the device-permutation fast path relies on finite non-negative sort
    # keys (score_loc passes the shard_select isfinite gate; label gets one
    # upfront check here — non-finite labels would sort inconsistently
    # across backends)
    if not bool(torch.isfinite(label_loc).all().item()):
        raise ValueError("dense-cache label vector contains non-finite "
                         "entries; check the trace/cache integrity")
    den_at = cached["den_at"]
    recon_steps = cached["recon_steps"]
    label_total = cached["label_total"]
    snap_loc = {}
    for h in eff_horizons:
        snap_loc[h] = {}
        for e in sp.entries:
            t = cached["snap_d"][h][e["name"]].reshape(-1)
            snap_loc[h][e["name"]] = t[e["p_off"]:e["p_off"] + e["n"]] \
                .clone()
    dense_cache_hit = cfg["dense_cache_hit"]  # parent-side load outcome;
    # the streaming pass itself only ever consumes the cache
    del cached
    if _S0_PROFILE:
        _tf_cache = time.perf_counter() - _tf0
    else:
        _tf_cache = 0.0

    # ---- score / label statistics (sums shard-partial; min/max exact) ------
    stats = {}
    for s in score_names:
        stats[s] = _shard_stats(score_loc[s], comms)
    stats_label = _shard_stats(label_loc, comms)

    # ---- branch selections + label selections + ranking metrics -----------
    ratios = cfg["ratios"]
    cycle_lens = cfg["cycle_lens"]
    ks = {r: max(1, int(round(r / 100.0 * N))) for r in ratios}
    idcg, sel0, thresh0, rankmet = {}, {}, {}, {}
    _tf_sel = 0.0
    _tf_rank = 0.0
    for r in ratios:
        _tq = time.perf_counter() if _S0_PROFILE else 0.0
        lsel, _lt = shard_select(label_loc, ks[r], comms)
        parts = comms.all_gather_pad(label_loc[lsel].clone())
        if _S0_PROFILE:
            _tf_sel += time.perf_counter() - _tq
            _tq = time.perf_counter()
        if comms.global_rank == 0:
            idcg[r] = _idcg_of_parts(parts, dev=sort_dev)
        if _S0_PROFILE:
            _tf_rank += time.perf_counter() - _tq
        for s in score_names:
            # the branch-point selection is a function of (score, ratio)
            # only, and shard_select is deterministic on its inputs — every
            # T arm of one (s, r) shares a single select instead of paying
            # len(cycle_lens) redundant global selects; sel0 is only read to
            # build the per-config device masks (never mutated)
            _tq = time.perf_counter() if _S0_PROFILE else 0.0
            sel, th = shard_select(score_loc[s], ks[r], comms)
            sc_parts = comms.all_gather_pad(score_loc[s][sel].clone())
            lb_parts = comms.all_gather_pad(label_loc[sel].clone())
            ls_parts = comms.all_gather_pad(lsel[sel].clone())
            if _S0_PROFILE:
                _tf_sel += time.perf_counter() - _tq
                _tq = time.perf_counter()
            met_sr = None
            if comms.global_rank == 0:
                met_sr = _fused_ranking(
                    torch.cat(sc_parts), torch.cat(lb_parts), ls_parts,
                    ks[r], stats[s]["sum"], label_total, idcg[r],
                    dev=sort_dev)
            if _S0_PROFILE:
                _tf_rank += time.perf_counter() - _tq
            for t in cycle_lens:
                sel0[(s, r, t)] = sel
                thresh0[(s, r, t)] = th
                if comms.global_rank == 0:
                    rankmet[(s, r, t)] = met_sr
    if _S0_PROFILE and comms.global_rank == 0:
        # front-phase decomposition (diagnostic labels, not exact
        # accounting): branch-score math / dense-cache load / shard_select
        # + gathers / ranking+idcg sorts
        print(f"S0FRONT rank={rank} scores_s={_tf_scores:.1f} "
              f"cache_s={_tf_cache:.1f} select_s={_tf_sel:.1f} "
              f"rank_s={_tf_rank:.1f} "
              f"front_sort={'cpu' if sort_dev is None else 'dev'}",
              flush=True)

    # ---- streaming passes over config groups ------------------------------
    # (score, ratio, T) arms; a single T enumerates exactly the historical
    # (score, ratio) order, so the max_configs partition is unchanged
    configs = [(s, r, t) for r in ratios for s in score_names
               for t in cycle_lens]
    # group-size cap: with a state byte budget set (>0) the cap is derived
    # from the resident {w,m,v,res} bytes of ONE config over the WIDEST
    # shard — ceil(N/world) elements on every rank, so the partition stays a
    # deterministic pure function (a per-shard sp.n basis could split the
    # ranks at the floor and tear the qn2/num2 collectives). The budget
    # overrides --fused-max-configs; the bool sel mask, the --chunk-mb
    # working set and g-residency buffers sit outside the accounting (the
    # plan-level budget table counts the four state tensors only).
    itemsize = torch.finfo(replay_dtype).bits // 8
    budget_bytes = cfg["state_budget_mb"] * 2 ** 20
    if budget_bytes > 0:
        per_cfg = ((N + world - 1) // world) * 4 * itemsize
        # per_cfg > 0 whenever N >= 1 (empty traces fail loudly in discover_trace)
        assert per_cfg > 0
        eff_cap = max(1, int(budget_bytes // per_cfg))
    else:
        eff_cap = cfg["max_configs"]
    groups = [configs[i:i + eff_cap]
              for i in range(0, len(configs), eff_cap)]
    num_at = {(s, r, t): {} for s in score_names for r in ratios
              for t in cycle_lens}
    # one counter dict per config group, stashed into shard 0's fragment
    # alongside the S0PROFILE print (env-gated: None when the env is absent)
    s0_profs = [] if _S0_PROFILE else None
    t_pass0 = time.time()
    for gi, group in enumerate(groups):
        states = {}
        # sel0[(s,r,t)] is the SAME tensor for every t of one (s,r) — key
        # the upload cache by (s, r, entry) so "same selection across T
        # arms" is expressed directly and the slice + upload happens once
        # per distinct selection instead of once per arm (the masks are
        # read-only downstream; reselection replaces the tensor, never
        # mutates in place, so aliasing is safe)
        sel_dev_cache = {}
        for (s, r, t) in group:
            sel = sel0[(s, r, t)]
            stt = {}
            for e in sp.entries:
                # _fresh_state: never alias the shard's w0/m0/v0 — every
                # arm in the group must start from the pristine state
                sel_key = (s, r, e["name"])
                if sel_key not in sel_dev_cache:
                    sel_dev_cache[sel_key] = \
                        sel[e["a"]:e["a"] + e["n"]].to(dev)
                stt[e["name"]] = {
                    "ms": _fresh_state(e["m0"], dev),
                    "vs": _fresh_state(e["v0"], dev),
                    "ws": _fresh_state(e["w0"], dev),
                    "res": torch.zeros(e["n"], dtype=replay_dtype,
                                       device=dev),
                    "sel": sel_dev_cache[sel_key],
                }
            states[(s, r, t)] = stt
        chunk_dev = R._chunk_elems(cfg["chunk_mb"], itemsize)
        norm_fp32 = cfg["norm_fp32"]

        res_on = bool(cfg.get("g_residency"))
        red_on = bool(cfg.get("dev_reduce"))
        pre = GPrefetcher(g_files, window_steps, sp, cfg["upcast_bf16"],
                          pin=(res_on and dev != "cpu"))
        # boundary steps without a g payload read zeros; the resident path
        # caches one device zeros buffer per entry instead of materializing
        # a host zeros tensor per (config, entry, step)
        zeros_dev = {}
        # S0 counters are diagnostic labels, not exact measurements:
        # gloo_ar counts logical sum_f64 calls (nonzero even at world 1,
        # where the op is an identity), syncs counts only the norm/numerator
        # fetches, mem_peak_gb (printed at group end) is the process-lifetime
        # cumulative peak, and residency books every shared upload byte to
        # pass 1
        prof = {"sync": 0, "ar": 0, "h2d": [0, 0, 0],
                "pt": [0.0, 0.0, 0.0]} if _S0_PROFILE else None
        if prof:
            # diagnostic memory accounting: the group's resident state at
            # its full per-rank truth ({w,m,v,res} at replay itemsize plus
            # the 1 B/elem bool sel) — the budget knob itself counts the
            # four state tensors only, this field books everything
            prof["state_gb"] = (len(group) * sp.n * (4 * itemsize + 1)
                                / 2 ** 30)
        for si, (_gstep, gmeta, local) in enumerate(pre):
            coef_dense, _flag = R.coef_dense_for(gmeta)
            clip_grad = float(gmeta.get("clip_grad", 0.0))
            count = t0 + si + 1
            # per-config calendars: arm (s, r, t) full-syncs iff si % t == 0.
            # Mixed-T runs reproduce solo-T arms bitwise because each arm's
            # state machine consumes only its own calendar/states, the
            # qn2/num2 exchanges are per-arm-indexed lists (elementwise
            # reductions — the list length never enters a lane's value) and
            # chunking / entry order are group-invariant
            boundary_c = [si % t == 0 for (_s, _r, t) in group]
            if res_on:
                # one H2D per entry per step, shared by every config and
                # all three passes (the legacy path re-uploads per config
                # per pass: max_configs x passes x the same bytes). The
                # coef_dense division is per-step too, so it is folded in
                # here once instead of per chunk per config per pass —
                # elementwise, identical values either way.
                g_res = {}
                for e in sp.entries:
                    if e["name"] in local:
                        g_res[e["name"]] = local[e["name"]].to(
                            dev, non_blocking=(dev != "cpu"))
                    else:
                        z = zeros_dev.get(e["name"])
                        if z is None:
                            z = torch.zeros(e["n"], dtype=torch.float32,
                                            device=dev)
                            zeros_dev[e["name"]] = z
                        g_res[e["name"]] = z
                # must stay out-of-place: boundary steps read the cached
                # zeros_dev buffer, an in-place div_ would corrupt the zeros
                g_res = {k: t / coef_dense for k, t in g_res.items()}
                if prof:
                    prof["h2d"][0] += sum(local[e["name"]].numel()
                                          for e in sp.entries
                                          if e["name"] in local)

            # pass 1: per-config clip norms (local sums -> fp64 allreduce)
            _t0p = time.perf_counter() if prof else 0.0
            qn2 = [0.0] * len(group)
            # dev_reduce: same addends, same chunk order — the per-chunk
            # device partials land in per-config fp64 device scalars and
            # reach the host once per pass instead of per chunk
            qn2_dev = (torch.zeros(len(group), dtype=torch.float64,
                                   device=dev) if red_on else None)
            for ci, (s, r, t) in enumerate(group):
                bnd = boundary_c[ci]
                for e in sp.entries:
                    g = local.get(e["name"])
                    if g is None and not bnd:
                        continue
                    g_el = None if res_on else (
                        torch.zeros(e["n"], dtype=torch.float32)
                        if g is None else g)
                    stt = states[(s, r, t)][e["name"]]
                    for a in range(0, e["n"], chunk_dev):
                        b = min(e["n"], a + chunk_dev)
                        if res_on:
                            gd = g_res[e["name"]][a:b]
                        else:
                            gd = g_el[a:b].to(dev) / coef_dense
                            if prof:
                                prof["h2d"][0] += b - a
                        if bnd:
                            q_pre = gd + stt["res"][a:b]
                        else:
                            q_pre = gd * stt["sel"][a:b]
                        if red_on:
                            if norm_fp32:
                                q32 = q_pre.to(torch.float32)
                                qn2_dev[ci] += (q32 * q32).sum()
                            else:
                                qn2_dev[ci] += (q_pre.double()
                                                ** 2.0).sum()
                        elif norm_fp32:
                            q32 = q_pre.to(torch.float32)
                            qn2[ci] += float((q32 * q32).sum())
                            if prof:
                                prof["sync"] += 1
                        else:
                            qn2[ci] += float((q_pre.double() ** 2.0).sum())
                            if prof:
                                prof["sync"] += 1
            if red_on:
                qn2 = qn2_dev.cpu().tolist()
                if prof:
                    prof["sync"] += 1
            qn2 = comms.sum_f64(qn2)
            if prof:
                prof["ar"] += 1
                prof["pt"][0] += time.perf_counter() - _t0p
            coefs = []
            for ci in range(len(group)):
                coef = 1.0
                if clip_grad > 0.0:
                    qnorm = math.sqrt(qn2[ci])
                    coef = min(1.0, clip_grad / (qnorm + 1.0e-6))
                coefs.append(coef)

            # boundary re-selection (same gating as
            # tests/test_replay_cycle.reference_cycle):
            # skip si=0 (sel0 encodes it) and the final boundary (no masked
            # step remains to consume the new mask); only arms whose own
            # calendar reaches this step re-select, the others leave mask
            # and states untouched
            _t0p = time.perf_counter() if prof else 0.0
            for ci, (s, r, t) in enumerate(group):
                if not (boundary_c[ci] and si > 0 and si + 1 < W):
                    continue
                spec = specs[s]
                score_flat = torch.empty(sp.n, dtype=torch.float32,
                                         device=dev)
                for e in sp.entries:
                    g = local.get(e["name"])
                    g_el = None if res_on else (
                        torch.zeros(e["n"], dtype=torch.float32)
                        if g is None else g)
                    stt = states[(s, r, t)][e["name"]]
                    b1, b2, eps, wd = e["b1"], e["b2"], e["eps"], e["wd"]
                    bc1c = 1.0 - b1 ** count
                    bc2c = 1.0 - b2 ** count
                    for a in range(0, e["n"], chunk_dev):
                        b = min(e["n"], a + chunk_dev)
                        if res_on:
                            q = (g_res[e["name"]][a:b]
                                 + stt["res"][a:b]) * coefs[ci]
                        else:
                            q = (g_el[a:b].to(dev) / coef_dense
                                 + stt["res"][a:b]) * coefs[ci]
                            if prof:
                                prof["h2d"][1] += b - a
                        sc = spec.formula(
                            stt["ws"][a:b], stt["ms"][a:b],
                            stt["vs"][a:b], q, b1, b2, eps, wd,
                            bc1c, bc2c, update_includes_wd=True)
                        score_flat[e["a"] + a:e["a"] + b] = sc.abs() \
                            .float()
                newsel, _th = shard_select(score_flat, ks[r], comms)
                for e in sp.entries:
                    stt = states[(s, r, t)][e["name"]]
                    stt["sel"] = newsel[e["a"]:e["a"] + e["n"]].to(dev)
                del score_flat
            if prof:
                prof["pt"][1] += time.perf_counter() - _t0p

            # pass 3: commit + residual + horizon numerators
            hz = (si + 1) if (si + 1) in eff_horizons else None
            _t0p = time.perf_counter() if prof else 0.0
            num2 = [0.0] * len(group)
            num2_dev = (torch.zeros(len(group), dtype=torch.float64,
                                    device=dev)
                        if (red_on and hz is not None) else None)
            for ci, (s, r, t) in enumerate(group):
                bnd = boundary_c[ci]
                for e in sp.entries:
                    g = local.get(e["name"])
                    g_el = None if res_on else (
                        torch.zeros(e["n"], dtype=torch.float32)
                        if g is None else g)
                    stt = states[(s, r, t)][e["name"]]
                    lr = R._lr_for_step(gmeta, e["group"])
                    b1, b2, eps, wd = e["b1"], e["b2"], e["eps"], e["wd"]
                    bc1c = 1.0 - b1 ** count
                    bc2c = 1.0 - b2 ** count
                    for a in range(0, e["n"], chunk_dev):
                        b = min(e["n"], a + chunk_dev)
                        if res_on:
                            gd = g_res[e["name"]][a:b]
                        else:
                            gd = g_el[a:b].to(dev) / coef_dense
                            if prof:
                                prof["h2d"][2] += b - a
                        if bnd:
                            q = (gd + stt["res"][a:b]) * coefs[ci]
                            stt["res"][a:b] = 0.0
                        else:
                            sel_c = stt["sel"][a:b]
                            q = gd * sel_c * coefs[ci]
                            stt["res"][a:b] += gd * (~sel_c)
                        if replay_dtype != torch.float32:
                            q = q.to(replay_dtype)
                        R._adam_step_inplace(
                            q, stt["ms"][a:b], stt["vs"][a:b], stt["ws"][a:b],
                            None, lr, wd, b1, b2, bc1c, bc2c, eps)
                        if hz is not None:
                            d = (stt["ws"][a:b]
                                 - snap_loc[hz][e["name"]][a:b].to(dev)) \
                                .double()
                            if red_on:
                                num2_dev[ci] += (d * d).sum()
                            else:
                                num2[ci] += float((d * d).sum())
                                if prof:
                                    prof["sync"] += 1
            if prof:
                prof["pt"][2] += time.perf_counter() - _t0p
            if hz is not None:
                if red_on:
                    num2 = num2_dev.cpu().tolist()
                    if prof:
                        prof["sync"] += 1
                num2 = comms.sum_f64(num2)
                if prof:
                    prof["ar"] += 1
                for ci, (s, r, t) in enumerate(group):
                    num_at[(s, r, t)][hz] = math.sqrt(num2[ci])

        del states
        if dev != "cpu":
            torch.cuda.empty_cache()
        if comms.global_rank == 0 and cfg["verbose"]:
            print(f"[fused t0={t0}] group {gi + 1}/{len(groups)} "
                  f"({len(group)} configs) done in "
                  f"{time.time() - t_pass0:.0f}s", flush=True)
            t_pass0 = time.time()
        if prof:
            # g slices are fp32 host-side (prefetcher upcasts), 4 B/elem
            mem_gb = (torch.cuda.max_memory_allocated() / 2 ** 30
                      if dev != "cpu" else 0.0)
            print(f"S0PROFILE rank={rank} group={gi} steps={W} "
                  f"syncs={prof['sync']} gloo_ar={prof['ar']} "
                  f"state_gb={prof['state_gb']:.2f} "
                  f"h2d_gb_p1={prof['h2d'][0] * 4 / 2 ** 30:.1f} "
                  f"h2d_gb_p2={prof['h2d'][1] * 4 / 2 ** 30:.1f} "
                  f"h2d_gb_p3={prof['h2d'][2] * 4 / 2 ** 30:.1f} "
                  f"pass_s_p1={prof['pt'][0]:.0f} "
                  f"pass_s_p2={prof['pt'][1]:.0f} "
                  f"pass_s_p3={prof['pt'][2]:.0f} mem_peak_gb={mem_gb:.1f}",
                  flush=True)
            s0_profs.append(dict(prof))

    # ---- global rank 0 assembles the full branch fragment ------------------
    # (node 0's local shard 0 in the multi-node topology — the launcher's
    # frag_path is only meaningful on the launcher's own machine; node 0
    # itself is launched by ssh-to-self like every other node, so its /tmp
    # IS the launcher's filesystem view)
    out = None
    if comms.global_rank == 0:
        per_ratio = {}
        for r in ratios:
            scores = {}
            for s in score_names:
                # a single T keeps the historical bare score key; with a T
                # set the same score name recurs per T and would collide
                for t in cycle_lens:
                    key = s if len(cycle_lens) == 1 else f"{s}@T{t}"
                    distortion = {}
                    for h in cfg["horizons"]:
                        hh = max(1, min(h, W))
                        den = den_at.get(hh)
                        num = num_at[(s, r, t)].get(hh)
                        distortion[str(h)] = (
                            num / den if (den is not None and num is not None
                                          and den > 0) else None)
                    scores[key] = {
                        "threshold": thresh0[(s, r, t)],
                        "distortion": distortion,
                        **rankmet[(s, r, t)],
                    }
            per_ratio[str(r)] = {"k": ks[r], "scores": scores}
        out = {
            "t0": t0,
            "state_absent": bool(meta.get("state_absent", False)),
            "window_len": W,
            "window_steps": window_steps,
            "cycle_lens": cycle_lens,
            "horizon_steps": {str(h): max(1, min(h, W))
                              for h in cfg["horizons"]},
            "meta": {
                "lr": R.scalar_list(meta.get("lr", [])),
                "grad_norm": float(meta.get("grad_norm", float("nan"))),
                "clip_grad": float(meta.get("clip_grad", 0.0)),
                "weight_decay": float(meta.get("weight_decay", 0.0)),
                "adam_beta1": float(meta.get("adam_beta1", 0.9)),
                "adam_beta2": float(meta.get("adam_beta2", 0.999)),
                "adam_eps": float(meta.get("adam_eps", 1e-8)),
            },
            "replay": {
                "dtype": cfg["replay_dtype"],
                "trace_dtype": cfg["trace_dtype"],
                "dense_cache_hit": dense_cache_hit,
                "g_residency": bool(cfg.get("g_residency")),
                "dev_reduce": bool(cfg.get("dev_reduce")),
                "per_step": recon_steps,
                "den_norm_at_horizon": {
                    str(h): den_at.get(h) for h in eff_horizons
                    if den_at.get(h) is not None
                },
                "label_total": label_total,
                "label_stats": stats_label,
            },
            "scores": stats,
            "per_ratio": per_ratio,
        }
        # conditional echoes: the default (budget off) report must stay
        # byte-identical to budget-off artifacts of this version (cycle_lens
        # above is echoed unconditionally, so pre-knob artifacts already
        # differ at byte level), so these keys only exist when the knob is on
        if cfg["state_budget_mb"] > 0:
            out["replay"]["state_budget_mb"] = float(cfg["state_budget_mb"])
            out["replay"]["group_cap"] = int(eff_cap)
        if _S0_PROFILE:
            out["s0_profile"] = s0_profs
        torch.save(out, frag_path)
    if comms.distributed:
        dist.destroy_process_group()
    return out


def _shard_stats(flat, comms):
    """_score_stats over a sharded vector: sums are shard-partial fp32 sums
    combined in fp64 (pure-reordering class); min/max/zero-count/numel
    exact. Empty shards contribute no min/max."""
    total = flat.numel()
    row = torch.tensor([
        float(total),
        float(flat.sum().item()) if total else 0.0,
        float((flat == 0).sum().item()) if total else 0.0,
        float(flat.min().item()) if total else float("inf"),
        float(flat.max().item()) if total else float("-inf"),
    ], dtype=torch.float64)
    parts = comms.all_gather_pad(row)
    gtotal = int(sum(int(p[0].item()) for p in parts))
    ssum = sum(p[1].item() for p in parts)
    nzero = int(sum(p[2].item() for p in parts))
    mins = [p[3].item() for p in parts if p[0].item() > 0]
    maxs = [p[4].item() for p in parts if p[0].item() > 0]
    return {
        "numel": gtotal,
        "sum": ssum,
        "mean": ssum / gtotal if gtotal else 0.0,
        "min": min(mins) if mins else 0.0,
        "max": max(maxs) if maxs else 0.0,
        "fraction_zero": nzero / gtotal if gtotal else 0.0,
    }


def _compute_dense_and_cache(cfg, R, t0, window_steps, eff_horizons,
                             replay_dtype):
    """Cache-miss fallback: run the classic dense pass (full params) in the
    parent process and save the standard dense cache — once per trace."""
    catalog, names, offsets, N, state_files, g_files = R.discover_trace(
        cfg["trace_dir"])
    meta, params, *_ = R.build_branch_state(
        state_files[t0], names, replay_dtype, catalog=catalog)
    if cfg["device"] == "cpu":
        # mirror the classic cpu path exactly (dense_replay quantizes the
        # horizon diffs to fp32 before squaring; dense_replay_dev does not)
        for prm in params:
            prm["w"] = prm["w0"].clone()
            prm["m"] = prm["m0"].clone()
            prm["v"] = prm["v0"].clone()
            prm["label"] = torch.zeros(prm["n"], dtype=replay_dtype)
        label_total, den_at, recon_steps, snap_d = R.dense_replay(
            params, g_files, window_steps, meta, t0, eff_horizons,
            replay_dtype, cfg["chunk_mb"], "cpu",
            upcast_bf16=cfg["upcast_bf16"])
    else:
        label_total, den_at, recon_steps, snap_d = R.dense_replay_dev(
            params, g_files, window_steps, meta, t0, eff_horizons,
            replay_dtype, cfg["chunk_mb"], "cuda:0",
            upcast_bf16=cfg["upcast_bf16"])
    label_flat = torch.cat([p["label"].float() for p in params])
    R.save_dense_cache(cfg["dense_cache"], R._dense_cache_ident(
        cfg["trace_dir"], t0, window_steps, eff_horizons, replay_dtype,
        cfg["trace_dtype"],
        "cpu" if cfg["device"] == "cpu" else "dev"), label_total, den_at,
        recon_steps, snap_d, label_flat)
    print(f"[fused t0={t0}] dense cache populated on miss "
          f"({cfg['dense_cache']})", flush=True)


# ----------------------------------------------------------------------------
# entry points (called from sga.replay main with a plain config dict)
# ----------------------------------------------------------------------------

def _common_gates(args):
    """Validation shared by the single-node and multi-node fused entries;
    returns the resolved score_names (registry order, unknowns rejected)."""
    import sga.replay as R
    if args.replay_mode != "cycle":
        raise ValueError("the fused engine implements --replay-mode cycle "
                         "only")
    if args.self_check is not None:
        raise ValueError("--replay-mode cycle cannot be combined with "
                         "--self-check")
    if not args.dense_cache:
        raise ValueError(
            "--replay-mode cycle requires --dense-cache (the dense arm is "
            "consumed from the cache; a first run populates it "
            "automatically)")
    if args.fused_max_configs < 1:
        raise ValueError("--fused-max-configs must be >= 1")
    for r in args.ratios:
        if r <= 0.0 or r > 100.0:
            raise ValueError(f"ratio {r} out of (0,100]")
    if args.device not in ("cpu", "cuda"):
        # a shard always maps to its node-local cuda index, so a bare index
        # on --device has no meaning; the visible device set is constrained
        # via CUDA_VISIBLE_DEVICES
        raise ValueError(
            "--device cuda:N is not supported: each shard always maps to "
            "its node-local cuda index. Use --device cuda and restrict the "
            "visible devices with CUDA_VISIBLE_DEVICES (e.g. "
            "CUDA_VISIBLE_DEVICES=4,5 for a 2-shard run on physical GPUs "
            "4 and 5)")
    # mirror the classic path: registry order + unknown-name rejection
    allowed = list(R.all_score_names())
    if args.score_names:
        unknown = [s for s in args.score_names if s not in allowed]
        if unknown:
            raise ValueError(
                f"--score-names {unknown} not a registered score "
                f"(allowed: {allowed})")
        score_names = [s for s in allowed if s in args.score_names]
    else:
        score_names = allowed
    world = args.fused_shards
    if world < 1:
        raise ValueError("--fused-shards must be >= 1")
    if args.fused_max_configs < 1:
        raise ValueError("--fused-max-configs must be >= 1")
    # getattr: programmatic callers (tests) build their own argparse
    # Namespace and must keep defaulting to the count-knob behavior
    state_budget_mb = float(getattr(args, "fused_state_budget_mb", 0.0)
                            or 0.0)
    if not math.isfinite(state_budget_mb) or state_budget_mb < 0:
        raise ValueError(
            "--fused-state-budget-mb must be a finite value >= 0, got "
            f"{state_budget_mb}")
    # the T set rides the cfg as a list; direct programmatic callers (tests)
    # predate the set spelling and pass the scalar cycle_len — resolve and
    # normalize here, mirroring the CLI gate
    cycle_lens = getattr(args, "cycle_lens", None)
    if cycle_lens is None:
        cycle_lens = [args.cycle_len]
    if (not cycle_lens
            or [t for t in cycle_lens
                if not isinstance(t, int) or t < 1]):
        raise ValueError(
            "cycle_lens must be a non-empty list of ints >= 1, "
            f"got {cycle_lens!r}")
    cycle_lens = sorted(set(cycle_lens))
    for r in args.ratios:
        if r <= 0.0 or r > 100.0:
            raise ValueError(f"ratio {r} out of (0,100]")
    if args.device != "cpu":
        # shard r maps to cuda:r, so a bare index on --device has no meaning;
        # the visible device set is constrained via CUDA_VISIBLE_DEVICES
        if args.device != "cuda":
            raise ValueError(
                "--device cuda:N is not supported: shard r always maps to "
                "cuda:r. Use --device cuda and restrict the visible devices "
                "with CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=4,5 for "
                "a 2-shard run on physical GPUs 4 and 5)")
        if world > torch.cuda.device_count():
            raise ValueError(
                f"--fused-shards {world} exceeds the visible CUDA devices "
                f"({torch.cuda.device_count()})")
    return score_names


def _fused_cfg(args, ctx, t0, score_names):
    """The plain-dict worker config (json-serializable: the multi-node
    launcher ships it to every node verbatim)."""
    # the same T-set resolution as the gate (validated there): scalar
    # cycle_len from legacy programmatic callers fans out to a 1-element set
    cycle_lens = getattr(args, "cycle_lens", None)
    if cycle_lens is None:
        cycle_lens = [args.cycle_len]
    return {
        "trace_dir": ctx["trace_dir"],
        "t0": t0,
        "window": args.window,
        "horizons": args.horizons,
        "ratios": args.ratios,
        "score_names": score_names,
        "cycle_lens": sorted(set(cycle_lens)),
        "cycle_mode": True,
        "replay_dtype": args.replay_dtype,
        "trace_dtype": args.trace_dtype,
        "upcast_bf16": args.trace_dtype == "bf16",
        "norm_fp32": args.norm_dtype == "fp32",
        "chunk_mb": args.chunk_mb,
        "device": ("cpu" if args.device == "cpu" else "cuda"),
        "dense_cache": args.dense_cache,
        "max_configs": args.fused_max_configs,
        "state_budget_mb": float(getattr(args, "fused_state_budget_mb", 0.0)
                                 or 0.0),
        # getattr: programmatic callers (tests) build their own argparse
        # Namespace and must keep defaulting to the legacy upload path
        "g_residency": getattr(args, "fused_g_residency", False),
        "dev_reduce": getattr(args, "fused_dev_reduce", False),
        "seed": args.seed,
        "verbose": args.verbose,
    }


def _populate_dense_cache(args, ctx, cfg, t0):
    """Populate the dense cache parent-side, before any shard exists: on a
    cold cache the dense pass can run for hours and must not happen inside
    the process group (the waiting shards would hit the gloo timeout).
    dense_cache_hit reports the parent-side LOAD outcome (False on the
    populating run), matching the classic report field semantics. Returns
    (window_steps, eff_horizons, ident)."""
    import sga.replay as R
    replay_dtype = (torch.float64 if args.replay_dtype == "fp64"
                    else torch.float32)
    window_steps, _W, eff_horizons = _window_and_horizons(
        ctx["g_files"], t0, args.window, args.horizons)
    ident = R._dense_cache_ident(ctx["trace_dir"], t0, window_steps,
                                 eff_horizons, replay_dtype,
                                 args.trace_dtype,
                                 "cpu" if args.device == "cpu" else "dev")
    hit = R.load_dense_cache(args.dense_cache, ident) is not None
    if not hit:
        _compute_dense_and_cache(cfg, R, t0, window_steps, eff_horizons,
                                 replay_dtype)
        if args.device != "cpu":
            # the populate pass ran on cuda:0 in THIS (parent) process and
            # its peak stays cached in the allocator, invisible to per-process
            # accounting but visible to the workers' sort guard, which reads
            # device-wide free memory — hand the blocks back before any
            # shard spawns (no device tensor is live at this point)
            torch.cuda.empty_cache()
    cfg["dense_cache_hit"] = hit
    return window_steps, eff_horizons, ident


def run_fused_branch(args, ctx, t0):
    """One branch point through the fused engine (single node); returns the
    complete branch dict (same JSON shape as process_branch). Multi-node
    runs go through sga.replay_launch.run_fused_branch_multinode."""
    import tempfile
    score_names = _common_gates(args)
    world = args.fused_shards
    if world < 1:
        raise ValueError("--fused-shards must be >= 1")
    if args.device == "cuda" and world > torch.cuda.device_count():
        raise ValueError(
            f"--fused-shards {world} exceeds the visible CUDA devices "
            f"({torch.cuda.device_count()})")
    cfg = _fused_cfg(args, ctx, t0, score_names)
    _populate_dense_cache(args, ctx, cfg, t0)
    fd, frag_path = tempfile.mkstemp(suffix=".pt",
                                     prefix="fused_branch_")
    os.close(fd)
    try:
        if world == 1:
            _worker(0, 1, cfg, 0, frag_path)
        else:
            port = _free_port()
            import torch.multiprocessing as mp
            mp.spawn(_worker, args=(world, cfg, port, frag_path),
                     nprocs=world, join=True)
        return torch.load(frag_path, map_location="cpu", weights_only=False)
    finally:
        if os.path.exists(frag_path):
            os.remove(frag_path)
