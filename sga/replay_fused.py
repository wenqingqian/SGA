# Copyright (c) 2026, SGA framework contributors.
"""Fused single-pass streaming replay (config-grouped, param-sharded).

A naive replay reads the whole window once per arm. This engine advances
all arms of one config group together, so the window is read once per
group — sharded across processes:

- ``--fused-shards`` shard processes (one per GPU; gloo over localhost for
  the tiny control-plane exchanges, plus an optional single-node NCCL data
  plane for the front-phase payload gathers), each owning a contiguous
  slice of the global flat index space (catalog order). Every shard mmaps
  the same g files and reads only its slice; the host page cache dedups
  the NFS pulls (each file crosses the machine once per pass). A prefetch
  thread faults in the next step's slices while the current step computes.
- Per step the shards exchange only per-config clip-norm sums (fp64
  scalars); at cycle boundaries they additionally exchange the 32769-bin
  integer score histograms and per-shard (above, tie) counts — the global
  top-k stays exact, with the tie layer filled in global flat-index order
  (= shard order, then in-shard order), bit-identical to
  ``select_topk_flat`` on the concatenated vector. With
  ``--selection-semantics balanced-histogram`` (the default) the two select
  sites swap
  the exact tie-filled topk for the online engine's whole-crossing-bin
  histogram select — the same int64 histogram exchange, without the tie
  counts, reconstructed catalog-wide across shards
  (``shard_select_balanced``; single node only).
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

Arithmetic contract: per-config math runs per-param per-chunk in the
same order as ``tests/test_replay_cycle.reference_cycle``, so an explicit
``--selection-semantics topk`` single-shard run reproduces the reference
bitwise. Balanced runs instead follow the online whole-crossing-bin
semantics at both selection sites. At world > 1 the
only deviations are cross-shard fp64 scalar-sum reorderings (covered by
the frozen pure-reordering band), plus the optional ``--norm-dtype fp32``
quantization (gated by its own frozen band). The
score/label SUM statistics are shard-partial sums (same reordering class);
min/max/zero-fractions stay exact. Ranking metrics are exact: the selected
coordinates' (score, label, label-sel) values are gathered to shard 0 in
global index order and the same arithmetic as ``ranking_metrics`` runs on
the reconstructed vectors.

Per-config axes (``--optimizer`` / ``--fold`` / ``--dual-lr``): every
(score, ratio, T) tuple fans out over the three axes and all arms advance
in the same lockstep window pass. ``optimizer`` selects the step recursion
(adamw | adams — AdamS rebuilds v from the pre-commit m with bias
correction dropped, see sga.replay.ADAMS_V_BASIS; the dense reference —
label/den/snapshots — is per-optimizer, one dense cache file each).
``fold=on`` divides the accumulated residual by T at the boundary commit
(the online residual_mean fold), in place ahead of the boundary clip norm
so the norm, the boundary re-selection score and the commit all read the
folded value. ``--dual-lr`` resolves T-relative tokens (``T``, ``T/2``)
to the full-sync inactive-lr scale: at boundaries the never-communicated
tail applies the whole step (weight-decay component included) at
lr * scale against the OUTGOING window's mask. The default singletons
(adamw / off / 1) reduce the config space to the historical
(score, ratio, T) behavior; explicit topk also retains the legacy report
shape, while balanced adds its semantic echo.
"""
import datetime
import io
import math
import os
import pickle
import socket
import struct
import threading
import time
import queue
import zipfile
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# opt-in stage-0 (S0) profile: counts host syncs / gloo
# ops / H2D bytes per pass and per-pass wall time; zero effect when unset
_S0_PROFILE = os.environ.get("REPLAY_S0_PROFILE", "") == "1"
# replay selection is balanced-histogram unless explicit legacy topk is used
BALANCED_HISTOGRAM = "balanced-histogram"


# ----------------------------------------------------------------------------
# small comms wrapper: exact reductions, identity when world == 1
# ----------------------------------------------------------------------------

class Comms:
    """gloo wrapper for the tiny control-plane exchanges (exact reductions,
    length/pad gathers, unanimous votes); the same object optionally owns a
    single-node NCCL data plane for the front-phase device payload gathers
    (``all_gather_pad_dev``) — gloo remains THE control plane either way.

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
        self._nccl_group = None
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

    def vote_unanimous(self, local_ok):
        """One all-rank AND vote (identity when world == 1). Device-path
        decisions must be unanimous: a partial fallback would split the
        ranks across two different collectives (one gloo, one NCCL) and
        deadlock the boundary."""
        if not self.distributed:
            return bool(local_ok)
        vote = torch.tensor([1 if local_ok else 0], dtype=torch.int64)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        return bool(vote.item())

    def vote_gather_backend(self):
        """Unanimous transport vote for device-resident gathers. The NCCL
        fast path is single-node only by design (the multi-node rendezvous
        keeps the byte-identical gloo path); world == 1 needs no transport
        at all."""
        if not self.distributed:
            return "gloo"
        local = (os.environ.get("SGA_MASTER_ADDR") is None
                 and dist.is_nccl_available())
        return "nccl" if self.vote_unanimous(local) else "gloo"

    def _dev_gather_group(self):
        """Lazily create the NCCL group after a unanimous availability
        vote. A failure here is post-vote (the environment is broken):
        abort loudly rather than deadlock on a split collective."""
        if self._nccl_group is None:
            try:
                self._nccl_group = dist.new_group(backend="nccl")
            except Exception as exc:
                raise RuntimeError(
                    "NCCL group init failed after a unanimous availability "
                    "vote — aborting loudly instead of deadlocking") from exc
        return self._nccl_group

    def all_gather_pad_dev(self, t, backend):
        """Device-payload variant of all_gather_pad: same contract (per-rank
        CPU tensors in rank order, padding never enters the results) but the
        payload is produced on a device and moved over the NCCL fast path
        when ``backend`` says so. Transport is pure data movement — values
        (and every downstream CPU expression) are bit-identical across
        backends. bool payloads move as uint8 (NCCL has no bool)."""
        assert t.is_contiguous(), \
            "all_gather_pad_dev needs a contiguous 1-D payload"
        if not self.distributed:
            return [t.cpu()]
        if backend != "nccl":
            return self.all_gather_pad(t.cpu())
        n = torch.tensor([t.numel()], dtype=torch.int64)
        ns = [torch.zeros(1, dtype=torch.int64) for _ in range(self.world)]
        dist.all_gather(ns, n)
        m = max(int(x.item()) for x in ns)
        cast = t.dtype == torch.bool
        tt = t.view(torch.uint8) if cast else t
        buf = torch.zeros(m, dtype=tt.dtype, device=tt.device)
        buf[:tt.numel()] = tt
        out = [torch.zeros_like(buf) for _ in range(self.world)]
        dist.all_gather(out, buf, group=self._dev_gather_group())
        parts = [out[r][:int(ns[r].item())] for r in range(self.world)]
        if cast:
            parts = [p.view(torch.bool) for p in parts]
        return [p.cpu() for p in parts]


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

def shard_select(score, k, comms, prof=None):
    """Exact global top-k of a non-negative fp32 score vector sharded across
    ranks (each rank holds its contiguous slice; ``score`` may live on cpu or
    the shard's cuda device). Returns (local bool mask on the same device,
    global threshold) with the tie layer filled in GLOBAL flat-index order —
    bit-identical to select_topk_flat on the concatenated vector.

    ``prof`` (env-gated diagnostic, None on the default path): when given,
    device-sync-bracketed sub-bucket timers accumulate into the keys
    p2s_hist (finite guard + top16 + bincount + D2H), p2s_ar (histogram
    all_reduce), p2s_cand (cumsum/searchsorted + candidate extraction +
    gather + kthvalue) and p2s_fill (gt/eq + counts + tie-fill).

    When the replay-kernel arm engages (``arm_select``: CUDA fp32
    contiguous + killswitch auto + pinned triton), the histogram sweep
    (with the finite guard folded in), the candidate extraction and the
    compare+count sweep run as Triton kernels — integer/compare-identical
    to the torch chain, the candidate vector in the same global index
    order; the collectives and kthvalue always stay torch."""
    _t = time.perf_counter() if prof is not None else 0.0

    def _prof_mark(key):
        nonlocal _t
        if score.is_cuda:
            torch.cuda.synchronize()
        _now = time.perf_counter()
        prof[key] += _now - _t
        _t = _now

    from sga import replay_kernels as K
    k_sel = K.arm_select(score)
    if k_sel:
        hist, _nf = K.hist_top16(score)
        if _nf:
            raise ValueError(
                "shard_select requires finite scores (NaN/Inf would land "
                "in the top bins and silently pollute the topk)")
    else:
        if not bool(torch.isfinite(score).all().item()):
            # NaN/Inf sit in the histogram's top bit-pattern bins: they
            # would be selected FIRST, silently displacing real
            # coordinates from the topk.
            raise ValueError(
                "shard_select requires finite scores (NaN/Inf would land "
                "in the top bins and silently pollute the topk)")
        top16 = ((score.view(torch.int32) & 0x7FFFFFFF) >> 16)
        hist = torch.bincount(top16, minlength=32769).cpu()
    if prof is not None:
        _prof_mark("p2s_hist")
    if comms.distributed:
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
    if prof is not None:
        _prof_mark("p2s_ar")
    cum = torch.cumsum(hist, 0)
    total = int(cum[-1].item())
    if k > total:
        raise ValueError(f"k={k} > N={total}")
    r = total - k + 1
    b = int(torch.searchsorted(cum, r).item())
    n_above = total - int(cum[b].item())

    # exact k-th value: kthvalue over the union of the shards' candidates in
    # bin b (the same multiset the classic hist_threshold refines over)
    cand = (K.cand_extract(score, b) if k_sel
            else score[top16 == b].cpu().clone())
    parts = comms.all_gather_pad(cand)
    whole = torch.cat(parts)
    kk = whole.numel() - (k - n_above) + 1
    if whole.numel() == 0 or not (1 <= kk <= whole.numel()):
        raise RuntimeError(
            f"fused hist select math out of range: cand={whole.numel()} "
            f"kk={kk} r={r}")
    thresh = float(torch.kthvalue(whole, kk).values.item())
    if prof is not None:
        _prof_mark("p2s_cand")

    # tie-fill: first `need` tied coords in global index order = shard
    # order, then in-shard order. counting runs chunked and the fill marks
    # the tie layer's nonzero indices: bool .sum()/.count_nonzero()/.cumsum()
    # all materialize a full-size int64 copy (9+ GB per call at 1.2B elements
    # on one shard), which OOMs the world=1 oracle mode
    if k_sel:
        gt, eq, counts = K.gt_eq_counts(score, thresh)
    else:
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
    if prof is not None:
        _prof_mark("p2s_fill")
    return gt, thresh


def shard_select_balanced(score, k, comms, entries):
    """Bit-faithful replay of the online engine's balanced-histogram
    selection (``sga.plugins.selection._histogram.select_masks_histogram(
    mode="balanced", global_allreduce=False)``) for the replay select
    sites, single- or multi-shard on ONE node.

    The online engine selects on per-param score views with the histogram
    built on a stride-8 subsample — per view, per ``_HIST_CHUNK`` block
    (the chunk length is a stride multiple, so both reduce to the view's
    param-relative ``[::8]``); the crossing bin is searched with ``r =
    max(1, round(n_sampled * (n_total - k + 1) / n_total))`` in world-1
    local units (the engine's replicated full-sync scores make every
    rank's local histogram the full one, so the un-merged select derives
    the same bin the merged form would); the mask keeps the WHOLE
    crossing bin — approximate by construction (the ±0.5pp two-sided
    executable criterion), which is exactly the selection-set difference
    vs the exact topk replay.

    Sharded replay reconstructs that full-catalog select exactly.
    ``entries`` are the shard's ``sp.entries`` records (``{"a": shard-local
    flat offset, "p_off": param-relative slice start, "n": slice numel}``;
    whole-view callers may omit ``p_off`` — it is 0 there) and every
    per-param subsample keeps the ONLINE plugin's param-relative grid:
    the sampled local indices are ``(-p_off) % stride`` plus stride
    steps, so the shard partial histograms SUM to exactly the world-1
    histogram (one int64 all_reduce) and the shard-local n_total /
    n_sampled sum to the catalog-wide values (one int64 all_reduce).
    ``r`` stays in world-1 units — NOT scaled by the replay's shard
    world, because the partials are disjoint slices, not the online
    engine's replicated copies. The crossing bin b is therefore identical
    on every rank: the returned mask is that rank's exact slice of the
    world-1 mask and the threshold is bitwise the same everywhere.

    Only the single-node engine is supported (the mp.spawn shards; a
    multi-node rendezvous sets SGA_MASTER_ADDR and is rejected here AND
    by the CLI gate).

    Unlike the online plugin (which has no guard) this entry raises on
    non-finite scores — the replay's NaN/Inf canary, mirroring
    shard_select; the check only fires where parity is undefined anyway,
    so finite-input results stay bit-identical (a deliberate asymmetry:
    replay is the measurement instrument).
    """
    if os.environ.get("SGA_MASTER_ADDR") is not None:
        raise ValueError(
            "--selection-semantics balanced-histogram supports only the "
            "single-node fused engine (the multi-node launcher's shards "
            "do not implement the sharded balanced select)")
    from sga.plugins.selection._histogram import (
        _HIST_MODES, _abs_bit_top, _accumulate_histogram,
        _balanced_crossing_bin, _bin_edge_float, _sampled_count)
    if not bool(torch.isfinite(score).all().item()):
        # NaN/Inf land in the histogram's top bit-pattern bins and would
        # be selected FIRST — mirror shard_select's canary (the online
        # plugin has none; deliberate asymmetry, see the docstring)
        raise ValueError(
            "shard_select_balanced requires finite scores (NaN/Inf land "
            "in the top bit-pattern bins and would be selected; parity "
            "is undefined there)")

    shift, nbins, stride = _HIST_MODES["balanced"]
    hist = torch.zeros(nbins, dtype=torch.int64)
    # [n_total, n_sampled] shard partials: the shards partition every
    # param view, so their sum is the full-catalog pair
    counts = torch.zeros(2, dtype=torch.int64)
    for e in entries:
        v = score[e["a"]:e["a"] + e["n"]]
        n = v.numel()
        counts[0] += n
        phase = (-int(e.get("p_off", 0))) % stride
        counts[1] += _sampled_count(n, stride, phase)
        _accumulate_histogram(hist, v, shift, stride, phase)
    if comms.distributed:
        # gloo all_reduce of the int64 histogram + counts: the shard
        # partials are DISJOINT slices of the catalog, so the sum is the
        # online engine's full histogram and the crossing-bin search
        # below reads world-1 quantities (no world factor anywhere)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    n_total = int(counts[0].item())
    n_sampled = int(counts[1].item())
    b = _balanced_crossing_bin(hist, n_total, n_sampled, k, replicas=1)
    mask = _abs_bit_top(score, shift) >= b
    thresh = _bin_edge_float(b, shift)
    return mask, thresh


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
    # physically fitting. On the main thread no sort tensor is live between
    # calls (the .cpu() fetch syncs); from the rank worker's vantage sorts
    # run concurrently with the main thread's front-select transients, so
    # the free-VRAM sample can differ between the rank_bg arms — one arm
    # may take this loud CPU fallback where the other sorted on device.
    # Both outcomes stay bit-identical (the stable-sort guarantee), the
    # worker's rank_s may then include CPU-fallback sorts, and the
    # worker's empty_cache() can hand back blocks the main thread would
    # have reused (allocation churn, perf-only).
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(dev)
    need = numel * 32
    ok = need < free * 0.85
    if not ok:
        print(f"[fused] ranking permutation on {dev} skipped: "
              f"need ~{need / 2 ** 30:.1f} GiB, free {free / 2 ** 30:.1f} "
              f"GiB — CPU fallback (bit-identical)", flush=True)
    return ok


def _device_ok_for_select(numel, k_max, world, n_scores, dev):
    """Conservative VRAM guard for the device front-select block (not the
    sort guard's 32 B/elem): the resident fp32 vectors (4 * (n_scores + 1)
    B/elem: one vector per selected score + the label), the per-call select
    peak (14 B/elem: top16 int32 4 + gt/eq bool 2 + worst-case all-tie
    nonzero int64 8), and the padded gather fan-out (~k/world per-rank
    payload times world+1 buffers) — the expected case with margin, not an
    exhaustive booking (unbooked transients exist: candidate/isfinite/
    payload copies and a selection skewed onto one rank). Runtime free-mem
    based with the same empty_cache-first caveat as _device_ok_for_sort
    (mem_get_info counts this process's own cached-but-free blocks). At
    refusal the run takes the loud, bit-identical CPU path — guaranteed at
    refusal only: a post-vote VRAM grab (TOCTOU) aborts the run by design
    (collectives cannot be unwound mid-block)."""
    if dev is None:
        return False
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(dev)
    need = numel * (4 * (n_scores + 1) + 14) \
        + (k_max // world + 1) * 4 * (world + 1)
    ok = need < free * 0.85
    if not ok:
        print(f"[fused] front select on {dev} skipped: "
              f"need ~{need / 2 ** 30:.1f} GiB, free {free / 2 ** 30:.1f} "
              f"GiB — CPU fallback (bit-identical); a post-vote VRAM grab "
              f"(TOCTOU) aborts instead — collectives cannot be unwound "
              f"mid-block", flush=True)
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


class _FrontRankWorker:
    """rank0's front-phase ranking chain on one bounded background thread.

    The main pipeline enqueues each (s, r)'s gathered parts and proceeds to
    the next shard_select collective immediately; the thread applies
    _idcg_of_parts / _fused_ranking to the same inputs in FIFO (= loop)
    order as the single writer of idcg/rankmet, so results are
    bit-identical to the synchronous path. Quarantine contract (pinned at
    the enqueue site too): the queue carries plain CPU tensors/floats
    only — the worker never receives a comms object and never issues a
    dist call, so every collective stays single-threaded on the main
    thread. With a device sort arm the worker sets its own
    torch.cuda.set_device (the device is thread-local); the helpers'
    reductions keep their original CPU expression path either way. The
    worker deliberately does NOT call torch.set_num_threads — the fp32 sum
    partitioning depends on the process-wide intra-op configuration, and
    bit-identity additionally assumes the OpenMP runtime grants the worker
    thread's root-level parallel region a full team (the libgomp default
    for a foreign host thread, with OMP_DYNAMIC unset/false).

    Failure discipline: a worker exception is captured, not swallowed —
    a health-checked put refuses to block forever behind a dead worker
    (bounded queue + dead consumer would otherwise deadlock the main
    thread on backpressure), and finish() re-raises it before the
    streaming phase."""

    _SENTINEL = object()

    def __init__(self, sort_dev, ks, stats, label_total,
                 maxsize=4):
        self.sort_dev = sort_dev
        # read-only inputs shared with the main thread (never mutated after
        # the pre-loop computation — the FIFO hand-off needs no copies);
        # stats is keyed (score, optimizer), label_total by optimizer
        self.ks, self.stats, self.label_total = ks, stats, label_total
        self.idcg, self.rankmet = {}, {}
        self.rank_s, self.qwait_s = 0.0, 0.0
        self.error = None
        # identity (op/score/ratio) of the item in flight, for the failure
        # message; read by the main thread only once error is visible
        self._cur = "startup"
        self._q = queue.Queue(maxsize=maxsize)
        self._finished = False
        self._th = threading.Thread(target=self._run, daemon=True,
                                    name="front-rank-worker")

    def start(self):
        self._th.start()

    def _run(self):
        try:
            # context setup inside the try: a failure here must surface as
            # self.error (finish() re-raises it), never as a dead worker
            # with a silently empty rankmet
            if self.sort_dev is not None:
                torch.cuda.set_device(int(self.sort_dev.split(":")[1]))
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    return
                _t = time.perf_counter()
                if item[0] == "idcg":
                    _, opt, r, parts = item
                    self._cur = f"idcg opt={opt} r={r}"
                    self.idcg[(opt, r)] = _idcg_of_parts(
                        parts, dev=self.sort_dev)
                else:
                    _, s, opt, r, sc_parts, lb_parts, ls_parts = item
                    self._cur = f"rank s={s} opt={opt} r={r}"
                    met = _fused_ranking(
                        torch.cat(sc_parts), torch.cat(lb_parts), ls_parts,
                        self.ks[r], self.stats[(s, opt)]["sum"],
                        self.label_total[opt], self.idcg[(opt, r)],
                        dev=self.sort_dev)
                    # the ranking metrics depend on (score, ratio,
                    # optimizer) only — every T/fold/dual-lr arm of one
                    # (s, r, opt) shares this entry
                    self.rankmet[(s, r, opt)] = met
                self.rank_s += time.perf_counter() - _t
        except BaseException as e:  # captured for the main thread
            self.error = e

    def _raise_worker_failed(self):
        raise RuntimeError(
            f"front rank worker failed ({self._cur})") from self.error

    def submit(self, item):
        # the health checks detect a DEAD worker only, never a hung one —
        # a live-but-hung worker backpressures this put exactly as the
        # synchronous path would stall the pipeline
        if self.error is not None:
            self._raise_worker_failed()
        while True:
            try:
                _t = time.perf_counter()
                self._q.put(item, timeout=1.0)
                self.qwait_s += time.perf_counter() - _t
                return
            except queue.Full:
                self.qwait_s += time.perf_counter() - _t
                if self.error is not None:
                    self._raise_worker_failed()
                if not self._th.is_alive():
                    raise RuntimeError(
                        "front rank worker died without an exception "
                        "record")

    def finish(self):
        """Drain + join + adopt the results; re-raises a worker failure.
        Runs after the front loop, before streaming setup / report
        assembly (and before destroy_process_group)."""
        # before the submit: a failed sentinel put still marks the worker
        # closed (its join would return instantly either way)
        self._finished = True
        self.submit(self._SENTINEL)
        self._th.join()
        if self.error is not None:
            self._raise_worker_failed()
        return self.rankmet

    def close_best_effort(self):
        """Exception-path shutdown: drop a sentinel if the queue has room
        and wait out at most join(timeout=5.0) of the drain — bounded, but
        it CAN block, so only the failure path calls it (the success
        path's single sentinel+join is finish(), after the front-only
        release block). A full queue skips the sentinel; the daemon flag
        is the final backstop."""
        if self._finished:
            return
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass
        self._th.join(timeout=5.0)


# ----------------------------------------------------------------------------
# g-slice prefetch (fault the NFS pages of upcoming steps off the hot path)
# ----------------------------------------------------------------------------

# pread-arm trace index: read a torch.save zip's data.pkl + record offsets
# WITHOUT faulting any storage page, so the shard's slices can be pulled
# with plain pread (page faults and readahead, not the wire, cap the mmap
# path)
_TRACE_DTYPES = {
    "DoubleStorage": torch.float64, "FloatStorage": torch.float32,
    "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64, "IntStorage": torch.int32,
    "ShortStorage": torch.int16, "CharStorage": torch.int8,
    "ByteStorage": torch.uint8, "BoolStorage": torch.bool,
}

# static element sizes for the trace dtype vocabulary — a KeyError here is
# the same fail-loudly contract as _TRACE_DTYPES (no per-entry 0-d tensor
# allocation just to read a constant)
_ELSIZE = {
    torch.float64: 8, torch.float32: 4, torch.float16: 2,
    torch.bfloat16: 2, torch.int64: 8, torch.int32: 4,
    torch.int16: 2, torch.int8: 1, torch.uint8: 1, torch.bool: 1,
}


class _StubTensor:
    """A payload tensor's index card: which storage record, which dtype,
    which element offset/shape/stride — no payload bytes."""
    __slots__ = ("key", "dtype", "offset", "size", "stride")

    def __init__(self, key, dtype, offset, size, stride):
        self.key, self.dtype, self.offset = key, dtype, offset
        self.size, self.stride = size, stride

    def contiguous(self):
        exp, acc = [], 1
        for d in reversed(self.size):
            exp.append(acc)
            acc *= d
        return self.stride == tuple(reversed(exp))

    def numel(self):
        n = 1
        for d in self.size:
            n *= d
        return n


def _stub_rebuild_tensor_v2(storage, offset, size, stride, *rest):
    key, dtype, _numel = storage
    return _StubTensor(key, dtype, offset, tuple(size), tuple(stride))


class _TraceIndexUnpickler(pickle.Unpickler):
    """data.pkl → meta + payload stubs. The trace writer's vocabulary is
    fixed (containers + _rebuild_tensor_v2 + typed-storage dtype tokens);
    anything else must fail loudly, never guess."""

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _stub_rebuild_tensor_v2
        if module == "torch" and name in _TRACE_DTYPES:
            return _TRACE_DTYPES[name]
        if module == "collections" and name == "OrderedDict":
            return dict
        raise pickle.UnpicklingError(
            f"unexpected {module}.{name} in trace index")

    def persistent_load(self, pid):
        # arity validated before the unpack: a malformed id must raise the
        # UnpicklingError contract, not a bare ValueError from the tuple
        if not (isinstance(pid, tuple) and len(pid) == 5):
            raise pickle.UnpicklingError(
                f"unexpected persistent id {pid!r}")
        kind, dtype, key, loc, numel = pid
        if kind != "storage" or loc != "cpu":
            raise pickle.UnpicklingError(
                f"unexpected persistent id {pid!r}")
        return (key, dtype, numel)


def _pt_index(path):
    """(meta, stubs, data_offsets) for a torch.save trace file: stubs map
    payload name → _StubTensor; data_offsets map storage key → absolute
    file offset of the storage bytes. The data offset must come from the
    LOCAL header (torch pads each storage record's start to
    .storage_alignment via the local extra field, so central-directory
    arithmetic alone lands short)."""
    with zipfile.ZipFile(path) as zf:
        recs = {}
        pkl_bytes = None
        for inf in zf.infolist():
            parts = inf.filename.split("/")
            if parts[-1] == "data.pkl":
                pkl_bytes = zf.read(inf)
            elif len(parts) == 3 and parts[-2] == "data":
                if inf.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(
                        f"{path}: compressed storage record {inf.filename}")
                recs[parts[-1]] = inf
    if pkl_bytes is None:
        raise ValueError(f"{path}: no data.pkl record")
    fd = os.open(path, os.O_RDONLY)
    try:
        offs = {}
        for key, inf in recs.items():
            hdr = os.pread(fd, 30, inf.header_offset)
            if len(hdr) != 30:
                # named parse error, not an opaque struct.error (same
                # short-read discipline as the payload preadv loop)
                raise ValueError(
                    f"{path}: short local header read "
                    f"@{inf.header_offset} ({len(hdr)}/30)")
            (sig, _ver, _flag, _meth, _mt, _dt, _crc, _csz, _usz,
             nlen, elen) = struct.unpack("<IHHHHHIIIHH", hdr)
            if sig != 0x04034B50:
                raise ValueError(
                    f"{path}: bad local header sig @{inf.header_offset}")
            offs[key] = (inf.header_offset + 30 + nlen + elen,
                         inf.file_size)
    finally:
        os.close(fd)
    d = _TraceIndexUnpickler(io.BytesIO(pkl_bytes)).load()
    return d["meta"], dict(d["payload"]), offs


class GPrefetcher:
    """Background thread materializing the shard's per-step g slices as
    contiguous fp32 CPU tensors (page-cache faults + upcast overlap the
    compute of the current step); the compute side only pays the H2D.

    The pread arm's in-flight host-memory bound is ``depth`` raw slice
    bytearrays in _raw_q PLUS ``depth`` converted (fp32 / pinned) tensors
    in q — roughly 2x the mmap arm's peak (for bf16 traces the raw side is
    half-width, so the increment is modest). Matters when sizing host
    memory for large shards with ``pin=True``."""

    def __init__(self, g_files, window_steps, sp, upcast_bf16, depth=2,
                 pin=False, timed=False, read_mode="mmap", index_cache=None):
        self.g_files, self.window_steps, self.sp = g_files, window_steps, sp
        self.upcast_bf16 = upcast_bf16
        # pinned host buffers make the consumer's H2D non_blocking-able;
        # requires CUDA (callers gate this on the device)
        self.pin = pin
        # timed brackets the consumer's q.get() (S0PROFILE diagnostics only —
        # the value path is untouched)
        self.timed = timed
        self.qwait_s = 0.0
        # pread arm: one reader thread pulls the shard's raw slice bytes
        # (single-threaded by design — the NFS aggregate is wire-capped and
        # read threads do not scale it) while a second thread does the
        # upcast/pin CPU work, so conversion never serializes into the read
        # syscalls; the consumer-facing queue/semantics are unchanged
        self.read_mode = read_mode
        # zip-index cache: the worker hands one dict to every group's
        # prefetcher so each file's index is parsed once per worker, not
        # once per group; per-instance dict when the caller passes none
        self.index_cache = {} if index_cache is None else index_cache
        self.q = queue.Queue(maxsize=depth)
        self.exc = None
        if read_mode == "pread":
            self._raw_q = queue.Queue(maxsize=depth)
            threading.Thread(target=self._run_read, daemon=True).start()
            threading.Thread(target=self._run_upcast, daemon=True).start()
        else:
            threading.Thread(target=self._run, daemon=True).start()

    def _run_read(self):
        try:
            for gstep in self.window_steps:
                path = self.g_files[gstep]
                # the index is pure metadata; one reader thread per
                # prefetcher walks the groups sequentially, so the shared
                # dict has no concurrent writers and needs no lock
                if path not in self.index_cache:
                    self.index_cache[path] = _pt_index(path)
                meta, stubs, offs = self.index_cache[path]
                bf16_keys = [k for k, v in stubs.items()
                             if isinstance(v, _StubTensor)
                             and v.dtype == torch.bfloat16]
                if bf16_keys and not self.upcast_bf16:
                    raise ValueError(
                        f"{path}: bf16 payload(s) {bf16_keys} "
                        "but --trace-dtype is fp32; rerun with --trace-dtype "
                        "bf16 (payloads are upcast to fp32 at load)")
                raw = {}
                fd = os.open(path, os.O_RDONLY)
                try:
                    for e in self.sp.entries:
                        st = stubs.get(e["name"])
                        if st is None:
                            continue
                        if not st.contiguous():
                            raise ValueError(
                                f"{path}: non-contiguous "
                                f"payload {e['name']} stride={st.stride}")
                        elsize = _ELSIZE[st.dtype]
                        data_off, rec_bytes = offs[st.key]
                        if rec_bytes < (st.offset + st.numel()) * elsize:
                            raise ValueError(
                                f"{path}: storage record "
                                f"{st.key} too short for {e['name']}")
                        if (st.offset + e["p_off"] + e["n"]) * elsize \
                                > rec_bytes:
                            # a stale catalog must fail loudly here: past
                            # this check preadv would read the adjacent
                            # record's bytes where the mmap arm silently
                            # clamps — the arms would diverge on corrupt
                            # input instead of both raising
                            raise ValueError(
                                f"{path}: slice [{e['p_off']}, "
                                f"{e['p_off'] + e['n']}) of {e['name']} "
                                f"outruns storage record {st.key}")
                        lo = data_off + (st.offset + e["p_off"]) * elsize
                        n = e["n"] * elsize
                        # preadv: the kernel copies straight into the buffer
                        # with the GIL released — a python-side assign would
                        # hold the GIL for the whole memcpy and convoy the
                        # consumer's CUDA launches
                        buf = bytearray(n)
                        mv = memoryview(buf)
                        got = 0
                        while got < n:
                            r = os.preadv(fd, [mv[got:]], lo + got)
                            if r == 0:
                                raise OSError(
                                    f"short pread {path} "
                                    f"@{lo + got} ({got}/{n})")
                            got += r
                        raw[e["name"]] = (st.dtype, buf)
                finally:
                    os.close(fd)
                self._raw_q.put((gstep, meta, raw))
            # the upcast thread exits only on this sentinel — without the
            # success-path put it parks on _raw_q.get() forever (one
            # leaked daemon thread per config group)
            self._raw_q.put(None)
        except BaseException as e:
            self.exc = e
            self._raw_q.put(None)

    def _run_upcast(self):
        try:
            while True:
                item = self._raw_q.get()
                if item is None:
                    self.q.put(None)
                    return
                gstep, meta, raw = item
                # same slice→upcast commute as the mmap arm: the buffer
                # holds exactly the shard's slice bytes, so the fp32 values
                # are bit-identical to slicing the mmap'd payload first
                local = {}
                for e in self.sp.entries:
                    got = raw.get(e["name"])
                    if got is None:
                        continue
                    dtype, buf = got
                    src = torch.frombuffer(buf, dtype=dtype)
                    out_dtype = (torch.float32
                                 if dtype == torch.bfloat16 else dtype)
                    if self.pin:
                        pb = torch.empty(src.shape, dtype=out_dtype,
                                         pin_memory=True)
                        pb.copy_(src)
                        local[e["name"]] = pb
                    elif out_dtype is not dtype:
                        local[e["name"]] = src.float()
                    else:
                        local[e["name"]] = src.clone()
                self.q.put((gstep, meta, local))
        except BaseException as e:
            self.exc = e
            self.q.put(None)

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
            if self.timed:
                _t = time.perf_counter()
                item = self.q.get()
                self.qwait_s += time.perf_counter() - _t
            else:
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

    R.require_scan_dtypes(
        cfg.get("replay_dtype"),
        "fp32" if cfg.get("norm_fp32") is True else "fp64",
    )
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
    # select kill-switch (A/B arms and emergencies): force the bit-identical
    # CPU front-select path — the S0FRONT print echoes the arm as
    # front_sel=cpu. Same unknown-values-must-raise contract as
    # REPLAY_FUSED_FRONT_SORT. The streaming boundary re-selections are
    # unaffected (their scores are device-constructed under --device cuda
    # already); this flag governs the FRONT selection block only.
    front_sel = os.environ.get("REPLAY_FUSED_SELECT", "auto")
    if front_sel not in ("auto", "cpu"):
        raise ValueError(
            f"REPLAY_FUSED_SELECT={front_sel!r} is not one of the "
            "allowed values ('auto', 'cpu')")
    # rank-background kill-switch (A/B arms and emergencies): force the
    # synchronous front ranking path — the S0FRONT print echoes the arm as
    # rank_bg=off. Same unknown-values-must-raise contract as
    # REPLAY_FUSED_FRONT_SORT.
    rank_bg = os.environ.get("REPLAY_FUSED_FRONT_RANK_BG", "auto")
    if rank_bg not in ("auto", "off"):
        raise ValueError(
            f"REPLAY_FUSED_FRONT_RANK_BG={rank_bg!r} is not one of the "
            "allowed values ('auto', 'off')")
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    comms = Comms(rank, world, port)

    t0 = cfg["t0"]
    replay_dtype = R.resolve_replay_dtype(cfg["replay_dtype"])
    # replay-kernel kill-switch (A/B arms and emergencies): force the
    # torch eager path — same unknown-values-must-raise contract as the
    # other replay flags
    kmode = os.environ.get("REPLAY_FUSED_KERNEL", "auto")
    if kmode not in ("auto", "off"):
        raise ValueError(
            f"REPLAY_FUSED_KERNEL={kmode!r} is not one of the "
            "allowed values ('auto', 'off')")
    from sga import replay_kernels as K
    K.MODE = kmode
    k_on = K.arm(dev.startswith("cuda"), replay_dtype)
    if comms.global_rank == 0:
        print(f"[fused] replay kernels: {'triton' if k_on else 'torch'}"
              f" (REPLAY_FUSED_KERNEL={kmode})", flush=True)
    # g-read kill-switch (A/B arms and emergencies): pread pulls the shard's
    # slice bytes straight; mmap keeps the page-fault reader. Same
    # unknown-values-must-raise contract as the other replay flags
    p1_read = os.environ.get("REPLAY_FUSED_P1_READ", "pread")
    if p1_read not in ("pread", "mmap"):
        raise ValueError(
            f"REPLAY_FUSED_P1_READ={p1_read!r} is not one of the "
            "allowed values ('pread', 'mmap')")
    if comms.global_rank == 0:
        print(f"[fused] g read: {p1_read} (REPLAY_FUSED_P1_READ={p1_read})",
              flush=True)
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
    optimizers = cfg["optimizers"]
    specs = {s: R.get_score_spec(s) for s in score_names}
    t0step = int(meta["step"])
    chunk_n = R._chunk_elems(cfg["chunk_mb"], 8)
    # score vectors are per (score, optimizer): the branch-point selection
    # and stats depend on the optimizer through the formula's v-basis and
    # bias corrections (fold/dual-lr do not touch the branch point — no
    # residual exists yet and the split applies at commits only)
    score_loc = {(s, o): torch.empty(sp.n, dtype=torch.float32)
                 for s in score_names for o in optimizers}
    for e in sp.entries:
        b1, b2, eps, wd = e["b1"], e["b2"], e["eps"], e["wd"]
        bcs = {opt: R._bc_pair(opt, t0step + 1, b1, b2)
               for opt in optimizers}
        # the fp64 chunk casts/upload read the SAME w0/m0/v0/g0 basis for
        # every optimizer family — build them once per chunk, shared
        # across the optimizer loop, so an N-optimizer run pays the input
        # cost once instead of N times (the per-opt outputs and the bc
        # pairs are the only optimizer-dependent state)
        for a in range(0, e["n"], chunk_n):
            b = min(e["n"], a + chunk_n)
            w64 = e["w0"][a:b].to(dtype=torch.float64, device=dev)
            m64 = e["m0"][a:b].to(dtype=torch.float64, device=dev)
            v64 = e["v0"][a:b].to(dtype=torch.float64, device=dev)
            g64 = e["g0"][a:b].to(dtype=torch.float64, device=dev)
            for opt in optimizers:
                bc1p, bc2p = bcs[opt]
                for s in score_names:
                    sc = R.score_formula(specs[s], w64, m64, v64, g64,
                                         b1, b2, eps, wd, bc1p, bc2p, opt,
                                         update_includes_wd=cfg["cycle_mode"])
                    score_loc[(s, opt)][e["a"] + a:e["a"] + b] = sc.abs()

    # ---- dense artifacts: the parent populated the cache before spawn -----
    # (the cache roundtrip is bitwise — pinned by test_dense_cache_roundtrip;
    # one cache per optimizer: the dense reference trajectory — label, den,
    # snapshots — depends on the step recursion)
    if _S0_PROFILE:
        _tf_scores = time.perf_counter() - _tf0
        _tf0 = time.perf_counter()
    else:
        _tf_scores = 0.0
    label_loc, den_at, label_total, snap_loc, stats_label = {}, {}, {}, {}, {}
    recon_steps = None
    for opt in optimizers:
        ident = R._dense_cache_ident(
            cfg["trace_dir"], t0, window_steps, eff_horizons, replay_dtype,
            cfg["trace_dtype"],
            "cpu" if cfg["device"] == "cpu" else "dev", opt)
        cached = R.load_dense_cache(cfg["dense_cache"], ident)
        if cached is None:
            raise RuntimeError(
                "dense cache still missing after parent populate "
                f"(optimizer {opt})")
        lv = cached["label_flat"][lo:hi].clone()
        # the device-permutation fast path relies on finite non-negative sort
        # keys (score_loc passes the shard_select isfinite gate; label gets one
        # upfront check here — non-finite labels would sort inconsistently
        # across backends)
        if not bool(torch.isfinite(lv).all().item()):
            raise ValueError("dense-cache label vector contains non-finite "
                             "entries; check the trace/cache integrity")
        label_loc[opt] = lv
        den_at[opt] = cached["den_at"]
        # the per-step pre-clip reconstruction diagnostics are pure functions
        # of the logged gradients (no optimizer state) — identical in every
        # optimizer's cache; report the first optimizer's copy
        if recon_steps is None:
            recon_steps = cached["recon_steps"]
        label_total[opt] = cached["label_total"]
        snap_loc[opt] = {}
        for h in eff_horizons:
            snap_loc[opt][h] = {}
            for e in sp.entries:
                t = cached["snap_d"][h][e["name"]].reshape(-1)
                snap_loc[opt][h][e["name"]] = \
                    t[e["p_off"]:e["p_off"] + e["n"]].clone()
        del cached
    dense_cache_hit = cfg["dense_cache_hit"]  # parent-side load outcome;
    # the streaming pass itself only ever consumes the cache
    if _S0_PROFILE:
        _tf_cache = time.perf_counter() - _tf0
    else:
        _tf_cache = 0.0

    # ---- score / label statistics (sums shard-partial; min/max exact) ------
    stats = {}
    for opt in optimizers:
        for s in score_names:
            stats[(s, opt)] = _shard_stats(score_loc[(s, opt)], comms)
        stats_label[opt] = _shard_stats(label_loc[opt], comms)

    # ---- branch selections + label selections + ranking metrics -----------
    ratios = cfg["ratios"]
    cycle_lens = cfg["cycle_lens"]
    folds = cfg["folds"]
    dual_lrs = cfg["dual_lrs"]
    ks = {r: max(1, int(round(r / 100.0 * N))) for r in ratios}
    idcg, sel0, thresh0, rankmet = {}, {}, {}, {}
    # selection semantics applies to BOTH select sites (the branch-point
    # front select below AND the boundary reselection in the streaming
    # loop): the online engine selects with the same semantics at the
    # branch point and at every boundary — an arm wired at only one site
    # diverges from step 0. Read once here; both select sites below
    # reuse the same name.
    sel_semantics = cfg.get("selection_semantics", "balanced-histogram")
    # device front-select decision: unanimous across ranks (a partial
    # fallback would split the gather collectives and deadlock). With the
    # arm engaged, the CPU originals stay untouched for _shard_stats (its
    # fp32 sum is device-divergent) while device copies feed shard_select,
    # the mask indexing, and the gather payloads; gather parts come back
    # to CPU (exact movement) so every downstream reduction keeps its
    # original CPU expression path.
    _sel_ok = (front_sel == "auto" and dev.startswith("cuda")
               and _device_ok_for_select(sp.n, ks[max(ratios)], world,
                                         len(score_names) * len(optimizers),
                                         dev))
    sel_dev_used = dev if comms.vote_unanimous(_sel_ok) else None
    gather_backend = (comms.vote_gather_backend()
                      if sel_dev_used is not None else "gloo")
    if sel_dev_used is not None:
        score_sel = {k: v.to(sel_dev_used) for k, v in score_loc.items()}
        label_sel = {o: label_loc[o].to(sel_dev_used) for o in optimizers}
    else:
        score_sel, label_sel = score_loc, label_loc

    def _gather(payload):
        if sel_dev_used is not None:
            return comms.all_gather_pad_dev(payload, gather_backend)
        return comms.all_gather_pad(payload)

    _tf_sel = 0.0
    _tf_rank = 0.0
    # rank0's ranking chain runs on a bounded background thread (the
    # _FrontRankWorker quarantine contract: plain CPU tensors/floats cross
    # the queue, never comms, never a dist call) so the main pipeline
    # reaches the next shard_select collective without waiting out the
    # ranking; world==1 takes the same path (shard_select is heavy local
    # compute there too). rank_bg=off keeps the synchronous path.
    rank_worker = None
    if rank_bg == "auto" and comms.global_rank == 0:
        rank_worker = _FrontRankWorker(sort_dev, ks, stats, label_total)
        rank_worker.start()
    _front_failed = False
    try:
        for opt in optimizers:
            for r in ratios:
                _tq = time.perf_counter() if _S0_PROFILE else 0.0
                lsel, _lt = shard_select(label_sel[opt], ks[r], comms)
                parts = _gather(label_sel[opt][lsel].clone())
                if _S0_PROFILE:
                    _tf_sel += time.perf_counter() - _tq
                    _tq = time.perf_counter()
                if rank_worker is not None:
                    rank_worker.submit(("idcg", opt, r, parts))
                elif comms.global_rank == 0:
                    idcg[(opt, r)] = _idcg_of_parts(parts, dev=sort_dev)
                if _S0_PROFILE and rank_worker is None:
                    _tf_rank += time.perf_counter() - _tq
                for s in score_names:
                    # the branch-point selection is a function of (score,
                    # ratio, optimizer) only, and the selector is
                    # deterministic on its inputs — every T/fold/dual-lr arm
                    # of one (s, r, opt) shares a single select instead of
                    # paying the axis product in redundant global selects;
                    # sel0 is only read to build the per-config device masks
                    # (never mutated)
                    _tq = time.perf_counter() if _S0_PROFILE else 0.0
                    # the ranking diagnostics (idcg/DCG/rankmet below) are
                    # size-pinned to exact k and cross-arm comparable only
                    # under ONE selection rule — they stay on exact topk
                    # for every semantics; the balanced arm swaps ONLY the
                    # trajectory-driving mask (sel0/thresh0).
                    sel, th = shard_select(score_sel[(s, opt)], ks[r],
                                           comms)
                    sc_parts = _gather(score_sel[(s, opt)][sel].clone())
                    lb_parts = _gather(label_sel[opt][sel].clone())
                    ls_parts = _gather(lsel[sel].clone())
                    if _S0_PROFILE:
                        _tf_sel += time.perf_counter() - _tq
                        _tq = time.perf_counter()
                    met_sr = None
                    if rank_worker is not None:
                        rank_worker.submit(("rank", s, opt, r, sc_parts,
                                            lb_parts, ls_parts))
                    elif comms.global_rank == 0:
                        met_sr = _fused_ranking(
                            torch.cat(sc_parts), torch.cat(lb_parts),
                            ls_parts, ks[r], stats[(s, opt)]["sum"],
                            label_total[opt], idcg[(opt, r)], dev=sort_dev)
                    if _S0_PROFILE and rank_worker is None:
                        _tf_rank += time.perf_counter() - _tq
                    if sel_semantics == BALANCED_HISTOGRAM:
                        # S0: the full-vector balanced pass is a front
                        # select too — bracket it into select_s instead of
                        # leaving its cost attributed to nothing
                        _tq = time.perf_counter() if _S0_PROFILE else 0.0
                        sel0[(s, r, opt)], thresh0[(s, r, opt)] = (
                            shard_select_balanced(
                                score_sel[(s, opt)], ks[r], comms,
                                sp.entries))
                        if _S0_PROFILE:
                            _tf_sel += time.perf_counter() - _tq
                    else:
                        sel0[(s, r, opt)] = sel
                        thresh0[(s, r, opt)] = th
                    if rank_worker is None and comms.global_rank == 0:
                        rankmet[(s, r, opt)] = met_sr
    except BaseException:
        _front_failed = True
        raise
    finally:
        # failure-path shutdown; the success path joins via finish() below
        if rank_worker is not None and _front_failed:
            rank_worker.close_best_effort()
    # release the front-only device residents before the streaming phase's
    # own peak (g-residency + per-group states run the card near full):
    # the uploaded vectors and the loop-local select masks (label sel + the
    # last score sel) are dead (the vectors' CPU originals persist), and
    # the sel0 masks go back to CPU so the streaming setup's slice-upload
    # path behaves exactly as in the CPU arm — a device-resident mask would
    # otherwise alias-hold every full-size mask through streaming via the
    # per-entry slice views, and the whole set plus the NCCL buffers does
    # not fit under the streaming peak
    del score_sel, label_sel, lsel, sel
    if sel_dev_used is not None:
        for _k, _v in sel0.items():
            sel0[_k] = _v.cpu()
        torch.cuda.empty_cache()
    # the release block above overlaps the worker's queue drain; the join
    # lands here so a worker failure re-raises ahead of any use
    rank_joinwait = 0.0
    if rank_worker is not None:
        _tj = time.perf_counter()
        rankmet = rank_worker.finish()
        rank_joinwait = time.perf_counter() - _tj
    if _S0_PROFILE and comms.global_rank == 0:
        # front-phase decomposition (diagnostic labels, not exact
        # accounting): branch-score math / dense-cache load / shard_select
        # + gathers / ranking+idcg sorts. With rank_bg=on the rank figure
        # is the worker's own accumulation and the main thread's share of
        # the bracket shows up as backpressure (qwait) plus the front-end
        # drain (joinwait); with rank_bg=off rank_s is the main-thread
        # bracket as before
        if rank_worker is not None:
            _rank_f = (f"rank_s={rank_worker.rank_s:.1f} "
                       f"qwait_s={rank_worker.qwait_s:.1f} "
                       f"joinwait_s={rank_joinwait:.1f}")
        else:
            _rank_f = f"rank_s={_tf_rank:.1f}"
        print(f"S0FRONT rank={rank} scores_s={_tf_scores:.1f} "
              f"cache_s={_tf_cache:.1f} select_s={_tf_sel:.1f} "
              f"{_rank_f} "
              f"front_sort={'cpu' if sort_dev is None else 'dev'} "
              f"front_sel={'cpu' if sel_dev_used is None else 'dev'} "
              f"gather={gather_backend} "
              f"rank_bg={'on' if rank_worker is not None else 'off'}",
              flush=True)

    # ---- streaming passes over config groups ------------------------------
    # The three optimizer-axis values enumerate innermost; default singletons
    # preserve the historical (score, ratio, T) order and group partition.
    configs = [(s, r, t, opt, fold, dl)
               for r in ratios for s in score_names for t in cycle_lens
               for opt in optimizers for fold in folds for dl in dual_lrs]
    # T-relative dual-lr tokens resolve against each arm's own cycle length.
    dl_scale_of = {c: R.resolve_dual_lr_token(c[5], c[2]) for c in configs}
    # group-size cap: with a state byte budget set (>0) the cap is derived
    # from the resident {w,m,v,res} bytes of ONE config over the WIDEST
    # shard — ceil(N/world) elements on every rank, so the partition stays a
    # deterministic pure function (a per-shard sp.n basis could split the
    # ranks at the floor and tear the qn2/num2 collectives). The budget
    # overrides --fused-max-configs; the bool sel mask, the --chunk-mb
    # working set and g-residency buffers sit outside the accounting.
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
    num_at = {c: {} for c in configs}
    # one counter dict per config group, stashed into shard 0's fragment
    # alongside the S0PROFILE print (env-gated: None when the env is absent)
    s0_profs = [] if _S0_PROFILE else None
    t_pass0 = time.time()
    # one zip-index cache per worker, handed to every group's prefetcher:
    # the streaming loop re-reads the same window files once per group
    index_cache = {}
    # group-wall clock matching the "done in Ns" print's span, so the gap
    # buckets can be coverage-checked against it:
    # (stp + qw + h2dl + pt[] + hzar + cln) / gw
    _tgw0 = time.perf_counter() if _S0_PROFILE else 0.0
    for gi, group in enumerate(groups):
        states = {}
        # sel0[(s,r,t)] is the SAME CPU selection mask for every t of one
        # (s,r) — CPU-resident in every arm here (the front block released
        # its device copies or fetched the masks back before this loop) —
        # key the upload cache by (s, r, entry) so "same selection across T
        # arms" is expressed directly and the slice + upload happens once
        # per distinct selection instead of once per arm (the masks are
        # read-only downstream; reselection replaces the tensor, never
        # mutates in place, so aliasing is safe)
        sel_dev_cache = {}
        _t0s = time.perf_counter() if _S0_PROFILE else 0.0
        for c in group:
            s, r, t, opt, _fold, _dl = c
            sel = sel0[(s, r, opt)]
            stt = {}
            for e in sp.entries:
                # _fresh_state: never alias the shard's w0/m0/v0 — every
                # arm in the group must start from the pristine state
                sel_key = (s, r, opt, e["name"])
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
            states[c] = stt
        _stp_s = (time.perf_counter() - _t0s) if _S0_PROFILE else 0.0
        chunk_dev = R._chunk_elems(cfg["chunk_mb"], itemsize)
        norm_fp32 = cfg["norm_fp32"]

        res_on = bool(cfg.get("g_residency"))
        red_on = bool(cfg.get("dev_reduce"))
        pre = GPrefetcher(g_files, window_steps, sp, cfg["upcast_bf16"],
                          pin=(res_on and dev != "cpu"), timed=_S0_PROFILE,
                          read_mode=p1_read, index_cache=index_cache)
        # boundary steps without a g payload read zeros; the resident path
        # caches one device zeros buffer per entry instead of materializing
        # a host zeros tensor per (config, entry, step)
        zeros_dev = {}
        # S0 counters are diagnostic labels, not exact measurements:
        # gloo_ar counts logical sum_f64 calls (nonzero even at world 1,
        # where the op is an identity), syncs counts only the norm/numerator
        # fetches, mem_peak_gb (printed at group end) is the process-lifetime
        # cumulative peak, and residency books every shared upload byte to
        # pass 1. The gap buckets (qw/h2dl/p1d2h/p1ar/hzar/stp/cln/gw) split
        # the group wall three ways — own read-wait (qw), lockstep wait
        # (p1ar/hzar), real work (pt[] + h2d_launch + stp/cln) — so the
        # streaming residual outside pt[] is attributed, not inferred;
        # p1d2h stays 0 when dev_reduce is off (the per-chunk float()
        # syncs are then fused into pt[0] and not separable cheaply)
        prof = {"sync": 0, "ar": 0, "h2d": [0, 0, 0],
                "pt": [0.0, 0.0, 0.0], "p2f": 0.0,
                "p2s": 0.0, "p2s_hist": 0.0, "p2s_ar": 0.0,
                "p2s_cand": 0.0, "p2s_fill": 0.0,
                "qw": 0.0, "h2dl": 0.0, "p1d2h": 0.0, "p1ar": 0.0,
                "hzar": 0.0, "stp": 0.0, "cln": 0.0, "gw": 0.0} \
            if _S0_PROFILE else None
        if prof:
            # diagnostic memory accounting: the group's resident state at
            # its full per-rank truth ({w,m,v,res} at replay itemsize plus
            # the 1 B/elem bool sel) — the budget knob itself counts the
            # four state tensors only, this field books everything
            prof["state_gb"] = (len(group) * sp.n * (4 * itemsize + 1)
                                / 2 ** 30)
            prof["stp"] = _stp_s
        for si, (_gstep, gmeta, local) in enumerate(pre):
            coef_dense, _flag = R.coef_dense_for(gmeta)
            clip_grad = float(gmeta.get("clip_grad", 0.0))
            count = t0 + si + 1
            # per-config calendars: arm (s, r, t, opt, fold, dl) full-syncs
            # iff si % t == 0. Mixed-T (and mixed-axis) runs reproduce
            # solo-axis arms bitwise because each arm's state machine
            # consumes only its own calendar/states, the qn2/num2 exchanges
            # are per-arm-indexed lists (elementwise reductions — the list
            # length never enters a lane's value) and chunking / entry order
            # are group-invariant
            boundary_c = [si % c[2] == 0 for c in group]
            fold_c = [c[4] for c in group]
            dl_scale = [dl_scale_of[c] for c in group]
            # residual fold (fold=on arms): divide the accumulated residual
            # by the cycle length T at the boundary, IN PLACE and BEFORE the
            # pass-1 clip norm — the boundary norm (pass 1), the boundary
            # re-selection score (pass 2) and the commit (pass 3) all read
            # the folded value, exactly where the online residual_mean fold
            # lands it (the grad buffer on the pre-boundary step, ahead of
            # everything the boundary step computes; folding only the commit
            # would re-clip against an unfolded norm — a semantics the
            # online path never produces). The online kernel multiplies by
            # the fp32 host reciprocal of the window; the oracle divides
            # exactly in replay_dtype
            if si > 0:
                for ci, c in enumerate(group):
                    if boundary_c[ci] and fold_c[ci]:
                        for e in sp.entries:
                            states[c][e["name"]]["res"].div_(float(c[2]))
            # dual-lr boundary split: the outgoing window's mask (the mask
            # the cycle's sparse steps used) gates the applied lr — pass 2
            # replaces the masks, so the pre-replacement slices are stashed
            # here for pass 3 (references only; the mask tensors are never
            # mutated, reselection replaces them)
            out_sel = {}
            if res_on:
                # one H2D per entry per step, shared by every config and
                # all three passes (the legacy path re-uploads per config
                # per pass: max_configs x passes x the same bytes). The
                # coef_dense division is per-step too, so it is folded in
                # here once instead of per chunk per config per pass —
                # elementwise, identical values either way.
                _t0h = time.perf_counter() if prof else 0.0
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
                    # launch-side time only (pinned H2D and the div kernels
                    # are async); the completion wait lands in p1d2h below
                    prof["h2dl"] += time.perf_counter() - _t0h
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
            for ci, c in enumerate(group):
                s, r, t, opt, fold, dl = c
                bnd = boundary_c[ci]
                for e in sp.entries:
                    g = local.get(e["name"])
                    if g is None and not bnd:
                        continue
                    g_el = None if res_on else (
                        torch.zeros(e["n"], dtype=torch.float32)
                        if g is None else g)
                    stt = states[c][e["name"]]
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
                _t0d = time.perf_counter() if prof else 0.0
                qn2 = qn2_dev.cpu().tolist()
                if prof:
                    # first device sync of the step: absorbs completion of
                    # the resident H2D burst + the norm partials
                    prof["p1d2h"] += time.perf_counter() - _t0d
                    prof["sync"] += 1
            _t0a = time.perf_counter() if prof else 0.0
            qn2 = comms.sum_f64(qn2)
            if prof:
                # clip-norm all_reduce: the per-step lockpoint — measures
                # rank-skew wait directly (sub-bucket of pt[0])
                prof["p1ar"] += time.perf_counter() - _t0a
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
            for ci, c in enumerate(group):
                s, r, t, opt, fold, dl = c
                if not (boundary_c[ci] and si > 0 and si + 1 < W):
                    continue
                spec = specs[s]
                # the replay-kernel formula arm hardcodes the AdamW v-basis;
                # AdamS scores take the torch formula path with the v
                # substitution (score_formula)
                _kf = K.formula(s) if (k_on and opt == "adamw") else None
                score_flat = torch.empty(sp.n, dtype=torch.float32,
                                         device=dev)
                if prof:
                    # p2 attribution split (formula+q-build vs the select
                    # collective): device-sync brackets, otherwise async
                    # kernels bleed across the boundary and the split
                    # measures launch order instead of execution
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    _t0f = time.perf_counter()
                for e in sp.entries:
                    g = local.get(e["name"])
                    g_el = None if res_on else (
                        torch.zeros(e["n"], dtype=torch.float32)
                        if g is None else g)
                    stt = states[c][e["name"]]
                    b1, b2, eps, wd = e["b1"], e["b2"], e["eps"], e["wd"]
                    bc1c, bc2c = R._bc_pair(opt, count, b1, b2)
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
                        sc = (_kf(stt["ws"][a:b], stt["ms"][a:b],
                                  stt["vs"][a:b], q, b1, b2, eps, wd,
                                  bc1c, bc2c)
                              if _kf is not None else
                              R.score_formula(spec, stt["ws"][a:b],
                                              stt["ms"][a:b], stt["vs"][a:b],
                                              q, b1, b2, eps, wd, bc1c, bc2c,
                                              opt,
                                              update_includes_wd=True))
                        score_flat[e["a"] + a:e["a"] + b] = sc.abs() \
                            .float()
                if prof:
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    prof["p2f"] += time.perf_counter() - _t0f
                    _t0f = time.perf_counter()
                if sel_semantics == BALANCED_HISTOGRAM:
                    # no prof= here: the p2s bracket below is the single
                    # accounting for both semantics
                    newsel, _th = shard_select_balanced(
                        score_flat, ks[r], comms, sp.entries)
                else:
                    newsel, _th = shard_select(score_flat, ks[r], comms,
                                               prof=prof)
                if prof:
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    prof["p2s"] += time.perf_counter() - _t0f
                for e in sp.entries:
                    stt = states[c][e["name"]]
                    if dl_scale[ci] != 1.0:
                        # stash the OUTGOING mask ahead of the replacement:
                        # the boundary dual-lr split applies against it (the
                        # online reselection runs after the commit)
                        out_sel.setdefault(ci, {})[e["name"]] = stt["sel"]
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
            for ci, c in enumerate(group):
                s, r, t, opt, fold, dl = c
                bnd = boundary_c[ci]
                for e in sp.entries:
                    g = local.get(e["name"])
                    g_el = None if res_on else (
                        torch.zeros(e["n"], dtype=torch.float32)
                        if g is None else g)
                    stt = states[c][e["name"]]
                    lr = R._lr_for_step(gmeta, e["group"])
                    b1, b2, eps, wd = e["b1"], e["b2"], e["eps"], e["wd"]
                    bc1c, bc2c = R._bc_pair(opt, count, b1, b2)
                    split = (bnd and dl_scale[ci] != 1.0)
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
                        if split:
                            # boundary dual-lr: the whole step displacement
                            # (wd component included) applies at lr*scale on
                            # the mask-inactive tail, at lr on the
                            # communicated coords — against the OUTGOING
                            # window's mask
                            sel_out = (out_sel[ci][e["name"]] if ci in
                                       out_sel else stt["sel"])
                            inactive = ~sel_out[a:b]
                            lr_i = lr * dl_scale[ci]
                        else:
                            inactive = lr_i = None
                        if k_on:
                            K.opt_step_inplace(
                                q, stt["ms"][a:b], stt["vs"][a:b],
                                stt["ws"][a:b], None, lr, wd, b1, b2,
                                bc1c, bc2c, eps, optimizer=opt,
                                inactive=inactive, lr_inactive=lr_i)
                        else:
                            R._opt_step_inplace(
                                q, stt["ms"][a:b], stt["vs"][a:b],
                                stt["ws"][a:b], None, lr, wd, b1, b2,
                                bc1c, bc2c, eps, optimizer=opt,
                                inactive=inactive, lr_inactive=lr_i)
                        if hz is not None:
                            d = (stt["ws"][a:b]
                                 - snap_loc[opt][hz][e["name"]][a:b]
                                 .to(dev)) \
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
                _t0z = time.perf_counter() if prof else 0.0
                num2 = comms.sum_f64(num2)
                if prof:
                    prof["hzar"] += time.perf_counter() - _t0z
                    prof["ar"] += 1
                for ci, c in enumerate(group):
                    num_at[c][hz] = math.sqrt(num2[ci])

        _t0c = time.perf_counter() if prof else 0.0
        del states
        if dev != "cpu":
            torch.cuda.empty_cache()
        if prof:
            prof["cln"] += time.perf_counter() - _t0c
        if comms.global_rank == 0 and cfg["verbose"]:
            print(f"[fused t0={t0}] group {gi + 1}/{len(groups)} "
                  f"({len(group)} configs) done in "
                  f"{time.time() - t_pass0:.0f}s", flush=True)
            t_pass0 = time.time()
        if prof:
            # g slices are fp32 host-side (prefetcher upcasts), 4 B/elem
            mem_gb = (torch.cuda.max_memory_allocated() / 2 ** 30
                      if dev != "cpu" else 0.0)
            prof["qw"] = pre.qwait_s
            prof["gw"] = time.perf_counter() - _tgw0
            _tgw0 = time.perf_counter()
            _cov = (prof["stp"] + prof["qw"] + prof["h2dl"]
                    + sum(prof["pt"]) + prof["hzar"] + prof["cln"]
                    ) / prof["gw"] if prof["gw"] > 0 else 0.0
            print(f"S0PROFILE rank={rank} group={gi} steps={W} "
                  f"syncs={prof['sync']} gloo_ar={prof['ar']} "
                  f"state_gb={prof['state_gb']:.2f} "
                  f"h2d_gb_p1={prof['h2d'][0] * 4 / 2 ** 30:.1f} "
                  f"h2d_gb_p2={prof['h2d'][1] * 4 / 2 ** 30:.1f} "
                  f"h2d_gb_p3={prof['h2d'][2] * 4 / 2 ** 30:.1f} "
                  f"pass_s_p1={prof['pt'][0]:.0f} "
                  f"pass_s_p2={prof['pt'][1]:.0f} "
                  f"pass_s_p3={prof['pt'][2]:.0f} "
                  f"p2_formula_s={prof['p2f']:.1f} "
                  f"p2_select_s={prof['p2s']:.1f} "
                  f"p2s_hist={prof['p2s_hist']:.1f} "
                  f"p2s_ar={prof['p2s_ar']:.1f} "
                  f"p2s_cand={prof['p2s_cand']:.1f} "
                  f"p2s_fill={prof['p2s_fill']:.1f} "
                  f"gw_s={prof['gw']:.0f} "
                  f"qwait_s={prof['qw']:.0f} "
                  f"h2d_launch_s={prof['h2dl']:.1f} "
                  f"p1_d2h_s={prof['p1d2h']:.1f} "
                  f"p1_ar_s={prof['p1ar']:.1f} "
                  f"hz_ar_s={prof['hzar']:.1f} "
                  f"setup_s={prof['stp']:.1f} "
                  f"clean_s={prof['cln']:.1f} "
                  f"cov={_cov:.2f} "
                  f"mem_peak_gb={mem_gb:.1f}",
                  flush=True)
            s0_profs.append(dict(prof))

    # ---- global rank 0 assembles the full branch fragment ------------------
    # (node 0's local shard 0 in the multi-node topology — the launcher's
    # frag_path is only meaningful on the launcher's own machine; node 0
    # itself is launched by ssh-to-self like every other node, so its /tmp
    # IS the launcher's filesystem view)
    out = None
    if comms.global_rank == 0:
        # key-suffix rules (the collision-avoidance precedent of the T set):
        # an axis suffixes the bare score name only when that axis actually
        # recurs in the run, so every default-singleton run (one T, adamw,
        # fold off, dl 1) keeps the historical bare keys. Selection semantics
        # are echoed separately by the CLI report assembler.
        multi_t = len(cycle_lens) > 1
        multi_opt = len(optimizers) > 1
        multi_fold = len(folds) > 1
        multi_dl = len(dual_lrs) > 1

        def _arm_key(s, t, opt, fold, dl):
            key = s
            if multi_t:
                key += f"@T{t}"
            if multi_opt:
                key += f"@{opt}"
            if multi_fold and fold:
                key += "@fold"
            if multi_dl:
                key += f"@dl{dl:g}"
            return key

        per_ratio = {}
        for r in ratios:
            scores = {}
            for s in score_names:
                for t in cycle_lens:
                    for opt in optimizers:
                        for fold in folds:
                            for dl in dual_lrs:
                                key = _arm_key(s, t, opt, fold,
                                               dl_scale_of[(s, r, t, opt,
                                                            fold, dl)])
                                distortion = {}
                                for h in cfg["horizons"]:
                                    hh = max(1, min(h, W))
                                    den = den_at[opt].get(hh)
                                    num = num_at[(s, r, t, opt, fold,
                                                  dl)].get(hh)
                                    distortion[str(h)] = (
                                        num / den if (den is not None
                                                      and num is not None
                                                      and den > 0)
                                        else None)
                                scores[key] = {
                                    "threshold": thresh0[(s, r, opt)],
                                    "distortion": distortion,
                                    **rankmet[(s, r, opt)],
                                }
            per_ratio[str(r)] = {"k": ks[r], "scores": scores}
        # one den/label view at one optimizer keeps the historical flat
        # shape; with several optimizers the per-optimizer views nest under
        # the optimizer name (each arm family's distortion above already
        # divides by its own family's denominator)
        if len(optimizers) == 1:
            den_view = {str(h): den_at[optimizers[0]].get(h)
                        for h in eff_horizons
                        if den_at[optimizers[0]].get(h) is not None}
            label_total_view = label_total[optimizers[0]]
            label_stats_view = stats_label[optimizers[0]]
        else:
            den_view = {
                opt: {str(h): den_at[opt].get(h) for h in eff_horizons
                      if den_at[opt].get(h) is not None}
                for opt in optimizers}
            label_total_view = dict(label_total)
            label_stats_view = dict(stats_label)
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
                "den_norm_at_horizon": den_view,
                "label_total": label_total_view,
                "label_stats": label_stats_view,
            },
            "scores": {(s if not multi_opt else f"{s}@{opt}"):
                       stats[(s, opt)]
                       for s in score_names for opt in optimizers},
            "per_ratio": per_ratio,
        }
        # conditional echoes: the budget-off report shape is frozen — these
        # keys only exist when the knob is on (cycle_lens above is echoed
        # unconditionally)
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
                             replay_dtype, optimizer):
    """Cache-miss fallback: run the classic dense pass (full params) in the
    parent process and save the standard dense cache — once per trace and
    optimizer (the dense reference trajectory — label, den, snapshots —
    depends on the step recursion, so each optimizer owns its cache file)."""
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
            upcast_bf16=cfg["upcast_bf16"], optimizer=optimizer)
    else:
        label_total, den_at, recon_steps, snap_d = R.dense_replay_dev(
            params, g_files, window_steps, meta, t0, eff_horizons,
            replay_dtype, cfg["chunk_mb"], "cuda:0",
            upcast_bf16=cfg["upcast_bf16"], optimizer=optimizer)
    label_flat = torch.cat([p["label"].float() for p in params])
    R.save_dense_cache(cfg["dense_cache"], R._dense_cache_ident(
        cfg["trace_dir"], t0, window_steps, eff_horizons, replay_dtype,
        cfg["trace_dtype"],
        "cpu" if cfg["device"] == "cpu" else "dev", optimizer), label_total,
        den_at, recon_steps, snap_d, label_flat)
    print(f"[fused t0={t0}] dense cache populated on miss for {optimizer} "
          f"({cfg['dense_cache']})", flush=True)


# ----------------------------------------------------------------------------
# entry points (called from sga.replay main with a plain config dict)
# ----------------------------------------------------------------------------

def _resolve_axes(args):
    """Resolve the per-config cycle-engine axes (optimizer / fold /
    dual-lr) for one args namespace, mirroring the CLI gate for programmatic
    callers: getattr defaults keep Namespaces built before the axes existed
    on the historical default singletons (adamw / off / 1), which reduce the
    config space to the historical (score, ratio, T) tuples. Returns
    (optimizers, folds, dual_lr_tokens)."""
    import sga.replay as R
    cycle_lens = getattr(args, "cycle_lens", None)
    if cycle_lens is None:
        cycle_lens = [args.cycle_len]
    opts = getattr(args, "optimizer_set", None)
    if opts is None:
        raw = getattr(args, "optimizer", None)
        opts = (R.parse_optimizer_set(raw) if raw is not None
                else [R.DEFAULT_OPTIMIZER])
    folds = getattr(args, "fold_set", None)
    if folds is None:
        raw = getattr(args, "fold", None)
        folds = R.parse_fold_set(raw) if raw is not None else [False]
    dl = getattr(args, "dual_lr_set", None)
    if dl is None:
        raw = getattr(args, "dual_lr", None)
        dl = (R.parse_dual_lr_set(raw, cycle_lens) if raw is not None
              else ["1"])
    return opts, folds, dl


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
    R.require_scan_dtypes(args.replay_dtype, args.norm_dtype)
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
    # the per-config axes ride the same resolution as the CLI gate (unknown
    # optimizer names, bad fold spellings and bad/ambiguous dual-lr tokens
    # fail loudly here for direct programmatic callers too)
    _resolve_axes(args)
    for r in args.ratios:
        if r <= 0.0 or r > 100.0:
            raise ValueError(f"ratio {r} out of (0,100]")
    if args.device != "cpu":
        if args.device != "cuda":
            raise ValueError(
                "--device cuda:N is not supported: shard r always maps to "
                "cuda:r. Use --device cuda and restrict the visible devices "
                "with CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=4,5 for "
                "a 2-shard run on physical GPUs 4 and 5)")
        nodes = getattr(args, "nodes", None)
        if nodes:
            # multinode early fail: the launcher runs no shards, so check
            # the per-node share (ceil — divisibility is the launcher's
            # own gate); replay_launch._cuda_shard_gate re-checks the real
            # local count on every node in both phases
            share = -(-world // len(nodes))
            if share > torch.cuda.device_count():
                raise ValueError(
                    f"--fused-shards {world} over {len(nodes)} nodes puts "
                    f"{share} shards on each, exceeding the visible CUDA "
                    f"devices on this node ({torch.cuda.device_count()})")
        elif world > torch.cuda.device_count():
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
    optimizers, folds, dual_lrs = _resolve_axes(args)
    return {
        "trace_dir": ctx["trace_dir"],
        "t0": t0,
        "window": args.window,
        "horizons": args.horizons,
        "ratios": args.ratios,
        "score_names": score_names,
        "cycle_lens": sorted(set(cycle_lens)),
        "cycle_mode": True,
        "optimizers": optimizers,
        "folds": folds,
        "dual_lrs": dual_lrs,
        "replay_dtype": args.replay_dtype,
        "trace_dtype": args.trace_dtype,
        "upcast_bf16": args.trace_dtype == "bf16",
        "selection_semantics": getattr(args, "selection_semantics",
                                      "balanced-histogram"),
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
    """Populate the dense cache(s) parent-side, before any shard exists: on
    a cold cache the dense pass can run for hours and must not happen inside
    the process group (the waiting shards would hit the gloo timeout).
    One cache per optimizer of the run (the dense reference trajectory
    depends on the step recursion; the ident carries the optimizer for
    non-adamw values so existing adamw caches stay hits). dense_cache_hit
    reports the parent-side LOAD outcome (False on the populating run),
    matching the classic report field semantics — a bare bool for a
    single-optimizer run (the historical shape), a per-optimizer dict
    otherwise. Returns {optimizer: ident}."""
    import sga.replay as R
    R.require_scan_dtypes(args.replay_dtype, args.norm_dtype)
    replay_dtype = R.resolve_replay_dtype(args.replay_dtype)
    window_steps, _W, eff_horizons = _window_and_horizons(
        ctx["g_files"], t0, args.window, args.horizons)
    engine = "cpu" if args.device == "cpu" else "dev"
    optimizers, _folds, _dl = _resolve_axes(args)
    idents, hits = {}, {}
    for opt in optimizers:
        ident = R._dense_cache_ident(ctx["trace_dir"], t0, window_steps,
                                     eff_horizons, replay_dtype,
                                     args.trace_dtype, engine, opt)
        idents[opt] = ident
        hits[opt] = R.load_dense_cache(args.dense_cache, ident) is not None
        if not hits[opt]:
            _compute_dense_and_cache(cfg, R, t0, window_steps, eff_horizons,
                                     replay_dtype, opt)
    if args.device != "cpu":
        # a populate pass ran on cuda:0 in THIS (parent) process and
        # its peak stays cached in the allocator, invisible to per-process
        # accounting but visible to the workers' sort guard, which reads
        # device-wide free memory — hand the blocks back before any
        # shard spawns (no device tensor is live at this point)
        torch.cuda.empty_cache()
    cfg["dense_cache_hit"] = (hits[optimizers[0]]
                              if len(optimizers) == 1 else hits)
    return idents


def run_fused_branch(args, ctx, t0):
    """One branch point through the fused engine (single node); returns the
    complete branch dict (same JSON shape as process_branch). Multi-node
    runs go through sga.replay_launch.run_fused_branch_multinode."""
    score_names = _common_gates(args)
    world = args.fused_shards
    if world < 1:
        raise ValueError("--fused-shards must be >= 1")
    if args.device == "cuda" and world > torch.cuda.device_count():
        raise ValueError(
            f"--fused-shards {world} exceeds the visible CUDA devices "
            f"({torch.cuda.device_count()})")
    cfg = _fused_cfg(args, ctx, t0, score_names)
    if cfg.get("selection_semantics", "balanced-histogram") != "topk" and \
            getattr(args, "nodes", None):
        # mirror of the CLI gate in sga.replay.main for programmatic
        # callers — fail before _populate_dense_cache (a cold-cache dense
        # pass can run for hours) and before any shard spawns; any
        # single-node shard count is fine (the sharded balanced select
        # reconstructs the catalog histogram from the shard partials)
        raise ValueError(
            "--selection-semantics balanced-histogram requires a "
            "single-node run (--nodes is not supported; single-node "
            "runs may use any --fused-shards)")
    _populate_dense_cache(args, ctx, cfg, t0)
    fd, frag_path = tempfile.mkstemp(suffix=".pt",
                                     prefix="fused_branch_")
    os.close(fd)
    try:
        if world == 1:
            _worker(0, 1, cfg, 0, frag_path)
        else:
            port = _free_port()
            mp.spawn(_worker, args=(world, cfg, port, frag_path),
                     nprocs=world, join=True)
        return torch.load(frag_path, map_location="cpu", weights_only=False)
    finally:
        if os.path.exists(frag_path):
            os.remove(frag_path)
