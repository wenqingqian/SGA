#!/usr/bin/env python3
"""Unit tests for the fused single-pass streaming engine (sga.replay_fused).

Run from anywhere (world=1 and world=2 both work on CPU; gloo over
localhost carries the tiny control-plane exchanges):

    python SGA/tests/test_replay_fused.py

Contract under test (the fused engine is the only cycle engine; cycle
auto-routes here):
1. world=1: the report's distortions match the from-spec reference
   (test_replay_cycle.reference_cycle) for every score x ratio x horizon;
   a miss run populates the cache and reports dense_cache_hit False, the
   hit run reproduces the whole report; the deprecated --fused flag is a
   no-op (identical report).
2. world=2: params straddle the shard boundary; the only deviations from
   the world=1 run are cross-shard fp64 scalar-sum reorderings —
   distortions and sum-type statistics stay within the pure-reorder band
   (rel <= 1e-6), everything that is not a sum (thresholds, ranking
   metrics, cache-sourced dense artifacts, min/max/zero fractions) is
   exact.
3. --norm-dtype fp32 threads through the fused engine (world=2 vs
   world=1, both arms fp32-norm, same band).
4. Cold cache with no prior run: the parent populates the cache before
   the shards start (dense_cache_hit False on the populating run), and
   the populated cache then serves a world=1 run.
5. CLI gates: cycle without --dense-cache, --fused outside cycle mode,
   --self-check with cycle
   or with --fused, shard count < 1, max-configs < 1, unknown scores,
   out-of-range ratios and an explicit cuda:N --device, each with the
   option named.
6. CUDA smoke (skipped without a GPU): world=1 on cuda (dense via
   dense_replay_dev) matches the from-spec reference within the
   sanitize/report band.
7. --cycle-lens T sets: every f"{score}@T{t}" arm of a mixed-T group is
   bitwise identical to the same arm under the solo-T runs (world 1 and
   2); --cycle-lens N equals --cycle-len N modulo the echo keys; the
   mutually-exclusive/invalid knob values are rejected by name.
8. Comms multi-node rendezvous env (SGA_MASTER_ADDR + SGA_MASTER_PORT,
   per-process SGA_GLOBAL_RANK): real gloo init over the master endpoint
   with the global process-group rank while the local shard index stays
   in ``rank``, collectives come back in global-rank order, and both env
   misconfigurations raise loudly.
9. Perf-flag mechanism (not just report values): the --fused-g-residency
   / --fused-dev-reduce echoes, the REPLAY_S0_PROFILE counters carried in
   the fragment, pinned prefetcher buffers on CUDA, and a --trace-dtype
   bf16 leg of the slice-first prefetcher (reference anchor + bitwise
   fp32-twin A/B). The same gates pin the pread arm (REPLAY_FUSED_P1_READ)
   bitwise against the mmap arm (across payload dtypes, range splits, and
   the pinned construction, plus its torch.save zip-index parser against
   torch.load ground truth), the p2 attribution split (p2f/p2s counter
   keys, p2_formula_s/p2_select_s line fields, and the p2s_hist/p2s_ar/
   p2s_cand/p2s_fill sub-buckets — all pinned by sub-timer nesting
   bounds), and the gap buckets (qw/h2dl/p1d2h/p1ar/hzar/stp/cln/gw:
   read-wait vs lockstep wait vs real work outside the pass brackets,
   coverage-checked against the group wall).
10. --fused-state-budget-mb: the resident-state byte budget derives the
   config group size (overrides --fused-max-configs); reports stay
   bit-identical to the count knob at the same partition AND across
   partitions; the echoes are conditional so budget-off artifacts keep
   their exact bytes; the cap derivation (floor + >=1 clamp) is pinned
   through the group_cap echo; REPLAY_S0_PROFILE gains a state_gb field.
11. Front select on device + NCCL transport + REPLAY_FUSED_SELECT
   kill-switch: the dev arm's reports are bit-identical to the forced-CPU
   arm (world 1 and, with 2 GPUs, world 2), and at world 2 the nccl
   payload gather is pinned bitwise against the gloo all_gather_pad
   ground truth on uneven and zero-length per-rank payloads (bool and
   fp32).
12. rank0 front ranking on a bounded worker thread
   (REPLAY_FUSED_FRONT_RANK_BG): the on arm's reports are bit-identical
   to the forced-synchronous arm (world 1, CPU and cuda device arms; and
   at world 2 with 2 visible GPUs, where the worker's sorts overlap real
   collectives); the worker applies the chain FIFO as single writer
   (cross-ratio ordering included), re-raises a task failure at finish(),
   and a submit behind a dead worker raises fast — through the entry
   check, and through the queue.Full health check when the queue is full
   and the only consumer dies mid-backlog — instead of deadlocking the
   bounded queue; unknown knob values raise.
13. REPLAY_FUSED_KERNEL replay-kernel kill-switch: the auto arm on
   CUDA+fp32 engages the triton kernels (echo line) with reports
   bit-identical to the forced-off arm across two chunk-mb settings
   (formula/select/adam sweeps all bit-covered); the CPU device arm
   stays torch under auto; a bogus value raises with the knob named.
   The per-kernel adversarial conformance matrix (per-score, adam,
   select passes, stack canaries, dispatch gates) lives in
   tests/test_replay_kernels.py.
14. --selection-semantics balanced-histogram: allowed end to end on a
   single node at world=2 (the sharded balanced select reconstructs the
   catalog histogram from the shard partials), report inside the
   standard world=2 comparator, threshold distinct from the exact-topk
   arm's (the semantics actually routed the select sites); --nodes and
   zerofill stay rejected by name.
"""
import io
import json
import os
import pickle
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sga import replay as soa
from sga import replay_fused
from test_replay_cycle import reference_cycle

# runners may export REPLAY_S0_PROFILE globally for their GPU arms; the
# suite must stay immune (the s0_profile fragment key carries wall-clock
# counters that break the bitwise A/Bs). Tests that need it set it
# themselves. Same for REPLAY_FUSED_FRONT_SORT: a suite-wide export would
# force test_fused_sort_device_bitwise's dev arm onto the CPU path and fail
# with a misleading "patch did not engage". Covers both the pytest and the
# main() entry paths.
# NOT under the __mp_main__ spelling: an mp.spawn child re-executes this
# module under that name before unpickling the worker, and the world-2
# fd-captured A/Bs need the env set in the parent to survive into the
# child's import-time _S0_PROFILE bind (it is the only route there) —
# deleting it here would silence the S0FRONT echo those tests assert on.
if __name__ != "__mp_main__":
    if "REPLAY_S0_PROFILE" in os.environ:
        del os.environ["REPLAY_S0_PROFILE"]
        replay_fused._S0_PROFILE = False
    if "REPLAY_FUSED_FRONT_SORT" in os.environ:
        del os.environ["REPLAY_FUSED_FRONT_SORT"]
    # same suite-immunity rationale as REPLAY_FUSED_FRONT_SORT above
    if "REPLAY_FUSED_SELECT" in os.environ:
        del os.environ["REPLAY_FUSED_SELECT"]
    if "REPLAY_FUSED_FRONT_RANK_BG" in os.environ:
        del os.environ["REPLAY_FUSED_FRONT_RANK_BG"]
    if "REPLAY_FUSED_KERNEL" in os.environ:
        del os.environ["REPLAY_FUSED_KERNEL"]
    if "REPLAY_FUSED_P1_READ" in os.environ:
        del os.environ["REPLAY_FUSED_P1_READ"]


B1, B2, EPS, WD = 0.9, 0.95, 1e-8, 0.1
BASE_META = {"adam_beta1": B1, "adam_beta2": B2, "adam_eps": EPS,
             "weight_decay": WD}

# sizes chosen so the world=2 split of N=870000 at 435000 cuts INSIDE p1,
# and (with --chunk-mb 1 in _base_args) every param spans multiple score and
# streaming chunks — the chunked write path (shard offset + chunk offset) is
# load-bearing, a single-chunk fixture cannot see an indexing bug there
SIZES = [400_000, 300_000, 170_000]
NAMES = [f"p{i}" for i in range(len(SIZES))]
N_TOTAL = sum(SIZES)


def _make_trace_multi(tmpdir, pre_grads, lrs, clip, t0, absent=None,
                      payload_dtype=torch.float32):
    """Multi-param synthetic trace; ``absent`` maps step-index -> param name
    that is missing from that step's g payload (the zero-fill path).
    ``payload_dtype`` is the stored g dtype (bf16 for --trace-dtype bf16
    screening-copy fixtures). Returns (w0s, m0s, v0s, g0) so from-spec
    references can replay the exact saved branch state."""
    absent = absent or {}
    torch.save({"catalog": {n: {"shape": [s]}
                            for n, s in zip(NAMES, SIZES)}},
               os.path.join(tmpdir, "catalog.pt"))
    rng = torch.Generator().manual_seed(2026)
    w0s = {n: torch.randn(s, generator=rng) for n, s in zip(NAMES, SIZES)}
    m0s = {n: torch.randn(s, generator=rng) * 0.1
           for n, s in zip(NAMES, SIZES)}
    v0s = {n: torch.rand(s, generator=rng) * 0.01 + 1e-4
           for n, s in zip(NAMES, SIZES)}
    for si, pre in enumerate(pre_grads):
        step = t0 + si
        norm = 0.0
        for n in NAMES:
            norm += float((pre[n] ** 2).sum())
        norm = norm ** 0.5
        coef = 1.0
        if clip > 0.0:
            coef = min(1.0, clip / (norm + 1.0e-6))
        payload = {n: (pre[n] * coef).to(payload_dtype)
                   for n in NAMES if absent.get(si) != n}
        meta = {"lr": [lrs[si]], "grad_norm": norm, "clip_grad": clip,
                "adam_beta1": B1, "adam_beta2": B2, "adam_eps": EPS,
                "weight_decay": WD, "step": step}
        torch.save({"payload": payload, "meta": meta},
                   os.path.join(tmpdir, f"step_{step:06d}_g.pt"))
    g0 = {n: torch.zeros(s) for n, s in zip(NAMES, SIZES)}
    state = {"payload": {"w": w0s, "m": m0s, "v": v0s, "g": g0},
             "meta": {**BASE_META, "step": t0}}
    torch.save(state, os.path.join(tmpdir, f"step_{t0:06d}_state.pt"))
    return w0s, m0s, v0s, g0


def _build_fixture(tmpdir, payload_dtype=torch.float32):
    """Clipped 6-step trace (the clip binds every step), 3 params, p2 absent
    at the si=2 boundary and p1 absent at the si=3 masked step. Returns
    everything a from-spec reference needs (the state's g — the
    branch-point selection input — is zeros in this fixture)."""
    torch.manual_seed(11)
    t0, W = 900, 6
    clip = 0.3
    lrs = [5e-5] * W
    pre = [{n: torch.randn(s) * 0.5 for n, s in zip(NAMES, SIZES)}
           for _ in range(W)]
    absent = {2: "p2", 3: "p1"}
    w0s, m0s, v0s, g0s = _make_trace_multi(tmpdir, pre, lrs, clip, t0,
                                           absent=absent,
                                           payload_dtype=payload_dtype)
    return t0, W, pre, lrs, clip, absent, w0s, m0s, v0s, g0s


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return soa.main(argv)


def _base_args(td, cache, t0, extra=(), cycle=("--cycle-len", "2"),
               selection="topk"):
    # ``cycle`` is swappable because the two T spellings are mutually
    # exclusive on the CLI — a --cycle-lens run cannot just append the flag.
    # Legacy exact-oracle tests pin topk centrally; selection=None exercises
    # the CLI's balanced default, and an appended explicit flag also wins.
    selection_args = (() if selection is None else
                      ("--selection-semantics", selection))
    return ["--trace-dir", td, "--branch-point", str(t0), "--window", "6",
            "--horizons", "6", "--ratios", "20,30",
            "--score-names", "abs_g,update", "--replay-mode", "cycle",
            *cycle, "--dense-cache", cache, "--device", "cpu",
            "--chunk-mb", "1", *selection_args, *extra]



def test_scan_dtype_guards():
    """Fused entries and serialized workers reject fp64 before replay work."""
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        for extra in (["--replay-dtype", "fp64"],
                      ["--norm-dtype", "fp64"]):
            try:
                _run(_base_args(td, cache, t0, extra))
            except ValueError as e:
                assert "fp64 is reserved for --self-check" in str(e), e
            else:
                raise AssertionError(f"fp64 scan accepted: {extra}")

        worker_cases = [
            {"replay_dtype": "fp64", "norm_fp32": True},
            {"replay_dtype": "fp32", "norm_fp32": False},
        ]
        for case in worker_cases:
            cfg = {"seed": 1234, "device": "cpu", **case}
            try:
                replay_fused._worker(
                    0, 1, cfg, 0, os.path.join(td, "frag.pt"))
            except ValueError as e:
                assert "fp64 is reserved for --self-check" in str(e), e
            else:
                raise AssertionError(
                    f"serialized worker accepted invalid scan cfg: {case}")
    print("PASS test_scan_dtype_guards")

def _close(a, b, rel):
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom == 0.0 or abs(a - b) / denom <= rel


def _deep_eq_nan(a, b):
    """Deep equality treating NaN == NaN: the fixture's grad_norm is NaN,
    and dict == compares two distinct NaN objects as unequal even when the
    reports are bit-identical."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _deep_eq_nan(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _deep_eq_nan(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return type(a) is type(b) and a == b


def _strip_cycle_echo(rep):
    """Drop the T-set provenance echoes (config cycle_len/cycle_lens, the
    branch fragment's cycle_lens) so reports produced through different
    spellings or T sets compare equal on the numbers."""
    cfg = dict(rep["config"])
    cfg.pop("cycle_len", None)
    cfg.pop("cycle_lens", None)
    brs = {}
    for t0k, br in rep["branch_results"].items():
        br = dict(br)
        br.pop("cycle_lens", None)
        brs[t0k] = br
    return {"config": cfg, "branch_results": brs}


def _ref_distortion(pre, lrs, clip, t0, absent, w0s, m0s, v0s, g0s, score,
                    ratio, T, W, payload_dtype=torch.float32):
    """From-spec distortion {str(h): num/den} for one score x ratio arm on
    the multi-param fixture (params concatenated in catalog order — the
    engine's flat index space). Rebuilds the exact logged gradients and
    meta norms the fixture writer produced (the trace inputs the engine
    and the reference both consume); absent params replay as zero
    gradients; g0s is the branch state's g — the branch-point selection
    input. ``payload_dtype`` mirrors the stored g dtype: the reference
    quantizes the logged gradients exactly as the fixture writer did
    (bf16 payloads reach the engine upcast to fp32)."""
    w0 = torch.cat([w0s[n] for n in NAMES]).float()
    m0 = torch.cat([m0s[n] for n in NAMES]).float()
    v0 = torch.cat([v0s[n] for n in NAMES]).float()
    branch_grad = torch.cat([g0s[n] for n in NAMES]).float()
    logged_flat, norms = [], []
    for si, g in enumerate(pre):
        norm = 0.0
        for n in NAMES:
            norm += float((g[n] ** 2).sum())
        norm = norm ** 0.5
        coef = min(1.0, clip / (norm + 1.0e-6)) if clip > 0.0 else 1.0
        parts = [torch.zeros(SIZES[i]) if absent.get(si) == n
                 else (g[n] * coef).to(payload_dtype).float()
                 for i, n in enumerate(NAMES)]
        logged_flat.append(torch.cat(parts))
        norms.append(norm)
    k = max(1, int(round(ratio / 100.0 * N_TOTAL)))
    dense_ws, sparse_ws = reference_cycle(w0, m0, v0, logged_flat, norms,
                                          lrs, clip, t0, score, k, T,
                                          branch_grad=branch_grad)
    out = {}
    for h in range(1, W + 1):
        den = float((dense_ws[h - 1] - w0).norm())
        num = float((sparse_ws[h - 1] - dense_ws[h - 1]).norm())
        out[str(h)] = None if den == 0.0 else num / den
    return out


def _assert_ref_anchor(report, t0, ref_args, horizons):
    """Every score x ratio distortion in the report matches the from-spec
    reference. The report is sanitized to 8 decimals, so the band is the
    rounding step plus a relative slack for chunked fp32 replay arithmetic."""
    for r in ref_args["ratios"]:
        for s in ref_args["scores"]:
            got = report["branch_results"][str(t0)]["per_ratio"][str(r)] \
                ["scores"][s]["distortion"]
            want = _ref_distortion(*ref_args["fixture"], s, r,
                                   ref_args["T"], ref_args["W"],
                                   payload_dtype=ref_args.get(
                                       "payload_dtype", torch.float32))
            for h in horizons:
                g, w = got[str(h)], want[str(h)]
                assert (g is None) == (w is None), (s, r, h, g, w)
                if g is not None:
                    # fp32 state updates have an absolute quantization floor
                    # when dense and sparse trajectories nearly coincide.
                    assert abs(g - w) <= 5e-6 + 1e-3 * abs(w), \
                        (s, r, h, g, w)


def test_fused_world1_bitwise():
    """Single shard (inline, no comms): a fresh run populates the cache and
    reports dense_cache_hit False, the hit run is bitwise-identical to the
    same run invoked with the deprecated --fused flag (the comparison arm
    is a second cache HIT — the miss->hit roundtrip equality is anchored by
    test_replay_cycle.test_dense_cache_roundtrip), and every distortion
    matches the from-spec reference (reference_cycle is the spec anchor)."""
    with tempfile.TemporaryDirectory() as td:
        (t0, W, pre, lrs, clip, absent, w0s, m0s, v0s, g0s) = \
            _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        ref_args = {
            "fixture": (pre, lrs, clip, t0, absent, w0s, m0s, v0s, g0s),
            "ratios": (20.0, 30.0), "scores": ("abs_g", "update"),
            "T": 2, "W": W,
        }
        miss = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        assert miss["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is False
        hit = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        assert hit["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is True
        # "fused" echoes the CLI flag (not passed on this run);
        # "fused_effective" names the engine actually serving it — cycle
        # always routes to the fused engine
        assert hit["config"]["fused"] is False
        assert hit["config"]["fused_effective"] is True
        assert hit["config"]["fused_shards"] == 1
        # the deprecated --fused flag must be a no-op on an identical run
        noop = _run(_base_args(td, cache, t0,
                               ["--fused", "--fused-shards", "1",
                                "--fused-max-configs", "2"]))
        assert noop["config"]["fused_max_configs"] == 2
        assert noop["config"]["fused"] is True  # CLI echo of --fused
        assert noop["config"]["fused_effective"] is True
        # env-gated: no s0_profile key unless REPLAY_S0_PROFILE=1
        assert "s0_profile" not in hit["branch_results"][str(t0)]
        a = json.dumps(hit["branch_results"], sort_keys=True)
        b = json.dumps(noop["branch_results"], sort_keys=True)
        assert a == b, "--fused (deprecated no-op) changed the report"
        _assert_ref_anchor(hit, t0, ref_args, (str(W),))
    print("PASS test_fused_world1_bitwise")


def _compare_world2(world1, world2, label):
    bc = world1["branch_results"]
    bf = world2["branch_results"]
    assert set(bc) == set(bf)
    for t0, cbr in bc.items():
        fbr = bf[t0]
        # non-sum fields: exact (meta carries a NaN grad_norm for synthetic
        # traces — compare the canonical JSON text, NaN != NaN under ==)
        for key in ("t0", "state_absent", "window_len", "window_steps",
                    "horizon_steps"):
            assert cbr[key] == fbr[key], (t0, key)
        assert json.dumps(cbr["meta"], sort_keys=True) \
            == json.dumps(fbr["meta"], sort_keys=True), (t0, "meta")
        for key in ("per_step", "den_norm_at_horizon", "label_total"):
            assert cbr["replay"][key] == fbr["replay"][key], (t0, key)
        for s, st in cbr["scores"].items():
            fst = fbr["scores"][s]
            for key in ("numel", "min", "max", "fraction_zero"):
                assert st[key] == fst[key], (t0, s, key)
            for key in ("sum", "mean"):
                assert _close(st[key], fst[key], 1e-6), (t0, s, key, label)
        for key in ("numel", "min", "max", "fraction_zero"):
            assert cbr["replay"]["label_stats"][key] \
                == fbr["replay"]["label_stats"][key], (t0, key)
        for key in ("sum", "mean"):
            assert _close(cbr["replay"]["label_stats"][key],
                          fbr["replay"]["label_stats"][key], 1e-6), \
                (t0, key, label)
        for r, prc in cbr["per_ratio"].items():
            prf = fbr["per_ratio"][r]
            assert prc["k"] == prf["k"], (t0, r)
            for s, mc in prc["scores"].items():
                mf = prf["scores"][s]
                assert mc["threshold"] == mf["threshold"], (t0, r, s)
                for key in ("ndcg_at_k", "recall_at_k", "missed_label_mass"):
                    assert mc[key] == mf[key], (t0, r, s, key, label)
                assert _close(mc["missed_score_mass"],
                              mf["missed_score_mass"], 1e-6), (t0, r, s)
                for h, dc in mc["distortion"].items():
                    df = mf["distortion"][h]
                    if dc is None:
                        assert df is None, (t0, r, s, h)
                    else:
                        assert _close(df, dc, 1e-6), \
                            (t0, r, s, h, dc, df, label)


def test_fused_world2_band():
    """Two shards (mp.spawn + gloo): the score vectors themselves are
    elementwise (bitwise identical across shardings); only the fp64 scalar
    SUMS reorder (clip norms, distortion numerators, sum statistics) —
    world=2 against the world=1 run (self-consistency)."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        f1 = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        f2 = _run(_base_args(td, cache, t0,
                             ["--fused-shards", "2",
                              "--fused-max-configs", "2"]))
        assert f2["config"]["fused_shards"] == 2
        _compare_world2(f1, f2, "fp64")
    print("PASS test_fused_world2_band")


def test_fused_world2_norm_fp32():
    """--norm-dtype fp32 on both arms: same quantization, so the world=2
    vs world=1 deviation stays pure-reorder (the fp32-norm effect itself is
    gated by the frozen norm-dtype band, not by this test)."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        f1 = _run(_base_args(td, cache, t0,
                             ["--norm-dtype", "fp32", "--fused-shards", "1"]))
        f2 = _run(_base_args(td, cache, t0,
                             ["--norm-dtype", "fp32", "--fused-shards", "2"]))
        _compare_world2(f1, f2, "fp32-norm")
    print("PASS test_fused_world2_norm_fp32")


def test_fused_cold_cache_world2():
    """Cold cache with no prior run: run_fused_branch populates the dense
    cache parent-side before the shards start (dense_cache_hit False on
    the populating run), so --fused-shards 2 runs directly on a fresh cache
    dir; the populated cache then serves a world=1 run (hit True). The
    world=2 fp32 sum-statistic reordering survives the report's 8-decimal
    sanitize (~4e-6 abs on the update-score sum), so the branch_results go
    through the standard world=2 comparator rather than the world=1
    canonical-JSON equality."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        f2 = _run(_base_args(td, cache, t0, ["--fused-shards", "2"]))
        assert f2["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is False
        f1 = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        assert f1["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is True
        _compare_world2(f1, f2, "cold-cache")
    print("PASS test_fused_cold_cache_world2")


def test_fused_world2_balanced_histogram():
    """Omitting --selection-semantics selects balanced-histogram end to end
    at world=2 on one node: the sharded balanced select reconstructs the online engine's
    full-catalog histogram from the shard partials (per-param subsample
    phase, catalog-wide n_total/n_sampled, int64 histogram all_reduce, no
    world factor), so the whole-crossing-bin threshold is exact across
    shardings and the report stays inside the standard world=2 comparator
    (fp64 scalar sums reorder exactly as in the topk band). The balanced
    threshold (crossing-bin lower edge) must differ from the exact-topk
    threshold on at least one arm — proof the semantics actually routed
    the select sites instead of silently running topk."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        top = _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        b1 = _run(_base_args(td, cache, t0, ["--fused-shards", "1"],
                             selection=None))
        b2 = _run(_base_args(td, cache, t0,
                             ["--fused-shards", "2",
                              "--fused-max-configs", "2"], selection=None))
        assert b1["config"]["selection_semantics"] == "balanced-histogram"
        assert b2["config"]["selection_semantics"] == "balanced-histogram"
        _compare_world2(b1, b2, "balanced")
        tbr = top["branch_results"][str(t0)]["per_ratio"]
        bbr = b1["branch_results"][str(t0)]["per_ratio"]
        assert any(tbr[r]["scores"][s]["threshold"]
                   != bbr[r]["scores"][s]["threshold"]
                   for r in tbr for s in tbr[r]["scores"]), \
            "balanced-histogram arm reproduced exact-topk thresholds"
    print("PASS test_fused_world2_balanced_histogram")


def test_fused_cli_gates():
    """Each invalid combination is rejected with the option named, before
    any replay work."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        base = ["--trace-dir", td, "--branch-point", str(t0),
                "--ratios", "20", "--horizons", "3", "--window", "3",
                "--device", "cpu", "--selection-semantics", "topk"]
        default_balanced = base[:-2]
        cases = [
            # The unconditional balanced default rejects unsupported modes
            # when the caller omits an explicit legacy topk selection.
            (default_balanced,
             "--selection-semantics balanced-histogram requires"),
            (default_balanced + ["--replay-mode", "cycle",
                                 "--dense-cache", cache,
                                 "--nodes", "nodeA.example"],
             "--selection-semantics balanced-histogram requires"),
            # zerofill (the default) has no fused form; --fused is a
            # deprecated no-op only where it ever made sense
            (base + ["--fused"], "--fused is deprecated"),
            (base + ["--replay-mode", "cycle"],
             "--replay-mode cycle requires --dense-cache"),
            (["--trace-dir", "/nonexistent-sga-trace-dir", "--replay-mode",
              "cycle", "--self-check", "900", "901"], "--self-check"),
            (["--trace-dir", "/nonexistent-sga-trace-dir", "--fused",
              "--self-check", "900", "901"],
             "--fused is deprecated"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--fused-shards", "0"],
             "--fused-shards must be >= 1"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--fused-max-configs", "0"],
             "--fused-max-configs must be >= 1"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--fused-max-configs", "-1"],
             "--fused-max-configs must be >= 1"),
            # the state byte budget knob: rejected on non-finite/negative
            # under cycle ...
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--fused-state-budget-mb", "-1"],
             "--fused-state-budget-mb must be a finite value >= 0"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--fused-state-budget-mb", "inf"],
             "--fused-state-budget-mb must be a finite value >= 0"),
            # ... and under zerofill (fused-only knobs are unused there but
            # still validated with the option named)
            (base + ["--fused-state-budget-mb", "-0.5"],
             "--fused-state-budget-mb must be a finite value >= 0"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--score-names", "nope"],
             "not a registered score"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--ratios", "150"],
             "in (0, 100]"),
            # the T-set knob: the two spellings are mutually exclusive (the
            # gate fires under zerofill too — the knob is only echoed there),
            # and the set itself is validated value by value
            (base + ["--cycle-len", "2", "--cycle-lens", "3,5"],
             "--cycle-len and --cycle-lens are mutually exclusive"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--cycle-lens", ""],
             "names no cycle length"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--cycle-lens", "2,0"],
             "must be >= 1"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--cycle-lens", "2,x"],
             "must be integers >= 1"),
            # balanced-histogram semantics: zerofill has no fused select
            (base + ["--selection-semantics", "balanced-histogram"],
             "--selection-semantics balanced-histogram requires"),
            (base + ["--horizons", "1,3"],
             "--horizons must equal the window"),
            (base + ["--horizons", ""],
             "--horizons must equal the window"),
            # ... and --nodes stays rejected under cycle: the sharded
            # balanced select is single-node only (any shard count is fine
            # on ONE node — exercised end to end by
            # test_fused_world2_balanced_histogram)
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--selection-semantics", "balanced-histogram",
                     "--nodes", "nodeA.example"],
             "--selection-semantics balanced-histogram requires"),
        ]
        if torch.cuda.is_available():
            # --device cpu: shards are plain processes with no CUDA bound,
            # so the "exceeds" gate needs a cuda device to fire
            cases += [
                # the rejection must point at CUDA_VISIBLE_DEVICES (shard r
                # always maps to cuda:r)
                (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                         "--device", "cuda:1"],
                 "CUDA_VISIBLE_DEVICES"),
                (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                         "--device", "cuda", "--fused-shards", "99"],
                 "exceeds the visible CUDA devices"),
            ]
        for argv, msg in cases:
            try:
                _run(argv)
            except ValueError as e:
                assert msg in str(e), (msg, str(e))
            else:
                raise AssertionError(f"gate accepted: {argv}")
    print("PASS test_fused_cli_gates")


def test_cycle_lens_mixed_equivalence():
    """Mixed-T groups (--cycle-lens 2,3): every f"{score}@T{t}" arm must be
    bitwise identical to the same (score, ratio) arm in the solo-T run
    (--cycle-len t), at world 1 and world 2 — each arm's state machine
    consumes only its own calendar/states and the qn2/num2 collectives carry
    one independent scalar lane per arm, so no band is needed here."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        # warm the shared dense cache once: the populating run must not sit
        # in any A/B arm
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        for shards in ("1", "2"):
            mixed = _run(_base_args(td, cache, t0,
                                    ["--fused-shards", shards],
                                    cycle=("--cycle-lens", "2,3")))
            solos = {t: _run(_base_args(td, cache, t0,
                                        ["--fused-shards", shards],
                                        cycle=("--cycle-len", t)))
                     for t in ("2", "3")}
            assert mixed["branch_results"][str(t0)]["cycle_lens"] == [2, 3]
            mixed_br = mixed["branch_results"][str(t0)]
            for t in ("2", "3"):
                s = _strip_cycle_echo(solos[t])
                # rebuild the mixed per_ratio with only this T's arms,
                # renamed to the solo keys (no mutation of the shared
                # nested dicts — the other T's arms stay intact)
                new_per_ratio = {}
                for r, pr in mixed_br["per_ratio"].items():
                    solo_keys = set(
                        s["branch_results"][str(t0)]["per_ratio"][r]
                        ["scores"])
                    picked = {}
                    for k in sorted(pr["scores"]):
                        base_k, _, tag = k.partition("@T")
                        if tag == t:
                            picked[base_k] = pr["scores"][k]
                        else:
                            assert tag in ("2", "3"), k
                    assert set(picked) == solo_keys, (shards, t, r)
                    new_per_ratio[r] = {"k": pr["k"], "scores": picked}
                m = _strip_cycle_echo(mixed)
                m["branch_results"][str(t0)]["per_ratio"] = new_per_ratio
                assert _deep_eq_nan(m, s), \
                    f"mixed-T arm diverged from solo T={t} (shards={shards})"
    print("PASS test_cycle_lens_mixed_equivalence")


def test_cycle_lens_single_compat():
    """--cycle-lens 2 and --cycle-len 2 produce the same report once the
    echo keys are stripped (the set spelling echoes config cycle_len None
    and per_ratio keeps the bare score keys) — single-T products are
    unaffected by the schema extension (world 1 and 2)."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        for shards in ("1", "2"):
            via_set = _run(_base_args(td, cache, t0,
                                      ["--fused-shards", shards],
                                      cycle=("--cycle-lens", "2")))
            via_scalar = _run(_base_args(td, cache, t0,
                                         ["--fused-shards", shards]))
            assert via_set["config"]["cycle_lens"] == [2]
            assert via_set["config"]["cycle_len"] is None
            assert via_scalar["config"]["cycle_len"] == 2
            br = via_set["branch_results"][str(t0)]
            assert br["cycle_lens"] == [2]
            assert "abs_g" in br["per_ratio"]["20.0"]["scores"]
            assert _deep_eq_nan(_strip_cycle_echo(via_set),
                                _strip_cycle_echo(via_scalar)), shards
    print("PASS test_cycle_lens_single_compat")


def _env_comms_worker(rank, world, port, per_node, expected_sum):
    """One simulated multi-node shard (rank = the global process-group
    rank, local shard index = rank % per_node). The launcher contract is
    env-per-process, so each worker sets its own SGA_GLOBAL_RANK; the
    master addr/port must arrive by mp.spawn os.environ inheritance from
    the parent. port is deliberately 0 — unusable — so the rendezvous can
    only succeed through the inherited env path."""
    glob, local = rank, rank % per_node
    os.environ["SGA_GLOBAL_RANK"] = str(glob)
    c = replay_fused.Comms(local, world, port)
    assert (c.rank, c.global_rank, c.world) == (local, glob, world), \
        (c.rank, c.global_rank, c.world, local, glob, world)
    assert c.sum_f64([float(glob) + 0.5]) == [expected_sum]
    parts = c.all_gather_pad(torch.tensor([float(glob)]))
    assert [int(p[0].item()) for p in parts] == list(range(world))
    if c.distributed:
        torch.distributed.destroy_process_group()


def _with_rendezvous_env(fn):
    """Run fn() with SGA_MASTER_ADDR/SGA_MASTER_PORT set to a free local
    endpoint, restoring any SGA_* rendezvous vars afterwards."""
    keys = ("SGA_MASTER_ADDR", "SGA_MASTER_PORT", "SGA_GLOBAL_RANK")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["SGA_MASTER_ADDR"] = "127.0.0.1"
        os.environ["SGA_MASTER_PORT"] = str(replay_fused._free_port())
        fn()
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def test_comms_env_multinode_world2():
    """SGA_MASTER_ADDR + SGA_MASTER_PORT set in the parent and inherited by
    the mp.spawn children, per-worker SGA_GLOBAL_RANK: a real 2-rank gloo
    init over the master endpoint (the port argument is 0, so only the env
    path can rendezvous) plus one sum_f64 and one all_gather_pad
    roundtrip."""
    _with_rendezvous_env(lambda: mp.spawn(
        _env_comms_worker, args=(2, 0, 2, 2.0), nprocs=2, join=True))
    print("PASS test_comms_env_multinode_world2")


def test_comms_env_global_local_split():
    """Two simulated nodes x 2 local shards (world=4): global rank != local
    rank on the second node, init_process_group uses the global rank (all
    4 meet), collectives return in global process-group rank order (the
    order shard_select's tie-fill indexes), and self.rank stays the local
    shard index."""
    _with_rendezvous_env(lambda: mp.spawn(
        _env_comms_worker, args=(4, 0, 2, 8.0), nprocs=4, join=True))
    print("PASS test_comms_env_global_local_split")


def test_comms_env_errors():
    """The two env misconfigurations raise loudly, naming the orphaned or
    missing variable, before any process group is touched (world=1, so no
    init would happen anyway)."""
    keys = ("SGA_MASTER_ADDR", "SGA_MASTER_PORT", "SGA_GLOBAL_RANK")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["SGA_GLOBAL_RANK"] = "1"
        try:
            replay_fused.Comms(0, 1, 0)
        except ValueError as e:
            assert "SGA_MASTER_ADDR" in str(e), str(e)
        else:
            raise AssertionError(
                "SGA_GLOBAL_RANK accepted without SGA_MASTER_ADDR")
        del os.environ["SGA_GLOBAL_RANK"]
        os.environ["SGA_MASTER_ADDR"] = "127.0.0.1"
        try:
            replay_fused.Comms(0, 1, 0)
        except ValueError as e:
            assert "SGA_MASTER_PORT" in str(e), str(e)
        else:
            raise AssertionError(
                "SGA_MASTER_ADDR accepted without SGA_MASTER_PORT")
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
    print("PASS test_comms_env_errors")


def test_fused_cuda_world1():
    """CUDA smoke: world=1 on cuda (dense via dense_replay_dev) matches the
    from-spec reference within the sanitize/report band (fp64 dev engine,
    no horizon-diff quantization)."""
    if not torch.cuda.is_available():
        print("SKIP test_fused_cuda_world1 (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        (t0, W, pre, lrs, clip, absent, w0s, m0s, v0s, g0s) = \
            _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        report = _run(_base_args(td, cache, t0,
                                 ["--device", "cuda", "--fused-shards", "1"]))
        ref_args = {
            "fixture": (pre, lrs, clip, t0, absent, w0s, m0s, v0s, g0s),
            "ratios": (20.0, 30.0), "scores": ("abs_g", "update"),
            "T": 2, "W": W,
        }
        _assert_ref_anchor(report, t0, ref_args, (str(W),))
    print("PASS test_fused_cuda_world1")


def test_fused_bf16_trace_world1():
    """--trace-dtype bf16 (screening copies): the prefetcher's unconditional
    slice-first leg (slice the mmap'd bf16 payload, THEN widen) exercised
    three ways — (a) a loud rejection when the bf16 fixture is met with the
    fp32 declaration, so this test cannot silently degrade into an fp32
    duplicate; (b) the from-spec reference with the identical bf16->fp32
    upcast mirrored in the logged gradients; (c) a bitwise A/B against a
    fused run on the fp32 twin trace (the same payloads pre-upcast at
    rest, i.e. exactly what the bf16 load hands the engine)."""
    with tempfile.TemporaryDirectory() as td:
        bf16_dir = os.path.join(td, "bf16")
        os.makedirs(bf16_dir)
        (t0, W, pre, lrs, clip, absent, w0s, m0s, v0s, g0s) = \
            _build_fixture(bf16_dir, payload_dtype=torch.bfloat16)
        cache = os.path.join(bf16_dir, "dense_cache")
        try:
            _run(_base_args(bf16_dir, cache, t0, ["--fused-shards", "1"]))
        except ValueError as e:
            assert "--trace-dtype" in str(e), str(e)
        else:
            raise AssertionError(
                "bf16 payload accepted with --trace-dtype fp32")
        report = _run(_base_args(bf16_dir, cache, t0,
                                 ["--fused-shards", "1",
                                  "--trace-dtype", "bf16"]))
        assert report["config"]["trace_dtype"] == "bf16"
        assert report["branch_results"][str(t0)]["replay"][
            "trace_dtype"] == "bf16"
        ref_args = {
            "fixture": (pre, lrs, clip, t0, absent, w0s, m0s, v0s, g0s),
            "ratios": (20.0, 30.0), "scores": ("abs_g", "update"),
            "T": 2, "W": W, "payload_dtype": torch.bfloat16,
        }
        _assert_ref_anchor(report, t0, ref_args, (str(W),))
        # fp32 twin: identical payloads pre-upcast at rest — every replay
        # path must consume exactly the same fp32 inputs either way, so
        # the two fused runs are bit-identical (the dense caches populate
        # separately per trace_dtype ident; both arms are populate runs,
        # so dense_cache_hit matches)
        twin = os.path.join(td, "twin_fp32")
        os.makedirs(twin)
        for f in os.listdir(bf16_dir):
            src_path = os.path.join(bf16_dir, f)
            if f.endswith("_g.pt"):
                d = torch.load(src_path, weights_only=False)
                for k, t in d["payload"].items():
                    d["payload"][k] = t.float()
                torch.save(d, os.path.join(twin, f))
            elif f == "catalog.pt" or f.endswith("_state.pt"):
                torch.save(torch.load(src_path, weights_only=False),
                           os.path.join(twin, f))
        twin_rep = _run(_base_args(twin, os.path.join(twin, "dense_cache"),
                                   t0, ["--fused-shards", "1"]))
        for rep in (report, twin_rep):
            rep["branch_results"][str(t0)]["replay"].pop("trace_dtype")
        assert _deep_eq_nan(report["branch_results"],
                            twin_rep["branch_results"]), \
            "the bf16 load path diverged from its fp32 twin"
    print("PASS test_fused_bf16_trace_world1")


def test_g_residency_bitwise():
    """--fused-g-residency uploads each step's g slices once and shares the
    resident copy across configs and passes 1/2/3 (boundary steps missing a
    g payload read a cached device zeros buffer). The arithmetic per element
    is unchanged, so the whole report must be bit-identical to the default
    per-chunk-upload path (world 1 and 2, warm cache)."""
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        # warm the cache first: the populating run reports
        # dense_cache_hit False and must not sit in the A/B
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        for shards in ("1", "2"):
            base = _run(_base_args(td, cache, t0,
                                   ["--fused-shards", shards]))
            res = _run(_base_args(td, cache, t0,
                                  ["--fused-shards", shards,
                                   "--fused-g-residency"]))
            assert base["config"]["fused_g_residency"] is False
            assert res["config"]["fused_g_residency"] is True
            assert res["branch_results"][str(t0)]["replay"][
                "g_residency"] is True
            assert base["branch_results"][str(t0)]["replay"][
                "g_residency"] is False
            # strip the provenance echoes; everything else must be identical
            res["config"].pop("fused_g_residency")
            base["config"].pop("fused_g_residency")
            res["branch_results"][str(t0)]["replay"].pop("g_residency")
            base["branch_results"][str(t0)]["replay"].pop("g_residency")
            assert _deep_eq_nan(base, res), \
                f"residency moved a bit (shards={shards})"
    print("PASS test_g_residency_bitwise")


def test_g_residency_cuda_pinned():
    """On CUDA the residency path also pins the prefetcher's host buffers
    (async H2D). Numbers stay bit-identical to the unpinned default path
    (world 1 wherever a GPU is visible; the world-2 leg needs 2 GPUs)."""
    if not torch.cuda.is_available():
        print("SKIP test_g_residency_cuda_pinned (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        # the dense-cache ident is world-independent: one warm run serves
        # both shard legs (the populating run reports dense_cache_hit False
        # and must not sit in the A/B)
        _run(_base_args(td, cache, t0, ["--device", "cuda",
                                        "--fused-shards", "1"]))
        # pin is world-independent (pin=(residency and device != cpu)), so
        # world 1 already exercises it; world 2 only where 2 GPUs are up
        shard_legs = ["1"]
        if torch.cuda.device_count() >= 2:
            shard_legs.append("2")
        else:
            print("NOTE test_g_residency_cuda_pinned: world-2 leg skipped "
                  "(needs 2 visible GPUs)")
        for shards in shard_legs:
            common = ["--device", "cuda", "--fused-shards", shards]
            base = _run(_base_args(td, cache, t0, common))
            res = _run(_base_args(td, cache, t0,
                                  [*common, "--fused-g-residency"]))
            assert res["config"]["fused_g_residency"] is True
            for rep in (base, res):
                rep["config"].pop("fused_g_residency")
                rep["branch_results"][str(t0)]["replay"].pop("g_residency")
            assert _deep_eq_nan(base, res), f"shards={shards}"
    print("PASS test_g_residency_cuda_pinned")


def test_gprefetcher_pinned_buffers():
    """GPrefetcher(pin=True) — the construction the CUDA residency path
    makes (pin=(residency and device != cpu)) — hands the consumer pinned
    fp32 host buffers, so the H2D can go out with non_blocking. Needs a
    CUDA context (pin_memory), skipped without one."""
    if not torch.cuda.is_available():
        print("SKIP test_gprefetcher_pinned_buffers (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, W = _build_fixture(td)[0:2]
        catalog, names, offsets, N, state_files, g_files = \
            soa.discover_trace(td)
        sp = replay_fused.ShardParams(names, catalog,
                                      {**BASE_META, "step": t0}, 0, N)
        window_steps = soa.window_steps_for(g_files, t0, W)
        pre = replay_fused.GPrefetcher(g_files, window_steps, sp, False,
                                       pin=True)
        # one generator only: each __iter__() call restarts q.get() from
        # scratch, but the producer emits exactly len(window_steps) items —
        # a second iterator would wait forever for items already consumed
        it = iter(pre)
        _gstep, _gmeta, local = next(it)
        assert set(local) == set(names)
        for t in local.values():
            assert t.is_pinned()
            assert t.dtype == torch.float32
        for _ in it:  # drain: let the producer thread finish cleanly
            pass
    print("PASS test_gprefetcher_pinned_buffers")


class _ForeignPayload:
    pass


def test_pread_trace_index():
    """_pt_index reads a torch.save zip's index without touching payload
    records: stubs carry key/dtype/offset/shape/stride, record data offsets
    come from the local headers (alignment padding via the extra field),
    and slicing bytes at (offset + p_off) * elsize reproduces torch.load's
    slice bitwise across dtypes (bf16/fp32/int64, 1-D and 2-D) — including
    a sliced/view payload with a nonzero storage offset, so the
    (st.offset + p_off) arithmetic the reader performs is pinned against
    torch.load ground truth. A non-contiguous payload is caught by the
    contiguous() guard, and a foreign class in the pickle fails loudly
    (fixed-vocabulary contract)."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "toy.pt")
        payload = {
            "a_bf16_2d": torch.randn(64, 96).to(torch.bfloat16),
            "b_fp32_1d": torch.randn(1000),
            "c_int64_2d": torch.arange(24).reshape(4, 6),
            "d_fp32_view": torch.randn(80)[17:],
        }
        meta = {"step": 7, "lr": [1e-5], "clip_grad": 0.5}
        torch.save({"payload": payload, "meta": meta}, path)
        got_meta, stubs, offs = replay_fused._pt_index(path)
        assert got_meta == meta
        truth = torch.load(path, map_location="cpu", weights_only=False)
        assert set(stubs) == set(payload)
        assert stubs["d_fp32_view"].offset == 17, \
            "fixture lost its nonzero-storage-offset payload"
        for name, st in stubs.items():
            t = truth["payload"][name]
            assert st.dtype == t.dtype, name
            assert st.size == tuple(t.shape), name
            assert st.offset == t.storage_offset() and st.contiguous(), name
            elsize = t.element_size()
            data_off, rec_bytes = offs[st.key]
            assert rec_bytes >= (st.offset + st.numel()) * elsize, name
            flat = t.reshape(-1)
            for p_off, n in ((0, flat.numel()), (3, 17),
                             (flat.numel() - 5, 5)):
                with open(path, "rb") as f:
                    f.seek(data_off + (st.offset + p_off) * elsize)
                    buf = f.read(n * elsize)
                back = torch.frombuffer(bytearray(buf), dtype=st.dtype)
                assert torch.equal(back, flat[p_off:p_off + n]), \
                    (name, p_off, n)
        nc_path = os.path.join(td, "nc.pt")
        torch.save({"payload": {"nc": torch.randn(8, 9).t()},
                    "meta": {}}, nc_path)
        _m, nc_stubs, _o = replay_fused._pt_index(nc_path)
        assert not nc_stubs["nc"].contiguous()
        fk_path = os.path.join(td, "fk.pt")
        torch.save({"payload": {"x": torch.randn(4)}, "meta": {},
                    "foreign": _ForeignPayload()}, fk_path)
        try:
            replay_fused._pt_index(fk_path)
        except pickle.UnpicklingError as e:
            assert "unexpected" in str(e), str(e)
        else:
            raise AssertionError("foreign class accepted by index unpickler")
    print("PASS test_pread_trace_index")


def test_pread_prefetch_bitwise():
    """GPrefetcher pread arm == mmap arm bitwise: both payload dtypes
    (fp32 clone path, bf16 upcast path), full range and a mid split
    (offsets + partial slices), every step's meta and every entry's fp32
    values torch.equal; on CUDA also the pinned construction. The
    REPLAY_FUSED_P1_READ knob echoes its arm and a bogus value raises with
    the knob named."""
    for payload_dtype, upcast in ((torch.float32, False),
                                  (torch.bfloat16, True)):
        with tempfile.TemporaryDirectory() as td:
            t0, W = _build_fixture(td, payload_dtype=payload_dtype)[0:2]
            catalog, names, offsets, N, state_files, g_files = \
                soa.discover_trace(td)
            window_steps = soa.window_steps_for(g_files, t0, W)
            for lo, hi in ((0, N), (N // 3, 2 * N // 3)):
                sp = replay_fused.ShardParams(
                    names, catalog, {**BASE_META, "step": t0}, lo, hi)
                arms = {}
                for mode in ("mmap", "pread"):
                    pre = replay_fused.GPrefetcher(
                        g_files, window_steps, sp, upcast, read_mode=mode)
                    arms[mode] = [(gs, gm, {k: v.clone()
                                            for k, v in local.items()})
                                  for gs, gm, local in pre]
                assert len(arms["mmap"]) == len(arms["pread"]) == W
                for (gs_a, gm_a, la), (gs_b, gm_b, lb) in zip(
                        arms["mmap"], arms["pread"]):
                    assert (gs_a, gm_a) == (gs_b, gm_b)
                    assert set(la) == set(lb)
                    for k in la:
                        assert la[k].dtype == lb[k].dtype
                        assert torch.equal(la[k], lb[k]), \
                            (payload_dtype, lo, k)
            if torch.cuda.is_available():
                sp = replay_fused.ShardParams(
                    names, catalog, {**BASE_META, "step": t0}, 0, N)
                pinned = {}
                for mode in ("mmap", "pread"):
                    pre = replay_fused.GPrefetcher(
                        g_files, window_steps, sp, upcast, pin=True,
                        read_mode=mode)
                    pinned[mode] = [{k: v for k, v in local.items()}
                                    for _gs, _gm, local in pre]
                for la, lb in zip(pinned["mmap"], pinned["pread"]):
                    for k in la:
                        assert lb[k].is_pinned()
                        assert torch.equal(la[k], lb[k]), ("pin", k)
    old_env = os.environ.get("REPLAY_FUSED_P1_READ")
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            reps = []
            for arm, echo in ((None, "g read: pread"),
                              ("mmap", "g read: mmap")):
                if arm is None:
                    os.environ.pop("REPLAY_FUSED_P1_READ", None)
                else:
                    os.environ["REPLAY_FUSED_P1_READ"] = arm
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rep = soa.main(_base_args(td, cache, t0,
                                              ["--fused-shards", "1"]))
                assert echo in buf.getvalue(), (echo, buf.getvalue()[-500:])
                reps.append(rep)
            # the read arm must not move a report bit
            for rep in reps:
                rep["branch_results"][str(t0)]["replay"].pop(
                    "dense_cache_hit")
            assert _deep_eq_nan(reps[0], reps[1]), \
                "the pread read arm moved a report bit"
            os.environ["REPLAY_FUSED_P1_READ"] = "bogus"
            try:
                soa.main(_base_args(td, cache, t0, ["--fused-shards", "1"]))
            except ValueError as e:
                assert "REPLAY_FUSED_P1_READ" in str(e), str(e)
            else:
                raise AssertionError("bogus REPLAY_FUSED_P1_READ accepted")
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_P1_READ", None)
        else:
            os.environ["REPLAY_FUSED_P1_READ"] = old_env
    print("PASS test_pread_prefetch_bitwise")


def test_gprefetcher_index_cache_shared_across_groups():
    """The zip-index cache is worker-lifetime, not per prefetcher: two
    sequential pread-arm GPrefetcher instances sharing one index_cache dict
    (the _worker pattern — one dict handed to every config group's
    prefetcher) parse each g file's index exactly once and yield
    bitwise-equal items; an instance without a shared cache re-parses per
    instance (the documented fallback). The counting wrapper around
    _pt_index makes a silent never-hitting cache visible."""
    with tempfile.TemporaryDirectory() as td:
        t0, W = _build_fixture(td)[0:2]
        catalog, names, offsets, N, state_files, g_files = \
            soa.discover_trace(td)
        sp = replay_fused.ShardParams(names, catalog,
                                      {**BASE_META, "step": t0}, 0, N)
        window_steps = soa.window_steps_for(g_files, t0, W)
        calls = []
        real_index = replay_fused._pt_index

        def counting_index(path):
            calls.append(path)
            return real_index(path)

        replay_fused._pt_index = counting_index
        try:
            index_cache = {}
            runs = []
            for _ in range(2):
                pre = replay_fused.GPrefetcher(
                    g_files, window_steps, sp, False,
                    index_cache=index_cache, read_mode="pread")
                runs.append([(gs, dict(gm),
                              {k: v.clone() for k, v in local.items()})
                             for gs, gm, local in pre])
            assert calls == [g_files[s] for s in window_steps], calls
            calls.clear()
            pre = replay_fused.GPrefetcher(g_files, window_steps, sp, False,
                                           read_mode="pread")
            fallback = [(gs, dict(gm),
                         {k: v.clone() for k, v in local.items()})
                        for gs, gm, local in pre]
            assert len(calls) == W, calls
        finally:
            replay_fused._pt_index = real_index
        for a, b in zip(runs[0], runs[1]):
            assert a[0] == b[0] and a[1] == b[1]
            assert set(a[2]) == set(b[2])
            for k in a[2]:
                assert torch.equal(a[2][k], b[2][k]), k
        for run in runs + [fallback]:
            assert [gs for gs, _gm, _l in run] == window_steps
        for a, b in zip(runs[0], fallback):
            for k in a[2]:
                assert torch.equal(a[2][k], b[2][k]), k
    print("PASS test_gprefetcher_index_cache_shared_across_groups")


def test_pread_pipeline_lifecycle():
    """Two-stage pread pipeline lifecycle: at depth=1 the reader -> _raw_q
    -> upcast -> q handoff backpressures yet delivers every step in order
    without deadlock, and after a clean drain BOTH producer threads are
    gone — the success path must post the sentinel that releases the
    upcast thread (pre-fix, one parked daemon thread leaked per config
    group; the consumer could never see the difference, so this is pinned
    by thread identity, not by values)."""
    with tempfile.TemporaryDirectory() as td:
        t0, W = _build_fixture(td)[0:2]
        catalog, names, offsets, N, state_files, g_files = \
            soa.discover_trace(td)
        sp = replay_fused.ShardParams(names, catalog,
                                      {**BASE_META, "step": t0}, 0, N)
        window_steps = soa.window_steps_for(g_files, t0, W)
        made = []
        real_thread = threading.Thread

        def rec_thread(*a, **k):
            th = real_thread(*a, **k)
            made.append(th)
            return th

        threading.Thread = rec_thread
        try:
            pre = replay_fused.GPrefetcher(g_files, window_steps, sp, False,
                                           depth=1, read_mode="pread")
        finally:
            threading.Thread = real_thread
        assert len(made) == 2, "pread arm must run exactly two threads"
        items = [(gs, gm, {k: v.clone() for k, v in local.items()})
                 for gs, gm, local in pre]
        assert [gs for gs, _gm, _l in items] == window_steps, \
            "depth=1 reordered the two-stage pipeline"
        for th in made:
            th.join(timeout=5.0)
            assert not th.is_alive(), \
                "pread producer thread parked past a clean drain " \
                "(success-path sentinel missing)"
    print("PASS test_pread_pipeline_lifecycle")


def test_pread_failure_injection():
    """Pinned by injection: any producer-thread exception surfaces at the
    consumer as RuntimeError('g prefetch failed') with the original
    exception chained, never a silent hang. Three injections: a short
    pread where the payload record outruns the file (OSError from the
    r==0 guard), a corrupt zip local-header signature (ValueError from
    the index parser), and an upcast-thread failure — all take the same
    path."""
    with tempfile.TemporaryDirectory() as td:
        t0, W = _build_fixture(td)[0:2]
        catalog, names, offsets, N, state_files, g_files = \
            soa.discover_trace(td)
        sp = replay_fused.ShardParams(names, catalog,
                                      {**BASE_META, "step": t0}, 0, N)
        window_steps = soa.window_steps_for(g_files, t0, W)

        def _expect_consumer_failure(cause_type, cause_msg, build):
            pre = build()
            try:
                next(iter(pre))
            except RuntimeError as e:
                assert "g prefetch failed" in str(e), str(e)
                assert isinstance(e.__cause__, cause_type), e.__cause__
                assert cause_msg in str(e.__cause__), \
                    (cause_msg, str(e.__cause__))
            else:
                raise AssertionError(
                    f"producer failure '{cause_msg}' did not surface at "
                    "the consumer")

        # (a) short pread: preadv hits EOF inside the record (the shape a
        # truncated trace file takes once the central directory survives)
        real_preadv = os.preadv

        def eof_preadv(fd, bufs, offset):
            return 0

        os.preadv = eof_preadv
        try:
            _expect_consumer_failure(
                OSError, "short pread",
                lambda: replay_fused.GPrefetcher(
                    g_files, window_steps, sp, False, read_mode="pread"))
        finally:
            os.preadv = real_preadv

        # (b) corrupt zip local header: clobber one storage record's
        # signature on disk (data.pkl stays intact so the failure lands in
        # the _pt_index header arithmetic, not zipfile); restore right
        # after, otherwise (c) would see the reader fail on the same file
        # before the upcast injection fires
        path = g_files[window_steps[0]]
        with zipfile.ZipFile(path) as zf:
            hdr_off = next(inf.header_offset for inf in zf.infolist()
                           if len(inf.filename.split("/")) == 3
                           and inf.filename.split("/")[-2] == "data")
        with open(path, "r+b") as f:
            f.seek(hdr_off)
            orig_sig = f.read(4)
            f.seek(hdr_off)
            f.write(b"\x00\x00\x00\x00")
        try:
            _expect_consumer_failure(
                ValueError, "bad local header sig",
                lambda: replay_fused.GPrefetcher(
                    g_files, window_steps, sp, False, read_mode="pread"))
        finally:
            with open(path, "r+b") as f:
                f.seek(hdr_off)
                f.write(orig_sig)

        # (c) upcast-thread failure: same contract, same chaining; depth
        # >= W+1 keeps the reader off a full _raw_q so both threads finish
        real_frombuffer = torch.frombuffer

        def boom(*a, **k):
            raise ValueError("simulated upcast failure")

        torch.frombuffer = boom
        try:
            _expect_consumer_failure(
                ValueError, "simulated upcast failure",
                lambda: replay_fused.GPrefetcher(
                    g_files, window_steps, sp, False, depth=W + 1,
                    read_mode="pread"))
        finally:
            torch.frombuffer = real_frombuffer
    print("PASS test_pread_failure_injection")


def test_pread_guards_through_prefetcher():
    """The pread arm's payload guards raise THROUGH GPrefetcher (the
    consumer sees RuntimeError('g prefetch failed') from the guard's
    ValueError, not just the bare predicate): a non-contiguous payload, a
    catalog record shorter than its storage needs, and a stale catalog
    whose slice extent outruns the payload record (silent wrong-bytes must
    stay loud). Both arms — the default pread and the mmap emergency
    fallback — reject bf16 payloads with the upcast off under the same
    ValueError contract."""
    with tempfile.TemporaryDirectory() as td:
        t0, W = _build_fixture(td)[0:2]
        catalog, names, offsets, N, state_files, g_files = \
            soa.discover_trace(td)
        sp = replay_fused.ShardParams(names, catalog,
                                      {**BASE_META, "step": t0}, 0, N)
        window_steps = soa.window_steps_for(g_files, t0, W)

        def _expect_guard(msg, build):
            pre = build()
            try:
                next(iter(pre))
            except RuntimeError as e:
                assert "g prefetch failed" in str(e), str(e)
                assert isinstance(e.__cause__, ValueError), e.__cause__
                assert msg in str(e.__cause__), (msg, str(e.__cause__))
            else:
                raise AssertionError(f"guard '{msg}' did not fire")

        # (a) non-contiguous payload (transposed storage layout) — the
        # contiguous() predicate pinned by test_pread_trace_index, now
        # through the prefetcher itself
        with tempfile.TemporaryDirectory() as td2:
            torch.save({"catalog": {"nc": {"shape": [72]}}},
                       os.path.join(td2, "catalog.pt"))
            torch.save({"payload": {"nc": torch.randn(8, 9).t()},
                        "meta": {"step": t0}},
                       os.path.join(td2, f"step_{t0:06d}_g.pt"))
            torch.save({"payload": {}, "meta": {"step": t0}},
                       os.path.join(td2, f"step_{t0:06d}_state.pt"))
            cat2, names2, _off2, n2, _sf2, gf2 = soa.discover_trace(td2)
            sp_nc = replay_fused.ShardParams(
                names2, cat2, {**BASE_META, "step": t0}, 0, n2)
            _expect_guard(
                "non-contiguous",
                lambda: replay_fused.GPrefetcher(
                    gf2, soa.window_steps_for(gf2, t0, 1), sp_nc, False,
                    read_mode="pread"))

        # (b) record-too-short guard: the catalog claims one element less
        # per storage record than the payload stores (a corrupt/stale
        # catalog — honest torch.save output can never produce this)
        real_index = replay_fused._pt_index

        def short_record(path):
            meta, stubs, offs = real_index(path)
            for key in offs:
                data_off, rec_bytes = offs[key]
                offs[key] = (data_off, rec_bytes - 4)
            return meta, stubs, offs

        replay_fused._pt_index = short_record
        try:
            _expect_guard(
                "too short",
                lambda: replay_fused.GPrefetcher(
                    g_files, window_steps, sp, False, read_mode="pread"))
        finally:
            replay_fused._pt_index = real_index

        # (c) slice-extent guard: an entry whose [p_off, p_off + n) window
        # outruns the payload record — in-repo ShardParams construction
        # cannot produce this, so the stale catalog is built by hand and
        # the guard is the loud backstop (pread would otherwise read the
        # adjacent record's bytes where the mmap arm silently clamps)
        sp_wide = replay_fused.ShardParams(names, catalog,
                                           {**BASE_META, "step": t0}, 0, N)
        sp_wide.entries[0]["n"] += 1
        _expect_guard(
            "outruns storage record",
            lambda: replay_fused.GPrefetcher(
                g_files, window_steps, sp_wide, False, read_mode="pread"))

        with tempfile.TemporaryDirectory() as td3:
            _build_fixture(td3, payload_dtype=torch.bfloat16)
            cat3, names3, _off3, n3, _sf3, gf3 = soa.discover_trace(td3)
            sp_bf = replay_fused.ShardParams(
                names3, cat3, {**BASE_META, "step": t0}, 0, n3)
            ws3 = soa.window_steps_for(gf3, t0, W)
            for mode in ("mmap", "pread"):
                pre = replay_fused.GPrefetcher(gf3, ws3, sp_bf, False,
                                               read_mode=mode)
                try:
                    next(iter(pre))
                except RuntimeError as e:
                    assert "g prefetch failed" in str(e), (mode, str(e))
                    assert isinstance(e.__cause__, ValueError), \
                        (mode, e.__cause__)
                    assert "--trace-dtype" in str(e.__cause__), \
                        (mode, str(e.__cause__))
                else:
                    raise AssertionError(
                        f"{mode} arm accepted bf16 with upcast off")
    print("PASS test_pread_guards_through_prefetcher")


def test_dev_reduce_bitwise():
    """--fused-dev-reduce accumulates the pass-1 clip norms and pass-3
    horizon numerators in per-config fp64 device scalars fetched once per
    pass. The addends and their order are unchanged, so the whole report
    must be bit-identical to the per-chunk host-sync path (world 1 and 2,
    warm cache, fp32 norm; on CUDA also combined with residency)."""
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        for shards in ("1", "2"):
            base = _run(_base_args(td, cache, t0,
                                   ["--fused-shards", shards]))
            red = _run(_base_args(td, cache, t0,
                                  ["--fused-shards", shards,
                                   "--fused-dev-reduce"]))
            assert base["config"]["fused_dev_reduce"] is False
            assert red["config"]["fused_dev_reduce"] is True
            for rep in (base, red):
                rep["config"].pop("fused_dev_reduce")
                frag = rep["branch_results"][str(t0)]
                frag["replay"].pop("dev_reduce")
            assert _deep_eq_nan(base, red), \
                f"dev_reduce moved a bit (shards={shards})"
    print("PASS test_dev_reduce_bitwise")


def test_dev_reduce_cuda():
    """CUDA leg: dev-reduce alone and stacked with g residency (the
    intended production combo) both stay bit-identical AND echo the flags
    (world 1 wherever a GPU is visible; the world-2 leg needs 2 GPUs)."""
    if not torch.cuda.is_available():
        print("SKIP test_dev_reduce_cuda (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--device", "cuda",
                                        "--fused-shards", "1"]))
        shard_legs = ["1"]
        if torch.cuda.device_count() >= 2:
            shard_legs.append("2")
        else:
            print("NOTE test_dev_reduce_cuda: world-2 leg skipped (needs "
                  "2 visible GPUs)")
        for shards in shard_legs:
            common = ["--device", "cuda", "--fused-shards", shards]
            base = _run(_base_args(td, cache, t0, common))
            red = _run(_base_args(td, cache, t0,
                                  [*common, "--fused-dev-reduce"]))
            combo = _run(_base_args(td, cache, t0,
                                    [*common, "--fused-dev-reduce",
                                     "--fused-g-residency"]))
            # the flags must do more than parse: the single-flag run
            # reports dev_reduce with residency off, the combo both
            assert red["config"]["fused_dev_reduce"] is True
            assert red["config"]["fused_g_residency"] is False
            assert red["branch_results"][str(t0)]["replay"][
                "dev_reduce"] is True
            assert red["branch_results"][str(t0)]["replay"][
                "g_residency"] is False
            assert combo["config"]["fused_dev_reduce"] is True
            assert combo["config"]["fused_g_residency"] is True
            assert combo["branch_results"][str(t0)]["replay"][
                "dev_reduce"] is True
            assert combo["branch_results"][str(t0)]["replay"][
                "g_residency"] is True
            for rep in (base, red, combo):
                rep["config"].pop("fused_dev_reduce")
                rep["config"].pop("fused_g_residency")
                frag = rep["branch_results"][str(t0)]
                frag["replay"].pop("dev_reduce")
                frag["replay"].pop("g_residency")
            assert _deep_eq_nan(base, red), f"shards={shards}"
            assert _deep_eq_nan(base, combo), f"shards={shards}"
    print("PASS test_dev_reduce_cuda")


def _adv_sort_vectors():
    """Non-negative finite sort keys engineered against stable-sort tie
    rules: all-equal, +0.0 runs interleaved with duplicated positives, and
    a duplicate-heavy random draw (the fused engine's score/label vectors
    are exactly this shape — non-negative, finite, tie-rich)."""
    v_tie = torch.zeros(4096)
    v_tie[::7] = 1.5
    v_tie[3::7] = 1.5
    v_tie[1::11] = 0.25
    g = torch.Generator().manual_seed(1234)
    return {
        "all_equal": torch.zeros(4096),
        "tie_blocks": v_tie,
        "dup_random": torch.randint(0, 64, (16384,), generator=g).float()
                      / 8.0,
    }


def test_stable_order_desc_adversarial():
    """The device-permutation fast path must return the SAME permutation as
    the CPU path on adversarial keys (the whole engine's byte-identity
    rests on stable sort being backend-independent for finite non-negative
    keys) — including when the VRAM guard is forced to fall back."""
    if not torch.cuda.is_available():
        print("SKIP test_stable_order_desc_adversarial (no CUDA)")
        return
    for name, v in _adv_sort_vectors().items():
        cpu_order = replay_fused._stable_order_desc(v, None)
        gpu_order = replay_fused._stable_order_desc(v, "cuda:0")
        assert torch.equal(cpu_order, gpu_order), \
            f"permutation diverged on {name}"
        # guard forced to refuse → loud CPU fallback, still the same order
        real_get = torch.cuda.mem_get_info
        torch.cuda.mem_get_info = lambda *a, **k: (0, 1)
        try:
            fb_order = replay_fused._stable_order_desc(v, "cuda:0")
        finally:
            torch.cuda.mem_get_info = real_get
        assert torch.equal(fb_order, cpu_order), \
            f"forced fallback diverged on {name}"
    print("PASS test_stable_order_desc_adversarial")


def test_ranking_idcg_device_bitwise():
    """_fused_ranking/_idcg_of_parts with adversarial keys: the device only
    produces the permutation/sorted values, every reduction stays on the
    callers' CPU expression — so the returned metric scalars must be
    bitwise-equal to the pure-CPU call."""
    if not torch.cuda.is_available():
        print("SKIP test_ranking_idcg_device_bitwise (no CUDA)")
        return
    for i, (name, lb) in enumerate(_adv_sort_vectors().items()):
        n = lb.numel()
        # the engine's call sites pass k == sc_all.numel() (the gathered
        # selection IS the top-k), so wts and the gathered labels align
        k = n
        g = torch.Generator().manual_seed(100 + i)
        sc = torch.randint(0, 32, (n,), generator=g).float() / 4.0
        lsel = torch.rand(n, generator=g) < 0.3
        parts = [lb[: n // 2], lb[n // 2:]]
        idcg_cpu = replay_fused._idcg_of_parts(parts)
        idcg_dev = replay_fused._idcg_of_parts(parts, dev="cuda:0")
        assert idcg_cpu == idcg_dev, f"idcg moved a bit on {name}"
        m_cpu = replay_fused._fused_ranking(
            sc, lb, [lsel], k, float(sc.sum()), float(lb.sum()), idcg_cpu)
        m_dev = replay_fused._fused_ranking(
            sc, lb, [lsel], k, float(sc.sum()), float(lb.sum()), idcg_cpu,
            dev="cuda:0")
        assert m_cpu == m_dev, f"ranking moved a bit on {name}"
    # boundary documentation: LABELS may carry -0.0 (finite, asserted) —
    # radix sort groups -0.0/+0.0 into different key classes where CPU
    # stability interleaves them, but labels only feed VALUES (idcg mul-sum
    # is immune to the sign of zero; the ranking permutation is keyed on sc,
    # which comes from .abs() and can never be -0.0), so the scalars stay
    # bitwise. This is the ONLY -0.0 tolerance — never feed -0.0 as sc.
    n = 8192
    g = torch.Generator().manual_seed(99)
    lb = torch.randint(0, 16, (n,), generator=g).float() / 4.0
    lb[::5] = -0.0
    sc = torch.randint(0, 32, (n,), generator=g).float() / 4.0
    lsel = torch.rand(n, generator=g) < 0.3
    k = n
    parts = [lb[: n // 2], lb[n // 2:]]
    idcg_cpu = replay_fused._idcg_of_parts(parts)
    idcg_dev = replay_fused._idcg_of_parts(parts, dev="cuda:0")
    assert idcg_cpu == idcg_dev, "idcg moved a bit on label_neg_zero"
    m_cpu = replay_fused._fused_ranking(
        sc, lb, [lsel], k, float(sc.sum()), float(lb.sum()), idcg_cpu)
    m_dev = replay_fused._fused_ranking(
        sc, lb, [lsel], k, float(sc.sum()), float(lb.sum()), idcg_cpu,
        dev="cuda:0")
    assert m_cpu == m_dev, "ranking moved a bit on label_neg_zero"
    print("PASS test_ranking_idcg_device_bitwise")


def test_fused_sort_device_bitwise():
    """End-to-end on cuda: forcing the ranking/idcg sorts onto the CPU
    fallback (the guard patched to refuse) must leave the whole fused
    report bit-identical to the device-permutation run — the device
    computes permutations only. The counting wrapper asserts the patch
    actually engaged (a spawned worker would bypass it and pass
    vacuously)."""
    if not torch.cuda.is_available():
        print("SKIP test_fused_sort_device_bitwise (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        common = ["--device", "cuda", "--fused-shards", "1"]
        base = _run(_base_args(td, cache, t0, common))
        calls = {"n": 0}

        def refusing_guard(numel, dev):
            calls["n"] += 1
            return False

        real_guard = replay_fused._device_ok_for_sort
        replay_fused._device_ok_for_sort = refusing_guard
        try:
            fb = _run(_base_args(td, cache, t0, common))
        finally:
            replay_fused._device_ok_for_sort = real_guard
        assert calls["n"] > 0, "guard never consulted — patch did not engage"
        for rep in (base, fb):
            rep["branch_results"][str(t0)]["replay"].pop("dense_cache_hit")
        assert _deep_eq_nan(base, fb), \
            "the device-permutation fast path moved a report bit"
    print("PASS test_fused_sort_device_bitwise")


def test_sort_oom_fallback_loud_cpu():
    """The TOCTOU completion of the loud-fallback contract: a CUDA OOM
    raised at the device sort itself (the guard already passed — another
    process grabbed the VRAM in between) must fall back to the bit-identical
    CPU path with a loud one-line print, not abort the run. torch.argsort /
    torch.sort are patched to raise only for CUDA inputs, so the CPU
    fallback exercise is real; restored in finally."""
    if not torch.cuda.is_available():
        print("SKIP test_sort_oom_fallback_loud_cpu (no CUDA)")
        return
    real_argsort, real_sort = torch.argsort, torch.sort
    calls = []

    def oom_argsort(t, *a, **k):
        if t.is_cuda:
            calls.append("dev_argsort")
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        calls.append("cpu_argsort")
        return real_argsort(t, *a, **k)

    def oom_sort(t, *a, **k):
        if t.is_cuda:
            calls.append("dev_sort")
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        calls.append("cpu_sort")
        return real_sort(t, *a, **k)

    v = _adv_sort_vectors()["tie_blocks"]
    expected_order = replay_fused._stable_order_desc(v, None)
    expected_idcg = replay_fused._idcg_of_parts([v], dev=None)
    try:
        torch.argsort = oom_argsort
        torch.sort = oom_sort
        buf = io.StringIO()
        with redirect_stdout(buf):
            fb_order = replay_fused._stable_order_desc(v, "cuda:0")
            fb_idcg = replay_fused._idcg_of_parts([v], dev="cuda:0")
    finally:
        torch.argsort, torch.sort = real_argsort, real_sort
    assert calls.count("dev_argsort") == 1 and \
        calls.count("cpu_argsort") == 1, calls
    assert calls.count("dev_sort") == 1 and calls.count("cpu_sort") == 1, \
        calls
    out = buf.getvalue()
    assert out.count("hit CUDA OOM") == 2, out[-500:]
    assert "CPU fallback (bit-identical)" in out, out[-500:]
    assert torch.equal(fb_order, expected_order), \
        "OOM fallback changed the ranking permutation"
    assert fb_idcg == expected_idcg, "OOM fallback moved the idcg bit"
    print("PASS test_sort_oom_fallback_loud_cpu")


def test_sort_guard_empty_cache_before_measure():
    """Pins the guard's call order: empty_cache before mem_get_info (the
    rationale is _device_ok_for_sort's: device-wide free-mem accounting
    counts this process's cached-but-free blocks as used). Recorded by
    monkeypatching both (restored in finally)."""
    if not torch.cuda.is_available():
        print("SKIP test_sort_guard_empty_cache_before_measure (no CUDA)")
        return
    calls = []
    real_empty = torch.cuda.empty_cache
    real_get = torch.cuda.mem_get_info

    def rec_empty_cache():
        calls.append("empty_cache")

    def rec_mem_get_info(dev=None):
        calls.append("mem_get_info")
        return real_get(dev)

    try:
        torch.cuda.empty_cache = rec_empty_cache
        torch.cuda.mem_get_info = rec_mem_get_info
        replay_fused._device_ok_for_sort(1, "cuda:0")
    finally:
        torch.cuda.empty_cache = real_empty
        torch.cuda.mem_get_info = real_get
    assert calls == ["empty_cache", "mem_get_info"], calls
    print("PASS test_sort_guard_empty_cache_before_measure")


def test_front_sort_env_killswitch():
    """REPLAY_FUSED_FRONT_SORT=cpu forces the bit-identical CPU sort path on
    a cuda run: the S0FRONT echo must read front_sort=cpu (dev arm:
    front_sort=dev) and the two arms' reports must be byte-identical — the
    small-field A/B harness drives its arms through this public knob."""
    if not torch.cuda.is_available():
        print("SKIP test_front_sort_env_killswitch (no CUDA)")
        return
    old_env = os.environ.get("REPLAY_FUSED_FRONT_SORT")
    old_gate = replay_fused._S0_PROFILE
    replay_fused._S0_PROFILE = True
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            common = ["--device", "cuda", "--fused-shards", "1"]
            bufs = []
            reps = []
            for arm in ("auto", "cpu"):
                os.environ["REPLAY_FUSED_FRONT_SORT"] = arm
                buf = io.StringIO()
                with redirect_stdout(buf):
                    reps.append(soa.main(_base_args(td, cache, t0, common)))
                bufs.append(buf.getvalue())
            assert "front_sort=dev" in bufs[0], bufs[0][-500:]
            assert "front_sort=cpu" in bufs[1], bufs[1][-500:]
            for rep in reps:
                rep["branch_results"][str(t0)]["replay"].pop(
                    "dense_cache_hit")
                rep["branch_results"][str(t0)].pop("s0_profile", None)
            assert _deep_eq_nan(reps[0], reps[1]), \
                "the front-sort kill-switch moved a report bit"
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_FRONT_SORT", None)
        else:
            os.environ["REPLAY_FUSED_FRONT_SORT"] = old_env
        replay_fused._S0_PROFILE = old_gate
    print("PASS test_front_sort_env_killswitch")


def test_front_rank_bg_env_killswitch():
    """REPLAY_FUSED_FRONT_RANK_BG=off forces the synchronous front ranking
    path: the on arm's reports must be bit-identical to the off arm's
    (same arithmetic, same application order, only the executing thread
    differs) and the S0FRONT echo must read rank_bg=on|off. Runs the
    world-1 CPU device pair always and the cuda pair when a GPU is
    present (the worker's set_device/sort path engages there); a bogus
    knob value must raise with the knob named."""
    old_env = os.environ.get("REPLAY_FUSED_FRONT_RANK_BG")
    old_gate = replay_fused._S0_PROFILE
    replay_fused._S0_PROFILE = True
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            device_pairs = [("cpu", ["--device", "cpu"])]
            if torch.cuda.is_available():
                device_pairs.append(("cuda", ["--device", "cuda"]))
            for dev_name, dev_args in device_pairs:
                bufs, reps = [], []
                for arm in ("auto", "off"):
                    os.environ["REPLAY_FUSED_FRONT_RANK_BG"] = arm
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        reps.append(soa.main(_base_args(
                            td, cache, t0,
                            [*dev_args, "--fused-shards", "1"])))
                    bufs.append(buf.getvalue())
                assert "rank_bg=on" in bufs[0], (dev_name, bufs[0][-500:])
                assert "rank_bg=off" in bufs[1], (dev_name, bufs[1][-500:])
                for rep in reps:
                    rep["branch_results"][str(t0)]["replay"].pop(
                        "dense_cache_hit")
                    rep["branch_results"][str(t0)].pop("s0_profile", None)
                assert _deep_eq_nan(reps[0], reps[1]), \
                    f"the rank-background arm moved a report bit ({dev_name})"
            os.environ["REPLAY_FUSED_FRONT_RANK_BG"] = "1"
            try:
                soa.main(_base_args(td, cache, t0,
                                    ["--device", "cpu",
                                     "--fused-shards", "1"]))
            except ValueError as e:
                assert "REPLAY_FUSED_FRONT_RANK_BG" in str(e), str(e)
            else:
                raise AssertionError("bogus rank_bg value accepted")
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_FRONT_RANK_BG", None)
        else:
            os.environ["REPLAY_FUSED_FRONT_RANK_BG"] = old_env
        replay_fused._S0_PROFILE = old_gate
    print("PASS test_front_rank_bg_env_killswitch")


def test_replay_kernels_killswitch():
    """REPLAY_FUSED_KERNEL kill-switch: on CUDA+fp32 the auto arm engages
    the triton kernels (echo line) and its report is bit-identical to the
    forced-off arm's — the formula/select/adam sweeps are all covered by
    the conformance matrix in tests/test_replay_kernels.py, this leg pins
    the end-to-end wiring including the chunk-boundary invariance (two
    chunk-mb settings). The CPU device arm stays torch under auto; a
    bogus value raises with the knob named."""
    old_env = os.environ.get("REPLAY_FUSED_KERNEL")
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            if torch.cuda.is_available():
                for chunk in ("1", "3"):
                    bufs, reps = [], []
                    for arm in ("auto", "off"):
                        os.environ["REPLAY_FUSED_KERNEL"] = arm
                        buf = io.StringIO()
                        with redirect_stdout(buf):
                            reps.append(soa.main(_base_args(
                                td, cache, t0,
                                ["--device", "cuda", "--fused-shards", "1",
                                 "--replay-dtype", "fp32",
                                 "--chunk-mb", chunk])))
                        bufs.append(buf.getvalue())
                    assert "replay kernels: triton" in bufs[0], \
                        ("auto arm", chunk, bufs[0][-500:])
                    assert "replay kernels: torch" in bufs[1], \
                        ("off arm", chunk, bufs[1][-500:])
                    for rep in reps:
                        rep["branch_results"][str(t0)]["replay"].pop(
                            "dense_cache_hit")
                    assert _deep_eq_nan(reps[0], reps[1]), \
                        f"the kernel arm moved a report bit (chunk {chunk})"
            # the CPU device arm stays torch even under auto
            os.environ["REPLAY_FUSED_KERNEL"] = "auto"
            buf = io.StringIO()
            with redirect_stdout(buf):
                soa.main(_base_args(td, cache, t0,
                                    ["--device", "cpu",
                                     "--fused-shards", "1"]))
            assert "replay kernels: torch" in buf.getvalue(), \
                buf.getvalue()[-500:]
            os.environ["REPLAY_FUSED_KERNEL"] = "bogus"
            try:
                soa.main(_base_args(td, cache, t0,
                                    ["--device", "cpu",
                                     "--fused-shards", "1"]))
            except ValueError as e:
                assert "REPLAY_FUSED_KERNEL" in str(e), str(e)
            else:
                raise AssertionError("bogus REPLAY_FUSED_KERNEL accepted")
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_KERNEL", None)
        else:
            os.environ["REPLAY_FUSED_KERNEL"] = old_env
    print("PASS test_replay_kernels_killswitch")


def test_front_rank_bg_world2():
    """World-2 e2e A/B across the rank-background arm: the genuinely new
    hazard — worker sort/reduction activity overlapping the main thread's
    collectives and allocator traffic — exists only at world >= 2, so the
    on/off report equality is pinned there too. The shard echoes (captured
    at fd level) pin rank_bg=on|off engagement in both arms."""
    if torch.cuda.device_count() < 2:
        print("SKIP test_front_rank_bg_world2 (needs 2 visible GPUs)")
        return
    old_env = os.environ.get("REPLAY_FUSED_FRONT_RANK_BG")
    old_s0_env = os.environ.get("REPLAY_S0_PROFILE")
    old_gate = replay_fused._S0_PROFILE
    # the env var (not just the module flag): the spawned shards bind
    # _S0_PROFILE at their own import, and rank 0's S0FRONT print is the
    # engagement evidence
    os.environ["REPLAY_S0_PROFILE"] = "1"
    replay_fused._S0_PROFILE = True
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            common = ["--device", "cuda", "--fused-shards", "2"]
            outs = []
            for arm in ("auto", "off"):
                os.environ["REPLAY_FUSED_FRONT_RANK_BG"] = arm
                outs.append(_run_fd_captured(
                    _base_args(td, cache, t0, common)))
            assert "rank_bg=on" in outs[0][1], outs[0][1][-500:]
            assert "rank_bg=off" in outs[1][1], outs[1][1][-500:]
            reps = [rep for rep, _txt in outs]
            for rep in reps:
                rep["branch_results"][str(t0)]["replay"].pop(
                    "dense_cache_hit")
                rep["branch_results"][str(t0)].pop("s0_profile", None)
            assert _deep_eq_nan(reps[0], reps[1]), \
                "the rank-background arm moved a report bit (world 2)"
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_FRONT_RANK_BG", None)
        else:
            os.environ["REPLAY_FUSED_FRONT_RANK_BG"] = old_env
        if old_s0_env is None:
            os.environ.pop("REPLAY_S0_PROFILE", None)
        else:
            os.environ["REPLAY_S0_PROFILE"] = old_s0_env
        replay_fused._S0_PROFILE = old_gate
    print("PASS test_front_rank_bg_world2")


def test_front_rank_bg_worker_unit():
    """_FrontRankWorker applies the ranking chain in FIFO order as the
    single writer of idcg/rankmet: tiny-tensor results match the
    synchronous calls exactly across two ratios x two optimizer families
    (the cross-ratio FIFO through the shared idcg map AND the
    (score, optimizer) stats / per-optimizer label-total plumbing are
    pinned in the unit); a task exception re-raises at finish() (never a
    silent drain); a submit behind a dead worker raises fast through the
    entry check, and a submit backpressured on a full queue raises fast
    through the queue.Full health check once the only consumer dies
    mid-backlog — neither leg may deadlock the bounded queue."""
    gen = torch.Generator().manual_seed(7)
    ks_map = {20: 5, 30: 3}
    idcg_parts = {20: [torch.rand(3, generator=gen),
                       torch.rand(2, generator=gen)],
                  30: [torch.rand(4, generator=gen)]}
    sc_parts = {20: [torch.rand(3, generator=gen),
                     torch.rand(2, generator=gen)],
                30: [torch.rand(2, generator=gen),
                     torch.rand(1, generator=gen)]}
    lb_parts = {20: [torch.rand(3, generator=gen),
                     torch.rand(2, generator=gen)],
                30: [torch.rand(2, generator=gen),
                     torch.rand(1, generator=gen)]}
    ls_parts = {20: [torch.zeros(3, dtype=torch.bool),
                     torch.ones(2, dtype=torch.bool)],
                30: [torch.ones(2, dtype=torch.bool),
                     torch.zeros(1, dtype=torch.bool)]}
    # per-(score, optimizer) stats and per-optimizer label totals — the
    # axis-aware plumbing the worker must keep apart
    stats = {("abs_g", "adamw"): {"sum": 100.0},
             ("abs_g", "adams"): {"sum": 40.0}}
    label_total = {"adamw": 50.0, "adams": 8.0}
    worker = replay_fused._FrontRankWorker(None, ks_map, stats, label_total)
    worker.start()
    for opt in ("adamw", "adams"):
        for r in (20, 30):
            worker.submit(("idcg", opt, r, idcg_parts[r]))
            worker.submit(("rank", "abs_g", opt, r, sc_parts[r],
                           lb_parts[r], ls_parts[r]))
    rankmet = worker.finish()
    exp_met = {}
    for opt in ("adamw", "adams"):
        for r in (20, 30):
            exp_idcg = replay_fused._idcg_of_parts(idcg_parts[r])
            exp_met[("abs_g", r, opt)] = replay_fused._fused_ranking(
                torch.cat(sc_parts[r]), torch.cat(lb_parts[r]), ls_parts[r],
                ks_map[r], stats[("abs_g", opt)]["sum"], label_total[opt],
                exp_idcg)
    assert set(rankmet) == set(exp_met), rankmet.keys()
    for key, want in exp_met.items():
        got = rankmet[key]
        assert all(got[f] == want[f] for f in want), (key, got, want)
    # two optimizer families on one score must land distinct metrics (the
    # sums/label totals differ) — a keyed-by-score regression would fold
    # them onto one entry
    for r in (20, 30):
        assert rankmet[("abs_g", r, "adamw")] != rankmet[("abs_g", r,
                                                          "adams")]

    bad = replay_fused._FrontRankWorker(None, {20: ks_map[20]},
                                        {("abs_g", "adamw"):
                                         {"sum": 100.0}}, {"adamw": 50.0})
    bad.start()
    bad.submit(("rank", "abs_g", "adamw", 20, [torch.rand(2), "junk"],
                [torch.rand(2)], [torch.ones(2, dtype=torch.bool)]))
    try:
        bad.finish()
    except RuntimeError as e:
        assert "front rank worker failed" in str(e), str(e)
    else:
        raise AssertionError("worker failure did not re-raise at finish")
    _tq = time.perf_counter()
    try:
        bad.submit(("idcg", "adamw", 20, idcg_parts[20]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("submit behind a dead worker did not raise")
    assert time.perf_counter() - _tq < 1.0, "submit blocked behind a dead worker"

    # queue-full leg: the first task sleeps inside a patched
    # _fused_ranking before raising, so the submits behind it fill the
    # bounded queue while the only consumer is still alive; the submit
    # that lands on backpressure must discover the death through its
    # health check and raise bounded, not block forever on a queue
    # nothing drains anymore
    real_ranking = replay_fused._fused_ranking

    def slow_poison(*_a, **_kw):
        time.sleep(0.5)
        raise ValueError("poison")

    replay_fused._fused_ranking = slow_poison
    try:
        full = replay_fused._FrontRankWorker(None, ks_map, stats,
                                             label_total)
        full.start()
        for _i in range(5):
            full.submit(("rank", "abs_g", "adamw", 20, sc_parts[20],
                         lb_parts[20], ls_parts[20]))
        _tq = time.perf_counter()
        try:
            full.submit(("rank", "abs_g", "adamw", 20, sc_parts[20],
                         lb_parts[20], ls_parts[20]))
        except RuntimeError as e:
            assert "front rank worker failed" in str(e), str(e)
        else:
            raise AssertionError(
                "a submit backpressured behind a dying worker did not "
                "raise")
        assert time.perf_counter() - _tq < 5.0, \
            "the queue-full health check did not bound the submit"
    finally:
        replay_fused._fused_ranking = real_ranking
    print("PASS test_front_rank_bg_worker_unit")


def _adversarial_scores(n, gen):
    """Non-negative finite fp32 vectors engineered with heavy tie structure
    in the threshold bin."""
    plateau = torch.ones(n)
    plateau[: int(0.4 * n)] = 2.0
    zh = torch.zeros(n)
    zh[: max(8, n // 1000)] = torch.rand(max(8, n // 1000), generator=gen) \
        + 0.5
    return {
        "all-equal": torch.full((n,), 1.0),
        "boundary-plateau": plateau,
        "quantized": (torch.rand(n, generator=gen)
                      .mul(1000).round().div(1000)),
        "zero-heavy": zh,
        "uniform": torch.rand(n, generator=gen) + 1e-6,
    }


def test_shard_select_device_bitwise():
    """shard_select on a cuda resident score is bit-identical to the cpu
    execution (mask bitwise + threshold exact) across adversarial tie
    structures and k values, including tie layers filled in flat-index
    order (the torch.nonzero ordering contract)."""
    if not torch.cuda.is_available():
        print("SKIP test_shard_select_device_bitwise (no CUDA)")
        return
    gen = torch.Generator().manual_seed(20260912)
    comms = replay_fused.Comms(rank=0, world=1, port=29613)
    n = 200_000
    for cname, score in _adversarial_scores(n, gen).items():
        n_pos = max(8, n // 1000)
        ks = sorted({1, n // 100, 3 * n // 10, n - 1, n})
        if cname == "zero-heavy":
            ks = sorted(set(ks) | {max(1, n_pos // 2), min(n, n_pos * 4)})
        for k in ks:
            m_cpu, t_cpu = replay_fused.shard_select(score, k, comms)
            m_dev, t_dev = replay_fused.shard_select(
                score.to("cuda:0"), k, comms)
            assert torch.equal(m_cpu, m_dev.cpu()), (cname, k, "mask")
            assert t_cpu == t_dev, (cname, k, t_cpu, t_dev)
    print("PASS test_shard_select_device_bitwise")


def test_all_gather_pad_dev_world1():
    """world=1 contract of the device-payload gather: this pins ONLY the
    ``not self.distributed`` identity shortcut (both arms below take the
    early return — no transport runs; the nccl movement itself is covered
    by the world-2 test). Values exactly preserved, bool and fp32."""
    if not torch.cuda.is_available():
        print("SKIP test_all_gather_pad_dev_world1 (no CUDA)")
        return
    comms = replay_fused.Comms(rank=0, world=1, port=29614)
    t = torch.rand(100_000, device="cuda:0")
    (got,) = comms.all_gather_pad_dev(t, "nccl")
    assert torch.equal(got, t.cpu())
    b = (torch.rand(100_000, device="cuda:0") < 0.5)
    (gotb,) = comms.all_gather_pad_dev(b, "gloo")
    assert gotb.dtype == torch.bool and torch.equal(gotb, b.cpu())
    print("PASS test_all_gather_pad_dev_world1")


def _agp_dev_world2_worker(rank, world, port):
    """One rank of the world-2 all_gather_pad_dev check: uneven per-rank
    payload lengths (rank 1 also contributes a ZERO-length payload), bool
    and fp32 payloads, every nccl result asserted bitwise against the gloo
    all_gather_pad ground truth on the same values."""
    torch.cuda.set_device(rank)  # production mapping: shard r -> cuda:r
    comms = replay_fused.Comms(rank, world, port)
    try:
        assert comms.distributed and comms.world == 2
        # the same unanimous vote the engine's front block runs must pick
        # nccl on a single node; the first nccl gather then builds the lazy
        # group — a gloo fallback would leave it None and fail below
        assert comms.vote_gather_backend() == "nccl"
        gen = torch.Generator().manual_seed(31 + rank)
        dev = f"cuda:{rank}"
        for dtype in (torch.float32, torch.bool):
            # uneven per-rank sizes; the last slot is zero-length on rank 1
            # (padding must never leak into any rank's result)
            for ln in (4096 + rank * 137, 1, 0 if rank == 1 else 511):
                t = (torch.rand(ln, generator=gen) < 0.5).to(dev) \
                    if dtype is torch.bool \
                    else torch.rand(ln, generator=gen).to(dev)
                got = comms.all_gather_pad_dev(t, "nccl")
                assert comms._nccl_group is not None
                want = comms.all_gather_pad(t.cpu())
                assert [p.numel() for p in got] \
                    == [p.numel() for p in want], (dtype, ln)
                for r in range(world):
                    assert got[r].dtype == dtype \
                        and got[r].device.type == "cpu", (dtype, ln, r)
                    assert torch.equal(got[r], want[r]), (dtype, ln, r)
    finally:
        if comms.distributed:
            torch.distributed.destroy_process_group()


def test_all_gather_pad_dev_world2():
    """world=2 NCCL transport of the device-payload gather: uneven per-rank
    payload sizes including a zero-length payload on one rank, bool and
    fp32, asserted bitwise against the all_gather_pad ground truth on the
    same values (transport is exact movement); the lazy nccl group assert
    pins that the nccl branch — not the world==1 shortcut or the gloo
    fallback — moved the bytes."""
    if torch.cuda.device_count() < 2:
        print("SKIP test_all_gather_pad_dev_world2 (needs 2 visible GPUs)")
        return
    mp.spawn(_agp_dev_world2_worker,
             args=(2, replay_fused._free_port()), nprocs=2, join=True)
    print("PASS test_all_gather_pad_dev_world2")


def test_fused_select_device_bitwise():
    """End-to-end on cuda: forcing the front select block onto the CPU
    fallback (the guard patched to refuse) must leave the whole fused
    report bit-identical to the device-select run — the device computes
    selections/gathers only, every downstream reduction keeps its original
    CPU expression. The counting wrapper asserts the patch engaged."""
    if not torch.cuda.is_available():
        print("SKIP test_fused_select_device_bitwise (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        common = ["--device", "cuda", "--fused-shards", "1"]
        base = _run(_base_args(td, cache, t0, common))
        calls = {"n": 0}

        def refusing_guard(numel, k_max, world, n_scores, dev):
            calls["n"] += 1
            return False

        real_guard = replay_fused._device_ok_for_select
        replay_fused._device_ok_for_select = refusing_guard
        try:
            fb = _run(_base_args(td, cache, t0, common))
        finally:
            replay_fused._device_ok_for_select = real_guard
        assert calls["n"] > 0, "guard never consulted — patch did not engage"
        for rep in (base, fb):
            rep["branch_results"][str(t0)]["replay"].pop("dense_cache_hit")
        assert _deep_eq_nan(base, fb), \
            "the device front-select fast path moved a report bit"
    print("PASS test_fused_select_device_bitwise")


def test_select_env_killswitch():
    """REPLAY_FUSED_SELECT=cpu forces the bit-identical CPU front-select
    path on a cuda run: the S0FRONT echo must read front_sel=cpu (dev arm:
    front_sel=dev) and the two arms' reports must be byte-identical."""
    if not torch.cuda.is_available():
        print("SKIP test_select_env_killswitch (no CUDA)")
        return
    old_env = os.environ.get("REPLAY_FUSED_SELECT")
    old_gate = replay_fused._S0_PROFILE
    replay_fused._S0_PROFILE = True
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            common = ["--device", "cuda", "--fused-shards", "1"]
            bufs = []
            reps = []
            for arm in ("auto", "cpu"):
                os.environ["REPLAY_FUSED_SELECT"] = arm
                buf = io.StringIO()
                with redirect_stdout(buf):
                    reps.append(soa.main(_base_args(td, cache, t0, common)))
                bufs.append(buf.getvalue())
            assert "front_sel=dev" in bufs[0], bufs[0][-500:]
            assert "front_sel=cpu" in bufs[1], bufs[1][-500:]
            for rep in reps:
                rep["branch_results"][str(t0)]["replay"].pop(
                    "dense_cache_hit")
                rep["branch_results"][str(t0)].pop("s0_profile", None)
            assert _deep_eq_nan(reps[0], reps[1]), \
                "the select kill-switch moved a report bit"
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_SELECT", None)
        else:
            os.environ["REPLAY_FUSED_SELECT"] = old_env
        replay_fused._S0_PROFILE = old_gate
    print("PASS test_select_env_killswitch")


def _run_fd_captured(argv):
    """soa.main with the real stdout fd captured to a string, returned as
    (report, text): mp.spawn shards inherit fd 1, so the parent-level
    redirect_stdout in _run never sees their S0FRONT/S0PROFILE prints —
    capture at the fd the children actually write to."""
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+")
    try:
        os.dup2(tmp.fileno(), 1)
        rep = soa.main(argv)
        sys.stdout.flush()
        tmp.seek(0)
        return rep, tmp.read()
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        tmp.close()


def test_select_nccl_world2():
    """World-2 e2e A/B across the front-select transport: the default arm
    (auto -> device select + nccl gather on a single node) and the
    REPLAY_FUSED_SELECT=cpu arm must produce bit-identical reports. The
    shard echoes (captured at fd level) pin both arms' engagement:
    front_sel=dev with gather=nccl vs front_sel=cpu."""
    if torch.cuda.device_count() < 2:
        print("SKIP test_select_nccl_world2 (needs 2 visible GPUs)")
        return
    old_env = os.environ.get("REPLAY_FUSED_SELECT")
    old_s0_env = os.environ.get("REPLAY_S0_PROFILE")
    old_gate = replay_fused._S0_PROFILE
    # the env var (not just the module flag): the spawned shards bind
    # _S0_PROFILE at their own import, and rank 0's S0FRONT print is the
    # engagement evidence
    os.environ["REPLAY_S0_PROFILE"] = "1"
    replay_fused._S0_PROFILE = True
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, *_ = _build_fixture(td)
            cache = os.path.join(td, "dense_cache")
            common = ["--device", "cuda", "--fused-shards", "2"]
            outs = []
            for arm in ("auto", "cpu"):
                os.environ["REPLAY_FUSED_SELECT"] = arm
                outs.append(_run_fd_captured(
                    _base_args(td, cache, t0, common)))
            assert "front_sel=dev" in outs[0][1], outs[0][1][-500:]
            assert "gather=nccl" in outs[0][1], outs[0][1][-500:]
            assert "front_sel=cpu" in outs[1][1], outs[1][1][-500:]
            reps = [rep for rep, _txt in outs]
            for rep in reps:
                rep["branch_results"][str(t0)]["replay"].pop(
                    "dense_cache_hit")
                rep["branch_results"][str(t0)].pop("s0_profile", None)
            assert _deep_eq_nan(reps[0], reps[1]), \
                "the nccl gather fast path moved a report bit"
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_FUSED_SELECT", None)
        else:
            os.environ["REPLAY_FUSED_SELECT"] = old_env
        if old_s0_env is None:
            os.environ.pop("REPLAY_S0_PROFILE", None)
        else:
            os.environ["REPLAY_S0_PROFILE"] = old_s0_env
        replay_fused._S0_PROFILE = old_gate
    print("PASS test_select_nccl_world2")


def _budget_mb(shards, cap, replay_dtype="fp32"):
    """The --fused-state-budget-mb value that derives group size == cap,
    recomputed from the fixture spec constants (sum(SIZES)) via the
    engine's public accounting: {w,m,v,res} x itemsize over the widest
    shard (ceil(N/world)). Dividing by 2**20 is exact in binary floats
    (power-of-two divisor), so the engine's floor lands exactly on cap."""
    basis = (sum(SIZES) + shards - 1) // shards
    itemsize = 8 if replay_dtype == "fp64" else 4
    return basis * 4 * itemsize * cap / 2 ** 20


def test_state_budget_bitwise():
    """--fused-state-budget-mb derives the fused group size from the
    resident-state byte budget instead of --fused-max-configs (which it
    overrides). The report must be bit-identical to the count-knob run at
    the same partition AND across partitions (each arm's value path is
    group-membership invariant: per-arm state machines, per-lane fp64
    collectives, per-config shard_select calls). The echo keys
    (fused_state_budget_mb / state_budget_mb / group_cap) are conditional —
    the budget-off report gains no keys at all, so default-path artifacts
    stay byte-identical."""
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        # warm the shared dense cache once (the populating run must not sit
        # in any A/B arm)
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        for shards in ("1", "2"):
            w = int(shards)

            def _arm(extra):
                return _run(_base_args(
                    td, cache, t0, ["--fused-shards", shards, *extra],
                    cycle=("--cycle-lens", "2,3")))

            cnt2 = _arm(["--fused-max-configs", "2"])          # 4 groups
            bud2 = _arm(["--fused-state-budget-mb",
                         repr(_budget_mb(w, 2))])              # same 4 groups
            cnt8 = _arm(["--fused-max-configs", "8"])          # 1 group
            bud8 = _arm(["--fused-state-budget-mb",
                         repr(_budget_mb(w, 8))])              # same 1 group
            # echoes: absent when off (byte-stability), present and correct
            # when on, with the derived cap reported
            assert "fused_state_budget_mb" not in cnt2["config"]
            br2 = cnt2["branch_results"][str(t0)]["replay"]
            assert "state_budget_mb" not in br2 and "group_cap" not in br2
            # the report sanitizes floats to 8 decimals — compare echoed
            # values on the same grid
            assert bud2["config"]["fused_state_budget_mb"] \
                == round(_budget_mb(w, 2), 8)
            brr = bud2["branch_results"][str(t0)]["replay"]
            assert brr["state_budget_mb"] == round(_budget_mb(w, 2), 8)
            assert brr["group_cap"] == 2
            assert bud8["branch_results"][str(t0)]["replay"][
                "group_cap"] == 8
            # strip the knob echoes; all four arms must coincide bitwise —
            # same partition (count vs budget) and across partitions (4
            # groups vs 1 group). fused_max_configs is a knob echo too: the
            # count arms set it to the partition size while the budget arms
            # leave the default (the budget overrides it), and the
            # cross-partition compare (cnt2 vs cnt8) differs on it by
            # construction — the partition-invariance claim is about
            # VALUES, not which knob chose the partition
            for rep in (cnt2, bud2, cnt8, bud8):
                rep["config"].pop("fused_max_configs", None)
                rep["config"].pop("fused_state_budget_mb", None)
                frag = rep["branch_results"][str(t0)]["replay"]
                frag.pop("state_budget_mb", None)
                frag.pop("group_cap", None)
            assert _deep_eq_nan(cnt2, bud2), \
                f"same-partition budget run moved a bit (shards={shards})"
            assert _deep_eq_nan(cnt8, bud8), \
                f"same-partition budget run moved a bit (shards={shards})"
            assert _deep_eq_nan(cnt2, cnt8), \
                f"partition changed the values (shards={shards})"
        # uneven-shard leg: world 7 does not divide N (870000 / 7 ->
        # 124285.71), so the budget cap basis ceil(N/world) = 124286
        # differs from six of the seven rank-local sp.n values — a
        # regression to a per-shard basis would disagree across ranks at
        # the floor and tear the qn2/num2 collectives. A non-hanging run
        # implicitly pins cross-rank partition agreement; the echo and
        # the bitwise A/B vs the count knob pin the derived cap itself
        shards = "7"

        def _arm7(extra):
            return _run(_base_args(
                td, cache, t0, ["--fused-shards", shards, *extra],
                cycle=("--cycle-lens", "2,3")))

        cnt7 = _arm7(["--fused-max-configs", "2"])
        bud7 = _arm7(["--fused-state-budget-mb", repr(_budget_mb(7, 2))])
        assert bud7["branch_results"][str(t0)]["replay"]["group_cap"] == 2
        for rep in (cnt7, bud7):
            rep["config"].pop("fused_max_configs", None)
            rep["config"].pop("fused_state_budget_mb", None)
            frag = rep["branch_results"][str(t0)]["replay"]
            frag.pop("state_budget_mb", None)
            frag.pop("group_cap", None)
        assert _deep_eq_nan(cnt7, bud7), \
            "uneven-shard budget run moved a bit (shards=7)"
    print("PASS test_state_budget_bitwise")


def test_state_budget_cap_semantics():
    """Floor semantics and the >=1 clamp, pinned through the group_cap echo
    with budgets recomputed from the spec constants: half a config's bytes
    clamps to cap 1, 2.5 configs floors to cap 2, one config per exact
    budget unit steps cap 1 -> 8."""
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        one = _budget_mb(1, 1)
        for budget, want in ((one / 2, 1), (one, 1),
                             (_budget_mb(1, 2) + one / 2, 2),
                             (_budget_mb(1, 8), 8)):
            rep = _run(_base_args(
                td, cache, t0,
                ["--fused-shards", "1",
                 "--fused-state-budget-mb", repr(budget)],
                cycle=("--cycle-lens", "2,3")))
            assert rep["branch_results"][str(t0)]["replay"]["group_cap"] \
                == want, (budget, want)
    print("PASS test_state_budget_cap_semantics")


def test_state_budget_cuda():
    """CUDA leg: the budget-derived partition stays bit-identical to the
    count knob, and a narrower budget stacked with the production perf
    flags (g residency + dev reduce) stays bit-identical too — the
    intended combination. World 1 wherever a GPU is visible; the world-2
    leg needs 2 GPUs."""
    if not torch.cuda.is_available():
        print("SKIP test_state_budget_cuda (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--device", "cuda",
                                        "--fused-shards", "1"]))
        shard_legs = ["1"]
        if torch.cuda.device_count() >= 2:
            shard_legs.append("2")
        else:
            print("NOTE test_state_budget_cuda: world-2 leg skipped "
                  "(needs 2 visible GPUs)")
        for shards in shard_legs:
            w = int(shards)

            def _arm(extra):
                return _run(_base_args(
                    td, cache, t0,
                    ["--device", "cuda", "--fused-shards", shards, *extra],
                    cycle=("--cycle-lens", "2,3")))

            base = _arm(["--fused-max-configs", "8"])
            wide = _arm(["--fused-state-budget-mb", repr(_budget_mb(w, 8))])
            combo = _arm(["--fused-state-budget-mb", repr(_budget_mb(w, 2)),
                          "--fused-g-residency", "--fused-dev-reduce"])
            # strip the provenance echoes: the budget knob echoes only on
            # the budget arms, the count-knob echo differs by construction
            # (base sets 8, the budget arms keep the default 12), and the
            # combo arm's perf-flag echoes (fused_g_residency/fused_dev_reduce
            # in config, g_residency/dev_reduce in the fragment) are
            # unconditional — the partition/flag-invariance claim is about
            # VALUES, not which knob chose the partition or which flags ran
            for rep in (base, wide, combo):
                rep["config"].pop("fused_max_configs", None)
                rep["config"].pop("fused_state_budget_mb", None)
                frag = rep["branch_results"][str(t0)]["replay"]
                frag.pop("state_budget_mb", None)
                frag.pop("group_cap", None)
            for rep in (base, wide, combo):
                rep["config"].pop("fused_g_residency", None)
                rep["config"].pop("fused_dev_reduce", None)
                frag = rep["branch_results"][str(t0)]["replay"]
                frag.pop("g_residency", None)
                frag.pop("dev_reduce", None)
            assert _deep_eq_nan(base, wide), f"shards={shards}"
            assert _deep_eq_nan(base, combo), f"shards={shards}"
    print("PASS test_state_budget_cuda")


def test_s0_profile_counters():
    """REPLAY_S0_PROFILE=1 (env-gated diagnostic): the fragment gains one
    counter dict per config group and the S0PROFILE line prints. With
    residency on, every shared upload byte is booked to pass 1 (p2/p3
    stay 0); with dev-reduce on, host syncs collapse to the per-pass
    fetches (one per step, plus the horizon fetch when a horizon lands)
    instead of the per-chunk storm. The gate binds at import, so the test
    sets the env AND the module flag, restored in finally (monkeypatch
    semantics without a pytest fixture, so main() below keeps working)."""
    old_env = os.environ.get("REPLAY_S0_PROFILE")
    old_gate = replay_fused._S0_PROFILE
    os.environ["REPLAY_S0_PROFILE"] = "1"
    replay_fused._S0_PROFILE = True
    try:
        with tempfile.TemporaryDirectory() as td:
            t0, W = _build_fixture(td)[0:2]
            cache = os.path.join(td, "dense_cache")
            buf = io.StringIO()
            with redirect_stdout(buf):
                report = soa.main(_base_args(
                    td, cache, t0,
                    ["--fused-shards", "1", "--fused-g-residency",
                     "--fused-dev-reduce"]))
                # the budget knob: state_gb books the resident state (the
                # derived cap shows up in the fragment echo)
                report_b = soa.main(_base_args(
                    td, cache, t0,
                    ["--fused-shards", "1", "--fused-g-residency",
                     "--fused-dev-reduce",
                     "--fused-state-budget-mb", repr(_budget_mb(1, 4))]))
            assert "S0PROFILE" in buf.getvalue()
            # the p2 attribution split rides the same line, with the
            # select sub-buckets nested inside it
            assert "p2_formula_s=" in buf.getvalue()
            assert "p2_select_s=" in buf.getvalue()
            for field in ("p2s_hist=", "p2s_ar=", "p2s_cand=", "p2s_fill="):
                assert field in buf.getvalue(), field
            # the gap buckets (three-way split of the streaming residual:
            # own read-wait / lockstep wait / real work, coverage-checked
            # against the group wall) ride the same line
            for field in ("gw_s=", "qwait_s=", "h2d_launch_s=", "p1_d2h_s=",
                          "p1_ar_s=", "hz_ar_s=", "setup_s=", "clean_s=",
                          "cov="):
                assert field in buf.getvalue(), field
            # the front-phase decomposition line rides the same gate (the
            # cpu device arm echoes front_sort=cpu)
            assert "S0FRONT" in buf.getvalue()
            assert "front_sort=cpu" in buf.getvalue()
            profs = report["branch_results"][str(t0)]["s0_profile"]
            assert isinstance(profs, list) and profs
            for prof in profs:
                for key in ("sync", "ar", "h2d", "state_gb", "p2f", "p2s",
                            "p2s_hist", "p2s_ar", "p2s_cand", "p2s_fill",
                            "qw", "h2dl", "p1d2h", "p1ar", "hzar", "stp",
                            "cln", "gw"):
                    assert key in prof, key
                # the split brackets nest inside the pass-2 bracket
                assert prof["p2f"] >= 0.0 and prof["p2s"] >= 0.0
                assert prof["p2f"] + prof["p2s"] <= prof["pt"][1] + 0.01, \
                    (prof["p2f"], prof["p2s"], prof["pt"][1])
                # the select sub-buckets nest inside the select bracket
                sub = (prof["p2s_hist"] + prof["p2s_ar"] + prof["p2s_cand"]
                       + prof["p2s_fill"])
                assert 0.0 <= sub <= prof["p2s"] + 0.01, \
                    (sub, prof["p2s"])
                # the p1 sub-buckets nest inside the pass-1 bracket; all
                # gap buckets are nonnegative and bounded by the group wall
                assert 0.0 <= prof["p1d2h"] + prof["p1ar"] \
                    <= prof["pt"][0] + 0.01, \
                    (prof["p1d2h"], prof["p1ar"], prof["pt"][0])
                assert prof["gw"] > 0.0
                for key in ("qw", "h2dl", "hzar", "stp", "cln"):
                    assert 0.0 <= prof[key] <= prof["gw"] + 0.01, \
                        (key, prof[key], prof["gw"])
                # coverage: the buckets account for the group wall — loose
                # bounds on this tiny CPU fixture (the glue outside the
                # brackets is proportionally larger here); the production
                # gate (>=80%) is not checked here
                cov = (prof["stp"] + prof["qw"] + prof["h2dl"]
                       + sum(prof["pt"]) + prof["hzar"] + prof["cln"]
                       ) / prof["gw"]
                assert 0.3 <= cov <= 1.05, (cov, prof)
                # residency books every shared upload to pass 1
                assert prof["h2d"][0] > 0
                assert prof["h2d"][1] == 0 and prof["h2d"][2] == 0
                # dev-reduce: one norm fetch per step plus one horizon
                # fetch on the 3 horizon steps of this window — generous
                # bound, but far under the per-chunk sync storm
                assert prof["sync"] <= 2 * W, (prof["sync"], W)
            profs_b = report_b["branch_results"][str(t0)]["s0_profile"]
            assert isinstance(profs_b, list) and profs_b
            # value pin: len(group) * sp.n * (4 * itemsize + 1 B sel) /
            # 2**30 — cap 4 over the 4 (score, ratio, T) arms is one full
            # group, world 1 so sp.n == N == sum(SIZES), itemsize from the
            # replay_dtype echo; the report sanitizes floats to 8 decimals
            itemsize = 8 if report_b["config"]["replay_dtype"] == "fp64" \
                else 4
            expected_gb = 4 * sum(SIZES) * (4 * itemsize + 1) / 2 ** 30
            for prof in profs_b:
                assert prof["state_gb"] > 0
                assert prof["state_gb"] == round(expected_gb, 8), \
                    (prof["state_gb"], expected_gb)
            assert report_b["branch_results"][str(t0)]["replay"][
                "group_cap"] == 4
    finally:
        if old_env is None:
            os.environ.pop("REPLAY_S0_PROFILE", None)
        else:
            os.environ["REPLAY_S0_PROFILE"] = old_env
        replay_fused._S0_PROFILE = old_gate
    print("PASS test_s0_profile_counters")


def main():
    test_fused_world1_bitwise()
    test_fused_world2_band()
    test_fused_world2_norm_fp32()
    test_fused_cold_cache_world2()
    test_fused_world2_balanced_histogram()
    test_fused_cli_gates()
    test_cycle_lens_mixed_equivalence()
    test_cycle_lens_single_compat()
    test_comms_env_multinode_world2()
    test_comms_env_global_local_split()
    test_comms_env_errors()
    test_fused_cuda_world1()
    test_fused_bf16_trace_world1()
    test_g_residency_bitwise()
    test_g_residency_cuda_pinned()
    test_gprefetcher_pinned_buffers()
    test_pread_trace_index()
    test_pread_prefetch_bitwise()
    test_gprefetcher_index_cache_shared_across_groups()
    test_pread_pipeline_lifecycle()
    test_pread_failure_injection()
    test_pread_guards_through_prefetcher()
    test_dev_reduce_bitwise()
    test_dev_reduce_cuda()
    test_stable_order_desc_adversarial()
    test_ranking_idcg_device_bitwise()
    test_fused_sort_device_bitwise()
    test_sort_oom_fallback_loud_cpu()
    test_sort_guard_empty_cache_before_measure()
    test_front_sort_env_killswitch()
    test_front_rank_bg_env_killswitch()
    test_front_rank_bg_world2()
    test_front_rank_bg_worker_unit()
    test_shard_select_device_bitwise()
    test_all_gather_pad_dev_world1()
    test_all_gather_pad_dev_world2()
    test_fused_select_device_bitwise()
    test_select_env_killswitch()
    test_select_nccl_world2()
    test_scan_dtype_guards()
    test_state_budget_bitwise()
    test_state_budget_cap_semantics()
    test_state_budget_cuda()
    test_s0_profile_counters()
    test_replay_kernels_killswitch()
    print("\nall replay-fused tests passed")


if __name__ == "__main__":
    main()
