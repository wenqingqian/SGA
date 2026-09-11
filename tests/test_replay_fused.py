#!/usr/bin/env python3
"""Unit tests for the fused single-pass streaming engine (sga.replay_fused).

Run from anywhere (world=1 and world=2 both work on CPU; gloo over
localhost carries the tiny control-plane exchanges):

    python SGA/tests/test_replay_fused.py

Contract under test (the fused engine is the ONLY cycle engine — the
classic in-process cycle replay was removed; cycle auto-routes here):
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
5. CLI gates: cycle without --dense-cache (the post-consolidation
   behavior change), --fused outside cycle mode, --self-check with cycle
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
   the fragment (residency books every shared H2D byte to pass 1,
   dev-reduce collapses the host syncs to the per-pass fetches), the
   pinned prefetcher buffers on CUDA, and a --trace-dtype bf16 leg of
   the slice-first prefetcher (reference anchor + bitwise fp32-twin A/B).
10. --fused-state-budget-mb: the resident-state byte budget derives the
   config group size (overrides --fused-max-configs); reports stay
   bit-identical to the count knob at the same partition AND across
   partitions; the echoes are conditional so budget-off artifacts keep
   their exact bytes; the cap derivation (floor + >=1 clamp) is pinned
   through the group_cap echo; REPLAY_S0_PROFILE gains a state_gb field.
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sga import replay as soa
from sga import replay_fused
from test_replay_cycle import reference_cycle

# runners may export REPLAY_S0_PROFILE globally for their GPU arms; the
# suite must stay immune (the s0_profile fragment key carries wall-clock
# counters that break the bitwise A/Bs). Tests that need it set it
# themselves. Covers both the pytest and the main() entry paths.
if "REPLAY_S0_PROFILE" in os.environ:
    del os.environ["REPLAY_S0_PROFILE"]
    replay_fused._S0_PROFILE = False


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


def _base_args(td, cache, t0, extra=(), cycle=("--cycle-len", "2")):
    # ``cycle`` is swappable because the two T spellings are mutually
    # exclusive on the CLI — a --cycle-lens run cannot just append the flag
    return ["--trace-dir", td, "--branch-point", str(t0), "--window", "6",
            "--horizons", "1,3,6", "--ratios", "20,30",
            "--score-names", "abs_g,update", "--replay-mode", "cycle",
            *cycle, "--dense-cache", cache, "--device", "cpu",
            "--chunk-mb", "1", *extra]


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
    w0 = torch.cat([w0s[n] for n in NAMES]).double()
    m0 = torch.cat([m0s[n] for n in NAMES]).double()
    v0 = torch.cat([v0s[n] for n in NAMES]).double()
    branch_grad = torch.cat([g0s[n] for n in NAMES]).double()
    logged_flat, norms = [], []
    for si, g in enumerate(pre):
        norm = 0.0
        for n in NAMES:
            norm += float((g[n] ** 2).sum())
        norm = norm ** 0.5
        coef = min(1.0, clip / (norm + 1.0e-6)) if clip > 0.0 else 1.0
        parts = [torch.zeros(SIZES[i]) if absent.get(si) == n
                 else (g[n] * coef).to(payload_dtype).float().double()
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
    rounding step plus a relative slack for the engine's chunked fp64
    summation order."""
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
                    assert abs(g - w) <= 5e-8 + 1e-6 * abs(w), \
                        (s, r, h, g, w)


def test_fused_world1_bitwise():
    """Single shard (inline, no comms): a fresh run populates the cache and
    reports dense_cache_hit False, the hit run is bitwise-identical to the
    same run invoked with the deprecated --fused flag (the comparison arm
    is a second cache HIT — the miss->hit roundtrip equality is anchored by
    test_replay_cycle.test_dense_cache_roundtrip), and every distortion
    matches the from-spec reference (the classic comparison arm is gone
    with the classic engine — reference_cycle is the spec anchor)."""
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
        _assert_ref_anchor(hit, t0, ref_args, ("1", "3", "6"))
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
    world=2 against the world=1 run (self-consistency; the classic
    comparison arm is gone with the classic engine)."""
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
    gated by the frozen a2 band, not by this test)."""
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


def test_fused_cli_gates():
    """Each invalid combination is rejected with the option named, before
    any replay work."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)[0:2]
        cache = os.path.join(td, "dense_cache")
        base = ["--trace-dir", td, "--branch-point", str(t0),
                "--ratios", "20", "--horizons", "3", "--device", "cpu"]
        cases = [
            # zerofill (the default) has no fused form; --fused is a
            # deprecated no-op only where it ever made sense
            (base + ["--fused"], "--fused is deprecated"),
            # the post-consolidation behavior change, named loudly
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
    import torch.multiprocessing as mp
    _with_rendezvous_env(lambda: mp.spawn(
        _env_comms_worker, args=(2, 0, 2, 2.0), nprocs=2, join=True))
    print("PASS test_comms_env_multinode_world2")


def test_comms_env_global_local_split():
    """Two simulated nodes x 2 local shards (world=4): global rank != local
    rank on the second node, init_process_group uses the global rank (all
    4 meet), collectives return in global process-group rank order (the
    order shard_select's tie-fill indexes), and self.rank stays the local
    shard index."""
    import torch.multiprocessing as mp
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
        _assert_ref_anchor(report, t0, ref_args, ("1", "3", "6"))
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
        _assert_ref_anchor(report, t0, ref_args, ("1", "3", "6"))
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


def test_dev_reduce_bitwise():
    """--fused-dev-reduce accumulates the pass-1 clip norms and pass-3
    horizon numerators in per-config fp64 device scalars fetched once per
    pass. The addends and their order are unchanged, so the whole report
    must be bit-identical to the per-chunk host-sync path (world 1 and 2,
    warm cache, both norm dtypes; on CUDA also combined with residency)."""
    with tempfile.TemporaryDirectory() as td:
        t0, *_ = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--fused-shards", "1"]))
        for shards in ("1", "2"):
            for nd in ("fp64", "fp32"):
                base = _run(_base_args(td, cache, t0,
                                       ["--fused-shards", shards,
                                        "--norm-dtype", nd]))
                red = _run(_base_args(td, cache, t0,
                                      ["--fused-shards", shards,
                                       "--norm-dtype", nd,
                                       "--fused-dev-reduce"]))
                assert base["config"]["fused_dev_reduce"] is False
                assert red["config"]["fused_dev_reduce"] is True
                for rep in (base, red):
                    rep["config"].pop("fused_dev_reduce")
                    frag = rep["branch_results"][str(t0)]
                    frag["replay"].pop("dev_reduce")
                assert _deep_eq_nan(base, red), \
                    f"dev_reduce moved a bit (shards={shards} nd={nd})"
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


def _budget_mb(shards, cap, replay_dtype="fp64"):
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
    stay byte-identical (absorb-style byte cmps)."""
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
            profs = report["branch_results"][str(t0)]["s0_profile"]
            assert isinstance(profs, list) and profs
            for prof in profs:
                for key in ("sync", "ar", "h2d", "state_gb"):
                    assert key in prof, key
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
    test_dev_reduce_bitwise()
    test_dev_reduce_cuda()
    test_state_budget_bitwise()
    test_state_budget_cap_semantics()
    test_state_budget_cuda()
    test_s0_profile_counters()
    print("\nall replay-fused tests passed")


if __name__ == "__main__":
    main()
