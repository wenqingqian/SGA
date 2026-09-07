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
from test_replay_cycle import reference_cycle


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


def _make_trace_multi(tmpdir, pre_grads, lrs, clip, t0, absent=None):
    """Multi-param synthetic trace; ``absent`` maps step-index -> param name
    that is missing from that step's g payload (the zero-fill path).
    Returns (w0s, m0s, v0s, g0) so from-spec references can replay the exact
    saved branch state."""
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
        payload = {n: (pre[n] * coef).float()
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


def _build_fixture(tmpdir):
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
                                           absent=absent)
    return t0, W, pre, lrs, clip, absent, w0s, m0s, v0s, g0s


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return soa.main(argv)


def _base_args(td, cache, t0, extra=()):
    return ["--trace-dir", td, "--branch-point", str(t0), "--window", "6",
            "--horizons", "1,3,6", "--ratios", "20,30",
            "--score-names", "abs_g,update", "--replay-mode", "cycle",
            "--cycle-len", "2", "--dense-cache", cache, "--device", "cpu",
            "--chunk-mb", "1", *extra]


def _close(a, b, rel):
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom == 0.0 or abs(a - b) / denom <= rel


def _ref_distortion(pre, lrs, clip, t0, absent, w0s, m0s, v0s, g0s, score,
                    ratio, T, W):
    """From-spec distortion {str(h): num/den} for one score x ratio arm on
    the multi-param fixture (params concatenated in catalog order — the
    engine's flat index space). Rebuilds the exact logged gradients and
    meta norms the fixture writer produced (the trace inputs the engine
    and the reference both consume); absent params replay as zero
    gradients; g0s is the branch state's g — the branch-point selection
    input."""
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
                 else (g[n] * coef).float().double()
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
                                   ref_args["T"], ref_args["W"])
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
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--score-names", "nope"],
             "not a registered score"),
            (base + ["--replay-mode", "cycle", "--dense-cache", cache,
                     "--ratios", "150"],
             "in (0, 100]"),
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


def main():
    test_fused_world1_bitwise()
    test_fused_world2_band()
    test_fused_world2_norm_fp32()
    test_fused_cold_cache_world2()
    test_fused_cli_gates()
    test_fused_cuda_world1()
    print("\nall replay-fused tests passed")


if __name__ == "__main__":
    main()
