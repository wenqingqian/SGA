#!/usr/bin/env python3
"""Unit tests for the fused single-pass streaming engine (sga.replay_fused).

Run from anywhere (world=1 and world=2 both work on CPU; gloo over
localhost carries the tiny control-plane exchanges):

    python SGA/tests/test_replay_fused.py

Contract under test (fused-engine equivalence):
1. world=1: the --fused report equals the classic --replay-mode cycle
   report bitwise (json equality of branch_results against a dense-cache
   HIT run of the classic path).
2. world=2: params straddle the shard boundary; the only deviations are
   cross-shard fp64 scalar-sum reorderings — distortions and sum-type
   statistics stay within the pure-reorder band (rel <= 1e-6), everything
   that is not a sum (thresholds, ranking metrics, cache-sourced dense
   artifacts, min/max/zero fractions) is exact.
3. --norm-dtype fp32 threads through the fused engine (same comparison
   against the classic fp32-norm arm, same band).
4. CLI gates: --fused rejects zerofill mode, missing --dense-cache,
   --self-check, shard count < 1, max-configs < 1, unknown scores,
   out-of-range ratios and an explicit cuda:N --device, each with the
   option named.
5. CUDA smoke (skipped without a GPU): world=1 on cuda reproduces the
   classic cuda report bitwise.
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga import replay as soa


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
    that is missing from that step's g payload (the zero-fill path)."""
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


def _build_fixture(tmpdir):
    """Clipped 6-step trace (the clip binds every step), 3 params, p2 absent
    at the si=2 boundary and p1 absent at the si=3 masked step."""
    torch.manual_seed(11)
    t0, W = 900, 6
    pre = [{n: torch.randn(s) * 0.5 for n, s in zip(NAMES, SIZES)}
           for _ in range(W)]
    _make_trace_multi(tmpdir, pre, [5e-5] * W, 0.3, t0,
                      absent={2: "p2", 3: "p1"})
    return t0, W


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


def test_fused_world1_bitwise():
    """Single shard (inline, no comms): the fused report equals the classic
    cycle report bitwise — same dense cache artifacts, same per-config
    arithmetic order. The synthetic state meta has no grad_norm (NaN != NaN
    under plain dict equality), so compare the canonical JSON text."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0))                    # classic, miss
        classic = _run(_base_args(td, cache, t0))          # classic, hit
        fused = _run(_base_args(td, cache, t0,
                                ["--fused", "--fused-shards", "1",
                                 "--fused-max-configs", "2"]))
        assert classic["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is True
        assert fused["config"]["fused"] is True
        assert fused["config"]["fused_shards"] == 1
        assert fused["config"]["fused_max_configs"] == 2
        a = json.dumps(classic["branch_results"], sort_keys=True)
        b = json.dumps(fused["branch_results"], sort_keys=True)
        assert a == b, "world=1 fused report differs from the classic report"
    print("PASS test_fused_world1_bitwise")


def _compare_world2(classic, fused, label):
    bc = classic["branch_results"]
    bf = fused["branch_results"]
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
    SUMS reorder (clip norms, distortion numerators, sum statistics)."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0))                    # populate cache
        classic = _run(_base_args(td, cache, t0))
        fused = _run(_base_args(td, cache, t0,
                                ["--fused", "--fused-shards", "2",
                                 "--fused-max-configs", "2"]))
        assert fused["config"]["fused_shards"] == 2
        _compare_world2(classic, fused, "fp64")
    print("PASS test_fused_world2_band")


def test_fused_world2_norm_fp32():
    """--norm-dtype fp32 on both arms: same quantization, so the fused-vs-
    classic deviation stays pure-reorder (the fp32-norm effect itself is
    gated by the frozen a2 band, not by this test)."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        _run(_base_args(td, cache, t0, ["--norm-dtype", "fp32"]))
        classic = _run(_base_args(td, cache, t0, ["--norm-dtype", "fp32"]))
        fused = _run(_base_args(td, cache, t0,
                                ["--norm-dtype", "fp32", "--fused",
                                 "--fused-shards", "2"]))
        _compare_world2(classic, fused, "fp32-norm")
    print("PASS test_fused_world2_norm_fp32")


def test_fused_cold_cache_world2():
    """Cold cache with no classic pre-run: run_fused_branch populates the
    dense cache parent-side before the shards start, so --fused
    --fused-shards 2 runs directly on a fresh cache dir; the populated cache
    then serves the classic arm. The world=2 fp32 sum-statistic reordering
    survives the report's 8-decimal sanitize (~4e-6 abs on the update-score
    sum), so the branch_results go through the standard world=2 comparator
    rather than the world=1 canonical-JSON equality."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        fused = _run(_base_args(td, cache, t0,
                                ["--fused", "--fused-shards", "2"]))
        classic = _run(_base_args(td, cache, t0))
        assert classic["branch_results"][str(t0)]["replay"] \
            ["dense_cache_hit"] is True
        _compare_world2(classic, fused, "cold-cache")
    print("PASS test_fused_cold_cache_world2")


def test_fused_cli_gates():
    """Each invalid --fused combination is rejected with the option named,
    before any replay work."""
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        base = ["--trace-dir", td, "--branch-point", str(t0),
                "--ratios", "20", "--horizons", "3", "--device", "cpu"]
        cases = [
            (base + ["--fused"], "--fused implements --replay-mode cycle"),
            (base + ["--fused", "--replay-mode", "cycle"],
             "--fused requires --dense-cache"),
            (["--trace-dir", "/nonexistent-sga-trace-dir", "--fused",
              "--self-check", "900", "901"],
             "--fused cannot be combined with --self-check"),
            (base + ["--fused", "--replay-mode", "cycle",
                     "--dense-cache", cache, "--fused-shards", "0"],
             "--fused-shards must be >= 1"),
            (base + ["--fused", "--replay-mode", "cycle",
                     "--dense-cache", cache, "--fused-max-configs", "0"],
             "--fused-max-configs must be >= 1"),
            (base + ["--fused", "--replay-mode", "cycle",
                     "--dense-cache", cache, "--fused-max-configs", "-1"],
             "--fused-max-configs must be >= 1"),
            (base + ["--fused", "--replay-mode", "cycle",
                     "--dense-cache", cache, "--score-names", "nope"],
             "not a registered score"),
            (base + ["--fused", "--replay-mode", "cycle",
                     "--dense-cache", cache, "--ratios", "150"],
             "in (0, 100]"),
        ]
        if torch.cuda.is_available():
            # --device cpu: shards are plain processes with no CUDA bound,
            # so the "exceeds" gate needs a cuda device to fire
            cases += [
                # fires the named fused cuda:N rejection on any CUDA box:
                # main's earlier gate only rejects --device cuda when CUDA
                # is unavailable
                (base + ["--fused", "--replay-mode", "cycle",
                         "--dense-cache", cache, "--device", "cuda:1"],
                 "an explicit cuda:N index is not supported"),
                (base + ["--fused", "--replay-mode", "cycle",
                         "--dense-cache", cache, "--device", "cuda",
                         "--fused-shards", "99"],
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
    """CUDA smoke: world=1 on cuda (both arms on cuda:0, dense via
    dense_replay_dev) reproduces the classic report bitwise."""
    if not torch.cuda.is_available():
        print("SKIP test_fused_cuda_world1 (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as td:
        t0, _W = _build_fixture(td)
        cache = os.path.join(td, "dense_cache")
        base = _base_args(td, cache, t0)
        _run(base + ["--device", "cuda"])
        classic = _run(base + ["--device", "cuda"])
        fused = _run(base + ["--device", "cuda", "--fused",
                             "--fused-shards", "1"])
        a = json.dumps(classic["branch_results"], sort_keys=True)
        b = json.dumps(fused["branch_results"], sort_keys=True)
        assert a == b, "world=1 cuda fused report differs from classic"
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
