#!/usr/bin/env python3
"""Unit tests for the cycle-mode (residual + multi-refresh) replay oracle.

Run from anywhere (CPU only, torch + stdlib):

    python SGA/tests/test_replay_cycle.py

The cycle replay (sga.replay.sparse_replay_cycle) is the offline twin of the
online cyclic semantics: masked steps communicate ``q = g ⊙ sel`` (clipped by
the masked norm) and accumulate the unselected pre-clip gradient into a
residual; every ``cycle_len`` steps a full-sync boundary commits
``q = g + residual`` (clipped by the folded norm), clears the residual, and
re-selects the next cycle's mask from the boundary score computed on the
pre-commit state.

Tests (ported from the legacy oracle suite, importing sga.replay):
1. T=1 degenerates to the dense trajectory (distortion exactly 0).
2. T > window: residual accumulates but never folds (fixed mask from the
   branch selection) — matches an independent reference loop.
3. Multi-cycle hand example (T=2, 6 steps, clipping active): w/m/v and the
   residual/fold math match a from-spec reference; mask re-selection fires
   at the expected boundaries.
4. Boundary re-selection actually changes the support when momentum flips.
5. Score formulas: the cycle scores match direct fp64 formulas.
6. abs_m_sticky loyalty-boost semantics.
7. Determinism: two identical runs produce bitwise-equal distortion numbers.
8. Per-group hyperparameter resolution (new-trace additive extension with
   legacy global fallback).
"""
import io
import math
import os
import sys
import tempfile
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga import replay as soa


# ---------------------------------------------------------------------------
# independent from-spec reference implementation (whole model, no chunking)
# ---------------------------------------------------------------------------

B1, B2, EPS, WD = 0.9, 0.95, 1e-8, 0.1
BASE_META = {"adam_beta1": B1, "adam_beta2": B2, "adam_eps": EPS,
             "weight_decay": WD}


def _bc(count, b1=B1, b2=B2):
    return 1.0 - b1 ** count, 1.0 - b2 ** count


def _adamw(w, m, v, q, lr, count):
    bc1, bc2 = _bc(count)
    m = B1 * m + (1.0 - B1) * q
    v = B2 * v + (1.0 - B2) * q * q
    upd = (m / bc1) / (torch.sqrt(v / bc2) + EPS)
    return w * (1.0 - lr * WD) - lr * upd, m, v


def _ref_score(name, w, m, v, g, count, b1=B1, b2=B2):
    bc1, bc2 = _bc(count, b1, b2)
    if name == "abs_g":
        return g.abs()
    if name in ("abs_m", "abs_m_sticky"):
        return m.abs()
    if name == "abs_m_post":
        return (b1 * m + (1.0 - b1) * g).abs()
    if name in ("snr_m", "radius"):
        # state-matched (pre-commit) corrections: the moments have seen
        # count-1 commits
        bc1p, bc2p = _bc(count - 1, b1, b2)
        if bc1p <= 0.0 or bc2p <= 0.0:
            return torch.zeros_like(m)
        snr = (m / bc1p) / (torch.sqrt(v / bc2p) + EPS)
        return snr.abs() if name == "snr_m" else (WD * w + snr).abs()
    mg = (b1 * m + (1.0 - b1) * g) / bc1
    vg = (b2 * v + (1.0 - b2) * g * g) / bc2
    ug = mg / (torch.sqrt(vg) + EPS)
    if name == "full_update":
        return ug.abs()
    if name == "update":
        return (WD * w + ug).abs()
    u0 = (b1 * m / bc1) / (torch.sqrt(b2 * v / bc2) + EPS)
    return (ug - u0).abs()


def _topk_mask(sc, k):
    idx = torch.topk(sc, k).indices
    mask = torch.zeros(sc.numel(), dtype=torch.bool)
    mask[idx] = True
    return mask


def reference_cycle(w0, m0, v0, pre_grads, lrs, clip, t0, score_name, k, T,
                    sticky_lambda=0.5):
    """From-spec reference: returns (dense_w_at_each_step, sparse_w_at_each_step).

    ``pre_grads`` are the PRE-clip gradients (the logged trace gradient is
    pre * coef_dense; coef_dense is what the dense run applied).
    For abs_m_sticky the re-selection score gets the support-loyalty boost
    (1 + sticky_lambda * prev_sel); the first selection has no previous
    support and degenerates to plain abs_m.
    """
    # dense trajectory
    md, vd, wdt = m0.clone(), v0.clone(), w0.clone()
    dense_ws = []
    for si, g in enumerate(pre_grads):
        count = t0 + si + 1
        coef = 1.0
        if clip > 0.0:
            coef = min(1.0, clip / (float(g.norm()) + 1.0e-6))
        wdt, md, vd = _adamw(wdt, md, vd, g * coef, lrs[si], count)
        dense_ws.append(wdt.clone())
    # sparse cycle trajectory
    ms, vs, ws = m0.clone(), v0.clone(), w0.clone()
    res = torch.zeros_like(w0)
    q0 = pre_grads[0]
    if clip > 0.0:
        q0 = q0 * min(1.0, clip / (float(q0.norm()) + 1.0e-6))
    sel = _topk_mask(_ref_score(score_name, ws, ms, vs, q0, t0 + 1), k)
    sparse_ws = []
    for si, g in enumerate(pre_grads):
        count = t0 + si + 1
        if si % T == 0:
            q_pre = g + res
            coef = 1.0
            if clip > 0.0:
                coef = min(1.0, clip / (float(q_pre.norm()) + 1.0e-6))
            q = q_pre * coef
            if si > 0 and si + 1 < len(pre_grads):
                sc = _ref_score(score_name, ws, ms, vs, q, count)
                if score_name == "abs_m_sticky":
                    sc = sc * (1.0 + sticky_lambda * sel.to(sc.dtype))
                sel = _topk_mask(sc, k)
            res = torch.zeros_like(res)
        else:
            coef = 1.0
            if clip > 0.0:
                coef = min(1.0, clip / (float((g * sel).norm()) + 1.0e-6))
            q = g * sel * coef
            res = res + g * (~sel)
        ws, ms, vs = _adamw(ws, ms, vs, q, lrs[si], count)
        sparse_ws.append(ws.clone())
    return dense_ws, sparse_ws


# ---------------------------------------------------------------------------
# harness: synthetic trace on disk + module replays
# ---------------------------------------------------------------------------

def _make_trace(tmpdir, pre_grads, lrs, clip, t0, n):
    """Write catalog/state/g files for a one-param synthetic trace.

    The logged gradient is post-clip (g_logged = pre * coef_dense), matching
    the real trace collector; meta carries the dense-run grad_norm so the
    replay reconstructs pre-clip via g_logged / coef_dense.
    """
    name = "p"
    torch.save({"catalog": {name: {"shape": [n]}}},
               os.path.join(tmpdir, "catalog.pt"))
    w0 = torch.randn(n, dtype=torch.float32)
    m0 = torch.randn(n, dtype=torch.float32) * 0.1
    v0 = torch.rand(n, dtype=torch.float32) * 0.01 + 1e-4
    g0_logged = None
    for si, pre in enumerate(pre_grads):
        step = t0 + si
        norm = float(pre.norm())
        coef = 1.0
        if clip > 0.0:
            coef = min(1.0, clip / (norm + 1.0e-6))
        logged = (pre * coef).float()
        meta = {"lr": [lrs[si]], "grad_norm": norm, "clip_grad": clip,
                "adam_beta1": B1, "adam_beta2": B2, "adam_eps": EPS,
                "weight_decay": WD, "step": step}
        torch.save({"payload": {name: logged}, "meta": meta},
                   os.path.join(tmpdir, f"step_{step:06d}_g.pt"))
        if si == 0:
            g0_logged = logged
    state = {"payload": {"w": {name: w0}, "m": {name: m0}, "v": {name: v0},
                         "g": {name: g0_logged}},
             "meta": {**BASE_META, "step": t0}}
    torch.save(state, os.path.join(tmpdir, f"step_{t0:06d}_state.pt"))
    return name, w0, m0, v0


def _run_module(tmpdir, name, w0, m0, v0, t0, W, score_name, ratio_pct, T,
                replay_dtype=torch.float64, sel_trace=None, sticky_lambda=0.5):
    """Run the module's dense + cycle sparse replays; return num_at {h: num}."""
    catalog, names, offsets, N, state_files, g_files = soa.discover_trace(tmpdir)
    meta, params, w_parts, m_parts, v_parts, g_parts = soa.build_branch_state(
        state_files[t0], names, replay_dtype)
    score_flats = soa.compute_scores_flat(
        w_parts, m_parts, v_parts, g_parts, offsets, N, meta,
        chunk_mb=1, device="cpu",
        score_names=soa.SCORE_NAMES_CYCLE + soa.SCORE_NAMES_CYCLE_EXTRA,
        update_includes_wd=True)
    k = max(1, int(round(ratio_pct / 100.0 * N)))
    sel0, _ = soa.select_topk_flat(score_flats[score_name], k)
    window_steps = list(range(t0, t0 + W))
    eff = set(range(1, W + 1))
    for prm in params:
        prm["w"] = prm["w0"].clone()
        prm["m"] = prm["m0"].clone()
        prm["v"] = prm["v0"].clone()
        prm["label"] = torch.zeros(prm["n"], dtype=replay_dtype)
    _label, _den, _recon, snap_d = soa.dense_replay(
        params, g_files, window_steps, meta, t0, eff, replay_dtype, 1, "cpu")
    num_at = soa.sparse_replay_cycle(
        params, g_files, window_steps, meta, t0, eff, score_name, k, T, sel0,
        snap_d, replay_dtype, 1, "cpu", sel_trace=sel_trace,
        sticky_lambda=sticky_lambda)
    return num_at


def _ref_num_at(w0, m0, v0, pre_grads, lrs, clip, t0, score_name, ratio_pct, T,
                sticky_lambda=0.5):
    k = max(1, int(round(ratio_pct / 100.0 * w0.numel())))
    dense_ws, sparse_ws = reference_cycle(
        w0.double(), m0.double(), v0.double(),
        [g.double() for g in pre_grads], lrs, clip, t0, score_name, k, T,
        sticky_lambda=sticky_lambda)
    return {h: float((sparse_ws[h - 1] - dense_ws[h - 1]).norm())
            for h in range(1, len(pre_grads) + 1)}


def _close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_T1_is_dense():
    """cycle_len=1: every step is a full-sync boundary -> dense trajectory."""
    torch.manual_seed(0)
    t0, W, n = 1600, 8, 16
    pre = [torch.randn(n) * 0.01 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0 = _make_trace(td, pre, lrs, 1.0, t0, n)
        num_at = _run_module(td, name, w0, m0, v0, t0, W, "coi", 50.0, 1)
    for h, v in num_at.items():
        assert v == 0.0, f"T=1 must be dense; d@{h} = {v}"
    print("PASS test_T1_is_dense")


def test_no_fold_within_window():
    """T > W: residual accumulates but never folds; fixed mask from si=0."""
    torch.manual_seed(1)
    t0, W, n = 2000, 6, 32
    pre = [torch.randn(n) * 0.02 for _ in range(W)]
    lrs = [4e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0 = _make_trace(td, pre, lrs, 0.0, t0, n)
        for score in soa.SCORE_NAMES_CYCLE + soa.SCORE_NAMES_CYCLE_EXTRA:
            num_at = _run_module(td, name, w0, m0, v0, t0, W, score, 25.0,
                                 T=W + 1)
            ref = _ref_num_at(w0, m0, v0, pre, lrs, 0.0, t0, score, 25.0,
                              T=W + 1)
            for h in range(1, W + 1):
                assert _close(num_at[h], ref[h], 1e-9), (
                    f"{score} d@{h}: module {num_at[h]} vs ref {ref[h]}")
    print("PASS test_no_fold_within_window")


def test_multicycle_with_clipping():
    """T=2 multi-cycle with active clipping: residual fold, per-step re-clip
    and boundary re-selection all match the from-spec reference."""
    torch.manual_seed(2)
    t0, W, n = 4000, 6, 8
    pre = [torch.randn(n) * 0.5 for _ in range(W)]  # norms >> clip -> active
    lrs = [5e-5, 4.5e-5, 4e-5, 3.5e-5, 3e-5, 2.5e-5]
    clip = 0.3
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0 = _make_trace(td, pre, lrs, clip, t0, n)
        for score in soa.SCORE_NAMES_CYCLE + soa.SCORE_NAMES_CYCLE_EXTRA:
            trace = []
            num_at = _run_module(td, name, w0, m0, v0, t0, W, score, 50.0, 2,
                                 sel_trace=trace)
            ref = _ref_num_at(w0, m0, v0, pre, lrs, clip, t0, score, 50.0, 2)
            for h in range(1, W + 1):
                assert _close(num_at[h], ref[h], 1e-9), (
                    f"{score} d@{h}: module {num_at[h]} vs ref {ref[h]}")
            # boundaries at si=0 (sel0, trace entry -1), si=2, si=4
            # (si+1 < W so re-selection happens)
            assert [s for s, _ in trace] == [-1, 2, 4], (
                f"{score}: re-selection boundaries {trace!r}")
    print("PASS test_multicycle_with_clipping")


def test_boundary_reselect_changes_mask():
    """A momentum flip between boundaries must change the selected support."""
    torch.manual_seed(3)
    t0, W, n = 1600, 5, 4
    # step 0 pushes coords 0-1; steps 1..4 push coords 2-3 hard, so the
    # abs_g/update/coi supports must rotate at the si=2 boundary (T=2).
    pre = [torch.tensor([1.0, 1.0, 0.01, 0.01])]
    pre += [torch.tensor([0.01, 0.01, 1.0, 1.0]) for _ in range(W - 1)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0 = _make_trace(td, pre, lrs, 0.0, t0, n)
        trace = []
        _run_module(td, name, w0, m0, v0, t0, W, "abs_g", 50.0, 2,
                    sel_trace=trace)
    sel0 = trace[0][1]
    sel2 = trace[1][1]
    assert sel0[:2].all() and not sel0[2:].any(), f"sel0={sel0}"
    assert sel2[2:].all() and not sel2[:2].any(), f"sel@si2={sel2}"
    print("PASS test_boundary_reselect_changes_mask")


def test_cycle_score_formulas():
    """_cycle_score_chunk vs direct fp64 formulas on a random state."""
    torch.manual_seed(4)
    n = 64
    w = torch.randn(n, dtype=torch.float64)
    m = torch.randn(n, dtype=torch.float64) * 0.1
    v = torch.rand(n, dtype=torch.float64) * 0.01 + 1e-4
    g = torch.randn(n, dtype=torch.float64) * 0.01
    count = 1234
    bc1, bc2 = _bc(count)
    for name in soa.SCORE_NAMES_CYCLE + soa.SCORE_NAMES_CYCLE_EXTRA:
        got = soa._cycle_score_chunk(name, w, m, v, g, WD, B1, B2, bc1, bc2,
                                     EPS)
        want = _ref_score(name, w, m, v, g, count)
        assert torch.equal(got, want), f"{name}: chunk formula mismatch"
    # edge: count=1 (state has zero commits) -> snr_m/radius are zeros,
    # matching compute_scores_flat's t0=0 zero-moments convention
    bc1_1, bc2_1 = _bc(1)
    for name in ("snr_m", "radius"):
        got = soa._cycle_score_chunk(name, w, m, v, g, WD, B1, B2,
                                     bc1_1, bc2_1, EPS)
        assert torch.equal(got, torch.zeros_like(m)), f"{name} count=1 edge"
    # mid-range counts: the derived pre-commit corrections carry ~1 ulp vs
    # the direct form (bitwise equality is NOT achievable there; at 1234
    # both corrections saturate to exactly 1.0, so the loop above cannot
    # distinguish derivations at all) — pin the derivation by tolerance
    for count in (50, 500):
        bc1, bc2 = _bc(count)
        for name in ("snr_m", "radius"):
            got = soa._cycle_score_chunk(name, w, m, v, g, WD, B1, B2,
                                         bc1, bc2, EPS)
            want = _ref_score(name, w, m, v, g, count)
            torch.testing.assert_close(got, want, rtol=1e-12, atol=0.0)
    # real-Adam regime: 0.999**1000 ~ 0.37 does not saturate, so the derived
    # formula carries its ulp against the direct 1 - beta**(count-1)
    bc1, bc2 = _bc(1000, B1, 0.999)
    for name in ("snr_m", "radius"):
        got = soa._cycle_score_chunk(name, w, m, v, g, WD, B1, 0.999,
                                     bc1, bc2, EPS)
        want = _ref_score(name, w, m, v, g, 1000, b2=0.999)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=0.0)
    print("PASS test_cycle_score_formulas")


def test_sticky_semantics():
    """abs_m_sticky: sel0 degenerates to abs_m (no previous support); the
    loyalty boost keeps a just-overtaken incumbent in the support."""
    torch.manual_seed(6)
    t0, W, n = 1600, 5, 4
    # m0: coord0 barely leads coord1; g pushes coord1 hard so its momentum
    # overtakes coord0's by the si=2 boundary -- plain abs_m rotates to
    # coord1, sticky (x1.5 incumbent boost) keeps coord0.
    pre = [torch.tensor([0.005, 0.5, 0.005, 0.005]) for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0 = _make_trace(td, pre, lrs, 0.0, t0, n)
        # m0 engineered post-trace-creation (harness randomizes it)
        m0_eng = torch.tensor([1.0, 0.96, 0.3, 0.2])
        st = torch.load(os.path.join(td, f"step_{t0:06d}_state.pt"),
                        weights_only=False)
        st["payload"]["m"][name] = m0_eng
        torch.save(st, os.path.join(td, f"step_{t0:06d}_state.pt"))
        tr_abs_m, tr_sticky = [], []
        _run_module(td, name, w0, m0, v0, t0, W, "abs_m", 25.0, 2,
                    sel_trace=tr_abs_m)
        _run_module(td, name, w0, m0, v0, t0, W, "abs_m_sticky", 25.0, 2,
                    sel_trace=tr_sticky)
    # sel0 identical (sticky has no history at the branch point)
    assert torch.equal(tr_abs_m[0][1], tr_sticky[0][1]), "sel0 must match"
    assert tr_abs_m[0][1][0] and not tr_abs_m[0][1][1:].any(), (
        f"sel0 should pick coord0: {tr_abs_m[0][1]}")
    # at si=2: abs_m rotates to coord1, sticky keeps coord0
    assert tr_abs_m[1][1][1] and not tr_abs_m[1][1][0], (
        f"abs_m@si2 should rotate to coord1: {tr_abs_m[1][1]}")
    assert tr_sticky[1][1][0] and not tr_sticky[1][1][1], (
        f"sticky@si2 should keep coord0: {tr_sticky[1][1]}")
    print("PASS test_sticky_semantics")


def test_determinism():
    """Two identical cycle replays produce bitwise-equal distortion numbers."""
    torch.manual_seed(5)
    t0, W, n = 1600, 6, 32
    pre = [torch.randn(n) * 0.02 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0 = _make_trace(td, pre, lrs, 1.0, t0, n)
        a = _run_module(td, name, w0, m0, v0, t0, W, "coi", 20.0, 2)
        b = _run_module(td, name, w0, m0, v0, t0, W, "coi", 20.0, 2)
    assert a == b, f"non-deterministic replay: {a} vs {b}"
    print("PASS test_determinism")


def test_per_group_hyperparams_resolve():
    """New-trace additive extension: meta["param_groups"] +
    catalog[name]["param_group"] resolve per-param betas/eps/wd and exact
    per-group lr; traces without them fall back to the global meta scalars
    (legacy behavior)."""
    torch.manual_seed(7)
    t0, n = 100, 8
    with tempfile.TemporaryDirectory() as td:
        # two params, one per optimizer group
        torch.save({"catalog": {
            "p": {"shape": [n], "param_group": 0},
            "q": {"shape": [n], "param_group": 1},
        }}, os.path.join(td, "catalog.pt"))
        groups = [
            {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8,
             "weight_decay": 0.1},
            {"adam_beta1": 0.5, "adam_beta2": 0.9, "adam_eps": 1e-6,
             "weight_decay": 0.0},
        ]
        meta = {**BASE_META, "step": t0, "param_groups": groups}
        state = {"payload": {
            "w": {"p": torch.randn(n), "q": torch.randn(n)},
            "m": {"p": torch.randn(n) * 0.1, "q": torch.randn(n) * 0.1},
            "v": {"p": torch.rand(n) * 0.01 + 1e-4,
                  "q": torch.rand(n) * 0.01 + 1e-4},
            "g": {"p": torch.randn(n) * 0.01, "q": torch.randn(n) * 0.01},
        }, "meta": meta}
        torch.save(state, os.path.join(td, f"step_{t0:06d}_state.pt"))

        catalog, names, offsets, N, state_files, g_files = soa.discover_trace(td)
        _meta, params, *_ = soa.build_branch_state(
            state_files[t0], names, torch.float64, catalog=catalog)
        by_name = {p["name"]: p for p in params}
        assert by_name["p"]["group"] == 0 and by_name["q"]["group"] == 1
        # per-group values take precedence over the global meta scalars
        assert by_name["p"]["b1"] == 0.9 and by_name["p"]["b2"] == 0.95
        assert by_name["p"]["eps"] == 1e-8 and by_name["p"]["wd"] == 0.1
        assert by_name["q"]["b1"] == 0.5 and by_name["q"]["b2"] == 0.9
        assert by_name["q"]["eps"] == 1e-6 and by_name["q"]["wd"] == 0.0

        # exact per-group lr when the group is known; legacy mean fallback
        # when it is not
        gmeta = {"lr": [1e-4, 3e-4]}
        assert soa._lr_for_step(gmeta, 0) == 1e-4
        assert soa._lr_for_step(gmeta, 1) == 3e-4
        assert abs(soa._lr_for_step(gmeta, None) - 2e-4) < 1e-12

        # legacy-shaped catalog (no param_group) -> global fallback
        torch.save({"catalog": {"p": {"shape": [n]}, "q": {"shape": [n]}}},
                   os.path.join(td, "catalog.pt"))
        catalog2, names2, _, _, state_files2, _ = soa.discover_trace(td)
        _meta2, params2, *_ = soa.build_branch_state(
            state_files2[t0], names2, torch.float64, catalog=catalog2)
        for p in params2:
            assert p["group"] is None
            assert (p["b1"], p["b2"], p["eps"], p["wd"]) == (B1, B2, EPS, WD)
    print("PASS test_per_group_hyperparams_resolve")


def test_select_topk_flat_contract():
    """Direct pin for the selection primitives: the k-th-largest threshold
    plus index-ordered tie-fill (the FIRST `need` tied coords by flat index
    win when more candidates are tied than slots remain) and the documented
    rejections — the cross-rank/cross-run determinism contract."""
    # hand-built tie layer with MORE tied candidates than slots
    t = torch.zeros(16)
    t[:8] = torch.arange(1, 9, dtype=torch.float32)  # strictly above the tie
    t[8:] = 0.5                                      # 8 tied at the k-th value
    mask, thresh = soa.select_topk_flat(t, 12)
    want = torch.zeros(16, dtype=torch.bool)
    want[:8] = True
    want[8:12] = True
    assert torch.equal(mask, want) and thresh == 0.5

    # random vector with an injected oversized tie at the k-th value
    torch.manual_seed(8)
    t = torch.rand(200)
    k = 50
    v = float(torch.kthvalue(t, t.numel() - k + 1).values)
    t[100:160] = v
    mask, thresh = soa.select_topk_flat(t, k)
    assert thresh == v
    need = k - int((t > v).sum())
    tied = (t == v).nonzero(as_tuple=True)[0]
    want = t > v
    want[tied[:need]] = True
    assert torch.equal(mask, want)

    # k=1 takes the first of tied maxima; k=N selects everything
    mask, thresh = soa.select_topk_flat(torch.tensor([0.1, 5.0, 3.0, 5.0]), 1)
    assert torch.equal(mask, torch.tensor([False, True, False, False]))
    assert thresh == 5.0
    t = torch.rand(7)
    mask, thresh = soa.select_topk_flat(t, 7)
    assert bool(mask.all()) and thresh == float(t.min())

    # all-equal / all-zero scores: exact index-ordered prefix
    mask, thresh = soa.select_topk_flat(torch.full((10,), 2.5), 4)
    want = torch.zeros(10, dtype=torch.bool)
    want[:4] = True
    assert torch.equal(mask, want) and thresh == 2.5
    mask, thresh = soa.select_topk_flat(torch.zeros(6), 3)
    want = torch.zeros(6, dtype=torch.bool)
    want[:3] = True
    assert torch.equal(mask, want) and thresh == 0.0

    # documented rejections
    try:
        soa.select_topk_flat(torch.zeros(5), 6)
        raise AssertionError("k > N accepted")
    except ValueError as e:
        assert "k=6 > N=5" in str(e), e
    try:
        soa.select_topk_flat(torch.tensor([0.5, -0.1]), 1)
        raise AssertionError("negative scores accepted")
    except ValueError as e:
        assert "non-negative" in str(e), e
    print("PASS test_select_topk_flat_contract")


def test_cli_rejects_invalid_options():
    """The CLI gate rejects each invalid option value with the option named,
    before any trace access (deep ZeroDivisionErrors and silently empty
    reports are not acceptable diagnostics)."""
    base = ["--trace-dir", "/nonexistent-sga-trace-dir"]
    cases = [
        (["--replay-mode", "cycle", "--cycle-len", "0"], "--cycle-len"),
        (["--chunk-mb", "0"], "--chunk-mb"),
        (["--window", "0"], "--window"),
        (["--horizons", "0"], "--horizons"),
        (["--ratios", "nan"], "--ratios"),
        (["--branch-point", ","], "--branch-point"),
        (["--sticky-lambda", "-0.1"], "--sticky-lambda"),
    ]
    for extra, opt in cases:
        try:
            soa.main(base + extra)
        except ValueError as e:
            assert opt in str(e), (opt, str(e))
        else:
            raise AssertionError(f"{opt}: invalid value accepted")
    # sticky-lambda NaN is rejected by the same finiteness arm
    try:
        soa.main(base + ["--sticky-lambda", "nan"])
    except ValueError as e:
        assert "--sticky-lambda" in str(e), e
    else:
        raise AssertionError("--sticky-lambda nan accepted")
    print("PASS test_cli_rejects_invalid_options")


def test_print_summary_zero_distortion():
    """lr=0: the dense arm never moves -> den == 0 -> distortion None (the
    JSON path carries null). print_summary must render the N/A cell
    end-to-end instead of crashing after all replay work is done."""
    torch.manual_seed(9)
    t0, W, n = 100, 3, 8
    pre = [torch.randn(n) * 0.01 for _ in range(W)]
    lrs = [0.0] * W
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as td:
        _make_trace(td, pre, lrs, 0.0, t0, n)
        with redirect_stdout(buf):
            report = soa.main(["--trace-dir", td, "--branch-point", str(t0),
                               "--window", str(W), "--ratios", "50",
                               "--horizons", "2"])
    met = report["branch_results"][str(t0)]["per_ratio"]["50.0"]["scores"]
    for m in met.values():
        assert m["distortion"] == {"2": None}
    assert "N/A" in buf.getvalue(), "printer must render the None cell"
    print("PASS test_print_summary_zero_distortion")


def main():
    test_T1_is_dense()
    test_no_fold_within_window()
    test_multicycle_with_clipping()
    test_boundary_reselect_changes_mask()
    test_cycle_score_formulas()
    test_sticky_semantics()
    test_determinism()
    test_per_group_hyperparams_resolve()
    test_select_topk_flat_contract()
    test_cli_rejects_invalid_options()
    test_print_summary_zero_distortion()
    print("\nall replay-cycle tests passed")


if __name__ == "__main__":
    main()
