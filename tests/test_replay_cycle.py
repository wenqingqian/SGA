#!/usr/bin/env python3
"""Unit tests for the cycle-mode (residual + multi-refresh) replay oracle.

Run from anywhere (CPU only, torch + stdlib):

    python SGA/tests/test_replay_cycle.py

The cycle semantics (offline twin of the online cyclic semantics) are served
by the fused single-pass engine (sga.replay_fused, --replay-mode cycle):
masked steps communicate ``q = g ⊙ sel`` (clipped by the masked norm) and
accumulate the unselected pre-clip gradient into a residual; every
``cycle_len`` steps a full-sync boundary commits ``q = g + residual``
(clipped by the folded norm), clears the residual, and re-selects the next
cycle's mask from the boundary score computed on the pre-commit state.
The classic in-process cycle engine (sparse_replay_cycle) was removed; this
suite's FROM-SPEC REFERENCE (reference_cycle below) is the sole independent
spec anchor for the semantics — the fused engine is validated against it
(the pre-deletion classic engine is reproducible from the old tag).

Tests (ported from the legacy oracle suite, importing sga.replay):
1. T=1 degenerates to the dense trajectory (distortion exactly 0).
2. T > window: residual accumulates but never folds (fixed mask from the
   branch selection) — matches an independent reference loop.
3. Multi-cycle hand example (T=2, 6 steps, clipping active): w/m/v and the
   residual/fold math match a from-spec reference.
4. Boundary re-selection actually changes the support when momentum flips
   (the frozen-mask counterfactual diverges; the engine follows the
   re-selecting reference).
5. Branch-point score formulas (registry, fp64) match direct fp64 references
   for every registered score, including the zero-moment t0=0 edge.
6. Determinism: two identical runs produce bitwise-equal distortion numbers.
7. Per-group hyperparameter resolution (new-trace additive extension with
   legacy global fallback).
8. Dense-artifact cache (--dense-cache): miss writes, hit reproduces the
   whole report, a tampered cache ident fails loudly.
9. bf16 screening copies (--trace-dtype bf16): bitwise no-op on fp32 traces;
   on bf16 copies the upcast path stays within the quantization band; a bf16
   payload replayed with the upcast off fails loudly.
10. sga.trace_convert: one-way fp32->bf16 copy contract (payloads bf16,
    meta/catalog/states verbatim), step-range selection, rejections,
    --overwrite removes stale out-of-range files.
11. --self-check provenance: --trace-dtype is threaded into the dense replay
    and echoed in the report; bf16 payloads without the flag are rejected.
12. --norm-dtype fp32: clip-inactive runs are bitwise identical, clip-active
    runs stay within the fp32-norm band.
"""
import argparse
import glob
import io
import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga import replay as soa
from sga import trace_convert as tc


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
    if name == "abs_m":
        return m.abs()
    mg = (b1 * m + (1.0 - b1) * g) / bc1
    vg = (b2 * v + (1.0 - b2) * g * g) / bc2
    ug = mg / (torch.sqrt(vg) + EPS)
    if name == "update":
        return (WD * w + ug).abs()
    u0 = (b1 * m / bc1) / (torch.sqrt(b2 * v / bc2) + EPS)
    return (ug - u0).abs()


def _topk_mask(sc, k):
    # index-ordered tie-fill — the selection contract pinned by
    # test_select_topk_flat_contract: among coords tied at the k-th value,
    # the FIRST ones by flat index win. torch.topk does NOT guarantee this
    # on ties (it returns a different tie set on all-equal scores), so the
    # reference implements the contract itself.
    v = torch.kthvalue(sc, sc.numel() - k + 1).values
    mask = sc > v
    need = k - int(mask.sum())
    if need:
        tied = (sc == v).nonzero(as_tuple=True)[0]
        mask[tied[:need]] = True
    return mask


def reference_cycle(w0, m0, v0, logged_grads, grad_norms, lrs, clip, t0,
                    score_name, k, T, reselect=True, branch_grad=None):
    """From-spec reference: returns (dense_w_at_each_step, sparse_w_at_each_step).

    ``logged_grads`` are the POST-CLIP gradients as logged in the trace
    (what the dense run committed); ``grad_norms`` are the per-step norms
    the trace meta carries. The sparse path reconstructs the pre-clip
    gradient via g / coef_dense with coef_dense = min(1, clip /
    (grad_norm + 1e-6)) — the replay contract; the reference consumes the
    trace inputs, not the idealized pre-clip gradients.
    ``reselect=False`` freezes the mask after the branch selection — the
    counterfactual used to prove that boundary re-selection is
    load-bearing (test_boundary_reselect_changes_mask).
    ``branch_grad`` is the gradient the branch STATE dump carries (what
    the engine's branch-point selection consumes)."""
    # dense trajectory: the logged (already clipped) gradient as committed
    md, vd, wdt = m0.clone(), v0.clone(), w0.clone()
    dense_ws = []
    for si, g in enumerate(logged_grads):
        count = t0 + si + 1
        wdt, md, vd = _adamw(wdt, md, vd, g, lrs[si], count)
        dense_ws.append(wdt.clone())
    # sparse cycle trajectory
    ms, vs, ws = m0.clone(), v0.clone(), w0.clone()
    res = torch.zeros_like(w0)
    q0 = branch_grad
    # the branch-point selection consumes the state dump's g as-is (it is
    # already post-clip); fp64 math / fp32 storage per the registry
    # contract, index-ordered tie-fill per the select contract
    sel = _topk_mask(_ref_score(score_name, ws, ms, vs, q0, t0 + 1)
                     .float(), k)
    sparse_ws = []
    for si, g_logged in enumerate(logged_grads):
        count = t0 + si + 1
        coef_dense = min(1.0, clip / (grad_norms[si] + 1.0e-6)) \
            if clip > 0.0 else 1.0
        g = g_logged / coef_dense
        if si % T == 0:
            q_pre = g + res
            coef = 1.0
            if clip > 0.0:
                coef = min(1.0, clip / (float(q_pre.norm()) + 1.0e-6))
            q = q_pre * coef
            if reselect and si > 0 and si + 1 < len(logged_grads):
                sc = _ref_score(score_name, ws, ms, vs, q, count)
                sel = _topk_mask(sc.float(), k)
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
    logged_all, norms = [], []
    for si, pre in enumerate(pre_grads):
        step = t0 + si
        norm = float(pre.norm())
        norms.append(norm)
        coef = 1.0
        if clip > 0.0:
            coef = min(1.0, clip / (norm + 1.0e-6))
        logged = (pre * coef).float()
        logged_all.append(logged)
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
    return name, w0, m0, v0, logged_all, norms


def _run_module(tmpdir, name, w0, m0, v0, t0, W, score_name, ratio_pct, T,
                replay_dtype=torch.float64, upcast_bf16=False,
                norm_fp32=False):
    """Run the fused engine (single shard, inline, cpu) for one score x
    ratio arm via run_fused_branch — the direct call returns the RAW branch
    dict (no 8-decimal sanitize), so num_at keeps full fp64 precision — and
    return num_at {h: num} as distortion[h] * den[h]."""
    from sga import replay_fused

    catalog, names, offsets, N, state_files, g_files = soa.discover_trace(
        tmpdir)
    ctx = {"trace_dir": os.path.abspath(tmpdir), "names": names,
           "offsets": offsets, "N": N, "state_files": state_files,
           "g_files": g_files, "catalog": catalog}
    args = argparse.Namespace(
        replay_mode="cycle", self_check=None,
        dense_cache=os.path.join(tmpdir, "dense_cache"),
        score_names=[score_name], fused_shards=1, fused_max_configs=12,
        ratios=[ratio_pct], horizons=list(range(1, W + 1)), window=W,
        cycle_len=T,
        replay_dtype=("fp64" if replay_dtype is torch.float64 else "fp32"),
        trace_dtype=("bf16" if upcast_bf16 else "fp32"),
        norm_dtype=("fp32" if norm_fp32 else "fp64"),
        chunk_mb=1, device="cpu", seed=1234, verbose=False,
    )
    br = replay_fused.run_fused_branch(args, ctx, t0)
    den = br["replay"]["den_norm_at_horizon"]
    dist = br["per_ratio"][str(ratio_pct)]["scores"][score_name]["distortion"]
    return {int(h): (None if dist[str(h)] is None
                     else dist[str(h)] * den[str(h)])
            for h in range(1, W + 1)}


def _ref_num_at(w0, m0, v0, logged, norms, lrs, clip, t0, score_name,
                ratio_pct, T):
    k = max(1, int(round(ratio_pct / 100.0 * w0.numel())))
    dense_ws, sparse_ws = reference_cycle(
        w0.double(), m0.double(), v0.double(),
        [g.double() for g in logged], norms, lrs, clip, t0, score_name, k, T,
        branch_grad=logged[0].double())
    return {h: float((sparse_ws[h - 1] - dense_ws[h - 1]).norm())
            for h in range(1, len(logged) + 1)}


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
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 1.0, t0, n)
        num_at = _run_module(td, name, w0, m0, v0, t0, W, "coi", 50.0, 1)
    # exactness relies on coef_dense == 1.0 (clip inactive in this fixture);
    # with active clipping T=1 is dense only up to the reconstruction band
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
        name, w0, m0, v0, logged, norms = _make_trace(td, pre, lrs, 0.0, t0, n)
        for score in soa.all_score_names():
            num_at = _run_module(td, name, w0, m0, v0, t0, W, score, 25.0,
                                 T=W + 1)
            ref = _ref_num_at(w0, m0, v0, logged, norms, lrs, 0.0, t0, score, 25.0,
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
        name, w0, m0, v0, logged, norms = _make_trace(td, pre, lrs, clip, t0, n)
        for score in soa.all_score_names():
            num_at = _run_module(td, name, w0, m0, v0, t0, W, score, 50.0, 2)
            ref = _ref_num_at(w0, m0, v0, logged, norms, lrs, clip, t0, score, 50.0, 2)
            for h in range(1, W + 1):
                assert _close(num_at[h], ref[h], 1e-9), (
                    f"{score} d@{h}: module {num_at[h]} vs ref {ref[h]}")
    print("PASS test_multicycle_with_clipping")


def test_boundary_reselect_changes_mask():
    """A momentum flip between boundaries must change the selected support.
    The fused engine exposes no selection hook, so this is pinned
    behaviorally: the re-selecting from-spec reference and the frozen-mask
    counterfactual (reselect=False) diverge on this fixture, and the engine
    tracks the re-selecting one."""
    torch.manual_seed(3)
    t0, W, n = 1600, 5, 4
    # step 0 pushes coords 0-1; steps 1..4 push coords 2-3 hard, so the
    # abs_g/update/coi supports must rotate at the si=2 boundary (T=2).
    pre = [torch.tensor([1.0, 1.0, 0.01, 0.01])]
    pre += [torch.tensor([0.01, 0.01, 1.0, 1.0]) for _ in range(W - 1)]
    lrs = [5e-5] * W
    score, ratio_pct, T = "abs_g", 50.0, 2
    k = max(1, int(round(ratio_pct / 100.0 * n)))
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, logged, norms = _make_trace(td, pre, lrs, 0.0, t0, n)
        # re-selecting vs frozen references on the real fixture state
        _, resel_ws = reference_cycle(w0.double(), m0.double(), v0.double(),
                                      [g.double() for g in logged], norms,
                                      lrs, 0.0, t0, score, k, T,
                                      branch_grad=logged[0].double())
        _, frozen_ws = reference_cycle(w0.double(), m0.double(), v0.double(),
                                       [g.double() for g in logged], norms,
                                       lrs, 0.0, t0, score, k, T,
                                       reselect=False,
                                       branch_grad=logged[0].double())
        # the two references must genuinely diverge (masks rotated)
        assert any(not torch.equal(a, b)
                   for a, b in zip(resel_ws, frozen_ws)), \
            "fixture failed: frozen-mask and re-selecting references agree"
        # the engine follows the re-selecting reference, not the frozen one
        num_at = _run_module(td, name, w0, m0, v0, t0, W, score, ratio_pct, T)
        ref = _ref_num_at(w0, m0, v0, logged, norms, lrs, 0.0, t0, score, ratio_pct, T)
        for h in range(1, W + 1):
            assert _close(num_at[h], ref[h], 1e-9), (
                f"d@{h}: module {num_at[h]} vs re-selecting ref {ref[h]}")
    print("PASS test_boundary_reselect_changes_mask")


def test_branch_score_formulas():
    """compute_scores_flat (the registry formulas, fp64 math / fp32 storage)
    vs direct fp64 references for every registered score, including the
    zero-moment t0=0 edge. The cycle-boundary re-selection path consumes the
    same registry formulas (its per-param call site), pinned elementwise by
    tests/test_plugins_unit.py::test_score_formulas_match_fp64_oracles."""
    torch.manual_seed(4)
    n = 64
    w = torch.randn(n, dtype=torch.float32)
    m = torch.randn(n, dtype=torch.float32) * 0.1
    v = torch.rand(n, dtype=torch.float32) * 0.01 + 1e-4
    g = torch.randn(n, dtype=torch.float32) * 0.01
    count = 1234  # upcoming commit; the branch state dump carries step=t0

    def _flats(step):
        meta = {**BASE_META, "step": step}
        return soa.compute_scores_flat(
            [("p", w)], {"p": m}, {"p": v}, {"p": g}, {"p": 0}, n, meta,
            chunk_mb=1, device="cpu", update_includes_wd=True)

    flats = _flats(count - 1)
    for name in soa.all_score_names():
        got = flats[name]
        want = _ref_score(name, w.double(), m.double(), v.double(),
                          g.double(), count)
        # fp32 storage rounding only: max abs error within 1e-6 of the peak
        err = (got.double() - want).abs().max()
        scale = want.abs().max().clamp(min=1e-30)
        assert err / scale < 1e-6, f"{name}: {float(err / scale):.3e}"

    # zero-state edge (t0=0, state_absent: zero moments, upcoming commit 1)
    m, v = torch.zeros_like(m), torch.zeros_like(v)
    flats0 = _flats(0)
    for name in soa.all_score_names():
        want = _ref_score(name, w.double(), torch.zeros(n, dtype=torch.float64),
                          torch.zeros(n, dtype=torch.float64), g.double(), 1)
        err = (flats0[name].double() - want).abs().max()
        scale = want.abs().max().clamp(min=1e-30)
        assert err / scale < 1e-6, f"{name} zero-state: {float(err / scale):.3e}"
    print("PASS test_branch_score_formulas")


def test_determinism():
    """Two identical cycle replays produce bitwise-equal distortion numbers."""
    torch.manual_seed(5)
    t0, W, n = 1600, 6, 32
    pre = [torch.randn(n) * 0.02 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 1.0, t0, n)
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
    # NaN/Inf would sit in the histogram's top bit-pattern bins and displace
    # real coordinates from the topk silently (min() does not catch NaN)
    for bad in (float("nan"), float("inf")):
        try:
            soa.select_topk_flat(torch.tensor([0.5, bad, 0.2]), 2)
            raise AssertionError(f"{bad} scores accepted")
        except ValueError as e:
            assert "finite" in str(e), e
    print("PASS test_select_topk_flat_contract")


def test_param_hp_group_index_out_of_range():
    """A catalog param_group index outside the step meta's param_groups is a
    trace-integrity violation and must fail with a named error (a raw
    IndexError from groups[gi] would be cryptic and point nowhere)."""
    meta = {
        "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "param_groups": [
            {"lr": 1e-3, "weight_decay": 0.0,
             "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_eps": 1e-8},
            {"lr": 2e-3, "weight_decay": 0.1,
             "adam_beta1": 0.85, "adam_beta2": 0.95, "adam_eps": 1e-7},
        ],
    }
    b1, b2, eps, wd, gi = soa._param_hp(meta, {"param_group": 1})
    assert (b1, b2, eps, wd, gi) == (0.85, 0.95, 1e-7, 0.1, 1)
    for bad_gi in (-1, 2):
        try:
            soa._param_hp(meta, {"param_group": bad_gi})
            raise AssertionError(f"group index {bad_gi} accepted")
        except ValueError as e:
            assert "outside the step meta param_groups" in str(e), e
    print("PASS test_param_hp_group_index_out_of_range")


def test_cli_rejects_invalid_options():
    """The CLI gate rejects each invalid option value with the option named,
    before any trace access (deep ZeroDivisionErrors and silently empty
    reports are not acceptable diagnostics)."""
    base = ["--trace-dir", "/nonexistent-sga-trace-dir"]
    cases = [
        (["--replay-mode", "cycle", "--cycle-len", "0"], "--cycle-len"),
        # the loud post-consolidation gate: classic cycle-without-cache was
        # removed; cycle now always routes to the fused engine, which
        # consumes --dense-cache (populated on the first run)
        (["--replay-mode", "cycle", "--branch-point", "5"], "--dense-cache"),
        # --fused is a deprecated no-op, but only where it ever made sense
        (["--fused"], "--fused is deprecated"),
        (["--chunk-mb", "0"], "--chunk-mb"),
        (["--window", "0"], "--window"),
        (["--horizons", "0"], "--horizons"),
        (["--ratios", "nan"], "--ratios"),
        (["--branch-point", ","], "--branch-point"),
        (["--dense-cache", "/tmp/x", "--self-check", "1", "2"],
         "--dense-cache"),
    ]
    for extra, opt in cases:
        try:
            soa.main(base + extra)
        except ValueError as e:
            assert opt in str(e), (opt, str(e))
        else:
            raise AssertionError(f"{opt}: invalid value accepted")
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
    # pure zerofill run: the CLI echo stays False and so does the
    # effective-engine flag (only cycle routes to the fused engine)
    assert report["config"]["fused"] is False
    assert report["config"]["fused_effective"] is False
    assert "N/A" in buf.getvalue(), "printer must render the None cell"
    print("PASS test_print_summary_zero_distortion")


def test_dense_cache_roundtrip():
    """--dense-cache: run 1 writes the dense artifacts (miss), run 2 loads
    them (hit) and the whole JSON report is identical; a tampered ident
    inside the cache file fails loudly instead of serving wrong numbers."""
    torch.manual_seed(10)
    t0, W, n = 300, 5, 16
    pre = [torch.randn(n) * 0.02 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        _make_trace(td, pre, lrs, 1.0, t0, n)
        argv = ["--trace-dir", td, "--branch-point", str(t0),
                "--window", str(W), "--ratios", "25,50", "--horizons", "2,5",
                "--replay-mode", "cycle", "--cycle-len", "2",
                "--score-names", "abs_g,update",
                "--fused-shards", "1",
                "--dense-cache", os.path.join(td, "dense_cache")]
        buf = io.StringIO()
        with redirect_stdout(buf):
            r1 = soa.main(argv + ["--out", os.path.join(td, "r1.json")])
        b1 = r1["branch_results"][str(t0)]["replay"]
        assert b1["dense_cache_hit"] is False, b1["dense_cache_hit"]
        with redirect_stdout(buf):
            r2 = soa.main(argv + ["--out", os.path.join(td, "r2.json")])
        b2 = r2["branch_results"][str(t0)]["replay"]
        assert b2["dense_cache_hit"] is True, b2["dense_cache_hit"]
        # everything but the hit marker itself must round-trip identically
        # (compare via json: the synthetic state meta carries a NaN
        # grad_norm, and NaN != NaN under plain dict equality)
        b1.pop("dense_cache_hit")
        b2.pop("dense_cache_hit")
        assert json.dumps(r1) == json.dumps(r2), \
            "cache HIT report differs from MISS report"

        # tampered ident -> loud failure, never silent wrong numbers
        cpath = glob.glob(os.path.join(td, "dense_cache", "dense_*.pt"))[0]
        blob = torch.load(cpath, map_location="cpu", weights_only=False)
        blob["ident"] = "tampered"
        torch.save(blob, cpath)
        try:
            with redirect_stdout(buf):
                soa.main(argv + ["--out", os.path.join(td, "r3.json")])
            raise AssertionError("tampered cache ident accepted")
        except ValueError as e:
            assert "ident" in str(e), e
    print("PASS test_dense_cache_roundtrip")


def test_dense_cache_ident_engine_fence():
    """cpu and dev dense engines must never share a cache file: dense_replay
    quantizes horizon diffs to fp32 before squaring, dense_replay_dev does
    not — a shared ident would silently serve wrong denominators across
    device migrations of the same trace."""
    base = ("trace", 0, [1, 2], [1, 2], torch.float32, "fp32")
    i_cpu = soa._dense_cache_ident(*base, "cpu")
    i_dev = soa._dense_cache_ident(*base, "dev")
    assert i_cpu != i_dev
    assert (soa._dense_cache_path("cache", i_cpu)
            != soa._dense_cache_path("cache", i_dev))
    print("PASS test_dense_cache_ident_engine_fence")


def test_trace_dtype_bf16():
    """--trace-dtype bf16: on fp32 traces the flag is a bitwise no-op; on
    bf16 screening copies the upcast path keeps the distortion within the
    bf16 quantization band of the fp32 trajectory."""
    torch.manual_seed(11)
    t0, W, n = 500, 6, 32
    pre = [torch.randn(n) * 0.05 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 1.0, t0, n)
        a = _run_module(td, name, w0, m0, v0, t0, W, "coi", 30.0, 2)
        b = _run_module(td, name, w0, m0, v0, t0, W, "coi", 30.0, 2,
                        upcast_bf16=True)
        assert a == b, "--trace-dtype bf16 must be a no-op on fp32 traces"

        bf = os.path.join(td, "bf16")
        os.makedirs(bf)
        shutil.copy2(os.path.join(td, "catalog.pt"),
                     os.path.join(bf, "catalog.pt"))
        shutil.copy2(os.path.join(td, f"step_{t0:06d}_state.pt"),
                     os.path.join(bf, f"step_{t0:06d}_state.pt"))
        for si in range(W):
            f = os.path.join(td, f"step_{t0 + si:06d}_g.pt")
            d = torch.load(f, map_location="cpu", weights_only=False)
            d["payload"] = {k: v.to(torch.bfloat16)
                            for k, v in d["payload"].items()}
            torch.save(d, os.path.join(bf, f"step_{t0 + si:06d}_g.pt"))
        # the bf16 gate: replaying a bf16 payload with the upcast off is a
        # hard error naming the flag (the sparse paths would divide by
        # coef_dense in bf16), never a silently degraded run
        try:
            _run_module(bf, name, w0, m0, v0, t0, W, "coi", 30.0, 2,
                        upcast_bf16=False)
            raise AssertionError("bf16 payload accepted with upcast off")
        except ValueError as e:
            assert "--trace-dtype" in str(e), e
        c = _run_module(bf, name, w0, m0, v0, t0, W, "coi", 30.0, 2,
                        upcast_bf16=True)
        for h in a:
            assert (a[h] == 0.0) == (c[h] == 0.0), f"d@{h}: {a[h]} vs {c[h]}"
            if a[h] != 0.0:
                rel = abs(c[h] - a[h]) / abs(a[h])
                assert rel < 0.05, f"bf16 copy d@{h}: rel err {rel:.3e}"
    print("PASS test_trace_dtype_bf16")


def test_trace_convert_module():
    """sga.trace_convert: one-way fp32->bf16 copy — g payloads become bf16
    with meta preserved and values exactly the bf16 rounding of the source,
    states are copied verbatim (fp32) and the catalog verbatim except its
    self-describing g_dtype flip, the step range selects which
    g/state files land in dst, an already-bf16 source converts no-op but
    warns loudly, and dst==src / non-empty dst are rejected."""
    torch.manual_seed(12)
    t0, W, n = 700, 4, 8
    pre = [torch.randn(n) * 0.05 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 1.0, t0, n)
        dst = os.path.join(td, "dst")
        buf = io.StringIO()
        with redirect_stdout(buf):
            tc.main(["--src", td, "--dst", dst])

        src_g = torch.load(os.path.join(td, f"step_{t0:06d}_g.pt"),
                           map_location="cpu", weights_only=False)
        dst_g = torch.load(os.path.join(dst, f"step_{t0:06d}_g.pt"),
                           map_location="cpu", weights_only=False)
        assert dst_g["payload"][name].dtype == torch.bfloat16
        assert dst_g["meta"] == src_g["meta"]
        assert torch.equal(dst_g["payload"][name],
                           src_g["payload"][name].to(torch.bfloat16))
        dst_st = torch.load(os.path.join(dst, f"step_{t0:06d}_state.pt"),
                            map_location="cpu", weights_only=False)
        assert dst_st["payload"]["w"][name].dtype == torch.float32
        assert torch.equal(dst_st["payload"]["w"][name], w0)
        src_cat = torch.load(os.path.join(td, "catalog.pt"),
                             map_location="cpu", weights_only=False)
        dst_cat = torch.load(os.path.join(dst, "catalog.pt"),
                             map_location="cpu", weights_only=False)
        # catalog verbatim except g_dtype, which flips to bf16 so the
        # converted dir is self-describing (same shape a runtime bf16
        # dump's collector writes)
        assert dst_cat == {**src_cat, "g_dtype": "bf16"}
        assert sorted(f for f in os.listdir(dst) if f.endswith("_g.pt")) == \
            sorted(f"step_{t0 + si:06d}_g.pt" for si in range(W))

        # re-converting an already-bf16 source is a correct no-op (the
        # per-tensor guard skips non-fp32 payloads) but must say so loudly
        dst3 = os.path.join(td, "dst3")
        buf3 = io.StringIO()
        with redirect_stdout(buf3):
            tc.main(["--src", dst, "--dst", dst3])
        assert "already bf16" in buf3.getvalue()
        g3 = torch.load(os.path.join(dst3, f"step_{t0:06d}_g.pt"),
                        map_location="cpu", weights_only=False)
        assert torch.equal(g3["payload"][name], dst_g["payload"][name])

        # step range: only g files in [from, to) land in dst2 (the t0 state
        # is outside the range and must not be copied)
        dst2 = os.path.join(td, "dst2")
        with redirect_stdout(buf):
            tc.main(["--src", td, "--dst", dst2,
                     "--from-step", str(t0 + 2), "--to-step", str(t0 + 4)])
        assert sorted(f for f in os.listdir(dst2) if f.endswith("_g.pt")) == \
            [f"step_{t0 + 2:06d}_g.pt", f"step_{t0 + 3:06d}_g.pt"]
        assert not [f for f in os.listdir(dst2) if f.endswith("_state.pt")]

        # --overwrite is deterministic: re-converting a narrower range into
        # dst removes the previous selection's out-of-range files (the dst is
        # a trace, not a manifest — discover_trace must not see the union of
        # two step sets)
        with redirect_stdout(buf):
            tc.main(["--src", td, "--dst", dst, "--overwrite",
                     "--from-step", str(t0 + 2), "--to-step", str(t0 + 4)])
        assert sorted(f for f in os.listdir(dst) if f.endswith("_g.pt")) == \
            [f"step_{t0 + 2:06d}_g.pt", f"step_{t0 + 3:06d}_g.pt"]
        assert not [f for f in os.listdir(dst) if f.endswith("_state.pt")], \
            "overwrite must drop stale state files outside the new range"
        assert os.path.isfile(os.path.join(dst, "catalog.pt"))

        for bad, why in (
            (["--src", td, "--dst", dst], "non-empty dst accepted"),
            (["--src", td, "--dst", td], "dst == src accepted"),
        ):
            try:
                tc.main(bad)
                raise AssertionError(why)
            except ValueError:
                pass
    print("PASS test_trace_convert_module")


def test_self_check_trace_dtype():
    """--self-check provenance: --trace-dtype is threaded into the dense
    replay and echoed in the report (on an fp32 trace the numbers are
    identical with and without the flag — only the echoed field differs);
    on a bf16 copy the self-check without the flag fails loudly (the bf16
    gate) instead of silently ignoring it, and with the flag it reports
    trace_dtype "bf16"."""
    torch.manual_seed(13)
    t0, W, n = 900, 5, 16
    pre = [torch.randn(n) * 0.02 for _ in range(W)]
    lrs = [5e-5] * W
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 1.0, t0, n)
        # self-check compares against the real state dump at TO_STEP
        torch.save({"payload": {"w": {name: w0}, "m": {name: m0},
                                "v": {name: v0}, "g": {name: torch.randn(n)}},
                    "meta": {**BASE_META, "step": t0 + W}},
                   os.path.join(td, f"step_{t0 + W:06d}_state.pt"))
        argv = ["--trace-dir", td, "--self-check", str(t0), str(t0 + W)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            r_fp = soa.main(list(argv))
            r_up = soa.main(argv + ["--trace-dtype", "bf16"])
        sc_fp, sc_up = r_fp["self_check"], r_up["self_check"]
        assert sc_fp.pop("trace_dtype") == "fp32"
        assert sc_up.pop("trace_dtype") == "bf16"
        # fp32 payloads: the flag changes nothing but the echoed field
        # (compare via json: NaN != NaN under plain dict equality)
        assert json.dumps(sc_fp) == json.dumps(sc_up), (
            "--trace-dtype bf16 must be a no-op on fp32 traces")

        # bf16 screening copy of the same window (states/catalog verbatim)
        bf = os.path.join(td, "bf16")
        os.makedirs(bf)
        shutil.copy2(os.path.join(td, "catalog.pt"),
                     os.path.join(bf, "catalog.pt"))
        for st in (t0, t0 + W):
            shutil.copy2(os.path.join(td, f"step_{st:06d}_state.pt"),
                         os.path.join(bf, f"step_{st:06d}_state.pt"))
        for si in range(W):
            f = os.path.join(td, f"step_{t0 + si:06d}_g.pt")
            d = torch.load(f, map_location="cpu", weights_only=False)
            d["payload"] = {k: v.to(torch.bfloat16)
                            for k, v in d["payload"].items()}
            torch.save(d, os.path.join(bf, f"step_{t0 + si:06d}_g.pt"))

        argv_bf = ["--trace-dir", bf, "--self-check", str(t0), str(t0 + W)]
        try:
            with redirect_stdout(buf):
                soa.main(list(argv_bf))
            raise AssertionError("bf16 self-check accepted without "
                                 "--trace-dtype bf16")
        except ValueError as e:
            assert "--trace-dtype" in str(e), e
        with redirect_stdout(buf):
            r_bf = soa.main(argv_bf + ["--trace-dtype", "bf16"])
        assert r_bf["self_check"]["trace_dtype"] == "bf16"
    print("PASS test_self_check_trace_dtype")


def test_norm_dtype_fp32():
    """--norm-dtype fp32: with clipping inactive the norm is unused and the
    replay is bitwise identical; with active clipping the fp32-quantized
    pass-1 norms perturb the trajectory only at the band scale."""
    torch.manual_seed(13)
    t0, W, n = 900, 6, 24
    lrs = [5e-5] * W
    # clip inactive (clip=0): coef is structurally 1, the norm is never
    # consumed
    pre = [torch.randn(n) * 0.02 for _ in range(W)]
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 0.0, t0, n)
        a = _run_module(td, name, w0, m0, v0, t0, W, "abs_g", 25.0, 2,
                        norm_fp32=False)
        b = _run_module(td, name, w0, m0, v0, t0, W, "abs_g", 25.0, 2,
                        norm_fp32=True)
        assert a == b, "clip-inactive replay must not depend on norm dtype"
    # clip active (norms >> clip): fp32 quantization of the pass-1 norms
    # perturbs coef at ~1e-7 relative; the trajectory stays within the band
    pre = [torch.randn(n) * 0.5 for _ in range(W)]
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, _logged, _norms = _make_trace(td, pre, lrs, 0.3, t0, n)
        a = _run_module(td, name, w0, m0, v0, t0, W, "abs_g", 25.0, 2,
                        norm_fp32=False)
        b = _run_module(td, name, w0, m0, v0, t0, W, "abs_g", 25.0, 2,
                        norm_fp32=True)
        # the flag must actually fire on the clip-active arm: a no-op
        # norm_fp32 would produce b == a, trivially inside the band
        assert any(b[h] != a[h] for h in a), \
            "fp32-norm quantization must perturb the clipped trajectory"
        for h in a:
            assert (a[h] == 0.0) == (b[h] == 0.0), f"d@{h}: {a[h]} vs {b[h]}"
            if a[h] != 0.0:
                assert _close(b[h], a[h], 1e-4), (
                    f"fp32-norm d@{h} outside band: {b[h]} vs {a[h]}")
    print("PASS test_norm_dtype_fp32")


def test_replay_dtype_fp32():
    """--replay-dtype fp32 anchors the fused worker's fp32 branch: state,
    residual and q stay fp32 (no fp64 cast at the commit), so the
    trajectory matches the from-spec fp64 reference only to the fp32 band,
    and the flag visibly fires (the fp32 numbers differ from the fp64
    run's)."""
    torch.manual_seed(14)
    t0, W, n = 1100, 6, 24
    pre = [torch.randn(n) * 0.5 for _ in range(W)]  # norms >> clip -> active
    lrs = [5e-5] * W
    clip = 0.3
    with tempfile.TemporaryDirectory() as td:
        name, w0, m0, v0, logged, norms = _make_trace(td, pre, lrs, clip, t0, n)
        f64 = _run_module(td, name, w0, m0, v0, t0, W, "coi", 50.0, 2)
        f32 = _run_module(td, name, w0, m0, v0, t0, W, "coi", 50.0, 2,
                          replay_dtype=torch.float32)
        ref = _ref_num_at(w0, m0, v0, logged, norms, lrs, clip, t0, "coi",
                          50.0, 2)
        # the branch must actually fire: fp32 state/q/res move the numbers
        assert any(f32[h] != f64[h] for h in f64), \
            "--replay-dtype fp32 must not be a silent no-op"
        # sub-ulp distortions collapse to exact 0.0 in fp32 (h=1 here:
        # ref ~4e-13 -> 0.0); above that noise floor the fp32 trajectory
        # tracks the from-spec fp64 reference to ~5e-4 relative -> band 1e-3
        floor = 1.0e-8
        for h in f32:
            if ref[h] < floor:
                assert f32[h] < floor, \
                    f"d@{h}: {f32[h]} above the fp32 noise floor " \
                    f"(ref {ref[h]})"
            else:
                rel = abs(f32[h] - ref[h]) / abs(ref[h])
                assert rel < 1e-3, (
                    f"fp32 d@{h} outside band: {f32[h]} vs ref {ref[h]}")
    print("PASS test_replay_dtype_fp32")


def main():
    test_T1_is_dense()
    test_no_fold_within_window()
    test_multicycle_with_clipping()
    test_boundary_reselect_changes_mask()
    test_branch_score_formulas()
    test_determinism()
    test_per_group_hyperparams_resolve()
    test_select_topk_flat_contract()
    test_param_hp_group_index_out_of_range()
    test_cli_rejects_invalid_options()
    test_print_summary_zero_distortion()
    test_dense_cache_roundtrip()
    test_dense_cache_ident_engine_fence()
    test_trace_dtype_bf16()
    test_trace_convert_module()
    test_self_check_trace_dtype()
    test_norm_dtype_fp32()
    test_replay_dtype_fp32()
    print("\nall replay-cycle tests passed")


if __name__ == "__main__":
    main()
