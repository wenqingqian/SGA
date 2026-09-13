#!/usr/bin/env python3
"""Replay-kernel conformance matrix (sga/replay_kernels.py vs the torch
eager chains it replaces), CUDA-only — without a GPU or triton every
test here is a vacuous pass (the dispatch falls back to torch there by
design).

- per-score adversarial bitwise A/B (denormal / ±0 / 1±ulp neighbors /
  extreme magnitudes): update, coi, abs_g, abs_m;
- the _adam_step_inplace twin: m/v/w(/lab) bitwise across scalar
  settings and the lab on/off arms;
- select passes: the one-sweep top16 histogram (bin values plus the
  folded finite-guard count), candidate extraction (global index order,
  populated and always-empty bins), gt/eq masks + counts (tie-heavy at
  the threshold — the f32 threshold cast is load-bearing there);
- stack canaries: enable_fp_fusion=False really leaves a*b+c
  uncontracted (on vectors proven to expose an FMA), and the f64-upcast
  div/sqrt recipe matches torch on the ulp vectors including denormals;
- dispatch gates: arm()/arm_select() over the killswitch/device/dtype/
  contiguity matrix and formula() unknown-name -> None.

Run: python tests/test_replay_kernels.py  (or python -m pytest)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

HAVE = torch.cuda.is_available()
try:
    import triton
    import triton.language as tl
except ImportError:
    HAVE = False

from sga import replay_kernels as K
from sga.plugins.scores import get_score_spec
from sga.replay import _adam_step_inplace

DEV = "cuda"


def _adv(n, seed, nonneg=False):
    """Adversarial fp32 lanes: specials (±0, min/max normal, min
    denormal, 1e±20, near-tie constants) + random finite bit patterns."""
    g = np.random.default_rng(seed)
    parts = [np.array([0.0, -0.0, 1.0, -1.0, 2.0 ** -126, 2.0 ** -149,
                       2.0 ** 127, -(2.0 ** 127), 1e-8, 1e-20, 1e20,
                       0.9, 0.999], dtype=np.float32)]
    got = parts[0].size
    while got < n:
        b = g.integers(-2 ** 31, 2 ** 31, size=n, dtype=np.int32)
        f = b.view(np.float32)
        f = f[np.isfinite(f)]
        parts.append(f)
        got += f.size
    v = np.concatenate(parts)[:n]
    return np.abs(v) if nonneg else v


def _bits(t):
    return t.view(torch.int32)


def _bit_eq(a, b):
    return int((_bits(a) != _bits(b)).sum())


if HAVE:

    @triton.jit
    def _c_mad(a, b, c, out, n, BS: tl.constexpr):
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        x = tl.load(a + i, mask=mk)
        y = tl.load(b + i, mask=mk)
        z = tl.load(c + i, mask=mk)
        tl.store(out + i, x * y + z, mask=mk)

    @triton.jit
    def _c_div(a, b, out, n, BS: tl.constexpr):
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        x = tl.load(a + i, mask=mk)
        y = tl.load(b + i, mask=mk)
        tl.store(out + i,
                 (x.to(tl.float64) / y.to(tl.float64)).to(tl.float32),
                 mask=mk)

    @triton.jit
    def _c_sqrt(x, out, n, BS: tl.constexpr):
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        v = tl.load(x + i, mask=mk)
        tl.store(out + i, tl.sqrt(v.to(tl.float64)).to(tl.float32),
                 mask=mk)


def test_canary_no_contraction():
    """enable_fp_fusion=False must leave a*b+c uncontracted: the kernel
    matches torch's two-op chain bitwise, on vectors proven (by exact
    arithmetic on a sample) to expose an FMA on some lanes — otherwise
    the canary would pass vacuously on non-discriminating data."""
    if not HAVE:
        print("SKIP test_canary_no_contraction (no CUDA/triton)")
        return
    n = 1 << 18
    a = torch.from_numpy(_adv(n, 31)).to(DEV)
    b = torch.from_numpy(_adv(n, 32)).to(DEV)
    c = torch.from_numpy(_adv(n, 33)).to(DEV)
    out = torch.empty(n, dtype=torch.float32, device=DEV)
    _c_mad[(triton.cdiv(n, 1024),)](a, b, c, out, n, BS=1024,
                                    enable_fp_fusion=False)
    ref = a * b + c
    assert _bit_eq(out, ref) == 0, "fusion off still contracted"
    # discrimination proof on a sample: exact fma != two-op chain somewhere
    from fractions import Fraction
    idx = torch.randint(0, n, (4096,)).cpu()
    diff = 0
    for i in idx.tolist():
        exact = Fraction(float(a[i].item())) * Fraction(
            float(b[i].item())) + Fraction(float(c[i].item()))
        fma32 = np.float32(float(exact))
        if fma32.tobytes() != np.float32(ref[i].item()).tobytes():
            diff += 1
    assert diff > 0, "sample exposes no fma/two-op split — canary blind"
    print("PASS test_canary_no_contraction")


def test_canary_div_sqrt_recipe():
    """The f64-upcast div/sqrt recipe matches torch's correctly-rounded
    fp32 ops on every adversarial lane (denormals included — the lanes
    where the stack's plain f32 paths flush to zero)."""
    if not HAVE:
        print("SKIP test_canary_div_sqrt_recipe (no CUDA/triton)")
        return
    n = 1 << 18
    a = torch.from_numpy(_adv(n, 41)).to(DEV)
    b = torch.from_numpy(np.abs(_adv(n, 42)) + 2.0 ** -149).to(DEV)
    x = torch.from_numpy(_adv(n, 43, nonneg=True)).to(DEV)
    out = torch.empty(n, dtype=torch.float32, device=DEV)
    grid = (triton.cdiv(n, 1024),)
    _c_div[grid](a, b, out, n, BS=1024, enable_fp_fusion=False)
    assert _bit_eq(out, a / b) == 0, "f64-upcast div diverged from torch"
    _c_sqrt[grid](x, out, n, BS=1024, enable_fp_fusion=False)
    assert _bit_eq(out, torch.sqrt(x)) == 0, \
        "f64-upcast sqrt diverged from torch"
    print("PASS test_canary_div_sqrt_recipe")


def test_formula_bitwise():
    """Every dispatched score formula against its torch registry twin on
    adversarial state — bitwise, plus the unknown-name fallback."""
    if not HAVE:
        print("SKIP test_formula_bitwise (no CUDA/triton)")
        return
    n = 1 << 18
    w = torch.from_numpy(_adv(n, 51)).to(DEV)
    m = torch.from_numpy(_adv(n, 52)).to(DEV)
    v = torch.from_numpy(_adv(n, 53, nonneg=True)).to(DEV)
    g = torch.from_numpy(_adv(n, 54)).to(DEV)
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
    bc1c, bc2c = 1.0 - b1 ** 7, 1.0 - b2 ** 7
    for name in ("update", "coi", "abs_g", "abs_m"):
        kf = K.formula(name)
        assert kf is not None, name
        ref = get_score_spec(name).formula(
            w, m, v, g, b1, b2, eps, wd, bc1c, bc2c,
            update_includes_wd=True)
        out = kf(w, m, v, g, b1, b2, eps, wd, bc1c, bc2c)
        assert out.dtype == ref.dtype and out.shape == ref.shape, name
        assert _bit_eq(out, ref) == 0, f"{name} moved a bit"
    assert K.formula("no_such_score") is None
    print("PASS test_formula_bitwise")


def test_adam_bitwise():
    """The adam twin against _adam_step_inplace across scalar settings
    and lab arms — all four outputs bitwise."""
    if not HAVE:
        print("SKIP test_adam_bitwise (no CUDA/triton)")
        return
    n = 1 << 18
    for lr, wd, count, has_lab in ((1e-3, 0.01, 7, True),
                                   (3e-4, 0.01, 1, True),
                                   (1e-3, 0.0, 100, False)):
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1c, bc2c = 1.0 - b1 ** count, 1.0 - b2 ** count
        q = torch.from_numpy(_adv(n, 61)).to(DEV)
        m = torch.from_numpy(_adv(n, 62)).to(DEV)
        v = torch.from_numpy(_adv(n, 63, nonneg=True)).to(DEV)
        w = torch.from_numpy(_adv(n, 64)).to(DEV)
        lab = torch.from_numpy(_adv(n, 65, nonneg=True)).to(DEV)
        tm, tv, tw, tlab = m.clone(), v.clone(), w.clone(), lab.clone()
        _adam_step_inplace(q, tm, tv, tw, tlab if has_lab else None,
                           lr, wd, b1, b2, bc1c, bc2c, eps)
        K.adam_step_inplace(q, m, v, w, lab if has_lab else None,
                            lr, wd, b1, b2, bc1c, bc2c, eps)
        for nm, a, b_ in (("m", m, tm), ("v", v, tv), ("w", w, tw)):
            assert _bit_eq(a, b_) == 0, f"{nm} ({lr},{wd},{count})"
        if has_lab:
            assert _bit_eq(lab, tlab) == 0, "lab"
    print("PASS test_adam_bitwise")


def test_select_hist_bitwise():
    """One-sweep histogram vs the torch bitcast+bincount chain, with the
    folded finite guard counting exactly the nonfinite lanes."""
    if not HAVE:
        print("SKIP test_select_hist_bitwise (no CUDA/triton)")
        return
    score = torch.from_numpy(_adv(1 << 18, 71, nonneg=True)).to(DEV)
    h, nf = K.hist_top16(score)
    top16 = ((score.view(torch.int32) & 0x7FFFFFFF) >> 16)
    assert torch.equal(h, torch.bincount(top16, minlength=32769).cpu())
    assert nf == 0
    dirty = score.clone()
    dirty[5] = float("nan")
    dirty[7] = float("inf")
    dirty[9] = float("-inf")
    h2, nf2 = K.hist_top16(dirty)
    assert nf2 == 3, nf2
    top16d = ((dirty.view(torch.int32) & 0x7FFFFFFF) >> 16)
    assert torch.equal(h2, torch.bincount(top16d, minlength=32769).cpu())
    print("PASS test_select_hist_bitwise")


def test_select_cand_bitwise():
    """Candidate extraction reproduces torch's masked extraction in
    global index order — populated bins and the always-empty top bin."""
    if not HAVE:
        print("SKIP test_select_cand_bitwise (no CUDA/triton)")
        return
    score = torch.from_numpy(_adv(1 << 18, 81, nonneg=True)).to(DEV)
    top16 = ((score.view(torch.int32) & 0x7FFFFFFF) >> 16)
    bins = [0, 32768, int(top16[10].item()), int(top16[1000].item())]
    for b in bins:
        got = K.cand_extract(score, b)
        ref = score[top16 == b].cpu()
        assert got.dtype == ref.dtype and got.shape == ref.shape, b
        assert torch.equal(got, ref), f"bin {b}: order or content moved"
    print("PASS test_select_cand_bitwise")


def test_select_gt_eq_bitwise():
    """gt/eq masks + counts at thresholds chosen to stress the f32
    scalar cast: an in-data value (guaranteed ties) and an f64 value
    whose f32 cast lands on a representable neighbor."""
    if not HAVE:
        print("SKIP test_select_gt_eq_bitwise (no CUDA/triton)")
        return
    score = torch.from_numpy(_adv(1 << 18, 91, nonneg=True)).to(DEV)
    t_in = float(score[100].item())
    # f64-distinct from any f32, casts to exactly 0.1f
    t_cast = float(np.float32(0.1)) + 1e-20
    score[200] = 0.1
    for thresh in (t_in, t_cast):
        gt, eq, counts = K.gt_eq_counts(score, thresh)
        rg = score > thresh
        re_ = score == thresh
        assert torch.equal(gt, rg) and torch.equal(eq, re_), thresh
        assert int(counts[0]) == int(rg.sum()) and \
            int(counts[1]) == int(re_.sum()), thresh
    print("PASS test_select_gt_eq_bitwise")


def test_dispatch_gates():
    """arm()/arm_select() over the killswitch/device/dtype/contiguity
    matrix; formula() unknown-name -> None."""
    old_mode = K.MODE
    try:
        K.MODE = "off"
        assert not K.arm(True, torch.float32)
        K.MODE = "auto"
        # arm() trusts the device/dtype facts it is handed; only the
        # killswitch and the triton import gate inside it
        assert K.arm(True, torch.float32) == (triton is not None)
        assert not K.arm(False, torch.float32)
        assert not K.arm(True, torch.float64)
        assert not K.arm(True, torch.bfloat16)
        if HAVE:
            x = torch.zeros(64, device=DEV)
            assert K.arm_select(x)
            assert not K.arm_select(x[::2])
            assert not K.arm_select(x.double())
            assert not K.arm_select(x.cpu())
            assert K.formula("update") is not None
        assert K.formula("bogus_name") is None
    finally:
        K.MODE = old_mode
    print("PASS test_dispatch_gates")


def main():
    test_canary_no_contraction()
    test_canary_div_sqrt_recipe()
    test_formula_bitwise()
    test_adam_bitwise()
    test_select_hist_bitwise()
    test_select_cand_bitwise()
    test_select_gt_eq_bitwise()
    test_dispatch_gates()
    print("all replay-kernel tests passed")


if __name__ == "__main__":
    main()
