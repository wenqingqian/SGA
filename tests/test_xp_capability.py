#!/usr/bin/env python3
"""Capability tests for the plugin selection/optimizer pieces; the
megatron-integration plumbing (DDP stamps, wrapper state, phase dispatch)
is owned by the engine/EF unit tests:

- the bit-pattern histogram select: exact tier bit-exact vs torch.kthvalue
  across adversarial distributions and k extremes, world-replica scale
  invariance, monotone bin ordering, and the global-allreduce gate (the
  local-only select issues no collective and is bitwise the merged form);
- the approximate tiers' production closeness (balanced within 0.5pp, fast
  within 2pp of the exact count) and the all-equal degenerate;
- the adaptive threshold v3.1: dead-band wobble, log-space steps + clamps,
  gated warmup with consecutive-pass reset, the zero-target short circuit,
  persistent-tau feedback and the collapse guard's warmup mode;
- the coi score against its fp64 offline oracle, incl. the zero-state edge.

Run: python tests/test_xp_capability.py
"""

import math
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sga.plugins.selection._adaptive as adapt_mod
from sga.config import set_args
from sga.config.validation import sga_defaults
from sga.plugins.optimizer.adamw import adamw_grad_correct
from sga.plugins.optimizer.adamw_torch import grad_correct_torch
from sga.plugins.selection._histogram import (
    _HIST_BINS,
    HistogramSelect,
    _abs_bit_top16,
    _build_histogram,
    _histogram_select_threshold,
    select_masks_histogram,
)
from sga.plugins.selection._adaptive import _adapt_threshold
from sga.plugins.selection.topk import TopkSelection


class _SelCtx:
    def __init__(self, grad_views, warmup_target=None):
        self.grad_views = grad_views
        self.comm_group = None
        self.warmup_target = warmup_target


def _kthvalue_threshold(grads, k):
    a = torch.cat(grads).abs()
    return torch.kthvalue(a, a.numel() - k + 1).values.item()


# ------------------------------------------------------- histogram exact tier

def test_exact_threshold_vs_kthvalue_all_distributions():
    """Threshold VALUE is bit-identical to kthvalue for every distribution
    (heavy ties, skew, all-equal, zeros, extreme exponent spread) and every
    k extreme; the derived masks match the kthvalue masks with ties kept."""
    N = 4097  # odd size: exercises partial bins
    g = torch.Generator().manual_seed(1234)
    distributions = [
        torch.randn(N, generator=g) * 1e-3,
        (torch.randn(N, generator=g) * 1e3).round() / 1e3,  # heavy ties
        torch.randn(N, generator=g).exp()
        * torch.sign(torch.randn(N, generator=g)),           # lognormal skew
        torch.full((N,), 0.5),                                # all equal
        torch.randn(N, generator=g) * 1e-3,                   # zero mix below
        torch.randn(N, generator=g) * 1e-3,                   # tiny/huge below
    ]
    distributions[4][::3] = 0.0
    distributions[5][::7] *= 1e30

    for x in distributions:
        grads = [x[:N // 2].contiguous(), x[N // 2:].contiguous()]
        n = sum(gv.numel() for gv in grads)
        for k in (1, 2, n // 10, n // 2, n - 1, n):
            hist = _build_histogram(grads, 16, _HIST_BINS, 1)
            thresh = _histogram_select_threshold(hist, k, 1, grads)
            ref = _kthvalue_threshold(grads, k)
            assert thresh == ref, (thresh, ref, k)
            t2, masks = select_masks_histogram(grads, k, None, mode="exact")
            assert t2 == thresh
            for gv, m in zip(grads, masks):
                assert torch.equal(m, (gv.abs() >= ref).to(torch.uint8))
    print("ok: exact histogram == kthvalue across distributions and k extremes")


def test_scale_invariance_merged_histogram():
    """Merged (world x local) histogram + world-scaled k -> identical bin."""
    x = torch.randn(4097, generator=torch.Generator().manual_seed(99)) * 1e-3
    grads = [x.contiguous()]
    n = x.numel()
    for world in (2, 8, 24):
        for k in (1, n // 5, n // 2, n):
            hist_local = _build_histogram(grads, 16, _HIST_BINS, 1)
            t_local = _histogram_select_threshold(hist_local, k, 1, grads)
            t_merged = _histogram_select_threshold(
                hist_local * world, k, world, grads)
            assert t_local == t_merged, (world, k)
    print("ok: world-replica scale invariance (world 2/8/24)")


def test_top16_monotone_and_sign_stripped():
    vals = torch.tensor([0.0, 1e-45, 1e-30, 0.5, 1.0, 3.4e38, float("inf")])
    top = _abs_bit_top16(vals)
    assert (top[1:] >= top[:-1]).all()
    assert torch.equal(_abs_bit_top16(-vals), top)
    print("ok: top16 monotone in |x|, sign bit stripped")


# ---------------------------------------------- global-allreduce gate flag

def test_global_allreduce_gate_skips_collective():
    """global_allreduce=False (the add-global-allreduce flag's default)
    issues NO collective even with dist initialized and a comm_group — and
    under replicated inputs (the engine's full-sync scores) the local-only
    select is bit-identical to the merged (world-scaled) form, every tier.
    The k corners include k=n, which drives the approximate tiers' rank
    formula through max(1, round(...)==0) before the world multiply."""
    N = 100_003
    g = torch.Generator().manual_seed(7)
    x = torch.randn(N, generator=g) * 1e-3
    x[::991] *= 50.0
    grads = [x[:N // 2].contiguous(), x[N // 2:].contiguous()]
    n = sum(gv.numel() for gv in grads)
    world = 4
    grp = object()
    calls = []

    def fake_all_reduce(t, op=None, group=None):
        calls.append((op, group))
        t.mul_(world)  # an exact int64 merge of `world` identical replicas

    with mock.patch.object(torch.distributed, "is_initialized",
                           return_value=True), \
         mock.patch.object(torch.distributed, "all_reduce",
                           side_effect=fake_all_reduce), \
         mock.patch.object(torch.distributed, "get_world_size",
                           return_value=world):
        for mode in ("exact", "balanced", "fast"):
            for k in (1, int(round(n * 0.3)), n):
                calls.clear()
                t_off, m_off = select_masks_histogram(
                    grads, k, grp, mode=mode, global_allreduce=False)
                assert calls == [], (mode, k, calls)
                t_on, m_on = select_masks_histogram(
                    grads, k, grp, mode=mode, global_allreduce=True)
                assert calls == [(torch.distributed.ReduceOp.SUM, grp)], (
                    mode, k, calls)
                assert t_off == t_on, (mode, k)
                for a, b in zip(m_off, m_on):
                    assert torch.equal(a, b), (mode, k)
    print("ok: gate skips the collective; local select == merged bitwise")


def test_global_allreduce_gate_block_path():
    """The block-granular select threads the gate identically: off issues
    no collective and the expanded element masks are bitwise the merged
    form's (block32 + balanced, the production block geometry)."""
    N = 8192
    x = torch.randn(N, generator=torch.Generator().manual_seed(11)) * 1e-3
    x[::37] *= 30.0
    grads = [x.contiguous()]
    k = int(round(N * 0.4))
    world = 4
    grp = object()
    calls = []

    def fake_all_reduce(t, op=None, group=None):
        calls.append((op, group))
        t.mul_(world)

    with mock.patch.object(torch.distributed, "is_initialized",
                           return_value=True), \
         mock.patch.object(torch.distributed, "all_reduce",
                           side_effect=fake_all_reduce), \
         mock.patch.object(torch.distributed, "get_world_size",
                           return_value=world):
        sel_off = HistogramSelect("balanced", block_size=32,
                                  block_reduce="mean", global_allreduce=False)
        sel_on = HistogramSelect("balanced", block_size=32,
                                 block_reduce="mean", global_allreduce=True)
        t_off, m_off = sel_off.select(grads, k, grp, None)
        assert calls == [], calls
        t_on, m_on = sel_on.select(grads, k, grp, None)
        assert len(calls) == 1, calls
        assert t_off == t_on
        assert torch.equal(m_off[0], m_on[0])
    print("ok: block path threads the gate (off == on bitwise)")


def test_topk_selection_reads_global_allreduce_flag():
    """TopkSelection wires the add-global-allreduce flag into both
    histogram-family impls (the flag's single runtime reader); non-bool
    values are refused by both impl constructors' strictness guards."""
    for method in ("histogram", "adaptive"):
        for flag in (False, True):
            set_args(SimpleNamespace(**{
                **sga_defaults(),
                "sga_selection_method": method,
                "sga_selection_ratio_percent": 30.0,
                "sga_selection_topk_histogram_add_global_allreduce": flag,
            }))
            sel = TopkSelection()
            assert sel._impl.global_allreduce is flag, (method, flag)
    for bad in (1, "true", None):
        for ctor in (lambda b: HistogramSelect("exact", global_allreduce=b),
                     lambda b: adapt_mod.AdaptiveTau(
                         40.0, 5, "fast", global_allreduce=b)):
            try:
                ctor(bad)
                raise AssertionError(
                    f"non-bool global_allreduce accepted: {bad!r}")
            except ValueError:
                pass
    print("ok: topk selection wires the flag; non-bool gate values refused")


def test_adaptive_warmup_threads_global_allreduce():
    """AdaptiveTau's histogram warmup AND collapse re-bootstrap selects
    inherit the global_allreduce wiring (the constructor argument reaches
    every select_masks_histogram call)."""
    grads = [torch.linspace(-1, 1, 1000)]
    seen = []
    real = adapt_mod.select_masks_histogram

    def spy(*a, **kw):
        seen.append(kw.get("global_allreduce", "__missing__"))
        return real(*a, **kw)

    with mock.patch.object(adapt_mod, "select_masks_histogram",
                           side_effect=spy):
        impl = adapt_mod.AdaptiveTau(40.0, None, "fast",
                                     global_allreduce=False)
        impl.select(grads, 400, None, None)  # warmup bootstrap (call 1)
        impl.tau = 1e9                       # above every |g| -> measured 0
        impl.warmup_passed = 1               # past the warmup gate
        impl.select(grads, 400, None, None)  # collapse re-bootstrap (call 2)
        assert seen == [False, False], seen
    print("ok: adaptive warmup + collapse thread global_allreduce through")


# ------------------------------------------------- approximate tiers (tier 2)

def test_histogram_mode_production_closeness():
    """balanced within 0.5pp / fast within 2pp of the exact selected count on
    a 1M production-shaped gradient (periodic large outliers); every selected
    element passes the returned bin edge; all-equal degenerate selects all."""
    N = 1_000_003
    g = torch.Generator().manual_seed(42)
    x = torch.randn(N, generator=g) * 1e-3
    x[::997] *= 100.0
    grads = [x[:N // 2].contiguous(), x[N // 2:].contiguous()]
    n = sum(gv.numel() for gv in grads)
    k = int(round(n * 0.4))

    t_e, m_e = select_masks_histogram(grads, k, None, mode="exact")
    assert t_e == _kthvalue_threshold(grads, k)
    n_e = sum(int(mm.sum().item()) for mm in m_e)

    t_b, m_b = select_masks_histogram(grads, k, None, mode="balanced")
    n_b = sum(int(mm.sum().item()) for mm in m_b)
    assert abs(n_b - n_e) / n < 0.005
    for gv, m in zip(grads, m_b):
        assert torch.all(gv.abs()[m.bool()] >= t_b)
    assert 0.9 * t_e <= t_b <= 1.05 * t_e

    t_f, m_f = select_masks_histogram(grads, k, None, mode="fast")
    n_f = sum(int(mm.sum().item()) for mm in m_f)
    assert abs(n_f - n_e) / n < 0.02
    for gv, m in zip(grads, m_f):
        assert torch.all(gv.abs()[m.bool()] >= t_f)

    # all-equal updates: one coarse bin holds everything -> select all
    eq = [torch.full((5000,), 1e-3), torch.full((5000,), 1e-3)]
    t_eq, m_eq = select_masks_histogram(eq, 4000, None, mode="fast")
    assert sum(int(mm.sum().item()) for mm in m_eq) == 10_000
    assert 0 < t_eq <= 1e-3

    try:
        select_masks_histogram(grads, 100, None, mode="bogus")
        raise AssertionError("bogus histogram_mode accepted")
    except ValueError:
        pass
    print("ok: histogram mode production closeness (0.5pp/2pp) + degenerate")


# -------------------------------------------------------- adaptive threshold

def test_adapt_rule_band_wobble_and_steps():
    """±1pp dead band with its asymmetric float wobble (0.41 in, 0.39 out at
    a 0.40 target), log-space out-of-band steps, [0.9, 1.15] clamps."""
    for ratio in (0.40, 0.391, 0.409, 0.41):
        assert _adapt_threshold(1e-3, ratio, 0.40) == 1e-3, ratio
    # 0.40 - 0.39 = 0.01000...09 lands OUTSIDE the band
    assert math.isclose(_adapt_threshold(1e-3, 0.39, 0.40),
                        1e-3 * (0.39 / 0.40) ** 0.5, rel_tol=1e-12)
    assert _adapt_threshold(1e-3, 0.50, 0.40) > 1e-3   # over-select -> raise
    assert _adapt_threshold(1e-3, 0.30, 0.40) < 1e-3   # under-select -> lower
    assert math.isclose(_adapt_threshold(1e-3, 0.50, 0.40),
                        1e-3 * 1.25 ** 0.5, rel_tol=1e-12)
    assert math.isclose(_adapt_threshold(1e-3, 0.30, 0.40), 1e-3 * 0.9,
                        rel_tol=1e-12)  # (0.75)^0.5 = 0.866 -> clamped
    assert math.isclose(_adapt_threshold(1e-3, 1.0, 0.40), 1e-3 * 1.15,
                        rel_tol=1e-12)
    assert math.isclose(_adapt_threshold(1e-3, 0.0, 0.40), 1e-3 * 0.9,
                        rel_tol=1e-12)
    # self-scaling: gentle at the band edge, clamped when far
    near = _adapt_threshold(1e-3, 0.42, 0.40) / 1e-3
    assert math.isclose(near, 1.05 ** 0.5, rel_tol=1e-12)
    assert math.isclose(_adapt_threshold(1e-3, 0.80, 0.40) / 1e-3, 1.15,
                        rel_tol=1e-12)
    print("ok: adapt rule band wobble, log-space steps, clamps")


def test_adaptive_warmup_gate_and_consecutive_reset():
    """Gated warmup: a miss resets the consecutive count (warmup EXTENDS,
    never escalates); warmup_target=0 short-circuits to a single bootstrap
    and the feedback path takes over from the second sync; the bootstrap and
    collapse re-bootstrap run in the configured warmup mode (fast)."""
    grads = [torch.linspace(-1, 1, 1000)]
    good = torch.zeros(1000, dtype=torch.uint8)
    good[:400] = 1  # 40% == target -> gate passes
    bad = torch.zeros(1000, dtype=torch.uint8)
    bad[:800] = 1   # 80% >> target -> gate fails

    set_args(SimpleNamespace(**{**sga_defaults(),
                                 "sga_selection_method": "adaptive",
                                 "sga_selection_ratio_percent": 40.0}))
    sel = TopkSelection()
    seq = [(0.6, [good]), (0.6, [good]), (0.6, [bad]), (0.6, [good])]
    with mock.patch.object(
        adapt_mod, "select_masks_histogram", side_effect=seq):
        sel.select(_SelCtx(grads, warmup_target=3))
        assert sel.warmup_passed == 1
        sel.select(_SelCtx(grads, warmup_target=3))
        assert sel.warmup_passed == 2
        sel.select(_SelCtx(grads, warmup_target=3))
        assert sel.warmup_passed == 0  # reset, not 3
        sel.select(_SelCtx(grads, warmup_target=3))
        assert sel.warmup_passed == 1

    set_args(SimpleNamespace(**{**sga_defaults(),
                                 "sga_selection_method": "adaptive",
                                 "sga_selection_ratio_percent": 40.0}))
    sel2 = TopkSelection()
    with mock.patch.object(
        adapt_mod, "select_masks_histogram", return_value=(0.6, [good])
    ) as hist:
        sel2.select(_SelCtx(grads, warmup_target=0))
        assert hist.call_count == 1
        assert hist.call_args.kwargs.get("mode") == "fast"
        assert sel2.warmup_passed == 1
        sel2.select(_SelCtx(grads, warmup_target=0))  # tau path from here
        assert hist.call_count == 1
        assert sel2.tau == 0.6  # 40% measured, in band, frozen
    print("ok: adaptive warmup gate (consecutive reset, zero short-circuit)")


def test_adaptive_tau_feedback_and_collapse_mode():
    """Past the gate the persistent tau drives the mask directly and adapts by
    the exact log-space factor; in-band measurements freeze it; the collapse
    guard re-bootstraps in the CONFIGURED warmup mode."""
    grads = [torch.linspace(-1, 1, 1000)]

    set_args(SimpleNamespace(**{**sga_defaults(),
                                 "sga_selection_method": "adaptive",
                                 "sga_selection_ratio_percent": 40.0}))
    sel = TopkSelection()
    sel._impl.tau = 0.5           # 50% selected -> over target
    sel._impl.warmup_passed = 5   # past the default warmup
    masks = sel.select(_SelCtx(grads))
    assert sel.last_threshold == 0.5  # masks built with the OLD tau
    assert torch.equal(masks[0], (grads[0].abs() >= 0.5).to(torch.uint8))
    assert math.isclose(sel.tau, 0.5 * 1.25 ** 0.5, rel_tol=1e-12)

    set_args(SimpleNamespace(**{**sga_defaults(),
                                 "sga_selection_method": "adaptive",
                                 "sga_selection_ratio_percent": 40.0}))
    sel2 = TopkSelection()
    sel2._impl.tau = 0.6           # 40% selected, in band
    sel2._impl.warmup_passed = 5
    sel2.select(_SelCtx(grads))
    assert sel2.tau == 0.6

    # collapse: nothing passes tau -> re-bootstrap in the warmup mode
    canned = (0.6, [(grads[0].abs() >= 0.6).to(torch.uint8)])
    set_args(SimpleNamespace(**{**sga_defaults(),
                                 "sga_selection_method": "adaptive",
                                 "sga_selection_ratio_percent": 40.0,
                                 "sga_selection_adaptive_warmup_mode": "exact"}))
    sel3 = TopkSelection()
    sel3._impl.tau = 1e9
    sel3._impl.warmup_passed = 5
    with mock.patch.object(
        adapt_mod, "select_masks_histogram", return_value=canned
    ) as hist:
        masks = sel3.select(_SelCtx(grads))
    hist.assert_called_once()
    assert hist.call_args.kwargs.get("mode") == "exact"
    assert sel3.tau == 0.6
    assert int(masks[0].sum().item()) == 400
    print("ok: adaptive tau feedback exact factor + collapse mode")


# ------------------------------------------------------------- coi fp64 oracle

def _coi_reference(g, m, v, beta1, beta2, eps, step):
    """fp64 oracle of |U(g) - U(0)| (weight-decay-free, bias corrections of
    the upcoming step)."""
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    g64, m64, v64 = g.double(), m.double(), v.double()
    u_g = ((beta1 * m64 + (1.0 - beta1) * g64) / bc1) / (
        ((beta2 * v64 + (1.0 - beta2) * g64 * g64) / bc2).sqrt() + eps
    )
    u_0 = ((beta1 * m64) / bc1) / ((beta2 * v64 / bc2).sqrt() + eps)
    return (u_g - u_0).abs()


def test_coi_score_matches_fp64_oracle():
    b1, b2, eps = 0.9, 0.95, 1e-8
    g = torch.Generator().manual_seed(0)
    for step in (1, 2, 17, 1501):
        n = 4097
        grad = torch.randn(n, generator=g) * 1e-3
        m = torch.randn(n, generator=g) * 1e-3
        v = torch.rand(n, generator=g) * 1e-6
        w = torch.randn(n, generator=g) * 0.02
        buf = grad.clone()  # becomes the coi score under the full-step emit
        grad_correct_torch(
            w, buf, m.clone(), v.clone(), 0.0, b1, b2, eps, 0.0,
            1 - b1 ** step, 1 - b2 ** step,
            None, "coi",
        )
        ref = _coi_reference(grad, m, v, b1, b2, eps, step)
        rel = ((buf.double() - ref).abs().max() / ref.max()).item()
        assert rel < 1e-5, (step, rel)
        assert buf.dtype == torch.float32

    # zero-state edge (m = v = 0): U(0) = 0, score = |U(g)|
    grad = torch.randn(1000, generator=g) * 1e-3
    buf = grad.clone()
    grad_correct_torch(
        torch.zeros(1000), buf, torch.zeros(1000), torch.zeros(1000),
        0.0, b1, b2, eps, 0.0, 1 - b1, 1 - b2, None, "coi",
    )
    ref = _coi_reference(grad, torch.zeros(1000), torch.zeros(1000),
                         b1, b2, eps, 1)
    assert torch.allclose(buf.double(), ref, rtol=1e-5, atol=1e-12)

    # empty input: noop via the public wrapper
    adamw_grad_correct(torch.empty(0), torch.empty(0), torch.empty(0),
                        torch.empty(0), 1e-3, b1, b2, eps, 0.0, 1,
                        emit_score="coi")
    print("ok: coi score == fp64 oracle (steps 1/2/17/1501, zero-state, empty)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
