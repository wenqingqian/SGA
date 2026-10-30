#!/usr/bin/env python3
"""Unit tests for the AdamS optimizer template (sga/plugins/optimizer/adams.py)
and the optimizer-aware score registry legs, CPU-only.

Covers:
- the AdamS score formulas (optimizer="adams") vs independent fp64 oracles,
  incl. the bias-correction channels being FORCED to 1.0 inside the formula
  (whatever the caller passes) and the unknown-optimizer refusal;
- the ordering-directed fixture: v is rebuilt from the PRE-commit m_old
  (same read point as abs_m), discriminating the two wrong bases
  (post-commit m_new, stored v_old) with opposite-sign and cross-scale
  m_old/g pairs;
- the torch reference path: sparse (mask/coeff/clear/fold) and full steps
  (every emittable score) vs hand oracles; fused sparse knobs == the
  standalone sequence, bitwise;
- the wrapper guards (unbranched score name, sparse-fusion knobs, the
  dual-lr domain/shape rules) and the no-step (no bias correction) surface;
- SgaAdamS at the plugin level: zero-arg build off the singleton, score
  emission, the sparse end-state invariant, the CPU compact dispatch
  (table == bisection on the fallback), the boundary dual-lr semantics;
- config validation: AdamS x shadow refused, the existing optimizer gates
  (the boundary dual-lr rules) isomorphic across both optimizer names,
  and the registry face (build_optimizer("adams"), duplicate refusal,
  curated unknown-name error listing both templates).

Run: python tests/test_adams_unit.py  (or pytest tests/test_adams_unit.py)
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.config import set_args
from sga.config.validation import sga_defaults, validate_sga_args
from sga.plugins import (
    build_optimizer,
    plugin_class,
    register_default_plugins,
    register_optimizer,
)
from sga.plugins.optimizer import OptContext
from sga.plugins.optimizer.adams import (
    SgaAdamS,
    _TRITON_SCORE_MAP,
    adams_grad_correct,
)
from sga.plugins.optimizer.adams_torch import adams_grad_correct_torch
from sga.plugins.optimizer.adamw import SgaAdamW
from sga.plugins.scores import get_score_spec, score_names

torch.manual_seed(0)


def _seed(**over):
    """Seed the sga args singleton for a zero-arg plugin build
    (declared defaults + a buildable sga_score + overrides)."""
    ns = SimpleNamespace(**{**sga_defaults(), "sga_score": "update", **over})
    set_args(ns)
    return ns


class _Rec:
    """Minimal ParamRecord stand-in (SgaAdamS.full_step / sparse_step)."""

    __hash__ = None

    def __init__(self, param, grad):
        self.main_param = param
        self.grad = grad
        self._state = {}
        self.lr, self.beta1, self.beta2 = 1e-3, 0.9, 0.95
        self.eps, self.weight_decay, self.step = 1e-8, 1e-2, 7

    def mask_u8(self):
        raise AssertionError("full steps take no mask")


class _RecCtx:
    def __init__(self, records):
        self.records = records


class _SparseRec(_Rec):
    def __init__(self, param, grad, mask):
        super().__init__(param, grad)
        self._mask = mask

    def mask_u8(self):
        return self._mask


class _Bucket:
    def __init__(self, grad_data, mask):
        self.grad_data = grad_data
        self.active_mask_u8 = mask
        self.extra_main_grads = [torch.randn(3)]


# ------------------------------------------------- optimizer-aware formulas

def _adams_formula_oracles(w, m, v, g, b1, b2, eps, wd):
    """Straightforward fp64 oracles for the AdamS scores, written
    independently of the formulas (v rebuilt from the PRE-commit m, no
    bias correction)."""
    w64, m64, g64 = (t.double() for t in (w, m, g))
    m_new = b1 * m64 + (1 - b1) * g64
    v_new = b2 * (m64 * m64) + (1 - b2) * g64 * g64
    u_g = m_new / (v_new.sqrt() + eps)
    u_0 = (b1 * m64) / ((b2 * (m64 * m64)).sqrt() + eps)
    return {
        "abs_g": g64.abs(),
        "abs_m": m64.abs(),
        "update": wd * w64 + u_g,
        "coi": (u_g - u_0).abs(),
        # coi's fp32 error lives at the OPERAND scale (|u_g - u_0| can sit
        # many binades below both terms — a result-relative bound is
        # meaningless near cancellation, the same policy as the wd-free
        # update test)
        "coi_operand_scale": torch.maximum(u_g.abs(), u_0.abs()),
    }


def _assert_score_close(got, oracle64, mode, oracles):
    """Score-vs-fp64-oracle compare: uniform tolerances everywhere except
    coi, whose bound is taken at the operand scale."""
    if mode == "coi":
        diff = (got.double() - oracle64).abs()
        scale = oracles["coi_operand_scale"].clamp_min(1e-30)
        assert float((diff / scale).max()) < 1e-5, (mode, diff.max())
    else:
        torch.testing.assert_close(got.double(), oracle64,
                                   rtol=1e-5, atol=1e-7)


def test_adams_score_formulas_match_fp64_oracles():
    torch.manual_seed(3)
    n, b1, b2, eps, wd, step = 513, 0.9, 0.95, 1e-8, 1e-2, 7
    # deliberately NON-1.0 bias corrections: AdamS must ignore the channels
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    w = torch.randn(n)
    m = torch.randn(n) * 0.1
    v = torch.rand(n) * 0.1 + 1e-4   # inert on the AdamS path (never read)
    g = torch.randn(n)
    oracles = _adams_formula_oracles(w, m, v, g, b1, b2, eps, wd)
    for name in score_names():
        spec = get_score_spec(name)
        w0, m0, v0, g0 = w.clone(), m.clone(), v.clone(), g.clone()
        out = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                           update_includes_wd=True, optimizer="adams")
        _assert_score_close(out, oracles[name], name, oracles)
        # no training input mutated (shadow's contract)
        assert torch.equal(w, w0) and torch.equal(m, m0)
        assert torch.equal(v, v0) and torch.equal(g, g0)
    # the correction channels are forced to 1.0 INSIDE the formula: any
    # caller-passed values are inert, bitwise (dividing by 1.0 is the
    # identity — and the division is elided entirely)
    for name in ("update", "coi"):
        spec = get_score_spec(name)
        a = spec.formula(w, m, v, g, b1, b2, eps, wd, 0.5, 0.75,
                         update_includes_wd=True, optimizer="adams")
        b = spec.formula(w, m, v, g, b1, b2, eps, wd, 1.0, 1.0,
                         update_includes_wd=True, optimizer="adams")
        assert torch.equal(a, b), name
    print("ok: AdamS score formulas match fp64 oracles; bc channels inert")


def test_score_formula_optimizer_vocabulary():
    """An unknown optimizer name fails loud through update/coi (the two
    formulas that branch on the recursion); abs_g/abs_m are inert by
    construction (the keyword changes nothing they read) and run on."""
    n = 64
    w, m, v, g = (torch.randn(n) for _ in range(4))
    kw = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=1e-2)
    for name in ("update", "coi"):
        try:
            get_score_spec(name).formula(w, m, v, g, bias_correction1=0.9,
                                         bias_correction2=0.99,
                                         update_includes_wd=True,
                                         optimizer="adam", **kw)
            raise AssertionError(f"unknown optimizer accepted by {name}")
        except ValueError as e:
            assert "adam" in str(e) and "adams" in str(e), e
    for name in ("abs_g", "abs_m"):
        out = get_score_spec(name).formula(w, m, v, g, bias_correction1=0.9,
                                           bias_correction2=0.99,
                                           update_includes_wd=True,
                                           optimizer="adams", **kw)
        assert out.shape == w.shape
    # the default (unset optimizer) keeps AdamW semantics: the v-fold base
    ref_adamw = get_score_spec("update").formula(
        w, m, v, g, bias_correction1=1.0, bias_correction2=1.0,
        update_includes_wd=True, **kw)
    ref_adams = get_score_spec("update").formula(
        w, m, v, g, bias_correction1=1.0, bias_correction2=1.0,
        update_includes_wd=True, optimizer="adams", **kw)
    assert not torch.equal(ref_adamw, ref_adams), \
        "the optimizer keyword changed nothing (v recursion not branched)"
    print("ok: score formula optimizer vocabulary (loud unknown, inert "
          "abs_*, AdamW default unchanged)")


# ------------------------------------- v base ordering (the spec's core)

def test_adams_v_base_is_pre_commit_m():
    """Ordering-directed fixture: the second moment must read the
    PRE-commit first moment. m_old and g are built with opposite signs
    and crossed scales so the three candidate bases give pairwise
    DIFFERENT v_new — beta1/beta2/values chosen so every arithmetic step
    is exact in fp32:
        pre-commit m (the spec):   v = b2*m^2 + (1-b2)*g^2
        post-commit m_new (wrong): m_new = b1*m + (1-b1)*g enters v
        stored v_old (AdamW, wrong): v = b2*v_old + (1-b2)*g^2
    Pinned at BOTH levels: the registry formula and the plugin step
    (torch reference, CPU)."""
    b1, b2, lr, wd, eps = 0.5, 0.5, 1.0, 0.0, 0.0
    # (m_old, g, v_old): opposite signs (with m_new != 0 — an exactly-zero
    # m_new cannot discriminate at the update level), then crossed scales
    for m_val, g_val, v_old in ((-2.0, 0.5, 8.0),
                                (2.0 ** -10, 2.0 ** 6, 1.0)):
        m = torch.tensor([m_val])
        g = torch.tensor([g_val])
        v = torch.tensor([v_old])
        w = torch.tensor([1.0])
        m_new = b1 * m + (1 - b1) * g
        v_spec = b2 * (m * m) + (1 - b2) * g * g
        v_post_m = b2 * (m_new * m_new) + (1 - b2) * g * g
        v_adamw = b2 * v + (1 - b2) * g * g
        # the three hypotheses are pairwise distinct (the fixture bites)
        assert v_spec.item() != v_post_m.item()
        assert v_spec.item() != v_adamw.item()
        assert v_post_m.item() != v_adamw.item()

        # registry formula level: update tracks the spec's v exactly
        upd = get_score_spec("update").formula(
            w, m, v, g, b1, b2, eps, wd, 1.0, 1.0,
            update_includes_wd=True, optimizer="adams")
        upd_spec = (wd * w + m_new / (v_spec.sqrt() + eps))
        upd_post = (wd * w + m_new / (v_post_m.sqrt() + eps))
        assert torch.equal(upd, upd_spec), (m_val, g_val)
        assert not torch.equal(upd, upd_post), (m_val, g_val)

        # plugin level: the committed v buffer holds the spec's value
        p, gg, mm, vv = w.clone(), g.clone(), m.clone(), v.clone()
        adams_grad_correct_torch(p, gg, mm, vv, lr, b1, b2, eps, wd,
                                 mask=None, emit_score=None)
        assert vv.item() == v_spec.item(), (m_val, g_val)
        assert mm.item() == m_new.item()
        # and an AdamW step on the same state commits a DIFFERENT v
        from sga.plugins.optimizer.adamw_torch import grad_correct_torch
        p2, gg2, mm2, vv2 = w.clone(), g.clone(), m.clone(), v.clone()
        grad_correct_torch(p2, gg2, mm2, vv2, lr, b1, b2, eps, wd,
                           1.0, 1.0, mask=None, emit_score=None)
        assert vv2.item() == v_adamw.item()
        assert vv2.item() != vv.item()
    print("ok: AdamS v base is the pre-commit m (both formula and plugin)")


# ---------------------------------------------------- torch reference path

def test_adams_torch_sparse_and_full_vs_oracles():
    """The torch reference step vs hand oracles: sparse masked commit
    (coeff/clear/fold) and the full step with every emittable score, on
    pre-existing moments (the AdamS tail is non-trivial from step one)."""
    torch.manual_seed(7)
    n, b1, b2, eps, wd, lr = 256, 0.9, 0.95, 1e-8, 1e-2, 1e-3
    mask = (torch.rand(n) < 0.3).to(torch.uint8)
    active = mask > 0
    p0, g0 = torch.randn(n), torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1

    # sparse: g committed at active coords only; v from the pre-commit m
    p, g, m, v = p0.clone(), g0.clone(), m0.clone(), v0.clone()
    adams_grad_correct_torch(p, g, m, v, lr, b1, b2, eps, wd,
                             mask=mask, emit_score=None,
                             coeff=0.5, clear=True, fold=True, interval=8.0)
    g_eff = torch.where(active, g0 * 0.5, torch.zeros(n))
    m_ref = b1 * m0 + (1 - b1) * g_eff
    v_ref = b2 * (m0 * m0) + (1 - b2) * g_eff * g_eff
    upd_ref = wd * p0 + m_ref / (v_ref.sqrt() + eps)
    torch.testing.assert_close(m, m_ref, rtol=0, atol=0)
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)
    torch.testing.assert_close(p, p0 - lr * upd_ref, rtol=0, atol=0)
    # grad end state: actives zeroed, inactives folded by the window
    assert torch.all(g[active] == 0)
    torch.testing.assert_close(g[~active], g0[~active] / 8.0, rtol=0, atol=0)

    # full: every score materializes; the apply fuses in the same pass
    oracles = _adams_formula_oracles(p0, m0, v0, g0, b1, b2, eps, wd)
    for mode in score_names():
        p, g, m, v = p0.clone(), g0.clone(), m0.clone(), v0.clone()
        adams_grad_correct_torch(p, g, m, v, lr, b1, b2, eps, wd,
                                 mask=None, emit_score=mode)
        _assert_score_close(g, oracles[mode], mode, oracles)
        torch.testing.assert_close(m, m_ref_full(m0, g0, b1, b2),
                                   rtol=0, atol=0)
        torch.testing.assert_close(
            v, v_ref_full(m0, g0, b1, b2), rtol=0, atol=0)
        torch.testing.assert_close(p, p0 - lr * oracles["update"].float(),
                                   rtol=1e-5, atol=1e-7)
    print("ok: AdamS torch reference (sparse fusion + every full score) "
          "matches oracles")


def m_ref_full(m0, g0, b1, b2):
    return b1 * m0 + (1 - b1) * g0


def v_ref_full(m0, g0, b1, b2):
    return b2 * (m0 * m0) + (1 - b2) * g0 * g0


def test_adams_sparse_fusion_bitwise():
    """Fused knobs == the standalone sequence, bitwise on every buffer
    (coeff is a masked prescale, clear zeroes actives, fold divides by
    the window) — the AdamW contract, replicated."""
    torch.manual_seed(11)
    n, b1, b2 = 128, 0.9, 0.95
    for fold, interval in [(False, 8.0), (True, 8.0)]:
        for coeff in (0.5, 1.0):
            g0 = torch.randn(n)
            mask = (torch.rand(n) < 0.25).to(torch.uint8)
            p0, m0, v0 = torch.randn(n), torch.randn(n) * .1, torch.rand(n) * .1

            p1, g1, m1, v1 = p0.clone(), g0.clone(), m0.clone(), v0.clone()
            adams_grad_correct_torch(p1, g1, m1, v1, 1e-3, b1, b2, 1e-8, 1e-2,
                                     mask=mask, emit_score=None,
                                     coeff=coeff, clear=True, fold=fold,
                                     interval=interval)
            p2, m2, v2 = p0.clone(), m0.clone(), v0.clone()
            g2 = g0.clone()
            if coeff != 1.0:
                g2 = torch.where(mask > 0, g2 * coeff, g2)
            adams_grad_correct_torch(p2, g2, m2, v2, 1e-3, b1, b2, 1e-8, 1e-2,
                                     mask=mask, emit_score=None,
                                     clear=True, fold=fold,
                                     interval=interval)
            for got, ref, name in ((p1, p2, "param"), (m1, m2, "exp_avg"),
                                   (v1, v2, "exp_avg_sq"), (g1, g2, "grad")):
                assert torch.equal(got, ref), (name, coeff, fold)


def test_adams_wrapper_guards_and_no_step_surface():
    """The wrapper owns the same refusals as the AdamW one (unbranched
    score name, sparse-fusion knobs off their step shape, the dual-lr
    domain/shape rules, the dormant masked-commit hybrid) — and AdamS
    has NO step argument: bias corrections do not exist on this path, so
    passing one is a loud TypeError, not a silently-ignored value."""
    n = 32
    p, g, m, v = (torch.randn(n) for _ in range(4))
    mask = (torch.rand(n) < .5).to(torch.uint8)
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=1e-2)
    # unbranched score name: owned message, before any buffer work
    try:
        adams_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                           emit_score="bogus", **kw)
        raise AssertionError("unbranched score accepted")
    except ValueError as e:
        assert "_adams_grad_correct_kernel branch" in str(e), e
    # sparse-fusion knobs refuse on full steps / mask-less sparse steps
    for kwargs in (dict(emit_score=None, mask=None, coeff=0.5),
                   dict(emit_score=None, mask=None, clear=True),
                   dict(emit_score=None, mask=None, fold=True),
                   dict(emit_score="update", mask=mask, clear=True)):
        try:
            adams_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                               **kwargs, **kw)
        except ValueError as e:
            assert "requires a sparse step" in str(e), e
        else:
            raise AssertionError(f"fusion guard did not fire for {kwargs}")
    # dual-lr: sparse steps, mask-less full steps, and the domain rule
    for kwargs in (dict(emit_score=None, mask=mask, inactive_lr_scale=0.5),
                   dict(emit_score="update", mask=None, inactive_lr_scale=0.5),
                   dict(emit_score="update", mask=mask, inactive_lr_scale=-1.0),
                   dict(emit_score="update", mask=mask,
                        inactive_lr_scale=float("nan")),
                   dict(emit_score="update", mask=mask, inactive_lr_scale=1.0)):
        try:
            adams_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                               **kwargs, **kw)
        except ValueError as e:
            assert "inactive_lr_scale" in str(e) or \
                "masked-commit full step" in str(e), e
        else:
            raise AssertionError(f"dual-lr guard did not fire for {kwargs}")
    # the legal combination runs
    adams_grad_correct(p, g, m, v, mask=mask, emit_score="update",
                       inactive_lr_scale=0.5, **kw)
    # no-step surface: there is no bias correction to compute, so a step
    # argument is not accepted silently
    try:
        adams_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                           step=5, **kw)
        raise AssertionError("a step argument was accepted")
    except TypeError:
        pass
    # the constructor repeats the domain guard for direct construction
    _seed(sga_optimizer_full_sync_inactive_lr_scale=-0.5)
    try:
        SgaAdamS()
    except ValueError as e:
        assert "full_sync_inactive_lr_scale" in str(e), e
    else:
        raise AssertionError("constructor domain guard did not fire")
    print("ok: AdamS wrapper guards + no-step surface")


# ---------------------------------------------------------- plugin level

def test_sga_adams_build_face():
    """Zero-arg build off the singleton, knob reads, and the loud
    unbranched-score refusal at construction (the registry alignment)."""
    _seed()
    opt = build_optimizer("adams")
    assert isinstance(opt, SgaAdamS) and opt.emit_score == "update"
    assert opt.compact_search == "table"
    assert opt.full_sync_inactive_lr_scale == 1.0
    assert opt.residual_mean is False
    _seed(sga_score="abs_m", sga_optimizer_residual_mean=True,
          sga_optimizer_compact_search="bisection",
          sga_optimizer_full_sync_inactive_lr_scale=0.5)
    opt = SgaAdamS()
    assert opt.emit_score == "abs_m" and opt.residual_mean
    assert opt.compact_search == "bisection"
    assert opt.full_sync_inactive_lr_scale == 0.5
    assert set(_TRITON_SCORE_MAP) == set(score_names())
    _seed(sga_score="bogus")
    try:
        SgaAdamS()
        raise AssertionError("unbranched score accepted at build time")
    except ValueError as e:
        assert "no EMIT_MODE branch" in str(e) and "bogus" in str(e), e
    print("ok: SgaAdamS build face (singleton knobs, branch refusal)")


def test_sga_adams_full_step_score_emission():
    """Each emitted score and the applied update match hand oracles on
    the PRE-commit moments (state pre-seeded — the AdamS tail is
    non-trivial), and the moments commit to the AdamS recursion."""
    torch.manual_seed(5)
    n, b1, b2, eps, wd, lr = 512, 0.9, 0.95, 1e-8, 1e-2, 1e-3
    p0, g0 = torch.randn(n), torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    oracles = _adams_formula_oracles(p0, m0, v0, g0, b1, b2, eps, wd)
    for mode in score_names():
        w = p0.clone().requires_grad_(False)
        g = g0.clone().requires_grad_(False)
        rec = _Rec(w, g)
        rec._state = {"exp_avg": m0.clone(), "exp_avg_sq": v0.clone()}
        _seed(sga_score=mode)
        SgaAdamS().full_step(_RecCtx([rec]))
        _assert_score_close(g.data, oracles[mode], mode, oracles)
        torch.testing.assert_close(w.data, p0 - lr * oracles["update"].float(),
                                   rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(rec._state["exp_avg"],
                                   m_ref_full(m0, g0, b1, b2),
                                   rtol=0, atol=0)
        torch.testing.assert_close(rec._state["exp_avg_sq"],
                                   v_ref_full(m0, g0, b1, b2),
                                   rtol=0, atol=0)
    print("ok: SgaAdamS full-step score emission matches oracles")


def test_sga_adams_sparse_end_state_invariant():
    """The sparse-fusion contract at SgaAdamS level: mask-active coords
    bitwise 0 after sparse_step, mask-inactive coords keep the residual
    (folded by the window only on the pre-boundary step with
    residual_mean); reset() must NOT zero the bucket on a sparse step and
    must zero everything on a full step / a degenerate mask-less bucket."""
    torch.manual_seed(11)
    n = 256
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    active = mask > 0
    for residual_mean, next_full, fold_window in [(False, False, 8),
                                                  (True, False, 8),
                                                  (True, True, 8)]:
        w = torch.randn(n)
        g0 = torch.randn(n)
        rec = _SparseRec(w, g0.clone(), mask)
        ctx = OptContext(records=[rec], phase="sparse", bucket_states=[],
                         fold_window=fold_window, next_step_is_full=next_full,
                         apply_coeff=0.5)
        _seed(sga_optimizer_residual_mean=residual_mean)
        opt = SgaAdamS()
        opt.sparse_step(ctx)

        assert torch.all(rec.grad.data[active] == 0)
        if residual_mean and next_full:
            torch.testing.assert_close(
                rec.grad.data[~active], g0[~active] / fold_window,
                rtol=0, atol=0)
        else:
            assert torch.equal(rec.grad.data[~active], g0[~active])

        bucket = _Bucket(rec.grad.data, mask)
        ctx.bucket_states = [bucket]
        post_step = rec.grad.data.clone()
        opt.reset(ctx, was_full=False)
        assert torch.equal(bucket.grad_data, post_step)
        assert torch.all(bucket.extra_main_grads[0] == 0)
        opt.reset(ctx, was_full=True)
        assert torch.all(bucket.grad_data == 0)

    # degenerate mask-less bucket: reset zeroes it even on a sparse step
    bare = _Bucket(torch.randn(n), None)
    ctx = OptContext(records=[], phase="sparse", bucket_states=[bare])
    _seed()
    SgaAdamS().reset(ctx, was_full=False)
    assert torch.all(bare.grad_data == 0)
    print("ok: SgaAdamS sparse end-state invariant + reset contract")


def test_sga_adams_compact_dispatch_cpu():
    """Compact-emission sparse step through SgaAdamS.sparse_step on the
    CPU fallback: compact_search 'table' must engage the fallback
    SILENTLY (no eager dir_table build) and be bitwise 'bisection'; the
    values are actually consumed and the communicated coords re-cleared.
    The committed v equals the AdamS recursion (m-base) on the selected
    coords — the compact path carries the same spec."""
    from sga.support import BucketSupport, Support

    torch.manual_seed(23)
    n = 512
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    idx = mask.nonzero(as_tuple=True)[0]
    values = torch.randn(idx.numel())
    param0, g0 = torch.randn(n), torch.randn(n)
    m0 = torch.randn(n) * 0.3

    def run(compact_search):
        bs = SimpleNamespace(support_index=0)
        rec = _CompactRecPack(param0.clone(), g0.clone(), bs, 0, m0.clone())
        support = Support(buckets=[BucketSupport(mask_u8=mask)])
        ctx = OptContext(records=[rec], phase="sparse", support=support,
                         bucket_states=[], fold_window=8,
                         next_step_is_full=False, apply_coeff=0.5,
                         compact_values={id(bs): values.clone()})
        _seed(sga_optimizer_compact_search=compact_search)
        SgaAdamS().sparse_step(ctx)
        return rec

    rec_tab = run("table")
    rec_bin = run("bisection")
    for attr in ("main_param", ):
        assert torch.equal(getattr(rec_tab, attr).data,
                           getattr(rec_bin, attr).data)
    assert torch.equal(rec_tab.grad.data, rec_bin.grad.data)
    assert torch.equal(rec_tab._state["exp_avg"],
                       rec_bin._state["exp_avg"])
    assert torch.equal(rec_tab._state["exp_avg_sq"],
                       rec_bin._state["exp_avg_sq"])
    # not a silent no-op: the values entered m from the pre-commit m0 base
    assert torch.count_nonzero(rec_tab._state["exp_avg"]).item() > 0
    assert torch.count_nonzero(rec_tab.grad.data[idx]).item() == 0
    # the AdamS spec on the selected coords (fresh state, coeff folded in)
    g_eff = torch.zeros(n)
    g_eff[idx] = values * 0.5
    v_ref = 0.95 * (m0 * m0) + 0.05 * g_eff * g_eff
    torch.testing.assert_close(rec_tab._state["exp_avg_sq"], v_ref,
                               rtol=0, atol=0)
    print("ok: SgaAdamS compact CPU dispatch (table == bisection, spec v)")


class _CompactRecPack(_Rec):
    def __init__(self, param, grad, bucket_state, start, m0):
        super().__init__(param, grad)
        self.bucket_state = bucket_state
        self.start = start
        self._state = {"exp_avg": m0.clone(),
                       "exp_avg_sq": torch.zeros_like(m0)}


def test_sga_adams_dual_lr_semantics():
    """Boundary dual-lr at SgaAdamS level: the m/v commit and the
    materialized update score are BITWISE the single-lr run (mask-blind),
    active coords' apply bitwise unchanged, the tail steps at exactly
    lr*scale (the split multiplies the whole AdamS update, wd included),
    a mask-less record keeps plain lr, and scale 0 freezes the tail."""
    torch.manual_seed(13)
    n = 256
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    active = mask > 0
    for scale in (0.0, 0.5, 2.0):
        w = torch.randn(n)
        g0 = torch.randn(n)
        m0 = torch.randn(n) * 0.1
        rec_a = _SeedRec(w.clone(), g0.clone(), mask, m0)
        rec_b = _SeedRec(w.clone(), g0.clone(), mask, m0)
        rec_c = _SeedRec(w.clone(), g0.clone(), None, m0)
        _seed(sga_optimizer_full_sync_inactive_lr_scale=scale)
        opt_split = SgaAdamS()
        _seed()
        opt_plain = SgaAdamS()
        opt_split.full_step(_RecCtx([rec_a]))
        opt_plain.full_step(_RecCtx([rec_b]))
        opt_split.full_step(_RecCtx([rec_c]))

        # commit + score mask-blind: bitwise the single-lr run
        assert torch.equal(rec_a._state["exp_avg"],
                           rec_b._state["exp_avg"]), scale
        assert torch.equal(rec_a._state["exp_avg_sq"],
                           rec_b._state["exp_avg_sq"]), scale
        assert torch.equal(rec_a.grad.data, rec_b.grad.data), scale
        # mask-less record: plain lr everywhere, bitwise the plain run
        assert torch.equal(rec_c.main_param.data, rec_b.main_param.data)
        # exact apply oracle: the materialized update IS the update vector
        update = rec_a.grad.data
        lr_eff = torch.where(active, 1e-3, 1e-3 * scale)
        assert torch.equal(rec_a.main_param.data, w - lr_eff * update), scale
        if scale == 0.0:
            assert torch.equal(rec_a.main_param.data[~active], w[~active])

    # default knob (1.0): full_step must not consult the mask at all —
    # _Rec.mask_u8() raises by contract, so this call passing IS the pin
    _seed()
    SgaAdamS().full_step(_RecCtx([_Rec(torch.randn(n), torch.randn(n))]))
    print("ok: SgaAdamS dual-lr semantics (mask-blind commit, tail lr*scale)")


class _SeedRec(_Rec):
    def __init__(self, param, grad, mask, m0):
        super().__init__(param, grad)
        self._mask = mask
        self._state = {"exp_avg": m0.clone(),
                       "exp_avg_sq": torch.rand_like(m0) * 0.05}

    def mask_u8(self):
        return self._mask


# ------------------------------------------------------- config validation

def _valid_ns(**over):
    """A minimal VALID enabled namespace (histogram-balanced selection,
    torch_pipelined_index backend) with overrides."""
    ns = {
        **sga_defaults(),
        "sga_enabled": True,
        "sga_score": "update",
        "sga_selection": "topk",
        "sga_selection_method": "histogram",
        "sga_selection_histogram_mode": "balanced",
        "sga_selection_ratio_percent": 30.0,
        "sga_backend": "torch_pipelined_index",
        "sga_optimizer": "adamw",
        "sga_cycle_period": 8,
    }
    ns.update(over)
    return SimpleNamespace(**ns)


def test_validation_adams_shadow_refused():
    """AdamS x shadow is refused: shadow is the one run shape whose
    optimizer step is the STOCK megatron optimizer (a literal replication
    of MixedPrecisionOptimizer.step), so an AdamS-configured shadow run
    would silently train AdamW — the every-path-own-step guarantee is
    enforced here. AdamW x shadow stays legal."""
    validate_sga_args(_valid_ns(sga_optimizer="adams"))  # adams alone: OK
    validate_sga_args(_valid_ns(sga_shadow=True))        # adamw shadow: OK
    try:
        validate_sga_args(_valid_ns(sga_optimizer="adams", sga_shadow=True))
        raise AssertionError("AdamS x shadow accepted")
    except ValueError as e:
        assert "adams" in str(e) and "sga_shadow" in str(e), e
        assert "stock" in str(e), e
    print("ok: validation refuses AdamS x shadow (stock-step fallback)")


def test_validation_optimizer_gates_isomorphic():
    """The existing optimizer-axis gates fire IDENTICALLY for both
    template names: the boundary dual-lr knob's shadow refusal, its
    period-1 dead-config refusal and its domain rule, plus the
    legal-knob pass — one loop, two optimizers."""
    for name in ("adamw", "adams"):
        # legal dual-lr on a period-8 non-shadow run passes for both
        validate_sga_args(_valid_ns(
            sga_optimizer=name,
            sga_optimizer_full_sync_inactive_lr_scale=0.5))
        # shadow + dual-lr refuses for both: adamw via the dual-lr rule,
        # adams via its own (earlier) shadow refusal — the GATE is
        # isomorphic (the combination never validates), the firing rule
        # differs by design
        try:
            validate_sga_args(_valid_ns(
                sga_optimizer=name, sga_shadow=True,
                sga_optimizer_full_sync_inactive_lr_scale=0.5))
            raise AssertionError(f"dual-lr+shadow accepted for {name}")
        except ValueError as e:
            assert ("full_sync_inactive_lr_scale" in str(e)
                    or "sga_shadow" in str(e)), (name, e)
        # period 1 + dual-lr refuses for both (dead config)
        try:
            validate_sga_args(_valid_ns(
                sga_optimizer=name, sga_cycle_period=1,
                sga_optimizer_full_sync_inactive_lr_scale=0.5))
            raise AssertionError(f"dual-lr+period1 accepted for {name}")
        except ValueError as e:
            assert "sga_cycle_period 1" in str(e), (name, e)
        # domain rule: negative / nan refuse for both
        for bad in (-1.0, float("nan")):
            try:
                validate_sga_args(_valid_ns(
                    sga_optimizer=name,
                    sga_optimizer_full_sync_inactive_lr_scale=bad))
                raise AssertionError(f"{bad} accepted for {name}")
            except ValueError as e:
                assert "full_sync_inactive_lr_scale" in str(e), (name, e)
    print("ok: optimizer gates isomorphic across adamw/adams")


def test_registry_adams_face():
    """The class-registry face for the new template: idempotent default
    fill, duplicate-registration refusal, curated unknown-name error
    listing BOTH templates, and the SgaAdamW/SgaAdamS classes stay
    distinct (no accidental aliasing in the registry)."""
    from sga.plugins import _OPTIMIZERS

    register_default_plugins()
    register_default_plugins()  # idempotent
    assert plugin_class("optimizer", "adams") is SgaAdamS
    assert plugin_class("optimizer", "adamw") is SgaAdamW
    assert _OPTIMIZERS["adamw"] is not _OPTIMIZERS["adams"]
    try:
        register_optimizer("adams", SgaAdamS)
        raise AssertionError("duplicate adams accepted")
    except ValueError as e:
        assert "adams" in str(e), e
    try:
        plugin_class("optimizer", "nope")
        raise AssertionError("unknown optimizer accepted")
    except ValueError as e:
        assert "Unknown SGA optimizer plugin" in str(e), e
        assert "adams" in str(e) and "adamw" in str(e), e
    print("ok: registry face (adams registered, duplicates, unknown list)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
