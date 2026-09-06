#!/usr/bin/env python3
"""Unit tests for the SGA plugin layer (score registry / selections / optimizer), CPU-only.

Covers:
- score registry formulas vs straightforward fp64 oracles (all four scores,
  scratch-routed == freshly-allocated, wd-free update variant);
- selective reset semantics as fused into the sparse grad-correct pass
  (clear / fold / coeff == prescale, plus the SgaAdamW end-state invariant);
- topk: histogram threshold == exact torch.kthvalue threshold,
  mask equality, k rounding bound, three-tier geometry + chunking;
- adaptive controller band logic, clamps and lifecycle;
- declaration-based combination validation + build injection + decl coverage
  (incl. the emission registration gate and the compact capability rule);

Run: python tests/test_plugins_unit.py  (or pytest tests/test_plugins_unit.py)
"""

import os
import sys
import warnings
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.plugins import (
    BackendDecl,
    OptimizerDecl,
    _register,
    get_backend_plugin_decl,
    get_optimizer_decl,
    get_selection_decl,
    register_default_plugins,
    validate_combination,
)
from sga.plugins.optimizer import OptContext
from sga.plugins.optimizer.adamw import SgaAdamW, adamw_grad_correct
from sga.plugins.optimizer.adamw_torch import grad_correct_torch
from sga.plugins.scores import SCRATCH_SCORES, get_score_spec, score_names
from sga.plugins.selection._exact import select_masks_exact_kthvalue
from sga.plugins.selection._histogram import (
    _HIST_BINS,
    _abs_bit_top16,
    _histogram_select_threshold,
    select_masks_histogram,
)
from sga.plugins.selection.topk import TopkSelection


def _method_allows(method: str, key: str) -> bool:
    """Is `key` a legal topk param under `method`? (topk's own literal
    per-method vocabularies: base params always, guarded params under
    the method that unlocks them.)"""
    guarded = {"sga_selection_histogram_mode": "histogram",
               "sga_selection_adaptive_warmup_syncs": "adaptive",
               "sga_selection_adaptive_warmup_mode": "adaptive"}
    return (key in ("sga_selection_method", "sga_selection_ratio_percent")
            or (key in guarded and guarded[key] == method))

torch.manual_seed(0)


class _Ctx:
    def __init__(self, grad_views, comm_group=None, warmup_target=None):
        self.grad_views = grad_views
        self.comm_group = comm_group
        self.warmup_target = warmup_target


# ------------------------------------------------------------- score registry

def test_old_selection_name_fails_loud():
    """circular_topk (the pre-rename name of topk) must fail with the
    curated unknown-plugin error — not a KeyError from the dispatch dict."""
    from sga.plugins import build_selection
    try:
        build_selection("circular_topk", {"sga_selection_ratio_percent": 20.0})
        raise AssertionError("old name accepted")
    except ValueError as e:
        assert "circular_topk" in str(e) and "['topk']" in str(e), e
    print("ok: old selection name fails loud with the registered list")


def test_old_backend_name_fails_loud():
    """Pre-unification backend names must fail with a migration hint that
    names the reproducing configuration — not a silent dict miss."""
    from sga.plugins import get_backend_plugin_decl
    for old, hint in (
        ("indexed_sparse", "high_priority_group: false"),
        ("pipelined_idx_sorted", "high_priority_group: true, index_slices: 3"),
        ("masked_pipelined", "torch_pipelined_mask"),
    ):
        try:
            get_backend_plugin_decl(old)
            raise AssertionError(f"old name accepted: {old}")
        except ValueError as e:
            assert old in str(e) and hint in str(e), (old, e)
    print("ok: old backend names fail loud with migration hints")


def test_registry_contents():
    assert score_names() == ["abs_g", "abs_m", "coi", "update"]
    for name in score_names():
        spec = get_score_spec(name)
        assert callable(spec.formula) and spec.doc
    try:
        get_score_spec("nope")
        raise AssertionError("unknown score accepted")
    except ValueError as e:
        assert "nope" in str(e) and "abs_g" in str(e), e


def _formula_oracles(w, m, v, g, b1, b2, eps, wd, bc1, bc2):
    """Straightforward fp64 oracles, written independently of the formulas."""
    w64, m64, v64, g64 = (t.double() for t in (w, m, v, g))
    m_new = b1 * m64 + (1 - b1) * g64
    v_new = b2 * v64 + (1 - b2) * g64 * g64
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    u_g = m_hat / (v_hat.sqrt() + eps)
    u_0 = (b1 * m64 / bc1) / ((b2 * v64 / bc2).sqrt() + eps)
    return {
        "abs_g": g64.abs(),
        "abs_m": m64.abs(),
        "update": wd * w64 + u_g,
        "coi": (u_g - u_0).abs(),
    }


def test_score_formulas_match_fp64_oracles():
    torch.manual_seed(3)
    n, b1, b2, eps, wd, step = 513, 0.9, 0.95, 1e-8, 1e-2, 7
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    w = torch.randn(n)
    m = torch.randn(n) * 0.1
    v = torch.rand(n) * 0.1 + 1e-4
    g = torch.randn(n)
    oracles = _formula_oracles(w, m, v, g, b1, b2, eps, wd, bc1, bc2)
    for name in score_names():
        spec = get_score_spec(name)
        # no training input is mutated by any formula (shadow's contract):
        # the snapshots are taken BEFORE the call — comparing against
        # post-call clones would compare the formula against itself and
        # could never fail
        w0, m0, v0, g0 = w.clone(), m.clone(), v.clone(), g.clone()
        out = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                           update_includes_wd=True)
        torch.testing.assert_close(out.double(), oracles[name],
                                   rtol=1e-5, atol=1e-7)
        assert torch.equal(w, w0) and torch.equal(m, m0)
        assert torch.equal(v, v0) and torch.equal(g, g0)
    print("ok: score formulas match fp64 oracles")


def test_score_formula_scratch_routing_bitwise():
    """scratch-routed evaluation is bitwise identical to the freshly-allocated
    one (the out= chains must be op-for-op the same elementwise sequence),
    outputs never alias the scratch pair, the unused-scratch scores (outside
    SCRATCH_SCORES) ignore the pair entirely, and records of DIFFERENT numel
    AND rank share ONE pair sized to the max (the [:n].view_as(w) narrowing
    keeps out= ops off the shared buffers' shape — a resize would shrink the
    pair under the next, larger record). The whole scratch-routed evaluation
    runs under warnings-as-errors: an out= resize emits a UserWarning and
    must fail here, not silently rely on deprecated behavior."""
    torch.manual_seed(4)
    n = 1024
    w = torch.randn(n)
    m = torch.randn(n) * 0.1
    v = torch.rand(n) + 1e-3
    g = torch.randn(n)
    b1, b2, eps, wd, step = 0.9, 0.95, 1e-8, 1e-2, 9
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    for name in score_names():
        spec = get_score_spec(name)
        fresh = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                             update_includes_wd=True)
        scratch = (
            torch.full((n + 7,), float("nan")),  # >= numel, ragged on purpose
            torch.full((n + 7,), float("nan")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with_scratch = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                                        update_includes_wd=True,
                                        scratch=scratch)
        assert torch.equal(fresh, with_scratch), name
        assert with_scratch.data_ptr() != scratch[0].data_ptr()
        assert with_scratch.data_ptr() != scratch[1].data_ptr()
    assert SCRATCH_SCORES == frozenset(
        n for n in score_names() if get_score_spec(n).scratch
    )
    assert SCRATCH_SCORES == frozenset(("update", "coi"))

    # the wd-free variant must ALSO return a fresh tensor with a caller
    # scratch (never alias the pair — per-param outputs are held
    # simultaneously) and stay bitwise equal to its fresh evaluation
    spec = get_score_spec("update")
    fresh_nowd = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                              update_includes_wd=False)
    scratch = (torch.full((n + 7,), float("nan")),
               torch.full((n + 7,), float("nan")))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        scratch_nowd = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                                    update_includes_wd=False, scratch=scratch)
    assert torch.equal(fresh_nowd, scratch_nowd)
    assert scratch_nowd.data_ptr() not in (scratch[0].data_ptr(),
                                           scratch[1].data_ptr())

    # mixed-numel AND mixed-rank records over ONE shared pair sized to the
    # max numel (exactly the engine's per-boundary flat max_n pair), smaller
    # record first: the narrowing must keep both evaluations warning-free
    # (an out= resize would shrink the SHARED pair under the first record)
    # and leave the pair at full capacity for the larger record
    shared = (torch.full((200,), float("nan")), torch.full((200,), float("nan")))
    for shape in ((50,), (10, 20)):
        w2 = torch.randn(shape)
        m2 = torch.randn(shape) * 0.1
        v2 = torch.rand(shape) + 1e-3
        g2 = torch.randn(shape)
        fresh2 = spec.formula(w2, m2, v2, g2, b1, b2, eps, wd, bc1, bc2,
                              update_includes_wd=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            routed2 = spec.formula(w2, m2, v2, g2, b1, b2, eps, wd, bc1, bc2,
                                   update_includes_wd=True, scratch=shared)
        assert torch.equal(fresh2, routed2), shape
        assert routed2.data_ptr() not in (shared[0].data_ptr(),
                                          shared[1].data_ptr()), shape
        assert shared[0].numel() == 200 and shared[1].numel() == 200, \
            "scratch pair was resized in place"
    print("ok: score formula scratch routing is bitwise identical")


def test_update_formula_wd_free_variant():
    """update_includes_wd=False returns the wd-free U(g) (the offline
    zerofill-mode convention); True returns the online-exact signed form."""
    torch.manual_seed(5)
    n = 256
    w = torch.randn(n)
    m = torch.randn(n) * 0.1
    v = torch.rand(n) + 1e-3
    g = torch.randn(n)
    b1, b2, eps, wd, step = 0.9, 0.95, 1e-8, 1e-2, 4
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    spec = get_score_spec("update")
    with_wd = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                           update_includes_wd=True)
    without_wd = spec.formula(w, m, v, g, b1, b2, eps, wd, bc1, bc2,
                              update_includes_wd=False)
    # (a + b) - b != a exactly in fp32: the identity holds only to fp32
    # rounding of the LARGER operands, so the error is checked against the
    # operand scale, not the (cancelling) difference scale
    diff = (with_wd - without_wd).double() - wd * w.double()
    scale = max(float((wd * w.double()).abs().max()),
                float(without_wd.double().abs().max()))
    assert float(diff.abs().max()) <= 1e-6 * scale


# ------------------------------------------- sparse fusion (CPU)

def test_grad_correct_sparse_fusion_semantics():
    """grad_correct_torch fusion knobs == the standalone sequence: coeff is
    a masked prescale, clear zeroes the active positions, fold divides the
    inactive residual by the window — every buffer bitwise."""
    torch.manual_seed(7)
    n = 64
    b1, b2, step = 0.9, 0.95, 5
    kw = dict(lr=1e-3, beta1=b1, beta2=b2, eps=1e-8, weight_decay=1e-2,
              bias_correction1=1 - b1 ** step,
              bias_correction2=1 - b2 ** step)
    for fold, interval in [(False, 8), (True, 8)]:
        for coeff in (0.5, 1.0):
            g0 = torch.randn(n)
            mask = (torch.rand(n) < 0.25).to(torch.uint8)
            p, m, v = torch.randn(n), torch.randn(n) * .1, torch.rand(n) * .1

            p1, g1, m1, v1 = p.clone(), g0.clone(), m.clone(), v.clone()
            grad_correct_torch(p1, g1, m1, v1, mask=mask, emit_mode=None,
                               coeff=coeff, clear=True, fold=fold,
                               interval=float(interval), **kw)

            p2, m2, v2 = p.clone(), m.clone(), v.clone()
            g2 = g0.clone()
            if coeff != 1.0:
                g2 = torch.where(mask > 0, g2 * coeff, g2)
            grad_correct_torch(p2, g2, m2, v2, mask=mask, emit_mode=None,
                               clear=True, fold=fold,
                               interval=float(interval), **kw)

            for got, ref, name in ((p1, p2, "param"), (m1, m2, "exp_avg"),
                                   (v1, v2, "exp_avg_sq"), (g1, g2, "grad")):
                assert torch.equal(got, ref), (
                    f"fused sparse path diverged ({name}, coeff={coeff}, "
                    f"fold={fold})")


# --------------------------------------------------------------- topk

def test_histogram_matches_exact_method():
    for trial in range(5):
        grads = [torch.randn(1000), torch.randn(317), torch.randn(1)]
        k = 100 + trial * 37
        t_h, m_h = select_masks_histogram(grads, k, None)
        t_e, m_e = select_masks_exact_kthvalue(grads, k, None)
        assert t_h == t_e, (t_h, t_e)
        for a, b in zip(m_h, m_e):
            assert torch.equal(a, b)
        n_sel = sum(int(torch.count_nonzero(m).item()) for m in m_h)
        assert n_sel >= k  # ties at the boundary are all selected


def test_topk_k_rounding():
    sel = TopkSelection({"sga_selection_method": "histogram", "sga_selection_ratio_percent": 10.0})
    n = 1000
    assert sel.k_for(n) == 100
    # rounding bound from k = max(1, round(n * r)): |selected/n - r| <= 0.5/n
    assert abs(sel.k_for(999) / 999 - 0.10) <= 0.5 / 999 + 1e-12


def test_warmup_sync_target_anchor():
    """v3.1 warmup anchor, owned by the selection itself (the engine only
    calls the optional hook with start_iter/period): a window of N cycles of
    T iters from the run start leaves max(0, (N*T - 1 - start) // T) gated
    syncs for a selection starting at start_iter; a late start (window
    already over) leaves a single bootstrap; an unset window yields None."""
    sel = TopkSelection({"sga_selection_method": "adaptive", "sga_selection_ratio_percent": 40.0,
                                 "sga_selection_adaptive_warmup_syncs": 5})
    assert sel.warmup_sync_target(start_iter=0, period=4) == 4  # syncs at 4,8,12,16
    assert sel.warmup_sync_target(start_iter=6, period=4) == (20 - 1 - 6) // 4
    assert sel.warmup_sync_target(start_iter=25, period=4) == 0

    unset = TopkSelection({"sga_selection_method": "adaptive", "sga_selection_ratio_percent": 40.0,
                                   "sga_selection_adaptive_warmup_syncs": None})
    assert unset.params["sga_selection_adaptive_warmup_syncs"] is None
    assert unset.warmup_sync_target(start_iter=0, period=4) is None

    # null syncs + no engine anchor (warmup_target=None): the gate
    # short-circuits to a single bootstrap, then the persistent tau path
    # takes over (an earlier revision crashed here with max(None, 1))
    torch.manual_seed(4)
    grads = [torch.randn(2048)]
    sel0 = TopkSelection({"sga_selection_method": "adaptive", "sga_selection_ratio_percent": 10.0,
                          "sga_selection_adaptive_warmup_syncs": None})
    masks = sel0.select(_Ctx(grads, comm_group=None, warmup_target=None))
    assert sum(int(torch.count_nonzero(m).item()) for m in masks) > 0
    assert sel0.tau is not None and sel0.tau > 0.0
    sel0.select(_Ctx(grads, comm_group=None, warmup_target=None))
    assert sel0.last_threshold == sel0.tau  # steady state: tau-driven


def test_histogram_world_replica_threshold():
    """World>1 scaling terms: with two bit-identical replicas the merged
    histogram is exactly 2x the local one, and k stays in LOCAL units —
    the threshold must equal one copy's own torch.kthvalue result."""
    g = torch.randn(2048)
    k = 333
    hist_local = torch.zeros(_HIST_BINS, dtype=torch.int64)
    hist_local += torch.bincount(_abs_bit_top16(g), minlength=_HIST_BINS)
    merged = hist_local * 2

    t_world2 = _histogram_select_threshold(merged.clone(), k, world=2,
                                           grads=[g.clone()])
    ref = g.abs()
    t_ref = torch.kthvalue(ref, ref.numel() - k + 1).values.item()
    assert t_world2 == t_ref, (t_world2, t_ref)


def test_hist_tiers_geometry_and_chunking():
    """Three-tier geometry: chunked passes are bitwise identical
    to unchunked ones, exact stays bit-exact vs the kthvalue control, and
    the approximate tiers meet the two-sided production closeness criterion
    vs exact (0.5pp balanced / 2pp fast). The tiers carry an over-selection
    BIAS but NOT a strict-superset guarantee — stride-subsampling
    fluctuation allows small under-selection (see _histogram.py's geometry
    note); do not assert supersets here."""
    import sga.plugins.selection._histogram as _h

    torch.manual_seed(11)
    # wide dynamic range so both the fine (16-bit) and coarse (13-bit) bin
    # geometries are actually crossed; PRODUCTION SCALE (~1M elements) —
    # the two-sided count criterion below is only meaningful there (at
    # n~10k the stride subsample's own fluctuation exceeds the bands)
    grads = [torch.randn(1000000) * 1e-3, torch.randn(31000) * 1e3,
             torch.randn(3000)]
    n_total = sum(g.numel() for g in grads)
    k = int(round(n_total * 0.10))

    t_exact, m_exact = select_masks_histogram(grads, k, None)
    t_ctl, m_ctl = select_masks_exact_kthvalue(grads, k, None)
    assert t_exact == t_ctl
    for a, b in zip(m_exact, m_ctl):
        assert torch.equal(a, b)

    # chunked == unchunked (chunk patched to a stride multiple)
    old_chunk = _h._HIST_CHUNK
    assert old_chunk % _h._HIST_SAMPLE_STRIDE == 0
    _h._HIST_CHUNK = 1024
    try:
        t_ch, m_ch = select_masks_histogram(grads, k, None)
        assert t_ch == t_exact
        for a, b in zip(m_ch, m_exact):
            assert torch.equal(a, b)
        shift, nbins, stride = _h._HIST_MODES["fast"]
        h_chunked = _h._build_histogram(grads, shift, nbins, stride)
    finally:
        _h._HIST_CHUNK = old_chunk
    assert torch.equal(h_chunked, _h._build_histogram(grads, shift, nbins, stride))

    # approximate tiers: two-sided closeness of the selected count vs exact
    # (the same criterion production uses — NOT a superset relation) and
    # bin-edge thresholds
    n_exact = sum(int(torch.count_nonzero(m).item()) for m in m_exact)
    tol = {"balanced": 0.005, "fast": 0.02}
    for mode in ("balanced", "fast"):
        t_app, m_app = _h.select_masks_histogram(grads, k, None, mode=mode)
        n_app = sum(int(torch.count_nonzero(m).item()) for m in m_app)
        assert abs(n_app - n_exact) / n_total < tol[mode], \
            (mode, n_app, n_exact)
        lo_bits = 16 if mode == "balanced" else 19
        bits = torch.tensor(t_app, dtype=torch.float32).view(torch.int32).item()
        assert (bits & ((1 << lo_bits) - 1)) == 0, (mode, hex(bits))

    # the approximate crossing-bin rank is scale-invariant in world size
    # (merged hist = world x local; r scales by world -> identical bin)
    r1 = max(1, round(sum((g.numel() + stride - 1) // stride for g in grads)
                      * (n_total - k + 1) / n_total))
    hist = _h._build_histogram(grads, shift, nbins, stride)
    b1 = int(torch.searchsorted(torch.cumsum(hist, 0), r1).item())
    b3 = int(torch.searchsorted(torch.cumsum(hist * 3, 0), r1 * 3).item())
    assert b1 == b3

    # selection plumbing: histogram_mode flows through, unknown modes rejected
    sel = TopkSelection({"sga_selection_method": "histogram", "sga_selection_ratio_percent": 10.0,
                        "sga_selection_histogram_mode": "fast"})
    sel.select(_Ctx(grads, comm_group=None))
    bits = torch.tensor(sel.last_threshold, dtype=torch.float32).view(torch.int32).item()
    assert (bits & ((1 << 19) - 1)) == 0
    try:
        TopkSelection({"sga_selection_method": "histogram", "sga_selection_ratio_percent": 10.0,
                        "sga_selection_histogram_mode": "turbo"})
        raise AssertionError("unknown histogram_mode accepted")
    except ValueError:
        pass
    print("ok: hist tiers (exact/balanced/fast) geometry, chunking, closeness")


def test_histogram_block_selection():
    """Block-granular histogram selection: block-mean/max
    reduce, real-length tail blocks, mask expansion structure, k_blocks
    rounding, exact-vs-histogram threshold identity on block scores,
    empty-view tolerance, and constructor/foreign-key guards."""
    import sga.plugins.selection._histogram as _h
    from sga.plugins.selection._histogram import (
        HistogramSelect, _block_reduce_views)

    # --- reduce correctness incl. real-length tails ---
    torch.manual_seed(21)
    g = torch.randn(1000)          # 1000 = 62*16 + 8 -> tail block of 8
    red_mean = _block_reduce_views([g], 16, "mean")[0]
    ref = torch.cat([g[:62 * 16].abs().view(62, 16).mean(1),
                     g[62 * 16:].abs().mean().reshape(1)])
    assert torch.equal(red_mean, ref)
    red_max = _block_reduce_views([g], 16, "max")[0]
    ref_max = torch.cat([g[:62 * 16].abs().view(62, 16).amax(1),
                         g[62 * 16:].abs().amax().reshape(1)])
    assert torch.equal(red_max, ref_max)
    # chunked == unchunked reduce (chunk patched small but x-aligned)
    old_chunk = _h._HIST_CHUNK
    _h._HIST_CHUNK = 16 * 40
    try:
        red_ch = _block_reduce_views([g], 16, "mean")[0]
    finally:
        _h._HIST_CHUNK = old_chunk
    assert torch.equal(red_ch, ref)

    # --- tail block scored by REAL length (a sum would discriminate) ---
    # 1-element tail holding the global max must beat a full block of
    # mediums under mean (and under max).
    grads = [torch.cat([torch.full((64,), 10.0), torch.tensor([50.0])])]
    for reduce in ("mean", "max"):
        _t, m = HistogramSelect("exact", block_size=64,
                                block_reduce=reduce).select(grads, 64, None)
        # k_blocks = round(64/64) = 1: the tail block wins
        assert int(m[0][64].item()) == 1 and int(m[0][:64].sum().item()) == 0

    # --- mask structure: all-or-nothing blocks, tail clamped ---
    torch.manual_seed(22)
    grads = [torch.randn(1000), torch.randn(37)]
    x, k = 16, 200
    _t, masks = HistogramSelect("exact", block_size=x).select(grads, k, None)
    for mm, gg in zip(masks, grads):
        assert mm.dtype == torch.uint8 and mm.numel() == gg.numel()
        n = gg.numel()
        pad = (-n) % x
        rows = torch.cat([mm, torch.zeros(pad, dtype=torch.uint8)]
                         ).view(-1, x).sum(1)
        lens = [x] * (rows.numel() - 1) + [x - pad]
        assert all(int(s) in (0, L) for s, L in zip(rows.tolist(), lens))
    n_sel = sum(int(mm.sum()) for mm in masks)
    assert abs(n_sel - max(1, round(k / x)) * x) <= x

    # --- histogram block threshold == exact kthvalue on the block scores,
    #     and the expanded mask is exactly per-block (score >= thresh) ---
    torch.manual_seed(23)
    grads = [torch.randn(100000), torch.randn(317)]
    x, k = 32, 5000
    t_h, m_h = HistogramSelect("exact", block_size=x).select(grads, k, None)
    bviews = _block_reduce_views(grads, x, "mean")
    bcat = torch.cat(bviews)
    kb = max(1, round(k / x))
    t_ref = torch.kthvalue(bcat, bcat.numel() - kb + 1).values.item()
    assert t_h == t_ref, (t_h, t_ref)
    n_sel = 0
    for mm, bv, gg in zip(m_h, bviews, grads):
        ref_m = (bv >= t_ref).repeat_interleave(x)[:gg.numel()]
        assert torch.equal(mm, ref_m.to(torch.uint8))
        n_sel += int(ref_m.sum())
    assert n_sel >= kb  # boundary ties all selected

    # --- mean vs max genuinely differ on spiky blocks ---
    spike = torch.zeros(64)
    spike[0] = 100.0                                # mean 1.5625, max 100
    flat = torch.full((64,), 10.0)                  # mean 10, max 10
    grads = [torch.cat([spike, flat])]
    _t, m_mean = HistogramSelect("exact", block_size=64,
                                 block_reduce="mean").select(grads, 64, None)
    _t, m_max = HistogramSelect("exact", block_size=64,
                                block_reduce="max").select(grads, 64, None)
    assert int(m_mean[0][64:].sum()) == 64 and int(m_mean[0][:64].sum()) == 0
    assert int(m_max[0][:64].sum()) == 64 and int(m_max[0][64:].sum()) == 0

    # --- empty grad view: tolerated, not a crash (an empty view enters no
    #     reduce chunk and must come back as an empty mask, like the
    #     element path) ---
    grads = [torch.randn(48), torch.empty(0)]
    bviews = _block_reduce_views(grads, 16, "mean")
    assert bviews[0].numel() == 3 and bviews[1].numel() == 0
    _t, masks = HistogramSelect("exact", block_size=16).select(grads, 16, None)
    assert masks[1].dtype == torch.uint8 and masks[1].numel() == 0
    assert masks[0].numel() == 48

    # --- TopkSelection plumbing + constructor/foreign-key guards ---
    sel = TopkSelection({"sga_selection_method": "histogram", "sga_selection_ratio_percent": 10.0,
                         "sga_selection_block_size": 16})
    assert sel.block_size == 16 and sel.params["sga_selection_block_reduce"] == "mean"
    assert TopkSelection({"sga_selection_method": "histogram",
                          "sga_selection_ratio_percent": 10.0}).block_size == 1
    assert TopkSelection({"sga_selection_method": "exact",
                          "sga_selection_ratio_percent": 10.0}).block_size == 1
    for bad in (0, 3, True, -16):
        try:
            HistogramSelect("exact", block_size=bad)
            raise AssertionError(f"bad block_size accepted: {bad}")
        except ValueError:
            pass
    try:
        HistogramSelect("exact", block_size=16, block_reduce="sum")
        raise AssertionError("bad block_reduce accepted")
    except ValueError:
        pass
    try:
        TopkSelection({"sga_selection_method": "exact", "sga_selection_ratio_percent": 10.0,
                       "sga_selection_block_size": 16})
        raise AssertionError("block_size accepted under method=exact")
    except ValueError as e:
        assert "do not belong" in str(e)
    print("ok: block histogram selection (reduce/tail/structure/k_blocks)")


def test_adaptive_threshold_band_and_clamps():
    from sga.plugins.selection._adaptive import _adapt_threshold

    tau = 1.0
    # inside the ±1pp dead band: unchanged
    assert _adapt_threshold(tau, 0.105, 0.10) == tau
    assert _adapt_threshold(tau, 0.095, 0.10) == tau
    # out of band: log-space step (measured/target)^gain
    assert _adapt_threshold(tau, 0.13, 0.10) == 1.3 ** 0.5
    assert _adapt_threshold(tau, 0.085, 0.10) == 0.85 ** 0.5
    # clamps: a huge deviation steps at most +15% / -10%
    assert _adapt_threshold(tau, 0.20, 0.10) == 1.15
    assert _adapt_threshold(tau, 0.0001, 0.10) == 0.9
    print("ok: adaptive threshold band + log-space clamps")


def test_topk_adaptive_lifecycle():
    """Bootstrap -> gated warmup -> persistent tau feedback -> collapse
    re-bootstrap (v3.1 semantics, default fast-mode warmup histogram)."""
    torch.manual_seed(12)
    grads = [torch.randn(4096), torch.randn(1000)]
    sel = TopkSelection({"sga_selection_method": "adaptive", "sga_selection_ratio_percent": 10.0})
    assert sel._impl.warmup_mode == "fast"

    # 1) bootstrap from the warmup-mode histogram; tau is its bin edge
    sel.select(_Ctx(grads, comm_group=None, warmup_target=3))
    bits = torch.tensor(sel.last_threshold, dtype=torch.float32).view(torch.int32).item()
    assert (bits & ((1 << 19) - 1)) == 0
    assert sel.tau == sel.last_threshold
    assert sel.warmup_passed in (0, 1)  # gated on the measured ratio

    # 2) a miss extends the warmup (never escalates): still bootstrapping
    sel._impl.warmup_passed = 0
    sel.select(_Ctx(grads, comm_group=None, warmup_target=3))
    assert sel.tau == sel.last_threshold

    # 3) after the gate: masks == |g| >= tau, built with the OLD tau while
    # the feedback moves the persistent tau for the next cycle
    sel._impl.warmup_passed = 3
    tau = sel.tau
    masks = sel.select(_Ctx(grads, comm_group=None, warmup_target=3))
    for g, m in zip(grads, masks):
        assert torch.equal(m, (g.abs() >= tau).to(torch.uint8))
    assert sel.last_threshold == tau

    # 4) collapse guard: everything below tau -> re-bootstrap from histogram
    sel._impl.tau = 1e30
    masks = sel.select(_Ctx(grads, comm_group=None, warmup_target=3))
    assert sel.tau < 1e30
    assert any(int(torch.count_nonzero(m).item()) for m in masks)

    # 5) a latched zero tau cannot persist: the collapse guard treats
    #    tau <= 0 like a zero measurement, re-bootstraps, and clamps the new
    #    threshold to the smallest strictly-positive bin edge of the warmup
    #    geometry (bin 0's lower edge is exactly 0.0; the multiplicative
    #    feedback could otherwise never leave zero).
    import unittest.mock as mock

    import sga.plugins.selection._adaptive as adapt_mod
    from sga.plugins.selection._histogram import _bin_edge_float

    sel._impl.warmup_passed = 5
    sel._impl.tau = 0.0
    canned_zero = (0.0, [(grads[0].abs() >= 0.10).to(torch.uint8)])
    with mock.patch.object(adapt_mod, "select_masks_histogram",
                           return_value=canned_zero):
        masks = sel.select(_Ctx(grads, comm_group=None, warmup_target=0))
    assert sel.tau > 0.0, sel.tau
    assert sel.tau == _bin_edge_float(1, 19), sel.tau  # fast mode floor
    assert all(int(torch.count_nonzero(m).item()) for m in masks)

    # per-method vocabulary: a foreign method's knob is rejected loudly —
    # it would be silently dead under the configured method (full product)
    for method in ("adaptive", "exact", "histogram"):
        for key, val in (("sga_selection_histogram_mode", "exact"),
                         ("sga_selection_adaptive_warmup_syncs", 3),
                         ("sga_selection_adaptive_warmup_mode", "fast")):
            cfg = {"sga_selection_method": method, "sga_selection_ratio_percent": 10.0, key: val}
            if _method_allows(method, key):
                TopkSelection(cfg)  # legal under its own method
                continue
            try:
                TopkSelection(cfg)
                raise AssertionError(f"{method} + {key} accepted")
            except ValueError as e:
                assert key in str(e) and method in str(e), e
    print("ok: topk adaptive lifecycle (bootstrap/warmup/tau/collapse/"
          "tau-zero latch) + per-method foreign-key rejection")


# ------------------------------------------------- score emission (full step)

class _Rec:
    """Minimal ParamRecord stand-in for SgaAdamW.full_step.

    __hash__ = None mirrors the engine's @dataclass ParamRecord (value-eq
    classes are unhashable): any dict keyed by the record itself fails here
    first, not on the cluster."""

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


def test_score_emission_matches_oracles():
    """Each emitted score, its buffer swap, and the applied update match a
    hand oracle on pre/post-commit state (fallback path, CPU)."""
    torch.manual_seed(3)
    n, b1, b2, step = 512, 0.9, 0.95, 7
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    for mode in ("coi", "abs_g", "abs_m"):
        param0 = torch.randn(n)
        grad0 = torch.randn(n)
        m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1

        w = param0.clone().requires_grad_(False)
        g = grad0.clone().requires_grad_(False)
        rec = _Rec(w, g)
        opt = SgaAdamW({"sga_optimizer_residual_mean": False}, emit_score=mode)
        opt.full_step(_RecCtx([rec]))

        # m0/v0 start at zero here (state is lazily created)
        m_ref = b1 * torch.zeros(n) + (1 - b1) * grad0
        v_ref = b2 * torch.zeros(n) + (1 - b2) * grad0 * grad0
        u_ref = 1e-2 * param0 + (m_ref / bc1) / ((v_ref / bc2).sqrt() + 1e-8)
        if mode == "coi":
            u0_ref = torch.zeros(n)
            score_ref = ((m_ref / bc1) / ((v_ref / bc2).sqrt() + 1e-8) - u0_ref).abs()
        elif mode == "abs_g":
            score_ref = grad0.abs()
        else:
            score_ref = torch.zeros(n)  # |m| with m pre-commit zero
        torch.testing.assert_close(g.data, score_ref, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(w.data, param0 - 1e-3 * u_ref,
                                   rtol=1e-5, atol=1e-7)

    # with pre-existing moments, coi is non-trivial: |U(g) - U(0)|
    param0 = torch.randn(n)
    grad0 = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    w = param0.clone().requires_grad_(False)
    g = grad0.clone().requires_grad_(False)
    rec = _Rec(w, g)
    rec._state = {"exp_avg": m0.clone(), "exp_avg_sq": v0.clone()}
    opt = SgaAdamW({}, emit_score="coi")
    opt.full_step(_RecCtx([rec]))
    oracles = _formula_oracles(param0, m0, v0, grad0, b1, b2, 1e-8, 1e-2,
                               bc1, bc2)
    m_new = b1 * m0 + (1 - b1) * grad0
    v_new = b2 * v0 + (1 - b2) * grad0 * grad0
    torch.testing.assert_close(g.data, oracles["coi"].float(),
                               rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(w.data, param0 - 1e-3 * oracles["update"].float(),
                               rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(rec._state["exp_avg"], m_new, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(rec._state["exp_avg_sq"], v_new, rtol=1e-4, atol=1e-7)
    print("ok: score emission matches oracles (coi/abs_g/abs_m)")


# ------------------------------------- sparse end-state (CPU)

class _SparseRec(_Rec):
    """_Rec with a realizable mask for the sparse step."""

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


def test_sparse_step_end_state_invariant():
    """The sparse-fusion contract pinned end-to-end at SgaAdamW level: after
    sparse_step + reset, mask-active coords are bitwise 0 and mask-inactive
    coords keep the accumulating residual (folded by the window only on the
    pre-boundary step with residual_mean); reset() on a sparse step must
    NOT zero the bucket (zeroing would destroy the residual), on a full
    step it zeroes everything."""
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
        opt = SgaAdamW({"sga_optimizer_residual_mean": residual_mean})
        opt.sparse_step(ctx)

        assert torch.all(rec.grad.data[active] == 0)
        if residual_mean and next_full:
            torch.testing.assert_close(
                rec.grad.data[~active], g0[~active] / fold_window,
                rtol=0, atol=0)
        else:
            assert torch.equal(rec.grad.data[~active], g0[~active])

        # reset on the sparse step: snapshot compare (bucket.grad_data IS
        # rec.grad.data here, so assert against values, not identity)
        bucket = _Bucket(rec.grad.data, mask)
        ctx.bucket_states = [bucket]
        post_step = rec.grad.data.clone()
        extra_before = bucket.extra_main_grads[0].clone()
        opt.reset(ctx, was_full=False)
        assert torch.equal(bucket.grad_data, post_step)
        assert torch.all(bucket.grad_data[active] == 0)
        assert torch.all(bucket.extra_main_grads[0] == 0)
        assert not torch.equal(bucket.extra_main_grads[0], extra_before)

        opt.reset(ctx, was_full=True)
        assert torch.all(bucket.grad_data == 0)

    # degenerate mask-less bucket: reset zeroes it even on a sparse step
    bare = _Bucket(torch.randn(n), None)
    ctx = OptContext(records=[], phase="sparse", bucket_states=[bare])
    SgaAdamW({}).reset(ctx, was_full=False)
    assert torch.all(bare.grad_data == 0)


def test_compact_search_param_resolution():
    """compact_search: default 'table' (the benchmark-adjudicated prod
    form), enum validated loudly, bisection accepted. Silent no-op on
    writeback-emission runs / torch fallback is the DECL's documented
    contract (they never reach the compact merge-join)."""
    assert SgaAdamW({}).compact_search == "table"
    assert SgaAdamW({"sga_optimizer_compact_search": "bisection"}).compact_search == "bisection"
    try:
        SgaAdamW({"sga_optimizer_compact_search": "quadtree"})
        raise AssertionError("bogus compact_search accepted")
    except ValueError as e:
        assert "compact_search" in str(e), e
    print("ok: compact_search resolution (default table, enum validated)")


def test_backend_emission_param_resolution():
    """sga_backend emission: 'auto' (the flag default) resolves to each
    backend's inherent form — compact ONLY on torch_pipelined_index;
    explicit 'compact' on the writeback-inherent backends is refused with
    a pointer (direct-construction backstop; the flag-layer rule catches
    real runs earlier); bogus values fail the enum loudly."""
    from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend
    from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
    from sga.backends.torch_pipelined_mask import TorchPipelinedMaskBackend

    assert TorchPipelinedIndexBackend({}).emission == "compact"
    assert TorchPipelinedIndexBackend(
        {"sga_backend_emission": "writeback"}).emission == "writeback"
    assert TorchPipelinedIndexBackend(
        {"sga_backend_emission": "compact"}).emission == "compact"
    assert TorchPipelinedMaskBackend({}).emission == "writeback"
    assert TorchPipelinedMaskBackend(
        {"sga_backend_emission": "writeback"}).emission == "writeback"
    # mask_slices: the declared default expands to 2, and an explicit 0
    # (silently normalized to 2 before the flat config) fails loud
    assert TorchPipelinedMaskBackend({})._mask_slices == 2  # declared default
    try:
        TorchPipelinedMaskBackend({"sga_backend_mask_slices": 0})
        raise AssertionError("mask_slices 0 accepted")
    except ValueError as e:
        assert "mask_slices must be >= 1" in str(e), e
    assert NcclPipelinedIndexBackend({}).emission == "writeback"
    for cls in (TorchPipelinedMaskBackend, NcclPipelinedIndexBackend):
        try:
            cls({"sga_backend_emission": "compact"})
            raise AssertionError(f"compact accepted by {cls.__name__}")
        except ValueError as e:
            assert "only supported" in str(e)
            assert "torch_pipelined_index" in str(e), e
    try:
        TorchPipelinedIndexBackend({"sga_backend_emission": "quadtree"})
        raise AssertionError("bogus emission accepted")
    except ValueError as e:
        assert "quadtree" in str(e), e
    print("ok: backend emission resolution (auto->inherent, compact tpi-only)")


def test_validate_combination_emission_value_aware():
    """The compact-capability rule keys on the RESOLVED emission, not the
    decl field: torch_pipelined_index (decl emission=compact) with an
    explicit writeback may pair with a non-compact optimizer; explicit
    compact still requires consumes_compact."""
    backend = get_backend_plugin_decl("torch_pipelined_index")
    non_consumer = OptimizerDecl(
        name="fake", reset_mode="selective", consumes_compact=False,
        emittable_scores=frozenset({"update"}))
    # decl says compact; the run says writeback -> legal
    validate_combination("update", get_selection_decl("topk"), non_consumer,
                         backend, backend_emission="writeback")
    # auto/None -> the decl's inherent compact -> still requires the consumer
    for em in (None, "auto"):
        try:
            validate_combination("update", get_selection_decl("topk"),
                                 non_consumer, backend, backend_emission=em)
            raise AssertionError("non-consumer accepted under compact")
        except ValueError as e:
            assert "consumes_compact" in str(e), e
    print("ok: combination check resolves emission by value")


class _CompactRec(_Rec):
    """_Rec with the bucket fields _records_pass_compact consumes."""

    def __init__(self, param, grad, bucket_state, start):
        super().__init__(param, grad)
        self.bucket_state = bucket_state
        self.start = start


def test_compact_search_cpu_compact_dispatch():
    """Fallback contract pin: compact_search='table' must take the
    torch fallback SILENTLY on the CPU compact path — the eager
    bucket.dir_table build (a triton assert, or a CUDA kernel launched on
    CPU tensors) must never run before the fallback engages (the DECL's
    "silently ignored on ... the torch fallback path" contract). Drives
    SgaAdamW.sparse_step -> _records_pass_compact with compact_values on
    CPU tensors: the 'table' end state must be bitwise the 'bisection' end
    state, and the values must actually be consumed (the fallback is not
    a silent no-op)."""
    from sga.support import BucketSupport, Support

    torch.manual_seed(23)
    n = 512
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    idx = mask.nonzero(as_tuple=True)[0]
    values = torch.randn(idx.numel())
    param0, g0 = torch.randn(n), torch.randn(n)

    def run(compact_search):
        bs = SimpleNamespace(support_index=0)
        rec = _CompactRec(param0.clone(), g0.clone(), bs, 0)
        support = Support(buckets=[BucketSupport(mask_u8=mask)])
        ctx = OptContext(records=[rec], phase="sparse", support=support,
                         bucket_states=[], fold_window=8,
                         next_step_is_full=False, apply_coeff=0.5,
                         compact_values={id(bs): values.clone()})
        SgaAdamW({"sga_optimizer_compact_search": compact_search}).sparse_step(ctx)
        return rec

    rec_tab = run("table")
    rec_bin = run("bisection")
    # bitwise-identical end states: the gated dispatch hands BOTH configs
    # dir_table=None on CPU, so the fallback consumes the values the same way
    assert torch.equal(rec_tab.main_param.data, rec_bin.main_param.data)
    assert torch.equal(rec_tab.grad.data, rec_bin.grad.data)
    assert torch.equal(rec_tab._state["exp_avg"], rec_bin._state["exp_avg"])
    assert torch.equal(rec_tab._state["exp_avg_sq"],
                       rec_bin._state["exp_avg_sq"])
    # not a silent no-op: the values entered m (fresh state) and the
    # communicated coords were re-cleared by the fallback
    assert torch.count_nonzero(rec_tab._state["exp_avg"]).item() > 0
    assert torch.count_nonzero(rec_tab.grad.data[idx]).item() == 0
    print("ok: compact_search CPU compact dispatch (table == bisection end "
          "state on the torch fallback)")


def test_grad_correct_sparse_fusion_knob_guard():
    """The fusion knobs are sparse-step-only: the dispatch refuses them on
    full steps (emit_mode) and without a mask — owned message, before any
    buffer work."""
    n = 32
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=1e-2,
              step=5)
    p, g, m, v = (torch.randn(n) for _ in range(4))
    for kwargs in (dict(emit_mode=None, mask=None, coeff=0.5),
                   dict(emit_mode=None, mask=None, clear=True),
                   dict(emit_mode=None, mask=None, fold=True),
                   dict(emit_mode=0, mask=(torch.rand(n) < .5).to(torch.uint8),
                        clear=True)):
        try:
            adamw_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                               **kwargs, **kw)
        except ValueError as e:
            assert "requires a sparse step" in str(e), e
        else:
            raise AssertionError(f"guard did not fire for {kwargs}")


def test_full_step_dual_lr_semantics():
    """Boundary dual-lr pinned at SgaAdamW level: on a full step
    with full_sync_inactive_lr_scale != 1.0, the m/v commit and the
    materialized update score are BITWISE the single-lr run (mask-blind),
    the mask-active coords' apply is BITWISE unchanged, and the
    never-communicated tail steps at exactly lr * scale (the split sits
    outside the AdamW normalization, so the displacement is exactly
    proportional — s=0 freezes the tail while m/v still commit). A
    mask-less record (bootstrap / degenerate bucket = fully communicated)
    keeps plain lr everywhere."""
    torch.manual_seed(13)
    n = 256
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    active = mask > 0
    for scale in (0.0, 0.5, 2.0, 0.7):
        w = torch.randn(n)
        g0 = torch.randn(n)  # the boundary buffer: fresh g + residual
        rec_a = _SparseRec(w.clone().requires_grad_(False),
                           g0.clone().requires_grad_(False), mask)
        rec_b = _SparseRec(w.clone().requires_grad_(False),
                           g0.clone().requires_grad_(False), mask)
        rec_c = _SparseRec(w.clone().requires_grad_(False),
                           g0.clone().requires_grad_(False), None)
        opt_split = SgaAdamW({"sga_optimizer_full_sync_inactive_lr_scale": scale})
        opt_plain = SgaAdamW({})
        opt_split.full_step(_RecCtx([rec_a]))
        opt_plain.full_step(_RecCtx([rec_b]))
        opt_split.full_step(_RecCtx([rec_c]))

        # commit + score are mask-blind: bitwise the single-lr run
        assert torch.equal(rec_a._state["exp_avg"],
                           rec_b._state["exp_avg"]), scale
        assert torch.equal(rec_a._state["exp_avg_sq"],
                           rec_b._state["exp_avg_sq"]), scale
        assert torch.equal(rec_a.grad.data, rec_b.grad.data), scale
        # mask-less record: plain lr everywhere, bitwise the plain run
        assert torch.equal(rec_c.main_param.data, rec_b.main_param.data)
        assert torch.equal(rec_c._state["exp_avg"],
                           rec_b._state["exp_avg"])

        disp_a = w - rec_a.main_param.data
        disp_b = w - rec_b.main_param.data
        assert torch.equal(disp_a[active], disp_b[active]), scale
        # exact apply oracle (same rounding sequence as the implementation:
        # the materialized update score IS the update vector, and the apply
        # is one rounded multiply-add per coordinate at the split lr)
        update = rec_a.grad.data  # mask-blind: bitwise the plain run's
        assert torch.equal(update, rec_b.grad.data), scale
        lr_eff = torch.where(active, 1e-3, 1e-3 * scale)
        assert torch.equal(rec_a.main_param.data, w - lr_eff * update), scale
        if scale == 0.0:
            # frozen tail: the param is bitwise untouched at inactive coords
            assert torch.equal(rec_a.main_param.data[~active], w[~active])

    # named emit score under the split: the score path is mask-blind too —
    # the materialized abs_g score is bitwise the plain run's (the split
    # lives only in the apply's lr)
    w = torch.randn(n)
    g0 = torch.randn(n)
    rec_s = _SparseRec(w.clone().requires_grad_(False),
                       g0.clone().requires_grad_(False), mask)
    rec_p = _SparseRec(w.clone().requires_grad_(False),
                       g0.clone().requires_grad_(False), mask)
    SgaAdamW({"sga_optimizer_full_sync_inactive_lr_scale": 0.5},
             emit_score="abs_g").full_step(_RecCtx([rec_s]))
    SgaAdamW({}, emit_score="abs_g").full_step(_RecCtx([rec_p]))
    assert torch.equal(rec_s.grad.data, rec_p.grad.data)

    # default knob (1.0): full_step must not consult the mask at all —
    # _Rec.mask_u8() raises by contract, so this call passing IS the pin
    SgaAdamW({}).full_step(_RecCtx([_Rec(torch.randn(n), torch.randn(n))]))
    print("ok: full-step dual-lr semantics (commit/score mask-blind, "
          "tail exactly lr*scale, mask-less = plain lr, named score "
          "bitwise)")


def test_grad_correct_dual_lr_knob_guard():
    """inactive_lr_scale is full-step-with-mask only: refused on sparse
    steps (emit_mode=None), without a mask, and when negative/non-finite —
    owned messages, before any buffer work; the sparse-fusion guard
    precedes this one, and the dormant masked-commit hybrid (mask + full
    step at scale 1.0) is refused outright; the constructor repeats the
    domain guard for direct construction (tests/replay bypass config
    validation)."""
    n = 32
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=1e-2,
              step=5)
    p, g, m, v = (torch.randn(n) for _ in range(4))
    mask = (torch.rand(n) < .5).to(torch.uint8)
    for kwargs in (dict(emit_mode=None, mask=mask, inactive_lr_scale=0.5),
                   dict(emit_mode=0, mask=None, inactive_lr_scale=0.5),
                   dict(emit_mode=0, mask=mask, inactive_lr_scale=-1.0),
                   dict(emit_mode=0, mask=mask, inactive_lr_scale=float("inf")),
                   dict(emit_mode=0, mask=mask, inactive_lr_scale=float("nan"))):
        try:
            adamw_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                               **kwargs, **kw)
        except ValueError as e:
            assert "inactive_lr_scale" in str(e), e
        else:
            raise AssertionError(f"guard did not fire for {kwargs}")
    # guard order: the sparse-fusion guard precedes the dual-lr guard — a
    # combined coeff+clear+mask+scale violation reports the sparse-step rule
    try:
        adamw_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                           emit_mode=0, mask=mask, coeff=0.5, clear=True,
                           inactive_lr_scale=0.5, **kw)
    except ValueError as e:
        assert "requires a sparse step" in str(e), e
    else:
        raise AssertionError("fusion guard did not fire for the combined leg")
    # the dormant hybrid (predates the dual-lr knob, nothing in production
    # produces it): mask + full step without a split is refused outright
    try:
        adamw_grad_correct(p.clone(), g.clone(), m.clone(), v.clone(),
                           emit_mode=0, mask=mask, inactive_lr_scale=1.0,
                           **kw)
    except ValueError as e:
        assert "masked-commit full step" in str(e), e
    else:
        raise AssertionError("hybrid masked-commit full step was accepted")
    # legal combination (full step + mask) runs; the constructor's own
    # domain guard fires for direct construction
    adamw_grad_correct(p, g, m, v, mask=mask, emit_mode=0,
                       inactive_lr_scale=0.5, **kw)
    try:
        SgaAdamW({"sga_optimizer_full_sync_inactive_lr_scale": -0.5})
    except ValueError as e:
        assert "full_sync_inactive_lr_scale" in str(e), e
    else:
        raise AssertionError("constructor domain guard did not fire")
    print("ok: dual-lr knob guarded (sparse/mask-less/domain/fusion-order/"
          "hybrid)")


# ----------------------------------------------------- combination validation

def test_combination_validator_and_build_injection():
    from sga.plugins import (
        OptimizerDecl,
        build_backend_plugin,
        build_optimizer,
        build_selection,
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_selection_decl,
        registered_names,
        register_default_plugins,
        validate_combination,
    )

    register_default_plugins()
    opt_decl = get_optimizer_decl("adamw")
    for name in ("update", "coi", "abs_g", "abs_m"):
        assert name in opt_decl.emittable_scores
        validate_combination(name, get_selection_decl("topk"),
                             opt_decl, get_backend_plugin_decl("torch_pipelined_index"))
    # an unknown score name is rejected with the registered list
    try:
        validate_combination("nope", get_selection_decl("topk"),
                             opt_decl, get_backend_plugin_decl("torch_pipelined_index"))
        raise AssertionError("unknown score accepted")
    except ValueError as e:
        assert "nope" in str(e) and "abs_g" in str(e), e
    # a registry score outside the optimizer's emittable set is
    # offline-only: configs must reject it (simulated via a decl whose
    # emittable set lacks the name — exactly the future offline-candidate
    # promotion gate)
    non_emitting = OptimizerDecl(name="stub", emittable_scores=frozenset())
    try:
        validate_combination("abs_m", get_selection_decl("topk"),
                             non_emitting,
                             get_backend_plugin_decl("torch_pipelined_index"))
        raise AssertionError("offline-only score accepted")
    except ValueError as e:
        assert "emittable" in str(e), e
    # (the cycle-calendar legality rules moved to sga.config.validation
    # with the flat keys — pinned in test_config.py)

    # build injection: the optimizer instance carries the requested emission
    opt = build_optimizer("adamw", {}, emit_score="abs_m")
    assert opt.emit_score == "abs_m"
    default_opt = build_optimizer("adamw", {})
    assert default_opt.emit_score == "update"
    try:
        build_optimizer("adamw", {}, emit_score="nope")
        raise AssertionError("unemittable score accepted at build time")
    except ValueError:
        pass

    # every BUILT plugin object exposes its declaration (the engine reads
    # e.g. backend.decl.consumes at the first reselection — a class missing
    # its decl must fail here, not on the cluster)
    names = registered_names()
    assert names["score"] == score_names()
    assert names["optimizer"] == ["adamw"]
    assert names["selection"] == ["topk"]
    for n in names["backend"]:
        be = build_backend_plugin(n, {})
        assert be.decl.consumes in ("mask", "idx"), n
    for n in names["optimizer"]:
        assert hasattr(build_optimizer(n, {}), "decl"), n
    for n in names["selection"]:
        assert hasattr(build_selection(n, {"sga_selection_method": "histogram", "sga_selection_ratio_percent": 40.0}), "decl"), n
    print("ok: combination validator + build injection + decl coverage")


def test_emission_declaration_and_compact_capability():
    """Emission-legality validation: a typo'd emission is refused at
    REGISTRATION (the engine routes by exact equality, so 'comapct' would
    otherwise silently behave as writeback while the backend skips the
    writeback scatter), and a compact-emission backend combined with an
    optimizer that does not declare consumes_compact is refused at config
    time (the communicated values would never reach the m/v commit)."""

    seen = {}
    try:
        for bad in ("comapct", "write", "", None):
            try:
                _register("backend", seen, "stub",
                          BackendDecl(name="stub", emission=bad))
                raise AssertionError(f"emission {bad!r} accepted at registration")
            except ValueError as e:
                assert "emission" in str(e) and "compact" in str(e), e
        _register("backend", seen, "stub",
                  BackendDecl(name="stub", emission="compact"))  # legal passes
        assert isinstance(seen["stub"].emission, str)
    finally:
        pass

    register_default_plugins()
    compact = get_backend_plugin_decl("torch_pipelined_index")
    assert compact.emission == "compact"
    assert get_optimizer_decl("adamw").consumes_compact is True
    # the shipped pair is legal (also covered by the emittable-scores loop
    # above, restated here so the capability rule's PASS case is explicit)
    validate_combination("update", get_selection_decl("topk"),
                         get_optimizer_decl("adamw"), compact)
    # an optimizer WITHOUT the compact commit: refused on the compact
    # backend, fine on writeback backends
    stub = OptimizerDecl(name="stub", emittable_scores=frozenset({"update"}))
    try:
        validate_combination("update", get_selection_decl("topk"), stub,
                             compact)
        raise AssertionError("compact backend without consumes_compact accepted")
    except ValueError as e:
        assert "consumes_compact" in str(e), e
        assert "m/v commit" in str(e), e
    validate_combination("update", get_selection_decl("topk"), stub,
                         get_backend_plugin_decl("torch_pipelined_mask"))
    print("ok: emission registration gate + compact capability rule")


# ------------------------------------------------------------------- support

def test_support_idx_pool_chunked_equals_nonzero():
    from sga.support import BucketSupport, IndexPool, compact_mask_u8

    torch.manual_seed(4)
    m = (torch.rand(70_000_000) < 0.4).to(torch.uint8)  # > 32M chunk
    pool = IndexPool()
    a = compact_mask_u8(m, pool)
    ref = m.nonzero(as_tuple=True)[0].to(torch.int32)
    assert torch.equal(a, ref)
    assert a.data_ptr() == pool.buf.data_ptr()
    cap0 = pool.buf.numel()
    smaller = (torch.rand(70_000_000) < 0.1).to(torch.uint8)
    b = compact_mask_u8(smaller, pool)
    assert b.numel() < a.numel()
    assert pool.buf.numel() == cap0  # never shrinks
    bigger = (torch.rand(70_000_000) < 0.9).to(torch.uint8)
    c = compact_mask_u8(bigger, pool)
    assert c.numel() > cap0 and pool.buf.numel() >= c.numel()  # growth replaces the buffer
    bs = BucketSupport((torch.rand(64) < 0.5).to(torch.uint8), idx_pool=None)
    assert torch.equal(bs.idx, bs.mask_u8.nonzero(as_tuple=True)[0].to(torch.int32))
    print("ok: chunked compaction == nonzero; grow-only pool")


def test_support_split_and_idx_positional_equivalence():
    """BucketSupport.split: cached searchsorted midpoint split of the cached
    flat-nonzero indices (the legacy topk_indices/topk_split pair); the
    indices must address exactly the mask's set positions."""
    from sga.support import BucketSupport

    mask = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.uint8)  # mid = 3
    bs = BucketSupport(mask, idx_pool=None)
    idx = bs.idx
    assert idx.dtype == torch.int32
    assert idx.tolist() == [1, 2, 4]
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert torch.equal(grad[idx.long()], grad[mask.bool()])
    assert bs.split == 2  # indices 1,2 below mid=3; 4 above
    assert bs.split == int(torch.searchsorted(idx, 3).item())
    assert bs.split == int((idx < 3).sum().item())
    # both derivations are cached: repeated access returns identical objects
    assert bs.idx.data_ptr() == idx.data_ptr()
    assert bs.split == bs.split
    print("ok: BucketSupport idx/split == flatnonzero + midpoint split")


def test_support_block_idx_from_mask():
    """block_idx derives FROM THE MASK + spans: one int32 start per
    x-block with ANY set bit, per-span tiling (span tails are short
    blocks), ascending by construction; x=1 degenerates to the element
    idx; block_size > 1 without spans is a loud error."""
    from sga.support import BucketSupport, block_idx_from_mask

    # two spans: [0,10) tiles of 4 -> [0-3] [4-7] tail [8-9];
    #            [10,26) -> [10-13] [14-17] [18-21] tail [22-25]
    mask = torch.zeros(26, dtype=torch.uint8)
    mask[5] = 1        # partial set in block [4-7]: ANY rule lists it
    mask[8] = 1        # tail block of span 0
    mask[10:14] = 1    # full block of span 1
    spans = [(0, 10), (10, 26)]
    got = block_idx_from_mask(mask, spans, 4)
    assert got.dtype == torch.int32
    assert got.tolist() == [4, 8, 10]

    # x=1 is exactly the element idx
    torch.manual_seed(5)
    m = (torch.rand(1000) < 0.3).to(torch.uint8)
    assert torch.equal(block_idx_from_mask(m, [(0, 1000)], 1),
                       m.nonzero(as_tuple=True)[0].to(torch.int32))

    # empty support -> empty int32
    z = block_idx_from_mask(torch.zeros(20, dtype=torch.uint8),
                            [(0, 20)], 8)
    assert z.dtype == torch.int32 and z.numel() == 0

    # BucketSupport wiring: lazy derive + x=1 identity + block_split +
    # the spans-required refusal
    bs = BucketSupport(mask.clone(), block_size=4, block_spans=spans)
    assert torch.equal(bs.block_idx, got)
    assert bs.block_split == 3  # mid = 26//2 = 13; all of [4,8,10] below
    bs1 = BucketSupport(m.clone())
    assert torch.equal(bs1.block_idx, bs1.idx)
    try:
        BucketSupport(mask.clone(), block_size=4).block_idx
        raise AssertionError("block_idx without spans accepted")
    except ValueError:
        pass
    print("ok: block_idx from mask+spans (ANY rule, tails, x=1 identity)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
