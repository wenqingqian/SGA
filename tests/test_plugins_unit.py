#!/usr/bin/env python3
"""Unit tests for the SGA plugin layer (scores / selections / EFs), CPU-only.

Covers (acceptance + expert Q7 additions):
- abs_g compute_mask (element + block) vs a plain-torch reference;
- masked AdamW torch fallback vs an fp64 oracle (standard + acc-aware +
  freeze_inactive), and full-mask equivalence to dense AdamW;
- acc-count update semantics incl. the uint8 saturation at the ceiling;
- circular topk: histogram threshold == exact torch.kthvalue threshold,
  mask equality, k rounding bound;
- threshold selection: OR-merge with a mocked collective, always-sync
  forcing, adaptive controller band logic and caps;
- radius reset fold semantics (undivided / /T / full clear).

Run: python tests/test_plugins_unit.py  (or pytest tests/test_plugins_unit.py)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.plugins.error_feedback.masked_adamw import (
    MaskedAdamWEF,
    _masked_adamw_step_torch,
    update_grad_acc_count,
)
from sga.plugins.error_feedback.radius_opt import radius_reset
from sga.plugins.scores.abs_g import compute_mask
from sga.plugins.selection.circular_topk import (
    CircularTopkSelection,
    _HIST_BINS,
    _abs_bit_top16,
    _histogram_select_threshold,
    _select_masks_exact_kthvalue,
    _select_masks_histogram,
)
from sga.plugins.selection.threshold import ThresholdSelection

torch.manual_seed(0)


class _Ctx:
    def __init__(self, grad_views, always=None, comm_group=None):
        self.grad_views = grad_views
        self.always_sync_masks = always or [None] * len(grad_views)
        self.comm_group = comm_group


# ---------------------------------------------------------------- abs_g masks

def test_compute_mask_elementwise():
    g = torch.tensor([-0.5, 0.05, 2.0, -0.00001, 0.0])
    m = compute_mask(g, 0.1)
    ref = (g.abs() >= 0.1).to(torch.uint8)
    assert torch.equal(m, ref), (m, ref)


def test_compute_mask_blockwise():
    g = torch.arange(16, dtype=torch.float32).reshape(4, 4) * 0.01 - 0.02
    m = compute_mask(g.reshape(-1), 0.03, block_size=4)
    ref = (g.abs().amax(dim=1) >= 0.03).to(torch.uint8)
    assert torch.equal(m, ref), (m, ref)


def test_compute_mask_empty():
    m = compute_mask(torch.empty(0), 0.1)
    assert m.numel() == 0


# ------------------------------------------------------------- masked adamw

def _oracle_adamw(param, grad, m, v, step, lr, b1, b2, eps, wd, active, count, freeze):
    p, g, mm, vv = (t.double().clone() for t in (param, grad, m, v))
    cnt = count.double().clamp(min=1)
    gg = torch.where(active, g / cnt, torch.zeros_like(g))
    b1p = torch.exp(cnt * torch.log(torch.tensor(b1, dtype=torch.float64)))
    b2p = torch.exp(cnt * torch.log(torch.tensor(b2, dtype=torch.float64)))
    m_new = b1 * mm + (1 - b1p) * gg
    v_new = b2 * vv + (1 - b2p) * gg * gg
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    upd = torch.where(active, lr * cnt, lr * (1 + 0 * cnt)) * (m_new / bc1) / (
        (v_new / bc2).sqrt() + eps
    )
    p = p - lr * wd * p
    if freeze:
        m_new = torch.where(active, m_new, mm)
        v_new = torch.where(active, v_new, vv)
        upd = torch.where(active, upd, torch.zeros_like(upd))
    p = p - upd
    return p, m_new, v_new


def test_masked_adamw_matches_oracle():
    n = 512
    param0 = torch.randn(n)
    grad = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    step, lr, b1, b2, eps, wd = 7, 1e-3, 0.9, 0.95, 1e-8, 1e-2

    for active_ratio, count_on, freeze in [
        (0.3, False, False), (0.3, True, False), (0.3, True, True), (1.0, True, False),
    ]:
        active = torch.rand(n) < active_ratio
        count = torch.randint(1, 9, (n,)).to(torch.uint8) if count_on else None
        p = param0.clone(); m = m0.clone(); v = v0.clone()
        _masked_adamw_step_torch(
            p, grad, m, v, active, lr, b1, b2, eps, wd,
            1 - b1 ** step, 1 - b2 ** step, freeze, count,
        )
        po, mo, vo = _oracle_adamw(
            param0, grad, m0, v0, step, lr, b1, b2, eps, wd,
            active, count if count is not None else torch.ones(n), freeze,
        )
        torch.testing.assert_close(p, po.float(), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(m, mo.float(), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(v, vo.float(), rtol=1e-4, atol=1e-7)


def test_full_mask_equals_dense_adamw():
    """All-ones mask + unit acc counts must equal standard AdamW."""
    n = 256
    param0 = torch.randn(n)
    grad = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    step, lr, b1, b2, eps, wd = 5, 1e-3, 0.9, 0.95, 1e-8, 1e-2
    ones = torch.ones(n, dtype=torch.bool)

    p = param0.clone(); m = m0.clone(); v = v0.clone()
    _masked_adamw_step_torch(
        p, grad, m, v, ones, lr, b1, b2, eps, wd,
        1 - b1 ** step, 1 - b2 ** step, False, torch.ones(n, dtype=torch.uint8),
    )
    # dense reference (decoupled AdamW)
    pd = param0.clone(); md = m0.clone(); vd = v0.clone()
    md.mul_(b1).add_(grad, alpha=1 - b1)
    vd.mul_(b2).addcmul_(grad, grad, value=1 - b2)
    denom = (vd / (1 - b2 ** step)).sqrt().add_(eps)
    pd.add_(pd, alpha=-lr * wd)
    pd.addcdiv_(md, denom, value=-lr / (1 - b1 ** step))
    torch.testing.assert_close(p, pd, rtol=1e-5, atol=1e-7)


# ------------------------------------------------------------------ acc count

def test_acc_count_update_saturates():
    count = torch.tensor([1, 5, 255, 10], dtype=torch.uint8)
    active = torch.tensor([1, 0, 0, 1], dtype=torch.uint8)
    update_grad_acc_count(count, active)
    assert count.tolist() == [1, 6, 255, 1]  # ceiling SATURATES instead of wrapping


def test_acc_count_saturation_no_wrap_over_long_streak():
    """A coordinate unselected for hundreds of steps must saturate at 255,
    never wrap back to small counts."""
    count = torch.full((4,), 250, dtype=torch.uint8)
    inactive = torch.zeros(4, dtype=torch.uint8)
    for _ in range(300):
        update_grad_acc_count(count, inactive)
    assert count.tolist() == [255] * 4


def test_saturated_residual_is_averaged_not_reweighted():
    """Compose saturation WITH a stale residual: a coordinate holding a
    255-step accumulated gradient enters its next update divided by 255
    (bounded moment), NOT divided by ~1 as a wrapped/clamped count would."""
    n = 64
    stale_grad = torch.full((n,), 255.0)          # the long-accumulated residual
    param = torch.zeros(n)
    m0 = torch.zeros(n)
    v0 = torch.zeros(n)
    saturated = torch.full((n,), 255, dtype=torch.uint8)

    step, lr, b1, b2, eps, wd = 256, 1e-3, 0.9, 0.95, 1e-8, 0.0
    p = param.clone(); m = m0.clone(); v = v0.clone()
    _masked_adamw_step_torch(
        p, stale_grad, m, v, torch.ones(n, dtype=torch.bool), lr, b1, b2,
        eps, wd, 1 - b1 ** step, 1 - b2 ** step, False, saturated,
    )
    # explicit reference: g/count with 1-beta**count weights, count == 255
    cnt = torch.full((n,), 255, dtype=torch.float64)
    ref_m = b1 * 0.0 + (1 - torch.exp(cnt * float(torch.log(torch.tensor(b1))))) * (
        stale_grad.double() / cnt
    )
    torch.testing.assert_close(m, ref_m.float(), rtol=1e-5, atol=1e-7)
    # bound: no coordinate's moment may approach the raw residual scale
    # (the wrap bug made count read 0 -> clamped 1 -> m jump to ~(g))
    assert m.abs().max().item() < 2.0


# --------------------------------------------------------------- circular topk

def test_histogram_matches_exact_kthvalue():
    for trial in range(5):
        grads = [torch.randn(1000), torch.randn(317), torch.randn(1)]
        k = 100 + trial * 37
        t_h, m_h = _select_masks_histogram(grads, k, None)
        t_e, m_e = _select_masks_exact_kthvalue(grads, k, None)
        assert t_h == t_e, (t_h, t_e)
        for a, b in zip(m_h, m_e):
            assert torch.equal(a, b)
        n_sel = sum(int(torch.count_nonzero(m).item()) for m in m_h)
        assert n_sel >= k  # ties at the boundary are all selected


def test_circular_topk_k_rounding():
    sel = CircularTopkSelection({"ratio_percent": 10.0})
    n = 1000
    assert sel.k_for(n) == 100
    # rounding bound from k = max(1, round(n * r)): |selected/n - r| <= 0.5/n
    assert abs(sel.k_for(999) / 999 - 0.10) <= 0.5 / 999 + 1e-12


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


# ------------------------------------------------------ threshold + adaptivity

def test_threshold_select_or_merge_and_always_sync(monkeypatch=None):
    class _Score:
        has_threshold_path = True

        def threshold_mask(self, views, threshold, block_size):
            return [compute_mask(g, threshold) for g in views]

    views = [torch.tensor([0.5, 0.01, 0.2]), torch.tensor([0.3, 0.001])]
    always = [torch.tensor([0, 1, 0], dtype=torch.uint8), None]

    sel = ThresholdSelection({"threshold": 0.1}, score=_Score())
    masks = sel.select(_Ctx(views, always, comm_group=None))
    # 0.5>=0.1 -> 1; 0.01 -> always-sync forced to 1; 0.2 -> 1; 0.3 -> 1; 0.001 -> 0
    assert masks[0].tolist() == [1, 1, 1]
    assert masks[1].tolist() == [1, 0]


def test_or_merge_survives_world_256():
    """The mask OR-merge must be a MAX, not a SUM.

    At DP world 256 a uint8 SUM of all-ones wraps to 0: universally-selected
    coordinates silently drop, identically on every rank — invisible to
    every consistency layer. MAX of 0/1 is exactly OR and cannot wrap at any
    world size; staying on uint8 also keeps the collective inside NCCL's
    supported dtype/op set (int16 is rejected by NCCL outright).
    """
    import contextlib
    from types import SimpleNamespace

    import torch.distributed as dist

    class _Score:
        has_threshold_path = True

        def threshold_mask(self, views, threshold, block_size):
            return [compute_mask(g, threshold) for g in views]

    views = [torch.tensor([0.5, 0.01, 0.2, 4.0, 0.001])]
    sel = ThresholdSelection({"threshold": 0.1}, score=_Score())

    saved_ar, saved_init = dist.all_reduce, dist.is_initialized

    @contextlib.contextmanager
    def _world256():
        def fake_allreduce(t, op=None, group=None, async_op=False):
            if op is dist.ReduceOp.SUM:
                t.mul_(256)  # equal replicas: SUM == 256 * local (uint8 wraps)
            # MAX over equal replicas: t already holds the result
            return SimpleNamespace(wait=lambda: None)

        dist.all_reduce = fake_allreduce
        try:
            yield
        finally:
            dist.all_reduce = saved_ar

    try:
        dist.is_initialized = lambda: True
        with _world256():
            masks = sel.select(_Ctx(views, [None], comm_group=object()))
        # a SUM implementation would wrap the ones to 256 % 256 == 0
        assert masks[0].tolist() == [1, 0, 1, 1, 0]
    finally:
        dist.is_initialized = saved_init


def test_threshold_adapt_band_and_cap():
    sel = ThresholdSelection({
        "threshold": 1e-4, "target_ratio_low": 0.2, "target_ratio_high": 0.3,
        "threshold_max": 1e-3, "adapt_factor": 0.1,
    })
    # ratio 0.9 >> target: threshold increases, capped by threshold_max
    n = 1000
    masks = [torch.ones(n, dtype=torch.uint8)]
    always = [None]
    for _ in range(200):
        sel.adapt(masks, always)
    assert sel.threshold == 1e-3
    # ratio 0.01 << target: threshold decreases, floored at 0.9x per step
    sel2 = ThresholdSelection({"threshold": 1e-4, "target_ratio_low": 0.2,
                               "target_ratio_high": 0.3})
    masks = [(torch.rand(n) < 0.01).to(torch.uint8)]
    t0 = sel2.threshold
    sel2.adapt(masks, [None])
    assert sel2.threshold < t0 and sel2.threshold >= 0.9 * t0


def test_threshold_warmup_anneal_bounds():
    sel = ThresholdSelection({
        "threshold": 1e-4, "target_ratio_low": 0.2, "target_ratio_high": 0.3,
        "warmup_iters": 10, "warmup_start_ratio": 1.0,
    })
    # Deterministic ratio 0.99. At adapt_iter 1 the annealed band is
    # [0.92, 0.93] (progress 0.1 from start 1.0) -> mid 0.925 -> factor
    # 1 + (0.99 - 0.925) * 0.1 == 1.0065. Asserting the EXACT factor makes
    # this fail on any warmup-term regression (a non-warmup controller
    # yields 1.074 against the final band).
    masks = [torch.tensor([1] * 990 + [0] * 10, dtype=torch.uint8)]
    t0 = sel.threshold
    sel.adapt(masks, [None])
    torch.testing.assert_close(sel.threshold, t0 * 1.0065, rtol=1e-9, atol=0.0)
    # control: without warmup the SAME ratio moves the threshold by a clearly
    # different factor — proves the annealed band was actually consulted.
    plain = ThresholdSelection({
        "threshold": 1e-4, "target_ratio_low": 0.2, "target_ratio_high": 0.3,
    })
    plain.adapt([masks[0].clone()], [None])
    torch.testing.assert_close(plain.threshold, t0 * 1.074, rtol=1e-9, atol=0.0)


def test_threshold_adapt_single_transfer_and_forced_cache():
    """The controller aggregates selected counts device-side into ONE host
    transfer and caches static always-sync counts across reselections."""
    sel = ThresholdSelection({
        "threshold": 1e-4, "target_ratio_low": 0.2, "target_ratio_high": 0.3,
    })
    n = 512
    forced = torch.zeros(n, dtype=torch.uint8)
    forced[:7] = 1
    masks = [(torch.rand(n) < 0.5).to(torch.uint8)]
    transfers = {"n": 0}
    real_item = torch.Tensor.item

    def counting_item(self, *a):
        transfers["n"] += 1
        return real_item(self, *a)

    torch.Tensor.item = counting_item
    try:
        sel.adapt([masks[0].clone()], [forced])
        first = transfers["n"]
        assert first >= 1
        sel.adapt([masks[0].clone()], [forced])
        second = transfers["n"] - first
        # cached forced count + single accumulator: steady state keeps the
        # per-call transfer count at exactly 1 regardless of param count.
        assert second == 1, second
    finally:
        torch.Tensor.item = real_item


# ------------------------------------------------------------- radius reset

def test_radius_reset_fold_semantics():
    for fold, interval in [(False, 8), (True, 8)]:
        grad = torch.randn(64)
        mask = (torch.rand(64) < 0.25).to(torch.uint8)
        ref = grad.clone()
        radius_reset(grad, mask, float(interval), fold)
        active = mask > 0
        assert torch.all(grad[active] == 0)
        if fold:
            torch.testing.assert_close(
                grad[~active], ref[~active] / interval, rtol=0, atol=0,
            )
        else:
            assert torch.equal(grad[~active], ref[~active])


# --------------------------------------------------------------- EF reset (origin)

def test_origin_ef_selective_reset_and_counts():
    ef = MaskedAdamWEF({"grad_acc_count": True, "freeze_inactive": False})

    class _BS:
        def __init__(self, n):
            self.grad_data = torch.randn(n)
            self.active_mask_u8 = (torch.rand(n) < 0.3).to(torch.uint8)
            self.grad_acc_count = torch.full((n,), 3, dtype=torch.uint8)
            self.ones_mask_u8 = None
            self.extra_main_grads = [torch.randn(4)]

    bs = _BS(128)
    residual_ref = bs.grad_data.clone()

    class _C:
        support = None
        bucket_states = [bs]
        records = []

    ef.reset(_C, was_full=False)
    active = bs.active_mask_u8 > 0
    assert torch.all(bs.grad_data[active] == 0)
    assert torch.equal(bs.grad_data[~active], residual_ref[~active])
    # communicated -> 1, uncommunicated -> +1 (from 3)
    assert bs.grad_acc_count[active].min().item() == 1
    assert bs.grad_acc_count[~active].min().item() == 4
    assert torch.all(bs.extra_main_grads[0] == 0)

    ef.reset(_C, was_full=True)
    assert torch.all(bs.grad_data == 0)
    assert torch.all(bs.grad_acc_count == 1)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
