#!/usr/bin/env python3
"""Single-GPU Triton-vs-torch A/B tests for the SGA kernels (R9 gap 3).

Runs only when CUDA + Triton are available (the container has GPUs); each
fused kernel path is compared against the pure-torch fallback that the CPU
unit tests already pin to fp64 oracles:

- masked AdamW (acc-aware) Triton kernel vs _masked_adamw_step_torch;
- radius grad-correct Triton kernel vs its torch reference;
- masked_scale / scatter_selected: bitwise-equality contracts;
- masked_norm_sq_contribution: same aggregate within atomic-order jitter.

Run: python tests/test_kernels_gpu.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.kernels import (
    masked_norm_sq_contribution,
    masked_scale,
    scatter_selected,
)
from sga.plugins.error_feedback.masked_adamw import (
    _masked_adamw_step_torch,
    masked_adamw_step,
)
from sga.plugins.error_feedback.radius_opt import (
    _radius_grad_correct_torch,
    radius_grad_correct,
)

torch.manual_seed(0)


def test_masked_adamw_acc_aware_triton_matches_fallback():
    device = "cuda"
    n = 4096
    param0 = torch.randn(n)
    grad = torch.randn(n)
    m0 = torch.randn(n) * 0.1
    v0 = torch.rand(n) * 0.1
    mask = (torch.rand(n) < 0.4).to(torch.uint8)
    count = torch.randint(1, 9, (n,)).to(torch.uint8)
    lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.95, 1e-8, 1e-2, 7

    # Triton device path (masked_adamw_step dispatches when buffers are CUDA)
    p_dev = param0.to(device).clone()
    m_dev, v_dev = m0.to(device).clone(), v0.to(device).clone()
    g_dev = grad.clone().to(device)
    c_dev = count.to(device).clone()
    mk_dev = mask.clone().to(device)
    masked_adamw_step(p_dev, g_dev, m_dev, v_dev, mk_dev, lr, b1, b2, eps,
                      wd, step, False, c_dev)

    # CPU torch fallback on identical inputs
    p_ref = param0.clone(); m_ref = m0.clone(); v_ref = v0.clone()
    _masked_adamw_step_torch(
        p_ref, grad, m_ref, v_ref, mask.bool(), lr, b1, b2, eps, wd,
        1 - b1 ** step, 1 - b2 ** step, False, count,
    )
    torch.testing.assert_close(p_dev.cpu(), p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(m_dev.cpu(), m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(v_dev.cpu(), v_ref, rtol=1e-4, atol=1e-6)


def test_radius_triton_matches_fallback():
    device = "cuda"
    n = 4096
    lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.95, 1e-8, 1e-2, 11
    param0 = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    mask = (torch.rand(n) < 0.4).to(torch.uint8)

    # Sparse path (write_update=False): fused apply; grad buffer untouched.
    p_d = param0.clone().to(device); mm_d = m0.clone().to(device)
    vv_d = v0.clone().to(device); g_d = torch.randn(n).to(device)
    mk_d = mask.clone().to(device)
    radius_grad_correct(p_d, g_d, mm_d, vv_d, lr, b1, b2, eps, wd, step,
                        mask=mk_d, write_update=False)
    p_ref, m_ref, v_ref = map(torch.clone, (param0, m0, v0))
    g_init = g_d.cpu().clone()
    _radius_grad_correct_torch(
        p_ref, g_init, m_ref, v_ref, lr, b1, b2, eps, wd,
        1 - b1 ** step, 1 - b2 ** step,
        1 - b1 ** (step - 1) if step > 1 else 0.0,
        1 - b2 ** (step - 1) if step > 1 else 0.0,
        mask, False,
    )
    torch.testing.assert_close(p_d.cpu(), p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(mm_d.cpu(), m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(vv_d.cpu(), v_ref, rtol=1e-4, atol=1e-6)
    assert torch.equal(g_d.cpu(), g_init), "sparse step must not touch grads"

    # Full path (write_update=True): materialize the update INTO the grad.
    p_d = param0.clone().to(device); mm_d = m0.clone().to(device)
    vv_d = v0.clone().to(device)
    grads = torch.randn(n)          # fresh residual-shaped input
    g_d = grads.clone().to(device)
    radius_grad_correct(p_d, g_d, mm_d, vv_d, lr, b1, b2, eps, wd, step,
                        mask=None, write_update=True)
    p_ref, m_ref, v_ref = map(torch.clone, (param0, m0, v0))
    g_ref = grads.clone()
    _radius_grad_correct_torch(
        p_ref, g_ref, m_ref, v_ref, lr, b1, b2, eps, wd,
        1 - b1 ** step, 1 - b2 ** step,
        1 - b1 ** (step - 1) if step > 1 else 0.0,
        1 - b2 ** (step - 1) if step > 1 else 0.0,
        None, True,
    )
    torch.testing.assert_close(g_d.cpu(), g_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p_d.cpu(), p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(mm_d.cpu(), m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(vv_d.cpu(), v_ref, rtol=1e-4, atol=1e-6)


def test_masked_scale_bitwise_and_norm_aggregate():
    device = "cuda"
    n = 8192
    grad0 = torch.randn(n)
    mask = (torch.rand(n) < 0.3).to(torch.uint8)
    coeff = 0.125  # power of two -> scaling is exact on both paths

    dev = grad0.clone().to(device)
    masked_scale(dev.float(), mask.to(device), coeff)
    ref = grad0.clone()
    ref[mask.bool()] *= coeff
    # contract of the fused scale kernel: bitwise identical to the fallback
    assert torch.equal(dev.cpu(), ref), "masked_scale lost bitwise equality"

    total_dev = torch.zeros(1, device=device)
    masked_norm_sq_contribution(dev, mask.to(device), total_dev)
    ref_sq = float((ref ** 2)[mask.bool()].sum())
    assert abs(total_dev.item() - ref_sq) <= 1e-4 * max(1.0, abs(ref_sq)), (
        total_dev.item(), ref_sq
    )


def test_scatter_selected_roundtrip():
    device = "cuda"
    numel = 1024
    k = 377
    idx = torch.randperm(numel)[:k].to(torch.int32).to(device)
    src = torch.randn(k, device=device)
    dst = torch.zeros(numel, device=device)
    scatter_selected(dst, idx, src)
    assert torch.equal(dst[idx.long()].cpu(), src.cpu())
    # untouched coordinates remain zero
    rest = torch.ones(numel, dtype=torch.bool)
    rest[idx.long()] = False
    assert torch.all(dst.cpu()[rest] == 0)


def test_radius_score_modes_purity_and_content():
    """Triton full-step score emission : with a named score the
    two outputs swap targets — the score lands in the grad buffer, the signed
    update in score_out — and moments/params stay bitwise identical to the
    default run; the emitted score matches its reference from the PRE-commit
    state (migrated from test_sga_scheme_xp_mask_score.py)."""
    device = "cuda"
    n = 999_999
    lr, b1, b2, eps, wd, step = 5e-5, 0.9, 0.95, 1e-8, 0.1, 1501
    torch.manual_seed(3)
    w0 = torch.randn(n) * 0.02
    g0 = torch.randn(n) * 1e-3
    m0 = torch.randn(n) * 1e-3
    v0 = torch.rand(n) * 1e-6

    def run(score_mode=None):
        w = w0.clone().to(device)
        g = g0.clone().to(device)
        m, v = m0.clone().to(device), v0.clone().to(device)
        side = None if score_mode is None else torch.empty(n, device=device)
        radius_grad_correct(w, g, m, v, lr, b1, b2, eps, wd, step,
                            mask=None, write_update=True,
                            score_out=side, score_mode=score_mode or "coi")
        return w, g, m, v, side

    w1, g1, m1, v1, _ = run(None)               # default: update in grads
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    bc1_pre, bc2_pre = 1 - b1 ** (step - 1), 1 - b2 ** (step - 1)
    m1c = b1 * m0 + (1 - b1) * g0
    v1c = b2 * v0 + (1 - b2) * g0 * g0
    u_g = (m1c / bc1) / ((v1c / bc2).sqrt() + eps)

    for mode, ref in (
        ("coi", ((m1c / bc1) / ((v1c / bc2).sqrt() + eps)
                 - (b1 * m0 / bc1) / ((b2 * v0 / bc2).sqrt() + eps)).abs()),
        ("abs_g_synced", g0.abs()),
        ("abs_m", m0.abs()),
        # snr_m: state-matched (pre-commit) corrections, exponent step - 1
        ("snr_m", (m0 / bc1_pre).abs() / ((v0 / bc2_pre).sqrt() + eps)),
        ("abs_m_post", m1c.abs()),
    ):
        w2, g2, m2, v2, side = run(mode)
        assert torch.equal(m1.cpu(), m2.cpu())
        assert torch.equal(v1.cpu(), v2.cpu())
        assert torch.equal(w1.cpu(), w2.cpu())   # write_update never touches w
        torch.testing.assert_close(side.cpu(), g1.cpu(), rtol=1e-5, atol=1e-6)
        rel = ((g2.cpu().double() - ref.double()).abs().max()
               / ref.double().max().clamp(min=1e-30)).item()
        assert rel < 1e-5, (mode, rel)

    # snr_m at step == 1: no commit applied yet -> score collapses to 0
    w = w0.clone().to(device)
    g = g0.clone().to(device)
    m, v = m0.clone().to(device), v0.clone().to(device)
    radius_grad_correct(w, g, m, v, lr, b1, b2, eps, wd, 1,
                        mask=None, write_update=True,
                        score_out=torch.empty(n, device=device),
                        score_mode="snr_m")
    assert torch.all(g == 0).item(), "snr_m at step 1 must be exactly 0"

    # unknown score mode is rejected host-side before any launch
    try:
        radius_grad_correct(w0.clone(), g0.clone(), m0.clone(), v0.clone(),
                            lr, b1, b2, eps, wd, step, mask=None,
                            write_update=True,
                            score_out=torch.empty(n), score_mode="bogus")
        raise AssertionError("bogus score_mode accepted")
    except ValueError as e:
        assert "score_mode" in str(e), e


if __name__ == "__main__":
    try:
        import triton  # noqa: F401

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not has_gpu:
        print(f"SKIP {len(fns)} tests (CUDA+Triton unavailable)")
        sys.exit(0)
    dev_kind = torch.cuda.get_device_name(0)
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__} [{dev_kind}]")
    print(f"{len(fns)} tests OK")
