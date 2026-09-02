#!/usr/bin/env python3
"""Single-GPU Triton-vs-torch A/B tests for the SGA kernels (R9 gap 3).

Runs only when CUDA + Triton are available (the container has GPUs); each
fused kernel path is compared against the pure-torch fallback that the CPU
unit tests already pin to fp64 oracles:

- adamw grad-correct Triton kernel vs its torch reference (sparse apply and
  full-step score materialization, every emittable score);
- scatter_selected: bitwise-equality contract;
- masked_norm_sq_contribution: same aggregate within atomic-order jitter;
- the fused sparse grad-correct path (coeff/clear/fold knobs) vs
  the standalone sequence it replaces: bitwise per buffer.

Run: python tests/test_kernels_gpu.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.kernels import (
    masked_norm_sq_contribution,
    scatter_selected,
)
from sga.plugins.optimizer.adamw import adamw_grad_correct
from sga.plugins.optimizer.adamw_torch import grad_correct_torch
from sga.plugins.scores import Score, emittable_score_names, get_score_spec

torch.manual_seed(0)


def test_grad_correct_triton_matches_fallback():
    device = "cuda"
    n = 4096
    lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.95, 1e-8, 1e-2, 11
    param0 = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    mask = (torch.rand(n) < 0.4).to(torch.uint8)

    # Sparse path (emit_mode=None): fused apply; grad buffer untouched.
    p_d = param0.clone().to(device); mm_d = m0.clone().to(device)
    vv_d = v0.clone().to(device); g_d = torch.randn(n).to(device)
    mk_d = mask.clone().to(device)
    adamw_grad_correct(p_d, g_d, mm_d, vv_d, lr, b1, b2, eps, wd, step,
                       mask=mk_d, emit_mode=None)
    p_ref, m_ref, v_ref = map(torch.clone, (param0, m0, v0))
    g_init = g_d.cpu().clone()
    grad_correct_torch(
        p_ref, g_init, m_ref, v_ref, lr, b1, b2, eps, wd,
        1 - b1 ** step, 1 - b2 ** step,
        mask, None,
    )
    torch.testing.assert_close(p_d.cpu(), p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(mm_d.cpu(), m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(vv_d.cpu(), v_ref, rtol=1e-4, atol=1e-6)
    assert torch.equal(g_d.cpu(), g_init), "sparse step must not touch grads"

    # Full path (emit_mode=update): materialize the update INTO the grad,
    # with the apply fused in the same pass on both paths.
    p_d = param0.clone().to(device); mm_d = m0.clone().to(device)
    vv_d = v0.clone().to(device)
    grads = torch.randn(n)          # fresh residual-shaped input
    g_d = grads.clone().to(device)
    adamw_grad_correct(p_d, g_d, mm_d, vv_d, lr, b1, b2, eps, wd, step,
                       mask=None, emit_mode=int(Score.update))
    p_ref, m_ref, v_ref = map(torch.clone, (param0, m0, v0))
    g_ref = grads.clone()
    grad_correct_torch(
        p_ref, g_ref, m_ref, v_ref, lr, b1, b2, eps, wd,
        1 - b1 ** step, 1 - b2 ** step,
        None, int(Score.update),
    )
    torch.testing.assert_close(g_d.cpu(), g_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p_d.cpu(), p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(mm_d.cpu(), m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(vv_d.cpu(), v_ref, rtol=1e-4, atol=1e-6)


def test_grad_correct_sparse_fusion_bitwise():
    """coeff+clear+fold fused into grad-correct == the standalone
    three-pass sequence, bitwise, on every buffer.

    Reference mirrors: prescale via a torch elementwise multiply (a lone fp32
    multiply is exactly rounded on every backend), the plain kernel with no
    fusion knobs, then masked zero-fill and multiply-by-host-reciprocal fold
    (interval 7.0 exercises the non-power-of-two reciprocal; within ONE
    backend mul-by-reciprocal is self-consistent — the CPU fallback's true
    division may sit 1 ulp from it). Any coeff value is fair game — there is
    no add adjacent for FMA contraction, which the C1 on-GPU bench (99
    records, real coeffs) already confirmed at full scale.
    """
    device = "cuda"
    n = 8192
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1, step=1000)
    for coeff in (0.125, 0.7, 1.0):
        for fold, interval in [(False, 8.0), (True, 8.0), (True, 7.0)]:
            g0 = torch.randn(n)
            mask = (torch.rand(n) < 0.3).to(torch.uint8)
            p, m, v = torch.randn(n), torch.randn(n) * .1, torch.rand(n) * .01

            fp, fg, fm, fv = (t.clone().to(device) for t in (p, g0, m, v))
            adamw_grad_correct(fp, fg, fm, fv, mask=mask.to(device),
                               emit_mode=None, coeff=coeff, clear=True,
                               fold=fold, interval=interval, **kw)

            rp, rg, rm, rv = (t.clone().to(device) for t in (p, g0, m, v))
            dmask = mask.to(device)
            if coeff != 1.0:
                rg = torch.where(dmask > 0, rg * coeff, rg)
            adamw_grad_correct(rp, rg, rm, rv, mask=dmask, emit_mode=None, **kw)
            rg.masked_fill_(dmask > 0, 0.0)
            if fold:
                inv = float((torch.ones(1, dtype=torch.float32) / interval).item())
                rg.mul_(inv)
            for got, ref, name in ((fp, rp, "param"), (fm, rm, "exp_avg"),
                                   (fv, rv, "exp_avg_sq"), (fg, rg, "grad")):
                assert torch.equal(got, ref), (
                    f"fused sparse path diverged ({name}, coeff={coeff}, "
                    f"fold={fold})"
                )


def test_masked_norm_sq_aggregate():
    device = "cuda"
    n = 8192
    grad0 = torch.randn(n)
    mask = (torch.rand(n) < 0.3).to(torch.uint8)

    dev = grad0.clone().to(device)
    total_dev = torch.zeros(1, device=device)
    masked_norm_sq_contribution(dev.float(), mask.to(device), total_dev)
    ref_sq = float((grad0 ** 2)[mask.bool()].sum())
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


def test_score_modes_purity_and_content():
    """Triton full-step score emission: every named score lands
    in the grad buffer with the apply fused in the same pass, and the
    emitted score matches its reference from the PRE-commit state; the
    bare-int kernel modes equal the Score enum members BY NAME (the enum is
    the namespace; migrated from test_sga_scheme_xp_mask_score.py)."""
    device = "cuda"
    n = 999_999
    lr, b1, b2, eps, wd, step = 5e-5, 0.9, 0.95, 1e-8, 0.1, 1501
    torch.manual_seed(3)
    w0 = torch.randn(n) * 0.02
    g0 = torch.randn(n) * 1e-3
    m0 = torch.randn(n) * 1e-3
    v0 = torch.rand(n) * 1e-6

    def run(score):
        w = w0.clone().to(device)
        g = g0.clone().to(device)
        m, v = m0.clone().to(device), v0.clone().to(device)
        adamw_grad_correct(w, g, m, v, lr, b1, b2, eps, wd, step,
                           mask=None, emit_mode=int(score))
        return w, g, m, v

    w1, g1, m1, v1 = run(Score.update)        # default: update in grads
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step

    # Content + purity for EVERY emittable score, iterated from the registry
    # (derived: a score registered without a kernel branch fails HERE at the
    # GPU gate instead of mid-cluster). The reference is the registry
    # formula — the same single-source math the torch fallback calls,
    # fp64-oracle-pinned on CPU by tests/test_xp_capability.py — computed
    # from the same PRE-commit state.
    for name in sorted(emittable_score_names()):
        w2, g2, m2, v2 = run(Score[name])
        ref = get_score_spec(name).formula(
            w0, m0, v0, g0, b1, b2, eps, wd, bc1, bc2,
            update_includes_wd=True,
        )
        # purity: the m/v commit is identical (bitwise) across every mode
        assert torch.equal(m1.cpu(), m2.cpu())
        assert torch.equal(v1.cpu(), v2.cpu())
        # content: the emitted score matches its registry formula
        rel = ((g2.cpu().double() - ref.double()).abs().max()
               / ref.double().max().clamp(min=1e-30)).item()
        assert rel < 1e-5, (name, rel)
        # the apply is fused in the same pass: every mode's PARAMS must
        # equal the default run's (same update, same fused arithmetic)
        torch.testing.assert_close(w2.cpu(), w1.cpu(), rtol=1e-5, atol=1e-7)

    # the bare kernel-mode ints equal the enum values (Score is the single
    # namespace: the kernel's EMIT_MODE comments must not drift)
    for name, mode in (("update", 0), ("abs_g", 1), ("abs_m", 2), ("coi", 3)):
        assert int(Score[name]) == mode, name

    # offline sentinel mode is rejected host-side before any launch
    try:
        adamw_grad_correct(w0.clone(), g0.clone(), m0.clone(), v0.clone(),
                           lr, b1, b2, eps, wd, step, mask=None,
                           emit_mode=int(Score.OFFLINE))
        raise AssertionError("offline emit_mode accepted")
    except ValueError as e:
        assert "OFFLINE" in str(e), e

    # any out-of-range int gets the same host-side rejection with owned text
    # (not the raw enum-cast error) — the public unknown-mode contract
    try:
        adamw_grad_correct(w0.clone(), g0.clone(), m0.clone(), v0.clone(),
                           lr, b1, b2, eps, wd, step, mask=None, emit_mode=7)
        raise AssertionError("out-of-range emit_mode accepted")
    except ValueError as e:
        assert "not a valid Score" in str(e), e


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
