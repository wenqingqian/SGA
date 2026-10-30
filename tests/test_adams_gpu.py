#!/usr/bin/env python3
"""Single-GPU Triton-vs-torch A/B tests for the AdamS kernels
(sga/plugins/optimizer/adams.py).

Runs only when CUDA + Triton are available; each fused kernel path is
compared against the pure-torch reference that tests/test_adams_unit.py
pins to fp64 oracles on CPU. The AdamW suite's criterion classes carry
over unchanged (values-level and grad end-state stores are bitwise; the
m/v/param arithmetic tail is ulp-class — per-compilation FMA contraction
forbids bitwise across kernel bodies — and the same binary reruns
bitwise-stable):

- AdamS grad-correct Triton kernel vs its torch reference (sparse apply
  with the fused coeff/clear/fold knobs, and the full-step score
  materialization for every emittable score; the registry formula with
  optimizer="adams" is the content oracle);
- the ordering leg: the committed v is the PRE-commit-m recursion, on
  GPU, with the opposite-sign / crossed-scale fixture whose three
  candidate v bases are pairwise distinct;
- the boundary dual-lr (LR_SPLIT) vs the torch reference and the
  single-lr run;
- the compact-emission merge-join (bisection and table forms) vs the
  writeback chain: values/grad bitwise, tail ulp-class, fold arms, the
  k == 0 boundary.

Run: python tests/test_adams_gpu.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.kernels import pack_selected, scatter_selected
from sga.plugins.optimizer.adams import (
    SgaAdamS,
    _TRITON_SCORE_MAP,
    adams_grad_correct,
    adams_grad_correct_compact,
)
from sga.plugins.optimizer.adams import build_dir_table
from sga.plugins.optimizer.adams_torch import adams_grad_correct_torch
from sga.plugins.scores import get_score_spec, score_names

torch.manual_seed(0)

# Measured contraction class of the compact merge-join tail vs the masked
# kernel tail (the AdamW suite's yardstick, carried over: the search
# machinery is identical and the tail is one op shorter — no correction
# divisions — so the class cannot widen). The bound is pinned just above
# the measurement so re-seeds do not flap.
_COMPACT_TAIL_ULP = 4


def _ulp_distance(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Ordered-int ULP distance between two finite fp32 tensors (the
    AdamW suite's helper, verbatim)."""
    ia = got.contiguous().view(torch.int32).to(torch.int64)
    ib = ref.contiguous().view(torch.int32).to(torch.int64)
    ka = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    kb = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return (ka - kb).abs()


def _assert_tail_ulp_class(got, ref, op_scale, ctx):
    """Pin a kernel tail to ``_COMPACT_TAIL_ULP`` ulps at the operand
    scale (the AdamW suite's helper, verbatim)."""
    op_scale = op_scale.to(got.device)
    fin = torch.isfinite(got) & torch.isfinite(ref)
    assert torch.equal(got.view(torch.int32)[~fin],
                       ref.view(torch.int32)[~fin]), (
        f"non-finite tail values diverged ({ctx})"
    )
    if not fin.any():
        return
    x, y = got[fin], ref[fin]
    d = _ulp_distance(x, y).double()
    tiny = torch.finfo(torch.float32).tiny
    mag = torch.maximum(x.abs(), y.abs()).clamp_min(tiny)
    bin_gap = (op_scale[fin].clamp_min(tiny).log2().floor()
               - mag.log2().floor()).clamp_min(0)
    worst = float((d * torch.pow(2.0, -bin_gap.double())).max())
    assert worst <= _COMPACT_TAIL_ULP, (
        f"compact tail beyond the measured contraction class ({ctx}): "
        f"{worst:.1f} operand-scale ulps — lookup bug, not contraction"
    )


def test_adams_grad_correct_triton_matches_torch():
    """Triton kernel vs torch reference, sparse and full legs, plus the
    every-score content oracle (the registry formula with
    optimizer="adams" on the same PRE-commit state)."""
    device = "cuda"
    n = 4096
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.95, 1e-8, 1e-2
    torch.manual_seed(3)
    param0 = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    g0 = torch.randn(n)
    mask = (torch.rand(n) < 0.4).to(torch.uint8)

    # Sparse path: fused apply; grad buffer untouched
    p_d = param0.clone().to(device); mm_d = m0.clone().to(device)
    vv_d = v0.clone().to(device); g_d = g0.clone().to(device)
    mk_d = mask.clone().to(device)
    adams_grad_correct(p_d, g_d, mm_d, vv_d, lr, b1, b2, eps, wd,
                       mask=mk_d, emit_score=None)
    p_ref, m_ref, v_ref = param0.clone(), m0.clone(), v0.clone()
    g_ref = g0.clone()
    adams_grad_correct_torch(p_ref, g_ref, m_ref, v_ref, lr, b1, b2, eps,
                             wd, mask, None)
    torch.testing.assert_close(p_d.cpu(), p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(mm_d.cpu(), m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(vv_d.cpu(), v_ref, rtol=1e-5, atol=1e-7)
    assert torch.equal(g_d.cpu(), g_ref), "sparse step must not touch grads"

    # Full path: every registered score, content vs the registry formula
    # (optimizer="adams"), purity of the m/v commit across modes, and the
    # fused apply equal to the default run's
    def run(score):
        w = param0.clone().to(device)
        g = g0.clone().to(device)
        m, v = m0.clone().to(device), v0.clone().to(device)
        adams_grad_correct(w, g, m, v, lr, b1, b2, eps, wd,
                           mask=None, emit_score=score)
        return w, g, m, v

    w1, g1, m1, v1 = run("update")
    for name in sorted(score_names()):
        w2, g2, m2, v2 = run(name)
        ref = get_score_spec(name).formula(
            param0, m0, v0, g0, b1, b2, eps, wd, 1.0, 1.0,
            update_includes_wd=True, optimizer="adams")
        # purity: the m/v commit is bitwise identical across every mode
        assert torch.equal(m1.cpu(), m2.cpu())
        assert torch.equal(v1.cpu(), v2.cpu())
        rel = ((g2.cpu().double() - ref.double()).abs().max()
               / ref.double().abs().max().clamp(min=1e-30)).item()
        assert rel < 1e-5, (name, rel)
        torch.testing.assert_close(w2.cpu(), w1.cpu(), rtol=1e-5, atol=1e-7)
    # the name->int map equals the branch order (the AdamW pin's twin)
    assert _TRITON_SCORE_MAP == {"update": 0, "abs_g": 1,
                                 "abs_m": 2, "coi": 3}
    # unknown name refused host-side with the owned message
    try:
        adams_grad_correct(param0.clone().to(device), g0.clone().to(device),
                           m0.clone().to(device), v0.clone().to(device),
                           lr, b1, b2, eps, wd, mask=None, emit_score="bogus")
        raise AssertionError("unknown emit_score accepted")
    except ValueError as e:
        assert "_adams_grad_correct_kernel branch" in str(e), e
    print("ok: AdamS grad-correct triton == torch reference (sparse/full, "
          "every score)")


def test_adams_v_base_ordering_on_gpu():
    """The committed v on the compiled kernel path is the PRE-commit-m
    recursion: the opposite-sign / crossed-scale fixture whose three
    candidate bases (pre-commit m, post-commit m_new, stored v_old) are
    pairwise distinct, with beta values that make every arithmetic step
    exact in fp32."""
    device = "cuda"
    b1, b2, lr, wd, eps = 0.5, 0.5, 1.0, 0.0, 0.0
    for m_val, g_val, v_old in ((-2.0, 2.0, 8.0), (2.0 ** -10, 2.0 ** 6, 1.0)):
        m = torch.tensor([m_val]).to(device)
        g = torch.tensor([g_val]).to(device)
        v = torch.tensor([v_old]).to(device)
        p = torch.tensor([1.0]).to(device)
        m_new = (b1 * m + (1 - b1) * g).cpu()
        v_spec = b2 * (m.cpu() * m.cpu()) + (1 - b2) * g.cpu() * g.cpu()
        v_post_m = b2 * (m_new * m_new) + (1 - b2) * g.cpu() * g.cpu()
        v_adamw = b2 * v.cpu() + (1 - b2) * g.cpu() * g.cpu()
        assert len({v_spec.item(), v_post_m.item(), v_adamw.item()}) == 3
        adams_grad_correct(p, g, m, v, lr, b1, b2, eps, wd,
                           mask=None, emit_score=None)
        assert v.cpu().item() == v_spec.item(), (m_val, g_val)
        assert m.cpu().item() == m_new.item(), (m_val, g_val)
    print("ok: AdamS kernel v base is the pre-commit m (GPU)")


def test_adams_sparse_fusion_bitwise():
    """coeff+clear+fold fused into grad-correct == the standalone
    three-pass sequence, bitwise, on every buffer (the AdamW GPU pin,
    replicated on the AdamS kernel)."""
    device = "cuda"
    n = 8192
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1)
    for coeff in (0.125, 0.7, 1.0):
        for fold, interval in [(False, 8.0), (True, 8.0), (True, 7.0)]:
            g0 = torch.randn(n)
            mask = (torch.rand(n) < 0.3).to(torch.uint8)
            p, m, v = torch.randn(n), torch.randn(n) * .1, torch.rand(n) * .01

            fp, fg, fm, fv = (t.clone().to(device) for t in (p, g0, m, v))
            adams_grad_correct(fp, fg, fm, fv, mask=mask.to(device),
                               emit_score=None, coeff=coeff, clear=True,
                               fold=fold, interval=interval, **kw)

            rp, rg, rm, rv = (t.clone().to(device) for t in (p, g0, m, v))
            dmask = mask.to(device)
            if coeff != 1.0:
                rg = torch.where(dmask > 0, rg * coeff, rg)
            adams_grad_correct(rp, rg, rm, rv, mask=dmask, emit_score=None,
                               **kw)
            rg.masked_fill_(dmask > 0, 0.0)
            if fold:
                inv = float((torch.ones(1, dtype=torch.float32)
                             / interval).item())
                rg.mul_(inv)
            for got, ref, name in ((fp, rp, "param"), (fm, rm, "exp_avg"),
                                   (fv, rv, "exp_avg_sq"), (fg, rg, "grad")):
                assert torch.equal(got, ref), (
                    f"fused sparse path diverged ({name}, coeff={coeff}, "
                    f"fold={fold})"
                )


def test_adams_full_step_dual_lr_triton_vs_torch():
    """Boundary dual-lr on GPU: the LR_SPLIT kernel matches the torch
    reference on every buffer, the commit/score stay bitwise the
    single-lr run, active coords keep lr, the tail's displacement is the
    scale multiple, and scale 0 freezes the tail bitwise while the score
    still materializes."""
    torch.manual_seed(45)
    device = "cuda"
    n = 4096
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.95, 1e-8, 1e-2
    scale = 0.5
    param0 = torch.randn(n)
    m0, v0 = torch.randn(n) * 0.1, torch.rand(n) * 0.1
    g0 = torch.randn(n)
    mask = (torch.rand(n) < 0.4).to(torch.uint8)
    active = mask > 0

    def run(scale, with_mask):
        p = param0.clone().to(device)
        mm, vv = m0.clone().to(device), v0.clone().to(device)
        g = g0.clone().to(device)
        adams_grad_correct(p, g, mm, vv, lr, b1, b2, eps, wd,
                           mask=mask.to(device) if with_mask else None,
                           emit_score="update",
                           inactive_lr_scale=scale)
        return p.cpu(), g.cpu(), mm.cpu(), vv.cpu()

    p_s, g_s, m_s, v_s = run(scale, True)
    p_n, g_n, m_n, v_n = run(1.0, False)

    p_ref, m_ref, v_ref = param0.clone(), m0.clone(), v0.clone()
    g_ref = g0.clone()
    adams_grad_correct_torch(p_ref, g_ref, m_ref, v_ref, lr, b1, b2, eps, wd,
                             mask, "update", inactive_lr_scale=scale)
    torch.testing.assert_close(p_s, p_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g_s, g_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m_s, m_ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(v_s, v_ref, rtol=1e-4, atol=1e-6)

    # vs the single-lr boundary run: same commit and score (few-ulp),
    # active coords at the same lr, split apply == the where-lr oracle
    torch.testing.assert_close(g_s, g_n, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m_s, m_n, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(v_s, v_n, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(p_s[active], p_n[active],
                               rtol=1e-5, atol=1e-7)
    lr_eff = torch.where(active, lr, lr * scale)
    torch.testing.assert_close(p_s, param0 - lr_eff * g_s,
                               rtol=1e-5, atol=1e-7)

    # scale-0 corner: score materializes bitwise, tail bitwise untouched
    p_z, g_z, _, _ = run(0.0, True)
    assert torch.equal(g_z, g_s), \
        "scale-0 score must materialize bitwise like any split run"
    assert torch.equal(p_z[~active], param0[~active]), \
        "scale-0 tail must be bitwise untouched"
    assert not torch.equal(p_z[active], param0[active]), \
        "scale-0 active coords must still step"
    print("ok: AdamS full-step dual-lr triton == torch; tail lr*scale")


def test_adams_compact_grad_correct_ab():
    """Compact chain (fused pack -> merge-join grad-correct, both search
    forms) vs the writeback chain (index_select*scale -> scatter ->
    masked fused kernel): values/grad bitwise, m/v/param in the measured
    contraction class (operand-scale ulps), fold arms included, the
    k == 0 boundary bitwise, rerun stability."""
    device = "cuda"
    n = 200_000
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1)
    coeff, factor = 0.7, 0.125
    for fold, interval in [(False, 8.0), (True, 8.0), (True, 7.0)]:
        g0 = torch.randn(n)
        mask = (torch.rand(n) < 0.3).to(torch.uint8)
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        k = idx.numel()
        base = {t: torch.randn(n) * s for t, s in (("p", 1.0), ("m", 0.1))}
        base["v"] = torch.rand(n) * 0.01   # inert load/store only
        base["g"] = g0

        def chain_a(dir_table):
            a = {kk: vv.clone().to(device) for kk, vv in base.items()}
            vals = torch.empty(k, device=device)
            pack_selected(a["g"], idx, vals, factor)
            adams_grad_correct_compact(
                a["p"], a["g"], a["m"], a["v"], idx, vals, 0,
                coeff=coeff, fold=fold, interval=interval,
                dir_table=dir_table, **kw)
            return a

        b = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vb = torch.empty(k, device=device)
        torch.index_select(b["g"], 0, idx, out=vb)
        vb *= factor
        scatter_selected(b["g"], idx, vb)
        adams_grad_correct(b["p"], b["g"], b["m"], b["v"],
                           mask=mask.to(device), emit_score=None,
                           coeff=coeff, clear=True, fold=fold,
                           interval=interval, **kw)
        a_bin = chain_a(None)
        a_tab = chain_a(build_dir_table(idx, n))
        a_tab2 = chain_a(build_dir_table(idx, n))  # rerun stability
        # values / grad end state bitwise (vs the writeback oracle AND
        # across the two search forms)
        assert torch.equal(a_bin["g"], b["g"]), "grad end state diverged"
        assert torch.equal(a_tab["g"], a_bin["g"]), \
            "table/bisection grad end state diverged"
        bits = lambda t: t.view(torch.int32)
        assert all(torch.equal(bits(a_tab2[x]), bits(a_tab[x]))
                   for x in ("p", "m", "v", "g")), \
            "table kernel not rerun-bitwise-stable"
        # tail: measured contraction class, AdamS operand scales (v is
        # rebuilt from the pre-commit m, no correction divides)
        gc = base["g"].to(device) * factor * coeff * mask.float().to(device)
        m0d = base["m"].to(device)
        v_new = kw["beta2"] * (m0d * m0d) + (1 - kw["beta2"]) * gc * gc
        m_new = kw["beta1"] * m0d + (1 - kw["beta1"]) * gc
        upd = (kw["weight_decay"] * base["p"].abs().to(device)
               + m_new.abs() / (v_new.sqrt() + kw["eps"]))
        scales = {
            "p": torch.maximum(base["p"].abs().to(device), kw["lr"] * upd),
            "m": torch.maximum(kw["beta1"] * m0d.abs(),
                               (1 - kw["beta1"]) * gc.abs()),
            "v": torch.maximum(kw["beta2"] * (m0d * m0d).abs(),
                               (1 - kw["beta2"]) * gc.abs() ** 2),
        }
        for name in ("p", "m", "v"):
            _assert_tail_ulp_class(a_bin[name], b[name], scales[name],
                                   f"bisect {name}, fold={fold}")
            _assert_tail_ulp_class(a_tab[name], b[name], scales[name],
                                   f"table {name}, fold={fold}")

    # k == 0 boundary: empty idx == all-zero mask, pure decay, bitwise
    n0 = 4096
    z = {"p": torch.randn(n0), "m": torch.randn(n0) * .1,
         "v": torch.rand(n0) * .01, "g": torch.randn(n0)}
    a = {kk: vv.clone().to(device) for kk, vv in z.items()}
    b0 = {kk: vv.clone().to(device) for kk, vv in z.items()}
    idx0 = torch.empty(0, dtype=torch.int32, device=device)
    vals0 = torch.empty(0, device=device)
    adams_grad_correct_compact(a["p"], a["g"], a["m"], a["v"], idx0, vals0,
                               0, coeff=coeff, **kw)
    zero_mask = torch.zeros(n0, dtype=torch.uint8, device=device)
    adams_grad_correct(b0["p"], b0["g"], b0["m"], b0["v"], mask=zero_mask,
                       emit_score=None, **kw)
    for name in ("p", "m", "v", "g"):
        assert torch.equal(a[name], b0[name]), f"k==0 mismatch ({name})"
    print("ok: AdamS compact A/B (values/grad bitwise, tail ulp-class, "
          "table == bisection)")


def test_sga_adams_class_face_on_gpu():
    """One end-to-end SgaAdamS full_step on CUDA (the engine's own call
    path through the class), pinned against the torch reference: the
    plugin routes its records to the Triton kernel on GPU."""
    from types import SimpleNamespace

    from sga.config import set_args
    from sga.config.validation import sga_defaults

    torch.manual_seed(9)
    device = "cuda"
    n = 8192
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.95, 1e-8, 1e-2
    set_args(SimpleNamespace(**{**sga_defaults(), "sga_score": "update"}))
    opt = SgaAdamS()
    param0 = torch.randn(n)
    g0 = torch.randn(n)

    rec = SimpleNamespace(
        main_param=SimpleNamespace(
            data=param0.clone().to(device).requires_grad_(False)),
        grad=SimpleNamespace(data=g0.clone().to(device)),
        _state={}, lr=lr, beta1=b1, beta2=b2, eps=eps,
        weight_decay=wd, step=11, mask_u8=lambda: None,
    )
    ctx = SimpleNamespace(records=[rec])
    opt.full_step(ctx)

    p_ref = param0.clone()
    g_ref = g0.clone()
    m_ref = torch.zeros(n)
    v_ref = torch.zeros(n)
    adams_grad_correct_torch(p_ref, g_ref, m_ref, v_ref, lr, b1, b2, eps, wd,
                             mask=None, emit_score="update")
    torch.testing.assert_close(rec.main_param.data.cpu(), p_ref,
                               rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(rec.grad.data.cpu(), g_ref,
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rec._state["exp_avg"].cpu(), m_ref,
                               rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(rec._state["exp_avg_sq"].cpu(), v_ref,
                               rtol=1e-5, atol=1e-7)
    print("ok: SgaAdamS class path on GPU (full_step -> Triton kernel)")


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
