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
  the standalone sequence it replaces: bitwise per buffer;
- the compact-emission path: fused pack + merge-join
  grad-correct vs the writeback chain — three-leg criterion (values/grad
  bitwise; m/v/param ulp-class: Triton FMA contraction differs per
  compilation, same-binary reruns bitwise-stable); pack contract incl. the
  factor==1.0 constexpr skip; compact norm determinism (no atomics — the
  run-to-run jitter source the masked kernel carried is gone).

Run: python tests/test_kernels_gpu.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.kernels import (
    compact_norm_sq_contribution,
    masked_norm_sq_contribution,
    pack_selected,
    scatter_selected,
)
from sga.kernels_torch import pack_selected_torch
from sga.plugins.optimizer.adamw import (
    adamw_grad_correct,
    adamw_grad_correct_compact,
    build_dir_table,
)
from sga.plugins.optimizer.adamw_torch import grad_correct_torch
from sga.plugins.scores import Score, emittable_score_names, get_score_spec

torch.manual_seed(0)

# Measured contraction class of the compact merge-join tail vs the masked
# kernel tail: worst ~2.3 ulps AT THE OPERAND SCALE across every A/B arm,
# dense-run/rec_start geometry leg and seed (FMA contraction is decided per
# compilation; the yardstick must be the operand scale — near a cancelling
# coordinate the result falls many binades below its inputs, where a
# result-relative bound is meaningless). A real lookup bug moves
# committed-grad-sized chunks: ~1e6 operand-scale ulps. The class is pinned
# just above the measurement so re-seeds do not flap.
_COMPACT_TAIL_ULP = 4


def _ulp_distance(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Ordered-int ULP distance between two finite fp32 tensors: the
    IEEE-754 bit patterns remapped to a monotonic integer rank (sign bit
    folded into the magnitude order), so the difference is the exact ulp
    distance."""
    ia = got.contiguous().view(torch.int32).to(torch.int64)
    ib = ref.contiguous().view(torch.int32).to(torch.int64)
    ka = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    kb = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return (ka - kb).abs()


def _assert_tail_ulp_class(got, ref, op_scale, ctx):
    """Pin a kernel tail to ``_COMPACT_TAIL_ULP`` ulps at the operand scale.

    ``op_scale`` is the per-element magnitude of the largest operand the
    tail consumes (re-derived from the pre-commit state; the right SCALE is
    required, not the exact operands). The ulp distance is computed on the
    results (ordered-int) and rescaled by the binade gap to the operand
    scale before the bound applies. Non-finite values must additionally
    match bit-exactly — same specials on both sides (int32 view).
    """
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


def test_compact_grad_correct_ab():
    """Three-leg A/B: compact chain (fused pack -> merge-join
    grad-correct) vs the writeback chain (index_select*scale -> scatter ->
    masked fused kernel).

    LEG-1 bitwise: values buffer and the grad buffer END STATE (single-mul
    stores only). LEG-2 ulp-class: m/v/param pinned to the MEASURED
    contraction class — ordered-int ulp distance at the operand scale
    (see _assert_tail_ulp_class): Triton's FMA contraction is decided per
    compilation, so two different kernel bodies can never be bitwise-equal
    in the tail arithmetic, while a real lookup bug moves committed-grad-
    sized chunks and misses the bound by orders of magnitude. Non-finite
    values must match bit-exactly. LEG-3: the same compact binary reruns
    bitwise-stable. Fold arms included (the raw residual read).
    """
    device = "cuda"
    n = 200_000
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1, step=1000)
    coeff, factor = 0.7, 0.125
    for fold, interval in [(False, 8.0), (True, 8.0), (True, 7.0)]:
        g0 = torch.randn(n)
        mask = (torch.rand(n) < 0.3).to(torch.uint8)
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        k = idx.numel()
        # v must be POSITIVE (rand, not randn): the AdamW tail takes
        # sqrt(v_hat) — a signed v would NaN ~half the params identically
        # in both runs and torch.equal reads NaN != NaN (false alarm)
        base = {t: torch.randn(n) * s for t, s in
                (("p", 1.0), ("m", 0.1))} | {
            "v": torch.rand(n) * 0.01}
        base["g"] = g0

        def chain_a():
            a = {kk: vv.clone().to(device) for kk, vv in base.items()}
            vals = torch.empty(k, device=device)
            pack_selected(a["g"], idx, vals, factor)
            adamw_grad_correct_compact(
                a["p"], a["g"], a["m"], a["v"], idx, vals, 0,
                coeff=coeff, fold=fold, interval=interval, **kw)
            return a, vals

        b = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vb = torch.empty(k, device=device)
        torch.index_select(b["g"], 0, idx, out=vb)
        vb *= factor
        scatter_selected(b["g"], idx, vb)
        adamw_grad_correct(b["p"], b["g"], b["m"], b["v"], mask=mask.to(device),
                           emit_mode=None, coeff=coeff, clear=True,
                           fold=fold, interval=interval, **kw)
        a, vals = chain_a()
        a2, vals2 = chain_a()  # LEG-3: same binary, fresh buffers
        # LEG-1: pure memory / single-mul stores are bitwise
        assert torch.equal(vals, vb), "values diverged (must be bitwise)"
        assert torch.equal(a["g"], b["g"]), "grad end state diverged"
        # LEG-3 compares raw BITS (int views): torch.equal would read a
        # NaN as != itself even against an identical bit pattern
        bits = lambda t: t.view(torch.int32)
        assert all(torch.equal(bits(x), bits(y)) for x, y in
                   [(a2["p"], a["p"]), (a2["m"], a["m"]),
                    (a2["v"], a["v"]), (a2["g"], a["g"]), (vals2, vals)]), \
            "compact kernel is not rerun-bitwise-stable"
        # LEG-2: arithmetic tail pinned to the measured contraction class
        # (operand-scale ulps). The scale re-derives each element's largest
        # operand from the pre-commit state: param's update magnitude uses
        # the committed moments as proxies.
        gc = base["g"] * factor * coeff * mask.float()
        bc1 = 1 - kw["beta1"] ** kw["step"]
        bc2 = 1 - kw["beta2"] ** kw["step"]
        upd = (kw["weight_decay"] * base["p"].abs()
               + (a["m"].cpu().abs() / bc1)
               / ((a["v"].cpu() / bc2).sqrt() + kw["eps"]))
        scales = {
            "p": torch.maximum(base["p"].abs(), kw["lr"] * upd),
            "m": torch.maximum(kw["beta1"] * base["m"].abs(),
                               (1 - kw["beta1"]) * gc.abs()),
            "v": torch.maximum(kw["beta2"] * base["v"].abs(),
                               (1 - kw["beta2"]) * gc.abs() ** 2),
        }
        for name in ("p", "m", "v"):
            _assert_tail_ulp_class(a[name], b[name], scales[name],
                                   f"{name}, fold={fold}")

    # k == 0 boundary: empty idx == all-zero mask, pure decay, bitwise
    n = 4096
    z = {"p": torch.randn(n), "m": torch.randn(n) * .1,
         "v": torch.rand(n) * .01}
    z["g"] = torch.randn(n)
    a = {kk: vv.clone().to(device) for kk, vv in z.items()}
    b = {kk: vv.clone().to(device) for kk, vv in z.items()}
    idx0 = torch.empty(0, dtype=torch.int32, device=device)
    vals0 = torch.empty(0, device=device)
    adamw_grad_correct_compact(a["p"], a["g"], a["m"], a["v"], idx0, vals0,
                               0, coeff=coeff, **kw)
    zero_mask = torch.zeros(n, dtype=torch.uint8, device=device)
    adamw_grad_correct(b["p"], b["g"], b["m"], b["v"], mask=zero_mask,
                       emit_mode=None, **kw)
    for name in ("p", "m", "v", "g"):
        assert torch.equal(a[name], b[name]), f"k==0 mismatch ({name})"

    # k in {1, 2, 3}: minimal non-empty supports beside the k == 0 leg —
    # the bracket loops at their shortest lengths. Values/grad bitwise;
    # m/v/param within the measured contraction class (different kernel
    # bodies are never bitwise in the FMA tail).
    bc1 = 1 - kw["beta1"] ** kw["step"]
    bc2 = 1 - kw["beta2"] ** kw["step"]
    n3 = 4096
    for k in (1, 2, 3):
        mask = torch.zeros(n3, dtype=torch.uint8)
        mask[torch.randperm(n3)[:k]] = 1
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        z = {"p": torch.randn(n3), "m": torch.randn(n3) * .1,
             "v": torch.rand(n3) * .01, "g": torch.randn(n3)}
        a = {name: t.clone().to(device) for name, t in z.items()}
        vals = torch.empty(k, device=device)
        pack_selected(a["g"], idx, vals, 1.0)
        adamw_grad_correct_compact(a["p"], a["g"], a["m"], a["v"],
                                   idx, vals, 0, coeff=coeff, **kw)
        b = {name: t.clone().to(device) for name, t in z.items()}
        adamw_grad_correct(b["p"], b["g"], b["m"], b["v"],
                           mask=mask.to(device), emit_mode=None,
                           coeff=coeff, clear=True, **kw)
        assert torch.equal(vals, z["g"].to(device)[idx.long()]), \
            f"k={k} values diverged"
        assert torch.equal(a["g"], b["g"]), f"k={k} grad end state diverged"
        gc = z["g"] * coeff * mask.float()
        upd = (kw["weight_decay"] * z["p"].abs()
               + (a["m"].cpu().abs() / bc1)
               / ((z["v"] / bc2).sqrt() + kw["eps"]))
        scales = {
            "p": torch.maximum(z["p"].abs(), kw["lr"] * upd),
            "m": torch.maximum(kw["beta1"] * z["m"].abs(),
                               (1 - kw["beta1"]) * gc.abs()),
            "v": torch.maximum(kw["beta2"] * z["v"].abs(),
                               (1 - kw["beta2"]) * gc.abs() ** 2),
        }
        for name in ("p", "m", "v"):
            _assert_tail_ulp_class(a[name], b[name], scales[name],
                                   f"k={k}, {name}")


def test_compact_grad_correct_dense_runs_and_rec_start():
    """Merge-join geometry the Bernoulli A/B cannot reach: fully selected
    1024-blocks amid sparse background (a lower bound over d elements has
    d+1 candidate positions, i.e. ceil(log2(d+1)) iterations — a lane
    search one iteration short leaves (la, lb) un-converged inside a full
    block and silently drops the second coordinate, committing grad=0
    there), exactly-block and one-short lengths, single selections at the
    first/second/last position of a block, and the rec_start > 0 span
    arithmetic (bucket-level idx vs record-local coords) with a segment
    straddling a record-local block boundary and an empty record segment."""
    device = "cuda"
    n = 16384
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1, step=1000)
    coeff, factor = 0.7, 0.125
    bc1 = 1 - kw["beta1"] ** kw["step"]
    bc2 = 1 - kw["beta2"] ** kw["step"]

    def ab(mask, fold=False, interval=8.0, rec_lo=0, rec_hi=None):
        """One A/B on the [rec_lo, rec_hi) record slice of a bucket-wide
        mask: the compact call consumes the bucket-level (idx, values)
        with rec_start == rec_lo; the reference runs the masked chain on
        the full buffers. Values/grad bitwise, tail within the measured
        contraction class."""
        rec_hi = n if rec_hi is None else rec_hi
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        k = idx.numel()
        base = {"p": torch.randn(n), "m": torch.randn(n) * 0.1,
                "v": torch.rand(n) * 0.01, "g": torch.randn(n)}
        gc = base["g"] * factor * coeff * mask.float()

        a = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vals = torch.empty(k, device=device)
        pack_selected(a["g"], idx, vals, factor)
        sl = slice(rec_lo, rec_hi)
        adamw_grad_correct_compact(
            a["p"][sl], a["g"][sl], a["m"][sl], a["v"][sl],
            idx, vals, rec_lo, coeff=coeff, fold=fold,
            interval=interval, **kw)

        b = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vb = torch.empty(k, device=device)
        torch.index_select(b["g"], 0, idx, out=vb)
        vb *= factor
        scatter_selected(b["g"], idx, vb)
        adamw_grad_correct(b["p"], b["g"], b["m"], b["v"],
                           mask=mask.to(device), emit_mode=None,
                           coeff=coeff, clear=True, fold=fold,
                           interval=interval, **kw)

        assert torch.equal(a["g"][sl], b["g"][sl]), \
            f"grad end state diverged (rec {rec_lo}:{rec_hi}, fold={fold})"
        upd = (kw["weight_decay"] * base["p"].abs()
               + (a["m"].cpu().abs() / bc1)
               / ((a["v"].cpu() / bc2).sqrt() + kw["eps"]))
        scales = {
            "p": torch.maximum(base["p"].abs(), kw["lr"] * upd),
            "m": torch.maximum(kw["beta1"] * base["m"].abs(),
                               (1 - kw["beta1"]) * gc.abs()),
            "v": torch.maximum(kw["beta2"] * base["v"].abs(),
                               (1 - kw["beta2"]) * gc.abs() ** 2),
        }
        for name in ("p", "m", "v"):
            _assert_tail_ulp_class(
                a[name][sl], b[name][sl], scales[name][sl],
                f"{name}, rec {rec_lo}:{rec_hi}, fold={fold}")

    torch.manual_seed(11)
    sparse = (torch.rand(n) < 0.05).to(torch.uint8)
    # dense runs of block-boundary lengths (1025 / 1024 / 1023) amid a
    # sparse background
    for lo, hi in ((4096, 5121), (2048, 3072), (8192, 9215)):
        for fold, interval in ((False, 8.0), (True, 7.0)):
            m = sparse.clone()
            m[lo:hi] = 1
            ab(m, fold=fold, interval=interval)
    # single selections at a block's first / second / last position: the
    # lane search's only equality probe is idx[la] vs coords, so each
    # must resolve exactly
    for pos in (4096, 4097, 5119):
        m = sparse.clone()
        m[pos] = 1
        ab(m)

    # rec_start > 0: record 1 of a two-record bucket, its dense run
    # straddling the record-local block boundary at rec_start + 1024 and
    # a fully selected record-local block ([rec_start+3072, +4096)) deeper
    # in; record 0 is the empty segment (no idx entries below rec_start).
    rs = 1000
    m = (torch.rand(n) < 0.05).to(torch.uint8)
    m[:rs] = 0
    m[rs + 1000:rs + 1060] = 1
    m[rs + 3072:rs + 4096] = 1
    ab(m, rec_lo=rs, rec_hi=n)                      # rec_start = 1000
    ab(m, rec_lo=0, rec_hi=rs)                      # the empty segment
    ab(m, fold=True, interval=7.0, rec_lo=rs, rec_hi=n)
    # bucket idx holding selected coords BELOW rec_start (records are not
    # aligned with selection density): the record's lower bounds must
    # skip them
    m2 = (torch.rand(n) < 0.05).to(torch.uint8)
    m2[100:190] = 1
    m2[rs + 1000:rs + 1060] = 1
    m2[rs + 3072:rs + 4096] = 1
    ab(m2, rec_lo=rs, rec_hi=n)


def test_compact_grad_correct_full_density_clustered_and_bracket_fill():
    """Legs for the bounded-bracket merge-join. (a) Full density:
    every block's segment spans exactly BLOCK (q - p == 1024), the second
    lower_bound's [p, min(k, p+BLOCK)] bound is tight at its maximum — and
    a too-tight bound (p + BLOCK - 1) would drop the last coordinate of
    every block, so this leg bites bracket-bound regressions. (b)
    Heterogeneous occupancy: full, ~30%-partial and empty blocks side by
    side, built deterministically and asserted — global top-k of IID
    scores yields exchangeable per-block occupancy instead (no empty
    blocks), so the heterogeneity must be constructed. (c) Adversarial
    bracket fill: a record whose length is not a multiple of BLOCK ends
    mid-block dense, and the NEXT record's selections complete that
    bracket — those entries sit inside the record's [c0, c1) window at
    coords equal to the out-of-range lanes', so what blocks their commit
    is the ``valid = offs < n_elements`` mask on found/store in the
    kernel (adamw.py), not the c-bounds."""
    device = "cuda"
    n = 16384
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1, step=1000)
    coeff, factor = 0.7, 0.125
    bc1 = 1 - kw["beta1"] ** kw["step"]
    bc2 = 1 - kw["beta2"] ** kw["step"]

    def ab(mask, fold=False, interval=8.0, rec_lo=0, rec_hi=None):
        rec_hi = n if rec_hi is None else rec_hi
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        k = idx.numel()
        base = {"p": torch.randn(n), "m": torch.randn(n) * 0.1,
                "v": torch.rand(n) * 0.01, "g": torch.randn(n)}
        gc = base["g"] * factor * coeff * mask.float()

        a = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vals = torch.empty(k, device=device)
        pack_selected(a["g"], idx, vals, factor)
        sl = slice(rec_lo, rec_hi)
        adamw_grad_correct_compact(
            a["p"][sl], a["g"][sl], a["m"][sl], a["v"][sl],
            idx, vals, rec_lo, coeff=coeff, fold=fold,
            interval=interval, **kw)

        b = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vb = torch.empty(k, device=device)
        torch.index_select(b["g"], 0, idx, out=vb)
        vb *= factor
        scatter_selected(b["g"], idx, vb)
        adamw_grad_correct(b["p"], b["g"], b["m"], b["v"],
                           mask=mask.to(device), emit_mode=None,
                           coeff=coeff, clear=True, fold=fold,
                           interval=interval, **kw)

        assert torch.equal(a["g"][sl], b["g"][sl]), \
            f"grad end state diverged (rec {rec_lo}:{rec_hi}, fold={fold})"
        upd = (kw["weight_decay"] * base["p"].abs()
               + (a["m"].cpu().abs() / bc1)
               / ((a["v"].cpu() / bc2).sqrt() + kw["eps"]))
        scales = {
            "p": torch.maximum(base["p"].abs(), kw["lr"] * upd),
            "m": torch.maximum(kw["beta1"] * base["m"].abs(),
                               (1 - kw["beta1"]) * gc.abs()),
            "v": torch.maximum(kw["beta2"] * base["v"].abs(),
                               (1 - kw["beta2"]) * gc.abs() ** 2),
        }
        for name in ("p", "m", "v"):
            _assert_tail_ulp_class(
                a[name][sl], b[name][sl], scales[name][sl],
                f"{name}, rec {rec_lo}:{rec_hi}, fold={fold}")

    torch.manual_seed(22)
    # (a) 100% density: every block span == BLOCK, single record and a
    # mid-bucket record slice; both fold arms (all-dense masks: the FOLD
    # store only fires on (b)'s non-dense fold leg)
    full = torch.ones(n, dtype=torch.uint8)
    for fold, interval in ((False, 8.0), (True, 7.0)):
        ab(full.clone(), fold=fold, interval=interval)
        ab(full.clone(), fold=fold, interval=interval, rec_lo=5000)

    # (b) heterogeneous occupancy, constructed explicitly: 5 full + 4
    # partial (~30% Bernoulli) + 7 empty 1024-blocks — the shape the
    # bench's real arm measures (empty_frac 0.63 / full_frac 0.15). A
    # global top-k of IID scores cannot produce it (per-block occupancy is
    # exchangeable around ~k/BLOCK, no empty blocks), so the construction
    # is deterministic and its heterogeneity is asserted.
    bsz = 1024
    m = torch.zeros(n, dtype=torch.uint8)
    m[:5 * bsz] = 1
    m[5 * bsz:9 * bsz] = (torch.rand(4 * bsz) < 0.3).to(torch.uint8)
    occ = m.view(-1, bsz).sum(dim=1)
    assert int((occ == 0).sum()) > 0 and int((occ == bsz).sum()) > 0, (
        "leg (b) lost its occupancy heterogeneity (empty/full blocks)"
    )
    ab(m)
    # fold=True on a NON-dense mask: the compact FOLD store
    # (valid & ~found) actually fires at inactive coords
    ab(m, fold=True, interval=7.0)

    # (c) bracket fill: record 0 is dense and ends mid-block (1500 is not
    # a multiple of 1024), record 1 continues dense from the next coord —
    # record 1's idx entries land INSIDE record 0's final-block window at
    # coords equal to the out-of-range lanes', and only the
    # valid = offs < n_elements mask on found/store keeps them from
    # committing (not the [c0, c1) bounds); record 1's own first block
    # starts at an unaligned c0 = 1500
    m = torch.ones(n, dtype=torch.uint8)
    ab(m, rec_lo=0, rec_hi=1500)
    ab(m, rec_lo=1500, rec_hi=n)
    ab(m, fold=True, interval=9.0, rec_lo=1500, rec_hi=n)
    # sparse tail on record 1 so its last blocks are partial too
    m2 = torch.ones(n, dtype=torch.uint8)
    m2[12000:] = (torch.rand(n - 12000) < 0.1).to(torch.uint8)
    ab(m2, rec_lo=1500, rec_hi=n)


def test_compact_grad_correct_dir_table():
    """compact_search=table legs.

    The directory table's exactness is the ONLY hard pin of the four-link
    invariance chain: a containment-PRESERVING table error is invisible to
    every downstream instrument (a bracket containing the true p/q
    bisects to the same unique fixed point), so table ==
    torch.searchsorted 'left' is asserted elementwise over adversarial
    geometries. The corrupted-entry leg guards that comparator staying
    exact — it pins the pin (a future tolerance-comparator swap), not a
    plausible kernel fault (the poisoned entry differs from `ref` by
    construction). The consumer legs pin: fired-set equality vs the
    bisection form (the integer path), the fp tail to the measured
    contraction class, same-binary rerun stability, the r=1023
    extreme-residue geometry (every p/q at its bracket's far edge)
    through the writeback oracle chain, and bisecting brackets over
    sparse idx (rec_start 1/500 legs put r = c0 mod 1024 != 0 in every
    block). The r=1023 leg BITES TABLE_DEPTH=9: closed width-1023
    brackets with the true p/q at the far edge exit 9 rounds with
    a = true - 1, and the q-side drop makes every fully-selected block's
    last coordinate fail its equality probe (committed 0) — exactly what
    the m/v ulp asserts catch (the p-side under-convergence alone is
    lane-absorbed: the lane search re-derives within [p, q)).
    """
    from sga.support import BucketSupport

    device = "cuda"
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
              weight_decay=0.1, step=1000)
    coeff = 0.7
    bc1 = 1 - kw["beta1"] ** kw["step"]
    bc2 = 1 - kw["beta2"] ** kw["step"]

    # --- TB: table == searchsorted 'left' on adversarial geometries ---
    torch.manual_seed(41)  # per-leg seed: the masks must not shift when a
    n = 1 << 20            # test is inserted above
    runs = torch.zeros(n, dtype=torch.uint8).view(-1, 1024)
    runs[(torch.arange(runs.shape[0]) % 8) < 3] = 1  # full blocks adjacent
    for tag, mask in (
        ("empty", torch.zeros(8192, dtype=torch.uint8)),
        ("short-777", (torch.rand(777) < 0.5).to(torch.uint8)),
        ("sparse5%", (torch.rand(n) < 0.05).to(torch.uint8)),
        ("full-dense", torch.ones(8192, dtype=torch.uint8)),
        ("full-block-runs", runs.reshape(-1)),
    ):
        nn = mask.numel()
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        table = build_dir_table(idx, nn)
        keys = torch.arange(table.numel(), device=device,
                            dtype=torch.int64) * 1024
        ref = torch.searchsorted(idx.long(), keys)
        assert torch.equal(table.long(), ref), (
            f"table != searchsorted 'left' ({tag}) — the only hard pin "
            "of the invariance chain failed")
        if idx.numel() > 100:
            # guards the comparator, not a kernel fault: bad[mid] += 1
            # differs from ref by construction — the leg only proves the
            # pin stays exact (no tolerance comparator may replace it)
            bad = table.clone()
            bad[bad.numel() // 2] += 1
            assert not torch.equal(bad.long(), ref), (
                f"({tag}) a corrupted table entry evaded the searchsorted "
                "pin — the comparator went tolerant")

    # int32 table-key contract: the assert fires on host geometry alone
    # (no oversized tensor is built — the guard reads total_n before any
    # allocation/launch)
    try:
        build_dir_table(idx[:8].contiguous(), (1 << 31) - 1)
        raise AssertionError("int32 table-key overflow accepted silently")
    except AssertionError as e:
        assert "2^31" in str(e), e
    # companion happy path just below the guard's edge (host arithmetic:
    # ((2^31 - 2048) >> 10 + 1) * 1024 = 2^31 - 1024 < 2^31) must NOT
    # raise — no oversized tensor is constructed, the 8-entry idx slice
    # keeps the launch trivial
    build_dir_table(idx[:8].contiguous(), (1 << 31) - 1024 * 2)

    # BucketSupport.dir_table: lazy, cached per cycle, == searchsorted
    torch.manual_seed(42)  # per-leg seed (cf. the TB leg above)
    mask = (torch.rand(200_000) < 0.3).to(torch.uint8).to(device)
    bs = BucketSupport(mask_u8=mask)
    t1 = bs.dir_table
    t2 = bs.dir_table
    assert t1 is t2, "dir_table not cached (would rebuild per record)"
    assert torch.equal(
        t1.long(),
        torch.searchsorted(bs.idx.long(),
                           torch.arange(t1.numel(), device=device,
                                        dtype=torch.int64) * 1024)), \
        "BucketSupport.dir_table != searchsorted"
    del bs, t1, t2

    # --- on/off A/B (compact_search table vs bisection), fold arms included:
    # LEG-1 grad end state bitwise (found is the kernels' integer
    # predicate — identical when the integer path holds); fired-set
    # symmetric difference within the rounding-absorption tail (the
    # m-based detector flips only where |0.1*g| sinks below the stored
    # ulp of 0.9*m — each kernel's own summation rounding decides that
    # boundary); m/v/param in the measured contraction class. The
    # rec_start legs matter beyond the whole-bucket form: rec_start 1/500
    # leave r = c0 mod 1024 != 0 in every block, so the TABLE_DEPTH
    # bracket loops actually bisect over sparse idx (at r == 0 both
    # brackets collapse to width 0 — p/q are direct table reads, zero
    # probes executed).
    torch.manual_seed(7)
    n = 200_000
    for fold, interval in [(False, 8.0), (True, 8.0)]:
        g0 = torch.randn(n)
        mask = (torch.rand(n) < 0.3).to(torch.uint8)
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        k = idx.numel()
        base = {t: torch.randn(n) * s for t, s in (("p", 1.0), ("m", 0.1))}
        base["v"] = torch.rand(n) * 0.01
        base["g"] = g0
        for rec_start in (0, 1, 500):
            sl = slice(rec_start, n)

            def chain(dir_table):
                a = {kk: vv.clone().to(device) for kk, vv in base.items()}
                vals = torch.empty(k, device=device)
                pack_selected(a["g"], idx, vals, 0.125)
                adamw_grad_correct_compact(
                    a["p"][sl], a["g"][sl], a["m"][sl], a["v"][sl],
                    idx, vals, rec_start,
                    coeff=coeff, fold=fold, interval=interval,
                    dir_table=dir_table, **kw)
                return a

            table = build_dir_table(idx, n)
            a_bin = chain(None)
            a_tab = chain(table)
            a_tab2 = chain(table)  # rerun stability of the table binary
            # rerun: raw bits identical
            bits = lambda t: t.view(torch.int32)
            assert all(torch.equal(bits(a_tab2[x]), bits(a_tab[x]))
                       for x in ("p", "m", "v", "g")), \
                "table kernel not rerun-bitwise-stable"
            # LEG-1: grad end state bitwise. Vacuous at fold=False
            # (neither compact form writes g — the pack zeroed both
            # sides' selected coords before the call); at fold=True the
            # (~found)-masked fold store detects a ghost-found (an
            # unselected coord claimed) but not missed coords (raw was
            # pre-zeroed by pack)
            assert torch.equal(a_bin["g"], a_tab["g"]), \
                "grad end state diverged (fold-arm ghost-found)"
            # fired-set: symmetric difference within the absorption tail,
            # counted on the record's slice only — coords below rec_start
            # are untouched by this leg's calls (their m stays m0, which
            # the m != 0.9*m0 probe would misread as fired)
            m0 = base["m"].to(device)
            fired_bin = a_bin["m"][sl] != 0.9 * m0[sl]
            fired_tab = a_tab["m"][sl] != 0.9 * m0[sl]
            sel = mask.to(device).bool()[sl]
            assert int((fired_tab & ~sel).sum()) == 0, "ghost lookup (table)"
            assert int((fired_bin & ~sel).sum()) == 0, "ghost lookup (bisection)"
            sym = int((fired_tab ^ fired_bin).sum())
            # absorption tail: the m-detector flips only at summation
            # rounding ties (~2e-8/element — expected sym is 0 at this
            # k); the bound is deliberately coarse, the ulp legs carry
            # the arithmetic pin
            assert sym <= max(4, k >> 16), (
                f"fired-set symmetric difference {sym} >> absorption tail "
                f"({k} selected) — integer lookup path diverged")
            # m/v/param: measured contraction class (same yardstick as the
            # masked-vs-compact A/B), on the record's slice (the rest of
            # the buffer is identical untouched input on both arms)
            gc = base["g"].to(device) * 0.125 * coeff * mask.float().to(device)
            upd = (kw["weight_decay"] * base["p"].abs().to(device)
                   + (a_tab["m"].abs() / bc1)
                   / ((base["v"].to(device) / bc2).sqrt() + kw["eps"]))
            scales = {
                "p": torch.maximum(base["p"].abs().to(device), kw["lr"] * upd),
                "m": torch.maximum(kw["beta1"] * base["m"].abs().to(device),
                                   (1 - kw["beta1"]) * gc.abs()),
                "v": torch.maximum(kw["beta2"] * base["v"].abs().to(device),
                                   (1 - kw["beta2"]) * gc.abs() ** 2),
            }
            for name in ("p", "m", "v"):
                _assert_tail_ulp_class(a_tab[name][sl], a_bin[name][sl],
                                       scales[name][sl],
                                       f"table-vs-bisection {name}, fold={fold}, "
                                       f"rec_start={rec_start}")

    # --- two records, one shared table (the production loop's shape:
    # _records_pass_compact hands every record the SAME bucket-level
    # table and values, only rec_start differs — record 1's b = c0 >> 10
    # must index the table record 0 consumed, built once over the full
    # bucket width; a per-record rebuild would hide a shared-table
    # indexing bug). Two-record harness pattern of the bisection-path test,
    # record boundary NOT block-aligned so record 1 starts mid-block
    # with a nonzero residue; instruments identical to the A/B legs
    # above.
    torch.manual_seed(43)
    mask = (torch.rand(n) < 0.3).to(torch.uint8)
    idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
    k = idx.numel()
    base = {t: torch.randn(n) * s for t, s in (("p", 1.0), ("m", 0.1))}
    base["v"] = torch.rand(n) * 0.01
    base["g"] = torch.randn(n)
    table = build_dir_table(idx, n)  # BucketSupport-style: once per bucket

    def chain2(dir_table):
        a = {kk: vv.clone().to(device) for kk, vv in base.items()}
        vals = torch.empty(k, device=device)
        pack_selected(a["g"], idx, vals, 0.125)
        for lo, hi in ((0, 1500), (1500, n)):
            adamw_grad_correct_compact(
                a["p"][lo:hi], a["g"][lo:hi], a["m"][lo:hi], a["v"][lo:hi],
                idx, vals, lo,
                coeff=coeff, dir_table=dir_table, **kw)
        return a

    a_bin = chain2(None)
    a_tab = chain2(table)
    a_tab2 = chain2(table)
    bits = lambda t: t.view(torch.int32)
    assert all(torch.equal(bits(a_tab2[x]), bits(a_tab[x]))
               for x in ("p", "m", "v", "g")), \
        "table kernel not rerun-bitwise-stable (two records)"
    assert torch.equal(a_bin["g"], a_tab["g"]), \
        "grad end state diverged (two records, one table)"
    m0 = base["m"].to(device)
    fired_bin = a_bin["m"] != 0.9 * m0
    fired_tab = a_tab["m"] != 0.9 * m0
    sel = mask.to(device).bool()
    assert int((fired_tab & ~sel).sum()) == 0, "ghost lookup (table)"
    assert int((fired_bin & ~sel).sum()) == 0, "ghost lookup (bisection)"
    sym = int((fired_tab ^ fired_bin).sum())
    assert sym <= max(4, k >> 16), (
        f"fired-set symmetric difference {sym} >> absorption tail "
        f"({k} selected) — integer lookup path diverged")
    gc = base["g"].to(device) * 0.125 * coeff * mask.float().to(device)
    upd = (kw["weight_decay"] * base["p"].abs().to(device)
           + (a_tab["m"].abs() / bc1)
           / ((base["v"].to(device) / bc2).sqrt() + kw["eps"]))
    scales = {
        "p": torch.maximum(base["p"].abs().to(device), kw["lr"] * upd),
        "m": torch.maximum(kw["beta1"] * base["m"].abs().to(device),
                           (1 - kw["beta1"]) * gc.abs()),
        "v": torch.maximum(kw["beta2"] * base["v"].abs().to(device),
                           (1 - kw["beta2"]) * gc.abs() ** 2),
    }
    for name in ("p", "m", "v"):
        _assert_tail_ulp_class(a_tab[name], a_bin[name], scales[name],
                               f"table-vs-bisection {name}, two records")

    # --- r=1023 extreme residue: start 1023 over a full-dense bucket —
    # every block's true p/q sits exactly at table[b]+1023 /
    # table[b+1]+1023, the widest 1024-candidate brackets. This leg BITES
    # a TABLE_DEPTH shortfall: 9 rounds exit with a = true - 1 and the
    # q-side drop commits each fully-selected block's last coordinate as
    # 0, which the m/v ulp asserts below catch (the p-side
    # under-convergence alone is lane-absorbed — the lane search
    # re-derives within [p, q)). No bitwise instrument here by construction: a["g"] is
    # never written (no fold) and the oracle zeroes b["g"] — found-set
    # sensitivity is carried by the ulp legs; the grad-bitwise pin lives
    # in the A/B legs' rerun-stability assert.
    n = 8192
    rs = 1023
    torch.manual_seed(5)
    z = {"p": torch.randn(n - rs), "m": torch.randn(n - rs) * 0.1,
         "v": torch.rand(n - rs) * 0.01, "g": torch.randn(n - rs)}
    idx = torch.arange(n, dtype=torch.int32, device=device)
    vals = torch.randn(n, device=device)
    table = build_dir_table(idx, n)
    a = {kk: vv.clone().to(device) for kk, vv in z.items()}
    adamw_grad_correct_compact(a["p"], a["g"], a["m"], a["v"], idx, vals,
                               rs, coeff=coeff, dir_table=table, **kw)
    # oracle: record-local buffers (n - rs elements); the record's value
    # segment is vals[rs:] scattered raw, coeff applied by the masked
    # kernel — same single-rounding as the compact in-register multiply
    # (the torch-fallback construction inside the wrapper)
    b = {kk: vv.clone().to(device) for kk, vv in z.items()}
    b["g"] = torch.zeros(n - rs, device=device)
    b["g"][:] = vals[rs:]
    mask = torch.ones(n - rs, dtype=torch.uint8, device=device)
    adamw_grad_correct(b["p"], b["g"], b["m"], b["v"], mask=mask,
                       emit_mode=None, coeff=coeff, clear=True, **kw)
    gc = torch.zeros(n - rs, device=device)
    gc[:] = vals[rs:] * coeff
    upd = (kw["weight_decay"] * z["p"].abs().to(device)
           + (a["m"].abs() / bc1)
           / ((z["v"].to(device) / bc2).sqrt() + kw["eps"]))
    scales = {
        "p": torch.maximum(z["p"].abs().to(device), kw["lr"] * upd),
        "m": torch.maximum(kw["beta1"] * z["m"].abs().to(device),
                           (1 - kw["beta1"]) * gc.abs()),
        "v": torch.maximum(kw["beta2"] * z["v"].abs().to(device),
                           (1 - kw["beta2"]) * gc.abs() ** 2),
    }
    for name in ("p", "m", "v"):
        _assert_tail_ulp_class(a[name], b[name], scales[name],
                               f"r=1023 extreme residue {name}")

    # --- k in {0, 1, 2, 3} through the table path: the bracket loops at
    # their shortest, table degenerate (2-3 entries) ---
    n = 4096
    for k in (0, 1, 2, 3):
        torch.manual_seed(44 + k)  # per-leg seed: each geometry pinned
        mask = torch.zeros(n, dtype=torch.uint8)
        if k:
            mask[torch.randperm(n)[:k]] = 1
        idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
        kk = idx.numel()
        z = {"p": torch.randn(n), "m": torch.randn(n) * 0.1,
             "v": torch.rand(n) * 0.01, "g": torch.randn(n)}
        a = {name: t.clone().to(device) for name, t in z.items()}
        vals = torch.empty(kk, device=device)
        pack_selected(a["g"], idx, vals, 1.0)  # values out, selected zeroed
        table = build_dir_table(idx, n)
        adamw_grad_correct_compact(a["p"], a["g"], a["m"], a["v"], idx,
                                   vals, 0, coeff=coeff, dir_table=table,
                                   **kw)
        b = {name: t.clone().to(device) for name, t in z.items()}
        adamw_grad_correct(b["p"], b["g"], b["m"], b["v"],
                           mask=mask.to(device), emit_mode=None,
                           coeff=coeff, clear=True, **kw)
        # End-state equality re-pins pack-zeroing == clear-zeroing, NOT
        # the found set: at fold=False neither form writes g, so a
        # found-set error cannot show here (found sensitivity is the ulp
        # legs')
        assert torch.equal(a["g"], b["g"]), f"k={k} grad end state diverged"
        gsel = z["g"].to(device) * coeff * mask.float().to(device)
        upd = (kw["weight_decay"] * z["p"].abs().to(device)
               + (a["m"].abs() / bc1)
               / ((z["v"].to(device) / bc2).sqrt() + kw["eps"]))
        scales = {
            "p": torch.maximum(z["p"].abs().to(device), kw["lr"] * upd),
            "m": torch.maximum(kw["beta1"] * z["m"].abs().to(device),
                               (1 - kw["beta1"]) * gsel.abs()),
            "v": torch.maximum(kw["beta2"] * z["v"].abs().to(device),
                               (1 - kw["beta2"]) * gsel.abs() ** 2),
        }
        for name in ("p", "m", "v"):
            _assert_tail_ulp_class(a[name], b[name], scales[name],
                                   f"k={k} table {name}")
    print("ok: compact dir-table legs (searchsorted pin + bite, on/off A/B "
          "fired-set/ulp/rerun, r=1023 extreme, k=0..3, int32 guard)")


def test_pack_selected_contract():
    """Fused pack vs index_select + payload scale + zero-scatter reference:
    bitwise at every factor, including the factor==1.0 constexpr skip; the
    torch fallback agrees on the same contract (CPU)."""
    device = "cuda"
    n = 300_000
    g0 = torch.randn(n)
    mask = (torch.rand(n) < 0.25).to(torch.uint8)
    idx = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(device)
    k = idx.numel()
    for factor in (0.125, 1.0):
        g = g0.clone().to(device)
        vals = torch.empty(k, device=device)
        pack_selected(g, idx, vals, factor)
        ref = g0.clone().to(device)
        ref_vals = ref[idx.long()].clone()
        if factor != 1.0:
            ref_vals *= factor
        ref[idx.long()] = 0.0
        assert torch.equal(vals, ref_vals), f"pack values (factor={factor})"
        assert torch.equal(g, ref), f"pack zero-store (factor={factor})"
    # torch fallback parity (small, CPU)
    gc_ = g0.clone()
    vc_ = torch.empty(k)
    idx_cpu = idx.cpu()
    pack_selected_torch(gc_, idx_cpu, vc_, 0.5)
    ref = g0[idx_cpu.long()] * 0.5
    assert torch.equal(vc_, ref)
    assert torch.count_nonzero(gc_[idx_cpu.long()]).item() == 0


def test_compact_norm_deterministic():
    """Compact norm: same aggregate as the masked contribution (within the
    declared ulp-class reduction-order difference) and DETERMINISTIC across
    reruns — the masked atomic kernel this replaces was the run-to-run
    clip-coeff jitter source the masked atomic kernel carried; the
    compact path removes it from the sparse path)."""
    device = "cuda"
    n = 8192 * 64
    vals = torch.rand(n, device=device) * 2 - 1
    ref_sq = float((vals.cpu() ** 2).sum())
    got = torch.zeros(1, device=device)
    compact_norm_sq_contribution(vals, got)
    assert abs(got.item() - ref_sq) <= 1e-5 * max(1.0, abs(ref_sq))
    # determinism: 20 runs, exactly one distinct sum (pre-registered probe)
    sums = set()
    for _ in range(20):
        t = torch.zeros(1, device=device)
        compact_norm_sq_contribution(vals, t)
        sums.add(t.item())
    assert len(sums) == 1, (
        f"compact norm is nondeterministic ({len(sums)} distinct sums in "
        "20 runs) — the jitter source would be BACK"
    )


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
