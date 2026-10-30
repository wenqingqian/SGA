# Copyright (c) 2026, SGA framework contributors.
"""Triton kernels for the fused replay's fp32 elementwise passes.

Every kernel here replicates one torch eager fp32 chain bit-for-bit (pinned on the
torch 2.9.0+cu128 / triton 3.5.0 stack; the conformance matrix lives in
tests/test_replay_kernels.py):

- plain tensor mul/add/sub are correctly-rounded fp32 with no FMA
  contraction — plain f32 ops with ``enable_fp_fusion=False``;
- ``add_(alpha=)`` and ``addcmul_(value=)`` contract to an FMA inside the
  torch kernel — those points take an explicit ``tl.math.fma`` (with the
  multiplier operand itself rounded as a separate correctly-rounded mul);
- torch's fp32 tensor-tensor div and sqrt are IEEE correctly-rounded on
  every lane including denormals — reproducing them takes an f64 upcast
  with a single round back to f32;
- torch's tensor/python-scalar div is a reciprocal multiply by
  ``f32(1/f32(s))`` computed on the host — the wrappers precompute that
  reciprocal; tensor-scalar mul/add cast the scalar to f32, correctly
  rounded.

Dispatch (killswitch ``REPLAY_FUSED_KERNEL``): ``auto`` engages a kernel
only on CUDA fp32 contiguous tensors with the name in the dispatch table;
``off``, CPU tensors, other dtypes and unknown score names all fall back
to the torch eager path the kernels replicate. Unknown flag values raise
at run entry (same contract as the other replay flags). ``auto`` with no
importable triton warns once and falls back, keeping triton-less nodes
usable. triton itself is pinned == 3.5.0 at import (the conformance
recipe is stack-specific) — a wrong version raises, an absent triton does
not.

Launch configs are fixed (``_BS``/``_WARPS``, no autotune): every value
these kernels produce is launch-config invariant — one lane per element
with masked stores, integer histogram/count atomics whose results are
order-free, and an index-ordered candidate compaction that two
deterministic sweeps make independent of the block partition.
"""
import os
import struct
import warnings

import torch

MODE = os.environ.get("REPLAY_FUSED_KERNEL", "auto")

_BS = 1024
_WARPS = 4

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
else:
    if triton.__version__ != "3.5.0":
        raise RuntimeError(
            "sga.replay_kernels pins triton==3.5.0 (the conformance "
            f"recipe is stack-specific), found {triton.__version__}")

_WARNED = False


def _f32(x):
    """Round a host python float to fp32 (round-to-nearest-even)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _recip32(s):
    """torch's tensor/python-scalar div computes x * f32(1/f32(s))."""
    return _f32(1.0 / _f32(s))


def arm(dev_is_cuda, dtype):
    """The run-level kernel arm: killswitch auto + importable pinned
    triton + CUDA device + fp32 replay dtype. Anything else means the
    torch eager path runs; triton absent warns once."""
    global _WARNED
    if MODE != "auto":
        return False
    if triton is None:
        if not _WARNED:
            warnings.warn(
                "REPLAY_FUSED_KERNEL=auto but triton is not importable; "
                "falling back to the torch eager path")
            _WARNED = True
        return False
    return dev_is_cuda and dtype == torch.float32


def arm_select(score):
    """The per-tensor select arm: same gate plus the contiguous fp32
    storage the bitcast histogram reads lane-by-lane."""
    return arm(score.is_cuda, score.dtype) and score.is_contiguous()


def formula(name):
    """Kernel for a registered score name, or None (unknown names run
    their torch formula — the per-name scope-shrink rule)."""
    return _FORMULAS.get(name)


def _grid(n):
    return (triton.cdiv(n, _BS),)


if triton is not None:

    @triton.jit
    def _k_abs(x, out, n, BS: tl.constexpr):
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        tl.store(out + i, tl.abs(tl.load(x + i, mask=mk)), mask=mk)

    @triton.jit
    def _k_update(w, m, v, g, out, n,
                  b1f, om1f, b2f, om2f, wdf, r_bc1, r_bc2, epsf,
                  BS: tl.constexpr):
        # the materialized AdamW update wd*w + U(g): every mul/add is a
        # separate correctly-rounded op (fusion off), the bias-correction
        # divides are host reciprocals, sqrt and the final div upcast f64
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        tw = tl.load(w + i, mask=mk)
        tm = tl.load(m + i, mask=mk)
        tv = tl.load(v + i, mask=mk)
        tg = tl.load(g + i, mask=mk)
        sa = tm * b1f + tg * om1f
        sb = tv * b2f + (tg * om2f) * tg
        sa = sa * r_bc1
        sb = sb * r_bc2
        den = tl.sqrt(sb.to(tl.float64)).to(tl.float32)
        den = den + epsf
        u = (sa.to(tl.float64) / den.to(tl.float64)).to(tl.float32)
        tl.store(out + i, tw * wdf + u, mask=mk)

    @triton.jit
    def _k_coi(w, m, v, g, out, n,
               b1f, om1f, b2f, om2f, r_bc1, r_bc2, epsf,
               BS: tl.constexpr):
        # |U(g) - U(0)|: two moment chains with the same rounding rules
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        tm = tl.load(m + i, mask=mk)
        tv = tl.load(v + i, mask=mk)
        tg = tl.load(g + i, mask=mk)
        sa = tm * b1f + tg * om1f
        sb = tv * b2f + (tg * om2f) * tg
        sa = sa * r_bc1
        sb = sb * r_bc2
        res = tm * b1f
        res = res * r_bc1
        den = tv * b2f
        den = den * r_bc2
        den = tl.sqrt(den.to(tl.float64)).to(tl.float32)
        den = den + epsf
        res = (res.to(tl.float64) / den.to(tl.float64)).to(tl.float32)
        inner = tl.sqrt(sb.to(tl.float64)).to(tl.float32)
        inner = inner + epsf
        inner = (sa.to(tl.float64) / inner.to(tl.float64)).to(tl.float32)
        tl.store(out + i, tl.abs(inner - res), mask=mk)

    @triton.jit
    def _k_adam(q, m, v, w, lab, sel, n,
                b1f, om1f, b2f, om2f, r_bc1, r_bc2, epsf,
                omlwdf, neglrf, lr2f, lrf, lri_f, wdf,
                HAS_LAB: tl.constexpr, ADAMS: tl.constexpr,
                SPLIT: tl.constexpr, BS: tl.constexpr):
        # one in-place commit (AdamW's recursion, or the AdamS v basis under
        # ADAMS): mul_ / add_(alpha=) / addcmul_ are the FMA-contraction
        # points, the rest follows the recipe. ADAMS rebuilds v from the
        # PRE-commit m (the torch twin's copy_/mul_/mul_ chain — three
        # correctly-rounded muls — then the addcmul_ FMA) before m updates;
        # SPLIT is the boundary dual-lr apply (_apply_lr_split): the step
        # rate is lr on the sel-active coords and lri on the inactive tail,
        # the applied vector is wd*w + upd (two correctly-rounded ops) and
        # the final subtract rides the addcmul_ FMA with its tensor product
        # rounded first — the split scales the WHOLE step, weight-decay
        # component included
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        tq = tl.load(q + i, mask=mk)
        tm = tl.load(m + i, mask=mk)
        tv = tl.load(v + i, mask=mk)
        tw = tl.load(w + i, mask=mk)
        if ADAMS:
            tqq = tq * tq
            tmm = tm * tm
            tv = tl.math.fma(om2f, tqq, tmm * b2f)
            tm = tl.math.fma(om1f, tq, tm * b1f)
        else:
            tm = tl.math.fma(om1f, tq, tm * b1f)
            tqq = tq * tq
            tv = tl.math.fma(om2f, tqq, tv * b2f)
        tmh = tm * r_bc1
        tvh = tl.sqrt((tv * r_bc2).to(tl.float64)).to(tl.float32)
        upd = (tmh.to(tl.float64)
               / (tvh + epsf).to(tl.float64)).to(tl.float32)
        if SPLIT:
            # sel carries the INACTIVE mask bytes (True = the
            # never-communicated tail): those lanes step at lri, the rest
            # at lr
            act = tl.load(sel + i, mask=mk, other=0)
            stepv = tl.where(mk & (act != 0), lri_f, lrf)
            fullv = tw * wdf
            fullv = fullv + upd
            prodv = fullv * stepv
            tw = tl.math.fma(-1.0, prodv, tw)
        else:
            tw = tl.math.fma(neglrf, upd, tw * omlwdf)
        tl.store(m + i, tm, mask=mk)
        tl.store(v + i, tv, mask=mk)
        tl.store(w + i, tw, mask=mk)
        if HAS_LAB:
            tlab = tl.load(lab + i, mask=mk)
            uu = upd * upd
            tl.store(lab + i, tl.math.fma(lr2f, uu, tlab), mask=mk)

    @triton.jit
    def _k_hist(score, hist, nfcnt, n, BS: tl.constexpr):
        # one sweep: the 32769-bin top-16 histogram as int64 atomics
        # (integer adds — the counts are order-free) plus the finite
        # guard's nonfinite-lane count
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        x = tl.load(score + i, mask=mk, other=0.0)
        nf = (x != x) | (tl.abs(x) == float("inf"))
        tl.atomic_add(nfcnt, tl.sum(nf.to(tl.int32)))
        t = (x.to(tl.int32, bitcast=True) & 0x7FFFFFFF) >> 16
        tl.atomic_add(hist + t, 1, mask=mk)

    @triton.jit
    def _k_cand_count(score, blk, n, b, BS: tl.constexpr):
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        x = tl.load(score + i, mask=mk, other=0.0)
        t = (x.to(tl.int32, bitcast=True) & 0x7FFFFFFF) >> 16
        tl.store(blk + tl.program_id(0),
                 tl.sum(((t == b) & mk).to(tl.int32)))

    @triton.jit
    def _k_cand_fill(score, cand, off, n, b, BS: tl.constexpr):
        # deterministic index-ordered compaction: per-block exclusive
        # offsets plus in-block exclusive cumsum reproduce torch's
        # masked-extraction order for any block partition
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        x = tl.load(score + i, mask=mk, other=0.0)
        t = (x.to(tl.int32, bitcast=True) & 0x7FFFFFFF) >> 16
        hit = (t == b) & mk
        ex = tl.cumsum(hit.to(tl.int32)) - hit.to(tl.int32)
        base = tl.load(off + tl.program_id(0))
        tl.store(cand + base + ex, x, mask=hit)

    @triton.jit
    def _k_gt_eq(score, gt, eq, cnts, n, threshf, BS: tl.constexpr):
        # one sweep writing both compare masks and their int64 counts —
        # compares against the f32-cast threshold are exact, the counts
        # are integer atomics (order-free)
        i = tl.program_id(0) * BS + tl.arange(0, BS)
        mk = i < n
        x = tl.load(score + i, mask=mk, other=0.0)
        g = (x > threshf) & mk
        e = (x == threshf) & mk
        tl.store(gt + i, g.to(tl.int1), mask=mk)
        tl.store(eq + i, e.to(tl.int1), mask=mk)
        tl.atomic_add(cnts + 0, tl.sum(g.to(tl.int32)))
        tl.atomic_add(cnts + 1, tl.sum(e.to(tl.int32)))


def _check_contig(**ts):
    for name, t in ts.items():
        if not t.is_contiguous():
            raise ValueError(
                f"replay_kernels requires contiguous tensors ({name} is "
                "not); the fused call sites pass contiguous slices")


def _formula_update_k(w, m, v, g, b1, b2, eps, wd, bc1c, bc2c):
    _check_contig(w=w, m=m, v=v, g=g)
    n = w.numel()
    out = torch.empty(n, dtype=torch.float32, device=w.device)
    _k_update[_grid(n)](
        w, m, v, g, out, n,
        _f32(b1), _f32(1.0 - b1), _f32(b2), _f32(1.0 - b2), _f32(wd),
        _recip32(bc1c), _recip32(bc2c), _f32(eps),
        BS=_BS, num_warps=_WARPS, enable_fp_fusion=False)
    return out


def _formula_coi_k(w, m, v, g, b1, b2, eps, wd, bc1c, bc2c):
    _check_contig(w=w, m=m, v=v, g=g)
    n = w.numel()
    out = torch.empty(n, dtype=torch.float32, device=w.device)
    _k_coi[_grid(n)](
        w, m, v, g, out, n,
        _f32(b1), _f32(1.0 - b1), _f32(b2), _f32(1.0 - b2),
        _recip32(bc1c), _recip32(bc2c), _f32(eps),
        BS=_BS, num_warps=_WARPS, enable_fp_fusion=False)
    return out


def _abs_out(x):
    n = x.numel()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    _k_abs[_grid(n)](x, out, n, BS=_BS, num_warps=_WARPS,
                       enable_fp_fusion=False)
    return out


def _formula_abs_g_k(w, m, v, g, b1, b2, eps, wd, bc1c, bc2c):
    _check_contig(g=g)
    return _abs_out(g)


def _formula_abs_m_k(w, m, v, g, b1, b2, eps, wd, bc1c, bc2c):
    _check_contig(m=m)
    return _abs_out(m)


_FORMULAS = {}
if triton is not None:
    _FORMULAS.update({
        "update": _formula_update_k,
        "coi": _formula_coi_k,
        "abs_g": _formula_abs_g_k,
        "abs_m": _formula_abs_m_k,
    })


def opt_step_inplace(q, m, v, w, lab, lr, wd, b1, b2, bc1c, bc2c, eps,
                     optimizer="adamw", inactive=None, lr_inactive=None):
    """The kernel twin of sga.replay._opt_step_inplace (fp32 CUDA,
    contiguous, in place): the AdamW recursion or the AdamS v basis
    (``optimizer='adams'`` — v rebuilt from the pre-commit m, the bc
    channels already pinned to 1.0 by the caller), plus the optional
    boundary dual-lr split (``inactive``: bool mask, True = the
    never-communicated tail applying at ``lr_inactive``)."""
    _check_contig(q=q, m=m, v=v, w=w)
    if lab is not None:
        _check_contig(lab=lab)
    if inactive is not None:
        _check_contig(inactive=inactive)
    if (inactive is None) != (lr_inactive is None):
        raise ValueError(
            "opt_step_inplace: inactive and lr_inactive come as a pair "
            f"(got inactive={inactive is not None}, "
            f"lr_inactive={lr_inactive!r})")
    n = q.numel()
    # torch.bool's byte-per-element buffer is not a pointer type a triton
    # load honors bit-exactly — reinterpret the SAME bytes as uint8
    # (one 0/1 byte per lane) for the kernel's step-rate select
    _k_adam[_grid(n)](
        q, m, v, w, lab if lab is not None else q,
        (inactive.view(torch.uint8)
         if inactive is not None else q), n,
        _f32(b1), _f32(1.0 - b1), _f32(b2), _f32(1.0 - b2),
        _recip32(bc1c), _recip32(bc2c), _f32(eps),
        _f32(1.0 - lr * wd), _f32(-lr), _f32(lr * lr),
        _f32(lr), _f32(lr_inactive if lr_inactive is not None else 0.0),
        _f32(wd),
        HAS_LAB=lab is not None, ADAMS=(optimizer == "adams"),
        SPLIT=inactive is not None, BS=_BS, num_warps=_WARPS,
        enable_fp_fusion=False)


def adam_step_inplace(q, m, v, w, lab, lr, wd, b1, b2, bc1c, bc2c, eps):
    """The kernel twin of sga.replay._adam_step_inplace (fp32 CUDA,
    contiguous, in place)."""
    opt_step_inplace(q, m, v, w, lab, lr, wd, b1, b2, bc1c, bc2c, eps)


def hist_top16(score):
    """(32769-bin int64 cpu histogram, nonfinite lane count) in one
    sweep — the histogram values and the guard verdict are exactly the
    torch chain's."""
    n = score.numel()
    hist = torch.zeros(32769, dtype=torch.int64, device=score.device)
    nf = torch.zeros(1, dtype=torch.int32, device=score.device)
    _k_hist[_grid(n)](score, hist, nf, n, BS=_BS, num_warps=_WARPS,
                        enable_fp_fusion=False)
    return hist.cpu(), int(nf.cpu().item())


def cand_extract(score, b):
    """The threshold bin's candidates on cpu in global index order —
    bitwise the same vector as torch's masked extraction."""
    n = score.numel()
    nblk = triton.cdiv(n, _BS)
    blk = torch.empty(nblk, dtype=torch.int32, device=score.device)
    _k_cand_count[_grid(n)](score, blk, n, b, BS=_BS, num_warps=_WARPS,
                              enable_fp_fusion=False)
    incl = torch.cumsum(blk, 0, dtype=torch.int64)
    total = int(incl[-1].item())
    cand = torch.empty(total, dtype=torch.float32, device=score.device)
    if total:
        _k_cand_fill[_grid(n)](
            score, cand, incl - blk, n, b, BS=_BS, num_warps=_WARPS,
            enable_fp_fusion=False)
    return cand.cpu()


def gt_eq_counts(score, thresh):
    """(gt, eq bool device masks, int64 cpu counts pair) in one sweep —
    the threshold is cast to fp32 exactly as torch's tensor-scalar
    compare casts its wrapped number."""
    n = score.numel()
    gt = torch.empty(n, dtype=torch.bool, device=score.device)
    eq = torch.empty(n, dtype=torch.bool, device=score.device)
    cnts = torch.zeros(2, dtype=torch.int32, device=score.device)
    _k_gt_eq[_grid(n)](score, gt, eq, cnts, n, _f32(thresh),
                         BS=_BS, num_warps=_WARPS, enable_fp_fusion=False)
    c = cnts.cpu()
    return gt, eq, torch.tensor([int(c[0]), int(c[1])],
                                dtype=torch.int64)
