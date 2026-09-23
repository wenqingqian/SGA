#!/usr/bin/env python3
"""Distributed (2-rank CUDA) tests for the sparse allreduce backends.

Runs the real collectives through torchrun, mirroring the nccl-space
prototype's methodology:

- payload scaling under a real SUM allreduce: non-selected coordinates are
  BITWISE unchanged, selected ones equal (a + b) * factor;
- torch_pipelined_mask (writeback emission): the communicated values are
  scattered into the full buffer and the call returns None;
- torch_pipelined_index (compact emission): the call returns the
  values view — content bitwise-equal to the mask backend's written-back
  values — and the full buffer's selected coords are ZERO (the fused pack
  cleared them); non-selected coords keep the residual;
- an empty support is a clean no-op (empty Tensor return on the compact
  backend — a Tensor, not None, so the engine can tell zero-selection
  apart from a writeback backend);
- the engine's layer-1 count check passes on identical supports and REFUSES
  (raises, no collective mismatch) on diverged ones;
- the rank-split histogram-allreduce gate: both ranks refuse
  loudly instead of splitting collective participation.

Run: python tests/test_backend_dist.py   (re-execs itself under torchrun)
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

SGA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SGA_ROOT)

from sga.backends.torch_pipelined_index import TorchPipelinedIndexBackend
from sga.backends.torch_pipelined_mask import TorchPipelinedMaskBackend
from sga.config import set_args
from sga.config.validation import sga_defaults
from sga.engine import SGAEngineError
from sga.support import BucketSupport


def _seed(**over):
    """Seed the sga singleton for a zero-arg backend build
    (backends read their knobs off sga.config.get_args())."""
    set_args(SimpleNamespace(**{**sga_defaults(), **over}))


def _engine_like():
    """Minimal stand-in exposing only what _check_counts touches."""
    return SimpleNamespace()


def main():
    rank = dist.get_rank()
    torch.manual_seed(1234 + rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    group = dist.group.WORLD
    world = dist.get_world_size()
    assert world == 2

    n = 8192
    gen = torch.Generator(device="cpu").manual_seed(7)
    mask_cpu = (torch.rand(n, generator=gen) < 0.25).to(torch.uint8)
    support = BucketSupport(mask_cpu.to(device))
    factor = 0.125
    op = dist.ReduceOp.SUM

    def _vec(seed):
        """Deterministic vector, identical on both ranks (CPU RNG, then move)."""
        return torch.randn(n, generator=torch.Generator().manual_seed(seed)).to(device)

    local = _vec(100 + rank)
    expected_selected = (_vec(100) + _vec(101)) * factor  # (a + b) * factor

    # --- masked backend (writeback emission) -------------------------------
    g_mask = local.clone()
    _seed()
    ret_mask = TorchPipelinedMaskBackend().allreduce_sparse(
        g_mask, support, group, factor, op, key=0)
    active = support.mask_u8 > 0
    assert ret_mask is None, "writeback backend must return None"
    assert torch.equal(g_mask[~active], local[~active]), "residual changed"
    assert torch.equal(g_mask[active], expected_selected[active]), \
        f"masked backend selected coords wrong (rank {rank})"

    # --- indexed backend (compact emission) --------------------------------
    g_idx = local.clone()
    _seed()
    ret_idx = TorchPipelinedIndexBackend().allreduce_sparse(
        g_idx, support, group, factor, op, key=0)
    assert isinstance(ret_idx, torch.Tensor), (
        "compact backend must return the values view (a Tensor)"
    )
    assert torch.equal(g_idx[~active], local[~active]), "residual changed"
    assert torch.equal(g_idx[active], torch.zeros_like(g_idx[active])), \
        "compact backend must leave the selected coords ZERO (pack cleared)"
    # idx order is ascending, boolean indexing is ascending: elementwise.
    assert torch.equal(ret_idx, expected_selected[active]), \
        f"indexed backend values wrong (rank {rank})"

    # --- the two emissions agree bitwise on the communicated values -------
    assert torch.equal(ret_idx, g_mask[active])

    # --- indexed backend, emission=writeback (the restored historical
    # choreography): pure-read gather + scatter-back, None return — the
    # written-back buffer must equal the mask backend's bitwise (same
    # payload elements in ascending order, same per-element SUMs) ---
    g_wb = local.clone()
    _seed(sga_backend_emission="writeback")
    ret_wb = TorchPipelinedIndexBackend().allreduce_sparse(
            g_wb, support, group, factor, op, key=0)
    assert ret_wb is None, "writeback emission must return None"
    assert torch.equal(g_wb[~active], local[~active]), "residual changed"
    assert torch.equal(g_wb[active], expected_selected[active]), \
        f"writeback emission selected coords wrong (rank {rank})"
    assert torch.equal(g_wb[active], g_mask[active]), \
        "writeback emission must match the mask backend's buffer bitwise"
    assert torch.equal(
        g_wb, g_mask), "writeback emission must equal the masked backend"

    # --- block-granular arm (sga_selection_block_size > 1): tpi writeback
    # on a block-structured support must land the SAME end state as the
    # mask backend on the same mask, bitwise — the payload holds the same
    # elements in ascending order and the zero-padded inactive lanes are
    # allreduce-neutral, so every mask-set coordinate gets the same SUM.
    # The mask exercises a mask-partial listed block (drift) and both
    # span-tail blocks.
    x = 16
    spans = [(0, 8100), (8100, 8192)]
    genb = torch.Generator(device="cpu").manual_seed(11)
    bmask = torch.zeros(n, dtype=torch.uint8)
    bview = bmask[:8096].view(506, x)      # span 0's full blocks
    bview[(torch.rand(506, generator=genb) < 0.25)
          .nonzero(as_tuple=True)[0]] = 1
    bview[3, 1::2] = 0                     # a mask-partial listed block
    bview[3, 0] = 1                        # ...still listed (ANY rule)
    bmask[8097] = 1                        # span-0 tail block [8096, 8100)
    bmask[8100 + 3 * x:8100 + 4 * x] = 1   # a full block of span 1
    bmask[8180] = 1                        # span-1 tail block [8180, 8192)
    bsupport = BucketSupport(bmask.to(device), block_size=x,
                             block_spans=spans)
    g_blk = local.clone()
    _seed(sga_backend_emission="writeback")
    ret_blk = TorchPipelinedIndexBackend().allreduce_sparse(
            g_blk, bsupport, group, factor, op, key=2)
    assert ret_blk is None, "block writeback must return None"
    g_bmask = local.clone()
    _seed()
    TorchPipelinedMaskBackend().allreduce_sparse(
        g_bmask, BucketSupport(bmask.to(device)), group, factor, op, key=2)
    bact = bmask.to(device) > 0
    assert torch.equal(g_blk[~bact], local[~bact]), "block residual changed"
    assert torch.equal(g_blk[bact], expected_selected[bact]), \
        f"block writeback selected coords wrong (rank {rank})"
    assert torch.equal(g_blk, g_bmask), \
        "block writeback must equal the mask backend's buffer bitwise"

    # block + empty support: the writeback signature (None, untouched)
    bempty = BucketSupport(torch.zeros(n, dtype=torch.uint8, device=device),
                           block_size=x, block_spans=[(0, n)])
    g_be = local.clone()
    _seed(sga_backend_emission="writeback")
    assert TorchPipelinedIndexBackend().allreduce_sparse(
            g_be, bempty, group, factor, op, key=3) is None
    assert torch.equal(g_be, local)

    # block + tpi compact: refused loudly (direct-construction belt;
    # config validation is the front gate, pinned in test_config_flat)
    try:
        _seed(sga_backend_emission="compact")
        TorchPipelinedIndexBackend().allreduce_sparse(
            local.clone(), bsupport, group, factor, op, key=4)
        raise AssertionError("compact emission accepted a block support")
    except ValueError as e:
        assert "writeback" in str(e), e
    # nccl bb outside the descriptor x domain [4, 64]: refused loudly
    # (the kernel ABI floor is blockShift >= 2). Inside the domain a
    # block support is a LEGITIMATE bb config — the descriptor arm
    # below exercises it end to end.
    from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend
    nsup = BucketSupport(torch.ones(n, dtype=torch.uint8, device=device),
                         block_size=2, block_spans=[(0, n)])
    try:
        _seed()
        NcclPipelinedIndexBackend().allreduce_sparse(
            local.clone(), nsup, group, factor, op, key=4)
        raise AssertionError("nccl accepted an out-of-domain block size")
    except ValueError as e:
        assert "[4, 64]" in str(e), e

    # --- raw-kernel backend (needs the sparse library; skip loudly) -------
    lib = os.environ.get("SGA_NCCL_SAR_LIB")
    if lib:
        from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend

        _seed()
        be = NcclPipelinedIndexBackend()
        be.prepare(group, device=device)
        be.prepare(group, device=device)  # idempotent: one comm per group
        assert len(be._comms) == 1
        g_nccl = local.clone()
        be.allreduce_sparse(g_nccl, support, group, factor, op, key=9)
        torch.cuda.synchronize()
        assert torch.equal(g_nccl[~active], local[~active]), "residual changed"
        # pow2 factor + one add per coordinate: bitwise agreement with the
        # torch backends (validated on A100 singlesocket + cross-node IB)
        assert torch.equal(g_nccl[active], expected_selected[active]), \
            f"nccl backend selected coords wrong (rank {rank})"
        assert torch.equal(g_nccl[active], ret_idx)

        # a second backend instance exercising the factor-1.0 fast path:
        # no pre-scale, plain SUM over 2 ranks
        _seed()
        be3 = NcclPipelinedIndexBackend()
        be3.prepare(group, device=device)
        g_fast = local.clone()
        be3.allreduce_sparse(g_fast, support, group, 1.0, op, key=10)
        torch.cuda.synchronize()
        assert torch.equal(g_fast[~active], local[~active]), "residual changed"
        expected_fast = _vec(100) + _vec(101)  # (a + b) * 1.0: no scaling
        assert torch.equal(g_fast[active], expected_fast[active]), \
            f"nccl fast-path selected coords wrong (rank {rank})"

        # --- bb descriptor frozen grid (C2): the same whole-run
        # selected set through FOUR transports — mask backend, torch
        # element writeback (eb form), torch block heads (bb), nccl
        # packed descriptors (bb) — must land identical end states,
        # bitwise, on every cell of the frozen grid: x in
        # {4,8,16,32,64} (x=4 is the kernel's ABI floor — the in-domain
        # smallest, the live control for the out-of-domain refusals)
        # x span-tail on/off x factor in {1.0, 0.125} (a non-1 power of
        # two keeps the pre-scale exact in fp32).
        #
        # Why bitwise agreement holds despite different compact domains
        # (bb packs k_b*x lanes with tail runs riding short blocks; the
        # element paths pack k_e indices; the mask backend communicates
        # nothing compact at all): at 2 ranks every selected coordinate
        # sees exactly one a-term and one b-term — fp32 addition is
        # commutative and the pow2 factor makes each scaled term exact —
        # and every transport writes ONLY selected coordinates
        # (descriptor runs are whole-run by the contract belt; the torch
        # block lanes are mask-guarded; the element paths move exactly
        # the set bits). Beyond 2 ranks this is an empirical property,
        # not an identity (reduction trees enter) — the suite's scope
        # statement.
        _seed(sga_backend_layout="bb")
        be_bb = NcclPipelinedIndexBackend()
        assert be_bb.layout == "bb"
        be_bb.prepare(group, device=device)  # block-form warmup (C6)

        def _grid_case(x, with_tail, fac):
            a = (n // 3 // x) * x
            if with_tail:
                a += x - 1                 # span-0 tail-short end
            b = a if a % x == 0 else (a + x - 1) // x * x
            spans = [(0, a), (b, n)]       # aligned starts; gap = dead space
            gsel = torch.Generator(device="cpu").manual_seed(
                4242 + x + 8 * int(with_tail) + int(fac != 1.0))
            m = torch.zeros(n, dtype=torch.uint8)
            nb, rem = divmod(a, x)
            keep = torch.rand(nb, generator=gsel) < 0.25
            for i in keep.nonzero(as_tuple=True)[0].tolist():
                m[i * x:(i + 1) * x] = 1
            if with_tail and rem:
                m[nb * x:a] = 1            # whole tail (prefix contract)
            m[b + x:n - x] = 1             # span 1: all but edge blocks
            dact = m.to(device) > 0
            dsup = BucketSupport(m.to(device), block_size=x,
                                 block_spans=spans)
            esup = BucketSupport(m.to(device))  # element form, same mask
            exp = (_vec(100) + _vec(101)) * fac
            _seed()
            g_mask = local.clone()
            TorchPipelinedMaskBackend().allreduce_sparse(
                g_mask, esup, group, fac, op, key=20)
            _seed(sga_backend_emission="writeback")
            g_elem = local.clone()
            TorchPipelinedIndexBackend().allreduce_sparse(
                g_elem, esup, group, fac, op, key=21)
            g_tblk = local.clone()
            TorchPipelinedIndexBackend().allreduce_sparse(
                g_tblk, dsup, group, fac, op, key=22)
            g_desc = local.clone()
            be_bb.allreduce_sparse(g_desc, dsup, group, fac, op, key=23)
            torch.cuda.synchronize()
            for tag, g in (("mask", g_mask), ("element", g_elem),
                           ("torch-block", g_tblk), ("nccl-desc", g_desc)):
                assert torch.equal(g[~dact], local[~dact]), \
                    f"{tag} residual changed x={x} t={with_tail} f={fac}"
                assert torch.equal(g[dact], exp[dact]), \
                    f"{tag} wrong x={x} t={with_tail} f={fac} r={rank}"
                assert torch.equal(g, g_mask), \
                    f"{tag} diverged from mask x={x} t={with_tail} f={fac}"
            # one auto-layout spot check through the plain backend: the
            # dispatch belt derives bb from the support's granularity
            if x == 16 and with_tail:
                g_auto = local.clone()
                be.allreduce_sparse(g_auto, dsup, group, fac, op, key=24)
                torch.cuda.synchronize()
                assert torch.equal(g_auto, g_mask), \
                    "auto-belt dispatch diverged"

        for x in (4, 8, 16, 32, 64):
            for with_tail in (True, False):
                for fac in (1.0, 0.125):
                    _grid_case(x, with_tail, fac)
        if rank == 0:
            print("ok: bb frozen grid (5x2x2) four-transport bitwise, C2")

        # --- old-library belt (C7): a block-incapable library under a
        # bb config must refuse at LOAD time — before any raw comm is
        # created (no half-built communicator may survive a config the
        # library cannot serve). v5 exports the element API only and is
        # staged as a SIBLING of the pinned generation (lib is
        # <libsroot>/<gen>/libnccl-mask.so — two dirname levels reach
        # <libsroot>).
        v5 = os.path.join(os.path.dirname(os.path.dirname(lib)),
                          "v5", "libnccl-mask.so")
        if os.path.exists(v5):
            saved = os.environ.get("SGA_NCCL_SAR_LIB")
            os.environ["SGA_NCCL_SAR_LIB"] = v5
            try:
                be_old = NcclPipelinedIndexBackend()
                be_old.layout = "bb"  # production ctor reads it concrete
                try:
                    be_old.prepare(group, device=device)
                    raise AssertionError("bb config accepted a library "
                                         "without any block API")
                except RuntimeError as e:
                    assert "refusing before comm creation" in str(e), e
                assert not be_old._comms, "a comm survived the refusal"
            finally:
                if saved is None:
                    del os.environ["SGA_NCCL_SAR_LIB"]
                else:
                    os.environ["SGA_NCCL_SAR_LIB"] = saved
        else:
            if rank == 0:
                print(f"skip: C7 old-library belt (v5 lib not staged "
                      f"at {v5})")
    else:
        if rank == 0:
            print("skip: nccl_pipelined_index arm (SGA_NCCL_SAR_LIB unset)")

    # --- empty support is a clean no-op ------------------------------------
    empty = BucketSupport(torch.zeros(n, dtype=torch.uint8, device=device))
    g_empty = local.clone()
    _seed()
    ret_m = TorchPipelinedMaskBackend().allreduce_sparse(
        g_empty, empty, group, factor, op, key=1)
    ret_i = TorchPipelinedIndexBackend().allreduce_sparse(
        g_empty, empty, group, factor, op, key=1)
    assert ret_m is None
    # zero-selection is an EMPTY TENSOR on compact backends (never None —
    # that is the writeback signature)
    assert isinstance(ret_i, torch.Tensor) and ret_i.numel() == 0
    assert torch.equal(g_empty, local)
    # writeback emission: zero-selection is None (the writeback signature
    # the engine keys on) and the buffer stays untouched
    g_empty2 = local.clone()
    _seed(sga_backend_emission="writeback")
    ret_wb_empty = TorchPipelinedIndexBackend().allreduce_sparse(
            g_empty2, empty, group, factor, op, key=1)
    assert ret_wb_empty is None
    assert torch.equal(g_empty2, local)
    dist.barrier()

    # --- layer-1 count check: pass on identical supports -------------------
    from sga.engine import SGAEngine

    eng = object.__new__(SGAEngine)
    eng._dp_cp_group = lambda: group
    eng._bucket_states = [SimpleNamespace(grad_data=local)]
    eng._active_support = SupportLike(support)
    eng._check_counts()  # identical across ranks -> must pass

    # --- layer-1 count check: refuse on diverged supports ------------------
    diverged = mask_cpu.clone()
    flip = 100 + 50 * rank  # different flip count per rank -> different k
    diverged[:flip] = 1 - diverged[:flip]
    eng._active_support = SupportLike(BucketSupport(diverged.to(device)))
    try:
        eng._check_counts()
        raise AssertionError("diverged support counts were not refused")
    except SGAEngineError:
        pass

    # --- rank-split histogram allreduce gate: loud symmetric abort -------
    # One rank flag-on and one flag-off would split collective
    # participation at the first reselection (one enters the histogram
    # SUM, the other advances) — an undiagnosable hang. The one-time
    # packed MIN/MAX probe in TopkSelection.select must refuse on BOTH
    # ranks, naming the flag.
    from sga.plugins.selection.topk import TopkSelection

    _seed(sga_selection_method="histogram",
          sga_selection_ratio_percent=30.0,
          sga_selection_topk_histogram_add_global_allreduce=bool(rank == 0))
    split = TopkSelection()
    try:
        split.select(SimpleNamespace(
            grad_views=[torch.ones(64, device=device)], comm_group=group))
        raise AssertionError("rank-split gate was not refused")
    except ValueError as e:
        assert "sga_selection_topk_histogram_add_global_allreduce" in str(e), e

    # Agreement restored (both off, the default): the probe passes once and
    # the select itself runs the local-histogram path — no collective, no
    # hang; the barrier proves both ranks reached the end.
    _seed(sga_selection_method="histogram", sga_selection_ratio_percent=30.0)
    agree = TopkSelection()
    masks = agree.select(SimpleNamespace(
        grad_views=[torch.linspace(-1, 1, 1024, device=device)],
        comm_group=group))
    assert len(masks) == 1 and masks[0].numel() == 1024
    dist.barrier()

    # --- backend_timer on: identical results + records ------
    # The instrument must not change what the backend computes
    # (bitwise on/off; records carry the engine iter stamp).
    import json
    import tempfile

    from sga import backend_timer

    tmp = tempfile.mkdtemp(prefix="sga_timer_dist_")
    backend_timer.configure(tmp, rank=rank)
    backend_timer.set_iter(1)
    g_tim = local.clone()
    _seed()
    ret_tim = TorchPipelinedIndexBackend().allreduce_sparse(
        g_tim, support, group, factor, op, key=0)
    assert torch.equal(g_tim, g_idx), f"timer-on buffer differs (rank {rank})"
    assert torch.equal(ret_tim, ret_idx), f"timer-on values differ (rank {rank})"
    rec_path = os.path.join(tmp, f"backend_timer_rank{rank}.jsonl")
    with open(rec_path) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    trecs = [r for r in recs if r["backend"] == "torch_pipelined_index"]
    assert trecs and all(r["phase"] == "sparse" for r in trecs), trecs
    assert all(r["iter"] == 1 and r["bucket"] == 0 for r in trecs), trecs
    assert all(r["wall_ms"] >= 0.0 for r in trecs), trecs
    assert not any(r.get("error") for r in trecs), trecs

    # --- nccl arm of the timer-on section (needs the sparse library) ---
    # Same A/B identity contract for both APIs plus a magnitude
    # guard: the timed region now chains the kernel back onto the
    # calling stream INSIDE the bracket, so the sparse record blocks on
    # the comm kernel — an enqueue-only region would record sub-noise
    # µs instead.
    if lib:
        _seed()
        be_t = NcclPipelinedIndexBackend()
        be_t.prepare(group, device=device)
        g_nccl_tim = local.clone()
        be_t.allreduce_sparse(g_nccl_tim, support, group, factor, op,
                              key=30)
        torch.cuda.synchronize()
        # bitwise on/off identity (element API): g_nccl is the same
        # allreduce from the timer-off arm above
        assert torch.equal(g_nccl_tim, g_nccl), \
            f"timer-on nccl element result differs (rank {rank})"

        # block API on a dedicated bb-SAFE support: aligned span, whole
        # selected runs only (the descriptor contract's whole-run belt —
        # bsupport/bmask above carry mask-partial listed blocks and a
        # mid-block span start, torch-block-only shapes). At 2 ranks
        # with the pow2 factor the ground truth is bitwise
        # expected_selected — the same anchor the frozen grid asserts.
        xt = 16
        tmask = torch.zeros(n, dtype=torch.uint8)
        tgen = torch.Generator().manual_seed(99)
        for i in (torch.rand(n // xt, generator=tgen) < 0.25) \
                .nonzero(as_tuple=True)[0].tolist():
            tmask[i * xt:(i + 1) * xt] = 1
        tsup = BucketSupport(tmask.to(device), block_size=xt,
                             block_spans=[(0, n)])
        tact = tmask.to(device) > 0
        _seed(sga_backend_layout="bb")
        be_tblk = NcclPipelinedIndexBackend()
        be_tblk.prepare(group, device=device)
        g_blk_tim = local.clone()
        be_tblk.allreduce_sparse(g_blk_tim, tsup, group, factor, op,
                                 key=31)
        torch.cuda.synchronize()
        assert torch.equal(g_blk_tim[~tact], local[~tact]), \
            f"timer-on nccl block residual changed (rank {rank})"
        assert torch.equal(g_blk_tim[tact], expected_selected[tact]), \
            f"timer-on nccl block selected coords wrong (rank {rank})"

        with open(rec_path) as f:
            nrecs = [json.loads(x) for x in f if x.strip()]
        nrecs = [r for r in nrecs
                 if r["backend"] == "nccl_pipelined_index"]
        assert all(r["iter"] == 1 for r in nrecs), nrecs
        assert not any(r.get("error") for r in nrecs), nrecs
        by_key = {(r["api"], r["phase"]): r for r in nrecs}
        assert set(by_key) >= {("index", "prescale"), ("index", "sparse"),
                               ("block", "prescale"),
                               ("block", "sparse")}, sorted(by_key)
        assert "k" in by_key[("index", "sparse")], nrecs       # elements
        assert "k_blocks" in by_key[("block", "sparse")], nrecs
        assert "k" not in by_key[("block", "sparse")], nrecs
        # magnitude guard: sparse must clear the sub-noise floor or
        # cover the pre-scale segment.
        for api in ("index", "block"):
            sp = by_key[(api, "sparse")]["wall_ms"]
            ps = by_key[(api, "prescale")]["wall_ms"]
            assert sp >= 0.05 or sp >= ps, \
                f"nccl {api} sparse {sp}ms reads as enqueue noise " \
                f"(prescale {ps}ms, rank {rank})"
        if rank == 0:
            print("ok: nccl timer-on records (element+block): " + ", ".join(
                f"{api}/{ph}={by_key[(api, ph)]['wall_ms']:.3f}ms"
                for api in ("index", "block")
                for ph in ("prescale", "sparse")))
    backend_timer.reset()  # back to off (closes the record file)
    dist.barrier()

    if rank == 0:
        print("test_backend_dist: ALL PASS (2-rank CUDA)")


class SupportLike:
    def __init__(self, *buckets):
        self.buckets = buckets


if __name__ == "__main__":
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        os.execvp("torchrun", [
            "torchrun", "--nproc_per_node=2",
            "--master_port=29601", os.path.abspath(__file__),
        ])
    dist.init_process_group(backend="nccl")
    try:
        main()
    finally:
        dist.destroy_process_group()
