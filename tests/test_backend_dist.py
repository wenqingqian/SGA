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
  (raises, no collective mismatch) on diverged ones.

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
from sga.engine import SGAEngineError
from sga.support import BucketSupport


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
    ret_mask = TorchPipelinedMaskBackend({"mask_slices": 2}).allreduce_sparse(
        g_mask, support, group, factor, op, key=0)
    active = support.mask_u8 > 0
    assert ret_mask is None, "writeback backend must return None"
    assert torch.equal(g_mask[~active], local[~active]), "residual changed"
    assert torch.equal(g_mask[active], expected_selected[active]), \
        f"masked backend selected coords wrong (rank {rank})"

    # --- indexed backend (compact emission) --------------------------------
    g_idx = local.clone()
    ret_idx = TorchPipelinedIndexBackend({}).allreduce_sparse(
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
    ret_wb = TorchPipelinedIndexBackend(
        {"emission": "writeback"}).allreduce_sparse(
            g_wb, support, group, factor, op, key=0)
    assert ret_wb is None, "writeback emission must return None"
    assert torch.equal(g_wb[~active], local[~active]), "residual changed"
    assert torch.equal(g_wb[active], expected_selected[active]), \
        f"writeback emission selected coords wrong (rank {rank})"
    assert torch.equal(g_wb[active], g_mask[active]), \
        "writeback emission must match the mask backend's buffer bitwise"
    assert torch.equal(
        g_wb, g_mask), "writeback emission must equal the masked backend"

    # --- raw-kernel backend (needs the sparse library; skip loudly) -------
    lib = os.environ.get("SGA_NCCL_SAR_LIB")
    if lib:
        from sga.backends.nccl_pipelined_index import NcclPipelinedIndexBackend

        be = NcclPipelinedIndexBackend({})
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
        be3 = NcclPipelinedIndexBackend({})
        be3.prepare(group, device=device)
        g_fast = local.clone()
        be3.allreduce_sparse(g_fast, support, group, 1.0, op, key=10)
        torch.cuda.synchronize()
        assert torch.equal(g_fast[~active], local[~active]), "residual changed"
        expected_fast = _vec(100) + _vec(101)  # (a + b) * 1.0: no scaling
        assert torch.equal(g_fast[active], expected_fast[active]), \
            f"nccl fast-path selected coords wrong (rank {rank})"
    else:
        if rank == 0:
            print("skip: nccl_pipelined_index arm (SGA_NCCL_SAR_LIB unset)")

    # --- empty support is a clean no-op ------------------------------------
    empty = BucketSupport(torch.zeros(n, dtype=torch.uint8, device=device))
    g_empty = local.clone()
    ret_m = TorchPipelinedMaskBackend({"mask_slices": 2}).allreduce_sparse(
        g_empty, empty, group, factor, op, key=1)
    ret_i = TorchPipelinedIndexBackend({}).allreduce_sparse(
        g_empty, empty, group, factor, op, key=1)
    assert ret_m is None
    # zero-selection is an EMPTY TENSOR on compact backends (never None —
    # that is the writeback signature)
    assert isinstance(ret_i, torch.Tensor) and ret_i.numel() == 0
    assert torch.equal(g_empty, local)
    # writeback emission: zero-selection is None (the writeback signature
    # the engine keys on) and the buffer stays untouched
    g_empty2 = local.clone()
    ret_wb_empty = TorchPipelinedIndexBackend(
        {"emission": "writeback"}).allreduce_sparse(
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
