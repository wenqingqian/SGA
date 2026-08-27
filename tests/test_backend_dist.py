#!/usr/bin/env python3
"""Distributed (2-rank CUDA) tests for the sparse allreduce backends.

Runs the real collectives through torchrun, mirroring the nccl-space
prototype's methodology:

- payload scaling under a real SUM allreduce: non-selected coordinates are
  BITWISE unchanged, selected ones equal (a + b) * factor;
- the two backends (masked_pipelined / indexed_sparse) produce bitwise-
  identical communicated coordinates (same element set, single fp32 add);
- an empty support is a clean no-op;
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

from sga.backends.indexed_sparse import IndexedSparseBackend
from sga.backends.masked_pipelined import MaskedPipelinedBackend
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

    # --- masked backend ---------------------------------------------------
    g_mask = local.clone()
    MaskedPipelinedBackend({"num_slices": 2}).allreduce_sparse(
        g_mask, support, group, factor, op, key=0)
    active = support.mask_u8 > 0
    assert torch.equal(g_mask[~active], local[~active]), "residual changed"
    assert torch.equal(g_mask[active], expected_selected[active]), \
        f"masked backend selected coords wrong (rank {rank})"

    # --- indexed backend --------------------------------------------------
    g_idx = local.clone()
    IndexedSparseBackend({}).allreduce_sparse(
        g_idx, support, group, factor, op, key=0)
    assert torch.equal(g_idx[~active], local[~active]), "residual changed"
    assert torch.equal(g_idx[active], expected_selected[active]), \
        f"indexed backend selected coords wrong (rank {rank})"

    # --- backends agree bitwise on the communicated coordinates -----------
    assert torch.equal(g_idx[active], g_mask[active])

    # --- empty support is a clean no-op ------------------------------------
    empty = BucketSupport(torch.zeros(n, dtype=torch.uint8, device=device))
    g_empty = local.clone()
    MaskedPipelinedBackend({"num_slices": 2}).allreduce_sparse(
        g_empty, empty, group, factor, op, key=1)
    IndexedSparseBackend({}).allreduce_sparse(
        g_empty, empty, group, factor, op, key=1)
    assert torch.equal(g_empty, local)
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
