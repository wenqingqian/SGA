#!/usr/bin/env python3
"""Replay's balanced-histogram selection emulation must be BIT-FAITHFUL
to the online engine's plugin selection.

Compares replay_fused.shard_select_balanced (flat score + per-param entries)
against plugins.selection._histogram.select_masks_histogram(mode='balanced',
global_allreduce=False) (the per-view form the online engine runs under its
replicated full-sync scores) on random score vectors with awkward view
geometries (odd sizes, tiny views, duplicate values to land inside crossing
bins) and several k. Pass criteria: threshold floats bit-equal (repr) AND
masks exactly equal.

A second battery pins the SHARDED form (single node, real gloo process
groups via mp.spawn): for worlds 2/3/8 over the non-8-aligned geometry the
per-shard masks reassemble (in shard_bounds order) to exactly the world-1
mask and every rank's threshold is bit-equal to it — the per-param
subsample phase, the catalog-wide n_total/n_sampled and the int64
histogram all_reduce reconstruct the world-1 select exactly (no world
factor). Container: PYTHONPATH=<0072-tree>/SGA python3 this file.
"""
import os
import struct
import sys
import tempfile

import torch
import torch.multiprocessing as mp

from sga import replay_fused
from sga.plugins.selection._histogram import (
    _HIST_MODES, _accumulate_histogram, _balanced_crossing_bin,
    _sampled_count, select_masks_histogram)
from sga.replay_fused import shard_select_balanced


class _Comms1:
    distributed = False
    world = 1
    global_rank = 0


def _f32_bits(x):
    return struct.pack("<f", float(x))


def _entries_of(sizes, lo=0, hi=None):
    """ShardParams-style per-param entries for the [lo, hi) slice of the
    catalog: "a" is the slice-local flat offset, "p_off" the param-relative
    slice start (the subsample phase input), "n" the slice length."""
    hi = sum(sizes) if hi is None else hi
    entries, off = [], 0
    for n in sizes:
        a, b = max(off, lo), min(off + n, hi)
        if b > a:
            entries.append({"a": a - lo, "p_off": a - off, "n": b - a})
        off += n
    return entries


def run_case(views, k, tag):
    thresh_on, masks_on = select_masks_histogram(
        views, k, None, mode="balanced", global_allreduce=False)
    flat = torch.cat(views)
    mask_rp, thresh_rp = shard_select_balanced(
        flat, k, _Comms1(), _entries_of([v.numel() for v in views]))
    on_flat = torch.cat([m.to(torch.bool) for m in masks_on])
    ok_t = _f32_bits(thresh_on) == _f32_bits(thresh_rp)
    ok_m = bool(torch.equal(on_flat, mask_rp))
    # over-selection ratio is printed for eyeball, not asserted (the
    # approximate tier keeps the whole crossing bin, two-sided by
    # construction — the realized count may also land under k)
    n = flat.numel()
    got = int(mask_rp.sum().item())
    status = "PASS" if (ok_t and ok_m) else "FAIL"
    print(f"[{status}] {tag}: k={k} thresh_on={thresh_on!r} "
          f"thresh_rp={thresh_rp!r} bits_eq={ok_t} masks_eq={ok_m} "
          f"selected={got}/{n} (+{100.0 * (got / k - 1):.3f}pp-ish vs k)")
    return ok_t and ok_m


def run_rank_scale_case(views, k):
    """Replicated online ranks and disjoint replay shards must select the
    same crossing bin with different explicit rank scaling."""
    shift, nbins, stride = _HIST_MODES["balanced"]
    hist = torch.zeros(nbins, dtype=torch.int64)
    for view in views:
        _accumulate_histogram(hist, view, shift, stride)
    n_total = sum(v.numel() for v in views)
    n_sampled = sum(_sampled_count(v.numel(), stride) for v in views)
    replay_bin = _balanced_crossing_bin(
        hist, n_total, n_sampled, k, replicas=1)
    bins = {world: _balanced_crossing_bin(
        hist * world, n_total, n_sampled, k, replicas=world)
        for world in (2, 3, 8)}
    ok = all(b == replay_bin for b in bins.values())
    print(f"[{'PASS' if ok else 'FAIL'}] rank-scale: replay={replay_bin} "
          f"replicated={bins}")
    return ok


def _shard_worker(rank, world, port, flat, sizes, bounds, k, out_dir):
    """One replay shard: select on ITS slice with the real Comms process
    group (gloo over localhost) and drop the mask piece + threshold to
    disk (mp.spawn workers return nothing)."""
    comms = replay_fused.Comms(rank, world, port)
    lo, hi = bounds[comms.global_rank]
    mask, thresh = shard_select_balanced(
        flat[lo:hi], k, comms, _entries_of(sizes, lo, hi))
    torch.save({"mask": mask.cpu(), "thresh": thresh},
               os.path.join(out_dir, f"shard{rank}.pt"))


def run_shard_parity(views, k, world, tag, bounds=None):
    """world-shard parity: the sharded balanced select must reproduce the
    world-1 select BITWISE — assembled mask (shard pieces in bounds order)
    and per-rank thresholds — and both must match the online plugin's own
    selection. ``bounds`` overrides shard_bounds (a hand-placed cut can
    land a tiny param on a phase-7 offset, whose chunk subsample is the
    empty slice the engine can produce on any unlucky boundary)."""
    flat = torch.cat(views)
    sizes = [v.numel() for v in views]
    n = flat.numel()
    thresh_on, masks_on = select_masks_histogram(
        views, k, None, mode="balanced", global_allreduce=False)
    on_flat = torch.cat([m.to(torch.bool) for m in masks_on])
    mask1, thresh1 = shard_select_balanced(
        flat, k, _Comms1(), _entries_of(sizes))
    if bounds is None:
        bounds = replay_fused.shard_bounds(n, world)
    with tempfile.TemporaryDirectory() as td:
        port = replay_fused._free_port()
        mp.spawn(_shard_worker,
                 args=(world, port, flat, sizes, bounds, k, td),
                 nprocs=world, join=True)
        parts = [torch.load(os.path.join(td, f"shard{r}.pt"),
                            weights_only=True) for r in range(world)]
    mask_w = torch.cat([p["mask"] for p in parts])
    ok_t = all(_f32_bits(p["thresh"]) == _f32_bits(thresh1)
               for p in parts) and _f32_bits(thresh1) == _f32_bits(thresh_on)
    ok_m = (bool(torch.equal(on_flat, mask1))
            and bool(torch.equal(mask1, mask_w)))
    status = "PASS" if (ok_t and ok_m) else "FAIL"
    print(f"[{status}] {tag}: world={world} k={k} thresh_bits_eq={ok_t} "
          f"masks_assembled_eq={ok_m} selected={int(mask_w.sum().item())}/{n}")
    return ok_t and ok_m


def _cuda_case():
    """cuda:0 score vector — the boundary-reselection device path (hist on
    cpu, mask returned on the input's device). Returns (ok, n_skipped)."""
    if not torch.cuda.is_available():
        print("[SKIP] cuda case (no device)")
        return True, 1
    g = torch.Generator().manual_seed(7)
    sizes = [40001, 7, 65536, 100009]
    views = [(torch.rand(n, generator=g) ** 2).float() for n in sizes]
    k = sum(sizes) // 100
    thresh_on, masks_on = select_masks_histogram(
        views, k, None, mode="balanced", global_allreduce=False)
    flat = torch.cat(views).cuda()
    mask_rp, thresh_rp = shard_select_balanced(
        flat, k, _Comms1(), _entries_of(sizes))
    on_flat = torch.cat([m.to(torch.bool) for m in masks_on]).cuda()
    ok = (_f32_bits(thresh_on) == _f32_bits(thresh_rp)
          and bool(torch.equal(on_flat, mask_rp))
          and mask_rp.device.type == "cuda")
    print(f"[{'PASS' if ok else 'FAIL'}] cuda: bits/mask/device ok={ok}")
    return ok, 0


def main():
    torch.manual_seed(20261004)
    g = torch.Generator().manual_seed(20261004)
    ok = True
    # awkward geometries: odd sizes / tiny view / pow2-heavy middle view
    sizes = [40001, 7, 65536, 100009]
    for trial, scale in enumerate((1e-3, 1.0, 37.5)):
        views = [(
            (torch.rand(n, generator=g) ** 2 * scale).float()
            # duplicates: force populated bins / ties inside the crossing bin
            if n != 65536 else
            torch.full((n,), 0.125 * scale, dtype=torch.float32))
            for n in sizes]
        n_total = sum(sizes)
        for k in (n_total // 100, n_total // 17, 4096, 3):
            ok &= run_case(views, k, f"trial{trial}")
    cuda_ok, n_skipped = _cuda_case()
    ok &= cuda_ok
    # sharded parity (single node, real gloo groups): the catalog is NOT
    # 8-aligned against the shard bounds (worlds 2/3/8 all cut params at
    # non-multiple offsets), so the per-param subsample phases engage;
    # the duplicate-heavy middle view keeps the crossing bin populated.
    # SGA_MASTER_ADDR would flip Comms to the multi-node rendezvous, which
    # the balanced select rejects — pin the single-node form.
    os.environ.pop("SGA_MASTER_ADDR", None)
    views_p = [((torch.rand(n, generator=g) ** 2).float()
                if n != 65536 else
                torch.full((n,), 0.125, dtype=torch.float32))
               for n in sizes]
    n_p = sum(sizes)
    ok &= run_rank_scale_case(views_p, n_p // 100)
    for world in (2, 3, 8):
        for k in (n_p // 100, n_p // 17):
            ok &= run_shard_parity(views_p, k, world, "shard-parity")
    # hand-placed cut through the 7-element param at param-relative
    # offset 1: the right piece runs phase 7 over 6 elements — an EMPTY
    # chunk subsample (bincount of nothing must stay a zero histogram,
    # consistent with its n_sampled count of 0)
    cut = 40002
    ok &= run_shard_parity(views_p, n_p // 100, 2, "shard-parity-empty",
                           bounds=[(0, cut), (cut, n_p)])
    print(("ALL PASS" if ok else "FAILURES PRESENT")
          + (f" ({n_skipped} skipped)" if n_skipped else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
