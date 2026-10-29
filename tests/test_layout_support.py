# Copyright (c) 2026, SGA framework contributors.
"""Layout-taxonomy unit tests: the block_desc packing contract.

The packed descriptor form (``head | tail_len`` in the low log2(x) bits)
is the nccl transport's wire format, so its derivation from the
canonical mask is pinned three ways here:

- oracle parity: an INDEPENDENT first-principles packer (build the
  block list per span, encode head + valid length) must agree with
  ``block_desc_from_mask`` bitwise;
- roundtrip identity: decoding the descriptors (aligned head + tail
  length) must reproduce exactly the mask's selected set;
- cross-repo parity: when the sibling nccl-sar checkout is present, the
  fork's own ``build_descriptors`` bench reference must agree bitwise
  (the two packers were written against the same descriptor contract
  from opposite sides — engine vs kernel test harness).

Negatives pin the contract belts (non-pow2 block size, misaligned
span start, partially selected run) and the property guards (element-
granular support, missing spans), plus the engine's attach-time
span-alignment gate for descriptor-form backends and the layout
resolution surface.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_engine_unit import _FakeChunk, _seed  # noqa: E402

from sga.engine import SGAEngine, SGAEngineError  # noqa: E402
from sga.plugins import resolve_layout  # noqa: E402
from sga.support import (  # noqa: E402
    BucketSupport,
    block_desc_from_mask,
    block_idx_from_mask,
)


def _oracle_pack(mask_u8, spans, x):
    """First-principles reference: tile each span from its start, list
    the all-set blocks (the contract geometry), encode head | tail_len
    (0 for complete blocks). Deliberately written per-block in a loop —
    a different derivation shape than the vectorized SGA packer."""
    descs = []
    prev_end = 0
    for start, end in spans:
        assert start >= prev_end
        prev_end = end
        n = end - start
        nb, rem = divmod(n, x)
        for i in range(nb):
            if bool(mask_u8[start + i * x:start + (i + 1) * x].all()):
                descs.append(start + i * x)  # complete: tail bits 0
        if rem and bool(mask_u8[start + nb * x:end].all()):
            descs.append((start + nb * x) | rem)  # tail-short: | valid
    return torch.tensor(descs, dtype=torch.int32, device=mask_u8.device)


def _decode_cover(desc, n, x):
    """Roundtrip decode: each descriptor heads a run of tail_len (or x
    when the low bits are zero) elements; the union must equal the
    mask's selected set."""
    covered = torch.zeros(n, dtype=torch.bool)
    for d in desc.tolist():
        head = d & ~(x - 1)
        run = (d & (x - 1)) or x
        covered[head:head + run] = True
    return covered


def _aligned_spans(n, x, with_tail):
    """Two ascending spans with x-aligned starts. A tail-short span-0
    end forces a dead-space gap up to the next aligned start (contiguous
    spans would start span 1 off the grid — the unencodable geometry the
    packer's belt refuses); a grid-aligned end allows a contiguous split.
    Mirrors the production layout (params padded so span starts land on
    the block grid; the gap is inter-param dead space, never selected).
    """
    a = (n // 3 // x) * x  # span-0 end on the block grid
    if with_tail and n - a > 2 * x:
        a += x - 1  # tail-short span-0 end (not x-divisible)
    if a % x == 0:
        b = a  # contiguous aligned split
    else:
        b = ((a + x - 1) // x) * x  # gap up to the next aligned start
    return [(0, a), (b, n)]


def _contract_mask(n, spans, x, keep=0.3, tail_on=True, seed=0):
    """A whole-run selection over aligned spans (the representable
    geometry): keep fraction of each span's full blocks, plus each span
    tail block when tail_on."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(n, dtype=torch.uint8)
    for start, end in spans:
        size = end - start
        nb, rem = divmod(size, x)
        keep_rows = torch.rand(nb, generator=g) < keep
        for i in torch.nonzero(keep_rows).flatten().tolist():
            mask[start + i * x:start + (i + 1) * x] = 1
        if tail_on and rem:
            mask[start + nb * x:end] = 1  # whole tail (prefix contract)
    return mask


def test_pack_oracle_and_roundtrip_grid():
    """Positive grid: x over the full contract domain {4,8,16,32,64}
    (4 = the kernel ABI floor) × span tail on/off — oracle parity,
    roundtrip identity, and sort/alignment invariants on every cell."""
    n = 4096
    for x in (4, 8, 16, 32, 64):
        for with_tail in (True, False):
            spans = _aligned_spans(n, x, with_tail)
            if spans[-1][1] - spans[-1][0] < x:
                continue  # degenerate second span at big x
            mask = _contract_mask(n, spans, x, tail_on=with_tail,
                                  seed=x + int(with_tail))
            desc = block_desc_from_mask(mask, spans, x)
            assert torch.equal(desc, _oracle_pack(mask, spans, x)), \
                f"oracle mismatch x={x} tail={with_tail}"
            assert torch.equal(_decode_cover(desc, n, x), mask > 0), \
                f"roundtrip mismatch x={x} tail={with_tail}"
            # invariants: ascending, x-aligned heads (the low log2(x)
            # bits are pure tail encoding)
            d64 = desc.to(torch.int64)
            assert bool((d64[1:] > d64[:-1]).all()), "not ascending"
            assert not bool(((d64 & ~(x - 1)) % x).any()), \
                "head not x-aligned"
    print("ok: block_desc oracle parity + roundtrip over the x grid")


def test_pack_parity_with_bench_reference():
    """Cross-repo parity vs the nccl-sar fork's build_descriptors (the
    kernel test harness's packer, written from the kernel side). Skips
    loudly when the sibling checkout (or its staged lib set) is absent —
    the bench module CDLLs the nomask baseline at import, so it needs a
    library directory, not just the source."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench = os.path.join(root, "..", "nccl-sar", "tests", "mask_utils.py")
    if not os.path.exists(bench):
        print("skip: bench parity (no sibling nccl-sar checkout)")
        return
    lib_roots = [os.environ.get("NCCL_LIBS_ROOT"),
                 os.path.join(root, "..", "nccl-sar", "libs")]
    lib_root = next((p for p in lib_roots
                     if p and os.path.exists(
                         os.path.join(p, "nomask", "libnccl-nomask.so"))),
                    None)
    if lib_root is None:
        print("skip: bench parity (no staged nccl lib set for its import)")
        return
    saved = os.environ.get("NCCL_LIBS_ROOT")
    os.environ["NCCL_LIBS_ROOT"] = lib_root
    try:
        spec = importlib.util.spec_from_file_location(
            "bench_mask_utils", bench)
        bm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bm)
    finally:
        if saved is None:
            del os.environ["NCCL_LIBS_ROOT"]
        else:
            os.environ["NCCL_LIBS_ROOT"] = saved
    n = 8192
    for x in (4, 8, 16, 32, 64):
        spans = _aligned_spans(n, x, with_tail=True)
        if spans[-1][1] - spans[-1][0] < x:
            continue
        mask = _contract_mask(n, spans, x, seed=100 + x)
        ours = block_desc_from_mask(mask, spans, x)
        heads = block_idx_from_mask(mask, spans, x)
        theirs = bm.build_descriptors(heads, mask, x, spans)
        assert torch.equal(ours.cpu(), theirs.cpu()), f"bench parity x={x}"
    print("ok: block_desc parity with the nccl-sar bench reference")


def test_pack_negative_misaligned_span():
    """A span START off the x grid is unencodable (the head's low bits
    carry the tail length) — the packer's belt must refuse it."""
    x = 8
    spans = [(0, 12), (12, 32)]  # span 1 starts at 12: 12 % 8 != 0
    mask = torch.zeros(32, dtype=torch.uint8)
    mask[:8] = 1      # span 0's aligned head (would be fine alone)
    mask[12:20] = 1   # span 1's first block heads at 12 — unaligned
    try:
        block_desc_from_mask(mask, spans, x)
        raise AssertionError("misaligned span start packed silently")
    except AssertionError as e:
        if "block_desc" not in str(e):
            raise  # the packer's belt, not our own control-flow raise


def test_pack_negative_partial_block():
    """A partially selected block breaks the whole-run contract (the
    descriptor transfers complete runs — unselected elements would ride
    the wire): the belt must refuse it."""
    x = 8
    spans = [(0, 64)]
    mask = torch.zeros(64, dtype=torch.uint8)
    mask[0:5] = 1  # 5 of 8 lanes: listed by the ANY rule, not whole
    try:
        block_desc_from_mask(mask, spans, x)
        raise AssertionError("partially selected block packed silently")
    except AssertionError as e:
        if "partially-selected" not in str(e):
            raise


def test_pack_negative_non_pow2_x():
    """x outside the pow2 >= 4 ABI domain (e.g. 6) is refused up front —
    the packing belts assume pow2 geometry (the alignment mask and the
    log2(x) tail bits are meaningless otherwise), so a non-pow2 x must
    not reach them and masquerade as a misalignment."""
    spans = [(0, 64)]
    mask = torch.zeros(64, dtype=torch.uint8)
    mask[:6] = 1  # a whole 6-block: even the honest geometry must refuse
    try:
        block_desc_from_mask(mask, spans, 6)
        raise AssertionError("non-pow2 block size packed silently")
    except AssertionError as e:
        if "power of two" not in str(e):
            raise  # the packer's ABI belt, not our own control-flow raise


def test_block_desc_property_guards():
    mask = torch.zeros(64, dtype=torch.uint8)
    mask[:8] = 1
    sup = BucketSupport(mask, block_size=1)
    try:
        sup.block_desc
        raise AssertionError("element-granular support offered block_desc")
    except ValueError as e:
        assert "block_size > 1" in str(e), e
    sup2 = BucketSupport(mask.clone(), block_size=8)  # no block_spans
    try:
        sup2.block_desc
        raise AssertionError("block_desc built without spans")
    except ValueError as e:
        assert "block_spans" in str(e), e
    print("ok: block_desc property guards (element support, missing spans)")


def test_resolve_layout_surface():
    assert resolve_layout("auto", 1) == "er"
    assert resolve_layout("auto", 1024) == "bb"
    assert resolve_layout("eb", 1) == "eb"  # explicit wins (config owns legality)
    assert resolve_layout("bb", 8) == "bb"
    print("ok: resolve_layout auto derivation + explicit passthrough")


def test_attach_alignment_gate():
    """The engine's attach gate: a bb config on a descriptor-form backend
    refuses a param span starting off the x grid, naming the bucket and
    param; aligned layouts attach cleanly, and the torch block form
    (span-relative tiling) is exempt. The fake engine carries only the
    attributes the attach path reads (object.__new__ skips the ctor)."""
    def _mk_engine(block_form):
        _seed({"sga_selection_block_size": 16})
        eng = object.__new__(SGAEngine)
        eng.selection = SimpleNamespace(block_size=16)
        eng.backend = SimpleNamespace(consumes="idx", layout="bb",
                                      block_form=block_form)
        eng._backend_layout = "bb"
        eng._selection_block_size = 16
        eng._chunks = {}
        eng._bucket_states = []
        eng._bucket_state_of = {}
        eng._param_locations = {}
        eng._selection_meta = None
        eng._always_sync_layers = None
        eng._forced_param_ids = set()
        eng.opt = SimpleNamespace(ensure_state=lambda bs: None)
        return eng

    # numel_per=60: param p1 spans [60, 120) — 60 % 16 != 0 -> refuse
    try:
        _mk_engine("block_desc").attach(_FakeChunk(n_params=2, numel_per=60))
        raise AssertionError("misaligned span attached under bb")
    except SGAEngineError as e:
        assert "p1" in str(e) and "16-aligned" in str(e), e
    # numel_per=64: spans [0,64) [64,128) — aligned -> clean attach
    eng2 = _mk_engine("block_desc")
    eng2.attach(_FakeChunk(n_params=2, numel_per=64))
    assert len(eng2._bucket_states) == 1
    # torch-form bb (block_idx) is exempt: tiling is span-relative
    _mk_engine("block_idx").attach(_FakeChunk(n_params=2, numel_per=60))
    print("ok: attach alignment gate (misaligned refuse + aligned pass)")


if __name__ == "__main__":
    test_pack_oracle_and_roundtrip_grid()
    test_pack_parity_with_bench_reference()
    test_pack_negative_misaligned_span()
    test_pack_negative_partial_block()
    test_pack_negative_non_pow2_x()
    test_block_desc_property_guards()
    test_resolve_layout_surface()
    test_attach_alignment_gate()
    print("ALL LAYOUT SUPPORT TESTS PASSED")
