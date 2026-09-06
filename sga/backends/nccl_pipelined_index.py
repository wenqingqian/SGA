# Copyright (c) 2026, SGA framework contributors.
"""Index-form sparse allreduce backend on the raw ncclSparseAllReduce kernel.

Transport is the nccl-sar fork's ``ncclSparseAllReduce`` (Ring+Simple,
compact-domain wire cost that scales with idxCount): the kernel itself
accepts the bucket's DENSE gradient buffer plus idx and reads/writes it in
place, so no gather is needed for communication — selection, reduction and
writeback all happen inside the kernel; the only backend-side data
movement is the optional pre-scale (an ``index_select`` gather plus a
``scatter_selected`` writeback) when ``gradient_scaling_factor != 1``.
The idx contract is layered: identical counts come from the engine's
count guard, identical sets from its layer-2 dual checksum, and the
unique/sorted int32 indices from ``compact_mask_u8``.

The fork also changed ``ncclAllReduce``'s signature, so its library cannot
be LD_PRELOADed next to torch's bundled NCCL. The backend therefore owns a
raw ``ncclComm`` built with ``ncclGetUniqueId``/``ncclCommInitRank``: the
uniqueId crosses ranks through the training world's existing torch
distributed group (``broadcast_object_list``), and every collective is a
direct ctypes call on a backend-owned CUDA stream — torch's communicators
are untouched.

Attach-time discipline mirrors the twin-group rules (see
torch_pipelined_index): ``prepare()`` eagerly builds the comm and runs a
warmup pair (one op at/above the largest production op, one 1-element op)
so lazily-created NCCL structures appear at the process's cleanest memory
moment; ``validate_runtime`` refuses every topology whose comm creation
order is not provably deterministic (single DP-CP group spanning the
world) and refuses resume runs on the same residency grounds
(``SGA_ALLOW_RESUME_PIPELINED=1`` overrides on hardware with headroom;
raw-comm residency is smaller than a twin process group, but no 40GB
resume steady-state measurement exists yet, so the refusal stays closed
until multi-node data lands).

One kernel launch per sync, unconditionally: the measured cost of an extra
slice is one more full kernel invocation (per-call fixed cost, cross-node
amplified), with no overlap on the kernel stream to pay for it, so no
slicing is offered.

Stream semantics satisfy the calling-stream completion contract: the
selected payload is pre-scaled on the calling stream (only when
``gradient_scaling_factor != 1``), an event orders the calling stream
ahead of the kernel stream, and a recorded event orders the kernel stream
back ahead of the calling stream before ``allreduce_sparse`` returns —
subsequent ops on the calling stream observe the reduced gradients without
any host synchronization.

The kernel is pinned to fp32 SUM; other dtypes/ops are refused loudly.
Payload scaling happens before the reduction (the SGA convention: the
communicated coordinate carries ``local * factor``, so the reduction sums
equally-scaled replicas). Bitwise equality with the torch backends is an
empirical result within the dist suite's validated scope — 2 ranks, where
fp32 addition is commutative and the power-of-two test factor makes the
pre-scale exact (A100 singlesocket and cross-node IB) — not a property of
the scaling convention alone: a larger DP world must additionally match
reduction trees, and nothing pins DP worlds to powers of two.
"""

import ctypes
import os

import torch

from sga.backends import COMM_WARMUP_ENV, RESUME_OVERRIDE_ENV
from sga.kernels import scatter_selected
from sga.plugins import BackendDecl, expand_plugin_params

DECL = BackendDecl(
    name="nccl_pipelined_index",
    consumes="idx",
)

# This backend's own vocabulary: its params as FULL flat dests.
PARAMS = ("sga_backend_emission",)

# Path to the nccl-sar sparse library exporting ncclSparseAllReduce (the
# fork's ncclAllReduce signature differs from stock, so it can never be
# picked up via the normal loader). env.sh is the single source of truth
# for the node layout; nothing sane can be defaulted here.
_LIB_ENV = "SGA_NCCL_SAR_LIB"
# Same escape hatch and warmup-bypass semantics as torch_pipelined_index's
# twin mode (exactly "1" parsed, identical on every rank — see that
# backend for the deadlock/refusal rationale each flag documents).
_RESUME_OVERRIDE_ENV = RESUME_OVERRIDE_ENV
_WARMUP_ENV = COMM_WARMUP_ENV
# Sized at/above the largest single-op allreduce possible in production
# (one whole bucket, ~40M elements — the ratio-100% corner), so the warmup
# connects every NCCL channel the real path can use.
_WARMUP_ELEMS = 40 << 20

_NCCL_FLOAT32 = 7
_NCCL_SUM = 0


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


class _RawComm:
    """A loaded sparse NCCL library plus the raw comm and stream it serves."""

    def __init__(self, lib, comm, stream):
        self.lib = lib
        self.comm = comm
        self.stream = stream


class NcclPipelinedIndexBackend:
    """Consumes the canonical sorted-idx form of a bucket's support."""

    decl = DECL

    def __init__(self, params: dict):
        # Validation-only beyond emission (this backend takes no other
        # params): unknown keys fail here, loudly.
        params = expand_plugin_params(PARAMS, dict(params),
                                      "sga_backend (nccl_pipelined_index)")
        # Emission is inherent here (writeback): the communicated values
        # are scattered back into the dense grad buffer for the masked
        # optimizer kernel. 'auto' (the flag default) resolves to that;
        # an explicit 'writeback' is the same thing spelled out; an
        # explicit 'compact' is refused with a pointer — compact
        # emission is a torch_pipelined_index capability (the flag-layer
        # rule catches it earlier in real runs; this is the
        # direct-construction backstop).
        emission = params["sga_backend_emission"]
        if emission == "compact":
            raise ValueError(
                "sga_backend (nccl_pipelined_index): emission 'compact' is only supported "
                "by torch_pipelined_index; this backend is "
                "writeback-inherent (set sga_backend: "
                "torch_pipelined_index to switch emission)")
        self.emission = self.decl.resolve_emission(emission)

        # comm_group -> _RawComm. Keyed by the group OBJECT with a strong
        # reference held on both sides (an id() key could collide after a
        # source group is released and Python reuses its id). Populated
        # only by prepare() — eagerly, never inside the sync path.
        self._comms: dict = {}
        self._scale_bufs: dict = {}  # bucket key -> persistent compact buffer

    # --------------------------------------------------------- raw NCCL comm

    def _load_lib(self):
        path = os.environ.get(_LIB_ENV)
        if not path:
            raise ValueError(
                f"nccl_pipelined_index needs {_LIB_ENV} to point at the "
                "nccl-sar library exporting ncclSparseAllReduce (the fork "
                "cannot be loaded through torch's bundled NCCL)")
        try:
            # ctypes.CDLL's default RTLD_LOCAL is load-bearing: the fork's
            # altered ncclAllReduce must never interpose on torch's NCCL.
            lib = ctypes.CDLL(path)
        except OSError as e:
            raise RuntimeError(
                f"nccl_pipelined_index could not load {_LIB_ENV}={path!r}"
            ) from e
        try:
            lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
            lib.ncclGetUniqueId.restype = ctypes.c_int
            lib.ncclCommInitRank.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                _NcclUniqueId, ctypes.c_int,
            ]
            lib.ncclCommInitRank.restype = ctypes.c_int
            lib.ncclSparseAllReduce.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.ncclSparseAllReduce.restype = ctypes.c_int
        except AttributeError as e:
            raise RuntimeError(
                f"{path!r} does not export ncclSparseAllReduce — wrong "
                "library build?") from e
        return lib

    def prepare(self, comm_group, device=None) -> None:
        """Eagerly build the raw NCCL comm for this group and warm it up.

        Optional engine hook, called once per distinct DP-CP group at
        attach time — symmetrically on every rank, before any
        training-loop allocation expands the caching allocator. The
        uniqueId is generated by group rank 0 and crosses the group over
        torch distributed (control plane only; the kernel traffic never
        touches torch's communicators). ``device`` is the bucket
        grad-buffer device.
        """
        if comm_group is None or comm_group in self._comms:
            return
        if not (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            raise RuntimeError(
                "nccl_pipelined_index prepare() needs an initialized torch "
                "distributed world for the uniqueId exchange")
        dev = device if device is not None else torch.device(
            "cuda", torch.cuda.current_device())
        if dev.type != "cuda":
            raise ValueError(f"raw NCCL comm needs a cuda device, got {dev}")

        lib = self._load_lib()
        ranks = torch.distributed.get_process_group_ranks(comm_group)
        size = torch.distributed.get_world_size(comm_group)
        my_rank = torch.distributed.get_rank(comm_group)

        uid = _NcclUniqueId()
        if my_rank == 0:
            ret = lib.ncclGetUniqueId(ctypes.byref(uid))
            if ret != 0:
                raise RuntimeError(f"ncclGetUniqueId failed: {ret}")
            payload = bytes(ctypes.cast(
                ctypes.addressof(uid),
                ctypes.POINTER(ctypes.c_char * 128)).contents)
        else:
            payload = b""
        # Object collective over the training group: NCCL process groups
        # need the device pinned explicitly for the internal broadcast.
        box = [payload]
        torch.distributed.broadcast_object_list(
            box, src=ranks[0], group=comm_group, device=dev)
        if my_rank != 0:
            if len(box[0]) != 128:
                raise RuntimeError(
                    f"uniqueId broadcast delivered {len(box[0])} bytes, "
                    "expected 128 — corrupted control-plane payload")
            ctypes.memmove(ctypes.addressof(uid), box[0], 128)

        # NCCL binds the new comm to the calling thread's current device —
        # pin it to dev around init and warmup so it can never attach to
        # the wrong GPU.
        with torch.cuda.device(dev):
            comm = ctypes.c_void_p()
            ret = lib.ncclCommInitRank(ctypes.byref(comm), size, uid, my_rank)
            if ret != 0:
                raise RuntimeError(
                    f"ncclCommInitRank failed: {ret} (size={size}, "
                    f"rank={my_rank})")
            stream = torch.cuda.Stream(device=dev)

            flag = os.environ.get(_WARMUP_ENV, "1")  # exactly "1" enables
            if flag == "1":
                self._warm_up_comm(lib, comm, stream, dev, flag)
            elif torch.distributed.get_rank() == 0:
                print(f"[SGA] WARNING: {_WARMUP_ENV}={flag} disables the "
                      "raw-comm warmup; NCCL lazy init at first collective "
                      "is back in effect (diagnostic bypass)",
                      flush=True)
        # Only after a successful warmup: a half-initialized comm that
        # later collectives could pick up is strictly worse than dying at
        # attach.
        self._comms[comm_group] = _RawComm(lib, comm, stream)

    def _warm_up_comm(self, lib, comm, stream, device, flag) -> None:
        """Build every lazily-created NCCL structure at attach time with a
        dense-index op at/above the largest production op plus a
        1-element op; the synchronize turns a sticky CUDA error into an
        immediate, well-located failure. ``flag`` is the resolved
        ``_WARMUP_ENV`` value, echoed on rank 0 so a divergent export is
        visible in the logs."""
        try:
            with torch.cuda.stream(stream):
                big_idx = torch.arange(_WARMUP_ELEMS, dtype=torch.int32,
                                       device=device)
                big = torch.zeros(_WARMUP_ELEMS, dtype=torch.float32,
                                  device=device)
                self._launch(lib, comm, stream, big, big_idx)
                one_idx = torch.zeros(1, dtype=torch.int32, device=device)
                one = torch.zeros(1, dtype=torch.float32, device=device)
                self._launch(lib, comm, stream, one, one_idx)
            stream.synchronize()
        except Exception as e:
            raise RuntimeError(
                "nccl_pipelined_index comm warmup failed at attach (the "
                "process's cleanest memory moment). Diagnostic only: "
                f"setting {_WARMUP_ENV}=0 bypasses the warmup, which "
                "returns the NCCL lazy init at first collective.") from e
        if torch.distributed.get_rank() == 0:
            print(f"[SGA] raw-comm warmup on {device} "
                  f"({_WARMUP_ENV}={flag}) ok", flush=True)

    def validate_runtime(self, args) -> None:
        """Config-time requirements (invoked generically by
        sga.runtime.build_engine via the optional ``validate_runtime`` hook
        probe). Same contract as the twin mode of torch_pipelined_index:
        comm creation is only order-provable under a single DP-CP group
        spanning the world, and resume is refused on residency grounds."""

        def _require(cond, why):
            if not cond:
                raise ValueError(f"nccl_pipelined_index requires {why}")

        _require(getattr(args, "pipeline_model_parallel_size", 1) == 1,
                 "pipeline_model_parallel_size == 1 (its raw NCCL "
                 "communicator over the DP-CP ranks is only order-safe "
                 "under the single-pipe phase calendar)")
        _require(getattr(args, "tensor_model_parallel_size", 1) == 1,
                 "tensor_model_parallel_size == 1 (single DP-CP group: every "
                 "world rank must share one sparse communicator)")
        _require(getattr(args, "context_parallel_size", 1) == 1,
                 "context_parallel_size == 1 (single DP-CP group: every "
                 "world rank must share one sparse communicator)")
        _require(getattr(args, "expert_model_parallel_size", 1) == 1,
                 "expert_model_parallel_size == 1 (expert ranks form separate "
                 "data-parallel groups and per-rank expert bucket counts "
                 "diverge)")
        _require(not getattr(args, "ddp_average_in_collective", False),
                 "a SUM-only kernel (--ddp-average-in-collective is "
                 "unsupported)")
        # Resume (checkpoint tracker present) is refused on residency
        # grounds, conservatively matching the twin-group policy until a
        # 40GB resume steady-state measurement exists for the raw comm.
        load = getattr(args, "load", None)
        tracker = (os.path.join(load, "latest_checkpointed_iteration.txt")
                   if load else None)
        if (tracker and os.path.exists(tracker)
                and os.environ.get(_RESUME_OVERRIDE_ENV, "0") != "1"):
            raise ValueError(
                "nccl_pipelined_index does not support resume runs on this "
                "memory budget (raw-communicator residency is unmeasured "
                "for resume; conservative refusal). Set "
                f"{_RESUME_OVERRIDE_ENV}=1 to override on hardware with "
                "headroom.")

    # ------------------------------------------------------------- the sync

    def _launch(self, lib, comm, stream, buf: torch.Tensor,
                idx: torch.Tensor) -> None:
        ret = lib.ncclSparseAllReduce(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            idx.numel(),
            buf.numel(),
            _NCCL_FLOAT32,
            _NCCL_SUM,
            comm,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(
                f"ncclSparseAllReduce failed: {ret} (k={idx.numel()}, "
                f"count={buf.numel()})")

    def _scale_buf(self, key: int, k: int, dtype, device) -> torch.Tensor:
        # Grow-only, rounded up to 4M elements (16MB fp32): exact-size
        # reallocation on every selected-count change re-triggers the
        # allocator map/unmap storms the persistent-buffer discipline
        # exists to avoid.
        buf = self._scale_bufs.get(key)
        cap = -(-(k) // (4 << 20)) * (4 << 20)
        if (buf is None or buf.dtype != dtype or buf.device != device
                or buf.numel() < cap):
            new_cap = cap if buf is None else max(cap, buf.numel())
            buf = torch.empty(new_cap, dtype=dtype, device=device)
            self._scale_bufs[key] = buf
        return buf

    def allreduce_sparse(self, grad_data: torch.Tensor, support, comm_group,
                         gradient_scaling_factor: float, reduce_op,
                         key: int) -> None:
        """In-place sparse allreduce of ``grad_data`` at ``support.idx``
        positions via ncclSparseAllReduce on the backend-owned comm.

        ``support`` is a ``sga.support.BucketSupport``; ``key`` identifies
        the owning bucket state for the persistent scale buffer. The
        selected payload is scaled by ``gradient_scaling_factor`` before
        the SUM reduction (payload-only; residual coordinates are never
        touched — the kernel writes only at idx positions).
        """
        if reduce_op != torch.distributed.ReduceOp.SUM:
            raise ValueError(
                f"nccl_pipelined_index is pinned to SUM, got {reduce_op}")
        if grad_data.dtype != torch.float32:
            raise ValueError(
                "nccl_pipelined_index is pinned to fp32, got "
                f"{grad_data.dtype}")
        # int32 canonical indices: refuse oversized buckets loudly instead
        # of wrapping silently (buckets ~40M elements today, 2^31 is far).
        if grad_data.numel() >= 2 ** 31:
            raise ValueError(
                f"int32 support indices assume < 2^31 bucket elements, "
                f"got {grad_data.numel()} (bucket key {key})"
            )
        # _launch hands grad_data's raw pointer to the kernel as a dense
        # contiguous fp32 array (the torch twins' index_select is
        # layout-agnostic; this backend is not): a non-contiguous buffer
        # would be silently misread.
        if not grad_data.is_contiguous():
            raise ValueError(
                f"nccl_pipelined_index needs a contiguous grad buffer, got "
                f"non-contiguous (bucket key {key})")
        # The fork kernel is element-granular: an idx entry moves exactly
        # one element, and there is no block-size argument to teach it
        # contiguous runs (config validation refuses this combination
        # earlier; this is the direct-construction belt).
        if getattr(support, "block_size", 1) > 1:
            raise ValueError(
                "nccl_pipelined_index cannot carry block-granular supports "
                "(sga_selection_block_size > 1): the fork kernel's idx is "
                "element-granular. Use torch_pipelined_index with "
                "sga_backend_emission=writeback, or torch_pipelined_mask")
        indices = support.idx
        k = indices.numel()
        if k == 0:
            return
        entry = self._comms.get(comm_group)
        if entry is None:
            raise RuntimeError(
                "nccl_pipelined_index has no comm for this group — "
                "prepare() must run at attach time (the engine calls it "
                "once per DP-CP group)")
        if grad_data.device != entry.stream.device:
            raise ValueError(
                f"nccl_pipelined_index grad buffer device "
                f"{grad_data.device} does not match the prepared comm "
                f"stream device {entry.stream.device} (bucket key {key})")

        calling = torch.cuda.current_stream(grad_data.device)
        if gradient_scaling_factor != 1.0:
            sel = self._scale_buf(key, k, grad_data.dtype,
                                  grad_data.device)[:k]
            torch.index_select(grad_data, 0, indices, out=sel)
            sel *= gradient_scaling_factor
            scatter_selected(grad_data, indices, sel)

        # Order the (optional) pre-scale on the calling stream ahead of
        # the kernel stream, and the kernel stream back ahead of the
        # calling stream — the calling-stream completion contract, with no
        # host synchronization anywhere.
        ready = torch.cuda.Event()
        ready.record(calling)
        ready.wait(entry.stream)
        self._launch(entry.lib, entry.comm, entry.stream, grad_data, indices)
        done = torch.cuda.Event()
        done.record(entry.stream)
        done.wait(calling)
