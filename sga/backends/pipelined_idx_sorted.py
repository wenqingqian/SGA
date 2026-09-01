# Copyright (c) 2026, SGA framework contributors.
"""Pipelined index-form sparse allreduce on a high-priority NCCL stream.

Twin-process-group variant of indexed_sparse : the sparse
collectives run on a second NCCL process group over the SAME ranks created
with ``ProcessGroupNCCL.Options(is_high_priority_stream=True)``, so the
compact payloads do not queue behind work on the training stream's
communicator. The gather/writeback stays pure torch — ``index_select`` /
``index_put_`` accept the canonical int32 sorted indices directly (verified
on torch 2.9) — and the compact set K is split into ``slices`` (default 3)
evenly-spaced contiguous ranges (``i*k//S`` bounds), each allreduce launched
async so the next slice's gather overlaps the previous slice's NCCL
transfer, exactly like the two-slice midpoint split of indexed_sparse.

The twin group is created EAGERLY by ``prepare()``, which the engine calls
once at attach time — symmetrically on every rank, BEFORE the training
loop's allocator expansion. Creation must not be lazy at first use, for two
independent reasons: (a) world ordering — ``torch.distributed.new_group``
is a world collective that every rank, including non-members, must enter in
one deterministic global order; that is provable only under the single
DP-CP-group world ``validate_runtime`` pins at config time (tp == cp ==
ep == pp == 1), where every rank issues an identical creation call (under
expert parallelism even per-rank bucket counts can diverge the arrival
order); and (b) memory timing — NCCL establishes a communicator's transport
connects, persistent channel buffers, and the high-priority stream LAZILY,
at the first COLLECTIVE on the group, not at ``new_group`` (which only
builds the comm object; eager ``device_id`` init likewise stops short of
transport). When that first collective is the first sparse sync deep in a
run, its internal cudaMalloc competes with the caching allocator's
reservations: under ``expandable_segments:True``, resume runs (first sparse
sync at step 68) crashed deterministically with
``ncclUnhandledCudaError: Failed to CUDA calloc async`` while fresh runs
(first sparse sync at step 8) passed — and an eager new_group alone did
NOT fix the resume crash. ``prepare()`` therefore finishes with a dummy
warmup allreduce pair — one op sized at/above the largest production
slice (covers the Simple-protocol path, all channels), one 1-element op
(the LL path of the final short slice) — forcing every lazy structure
to be built at attach, the process's cleanest and most symmetric moment.
``SGA_COMM_WARMUP=0`` disables the warmup (diagnostics / emergency
bypass to indexed_sparse).

Requires pipeline_model_parallel_size == 1 plus a single DP-CP group
covering the whole world (rejected config-time by this backend's
``validate_runtime``), which also refuses resume runs — the twin's
residency does not fit a resume run's steady state (see
``validate_runtime``). Payload-only scaling as in the mask backend
(frozen decision). Persistent grow-only gather buffers
(notes/sga-engine-structure.md section 7).
"""

import os

import torch

from sga.plugins import BackendDecl, PluginParam, resolve_plugin_params

DECL = BackendDecl(
    name="pipelined_idx_sorted",
    consumes="idx",
    params=(
        PluginParam(
            "slices", int, default=3,
            doc="number of idx slices pipelined against the twin allreduces",
        ),
    ),
)

# Emergency/diagnostic bypass for this backend's twin-PG warmup. Parsed as
# exactly "1" (the unset default): any OTHER value — "0", "true", "yes"
# alike — silently disables the warmup. Must be set identically on every
# rank (export via env.sh, not per-node): ranks that skip the warmup's
# blocking allreduce while the rest enter it deadlock the job inside NCCL
# at attach.
_WARMUP_ENV = "SGA_COMM_WARMUP"
# Escape hatch for the resume fail-closed below (hardware with memory
# headroom, e.g. 80GB cards). Parsed as exactly "1": any other value —
# "yes" included — keeps the safe refusal. Must be set identically on
# every rank (export via env.sh, not per-node): a partial export raises on
# only some ranks while the rest proceed into the next world collective.
_RESUME_OVERRIDE_ENV = "SGA_ALLOW_RESUME_PIPELINED"
# 8M fp32 elements = 32MB: at/above the largest single-slice allreduce in
# production (buckets ~40M elements × 40% support / S=3 slices ≈ 5.3M), so
# the warmup connects every NCCL channel the real path will ever use.
# Assumed geometry — NOT scaled with self._slices: sized for the default
# slices=3; with slices: 1 the largest production slice is ~16M elems and
# the first real allreduce exceeds this warmup op.
_TWIN_WARMUP_ELEMS = 8 << 20


class PipelinedIdxSortedBackend:
    """Consumes the canonical sorted-idx form of a bucket's support."""

    decl = DECL

    def __init__(self, params: dict):
        params = resolve_plugin_params("backend", "pipelined_idx_sorted",
                                       DECL, dict(params))
        self._slices = int(params["slices"])
        if self._slices < 1:
            raise ValueError(f"slices must be >= 1, got {self._slices}")
        self._gather_bufs: dict = {}  # bucket key -> persistent compact buffer
        # comm_group -> twin collect-on group. Keyed by the group OBJECT with
        # a strong reference held on both sides (an id() key could collide
        # after a source group is released and Python reuses its id).
        # Populated only by prepare() — eagerly, never inside the sync path.
        self._twins: dict = {}

    def prepare(self, comm_group, device=None) -> None:
        """Eagerly create this backend's twin high-priority NCCL group and
        force its lazy NCCL setup now.

        Optional engine hook, called once per distinct DP-CP group at attach
        time — symmetrically on every rank (DDP attaches identically
        everywhere), before any training-loop allocation expands the caching
        allocator. See the module docstring for why creation and the warmup
        collective cannot be left to first use. ``device`` is the bucket
        grad-buffer device (the ground truth the sparse collectives will run
        on); without it the current CUDA device is used.

        Non-NCCL groups (gloo, CPU unit tests) keep the plain group: the
        priority stream is a perf knob with no correctness content.
        """
        if comm_group is None or comm_group in self._twins:
            return
        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_backend(comm_group) == "nccl"):
            from torch.distributed import ProcessGroupNCCL

            dev = device if device is not None else torch.device(
                "cuda", torch.cuda.current_device())
            if dev.type != "cuda":
                raise ValueError(
                    f"twin NCCL group needs a cuda device, got {dev}")
            opts = ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = True
            ranks = torch.distributed.get_process_group_ranks(comm_group)
            twin = torch.distributed.new_group(
                ranks=ranks, backend="nccl", pg_options=opts, device_id=dev)
            flag = os.environ.get(_WARMUP_ENV, "1")  # exactly "1" enables
            if flag == "1":
                self._warm_up_twin(twin, dev, flag)
            elif torch.distributed.get_rank() == 0:
                print(f"[SGA] WARNING: {_WARMUP_ENV}={flag} disables the "
                      "twin-PG warmup; the backend stays pipelined and "
                      "NCCL lazy init at first collective is back in "
                      "effect (diagnostic bypass)",
                      flush=True)
        else:
            twin = comm_group
        # Only after a successful warmup: a half-initialized twin that later
        # collectives could pick up is strictly worse than dying at attach.
        self._twins[comm_group] = twin

    def _warm_up_twin(self, twin, device, flag) -> None:
        """Build every lazily-created NCCL structure at attach time.

        new_group creates only the comm object: transport connects, channel
        buffers, and the priority stream appear at the FIRST collective
        (see module docstring). The blocking ops + synchronize also turn a
        sticky CUDA error into an immediate, well-located failure.
        ``flag`` is the resolved ``_WARMUP_ENV`` value, echoed on rank 0
        so a rank with a divergent export is visible in the logs.
        """
        rank = torch.distributed.get_rank()
        try:
            free0, _total = torch.cuda.mem_get_info(device)
            big = torch.zeros(_TWIN_WARMUP_ELEMS, dtype=torch.float32,
                              device=device)
            torch.distributed.all_reduce(big, group=twin)
            small = torch.zeros(1, dtype=torch.float32, device=device)
            torch.distributed.all_reduce(small, group=twin)
            torch.cuda.synchronize(device)
        except Exception as e:
            raise RuntimeError(
                "pipelined_idx_sorted twin-PG warmup failed at attach "
                "(the process's cleanest memory moment). Diagnostic only: "
                f"setting {_WARMUP_ENV}=0 bypasses the warmup, which "
                "returns the NCCL lazy init at first collective; the "
                "resume-safe path is the indexed_sparse backend") from e
        if rank == 0:
            free1, total = torch.cuda.mem_get_info(device)
            print(f"[SGA] twin-PG warmup on {device} "
                  f"({_WARMUP_ENV}={flag}): "
                  f"{(free0 - free1) / 2**20:.0f}MB free-memory delta "
                  f"during warmup "
                  f"(free {free1 / 2**30:.2f}/{total / 2**30:.2f}GB)",
                  flush=True)

    def validate_runtime(self, args) -> None:
        """Config-time requirements specific to THIS backend (invoked
        generically by sga.runtime.build_engine via the optional
        ``validate_runtime`` hook probe)."""
        def _require(cond, why):
            if not cond:
                raise ValueError(f"pipelined_idx_sorted requires {why}")

        # The original single-pipe requirement (moved here from the shared
        # builder's name-keyed branch — backend policy lives in the backend).
        _require(getattr(args, "pipeline_model_parallel_size", 1) == 1,
                 "pipeline_model_parallel_size == 1 (its twin high-priority "
                 "NCCL communicator over the DP-CP ranks is only order-safe "
                 "under the single-pipe phase calendar)")
        # Exactly ONE DP-CP group spanning the whole world. new_group is a
        # WORLD collective that every rank — member or not — must enter in
        # one deterministic global order; with a single DP-CP group every
        # rank issues the identical creation call, and anything else
        # (multiple pipeline stages' data groups, tensor/context shards,
        # expert-parallel data groups with diverging per-rank bucket counts)
        # is refused here instead of hanging in a half-ordered new_group.
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
        # Resume (checkpoint tracker present) is refused on memory grounds:
        # the twin communicator costs ~330MB of permanent residency per GPU,
        # and a resume run's checkpoint-load/first-step steady state leaves
        # less than that — five probes  showed every warmup
        # ordering dies: the first still-cold communicator at steady-state
        # memory crashes with ``ncclUnhandledCudaError: Failed to CUDA calloc
        # async``, and warming everything shifts the failure into the
        # checkpoint load itself (CUDA OOM). Fresh runs are unaffected —
        # every communicator initializes at steps 1-8 while memory is clean
        # (60/60 smoke passes). Resume with indexed_sparse instead; set
        # SGA_ALLOW_RESUME_PIPELINED=1 to override on hardware with headroom.
        load = getattr(args, "load", None)
        tracker = (os.path.join(load, "latest_checkpointed_iteration.txt")
                   if load else None)
        if (tracker and os.path.exists(tracker)
                and os.environ.get(_RESUME_OVERRIDE_ENV, "0") != "1"):
            raise ValueError(
                "pipelined_idx_sorted does not support resume runs on this "
                f"memory budget (twin-PG residency does not fit; load dir "
                f"{load!r} has a checkpoint tracker). Use backend "
                "indexed_sparse for resume, or set "
                f"{_RESUME_OVERRIDE_ENV}=1 to override on hardware with "
                "headroom (known-issue, iteration probes 5-5e).")

    def _gather_buf(self, key: int, k: int, dtype, device) -> torch.Tensor:
        # Grow-only, rounded up to 4M elements (16MB fp32): exact-size
        # reallocation on every selected-count change (including shrinks)
        # re-triggers the allocator map/unmap storms the persistent-buffer
        # discipline exists to avoid. Slices below are bounded by k anyway.
        buf = self._gather_bufs.get(key)
        cap = -(-(k) // (4 << 20)) * (4 << 20)
        if (buf is None or buf.dtype != dtype or buf.device != device
                or buf.numel() < cap):
            new_cap = cap if buf is None else max(cap, buf.numel())
            buf = torch.empty(new_cap, dtype=dtype, device=device)
            self._gather_bufs[key] = buf
        return buf

    def _group_for(self, comm_group):
        """The group the sparse collectives run on: the eager twin built by
        ``prepare()`` (kept plain for non-NCCL groups / any group prepare was
        never called on — e.g. comm_group None in CPU unit tests)."""
        if comm_group is None:
            return None
        return self._twins.get(comm_group, comm_group)

    def allreduce_sparse(self, grad_data: torch.Tensor, support, comm_group,
                         gradient_scaling_factor: float, reduce_op,
                         key: int) -> None:
        """Sparse allreduce of ``grad_data`` at ``support.idx`` positions.

        ``support`` is a ``sga.support.BucketSupport``; ``key`` identifies the
        owning bucket state for the persistent gather buffer.
        """
        # int32 canonical indices: refuse oversized buckets loudly instead
        # of wrapping silently (buckets ~40M elements today, 2^31 is far).
        assert grad_data.numel() < 2 ** 31, (
            f"int32 support indices assume < 2^31 bucket elements, "
            f"got {grad_data.numel()} (bucket key {key})"
        )
        indices = support.idx
        k = indices.numel()
        if k == 0:
            return

        gather_buf = self._gather_buf(key, k, grad_data.dtype, grad_data.device)
        group = self._group_for(comm_group)
        s = self._slices
        bounds = [i * k // s for i in range(s + 1)]

        def _scale(payload: torch.Tensor) -> torch.Tensor:
            if (reduce_op == torch.distributed.ReduceOp.SUM
                    and gradient_scaling_factor != 1.0):
                payload *= gradient_scaling_factor
            return payload

        # One allreduce in flight at a time: slice i+1's gather (compute
        # stream) overlaps slice i's NCCL transfer (priority stream).
        prev = None  # (work, idx, sel)
        for i in range(s):
            lo, hi = bounds[i], bounds[i + 1]
            if hi <= lo:
                continue
            idx = indices[lo:hi]
            sel = gather_buf[lo:hi]
            torch.index_select(grad_data, 0, idx, out=sel)
            _scale(sel)
            if prev is not None:
                prev[0].wait()
                grad_data.index_put_((prev[1],), prev[2])
            prev = (torch.distributed.all_reduce(
                sel, op=reduce_op, group=group, async_op=True), idx, sel)
        if prev is not None:
            prev[0].wait()
            grad_data.index_put_((prev[1],), prev[2])
