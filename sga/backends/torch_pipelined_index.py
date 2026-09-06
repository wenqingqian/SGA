# Copyright (c) 2026, SGA framework contributors.
"""Index-form pipelined sparse allreduce backend (the unified torch path).

One backend, two transport modes chosen by ``high_priority_group``:

- ``false`` (default): collectives run on the training world's existing
  DP-CP group — the legacy ``indexed_sparse`` transport. With the default
  ``index_slices: 2`` the split is the bucket-midpoint partition
  (``support.split``).
- ``true``: collectives run on a twin NCCL process group over the same
  ranks created with ``ProcessGroupNCCL.Options(is_high_priority_stream=
  True)`` — the legacy ``pipelined_idx_sorted`` transport (compact payloads
  do not queue behind the training stream's communicator).

EMISSION (``sga_backend_emission``, default ``auto`` -> compact here):
how the sparse allreduce hands its result to the optimizer.

- ``compact`` (the inherent default): the pack is a fused gather+zero+scale
  Triton kernel into a persistent grow-only compact buffer, the allreduce
  runs on that buffer, and the communicated values STAY there —
  ``allreduce_sparse`` returns the values view and never writes the full
  grad buffer back (the zero store inside pack establishes the
  buffer-state invariant at grad_sync time; clip/grad_correct consume
  (idx, values) via the engine's compact branches).
- ``writeback``: the historical choreography (the pre-compact form; the
  end state is bitwise-equal to the nccl backend's within the dist
  suite's 2-rank scope — the mechanism itself differs) — pure-read
  ``index_select`` gather,
  payload-only scale, pipelined allreduce, ``scatter_selected`` writes the
  communicated values back into the dense grad buffer; the optimizer's
  masked kernel reads them there and does the selective clear post-commit.
  The engine's deterministic compact grad-norm reverts to the masked
  atomic path with it.

Twin-mode runtime gates (probed via the optional ``validate_runtime`` /
``prepare`` hooks, only active when ``high_priority_group`` is true):

- Topology: pipeline/tensor/context/expert parallel sizes all 1.
  ``new_group`` is a world collective every rank must enter in one
  deterministic global order; provable only under a single DP-CP group
  spanning the whole world (see ``validate_runtime``).
- Resume runs are refused on memory grounds (override:
  ``SGA_ALLOW_RESUME_PIPELINED=1`` on hardware with headroom; the
  plain-group mode needs no override — turn this backend's
  ``high_priority_group`` off instead). The twin communicator costs
  ~330MB/GPU of permanent residency; a resume run's checkpoint-load
  steady state on 40GB cards leaves less than that, and every warmup
  ordering was probed: the failures just move (first sparse sync
  ``ncclUnhandledCudaError: Failed to CUDA calloc async`` → checkpoint
  load OOM). Not a scheduling bug — arithmetic.

The twin group is created EAGERLY by ``prepare()`` at attach time —
symmetrically on every rank, BEFORE the training loop's allocator
expansion — and finished with a dummy warmup allreduce pair (one op sized
at/above the largest production slice, one 1-element op) so every lazily
created NCCL structure (transport connects, channel buffers, the priority
stream) is built at the process's cleanest memory moment.
``SGA_COMM_WARMUP=0`` disables the warmup (diagnostics / emergency
bypass); it must be set identically on every rank.
"""

import os

import torch

from sga.backends import COMM_WARMUP_ENV, RESUME_OVERRIDE_ENV
from sga.kernels import (gather_selected_blocks, pack_selected,
                         scatter_selected, scatter_selected_blocks)
from sga.plugins import BackendDecl, expand_plugin_params

DECL = BackendDecl(
    name="torch_pipelined_index",
    consumes="idx",
    emission="compact",
)

# This backend's own vocabulary: its params as FULL flat dests (defaults
# and required-ness live in the argparse declarations).
PARAMS = ("sga_backend_high_priority_group", "sga_backend_index_slices",
          "sga_backend_emission")

# Escape hatch for the resume fail-closed in twin mode. Parsed as exactly
# "1": any other value — "yes" included — keeps the safe refusal. Must be
# set identically on every rank (export via env.sh, not per-node): a
# partial export raises on only some ranks while the rest proceed into the
# next world collective.
_RESUME_OVERRIDE_ENV = RESUME_OVERRIDE_ENV
# Emergency/diagnostic bypass for the twin-PG warmup. Parsed as exactly "1"
# (the unset default); any OTHER value disables the warmup. Must be set
# identically on every rank: ranks that skip the warmup's blocking
# allreduce while the rest enter it deadlock the job inside NCCL at attach.
_WARMUP_ENV = COMM_WARMUP_ENV
# 8M fp32 elements = 32MB: at/above the largest single-slice allreduce in
# production (buckets ~40M elements × 40% support / 2 slices ≈ 8M), so the
# warmup connects every NCCL channel the real path will ever use. Assumed
# geometry — NOT scaled with self._slices (sized for the default
# index_slices=2): with index_slices=1 the largest production slice is
# ~16M elems and the first real allreduce exceeds this warmup op.
_TWIN_WARMUP_ELEMS = 8 << 20


class TorchPipelinedIndexBackend:
    """Consumes the canonical sorted-idx form of a bucket's support."""

    decl = DECL

    def __init__(self, params: dict):
        params = expand_plugin_params(
            PARAMS, dict(params), "sga_backend (torch_pipelined_index)")
        # Emission switch: how the sparse allreduce hands its result to
        # the optimizer. 'auto' (the flag default) resolves to this
        # backend's inherent compact form; explicit 'writeback' restores
        # the historical choreography (see allreduce_sparse). The DECL's
        # emission field is the declared inherent default the resolution
        # falls back to; the RESOLVED instance attribute is what the
        # engine routes on.
        self.emission = self.decl.resolve_emission(
            params["sga_backend_emission"])
        self._hiprio = bool(params["sga_backend_high_priority_group"])
        self._slices = int(params["sga_backend_index_slices"])
        if self._slices < 1:
            raise ValueError(f"index_slices must be >= 1, got {self._slices}")
        self._gather_bufs: dict = {}  # bucket key -> persistent compact buffer
        # comm_group -> twin high-priority group. Keyed by the group OBJECT
        # with a strong reference held on both sides (an id() key could
        # collide after a source group is released and Python reuses its
        # id). Populated only by prepare() — eagerly, never inside the sync
        # path, and only in twin mode.
        self._twins: dict = {}

    # ------------------------------------------------- transport selection

    def prepare(self, comm_group, device=None) -> None:
        """Eagerly create the twin high-priority NCCL group (twin mode only)
        and force its lazy NCCL setup now.

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
        if not self._hiprio or comm_group is None or comm_group in self._twins:
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
        buffers, and the priority stream appear at the FIRST collective.
        The blocking ops + synchronize also turn a sticky CUDA error into an
        immediate, well-located failure. ``flag`` is the resolved
        ``_WARMUP_ENV`` value, echoed on rank 0 so a rank with a divergent
        export is visible in the logs.
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
                "torch_pipelined_index twin-PG warmup failed at attach "
                "(the process's cleanest memory moment). Diagnostic only: "
                f"setting {_WARMUP_ENV}=0 bypasses the warmup, which "
                "returns the NCCL lazy init at first collective; the "
                "memory-safe path is high_priority_group: false") from e
        if rank == 0:
            free1, total = torch.cuda.mem_get_info(device)
            print(f"[SGA] twin-PG warmup on {device} "
                  f"({_WARMUP_ENV}={flag}): "
                  f"{(free0 - free1) / 2**20:.0f}MB free-memory delta "
                  "during warmup "
                  f"(free {free1 / 2**30:.2f}/{total / 2**30:.2f}GB)",
                  flush=True)

    def validate_runtime(self, args) -> None:
        """Config-time requirements specific to TWIN mode (invoked
        generically by sga.runtime.build_engine via the optional
        ``validate_runtime`` hook probe). The plain-group mode imposes
        nothing beyond the shared v1 set."""
        if not self._hiprio:
            return

        def _require(cond, why):
            if not cond:
                raise ValueError(f"torch_pipelined_index (high_priority_"
                                 f"group: true) requires {why}")

        # new_group is a WORLD collective that every rank — member or not —
        # must enter in one deterministic global order; with a single DP-CP
        # group every rank issues the identical creation call, and anything
        # else (multiple pipeline stages' data groups, tensor/context
        # shards, expert-parallel data groups with diverging per-rank bucket
        # counts) is refused here instead of hanging in a half-ordered
        # new_group.
        _require(getattr(args, "pipeline_model_parallel_size", 1) == 1,
                 "pipeline_model_parallel_size == 1 (its twin high-priority "
                 "NCCL communicator over the DP-CP ranks is only order-safe "
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
        # Resume (checkpoint tracker present) is refused on memory grounds:
        # the twin communicator costs ~330MB of permanent residency per GPU,
        # and a resume run's checkpoint-load/first-step steady state leaves
        # less than that. Fresh runs are unaffected. The plain-group mode
        # (high_priority_group: false) is the resume path; set
        # SGA_ALLOW_RESUME_PIPELINED=1 to override on hardware with headroom.
        load = getattr(args, "load", None)
        tracker = (os.path.join(load, "latest_checkpointed_iteration.txt")
                   if load else None)
        if (tracker and os.path.exists(tracker)
                and os.environ.get(_RESUME_OVERRIDE_ENV, "0") != "1"):
            raise ValueError(
                "torch_pipelined_index with high_priority_group: true does "
                "not support resume runs on this memory budget (twin-PG "
                "residency does not fit; load dir "
                f"{load!r} has a checkpoint tracker). Set this backend's "
                "high_priority_group to false for resume, or set "
                f"{_RESUME_OVERRIDE_ENV}=1 to override on hardware with "
                "headroom.")

    # ------------------------------------------------------------- the sync

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
        """The group the sparse collectives run on: the eager twin in twin
        mode (kept plain for non-NCCL groups / any group prepare was never
        called on — e.g. comm_group None in CPU unit tests); the given
        group itself in plain mode."""
        if comm_group is None:
            return None
        return self._twins.get(comm_group, comm_group)

    def _bounds(self, support, k):
        """(lo, hi) index ranges into ``support.idx`` per slice.

        Plain mode with the default 2 slices uses the bucket-midpoint split
        (``support.split``) — the historical partition, kept for arm
        continuity. Everything else splits the selected set evenly
        (``i*k//s``).
        """
        if not self._hiprio and self._slices == 2:
            n0 = support.split
            return [(0, n0), (n0, k)]
        s = self._slices
        return [(i * k // s, (i + 1) * k // s) for i in range(s)]

    def _allreduce_sparse_block(self, grad_data, support, comm_group,
                                gradient_scaling_factor, reduce_op, key):
        """Block-granular writeback sync (sga_selection_block_size > 1).

        Each block-idx entry heads a contiguous x-element run; the payload
        packs k_blocks*x slots whose lanes are mask-guarded (tail-short
        blocks contribute zeros — allreduce-neutral, and identical on
        every rank since all ranks hold the same verified mask). Slices
        pipeline over BLOCK counts (the default 2-slice plain mode splits
        at the bucket midpoint in block space, ``support.block_split``).
        The reduced values scatter back into the dense grad on the same
        lane mask (writeback contract: returns None; the optimizer's
        masked path reads them there and clears post-commit).

        Config validation refuses compact emission for block supports (the
        compact consumers — merge-join, compact norm — are
        element-granular); the raise below is the direct-construction
        belt.
        """
        if self.emission != "writeback":
            raise ValueError(
                "torch_pipelined_index: block supports "
                "(sga_selection_block_size > 1) run writeback emission "
                "only — the compact consumers are element-granular; set "
                "sga_backend_emission=writeback")
        x = int(support.block_size)
        blk = support.block_idx
        kb = blk.numel()
        if kb == 0:
            return None
        gather_buf = self._gather_buf(key, kb * x, grad_data.dtype,
                                      grad_data.device)
        group = self._group_for(comm_group)
        factor = (float(gradient_scaling_factor)
                  if (reduce_op == torch.distributed.ReduceOp.SUM
                      and gradient_scaling_factor != 1.0) else 1.0)
        if not self._hiprio and self._slices == 2:
            n0 = support.block_split
            bounds = [(0, n0), (n0, kb)]
        else:
            s = self._slices
            bounds = [(i * kb // s, (i + 1) * kb // s) for i in range(s)]

        # One allreduce in flight at a time (same overlap choreography as
        # the element path): slice i+1's gather touches grad/buffer memory
        # disjoint from slice i's in-flight transfer.
        prev = None  # (work, block_idx slice, payload slice)
        for lo, hi in bounds:
            if hi <= lo:
                continue
            bidx = blk[lo:hi]
            sel = gather_buf[lo * x:hi * x]
            gather_selected_blocks(grad_data, support.mask_u8, bidx, sel,
                                   x, factor)
            if prev is not None:
                prev[0].wait()
                scatter_selected_blocks(grad_data, support.mask_u8,
                                        prev[1], prev[2], x)
            prev = (torch.distributed.all_reduce(
                sel, op=reduce_op, group=group, async_op=True), bidx, sel)
        if prev is not None:
            prev[0].wait()
            scatter_selected_blocks(grad_data, support.mask_u8,
                                    prev[1], prev[2], x)
        return None

    def allreduce_sparse(self, grad_data: torch.Tensor, support, comm_group,
                         gradient_scaling_factor: float, reduce_op,
                         key: int):
        """Sparse allreduce of ``grad_data`` at ``support.idx`` positions —
        the contract is conditional on the resolved ``self.emission``.

        COMPACT EMISSION: returns the communicated values view.

        ``support`` is a ``sga.support.BucketSupport``; ``key`` identifies
        the owning bucket state for the persistent gather buffer. The fused
        pack kernel gathers ``grad_data[idx]`` into that buffer (scaled by
        ``gradient_scaling_factor`` when the reduce op is SUM — payload-only
        math, never the whole buffer) AND zeroes those positions of
        ``grad_data`` in the same pass; the allreduce runs on the buffer and
        the result stays there. The return value is ``buf[:k]`` — valid
        until this bucket's next pack; an empty tensor when nothing was
        selected (k == 0), still a Tensor so the engine can distinguish
        compact zero-selection from a writeback backend's None.

        WRITEBACK EMISSION: pure-read gather (``grad_data`` is never
        zeroed), payload-only scale, pipelined allreduce, then the reduced
        values are scattered back into ``grad_data`` — returns ``None``
        (also at k == 0); the engine treats None as the writeback
        signature and keeps clip/grad_correct on the masked paths.
        """
        # int32 canonical indices: refuse oversized buckets loudly instead
        # of wrapping silently (buckets ~40M elements today, 2^31 is far).
        if grad_data.numel() >= 2 ** 31:
            raise ValueError(
                f"int32 support indices assume < 2^31 bucket elements, "
                f"got {grad_data.numel()} (bucket key {key})"
            )
        if getattr(support, "block_size", 1) > 1:
            return self._allreduce_sparse_block(
                grad_data, support, comm_group, gradient_scaling_factor,
                reduce_op, key)
        indices = support.idx
        k = indices.numel()
        if k == 0:
            if self.emission == "writeback":
                return None
            return torch.empty(0, dtype=grad_data.dtype,
                               device=grad_data.device)

        gather_buf = self._gather_buf(key, k, grad_data.dtype, grad_data.device)
        group = self._group_for(comm_group)
        factor = (float(gradient_scaling_factor)
                  if (reduce_op == torch.distributed.ReduceOp.SUM
                      and gradient_scaling_factor != 1.0) else 1.0)

        if self.emission == "writeback":
            # The historical choreography (bit for bit the pre-compact
            # form; end state bitwise-equal to the nccl backend's within
            # the dist suite's 2-rank scope): pure-read gather, payload-only
            # scale, pipelined allreduce, scatter the communicated values
            # back into the dense grad; the optimizer's masked kernel
            # reads them there and does the selective clear post-commit.
            prev = None  # (work, idx, sel)
            for lo, hi in self._bounds(support, k):
                if hi <= lo:
                    continue
                idx = indices[lo:hi]
                sel = gather_buf[lo:hi]
                torch.index_select(grad_data, 0, idx, out=sel)
                if factor != 1.0:
                    sel *= factor
                if prev is not None:
                    prev[0].wait()
                    scatter_selected(grad_data, prev[1], prev[2])
                prev = (torch.distributed.all_reduce(
                    sel, op=reduce_op, group=group, async_op=True), idx, sel)
            if prev is not None:
                prev[0].wait()
                scatter_selected(grad_data, prev[1], prev[2])
            return None

        # One allreduce in flight at a time: slice i+1's pack (compute
        # stream) overlaps slice i's NCCL transfer. The pack of slice i+1
        # (grad reads/clears at idx[i+1:], buf writes at [lo:hi]) touches
        # disjoint memory from the in-flight allreduce on buf[:lo].
        prev = None  # (work,)
        for lo, hi in self._bounds(support, k):
            if hi <= lo:
                continue
            idx = indices[lo:hi]
            sel = gather_buf[lo:hi]
            pack_selected(grad_data, idx, sel, factor)
            if prev is not None:
                prev[0].wait()
            prev = (torch.distributed.all_reduce(
                sel, op=reduce_op, group=group, async_op=True),)
        if prev is not None:
            prev[0].wait()
        return gather_buf[:k]
