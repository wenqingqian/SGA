# Copyright (c) 2026, SGA framework contributors.
"""Sync-isolated comm-process timing (the ``sga_backend_timer`` instrument).

Off unless configured (the knob's value is an output directory). When on,
every timed region is bracketed by a device-wide ``torch.cuda.synchronize``
on entry and a calling-stream event synchronize on exit — the sync brackets
destroy pipelining exposure ON PURPOSE: the recorded numbers are the comm
process's ISOLATED duration (breakdown caliber), never an e2e reference.
The e2e effect of a backend is read from the un-instrumented runs only.

Records: one JSON line per timed region, appended to
``<dir>/backend_timer_rank{R}.jsonl`` — per-rank files, because NFS gives
no cross-client append atomicity; the analysis reduces them to per-iter ×
phase sums with max over ranks (a synchronous collective is closed by its
slowest rank). Phases:

  sparse    torch backends: the whole ``allreduce_sparse`` call — the
            pack/scatter that overlaps the transfer is part of the comm
            process; nccl: the kernel launch choreography plus the
            kernel itself — the region chains the kernel back onto the
            calling stream before end(), so the exit event observes it
            and the record is the kernel's isolated duration (the
            ready/done ordering enqueues are µs-scale launch cost).
  prescale  nccl's optional payload pre-scale segment (three passes over
            k), recorded apart so the analysis can add or subtract it
            when aligning the torch/nccl calibers.
  fullsync  a model chunk's ``finish_grad_sync`` region — the dense arm's
            stock grad sync and the sparse arms' boundary-step full sync
            (one caliber, one phase; includes Megatron's bucket pre-scale).

iter semantics: the step stamp lives in ``SGAEngine.grad_sync``, so
fullsync records from the dense arm (``sga_enabled=false``) and from
shadow mode carry ``iter: null`` — the analysis reconstructs their
grouping from seq/ts ordering (stamping the dense arm would need a new
call point inside Megatron-LM's training loop, which the fork's
narrow-waist rule forbids). On a resume, the first grad_sync is stamped
``iter: 1`` instead of the fork's absolute iteration K (the stamp
realigns at the first optimizer_step): discard the first post-resume
step in analysis.

Compact-emission note: the fused pack zeroes the communicated grad
positions inside the timed region; its writeback counterpart (the masked
clear post-commit) lives on the optimizer side, outside. Cross-emission
reads of phase=sparse carry that sync/consume boundary split — the
emission axis's verdict belongs to the e2e runs.
"""

import functools
import json
import os
import sys
import time

import torch

_STATE = None  # _TimerState when configured, None = off (identity path)


class _TimerState:
    def __init__(self, out_dir, rank):
        self.out_dir = out_dir
        self.rank = rank
        self.iter = None  # engine step stamp (None until the first grad_sync)
        self.seq = 0
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"backend_timer_rank{rank}.jsonl")
        # one line per record, flushed as written — crash-safe, and the
        # per-rank file is this process's alone (no cross-writer races)
        self.fh = open(path, "a", buffering=1)

    def write(self, record):
        self.seq += 1
        self.fh.write(json.dumps(record, sort_keys=True) + "\n")


def active() -> bool:
    return _STATE is not None


def configure(out_dir, rank) -> None:
    """Activate the instrument (the knob consumer's one-time call)."""
    global _STATE
    if _STATE is not None:
        raise RuntimeError(
            f"sga_backend_timer already configured for rank "
            f"{_STATE.rank} ({_STATE.out_dir!r}) — configure once")
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise ValueError(
            f"sga_backend_timer must be a non-empty output directory path "
            f"(got {out_dir!r})")
    _STATE = _TimerState(out_dir, rank)
    print(f"[SGA] backend_timer active: rank {rank} -> "
          f"{_STATE.fh.name} (sync-isolated comm timing; NOT e2e numbers)",
          flush=True)


def set_iter(step) -> None:
    """Engine stamp: the step currently syncing (cheap no-op when off)."""
    if _STATE is not None:
        _STATE.iter = step


def reset() -> None:
    """Close the record file and drop the state (back to off).

    A test / graceful-shutdown helper: production configures once per
    process and never resets. Tests use this instead of poking the
    private ``_STATE``.
    """
    global _STATE
    if _STATE is not None:
        _STATE.fh.close()
        _STATE = None


def begin():
    """Open a timed region: drain the device, start the host clock.

    Returns an opaque token (None when the instrument is off — the
    caller's ``end(None, ...)`` is then a no-op, so the off path costs
    one attribute check per call site).
    """
    if _STATE is None:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def end(token, phase, **fields) -> None:
    """Close a timed region and append its record.

    The region's work is drained via an event on the CALLING stream: the
    backends order their kernel streams back onto the calling stream
    before returning, so a calling-stream event placed after the region
    observes all of it — without a device-wide sync waiting on unrelated
    streams.
    """
    if token is None:
        return
    if torch.cuda.is_available():
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        ev.synchronize()
    wall_ms = (time.perf_counter() - token) * 1e3
    record = {"phase": phase, "wall_ms": round(wall_ms, 6),
              "rank": _STATE.rank, "iter": _STATE.iter, "seq": _STATE.seq,
              "ts": round(time.time(), 6)}
    if sys.exc_info()[0] is not None:
        # the region raised (call sites run end() in a finally) — the
        # field is present ONLY on error: a partial sample can't pass
        # for a good one
        record["error"] = True
    record.update(fields)
    _STATE.write(record)


def _wrap_finish_grad_sync(chunk_cls) -> None:
    """Class-level wrap of ``finish_grad_sync`` (the fullsync phase point).

    Class-level (not per instance): the dense arm has no engine to walk
    the chunks with, and both callers of the chunk-level method — the
    stock training loop and the engine's boundary delegation — go through
    the same class attribute. Idempotent via the marker attribute.
    """
    marker = "_sga_backend_timer"
    orig = getattr(chunk_cls, "finish_grad_sync")
    if getattr(orig, marker, False):
        return

    @functools.wraps(orig)
    def finish_grad_sync(self, *args, **kwargs):
        ddp_config = getattr(self, "ddp_config", None)
        if getattr(ddp_config, "overlap_grad_reduce", False):
            raise RuntimeError(
                "sga_backend_timer refuses overlap_grad_reduce=true: "
                "overlapped grad sync bypasses finish_grad_sync, so the "
                "fullsync phase would record nothing real")
        token = begin()
        try:
            return orig(self, *args, **kwargs)
        finally:
            end(token, "fullsync", backend="megatron_ddp", bucket=None)

    setattr(finish_grad_sync, marker, True)
    chunk_cls.finish_grad_sync = finish_grad_sync


def install_fullsync_wrap() -> None:
    """Install the fullsync phase point on Megatron's DDP (knob-gated).

    Gated on the timer knob, NOT on sga_enabled: the dense arm runs with
    sga_enabled=false and still needs its fullsync records.
    """
    if _STATE is None:
        return
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    _wrap_finish_grad_sync(DistributedDataParallel)
