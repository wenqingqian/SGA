# Copyright (c) 2026, SGA framework contributors.
"""Engine-side pre-commit trace collector (offline replay data source).

Dumps the optimizer's fp32 main-param view right before the AdamW commit:
per-step gradients ``g`` (post unscale + clip) inside configured windows,
and full ``w/m/v`` state at branch points. Consumed offline by
``python -m sga.replay`` to evaluate oracle score candidates against future
update distortion without touching training semantics. Read-only: the
collector never mutates any tensor consumed by training (world-rank-0
only). The default is fully synchronous — the dump runs at most once per
step inside a configured window and ``sga_trace_async`` (opt-in,
default false) selects a background writer. Its default ``cpu_snapshot``
backend keeps the blocking independent CPU snapshot semantics.
The separate explicit ``pinned_cuda`` backend submits nonblocking D2H directly
from immutable dump-slot gradients into bounded reusable pinned buffers on a
dedicated CUDA stream, synchronizes that stream before optimizer mutation, and
then overlaps one or more atomic file writers with subsequent training. The
pinned_cuda pool is sized at the first traced step (a cap too small for one
slot per writer fails loudly there, not at startup), and the gradient layout
and single CUDA device are frozen from that first dump. Catalog and
branch-point state files always stay synchronous.
The writer preserves state-before-gradient publication, bounds queued
snapshots independently of synchronous state saves, and records native
per-phase timers.
Format is schema-compatible with the legacy in-fork collector (catalog.pt /
step_NNNNNN_state.pt / step_NNNNNN_g.pt): mutually readable, with three
ADDITIVE extensions the legacy replay simply ignores — catalog entries
carry ``param_group`` and the meta carries ``param_groups`` (per-group
lr/betas/eps/wd), letting the new replay resolve exact per-group
hyperparameters, and the catalog carries a top-level ``g_dtype``
storage-dtype tag for the per-step g payloads (fp32|bf16; the runtime bf16
dump's self-describing marker, enforced by the resume guard below and read
by ``sga.trace_convert``). Catalog keys are canonical parameter names with
every leading ``module.`` wrapper stripped; the engine derives names from
the attached chunks and reads the grad norm from its own clip.

Numbering is the fork's ABSOLUTE iteration (0-indexed at the optimizer-step
call point, matching the legacy local counter for runs starting from
scratch: dump at step t is the pre-commit state with exactly t commits
applied, and g at step t is consumed by commit t+1). Absolute numbering
fixes the legacy resume-overwrite hazard (a resumed run's local counter
restarted at 0 and silently overwrote early dumps; now a resumed run's
dumps continue at its resume iteration and never collide). A found-inf
skipped step leaves a GAP in the g-file sequence; the replay's contiguous
window discovery stops there instead of silently renumbering.

Dump slot (frozen ordering contract): in ``SGAEngine.optimizer_step``
AFTER step 3 (count_zeros) and BEFORE step 4 (the optimizer template owns
the inner step
and rewrites main_param.grad). At that point prepare_grads + the masked
clip have run, so main_param.grad is the post-unscale post-clip gradient —
exactly the legacy hook's view at the top of step_with_ready_grads. On
sparse steps that gradient holds reduced values only at support coords
(masked clip semantics); replay-grade traces come from full/dense (shadow)
runs where every coord is reduced — the format does not distinguish.
Compact-emission sparse steps: the values are scattered back
into the full layout for the dump window and re-zeroed right after, so
the dump reads the same view the writeback path produced; outside the
dump window the unselected coords hold the raw accumulated residual
(pack clears only the communicated positions) — same semantic as the
writeback path, nothing dump-visible changes.

Enabled by ``sga_trace_dir`` (yaml ``sga:`` section); absent (default)
disables collection entirely: the engine holds ``_trace = None`` and the
slot costs one None-check per step (zero-overhead-when-off, asserted
structurally in tests).
"""

import atexit
import math
import os
import re
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import torch

# The async publication protocol's temporary-file spelling:
# <target>.tmp.<pid>.<seq> — same directory, unique per writing process.
_TMP_SUFFIX_RE = re.compile(r"\.tmp\.(\d+)\.(\d+)$")

# Best-effort atexit drain bound: the success path drains inline at the
# last target step, so this only bounds the early-exception exit (a hung
# writer must not turn a crashed run into a stuck process).
_ATEXIT_CLOSE_TIMEOUT_S = 300.0

# torch.save streams tensor storage rather than cloning a full payload.  Bound
# its non-payload bookkeeping/serialization allowance per in-flight item.
_SERIALIZATION_TRANSIENT_CAP_BYTES = 256 << 20


def parse_branch_points(spec) -> List[int]:
    """Normalize a branch-point spec (comma string or int list) into a
    sorted list of unique non-negative ints. Empty/None -> []. Non-integral
    floats are rejected: int() would silently truncate 10.5 to 10 and land
    the window on the wrong step."""
    if spec is None:
        return []
    if isinstance(spec, str):
        toks = [t for t in spec.replace(" ", "").split(",") if t]
    else:
        toks = list(spec)
    points = set()
    for tok in toks:
        if isinstance(tok, float) and not tok.is_integer():
            raise ValueError(
                f"trace branch point must be an integer, got {tok!r}"
            )
        try:
            t = int(tok)
        except (TypeError, ValueError):
            raise ValueError(
                f"trace branch point must be an integer, got {tok!r}"
            ) from None
        if t < 0:
            raise ValueError(f"trace branch point must be >= 0, got {t}")
        points.add(t)
    return sorted(points)


def _canonical_name(name: str) -> str:
    """Strip every leading ``module.`` wrapper segment (DDP / Float16Module)
    so catalog keys do not depend on the wrapping stack."""
    while name.startswith("module."):
        name = name[len("module."):]
    return name


def _dump_phase(engine) -> str:
    """Engine mode at dump time: shadow, or the live step's sync phase."""
    if getattr(engine, "_shadow", False):
        return "shadow"
    return "live_full" if getattr(engine, "_step_full", None) else "live_sparse"


def _catalog_mismatch(old, expected: dict):
    """First offending (name, field) between an on-disk catalog and the one
    this run would write — key set plus shape/dtype/param_group per entry —
    or None when identical. The resume identity anchor: a catalog from a
    different model/config would otherwise silently combine with new step
    payloads and poison replay."""
    old = old if isinstance(old, dict) else {}
    if set(old) != set(expected):
        for name in expected:
            if name not in old:
                return name, "name"
        for name in old:
            if name not in expected:
                return name, "name"
    for name, ent in expected.items():
        for field in ("shape", "dtype", "param_group"):
            if old[name].get(field) != ent[field]:
                return name, field
    return None


def _values_match(a, b) -> bool:
    """Recursive meta equality with NaN==NaN (grad_norm legitimately
    records NaN; plain ``==`` would refuse a bitwise-identical resume)."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_match(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _values_match(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _process_alive(pid: int) -> bool:
    """os.kill(pid, 0) liveness probe: ESRCH = dead, EPERM = alive but
    foreign-owned; anything else (bad pid spelling) reads as alive — the
    conservative direction for tmp cleanup (never delete what might be
    another live writer's in-flight file)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _record_phase(stats: dict, phase: str, elapsed: float) -> None:
    """Fold one phase measurement into its {n, s, max_s} triple. The keys
    are pre-initialized in the stats dict (a stable diagnostics shape).
    Best-effort: a concurrent writer may lose a race on a counter —
    diagnostics only, never control flow."""
    stats[f"{phase}_n"] += 1
    stats[f"{phase}_s"] += float(elapsed)
    if elapsed > stats[f"{phase}_max_s"]:
        stats[f"{phase}_max_s"] = float(elapsed)


class _PendingWrite:
    """One accepted g item on its way to disk."""

    __slots__ = ("step", "meta", "payload", "reservation", "seq",
                 "release_slot")

    def __init__(self, step, meta, payload, reservation, seq,
                 release_slot=None):
        self.step = step
        self.meta = meta
        self.payload = payload
        self.reservation = reservation
        self.seq = seq
        self.release_slot = release_slot


class _PinnedCudaSlot:
    """Reusable per-item pinned host buffers, shape-keyed by parameter name."""

    __slots__ = ("payload",)

    def __init__(self, entries, output_dtype=None):
        self.payload = {}
        for name, grad in entries:
            # Async D2H requires a pinned destination; the copy stream is
            # synchronized before this storage is handed to the writer.
            dtype = output_dtype or grad.dtype
            tensor = torch.empty(grad.shape, dtype=dtype, device="cpu",
                                 pin_memory=True)
            if not tensor.is_pinned():
                raise RuntimeError(
                    "trace pinned_cuda failed to allocate pinned memory")
            self.payload[name] = tensor


class _AsyncGradWriter:
    """The background writer publishing ``step_NNNNNN_g.pt`` files
    (constructed lazily, world-rank-0 only).

    The enqueue face reserves an exact caller-computed transient byte budget,
    then submits an independent CPU snapshot.  Reservation blocks on depth or
    bytes and re-raises the latched writer exception.  One or more daemon
    workers atomically publish independent step files; the first exception is
    latched and the queue fails closed.

    ``_cv`` guards: ``_queue``, ``_reserved`` (bytes), ``_inflight``
    (accepted-but-unfinished items — the depth budget counts queue + the
    item under a writer's pen), ``_error``, ``_closed``. ``stats``
    diagnostics are updated by the caller and every worker (best-effort
    counters; a concurrent update may be lost); a snapshot for reporting
    is taken under the lock.

    ``stats`` phase timers are async-only and expose ``<name>_n``,
    ``<name>_s`` and ``<name>_max_s`` for ``snapshot``, ``d2h_submit``,
    ``d2h_sync``, ``slot_wait``, ``cast``, ``write`` and ``drain``. Queue
    wait uses the ``backpressure_*`` counters.
    """

    def __init__(self, dir: str, queue_depth: int, max_workset_bytes: int,
                 worker_count: int = 1):
        if (not isinstance(worker_count, int)
                or isinstance(worker_count, bool) or worker_count < 1):
            raise ValueError(
                f"trace async writer worker_count must be an int >= 1 "
                f"(got {worker_count!r})")
        self._dir = dir
        self._max_depth = int(queue_depth)
        self._cap = int(max_workset_bytes)
        self._worker_count = int(worker_count)
        self._cv = threading.Condition(threading.Lock())
        self._queue = deque()
        self._reserved = 0
        self._inflight = 0
        self._error = None              # the latch (first writer exception)
        self._error_raised = False      # close raises it exactly once
        self._closed = False
        self._threads = []
        self._seq = 0
        self.stats = {
            "enqueued": 0, "written": 0, "written_bytes": 0,
            "skipped_existing": 0, "backpressure_waits": 0,
            "backpressure_wait_s": 0.0, "peak_reserved_bytes": 0,
            "peak_inflight": 0, "error": None, "error_step": None,
            # native phase timers ({n, s, max_s} per phase; async-only)
            "snapshot_n": 0, "snapshot_s": 0.0, "snapshot_max_s": 0.0,
            "d2h_submit_n": 0, "d2h_submit_s": 0.0,
            "d2h_submit_max_s": 0.0,
            "d2h_sync_n": 0, "d2h_sync_s": 0.0,
            "d2h_sync_max_s": 0.0,
            "slot_wait_n": 0, "slot_wait_s": 0.0,
            "slot_wait_max_s": 0.0,
            "cast_n": 0, "cast_s": 0.0, "cast_max_s": 0.0,
            "write_n": 0, "write_s": 0.0, "write_max_s": 0.0,
            "drain_n": 0, "drain_s": 0.0, "drain_max_s": 0.0,
        }

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Clean this protocol's stale tmp files, then spawn the writer."""
        self._cleanup_stale_tmp()
        self._threads = [threading.Thread(
            target=self._writer_loop,
            name=f"sga-trace-async-writer-{i}", daemon=True)
            for i in range(self._worker_count)]
        for thread in self._threads:
            thread.start()
        # Best-effort close for training-side early exceptions: the success
        # path drains inline at the last target step, so this fires only
        # when training is already unwinding — it must not mask anything
        # (print-only) nor hang the exit (bounded join).
        atexit.register(self._atexit_close)

    def _cleanup_stale_tmp(self) -> None:
        """Startup hygiene: remove ONLY this protocol's orphaned tmp files
        (``*.tmp.<pid>.<seq>`` whose owner pid is dead). A live owner's tmp
        is another run's in-flight write — untouched; anything not matching
        the pattern is not ours to judge — untouched."""
        try:
            entries = os.listdir(self._dir)
        except OSError:
            return  # no directory yet: nothing stale
        for name in entries:
            m = _TMP_SUFFIX_RE.search(name)
            if m is None:
                continue
            pid = int(m.group(1))
            if pid == os.getpid() or _process_alive(pid):
                continue
            try:
                os.remove(os.path.join(self._dir, name))
            except OSError:
                pass

    # ---------------------------------------------------------- enqueue face

    def check_open(self) -> None:
        """Raise a latched writer error or closed-state failure."""
        with self._cv:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError("trace async writer is closed")

    def reserve(self, reservation_bytes: int) -> None:
        """Claim one item slot and an exact caller-computed byte budget."""
        reservation = int(reservation_bytes)
        with self._cv:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError(
                    "trace async writer is closed (enqueue after the final "
                    "drain — a later window than the configured branch "
                    "points?)")
            if reservation > self._cap:
                raise ValueError(
                    f"trace async: single item reservation {reservation} "
                    f"bytes exceeds the max workset cap {self._cap} bytes "
                    f"(sga_trace_async_max_workset_gib) — refusing before "
                    "the snapshot allocation (no oversized bypass); raise "
                    "the cap or disable sga_trace_async")
            waited = False
            t0 = 0.0
            while True:
                if self._error is not None:
                    raise self._error
                if (self._inflight < self._max_depth
                        and self._reserved + reservation <= self._cap):
                    break
                if not waited:
                    waited = True
                    t0 = time.perf_counter()
                self._cv.wait()
            if waited:
                self.stats["backpressure_waits"] += 1
                self.stats["backpressure_wait_s"] += time.perf_counter() - t0
            self._reserved += reservation
            self._inflight += 1
            if self._reserved > self.stats["peak_reserved_bytes"]:
                self.stats["peak_reserved_bytes"] = self._reserved
            if self._inflight > self.stats["peak_inflight"]:
                self.stats["peak_inflight"] = self._inflight

    def release(self, reservation_bytes: int) -> None:
        """Free a reservation whose snapshot was never enqueued."""
        with self._cv:
            self._reserved -= int(reservation_bytes)
            self._inflight -= 1
            self._cv.notify_all()

    def submit(self, step: int, meta: dict, payload: dict,
               reservation_bytes: int, *, release_slot=None) -> None:
        """Hand a completed independent CPU snapshot to the writer."""
        with self._cv:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError(
                    "trace async writer is closed (submit after the final "
                    "drain)")
            self._seq += 1
            self._queue.append(_PendingWrite(
                step, meta, payload, int(reservation_bytes), self._seq,
                release_slot=release_slot))
            self.stats["enqueued"] += 1
            self._cv.notify_all()

    def existing_matches(self, step: int, meta: dict, payload: dict) -> bool:
        """Resume-safe publication check (caller thread, before submit —
        synchronous, outside the writer's async budget):
        an existing target with bitwise-equal payload and matching meta is
        skipped; anything else (unreadable, different content) refuses the
        overwrite — a rerun must name an empty directory."""
        path = os.path.join(self._dir, f"step_{step:06d}_g.pt")
        if not os.path.exists(path):
            return False
        try:
            on_disk = torch.load(path, weights_only=False)
        except Exception as e:  # unreadable/corrupt: refuse, never guess
            raise ValueError(
                f"trace async: {os.path.basename(path)} already exists but "
                f"cannot be read ({e}); rerun into an empty directory"
            ) from None
        disk_meta = on_disk.get("meta") if isinstance(on_disk, dict) else None
        disk_payload = (on_disk.get("payload")
                        if isinstance(on_disk, dict) else None)
        if not isinstance(disk_meta, dict) or not isinstance(disk_payload,
                                                             dict):
            raise ValueError(
                f"trace async: {os.path.basename(path)} already exists with "
                "an unrecognized layout; rerun into an empty directory")
        if not _values_match(meta, disk_meta):
            raise ValueError(
                f"trace async: {os.path.basename(path)} already exists with "
                "different metadata; rerun into an empty directory")
        if set(disk_payload) != set(payload):
            raise ValueError(
                f"trace async: {os.path.basename(path)} already exists with "
                "a different parameter set; rerun into an empty directory")
        for name, tensor in payload.items():
            if not torch.equal(disk_payload[name], tensor):
                raise ValueError(
                    f"trace async: {os.path.basename(path)} already exists "
                    f"with different values for '{name}'; rerun into an "
                    "empty directory")
        self.stats["skipped_existing"] += 1
        return True

    # ----------------------------------------------------------- writer face

    def _writer_loop(self) -> None:
        while True:
            with self._cv:
                while (not self._queue and self._error is None
                        and not self._closed):
                    self._cv.wait()
                if self._error is not None:
                    # One worker owns fail-closed backlog cleanup; others exit.
                    if self._queue:
                        while self._queue:
                            self._release_item_locked(self._queue.popleft())
                    self._cv.notify_all()
                    return
                if not self._queue:
                    self._cv.notify_all()
                    return
                item = self._queue.popleft()
            self._write_one(item)

    def _write_one(self, item: _PendingWrite) -> None:
        path = os.path.join(self._dir, f"step_{item.step:06d}_g.pt")
        tmp = f"{path}.tmp.{os.getpid()}.{item.seq}"
        t0 = time.perf_counter()
        try:
            torch.save({"meta": item.meta, "payload": item.payload}, tmp)
            os.replace(tmp, path)
        except BaseException as e:
            # a failed write still accounts its elapsed time, then latches
            _record_phase(self.stats, "write", time.perf_counter() - t0)
            try:
                os.remove(tmp)  # never leave our own partial behind
            except OSError:
                pass
            with self._cv:
                if self._error is None:
                    self._error = e  # latch the ORIGINAL exception object
                    self.stats["error"] = repr(e)
                    self.stats["error_step"] = item.step
                    # fail-closed: the queued backlog is dropped (an item
                    # already under a writer's pen may still publish)
                    while self._queue:
                        self._release_item_locked(self._queue.popleft())
                self._release_item_locked(item)
                self._cv.notify_all()
            return
        _record_phase(self.stats, "write", time.perf_counter() - t0)
        try:
            written = os.path.getsize(path)
        except OSError:
            written = item.reservation
        if item.release_slot is not None:
            item.release_slot()
            item.release_slot = None
        with self._cv:
            self.stats["written"] += 1
            self.stats["written_bytes"] += written
            self._release_item_locked(item)
            self._cv.notify_all()

    def _release_item_locked(self, item: _PendingWrite) -> None:
        if item.release_slot is not None:
            item.release_slot()
            item.release_slot = None
        self._reserved -= item.reservation
        self._inflight -= 1

    # ---------------------------------------------------------------- close

    def close(self, timeout: Optional[float] = None,
              raise_error: bool = True) -> dict:
        """Idempotent drain: mark closed, wake the writer, join it (None =
        unbounded — the success path must not return before every target
        file is on disk), then re-raise the latched writer error exactly
        once. An actual wait (the writer still alive at entry) is recorded
        in the ``drain`` phase timer; an idempotent no-op close is not.
        Returns the diagnostics snapshot."""
        with self._cv:
            first = not self._closed
            self._closed = True
            self._cv.notify_all()
        current = threading.current_thread()
        waited_threads = [t for t in self._threads
                          if t is not current and t.is_alive()]
        if waited_threads:
            t0 = time.perf_counter()
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in self._threads:
            if thread is current:
                continue
            remaining = (None if deadline is None else
                         max(0.0, deadline - time.monotonic()))
            thread.join(remaining)
        if waited_threads:
            _record_phase(self.stats, "drain", time.perf_counter() - t0)
        alive = [t.name for t in self._threads if t is not current and t.is_alive()]
        if alive:
            msg = (f"trace async writer did not drain within {timeout}s — "
                   f"workers still alive: {alive}; target files may be missing")
            if raise_error:
                raise RuntimeError(msg)
            print(f"[SGA] {msg}", flush=True)
        with self._cv:
            latched = self._error
            do_raise = (raise_error and latched is not None
                        and not self._error_raised)
            if do_raise:
                self._error_raised = True
            stats = dict(self.stats)
        if first and self._threads:
            peak = stats["peak_reserved_bytes"] / (1 << 20)
            cap_mib = self._cap / (1 << 20)
            print(f"[SGA] trace async writer closed: {stats['written']} "
                  f"items, {stats['written_bytes'] / (1 << 20):.1f} MiB, "
                  f"peak workset {peak:.1f} MiB of {cap_mib:.0f} "
                  f"MiB cap, {stats['backpressure_waits']} backpressure "
                  f"waits ({stats['backpressure_wait_s']:.3f}s), "
                  f"{stats['skipped_existing']} skipped-existing, "
                  f"phases snapshot={stats['snapshot_s']:.3f}s/"
                  f"d2h_submit={stats['d2h_submit_s']:.3f}s/"
                  f"d2h_sync={stats['d2h_sync_s']:.3f}s/"
                  f"slot_wait={stats['slot_wait_s']:.3f}s/"
                  f"cast={stats['cast_s']:.3f}s/"
                  f"write={stats['write_s']:.3f}s/"
                  f"drain={stats['drain_s']:.3f}s, "
                  f"error={stats['error']!r}", flush=True)
        if do_raise:
            raise latched
        return stats

    def _atexit_close(self) -> None:
        """Best-effort bounded close for the early-exception exit path:
        never raises (the training error already owns the exit), never
        hangs (bounded join), and reports a writer failure the process
        never surfaced (a close/drain raise already did its job — the
        exit stays quiet)."""
        try:
            stats = self.close(timeout=_ATEXIT_CLOSE_TIMEOUT_S,
                               raise_error=False)
        except Exception as e:  # pragma: no cover — defensive
            print(f"[SGA] trace async writer atexit close failed: {e!r}",
                  flush=True)
            return
        if (stats.get("error") is not None
                and not self._error_raised):
            print(f"[SGA] trace async writer failed earlier (latched, "
                  f"step {stats.get('error_step')}): {stats['error']!r}",
                  flush=True)


class TraceCollector:
    """Per-engine trace state; constructed only when sga_trace_dir is set.

    The engine passes the canonical values of ``sga_trace_dir`` /
    ``sga_trace_branch_points`` / ``sga_trace_window`` /
    ``sga_trace_dump_dtype`` and the async knobs straight through
    (already typed and validated; ``parse_branch_points`` below is the
    idempotent guard for direct construction)."""

    def __init__(self, dir: str, branch_points, window: int,
                 dump_dtype: str = "fp32", async_enabled: bool = False,
                 async_backend: str = "cpu_snapshot",
                 async_queue_depth: int = 2,
                 async_worker_count: int = 1,
                 async_max_workset_gib: float = 16.0):
        self.dir = dir
        self.branch_points = parse_branch_points(branch_points)
        self.window = int(window)
        if dump_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"dump_dtype must be 'fp32' or 'bf16' (got {dump_dtype!r})")
        self.dump_dtype = dump_dtype
        self._g_dtype = torch.bfloat16 if dump_dtype == "bf16" else None
        self.async_enabled = bool(async_enabled)
        if async_backend not in ("cpu_snapshot", "pinned_cuda"):
            raise ValueError(
                "async_backend must be 'cpu_snapshot' or 'pinned_cuda' "
                f"(got {async_backend!r})")
        if async_backend != "cpu_snapshot" and not self.async_enabled:
            raise ValueError(
                "async_backend='pinned_cuda' requires async_enabled=True")
        self.async_backend = async_backend
        # Direct-construction backstops (the engine-test route seeds config
        # without validation; same discipline as the dump_dtype guard).
        if (not isinstance(async_queue_depth, int)
                or isinstance(async_queue_depth, bool)
                or async_queue_depth < 1):
            raise ValueError(
                f"async_queue_depth must be an int >= 1 "
                f"(got {async_queue_depth!r})")
        if (not isinstance(async_max_workset_gib, (int, float))
                or isinstance(async_max_workset_gib, bool)
                or not math.isfinite(async_max_workset_gib)
                or async_max_workset_gib <= 0.0):
            raise ValueError(
                f"async_max_workset_gib must be a finite float > 0 "
                f"(got {async_max_workset_gib!r})")
        self.async_queue_depth = int(async_queue_depth)
        if (not isinstance(async_worker_count, int)
                or isinstance(async_worker_count, bool)
                or async_worker_count < 1):
            raise ValueError(
                f"async_worker_count must be an int >= 1 "
                f"(got {async_worker_count!r})")
        if async_worker_count > self.async_queue_depth:
            raise ValueError(
                "async_worker_count must be <= async_queue_depth "
                f"({async_worker_count} > {self.async_queue_depth})")
        self.async_worker_count = int(async_worker_count)
        self.async_max_workset_gib = float(async_max_workset_gib)
        self._async_cap_bytes = int(self.async_max_workset_gib * (1 << 30))
        self._entries: Optional[list] = None   # [(name, rec, group_index)]
        self._catalog_written = False          # per process (resume-safe)
        self._writer = None                    # lazy, async mode, rank 0
        self._drained = False                  # final drain happened
        self._copy_stream = None  # pinned_cuda; one stream, single-device dumps
        self._slot_cv = threading.Condition(threading.Lock())
        self._pinned_slots = []                # bounded reusable host buffers
        self._free_pinned_slots = deque()
        self._slot_signature = None

    def wants(self, step: int):
        """(want_state, want_g) for an absolute engine step."""
        want_state = step in self.branch_points
        want_g = want_state or any(
            t0 <= step < t0 + self.window for t0 in self.branch_points
        )
        return want_state, want_g

    @property
    def _last_g_step(self) -> Optional[int]:
        """The last engine step any g dump targets (None: no windows)."""
        if not self.branch_points:
            return None
        return self.branch_points[-1] + self.window - 1

    # ------------------------------------------------------------------ bind

    def _bind(self, engine, mp_opt) -> None:
        """Resolve (canonical name, record, optimizer group) per record once.

        Names come from the attached chunks' ``module.named_parameters()``
        (no param tagging); the group index from the inner optimizer's
        param_groups. Both maps are identity-keyed and stable for the run.
        """
        name_of: Dict[int, str] = {}
        for chunk in engine._chunks.values():
            for n, p in chunk.module.named_parameters():
                name_of[id(p)] = _canonical_name(n)
        group_of: Dict[int, int] = {}
        for gi, group in enumerate(mp_opt.optimizer.param_groups):
            for p in group["params"]:
                if id(p) in group_of and group_of[id(p)] != gi:
                    raise ValueError(
                        "trace: main param appears in multiple optimizer "
                        f"param_groups ({group_of[id(p)]} and {gi})"
                    )
                group_of[id(p)] = gi
        entries = []
        seen = set()
        for rec in engine._records:
            name = name_of.get(id(rec.model_param))
            if name is None:
                raise ValueError(
                    "trace: model param not named by any attached chunk "
                    f"(shape {tuple(rec.model_param.shape)}); attach() must "
                    "precede the first dump"
                )
            if name in seen:
                raise ValueError(
                    f"trace: duplicate canonical param name '{name}' after "
                    "module.-prefix stripping"
                )
            seen.add(name)
            gi = group_of.get(id(rec.main_param))
            if gi is None:
                raise ValueError(
                    "trace: main param not found in any optimizer param_group "
                    f"(model param shape {tuple(rec.model_param.shape)})"
                )
            entries.append((name, rec, gi))
        self._entries = entries

    # ------------------------------------------------------------------ dump

    def _meta(self, mp_opt, step: int, grad_norm, phase: Optional[str]) -> dict:
        cfg = mp_opt.config
        groups = []
        for g in mp_opt.optimizer.param_groups:
            betas = g.get("betas", (0.9, 0.999))
            groups.append({
                "lr": float(g.get("lr", 0.0)),
                "weight_decay": float(g.get("weight_decay", 0.0)),
                "adam_beta1": float(betas[0]),
                "adam_beta2": float(betas[1]),
                "adam_eps": float(g.get("eps", 1e-8)),
            })
        meta = {
            "step": step,
            # Engine mode at dump time: 'shadow' | 'live_full' | 'live_sparse'.
            # Replay-grade traces come from shadow/dense runs; the field makes
            # a mixed directory self-describing instead of inferrable.
            "phase": phase,
            "lr": [g["lr"] for g in groups],
            "grad_norm": float(grad_norm) if grad_norm is not None else float("nan"),
            "clip_grad": float(getattr(cfg, "clip_grad", 0.0)),
            # legacy global scalars (uniform-run replay fallback; per-group
            # values take precedence when the catalog carries param_group)
            "weight_decay": float(getattr(cfg, "weight_decay", 0.0)),
            "adam_beta1": float(getattr(cfg, "adam_beta1", 0.9)),
            "adam_beta2": float(getattr(cfg, "adam_beta2", 0.999)),
            "adam_eps": float(getattr(cfg, "adam_eps", 1e-8)),
            "param_groups": groups,
        }
        return meta

    @torch.no_grad()
    def maybe_dump(self, engine, mp_opt, step: int, grad_norm) -> None:
        """Dump slot: called once per optimizer_step when tracing is on."""
        want_state, want_g = self.wants(step)
        if not (want_state or want_g):
            # Async drain fence: a found-inf skip of the last target step
            # means no enqueue ever fires for it — the first step past the
            # last target drains the queue here (no-op in sync mode).
            self._final_drain(step)
            return
        if engine._rank != 0:
            return
        records = engine._records or engine._refresh_records(mp_opt)
        if self._entries is None:
            self._bind(engine, mp_opt)

        os.makedirs(self.dir, exist_ok=True)
        meta = self._meta(mp_opt, step, grad_norm, _dump_phase(engine))

        if not self._catalog_written:
            cat_path = os.path.join(self.dir, "catalog.pt")
            catalog = {
                name: {"shape": list(rec.main_param.shape),
                       "dtype": str(rec.main_param.dtype),
                       "param_group": gi}
                for name, rec, gi in self._entries
            }
            if os.path.exists(cat_path):
                # Resume: verify the producing run's identity, never rewrite.
                # With the catalog verified, re-dumping the same absolute
                # steps is a legitimate same-step recomputation after a
                # mid-window crash, so existing step files stay unguarded
                # (the async path re-verifies its own targets bitwise).
                on_disk = torch.load(cat_path, weights_only=False)
                old = on_disk.get("catalog") if isinstance(on_disk, dict) else None
                mismatch = _catalog_mismatch(old, catalog)
                if (mismatch is None and on_disk.get("g_dtype", "fp32")
                        != self.dump_dtype):
                    mismatch = ("g_dtype",
                                f"{on_disk.get('g_dtype', 'fp32')} on disk "
                                f"vs {self.dump_dtype} this run")
                if mismatch is not None:
                    raise ValueError(
                        "trace: existing catalog.pt in "
                        f"{self.dir} does not match this run "
                        f"({mismatch[1]} for {mismatch[0]})"
                    )
            else:
                torch.save({"meta": meta, "catalog": catalog,
                            "g_dtype": self.dump_dtype}, cat_path)
            self._catalog_written = True

        if want_state:
            self._dump_state(step, meta)
        if want_g:
            self._dump_grad(step, meta)
        # The last target gradient enqueued: drain inline — training may
        # not proceed past this step until every g file is on disk.
        self._final_drain(step)

    def _dump_state(self, step: int, meta: dict) -> None:
        """Dump w/m/v (+ g) at a branch point. Missing moments (lazy AdamW
        state before the first commit) are dumped as zeros and flagged."""
        state_absent = False
        payload = {"w": {}, "m": {}, "v": {}, "g": {}}
        for name, rec, _gi in self._entries:
            payload["w"][name] = rec.main_param.data.detach().cpu()
            if rec.main_param.grad is not None:
                payload["g"][name] = rec.main_param.grad.detach().cpu()
            st = rec._state or {}
            if "exp_avg" in st and "exp_avg_sq" in st:
                payload["m"][name] = st["exp_avg"].detach().cpu()
                payload["v"][name] = st["exp_avg_sq"].detach().cpu()
            else:
                state_absent = True
                payload["m"][name] = torch.zeros_like(payload["w"][name])
                payload["v"][name] = torch.zeros_like(payload["w"][name])
        step_meta = dict(meta)
        step_meta["state_absent"] = state_absent
        torch.save({"meta": step_meta, "payload": payload},
                   os.path.join(self.dir, f"step_{step:06d}_state.pt"))

    def _dump_grad(self, step: int, meta: dict) -> None:
        # g_dtype=bfloat16 stores the per-tensor round-to-nearest bf16 value
        # — bit-identical to converting an fp32 dump with sga.trace_convert
        # (same .to(bfloat16) on the same fp32 source), so a runtime-bf16
        # trace and a converted trace are interchangeable replay inputs.
        if not self.async_enabled:
            payload = {}
            for name, rec, _gi in self._entries:
                if rec.main_param.grad is not None:
                    g = rec.main_param.grad.detach().cpu()
                    if self._g_dtype is not None:
                        g = g.to(self._g_dtype)
                    payload[name] = g
            torch.save({"meta": dict(meta), "payload": payload},
                       os.path.join(self.dir, f"step_{step:06d}_g.pt"))
            return
        if self.async_backend == "pinned_cuda":
            self._dump_grad_pinned_cuda(step, meta)
        else:
            self._dump_grad_async(step, meta)

    def _dump_grad_async(self, step: int, meta: dict) -> None:
        """Async g dump: reserve BEFORE any allocation, take the blocking
        independent CPU snapshot on the main thread (timed into the
        ``snapshot`` phase, its bf16 cast into ``cast``), then hand
        serialization+write to the background writer(s). Nothing the queue holds
        aliases training memory, and the payload value semantics are
        identical to the sync path above."""
        entries = [(name, rec) for name, rec, _gi in self._entries
                   if rec.main_param.grad is not None]
        # fp32 source-snapshot size, from shapes only — computed BEFORE the
        # copy so an oversized item dies before allocating anything (the
        # bf16 cast transient is covered by the conservative 2x margin).
        snapshot_nbytes = sum(
            rec.main_param.grad.numel() * rec.main_param.grad.element_size()
            for _name, rec in entries)
        reservation_bytes = 2 * snapshot_nbytes
        writer = self._ensure_writer()
        writer.reserve(reservation_bytes)
        submitted = False
        try:
            payload = {}
            snap_s = 0.0
            cast_s = 0.0
            for name, rec in entries:
                # to("cpu", copy=True): on GPU this is the one blocking D2H
                # copy (.cpu() would copy anyway); on CPU it forces the
                # independent copy .cpu() would alias away — the queued
                # tensor must never share storage with training state.
                t0 = time.perf_counter()
                g = rec.main_param.grad.detach().to("cpu", copy=True)
                snap_s += time.perf_counter() - t0
                if self._g_dtype is not None:
                    t1 = time.perf_counter()
                    g = g.to(self._g_dtype)
                    cast_s += time.perf_counter() - t1
                payload[name] = g
            _record_phase(writer.stats, "snapshot", snap_s)
            if self._g_dtype is not None:
                _record_phase(writer.stats, "cast", cast_s)
            if not writer.existing_matches(step, dict(meta), payload):
                writer.submit(step, dict(meta), payload, reservation_bytes)
                submitted = True
        finally:
            if not submitted:
                writer.release(reservation_bytes)

    def _acquire_pinned_slot(self, entries, writer):
        output_dtype = self._g_dtype or entries[0][1].dtype
        signature = tuple((name, tuple(src.shape), output_dtype)
                          for name, src in entries)
        wait0 = time.perf_counter()
        waited = False
        with self._slot_cv:
            if self._slot_signature is None:
                output_bytes = sum(src.numel() * output_dtype.itemsize
                                   for _name, src in entries)
                transient_bytes = min(
                    output_bytes, _SERIALIZATION_TRANSIENT_CAP_BYTES)
                max_slots = ((self._async_cap_bytes -
                              self.async_queue_depth * transient_bytes)
                             // output_bytes)
                wanted = min(self.async_queue_depth,
                             self.async_worker_count + 1)
                slot_count = min(wanted, max_slots)
                if slot_count < self.async_worker_count:
                    raise ValueError(
                        "trace pinned_cuda workset cannot hold one pinned "
                        "slot per writer plus bounded queue serialization "
                        "transients within sga_trace_async_max_workset_gib")
                self._pinned_slots = [
                    _PinnedCudaSlot(entries, output_dtype=self._g_dtype)
                    for _ in range(slot_count)]
                self._free_pinned_slots.extend(self._pinned_slots)
                self._slot_signature = signature
            elif signature != self._slot_signature:
                raise RuntimeError(
                    "trace pinned_cuda gradient layout changed after the "
                    "buffer pool was created")
        while True:
            with self._slot_cv:
                if self._free_pinned_slots:
                    slot = self._free_pinned_slots.popleft()
                    break
                waited = True
                self._slot_cv.wait(timeout=0.1)
            # Check outside _slot_cv: writer failure cleanup returns slots
            # while holding its own condition, so nested locks here would
            # invert that order and deadlock.
            writer.check_open()
        if waited:
            _record_phase(writer.stats, "slot_wait",
                          time.perf_counter() - wait0)
        return slot

    def _release_pinned_slot(self, slot) -> None:
        with self._slot_cv:
            self._free_pinned_slots.append(slot)
            self._slot_cv.notify_all()

    def _dump_grad_pinned_cuda(self, step: int, meta: dict) -> None:
        """Synchronously complete pinned D2H, then enqueue the host snapshot.

        Gradients remain immutable until this dump slot returns.  A dedicated
        copy stream waits for the producer stream, copies directly from those
        gradients into a reusable pinned slot, and is synchronized before the
        optimizer may mutate them.  File serialization then overlaps later
        training. BF16 is cast on the GPU before D2H; parity is test-pinned.
        """
        entries = [(name, rec.main_param.grad.detach())
                   for name, rec, _gi in self._entries
                   if rec.main_param.grad is not None]
        if not entries:
            return self._dump_grad_async(step, meta)
        if any(not grad.is_cuda for _name, grad in entries):
            raise RuntimeError(
                "trace async backend 'pinned_cuda' requires CUDA gradients")
        devices = {grad.device for _name, grad in entries}
        if len(devices) != 1:
            raise RuntimeError(
                "trace async backend 'pinned_cuda' requires one CUDA device")
        writer = self._ensure_writer()
        slot = self._acquire_pinned_slot(entries, writer)
        output_bytes = sum(t.numel() * t.element_size()
                           for t in slot.payload.values())
        # Pinned slots are persistent pool storage. Per-item reservation
        # covers the bounded serialization transient only; the pool itself
        # was sized against the same global cap at first-slot acquisition.
        transient_bytes = min(output_bytes,
                              _SERIALIZATION_TRANSIENT_CAP_BYTES)
        submitted = False
        reserved = False
        try:
            writer.reserve(transient_bytes)
            reserved = True
            device = next(iter(devices))
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=device)
            producer = torch.cuda.current_stream(device=device)
            submit0 = time.perf_counter()
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_stream(producer)
                for name, grad in entries:
                    src = (grad.to(self._g_dtype)
                           if self._g_dtype is not None else grad)
                    slot.payload[name].copy_(src, non_blocking=True)
            _record_phase(writer.stats, "d2h_submit",
                          time.perf_counter() - submit0)
            sync0 = time.perf_counter()
            self._copy_stream.synchronize()
            _record_phase(writer.stats, "d2h_sync",
                          time.perf_counter() - sync0)
            if not writer.existing_matches(step, dict(meta), slot.payload):
                writer.submit(
                    step, dict(meta), slot.payload, transient_bytes,
                    release_slot=lambda s=slot: self._release_pinned_slot(s))
                submitted = True
        finally:
            if not submitted:
                self._release_pinned_slot(slot)
                if reserved:
                    writer.release(transient_bytes)

    def _ensure_writer(self) -> _AsyncGradWriter:
        """Lazy writer construction (rank 0, async mode, first g
        dump only — a traced run with no window never spawns it)."""
        if self._writer is None:
            self._writer = _AsyncGradWriter(
                self.dir, self.async_queue_depth,
                self._async_cap_bytes, self.async_worker_count)
            self._writer.start()
        return self._writer

    def _final_drain(self, step: int) -> None:
        """Inline drain once the last target gradient step has passed
        (idempotent; sync mode and non-target early steps are no-ops)."""
        writer = self._writer
        if writer is None or self._drained:
            return
        if self._last_g_step is not None and step >= self._last_g_step:
            self._drained = True
            writer.close()  # unbounded: raises the latched error fail-closed

    def close(self, timeout: Optional[float] = None) -> dict:
        """Idempotent drain of the async writer (no-op in sync mode).
        Returns the writer's diagnostics snapshot ({} in sync mode)."""
        if self._writer is None:
            return {}
        return self._writer.close(timeout=timeout)

    def diagnostics(self) -> dict:
        """The async writer's diagnostics snapshot ({} in sync mode) —
        items/bytes written, peak workset, backpressure waits, latched
        error; reportable from tests and harnesses."""
        writer = self._writer
        if writer is None:
            return {}
        with writer._cv:
            return dict(writer.stats)
