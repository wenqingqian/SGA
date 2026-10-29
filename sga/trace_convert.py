# Copyright (c) 2026, SGA framework contributors.
"""One-way trace converter: build a bf16 screening copy of a trace window.

Replay IO is dominated by the per-step g dumps (one fp32 gradient per model
parameter per step). A bf16 copy halves those bytes at the cost of rounding
the logged gradients to bf16; replays on the copy run with ``python -m
sga.replay --trace-dtype bf16`` (payloads upcast to fp32 at load, replay
arithmetic unchanged). The source trace is never modified.

catalog.pt and the step_NNNNNN_state.pt dumps are copied verbatim except the
catalog's self-describing ``g_dtype`` tag, which flips to bf16: they hold
the replay branch states (w/m/v), which stay fp32 — only the per-step
gradients belong to the IO-halving being screened.

Usage (SGA repo root on PYTHONPATH):

    python -m sga.trace_convert --src <trace-dir> --dst <dst-dir> \
        [--from-step S] [--to-step E] [--overwrite]

S/E select the half-open step range [S, E) (default: every g/state file in
the trace).
"""
import argparse
import os
import re
import shutil
import sys

import torch

_G_RE = re.compile(r"step_(\d+)_g\.pt$")
_S_RE = re.compile(r"step_(\d+)_state\.pt$")


def convert(src, dst, from_step=None, to_step=None, overwrite=False,
            log=None):
    """Copy ``src`` into a new ``dst`` dir with the g payloads of steps in
    [from_step, to_step) rounded to bf16; state files verbatim (fp32), the
    catalog verbatim except the self-describing ``g_dtype`` flip to bf16.
    Returns (n_g, n_state, src_bytes, dst_bytes) over the converted g files."""
    if not os.path.isfile(os.path.join(src, "catalog.pt")):
        raise ValueError(f"no catalog.pt in --src {src}")
    if os.path.realpath(src) == os.path.realpath(dst):
        raise ValueError("--dst must differ from --src (the source trace is "
                         "never modified in place)")
    if (os.path.isdir(dst) and os.listdir(dst) and not overwrite):
        raise ValueError(
            f"--dst {dst} exists and is not empty (use --overwrite to reuse)")
    # overwrite is deterministic: the dst is presented as a trace, not a
    # manifest, so stale trace-pattern files outside the new selection must
    # go — discover_trace would otherwise see the union of two step sets.
    if overwrite and os.path.isdir(dst):
        for f in os.listdir(dst):
            if _G_RE.match(f) or _S_RE.match(f) or f == "catalog.pt":
                os.unlink(os.path.join(dst, f))

    os.makedirs(dst, exist_ok=True)
    # catalog: verbatim except g_dtype, which flips to bf16 to keep the
    # converted dir self-describing (runtime-bf16 dumps land on the same
    # catalog shape via the collector's own field)
    cat = torch.load(os.path.join(src, "catalog.pt"),
                     map_location="cpu", weights_only=False)
    if str(cat.get("g_dtype", "fp32")) == "bf16":
        # loud but not fatal: the output is still correct (idempotent — the
        # per-tensor guard below skips non-fp32 payloads), so this pass just
        # re-copies bytes at full volume with zero rounding
        msg = ("source catalog g_dtype is already bf16: this conversion "
               "rounds nothing (an idempotent re-copy of an already-bf16 "
               "trace)")
        if log is not None:
            log(f"WARNING: {msg}", flush=True)
        else:
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)
    cat["g_dtype"] = "bf16"
    torch.save(cat, os.path.join(dst, "catalog.pt"))

    def _in_range(step):
        if from_step is not None and step < from_step:
            return False
        if to_step is not None and step >= to_step:
            return False
        return True

    n_g = n_state = 0
    src_bytes = dst_bytes = 0
    for f in sorted(os.listdir(src)):
        m = _G_RE.match(f)
        if m:
            step = int(m.group(1))
            if not _in_range(step):
                continue
            s_path = os.path.join(src, f)
            d = torch.load(s_path, map_location="cpu", weights_only=False,
                           mmap=True)
            d["payload"] = {
                k: (v.to(torch.bfloat16)
                    if torch.is_tensor(v) and v.dtype == torch.float32 else v)
                for k, v in d["payload"].items()
            }
            d_path = os.path.join(dst, f)
            torch.save(d, d_path)
            n_g += 1
            src_bytes += os.path.getsize(s_path)
            dst_bytes += os.path.getsize(d_path)
            if log and n_g % 100 == 0:
                log(f"g {n_g} files converted (last step {step})", flush=True)
            continue
        m = _S_RE.match(f)
        if m and _in_range(int(m.group(1))):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            n_state += 1
    return n_g, n_state, src_bytes, dst_bytes


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="build a bf16 screening copy of a trace (g payloads "
                    "fp32->bf16; states verbatim; catalog verbatim except "
                    "the g_dtype=bf16 tag; source untouched)")
    ap.add_argument("--src", required=True, help="source trace directory")
    ap.add_argument("--dst", required=True, help="destination directory")
    ap.add_argument("--from-step", type=int, default=None,
                    help="convert g/state files from this step (inclusive)")
    ap.add_argument("--to-step", type=int, default=None,
                    help="up to this step (exclusive); default: whole trace")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a non-empty --dst")
    args = ap.parse_args(argv)
    if args.to_step is not None and args.from_step is not None \
            and args.to_step <= args.from_step:
        raise ValueError(f"--to-step must be > --from-step, got "
                         f"{args.from_step}..{args.to_step}")

    n_g, n_state, src_bytes, dst_bytes = convert(
        args.src, args.dst, args.from_step, args.to_step,
        overwrite=args.overwrite, log=print)
    print(f"done: {n_g} g files converted "
          f"({src_bytes / 1e9:.2f} GB -> {dst_bytes / 1e9:.2f} GB), "
          f"{n_state} state files copied")


if __name__ == "__main__":
    main()
