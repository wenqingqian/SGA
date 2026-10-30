# Minimal SGA example: dense trace -> offline replay -> sparse training

This directory is a runnable, single-node walkthrough of the SGA workflow.
It is deliberately small: a toy llama-architecture model, a short gradient
window, and the same 48-cell grid used by the E-series replay experiments.

The three steps are:

1. **Dense trace capture** — run a shadow-mode dense training for a short
   range, dumping BF16 gradient payloads asynchronously.
2. **Offline replay** — evaluate the 48-cell grid
   `score {abs_m, abs_g, update, coi} x ratio {20, 30, 40} x
   cycle_len {10, 50, 100, 200}` on the captured window.
3. **Sparse training** — take the replay argmin and launch a real SGA
   sparse run with that score/ratio/cycle length.

## Prerequisites

- Run inside the training container (`flagscale-gzlu`) with the
  `flagscale` conda environment.
- One node with 8 visible GPUs and the workspace mounted at
  `/workspace/sga_framework` (the scripts use container-side paths).
- `SGA/env.sh` resolves Megatron-LM from the sibling checkout.
- The dataset paths in `configs/*.yaml` exist
  (`data/openwebtext`, `data/llama3_dataset`).

## Quick start

```bash
cd /workspace/sga_framework/SGA
source env.sh

NODE=<node-ip> SSH_PORT=2222 bash experiments/example/run_example.sh
```

`NODE` must be reachable over ssh from the launcher process; for a
non-default ssh port pass `SSH_PORT` (for example `4399`).

Outputs go to `WORKDIR` (default `/workspace/sga_framework/example_run`):

```text
example_run/
  hostfile.txt
  trace/                 # catalog.pt, step_*_state.pt, step_*_g.pt
  cache/                 # dense replay reference cache
  replay_report.json     # 48-cell replay result
  dense/                 # step-1 hydra/log output
  sga/                   # step-3 hydra/log output
  best.env               # selected score/ratio/cycle length
```

## What each step does

### 1. Dense trace capture

`configs/llama_tiny_dense_trace.yaml` runs shadow mode: training follows
the stock dense optimizer exactly, while the SGA engine records the
pre-commit gradient. The trace collector writes:

- `catalog.pt` — parameter names and sizes;
- `step_<branch>_state.pt` — full `w/m/v` at the branch point;
- `step_<t>_g.pt` — per-step gradient payloads for the window.

The example sets `sga_trace_dump_dtype: bf16` (gradient payload only;
optimizer state stays fp32) and `sga_trace_async: true` (serialization
and file writes run on a bounded background writer after the blocking CPU
snapshot). The run stops at `branch + window + 1` steps.

### 2. Offline replay

`sga.replay` consumes the trace and runs the fused engine in cycle mode
over the 48-cell grid. Selection uses the production-default
`balanced-histogram` semantics, so replay and online selection share the
same histogram implementation. The report contains, for every cell,
`distortion` at each horizon.

For each cell, distortion is

```text
||w_sparse(h) - w_dense(h)|| / ||w_dense(h) - w_start||
```

so lower is better. `select_best.py` picks the smallest distortion at the
largest common horizon and writes shell assignments to `best.env`.

### 3. Sparse training

`configs/llama_tiny_sga.yaml` is a normal SGA configuration. The launcher
overrides only `sga_score`, `sga_selection_ratio_percent`,
`sga_cycle_period`, and the step count, using the replay argmin. This is
the same path a production run uses; no replay-specific code is involved.
If `SGA_STEPS` would stop before the first cycle boundary
(`start_iter + cycle_period`), the launcher extends the run to cross one
boundary and execute at least one sparse step.

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `NODE` | required | node IP reachable over ssh |
| `SSH_PORT` | `2222` | ssh port of that node |
| `GPUS` | `8` | visible GPUs and replay shards |
| `WORKDIR` | `/workspace/sga_framework/example_run` | output root (NFS-visible) |
| `BRANCH` | `4` | trace branch point (training step) |
| `WINDOW` | `64` | captured gradient steps after the branch point |
| `SGA_STEPS` | `16` | steps for the final sparse run |
| `MASTER_PORT` | `29620` | torchrun rendezvous port |
| `HORIZONS` | `1,2,4,8,16,32,64` | replay horizons (truncated to `WINDOW`) |

`WINDOW` must be long enough to make the larger cycle lengths observable.
With the default `WINDOW=64`, `T=10` and `T=50` reach a boundary; `T=100`
and `T=200` do not, so those cells reduce to the frozen-mask behavior.
Use a longer window for a meaningful cycle-length comparison.

## Minimal, not production

- The model is a toy llama (4 layers, hidden 512) trained from scratch.
  Swap in `examples/llama3/train_1b_*.yaml` and a checkpoint for a real
  study.
- The window and step counts are intentionally short. The replay decision
  is only meaningful on a representative window (the E-series uses 1000+
  steps).
- Single node only. Multi-node replay currently requires
  `--selection-semantics topk`; balanced-histogram replay is single-node.

## Cleanup

```bash
rm -rf /workspace/sga_framework/example_run
```
