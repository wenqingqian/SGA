# SGA

SGA (selective gradient accumulation) is a plugin-based framework for
training with sparse gradient communication on Megatron-LM. A scheme is
pure yaml configuration: pick one value on each plugin axis (score,
selection, optimizer, backend) — no scheme ships dedicated code.

## Installation

Clone this repository and the Megatron-LM fork side by side — same
parent directory, Megatron-LM on the `sga` branch:

```bash
git clone https://github.com/wenqingqian/SGA
git clone -b sga https://github.com/wenqingqian/Megatron-LM
```

Megatron-LM is used from source, never pip-installed. Sourcing `env.sh`
resolves the sibling checkout (or `$MEGATRON_SGA_DIR` when set), puts
both trees on `PYTHONPATH`, and fails loudly if a pip-installed
megatron-core would shadow the source:

```bash
cd SGA && source env.sh
```

## Repository structure

- `sga/plugins/` — the plugin axes (score / selection / optimizer):
  scoring functions, top-k selection methods, and the residual-aware
  AdamW optimizer.
- `sga/backends/` — gradient-sync backends: `torch_pipelined_index` and
  `torch_pipelined_mask` (active), and `nccl_pipelined_index`
  (deprecated for now, but code and tests are kept). `dense` is not a
  backend class — it is the SGA-off value of `sga_backend`.
- `sga/config/` — single-layer yaml validation. Keys under the `sga:`
  section reach the training entry as flat `--sga-*` flags, are
  validated there, and never mix into megatron's `train.*` arguments.
- `sga/train_gpt.py` — the torchrun entrypoint. It reuses megatron's
  stock pretraining implementations and GPT build chain unchanged.
- `sga/launcher/` — multi-node ssh launcher for the training path
  (ssh + torchrun + megatron). Job control goes through
  `run.py` + hydra.
- `sga/replay.py`, `sga/replay_fused.py` — offline replay. Record one
  training window, then evaluate any score × ratio × cycle-length
  configuration offline. The fused engine advances a whole config group
  in a single pass over the window, sharded across GPUs, with an exact
  distributed top-k and a reusable dense-reference cache.
- `experiments/example/` — a runnable single-node walkthrough of the
  full workflow: dense trace capture, offline replay, and a sparse
  training run.
- `examples/` — training configs: llama3 1B (dense / shadow / SGA
  presets) and qwen3 0.6B.
- `tests/` — unit tests plus the megatron argument audit tool.
- `env.sh` — environment setup in one place: locates the Megatron-LM
  source checkout, exports PYTHONPATH, and fails loudly if a
  pip-installed megatron would shadow the source tree. `setup.sh`
  prints an environment and sibling-repo report against
  `manifest.yaml`.

## Running a training job

`run.py` is a hydra shim over the launcher. Given a config it wires
MegatronBackend + SshLauncher: ssh to each node named in the hostfile,
run the config's `cmds.before_start` hook there (activate the training
environment and `source env.sh`, which puts the Megatron-LM source on
`PYTHONPATH`), then start torchrun with the entrypoint
`sga/train_gpt.py` — which drives megatron's stock pretraining loop
unchanged.

A config is one flat yaml with three sections:

- `experiment:` — launch and outputs: `exp_name`, `exp_dir`, `runner`
  (`nnodes`, `nproc_per_node`, `hostfile`, `ssh_port`, `master_port`),
  the `cmds.before_start` hook, and `envs`.
- `train:` — megatron's own arguments. Leaf keys are megatron CLI arg
  names, and node-count-dependent values (e.g. `global_batch_size`)
  are set per launch.
- `sga:` — the plugin axes, validated before launch and handed to the
  entrypoint as flat `--sga-*` flags. `sga_enabled: false` (the dense
  presets) leaves the job a plain megatron run.

The top-level `action` key selects what the launcher does:

| action | effect |
| --- | --- |
| `run` | background launch (remote nohup). The console tails node 0's log during startup, then the launcher returns while the job runs on |
| `test` | foreground launch: stream the job output and return when it exits |
| `dryrun` | print the per-node launch plan without touching the nodes |
| `query` | report the status of a running job |
| `stop` | stop a running job |

```bash
python run.py --config-path examples/llama3 --config-name train_1b_dense \
    action=run experiment.exp_name=dense-1b experiment.exp_dir=./dense-1b \
    experiment.runner.nnodes=2 experiment.runner.hostfile=hostfile.txt \
    train.model.global_batch_size=16
```

The hostfile is not committed. The example configs default to
`examples/<model>/hostfile.txt`: create it there with one
`<ip> slots=8` line per node. A missing hostfile only WARNs, and the
job runs local-only.

For a sparse run, point `--config-name` at an SGA preset or override
the `sga:` keys. `experiments/example/` walks the whole
trace → replay → sparse-run workflow end to end.

## Offline replay (fused)

Scheme research needs the answer to "what would training look like
under a different selection scheme?" many times over. Record a training
window once with the trace options (`sga_trace_dir` +
`sga_trace_branch_points` + `sga_trace_window`, and optionally
`sga_trace_dump_dtype=bf16` to store the per-step gradient dumps as
bf16) on a shadow run (a run that trains exactly like a dense run
while the engine records gradients on the side), then replay any
configuration offline. The fused engine evaluates a whole group of
score × ratio × cycle-length combinations in a single pass over the
window (mixed cycle lengths via `--cycle-lens`, group size derived
from a state-byte budget via `--fused-state-budget-mb`), sharded
across GPUs, and spans nodes with `--nodes` (ssh-launched shards over
a shared filesystem). A single-shard run reproduces per-config replay
bitwise. A multi-shard run stays within a small pure reordering band.
The branch point is a global training step number (train iteration):
`--branch-point 40` replays from the optimizer-state snapshot saved at
the 40th train iteration and consumes the `--window` recorded gradient
steps after it (steps 41..1040 for `--window 1000`):

    CUDA_VISIBLE_DEVICES=0,...,7 python -m sga.replay \
        --trace-dir <trace> --branch-point 40 --window 1000 \
        --ratios 20,30,40 --score-names abs_m,update \
        --replay-mode cycle --cycle-len 20 --device cuda --fused-shards 8 \
        --dense-cache <cache-dir> --out report.json
