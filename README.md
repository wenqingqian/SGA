# SGA

Plugin-based **selective gradient accumulation (SGA)** framework: sparse-gradient
communication training on Megatron-LM, where an SGA scheme (score × selection ×
error_feedback × backend) is pure configuration, not code.

- `sga/launcher/` — multi-node ssh launcher (deep-copied from FlagScale's
  runner, train/ssh/megatron only; run/stop/query via `run.py` + hydra).
- `sga/config/` — single-layer yaml validation; the `sga:` algorithm section
  never reaches the megatron command line.
- `sga/train_gpt.py` — the torchrun entrypoint, reusing megatron's stock
  pretrain implementations and stock GPT build chain verbatim.
- `sga/plugins/`, `sga/backends/` — reserved for SGA algorithm plugins and
  gradient-sync backends (day-one backend: `dense`).
- `sga/replay.py`, `sga/replay_fused.py` — offline replay oracle: record a
  training window once (shadow-mode trace), then sweep any
  score × ratio × cycle-length configuration offline; the fused engine
  streams the window once for a whole config group, sharded across GPUs
  with an exact distributed top-k and a reusable dense-reference cache
  (see docs/fused_replay.md).
- `examples/llama3/train_1b_dense.yaml` — dense llama 1B config.
- `tests/` — unit tests + the megatron arg audit tool.
- `env.sh` — environment single source of truth (megatron source resolution +
  anti-shadowing assertion); `setup.sh` — environment/manifest check;
  `manifest.yaml` — expected sibling-repo states.

## Offline replay (fused)

SGA scheme research needs "what would training look like under a different
selection scheme?" answered many times. Record a training window once
(`sga_trace_dir` + `sga_trace_branch_points` + `sga_trace_window` on a
shadow run), then replay any configuration offline — the fused engine
advances a whole group of score × ratio arms over a single window pass,
sharded across GPUs, reproducing per-config replay bitwise (single shard)
or within the frozen pure-reordering band (multi-shard). The branch point
is a global training step number (train iteration): `--branch-point 40`
replays from the optimizer-state snapshot saved at the 40th train
iteration, consuming the `--window` recorded gradient steps after it
(steps 41..1040 for `--window 1000`):

    CUDA_VISIBLE_DEVICES=0,...,7 python -m sga.replay \
        --trace-dir <trace> --branch-point 40 --window 1000 \
        --ratios 20,30,40 --score-names abs_m,update \
        --replay-mode cycle --cycle-len 20 --device cuda --fused-shards 8 \
        --dense-cache <cache-dir> --out report.json

Details, precision guarantees, and a full parameter reference:
[docs/fused_replay.md](docs/fused_replay.md) (中文).

## Quick start (in the training container)

```bash
source env.sh            # PYTHONPATH: SGA + Megatron-LM, anti-shadowing assert
bash setup.sh            # env + sibling-repo state report
python tests/audit_megatron_args.py check examples/llama3/train_1b_dense.yaml
python run.py --config-path examples/llama3 --config-name train_1b_dense \
    action=run experiment.exp_name=exp1 experiment.exp_dir=/path/to/exp1 \
    experiment.runner.nnodes=3 experiment.runner.hostfile=examples/llama3/hostfile.txt \
    train.model.global_batch_size=24 train.model.optimizer.lr_scheduler.lr=3e-05 \
    train.model.optimizer.lr_scheduler.min_lr=3e-06
python run.py ... action=query   # / action=stop
```

## Out of scope — never built here

elastic training · auto_tuner · cloud launcher · straggler detection ·
multi-chip backends · inference · serving · evaluation · node monitoring.
(The launcher keeps none of FlagScale's corresponding code paths.)
