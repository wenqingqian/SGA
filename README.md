# SGA

Plugin-based **selective gradient accumulation (SGA)** framework: sparse-gradient
communication training on Megatron-LM, where an SGA scheme (score × selection ×
error_feedback × backend) is pure configuration, not code.

- `sga/launcher/` — multi-node ssh launcher (deep-copied from FlagScale's
  runner, train/ssh/megatron only; run/stop/query via `run.py` + hydra).
- `sga/config/` — single-layer yaml validation; the `sga:` algorithm section
  never reaches the megatron command line.
- `sga/models/` — model registry (day-one: `llama-1b`); `sga/train_gpt.py` is
  the torchrun entrypoint reusing megatron's stock pretrain implementations.
- `sga/plugins/`, `sga/backends/` — reserved for SGA algorithm plugins and
  gradient-sync backends (day-one backend: `dense`).
- `examples/llama3/train_1b_dense.yaml` — dense llama 1B config.
- `tests/` — unit tests + the megatron arg audit tool.
- `env.sh` — environment single source of truth (megatron source resolution +
  anti-shadowing assertion); `setup.sh` — environment/manifest check;
  `manifest.yaml` — expected sibling-repo states.

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
