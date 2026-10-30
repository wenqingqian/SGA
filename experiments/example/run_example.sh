#!/usr/bin/env bash
# Minimal three-step SGA example: dense trace -> offline replay -> sparse run.
# Run inside the training container on the target node (single node only).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGA_ROOT="$(cd "$HERE/../.." && pwd)"

NODE="${NODE:?set NODE to the single node IP reachable over ssh}"
SSH_PORT="${SSH_PORT:-2222}"
GPUS="${GPUS:-8}"
WORKDIR="${WORKDIR:-/workspace/sga_framework/example_run}"
BRANCH="${BRANCH:-4}"
WINDOW="${WINDOW:-64}"
SGA_STEPS="${SGA_STEPS:-16}"
MASTER_PORT="${MASTER_PORT:-29620}"
HORIZONS="${HORIZONS:-1,2,4,8,16,32,64}"
TRACE_DTYPE="${TRACE_DTYPE:-bf16}"

if (( SGA_STEPS < 2 )); then
  echo "SGA_STEPS must be >= 2 (a sparse step needs a cycle)" >&2
  exit 2
fi
CYCLE_START=$(( SGA_STEPS < 4 ? 1 : 4 ))
DENSE_STEPS=$(( BRANCH + WINDOW + 1 ))

HOSTFILE="$WORKDIR/hostfile.txt"
TRACE_DIR="$WORKDIR/trace"
CACHE_DIR="$WORKDIR/cache"
REPORT="$WORKDIR/replay_report.json"
BEST_ENV="$WORKDIR/best.env"

mkdir -p "$WORKDIR" "$TRACE_DIR" "$CACHE_DIR"
rm -rf "$WORKDIR/dense" "$WORKDIR/sga"
printf '%s slots=%s\n' "$NODE" "$GPUS" > "$HOSTFILE"

cd "$SGA_ROOT"
source env.sh --check

run_train () {  # $1=config name, $2=exp dir, rest=hydra overrides
  local config="$1" exp_dir="$2"; shift 2
  python run.py \
    --config-path "$HERE/configs" \
    --config-name "$config" \
    action=test \
    experiment.exp_name="example_$config" \
    experiment.exp_dir="$exp_dir" \
    experiment.runner.hostfile="$HOSTFILE" \
    experiment.runner.ssh_port="$SSH_PORT" \
    experiment.runner.master_port="$MASTER_PORT" \
    experiment.runner.nproc_per_node="$GPUS" \
    "$@"
}

echo "== [1/3] dense trace capture: branch=$BRANCH window=$WINDOW steps=$DENSE_STEPS =="
rm -rf "$TRACE_DIR"
mkdir -p "$TRACE_DIR"
run_train llama_tiny_dense_trace "$WORKDIR/dense" \
  ++sga.sga_trace_dir="$TRACE_DIR" \
  ++sga.sga_trace_branch_points="$BRANCH" \
  ++sga.sga_trace_window="$WINDOW" \
  ++sga.sga_trace_dump_dtype="$TRACE_DTYPE" \
  ++train.model.train_iters="$DENSE_STEPS" \
  ++train.model.exit_interval="$DENSE_STEPS"
test -f "$TRACE_DIR/catalog.pt" || { echo "trace catalog missing" >&2; exit 1; }

echo "== [2/3] offline replay: 48 cells (4 scores x 3 ratios x 4 cycle lengths) =="
python -m sga.replay \
  --trace-dir "$TRACE_DIR" \
  --branch-point "$BRANCH" \
  --window "$WINDOW" \
  --horizons "$HORIZONS" \
  --ratios 20,30,40 \
  --score-names abs_m,abs_g,update,coi \
  --replay-mode cycle \
  --cycle-lens 10,50,100,200 \
  --selection-semantics balanced-histogram \
  --trace-dtype "$TRACE_DTYPE" \
  --device cuda \
  --fused-shards "$GPUS" \
  --dense-cache "$CACHE_DIR" \
  --out "$REPORT"

echo "== [3/3] sparse training with the replay argmin =="
python "$HERE/select_best.py" "$REPORT" > "$BEST_ENV"
cat "$BEST_ENV"
# shellcheck disable=SC1090
source "$BEST_ENV"
# The first SGA boundary is at cycle_start + cycle_period; extend the run if
# the requested step count would stop before any sparse step exists.
SPARSE_STEPS=$SGA_STEPS
if (( SPARSE_STEPS <= CYCLE_START + BEST_CYCLE )); then
  SPARSE_STEPS=$(( CYCLE_START + BEST_CYCLE + 2 ))
  echo "extending the sparse run to $SPARSE_STEPS steps to cross one boundary"
fi
run_train llama_tiny_sga "$WORKDIR/sga" \
  ++sga.sga_score="$BEST_SCORE" \
  ++sga.sga_selection_ratio_percent="$BEST_RATIO" \
  ++sga.sga_cycle_period="$BEST_CYCLE" \
  ++sga.sga_cycle_start_iter="$CYCLE_START" \
  ++train.model.train_iters="$SPARSE_STEPS" \
  ++train.model.exit_interval="$SPARSE_STEPS"

echo "== example complete =="
echo "trace:  $TRACE_DIR"
echo "report: $REPORT"
echo "best:   $BEST_ENV"
echo "logs:   $WORKDIR/dense $WORKDIR/sga"
