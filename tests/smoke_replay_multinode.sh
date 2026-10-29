#!/bin/bash
# Cross-node fused replay smoke: CPU world=2 over 2 nodes (tiny synthetic
# fixture, one cold-cache populate on the master node + out-of-band cache
# publish confirmation + ssh node-worker launch). Deliberately light: runs
# on CPU so it can share nodes with GPU jobs.
#
# Usage (HOST, from the SGA repo checkout under test):
#   bash tests/smoke_replay_multinode.sh <master-ip> <worker-ip> <host-workdir>
#   <host-workdir> e.g. $HOST_WS/AGENTSPACE/iterations/<iteration-dir>/data/it2_smoke
#
# <host-workdir> must live on the shared NFS (both nodes read the fixture
# and the published dense cache through /workspace view). PASS criterion:
# the cross-node world=2 branch report sits in the pure-reordering band
# against the master-local world=1 reference (same comparator as the
# unit tests: sums rel<=1e-6, everything else exact).
set -euo pipefail

MASTER="${1:?master node ip}"
WORKER="${2:?worker node ip}"
HOST_WORK="${3:?host-side NFS workdir, e.g. /nfs/gzlu/sga_framework/AGENTSPACE/iterations/<iteration-dir>/data/it2_smoke}"

HOST_WS="/nfs/gzlu/sga_framework"
REMOTE_WS="/workspace/sga_framework"
HOST_SGA="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SGA="${HOST_SGA/#$HOST_WS/$REMOTE_WS}"
[ "$REMOTE_SGA" != "$HOST_SGA" ] || { echo "SGA checkout must live under $HOST_WS (got $HOST_SGA)" >&2; exit 2; }
REMOTE_WORK="${HOST_WORK/#$HOST_WS/$REMOTE_WS}"
[ "$REMOTE_WORK" != "$HOST_WORK" ] || { echo "workdir must live under $HOST_WS (got $HOST_WORK)" >&2; exit 2; }

SSH="ssh -p 2222 -i $HOST_WS/cluster_ssh_key -o BatchMode=yes -o StrictHostKeyChecking=no"
run() { $SSH "root@$1" "$2"; }
COND="source /root/miniconda3/etc/profile.d/conda.sh && conda activate flagscale"
# worktree checkouts have no sibling Megatron-LM: point env.sh back at the
# main tree (the launcher's own node-worker lines carry an explicit
# PYTHONPATH prepend and do not depend on this)
MEGATRON_SGA_DIR_OPT="MEGATRON_SGA_DIR=$REMOTE_WS/Megatron-LM"

T0=900
ARGS="--trace-dir $REMOTE_WORK/trace --branch-point $T0 --window 6 \
 --horizons 1,3,6 --ratios 20,30 --score-names abs_g,update \
 --replay-mode cycle --cycle-len 2 --dense-cache $REMOTE_WORK/cache \
 --device cpu --chunk-mb 1"

echo "== [1/5] build tiny fixture on master ($MASTER) -> $REMOTE_WORK =="
rm -rf "$HOST_WORK"
mkdir -p "$HOST_WORK"
run "$MASTER" "$COND && python - <<'PYEOF'
import os, sys
sys.path[:0] = ['$REMOTE_SGA', '$REMOTE_SGA/tests']
from test_replay_fused import _build_fixture
os.makedirs('$REMOTE_WORK/trace', exist_ok=True)
_build_fixture('$REMOTE_WORK/trace')
print('fixture ok')
PYEOF"

echo "== [2/5] guard: helper signatures (cheap; drift must fail BEFORE the"
echo "   multi-node run pays for itself, not at the final compare) =="
run "$MASTER" "$COND && python - <<'PYEOF'
import inspect, sys
sys.path[:0] = ['$REMOTE_SGA', '$REMOTE_SGA/tests']
from test_replay_fused import _build_fixture, _compare_world2
assert len(inspect.signature(_build_fixture).parameters) == 2, \
    '_build_fixture signature drifted'
assert len(inspect.signature(_compare_world2).parameters) == 3, \
    '_compare_world2 signature drifted'
print('helper signatures ok')
PYEOF"

echo "== [3/5] world=2 cross-node ($MASTER + $WORKER), cold cache =="
echo "   (exercises: node-0 parent populate -> publish -> out-of-band"
echo "    confirmation on $WORKER -> bounded preflight -> ssh node-workers)"
run "$MASTER" "$COND && cd $REMOTE_SGA && $MEGATRON_SGA_DIR_OPT source env.sh >/dev/null 2>&1; \
  PYTHONUNBUFFERED=1 python sga/replay.py $ARGS --fused-shards 2 \
  --nodes $MASTER,$WORKER --launch-log-dir $REMOTE_WORK/logs \
  --out $REMOTE_WORK/report_w2.json"

echo "== [4/5] world=1 reference on master (cache hit) =="
run "$MASTER" "$COND && cd $REMOTE_SGA && $MEGATRON_SGA_DIR_OPT source env.sh >/dev/null 2>&1; \
  PYTHONUNBUFFERED=1 python sga/replay.py $ARGS --fused-shards 1 \
  --out $REMOTE_WORK/report_w1.json"

echo "== [5/5] pure-reordering band compare (world2 vs world1) =="
run "$MASTER" "$COND && python - <<'PYEOF'
import json, sys
sys.path[:0] = ['$REMOTE_SGA', '$REMOTE_SGA/tests']
from test_replay_fused import _compare_world2
w1 = json.load(open('$REMOTE_WORK/report_w1.json'))
w2 = json.load(open('$REMOTE_WORK/report_w2.json'))
assert w2['config']['nodes'] == ['$MASTER', '$WORKER'], w2['config']['nodes']
_compare_world2(w1, w2, 'cross-node-world2')
print('SMOKE PASS: cross-node world=2 matches world=1 within the band')
PYEOF"
echo "== smoke done: logs and reports under $HOST_WORK =="
