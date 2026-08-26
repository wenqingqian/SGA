#!/bin/bash
# SGA setup check: environment sanity + sibling repo state vs manifest.yaml.
# Non-destructive: prints a report, exits 0 unless the environment itself is
# unusable (env.sh --check failure). Repo/commit mismatches are WARNs only.

set -u

_SGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_SGA_ROOT"

echo "== SGA setup check =="

# 1. env.sh: megatron lookup + anti-shadowing (hard fail if broken)
if ! source "$_SGA_ROOT/env.sh" --check; then
    echo "FAIL: env.sh --check did not pass" >&2
    exit 1
fi

# 2. manifest comparison (WARN only): branch divergence + commit ancestry
python3 - <<'PYEOF'
import os, subprocess, sys
import yaml

root = os.environ["SGA_ROOT"]
with open(os.path.join(root, "manifest.yaml")) as f:
    manifest = yaml.safe_load(f)

warn = 0
for name, spec in manifest["repos"].items():
    path = os.path.normpath(os.path.join(root, spec["path"]))
    if not os.path.isdir(os.path.join(path, ".git")):
        print(f"WARN {name}: {path} is not a git checkout")
        warn += 1
        continue
    def git(*args):
        return subprocess.run(["git", "-C", path, *args],
                              capture_output=True, text=True).stdout.strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    head = git("rev-parse", "--short", "HEAD")
    if branch != spec["branch"]:
        print(f"WARN {name}: on branch '{branch}', manifest expects '{spec['branch']}'")
        warn += 1
    if head.startswith(spec["base_commit"]):
        print(f"OK   {name}: {head} == manifest base on branch '{branch}'")
    else:
        is_anc = subprocess.run(
            ["git", "-C", path, "merge-base", "--is-ancestor",
             spec["base_commit"], "HEAD"]).returncode == 0
        if is_anc:
            print(f"OK   {name}: {head} (descends from manifest base {spec['base_commit']})")
        else:
            print(f"WARN {name}: HEAD {head} does not descend from manifest base "
                  f"{spec['base_commit']}")
            warn += 1
sys.exit(0)
PYEOF

# 3. runtime dependency versions (recorded for post-hoc attribution)
python3 - <<'PYEOF'
import torch
print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
for mod in ("transformer_engine", "flash_attn"):
    try:
        m = __import__(mod)
        print(f"{mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"WARN {mod}: import failed: {e}")
PYEOF

echo "== setup check done =="
