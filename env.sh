#!/bin/bash
# SGA environment: single source of truth for PYTHONPATH-style env setup.
#
# Usage:
#   source env.sh              # locate megatron, export PYTHONPATH, assert no shadowing
#   source env.sh --check      # same, plus fail loudly (exit 1) on any problem
#
# Megatron-LM source resolution order (first hit wins):
#   1. $MEGATRON_SGA_DIR (explicit override)
#   2. sibling checkout:  <workspace>/Megatron-LM   (next to this SGA repo)
#   3. vendored copy:     <SGA_ROOT>/third_party/megatron-lm   (unverified, reserved)
# Megatron-LM is used from source via PYTHONPATH; it is never pip-installed here.

set -u

_SGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGA_ROOT="${SGA_ROOT:-$_SGA_ROOT}"

_SGA_CHECK=0
if [ "${1:-}" = "--check" ]; then
    _SGA_CHECK=1
fi

# --- locate megatron source -------------------------------------------------
if [ -n "${MEGATRON_SGA_DIR:-}" ] && [ -d "${MEGATRON_SGA_DIR}/megatron" ]; then
    _MEGATRON_DIR="${MEGATRON_SGA_DIR}"
elif [ -d "${_SGA_ROOT}/../Megatron-LM/megatron" ]; then
    _MEGATRON_DIR="$(cd "${_SGA_ROOT}/../Megatron-LM" && pwd)"
elif [ -d "${_SGA_ROOT}/third_party/megatron-lm/megatron" ]; then
    _MEGATRON_DIR="${_SGA_ROOT}/third_party/megatron-lm"
else
    echo "SGA env error: no Megatron-LM source found (tried \$MEGATRON_SGA_DIR," \
         "sibling ../Megatron-LM, vendored third_party/megatron-lm)" >&2
    [ "$_SGA_CHECK" -eq 1 ] && exit 1
    return 1
fi
export MEGATRON_SGA_DIR="$_MEGATRON_DIR"
export PYTHONPATH="${_SGA_ROOT}:${_MEGATRON_DIR}:${PYTHONPATH:-}"

# --- anti-shadowing assertion ------------------------------------------------
# A pip-installed megatron-core in the active environment would silently win
# over the PYTHONPATH source. Fail instead of training against the wrong tree.
_sga_check_megatron_source() {
    python3 - <<PYEOF
import os, sys
# 'megatron' is a PEP420 namespace package (its __path__ can also absorb a
# pip-installed megatron-core), so assert on a concrete submodule instead.
import megatron.training
actual = os.path.realpath(os.path.dirname(megatron.training.__file__))
expected = os.path.realpath(os.environ["MEGATRON_SGA_DIR"])
if not actual.startswith(expected + os.sep):
    sys.stderr.write(
        "SGA anti-shadowing check FAILED: 'import megatron.training' resolved to "
        f"{actual}, expected under {expected}. Uninstall the shadowing "
        "megatron-core or fix PYTHONPATH order before training.\n"
    )
    sys.exit(1)
print(f"SGA megatron source: {actual}")
PYEOF
}

if [ "$_SGA_CHECK" -eq 1 ]; then
    _sga_check_megatron_source || exit 1
else
    _sga_check_megatron_source || return 1
fi

unset _SGA_ROOT _SGA_CHECK _MEGATRON_DIR
