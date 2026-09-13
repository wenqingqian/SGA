#!/bin/bash
# SGA environment: single source of truth for PYTHONPATH-style env setup.
#
# Usage:
#   source env.sh              # locate megatron, export PYTHONPATH, assert no shadowing
#   source env.sh --check      # same, plus fail loudly (nonzero status) on any problem
#                              # (this is what setup.sh guards with `if !`)
#   bash env.sh [--check]      # executed form also works; failures exit nonzero
#
# Contract (fail loudly, both modes): every failure path yields a nonzero
# status to its caller — `return` when sourced (propagating through setup.sh's
# `if !` without killing the shell), `exit` when executed. Sourced vs executed
# is detected via `$0 != "$BASH_SOURCE"`. Note an executed run cannot export
# variables into your shell; sourcing is the normal mode. `--check` is kept
# for callers that want to signal intent; behavior is strict either way.
#
# Sourcing is idempotent: re-sourcing never duplicates PYTHONPATH entries,
# and all helpers/options this script introduces are undone afterwards.
#
# Megatron-LM source resolution order (first hit wins):
#   1. $MEGATRON_SGA_DIR (explicit override)
#   2. sibling checkout:  <workspace>/Megatron-LM   (next to this SGA repo)
#   3. vendored copy:     <SGA_ROOT>/third_party/megatron-lm   (unverified, reserved)
# Megatron-LM is used from source via PYTHONPATH; it is never pip-installed here.

_sga_sourced=1
[ "$0" != "${BASH_SOURCE[0]}" ] || _sga_sourced=0

# Remember whether the sourcing shell already ran with nounset, so we can
# restore its exact option state instead of leaking ours into it.
_sga_had_u=0
case $- in *u*) _sga_had_u=1 ;; esac
set -u

_SGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGA_ROOT="${SGA_ROOT:-$_SGA_ROOT}"

# Prepend $1 to PYTHONPATH unless already present (idempotent re-source);
# never leaves a trailing empty entry when PYTHONPATH started out unset.
_pythonpath_prepend() {
    case ":${PYTHONPATH:-}:" in
        *":$1:"*) ;;
        *) if [ -n "${PYTHONPATH:-}" ]; then PYTHONPATH="$1:$PYTHONPATH"; else PYTHONPATH="$1"; fi ;;
    esac
}

# --- anti-shadowing assertion ------------------------------------------------
# A pip-installed megatron-core in the active environment would silently win
# over the PYTHONPATH source. Fail instead of training against the wrong tree.
_sga_check_megatron_source() {
    python3 - <<'PYEOF'
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

_sga_main() {
    # --- locate megatron source ---------------------------------------------
    if [ -n "${MEGATRON_SGA_DIR:-}" ] && [ -d "${MEGATRON_SGA_DIR}/megatron" ]; then
        _MEGATRON_DIR="${MEGATRON_SGA_DIR}"
    elif [ -d "${_SGA_ROOT}/../Megatron-LM/megatron" ]; then
        _MEGATRON_DIR="$(cd "${_SGA_ROOT}/../Megatron-LM" && pwd)"
    elif [ -d "${_SGA_ROOT}/third_party/megatron-lm/megatron" ]; then
        _MEGATRON_DIR="${_SGA_ROOT}/third_party/megatron-lm"
    else
        echo "SGA env error: no Megatron-LM source found (tried \$MEGATRON_SGA_DIR," \
             "sibling ../Megatron-LM, vendored third_party/megatron-lm)" >&2
        return 1
    fi
    export MEGATRON_SGA_DIR="$_MEGATRON_DIR"

    _pythonpath_prepend "$_MEGATRON_DIR"
    _pythonpath_prepend "$_SGA_ROOT"
    export PYTHONPATH

    # Sparse NCCL library for the nccl_pipelined_index backend: the nccl-sar
    # fork's modified ncclAllReduce signature makes it mutually exclusive
    # with torch's bundled NCCL, so it is only ever loaded explicitly via
    # ctypes. The v7 build is the API-split alias window — it
    # exports the new ncclSparseAllReduceIndex/BlockIndex spellings while
    # keeping the pre-split names as deprecated forwarders, so er/eb (and
    # bb, which v5 could not serve) all resolve from one library. Unset
    # when the sibling checkout has no built library — that backend then
    # fails its own loud prepare() check instead of here.
    _SPARSE_LIB="${_SGA_ROOT}/../nccl-sar/libs/v7/libnccl-mask.so"
    if [ -z "${SGA_NCCL_SAR_LIB:-}" ] && [ -f "$_SPARSE_LIB" ]; then
        export SGA_NCCL_SAR_LIB="$_SPARSE_LIB"
    fi
    unset _SPARSE_LIB

    if ! _sga_check_megatron_source; then
        return 1
    fi
    return 0
}

_sga_cleanup() {
    # Restore the caller's nounset option state before dropping the memory of it.
    [ "$_sga_had_u" -eq 1 ] || set +u
    unset _SGA_ROOT _SGA_CHECK _MEGATRON_DIR _sga_sourced _sga_had_u _sga_rc
    unset -f _pythonpath_prepend _sga_check_megatron_source _sga_main _sga_cleanup
}

_sga_rc=0
_sga_main || _sga_rc=$?

# Teardown happens before the terminal verb, so the trailing literal carries
# the failure status and nothing leaks into the calling shell.
if [ "$_sga_sourced" -eq 1 ]; then
    if [ "$_sga_rc" -eq 0 ]; then
        _sga_cleanup
        return 0
    else
        _sga_cleanup
        return 1
    fi
else
    if [ "$_sga_rc" -eq 0 ]; then
        _sga_cleanup
        exit 0
    else
        _sga_cleanup
        exit 1
    fi
fi
