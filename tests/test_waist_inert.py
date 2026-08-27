#!/usr/bin/env python3
"""Fork-side inertness tests: without --sga-config the waist is a no-op.

The megatron fork's contract (notes/sga-waist-api-v1.md): the unified plugin
interface must be invisible to pure dense runs. Verifies on the real fork
source (PYTHONPATH includes Megatron-LM):

- add_sga_args adds exactly the two reserved flags, default-inactive;
- without a config the waist never builds an engine, never imports ``sga``,
  and attach/grad_sync/optimizer_step are inert;
- _config_from_args parses the canonical JSON string (never a filesystem
  path) and rejects malformed payloads with the offending input named.

Run in the container: PYTHONPATH=SGA:Megatron-LM python tests/test_waist_inert.py
"""

import os
import sys
from types import SimpleNamespace

SGA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SGA_ROOT)
sys.path.insert(0, os.path.dirname(SGA_ROOT) + "/Megatron-LM")

import megatron.core.sga_waist as sga_waist
import megatron.training as _mt


def _restore_modules(snapshot):
    """Restore sys.modules exactly (alphabetically-earlier suites may have
    imported sga.runtime; later assertions must see their own reality)."""
    for name in list(sys.modules):
        if name not in snapshot:
            del sys.modules[name]
    sys.modules.update(snapshot)


def _purge_sga_modules():
    """Drop pre-imported sga modules so lazy-import assertions observe only
    what the exercised path itself triggers; _restore_modules brings the
    original entries back from the test's snapshot."""
    for name in list(sys.modules):
        if name == "sga" or name.startswith("sga."):
            del sys.modules[name]


class _StubGroup:
    def __init__(self):
        self.flags = []

    def add_argument(self, *flags, **kw):
        self.flags.extend(flags)


class _StubParser:
    def __init__(self):
        self.groups = []

    def add_argument_group(self, title):
        group = _StubGroup()
        self.groups.append((title, group))
        return group


class _Any:
    """Any attribute touch is a failure (inertness canary)."""

    def __getattr__(self, name):
        raise AssertionError(f"inert waist touched .{name} of a dummy object")


def _reset_waist():
    sga_waist._engine = None
    sga_waist._engine_failed = False


def _with_args(**kw):
    saved = _mt.get_args
    _mt.get_args = lambda: SimpleNamespace(**kw)
    return saved


def test_add_sga_args_flags():
    parser = sga_waist.add_sga_args(_StubParser())
    flags = [f for _, g in parser.groups for f in g.flags]
    assert sorted(flags) == ["--sga-allow-config-override", "--sga-config"]
    titles = [t for t, _ in parser.groups]
    assert titles == ["SGA"]


def test_inactive_without_config():
    _reset_waist()
    snapshot = dict(sys.modules)
    _purge_sga_modules()
    saved = _with_args(sga_config="", sga_allow_config_override=False)
    try:
        assert sga_waist._config_from_args() is None
        assert sga_waist._get_engine() is None
        assert sga_waist.active() is False
        assert sga_waist.attach(_Any()) is None
        assert sga_waist.grad_sync(_Any(), force_all_reduce=False) is False
        assert sga_waist.optimizer_step(_Any(), [_Any()], 1) is None
        assert "sga.runtime" not in sys.modules  # never imported when inert
    finally:
        _mt.get_args = saved
        _restore_modules(snapshot)
        _reset_waist()


def test_fail_fast_when_sga_unimportable():
    """--sga-config set but the sga package unimportable: the failure must
    surface at engine-build time with an actionable message, never at the
    first optimizer step."""
    _reset_waist()
    snapshot = dict(sys.modules)
    canonical = '{"cycle": {"start_iter": 0, "period": 1}}'
    saved = _with_args(sga_config=canonical, sga_allow_config_override=False)
    # Block the submodule import: a None entry in sys.modules makes any
    # import of it raise ImportError ("halted"), simulating a missing package.
    sys.modules["sga.runtime"] = None
    try:
        try:
            sga_waist._get_engine()
            raise AssertionError("unimportable sga did not fail fast")
        except ImportError as e:
            assert "--sga-config" in str(e) and "PYTHONPATH" in str(e), e
        assert sga_waist._engine_failed is True  # never retried silently
        assert sga_waist.active() is False
    finally:
        _mt.get_args = saved
        _restore_modules(snapshot)
        _reset_waist()


def test_api_version_mismatch_rejected_at_build():
    """The fork speaks exactly one API version; a mismatching build request
    fails loudly (contract pinning for the version handshake)."""
    # Import-scope hygiene: this test legitimately imports sga.runtime, but
    # the inertness canary below must still observe an unimported package.
    snapshot = dict(sys.modules)
    try:
        from sga.runtime import SGA_API_VERSION, build_engine

        config = {
            "score": {"name": "abs_g"},
            "selection": {"name": "threshold"},
            "error_feedback": {"name": "masked_adamw"},
            "backend": {"name": "masked_pipelined"},
            "cycle": {},
        }
        try:
            build_engine(config, api_version=SGA_API_VERSION + 1)
            raise AssertionError("API-version mismatch was not rejected")
        except ValueError as e:
            assert "version" in str(e), e
    finally:
        _restore_modules(snapshot)


def test_config_from_args_parses_json_string_not_path():
    _reset_waist()
    canonical = '{"cycle": {"start_iter": 0, "period": 1, "full_sync_interval": 10}}'
    saved = _with_args(sga_config=canonical, sga_allow_config_override=False)
    try:
        cfg = sga_waist._config_from_args()
        assert isinstance(cfg, dict) and cfg["cycle"]["period"] == 1
        # a filesystem path is not a JSON payload
        _mt.get_args = lambda: SimpleNamespace(
            sga_config="/tmp/nonexistent.json", sga_allow_config_override=False)
        try:
            sga_waist._config_from_args()
            raise AssertionError("a path was accepted as the config payload")
        except ValueError:
            pass
    finally:
        _mt.get_args = saved
        _reset_waist()


def test_runtime_args_guards():
    """Build-time rejection set on the real _assert_args_compatible.

    vpp: megatron leaves the attribute ABSENT or None when the feature is
    off, so both off-forms must pass while vpp > 1 must fail loudly. fp16
    is rejected: the dynamic-loss-scaling unscale decays the residual
    through its aliased fp32 bucket buffer.
    """
    from unittest.mock import patch

    from megatron.core.rerun_state_machine import RerunMode
    from sga.runtime import _assert_args_compatible

    def _args(**over):
        base = dict(
            use_distributed_optimizer=False, overlap_grad_reduce=False,
            calculate_per_token_loss=False,
            reuse_grad_buf_for_mxfp8_param_ag=False,
            tensor_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            expert_model_parallel_size=1, num_experts=None,
            pipeline_model_parallel_size=1,
            param_name_patterns_for_fp32_local_accumulation=None,
            cuda_graph_impl="none", fp16=False, bf16=True,
            accumulate_allreduce_grads_in_fp32=True,
            check_for_nan_in_grad=False,
            check_for_large_grads_in_bucket=False,
            optimizer="adam", decoupled_weight_decay=True,
        )
        base.update(over)
        return SimpleNamespace(**base)

    saved = _mt.get_args
    try:
        with patch("megatron.core.rerun_state_machine.get_rerun_state_machine") as g:
            g.return_value.mode = RerunMode.DISABLED
            # every off-form of vpp passes
            ns = _args()
            del ns.virtual_pipeline_model_parallel_size
            for a in (ns, _args(virtual_pipeline_model_parallel_size=None),
                      _args(virtual_pipeline_model_parallel_size=1)):
                _mt.get_args = lambda a=a: a
                _assert_args_compatible()
            # vpp > 1 fails loudly at build time
            _mt.get_args = lambda: _args(virtual_pipeline_model_parallel_size=2)
            try:
                _assert_args_compatible()
                raise AssertionError("vpp=2 was not rejected")
            except ValueError as e:
                assert "virtual-pipeline" in str(e), e
            # fp16 fails loudly (even with bf16 also set: fp16 wins in megatron)
            _mt.get_args = lambda: _args(fp16=True)
            try:
                _assert_args_compatible()
                raise AssertionError("fp16 was not rejected")
            except ValueError as e:
                assert "fp16" in str(e), e
    finally:
        _mt.get_args = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
