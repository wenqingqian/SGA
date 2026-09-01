#!/usr/bin/env python3
"""Launcher script-shape gate (container-only; run after `source env.sh`).

Pins the plan's central day-one invariants that nothing else exercised:

  a. golden shape: the real launch pipeline over examples/llama3/train_1b_dense.yaml
     (_update_config_train -> _get_args_megatron -> _get_runner_cmd_train) emits
     the expected megatron/torchrun flags and never leaks launcher-only keys;
  b. generated run-script skeleton (shebang/mkdir/cd/PYTHONPATH/cmd/nohup|tee);
  c. regression: run_node tolerates hosts without any *_VISIBLE_DEVICES env
     (UnboundLocalError fixed) and still honors them when present;
  d. regression: runner.no_shared_fs=true must not inject --no-shared-fs into
     the megatron command line (the strict fork parser rejects it);
  e. regression: the audit tool's parser equals the training-entry parser
     under both modelopt-provider-present and provider-absent environments.

Run: python tests/test_launcher_shape.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf

import audit_megatron_args

from sga.launcher.backend_megatron import MegatronBackend, _get_args_megatron, _update_config_train
from sga.launcher.launcher_ssh import SshLauncher, _get_runner_cmd_train, run_node
from sga.launcher.utils import get_pkg_dir

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "llama3", "train_1b_dense.yaml")

BASE = """
experiment:
  exp_name: t
  exp_dir: PLACEHOLDER_EXP_DIR
  task: {type: train, backend: megatron, entrypoint: ./sga/train_gpt.py}
  runner: {type: ssh}
sga:
  score: {enabled: false}
  selection: {enabled: false}
  optimizer: {enabled: false}
  backend: {name: dense}
train:
  system: {tensor_model_parallel_size: 1}
  model: {num_layers: 2}
  data: {data_path: /tmp/x}
"""


def load_example_config(exp_dir):
    """Day-one yaml with exp_dir redirected to a private temp dir.

    Deliberately NO container round-trip:OmegaConf.to_container(resolve=True)
    would turn the yaml's escaped ``\\${VAR}`` sequences in cmds.before_start
    into live interpolations, which the pipeline never sees.
    """
    cfg = OmegaConf.load(EXAMPLE)
    cfg.experiment.exp_dir = exp_dir
    # hostfile.txt is deliberately absent; resources fall back to localhost,
    # matching how parse_hostfile degrades (same warning path as production).
    return cfg


def test_golden_shape():
    tmp = tempfile.mkdtemp(prefix="sga-shape-exp-")
    try:
        cfg = load_example_config(os.path.join(tmp, "exp"))
        _update_config_train(cfg)
        args = _get_args_megatron(cfg)

        for token in [
            "--seed", "1234",
            "--exit-interval", "200",
            "--rotary-base", "500000",
            "--num-layers", "16",
            "--seq-length", "2048",
            "--tokenizer-type", "HuggingFaceTokenizer",
            "--tokenizer-model",
            "--tensorboard-dir",
            "--save",  # launcher-injected checkpoint dir survives flattening
        ]:
            assert token in args, f"missing essential flag/token in megatron args: {token}"

        for forbidden in [
            "--type", "--backend", "--hostfile", "--ssh-port",
            "--no-shared-fs",
            "--log-dir", "--details-dir", "--pids-dir", "--scripts-dir",
            "--nsys-bin-path", "--nsys-rep-file-path",
            "--enable-perf-monitor", "--perf-log-interval", "--perf-log-dir",
        ]:
            assert forbidden not in args, f"launcher-only key leaked: {forbidden}"

        runner_cmd = _get_runner_cmd_train("localhost", "localhost", 29510, 1, 0, 8, cfg)
        assert runner_cmd[0] == "torchrun"
        pairs = dict(zip(runner_cmd[1::2], runner_cmd[2::2]))
        # flags come straight from runner_config keys (underscore style)
        assert pairs["--nnodes"] == "1"
        assert pairs["--node_rank"] == "0"
        assert pairs["--nproc_per_node"] == "8"
        assert pairs["--rdzv_endpoint"] == "localhost:29510"
        assert "--log_dir" in pairs  # torchrun consumes this one legitimately
        assert "--tee" in pairs
        forbidden_runner_tokens = [
            "--type", "--backend", "--hostfile",
            "--ssh-port", "--ssh_port",
            "--per-node-task", "--per_node_task",
            "--master-addr", "--master-addr".replace("-", "_"),
            "--master-port", "--master-port".replace("-", "_"),
            "--nsys-bin-path", "--nsys-rep-file-path",
            "--enable-perf-monitor", "--perf-log-dir",
        ]
        for forbidden in forbidden_runner_tokens:
            assert forbidden not in runner_cmd, f"launcher-only key leaked: {forbidden}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: golden launch shape (megatron args + torchrun runner cmd)")


def test_run_script_skeleton():
    tmp = tempfile.mkdtemp(prefix="sga-shape-exp-")
    try:
        cfg = load_example_config(os.path.join(tmp, "exp"))
        backend = MegatronBackend(cfg)
        launcher = SshLauncher(cfg, backend)
        pkg_dir = get_pkg_dir()

        background_script = backend.generate_run_script(
            cfg, "localhost", 0, cmd="echo hi", background=True, pkg_dir=pkg_dir
        )
        lines = open(background_script).read().splitlines()
        assert lines[0] == "#!/bin/bash"
        mkdir_lines = [line for line in lines if line.startswith("mkdir -p ")]
        assert len(mkdir_lines) >= 6, "expected >=6 mkdir -p skeleton lines"
        assert any(line.startswith(f"cd {pkg_dir}") for line in lines)
        assert any(
            line.startswith("export PYTHONPATH=") and line.endswith(":${PYTHONPATH}")
            for line in lines
        )
        # shlex.quote keeps the payload shell-safe (e.g. the --sga-config JSON)
        assert "cmd='echo hi'" in lines
        nohup_lines = [line for line in lines if line.startswith('nohup bash -c "$cmd; sync"')]
        assert len(nohup_lines) == 1 and ">> " in nohup_lines[0] and "2>&1" in nohup_lines[0]

        foreground_script = backend.generate_run_script(
            cfg, "localhost", 0, cmd="echo hi", background=False, pkg_dir=pkg_dir
        )
        flines = open(foreground_script).read().splitlines()
        assert any("| tee -a " in line for line in flines)

        stop_script = backend.generate_stop_script("localhost", 0)
        slines = open(stop_script).read().splitlines()
        assert any(line.startswith("if [ -f ") for line in slines)
        assert any("pid=$(cat " in line for line in slines)
        query_script = launcher._generate_query_script("localhost", 0)
        qlines = open(query_script).read().splitlines()
        assert any("ps -p $pid" in line for line in qlines)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: generated script skeleton (run/stop/query)")


def test_run_node_without_visible_devices():
    captured = {}

    def fake_func(host, master_addr, master_port, nnodes, node_rank, nproc_per_node,
                  device_type=None, background=True, dryrun=False, cur_envs=None):
        captured.update(
            host=host, master_addr=master_addr, master_port=master_port,
            nnodes=nnodes, node_rank=node_rank, nproc_per_node=nproc_per_node,
            device_type=device_type, cur_envs=cur_envs,
        )

    resource_info = {"slots": 8, "type": None}
    runner_config = OmegaConf.create({})

    # No *_VISIBLE_DEVICES anywhere: previously UnboundLocalError.
    run_node(fake_func, 0, "localhost", resource_info, {}, runner_config,
             1, "localhost", 29510, True, True, {})
    assert captured["nproc_per_node"] == 8
    assert captured["cur_envs"] == {}
    print("ok: run_node without *_VISIBLE_DEVICES (no UnboundLocalError)")

    run_node(fake_func, 0, "localhost", resource_info, {"CUDA_VISIBLE_DEVICES": "0,1"},
             runner_config, 1, "localhost", 29510, True, True, {"CUDA_VISIBLE_DEVICES": "0,1"})
    assert captured["nproc_per_node"] == 2
    print("ok: run_node honors CUDA_VISIBLE_DEVICES when present")


def test_no_shared_fs_never_reaches_cmdline():
    tmp = tempfile.mkdtemp(prefix="sga-shape-exp-")
    try:
        data = OmegaConf.to_container(OmegaConf.create(BASE), resolve=True)
        data["experiment"]["exp_dir"] = os.path.join(tmp, "exp")
        data["experiment"]["runner"]["no_shared_fs"] = True
        data["train"]["system"]["logging"] = {}
        cfg = OmegaConf.create(data)
        _update_config_train(cfg)
        args = _get_args_megatron(cfg)
        assert "--no-shared-fs" not in args, (
            "runner.no_shared_fs injected a --no-shared-fs flag "
            "(the strict fork parser rejects unknown args)"
        )
        # sanity: pipeline still injected the standard directories
        assert "--save" in args and "--load" in args
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: runner.no_shared_fs=true leaves no --no-shared-fs in megatron args")


def test_audit_parser_matches_training_entry():
    """The audit tool's parser must equal the training entry's parser under
    both provider-present and provider-absent environments."""
    detect = audit_megatron_args.detect_modelopt()

    base_table = audit_megatron_args.dump_table(
        audit_megatron_args.build_megatron_parser(False)
    )
    chosen_table = audit_megatron_args.dump_table(audit_megatron_args.load_training_entry_parser())

    if detect:
        full_table = audit_megatron_args.dump_table(
            audit_megatron_args.build_megatron_parser(True)
        )
        assert set(chosen_table) == set(full_table), (
            "training-entry parser must include modelopt extras when available"
        )
        extra_flags = set(full_table) - set(base_table)
        assert extra_flags, "modelopt provider added no flags (wired wrong?)"
        print(f"ok: audit parser matches training entry (provider PRESENT, +{len(extra_flags)} flags)")
    else:
        assert set(chosen_table) == set(base_table), (
            "provider-absent environment must behave exactly like the training entry"
        )
        print("ok: audit parser matches training entry (provider ABSENT, identical tables)")


if __name__ == "__main__":
    test_golden_shape()
    test_run_script_skeleton()
    test_run_node_without_visible_devices()
    test_no_shared_fs_never_reaches_cmdline()
    test_audit_parser_matches_training_entry()
    print("test_launcher_shape: ALL PASS")
