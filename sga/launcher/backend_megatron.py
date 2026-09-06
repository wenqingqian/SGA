"""Megatron train backend: config -> megatron args + per-host run/stop scripts.

Surgery note (from flagscale/runner/backend/backend_megatron.py @ a161153,
with `_get_args_megatron` / `_update_config_train` merged in from
flagscale/runner/runner_train.py): removed the perf-monitor config block,
the elastic monitor_launcher hook, and straggler logging. The generated
script no longer prepends a ``flagscale/train`` directory to PYTHONPATH —
the SGA repo root (which carries the ``sga`` package) is prepended instead;
Megatron-LM must be provided by the user's ``cmds.before_start`` (normally
``source env.sh``).
"""

import os
import shlex

from omegaconf import DictConfig, OmegaConf

from sga.launcher.backend_base import BackendBase
from sga.launcher.utils import (
    flatten_dict_to_args,
    get_pkg_dir,
    parse_hostfile,
    resolve_path,
    setup_exp_dir,
    setup_logging_dirs,
)
from sga.logger import logger


def _sga_section_to_flags(config) -> list:
    """Render the yaml ``sga:`` section into the exact ``--sga-*`` flag
    list the training entry will see (FLAT keys — the same vocabulary as
    the flags, underscores instead of hyphens). The sga_ prefix contract
    is re-asserted at this transport point: every yaml sga: key IS the
    full flat dest, no short form exists."""
    sga = OmegaConf.to_container(config.sga, resolve=True)
    assert all(isinstance(k, str) and k.startswith("sga_") for k in sga), (
        f"sga: keys must be the full flat sga_* names (got {sorted(sga)})")
    from sga.config.arguments import sga_int_or_null
    from sga.config.validation import sga_actions

    flags = []
    for dest in sorted(sga):
        value = sga[dest]
        action = sga_actions()[dest]  # unknown keys already rejected by parse
        if value is None:
            if dest == "sga_enabled":
                raise ValueError(
                    "sga_enabled is explicitly null in the sga: section — "
                    "spell it true or false (null silently means disabled "
                    "here)")
            if action.type is sga_int_or_null:
                flags.extend([action.option_strings[0], "null"])
            # any other flag's explicit null IS the unset spelling —
            # leave it at the declared default (emitting a literal would
            # fail its typing)
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        else:
            value = str(value)
        flags.extend([action.option_strings[0], value])
    return flags


def _prevalidate_sga(config) -> None:
    """Launch-time pre-validation of the ``sga:`` section: parse the very
    flags about to be emitted with the REAL parser (one argparse code
    path — the training entry sees identical values), then run the same
    ``validate_sga_args`` build_engine will run. Fails fast before any
    process is spawned; disabled configs validate trivially (megatron
    semantics: zero checks under sga_enabled false)."""
    sga = config.get("sga", None)
    if sga is None:
        raise ValueError(
            "Config is missing the 'sga:' section (declare it with just "
            "sga_enabled: false for pure dense training)."
        )
    # A scalar/ill-typed section would die inside OmegaConf.to_container
    # with an opaque internal error — refuse it by name; an EMPTY mapping
    # is the missing section wearing braces and meets the same loudness.
    if isinstance(sga, DictConfig):
        sga = OmegaConf.to_container(sga, resolve=True)
    if not isinstance(sga, dict):
        raise ValueError(
            f"sga section must be a mapping (got {type(sga).__name__})")
    if not sga:
        raise ValueError(
            "Config has an empty 'sga:' section (declare it with just "
            "sga_enabled: false for pure dense training)."
        )
    # `sga:` must not leak into the megatron arg flattening; the launcher
    # only reads train.{system,model,data}, but assert the boundary here
    # so a future flatten change cannot silently start forwarding it.
    assert "sga" not in OmegaConf.to_container(config.train, resolve=True), \
        "sga section leaked into train.*"

    import argparse

    from sga.config.arguments import add_sga_args
    from sga.config.validation import validate_sga_args

    class _LoudParser(argparse.ArgumentParser):
        """argparse's own complaints (typing, choices) as ValueError with
        the FLAG NAMED — a bare SystemExit(2) would swallow the message
        and name no key."""

        def error(self, message):
            raise ValueError(message)

    parser = _LoudParser(add_help=False)
    add_sga_args(parser)
    try:
        ns = parser.parse_args(_sga_section_to_flags(config))
    except KeyError as e:  # a yaml key no declared flag owns
        raise ValueError(
            f"unknown sga: key {e} — the keys must be the full flat sga_* "
            "flag dests declared in sga/config/arguments.py") from None
    # The two non-sga keys validation reads, from the same run config —
    # selected through both train subtrees in _get_args_megatron's
    # flattening order (system first, model overrides): the shipped
    # layout carries pipeline_model_parallel_size under train.system,
    # and a single-subtree select would leave the pp>1 rule dead at
    # launch time.
    it = OmegaConf.select(config, "train.system.train_iters", default=None)
    ns.train_iters = OmegaConf.select(
        config, "train.model.train_iters", default=it)
    pp = OmegaConf.select(
        config, "train.system.pipeline_model_parallel_size", default=1)
    ns.pipeline_model_parallel_size = OmegaConf.select(
        config, "train.model.pipeline_model_parallel_size", default=pp)
    validate_sga_args(ns)


def _get_args_megatron(config: DictConfig):
    assert config.experiment.task.backend == "megatron", (
        "This function only supports megatron backend."
    )

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict["train"]

    new_config_dict = {}
    new_config_dict.update(config_dict["system"])
    new_config_dict.update(config_dict["model"])
    new_config_dict.update(config_dict["data"])

    ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir", "no_shared_fs"]
    # Flatten the dictionary to a list of arguments
    args = flatten_dict_to_args(new_config_dict, ignore_keys)

    # The sga: section becomes the visible flags themselves, pre-validated
    # here (fail-fast before any process is spawned; the training entry
    # re-validates on its parsed namespace). There is no translation
    # layer and no JSON form.
    _prevalidate_sga(config)
    args.extend(_sga_section_to_flags(config))

    return args


def _update_config_train(config: DictConfig):
    exp_dir = setup_exp_dir(config)

    OmegaConf.set_struct(config, False)

    system = config.train.system

    if system.get("checkpoint", None) is None:
        system.checkpoint = DictConfig({})

    if system.get("logging", None) is None:
        system.logging = DictConfig({})

    # Checkpoint directories
    system.checkpoint.save = (
        resolve_path(system.checkpoint.save, "checkpoint.save")
        if system.checkpoint.get("save", None)
        else os.path.join(exp_dir, "checkpoints")
    )
    system.checkpoint.load = (
        resolve_path(system.checkpoint.load, "checkpoint.load")
        if system.checkpoint.get("load", None)
        else os.path.join(exp_dir, "checkpoints")
    )

    # Logging directories
    log_dir = setup_logging_dirs(system.logging, exp_dir)
    system.logging.details_dir = os.path.join(log_dir, "details")
    system.logging.tensorboard_dir = (
        resolve_path(system.logging.tensorboard_dir, "logging.tensorboard_dir")
        if system.logging.get("tensorboard_dir", None)
        else os.path.join(exp_dir, "tensorboard")
    )
    system.logging.wandb_save_dir = (
        resolve_path(system.logging.wandb_save_dir, "logging.wandb_save_dir")
        if system.logging.get("wandb_save_dir", None)
        else os.path.join(exp_dir, "wandb")
    )

    # Tokenizer file paths — resolve before passing to the training subprocess,
    # which may run with a different cwd (e.g. site-packages when pip-installed).
    data = config.train.get("data", None)
    if data:
        tokenizer = data.get("tokenizer", None)
        if tokenizer:
            _TOKENIZER_FILE_KEYS = (
                "vocab_file",
                "merge_file",
                "special_tokens_file",
                "tokenizer_model",
            )
            for key in _TOKENIZER_FILE_KEYS:
                val = tokenizer.get(key, None)
                if val is not None:
                    tokenizer[key] = resolve_path(val, f"data.tokenizer.{key}", raise_missing=True)

    OmegaConf.set_struct(system, False)


class MegatronBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "train", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        _update_config_train(self.config)
        self.user_args = _get_args_megatron(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_script = self.config.experiment.task.entrypoint
        self.resources = parse_hostfile(self.config.experiment.runner.get("hostfile", None))
        self.device_type_specific = self.config.get("device_type_specific", None)
        self.node_specific = self.config.get("node_specific", None)
        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def generate_run_script(
        self,
        config,
        host,
        node_rank,
        cmd,
        background=False,
        pkg_dir=None,
    ):
        system_config = config.train.system
        logging_config = config.train.system.logging

        no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
        if no_shared_fs:
            host_output_file = os.path.join(logging_config.log_dir, "host.output")
        else:
            host_output_file = os.path.join(
                logging_config.log_dir, f"host_{node_rank}_{host}.output"
            )
        host_run_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
        )
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)
        pkg_dir = (
            get_pkg_dir()
            if pkg_dir is None
            else resolve_path(pkg_dir, "build_dir", raise_missing=True)
        )
        assert os.path.exists(pkg_dir), f"PKG_DIR {pkg_dir} does not exist."
        cmds_config = config.experiment.get("cmds", None)
        if cmds_config:
            before_start = cmds_config.get("before_start", "")
        else:
            before_start = ""
        with open(host_run_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"{before_start}\n")
            # shlex.quote keeps today's safe path values byte-identical while
            # making whitespace/metacharacter-bearing paths (or injected
            # config values) inert.
            f.write(f"mkdir -p {shlex.quote(system_config.checkpoint.load)}\n")
            f.write(f"mkdir -p {shlex.quote(system_config.checkpoint.save)}\n")
            f.write(f"mkdir -p {shlex.quote(system_config.logging.log_dir)}\n")
            f.write(f"mkdir -p {shlex.quote(system_config.logging.pids_dir)}\n")
            f.write(f"mkdir -p {shlex.quote(system_config.logging.details_dir)}\n")
            f.write(f"mkdir -p {shlex.quote(system_config.logging.tensorboard_dir)}\n")
            f.write(f"mkdir -p {shlex.quote(system_config.logging.wandb_save_dir)}\n")
            f.write("\n")
            f.write(f"cd {shlex.quote(pkg_dir)}\n")
            f.write("\n")
            f.write(f"export PYTHONPATH={shlex.quote(pkg_dir)}:${{PYTHONPATH}}\n")
            f.write("\n")
            # shlex.quote: the cmd already embeds shlex-quoted words (e.g.
            # flat --sga-* flag values); a plain double-quoted assignment
            # would eat quoted values' inner quotes.
            f.write(f"cmd={shlex.quote(cmd)}\n")
            f.write("\n")

            if background:
                f.write(
                    f'nohup bash -c "$cmd; sync" >> {shlex.quote(host_output_file)} 2>&1'
                    f' & echo $! > {shlex.quote(host_pid_file)}\n'
                )
            else:
                # capture the training's own exit status: `bash -c "$cmd;
                # sync"` reports sync's rc (0) and masks a crash
                f.write("set -o pipefail\n")
                f.write(f'bash -c "$cmd" 2>&1 | tee -a {shlex.quote(host_output_file)}\n')
                f.write("rc=$?\nsync\nexit $rc\n")
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_run_script_file, 0o755)

        return host_run_script_file

    def generate_stop_script(self, host, node_rank):
        if getattr(self.config, "train", None):
            logging_config = self.config.train.system.logging
        else:
            raise ValueError("stop requires a train config")

        host_stop_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        cmds_config = self.config.experiment.get("cmds", None)
        if cmds_config:
            after_stop = cmds_config.get("after_stop", "")
        else:
            after_stop = ""
        with open(host_stop_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + shlex.quote(host_pid_file) + " ]; then\n")
            f.write("    pid=$(cat " + shlex.quote(host_pid_file) + ")\n")
            f.write("    pkill -P $pid\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to stop the job.
            f.write("    pkill -f 'torchrun'\n")
            f.write("fi\n")
            f.write(f"{after_stop}\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_stop_script_file, 0o755)

        return host_stop_script_file
