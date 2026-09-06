"""SGA hydra entry: single task type (train), single runner type (ssh).

Surgery note (from flagscale/run.py @ a161153): removed autotuner,
inference/serve/compress/rl runners, cloud runner, legacy runner map and the
FLAGSCALE_USE_V1 switch. SGA always dispatches to Runner, which wires
MegatronBackend + SshLauncher directly (no factory registry).
"""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from sga.launcher.runner_base import Runner
from sga.logger import logger

# To accommodate the scenario where the before_start field is used to switch to the actual environment during program execution,
# we have placed the import statements inside the function body rather than at the beginning of the file.

VALID_TASKS = {"train"}

# task_type -> allowed actions
TASK_ACTIONS = {
    "train": {"run", "dryrun", "test", "stop", "query"},
}


def validate_task(task_type: str, action: str) -> None:
    if task_type not in VALID_TASKS:
        raise ValueError(f"Invalid task_type '{task_type}', must be one of {sorted(VALID_TASKS)}")

    allowed_actions = TASK_ACTIONS[task_type]
    if action not in allowed_actions:
        raise ValueError(
            f"Action '{action}' is not allowed for task_type '{task_type}'. "
            f"Allowed actions: {sorted(allowed_actions)}"
        )


def get_runner(config: DictConfig, task_type: str):
    runner_type = config.experiment.runner.get("type", "ssh")
    if runner_type != "ssh":
        raise NotImplementedError(
            f"Runner type '{runner_type}' is not supported. SGA only supports the ssh runner."
        )
    return Runner(config)


def execute_action(runner, action: str, task_type: str, config: DictConfig) -> None:
    if action == "run":
        runner.run()
    elif action == "dryrun":
        runner.run(dryrun=True)
    elif action == "test":
        runner.run(background=False)
    elif action == "stop":
        runner.stop()
    elif action == "query":
        runner.query()
    else:
        raise ValueError(f"Unknown action '{action}'")


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    # Restore the user's original working directory.
    # Hydra defaults to hydra.job.chdir=True, which changes cwd to a run-specific
    # output dir (e.g. outputs/2024-01-01/12-00-00/).  This breaks all relative
    # paths in configs (tokenizer files, data paths, hostfiles, entrypoints, etc.).
    # SGA manages output directories explicitly (exp_dir, log_dir, etc.),
    # so the cwd change provides no benefit.  Restoring cwd here is equivalent to
    # hydra.job.chdir=False but works for all invocation paths (CLI, direct python,
    # tests) without requiring Hydra config file changes.
    os.chdir(hydra.utils.get_original_cwd())

    try:
        _main(config)
    except OmegaConfBaseException as e:
        logger.error(f"Config validation error: {e}")
        try:
            config_yaml = OmegaConf.to_yaml(config, resolve=False)
        except Exception:
            config_yaml = repr(config)
        logger.error(f"Config content:\n{config_yaml}")
        raise SystemExit(
            f"Config error: {e}\nHint: Check your YAML config files for missing or invalid fields."
        ) from e


def _main(config: DictConfig) -> None:
    """Core logic, separated from main() for error wrapping.

    The sga: section is pre-validated inside the backend's flag assembly
    (backend_megatron._prevalidate_sga), still fail-fast before any
    process is spawned."""
    task_type = config.experiment.task.get("type", None)
    action = config.action
    validate_task(task_type, action)

    runner = get_runner(config, task_type)
    execute_action(runner, action, task_type, config)


def resolve_config_path_in_argv():
    """Convert relative --config-path in sys.argv to an absolute path.

    Hydra interprets a relative config_path as relative to the calling
    module's package directory, not the current working directory.
    Converting to an absolute path ensures it is treated as a filesystem path.
    """
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config-path="):
            path = arg[len("--config-path=") :]
            if not os.path.isabs(path):
                sys.argv[i] = f"--config-path={os.path.abspath(path)}"
        elif arg == "--config-path" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            if not os.path.isabs(path):
                sys.argv[i + 1] = os.path.abspath(path)


if __name__ == "__main__":
    resolve_config_path_in_argv()
    main()
