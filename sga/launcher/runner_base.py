"""Runner: config -> MegatronBackend + SshLauncher.

Surgery note (from flagscale/runner/runner_base.py @ a161153): removed the
RunnerFactory registry and every non-train task type. SGA supports exactly
one backend (megatron) and one launcher (ssh), wired directly.
"""

from abc import ABC

from omegaconf import DictConfig

from sga.launcher.backend_megatron import MegatronBackend
from sga.launcher.launcher_ssh import SshLauncher

TASK_TO_BACKEND_MAP = {
    "train": ["megatron"],
}


class Runner(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = self.config.experiment.task.get("type", None)
        assert self.task_type in TASK_TO_BACKEND_MAP, f"Unsupported task type: {self.task_type}"

        backend_attr = self.config.experiment.task.get("backend", None)

        assert backend_attr is not None, (
            f"backend_type is required for task_type='{self.task_type}'. "
            f"Allowed backends: {TASK_TO_BACKEND_MAP[self.task_type]}"
        )

        allowed_backends = TASK_TO_BACKEND_MAP[self.task_type]
        assert backend_attr in allowed_backends, (
            f"Unsupported backend type '{backend_attr}' for task_type='{self.task_type}'. "
            f"Allowed backends: {allowed_backends}"
        )

        self.backend = MegatronBackend(self.config)
        self.launcher = SshLauncher(self.config, self.backend)

    def run(self, *args, **kwargs):
        return self.launcher.run(*args, **kwargs)

    def stop(self, *args, **kwargs):
        """Optional method to override."""
        return self.launcher.stop(*args, **kwargs)

    def query(self, *args, **kwargs):
        """Optional method to override."""
        return self.launcher.query(*args, **kwargs)
