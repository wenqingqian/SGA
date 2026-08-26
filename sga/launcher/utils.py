import collections
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
from enum import Enum

from omegaconf import DictConfig, OmegaConf

from sga.logger import logger


def resolve_path(path, config_key="", check_exists=False, raise_missing=False):
    """Resolve a config path to an absolute path and optionally validate it.

    Handles all path forms consistently:
      - Absolute paths (starting with '/') are returned as-is.
      - Relative paths ('./…', '../…', or bare like 'data/…') are resolved
        against the current working directory via ``os.path.abspath``.

    Note: ``run.py`` restores the original cwd before any runner code executes
    (equivalent to ``hydra.job.chdir=False``), so ``os.path.abspath`` always
    resolves against the user's project directory, not Hydra's output dir.

    Args:
        path: The path string to resolve.
        config_key: Dotted config key for log messages (e.g. 'train.data.tokenizer.vocab_file').
        check_exists: If True, log a warning when the resolved path does not exist.
        raise_missing: If True, raise FileNotFoundError when the resolved path does not exist.
            Takes precedence over *check_exists*.
    """
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        if raise_missing:
            raise FileNotFoundError(
                f"Config '{config_key}': resolved path '{path}' to '{resolved}', "
                f"but the path does not exist."
            )
        if check_exists:
            logger.warning(
                f"Config '{config_key}': resolved path '{path}' to '{resolved}', "
                f"but the path does not exist."
            )
    return resolved


def setup_exp_dir(config):
    """Resolve experiment directory and ensure it exists.

    Common setup shared by all task types: resolves ``config.experiment.exp_dir``
    to an absolute path, creates it if necessary, and returns the resolved path.
    """
    exp_dir = resolve_path(config.experiment.exp_dir, "experiment.exp_dir")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def get_pkg_dir():
    """Return the root directory of the SGA repo.

    Used to locate package-internal files (the ``sga`` package, scripts, etc.)
    in generated run scripts for PYTHONPATH, ``cd``, and script paths.

    Resolves to the parent of the ``sga/`` package directory (repo root).
    """
    # sga/launcher/utils.py -> sga/launcher/ -> sga/ -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging_dirs(logging_config, exp_dir, log_subdir="logs"):
    """Set up standard logging directory layout under exp_dir.

    Creates the directory tree and writes paths back into *logging_config*::

        <exp_dir>/<log_subdir>/
        ├── scripts/
        └── pids/

    Args:
        logging_config: The OmegaConf logging section to update in-place.
        exp_dir: Absolute path to the experiment directory.
        log_subdir: Name of the logging subdirectory under exp_dir.

    Returns:
        The absolute log_dir path.
    """
    log_dir = (
        resolve_path(logging_config.log_dir, "logging.log_dir")
        if logging_config.get("log_dir", None)
        else os.path.join(exp_dir, log_subdir)
    )
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    logging_config.log_dir = log_dir
    logging_config.scripts_dir = scripts_dir
    logging_config.pids_dir = pids_dir

    return log_dir


class JobStatus(Enum):
    RUNNING = "Running"
    TRANSITIONAL = "Transitional (Stopping or Starting)"
    COMPLETED_OR_IDLE = "Completed or Not Started"


def log_and_raise_error(message):
    logger.error(message)
    raise ValueError(message)


def parse_hostfile(hostfile_path):
    if hostfile_path is None or not os.path.isfile(hostfile_path):
        logger.warning(
            f"Hostfile {hostfile_path} not found. The training will proceed using only local resources."
        )
        return None

    # e.g., worker0 slots=8 type=A100
    pattern = re.compile(r"^(\S+)\s+slots=(\d+)(?:\s+type=(\S+))?")

    resources = collections.OrderedDict()

    with open(hostfile_path, "r") as fd:
        hostfile_lines = fd.readlines()

    for line in hostfile_lines:
        line = line.strip()
        match = pattern.search(line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            machine_type = match.group(3) if match.group(3) else None
            if host in resources:
                log_and_raise_error(f"Hostfile contains multiple entries for host: {host}.")
            resources[host] = {"slots": num_slots, "type": machine_type}
        else:
            log_and_raise_error(f"Invalid entry in hostfile: {line}.")

    assert all(info["type"] is None for _, info in resources.items()) or all(
        info["type"] is not None for _, info in resources.items()
    ), "All hosts must have the a machine type or no machine type specified."

    if len(resources) == 0:
        log_and_raise_error(
            "Hostfile is empty or not formatted correctly. Please check the hostfile."
        )

    return resources


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_local_command(cmd, dryrun=False, query=False, stream_output=False):
    logger.info(f"Run the local command: {cmd}")
    if dryrun:
        return
    if stream_output:
        # Stdout/stderr go directly to the console (no capture).
        result = subprocess.run(cmd, shell=True, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)
        return
    result = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if query:
        return result
    if result.returncode != 0:
        print(f"Command {cmd} failed with return code {result.returncode}.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        sys.exit(result.returncode)


def _get_ssh_env():
    """Return an env dict for system ssh/scp to avoid OpenSSL mismatches."""
    env = os.environ.copy()
    ld_path = env.get("LD_LIBRARY_PATH", "")
    cleaned = ":".join(p for p in ld_path.split(":") if "miniconda3/envs/" not in p)
    if cleaned:
        env["LD_LIBRARY_PATH"] = cleaned
    elif "LD_LIBRARY_PATH" in env:
        del env["LD_LIBRARY_PATH"]
    return env


def run_ssh_command(
    host, cmd, port=None, dryrun=False, query=False, background=True, stream_output=False
):
    # Build SSH command — only background mode adds -f
    flags = "-f -n" if (background and not stream_output) else "-n"
    port_flag = f"-p {port} " if port else ""
    ssh_cmd = f"ssh {flags} {port_flag}{host} '{cmd}'"

    if not query:
        logger.info(f"Running the ssh command: {ssh_cmd}")
    if dryrun:
        return

    ssh_env = _get_ssh_env()

    if stream_output:
        # Stdout/stderr stream directly to the login node console.
        result = subprocess.run(ssh_cmd, shell=True, check=False, env=ssh_env)
        if result.returncode != 0:
            sys.exit(result.returncode)
        return

    result = subprocess.run(
        ssh_cmd,
        shell=True,
        check=(background),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=ssh_env,
    )
    if result.returncode != 0 and not query:
        print(f"SSH command {ssh_cmd} failed with return code {result.returncode}.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        sys.exit(result.returncode)
    if query:
        return result


def run_scp_command(host, src, dst, port=None, dryrun=False):
    if port:
        scp_cmd = f"scp -P {port} -r {src} {host}:{dst} "
    else:
        scp_cmd = f"scp -r {src} {host}:{dst} "
    logger.info(f"Run the scp command: {scp_cmd}")
    if dryrun:
        return
    result = subprocess.run(
        scp_cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_get_ssh_env(),
    )
    if result.returncode != 0:
        print(f"SCP command {scp_cmd} failed with return code {result.returncode}.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        sys.exit(result.returncode)


def flatten_dict_to_args(config_dict, ignore_keys=[], do_dash_replace=True):
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        if value is None:
            continue
        if do_dash_replace:
            key = key.replace("_", "-")
        if isinstance(value, dict):
            args.extend(flatten_dict_to_args(value, ignore_keys, do_dash_replace=do_dash_replace))
        elif isinstance(value, list):
            args.append(f"--{key}")
            for v in value:
                args.append(f"{v}")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(f"{value}")
    return args


def get_nnodes(nnodes_from_hostfile=None, nnodes_from_args=None):
    assert nnodes_from_hostfile is not None or nnodes_from_args is not None
    if nnodes_from_hostfile is not None and nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return min(nnodes_from_hostfile, int(nnodes_from_args))
    elif nnodes_from_hostfile is not None:
        return nnodes_from_hostfile
    elif nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return int(nnodes_from_args)


def get_nproc_per_node(nproc_from_hostfile=None, nproc_from_args=None, num_visible_devices=None):
    if nproc_from_hostfile is not None and nproc_from_args is not None:
        nproc = min(nproc_from_hostfile, int(nproc_from_args))
        if num_visible_devices:
            return min(nproc, num_visible_devices)
        else:
            return nproc
    elif nproc_from_hostfile is not None:
        if num_visible_devices:
            return min(nproc_from_hostfile, num_visible_devices)
        else:
            return nproc_from_hostfile
    elif nproc_from_args is not None:
        if num_visible_devices:
            return min(int(nproc_from_args), num_visible_devices)
        else:
            return nproc_from_args
    else:
        if num_visible_devices:
            return num_visible_devices
        else:
            return 1


def update_nodes_envs(env_config, ip_addr, resource_info):
    device_type = resource_info.get("type", None)
    assert ip_addr is not None, "ip address is required in resource_info"

    cur_node_config = {}
    if isinstance(env_config, DictConfig):
        cur_node_config = OmegaConf.to_container(env_config, resolve=True)
    else:
        cur_node_config = env_config.copy()

    device_types_envs = cur_node_config.pop("device_type_specific", None)
    nodes_envs = cur_node_config.pop("node_specific", None)
    if device_types_envs is None and nodes_envs is None:
        logger.warning(
            "type in hostfile is not specified. All the nodes use the same arguments inlucding evnironment variables."
        )
        return cur_node_config

    # update then envs according to the device type
    if device_types_envs is not None and device_type is not None:
        cur_node_config.update(device_types_envs.get(device_type, {}))
    # update the envs according to the ip address
    if nodes_envs is not None:
        cur_node_config.update(nodes_envs.get(ip_addr, {}))

    return cur_node_config


def update_cmd_with_node_specific_config(cmd, node_specific_config=None):
    """
    Update the command string with additional configuration options for speicial device.
    Parameters:
        cmd (str): The original command string to be updated.
        node_specific_config (dict): A dictionary containing configuration
                                     options to be applied to the command.
    Returns:
        str: The updated command string after applying the configuration.
    """
    if node_specific_config is None or len(node_specific_config) == 0:
        return cmd
    cmd_parts = shlex.split(cmd)

    for key, value in node_specific_config.items():
        key = key.replace("_", "-")
        option = f"--{key}"
        if option in cmd_parts:
            idx = cmd_parts.index(option)
            if value.lower() == "true":
                # If the option has a value, remove it.
                if idx + 1 < len(cmd_parts) and not cmd_parts[idx + 1].startswith("--"):
                    cmd_parts.pop(idx + 1)
                continue
            elif value.lower() == "false":
                # If the option has a value, remove it along with the option.
                if idx + 1 < len(cmd_parts) and not cmd_parts[idx + 1].startswith("--"):
                    del cmd_parts[idx : idx + 2]
                else:
                    cmd_parts.pop(idx)
        else:
            if value.lower() == "true":
                cmd_parts.append(option)
            # NOTE: disable adding new options
            # elif value.lower() != "false":
            #     cmd_parts.extend([option, value])
    return shlex.join(cmd_parts)


def find_latest_stdout_log(start_path):
    """Find the latest stdout.log for the highest rank in the latest attempt.

    Used to locate the training log to tail on the login node console.
    Directory structure created by torchrun::

        start_path/host_0_*/timestamp/run_id/attempt_N/rank/stdout.log

    Returns the path to stdout.log, or None if not found.
    """
    if not os.path.exists(start_path):
        return None

    folders_with_attempts = []
    for root, dirs, _ in os.walk(start_path):
        attempt_dirs = [d for d in dirs if d.startswith("attempt_")]
        if attempt_dirs:
            folders_with_attempts.append(root)

    if not folders_with_attempts:
        return None

    folders_with_attempts.sort(reverse=True)
    latest_folder = folders_with_attempts[0]

    attempt_dirs = [d for d in os.listdir(latest_folder) if d.startswith("attempt_")]
    if not attempt_dirs:
        return None

    attempt_dirs.sort(
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else -1,
        reverse=True,
    )
    latest_attempt = os.path.join(latest_folder, attempt_dirs[0])

    try:
        rank_dirs = os.listdir(latest_attempt)
        rank_dirs.sort(key=lambda x: int(x) if x.isdigit() else float("inf"), reverse=True)
        for rank_dir in rank_dirs:
            log_path = os.path.join(latest_attempt, rank_dir, "stdout.log")
            if os.path.exists(log_path):
                return log_path
    except OSError:
        pass

    return None


def tail_log_to_console(log_file_or_finder, stop_event, poll_interval=1.0):
    """Tail a log file and print new lines to the login node console.

    Runs in a daemon thread. Opens the log file in **read-only** mode, so
    Ctrl+C or any interruption on the login node will never corrupt or
    affect the log files written by the worker nodes.

    Args:
        log_file_or_finder: Either a file path string, or a callable returning
            a file path (for dynamic discovery, e.g. training stdout.log).
        stop_event: threading.Event to signal when to stop tailing.
        poll_interval: Seconds between polls for new content.
    """
    try:
        log_file = None
        while not stop_event.is_set():
            if callable(log_file_or_finder):
                log_file = log_file_or_finder()
            else:
                log_file = log_file_or_finder
            if log_file and os.path.exists(log_file):
                break
            stop_event.wait(poll_interval)

        if stop_event.is_set() or not log_file:
            return

        with open(log_file, "r") as f:
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    stop_event.wait(poll_interval)
    except Exception:
        # Silently exit on any error (file deleted, permission change, etc.).
        # The log files in log_dir are unaffected since we only read them.
        pass


def start_tail_log(log_file_or_finder):
    """Start tailing a log file in a daemon thread on the login node.

    Args:
        log_file_or_finder: Either a file path string, or a callable returning
            a file path (for dynamic discovery, e.g. training stdout.log).

    Returns:
        Tuple of (thread, stop_event). Call stop_event.set() to stop tailing.
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=tail_log_to_console,
        args=(log_file_or_finder, stop_event),
        daemon=True,
    )
    thread.start()
    return thread, stop_event


