import multiprocessing
import os
import shlex
import time
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from sga.launcher.launcher_base import LauncherBase
from sga.launcher.utils import (
    JobStatus,
    find_latest_stdout_log,
    get_free_port,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    resolve_path,
    run_local_command,
    run_scp_command,
    run_ssh_command,
    start_tail_log,
    update_cmd_with_node_specific_config,
    update_nodes_envs,
)

_MAX_CPU_COUNT = multiprocessing.cpu_count()


def _get_runner_cmd_train(
    host, master_addr, master_port, nnodes, node_rank, nproc_per_node, config: DictConfig
):
    runner_config = config.experiment.runner
    logging_config = config.train.system.logging

    if runner_config.get("per_node_task", False):
        nnodes = 1
        node_rank = 0
        master_addr = "localhost"

    rdzv_id = runner_config.get("rdzv_id", "default")
    log_dir = runner_config.get("log_dir", logging_config.details_dir)
    log_dir = resolve_path(log_dir, "runner.log_dir")
    no_shared_fs = runner_config.get("no_shared_fs", False)
    if no_shared_fs:
        log_dir = os.path.join(log_dir, "host")
    else:
        log_dir = os.path.join(log_dir, f"host_{node_rank}_{host}")
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    rdzv_backend = runner_config.get("rdzv_backend", "c10d")
    rdzv_endpoint = runner_config.get("rdzv_endpoint", f"{master_addr}:{master_port}")
    tee = runner_config.get("tee", "3")

    runner_args = OmegaConf.to_container(runner_config, resolve=True)
    if "type" in runner_args:
        del runner_args["type"]
    if "backend" in runner_args:
        del runner_args["backend"]
    if "per_node_task" in runner_args:
        del runner_args["per_node_task"]
    if "hostfile" in runner_args:
        del runner_args["hostfile"]
    if "ssh_port" in runner_args:
        del runner_args["ssh_port"]
    if "master_addr" in runner_args:
        del runner_args["master_addr"]
    if "master_port" in runner_args:
        del runner_args["master_port"]
    if "enable_monitoring" in runner_args:
        del runner_args["enable_monitoring"]
    if "enable_gpu_health_check" in runner_args:
        del runner_args["enable_gpu_health_check"]
    # Profiling options are consumed by launcher; torchrun doesn't accept them.
    if "nsys_bin_path" in runner_args:
        del runner_args["nsys_bin_path"]
    if "nsys_rep_file_path" in runner_args:
        del runner_args["nsys_rep_file_path"]
    if "deploy" in runner_args:
        del runner_args["deploy"]
    if "enable_perf_monitor" in runner_args:
        del runner_args["enable_perf_monitor"]
    if "perf_log_interval" in runner_args:
        del runner_args["perf_log_interval"]
    if "perf_log_dir" in runner_args:
        del runner_args["perf_log_dir"]
    if "perf_console_output" in runner_args:
        del runner_args["perf_console_output"]
    if "perf_log_format" in runner_args:
        del runner_args["perf_log_format"]
    if "perf_memory_tracking" in runner_args:
        del runner_args["perf_memory_tracking"]
    if "perf_breakdown" in runner_args:
        del runner_args["perf_breakdown"]
    if "perf_max_log_files" in runner_args:
        del runner_args["perf_max_log_files"]
    if "perf_model_type" in runner_args:
        del runner_args["perf_model_type"]
    runner_args["rdzv_id"] = rdzv_id
    runner_args["nnodes"] = nnodes
    runner_args["node_rank"] = node_rank
    runner_args["nproc_per_node"] = nproc_per_node
    runner_args["rdzv_backend"] = rdzv_backend
    runner_args["rdzv_endpoint"] = rdzv_endpoint

    runner_args["log_dir"] = log_dir
    runner_args["tee"] = tee

    runner_cmd = ["torchrun"]
    for key, value in runner_args.items():
        if isinstance(value, bool):
            if value:
                runner_cmd.append(f"--{key}")
        else:
            runner_cmd.append(f"--{key}")
            runner_cmd.append(f"{value}")
    return runner_cmd


def run_node(
    func,
    node_rank,
    host,
    resource_info,
    user_envs,
    runner_config,
    nnodes,
    available_ip,
    available_port,
    background,
    dryrun,
    cur_envs,
):
    cur_envs = update_nodes_envs(user_envs, host, resource_info)
    # Get the number of visible devices from the environment variable, e.g. CUDA_VISIBLE_DEVICES, MLU_VISIBLE_DEVICES
    visible_devices = next((v for k, v in cur_envs.items() if k.endswith("_VISIBLE_DEVICES")), None)
    num_visible_devices = None
    if visible_devices is not None and isinstance(visible_devices, str):
        visible_devices = visible_devices.split(",")
        num_visible_devices = len(visible_devices)
    nproc_from_hostfile = resource_info["slots"]
    nproc_from_args = runner_config.get("nproc_per_node", None)
    nproc_per_node = get_nproc_per_node(nproc_from_hostfile, nproc_from_args, num_visible_devices)
    master_addr = runner_config.get("master_addr", available_ip)
    master_port = runner_config.get("master_port", available_port)
    func(
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        device_type=resource_info["type"],
        background=background,
        dryrun=dryrun,
        cur_envs=cur_envs,
    )


class SshLauncher(LauncherBase):
    def __init__(self, config, backend):
        self.config = config
        hostfile = self.config.experiment.runner.get("hostfile", None)
        self.resources = parse_hostfile(hostfile) if hostfile else None
        self.task_type = getattr(self.config.experiment.task, "type", None)
        self.backend = backend
        self.user_args = self.backend.user_args
        self.user_envs = self.backend.user_envs
        self.user_script = self.backend.user_script

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        device_type=None,
        background=True,
        dryrun=False,
        cur_envs=None,
    ):
        export_cmd = []
        if cur_envs:
            for k, v in cur_envs.items():
                if k != "nodes_envs":
                    export_cmd += [f"{k}={v}"]
        runner_cmd = _get_runner_cmd_train(
            host, master_addr, master_port, nnodes, node_rank, nproc_per_node, self.config
        )

        # Optional Nsight Systems profiling (nsys)
        runner_cfg = self.config.experiment.runner
        nsys_bin_path = runner_cfg.get("nsys_bin_path", None)
        nsys_rep_file_path = runner_cfg.get("nsys_rep_file_path", None)
        if nsys_bin_path and nsys_rep_file_path:
            # Allow passing either a full path to `nsys` or a directory containing it.
            nsys_exe = (
                nsys_bin_path
                if os.path.basename(str(nsys_bin_path)) == "nsys"
                else os.path.join(str(nsys_bin_path), "nsys")
            )
            # If a directory is provided, write a per-host report file under it.
            # Keep `$HOSTNAME` unexpanded so the remote shell expands it.
            rep_path = str(nsys_rep_file_path)
            if rep_path.endswith("/") or (
                ".nsys-rep" not in rep_path and "$HOSTNAME" not in rep_path
            ):
                rep_path = rep_path.rstrip("/") + "/$HOSTNAME.nsys-rep"
            nsys_cmd = [
                nsys_exe,
                "profile",
                "-s",
                "none",
                "-t",
                "nvtx,cuda,osrt",
                "-o",
                rep_path,
                "--force-overwrite",
                "true",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
            ]
            runner_cmd = nsys_cmd + runner_cmd

        # update hetero-current-device-type according to the device_type in hostfile
        if device_type is not None:
            if "--hetero-current-device-type" in self.user_args:
                idx = self.user_args.index("--hetero-current-device-type")
                self.user_args[idx + 1] = device_type
            else:
                self.user_args += ["--hetero-current-device-type", device_type]
        cmd = shlex.join(export_cmd + runner_cmd + [self.user_script] + self.user_args)
        node_specific_config = {}
        if device_type is not None:
            node_specific_config = (
                self.backend.device_type_specific.get(device_type, {})
                if self.backend.device_type_specific
                else {}
            )
        node_specific_config.update(
            self.backend.node_specific.get(host, {}) if self.backend.node_specific else {}
        )
        cmd = update_cmd_with_node_specific_config(cmd, node_specific_config)

        logging_config = self.config.train.system.logging
        host_run_script_file = self.backend.generate_run_script(
            self.config,
            host,
            node_rank,
            cmd,
            background=background,
            pkg_dir=node_specific_config.get("build_dir", None),
        )

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, dryrun)

            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_run_script_file, logging_config.scripts_dir, ssh_port, dryrun
                )

            # For foreground + node 0, stream stdout through SSH to the
            # login node console so logs are visible without shared FS.
            run_ssh_command(
                host,
                f"bash {host_run_script_file}",
                ssh_port,
                dryrun,
                stream_output=(not background and node_rank == 0),
            )
        else:
            run_local_command(
                f"bash {host_run_script_file}",
                dryrun,
                stream_output=(not background and node_rank == 0),
            )

    def run(
        self,
        background=True,
        dryrun=False,
        monitor=False,
        interval=10,
    ):
        num_visible_devices = None
        # visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        visible_devices = next(
            (v for k, v in self.user_envs.items() if k.endswith("_VISIBLE_DEVICES")), None
        )
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

        # In background mode, tail node 0's log file on the login node console.
        # In foreground mode, tee already streams stdout directly.
        _tail_stop = None
        if not dryrun and background:
            details_dir = self.config.train.system.logging.details_dir
            _, _tail_stop = start_tail_log(lambda: find_latest_stdout_log(details_dir))

        try:
            if self.resources is not None:
                nnodes_from_hostfile = len(self.resources.keys())
                nnodes_from_args = runner_config.get("nnodes", None)
                nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)
                available_ip = next(iter(self.resources.keys()))
                available_port = get_free_port()
                num_processes = min(nnodes, _MAX_CPU_COUNT)
                with multiprocessing.Pool(processes=num_processes) as pool:
                    tasks = []
                    for node_rank, (host, resource_info) in enumerate(self.resources.items()):
                        if node_rank >= nnodes:
                            break
                        args = (
                            self._run_each,
                            node_rank,
                            host,
                            resource_info,
                            self.user_envs,
                            runner_config,
                            nnodes,
                            available_ip,
                            available_port,
                            background,
                            dryrun,
                            None,
                        )
                        tasks.append(args)
                    pool.starmap(run_node, tasks)
            else:
                nproc_from_args = runner_config.get("nproc_per_node", None)
                nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
                available_addr = runner_config.get("master_addr", "localhost")
                available_port = runner_config.get("master_port", get_free_port())
                self._run_each(
                    "localhost",
                    available_addr,
                    available_port,
                    1,
                    0,
                    nproc_per_node,
                    background=background,
                    dryrun=dryrun,
                    cur_envs=self.user_envs,
                )

            if monitor:
                # sleep to wait task already started
                time.sleep(interval)
                try:
                    while True:
                        status = self._query_status()
                        logger.info(f"Job Status: {status.name}")
                        if status == JobStatus.COMPLETED_OR_IDLE:
                            break
                        time.sleep(interval)
                    logger.info("Job Ended.")
                except Exception as e:
                    logger.info(e)
        finally:
            if _tail_stop:
                _tail_stop.set()

        return None

    def _stop_each(self, host, node_rank):
        host_stop_script_file = self.backend.generate_stop_script(host, node_rank)
        logging_config = self.config.train.system.logging

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port)
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_stop_script_file, logging_config.scripts_dir, ssh_port
                )
            run_ssh_command(host, f"bash {host_stop_script_file}", ssh_port)
        else:
            run_local_command(f"bash {host_stop_script_file}")

    def stop(self):
        if self.resources is None:
            self._stop_each("localhost", 0)
            return

        nnodes = get_nnodes(len(self.resources), self.config.experiment.runner.get("nnodes", None))

        num_processes = min(nnodes, _MAX_CPU_COUNT)
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = []
            for node_rank, (host, _) in enumerate(self.resources.items()):
                if node_rank >= nnodes:
                    break
                args = (host, node_rank)
                tasks.append(args)
            pool.starmap(self._stop_each, tasks)

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.train.system.logging

        host_query_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + shlex.quote(host_pid_file) + " ]; then\n")
            f.write("    pid=$(cat " + shlex.quote(host_pid_file) + ")\n")
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("else\n")
            # Fallback when the pid file is missing: pick the first torchrun process.
            f.write(
                "    pid=$(ps aux | grep 'torchrun' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file


    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        logging_config = self.config.train.system.logging
        result = ""
        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            run_ssh_command(
                host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, query=True
            )
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_query_script_file, logging_config.scripts_dir, ssh_port
                )
            try:
                result = run_ssh_command(
                    host, f"bash {host_query_script_file}", ssh_port, query=True
                )
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        else:
            try:
                result = run_local_command(f"bash {host_query_script_file}", query=True)
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result


    def _query_status(self):
        "Query Job status."
        results = []
        if self.resources is None:
            result = self._query_each("localhost", 0)
            results.append(result)
        else:
            host_list = list(self.resources.keys())
            for host, _ in self.resources.items():
                node_rank = host_list.index(host)
                result = self._query_each(host, node_rank)
                results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status


    def query_once(self):
        """
        Query job status once (non-blocking).
        There are three kinds of status for a Job:
            RUNNING: The job is running.
            COMPLETED_OR_IDLE: The job is completed or idle.
            TRANSITIONAL: The job is starting or stopping.

        Returns:
            JobStatus: Current job status
        """
        return self._query_status()

    def query(self, interval=10, timeout=None):
        """
        Query job status and log with optional timeout (blocking).

        Args:
            interval (int, optional): The interval of querying job status. Default: 10.
            timeout (float, optional): The timeout of query job status, if None, the query will keep indefinitely. Default: None.

        Returns:
            None

        Warning:
                    This method is blocking and should be used with caution.
                                Consider using query_once() for a non-blocking alternative.
        """
        logger.warning(
            "Using blocking query method. Consider using query_once()"
        )

        if timeout is None:
            logger.warning("Entering indefinite blocking query loop. Press Ctrl+C to exit.")
            try:
                while True:
                    job_status = self._query_status()
                    logger.info(f"Job status: {job_status.name}")
                    time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Query interrupted by user")
        else:
            start_time = time.time()
            cur_time = time.time()
            while cur_time - start_time < timeout:
                job_status = self._query_status()
                logger.info(f"Job status: {job_status.name}")
                time.sleep(interval)
                cur_time = time.time()
            logger.info(f"Query timeout reached ({timeout}s)")


