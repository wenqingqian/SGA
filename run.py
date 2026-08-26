"""SGA CLI shim: forwards to the hydra entry in sga.launcher.run."""

from sga.launcher.run import main, resolve_config_path_in_argv

if __name__ == "__main__":
    resolve_config_path_in_argv()
    main()
