# Copyright (c) 2026, SGA framework contributors.
"""The SGA args singleton — how every SGA consumer reads configuration.

Plugins, backends and the engine read config as plain attribute access
(``args.sga_optimizer_residual_mean``) off the namespace this module holds,
mirroring how megatron code reads ``megatron.training.get_args()``. The
namespace is a COPY of the megatron one, seeded once by
``copy_from_megatron_args`` inside ``build_engine`` (the single validation
point); validation writes derived values back onto the COPY, never onto the
megatron namespace. Tests seed it with ``set_args`` (usually
``SimpleNamespace(**dict(sga_defaults(), **overrides))``).

This module deliberately imports nothing from the rest of sga — it sits at
the bottom of the import graph, so every other module can reach get_args
without cycles.
"""

from types import SimpleNamespace

__all__ = [
    "NON_CONFIG_SGA_KEYS",
    "set_args",
    "get_args",
    "sga_config_attrs",
    "copy_from_megatron_args",
]

# sga-prefixed keys that are NOT config: the waist's resume-override switch
# and the retired v1 JSON-transport key (still present on old checkpoints).
# The single exclusion list — copy_from_megatron_args, the engine config
# print and the resume guard all read through it.
NON_CONFIG_SGA_KEYS = ("sga_allow_config_override", "sga_config")

_ARGS = None


def set_args(ns) -> None:
    """Install the SGA args namespace (tests and copy_from_megatron_args)."""
    global _ARGS
    _ARGS = ns


def get_args():
    """The SGA args namespace; loud when accessed before it was seeded."""
    if _ARGS is None:
        raise RuntimeError(
            "sga args not set — expected copy_from_megatron_args() (via "
            "build_engine) or set_args() (tests) to run first")
    return _ARGS


def sga_config_attrs(attrs_owner) -> dict:
    """One namespace's sga_* config attributes, RAW (no validation
    write-backs): the single value path for copy_from_megatron_args, the
    engine's config print and the resume guard comparison."""
    return {key: value for key, value in vars(attrs_owner).items()
            if key.startswith("sga_") and key not in NON_CONFIG_SGA_KEYS}


def copy_from_megatron_args(megatron_args) -> None:
    """Seed the SGA namespace from the parsed megatron namespace: every
    sga_* config attribute plus the few megatron keys SGA validation needs
    (train_iters anchors the percent spelling; pipeline parallelism gates
    the always-sync axis). Later needs mean one more line here."""
    ns = SimpleNamespace(**sga_config_attrs(megatron_args))
    ns.train_iters = getattr(megatron_args, "train_iters", None)
    ns.pipeline_model_parallel_size = getattr(
        megatron_args, "pipeline_model_parallel_size", 1)
    set_args(ns)
