"""Gradient-sync backend registry.

A backend owns how gradients reach their allreduced/updated form (today: the
null dense path; later plans add torch pipelined sparse allreduce and the
nccl-sar kernel). Registration is explicit — `register_default_backends()` is
called by the config loader; plugin backends register themselves through the
same API when their plugin is activated.
"""

_BACKEND_REGISTRY: dict = {}


def register_backend(name: str, backend) -> None:
    if name in _BACKEND_REGISTRY:
        raise ValueError(f"Backend '{name}' is already registered")
    _BACKEND_REGISTRY[name] = backend


def get_backend(name: str):
    try:
        return _BACKEND_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown SGA backend '{name}'. Registered backends: {sorted(_BACKEND_REGISTRY)}"
        ) from None


def registered_backends() -> list:
    return sorted(_BACKEND_REGISTRY)


def register_default_backends() -> None:
    """Idempotent registration of the built-in backends."""
    from sga.backends.dense import DenseBackend

    if "dense" not in _BACKEND_REGISTRY:
        register_backend("dense", DenseBackend)
