"""Dense backend: no selective gradient accumulation, plain megatron training."""

_BACKEND_NAME = "dense"


class DenseBackend:
    """Null backend — the framework's default when every SGA subfield is off.

    Kept as a class so later plans can hang capabilities (e.g. a `describe()`
    contract) off the registry entries uniformly.
    """

    name = _BACKEND_NAME
    sparse = False
