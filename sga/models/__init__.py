"""Model registry: one file per model under sga/models/.

A registry entry provides the megatron-facing build chain for a model family:
``builder`` (the gpt_builder-style callable) plus its ModelType. The training
entry (sga.train_gpt) selects the entry via the SGA_MODEL environment variable
(default: the day-one llama-1b dense model) and never hardcodes a model.

Registration is explicit — train_gpt calls register_default_models(); new
models add one file here and register themselves inside that function.
"""

from megatron.core.enums import ModelType

_MODEL_REGISTRY: dict = {}


def register_model(name: str, builder, model_type: ModelType) -> None:
    if name in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered")
    _MODEL_REGISTRY[name] = {"builder": builder, "model_type": model_type}


def get_model(name: str):
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}' (SGA_MODEL). Registered models: {sorted(_MODEL_REGISTRY)}"
        ) from None


def registered_models() -> list:
    return sorted(_MODEL_REGISTRY)


def register_default_models() -> None:
    """Idempotent registration of the built-in models."""
    from sga.models import llama_1b

    if llama_1b.MODEL_NAME not in _MODEL_REGISTRY:
        register_model(llama_1b.MODEL_NAME, llama_1b.builder, llama_1b.MODEL_TYPE)
