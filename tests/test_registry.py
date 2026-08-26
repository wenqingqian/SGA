#!/usr/bin/env python3
"""Unit tests for the model/backend registries (run: python tests/test_registry.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.core.enums import ModelType

from sga.backends import get_backend, register_backend, register_default_backends, registered_backends
from sga.models import get_model, register_default_models, register_model, registered_models


def test_backends():
    register_default_backends()
    register_default_backends()  # idempotent
    assert registered_backends() == ["dense"]
    assert get_backend("dense").sparse is False
    try:
        get_backend("nope")
    except ValueError as e:
        assert "dense" in str(e)
    else:
        raise SystemExit("unknown backend not rejected")
    print("ok: backend registry (explicit, idempotent, dense-only)")


def test_models():
    register_default_models()
    entry = get_model("llama-1b")
    assert entry["model_type"] == ModelType.encoder_or_decoder
    assert callable(entry["builder"])
    try:
        get_model("gpt-99b")
    except ValueError as e:
        assert "llama-1b" in str(e)
    else:
        raise SystemExit("unknown model not rejected")
    print("ok: model registry (llama-1b default entry)")


if __name__ == "__main__":
    test_backends()
    test_models()
    print("test_registry: ALL PASS")
