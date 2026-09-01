#!/usr/bin/env python3
"""Unit tests for the backend registry (run: python tests/test_registry.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sga.backends import get_backend, register_backend, register_default_backends, registered_backends


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


if __name__ == "__main__":
    test_backends()
    print("test_registry: ALL PASS")
