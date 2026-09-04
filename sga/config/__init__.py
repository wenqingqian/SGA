# Copyright (c) 2026, SGA framework contributors.
"""Flat SGA config surface: declarations in arguments.py, validation here.

See sga/config/arguments.py (the developer-facing file) and
sga/config/validation.py (the validation core)."""

from sga.config.validation import (
    canonical_flat_json,
    validate_run_config,
    validate_sga_args,
    validate_sga_flat,
)

__all__ = [
    "canonical_flat_json",
    "validate_run_config",
    "validate_sga_args",
    "validate_sga_flat",
]
