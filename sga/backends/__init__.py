# Copyright (c) 2026, SGA framework contributors.
"""Sparse gradient-sync backends.

Backend classes live in this package; their declarations, registration, and
construction go through the single plugin registry in ``sga.plugins``
(``register_default_plugins`` / ``build_backend_plugin``) — there is no
separate registry here. "dense" is not a backend class: the config loader
treats the name as the SGA-off representation, and the engine's full-sync
steps delegate to the fork's dense allreduce directly.
"""

# The pipelined backends' single operator knobs — the resume fail-closed
# escape hatch and the comm-warmup bypass — share these env-var NAMES. Only
# the names live here (this package stays import-light): each backend keeps
# its own parsing rules and policy prose next to its local alias.
RESUME_OVERRIDE_ENV = "SGA_ALLOW_RESUME_PIPELINED"
COMM_WARMUP_ENV = "SGA_COMM_WARMUP"
