# Copyright (c) 2026, SGA framework contributors.
"""Sparse gradient-sync backends.

Backend classes live in this package; their declarations, registration, and
construction go through the single plugin registry in ``sga.plugins``
(``register_default_plugins`` / ``build_backend_plugin``) — there is no
separate registry here. "dense" is not a backend class: the config loader
treats the name as the SGA-off representation, and the engine's full-sync
steps delegate to the fork's dense allreduce directly. The nccl-sar kernel
backend joins this package as a future sibling.
"""
