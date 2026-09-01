"""SGA: plugin-based selective gradient accumulation framework.

Subpackages:
- ``sga.launcher``: multi-node ssh launcher (run/stop/query).
- ``sga.config``: single-layer yaml config loading and SGA-section validation.
- ``sga.plugins`` / ``sga.backends``: SGA algorithm plugins (score /
  selection / optimizer) and sparse-allreduce backends.
"""

__version__ = "0.1.0"
