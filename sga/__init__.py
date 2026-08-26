"""SGA: plugin-based selective gradient accumulation framework.

Subpackages:
- ``sga.launcher``: multi-node ssh launcher (run/stop/query).
- ``sga.config``: single-layer yaml config loading and SGA-section validation.
- ``sga.models``: model registry (one file per model).
- ``sga.plugins`` / ``sga.backends``: reserved for SGA algorithm plugins and
  sparse-allreduce backends (populated by later plans).
"""

__version__ = "0.1.0"
