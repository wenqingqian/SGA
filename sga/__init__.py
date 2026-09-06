"""SGA: plugin-based selective gradient accumulation framework.

Subpackages:
- ``sga.config``: the sga args singleton + the flat ``--sga-*`` flag
  vocabulary and its validation.
- ``sga.plugins`` / ``sga.backends``: SGA algorithm plugins (score /
  selection / optimizer) and sparse-allreduce backends.
- ``sga.launcher``: multi-node ssh launcher (run/stop/query).
"""

__version__ = "0.1.0"
