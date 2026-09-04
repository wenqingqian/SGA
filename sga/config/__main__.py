# Copyright (c) 2026, SGA framework contributors.
"""``python -m sga.config`` — the sga ``--help`` equivalents.

- ``schema``: the full config surface, introspected from the declarations
  (sga/config/arguments.py): every flag with its dest/default/choices/
  help. (Which params belong to which plugin is the plugins' own
  business — there is no central attribution table to dump.)
- ``resolve <run.yaml>``: load a run config, validate the ``sga:``
  section, and dump the RESOLVED canonical flat dict — the exact form
  the engine prints at build time (percent resolved away, defaults
  filled, floats canonicalized).

Usage: python -m sga.config schema | resolve examples/llama3/train_1b_adamw.yaml
"""

import json
import sys

from sga.config.validation import (
    _param_kind,
    sga_actions,
    validate_run_config,
)


def _schema() -> dict:
    flags = {}
    for dest, action in sorted(sga_actions().items()):
        kind = _param_kind(dest)
        if dest == "sga_trace_branch_points":
            # lists are accepted on the yaml route (_coerce's carve-out)
            kind = "str|list"
        flags[dest] = {
            "flag": "--" + dest.replace("_", "-"),
            "default": action.default,
            "type": kind,
            **({"choices": list(action.choices)} if action.choices else {}),
            "help": action.help,
        }
    return {"flags": flags}


def _resolved(path: str) -> dict:
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
    except Exception as e:  # a raw OmegaConf traceback helps nobody
        raise SystemExit(f"sga.config: cannot load {path}: {e}") from None
    canonical = validate_run_config(cfg)
    if canonical is None:
        return {"sga": "disabled (dense)"}
    return canonical


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["schema"]:
        print(json.dumps(_schema(), indent=2, sort_keys=True))
        return 0
    if argv[:1] == ["resolve"] and len(argv) == 2:
        print(json.dumps(_resolved(argv[1]), indent=2, sort_keys=True))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
