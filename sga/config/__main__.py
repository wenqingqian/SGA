# Copyright (c) 2026, SGA framework contributors.
"""``python -m sga.config`` — the sga ``--help`` equivalents.

- ``schema``: the full config surface from the declarations — every axis,
  every registered name, every declared param with type/default/choices/doc,
  plus the engine sections and the flat-flag spelling of each.
- ``resolve <run.yaml>``: load a run config, validate the ``sga:`` section,
  and dump the RESOLVED engine config with defaults expanded (the complete
  form of the canonical ``args.sga_config`` JSON: what the engine would
  actually run).

Usage: python -m sga.config schema | resolve examples/llama3/train_1b_adamw.yaml
"""

import json
import sys

from omegaconf import OmegaConf

from sga.config.arguments import _ENGINE_SECTIONS, _ENGINE_TOP, flat_flags
from sga.config.loader import sga_section_to_config
from sga.plugins import (
    register_default_plugins,
    registered_names,
    resolve_plugin_params,
)


def _schema() -> dict:
    from sga.plugins import (
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_selection_decl,
    )

    register_default_plugins()  # populate before listing (idempotent)
    names = registered_names()
    decls = {}
    getter = {"selection": get_selection_decl,
              "optimizer": get_optimizer_decl,
              "backend": get_backend_plugin_decl}
    for axis in ("selection", "optimizer", "backend"):
        decls[axis] = {
            plugin_name: [
                {"name": p.name, "type": p.type.__name__,
                 "required": p.required, "nullable": p.nullable,
                 **({"default": p.default} if not p.required else {}),
                 **({"choices": list(p.choices)} if p.choices else {}),
                 "doc": p.doc}
                for p in getter[axis](plugin_name).params
            ]
            for plugin_name in names[axis]
        }

    flag_of = {}
    for f in flat_flags():
        section = f.section or "(top)"
        flag_of.setdefault(section, {})[f.key] = f.flag

    return {
        "score": {"names": names["score"], "params": "none (registry data)"},
        "selection": decls["selection"],
        "optimizer": decls["optimizer"],
        "backend": decls["backend"],
        "engine_sections": {
            "cycle": [k for k, _t in _ENGINE_SECTIONS["cycle"]] + ["start_iter_percent (yaml-only)"],
            "checks": [k for k, _t in _ENGINE_SECTIONS["checks"]],
            "trace": [k for k, _t in _ENGINE_SECTIONS["trace"]],
            "top_level": [k for k, _t in _ENGINE_TOP],
        },
        "flat_flags": flag_of,
    }


def _expanded(path: str) -> dict:
    from sga.plugins import (
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_selection_decl,
    )

    try:
        cfg = OmegaConf.load(path)
    except Exception as e:  # a raw OmegaConf traceback helps nobody
        raise SystemExit(f"sga.config: cannot load {path}: {e}") from None
    engine_config = sga_section_to_config(cfg)
    if engine_config is None:
        return {"sga": "disabled (dense)"}
    # Expand plugin defaults through the same declarations the constructors
    # read (the canonical JSON carries only explicit keys; this is the full
    # picture an operator wants to inspect).
    getter = {"selection": get_selection_decl,
              "optimizer": get_optimizer_decl,
              "backend": get_backend_plugin_decl}
    for axis in ("selection", "optimizer", "backend"):
        name = engine_config[axis]["name"]
        decl = getter[axis](name)
        params = {k: v for k, v in engine_config[axis].items() if k != "name"}
        resolved = resolve_plugin_params(axis, name, decl, params)
        if decl.effective is not None:
            # vocabulary-aware view (e.g. topk expands only the configured
            # method's params — the union defaults would not reconstruct)
            resolved = decl.effective(resolved)
        engine_config[axis] = {"name": name, **resolved}
    engine_config.setdefault("cycle", {})
    # full_sync_interval has no default: absent means "not decoupled" (the
    # validator rejects a value without decouple=true).
    for key, default in (("start_iter", 0), ("period", 1),
                         ("decouple_full_sync_interval_and_period", False)):
        engine_config["cycle"].setdefault(key, default)
    return engine_config


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["schema"]:
        print(json.dumps(_schema(), indent=2, sort_keys=True))
        return 0
    if argv[:1] == ["resolve"] and len(argv) == 2:
        print(json.dumps(_expanded(argv[1]), indent=2, sort_keys=True))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
