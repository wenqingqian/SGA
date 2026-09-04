# Copyright (c) 2026, SGA framework contributors.
"""``python -m sga.config`` — the sga ``--help`` equivalents.

- ``schema``: the full config surface from the catalog — every axis, every
  registered name, every declared param with type/default/choices/doc,
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

from sga.config.arguments import (
    ENGINE_SECTIONS,
    PLUGIN_PARAMS,
    SCORE_NAMES,
    TOP,
)
from sga.config.flatargs import flat_flags
from sga.config.loader import sga_section_to_config
from sga.plugins import effective_param_names, resolve_plugin_params


def _param_view(p):
    return {"name": p.name, "type": p.type.__name__,
            "required": p.required, "nullable": p.nullable,
            **({"default": p.default} if not p.required else {}),
            **({"choices": list(p.choices)} if p.choices else {}),
            "doc": p.doc}


def _schema() -> dict:
    flag_of = {}
    for f in flat_flags():
        section = f.section or "(top)"
        flag_of.setdefault(section, {})[f.key] = f.flag

    return {
        "score": {"names": list(SCORE_NAMES), "params": "none (registry data)"},
        "selection": {name: [_param_view(p) for p in PLUGIN_PARAMS["selection"][name]]
                      for name in sorted(PLUGIN_PARAMS["selection"])},
        "optimizer": {name: [_param_view(p) for p in PLUGIN_PARAMS["optimizer"][name]]
                      for name in sorted(PLUGIN_PARAMS["optimizer"])},
        "backend": {name: [_param_view(p) for p in PLUGIN_PARAMS["backend"][name]]
                    for name in sorted(PLUGIN_PARAMS["backend"])},
        "engine_sections": {
            section: [p.name for p in decl if not p.yaml_only]
                     + [f"{p.name} (yaml-only)" for p in decl if p.yaml_only]
            for section, decl in ENGINE_SECTIONS.items()
        } | {"top_level": [p.name for p in TOP]},
        "flat_flags": flag_of,
    }


def _expanded(path: str) -> dict:
    try:
        cfg = OmegaConf.load(path)
    except Exception as e:  # a raw OmegaConf traceback helps nobody
        raise SystemExit(f"sga.config: cannot load {path}: {e}") from None
    engine_config = sga_section_to_config(cfg)
    if engine_config is None:
        return {"sga": "disabled (dense)"}
    # Expand plugin defaults through the catalog (the canonical JSON carries
    # only explicit keys; this is the full picture an operator wants), then
    # keep only the params the configured method consumes — the union's
    # other defaults would be foreign keys (the dump's own output would be
    # rejected by the constructor's per-method check).
    for axis in ("selection", "optimizer", "backend"):
        name = engine_config[axis]["name"]
        params = {k: v for k, v in engine_config[axis].items() if k != "name"}
        resolved = resolve_plugin_params(axis, name, params)
        keep = effective_param_names(axis, name, resolved)
        engine_config[axis] = {"name": name,
                               **{k: v for k, v in resolved.items() if k in keep}}
    # Expand only the keys that HAVE a default: full_sync_interval and
    # start_iter_percent default to None = absent-by-meaning ("not
    # decoupled" / percent already resolved away), and the validator
    # rejects a full_sync_interval value without decouple=true.
    cycle_defaults = {p.name: p.default for p in ENGINE_SECTIONS["cycle"]
                      if not p.required and p.default is not None}
    engine_config.setdefault("cycle", {})
    for key, default in cycle_defaults.items():
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
