#!/usr/bin/env python3
"""Audit SGA run-config megatron args against the Megatron-LM arg parser.

core_r0.18.0 builds most CLI args at runtime from dataclasses
(TransformerConfig / TrainingConfig / TokenizerConfig via ArgumentGroupFactory),
so text grep over arguments.py systematically misses args. This tool instead
instantiates the real parser and checks programmatically.

Modes:
  dump    <out.json>                 dump every accepted flag (flags/dest/default/choices)
  check   <yaml> [<out.json>]        flatten the yaml's train.* sections the way the
                                     launcher does, parse them with the real parser,
                                     report rejected/unknown flags, then dump the table
  diff-defaults <old.json> <new.json> compare two dumps flag-by-flag (old vs new fork)

Run inside the container conda env after `source ../SGA/env.sh` so
`megatron.training` resolves to the intended tree (or point PYTHONPATH at
another tree to dump it).
"""

import argparse
import json
import sys


def add_megtron_safe(add_fn):
    parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
    return add_fn(parser)


def dump_table(parser):
    table = {}
    for action in parser._actions:
        flags = [f for f in action.option_strings if f.startswith("--")]
        if not flags:
            continue
        entry = {
            "dest": action.dest,
            "default": repr(action.default),
            "type": getattr(action.type, "__name__", None) or str(action.type),
        }
        if getattr(action, "choices", None) is not None:
            entry["choices"] = [str(c) for c in action.choices]
        for flag in flags:
            table[flag] = entry
    return table


def flatten_yaml_train_sections(yaml_path):
    """Flatten train.{system,model,data} leaf keys exactly like the launcher:
    nested dicts lose their parent keys, underscores become dashes."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)["train"]
    flat = {}
    for section in ("system", "model", "data"):
        stack = [("", cfg[section])]
        while stack:
            prefix, cur = stack.pop()
            for k, v in cur.items():
                if isinstance(v, dict):
                    stack.append((k, v))
                else:
                    flat[k.replace("_", "-")] = v
    return flat


def build_cmdline(flat):
    args = []
    for k, v in flat.items():
        if v is None or v is False:
            continue  # mirrors flatten_dict_to_args: False/None silently dropped
        if v is True:
            args.append(f"--{k}")
        elif isinstance(v, list):
            args.append(f"--{k}")
            args.extend(str(x) for x in v)
        else:
            args.append(f"--{k}")
            args.append(str(v))
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "check", "diff-defaults"])
    ap.add_argument("paths", nargs="+")
    ns = ap.parse_args()

    if ns.mode == "diff-defaults":
        with open(ns.paths[0]) as f:
            old = json.load(f)
        with open(ns.paths[1]) as f:
            new = json.load(f)
        for flag in sorted(set(old) | set(new)):
            o = old.get(flag)
            n = new.get(flag)
            if o is None:
                print(f"ADDED      {flag}  default={n['default']}")
            elif n is None:
                print(f"REMOVED    {flag}  (old default={o['default']})")
            elif o["default"] != n["default"]:
                print(f"DEFAULT-DIFF {flag}: {o['default']} -> {n['default']}")
        return 0

    parser = add_megtron_safe(_import_add_args())

    if ns.mode == "dump":
        table = dump_table(parser)
        with open(ns.paths[0], "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
        print(f"dumped {len(table)} flags to {ns.paths[0]}")
        return 0

    # check mode
    yaml_path, out_path = ns.paths[0], (ns.paths[1:] + [None])[0]
    flat = flatten_yaml_train_sections(yaml_path)
    table = dump_table(parser)

    unknown = [k for k in flat if f"--{k}" not in table]
    cmdline = build_cmdline(flat)
    try:
        ns2, unrecognized = parser.parse_known_args(cmdline)
        parse_errors = []
    except SystemExit:
        print("PARSE ERROR: parser rejected the command line (see usage above)")
        parse_errors = ["parse_args exited"]
        unrecognized = []
    if unknown:
        print("UNKNOWN FLAGS (not in parser table):")
        for k in sorted(unknown):
            print(f"  --{k}  (yaml value: {flat[k]!r})")
    if unrecognized:
        print("UNRECOGNIZED ON CMDLINE:", unrecognized)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
        print(f"dumped {len(table)} flags to {out_path}")
    if unknown or unrecognized or parse_errors:
        return 1
    print(f"OK: all {len(flat)} yaml keys accepted by the parser")
    return 0


def _import_add_args():
    from megatron.training.arguments import add_megatron_arguments

    return add_megatron_arguments


if __name__ == "__main__":
    sys.exit(main())
