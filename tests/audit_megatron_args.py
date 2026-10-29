#!/usr/bin/env python3
"""Audit SGA run-config megatron args against the Megatron-LM arg parser.

core_r0.18.0 builds most CLI args at runtime from dataclasses
(TransformerConfig / TrainingConfig / TokenizerConfig via ArgumentGroupFactory),
so text grep over arguments.py systematically misses args. This tool instead
instantiates the real parser and checks programmatically.

Two flatten views are reported:

  PRIMARY   the REAL launch pipeline: the yaml is wrapped in a minimal
            DictConfig and pushed through sga.launcher.backend_megatron
            ._update_config_train + ._get_args_megatron (the same functions
            MegatronBackend._prepare calls), so launcher-injected keys
            (checkpoint/log/tensorboard/wandb dirs, ...) and the launcher's
            ignore_keys match production exactly. Problems here decide the
            exit code.
  SECONDARY the raw yaml's train.{system,model,data} leaves only, flattened
            directly (no mutation, no ignore_keys). Informational: it shows
            what the authored file maps to before the launcher touches it,
            but it is not what training actually receives.

Parser construction follows the training entry (sga/train_gpt.py), i.e. the
same path as megatron.training.arguments.parse_and_validate_args ->
parse_args: an ArgumentParser(allow_abbrev=False), add_megatron_arguments(),
then upstream pretrain_gpt's optional ModelOpt extra_args_provider when
megatron.post_training is importable. parse_and_validate_args exposes no
public parser getter, so that three-line construction is mirrored verbatim
here (see build_megatron_parser) and pinned by tests/test_launcher_shape.py.

Modes:
  dump    <out.json>                 dump every accepted flag (flags/dest/default/choices)
  check   <yaml> [<out.json>]        run the views above through the real
                                     parser, report rejected/unknown flags,
                                     then optionally dump the table
  diff-defaults <old.json> <new.json> compare two dumps flag-by-flag (old vs new fork)

Run inside the container conda env after `source env.sh` from the SGA
repo root so `megatron.training` resolves to the intended tree (or point PYTHONPATH at
another tree to dump it).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_modelopt() -> bool:
    """True iff the ModelOpt extra_args_provider is available, mirroring the
    import guard used by pretrain_gpt.__main__ / sga/train_gpt.py."""
    try:
        import megatron.post_training.arguments  # noqa: F401
    except ImportError:
        return False
    return True


def build_megatron_parser(include_modelopt: bool) -> argparse.ArgumentParser:
    """Build the parser exactly as the training entry does.

    Mirrors megatron.training.arguments.parse_args() (called internally by
    parse_and_validate_args(), which SGA's train_gpt.py invokes with the
    optional ModelOpt extra provider): no public parser-builder exists in
    megatron.training, so these three lines replicate its construction.
    """
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
    parser = add_megatron_arguments(parser)
    if include_modelopt:
        from megatron.post_training.arguments import add_modelopt_args

        parser = add_modelopt_args(parser)
    return parser


def load_training_entry_parser():
    """The parser exactly as the SGA training entry would build it now."""
    return build_megatron_parser(detect_modelopt())


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
    """SECONDARY view: flatten raw train.{system,model,data} leaf keys without
    any launcher mutation (nested dicts lose their parent keys, underscores
    become dashes)."""
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


def launcher_cmdline_from_yaml(yaml_path):
    """PRIMARY view: run the actual launch pipeline over the yaml.

    Loads the yaml, fills in the minimal experiment.* scaffolding the hydra
    entry normally provides (exp_dir is redirected to a throwaway temp dir so
    the audit never creates directories at the user's configured location --
    directory *values* do not affect parser acceptance, only flag names do),
    then calls the real _update_config_train + _get_args_megatron pair.

    The loaded DictConfig is used as-is (no to_container/create round-trip:
    that would turn escaped ``\\${VAR}`` sequences in cmds.* into live
    interpolations). Missing scaffolding is filled via .get() probes so keys
    are only added when genuinely absent.

    Returns the resulting megatron arg list.
    """
    from omegaconf import OmegaConf

    from sga.launcher.backend_megatron import _get_args_megatron, _update_config_train

    cfg = OmegaConf.load(yaml_path)

    experiment = cfg.get("experiment", None)
    if experiment is None or not OmegaConf.is_dict(experiment):
        cfg.experiment = {}
        experiment = cfg.experiment

    if experiment.get("exp_name", None) is None:
        experiment.exp_name = os.path.splitext(os.path.basename(yaml_path))[0]

    exp_dir = tempfile.mkdtemp(prefix="sga-audit-exp-")
    experiment.exp_dir = exp_dir

    task = experiment.get("task", None)
    if task is None or not OmegaConf.is_dict(task):
        experiment.task = {}
        task = experiment.task
    if task.get("type", None) is None:
        task.type = "train"
    if task.get("backend", None) is None:
        task.backend = "megatron"
    if task.get("entrypoint", None) is None:
        task.entrypoint = "./sga/train_gpt.py"

    runner = experiment.get("runner", None)
    if runner is None or not OmegaConf.is_dict(runner):
        experiment.runner = {}

    try:
        _update_config_train(cfg)
        args_list = _get_args_megatron(cfg)
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)
    return args_list


def args_list_to_flat(args_list):
    """Inverse view of flatten_dict_to_args output: flag name -> value(s).
    Bare flags (bool True) map to True; False/None flags never appear."""
    flat = {}
    i = 0
    while i < len(args_list):
        token = args_list[i]
        assert token.startswith("--"), f"non-flag token in launcher args: {token!r}"
        key = token[2:]
        values = []
        j = i + 1
        while j < len(args_list) and not args_list[j].startswith("--"):
            values.append(args_list[j])
            j += 1
        flat[key] = values if values else True
        i = j
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


def report_section(name, flat, table, parser, fail_hard=True):
    """Check one flattened view against the parser.

    fail_hard=False makes the section informational: problems are reported
    but not counted (the raw-yaml view lacks launcher mutation/ignore_keys,
    so e.g. an authored logging.log_dir is expected to look 'unknown' there).
    Returns the problem count.
    """
    unknown = [k for k in flat if f"--{k}" not in table]
    cmdline = build_cmdline(flat)
    try:
        _parsed, unrecognized = parser.parse_known_args(cmdline)
        parse_errors = []
    except SystemExit:
        print("PARSE ERROR: parser rejected the command line (see usage above)")
        parse_errors = ["parse_args exited"]
        unrecognized = []
    if unknown:
        print(("UNKNOWN FLAGS (not in parser table):" if fail_hard else "(info) keys outside the parser table:"))
        for k in sorted(unknown):
            print(f"  --{k}  (value: {flat[k]!r})")
    if unrecognized:
        print("UNRECOGNIZED ON CMDLINE:", unrecognized)
    n_unknown_or_unrecognized = len(unknown) + len(unrecognized) + len(parse_errors)
    if n_unknown_or_unrecognized == 0:
        print(f"ok: {len(flat)} keys checked ({name})")
    elif fail_hard:
        print(f"FAIL: {n_unknown_or_unrecognized} problem(s) ({name})")
    else:
        print(f"(info) {n_unknown_or_unrecognized} raw-yaml key(s) differ from the launcher view")
    return n_unknown_or_unrecognized if fail_hard else 0


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

    # Same parser-construction path as the training entry (ModelOpt included
    # exactly when the training entry would include it).
    include_modelopt = detect_modelopt()
    print(
        "parser: training-entry path "
        f"(extra_args_provider={'add_modelopt_args' if include_modelopt else 'None'})"
    )
    parser = build_megatron_parser(include_modelopt)

    if ns.mode == "dump":
        table = dump_table(parser)
        with open(ns.paths[0], "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
        print(f"dumped {len(table)} flags to {ns.paths[0]}")
        return 0

    # check mode
    yaml_path, out_path = ns.paths[0], (ns.paths[1:] + [None])[0]
    table = dump_table(parser)

    problems = 0

    print("== PRIMARY: real launch pipeline (_update_config_train -> _get_args_megatron) ==")
    primary_flat = args_list_to_flat(launcher_cmdline_from_yaml(yaml_path))
    problems += report_section("launcher pipeline", primary_flat, table, parser)

    print("== SECONDARY: raw-yaml train.* leaves (informational) ==")
    secondary_flat = flatten_yaml_train_sections(yaml_path)
    extra_raw_only = sorted(set(secondary_flat) - set(primary_flat))
    report_section("raw yaml", secondary_flat, table, parser, fail_hard=False)
    if extra_raw_only:
        print("(info) raw-yaml-only keys absent from launcher args:")
        for k in extra_raw_only:
            print(f"  --{k}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
        print(f"dumped {len(table)} flags to {out_path}")
    if problems:
        return 1
    print(
        f"OK: {len(primary_flat)} launcher-pipeline keys and {len(secondary_flat)}"
        " raw-yaml keys all accepted by the parser"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
