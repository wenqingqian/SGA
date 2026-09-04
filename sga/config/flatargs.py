# Copyright (c) 2026, SGA framework contributors.
"""The flat ``--sga-*`` flag pipeline — driven by the catalog (arguments.py).

Every config path meets HERE:

    nested yaml ``sga:`` section --(launcher translation)--> flat flags
    hand-written flat flags --------------------------------------> argparse
                                                                     |
    ``--sga-config`` canonical JSON (kept; fork-side flag) --------> args
                                                                     |
                                              assemble_sga_args(args)
                                                                     |
                                        ``args.sga_config`` canonical JSON

The vocabulary is DATA-DRIVEN from the catalog declarations (the engine
sections and every plugin axis' params — sga/config/arguments.py, the
single source of truth). Adding a catalog row automatically gets its
``--sga-<axis>-<param>`` flag; ``yaml_only`` rows get none.

Inertness invariant: every flag defaults to None; no flag and no JSON means
SGA-off, byte-identical to stock (the waist still keys off
``args.sga_config`` alone, and ``--sga-config`` remains the fork's only
interface — the fork is untouched). Hand-writing BOTH the JSON and flat
flags fails loud.
"""

import argparse
from dataclasses import dataclass

from sga.config.arguments import (
    ENGINE_SECTIONS,
    PLUGIN_PARAMS,
    SCORE_NAMES,
    TOP,
)

_AXES = ("score", "selection", "optimizer", "backend")


@dataclass(frozen=True)
class FlatFlag:
    """One flat flag and its nesting coordinates."""

    flag: str      # --sga-cycle-period
    dest: str      # sga_cycle_period (argparse's auto-dest spelling)
    section: str   # engine section or axis name; "" = top-level sga key
    key: str       # key inside the section ("name" for axis name flags)
    type: object   # bool / int / float / str / list (comma-joined str)
    choices: tuple = ()
    help: str = ""
    nullable: bool = False  # accepts the "null" spelling (explicit null)


def flat_flags() -> list:
    """The full flag table (engine sections + per-axis plugin vocabularies).

    Same-name params from different plugins of one axis must declare
    identically (same type/choices) or the collision fails at table build.
    """
    flags = []

    def _add(section, key, typ, choices=(), help="", nullable=False):
        flag = f"--sga-{(section + '-' if section else '')}{key}".replace("_", "-")
        dest = flag[2:].replace("-", "_")
        flags.append(FlatFlag(flag, dest, section, key, typ, tuple(choices),
                              help, nullable))

    for p in TOP:
        _add("", p.name, p.type, choices=p.choices or (), help=p.doc,
             nullable=p.nullable)
    for section, decl in ENGINE_SECTIONS.items():
        for p in decl:
            if p.yaml_only:
                continue  # e.g. start_iter_percent: launcher-side only
            _add(section, p.name, p.type, choices=p.choices or (),
                 help=p.doc, nullable=p.nullable)

    for axis in _AXES:
        if axis == "score":
            names = SCORE_NAMES  # registry data: no decls, no params
        else:
            names = sorted(PLUGIN_PARAMS[axis])
        _add(axis, "name", str, choices=names,
             help=f"the {axis} plugin (registered names are the choices)")
        if axis == "score":
            continue
        first = {}  # param name -> declaring Param (first wins)
        for decl in PLUGIN_PARAMS[axis].values():
            for p in decl:
                prior = first.get(p.name)
                if prior is not None and (prior.type, prior.choices) != (p.type, p.choices):
                    raise ValueError(
                        f"flat vocabulary collision on --sga-{axis}-"
                        f"{p.name}: plugins of one axis must declare it "
                        f"identically ({(prior.type, prior.choices)} vs "
                        f"{(p.type, p.choices)})"
                    )
                first.setdefault(p.name, p)
        for name, p in sorted(first.items()):
            _add(axis, name, p.type, choices=p.choices or (), help=p.doc,
                 nullable=p.nullable)
    return flags


_FLAG_TABLE = None


def _flags():
    global _FLAG_TABLE
    if _FLAG_TABLE is None:
        _FLAG_TABLE = {f.dest: f for f in flat_flags()}
    return _FLAG_TABLE


# argparse keys the fork already owns; never generated here.
_FORK_DESTS = frozenset(("sga_config", "sga_allow_config_override"))


class _ExplicitNull:
    """Argparse-side marker for the ``null`` spelling (None itself means
    "flag absent" under the all-default-None inertness invariant)."""

    __slots__ = ()

    def __repr__(self):
        return "null"


NULL = _ExplicitNull()


def _nullable_type(base):
    def parse(text):
        return NULL if text == "null" else base(text)

    return parse


def add_sga_flat_args(parser: argparse.ArgumentParser):
    """Register every flat flag on a megatron-side parser (all default None).

    Called from the training entry's ``extra_args_provider`` chain; bools are
    ``true``/``false`` strings (megatron style), lists ride as one
    comma-joined token, nullable params also accept ``null``.
    """
    for f in flat_flags():
        kw = {"default": None, "help": f.help or f.flag}
        if f.type is bool:
            kw.update(type=str.lower, choices=("true", "false"))
        elif f.type is list:
            kw.update(type=str)
        elif f.nullable:
            kw.update(type=_nullable_type(f.type))
            if f.choices:
                kw["choices"] = f.choices
        else:
            kw.update(type=f.type)
            if f.choices:
                kw["choices"] = f.choices
        parser.add_argument(f.flag, dest=f.dest, **kw)
    return parser


def sga_config_to_flags(engine_config: dict) -> list:
    """Translate a validated engine config into the flat flag sequence.

    The launcher's nested-yaml exit: every sga key becomes a visible
    flag. Round-trip contract: parse these flags and run
    ``assemble_sga_args`` and the canonical JSON comes out byte-identical
    (pinned by tests/test_arguments.py).
    """
    argv = []
    flags = _flags()

    def _emit(section, key, value):
        dest = f"sga_{(section + '_' if section else '')}{key}".replace("-", "_")
        f = flags.get(dest)
        if f is None:
            raise ValueError(
                f"engine config key '{(section + '.' if section else '')}{key}' "
                "has no flat flag — the vocabulary and the engine config "
                "disagree (stale declarations?)"
            )
        if value is None:
            if not f.nullable:
                raise ValueError(
                    f"{f.flag} got an explicit null but its declaration is "
                    "not nullable"
                )
            value = "null"
        elif f.type is bool:
            value = "true" if value else "false"
        elif f.type is list:
            value = ",".join(str(v) for v in value)
        else:
            value = str(value)
        argv.extend([f.flag, value])

    # The engine config carries no "enabled" key (it only exists when on):
    # emit the switch explicitly — the assembler requires it.
    _emit("", "enabled", True)
    for key in ("shadow", "shadow_score"):
        if key in engine_config and engine_config[key] is not None:
            _emit("", key, engine_config[key])
    for section in ENGINE_SECTIONS:
        for key, val in engine_config.get(section, {}).items():
            _emit(section, key, val)
    for axis in _AXES:
        for key, val in engine_config[axis].items():
            _emit(axis, key, val)
    return argv


def assemble_sga_args(args) -> None:
    """Fold the parsed flat flags back into ``args.sga_config`` (in place).

    The training entry calls this right after ``parse_and_validate_args`` —
    before the resume guard and the waist read ``args.sga_config``. Rules:
    no flags and no JSON -> disabled (sga_config stays ""); JSON only ->
    untouched (the fork-native form); flags only -> re-nested through the
    loader's shared validation core and written back as canonical JSON;
    BOTH -> fail loud (ambiguous precedence is never guessed).
    """
    flags = _flags()
    flat = {}
    for dest, val in vars(args).items():
        if dest in _FORK_DESTS or not dest.startswith("sga_") or val is None:
            continue
        if dest not in flags:
            raise ValueError(
                f"unknown flat sga argument {dest!r} on the namespace — the "
                "vocabulary is generated from the catalog declarations"
            )
        flat[dest] = None if val is NULL else val
    if args.sga_config and flat:
        raise ValueError(
            "--sga-config (JSON) and flat --sga-* flags are mutually "
            "exclusive; the config would be ambiguous"
        )
    if not flat:
        return  # disabled, or the fork-native JSON form: leave as-is

    # a lone explicit off-switch is a complete (disabled) specification
    # (bool flags still ride as "true"/"false" strings at this point)
    if flat == {"sga_enabled": "false"}:
        return

    sga = {}
    for dest, val in flat.items():
        f = flags[dest]
        if val is not None:
            if f.type is bool:
                if val not in ("true", "false"):
                    raise ValueError(f"{f.flag} must be true or false (got {val!r})")
                val = val == "true"
            elif f.type is list:
                val = str(val)  # comma string; the core normalizes via the parser
        # int/float flags were already parsed to numbers by argparse;
        # None here is the explicit-null spelling (NULL was unmapped above).
        target = sga if not f.section else sga.setdefault(f.section, {})
        target[f.key] = val
    if sga.get("enabled") is not True:
        raise ValueError(
            "flat sga flags require --sga-enabled true — a dense run carries "
            "no sga flags at all (pass none)"
        )
    from sga.config.loader import engine_config_from_sga_dict
    from sga.runtime import canonical_config_json

    engine_config = engine_config_from_sga_dict(sga)
    args.sga_config = canonical_config_json(engine_config)
