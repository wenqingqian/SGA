#!/usr/bin/env python3
"""The flat-flag vocabulary tests: single parsing pipeline + inertness.

Pins the single-pipeline contract: the nested ``sga:`` yaml section and the
hand-written ``--sga-*`` flags are two spellings of ONE pipeline — the
launcher translates the section into flags, the training entry registers
them into megatron's argparse and folds them back into the canonical
``args.sga_config`` JSON. Byte-identical canonical output is the contract.

Run: python tests/test_arguments.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config.arguments import (
    add_sga_flat_args,
    assemble_sga_args,
    flat_flags,
    sga_config_to_flags,
)
from sga.config.loader import sga_section_to_config
from sga.runtime import canonical_config_json


def expect_error(fn, needle):
    try:
        fn()
    except (ValueError, SystemExit) as e:
        assert needle in str(e), f"error missing '{needle}': {e}"
        return
    raise SystemExit(f"expected error containing '{needle}'")


class _LoudParser(argparse.ArgumentParser):
    """Surface argparse diagnostics as exceptions (default: print + exit 2)."""

    def error(self, message):
        raise ValueError(message)


def _parse(argv):
    """Parse flat flags exactly like the training entry (fork flags included)."""
    parser = _LoudParser()
    parser.add_argument("--sga-config", dest="sga_config", default="")
    parser.add_argument("--sga-allow-config-override",
                        dest="sga_allow_config_override",
                        action="store_true")
    add_sga_flat_args(parser)
    args = parser.parse_args(argv)
    assemble_sga_args(args)
    return args


def _nested_section(extra_cycle=None, **kw):
    section = {
        "enabled": True,
        "cycle": {"start_iter": 20, "period": 8, "full_sync_interval": 0},
        "score": {"name": "update"},
        "selection": {"name": "topk", "method": "histogram", "ratio_percent": 10.0},
        "optimizer": {"name": "adamw"},
        "backend": {"name": "indexed_sparse"},
    }
    if extra_cycle:
        section["cycle"].update(extra_cycle)
    section.update(kw)
    return OmegaConf.create({"sga": section, "train": {"model": {"train_iters": 5000}}})


def test_flag_table_generated_from_declarations():
    flags = {f.flag: f for f in flat_flags()}
    # engine sections
    for f in ("--sga-enabled", "--sga-shadow", "--sga-shadow-score",
              "--sga-cycle-start-iter", "--sga-cycle-period",
              "--sga-cycle-full-sync-interval", "--sga-checks-count-guard",
              "--sga-trace-dir", "--sga-trace-branch-points",
              "--sga-trace-window"):
        assert f in flags, f
    # per-axis plugin params (data-driven: a decl param implies its flag)
    assert flags["--sga-selection-ratio-percent"].type is float
    assert flags["--sga-selection-method"].choices == ("adaptive", "exact", "histogram")
    assert flags["--sga-selection-histogram-mode"].choices == ("balanced", "exact", "fast")
    assert flags["--sga-optimizer-residual-mean"].type is bool
    assert flags["--sga-backend-num-slices"].type is int
    assert flags["--sga-backend-slices"].type is int
    # score axis: names from the registry, no params
    assert flags["--sga-score-name"].choices == ("abs_g", "abs_m", "coi", "update")
    assert not [f for f in flags if f.startswith("--sga-score-")
                and f != "--sga-score-name"]
    # no start_iter_percent flag (yaml-only, needs the train_iters context)
    assert "--sga-cycle-start-iter-percent" not in flags
    # fork-owned flags are never generated
    assert "--sga-config" not in flags
    print("ok: flag table generated from declarations")


def test_dual_form_equivalence():
    """nested yaml -> flags -> argparse -> assemble == direct canonical, for
    the standard combo AND the full-surface combo (checks/trace/shadow)."""
    for cfg in (_nested_section(),
                _nested_section(checks={"count_guard": False},
                                trace={"dir": "/tmp/sga-trace-x",
                                       "branch_points": [40, 80], "window": 8},
                                shadow=True, shadow_score=False),
                _nested_section(optimizer={"name": "adamw",
                                           "residual_mean": True}),
                # int-spelled float: canonical is spelling-invariant (10 -> 10.0)
                _nested_section(selection={"name": "topk", "method": "adaptive",
                                           "ratio_percent": 10,
                                           "adaptive_warmup_syncs": None})):
        engine_config = sga_section_to_config(cfg)
        assert engine_config["selection"]["ratio_percent"] == 10.0
        if engine_config["selection"].get("adaptive"):
            assert "adaptive_warmup_syncs" in engine_config["selection"]
            assert engine_config["selection"]["adaptive_warmup_syncs"] is None
        want = canonical_config_json(engine_config)
        argv = sga_config_to_flags(engine_config)
        if engine_config["selection"].get("adaptive"):
            assert argv[argv.index("--sga-selection-adaptive-warmup-syncs") + 1] == "null"
        args = _parse(argv)
        assert args.sga_config == want, (args.sga_config, want)
    print("ok: nested-yaml and flat-flag forms produce identical canonical JSON")


def test_handwritten_flat_flags_equal():
    """A user hand-writing the flags reaches the same canonical JSON."""
    cfg = _nested_section(selection={"name": "topk", "method": "adaptive",
                                     "ratio_percent": 40.0})
    want = canonical_config_json(sga_section_to_config(cfg))
    args = _parse([
        "--sga-enabled", "true",
        "--sga-cycle-start-iter", "20",
        "--sga-cycle-period", "8",
        "--sga-cycle-full-sync-interval", "0",
        "--sga-score-name", "update",
        "--sga-selection-name", "topk",
        "--sga-selection-method", "adaptive",
        "--sga-selection-ratio-percent", "40.0",
        "--sga-optimizer-name", "adamw",
        "--sga-backend-name", "indexed_sparse",
    ])
    assert args.sga_config == want, (args.sga_config, want)
    # the engine reads the expanded adaptive defaults at build time
    got = json.loads(args.sga_config)
    assert got["selection"] == {"name": "topk", "method": "adaptive",
                                "ratio_percent": 40.0}
    print("ok: hand-written flat flags equal the nested form")


def test_percent_resolves_during_translation():
    """start_iter_percent has no flag: the launcher translation resolves it
    to the absolute start_iter BEFORE emitting flags."""
    cfg = _nested_section()
    OmegaConf.update(cfg, "sga.cycle", {"start_iter_percent": 10.0, "period": 8,
                                        "full_sync_interval": 0},
                     merge=False)
    engine_config = sga_section_to_config(cfg)
    assert engine_config["cycle"]["start_iter"] == 500
    argv = sga_config_to_flags(engine_config)
    assert "--sga-cycle-start-iter" in argv
    assert argv[argv.index("--sga-cycle-start-iter") + 1] == "500"
    print("ok: start_iter_percent resolves during the translation")


def test_json_and_flat_conflict_rejected():
    expect_error(
        lambda: _parse(["--sga-config", '{"score": {"name": "abs_m"}}',
                        "--sga-enabled", "true"]),
        "mutually exclusive",
    )
    print("ok: JSON + flat flags conflict fails loud")


def test_json_only_form_passthrough():
    canonical = json.dumps({"score": {"name": "abs_m"}}, sort_keys=True)
    args = _parse(["--sga-config", canonical])
    assert args.sga_config == canonical
    print("ok: JSON-only form passes through untouched")


def test_no_input_disabled_and_flags_inert():
    args = _parse([])
    assert args.sga_config == ""
    table = {f.dest: f for f in flat_flags()}
    for dest, val in vars(args).items():
        if dest in table:
            assert val is None, f"{dest} not inert: {val!r}"
    print("ok: no input = disabled; every flat flag defaults to None")


def test_disabled_yaml_emits_zero_flags():
    from sga.launcher.backend_megatron import _get_args_megatron, _update_config_train
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp(prefix="sga-flat-disabled-")
    try:
        cfg = OmegaConf.create({
            "experiment": {"exp_name": "t", "exp_dir": os.path.join(tmp, "exp"),
                           "task": {"type": "train", "backend": "megatron",
                                    "entrypoint": "./sga/train_gpt.py"},
                           "runner": {"type": "ssh"}},
            "sga": {"enabled": False, "backend": {"name": "dense"}},
            "train": {"system": {"tensor_model_parallel_size": 1,
                                 "logging": {}},
                      "model": {"num_layers": 2}, "data": {"data_path": "/tmp/x"}},
        })
        _update_config_train(cfg)
        args = _get_args_megatron(cfg)
        assert not [a for a in args if a.startswith("--sga")], args
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: disabled yaml emits zero --sga-* tokens")


def test_unknown_flat_values_rejected():
    # vocabulary domains are enforced by argparse choices at parse time
    expect_error(lambda: _parse(["--sga-enabled", "true",
                                 "--sga-score-name", "update",
                                 "--sga-selection-name", "topk",
                                 "--sga-selection-method", "histogram",
                                 "--sga-selection-ratio-percent", "30",
                                 "--sga-optimizer-name", "adamw",
                                 "--sga-backend-name", "quantum"]),
                 "invalid choice")
    expect_error(lambda: _parse(["--sga-enabled", "yes"]), "invalid choice")
    print("ok: unknown flat values rejected (choices)")


def test_enabled_switch_required_with_flat_flags():
    # flat flags without --sga-enabled would silently resolve to disabled ->
    # loud refusal instead, naming the missing switch
    expect_error(
        lambda: _parse(["--sga-score-name", "update"]),
        "--sga-enabled",
    )
    print("ok: flat flags without the enabled switch fail loud")


def test_explicit_false_and_null_spellings():
    # a lone explicit off-switch is a complete disabled specification
    args = _parse(["--sga-enabled", "false"])
    assert args.sga_config == ""
    # ...but false alongside any other sga flag is a configuration error
    expect_error(lambda: _parse(["--sga-enabled", "false",
                                 "--sga-score-name", "update"]),
                 "require --sga-enabled true")
    # hand-written null spelling reaches the engine as a real null
    cfg = _nested_section(selection={"name": "topk", "method": "adaptive",
                                     "ratio_percent": 40.0,
                                     "adaptive_warmup_syncs": None})
    want = canonical_config_json(sga_section_to_config(cfg))
    args = _parse([
        "--sga-enabled", "true",
        "--sga-cycle-start-iter", "20",
        "--sga-cycle-period", "8",
        "--sga-cycle-full-sync-interval", "0",
        "--sga-score-name", "update",
        "--sga-selection-name", "topk",
        "--sga-selection-method", "adaptive",
        "--sga-selection-ratio-percent", "40.0",
        "--sga-selection-adaptive-warmup-syncs", "null",
        "--sga-optimizer-name", "adamw",
        "--sga-backend-name", "indexed_sparse",
    ])
    assert args.sga_config == want, (args.sga_config, want)
    assert json.loads(args.sga_config)["selection"]["adaptive_warmup_syncs"] is None
    # null on a non-nullable flag fails loud at parse time
    expect_error(lambda: _parse(["--sga-cycle-period", "null"]),
                 "invalid int value")
    print("ok: explicit false / hand-written null spellings")


def test_schema_and_resolve_cli():
    import subprocess

    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-m", "sga.config", "schema"],
                         capture_output=True, text=True, cwd=root, env=env)
    assert out.returncode == 0, out.stderr
    schema = json.loads(out.stdout)
    assert schema["score"]["names"] == ["abs_g", "abs_m", "coi", "update"]
    circular = schema["selection"]["topk"]
    by_name = {p["name"]: p for p in circular}
    assert by_name["ratio_percent"]["required"] is True
    assert by_name["method"]["required"] is True
    assert by_name["histogram_mode"]["default"] == "exact"
    assert by_name["adaptive_warmup_syncs"]["nullable"] is True
    assert schema["flat_flags"]["selection"]["histogram_mode"] == \
        "--sga-selection-histogram-mode"
    # argc-strict: trailing junk on schema exits with usage

    example = os.path.join(root, "examples", "llama3", "train_1b_adamw.yaml")
    out = subprocess.run([sys.executable, "-m", "sga.config", "resolve", example],
                         capture_output=True, text=True, cwd=root, env=env)
    assert out.returncode == 0, out.stderr
    resolved = json.loads(out.stdout)
    assert resolved["selection"]["ratio_percent"] == 10.0
    # plugin defaults expanded (canonical JSON would omit them) — and the
    # expansion is method-aware: only the configured method's keys appear,
    # and the block rebuilds through build_selection verbatim (a union-
    # default dump would be rejected by the constructor's foreign-key check)
    from sga.plugins import build_selection, register_default_plugins
    register_default_plugins()
    sel = resolved["selection"]
    assert set(sel) == {"name", "method", "ratio_percent", "histogram_mode"}, sel
    build_selection(sel["name"], {k: v for k, v in sel.items() if k != "name"})
    assert resolved["optimizer"]["residual_mean"] is False
    assert resolved["cycle"] == {"start_iter": 20, "period": 8,
                                 "full_sync_interval": 0}

    # schema is argc-strict (trailing junk prints usage, exits 2)
    out = subprocess.run([sys.executable, "-m", "sga.config", "schema", "x"],
                         capture_output=True, text=True, cwd=root, env=env)
    assert out.returncode == 2 and "schema" in out.stderr
    # resolve on a disabled config says so instead of dumping an empty dict
    disabled = os.path.join(root, "tests", "_tmp_disabled.yaml")
    with open(disabled, "w") as f:
        f.write("sga: {enabled: false, backend: {name: dense}}\n")
    try:
        out = subprocess.run([sys.executable, "-m", "sga.config", "resolve",
                              disabled], capture_output=True, text=True,
                             cwd=root, env=env)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout) == {"sga": "disabled (dense)"}
    finally:
        os.unlink(disabled)

    # the other methods expand exactly their own vocabulary (adaptive:
    # +2 adaptive keys; exact: none) and equally reconstruct
    for method, extra in (("adaptive", {"adaptive_warmup_syncs",
                                        "adaptive_warmup_mode"}),
                          ("exact", set())):
        tmp = os.path.join(root, "tests", "_tmp_method.yaml")
        with open(tmp, "w") as f:
            f.write(f"sga: {{enabled: true, backend: {{name: indexed_sparse}}, "
                    f"score: {{name: update}}, optimizer: {{name: adamw}}, "
                    f"selection: {{name: topk, "
                    f"method: {method}, ratio_percent: 10.0}}}}\n")
        try:
            out = subprocess.run([sys.executable, "-m", "sga.config", "resolve",
                                  tmp], capture_output=True, text=True,
                                 cwd=root, env=env)
            assert out.returncode == 0, out.stderr
            sel = json.loads(out.stdout)["selection"]
            assert set(sel) == {"name", "method", "ratio_percent"} | extra, sel
            build_selection(sel["name"],
                            {k: v for k, v in sel.items() if k != "name"})
        finally:
            os.unlink(tmp)
    print("ok: schema + resolve CLI")


def test_typo_plugin_param_rejected_both_forms():
    # nested form: unknown plugin param named in the error (loader gate)
    cfg = _nested_section(selection={"name": "topk", "method": "histogram",
                                     "ratio_percent": 10.0, "rtaio": 5})
    expect_error(lambda: sga_section_to_config(cfg), "unknown params ['rtaio']")
    # flat form: the vocabulary has no such flag -> argparse rejects it
    expect_error(lambda: _parse(["--sga-selection-rtaio", "5"]), "unrecognized")
    print("ok: typo'd plugin param rejected in both forms")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests OK")
