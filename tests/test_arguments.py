#!/usr/bin/env python3
"""The flat-flag pipeline tests: one vocabulary, one transport.

Pins the flag-as-transport contract: the yaml ``sga:`` section keys and
the hand-written ``--sga-*`` flags are two spellings of ONE flat
vocabulary declared in sga/config/arguments.py. The launcher emits
flags; megatron's argparse parses them onto the namespace; validation
reads the namespace's explicit diff. Canonical flat-dict equality is
the contract (the engine print and the resume guard both speak it).

Run: python tests/test_arguments.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config.arguments import add_sga_args
from sga.config.validation import (
    canonical_flat_json,
    explicit_from_args,
    sga_actions,
    sga_defaults,
    validate_sga_flat,
)
from sga.launcher.backend_megatron import _get_args_megatron

ENABLED = {
    "sga_enabled": True,
    "sga_cycle_start_iter": 20,
    "sga_cycle_period": 8,
    "sga_score": "update",
    "sga_selection_method": "histogram",
    "sga_selection_ratio_percent": 10.0,
    "sga_optimizer": "adamw",
    "sga_backend": "torch_pipelined_index",
}


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
    """Parse flat flags exactly like the training entry (waist flags too)."""
    parser = _LoudParser()
    parser.add_argument("--sga-allow-config-override",
                        dest="sga_allow_config_override",
                        action="store_true")
    add_sga_args(parser)
    return parser.parse_args(argv)


RUN_BASE = {
    "experiment": {"exp_name": "t", "exp_dir": "/tmp/x",
                   "task": {"type": "train", "backend": "megatron",
                            "entrypoint": "./sga/train_gpt.py"},
                   "runner": {"type": "ssh"}},
    "train": {"system": {"tensor_model_parallel_size": 1},
              "model": {"num_layers": 2, "train_iters": 5000},
              "data": {"data_path": "/tmp/x"}},
}


def _cfg_with_sga(sga):
    cfg = OmegaConf.create(RUN_BASE)
    cfg.sga = sga
    return cfg


def _flags_of(cfg):
    args = _get_args_megatron(cfg)
    tokens, it = [], iter(args)
    for tok in it:
        if tok.startswith("--sga-"):
            tokens.extend([tok, next(it)])
    return tokens


def _canon_from_namespace(argv, train_iters=5000):
    args = _parse(argv)
    return validate_sga_flat(explicit_from_args(args), train_iters=train_iters)


def test_flag_table_declared_in_arguments_py():
    """The vocabulary IS the add_argument set: every dest hyphenates to its
    flag and the historical surface is pinned (28 dests)."""
    actions = sga_actions()
    assert len(actions) == 28, sorted(actions)
    for dest, action in actions.items():
        assert action.option_strings == ["--" + dest.replace("_", "-")], dest
    assert set(sga_defaults()) == {
        "sga_enabled", "sga_shadow", "sga_shadow_score",
        "sga_always_sync_layers",
        "sga_cycle_start_iter", "sga_cycle_start_iter_percent",
        "sga_cycle_period", "sga_cycle_decouple_full_sync_interval_and_period",
        "sga_cycle_full_sync_interval",
        "sga_checks_count_guard",
        "sga_trace_dir", "sga_trace_branch_points", "sga_trace_window",
        "sga_score",
        "sga_selection", "sga_selection_method",
        "sga_selection_ratio_percent", "sga_selection_histogram_mode",
        "sga_selection_adaptive_warmup_syncs",
        "sga_selection_adaptive_warmup_mode",
        "sga_optimizer", "sga_optimizer_residual_mean",
        "sga_optimizer_compact_search",
        "sga_backend", "sga_backend_emission",
        "sga_backend_high_priority_group",
        "sga_backend_index_slices", "sga_backend_mask_slices",
    }
    print("ok: 28 flat flags declared, hyphenation 1:1")


def test_dual_form_equivalence():
    """yaml sga: section -> launcher flags -> argparse namespace ->
    validation produces the SAME canonical flat dict as validating the
    section directly. The two spellings are one pipeline."""
    cfg = _cfg_with_sga(dict(ENABLED))
    canon_direct = validate_sga_flat(
        OmegaConf.to_container(cfg.sga, resolve=True), train_iters=5000)
    canon_ns = _canon_from_namespace(_flags_of(cfg))
    assert canonical_flat_json(canon_direct) == canonical_flat_json(canon_ns)
    print("ok: yaml and hand-written flags canonicalize identically")


def test_dual_form_explicit_null_round_trip():
    """R5: the explicit-null spelling of the nullable warmup flag
    survives the dual-form round trip — a yaml flat section carrying
    sga_selection_adaptive_warmup_syncs: null and the CLI
    --sga-selection-adaptive-warmup-syncs null canonicalize identically
    (a launcher/OmegaConf conversion that dropped the null or reset it
    to the default would fail here)."""
    adaptive = dict(ENABLED, sga_selection_method="adaptive",
                    sga_selection_adaptive_warmup_syncs=None)
    cfg = _cfg_with_sga(dict(adaptive))
    canon_direct = validate_sga_flat(
        OmegaConf.to_container(cfg.sga, resolve=True), train_iters=5000)
    assert canon_direct["sga_selection_adaptive_warmup_syncs"] is None
    tokens = _flags_of(cfg)
    assert tokens[tokens.index("--sga-selection-adaptive-warmup-syncs") + 1] \
        == "null"
    canon_ns = _canon_from_namespace(tokens)
    assert canon_ns["sga_selection_adaptive_warmup_syncs"] is None
    assert canonical_flat_json(canon_direct) == canonical_flat_json(canon_ns)
    print("ok: explicit null round-trips through both spellings")


def test_decoupled_cycle_round_trip():
    flat = dict(ENABLED,
                sga_cycle_decouple_full_sync_interval_and_period=True,
                sga_cycle_full_sync_interval=4,
                sga_cycle_period=8)
    canon_direct = validate_sga_flat(dict(flat))
    args = _parse(["--sga-enabled", "true",
                   "--sga-cycle-start-iter", "20",
                   "--sga-cycle-period", "8",
                   "--sga-cycle-decouple-full-sync-interval-and-period",
                   "true", "--sga-cycle-full-sync-interval", "4",
                   "--sga-score", "update",
                   "--sga-selection-method", "histogram",
                   "--sga-selection-ratio-percent", "10",
                   "--sga-optimizer", "adamw",
                   "--sga-backend", "torch_pipelined_index"])
    canon_ns = validate_sga_flat(explicit_from_args(args))
    assert canonical_flat_json(canon_direct) == canonical_flat_json(canon_ns)
    print("ok: decoupled cycle round-trips through the flags")


def test_percent_resolves_train_side():
    """start_iter_percent is a real flag resolved against train_iters at
    validation time (the launcher pre-resolves with the yaml's value)."""
    flat = {k: v for k, v in ENABLED.items()
            if k != "sga_cycle_start_iter"}
    flat["sga_cycle_start_iter_percent"] = 0.4
    canon = validate_sga_flat(flat, train_iters=5000)
    from sga.runtime import materialize_engine_config
    assert materialize_engine_config(canon)["cycle"]["start_iter"] == 20
    assert canon["sga_cycle_start_iter"] == 20
    assert "sga_cycle_start_iter_percent" not in canon  # resolved away
    print("ok: percent resolves train-side and leaves the canonical form")


def test_no_input_disabled_and_flags_inert():
    """No sga flags at all: every dest at its default, the explicit diff is
    empty and validation reads disabled-clean (dense, byte-identical)."""
    args = _parse([])
    assert explicit_from_args(args) == {}
    assert validate_sga_flat(explicit_from_args(args)) is None
    # waist contract: inactive without sga_enabled
    assert getattr(args, "sga_enabled", False) is False
    print("ok: zero-flag run is dense-clean")


def test_disabled_yaml_emits_single_flag():
    cfg = _cfg_with_sga({"sga_enabled": False})
    tokens = _flags_of(cfg)
    assert tokens == ["--sga-enabled", "false"]
    assert _canon_from_namespace(tokens) is None
    # an explicit sga_backend (even 'dense', the default) is a set key on
    # the yaml route and errors while disabled
    cfg2 = _cfg_with_sga({"sga_enabled": False, "sga_backend": "dense"})
    expect_error(lambda: validate_sga_flat(
        OmegaConf.to_container(cfg2.sga, resolve=True)),
        "sga_backend is set but sga_enabled is false")
    print("ok: disabled yaml emits exactly one flag; set keys error")


def test_cli_disabled_with_extra_flags_rejected():
    """The flag route enforces the same disabled-cleanliness contract as
    the yaml route: any --sga-* flag without --sga-enabled true fails
    loud (never silently dense)."""
    args = _parse(["--sga-backend", "torch_pipelined_index"])
    expect_error(lambda: validate_sga_flat(explicit_from_args(args)),
                 "sga_backend is set but sga_enabled is false")
    print("ok: CLI disabled + extra flags rejected loud")


def test_unknown_flat_values_rejected():
    expect_error(lambda: _parse(["--sga-enabled", "true",
                                 "--sga-score", "update",
                                 "--sga-selection-method", "hist",
                                 "--sga-selection-ratio-percent", "10",
                                 "--sga-optimizer", "adamw",
                                 "--sga-backend", "torch_pipelined_index"]),
                 "invalid choice")
    expect_error(lambda: _parse(["--sga-enabled", "maybe"]),
                 "must be 'true' or 'false'")
    print("ok: argparse rejects unknown values at parse time")


def test_explicit_false_and_null_spellings():
    """bools carry explicit false; the nullable warmup accepts null."""
    args = _parse(["--sga-enabled", "true", "--sga-score", "update",
                   "--sga-selection-method", "adaptive",
                   "--sga-selection-ratio-percent", "10",
                   "--sga-optimizer", "adamw",
                   "--sga-backend", "torch_pipelined_index",
                   "--sga-shadow", "false",
                   "--sga-selection-adaptive-warmup-syncs", "null"])
    explicit = explicit_from_args(args)
    assert explicit["sga_selection_adaptive_warmup_syncs"] is None
    canon = validate_sga_flat(explicit)
    assert canon["sga_selection_adaptive_warmup_syncs"] is None
    assert canon["sga_shadow"] is False
    print("ok: explicit false and null spellings survive the pipeline")


def test_legacy_nested_keys_are_unknown():
    """A pre-flat nested section fails loud as unknown keys, never as a
    silently-ignored config."""
    for legacy in ("cycle", "selection", "score", "trace", "checks"):
        flat = dict(ENABLED)
        flat[legacy] = {"start_iter": 1}
        expect_error(lambda l=legacy: validate_sga_flat(flat), "unknown keys")
    print("ok: legacy nested keys rejected as unknown")


def test_typo_plugin_param_rejected():
    base = dict(ENABLED)
    expect_error(lambda: validate_sga_flat({**base,
                                             "sga_selection_ration": 10.0}),
                 "unknown keys")
    expect_error(lambda: _parse([
        "--sga-enabled", "true", "--sga-score", "update",
        "--sga-selection-method", "histogram",
        "--sga-selection-ratio-percent", "10",
        "--sga-optimizer", "adamw",
        "--sga-backend", "torch_pipelined_index",
        "--sga-selection-histogram-mdoe", "fast"]),
        "unrecognized arguments")
    # a DECLARED flag with a typo'd VALUE fails at parse (argparse IS part
    # of the shared validator; the yaml route hits the same check in
    # validation — pinned in test_config.py)
    expect_error(lambda: _parse([
        "--sga-enabled", "true", "--sga-score", "update",
        "--sga-selection-method", "histogram",
        "--sga-selection-ratio-percent", "10",
        "--sga-optimizer", "adamw",
        "--sga-backend", "torch_pipelined_index",
        "--sga-selection-histogram-mode", "faster"]), "invalid choice")
    print("ok: typo'd params fail on both spellings")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL ARGUMENT TESTS PASSED")




def test_always_sync_layers_dual_form():
    """The list-kind always-sync flag: a yaml string list and the
    comma-joined CLI token (the launcher's list spelling) canonicalize
    to the SAME parsed pattern list."""
    flat = dict(ENABLED, sga_always_sync_layers=["embed", "norm"])
    cfg = _cfg_with_sga(dict(flat))
    canon_direct = validate_sga_flat(
        OmegaConf.to_container(cfg.sga, resolve=True), train_iters=5000)
    assert canon_direct["sga_always_sync_layers"] == ["embed", "norm"]
    tokens = _flags_of(cfg)
    assert tokens[tokens.index("--sga-always-sync-layers") + 1] == "embed,norm"
    canon_ns = _canon_from_namespace(tokens)
    assert canon_ns["sga_always_sync_layers"] == ["embed", "norm"]
    assert canonical_flat_json(canon_direct) == canonical_flat_json(canon_ns)
    print("ok: always-sync layers list <-> comma token canonicalize identically")


def test_always_sync_layers_empty_is_off_on_both_routes():
    """The yaml [] OFF spelling round-trips: the launcher emits it as the
    single empty token (",".join([])), and BOTH that token and a
    hand-written empty flag value canonicalize to None — identical to
    the unset flag (the broken-join defense only refuses holes BETWEEN
    patterns, never the whole-empty spelling)."""
    cfg = _cfg_with_sga(dict(ENABLED, sga_always_sync_layers=[]))
    # yaml route: the empty list itself is an OFF spelling
    canon_direct = validate_sga_flat(
        OmegaConf.to_container(cfg.sga, resolve=True), train_iters=5000)
    assert canon_direct["sga_always_sync_layers"] is None
    # launcher route: [] is emitted as the single empty token
    tokens = _flags_of(cfg)
    assert tokens[tokens.index("--sga-always-sync-layers") + 1] == ""
    canon_ns = _canon_from_namespace(tokens)
    assert canon_ns["sga_always_sync_layers"] is None
    # a hand-written empty flag value is that same token
    canon_cli = _canon_from_namespace(
        _flags_of(_cfg_with_sga(dict(ENABLED)))
        + ["--sga-always-sync-layers", ""])
    assert canon_cli["sga_always_sync_layers"] is None
    # all three spellings are identical to the unset flag
    unset = _canon_from_namespace(_flags_of(_cfg_with_sga(dict(ENABLED))))
    assert canonical_flat_json(canon_direct) == canonical_flat_json(unset)
    assert canonical_flat_json(canon_ns) == canonical_flat_json(unset)
    assert canonical_flat_json(canon_cli) == canonical_flat_json(unset)
    print("ok: empty list / empty token <-> off equivalence on both routes")


if __name__ == "__main__":
    _run_all()
