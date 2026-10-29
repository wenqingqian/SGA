#!/usr/bin/env python3
"""The flat-flag pipeline tests: one vocabulary, one transport.

Pins the flag-as-transport contract: the yaml ``sga:`` section keys and
the hand-written ``--sga-*`` flags are two spellings of ONE flat
vocabulary declared in sga/config/arguments.py. The launcher emits
flags; megatron's argparse parses them onto the namespace; validation
(validate_sga_args, megatron validate_args style) runs on the namespace
and writes derived values back onto it. Raw attrs equality (defaults +
the namespace's sga_* attrs, the same comparison the resume guard
speaks) is the contract.

Run: python tests/test_arguments.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from sga.config import sga_config_attrs
from sga.config.arguments import add_sga_args
from sga.config.validation import sga_actions, sga_defaults, validate_sga_args
from sga.launcher.backend_megatron import _get_args_megatron, _prevalidate_sga

ENABLED = {
    "sga_enabled": True,
    "sga_cycle_start_iter": 20,
    "sga_cycle_period": 8,
    "sga_score": "update",
    "sga_selection_method": "histogram",
    # the frozen selection combo: validation refuses any other
    # method/mode pairing
    "sga_selection_histogram_mode": "balanced",
    "sga_selection_ratio_percent": 10.0,
    "sga_optimizer": "adamw",
    "sga_backend": "torch_pipelined_index",
}


def expect_error(fn, needle):
    try:
        fn()
    except (ValueError, SystemExit, AssertionError) as e:
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


def _ns_from_flags(tokens, train_iters=5000, pp=1):
    """The namespace route: parse flags, attach the two non-sga keys,
    validate (write-backs land on the namespace)."""
    ns = _parse(tokens)
    ns.train_iters = train_iters
    ns.pipeline_model_parallel_size = pp
    validate_sga_args(ns)
    return ns


def _raw_json(attrs_owner):
    """The raw comparison form (the resume guard's _config_json contract):
    declared defaults + the namespace's sga_* attrs."""
    return json.dumps(dict(sga_defaults(), **sga_config_attrs(attrs_owner)),
                      sort_keys=True, default=repr)


def test_flag_table_declared_in_arguments_py():
    """The vocabulary IS the add_argument set: every dest hyphenates to its
    flag and the historical surface is pinned (39 dests)."""
    actions = sga_actions()
    assert len(actions) == 39, sorted(actions)
    for dest, action in actions.items():
        assert action.option_strings == ["--" + dest.replace("_", "-")], dest
    assert set(sga_defaults()) == {
        "sga_enabled", "sga_shadow", "sga_shadow_score",
        "sga_always_sync_layers",
        "sga_cycle_start_iter", "sga_cycle_start_iter_percent",
        "sga_cycle_period", "sga_cycle_decouple_full_sync_interval_and_period",
        "sga_cycle_full_sync_interval",
        "sga_checks", "sga_checks_count_guard",
        "sga_checks_support_consistency", "sga_checks_norm_spread",
        "sga_checks_rng_bracket",
        "sga_trace_dir", "sga_trace_branch_points", "sga_trace_window",
        "sga_trace_dump_dtype",
        "sga_score",
        "sga_selection", "sga_selection_method",
        "sga_selection_ratio_percent", "sga_selection_histogram_mode",
        "sga_selection_topk_histogram_add_global_allreduce",
        "sga_selection_adaptive_warmup_syncs",
        "sga_selection_adaptive_warmup_mode",
        "sga_selection_block_size", "sga_selection_block_reduce",
        "sga_optimizer", "sga_optimizer_residual_mean",
        "sga_optimizer_full_sync_inactive_lr_scale",
        "sga_optimizer_compact_search",
        "sga_backend", "sga_backend_emission", "sga_backend_layout",
        "sga_backend_high_priority_group",
        "sga_backend_index_slices", "sga_backend_mask_slices",
        "sga_backend_timer",
    }
    print("ok: 39 flat flags declared, hyphenation 1:1")


def test_dual_form_equivalence():
    """yaml sga: section -> launcher flags -> argparse namespace ->
    validation produces the SAME raw attrs as the hand-written flag
    spelling. The two spellings are one pipeline."""
    cfg = _cfg_with_sga(dict(ENABLED))
    ns_launcher = _ns_from_flags(_flags_of(cfg))
    hand = ["--sga-enabled", "true", "--sga-cycle-start-iter", "20",
            "--sga-cycle-period", "8", "--sga-score", "update",
            "--sga-selection-method", "histogram",
            "--sga-selection-histogram-mode", "balanced",
            "--sga-selection-ratio-percent", "10",
            "--sga-optimizer", "adamw",
            "--sga-backend", "torch_pipelined_index"]
    ns_hand = _ns_from_flags(hand)
    assert _raw_json(ns_launcher) == _raw_json(ns_hand)
    print("ok: yaml and hand-written flags validate identically")


def test_dual_form_explicit_null_round_trip():
    """The explicit-null spelling of the nullable warmup flag
    survives the dual-form round trip — a yaml flat section carrying
    sga_selection_adaptive_warmup_syncs: null and the CLI
    --sga-selection-adaptive-warmup-syncs null validate identically
    (a launcher/OmegaConf conversion that dropped the null or reset it
    to the default would fail here). The nullable flag rides the frozen
    histogram-balanced selection (under it the knob is dead config, but
    the spelling still round-trips)."""
    null_syncs = dict(ENABLED,
                      sga_selection_adaptive_warmup_syncs=None)
    cfg = _cfg_with_sga(dict(null_syncs))
    tokens = _flags_of(cfg)
    assert tokens[tokens.index("--sga-selection-adaptive-warmup-syncs") + 1] \
        == "null"
    ns_launcher = _ns_from_flags(tokens)
    assert ns_launcher.sga_selection_adaptive_warmup_syncs is None
    # the hand spelling written out independently of the launcher tokens
    # (a copied token list would make the raw-equality assert vacuous)
    hand = ["--sga-enabled", "true",
            "--sga-backend", "torch_pipelined_index",
            "--sga-cycle-period", "8",
            "--sga-cycle-start-iter", "20",
            "--sga-optimizer", "adamw",
            "--sga-score", "update",
            "--sga-selection-adaptive-warmup-syncs", "null",
            "--sga-selection-method", "histogram",
            "--sga-selection-histogram-mode", "balanced",
            "--sga-selection-ratio-percent", "10.0"]
    ns_hand = _ns_from_flags(hand)
    assert ns_hand.sga_selection_adaptive_warmup_syncs is None
    assert _raw_json(ns_launcher) == _raw_json(ns_hand)
    print("ok: explicit null round-trips through both spellings")


def test_decoupled_cycle_round_trip():
    cfg = _cfg_with_sga(dict(
        ENABLED, sga_cycle_decouple_full_sync_interval_and_period=True,
        sga_cycle_full_sync_interval=4, sga_cycle_period=8))
    ns_launcher = _ns_from_flags(_flags_of(cfg))
    hand = ["--sga-enabled", "true",
            "--sga-cycle-start-iter", "20",
            "--sga-cycle-period", "8",
            "--sga-cycle-decouple-full-sync-interval-and-period",
            "true", "--sga-cycle-full-sync-interval", "4",
            "--sga-score", "update",
            "--sga-selection-method", "histogram",
            "--sga-selection-histogram-mode", "balanced",
            "--sga-selection-ratio-percent", "10",
            "--sga-optimizer", "adamw",
            "--sga-backend", "torch_pipelined_index"]
    ns_hand = _ns_from_flags(hand)
    assert _raw_json(ns_launcher) == _raw_json(ns_hand)
    print("ok: decoupled cycle round-trips through the flags")


def test_percent_resolves_train_side():
    """start_iter_percent is a real flag resolved against train_iters at
    validation time (the launcher pre-resolves with the yaml's value)."""
    tokens = [tok for k, v in ENABLED.items() if k != "sga_cycle_start_iter"
              for tok in ("--" + k.replace("_", "-"), str(v))]
    tokens += ["--sga-cycle-start-iter-percent", "0.4"]
    ns = _ns_from_flags(tokens, train_iters=5000)
    assert ns.sga_cycle_start_iter == 20
    assert ns.sga_cycle_start_iter_percent is None  # resolved away
    print("ok: percent resolves train-side onto the namespace")


def test_no_input_disabled_and_flags_inert():
    """No sga flags at all: every dest at its default, validation reads
    disabled and runs zero checks (dense, byte-identical)."""
    args = _parse([])
    assert args.sga_enabled is False
    validate_sga_args(_ns_from_flags([]))
    # waist contract: inactive without sga_enabled
    print("ok: zero-flag run is dense-clean")


def test_disabled_yaml_emits_single_flag():
    cfg = _cfg_with_sga({"sga_enabled": False})
    tokens = _flags_of(cfg)
    assert tokens == ["--sga-enabled", "false"]
    validate_sga_args(_ns_from_flags(tokens))  # disabled: zero checks
    # an explicit sga_backend under disabled is silently ignored
    # (megatron semantics: a disabled feature's sub-args are never
    # validated)
    cfg2 = _cfg_with_sga({"sga_enabled": False, "sga_backend": "dense"})
    validate_sga_args(_ns_from_flags(_flags_of(cfg2)))
    print("ok: disabled yaml emits exactly one flag; extra keys ignored")


def test_cli_disabled_with_extra_flags_accepted_silently():
    """The flag route carries the same disabled semantics as the yaml
    route: any --sga-* flag without --sga-enabled true is silently
    ignored (megatron semantics: a disabled feature's sub-args are
    never validated)."""
    ns = _parse(["--sga-backend", "torch_pipelined_index"])
    ns.train_iters = 5000
    ns.pipeline_model_parallel_size = 1
    validate_sga_args(ns)  # must not raise
    print("ok: CLI disabled + extra flags silently ignored")


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
    ns = _ns_from_flags([
        "--sga-enabled", "true", "--sga-score", "update",
        "--sga-selection-method", "histogram",
        "--sga-selection-histogram-mode", "balanced",
        "--sga-selection-ratio-percent", "10",
        "--sga-optimizer", "adamw",
        "--sga-backend", "torch_pipelined_index",
        "--sga-shadow", "false",
        "--sga-selection-adaptive-warmup-syncs", "null"])
    assert ns.sga_selection_adaptive_warmup_syncs is None
    assert ns.sga_shadow is False
    print("ok: explicit false and null spellings survive the pipeline")


def test_legacy_nested_keys_are_unknown():
    """A pre-flat nested section fails loud at the launcher's prefix gate
    (there is no short form and no nested section), never as a
    silently-ignored config; a prefixed but undeclared key fails with
    the unknown-key refusal."""
    for legacy in ("cycle", "selection", "score", "trace", "checks"):
        cfg = _cfg_with_sga(dict(ENABLED, **{legacy: {"start_iter": 1}}))
        expect_error(lambda c=cfg: _prevalidate_sga(c),
                     "must be the full flat sga_* names")
    cfg = _cfg_with_sga(dict(ENABLED, sga_cycle_legacy=1))
    expect_error(lambda: _prevalidate_sga(cfg), "unknown sga: key")
    print("ok: legacy nested keys rejected at the prefix gate")


def test_typo_plugin_param_rejected():
    expect_error(lambda: _prevalidate_sga(
        _cfg_with_sga(dict(ENABLED, sga_selection_ration=10.0))),
        "unknown sga: key")
    expect_error(lambda: _parse([
        "--sga-enabled", "true", "--sga-score", "update",
        "--sga-selection-method", "histogram",
        "--sga-selection-ratio-percent", "10",
        "--sga-optimizer", "adamw",
        "--sga-backend", "torch_pipelined_index",
        "--sga-selection-histogram-mdoe", "fast"]),
        "unrecognized arguments")
    # a DECLARED flag with a typo'd VALUE fails at parse (argparse IS part
    # of the shared validator; the yaml route hits the same check in the
    # launcher pre-validation — pinned in test_config.py)
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
    comma-joined CLI token (the launcher's list spelling) parse to the
    SAME sorted pattern list on the namespace."""
    cfg = _cfg_with_sga(dict(ENABLED, sga_always_sync_layers=["embed", "norm"]))
    tokens = _flags_of(cfg)
    assert tokens[tokens.index("--sga-always-sync-layers") + 1] == "embed,norm"
    ns_launcher = _ns_from_flags(tokens)
    assert ns_launcher.sga_always_sync_layers == ["embed", "norm"]
    ns_hand = _ns_from_flags(
        _flags_of(_cfg_with_sga(dict(ENABLED)))
        + ["--sga-always-sync-layers", "norm,embed"])
    assert ns_hand.sga_always_sync_layers == ["embed", "norm"]
    assert _raw_json(ns_launcher) == _raw_json(ns_hand)
    print("ok: always-sync layers list <-> comma token parse identically")


def test_always_sync_layers_empty_is_off_on_both_routes():
    """The yaml [] OFF spelling round-trips: the launcher emits it as the
    single empty token (",".join([])), and BOTH that token and a
    hand-written empty flag value parse to None — identical to the unset
    flag (the broken-join defense only refuses holes BETWEEN patterns,
    never the whole-empty spelling)."""
    cfg = _cfg_with_sga(dict(ENABLED, sga_always_sync_layers=[]))
    tokens = _flags_of(cfg)
    assert tokens[tokens.index("--sga-always-sync-layers") + 1] == ""
    ns_launcher = _ns_from_flags(tokens)
    assert ns_launcher.sga_always_sync_layers is None
    # a hand-written empty flag value is that same token
    ns_cli = _ns_from_flags(
        _flags_of(_cfg_with_sga(dict(ENABLED)))
        + ["--sga-always-sync-layers", ""])
    assert ns_cli.sga_always_sync_layers is None
    # all three spellings are identical to the unset flag
    ns_unset = _ns_from_flags(_flags_of(_cfg_with_sga(dict(ENABLED))))
    assert ns_unset.sga_always_sync_layers is None
    assert _raw_json(ns_launcher) == _raw_json(ns_unset)
    assert _raw_json(ns_cli) == _raw_json(ns_unset)
    print("ok: empty list / empty token <-> off equivalence on both routes")


if __name__ == "__main__":
    _run_all()
