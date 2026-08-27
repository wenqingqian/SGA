# Copyright (c) 2026, SGA framework contributors.
"""Engine builder — the single entry the fork's sga_waist imports.

``build_engine(config, api_version)`` validates the four-field config against
the plugin registries (declaration-based, no scheme names), enforces the v1
config-time assertion set against the megatron args, constructs the four
plugin objects and returns the SGAEngine. Failures surface HERE (during model
construction), never at the first optimizer step.
"""

import json

import torch

SGA_API_VERSION = 1


def _assert_args_compatible() -> None:
    """v1 config-time assertion set (notes/sga-waist-api-v1.md section 6)."""
    from megatron.training import get_args

    args = get_args()

    def _require(cond, why):
        if not cond:
            raise ValueError(f"SGA v1 is incompatible with this configuration: {why}")

    _require(not getattr(args, "use_distributed_optimizer", False),
             "--use-distributed-optimizer must be off (v1 supports the "
             "allreduce 'replicated world' path only)")
    _require(not getattr(args, "overlap_grad_reduce", False),
             "--overlap-grad-reduce must be off (future-work axis)")
    _require(not getattr(args, "calculate_per_token_loss", False),
             "--calculate-per-token-loss must be off (scale_gradients would "
             "rescale the whole buffer including residuals)")
    _require(not getattr(args, "reuse_grad_buf_for_mxfp8_param_ag", False),
             "--reuse-grad-buf-for-mxfp8-param-ag must be off (its eval path "
             "calls zero_grad_buffer unconditionally)")
    _require(getattr(args, "tensor_model_parallel_size", 1) == 1,
             "tensor_model_parallel_size must be 1 in v1 (sequence-parallel "
             "non-tensor grads allreduce would write residual coordinates)")
    # megatron leaves vpp either unset or None when the feature is off
    # (arguments.py picks per config path), so normalize before comparing.
    _require((getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) == 1,
             "virtual-pipeline model parallelism is excluded in v1 "
             "(multiple local chunks per rank consume the per-step full/"
             "sparse phase latch independently and would mix dense and "
             "sparse sync within one step)")
    _require(
        getattr(args, "expert_model_parallel_size", 1) == 1,
        "expert_model_parallel_size must be 1 in v1 (expert params create a "
        "second inner optimizer behind the chain-of-1 assert, and the "
        "ranked-selection world-scaling assumes dp_cp replication)",
    )
    _require(
        not getattr(args, "num_experts", None),
        "num_experts must be unset in v1 (expert routing creates the same "
        "second-inner-optimizer / non-replicated-world problems as EP)",
    )
    _require(
        getattr(args, "pipeline_model_parallel_size", 1) == 1
        or getattr(args, "untie_embeddings_and_output_weights", False),
        "tied embeddings with pipeline parallelism are excluded in v1 "
        "(embedding grads allreduce touches the shared buffer)",
    )
    _require(
        not getattr(args, "param_name_patterns_for_fp32_local_accumulation", None),
        "param-name-patterns-for-fp32-local-accumulation must be empty "
        "(extra main grads detach from the bucket buffer)",
    )
    _require(
        getattr(args, "cuda_graph_impl", "none") in (None, "none"),
        "CUDA graph capture must be off (host-side sparse/dense branching "
        "cannot be captured)",
    )
    # bf16-only. Under fp16 with an active loss scaler, prepare_grads()
    # (_amp_foreach_non_finite_check_and_unscale_) unscales the fp32 main
    # grads IN PLACE every step — and main_param.grad aliases the fp32
    # bucket buffer (main_param.grad = model_param.main_grad.float()), so
    # each unscale also multiplies the uncommunicated residual accumulated
    # across sparse steps, which decays by ~inv_scale per step and mixes
    # scaled/unscaled units under dynamic scaling. Proper fp16 support
    # (unscale at accumulation) is out of v1 scope; bf16 runs scaler-free.
    _require(not getattr(args, "fp16", False),
             "--fp16 is unsupported in v1 (the dynamic-loss-scaling unscale "
             "decays the uncommunicated residual through its aliased fp32 "
             "bucket buffer)")
    _require(bool(getattr(args, "bf16", False)),
             "SGA v1 expects bf16 mixed-precision training (fp32 main grads); "
             "fp16 and fp32 paths are rejected (see the fp16 note above)")
    _require(
        getattr(args, "accumulate_allreduce_grads_in_fp32", True),
        "accumulate-allreduce-grads-in-fp32 must hold (fp32 bucket buffers)",
    )

    try:
        from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

        _require(get_rerun_state_machine().mode == RerunMode.DISABLED,
                 "rerun state machine must be disabled (a rerun would "
                 "double-accumulate into the residual)")
    except ImportError:
        pass

    # check_grads nan/large check inside start_grad_sync (v1 sparse path
    # skips the dense start_grad_sync; keep the flags off for parity).
    _require(not getattr(args, "check_for_nan_in_grad", False),
             "--check-for-nan-in-grad must be off")
    _require(not getattr(args, "check_for_large_grads_in_bucket", False),
             "--check-for-large-grads must be off")

    # The EF's dense-fallback math is decoupled (AdamW-style); FusedAdam must
    # run in adam_w_mode for the two to agree (torch AdamW always does).
    if getattr(args, "optimizer", "adam") == "adam":
        _require(getattr(args, "decoupled_weight_decay", True),
                 "--decoupled-weight-decay must stay on (SGA's optimizer math "
                 "is decoupled AdamW)")


def build_engine(config: dict, api_version: int):
    """Validate + assemble the engine from the four-field config dict."""
    from sga.plugins import (
        build_backend_plugin,
        build_ef,
        build_score,
        build_selection,
        get_backend_plugin_decl,
        get_ef_decl,
        get_score_decl,
        get_selection_decl,
        register_default_plugins,
        validate_combination,
    )

    if api_version != SGA_API_VERSION:
        raise ValueError(
            f"SGA API version mismatch: fork waist speaks v{api_version}, "
            f"sga package speaks v{SGA_API_VERSION}"
        )

    register_default_plugins()
    _assert_args_compatible()

    for axis in ("score", "selection", "error_feedback", "backend"):
        if not isinstance(config.get(axis), dict) or "name" not in config[axis]:
            raise ValueError(
                f"SGA config: axis '{axis}' must be a dict with a 'name' field"
            )
    cycle = config.get("cycle", {})
    if int(cycle.get("period", 1)) < 1:
        raise ValueError("SGA config: cycle.period must be >= 1")

    score_name = config["score"]["name"]
    selection_name = config["selection"]["name"]
    ef_name = config["error_feedback"]["name"]
    backend_name = config["backend"]["name"]

    score_decl = get_score_decl(score_name)
    selection_decl = get_selection_decl(selection_name)
    ef_decl = get_ef_decl(ef_name)
    backend_decl = get_backend_plugin_decl(backend_name)

    validate_combination(score_decl, selection_decl, ef_decl, backend_decl, cycle)

    score = build_score(score_name, dict(config["score"]))
    selection = build_selection(selection_name, dict(config["selection"]), score=score)
    ef = build_ef(ef_name, dict(config["error_feedback"]))
    backend = build_backend_plugin(backend_name, dict(config["backend"]))

    from sga.engine import SGAEngine

    engine = SGAEngine(config, score, selection, ef, backend)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        canonical = json.dumps(config, sort_keys=True)
        print(f"[SGA] engine built (API v{api_version}); config: {canonical}", flush=True)
    return engine


def canonical_config_json(config: dict) -> str:
    """Canonical (sorted-key) resolved JSON — the ONLY form --sga-config takes."""
    return json.dumps(config, sort_keys=True)
