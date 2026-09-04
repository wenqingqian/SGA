# Copyright (c) 2026, SGA framework contributors.
"""Engine builder — the single entry the fork's sga_waist imports.

``build_engine(config, api_version)`` validates the flat ``sga_*`` config
against the plugin registries (declaration-based, no scheme names),
enforces the config-time assertion set against the megatron args,
constructs the four plugin objects and returns the SGAEngine. Failures
surface HERE (during model construction), never at the first optimizer
step.
"""

import torch

from sga.config.validation import (
    canonical_flat_json,
    explicit_from_flat,
    validate_sga_flat,
)

SGA_API_VERSION = 2


def _assert_args_compatible() -> None:
    """Config-time assertion set (the v2 flat-args contract).

    Backend-specific requirements do NOT live here as name branches — a
    backend plugin may expose an optional ``validate_runtime(args)`` hook
    (probed generically after construction in ``build_engine``, the same
    discipline as the engine's adapt()/wants_force_full() probes).
    """
    from megatron.training import get_args

    args = get_args()

    def _require(cond, why):
        if not cond:
            raise ValueError(f"SGA is incompatible with this configuration: {why}")

    _require(not getattr(args, "use_distributed_optimizer", False),
             "--use-distributed-optimizer must be off (only the "
             "allreduce 'replicated world' path is supported)")
    _require(not getattr(args, "overlap_grad_reduce", False),
             "--overlap-grad-reduce must be off (future-work axis)")
    _require(not getattr(args, "calculate_per_token_loss", False),
             "--calculate-per-token-loss must be off (scale_gradients would "
             "rescale the whole buffer including residuals)")
    _require(not getattr(args, "reuse_grad_buf_for_mxfp8_param_ag", False),
             "--reuse-grad-buf-for-mxfp8-param-ag must be off (its eval path "
             "calls zero_grad_buffer unconditionally)")
    _require(getattr(args, "tensor_model_parallel_size", 1) == 1,
             "tensor_model_parallel_size must be 1 (sequence-parallel "
             "non-tensor grads allreduce would write residual coordinates)")
    # megatron leaves vpp either unset or None when the feature is off
    # (arguments.py picks per config path), so normalize before comparing.
    _require((getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) == 1,
             "virtual-pipeline model parallelism is excluded "
             "(multiple local chunks per rank consume the per-step full/"
             "sparse phase latch independently and would mix dense and "
             "sparse sync within one step)")
    _require(
        getattr(args, "expert_model_parallel_size", 1) == 1,
        "expert_model_parallel_size must be 1 (expert params create a "
        "second inner optimizer behind the chain-of-1 assert, and the "
        "ranked-selection world-scaling assumes dp_cp replication)",
    )
    _require(
        not getattr(args, "num_experts", None),
        "num_experts must be unset (expert routing creates the same "
        "second-inner-optimizer / non-replicated-world problems as EP)",
    )
    _require(
        getattr(args, "pipeline_model_parallel_size", 1) == 1
        or getattr(args, "untie_embeddings_and_output_weights", False),
        "tied embeddings with pipeline parallelism are excluded "
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
    # (unscale at accumulation) is out of scope; bf16 runs scaler-free.
    _require(not getattr(args, "fp16", False),
             "--fp16 is unsupported (the dynamic-loss-scaling unscale "
             "decays the uncommunicated residual through its aliased fp32 "
             "bucket buffer)")
    _require(bool(getattr(args, "bf16", False)),
             "SGA expects bf16 mixed-precision training (fp32 main grads); "
             "fp16 and fp32 paths are rejected (see the fp16 note above)")
    # The unscale hazard is NOT fp16-only: megatron builds a
    # ConstantGradScaler for bf16 too whenever --loss-scale is set
    # (megatron/core/optimizer/__init__.py), and the same in-place
    # _amp_foreach_non_finite_check_and_unscale_ runs every step. The
    # compact-emission path makes the hole toxic: communicated
    # values live in the backend's gather buffer, which the unscale never
    # touches — every sparse step's committed grads would be off by exactly
    # the constant scale. bf16 must run scaler-free.
    _require(getattr(args, "loss_scale", None) is None,
             "--loss-scale must be unset under bf16 (a constant grad scaler "
             "unscales the fp32 bucket grads in place every step while the "
             "compact communicated values escape it — scaled/unscaled units "
             "mix across every sparse step)")
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

    # check_grads nan/large check inside start_grad_sync (the sparse path
    # skips the dense start_grad_sync; keep the flags off for parity).
    _require(not getattr(args, "check_for_nan_in_grad", False),
             "--check-for-nan-in-grad must be off")
    _require(not getattr(args, "check_for_large_grads_in_bucket", False),
             "--check-for-large-grads must be off")

    # The optimizer template's dense-fallback math is decoupled (AdamW-style);
    # FusedAdam must
    # run in adam_w_mode for the two to agree (torch AdamW always does).
    if getattr(args, "optimizer", "adam") == "adam":
        _require(getattr(args, "decoupled_weight_decay", True),
                 "--decoupled-weight-decay must stay on (SGA's optimizer math "
                 "is decoupled AdamW)")


def materialize_engine_config(canonical: dict) -> dict:
    """The engine's internal NESTED read shape, assembled from the
    validated canonical flat dict (engine.py's historical contract — the
    rest of the pipeline is flat end to end). Axis sections carry the
    explicit non-default params only; plugin constructors expand their
    declared defaults through the plugins' ``expand_plugin_params``, so a
    spelling equal to the default lands in the same final values either
    way.
    """
    explicit = explicit_from_flat(canonical)

    def _axis(axis):
        prefix = f"sga_{axis}_"
        params = {dest[len(prefix):]: explicit[dest]
                  for dest in explicit if dest.startswith(prefix)}
        return {"name": canonical[f"sga_{axis}"], **params}

    engine_config = {
        "cycle": {
            "start_iter": canonical["sga_cycle_start_iter"],
            "period": canonical["sga_cycle_period"],
            "decouple_full_sync_interval_and_period":
                canonical["sga_cycle_decouple_full_sync_interval_and_period"],
            "full_sync_interval": canonical["sga_cycle_full_sync_interval"],
        },
        "checks": {"count_guard": canonical["sga_checks_count_guard"]},
        "shadow": canonical["sga_shadow"],
        "shadow_score": canonical["sga_shadow_score"],
        "score": _axis("score"),
        "selection": _axis("selection"),
        "optimizer": _axis("optimizer"),
        "backend": _axis("backend"),
    }
    if canonical["sga_trace_dir"] is not None:
        engine_config["trace"] = {
            "dir": canonical["sga_trace_dir"],
            "branch_points": canonical["sga_trace_branch_points"],
            "window": canonical["sga_trace_window"],
        }
    return engine_config


def build_engine(flat: dict, api_version: int):
    """Validate + assemble the engine from the flat ``sga_*`` dict the
    fork waist collected off the parsed megatron namespace (flags ARE the
    transport — there is no JSON form anymore)."""
    from sga.plugins import (
        build_backend_plugin,
        build_optimizer,
        build_selection,
        get_backend_plugin_decl,
        get_optimizer_decl,
        get_score_spec,
        get_selection_decl,
        register_default_plugins,
        validate_combination,
    )

    # Hard-coupled to megatron's get_args() by design: the fork waist is
    # this builder's only caller. The version check deliberately precedes
    # the megatron import so a mismatch raises even without megatron
    # importable.
    if api_version != SGA_API_VERSION:
        raise ValueError(
            f"SGA API version mismatch: fork waist speaks v{api_version}, "
            f"sga package speaks v{SGA_API_VERSION}"
        )

    # The waist hands the RAW namespace collection (argparse defaults
    # included); validation consumes the explicit diff.
    from megatron.training import get_args

    args = get_args()
    canonical = validate_sga_flat(
        explicit_from_flat(flat),
        train_iters=getattr(args, "train_iters", None))
    if canonical is None:
        raise ValueError(
            "build_engine called with a disabled/absent SGA configuration "
            "(the waist must key off sga_enabled)"
        )
    engine_config = materialize_engine_config(canonical)

    _assert_args_compatible()

    # Defensive re-derivation (validation already checked membership and
    # the cross-axis rules; the decls are needed for construction anyway).
    score_name = engine_config["score"]["name"]
    selection_name = engine_config["selection"]["name"]
    optimizer_name = engine_config["optimizer"]["name"]
    backend_name = engine_config["backend"]["name"]

    register_default_plugins()
    get_score_spec(score_name)  # unknown score fails with the registered list
    selection_decl = get_selection_decl(selection_name)
    optimizer_decl = get_optimizer_decl(optimizer_name)
    backend_decl = get_backend_plugin_decl(backend_name)

    # backend_emission must match the flat layer's resolution, else the two
    # validation layers contradict each other for explicit non-auto values
    validate_combination(score_name, selection_decl, optimizer_decl,
                         backend_decl,
                         backend_emission=canonical.get("sga_backend_emission"))

    selection = build_selection(selection_name, dict(engine_config["selection"]))
    # Scores are materialized by the optimizer template on full steps; inject
    # WHICH score to emit so the kernel picks the right mode (the validator
    # above already proved the template can emit it).
    opt = build_optimizer(optimizer_name, dict(engine_config["optimizer"]),
                          emit_score=score_name)
    backend = build_backend_plugin(backend_name, dict(engine_config["backend"]))

    # Optional backend-owned runtime validation, probed generically (the
    # same optional-hook discipline as adapt()/wants_force_full()): topology
    # requirements specific to a concrete backend live IN that backend
    # plugin — never as name-keyed branches in this shared builder.
    validate = getattr(backend, "validate_runtime", None)
    if callable(validate):
        from megatron.training import get_args
        validate(get_args())

    from sga.engine import SGAEngine

    engine = SGAEngine(engine_config, score_name, selection, opt, backend)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[SGA] engine built (API v{api_version}); config: "
              f"{canonical_flat_json(canonical)}", flush=True)
    return engine
