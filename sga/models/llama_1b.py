"""llama 1B dense model (day-one registry entry).

The build chain is megatron's stock GPT builder: for this config (dense,
transformer_engine impl, no MoE/hetero/MTP) the old FlagScale wrapper was
byte-for-byte the upstream path, verified by diff against
flagscale/train/megatron/gpt_builders.py @ daf4c350d. All numerics-relevant
knobs (rotary_base, swiglu, RMSNorm, untie, ...) come from the run yaml.
"""

from megatron.core.enums import ModelType

MODEL_NAME = "llama-1b"
MODEL_TYPE = ModelType.encoder_or_decoder


def builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    # Late import: gpt_builders lives at the Megatron-LM repo root, which env.sh
    # puts on PYTHONPATH alongside the sga package.
    from gpt_builders import gpt_builder

    return gpt_builder(
        args, pre_process, post_process, vp_stage=vp_stage, config=config, pg_collection=pg_collection
    )


__all__ = ["MODEL_NAME", "MODEL_TYPE", "builder"]
