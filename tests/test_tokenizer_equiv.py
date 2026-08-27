#!/usr/bin/env python3
"""Tokenizer equivalence assertion (container-only; run after `source env.sh`).

The old baseline used FlagScale's Llama3TokenizerFS shim (HF AutoTokenizer over
data/llama3_dataset, eod = eos_token_id, vocab = vocab_size + added tokens =
64002 + 305 = 64307, padded to 64320). Megatron 0.18's HuggingFaceTokenizer
reports vocab = len(tokenizer) = 64002 instead, and --vocab-size is unused on
that branch: build_tokenizer pads from the tokenizer property alone whenever
--padded-vocab-size is absent. The bridge therefore pins the model vocab
explicitly via `padded_vocab_size: 64320` in both run configs (the exact
value the old runs derived and the base5k_owt checkpoint/embedding and the
pretokenized indexed datasets were built with).

This asserts (1) eod equality with the old shim derivation and (2) that both
run configs still pin padded_vocab_size so the embedding never silently
derives from the diverging tokenizer property.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

TOKENIZER_DIR = "/workspace/sga_framework/data/llama3_dataset"
PINNED_PADDED_VOCAB = 64320  # old shim 64307 padded by 64 -> ckpt/dataset vocab
CONFIGS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "examples", "llama3", name)
    for name in ("train_1b_dense.yaml", "production_abs_m.yaml")
]


def main():
    class _Args:
        # attrs the 0.18 HuggingFaceTokenizer branch of build_tokenizer reads
        tokenizer_type = "HuggingFaceTokenizer"
        tokenizer_model = TOKENIZER_DIR
        chat_template = None
        vocab_file = None
        merge_file = None
        special_tokens = None
        tokenizer_hf_no_use_fast = False
        trust_remote_code = False
        tokenizer_hf_no_include_special_tokens = False
        metadata_path = None
        pad_vocab_size = None

    tok = build_tokenizer(_Args())

    # The old FlagScale shim (Llama3TokenizerFS) derived eod as
    # AutoTokenizer.from_pretrained(dir).eos_token_id over the same directory
    # (md5-identical files); 0.18's HuggingFaceTokenizer derives eod the same
    # way. The vocab_size *property* formulas differ numerically (len(tok) =
    # 64002 vs vocab+added = 64307), so the embedding vocab must come from the
    # explicit padded_vocab_size pin, never from the property.
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    hf_vocab = hf.vocab_size + len(hf.get_added_vocab())
    print(f"megatron tokenizer vocab/eod: {tok.vocab_size} / {tok.eod}")
    print(f"AutoTokenizer vocab/eos:      {hf_vocab} / {hf.eos_token_id}")

    assert tok.eod == hf.eos_token_id, "eod derivation diverged from the old shim"
    assert tok.vocab_size <= PINNED_PADDED_VOCAB, (
        "tokenizer property exceeds the pinned vocab — datasets would overflow"
    )
    assert hf_vocab <= PINNED_PADDED_VOCAB, (
        "old-shim vocab exceeds the pinned vocab — datasets would overflow"
    )
    assert tok.eod < PINNED_PADDED_VOCAB, (
        f"eod {tok.eod} outside pinned vocab {PINNED_PADDED_VOCAB}"
    )
    for cfg in CONFIGS:
        text = open(cfg).read()
        assert re.search(r"^\s*padded_vocab_size:\s*64320\s*$", text, re.M), (
            f"{cfg} lost the padded_vocab_size pin — the model vocab would "
            "silently derive from the tokenizer property (64064) and mismatch "
            "the 64320 checkpoint/datasets"
        )
    print("test_tokenizer_equiv: PASS")


if __name__ == "__main__":
    main()
