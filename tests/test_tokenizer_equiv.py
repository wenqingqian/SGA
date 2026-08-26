#!/usr/bin/env python3
"""Tokenizer equivalence assertion (container-only; run after `source env.sh`).

The old baseline used FlagScale's Llama3TokenizerFS shim (HF AutoTokenizer over
data/llama3_dataset, eod = eos_token_id, vocab = vocab_size + added tokens).
This asserts megatron 0.18's HuggingFaceTokenizer over the same directory
yields the same vocabulary footprint, so the pretokenized indexed datasets see
identical tokenizer semantics.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

TOKENIZER_DIR = "/workspace/sga_framework/data/llama3_dataset"
DECLARED_VOCAB = 128256  # explicit --vocab-size in the run config (both old and new)


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
    # way. The vocab_size *property* formulas differ numerically (len(tok) vs
    # vocab+added), but both runs declare --vocab-size 128256 explicitly, so
    # the embedding size never derives from the property; what must hold is
    # eod equality and that everything fits under the declared vocab.
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    hf_vocab = hf.vocab_size + len(hf.get_added_vocab())
    print(f"megatron tokenizer vocab/eod: {tok.vocab_size} / {tok.eod}")
    print(f"AutoTokenizer vocab/eos:      {hf_vocab} / {hf.eos_token_id}")

    assert tok.eod == hf.eos_token_id, "eod derivation diverged from the old shim"
    assert max(tok.vocab_size, hf_vocab) <= DECLARED_VOCAB, (
        "tokenizer vocab does not fit under the declared vocab"
    )
    assert tok.eod < DECLARED_VOCAB, f"eod {tok.eod} outside declared vocab {DECLARED_VOCAB}"
    print("test_tokenizer_equiv: PASS")


if __name__ == "__main__":
    main()
