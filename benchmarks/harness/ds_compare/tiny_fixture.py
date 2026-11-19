"""A tiny Llama-style model + tokenizer built locally, for CPU rehearsals (no download).

The tokenizer is a byte-level BPE trained on the word pool of ``make_rows.py`` with the four
Llama-3 special tokens added and the pinned model's chat template, so alpaca rows render
through the same template text as in the real run (the token counts differ, of course).
"""

from __future__ import annotations

from pathlib import Path

# Chat template of NousResearch/Meta-Llama-3.1-8B-Instruct at the pinned revision (the
# tokenizer_config.json "chat_template" value; sha256 of this string is recorded in
# ds_compare_rows_manifest.json as tokenizer.chat_template_sha256).
NOUS_LLAMA31_TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
)

SPECIAL_TOKENS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def build_tiny_model_dir(target: str | Path, with_model: bool = True) -> str:
    from make_rows import TOPICS, WORD_POOL
    from tokenizers import ByteLevelBPETokenizer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    corpus = [f" {word}" for word in WORD_POOL] + [f"Write a note about the {t}." for t in TOPICS]
    corpus += [f" {word.capitalize()}" for word in WORD_POOL]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=1000, min_frequency=1, special_tokens=SPECIAL_TOKENS)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        pad_token="<|eot_id|>",
    )
    tokenizer.chat_template = NOUS_LLAMA31_TEMPLATE
    tokenizer.save_pretrained(str(target))
    if with_model:
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=len(tokenizer),
            max_position_embeddings=4096,
        )
        LlamaForCausalLM(config).save_pretrained(str(target))
    return str(target)
