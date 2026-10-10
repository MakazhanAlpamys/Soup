"""Shared tiny-model builders for tests that need a real transformers model
and tokenizer, not a mock, but not one pulled from the Hub either.

#1356 found 76 tests across 10 files downloading
``hf-internal-testing/tiny-random-LlamaForCausalLM``,
``hf-internal-testing/tiny-random-gpt2``, ``sshleifer/tiny-gpt2`` or
``HuggingFaceTB/SmolLM2-135M-Instruct`` on every run for nothing any of them
checked depends on real pretrained weights or vocabulary -- only a working
tokenizer and a save/reload round-trip. The first fix defined the same
builder in seven files and an inline version in three more; this module is
the one place that builds them now, mirroring ``tests/_windows_ci.py`` as
the existing precedent for a shared test helper pulled out after a predicate
had been copied instead (#372, #392, #424).

``tiny_model_dir(arch)`` is cached per architecture and per process, and the
directory is removed at interpreter exit rather than left in the system temp
directory the way ten separate ``tempfile.mkdtemp()`` calls were.
``build_tokenizer`` / ``build_gpt2`` / ``build_llama`` are exposed
separately for the two tests that want the in-memory objects directly
(``test_rewind_hf.py``, ``test_v05311.py``) rather than a saved directory.
"""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from transformers import GPT2LMHeadModel, LlamaForCausalLM, PreTrainedTokenizerFast

_TEXTS = (
    "hi there friend",
    "hello world",
    "the quick brown fox jumps",
    "message 0 is ham",
    "message 1 is spam",
)


def build_tokenizer(
    texts: tuple[str, ...] | None = None, *, eos: str = "<eos>"
) -> "PreTrainedTokenizerFast":
    """A byte-level BPE tokenizer trained locally (no download)."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        list(texts or _TEXTS), vocab_size=300, min_frequency=1, special_tokens=[eos]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe, eos_token=eos, bos_token=eos, pad_token=eos
    )


def build_gpt2(tok: "PreTrainedTokenizerFast") -> "GPT2LMHeadModel":
    """A tiny, randomly-initialised GPT-2 sized to ``tok``'s vocabulary."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=32, n_layer=2, n_head=4)
    return GPT2LMHeadModel(config)


def build_llama(tok: "PreTrainedTokenizerFast") -> "LlamaForCausalLM":
    """A tiny, randomly-initialised Llama sized to ``tok``'s vocabulary."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tok),
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config)


_BUILDERS = {"gpt2": build_gpt2, "llama": build_llama}


@functools.lru_cache(maxsize=None)
def tiny_model_dir(arch: str) -> str:
    """Build (once per process) and return a directory holding a tiny
    ``arch`` model + tokenizer, cleaned up at interpreter exit."""
    tok = build_tokenizer()
    model = _BUILDERS[arch](tok)
    out_dir = tempfile.mkdtemp(prefix=f"soup-test-tiny-{arch}-")
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
