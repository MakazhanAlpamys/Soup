"""Kimi K2 tokenizer without Hub code.

``tests/fixtures/kimi_k2_tokenizer_reference.json`` holds token ids that Moonshot's own
``tokenization_kimi.py`` produced once (see its ``provenance`` field). The parity tests need
the real ``tiktoken.model`` (2.8 MB, not in the repo). Point ``SOUP_K2_TOKENIZER_DIR`` at a
folder holding ``tiktoken.model``, ``tokenizer_config.json`` and ``chat_template.jinja`` from
``moonshotai/Kimi-K2-Instruct`` at the fixture's revision.
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from soup_cli.utils.kimi_k2 import (
    _K2_SPLIT_PATTERN,
    has_native_kimi_k2_tokenizer,
    load_kimi_k2_tokenizer,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "kimi_k2_tokenizer_reference.json"
_REFERENCE: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_REAL_DIR = os.environ.get("SOUP_K2_TOKENIZER_DIR")


def _special(content: str, special: bool = True) -> dict[str, Any]:
    return {
        "content": content,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": special,
    }


@pytest.fixture
def tiny_folder(tmp_path: Path) -> Path:
    """A K2-shaped folder: 256 byte tokens, a few merges, specials at K2's relative ids."""
    root = tmp_path / "k2"
    root.mkdir()
    base_tokens = 300
    lines = [f"{base64.b64encode(bytes([i])).decode()} {i}" for i in range(256)]
    merges = [b"he", b"ll", b"hell", b"hello", b" w", b"or", b" wor", b"ld", b" world"]
    merges += [bytes([0xE4, 0xBD]), "你".encode()]
    merges += [f"x{i}".encode() for i in range(base_tokens - 256 - len(merges))]
    for offset, token in enumerate(merges):
        lines.append(f"{base64.b64encode(token).decode()} {256 + offset}")
    (root / "tiktoken.model").write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = base_tokens
    decoder = {
        str(first): _special("[BOS]"),
        str(first + 1): _special("[EOS]"),
        str(first + 2): _special("<|im_end|>"),
        str(first + 3): _special("<|im_user|>"),
        str(first + 5): _special("<|tool_call_begin|>", special=False),
        str(first + 254): _special("[UNK]"),
        str(first + 255): _special("[PAD]"),
    }
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": decoder,
                "additional_special_tokens": ["<|im_end|>", "<|im_user|>"],
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
            }
        ),
        encoding="utf-8",
    )
    (root / "chat_template.jinja").write_text(
        "{% for m in messages %}<|im_user|>{{ m['content'] }}<|im_end|>{% endfor %}",
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def tokenizer_dependencies():
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")


@pytest.fixture
def tiny_tokenizer(tiny_folder, tokenizer_dependencies):
    return load_kimi_k2_tokenizer(str(tiny_folder))


def test_loads_without_hub_code_or_tiktoken_package(
    tiny_folder, tokenizer_dependencies, monkeypatch
):
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # import tiktoken raises ImportError
    tok = load_kimi_k2_tokenizer(str(tiny_folder))
    assert type(tok).__module__.startswith("transformers")


def test_special_tokens_sit_at_their_configured_ids(tiny_tokenizer):
    tok = tiny_tokenizer
    assert tok.convert_tokens_to_ids("[BOS]") == 300
    assert tok.convert_tokens_to_ids("<|im_end|>") == 302
    assert tok.convert_tokens_to_ids("<|tool_call_begin|>") == 305
    assert tok.convert_tokens_to_ids("[UNK]") == 554
    assert tok.convert_tokens_to_ids("[PAD]") == 555
    assert tok.bos_token == "[BOS]" and tok.eos_token == "[EOS]"
    assert tok.pad_token == "[PAD]" and tok.unk_token == "[UNK]"
    assert len(tok) == 556


def test_unlisted_ids_are_reserved_tokens(tiny_tokenizer):
    tok = tiny_tokenizer
    assert tok.convert_ids_to_tokens(304) == "<|reserved_token_304|>"
    assert tok.encode("<|reserved_token_304|>") == [304]


def test_special_text_is_one_token_and_round_trips(tiny_tokenizer):
    tok = tiny_tokenizer
    ids = tok.encode("hello world<|im_end|>你")
    assert ids == [259, 264, 302, 266]
    assert tok.decode(ids) == "hello world<|im_end|>你"


def test_no_bos_is_added(tiny_tokenizer):
    assert tiny_tokenizer.encode("hello") == [259]


def test_chat_template_comes_from_the_jinja_file(tiny_tokenizer):
    text = tiny_tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    assert text == "<|im_user|>hi<|im_end|>"


def test_refuses_trust_remote_code(tiny_folder):
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder), trust_remote_code=True)
    assert str(error.value) == (
        "the native Kimi K2 tokenizer replaces Moonshot's tokenizer code, so "
        "--trust-remote-code has no use here. Run again without --trust-remote-code."
    )


def test_refuses_missing_vocabulary(tiny_folder):
    (tiny_folder / "tiktoken.model").unlink()
    with pytest.raises(FileNotFoundError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"tiktoken.model not found in {tiny_folder}"


@pytest.mark.parametrize("line", ["YQ== invalid\n", "%%%% 0\n"])
def test_refuses_malformed_rank_line(tiny_folder, tokenizer_dependencies, line):
    vocab_file = tiny_folder / "tiktoken.model"
    vocab_file.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{vocab_file}:1: not a 'base64-token rank' line"


@pytest.mark.parametrize("package", ["tokenizers", "transformers"])
def test_refuses_missing_tokenizer_dependency(tiny_folder, monkeypatch, package):
    if package == "transformers":
        pytest.importorskip("tokenizers")
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == (
        f"The native Kimi K2 tokenizer requires {package}. Run pip install {package}."
    )


@pytest.mark.parametrize(
    ("model_type", "has_ranks", "expected"),
    [("kimi_k2", True, True), ("kimi_k2", False, False), ("deepseek_v3", True, False)],
)
def test_native_tokenizer_requires_a_local_k2_vocabulary(
    tiny_folder, tokenizer_dependencies, model_type, has_ranks, expected
):
    (tiny_folder / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    if not has_ranks:
        (tiny_folder / "tiktoken.model").unlink()
        (tiny_folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert has_native_kimi_k2_tokenizer(str(tiny_folder)) is expected


def test_native_tokenizer_does_not_dispatch_hub_ids():
    assert not has_native_kimi_k2_tokenizer("moonshotai/Kimi-K2-Instruct")


def test_split_pattern_is_the_one_the_reference_was_made_with():
    """Runs in CI, where the real-file parity tests below skip."""
    digest = hashlib.sha256(_K2_SPLIT_PATTERN.encode()).hexdigest()
    assert digest == _REFERENCE["provenance"]["split_pattern_sha256"]


# --- real tiktoken.model against Moonshot's own ids ---

_needs_real = pytest.mark.skipif(
    not _REAL_DIR or not (Path(_REAL_DIR or "") / "tiktoken.model").is_file(),
    reason="SOUP_K2_TOKENIZER_DIR does not hold the Kimi-K2-Instruct tokenizer files",
)


@pytest.fixture(scope="module")
def real_tokenizer(tokenizer_dependencies):
    return load_kimi_k2_tokenizer(str(_REAL_DIR))


@_needs_real
def test_real_files_are_the_ones_the_reference_came_from():
    pinned = _REFERENCE["provenance"]["files_sha256"]
    for name in ("tiktoken.model", "chat_template.jinja", "tokenizer_config.json"):
        digest = hashlib.sha256((Path(_REAL_DIR) / name).read_bytes()).hexdigest()
        assert digest == pinned[name], name


@_needs_real
def test_vocab_size_and_special_ids_match(real_tokenizer):
    assert len(real_tokenizer) == _REFERENCE["vocab_size"]
    for token, index in _REFERENCE["special_ids"].items():
        assert real_tokenizer.convert_tokens_to_ids(token) == index, token


@_needs_real
@pytest.mark.parametrize("index", range(len(_REFERENCE["texts"])))
def test_text_ids_match_moonshots_tokenizer(real_tokenizer, index):
    case = _REFERENCE["texts"][index]
    assert real_tokenizer.encode(case["text"]) == case["ids"], repr(case["text"])


@_needs_real
@pytest.mark.parametrize("index", range(len(_REFERENCE["chats"])))
def test_chat_template_matches_moonshots_rendering(real_tokenizer, index):
    case = _REFERENCE["chats"][index]
    rendered = real_tokenizer.apply_chat_template(
        case["messages"], tokenize=False, add_generation_prompt=case["add_generation_prompt"]
    )
    assert rendered == case["text"]
    assert real_tokenizer.encode(rendered) == case["ids"]
