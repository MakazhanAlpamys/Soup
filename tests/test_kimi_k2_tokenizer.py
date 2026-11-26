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


@pytest.mark.parametrize(
    "line",
    ["YQ== invalid\n", "%%%% 0\n", "YQ== +0\n", "YQ== 0_0\n"],
    ids=["non-integer-rank", "invalid-base64", "signed-positive-rank", "underscore-in-rank"],
)
def test_refuses_malformed_rank_line(tiny_folder, line):
    vocab_file = tiny_folder / "tiktoken.model"
    vocab_file.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{vocab_file}:1: not a 'base64-token rank' line"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (b"YQ== 0\nYQ== 1\n", "2: duplicate token (first seen on line 1)"),
        (b"YQ== 0\nYg== 0\n", "2: duplicate rank 0 (first seen on line 1)"),
        (b"YQ== -1\n", "1: negative rank -1"),
        (b"YQ== 0\nYg== 2\n", "2: rank 2 is outside the contiguous range 0..1"),
        (b"", "1: no token ranks found"),
    ],
    ids=["duplicate-token", "duplicate-rank", "negative-rank", "gap", "empty"],
)
def test_refuses_invalid_rank_table(tiny_folder, lines, message):
    vocab_file = tiny_folder / "tiktoken.model"
    vocab_file.write_bytes(lines)
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{vocab_file}:{message}"


@pytest.mark.parametrize(
    ("filename", "limit", "message"),
    [
        ("tiktoken.model", 64, ":1: file exceeds the 64 MiB size limit"),
        ("tokenizer_config.json", 16, ": file exceeds the 16 MiB size limit"),
    ],
)
def test_checks_tokenizer_file_size_before_opening(
    tiny_folder, monkeypatch, filename, limit, message,
):
    path = tiny_folder / filename
    getsize = os.path.getsize
    path_open = Path.open

    def reported_size(candidate):
        return limit * 1024**2 + 1 if Path(candidate) == path else getsize(candidate)

    def guarded_open(candidate, *args, **kwargs):
        assert candidate != path, f"{path} was opened before its size was checked"
        return path_open(candidate, *args, **kwargs)

    monkeypatch.setattr(os.path, "getsize", reported_size)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{path}{message}"


def test_checks_chat_template_size_before_opening(tiny_folder, monkeypatch):
    path = tiny_folder / "chat_template.jinja"
    getsize = os.path.getsize
    path_open = Path.open

    def reported_size(candidate):
        return 16 * 1024**2 + 1 if Path(candidate) == path else getsize(candidate)

    def guarded_open(candidate, *args, **kwargs):
        assert candidate != path, f"{path} was opened before its size was checked"
        return path_open(candidate, *args, **kwargs)

    monkeypatch.setattr(os.path, "getsize", reported_size)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{path}: file exceeds the 16 MiB size limit"


def test_refuses_chat_template_longer_than_limit_when_size_lies(tiny_folder, monkeypatch):
    path = tiny_folder / "chat_template.jinja"
    path.write_bytes(b"a" * (16 * 1024**2 + 1))
    getsize = os.path.getsize
    monkeypatch.setattr(os.path, "getsize", lambda c: 1 if Path(c) == path else getsize(c))
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{path}: file exceeds the 16 MiB size limit"


def test_refuses_chat_template_that_is_not_utf8(tiny_folder):
    path = tiny_folder / "chat_template.jinja"
    path.write_bytes(b"\xff\xfe{{ messages }}")
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{path}: not valid UTF-8"


def test_refuses_file_swapped_after_the_containment_check(tiny_folder, monkeypatch):
    path = tiny_folder / "chat_template.jinja"
    other = tiny_folder / "other.jinja"
    other.write_text("swapped", encoding="utf-8")
    real_stat = os.stat
    resolved = Path(os.path.realpath(path))

    def stat_of_another_file(candidate, *args, **kwargs):
        if Path(candidate) == resolved:
            return real_stat(other, *args, **kwargs)
        return real_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_of_another_file)
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{path}: changed while it was being opened"


def test_refuses_duplicate_normalized_special_token_ids(tiny_folder):
    config_file = tiny_folder / "tokenizer_config.json"
    entry = _special("[BOS]")
    config_file.write_text(
        json.dumps({"added_tokens_decoder": {"300": entry, "0300": entry}}), encoding="utf-8"
    )
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == (
        f"{config_file}: duplicate special-token id 300 (keys '300' and '0300')"
    )


@pytest.mark.parametrize("special", ["true", 1, None])
def test_refuses_non_boolean_special_flag(tiny_folder, special):
    config_file = tiny_folder / "tokenizer_config.json"
    entry = {"content": "[BOS]", "special": special}
    config_file.write_text(json.dumps({"added_tokens_decoder": {"300": entry}}), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{config_file}: special-token entry 300 special must be a boolean"


def test_refuses_same_content_declared_for_two_ids(tiny_folder, tokenizer_dependencies):
    config_file = tiny_folder / "tokenizer_config.json"
    decoder = {"300": _special("[BOS]"), "301": _special("[BOS]")}
    config_file.write_text(json.dumps({"added_tokens_decoder": decoder}), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == (
        f"{config_file}: token content '[BOS]' is declared for both ids 300 and 301"
    )


def test_refuses_token_that_did_not_receive_its_declared_id(
    tiny_folder, tokenizer_dependencies, monkeypatch
):
    from transformers.convert_slow_tokenizer import TikTokenConverter

    real_converted = TikTokenConverter.converted

    class _DropsAddedTokens:
        def __init__(self, inner):
            self._inner = inner

        def add_tokens(self, tokens):
            return 0

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        TikTokenConverter, "converted", lambda self: _DropsAddedTokens(real_converted(self))
    )
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == (
        f"{tiny_folder / 'tokenizer_config.json'}: special token '[BOS]' received id None, "
        "declared id 300"
    )


def test_refuses_more_than_one_million_vocab_lines(tiny_folder):
    vocab_file = tiny_folder / "tiktoken.model"
    vocab_file.write_bytes(b"YQ== 0\n" + b"\n" * 1_000_000)
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{vocab_file}:1000001: file exceeds the 1000000-line limit"


def test_refuses_missing_tokenizer_config(tiny_folder):
    config_file = tiny_folder / "tokenizer_config.json"
    config_file.unlink()
    with pytest.raises(FileNotFoundError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{config_file}: file not found"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{", "invalid JSON"),
        ("[]", "expected a JSON object"),
        ('{"added_tokens_decoder": []}', "added_tokens_decoder must be an object"),
        (
            '{"added_tokens_decoder": {"+300": {}}}',
            "special-token id '+300' is not a decimal integer",
        ),
        ('{"added_tokens_decoder": {"300": []}}', "special-token entry 300 must be an object"),
        (
            '{"added_tokens_decoder": {"300": {}}}',
            "special-token entry 300 must have string content",
        ),
        (
            '{"added_tokens_decoder": {"300": {"content": 300}}}',
            "special-token entry 300 must have string content",
        ),
        (
            '{"added_tokens_decoder": {"299": {"content": "[BOS]", "special": true}}}',
            "special-token id 299 is outside the reserved range [300, 556)",
        ),
        (
            '{"added_tokens_decoder": {"556": {"content": "[BOS]", "special": true}}}',
            "special-token id 556 is outside the reserved range [300, 556)",
        ),
    ],
    ids=[
        "invalid-json", "root-not-object", "decoder-not-object", "key-not-decimal",
        "entry-not-object", "missing-content", "non-string-content", "id-below", "id-above",
    ],
)
def test_refuses_invalid_tokenizer_config(tiny_folder, contents, message):
    config_file = tiny_folder / "tokenizer_config.json"
    config_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{config_file}: {message}"


@pytest.mark.requires_symlink
@pytest.mark.parametrize(
    "filename", ["tiktoken.model", "tokenizer_config.json", "chat_template.jinja"],
)
def test_refuses_tokenizer_file_link_outside_model_folder(tiny_folder, filename):
    path = tiny_folder / filename
    outside = path.replace(tiny_folder.parent / filename)
    os.symlink(outside, path)
    with pytest.raises(ValueError) as error:
        load_kimi_k2_tokenizer(str(tiny_folder))
    assert str(error.value) == f"{path}: resolves outside the model folder"


def test_expands_home_for_native_tokenizer(tiny_folder, tokenizer_dependencies, monkeypatch):
    (tiny_folder / "config.json").write_text('{"model_type": "kimi_k2"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tiny_folder.parent))
    monkeypatch.setenv("USERPROFILE", str(tiny_folder.parent))
    assert has_native_kimi_k2_tokenizer("~/k2")
    assert load_kimi_k2_tokenizer("~/k2").encode("hello") == [259]


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
