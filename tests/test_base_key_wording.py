"""User-facing text names the model key as ``base``, which is a top-level config field.

The layer-streaming refusal of a quantised source (``utils/fp8_block.py``) and the
streaming docs told users to "point ``training.base``" at a bf16 copy. There is no such
key: ``base`` sits at the top of ``SoupConfig``, and a ``training: {base: ...}`` block is
refused as an unknown key. A user following the message edited the wrong place.
"""

from __future__ import annotations

import os
import re

import pytest

from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.utils.fp8_block import (
    SUPPORTED_SOURCE_FORMATS,
    UnsupportedQuantizedSourceError,
    check_source_quantization,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WRONG_KEY = re.compile(r"training\s*\.\s*base\b")
#: Text a user reads: the shipped package and the docs. Tests are not scanned.
_SCANNED_ROOTS = ("src", "docs")
_SCANNED_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".txt")


def test_base_is_a_top_level_field_of_soup_config_and_not_a_training_field():
    assert "base" in SoupConfig.model_fields
    assert "base" not in TrainingConfig.model_fields


def test_a_training_base_key_is_refused_by_the_config_loader():
    from soup_cli.config.loader import config_from_mapping

    with pytest.raises(ValueError) as excinfo:
        config_from_mapping(
            {"base": "org/model", "data": {"train": "x.jsonl"}, "training": {"base": "org/other"}}
        )
    assert "unknown config key" in str(excinfo.value)
    assert "refused" in str(excinfo.value)

    # the same mapping with `base` at the top level, where the text now sends users, loads
    config = config_from_mapping({"base": "org/model", "data": {"train": "x.jsonl"}})
    assert config.base == "org/model"


def test_the_quantised_source_refusal_points_at_base_not_training_base():
    assert "Point base at a bf16 or FP8-block copy" in SUPPORTED_SOURCE_FORMATS
    assert "training.base" not in SUPPORTED_SOURCE_FORMATS


def test_the_refusal_a_user_actually_receives_names_base():
    quant = {"quant_method": "compressed-tensors", "format": "pack-quantized"}
    with pytest.raises(UnsupportedQuantizedSourceError) as excinfo:
        check_source_quantization(quant, arch="deepseek_v3")
    message = str(excinfo.value)
    assert "Point base at a bf16 or FP8-block copy" in message
    assert "training.base" not in message


def _scanned_files():
    for top in _SCANNED_ROOTS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(_ROOT, top)):
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
            for filename in filenames:
                if filename.endswith(_SCANNED_SUFFIXES):
                    yield os.path.join(dirpath, filename)


def test_no_src_or_docs_text_says_training_base():
    files = list(_scanned_files())
    assert files, "the scanner found no src/ or docs/ files; _ROOT is wrong"
    offenders = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if _WRONG_KEY.search(line):
                    offenders.append(f"{os.path.relpath(path, _ROOT)}:{number}")
    assert not offenders, (
        "`base` is a top-level config key, not `training.base`; fix the text at: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize(
    "text",
    [
        "Point training.base at x",
        "point `training.base` at x",
        "training . base",
        "TrainingConfig: see training.base.",
    ],
)
def test_the_scanner_pattern_catches_each_spelling_of_the_wrong_key(text):
    assert _WRONG_KEY.search(text)


@pytest.mark.parametrize(
    "text", ["Point base at x", "`training.stream_layers`", "training: {lora: 1}"]
)
def test_the_scanner_pattern_leaves_correct_text_alone(text):
    assert _WRONG_KEY.search(text) is None
