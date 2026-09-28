# Copyright 2026 Soup Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recipe declared size validation and base parameter count ratchet (#1161).

Every recipe declared in ``src/soup_cli/recipes/catalog.py`` carries a user-facing
``size`` attribute (printed by ``soup recipes`` and used to assess GPU memory fit).
For sparse MoE models, ``size`` represents total parameters (consistent with
MiniMax M3 at 428B, DeepSeek V3 at 671B, GLM-5.1 at 754B, and Qwen3.5-397B at 397B).

This suite asserts that every recipe with an anonymously resolvable base model
has a declared ``size`` that matches the parameter count recorded in
``tests/fixtures/recipe_base_architectures.json`` within a tolerance band
[0.70x .. 1.15x].
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
import yaml

from soup_cli.recipes.catalog import RECIPES

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_base_architectures.json"
)

#: Bases that have no safetensors metadata on the Hugging Face Hub.
EXCLUDED_SIZE_BASES = {
    "BioMistral/BioMistral-7B",
    "SparkAudio/Spark-TTS-0.5B",
    "baichuan-inc/Baichuan2-13B-Chat",
    "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "mistralai/Pixtral-12B-2409",
    "nvidia/Nemotron-4-340B-Instruct",
}

#: Recipes whose size is explicitly declared as "N/A" (e.g. reasoning wrappers
#: or unreleased models). Pinned by name.
KNOWN_NA_SIZE_RECIPES = {
    "deepseek-ocr-sft",
    "deepseek-v3-reasoning",
    "deepseek-v4-flash-dpo",
    "deepseek-v4-flash-grpo",
    "deepseek-v4-flash-sft",
    "deepseek-v4-pro-dpo",
    "deepseek-v4-pro-grpo",
    "deepseek-v4-pro-sft",
    "kimi-k2-sft",
    "kimi-k2-thinking-grpo",
    "paddle-ocr-sft",
    "qwen-image-sft",
    "ra-dit-retriever",
}

#: Tolerance band for declared parameter ratio: 0.70x to 1.15x.
_MIN_SIZE_RATIO = 0.70
_MAX_SIZE_RATIO = 1.15


def parse_size(size_str: str) -> float | None:
    """Parse human-readable size string (e.g., '8B', '300M', '1T') to parameter count."""
    if not size_str or size_str == "N/A":
        return None
    match = re.match(r"^([\d\.]+)\s*([MBT])$", size_str.strip())
    if not match:
        return None
    val, unit = float(match.group(1)), match.group(2)
    multiplier = {"M": 1e6, "B": 1e9, "T": 1e12}[unit]
    return val * multiplier


def _load_base_architectures() -> dict[str, dict]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def size_problems(recipes, arch):
    """Every recipe whose label disagrees with its base, and how many were compared."""
    problems, checked = [], 0
    for name, meta in recipes.items():
        base = (yaml.safe_load(meta.yaml_str) or {}).get("base")
        if base in EXCLUDED_SIZE_BASES:
            continue
        if meta.size == "N/A":
            if name not in KNOWN_NA_SIZE_RECIPES:
                problems.append(f"{name}: size 'N/A' is not declared in KNOWN_NA_SIZE_RECIPES")
            continue
        declared = parse_size(meta.size)
        if declared is None:
            problems.append(f"{name}: size {meta.size!r} is not a total parameter count like '35B'")
            continue
        actual = arch.get(base, {}).get("parameters")
        if actual is None:
            problems.append(
                f"{name}: base {base} has no recorded parameter count and is not excluded"
            )
            continue
        checked += 1
        ratio = declared / actual
        if not _MIN_SIZE_RATIO <= ratio <= _MAX_SIZE_RATIO:
            problems.append(f"{name}: declared {meta.size} vs {actual:,} on {base} ({ratio:.3f}x)")
    return problems, checked


def test_every_recipe_label_matches_its_base():
    problems, checked = size_problems(RECIPES, _load_base_architectures())
    assert problems == [], "\n".join(problems)
    assert checked >= 150, checked


@pytest.mark.parametrize(("name", "old_label"), [
    ("glm-4.6-sft", "9B"), ("glm-5-sft", "9B"), ("minimax-m2-sft", "9B"),
    ("deepseek-v3-7b-sft", "7B"), ("voxtral-sft", "3B"), ("llama4-scout-17b-sft", "17B"),
    ("qwen2.5-7b-sft", "10B"),  # overstated: the upper bound has to bite too
])
def test_the_real_check_rejects_the_old_label(name, old_label):
    mutated = {**RECIPES, name: dataclasses.replace(RECIPES[name], size=old_label)}
    problems, _ = size_problems(mutated, _load_base_architectures())
    assert any(p.startswith(f"{name}:") for p in problems), problems


def test_a_missing_count_fails_naming_the_recipe():
    arch = _load_base_architectures()
    arch["zai-org/GLM-4.6"].pop("parameters")
    problems, _ = size_problems(RECIPES, arch)
    assert any(
        p.startswith("glm-4.6-sft:") and "no recorded parameter count" in p for p in problems
    ), problems


def test_every_exclusion_is_live_and_has_no_count():
    arch = _load_base_architectures()
    used = {(yaml.safe_load(m.yaml_str) or {}).get("base") for m in RECIPES.values()}
    assert sorted(set(EXCLUDED_SIZE_BASES) - used) == []
    assert sorted(b for b in EXCLUDED_SIZE_BASES if "parameters" in arch.get(b, {})) == []


def test_every_known_na_entry_is_live():
    stale = [n for n in KNOWN_NA_SIZE_RECIPES if n not in RECIPES or RECIPES[n].size != "N/A"]
    assert sorted(stale) == []


def test_the_band_is_pinned():
    assert (_MIN_SIZE_RATIO, _MAX_SIZE_RATIO) == (0.70, 1.15)


def test_parse_size_valid_and_invalid():
    """Coverage for size parsing helper across supported units and invalid values."""
    assert parse_size("357B") == 357e9
    assert parse_size("754B") == 754e9
    assert parse_size("1.5B") == 1.5e9
    assert parse_size("300M") == 300e6
    assert parse_size("1T") == 1e12
    assert parse_size("N/A") is None
    assert parse_size("") is None
    assert parse_size("invalid") is None
    assert parse_size("12KB") is None
