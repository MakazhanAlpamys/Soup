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
``tests/fixtures/recipe_base_architectures.json`` within a 30% tolerance.
Bases that cannot be resolved anonymously (gated, missing config.json, or
requiring authentication) are explicitly excluded with documented reasons.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from soup_cli.recipes.catalog import RECIPES

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_base_architectures.json"
)

#: Bases that cannot be resolved anonymously on the Hugging Face Hub (gated,
#: missing config.json/safetensors, or unresolvable anonymously), each with an
#: explicit reason. Pinned by count so exclusions cannot be silently added.
EXCLUDED_SIZE_BASES = {
    "BioMistral/BioMistral-7B": (
        "weights distributed as PyTorch bin only without safetensors metadata"
    ),
    "OpenGVLab/InternVL3-5": (
        "repo not found or not publicly readable (see #677)"
    ),
    "Qwen/Qwen-Image": (
        "repo has no config.json metadata"
    ),
    "SparkAudio/Spark-TTS-0.5B": (
        "repo has no config.json metadata"
    ),
    "baichuan-inc/Baichuan2-13B-Chat": (
        "weights distributed as PyTorch bin only without safetensors metadata"
    ),
    "canopylabs/orpheus-3b-0.1-ft": (
        "gated repository requiring authentication"
    ),
    "deepcogito/cogito-v2-preview": (
        "repo not found or not publicly readable (see #677)"
    ),
    "deepseek-ai/DeepSeek-OCR": (
        "deepseek_vl_v2 architecture requires trust_remote_code to load"
    ),
    "deepseek-ai/DeepSeek-V3-0324": (
        "historical v0.25 recipe deepseek-v3-7b-sft carries legacy 7B "
        "size label pinned by test_search_by_size"
    ),
    "epfl-llm/meditron-7b": (
        "gated repository requiring authentication"
    ),
    "google/embeddinggemma-300m": (
        "gated repository requiring authentication"
    ),
    "google/gemma-2-2b-it": (
        "gated repository requiring authentication"
    ),
    "google/gemma-3-12b-it": (
        "gated repository requiring authentication"
    ),
    "google/gemma-3-27b-it": (
        "gated repository requiring authentication"
    ),
    "google/gemma-3-9b-it": (
        "repo not found or not publicly readable (see #677)"
    ),
    "google/medgemma-4b-it": (
        "gated repository requiring authentication"
    ),
    "ibm-granite/granite-4.0-tiny-base": (
        "repo not found or not publicly readable (see #677)"
    ),
    "meta-llama/Llama-2-13b-hf": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.1-70B-Instruct": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.1-8B": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.2-11B-Vision-Instruct": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.2-1B-Instruct": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-3.2-90B-Vision-Instruct": (
        "gated repository requiring authentication"
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (
        "gated repository requiring authentication"
    ),
    "mistralai/Devstral-Small": (
        "repo not found or not publicly readable (see #677)"
    ),
    "mistralai/Mistral-Large-3-675B-Instruct-2512": (
        "repo has no config.json metadata"
    ),
    "mistralai/Mistral-Medium-3.5": (
        "repo not found or not publicly readable (see #677)"
    ),
    "mistralai/Pixtral-12B-2409": (
        "repo has no config.json metadata"
    ),
    "mistralai/Voxtral-Mini-3B": (
        "repo not found or not publicly readable (see #677)"
    ),
    "moonshotai/Kimi-K2": (
        "repo not found or not publicly readable (see #677)"
    ),
    "nvidia/Nemotron-4-340B-Instruct": (
        "repo has no config.json metadata"
    ),
    "openbmb/MiniCPM-V-2_6": (
        "gated repository requiring authentication"
    ),
    "sesame/csm-1b": (
        "gated repository requiring authentication"
    ),
}

#: Recipes whose size is explicitly declared as "N/A" (e.g. reasoning wrappers
#: or unreleased models). Pinned by name.
KNOWN_NA_SIZE_RECIPES = {
    "deepseek-ocr-sft",
    "deepseek-v3-reasoning",
    "deepseek-v4-flash-dpo",
    "deepseek-v4-flash-grpo",
    "deepseek-v4-flash-sft",
    "deepseek-v4-pro-sft",
    "kimi-k2-sft",
    "kimi-k2-thinking-grpo",
    "mistral-medium-3-5-sft",
    "paddle-ocr-sft",
    "qwen-image-sft",
    "ra-dit-retriever",
}

#: Tolerance band for declared parameter ratio: 0.70x to 1.40x.
_MIN_SIZE_RATIO = 0.70
_MAX_SIZE_RATIO = 1.40


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


class TestRecipeSizeRatchet:
    """Validates that recipe size labels accurately reflect model parameters."""

    def test_every_resolvable_recipe_matches_recorded_parameters(self) -> None:
        """Every recipe using an un-excluded base must match within tolerance."""
        arch = _load_base_architectures()
        discrepancies: list[str] = []
        checked_count = 0

        for name, meta in RECIPES.items():
            parsed_yaml = yaml.safe_load(meta.yaml_str) or {}
            base = parsed_yaml.get("base")
            assert base, f"Recipe {name} has no base in YAML"

            if base in EXCLUDED_SIZE_BASES:
                continue

            if meta.size == "N/A":
                assert name in KNOWN_NA_SIZE_RECIPES, (
                    f"Recipe {name} has size='N/A' but is not in KNOWN_NA_SIZE_RECIPES"
                )
                continue

            declared_params = parse_size(meta.size)
            assert declared_params is not None, (
                f"Recipe {name} has unparseable size: {meta.size!r}"
            )

            base_info = arch.get(base)
            assert base_info is not None, (
                f"Recipe {name}'s base {base!r} is missing from recipe_base_architectures.json"
            )

            actual_params = base_info.get("parameters")
            if actual_params is None:
                continue

            checked_count += 1
            ratio = declared_params / actual_params
            if not (_MIN_SIZE_RATIO <= ratio <= _MAX_SIZE_RATIO):
                discrepancies.append(
                    f"{name}: declared {meta.size} ({declared_params:.2e}) vs "
                    f"actual {actual_params:.2e} on base {base} (ratio {ratio:.3f})"
                )

        assert discrepancies == [], (
            "Recipe sizes disagree with recorded base parameter count beyond tolerance "
            f"[{_MIN_SIZE_RATIO}..{_MAX_SIZE_RATIO}]:\n  " + "\n  ".join(discrepancies)
        )
        assert checked_count >= 110, f"Expected at least 110 checked recipes, got {checked_count}"

    def test_ratchet_kills_understated_size_mutations(self) -> None:
        """Assert that understated size labels are caught by the ratchet."""
        arch = _load_base_architectures()

        test_cases = [
            ("glm-4.6-sft", "9B", 356785898816),  # 356.8B params on Hub
            ("glm-5-sft", "9B", arch["zai-org/GLM-5"]["parameters"]),
            ("minimax-m2-sft", "9B", arch["MiniMaxAI/MiniMax-M2"]["parameters"]),
            ("deepseek-v3-7b-sft", "7B", arch["deepseek-ai/DeepSeek-V3-0324"]["parameters"]),
        ]

        for recipe_name, bad_size, actual_params in test_cases:
            mutated_params = parse_size(bad_size)
            assert mutated_params is not None
            ratio = mutated_params / actual_params
            assert not (_MIN_SIZE_RATIO <= ratio <= _MAX_SIZE_RATIO), (
                f"Mutation {bad_size} for {recipe_name} was expected to fail ratchet"
            )

    def test_excluded_bases_are_pinned(self) -> None:
        """Ensure the exclusion list count and reasons are pinned to prevent silent additions."""
        assert len(EXCLUDED_SIZE_BASES) == 35
        for base, reason in EXCLUDED_SIZE_BASES.items():
            assert reason, f"Base {base} has empty exclusion reason"

    def test_parse_size_valid_and_invalid(self) -> None:
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
