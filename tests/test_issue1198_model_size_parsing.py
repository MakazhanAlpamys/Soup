"""Tests for Issue #1198: Hardware-fit gate Hub model size estimation.

Validates that model_size_from_name correctly parses model sizes from Hub IDs
using boundary-safe regex, handles MoE (NxMb) models, respects total parameters
over active-parameter markers (-aNNb), and resolves recipe bases in catalog.py
within 10% of their actual parameter count or explicitly allowlists them.
"""

from __future__ import annotations

import re

import pytest

from soup_cli.recipes.catalog import RECIPES
from soup_cli.utils.gpu import model_size_from_name

# Hub base models in the recipe catalogue whose repository IDs carry no parameter
# count marker (such as "14B" or "8x7B") and therefore fall back to the default 7.0B
# guess, or whose size is not specified in the recipe metadata ("N/A").
_UNRESOLVABLE_CATALOG_REASONS = {
    "MiniMaxAI/MiniMax-M2": "Model ID specifies version 'M2', carrying no parameter count (9B).",
    "MiniMaxAI/MiniMax-M3": "Model ID specifies version 'M3', carrying no parameter count (428B).",
    "OpenGVLab/InternVL3-5": "Model ID specifies series '3-5' without parameter count (8B).",
    "PaddlePaddle/PaddleOCR-VL": "Specialized OCR VL model with no parameter count (size N/A).",
    "Qwen/Qwen-Image": "Qwen image foundation model with no parameter count (size N/A).",
    "THUDM/glm-4.6": "Model ID specifies release '4.6' without parameter count (9B).",
    "deepcogito/cogito-v2-preview": "Preview release with no parameter count in ID (14B).",
    "deepseek-ai/DeepSeek-OCR": "Domain OCR model with no parameter count (size N/A).",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3 671B MoE carrying no parameter count in repo ID.",
    "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek V4 Flash architecture with no parameter count.",
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek V4 Pro architecture with no parameter count.",
    "facebook/seamless-m4t-v2-large": "Uses 'v2-large' designation rather than param count (2.3B).",
    "ibm-granite/granite-4.0-tiny-base": "Uses 'tiny-base' descriptor without param count (3B).",
    "microsoft/Phi-3.5-mini-instruct": "Uses 'mini' descriptor rather than param count (3.8B).",
    "microsoft/phi-4": "Model ID specifies release 'phi-4' without parameter count (14B).",
    "mistralai/Devstral-Small": "Uses 'Small' descriptor rather than param count (24B).",
    "mistralai/Magistral-Small": "Uses 'Small' descriptor rather than param count (24B).",
    "mistralai/Mistral-Medium-3.5": "Uses 'Medium-3.5' descriptor without param count (size N/A).",
    "moonshotai/Kimi-K2": "Model ID specifies version 'K2' without parameter count (size N/A).",
    "moonshotai/Kimi-K2-Thinking": "Reasoning variant of K2 with no parameter count (size N/A).",
    "moonshotai/Kimi-K2.5": "Model ID specifies version 'K2.5' without parameter count (1T).",
    "moonshotai/Kimi-K2.6": "Model ID specifies version 'K2.6' without parameter count (1T).",
    "openbmb/MiniCPM-V-2_6": "Model ID specifies version '2_6' without parameter count (8B).",
    "sentence-transformers/all-mpnet-base-v2": "Embedding model with no parameter count.",
    "zai-org/GLM-5": "Model ID specifies release '5' without parameter count (9B).",
    "zai-org/GLM-5.1": "Model ID specifies release '5.1' without parameter count (754B).",
}


class TestIssue1198HubModelSizeParsing:
    """Acceptance criteria tests for Issue #1198."""

    @pytest.mark.parametrize(
        ("model_id", "expected_size"),
        [
            ("Qwen/Qwen2.5-72B-Instruct", 72.0),
            ("Qwen/Qwen2.5-32B", 32.0),
            ("Qwen/Qwen2.5-14B", 14.0),
            ("meta-llama/Llama-3.1-405B", 405.0),
            ("Qwen/Qwen3-235B-A22B", 235.0),
            ("mistralai/Mixtral-8x7B-v0.1", 46.7),
            ("mistralai/Mixtral-8x22B-v0.1", 141.0),
            ("Qwen/Qwen3-4B", 4.0),
        ],
    )
    def test_issue_reported_models(self, model_id: str, expected_size: float) -> None:
        """Verify the exact models reported in issue #1198 resolve accurately."""
        assert model_size_from_name(model_id) == pytest.approx(expected_size)

    @pytest.mark.parametrize(
        ("model_id", "expected_size"),
        [
            # Models > 8B must never resolve to <= 8.0
            ("meta-llama/Llama-3.1-70B", 70.0),
            ("meta-llama/Llama-3.2-90B-Vision-Instruct", 90.0),
            ("meta-llama/Llama-3.1-405B", 405.0),
            ("Qwen/Qwen2.5-72B-Instruct", 72.0),
            ("Qwen/Qwen2.5-32B", 32.0),
            ("Qwen/Qwen2.5-Coder-14B-Instruct", 14.0),
            ("meta-llama/Llama-3.2-11B-Vision-Instruct", 11.0),
            ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 17.0),
            ("mistralai/Pixtral-12B-2409", 12.0),
            ("mistralai/Mistral-Small-24B-Instruct-2501", 24.0),
            ("Qwen/Qwen3.5-27B", 27.0),
            ("Qwen/Qwen3-Coder-30B-A3B-Instruct", 30.0),
            ("Qwen/Qwen3.5-35B-A3B", 35.0),
            ("Qwen/Qwen3.5-122B-A10B", 122.0),
            ("Qwen/Qwen3.5-397B-A17B", 397.0),
            ("nvidia/Nemotron-4-340B-Instruct", 340.0),
            ("mistralai/Mistral-Large-3-675B-Instruct-2512", 675.0),
            ("openai/gpt-oss-20b", 20.0),
            ("openai/gpt-oss-120b", 120.0),
            ("mistralai/Mixtral-8x7B-v0.1", 46.7),
            ("mistralai/Mixtral-8x22B-v0.1", 141.0),
        ],
    )
    def test_large_models_never_resolve_to_eight_or_less(
        self, model_id: str, expected_size: float
    ) -> None:
        """Acceptance Criterion 2: No Hub id of a model larger than 8B resolves to 8 or less."""
        resolved = model_size_from_name(model_id)
        assert resolved > 8.0
        assert resolved == pytest.approx(expected_size)

    def test_active_parameter_variations(self) -> None:
        """Active parameter tokens must never mask the total parameter count."""
        assert model_size_from_name("Qwen/Qwen3-235B-A22B") == 235.0
        assert model_size_from_name("Qwen/Qwen3-235B-a22b") == 235.0
        assert model_size_from_name("Qwen/Qwen3-235B-act22b") == 235.0
        assert model_size_from_name("Qwen/Qwen3-235B-active22b") == 235.0
        assert model_size_from_name("Qwen/Qwen3-a-22b-235b") == 235.0
        assert model_size_from_name("Qwen/Qwen3-act-22b-235b") == 235.0
        assert model_size_from_name("Qwen/Qwen3-Coder-30B-A3B-Instruct") == 30.0
        assert model_size_from_name("Qwen/Qwen3.5-122B-A10B") == 122.0
        assert model_size_from_name("Qwen/Qwen3.5-397B-A17B") == 397.0

    def test_boundary_lookaround_prevents_false_matches(self) -> None:
        """Boundary lookarounds prevent alphanumeric prefixes (e.g. 'v3b' beta) from matching."""
        assert model_size_from_name("org/model-v3b-7b") == 7.0

    def test_generic_moe_fallback(self) -> None:
        """Generic NxMb pattern computes expert product as upper bound for hardware fit."""
        assert model_size_from_name("custom-org/moe-4x7b-base") == 28.0
        assert model_size_from_name("custom-org/moe-16x12b-instruct") == 192.0

    def test_decimal_size_tokens(self) -> None:
        """Decimals are parsed correctly without substring collisions."""
        assert model_size_from_name("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
        assert model_size_from_name("Qwen/Qwen3.5-0.8B") == 0.8
        assert model_size_from_name("Qwen/Qwen2.5-1.5B-Instruct") == 1.5
        assert model_size_from_name("HuggingFaceTB/SmolLM2-1.7B-Instruct") == 1.7
        assert model_size_from_name("LiquidAI/LFM2-1.2B") == 1.2
        assert model_size_from_name("ise-uiuc/Magicoder-S-DS-6.7B") == 6.7

    def test_recipe_catalog_resolution_or_allowlist(self) -> None:
        """Acceptance Criterion 1: Every recipe base resolves within 10% or is documented."""
        seen_models = set()
        for name, recipe in RECIPES.items():
            model = recipe.model
            if model in seen_models:
                continue
            seen_models.add(model)

            parsed = model_size_from_name(model)

            # Match recipe size like "8B", "135M", "0.5B", etc.
            size_match = re.match(r"^(\d+(?:\.\d+)?)([BM])$", recipe.size)
            if size_match:
                val = float(size_match.group(1))
                unit = size_match.group(2)
                expected_b = val if unit == "B" else val / 1000.0
                rel_diff = abs(parsed - expected_b) / expected_b

                if rel_diff > 0.10:
                    assert model in _UNRESOLVABLE_CATALOG_REASONS, (
                        f"Recipe '{name}' with model '{model}' resolved to {parsed}B, "
                        f"deviating from recipe size {recipe.size} ({expected_b}B) "
                        f"by {rel_diff:.1%}, but not in _UNRESOLVABLE_CATALOG_REASONS."
                    )
            else:
                # Recipe size is "N/A" or non-standard
                assert model in _UNRESOLVABLE_CATALOG_REASONS, (
                    f"Recipe '{name}' with model '{model}' has non-standard size '{recipe.size}', "
                    "but is not documented in _UNRESOLVABLE_CATALOG_REASONS."
                )
