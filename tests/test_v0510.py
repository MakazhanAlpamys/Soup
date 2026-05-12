"""v0.51.0 — Model Catalog Expansion + Alternative Model Hubs.

Covers Parts A/B/C (25 new recipes), Part D (MULTIPACK_ARCHITECTURES extension),
Part E (hub adapters + ``hub`` field on TrainingConfig).
"""
from __future__ import annotations

from types import MappingProxyType

import pytest
import yaml
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TEMPLATES, SoupConfig, TrainingConfig
from soup_cli.recipes.catalog import RECIPES, get_recipe, list_recipes, search_recipes
from soup_cli.utils import hubs as hubs_mod
from soup_cli.utils.hubs import (
    SUPPORTED_HUBS,
    default_endpoint,
    endpoint_env_var,
    is_hf,
    required_hub_package,
    resolve_endpoint,
    validate_hub_endpoint,
    validate_hub_name,
)
from soup_cli.utils.multipack_sampler import (
    MULTIPACK_ARCHITECTURES,
    validate_multipack_architecture,
)

# =====================================================================
# Part E — hubs.validate_hub_name
# =====================================================================


class TestValidateHubName:
    @pytest.mark.parametrize("name", ["hf", "modelscope", "modelers"])
    def test_known_accepted(self, name: str) -> None:
        assert validate_hub_name(name) == name

    @pytest.mark.parametrize("name", ["HF", "ModelScope", "MODELERS"])
    def test_case_insensitive(self, name: str) -> None:
        assert validate_hub_name(name) == name.lower()

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            validate_hub_name("github")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_hub_name("")

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="null bytes"):
            validate_hub_name("hf\x00")

    def test_oversize_rejected(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            validate_hub_name("a" * 33)

    def test_non_string_rejected(self) -> None:
        with pytest.raises(TypeError):
            validate_hub_name(123)  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            validate_hub_name(True)  # type: ignore[arg-type]

    def test_supported_hubs_frozen(self) -> None:
        with pytest.raises(AttributeError):
            SUPPORTED_HUBS.add("evil")  # type: ignore[attr-defined]

    def test_supported_hubs_count(self) -> None:
        assert SUPPORTED_HUBS == frozenset({"hf", "modelscope", "modelers"})


# =====================================================================
# Part E — required_hub_package / default_endpoint / endpoint_env_var
# =====================================================================


class TestHubMetadata:
    @pytest.mark.parametrize(
        "hub,pkg",
        [("hf", "huggingface-hub"), ("modelscope", "modelscope"),
         ("modelers", "openmind-hub")],
    )
    def test_required_package(self, hub: str, pkg: str) -> None:
        assert required_hub_package(hub) == pkg

    def test_required_package_unknown(self) -> None:
        assert required_hub_package("github") is None

    def test_required_package_non_string(self) -> None:
        assert required_hub_package(123) is None  # type: ignore[arg-type]

    def test_required_package_case_insensitive(self) -> None:
        assert required_hub_package("HF") == "huggingface-hub"

    def test_default_endpoint_hf_https(self) -> None:
        assert default_endpoint("hf").startswith("https://")

    def test_default_endpoint_modelscope_https(self) -> None:
        assert default_endpoint("modelscope").startswith("https://")

    def test_default_endpoint_modelers_https(self) -> None:
        assert default_endpoint("modelers").startswith("https://")

    def test_default_endpoint_unknown_rejected(self) -> None:
        with pytest.raises(ValueError):
            default_endpoint("github")

    def test_default_endpoint_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            default_endpoint(True)  # type: ignore[arg-type]

    def test_endpoint_env_var_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            endpoint_env_var(True)  # type: ignore[arg-type]

    def test_endpoint_env_var_hf(self) -> None:
        assert endpoint_env_var("hf") == "HF_ENDPOINT"

    def test_endpoint_env_var_modelscope(self) -> None:
        assert endpoint_env_var("modelscope") == "MODELSCOPE_ENDPOINT"

    def test_endpoint_env_var_modelers(self) -> None:
        assert endpoint_env_var("modelers") == "MODELERS_ENDPOINT"

    def test_endpoint_env_var_unknown_rejected(self) -> None:
        with pytest.raises(ValueError):
            endpoint_env_var("github")


# =====================================================================
# Part E — validate_hub_endpoint (SSRF policy)
# =====================================================================


class TestValidateHubEndpoint:
    def test_https_remote_ok(self) -> None:
        assert validate_hub_endpoint("https://example.com") == "https://example.com"

    def test_strips_trailing_slash(self) -> None:
        assert validate_hub_endpoint("https://example.com/") == "https://example.com"

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
    def test_http_loopback_ok(self, host: str) -> None:
        url = f"http://{host}:8080"
        assert validate_hub_endpoint(url) == url

    def test_http_remote_rejected(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            validate_hub_endpoint("http://example.com")

    def test_http_private_ip_rejected(self) -> None:
        with pytest.raises(ValueError, match="loopback"):
            validate_hub_endpoint("http://192.168.1.1")

    def test_http_link_local_rejected(self) -> None:
        # AWS metadata endpoint
        with pytest.raises(ValueError, match="loopback"):
            validate_hub_endpoint("http://169.254.169.254")

    def test_zero_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="0.0.0.0"):
            validate_hub_endpoint("http://0.0.0.0")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            validate_hub_endpoint("ftp://example.com")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            validate_hub_endpoint("file:///etc/passwd")

    def test_no_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            validate_hub_endpoint("example.com")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_hub_endpoint("")

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="null bytes"):
            validate_hub_endpoint("https://example.com\x00")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(TypeError):
            validate_hub_endpoint(123)  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            validate_hub_endpoint(True)  # type: ignore[arg-type]

    def test_label_in_error_message(self) -> None:
        with pytest.raises(ValueError, match="modelscope"):
            validate_hub_endpoint("ftp://example.com", hub="modelscope")

    def test_missing_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="host"):
            validate_hub_endpoint("https://")

    def test_control_chars_rejected(self) -> None:
        # CRLF injection defence (review fix).
        with pytest.raises(ValueError, match="control characters"):
            validate_hub_endpoint("https://example.com\r\nX-Evil: 1")

    def test_http_ipv6_mapped_private_rejected(self) -> None:
        with pytest.raises(ValueError, match="loopback"):
            validate_hub_endpoint("http://[::ffff:192.168.1.1]")

    def test_http_ipv6_loopback_ok(self) -> None:
        # `::1` is in `_LOOPBACK_HOSTS` so HTTP is permitted (parity with v0.29.0).
        assert (
            validate_hub_endpoint("http://[::1]:8080")
            == "http://[::1]:8080"
        )


# =====================================================================
# Part E — resolve_endpoint (env-var override)
# =====================================================================


class TestResolveEndpoint:
    def test_default_when_env_unset(self) -> None:
        out = resolve_endpoint("hf", env={})
        assert out.startswith("https://huggingface.co")

    def test_modelscope_default_when_env_unset(self) -> None:
        out = resolve_endpoint("modelscope", env={})
        assert "modelscope" in out

    def test_modelers_default_when_env_unset(self) -> None:
        out = resolve_endpoint("modelers", env={})
        assert "modelers" in out

    def test_env_override_validated(self) -> None:
        out = resolve_endpoint(
            "modelscope",
            env={"MODELSCOPE_ENDPOINT": "https://mirror.example.com/"},
        )
        assert out == "https://mirror.example.com"

    def test_env_override_rejects_http_remote(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            resolve_endpoint(
                "modelers",
                env={"MODELERS_ENDPOINT": "http://example.com"},
            )

    def test_env_override_loopback_http_ok(self) -> None:
        out = resolve_endpoint(
            "hf", env={"HF_ENDPOINT": "http://localhost:9000"}
        )
        assert out == "http://localhost:9000"

    def test_env_override_null_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="null bytes"):
            resolve_endpoint(
                "hf", env={"HF_ENDPOINT": "https://x\x00.com"}
            )

    def test_unknown_hub_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            resolve_endpoint("github", env={})

    def test_empty_env_value_falls_back_to_default(self) -> None:
        out = resolve_endpoint("hf", env={"HF_ENDPOINT": ""})
        assert out.startswith("https://huggingface.co")

    def test_default_env_uses_os_environ_when_none(self, monkeypatch) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        out = resolve_endpoint("hf")
        assert out.startswith("https://huggingface.co")


# =====================================================================
# Part E — is_hf convenience
# =====================================================================


class TestIsHf:
    def test_hf_true(self) -> None:
        assert is_hf("hf")

    def test_hf_uppercase_true(self) -> None:
        assert is_hf("HF")

    def test_modelscope_false(self) -> None:
        assert not is_hf("modelscope")

    def test_modelers_false(self) -> None:
        assert not is_hf("modelers")

    def test_non_string_false(self) -> None:
        assert not is_hf(None)  # type: ignore[arg-type]
        assert not is_hf(123)  # type: ignore[arg-type]

    def test_bool_false(self) -> None:
        # bool is a subclass of int but never a hub name; reject silently.
        assert not is_hf(True)  # type: ignore[arg-type]
        assert not is_hf(False)  # type: ignore[arg-type]


# =====================================================================
# Part E — TrainingConfig.hub schema integration
# =====================================================================


class TestTrainingConfigHub:
    def test_default_is_hf(self) -> None:
        cfg = TrainingConfig()
        assert cfg.hub == "hf"

    @pytest.mark.parametrize("hub", ["hf", "modelscope", "modelers"])
    def test_accepts_supported(self, hub: str) -> None:
        cfg = TrainingConfig(hub=hub)
        assert cfg.hub == hub

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(hub="github")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(hub="")

    def test_case_insensitive_normalised(self) -> None:
        # Review fix: field_validator(mode='before') normalises to lower
        # (matches v0.41.0 optimizer / v0.50.0 grpo_variant policy).
        cfg = TrainingConfig(hub="HF")
        assert cfg.hub == "hf"
        cfg2 = TrainingConfig(hub="ModelScope")
        assert cfg2.hub == "modelscope"

    def test_mlx_backend_rejects_non_hf_hub(self) -> None:
        yaml_str = """\
base: Qwen/Qwen2.5-7B
task: sft
backend: mlx
data:
  train: ./data/train.jsonl
  format: auto
training:
  epochs: 1
  lr: 2e-4
  hub: modelscope
output: ./output
"""
        # load_config_from_string wraps ValidationError as ValueError.
        with pytest.raises((ValidationError, ValueError), match="mlx"):
            load_config_from_string(yaml_str)

    def test_hub_none_rejected(self) -> None:
        # Pydantic Literal must reject None even after _normalize_hub.
        with pytest.raises(ValidationError):
            TrainingConfig(hub=None)  # type: ignore[arg-type]

    def test_modelers_on_transformers_accepted(self) -> None:
        yaml_str = """\
base: Qwen/Qwen2.5-7B
task: sft
backend: transformers
data:
  train: ./data/train.jsonl
  format: auto
training:
  epochs: 1
  lr: 2e-4
  hub: modelers
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.hub == "modelers"

    def test_mlx_backend_with_hf_hub_accepted(self) -> None:
        yaml_str = """\
base: Qwen/Qwen2.5-7B
task: sft
backend: mlx
data:
  train: ./data/train.jsonl
  format: auto
training:
  epochs: 1
  lr: 2e-4
  hub: hf
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.hub == "hf"

    def test_yaml_roundtrip_modelscope(self) -> None:
        yaml_str = """\
base: Qwen/Qwen2.5-7B
task: sft
data:
  train: ./data/train.jsonl
  format: auto
training:
  epochs: 1
  lr: 2e-4
  hub: modelscope
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.hub == "modelscope"


# =====================================================================
# Part D — MULTIPACK_ARCHITECTURES extension
# =====================================================================


class TestMultipackArchitecturesV0510:
    @pytest.mark.parametrize("arch", [
        "GraniteForCausalLM",
        "GraniteMoeForCausalLM",
        "Glm4ForCausalLM",
        "Glm5ForCausalLM",
        "KimiForCausalLM",
        "MiniMaxForCausalLM",
        "QwQForCausalLM",
        "QVQForCausalLM",
        "GptOssForCausalLM",
        "MagistralForCausalLM",
        "DevstralForCausalLM",
        "MinistralForCausalLM",
        "MedGemmaForCausalLM",
        "Lfm2ForCausalLM",
        "CogitoForCausalLM",
        "HunyuanForCausalLM",
        "ErnieForCausalLM",
        "YiForCausalLM",
        "BaichuanForCausalLM",
        "ChatGLMForConditionalGeneration",
    ])
    def test_arch_in_allowlist(self, arch: str) -> None:
        assert arch in MULTIPACK_ARCHITECTURES
        validate_multipack_architecture(arch)  # must not raise

    def test_legacy_arches_still_present(self) -> None:
        # Sanity — v0.37.0 entries must still be there
        for arch in ("LlamaForCausalLM", "Qwen2ForCausalLM",
                     "Phi3ForCausalLM", "Gemma2ForCausalLM"):
            assert arch in MULTIPACK_ARCHITECTURES

    def test_count_exactly_38(self) -> None:
        # 18 v0.37.0 + 20 v0.51.0 additions = 38. Exact-count assertion
        # so accidental deletion fails loudly (review fix).
        assert len(MULTIPACK_ARCHITECTURES) == 38

    def test_frozen_set(self) -> None:
        with pytest.raises(AttributeError):
            MULTIPACK_ARCHITECTURES.add("Evil")  # type: ignore[attr-defined]


# =====================================================================
# Parts A/B/C — 25 new recipes
# =====================================================================


V0510_RECIPE_NAMES = [
    # Part A — reasoning / agent
    "gpt-oss-20b-sft", "gpt-oss-120b-sft", "glm-4.6-sft", "glm-5-sft",
    "kimi-k2-sft", "kimi-k2-thinking-grpo", "minimax-m2-sft",
    "qwq-32b-grpo", "qvq-72b-sft",
    # Part B — small / specialist
    "granite-4-sft", "lfm2-sft", "cogito-v2-sft", "mistral-small-3-sft",
    "mistral-medium-3-5-sft", "magistral-small-sft", "devstral-sft",
    "ministral-sft", "medgemma-sft", "embedding-gemma-sft",
    # Part C — vision / multimodal
    "llava-next-sft", "internvl-3-5-sft", "voxtral-sft", "baichuan-sft",
    "qwen-image-sft", "deepseek-ocr-sft", "paddle-ocr-sft",
]


class TestV0510Recipes:
    def test_recipe_count_target(self) -> None:
        # 25 new entries (we shipped 26 to better cover the catalogue).
        assert len(V0510_RECIPE_NAMES) >= 25

    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_recipe_registered(self, name: str) -> None:
        assert name in RECIPES, f"Recipe {name!r} missing from catalog"
        assert get_recipe(name) is not None

    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_recipe_metadata_well_formed(self, name: str) -> None:
        meta = RECIPES[name]
        assert meta.model
        assert meta.task in {
            "sft", "dpo", "grpo", "kto", "orpo", "simpo", "ipo",
            "ppo", "reward_model", "pretrain", "embedding", "bco",
            "preference", "prm",
        }
        assert meta.size
        assert meta.tags
        assert meta.description
        assert meta.yaml_str

    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_recipe_yaml_parses_as_soup_config(self, name: str) -> None:
        yaml_str = RECIPES[name].yaml_str
        cfg = load_config_from_string(yaml_str)
        assert isinstance(cfg, SoupConfig)
        assert cfg.base == RECIPES[name].model
        assert cfg.task == RECIPES[name].task

    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_recipe_yaml_is_safe_loadable(self, name: str) -> None:
        # Defence-in-depth — yaml.safe_load must succeed (no Python tags etc.)
        parsed = yaml.safe_load(RECIPES[name].yaml_str)
        assert isinstance(parsed, dict)
        assert "base" in parsed and "task" in parsed

    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_recipe_model_id_no_null_or_whitespace(self, name: str) -> None:
        meta = RECIPES[name]
        assert "\x00" not in meta.model
        assert " " not in meta.model
        # ``owner/name`` shape OR plain name; every component must be non-empty
        # (rejects leading/trailing slashes — review fix).
        parts = meta.model.split("/")
        assert 1 <= len(parts) <= 2
        assert all(p for p in parts), f"empty component in {meta.model!r}"

    def test_baichuan_recipe_uses_modelscope_hub(self) -> None:
        cfg = load_config_from_string(RECIPES["baichuan-sft"].yaml_str)
        assert cfg.training.hub == "modelscope"

    def test_search_finds_v0510_recipes(self) -> None:
        results = search_recipes(query="gpt-oss")
        assert any(r.model.startswith("openai/") for r in results)

    def test_search_kimi(self) -> None:
        results = search_recipes(query="kimi")
        assert len(results) >= 2  # k2 + k2-thinking

    def test_total_recipe_count_increased(self) -> None:
        # The v0.31.0 baseline shipped 80 recipes; v0.51.0 adds ~26.
        assert len(list_recipes()) >= 80 + 25


# =====================================================================
# Parts A/B/C — invariants from test_recipes_v031.py mirrored
# =====================================================================


class TestV0510RecipeInvariants:
    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_max_length_within_bounds(self, name: str) -> None:
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        # Schema bounds: 64 <= max_length <= 1_048_576
        assert 64 <= cfg.data.max_length <= 1_048_576

    @pytest.mark.parametrize("name", V0510_RECIPE_NAMES)
    def test_task_has_required_fields(self, name: str) -> None:
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        if cfg.task == "grpo":
            assert cfg.training.reward_fn is not None
            assert cfg.training.num_generations >= 1


# =====================================================================
# Cross-cutting — module surface
# =====================================================================


class TestHubsModuleSurface:
    def test_module_exposes_validators(self) -> None:
        for sym in (
            "SUPPORTED_HUBS",
            "validate_hub_name",
            "validate_hub_endpoint",
            "resolve_endpoint",
            "default_endpoint",
            "endpoint_env_var",
            "required_hub_package",
            "is_hf",
        ):
            assert hasattr(hubs_mod, sym)

    def test_default_endpoints_are_mappingproxytype(self) -> None:
        # Internal but useful invariant — registry must not be mutable.
        assert isinstance(hubs_mod._HUB_DEFAULT_ENDPOINTS, MappingProxyType)
        assert isinstance(hubs_mod._HUB_ENDPOINT_ENV, MappingProxyType)
        assert isinstance(hubs_mod._HUB_PACKAGE, MappingProxyType)


# =====================================================================
# Templates — sanity (no template additions in v0.51.0, but the manifest
# must still load without regression)
# =====================================================================


class TestTemplateRegressionGuard:
    def test_templates_dict_intact(self) -> None:
        # v0.40.0 baseline = 17 inline templates
        assert len(TEMPLATES) >= 17
