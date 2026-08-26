"""#502/#503 — Transformers 5 Qwen3.5 text training compatibility."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKENDS_DOC = REPO_ROOT / "docs" / "backends-and-ops.md"
_DOCS_WARNING = "cannot be installed together"
_PROBE_VERSIONS = tuple(
    Version(f"{major}.{minor}.0") for major in range(3, 10) for minor in range(60)
)


def _extra_requirement(extra: str, package: str) -> Requirement:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(rf"^{re.escape(extra)}\s*=\s*\[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert block is not None
    requirements = [
        Requirement(raw)
        for raw in re.findall(r'"([^"\n]+)"', block.group(1))
        if Requirement(raw).name == package
    ]
    assert len(requirements) == 1
    return requirements[0]


def _single_bound(requirement: Requirement, operator: str) -> str:
    versions = [clause.version for clause in requirement.specifier if clause.operator == operator]
    assert len(versions) == 1, (
        f"expected one {operator} bound for {requirement.name}, found {versions}"
    )
    return versions[0]


_RUNTIME_FLOORS = {
    "transformers": "5.16.1",
    "trl": "0.29.0",
    "peft": "0.20.0",
}


def _missing_runtime_floors() -> list[str]:
    missing: list[str] = []
    for package, floor in _RUNTIME_FLOORS.items():
        try:
            installed = distribution_version(package)
            if Version(installed) < Version(floor):
                missing.append(f"{package}=={installed} (need >={floor})")
        except (PackageNotFoundError, InvalidVersion):
            missing.append(f"{package} (need >={floor})")
    return missing


_MISSING_RUNTIME_FLOORS = _missing_runtime_floors()
_RUNTIME_FLOOR_REASON = (
    "requires the validated training floor; install soup-cli[train] with "
    "transformers>=5.16.1, trl>=0.29.0, peft>=0.20.0; missing/outdated: "
    + ", ".join(_MISSING_RUNTIME_FLOORS)
)
_REQUIRES_TRAINING_FLOOR = pytest.mark.skipif(
    bool(_MISSING_RUNTIME_FLOORS),
    reason=_RUNTIME_FLOOR_REASON,
)


def _transformers_extras_are_disjoint() -> bool:
    """Return whether the declared train/mlx ranges have no shared version."""
    train = _extra_requirement("train", "transformers").specifier
    mlx = _extra_requirement("mlx", "transformers").specifier
    probes = list(_PROBE_VERSIONS)
    for specifier in (train, mlx):
        for clause in specifier:
            probes.append(Version(clause.version))
    return not any(
        train.contains(version, prereleases=True)
        and mlx.contains(version, prereleases=True)
        for version in probes
    )


def _tiny_qwen35_config():
    from transformers import Qwen3_5Config, Qwen3_5TextConfig

    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
    )
    return Qwen3_5Config(text_config=text.to_dict())


def test_training_and_mlx_extras_share_the_transformers5_range():
    train = _extra_requirement("train", "transformers")
    mlx = _extra_requirement("mlx", "transformers")

    assert train.specifier == mlx.specifier
    assert Version("5.16.1") in train.specifier
    assert Version("5.16.0") not in train.specifier
    assert Version("5.15.1") not in train.specifier
    assert Version("6.0.0") not in train.specifier


def test_docs_warn_exactly_when_training_and_mlx_extras_are_disjoint():
    disjoint = _transformers_extras_are_disjoint()
    warned = _DOCS_WARNING in BACKENDS_DOC.read_text(encoding="utf-8")

    assert warned is disjoint, (
        "docs/backends-and-ops.md must say the train/mlx extras cannot be installed "
        "together exactly when their declared transformers ranges are disjoint"
    )


def test_trl_floor_crosses_the_transformers5_import_fix():
    trl = _extra_requirement("train", "trl")

    assert Version("0.28.0") not in trl.specifier
    assert Version("0.29.0") in trl.specifier
    assert Version("1.0.0") not in trl.specifier


def test_peft_floor_is_the_validated_transformers5_adapter_stack():
    peft = _extra_requirement("train", "peft")

    assert Version("0.19.0") not in peft.specifier
    assert Version("0.20.0") in peft.specifier
    assert Version("1.0.0") not in peft.specifier


def test_doctor_training_bounds_match_declared_extra():
    from soup_cli.commands.doctor import _MAX_EXCLUSIVE, DEPS

    doctor_floors = {package: floor for _, package, floor, _ in DEPS}
    for package in _RUNTIME_FLOORS:
        requirement = _extra_requirement("train", package)
        assert doctor_floors[package] == _single_bound(requirement, ">=")
        assert _MAX_EXCLUSIVE[package] == _single_bound(requirement, "<")


def test_runtime_floor_skip_explains_the_required_upgrade():
    for package, floor in _RUNTIME_FLOORS.items():
        assert f"{package}>={floor}" in _RUNTIME_FLOOR_REASON


@_REQUIRES_TRAINING_FLOOR
def test_trl029_trainers_and_experimental_pairwise_judge_import():
    import trl.trainer.dpo_trainer  # noqa: F401
    import trl.trainer.grpo_trainer  # noqa: F401

    from soup_cli.eval.judge import _base_pairwise_judge_cls

    base_cls = _base_pairwise_judge_cls()
    assert base_cls.__name__ == "BasePairwiseJudge"
    assert base_cls.__module__.startswith("trl.experimental.judges")


@_REQUIRES_TRAINING_FLOOR
def test_qwen35_outer_config_selects_text_causal_model_without_vision():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_config(_tiny_qwen35_config(), dtype=torch.float32)

    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    assert model.config.model_type == "qwen3_5_text"
    assert not [
        name
        for name, _ in model.named_parameters()
        if "vision" in name or "visual" in name
    ]


@_REQUIRES_TRAINING_FLOOR
def test_qwen35_auto_targets_attach_and_reload_a_nonempty_adapter(tmp_path):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    config = _tiny_qwen35_config()
    base = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    targets = resolve_lora_target_modules(base, "auto")
    assert targets == ["q_proj", "v_proj", "in_proj_qkv", "out_proj"]

    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert sum(parameter.numel() for parameter in trainable) > 0

    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    reloaded_base = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter)
    reloaded_lora = [name for name, _ in reloaded.named_parameters() if "lora_" in name]
    assert len(reloaded_lora) == 8


def test_explicit_peft_targets_and_mlx_defaults_are_unchanged():
    from soup_cli.trainer.mlx_sft import MLX_DEFAULT_TARGET_KEYS
    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    explicit = ["custom_proj"]
    assert resolve_lora_target_modules(object(), explicit) is explicit
    assert resolve_lora_target_modules(object(), None) is None
    assert MLX_DEFAULT_TARGET_KEYS == ["self_attn.q_proj", "self_attn.v_proj"]
