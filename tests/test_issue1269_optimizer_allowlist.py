"""#1269 — every allowlisted optimizer must survive TrainingArguments.

transformers 5.x rejects ten names Soup used to accept, so a config that
loaded fine crashed inside the trainer after the model was already loaded.
This suite builds TrainingArguments(optim=name) for every entry of
SUPPORTED_OPTIMIZERS (minus the two MLX-only names, which never reach
TrainingArguments), so a name transformers refuses fails here instead of a
user's run, and checks the retired names are refused at config load with
an actionable message that only recommends names a run can actually build.
"""

from __future__ import annotations

import re

import pytest
from torch import nn
from transformers import Trainer, TrainingArguments

from soup_cli.utils.optimizer_zoo import (
    MLX_ONLY_OPTIMIZERS,
    SUPPORTED_OPTIMIZERS,
    validate_optimizer_name,
)

# Names transformers 5.x rejects (verified on 5.17, #1269). Retired from the
# allowlist; the config loader must refuse them before any model loads.
_RETIRED = (
    "adam_mini",
    "adamw_apex_fused",
    "ao_adamw_4bit",
    "ao_adamw_8bit",
    "ao_adamw_fp8",
    "badam",
    "came_pytorch",
    "dion",
)


@pytest.mark.parametrize(
    "name", sorted(SUPPORTED_OPTIMIZERS - MLX_ONLY_OPTIMIZERS)
)
def test_every_allowlisted_optimizer_passes_trainingarguments(name, tmp_path):
    """Each allowlisted name must be accepted by transformers' own gate."""
    TrainingArguments(output_dir=str(tmp_path), optim=name, report_to=[])


@pytest.mark.parametrize("name", _RETIRED)
def test_retired_optimizer_refused_at_config_load(name):
    """Retired names must fail fast with an actionable message."""
    with pytest.raises(ValueError, match="no longer supported"):
        validate_optimizer_name(name)


def test_retired_names_not_in_allowlist():
    assert not (SUPPORTED_OPTIMIZERS & frozenset(_RETIRED))


@pytest.mark.parametrize("name", ["BAdam", "Adam_Mini", "AO_ADAMW_8BIT"])
def test_retired_names_are_refused_case_insensitively(name):
    """The refusal must normalise the name before the lookup (#1269)."""
    with pytest.raises(ValueError, match="no longer supported"):
        validate_optimizer_name(name)


def _recommended_alternatives() -> list[str]:
    with pytest.raises(ValueError) as excinfo:
        validate_optimizer_name("badam")
    match = re.search(r"Use one of: ([a-z0-9_, ]+)\.", str(excinfo.value))
    assert match, str(excinfo.value)
    return match.group(1).split(", ")


@pytest.mark.parametrize("name", _recommended_alternatives())
def test_every_recommended_alternative_builds_an_optimizer(name, tmp_path):
    """#1269: no config-load message may recommend a name the run then rejects.

    TrainingArguments accepting a name is not enough: apollo_adamw passes it and
    then fails when the optimizer is built, because Soup never sets
    optim_target_modules.
    """
    assert name in SUPPORTED_OPTIMIZERS, name
    args = TrainingArguments(output_dir=str(tmp_path), optim=name, report_to=[])
    Trainer.get_optimizer_cls_and_kwargs(args, nn.Linear(4, 4))


def test_mlx_backend_accepts_mlx_only_optimizers(tmp_path):
    """#1283: `muon` / `adamw_hf` map to native MLX optimizers, not transformers."""
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

    for name in sorted(MLX_ONLY_OPTIMIZERS):
        SoupConfig(
            base="mlx-community/Llama-3.1-8B-Instruct-4bit",
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(epochs=1, optimizer=name),
            output=str(tmp_path),
        )


def test_transformers_backend_refuses_mlx_only_optimizers(tmp_path):
    """#1283: a non-MLX run must name the backend and refuse before load."""
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

    with pytest.raises(ValueError, match=r"only works on backend: mlx"):
        SoupConfig(
            base="hf-internal-testing/tiny-random-LlamaForCausalLM",
            task="sft",
            backend="transformers",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(epochs=1, optimizer="muon"),
            output=str(tmp_path),
        )


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
@pytest.mark.parametrize("optimizer", sorted(MLX_ONLY_OPTIMIZERS))
def test_every_non_mlx_backend_refuses_mlx_only_optimizers(backend, optimizer):
    """#1283: both TrainingArguments backends refuse `muon` / `adamw_hf` at load."""
    from soup_cli.config.loader import load_config_from_string

    text = (
        "base: some-org/some-model\n"
        "task: sft\n"
        f"backend: {backend}\n"
        "data:\n  train: ./train.jsonl\n"
        f"training:\n  optimizer: {optimizer}\n"
    )
    with pytest.raises(ValueError, match="only works on backend: mlx"):
        load_config_from_string(text)


def test_lorafa_config_with_adamw_hf_is_refused_at_load():
    """The LoRA-FA compat tuple must not accept a non-AdamW-projection name."""
    from soup_cli.config.schema import TrainingConfig

    with pytest.raises(ValueError, match="incompatible"):
        TrainingConfig(use_lorafa=True, optimizer="adamw_hf")
