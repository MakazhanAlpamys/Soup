"""#1269 — every allowlisted optimizer must survive TrainingArguments.

transformers 5.x rejects ten names Soup used to accept, so a config that
loaded fine crashed inside the trainer after the model was already loaded.
This suite builds TrainingArguments(optim=name) for every entry of
SUPPORTED_OPTIMIZERS, so a name transformers refuses fails here instead of
a user's run, and checks the retired names are refused at config load with
an actionable message.
"""

from __future__ import annotations

import pytest
from transformers import TrainingArguments

from soup_cli.utils.optimizer_zoo import (
    SUPPORTED_OPTIMIZERS,
    validate_optimizer_name,
)

# Names transformers 5.x rejects (verified on 5.17, #1269). Retired from the
# allowlist; the config loader must refuse them before any model loads.
_RETIRED = (
    "adam_mini",
    "adamw_apex_fused",
    "adamw_hf",
    "ao_adamw_4bit",
    "ao_adamw_8bit",
    "ao_adamw_fp8",
    "badam",
    "came_pytorch",
    "dion",
    "muon",
)


@pytest.mark.parametrize("name", sorted(SUPPORTED_OPTIMIZERS))
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


def test_adamw_hf_not_recommended_by_lorafa_validator():
    """The LoRA-FA compat tuple must not recommend a retired name (#1269)."""
    from soup_cli.config.schema import TrainingConfig

    with pytest.raises(Exception) as excinfo:
        TrainingConfig(use_lorafa=True, optimizer="adamw_hf")
    assert "no longer supported" in str(excinfo.value) or "incompatible" in str(
        excinfo.value
    )
