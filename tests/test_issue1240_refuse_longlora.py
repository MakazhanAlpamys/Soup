"""#1240 -- ``training.use_longlora: true`` is refused at config load.

``use_longlora: true`` installed ``apply_longlora_forward_override`` for the
whole SFT run. The override rolls the q/k projection outputs of half the heads
along the sequence with ``torch.roll``, which wraps the last
``group_size // 2`` positions around to the front, while attention stays full
causal. The issue's CPU repro (a tiny random-init Llama, default
``group_size=4``) changed only the last of eight tokens and saw the logits move
at positions 1-6 as well as 7, so the training loss at earlier positions could
read the end of the sequence. No grouped attention ran anywhere: it was not
S^2, and it delivered none of S^2's saving.

The ruling is "refuse until real S^2 attention exists". This file pins:

* the refusal, for every spelling the schema reads as true, through the loader
  and through direct construction, ahead of every older LongLoRA gate
  (architecture, task, backend, ring attention, FlashAttention v3) and of the
  three LoRA-conflict validators that list ``use_longlora``;
* the message: it names the setting, says the path leaks future tokens and is
  not S^2, points at ``rope_scaling_type`` with plain LoRA, and at the issue;
* ``soup train`` and ``soup doctor --config`` stop at config load;
* the values that must NOT change -- every false spelling, the default, and the
  ``rope_scaling_type`` + plain-LoRA configs the message recommends. Those
  controls pass before and after the fix by design;
* every shipped recipe, template and example still loads;
* no doc still shows ``use_longlora: true`` as a setting to copy.
"""

from __future__ import annotations

import re
import typing
from decimal import Decimal
from pathlib import Path

import pytest
import typer
import yaml
from pydantic import TypeAdapter, ValidationError
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every YAML spelling the schema reads as ``True``, enumerated against the
#: unfixed schema: the YAML 1.1 booleans, the YAML integers and floats equal to
#: one, pydantic's lax bool strings left as strings by YAML (bare ``y``/``t``
#: and any case of them), and the same strings quoted.
#: ``test_each_spelling_is_read_as_true`` keeps the list honest.
YAML_SPELLINGS_READ_AS_TRUE = [
    "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON",
    "y", "Y", "t", "T", "tRuE", "1", "+1", "0x1", "1.0",
    "'true'", '"True"', "'1'", "'yes'", "'y'", "'on'", "'t'",
]

#: The same property for a caller that builds the model directly (``soup
#: sweep``, ``soup rewind`` and autopilot construct it).
PYTHON_VALUES_READ_AS_TRUE = [
    True, 1, 1.0, Decimal(1), "true", "True", "TRUE", "yes", "y", "on", "t", "1",
    b"true", b"1",
]

#: Spellings that mean "off". They must keep loading, as ``False``.
YAML_SPELLINGS_READ_AS_FALSE = [
    "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF",
    "n", "N", "f", "F", "0", "0.0", "'false'", "'0'", "'no'", "'off'", "'n'", "'f'",
]

_LLAMA = "meta-llama/Llama-3.1-8B"


def _plain(text: "str | None") -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(strip_ansi(text).split())


def _yaml(
    value: str,
    *,
    base: str = _LLAMA,
    task: str = "sft",
    backend: str = "transformers",
    training_extra: str = "",
) -> str:
    return (
        f"base: {base}\ntask: {task}\nbackend: {backend}\n"
        "data: {train: ./data.jsonl}\n"
        "training:\n  quantization: none\n"
        f"  use_longlora: {value}\n{training_extra}"
    )


def _assert_is_the_1240_refusal(message: str) -> None:
    flat = " ".join(message.split())
    assert "training.use_longlora" in flat, flat
    assert "#1240" in flat, flat


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------


def test_the_message_names_the_setting_the_reason_the_alternative_and_the_issue():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true"))
    message = " ".join(str(excinfo.value).split())

    assert "training.use_longlora" in message
    assert "#1240" in message
    # The reason: what the installed path actually does.
    assert "leaks future tokens" in message
    assert "is not S^2" in message
    assert "wrap-around" in message
    assert "no grouped attention" in message
    # The way forward, and how to make the config load again.
    assert "rope_scaling_type" in message
    assert "plain LoRA" in message
    assert "remove use_longlora or set it to false" in message


@pytest.mark.parametrize("spelling", YAML_SPELLINGS_READ_AS_TRUE)
def test_the_loader_refuses_every_spelling_read_as_true(spelling):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(spelling))
    _assert_is_the_1240_refusal(str(excinfo.value))


@pytest.mark.parametrize("value", PYTHON_VALUES_READ_AS_TRUE, ids=repr)
def test_direct_construction_refuses_every_value_read_as_true(value):
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(use_longlora=value)
    _assert_is_the_1240_refusal(str(excinfo.value))


def test_the_error_is_located_at_the_field_so_the_cli_names_it():
    """The loader prints ``loc: msg``; the loc must be the field itself, not
    an empty model-level location."""
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig(
            base=_LLAMA,
            data={"train": "./data.jsonl"},
            training={"use_longlora": True},
        )
    locations = [error["loc"] for error in excinfo.value.errors()]
    assert locations == [("training", "use_longlora")], locations


#: Configs that the older gates refused with their OWN message, each of which
#: loads once ``use_longlora`` is removed. The #1240 refusal must come first:
#: the actionable problem is that the path itself cannot train correctly, not
#: the architecture, task, backend or LoRA-conflict rule it tripped next.
_OLDER_GATES = {
    "arch-allowlist": {"base": "google/gemma-2-9b"},
    "task-sft-only": {"task": "dpo"},
    "backend-mlx": {"base": "mlx-community/Llama-3.1-8B-Instruct-4bit", "backend": "mlx"},
    "backend-unsloth": {"backend": "unsloth"},
    "ring-attention": {"training_extra": "  use_ring_attention: true\n"},
    "full-ft-lora-conflict": {"training_extra": "  lora: {r: 0}\n"},
    "spectrum-lora-conflict": {
        "training_extra": "  unfrozen_parameters: [model.layers.0.mlp.down_proj]\n"
    },
    "lisa-lora-conflict": {"training_extra": "  lisa_enabled: true\n"},
}


@pytest.mark.parametrize("gate", sorted(_OLDER_GATES))
def test_the_refusal_comes_before_every_older_gate(gate):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true", **_OLDER_GATES[gate]))
    _assert_is_the_1240_refusal(str(excinfo.value))


@pytest.mark.parametrize("gate", sorted(_OLDER_GATES))
def test_each_older_gate_config_loads_without_use_longlora(gate):
    """Control for the test above: without the key, each config is valid, so
    the refusal there is the only reason it stops."""
    cfg = load_config_from_string(_yaml("false", **_OLDER_GATES[gate]))
    assert cfg.training.use_longlora is False


def test_the_refusal_comes_before_the_flash_attention_v3_gate(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.utils.flash_attn.is_flash_attn_v3_available", lambda: True
    )
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true"))
    _assert_is_the_1240_refusal(str(excinfo.value))


# --------------------------------------------------------------------------
# the entry points a user runs
# --------------------------------------------------------------------------


def _write_config(tmp_path: Path, value: str, *, backend: str = "transformers") -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n',
        encoding="utf-8",
    )
    path = tmp_path / "soup.yaml"
    path.write_text(
        f"base: {_LLAMA}\ntask: sft\n"
        f"backend: {backend}\n"
        f"data:\n  train: {train.as_posix()}\n  format: chatml\n"
        "training:\n  quantization: none\n"
        f"  use_longlora: {value}\n"
        "  lora: {r: 8, alpha: 16}\n"
        f"output: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("extra_args", [[], ["--dry-run"]], ids=["train", "dry-run"])
def test_soup_train_stops_at_config_load(tmp_path, monkeypatch, extra_args):
    """Nothing past the loader may run: no setup panel, and no line saying the
    override is active on a run that could not train correctly."""
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, "true")

    result = CliRunner().invoke(app, ["train", "--config", str(config), *extra_args])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "training -> use_longlora" in output, output
    assert "#1240" in output, output
    assert "leaks future tokens" in output, output
    assert "rope_scaling_type" in output, output
    for later_line in ("Training Setup", "attention override active", "LoRA applied"):
        assert later_line not in output, (later_line, output)


def test_soup_doctor_config_reports_the_config_as_unloadable(tmp_path, capsys):
    """Before the fix this config loaded and doctor reported on it; now it
    cannot load, and an unloadable config is exit 2."""
    from soup_cli.commands.doctor import doctor

    config = _write_config(tmp_path, "true")

    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(config))

    assert excinfo.value.exit_code == 2
    output = _plain(capsys.readouterr().out)
    assert "training -> use_longlora" in output, output
    assert "#1240" in output, output


# --------------------------------------------------------------------------
# controls -- these pass with and without the fix, by design
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", YAML_SPELLINGS_READ_AS_TRUE)
def test_each_spelling_is_read_as_true(spelling):
    """The enumeration above is the field's own coercion, not a guess."""
    annotation = TrainingConfig.model_fields["use_longlora"].annotation
    raw = yaml.safe_load(f"value: {spelling}")["value"]
    assert TypeAdapter(annotation).validate_python(raw) is True


@pytest.mark.parametrize("value", PYTHON_VALUES_READ_AS_TRUE, ids=repr)
def test_each_python_value_is_read_as_true(value):
    annotation = TrainingConfig.model_fields["use_longlora"].annotation
    assert TypeAdapter(annotation).validate_python(value) is True


class TestTheValuesThatStillLoadAreUnchanged:
    @pytest.mark.parametrize("spelling", YAML_SPELLINGS_READ_AS_FALSE)
    def test_every_false_spelling_still_loads_as_off(self, spelling):
        cfg = load_config_from_string(_yaml(spelling))
        assert cfg.training.use_longlora is False

    def test_the_default_is_still_off(self):
        cfg = SoupConfig(base=_LLAMA, data={"train": "./data.jsonl"})
        assert cfg.training.use_longlora is False

    def test_direct_construction_with_false_still_works(self):
        assert TrainingConfig(use_longlora=False).use_longlora is False

    @pytest.mark.parametrize(
        "rope_type",
        typing.get_args(
            typing.get_args(TrainingConfig.model_fields["rope_scaling_type"].annotation)[0]
        ),
    )
    def test_the_recommended_alternative_loads(self, rope_type):
        """The message sends users to ``rope_scaling_type`` with plain LoRA;
        every such config must load on the same base and task."""
        cfg = load_config_from_string(
            f"base: {_LLAMA}\ntask: sft\nbackend: transformers\n"
            "data: {train: ./data.jsonl}\n"
            "training:\n  quantization: none\n"
            f"  rope_scaling_type: {rope_type}\n"
            "  lora: {r: 16, alpha: 32}\n"
        )
        assert cfg.training.rope_scaling_type == rope_type
        assert cfg.training.lora.r == 16
        assert cfg.training.use_longlora is False


# --------------------------------------------------------------------------
# shipped configs
# --------------------------------------------------------------------------


def _shipped_config_texts() -> list[tuple[str, str]]:
    from soup_cli.recipes.catalog import RECIPES

    texts = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    for directory in (REPO_ROOT / "src" / "soup_cli" / "templates", REPO_ROOT / "examples"):
        assert directory.is_dir(), directory
        for path in sorted(directory.rglob("*.yaml")):
            texts.append(
                (path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8"))
            )
    return texts


def test_every_shipped_recipe_template_and_example_still_loads():
    texts = _shipped_config_texts()
    assert len(texts) > 200, len(texts)
    refused = []
    for name, text in texts:
        try:
            cfg = load_config_from_string(text)
        except ValueError as exc:
            refused.append((name, str(exc)[:200]))
            continue
        assert cfg.training.use_longlora is False, name
    assert refused == []


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------

_TRUE_SETTING = re.compile(
    r"^\s*use_longlora:\s*(true|yes|on|1)\b", re.IGNORECASE | re.MULTILINE
)


def test_no_doc_still_shows_use_longlora_true_as_a_setting_to_copy():
    targets = [*sorted(REPO_ROOT.glob("README*.md")), *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in targets
        if _TRUE_SETTING.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"these docs still show use_longlora: true: {offenders}"


def test_the_longlora_docs_say_it_is_refused_and_why():
    text = (REPO_ROOT / "docs" / "peft-and-efficiency.md").read_text(encoding="utf-8")
    section = text.split("## LongLoRA Forward Override", 1)[1].split("\n## ", 1)[0]
    flat = " ".join(section.split())
    assert "#1240" in flat or "issues/1240" in flat, flat
    assert "refused" in flat, flat
    assert "future tokens" in flat, flat
    assert "rope_scaling_type" in flat, flat
