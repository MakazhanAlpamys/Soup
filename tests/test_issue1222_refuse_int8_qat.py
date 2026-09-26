"""#1222 -- ``training.quantization_aware: true`` is refused at config load.

``true`` never selected quantization-aware training. It handed torchao's
``Int8WeightOnlyConfig`` -- a post-training, weight-only quantizer -- to
``quantize_`` after LoRA was attached, over every ``nn.Linear`` including
``lora_A`` and ``lora_B``. The issue's CPU repro went from 8 trainable tensors
to 0 and then died in backward (``element 0 of tensors does not require
grad``); through the real ``soup train`` every path tried (sft, sft with
gradient checkpointing, full fine-tuning, dpo) exited 1.

The ruling is "refuse until real QAT exists". This file pins:

* the refusal, for every spelling the schema reads as true, on every task and
  backend that accepted it (unsloth included, which supersedes #1248), and on
  every entry point that builds a config (the string loader the Web UI uses,
  direct construction, ``soup train`` and ``soup doctor --config``);
* the message: it names the setting, says why, says what still works and
  points at the issue;
* that no other message recommends the refused value (``soup train``'s
  unsloth refusal used to);
* the values that must NOT change -- ``fp8``, ``quest`` and every false
  spelling. Those controls pass before and after the fix by design; the
  refusal tests are the ones that fail without it.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

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
#: unfixed schema: the YAML 1.1 booleans, pydantic's lax bool strings (quoted,
#: plus the bare ``y``/``Y`` that YAML leaves as a string) and the numbers 1
#: and 1.0. ``test_each_spelling_is_read_as_true`` keeps the list honest.
YAML_SPELLINGS_READ_AS_TRUE = [
    "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON",
    "y", "Y", "1", "1.0", "'true'", '"True"', "'1'", "'yes'", "'on'", "'t'",
]

#: The same property for a caller that builds the model directly (``soup
#: sweep``, ``soup rewind``, autopilot and the Web UI all construct it).
PYTHON_VALUES_READ_AS_TRUE = [True, 1, 1.0, "true", "True", "yes", "on", "t", "y", "1"]

#: Spellings that mean "off". They must keep loading, as ``False``.
YAML_SPELLINGS_READ_AS_FALSE = ["false", "False", "no", "off", "0", "'false'", "'0'", "'n'"]

#: Every task whose own trainer called ``prepare_model_for_qat`` after LoRA:
#: the nine the issue lists, plus simpo, which carries the same branch. ``tts``
#: and ``preference`` reached it through another trainer; they have their own
#: test below.
TASKS_THAT_REACHED_THE_INT8_HELPER = [
    "sft", "dpo", "kto", "orpo", "ipo", "bco", "simpo", "grpo", "ppo", "pretrain",
]


def _plain(text: "str | None") -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(strip_ansi(text).split())


def _yaml(value: str, *, task: str = "sft", backend: str = "transformers") -> str:
    return (
        f"base: some-model\ntask: {task}\nbackend: {backend}\n"
        "data: {train: ./data.jsonl}\n"
        f"training: {{quantization_aware: {value}}}\n"
    )


def _assert_is_the_1222_refusal(message: str) -> None:
    flat = " ".join(message.split())
    assert "training.quantization_aware" in flat, flat
    assert "#1222" in flat, flat


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------


def test_the_message_names_the_setting_the_reason_what_still_works_and_the_issue():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true"))
    message = " ".join(str(excinfo.value).split())

    assert "training.quantization_aware" in message
    # The reason, and what the old path actually did to the model.
    assert "int8 quantization-aware training is not implemented" in message
    assert "adapter included" in message
    assert "zero trainable parameters" in message
    # Scoped to where that path ran. MLX ignored the field with a warning, no
    # unsloth setup ever applied it (#1248), and many trainers never read it,
    # so those runs completed without QAT: the message must not claim that no
    # run ever did.
    assert "Where it was applied, it ran" in message
    assert (
        "Elsewhere, including backend: mlx and backend: unsloth, it was not applied at all"
        in message
    )
    assert "no run completed" not in message
    # What still works, and why it is unaffected.
    assert "quantization_aware: fp8" in message
    assert "quantization_aware: quest" in message
    assert "own code paths" in message
    assert "#1222" in message


@pytest.mark.parametrize("spelling", YAML_SPELLINGS_READ_AS_TRUE)
def test_the_loader_refuses_every_spelling_read_as_true(spelling):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(spelling))
    _assert_is_the_1222_refusal(str(excinfo.value))


@pytest.mark.parametrize("value", PYTHON_VALUES_READ_AS_TRUE, ids=repr)
def test_direct_construction_refuses_every_value_read_as_true(value):
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(quantization_aware=value)
    _assert_is_the_1222_refusal(str(excinfo.value))


def test_the_error_is_located_at_the_field_so_the_cli_names_it():
    """The loader prints ``loc: msg``; the loc must be the field itself, not
    an empty model-level location."""
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": True},
        )
    locations = [error["loc"] for error in excinfo.value.errors()]
    assert locations == [("training", "quantization_aware")], locations


@pytest.mark.parametrize("task", TASKS_THAT_REACHED_THE_INT8_HELPER)
def test_every_task_that_reached_the_int8_helper_refuses_it(task):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true", task=task))
    _assert_is_the_1222_refusal(str(excinfo.value))


@pytest.mark.parametrize(
    "config",
    [
        "task: tts\nmodality: audio_out\ndata: {train: ./data.jsonl}\n"
        "training: {quantization_aware: true, tts_family: orpheus}\n",
        "task: preference\ndata: {train: ./data.jsonl}\n"
        "training: {quantization_aware: true, preference_loss: dpo}\n",
    ],
    ids=["tts", "preference"],
)
def test_the_tasks_that_reached_it_through_another_trainer_refuse_it(config):
    """Neither trainer names the field, but both applied the int8 path: ``tts``
    inherits SFT's model setup, and ``preference`` runs one of the dpo, simpo,
    orpo, ipo and bco trainers."""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string("base: some-model\n" + config)
    _assert_is_the_1222_refusal(str(excinfo.value))


@pytest.mark.parametrize("backend", ["unsloth", "mlx"])
def test_the_other_backends_refuse_it_too(backend):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true", backend=backend))
    _assert_is_the_1222_refusal(str(excinfo.value))


@pytest.mark.parametrize("task", ["sft", "dpo"])
def test_unsloth_refuses_it_on_sft_and_a_preference_task(task):
    """#1248: on unsloth only ``soup train``'s pre-flight refused it, so
    ``soup sweep``, ``soup bench train`` and the Python API trained without
    QAT. The load refuses it now, on every task."""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true", task=task, backend="unsloth"))
    _assert_is_the_1222_refusal(str(excinfo.value))


def test_vision_refuses_it_too():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(
            "base: some-model\ntask: sft\nmodality: vision\n"
            "data: {train: ./data.jsonl, format: llava, image_dir: ./images}\n"
            "training: {quantization_aware: true, quantization: 4bit}\n"
        )
    _assert_is_the_1222_refusal(str(excinfo.value))


def test_the_refusal_wins_over_the_pre_quantized_cross_validator():
    """``true`` next to a pre-quantized format used to be reported as a format
    clash; the actionable problem is that ``true`` itself cannot run."""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(
            "base: some-model\ntask: sft\ndata: {train: ./data.jsonl}\n"
            "training: {quantization: gptq, quantization_aware: true}\n"
        )
    _assert_is_the_1222_refusal(str(excinfo.value))


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
        "base: some-model\ntask: sft\n"
        f"backend: {backend}\n"
        f"data:\n  train: {train.as_posix()}\n  format: chatml\n"
        "training:\n  quantization: none\n"
        f"  quantization_aware: {value}\n"
        "  lora: {r: 8, alpha: 16}\n"
        f"output: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("extra_args", [[], ["--dry-run"]], ids=["train", "dry-run"])
def test_soup_train_stops_at_config_load(tmp_path, monkeypatch, extra_args):
    """Nothing past the loader may run: no setup panel, no "QAT enabled", no
    trainable-parameter line describing a model that can no longer train."""
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, "true")

    result = CliRunner().invoke(app, ["train", "--config", str(config), *extra_args])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "training -> quantization_aware" in output, output
    assert "#1222" in output, output
    assert "quantization_aware: fp8" in output, output
    # The refusal itself says "zero trainable parameters", so the check is on
    # the prefixes of the lines that report the model, not on the bare word.
    for later_line in ("Training Setup", "QAT enabled", "LoRA applied", "Full fine-tuning"):
        assert later_line not in output, (later_line, output)


def test_soup_train_on_unsloth_stops_at_load_without_a_pip_hint(tmp_path, monkeypatch):
    """#1248: ``soup train`` used to refuse unsloth + ``true`` only at its QAT
    pre-flight, and with torchao missing it also said ``pip install torchao``,
    which cannot help when the backend is the problem. The load refuses it
    now, before that pre-flight runs."""
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, "true", backend="unsloth")

    result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "#1222" in output, output
    assert "pip install" not in output, output
    assert "QAT error" not in output, output


def test_soup_doctor_config_reports_the_config_as_unloadable(tmp_path, capsys):
    """Before the fix this MLX config loaded and doctor listed
    ``training.quantization_aware`` as an ignored setting; now it cannot load,
    and an unloadable config is exit 2."""
    from soup_cli.commands.doctor import doctor

    config = _write_config(tmp_path, "true", backend="mlx")

    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(config))

    assert excinfo.value.exit_code == 2
    output = _plain(capsys.readouterr().out)
    assert "#1222" in output, output


# --------------------------------------------------------------------------
# no message may recommend the refused value
# --------------------------------------------------------------------------

#: What ``soup train`` says when ``quantization_aware: fp8`` meets
#: ``backend: unsloth``. It used to end "Use backend: transformers with
#: quantization_aware: true.", sending the user to the value refused above.
UNSLOTH_REFUSAL = "QAT is not compatible with the unsloth backend. Use backend: transformers."


def test_the_unsloth_refusal_points_at_the_transformers_backend_only():
    from soup_cli.utils.qat import validate_qat_config

    errors = validate_qat_config(quantization="none", backend="unsloth", modality="text")
    assert UNSLOTH_REFUSAL in errors, errors
    assert not any("quantization_aware: true" in error for error in errors), errors


def test_fp8_on_unsloth_prints_the_new_refusal(tmp_path, monkeypatch):
    """``fp8`` with ``backend: unsloth`` still loads, and ``soup train``'s QAT
    check refuses it; that refusal must not name ``quantization_aware: true``."""
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, "fp8", backend="unsloth")

    result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"QAT error: {UNSLOTH_REFUSAL}" in output, output
    assert "quantization_aware: true" not in output, output


# --------------------------------------------------------------------------
# controls -- these pass with and without the fix, by design
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", YAML_SPELLINGS_READ_AS_TRUE)
def test_each_spelling_is_read_as_true(spelling):
    """The enumeration above is the field's own coercion, not a guess."""
    annotation = TrainingConfig.model_fields["quantization_aware"].annotation
    raw = yaml.safe_load(f"value: {spelling}")["value"]
    assert TypeAdapter(annotation).validate_python(raw) is True


class TestTheValuesThatStillLoadAreUnchanged:
    @pytest.mark.parametrize("spelling", YAML_SPELLINGS_READ_AS_FALSE)
    def test_every_false_spelling_still_loads_as_off(self, spelling):
        cfg = load_config_from_string(_yaml(spelling))
        assert cfg.training.quantization_aware is False

    def test_the_default_is_still_off(self):
        cfg = SoupConfig(base="some-model", data={"train": "./data.jsonl"})
        assert cfg.training.quantization_aware is False

    @pytest.mark.parametrize("task", ["sft", "dpo"])
    def test_unsloth_without_qat_still_loads(self, task):
        """#1248's control: only the value is refused, not the backend."""
        cfg = load_config_from_string(_yaml("false", task=task, backend="unsloth"))
        assert cfg.backend == "unsloth"
        assert cfg.training.quantization_aware is False

    @pytest.mark.parametrize("task", TASKS_THAT_REACHED_THE_INT8_HELPER)
    def test_fp8_still_loads_on_every_task_that_took_true(self, task):
        cfg = load_config_from_string(_yaml("fp8", task=task))
        assert cfg.training.quantization_aware == "fp8"
        assert cfg.training.fp8_recipe == "tensorwise"

    def test_quest_still_loads(self):
        cfg = load_config_from_string(
            "base: ahxt/LiteLlama-460M-1T\ntask: sft\nbackend: transformers\n"
            "modality: text\ndata: {train: ./data.jsonl}\n"
            "training: {quantization_aware: quest, quantization: none, "
            "batch_size: 2, lora: {r: 0}}\n"
        )
        assert cfg.training.quantization_aware == "quest"


def _sft_wrapper(training: dict):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    cfg = SoupConfig(
        base="some-model",
        task="sft",
        data={"train": "./data.jsonl"},
        training=training,
    )
    wrapper = SFTTrainerWrapper(config=cfg, device="cpu")
    wrapper.model = MagicMock()
    return wrapper, cfg


def _the_int8_helper_must_not_run(*args, **kwargs):
    raise AssertionError("prepare_model_for_qat is unreachable from a config that loads")


class TestTheInt8HelperIsUnreachableFromAConfigThatLoads:
    """The SFT dispatcher, driven with each value the schema still accepts."""

    def test_off_never_reaches_it(self):
        wrapper, cfg = _sft_wrapper({"quantization_aware": False})
        with patch(
            "soup_cli.utils.qat.prepare_model_for_qat", _the_int8_helper_must_not_run
        ), patch("soup_cli.utils.v028_features.apply_v028_speed_memory") as v028:
            wrapper._apply_quantization_aware(cfg.training)
        v028.assert_called_once()
        assert v028.call_args.kwargs["skip_cut_ce"] is True

    def test_fp8_takes_its_own_path(self):
        wrapper, cfg = _sft_wrapper({"quantization_aware": "fp8"})
        model = wrapper.model
        with patch(
            "soup_cli.utils.qat.prepare_model_for_qat", _the_int8_helper_must_not_run
        ), patch("soup_cli.utils.fp8.apply_fp8_training", return_value=True) as fp8:
            wrapper._apply_quantization_aware(cfg.training)
        fp8.assert_called_once_with(model, recipe="tensorwise")

    def test_quest_takes_its_own_path(self):
        cfg = load_config_from_string(
            "base: ahxt/LiteLlama-460M-1T\ntask: sft\nbackend: transformers\n"
            "modality: text\ndata: {train: ./data.jsonl}\n"
            "training: {quantization_aware: quest, quantization: none, "
            "batch_size: 2, lora: {r: 0}}\n"
        )
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper(config=cfg, device="cpu")
        model = MagicMock()
        wrapper.model = model
        with patch(
            "soup_cli.utils.qat.prepare_model_for_qat", _the_int8_helper_must_not_run
        ), patch(
            "soup_cli.utils.v028_features.apply_v028_speed_memory",
            _the_int8_helper_must_not_run,
        ):
            wrapper._apply_quantization_aware(cfg.training)
        assert wrapper.model is model


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------

_TRUE_SETTING = re.compile(
    r"^\s*quantization_aware:\s*(true|yes|on|1)\b", re.IGNORECASE | re.MULTILINE
)


def test_no_doc_still_shows_quantization_aware_true_as_a_setting_to_copy():
    targets = [*sorted(REPO_ROOT.glob("README*.md")), *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in targets
        if _TRUE_SETTING.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"these docs still show quantization_aware: true: {offenders}"


def test_the_qat_docs_say_true_is_refused_and_why():
    text = (REPO_ROOT / "docs" / "performance-and-quantization.md").read_text(encoding="utf-8")
    section = text.split("## Quantization-Aware Training (QAT)", 1)[1].split("\n## ", 1)[0]
    flat = " ".join(section.split())
    assert "#1222" in flat or "issues/1222" in flat, flat
    assert "refused" in flat, flat
    assert "quantization_aware: fp8" in flat and "quantization_aware: quest" in flat, flat
