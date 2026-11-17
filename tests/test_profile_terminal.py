"""``soup profile`` shows paths, model names and typed values as typed (#2489).

The config path, a typed ``--method`` / ``--optimizer`` / ``--gradient-checkpointing``
/ ``--gpu`` value, the ``--model`` name and the config's ``lora.target_modules`` keep
``:name:`` aliases and backslashes before brackets, and drop control and bidi
characters. The labels keep their colours.
"""

import sys
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import profile as profile_cmd
from tests._tiny_checkpoint import write_tiny_model
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

CONTROLS = "\x9b\u202e"
VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x" + CONTROLS + "n", "x\x1b\x07n"]
#: Values that can also be part of a directory name on every CI platform.
DIR_VALUES = [
    "plain",
    "x[red]n",
    "x" + CONTROLS + "n",
    pytest.param(
        "x\\[n",
        marks=pytest.mark.skipif(sys.platform == "win32", reason="backslash in a name"),
    ),
    pytest.param("n:rocket:", marks=skip_colon_on_windows),
]

#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")


def shown(value: str) -> str:
    for char in DROPPED:
        value = value.replace(char, "")
    return value


def plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(profile_cmd, "console", Console(width=1000, height=100))
    write_tiny_model(work / "tiny")


def _config(targets=("q_proj", "v_proj")) -> None:
    document = {
        "base": "tiny",
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "lora": {"r": 8, "target_modules": list(targets)}},
    }
    Path("soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _profile(*args: str):
    return runner.invoke(app, ["profile", *args])


@pytest.mark.parametrize("value", VALUES)
def test_a_missing_config_is_named_as_typed(value):
    result = _profile("--config", f"missing-{value}.yaml")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Config file not found: missing-{shown(value)}.yaml" in plain(result)


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("flag", ["--method", "--optimizer", "--gradient-checkpointing"])
def test_an_unknown_choice_is_named_as_typed(flag, value):
    result = _profile("--model", "tiny", flag, value)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Unknown {flag}: {shown(value)} Valid options:" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_an_unknown_gpu_is_named_as_typed(value):
    result = _profile("--model", "tiny", "--gpu", value)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Unknown GPU: {shown(value)} Valid options:" in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_an_unreadable_model_is_named_as_typed(value):
    model = f"./missing-{value}"

    result = _profile("--model", model, "--gpu", "a100")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Cannot read {shown(model)}: it is not a local model directory" in plain(result)


@pytest.mark.parametrize("value", DIR_VALUES)
def test_the_panel_names_the_model_as_typed(value):
    name = f"tiny-{value}"
    write_tiny_model(Path(name))

    result = _profile("--model", name, "--gpu", "a100")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Model: {shown(name)} " in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_the_panel_shows_any_model_name_as_typed(monkeypatch, value):
    """The name the estimate carries, whatever it is; runs on every platform."""
    real = profile_cmd.profile_from_model
    monkeypatch.setattr(
        profile_cmd,
        "profile_from_model",
        lambda *args, **kwargs: {**real(*args, **kwargs), "model_name": value},
    )

    result = _profile("--model", "tiny", "--gpu", "a100")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Model: {shown(value)} " in plain(result)


@pytest.mark.parametrize("value", VALUES)
def test_the_panel_names_the_target_modules_as_typed(value):
    _config(targets=(f"q_{value}", "v_proj"))

    result = _profile("--config", "soup.yaml", "--gpu", "a100")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"on q_{shown(value)}, v_proj)" in plain(result)


# ---------------------------------------------------------------------------
# The labels keep their colours
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(monkeypatch):
    monkeypatch.setattr(
        profile_cmd,
        "console",
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
    )


def test_a_refusal_keeps_its_red_label_and_dim_options(coloured):
    result = _profile("--model", "tiny", "--gpu", "nope")

    assert result.exit_code == 1
    assert "\x1b[31mUnknown GPU:\x1b[0m nope\n\x1b[2mValid options: " in result.output


@pytest.mark.parametrize("flag", ["--method", "--optimizer", "--gradient-checkpointing"])
def test_a_flag_refusal_keeps_its_red_label_and_dim_options(coloured, flag):
    result = _profile("--model", "tiny", flag, "nope")

    assert result.exit_code == 1
    assert f"\x1b[31mUnknown {flag}:\x1b[0m nope\n\x1b[2mValid options: " in result.output


def test_a_missing_config_keeps_its_red_label(coloured):
    result = _profile("--config", "missing.yaml")

    assert result.exit_code == 1
    assert "\x1b[31mConfig file not found:\x1b[0m missing.yaml" in result.output


def test_an_unreadable_model_keeps_its_red_label(coloured):
    result = _profile("--model", "./missing", "--gpu", "a100")

    assert result.exit_code == 1
    assert "\x1b[31mCannot read ./missing:\x1b[0m it is not a local" in result.output


def test_the_panel_keeps_its_bold_values(coloured):
    result = _profile("--model", "tiny", "--gpu", "a100", "--gradient-checkpointing", "full")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Model:     \x1b[1mtiny\x1b[0m" in result.output
    assert "Quantization: \x1b[1m4bit\x1b[0m" in result.output
    assert "Gradient checkpointing: \x1b[1;32menabled\x1b[0m (full)" in result.output
