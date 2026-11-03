"""The train pre-flight says why it does not check the fit, for every cause (#2363).

With ``batch_size: auto``, the default, the pre-flight returned without a word, so
nothing told the user the fit was never checked. It now prints one line for each
cause the estimate cannot price, and never shows an assumed batch as the run's own.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml
from rich.panel import Panel
from rich.text import Text

from soup_cli.commands import train as train_mod
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from soup_cli.utils import hardware_fit, model_shape

SHAPE = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}
CARD = {"memory_total_bytes": 6 * 1024**3}
AUTO_LINE = (
    "Pre-flight: the fit is not checked before training: "
    "training.batch_size is auto, which is chosen after the model loads."
)
_REAL_HUB_LOOKUP = model_shape._hub_lookup
_REAL_HUB_PARAM_COUNT = model_shape._hub_param_count


class RecordingConsole:
    def __init__(self) -> None:
        self.printed: list[str] = []
        self.renderables: list[object] = []

    def print(self, renderable: object) -> None:
        self.renderables.append(renderable)
        if isinstance(renderable, str):
            self.printed.append(Text.from_markup(renderable).plain)


def config(base: str, *, batch_size: int | str = "auto", quantization: str = "4bit") -> SoupConfig:
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "data": {"train": "train.jsonl", "max_length": 2048},
                "training": {
                    "batch_size": batch_size,
                    "quantization": quantization,
                    "gradient_checkpointing": True,
                },
            }
        )
    )


@pytest.fixture
def model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(SHAPE), encoding="utf-8")
    return str(folder)


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch) -> RecordingConsole:
    recorder = RecordingConsole()
    monkeypatch.setattr(train_mod, "console", recorder)
    return recorder


@pytest.mark.parametrize("allow_oom_attempt", [False, True])
def test_auto_prints_one_dim_line(
    model: str, console: RecordingConsole, allow_oom_attempt: bool
) -> None:
    train_mod._hardware_fit_preflight(
        config(model), CARD, allow_oom_attempt=allow_oom_attempt
    )
    assert console.printed == [AUTO_LINE]
    assert console.renderables == [f"[dim]{AUTO_LINE}[/]"]


def test_explicit_batch_keeps_the_refusal_panel(model: str, console: RecordingConsole) -> None:
    with pytest.raises(typer.Exit) as error:
        train_mod._hardware_fit_preflight(
            config(model, batch_size=1), CARD, allow_oom_attempt=False
        )
    assert error.value.exit_code == 1
    assert len(console.renderables) == 1
    panel = console.renderables[0]
    assert isinstance(panel, Panel)
    assert Text.from_markup(panel.title).plain == "Hardware-fit gate"
    lines = Text.from_markup(panel.renderable).plain.splitlines()
    assert lines[0] == (
        "Predicted peak VRAM 14.4 GB (+10% margin = 15.8 GB) exceeds 6.4 GB available."
    )


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Real readers over an empty cache, with every network call recorded."""
    import huggingface_hub
    from huggingface_hub import constants

    from soup_cli.utils import hf

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(model_shape, "_hub_lookup", _REAL_HUB_LOOKUP)
    monkeypatch.setattr(model_shape, "_hub_param_count", _REAL_HUB_PARAM_COUNT)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "config.json").write_text(json.dumps(SHAPE), encoding="utf-8")
    events: list[str] = []

    def head(url: str, **kwargs: object) -> SimpleNamespace:
        events.append("head")
        return SimpleNamespace(size=(downloaded / "config.json").stat().st_size)

    def download(repo: str, filename: str, **kwargs: object) -> str:
        events.append("download")
        return str(downloaded / filename)

    class Api:
        def get_safetensors_metadata(self, repo: str) -> SimpleNamespace:
            events.append("count")
            return SimpleNamespace(parameter_count={"BF16": 8_030_261_248})

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", head)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda repo, name: f"hub:{repo}/{name}")
    monkeypatch.setattr(hf, "get_hf_api", lambda *args, **kwargs: Api())
    return events


def test_auto_sends_no_request_and_prints_only_its_line(
    hub: list[str], console: RecordingConsole
) -> None:
    # A Hub id with nothing cached: reading its shape would send requests and print
    # the line before the Hub.
    train_mod._hardware_fit_preflight(
        config("someone/uncached-model"), CARD, allow_oom_attempt=False
    )
    assert hub == []
    assert console.printed == [AUTO_LINE]
    assert console.renderables == [f"[dim]{AUTO_LINE}[/]"]


def test_batch_outside_estimate_range_names_the_reason(
    model: str, console: RecordingConsole
) -> None:
    train_mod._hardware_fit_preflight(config(model, batch_size=2048), CARD, allow_oom_attempt=False)
    assert console.printed == [
        "Pre-flight skipped: training.batch_size is outside 1 to 1,024, "
        "the range the estimate prices."
    ]


def test_model_over_1000b_names_the_reason(
    console: RecordingConsole, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kwargs: (dict(SHAPE), None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kwargs: 2 * 10**12)
    train_mod._hardware_fit_preflight(
        config("someone/huge-model", batch_size=1), CARD, allow_oom_attempt=False
    )
    assert console.printed == [
        "Pre-flight skipped: the model in base has over 1,000B parameters, "
        "the most the estimate prices."
    ]


def test_max_length_outside_estimate_range_names_the_reason(
    model: str, console: RecordingConsole
) -> None:
    cfg = config(model, batch_size=1)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"max_length": 10**7})})
    train_mod._hardware_fit_preflight(cfg, CARD, allow_oom_attempt=False)
    assert console.printed == [
        "Pre-flight skipped: data.max_length is outside 64 to 1,048,576, "
        "the range the estimate prices."
    ]


def test_generic_refusal_names_the_reason(
    model: str, console: RecordingConsole, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RefusedInput:
        def __init__(self, **kwargs: object) -> None:
            raise ValueError("refused")

    monkeypatch.setattr(hardware_fit, "HardwareFitInput", RefusedInput)
    train_mod._hardware_fit_preflight(config(model, batch_size=1), CARD, allow_oom_attempt=False)
    assert console.printed == [
        "Pre-flight skipped: the run's inputs are outside the range the estimate prices."
    ]


def test_hqq_takes_precedence_over_auto(model: str, console: RecordingConsole) -> None:
    train_mod._hardware_fit_preflight(
        config(model, quantization="hqq:4bit"), CARD, allow_oom_attempt=False
    )
    assert console.printed == ["Pre-flight skipped: no memory model for quantization: hqq:4bit"]
    assert console.renderables == [
        "[yellow]Pre-flight skipped:[/] no memory model for quantization: hqq:4bit"
    ]
