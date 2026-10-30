"""``training.activation_offloading`` reaches the VRAM estimate, which says it is not modelled.

The estimate charges every saved tensor as if it stayed on the card, with or without
offloading: what ``cpu`` or ``disk`` moves off it has not been measured. So the mode is
carried from the config into ``HardwareFitInput`` and ``RunSpec`` (and read by the
streamed estimate's caller), the terms stay as they are, and every place that shows the
estimate says so: ``soup profile`` (text and ``--json``), the train pre-flight, the
layer-streaming plan panel and ``soup plan``. The pre-flight's refuse / warn decision is
the same as with the mode unset.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import typer
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.hardware_fit import (
    decide_hardware_fit,
    hardware_fit_input_from_config,
    outside_calibrated_range,
)
from soup_cli.utils.profiler import profile_from_config
from soup_cli.utils.vram_estimate import GB, RunSpec, estimate_vram
from tests.conftest import strip_ansi

MODES = (None, "cpu", "disk")
SET_MODES = ("cpu", "disk")

# Llama-3.1-8B dimensions, as in the issue: shape metadata only, no weights.
_LLAMA_8B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}

# Qwen2.5-0.5B dimensions: inside the calibrated range, so a peak that does not fit is
# refused there.
_SMALL = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _note(mode: str, activations_gb: float) -> str:
    return (
        f"training.activation_offloading: {mode} is set: the activation term "
        f"({activations_gb:.2f} GB) is charged as if every saved tensor stayed on the card; "
        "what offloading moves off it is not modelled."
    )


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for name, config in (("model", _LLAMA_8B), ("small", _SMALL)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "data.jsonl").write_text('{"text":"example"}\n', encoding="utf-8")
    return tmp_path


def _document(
    mode, *, checkpointing=False, base="./model", max_length=4096, batch_size=4
) -> dict:
    return {
        "base": base,
        "backend": "transformers",
        "task": "sft",
        "data": {"train": "data.jsonl", "max_length": max_length},
        "training": {
            "batch_size": batch_size,
            "quantization": "none",
            "gradient_checkpointing": checkpointing,
            "activation_offloading": mode,
            "lora": {"r": 64, "target_modules": "auto"},
        },
        "output": "./output",
    }


def _cfg(mode, **kwargs):
    return load_config_from_string(json.dumps(_document(mode, **kwargs)))


# --- the issue's reproduction --------------------------------------------------------


@pytest.mark.parametrize("checkpointing", [False, True])
def test_the_mode_reaches_the_input_and_the_run_and_the_terms_are_unchanged(
    workdir, checkpointing
):
    seen = {}
    for mode in MODES:
        cfg = _cfg(mode, checkpointing=checkpointing)
        inp = hardware_fit_input_from_config(cfg, allow_network=False)
        report = decide_hardware_fit(inp, available_vram_gb=96)
        run = report.breakdown.estimate.run
        est = estimate_vram(inp.shape, run)
        profile = profile_from_config(cfg, allow_network=False)
        assert est.total_bytes == profile["total_memory_bytes"]
        assert inp.activation_offloading == mode
        assert run.activation_offloading == mode
        seen[mode] = (est.terms, report.ok, report.required_with_margin_gb)
    assert seen["cpu"] == seen[None]
    assert seen["disk"] == seen[None]


# --- the note ------------------------------------------------------------------------


def _estimate(mode, **kwargs):
    inp = hardware_fit_input_from_config(_cfg(mode, **kwargs), allow_network=False)
    return decide_hardware_fit(inp, available_vram_gb=96).breakdown.estimate


@pytest.mark.parametrize("mode", SET_MODES)
def test_the_estimate_says_the_activation_term_is_charged_as_resident(workdir, mode):
    est = _estimate(mode)
    notes = [note for note in est.notes if "activation_offloading" in note]
    assert notes == [_note(mode, est.gb("activations"))]
    assert "(92.56 GB)" in notes[0]


def test_no_note_without_the_mode(workdir):
    assert not any("activation_offloading" in note for note in _estimate(None).notes)


def test_the_other_notes_are_kept(workdir):
    plain, offloaded = _estimate(None), _estimate("cpu")
    assert offloaded.notes[: len(plain.notes)] == plain.notes
    assert len(offloaded.notes) == len(plain.notes) + 1


@pytest.mark.parametrize("value", ["gpu", "CPU", "", True, 1])
def test_run_spec_refuses_an_unknown_mode(value):
    with pytest.raises(ValueError, match="activation_offloading"):
        RunSpec(seq_len=128, batch_size=1, activation_offloading=value)


@pytest.mark.parametrize("mode", MODES)
def test_run_spec_takes_the_known_modes(mode):
    run = RunSpec(seq_len=128, batch_size=1, activation_offloading=mode)
    assert run.activation_offloading == mode


# --- soup profile ----------------------------------------------------------------------


_PROFILE = ["profile", "--config", "soup.yaml", "--gpu", "h100"]


def _write_config(mode, **kwargs) -> None:
    Path("soup.yaml").write_text(yaml.safe_dump(_document(mode, **kwargs)), encoding="utf-8")


@pytest.mark.parametrize("mode", SET_MODES)
def test_soup_profile_json_carries_the_note(workdir, mode):
    _write_config(mode)
    result = CliRunner().invoke(app, [*_PROFILE, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert _note(mode, data["memory_terms_bytes"]["activations"] / GB) in data["notes"]


@pytest.mark.parametrize("mode", SET_MODES)
def test_soup_profile_text_carries_the_note(workdir, mode):
    _write_config(mode)
    result = CliRunner().invoke(app, _PROFILE)
    assert result.exit_code == 0, result.output
    assert _note(mode, 92.56) in _plain(result.output)


def test_soup_profile_without_the_mode_says_nothing_about_it(workdir):
    _write_config(None)
    as_json = CliRunner().invoke(app, [*_PROFILE, "--json"])
    assert as_json.exit_code == 0, as_json.output
    assert not any("activation_offloading" in note for note in json.loads(as_json.output)["notes"])
    text = CliRunner().invoke(app, _PROFILE)
    assert "activation_offloading" not in text.output


# --- the train pre-flight ----------------------------------------------------------------


@pytest.fixture
def screen(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
    )
    return buffer


def _run(cfg, card_gb: float, *, allow_oom_attempt: bool = False):
    try:
        _hardware_fit_preflight(
            cfg, {"memory_total_bytes": int(card_gb * 1e9)}, allow_oom_attempt=allow_oom_attempt
        )
    except typer.Exit as exc:
        return exc.exit_code
    return None


@pytest.mark.parametrize("mode", SET_MODES)
def test_the_pre_flight_footnote_carries_the_note(workdir, screen, mode):
    assert _run(_cfg(mode), 96) is None  # 8B: outside the range, so a warning
    text = _plain(screen.getvalue())
    assert "Hardware-fit warning" in text
    assert _note(mode, 92.56) in text
    # With the other notes, after the advice (#1631's layout).
    assert text.index("Warning only: launching anyway.") < text.index(_note(mode, 92.56))


#: (config, card in GB, what the pre-flight does with the mode unset)
_DECISIONS = {
    "inside the range, does not fit: refused": (
        {"base": "./small", "max_length": 2048}, 6, 1, "Hardware-fit gate"
    ),
    "outside the range, does not fit: warned": ({}, 96, None, "Hardware-fit warning"),
    "outside the range, fits: nothing printed": ({"checkpointing": True}, 96, None, ""),
    "the floor alone does not fit: refused": ({}, 8, 1, "Hardware-fit gate"),
}


@pytest.mark.parametrize("case", sorted(_DECISIONS))
def test_the_pre_flight_decision_is_the_one_without_the_mode(workdir, monkeypatch, case):
    kwargs, card_gb, exit_code, title = _DECISIONS[case]
    outcomes = {}
    for mode in MODES:
        buffer = io.StringIO()
        monkeypatch.setattr(
            train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
        )
        cfg = _cfg(mode, **kwargs)
        code = _run(cfg, card_gb)
        text = _plain(buffer.getvalue())
        inp = hardware_fit_input_from_config(cfg, allow_network=False)
        shown = [name for name in ("Hardware-fit gate", "Hardware-fit warning") if name in text]
        outcomes[mode] = (code, shown, outside_calibrated_range(inp, backend="transformers"))
    assert outcomes[None][0] == exit_code
    assert outcomes[None][1] == ([title] if title else [])
    assert outcomes["cpu"] == outcomes[None]
    assert outcomes["disk"] == outcomes[None]
    assert not any("offload" in reason for reason in outcomes["cpu"][2])


@pytest.mark.parametrize("mode", SET_MODES)
def test_the_refusal_panel_carries_the_note(workdir, screen, mode):
    kwargs = {"base": "./small", "max_length": 2048}
    assert _run(_cfg(mode, **kwargs), 6) == 1  # inside the range, does not fit: refused
    text = _plain(screen.getvalue())
    assert "Hardware-fit gate" in text
    assert _note(mode, _estimate(mode, **kwargs).gb("activations")) in text


# --- the layer-streaming plan panel --------------------------------------------------


class _Lora:
    r = 8
    target_modules = ["q_proj", "v_proj"]


class _ModelConfig:
    vocab_size = 64
    hidden_size = 4096
    intermediate_size = 14336
    num_hidden_layers = 32


class _Data:
    max_length = 4096


class _Config:
    data = _Data()


class _TrainConfig:
    batch_size = 4
    stream_buffers = 2
    stream_vram_probe = False
    stream_vram_override = None
    stream_nf4_scale_cache = False
    lora = _Lora()

    def __init__(self, mode):
        self.activation_offloading = mode


class _Index:
    n_layers = 32


def _stream_lines(mode) -> list:
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    lines, _plan = StreamingSetupMixin()._stream_budget_lines(
        _Config(),
        _TrainConfig(mode),
        model_config=_ModelConfig(),
        layer_bytes=1000,
        embed_bytes=0,
        index=_Index(),
        on_cuda=False,
    )
    return list(lines)


@pytest.mark.parametrize("mode", SET_MODES)
def test_the_streaming_panel_carries_one_line_with_the_same_wording(mode):
    from soup_cli.utils.layer_stream import estimate_activation_bytes

    activations = estimate_activation_bytes(
        hidden_size=4096, intermediate_size=14336, n_layers=32, seq_len=4096, batch_size=4
    )
    lines = _stream_lines(mode)
    said = [line for line in lines if "activation_offloading" in line]
    assert len(said) == 1
    assert said[0].strip().endswith(_note(mode, activations / GB))
    # The streamed estimate is unchanged: every other line is the one without the mode.
    assert [line for line in lines if line not in said] == _stream_lines(None)


def test_the_streaming_panel_says_nothing_without_the_mode():
    assert not any("activation_offloading" in line for line in _stream_lines(None))


def _streamed_refusal(monkeypatch, mode) -> str:
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    # 1 GB free: the step is refused before anything is measured.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (int(1 * GB), 0))
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_gemm_tflops", lambda *_a, **_k: None
    )
    mixin = StreamingSetupMixin()
    mixin.device = "cuda"
    with pytest.raises(ValueError) as exc:
        mixin._stream_budget_lines(
            _Config(),
            _TrainConfig(mode),
            model_config=_ModelConfig(),
            layer_bytes=1000,
            embed_bytes=0,
            index=_Index(),
            on_cuda=True,
        )
    return " ".join(str(exc.value).split())


@pytest.mark.parametrize("mode", SET_MODES)
def test_a_streamed_refusal_carries_the_note(monkeypatch, mode):
    from soup_cli.utils.layer_stream import estimate_activation_bytes

    activations = estimate_activation_bytes(
        hidden_size=4096, intermediate_size=14336, n_layers=32, seq_len=4096, batch_size=4
    )
    message = _streamed_refusal(monkeypatch, mode)
    assert "predicted to need" in message
    assert message.endswith(_note(mode, activations / GB))


@pytest.mark.parametrize("mode", SET_MODES)
def test_a_probe_ceiling_refusal_carries_the_note(monkeypatch, mode):
    from soup_cli.utils.layer_stream import estimate_activation_bytes

    # About 8.8 GB predicted with 1 GB free is past the 4x ceiling: the probe cannot defer it.
    monkeypatch.setattr(_TrainConfig, "stream_vram_probe", True)
    activations = estimate_activation_bytes(
        hidden_size=4096, intermediate_size=14336, n_layers=32, seq_len=4096, batch_size=4
    )
    message = _streamed_refusal(monkeypatch, mode)
    assert "cannot overrule a prediction this far over budget" in message
    assert _note(mode, activations / GB) in message


def test_a_streamed_refusal_without_the_mode_says_nothing_about_it(monkeypatch):
    message = _streamed_refusal(monkeypatch, None)
    assert "predicted to need" in message
    assert "activation_offloading" not in message


# --- soup plan -----------------------------------------------------------------------------


_PLAN_LINE = (
    "training.activation_offloading: {mode} is not modelled; activations are charged as resident"
)


def _soup_plan(monkeypatch, mode, **kwargs) -> str:
    monkeypatch.setattr("soup_cli.commands.plan.console", Console(width=200))
    _write_config(mode, **kwargs)
    result = CliRunner().invoke(app, ["plan", "--config", "soup.yaml"])
    assert result.exit_code == 0, result.output
    return _plain(result.output)


@pytest.mark.parametrize("mode", SET_MODES)
def test_soup_plan_says_so_next_to_the_figure(workdir, monkeypatch, mode):
    text = _soup_plan(monkeypatch, mode)
    assert _PLAN_LINE.format(mode=mode) in text
    assert text.index("peak_vram_gb") < text.index(_PLAN_LINE.format(mode=mode))
    assert text.index(_PLAN_LINE.format(mode=mode)) < text.index("spot_price_usd_per_hour")


def test_soup_plan_without_the_mode_says_nothing_about_it(workdir, monkeypatch):
    assert "activation_offloading" not in _soup_plan(monkeypatch, None)


def test_soup_plan_with_no_figure_has_no_line(workdir, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    text = _soup_plan(monkeypatch, "cpu", base="someone/unreadable-7B")
    assert "unknown:" in text
    assert "activation_offloading" not in text
