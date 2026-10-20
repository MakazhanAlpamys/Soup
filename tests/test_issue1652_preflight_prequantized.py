"""#1652 — the ``soup train`` VRAM pre-flight covers the pre-quantized formats.

``_build_hardware_fit_input`` mapped ``none`` / ``4bit`` / ``8bit`` (and ``mxfp4``
since #1631) and returned ``None`` for the rest, so twelve formats started without
a check and without a word. The predictor already had a figure for five of them.

- ``gptq`` / ``awq`` / ``aqlm`` / ``eetq`` stay packed: priced at the table's figure.
- ``fp8`` is dequantized on load, like ``mxfp4``: priced as bf16. The table said
  1.0 byte per parameter; a load through the real config holds 2.0.
- ``hqq:*`` has no figure: the pre-flight says it skipped the run.

No GPU here: the gate is called with a card size, and the ``fp8`` load is a tiny
model on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from rich.text import Text

from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _build_hardware_fit_input, _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.hardware_fit import (
    _BYTES_PER_PARAM_BY_QUANT,
    HardwareFitInput,
    estimate_peak_vram_gb,
)

_YAML = """
base: openai/gpt-oss-20b
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: {batch_size}
  quantization: {quant}
"""

# A 20B base: 10 GB of weights at the smallest figure (aqlm), 40 GB at bf16.
SMALL_CARD = {"memory_total_bytes": int(8e9)}
LARGE_CARD = {"memory_total_bytes": int(96e9)}

STAY_PACKED = ["gptq", "awq", "aqlm", "eetq"]
PRICED = [*STAY_PACKED, "fp8"]
HQQ = ["hqq:1bit", "hqq:2bit", "hqq:3bit", "hqq:4bit", "hqq:5bit", "hqq:6bit", "hqq:8bit"]


def _cfg(quant: str, *, batch_size: object = 1):
    return load_config_from_string(_YAML.format(batch_size=batch_size, quant=quant))


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, renderable):
        self.printed.append(renderable)


@pytest.fixture
def console(monkeypatch):
    fake = _Console()
    monkeypatch.setattr(train_mod, "console", fake)
    return fake


def _weights_gb(quant: str) -> float:
    return estimate_peak_vram_gb(
        HardwareFitInput(
            params_b=20.0,
            seq_len=2048,
            batch_size=1,
            optimizer="adamw_torch",
            quant=quant,
            peft="lora",
            gradient_checkpointing=False,
        )
    ).weights_gb


class TestTheInput:
    @pytest.mark.parametrize("quant", PRICED)
    def test_the_format_gets_an_input_under_its_own_name(self, quant):
        inp = _build_hardware_fit_input(_cfg(quant))
        assert inp is not None
        assert inp.quant == quant

    @pytest.mark.parametrize("quant", PRICED)
    def test_the_format_is_budgeted_the_way_8bit_is(self, quant):
        fmt, eight = _build_hardware_fit_input(_cfg(quant)), _build_hardware_fit_input(_cfg("8bit"))
        assert fmt.peft == eight.peft == "lora"
        assert (fmt.params_b, fmt.seq_len, fmt.batch_size, fmt.optimizer) == (
            eight.params_b,
            eight.seq_len,
            eight.batch_size,
            eight.optimizer,
        )

    @pytest.mark.parametrize("quant", HQQ)
    def test_hqq_still_gets_no_input(self, quant):
        assert _build_hardware_fit_input(_cfg(quant)) is None


class TestThePrices:
    @pytest.mark.parametrize(
        "quant, bytes_per_param",
        [("gptq", 0.55), ("awq", 0.55), ("aqlm", 0.5), ("eetq", 1.0), ("fp8", 2.0)],
    )
    def test_the_weights_are_priced_at_the_format_s_figure(self, quant, bytes_per_param):
        assert _weights_gb(quant) == pytest.approx(20.0 * bytes_per_param)

    def test_fp8_weights_are_priced_as_bf16(self):
        assert _weights_gb("fp8") == pytest.approx(_weights_gb("none"))
        assert _weights_gb("fp8") == pytest.approx(_weights_gb("mxfp4"))


def _fp8_checkpoint(path: Path) -> tuple[str, int]:
    """A two-layer Llama whose linear weights are FP8 on disk, written with
    transformers' own ``Fp8Quantize``. Returns the path and its parameter count."""
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers.integrations.finegrained_fp8 import Fp8Quantize

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=128,  # the default 128 x 128 block must divide every linear
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path)
    params = sum(p.numel() for p in model.parameters())

    quant_config = transformers.FineGrainedFP8Config()
    quantize = Fp8Quantize(type("Quantizer", (), {"quantization_config": quant_config})())
    linears = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    }
    weights_file = path / "model.safetensors"
    tensors = {}
    for key, value in load_file(weights_file).items():
        tensors.update(quantize.convert({key: value}) if key in linears else {key: value})
    save_file(tensors, weights_file, metadata={"format": "pt"})
    # The checkpoint must really be FP8, or the load below proves nothing.
    assert sum(t.dtype == torch.float8_e4m3fn for t in tensors.values()) == len(linears) == 14

    config_file = path / "config.json"
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["quantization_config"] = quant_config.to_dict()
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    return str(path), params


@pytest.fixture(scope="module")
def fp8_checkpoint(tmp_path_factory) -> tuple[str, int]:
    """One checkpoint for both tests below: neither writes to it."""
    return _fp8_checkpoint(tmp_path_factory.mktemp("fp8") / "base")


def _skip_below_the_fp8_minimum() -> None:
    """#1731: below this version transformers cannot load a dense base with an
    FP8 config (its quantizer raises in ``update_tp_plan``), and Soup refuses
    ``fp8`` when it builds the config."""
    import transformers
    from packaging.version import Version

    from soup_cli.utils.quant_menu import FP8_MIN_TRANSFORMERS

    if Version(transformers.__version__) < Version(FP8_MIN_TRANSFORMERS):
        pytest.skip(f"quantization: fp8 needs transformers >= {FP8_MIN_TRANSFORMERS}")


class TestAnFp8BaseOnceLoaded:
    """The figure in the table is what a load through the real config holds."""

    def test_the_loader_s_config_dequantizes(self, fp8_checkpoint):
        import transformers

        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        _skip_below_the_fp8_minimum()
        base, _ = fp8_checkpoint
        quant = build_quantization_config_for_loader(tcfg=_cfg("fp8").training, base=base)
        assert isinstance(quant, transformers.FineGrainedFP8Config)
        assert quant.dequantize is True

    def test_a_loaded_fp8_base_holds_the_bytes_per_parameter_the_table_charges(
        self, fp8_checkpoint
    ):
        """Without a GPU transformers dequantizes an FP8 load whatever the flag
        says, so on a CPU box this test alone cannot tell the flag's value; the
        test above pins it."""
        import torch
        import transformers

        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        _skip_below_the_fp8_minimum()
        base, params = fp8_checkpoint
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype="auto",  # what Soup passes for a frozen base
            quantization_config=build_quantization_config_for_loader(
                tcfg=_cfg("fp8").training, base=base
            ),
        )
        assert sum(p.numel() for p in model.parameters()) == params
        assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
        held = sum(p.numel() * p.element_size() for p in model.parameters())
        assert held / params == 2.0
        assert _BYTES_PER_PARAM_BY_QUANT["fp8"] == held / params
        assert _BYTES_PER_PARAM_BY_QUANT["fp8"] == _BYTES_PER_PARAM_BY_QUANT["none"]


@pytest.mark.parametrize("quant", [*PRICED, "none"])
class TestTheGate:
    """``none`` runs beside the formats as the control."""

    def test_a_card_too_small_for_the_weights_is_refused(self, quant, console):
        with pytest.raises(typer.Exit) as exit_info:
            _hardware_fit_preflight(_cfg(quant), SMALL_CARD, allow_oom_attempt=False)
        assert exit_info.value.exit_code == 1
        assert len(console.printed) == 1
        assert "Predicted peak VRAM" in console.printed[0].renderable

    def test_allow_oom_attempt_warns_and_launches(self, quant, console):
        _hardware_fit_preflight(_cfg(quant), SMALL_CARD, allow_oom_attempt=True)
        assert len(console.printed) == 1
        assert "--allow-oom-attempt set" in console.printed[0].renderable

    def test_a_card_that_fits_is_let_through_without_a_word(self, quant, console):
        _hardware_fit_preflight(_cfg(quant), LARGE_CARD, allow_oom_attempt=False)
        assert console.printed == []


def test_a_packed_base_fits_where_the_same_base_unquantized_does_not(console):
    # 24 GB: 11 GB of gptq weights fit, 40 GB of bf16 weights do not.
    card = {"memory_total_bytes": int(24e9)}
    _hardware_fit_preflight(_cfg("gptq"), card, allow_oom_attempt=False)
    assert console.printed == []
    with pytest.raises(typer.Exit):
        _hardware_fit_preflight(_cfg("none"), card, allow_oom_attempt=False)


def _refusal(console, quant: str, *, allow_oom_attempt: bool = False) -> str:
    try:
        _hardware_fit_preflight(_cfg(quant), SMALL_CARD, allow_oom_attempt=allow_oom_attempt)
    except typer.Exit:
        pass
    return console.printed[-1].renderable


FP8_NOTE = "An FP8 base is dequantized on load, so this estimate prices the bf16 model.\n"
MXFP4_NOTE = "An MXFP4 base is dequantized on load, so this estimate prices the bf16 model.\n"


class TestTheDequantizedSentence:
    def test_fp8_and_none_are_refused_at_the_same_prediction(self, console):
        fp8, none = _refusal(console, "fp8"), _refusal(console, "none")
        assert "Predicted peak VRAM" in none
        assert fp8.replace(FP8_NOTE, "") == none

    @pytest.mark.parametrize("allow_oom_attempt", [False, True])
    def test_the_fp8_panel_says_the_base_is_dequantized(self, console, allow_oom_attempt):
        panel = _refusal(console, "fp8", allow_oom_attempt=allow_oom_attempt)
        assert panel.count(FP8_NOTE) == 1
        assert "MXFP4" not in panel

    def test_the_mxfp4_panel_keeps_its_own_name(self, console):
        panel = _refusal(console, "mxfp4")
        assert panel.count(MXFP4_NOTE) == 1
        assert "FP8" not in panel

    @pytest.mark.parametrize("quant", STAY_PACKED)
    def test_a_format_that_stays_packed_does_not_get_the_sentence(self, console, quant):
        panel = _refusal(console, quant)
        assert "Predicted peak VRAM" in panel
        assert "dequantized" not in panel
        assert " GB\n\nReduce batch_size" in panel


def _plain(printed) -> str:
    return Text.from_markup(printed).plain


class TestTheSkipLine:
    @pytest.mark.parametrize("quant", HQQ)
    @pytest.mark.parametrize("card", [SMALL_CARD, LARGE_CARD])
    @pytest.mark.parametrize("allow_oom_attempt", [False, True])
    def test_hqq_is_let_through_with_one_line_naming_the_format(
        self, console, quant, card, allow_oom_attempt
    ):
        _hardware_fit_preflight(_cfg(quant), card, allow_oom_attempt=allow_oom_attempt)
        assert [_plain(line) for line in console.printed] == [
            f"Pre-flight skipped: no memory model for quantization: {quant}"
        ]

    @pytest.mark.parametrize("quant", [*PRICED, "none", "4bit", "8bit", "mxfp4"])
    def test_a_priced_format_never_gets_the_line(self, console, quant):
        _hardware_fit_preflight(_cfg(quant), LARGE_CARD, allow_oom_attempt=False)
        assert console.printed == []

    @pytest.mark.parametrize("gpu_info", [{}, {"memory_total_bytes": 0}, None])
    def test_no_line_without_a_card_to_check_against(self, console, gpu_info):
        _hardware_fit_preflight(_cfg("hqq:4bit"), gpu_info, allow_oom_attempt=False)
        assert console.printed == []

    @pytest.mark.parametrize("quant", ["gptq", "none"])
    def test_the_other_reason_to_skip_stays_silent(self, console, quant):
        # The line is about the format. batch_size: auto is skipped as before.
        cfg = _cfg(quant, batch_size="auto")
        assert _build_hardware_fit_input(cfg) is None
        _hardware_fit_preflight(cfg, SMALL_CARD, allow_oom_attempt=False)
        assert console.printed == []

    def test_hqq_with_batch_size_auto_still_gets_the_line(self, console):
        _hardware_fit_preflight(
            _cfg("hqq:4bit", batch_size="auto"), SMALL_CARD, allow_oom_attempt=False
        )
        assert len(console.printed) == 1
