"""#1466 — a hand-written ``use_dora: true`` on a GPTQ / AWQ / AQLM / EETQ base
loaded, then failed when the adapter was attached.

peft's LoRA layers for those four formats raise ``<Layer> does not support DoRA
yet`` in their constructor, which runs after the checkpoint download and the
model load. The pairing is now refused when the config is loaded. Autopilot
already keeps DoRA off for the same formats (#1473); both read one set.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string

NO_DORA = ("gptq", "awq", "aqlm", "eetq")


def _yaml(quantization: str, *, use_dora: bool, task: str = "sft") -> str:
    return (
        "base: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ\n"
        f"task: {task}\n"
        "data:\n  train: t.jsonl\n"
        "training:\n"
        f"  quantization: {quantization}\n"
        f"  lora:\n    use_dora: {str(use_dora).lower()}\n"
        "output: out\n"
    )


@pytest.mark.parametrize("quantization", NO_DORA)
def test_dora_on_prequantized_base_is_refused_at_load(quantization):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(quantization, use_dora=True))
    message = str(excinfo.value)
    assert "use_dora" in message
    assert repr(quantization) in message
    # Says what to change, not only what is wrong.
    assert "use_dora: false" in message


@pytest.mark.parametrize("quantization", NO_DORA)
def test_plain_lora_on_prequantized_base_still_loads(quantization):
    cfg = load_config_from_string(_yaml(quantization, use_dora=False))
    assert cfg.training.quantization == quantization
    assert cfg.training.lora.use_dora is False


@pytest.mark.parametrize("quantization", ["4bit", "8bit", "none", "hqq:4bit", "fp8"])
def test_dora_stays_allowed_where_peft_builds_it(quantization):
    """peft has a DoRA variant for bitsandbytes, HQQ and unquantised layers."""
    cfg = load_config_from_string(_yaml(quantization, use_dora=True))
    assert cfg.training.lora.use_dora is True


def test_one_format_set_serves_autopilot_and_the_schema():
    from soup_cli.autopilot.decisions import decide_peft
    from soup_cli.utils.quant_menu import DORA_UNSUPPORTED_FORMATS

    assert DORA_UNSUPPORTED_FORMATS == frozenset(NO_DORA)
    headroom = {"data_size": 200_000, "model_size_b": 7.0, "vram_gb": 80.0}
    for quantization in DORA_UNSUPPORTED_FORMATS:
        assert decide_peft(**headroom, quantization=quantization)["use_dora"] is False


def test_autopilot_yaml_for_a_gptq_base_reloads(tmp_path):
    """What Autopilot writes must pass the validator it now shares a set with."""
    import json

    from soup_cli.autopilot.generate_config import build_soup_config, write_yaml
    from soup_cli.config.loader import load_config

    data_file = tmp_path / "big.jsonl"
    data_file.write_text(
        "".join(
            json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n"
            for i in range(100_001)
        ),
        encoding="utf-8",
    )
    cfg = build_soup_config(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        data_path=str(data_file),
        goal="chat",
        vram_gb=24.0,
    )
    out = tmp_path / "soup.yaml"
    write_yaml(cfg, out)
    reloaded = load_config(out)
    assert reloaded.training.quantization == "gptq"
    assert reloaded.training.lora.use_dora is False


@pytest.mark.parametrize("quantization", ["gptq", "awq"])
def test_the_mlx_backend_keeps_its_own_refusal(quantization):
    """mlx refuses these formats outright; sending the user to `use_dora: false`
    first would only lead them to that second refusal."""
    yaml = _yaml(quantization, use_dora=True).replace("task: sft\n", "task: sft\nbackend: mlx\n")
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(yaml)
    message = str(excinfo.value)
    assert "not supported on the mlx backend" in message
    assert "use_dora: false" not in message


def _on_unsloth(yaml: str) -> str:
    return yaml.replace("task: sft\n", "task: sft\nbackend: unsloth\n")


@pytest.mark.parametrize("quantization", NO_DORA)
def test_the_unsloth_backend_gets_the_dora_refusal(quantization):
    """The early return is for mlx alone: on unsloth the pairing is refused
    with the same message as on transformers."""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_on_unsloth(_yaml(quantization, use_dora=True)))
    message = str(excinfo.value)
    assert repr(quantization) in message
    assert "use_dora: false" in message


@pytest.mark.parametrize("quantization", NO_DORA)
def test_plain_lora_on_the_unsloth_backend_still_loads(quantization):
    cfg = load_config_from_string(_on_unsloth(_yaml(quantization, use_dora=False)))
    assert cfg.backend == "unsloth"
    assert cfg.training.quantization == quantization
