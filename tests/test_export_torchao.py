"""``soup export --format torchao`` on a LoRA adapter directory exports the merged model.

The other export formats merge an adapter into its base before they convert. The
torchao path handed the adapter directory straight to ``from_pretrained``, which
attaches the adapter to its base, and ``save_pretrained`` then wrote the adapter
alone: the output held ``adapter_config.json`` and the LoRA tensors, no model, and
the command reported "TorchAO Export Complete". Real transformers and peft run
here on a tiny model; only torchao (a quantize step that records its model) is
faked.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("peft")
torch = pytest.importorskip("torch")

from soup_cli.cli import app  # noqa: E402
from tests._tiny_hf_models import build_llama, build_tokenizer  # noqa: E402

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """A tiny base model and a LoRA adapter on it that changes q_proj."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    tok = build_tokenizer()
    base_dir = tmp_path / "base"
    build_llama(tok).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)

    # Loaded from its directory, so the adapter records it as base_model_name_or_path.
    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], init_lora_weights=False),
    )
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    merged_q = (
        peft_model.merge_and_unload().model.layers[0].self_attn.q_proj.weight.detach().clone()
    )
    (tmp_path / "q.yaml").write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
    return types.SimpleNamespace(dir=adapter_dir, base=base_dir, merged_q=merged_q)


@pytest.fixture
def quantized(monkeypatch):
    """Stand in for torchao: record the model ``quantize_`` receives, change nothing."""
    from soup_cli.utils import torchao_compat

    seen = []

    def resolve(name):
        if name == "quantize_":
            return lambda model, config: seen.append(model)
        return lambda **kwargs: ("config", name, kwargs)

    monkeypatch.setattr(torchao_compat, "resolve_torchao_class", resolve)
    return seen


def _export(*extra):
    return CliRunner().invoke(
        app,
        ["export", "--model", "adapter", "--format", "torchao", "--quant-config", "q.yaml",
         *extra],
    )


def test_adapter_is_merged_before_quantizing_and_the_output_is_a_model(
    adapter, quantized, tmp_path
):
    result = _export("-o", "out")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = tmp_path / "out"
    files = sorted(p.name for p in out.iterdir())
    assert "adapter_config.json" not in files, files
    assert "config.json" in files, files
    assert any(name.endswith(".safetensors") for name in files), files

    (model,) = quantized
    assert not any("lora_" in name for name, _ in model.named_modules())

    from transformers import AutoModelForCausalLM

    exported = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    q = exported.model.layers[0].self_attn.q_proj.weight
    assert torch.allclose(q, adapter.merged_q, atol=1e-2)
    # The scratch merge is gone; nothing was left next to the adapter.
    assert list(tmp_path.glob(".soup_merge_tmp_*")) == []


def test_default_output_name_still_follows_the_adapter_dir(adapter, quantized, tmp_path):
    result = _export()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = tmp_path / "adapter.torchao.Int4WeightOnly"
    assert (out / "config.json").is_file()


def test_adapter_without_a_recorded_base_asks_for_base(adapter, quantized, tmp_path):
    cfg_path = adapter.dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.pop("base_model_name_or_path", None)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = _export("-o", "out")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "Pass --base" in _plain(result.output)
    assert quantized == []
    assert not (tmp_path / "out").exists()

    result = _export("-o", "out", "--base", str(adapter.base))

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "out" / "config.json").is_file()


def test_failed_quantize_removes_the_scratch_merge(adapter, monkeypatch, tmp_path):
    from soup_cli.utils import torchao_compat

    def resolve(name):
        if name == "quantize_":
            def quantize_(model, config):
                raise RuntimeError("no kernel for this device")
            return quantize_
        return lambda **kwargs: ("config", name, kwargs)

    monkeypatch.setattr(torchao_compat, "resolve_torchao_class", resolve)

    result = _export("-o", "out")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "TorchAO export failed: no kernel for this device" in _plain(result.output)
    assert list(tmp_path.glob(".soup_merge_tmp_*")) == []


def test_full_model_dir_is_exported_as_is(adapter, quantized, tmp_path):
    result = CliRunner().invoke(
        app,
        ["export", "--model", "base", "--format", "torchao", "--quant-config", "q.yaml",
         "-o", "out_base"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "merging" not in _plain(result.output)
    assert (Path("out_base") / "config.json").is_file()
