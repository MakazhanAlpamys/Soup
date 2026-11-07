"""Streaming does not install the resident CCE patch; refuse the dead switch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig


def _config(**training) -> SoupConfig:
    return SoupConfig(
        base="local/model", data={"train": "train.jsonl"},
        training={"batch_size": 1, "quantization": "none", **training},
    )


@pytest.mark.parametrize("task", ["sft", "dpo", "kto", "orpo", "simpo"])
def test_streamed_cce_refused(task):
    with pytest.raises(ValueError, match="stream_layers.*use_cut_ce"):
        SoupConfig(
            base="local/model", task=task, data={"train": "train.jsonl"},
            training={"batch_size": 2, "stream_layers": True, "use_cut_ce": True},
        )


def test_public_yaml_refuses_streamed_cce():
    with pytest.raises(ValueError, match="stream_layers.*use_cut_ce"):
        load_config_from_string(
            "base: local/model\ndata:\n  train: train.jsonl\ntraining:\n"
            "  batch_size: 1\n  stream_layers: true\n  use_cut_ce: true\n"
        )


def test_config_file_diagnostic(tmp_path, capsys):
    from soup_cli.config.loader import load_config
    from tests.conftest import strip_ansi

    path = tmp_path / "soup.yaml"
    path.write_text(
        "base: local/model\ndata:\n  train: train.jsonl\ntraining:\n"
        "  batch_size: 1\n  stream_layers: true\n  use_cut_ce: true\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        load_config(path)
    assert exc.value.code == 1
    output = strip_ansi(capsys.readouterr().out)
    assert "stream_layers" in output and "use_cut_ce" in output
    assert "resident" in output


def test_supported_controls():
    assert _config(stream_layers=True).training.stream_layers
    assert not _config(stream_layers=True, use_cut_ce=False).training.use_cut_ce
    assert _config(use_cut_ce=True).training.use_cut_ce


def test_real_streamed_setup_uses_ordinary_ce(tmp_path, monkeypatch):
    """CPU model, real sharding/setup/forward/backward, no CCE or model download.

    Bypass the new schema gate deliberately to inspect the formerly accepted
    implementation. Only tokenizer loading and the cache destination are isolated;
    model construction, shards, streaming and the loss are real.
    """
    import torch
    import torch.nn.functional as functional
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    torch.manual_seed(7)
    model_config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )
    weights = tmp_path / "model"
    LlamaForCausalLM(model_config).save_pretrained(weights)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(pad_token=None, eos_token="</s>"),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_shard.resolve_shard_dir", lambda *_: str(tmp_path / "shards"),
    )

    def unexpected_patch(*args, **kwargs):
        pytest.fail("streamed setup now reaches CCE: reassess the refusal and VRAM budget")

    monkeypatch.setattr("soup_cli.utils.cut_ce.apply_cut_ce", unexpected_patch)

    class Wrapper(StreamingSetupMixin):
        device = "cpu"
        _trust_remote_code = False

    cfg = _config(stream_layers=True)
    cfg.base = str(weights)
    cfg.training = cfg.training.model_copy(update={"use_cut_ce": True})
    wrapper = Wrapper()
    try:
        wrapper._setup_streaming_transformers(cfg, cfg.training)
        assert wrapper._stream_runtime.stats()["n_layers"] == 2
        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        output = wrapper.model(input_ids=inputs, labels=inputs)
        assert output.logits.shape == (1, 5, 32)
        oracle = functional.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, 32), inputs[:, 1:].reshape(-1),
        )
        torch.testing.assert_close(output.loss, oracle)
        output.loss.backward()
        gradients = [
            parameter.grad for name, parameter in wrapper.model.named_parameters()
            if "lora_" in name and parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    finally:
        wrapper._close_stream_runtime()
