"""Native Arrow-cache lifecycle; optional reader installed by mlx-cache CI."""

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm import load  # noqa: E402

from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402
from tests.test_mlx_dora_native import _config, _selective_fixture  # noqa: E402


@pytest.mark.parametrize("mode", ["train_field", "responses_only"])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_real_preprocess_cache_loss_training_and_reload(
    tmp_path, monkeypatch, mode, quantized,
):
    from datasets import load_from_disk
    from mlx_lm.tuner.datasets import CacheDataset
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.trainer.mlx_masking import MaskedChatDataset
    from soup_cli.trainer.mlx_pretokenized import labeled_iterate_batches, labeled_loss

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    base = tmp_path / "base"
    messages, ids = _selective_fixture(base, quantized)
    cfg = _config(base, tmp_path / "out", dora=False, quantized=quantized)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0.0
    cfg.data.train_on_responses_only = mode != "train_field"
    cfg.data.train_on_messages_with_train_field = mode == "train_field"
    raw = [{"messages": messages}] * 4
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    cfg.data.train = str(data)
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    cache = tmp_path / "cache"
    result = CliRunner().invoke(app, ["data", "preprocess", str(config),
                                      "--output", str(cache), "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    (cache,) = [path for path in cache.iterdir() if path.is_dir()]
    cached = list(load_from_disk(str(cache)))
    assert len(cached) == 4 and cached[0]["input_ids"] == ids
    uncached = MaskedChatDataset(raw, load(str(base))[1], train_field=mode == "train_field")
    raw_ids, raw_mask = uncached.process(raw[0])
    assert raw_ids == ids
    if mode == "train_field":
        assert cached[0]["labels"] == [token if keep else -100
                                       for token, keep in zip(raw_ids, raw_mask)]
    else:
        # The producer freezes the shared Transformers objective. Native raw
        # template heuristics may differ; the consumer must retain stored labels.
        from transformers import AutoTokenizer

        from soup_cli.data.loss_mask import build_assistant_only_labels

        shared = build_assistant_only_labels(
            messages, AutoTokenizer.from_pretrained(str(base)), max_length=64,
        )
        assert cached[0]["labels"] == shared["labels"]
    cfg.data.format = "pre_tokenized"
    cfg.data.tokenized_path = str(cache)
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": []})
    assert list(wrapper._dataset["train"]) == cached
    frozen = dict(tree_flatten(wrapper.model.parameters()))
    mx.eval(frozen)
    wrapper._apply_lora(wrapper.model)
    wrapper.model.eval()
    batch, labels = next(iter(labeled_iterate_batches(
        CacheDataset(wrapper._dataset["train"]), batch_size=2, max_seq_length=64,
    )))
    assert batch[0, :len(ids)].tolist() == ids
    assert labels[0, :len(ids)].tolist() == cached[0]["labels"]
    assert (labels[:, len(ids):] == -100).all().item()  # PAD=EOS cannot create targets.

    def oracle(model):
        logits = model(batch[:, :-1]).astype(mx.float32)
        selected = labels[:, 1:] != -100
        targets = mx.maximum(labels[:, 1:], 0)
        ce = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, targets[..., None], axis=-1,
        ).squeeze(-1)
        return (ce * selected).sum() / selected.sum()

    actual, gradients = nn.value_and_grad(wrapper.model, labeled_loss)(
        wrapper.model, batch, labels,
    )
    expected, expected_gradients = nn.value_and_grad(wrapper.model, oracle)(wrapper.model)
    assert actual[1].item() == 2 * sum(value != -100 for value in cached[0]["labels"][1:])
    assert actual[0].item() == pytest.approx(expected.item(), abs=2e-6)
    expected_flat = dict(tree_flatten(expected_gradients))
    for name, value in tree_flatten(gradients):
        assert mx.allclose(value, expected_flat[name], atol=2e-6, rtol=2e-6).item(), name
    del wrapper

    observed = {}
    setup = MLXSFTTrainerWrapper.setup

    def no_tokenization(*args, **kwargs):
        raise AssertionError("cached training must not tokenize or render")

    def observe_setup(self, dataset):
        setup(self, dataset)
        observed["wrapper"] = self
        # Observe the real loaded native tokenizer; never replace model/trainer outputs.
        self.tokenizer.encode = no_tokenization
        self.tokenizer.apply_chat_template = no_tokenization

    monkeypatch.setattr(MLXSFTTrainerWrapper, "setup", observe_setup)
    config.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    trained_wrapper = observed["wrapper"]
    trained_wrapper.model.eval()
    restored, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    restored.eval()
    trained = dict(tree_flatten(restored.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        assert mx.array_equal(trained.get(f"{parent}.linear.{leaf}", trained.get(name)),
                              value).item(), name
    weights = mx.load(str(tmp_path / "out" / "adapters.safetensors"))
    assert any(mx.abs(value).sum().item() > 0 for name, value in weights.items()
               if name.endswith("lora_b"))
    assert mx.allclose(restored(batch), trained_wrapper.model(batch), atol=1e-5).item()
    assert mx.abs(restored(batch) - load(str(base))[0](batch)).max().item() > 1e-4
    records = [json.loads(line) for line in (tmp_path / "out" / "rewind.jsonl").read_text()
               .splitlines()]
    steps = [record for record in records if "row_tokens" in record]
    assert steps
    expected_count = sum(value != -100 for value in cached[0]["labels"][1:])
    assert all(all(value == expected_count for value in record["row_tokens"]) for record in steps)
    metadata = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert metadata["pre_tokenized"] is True


def test_native_explicit_cached_targets_and_rewind_share_the_numerical_objective():
    from types import SimpleNamespace

    from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset, labeled_loss
    from soup_cli.trainer.rewind_mlx import ROW_LOSS_KEY, ROW_TOKENS_KEY, make_rewind_loss

    class FixedTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.arange(64, dtype=mx.float32).reshape(8, 8) / 10

        def __call__(self, tokens):
            return self.weight[tokens]

    row = {"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 6, 2]}
    data = PretokenizedDataset([row], vocab_size=8, max_length=4)
    tokens, targets = data.process(data[0])
    batch, labels = mx.array([tokens]), mx.array([targets])
    model = FixedTable()
    model.train()
    state = SimpleNamespace(failed=False, optimizer_state={}, mark_pending=lambda: None)
    recorded = make_rewind_loss(state, kind="labels")
    logits = model(batch[:, :-1])
    expected = ((mx.logsumexp(logits, axis=-1)[0, 1] - logits[0, 1, 6]) +
                (mx.logsumexp(logits, axis=-1)[0, 2] - logits[0, 2, 2])) / 2
    loss, count = labeled_loss(model, batch, labels)
    tracked, tracked_count = recorded(model, batch, labels)
    assert count.item() == tracked_count.item() == 2
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert tracked.item() == pytest.approx(expected.item(), abs=1e-6)
    assert state.optimizer_state[ROW_LOSS_KEY].item() == pytest.approx(expected.item(), abs=1e-6)
    assert state.optimizer_state[ROW_TOKENS_KEY].item() == 2
    _, actual_grad = nn.value_and_grad(model, labeled_loss)(model, batch, labels)
    _, rewind_grad = nn.value_and_grad(model, recorded)(model, batch, labels)
    assert mx.allclose(actual_grad["weight"], rewind_grad["weight"], atol=1e-6).item()
    assert actual_grad["weight"][3, 6].item() < 0
    assert actual_grad["weight"][3, 4].item() > 0
