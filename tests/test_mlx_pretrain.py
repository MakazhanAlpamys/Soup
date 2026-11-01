"""Portable contracts for native raw-text pretraining on MLX."""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string


class RawTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def __init__(self, tokens=None):
        self.tokens = tokens
        self.seen = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is True
        self.seen.append(text)
        return list(self.tokens if self.tokens is not None else [1, *map(int, text.split())])

    def apply_chat_template(self, *args, **kwargs):
        pytest.fail("raw pretraining must never apply a chat template")


@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_public_config_and_dispatch_support_native_pretraining(quantization: str) -> None:
    from soup_cli.trainer.dispatch import build_trainer

    cfg = load_config_from_string(
        "base: ./local-mlx-model\nbackend: mlx\ntask: pretrain\n"
        "data: {train: corpus.jsonl, format: plaintext, max_length: 64}\n"
        f"training: {{quantization: {quantization}, batch_size: 1}}\n"
    )
    wrapper = build_trainer(cfg)
    assert type(wrapper).__name__ == "MLXPretrainTrainerWrapper"
    assert wrapper.config.task == "pretrain"
    from soup_cli.config.schema import SoupConfig

    assert cfg.data.train_on_responses_only is False
    assert cfg.training.rewind_log is False
    assert SoupConfig(**cfg.model_dump()).model_dump() == cfg.model_dump()


def test_native_pretraining_has_its_own_route() -> None:
    from soup_cli.trainer.mlx_routing import get_mlx_trainer

    assert get_mlx_trainer("pretrain").__name__ == "MLXPretrainTrainerWrapper"


def test_exact_overlap_tail_and_pad_equals_eos():
    from soup_cli.data.mlx_pretrain import batch_windows, prepare_windows

    windows, stats = prepare_windows([{"text": "10 11 12"}], RawTokenizer(), 3)
    assert [(w.tokens, w.document) for w in windows] == [((1, 10, 11), 0), ((11, 12, 2), 0)]
    assert batch_windows(windows, 2) == (
        [[1, 10], [11, 12]], [[10, 11], [12, 2]], [[1., 1.], [1., 1.]],
    )
    windows, stats = prepare_windows([{"text": "10 11 12 13"}, {"text": "20"}],
                                     RawTokenizer(), 3)
    assert [(w.tokens, w.document) for w in windows] == [
        ((1, 10, 11), 0), ((11, 12, 13), 0), ((13, 2), 0), ((1, 20, 2), 1),
    ]
    assert batch_windows(windows[-2:], 2) == (
        [[13, 2], [1, 20]], [[2, 2], [20, 2]], [[1., 0.], [1., 1.]],
    )
    assert stats["supervised_tokens"] == 7
    assert stats["context_only_tokens"] == 2
    assert stats["tokenized_tokens"] == 9
    with pytest.raises(ValueError, match="truncate"):
        batch_windows(windows, 2, width=1)


@pytest.mark.parametrize("tokens,eos,expected", [
    ([1, 10, 2], 2, (1, 10, 2)), ([10, 11], None, (10, 11)),
    ([10], 2, (10, 2)), ([], 2, None), ([10], None, None),
])
def test_special_tokens_and_zero_target_policy(tokens, eos, expected):
    from soup_cli.data.mlx_pretrain import prepare_windows

    tokenizer = RawTokenizer(tokens)
    tokenizer.eos_token_id = eos
    if expected is None:
        with pytest.raises(ValueError, match="no supervised next-token targets"):
            prepare_windows([{"text": "<|assistant|> literal text"}], tokenizer, 4)
    else:
        windows, _ = prepare_windows([{"text": "<|assistant|> literal text"}], tokenizer, 4)
        assert windows[0].tokens == expected
    assert tokenizer.seen == ["<|assistant|> literal text"]


def test_real_files_preserve_documents_and_report_drops(tmp_path, capsys):
    import json

    from soup_cli.config.schema import DataConfig
    from soup_cli.data.mlx_pretrain import file_sha256, load_pretrain_corpus

    train = tmp_path / "train.txt"
    train.write_text("\ufeff  first  \n\n second \n", encoding="utf-8")
    val = tmp_path / "val.jsonl"
    text = "\n  def f():\n    return 1\n"
    val.write_text("\ufeff" + json.dumps({"text": text}) +
                   '\n{}\n{"text":3}\n{"text":"  "}\nnull\nnot-json\n', encoding="utf-8")
    data = load_pretrain_corpus(DataConfig(train=str(train), validation=str(val),
                                           format="plaintext", val_split=0))
    assert data["train"] == [{"text": "first"}, {"text": "second"}]
    assert data["val"] == [{"text": text}]
    stats = data["pretrain_data_stats"]
    assert stats["source"]["source_sha256"] == file_sha256(train)
    assert stats["validation_source"]["source_documents"] == 6
    assert stats["validation_source"]["accepted_documents"] == 1
    assert stats["validation_source"]["dropped_documents"] == 5
    from tests.conftest import strip_ansi

    diagnostics = strip_ansi(capsys.readouterr().out).lower()
    assert "invalid json on line 6" in diagnostics
    assert "json null, not an object" in diagnostics
    assert "dropped" in diagnostics


def test_document_split_happens_before_windows(tmp_path):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.mlx_pretrain import prepare_windows

    path = tmp_path / "corpus.txt"
    path.write_text("10 11 12 13 14\n20 21 22 23 24\n")
    data = load_dataset(DataConfig(train=str(path), format="plaintext", val_split=.5),
                        native_pretrain=True)
    train, _ = prepare_windows(data["train"], RawTokenizer(), 3)
    val, _ = prepare_windows(data["val"], RawTokenizer(), 3)
    assert {t for w in train for t in w.tokens} - {1, 2} == {10, 11, 12, 13, 14}
    assert {t for w in val for t in w.tokens} - {1, 2} == {20, 21, 22, 23, 24}


def test_missing_and_unusable_corpora_are_refused(tmp_path):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.mlx_pretrain import load_pretrain_corpus

    with pytest.raises(FileNotFoundError, match="not found"):
        load_pretrain_corpus(DataConfig(train=str(tmp_path / "absent.txt")))
    path = tmp_path / "empty.jsonl"
    path.write_text('{"text":null}\n{}\n')
    with pytest.raises(ValueError, match="training split has no usable"):
        load_pretrain_corpus(DataConfig(train=str(path), val_split=0))


@pytest.mark.parametrize("count,batch,accum,epochs", [(1, 4, 3, 1), (7, 2, 2, 3),
                                                           (8, 2, 2, 2), (3, 2, 5, 1)])
def test_update_plan_consumes_every_window_once_per_epoch(count, batch, accum, epochs):
    import math

    from soup_cli.data.mlx_pretrain import plan_updates

    plan = plan_updates(count, batch, accum, epochs, 17)
    assert len(plan) == epochs * math.ceil(math.ceil(count / batch) / accum)
    for epoch in range(1, epochs + 1):
        indices = [i for e, group in plan if e == epoch for mb in group for i in mb]
        assert sorted(indices) == list(range(count))
    assert plan == plan_updates(count, batch, accum, epochs, 17)
    if count > 2:
        assert plan != plan_updates(count, batch, accum, epochs, 18)


@pytest.mark.parametrize("setting,name", [
    ("modality: vision\n", "modality"), ("data: {train: corpus.txt, format: chatml}\n", "format"),
    ("training: {lora: {r: 0}}\n", "lora.r"),
    ("training: {packing: true}\n", "packing"),
    ("training: {lora: {use_dora: true}}\n", "use_dora"),
    ("training: {gradient_checkpointing: true}\n", "gradient_checkpointing"),
    ("training: {epochs: 1.5}\n", "epochs"),
    ("training: {quantization: 8bit}\n", "8bit"),
    ("data: {train: corpus.txt, format: plaintext, streaming: true}\n", "streaming"),
])
def test_initial_scope_has_named_refusals(setting, name):
    import yaml

    from soup_cli.config.schema import SoupConfig

    raw = dict(base="local", backend="mlx", task="pretrain",
               data=dict(train="corpus.txt", format="plaintext"))
    patch = yaml.safe_load(setting)
    for key, value in patch.items():
        if isinstance(value, dict):
            raw.setdefault(key, {}).update(value)
        else:
            raw[key] = value
    with pytest.raises(ValueError, match=name):
        SoupConfig(**raw)


def test_explicit_validation_and_eval_steps_are_exclusive_to_native_pretrain():
    from soup_cli.config.schema import SoupConfig

    raw = dict(base="local", backend="mlx", task="pretrain",
               data=dict(train="corpus.txt", validation="val.txt", format="plaintext",
                         val_split=0), training=dict(eval_steps=2))
    assert SoupConfig(**raw).training.eval_steps == 2
    with pytest.raises(ValueError, match="data.validation"):
        SoupConfig(**{**raw, "backend": "transformers"})


def test_missing_mlx_and_non_apple_errors_are_actionable(monkeypatch):
    import sys

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_pretrain import MLXPretrainTrainerWrapper
    from soup_cli.utils import mlx

    cfg = SoupConfig(base="local", backend="mlx", task="pretrain",
                     data=dict(train="corpus.txt", format="plaintext"))
    trainer = MLXPretrainTrainerWrapper(cfg)
    monkeypatch.setattr(mlx, "is_apple_silicon", lambda: False)
    with pytest.raises(RuntimeError, match="Apple Silicon Mac"):
        trainer.setup({"train": [{"text": "x"}]})
    monkeypatch.setattr(mlx, "is_apple_silicon", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(ImportError, match=r"soup-cli\[mlx\]"):
        trainer.setup({"train": [{"text": "x"}]})


def test_required_validation_and_invalid_numeric_limits_fail(tmp_path):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.mlx_pretrain import load_pretrain_corpus, prepare_windows

    train, val = tmp_path / "train.txt", tmp_path / "empty.txt"
    train.write_text("text\n")
    val.write_text(" \n")
    with pytest.raises(ValueError, match="required validation split"):
        load_pretrain_corpus(DataConfig(train=str(train), validation=str(val)))
    for value in (0, 1, True, 2.5):
        with pytest.raises(ValueError, match="max_length"):
            prepare_windows([{"text": "1"}], RawTokenizer(), value)


@pytest.mark.parametrize("tokens,expected", [([1, 10], [(1, 10, 2)]),
    ([1, 10, 11, 12, 13, 14], [(1, 10, 11), (11, 12, 13), (13, 14, 2)])])
def test_short_and_exact_boundary_windows_have_no_extra_empty_window(tokens, expected):
    from soup_cli.data.mlx_pretrain import prepare_windows

    windows, stats = prepare_windows([{"text": "raw"}], RawTokenizer(tokens), 3)
    assert [w.tokens for w in windows] == [tuple(w) for w in expected]
    assert stats["supervised_tokens"] == len(tokens)
