"""#3371: ``soup data preprocess`` preserves both train and validation splits."""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from soup_cli.config.schema import DataConfig
from soup_cli.utils import data_pipeline as dp
from soup_cli.utils.pretokenized import maybe_load_pretokenized
from tests.test_issue1038_preprocess_to_pretokenized import _BASE, _ROW, _tokenizer


def _legacy_dataset_key(dcfg: DataConfig) -> dict:
    """The pre-#3371 dataset_key shape containing only ``_DATASET_KEY_FIELDS``."""
    return {name: getattr(dcfg, name, None) for name in dp._DATASET_KEY_FIELDS}


class TestConditionalSplitLayoutMarker:
    def test_val_split_zero_retains_exact_old_key_input(self):
        dcfg = DataConfig(train="./d.jsonl", val_split=0.0)
        key_input = dp.preprocess_dataset_key_input(dcfg)
        parsed = json.loads(key_input)

        assert "split_layout" not in parsed
        expected_legacy = json.dumps(_legacy_dataset_key(dcfg), sort_keys=True)
        assert key_input == expected_legacy

    def test_val_split_positive_adds_marker_to_key_and_changes_cache_key(self):
        dcfg = DataConfig(train="./d.jsonl", val_split=0.2)
        key_input = dp.preprocess_dataset_key_input(dcfg)
        parsed = json.loads(key_input)

        assert parsed["split_layout"] == dp._SPLIT_LAYOUT_V1
        legacy_json = json.dumps(_legacy_dataset_key(dcfg), sort_keys=True)
        common = dict(
            tokenizer_name=_BASE,
            max_length=128,
            format_name="chatml",
            chat_template=None,
            mask_mode="responses_only",
            task="sft",
        )
        new_key = dp.make_preprocess_cache_key(dataset_path=key_input, **common)
        old_key = dp.make_preprocess_cache_key(dataset_path=legacy_json, **common)
        assert new_key != old_key


class TestLegacyCacheRejectionAndCompatibility:
    def test_legacy_train_only_cache_with_val_split_rejected_with_migration_message(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "legacy-cache"
        cache_dir.mkdir(parents=True)

        src_cfg = DataConfig(train="./d.jsonl", format="chatml", max_length=128, val_split=0.2)
        legacy_dict = _legacy_dataset_key(src_cfg)
        legacy_json = json.dumps(legacy_dict, sort_keys=True)
        legacy_cache_key = dp.make_preprocess_cache_key(
            dataset_path=legacy_json,
            tokenizer_name=_BASE,
            max_length=128,
            format_name="chatml",
            chat_template=None,
            mask_mode="responses_only",
            task="sft",
        )
        (cache_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_key": legacy_cache_key,
                    "row_count": 4,
                    "tokenizer_name": _BASE,
                    "max_length": 128,
                    "format": "chatml",
                    "chat_template": None,
                    "mask_mode": "responses_only",
                    "task": "sft",
                    "key_schema": dp._PREPROCESS_TOKENIZE_SCHEMA,
                    "dataset_key": legacy_dict,
                }
            ),
            encoding="utf-8",
        )

        gate_cfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path="legacy-cache",
            max_length=128,
            val_split=0.2,
        )
        assert (
            dp.preprocess_dataset_key_diff(
                legacy_dict, dp.preprocess_dataset_key_input(gate_cfg)
            )
            == ["split_layout"]
        )

        with pytest.raises(ValueError, match="cache hash mismatch") as exc:
            maybe_load_pretokenized(gate_cfg, _BASE, Console(), task="sft")

        msg = str(exc.value)
        assert "before validation rows were preserved" in msg
        assert "soup data preprocess" in msg

    def test_legacy_flat_cache_with_val_split_zero_still_loads(
        self, tmp_path, monkeypatch
    ):
        datasets = pytest.importorskip("datasets")
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / "flat-legacy"
        rows = [
            {"input_ids": [1, 3, 2], "labels": [-100, 3, 2], "attention_mask": [1, 1, 1]},
            {"input_ids": [1, 4, 2], "labels": [-100, 4, 2], "attention_mask": [1, 1, 1]},
        ]
        datasets.Dataset.from_list(rows).save_to_disk(str(cache_dir))

        src_cfg = DataConfig(train="./d.jsonl", format="chatml", max_length=128, val_split=0.0)
        legacy_dict = _legacy_dataset_key(src_cfg)
        legacy_json = json.dumps(legacy_dict, sort_keys=True)
        cache_key = dp.make_preprocess_cache_key(
            dataset_path=legacy_json,
            tokenizer_name=_BASE,
            max_length=128,
            format_name="chatml",
            chat_template=None,
            mask_mode="responses_only",
            task="sft",
        )
        (cache_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "row_count": len(rows),
                    "tokenizer_name": _BASE,
                    "max_length": 128,
                    "format": "chatml",
                    "chat_template": None,
                    "mask_mode": "responses_only",
                    "task": "sft",
                    "key_schema": dp._PREPROCESS_TOKENIZE_SCHEMA,
                    "dataset_key": legacy_dict,
                }
            ),
            encoding="utf-8",
        )

        gate_cfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path="flat-legacy",
            max_length=128,
            val_split=0.0,
        )
        loaded = maybe_load_pretokenized(gate_cfg, _BASE, Console(), task="sft")
        assert loaded is not None
        train_ds, eval_ds = loaded
        assert len(train_ds) == 2
        assert eval_ds is None


class TestPreprocessPreservesTrainAndValSplits:
    def test_end_to_end_preprocess_and_gate_preserve_train_and_val_splits(
        self, tmp_path, monkeypatch
    ):
        transformers = pytest.importorskip("transformers")
        datasets = pytest.importorskip("datasets")
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.chdir(tmp_path)
        (tmp_path / "d.jsonl").write_text(
            "".join(json.dumps(_ROW) + "\n" for _ in range(10)),
            encoding="utf-8",
        )
        tok = _tokenizer()
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
        )
        (tmp_path / "soup.yaml").write_text(
            f"base: {_BASE}\n"
            "task: sft\n"
            "data:\n"
            "  train: d.jsonl\n"
            "  format: chatml\n"
            "  max_length: 128\n"
            "  val_split: 0.2\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
        assert result.exit_code == 0, result.output

        (cache_dir,) = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
        meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["dataset_key"]["split_layout"] == dp._SPLIT_LAYOUT_V1
        assert meta["dataset_key"]["val_split"] == 0.2
        assert meta["row_count"] == 8

        on_disk = datasets.load_from_disk(str(cache_dir))
        assert isinstance(on_disk, datasets.DatasetDict)
        assert set(on_disk.keys()) == {"train", "val"}
        assert len(on_disk["train"]) == 8
        assert len(on_disk["val"]) == 2

        gate_cfg = load_config_from_string(
            f"base: {_BASE}\n"
            "task: sft\n"
            "data:\n"
            "  train: d.jsonl\n"
            "  format: pre_tokenized\n"
            f"  tokenized_path: {cache_dir.relative_to(tmp_path).as_posix()}\n"
            "  max_length: 128\n"
            "  val_split: 0.2\n"
            "output: ./out\n"
        )
        loaded = maybe_load_pretokenized(
            gate_cfg.data, gate_cfg.base, Console(), gate_cfg.training, task="sft"
        )
        assert loaded is not None
        train_ds, eval_ds = loaded
        assert len(train_ds) == 8
        assert eval_ds is not None
        assert len(eval_ds) == 2
        assert {"input_ids", "labels", "attention_mask"} <= set(eval_ds.column_names)

    def test_untokenizable_validation_row_refused_and_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi
        from tests.test_issue1180_preprocess_refuses_bad_rows import _STRICT, _TWO_USERS, _ok

        monkeypatch.chdir(tmp_path)
        rows = [_ok(0), _ok(1), _ok(2), _ok(3), _TWO_USERS]
        (tmp_path / "d.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows),
            encoding="utf-8",
        )
        tok = _tokenizer()
        tok.chat_template = _STRICT
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
        )
        (tmp_path / "soup.yaml").write_text(
            f"base: {_BASE}\n"
            "task: sft\n"
            "data:\n"
            "  train: d.jsonl\n"
            "  format: chatml\n"
            "  max_length: 64\n"
            "  val_split: 0.2\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, out
        assert "Val row 1 cannot be tokenized" in out
        assert "Conversation roles must alternate" in out
        assert not (tmp_path / ".soup-tokenized").exists()

    def test_val_split_preserves_dataset_dict_when_validation_yields_zero_rows(
        self, tmp_path, monkeypatch
    ):
        transformers = pytest.importorskip("transformers")
        datasets = pytest.importorskip("datasets")
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.chdir(tmp_path)
        (tmp_path / "d.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
        tok = _tokenizer()
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
        )
        monkeypatch.setattr(
            "soup_cli.data.loader.load_dataset",
            lambda *a, **k: {"train": [{"text": "Paris"}], "val": [{"text": ""}]},
        )
        (tmp_path / "soup.yaml").write_text(
            f"base: {_BASE}\n"
            "task: pretrain\n"
            "data:\n"
            "  train: d.jsonl\n"
            "  format: plaintext\n"
            "  max_length: 128\n"
            "  val_split: 0.2\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
        assert result.exit_code == 0, result.output

        (cache_dir,) = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
        on_disk = datasets.load_from_disk(str(cache_dir))
        assert isinstance(on_disk, datasets.DatasetDict)
        assert set(on_disk.keys()) == {"train", "val"}
        assert len(on_disk["train"]) == 1
        assert len(on_disk["val"]) == 0
        assert on_disk["val"].features == on_disk["train"].features

        gate_cfg = load_config_from_string(
            f"base: {_BASE}\n"
            "task: pretrain\n"
            "data:\n"
            "  train: d.jsonl\n"
            "  format: pre_tokenized\n"
            f"  tokenized_path: {cache_dir.relative_to(tmp_path).as_posix()}\n"
            "  max_length: 128\n"
            "  val_split: 0.2\n"
            "output: ./out\n"
        )
        loaded = maybe_load_pretokenized(
            gate_cfg.data, gate_cfg.base, Console(), gate_cfg.training, task="pretrain"
        )
        assert loaded is not None
        train_ds, eval_ds = loaded
        assert len(train_ds) == 1
        assert eval_ds is None
