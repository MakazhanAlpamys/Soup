"""#1217: a load that ends with zero training rows stops `soup train`.

The loader drops rows its format cannot convert, so a format mismatch loads
nothing and still returns cleanly. `soup train --dry-run` then printed
"Data OK: 0 train samples" and "Config valid. Ready to train!", and the real
run loaded the model before failing on an empty dataset. Both now exit 1
with the format and the first drop reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig
from soup_cli.data.loader import last_load_outcome, load_dataset
from tests.conftest import strip_ansi

_BASE = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_CHAT = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text)).strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _loaded_rows(path: str, data_format: str) -> list[dict]:
    config = DataConfig(train=path, format=data_format, val_split=0.0)
    return list(load_dataset(config)["train"])



def _write_config(path: Path, data_file: str, data_format: str) -> None:
    path.write_text(
        f"base: {_BASE}\ntask: sft\ndata:\n  train: {data_file}\n"
        f"  format: {data_format}\n  val_split: 0.0\noutput: ./out\n",
        encoding="utf-8",
    )


def test_dry_run_with_zero_rows_exits_non_zero_naming_format_and_reason(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(Path("data.jsonl"), [{"instruction": "q", "output": "a"}] * 2)
    _write_config(Path("soup.yaml"), "data.jsonl", "chatml")

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    output = _plain(result.output)
    assert result.exit_code == 1, output
    assert "Ready to train" not in output
    assert "Data OK" not in output
    assert (
        "No training rows: data.jsonl loaded 0 train samples as format 'chatml'."
    ) in output
    assert "First: row 0 (as 'chatml'): 'messages'." in output


def test_the_real_run_with_zero_rows_stops_before_the_model_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(Path("data.jsonl"), [{"instruction": "q", "output": "a"}])
    _write_config(Path("soup.yaml"), "data.jsonl", "chatml")

    import soup_cli.trainer.sft as sft

    def _must_not_run(*args, **kwargs):
        raise AssertionError("the trainer was set up for a run with no rows")

    monkeypatch.setattr(sft.SFTTrainerWrapper, "setup", _must_not_run)

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])

    output = _plain(result.output)
    assert result.exit_code == 1, output
    assert "No training rows: data.jsonl loaded 0 train samples as format 'chatml'." in output
    assert "Loaded:" not in output


def test_zero_rows_without_a_drop_says_no_row_failed_to_convert(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # A drop in an earlier load must not leak into this one's message.
    _write_jsonl(Path("bad.jsonl"), [{"instruction": "q", "output": "a"}])
    _loaded_rows("bad.jsonl", "chatml")
    assert last_load_outcome().first_drop is not None
    Path("empty.jsonl").write_text("", encoding="utf-8")
    _write_config(Path("soup.yaml"), "empty.jsonl", "chatml")

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    output = _plain(result.output)
    assert result.exit_code == 1, output
    assert "loaded 0 train samples as format 'chatml'." in output
    assert "No row failed to convert" in output
    assert "First: row" not in output


def test_dry_run_with_rows_still_says_ready_to_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(Path("data.jsonl"), [{"messages": _CHAT}])
    _write_config(Path("soup.yaml"), "data.jsonl", "chatml")

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    output = _plain(result.output)
    assert result.exit_code == 0, output
    assert "Data OK: 1 train samples" in output
    assert "Config valid. Ready to train!" in output


def test_a_pre_tokenized_run_counts_its_cache_not_the_loaders_view(tmp_path):
    # #1054: under `pre_tokenized` the loader sees the original file and keeps
    # nothing, while training reads the Arrow cache. The stop must not fire.
    import typer

    from soup_cli.commands.train import _refuse_empty_train

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(json.dumps({"row_count": 6}), encoding="utf-8")
    dcfg = SimpleNamespace(format="pre_tokenized", tokenized_path=str(cache), train="d.jsonl")

    _refuse_empty_train(dcfg, {"train": []})

    (cache / "metadata.json").unlink()
    with pytest.raises(typer.Exit):
        _refuse_empty_train(dcfg, {"train": []})


def test_the_outcome_names_the_resolved_format_not_auto(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(Path("data.jsonl"), [{"messages": _CHAT}, {"instruction": "q"}])

    assert len(_loaded_rows("data.jsonl", "auto")) == 1

    outcome = last_load_outcome()
    assert outcome.fmt == "chatml"
    assert outcome.first_drop == ("chatml", 1, "'messages'", "data.jsonl")
