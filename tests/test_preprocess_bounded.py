"""The local AOT path preserves eager results without keeping the corpus in RAM."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi
from tests.test_issue1038_preprocess_to_pretokenized import _ROW, _tokenizer


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(tmp_path, monkeypatch, *, task="sft", fmt="chatml", extra=""):
    import transformers

    from soup_cli.config.loader import load_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kw: _tokenizer(),
    )
    (tmp_path / "soup.yaml").write_text(
        f"base: x/tiny\ntask: {task}\ndata:\n  train: data.jsonl\n  format: {fmt}\n"
        "  max_length: 64\n  val_split: 0.0\n" + extra,
    )
    return load_config("soup.yaml")


def _invoke():
    from soup_cli.cli import app

    return CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"], color=True)


def _cache(tmp_path):
    from datasets import load_from_disk

    path, = [entry for entry in (tmp_path / ".soup-tokenized").iterdir() if entry.is_dir()]
    metadata = json.loads((path / "metadata.json").read_text())
    return path, load_from_disk(str(path)).to_list(), metadata


@pytest.mark.parametrize("pool_size,ratio", [(0, 0.4), (3, 0.001), (3, 0.5), (25, 0.4),
                                            (200, 0.1), (200, 0.5)])
@pytest.mark.parametrize("seed", [None, 0, 19])
@pytest.mark.parametrize("val_split", [0.0, 0.25])
def test_split_and_replay_match_existing_loader(
    tmp_path, monkeypatch, pool_size, ratio, seed, val_split,
):
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.preprocess import preprocessing_rows

    rows = [{"messages": [{"role": "user", "content": str(index)},
                           {"role": "assistant", "content": "Paris ."}]} for index in range(20)]
    replay = [{"instruction": f"replay {index}", "output": "What ."} for index in range(pool_size)]
    _write(tmp_path / "data.jsonl", rows)
    _write(tmp_path / "replay.jsonl", replay)
    cfg = _config(tmp_path, monkeypatch)
    cfg.data.val_split = val_split
    cfg.data.replay = "replay.jsonl"
    cfg.data.replay_ratio = ratio
    cfg.data.replay_seed = seed
    # An empty replay file with auto detection was always an error, even at ratio zero.
    if not pool_size:
        with pytest.raises(ValueError, match="Empty dataset"):
            load_dataset(cfg.data)
        with pytest.raises(ValueError, match="Empty dataset"):
            with preprocessing_rows(cfg.data, cfg.task):
                pass
        return
    expected = load_dataset(cfg.data)["train"]
    with preprocessing_rows(cfg.data, cfg.task) as actual:
        assert list(actual) == expected


@pytest.mark.parametrize("task,fmt,rows", [
    ("sft", "chatml", [_ROW] * 6),
    ("sft", "auto", [_ROW] * 6),
    ("sft", "alpaca", [{"instruction": "What ?", "output": "Paris ."}] * 6),
    ("sft", "sharegpt", [{"conversations": [
        {"from": "human", "value": "What ?"}, {"from": "gpt", "value": "Paris ."},
    ]}] * 6),
    ("pretrain", "plaintext", [{"text": " What is Paris .\n"}] * 6),
    ("pretrain", "auto", [{"text": "Paris " * 100}] * 6),
])
def test_public_cli_rows_and_metadata_match_eager_path(tmp_path, monkeypatch, task, fmt, rows):
    from soup_cli.data import preprocess
    from soup_cli.data.loader import load_dataset

    _write(tmp_path / "data.jsonl", rows)
    _config(tmp_path, monkeypatch, task=task, fmt=fmt)
    result = _invoke()
    assert result.exit_code == 0, result.output
    _, actual_rows, actual_metadata = _cache(tmp_path)

    @contextmanager
    def eager_rows(config, task):
        yield load_dataset(config)["train"]

    monkeypatch.setattr(preprocess, "preprocessing_rows", eager_rows)
    result = _invoke()
    assert result.exit_code == 0, result.output
    _, expected_rows, expected_metadata = _cache(tmp_path)
    assert actual_rows == expected_rows
    assert actual_metadata == expected_metadata


def test_reader_and_formatter_diagnostics_match_eager_loader(tmp_path, monkeypatch, capsys):
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.preprocess import preprocessing_rows

    _config(tmp_path, monkeypatch, fmt="auto")
    (tmp_path / "data.jsonl").write_text(
        '\ufeff\n{"other": "ignored"}\nfalse\nnot-json\n' + json.dumps(_ROW) + '\n{}\n',
        encoding="utf-8",
    )
    from soup_cli.config.loader import load_config

    cfg = load_config("soup.yaml")
    expected = load_dataset(cfg.data)["train"]
    expected_output = " ".join(strip_ansi(capsys.readouterr().out).split())
    with preprocessing_rows(cfg.data, cfg.task) as rows:
        assert list(rows) == expected
    assert " ".join(strip_ansi(capsys.readouterr().out).split()) == expected_output


def test_unsupported_replay_format_keeps_media_validation(tmp_path, monkeypatch):
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.preprocess import preprocessing_rows

    _write(tmp_path / "data.jsonl", [_ROW])
    _write(tmp_path / "replay.jsonl", [{"image": "missing.png", "conversations": [
        {"from": "human", "value": "What ?"}, {"from": "gpt", "value": "Paris ."},
    ]}])
    cfg = _config(tmp_path, monkeypatch, extra="  replay: replay.jsonl\n")
    expected = load_dataset(cfg.data)["train"]
    with preprocessing_rows(cfg.data, cfg.task) as rows:
        assert list(rows) == expected


def test_batches_and_shards_finish_before_the_last_row(tmp_path, monkeypatch):
    import pyarrow as arrow

    from soup_cli.utils import preprocess_arrow

    monkeypatch.setattr(preprocess_arrow, "_BATCH_ROWS", 2)
    monkeypatch.setattr(preprocess_arrow, "_SHARD_ROWS", 4)
    record = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}

    def rows():
        for index in range(9):
            if index == 8:
                first = sorted(tmp_path.glob("*.arrow"))[0]
                with arrow.OSFile(str(first), "rb") as handle:
                    with arrow.ipc.open_stream(handle) as reader:
                        assert sum(batch.num_rows for batch in reader) == 4
            yield record

    dataset = preprocess_arrow.PreparedDataset(rows(), tmp_path)
    assert dataset.num_rows == 9
    from datasets import load_from_disk

    assert load_from_disk(str(tmp_path)).to_list() == [record] * 9


def test_token_bound_flushes_before_the_row_bound(tmp_path, monkeypatch):
    from soup_cli.utils import preprocess_arrow

    monkeypatch.setattr(preprocess_arrow, "_BATCH_TOKENS", 8)
    monkeypatch.setattr(preprocess_arrow, "_SHARD_BYTES", 1)
    record = {"input_ids": [1] * 8, "labels": [1] * 8, "attention_mask": [1] * 8}

    def rows():
        yield record
        assert list(tmp_path.glob("*.arrow"))
        yield record

    assert preprocess_arrow.PreparedDataset(rows(), tmp_path).num_rows == 2


def test_sequential_validation_checks_real_shards_and_closes_handles(tmp_path, monkeypatch):
    import pyarrow as arrow

    from soup_cli.utils import preprocess_arrow

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "staging"
    folder.mkdir()
    monkeypatch.setattr(preprocess_arrow, "_SHARD_ROWS", 2)
    monkeypatch.setattr(preprocess_arrow, "_BATCH_ROWS", 2)
    row = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}
    dataset = preprocess_arrow.PreparedDataset(iter([row] * 5), folder)
    real_open = arrow.OSFile
    handles = []

    def open_file(*args, **kwargs):
        assert all(handle.closed for handle in handles)
        handle = real_open(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(arrow, "OSFile", open_file)
    with patch("datasets.load_from_disk", side_effect=AssertionError("whole-cache load")):
        preprocess_arrow.validate_staged_arrow(folder, 5, dataset.features)
    assert len(handles) == 3 and all(handle.closed for handle in handles)
    folder.rename(tmp_path / "renamed")
    with pytest.raises(ValueError, match="does not match"):
        preprocess_arrow.validate_staged_arrow(tmp_path / "renamed", 6, dataset.features)


@pytest.mark.parametrize("fault", ["tokenization", "batch", "shard"])
def test_preparation_failure_cleans_private_files_and_preserves_cache(
    tmp_path, monkeypatch, fault,
):
    from soup_cli.commands import data
    from soup_cli.utils import preprocess_arrow

    _write(tmp_path / "data.jsonl", [_ROW] * 5)
    _config(tmp_path, monkeypatch)
    assert _invoke().exit_code == 0
    cache, previous, metadata = _cache(tmp_path)
    monkeypatch.setattr(preprocess_arrow, "_BATCH_ROWS", 1)
    monkeypatch.setattr(preprocess_arrow, "_SHARD_ROWS", 1)
    original = data._tokenize_cache_rows
    tempdirs = []
    # Observe owned directories through the standard tempfile seam.
    import tempfile

    actual_temporary = tempfile.TemporaryDirectory

    def temporary(*args, **kwargs):
        directory = actual_temporary(*args, **kwargs)
        if kwargs.get("prefix", "").startswith("soup-preprocess-"):
            tempdirs.append(Path(directory.name))
        return directory

    def fail_rows(*args, **kwargs):
        for index, row in enumerate(original(*args, **kwargs)):
            if index == 3:
                raise ValueError("injected tokenize failure")
            yield row

    monkeypatch.setattr(tempfile, "TemporaryDirectory", temporary)
    if fault == "tokenization":
        monkeypatch.setattr(data, "_tokenize_cache_rows", fail_rows)
    else:
        from datasets.arrow_writer import ArrowWriter

        method = "write_table" if fault == "batch" else "finalize"
        real_method = getattr(ArrowWriter, method)

        def fail(*args, **kwargs):
            real_method(*args, **kwargs)
            raise OSError("injected Arrow failure")

        monkeypatch.setattr(ArrowWriter, method, fail)
    result = _invoke()
    assert result.exit_code == 1
    assert "injected" in str(result.exception)
    assert _cache(tmp_path)[1:] == (previous, metadata)
    assert tempdirs and all(not path.exists() for path in tempdirs)
    assert not list(cache.parent.glob(".tmp_*"))


def test_row_cap_still_counts_rows_after_split_replay(tmp_path, monkeypatch):
    from soup_cli.commands import data

    _write(tmp_path / "data.jsonl", [_ROW] * 7)
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr(data, "_MAX_PREPROCESS_ROWS", 3)
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert _cache(tmp_path)[2]["row_count"] == 3
    assert "Reached row cap 3" in strip_ansi(result.output)


@pytest.mark.parametrize("replay", [False, True])
def test_failed_auto_detection_retains_prior_parse_diagnostics(
    tmp_path, monkeypatch, capsys, replay,
):
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.preprocess import preprocessing_rows

    cfg = _config(tmp_path, monkeypatch, fmt="auto")
    _write(tmp_path / "data.jsonl", [_ROW])
    bad = tmp_path / ("replay.jsonl" if replay else "data.jsonl")
    bad.write_text('not-json\n{"unknown":"row"}\n')
    if replay:
        cfg.data.replay = "replay.jsonl"
    with pytest.raises(ValueError) as expected:
        load_dataset(cfg.data)
    expected_output = " ".join(strip_ansi(capsys.readouterr().out).split())
    with pytest.raises(ValueError, match="Cannot detect format") as actual:
        with preprocessing_rows(cfg.data, cfg.task):
            pass
    assert str(actual.value) == str(expected.value)
    assert " ".join(strip_ansi(capsys.readouterr().out).split()) == expected_output
    assert "invalid JSON on line 1" in expected_output


@pytest.mark.parametrize("nonempty_tail", [True, False])
def test_empty_token_prefix_has_a_stable_schema(tmp_path, monkeypatch, nonempty_tail):
    from datasets import Dataset, load_from_disk

    from soup_cli.utils import preprocess_arrow

    monkeypatch.setattr(preprocess_arrow, "_BATCH_ROWS", 2)
    empty = {"input_ids": [], "labels": [], "attention_mask": []}
    record = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}
    rows = [empty] * 2 + ([record] if nonempty_tail else [empty])
    assert preprocess_arrow.PreparedDataset(iter(rows), tmp_path).num_rows == 3
    assert load_from_disk(str(tmp_path)).to_list() == Dataset.from_list(rows).to_list()


def test_cli_pretrain_normalizer_can_erase_first_batches(tmp_path, monkeypatch):
    import transformers
    from tokenizers.normalizers import Replace

    from soup_cli.utils import preprocess_arrow

    _config(tmp_path, monkeypatch, task="pretrain", fmt="plaintext")
    tokenizer = _tokenizer()
    tokenizer.backend_tokenizer.normalizer = Replace("erase", "")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kw: tokenizer,
    )
    monkeypatch.setattr(preprocess_arrow, "_BATCH_ROWS", 2)
    _write(tmp_path / "data.jsonl", [{"text": "erase"}] * 2 + [{"text": "Paris"}])
    result = _invoke()
    assert result.exit_code == 0, result.output
    rows = _cache(tmp_path)[1]
    assert rows[:2] == [{"input_ids": [], "labels": [], "attention_mask": []}] * 2
    assert rows[2]["input_ids"] == tokenizer("Paris")["input_ids"]


def test_cli_split_replay_matches_eager_full_corpus_oracle(tmp_path, monkeypatch):
    from datasets import Dataset

    from soup_cli.commands import data
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.utils.data_pipeline import preprocess_mask_mode

    rows = [{"messages": [{"role": "user", "content": "What ? " * (index + 1)},
                           {"role": "assistant", "content": "Paris . " * (10 - index)}]}
            for index in range(10)]
    _write(tmp_path / "data.jsonl", rows)
    _write(tmp_path / "replay.jsonl", [
        {"instruction": "France ? " * (index + 1), "output": "What ."}
        for index in range(8)
    ])
    _config(tmp_path, monkeypatch)
    yaml = tmp_path / "soup.yaml"
    yaml.write_text(yaml.read_text().replace("val_split: 0.0", "val_split: 0.3") +
                    "  replay: replay.jsonl\n  replay_ratio: 0.4\n  replay_seed: 21\n")
    cfg = load_config(str(yaml))
    expected = Dataset.from_list(list(data._tokenize_cache_rows(
        load_dataset(cfg.data)["train"], _tokenizer(),
        preprocess_mask_mode(cfg.data, cfg.training), 64, False, cfg.base,
    ))).to_list()
    result = _invoke()
    assert result.exit_code == 0, result.output
    _, actual, metadata = _cache(tmp_path)
    assert actual == expected
    assert metadata["row_count"] == len(expected) == 12
    assert metadata["dataset_key"]["val_split"] == 0.3


def _measure_cli_working_set(tmp_path):
    import gc
    import tracemalloc

    monkeypatch = pytest.MonkeyPatch()

    _config(tmp_path, monkeypatch)
    yaml = tmp_path / "soup.yaml"
    yaml.write_text(yaml.read_text().replace("max_length: 64", "max_length: 512"))
    row = {"messages": [{"role": "user", "content": "What ?"},
                        {"role": "assistant", "content": "Paris " * 256}]}
    _write(tmp_path / "data.jsonl", [row])
    assert _invoke().exit_code == 0
    peaks = []
    for count in (400, 6400):
        with (tmp_path / "data.jsonl").open("w") as handle:
            for _ in range(count):
                handle.write(json.dumps(row) + "\n")
        gc.collect()
        tracemalloc.start()
        try:
            result = _invoke()
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert result.exit_code == 0, result.output
    monkeypatch.undo()
    return peaks


def _memory_worker(root, output):
    try:
        output.put((_measure_cli_working_set(Path(root)), None))
    except Exception as error:
        output.put((None, repr(error)))
        raise


def test_full_cli_python_working_set_is_bounded(tmp_path):
    import multiprocessing

    # A fresh interpreter excludes tracer state and background work from other tests.
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_memory_worker, args=(str(tmp_path), output))
    process.start()
    try:
        process.join(timeout=300)
        assert not process.is_alive(), "preprocessing memory check did not finish"
        assert process.exitcode == 0
        peaks, error = output.get(timeout=2)
        assert error is None, error
        # Same row and token limits: 16x the corpus must not add a whole-corpus list.
        assert peaks[1] < peaks[0] + 4 * 1024 * 1024, peaks
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        output.close()
        output.join_thread()


def test_byte_limit_finishes_a_shard_before_the_row_limit(tmp_path, monkeypatch):
    import pyarrow as arrow
    from datasets import arrow_writer

    from soup_cli.utils import preprocess_arrow

    monkeypatch.setattr(preprocess_arrow, "_BATCH_ROWS", 2)
    monkeypatch.setattr(preprocess_arrow, "_SHARD_BYTES", 1)
    real_writer = arrow_writer.ArrowWriter
    writers = []

    def writer(*args, **kwargs):
        current = real_writer(*args, **kwargs)
        writers.append(current)
        return current

    monkeypatch.setattr(arrow_writer, "ArrowWriter", writer)
    record = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}

    def rows():
        yield record
        yield record
        assert writers[0].stream.closed
        first, = tmp_path.glob("*.arrow")
        with arrow.OSFile(str(first), "rb") as handle:
            with arrow.ipc.open_stream(handle) as reader:
                assert sum(batch.num_rows for batch in reader) == 2
        yield record

    assert preprocess_arrow.PreparedDataset(rows(), tmp_path).num_rows == 3


def test_validation_refuses_info_schema_that_disagrees_with_shards(tmp_path, monkeypatch):
    from soup_cli.utils import preprocess_arrow

    monkeypatch.chdir(tmp_path)
    row = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}
    dataset = preprocess_arrow.PreparedDataset(iter([row]), tmp_path)
    path = tmp_path / "dataset_info.json"
    info = json.loads(path.read_text())
    del info["features"]["labels"]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="does not match"):
        preprocess_arrow.validate_staged_arrow(tmp_path, 1, dataset.features)


@pytest.mark.parametrize("missing", ["train", "replay", "outside_replay"])
def test_source_failures_keep_loader_refusals(tmp_path, monkeypatch, missing):
    from soup_cli.data.preprocess import preprocessing_rows

    _write(tmp_path / "data.jsonl", [_ROW])
    cfg = _config(tmp_path, monkeypatch)
    if missing == "train":
        (tmp_path / "data.jsonl").unlink()
        error, message = FileNotFoundError, "Data file not found"
    elif missing == "replay":
        cfg.data.replay = "missing.jsonl"
        error, message = FileNotFoundError, "data.replay file not found"
    else:
        outside = tmp_path.parent / (tmp_path.name + "-outside.jsonl")
        _write(outside, [_ROW])
        cfg.data.replay = str(outside)
        error, message = ValueError, "outside the working directory"
    with pytest.raises(error, match=message):
        with preprocessing_rows(cfg.data, cfg.task):
            pass


@pytest.mark.parametrize("corruption", ["path", "schema", "truncated"])
def test_validation_refuses_real_shard_damage(tmp_path, monkeypatch, corruption):
    import pyarrow as arrow

    from soup_cli.utils import preprocess_arrow

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "staging"
    folder.mkdir()
    row = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}
    dataset = preprocess_arrow.PreparedDataset(iter([row]), folder)
    shard, = folder.glob("*.arrow")
    if corruption == "path":
        state = json.loads((folder / "state.json").read_text())
        state["_data_files"][0]["filename"] = "../outside.arrow"
        (folder / "state.json").write_text(json.dumps(state))
        (tmp_path / "outside.arrow").write_bytes(shard.read_bytes())
    elif corruption == "schema":
        table = arrow.Table.from_pydict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})
        with arrow.OSFile(str(shard), "wb") as handle:
            with arrow.ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
    else:
        shard.write_bytes(shard.read_bytes()[:20])
    with pytest.raises((ValueError, arrow.ArrowInvalid)):
        preprocess_arrow.validate_staged_arrow(folder, 1, dataset.features)


def test_spooled_diagnostic_path_cannot_emit_terminal_controls(tmp_path, monkeypatch, capsys):
    from soup_cli.data.preprocess import preprocessing_rows

    cfg = _config(tmp_path, monkeypatch)
    cfg.data.train = "bad\x1b[2J.jsonl"
    _write(tmp_path / cfg.data.train, [{}, _ROW])
    with preprocessing_rows(cfg.data, cfg.task) as rows:
        assert list(rows) == [_ROW]
    output = capsys.readouterr().out
    assert "rows dropped" in output
    assert "\x1b" not in output


def test_replayed_diagnostics_remove_rendering_escapes(tmp_path, monkeypatch, capsys):
    from rich.console import Console

    from soup_cli.data import loader
    from soup_cli.data.preprocess import preprocessing_rows

    cfg = _config(tmp_path, monkeypatch)
    _write(tmp_path / "data.jsonl", [{}, _ROW])
    monkeypatch.setattr(loader, "console", Console(
        force_terminal=True, color_system="standard", no_color=False, width=100,
    ))
    assert loader.load_dataset(cfg.data)["train"] == [_ROW]
    expected = capsys.readouterr().out
    assert "\x1b[" in expected
    with preprocessing_rows(cfg.data, cfg.task) as rows:
        assert list(rows) == [_ROW]
    actual = capsys.readouterr().out
    assert "\x1b" not in actual
    assert actual.endswith("\n")
    assert " ".join(actual.split()) == " ".join(strip_ansi(expected).split())
