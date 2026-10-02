"""Image and audio paths resolve the same way on every load path (#PR).

Only three loaders used to resolve the ``image`` / ``audio`` value of a vision
or audio row: the single local file, the eager per-file interleave and the
replay file. The streaming interleave, the remote URI loader and the Hub loader
left it as written, so a relative path was later opened against the working
directory instead of the data directory, and ``data.image_dir`` /
``data.audio_dir`` were ignored on those paths.

Every load path is driven here through the public ``load_dataset``, with
``datasets`` and ``fsspec`` replaced by in-process fakes, and must give the same
resolved paths and the same "rows skipped" warnings. One class repeats the
streaming interleave with the real ``datasets`` library on local files. A remote
URI or a Hub dataset has no data directory of its own, so a string path there
needs the setting, and the error names it. On the trainer side, only a regular
file is opened.

An autouse guard fails any test in which a network-share or device form, or a
value holding a NUL byte, reaches ``os.path.realpath``: those values must be
refused before any filesystem call. The guard also keeps the file offline
whatever the code under test does.
"""

from __future__ import annotations

import errno
import io
import json
import os
import re
import stat
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.data.loader as loader
from soup_cli.config.schema import DataConfig
from soup_cli.data.loader import load_dataset
from soup_cli.utils.paths import (
    is_device_namespace_path,
    is_network_or_device_path,
    require_regular_file,
)
from tests.conftest import strip_ansi


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed console output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


# Every network or device spelling carries "soup-sentinel", which the realpath
# guard below looks for.
NETWORK_FORMS = [
    r"\\soup-sentinel.invalid\share\x.bin",
    "//soup-sentinel.invalid/share/x.bin",
    r"\/soup-sentinel.invalid/share/x.bin",
    r"/\soup-sentinel.invalid\share\x.bin",
    r"\\?\UNC\soup-sentinel.invalid\share\x.bin",
    r"\\.\pipe\soup-sentinel",
    r"\??\UNC\soup-sentinel.invalid\share\x.bin",
    "/??/UNC/soup-sentinel.invalid/share/x.bin",
    # The same prefix once "." and ".." segments are collapsed, which pathlib
    # and os.path.realpath do before they look at the filesystem.
    r"\.\??\UNC\soup-sentinel.invalid\share\x.bin",
    "/./??/UNC/soup-sentinel.invalid/share/x.bin",
    r"\x\..\??\UNC\soup-sentinel.invalid\share\x.bin",
]

# The forms the trainer-side check refuses: device and object-manager paths.
# A UNC or extended-length path is not among them, because the loader's own
# resolve() returns one for a mapped drive.
DEVICE_NAMESPACE_FORMS = [
    r"\\.\pipe\soup-sentinel",
    "//./pipe/soup-sentinel",
    r"\??\UNC\soup-sentinel.invalid\share\x.bin",
    "/??/UNC/soup-sentinel.invalid/share/x.bin",
    r"\.\??\UNC\soup-sentinel.invalid\share\x.bin",
    r"\x\..\??\UNC\soup-sentinel.invalid\share\x.bin",
]

_KEY = {"llava": "image", "audio": "audio", "asr": "audio"}
_SETTING = {"llava": "image_dir", "audio": "audio_dir", "asr": "audio_dir"}
_FORMATS = ["llava", "audio", "asr"]
_PATH_KINDS = [
    "single_file",
    "eager_interleave",
    "streaming_interleave",
    "remote",
    "remote_streaming",
    "hub",
    "hub_streaming",
    "hub_interleave",
]
_REMOTE_KINDS = ["remote", "remote_streaming", "hub", "hub_streaming", "hub_interleave"]
_REMOTE_URI = "s3://soup-bucket/rows.jsonl"
_HUB_NAME = "org/media-set"
_TAG_KEY = "__soup_source_index__"  # loader._SOURCE_INDEX_KEY, spelled out


@pytest.fixture(autouse=True)
def _guard_realpath(monkeypatch):
    real_realpath = os.path.realpath

    def guarded(path, *args, **kwargs):
        text = os.fsdecode(path)
        if "soup-sentinel" in text or "\x00" in text:
            raise AssertionError(f"os.path.realpath reached {text!r}")
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", guarded)


@pytest.fixture
def loader_output(monkeypatch) -> io.StringIO:
    out = io.StringIO()
    monkeypatch.setattr(loader, "console", Console(file=out, width=400))
    return out


# ---------------------------------------------------------------------------
# Rows, files and fakes
# ---------------------------------------------------------------------------


def _row(fmt: str, media, tag: str) -> dict:
    """One row of ``fmt`` whose media value is ``media``; ``tag`` identifies it."""
    if fmt == "llava":
        return {
            "image": media,
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe."},
                {"from": "gpt", "value": tag},
            ],
        }
    if fmt == "audio":
        return {
            "audio": media,
            "messages": [
                {"role": "user", "content": "Transcribe."},
                {"role": "assistant", "content": tag},
            ],
        }
    return {"audio": media, "text": tag}


def _tag_of(row: dict) -> str:
    if "messages" in row:
        return row["messages"][-1]["content"]
    return row["text"]


def _layout(tmp_path: Path) -> dict[str, Path]:
    """``data/`` (the data files), ``media_root/`` (a configured media dir) and
    ``outside/``, siblings under ``tmp_path``; both base dirs hold ``clips/ok.bin``."""
    dirs = {
        "data": tmp_path / "data",
        "media": tmp_path / "media_root",
        "outside": tmp_path / "outside",
    }
    for base in (dirs["data"], dirs["media"]):
        (base / "clips").mkdir(parents=True)
        (base / "clips" / "ok.bin").write_bytes(b"x")
    dirs["outside"].mkdir()
    (dirs["outside"] / "secret.bin").write_bytes(b"x")
    return dirs


def _rows(fmt: str, base: Path, outside: Path) -> list[dict]:
    """Two rows that resolve under ``base`` and two that lie outside it (plus,
    for llava, one with an empty path; the audio converters drop that shape).

    Network and device spellings have tests of their own below, so the local
    load paths here behave exactly as they do on ``main`` and act as controls.
    """
    rows = [
        _row(fmt, "clips/ok.bin", "kept-relative"),
        _row(fmt, str(base / "clips" / "ok.bin"), "kept-absolute"),
        _row(fmt, "../outside/secret.bin", "outside-relative"),
        _row(fmt, str(outside / "secret.bin"), "outside-absolute"),
    ]
    if fmt == "llava":
        rows.append(_row(fmt, "", "empty"))
    return rows


def _resolved(base: Path, relative: str = "clips/ok.bin") -> str:
    return str((base / relative).resolve())


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class _FakeStream:
    """Stand-in for an HF ``IterableDataset``: rows, ``map``, ``shuffle``, no features."""

    features = None

    def __init__(self, rows, calls: list[str]):
        self._rows = [dict(row) for row in rows]
        self._calls = calls

    def shuffle(self, buffer_size=None):
        return self

    def map(self, function, features=None):
        self._calls.append("map")
        return _FakeStream([{**row, **function(row)} for row in self._rows], self._calls)

    def __iter__(self):
        return iter([dict(row) for row in self._rows])


def _install_fake_datasets(monkeypatch, *, files=None, hub=None) -> list[str]:
    """Install a fake ``datasets`` module; returns the list it records calls in.

    ``files`` maps a ``data_files`` value to rows (a local path not listed is
    read from disk); ``hub`` maps a dataset name to ``{split: rows}``.
    """
    files = dict(files or {})
    hub = dict(hub or {})
    calls: list[str] = []

    def load_dataset(path, data_files=None, split=None, streaming=False):
        if data_files is not None:
            assert split == "train" and streaming is True
            rows = files.get(data_files)
            if rows is None:
                rows = _read_jsonl(Path(data_files))
            return _FakeStream(rows, calls)
        splits = hub[path]
        if streaming:
            return {name: _FakeStream(rows, calls) for name, rows in splits.items()}
        return {name: [dict(row) for row in rows] for name, rows in splits.items()}

    def concatenate_datasets(streams):
        return _FakeStream([row for stream in streams for row in stream], calls)

    def interleave_datasets(streams, probabilities=None, stopping_strategy="first_exhausted"):
        pools = [list(stream) for stream in streams]
        if stopping_strategy == "all_exhausted":
            rounds = max(len(pool) for pool in pools)
        else:
            rounds = min(len(pool) for pool in pools)
        mixed = [pool[turn % len(pool)] for turn in range(rounds) for pool in pools]
        return _FakeStream(mixed, calls)

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    module.concatenate_datasets = concatenate_datasets
    module.interleave_datasets = interleave_datasets
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def _install_fake_fsspec(monkeypatch, objects: dict[str, list[dict]]) -> None:
    class _Handle:
        def __init__(self, lines):
            self._lines = lines

        def __enter__(self):
            return iter(self._lines)

        def __exit__(self, *exc_info):
            return False

    def fake_open(uri, mode="rt", encoding=None):
        return _Handle([json.dumps(row) + "\n" for row in objects[uri]])

    module = types.ModuleType("fsspec")
    module.open = fake_open
    monkeypatch.setitem(sys.modules, "fsspec", module)


def _load(kind, fmt, rows, extra_rows, tmp_path, monkeypatch, **data_overrides) -> dict:
    """Load ``rows`` through load path ``kind``. Interleave kinds add a second
    source holding ``extra_rows``, combined with ``concat``; the local ones keep
    both files in ``data/``."""
    data_dir = tmp_path / "data"
    common = {"format": fmt, "val_split": 0.0, **data_overrides}
    if kind in ("single_file", "eager_interleave", "streaming_interleave"):
        first = _write_jsonl(data_dir / "rows.jsonl", rows)
        if kind == "single_file":
            return load_dataset(DataConfig(train=str(first), **common))
        second = _write_jsonl(data_dir / "more.jsonl", extra_rows)
        streaming = kind == "streaming_interleave"
        if streaming:
            _install_fake_datasets(monkeypatch)
        return load_dataset(
            DataConfig(
                train=[str(first), str(second)],
                interleave="concat",
                streaming=streaming,
                **common,
            )
        )
    if kind in ("remote", "remote_streaming"):
        _install_fake_fsspec(monkeypatch, {_REMOTE_URI: rows})
        _install_fake_datasets(monkeypatch, files={_REMOTE_URI: rows})
        return load_dataset(
            DataConfig(train=_REMOTE_URI, streaming=kind == "remote_streaming", **common)
        )
    if kind in ("hub", "hub_streaming"):
        _install_fake_datasets(monkeypatch, hub={_HUB_NAME: {"train": rows}})
        return load_dataset(
            DataConfig(train=_HUB_NAME, streaming=kind == "hub_streaming", **common)
        )
    assert kind == "hub_interleave"
    _install_fake_datasets(
        monkeypatch,
        hub={_HUB_NAME: {"train": rows}, "org/more": {"train": extra_rows}},
    )
    return load_dataset(
        DataConfig(train=[_HUB_NAME, "org/more"], interleave="concat", **common)
    )


def _skip_lines(key: str, count: int, base: Path | str) -> str:
    """The "outside" warning the loader prints for ``count`` rows of ``key``."""
    if key == "image":
        return f"Warning: {count} rows skipped (image path outside {base})"
    return f"Warning: {count} rows skipped (audio path traversal blocked)"


def _assert_resolved_like_local(result: dict, fmt: str, base: Path, output: str) -> None:
    """The outcome every load path must share for ``_rows(fmt, base, ...)``."""
    kept = [row for row in result["train"] if _tag_of(row) != "second-source"]
    assert [_tag_of(row) for row in kept] == ["kept-relative", "kept-absolute"]
    assert [row[_KEY[fmt]] for row in kept] == [_resolved(base)] * 2

    plain = _plain(output)
    if fmt == "llava":
        assert plain.count("Warning: 1 rows skipped (missing image path)") == 1, plain
    assert plain.count(_skip_lines(_KEY[fmt], 2, base)) == 1, plain


# ---------------------------------------------------------------------------
# Parity across load paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("kind", _PATH_KINDS)
def test_every_load_path_resolves_against_the_configured_media_dir(
    kind, fmt, tmp_path, monkeypatch, loader_output
):
    dirs = _layout(tmp_path)
    media_root = dirs["media"]
    result = _load(
        kind,
        fmt,
        _rows(fmt, media_root, dirs["outside"]),
        [_row(fmt, "clips/ok.bin", "second-source")],
        tmp_path,
        monkeypatch,
        **{_SETTING[fmt]: str(media_root)},
    )

    _assert_resolved_like_local(result, fmt, media_root, loader_output.getvalue())
    if "interleave" in kind:
        assert _tag_of(result["train"][-1]) == "second-source"
        assert result["train"][-1][_KEY[fmt]] == _resolved(media_root)


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("kind", ["single_file", "eager_interleave", "streaming_interleave"])
def test_local_paths_default_to_the_data_files_directory(
    kind, fmt, tmp_path, monkeypatch, loader_output
):
    dirs = _layout(tmp_path)
    result = _load(
        kind,
        fmt,
        _rows(fmt, dirs["data"], dirs["outside"]),
        [_row(fmt, "clips/ok.bin", "second-source")],
        tmp_path,
        monkeypatch,
    )

    _assert_resolved_like_local(result, fmt, dirs["data"], loader_output.getvalue())


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("kind", ["streaming_interleave", *_REMOTE_KINDS])
def test_format_auto_resolves_like_an_explicit_format(
    kind, fmt, tmp_path, monkeypatch, loader_output
):
    # format: auto is the default. The media pass must use the detected format,
    # not the configured "auto", on every path that detects after loading.
    dirs = _layout(tmp_path)
    result = _load(
        kind,
        "auto",
        _rows(fmt, dirs["media"], dirs["outside"]),
        [_row(fmt, "clips/ok.bin", "second-source")],
        tmp_path,
        monkeypatch,
        **{_SETTING[fmt]: str(dirs["media"])},
    )

    _assert_resolved_like_local(result, fmt, dirs["media"], loader_output.getvalue())


# ---------------------------------------------------------------------------
# Streaming interleave: each row against its own entry
# ---------------------------------------------------------------------------


def _two_entry_dirs(tmp_path: Path, *, shared: bool = False) -> tuple[Path, Path]:
    first_dir = tmp_path / "first"
    second_dir = first_dir if shared else tmp_path / "second"
    for base in {first_dir, second_dir}:
        (base / "clips").mkdir(parents=True)
        (base / "clips" / "ok.bin").write_bytes(b"x")
    return first_dir, second_dir


def _streaming_config(train, interleave="concat", **overrides) -> DataConfig:
    common = {"format": "llava", "val_split": 0.0, **overrides}
    return DataConfig(train=[str(path) for path in train], interleave=interleave,
                      streaming=True, **common)


@pytest.mark.parametrize("fmt", ["llava", "asr", "auto"])
def test_streamed_rows_resolve_against_their_own_entry(
    fmt, tmp_path, monkeypatch, loader_output
):
    shape = "asr" if fmt == "asr" else "llava"
    first_dir, second_dir = _two_entry_dirs(tmp_path)
    # The first row converts to nothing, so formatted rows sit one place
    # before their raw rows; each must still find its own entry.
    dropped = {"image": "clips/ok.bin"} if shape == "llava" else {"audio": "clips/ok.bin"}
    first = _write_jsonl(
        first_dir / "rows.jsonl", [dropped, _row(shape, "clips/ok.bin", "from-first")]
    )
    second = _write_jsonl(
        second_dir / "rows.jsonl",
        [
            _row(shape, "clips/ok.bin", "from-second"),
            _row(shape, "../first/clips/ok.bin", "leaves-second"),
        ],
    )
    calls = _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second], format=fmt))

    assert {_tag_of(row): row[_KEY[shape]] for row in result["train"]} == {
        "from-first": _resolved(first_dir),
        "from-second": _resolved(second_dir),
    }
    assert calls.count("map") == 2
    assert _plain(loader_output.getvalue()).count(
        _skip_lines(_KEY[shape], 1, second_dir)
    ) == 1


@pytest.mark.parametrize(
    "first_rows",
    [["note", "from-first"], ["note"]],
    ids=["first-row-has-no-format", "first-entry-has-no-media"],
)
def test_format_auto_tags_when_any_first_row_carries_the_media_key(
    first_rows, tmp_path, monkeypatch
):
    # A row that matches no format is skipped by detection and dropped by the
    # converter, so the detected format comes from the rows after it, in any
    # entry: the check must read past it, and look at every entry.
    first_dir, second_dir = _two_entry_dirs(tmp_path)
    rows = {
        "note": {"note": "no format"},
        "from-first": _row("llava", "clips/ok.bin", "from-first"),
    }
    first = _write_jsonl(first_dir / "rows.jsonl", [rows[name] for name in first_rows])
    second = _write_jsonl(
        second_dir / "rows.jsonl",
        [{"note": "no format"}, _row("llava", "clips/ok.bin", "from-second")],
    )
    _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second], format="auto"))

    expected = {"from-first": _resolved(first_dir), "from-second": _resolved(second_dir)}
    assert {_tag_of(row): row["image"] for row in result["train"]} == {
        name: expected[name] for name in [*first_rows, "from-second"] if name != "note"
    }


@pytest.mark.parametrize("fmt", ["llava", "asr", "auto"])
def test_streamed_entries_sharing_a_directory_are_not_tagged(
    fmt, tmp_path, monkeypatch, loader_output
):
    shape = "asr" if fmt == "asr" else "llava"
    first_dir, _ = _two_entry_dirs(tmp_path, shared=True)
    first = _write_jsonl(first_dir / "a.jsonl", [_row(shape, "clips/ok.bin", "a")])
    second = _write_jsonl(
        first_dir / "b.jsonl",
        [_row(shape, "clips/ok.bin", "b"), _row(shape, "../x.bin", "b-outside")],
    )
    calls = _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second], format=fmt))

    assert [(_tag_of(row), row[_KEY[shape]]) for row in result["train"]] == [
        ("a", _resolved(first_dir)),
        ("b", _resolved(first_dir)),
    ]
    assert "map" not in calls
    assert _plain(loader_output.getvalue()).count(
        _skip_lines(_KEY[shape], 1, first_dir)
    ) == 1


@pytest.mark.parametrize("layout", ["one_directory", "two_directories"])
@pytest.mark.parametrize(
    "interleave", ["concat", "under", "over", {"strategy": "probs", "probs": [0.5, 0.5]}]
)
def test_every_streaming_strategy_resolves_each_row_against_its_entry(
    interleave, layout, tmp_path, monkeypatch, loader_output
):
    first_dir, second_dir = _two_entry_dirs(tmp_path, shared=layout == "one_directory")
    first = _write_jsonl(
        first_dir / "a.jsonl",
        [_row("llava", "clips/ok.bin", "a-kept"), _row("llava", "../x.bin", "a-outside")],
    )
    second = _write_jsonl(
        second_dir / "b.jsonl",
        [_row("llava", "clips/ok.bin", "b-kept"), _row("llava", "../x.bin", "b-outside")],
    )
    calls = _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second], interleave=interleave))

    assert sorted((_tag_of(row), row["image"]) for row in result["train"]) == [
        ("a-kept", _resolved(first_dir)),
        ("b-kept", _resolved(second_dir)),
    ]
    plain = _plain(loader_output.getvalue())
    if layout == "one_directory":
        assert plain.count(_skip_lines("image", 2, first_dir)) == 1, plain
        assert "map" not in calls
    else:
        assert plain.count(_skip_lines("image", 1, first_dir)) == 1, plain
        assert plain.count(_skip_lines("image", 1, second_dir)) == 1, plain
        assert calls.count("map") == 2


def test_streaming_over_with_a_validation_split_resolves_both_sides(tmp_path, monkeypatch):
    first_dir, second_dir = _two_entry_dirs(tmp_path)
    first = _write_jsonl(
        first_dir / "a.jsonl",
        [_row("llava", "clips/ok.bin", f"a{index}") for index in range(3)],
    )
    second = _write_jsonl(second_dir / "b.jsonl", [_row("llava", "clips/ok.bin", "b0")])
    _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second], interleave="over", val_split=0.5))

    expected = {"a": _resolved(first_dir), "b": _resolved(second_dir)}
    assert result["val"], result
    for row in [*result["train"], *result["val"]]:
        assert row["image"] == expected[_tag_of(row)[0]], row


def test_streaming_auto_format_does_not_leak_the_source_tag(tmp_path, monkeypatch):
    first_dir, second_dir = _two_entry_dirs(tmp_path)
    first = _write_jsonl(first_dir / "a.jsonl", [_row("llava", "clips/ok.bin", "a")])
    second = _write_jsonl(second_dir / "b.jsonl", [_row("llava", "clips/ok.bin", "b")])
    calls = _install_fake_datasets(monkeypatch)

    result = load_dataset(
        _streaming_config([first, second], format="auto"), preserve_source_columns=True
    )

    assert [(_tag_of(row), row["image"]) for row in result["train"]] == [
        ("a", _resolved(first_dir)),
        ("b", _resolved(second_dir)),
    ]
    assert calls.count("map") == 2
    assert all(_TAG_KEY not in row for row in result["train"])


def test_a_row_cannot_name_its_own_entry(tmp_path, monkeypatch):
    first_dir, second_dir = _two_entry_dirs(tmp_path)
    first = _write_jsonl(
        first_dir / "a.jsonl",
        [{**_row("llava", "clips/ok.bin", "claims-second"), _TAG_KEY: 1}],
    )
    second = _write_jsonl(second_dir / "b.jsonl", [_row("llava", "clips/ok.bin", "b")])
    _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second]))

    assert {_tag_of(row): row["image"] for row in result["train"]} == {
        "claims-second": _resolved(first_dir),
        "b": _resolved(second_dir),
    }


@pytest.mark.parametrize("fmt", ["plaintext", "auto"])
def test_streaming_text_entries_are_not_tagged(fmt, tmp_path, monkeypatch):
    first_dir, second_dir = _two_entry_dirs(tmp_path)
    first = _write_jsonl(first_dir / "a.jsonl", [{"text": "alpha"}])
    second = _write_jsonl(second_dir / "b.jsonl", [{"text": "beta"}])
    calls = _install_fake_datasets(monkeypatch)

    result = load_dataset(_streaming_config([first, second], format=fmt))

    assert len(result["train"]) == 2
    assert "map" not in calls


def test_streaming_remote_entry_needs_the_media_dir(tmp_path, monkeypatch):
    local = _write_jsonl(tmp_path / "data" / "rows.jsonl", [_row("llava", "a.png", "local")])
    remote = "s3://soup-bucket/more.jsonl"
    _install_fake_datasets(monkeypatch, files={remote: [_row("llava", "b.png", "remote")]})

    with pytest.raises(ValueError) as excinfo:
        load_dataset(_streaming_config([local, remote]))

    message = str(excinfo.value)
    assert f"data.train[1] {remote!r} is not a local file" in message
    assert "Set data.image_dir to the local directory that holds the image files" in message


def test_streaming_remote_entries_without_a_media_dir_name_every_entry(monkeypatch):
    first = "s3://soup-bucket/a.jsonl"
    second = "s3://soup-bucket/b.jsonl"
    calls = _install_fake_datasets(
        monkeypatch,
        files={first: [_row("llava", "a.png", "a")], second: [_row("llava", "b.png", "b")]},
    )

    with pytest.raises(ValueError) as excinfo:
        load_dataset(_streaming_config([first, second]))

    assert str(excinfo.value) == (
        f"data.train entries {first!r}, {second!r} are not local files, so their "
        "'image' paths have no directory to resolve against. Set data.image_dir to "
        "the local directory that holds the image files: relative paths resolve under "
        "it, and rows whose image is outside it are skipped."
    )
    assert "map" not in calls


def test_streaming_remote_entries_resolve_against_the_media_dir(
    tmp_path, monkeypatch, loader_output
):
    dirs = _layout(tmp_path)
    first = "s3://soup-bucket/a.jsonl"
    second = "s3://soup-bucket/b.jsonl"
    calls = _install_fake_datasets(
        monkeypatch,
        files={
            first: [_row("llava", "clips/ok.bin", "a")],
            second: [_row("llava", "../outside/secret.bin", "b")],
        },
    )

    result = load_dataset(_streaming_config([first, second], image_dir=str(dirs["media"])))

    assert [(_tag_of(row), row["image"]) for row in result["train"]] == [
        ("a", _resolved(dirs["media"]))
    ]
    assert "map" not in calls
    assert _skip_lines("image", 1, dirs["media"]) in _plain(loader_output.getvalue())


# ---------------------------------------------------------------------------
# Streaming interleave with the real ``datasets`` library (local files only)
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_parquet(path: Path, rows: list[dict], image_columns=()) -> Path:
    datasets = pytest.importorskip("datasets")
    table = datasets.Dataset.from_list(rows)
    for column in image_columns:
        table = table.cast_column(column, datasets.Image())
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(str(path))
    return path


class TestRealDatasetsStreaming:
    @pytest.fixture(autouse=True)
    def _real_datasets(self):
        pytest.importorskip("datasets")
        pytest.importorskip("pyarrow")
        pytest.importorskip("PIL")

    @pytest.fixture
    def no_map(self, monkeypatch):
        import datasets

        def refuse(self, *args, **kwargs):
            raise AssertionError("IterableDataset.map was called")

        monkeypatch.setattr(datasets.IterableDataset, "map", refuse)

    def test_a_jsonl_and_an_image_parquet_in_different_directories(self, tmp_path):
        first_dir, second_dir = _two_entry_dirs(tmp_path)
        first = _write_jsonl(
            first_dir / "rows.jsonl",
            [_row("llava", "clips/ok.bin", "a0"), _row("llava", "clips/ok.bin", "a1")],
        )
        thumb = {"bytes": _png_bytes(), "path": None}
        second = _write_parquet(
            second_dir / "rows.parquet",
            [
                {**_row("llava", "clips/ok.bin", "b0"), "thumb": thumb},
                {**_row("llava", "../x.bin", "b-outside"), "thumb": thumb},
            ],
            image_columns=["thumb"],
        )

        result = load_dataset(
            _streaming_config([first, second], format="auto"), preserve_source_columns=True
        )

        rows = result["train"]
        assert [(_tag_of(row), row["image"]) for row in rows] == [
            ("a0", _resolved(first_dir)),
            ("a1", _resolved(first_dir)),
            ("b0", _resolved(second_dir)),
        ]
        assert all(_TAG_KEY not in row for row in rows)
        # The JSONL rows carry the parquet's extra column as None, as they do
        # when the two streams are combined without a tag.
        assert [("thumb" in row, row.get("thumb") is None) for row in rows] == [
            (True, True),
            (True, True),
            (True, False),
        ]

    def test_entries_sharing_a_directory_are_not_tagged(self, tmp_path, no_map):
        first_dir, _ = _two_entry_dirs(tmp_path, shared=True)
        first = _write_jsonl(first_dir / "a.jsonl", [_row("llava", "clips/ok.bin", "a")])
        thumb = {"bytes": _png_bytes(), "path": None}
        second = _write_parquet(
            first_dir / "b.parquet",
            [{**_row("llava", "clips/ok.bin", "b"), "thumb": thumb}],
            image_columns=["thumb"],
        )

        result = load_dataset(_streaming_config([first, second], format="auto"))

        assert [(_tag_of(row), row["image"]) for row in result["train"]] == [
            ("a", _resolved(first_dir)),
            ("b", _resolved(first_dir)),
        ]

    def test_text_entries_in_different_directories_are_not_tagged(self, tmp_path, no_map):
        first = _write_parquet(tmp_path / "a" / "rows.parquet", [{"text": "alpha"}])
        second = _write_parquet(tmp_path / "b" / "rows.parquet", [{"text": "beta"}])

        result = load_dataset(_streaming_config([first, second], format="auto"))

        assert [row["text"] for row in result["train"]] == ["alpha", "beta"]

    def test_decoded_images_from_two_directories_pass_through(self, tmp_path):
        pil_image = pytest.importorskip("PIL.Image")
        image = {"bytes": _png_bytes(), "path": None}
        first = _write_parquet(
            tmp_path / "a" / "rows.parquet", [_row("llava", image, "a")], image_columns=["image"]
        )
        second = _write_parquet(
            tmp_path / "b" / "rows.parquet", [_row("llava", image, "b")], image_columns=["image"]
        )

        result = load_dataset(_streaming_config([first, second]))

        assert [_tag_of(row) for row in result["train"]] == ["a", "b"]
        assert all(isinstance(row["image"], pil_image.Image) for row in result["train"])


# ---------------------------------------------------------------------------
# Remote URI and Hub dataset: no data directory of their own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize(
    ("fmt", "setting"),
    [("llava", "data.image_dir"), ("audio", "data.audio_dir"), ("asr", "data.audio_dir")],
)
@pytest.mark.parametrize("kind", _REMOTE_KINDS)
def test_remote_and_hub_paths_need_the_media_dir(
    kind, fmt, setting, absolute, tmp_path, monkeypatch
):
    media = str(tmp_path / "clips" / "ok.bin") if absolute else "clips/ok.bin"

    with pytest.raises(ValueError) as excinfo:
        _load(
            kind,
            fmt,
            [_row(fmt, media, "path-row")],
            [_row(fmt, media, "second-source")],
            tmp_path,
            monkeypatch,
        )

    message = str(excinfo.value)
    assert f"Set {setting} to the local directory that holds the" in message
    source = _REMOTE_URI if kind.startswith("remote") else _HUB_NAME
    assert f"{source!r} is not a local file" in message


def test_the_media_dir_message_names_the_source_and_the_setting(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        _load("hub", "llava", [_row("llava", "clips/ok.bin", "row")], [], tmp_path, monkeypatch)

    assert str(excinfo.value) == (
        "Hub dataset 'org/media-set' is not a local file, so its 'image' paths have no "
        "directory to resolve against. Set data.image_dir to the local directory that "
        "holds the image files: relative paths resolve under it, and rows whose image "
        "is outside it are skipped."
    )


_DRY_RUN_CONFIG = (
    "base: hf-internal-testing/tiny-random-LlamaForCausalLM\ntask: sft\nmodality: vision\n"
    "data:\n  train: org/media-set\n  format: llava\n  val_split: 0.0\n{extra}output: ./out\n"
)


def test_dry_run_stops_on_a_hub_dataset_without_the_media_dir(tmp_path, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.utils import errors

    monkeypatch.chdir(tmp_path)
    _layout(tmp_path)
    _install_fake_datasets(
        monkeypatch, hub={_HUB_NAME: {"train": [_row("llava", "clips/ok.bin", "row")]}}
    )
    Path("soup.yaml").write_text(_DRY_RUN_CONFIG.format(extra=""), encoding="utf-8")

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Data OK" not in _plain(result.output)
    assert isinstance(result.exception, ValueError), repr(result.exception)
    rendered = io.StringIO()
    monkeypatch.setattr(errors, "console", Console(file=rendered, width=400))
    errors.format_friendly_error(result.exception)
    assert (
        "Error: ValueError: Hub dataset 'org/media-set' is not a local file, so its "
        "'image' paths have no directory to resolve against. Set data.image_dir to the "
        "local directory that holds the image files"
    ) in _plain(rendered.getvalue())

    Path("soup.yaml").write_text(
        _DRY_RUN_CONFIG.format(extra="  image_dir: media_root\n"), encoding="utf-8"
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Data OK: 1 train samples" in _plain(result.output)


def test_hub_decoded_images_pass_through_without_a_media_dir(monkeypatch):
    decoded = object()  # stands in for a PIL image decoded from an Image() feature
    rows = [{"image": decoded, "conversations": [{"from": "human", "value": "Hi"}]}]
    _install_fake_datasets(monkeypatch, hub={"org/decoded": {"train": rows}})

    result = load_dataset(DataConfig(train="org/decoded", format="llava", val_split=0.0))

    assert result["train"][0]["image"] is decoded


def test_hub_validation_split_is_resolved_too(tmp_path, monkeypatch, loader_output):
    dirs = _layout(tmp_path)
    _install_fake_datasets(
        monkeypatch,
        hub={
            _HUB_NAME: {
                "train": [_row("llava", "clips/ok.bin", "train")],
                "validation": [
                    _row("llava", "clips/ok.bin", "val"),
                    _row("llava", "../outside/secret.bin", "val-outside"),
                ],
            }
        },
    )

    result = load_dataset(
        DataConfig(
            train=_HUB_NAME, format="llava", val_split=0.0, image_dir=str(dirs["media"])
        )
    )

    assert [(_tag_of(row), row["image"]) for row in result["val"]] == [
        ("val", _resolved(dirs["media"]))
    ]
    assert _skip_lines("image", 1, dirs["media"]) in _plain(loader_output.getvalue())


@pytest.mark.parametrize("kind", ["remote", "hub"])
def test_image_dir_dot_resolves_under_the_working_directory(
    kind, tmp_path, monkeypatch, loader_output
):
    monkeypatch.chdir(tmp_path)
    _layout(tmp_path)
    rows = [
        _row("llava", "media_root/clips/ok.bin", "kept"),
        _row("llava", str(tmp_path.parent / "elsewhere.bin"), "outside-cwd"),
    ]

    result = _load(kind, "llava", rows, [], tmp_path, monkeypatch, image_dir=".")

    assert [(_tag_of(row), row["image"]) for row in result["train"]] == [
        ("kept", _resolved(tmp_path / "media_root"))
    ]
    assert _skip_lines("image", 1, ".") in _plain(loader_output.getvalue())


# ---------------------------------------------------------------------------
# Values refused before any filesystem call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["absolute_dir", "relative_dir", "relative_media_dir"])
@pytest.mark.parametrize("fmt", _FORMATS)
def test_network_device_and_nul_values_are_skipped_before_realpath(
    fmt, layout, tmp_path, monkeypatch, loader_output
):
    dirs = _layout(tmp_path)
    refused = [*NETWORK_FORMS, "clips/ok\x00.bin"]
    rows = [_row(fmt, "clips/ok.bin", "kept")] + [_row(fmt, value, "refused") for value in refused]
    if layout == "absolute_dir":
        result = _load("single_file", fmt, rows, [], tmp_path, monkeypatch)
        base: Path | str = dirs["data"]
    else:
        # A relative directory keeps a rooted value rooted when the two are
        # joined, so only the pre-check stands between it and realpath.
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("data") / "rows.jsonl", rows)
        overrides = {_SETTING[fmt]: "media_root"} if layout == "relative_media_dir" else {}
        result = load_dataset(
            DataConfig(train="data/rows.jsonl", format=fmt, val_split=0.0, **overrides)
        )
        base = "media_root" if overrides else "data"

    assert [(_tag_of(row), row[_KEY[fmt]]) for row in result["train"]] == [
        ("kept", _resolved(tmp_path / base))
    ]
    assert _skip_lines(_KEY[fmt], len(refused), base) in _plain(loader_output.getvalue())


@pytest.mark.parametrize("key", ["image", "audio"])
@pytest.mark.parametrize(
    "value",
    [b"clips/ok.bin", Path("clips/ok.bin"), os.fsencode(NETWORK_FORMS[0])],
    ids=["bytes", "path-object", "bytes-network"],
)
def test_a_bytes_or_path_object_value_is_skipped_not_passed_on(
    key, value, tmp_path, loader_output
):
    dirs = _layout(tmp_path)
    validate = loader._validate_vision_images if key == "image" else loader._validate_audio_files

    rows = validate([{"messages": [], key: value}], dirs["data"])

    assert rows == []
    assert _skip_lines(key, 1, dirs["data"]) in _plain(loader_output.getvalue())


def test_a_hub_bytes_value_is_skipped_like_a_path_outside(tmp_path, monkeypatch, loader_output):
    dirs = _layout(tmp_path)
    rows = [_row("llava", "clips/ok.bin", "kept"), _row("llava", b"clips/ok.bin", "bytes")]
    _install_fake_datasets(monkeypatch, hub={_HUB_NAME: {"train": rows}})

    result = load_dataset(
        DataConfig(train=_HUB_NAME, format="llava", val_split=0.0, image_dir=str(dirs["media"]))
    )

    assert [(_tag_of(row), row["image"]) for row in result["train"]] == [
        ("kept", _resolved(dirs["media"]))
    ]
    assert _skip_lines("image", 1, dirs["media"]) in _plain(loader_output.getvalue())


def test_a_hub_bytes_value_without_a_media_dir_is_skipped_not_refused(
    monkeypatch, loader_output
):
    decoded = object()
    rows = [_row("llava", b"clips/ok.bin", "bytes"), _row("llava", decoded, "decoded")]
    _install_fake_datasets(monkeypatch, hub={_HUB_NAME: {"train": rows}})

    result = load_dataset(DataConfig(train=_HUB_NAME, format="llava", val_split=0.0))

    assert [row["image"] for row in result["train"]] == [decoded]
    assert "Warning: 1 rows skipped (image value is not a path string)" in _plain(
        loader_output.getvalue()
    )


def test_a_non_string_image_value_is_kept_as_is():
    decoded = object()

    rows = loader._validate_vision_images([{"messages": [], "image": decoded}], Path("unused"))

    assert rows[0]["image"] is decoded


_DOS_DEVICE_VALUES = ["nul", "clips/CON.png", "com1.wav", "LPT9", "aux.tar.gz", "conin$"]


@pytest.mark.skipif(os.name != "nt", reason="DOS device names are a Windows rule")
@pytest.mark.parametrize("fmt", _FORMATS)
def test_dos_device_names_are_skipped_on_windows(fmt, tmp_path, monkeypatch, loader_output):
    dirs = _layout(tmp_path)
    looked_up: list[str] = []
    guarded = os.path.realpath

    def spy(path, *args, **kwargs):
        looked_up.append(os.path.normcase(os.fsdecode(path)))
        return guarded(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", spy)
    rows = [_row(fmt, "clips/ok.bin", "kept")] + [
        _row(fmt, value, "device") for value in _DOS_DEVICE_VALUES
    ]

    result = _load("single_file", fmt, rows, [], tmp_path, monkeypatch)

    assert [_tag_of(row) for row in result["train"]] == ["kept"]
    candidates = {os.path.normcase(str(dirs["data"] / value)) for value in _DOS_DEVICE_VALUES}
    assert candidates.isdisjoint(looked_up)
    plain = _plain(loader_output.getvalue())
    assert _skip_lines(_KEY[fmt], len(_DOS_DEVICE_VALUES), dirs["data"]) in plain


@pytest.mark.skipif(os.name == "nt", reason="on Windows these name devices")
def test_dos_device_names_are_ordinary_file_names_elsewhere(tmp_path, monkeypatch):
    dirs = _layout(tmp_path)
    rows = [_row("llava", value, value) for value in _DOS_DEVICE_VALUES]

    result = _load("single_file", "llava", rows, [], tmp_path, monkeypatch)

    assert [row["image"] for row in result["train"]] == [
        _resolved(dirs["data"], value) for value in _DOS_DEVICE_VALUES
    ]


# ---------------------------------------------------------------------------
# The string tests behind the checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (r"\\host\share\x.png", True),
        ("//host/share/x.png", True),
        (r"\/host/share/x.png", True),
        (r"/\host\share\x.png", True),
        (r"\\?\C:\x.png", True),  # a local extended-length path is refused too
        (r"\\?\UNC\host\share\x.png", True),
        (r"\\.\pipe\name", True),
        (r"\??\C:\x.png", True),
        ("/??/C:/x.png", True),
        (r"\.\??\UNC\h\s\x.png", True),
        ("/./??/UNC/h/s/x.png", True),
        (r"\x\..\??\UNC\h\s\x.png", True),
        ("\\\\", True),
        ("//", True),
        ("///x", True),
        ("//tmp/data/x.png", True),  # POSIX reads this as /tmp/data/x.png
        (r"C:\data\x.png", False),
        ("C:/data/x.png", False),
        ("imgs/x.png", False),
        ("/srv/data/x.png", False),
        (r"\x.png", False),
        (r"\x\..\y.png", False),
        (r"a\??x", False),
        ("\\??", False),
        ("/?", False),
        ("", False),
    ],
)
def test_network_or_device_spellings(value, expected):
    assert is_network_or_device_path(value, dos_device_names=False) is expected
    assert is_network_or_device_path(value, dos_device_names=True) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("nul", True),
        ("NUL", True),
        ("Nul.png", True),
        ("clips/con", True),
        (r"clips\aux.wav", True),
        ("com1", True),
        ("COM9.txt", True),
        ("LPT1", True),
        ("com" + chr(0xB9), True),
        ("nul.", True),
        ("nul .png", True),
        ("con:stream", True),
        ("CONIN$", True),
        ("conout$.txt", True),
        ("clips/nul/x.png", True),
        ("null.png", False),
        ("console.png", False),
        ("com10", False),
        ("com0", False),
        ("acon.png", False),
        ("com.png", False),
        ("clips/nulls/x.png", False),
        (" nul", False),
    ],
)
def test_dos_device_names_are_a_parameter_of_the_check(value, expected):
    assert is_network_or_device_path(value, dos_device_names=True) is expected
    assert is_network_or_device_path(value, dos_device_names=False) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        *[(value, True) for value in DEVICE_NAMESPACE_FORMS],
        (r"\\.\NUL", True),
        (r"\\host\share\x.png", False),
        ("//host/share/x.png", False),
        (r"\\?\C:\x.png", False),
        (r"\\?\UNC\host\share\x.png", False),
        (r"C:\x.png", False),
        ("/srv/x.png", False),
        ("nul", False),
    ],
)
def test_device_namespace_spellings(value, expected):
    assert is_device_namespace_path(value) is expected


# ---------------------------------------------------------------------------
# Trainer side: only a regular file is opened
# ---------------------------------------------------------------------------


def _regular_stat() -> os.stat_result:
    return os.stat_result((stat.S_IFREG | 0o644, 0, 0, 1, 0, 0, 1, 0, 0, 0))


def _spy_lookups(monkeypatch, *, result=None) -> list[str]:
    """Record ``os.stat`` / ``os.lstat`` / ``os.path.exists`` calls on sentinel paths.

    A recorded call raises FileNotFoundError, or returns ``result`` when given.
    """
    seen: list[str] = []
    real = {"stat": os.stat, "lstat": os.lstat, "exists": os.path.exists}

    def spy(name):
        def lookup(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                text = os.fsdecode(path)
                if "soup-sentinel" in text:
                    seen.append(text)
                    if name == "exists":
                        return result is not None
                    if result is not None:
                        return result
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), text)
            return real[name](path, *args, **kwargs)

        return lookup

    monkeypatch.setattr(os, "stat", spy("stat"))
    monkeypatch.setattr(os, "lstat", spy("lstat"))
    monkeypatch.setattr(os.path, "exists", spy("exists"))
    return seen


class TestRequireRegularFile:
    def test_an_absolute_regular_file_is_returned(self, tmp_path):
        target = tmp_path / "ok.bin"
        target.write_bytes(b"x")

        assert require_regular_file(str(target)) == str(target)

    def test_a_relative_path_is_refused(self):
        with pytest.raises(OSError, match="not an absolute local file path"):
            require_regular_file("ok.bin")

    def test_a_nul_byte_is_an_oserror_not_a_valueerror(self, tmp_path):
        with pytest.raises(OSError, match="not an absolute local file path"):
            require_regular_file(str(tmp_path / "ok\x00.bin"))

    def test_a_directory_is_refused(self, tmp_path):
        with pytest.raises(OSError, match="not a regular file"):
            require_regular_file(str(tmp_path))

    def test_a_missing_file_is_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            require_regular_file(str(tmp_path / "missing.bin"))

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
    def test_a_fifo_is_refused_without_blocking(self, tmp_path):
        fifo = tmp_path / "pipe.bin"
        os.mkfifo(fifo)

        with pytest.raises(OSError, match="not a regular file"):
            require_regular_file(str(fifo))

    @pytest.mark.skipif(not os.path.exists("/dev/null"), reason="needs /dev/null")
    def test_a_device_is_refused(self):
        with pytest.raises(OSError, match="not a regular file"):
            require_regular_file("/dev/null")

    @pytest.mark.skipif(os.name != "nt", reason="DOS device names")
    def test_a_dos_device_name_is_refused(self, tmp_path):
        with pytest.raises(OSError, match="not a regular file"):
            require_regular_file(str(tmp_path / "nul"))

    @pytest.mark.parametrize("value", DEVICE_NAMESPACE_FORMS)
    def test_device_namespace_forms_are_refused_before_stat(self, value, monkeypatch):
        looked_up = _spy_lookups(monkeypatch)

        with pytest.raises(OSError, match="not an absolute local file path"):
            require_regular_file(value)
        assert looked_up == []

    def test_a_unc_path_is_not_refused_by_its_spelling(self, monkeypatch):
        # The loader resolves a media dir on a mapped drive to its UNC form; a
        # regular file there must stay readable.
        unc = "//soup-sentinel.invalid/share/x.bin"
        looked_up = _spy_lookups(monkeypatch, result=_regular_stat())

        assert require_regular_file(unc) == unc
        assert looked_up == [unc]

    @pytest.mark.skipif(os.name != "nt", reason="extended-length paths are Windows paths")
    def test_an_extended_length_local_path_is_not_refused(self, tmp_path):
        target = tmp_path / "ok.bin"
        target.write_bytes(b"x")
        extended = "\\\\?\\" + str(target)

        assert require_regular_file(extended) == extended


class TestLoadAudioMonoReadsOnlyRegularFiles:
    @pytest.fixture(autouse=True)
    def _soundfile_must_not_be_reached(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("soundfile reached a non-regular file")

        fake = types.ModuleType("soundfile")
        fake.info = refuse
        fake.read = refuse
        monkeypatch.setitem(sys.modules, "soundfile", fake)

    def _load_audio_mono(self, path):
        from soup_cli.utils.tts_codec import load_audio_mono

        return load_audio_mono(path)

    def test_a_directory_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="audio path must be a regular file"):
            self._load_audio_mono(str(tmp_path))

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
    def test_a_fifo_is_refused_without_blocking(self, tmp_path):
        fifo = tmp_path / "pipe.wav"
        os.mkfifo(fifo)

        with pytest.raises(ValueError, match="audio path must be a regular file"):
            self._load_audio_mono(str(fifo))

    @pytest.mark.skipif(not os.path.exists("/dev/null"), reason="needs /dev/null")
    def test_a_character_device_is_refused(self):
        with pytest.raises(ValueError, match="audio path must be a regular file"):
            self._load_audio_mono("/dev/null")

    @pytest.mark.skipif(os.name != "nt", reason="DOS device names")
    def test_a_dos_device_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="audio path must be a regular file"):
            self._load_audio_mono(str(tmp_path / "nul"))

    @pytest.mark.parametrize("value", DEVICE_NAMESPACE_FORMS)
    def test_a_device_namespace_path_is_refused_before_any_lookup(self, value, monkeypatch):
        looked_up = _spy_lookups(monkeypatch)

        with pytest.raises(ValueError, match="audio path must not be a device path"):
            self._load_audio_mono(value)
        assert looked_up == []


def _non_regular_paths(tmp_path: Path) -> list[str]:
    """A directory and a relative path, plus a FIFO and a device where the OS has them."""
    folder = tmp_path / "folder"
    folder.mkdir()
    paths = [str(folder), "ok.bin"]
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.bin"
        os.mkfifo(fifo)
        paths.append(str(fifo))
    if os.path.exists("/dev/null"):
        paths.append("/dev/null")
    if os.name == "nt":
        paths.append(str(tmp_path / "nul"))
    return paths


def _vision_wrapper(monkeypatch):
    pytest.importorskip("datasets")
    from soup_cli.trainer import sft
    from soup_cli.trainer.sft import SFTTrainerWrapper

    out = io.StringIO()
    monkeypatch.setattr(sft, "console", Console(file=out, width=400))
    return object.__new__(SFTTrainerWrapper), out


def _open_spy(pil_image, opened: list):
    real_open = pil_image.open

    def spy_open(fp, *args, **kwargs):
        if isinstance(fp, pil_image.Image):
            # A decoded value: hand it back, as a reader that accepts one would.
            opened.append(fp)
            return fp
        if isinstance(fp, (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(fp))
            if not os.path.isfile(fp):  # never block on a FIFO, whatever the code does
                raise OSError("spy: not a regular file")
        return real_open(fp, *args, **kwargs)

    return spy_open


_VISION_MESSAGES = [{"role": "user", "content": "<image>\nDescribe."}]


def test_vision_trainer_opens_only_regular_files(tmp_path, monkeypatch):
    pil_image = pytest.importorskip("PIL.Image")
    wrapper, out = _vision_wrapper(monkeypatch)
    image_path = tmp_path / "ok.png"
    pil_image.new("RGB", (4, 4), "red").save(image_path)
    refused = _non_regular_paths(tmp_path)
    rows = [{"messages": _VISION_MESSAGES, "image": str(image_path)}] + [
        {"messages": _VISION_MESSAGES, "image": path} for path in refused
    ]
    opened: list = []

    # Scoped to the call: reading the result back decodes the stored PNG
    # through PIL.Image.open(BytesIO), which must use the real function.
    with monkeypatch.context() as patched:
        patched.setattr(pil_image, "open", _open_spy(pil_image, opened))
        train_ds, _ = wrapper._prepare_vision_dataset({"train": rows})

    assert opened == [str(image_path)]
    assert [len(images) for images in train_ds["images"]] == [1] + [0] * len(refused)
    assert _plain(out.getvalue()).count("Warning: cannot open image:") == len(refused)


def test_vision_trainer_checks_a_bytes_path_too(tmp_path, monkeypatch):
    pil_image = pytest.importorskip("PIL.Image")
    wrapper, out = _vision_wrapper(monkeypatch)
    refused = [os.fsencode(path) for path in _non_regular_paths(tmp_path)]
    rows = [{"messages": _VISION_MESSAGES, "image": path} for path in refused]
    opened: list = []

    with monkeypatch.context() as patched:
        patched.setattr(pil_image, "open", _open_spy(pil_image, opened))
        wrapper._prepare_vision_dataset({"train": rows})

    assert opened == []
    assert _plain(out.getvalue()).count("Warning: cannot open image:") == len(refused)


def test_vision_trainer_hands_a_decoded_image_to_the_reader(tmp_path, monkeypatch):
    pil_image = pytest.importorskip("PIL.Image")
    wrapper, _ = _vision_wrapper(monkeypatch)
    rows = [{"messages": _VISION_MESSAGES, "image": pil_image.new("RGB", (4, 4), "blue")}]
    opened: list = []

    with monkeypatch.context() as patched:
        patched.setattr(pil_image, "open", _open_spy(pil_image, opened))
        train_ds, _ = wrapper._prepare_vision_dataset({"train": rows})

    assert [isinstance(value, pil_image.Image) for value in opened] == [True]
    assert [len(images) for images in train_ds["images"]] == [1]


_AUDIO_MESSAGES = [
    {"role": "user", "content": "Transcribe."},
    {"role": "assistant", "content": "hello"},
]


def _audio_wrapper(monkeypatch):
    pytest.importorskip("datasets")
    from soup_cli.trainer import sft
    from soup_cli.trainer.sft import SFTTrainerWrapper

    out = io.StringIO()
    monkeypatch.setattr(sft, "console", Console(file=out, width=400))
    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.processor = types.SimpleNamespace()  # no chat template: turns are joined
    return wrapper, out


def _fake_librosa(loaded: list):
    numpy = pytest.importorskip("numpy")

    def load(path, sr=None, mono=True):
        loaded.append(os.fsdecode(path) if isinstance(path, (str, bytes)) else path)
        return numpy.zeros(8, dtype=numpy.float32), 16000

    return types.SimpleNamespace(load=load)


def test_audio_trainer_hands_a_value_that_is_no_file_name_to_the_reader(monkeypatch):
    wrapper, _ = _audio_wrapper(monkeypatch)
    clip = {"array": [0.0, 0.0], "sampling_rate": 16000}
    loaded: list = []

    with monkeypatch.context() as patched:
        patched.setitem(sys.modules, "librosa", _fake_librosa(loaded))
        wrapper._prepare_audio_dataset({"train": [{"messages": _AUDIO_MESSAGES, "audio": clip}]})

    assert loaded == [clip]


def test_audio_trainer_reads_only_regular_files(tmp_path, monkeypatch):
    wrapper, out = _audio_wrapper(monkeypatch)
    messages = _AUDIO_MESSAGES
    loaded: list = []
    paths = _non_regular_paths(tmp_path)
    refused = [{"messages": messages, "audio": path} for path in paths]
    # A bytes value is a file name to the reader too; Arrow keeps it in a
    # column of its own, so it goes in a call of its own.
    refused_bytes = [{"messages": messages, "audio": os.fsencode(path)} for path in paths]
    clip = tmp_path / "ok.wav"
    clip.write_bytes(b"RIFF")

    # Only non-readable rows go in the first calls: mixing read and unread rows
    # in one Dataset.map hits a column mismatch that predates this change.
    with monkeypatch.context() as patched:
        patched.setitem(sys.modules, "librosa", _fake_librosa(loaded))
        wrapper._prepare_audio_dataset({"train": refused})
        wrapper._prepare_audio_dataset({"train": refused_bytes})
        assert loaded == []
        wrapper._prepare_audio_dataset({"train": [{"messages": messages, "audio": str(clip)}]})

    assert loaded == [str(clip)]
    assert _plain(out.getvalue()).count("Warning: cannot open audio:") == 2 * len(paths)


# ---------------------------------------------------------------------------
# soup infer --task asr and soup data inspect use the same check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", NETWORK_FORMS)
def test_infer_asr_refuses_every_network_and_device_spelling(value):
    from soup_cli.commands.infer import _resolve_asr_audio

    with pytest.raises(ValueError, match="UNC"):
        _resolve_asr_audio(value, Path("audio"))


def test_soup_infer_asr_skips_network_and_device_rows(tmp_path, monkeypatch):
    import soup_cli.commands.infer as infer
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "audio").mkdir()
    monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", lambda audio_path: "hi")
    refused = [
        r"\??\UNC\soup-sentinel.invalid\share\x.wav",
        r"\/soup-sentinel.invalid/share/x.wav",
        r"\.\??\UNC\soup-sentinel.invalid\share\x.wav",
    ]
    _write_jsonl(
        tmp_path / "in.jsonl",
        [{"audio": value, "text": "x"} for value in refused] + [{"audio": "ok.wav", "text": "hi"}],
    )

    result = CliRunner().invoke(
        app,
        ["infer", "--task", "asr", "--model", "m", "--input", "in.jsonl",
         "--output", "out.jsonl", "--audio-dir", "audio"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    written = _read_jsonl(tmp_path / "out.jsonl")
    assert [row["audio"] for row in written] == ["ok.wav"]
    plain = _plain(result.output)
    assert plain.count("audio path must not be a UNC / network / device path") == len(refused)


def _inspect_rows(tmp_path: Path, rows: list[dict]):
    from soup_cli.commands.data import app as data_app

    _write_jsonl(tmp_path / "rows.jsonl", rows)
    result = CliRunner().invoke(data_app, ["inspect", "rows.jsonl", "--rows", "0"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    found = re.search(r"Images found on disk\W+(\d+)", _plain(result.output))
    assert found, _plain(result.output)
    return int(found.group(1))


def test_data_inspect_never_looks_up_network_or_device_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ok.png").write_bytes(b"x")
    looked_up = _spy_lookups(monkeypatch)
    rows = [{"image": "ok.png", "conversations": []}] + [
        {"image": value, "conversations": []} for value in [*NETWORK_FORMS, "ok\x00.png"]
    ]

    found = _inspect_rows(tmp_path, rows)

    assert found == 1
    assert looked_up == []


@pytest.mark.skipif(os.name != "nt", reason="DOS device names")
def test_data_inspect_does_not_count_a_dos_device_as_an_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _inspect_rows(tmp_path, [{"image": "nul", "conversations": []}]) == 0


def test_data_inspect_skips_a_value_that_is_not_a_string(monkeypatch):
    from soup_cli.commands import data

    out = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=out, width=400))
    looked_up = _spy_lookups(monkeypatch)

    data._show_vision_stats(
        [{"image": os.fsencode(NETWORK_FORMS[0]), "conversations": []}, {"image": {"bytes": b""}}]
    )

    assert looked_up == []
    assert re.search(r"Images found on disk\W+0", _plain(out.getvalue()))
