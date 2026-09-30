"""#1181: ``load_dataset`` dropped rows its format converter rejected, silently.

The drop is the documented contract (``format_to_messages`` returns ``None`` "so
one bad line does not abort a whole dataset load"); the defect was that nothing
said so at train time. Measured with ``soup train`` on a 4-row ``chatml`` file
with one misspelled ``messages`` key: ``Map: 3/3``, exit 0, not a word. The media
paths in the same module already print ``rows skipped``.

The loader's module-level console is swapped for one writing to a buffer at a
fixed width with no colour, so the assertions see the whole line.
"""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

import soup_cli.data.loader as loader
from soup_cli.config.loader import load_config_from_string


def _ok(i):
    return {"messages": [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": f"a{i}"},
    ]}


_TYPO = {"mesages": [
    {"role": "user", "content": "typo"},
    {"role": "assistant", "content": "x"},
]}


@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buf, width=400, color_system=None)
    )
    return buf


def _load(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    cfg = load_config_from_string(
        "base: x\ntask: sft\ndata:\n  train: d.jsonl\n  format: chatml\n  val_split: 0\n"
    )
    return loader.load_dataset(cfg.data)


class TestADroppedRowIsReported:
    def test_the_count_the_row_and_the_reason_are_printed(self, tmp_path, monkeypatch, out):
        ds = _load(tmp_path, monkeypatch, [_ok(0), _TYPO, _ok(2), _ok(3)])

        assert len(ds["train"]) == 3, "the drop itself is the documented contract"
        text = " ".join(out.getvalue().split())
        assert "Warning: 1 of 4 rows dropped" in text
        assert "First: row 1: 'messages'" in text
        assert "soup data validate d.jsonl --format chatml" in text

    def test_the_count_matches_soup_data_validate(self, tmp_path, monkeypatch, out):
        """#695 made validate agree with the loader; this keeps the loader's own
        line agreeing with validate."""
        from soup_cli.data.validator import validate_and_stats

        rows = [_ok(0), _TYPO, _ok(2), _TYPO, _ok(4)]
        _load(tmp_path, monkeypatch, rows)
        stats = validate_and_stats(rows, "chatml")

        dropped = len(rows) - stats["valid_rows"]
        assert dropped == 2
        assert f"Warning: {dropped} of {len(rows)} rows dropped" in out.getvalue()
        assert "First: row 1:" in out.getvalue(), "the first dropped row, not the last (3)"

    def test_without_a_local_file_there_is_no_validate_hint(self, out):
        """A Hub or remote split has no path ``soup data validate`` can take."""
        loader._format_rows([_ok(0), _TYPO], "chatml")

        text = out.getvalue()
        assert "1 of 2 rows dropped" in text
        assert "soup data validate" not in text


class TestTheControl:
    def test_a_clean_file_prints_no_drop_line(self, tmp_path, monkeypatch, out):
        ds = _load(tmp_path, monkeypatch, [_ok(i) for i in range(4)])

        assert len(ds["train"]) == 4
        assert "dropped" not in out.getvalue()
