"""#2057: document readers have an installable extra and actionable CLI errors."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.requirements import Requirement
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import data
from tests.conftest import strip_ansi

ROOT = Path(__file__).resolve().parents[1]
INSTALL = 'pip install "soup-cli[documents]"'


@pytest.fixture(autouse=True)
def ingest_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "console", Console(
        width=100, force_terminal=True, color_system="standard", no_color=False,
        legacy_windows=False,
    ))


def _ingest(source: Path, output: str = "rows.jsonl") -> list[dict]:
    result = CliRunner().invoke(app, ["data", "ingest", source.name, "--output", output])
    assert result.exit_code == 0, (result.output, result.exception)
    return [json.loads(line) for line in source.with_name(output).read_text("utf-8").splitlines()]


def test_documents_extra_declares_both_reader_floors() -> None:
    # Pin the declared requirements independently of optional reader installs.
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    table = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    match = re.search(r"^documents\s*=\s*(\[[^\]]*\])", table, re.M)
    assert match, "no installable documents extra"
    requirements = [Requirement(value) for value in ast.literal_eval(match.group(1))]
    assert {str(requirement) for requirement in requirements} == {
        "pypdf>=6.21.2", "python-docx>=1.2.0",
    }


def test_documents_stays_opt_in_with_the_all_extra() -> None:
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    match = re.search(r"^all\s*=\s*(\[.*\])$", text, re.M)
    assert match, "no all extra"
    assert all(
        "documents" not in Requirement(value).extras for value in ast.literal_eval(match.group(1))
    )


@pytest.mark.parametrize("extension,module,package", [
    ("pdf", "pypdf", "pypdf"), ("docx", "docx", "python-docx"),
])
def test_missing_reader_names_package_and_installable_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str, module: str, package: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, module, None)
    source = tmp_path / f"document.{extension}"
    source.write_bytes(b"reader must not be called")
    output = tmp_path / "rows.jsonl"
    output.write_text("existing output\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["data", "ingest", source.name, "--output", output.name], color=True,
    )
    assert result.exit_code == 1, (result.output, result.exception)
    assert "\x1b[" in result.output  # The real rendering console exercised colour.
    text = " ".join(strip_ansi(result.output).split())
    assert f"{package} not installed." in text
    assert INSTALL in text
    assert output.read_text("utf-8") == "existing output\n"


def test_pdf_reader_keeps_each_page_and_empty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "report.pdf"
    source.write_bytes(b"stubbed PDF")
    calls = []

    def reader(path: str) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(pages=[
            SimpleNamespace(extract_text=lambda: "First page"),
            SimpleNamespace(extract_text=lambda: None),
        ])

    module = ModuleType("pypdf")
    module.PdfReader = reader
    monkeypatch.setitem(sys.modules, "pypdf", module)
    assert _ingest(source) == [
        {"text": "First page", "source": "report.pdf", "page": 0},
        {"text": "", "source": "report.pdf", "page": 1},
    ]
    assert calls == [str(source)]


def test_docx_reader_strips_text_but_preserves_physical_paragraph_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.docx"
    source.write_bytes(b"stubbed DOCX")
    calls = []

    def document(path: str) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(paragraphs=[
            SimpleNamespace(text=text) for text in ("  First paragraph  ", " ", "Second paragraph")
        ])

    module = ModuleType("docx")
    module.Document = document
    monkeypatch.setitem(sys.modules, "docx", module)
    assert _ingest(source) == [
        {"text": "First paragraph", "source": "notes.docx", "para": 0},
        {"text": "Second paragraph", "source": "notes.docx", "para": 2},
    ]
    assert calls == [str(source)]


@pytest.mark.parametrize("extension,text,expected", [
    ("txt", "Hello\nWorld", [{"text": "Hello\nWorld", "source": "notes.txt"}]),
    ("md", "# Title\n\nBody", [
        {"text": "Body", "section": "Title", "level": 1, "source": "notes.md"},
    ]),
])
def test_builtin_formats_need_no_document_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str, text: str,
    expected: list[dict],
) -> None:
    monkeypatch.chdir(tmp_path)
    for module in ("pypdf", "docx", "markdown"):
        monkeypatch.setitem(sys.modules, module, None)
    source = tmp_path / f"notes.{extension}"
    source.write_text(text, encoding="utf-8")
    assert _ingest(source) == expected


def test_real_pdf_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pypdf = pytest.importorskip("pypdf")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "report.pdf"
    # A tiny text PDF using only the reader's dependency; no PDF renderer required.
    writer = pypdf.PdfWriter()
    for text in ("First page", "Second page"):
        page = writer.add_blank_page(width=300, height=200)
        font = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
            pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
            pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica"),
        })
        page[pypdf.generic.NameObject("/Resources")] = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Font"): pypdf.generic.DictionaryObject({
                pypdf.generic.NameObject("/F1"): font,
            }),
        })
        stream = pypdf.generic.DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii"))
        page[pypdf.generic.NameObject("/Contents")] = stream
    writer.write(source)
    assert _ingest(source) == [
        {"text": "First page", "source": "report.pdf", "page": 0},
        {"text": "Second page", "source": "report.pdf", "page": 1},
    ]


def test_real_docx_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docx = pytest.importorskip("docx")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.docx"
    document = docx.Document()
    for text in ("  First paragraph  ", "", "Второй абзац"):
        document.add_paragraph(text)
    document.save(source)
    assert _ingest(source) == [
        {"text": "First paragraph", "source": "notes.docx", "para": 0},
        {"text": "Второй абзац", "source": "notes.docx", "para": 2},
    ]
