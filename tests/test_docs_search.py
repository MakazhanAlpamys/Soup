"""Tests for documentation search, BM25 indexing, caching, and CLI interface."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.docs.bm25 import PureBM25
from soup_cli.docs.chunker import split_markdown_by_headings
from soup_cli.docs.retriever import (
    get_default_cache_dir,
    load_cache,
    load_docs,
    save_cache,
    search_docs,
)
from soup_cli.docs.tokenizer import tokenize
from soup_cli.utils.terminal import for_terminal
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.mark.unit
def test_tokenizer_technical_tokens():
    """Tokenizer preserves CLI flags, version strings, compound identifiers, and paths."""
    text = (
        "Use --gradient-accumulation-steps=4 with torch.nn.Linear and v0.77.1 "
        "at path/to/model.safetensors!"
    )
    tokens = tokenize(text)

    assert "--gradient-accumulation-steps" in tokens
    assert "gradient" in tokens
    assert "accumulation" in tokens
    assert "steps" in tokens
    assert "4" in tokens
    assert "torch.nn.linear" in tokens
    assert "torch" in tokens
    assert "nn" in tokens
    assert "linear" in tokens
    assert "v0.77.1" in tokens
    assert "0.77.1" in tokens
    assert "path/to/model.safetensors" in tokens
    assert "model.safetensors" in tokens


@pytest.mark.unit
def test_tokenizer_edge_cases():
    """Tokenizer handles empty, whitespace, and pure-punctuation strings."""
    assert tokenize("") == []
    assert tokenize("    \n\t  ") == []
    assert tokenize("!@#$%^&*()_+-=[]{}|;':\",./<>?") == []


@pytest.mark.unit
def test_chunker_heading_hierarchy():
    """Chunker tracks heading hierarchy and preserves sections."""
    md = """# Guide Title

Introductory overview paragraph.

## Section One

Body text for section one.

### Subsection A

Details for subsection A.

## Section Two

Body text for section two.
"""
    chunks = split_markdown_by_headings(md, "test-doc")
    assert len(chunks) == 4

    assert chunks[0]["doc_name"] == "test-doc"
    assert chunks[0]["section"] == "Guide Title"
    assert chunks[0]["heading_prefix"] == "Guide Title"
    assert "Introductory overview" in chunks[0]["text"]

    assert chunks[1]["section"] == "Section One"
    assert chunks[1]["heading_prefix"] == "Guide Title > Section One"
    assert "Body text for section one" in chunks[1]["text"]

    assert chunks[2]["section"] == "Subsection A"
    assert chunks[2]["heading_prefix"] == "Guide Title > Section One > Subsection A"
    assert "Details for subsection A" in chunks[2]["text"]

    assert chunks[3]["section"] == "Section Two"
    assert chunks[3]["heading_prefix"] == "Guide Title > Section Two"
    assert "Body text for section two" in chunks[3]["text"]


@pytest.mark.unit
def test_chunker_code_block_fence_protection():
    """Lines starting with '#' inside code fences must not be parsed as ATX headings."""
    md = """# Execution Guide

Overview text.

## Running Jobs

Run a command:

```bash
# Auto-detect latest checkpoint
soup train --resume auto

# Resume from a specific checkpoint
soup train --resume ./output/checkpoint-500
```

End of section.
"""
    chunks = split_markdown_by_headings(md, "execution")
    assert len(chunks) == 2

    assert chunks[0]["section"] == "Execution Guide"
    assert chunks[1]["section"] == "Running Jobs"
    assert chunks[1]["heading_prefix"] == "Execution Guide > Running Jobs"
    assert "# Auto-detect latest checkpoint" in chunks[1]["text"]
    assert "# Resume from a specific checkpoint" in chunks[1]["text"]
    assert "soup train --resume auto" in chunks[1]["text"]
    assert "soup train --resume ./output/checkpoint-500" in chunks[1]["text"]


@pytest.mark.unit
def test_pure_bm25_formula_and_negative_idf_floor():
    """PureBM25 computes IDF with average over all terms and applies epsilon floor."""
    # Term 'common' is in all 4 docs (df=4, N=4) -> negative raw IDF
    # Term 'rare' is in 1 doc (df=1, N=4) -> positive raw IDF
    docs = [
        ["common", "rare", "alpha"],
        ["common", "beta"],
        ["common", "gamma"],
        ["common", "delta"],
    ]
    bm25 = PureBM25(docs, k1=1.5, b=0.75, eps=0.25)

    assert bm25.n_docs == 4
    assert bm25.avgdl == (3 + 2 + 2 + 2) / 4.0

    # Raw IDF for 'common': ln((4 - 4 + 0.5) / (4 + 0.5)) = ln(0.5 / 4.5) = ln(1/9) < 0
    raw_common_idf = math.log(0.5 / 4.5)
    assert raw_common_idf < 0

    # The stored IDF must be floored to eps * average_idf
    assert bm25.idf["common"] > 0
    assert bm25.idf["rare"] > bm25.idf["common"]

    # Inverted index posting check
    assert len(bm25.postings["common"]) == 4
    assert len(bm25.postings["rare"]) == 1
    assert bm25.postings["rare"][0] == (0, 1)


@pytest.mark.unit
def test_pure_bm25_deterministic_tie_breaking():
    """Ties in score are broken deterministically by document ID ascending."""
    docs = [
        ["exact", "match"],
        ["exact", "match"],
        ["unrelated", "other"],
    ]
    bm25 = PureBM25(docs)
    ranked = bm25.rank(["exact", "match"], top_k=2)

    assert len(ranked) == 2
    assert math.isclose(ranked[0][1], ranked[1][1], rel_tol=1e-7)
    # Doc 0 must precede Doc 1 due to tie-breaking
    assert ranked[0][0] == 0
    assert ranked[1][0] == 1


@pytest.mark.unit
def test_pure_bm25_equivalence_with_rank_bm25():
    """Verify that PureBM25 produces identical scores and rankings to rank_bm25 when installed."""
    rank_bm25 = pytest.importorskip("rank_bm25")

    corpus = [
        ["fine", "tuning", "lora", "adapters", "safetensors", "model"],
        ["quantization", "bitsandbytes", "nf4", "fp8", "vram"],
        ["export", "merge", "safetensors", "checkpoint", "weights"],
        ["dpo", "preference", "dataset", "reward", "loss"],
        ["common", "term", "appears", "everywhere", "fine", "tuning"],
        ["common", "term", "appears", "everywhere", "lora"],
        ["common", "term", "appears", "everywhere", "quantization"],
        ["common", "term", "appears", "everywhere", "dpo"],
    ]

    ref = rank_bm25.BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)
    pure = PureBM25(corpus, k1=1.5, b=0.75, eps=0.25)

    queries = [
        ["lora", "safetensors"],
        ["common", "term"],
        ["nf4", "vram", "quantization"],
        ["nonexistent", "term"],
        ["dpo", "reward", "preference"],
    ]

    for q in queries:
        ref_scores = ref.get_scores(q)
        pure_scores = pure.score(q)

        assert len(ref_scores) == len(pure_scores)
        for r, p in zip(ref_scores, pure_scores):
            assert math.isclose(r, p, rel_tol=1e-5, abs_tol=1e-6)

        ref_top = sorted(range(len(ref_scores)), key=lambda i: (-ref_scores[i], i))
        pure_top = [idx for idx, _ in pure.rank(q, top_k=len(corpus))]
        assert ref_top == pure_top


@pytest.mark.unit
def test_pure_bm25_edge_cases():
    """PureBM25 handles empty queries, OOV terms, empty docs, single doc, and extreme lengths."""
    # 1. Empty corpus
    empty_bm25 = PureBM25([])
    assert empty_bm25.score(["test"]) == []
    assert empty_bm25.rank(["test"]) == []

    # 2. Corpus with empty docs and varied lengths
    docs = [
        [],
        ["single"],
        ["extreme", "length"] + ["filler"] * 500,
    ]
    bm25 = PureBM25(docs)

    # Empty query
    assert bm25.score([]) == [0.0, 0.0, 0.0]
    assert bm25.rank([]) == [(0, 0.0), (1, 0.0), (2, 0.0)]

    # Out-of-vocabulary query
    assert bm25.score(["completely", "absent"]) == [0.0, 0.0, 0.0]

    # Query matching single doc
    scores = bm25.score(["single"])
    assert scores[1] > 0
    assert scores[0] == 0.0
    assert scores[2] == 0.0

    # 3. Single document corpus: every term is in 100% of docs, producing negative IDF
    # floored to eps * avg_idf (matching rank_bm25 behavior exactly)
    single_doc_bm25 = PureBM25([["only", "one", "document"]])
    assert single_doc_bm25.n_docs == 1
    expected_score = 0.25 * math.log(0.5 / 1.5)
    assert math.isclose(single_doc_bm25.score(["only"])[0], expected_score, rel_tol=1e-5)
    assert single_doc_bm25.score(["absent"])[0] == 0.0


@pytest.mark.unit
def test_pure_bm25_serialization_roundtrip():
    """Serialization to dict and reconstitution via from_dict preserves all attributes."""
    docs = [
        ["lora", "target", "modules"],
        ["quantization", "bitsandbytes", "nf4"],
        ["export", "safetensors", "lora"],
    ]
    bm25 = PureBM25(docs)
    data = bm25.to_dict()

    rebuilt = PureBM25.from_dict(data)
    assert rebuilt.n_docs == bm25.n_docs
    assert rebuilt.avgdl == bm25.avgdl
    assert rebuilt.doc_lens == bm25.doc_lens
    assert rebuilt.idf == bm25.idf
    assert dict(rebuilt.postings) == dict(bm25.postings)

    # Score comparisons must be bit-exact
    q = ["lora", "safetensors"]
    assert bm25.score(q) == rebuilt.score(q)


@pytest.mark.integration
@pytest.mark.parametrize(
    "has_pyproject,has_source",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_load_docs_requires_checkout_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_pyproject: bool, has_source: bool
):
    """Stray docs beside site-packages must not override the wheel's bundled guides."""
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("checkout guide", encoding="utf-8")
    if has_pyproject:
        (root / "pyproject.toml").touch()
    if has_source:
        (root / "src" / "soup_cli").mkdir(parents=True)

    package = tmp_path / "package"
    (package / "bundled").mkdir(parents=True)
    (package / "bundled" / "guide.md").write_text("bundled guide", encoding="utf-8")
    module_path = root / "site-packages" / "soup_cli" / "docs" / "retriever.py"
    monkeypatch.setattr("soup_cli.docs.retriever.__file__", str(module_path))
    monkeypatch.setattr("soup_cli.docs.retriever.resources.files", lambda _package: package)

    expected = "checkout guide" if has_pyproject and has_source else "bundled guide"
    assert load_docs() == {"guide.md": expected}


@pytest.mark.unit
def test_cache_save_load_and_invalidation(tmp_path: Path):
    """Index cache saves atomically and invalidates on fingerprint or schema changes."""
    cache_file = tmp_path / "docs_index.json"
    fingerprint = "abc123sha256"
    chunks = [{"doc_name": "doc1", "section": "Sec", "heading_prefix": "P", "text": "body"}]
    index = PureBM25([["body"]])

    # 1. Save cache
    save_cache(cache_file, fingerprint, chunks, index)
    assert cache_file.is_file()

    # 2. Valid load
    loaded = load_cache(cache_file, fingerprint)
    assert loaded is not None
    loaded_index, loaded_chunks = loaded
    assert len(loaded_chunks) == 1
    assert loaded_chunks[0]["doc_name"] == "doc1"
    assert loaded_index.n_docs == 1

    # 3. Fingerprint mismatch forces None
    assert load_cache(cache_file, "wrong_fingerprint") is None

    # 4. Tampered schema version forces None
    with open(cache_file, "r", encoding="utf-8") as f:
        tampered = json.load(f)
    tampered["schema_version"] = 999
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(tampered, f)
    assert load_cache(cache_file, fingerprint) is None


@pytest.mark.unit
def test_cache_corrupted_json_recovery(tmp_path: Path):
    """Corrupted, empty, or truncated JSON cache recovers gracefully by rebuilding."""
    cache_file = tmp_path / "docs_index.json"
    fingerprint = "abc123sha256"

    # 1. Zero-byte file
    cache_file.write_text("", encoding="utf-8")
    assert load_cache(cache_file, fingerprint) is None

    # 2. Truncated / invalid JSON syntax
    cache_file.write_text('{"schema_version": 1, "chunks": [', encoding="utf-8")
    assert load_cache(cache_file, fingerprint) is None

    # 3. Valid JSON but missing required fields
    cache_file.write_text('{"schema_version": 1}', encoding="utf-8")
    assert load_cache(cache_file, fingerprint) is None


@pytest.mark.unit
def test_cache_read_only_directory_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When cache directory cannot be written to, search still succeeds in-memory."""
    ro_dir = tmp_path / "readonly_cache"
    ro_dir.mkdir(parents=True)

    # Mock save_cache to simulate PermissionError
    def mock_save_fail(*args, **kwargs):
        raise PermissionError("Read-only filesystem")

    monkeypatch.setattr("soup_cli.docs.retriever.save_cache", mock_save_fail)

    # Search should still succeed by computing index in-memory
    results = search_docs("safetensors", top_k=2, cache_dir=ro_dir)
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.unit
def test_cache_config_invalidation(tmp_path: Path):
    """Index cache invalidates if indexing hyperparameters change."""
    cache_file = tmp_path / "docs_index.json"
    fingerprint = "hash123"
    chunks = [{"doc_name": "d", "section": "s", "heading_prefix": "p", "text": "t"}]
    index = PureBM25([["t"]])

    save_cache(cache_file, fingerprint, chunks, index)
    assert load_cache(cache_file, fingerprint) is not None

    # Tamper config in file
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["config"]["k1"] = 999.0
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    assert load_cache(cache_file, fingerprint) is None


@pytest.mark.unit
def test_cache_default_dir_resolution():
    """Default cache directory resolves to ~/.soup/cache/docs."""
    expected = Path.home() / ".soup" / "cache" / "docs"
    assert get_default_cache_dir() == expected


@pytest.mark.integration
def test_bm25_zero_torch_import(tmp_path: Path):
    """BM25 search does not import torch or transformers."""
    probe = textwrap.dedent("""
        import sys
        from pathlib import Path
        from soup_cli.docs.retriever import search_docs

        for _ in range(2):
            results = search_docs("safetensors", top_k=3, cache_dir=Path(sys.argv[1]))
            assert results
        assert "torch" not in sys.modules, "BM25 search imported torch"
        assert "transformers" not in sys.modules, "BM25 search imported transformers"
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


@pytest.mark.unit
def test_cli_docs_search_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI 'soup docs search' returns table with relevant documents."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    result = runner.invoke(app, ["docs", "search", "safetensors", "--top-k", "3"])
    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    assert "Documentation Search:" in text
    assert "safetensors" in text
    assert (
        "adapters-and-governance" in text or "cli-reference" in text or "serving-and-export" in text
    )


@pytest.mark.unit
def test_cli_docs_search_json_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI 'soup docs search --format json' satisfies the public contract schema."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    result = runner.invoke(
        app, ["docs", "search", "safetensors", "--format", "json", "--top-k", "2"]
    )
    assert result.exit_code == 0, result.output
    # Output must be pure JSON with newline
    assert result.stdout.endswith("\n")
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert 1 <= len(data) <= 2

    # Assert exact contract keys
    required_keys = {"doc_name", "section", "heading_prefix", "text", "score"}
    for hit in data:
        assert set(hit.keys()) == required_keys
        assert isinstance(hit["doc_name"], str)
        assert isinstance(hit["section"], str)
        assert isinstance(hit["heading_prefix"], str)
        assert isinstance(hit["text"], str)
        assert isinstance(hit["score"], (float, int))
        assert hit["score"] > 0


@pytest.mark.unit
def test_cli_docs_search_no_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI handles query with no matching terms gracefully."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    result = runner.invoke(app, ["docs", "search", "nonexistentquerywordxyz12345"])
    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    assert "No matching documentation found" in text


@pytest.mark.unit
def test_cli_docs_search_terminal_hygiene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Control characters and Rich markup syntax in query/text are sanitized for terminal."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    hostile_query = "[red]safetensors[/red]\x1b[31m\x00\u202e"
    result = runner.invoke(app, ["docs", "search", hostile_query, "--top-k", "1"])
    assert result.exit_code == 0, result.output
    plain = " ".join(strip_ansi(result.stdout).split())
    assert "[red]safetensors[/red]" in plain
    # Verify no raw escape codes or raw unescaped control codes leak
    assert "\x1b[31m" not in result.stdout
    assert "\x00" not in result.stdout
    assert "\u202e" not in result.stdout

    # Verify for_terminal unit behavior on malicious strings
    sanitized = for_terminal("[bold red]hello[/bold red]\x1b[31m\x00world\u202e")
    assert "\\[bold red]" in sanitized
    assert "\x1b" not in sanitized
    assert "\x00" not in sanitized
    assert "\u202e" not in sanitized


@pytest.mark.unit
def test_cli_docs_search_invalid_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI exits with code 2 when an unknown output format is specified."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    result = runner.invoke(app, ["docs", "search", "safetensors", "--format", "yaml"])
    assert result.exit_code == 2
    plain = " ".join(strip_ansi(result.output).split())
    assert "Unknown --format" in plain
    assert "Choose 'table' or 'json'" in plain


@pytest.mark.unit
def test_cli_docs_search_invalid_top_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI exits with code 2 when --top-k is less than 1."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    r1 = runner.invoke(app, ["docs", "search", "safetensors", "--top-k", "0"])
    assert r1.exit_code == 2
    plain = " ".join(strip_ansi(r1.output).split())
    assert "--top-k must be greater than or equal to 1" in plain

    r2 = runner.invoke(app, ["docs", "search", "safetensors", "--top-k", "-1"])
    assert r2.exit_code == 2
    plain = " ".join(strip_ansi(r2.output).split())
    assert "--top-k must be greater than or equal to 1" in plain


@pytest.mark.unit
def test_cli_docs_search_missing_docs_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """CLI exits with code 1 and prints clean error when documentation cannot be loaded."""
    monkeypatch.setattr("soup_cli.docs.retriever.get_default_cache_dir", lambda: tmp_path)

    def mock_load_missing():
        raise FileNotFoundError("No documentation files found in docs/ or package data")

    monkeypatch.setattr("soup_cli.docs.retriever.load_docs", mock_load_missing)

    result = runner.invoke(app, ["docs", "search", "safetensors"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "No documentation files found" in result.output


@pytest.mark.unit
def test_wheel_force_include_parity():
    """Verify that every docs/*.md file is explicitly mapped in pyproject.toml force-include."""
    import tomllib

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    pyproject_file = repo_root / "pyproject.toml"

    assert docs_dir.is_dir()
    assert pyproject_file.is_file()

    actual_docs = {p.name for p in docs_dir.glob("*.md")}

    with open(pyproject_file, "rb") as f:
        pyproject = tomllib.load(f)

    force_include = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("force-include", {})
    )
    mapped = {Path(key).name for key in force_include}
    assert actual_docs == mapped, sorted(actual_docs ^ mapped)

    for doc in actual_docs:
        key = f"docs/{doc}"
        assert key in force_include, f"Missing force-include mapping for {doc}"
        assert force_include[key] == f"soup_cli/docs/bundled/{doc}"

    # Verify assets directory is excluded from wheel force-include
    for key in force_include:
        assert not key.startswith("docs/assets"), f"Assets mapped in force-include: {key}"
