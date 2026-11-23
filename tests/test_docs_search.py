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
    _query_expansions,
    _rank_chunks,
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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("fine-tune on a Mac", {"macs": 0.7, "mlx": 0.65}),
        ("train on Macs", {"mac": 0.7, "mlx": 0.65}),
        ("continue training", {"resume": 0.65}),
        ("continue training continue training", {"resume": 0.65}),
        ("batch size", {"batch-size": 0.65}),
        ("resume resume continue training", {}),
        ("mlx fine-tune on a Mac", {"macs": 0.7}),
        ("foo/mac --macs C:\\mac", {}),
        ("--continue training --batch size", {}),
        ("24GB v0.78.0 org/model learning_rate", {}),
        ("preference DPO KTO ORPO", {}),
    ],
)
def test_query_expansions_preserve_literal_and_technical_terms(query, expected):
    assert _query_expansions(query)[0] == expected


@pytest.mark.parametrize(
    "query", ["--no reference model", "--without a reference model", "no/reference model"]
)
def test_reference_free_rule_requires_prose_boundaries(query):
    assert not _query_expansions(query)[1]


@pytest.mark.parametrize("query", ["foo/mac", "--macs", "24GB", "org/model", "v0.78.0"])
def test_technical_only_queries_keep_original_ranking(query):
    from soup_cli.docs.retriever import build_index

    chunks = [
        {"section": word, "heading_prefix": word, "text": f"{word} other"}
        for word in ["Mac", "Macs", "24GB", "org/model", "v0.78.0", "alpha", "beta"]
    ]
    index = build_index(chunks)
    assert _rank_chunks(query, index, chunks, len(chunks)) == index.rank(
        tokenize(query), len(chunks)
    )


def test_generic_preference_keeps_methods_and_reference_free_is_not_orpo_only():
    from soup_cli.docs.retriever import build_index

    chunks = [
        {"section": heading, "heading_prefix": heading, "text": body}
        for heading, body in [
            ("ORPO (No Reference Model)", "preference training alignment"),
            ("SimPO (Reference-free)", "preference training alignment"),
            ("DPO", "preference training reference model"),
            ("KTO", "preference training examples"),
            ("Formats", "alpha beta gamma delta epsilon zeta"),
            ("Export", "eta theta iota kappa lambda mu"),
            ("Serving", "nu xi omicron pi rho sigma"),
            ("Hardware", "tau upsilon phi chi psi omega"),
            ("Checkpoints", "save checkpoint restore"),
        ]
    ]
    index = build_index(chunks)
    generic = _rank_chunks("preference", index, chunks, len(chunks))
    assert generic == index.rank(["preference"], len(chunks))
    assert {doc_id for doc_id, score in generic if score > 0} == {0, 1, 2, 3}
    reference_free = _rank_chunks("preference training without a reference model", index, chunks, 2)
    assert {doc_id for doc_id, _ in reference_free} == {0, 1}


@pytest.mark.parametrize(
    ("query", "sections", "bodies", "expected"),
    [
        ("Mac", ["Target", "Decoy"], ["macs", "unrelated"], 0),
        ("continue training", ["Target", "Decoy"], ["resume", "unrelated"], 0),
        ("batch size", ["Target", "Decoy"], ["batch-size alpha", "batch size"], 0),
        ("fine-tune on a Mac", ["Target", "Decoy"], ["mlx", "unrelated"], 0),
        (
            "preference training without a reference model",
            ["No Reference Model", "Preference Training", "Reference-free training"],
            ["alignment", "preference preference training", "alignment"],
            0,
        ),
    ],
)
def test_each_query_rule_changes_retrieval(query, sections, bodies, expected):
    chunks = [
        {"doc_name": "guide", "section": title, "heading_prefix": title, "text": body}
        for title, body in zip(sections, bodies)
    ]
    chunks.extend(
        {"doc_name": "guide", "section": f"Other {number}", "heading_prefix": "Other", "text": word}
        for number, word in enumerate(["alpha", "beta", "gamma", "delta"])
    )
    from soup_cli.docs.retriever import build_index

    index = build_index(chunks)
    baseline = index.rank(tokenize(query), top_k=1)
    actual = _rank_chunks(query, index, chunks, top_k=1)
    assert actual and actual[0][0] == expected and actual[0][1] > 0
    assert baseline[0][0] != expected or baseline[0][1] <= 0


def test_heading_boost_ranks_before_top_k_and_is_bounded():
    chunks = [
        {"section": "Other", "heading_prefix": "Guide > Other", "text": "vram"},
        {"section": "VRAM", "heading_prefix": "Guide > VRAM", "text": "other"},
        {"section": "Other", "heading_prefix": "Guide > Other", "text": "unrelated"},
    ]
    chunks.extend(
        {"section": "Other", "heading_prefix": "Guide > Other", "text": word}
        for word in ["alpha", "beta", "gamma", "delta"]
    )
    index = PureBM25([tokenize(c["heading_prefix"] + "\n\n" + c["text"]) for c in chunks])
    before = index.score(["vram"])
    assert index.rank(["vram"], 1)[0][0] == 0
    after = _rank_chunks("vram", index, chunks, 3)
    assert after[0][0] == 1
    assert before[1] < after[0][1] <= 1.25 * before[1]
    repeated_score = _rank_chunks("vram vram", index, chunks, 3)[0][1]
    assert repeated_score - 2 * before[1] <= 0.3 * index.idf["vram"] + 1e-12


def test_query_only_search_preserves_existing_cache(tmp_path, monkeypatch):
    from soup_cli.docs import retriever

    docs = {
        "guide.md": (
            "# Guide\n\n## Apple Silicon\n\nFine-tune with MLX on Macs.\n\n"
            "## Other\n\nalpha beta gamma\n\n## Unrelated\n\ndelta epsilon zeta"
        )
    }
    monkeypatch.setattr(retriever, "load_docs", lambda: docs)
    first = search_docs("fine-tune on a Mac", cache_dir=tmp_path)
    assert first
    cache = tmp_path / "docs_index.json"
    original = (cache.read_bytes(), cache.stat().st_mtime_ns)

    def unexpected_write(*args, **kwargs):
        pytest.fail("query-only rules must reuse the existing cache")

    monkeypatch.setattr(retriever, "save_cache", unexpected_write)
    monkeypatch.setattr(retriever, "build_index", unexpected_write)
    assert search_docs("fine-tune on a Mac", cache_dir=tmp_path) == first
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == original


def _comparison_fixture(monkeypatch, baseline, candidate, targets):
    from types import SimpleNamespace

    from scripts import check_docs_search_ranking as gate

    fixture = {
        "cohort": "gate contract test",
        "chunks": [{"chunk_id": str(number)} for number in range(12)],
        "queries": [
            {"id": str(number), "query": f"query{number}", "targets": target}
            for number, target in enumerate(targets)
        ],
    }

    def ranked(query, runs):
        return [(doc_id, 1.0) for doc_id in runs[int(query.removeprefix("query"))]]

    monkeypatch.setattr(
        gate,
        "build_index",
        lambda chunks: SimpleNamespace(rank=lambda tokens, limit: ranked(tokens[0], baseline)),
    )
    monkeypatch.setattr(gate, "_rank_chunks", lambda query, *args: ranked(query, candidate))
    return gate, fixture


def test_ranking_gate_accepts_aggregate_gains_and_discloses_individual_loss(monkeypatch):
    gate, fixture = _comparison_fixture(
        monkeypatch,
        [[9, 0], [9, 1, 8, 7, 6, 2]],
        [[9, 8, 0], [1, 9, 8, 2]],
        [["0"], ["1", "2"]],
    )
    result = gate.compare(fixture)
    assert result["passed"]
    assert result["metrics"]["candidate"]["recall5"] == 1
    assert result["metrics"]["baseline"]["recall5"] == 0.75
    assert result["rank_losses"] == [{"id": "0", "query": "query0", "old_rank": 2, "new_rank": 3}]


def test_ranking_gate_refuses_recall_loss_despite_first_rank_gain(monkeypatch):
    gate, fixture = _comparison_fixture(
        monkeypatch,
        [[9, 0], [1, 9, 8, 2]],
        [[0, 9], [1, 9, 8, 7, 6, 2]],
        [["0"], ["1", "2"]],
    )
    result = gate.compare(fixture)
    assert result["metrics"]["candidate"]["mrr10"] > result["metrics"]["baseline"]["mrr10"]
    assert not result["passed"]


def test_ranking_gate_scores_misses_as_zero_and_still_reports_full_rank_loss(monkeypatch):
    gate, fixture = _comparison_fixture(
        monkeypatch, [list(range(12))], [list(range(10)) + [11, 10]], [["10"]]
    )
    result = gate.compare(fixture)
    assert result["passed"]
    assert result["metrics"]["candidate"] == dict.fromkeys(["hit1", "hit5", "mrr10", "recall5"], 0)
    assert result["rank_losses"][0]["old_rank"] == 11
    assert result["rank_losses"][0]["new_rank"] == 12
    assert not gate.compare(fixture, require_hit5_gain=True)["passed"]


@pytest.mark.parametrize(
    "invalid", ["missing_target", "duplicate_query", "duplicate_chunk", "empty"]
)
def test_ranking_gate_refuses_invalid_comparisons(monkeypatch, invalid):
    gate, fixture = _comparison_fixture(monkeypatch, [[0]], [[0]], [["0"]])
    if invalid == "missing_target":
        fixture["queries"][0]["targets"] = ["unknown"]
    elif invalid == "duplicate_query":
        fixture["queries"].append(fixture["queries"][0])
    elif invalid == "duplicate_chunk":
        fixture["chunks"].append(fixture["chunks"][0])
    else:
        fixture["queries"] = []
    with pytest.raises(ValueError):
        gate.compare(fixture)


def test_ranking_gate_refuses_changed_baseline(monkeypatch):
    gate, fixture = _comparison_fixture(monkeypatch, [[0]], [[0]], [["0"]])
    fixture["queries"][0]["baseline_top10"] = ["1"]
    with pytest.raises(ValueError, match="baseline ranking changed"):
        gate.compare(fixture)


def test_ranking_gate_refuses_changed_document_or_chunk_snapshot(monkeypatch):
    gate, fixture = _comparison_fixture(monkeypatch, [[0]], [[0]], [["0"]])
    fixture["corpus_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="chunking differs"):
        gate.compare(fixture)
    fixture.pop("corpus_sha256")
    fixture["documents"] = {"guide.md": "incorrect"}
    monkeypatch.setattr(gate, "load_docs", lambda: {"guide.md": "# Guide"})
    with pytest.raises(ValueError, match="documentation snapshot differs"):
        gate.compare(fixture)


def test_ranking_gate_command_exit_codes_and_json(tmp_path):
    fixture = {
        "cohort": "command contract",
        "chunks": [
            {
                "chunk_id": "0",
                "doc_name": "guide",
                "section": "Guide",
                "heading_prefix": "Guide",
                "text": "alpha beta",
            }
        ],
        "queries": [{"id": "q1", "query": "unmatched", "targets": ["0"]}],
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "check_docs_search_ranking.py"
    command = [sys.executable, str(script), str(fixture_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"]
    original = fixture_path.read_bytes()
    result = subprocess.run(
        command + ["--output", str(fixture_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 2
    assert fixture_path.read_bytes() == original
    result = subprocess.run(
        command + ["--require-hit5-gain"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    assert not json.loads(result.stdout)["passed"]
    fixture["queries"][0]["targets"] = ["missing"]
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "Invalid ranking comparison" in " ".join(strip_ansi(result.stdout).split())


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
    text = " ".join(strip_ansi(result.stdout).split())
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
    text = " ".join(strip_ansi(result.stdout).split())
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
    plain = " ".join(strip_ansi(result.output).split())
    assert "Error:" in plain
    assert "No documentation files found" in plain


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
