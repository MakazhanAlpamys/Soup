"""Float32 cosine comparisons tolerate rounding without rewriting measured pairs."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.commands import data
from soup_cli.utils.embed import _l2_normalize
from soup_cli.utils.semdedup import greedy_semdedup
from tests.conftest import strip_ansi


def duplicate_vector():
    vector = _l2_normalize(np.array([[1, 1, 1]], dtype=np.float32))[0]
    assert float(vector @ vector) < 1.0
    return vector


@pytest.mark.parametrize("threshold", [1.0, 0.9999])
def test_duplicate_rounding_preserves_raw_pairs_and_order(threshold):
    vector = duplicate_vector()
    other = np.array([1, 0, 0], dtype=np.float32)
    report = greedy_semdedup(np.stack([vector, vector, other]), threshold=threshold)
    assert report.kept == (0, 2)
    assert report.dropped == (1,)
    assert report.pairs == ((1, 0, float(vector @ vector)),)
    assert report.threshold == threshold


@pytest.mark.parametrize("gap,dropped", [(0.5e-6, (1,)), (2e-6, ())])
def test_tolerance_boundary_at_general_threshold(gap, dropped):
    cosine = 0.8 - gap
    vectors = np.array([[1, 0], [cosine, np.sqrt(1 - cosine**2)]], dtype=np.float32)
    report = greedy_semdedup(vectors, threshold=0.8)
    assert report.dropped == dropped


def test_zero_vectors_remain_distinct_at_one():
    report = greedy_semdedup(np.zeros((2, 3), dtype=np.float32), threshold=1.0)
    assert report.kept == (0, 1) and report.pairs == ()


@pytest.mark.parametrize("threshold", [1e-6, 1e-7])
@pytest.mark.parametrize("vectors", [
    [[1.0, 0.0], [0.0, 0.0]],  # zero after a kept nonzero row
    [[0.0, 0.0], [0.0, 0.0]],  # zero after a kept zero row
    [[1.0, 0.0], [0.0, 1.0]],  # orthogonal row
])
def test_positive_low_threshold_keeps_exact_zero_similarity(threshold, vectors):
    report = greedy_semdedup(np.asarray(vectors, dtype=np.float32), threshold=threshold)
    assert report.kept == (0, 1)
    assert report.dropped == ()
    assert report.pairs == ()


def test_zero_threshold_keeps_inclusive_zero_similarity_match():
    report = greedy_semdedup(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), threshold=0.0
    )
    assert report.kept == (0,)
    assert report.dropped == (1,)
    assert report.pairs == ((1, 0, 0.0),)


@pytest.mark.parametrize("threshold", ["1.0", "0.9999"])
@pytest.mark.parametrize("duplicate", ["same", ""])
def test_cli_removes_identical_rows_without_model_download(
    tmp_path, monkeypatch, threshold, duplicate,
):
    monkeypatch.chdir(tmp_path)
    vector = duplicate_vector()
    def embed(texts, **kwargs):
        return np.stack([vector if text == duplicate else [1, 0, 0] for text in texts])
    monkeypatch.setattr(data, "embed_texts", embed)
    rows = [{"text": duplicate}, {"text": duplicate}, {"text": "other"}]
    (tmp_path / "in.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    result = CliRunner().invoke(data.app, [
        "dedup", "in.jsonl", "--semantic", "--threshold", threshold, "-o", "out.jsonl",
    ])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, output
    assert "3 -> 2 rows" in output and "-1 near-duplicates" in output
    kept = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert kept == [rows[0], rows[2]]
