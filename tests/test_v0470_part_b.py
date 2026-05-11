"""v0.47.0 Part B — Data Quality Moat tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import data_score  # noqa: F401

    for name in (
        "ScoreReport",
        "BENCHMARKS",
        "ngram_set",
        "ngram_overlap_ratio",
        "decontaminate_rows",
        "detect_pii",
        "detect_language",
        "score_toxicity",
        "score_educational_value",
        "compute_scorecard",
        "load_jsonl_rows",
    ):
        assert hasattr(data_score, name), f"missing public symbol {name}"


def test_benchmarks_immutable():
    from types import MappingProxyType

    from soup_cli.utils.data_score import BENCHMARKS

    assert isinstance(BENCHMARKS, MappingProxyType)
    assert "mmlu" in BENCHMARKS
    assert "gsm8k" in BENCHMARKS
    assert "humaneval" in BENCHMARKS


def test_score_report_frozen():
    import dataclasses

    from soup_cli.utils.data_score import ScoreReport

    rep = ScoreReport(
        total=10,
        pii_flagged=1,
        toxic_flagged=0,
        decontaminated_removed=2,
        languages={"en": 8, "fr": 2},
        educational_mean=0.5,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.total = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ngram_set + ngram_overlap_ratio
# ---------------------------------------------------------------------------


def test_ngram_set_basic():
    from soup_cli.utils.data_score import ngram_set

    s = ngram_set("hello world how are you", n=3)
    assert ("hello", "world", "how") in s


def test_ngram_set_short_returns_empty():
    from soup_cli.utils.data_score import ngram_set

    assert ngram_set("hi", n=5) == set()


def test_ngram_set_rejects_invalid_n():
    from soup_cli.utils.data_score import ngram_set

    with pytest.raises(ValueError):
        ngram_set("hi", n=0)
    with pytest.raises(ValueError):
        ngram_set("hi", n=-1)
    with pytest.raises(TypeError):
        ngram_set("hi", n=True)


def test_ngram_set_rejects_non_string():
    from soup_cli.utils.data_score import ngram_set

    with pytest.raises(TypeError):
        ngram_set(None, n=3)  # type: ignore[arg-type]


def test_ngram_set_oversize_rejected():
    from soup_cli.utils.data_score import _MAX_TEXT_CHARS, ngram_set

    with pytest.raises(ValueError):
        ngram_set("x " * (_MAX_TEXT_CHARS), n=3)


def test_ngram_overlap_ratio_identical_is_one():
    from soup_cli.utils.data_score import ngram_overlap_ratio

    assert ngram_overlap_ratio("a b c d e", "a b c d e", n=3) == 1.0


def test_ngram_overlap_ratio_disjoint_is_zero():
    from soup_cli.utils.data_score import ngram_overlap_ratio

    assert ngram_overlap_ratio("a b c d e", "x y z w v", n=3) == 0.0


def test_ngram_overlap_ratio_partial():
    from soup_cli.utils.data_score import ngram_overlap_ratio

    r = ngram_overlap_ratio("a b c d e f", "a b c x y z", n=3)
    assert 0.0 < r < 1.0


def test_ngram_overlap_ratio_empty_returns_zero():
    from soup_cli.utils.data_score import ngram_overlap_ratio

    assert ngram_overlap_ratio("", "a b c", n=3) == 0.0
    assert ngram_overlap_ratio("a b c", "", n=3) == 0.0


# ---------------------------------------------------------------------------
# decontaminate_rows
# ---------------------------------------------------------------------------


def test_decontaminate_rows_removes_overlap():
    from soup_cli.utils.data_score import decontaminate_rows

    rows = [
        {"text": "The capital of France is Paris and known for art."},
        {"text": "totally unrelated content about banana farming systems"},
    ]
    benchmark_texts = ["The capital of France is Paris and known for art."]
    kept, removed_idx = decontaminate_rows(rows, benchmark_texts, n=5, threshold=0.5)
    assert len(kept) == 1
    assert 0 in removed_idx


def test_decontaminate_rows_threshold_bounds():
    from soup_cli.utils.data_score import decontaminate_rows

    with pytest.raises(ValueError):
        decontaminate_rows([], [], n=3, threshold=1.5)
    with pytest.raises(ValueError):
        decontaminate_rows([], [], n=3, threshold=-0.1)
    with pytest.raises(TypeError):
        decontaminate_rows([], [], n=3, threshold=True)


def test_decontaminate_rows_empty_benchmark_keeps_all():
    from soup_cli.utils.data_score import decontaminate_rows

    rows = [{"text": "anything"}, {"text": "more"}]
    kept, removed = decontaminate_rows(rows, [], n=3, threshold=0.5)
    assert kept == rows
    assert removed == []


def test_decontaminate_rows_non_dict_row_skipped():
    from soup_cli.utils.data_score import decontaminate_rows

    rows = [{"text": "a b c"}, "not-a-dict", {"text": "x y z"}]  # type: ignore[list-item]
    kept, _ = decontaminate_rows(rows, ["x y z"], n=2, threshold=0.5)
    # The non-dict was dropped from the kept output
    assert "not-a-dict" not in kept
    assert all(isinstance(r, dict) for r in kept)


# ---------------------------------------------------------------------------
# detect_pii
# ---------------------------------------------------------------------------


def test_detect_pii_email():
    from soup_cli.utils.data_score import detect_pii

    hits = detect_pii("Contact me at user@example.com please")
    assert any(h["kind"] == "email" for h in hits)


def test_detect_pii_phone():
    from soup_cli.utils.data_score import detect_pii

    hits = detect_pii("Call me at +1-415-555-0123 if you can")
    assert any(h["kind"] == "phone" for h in hits)


def test_detect_pii_ssn_like():
    from soup_cli.utils.data_score import detect_pii

    hits = detect_pii("SSN 123-45-6789 is sensitive")
    assert any(h["kind"] == "ssn" for h in hits)


def test_detect_pii_credit_card_like():
    from soup_cli.utils.data_score import detect_pii

    # 16-digit-like sequence — basic regex match
    hits = detect_pii("Card: 4111 1111 1111 1111 expires soon")
    assert any(h["kind"] == "credit_card" for h in hits)


def test_detect_pii_no_match():
    from soup_cli.utils.data_score import detect_pii

    assert detect_pii("hello world how are you today friend") == []


def test_detect_pii_rejects_non_string():
    from soup_cli.utils.data_score import detect_pii

    with pytest.raises(TypeError):
        detect_pii(123)  # type: ignore[arg-type]


def test_detect_pii_caps_text_size():
    from soup_cli.utils.data_score import _MAX_TEXT_CHARS, detect_pii

    with pytest.raises(ValueError):
        detect_pii("x " * (_MAX_TEXT_CHARS))


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


def test_detect_language_english_heuristic():
    from soup_cli.utils.data_score import detect_language

    lang = detect_language("the quick brown fox jumps over the lazy dog")
    assert lang == "en"


def test_detect_language_empty_returns_unknown():
    from soup_cli.utils.data_score import detect_language

    assert detect_language("") == "unknown"
    assert detect_language("   ") == "unknown"


def test_detect_language_non_string_rejected():
    from soup_cli.utils.data_score import detect_language

    with pytest.raises(TypeError):
        detect_language(123)  # type: ignore[arg-type]


def test_detect_language_returns_string():
    from soup_cli.utils.data_score import detect_language

    result = detect_language("Hola mundo, esto es una prueba en español muy clara.")
    # Heuristic might return "es" or "unknown" — just check it's a string label
    assert isinstance(result, str)
    assert "\x00" not in result


# ---------------------------------------------------------------------------
# score_toxicity
# ---------------------------------------------------------------------------


def test_score_toxicity_bounds():
    from soup_cli.utils.data_score import score_toxicity

    s = score_toxicity("a perfectly normal sentence")
    assert 0.0 <= s <= 1.0


def test_score_toxicity_keyword_baseline():
    from soup_cli.utils.data_score import score_toxicity

    clean = score_toxicity("good morning sunshine flowers garden")
    bad = score_toxicity("i hate kill destroy violence attack")
    assert bad > clean


def test_score_toxicity_empty_zero():
    from soup_cli.utils.data_score import score_toxicity

    assert score_toxicity("") == 0.0


def test_score_toxicity_rejects_non_string():
    from soup_cli.utils.data_score import score_toxicity

    with pytest.raises(TypeError):
        score_toxicity(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# score_educational_value
# ---------------------------------------------------------------------------


def test_score_educational_value_bounds():
    from soup_cli.utils.data_score import score_educational_value

    s = score_educational_value("Photosynthesis converts sunlight into chemical energy.")
    assert 0.0 <= s <= 1.0


def test_score_educational_value_empty_zero():
    from soup_cli.utils.data_score import score_educational_value

    assert score_educational_value("") == 0.0


def test_score_educational_value_short_low():
    from soup_cli.utils.data_score import score_educational_value

    short = score_educational_value("yo")
    longer = score_educational_value(
        "The mitochondria is the powerhouse of the cell. "
        "It generates ATP through oxidative phosphorylation."
    )
    assert longer > short


def test_score_educational_value_rejects_non_string():
    from soup_cli.utils.data_score import score_educational_value

    with pytest.raises(TypeError):
        score_educational_value(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_scorecard
# ---------------------------------------------------------------------------


def test_compute_scorecard_full():
    from soup_cli.utils.data_score import compute_scorecard

    rows = [
        {"text": "the quick brown fox jumps over"},
        {"text": "contact me at hi@example.com"},
        {"text": "i hate violence attack destroy"},
        {"text": ""},
    ]
    rep = compute_scorecard(rows)
    assert rep.total == 4
    assert rep.pii_flagged >= 1
    assert rep.toxic_flagged >= 1
    from collections.abc import Mapping as MappingABC
    assert isinstance(rep.languages, MappingABC)


def test_compute_scorecard_handles_non_dict_rows():
    from soup_cli.utils.data_score import compute_scorecard

    rep = compute_scorecard([{"text": "hello"}, "garbage", 42])  # type: ignore[list-item]
    # Non-dict rows excluded from total
    assert rep.total == 1


def test_compute_scorecard_with_benchmarks():
    from soup_cli.utils.data_score import compute_scorecard

    rows = [
        {"text": "The capital of France is Paris and known for art."},
        {"text": "totally unrelated banana farming systems"},
    ]
    rep = compute_scorecard(
        rows,
        benchmarks=["mmlu"],
        decontaminate_texts={"mmlu": ["The capital of France is Paris and known for art."]},
        decontaminate_threshold=0.5,
    )
    assert rep.decontaminated_removed >= 1


# ---------------------------------------------------------------------------
# load_jsonl_rows
# ---------------------------------------------------------------------------


def test_compute_scorecard_rejects_nan_threshold():
    from soup_cli.utils.data_score import compute_scorecard

    with pytest.raises(ValueError, match="finite"):
        compute_scorecard([], decontaminate_threshold=float("nan"))


def test_require_str_rejects_null_byte():
    from soup_cli.utils.data_score import detect_pii

    with pytest.raises(ValueError, match="null byte"):
        detect_pii("hi\x00there")


def test_pii_redos_phone_pattern_is_bounded():
    """Confirm pathological near-miss inputs return promptly (ReDoS guard)."""
    from soup_cli.utils.data_score import detect_pii

    # Pre-cap means even a 100k pathological input is bounded to 50k.
    pathological = "1 " * 50_000 + "x"
    # Should return quickly (≤ a few hundred ms in practice). We just
    # assert that it doesn't hang and that it returns a list.
    hits = detect_pii(pathological)
    assert isinstance(hits, list)


def test_load_jsonl_rows_basic(tmp_path):
    from soup_cli.utils.data_score import load_jsonl_rows

    p = tmp_path / "rows.jsonl"
    p.write_text(
        '{"text":"hi"}\n{"text":"bye"}\n',
        encoding="utf-8",
    )
    os.chdir(tmp_path)
    rows = load_jsonl_rows(str(p))
    assert len(rows) == 2


def test_load_jsonl_rows_outside_cwd(tmp_path):
    from soup_cli.utils.data_score import load_jsonl_rows

    inside = tmp_path / "inside"
    inside.mkdir()
    p = tmp_path / "rows.jsonl"
    p.write_text('{"text":"hi"}', encoding="utf-8")
    os.chdir(inside)
    with pytest.raises(ValueError, match="cwd"):
        load_jsonl_rows(str(p))


def test_load_jsonl_rows_missing(tmp_path):
    from soup_cli.utils.data_score import load_jsonl_rows

    os.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_jsonl_rows("missing.jsonl")


def test_load_jsonl_rows_skips_malformed(tmp_path):
    from soup_cli.utils.data_score import load_jsonl_rows

    p = tmp_path / "rows.jsonl"
    p.write_text(
        '{"text":"ok"}\n!!!not json!!!\n{"text":"also-ok"}\n',
        encoding="utf-8",
    )
    os.chdir(tmp_path)
    rows = load_jsonl_rows(str(p))
    assert len(rows) == 2


def test_load_jsonl_rows_symlink_rejected(tmp_path):
    from soup_cli.utils.data_score import load_jsonl_rows

    if os.name == "nt":
        pytest.skip("symlink test POSIX-only")
    os.chdir(tmp_path)
    real = tmp_path / "real.jsonl"
    real.write_text('{"x":1}', encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        load_jsonl_rows(str(link))


def test_load_jsonl_rows_cap(tmp_path):
    from soup_cli.utils.data_score import _MAX_ROWS

    # Just confirm cap is defined
    assert _MAX_ROWS >= 1


# ---------------------------------------------------------------------------
# CLI: soup data score, decontaminate, toxicity, langdetect, pii, educational
# ---------------------------------------------------------------------------


def _make_app():
    from soup_cli.cli import app
    return app


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_data_score_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "score", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_score_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(
        p,
        [
            {"text": "the quick brown fox jumps"},
            {"text": "contact me at user@example.com"},
        ],
    )
    result = runner.invoke(
        _make_app(),
        ["data", "score", "--input", str(p)],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Scorecard" in result.output or "scorecard" in result.output.lower()


def test_data_score_cli_outside_cwd(tmp_path, monkeypatch):
    runner = CliRunner()
    inside = tmp_path / "inside"
    inside.mkdir()
    p = tmp_path / "rows.jsonl"
    _write_jsonl(p, [{"text": "x"}])
    monkeypatch.chdir(inside)
    result = runner.invoke(_make_app(), ["data", "score", "--input", str(p)])
    assert result.exit_code != 0


def test_data_decontaminate_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "decontaminate", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_decontaminate_cli_unknown_benchmark(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(p, [{"text": "hi"}])
    result = runner.invoke(
        _make_app(),
        [
            "data", "decontaminate",
            "--input", str(p),
            "--benchmarks", "bogus_benchmark",
        ],
    )
    assert result.exit_code != 0


def test_data_decontaminate_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(
        p,
        [
            {"text": "alpha beta gamma delta epsilon zeta"},
            {"text": "totally unrelated banana farming systems"},
        ],
    )
    result = runner.invoke(
        _make_app(),
        [
            "data", "decontaminate",
            "--input", str(p),
            "--benchmarks", "mmlu",
            "--output", "clean.jsonl",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_toxicity_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "toxicity", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_toxicity_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(p, [{"text": "i hate attack destroy"}, {"text": "lovely day"}])
    result = runner.invoke(
        _make_app(),
        ["data", "toxicity", "--input", str(p), "--output", "tox.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path("tox.jsonl").is_file()


def test_data_langdetect_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "langdetect", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_langdetect_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(p, [{"text": "the quick brown fox"}])
    result = runner.invoke(
        _make_app(),
        ["data", "langdetect", "--input", str(p), "--output", "lang.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_pii_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "pii", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_pii_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(p, [{"text": "email user@example.com"}, {"text": "clean"}])
    result = runner.invoke(
        _make_app(),
        ["data", "pii", "--input", str(p), "--output", "pii.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_educational_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "educational", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_data_educational_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rows.jsonl"
    _write_jsonl(p, [{"text": "Photosynthesis converts sunlight into ATP."}])
    result = runner.invoke(
        _make_app(),
        ["data", "educational", "--input", str(p), "--output", "edu.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
