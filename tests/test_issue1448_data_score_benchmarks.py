"""#1448 data-score slice: comparison evidence reaches the real consumer."""

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score
from tests.conftest import strip_ansi

CORPUS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
PARTIAL = "alpha bravo charlie delta echo foxtrot golf hotel other fresh tokens today"


@pytest.fixture
def score_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"text": CORPUS}) + "\n", encoding="utf-8")
    return path


def _file(tmp_path, rows):
    path = tmp_path / "bench.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _score(path, *args):
    return CliRunner().invoke(app, ["data", "score", "-i", str(path), *args])


def _plain(result):
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


@pytest.mark.parametrize("labels", [[], ["-b", "gsm8k,mmlu"]], ids=["file", "labels-and-file"])
def test_file_texts_and_threshold_reach_real_decontamination(
    score_input, tmp_path, monkeypatch, labels
):
    seen = []
    original = data_score.decontaminate_rows

    def spy(rows, texts, **kwargs):
        seen.append((texts, kwargs))
        return original(rows, texts, **kwargs)

    monkeypatch.setattr(data_score, "decontaminate_rows", spy)
    bench = _file(tmp_path, [{"text": CORPUS}])
    result = _score(score_input, *labels, "--benchmark-file", str(bench), "--threshold", "0.73")
    assert result.exit_code == 0, result.output
    assert seen == [([CORPUS], {"n": 8, "threshold": 0.73})]
    assert re.search(r"Decontaminated\s+1", _plain(result))


@pytest.mark.parametrize("threshold,removed", [("0.1", 1), ("0.9", 0)])
def test_threshold_changes_real_removed_count(score_input, tmp_path, threshold, removed):
    score_input.write_text(json.dumps({"text": PARTIAL}) + "\n", encoding="utf-8")
    bench = _file(tmp_path, [{"text": CORPUS}])
    result = _score(score_input, "--benchmark-file", str(bench), "--threshold", threshold)
    assert result.exit_code == 0, result.output
    assert re.search(rf"Decontaminated\s+{removed}", _plain(result))


def test_labels_alone_are_refused_instead_of_a_clean_zero(score_input):
    result = _score(score_input, "-b", "gsm8k,mmlu", "--threshold", "0.01")
    assert result.exit_code != 0
    output = _plain(result)
    assert "--benchmarks" in output and "--benchmark-file" in output
    assert "not bundled" in output
    assert "Data Scorecard" not in output


def test_plain_scoring_distinguishes_not_run_from_measured_zero(score_input):
    result = _score(score_input)
    assert result.exit_code == 0, result.output
    assert "Decontaminated" in _plain(result)
    assert "not run" in _plain(result)


@pytest.mark.parametrize(
    "row", [{"content": CORPUS}, {"messages": [{"role": "user", "content": CORPUS}]}]
)
def test_file_uses_the_shared_text_extractor(score_input, tmp_path, row):
    bench = _file(tmp_path, [row])
    result = _score(score_input, "--benchmark-file", str(bench))
    assert result.exit_code == 0, result.output
    assert re.search(r"Decontaminated\s+1", _plain(result))


@pytest.mark.parametrize("rows", [[], [{"text": ""}], [{"number": 42}], [{"text": "too short"}]])
def test_unusable_corpus_is_refused(score_input, tmp_path, rows):
    bench = _file(tmp_path, rows)
    result = _score(score_input, "--benchmark-file", str(bench))
    assert result.exit_code != 0
    assert "--benchmark-file" in _plain(result)
    assert "Data Scorecard" not in _plain(result)


def test_bad_corpus_path_is_refused(score_input, tmp_path):
    result = _score(score_input, "--benchmark-file", str(tmp_path.parent / "outside.jsonl"))
    assert result.exit_code != 0
    assert "--benchmark-file" in _plain(result)
    assert "Data Scorecard" not in _plain(result)


def test_unknown_label_still_fails_with_a_corpus(score_input, tmp_path):
    bench = _file(tmp_path, [{"text": CORPUS}])
    result = _score(score_input, "-b", "unknown", "--benchmark-file", str(bench))
    assert result.exit_code != 0
    assert "unknown benchmark" in _plain(result)


def test_help_explains_labels_do_not_select_comparison_texts():
    result = CliRunner().invoke(app, ["data", "score", "--help"])
    assert result.exit_code == 0, result.output
    output = _plain(result)
    assert "validated only" in output
    assert "do not select or filter" in output


@pytest.mark.parametrize("label", ["gsm8k", "mmlu"])
def test_labels_with_file_report_validation_only_without_selecting_texts(
    score_input, tmp_path, label
):
    bench = _file(tmp_path, [{"text": CORPUS}])
    result = _score(score_input, "-b", label, "--benchmark-file", str(bench))
    assert result.exit_code == 0, result.output
    output = _plain(result)
    assert f"--benchmarks labels ({label}) are validated only" in output
    assert "do not select or filter" in output
    assert re.search(r"Decontaminated\s+1", output)


def test_file_without_labels_does_not_print_a_label_note(score_input, tmp_path):
    bench = _file(tmp_path, [{"text": CORPUS}])
    result = _score(score_input, "--benchmark-file", str(bench))
    assert result.exit_code == 0, result.output
    assert "validated only" not in _plain(result)
    assert re.search(r"Decontaminated\s+1", _plain(result))


@pytest.mark.parametrize("bad_row", ["not-json", "[]", '{"number": 42}'])
def test_partially_unusable_corpus_cannot_report_a_clean_result(score_input, tmp_path, bad_row):
    bench = _file(tmp_path, [{"text": CORPUS}])
    bench.write_text(bench.read_text(encoding="utf-8") + bad_row + "\n", encoding="utf-8")
    result = _score(score_input, "--benchmark-file", str(bench))
    assert result.exit_code != 0
    assert "--benchmark-file" in _plain(result)
    assert "Data Scorecard" not in _plain(result)


def test_strict_loader_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _file(tmp_path, [{"text": CORPUS}])
    path.write_text(path.read_text(encoding="utf-8") + "bad-json\n", encoding="utf-8")
    assert data_score.load_jsonl_rows(str(path)) == [{"text": CORPUS}]
    with pytest.raises(ValueError, match="line 2"):
        data_score.load_jsonl_rows(str(path), strict=True)


def test_row_cap_cannot_silently_truncate_the_benchmark_evidence(
    score_input, tmp_path, monkeypatch
):
    monkeypatch.setattr(data_score, "_MAX_ROWS", 1)
    bench = _file(tmp_path, [{"text": CORPUS}, {"text": CORPUS}])
    result = _score(score_input, "--benchmark-file", str(bench))
    assert result.exit_code != 0
    assert "--benchmark-file" in _plain(result)
    assert "exceeds 1 rows" in _plain(result)
    assert "Data Scorecard" not in _plain(result)


def test_explicit_texts_do_not_need_a_false_builtin_label():
    report = data_score.compute_scorecard([{"text": CORPUS}], benchmark_texts=[CORPUS])
    assert report.decontaminated_removed == 1


def test_legacy_named_corpora_remain_supported():
    report = data_score.compute_scorecard(
        [{"text": CORPUS}], benchmarks=["mmlu"], decontaminate_texts={"mmlu": [CORPUS]}
    )
    assert report.decontaminated_removed == 1


@pytest.mark.parametrize("bad", [CORPUS, True, [42]])
def test_core_rejects_invalid_explicit_text_shapes(bad):
    with pytest.raises(TypeError):
        data_score.compute_scorecard([{"text": CORPUS}], benchmark_texts=bad)


@pytest.mark.parametrize("texts", [[], [""], ["too short"], ["!!!"], [CORPUS, "short"]])
def test_core_cannot_present_unusable_explicit_corpus_as_a_measured_zero(texts):
    with pytest.raises(ValueError, match="usable 8-gram"):
        data_score.compute_scorecard([{"text": CORPUS}], benchmark_texts=texts)


def test_core_none_preserves_legacy_no_comparison():
    report = data_score.compute_scorecard([{"text": CORPUS}], benchmark_texts=None)
    plain = data_score.compute_scorecard([{"text": CORPUS}])
    assert report.decontaminated_removed == plain.decontaminated_removed == 0
    assert report.total == plain.total == 1
