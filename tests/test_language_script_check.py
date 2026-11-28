"""#2785: a Cyrillic answer the text's letters rule out is ``unknown``, on both paths.

Kazakh rows came back ``ru`` from the stopword heuristic (one shared word in the
user turn) and ``ru`` / ``uk`` from langdetect, which has no Kazakh profile.
Letters outside the alphabet of the answer now turn it into ``unknown`` once they
are in at least 15 % of the row's Cyrillic words (pinned at 14 % and 15 % below);
plain Russian, Ukrainian under ``uk`` and every non-Cyrillic answer are unchanged.
"""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score
from soup_cli.utils.data_score import compute_scorecard, detect_language

KAZAKH_ROWS = [
    ("Қазақстанның астанасы қай қала?", "Қазақстан Республикасының астанасы — Астана қаласы."),
    (
        "Бұл заң не үшін қажет?",
        "Бұл заң азаматтардың құқықтары мен бостандықтарын қорғауға бағытталған.",
    ),
    (
        "Еңбек демалысы туралы айтып беріңізші.",
        "Жұмыс беруші қызметкерге жыл сайынғы ақылы еңбек демалысын беруге міндетті.",
    ),
    ("Сіз не істеп жатырсыз?", "Мен үйде кітап оқып отырмын, кейін досыммен кездесемін."),
    ("Ауа райы қандай болады?", "Ертең Астанада күн бұлтты болады, кешке қар жаууы мүмкін."),
]
# Whole rows, user and assistant turns joined, as both commands read them.
KAZAKH_TEXTS = [f"{q}\n{a}" for q, a in KAZAKH_ROWS]
KK_IDS = [f"kk{i}" for i in range(1, len(KAZAKH_TEXTS) + 1)]

RUSSIAN = "Республика Казахстан — государство в Центральной Азии, столица город Астана."
# One Kazakh place name must not change the answer for a Russian sentence.
RUSSIAN_WITH_KAZAKH_NAME = (
    "Вчера мы ездили в Қарағанды, чтобы встретиться с партнёрами по проекту."
)
# і and ї set this apart from Russian.
UKRAINIAN = "Цей закон спрямований на захист прав і свобод громадян країни."
ENGLISH = "the quick brown fox jumps over the lazy dog"


def _heuristic_only():
    return patch.object(data_score, "_langdetect_fast", return_value=None)


def _langdetect_says(code):
    return patch.object(data_score, "_langdetect_fast", return_value=code)


# Heuristic path. Rows 2 and 4 return ``ru`` on main (``не`` in the user turn);
# rows 1, 3 and 5 already return ``unknown`` and must stay so.
@pytest.mark.parametrize("text", KAZAKH_TEXTS, ids=KK_IDS)
def test_heuristic_does_not_tag_kazakh_as_russian(text):
    with _heuristic_only():
        assert detect_language(text) == "unknown"


# langdetect path: every Cyrillic label it has is ruled out by Kazakh letters.
@pytest.mark.parametrize("text", KAZAKH_TEXTS, ids=KK_IDS)
@pytest.mark.parametrize("wrong", ["ru", "uk", "bg", "mk"])
def test_langdetect_cyrillic_label_on_kazakh_is_unknown(text, wrong):
    with _langdetect_says(wrong):
        assert detect_language(text) == "unknown"


def test_uppercase_kazakh_is_checked_too():
    with _langdetect_says("ru"):
        assert detect_language(KAZAKH_TEXTS[0].upper()) == "unknown"


def test_ordinary_russian_still_returns_ru():
    with _heuristic_only():
        assert detect_language(RUSSIAN) == "ru"
    with _langdetect_says("ru"):
        assert detect_language(RUSSIAN) == "ru"


def test_a_kazakh_place_name_does_not_change_a_russian_answer():
    with _heuristic_only():
        assert detect_language(RUSSIAN_WITH_KAZAKH_NAME) == "ru"
    with _langdetect_says("ru"):
        assert detect_language(RUSSIAN_WITH_KAZAKH_NAME) == "ru"


def _row_with_share(marked, total):
    """``total`` Cyrillic words, ``marked`` of them with a Kazakh letter, and ``и``
    twice so the stopword heuristic answers ``ru``."""
    words = ["қала"] * marked + ["и"] * 2 + ["город"] * (total - marked - 2)
    return " ".join(words)


@pytest.mark.parametrize(
    "marked, total, expected",
    [(7, 50, "ru"), (3, 20, "unknown")],
    ids=["14pct-keeps-ru", "15pct-is-unknown"],
)
def test_share_boundary_on_both_paths(marked, total, expected):
    text = _row_with_share(marked, total)
    _, share = data_score._extra_cyrillic_letters(data_score._tokenise(text))
    assert share == marked / total
    with _heuristic_only():
        assert detect_language(text) == expected
    with _langdetect_says("ru"):
        assert detect_language(text) == expected


def test_heuristic_does_not_tag_ukrainian_as_russian():
    with _heuristic_only():
        assert detect_language(UKRAINIAN) == "unknown"


@pytest.mark.parametrize("ruled_out", ["ru", "bg"])
def test_langdetect_ru_or_bg_on_ukrainian_is_unknown(ruled_out):
    with _langdetect_says(ruled_out):
        assert detect_language(UKRAINIAN) == "unknown"


def test_langdetect_uk_on_ukrainian_is_kept():
    with _langdetect_says("uk"):
        assert detect_language(UKRAINIAN) == "uk"


def test_english_still_returns_en_and_non_cyrillic_labels_are_untouched():
    with _heuristic_only():
        assert detect_language(ENGLISH) == "en"
    with _langdetect_says("ja"):
        assert detect_language(ENGLISH) == "ja"


def test_real_langdetect_on_kazakh_is_unknown():
    pytest.importorskip("langdetect")
    for text in KAZAKH_TEXTS:
        assert detect_language(text) == "unknown"


def _kazakh_rows_as_chat():
    return [
        {"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]}
        for question, answer in KAZAKH_ROWS
    ]


@pytest.mark.parametrize("langdetect_answer", [None, "ru"])
def test_cli_langdetect_tags_kazakh_rows_unknown(tmp_path, monkeypatch, langdetect_answer):
    monkeypatch.chdir(tmp_path)
    with open(tmp_path / "kk.jsonl", "w", encoding="utf-8") as handle:
        for row in _kazakh_rows_as_chat():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with _langdetect_says(langdetect_answer):
        result = CliRunner().invoke(
            app, ["data", "langdetect", "--input", "kk.jsonl", "--output", "tagged.jsonl"],
        )
    assert result.exit_code == 0, result.output
    lines = (tmp_path / "tagged.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["_language"] for line in lines] == ["unknown"] * len(KAZAKH_ROWS)


def test_scorecard_does_not_count_a_kazakh_dataset_as_russian():
    with _langdetect_says("ru"):
        report = compute_scorecard(_kazakh_rows_as_chat())
    assert dict(report.languages) == {"unknown": len(KAZAKH_ROWS)}
