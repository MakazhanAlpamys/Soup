from __future__ import annotations

import pytest

from soup_cli.utils._eval_text import tokenize
from soup_cli.utils.diagnose.forgetting import score_forgetting
from soup_cli.utils.diagnose.memorization import score_memorization
from soup_cli.utils.diagnose.mode_collapse import score_mode_collapse
from soup_cli.utils.live_eval import token_f1

LAO = "ຂ້ອຍມັກກິນເຂົ້າໜຽວກັບປາແດກທຸກມື້"
KHMER = "ខ្ញុំចូលចិត្តញ៉ាំបាយជាមួយត្រីរាល់ថ្ងៃ"
MYANMAR = "ကျွန်တော်ထမင်းနဲ့ငါးကိုနေ့တိုင်းစားတယ်"
HALF_KATAKANA = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿ"
CJK_EXT_C = "".join(chr(0x2A700 + i) for i in range(8))
ENGLISH = "I really like to eat sticky rice with fermented fish every single day"

FINALS = {
    "english": (ENGLISH, ("today", "tomorrow", "always", "sometimes")),
    "lao": (LAO, ("ມື້ນີ້", "ມື້ອື່ນ", "ສະເໝີ", "ບາງຄັ້ງ")),
    "khmer": (KHMER, ("ថ្ងៃនេះ", "ស្អែក", "ជានិច្ច", "ពេលខ្លះ")),
    "myanmar": (MYANMAR, ("ဒီနေ့", "မနက်ဖြန်", "အမြဲ", "တခါတလေ")),
}


def _collapse(answers: list) -> tuple:
    result = score_mode_collapse(["p"], lambda prompt, k: answers, k=len(answers))
    return result.verdict, result.score


@pytest.mark.parametrize("text", [LAO, KHMER, MYANMAR, HALF_KATAKANA, CJK_EXT_C])
def test_unspaced_scripts_tokenize_into_more_than_one_token(text: str) -> None:
    assert len(tokenize(text)) > 1


@pytest.mark.parametrize("script", ["lao", "khmer", "myanmar"])
def test_unspaced_script_reads_like_english_in_all_three_probes(script: str) -> None:
    sentence, finals = FINALS[script]
    assert _collapse([f"{sentence} {word}" for word in finals])[0] == _collapse(
        [f"{ENGLISH} {word}" for word in FINALS["english"][1]]
    )[0] == "MAJOR"
    parts = [sentence[:10], sentence[5:15], sentence[:8], sentence[8:20], sentence[3:17]]
    row = " ".join(parts)
    echo = score_memorization(
        [{"text": row}],
        lambda prefix: " ".join(parts[1:]) + " extra",
        prefix_fraction=0.25,
    )
    assert echo.verdict == "MAJOR", echo
    target = sentence + " " + finals[0]
    base = token_f1(target + " " + finals[1], target)
    adapter = token_f1("...", target)
    assert score_forgetting({"heldout": base}, {"heldout": adapter}).verdict == "MAJOR"


SYMBOL_ONLY = {
    "punctuation": ["...", "?!", "***", "---"],
    "emoji": ["\U0001F44D", "❤️", "\U0001F600", "\U0001F680"],
    "math": ["√ ∞", "≈ ≠", "→ ←", "= + ="],
}


@pytest.mark.parametrize("kind", sorted(SYMBOL_ONLY))
def test_symbol_only_answers_read_in_both_directions(kind: str) -> None:
    answers = SYMBOL_ONLY[kind]
    assert _collapse(answers)[0] == "OK"
    if kind != "punctuation":
        assert _collapse([answers[0]] * 4)[0] == "MAJOR"


@pytest.mark.parametrize(
    "answers",
    [["No.", "No!", "No?", "No..."], ["OK.", "OK!", "OK?", "OK..."], ["No"] * 4],
)
def test_stopword_only_answers_differing_in_punctuation_stay_collapsed(answers: list) -> None:
    assert _collapse(answers)[0] == "MAJOR"


@pytest.mark.parametrize("answers", [["", "", "", ""], [" ", "\n", "  ", "\t"]])
def test_answers_with_no_evidence_return_a_valid_score(answers: list) -> None:
    verdict, score = _collapse(answers)
    assert verdict in {"OK", "NOT_RUN"} and 0.0 <= score <= 1.0


def test_memorization_reports_skipped_rows_in_the_evidence_not_on_stdout(capsys) -> None:
    rows = [
        {"text": "alpha beta gamma delta epsilon zeta eta theta"},
        {"text": "start of row it is a"},
    ]
    result = score_memorization(rows, lambda prefix: "unrelated words entirely")
    assert capsys.readouterr().out == ""
    assert "skipped" in result.evidence