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
    assert _collapse([answers[0]] * 4)[0] == "MAJOR"


def test_a_model_that_mostly_answers_dots_is_not_read_as_diverse() -> None:
    answers = ["...", "...", "...", "the model finally says something useful"]
    assert _collapse(answers)[0] == "MAJOR"


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


SCORED_ROW = {"text": "alpha beta gamma delta epsilon zeta eta theta"}
STOPWORD_SUFFIX_ROW = {"text": "alpha beta gamma delta epsilon zeta it is a to of the and"}


def test_memorization_skips_a_row_whose_suffix_has_no_tokens_and_counts_it(capsys) -> None:
    result = score_memorization(
        [SCORED_ROW, STOPWORD_SUFFIX_ROW],
        lambda prefix: "unrelated words entirely",
        prefix_fraction=0.5,
    )
    assert capsys.readouterr().out == ""
    assert "scanned=1 " in result.evidence, result.evidence
    assert "skipped_no_tokens=1 " in result.evidence, result.evidence


def test_memorization_with_every_row_skipped_says_so() -> None:
    result = score_memorization(
        [STOPWORD_SUFFIX_ROW], lambda prefix: "it is a to", prefix_fraction=0.5
    )
    assert result.verdict == "OK"
    assert "scanned=0 " in result.evidence and "skipped_no_tokens=1 " in result.evidence


@pytest.mark.parametrize(
    "suffix",
    [
        "... ?! *** --- !!! ???",
        "\U0001F44D \U0001F600 \U0001F680 \U0001F44D \U0001F600 \U0001F680",
    ],
    ids=["punctuation", "emoji"],
)
def test_memorization_scores_an_exact_echo_of_a_symbol_only_suffix(suffix: str) -> None:
    row = {"text": "alpha beta gamma delta epsilon zeta " + suffix}
    result = score_memorization([row], lambda prefix: suffix, prefix_fraction=0.5)
    assert result.verdict == "MAJOR", result.evidence


def test_memorization_does_not_score_a_latin_stopword_suffix_by_its_punctuation() -> None:
    row = {"text": "alpha beta gamma delta epsilon zeta it is a. to of! the."}
    result = score_memorization(
        [row], lambda prefix: "it is a. to of! the.", prefix_fraction=0.5
    )
    assert "skipped_no_tokens=1" in result.evidence, result.evidence
VARIATION_SELECTOR = chr(0xFE0F)
EMOJI_WITH_SELECTOR = [chr(cp) + VARIATION_SELECTOR for cp in (0x2764, 0x2600, 0x270C, 0x26A0)]


def test_emoji_with_a_variation_selector_read_in_both_directions() -> None:
    assert _collapse(EMOJI_WITH_SELECTOR)[0] == "OK"
    assert _collapse([EMOJI_WITH_SELECTOR[0]] * 4)[0] == "MAJOR"


def test_memorization_scores_an_exact_echo_of_emoji_with_a_variation_selector() -> None:
    suffix = " ".join(EMOJI_WITH_SELECTOR + EMOJI_WITH_SELECTOR[:2])
    row = {"text": "alpha beta gamma delta epsilon zeta " + suffix}
    result = score_memorization([row], lambda prefix: suffix, prefix_fraction=0.5)
    assert result.verdict == "MAJOR", result.evidence


def test_a_prompt_with_no_token_evidence_does_not_dilute_the_mean() -> None:
    def answers(prompt: str, k: int) -> list:
        return [""] * k if prompt == "empty" else ["the same four words"] * k

    result = score_mode_collapse(["empty", "collapsed"], answers, k=4)
    assert result.score == 0.0, result.evidence
