# -*- coding: utf-8 -*-
"""Tests for Issue #1233: Unicode tokenization support in soup diagnose and live_eval.

Validates that non-Latin scripts (Cyrillic, Greek, Arabic, Hebrew, Hindi,
Chinese, Japanese, Korean, Thai, Vietnamese, Turkish) and digits tokenize
accurately without collapsing or generating false MAJOR verdicts in mode
collapse, memorization, and forgetting probes.
"""

from __future__ import annotations

import time

import pytest

from soup_cli.utils._eval_text import tokenize
from soup_cli.utils.diagnose.forgetting import score_forgetting
from soup_cli.utils.diagnose.memorization import score_memorization
from soup_cli.utils.diagnose.mode_collapse import score_mode_collapse
from soup_cli.utils.live_eval import token_f1


class TestUnicodeTokenize:
    def test_russian_words_and_digits(self) -> None:
        text = "Быстрая коричневая лиса прыгает через 42 забора"
        toks = tokenize(text)
        assert toks == [
            "быстрая",
            "коричневая",
            "лиса",
            "прыгает",
            "через",
            "42",
            "забора",
        ]

    def test_greek_words(self) -> None:
        text = "Η γρήγορη καφέ αλεπού 42"
        toks = tokenize(text)
        assert "γρήγορη" in toks
        assert "αλεπού" in toks
        assert "42" in toks

    def test_arabic_words(self) -> None:
        text = "الثعلب البني السريع 42"
        toks = tokenize(text)
        assert toks == ["الثعلب", "البني", "السريع", "42"]

    def test_hebrew_words(self) -> None:
        text = "השועל החום הזריז 42"
        toks = tokenize(text)
        assert toks == ["השועל", "החום", "הזריז", "42"]

    def test_hindi_words_with_combining_marks(self) -> None:
        text = "तेज़ भूरी लोमड़ी 42"
        toks = tokenize(text)
        assert "तेज़" in toks
        assert "भूरी" in toks
        assert "लोमड़ी" in toks
        assert "42" in toks

    def test_chinese_character_bigrams(self) -> None:
        text = "敏捷的棕色狐狸"
        toks = tokenize(text)
        assert toks == ["敏捷", "捷的", "的棕", "棕色", "色狐", "狐狸"]

    def test_japanese_kana_and_kanji(self) -> None:
        text = "素早いキツネ"
        toks = tokenize(text)
        assert "素早" in toks
        assert "早い" in toks
        assert "キツ" in toks

    def test_korean_words(self) -> None:
        text = "빠른 갈색 여우 42"
        toks = tokenize(text)
        assert toks == ["빠른", "갈색", "여우", "42"]

    def test_thai_character_bigrams(self) -> None:
        text = "สุนัข"
        toks = tokenize(text)
        assert len(toks) > 1

    def test_vietnamese_tone_marks(self) -> None:
        text = "Con cáo nâu nhanh nhẹn"
        toks = tokenize(text)
        assert "cáo" in toks
        assert "nhanh" in toks
        assert "nhẹn" in toks

    def test_turkish_dotted_i_normalization(self) -> None:
        text = "İstanbul Hızlı"
        toks = tokenize(text)
        assert toks[0] == "istanbul"
        assert toks[1] == "hızlı"

    def test_french_accents(self) -> None:
        text = "Zürich château élève"
        toks = tokenize(text)
        assert toks == ["zürich", "château", "élève"]

    def test_stopwords_filtering(self) -> None:
        text = "the quick brown fox and a dog"
        filtered = tokenize(text, filter_stopwords=True)
        assert "the" not in filtered
        assert "and" not in filtered
        assert "quick" in filtered
        assert "fox" in filtered

        unfiltered = tokenize(text, filter_stopwords=False)
        assert "the" in unfiltered
        assert "and" in unfiltered

    def test_short_ascii_words_with_filter_stopwords(self) -> None:
        text = "a b c hello"
        assert tokenize(text, filter_stopwords=True) == ["hello"]
        assert tokenize(text, filter_stopwords=False) == ["a", "b", "c", "hello"]

    def test_empty_and_non_str_inputs(self) -> None:
        assert tokenize("") == []
        assert tokenize(None) == []  # type: ignore[arg-type]
        assert tokenize("   \n\t  ") == []
        assert tokenize("---___---") == []


class TestModeCollapseUnicode:
    def test_diverse_russian_outputs_ok(self) -> None:
        templates = [
            "Москва является столицей и крупнейшим городом России",
            "Париж знаменит Эйфелевой башней и музеем Лувр",
            "Токио сочетает ультрасовременные небоскребы и древние храмы",
            "Берлин известен Бранденбургскими воротами и богатой историей",
        ]

        def multi(_prompt: str, k: int) -> list[str]:
            return templates[:k]

        score = score_mode_collapse(["расскажи о городе"], multi, k=4, ngram_n=2)
        assert score.verdict == "OK"

    def test_collapsed_russian_outputs_major(self) -> None:
        multi = lambda p, k: ["абсолютно одинаковый ответ модели"] * k  # noqa: E731
        score = score_mode_collapse(["расскажи о городе"], multi, k=4)
        assert score.verdict == "MAJOR"

    def test_diverse_chinese_outputs_ok(self) -> None:
        templates = [
            "北京是中国的首都和文化中心拥有悠久的历史",
            "上海是重要的经济和金融中心拥有东方明珠",
            "广州是著名的岭南文化发源地和美食之都",
            "深圳是中国高科技产业和创新发展的代表城市",
        ]

        def multi(_prompt: str, k: int) -> list[str]:
            return templates[:k]

        score = score_mode_collapse(["介绍城市"], multi, k=4, ngram_n=2)
        assert score.verdict == "OK"

    def test_collapsed_chinese_outputs_major(self) -> None:
        multi = lambda p, k: ["完全相同的模型输出回复"] * k  # noqa: E731
        score = score_mode_collapse(["介绍城市"], multi, k=4)
        assert score.verdict == "MAJOR"

    def test_diverse_arabic_outputs_ok(self) -> None:
        templates = [
            "القاهرة عاصمة مصر وتشتهر بالأهرامات وتاريخها العريق",
            "الرياض عاصمة المملكة العربية السعودية ومركز اقتصادي كبير",
            "بغداد مدينة عريقة على ضفاف نهر دجلة ولها تراث أدبي",
            "دمشق من أقدم المدن المأهولة في العالم وتتميز بآثارها",
        ]

        def multi(_prompt: str, k: int) -> list[str]:
            return templates[:k]

        score = score_mode_collapse(["حدثني عن مدينة"], multi, k=4, ngram_n=2)
        assert score.verdict == "OK"

    def test_collapsed_arabic_outputs_major(self) -> None:
        multi = lambda p, k: ["نفس الاستجابة المتطابقة تماما هنا"] * k  # noqa: E731
        score = score_mode_collapse(["حدثني عن مدينة"], multi, k=4)
        assert score.verdict == "MAJOR"


class TestMemorizationUnicode:
    def test_russian_no_memorization(self) -> None:
        rows = [{"text": "быстрая коричневая лиса прыгает через забор на ферме"}]
        gen = lambda p: "совершенно другой несвязанный ответ ассистента"  # noqa: E731
        score = score_memorization(rows, gen, prefix_fraction=0.4)
        assert score.verdict == "OK"

    def test_russian_full_memorization(self) -> None:
        rows = [{"text": "быстрая коричневая лиса прыгает через забор на ферме"}]
        # Generator echoes suffix verbatim
        gen = lambda p: "прыгает через забор на ферме"  # noqa: E731
        score = score_memorization(rows, gen, prefix_fraction=0.4, echo_threshold=0.5)
        assert score.verdict == "MAJOR"

    def test_chinese_no_memorization(self) -> None:
        rows = [{"text": "敏捷的棕色狐狸跳过懒惰的狗并在森林深处奔跑"}]
        gen = lambda p: "完全不同且毫无关联的系统回复内容"  # noqa: E731
        score = score_memorization(rows, gen, prefix_fraction=0.4)
        assert score.verdict == "OK"

    def test_chinese_full_memorization(self) -> None:
        rows = [{"text": "敏捷的棕色狐狸跳过懒惰的狗并在森林深处奔跑"}]
        gen = lambda p: "并在森林深处奔跑"  # noqa: E731
        score = score_memorization(rows, gen, prefix_fraction=0.5, echo_threshold=0.5)
        assert score.verdict == "MAJOR"

    def test_tokenizer_aware_greek_does_not_falsely_flag_unrelated(self) -> None:
        rows = [{"text": "Η γρήγορη καφέ αλεπού πηδά πάνω από σαράντα δύο σκυλιά"}]
        # Completely unrelated Greek sentence
        gen = lambda p: "Ο καιρός σήμερα στην Αθήνα είναι πολύ καλός και ηλιόλουστος"  # noqa: E731
        try:
            score = score_memorization(
                rows, gen, prefix_fraction=0.4, echo_threshold=0.5, tokenizer="gpt2"
            )
        except (OSError, ValueError) as exc:
            pytest.skip(f"gpt2 tokenizer unavailable: {exc}")
        assert score.verdict == "OK"

    def test_tokenizer_aware_greek_catches_echo(self) -> None:
        rows = [{"text": "Η γρήγορη καφέ αλεπού πηδά πάνω από σαράντα δύο σκυλιά στο χωράφι"}]
        # Generator echoes Greek suffix
        gen = lambda p: "πηδά πάνω από σαράντα δύο σκυλιά στο χωράφι"  # noqa: E731
        try:
            score = score_memorization(
                rows, gen, prefix_fraction=0.4, echo_threshold=0.5, tokenizer="gpt2"
            )
        except (OSError, ValueError) as exc:
            pytest.skip(f"gpt2 tokenizer unavailable: {exc}")
        assert score.verdict == "MAJOR"


class TestTokenF1Unicode:
    def test_russian_exact_match(self) -> None:
        assert token_f1("Быстрая лиса", "Быстрая лиса") == pytest.approx(1.0)

    def test_russian_partial_match(self) -> None:
        score = token_f1("быстрая лиса", "быстрая собака")
        assert 0.4 < score < 0.6

    def test_russian_no_match(self) -> None:
        assert token_f1("быстрая лиса", "медленная черепаха") == 0.0

    def test_russian_target_punctuation_only_gives_zero(self) -> None:
        assert token_f1("быстрая лиса", "...") == 0.0

    def test_chinese_exact_match(self) -> None:
        assert token_f1("敏捷的狐狸", "敏捷的狐狸") == pytest.approx(1.0)

    def test_arabic_exact_match(self) -> None:
        assert token_f1("الثعلب السريع", "الثعلب السريع") == pytest.approx(1.0)

    def test_ascii_regression_safety(self) -> None:
        assert token_f1("a b c", "a b c") == pytest.approx(1.0)
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)
        assert token_f1("foo", "bar") == 0.0
        assert token_f1("", "x") == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "a" + "\u0301" * 50_000 + "\u0316" * 50_000,
        # U+0F73 has combining class 0 but decomposes into two non-starters,
        # so only a category-based filter bounds this run.
        "\u0f40" + "\u0f73\u0f71" * 50_000,
    ],
    ids=["reverse_ordered_marks", "tibetan_decomposing_sign"],
)
def test_long_mark_runs_tokenize_in_linear_time(text: str) -> None:
    start = time.perf_counter()
    tokenize(text)
    token_f1(text, text)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("state of the art model", "state-of-the-art model", 1.0),
        ("GPT 4 is a model", "GPT-4 is a model", 1.0),
        ("the snake case name", "the snake_case name", 1.0),
        ("well known fact", "well-known fact", 1.0),
        ("It costs 42 dollars", "It costs 43 dollars", 0.75),
    ],
)
def test_token_f1_english_scores_are_unchanged(pred: str, gold: str, expected: float) -> None:
    assert token_f1(pred, gold) == pytest.approx(expected)


def test_english_joiners_preserved_in_tokenize() -> None:
    assert tokenize("state-of-the-art snake_case gpt-4") == [
        "state-of-the-art",
        "snake_case",
        "gpt-4",
    ]


def test_composed_and_decomposed_accents_match() -> None:
    assert token_f1("café", "cafe\u0301") == pytest.approx(1.0)


def test_emoji_variation_selector_does_not_produce_token() -> None:
    assert tokenize("\u2764\ufe0f") == []


def test_forgetting_reads_major_when_a_greek_adapter_answers_dots() -> None:
    targets = [
        "Η πρωτεύουσα είναι η Αθήνα και το λιμάνι είναι ο Πειραιάς",
        "Μία ώρα έχει εξήντα λεπτά και ένα λεπτό εξήντα δευτερόλεπτα",
    ]
    base = sum(token_f1(t, t) for t in targets) / len(targets)
    adapter = sum(token_f1("...", t) for t in targets) / len(targets)
    assert score_forgetting({"heldout": base}, {"heldout": adapter}).verdict == "MAJOR"


SCRIPTS = {
    "greek": [
        "Η πρωτεύουσα είναι η Αθήνα και το λιμάνι είναι ο Πειραιάς",
        "Μία ώρα έχει εξήντα λεπτά και ένα λεπτό εξήντα δευτερόλεπτα",
        "Η μπανάνα είναι κίτρινη και έχει κάλιο",
        "Ο ουρανός είναι μπλε λόγω της σκέδασης του φωτός",
    ],
    "hebrew": [
        "ירושלים היא בירת ישראל ועיר עתיקה מאוד",
        "בשעה אחת יש שישים דקות ובדקה שישים שניות",
        "הבננה צהובה ומכילה הרבה אשלגן",
        "השמיים כחולים בגלל פיזור של אור השמש",
    ],
    "hindi": [
        "नई दिल्ली भारत की राजधानी और बड़ा शहर है",
        "एक घंटे में साठ मिनट और एक मिनट में साठ सेकंड होते हैं",
        "केला पीला होता है और उसमें पोटैशियम होता है",
        "आकाश नीला दिखता है क्योंकि सूर्य का प्रकाश बिखरता है",
    ],
    "japanese": [
        "東京は日本の首都で大きな港もあります",
        "一時間は六十分で一分は六十秒です",
        "バナナは黄色くてカリウムを含みます",
        "空が青いのは太陽の光が散乱するからです",
    ],
    "korean": [
        "서울은 한국의 수도이며 큰 항구가 있습니다",
        "한 시간은 육십 분이고 일 분은 육십 초입니다",
        "바나나는 노랗고 칼륨이 들어 있습니다",
        "하늘이 파란 이유는 햇빛이 산란되기 때문입니다",
    ],
    "thai": [
        "กรุงเทพเป็นเมืองหลวงของประเทศไทย",
        "หนึ่งชั่วโมงมีหกสิบนาที",
        "กล้วยมีสีเหลืองและมีโพแทสเซียม",
        "ท้องฟ้าเป็นสีฟ้าเพราะแสงอาทิตย์กระเจิง",
    ],
    "vietnamese": [
        "Hà Nội là thủ đô của Việt Nam và có nhiều hồ",
        "Một giờ có sáu mươi phút và một phút có sáu mươi giây",
        "Chuối có màu vàng và chứa nhiều kali",
        "Bầu trời màu xanh vì ánh nắng bị tán xạ",
    ],
    "turkish": [
        "İstanbul Türkiye'nin en büyük şehri ve önemli bir limandır",
        "Bir saatte altmış dakika ve bir dakikada altmış saniye vardır",
        "Muz sarıdır ve içinde bol potasyum bulunur",
        "Gökyüzü güneş ışığı saçıldığı için mavi görünür",
    ],
    "french": [
        "Ça coûte très cher à Zürich et le port est loin",
        "Une heure compte soixante minutes et une minute soixante secondes",
        "La banane est jaune et contient du potassium",
        "Le ciel paraît bleu parce que la lumière est diffusée",
    ],
}


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_probes_discriminate_in_both_directions(script: str) -> None:
    answers = SCRIPTS[script]
    diverse = score_mode_collapse(["p"], lambda p, k: answers[:k], k=4)
    same = score_mode_collapse(["p"], lambda p, k: [answers[0]] * k, k=4)
    assert (diverse.verdict, same.verdict) == ("OK", "MAJOR"), (diverse, same)

    row = answers[1]
    unrelated = score_memorization([{"text": row}], lambda prefix: answers[3])
    echo = score_memorization(
        [{"text": row}],
        lambda prefix: row[len(prefix) :].strip() if row.startswith(prefix) else row,
    )
    assert (unrelated.verdict, echo.verdict) == ("OK", "MAJOR"), (unrelated, echo)
