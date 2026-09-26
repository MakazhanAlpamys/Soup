# -*- coding: utf-8 -*-
"""Tests for Issue #1233: Unicode tokenization support in soup diagnose and live_eval.

Validates that non-Latin scripts (Cyrillic, Greek, Arabic, Hebrew, Hindi,
Chinese, Japanese, Korean, Thai, Vietnamese, Turkish) and digits tokenize
accurately without collapsing or generating false MAJOR verdicts in mode
collapse, memorization, and forgetting probes.
"""

from __future__ import annotations

import pytest

from soup_cli.utils._eval_text import tokenize
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
        rows = [
            {"text": "быстрая коричневая лиса прыгает через забор на ферме"}
        ]
        gen = lambda p: "совершенно другой несвязанный ответ ассистента"  # noqa: E731
        score = score_memorization(rows, gen, prefix_fraction=0.4)
        assert score.verdict == "OK"

    def test_russian_full_memorization(self) -> None:
        rows = [
            {"text": "быстрая коричневая лиса прыгает через забор на ферме"}
        ]
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
        rows = [
            {"text": "Η γρήγορη καφέ αλεπού πηδά πάνω από σαράντα δύο σκυλιά"}
        ]
        # Completely unrelated Greek sentence
        gen = lambda p: "Ο καιρός σήμερα στην Αθήνα είναι πολύ καλός και ηλιόλουστος"  # noqa: E731
        score = score_memorization(
            rows, gen, prefix_fraction=0.4, echo_threshold=0.5, tokenizer="gpt2"
        )
        assert score.verdict == "OK"

    def test_tokenizer_aware_greek_catches_echo(self) -> None:
        rows = [
            {"text": "Η γρήγορη καφέ αλεπού πηδά πάνω από σαράντα δύο σκυλιά στο χωράφι"}
        ]
        # Generator echoes Greek suffix
        gen = lambda p: "πηδά πάνω από σαράντα δύο σκυλιά στο χωράφι"  # noqa: E731
        score = score_memorization(
            rows, gen, prefix_fraction=0.4, echo_threshold=0.5, tokenizer="gpt2"
        )
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
