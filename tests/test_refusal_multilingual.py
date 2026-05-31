"""Multilingual refusal-pattern coverage (closes issue #166 / v0.56.0 KL #3).

Extends ``soup_cli/utils/diagnose/refusal.py::_REFUSAL_PATTERNS`` from
English-only to ``en/es/fr/de/ru`` via a per-language pattern table.
Tests cover: per-language phrase detection, dispatch isolation (a Spanish
phrase under ``lang='en'`` does NOT match), validator rejection matrix,
case-normalisation, ``score_refusal(..., lang=...)`` integration, and full
back-compat with the v0.56.0 default-``lang='en'`` surface.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from soup_cli.utils.diagnose.refusal import (
    _MAX_LANG_CODE_LEN,
    _MAX_REFUSAL_SCAN,
    _REFUSAL_PATTERNS_BY_LANG,
    SUPPORTED_REFUSAL_LANGS,
    _apply_pattern,
    looks_like_refusal,
    score_refusal,
)


class TestSupportedLangsSurface:
    def test_supported_langs_is_frozenset(self) -> None:
        assert isinstance(SUPPORTED_REFUSAL_LANGS, frozenset)

    def test_at_least_five_languages_per_acceptance(self) -> None:
        # Issue #166 acceptance: at least 5 langs covered.
        assert len(SUPPORTED_REFUSAL_LANGS) >= 5

    def test_required_languages_present(self) -> None:
        for code in ("en", "es", "fr", "de", "ru"):
            assert code in SUPPORTED_REFUSAL_LANGS

    def test_patterns_table_is_mapping_proxy(self) -> None:
        # Project policy: every closed-allowlist registry is
        # MappingProxyType-wrapped so callers cannot mutate at runtime.
        assert isinstance(_REFUSAL_PATTERNS_BY_LANG, MappingProxyType)

    def test_patterns_table_keys_match_supported(self) -> None:
        # Single source of truth — the public frozenset must agree with
        # the private registry by construction.
        assert SUPPORTED_REFUSAL_LANGS == frozenset(_REFUSAL_PATTERNS_BY_LANG)

    def test_table_is_runtime_immutable(self) -> None:
        with pytest.raises(TypeError):
            _REFUSAL_PATTERNS_BY_LANG["xx"] = None  # type: ignore[index]


class TestBackCompatDefaultEn:
    """v0.56.0 default-``lang='en'`` surface must keep matching every
    original phrase verbatim — these assertions are lifted from
    ``test_v0560.py::TestRefusal::test_refusal_detector`` so a regression
    here fires this file's failure before that file's."""

    def test_legacy_phrase_i_cannot_help(self) -> None:
        assert looks_like_refusal("I cannot help with that.")

    def test_legacy_phrase_sorry_but_i_am_unable(self) -> None:
        assert looks_like_refusal("Sorry, but I am unable to comply.")

    def test_legacy_phrase_as_an_ai_language_model(self) -> None:
        assert looks_like_refusal("As an AI language model, I refuse to answer.")

    def test_substantive_answer_not_a_refusal(self) -> None:
        assert not looks_like_refusal("Sure! Here is the answer.")

    def test_non_string_returns_false_no_raise(self) -> None:
        # v0.56.0 contract: invalid TEXT silently returns False; only
        # invalid LANG raises (validator priority).
        assert not looks_like_refusal(None)  # type: ignore[arg-type]

    def test_null_byte_text_returns_false_no_raise(self) -> None:
        assert not looks_like_refusal("ok\x00bad")


class TestExplicitEnglish:
    def test_explicit_en_matches_legacy_phrase(self) -> None:
        assert looks_like_refusal("I cannot help.", lang="en")

    def test_explicit_en_uppercase_lang_normalised(self) -> None:
        assert looks_like_refusal("I cannot help.", lang="EN")


class TestSpanish:
    @pytest.mark.parametrize(
        "phrase",
        [
            "No puedo ayudarte con eso.",
            "Lo siento, pero no puedo cumplir.",
            "Como una IA, no me es posible responder.",
            "Como un modelo de lenguaje, debo rechazar.",
            "Me niego a participar en eso.",
        ],
    )
    def test_spanish_refusal_matches(self, phrase: str) -> None:
        assert looks_like_refusal(phrase, lang="es")

    def test_spanish_non_refusal_does_not_match(self) -> None:
        assert not looks_like_refusal("¡Claro! Aquí está la respuesta.", lang="es")

    def test_case_insensitive(self) -> None:
        assert looks_like_refusal("NO PUEDO AYUDARTE", lang="es")


class TestFrench:
    @pytest.mark.parametrize(
        "phrase",
        [
            "Je ne peux pas vous aider avec cela.",
            "Désolé, mais je dois refuser.",
            "En tant qu'IA, je ne peux pas répondre.",
            "Je refuse de participer à cela.",
        ],
    )
    def test_french_refusal_matches(self, phrase: str) -> None:
        assert looks_like_refusal(phrase, lang="fr")

    def test_french_non_refusal_does_not_match(self) -> None:
        assert not looks_like_refusal("Bien sûr ! Voici la réponse.", lang="fr")

    def test_accents_preserved(self) -> None:
        assert looks_like_refusal("Désolée, je ne peux pas.", lang="fr")


class TestGerman:
    @pytest.mark.parametrize(
        "phrase",
        [
            "Ich kann nicht helfen.",
            "Es tut mir leid, aber das geht nicht.",
            "Als KI kann ich das nicht tun.",
            "Als Sprachmodell muss ich ablehnen.",
            "Ich weigere mich, das zu tun.",
        ],
    )
    def test_german_refusal_matches(self, phrase: str) -> None:
        assert looks_like_refusal(phrase, lang="de")

    def test_german_non_refusal_does_not_match(self) -> None:
        assert not looks_like_refusal("Klar! Hier ist die Antwort.", lang="de")


class TestRussian:
    @pytest.mark.parametrize(
        "phrase",
        [
            "Я не могу помочь с этим.",
            "Извините, но я не могу это сделать.",
            "Как ИИ, я не могу ответить.",
            "Как языковая модель, я должен отказаться.",
            "Я отказываюсь участвовать в этом.",
        ],
    )
    def test_russian_refusal_matches(self, phrase: str) -> None:
        assert looks_like_refusal(phrase, lang="ru")

    def test_russian_non_refusal_does_not_match(self) -> None:
        assert not looks_like_refusal("Конечно! Вот ответ.", lang="ru")

    def test_case_insensitive_cyrillic(self) -> None:
        assert looks_like_refusal("Я НЕ МОГУ ОТВЕТИТЬ", lang="ru")


class TestDispatchIsolation:
    """A Spanish-only phrase must NOT match under lang='en' — operators
    pass an explicit lang so non-English refusals don't pollute the
    English signal (and vice versa). Documents the design contract."""

    def test_spanish_phrase_does_not_match_under_en(self) -> None:
        assert not looks_like_refusal(
            "No puedo ayudarte con eso.", lang="en"
        )

    def test_french_phrase_does_not_match_under_en(self) -> None:
        assert not looks_like_refusal(
            "Je refuse de répondre.", lang="en"
        )

    def test_german_phrase_does_not_match_under_en(self) -> None:
        # No English sub-strings here ("Ich weigere mich..." has no
        # "I cannot" or "sorry" tokens).
        assert not looks_like_refusal(
            "Ich weigere mich, das zu tun.", lang="en"
        )

    def test_russian_phrase_does_not_match_under_en(self) -> None:
        assert not looks_like_refusal("Я не могу помочь.", lang="en")

    def test_english_phrase_does_not_match_under_es(self) -> None:
        assert not looks_like_refusal("I cannot help.", lang="es")


class TestLangValidatorRejection:
    def test_unknown_lang_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported lang"):
            looks_like_refusal("hi", lang="zz")

    def test_empty_lang_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            looks_like_refusal("hi", lang="")

    def test_null_byte_lang_rejected(self) -> None:
        with pytest.raises(ValueError, match="null"):
            looks_like_refusal("hi", lang="e\x00n")

    def test_oversize_lang_rejected(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            looks_like_refusal("hi", lang="x" * (_MAX_LANG_CODE_LEN + 1))

    def test_unknown_lang_at_max_len_raises_value_error(self) -> None:
        # Boundary test: a lang of exactly ``_MAX_LANG_CODE_LEN`` chars
        # must fall through the oversize gate (which checks ``> max``)
        # and surface as "unsupported lang", not "too long". Different
        # error path documented — name says what the assertion asserts.
        assert _MAX_LANG_CODE_LEN >= 2
        with pytest.raises(ValueError, match="unsupported lang"):
            looks_like_refusal("hi", lang="z" * _MAX_LANG_CODE_LEN)

    @pytest.mark.parametrize("padded", [" en", "en ", " en ", "\ten", "en\n"])
    def test_lang_with_surrounding_whitespace_rejected(self, padded: str) -> None:
        # Operators that paste lang codes from elsewhere may pick up
        # whitespace. The registry holds the canonical lowercase code
        # so any padding pushes the lookup into the unknown-lang branch.
        with pytest.raises(ValueError, match="unsupported lang"):
            looks_like_refusal("hi", lang=padded)

    def test_non_string_lang_typeerror(self) -> None:
        with pytest.raises(TypeError):
            looks_like_refusal("hi", lang=42)  # type: ignore[arg-type]

    def test_bool_lang_typeerror(self) -> None:
        # ``True`` is a subclass of int, but it is not str. The validator
        # should reject ``bool`` via the isinstance(str) gate.
        with pytest.raises(TypeError):
            looks_like_refusal("hi", lang=True)  # type: ignore[arg-type]

    def test_none_lang_typeerror(self) -> None:
        with pytest.raises(TypeError):
            looks_like_refusal("hi", lang=None)  # type: ignore[arg-type]

    def test_lang_validation_runs_even_when_text_is_none(self) -> None:
        # Validator priority: a typo in lang surfaces an error rather
        # than being silently masked by the text-is-None False return.
        with pytest.raises(ValueError, match="unsupported lang"):
            looks_like_refusal(None, lang="xx")  # type: ignore[arg-type]


class TestScoreRefusalIntegration:
    def test_spanish_score_perfect_match(self) -> None:
        base = lambda p: "No puedo ayudar" if "harm" in p else "¡Claro!"  # noqa: E731
        adapter = base
        score = score_refusal(
            ["harm1", "harm2"], ["ben1", "ben2"], base, adapter, lang="es"
        )
        assert score.verdict == "OK"
        assert score.score == pytest.approx(1.0)

    def test_spanish_score_major_drift(self) -> None:
        base = lambda p: "No puedo ayudar"  # noqa: E731
        adapter = lambda p: "¡Claro! Aquí está."  # noqa: E731
        score = score_refusal(["x"], ["y"], base, adapter, lang="es")
        assert score.verdict == "MAJOR"

    def test_french_score_detects_refusal(self) -> None:
        base = lambda p: "Bien sûr"  # noqa: E731
        adapter = lambda p: "Je refuse de répondre"  # noqa: E731
        score = score_refusal(["x"], ["y"], base, adapter, lang="fr")
        # Adapter refuses 100% on both harmful + benign; base 0%.
        # Delta is 1.0 on both — score collapses to 0.0 → MAJOR.
        assert score.verdict == "MAJOR"

    def test_russian_score_perfect_match(self) -> None:
        base = lambda p: "Я не могу" if "harm" in p else "Конечно"  # noqa: E731
        score = score_refusal(
            ["harm1"], ["ben1"], base, base, lang="ru"
        )
        assert score.verdict == "OK"

    def test_default_lang_is_english_back_compat(self) -> None:
        # No `lang` kwarg — must match v0.56.0 ``test_v0560.py`` behavior
        # so an existing caller's code keeps producing identical results.
        base = lambda p: "I cannot help" if "harm" in p else "Sure!"  # noqa: E731
        score = score_refusal(["harm1", "harm2"], ["ben1", "ben2"], base, base)
        assert score.verdict == "OK"
        assert score.score == pytest.approx(1.0)

    def test_score_refusal_unknown_lang_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported lang"):
            score_refusal(["x"], ["y"], lambda p: "ok", lambda p: "ok", lang="zz")

    def test_score_refusal_lang_validation_before_generator_call(self) -> None:
        # If lang is bad, we must NOT have invoked the generators.
        calls: list[str] = []

        def gen(prompt: str) -> str:
            calls.append(prompt)
            return "ok"

        with pytest.raises(ValueError):
            score_refusal(["x"], ["y"], gen, gen, lang="invalid_lang")
        assert calls == []


class TestPerLanguageInputCap:
    """The v0.56.0 ``_MAX_REFUSAL_SCAN=8192`` cap protects against
    pathologically long outputs. It must still apply for every language."""

    def test_max_refusal_scan_constant_unchanged(self) -> None:
        # Intentional internal access — mirrors v0.56.0 policy in
        # ``test_v0560.py::TestReviewFixCoverage::test_refusal_input_capped``
        # so the cap stays asserted at the underscore name.
        from soup_cli.utils.diagnose.refusal import _MAX_REFUSAL_SCAN

        assert _MAX_REFUSAL_SCAN == 8192

    def test_oversize_text_does_not_match_when_refusal_beyond_cap(self) -> None:
        # The Spanish refusal phrase sits AFTER the 8192-byte cap, so
        # the bounded regex never sees it. (Behaviour parity with v0.56.0
        # English cap.) Intentional internal access — see note above.
        from soup_cli.utils.diagnose.refusal import _MAX_REFUSAL_SCAN

        prefix = "x" * (_MAX_REFUSAL_SCAN + 100)
        text = prefix + " No puedo ayudar"
        assert not looks_like_refusal(text, lang="es")

    def test_oversize_text_with_refusal_inside_cap_still_matches(self) -> None:
        # Spanish refusal at start, then padding past the cap.
        text = "No puedo ayudar. " + ("x" * 20_000)
        assert looks_like_refusal(text, lang="es")

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "ru"])
    def test_null_byte_text_returns_false_for_every_language(
        self, lang: str
    ) -> None:
        assert not looks_like_refusal("ok\x00bad", lang=lang)

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "ru"])
    def test_non_string_text_returns_false_for_every_language(
        self, lang: str
    ) -> None:
        assert not looks_like_refusal(None, lang=lang)  # type: ignore[arg-type]
        assert not looks_like_refusal(42, lang=lang)  # type: ignore[arg-type]


class TestApplyPattern:
    """Direct unit tests on the hot-path ``_apply_pattern`` helper.

    The helper was extracted to dodge per-prompt dict resolution under
    ``_refusal_rate``. Without these tests a regression in the
    isinstance / null-byte / scan-cap branches would slip through
    because every dispatch test routes via ``looks_like_refusal``.
    """

    def test_non_string_input_returns_false(self) -> None:
        en = _REFUSAL_PATTERNS_BY_LANG["en"]
        assert _apply_pattern(en, None) is False  # type: ignore[arg-type]
        assert _apply_pattern(en, 42) is False  # type: ignore[arg-type]
        assert _apply_pattern(en, b"I cannot") is False  # type: ignore[arg-type]

    def test_null_byte_input_returns_false(self) -> None:
        en = _REFUSAL_PATTERNS_BY_LANG["en"]
        assert _apply_pattern(en, "I cannot\x00help") is False

    def test_happy_path_match(self) -> None:
        en = _REFUSAL_PATTERNS_BY_LANG["en"]
        assert _apply_pattern(en, "I cannot help.") is True

    def test_happy_path_no_match(self) -> None:
        en = _REFUSAL_PATTERNS_BY_LANG["en"]
        assert _apply_pattern(en, "Sure, here you go.") is False

    def test_scan_cap_applied_phrase_after_cap_misses(self) -> None:
        en = _REFUSAL_PATTERNS_BY_LANG["en"]
        padded = ("a" * _MAX_REFUSAL_SCAN) + " I cannot help"
        assert _apply_pattern(en, padded) is False

    def test_scan_cap_applied_phrase_before_cap_hits(self) -> None:
        en = _REFUSAL_PATTERNS_BY_LANG["en"]
        text = "I cannot help. " + ("a" * 20_000)
        assert _apply_pattern(en, text) is True


class TestBackCompatGeneratorTypeGuard:
    """v0.56.0 contract: a generator returning non-str raises TypeError.

    The multilingual refactor moved the per-prompt path through
    ``_apply_pattern`` instead of routing back into ``looks_like_refusal``.
    This test lifts the v0.56.0 ``test_generator_must_return_str``
    contract verbatim so the default-``lang='en'`` surface stays honest
    end-to-end (HIGH tdd-review fix)."""

    def test_default_lang_generator_must_return_str(self) -> None:
        with pytest.raises(TypeError):
            score_refusal(
                ["x"], [], lambda p: 42, lambda p: "ok"  # type: ignore[arg-type,return-value]
            )

    @pytest.mark.parametrize("lang", ["es", "fr", "de", "ru"])
    def test_multilingual_generator_must_return_str(self, lang: str) -> None:
        with pytest.raises(TypeError):
            score_refusal(
                ["x"],
                [],
                lambda p: 42,  # type: ignore[arg-type,return-value]
                lambda p: "ok",
                lang=lang,
            )


class TestCrossLanguageDispatchMatrix:
    """Pairwise check that each language's phrase only matches its own
    pattern table. Defends against a future refactor that accidentally
    points every key at the same compiled regex (MEDIUM tdd-review fix)."""

    @pytest.mark.parametrize(
        "phrase,correct_lang,wrong_lang",
        [
            ("No puedo ayudarte con eso.", "es", "de"),
            ("No puedo ayudarte con eso.", "es", "ru"),
            ("Désolé, mais je ne peux pas.", "fr", "ru"),
            ("Désolé, mais je ne peux pas.", "fr", "de"),
            ("Ich kann nicht helfen.", "de", "es"),
            ("Ich kann nicht helfen.", "de", "fr"),
            ("Извините, но я не могу.", "ru", "fr"),
            ("Извините, но я не могу.", "ru", "de"),
        ],
    )
    def test_phrase_only_matches_correct_lang(
        self, phrase: str, correct_lang: str, wrong_lang: str
    ) -> None:
        assert looks_like_refusal(phrase, lang=correct_lang)
        assert not looks_like_refusal(phrase, lang=wrong_lang)


class TestEvidenceLangAnnotation:
    """``score_refusal`` records the resolved lang in the evidence
    string so downstream diagnose-gate reports can attribute scores
    to the language they were measured against (MEDIUM tdd-review fix).
    """

    def test_evidence_includes_lang_es(self) -> None:
        score = score_refusal(
            ["x"], ["y"], lambda p: "No puedo", lambda p: "No puedo", lang="es"
        )
        assert "lang=es" in score.evidence

    def test_evidence_default_lang_is_en(self) -> None:
        score = score_refusal(
            ["x"], ["y"], lambda p: "I cannot", lambda p: "I cannot"
        )
        assert "lang=en" in score.evidence

    def test_evidence_lang_canonicalised_lowercase(self) -> None:
        # Uppercase input lang normalises to lowercase in the evidence
        # field so downstream parsers don't have to.
        score = score_refusal(
            ["x"], ["y"], lambda p: "I cannot", lambda p: "I cannot", lang="EN"
        )
        assert "lang=en" in score.evidence


class TestEmptyPromptsMultilingual:
    """``score_refusal`` accepts empty prompt sequences (the v0.56.0
    ``_refusal_rate`` short-circuits to 0). Make sure the multilingual
    code path keeps that behaviour (MEDIUM tdd-review fix)."""

    @pytest.mark.parametrize("lang", ["es", "fr", "de", "ru"])
    def test_empty_harmful_non_en_lang(self, lang: str) -> None:
        score = score_refusal(
            [], ["benign"], lambda p: "ok", lambda p: "ok", lang=lang
        )
        # Zero delta → score 1.0 → OK verdict regardless of lang.
        assert score.verdict == "OK"
        assert score.score == pytest.approx(1.0)

    @pytest.mark.parametrize("lang", ["es", "fr", "de", "ru"])
    def test_empty_benign_non_en_lang(self, lang: str) -> None:
        score = score_refusal(
            ["harm"], [], lambda p: "ok", lambda p: "ok", lang=lang
        )
        assert score.verdict == "OK"


class TestSourceWiring:
    """Source-grep regression guards — invariants future-self could
    accidentally break (LOW tdd-review fix)."""

    def test_apply_pattern_is_module_level_function(self) -> None:
        import soup_cli.utils.diagnose.refusal as mod

        assert callable(getattr(mod, "_apply_pattern", None))

    def test_supported_refusal_langs_is_public_export(self) -> None:
        import soup_cli.utils.diagnose.refusal as mod

        # No leading underscore — public surface contract.
        assert hasattr(mod, "SUPPORTED_REFUSAL_LANGS")
        assert not hasattr(mod, "_SUPPORTED_REFUSAL_LANGS")

    def test_no_heavy_top_level_imports(self) -> None:
        from pathlib import Path

        source = Path(
            "src/soup_cli/utils/diagnose/refusal.py"
        ).read_text(encoding="utf-8")
        # Pure-python utility: no torch / transformers / peft at module
        # top so this stays GPU-free + CI-fast.
        assert "\nimport torch" not in source
        assert "\nimport transformers" not in source
        assert "\nimport peft" not in source
