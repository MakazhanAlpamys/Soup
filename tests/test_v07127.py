"""v0.71.27 "Fine-tune Doctor" — chat-template doctor + loss-mask X-ray +
preference linter (closes the release's three headline items, plus the
diagnose.py O_NOFOLLOW evidence-loader housekeeping rider).

- ``soup data doctor`` — template-compat report (EOS-missing / BOS
  duplication / no-system-role / unknown-role / truncation risk) +
  ``--show-mask N`` loss-mask X-ray through the REAL collator path.
- ``soup data lint`` — preference-data linter (dpo/kto): length bias
  (Cohen's d effect size), label imbalance, near-dup pairs (MinHash),
  chosen==rejected, prompt-leaked-into-completion.
- ``commands/diagnose.py::_load_evidence`` — O_NOFOLLOW + fstat hardening
  (backport of the v0.71.25 ``soup ship`` pattern).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import datasketch  # noqa: F401

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


# ---------------------------------------------------------------------------
# Fake tokenizer — char-level, extends the v0.36.0 test_assistant_mask.py
# `_FakeTokenizer` pattern with controllable EOS / BOS / system-role knobs.
# ---------------------------------------------------------------------------

_BOS_ID = -1001
_EOS_ID = -1002


class _FakeTokenizer:
    """Deterministic char-level fake tokenizer for doctor/lint unit tests.

    Renders ``<role>:content\\n`` per turn; each char maps to
    ``(ord(c) % 200) + 1`` so ids never collide with the negative BOS/EOS
    sentinels. Knobs simulate the real-world footguns the doctor detects:
    ``emit_eos=False`` (chat template never appends EOS after assistant
    turns), ``double_bos=True`` (template + tokenizer both prepend BOS),
    ``reject_system=True`` (Mistral-style templates without a system role).
    """

    bos_token_id = _BOS_ID
    pad_token_id = 0

    def __init__(
        self,
        *,
        supports_assistant_mask: bool = True,
        has_chat_template: bool = True,
        emit_eos: bool = True,
        double_bos: bool = False,
        reject_system: bool = False,
        fail_render: bool = False,
        eos_token_id=_EOS_ID,
        trailing_trained: bool = False,
    ):
        self.supports_assistant_mask = supports_assistant_mask
        self.chat_template = "fake-template" if has_chat_template else None
        self.emit_eos = emit_eos
        self.double_bos = double_bos
        self.reject_system = reject_system
        self.fail_render = fail_render
        self.eos_token_id = eos_token_id
        # v0.71.27 real-tokenizer smoke finding: SmolLM2's ChatML template
        # renders `<|im_end|>\n` for the LAST assistant turn with the
        # trailing `\n` landing INSIDE the trained span too (no later
        # message to fold it into a masked prefix on the fallback delta
        # path) — EOS is present but is not the literal last trained token.
        self.trailing_trained = trailing_trained

    def _render(self, messages):
        if self.fail_render:
            raise ValueError("fake render failure")
        if self.reject_system and any(m.get("role") == "system" for m in messages):
            raise ValueError("System role not supported")
        ids: list[int] = []
        mask: list[int] = []
        for _ in range(2 if self.double_bos else 1):
            ids.append(self.bos_token_id)
            mask.append(0)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            is_assistant = role == "assistant"
            for c in f"<{role}>:":
                ids.append((ord(c) % 200) + 1)
                mask.append(0)
            for c in str(content):
                ids.append((ord(c) % 200) + 1)
                mask.append(1 if is_assistant else 0)
            if is_assistant and self.emit_eos and self.eos_token_id is not None:
                # A real tokenizer's chat template emits exactly ONE
                # concrete eos id even when eos_token_id is a multi-entry
                # list/tuple (e.g. Llama-3's [128001, 128009]) — pick the
                # first entry so this fake's rendered ids stay flat ints.
                eos_value = self.eos_token_id
                if isinstance(eos_value, (list, tuple)):
                    eos_value = eos_value[0]
                ids.append(eos_value)
                mask.append(1)
            ids.append((ord("\n") % 200) + 1)
            mask.append(1 if (is_assistant and self.trailing_trained) else 0)
        return ids, mask

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_assistant_tokens_mask=False,
        return_dict=False,
        **kwargs,
    ):
        ids, mask = self._render(messages)
        if not tokenize:
            return "".join(f"[{i}]" for i in ids)
        if return_assistant_tokens_mask and return_dict:
            if not self.supports_assistant_mask:
                raise TypeError("return_assistant_tokens_mask unsupported")
            return {"input_ids": ids, "assistant_masks": mask}
        if return_dict:
            return {"input_ids": ids}
        return ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids = [(ord(c) % 200) + 1 for c in str(text)]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return out

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 200) + 1 for c in str(text)]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(f"[{i}]" for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


class _JinjaLikeError(RuntimeError):
    """Stand-in for ``jinja2.exceptions.TemplateError`` (neither ValueError
    nor TypeError — a real Mistral-style template's ``raise_exception()``
    raises this shape, not one of the two the checks used to catch)."""


class _OddRaisingTokenizer(_FakeTokenizer):
    """Raises a non-ValueError/TypeError from ``apply_chat_template`` for
    any row containing a system message — mirrors the real
    jinja2.exceptions.TemplateError a Mistral-style no-system-role template
    raises (v0.71.27 real-SmolLM2 smoke finding: this must be a per-row
    skip, not an unhandled crash of the whole report)."""

    def apply_chat_template(self, messages, **kwargs):
        if any(m.get("role") == "system" for m in messages):
            raise _JinjaLikeError("System role not supported")
        return super().apply_chat_template(messages, **kwargs)


class _TypeErrorOnKwargTokenizer(_FakeTokenizer):
    """``.encode`` has no ``add_special_tokens`` parameter at all (an older
    tokenizer) — calling it WITH that kwarg raises a genuine Python
    TypeError, calling it WITHOUT (positional-only) succeeds. Exercises the
    `soup data lint --model` length_fn's inner TypeError fallback."""

    def encode(self, text):  # noqa: D102 — deliberately no add_special_tokens param
        return [(ord(c) % 200) + 1 for c in str(text)]


class _AlwaysFailEncodeTokenizer(_FakeTokenizer):
    """``.encode(...)`` always raises, regardless of args — exercises the
    length_fn word-count degrade-gracefully fallback."""

    def encode(self, text, add_special_tokens=False):
        raise RuntimeError("encode is broken")


class _PerRowEosTokenizer(_FakeTokenizer):
    """EOS emission varies PER ROW (via a "NOEOS" marker in the assistant
    content) rather than being all-or-nothing for the whole tokenizer
    instance — needed to construct an EXACT boundary fraction (e.g. 2/4 =
    50%) for a threshold check's cutoff, not just "all rows" or "no rows".
    """

    def _render(self, messages):
        skip = any(m.get("content") == "NOEOS" for m in messages if m.get("role") == "assistant")
        original = self.emit_eos
        self.emit_eos = not skip
        try:
            return super()._render(messages)
        finally:
            self.emit_eos = original


def _chat_row(*turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# data_doctor.py — dataclasses + taxonomy
# ---------------------------------------------------------------------------


class TestDoctorCheck:
    def test_valid_check(self):
        from soup_cli.utils.data_doctor import DoctorCheck

        c = DoctorCheck(name="chat_template", verdict="OK", message="fine", evidence="")
        assert c.verdict == "OK"

    def test_rejects_unknown_name(self):
        from soup_cli.utils.data_doctor import DoctorCheck

        with pytest.raises(ValueError, match="unknown"):
            DoctorCheck(name="not_a_real_check", verdict="OK", message="x")

    def test_rejects_unknown_verdict(self):
        from soup_cli.utils.data_doctor import DoctorCheck

        with pytest.raises(ValueError, match="verdict"):
            DoctorCheck(name="chat_template", verdict="BAD", message="x")

    def test_rejects_null_byte(self):
        from soup_cli.utils.data_doctor import DoctorCheck

        with pytest.raises(ValueError, match="null byte"):
            DoctorCheck(name="chat_template", verdict="OK", message="a\x00b")

    def test_rejects_oversize_message(self):
        from soup_cli.utils.data_doctor import DoctorCheck

        with pytest.raises(ValueError, match="too long"):
            DoctorCheck(name="chat_template", verdict="OK", message="x" * 3000)

    def test_is_frozen(self):
        from soup_cli.utils.data_doctor import DoctorCheck

        c = DoctorCheck(name="chat_template", verdict="OK", message="x")
        with pytest.raises(Exception):  # noqa: PT011 — dataclasses.FrozenInstanceError
            c.verdict = "MAJOR"


class TestOverallVerdict:
    def test_empty_is_ok(self):
        from soup_cli.utils.data_doctor import overall_verdict

        assert overall_verdict([]) == "OK"

    def test_worst_wins(self):
        from soup_cli.utils.data_doctor import DoctorCheck, overall_verdict

        checks = [
            DoctorCheck(name="chat_template", verdict="OK", message="x"),
            DoctorCheck(name="bos_duplication", verdict="MINOR", message="x"),
            DoctorCheck(name="eos_in_labels", verdict="MAJOR", message="x"),
        ]
        assert overall_verdict(checks) == "MAJOR"

    def test_rejects_non_check_entries(self):
        from soup_cli.utils.data_doctor import overall_verdict

        with pytest.raises(TypeError):
            overall_verdict(["not a check"])


class TestDoctorReport:
    def test_to_dict_roundtrip_shape(self):
        from soup_cli.utils.data_doctor import DoctorCheck, compose_doctor_report

        checks = [DoctorCheck(name="chat_template", verdict="OK", message="fine", evidence="e")]
        report = compose_doctor_report(checks, rows_scanned=5, total_rows=10)
        d = report.to_dict()
        assert d["overall"] == "OK"
        assert d["rows_scanned"] == 5
        assert d["total_rows"] == 10
        assert d["checks"][0]["name"] == "chat_template"

    def test_rows_scanned_cannot_exceed_total(self):
        from soup_cli.utils.data_doctor import DoctorReport

        with pytest.raises(ValueError, match="cannot exceed"):
            DoctorReport(checks=(), overall="OK", rows_scanned=10, total_rows=5)

    def test_negative_counts_rejected(self):
        from soup_cli.utils.data_doctor import DoctorReport

        with pytest.raises(ValueError):
            DoctorReport(checks=(), overall="OK", rows_scanned=-1, total_rows=5)


class TestSampleIndices:
    def test_empty(self):
        from soup_cli.utils.data_doctor import sample_indices

        assert sample_indices(0, 10) == []

    def test_n_greater_than_total_returns_all(self):
        from soup_cli.utils.data_doctor import sample_indices

        assert sample_indices(3, 100) == [0, 1, 2]

    def test_spans_full_range(self):
        from soup_cli.utils.data_doctor import sample_indices

        idxs = sample_indices(1000, 10)
        assert len(idxs) <= 10
        assert idxs == sorted(set(idxs))
        assert idxs[0] < 100  # not just the tail
        assert idxs[-1] > 800  # not just the head — evenly spaced

    def test_exact_indices_evenly_divisible(self):
        """Regression (tdd-review): sample_indices is a pure, deterministic
        function that gates row sampling for all 13 checks — property-only
        assertions (count/sorted/range) would still pass under a changed
        step formula (e.g. round() instead of int(), or // instead of /)
        that silently samples different rows. Assert the exact list."""
        from soup_cli.utils.data_doctor import sample_indices

        assert sample_indices(1000, 10) == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        assert sample_indices(10, 3) == [0, 3, 6]

    def test_exact_indices_non_round_ratio(self):
        """int()-truncation on a non-round total/n ratio can skip an index
        (7/5=1.4 -> 0,1,2,4,5 — index 3 is never produced); this is an
        accepted property of the even-spacing approach, not a bug, but it
        must not silently change."""
        from soup_cli.utils.data_doctor import sample_indices

        assert sample_indices(7, 5) == [0, 1, 2, 4, 5]


# ---------------------------------------------------------------------------
# resolve_tokenizer
# ---------------------------------------------------------------------------


class TestResolveTokenizer:
    def test_passthrough_object(self):
        from soup_cli.utils.data_doctor import resolve_tokenizer

        tok = _FakeTokenizer()
        assert resolve_tokenizer(tok) is tok

    def test_rejects_non_string_non_tokenizer(self):
        from soup_cli.utils.data_doctor import resolve_tokenizer

        with pytest.raises(TypeError):
            resolve_tokenizer(12345)

    def test_rejects_empty_string(self):
        from soup_cli.utils.data_doctor import resolve_tokenizer

        with pytest.raises(ValueError, match="non-empty"):
            resolve_tokenizer("")

    def test_delegates_to_real_auto_tokenizer_from_pretrained(self, monkeypatch):
        """Regression (tdd-review): every other test either passes a
        duck-typed fake (hits the early passthrough) or monkeypatches
        resolve_tokenizer away entirely — the actual
        transformers.AutoTokenizer.from_pretrained delegation was never
        exercised, string id in, trust_remote_code kwarg forwarded, and all.
        """
        import transformers

        captured = {}

        def fake_from_pretrained(model_id, trust_remote_code=False):
            captured["model_id"] = model_id
            captured["trust_remote_code"] = trust_remote_code
            return _FakeTokenizer()

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
        )
        from soup_cli.utils.data_doctor import resolve_tokenizer

        result = resolve_tokenizer("some-org/some-model", trust_remote_code=True)
        assert isinstance(result, _FakeTokenizer)
        assert captured == {"model_id": "some-org/some-model", "trust_remote_code": True}

    def test_wraps_a_real_load_failure_with_model_id_and_exception_type(self, monkeypatch):
        import transformers

        def fake_from_pretrained(model_id, trust_remote_code=False):
            raise OSError("model not found on the Hub")

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
        )
        from soup_cli.utils.data_doctor import resolve_tokenizer

        with pytest.raises(ValueError, match="nonexistent/model") as excinfo:
            resolve_tokenizer("nonexistent/model")
        assert "OSError" in str(excinfo.value)

    def test_missing_transformers_dependency_is_a_friendly_error(self, monkeypatch):
        """Mirrors TestCheckNearDuplicatesNoDep's ImportError-simulation
        pattern for the datasketch optional dep."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from soup_cli.utils.data_doctor import resolve_tokenizer

        with pytest.raises(ValueError, match="transformers"):
            resolve_tokenizer("some-org/some-model")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestCheckChatTemplate:
    def test_present(self):
        from soup_cli.utils.data_doctor import check_chat_template

        c = check_chat_template(_FakeTokenizer(has_chat_template=True))
        assert c.verdict == "OK"

    def test_missing_is_major(self):
        from soup_cli.utils.data_doctor import check_chat_template

        c = check_chat_template(_FakeTokenizer(has_chat_template=False))
        assert c.verdict == "MAJOR"


class TestCheckTemplateRender:
    def test_all_render_ok(self):
        from soup_cli.utils.data_doctor import check_template_render

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))]
        c = check_template_render(_FakeTokenizer(), rows)
        assert c.verdict == "OK"

    def test_render_failures_flagged(self):
        from soup_cli.utils.data_doctor import check_template_render

        rows = [_chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello"))] * 5
        c = check_template_render(_FakeTokenizer(reject_system=True), rows)
        assert c.verdict == "MAJOR"  # 5/5 fail -> frac=1.0 >= the 10% MAJOR cutoff
        assert "5/5" in c.message

    def test_no_template_skips(self):
        from soup_cli.utils.data_doctor import check_template_render

        tok = _FakeTokenizer(has_chat_template=False)
        c = check_template_render(tok, [_chat_row(("user", "hi"))])
        assert c.verdict == "OK"
        assert "skipped" in c.message


class TestCheckGenerationMarkers:
    def test_supported(self):
        from soup_cli.utils.data_doctor import check_generation_markers

        c = check_generation_markers(_FakeTokenizer(supports_assistant_mask=True))
        assert c.verdict == "OK"

    def test_unsupported_is_minor(self):
        from soup_cli.utils.data_doctor import check_generation_markers

        c = check_generation_markers(_FakeTokenizer(supports_assistant_mask=False))
        assert c.verdict == "MINOR"
        assert "generation" in c.message.lower() or "heuristic" in c.message.lower()


class TestEosTokenIds:
    """Regression (tdd-review): `_eos_token_ids`'s list/tuple normalisation
    exists specifically for real multi-EOS models (Llama-3's
    eos_token_id=[128001, 128009]) but was never exercised — every test
    tokenizer used a bare int or None."""

    def test_list_eos_token_id_normalised(self):
        from soup_cli.utils.data_doctor import _eos_token_ids

        tok = _FakeTokenizer(eos_token_id=[128001, 128009])
        assert _eos_token_ids(tok) == {128001, 128009}

    def test_tuple_eos_token_id_normalised(self):
        from soup_cli.utils.data_doctor import _eos_token_ids

        tok = _FakeTokenizer(eos_token_id=(2, 3))
        assert _eos_token_ids(tok) == {2, 3}

    def test_list_with_non_int_entries_filtered(self):
        from soup_cli.utils.data_doctor import _eos_token_ids

        tok = _FakeTokenizer(eos_token_id=[2, "not-an-id", None, True])
        assert _eos_token_ids(tok) == {2}

    def test_bool_eos_token_id_rejected(self):
        from soup_cli.utils.data_doctor import _eos_token_ids

        tok = _FakeTokenizer(eos_token_id=True)
        assert _eos_token_ids(tok) == set()

    def test_multi_eos_list_actually_used_by_the_check(self):
        """Not just _eos_token_ids in isolation — a real multi-EOS model's
        list must actually satisfy check_eos_in_labels's span search."""
        from soup_cli.utils.data_doctor import check_eos_in_labels

        tok = _FakeTokenizer(emit_eos=True, eos_token_id=[999, _EOS_ID])
        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 4
        c = check_eos_in_labels(tok, rows, max_length=2048)
        assert c.verdict == "OK"


class TestCheckEosInLabels:
    """The #1 'model never stops generating' bug."""

    def test_eos_present_is_ok(self):
        from soup_cli.utils.data_doctor import check_eos_in_labels

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 4
        c = check_eos_in_labels(_FakeTokenizer(emit_eos=True), rows, max_length=2048)
        assert c.verdict == "OK"

    def test_eos_missing_is_flagged(self):
        from soup_cli.utils.data_doctor import check_eos_in_labels

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 4
        c = check_eos_in_labels(_FakeTokenizer(emit_eos=False), rows, max_length=2048)
        assert c.verdict == "MAJOR"  # 4/4 missing -> frac=1.0 >= the 50% MAJOR cutoff
        assert "4/4" in c.message
        assert "stop" in c.message.lower()

    def test_mixed_missing_is_proportional(self):
        from soup_cli.utils.data_doctor import check_eos_in_labels

        bad = _FakeTokenizer(emit_eos=False)
        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 10
        # Every row is checked against a tokenizer that never emits EOS at
        # all, so every row should be flagged (proportional to the 4/4 case
        # above at a different N — proves the fraction, not just a fixed count).
        c = check_eos_in_labels(bad, rows, max_length=2048)
        assert "10/10" in c.message

    def test_exactly_at_the_major_boundary(self):
        """Regression (tdd-review): every prior fixture was 0% or 100%
        missing — none exercised the documented >=50% MAJOR cutoff at the
        exact boundary or just below it. An off-by-one (>= vs >) would not
        be caught by any test before this one."""
        from soup_cli.utils.data_doctor import check_eos_in_labels

        tok = _PerRowEosTokenizer(emit_eos=True)
        rows = [
            _chat_row(("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "NOEOS")),
            _chat_row(("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "NOEOS")),
        ]
        c = check_eos_in_labels(tok, rows, max_length=2048)
        assert "2/4" in c.message
        assert c.verdict == "MAJOR"  # frac=0.5 >= 0.5

    def test_just_below_the_major_boundary(self):
        from soup_cli.utils.data_doctor import check_eos_in_labels

        tok = _PerRowEosTokenizer(emit_eos=True)
        rows = [
            _chat_row(("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "NOEOS")),
        ]
        c = check_eos_in_labels(tok, rows, max_length=2048)
        assert "1/4" in c.message
        assert c.verdict == "MINOR"  # frac=0.25 < 0.5, but > 0

    def test_no_eos_token_id_is_advisory(self):
        from soup_cli.utils.data_doctor import check_eos_in_labels

        tok = _FakeTokenizer(eos_token_id=None)
        row = _chat_row(("user", "hi"), ("assistant", "hello"))
        c = check_eos_in_labels(tok, [row], max_length=2048)
        assert c.verdict == "MINOR"

    def test_no_template_skips(self):
        from soup_cli.utils.data_doctor import check_eos_in_labels

        c = check_eos_in_labels(_FakeTokenizer(has_chat_template=False), [], max_length=2048)
        assert c.verdict == "OK"

    def test_eos_present_but_not_literal_last_token_is_ok(self):
        """Regression (found via real SmolLM2-135M-Instruct smoke test,
        v0.71.27): ChatML-style templates render `<|im_end|>\\n` for the
        last assistant turn, and on the fallback delta-masking path the
        trailing `\\n` has no later message to fold into, so it stays
        INSIDE the trained span too — EOS is present but is not the
        literal last trained id. Must not be flagged as missing.
        """
        from soup_cli.utils.data_doctor import check_eos_in_labels

        tok = _FakeTokenizer(emit_eos=True, trailing_trained=True)
        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 4
        c = check_eos_in_labels(tok, rows, max_length=2048)
        assert c.verdict == "OK", c.message

    def test_non_valueerror_render_failure_is_skipped_not_raised(self):
        """Regression (v0.71.27 real-SmolLM2 smoke test): a row whose
        template rejects it with a non-ValueError/TypeError exception (a
        real jinja2.TemplateError shape) must be skipped, not propagate
        and crash the whole doctor report.
        """
        from soup_cli.utils.data_doctor import check_eos_in_labels

        rows = [
            _chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "hello")),
        ]
        c = check_eos_in_labels(_OddRaisingTokenizer(), rows, max_length=2048)
        assert c.verdict == "OK"

    def test_eos_missing_on_an_earlier_turn_is_still_flagged(self):
        """Regression: checking only the LAST trained span would miss a
        template bug that drops EOS on an EARLIER assistant turn while the
        final turn still closes correctly — every turn boundary matters for
        multi-turn inference, not just the last."""
        from soup_cli.utils.data_doctor import check_eos_in_labels

        class _FirstTurnMissingEos(_FakeTokenizer):
            def _render(self, messages):
                ids, mask = [], []
                ids.append(self.bos_token_id)
                mask.append(0)
                assistant_seen = 0
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    is_assistant = role == "assistant"
                    for c in f"<{role}>:":
                        ids.append((ord(c) % 200) + 1)
                        mask.append(0)
                    for c in str(content):
                        ids.append((ord(c) % 200) + 1)
                        mask.append(1 if is_assistant else 0)
                    if is_assistant:
                        # Only the FIRST assistant turn skips EOS.
                        if assistant_seen > 0:
                            ids.append(self.eos_token_id)
                            mask.append(1)
                        assistant_seen += 1
                    ids.append((ord("\n") % 200) + 1)
                    mask.append(0)
                return ids, mask

        multi_turn_row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "second reply"},
            ]
        }
        c = check_eos_in_labels(_FirstTurnMissingEos(), [multi_turn_row], max_length=2048)
        assert c.verdict == "MAJOR", c.message  # 1/1 missing -> frac=1.0 >= the 50% cutoff
        assert "1/1" in c.message

    def test_truncated_row_excluded_not_flagged_as_missing_eos(self):
        """Regression (code-review MEDIUM): data.loss_mask._truncate keeps
        only the FIRST max_length tokens, which can cut off the trailing
        EOS a long row would otherwise have. That's a max_length problem
        (already surfaced by truncation_risk), not a template bug — a
        truncated row must not count against eos_in_labels at all."""
        from soup_cli.utils.data_doctor import check_eos_in_labels

        tok = _FakeTokenizer(emit_eos=True)
        long_row = _chat_row(("user", "hi"), ("assistant", "x" * 200))
        # A tiny max_length guarantees truncation lands well before the
        # trailing EOS this fake always appends after assistant content.
        c = check_eos_in_labels(tok, [long_row], max_length=10)
        assert c.verdict == "OK", c.message
        assert "no assistant turns to check" in c.message


class TestCheckBosDuplication:
    def test_single_bos_ok(self):
        from soup_cli.utils.data_doctor import check_bos_duplication

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 4
        c = check_bos_duplication(_FakeTokenizer(double_bos=False), rows, max_length=2048)
        assert c.verdict == "OK"

    def test_double_bos_flagged(self):
        from soup_cli.utils.data_doctor import check_bos_duplication

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 4
        c = check_bos_duplication(_FakeTokenizer(double_bos=True), rows, max_length=2048)
        assert c.verdict == "MAJOR"  # 4/4 duplicated -> frac=1.0 >= the 50% MAJOR cutoff
        assert "4/4" in c.message

    def test_no_bos_token_id_skips(self):
        from soup_cli.utils.data_doctor import check_bos_duplication

        class _NoBos(_FakeTokenizer):
            bos_token_id = None

        row = _chat_row(("user", "hi"), ("assistant", "hi"))
        c = check_bos_duplication(_NoBos(), [row], max_length=2048)
        assert c.verdict == "OK"

    def test_non_valueerror_render_failure_is_skipped_not_raised(self):
        """Regression (v0.71.27 real-SmolLM2 smoke test) — see the matching
        eos_in_labels test for the full rationale."""
        from soup_cli.utils.data_doctor import check_bos_duplication

        rows = [
            _chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "hello")),
        ]
        c = check_bos_duplication(_OddRaisingTokenizer(), rows, max_length=2048)
        assert c.verdict == "OK"


class TestCheckSystemRole:
    def test_no_system_rows_is_ok(self):
        from soup_cli.utils.data_doctor import check_system_role

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))]
        c = check_system_role(_FakeTokenizer(reject_system=True), rows)
        assert c.verdict == "OK"

    def test_system_supported(self):
        from soup_cli.utils.data_doctor import check_system_role

        rows = [_chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello"))]
        c = check_system_role(_FakeTokenizer(reject_system=False), rows)
        assert c.verdict == "OK"

    def test_system_unsupported_is_major(self):
        from soup_cli.utils.data_doctor import check_system_role

        rows = [_chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello"))] * 3
        c = check_system_role(_FakeTokenizer(reject_system=True), rows)
        assert c.verdict == "MAJOR"
        assert "3 rows" in c.message


class TestCheckUnknownRoles:
    def test_known_roles_ok(self):
        from soup_cli.utils.data_doctor import check_unknown_roles

        rows = [_chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello"))]
        c = check_unknown_roles(rows)
        assert c.verdict == "OK"

    def test_unknown_role_flagged(self):
        from soup_cli.utils.data_doctor import check_unknown_roles

        rows = [_chat_row(("human", "hi"), ("assistant", "hello"))] * 3
        c = check_unknown_roles(rows)
        assert c.verdict == "MAJOR"  # 3/3 unknown -> frac=1.0 >= the 10% MAJOR cutoff
        assert "human" in c.evidence


class TestCheckTruncationRisk:
    def test_short_rows_ok(self):
        from soup_cli.utils.data_doctor import check_truncation_risk

        rows = [_chat_row(("user", "hi"), ("assistant", "ok"))] * 5
        c = check_truncation_risk(_FakeTokenizer(), rows, max_length=2048)
        assert c.verdict == "OK"

    def test_long_rows_flagged(self):
        from soup_cli.utils.data_doctor import check_truncation_risk

        long_content = "x" * 500
        rows = [_chat_row(("user", long_content), ("assistant", long_content))] * 10
        c = check_truncation_risk(_FakeTokenizer(), rows, max_length=32)
        assert c.verdict == "MAJOR"  # 10/10 over -> frac_over=1.0 >= the 20% MAJOR cutoff
        assert "32" in c.message

    def test_p95_within_bound_is_ok_despite_a_truncated_outlier(self):
        """Regression (tdd-review): this is a COMPOUND gate, not a plain
        fraction — `if p95 <= max_length: OK` short-circuits regardless of
        frac_over. 19 short rows + 1 huge outlier keeps p95 within bound
        even though that one row (5%) would individually be truncated;
        empirically verified p95=52 for this exact fixture."""
        from soup_cli.utils.data_doctor import check_truncation_risk

        rows = [_chat_row(("user", "hi"), ("assistant", "ok"))] * 19 + [
            _chat_row(("user", "hi"), ("assistant", "x" * 500))
        ]
        c = check_truncation_risk(_FakeTokenizer(), rows, max_length=52)
        assert c.verdict == "OK", c.message  # p95(52) <= max_length(52)
        assert "1/20" in c.message  # the outlier is still named, just not gating

    def test_p95_just_over_bound_flips_to_minor(self):
        from soup_cli.utils.data_doctor import check_truncation_risk

        rows = [_chat_row(("user", "hi"), ("assistant", "ok"))] * 19 + [
            _chat_row(("user", "hi"), ("assistant", "x" * 500))
        ]
        c = check_truncation_risk(_FakeTokenizer(), rows, max_length=51)
        assert c.verdict == "MINOR", c.message  # p95(52) > max_length(51), frac_over 5% < 20%


# ---------------------------------------------------------------------------
# run_doctor — end to end
# ---------------------------------------------------------------------------


class TestRunDoctor:
    def test_happy_path(self):
        from soup_cli.utils.data_doctor import run_doctor

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 5
        report = run_doctor(rows, _FakeTokenizer(), fmt="chatml", max_length=2048)
        assert report.overall == "OK"
        assert report.rows_scanned == 5
        assert report.total_rows == 5
        names = {c.name for c in report.checks}
        assert {"chat_template", "eos_in_labels", "bos_duplication", "truncation_risk"} <= names

    def test_missing_eos_yields_major_overall(self):
        from soup_cli.utils.data_doctor import run_doctor

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 5
        report = run_doctor(rows, _FakeTokenizer(emit_eos=False), fmt="chatml", max_length=2048)
        assert report.overall == "MAJOR"

    def test_minor_only_check_yields_minor_overall(self):
        """Regression (tdd-review): every existing 'bad tokenizer' fixture
        happens to trip a MAJOR check, so no test proves overall correctly
        surfaces MINOR when that's genuinely the worst verdict present."""
        from soup_cli.utils.data_doctor import run_doctor

        tok = _FakeTokenizer(supports_assistant_mask=False)  # only generation_markers fires
        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 5
        report = run_doctor(rows, tok, fmt="chatml", max_length=2048)
        names_to_verdicts = {c.name: c.verdict for c in report.checks}
        assert names_to_verdicts["generation_markers"] == "MINOR"
        others = {k: v for k, v in names_to_verdicts.items() if k != "generation_markers"}
        assert all(v == "OK" for v in others.values())
        assert report.overall == "MINOR"

    def test_worst_wins_against_a_real_competing_minor(self):
        """A single-bad-check test can't distinguish 'worst wins' from
        'first non-OK wins' — this fixture makes generation_markers MINOR
        and eos_in_labels MAJOR simultaneously, so only a correct rank
        table produces MAJOR overall."""
        from soup_cli.utils.data_doctor import run_doctor

        tok = _FakeTokenizer(supports_assistant_mask=False, emit_eos=False)
        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 5
        report = run_doctor(rows, tok, fmt="chatml", max_length=2048)
        names_to_verdicts = {c.name: c.verdict for c in report.checks}
        assert names_to_verdicts["generation_markers"] == "MINOR"
        assert names_to_verdicts["eos_in_labels"] == "MAJOR"
        assert report.overall == "MAJOR"

    def test_zero_convertible_rows_raises(self):
        from soup_cli.utils.data_doctor import run_doctor

        rows = [{"prompt": "x", "chosen": "y", "rejected": "z"}] * 3
        with pytest.raises(ValueError, match="soup data lint"):
            run_doctor(rows, _FakeTokenizer(), fmt="dpo", max_length=2048)

    def test_empty_dataset_does_not_raise(self):
        from soup_cli.utils.data_doctor import run_doctor

        report = run_doctor([], _FakeTokenizer(), fmt="chatml", max_length=2048)
        assert report.total_rows == 0

    def test_sample_size_bounds(self):
        from soup_cli.utils.data_doctor import run_doctor

        rows = [_chat_row(("user", "hi"), ("assistant", "ho"))] * 3
        with pytest.raises(ValueError):
            run_doctor(rows, _FakeTokenizer(), fmt="chatml", sample_size=0)
        with pytest.raises(ValueError):
            run_doctor(rows, _FakeTokenizer(), fmt="chatml", sample_size=10**9)

    def test_max_length_bounds(self):
        from soup_cli.utils.data_doctor import run_doctor

        rows = [_chat_row(("user", "hi"), ("assistant", "ho"))]
        with pytest.raises(ValueError):
            run_doctor(rows, _FakeTokenizer(), fmt="chatml", max_length=0)

    def test_masking_strategy_flags_are_actually_threaded_into_checks(self):
        """Regression (code-review MEDIUM): eos_in_labels/bos_duplication
        must use the SAME masking strategy as --show-mask, not hard-code
        assistant-only — verified by making the two strategies disagree.

        build_assistant_only_labels always trains role=="assistant"
        regardless of any 'train' key; build_per_message_train_labels reads
        'train' directly. Marking the assistant turn train=False and the
        user turn train=True flips which span gets checked — and since this
        fake only ever appends EOS after an assistant turn, that flip must
        turn an OK verdict into a flagged one.
        """
        from soup_cli.utils.data_doctor import run_doctor

        rows = [
            {
                "messages": [
                    {"role": "user", "content": "hi", "train": True},
                    {"role": "assistant", "content": "hello", "train": False},
                ]
            }
        ]
        tok = _FakeTokenizer(emit_eos=True)

        default_report = run_doctor(rows, tok, fmt="chatml", max_length=2048)
        default_eos = next(c for c in default_report.checks if c.name == "eos_in_labels")
        assert default_eos.verdict == "OK", default_eos.message

        train_field_report = run_doctor(
            rows, tok, fmt="chatml", max_length=2048,
            train_on_messages_with_train_field=True,
        )
        train_field_eos = next(c for c in train_field_report.checks if c.name == "eos_in_labels")
        # 1/1 missing -> frac=1.0 >= the 50% MAJOR cutoff
        assert train_field_eos.verdict == "MAJOR", train_field_eos.message

    def test_masking_strategy_flag_type_validation(self):
        from soup_cli.utils.data_doctor import run_doctor

        rows = [_chat_row(("user", "hi"), ("assistant", "ho"))]
        with pytest.raises(TypeError):
            run_doctor(rows, _FakeTokenizer(), fmt="chatml", train_on_responses_only="yes")
        with pytest.raises(TypeError):
            run_doctor(
                rows, _FakeTokenizer(), fmt="chatml", train_on_messages_with_train_field=1
            )


# ---------------------------------------------------------------------------
# --show-mask: MaskedToken / MaskPreviewRow / render_mask_preview
# ---------------------------------------------------------------------------


class TestMaskedToken:
    def test_valid(self):
        from soup_cli.utils.data_doctor import MaskedToken

        t = MaskedToken(text="hi", trained=True)
        assert t.weight == 1.0

    def test_rejects_negative_weight(self):
        from soup_cli.utils.data_doctor import MaskedToken

        with pytest.raises(ValueError):
            MaskedToken(text="hi", trained=True, weight=-1.0)


class TestRenderMaskPreview:
    def test_assistant_only_strategy(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))]
        previews = render_mask_preview(rows, _FakeTokenizer(), fmt="chatml", n=1)
        assert len(previews) == 1
        assert previews[0].strategy == "assistant_only"
        assert any(t.trained for t in previews[0].tokens)
        assert any(not t.trained for t in previews[0].tokens)

    def test_per_message_train_field_strategy(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))]
        previews = render_mask_preview(
            rows, _FakeTokenizer(), fmt="chatml", n=1, train_on_messages_with_train_field=True,
        )
        assert previews[0].strategy == "per_message_train"

    def test_legacy_text_strategy_trains_everything(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))]
        previews = render_mask_preview(
            rows, _FakeTokenizer(), fmt="chatml", n=1, train_on_responses_only=False,
        )
        assert previews[0].strategy == "legacy_text"
        assert all(t.trained for t in previews[0].tokens)

    def test_legacy_text_strategy_truncates_to_max_length(self):
        """Regression (code-review LOW): legacy_text must truncate like the
        other two strategies (both delegate to data.loss_mask._truncate) —
        otherwise --show-mask could render more tokens than the trainer
        would actually see."""
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [_chat_row(("user", "hi"), ("assistant", "x" * 200))]
        previews = render_mask_preview(
            rows, _FakeTokenizer(), fmt="chatml", n=1,
            train_on_responses_only=False, max_length=10,
        )
        assert len(previews) == 1
        assert len(previews[0].tokens) == 10

    def test_raft_strategy_uses_loss_weights(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [
            {
                "query": "What is the capital of France?",
                "golden_doc": "Paris is the capital of France.",
                "distractor_docs": ["Berlin is the capital of Germany."],
                "answer": "Paris",
            }
        ]
        previews = render_mask_preview(rows, _FakeTokenizer(), fmt="raft", n=1)
        assert len(previews) == 1
        assert previews[0].strategy == "raft"
        assert any(t.trained for t in previews[0].tokens)
        assert any(not t.trained for t in previews[0].tokens)

    def test_stops_at_n(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))] * 10
        previews = render_mask_preview(rows, _FakeTokenizer(), fmt="chatml", n=3)
        assert len(previews) == 3

    def test_skips_unconvertible_rows(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [{"garbage": True}, _chat_row(("user", "hi"), ("assistant", "hello"))]
        previews = render_mask_preview(rows, _FakeTokenizer(), fmt="chatml", n=5)
        assert len(previews) == 1

    def test_row_index_reflects_real_source_position_not_display_order(self):
        """Regression (code-review LOW): row_index must be the row's real
        index into raw_rows (what a user would find in their JSONL), not a
        0,1,2... display-order counter that drifts once an earlier row is
        skipped for failing to convert/render."""
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [
            {"garbage": True},  # index 0 — skipped
            {"garbage": True},  # index 1 — skipped
            _chat_row(("user", "hi"), ("assistant", "hello")),  # index 2 — survives
        ]
        previews = render_mask_preview(rows, _FakeTokenizer(), fmt="chatml", n=5)
        assert len(previews) == 1
        assert previews[0].row_index == 2

    def test_non_valueerror_render_failure_is_skipped_not_raised(self):
        """Regression (v0.71.27 real-SmolLM2 smoke test) — see the matching
        eos_in_labels test for the full rationale."""
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [
            _chat_row(("system", "s"), ("user", "hi"), ("assistant", "hello")),
            _chat_row(("user", "hi"), ("assistant", "hello")),
        ]
        previews = render_mask_preview(rows, _OddRaisingTokenizer(), fmt="chatml", n=5)
        assert len(previews) == 1

    def test_n_bounds(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        rows = [_chat_row(("user", "hi"), ("assistant", "hello"))]
        with pytest.raises(ValueError):
            render_mask_preview(rows, _FakeTokenizer(), fmt="chatml", n=0)
        with pytest.raises(ValueError):
            render_mask_preview(rows, _FakeTokenizer(), fmt="chatml", n=10_000)


# ---------------------------------------------------------------------------
# No top-level heavy imports (fast CLI convention)
# ---------------------------------------------------------------------------


class TestNoTopLevelHeavyImport:
    @pytest.mark.parametrize(
        "relpath",
        [
            "src/soup_cli/utils/data_doctor.py",
            "src/soup_cli/utils/data_lint.py",
            "src/soup_cli/commands/data_doctor.py",
        ],
    )
    def test_no_torch_or_transformers_at_module_level(self, relpath):
        source = (_PROJECT_ROOT / relpath).read_text(encoding="utf-8")
        # Column-0 (unindented) lines only — a lazy import inside a function
        # body is indented and must NOT trip this check.
        top_level_lines = [ln for ln in source.splitlines() if ln[:1] not in ("", " ", "\t")]
        for ln in top_level_lines:
            assert not ln.startswith("import torch"), relpath
            assert not ln.startswith("from torch"), relpath
            assert not ln.startswith("import transformers"), relpath
            assert not ln.startswith("from transformers"), relpath


# ===========================================================================
# data_lint.py
# ===========================================================================


class TestLintCheck:
    """Regression (tdd-review): the doctor/lint 'twins' had asymmetric
    coverage — DoctorCheck/DoctorReport had a dedicated test class,
    LintCheck/LintReport did not (0% coverage on their __post_init__
    validation per the coverage report)."""

    def test_valid_check(self):
        from soup_cli.utils.data_lint import LintCheck

        c = LintCheck(name="length_bias", verdict="OK", message="fine", evidence="")
        assert c.verdict == "OK"

    def test_rejects_unknown_name(self):
        from soup_cli.utils.data_lint import LintCheck

        with pytest.raises(ValueError, match="unknown"):
            LintCheck(name="not_a_real_check", verdict="OK", message="x")

    def test_rejects_unknown_verdict(self):
        from soup_cli.utils.data_lint import LintCheck

        with pytest.raises(ValueError, match="verdict"):
            LintCheck(name="length_bias", verdict="BAD", message="x")

    def test_rejects_null_byte(self):
        from soup_cli.utils.data_lint import LintCheck

        with pytest.raises(ValueError, match="null byte"):
            LintCheck(name="length_bias", verdict="OK", message="a\x00b")

    def test_rejects_oversize_message(self):
        from soup_cli.utils.data_lint import LintCheck

        with pytest.raises(ValueError, match="too long"):
            LintCheck(name="length_bias", verdict="OK", message="x" * 3000)

    def test_is_frozen(self):
        from soup_cli.utils.data_lint import LintCheck

        c = LintCheck(name="length_bias", verdict="OK", message="x")
        with pytest.raises(Exception):  # noqa: PT011 — dataclasses.FrozenInstanceError
            c.verdict = "MAJOR"


class TestLintReport:
    def test_to_dict_roundtrip_shape(self):
        from soup_cli.utils.data_lint import LintCheck, compose_lint_report

        checks = [LintCheck(name="length_bias", verdict="OK", message="fine", evidence="e")]
        report = compose_lint_report(checks, fmt="dpo", rows_scanned=5, total_rows=10)
        d = report.to_dict()
        assert d["fmt"] == "dpo"
        assert d["overall"] == "OK"
        assert d["rows_scanned"] == 5
        assert d["total_rows"] == 10
        assert d["checks"][0]["name"] == "length_bias"

    def test_rows_scanned_cannot_exceed_total(self):
        from soup_cli.utils.data_lint import LintReport

        with pytest.raises(ValueError, match="cannot exceed"):
            LintReport(checks=(), overall="OK", fmt="dpo", rows_scanned=10, total_rows=5)

    def test_negative_counts_rejected(self):
        from soup_cli.utils.data_lint import LintReport

        with pytest.raises(ValueError):
            LintReport(checks=(), overall="OK", fmt="dpo", rows_scanned=-1, total_rows=5)

    def test_rejects_unsupported_fmt(self):
        from soup_cli.utils.data_lint import LintReport

        with pytest.raises(ValueError, match="fmt"):
            LintReport(checks=(), overall="OK", fmt="chatml", rows_scanned=0, total_rows=0)


class TestExtractPrefText:
    def test_string_passthrough(self):
        from soup_cli.utils.data_lint import extract_pref_text

        assert extract_pref_text("hello") == "hello"

    def test_messages_list_joins_content(self):
        from soup_cli.utils.data_lint import extract_pref_text

        val = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "there"}]
        out = extract_pref_text(val)
        assert "hi" in out and "there" in out

    def test_none_becomes_empty(self):
        from soup_cli.utils.data_lint import extract_pref_text

        assert extract_pref_text(None) == ""


class TestCohensD:
    def test_identical_distributions_zero(self):
        from soup_cli.utils.data_lint import cohens_d

        assert cohens_d([10.0] * 20, [10.0] * 20) == 0.0

    def test_large_shift_large_effect(self):
        from soup_cli.utils.data_lint import cohens_d

        a = [100.0 + i for i in range(20)]
        b = [10.0 + i for i in range(20)]
        d = cohens_d(a, b)
        assert d > 0.8

    def test_direction_sign(self):
        from soup_cli.utils.data_lint import cohens_d

        a = [5.0] * 10
        b = [50.0] * 10
        assert cohens_d(a, b) < 0

    def test_empty_raises(self):
        from soup_cli.utils.data_lint import cohens_d

        with pytest.raises(ValueError):
            cohens_d([], [1.0])

    def test_single_sample_each_no_crash(self):
        from soup_cli.utils.data_lint import cohens_d

        assert cohens_d([5.0], [1.0]) > 0

    def test_asymmetric_degenerate_n_takes_sign_only_path(self):
        """Regression (tdd-review): the n<2 fallback is a single combined
        `n_a < 2 or n_b < 2` condition — only the SYMMETRIC (both n=1) case
        was tested. A larger, non-degenerate second sample must still take
        the sign-only path when the OTHER side is degenerate."""
        from soup_cli.utils.data_lint import cohens_d

        assert cohens_d([5.0], [1.0, 2.0, 9.0]) > 0
        assert cohens_d([1.0, 2.0, 9.0], [5.0]) < 0

    def test_degenerate_tie_is_exactly_zero(self):
        """Regression (tdd-review): the n<2 fallback's `mean_a == mean_b`
        True branch is only ever hit incidentally (inside a 1-row DPO
        fixture that never inspects length_bias) — assert it directly."""
        from soup_cli.utils.data_lint import cohens_d

        assert cohens_d([5.0], [5.0]) == 0.0


class TestCheckLengthBias:
    def test_balanced_lengths_ok(self):
        from soup_cli.utils.data_lint import check_length_bias

        rows = [{"chosen": "a" * 20, "rejected": "b" * 20} for _ in range(20)]
        c = check_length_bias(rows, length_fn=len)
        assert c.verdict == "OK"

    def test_chosen_systematically_longer_flagged(self):
        from soup_cli.utils.data_lint import check_length_bias

        rows = [{"chosen": "a" * 200, "rejected": "b" * 10} for _ in range(20)]
        c = check_length_bias(rows, length_fn=len)
        # Constant, zero-variance lengths -> cohens_d's sign-only fallback = 1.0 >= 0.8
        assert c.verdict == "MAJOR"
        assert "chosen" in c.message.lower()

    def test_just_at_or_above_the_major_boundary(self):
        """Regression (tdd-review): no test hit the documented |d| >= 0.8
        MAJOR cutoff near the exact boundary. Equal-n, equal-variance
        samples make Cohen's d hand-computable via mean_diff/pooled_std —
        note 0.8 itself has no exact binary float representation, so this
        targets a value confirmed (empirically, not just hand-derived) to
        land just above it, rather than an unreliable exact match. Uses a
        controlled length_fn (parses the row as a float) for exact numeric
        control instead of string length."""
        from soup_cli.utils.data_lint import check_length_bias

        # chosen: mean=10, var=4 (std=2); rejected: mean=8.36, var=4 (std=2)
        # equal n + equal variance -> pooled_std=2 -> d=1.64/2=0.82 (verified: 0.8200000000000004)
        rows = [
            {"chosen": "8.0", "rejected": "6.36"},
            {"chosen": "10.0", "rejected": "8.36"},
            {"chosen": "12.0", "rejected": "10.36"},
        ]
        c = check_length_bias(rows, length_fn=float)
        assert c.verdict == "MAJOR", c.message

    def test_just_below_the_major_boundary(self):
        from soup_cli.utils.data_lint import check_length_bias

        # Same shape, mean diff 1.6 -> d=1.6/2=0.8 mathematically, but
        # verified to resolve to 0.7999999999999998 in float64 (0.8 has no
        # exact binary representation) -> MINOR, not MAJOR. This IS the
        # near-boundary case, not a workaround for it.
        rows = [
            {"chosen": "8.0", "rejected": "6.4"},
            {"chosen": "10.0", "rejected": "8.4"},
            {"chosen": "12.0", "rejected": "10.4"},
        ]
        c = check_length_bias(rows, length_fn=float)
        assert c.verdict == "MINOR", c.message

    def test_empty_is_ok(self):
        from soup_cli.utils.data_lint import check_length_bias

        c = check_length_bias([], length_fn=len)
        assert c.verdict == "OK"


class TestCheckLabelImbalance:
    def test_balanced_ok(self):
        from soup_cli.utils.data_lint import check_label_imbalance

        rows = [{"label": True}] * 15 + [{"label": False}] * 15
        c = check_label_imbalance(rows)
        assert c.verdict == "OK"

    def test_severe_imbalance_flagged(self):
        from soup_cli.utils.data_lint import check_label_imbalance

        rows = [{"label": True}] * 99 + [{"label": False}] * 1
        c = check_label_imbalance(rows)
        assert c.verdict == "MAJOR"  # minority_frac=0.01 < the 5% MAJOR cutoff

    def test_empty_is_ok(self):
        from soup_cli.utils.data_lint import check_label_imbalance

        assert check_label_imbalance([]).verdict == "OK"


class TestCheckIdenticalPairs:
    def test_no_dupes_ok(self):
        from soup_cli.utils.data_lint import check_identical_pairs

        rows = [{"chosen": "a", "rejected": "b"} for _ in range(5)]
        c = check_identical_pairs(rows)
        assert c.verdict == "OK"

    def test_identical_pair_is_major(self):
        from soup_cli.utils.data_lint import check_identical_pairs

        rows = [{"chosen": "same text", "rejected": "same text"}] + [
            {"chosen": "a", "rejected": "b"} for _ in range(9)
        ]
        c = check_identical_pairs(rows)
        assert c.verdict == "MAJOR"
        assert "1/10" in c.message

    def test_messages_list_pairs_compared(self):
        from soup_cli.utils.data_lint import check_identical_pairs

        msgs = [{"role": "assistant", "content": "x"}]
        rows = [{"chosen": msgs, "rejected": list(msgs)}]
        c = check_identical_pairs(rows)
        assert c.verdict == "MAJOR"


class TestCheckPromptLeak:
    def test_no_leak_ok(self):
        from soup_cli.utils.data_lint import check_prompt_leak

        rows = [{"prompt": "x" * 60, "chosen": "totally unrelated", "rejected": "also unrelated"}]
        c = check_prompt_leak(rows, fmt="dpo")
        assert c.verdict == "OK"

    def test_leak_in_chosen_flagged(self):
        from soup_cli.utils.data_lint import check_prompt_leak

        prompt = "Please summarize the following article: " + ("lorem ipsum " * 5)
        rows = [{"prompt": prompt, "chosen": prompt + " Here is a summary.", "rejected": "n/a"}]
        c = check_prompt_leak(rows, fmt="dpo")
        assert c.verdict == "MAJOR"  # 1/1 flagged -> frac=1.0 >= the 10% MAJOR cutoff

    def test_kto_checks_completion(self):
        from soup_cli.utils.data_lint import check_prompt_leak

        prompt = "Please summarize the following article: " + ("lorem ipsum " * 5)
        rows = [{"prompt": prompt, "completion": prompt + " Summary.", "label": True}]
        c = check_prompt_leak(rows, fmt="kto")
        assert c.verdict == "MAJOR"  # 1/1 flagged -> frac=1.0 >= the 10% MAJOR cutoff

    def test_short_prompts_not_flagged(self):
        from soup_cli.utils.data_lint import check_prompt_leak

        rows = [{"prompt": "hi", "chosen": "hi there", "rejected": "no"}]
        c = check_prompt_leak(rows, fmt="dpo")
        assert c.verdict == "OK"


@pytest.mark.skipif(not _HAS_DATASKETCH, reason="datasketch not installed")
class TestCheckNearDuplicates:
    def test_no_dupes_ok(self):
        from soup_cli.utils.data_lint import check_near_duplicates

        rows = [
            {
                "prompt": f"unique question number {i} about topic {i}",
                "chosen": "x",
                "rejected": "y",
            }
            for i in range(10)
        ]
        c = check_near_duplicates(rows, key_fn=lambda r: r["prompt"])
        assert c.verdict == "OK"

    def test_duplicates_flagged(self):
        from soup_cli.utils.data_lint import check_near_duplicates

        base = "what is the capital of france and why is it important historically"
        rows = [{"prompt": base, "chosen": "x", "rejected": "y"} for _ in range(10)]
        c = check_near_duplicates(rows, key_fn=lambda r: r["prompt"])
        # 10 byte-identical prompts -> every row matches every other -> frac=1.0
        assert c.verdict == "MAJOR"


class TestCheckNearDuplicatesNoDep:
    def test_missing_dep_degrades_gracefully(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "datasketch":
                raise ImportError("no datasketch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from soup_cli.utils.data_lint import check_near_duplicates

        rows = [{"prompt": "x", "chosen": "a", "rejected": "b"}]
        c = check_near_duplicates(rows, key_fn=lambda r: r["prompt"])
        assert c.verdict == "OK"
        assert "skip" in c.message.lower()


class TestRunLint:
    def test_dpo_happy_path(self):
        from soup_cli.utils.data_lint import run_lint

        rows = [
            {"prompt": f"q{i}", "chosen": "answer text here", "rejected": "different answer"}
            for i in range(10)
        ]
        report = run_lint(rows, fmt="dpo")
        assert report.fmt == "dpo"
        names = {c.name for c in report.checks}
        assert "length_bias" in names
        assert "identical_pairs" in names
        assert "label_imbalance" not in names

    def test_kto_happy_path(self):
        from soup_cli.utils.data_lint import run_lint

        rows = [
            {"prompt": f"q{i}", "completion": "some completion", "label": i % 2 == 0}
            for i in range(10)
        ]
        report = run_lint(rows, fmt="kto")
        names = {c.name for c in report.checks}
        assert "label_imbalance" in names
        assert "length_bias" not in names

    def test_unsupported_format_raises(self):
        from soup_cli.utils.data_lint import run_lint

        with pytest.raises(ValueError, match="dpo|kto"):
            run_lint([{"messages": []}], fmt="chatml")

    def test_sample_size_bounds(self):
        """Regression (tdd-review): run_doctor has this bounds test;
        run_lint's identical validation was untested."""
        from soup_cli.utils.data_lint import run_lint

        rows = [{"prompt": "q", "chosen": "a", "rejected": "b"}] * 3
        with pytest.raises(ValueError):
            run_lint(rows, fmt="dpo", sample_size=0)
        with pytest.raises(ValueError):
            run_lint(rows, fmt="dpo", sample_size=10**9)

    def test_identical_pairs_overall_major(self):
        from soup_cli.utils.data_lint import run_lint

        rows = [{"prompt": "q", "chosen": "same", "rejected": "same"}]
        report = run_lint(rows, fmt="dpo")
        assert report.overall == "MAJOR"

    def test_worst_wins_against_a_real_competing_minor(self):
        """Regression (tdd-review): the only prior multi-row fixture puts
        every row in the SAME bucket, so it can't distinguish 'worst wins'
        from 'first non-OK wins'. Mix a MAJOR identical_pairs row (a binary
        any-occurrence rule) with a separate MINOR prompt_leak row (a
        fractional rule, <10% here) and confirm MAJOR — not MINOR — wins.
        """
        from soup_cli.utils.data_lint import run_lint

        clean_rows = [
            {
                "prompt": f"distinct question {i}",
                "chosen": f"distinct answer {i}",
                "rejected": f"other {i}",
            }
            for i in range(13)
        ]
        leak_prompt = "Please summarize the following unique article: " + ("lorem ipsum " * 5)
        leak_row = {"prompt": leak_prompt, "chosen": leak_prompt + " done.", "rejected": "n/a"}
        identical_row = {"prompt": "q-identical", "chosen": "same text", "rejected": "same text"}
        rows = clean_rows + [leak_row, identical_row]

        report = run_lint(rows, fmt="dpo")
        names_to_verdicts = {c.name: c.verdict for c in report.checks}
        assert names_to_verdicts["identical_pairs"] == "MAJOR"
        assert names_to_verdicts["prompt_leak"] == "MINOR", names_to_verdicts
        assert report.overall == "MAJOR"

    def test_auto_detects_format(self):
        from soup_cli.utils.data_lint import run_lint

        rows = [{"prompt": "q", "chosen": "a", "rejected": "b"}]
        report = run_lint(rows, fmt="auto")
        assert report.fmt == "dpo"

    def test_all_rows_unparseable_raises_not_silent_ok(self):
        """Regression (code-review HIGH, empirically reproduced): a dataset
        where every row is structurally broken (e.g. chosen/rejected
        corrupted to null by an upstream export bug) must not silently
        report a false "OK, 0 rows scanned" — mirrors run_doctor's
        `test_zero_convertible_rows_raises` guard."""
        from soup_cli.utils.data_lint import run_lint

        rows = [{"prompt": "q", "chosen": None, "rejected": None}] * 5
        with pytest.raises(ValueError, match="no rows converted"):
            run_lint(rows, fmt="dpo")

    def test_empty_dataset_does_not_raise_zero_rows_guard(self):
        """The zero-rows guard is for 'every row failed to convert', not
        'there were no rows to begin with' — an empty dataset must still
        hit the earlier, more specific error."""
        from soup_cli.utils.data_lint import run_lint

        with pytest.raises(ValueError, match="empty dataset"):
            run_lint([], fmt="auto")


# ===========================================================================
# CLI layer — soup data doctor / soup data lint
# ===========================================================================


def _fake_resolve_tokenizer_factory(tok):
    def _fake(model, *, trust_remote_code=False):
        return tok

    return _fake


def _patch_tokenizer(monkeypatch, tok=None):
    """Make ``commands/data_doctor.py``'s ``engine.resolve_tokenizer`` return ``tok``."""
    import soup_cli.utils.data_doctor as engine

    fake = _fake_resolve_tokenizer_factory(tok or _FakeTokenizer())
    monkeypatch.setattr(engine, "resolve_tokenizer", fake)


class TestForTerminal:
    def test_strips_esc_and_bell(self):
        from soup_cli.commands.data_doctor import _for_terminal

        assert _for_terminal("\x1b]0;PWNED\x07human") == "]0;PWNEDhuman"

    def test_preserves_tab_newline_cr(self):
        from soup_cli.commands.data_doctor import _for_terminal

        assert _for_terminal("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_strips_del(self):
        from soup_cli.commands.data_doctor import _for_terminal

        assert _for_terminal("a\x7fb") == "ab"

    def test_plain_text_unchanged(self):
        from soup_cli.commands.data_doctor import _for_terminal

        assert _for_terminal("nothing weird here") == "nothing weird here"


class TestLoadTokenizerUsesResolvedTrust:
    def test_resolved_trust_value_is_what_gets_passed_through(self, monkeypatch):
        """Regression (security-review LOW): _load_tokenizer must use
        resolve_trust_remote_code's RETURN value (mirrors chat.py/diff.py/
        merge.py/...), not silently re-use the raw CLI-supplied bool —
        a latent footgun if that function's contract ever transforms the
        flag rather than just gating/warning on it."""
        import soup_cli.commands.data_doctor as cli_mod

        captured = {}

        def fake_resolve_trust(model, requested, console, requires):
            return "SENTINEL"  # deliberately not the raw bool, to prove it's threaded through

        def fake_resolve_tokenizer(model, *, trust_remote_code):
            captured["trust_remote_code"] = trust_remote_code
            return _FakeTokenizer()

        monkeypatch.setattr(cli_mod, "resolve_trust_remote_code", fake_resolve_trust)
        monkeypatch.setattr(cli_mod.engine, "resolve_tokenizer", fake_resolve_tokenizer)

        cli_mod._load_tokenizer("fake/model", False)
        assert captured["trust_remote_code"] == "SENTINEL"


class TestDoctorCli:
    def test_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_happy_path_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))] * 5)

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", "fake/model"])
        assert result.exit_code == 0, result.output

    def test_trust_remote_code_shows_warning_panel(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app,
            ["data", "doctor", str(data_path), "--model", "some-org/model", "--trust-remote-code"],
        )
        assert result.exit_code == 0, result.output
        assert "remote code" in result.output.lower()

    def test_local_model_requiring_remote_code_is_refused_without_opt_in(
        self, tmp_path, monkeypatch
    ):
        """Regression (code-review MEDIUM): _load_tokenizer must probe
        model_requires_trust_remote_code like every other model-loading
        command (chat.py, diff.py, export.py, ...) — a local model dir with
        auto_map set must be refused with the standard gate message unless
        --trust-remote-code is passed, not silently attempted."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])

        model_dir = tmp_path / "custom-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"auto_map": {"AutoTokenizer": "custom.CustomTokenizer"}}),
            encoding="utf-8",
        )

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", str(model_dir)])
        assert result.exit_code == 1, result.output
        assert "trust" in result.output.lower() and "remote" in result.output.lower()

    def test_control_characters_in_dataset_content_stripped_from_terminal(
        self, tmp_path, monkeypatch
    ):
        """Regression (security-review MEDIUM): rich.markup.escape() only
        neutralises [...] tag syntax, not raw control bytes — an attacker-
        controlled 'role' field carrying a literal ESC byte (e.g. an OSC
        title-spoof / cursor-trick sequence) must never reach the terminal
        raw, even though it survives escape() untouched."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        evil_role = "\x1b]0;PWNED\x07human"
        _write_jsonl(
            data_path,
            [
                {
                    "messages": [
                        {"role": evil_role, "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                }
            ],
        )

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", "fake/model"])
        assert "\x1b" not in result.output
        assert "\x07" not in result.output

    def test_empty_model_value_is_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", ""])
        assert result.exit_code == 1, result.output
        assert not isinstance(result.exception, (KeyError, AttributeError))

    def test_missing_eos_exits_two(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))] * 5)

        _patch_tokenizer(monkeypatch, _FakeTokenizer(emit_eos=False))

        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", "fake/model"])
        assert result.exit_code == 2, result.output
        assert "stop" in result.output.lower() or "eos" in result.output.lower()

    def test_missing_file_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", "nope.jsonl", "--model", "fake/model"])
        assert result.exit_code == 1

    def test_dpo_format_routes_to_lint_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(data_path, [{"prompt": "q", "chosen": "a", "rejected": "b"}] * 3)

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", "fake/model"])
        assert result.exit_code == 2
        assert "lint" in result.output.lower()

    def test_raft_without_show_mask_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "raft.jsonl"
        _write_jsonl(
            data_path,
            [
                {
                    "query": "q",
                    "golden_doc": "the answer is here",
                    "distractor_docs": ["irrelevant"],
                    "answer": "here",
                }
            ]
            * 3,
        )

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "doctor", str(data_path), "--model", "fake/model", "--format", "raft"]
        )
        assert result.exit_code == 2
        assert "show-mask" in result.output.lower()

    def test_show_mask_renders_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))] * 3)

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "doctor", str(data_path), "--model", "fake/model", "--show-mask", "2"]
        )
        assert result.exit_code == 0, result.output

    def test_raft_show_mask_works_without_full_report(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "raft.jsonl"
        _write_jsonl(
            data_path,
            [
                {
                    "query": "q",
                    "golden_doc": "the answer is here",
                    "distractor_docs": ["irrelevant"],
                    "answer": "here",
                }
            ]
            * 3,
        )

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app,
            [
                "data", "doctor", str(data_path), "--model", "fake/model",
                "--format", "raft", "--show-mask", "1",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_output_json_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))] * 3)
        out_path = tmp_path / "report.json"

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app,
            ["data", "doctor", str(data_path), "--model", "fake/model", "--output", str(out_path)],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["overall"] == "OK"

    def test_output_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app,
            [
                "data",
                "doctor",
                str(data_path),
                "--model",
                "fake/model",
                "--output",
                "../outside.json",
            ],
        )
        assert result.exit_code == 1

    def test_empty_dataset_file_is_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "empty.jsonl"
        data_path.write_text("", encoding="utf-8")
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", "fake/model"])
        assert result.exit_code == 1, result.output
        assert "empty" in result.output.lower()

    def test_format_auto_detect_failure_is_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "undetectable.jsonl"
        _write_jsonl(data_path, [{"some_unknown_key": "value"}])
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "doctor", str(data_path), "--model", "fake/model"])
        assert result.exit_code == 1, result.output

    def test_show_mask_out_of_bounds_is_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "doctor", str(data_path), "--model", "fake/model", "--show-mask", "0"]
        )
        assert result.exit_code == 1, result.output

    def test_sample_zero_is_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "doctor", str(data_path), "--model", "fake/model", "--sample", "0"]
        )
        assert result.exit_code == 1, result.output

    def test_show_mask_no_renderable_rows_prints_warning(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "train.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])

        _patch_tokenizer(monkeypatch, _FakeTokenizer(has_chat_template=False))

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "doctor", str(data_path), "--model", "fake/model", "--show-mask", "1"]
        )
        # No chat_template -> check_chat_template is MAJOR (exit 2) AND
        # render_mask_preview finds nothing to render (the warning path).
        assert result.exit_code == 2, result.output
        assert "no rows could be rendered" in result.output.lower()


class TestLintCli:
    def test_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", "--help"])
        assert result.exit_code == 0

    def test_happy_path_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(
            data_path,
            [
                {
                    "prompt": f"q{i}",
                    "chosen": "a reasonable answer",
                    "rejected": "a different answer",
                }
                for i in range(10)
            ],
        )
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", str(data_path)])
        assert result.exit_code == 0, result.output

    def test_identical_pairs_exits_two(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        row = {"prompt": "q", "chosen": "same text here", "rejected": "same text here"}
        _write_jsonl(data_path, [row])
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", str(data_path)])
        assert result.exit_code == 2, result.output

    def test_output_json_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        rows = [{"prompt": f"q{i}", "chosen": "answer", "rejected": "other"} for i in range(5)]
        _write_jsonl(data_path, rows)
        out_path = tmp_path / "lint.json"
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", str(data_path), "--output", str(out_path)])
        assert result.exit_code == 0, result.output
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["fmt"] == "dpo"

    def test_unsupported_format_message(self, tmp_path, monkeypatch):
        """Regression (code-review MEDIUM): the 'wrong command for this
        format' routing case is a dedicated exit 2 (mirroring doctor's
        symmetric dpo/kto-routes-to-lint check), not a generic exit 1 —
        the module docstring documents this exact exit-code contract."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "chat.jsonl"
        _write_jsonl(data_path, [_chat_row(("user", "hi"), ("assistant", "hello"))])
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", str(data_path)])
        assert result.exit_code == 2, result.output
        assert "dpo" in result.output.lower() or "kto" in result.output.lower()

    def test_missing_file_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", "nope.jsonl"])
        assert result.exit_code == 1

    def test_empty_dataset_file_is_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "empty.jsonl"
        data_path.write_text("", encoding="utf-8")
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "lint", str(data_path)])
        assert result.exit_code == 1, result.output
        assert "empty" in result.output.lower()

    def test_output_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(data_path, [{"prompt": "q", "chosen": "a", "rejected": "b"}])
        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "lint", str(data_path), "--output", "../outside.json"]
        )
        assert result.exit_code == 1

    def test_model_flag_uses_tokenizer_for_length_bias(self, tmp_path, monkeypatch):
        """Exercises the --model / length_fn closure end to end (previously
        0% covered) — a real (fake) tokenizer must drive the length_bias
        effect size instead of the default word-count proxy."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(
            data_path,
            [
                {"prompt": f"q{i}", "chosen": "a" * 200, "rejected": "b"}
                for i in range(10)
            ],
        )

        _patch_tokenizer(monkeypatch)

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "lint", str(data_path), "--model", "fake/model"]
        )
        assert result.exit_code == 2, result.output
        assert "length_bias" in result.output.lower() or "cohen" in result.output.lower()

    def test_model_flag_not_loaded_for_kto(self, tmp_path, monkeypatch):
        """kto's checks never consult length_fn — --model must not even be
        resolved (mirrors doctor's format-before-tokenizer-load ordering)."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(
            data_path,
            [{"prompt": f"q{i}", "completion": "c", "label": i % 2 == 0} for i in range(10)],
        )

        import soup_cli.utils.data_doctor as engine

        def _boom(model, *, trust_remote_code=False):
            raise AssertionError("tokenizer should not be loaded for kto lint")

        monkeypatch.setattr(engine, "resolve_tokenizer", _boom)

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "lint", str(data_path), "--format", "kto", "--model", "fake/model"]
        )
        assert result.exit_code == 0, result.output

    def test_model_length_fn_type_error_fallback(self, tmp_path, monkeypatch):
        """length_fn's inner TypeError fallback (an older tokenizer whose
        .encode has no add_special_tokens parameter) must still succeed."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(
            data_path, [{"prompt": f"q{i}", "chosen": "aaaa", "rejected": "b"} for i in range(10)]
        )

        _patch_tokenizer(monkeypatch, _TypeErrorOnKwargTokenizer())

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "lint", str(data_path), "--model", "fake/model"]
        )
        # chosen="aaaa" (4 char-tokens) vs rejected="b" (1) via the TypeError
        # fallback's char-level tokenizer -> constant, zero-variance Cohen's
        # d=1.0 -> MAJOR.
        assert result.exit_code == 2, result.output

    def test_model_length_fn_encode_failure_degrades_to_word_count(self, tmp_path, monkeypatch):
        """length_fn's outer Exception fallback (a tokenizer whose .encode
        is completely broken) must degrade to word count, not crash."""
        monkeypatch.chdir(tmp_path)
        data_path = tmp_path / "pref.jsonl"
        _write_jsonl(
            data_path, [{"prompt": f"q{i}", "chosen": "aaaa", "rejected": "b"} for i in range(10)]
        )

        _patch_tokenizer(monkeypatch, _AlwaysFailEncodeTokenizer())

        from soup_cli.cli import app

        result = runner.invoke(
            app, ["data", "lint", str(data_path), "--model", "fake/model"]
        )
        # Both "aaaa" and "b" are 1 word -> word-count fallback gives d=0.0 -> OK.
        assert result.exit_code == 0, result.output
        assert "length_bias" in result.output.lower()


# ===========================================================================
# Housekeeping rider — diagnose.py O_NOFOLLOW evidence loader (v0.71.25
# known-limitation (4)); ships alongside the doctor/lint feature work.
# ===========================================================================


class TestDiagnoseEvidenceHardening:
    def test_source_uses_o_nofollow_and_fstat(self):
        source = (_PROJECT_ROOT / "src" / "soup_cli" / "commands" / "diagnose.py").read_text(
            encoding="utf-8"
        )
        assert "O_NOFOLLOW" in source
        assert "os.fstat(" in source

    def test_happy_path_still_loads(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = tmp_path / "ev.json"
        ev.write_text(json.dumps({"scores": {}}), encoding="utf-8")

        from soup_cli.commands.diagnose import _load_evidence

        payload = _load_evidence(str(ev))
        assert payload == {"scores": {}}

    def test_size_cap_still_enforced_via_fstat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = tmp_path / "ev.json"
        ev.write_text("{}", encoding="utf-8")

        from unittest.mock import MagicMock, patch

        from soup_cli.commands.diagnose import _load_evidence

        fake_stat = MagicMock(st_size=20 * 1024 * 1024)
        with patch("os.fstat", return_value=fake_stat):
            with pytest.raises(Exception, match="exceeds"):
                _load_evidence(str(ev))

    def test_not_a_dict_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = tmp_path / "ev.json"
        ev.write_text("[1, 2, 3]", encoding="utf-8")

        from soup_cli.commands.diagnose import _load_evidence

        with pytest.raises(Exception, match="JSON object"):
            _load_evidence(str(ev))

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        from soup_cli.commands.diagnose import _load_evidence

        with pytest.raises(Exception, match="unreadable"):
            _load_evidence(str(tmp_path / "missing.json"))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_symlink_still_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_path / "link.json"
        os.symlink(target, link)

        from soup_cli.commands.diagnose import _load_evidence

        with pytest.raises(ValueError, match="symlink"):
            _load_evidence(str(link))
