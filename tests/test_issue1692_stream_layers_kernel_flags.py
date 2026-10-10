"""Tests for issue #1692 - layer streaming accepts `use_liger` / `use_flash_attn`,
then ignores both.

A config with ``training.stream_layers: true`` plus either flag loaded, trained, and
said nothing about the switch being dropped, because both are read only by the
RESIDENT setup:

- ``use_liger`` patches the MLP forward in ``SFTTrainerWrapper._setup_transformers``
  (the Liger patch, in ``trainer/sft.py``);
- ``use_flash_attn`` is handed to ``from_pretrained`` as ``attn_implementation`` in the
  same method.

A streamed run goes to ``_setup_streaming_transformers`` (the streamed branch of
``SFTTrainerWrapper.setup``), whose model is built by
``AutoModelForCausalLM.from_config(...)`` in ``build_meta_skeleton``
(``utils/layer_stream_runtime.py``) with no attention argument and no Liger patch.
``_liger_applied`` is never set there, so ``use_liger_kernel`` misses
``TrainingArguments`` too.

Function names rather than line numbers: the readers move with every rebase, and a
stale ``:1588`` in a docstring is a small lie nobody checks. The scan tests below pin
the claim itself, so a reader moving is caught where it matters.

These are the switches a user reaches for when a streamed run will not fit, so dropping
them silently is the worst outcome: the refusal is the small step until each is wired
and shown to engage on a streamed model. ``use_cut_ce`` has the same shape and is #1206,
deliberately left loading here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

ROOT = Path(__file__).resolve().parents[1]
STREAM_SETUP = ROOT / "src" / "soup_cli" / "trainer" / "stream_setup.py"
LAYER_STREAM_MODULES = sorted((ROOT / "src" / "soup_cli" / "utils").glob("layer_stream*.py"))

TRAIN = "tests/fixtures/sample_train.jsonl"
RESIDENT = ("use_liger", "use_flash_attn")


def _cfg(**training) -> SoupConfig:
    return SoupConfig(
        base="test/model",
        task="sft",
        data={"train": TRAIN},
        training={"batch_size": 1, **training},
    )


def _message(training: dict) -> str:
    with pytest.raises(ValueError) as exc:
        _cfg(**training)
    return str(exc.value)


def _raw_message(training: dict) -> str:
    """The validator's own message, without pydantic's rendering around it.

    `str(exc.value)` is a multi-line report that ends with the field's location, so it
    can never end the way these messages do. Anything asserting on the tail of the
    message has to read `errors()[0]["msg"]`.
    """
    with pytest.raises(ValidationError) as exc:
        _cfg(**training)
    return exc.value.errors()[0]["msg"]


class TestTheSwitchesAreRefusedUnderStreaming:
    """The acceptance criterion: refused at load, naming the field."""

    @pytest.mark.parametrize("field", RESIDENT)
    def test_streaming_refuses_each_switch(self, field: str) -> None:
        message = _message({"stream_layers": True, field: True})
        assert "stream_layers is mutually exclusive with" in message
        assert field in message

    def test_both_switches_are_named_together(self) -> None:
        """One conflict list, so a config setting both names both rather than the first."""
        message = _message({"stream_layers": True, "use_liger": True, "use_flash_attn": True})
        assert "use_liger, use_flash_attn" in message

    def test_the_message_says_they_are_never_reached(self) -> None:
        """"Mutually exclusive" alone would read as a conflict. It is a dead switch:
        the streamed model is built without them, so say which and why."""
        message = _message({"stream_layers": True, "use_liger": True})
        assert "never reached at all" in message
        assert "from_config" in message
        assert "resident setup" in message

    def test_the_clause_agrees_in_number(self) -> None:
        """One switch named, one switch referred to.

        The two mutants that survive every other assertion here write "are"/"them" for a
        single switch, or "is"/"it" for both -- so a config with only `use_liger` reads
        "use_liger and use_flash_attn are ... reads them", naming a switch that is not
        set. Asserted on the raw message so the tail is the real tail.
        """
        one = _raw_message({"stream_layers": True, "use_liger": True})
        assert " use_liger is never reached at all" in one
        assert one.endswith("the only thing that reads it.")

        both = _raw_message(
            {"stream_layers": True, "use_liger": True, "use_flash_attn": True}
        )
        assert " use_liger and use_flash_attn are never reached at all" in both
        assert both.endswith("the only thing that reads them.")

    @pytest.mark.parametrize("field", RESIDENT)
    def test_only_the_switches_that_are_set_are_named(self, field: str) -> None:
        """A message naming a switch the config did not set reads as if it were refused
        for a reason that does not apply, so only name what is actually there."""
        message = _raw_message({"stream_layers": True, field: True})
        tail = message.split("the same layers. ", 1)[-1]
        assert tail.startswith(field)
        other = "use_flash_attn" if field == "use_liger" else "use_liger"
        assert other not in tail

    @pytest.mark.parametrize("field", ["packing", "multipack", "nvfp4", "fp8_attention"])
    def test_a_pre_existing_conflict_message_ends_exactly_as_before(self, field: str) -> None:
        """CONTROL, and the load-bearing one.

        Asserted on the raw validator message, not on `str(exc.value)`: pydantic's
        multi-line rendering appends its own location line, so an assertion about the
        tail of `str(...)` passes whether or not the space is there. These four are the
        conflicts that actually reach this list -- `unfrozen_parameters`,
        `expand_layers`, `quantization_aware`, `train_router_only` and `lisa_enabled` are
        all refused by an earlier gate, so a message assertion on them would be vacuous
        for the same reason.
        """
        message = _raw_message({"stream_layers": True, field: True})
        assert field in message
        assert "never reached at all" not in message
        assert message.endswith("re-freezes the same layers.")

    def test_a_kernel_switch_joins_the_other_conflicts_rather_than_replacing_them(self) -> None:
        message = _message(
            {"stream_layers": True, "use_liger": True, "packing": True, "nvfp4": True}
        )
        assert all(name in message for name in ("packing", "nvfp4", "use_liger"))


class TestWhatMustKeepLoading:
    """CONTROLS. The refusal is scoped to the streamed path and to these two fields."""

    @pytest.mark.parametrize("field", RESIDENT)
    def test_the_resident_text_path_still_takes_each_switch(self, field: str) -> None:
        """This is the path that DOES read them, so it must keep loading. Without it the
        guard would have been satisfied by refusing everywhere, which fixes nothing."""
        cfg = _cfg(quantization="4bit", **{field: True})
        assert getattr(cfg.training, field) is True

    def test_streaming_still_loads_without_either_switch(self) -> None:
        assert _cfg(stream_layers=True).training.stream_layers is True

    @pytest.mark.parametrize("field", RESIDENT)
    def test_the_switch_alone_on_a_resident_stream_free_config_loads(self, field: str) -> None:
        assert getattr(_cfg(**{field: True}).training, field) is True

    def test_use_cut_ce_still_loads_under_streaming(self) -> None:
        """SCOPE. #1206 owns `use_cut_ce`; this PR must not quietly widen into it."""
        assert _cfg(stream_layers=True, use_cut_ce=True).training.use_cut_ce is True


class TestTheStreamedPreferencePathsAreAlreadyCovered:
    """Acceptance box 2, resolved rather than coded.

    `stream_layers` does support dpo / kto / orpo / simpo, so these four were checked.
    They cannot reach a streamed path with either switch set, because
    `_validate_kernel_flags_task_gate` (`soup_cli/config/schema.py`) already refuses
    both outside `{sft, tts}` -- the task gate fires first. So no branch is needed here;
    these tests are what would fail if that ordering ever changed.
    """

    @pytest.mark.parametrize("task", ["dpo", "kto", "orpo", "simpo"])
    @pytest.mark.parametrize("field", RESIDENT)
    def test_those_tasks_are_refused_on_their_own_terms(self, task: str, field: str) -> None:
        with pytest.raises(ValueError, match=r"requires task in \['sft', 'tts'\]"):
            SoupConfig(
                base="test/model",
                task=task,
                data={"train": TRAIN},
                training={"batch_size": 2, "lora": {"r": 8}, field: True},
            )

    @pytest.mark.parametrize("task", ["dpo", "kto", "orpo", "simpo"])
    def test_the_task_alone_still_reaches_the_streamed_path(self, task: str) -> None:
        """CONTROL. The gate above is about the switch, not about streaming: the same four
        tasks load with stream_layers, so nothing here is a blanket ban."""
        cfg = SoupConfig(
            base="test/model",
            task=task,
            data={"train": TRAIN},
            training={"batch_size": 2, "lora": {"r": 8}, "stream_layers": True},
        )
        assert cfg.training.stream_layers is True


def _code_without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*#.*$", "", text)


class TestTheStreamedSetupReallyCannotReadThem:
    """The justification, pinned so it cannot rot into a fiction.

    A refusal whose stated reason is wrong is worse than no refusal, so this asserts the
    reason: no module on the streamed path names either switch. Wiring one up is the
    better end state -- delete this scan and the refusal together, with a test showing
    the switch engaging on a streamed model.
    """

    def test_the_stream_setup_module_reads_neither_switch(self) -> None:
        code = _code_without_comments(STREAM_SETUP.read_text(encoding="utf-8"))
        for field in RESIDENT:
            assert field not in code, (
                f"{STREAM_SETUP.name} now names {field}: the streamed path may read it, "
                "so #1692's refusal needs the conflict removed and a test that shows the "
                "switch engaging on a streamed model."
            )

    def test_no_layer_stream_module_reads_either_switch(self) -> None:
        assert LAYER_STREAM_MODULES, "the glob found no layer_stream module to check"
        for path in LAYER_STREAM_MODULES:
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            for field in RESIDENT:
                assert field not in code, f"{path.name} names {field}; see #1692"

    def test_the_resident_setup_still_reads_them(self) -> None:
        """CONTROL. Without this the scan above would pass on a tree where the flags are
        read nowhere at all, which is not the justification this refusal rests on."""
        sft = (ROOT / "src" / "soup_cli" / "trainer" / "sft.py").read_text(encoding="utf-8")
        assert "use_liger" in sft
        assert "use_flash_attn" in sft
