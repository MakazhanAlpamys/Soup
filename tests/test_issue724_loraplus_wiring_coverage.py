"""Issue #724 — the LoRA+ optimizer wiring, in every wrapper that should have it.

#724 fixed the failure in ``trainer/sft.py``, ``pretrain.py`` and ``embedding.py``:
``loraplus_lr_ratio`` was forwarded into ``TrainingArguments``, which is not a
field there, so the run crashed before the first step. The fix routes it through
``attach_loraplus_optimizer`` instead, called after the trainer is built.

That change made the failure mode *quieter*, which is exactly why it needs this
scan. Before #724 a wrapper that never wired LoRA+ **crashed** on the unknown
keyword; after #724 a wrapper that never calls ``attach_loraplus_optimizer`` runs
to completion with LoRA+ silently disabled — the loss curve looks normal and the
B matrices simply trained at the base rate. A test has to provide the signal the
crash used to give for free.

Coverage is derived by SCANNING ``soup_cli/trainer/`` (following
``tests/test_issue359_deepspeed_guard_coverage.py``) rather than a hand-written
list, so a wrapper added later cannot ship LoRA+-less without either wiring it or
being named — with a reason — in the exemption set below.

Finding this scan surfaced, recorded here so it is not lost: LoRA+ is a shared
``TrainingConfig`` option with no task gating, but only the three SFT-family
wrappers ever implemented it. The other PEFT-building wrappers (preference/RL/
specialised trainers) accept ``loraplus_lr_ratio`` and silently ignore it — they
build no custom optimizer at all. Extending LoRA+ to them, or rejecting it there,
is out of #724's scope (the three that crashed) and belongs in a follow-up; until
then they are listed in ``_LORAPLUS_NOT_IMPLEMENTED`` so the gap is explicit.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_TRAINER_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
_TRAINER_SOURCES = sorted(_TRAINER_DIR.glob("*.py"))

#: A wrapper applies a LoRA adapter when it calls ``get_peft_model(...)`` — the
#: point at which LoRA A/B matrices exist for LoRA+ to give different rates to.
_BUILDS_PEFT = re.compile(r"get_peft_model\s*\(")
_ATTACH_CALL = re.compile(r"attach_loraplus_optimizer\s*\(")

#: PEFT-building wrappers that do NOT implement LoRA+ today. They accept
#: ``loraplus_lr_ratio`` (it is a shared TrainingConfig field) and silently
#: ignore it, building no custom optimizer. This is a real gap, tracked for a
#: follow-up rather than fixed in #724's scope. A module is named here so that a
#: wrapper which later GAINS LoRA+ must move out of the set (the "stays earned"
#: test enforces that), and so a newly added LoRA-less wrapper has to be argued
#: for rather than slipping through.
_LORAPLUS_NOT_IMPLEMENTED = {
    "asr.py", "bco.py", "classifier.py", "distill.py", "dpo.py", "grpo.py",
    "ipo.py", "kto.py", "orpo.py", "ppo.py", "reward_model.py", "simpo.py",
    "unlearn.py",
}


def _code_without_comments(text: str) -> str:
    """Strip trailing comments so prose ABOUT the call is not read as a call."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _builds_peft(path: pathlib.Path) -> bool:
    return bool(_BUILDS_PEFT.search(_code_without_comments(path.read_text(encoding="utf-8"))))


class TestLoraPlusWiringCoverage:
    def test_the_scan_actually_sees_the_trainer_package(self):
        """Without this, a moved source tree turns the checks below into a
        vacuous pass over an empty file list."""
        assert _TRAINER_DIR.is_dir(), _TRAINER_DIR
        names = {path.name for path in _TRAINER_SOURCES}
        assert len(names) > 20
        assert {"sft.py", "pretrain.py", "embedding.py", "dpo.py", "ppo.py"} <= names

    def test_the_scan_finds_the_peft_builders(self):
        """If this ever collapses to a handful, the detector has broken rather
        than the codebase having shed its LoRA trainers."""
        building = [p.name for p in _TRAINER_SOURCES if _builds_peft(p)]
        assert len(building) >= 15, building
        assert {"sft.py", "pretrain.py", "embedding.py"} <= set(building)

    @pytest.mark.parametrize("path", _TRAINER_SOURCES, ids=[p.stem for p in _TRAINER_SOURCES])
    def test_every_peft_builder_wires_loraplus_or_is_exempt(self, path):
        code = _code_without_comments(path.read_text(encoding="utf-8"))
        if not _BUILDS_PEFT.search(code):
            return
        if path.name in _LORAPLUS_NOT_IMPLEMENTED:
            return
        assert _ATTACH_CALL.search(code), (
            f"{path.name} applies a LoRA adapter (get_peft_model) but never calls "
            "attach_loraplus_optimizer(); training.loraplus_lr_ratio would be "
            "silently ignored on this task (#724). Wire it, or add the module to "
            "_LORAPLUS_NOT_IMPLEMENTED with a reason."
        )

    def test_the_three_sft_family_wrappers_are_the_ones_wired(self):
        """Pins the positive set so the parametrized check above is not vacuous:
        exactly the wrappers #724 fixed must carry the call."""
        wired = {
            p.name for p in _TRAINER_SOURCES
            if _ATTACH_CALL.search(_code_without_comments(p.read_text(encoding="utf-8")))
        }
        assert wired == {"sft.py", "pretrain.py", "embedding.py"}, wired

    def test_no_peft_builder_quietly_loses_the_wiring(self):
        """Aggregate form of the per-file check: every LoRA-building wrapper is
        either wired or explicitly exempt. Catches a new wrapper that builds a
        PEFT model and does neither."""
        offenders = []
        for path in _TRAINER_SOURCES:
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            if not _BUILDS_PEFT.search(code):
                continue
            if path.name in _LORAPLUS_NOT_IMPLEMENTED:
                continue
            if not _ATTACH_CALL.search(code):
                offenders.append(path.name)
        assert not offenders, (
            f"{', '.join(offenders)} apply a LoRA adapter but never call "
            "attach_loraplus_optimizer(). Wire LoRA+ or add to "
            "_LORAPLUS_NOT_IMPLEMENTED with the reason."
        )

    def test_the_exemption_list_stays_earned(self):
        """An exemption that stops being true is worse than none. Each exempt
        module must still exist, must still build a PEFT model (else it does not
        belong in a PEFT-builder exemption), and must NOT already call the attach
        (if it does, it is wired and should leave the set)."""
        for name in _LORAPLUS_NOT_IMPLEMENTED:
            path = _TRAINER_DIR / name
            assert path.is_file(), f"_LORAPLUS_NOT_IMPLEMENTED names {name}, which no longer exists"
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            assert _BUILDS_PEFT.search(code), (
                f"{name} is exempt but no longer builds a PEFT model; drop it from "
                "_LORAPLUS_NOT_IMPLEMENTED."
            )
            assert not _ATTACH_CALL.search(code), (
                f"{name} now calls attach_loraplus_optimizer; remove it from "
                "_LORAPLUS_NOT_IMPLEMENTED — it is wired, not exempt."
            )

    def test_the_patterns_would_catch_the_unwired_shape(self):
        """A scanner nobody has watched fail is indistinguishable from a broken
        one. Both detectors are exercised here, comments included."""
        assert _BUILDS_PEFT.search("        self.model = get_peft_model(model, cfg)")
        assert not _BUILDS_PEFT.search("        # get_peft_model is applied elsewhere")
        assert _ATTACH_CALL.search("attach_loraplus_optimizer(self.trainer, tcfg)")
        assert not _ATTACH_CALL.search("# attach_loraplus_optimizer is needed here")

    def test_a_comment_mentioning_the_call_does_not_satisfy_it(self):
        """The positive check reads code, not prose — otherwise the note that
        explains the wiring would pass without calling it."""
        source = "self.trainer = SFTTrainer()  # attach_loraplus_optimizer(x)\n"
        assert not _ATTACH_CALL.search(_code_without_comments(source))
