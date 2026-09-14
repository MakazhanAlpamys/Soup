"""Issue #725 -- the LoRA-FA optimizer wiring, in every wrapper that should have it.

#725 introduces ``training.use_lorafa``, wired via ``attach_lorafa_optimizer``
in ``trainer/sft.py``, ``pretrain.py`` and ``embedding.py`` after the trainer is built.
This scan ensures every PEFT-building wrapper either calls ``attach_lorafa_optimizer``
or is explicitly named in ``_LORAFA_NOT_IMPLEMENTED``.

Mirrors ``tests/test_issue724_loraplus_wiring_coverage.py``.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_TRAINER_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
_TRAINER_SOURCES = sorted(_TRAINER_DIR.glob("*.py"))

#: A wrapper applies a LoRA adapter when it calls ``get_peft_model(...)`` -- the
#: point at which LoRA A/B matrices exist for LoRA-FA to freeze A and train B.
_BUILDS_PEFT = re.compile(r"get_peft_model\s*\(")
_ATTACH_CALL = re.compile(r"attach_lorafa_optimizer\s*\(")

#: PEFT-building wrappers that do NOT implement LoRA-FA today. They accept
#: ``use_lorafa`` (it is a shared TrainingConfig field) and silently
#: ignore it, building no custom optimizer. Mirrors _LORAPLUS_NOT_IMPLEMENTED.
_LORAFA_NOT_IMPLEMENTED = {
    "asr.py",
    "bco.py",
    "classifier.py",
    "distill.py",
    "dpo.py",
    "grpo.py",
    "ipo.py",
    "kto.py",
    "orpo.py",
    "ppo.py",
    "reward_model.py",
    "simpo.py",
    "unlearn.py",
}


def _code_without_comments(text: str) -> str:
    """Strip trailing comments so prose ABOUT the call is not read as a call."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _builds_peft(path: pathlib.Path) -> bool:
    return bool(_BUILDS_PEFT.search(_code_without_comments(path.read_text(encoding="utf-8"))))


class TestLoraFAWiringCoverage:
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
    def test_every_peft_builder_wires_lorafa_or_is_exempt(self, path):
        code = _code_without_comments(path.read_text(encoding="utf-8"))
        if not _BUILDS_PEFT.search(code):
            return
        if path.name in _LORAFA_NOT_IMPLEMENTED:
            return
        assert _ATTACH_CALL.search(code), (
            f"{path.name} applies a LoRA adapter (get_peft_model) but never calls "
            "attach_lorafa_optimizer(); training.use_lorafa would be "
            "silently ignored on this task (#725). Wire it, or add the module to "
            "_LORAFA_NOT_IMPLEMENTED with a reason."
        )

    def test_the_three_sft_family_wrappers_are_the_ones_wired(self):
        """Pins the positive set so the parametrized check above is not vacuous:
        exactly the wrappers #725 wired must carry the call."""
        wired = {
            p.name
            for p in _TRAINER_SOURCES
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
            if path.name in _LORAFA_NOT_IMPLEMENTED:
                continue
            if not _ATTACH_CALL.search(code):
                offenders.append(path.name)
        assert not offenders, (
            f"{', '.join(offenders)} apply a LoRA adapter but never call "
            "attach_lorafa_optimizer(). Wire LoRA-FA or add to "
            "_LORAFA_NOT_IMPLEMENTED with the reason."
        )

    def test_the_exemption_list_stays_earned(self):
        """An exemption that stops being true is worse than none. Each exempt
        module must still exist, must still build a PEFT model (else it does not
        belong in a PEFT-builder exemption), and must NOT already call the attach
        (if it does, it is wired and should leave the set)."""
        for name in _LORAFA_NOT_IMPLEMENTED:
            path = _TRAINER_DIR / name
            assert path.is_file(), f"_LORAFA_NOT_IMPLEMENTED names {name}, which no longer exists"
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            assert _BUILDS_PEFT.search(code), (
                f"{name} is exempt but no longer builds a PEFT model; drop it from "
                "_LORAFA_NOT_IMPLEMENTED."
            )
            assert not _ATTACH_CALL.search(code), (
                f"{name} now calls attach_lorafa_optimizer; remove it from "
                "_LORAFA_NOT_IMPLEMENTED -- it is wired, not exempt."
            )

    def test_the_patterns_would_catch_the_unwired_shape(self):
        """A scanner nobody has watched fail is indistinguishable from a broken
        one. Both detectors are exercised here, comments included."""
        assert _BUILDS_PEFT.search("        self.model = get_peft_model(model, cfg)")
        assert not _BUILDS_PEFT.search("        # get_peft_model is applied elsewhere")
        assert _ATTACH_CALL.search("attach_lorafa_optimizer(self.trainer, tcfg)")
        assert not _ATTACH_CALL.search("# attach_lorafa_optimizer is needed here")

    def test_a_comment_mentioning_the_call_does_not_satisfy_it(self):
        """The positive check reads code, not prose -- otherwise the note that
        explains the wiring would pass without calling it."""
        source = "self.trainer = SFTTrainer()  # attach_lorafa_optimizer(x)\n"
        assert not _ATTACH_CALL.search(_code_without_comments(source))
