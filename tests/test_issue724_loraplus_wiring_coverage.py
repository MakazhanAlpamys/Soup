"""Issue #724 / #745 — the LoRA+ optimizer wiring, in every trainer that should
have it.

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

**#745 extends the coverage from the three SFT-family wrappers to every wrapper
that builds a TRL/HF ``Trainer``.** LoRA+ is a shared ``TrainingConfig`` option
with no task gating, so a preference/RL/specialised wrapper that builds a PEFT
model and does not wire it silently ignores ``loraplus_lr_ratio``. The one
genuine exception is ``unlearn.py``: it runs a self-contained
``torch.optim.AdamW`` loop (not a ``Trainer`` subclass), so there is no
``trainer.optimizer`` to attach to. Rather than lump it in with the wired
wrappers or leave it silently ignoring the option, ``loraplus_lr_ratio`` is
**refused at config parse** for ``task='unlearn'`` — see
``test_unlearn_loraplus_is_refused_at_config_parse``. So the exemption below is
earned by a hard error, not by an unimplemented gap.

Coverage is derived by SCANNING ``soup_cli/trainer/`` (following
``tests/test_issue359_deepspeed_guard_coverage.py``) rather than a hand-written
list, so a wrapper added later cannot ship LoRA+-less without either wiring it or
being named — with a reason — in the exemption set below.
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

#: PEFT-building wrappers that do NOT wire ``attach_loraplus_optimizer``, each
#: with the reason it is allowed to. After #745 there is exactly one: ``unlearn``
#: is not a ``Trainer`` subclass (it drives ``torch.optim.AdamW`` directly), so
#: there is nothing to attach to and ``loraplus_lr_ratio`` is refused at parse
#: instead. A wrapper is named here so a newly added LoRA-less trainer has to be
#: argued for rather than slipping through, and the "stays earned" test below
#: keeps each reason true.
_LORAPLUS_EXEMPT = {
    "unlearn.py": (
        "not a Trainer subclass — runs torch.optim.AdamW directly (unlearn.py), "
        "so there is no trainer.optimizer to attach LoRA+ to. loraplus_lr_ratio "
        "is refused at config parse for task='unlearn' instead of being silently "
        "ignored (see test_unlearn_loraplus_is_refused_at_config_parse)."
    ),
}


def _code_without_comments(text: str) -> str:
    """Strip trailing comments so prose ABOUT the call is not read as a call."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _builds_peft(path: pathlib.Path) -> bool:
    return bool(_BUILDS_PEFT.search(_code_without_comments(path.read_text(encoding="utf-8"))))


def _wires_loraplus(path: pathlib.Path) -> bool:
    return bool(_ATTACH_CALL.search(_code_without_comments(path.read_text(encoding="utf-8"))))


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
        if path.name in _LORAPLUS_EXEMPT:
            return
        assert _ATTACH_CALL.search(code), (
            f"{path.name} applies a LoRA adapter (get_peft_model) but never calls "
            "attach_loraplus_optimizer(); training.loraplus_lr_ratio would be "
            "silently ignored on this task (#724/#745). Wire it, or add the module "
            "to _LORAPLUS_EXEMPT with a reason."
        )

    def test_every_builder_except_the_exempt_is_wired(self):
        """Pins the positive set so the parametrized check above is not vacuous:
        every PEFT-building wrapper except the earned exemptions carries the call.
        #724 wired three; #745 must wire all the rest (dpo, ppo, kto, ...)."""
        builders = {p.name for p in _TRAINER_SOURCES if _builds_peft(p)}
        wired = {p.name for p in _TRAINER_SOURCES if _wires_loraplus(p)}
        assert wired == builders - set(_LORAPLUS_EXEMPT), (
            "wired set must be exactly the PEFT builders minus the exemptions; "
            f"missing={sorted(builders - set(_LORAPLUS_EXEMPT) - wired)}, "
            f"unexpected={sorted(wired - (builders - set(_LORAPLUS_EXEMPT)))}"
        )

    def test_no_peft_builder_quietly_loses_the_wiring(self):
        """Aggregate form of the per-file check: every LoRA-building wrapper is
        either wired or explicitly exempt. Catches a new wrapper that builds a
        PEFT model and does neither."""
        offenders = []
        for path in _TRAINER_SOURCES:
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            if not _BUILDS_PEFT.search(code):
                continue
            if path.name in _LORAPLUS_EXEMPT:
                continue
            if not _ATTACH_CALL.search(code):
                offenders.append(path.name)
        assert not offenders, (
            f"{', '.join(offenders)} apply a LoRA adapter but never call "
            "attach_loraplus_optimizer(). Wire LoRA+ or add to _LORAPLUS_EXEMPT "
            "with the reason."
        )

    def test_the_exemption_list_stays_earned(self):
        """An exemption that stops being true is worse than none. Each exempt
        module must still exist, must still build a PEFT model (else it does not
        belong in a PEFT-builder exemption), and must NOT already call the attach
        (if it does, it is wired and should leave the set)."""
        for name in _LORAPLUS_EXEMPT:
            path = _TRAINER_DIR / name
            assert path.is_file(), f"_LORAPLUS_EXEMPT names {name}, which no longer exists"
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            assert _BUILDS_PEFT.search(code), (
                f"{name} is exempt but no longer builds a PEFT model; drop it from "
                "_LORAPLUS_EXEMPT."
            )
            assert not _ATTACH_CALL.search(code), (
                f"{name} now calls attach_loraplus_optimizer; remove it from "
                "_LORAPLUS_EXEMPT — it is wired, not exempt."
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


class TestUnlearnExemptionIsEarned:
    """The sole exemption (``unlearn``) is earned by a hard refusal, not silence.

    ``unlearn.py`` builds a PEFT model but drives its own ``torch.optim.AdamW``,
    so ``attach_loraplus_optimizer`` cannot reach it. #745's rule: refuse
    ``loraplus_lr_ratio`` for ``task='unlearn'`` at config parse, so a user is
    told rather than trained without the LoRA+ rates they asked for.
    """

    def _unlearn_config(self, *, loraplus: bool):
        """A valid ``task='unlearn'`` config (unlearn_method + forget set), with
        or without ``loraplus_lr_ratio`` — so the only variable is the option
        under test, not the unrelated unlearn-shape requirements."""
        from soup_cli.config.schema import SoupConfig

        training = {"unlearn_method": "npo"}
        if loraplus:
            training["loraplus_lr_ratio"] = 16.0
        return SoupConfig(
            base="hf-internal-testing/tiny-random-gpt2",
            task="unlearn",
            data={"train": "forget.jsonl", "forget_set": "forget.jsonl"},
            training=training,
        )

    def test_unlearn_loraplus_is_refused_at_config_parse(self):
        """task='unlearn' + loraplus_lr_ratio must raise at parse, naming the
        option, rather than parsing and silently ignoring it. The config is
        otherwise valid, so a pass here means the refusal fired (not the
        unrelated unlearn_method/forget requirements)."""
        with pytest.raises(ValueError, match="loraplus_lr_ratio"):
            self._unlearn_config(loraplus=True)

    def test_unlearn_without_loraplus_still_parses(self):
        """The refusal is scoped to the option, not the task: an otherwise
        identical unlearn config that does not set loraplus_lr_ratio parses."""
        self._unlearn_config(loraplus=False)

    def test_loraplus_is_allowed_on_a_wired_task(self):
        """Control: the same option on a Trainer-based task (sft) parses fine, so
        the refusal is specific to unlearn and not a blanket loraplus ban."""
        from soup_cli.config.schema import SoupConfig

        SoupConfig(
            base="hf-internal-testing/tiny-random-gpt2",
            task="sft",
            data={"train": "t.jsonl"},
            training={"loraplus_lr_ratio": 16.0},
        )
