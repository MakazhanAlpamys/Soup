"""#326 — the six preference trainers must survive trl moving under them.

Two independent trl changes broke ``soup train``, and they moved on different
schedules:

1. ``max_prompt_length`` was removed from the preference configs in STAGES —
   kto at 0.27.0, bco/orpo/simpo at 0.28.0, dpo/ipo at 0.29.0 — so any single
   spot-check gives the wrong boundary. There is no successor field.
2. ``ORPOConfig`` / ``CPOConfig`` / ``BCOConfig`` (and their trainers) left the
   public ``trl`` namespace at 0.29.0 for ``trl.experimental.<algo>``. That is
   an ``ImportError`` at ``setup()``, harder than a rejected keyword.

Both are now handled by capability probes in ``trainer/_trl_compat.py`` rather
than by version comparisons, because a version table is exactly what was wrong
twice before. These tests pin the probes' behaviour AGAINST WHATEVER TRL IS
INSTALLED, so they keep their meaning when the cap moves again — a test that
hard-coded "0.28 has no max_prompt_length" would itself become the next stale
version table.

Companion to ``tests/test_trl_preference_config_contract.py``, which drives the
real ``setup()`` for all six. This file covers the compat layer those wrappers
now depend on.
"""

import sys
import types

import pytest

_SIX = ("bco", "dpo", "ipo", "kto", "orpo", "simpo")


# --------------------------------------------------------------------------
# config_accepts / prompt_length_kwargs -- capability, not version
# --------------------------------------------------------------------------
class TestPromptLengthIsDecidedByCapabilityNotVersion:
    def test_kwarg_is_passed_when_the_config_accepts_it(self):
        from soup_cli.trainer._trl_compat import prompt_length_kwargs

        class OldStyleConfig:
            def __init__(self, output_dir=None, max_length=None, max_prompt_length=None):
                pass

        assert prompt_length_kwargs(OldStyleConfig, 32) == {"max_prompt_length": 32}

    def test_kwarg_is_dropped_when_the_config_removed_it(self):
        """The control for the test above. Without it, a helper that returned
        ``{}`` unconditionally — i.e. one that had silently stopped passing the
        cap on EVERY trl — would pass a one-sided suite."""
        from soup_cli.trainer._trl_compat import prompt_length_kwargs

        class NewStyleConfig:
            def __init__(self, output_dir=None, max_length=None):
                pass

        assert prompt_length_kwargs(NewStyleConfig, 32) == {}

    def test_the_probe_never_reads_the_trl_version(self):
        """The whole point of #326. A class that still has the field must get
        the keyword even though the INSTALLED trl may be one that removed it —
        which is only possible if the answer comes from the class, not from
        ``trl.__version__``."""
        from soup_cli.trainer._trl_compat import prompt_length_kwargs

        class StillHasIt:
            def __init__(self, max_prompt_length=None):
                pass

        assert prompt_length_kwargs(StillHasIt, 7) == {"max_prompt_length": 7}

    def test_config_accepts_sees_inherited_keywords(self):
        """trl's configs inherit most of their fields from ``TrainingArguments``.
        A probe that only looked at a class's OWN annotations would answer False
        for every inherited field and silently drop real arguments."""
        from soup_cli.trainer._trl_compat import config_accepts

        class Base:
            def __init__(self, inherited=None):
                pass

        class Child(Base):
            pass

        assert config_accepts(Child, "inherited") is True
        assert config_accepts(Child, "not_a_field") is False

    def test_an_unintrospectable_class_degrades_to_false(self):
        """Some C-level ``__init__``s raise from ``inspect.signature``. Dropping
        an optional kwarg is recoverable; raising out of ``setup()`` is not."""
        from soup_cli.trainer._trl_compat import config_accepts

        assert config_accepts(object(), "max_prompt_length") is False


# --------------------------------------------------------------------------
# resolve_trl_symbol -- the namespace move
# --------------------------------------------------------------------------
def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class TestResolveTrlSymbol:
    def test_public_namespace_is_used_when_available(self):
        import trl

        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        assert resolve_trl_symbol("DPOConfig", "trl.experimental.dpo") is trl.DPOConfig

    def test_public_namespace_wins_over_experimental(self, monkeypatch):
        """Order is load-bearing, not incidental: on a trl that still exports
        the symbol publicly, taking the experimental copy would opt users into
        an upstream-experimental code path — and emit a warning — for no reason.
        A fake experimental module holding a DIFFERENT object proves which one
        was chosen; asserting only "it returned something" would not."""
        import trl

        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        sentinel = object()
        monkeypatch.setitem(
            sys.modules, "soup_fake_exp", _fake_module("soup_fake_exp", DPOConfig=sentinel)
        )
        resolved = resolve_trl_symbol("DPOConfig", "soup_fake_exp")
        assert resolved is trl.DPOConfig
        assert resolved is not sentinel

    def test_experimental_fallback_is_used_when_the_symbol_left_trl(self, monkeypatch):
        """The 0.29.0 case. Uses a name trl has never exported, so the test does
        not depend on which trl is installed."""
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        sentinel = object()
        monkeypatch.setitem(
            sys.modules,
            "soup_fake_exp2",
            _fake_module("soup_fake_exp2", SoupAbsentConfig=sentinel),
        )
        assert resolve_trl_symbol("SoupAbsentConfig", "soup_fake_exp2") is sentinel

    def test_missing_everywhere_raises_an_import_error_naming_the_symbol(self):
        """The next removal should say what is missing and on which trl, rather
        than surfacing a bare AttributeError from inside a lazy module."""
        import trl

        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        with pytest.raises(ImportError) as excinfo:
            resolve_trl_symbol("SoupDefinitelyAbsent", "soup_module_that_does_not_exist")
        message = str(excinfo.value)
        assert "SoupDefinitelyAbsent" in message
        assert trl.__version__ in message
        assert "soup_module_that_does_not_exist" in message

    def test_the_original_failure_is_chained_not_swallowed(self):
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        with pytest.raises(ImportError) as excinfo:
            resolve_trl_symbol("SoupDefinitelyAbsent")
        assert excinfo.value.__cause__ is not None, "root cause was discarded"

    def test_no_experimental_module_given_still_raises_cleanly(self):
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        with pytest.raises(ImportError, match="SoupDefinitelyAbsent"):
            resolve_trl_symbol("SoupDefinitelyAbsent")


# --------------------------------------------------------------------------
# the trainers themselves
# --------------------------------------------------------------------------
class TestTheTrainersRouteThroughTheCompatLayer:
    """Guards the property, not the text: a trainer that goes back to passing
    the keyword directly would work on today's trl and break on the next one,
    which is exactly how this shipped broken twice."""

    def _trainer_source(self, name):
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "soup_cli"
            / "trainer"
            / f"{name}.py"
        )
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("task", _SIX)
    def test_no_trainer_passes_max_prompt_length_directly(self, task):
        source = self._trainer_source(task)
        assert "max_prompt_length=" not in source, (
            f"{task}.py passes `max_prompt_length=` directly again — it must go "
            f"through prompt_length_kwargs(), or the next trl removal breaks it"
        )

    @pytest.mark.parametrize("task", _SIX)
    def test_every_preference_trainer_uses_the_helper(self, task):
        assert "prompt_length_kwargs(" in self._trainer_source(task), task

    @pytest.mark.parametrize("task", ("orpo", "simpo", "bco"))
    def test_the_relocated_three_resolve_their_classes(self, task):
        """Only these three left the public trl namespace at 0.29.0. dpo / ipo /
        kto keep a plain import, so pinning all six here would assert a
        migration that deliberately did not happen."""
        assert "resolve_trl_symbol(" in self._trainer_source(task), task

    @pytest.mark.parametrize("task", ("dpo", "ipo", "kto"))
    def test_the_others_still_import_directly(self, task):
        source = self._trainer_source(task)
        assert "from trl import" in source, task


class TestTheDeclaredBoundMatchesTheInstalledTrl:
    """The floor shipped as ``>=0.7.0`` while ``setup()`` imported a symbol trl
    first exported at 0.14.0 — a declared bound the code could never have run
    on. Cheap to state as a property."""

    def test_every_symbol_the_six_need_is_resolvable(self):
        """Resolution, not a bare import: three of these legitimately live under
        ``trl.experimental`` on a new enough trl, so a plain ``from trl import``
        check would report a false failure there."""
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        needed = {
            "DPOConfig": None,
            "DPOTrainer": None,
            "KTOConfig": "trl.experimental.kto",
            "KTOTrainer": "trl.experimental.kto",
            "ORPOConfig": "trl.experimental.orpo",
            "ORPOTrainer": "trl.experimental.orpo",
            "CPOConfig": "trl.experimental.cpo",
            "CPOTrainer": "trl.experimental.cpo",
            "BCOConfig": "trl.experimental.bco",
            "BCOTrainer": "trl.experimental.bco",
        }
        broken = {}
        for name, experimental in needed.items():
            try:
                resolve_trl_symbol(name, experimental)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                broken[name] = f"{type(exc).__name__}: {exc}"
        assert not broken, broken

    def test_the_installed_trl_satisfies_the_declared_cap(self):
        """Reads the bound out of pyproject rather than restating it, so the two
        cannot drift. Skips when the metadata is not on disk (installed wheel)."""
        import pathlib
        import re

        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        if not pyproject.exists():  # pragma: no cover - source checkouts have it
            pytest.skip("pyproject.toml not present (installed, not a checkout)")
        match = re.search(r'"trl>=([0-9.]+),<([0-9.]+)"', pyproject.read_text(encoding="utf-8"))
        assert match, "could not find the trl bound in pyproject.toml"
        floor, cap = match.group(1), match.group(2)

        import trl

        def parts(version):
            return tuple(int(piece) for piece in re.findall(r"\d+", version)[:3])

        installed = parts(trl.__version__)
        assert installed >= parts(floor), (trl.__version__, floor)
        assert installed < parts(cap), (
            f"installed trl {trl.__version__} is at or above the declared cap "
            f"{cap} — the tests are exercising a version the package says it "
            f"does not support"
        )
