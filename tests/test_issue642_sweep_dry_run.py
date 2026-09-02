"""``soup sweep --dry-run`` must validate, not just print the grid (#642).

The dry-run branch returned before the config was ever loaded, so neither the
loader's unknown-key warning (#627) nor the sweep-parameter pre-flight (#628)
was reachable under it — the headline complaint of #627 (a dry run printing a
clean bill over a typo'd key) was fixed for ``train`` in #628 and left standing
here. ``--dry-run`` now means "validate, then print the grid without running
anything".

Every test runs the real command through the CLI runner and asserts on its
output and exit code — no source-text assertions, per the #628 review note
(that pattern passed three times while the caller was unwired). CPU-only: no
downloads, no model load, no training.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.unknown_keys import UNKNOWN_KEY_REJECTION_VERSION

_CLEAN = """\
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
output: ./o
"""

# The issue's own reproduction: one edit from `quantization`, silently dropped
# before #627 and silently unreported under sweep --dry-run before this fix.
_TYPO = """\
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
training:
  quantizaton: none
output: ./o
"""

_INVALID_SCHEMA = """\
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
training:
  epochs: not-a-number
output: ./o
"""


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, cfg_text: str, tmp_path, *extra: str, input: str | None = None):
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(cfg_text, encoding="utf-8")
    return runner.invoke(
        app,
        ["sweep", "--config", str(config_file), "--param", "lr=1e-5,2e-5", "--dry-run", *extra],
        input=input,
    )


class TestTheWarningBranch:
    """Criterion 1: the same block ``soup train --dry-run`` prints, same loader."""

    def test_typo_config_warns_and_still_exits_zero(self, runner, tmp_path) -> None:
        result = _invoke(runner, _TYPO, tmp_path)
        assert result.exit_code == 0, result.output
        assert "Warning:" in result.output
        assert "training.quantizaton" in result.output
        # The suggestion is the point of the block (pinned by #627's tests at
        # the unit level; here it must survive the trip through the command).
        assert "quantization" in result.output
        # Criterion 3: a config that merely warns keeps the dry-run exit at 0.
        assert "Dry run - no training will be executed." in result.output

    def test_the_warning_carries_the_loaders_deadline(self, runner, tmp_path) -> None:
        # Same loader path means the deadline sentence and the follow-up
        # explanation come along; a hand-rolled second warning would not.
        # Stripped for colour: under FORCE_COLOR rich's number highlighter
        # puts escapes INSIDE "v0.75", splitting the token (#633's class).
        from .conftest import strip_ansi

        result = _invoke(runner, _TYPO, tmp_path)
        out = strip_ansi(result.output)
        assert f"v{UNKNOWN_KEY_REJECTION_VERSION}" in out
        assert "An unapplied key is ignored, not defaulted" in out
        # One load, one report — a second load_config on the way out would
        # print the block twice.
        assert out.count("Warning:") == 1

    def test_the_grid_still_prints_for_a_warn_only_config(self, runner, tmp_path) -> None:
        result = _invoke(runner, _TYPO, tmp_path)
        assert "Sweep Plan" in result.output
        assert "Parameter Grid" in result.output


class TestTheHardParamBranch:
    """Criterion 2: the #628 pre-flight runs under --dry-run, before the grid."""

    @pytest.mark.parametrize("param", ["lora_rank=8,16", "training.lr_rate=1e-5,2e-5"])
    def test_typo_sweep_param_exits_nonzero_without_printing_a_grid(
        self, runner, tmp_path, param
    ) -> None:
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(_CLEAN, encoding="utf-8")
        result = runner.invoke(
            app,
            ["sweep", "--config", str(config_file), "--param", param, "--dry-run"],
        )
        assert result.exit_code == 1, result.output
        assert "does not match any config field" in result.output
        # "Should not print a grid it can never run" — the refusal comes first.
        assert "Sweep Plan" not in result.output
        assert "Parameter Grid" not in result.output


class TestTheSeveritySplit:
    """Criterion 3: matches the non-dry-run path; no third behaviour."""

    def test_schema_invalid_config_exits_nonzero_under_dry_run(self, runner, tmp_path) -> None:
        # load_config's hard refusal, identical to what the execution path has
        # always done with the same file.
        result = _invoke(runner, _INVALID_SCHEMA, tmp_path)
        assert result.exit_code == 1, result.output
        assert "Config validation error" in result.output


class TestTheControl:
    """Criterion 4: a clean config under --dry-run emits no warning at all."""

    def test_clean_config_warns_nothing_and_exits_zero(self, runner, tmp_path) -> None:
        result = _invoke(runner, _CLEAN, tmp_path)
        assert result.exit_code == 0, result.output
        assert "Warning:" not in result.output
        assert "quantizaton" not in result.output
        assert "Sweep Plan" in result.output
        assert "Dry run - no training will be executed." in result.output


class TestTheExecutionPathStillMatches:
    """The reorder must not invent a second behaviour for the real run.

    Validation now happens before the confirmation prompt instead of after
    it — same reports, same exit codes, earlier. Declining the prompt after a
    warning keeps the historical cancelled exit.
    """

    def test_warning_appears_before_the_prompt_and_once(self, runner, tmp_path) -> None:
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(_TYPO, encoding="utf-8")
        result = runner.invoke(
            app,
            ["sweep", "--config", str(config_file), "--param", "lr=1e-5,2e-5"],
            input="n\n",
        )
        assert result.exit_code == 0, result.output
        assert "Cancelled." in result.output
        assert result.output.count("Warning:") == 1
        warn_at = result.output.index("training.quantizaton")
        prompt_at = result.output.index("Start 2 training run(s)?")
        assert warn_at < prompt_at, "the operator must see the warning before confirming"

    def test_the_execution_path_does_not_load_the_config_again(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # The cancelled-at-prompt test never reaches the arm loop, so a second
        # load_config on the execution path would hide from it — a mutant
        # re-loading after the prompt survived until this test. _run_single is
        # stubbed because the contract here is "one load, one warning", not
        # training; the stub returns what the arm loop consumes.
        import soup_cli.commands.sweep as sweep_mod

        monkeypatch.setattr(
            sweep_mod,
            "_run_single",
            lambda *a, **k: {"final_loss": 0.5, "run_id": "r1", "duration_secs": 1},
        )
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(_TYPO, encoding="utf-8")
        result = runner.invoke(
            app,
            ["sweep", "--config", str(config_file), "--param", "lr=1e-5,2e-5", "-y"],
        )
        assert result.exit_code == 0, result.output
        assert result.output.count("Warning:") == 1
        assert "Sweep Results" in result.output

    def test_hard_param_refusal_is_unchanged_without_dry_run(self, runner, tmp_path) -> None:
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(_CLEAN, encoding="utf-8")
        result = runner.invoke(
            app,
            ["sweep", "--config", str(config_file), "--param", "lora_rank=8,16", "-y"],
        )
        assert result.exit_code == 1, result.output
        assert "does not match any config field" in result.output


class TestTheGenericAliasFilter:
    """Regression: the pre-flight walk crashed on a full dump — scoped exactly.

    The probe walks a full ``model_dump()``, which visits every declared
    field — including ``training.preference_loss_weights``, annotated
    ``Optional[dict[str, float]]``. Two conditions made that fatal, and BOTH
    had to hold:

    * Python 3.10, where a PEP-585 alias passes ``isinstance(c, type)``
      (3.11 made isinstance return False), and
    * a pydantic whose ``ModelMetaclass`` inherits ``__subclasscheck__`` from
      ``ABCMeta``, which raises ``TypeError`` on a non-class — verified
      raising on 2.10.6 and verified tolerant on 2.13.4 (which defines its
      own ``__subclasscheck__``) under the SAME 3.10.11 interpreter, so
      pydantic is the discriminating variable, not the Python patch level.

    The repo floor is ``pydantic>=2.0.0``, so on 3.10 the crash was live
    across the unmasked part of the supported range; a current pydantic
    masks it, which is how the review's box — and every CI cell — saw the
    filter-removal mutation survive. #628's unit tests fed the walker
    hand-built YAML dicts that never reached such a field.
    """

    def _full_dump(self) -> dict:
        import yaml

        from soup_cli.config.schema import SoupConfig

        return SoupConfig(**yaml.safe_load(_CLEAN)).model_dump()

    def test_walking_a_full_dump_does_not_raise(self) -> None:
        from soup_cli.config.unknown_keys import find_unknown_config_keys

        assert find_unknown_config_keys(self._full_dump()) == []

    def test_the_walker_still_finds_an_injected_key(self) -> None:
        # The control on the filter: "never crash" must not be achieved by
        # never walking. A key no model declares is still found in a full dump.
        from soup_cli.config.unknown_keys import find_unknown_config_keys

        dumped = self._full_dump()
        dumped["training"]["quantizaton"] = "none"
        found = find_unknown_config_keys(dumped)
        assert [u.path for u in found] == ["training.quantizaton"]

    def test_the_hazard_is_pinned_stdlib_only(self) -> None:
        """The two facts the filter rests on, with no pydantic involved."""
        import abc
        import sys

        class _AbcChecked(metaclass=abc.ABCMeta):
            pass

        alias = dict[str, float]
        if sys.version_info[:2] == (3, 10):
            assert isinstance(alias, type), (
                "the 3.10 GenericAlias isinstance quirk is gone — the filter's "
                "raison d'être needs re-examining"
            )
            with pytest.raises(TypeError):
                issubclass(alias, _AbcChecked)
        else:
            # 3.11+ filters the alias at isinstance already; there the
            # get_origin guard is belt-and-braces, and this documents the
            # scoping rather than pretending otherwise.
            assert not isinstance(alias, type)

    def test_the_filter_kills_the_crash_whatever_pydantic_does(self, monkeypatch) -> None:
        """The deterministic mutation killer.

        Removing the ``get_origin`` filter survives every other test on a box
        whose pydantic tolerates the alias (2.13.4+) — the review reproduced
        exactly that. This recreates the pre-override metaclass behaviour with
        an ABCMeta stand-in for ``pydantic.BaseModel``, so on any Python 3.10
        box the unfiltered ``issubclass`` raises ``TypeError`` HERE regardless
        of the installed pydantic — and it pins the real schema field the
        crash came through, so renaming or retyping it fails too.
        """
        import abc

        import pydantic

        from soup_cli.config import unknown_keys as uk
        from soup_cli.config.schema import TrainingConfig

        class _RaisingCheckBase(metaclass=abc.ABCMeta):
            pass

        assert "preference_loss_weights" in TrainingConfig.model_fields
        monkeypatch.setattr(pydantic, "BaseModel", _RaisingCheckBase)
        # Without the filter this is issubclass(dict[str, float], ABCMeta-class)
        # — TypeError on 3.10; with it, the alias never reaches issubclass.
        assert uk._nested_models(TrainingConfig, "preference_loss_weights") == []
