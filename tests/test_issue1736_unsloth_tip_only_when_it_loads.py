"""#1736 — ``soup train`` tips ``backend: unsloth`` only for a config that loads with it.

With unsloth installed, every run on ``backend: transformers`` was told to add
``backend: unsloth``. The check looked at the backend alone, so the tip was also
printed for configs the loader refuses the moment that line is added: a task
with no unsloth setup, ``freeze_layers`` on sft, ``moe_lora`` recipes. 35 of
the 184 shipped templates and recipes on transformers were among them.

The tip now asks the schema: the dumped config is validated again with
``backend: unsloth``, and the tip is printed only if that succeeds. No list in
``train.py`` has to follow the validators.

unsloth is not installed here; ``is_unsloth_available`` is stubbed.
"""

from __future__ import annotations

import builtins
import logging
import os
import warnings

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

import soup_cli.utils.unsloth as unsloth_mod
from soup_cli.cli import app
from soup_cli.commands.train import _loads_on_unsloth
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TEMPLATES, SoupConfig
from soup_cli.recipes.catalog import RECIPES
from tests.conftest import strip_ansi

TIP = "Tip: unsloth is installed"

_SHIPPED = {
    **{f"template:{name}": text for name, text in TEMPLATES.items()},
    **{f"recipe:{name}": recipe.yaml_str for name, recipe in RECIPES.items()},
}


def _raw(task="sft", **training):
    return {
        "base": "org/m", "task": task, "data": {"train": "./x.jsonl"}, "training": training,
    }


# A config on transformers, and whether the loader takes it with backend: unsloth.
_CASES = {
    "plain sft": (_raw(), True),
    "sft with a lora rank": (_raw(lora={"r": 8}), True),
    "freeze_layers on sft (#1581)": (_raw(freeze_layers=2), False),
    "rope_scaling_type on sft (#1697)": (_raw(rope_scaling_type="dynamic"), False),
    "a task with no unsloth setup (#1357)": (_raw("classifier", num_labels=2), False),
    "distill": (_raw("distill", teacher_model="org/t"), False),
}


def _loads(raw: dict) -> bool:
    """The oracle, independent of the helper: write the line and load the YAML."""
    try:
        load_config_from_string(yaml.safe_dump({**raw, "backend": "unsloth"}))
    except ValueError:
        return False
    return True


class TestTheHelper:
    @pytest.mark.parametrize("case", list(_CASES))
    def test_it_answers_what_the_loader_answers(self, case):
        raw, loads = _CASES[case]
        cfg = load_config_from_string(yaml.safe_dump(raw))

        assert _loads(raw) is loads, "the case table is out of step with the schema"
        assert _loads_on_unsloth(cfg) is loads

    def test_the_config_it_was_given_is_left_alone(self):
        cfg = load_config_from_string(yaml.safe_dump(_raw(freeze_layers=2)))
        before = cfg.model_dump()

        _loads_on_unsloth(cfg)

        assert cfg.backend == "transformers"
        assert cfg.model_dump() == before

    def test_only_a_validation_error_means_no(self, monkeypatch):
        """A bug in the dump or in a validator must surface, not hide the tip."""
        cfg = load_config_from_string(yaml.safe_dump(_raw()))

        def broken(cls, *args, **kwargs):
            raise RuntimeError("not a validation error")

        monkeypatch.setattr(SoupConfig, "model_validate", classmethod(broken))

        with pytest.raises(RuntimeError, match="not a validation error"):
            _loads_on_unsloth(cfg)

    def test_a_validation_error_is_what_a_refusal_raises(self):
        """Guard on the test above: the refusals are ValidationErrors, so
        catching only that still covers every one of them."""
        cfg = load_config_from_string(yaml.safe_dump(_raw(freeze_layers=2)))

        with pytest.raises(ValidationError):
            SoupConfig.model_validate({**cfg.model_dump(), "backend": "unsloth"})


@pytest.mark.parametrize("name", list(_SHIPPED))
class TestEveryShippedConfig:
    def test_the_tip_agrees_with_the_loader(self, name):
        """A template or recipe either gets the tip and loads with
        ``backend: unsloth``, or gets neither."""
        text = _SHIPPED[name]
        cfg = load_config_from_string(text)
        if cfg.backend != "transformers":
            pytest.skip("already on another backend: the tip is not offered")

        assert _loads_on_unsloth(cfg) is _loads(yaml.safe_load(text))

    def test_the_second_validation_is_silent_and_stays_off_the_disk(
        self, name, monkeypatch, capsys, caplog
    ):
        cfg = load_config_from_string(_SHIPPED[name])
        capsys.readouterr()
        caplog.clear()

        touched = []

        def spy(name, real):
            def call(*args, **kwargs):
                touched.append((name, args[:1]))
                return real(*args, **kwargs)

            return call

        for module, attr in (
            (builtins, "open"), (os, "stat"), (os, "lstat"), (os, "listdir"),
            (os, "scandir"), (os, "mkdir"), (os, "makedirs"), (os, "remove"),
        ):
            monkeypatch.setattr(module, attr, spy(attr, getattr(module, attr)))
        try:
            with caplog.at_level(logging.DEBUG), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _loads_on_unsloth(cfg)
        finally:
            monkeypatch.undo()  # before pytest itself needs the disk again

        out = capsys.readouterr()
        assert touched == []
        assert (out.out, out.err) == ("", "")
        assert [str(w.message) for w in caught] == []
        assert [record.getMessage() for record in caplog.records] == []


def test_some_shipped_configs_are_on_each_side():
    """Guard on the sweep: it must really see both answers."""
    answers = [
        _loads_on_unsloth(cfg)
        for cfg in (load_config_from_string(text) for text in _SHIPPED.values())
        if cfg.backend == "transformers"
    ]

    assert answers.count(True) > 100
    assert answers.count(False) > 20


def _dry_run(tmp_path, monkeypatch, raw, *, installed=True) -> str:
    monkeypatch.setattr(unsloth_mod, "is_unsloth_available", lambda: installed)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.jsonl").write_text('{"instruction": "q", "output": "a"}\n', encoding="utf-8")
    (tmp_path / "soup.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    # A crash would also leave the tip out; it must not pass for that reason.
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    return " ".join(strip_ansi(result.output).split())


class TestSoupTrain:
    @pytest.mark.parametrize("case", list(_CASES))
    def test_the_tip_is_printed_only_for_a_config_that_would_load(
        self, tmp_path, monkeypatch, case
    ):
        raw, loads = _CASES[case]

        assert (TIP in _dry_run(tmp_path, monkeypatch, raw)) is loads

    def test_without_unsloth_installed_nothing_is_said(self, tmp_path, monkeypatch):
        assert TIP not in _dry_run(tmp_path, monkeypatch, _raw(), installed=False)

    def test_without_unsloth_installed_the_config_is_not_validated_again(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.commands.train as train_mod

        def unexpected(cfg):
            raise AssertionError("validated again although unsloth is not installed")

        monkeypatch.setattr(train_mod, "_loads_on_unsloth", unexpected)

        assert TIP not in _dry_run(tmp_path, monkeypatch, _raw(), installed=False)

    def test_a_run_already_on_unsloth_gets_no_tip(self, tmp_path, monkeypatch):
        out = _dry_run(tmp_path, monkeypatch, {**_raw(), "backend": "unsloth"})

        assert TIP not in out
