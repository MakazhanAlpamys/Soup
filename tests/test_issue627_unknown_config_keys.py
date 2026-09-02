"""Unknown config keys must not be silently dropped (#627).

None of the 9 config models set ``extra="forbid"``, so Pydantic's default
``extra="ignore"`` applied everywhere: a misspelled or not-yet-released key
validated clean, printed "Config valid. Ready to train!" under
``soup train --dry-run``, and was discarded.

The keys that reach users are not exotic. ``quantizaton``,
``gradient_checkpoint``, ``lr_scheduler`` and ``max_len`` are each one edit from
a real field, so the run keeps the default quantization / does no checkpointing
/ uses the default schedule while the operator believes otherwise. Exit 0,
plausible logs, and the requested thing silently not done.

#623 is the live example: ``training.stream_pin`` landed on main two days after
0.73.3 shipped, a user on the released wheel wrote the documented escape hatch,
``--dry-run`` called it valid, the key was dropped, and the resulting OOM was
diagnosed as a layer-streaming bug across two long comments.

Every test here is CPU-only: no GPU, no downloads, no model load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.config.unknown_keys import (
    UNKNOWN_KEY_REJECTION_VERSION,
    find_unknown_config_keys,
    format_unknown_keys,
)

_VALID = """
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
output: ./o
"""


def _raw(extra_data: str = "", extra_training: str = "") -> dict:
    import yaml

    doc = yaml.safe_load(_VALID)
    if extra_data:
        doc["data"].update(yaml.safe_load(extra_data))
    if extra_training:
        doc["training"] = yaml.safe_load(extra_training)
    return doc


class TestTheFiveReportedCases:
    """Each of the silently-accepted keys from the issue, pinned by name."""

    @pytest.mark.parametrize(
        "key,value,expected_suggestion",
        [
            ("quantizaton", "4bit", "quantization"),
            ("gradient_checkpoint", True, "gradient_checkpointing"),
            ("lr_scheduler", "cosine", "scheduler"),
            ("stream_pin_typo", False, "stream_pin"),
        ],
    )
    def test_training_key_is_reported_with_a_suggestion(
        self, key, value, expected_suggestion
    ) -> None:
        import yaml

        doc = yaml.safe_load(_VALID)
        doc["training"] = {"epochs": 1, key: value}
        unknown = find_unknown_config_keys(doc)

        assert [u.key for u in unknown] == [key]
        assert unknown[0].path == f"training.{key}"
        # The suggestion is the point -- "unknown field 'quantizaton'" alone is
        # much less useful than naming the field the user meant.
        assert expected_suggestion in unknown[0].suggestions

    def test_data_key_is_reported(self) -> None:
        """A different model: a fix guarding training only must fail here."""
        unknown = find_unknown_config_keys(_raw(extra_data="{max_len: 512}"))
        assert [u.path for u in unknown] == ["data.max_len"]
        assert "max_length" in unknown[0].suggestions


class TestNesting:
    """Each model in the tree is walked, not just the top level."""

    def test_unknown_key_inside_lora_is_found(self) -> None:
        unknown = find_unknown_config_keys(
            _raw(extra_training="{epochs: 1, lora: {r: 8, alfa: 16}}")
        )
        assert [u.path for u in unknown] == ["training.lora.alfa"]
        assert "alpha" in unknown[0].suggestions

    def test_unknown_key_at_the_top_level_is_found(self) -> None:
        raw = _raw()
        raw["taks"] = "sft"
        assert "taks" in [u.key for u in find_unknown_config_keys(raw)]

    def test_several_unknowns_are_all_reported(self) -> None:
        raw = _raw(extra_data="{max_len: 512}", extra_training="{epochs: 1, quantizaton: 4bit}")
        paths = sorted(u.path for u in find_unknown_config_keys(raw))
        assert paths == ["data.max_len", "training.quantizaton"]


class TestTheControl:
    """The guard must not be satisfiable by rejecting everything."""

    def test_a_valid_config_reports_nothing(self) -> None:
        assert find_unknown_config_keys(_raw()) == []

    def test_a_config_using_many_real_keys_reports_nothing(self) -> None:
        import yaml

        doc = yaml.safe_load(_VALID)
        doc["data"].update({"max_length": 2048, "val_split": 0.1})
        doc["training"] = {
            "epochs": 3,
            "lr": 5e-6,
            "batch_size": "auto",
            "gradient_accumulation_steps": 8,
            "quantization": "4bit",
            "gradient_checkpointing": True,
            "dpo_beta": 0.1,
            "moe_lora": True,
            "lora": {"r": 16, "alpha": 32, "target_modules": "auto"},
        }
        assert find_unknown_config_keys(doc) == []

    def test_every_real_recipe_in_the_catalog_is_clean(self) -> None:
        """The strongest control available: 160 shipped configs, none flagged."""
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        offenders = {}
        for name, recipe in RECIPES.items():
            unknown = find_unknown_config_keys(yaml.safe_load(recipe.yaml_str))
            if unknown:
                offenders[name] = [u.path for u in unknown]
        assert offenders == {}

    def test_non_mapping_values_do_not_crash_the_walk(self) -> None:
        raw = _raw()
        raw["training"] = "not-a-mapping"
        find_unknown_config_keys(raw)  # must not raise

    def test_empty_and_none_sections_are_tolerated(self) -> None:
        raw = _raw()
        raw["training"] = None
        raw["eval"] = {}
        find_unknown_config_keys(raw)  # must not raise


class TestTheMessage:
    def test_message_names_the_path_and_the_suggestion(self) -> None:
        unknown = find_unknown_config_keys(
            _raw(extra_training="{epochs: 1, quantizaton: 4bit}")
        )
        msg = format_unknown_keys(unknown)
        assert "training.quantizaton" in msg
        assert "quantization" in msg
        assert "did you mean" in msg.lower()

    def test_a_key_with_no_close_match_still_reports_cleanly(self) -> None:
        unknown = find_unknown_config_keys(
            _raw(extra_training="{epochs: 1, zzzzzzzz: 1}")
        )
        assert unknown[0].suggestions == ()
        msg = format_unknown_keys(unknown)
        assert "training.zzzzzzzz" in msg
        assert "did you mean" not in msg.lower()


class TestEveryConstructionSiteIsGuarded:
    """A fourth ``SoupConfig(**...)`` must not be able to appear unguarded.

    The point of #627 is that nothing *reminds* a caller to check. A scanner is
    the only guard that survives someone adding a new entry point next year --
    the same shape as ``test_no_second_hand_rolled_prompt_remains_in_the_serve_backends``.
    """

    def test_all_soupconfig_construction_sites_check_unknown_keys(self) -> None:
        import re
        from pathlib import Path

        src = Path(__file__).parents[1] / "src" / "soup_cli"
        offenders = []
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if not re.search(r"SoupConfig\(\*\*", text):
                continue
            if "find_unknown_config_keys" not in text:
                offenders.append(str(path.relative_to(src)))
        assert offenders == [], (
            f"these build a SoupConfig without checking for unknown keys: {offenders}"
        )


class TestTheDeadline:
    """The warning has to be a deadline, not a decoration (#627).

    The maintainer's call on the thread was **option 3 -- warn now, forbid in
    the next minor**, and the reasoning is that a warning with no expiry is
    permanent. That only holds if the version is (a) named in the message,
    (b) stated in one place, and (c) checked against the version this tree
    actually declares.

    (c) is the part that needs a test rather than a convention. A version typed
    into a message survives the release it names and starts lying to users --
    still warning, while claiming the rejection already arrived. So the deadline
    is asserted against ``soup_cli.__version__`` here: while the tree is below
    the deadline the switch must read ``"warn"``, and the release that crosses
    it turns this test red until someone flips the switch.

    Every assertion below reads :data:`UNKNOWN_KEY_REJECTION_VERSION`; none of
    them repeats the number, which is the property being defended.
    """

    def _msg_for_one_typo(self) -> str:
        return format_unknown_keys(
            find_unknown_config_keys(_raw(extra_training="{epochs: 1, quantizaton: 4bit}"))
        )

    def test_the_warning_names_the_version_rather_than_a_vague_future(self) -> None:
        msg = self._msg_for_one_typo()
        assert f"v{UNKNOWN_KEY_REJECTION_VERSION}" in msg
        # "a future release" reads as "never" and gives a user nothing to decide
        # on, which is the whole reason the version is named.
        assert "future release" not in msg.lower()

    def test_the_version_is_written_out_in_exactly_one_source_file(self) -> None:
        """Derived, not duplicated: a second copy is what falls out of step.

        The two spellings a duplicate would take are both matched: the
        ``v``-prefixed one a hand-written message would use (``v0.75``), and the
        quoted one a second constant would use (``"0.75"``).

        It deliberately does **not** match a bare ``0.75``. The maintainer moved
        the deadline from 0.74 to 0.75 mid-review, and this test -- which had
        been scanning for the bare number -- went red on
        ``schema.py`` (``freeze_ratio``: "0.75 = freeze 75%"),
        ``adapter_scan.py`` (``_ENERGY_TOP1_WARN = 0.75``) and two more. A
        deadline is a version, ordinary ratios are not, and a guard that a
        routine deadline move turns red is a guard people delete. Its passing on
        0.74 was luck: that number happened to appear nowhere.

        ``__init__.py`` is exempt because it holds the *declared* version, a
        different fact that will legitimately equal the deadline -- as a quoted
        string -- the moment the deadline release ships. Found by mutation:
        bumping ``__version__`` to the deadline reddened this test as well as
        the one that should fire, burying the real signal under a false one.

        ``as_posix()`` rather than ``str()``: the first version compared
        ``str(path)`` against a ``/`` literal and passed on macOS and Ubuntu
        while failing all three Windows jobs on ``config\\unknown_keys.py``.
        The separator is the platform's; the expectation should not be.
        """
        import re
        from pathlib import Path

        version = re.escape(UNKNOWN_KEY_REJECTION_VERSION)
        # v-prefixed (a message) or quoted (a constant); never a bare float.
        pattern = re.compile(rf"""v{version}\b|["']{version}["']""")
        src = Path(__file__).parents[1] / "src" / "soup_cli"
        holders = sorted(
            p.relative_to(src).as_posix()
            for p in src.rglob("*.py")
            if p.name != "__init__.py" and pattern.search(p.read_text(encoding="utf-8"))
        )
        assert holders == ["config/unknown_keys.py"], (
            f"the deadline version is written out in more than one place: {holders}"
        )

    def test_the_deadline_is_enforced_against_the_declared_version(self) -> None:
        """The release that crosses the deadline must flip the switch.

        Asserted against the declared bound rather than a literal, so a slipped
        or an arrived release is caught here instead of in a user's log.
        """
        from soup_cli import __version__
        from soup_cli.config import loader

        declared = tuple(int(p) for p in __version__.split(".")[:2])
        deadline = tuple(int(p) for p in UNKNOWN_KEY_REJECTION_VERSION.split(".")[:2])

        if declared < deadline:
            assert loader.UNKNOWN_KEY_SEVERITY == "warn", (
                f"v{__version__} is before the v{UNKNOWN_KEY_REJECTION_VERSION} "
                "deadline, so unknown keys must warn, not refuse"
            )
        else:
            assert loader.UNKNOWN_KEY_SEVERITY == "error", (
                f"v{__version__} has reached the v{UNKNOWN_KEY_REJECTION_VERSION} "
                "deadline promised in the warning: set UNKNOWN_KEY_SEVERITY to "
                "'error' (and drop the deadline sentence), or move the deadline "
                "deliberately -- the message is currently promising a rejection "
                "that does not happen"
            )

    def test_the_deadline_is_dropped_once_the_switch_rejects(self) -> None:
        """Under ``"error"`` the message must not promise a future rejection."""
        unknown = find_unknown_config_keys(_raw(extra_training="{epochs: 1, quantizaton: 4bit}"))
        assert UNKNOWN_KEY_REJECTION_VERSION not in format_unknown_keys(
            unknown, include_deadline=False
        )

    def test_the_docs_state_the_same_deadline(self) -> None:
        """A dated warning is only worth having if the date is findable.

        Both facts have to land in the *same* Markdown section. Whole-file
        containment was the first version of this and a mutation walked
        through it: stripping the deadline out of the prose left a ``v0.75``
        elsewhere in the file, and the test stayed green while the section a
        reader actually lands on no longer said when the rejection arrives.
        """
        from pathlib import Path

        root = Path(__file__).parents[1]
        version = f"v{UNKNOWN_KEY_REJECTION_VERSION}"
        for rel in ("README.md", "docs/backends-and-ops.md"):
            text = (root / rel).read_text(encoding="utf-8")
            sections: list[list[str]] = [[]]
            for line in text.splitlines():
                if line.startswith("#"):
                    sections.append([])
                sections[-1].append(line)
            bodies = ["\n".join(s) for s in sections]
            assert any(
                version in body and "unknown config key" in body.lower()
                for body in bodies
            ), (
                f"{rel} has no section that both names the {version} deadline and "
                "says what expires -- one without the other is not a deadline"
            )


class TestTheDocumentedExamplesAreOnesThatActuallyBreak:
    """A documented example whose typo changes nothing teaches the wrong lesson.

    The first draft of these docs claimed ``quantizaton: 4bit`` "trained in full
    precision". It does not. ``quantization`` *defaults* to ``"4bit"``, so that
    particular typo is the harmless member of the population this fix exists for
    -- silently dropped and silently identical. Running the CLI caught it; the
    diff never would have, because the sentence reads plausibly.

    So each example is pinned to the schema default it contradicts. The property
    is "the value the user wrote differs from what they get when it is dropped",
    which is the only thing that makes an example worth printing, and it is a
    property a later schema change can quietly destroy.
    """

    def _defaults(self) -> dict:
        from soup_cli.config.schema import DataConfig, TrainingConfig

        training = TrainingConfig()
        data = DataConfig(train="./t.jsonl")
        return {
            "training.quantization": training.quantization,
            "training.gradient_checkpointing": training.gradient_checkpointing,
            "data.max_length": data.max_length,
        }

    #: ``real field -> the value the docs show a user writing (typo'd)``.
    INTENDED = {
        "training.quantization": "none",
        "training.gradient_checkpointing": True,
        "data.max_length": 512,
    }

    def test_each_documented_example_asks_for_something_the_default_is_not(self) -> None:
        defaults = self._defaults()
        vacuous = {
            field: value
            for field, value in self.INTENDED.items()
            if defaults[field] == value
        }
        assert vacuous == {}, (
            "these documented examples are indistinguishable from the schema "
            f"default, so dropping them changes nothing: {vacuous}"
        )

    def test_the_typo_the_docs_show_really_is_unknown(self) -> None:
        """The misspelling has to be one the walk actually flags."""
        raw = _raw(extra_training="{epochs: 1, quantizaton: none}")
        paths = [u.path for u in find_unknown_config_keys(raw)]
        assert paths == ["training.quantizaton"]

    def test_the_docs_do_not_still_carry_the_corrected_claim(self) -> None:
        """The wrong version was a specific sentence; pin it out of the tree.

        Prose repeats. The first correction fixed the docs and the changelog
        fragment and left the same false sentence in the module docstring that
        explains the fix, so the scan covers ``src/`` too -- a claim is no less
        wrong for being in a docstring. This file is excluded because the class
        above quotes the wrong example deliberately, as the thing being pinned.
        """
        from pathlib import Path

        root = Path(__file__).parents[1]
        named = [root / rel for rel in ("README.md", "docs/backends-and-ops.md", "CHANGELOG.md")]
        # The fragment too: it becomes CHANGELOG.md at release, so a claim
        # corrected only in the docs would come back at assembly time.
        candidates = (
            named
            + sorted((root / "changelog.d").rglob("*.md"))
            + sorted((root / "src" / "soup_cli").rglob("*.py"))
        )
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            assert "quantizaton: 4bit" not in text, (
                f"{path.relative_to(root)} still shows `quantizaton: 4bit` as a "
                "harmful typo; quantization defaults to 4bit, so that example "
                "changes nothing"
            )
            assert "trains in full precision" not in text, (
                f"{path.relative_to(root)} still claims a dropped quantization "
                "key means full precision; the default is 4bit, so it does not"
            )


class TestOneReportPerLoad:
    """Four typos, one panel -- the maintainer's call on #627.

    A per-key report is worse in exactly the case that matters: a config
    copied from a newer Soup trips several keys at once, and N panels bury the
    list they are supposed to present.
    """

    def _four_typos(self) -> list:
        raw = _raw(extra_data="{max_len: 512, val_splt: 0.1}")
        raw["training"] = {"epochs": 1, "quantizaton": "4bit", "gradient_checkpoint": True}
        return find_unknown_config_keys(raw)

    def test_every_unknown_key_appears_in_one_message(self) -> None:
        msg = format_unknown_keys(self._four_typos())
        for path in (
            "data.max_len",
            "data.val_splt",
            "training.quantizaton",
            "training.gradient_checkpoint",
        ):
            assert path in msg

    def test_the_deadline_sentence_appears_once_not_once_per_key(self) -> None:
        msg = format_unknown_keys(self._four_typos())
        assert msg.count(f"v{UNKNOWN_KEY_REJECTION_VERSION}") == 1

    def test_the_loader_prints_a_single_warning_block(self) -> None:
        """Four unknown keys must not produce four ``Warning:`` headers."""
        from soup_cli.config import loader

        printed: list[str] = []
        original = loader.console.print
        loader.console.print = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
        try:
            loader._report_unknown_keys(
                {
                    "base": "hf/model",
                    "task": "sft",
                    "data": {"train": "./t.jsonl", "format": "auto", "max_len": 512},
                    "training": {"epochs": 1, "quantizaton": "4bit"},
                    "output": "./o",
                }
            )
        finally:
            loader.console.print = original

        assert sum("Warning:" in line for line in printed) == 1


class TestTheSweepGuardIsIndependentOfTheDeadline:
    """``sweep.py`` raises whatever the switch says, so it makes no promise.

    A swept parameter that names no config field produces arms that are all
    identical -- there is no salvageable result to preserve compatibility for,
    which is a different failure class from a dropped training key.
    """

    def _swept(self, param: str, value: object) -> dict:
        """A config dict as ``_run_single`` builds it: dump plus one --param."""
        from soup_cli.commands.sweep import _set_nested_param

        config_dict = _raw()
        config_dict["experiment_name"] = "sweep-run-1"
        _set_nested_param(config_dict, param, value)
        return config_dict

    def test_a_parameter_naming_no_field_raises_before_the_first_arm(self) -> None:
        from soup_cli.commands.sweep import _reject_unknown_sweep_params

        with pytest.raises(ValueError) as excinfo:
            _reject_unknown_sweep_params(self._swept("lora_rank", 8))
        assert "lora_rank" in str(excinfo.value)
        assert "does not match any config field" in str(excinfo.value)

    def test_a_nested_parameter_naming_no_field_raises_too(self) -> None:
        """``--param training.lr_rate=...`` must not slip past a top-level check."""
        from soup_cli.commands.sweep import _reject_unknown_sweep_params

        with pytest.raises(ValueError, match="training.lr_rate"):
            _reject_unknown_sweep_params(self._swept("training.lr_rate", 1e-5))

    def test_the_error_does_not_promise_a_future_rejection(self) -> None:
        """It raises today, so a v-next deadline sentence would be false."""
        from soup_cli.commands.sweep import _reject_unknown_sweep_params

        with pytest.raises(ValueError) as excinfo:
            _reject_unknown_sweep_params(self._swept("lora_rank", 8))
        assert UNKNOWN_KEY_REJECTION_VERSION not in str(excinfo.value)

    def test_it_raises_even_though_the_loader_only_warns(self) -> None:
        """The switch governs the loader, not the sweep -- pinned, not assumed."""
        from soup_cli.commands.sweep import _reject_unknown_sweep_params
        from soup_cli.config import loader

        assert loader.UNKNOWN_KEY_SEVERITY == "warn"
        with pytest.raises(ValueError):
            _reject_unknown_sweep_params(self._swept("lora_rank", 8))

    def test_a_real_swept_parameter_is_not_rejected(self) -> None:
        """The control: a guard that rejects every sweep would pass the above."""
        from soup_cli.commands.sweep import _reject_unknown_sweep_params

        for param, value in (
            ("training.lr", 1e-5),
            ("training.lora.r", 16),
            ("training.epochs", 3),
        ):
            _reject_unknown_sweep_params(self._swept(param, value))  # must not raise

    def test_run_single_refuses_before_it_imports_the_training_stack(self) -> None:
        """The seam is only worth having while the caller uses it.

        This asserted on ``inspect.getsource`` until the #628 review: it
        checked that a string appeared in the function, which is not the same
        as the function refusing anything. It calls ``_run_single`` now.

        The heavy modules are poisoned rather than counted in ``sys.modules``.
        A first version asserted ``"torch" not in sys.modules`` after the call,
        which silently passed whenever an earlier test had already imported
        torch -- it let the "guard moved back below the imports" mutation live.
        Blocking the imports makes the ordering the *only* thing that decides
        the outcome: guard first gives ``ValueError``, guard second gives
        ``ImportError``, in any test order and on a machine with no GPU stack.
        """
        import sys

        from soup_cli.commands.sweep import _run_single
        from soup_cli.config.schema import SoupConfig

        base_cfg = SoupConfig(**_raw())
        heavy = (
            "soup_cli.data.loader",
            "soup_cli.experiment.tracker",
            "soup_cli.monitoring.display",
            "soup_cli.trainer.sft",
            "soup_cli.utils.gpu",
        )
        saved = {name: sys.modules.get(name) for name in heavy}
        for name in heavy:
            sys.modules[name] = None  # a None entry makes `import name` raise
        try:
            with pytest.raises(ValueError, match="does not match any config field"):
                _run_single(base_cfg, {"lora_rank": 8}, "sweep_1", Path("soup.yaml"))
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


class TestATypodSweepFailsTheCommandAndNotJustTheArm:
    """The guard raising is not the same as the sweep failing (#628 review).

    ``_reject_unknown_sweep_params`` was called from inside ``_run_single``,
    and the arm loop wraps every call in ``except Exception``. So the guard
    fired, the loop caught it, each arm was recorded ``failed``, the results
    table printed, and the command exited **0** -- the exact #627 shape (exit
    0, plausible output, the requested thing not done) reproduced one layer
    above the fix for it. No CI job or script could detect an entirely invalid
    sweep.

    Every test in the class above calls the guard directly, so all of them
    passed while this was broken. That is the gap: they tested the guard and
    never the caller around it.
    """

    def _config(self, tmp_path) -> str:
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(
            "base: test-model\n"
            "data:\n"
            "  train: ./data.jsonl\n"
        )
        return str(cfg)

    def test_a_typod_sweep_exits_non_zero(self, tmp_path) -> None:
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "sweep",
            "--config", self._config(tmp_path),
            "--param", "learnig_rate=1e-5,2e-5",
            "--yes",
        ])
        assert result.exit_code != 0, (
            "a sweep whose only swept parameter names no config field exited "
            f"{result.exit_code}; nothing downstream can detect that"
        )

    def test_it_names_the_parameter_it_refused(self, tmp_path) -> None:
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "sweep",
            "--config", self._config(tmp_path),
            "--param", "learnig_rate=1e-5,2e-5",
            "--yes",
        ])
        assert "learnig_rate" in result.output

    def test_no_arm_is_started_and_no_results_table_is_printed(self, tmp_path) -> None:
        """The fragment claims it raises 'before the first arm starts'."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "sweep",
            "--config", self._config(tmp_path),
            "--param", "learnig_rate=1e-5,2e-5",
            "--yes",
        ])
        assert "--- Run 1/2" not in result.output, "an arm started despite the guard"
        assert "failed" not in result.output.lower(), (
            "a per-arm failure table means the loop swallowed the guard"
        )

    def test_a_valid_sweep_reaches_the_first_arm(self, tmp_path) -> None:
        """Control: a precheck that refused every sweep would pass the above.

        It must not use ``--dry-run``: that branch returns *before* the
        precheck, so a dry run exercises none of it and would pass against a
        reject-everything precheck. The arm banner is the evidence the grid
        was let through -- the run itself then fails on the absent dataset,
        which is downstream of what this pins.
        """
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "sweep",
            "--config", self._config(tmp_path),
            "--param", "training.lr=1e-5",
            "--yes",
        ])
        assert "does not match any config field" not in result.output, (
            "the precheck refused a real config field"
        )
        assert "--- Run 1/1" in result.output, (
            "the sweep never reached its first arm, so the precheck blocked it"
        )
