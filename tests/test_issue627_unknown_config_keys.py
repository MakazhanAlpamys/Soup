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


    def test_every_bundled_fetch_example_is_clean(self) -> None:
        """``soup fetch examples`` ships configs too, and two of them were not.

        Both bundled YAML examples carried a top-level ``lora:`` block that the
        schema has never had (it lives under ``training``), so their LoRA
        settings had never applied -- and once the loader refuses unknown keys
        (v0.75.0) both would have refused to load. The recipe catalog had this
        guard; the fetch catalog did not. Loading through the real loader is
        the second half: a clean scan of a config that then fails validation
        would still be a broken example.
        """
        import os

        import yaml

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.fetch_examples import fetch_examples_dir, list_entries

        entries = [e for e in list_entries("examples").values() if e.filename.endswith(".yaml")]
        assert len(entries) >= 2, "the bundled example catalog shrank"
        for entry in entries:
            text = Path(os.path.join(fetch_examples_dir(), entry.filename)).read_text(
                encoding="utf-8"
            )
            unknown = find_unknown_config_keys(yaml.safe_load(text))
            assert unknown == [], (entry.name, [u.path for u in unknown])
            assert load_config_from_string(text).training.lora.r == 16, entry.name
            # The remap makes a root-level ``lora:`` LEGAL, so the two checks
            # above cannot see the canonical-spelling fix; pin the text itself.
            assert "lora" not in yaml.safe_load(text), (
                f"{entry.name}: the bundled example spells lora: at the root; "
                "the canonical spelling is training.lora"
            )
    def test_non_mapping_values_do_not_crash_the_walk(self) -> None:
        raw = _raw()
        raw["training"] = "not-a-mapping"
        find_unknown_config_keys(raw)  # must not raise

    def test_empty_and_none_sections_are_tolerated(self) -> None:
        raw = _raw()
        raw["training"] = None
        raw["eval"] = {}
        find_unknown_config_keys(raw)  # must not raise


class TestNonStringKeysAreSkippedNotCrashedOn:
    """The ``isinstance(key, str)`` guard in ``_walk`` (#628), which had no test.

    YAML permits non-string mapping keys. Without the guard the key reaches
    ``difflib.get_close_matches``, which iterates it:

        get_close_matches(1,    [...])  -> TypeError: 'int' object is not iterable
        get_close_matches(True, [...])  -> TypeError: 'bool' object is not iterable

    So a config carrying ``1:`` or ``true:`` would kill ``soup train --config``
    with a bare TypeError raised from inside the unknown-key reporter — the
    code whose whole purpose is to turn a confusing failure into an actionable
    one. Measured as the single uncovered statement in ``unknown_keys.py``.
    """

    def test_python_collapses_true_and_one_into_one_key(self) -> None:
        """Pin the fixture's own premise before relying on it.

        ``True == 1`` in Python, so a mapping with both ``1:`` and ``true:``
        holds ONE entry, not two. A test written without noticing this would
        claim to cover two non-string key types while covering one.
        """
        import yaml

        collapsed = yaml.safe_load("1: numeric\ntrue: boolean\n")
        assert list(collapsed) == [1]
        assert type(next(iter(collapsed))) is int

    def test_a_numeric_key_does_not_raise(self) -> None:
        doc = _raw(extra_training="1: numeric-key\nepochs: 3\n")
        assert find_unknown_config_keys(doc) == []

    def test_a_boolean_key_does_not_raise(self) -> None:
        """Constructed directly: YAML would collapse ``true:`` onto ``1:``."""
        doc = _raw(extra_training="epochs: 3\n")
        doc["training"][True] = "boolean-key"
        assert find_unknown_config_keys(doc) == []

    def test_a_non_string_key_does_not_mask_a_real_unknown_key(self) -> None:
        """The guard must ``continue``, not abort the loop.

        This is the assertion that would survive a guard rewritten as an early
        ``return``: the typo sits AFTER the non-string key in insertion order,
        so it is only reported if the walk kept going.
        """
        doc = _raw(extra_training="1: numeric-key\nepocs: 3\n")
        found = find_unknown_config_keys(doc)

        assert [u.path for u in found] == ["training.epocs"]
        assert "epochs" in found[0].suggestions

    def test_a_non_string_key_nested_in_a_subsection_does_not_raise(self) -> None:
        """``_walk`` recurses; the guard has to hold on the way down too."""
        doc = _raw(extra_training="epochs: 3\n")
        doc["training"]["lora"] = {2: "numeric-key", "r": 16, "alpah": 32}
        found = find_unknown_config_keys(doc)

        assert [u.path for u in found] == ["training.lora.alpah"]
        assert "alpha" in found[0].suggestions

    def test_control_string_keys_are_still_reported(self) -> None:
        """Reject-everything control: the walk has not simply stopped working."""
        doc = _raw(extra_training="epocs: 3\n")
        assert [u.path for u in find_unknown_config_keys(doc)] == ["training.epocs"]


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
    list they are supposed to present. The same class pins what the loader
    does with that one report on each side of the severity switch: return it
    (``"error"``) or print it once (``"warn"``).
    """

    TWO_TYPOS = {
        "base": "hf/model",
        "task": "sft",
        "data": {"train": "./t.jsonl", "format": "auto", "max_len": 512},
        "training": {"epochs": 1, "quantizaton": "4bit"},
        "output": "./o",
    }

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

    def test_the_loader_returns_a_single_refusal_naming_every_key(self, monkeypatch) -> None:
        """Under the shipped ``"error"`` switch: one message, nothing printed.

        Printing is the caller's job on this branch (``SystemExit`` for the
        CLI, ``ValueError`` for the API), so a report that printed *and*
        returned would show the operator the same list twice.
        """
        from soup_cli.config import loader

        printed: list[str] = []
        monkeypatch.setattr(
            loader.console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
        )
        message = loader._report_unknown_keys(dict(self.TWO_TYPOS))

        assert printed == []
        assert message is not None
        assert "data.max_len" in message and "training.quantizaton" in message
        # The refusal must not say the run proceeded without the key.
        assert "not applied" not in message.lower()
        assert "Refused." in message

    def test_the_loader_prints_a_single_warning_block_under_warn(self, monkeypatch) -> None:
        """The other side of the switch stays pinned: four keys, one ``Warning:``."""
        from soup_cli.config import loader

        printed: list[str] = []
        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", "warn")
        monkeypatch.setattr(
            loader.console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
        )
        assert loader._report_unknown_keys(dict(self.TWO_TYPOS)) is None
        assert sum("Warning:" in line for line in printed) == 1
        assert sum("Not applied." in line for line in printed) == 1


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

    @pytest.mark.parametrize("severity", ["warn", "error"])
    def test_it_raises_whatever_the_loader_switch_says(self, monkeypatch, severity) -> None:
        """The switch governs the loader, not the sweep -- pinned on both sides.

        v0.74 shipped the loader at ``"warn"`` and v0.75 at ``"error"``; the
        sweep guard raised under both, and this parametrisation is what keeps
        that true rather than assumed.
        """
        from soup_cli.commands.sweep import _reject_unknown_sweep_params
        from soup_cli.config import loader

        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", severity)
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

    def test_an_empty_grid_does_not_crash_the_probe(self, tmp_path) -> None:
        """The probe reads ``combinations[0]``, which an empty grid does not have.

        ``--max-runs -1`` empties the grid. Before the guard this raised an
        unhandled ``IndexError`` from inside the new precheck -- a regression
        this PR introduced, found in the #628 review. Nonsense input either
        way, but main printed an empty table and exited 0, and an unhandled
        traceback is a worse answer than that. Bounding ``--max-runs`` at
        ``ge=1`` would be the better fix and is pre-existing behaviour, so it
        is left alone here.
        """
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "sweep",
            "--config", self._config(tmp_path),
            "--param", "training.lr=1e-5",
            "--max-runs", "-1",
            "--yes",
        ])
        assert "IndexError" not in result.output, result.output
        assert result.exit_code == 0, result.output

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


class TestTheLoaderIsActuallyWiredUp:
    """The half every ``soup train`` user hits, tested through its callers.

    The #628 review reverted both call sites in ``config/loader.py`` -- leaving
    ``unknown_keys.py``, ``_report_unknown_keys`` and the import in place --
    and the whole suite stayed green: 43 passed while a config carrying
    ``quantizaton: none`` loaded silently, which is #627 restored exactly.

    Nothing saw it because ``TestOneReportPerLoad`` calls ``_report_unknown_keys``
    directly, and the construction-site scanner is a regex the surviving
    top-level import satisfies on its own. The scanner is still worth having --
    it catches a *fourth* call site appearing -- but it is not a test of these
    two. Same shape as the ``inspect.getsource`` gap on the sweep side; this is
    the third instance and the only user-visible one.

    Each call site is exercised separately: they are separate branches with
    separate contracts (``SystemExit`` for the CLI, ``ValueError`` for the
    API/UI), and a test of one does not cover the other.
    """

    TYPOD = (
        "base: test-model\n"
        "data:\n"
        "  train: ./data.jsonl\n"
        "training:\n"
        "  epochs: 1\n"
        "  quantizaton: none\n"
    )
    CLEAN = (
        "base: test-model\n"
        "data:\n"
        "  train: ./data.jsonl\n"
        "training:\n"
        "  epochs: 1\n"
    )

    @staticmethod
    def _recording_console(monkeypatch) -> list:
        """Record what the loader prints without matching against wrapped ANSI."""
        from soup_cli.config import loader

        printed: list[str] = []

        def record(*args, **kwargs) -> None:
            printed.append(str(args[0]) if args else "")

        monkeypatch.setattr(loader.console, "print", record)
        return printed

    def test_load_config_from_string_refuses_an_unknown_key(self, monkeypatch) -> None:
        """The shipped switch: the API/UI call site raises, and names the key."""
        from soup_cli.config.loader import load_config_from_string

        printed = self._recording_console(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(self.TYPOD)
        assert "quantizaton" in str(excinfo.value), "the refusal never named the typo"
        assert "unknown config key" in str(excinfo.value)
        # A refusal that still promised a *future* rejection would be lying,
        # and one that said "not applied" would read as if the run went ahead.
        assert UNKNOWN_KEY_REJECTION_VERSION not in str(excinfo.value)
        assert "Refused." in str(excinfo.value)
        assert "not applied" not in str(excinfo.value).lower()
        assert printed == [], "the API call site must raise, not print"

    def test_load_config_from_string_warns_under_warn_severity(self, monkeypatch) -> None:
        """The pre-v0.75 branch stays pinned: report, name the deadline, proceed."""
        from soup_cli.config import loader

        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", "warn")
        printed = self._recording_console(monkeypatch)
        loader.load_config_from_string(self.TYPOD)
        joined = "\n".join(printed)
        assert "quantizaton" in joined, "the API/UI call site never reported the typo"
        assert "unknown config key" in joined
        assert UNKNOWN_KEY_REJECTION_VERSION in joined, "the warning lost its deadline"

    def test_load_config_from_string_is_silent_on_a_clean_config(self, monkeypatch) -> None:
        """The control: wiring that warned on everything would pass the above."""
        from soup_cli.config.loader import load_config_from_string

        printed = self._recording_console(monkeypatch)
        load_config_from_string(self.CLEAN)
        assert printed == [], f"a clean config produced output: {printed}"

    def test_load_config_refuses_an_unknown_key(self, monkeypatch, tmp_path) -> None:
        """The file call site is a separate branch from the string one."""
        from soup_cli.config.loader import load_config

        path = tmp_path / "soup.yaml"
        path.write_text(self.TYPOD, encoding="utf-8")
        printed = self._recording_console(monkeypatch)
        with pytest.raises(SystemExit) as excinfo:
            load_config(path)
        assert excinfo.value.code == 1
        joined = "\n".join(printed)
        assert "quantizaton" in joined, "the CLI call site never reported the typo"
        assert "unknown config key" in joined
        assert "Refused." in joined and "Not applied." not in joined

    def test_load_config_warns_under_warn_severity(self, monkeypatch, tmp_path) -> None:
        from soup_cli.config import loader

        path = tmp_path / "soup.yaml"
        path.write_text(self.TYPOD, encoding="utf-8")
        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", "warn")
        printed = self._recording_console(monkeypatch)
        loader.load_config(path)
        joined = "\n".join(printed)
        assert "quantizaton" in joined, "the CLI call site never reported the typo"
        assert "unknown config key" in joined
        assert "Not applied." in joined, "a warning must say the key was ignored"
        assert UNKNOWN_KEY_REJECTION_VERSION in joined, "the warning lost its deadline"

    def test_load_config_is_silent_on_a_clean_config(self, monkeypatch, tmp_path) -> None:
        from soup_cli.config.loader import load_config

        path = tmp_path / "soup.yaml"
        path.write_text(self.CLEAN, encoding="utf-8")
        printed = self._recording_console(monkeypatch)
        load_config(path)
        assert printed == [], f"a clean config produced output: {printed}"

    def test_the_string_call_site_refuses_under_error_severity(self, monkeypatch) -> None:
        """Nothing else proves the v0.75 flip works, and a test will force it."""
        from soup_cli.config import loader

        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", "error")
        with pytest.raises(ValueError, match="quantizaton"):
            loader.load_config_from_string(self.TYPOD)

    def test_the_file_call_site_refuses_under_error_severity(
        self, monkeypatch, tmp_path
    ) -> None:
        from soup_cli.config import loader

        path = tmp_path / "soup.yaml"
        path.write_text(self.TYPOD, encoding="utf-8")
        self._recording_console(monkeypatch)
        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", "error")
        with pytest.raises(SystemExit):
            loader.load_config(path)

    def test_a_clean_config_still_loads_under_error_severity(self, monkeypatch) -> None:
        """The control for the flip: refusing everything would pass the two above."""
        from soup_cli.config import loader

        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", "error")
        assert loader.load_config_from_string(self.CLEAN).base == "test-model"


class TestTheDetectorSeesWhatTheValidatorSees:
    """A spelling ``SoupConfig`` accepts must never be one the detector refuses.

    ``SoupConfig`` has moved a root-level ``lora:`` block under ``training``
    since v0.40.1 (LlamaFactory / Axolotl convention). The v0.74.0 detector
    walked the RAW dict, so it flagged that spelling as unknown -- a warning
    then, and under v0.75.0's refusal a config the schema was built to accept
    would have failed to load (found in the release review of #879). Both
    sides now call the same ``remap_root_level_misplaced_keys``.
    """

    TOPLEVEL_LORA = (
        "base: test-model\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data.jsonl\n"
        "training:\n"
        "  epochs: 1\n"
        "lora:\n"
        "  r: 16\n"
        "  alpha: 32\n"
    )

    def test_a_root_level_lora_block_is_not_an_unknown_key(self) -> None:
        import yaml

        assert find_unknown_config_keys(yaml.safe_load(self.TOPLEVEL_LORA)) == []

    def test_a_root_level_lora_block_loads_under_the_shipped_refusal(self) -> None:
        from soup_cli.config import loader

        assert loader.UNKNOWN_KEY_SEVERITY == "error"
        assert loader.load_config_from_string(self.TOPLEVEL_LORA).training.lora.r == 16

    def test_a_typo_inside_a_root_level_lora_block_is_still_reported(self) -> None:
        """Remapping must not hide what is under the remapped key."""
        import yaml

        unknown = find_unknown_config_keys(
            yaml.safe_load(self.TOPLEVEL_LORA.replace("  alpha: 32\n", "  alpah: 32\n"))
        )
        assert [u.path for u in unknown] == ["training.lora.alpah"]
        assert "alpha" in unknown[0].suggestions

    def test_the_detector_reads_the_validator_key_list_not_a_copy(self) -> None:
        """Derived, not duplicated: every remapped key is accepted at the root."""
        from soup_cli.config.schema import ROOT_LEVEL_MISPLACED_KEYS

        assert ROOT_LEVEL_MISPLACED_KEYS, "the shared tuple went empty"
        for key in ROOT_LEVEL_MISPLACED_KEYS:
            raw = {"base": "m", "data": {"train": "t.jsonl"}, "training": {}, key: {}}
            assert find_unknown_config_keys(raw) == [], key

    def test_a_key_at_both_levels_is_the_validator_error_not_a_detector_one(self) -> None:
        """The detector steps aside; ``SoupConfig`` names the conflict precisely."""
        from soup_cli.config import loader

        both = self.TOPLEVEL_LORA.replace("  epochs: 1\n", "  epochs: 1\n  lora:\n    r: 8\n")
        with pytest.raises(ValueError, match="both root and training level"):
            loader.load_config_from_string(both)

    def test_the_caller_dict_is_not_mutated_by_the_detector(self) -> None:
        import yaml

        raw = yaml.safe_load(self.TOPLEVEL_LORA)
        before = repr(raw)
        find_unknown_config_keys(raw)
        assert repr(raw) == before


class TestTheReportIsSafeForTheTerminal:
    """Key names come from the config file; the report must not restyle the terminal.

    ``rich.markup.escape`` neutralises ``[...]`` and nothing else, and Rich
    passes a raw ESC byte straight through -- the class of bug six command
    modules already guard against with a private ``_for_terminal``. The
    loader printed the warning without either guard in v0.74.0, and v0.75.0's
    refusal printed the same text through a second, newly-live call. Both now
    go through ``soup_cli.utils.terminal.for_terminal``.
    """

    HOSTILE = (
        "base: test-model\n"
        "data:\n"
        "  train: ./data.jsonl\n"
        "training:\n"
        "  epochs: 1\n"
        '  "[bold red on white]INJECTED[/]": 1\n'
        '  "\\x1b]0;owned\\x07quantizaton": 4bit\n'
    )

    @staticmethod
    def _real_console(monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.config import loader

        buffer = StringIO()
        monkeypatch.setattr(loader, "console", Console(file=buffer, width=200, markup=True))
        return buffer

    @pytest.mark.parametrize("severity", ["error", "warn"])
    def test_markup_and_control_bytes_in_a_key_are_neutralised(
        self, monkeypatch, tmp_path, severity
    ) -> None:
        from soup_cli.config import loader

        monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", severity)
        buffer = self._real_console(monkeypatch)
        path = tmp_path / "soup.yaml"
        path.write_text(self.HOSTILE, encoding="utf-8")
        if severity == "error":
            with pytest.raises(SystemExit):
                loader.load_config(path)
        else:
            loader.load_config(path)
        out = buffer.getvalue()
        assert "INJECTED" in out and "quantizaton" in out, "the keys must still be named"
        assert "\x1b" not in out, "a raw ESC byte reached the terminal"
        assert "[bold red on white]" in out, "the markup was interpreted instead of shown"

    def test_the_shared_helper_does_both_halves(self) -> None:
        from soup_cli.utils.terminal import for_terminal

        assert for_terminal("\x1b[31m[bold]x[/]\x7f") == "[31m\\[bold]x\\[/]"
        assert for_terminal("tab\tnew\nline") == "tab\tnew\nline"

    @pytest.mark.parametrize("byte", [*range(0x00, 0x20), 0x7F])
    def test_every_control_byte_is_stripped_except_the_three_whitespace_ones(
        self, byte: int
    ) -> None:
        """ESC and DEL were pinned; BEL, CR and the other 28 were not (TDD review)."""
        from soup_cli.utils.terminal import for_terminal

        out = for_terminal(f"a{chr(byte)}b")
        if byte in (0x09, 0x0A, 0x0D):
            assert out == f"a{chr(byte)}b", f"whitespace byte {byte:#04x} must survive"
        else:
            assert out == "ab", f"control byte {byte:#04x} reached the terminal"


class TestTheWalkIsBounded:
    """A config string reaches the detector from the Web UI and MCP too.

    Every unknown key costs a difflib pass over ~240 declared names; 50,000
    bogus keys under one section measured ~31 s of CPU with no cap (release
    review of #879). The walk stops at ``_MAX_REPORTED_UNKNOWN_KEYS`` findings
    and the report says so; an absurdly long key gets no suggestion and is
    shown truncated.
    """

    def test_fifty_thousand_bogus_keys_are_capped_and_fast(self) -> None:
        import time

        from soup_cli.config.unknown_keys import _MAX_REPORTED_UNKNOWN_KEYS

        raw = {
            "base": "m",
            "data": {"train": "t.jsonl"},
            "training": {f"bogus_key_{i}": i for i in range(50_000)},
        }
        started = time.perf_counter()
        unknown = find_unknown_config_keys(raw)
        elapsed = time.perf_counter() - started
        # cap + 1: the extra finding is the overflow marker the report reads.
        assert len(unknown) == _MAX_REPORTED_UNKNOWN_KEYS + 1
        assert elapsed < 5.0, f"the capped walk took {elapsed:.1f}s"
        message = format_unknown_keys(unknown, include_deadline=False)
        assert f"capped at {_MAX_REPORTED_UNKNOWN_KEYS}" in message
        assert message.count("unknown config key") == _MAX_REPORTED_UNKNOWN_KEYS

    def test_exactly_the_cap_is_reported_in_full_without_the_cap_sentence(self) -> None:
        """A boundary the first version got wrong: 100 findings printed 'capped'."""
        from soup_cli.config.unknown_keys import _MAX_REPORTED_UNKNOWN_KEYS

        def raw(count: int) -> dict:
            return {
                "base": "m",
                "data": {"train": "t.jsonl"},
                "training": {f"bogus_key_{i}": i for i in range(count)},
            }

        at_cap = find_unknown_config_keys(raw(_MAX_REPORTED_UNKNOWN_KEYS))
        assert len(at_cap) == _MAX_REPORTED_UNKNOWN_KEYS
        at_cap_msg = format_unknown_keys(at_cap, include_deadline=False)
        assert "capped" not in at_cap_msg
        assert at_cap_msg.count("unknown config key") == _MAX_REPORTED_UNKNOWN_KEYS

        one_over = find_unknown_config_keys(raw(_MAX_REPORTED_UNKNOWN_KEYS + 1))
        assert len(one_over) == _MAX_REPORTED_UNKNOWN_KEYS + 1
        assert "capped" in format_unknown_keys(one_over, include_deadline=False)

    def test_a_report_below_the_cap_does_not_claim_to_be_capped(self) -> None:
        unknown = find_unknown_config_keys(
            {"base": "m", "data": {"train": "t.jsonl"}, "training": {"quantizaton": 1}}
        )
        assert "capped" not in format_unknown_keys(unknown)

    def test_a_megabyte_key_name_is_not_echoed(self) -> None:
        """The report names the key; it must not carry a megabyte of it."""
        from soup_cli.config.unknown_keys import _MAX_KEY_LEN

        huge = "q" * (1024 * 1024)
        unknown = find_unknown_config_keys(
            {"base": "m", "data": {"train": "t.jsonl"}, "training": {huge: 1}}
        )
        assert len(unknown) == 1
        assert len(unknown[0].path) < _MAX_KEY_LEN + 32
        assert unknown[0].path.endswith("...")
        assert len(format_unknown_keys(unknown)) < _MAX_KEY_LEN + 256


class TestTheWebUiKeepsTheHint:
    """``/api/train/start`` is where a Web UI user meets the refusal.

    Under v0.74.0's warning the loader never raised there, so the handler's
    generic ``"Invalid training configuration"`` was unreachable for this
    cause. Under the refusal it would have swallowed the one thing the
    feature exists to say -- which key, and what was probably meant. The
    SPA renders the detail through ``escapeHtml()`` (a code-review finding on
    #879).
    """

    def test_the_400_names_the_key_and_the_suggestion(self, monkeypatch) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.ui.app import create_app, get_auth_token

        client = TestClient(create_app())
        response = client.post(
            "/api/train/start",
            json={"config_yaml": TestTheLoaderIsActuallyWiredUp.TYPOD},
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail.startswith("Invalid training configuration")
        assert "quantizaton" in detail and "quantization" in detail
        assert "Refused." in detail


class TestTheLoaderRefusesANonMappingFile:
    """``load_config`` on a list-shaped YAML died with a TypeError traceback.

    ``load_config_from_string`` has guarded this shape since v0.40.x (its
    contract is ValueError-only); the file call site, touched in v0.75.0,
    never did — found by the post-release TDD review.
    """

    @pytest.mark.parametrize("text, shape", [("- a\n- b\n", "list"), ("just a string\n", "str")])
    def test_a_non_mapping_document_is_a_clean_exit_not_a_traceback(
        self, monkeypatch, tmp_path, text, shape
    ) -> None:
        from soup_cli.config import loader

        printed: list[str] = []
        monkeypatch.setattr(loader.console, "print", lambda *a, **k: printed.append(str(a[0])))
        path = tmp_path / "soup.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        assert any("YAML mapping" in line and shape in line for line in printed), printed


class TestTheWebUiErrorBranchesAreDistinct:
    """Both Web UI handlers split ValueError from everything else (TDD review).

    The split is the point of the v0.75.0 change: the loader's ValueError
    carries the key and the suggestion and is safe to show; anything else
    (a YAML parser error, an unexpected TypeError) must stay generic.
    """

    @staticmethod
    def _client():
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.ui.app import create_app, get_auth_token

        return TestClient(create_app()), {"Authorization": f"Bearer {get_auth_token()}"}

    def test_from_form_names_the_field_on_a_validation_error(self) -> None:
        client, headers = self._client()
        response = client.post(
            "/api/config/from-form",
            json={"base": "m", "data": {"train": "./t.jsonl"}, "training": {"epochs": 0}},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        error = response.json()["error"]
        assert error.startswith("Invalid configuration:")
        assert "epochs" in error, error

    def test_train_start_keeps_a_parser_error_generic(self) -> None:
        client, headers = self._client()
        response = client.post(
            "/api/train/start",
            json={"config_yaml": "base: [unclosed\n  data: {"},
            headers=headers,
        )
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail == "Invalid training configuration", detail


class TestPlanAndApplyRefuseUnknownKeys:
    """`soup plan` / `soup apply` must refuse what `soup train` refuses (#894).

    `_load_yaml_config` handed the raw mapping to `build_plan` without ever
    checking for unknown keys, so a typo'd key planned cleanly and was only
    refused one command later by `soup train`. Both commands now run the same
    unknown-key check the loader runs and exit 1 with the same message.
    """

    TYPOD = (
        "base: hf/model\n"
        "task: sft\n"
        "data:\n"
        "  train: ./t.jsonl\n"
        "  format: auto\n"
        "output: ./o\n"
        "training:\n"
        "  epochs: 1\n"
        "  quantizaton: none\n"
    )
    CLEAN = (
        "base: hf/model\n"
        "task: sft\n"
        "data:\n"
        "  train: ./t.jsonl\n"
        "  format: auto\n"
        "output: ./o\n"
        "training:\n"
        "  epochs: 1\n"
    )

    def test_plan_refuses_a_typod_key(self, tmp_path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(self.TYPOD, encoding="utf-8")
        result = CliRunner().invoke(app, ["plan", "--config", str(cfg)])
        assert result.exit_code == 1, result.output
        assert "unknown config key" in result.output
        assert "training.quantizaton" in result.output
        assert "quantization" in result.output

    def test_apply_dry_run_refuses_a_typod_key(self, tmp_path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(self.TYPOD, encoding="utf-8")
        result = CliRunner().invoke(
            app, ["apply", "--config", str(cfg), "--dry-run"]
        )
        assert result.exit_code == 1, result.output
        assert "unknown config key" in result.output
        assert "training.quantizaton" in result.output
        assert "quantization" in result.output

    def test_clean_config_still_plans_and_applies(self, tmp_path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(self.CLEAN, encoding="utf-8")
        runner = CliRunner()
        planned = runner.invoke(app, ["plan", "--config", str(cfg)])
        assert planned.exit_code == 0, planned.output
        applied = runner.invoke(app, ["apply", "--config", str(cfg), "--dry-run"])
        assert applied.exit_code == 0, applied.output
