"""``soup eval retention`` and ``utils/retention_ledger``: forgetting across a series.

``soup ship`` judges a candidate against one base on a fixed general suite, so an update
that quietly erodes a task an EARLIER update taught looks fine every time. The retention
ledger remembers every control task and every recorded version's score on it, and judges
a candidate against the BEST score each task ever had.

No model is loaded anywhere here: ``live_eval.make_generator`` is replaced by a fake whose
accuracy on each task is set by the test (``fake_model["profile"]``), so the numbers the
command prints are exactly the arithmetic under test. Each control task has 20 rows, so
scores move in steps of 0.05; 18/20 = 0.9 and 17/20 = 0.85 sit exactly one default
threshold apart, where a float comparison without an epsilon would misfire.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_cmd
from soup_cli.utils import retention_ledger as rl
from tests.conftest import strip_ansi

runner = CliRunner()
ROWS = 20
STAMP = "2026-11-05T00:00:00Z"
#: What a CPU run without --config stamps on every version it records.
CPU_NUMERICS = rl.Numerics(dtype="float32", device="cpu", quantization=None,
                           chat_template_kwargs=None)
CONFIG_4BIT = (
    "base: sshleifer/tiny-gpt2\ndata:\n  train: train.jsonl\n"
    "training:\n  quantization: 4bit\n"
)
HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (Rich colours and wraps)."""
    return " ".join(strip_ansi(text).split())


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    """Every test runs in its own directory: the ledger and control files live under cwd."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Pin the shared eval console wide so no message or table wraps mid-token."""
    monkeypatch.setattr(eval_cmd.console, "_width", 200)


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the generator factory; ``state['profile']`` is correct-answers per task."""
    from soup_cli.utils import live_eval

    state = {
        "profile": {}, "loads": [], "on_load": None,
        # tasks (or single prompts) that raise / return None: a failed generation, not a wrong
        # answer; ``empty_for`` tasks answer with an empty string, which IS a wrong answer
        "raise_for": set(), "none_for": set(), "empty_for": set(),
        # optional ``batch(prompts, generate)`` the generator exposes as ``.batch``
        "batch": None,
    }

    def factory(model_id, **kwargs):
        state["loads"].append((model_id, kwargs))
        if state["on_load"] is not None:
            state["on_load"]()
        profile = dict(state["profile"])

        def generate(prompt: str) -> str:
            task, _, index = prompt.rpartition("-")
            if task in state["raise_for"] or prompt in state["raise_for"]:
                raise RuntimeError("CUDA out of memory")
            if task in state["none_for"]:
                return None
            if task in state["empty_for"]:
                return ""
            return "ok" if int(index) < profile.get(task, 0) else "bad"

        if state["batch"] is not None:
            generate.batch = lambda prompts: state["batch"](prompts, generate)
            generate.batch_size = 8
        return generate

    monkeypatch.setattr(live_eval, "make_generator", factory)
    monkeypatch.setattr(live_eval, "resolve_device", lambda device=None: device or "cpu")
    return state


def _write_task_file(name: str, path: str | None = None, rows: int = ROWS) -> str:
    target = path or f"{name}.jsonl"
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"prompt": f"{name}-{index}", "expected": "ok"}) for index in range(rows)]
    Path(target).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _invoke(*args: str):
    return runner.invoke(app, ["eval", "retention", *args])


def _run(fake, profile: dict, label: str, *args: str, device: str = "cpu"):
    """One retention run of ``base-model`` with ``profile`` correct answers per task."""
    fake["profile"] = dict(profile)
    return _invoke("--model", "base-model", "--device", device, "--label", label, *args)


def _read_ledger() -> dict:
    return json.loads(Path("retention.json").read_text(encoding="utf-8"))


def _check(result, code: int) -> str:
    assert result.exit_code == code, (result.output, repr(result.exception))
    return _plain(result.output)


def _seed_two_tasks(fake) -> None:
    """v1 records alpha=0.9 and beta=0.8 with both control files registered."""
    _write_task_file("alpha")
    _write_task_file("beta")
    result = _run(
        fake, {"alpha": 18, "beta": 16}, "v1",
        "--add-task", "alpha=alpha.jsonl", "--add-task", "beta=beta.jsonl", "--record",
    )
    _check(result, 0)


def _task(name: str = "alpha") -> rl.RetentionTask:
    return rl.RetentionTask(name, f"{name}.jsonl", "a" * 64, STAMP, None)


def _version(label: str, scores: dict, numerics: rl.Numerics = CPU_NUMERICS):
    return rl.RetentionVersion(label, "base-model", None, STAMP, dict(scores), numerics)


def _valid_payload() -> dict:
    return {
        "schema": 1,
        "tasks": [
            {
                "name": "alpha", "path": "alpha.jsonl", "sha256": "a" * 64,
                "added_at": STAMP, "added_with": "v1",
            }
        ],
        "versions": [
            {
                "label": "v1", "model": "base-model", "adapter": None,
                "evaluated_at": STAMP, "scores": {"alpha": 0.9},
                "numerics": {
                    "dtype": "float32", "device": "cpu", "quantization": None,
                    "chat_template_kwargs": None,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Ledger format: validation
# ---------------------------------------------------------------------------

def _edit_task(**changes):
    def mutate(payload):
        payload["tasks"][0].update(changes)
        return payload

    return mutate


def _edit_version(**changes):
    def mutate(payload):
        payload["versions"][0].update(changes)
        return payload

    return mutate


def _drop_task_key(key: str):
    def mutate(payload):
        del payload["tasks"][0][key]
        return payload

    return mutate


def _edit_numerics(**changes):
    def mutate(payload):
        payload["versions"][0]["numerics"].update(changes)
        return payload

    return mutate


def _drop_numerics_key(key: str):
    def mutate(payload):
        del payload["versions"][0]["numerics"][key]
        return payload

    return mutate


def _drop_version_key(key: str):
    def mutate(payload):
        del payload["versions"][0][key]
        return payload

    return mutate


_BAD_LEDGERS = {
    "root-is-a-list": (lambda p: [], "ledger must be a JSON object, got list"),
    "schema-2": (lambda p: {**p, "schema": 2}, "unsupported ledger schema 2"),
    "schema-bool": (lambda p: {**p, "schema": True}, "'schema' must be an integer, got bool"),
    "schema-str": (lambda p: {**p, "schema": "1"}, "'schema' must be an integer, got str"),
    "missing-versions": (
        lambda p: {k: v for k, v in p.items() if k != "versions"},
        "ledger is missing key(s): 'versions'",
    ),
    "unknown-root-key": (lambda p: {**p, "extra": 1}, "ledger has unsupported key(s): 'extra'"),
    "tasks-is-a-dict": (lambda p: {**p, "tasks": {}}, "'tasks' must be a list, got dict"),
    "task-is-a-string": (
        lambda p: {**p, "tasks": ["alpha"]}, "ledger tasks[0] must be an object, got str"
    ),
    "task-missing-sha": (
        _drop_task_key("sha256"), "ledger tasks[0] is missing key(s): 'sha256'"
    ),
    "task-name-with-slash": (_edit_task(name="../x"), "task name '../x' is invalid"),
    "task-name-too-long": (_edit_task(name="a" * 65), "is invalid"),
    "task-bad-sha": (
        _edit_task(sha256="XYZ"), "ledger tasks[0].sha256 must be 64 lowercase hex characters"
    ),
    "task-upper-case-sha": (
        _edit_task(sha256="A" * 64), "must be 64 lowercase hex characters"
    ),
    "task-added-with-int": (
        _edit_task(added_with=5), "ledger tasks[0].added_with must be a string, got int"
    ),
    "task-unknown-key": (_edit_task(color="red"), "ledger tasks[0] has unsupported key(s)"),
    "duplicate-task": (
        lambda p: {**p, "tasks": p["tasks"] * 2}, "ledger task 'alpha' is listed twice"
    ),
    "version-scores-unknown-task": (
        _edit_version(scores={"ghost": 0.5}), "scores unknown task 'ghost'"
    ),
    "score-above-one": (
        _edit_version(scores={"alpha": 1.5}),
        "ledger versions[0].scores['alpha'] must be between 0.0 and 1.0, got 1.5",
    ),
    "score-negative": (_edit_version(scores={"alpha": -0.1}), "must be between 0.0 and 1.0"),
    "score-bool": (_edit_version(scores={"alpha": True}), "must be a number, got bool"),
    "score-string": (_edit_version(scores={"alpha": "0.9"}), "must be a number, got str"),
    "score-huge-int": (
        _edit_version(scores={"alpha": 10**400}), "must be between 0.0 and 1.0"
    ),
    "score-huge-negative-int": (
        _edit_version(scores={"alpha": -(10**400)}), "must be between 0.0 and 1.0"
    ),
    "version-missing-numerics": (
        _drop_version_key("numerics"), "ledger versions[0] is missing key(s): 'numerics'"
    ),
    "numerics-is-a-string": (
        _edit_version(numerics="cpu"), "ledger versions[0].numerics must be an object, got str"
    ),
    "numerics-unknown-key": (
        _edit_numerics(extra=1), "ledger versions[0].numerics has unsupported key(s): 'extra'"
    ),
    "numerics-missing-device": (
        _drop_numerics_key("device"), "ledger versions[0].numerics is missing key(s): 'device'"
    ),
    "numerics-device-int": (
        _edit_numerics(device=3), "ledger versions[0].numerics.device must be a string, got int"
    ),
    "numerics-device-empty": (_edit_numerics(device=""), "numerics.device"),
    "numerics-dtype-int": (
        _edit_numerics(dtype=3), "ledger versions[0].numerics.dtype must be a string, got int"
    ),
    "numerics-quantization-list": (
        _edit_numerics(quantization=["4bit"]),
        "ledger versions[0].numerics.quantization must be a string, got list",
    ),
    "numerics-chat-kwargs-too-long": (
        _edit_numerics(chat_template_kwargs="x" * 513), "numerics.chat_template_kwargs is too long"
    ),
    "numerics-chat-kwargs-control-char": (
        _edit_numerics(chat_template_kwargs='{"a":"\x07"}'),
        "must not contain control characters",
    ),
    "scores-is-a-list": (
        _edit_version(scores=[]), "ledger versions[0].scores must be an object, got list"
    ),
    "version-label-control-char": (
        _edit_version(label="v\x071"), "must not contain control characters"
    ),
    "version-label-too-long": (
        _edit_version(label="v" * 129), "version label is too long (129 characters, max 128)"
    ),
    "version-label-empty": (_edit_version(label=""), "must be non-empty"),
    "version-model-int": (
        _edit_version(model=3), "ledger versions[0].model must be a string, got int"
    ),
    "version-adapter-int": (_edit_version(adapter=3), "adapter must be a string, got int"),
    "duplicate-label": (
        lambda p: {**p, "versions": p["versions"] * 2},
        "ledger version label 'v1' is listed twice",
    ),
    "too-many-tasks": (
        lambda p: {**p, "tasks": [{}] * (rl.MAX_TASKS + 1)}, "ledger has too many tasks"
    ),
    "too-many-versions": (
        lambda p: {**p, "versions": [{}] * (rl.MAX_VERSIONS + 1)},
        "ledger has too many versions",
    ),
}


class TestLedgerValidation:
    def test_a_valid_payload_parses(self):
        ledger = rl.parse_ledger(_valid_payload())
        assert ledger.task_names() == ("alpha",)
        assert ledger.versions[0].scores == {"alpha": 0.9}
        assert ledger.versions[0].adapter is None
        assert ledger.tasks[0].added_with == "v1"

    @pytest.mark.parametrize("case", sorted(_BAD_LEDGERS))
    def test_a_bad_ledger_is_refused_with_a_message_naming_the_problem(self, case):
        build, expected = _BAD_LEDGERS[case]
        with pytest.raises(rl.LedgerError) as info:
            rl.parse_ledger(build(_valid_payload()))
        assert expected in str(info.value), str(info.value)

    def test_a_null_added_with_and_a_real_adapter_are_accepted(self):
        payload = _valid_payload()
        payload["tasks"][0]["added_with"] = None
        payload["versions"][0]["adapter"] = "out/adapter"
        ledger = rl.parse_ledger(payload)
        assert ledger.tasks[0].added_with is None
        assert ledger.versions[0].adapter == "out/adapter"

    def test_the_numerics_stamp_is_parsed_into_its_four_keys(self):
        payload = _valid_payload()
        payload["versions"][0]["numerics"] = {
            "dtype": None, "device": "cuda", "quantization": "4bit",
            "chat_template_kwargs": '{"enable_thinking":false}',
        }
        stamp = rl.parse_ledger(payload).versions[0].numerics
        assert stamp == rl.Numerics(
            dtype=None, device="cuda", quantization="4bit",
            chat_template_kwargs='{"enable_thinking":false}',
        )

    def test_the_round_trip_through_a_file_is_lossless(self):
        original = rl.parse_ledger(_valid_payload())
        rl.save_ledger(original, "retention.json")
        assert rl.load_ledger("retention.json") == original
        assert _read_ledger()["versions"][0]["numerics"] == {
            "dtype": "float32", "device": "cpu", "quantization": None,
            "chat_template_kwargs": None,
        }

    def test_a_400_digit_integer_score_is_a_ledger_error_not_an_overflow(self):
        """``float(10**400)`` raises OverflowError; the file must get a named refusal."""
        text = json.dumps(_valid_payload()).replace("0.9", "9" * 400)
        Path("retention.json").write_text(text, encoding="utf-8")
        with pytest.raises(rl.LedgerError) as info:
            rl.load_ledger("retention.json")
        message = str(info.value)
        assert "ledger versions[0].scores['alpha'] must be between 0.0 and 1.0" in message
        assert "9" * 100 not in message, "the message must not echo 400 digits"

    @pytest.mark.parametrize(
        "value", [10**400, -(10**400)], ids=["huge-positive", "huge-negative"]
    )
    def test_a_huge_integer_score_or_threshold_is_refused(self, value):
        with pytest.raises(rl.LedgerError, match="must be between 0.0 and 1.0"):
            rl.validate_score(value, "score")
        with pytest.raises(rl.LedgerError, match="threshold must be between 0.0 and 1.0"):
            rl.validate_threshold(value)

    def test_the_integers_zero_and_one_are_valid_scores(self):
        assert rl.validate_score(0, "score") == 0.0
        assert rl.validate_score(1, "score") == 1.0
        assert isinstance(rl.validate_score(1, "score"), float)

    def test_invalid_json_is_named_not_a_traceback(self):
        Path("retention.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(rl.LedgerError, match="is not valid JSON"):
            rl.load_ledger("retention.json")

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_number_is_refused_at_decode_time(self, literal):
        text = json.dumps(_valid_payload()).replace("0.9", literal)
        Path("retention.json").write_text(text, encoding="utf-8")
        with pytest.raises(rl.LedgerError, match="non-finite number"):
            rl.load_ledger("retention.json")

    def test_non_utf8_bytes_are_refused(self):
        Path("retention.json").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(rl.LedgerError, match="is not valid UTF-8"):
            rl.load_ledger("retention.json")

    def test_a_ledger_over_the_size_cap_is_refused_without_reading_it_whole(self):
        with open("retention.json", "wb") as handle:
            handle.write(b" " * (rl.MAX_LEDGER_BYTES + 1))
        with pytest.raises(rl.LedgerError, match=f"exceeds {rl.MAX_LEDGER_BYTES} bytes"):
            rl.load_ledger("retention.json")

    def test_a_missing_ledger_is_a_distinct_error(self):
        with pytest.raises(rl.LedgerNotFoundError, match="does not exist"):
            rl.load_ledger("retention.json")

    def test_a_ledger_outside_the_working_directory_is_refused(self):
        with pytest.raises(rl.LedgerError, match="must stay under cwd"):
            rl.load_ledger("../retention.json")

    def test_a_directory_is_not_a_ledger(self):
        os.mkdir("retention.json")
        with pytest.raises(rl.LedgerError):
            rl.load_ledger("retention.json")

    def test_a_file_that_is_not_a_regular_file_is_refused_before_it_is_read(self, monkeypatch):
        """A FIFO or device would block the read; the open itself must refuse it."""
        Path("retention.json").write_text(json.dumps(_valid_payload()), encoding="utf-8")
        monkeypatch.setattr(rl.stat, "S_ISREG", lambda mode: False)
        with pytest.raises(rl.LedgerError, match="must be a regular file"):
            rl.load_ledger("retention.json")

    def test_a_non_regular_file_is_refused_before_a_file_object_is_made(self, monkeypatch):
        """On Linux a directory opens as a descriptor and only ``os.fdopen`` raises.

        That ``IsADirectoryError`` would escape as a traceback, so the type check runs on the
        bare descriptor; Windows refuses the open itself, so the Linux order is simulated.
        """
        Path("retention.json").write_text(json.dumps(_valid_payload()), encoding="utf-8")
        monkeypatch.setattr(rl.stat, "S_ISREG", lambda mode: False)

        def _fdopen_of_a_directory(*_args, **_kwargs):
            raise IsADirectoryError(21, "Is a directory")

        monkeypatch.setattr(rl.os, "fdopen", _fdopen_of_a_directory)
        with pytest.raises(rl.LedgerError, match="must be a regular file"):
            rl.load_ledger("retention.json")

    @pytest.mark.requires_symlink
    def test_a_symlinked_ledger_is_refused_for_reading_and_writing(self):
        Path("real.json").write_text(json.dumps(_valid_payload()), encoding="utf-8")
        os.symlink("real.json", "retention.json")
        with pytest.raises(rl.LedgerError, match="must not be a symlink"):
            rl.load_ledger("retention.json")
        with pytest.raises(rl.LedgerError, match="must not be a symlink"):
            rl.save_ledger(rl.Ledger(), "retention.json")
        assert json.loads(Path("real.json").read_text(encoding="utf-8")) == _valid_payload()

    def test_a_hostile_name_is_quoted_without_its_control_bytes(self):
        payload = _valid_payload()
        payload["tasks"][0]["name"] = "bad\x1b]0;owned\x07name"
        with pytest.raises(rl.LedgerError) as info:
            rl.parse_ledger(payload)
        assert "\x1b" not in str(info.value)
        assert "\x07" not in str(info.value)
        assert "is invalid" in str(info.value)


# ---------------------------------------------------------------------------
# Ledger format: writing
# ---------------------------------------------------------------------------

class TestLedgerWriting:
    def test_save_replaces_the_file_and_leaves_no_temp_file(self):
        rl.save_ledger(rl.parse_ledger(_valid_payload()), "retention.json")
        assert sorted(os.listdir(".")) == ["retention.json"]

    def test_a_failed_write_leaves_the_old_ledger_intact(self, monkeypatch):
        """Atomic means the old bytes survive a failure in the middle of the swap."""
        original = rl.parse_ledger(_valid_payload())
        rl.save_ledger(original, "retention.json")
        before = Path("retention.json").read_bytes()

        def refuse(*_args, **_kwargs):
            raise OSError("disk went away")

        monkeypatch.setattr(os, "replace", refuse)
        bigger = rl.append_version(original, _version("v2", {"alpha": 0.5}))
        with pytest.raises(OSError, match="disk went away"):
            rl.save_ledger(bigger, "retention.json")
        assert Path("retention.json").read_bytes() == before
        assert sorted(os.listdir(".")) == ["retention.json"]

    def test_a_ledger_that_would_exceed_the_cap_is_not_written(self):
        names = [f"task{index:04d}" for index in range(rl.MAX_TASKS)]
        scores = {name: 0.123456 for name in names}
        ledger = rl.Ledger(
            tasks=tuple(_task(name) for name in names),
            versions=tuple(
                rl.RetentionVersion(
                    f"v{index}", "base-model", None, STAMP, scores, CPU_NUMERICS
                )
                for index in range(200)
            ),
        )
        with pytest.raises(rl.LedgerError, match="would exceed"):
            rl.save_ledger(ledger, "retention.json")
        assert not Path("retention.json").exists()

    def test_a_duplicate_label_is_refused_on_append(self):
        ledger = rl.parse_ledger(_valid_payload())
        with pytest.raises(rl.LedgerError, match="already recorded"):
            rl.append_version(ledger, _version("v1", {"alpha": 0.5}))

    def test_the_version_count_is_capped_on_append(self):
        full = rl.Ledger(
            tasks=(_task(),),
            versions=tuple(
                _version(f"v{index}", {"alpha": 0.9}) for index in range(rl.MAX_VERSIONS)
            ),
        )
        with pytest.raises(rl.LedgerError, match=f"{rl.MAX_VERSIONS} versions .the maximum"):
            rl.append_version(full, _version("one-more", {"alpha": 0.9}))

    def test_save_if_unchanged_writes_when_the_file_is_what_the_run_read(self):
        loaded = rl.parse_ledger(_valid_payload())
        rl.save_ledger(loaded, "retention.json")
        bigger = rl.append_version(loaded, _version("v2", {"alpha": 0.8}))
        rl.save_ledger_if_unchanged(bigger, "retention.json", loaded, label="v2")
        assert rl.load_ledger("retention.json") == bigger

    def test_save_if_unchanged_writes_a_new_file_when_none_existed_at_load(self):
        ledger = rl.Ledger(tasks=(_task(),), versions=(_version("v1", {"alpha": 0.9}),))
        rl.save_ledger_if_unchanged(ledger, "retention.json", None, label="v1")
        assert rl.load_ledger("retention.json") == ledger

    def test_save_if_unchanged_refuses_a_ledger_another_run_changed(self):
        loaded = rl.parse_ledger(_valid_payload())
        rl.save_ledger(loaded, "retention.json")
        other = rl.append_version(loaded, _version("v2-other", {"alpha": 0.7}))
        rl.save_ledger(other, "retention.json")
        before = Path("retention.json").read_bytes()
        mine = rl.append_version(loaded, _version("v2", {"alpha": 0.8}))
        with pytest.raises(rl.LedgerChangedError, match="changed since this run read it"):
            rl.save_ledger_if_unchanged(mine, "retention.json", loaded, label="v2")
        assert Path("retention.json").read_bytes() == before

    def test_save_if_unchanged_names_a_label_another_run_recorded_first(self):
        loaded = rl.parse_ledger(_valid_payload())
        rl.save_ledger(loaded, "retention.json")
        rl.save_ledger(rl.append_version(loaded, _version("v2", {"alpha": 0.7})), "retention.json")
        mine = rl.append_version(loaded, _version("v2", {"alpha": 0.8}))
        with pytest.raises(rl.LedgerChangedError, match="'v2' was recorded by another run"):
            rl.save_ledger_if_unchanged(mine, "retention.json", loaded, label="v2")

    def test_save_if_unchanged_refuses_a_ledger_that_appeared_during_the_run(self):
        ledger = rl.Ledger(tasks=(_task(),), versions=(_version("v1", {"alpha": 0.9}),))
        rl.save_ledger(ledger, "retention.json")
        before = Path("retention.json").read_bytes()
        with pytest.raises(rl.LedgerChangedError, match="changed since this run read it"):
            rl.save_ledger_if_unchanged(ledger, "retention.json", None, label="v9")
        assert Path("retention.json").read_bytes() == before


# ---------------------------------------------------------------------------
# Control tasks: registration and the hash pin
# ---------------------------------------------------------------------------

class TestControlTasks:
    def test_register_pins_the_sha256_and_stores_a_relative_path(self):
        _write_task_file("alpha", "ctrl/alpha.jsonl")
        ledger, status = rl.register_task(
            rl.Ledger(), "alpha", "ctrl/alpha.jsonl", label="v1", now=STAMP
        )
        assert status == "added"
        task = ledger.get_task("alpha")
        assert task.sha256 == _sha("ctrl/alpha.jsonl")
        assert task.path == "ctrl/alpha.jsonl"
        assert task.added_with == "v1"
        assert task.added_at == STAMP

    def test_re_registering_the_same_bytes_changes_nothing(self):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        again, status = rl.register_task(ledger, "alpha", "alpha.jsonl", label="v2", now=STAMP)
        assert status == "unchanged"
        assert again == ledger

    def test_the_same_bytes_at_a_new_path_move_the_task(self):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        os.makedirs("moved")
        os.replace("alpha.jsonl", "moved/alpha.jsonl")
        moved, status = rl.register_task(
            ledger, "alpha", "moved/alpha.jsonl", label="v2", now=STAMP
        )
        assert status == "moved"
        assert moved.get_task("alpha").path == "moved/alpha.jsonl"
        assert moved.get_task("alpha").sha256 == ledger.get_task("alpha").sha256

    def test_a_registered_name_with_different_bytes_is_refused_naming_both_hashes(self):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        old = _sha("alpha.jsonl")
        _write_task_file("alpha", rows=ROWS + 1)
        new = _sha("alpha.jsonl")
        assert old != new
        with pytest.raises(rl.LedgerError) as info:
            rl.register_task(ledger, "alpha", "alpha.jsonl", label="v2", now=STAMP)
        message = str(info.value)
        assert "control task 'alpha' changed" in message
        assert old in message and new in message

    def test_verify_catches_an_edited_and_a_missing_control_file(self):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        rl.verify_ledger_files(ledger)
        _write_task_file("alpha", rows=3)
        with pytest.raises(rl.LedgerError, match="control task 'alpha' changed"):
            rl.verify_ledger_files(ledger)
        os.remove("alpha.jsonl")
        with pytest.raises(rl.LedgerError, match="control task 'alpha' file .* is missing"):
            rl.verify_ledger_files(ledger)

    def test_a_missing_control_file_message_says_how_to_register_the_new_path(self):
        ledger = rl.Ledger(tasks=(_task("alpha"),))
        with pytest.raises(rl.LedgerError) as info:
            rl.verify_ledger_files(ledger)
        assert "if it moved, pass --add-task alpha=NEWPATH" in str(info.value)

    def test_the_bytes_that_are_hashed_are_the_bytes_a_caller_can_parse(self):
        _write_task_file("alpha")
        data = rl.read_control_bytes("alpha.jsonl", "control task alpha path")
        assert data == Path("alpha.jsonl").read_bytes()
        assert hashlib.sha256(data).hexdigest() == rl.file_sha256(
            "alpha.jsonl", "control task alpha path"
        )

    @pytest.mark.requires_symlink
    def test_read_control_bytes_refuses_a_symlink(self):
        _write_task_file("real")
        os.symlink("real.jsonl", "alpha.jsonl")
        with pytest.raises(rl.LedgerError, match="must not be a symlink"):
            rl.read_control_bytes("alpha.jsonl", "control task alpha path")

    def test_verify_refuses_a_ledger_path_that_leaves_the_working_directory(self):
        escaping = rl.Ledger(
            tasks=(rl.RetentionTask("alpha", "../a.jsonl", "a" * 64, STAMP, None),)
        )
        with pytest.raises(rl.LedgerError, match="must stay under cwd"):
            rl.verify_ledger_files(escaping)

    @pytest.mark.requires_symlink
    def test_a_symlinked_control_file_is_refused(self):
        _write_task_file("real")
        os.symlink("real.jsonl", "alpha.jsonl")
        with pytest.raises(rl.LedgerError, match="must not be a symlink"):
            rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("alpha", "--add-task expects NAME=PATH"),
            ("alpha=", "--add-task expects NAME=PATH"),
            ("=alpha.jsonl", "task name '' is invalid"),
            ("bad/name=alpha.jsonl", "task name 'bad/name' is invalid"),
            (f"{'n' * 65}=alpha.jsonl", "is invalid"),
        ],
    )
    def test_bad_task_specs_are_refused(self, spec, expected):
        with pytest.raises(rl.LedgerError) as info:
            rl.parse_task_spec(spec)
        assert expected in str(info.value)

    def test_only_the_first_equals_sign_splits_a_spec(self):
        assert rl.parse_task_spec("alpha=dir/a=b.jsonl") == ("alpha", "dir/a=b.jsonl")

    def test_the_task_count_is_capped(self):
        _write_task_file("extra")
        full = rl.Ledger(tasks=tuple(_task(f"t{index}") for index in range(rl.MAX_TASKS)))
        with pytest.raises(rl.LedgerError, match="the maximum"):
            rl.register_task(full, "extra", "extra.jsonl", label="v1", now=STAMP)

    def test_a_control_file_over_the_hash_cap_is_refused(self, monkeypatch):
        _write_task_file("alpha")
        size = Path("alpha.jsonl").stat().st_size
        monkeypatch.setattr(rl, "MAX_CONTROL_FILE_BYTES", size - 1)
        with pytest.raises(rl.LedgerError, match=f"exceeds {size - 1} bytes"):
            rl.file_sha256("alpha.jsonl", "control task alpha path")
        monkeypatch.setattr(rl, "MAX_CONTROL_FILE_BYTES", size)
        assert rl.file_sha256("alpha.jsonl", "control task alpha path") == _sha("alpha.jsonl")


# ---------------------------------------------------------------------------
# The numerics stamp
# ---------------------------------------------------------------------------

class TestNumericsStamp:
    def test_the_chat_template_kwargs_are_stamped_as_canonical_json(self):
        one = rl.make_numerics(
            dtype="float32", device="cpu", quantization=None,
            chat_template_kwargs={"b": 1, "a": {"y": True, "x": None}},
        )
        two = rl.make_numerics(
            dtype="float32", device="cpu", quantization=None,
            chat_template_kwargs={"a": {"x": None, "y": True}, "b": 1},
        )
        assert one == two
        assert one.chat_template_kwargs == '{"a":{"x":null,"y":true},"b":1}'

    @pytest.mark.parametrize("empty", [None, {}])
    def test_no_template_kwargs_and_an_empty_dict_stamp_the_same(self, empty):
        stamp = rl.make_numerics(
            dtype="float32", device="cpu", quantization=None, chat_template_kwargs=empty
        )
        assert stamp.chat_template_kwargs is None

    def test_template_kwargs_too_long_to_record_are_refused(self):
        with pytest.raises(rl.LedgerError, match="too long to record"):
            rl.make_numerics(
                dtype="float32", device="cpu", quantization=None,
                chat_template_kwargs={"k": "x" * 600},
            )

    def test_the_differences_name_every_key_in_a_fixed_order(self):
        mine = rl.Numerics("bfloat16", "cuda", None, '{"a":1}')
        theirs = rl.Numerics("float32", "cpu", None, None)
        assert mine.differences(theirs) == (
            ("dtype", "bfloat16", "float32"),
            ("device", "cuda", "cpu"),
            ("chat_template_kwargs", '{"a":1}', None),
        )
        assert mine.differences(mine) == ()

    def test_check_passes_for_an_empty_ledger_and_for_the_same_numerics(self):
        rl.check_numerics(rl.Ledger(), CPU_NUMERICS)
        ledger = rl.Ledger(tasks=(_task(),), versions=(_version("v1", {"alpha": 0.9}),))
        rl.check_numerics(ledger, CPU_NUMERICS)

    def test_check_refuses_a_different_stamp_naming_the_keys_and_both_values(self):
        ledger = rl.Ledger(tasks=(_task(),), versions=(_version("v1", {"alpha": 0.9}),))
        with pytest.raises(rl.NumericsMismatchError) as info:
            rl.check_numerics(ledger, rl.Numerics(None, "cuda", "4bit", None))
        message = str(info.value)
        assert "dtype None (this run) vs 'float32' (version 'v1')" in message
        assert "device 'cuda' (this run) vs 'cpu' (version 'v1')" in message
        assert "quantization '4bit' (this run) vs None (version 'v1')" in message
        assert "chat_template_kwargs" not in message, "an equal key is not named"
        assert "start a new ledger" in message

    def test_check_compares_with_every_recorded_version_not_only_the_last(self):
        odd = rl.Numerics("bfloat16", "cuda", None, None)
        ledger = rl.Ledger(
            tasks=(_task(),),
            versions=(_version("v1", {"alpha": 0.9}, odd), _version("v2", {"alpha": 0.9})),
        )
        with pytest.raises(rl.NumericsMismatchError, match="version 'v1'"):
            rl.check_numerics(ledger, CPU_NUMERICS)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def _report(versions: list, current: dict, *, threshold: float = 0.05, tasks=("alpha",)):
    ledger = rl.Ledger(
        tasks=tuple(_task(name) for name in tasks),
        versions=tuple(_version(label, scores) for label, scores in versions),
    )
    return rl.compute_retention(ledger, current, label="now", threshold=threshold)


class TestRetentionArithmetic:
    def test_the_first_version_is_a_baseline_entry_and_cannot_regress(self):
        report = _report([], {"alpha": 0.3})
        row = report.tasks[0]
        assert row.status == rl.STATUS_NEW
        assert row.best_previous is None and row.drop is None and row.first is None
        assert report.regressions == ()
        assert report.mean_forgetting() is None
        assert report.backward_transfer() is None
        assert report.summary()["new_tasks"] == 1

    def test_a_drop_beyond_the_threshold_regresses(self):
        report = _report([("v1", {"alpha": 0.9})], {"alpha": 0.8})
        assert report.tasks[0].status == rl.STATUS_REGRESSED
        assert report.tasks[0].drop == pytest.approx(0.1)
        assert [row.name for row in report.regressions] == ["alpha"]

    def test_a_drop_of_exactly_the_threshold_is_not_a_regression(self):
        # 0.90 - 0.85 is 0.05000000000000004 in floats: only comparing the rounded drop
        # (the number the table prints) with the threshold keeps it quiet.
        assert 0.90 - 0.85 > 0.05
        report = _report([("v1", {"alpha": 0.90})], {"alpha": 0.85})
        assert report.tasks[0].status == rl.STATUS_OK
        assert report.regressions == ()

    def test_a_drop_just_past_the_threshold_is_a_regression(self):
        report = _report([("v1", {"alpha": 0.90})], {"alpha": 0.84})
        assert report.tasks[0].status == rl.STATUS_REGRESSED

    def test_the_best_ever_score_is_the_reference_not_the_previous_one(self):
        """v1 0.9, v2 0.7 shipped anyway, v3 0.75: still 0.15 below the best."""
        report = _report([("v1", {"alpha": 0.9}), ("v2", {"alpha": 0.7})], {"alpha": 0.75})
        row = report.tasks[0]
        assert row.best_previous == 0.9
        assert row.drop == pytest.approx(0.15)
        assert row.status == rl.STATUS_REGRESSED
        assert row.scores == {"v1": 0.9, "v2": 0.7, "now": 0.75}

    def test_beating_the_best_is_an_improvement_with_a_negative_drop(self):
        report = _report([("v1", {"alpha": 0.6})], {"alpha": 0.8})
        row = report.tasks[0]
        assert row.status == rl.STATUS_OK
        assert row.drop == pytest.approx(-0.2)
        assert report.mean_forgetting() == 0.0

    def test_a_larger_threshold_forgives_the_same_drop(self):
        versions = [("v1", {"alpha": 0.9})]
        assert _report(versions, {"alpha": 0.75}, threshold=0.05).regressions
        assert not _report(versions, {"alpha": 0.75}, threshold=0.2).regressions
        assert _report(versions, {"alpha": 0.9}, threshold=0.0).regressions == ()
        assert _report(versions, {"alpha": 0.899}, threshold=0.0).regressions

    def test_mean_forgetting_and_backward_transfer_numbers(self):
        # alpha: 0.8 -> 0.6 -> now 0.7  (best 0.8, forgot 0.1, moved -0.1 from first)
        # beta : 0.5 -> 0.9 -> now 0.85 (best 0.9, forgot 0.05, moved +0.35 from first)
        # gamma: first seen now (no "before": in neither mean)
        report = _report(
            [("v1", {"alpha": 0.8, "beta": 0.5}), ("v2", {"alpha": 0.6, "beta": 0.9})],
            {"alpha": 0.7, "beta": 0.85, "gamma": 0.4},
            threshold=0.15,
            tasks=("alpha", "beta", "gamma"),
        )
        assert report.mean_forgetting() == pytest.approx(0.075)
        assert report.backward_transfer() == pytest.approx(0.125)
        summary = report.summary()
        assert (summary["tasks"], summary["comparable_tasks"], summary["new_tasks"]) == (3, 2, 1)
        assert summary["regressions"] == 0

    def test_forgetting_floors_each_task_at_zero_before_averaging(self):
        # alpha improved by 0.3 and beta forgot 0.1: improvement must not cancel forgetting.
        report = _report(
            [("v1", {"alpha": 0.5, "beta": 0.9})],
            {"alpha": 0.8, "beta": 0.8},
            tasks=("alpha", "beta"),
        )
        assert report.mean_forgetting() == pytest.approx(0.05)

    def test_a_task_added_later_uses_its_own_first_score(self):
        report = _report(
            [("v1", {"alpha": 0.9}), ("v2", {"alpha": 0.9, "beta": 0.4})],
            {"alpha": 0.9, "beta": 0.6},
            tasks=("alpha", "beta"),
        )
        beta = report.tasks[1]
        assert beta.first == 0.4 and beta.best_previous == 0.4
        assert beta.scores == {"v2": 0.4, "now": 0.6}
        assert report.backward_transfer() == pytest.approx((0.0 + 0.2) / 2)

    def test_a_label_that_is_already_recorded_is_refused(self):
        ledger = rl.Ledger(tasks=(_task(),), versions=(_version("v1", {"alpha": 0.9}),))
        with pytest.raises(rl.LedgerError, match="already recorded"):
            rl.compute_retention(ledger, {"alpha": 0.9}, label="v1", threshold=0.05)

    def test_the_scores_must_cover_exactly_the_ledger_tasks(self):
        ledger = rl.Ledger(tasks=(_task("alpha"), _task("beta")))
        with pytest.raises(rl.LedgerError, match="exactly the ledger's tasks"):
            rl.compute_retention(ledger, {"alpha": 0.9}, label="v1", threshold=0.05)

    @pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf"), True, "0.1"])
    def test_a_bad_threshold_is_refused(self, threshold):
        with pytest.raises(rl.LedgerError, match="threshold"):
            rl.validate_threshold(threshold)

    def test_the_report_dict_is_the_full_matrix(self):
        report = _report([("v1", {"alpha": 0.9})], {"alpha": 0.8})
        data = report.to_dict()
        assert data["schema"] == 1
        assert data["label"] == "now"
        assert data["versions"] == ["v1", "now"]
        assert data["tasks"][0]["scores"] == {"v1": 0.9, "now": 0.8}
        assert data["summary"]["regressed_tasks"] == ["alpha"]
        json.dumps(data, allow_nan=False)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

class TestRetentionCommand:
    def test_the_command_is_registered_under_soup_eval(self):
        result = runner.invoke(app, ["eval", "retention", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        for flag in ("--ledger", "--model", "--adapter", "--label", "--add-task", "--record",
                     "--threshold", "--output", "--device", "--config",
                     "--chat-template-kwargs", "--require-history"):
            assert flag in output, flag

    def test_the_first_version_registers_pins_and_records(self, fake_model):
        _write_task_file("alpha")
        result = _run(
            fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl", "--record"
        )
        output = _check(result, 0)
        assert "No ledger at retention.json; starting a new one." in output
        assert "Control task alpha: added." in output
        assert "the ledger has no earlier version" in output
        assert "NO RETENTION REGRESSION" in output
        assert "Recorded version v1 in retention.json" in output
        assert not re.search(r"\bpoints?\b", output), "no 'point' wording in printed text"
        ledger = _read_ledger()
        assert ledger["schema"] == 1
        task = ledger["tasks"][0]
        assert task["name"] == "alpha"
        assert task["path"] == "alpha.jsonl"
        assert task["sha256"] == _sha("alpha.jsonl")
        assert task["added_with"] == "v1"
        version = ledger["versions"][0]
        assert version["label"] == "v1"
        assert version["model"] == "base-model"
        assert version["adapter"] is None
        assert version["scores"] == {"alpha": 0.9}
        assert version["evaluated_at"].endswith("Z")

    def test_without_record_the_ledger_is_not_written(self, fake_model):
        _write_task_file("alpha")
        result = _run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl")
        output = _check(result, 0)
        assert "Dry run: pass --record" in output
        assert not Path("retention.json").exists()

    def test_one_model_is_loaded_once_for_every_task_with_the_adapter(self, fake_model):
        _seed_two_tasks(fake_model)
        fake_model["loads"].clear()
        _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--adapter", "out/adapter"), 0)
        assert len(fake_model["loads"]) == 1
        model_id, kwargs = fake_model["loads"][0]
        assert model_id == "base-model"
        assert kwargs["adapter"] == "out/adapter"
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] == "float32"
        assert kwargs["max_new_tokens"] == 256

    def test_the_adapter_is_stored_in_the_recorded_version(self, fake_model):
        _seed_two_tasks(fake_model)
        _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--adapter", "a/b", "--record"), 0)
        assert _read_ledger()["versions"][1]["adapter"] == "a/b"

    def test_the_drop_is_printed_with_every_digit_the_verdict_compares(self, fake_model):
        """The verdict compares the drop rounded to 6 places; the table prints all of them."""
        _write_task_file("alpha", rows=3)
        _check(_run(fake_model, {"alpha": 2}, "v1", "--add-task", "alpha=alpha.jsonl",
                    "--record"), 0)
        output = _check(_run(fake_model, {"alpha": 1}, "v2"), 2)
        # best 0.666667 (as recorded) - 1/3 = 0.333334 after rounding to 6 places
        assert "alpha (+0.333334)" in output
        assert output.count("+0.333334") == 2, "once in the table, once in the verdict line"
        assert "(threshold 0.050)" in output, "a short threshold keeps three decimals"

    def test_a_version_none_of_whose_tasks_were_scored_before_is_told_so(self, fake_model):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        ledger = rl.append_version(ledger, _version("v1", {}))
        rl.save_ledger(ledger, "retention.json")
        output = _check(_run(fake_model, {"alpha": 18}, "v2"), 0)
        assert "No earlier version scored these tasks; this version becomes the baseline" in output
        assert "the ledger has no earlier version" not in output
        assert not re.search(r"\bpoints?\b", output)

    def test_a_drop_beyond_the_threshold_exits_2_and_names_the_task(self, fake_model):
        _seed_two_tasks(fake_model)
        result = _run(fake_model, {"alpha": 14, "beta": 16}, "v2")
        output = _check(result, 2)
        assert "RETENTION REGRESSION: alpha (+0.200)" in output
        assert "beta" in output and "OK" in output
        assert "Regressions: 1 (threshold 0.050)" in output

    def test_a_drop_within_the_threshold_exits_0(self, fake_model):
        _seed_two_tasks(fake_model)
        # alpha 0.9 -> 0.85 is exactly one threshold: allowed.
        output = _check(_run(fake_model, {"alpha": 17, "beta": 16}, "v2"), 0)
        assert "NO RETENTION REGRESSION" in output
        assert "Regressions: 0" in output

    def test_the_threshold_flag_changes_the_verdict(self, fake_model):
        _seed_two_tasks(fake_model)
        profile = {"alpha": 15, "beta": 16}
        _check(_run(fake_model, profile, "v2"), 2)
        _check(_run(fake_model, profile, "v2", "--threshold", "0.2"), 0)
        _check(_run(fake_model, {"alpha": 17, "beta": 16}, "v2", "--threshold", "0"), 2)

    def test_the_best_ever_score_not_the_previous_one_is_the_reference(self, fake_model):
        """v1 0.9, v2 0.7 recorded despite the regression, v3 0.75 is still a regression."""
        _seed_two_tasks(fake_model)
        _check(_run(fake_model, {"alpha": 14, "beta": 16}, "v2", "--record"), 2)
        assert [v["label"] for v in _read_ledger()["versions"]] == ["v1", "v2"]
        output = _check(_run(fake_model, {"alpha": 15, "beta": 16}, "v3"), 2)
        assert "RETENTION REGRESSION: alpha (+0.150)" in output
        # Against the PREVIOUS version (0.7) this would be an improvement.

    def test_mean_forgetting_and_backward_transfer_are_printed_and_written(self, fake_model):
        _write_task_file("alpha")
        _write_task_file("beta")
        _write_task_file("gamma")
        _check(
            _run(fake_model, {"alpha": 16, "beta": 10}, "v1",
                 "--add-task", "alpha=alpha.jsonl", "--add-task", "beta=beta.jsonl", "--record"),
            0,
        )
        _check(_run(fake_model, {"alpha": 12, "beta": 18}, "v2", "--threshold", "1", "--record"), 0)
        result = _run(
            fake_model, {"alpha": 14, "beta": 17, "gamma": 8}, "v3",
            "--add-task", "gamma=gamma.jsonl", "--threshold", "0.15", "--output", "report.json",
        )
        output = _check(result, 0)
        assert "Tasks: 3 (2 comparable, 1 new)" in output
        assert "Mean forgetting: 0.075" in output
        assert "Backward transfer: +0.1250" in output
        report = json.loads(Path("report.json").read_text(encoding="utf-8"))
        assert report["summary"]["mean_forgetting"] == pytest.approx(0.075)
        assert report["summary"]["backward_transfer"] == pytest.approx(0.125)
        assert report["summary"]["new_tasks"] == 1

    def test_the_table_shows_every_version_best_drop_and_status(self, fake_model):
        _seed_two_tasks(fake_model)
        _check(_run(fake_model, {"alpha": 18, "beta": 14}, "v2", "--record"), 2)
        output = _check(_run(fake_model, {"alpha": 18, "beta": 13}, "v3"), 2)
        for text in ("Retention: v3 vs 2 earlier version(s)", "v1", "v2", "v3 (now)", "Best",
                     "Drop", "Status", "REGRESSED", "alpha", "beta", "0.900", "0.700", "0.650",
                     "0.800", "+0.150"):
            assert text in output, text

    def test_older_versions_beyond_the_table_width_are_noted(self, fake_model):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        for index in range(1, 11):
            ledger = rl.append_version(ledger, _version(f"v{index}", {"alpha": 0.9}))
        rl.save_ledger(ledger, "retention.json")
        output = _check(_run(fake_model, {"alpha": 18}, "v11"), 0)
        assert "2 earlier version(s) not shown; --output has the full matrix." in output

    def test_output_holds_the_whole_matrix_and_says_whether_it_was_recorded(self, fake_model):
        _seed_two_tasks(fake_model)
        _check(_run(fake_model, {"alpha": 14, "beta": 16}, "v2", "--output", "r.json"), 2)
        report = json.loads(Path("r.json").read_text(encoding="utf-8"))
        assert report["label"] == "v2" and report["recorded"] is False
        assert report["versions"] == ["v1", "v2"]
        assert report["threshold"] == 0.05
        alpha = next(row for row in report["tasks"] if row["name"] == "alpha")
        assert alpha["scores"] == {"v1": 0.9, "v2": 0.7}
        assert alpha["best_previous"] == 0.9
        assert alpha["status"] == "regressed"
        assert alpha["examples"] == ROWS
        assert report["summary"]["regressed_tasks"] == ["alpha"]
        assert len(_read_ledger()["versions"]) == 1
        _check(_run(fake_model, {"alpha": 14, "beta": 16}, "v2", "--record", "-o", "r2.json"), 2)
        assert json.loads(Path("r2.json").read_text(encoding="utf-8"))["recorded"] is True

    def test_a_regression_is_still_recorded_and_still_exits_2(self, fake_model):
        _seed_two_tasks(fake_model)
        output = _check(_run(fake_model, {"alpha": 10, "beta": 16}, "v2", "--record"), 2)
        assert "Recorded version v2" in output
        assert _read_ledger()["versions"][1]["scores"] == {"alpha": 0.5, "beta": 0.8}

    def test_a_new_task_in_a_later_version_is_recorded_with_that_label(self, fake_model):
        _seed_two_tasks(fake_model)
        _write_task_file("gamma")
        _check(
            _run(fake_model, {"alpha": 18, "beta": 16, "gamma": 12}, "v2",
                 "--add-task", "gamma=gamma.jsonl", "--record"),
            0,
        )
        ledger = _read_ledger()
        gamma = next(task for task in ledger["tasks"] if task["name"] == "gamma")
        assert gamma["added_with"] == "v2"
        assert ledger["versions"][1]["scores"]["gamma"] == 0.6
        assert "gamma" not in ledger["versions"][0]["scores"]

    def test_re_adding_a_task_with_the_same_bytes_is_unchanged(self, fake_model):
        _seed_two_tasks(fake_model)
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--add-task", "alpha=alpha.jsonl"),
            0,
        )
        assert "Control task alpha: unchanged." in output

    def test_the_same_bytes_moved_to_a_new_path_update_the_stored_path(self, fake_model):
        _seed_two_tasks(fake_model)
        os.makedirs("moved")
        os.replace("alpha.jsonl", "moved/alpha.jsonl")
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2",
                 "--add-task", "alpha=moved/alpha.jsonl", "--record"),
            0,
        )
        assert "Control task alpha: moved." in output
        assert next(t for t in _read_ledger()["tasks"] if t["name"] == "alpha")["path"] == (
            "moved/alpha.jsonl"
        )


def _first_run(fake_model, *flags: str, device: str = "cpu"):
    """Register alpha and beta and record v1 on ``out/adapter`` with extra ``flags``."""
    _write_task_file("alpha")
    _write_task_file("beta")
    return _run(
        fake_model, {"alpha": 18, "beta": 16}, "v1",
        "--add-task", "alpha=alpha.jsonl", "--add-task", "beta=beta.jsonl",
        "--adapter", "out/adapter", "--record", *flags, device=device,
    )


class TestLoadParityWithShip:
    """``soup eval retention`` loads the model exactly as ``soup ship`` would."""

    @staticmethod
    def _ship_kwargs(fake_model, *ship_args, device="cpu", **ship_kwargs):
        """The kwargs ``ship._resolve_generators`` gives ``make_generator`` for the adapter."""
        from soup_cli.commands import ship

        fake_model["loads"].clear()
        ship._resolve_generators(
            "base-model", None, "out/adapter", device, *ship_args, **ship_kwargs
        )
        assert len(fake_model["loads"]) == 2, "ship loads the base and the tuned model"
        model_id, kwargs = fake_model["loads"][1]
        assert model_id == "base-model"
        return kwargs

    @staticmethod
    def _retention_kwargs(fake_model):
        assert len(fake_model["loads"]) == 1, "one model load for every task"
        model_id, kwargs = fake_model["loads"][0]
        assert model_id == "base-model"
        return kwargs

    def test_without_flags_the_load_arguments_equal_ships(self, fake_model):
        expected = self._ship_kwargs(fake_model)
        fake_model["loads"].clear()
        _check(_first_run(fake_model), 0)
        assert self._retention_kwargs(fake_model) == expected
        assert expected["dtype"] == "float32" and expected["quantization"] is None

    def test_the_quantization_of_the_config_reaches_the_load_like_ship(self, fake_model):
        from soup_cli.commands import ship
        from soup_cli.config.loader import load_config_from_string

        quantization = ship._live_eval_quantization_from_config(
            load_config_from_string(CONFIG_4BIT)
        )
        expected = self._ship_kwargs(fake_model, quantization)
        assert expected["quantization"] == "4bit" and expected["dtype"] is None
        fake_model["loads"].clear()
        Path("soup.yaml").write_text(CONFIG_4BIT, encoding="utf-8")
        _check(_first_run(fake_model, "--config", "soup.yaml"), 0)
        assert self._retention_kwargs(fake_model) == expected

    def test_the_chat_template_kwargs_reach_the_load_like_ship(self, fake_model):
        template = {"enable_thinking": False}
        expected = self._ship_kwargs(fake_model, None, chat_template_kwargs=template)
        assert expected["chat_template_kwargs"] == template
        fake_model["loads"].clear()
        _check(_first_run(fake_model, "--chat-template-kwargs", '{"enable_thinking": false}'), 0)
        assert self._retention_kwargs(fake_model) == expected

    def test_a_cuda_device_loads_bfloat16_like_ship(self, fake_model):
        expected = self._ship_kwargs(fake_model, device="cuda")
        assert expected["dtype"] == "bfloat16" and expected["device"] == "cuda"
        fake_model["loads"].clear()
        _check(_first_run(fake_model, device="cuda"), 0)
        assert self._retention_kwargs(fake_model) == expected

    def test_a_bad_chat_template_kwargs_value_is_refused_before_any_model_load(self, fake_model):
        _write_task_file("alpha")
        for value, expected in (
            ("{not json", "--chat-template-kwargs:"),
            ("[1, 2]", "expected a JSON object"),
            ('{"tokenize": true}', "cannot override tokenize"),
        ):
            result = _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl",
                          "--chat-template-kwargs", value)
            assert expected in _check(result, 3)
        assert fake_model["loads"] == []

    def test_a_config_that_cannot_be_read_is_refused_before_any_model_load(self, fake_model):
        _write_task_file("alpha")
        result = _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl",
                      "--config", "missing.yaml")
        assert "--config" in _check(result, 3)
        assert fake_model["loads"] == []

    def test_template_kwargs_too_long_to_stamp_are_refused_before_any_model_load(
        self, fake_model
    ):
        _write_task_file("alpha")
        huge = json.dumps({"key": "x" * 600})
        result = _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl",
                      "--chat-template-kwargs", huge)
        assert "too long to record" in _check(result, 3)
        assert fake_model["loads"] == []


_NUMERICS_CPU_DICT = {
    "dtype": "float32", "device": "cpu", "quantization": None, "chat_template_kwargs": None,
}


class TestNumericsInTheLedger:
    def test_each_recorded_version_carries_the_numerics_of_its_run(self, fake_model):
        _seed_two_tasks(fake_model)
        assert _read_ledger()["versions"][0]["numerics"] == _NUMERICS_CPU_DICT

    def test_a_cuda_run_stamps_bfloat16_and_the_device_family(self, fake_model):
        _write_task_file("alpha")
        _check(_run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                    "--record", device="cuda:1"), 0)
        assert _read_ledger()["versions"][0]["numerics"] == {
            **_NUMERICS_CPU_DICT, "dtype": "bfloat16", "device": "cuda",
        }

    def test_a_quantized_run_with_template_kwargs_stamps_both(self, fake_model):
        _write_task_file("alpha")
        Path("soup.yaml").write_text(CONFIG_4BIT, encoding="utf-8")
        _check(_run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                    "--config", "soup.yaml", "--chat-template-kwargs",
                    '{"enable_thinking": false}', "--record"), 0)
        assert _read_ledger()["versions"][0]["numerics"] == {
            "dtype": None, "device": "cpu", "quantization": "4bit",
            "chat_template_kwargs": '{"enable_thinking":false}',
        }

    def test_the_numerics_are_printed_and_written_to_the_report(self, fake_model):
        _write_task_file("alpha")
        output = _check(_run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                             "--output", "r.json"), 0)
        assert "Numerics: dtype float32, device cpu, quantization none" in output
        report = json.loads(Path("r.json").read_text(encoding="utf-8"))
        assert report["numerics"] == _NUMERICS_CPU_DICT

    @pytest.mark.parametrize(
        ("flags", "device", "named"),
        [
            ((), "cuda", ["dtype 'bfloat16' (this run) vs 'float32'",
                          "device 'cuda' (this run) vs 'cpu'"]),
            (("--chat-template-kwargs", '{"enable_thinking": false}'), "cpu",
             ['chat_template_kwargs \'{"enable_thinking":false}\' (this run) vs None']),
        ],
    )
    def test_a_run_under_other_numerics_is_refused_naming_both_values(
        self, fake_model, flags, device, named
    ):
        _seed_two_tasks(fake_model)
        fake_model["loads"].clear()
        before = Path("retention.json").read_bytes()
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record", *flags,
                      device=device)
        output = _check(result, 3)
        for text in named:
            assert text in output, (text, output)
        assert "(version 'v1')" in output
        assert "start a new ledger" in output
        assert fake_model["loads"] == [], "refused before any model load"
        assert Path("retention.json").read_bytes() == before

    def test_a_quantized_run_against_a_full_precision_ledger_is_refused(self, fake_model):
        _seed_two_tasks(fake_model)
        Path("soup.yaml").write_text(CONFIG_4BIT, encoding="utf-8")
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--config", "soup.yaml")
        output = _check(result, 3)
        assert "dtype None (this run) vs 'float32' (version 'v1')" in output
        assert "quantization '4bit' (this run) vs None (version 'v1')" in output

    def test_the_same_numerics_on_another_cuda_index_are_accepted(self, fake_model):
        _write_task_file("alpha")
        _check(_run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                    "--record", device="cuda:0"), 0)
        _check(_run(fake_model, {"alpha": 18}, "v2", "--record", device="cuda:1"), 0)
        assert len(_read_ledger()["versions"]) == 2

    def test_a_dry_run_is_refused_too_when_the_numerics_differ(self, fake_model):
        _seed_two_tasks(fake_model)
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", device="cuda")
        assert "(this run)" in _check(result, 3)


class TestControlFileIntegrity:
    def test_an_edited_control_file_is_refused_naming_the_task_and_both_hashes(
        self, fake_model
    ):
        _seed_two_tasks(fake_model)
        old = _sha("alpha.jsonl")
        _write_task_file("alpha", rows=ROWS - 5)
        new = _sha("alpha.jsonl")
        fake_model["loads"].clear()
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record")
        output = _check(result, 3)
        assert "control task 'alpha' changed" in output
        assert old in output and new in output
        assert fake_model["loads"] == [], "refused before any model load"
        assert len(_read_ledger()["versions"]) == 1

    def test_re_adding_a_name_with_different_bytes_is_refused(self, fake_model):
        _seed_two_tasks(fake_model)
        _write_task_file("other", rows=ROWS - 1)
        old = _sha("alpha.jsonl")
        new = _sha("other.jsonl")
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--add-task", "alpha=other.jsonl"),
            3,
        )
        assert "control task 'alpha' changed" in output
        assert old in output and new in output

    def test_a_deleted_control_file_is_refused(self, fake_model):
        _seed_two_tasks(fake_model)
        os.remove("beta.jsonl")
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2"), 3)
        assert "control task 'beta' file" in output and "is missing" in output
        assert "if it moved, pass --add-task beta=NEWPATH" in output

    def test_the_hinted_add_task_registers_the_moved_file_and_the_run_goes_on(self, fake_model):
        _seed_two_tasks(fake_model)
        os.makedirs("moved")
        os.replace("beta.jsonl", "moved/beta.jsonl")
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2",
                 "--add-task", "beta=moved/beta.jsonl", "--record"),
            0,
        )
        assert "Control task beta: moved." in output

    def test_a_control_file_swapped_after_the_hash_check_and_put_back_is_still_caught(
        self, fake_model, monkeypatch
    ):
        """The bytes that are hashed must be the bytes that are parsed.

        The file is swapped right after the hash check and restored the moment it has been
        parsed: a re-hash at the end of the run sees the original and cannot tell.
        """
        from soup_cli.eval import custom

        _seed_two_tasks(fake_model)
        original = Path("alpha.jsonl").read_bytes()
        real_verify = rl.verify_ledger_files
        calls = {"verify": 0}

        def verify_then_swap(ledger):
            real_verify(ledger)
            calls["verify"] += 1
            if calls["verify"] == 1:
                _write_task_file("alpha", rows=ROWS - 7)

        real_load = custom.load_eval_tasks

        def load_then_restore(path):
            rows = real_load(path)
            Path("alpha.jsonl").write_bytes(original)
            return rows

        monkeypatch.setattr(rl, "verify_ledger_files", verify_then_swap)
        monkeypatch.setattr(custom, "load_eval_tasks", load_then_restore)
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record")
        output = _check(result, 3)
        assert "control task 'alpha' changed" in output
        assert len(_read_ledger()["versions"]) == 1

    def test_a_control_file_that_changes_while_it_is_read_fails_the_hash_of_those_bytes(
        self, fake_model, monkeypatch
    ):
        """The rows the model sees must hash to the ledger's pin, whatever the file does after.

        The file holds other rows while ``_load_control_tasks`` reads it and is put back
        before the end-of-run re-hash, which therefore sees the original.
        """
        _seed_two_tasks(fake_model)
        fake_model["loads"].clear()
        original = Path("alpha.jsonl").read_bytes()
        real_read = rl.read_control_bytes
        calls = {"n": 0}

        def read_while_swapped(path, field):
            calls["n"] += 1
            # reads 1 and 2: verify_ledger_files hashes alpha and beta; read 3: the parse
            # read of alpha in _load_control_tasks
            if calls["n"] != 3:
                return real_read(path, field)
            _write_task_file("alpha", rows=ROWS - 7)
            try:
                return real_read(path, field)
            finally:
                Path("alpha.jsonl").write_bytes(original)

        monkeypatch.setattr(rl, "read_control_bytes", read_while_swapped)
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 3)
        assert "control task 'alpha' changed" in output
        assert calls["n"] >= 3
        assert fake_model["loads"] == [], "refused before any model load"
        assert len(_read_ledger()["versions"]) == 1

    def test_a_control_file_swapped_during_the_run_is_caught_before_recording(
        self, fake_model
    ):
        _seed_two_tasks(fake_model)

        def swap():
            _write_task_file("alpha", rows=ROWS - 7)

        fake_model["on_load"] = swap
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 3)
        assert "control task 'alpha' changed" in output
        assert "nothing was recorded" in output
        assert len(_read_ledger()["versions"]) == 1

    def test_a_control_file_that_is_not_jsonl_is_refused(self, fake_model):
        Path("alpha.txt").write_text('{"prompt": "x", "expected": "y"}\n', encoding="utf-8")
        output = _check(
            _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.txt"), 3
        )
        assert "control task 'alpha' cannot be loaded" in output
        assert "Expected .jsonl" in output

    def test_a_control_file_with_a_bad_row_is_refused(self, fake_model):
        Path("alpha.jsonl").write_text('{"expected": "y"}\n', encoding="utf-8")
        output = _check(_run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl"), 3)
        assert "missing required field 'prompt'" in output

    def test_an_empty_control_file_is_refused(self, fake_model):
        Path("alpha.jsonl").write_text("", encoding="utf-8")
        output = _check(_run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl"), 3)
        assert "control task 'alpha' cannot be loaded" in output
        assert "No eval tasks found in alpha.jsonl" in output


class TestGenerationFailures:
    """A failed generation is not a wrong answer: it must never become a recorded score."""

    @staticmethod
    def _seed_and_snapshot(fake_model) -> bytes:
        _seed_two_tasks(fake_model)
        fake_model["loads"].clear()
        return Path("retention.json").read_bytes()

    def test_a_generator_that_raises_on_every_sample_exits_1_and_records_nothing(
        self, fake_model
    ):
        before = self._seed_and_snapshot(fake_model)
        fake_model["raise_for"] = {"alpha"}
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record", "--output", "r.json")
        output = _check(result, 1)
        assert "generation failed for 20 of 20 samples on control task 'alpha'" in output
        assert "first error: RuntimeError: CUDA out of memory" in output
        assert "nothing was recorded" in output
        assert "RETENTION REGRESSION" not in output, "a failed run is not a regression"
        assert "Recorded version" not in output
        assert Path("retention.json").read_bytes() == before
        assert not Path("r.json").exists(), "no report of scores that were never measured"

    def test_a_generator_that_returns_none_counts_as_a_failure_too(self, fake_model):
        before = self._seed_and_snapshot(fake_model)
        fake_model["none_for"] = {"beta"}
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record")
        output = _check(result, 1)
        assert "generation failed for 20 of 20 samples on control task 'beta'" in output
        assert Path("retention.json").read_bytes() == before

    def test_one_failed_sample_in_a_hundred_is_enough_and_is_counted(self, fake_model):
        before = self._seed_and_snapshot(fake_model)
        fake_model["raise_for"] = {"alpha-3"}
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record")
        output = _check(result, 1)
        assert "generation failed for 1 of 20 samples on control task 'alpha'" in output
        assert Path("retention.json").read_bytes() == before

    def test_a_new_series_whose_generation_fails_leaves_no_ledger_behind(self, fake_model):
        _write_task_file("alpha")
        fake_model["raise_for"] = {"alpha"}
        result = _run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                      "--record")
        _check(result, 1)
        assert not Path("retention.json").exists()

    def test_an_empty_answer_is_a_wrong_answer_not_a_failed_generation(self, fake_model):
        _seed_two_tasks(fake_model)
        fake_model["empty_for"] = {"alpha"}
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 2)
        assert "generation failed" not in output
        assert "RETENTION REGRESSION: alpha (+0.900)" in output
        assert _read_ledger()["versions"][1]["scores"]["alpha"] == 0.0

    def test_a_batch_that_raises_falls_back_to_one_by_one_and_is_not_a_failure(
        self, fake_model
    ):
        _seed_two_tasks(fake_model)

        def broken_batch(prompts, generate):
            raise RuntimeError("batch out of memory")

        fake_model["batch"] = broken_batch
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2"), 0)
        assert "generation failed" not in output

    def test_a_batch_that_raises_then_a_sample_that_raises_counts_that_sample_once(
        self, fake_model
    ):
        self._seed_and_snapshot(fake_model)

        def broken_batch(prompts, generate):
            raise RuntimeError("batch out of memory")

        fake_model["batch"] = broken_batch
        fake_model["raise_for"] = {"alpha-3"}
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 1)
        assert "generation failed for 1 of 20 samples on control task 'alpha'" in output

    def test_a_working_batch_is_used_and_scored(self, fake_model):
        _seed_two_tasks(fake_model)
        calls = []

        def batch(prompts, generate):
            calls.append(len(prompts))
            return [generate(prompt) for prompt in prompts]

        fake_model["batch"] = batch
        output = _check(_run(fake_model, {"alpha": 14, "beta": 16}, "v2"), 2)
        assert calls == [8, 8, 4, 8, 8, 4], "the generator's own batches reached run_eval"
        assert "RETENTION REGRESSION: alpha (+0.200)" in output

    def test_a_batch_of_the_wrong_length_fails_every_prompt_in_it(self, fake_model):
        before = self._seed_and_snapshot(fake_model)
        fake_model["batch"] = lambda prompts, generate: []
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 1)
        assert "generation failed for 20 of 20 samples on control task 'alpha'" in output
        assert Path("retention.json").read_bytes() == before

    def test_a_batch_with_a_non_string_item_fails_that_item(self, fake_model):
        self._seed_and_snapshot(fake_model)
        fake_model["batch"] = lambda prompts, generate: (
            [generate(prompt) for prompt in prompts[:-1]] + [None]
        )
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 1)
        assert "generation failed for 3 of 20 samples on control task 'alpha'" in output


class TestCountingGenerator:
    """``_CountingGenerator`` sits between the model and ``run_eval``."""

    @staticmethod
    def _wrap(inner):
        from soup_cli.commands import _eval_retention as command

        return command._CountingGenerator(inner)

    def test_a_raising_sample_is_counted_and_still_raised_to_the_scorer(self):
        def inner(prompt):
            raise RuntimeError("boom")

        wrapped = self._wrap(inner)
        with pytest.raises(RuntimeError, match="boom"):
            wrapped("p")
        assert wrapped.failures == 1
        assert wrapped.first_error == "RuntimeError: boom"

    def test_a_non_string_sample_is_counted_and_returned_unchanged(self):
        wrapped = self._wrap(lambda prompt: None)
        assert wrapped("p") is None
        assert wrapped.failures == 1

    def test_a_string_sample_including_the_empty_one_is_not_a_failure(self):
        wrapped = self._wrap(lambda prompt: "")
        assert wrapped("p") == ""
        assert wrapped.failures == 0

    def test_it_exposes_batch_only_when_the_model_does(self):
        assert not hasattr(self._wrap(lambda prompt: "x"), "batch")
        inner = lambda prompt: "x"  # noqa: E731 - a bare callable with attributes
        inner.batch = lambda prompts: ["x"] * len(prompts)
        inner.batch_size = 3
        wrapped = self._wrap(inner)
        assert wrapped.batch(["a", "b"]) == ["x", "x"]
        assert wrapped.batch_size == 3
        assert wrapped.failures == 0

    def test_a_malformed_batch_counts_every_prompt_and_a_bad_item_counts_once(self):
        inner = lambda prompt: "x"  # noqa: E731
        inner.batch = lambda prompts: ["x"]
        wrapped = self._wrap(inner)
        wrapped.batch(["a", "b", "c"])
        assert wrapped.failures == 3
        inner.batch = lambda prompts: ["x", None, 5]
        again = self._wrap(inner)
        again.batch(["a", "b", "c"])
        assert again.failures == 2

    def test_a_raising_batch_is_not_counted_the_retry_is(self):
        def inner(prompt):
            return "x" if prompt != "b" else None

        def broken(prompts):
            raise RuntimeError("batch out of memory")

        inner.batch = broken
        wrapped = self._wrap(inner)
        with pytest.raises(RuntimeError):
            wrapped.batch(["a", "b"])
        assert wrapped.failures == 0, "iter_outputs retries one by one through __call__"


class TestConcurrentRuns:
    """The ledger is re-read right before it is written: no run overwrites another's version."""

    def test_a_ledger_that_another_run_changed_is_not_overwritten(self, fake_model):
        _seed_two_tasks(fake_model)

        def another_run_records():
            ledger = rl.load_ledger("retention.json")
            other = _version("v2-other", {"alpha": 0.9, "beta": 0.8})
            rl.save_ledger(rl.append_version(ledger, other), "retention.json")

        fake_model["on_load"] = another_run_records
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record")
        output = _check(result, 1)
        assert "cannot record in --ledger" in output
        assert "changed since this run read it" in output
        assert "Recorded version" not in output and "Dry run" not in output
        assert [v["label"] for v in _read_ledger()["versions"]] == ["v1", "v2-other"]

    def test_the_same_label_recorded_by_the_other_run_is_refused_not_duplicated(
        self, fake_model
    ):
        _seed_two_tasks(fake_model)

        def another_run_records_v2():
            ledger = rl.load_ledger("retention.json")
            theirs = _version("v2", {"alpha": 0.5, "beta": 0.5})
            rl.save_ledger(rl.append_version(ledger, theirs), "retention.json")

        fake_model["on_load"] = another_run_records_v2
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record")
        output = _check(result, 1)
        assert "version label 'v2' was recorded by another run" in output
        versions = _read_ledger()["versions"]
        assert [v["label"] for v in versions] == ["v1", "v2"]
        assert versions[1]["scores"] == {"alpha": 0.5, "beta": 0.5}, "the other run's v2 stays"

    def test_a_ledger_that_appeared_while_a_new_series_ran_is_not_overwritten(self, fake_model):
        _write_task_file("alpha")

        def another_run_starts_the_series():
            ledger, _ = rl.register_task(
                rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP
            )
            rl.save_ledger(rl.append_version(ledger, _version("v1", {"alpha": 0.1})),
                           "retention.json")

        fake_model["on_load"] = another_run_starts_the_series
        result = _run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                      "--record")
        output = _check(result, 1)
        assert "version label 'v1' was recorded by another run" in output
        assert _read_ledger()["versions"][0]["scores"] == {"alpha": 0.1}

    def test_an_untouched_ledger_is_still_recorded(self, fake_model):
        _seed_two_tasks(fake_model)
        _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 0)
        assert [v["label"] for v in _read_ledger()["versions"]] == ["v1", "v2"]


class TestUsageErrors:
    @pytest.mark.parametrize(
        ("args", "missing"),
        [(["--label", "v1"], "--model"), (["--model", "m"], "--label")],
    )
    def test_missing_required_flags_exit_3_naming_the_flag(self, args, missing):
        result = runner.invoke(app, ["eval", "retention", *args])
        output = _plain(result.output)
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert f"Missing option '{missing}'" in output, output

    def test_a_missing_ledger_without_add_task_says_how_to_start_one(self, fake_model):
        output = _check(_run(fake_model, {}, "v1"), 3)
        assert "does not exist" in output
        assert "start a ledger with --add-task NAME=PATH" in output
        assert fake_model["loads"] == []

    def test_a_ledger_with_no_tasks_is_refused(self, fake_model):
        rl.save_ledger(rl.Ledger(), "retention.json")
        output = _check(_run(fake_model, {}, "v1"), 3)
        assert "the ledger has no control tasks" in output

    def test_a_malformed_ledger_is_named_not_a_traceback(self, fake_model):
        payload = _valid_payload()
        payload["versions"][0]["scores"]["alpha"] = "high"
        Path("retention.json").write_text(json.dumps(payload), encoding="utf-8")
        result = _run(fake_model, {}, "v2")
        output = _check(result, 3)
        assert "cannot read --ledger" in output
        assert "scores['alpha'] must be a number, got str" in output
        assert "Traceback" not in result.output

    def test_invalid_json_in_the_ledger_exits_3(self, fake_model):
        Path("retention.json").write_text("{nope", encoding="utf-8")
        output = _check(_run(fake_model, {}, "v1"), 3)
        assert "is not valid JSON" in output

    def test_a_400_digit_integer_score_in_the_ledger_exits_3_without_a_traceback(
        self, fake_model
    ):
        _seed_two_tasks(fake_model)
        text = Path("retention.json").read_text(encoding="utf-8").replace("0.9", "9" * 400)
        Path("retention.json").write_text(text, encoding="utf-8")
        fake_model["loads"].clear()
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2")
        output = _check(result, 3)
        assert "cannot read --ledger" in output
        assert "scores['alpha'] must be between 0.0 and 1.0" in output
        assert "Traceback" not in result.output and "OverflowError" not in result.output
        assert "9" * 100 not in output
        assert fake_model["loads"] == []

    def test_require_history_refuses_a_ledger_that_would_start_a_new_series(self, fake_model):
        """A lost or reset ledger makes every task new, so nothing can regress and CI passes."""
        _write_task_file("alpha")
        result = _run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl",
                      "--require-history", "--record")
        output = _check(result, 3)
        assert "--require-history: the ledger holds no earlier version" in output
        assert fake_model["loads"] == []
        assert not Path("retention.json").exists()

    def test_require_history_refuses_a_ledger_with_tasks_but_no_version(self, fake_model):
        _write_task_file("alpha")
        ledger, _ = rl.register_task(rl.Ledger(), "alpha", "alpha.jsonl", label="v1", now=STAMP)
        rl.save_ledger(ledger, "retention.json")
        result = _run(fake_model, {"alpha": 18}, "v1", "--require-history")
        assert "holds no earlier version" in _check(result, 3)
        assert fake_model["loads"] == []

    def test_require_history_lets_a_ledger_with_history_through(self, fake_model):
        _seed_two_tasks(fake_model)
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--require-history"), 0
        )
        assert "NO RETENTION REGRESSION" in output
        assert "the ledger has no earlier version" not in output

    def test_without_require_history_a_new_series_warns_and_names_the_flag(self, fake_model):
        _write_task_file("alpha")
        output = _check(_run(fake_model, {"alpha": 18}, "v1", "--add-task", "alpha=alpha.jsonl"), 0)
        assert "Warning: the ledger has no earlier version" in output
        assert "lost or reset" in output and "--require-history" in output

    def test_a_duplicate_label_is_refused_before_any_model_load(self, fake_model):
        _seed_two_tasks(fake_model)
        fake_model["loads"].clear()
        before = Path("retention.json").read_bytes()
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v1", "--record"), 3)
        assert "version label 'v1' is already recorded" in output
        assert "labels must be unique" in output
        assert fake_model["loads"] == []
        assert Path("retention.json").read_bytes() == before

    @pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "inf"])
    def test_a_bad_threshold_exits_3(self, fake_model, value):
        _write_task_file("alpha")
        output = _check(
            _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl", "--threshold", value), 3
        )
        assert "threshold must be between 0.0 and 1.0" in output

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            (["--add-task", "alpha"], "--add-task expects NAME=PATH"),
            (["--add-task", "bad/name=a.jsonl"], "task name 'bad/name' is invalid"),
            (["--add-task", "alpha=a.jsonl", "--add-task", "alpha=a.jsonl"],
             "--add-task names 'alpha' twice"),
        ],
    )
    def test_bad_add_task_flags_exit_3(self, fake_model, flags, expected):
        _write_task_file("a")
        assert expected in _check(_run(fake_model, {}, "v1", *flags), 3)

    @pytest.mark.parametrize("label", ["", " v1", "v1 ", "v\x07", "x" * 129])
    def test_a_bad_label_exits_3(self, fake_model, label):
        _write_task_file("alpha")
        result = _run(fake_model, {}, label, "--add-task", "alpha=alpha.jsonl")
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "version label" in _plain(result.output)

    def test_an_empty_model_exits_3(self, fake_model):
        _write_task_file("alpha")
        result = _invoke("--model", "", "--label", "v1", "--add-task", "alpha=alpha.jsonl")
        assert "--model '' must be non-empty" in _check(result, 3)


class TestPathContainment:
    def test_a_ledger_outside_the_working_directory_is_refused(self, fake_model):
        _write_task_file("alpha")
        output = _check(
            _run(fake_model, {}, "v1", "--ledger", "../retention.json",
                 "--add-task", "alpha=alpha.jsonl", "--record"),
            3,
        )
        assert "--ledger '../retention.json' must stay under cwd" in output
        assert fake_model["loads"] == []

    def test_an_absolute_ledger_outside_the_working_directory_is_refused(
        self, fake_model, tmp_path
    ):
        outside = str(tmp_path.parent / "retention.json")
        _write_task_file("alpha")
        output = _check(
            _run(fake_model, {}, "v1", "--ledger", outside, "--add-task", "alpha=alpha.jsonl"),
            3,
        )
        assert "must stay under cwd" in output

    def test_a_ledger_path_that_is_a_directory_is_refused(self, fake_model):
        os.mkdir("retention.json")
        _write_task_file("alpha")
        output = _check(_run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl"), 3)
        assert "cannot read --ledger" in output
        assert fake_model["loads"] == []

    def test_a_control_file_outside_the_working_directory_is_refused(self, fake_model):
        output = _check(_run(fake_model, {}, "v1", "--add-task", "alpha=../a.jsonl"), 3)
        assert "'../a.jsonl' must stay under cwd" in output
        assert fake_model["loads"] == []

    def test_an_output_outside_the_working_directory_is_refused_before_the_run(
        self, fake_model
    ):
        _write_task_file("alpha")
        output = _check(
            _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl", "--output", "../r.json"),
            3,
        )
        assert "--output '../r.json' must stay under cwd" in output
        assert fake_model["loads"] == []

    def test_an_output_may_not_overwrite_the_ledger(self, fake_model):
        _seed_two_tasks(fake_model)
        before = Path("retention.json").read_bytes()
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--output", "retention.json"), 3
        )
        assert "is the same file as --ledger" in output
        assert Path("retention.json").read_bytes() == before

    def test_an_output_may_not_overwrite_a_control_file(self, fake_model):
        _seed_two_tasks(fake_model)
        before = Path("alpha.jsonl").read_bytes()
        output = _check(
            _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--output", "alpha.jsonl"), 3
        )
        assert "is the same file as control task alpha" in output
        assert Path("alpha.jsonl").read_bytes() == before

    def test_an_output_that_is_a_directory_is_refused(self, fake_model):
        _write_task_file("alpha")
        os.mkdir("report")
        output = _check(
            _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl", "--output", "report"), 3
        )
        assert "must name a file, not a directory" in output

    @pytest.mark.requires_symlink
    def test_a_symlinked_ledger_is_refused(self, fake_model):
        _seed_two_tasks(fake_model)
        os.replace("retention.json", "real.json")
        os.symlink("real.json", "retention.json")
        output = _check(_run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record"), 3)
        assert "must not be a symlink" in output

    @pytest.mark.requires_symlink
    def test_a_symlinked_output_is_refused(self, fake_model):
        _write_task_file("alpha")
        Path("victim.txt").write_text("keep", encoding="utf-8")
        os.symlink("victim.txt", "report.json")
        output = _check(
            _run(fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl",
                 "--output", "report.json"),
            3,
        )
        assert "must not be a symlink" in output
        assert Path("victim.txt").read_text(encoding="utf-8") == "keep"


class TestRuntimeErrors:
    def test_a_model_that_fails_to_load_exits_1_and_records_nothing(
        self, fake_model, monkeypatch
    ):
        from soup_cli.utils import live_eval

        def boom(*_args, **_kwargs):
            raise OSError("no such model")

        monkeypatch.setattr(live_eval, "make_generator", boom)
        _write_task_file("alpha")
        result = _run(
            fake_model, {}, "v1", "--add-task", "alpha=alpha.jsonl", "--record"
        )
        output = _check(result, 1)
        assert "retention run failed: OSError: no such model" in output
        assert not Path("retention.json").exists()

    def test_a_ledger_that_cannot_be_written_exits_1_after_the_report(
        self, fake_model, monkeypatch
    ):
        _write_task_file("alpha")
        real_replace = os.replace

        def refuse_the_ledger(src, dst, *args, **kwargs):
            if os.path.basename(str(dst)) == "retention.json":
                raise OSError("read-only volume")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", refuse_the_ledger)
        result = _run(
            fake_model, {"alpha": 18}, "v1",
            "--add-task", "alpha=alpha.jsonl", "--record", "--output", "r.json",
        )
        output = _check(result, 1)
        assert "cannot record in --ledger: OSError: read-only volume" in output
        assert "NO RETENTION REGRESSION" in output
        assert "Recorded version" not in output
        assert "Dry run" not in output, "--record was passed: it was not a dry run"
        assert "pass --record" not in output
        assert not Path("retention.json").exists()
        report = json.loads(Path("r.json").read_text(encoding="utf-8"))
        assert report["recorded"] is False
        assert sorted(name for name in os.listdir(".") if name.startswith(".soup")) == []

    @staticmethod
    def _refuse_writes_to(monkeypatch, name: str) -> None:
        real_replace = os.replace

        def refuse(src, dst, *args, **kwargs):
            if os.path.basename(str(dst)) == name:
                raise OSError("read-only volume")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", refuse)

    def test_a_regression_is_not_hidden_by_a_ledger_write_failure(self, fake_model, monkeypatch):
        _seed_two_tasks(fake_model)
        self._refuse_writes_to(monkeypatch, "retention.json")
        result = _run(fake_model, {"alpha": 10, "beta": 16}, "v2", "--record")
        output = _check(result, 2)
        assert "RETENTION REGRESSION: alpha (+0.400)" in output
        assert "cannot record in --ledger: OSError: read-only volume" in output
        assert "Recorded version" not in output and "Dry run" not in output
        assert len(_read_ledger()["versions"]) == 1

    def test_a_regression_is_not_hidden_by_a_failed_output_write(self, fake_model, monkeypatch):
        _seed_two_tasks(fake_model)
        self._refuse_writes_to(monkeypatch, "r.json")
        result = _run(fake_model, {"alpha": 10, "beta": 16}, "v2", "--output", "r.json")
        output = _check(result, 2)
        assert "RETENTION REGRESSION: alpha (+0.400)" in output
        assert "cannot write --output: OSError: read-only volume" in output
        assert not Path("r.json").exists()

    def test_a_failed_output_write_without_a_regression_exits_1(self, fake_model, monkeypatch):
        _seed_two_tasks(fake_model)
        self._refuse_writes_to(monkeypatch, "r.json")
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--output", "r.json")
        output = _check(result, 1)
        assert "cannot write --output: OSError: read-only volume" in output
        assert "NO RETENTION REGRESSION" in output

    def test_both_a_failed_ledger_write_and_a_failed_output_write_are_reported(
        self, fake_model, monkeypatch
    ):
        _seed_two_tasks(fake_model)
        real_replace = os.replace

        def refuse(src, dst, *args, **kwargs):
            if os.path.basename(str(dst)) in ("retention.json", "r.json"):
                raise OSError("read-only volume")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", refuse)
        result = _run(fake_model, {"alpha": 18, "beta": 16}, "v2", "--record", "--output", "r.json")
        output = _check(result, 1)
        assert "cannot record in --ledger" in output
        assert "cannot write --output" in output


class TestTerminalSafety:
    def test_a_duplicate_label_with_markup_is_printed_as_typed_not_double_escaped(
        self, fake_model
    ):
        _write_task_file("alpha")
        _check(_run(fake_model, {"alpha": 18}, "[red]x", "--add-task", "alpha=alpha.jsonl",
                    "--record"), 0)
        output = _check(_run(fake_model, {"alpha": 18}, "[red]x"), 3)
        assert "version label '[red]x' is already recorded in the ledger" in output, output
        assert "\\" not in output, "no stray backslash from escaping twice"

    def test_markup_in_a_label_is_printed_literally(self, fake_model):
        _write_task_file("alpha")
        result = _run(
            fake_model, {"alpha": 18}, "[red]x[/]", "--add-task", "alpha=alpha.jsonl"
        )
        output = _check(result, 0)
        assert "Retention: [red]x[/] vs 0 earlier version(s)" in output
        assert "[red]x[/] (now)" in output

    def test_markup_in_an_earlier_label_and_in_paths_is_printed_literally(self, fake_model):
        _write_task_file("alpha")
        _check(
            _run(fake_model, {"alpha": 18}, "[red]v1[/]", "--add-task", "alpha=alpha.jsonl",
                 "--ledger", "[b]led.json", "--record"),
            0,
        )
        output = _check(
            _run(fake_model, {"alpha": 18}, "v2", "--ledger", "[b]led.json",
                 "--output", "[i]rep.json", "--record"),
            0,
        )
        assert "[red]v1[/]" in output, "earlier version column header"
        assert "Recorded version v2 in [b]led.json" in output
        assert "Wrote report [i]rep.json" in output
        started = _check(
            _run(fake_model, {"alpha": 18}, "v1", "--ledger", "[u]new.json",
                 "--add-task", "alpha=alpha.jsonl"),
            0,
        )
        assert "No ledger at [u]new.json; starting a new one." in started

    def test_a_row_name_that_bypassed_validation_is_still_escaped(self):
        """The renderer does not rely on the loader having cleaned the names."""
        import io

        from rich.console import Console

        from soup_cli.commands import _eval_retention as command

        hostile = "[blink]evil\x1b]0;owned\x07"
        row = rl.TaskRetention(
            name=hostile, scores={"[red]v1": 0.9, "[b]now": 0.8}, current=0.8, first=0.9,
            best_previous=0.9, drop=0.1, status=rl.STATUS_REGRESSED, examples=20,
        )
        report = rl.RetentionReport(
            label="[b]now", threshold=0.05, versions=("[red]v1", "[b]now"), tasks=(row,)
        )
        buffer = io.StringIO()
        console = Console(file=buffer, color_system=None, width=200)
        command._print_report(console, report, recorded=False, ledger_path="[u]x.json")
        shown = buffer.getvalue()
        assert "[blink]evil" in shown and "[red]v1" in shown and "[b]now (now)" in shown
        assert "\x1b" not in shown and "\x07" not in shown

    def test_a_hostile_ledger_never_reaches_the_terminal_raw(self, fake_model):
        payload = _valid_payload()
        payload["tasks"][0]["name"] = "evil\x1b]0;owned\x07[red]"
        Path("retention.json").write_text(json.dumps(payload), encoding="utf-8")
        result = _run(fake_model, {}, "v2")
        _check(result, 3)
        # Rich's own colour codes are fine (FORCE_COLOR=1 paints "Error:"); the hostile
        # OSC title sequence from the ledger must not survive them being stripped.
        shown = strip_ansi(result.output)
        assert "\x1b" not in shown and "\x07" not in shown
        # ... and the markup-looking tail of the name is shown, not interpreted.
        assert "owned[red]" in _plain(result.output)


# ---------------------------------------------------------------------------
# The documentation says what the code does
# ---------------------------------------------------------------------------

_EVALUATION_DOCS = Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"


def _retention_docs() -> str:
    text = _EVALUATION_DOCS.read_text(encoding="utf-8")
    start = text.index("## Retention across versions")
    return text[start:text.index("\n## ", start + 1)]


class TestRetentionDocs:
    def test_the_section_avoids_the_banned_wording(self):
        docs = _retention_docs()
        assert not re.search(r"\bpoints?\b", docs, re.IGNORECASE)
        assert "reward" not in docs.lower()

    def test_the_name_rule_names_the_first_character(self):
        assert "start with a letter or digit" in _retention_docs()

    def test_the_ledger_is_not_touched_without_record_but_output_may_write_a_file(self):
        docs = _retention_docs()
        assert "the ledger is not touched" in docs
        assert "Nothing is written" not in docs

    def test_a_moved_control_file_has_its_hint(self):
        docs = _retention_docs()
        assert "--add-task NAME=NEWPATH" in docs
        assert "updates the stored path" in docs

    def test_the_threshold_is_described_as_a_fraction(self):
        assert "absolute accuracy difference (a fraction, 0.05 = 5%)" in _retention_docs()

    def test_the_load_flags_the_stamp_and_the_history_flag_are_documented(self):
        docs = _retention_docs()
        for text in ("--config", "--chat-template-kwargs", "--require-history", "numerics",
                     "bfloat16"):
            assert text in docs, text

    def test_the_known_limitations_are_written_down(self):
        docs = _retention_docs()
        assert "Known limitations" in docs
        assert "same instant" in docs, "the residual race between two recording runs"

    def test_the_code_comments_avoid_the_banned_wording_too(self):
        root = Path(__file__).resolve().parents[1] / "src" / "soup_cli"
        for relative in ("commands/_eval_retention.py", "utils/retention_ledger.py"):
            text = (root / relative).read_text(encoding="utf-8")
            assert not re.search(r"\bpoints?\b", text, re.IGNORECASE), relative


# ---------------------------------------------------------------------------
# Import weight
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module", ["soup_cli.utils.retention_ledger", "soup_cli.commands._eval_retention"]
)
def test_the_modules_stay_torch_free(module):
    """The claim is about ``sys.modules`` in a fresh interpreter, not about syntax."""
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys, {module}; "
            f"print(','.join(sorted(m for m in {HEAVY!r} if m in sys.modules)))",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert proc.stdout.strip() == "", proc.stdout
