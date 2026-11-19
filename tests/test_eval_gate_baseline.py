"""``training.eval_gate.baseline``: refused forms and no silent empty baseline (#2523).

``baseline: previous`` was documented but never implemented: ``resolve_baseline``
read it as a file named ``previous``, and the epoch-end gate then logged one
warning and trained against an empty baseline, so no regression could fire.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import EvalGateConfig


class _StubDisplay:
    def start(self, total_steps=0):
        self.total_steps = total_steps


class _FakeArgs:
    pass


class _FakeState:
    def __init__(self, epoch=0, max_steps=1):
        self.epoch = epoch
        self.max_steps = max_steps
        self.global_step = 0


class _FakeControl:
    def __init__(self):
        self.should_training_stop = False


_PREVIOUS_REFUSED = (
    "training.eval_gate.baseline: 'previous' is not implemented; "
    "use a path to a baseline results file"
)


def _write_gate_suite(tmp_path, threshold=0.0):
    (tmp_path / "tasks.jsonl").write_text(
        '{"prompt": "2+2?", "expected": "4", "scoring": "exact"}\n', encoding="utf-8"
    )
    (tmp_path / "gate.yaml").write_text(
        "suite: t\n"
        "tasks:\n"
        "  - type: custom\n"
        "    name: math\n"
        f"    threshold: {threshold}\n"
        "    tasks: tasks.jsonl\n"
        "    scorer: exact\n",
        encoding="utf-8",
    )


def _callback(baseline, on_regression="stop"):
    from soup_cli.monitoring.callback import SoupTrainerCallback

    cfg = EvalGateConfig(
        enabled=True,
        suite="gate.yaml",
        baseline=baseline,
        on_regression=on_regression,
        regression_threshold=0.05,
    )
    cb = SoupTrainerCallback(_StubDisplay(), eval_gate_config=cfg)
    cb._gate_generate_fn = lambda prompt: "4"
    return cb


# ---------------------------------------------------------------------------
# Config load: 'previous' is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["previous", "Previous", " PREVIOUS ", "previous\n"])
def test_previous_baseline_is_refused(value):
    with pytest.raises(ValidationError) as exc_info:
        EvalGateConfig(enabled=True, suite="gate.yaml", baseline=value)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("baseline",)
    assert errors[0]["msg"] == f"Value error, {_PREVIOUS_REFUSED}"


_SOUP_YAML = (
    "base: some/model\n"
    "task: sft\n"
    "data:\n"
    "  train: data.jsonl\n"
    "training:\n"
    "  eval_gate:\n"
    "    enabled: true\n"
    "    suite: gate.yaml\n"
    "    baseline: {baseline}\n"
)


def test_previous_baseline_is_refused_from_soup_yaml():
    # The same document with a supported baseline loads, so the refusal below
    # is about the baseline value and nothing else.
    cfg = load_config_from_string(_SOUP_YAML.format(baseline="registry://v1"))
    assert cfg.training.eval_gate.baseline == "registry://v1"

    with pytest.raises(ValueError) as exc_info:
        load_config_from_string(_SOUP_YAML.format(baseline="previous"))
    assert str(exc_info.value) == (
        f"training -> eval_gate -> baseline: Value error, {_PREVIOUS_REFUSED}"
    )


@pytest.mark.parametrize(
    "value",
    [None, "", "registry://llama31-chat-v1", "baseline.json", "./previous", "previous.json"],
)
def test_supported_baseline_forms_still_load(value):
    cfg = EvalGateConfig(enabled=True, suite="gate.yaml", baseline=value)
    assert cfg.baseline == value


def test_field_description_no_longer_promises_previous():
    description = EvalGateConfig.model_fields["baseline"].description or ""
    assert "previous" not in description
    assert "registry://" in description


# ---------------------------------------------------------------------------
# Training callback: a requested baseline that cannot load is an error
# ---------------------------------------------------------------------------


def test_missing_baseline_file_aborts_gate_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_gate_suite(tmp_path)
    cb = _callback("missing.json")
    with pytest.raises(ValueError) as exc_info:
        cb.on_train_begin(_FakeArgs(), _FakeState(), _FakeControl())
    assert str(exc_info.value) == (
        "eval gate: could not resolve training.eval_gate.baseline 'missing.json': "
        "baseline file not found: missing.json"
    )
    assert cb._gate_baseline is None


def test_missing_registry_baseline_aborts_gate_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    _write_gate_suite(tmp_path)
    cb = _callback("registry://nonexistent_entry", on_regression="continue")
    with pytest.raises(ValueError, match="training.eval_gate.baseline"):
        cb.on_train_begin(_FakeArgs(), _FakeState(), _FakeControl())


def test_invalid_json_baseline_aborts_gate_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_gate_suite(tmp_path)
    (tmp_path / "baseline.json").write_text("{not json", encoding="utf-8")
    cb = _callback("baseline.json", on_regression="warn")
    with pytest.raises(ValueError, match="training.eval_gate.baseline"):
        cb.on_train_begin(_FakeArgs(), _FakeState(), _FakeControl())


# ---------------------------------------------------------------------------
# Unchanged: no baseline, and valid file / registry baselines
# ---------------------------------------------------------------------------


def test_no_baseline_still_means_empty_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_gate_suite(tmp_path)
    cb = _callback(None)
    args, state, control = _FakeArgs(), _FakeState(), _FakeControl()
    cb.on_train_begin(args, state, control)
    assert cb._gate_baseline == {}
    state.epoch = 1
    cb.on_epoch_end(args, state, control)
    assert control.should_training_stop is False


def test_file_baseline_still_drives_regression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_gate_suite(tmp_path)
    (tmp_path / "baseline.json").write_text(json.dumps({"math": 1.0}), encoding="utf-8")
    cb = _callback("baseline.json")
    cb._gate_generate_fn = lambda prompt: "wrong"
    args, state, control = _FakeArgs(), _FakeState(), _FakeControl()
    cb.on_train_begin(args, state, control)
    assert cb._gate_baseline == {"math": 1.0}
    state.epoch = 1
    cb.on_epoch_end(args, state, control)
    assert control.should_training_stop is True


def test_registry_baseline_still_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    from soup_cli.registry.store import RegistryStore

    store = RegistryStore()
    eid = store.push(
        name="baseline", tag="v1", base_model="llama", task="sft", run_id=None, config={}
    )
    store.close()
    _write_gate_suite(tmp_path)
    cb = _callback(f"registry://{eid}")
    cb.on_train_begin(_FakeArgs(), _FakeState(), _FakeControl())
    assert cb._gate_baseline == {}
