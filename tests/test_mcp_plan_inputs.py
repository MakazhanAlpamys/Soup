"""MCP train plans pin every local input the spawned run reads."""

from __future__ import annotations

import re
import typing
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import soup_cli.mcp_server.registry as reg
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager

_PATH_LIKE = re.compile(
    r"path|dir|file|model|set$|train|replay|suite|base|teacher|reward|adapter|checkpoint|"
    r"tokenizer|dataset|corpus|bank|prompts|tasks|baseline|output|resume|ref|vocab|template|"
    r"init|draft|judge"
)


def _path_like_fields() -> set[str]:
    found: set[str] = set()

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            ann = info.annotation
            subs = [
                a
                for a in typing.get_args(ann)
                if isinstance(a, type) and issubclass(a, BaseModel)
            ]
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                subs.append(ann)
            for sub in subs:
                walk(sub, f"{prefix}{name}.")
            if "str" in str(ann) and _PATH_LIKE.search(name):
                found.add(prefix + name)

    walk(SoupConfig, "")
    return found


def test_every_path_like_field_is_classified():
    classified = set(reg.PLAN_INPUT_FIELDS) | set(reg.NOT_PLAN_INPUT_FIELDS)
    missing = _path_like_fields() - classified
    assert not missing, f"classify in PLAN_INPUT_FIELDS or NOT_PLAN_INPUT_FIELDS: {sorted(missing)}"


def test_classified_fields_exist_on_schema():
    fields = _path_like_fields()
    for name in list(reg.PLAN_INPUT_FIELDS) + list(reg.NOT_PLAN_INPUT_FIELDS):
        assert name in fields, f"{name} is not a field of SoupConfig"


def test_no_overlap():
    assert not set(reg.PLAN_INPUT_FIELDS) & set(reg.NOT_PLAN_INPUT_FIELDS)


def test_not_plan_input_fields_carry_a_reason():
    for name, reason in reg.NOT_PLAN_INPUT_FIELDS.items():
        assert isinstance(reason, str) and reason.strip(), name


def test_nonexistent_fields_removed():
    assert "data.eval" not in reg.PLAN_INPUT_FIELDS
    assert "training.adapter" not in reg.PLAN_INPUT_FIELDS


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_text(
        '{"instruction": "hi", "output": "hello"}\n', encoding="utf-8"
    )
    return tmp_path


def _write_config(directory, text: str) -> None:
    # A schema error must fail the test loudly rather than read as a refusal.
    load_config_from_string(text)
    (directory / "soup.yaml").write_text(text, encoding="utf-8")


def _plan(manager: ExecutionManager) -> str:
    specs = reg.build_registry(allow_mutating=False, allow_execute=True, execution=manager)
    spec = next(s for s in specs if s.name == "train_start")
    return spec.handler({"config": "soup.yaml"})["confirmation_token"]


def _assert_refused(manager: ExecutionManager, token: str) -> None:
    with patch("subprocess.Popen") as mock_popen:
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="train")
        assert not mock_popen.called
    assert "planned input changed" in str(exc.value).lower()


def _mock_process() -> MagicMock:
    proc = MagicMock()
    proc.pid = 4242
    proc.wait.return_value = 0
    return proc


def test_reward_model_dir_change_refused(project):
    rm_dir = project / "rm"
    rm_dir.mkdir()
    (rm_dir / "model.safetensors").write_bytes(b"reward-weights-v1")
    _write_config(
        project,
        "base: Qwen/Qwen2.5-0.5B\ntask: ppo\ndata:\n  train: data.jsonl\n"
        "  format: chatml\ntraining:\n  reward_model: ./rm\n",
    )
    manager = ExecutionManager()
    token = _plan(manager)
    (rm_dir / "model.safetensors").write_bytes(b"reward-weights-v2")
    _assert_refused(manager, token)


def test_reward_fn_py_change_refused(project):
    reward = project / "my_reward.py"
    reward.write_text(
        "def reward_fn(completions, **kwargs):\n    return [1.0] * len(completions)\n",
        encoding="utf-8",
    )
    _write_config(
        project,
        "base: Qwen/Qwen2.5-0.5B\ntask: grpo\ndata:\n  train: data.jsonl\n"
        "training:\n  reward_fn: ./my_reward.py\n",
    )
    manager = ExecutionManager()
    token = _plan(manager)
    reward.write_text(
        "def reward_fn(completions, **kwargs):\n    return [0.0] * len(completions)\n",
        encoding="utf-8",
    )
    _assert_refused(manager, token)


def test_reward_fn_ensemble_segments_pinned(project):
    reward = project / "my_reward.py"
    reward.write_text(
        "def reward_fn(completions, **kwargs):\n    return [1.0] * len(completions)\n",
        encoding="utf-8",
    )
    _write_config(
        project,
        "base: Qwen/Qwen2.5-0.5B\ntask: grpo\ndata:\n  train: data.jsonl\n"
        'training:\n  reward_fn: "accuracy, ./my_reward.py"\n',
    )
    manager = ExecutionManager()
    token = _plan(manager)
    reward.write_text(
        "def reward_fn(completions, **kwargs):\n    return [0.5] * len(completions)\n",
        encoding="utf-8",
    )
    _assert_refused(manager, token)


def test_forget_set_change_refused(project):
    forget = project / "forget.jsonl"
    forget.write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    _write_config(
        project,
        "base: Qwen/Qwen2.5-0.5B\ntask: unlearn\ndata:\n  train: data.jsonl\n"
        "  forget_set: forget.jsonl\ntraining:\n  unlearn_method: npo\n",
    )
    manager = ExecutionManager()
    token = _plan(manager)
    forget.write_text('{"instruction": "a", "output": "c"}\n', encoding="utf-8")
    _assert_refused(manager, token)


def test_absent_input_created_after_plan_refused(project):
    _write_config(
        project,
        "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n"
        "  replay: later.jsonl\n",
    )
    manager = ExecutionManager()
    token = _plan(manager)
    (project / "later.jsonl").write_text(
        '{"instruction": "x", "output": "y"}\n', encoding="utf-8"
    )
    _assert_refused(manager, token)


def test_hub_id_base_does_not_block_execution(project):
    _write_config(
        project,
        "base: HuggingFaceTB/SmolLM2-135M\ntask: sft\ndata:\n  train: data.jsonl\n",
    )
    manager = ExecutionManager()
    token = _plan(manager)
    with patch("subprocess.Popen", return_value=_mock_process()):
        result = manager.execute(token=token, kind="train")
    assert result["status"] == "running"


def test_existing_input_outside_cwd_refused_at_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    outside = tmp_path / "outside"
    outside.mkdir()
    data = outside / "data.jsonl"
    data.write_text('{"instruction": "hi", "output": "hello"}\n', encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    _write_config(
        proj,
        "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n"
        f"  train: '{data.as_posix()}'\n",
    )
    manager = ExecutionManager()
    specs = reg.build_registry(allow_mutating=False, allow_execute=True, execution=manager)
    spec = next(s for s in specs if s.name == "train_start")
    with pytest.raises(reg.McpToolError) as exc:
        spec.handler({"config": "soup.yaml"})
    assert "data.train" in str(exc.value)
    assert "outside the working directory" in str(exc.value)
    assert not manager._plans
