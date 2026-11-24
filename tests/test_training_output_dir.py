"""``training_output_dir``: the one place that decides where a trainer writes.

``soup can run`` judges this directory before it extracts or runs a can, so a trainer that
joined the two config fields itself could write somewhere the check never looked. These
tests pin the helper and keep every trainer on it, with the same backend and task the
check passes for it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from soup_cli.utils.training_output import training_output_dir

_TRAINER_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer"
_HELPERS = frozenset({"training_output_dir", "_validated_output_dir"})
_CONFIG_NAMES = frozenset({"cfg", "config"})


class TestTheHelper:
    def test_the_experiment_name_is_appended_by_default(self):
        assert training_output_dir("out", "pilot") == str(Path("out") / "pilot")

    @pytest.mark.parametrize("experiment", [None, ""])
    def test_without_an_experiment_the_output_is_returned_as_given(self, experiment):
        assert training_output_dir("./out", experiment) == "./out"

    @pytest.mark.parametrize("backend", ["transformers", "unsloth", ""])
    def test_backends_that_nest_the_experiment_keep_doing_so(self, backend):
        assert training_output_dir("out", "pilot", backend=backend) == str(
            Path("out") / "pilot"
        )

    def test_mlx_writes_to_the_output_itself(self):
        assert training_output_dir("out", "pilot", backend="mlx") == "out"

    def test_unlearning_writes_to_the_output_itself(self):
        assert training_output_dir("out", "pilot", task="unlearn") == "out"

    @pytest.mark.parametrize("task", ["sft", "dpo", "grpo", "tts", "asr", "preference"])
    def test_other_tasks_nest_the_experiment(self, task):
        assert training_output_dir("out", "pilot", task=task) == str(Path("out") / "pilot")


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")


def _is_config_output(node: ast.AST) -> bool:
    """``cfg.output``, ``config.output`` or ``self.config.output``."""
    if not (isinstance(node, ast.Attribute) and node.attr == "output"):
        return False
    owner = node.value
    if isinstance(owner, ast.Name):
        return owner.id in _CONFIG_NAMES
    return isinstance(owner, ast.Attribute) and owner.attr in _CONFIG_NAMES


def _output_reads() -> list[tuple[str, ast.Attribute, ast.AST, dict]]:
    reads = []
    for path in sorted(_TRAINER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parents(tree)
        reads.extend(
            (path.name, node, tree, parents) for node in ast.walk(tree) if _is_config_output(node)
        )
    return reads


def _enclosing_helper_call(node: ast.AST, parents: dict) -> ast.Call | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Call) and _call_name(current) in _HELPERS:
            return current
        current = parents.get(current)
    return None


class TestEveryTrainerUsesTheHelper:
    def test_the_scan_sees_the_trainers_that_write_the_output(self):
        names = {name for name, *_ in _output_reads()}

        assert {"sft.py", "dpo.py", "mlx_sft.py", "mlx_dpo.py", "mlx_grpo.py", "unlearn.py"} <= (
            names
        )

    def test_no_trainer_reads_the_config_output_outside_the_helper(self):
        loose = [
            f"{name}:{node.lineno}"
            for name, node, _tree, parents in _output_reads()
            if _enclosing_helper_call(node, parents) is None
        ]

        assert not loose, (
            "a trainer reads the config's output directly instead of through "
            f"training_output_dir: {loose}"
        )

    def test_each_trainer_passes_the_backend_and_task_the_can_check_passes(self):
        wrong = []
        for name, node, _tree, parents in _output_reads():
            call = _enclosing_helper_call(node, parents)
            if call is None or _call_name(call) != "training_output_dir":
                continue
            keywords = {kw.arg: kw.value for kw in call.keywords}
            if name.startswith("mlx_"):
                ok = (
                    isinstance(keywords.get("backend"), ast.Constant)
                    and keywords["backend"].value == "mlx"
                )
            elif name == "unlearn.py":
                ok = (
                    isinstance(keywords.get("task"), ast.Constant)
                    and keywords["task"].value == "unlearn"
                )
            else:
                ok = "backend" not in keywords and "task" not in keywords
            if not ok:
                wrong.append(f"{name}:{call.lineno}")

        assert not wrong, f"these calls do not match the can check's routing: {wrong}"

    def test_unlearning_resolves_the_directory_with_its_task(self):
        tree = ast.parse((_TRAINER_DIR / "unlearn.py").read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == "training_output_dir"
        ]

        assert calls, "unlearn.py no longer calls training_output_dir"
        for call in calls:
            task = {kw.arg: kw.value for kw in call.keywords}.get("task")
            assert isinstance(task, ast.Constant) and task.value == "unlearn", call.lineno
