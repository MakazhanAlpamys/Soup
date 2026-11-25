"""The local selector reads code without importing Soup or running pytest."""

import shlex
import subprocess

import pytest

from scripts import select_tests

CORE = "src/soup_cli/utils/rank_score.py"
SOURCE = {
    CORE: "def score():\n    return 1\n",
    "src/soup_cli/utils/__init__.py": "",
    "src/soup_cli/utils/wrapper.py": "from . import rank_score\n",
    "src/soup_cli/commands/feature.py": "from soup_cli.utils import wrapper\n",
    "tests/_feature_helper.py": "from soup_cli.commands.feature import get_feature\n",
    "tests/test_direct.py": "import soup_cli.utils.rank_score as score\n",
    "tests/feature/feature_test.py": "from tests import _feature_helper\n",
    "tests/test_rank_score_cli.py": "",
    "tests/test_unrelated.py": "from math import log\n",
    "scripts/migrate.py": "VALUE = 3\n",
    "tests/test_migrate.py": "from scripts import migrate\n",
    "tests/test_dynamic.py": "importlib.import_module('soup_cli.utils.rank_score')\n",
    "tests/test_patch_only.py": "monkeypatch.setattr('soup_cli.utils.rank_score.score', fake)\n",
    "src/soup_cli/utils/uncovered.py": "",
}


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    for path, code in SOURCE.items():
        file = tmp_path / path
        file.parent.mkdir(exist_ok=True, parents=True)
        file.write_text(code, encoding="utf-8")
    monkeypatch.setattr(select_tests, "_candidate_paths", lambda root: set(SOURCE))
    return tmp_path


def test_uses_reverse_imports_including_helpers_and_name_conventions(checkout):
    plan = select_tests.plan_tests(checkout, [CORE])
    assert plan.reason is None
    assert plan.paths == (
        "tests/feature/feature_test.py",
        "tests/test_direct.py",
        "tests/test_dynamic.py",
        "tests/test_rank_score_cli.py",
    )


def test_unions_changed_tests_and_script_dependents_and_removes_duplicates(checkout):
    expected = ("tests/test_migrate.py", "tests/test_unrelated.py")
    changed = ["tests/test_unrelated.py", "scripts/migrate.py", "tests/test_unrelated.py"]
    assert select_tests.plan_tests(checkout, changed).paths == expected


def test_changes_to_test_helper_follow_imports(checkout):
    assert select_tests.plan_tests(checkout, ["tests/_feature_helper.py"]).paths == (
        "tests/feature/feature_test.py",
    )


def test_changed_file_unmatched_among_matched_files_is_full(checkout):
    plan = select_tests.plan_tests(checkout, [CORE, "src/soup_cli/utils/uncovered.py"])
    assert plan.paths == ("tests/",)
    assert "no related tests" in plan.reason
    assert "uncovered.py" in plan.reason


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("pyproject.toml", "shared"),
        ("src/soup_cli/config/schema.py", "shared"),
        ("src/soup_cli/cli.py", "shared"),
        ("tests/feature/conftest.py", "shared"),
        ("tests/fixtures/sample.json", "shared"),
        ("src/soup_cli/utils/__init__.py", "shared"),
        ("README.md", "no mapping"),
        ("tests/test_deleted.py", "missing"),
        ("../outside.py", "outside"),
    ],
)
def test_shared_or_uncertain_path_is_full(checkout, path, reason):
    plan = select_tests.plan_tests(checkout, [CORE, path])
    assert plan.paths == ("tests/",)
    assert reason in plan.reason


def test_a_conftest_consumer_of_changed_code_makes_scope_global(checkout, monkeypatch):
    path = "tests/conftest.py"
    (checkout / path).write_text("from soup_cli.utils import rank_score\n", encoding="utf-8")
    monkeypatch.setattr(select_tests, "_candidate_paths", lambda root: {*SOURCE, path})

    plan = select_tests.plan_tests(checkout, [CORE])

    assert plan.paths == ("tests/",)
    assert "conftest.py depends on" in plan.reason


def test_unparseable_module_makes_dependency_graph_uncertain(checkout):
    (checkout / "src/soup_cli/utils/wrapper.py").write_text("no =\n", encoding="utf-8")
    plan = select_tests.plan_tests(checkout, [CORE])
    assert plan.paths == ("tests/",)
    assert "could not parse" in plan.reason


def test_empty_and_broad_selections_have_visible_full_suite_results(checkout):
    empty = select_tests.plan_tests(checkout, [])
    assert empty.paths == ("tests/",)
    assert "no changed files" in empty.reason
    too_many = select_tests.plan_tests(checkout, [CORE], max_files=3)
    assert too_many.paths == ("tests/",)
    assert "more than 3" in too_many.reason


def test_staged_unstaged_and_untracked_discovery_includes_both_rename_paths(monkeypatch, tmp_path):
    sha = "a" * 40
    responses = [
        f"{sha}\n".encode(),
        b"src/soup_cli/utils/old.py\0src/soup_cli/utils/new.py\0tests/test_staged.py\0",
        b"tests/test_new.py\0",
    ]
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(select_tests.subprocess, "check_output", run)
    assert select_tests.changed_paths(tmp_path, "origin/main") == {
        "src/soup_cli/utils/old.py",
        "src/soup_cli/utils/new.py",
        "tests/test_staged.py",
        "tests/test_new.py",
    }
    assert commands[0][0] == [
        "git",
        "rev-parse",
        "--verify",
        "--end-of-options",
        "origin/main^{commit}",
    ]
    assert "--no-renames" in commands[1][0]
    assert commands[1][0][-2:] == [sha, "--"]
    assert commands[2][0] == ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    assert all(kwargs["cwd"] == tmp_path for _, kwargs in commands)


def test_git_failure_prints_a_full_command_on_stdout(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(select_tests.subprocess, "check_output", fail)
    assert select_tests.main(["--shell", "posix"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "python -m pytest tests/"
    assert "Full suite" in captured.err
    assert "Git" in captured.err


def test_shell_safe_focused_command_and_no_full_coverage_for_a_subset():
    path = "tests/test unusual'$file.py"
    focused = select_tests.TestPlan((path,))
    target = ["python", "-m", "pytest", "--no-cov", path]
    assert shlex.split(select_tests.pytest_command(focused, shell="posix")) == target
    assert select_tests.pytest_command(focused, shell="powershell") == (
        "python -m pytest --no-cov 'tests/test unusual''$file.py'"
    )
    full = select_tests.TestPlan(("tests/",), reason="uncertain")
    assert select_tests.pytest_command(full, shell="powershell") == "python -m pytest tests/"
