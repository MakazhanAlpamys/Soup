"""The local selector reads code without importing Soup or running pytest."""

import json
import shlex
import subprocess

import pytest

from scripts import select_tests
from tests.conftest import strip_ansi

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


def test_uses_reverse_imports_including_helpers_and_literals(checkout):
    plan = select_tests.plan_tests(checkout, [CORE])
    assert plan.reason is None
    assert plan.paths == (
        "tests/feature/feature_test.py",
        "tests/test_direct.py",
        "tests/test_dynamic.py",
        "tests/test_patch_only.py",
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


def test_reaching_conftest_does_not_expand_to_the_full_suite(checkout, monkeypatch):
    path = "tests/conftest.py"
    (checkout / path).write_text("from soup_cli.utils import rank_score\n", encoding="utf-8")
    monkeypatch.setattr(select_tests, "_candidate_paths", lambda root: {*SOURCE, path})

    plan = select_tests.plan_tests(checkout, [CORE])

    assert plan.reason is None
    assert "tests/test_direct.py" in plan.paths
    assert "tests/test_unrelated.py" not in plan.paths


def test_unparseable_module_makes_dependency_graph_uncertain(checkout):
    (checkout / "src/soup_cli/utils/wrapper.py").write_text("no =\n", encoding="utf-8")
    plan = select_tests.plan_tests(checkout, [CORE])
    assert plan.paths == ("tests/",)
    assert "could not parse" in plan.reason


def test_empty_and_broad_selections_have_visible_full_suite_results(checkout):
    empty = select_tests.plan_tests(checkout, [])
    assert empty.paths == ("tests/",)
    assert "no changed files" in empty.reason
    too_many = select_tests.plan_tests(checkout, [CORE], max_files=2)
    assert too_many.paths == ("tests/",)
    assert "more than 2" in too_many.reason


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
    status = " ".join(strip_ansi(captured.err).split())
    assert "Full suite" in status
    assert "Git" in status
    assert _selection_record(captured.err) == {
        "mode": "full",
        "reason": "Git could not run rev-parse",
        "scope": None,
        "unknown_dynamic_imports": 0,
    }


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


def _add_sources(checkout, monkeypatch, extra):
    for path, code in extra.items():
        file = checkout / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(code, encoding="utf-8")
    monkeypatch.setattr(select_tests, "_candidate_paths", lambda root: SOURCE.keys() | extra.keys())


def _selection_record(stderr):
    lines = [line for line in stderr.splitlines() if line.startswith("SOUP_TEST_SELECTION ")]
    assert len(lines) == 1
    return json.loads(lines[0].split(" ", 1)[1])


def test_string_only_paths_use_longest_existing_module_prefix(checkout, monkeypatch):
    extra = {
        "src/soup_cli/utils/rank_score_copy.py": "",
        "tests/test_patch_neighbor.py": "mock.patch('soup_cli.utils.rank_score_copy.score')\n",
        "tests/test_mock_only.py": "mock.patch('soup_cli.utils.rank_score.Engine.run')\n",
        "tests/test_registry.py": "routes = {'feature': 'soup_cli.utils.rank_score:score'}\n",
    }
    _add_sources(checkout, monkeypatch, extra)

    plan = select_tests.plan_tests(checkout, [CORE])

    assert plan.reason is None
    assert "tests/test_patch_only.py" in plan.paths
    assert "tests/test_mock_only.py" in plan.paths
    assert "tests/test_registry.py" in plan.paths
    assert "tests/test_patch_neighbor.py" not in plan.paths


def test_unknown_dynamic_imports_are_counted_without_selecting_tests(checkout, monkeypatch):
    extra = {
        "tests/test_lazy_builtin.py": "mod = __import__(name)\n",
        "tests/test_lazy_importlib.py": "mod = importlib.import_module(name=target)\n",
        "src/soup_cli/commands/plugin_loader.py": "module = importlib.import_module(target)\n",
        "tests/test_registry_client.py": "from soup_cli.commands import plugin_loader\n",
        "tests/test_dynamic_prefix.py": "importlib.import_module(f'soup_cli.envs.{name}')\n",
    }
    before = select_tests.plan_tests(checkout, [CORE])
    _add_sources(checkout, monkeypatch, extra)

    selected = select_tests.plan_tests(checkout, [CORE])

    assert selected.reason is None
    assert selected.paths == before.paths
    assert selected.unknown_dynamic_imports == 4

    unmapped = select_tests.plan_tests(checkout, ["src/soup_cli/utils/uncovered.py"])
    assert unmapped.paths == ("tests/",)
    assert "no related tests" in unmapped.reason
    assert unmapped.unknown_dynamic_imports == 4


def test_relative_import_inside_a_function_reaches_its_test(checkout, monkeypatch):
    extra = {
        "src/soup_cli/utils/other_wrapper.py": (
            "def run():\n    from . import rank_score\n    return rank_score.score()\n"
        ),
        "tests/test_other_wrapper.py": "from soup_cli.utils.other_wrapper import run\n",
    }
    _add_sources(checkout, monkeypatch, extra)

    assert "tests/test_other_wrapper.py" in select_tests.plan_tests(checkout, [CORE]).paths


@pytest.mark.parametrize(
    "stem",
    [
        "cli",
        "data",
        "train",
        "model",
        "eval",
        "serve",
        "config",
        "utils",
        "base",
        "common",
        "helpers",
    ],
)
def test_generic_stem_does_not_supply_evidence(checkout, monkeypatch, stem):
    generic = f"src/soup_cli/utils/{stem}.py"
    extra = {generic: "", f"tests/test_unrelated_{stem}_behavior.py": ""}
    _add_sources(checkout, monkeypatch, extra)

    plan = select_tests.plan_tests(checkout, [generic])

    assert plan.paths == ("tests/",)
    assert "no related tests" in plan.reason


def test_main_focused_returns_zero_and_prints_machine_readable_status(
    checkout, monkeypatch, capsys
):
    monkeypatch.setattr(select_tests, "REPO_ROOT", checkout)
    monkeypatch.chdir(checkout)

    assert select_tests.main(["--shell", "posix", CORE]) == 0

    captured = capsys.readouterr()
    plan = select_tests.plan_tests(checkout, [CORE])
    assert shlex.split(captured.out) == ["python", "-m", "pytest", "--no-cov", *plan.paths]
    assert _selection_record(captured.err) == {
        "mode": "focused",
        "reason": None,
        "scope": "transitive",
        "unknown_dynamic_imports": 0,
    }


def test_main_usage_error_retains_exit_two_and_empty_stdout(capsys):
    with pytest.raises(SystemExit) as exc:
        select_tests.main(["--base", "HEAD", CORE])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


def test_name_matching_is_only_for_unmapped_imports(checkout, monkeypatch):
    extra = {
        "src/soup_cli/utils/name_only.py": "",
        "tests/test_name_only_contract.py": "",
    }
    _add_sources(checkout, monkeypatch, extra)

    assert select_tests.plan_tests(checkout, ["src/soup_cli/utils/name_only.py"]).paths == (
        "tests/test_name_only_contract.py",
    )
    assert "tests/test_rank_score_cli.py" not in select_tests.plan_tests(checkout, [CORE]).paths


def test_broad_closure_keeps_complete_direct_and_test_helper_cohort(checkout, monkeypatch):
    source = "src/soup_cli/utils/scope_example.py"
    extra = {
        source: "",
        "tests/test_scope_direct.py": "from soup_cli.utils import scope_example\n",
        "tests/_scope_helper.py": "from soup_cli.utils import scope_example\n",
        "tests/test_scope_helper_consumer.py": "from tests import _scope_helper\n",
        "src/soup_cli/commands/scope_dispatch.py": "from soup_cli.utils import scope_example\n",
        "tests/test_scope_indirect.py": "from soup_cli.commands import scope_dispatch\n",
    }
    _add_sources(checkout, monkeypatch, extra)
    full_graph = select_tests.plan_tests(checkout, [source], max_files=3)
    focused = select_tests.plan_tests(checkout, [source], max_files=2)
    fallback = select_tests.plan_tests(checkout, [source], max_files=1)
    union = select_tests.plan_tests(checkout, [source, "tests/test_unrelated.py"], max_files=2)

    assert full_graph.scope == "transitive"
    assert "tests/test_scope_indirect.py" in full_graph.paths
    assert focused.scope == "direct"
    assert focused.paths == ("tests/test_scope_direct.py", "tests/test_scope_helper_consumer.py")
    assert fallback.paths == union.paths == ("tests/",)
    assert fallback.scope is None and union.scope is None
    assert "more than 1 related test files" in fallback.reason
    assert "more than 2 related test files" in union.reason


def test_main_focused_output_explains_direct_scope_and_unknown_edges(checkout, monkeypatch, capsys):
    monkeypatch.setattr(select_tests, "REPO_ROOT", checkout)
    monkeypatch.chdir(checkout)
    plan = select_tests.TestPlan(
        ("tests/test_direct.py",), scope="direct", unknown_dynamic_imports=3
    )
    monkeypatch.setattr(select_tests, "plan_tests", lambda *args: plan)

    assert select_tests.main(["--shell", "posix", CORE]) == 0

    captured = capsys.readouterr()
    status = " ".join(strip_ansi(captured.err).split())
    assert "direct import and test-helper" in status
    assert "3 unresolved dynamic imports" in status
    assert _selection_record(captured.err) == {
        "mode": "focused",
        "reason": None,
        "scope": "direct",
        "unknown_dynamic_imports": 3,
    }
    assert captured.out.strip() == "python -m pytest --no-cov tests/test_direct.py"


def test_help_describes_stdout_stderr_and_exit_codes(capsys):
    with pytest.raises(SystemExit) as exc:
        select_tests.main(["--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for detail in ("SOUP_TEST_SELECTION", "stderr", "stdout", "Exit code 0", "exit code 2"):
        assert detail in help_text


@pytest.mark.integration
def test_real_checkout_selects_related_tests_for_four_source_files():
    root = select_tests.REPO_ROOT
    related = {
        "src/soup_cli/eval/gate.py": "tests/test_eval_gate.py",
        "src/soup_cli/utils/yaml_limits.py": "tests/test_yaml_limits.py",
        "src/soup_cli/commands/doctor.py": "tests/test_doctor.py",
        "src/soup_cli/trainer/sft.py": "tests/test_issue341_seed_and_fullft.py",
    }
    try:
        candidates = select_tests._candidate_paths(root)
    except select_tests.SelectionError:
        pytest.skip("needs a full Git checkout")
    if not (related.keys() | set(related.values())) <= candidates or not all(
        (root / path).is_file() for path in candidates if path.endswith(".py")
    ):
        pytest.skip("needs a full checkout of Soup source and tests")
    graph = select_tests._build_graph(root, candidates=candidates)

    for path, expected in related.items():
        plan = select_tests.plan_tests(root, [path], graph=graph)
        assert plan.reason is None, (path, plan.reason)
        assert plan.scope in {"transitive", "direct"}
        assert 1 <= len(plan.paths) <= 300, (path, len(plan.paths))
        assert expected in plan.paths, (path, expected)
