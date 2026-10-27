"""External programs run by an absolute path found on PATH (utils/tool_paths.py).

Covers the search itself (``find_tool``: absolute PATH entries only, in order,
PATHEXT on Windows, the execute bit elsewhere, never the current directory), the
child environments (a filtered one keeps PATH, PATHEXT, SYSTEMROOT and
NoDefaultCurrentDirectoryInExePath), the copy carried by the generated RunPod and
Lambda controllers, and the places Soup starts an external program: each one hands
``subprocess`` the absolute path of the copy in the PATH directory, not a same-named
file in the current directory or in a relative PATH entry.

The ``tools`` fixture reproduces the Windows default by removing
NoDefaultCurrentDirectoryInExePath from the test process (some shells export it).
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.utils import tool_paths
from soup_cli.utils.tool_paths import (
    ALWAYS_KEPT_ENV_NAMES,
    ToolNotFoundError,
    filtered_environment,
    find_tool,
    require_tool,
    tool_environment,
)
from tests.conftest import strip_ansi

_WINDOWS = os.name == "nt"
_PATHEXT = ".COM;.EXE;.BAT;.CMD"
windows_only = pytest.mark.skipif(not _WINDOWS, reason="PATHEXT names Windows program files")
posix_only = pytest.mark.skipif(_WINDOWS, reason="Windows has no execute permission bit")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


def _program(directory: Path, name: str, *, suffix: str | None = None) -> Path:
    """An executable file named as this OS names programs (``.EXE`` on Windows)."""
    directory.mkdir(parents=True, exist_ok=True)
    if suffix is None:
        suffix = ".EXE" if _WINDOWS else ""
    path = directory / (name + suffix)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _same(left: str, right: object) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """``install(name)`` puts one copy of a program in the current directory, one
    under the relative PATH entry ``rel`` and one in an absolute PATH directory,
    and returns the absolute path of that last copy: the only one to be run."""
    project = tmp_path / "project"
    installed = tmp_path / "installed"
    project.mkdir()
    installed.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "rel", str(installed)]))
    monkeypatch.setenv("PATHEXT", _PATHEXT)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

    def install(name: str) -> str:
        _program(project, name)
        _program(project / "rel", name)
        return os.path.abspath(_program(installed, name))

    install.project = project
    install.installed = installed
    return install


@pytest.fixture
def no_tools(tmp_path, monkeypatch):
    """A PATH with one empty absolute directory: no program is found."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("PATHEXT", _PATHEXT)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)


class _Recorder:
    """A stand-in for ``subprocess.run`` that records each argv and env."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    def __call__(self, argv, *args, **kwargs):
        self.calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)

    @property
    def argv0(self) -> list[str]:
        return [argv[0] for argv, _ in self.calls]


# ---------------------------------------------------------------------------
# find_tool: the search
# ---------------------------------------------------------------------------


class TestTheSearch:
    def test_the_copy_in_the_absolute_path_directory_is_the_one_found(self, tools):
        expected = tools("soup-probe-tool")

        assert _same(find_tool("soup-probe-tool"), expected)

    def test_a_program_only_in_the_current_or_a_relative_directory_is_not_found(
        self, tools
    ):
        _program(tools.project, "soup-probe-tool")
        _program(tools.project / "rel", "soup-probe-tool")

        assert find_tool("soup-probe-tool") is None

    def test_empty_dot_and_relative_entries_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for directory in (tmp_path, tmp_path / "bin", tmp_path / "rel" / "bin"):
            _program(directory, "soup-probe-tool")
        relative = ["", " ", ".", "./bin", "bin", os.path.join("rel", "bin"), '""']
        monkeypatch.setenv("PATH", os.pathsep.join(relative))
        monkeypatch.setenv("PATHEXT", _PATHEXT)
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

        assert find_tool("soup-probe-tool") is None

    def test_the_entries_are_searched_in_order(self, tmp_path, monkeypatch):
        first = _program(tmp_path / "first", "soup-probe-tool")
        _program(tmp_path / "second", "soup-probe-tool")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        )
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert _same(find_tool("soup-probe-tool"), first)

    def test_a_quoted_entry_with_spaces_is_searched(self, tmp_path, monkeypatch):
        program = _program(tmp_path / "a dir with spaces", "soup-probe-tool")
        monkeypatch.setenv("PATH", f' "{tmp_path / "a dir with spaces"}" ')
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert _same(find_tool("soup-probe-tool"), program)

    def test_the_result_is_absolute_and_normalised(self, tmp_path, monkeypatch):
        program = _program(tmp_path / "bin", "soup-probe-tool")
        monkeypatch.setenv("PATH", str(tmp_path / "bin" / ".." / "bin"))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        found = find_tool("soup-probe-tool")

        assert os.path.isabs(found)
        assert ".." not in Path(found).parts
        assert _same(found, program)

    def test_a_directory_named_like_the_program_is_skipped(self, tmp_path, monkeypatch):
        decoy = tmp_path / "first" / ("soup-probe-tool" + (".EXE" if _WINDOWS else ""))
        decoy.mkdir(parents=True)
        program = _program(tmp_path / "second", "soup-probe-tool")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        )
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert _same(find_tool("soup-probe-tool"), program)

    def test_an_explicit_search_path_replaces_the_path_variable(self, tmp_path, monkeypatch):
        program = _program(tmp_path / "elsewhere", "soup-probe-tool")
        monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert find_tool("soup-probe-tool") is None
        assert _same(find_tool("soup-probe-tool", path=str(tmp_path / "elsewhere")), program)
        assert find_tool("soup-probe-tool", path="") is None

    def test_an_unset_path_does_not_search_the_current_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _program(tmp_path, "soup-probe-tool")
        monkeypatch.delenv("PATH", raising=False)
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

        assert find_tool("soup-probe-tool") is None

    @pytest.mark.parametrize(
        "name", ["", ".", "..", "bin/soup-probe-tool", os.path.join("bin", "soup-probe-tool")]
    )
    def test_a_name_with_a_directory_part_is_refused(self, name):
        with pytest.raises(ValueError, match="bare program name"):
            find_tool(name)

    @posix_only
    def test_a_file_without_the_execute_bit_is_skipped(self, tmp_path, monkeypatch):
        plain = tmp_path / "first" / "soup-probe-tool"
        plain.parent.mkdir()
        plain.write_text("data\n", encoding="utf-8")
        plain.chmod(0o644)
        program = _program(tmp_path / "second", "soup-probe-tool")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        )

        assert _same(find_tool("soup-probe-tool"), program)

    @windows_only
    def test_each_pathext_extension_is_tried_in_its_order(self, tmp_path, monkeypatch):
        # The reverse of the default order: the order comes from PATHEXT itself.
        directory = tmp_path / "bin"
        exe = _program(directory, "soup-probe-tool", suffix=".EXE")
        monkeypatch.setenv("PATH", str(directory))
        monkeypatch.setenv("PATHEXT", ".CMD;.EXE")

        assert _same(find_tool("soup-probe-tool"), exe)

        cmd = _program(directory, "soup-probe-tool", suffix=".CMD")
        assert _same(find_tool("soup-probe-tool"), cmd)

    @windows_only
    def test_an_extension_left_out_of_pathext_is_not_tried(self, tmp_path, monkeypatch):
        _program(tmp_path / "bin", "soup-probe-tool", suffix=".BAT")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", ".EXE")

        assert find_tool("soup-probe-tool") is None

    @windows_only
    def test_an_extension_outside_the_default_list_is_tried_when_pathext_names_it(
        self, tmp_path, monkeypatch
    ):
        program = _program(tmp_path / "bin", "soup-probe-tool", suffix=".SOUPX")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", ".EXE;.SOUPX")

        assert _same(find_tool("soup-probe-tool"), program)

    @windows_only
    def test_the_default_extensions_apply_when_pathext_is_unset(self, tmp_path, monkeypatch):
        bat = _program(tmp_path / "bin", "soup-probe-tool", suffix=".BAT")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.delenv("PATHEXT", raising=False)

        assert _same(find_tool("soup-probe-tool"), bat)

    @windows_only
    @pytest.mark.parametrize("pathext", ["", ";", " ; "])
    def test_a_pathext_that_names_no_extension_falls_back_to_the_default_extensions(
        self, tmp_path, monkeypatch, pathext
    ):
        bat = _program(tmp_path / "bin", "soup-probe-tool", suffix=".BAT")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", pathext)

        assert _same(find_tool("soup-probe-tool"), bat)

    @windows_only
    def test_a_name_that_ends_in_an_extension_is_tried_as_it_is(self, tmp_path, monkeypatch):
        cmd = _program(tmp_path / "bin", "soup-probe-tool", suffix=".cmd")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", ".EXE;.CMD")

        assert _same(find_tool("soup-probe-tool.cmd"), cmd)
        assert find_tool("soup-probe-tool.bat") is None

    @windows_only
    def test_a_file_without_an_extension_is_not_a_program(self, tmp_path, monkeypatch):
        _program(tmp_path / "bin", "soup-probe-tool", suffix="")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert find_tool("soup-probe-tool") is None

    @windows_only
    def test_rooted_and_drive_relative_entries_are_skipped(self, tmp_path, monkeypatch):
        _program(tmp_path / "bin", "soup-probe-tool")
        drive, rooted = os.path.splitdrive(str(tmp_path / "bin"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", os.pathsep.join([rooted, drive + "bin"]))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert find_tool("soup-probe-tool") is None


class TestRequireTool:
    def test_the_absolute_path_is_returned(self, tools):
        expected = tools("soup-probe-tool")

        assert _same(require_tool("soup-probe-tool", "Install it."), expected)

    def test_a_missing_program_is_named_with_how_to_install_it(self, no_tools):
        with pytest.raises(ToolNotFoundError) as excinfo:
            require_tool("soup-probe-tool", "Install it from https://example.invalid.")

        assert isinstance(excinfo.value, FileNotFoundError)
        message = str(excinfo.value)
        assert "soup-probe-tool was not found on PATH" in message
        assert "Install it from https://example.invalid." in message


# ---------------------------------------------------------------------------
# Child environments
# ---------------------------------------------------------------------------

_FOUR = {"PATH", "PATHEXT", "SYSTEMROOT", "NoDefaultCurrentDirectoryInExePath"}


def _folded(env) -> set[str]:
    return {name.upper() for name in env}


class TestFilteredEnvironment:
    def test_the_four_program_search_names_are_always_kept(self):
        assert set(ALWAYS_KEPT_ENV_NAMES) == _FOUR

    def test_exact_matching_keeps_the_named_and_the_four(self):
        environ = {
            "PATH": "/usr/bin",
            "PATHEXT": ".EXE",
            "SYSTEMROOT": "C:\\Windows",
            "NoDefaultCurrentDirectoryInExePath": "1",
            "LANG": "C.UTF-8",
            "HF_TOKEN": "hf-secret",
            "lang": "lower-case is another variable here",
        }

        env = filtered_environment({"LANG"}, environ, case_insensitive=False)

        assert env == {
            "PATH": "/usr/bin",
            "PATHEXT": ".EXE",
            "SYSTEMROOT": "C:\\Windows",
            "NoDefaultCurrentDirectoryInExePath": "1",
            "LANG": "C.UTF-8",
        }

    def test_windows_matching_ignores_case(self):
        environ = {
            "Path": "C:\\bin",
            "PathExt": ".EXE",
            "SystemRoot": "C:\\Windows",
            "NODEFAULTCURRENTDIRECTORYINEXEPATH": "1",
            "OPENAI_API_KEY": "sk-secret",
        }

        env = filtered_environment((), environ, case_insensitive=True)

        assert _folded(env) == {name.upper() for name in _FOUR}
        assert "OPENAI_API_KEY" not in env

    def test_the_default_follows_the_platform(self, monkeypatch):
        monkeypatch.setenv("SOUP_PROBE_VARIABLE", "x")
        monkeypatch.setenv("PATH", os.environ["PATH"])

        env = filtered_environment(())

        assert "PATH" in _folded(env)
        assert "SOUP_PROBE_VARIABLE" not in _folded(env)

    def test_windows_children_skip_the_current_directory_too(self):
        environ = {"PATH": "C:\\bin", "LANG": "C"}

        env = filtered_environment((), environ)

        if _WINDOWS:
            assert env == {"PATH": "C:\\bin", "NoDefaultCurrentDirectoryInExePath": "1"}
        else:
            assert env == {"PATH": "C:\\bin"}

    @windows_only
    def test_another_spelling_of_the_setting_is_replaced(self):
        environ = {"Path": "C:\\bin", "nodefaultcurrentdirectoryinexepath": "0"}

        env = filtered_environment((), environ)

        assert env == {"Path": "C:\\bin", "NoDefaultCurrentDirectoryInExePath": "1"}


class TestToolEnvironment:
    def test_the_named_variables_are_left_out_ignoring_case(self):
        environ = {"RUNPOD_API_KEY": "k", "runpod_api_key": "k2", "PATH": "/bin", "HOME": "/h"}

        env = tool_environment(drop=("RunPod_Api_Key",), environ=environ)

        assert "RUNPOD_API_KEY" not in _folded(env)
        assert env["PATH"] == "/bin"
        assert env["HOME"] == "/h"

    def test_windows_programs_skip_the_current_directory_too(self, monkeypatch):
        monkeypatch.setenv("NoDefaultCurrentDirectoryInExePath", "0")

        env = tool_environment()

        names = [name for name in env if name.upper() == "NODEFAULTCURRENTDIRECTORYINEXEPATH"]
        if _WINDOWS:
            assert names == ["NoDefaultCurrentDirectoryInExePath"]
            assert env["NoDefaultCurrentDirectoryInExePath"] == "1"
        else:
            assert names == ["NoDefaultCurrentDirectoryInExePath"]
            assert env["NoDefaultCurrentDirectoryInExePath"] == "0"

    def test_the_result_is_a_new_mapping(self):
        assert tool_environment() is not os.environ


_NO_CWD = "NoDefaultCurrentDirectoryInExePath"

_START_BY_BARE_NAME = """
import subprocess, sys
if sys.argv[1] == "exclude":
    from soup_cli.utils.tool_paths import exclude_current_directory_from_program_search
    exclude_current_directory_from_program_search()
try:
    subprocess.run(["soupcwdprobe"], capture_output=True, check=False, timeout=60)
except FileNotFoundError:
    print("not found")
else:
    print("started")
"""


def _unset_for_this_test(monkeypatch, name: str) -> None:
    """Remove ``name`` from the environment; teardown removes it again whether
    or not it was set before the test."""
    monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv(name)


def _values_seen_by_the_command(monkeypatch) -> list[str | None]:
    """Run ``soup version`` through ``cli.run()`` with the command replaced, and
    return the value of the variable each time the command was called."""
    from soup_cli import cli

    seen: list[str | None] = []

    def app():
        seen.append(os.environ.get(_NO_CWD))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "app", app)
    monkeypatch.setattr(cli, "_emit_audit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_emit_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["soup", "version"])

    with pytest.raises(SystemExit):
        cli.run()

    return seen


windows_setting_only = pytest.mark.skipif(
    not _WINDOWS,
    reason="NoDefaultCurrentDirectoryInExePath is read only by Windows; elsewhere soup leaves it "
    "alone (test_soup_sets_the_variable_for_itself_on_windows checks that)",
)


class TestSoupLeavesTheCurrentDirectoryOutOfItsProgramSearch:
    def test_soup_sets_the_variable_for_itself_on_windows(self, monkeypatch):
        _unset_for_this_test(monkeypatch, _NO_CWD)

        seen = _values_seen_by_the_command(monkeypatch)

        assert seen == (["1"] if _WINDOWS else [None])

    @windows_setting_only
    @pytest.mark.parametrize("spelling", [_NO_CWD, _NO_CWD.lower()])
    def test_a_value_already_set_is_kept(self, monkeypatch, spelling):
        monkeypatch.setenv(spelling, "0")

        seen = _values_seen_by_the_command(monkeypatch)

        assert seen == ["0"]
        assert os.environ[_NO_CWD] == "0"
        assert len([name for name in os.environ if name.upper() == _NO_CWD.upper()]) == 1

    @windows_only
    def test_a_program_only_in_the_current_directory_is_not_started_by_bare_name(
        self, tmp_path
    ):
        import shutil

        system_program = Path(os.environ.get("SYSTEMROOT", "C:\\Windows"), "System32", "whoami.exe")
        if not system_program.is_file():
            pytest.skip(f"{system_program} is not there to copy")
        shutil.copyfile(system_program, tmp_path / "soupcwdprobe.exe")
        env = {name: value for name, value in os.environ.items() if name.upper() != _NO_CWD.upper()}
        env["PYTHONPATH"] = str(Path(tool_paths.__file__).resolve().parents[2])

        def outcome(mode: str) -> str:
            completed = subprocess.run(
                [sys.executable, "-P", "-c", _START_BY_BARE_NAME, mode],
                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
            )
            assert completed.returncode == 0, completed.stderr
            return completed.stdout.strip()

        # Without the setting Windows starts the copy in the current directory ...
        assert outcome("keep") == "started"
        # ... and with it the bare name is not found there.
        assert outcome("exclude") == "not found"


def _with_the_four(monkeypatch) -> None:
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("PATHEXT", _PATHEXT)
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("NoDefaultCurrentDirectoryInExePath", "1")


# ---------------------------------------------------------------------------
# The generated cloud controllers carry the same search
# ---------------------------------------------------------------------------

_CONFIG = "base: gpt2\ndata:\n  train: data.jsonl\noutput: ./out\n"


def _lambda_namespace() -> dict:
    from soup_cli.cloud.lambda_labs import render_lambda_stub

    stub = render_lambda_stub(_CONFIG, gpu="a100", output_dir="./out", soup_version="0.76.0")
    namespace: dict = {"__name__": "controller"}
    exec(compile(stub, "soup_lambda_app.py", "exec"), namespace)
    return namespace


def _runpod_stub() -> str:
    from soup_cli.cloud.runpod import render_runpod_stub

    return render_runpod_stub(
        "base: public/model\ndata: {train: public/dataset}\noutput: out\n",
        gpu="a100",
        output_dir="out",
        soup_version="0.76.0",
    )


class TestTheGeneratedControllersCarryTheSearch:
    @pytest.mark.parametrize("render", ["runpod", "lambda"])
    def test_the_controller_holds_the_shared_source_and_imports_no_soup_cli(self, render):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        stub = (
            _runpod_stub()
            if render == "runpod"
            else render_lambda_stub(
                _CONFIG, gpu="a100", output_dir="./out", soup_version="0.76.0"
            )
        )

        assert tool_paths.controller_source() in stub
        assert not [
            line for line in stub.splitlines()
            if line.lstrip().startswith(("import soup_cli", "from soup_cli"))
        ]

    def test_the_runpod_template_defines_no_search_of_its_own(self):
        template = (
            Path(tool_paths.__file__).resolve().parents[1] / "cloud" / "_runpod_controller.py"
        ).read_text(encoding="utf-8")

        assert "def _find_tool" not in template
        assert "def find_tool" not in template

    @pytest.mark.parametrize("render", ["runpod", "lambda"])
    def test_the_controller_finds_the_absolute_copy_only(self, tools, render):
        expected = tools("ssh")
        if render == "runpod":
            namespace: dict = {"__name__": "controller"}
            exec(compile(_runpod_stub(), "soup_runpod_app.py", "exec"), namespace)
        else:
            namespace = _lambda_namespace()

        assert _same(namespace["_find_tool"]("ssh"), expected)
        _program(tools.project, "scp")
        assert namespace["_find_tool"]("scp") is None

    @pytest.mark.parametrize("case_insensitive", [True, False])
    @pytest.mark.parametrize("cloud", ["lambda_labs", "runpod"])
    def test_the_controller_environment_keeps_the_program_search_variables(
        self, cloud, case_insensitive
    ):
        import importlib

        module = importlib.import_module(f"soup_cli.cloud.{cloud}")
        environ = {name: "value" for name in _FOUR}

        env = module._controller_environment(environ, case_insensitive=case_insensitive)

        assert set(env) == _FOUR


def _lambda_run(namespace: dict, run) -> tuple[int, list[str]]:
    printed: list[str] = []
    namespace["_request_json"] = lambda api_key, path, **kw: (
        {"data": {"instance_ids": ["instance-1"]}} if path.endswith("/launch") else {"data": {}}
    )
    namespace["_wait_for_ip"] = lambda *_: "192.0.2.1"
    namespace["_wait_for_termination"] = lambda *_: None
    namespace["time"] = types.SimpleNamespace(sleep=lambda _s: None, monotonic=lambda: 0.0)
    namespace["subprocess"] = types.SimpleNamespace(
        run=run, TimeoutExpired=subprocess.TimeoutExpired
    )
    namespace["print"] = lambda *args, **kw: printed.append(" ".join(map(str, args)))
    return namespace["main"](), printed


@pytest.fixture
def lambda_credentials(tmp_path, monkeypatch):
    key = tmp_path / "lambda-key"
    key.write_text("test-only", encoding="utf-8")
    monkeypatch.setenv("LAMBDA_API_KEY", "lambda-secret")
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "my-key")
    monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
    monkeypatch.setenv("LAMBDA_REGION", "us-tx-1")


class TestTheLambdaControllerRunsSshAndScpByAbsolutePath:
    def test_ssh_and_scp_run_by_absolute_path_without_the_api_key(
        self, tools, lambda_credentials
    ):
        ssh, scp = tools("ssh"), tools("scp")
        namespace = _lambda_namespace()
        run = _Recorder(stdout="0\n")

        rc, printed = _lambda_run(namespace, run)

        assert rc == 0, printed
        assert {os.path.normcase(argv0) for argv0 in run.argv0} == {
            os.path.normcase(ssh), os.path.normcase(scp)
        }
        for _argv, kwargs in run.calls:
            assert "LAMBDA_API_KEY" not in _folded(kwargs["env"])
            if _WINDOWS:
                assert kwargs["env"]["NoDefaultCurrentDirectoryInExePath"] == "1"

    def test_the_log_tail_runs_ssh_by_absolute_path(self, tools, lambda_credentials):
        ssh = tools("ssh")
        namespace = _lambda_namespace()
        run = _Recorder(stdout="line\n")

        namespace["subprocess"] = types.SimpleNamespace(
            run=run, TimeoutExpired=subprocess.TimeoutExpired
        )
        namespace["print"] = lambda *args, **kw: None
        namespace["_fetch_log_tail"]("192.0.2.1", "key")

        assert [os.path.normcase(argv0) for argv0 in run.argv0] == [os.path.normcase(ssh)]
        assert "LAMBDA_API_KEY" not in _folded(run.calls[0][1]["env"])

    def test_missing_ssh_refuses_before_any_instance_is_launched(
        self, no_tools, lambda_credentials
    ):
        namespace = _lambda_namespace()
        run = _Recorder()

        rc, printed = _lambda_run(namespace, run)

        assert rc == 2
        assert "OpenSSH ssh and scp must be installed and on PATH" in " ".join(printed)
        assert not run.calls
        assert not any("launched" in line for line in printed)


# ---------------------------------------------------------------------------
# Every place Soup starts an external program
# ---------------------------------------------------------------------------


class TestNvidiaSmi:
    def test_doctor_runs_nvidia_smi_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.commands import doctor

        smi = tools("nvidia-smi")
        run = _Recorder(returncode=1)
        monkeypatch.setattr(subprocess, "run", run)

        doctor._nvidia_smi_cuda_version()
        doctor._detect_gpu_hw_without_torch_cuda()

        assert len(run.calls) == 2
        assert all(_same(argv0, smi) for argv0 in run.argv0)

    def test_doctor_reports_no_nvidia_smi_when_none_is_on_path(self, no_tools, monkeypatch):
        from soup_cli.commands import doctor

        monkeypatch.setattr(subprocess, "run", _Recorder())

        assert doctor._nvidia_smi_executable() is None
        assert doctor._nvidia_smi_cuda_version() is None
        assert doctor._detect_gpu_hw_without_torch_cuda() == ""

    def test_the_gpu_monitor_runs_nvidia_smi_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.utils import gpu_monitor

        smi = tools("nvidia-smi")
        run = _Recorder(stdout="")
        monkeypatch.setattr(gpu_monitor.subprocess, "run", run)

        gpu_monitor.query_nvidia_smi()

        assert len(run.calls) == 1
        assert _same(run.argv0[0], smi)

    def test_the_sm_clock_probe_runs_nvidia_smi_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import sm_clock_mhz

        smi = tools("nvidia-smi")
        run = _Recorder(stdout="1500\n")
        monkeypatch.setattr(subprocess, "run", run)

        assert sm_clock_mhz() == 1500
        assert _same(run.argv0[0], smi)

    def test_bench_train_runs_nvidia_smi_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.bench import train_run

        smi = tools("nvidia-smi")
        run = _Recorder(stdout="595.12\n")
        monkeypatch.setattr(subprocess, "run", run)

        assert train_run._driver_version() == "595.12"
        sampler = train_run.ClockSampler()
        assert sampler.available
        sampler._read()

        assert len(run.calls) == 2
        assert all(_same(argv0, smi) for argv0 in run.argv0)


class TestSystemToolsWithAFixedLocation:
    def test_a_fixed_location_wins_over_a_copy_on_path(self, tools, tmp_path):
        from soup_cli.utils.layer_stream import _resolve_tool

        tools("soup-probe-tool")
        fixed = _program(tmp_path / "system", "soup-probe-tool")

        assert _same(_resolve_tool("soup-probe-tool", str(fixed)), fixed)

    def test_path_is_searched_when_the_fixed_location_is_missing(self, tools, tmp_path):
        from soup_cli.utils.layer_stream import _resolve_tool

        expected = tools("soup-probe-tool")

        assert _same(_resolve_tool("soup-probe-tool", str(tmp_path / "missing")), expected)
        assert _same(_resolve_tool("soup-probe-tool"), expected)

    def test_windows_powershell_runs_from_its_fixed_location(self, tools, tmp_path, monkeypatch):
        import platform

        from soup_cli.utils import layer_stream

        tools("powershell")
        fixed = _program(tmp_path / "System32", "powershell")
        monkeypatch.setattr(layer_stream, "_WINDOWS_POWERSHELL", str(fixed))
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        run = _Recorder(stdout="")
        monkeypatch.setattr(subprocess, "run", run)

        layer_stream._probe_disk_kind(str(tmp_path))

        assert _same(run.argv0[0], fixed)

    def test_macos_diskutil_found_on_path_runs_by_absolute_path(self, tools, tmp_path, monkeypatch):
        import platform

        from soup_cli.utils import layer_stream

        diskutil = tools("diskutil")
        monkeypatch.setattr(layer_stream, "_MACOS_DISKUTIL", str(tmp_path / "no-diskutil"))
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        run = _Recorder(returncode=1)
        monkeypatch.setattr(subprocess, "run", run)

        layer_stream._probe_disk_kind(str(tmp_path))

        assert run.calls
        assert _same(run.argv0[0], diskutil)

    @pytest.mark.parametrize("probe", ["chip", "memory"])
    def test_mlx_reads_sysctl_from_its_fixed_location(self, tools, monkeypatch, probe):
        from soup_cli.utils import mlx

        tools("sysctl")
        monkeypatch.setattr(mlx, "is_apple_silicon", lambda: True)
        run = _Recorder(stdout="17179869184\n")
        monkeypatch.setattr(subprocess, "run", run)

        if probe == "chip":
            mlx.get_chip_info()
        else:
            assert mlx.get_unified_memory_bytes() == 17179869184

        assert run.argv0 == ["/usr/sbin/sysctl"]

    def test_doctor_reads_macos_memory_through_the_fixed_sysctl(self, tools, monkeypatch):
        import platform

        from soup_cli.commands import doctor

        tools("sysctl")
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        run = _Recorder(stdout=str(16 * 1024**3) + "\n")
        monkeypatch.setattr(subprocess, "run", run)

        assert doctor._get_ram_gb() == "16 GB"
        assert run.argv0 == ["/usr/sbin/sysctl"]


class TestOllama:
    @pytest.mark.parametrize("call", ["detect", "deploy", "list", "remove"])
    def test_every_ollama_call_runs_it_by_absolute_path(self, tools, monkeypatch, call):
        from soup_cli.utils import ollama

        expected = tools("ollama")
        run = _Recorder(stdout="ollama version is 0.6.2\n")
        monkeypatch.setattr(ollama.subprocess, "run", run)

        {
            "detect": lambda: ollama.detect_ollama(),
            "deploy": lambda: ollama.deploy_to_ollama("soup-test", "FROM x.gguf\n"),
            "list": lambda: ollama.list_soup_models(),
            "remove": lambda: ollama.remove_model("soup-test"),
        }[call]()

        assert len(run.calls) == 1
        assert _same(run.argv0[0], expected)

    def test_a_missing_ollama_keeps_its_messages(self, no_tools, monkeypatch):
        from soup_cli.utils import ollama

        run = _Recorder()
        monkeypatch.setattr(ollama.subprocess, "run", run)

        assert ollama.detect_ollama() is None
        assert ollama.deploy_to_ollama("soup-test", "FROM x.gguf\n") == (
            False, "Ollama not found. Install from https://ollama.com"
        )
        assert ollama.list_soup_models() == []
        assert ollama.remove_model("soup-test") == (
            False, "Ollama not found. Install from https://ollama.com"
        )
        assert not run.calls


class TestLlamaCpp:
    def test_the_llama_proxy_resolves_the_absolute_copy(self, tools):
        from soup_cli.utils.llama_proxy import build_argv, resolve

        expected = tools("llama-cli")

        argv = build_argv(resolve("cli", ["--help"]))

        assert _same(argv[0], os.path.realpath(expected))

    def test_the_llama_proxy_refuses_a_binary_that_is_only_in_the_current_directory(
        self, tools
    ):
        from soup_cli.utils.llama_proxy import resolve

        _program(tools.project, "llama-server")

        with pytest.raises(FileNotFoundError, match="llama-server not found on PATH"):
            resolve("server", [])

    def test_export_finds_llama_quantize_on_path_by_absolute_path(self, tools, tmp_path):
        from soup_cli.commands.export import _find_quantize_binary

        expected = tools("llama-quantize")
        checkout = tmp_path / "llama.cpp"
        checkout.mkdir()

        assert _same(_find_quantize_binary(checkout), expected)

    def test_export_returns_the_quantize_binary_of_a_dot_checkout_by_absolute_path(
        self, no_tools, tmp_path
    ):
        from soup_cli.commands.export import _find_quantize_binary

        expected = _program(tmp_path, "llama-quantize", suffix=".exe" if _WINDOWS else "")

        found = _find_quantize_binary(Path("."))

        assert os.path.isabs(found)
        assert _same(found, expected)

    @pytest.mark.parametrize("stage", ["quantize", "imatrix"])
    def test_the_advanced_gguf_tools_of_a_dot_checkout_run_by_absolute_path(
        self, no_tools, tmp_path, monkeypatch, stage
    ):
        from soup_cli.utils import gguf_quant

        name = "llama-quantize" if stage == "quantize" else "llama-imatrix"
        expected = _program(tmp_path, name, suffix=".exe" if _WINDOWS else "")
        run = _Recorder()
        monkeypatch.setattr(gguf_quant.subprocess, "run", run)

        if stage == "quantize":
            gguf_quant._run_quantize_binary(
                llama_cpp_dir=".", f16_path="f16.gguf", output_path="out.gguf", flavour="Q4_K_M"
            )
        else:
            gguf_quant._run_imatrix(
                llama_cpp_dir=".", f16_path="f16.gguf", calib_data="c.txt", imatrix_out="i.dat"
            )

        assert os.path.isabs(run.argv0[0])
        assert _same(run.argv0[0], expected)

    def test_export_clones_llama_cpp_with_the_absolute_git(self, tools, tmp_path, monkeypatch):
        from soup_cli.commands import export

        git = tools("git")
        monkeypatch.setattr(export, "_existing_llama_cpp", lambda user_path=None: None)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        run = _Recorder()
        monkeypatch.setattr(export.subprocess, "run", run)

        export._find_llama_cpp()

        assert _same(run.argv0[0], git)

    def test_export_without_git_keeps_its_message(self, no_tools, tmp_path, monkeypatch, capsys):
        import typer
        from rich.console import Console

        from soup_cli.commands import export

        monkeypatch.setattr(export, "_existing_llama_cpp", lambda user_path=None: None)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        monkeypatch.setattr(export, "console", Console(force_terminal=False, width=200))
        run = _Recorder()
        monkeypatch.setattr(export.subprocess, "run", run)

        with pytest.raises(typer.Exit):
            export._find_llama_cpp()

        assert "git not found." in _plain(capsys.readouterr().out)
        assert not run.calls

    def test_the_child_environment_keeps_the_program_search_variables(self, monkeypatch):
        from soup_cli.commands.llama import _filtered_env

        _with_the_four(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf-secret")

        env = _filtered_env()

        assert _folded(env) >= {name.upper() for name in _FOUR}
        assert "HF_TOKEN" not in _folded(env)


class TestTensorRT:
    def _export(self, tmp_path, monkeypatch, run):
        from rich.console import Console

        from soup_cli.commands import export

        for name in (
            "tensorrt_llm", "tensorrt_llm.commands", "tensorrt_llm.commands.convert_checkpoint"
        ):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(export, "console", Console(force_terminal=False, width=200))
        monkeypatch.setattr(export.subprocess, "run", run)
        return export._export_tensorrt(model, str(tmp_path / "out"), None)

    def test_trtllm_build_runs_by_absolute_path(self, tools, tmp_path, monkeypatch):
        expected = tools("trtllm-build")
        run = _Recorder()

        self._export(tmp_path, monkeypatch, run)

        assert run.argv0[0] == sys.executable
        assert _same(run.argv0[1], expected)

    def test_a_missing_trtllm_build_keeps_its_message(
        self, no_tools, tmp_path, monkeypatch, capsys
    ):
        import typer

        run = _Recorder()

        with pytest.raises(typer.Exit):
            self._export(tmp_path, monkeypatch, run)

        assert "trtllm-build not found in PATH." in _plain(capsys.readouterr().out)
        assert run.argv0 == [sys.executable]


class TestDocker:
    def test_the_aider_preflight_runs_docker_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.eval import aider_polyglot

        expected = tools("docker")
        run = _Recorder(stdout="27.0.0\n")
        monkeypatch.setattr(aider_polyglot.subprocess, "run", run)

        docker = aider_polyglot.preflight_docker("aider-benchmark")

        assert _same(docker, expected)
        assert len(run.calls) == 2
        assert all(_same(argv0, expected) for argv0 in run.argv0)

    def test_a_docker_only_in_the_current_directory_is_not_used(self, tools, monkeypatch):
        from soup_cli.eval import aider_polyglot

        _program(tools.project, "docker")
        run = _Recorder()
        monkeypatch.setattr(aider_polyglot.subprocess, "run", run)

        with pytest.raises(aider_polyglot.AiderEvalError, match="Docker CLI was not found"):
            aider_polyglot.preflight_docker("aider-benchmark")
        assert not run.calls


class TestModal:
    def _plan(self):
        from soup_cli.cloud.modal import CloudPlan

        return CloudPlan(
            cloud="modal", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="modal run x.py",
        )

    def test_modal_run_starts_the_absolute_cli(self, tools, monkeypatch):
        from soup_cli.cloud.modal import submit_modal_run

        expected = tools("modal")
        monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        rc = submit_modal_run(
            self._plan(), env={"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret"}
        )

        assert rc == 0
        assert _same(run.argv0[0], expected)
        assert run.calls[0][0][1:] == ["run", "x.py"]

    def test_a_missing_modal_cli_is_refused_by_name(self, no_tools, monkeypatch):
        from soup_cli.cloud.modal import submit_modal_run

        monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        with pytest.raises(RuntimeError, match="Modal CLI `modal` was not found on PATH"):
            submit_modal_run(
                self._plan(), env={"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret"}
            )
        assert not run.calls


class TestGitHubCli:
    def test_gh_runs_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.utils.adapter_pr import post_pr_comment

        expected = tools("gh")
        run = _Recorder(stdout='{"html_url": "u"}')

        post_pr_comment("o/r#1", "body", env={"GITHUB_TOKEN": "t"}, runner=run)

        assert _same(run.argv0[0], expected)

    def test_a_missing_gh_keeps_its_message(self, no_tools):
        from soup_cli.utils.adapter_pr import post_pr_comment

        run = _Recorder()

        with pytest.raises(RuntimeError, match="`gh` CLI not found"):
            post_pr_comment("o/r#1", "body", env={"GITHUB_TOKEN": "t"}, runner=run)
        assert not run.calls

    def test_the_child_environment_keeps_the_program_search_variables(self, tools, monkeypatch):
        from soup_cli.utils.adapter_pr import post_pr_comment

        tools("gh")
        _with_the_four(monkeypatch)
        monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "rel", str(tools.installed)]))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        run = _Recorder(stdout="{}")

        post_pr_comment("o/r#1", "body", runner=run)

        env = run.calls[0][1]["env"]
        assert _folded(env) >= {name.upper() for name in _FOUR}
        assert "OPENAI_API_KEY" not in _folded(env)


class TestTheCodeSandbox:
    def _reset(self, rewards, monkeypatch):
        monkeypatch.setattr(rewards, "_ISOLATION_STRATEGY_CACHE", None)

    def test_macos_sandbox_exec_is_probed_at_its_fixed_location(self, tmp_path, monkeypatch):
        from soup_cli.trainer import rewards

        self._reset(rewards, monkeypatch)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(rewards, "_MACOS_SANDBOX_EXEC", str(tmp_path / "missing"))
        assert rewards._compute_isolation_strategy() == "best-effort"

        fixed = _program(tmp_path / "usr-bin", "sandbox-exec", suffix="")
        monkeypatch.setattr(rewards, "_MACOS_SANDBOX_EXEC", str(fixed))
        assert rewards._compute_isolation_strategy() == "sandbox-exec"

    def test_macos_sandbox_exec_runs_from_its_fixed_location(self, tools, monkeypatch):
        from soup_cli.trainer import rewards

        tools("sandbox-exec")
        # Copies named as macOS names programs, too (``sys.platform`` says darwin).
        for directory in (tools.project, tools.project / "rel", tools.installed):
            _program(directory, "sandbox-exec", suffix="")
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "sandbox-exec")
        started: list[list[str]] = []

        def popen(argv, **kwargs):
            started.append(list(argv))
            raise OSError("not started in this test")

        monkeypatch.setattr(rewards.subprocess, "Popen", popen)

        result = rewards._run_sandboxed_subprocess([sys.executable, "-c", "pass"])

        assert result.launch_failed
        assert started[0][0] == rewards._MACOS_SANDBOX_EXEC == "/usr/bin/sandbox-exec"

    @posix_only
    def test_the_bash_tool_runs_bash_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.trainer import rewards

        bash = tools("bash")
        monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "namespaces")
        monkeypatch.setattr(rewards, "_show_code_exec_warning_once", lambda: None)
        started: list[list[str]] = []

        def popen(argv, **kwargs):
            started.append(list(argv))
            raise OSError("not started in this test")

        monkeypatch.setattr(rewards.subprocess, "Popen", popen)

        with pytest.raises(PermissionError):
            rewards._run_bash_sandbox("echo hi")

        assert _same(started[0][0], bash)

    def test_the_child_environment_keeps_the_program_search_variables(self, monkeypatch):
        from soup_cli.trainer.rewards import _get_safe_sandbox_env

        _with_the_four(monkeypatch)

        assert _folded(_get_safe_sandbox_env()) >= {name.upper() for name in _FOUR}


class TestTheMultiGpuLauncher:
    def test_windows_starts_accelerate_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.utils import launcher

        expected = tools("accelerate")
        monkeypatch.setattr(launcher, "_IS_WINDOWS", True)
        started: list[list[str]] = []

        class _Process:
            def __init__(self, argv, *args, **kwargs):
                started.append(list(argv))

            def wait(self):
                return 0

        monkeypatch.setattr(subprocess, "Popen", _Process)

        assert launcher.run_launcher(["accelerate", "launch", "--num_processes", "2"]) == 0
        assert _same(started[0][0], expected)
        assert started[0][1:] == ["launch", "--num_processes", "2"]

    def test_posix_execs_accelerate_by_absolute_path(self, tools, monkeypatch):
        from soup_cli.utils import launcher

        expected = tools("accelerate")
        monkeypatch.setattr(launcher, "_IS_WINDOWS", False)
        calls: list[tuple[str, list[str]]] = []

        def execv(path, argv):
            calls.append((path, list(argv)))
            raise SystemExit(0)  # a real exec does not return

        monkeypatch.setattr(os, "execv", execv)
        monkeypatch.setattr(os, "execvp", lambda *a: pytest.fail("PATH search by exec"))

        with pytest.raises(SystemExit):
            launcher.run_launcher(["accelerate", "launch"])

        assert _same(calls[0][0], expected)
        assert calls[0][1] == ["accelerate", "launch"]

    @pytest.mark.parametrize("windows", [True, False])
    def test_a_launcher_only_in_the_current_directory_is_not_started(
        self, tools, monkeypatch, windows
    ):
        from soup_cli.utils import launcher

        _program(tools.project, "accelerate")
        monkeypatch.setattr(launcher, "_IS_WINDOWS", windows)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
        monkeypatch.setattr(os, "execv", lambda *a: pytest.fail("exec'd"))
        monkeypatch.setattr(os, "execvp", lambda *a: pytest.fail("exec'd"))

        with pytest.raises(FileNotFoundError, match="launcher not found on PATH"):
            launcher.run_launcher(["accelerate", "launch"])


class TestAdapterBisectEvalCommand:
    def _invoke(self, eval_command: str, history: tuple[str, ...] = ("ckpt-1", "ckpt-2")):
        from soup_cli.cli import app

        return CliRunner().invoke(
            app, ["adapters", "bisect", *history, "--eval-command", eval_command]
        )

    def test_a_bare_program_name_runs_by_absolute_path(self, tools, monkeypatch):
        expected = tools("soup-eval-probe")
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        result = self._invoke("soup-eval-probe --model {ckpt}")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert run.calls
        assert all(_same(argv0, expected) for argv0 in run.argv0)
        assert run.calls[0][0][1] == "--model"

    def test_a_program_given_with_its_path_runs_as_given(self, tools, monkeypatch):
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        result = self._invoke(f'"{Path(sys.executable).as_posix()}" -c pass {{ckpt}}')

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert run.calls
        assert all(_same(argv0, Path(sys.executable).as_posix()) for argv0 in run.argv0)

    def test_a_program_that_is_not_on_path_is_refused_by_name(self, tools, monkeypatch):
        _program(tools.project, "soup-eval-probe")
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        result = self._invoke("soup-eval-probe --model {ckpt}")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert output.startswith(
            "--eval-command program 'soup-eval-probe' was not found on PATH"
        ), output
        # Refused before the bisect starts: no probe ran and no verdict was given.
        assert "Eval failed" not in output
        assert "Adapter bisect" not in output
        assert not run.calls

    @pytest.mark.parametrize(
        ("template", "program"), [("{ckpt}", "{}"), ("{ckpt}x --model m", "{}x")]
    )
    def test_a_program_named_by_the_checkpoint_id_runs_by_absolute_path(
        self, tools, monkeypatch, template, program
    ):
        history = ("soup-eval-a", "soup-eval-b")
        expected = [tools(program.format(ckpt)) for ckpt in history]
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        result = self._invoke(template, history)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(run.argv0) == len(expected)
        assert all(_same(argv0, path) for argv0, path in zip(run.argv0, expected))

    def test_a_program_named_by_the_checkpoint_id_that_is_not_on_path_is_not_started(
        self, tools, monkeypatch
    ):
        history = ("soup-eval-a", "soup-eval-b")
        for ckpt in history:
            _program(tools.project, ckpt)
            _program(tools.project / "rel", ckpt)
        run = _Recorder()
        monkeypatch.setattr(subprocess, "run", run)

        result = self._invoke("{ckpt} --model m", history)

        assert not run.calls
        output = _plain(result.output)
        assert (
            "Eval failed for soup-eval-a: --eval-command program 'soup-eval-a' was not "
            "found on PATH" in output
        )
        assert "BROKEN_AT" in output
        assert result.exit_code == 2, (result.output, repr(result.exception))
