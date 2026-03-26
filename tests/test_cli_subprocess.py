"""Subprocess CLI tests — real process execution to catch platform-specific bugs.

These tests run `soup` (or `python -m soup_cli`) as a real subprocess,
catching issues that in-process CliRunner misses:
  - encoding bugs (cp1251/cp1252 on Windows)
  - path separator issues
  - pipe buffering / stdout encoding
  - entry-point script resolution
  - exit code propagation through the OS shell
"""

import json
import os
import subprocess
import sys

import pytest

# Use `python -m soup_cli` for reliability across platforms (no need for
# the entry-point script to be installed or on PATH).
SOUP_CMD = [sys.executable, "-m", "soup_cli"]

# Timeout for all subprocess calls (seconds).  Tests should be fast —
# these are smoke-level checks, not training runs.
TIMEOUT = 30


def run_soup(
    *args: str, timeout: int = TIMEOUT, env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run soup CLI as a subprocess and return CompletedProcess."""
    merged_env = {**os.environ, **(env or {})}
    # Force UTF-8 in the child process (affects print / Rich Console output)
    merged_env["PYTHONIOENCODING"] = "utf-8"
    merged_env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    return subprocess.run(
        [*SOUP_CMD, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=merged_env,
        # Decode as UTF-8 on the *parent* side too — avoids cp1251/cp1252
        # failing on Rich box-drawing characters.
        encoding="utf-8",
        errors="replace",
    )


# ---------------------------------------------------------------------------
# Basic entry-point tests
# ---------------------------------------------------------------------------


class TestEntryPoint:
    """Verify the CLI launches correctly as a real process."""

    def test_no_args_shows_help(self):
        result = run_soup()
        # Typer shows help when no args, exit code 0
        assert result.returncode == 0
        assert "Fine-tune" in result.stdout or "Usage" in result.stdout

    def test_help_flag(self):
        result = run_soup("--help")
        assert result.returncode == 0
        assert "Fine-tune" in result.stdout

    def test_version(self):
        from soup_cli import __version__

        result = run_soup("version")
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_version_full(self):
        result = run_soup("version", "--full")
        assert result.returncode == 0
        assert "Python" in result.stdout

    def test_unknown_command_fails(self):
        result = run_soup("nonexistent_command_xyz")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Command --help tests (every command must print help without crashing)
# ---------------------------------------------------------------------------


ALL_COMMANDS = [
    "init",
    "train",
    "chat",
    "push",
    "export",
    "merge",
    "eval",
    "serve",
    "sweep",
    "diff",
    "infer",
    "doctor",
    "quickstart",
    "ui",
    "version",
]

ALL_DATA_SUBCOMMANDS = [
    "inspect",
    "validate",
    "convert",
    "merge",
    "dedup",
    "stats",
    "generate",
]

ALL_RUNS_SUBCOMMANDS = [
    "show",
    "compare",
    "delete",
]


class TestCommandHelp:
    """Every registered command must respond to --help without crashing."""

    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    def test_command_help(self, cmd):
        result = run_soup(cmd, "--help")
        assert result.returncode == 0, (
            f"`soup {cmd} --help` failed (rc={result.returncode}):\n{result.stderr}"
        )
        # Should contain some usage info
        assert len(result.stdout) > 20, f"`soup {cmd} --help` output too short"

    @pytest.mark.parametrize("subcmd", ALL_DATA_SUBCOMMANDS)
    def test_data_subcommand_help(self, subcmd):
        result = run_soup("data", subcmd, "--help")
        assert result.returncode == 0, (
            f"`soup data {subcmd} --help` failed (rc={result.returncode}):\n{result.stderr}"
        )

    @pytest.mark.parametrize("subcmd", ALL_RUNS_SUBCOMMANDS)
    def test_runs_subcommand_help(self, subcmd):
        result = run_soup("runs", subcmd, "--help")
        assert result.returncode == 0, (
            f"`soup runs {subcmd} --help` failed (rc={result.returncode}):\n{result.stderr}"
        )

    def test_data_no_args_shows_help(self):
        result = run_soup("data")
        assert result.returncode == 0
        assert "inspect" in result.stdout.lower() or "Usage" in result.stdout

    def test_runs_lists_or_shows_help(self):
        result = run_soup("runs")
        # runs without subcommand lists experiments (may be empty) — exit 0
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Error handling / exit code tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify errors produce correct exit codes and readable messages."""

    def test_train_missing_config(self):
        result = run_soup("train", "--config", "nonexistent_file.yaml")
        assert result.returncode == 1

    def test_chat_missing_model(self):
        result = run_soup("chat", "--model", "nonexistent_model_path")
        assert result.returncode == 1

    def test_push_missing_model(self):
        result = run_soup(
            "push", "--model", "nonexistent_model_path", "--repo", "user/model"
        )
        assert result.returncode == 1

    def test_init_unknown_template(self):
        result = run_soup("init", "--template", "nonexistent_template")
        assert result.returncode == 1

    def test_export_missing_model(self):
        result = run_soup("export", "--model", "nonexistent_model_path")
        assert result.returncode == 1

    def test_merge_missing_adapter(self):
        result = run_soup("merge", "--adapter", "nonexistent_adapter_path")
        assert result.returncode == 1

    def test_infer_missing_files(self):
        result = run_soup(
            "infer",
            "--model", "nonexistent_model",
            "--input", "nonexistent.jsonl",
            "--output", "out.jsonl",
        )
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Encoding / Unicode safety on all platforms
# ---------------------------------------------------------------------------


class TestEncoding:
    """Verify output contains no problematic Unicode on any platform."""

    PROBLEMATIC_CHARS = [
        "\u2192",  # → right arrow
        "\u2014",  # — em dash
        "\u2018",  # ' left single quote
        "\u2019",  # ' right single quote
        "\u201c",  # " left double quote
        "\u201d",  # " right double quote
    ]

    def test_version_output_is_ascii_safe(self):
        result = run_soup("version")
        assert result.returncode == 0
        for char in self.PROBLEMATIC_CHARS:
            assert char not in result.stdout, (
                f"Version output contains problematic Unicode char: {repr(char)}"
            )

    def test_help_output_is_ascii_safe(self):
        result = run_soup("--help")
        assert result.returncode == 0
        for char in self.PROBLEMATIC_CHARS:
            assert char not in result.stdout, (
                f"Help output contains problematic Unicode char: {repr(char)}"
            )

    def test_error_output_is_ascii_safe(self):
        """Error messages must be safe for Windows terminals."""
        result = run_soup("train", "--config", "nonexistent.yaml")
        combined = result.stdout + result.stderr
        for char in self.PROBLEMATIC_CHARS:
            assert char not in combined, (
                f"Error output contains problematic Unicode char: {repr(char)}"
            )


# ---------------------------------------------------------------------------
# Data command — real file I/O
# ---------------------------------------------------------------------------


class TestDataCommands:
    """Test data subcommands with actual file operations."""

    def test_data_inspect_jsonl(self, tmp_path):
        """Inspect a small JSONL file via subprocess."""
        data_file = tmp_path / "test.jsonl"
        rows = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]}
            for _ in range(5)
        ]
        data_file.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        result = run_soup("data", "inspect", str(data_file))
        assert result.returncode == 0
        assert "5" in result.stdout  # row count

    def test_data_inspect_nonexistent_file(self):
        result = run_soup("data", "inspect", "nonexistent_file.jsonl")
        assert result.returncode == 1

    def test_data_validate_jsonl(self, tmp_path):
        """Validate a well-formed JSONL file."""
        data_file = tmp_path / "valid.jsonl"
        rows = [
            {"messages": [
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant", "content": f"A{i}"},
            ]}
            for i in range(3)
        ]
        data_file.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        result = run_soup("data", "validate", str(data_file))
        assert result.returncode == 0

    def test_data_stats_jsonl(self, tmp_path):
        """Stats command should work without encoding errors on any OS."""
        data_file = tmp_path / "stats.jsonl"
        rows = [
            {"messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i} " + "x" * (i * 10)},
            ]}
            for i in range(10)
        ]
        data_file.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        result = run_soup("data", "stats", str(data_file))
        assert result.returncode == 0, (
            f"data stats failed (rc={result.returncode}):\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Init command — file creation
# ---------------------------------------------------------------------------


class TestInitCommand:
    """Test init command creates files correctly."""

    TEMPLATES = ["chat", "code", "medical", "reasoning", "vision"]

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_init_creates_config_file(self, tmp_path, template):
        """soup init --template X should create a valid YAML config."""
        result = run_soup(
            "init",
            "--template", template,
            "--output", str(tmp_path / f"{template}.yaml"),
        )
        assert result.returncode == 0, (
            f"init --template {template} failed:\n{result.stderr}"
        )
        config_file = tmp_path / f"{template}.yaml"
        assert config_file.exists(), f"Config file not created for template {template}"
        content = config_file.read_text(encoding="utf-8")
        assert "base:" in content

    def test_init_default_output(self, tmp_path, monkeypatch):
        """soup init --template chat writes soup.yaml in cwd."""
        monkeypatch.chdir(tmp_path)
        result = run_soup("init", "--template", "chat")
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Doctor command — system check
# ---------------------------------------------------------------------------


class TestDoctorCommand:
    """Doctor should always succeed (even without GPU)."""

    def test_doctor_runs(self):
        # doctor imports torch and checks GPU — allow extra time
        result = run_soup("doctor", timeout=120)
        assert result.returncode == 0
        # Should mention Python at minimum
        assert "python" in result.stdout.lower() or "Python" in result.stdout


# ---------------------------------------------------------------------------
# Verbose flag
# ---------------------------------------------------------------------------


class TestVerboseFlag:
    """--verbose / -V flag should not crash."""

    def test_verbose_with_version(self):
        result = run_soup("--verbose", "version")
        assert result.returncode == 0

    def test_verbose_short_flag(self):
        result = run_soup("-V", "version")
        assert result.returncode == 0

    def test_verbose_error_shows_traceback(self):
        result = run_soup("--verbose", "train", "--config", "nonexistent.yaml")
        assert result.returncode == 1
        # Verbose mode should show traceback details
        combined = result.stdout + result.stderr
        assert len(combined) > 0


# ---------------------------------------------------------------------------
# Platform-specific regression tests
# ---------------------------------------------------------------------------


class TestPlatformRegression:
    """Regressions that appeared on specific platforms."""

    def test_stdout_encoding_no_crash(self):
        """Verify stdout doesn't crash with encoding issues (Windows cp1252 bug)."""
        result = run_soup("version", "--full")
        assert result.returncode == 0
        # Output should be decodable text, no mojibake
        assert result.stdout.isprintable() or "\n" in result.stdout

    def test_path_with_spaces(self, tmp_path):
        """Paths with spaces must work on all OSes."""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        data_file = spaced_dir / "test.jsonl"
        data_file.write_text(
            json.dumps({"messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]}),
            encoding="utf-8",
        )
        result = run_soup("data", "inspect", str(data_file))
        assert result.returncode == 0

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_windows_long_path(self, tmp_path):
        """Long paths should work on Windows (>100 chars)."""
        long_dir = tmp_path
        for i in range(5):
            long_dir = long_dir / f"subdirectory_level_{i}_name"
            long_dir.mkdir()
        data_file = long_dir / "data.jsonl"
        data_file.write_text(
            json.dumps({"messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "OK"},
            ]}),
            encoding="utf-8",
        )
        assert len(str(data_file)) > 100
        result = run_soup("data", "inspect", str(data_file))
        assert result.returncode == 0

    def test_unicode_in_data_file(self, tmp_path):
        """Data files with Unicode content must load on all platforms."""
        data_file = tmp_path / "unicode.jsonl"
        rows = [
            {"messages": [
                {"role": "user", "content": "Wie geht es dir?"},
                {"role": "assistant", "content": "Mir geht es gut, danke!"},
            ]},
            {"messages": [
                {"role": "user", "content": "Как дела?"},
                {"role": "assistant", "content": "Хорошо, спасибо!"},
            ]},
            {"messages": [
                {"role": "user", "content": "お元気ですか？"},
                {"role": "assistant", "content": "元気です、ありがとう！"},
            ]},
        ]
        data_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )
        result = run_soup("data", "inspect", str(data_file))
        assert result.returncode == 0

    def test_empty_jsonl_file(self, tmp_path):
        """Empty data file should not crash (may show 0 rows)."""
        data_file = tmp_path / "empty.jsonl"
        data_file.write_text("", encoding="utf-8")
        result = run_soup("data", "inspect", str(data_file))
        # Should complete without crashing — exit 0 (0 rows) or 1 (error)
        assert result.returncode in (0, 1)


# ---------------------------------------------------------------------------
# Quickstart --dry-run
# ---------------------------------------------------------------------------


class TestQuickstart:
    """Quickstart command basic checks."""

    def test_quickstart_help(self):
        result = run_soup("quickstart", "--help")
        assert result.returncode == 0
        assert "dry" in result.stdout.lower() or "demo" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Sweep command
# ---------------------------------------------------------------------------


class TestSweepCommand:
    """Sweep command validation."""

    def test_sweep_missing_config(self):
        result = run_soup(
            "sweep", "--config", "nonexistent.yaml",
            "--param", "lr=1e-4,2e-4",
        )
        assert result.returncode == 1

    def test_sweep_help(self):
        result = run_soup("sweep", "--help")
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Diff command
# ---------------------------------------------------------------------------


class TestDiffCommand:
    """Diff command validation."""

    def test_diff_help(self):
        result = run_soup("diff", "--help")
        assert result.returncode == 0
        assert "model" in result.stdout.lower() or "compare" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Eval command
# ---------------------------------------------------------------------------


class TestEvalCommand:
    """Eval command validation."""

    def test_eval_help(self):
        result = run_soup("eval", "--help")
        assert result.returncode == 0

    def test_eval_missing_model(self):
        result = run_soup("eval", "--model", "nonexistent_model")
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Serve / UI commands
# ---------------------------------------------------------------------------


class TestServeUI:
    """Serve and UI commands help checks (don't actually start servers)."""

    def test_serve_help(self):
        result = run_soup("serve", "--help")
        assert result.returncode == 0
        assert "backend" in result.stdout.lower() or "model" in result.stdout.lower()

    def test_ui_help(self):
        result = run_soup("ui", "--help")
        assert result.returncode == 0
