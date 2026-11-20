"""Follow-up commands Soup prints repeat a configured path as a name, in every shell.

Covers the two places a configured path is printed inside a command that a person is
told to run:

* the "Run:" line of a cloud plan (``plan_modal_run`` / ``plan_lambda_run``), whose
  stub path is quoted the way the RunPod plan already quotes it;
* the "Quick test / Push / Merge / Export" lines of the "Training Complete!" panel.

The Modal stub's own download hint is covered in ``test_modal_stub_outputs.py``; the
quoting rule itself in ``test_paste_arg.py``.
"""

from __future__ import annotations

import ast
import os
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

_SOUP_YAML = """base: hf-internal-testing/tiny-random-gpt2
task: sft
data:
  train: data.jsonl
training:
  epochs: 1
output: ./output
"""


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
    return tmp_path


class TestCloudRunCommandQuotesTheStubPath:
    @pytest.mark.parametrize(
        ("stub_path", "quoted"),
        [
            ("soup_modal_app.py", "soup_modal_app.py"),
            ("my stub.py", "'my stub.py'"),
            ("stub$(cmd).py", "'stub$(cmd).py'"),
            ("stub;cmd.py", "'stub;cmd.py'"),
            ("stub&next.py", "'stub&next.py'"),
        ],
    )
    def test_modal_plan(self, project, stub_path, quoted):
        from soup_cli.cloud.modal import plan_modal_run

        plan = plan_modal_run(
            "soup.yaml",
            gpu="a100",
            output_dir="./output",
            soup_version="0.75.1",
            stub_path=stub_path,
        )

        assert plan.run_command == f"modal run {quoted}"

    @pytest.mark.parametrize(
        ("stub_path", "quoted"),
        [
            ("soup_lambda_app.py", "soup_lambda_app.py"),
            ("my stub.py", "'my stub.py'"),
            ("stub$(cmd).py", "'stub$(cmd).py'"),
            ("stub;cmd.py", "'stub;cmd.py'"),
            ("stub&next.py", "'stub&next.py'"),
        ],
    )
    def test_lambda_plan(self, project, stub_path, quoted):
        from soup_cli.cloud.lambda_labs import plan_lambda_run

        plan = plan_lambda_run(
            "soup.yaml",
            gpu="a100",
            output_dir="output",
            soup_version="0.75.1",
            stub_path=stub_path,
        )

        assert plan.run_command == f"python -I {quoted}"

    def test_default_commands_are_unchanged(self, project):
        from soup_cli.cloud.lambda_labs import plan_lambda_run
        from soup_cli.cloud.modal import plan_modal_run

        modal_plan = plan_modal_run(
            "soup.yaml", gpu="a100", output_dir="./output", soup_version="0.75.1"
        )
        lambda_plan = plan_lambda_run(
            "soup.yaml", gpu="a100", output_dir="output", soup_version="0.75.1"
        )

        assert modal_plan.run_command == "modal run soup_modal_app.py"
        assert lambda_plan.run_command == "python -I soup_lambda_app.py"


def _plain(markup: str) -> str:
    """What a terminal shows for ``markup``: rendered through Rich, markup consumed."""
    buffer = StringIO()
    Console(file=buffer, width=500, force_terminal=False, color_system=None).print(
        markup, emoji=False, highlight=False
    )
    return buffer.getvalue()


class TestTrainingCompleteCommands:
    def _commands(
        self, output_dir, merge_note=None, windows_separators=False, **hints
    ) -> str:
        from soup_cli.commands.train import _completion_commands

        return _plain(
            _completion_commands(
                output_dir, merge_note, windows_separators=windows_separators, **hints
            )
        )

    def test_the_backend_and_dequantize_flags_are_kept_next_to_the_path(self):
        shown = self._commands(
            "./output", backend_hint=" --backend mlx", dequantize_hint=" --dequantize"
        )

        assert "soup chat --backend mlx --model ./output" in shown
        assert "soup merge --backend mlx --adapter ./output --dequantize" in shown
        assert "soup push --model ./output" in shown
        assert "soup export --model ./output" in shown

    def test_the_flags_are_kept_when_the_path_is_left_out(self):
        shown = self._commands(
            "out$HOME", backend_hint=" --backend mlx", dequantize_hint=" --dequantize"
        )

        assert "soup chat --backend mlx --model <output directory>" in shown
        assert "soup merge --backend mlx --adapter <output directory> --dequantize" in shown

    def test_plain_output_is_repeated_as_it_was_before(self):
        shown = self._commands("./output")

        assert "soup chat --model ./output" in shown
        assert "soup push --model ./output" in shown
        assert "soup merge --adapter ./output" in shown
        assert "soup export --model ./output" in shown
        assert "<output directory>" not in shown

    def test_output_with_a_space_is_double_quoted(self):
        shown = self._commands("my outputs/run 1")

        assert 'soup chat --model "my outputs/run 1"' in shown
        assert 'soup merge --adapter "my outputs/run 1"' in shown

    @pytest.mark.parametrize(
        "output_dir",
        [
            "out$(cmd)",
            "out`cmd`",
            'out"; cmd; echo "',
            "out%COMSPEC%",
            "out!x",
            "out\\dir",
            "out\x1b[2Jx",
            "out“x",
        ],
    )
    def test_output_a_shell_would_interpret_is_left_out_of_every_command(self, output_dir):
        shown = self._commands(output_dir)

        for command in ("soup chat --model", "soup push --model", "soup merge --adapter",
                        "soup export --model"):
            assert f"{command} <output directory>" in shown, shown
        assert output_dir not in shown
        assert "\x1b" not in shown
        assert "is not repeated in these commands" in " ".join(shown.split())

    def test_a_leading_dash_cannot_become_an_option(self):
        shown = self._commands("-rf")

        assert "soup chat --model ./-rf" in shown
        assert "soup chat --model -rf" not in shown

    @pytest.mark.parametrize("output_dir", ["out[/bold]", "[bold red]out", "a[/]b", "[x"])
    def test_rich_markup_in_the_name_is_shown_literally_and_does_not_raise(self, output_dir):
        shown = self._commands(output_dir)

        assert f'soup chat --model "{output_dir}"' in shown

    def test_a_merge_note_replaces_the_merge_line(self):
        shown = self._commands("./output", "ReLoRA output is already dense; no soup merge")

        assert "ReLoRA output is already dense; no soup merge" in shown
        assert "soup merge --adapter" not in shown
        assert "soup export --model ./output" in shown

    def test_a_nested_windows_path_is_repeated_with_forward_slashes(self):
        shown = self._commands("output\\sub", windows_separators=True)

        assert "soup chat --model output/sub" in shown
        assert "soup merge --adapter output/sub" in shown
        assert "<output directory>" not in shown

    def test_the_same_path_is_left_out_where_a_backslash_is_a_name_character(self):
        shown = self._commands("output\\sub", windows_separators=False)

        assert "soup chat --model <output directory>" in shown

    def test_panel_builds_its_commands_through_the_helper(self):
        """The "Training Complete!" panel in ``train()`` calls the helper with the run's
        output directory and does not format the path itself."""
        import ast

        import soup_cli.commands.train as train_module

        source = Path(train_module.__file__).read_text(encoding="utf-8")
        assert "--model {result['output_dir']}" not in source
        assert "--adapter {result['output_dir']}" not in source
        assert "Output: [bold]{result['output_dir']}" not in source
        train_function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        calls = [
            node
            for node in ast.walk(train_function)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_completion_commands"
        ]
        assert len(calls) == 1
        assert ast.unparse(calls[0].args[0]) == "result['output_dir']"


class TestCloudPlanPanel:
    def test_the_plan_panel_does_not_print_control_characters_of_the_output(
        self, tmp_path, monkeypatch
    ):
        import yaml

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config
        from tests.conftest import make_cli_runner

        monkeypatch.chdir(tmp_path)
        runner = make_cli_runner()
        init = runner.invoke(app, ["init", "--template", "chat", "--output", "soup.yaml"])
        assert init.exit_code == 0, init.output
        cfg = load_config(Path("soup.yaml"))
        cfg.data.train = "teknium/OpenHermes-2.5"
        cfg.output = "out\x1b[2Jx"
        Path("soup.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )

        result = runner.invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "modal", "--gpu", "a100"]
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "\x1b[2J" not in result.output
        assert "out" in result.output


_BASE_PLACEHOLDER = "<base model>"


def _stream_preflight_text(monkeypatch, tmp_path, base: str) -> str:
    """The layer-streaming disk pre-flight panel for a base with one stale cache slot."""
    from soup_cli.trainer import stream_setup

    buffer = StringIO()
    monkeypatch.setattr(
        stream_setup, "console", Console(file=buffer, width=300, color_system=None)
    )
    monkeypatch.setattr(stream_setup, "_disk_volume", lambda path: (1, 10**15))
    slot = SimpleNamespace(
        name="old-slot", path=str(tmp_path / "old"), bytes=10, referenced=False,
        reason="", stripes=(),
    )
    stream_setup._render_stream_disk_preflight(
        source_bytes=0, materialized_copy_bytes=0, materialize_bytes=0,
        materialized_path=str(tmp_path), shard_bytes=1000, shard_write_bytes=0,
        shard_path=str(tmp_path / "cache"), other_slots=[slot], base=base,
    )
    return buffer.getvalue()


def _prune_command_line(text: str) -> str:
    (line,) = [
        line for line in text.splitlines()
        if "Remove unreferenced slots (opt-in): " in line
    ]
    return line.strip(" |\u2502").split("(opt-in): ", 1)[1]


class TestPruneCommandInTheStreamingPreflight:
    """The pre-flight lists stale cache slots and prints ``soup prune-layer-cache --model
    <base>`` with the configured base model: the base is spelled for every shell, or left
    out with a note."""

    @pytest.mark.parametrize(
        ("base", "spelled"),
        [
            ("org/model-1.5", "org/model-1.5"),
            ("my models/base", '"my models/base"'),
            ("org/a;b", '"org/a;b"'),
            ("org/a&b", '"org/a&b"'),
            ("-base", "./-base"),
            ("org/a$(x)", _BASE_PLACEHOLDER),
            ("org/a`x`", _BASE_PLACEHOLDER),
            ('org/a"b', _BASE_PLACEHOLDER),
            ("org/a%x%", _BASE_PLACEHOLDER),
            ("org/a!b", _BASE_PLACEHOLDER),
            ("org/a\x1b[2Jb", _BASE_PLACEHOLDER),
            ("", _BASE_PLACEHOLDER),
        ],
    )
    def test_the_base_is_one_argument_or_left_out(self, monkeypatch, tmp_path, base, spelled):
        text = _stream_preflight_text(monkeypatch, tmp_path, base)

        assert _prune_command_line(text) == f"soup prune-layer-cache --model {spelled} --yes"
        assert "\x1b" not in text

    def test_a_left_out_base_is_explained(self, monkeypatch, tmp_path):
        text = " ".join(_stream_preflight_text(monkeypatch, tmp_path, "org/a$(x)").split())

        assert f"Replace {_BASE_PLACEHOLDER} with the base model name" in text
        assert "is not repeated in the command" in text

    def test_a_plain_base_has_no_note(self, monkeypatch, tmp_path):
        text = _stream_preflight_text(monkeypatch, tmp_path, "org/model")

        assert "is not repeated in the command" not in text
        assert _BASE_PLACEHOLDER not in text

    def test_rich_markup_in_the_base_is_shown_literally(self, monkeypatch, tmp_path):
        text = _stream_preflight_text(monkeypatch, tmp_path, "org/[bold]model")

        assert _prune_command_line(text) == 'soup prune-layer-cache --model "org/[bold]model" --yes'

    def test_a_windows_style_path_follows_the_host_separator(self, monkeypatch, tmp_path):
        text = _stream_preflight_text(monkeypatch, tmp_path, "C:\\models\\base")

        expected = "C:/models/base" if os.sep == "\\" else _BASE_PLACEHOLDER
        assert _prune_command_line(text) == f"soup prune-layer-cache --model {expected} --yes"


def _stub_string(stub: str, name: str) -> str:
    """The string literal a rendered Modal stub assigns to ``name``."""
    for node in ast.parse(stub).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"the stub assigns no {name}")


class TestTheHostSeparatorDefault:
    """No test above passes ``windows_separators``; these call without it, as production
    does: a nested path written with backslashes is repeated with forward slashes on a
    Windows host and is a reason for the placeholder elsewhere."""

    def test_the_training_report_resolves_it_from_the_host(self):
        from soup_cli.commands.train import _completion_commands

        shown = _completion_commands("out\\dir")

        if os.sep == "\\":
            assert "soup chat --model out/dir[/]" in shown
            assert "<output directory>" not in shown
        else:
            assert "soup chat --model <output directory>[/]" in shown
            assert "out\\dir" not in shown

    def test_the_modal_retry_command_resolves_it_from_the_host(self):
        from soup_cli.cloud.modal import render_modal_stub

        stub = render_modal_stub(
            _SOUP_YAML, gpu="a100", output_dir="out\\dir", soup_version="0.75.0",
            run_name="run-abc",
        )

        destination = "out/dir" if os.sep == "\\" else "<local output directory>"
        assert _stub_string(stub, "_RETRY_COMMAND") == (
            f"modal volume get soup-outputs /run-abc {destination} --force"
        )
