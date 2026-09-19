"""`soup runs clean --keep-weights` must keep weights on every Click version.

Before this fix the option was declared as a bare `--keep-weights` with a
default of True. Click 8.2+ treats that as a flag whose presence means True,
but Click 8.1 treats a True-default flag with no negative form as a toggle:
typing `--keep-weights` yielded False and deleted whole checkpoints, the
opposite of what the user asked for. A paired `--keep-weights/--no-keep-weights`
flag has one meaning everywhere, and `--no-keep-weights` is the only way to
delete whole non-best checkpoints.
"""

import re
from pathlib import Path

import click
import pytest
import typer
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip Rich ANSI escapes and collapse whitespace (Rich wraps too)."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _clean_command() -> click.Command:
    root = typer.main.get_command(app)
    root_ctx = click.Context(root)
    runs_group = root.get_command(root_ctx, "runs")
    assert runs_group is not None
    clean = runs_group.get_command(click.Context(runs_group), "clean")
    assert clean is not None
    return clean


def _keep_weights_option() -> click.Option:
    clean = _clean_command()
    matches = [p for p in clean.params if p.name == "keep_weights"]
    assert len(matches) == 1, [p.name for p in clean.params]
    return matches[0]


class TestFlagShape:
    def test_option_is_a_paired_flag_defaulting_to_true(self) -> None:
        option = _keep_weights_option()
        assert option.opts == ["--keep-weights"]
        assert option.secondary_opts == ["--no-keep-weights"]
        assert option.default is True

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["--all"], True),
            (["--all", "--keep-weights"], True),
            (["--all", "--no-keep-weights"], False),
        ],
    )
    def test_parsed_value(self, argv: list, expected: bool) -> None:
        ctx = _clean_command().make_context("clean", list(argv))
        assert ctx.params["keep_weights"] is expected


_CKPT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt")


@pytest.fixture
def seeded_run(tmp_path: Path, monkeypatch):
    """A run under a temp cwd with checkpoint-10 (worse) and checkpoint-20 (best)."""
    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    out_dir = workdir / "output"
    for step in (10, 20):
        ckpt = out_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        for name in _CKPT_FILES:
            (ckpt / name).write_text(f"{name} at step {step}")

    tracker = ExperimentTracker(db_path=db_path)
    run_id = tracker.start_run(
        config_dict={}, device="cpu", device_name="CPU", gpu_info={},
    )
    tracker.finish_run(
        run_id=run_id,
        initial_loss=2.0,
        final_loss=1.0,
        total_steps=20,
        duration_secs=10.0,
        output_dir=str(out_dir),
    )
    tracker.log_metrics(run_id, step=10, loss=2.0)
    tracker.log_metrics(run_id, step=20, loss=1.0)
    return run_id, out_dir


def _invoke_clean(run_id: str, *extra: str):
    result = runner.invoke(app, ["runs", "clean", run_id, *extra])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return result


def _assert_best_untouched(out_dir: Path) -> None:
    best = out_dir / "checkpoint-20"
    for name in _CKPT_FILES:
        assert (best / name).exists(), name


def _assert_surgical(out_dir: Path) -> None:
    worse = out_dir / "checkpoint-10"
    assert (worse / "model.safetensors").exists()
    assert not (worse / "optimizer.pt").exists()
    assert not (worse / "scheduler.pt").exists()
    _assert_best_untouched(out_dir)


class TestBehaviour:
    def test_default_keeps_weights(self, seeded_run) -> None:
        run_id, out_dir = seeded_run
        _invoke_clean(run_id, "--force")
        _assert_surgical(out_dir)

    def test_explicit_keep_weights_keeps_weights(self, seeded_run) -> None:
        run_id, out_dir = seeded_run
        _invoke_clean(run_id, "--keep-weights", "--force")
        _assert_surgical(out_dir)

    def test_no_keep_weights_deletes_whole_non_best_checkpoints(self, seeded_run) -> None:
        run_id, out_dir = seeded_run
        _invoke_clean(run_id, "--no-keep-weights", "--force")
        assert not (out_dir / "checkpoint-10").exists()
        _assert_best_untouched(out_dir)

    def test_no_keep_weights_dry_run_deletes_nothing(self, seeded_run) -> None:
        run_id, out_dir = seeded_run
        result = _invoke_clean(run_id, "--no-keep-weights", "--dry-run")
        assert "Dry Run" in _plain(result.output)
        for step in (10, 20):
            for name in _CKPT_FILES:
                assert (out_dir / f"checkpoint-{step}" / name).exists()


def _walk(command: click.Command, path: tuple):
    yield path, command
    if isinstance(command, click.Group):
        ctx = click.Context(command)
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None:
                yield from _walk(sub, path + (name,))


class TestNoTrueDefaultFlagWithoutNegative:
    """A True-default flag with no `--no-` form means the opposite on Click 8.1."""

    def test_every_true_default_flag_has_a_negative_form(self) -> None:
        root = typer.main.get_command(app)
        offenders = []
        for path, command in _walk(root, ("soup",)):
            for param in command.params:
                if (
                    isinstance(param, click.Option)
                    and param.is_flag
                    and param.default is True
                    and not param.secondary_opts
                ):
                    offenders.append(f"{' '.join(path)} {'/'.join(param.opts)}")
        assert offenders == [], offenders
