"""Exercise the training completion panel with a mocked trainer result."""

from types import SimpleNamespace

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def test_completion_panel_shows_output_on_one_line(tmp_path, monkeypatch):
    import soup_cli.commands.train as train_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(train_mod.console, "width", 160)
    monkeypatch.setattr(train_mod, "detect_device", lambda **_: ("cpu", "test CPU"))
    monkeypatch.setattr(train_mod, "get_gpu_info", lambda **_: {"memory_total": "0 GB"})
    # Windows cannot create a directory whose name contains these characters.
    monkeypatch.setattr("soup_cli.utils.run_provenance.write_run_provenance", lambda *_: None)

    (tmp_path / "base").mkdir()
    (tmp_path / "train.jsonl").write_text(
        '{"instruction":"Question","output":"Answer"}\n', encoding="utf-8"
    )
    (tmp_path / "soup.yaml").write_text(
        "base: ./base\n"
        "output: ./configured\n"
        "data: {train: ./train.jsonl, format: alpaca, val_split: 0}\n"
        "training: {quantization: none, lora: {r: 4}}\n",
        encoding="utf-8",
    )
    result = {
        "output_dir": "./runs\nnew\rline\tend",
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "total_steps": 1,
        "duration_secs": 1.0,
    }
    trainer = SimpleNamespace(setup=lambda _: None, train=lambda **_: result)
    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", lambda *_args, **_kw: trainer)

    run = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])

    assert run.exit_code == 0, (run.output, repr(run.exception))
    assert r"Output: ./runs\nnew\rline\tend" in strip_ansi(run.output)
