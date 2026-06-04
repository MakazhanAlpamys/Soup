"""v0.68.0 Part E — ``soup local-rl`` personal-LLM flywheel daemon.

Wrap Ollama / MLX inference, capture thumbs into SQLite, harvest DPO pairs,
and (in v0.68.1) DPO-train nightly from systemd / launchd. v0.68.0 ships
the SQLite schema + thumbs recording + DPO-pair harvester live; nightly
train scheduler is the stub. Smaller-scope cousin of v0.58 ``soup loop``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import local_rl

        assert hasattr(local_rl, "SUPPORTED_LOCAL_RL_BACKENDS")
        assert hasattr(local_rl, "SUPPORTED_LOCAL_RL_TRAIN_METHODS")
        assert hasattr(local_rl, "validate_local_rl_backend")
        assert hasattr(local_rl, "validate_local_rl_train_method")
        assert hasattr(local_rl, "LocalRLConfig")
        assert hasattr(local_rl, "init_local_rl_db")
        assert hasattr(local_rl, "record_thumb")
        assert hasattr(local_rl, "harvest_dpo_pairs")
        assert hasattr(local_rl, "run_nightly_train")


class TestAllowlists:
    def test_backend_frozenset(self) -> None:
        from soup_cli.utils.local_rl import SUPPORTED_LOCAL_RL_BACKENDS

        assert isinstance(SUPPORTED_LOCAL_RL_BACKENDS, frozenset)
        assert "ollama" in SUPPORTED_LOCAL_RL_BACKENDS
        assert "mlx" in SUPPORTED_LOCAL_RL_BACKENDS

    def test_train_method_frozenset(self) -> None:
        from soup_cli.utils.local_rl import SUPPORTED_LOCAL_RL_TRAIN_METHODS

        assert isinstance(SUPPORTED_LOCAL_RL_TRAIN_METHODS, frozenset)
        assert "dpo" in SUPPORTED_LOCAL_RL_TRAIN_METHODS
        assert "kto" in SUPPORTED_LOCAL_RL_TRAIN_METHODS
        assert "orpo" in SUPPORTED_LOCAL_RL_TRAIN_METHODS

    def test_backend_immutable(self) -> None:
        from soup_cli.utils.local_rl import SUPPORTED_LOCAL_RL_BACKENDS

        with pytest.raises(AttributeError):
            SUPPORTED_LOCAL_RL_BACKENDS.add("x")  # type: ignore[attr-defined]


class TestValidators:
    def test_backend_happy(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        assert validate_local_rl_backend("ollama") == "ollama"

    def test_backend_case_insensitive(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        assert validate_local_rl_backend("OLLAMA") == "ollama"

    def test_backend_bool_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        with pytest.raises(TypeError):
            validate_local_rl_backend(True)  # type: ignore[arg-type]

    def test_backend_unknown_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        with pytest.raises(ValueError, match="unknown"):
            validate_local_rl_backend("evil")

    def test_train_method_happy(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        assert validate_local_rl_train_method("dpo") == "dpo"

    def test_train_method_unknown_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        with pytest.raises(ValueError, match="unknown"):
            validate_local_rl_train_method("ppo")  # PPO not in allowlist


class TestLocalRLConfig:
    def test_frozen(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.local_rl import LocalRLConfig

        monkeypatch.chdir(tmp_path)
        cfg = LocalRLConfig(
            backend="ollama",
            model="llama3:8b",
            db_path="local_rl.db",
            train_method="dpo",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.backend = "mlx"  # type: ignore[misc]

    def test_invalid_backend_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import LocalRLConfig

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            LocalRLConfig(
                backend="evil",
                model="m",
                db_path="db.db",
                train_method="dpo",
            )

    def test_invalid_train_method_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import LocalRLConfig

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            LocalRLConfig(
                backend="ollama",
                model="m",
                db_path="db.db",
                train_method="ppo",
            )


class TestInitDb:
    def test_creates_tables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sqlite3

        from soup_cli.utils.local_rl import init_local_rl_db

        monkeypatch.chdir(tmp_path)
        db_path = "rl.db"
        init_local_rl_db(db_path)
        assert os.path.exists(db_path)
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        names = {r[0] for r in rows}
        assert "interactions" in names
        assert "thumbs" in names

    def test_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        # Second call must not raise.
        init_local_rl_db("rl.db")

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db

        outside = tmp_path / "outside"
        outside.mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            init_local_rl_db(str(outside / "rl.db"))


class TestRecordThumb:
    def test_happy_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sqlite3

        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        record_thumb(
            db_path="rl.db",
            prompt="capital of france?",
            response="paris",
            thumb="up",
        )
        with sqlite3.connect("rl.db") as conn:
            rows = conn.execute("SELECT thumb FROM thumbs").fetchall()
        assert rows == [("up",)]

    def test_happy_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        record_thumb(
            db_path="rl.db", prompt="x", response="y", thumb="down"
        )

    def test_invalid_thumb_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(ValueError):
            record_thumb(
                db_path="rl.db", prompt="x", response="y", thumb="meh"
            )

    def test_null_byte_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(ValueError):
            record_thumb(
                db_path="rl.db",
                prompt="x\x00",
                response="y",
                thumb="up",
            )

    def test_bool_thumb_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(TypeError):
            record_thumb(
                db_path="rl.db",
                prompt="x",
                response="y",
                thumb=True,  # type: ignore[arg-type]
            )

    def test_oversize_prompt_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            MAX_PROMPT_LEN,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(ValueError):
            record_thumb(
                db_path="rl.db",
                prompt="a" * (MAX_PROMPT_LEN + 1),
                response="y",
                thumb="up",
            )


class TestHarvestDpoPairs:
    def test_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        assert harvest_dpo_pairs("rl.db") == ()

    def test_pairs_from_thumbs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        record_thumb(
            db_path="rl.db",
            prompt="q",
            response="good response",
            thumb="up",
        )
        record_thumb(
            db_path="rl.db",
            prompt="q",
            response="bad response",
            thumb="down",
        )
        pairs = harvest_dpo_pairs("rl.db")
        # One prompt with both up + down should yield exactly one DPO pair.
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.prompt == "q"
        assert pair.chosen == "good response"
        assert pair.rejected == "bad response"

    def test_returns_tuple(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        assert isinstance(harvest_dpo_pairs("rl.db"), tuple)


class TestNightlyTrainDeferred:
    def test_live_skips_when_no_pairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # v0.71.13 #229: live runner; an empty DB harvests 0 pairs and skips
        # (no NotImplementedError, no train call).
        from soup_cli.utils.local_rl import (
            LocalRLConfig,
            init_local_rl_db,
            run_nightly_train,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        cfg = LocalRLConfig(
            backend="ollama",
            model="org/model",
            db_path="rl.db",
            train_method="dpo",
        )
        res = run_nightly_train(
            cfg, min_pairs=1, train_fn=lambda **kw: pytest.fail("no pairs")
        )
        assert res.status == "skipped_insufficient_pairs"

    def test_non_config_rejected(self) -> None:
        from soup_cli.utils.local_rl import run_nightly_train

        with pytest.raises(TypeError):
            run_nightly_train({})  # type: ignore[arg-type]


class TestCli:
    def test_help(self) -> None:
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["local-rl", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_init_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["local-rl", "init", "--db", "rl.db"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "rl.db").exists()

    def test_record_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(app, ["local-rl", "init", "--db", "rl.db"])
        result = runner.invoke(
            app,
            [
                "local-rl",
                "record",
                "--db",
                "rl.db",
                "--prompt",
                "q",
                "--response",
                "a",
                "--thumb",
                "up",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_status_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(app, ["local-rl", "init", "--db", "rl.db"])
        result = runner.invoke(app, ["local-rl", "status", "--db", "rl.db"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_harvest_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(app, ["local-rl", "init", "--db", "rl.db"])
        result = runner.invoke(
            app, ["local-rl", "harvest", "--db", "rl.db", "--output", "p.jsonl"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_train_no_once_renders_scheduler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # v0.71.13 #229: `train` (no --once) renders the systemd/launchd
        # scaffold and exits 0 (no systemctl call).
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(app, ["local-rl", "init", "--db", "rl.db"])
        result = runner.invoke(
            app,
            [
                "local-rl",
                "train",
                "--db",
                "rl.db",
                "--backend",
                "ollama",
                "--model",
                "org/model",
                "--scheduler-dir",
                "sched",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "sched" / "soup-local-rl.timer").is_file()


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "local_rl.py"
        )
        text = path.read_text(encoding="utf-8")
        for token in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport ollama",
            "\nimport mlx",
        ):
            assert token not in text

    def test_cli_registered(self) -> None:
        from soup_cli.cli import app

        names = [t.name for t in app.registered_groups]
        assert "local-rl" in names
