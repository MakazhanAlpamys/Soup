"""v0.67.0 Part E — ``soup.lock`` shared run lockfile.

Tests for ``soup_cli/utils/soup_lock.py``:

- Frozen ``SoupLock`` dataclass (model SHA + dataset hash + env hash)
- ``compute_lock_closure`` deterministic + content-sensitive
- ``write_lock`` / ``read_lock`` atomic round-trip + cwd containment
- ``check_lock_drift`` detects each independent field drift
- Composes with v0.64 Part C ``soup env`` (env_hash is operator-supplied)
- CLI smoke (`soup lock`)
"""

from __future__ import annotations

import dataclasses
import os

import pytest


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import soup_lock

        assert hasattr(soup_lock, "SoupLock")
        assert hasattr(soup_lock, "LockDrift")
        assert hasattr(soup_lock, "compute_lock_closure")
        assert hasattr(soup_lock, "write_lock")
        assert hasattr(soup_lock, "read_lock")
        assert hasattr(soup_lock, "check_lock_drift")


class TestComputeLockClosure:
    def test_deterministic(self) -> None:
        from soup_cli.utils.soup_lock import compute_lock_closure

        a = compute_lock_closure(
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
        )
        b = compute_lock_closure(
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
        )
        assert a == b

    def test_content_sensitive(self) -> None:
        from soup_cli.utils.soup_lock import compute_lock_closure

        a = compute_lock_closure(
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
        )
        b = compute_lock_closure(
            base_model_sha="f" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
        )
        assert a != b

    def test_invalid_sha_rejected(self) -> None:
        from soup_cli.utils.soup_lock import compute_lock_closure

        with pytest.raises(ValueError):
            compute_lock_closure(
                base_model_sha="not-hex",
                dataset_sha="b" * 64,
                env_hash="c" * 64,
            )

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.soup_lock import compute_lock_closure

        with pytest.raises(TypeError):
            compute_lock_closure(
                base_model_sha=True,  # type: ignore[arg-type]
                dataset_sha="b" * 64,
                env_hash="c" * 64,
            )


class TestSoupLock:
    def test_construct(self) -> None:
        from soup_cli.utils.soup_lock import SoupLock

        lock = SoupLock(
            soup_version="0.67.0",
            base_model="meta-llama/Llama-3.1-8B",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
            created_at="2026-05-24T00:00:00Z",
        )
        assert lock.soup_version == "0.67.0"

    def test_frozen(self) -> None:
        from soup_cli.utils.soup_lock import SoupLock

        lock = SoupLock(
            soup_version="0.67.0",
            base_model="m",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
            created_at="2026-05-24",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            lock.soup_version = "9.99"  # type: ignore[misc]

    def test_sha_validation(self) -> None:
        from soup_cli.utils.soup_lock import SoupLock

        with pytest.raises(ValueError):
            SoupLock(
                soup_version="0.67.0",
                base_model="m",
                base_model_sha="too-short",
                dataset_sha="b" * 64,
                env_hash="c" * 64,
                closure_sha="d" * 64,
                created_at="2026-05-24",
            )

    def test_empty_base_model_rejected(self) -> None:
        from soup_cli.utils.soup_lock import SoupLock

        with pytest.raises(ValueError):
            SoupLock(
                soup_version="0.67.0",
                base_model="",
                base_model_sha="a" * 64,
                dataset_sha="b" * 64,
                env_hash="c" * 64,
                closure_sha="d" * 64,
                created_at="2026-05-24",
            )


class TestWriteReadLock:
    def test_roundtrip(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.soup_lock import (
            SoupLock,
            read_lock,
            write_lock,
        )

        monkeypatch.chdir(tmp_path)
        lock = SoupLock(
            soup_version="0.67.0",
            base_model="m",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
            created_at="2026-05-24",
        )
        path = str(tmp_path / "soup.lock")
        write_lock(lock, path)
        loaded = read_lock(path)
        assert loaded == lock

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.soup_lock import SoupLock, write_lock

        cwd = tmp_path / "work"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        lock = SoupLock(
            soup_version="0.67.0",
            base_model="m",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
            created_at="2026-05-24",
        )
        with pytest.raises(ValueError):
            write_lock(lock, str(tmp_path / "outside.lock"))

    def test_read_missing(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.soup_lock import read_lock

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            read_lock(str(tmp_path / "missing.lock"))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink test")
    def test_read_symlink_rejected(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.soup_lock import read_lock

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.lock"
        real.write_text('{"soup_version": "0.67.0"}', encoding="utf-8")
        sym = tmp_path / "sym.lock"
        os.symlink(real, sym)
        with pytest.raises(ValueError):
            read_lock(str(sym))


class TestCheckLockDrift:
    def _make_lock(self, **overrides):
        from soup_cli.utils.soup_lock import SoupLock

        base = dict(
            soup_version="0.67.0",
            base_model="m",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
            created_at="2026-05-24",
        )
        base.update(overrides)
        return SoupLock(**base)

    def test_no_drift(self) -> None:
        from soup_cli.utils.soup_lock import check_lock_drift

        a = self._make_lock()
        b = self._make_lock()
        drift = check_lock_drift(a, b)
        assert drift.ok is True
        assert drift.changes == ()

    def test_base_model_sha_drift(self) -> None:
        from soup_cli.utils.soup_lock import check_lock_drift

        a = self._make_lock()
        b = self._make_lock(base_model_sha="f" * 64)
        drift = check_lock_drift(a, b)
        assert drift.ok is False
        assert any("base_model_sha" in c for c in drift.changes)

    def test_dataset_sha_drift(self) -> None:
        from soup_cli.utils.soup_lock import check_lock_drift

        a = self._make_lock()
        b = self._make_lock(dataset_sha="f" * 64)
        drift = check_lock_drift(a, b)
        assert drift.ok is False
        assert any("dataset_sha" in c for c in drift.changes)

    def test_env_hash_drift(self) -> None:
        from soup_cli.utils.soup_lock import check_lock_drift

        a = self._make_lock()
        b = self._make_lock(env_hash="f" * 64)
        drift = check_lock_drift(a, b)
        assert drift.ok is False
        assert any("env_hash" in c for c in drift.changes)

    def test_non_lock_rejected(self) -> None:
        from soup_cli.utils.soup_lock import check_lock_drift

        a = self._make_lock()
        with pytest.raises(TypeError):
            check_lock_drift(a, "not-a-lock")  # type: ignore[arg-type]


class TestCliSmoke:
    def test_lock_help(self) -> None:
        from typer.testing import CliRunner

        from soup_cli.commands.lock import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_lock_write_command(self, tmp_path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from soup_cli.commands.lock import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "write",
                "--base-model", "test-model",
                "--base-sha", "a" * 64,
                "--dataset-sha", "b" * 64,
                "--env-hash", "c" * 64,
                "--output", "soup.lock",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "soup.lock").exists()

    # v0.71.1 #224 — auto-glue with `soup env lock`.
    def test_lock_write_auto_derives_env_hash(self, tmp_path, monkeypatch) -> None:
        import json

        from typer.testing import CliRunner

        from soup_cli.commands.lock import app
        from soup_cli.utils.env_lock import compute_env_hash, snapshot_env, write_lock

        monkeypatch.chdir(tmp_path)
        # Stand up a soup-env.lock the way `soup env lock` would.
        env = snapshot_env()
        write_lock(env, "soup-env.lock")
        expected_env_hash = compute_env_hash(env)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "write",
                "--base-model", "test-model",
                "--base-sha", "a" * 64,
                "--dataset-sha", "b" * 64,
                # NOTE: no --env-hash; must auto-derive from soup-env.lock.
                "--output", "soup.lock",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        written = json.loads((tmp_path / "soup.lock").read_text(encoding="utf-8"))
        assert written["env_hash"] == expected_env_hash

    def test_lock_write_missing_env_lock_errors(self, tmp_path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from soup_cli.commands.lock import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "write",
                "--base-model", "test-model",
                "--base-sha", "a" * 64,
                "--dataset-sha", "b" * 64,
                "--output", "soup.lock",
            ],
        )
        assert result.exit_code == 2
        assert "soup env lock" in result.output

    def test_lock_write_custom_env_lock_path(self, tmp_path, monkeypatch) -> None:
        import json

        from typer.testing import CliRunner

        from soup_cli.commands.lock import app
        from soup_cli.utils.env_lock import compute_env_hash, snapshot_env, write_lock

        monkeypatch.chdir(tmp_path)
        env = snapshot_env()
        write_lock(env, "custom-env.lock")
        expected = compute_env_hash(env)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "write",
                "--base-model", "test-model",
                "--base-sha", "a" * 64,
                "--dataset-sha", "b" * 64,
                "--env-lock", "custom-env.lock",
                "--output", "soup.lock",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        written = json.loads((tmp_path / "soup.lock").read_text(encoding="utf-8"))
        assert written["env_hash"] == expected

    def test_lock_write_explicit_env_hash_wins(self, tmp_path, monkeypatch) -> None:
        # Explicit --env-hash takes precedence over any soup-env.lock.
        import json

        from typer.testing import CliRunner

        from soup_cli.commands.lock import app
        from soup_cli.utils.env_lock import snapshot_env, write_lock

        monkeypatch.chdir(tmp_path)
        write_lock(snapshot_env(), "soup-env.lock")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "write",
                "--base-model", "test-model",
                "--base-sha", "a" * 64,
                "--dataset-sha", "b" * 64,
                "--env-hash", "c" * 64,
                "--output", "soup.lock",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        written = json.loads((tmp_path / "soup.lock").read_text(encoding="utf-8"))
        assert written["env_hash"] == "c" * 64


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "soup_lock.py").read_text(
            encoding="utf-8"
        )
        head_lines = [
            line
            for line in src.splitlines()[:50]
            if line.strip() and not line.strip().startswith("#")
        ]
        head = "\n".join(head_lines)
        for forbidden in ("import torch", "import transformers", "import peft"):
            assert forbidden not in head, f"top-level {forbidden!r}"

    def test_uses_atomic_write_helper(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "soup_lock.py").read_text(
            encoding="utf-8"
        )
        assert "atomic_write_text" in src
