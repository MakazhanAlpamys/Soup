"""Tests for Soup Cans shareable artifact format (v0.26.0 Part E)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


class TestManifest:
    def test_minimal_manifest(self):
        from soup_cli.cans.schema import Manifest

        m = Manifest(
            can_format_version=1, name="my-recipe",
            author="alice", created_at="2026-04-20",
            base_hash="abc",
        )
        assert m.name == "my-recipe"
        assert m.can_format_version == 1

    def test_unknown_format_version_rejected(self):
        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValueError, match="version"):
            Manifest(
                can_format_version=99, name="x",
                author="a", created_at="2026-04-20", base_hash="abc",
            )

    def test_data_ref_http_rejected(self):
        """data_ref URLs must be HTTPS."""
        from soup_cli.cans.schema import DataRef

        with pytest.raises(ValueError, match="https"):
            DataRef(kind="url", url="http://evil.example/data.jsonl")

    def test_data_ref_https_accepted(self):
        from soup_cli.cans.schema import DataRef

        ref = DataRef(kind="url", url="https://hub.example/data.jsonl")
        assert ref.url.startswith("https://")

    def test_data_ref_hf_dataset_validation(self):
        from soup_cli.cans.schema import DataRef

        DataRef(kind="hf", hf_dataset="alice/my-dataset")  # ok
        with pytest.raises(ValueError):
            DataRef(kind="hf", hf_dataset="a b/c")  # whitespace bad

    def test_name_validation(self):
        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValueError):
            Manifest(
                can_format_version=1, name="../evil",
                author="a", created_at="2026-04-20", base_hash="abc",
            )


# ---------------------------------------------------------------------------
# Pack / Unpack roundtrip
# ---------------------------------------------------------------------------


class TestPackUnpack:
    def _fake_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(
            name="recipe", tag="v1", base_model="llama-3.1-8b",
            task="sft", run_id=None,
            config={"base": "llama-3.1-8b", "task": "sft",
                    "training": {"lr": 2e-5}},
            notes="demo",
        )
        store.close()
        return eid

    def test_pack_creates_tarball(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.pack import pack_entry

        eid = self._fake_entry(tmp_path, monkeypatch)
        out = tmp_path / "recipe.can"
        pack_entry(entry_id=eid, out_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_inspect_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.pack import pack_entry
        from soup_cli.cans.unpack import inspect_can

        eid = self._fake_entry(tmp_path, monkeypatch)
        out = tmp_path / "recipe.can"
        pack_entry(entry_id=eid, out_path=str(out))

        manifest = inspect_can(str(out))
        assert manifest.name == "recipe"
        # v0.33.0 bumped CAN_FORMAT_VERSION 1 -> 2; v1 still loads on inspect.
        assert manifest.can_format_version in (1, 2)

    def test_verify_valid_can(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.pack import pack_entry
        from soup_cli.cans.verify import verify_can

        eid = self._fake_entry(tmp_path, monkeypatch)
        out = tmp_path / "recipe.can"
        pack_entry(entry_id=eid, out_path=str(out))

        report = verify_can(str(out))
        assert report.manifest_ok is True
        assert report.config_ok is True

    def test_unpack_rejects_path_traversal(self, tmp_path, monkeypatch):
        """tar with ../../ entries must be rejected by filter='data'."""
        monkeypatch.chdir(tmp_path)
        # Build a malicious tar with a path-traversal entry
        import io
        import tarfile

        from soup_cli.cans.unpack import extract_can

        mal_tar = tmp_path / "evil.can"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            content = b"pwned"
            info = tarfile.TarInfo(name="../escaped.bin")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        mal_tar.write_bytes(buf.getvalue())

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        with pytest.raises((ValueError, tarfile.TarError, OSError)):
            extract_can(str(mal_tar), str(extract_dir))

    def test_pack_missing_entry_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.cans.pack import pack_entry

        with pytest.raises(ValueError, match="not found|missing"):
            pack_entry(entry_id="nonexistent",
                       out_path=str(tmp_path / "x.can"))


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


class TestFork:
    def _make_can(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.cans.pack import pack_entry
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(
            name="recipe", tag="v1", base_model="llama-3.1-8b",
            task="sft", run_id=None,
            config={"base": "llama-3.1-8b", "training": {"lr": 2e-5}},
        )
        store.close()

        out = tmp_path / "recipe.can"
        pack_entry(entry_id=eid, out_path=str(out))
        return out

    def test_fork_applies_modifications(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.pack import fork_can

        can_path = self._make_can(tmp_path, monkeypatch)
        out = tmp_path / "fork.can"
        fork_can(
            source=str(can_path), out_path=str(out),
            modifications=["training.lr=5e-5"],
        )
        assert out.exists()

        # Verify the modification took effect
        from soup_cli.cans.unpack import read_config

        cfg = read_config(str(out))
        assert cfg["training"]["lr"] == 5e-5


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCanCLI:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(
            name="demo", tag="v1", base_model="llama", task="sft",
            run_id=None, config={"base": "llama", "task": "sft"},
        )
        store.close()
        return eid

    def test_pack_cli(self, tmp_path, monkeypatch):
        eid = self._setup(tmp_path, monkeypatch)
        result = runner.invoke(app, [
            "can", "pack", "--entry-id", eid, "--out", "demo.can",
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "demo.can").exists()

    def test_inspect_cli(self, tmp_path, monkeypatch):
        eid = self._setup(tmp_path, monkeypatch)
        runner.invoke(app, [
            "can", "pack", "--entry-id", eid, "--out", "demo.can",
        ])
        result = runner.invoke(app, ["can", "inspect", "demo.can"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "demo" in result.output

    def test_verify_cli(self, tmp_path, monkeypatch):
        eid = self._setup(tmp_path, monkeypatch)
        runner.invoke(app, [
            "can", "pack", "--entry-id", eid, "--out", "demo.can",
        ])
        result = runner.invoke(app, ["can", "verify", "demo.can"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_pack_rejects_path_traversal_out(self, tmp_path, monkeypatch):
        eid = self._setup(tmp_path, monkeypatch)
        result = runner.invoke(app, [
            "can", "pack", "--entry-id", eid,
            "--out", str(tmp_path.parent / "escape.can"),
        ])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_inspect_missing_file(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        result = runner.invoke(app, ["can", "inspect", "nope.can"])
        assert result.exit_code != 0, (result.output, repr(result.exception))


class TestForkSecurity:
    def _make_can(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.cans.pack import pack_entry
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="recipe", tag="v1", base_model="llama",
                         task="sft", run_id=None,
                         config={"base": "llama", "training": {"lr": 2e-5}})
        store.close()
        out = tmp_path / "recipe.can"
        pack_entry(entry_id=eid, out_path=str(out))
        return out

    def test_fork_rejects_source_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.cans.pack import fork_can

        monkeypatch.chdir(tmp_path)
        # Place source in a sibling dir — fork must refuse
        outside = tmp_path.parent / "escape_src.can"
        outside.write_bytes(b"dummy")
        try:
            with pytest.raises(ValueError, match="outside|cwd"):
                fork_can(source=str(outside),
                         out_path=str(tmp_path / "out.can"),
                         modifications=[])
        finally:
            outside.unlink(missing_ok=True)

    def test_fork_rejects_dunder_modification(self, tmp_path, monkeypatch):
        from soup_cli.cans.pack import fork_can

        can = self._make_can(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="dunder|forbidden"):
            fork_can(source=str(can), out_path=str(tmp_path / "fork.can"),
                     modifications=["__class__.__init__=evil"])

    def test_fork_rejects_null_byte(self, tmp_path, monkeypatch):
        from soup_cli.cans.pack import fork_can

        can = self._make_can(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="null byte"):
            fork_can(source=str(can), out_path=str(tmp_path / "fork.can"),
                     modifications=["k\x00ey=1"])

    def test_fork_with_empty_modifications_preserves_config(
        self, tmp_path, monkeypatch,
    ):
        from soup_cli.cans.pack import fork_can
        from soup_cli.cans.unpack import read_config

        can = self._make_can(tmp_path, monkeypatch)
        out = tmp_path / "fork.can"
        fork_can(source=str(can), out_path=str(out), modifications=[])
        cfg = read_config(str(out))
        assert cfg["training"]["lr"] == 2e-5


class TestManifestValidation:
    def test_author_rejects_newline(self):
        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValueError, match="null|newline"):
            Manifest(
                can_format_version=1, name="x",
                author="alice\ninject", created_at="2026-04-20",
                base_hash="abc",
            )

    def test_author_length_capped(self):
        from soup_cli.cans.schema import Manifest

        with pytest.raises(ValueError):
            Manifest(
                can_format_version=1, name="x",
                author="a" * 500, created_at="2026-04-20",
                base_hash="abc",
            )

    def test_created_at_parseability(self):
        from soup_cli.cans.schema import Manifest

        Manifest(can_format_version=1, name="x", author="a",
                 created_at="2026-04-20", base_hash="abc")
        Manifest(can_format_version=1, name="x", author="a",
                 created_at="2026-04-20T12:34:56", base_hash="abc")
        with pytest.raises(ValueError, match="ISO"):
            Manifest(can_format_version=1, name="x", author="a",
                     created_at="not-a-date", base_hash="abc")


class TestVerifyEdgeCases:
    def test_verify_corrupt_tar_reports_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.verify import verify_can

        fake = tmp_path / "junk.can"
        fake.write_bytes(b"not a tar file")
        report = verify_can(str(fake))
        assert report.manifest_ok is False

    def test_verify_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.verify import verify_can

        report = verify_can(str(tmp_path / "nope.can"))
        assert report.manifest_ok is False

    def test_inspect_rejects_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cans.unpack import inspect_can

        outside = tmp_path.parent / "escape.can"
        outside.write_bytes(b"x")
        try:
            with pytest.raises(ValueError, match="outside|cwd"):
                inspect_can(str(outside))
        finally:
            outside.unlink(missing_ok=True)
