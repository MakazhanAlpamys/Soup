"""Tests for v0.60.0 Part D — Namespace-pin verification (anti-AI-Jacking).

Coverage:
- ``NamespacePin`` frozen dataclass
- ``NamespacePinStore`` SQLite-backed cache (CRUD + verify_namespace)
- ``record_repo_first_seen`` + ``verify_namespace`` decisions
- Author change / created_at jump detection
- ``--allow-namespace-shift <author>`` opt-in path
- CLI smoke (advisory only — no top-level CLI in v0.60.0)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


class TestNamespacePin:
    def test_imports(self):
        from soup_cli.utils.namespace_pin import (
            NamespacePin,
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        assert callable(record_repo_first_seen)
        assert callable(verify_namespace)
        assert dataclasses.is_dataclass(NamespacePin)
        assert hasattr(NamespacePinStore, "get")
        assert hasattr(NamespacePinStore, "put")

    def test_pin_frozen(self):
        from soup_cli.utils.namespace_pin import NamespacePin

        pin = NamespacePin(
            repo_id="meta-llama/Llama-3.1-8B",
            author="meta-llama",
            created_at="2024-01-01T00:00:00+00:00",
            first_seen="2024-06-01T00:00:00+00:00",
        )
        assert pin.repo_id == "meta-llama/Llama-3.1-8B"
        with pytest.raises(dataclasses.FrozenInstanceError):
            pin.author = "attacker"  # type: ignore[misc]

    def test_pin_rejects_null_byte(self):
        from soup_cli.utils.namespace_pin import NamespacePin

        with pytest.raises(ValueError):
            NamespacePin(
                repo_id="a\x00b",
                author="x",
                created_at="2024-01-01T00:00:00+00:00",
                first_seen="2024-06-01T00:00:00+00:00",
            )

    def test_pin_rejects_empty_repo(self):
        from soup_cli.utils.namespace_pin import NamespacePin

        with pytest.raises(ValueError):
            NamespacePin(
                repo_id="",
                author="x",
                created_at="2024-01-01T00:00:00+00:00",
                first_seen="2024-06-01T00:00:00+00:00",
            )

    def test_pin_rejects_oversize_repo(self):
        from soup_cli.utils.namespace_pin import NamespacePin

        with pytest.raises(ValueError):
            NamespacePin(
                repo_id="a" * 1000,
                author="x",
                created_at="2024-01-01T00:00:00+00:00",
                first_seen="2024-06-01T00:00:00+00:00",
            )


class TestStore:
    def test_put_then_get(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePin,
            NamespacePinStore,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        pin = NamespacePin(
            repo_id="meta-llama/Llama-3.1-8B",
            author="meta-llama",
            created_at="2024-01-01T00:00:00+00:00",
            first_seen="2024-06-01T00:00:00+00:00",
        )
        store.put(pin)
        loaded = store.get("meta-llama/Llama-3.1-8B")
        assert loaded == pin

    def test_get_missing_returns_none(self, tmp_path):
        from soup_cli.utils.namespace_pin import NamespacePinStore
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        assert store.get("nobody/nothing") is None

    def test_put_idempotent(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePin,
            NamespacePinStore,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        pin = NamespacePin(
            repo_id="meta-llama/Llama-3.1-8B",
            author="meta-llama",
            created_at="2024-01-01T00:00:00+00:00",
            first_seen="2024-06-01T00:00:00+00:00",
        )
        store.put(pin)
        store.put(pin)
        assert store.get(pin.repo_id) == pin

    def test_put_does_not_overwrite_first_seen(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePin,
            NamespacePinStore,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        pin = NamespacePin(
            repo_id="r/x",
            author="alice",
            created_at="2024-01-01T00:00:00+00:00",
            first_seen="2024-06-01T00:00:00+00:00",
        )
        store.put(pin)
        # A second put with the same repo_id MUST NOT mutate first_seen.
        pin2 = NamespacePin(
            repo_id="r/x",
            author="alice",
            created_at="2024-01-01T00:00:00+00:00",
            first_seen="2025-01-01T00:00:00+00:00",
        )
        store.put(pin2)
        loaded = store.get("r/x")
        assert loaded is not None
        assert loaded.first_seen == "2024-06-01T00:00:00+00:00"

    def test_invalid_repo_id_rejected(self, tmp_path):
        from soup_cli.utils.namespace_pin import NamespacePinStore
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        with pytest.raises(ValueError):
            store.get("bad\x00id")


class TestVerify:
    def test_first_seen_records(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        pin = record_repo_first_seen(
            store,
            repo_id="meta-llama/Llama-3.1-8B",
            author="meta-llama",
            created_at="2024-01-01T00:00:00+00:00",
        )
        assert pin.author == "meta-llama"
        # Round-trip
        assert store.get(pin.repo_id) == pin

    def test_verify_namespace_clean_pass(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        record_repo_first_seen(
            store,
            repo_id="r/x",
            author="alice",
            created_at="2024-01-01T00:00:00+00:00",
        )
        report = verify_namespace(
            store,
            repo_id="r/x",
            current_author="alice",
            current_created_at="2024-01-01T00:00:00+00:00",
        )
        assert report.ok is True

    def test_verify_namespace_author_change_flagged(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        record_repo_first_seen(
            store,
            repo_id="r/x",
            author="alice",
            created_at="2024-01-01T00:00:00+00:00",
        )
        report = verify_namespace(
            store,
            repo_id="r/x",
            current_author="attacker",
            current_created_at="2024-01-01T00:00:00+00:00",
        )
        assert report.ok is False
        assert "author" in report.reason.lower()

    def test_verify_namespace_backward_created_at_flagged(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        record_repo_first_seen(
            store,
            repo_id="r/x",
            author="alice",
            created_at="2024-06-01T00:00:00+00:00",
        )
        # New created_at jumped backward — namespace was re-created.
        report = verify_namespace(
            store,
            repo_id="r/x",
            current_author="alice",
            current_created_at="2023-01-01T00:00:00+00:00",
        )
        assert report.ok is False
        assert "created" in report.reason.lower() or "backward" in report.reason.lower()

    def test_verify_namespace_unknown_repo_first_seen(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        # Unknown repo — by policy this is OK on first-seen (trust-on-first-use).
        report = verify_namespace(
            store,
            repo_id="r/x",
            current_author="alice",
            current_created_at="2024-01-01T00:00:00+00:00",
        )
        assert report.ok is True
        assert "first" in report.reason.lower()

    def test_verify_namespace_allow_shift(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        record_repo_first_seen(
            store,
            repo_id="r/x",
            author="alice",
            created_at="2024-01-01T00:00:00+00:00",
        )
        # With matching --allow-namespace-shift=attacker, the shift is allowed
        # and the pin is updated to the new author + created_at.
        report = verify_namespace(
            store,
            repo_id="r/x",
            current_author="attacker",
            current_created_at="2025-01-01T00:00:00+00:00",
            allow_namespace_shift="attacker",
        )
        assert report.ok is True
        # Allow opt-in updates the recorded pin.
        updated = store.get("r/x")
        assert updated is not None
        assert updated.author == "attacker"

    def test_verify_namespace_allow_shift_must_match(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        record_repo_first_seen(
            store,
            repo_id="r/x",
            author="alice",
            created_at="2024-01-01T00:00:00+00:00",
        )
        # Opt-in must name the new author. Mismatch keeps the gate closed.
        report = verify_namespace(
            store,
            repo_id="r/x",
            current_author="attacker",
            current_created_at="2025-01-01T00:00:00+00:00",
            allow_namespace_shift="otherperson",
        )
        assert report.ok is False

    def test_verify_namespace_rejects_bool_allow(self, tmp_path):
        """``allow_namespace_shift=True`` must NOT be a free-for-all bypass."""
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        with pytest.raises(TypeError):
            verify_namespace(
                store,
                repo_id="r/x",
                current_author="x",
                current_created_at="2024-01-01T00:00:00+00:00",
                allow_namespace_shift=True,  # type: ignore[arg-type]
            )


class TestSecurityReviewFixes:
    """Regression guards for the v0.60.0 Part D security-review fixes."""

    def test_store_rejects_outside_containment(self, tmp_path, monkeypatch):
        """Constructor must reject paths outside $HOME/$CWD/$TMPDIR.

        Defends against a caller smuggling /etc/passwd through the public
        ``NamespacePinStore(path=...)`` constructor.
        """
        # tmp_path lives under tempdir → accepted. Use a fabricated path
        # outside every allowed root.
        import sys

        from soup_cli.utils.namespace_pin import NamespacePinStore
        bogus = "Z:/no/such/dir/ns.db" if sys.platform == "win32" else "/no/such/dir/ns.db"
        with pytest.raises(ValueError):
            NamespacePinStore(bogus)

    @pytest.mark.skipif(__import__("sys").platform == "win32",
                        reason="POSIX symlink semantics")
    def test_store_rejects_symlink_at_db_path(self, tmp_path):
        import os as _os

        from soup_cli.utils.namespace_pin import NamespacePinStore

        link = tmp_path / "ns.db"
        target = tmp_path / "real_target.db"
        target.write_bytes(b"")
        _os.symlink(str(target), str(link))
        with pytest.raises(ValueError):
            NamespacePinStore(str(link))

    def test_iso_compare_uses_datetime_parser(self, tmp_path):
        """``_is_backward`` parses ISO offsets numerically, not lexicographically.

        Pin chronologically-distinguishing input: current='2024-01-01T01:00+02:00'
        (=2023-12-31T23:00 UTC) is BEFORE recorded='2024-01-01T00:00+00:00'.
        Lexicographically the current string sorts LATER ('T01' > 'T00'),
        so a string compare would miss the backward jump — but the datetime
        parser catches it.
        """
        from soup_cli.utils.namespace_pin import _is_backward

        assert _is_backward(
            "2024-01-01T01:00:00+02:00",
            "2024-01-01T00:00:00+00:00",
        ) is True
        # Plain backward case (works under either compare).
        assert _is_backward(
            "2023-01-01T00:00:00+00:00",
            "2024-06-01T00:00:00+00:00",
        ) is True
        # Forward case rejected.
        assert _is_backward(
            "2024-06-01T00:00:00+00:00",
            "2023-01-01T00:00:00+00:00",
        ) is False

    def test_author_override_case_insensitive(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )
        db = tmp_path / "ns.db"
        store = NamespacePinStore(str(db))
        record_repo_first_seen(
            store, repo_id="r/x", author="alice",
            created_at="2024-01-01T00:00:00+00:00",
        )
        # Operator types uppercase; opt-in must still accept.
        report = verify_namespace(
            store, repo_id="r/x",
            current_author="Attacker",
            current_created_at="2025-01-01T00:00:00+00:00",
            allow_namespace_shift="attacker",  # lowercase matches
        )
        assert report.ok is True


class TestSourceWiring:
    def test_module_imports(self):
        from soup_cli.utils import namespace_pin as m

        assert hasattr(m, "NamespacePin")
        assert hasattr(m, "NamespacePinStore")
        assert hasattr(m, "record_repo_first_seen")
        assert hasattr(m, "verify_namespace")

    def test_no_top_level_torch(self):
        src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "namespace_pin.py"
        text = src.read_text(encoding="utf-8")
        # no top-level torch/transformers imports
        for line in text.splitlines():
            if line.startswith("import torch") or line.startswith("from torch "):
                raise AssertionError(f"top-level torch import: {line}")
