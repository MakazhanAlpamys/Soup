"""v0.67.0 Part B — VeRA / VB-LoRA vector-bank storage format.

Tests for ``soup_cli/utils/vector_bank.py``:

- Frozen ``VectorBank`` / ``BankEntry`` dataclasses (shared random
  projection matrix + per-user MB-sized scaling vectors)
- Validation matrix on bank name, user id, scaling-vector shape
- ``estimate_bank_size`` for sizing decisions
- Stub-then-live: live serving wiring deferred to v0.67.1
- Atomic disk I/O via shared ``atomic_write_text`` (TOCTOU-safe)
"""

from __future__ import annotations

import dataclasses
import math
import os

import pytest

# -----------------------------------------------------------------------------
# Public surface
# -----------------------------------------------------------------------------


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import vector_bank

        assert hasattr(vector_bank, "VectorBank")
        assert hasattr(vector_bank, "BankEntry")
        assert hasattr(vector_bank, "validate_bank_name")
        assert hasattr(vector_bank, "validate_user_id")
        assert hasattr(vector_bank, "validate_scaling_vector")
        assert hasattr(vector_bank, "estimate_bank_size")
        assert hasattr(vector_bank, "apply_bank_to_serve")


# -----------------------------------------------------------------------------
# validate_bank_name
# -----------------------------------------------------------------------------


class TestValidateBankName:
    def test_happy(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        assert validate_bank_name("my-bank") == "my-bank"

    def test_case_insensitive(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        # Canonical: kebab-case, normalise to lowercase
        assert validate_bank_name("MY-BANK") == "my-bank"

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        with pytest.raises(ValueError):
            validate_bank_name("bad\x00name")

    def test_empty_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        with pytest.raises(ValueError):
            validate_bank_name("")

    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        with pytest.raises(ValueError):
            validate_bank_name("a" * 200)

    def test_non_string_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        with pytest.raises(TypeError):
            validate_bank_name(123)  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        with pytest.raises(TypeError):
            validate_bank_name(True)  # type: ignore[arg-type]

    def test_invalid_chars_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_bank_name

        with pytest.raises(ValueError):
            validate_bank_name("path/traversal")
        with pytest.raises(ValueError):
            validate_bank_name("..")
        with pytest.raises(ValueError):
            validate_bank_name("name with space")


# -----------------------------------------------------------------------------
# validate_user_id
# -----------------------------------------------------------------------------


class TestValidateUserId:
    def test_happy(self) -> None:
        from soup_cli.utils.vector_bank import validate_user_id

        assert validate_user_id("user-1234") == "user-1234"

    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_user_id

        with pytest.raises(ValueError):
            validate_user_id("u" * 300)

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_user_id

        with pytest.raises(ValueError):
            validate_user_id("u\x00")

    def test_non_string_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_user_id

        with pytest.raises(TypeError):
            validate_user_id(42)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# validate_scaling_vector
# -----------------------------------------------------------------------------


class TestValidateScalingVector:
    def test_happy(self) -> None:
        from soup_cli.utils.vector_bank import validate_scaling_vector

        v = validate_scaling_vector([0.1, 0.2, -0.3])
        assert v == (0.1, 0.2, -0.3)

    def test_returns_tuple(self) -> None:
        from soup_cli.utils.vector_bank import validate_scaling_vector

        v = validate_scaling_vector([1.0, 2.0])
        assert isinstance(v, tuple)

    def test_empty_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_scaling_vector

        with pytest.raises(ValueError):
            validate_scaling_vector([])

    def test_non_iterable_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_scaling_vector

        with pytest.raises(TypeError):
            validate_scaling_vector(42)  # type: ignore[arg-type]

    def test_non_finite_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_scaling_vector

        with pytest.raises(ValueError):
            validate_scaling_vector([1.0, math.nan])
        with pytest.raises(ValueError):
            validate_scaling_vector([math.inf, 1.0])

    def test_bool_in_vector_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_scaling_vector

        with pytest.raises(TypeError):
            validate_scaling_vector([True, 0.5])

    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.vector_bank import MAX_VECTOR_DIM, validate_scaling_vector

        with pytest.raises(ValueError):
            validate_scaling_vector([0.1] * (MAX_VECTOR_DIM + 1))


# -----------------------------------------------------------------------------
# BankEntry + VectorBank frozen dataclasses
# -----------------------------------------------------------------------------


class TestBankEntry:
    def test_construct(self) -> None:
        from soup_cli.utils.vector_bank import BankEntry

        entry = BankEntry(user_id="alice", scaling=(0.1, 0.2, 0.3))
        assert entry.user_id == "alice"
        assert entry.scaling == (0.1, 0.2, 0.3)

    def test_frozen(self) -> None:
        from soup_cli.utils.vector_bank import BankEntry

        entry = BankEntry(user_id="bob", scaling=(0.5,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.user_id = "eve"  # type: ignore[misc]


class TestVectorBank:
    def test_construct(self) -> None:
        from soup_cli.utils.vector_bank import BankEntry, VectorBank

        bank = VectorBank(
            name="mybank",
            base_model="meta-llama/Llama-3.1-8B",
            projection_seed=42,
            vector_dim=2,
            entries=(BankEntry(user_id="u1", scaling=(0.1, 0.2)),),
        )
        assert bank.name == "mybank"
        assert bank.vector_dim == 2

    def test_frozen(self) -> None:
        from soup_cli.utils.vector_bank import VectorBank

        bank = VectorBank(
            name="b",
            base_model="m",
            projection_seed=0,
            vector_dim=8,
            entries=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            bank.vector_dim = 99  # type: ignore[misc]

    def test_invalid_seed(self) -> None:
        from soup_cli.utils.vector_bank import VectorBank

        with pytest.raises(TypeError):
            VectorBank(
                name="b",
                base_model="m",
                projection_seed=True,  # type: ignore[arg-type]
                vector_dim=8,
                entries=(),
            )

    def test_invalid_vector_dim(self) -> None:
        from soup_cli.utils.vector_bank import VectorBank

        with pytest.raises(ValueError):
            VectorBank(
                name="b",
                base_model="m",
                projection_seed=0,
                vector_dim=0,
                entries=(),
            )

    def test_entries_must_be_tuple(self) -> None:
        from soup_cli.utils.vector_bank import VectorBank

        with pytest.raises(TypeError):
            VectorBank(
                name="b",
                base_model="m",
                projection_seed=0,
                vector_dim=8,
                entries=[],  # type: ignore[arg-type]
            )

    def test_max_entries_enforced(self) -> None:
        from soup_cli.utils.vector_bank import (
            MAX_ENTRIES_PER_BANK,
            BankEntry,
            VectorBank,
        )

        too_many = tuple(
            BankEntry(user_id=f"u{i}", scaling=(0.1,))
            for i in range(MAX_ENTRIES_PER_BANK + 1)
        )
        with pytest.raises(ValueError):
            VectorBank(
                name="b",
                base_model="m",
                projection_seed=0,
                vector_dim=1,
                entries=too_many,
            )

    def test_base_model_validation(self) -> None:
        from soup_cli.utils.vector_bank import VectorBank

        with pytest.raises(ValueError):
            VectorBank(
                name="b",
                base_model="",
                projection_seed=0,
                vector_dim=8,
                entries=(),
            )

    def test_per_entry_vector_dim_mismatch(self) -> None:
        from soup_cli.utils.vector_bank import BankEntry, VectorBank

        with pytest.raises(ValueError):
            VectorBank(
                name="b",
                base_model="m",
                projection_seed=0,
                vector_dim=4,
                entries=(BankEntry(user_id="u", scaling=(0.1, 0.2, 0.3)),),
            )


# -----------------------------------------------------------------------------
# estimate_bank_size
# -----------------------------------------------------------------------------


class TestEstimateBankSize:
    def test_basic(self) -> None:
        from soup_cli.utils.vector_bank import estimate_bank_size

        # 1000 users × 128 fp32 scaling vectors ≈ 512 KB + projection
        # matrix (128×128×4 = 64 KB)
        size_bytes = estimate_bank_size(num_users=1000, vector_dim=128)
        assert 400_000 < size_bytes < 1_000_000

    def test_zero_users(self) -> None:
        from soup_cli.utils.vector_bank import estimate_bank_size

        # Only projection matrix
        size_bytes = estimate_bank_size(num_users=0, vector_dim=128)
        assert size_bytes > 0  # projection still allocated

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.vector_bank import estimate_bank_size

        with pytest.raises(TypeError):
            estimate_bank_size(num_users=True, vector_dim=128)  # type: ignore[arg-type]

    def test_negative_rejected(self) -> None:
        from soup_cli.utils.vector_bank import estimate_bank_size

        with pytest.raises(ValueError):
            estimate_bank_size(num_users=-1, vector_dim=128)
        with pytest.raises(ValueError):
            estimate_bank_size(num_users=10, vector_dim=0)


# -----------------------------------------------------------------------------
# Atomic JSON write + read
# -----------------------------------------------------------------------------


class TestWriteLoadBank:
    def test_roundtrip(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.vector_bank import (
            BankEntry,
            VectorBank,
            load_bank,
            write_bank,
        )

        monkeypatch.chdir(tmp_path)
        bank = VectorBank(
            name="round-trip",
            base_model="meta-llama/Llama-3.1-8B",
            projection_seed=42,
            vector_dim=4,
            entries=(
                BankEntry(user_id="alice", scaling=(0.1, 0.2, 0.3, 0.4)),
                BankEntry(user_id="bob", scaling=(0.5, 0.6, 0.7, 0.8)),
            ),
        )
        path = str(tmp_path / "bank.json")
        write_bank(bank, path)
        loaded = load_bank(path)
        assert loaded.name == bank.name
        assert loaded.entries == bank.entries

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.vector_bank import VectorBank, write_bank

        cwd = tmp_path / "work"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        bank = VectorBank(
            name="b",
            base_model="m",
            projection_seed=0,
            vector_dim=1,
            entries=(),
        )
        # Path outside cwd
        with pytest.raises(ValueError):
            write_bank(bank, str(tmp_path / "outside.json"))

    def test_load_missing_file(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.vector_bank import load_bank

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_bank(str(tmp_path / "missing.json"))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink test")
    def test_load_symlink_rejected(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.vector_bank import load_bank

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.json"
        real.write_text('{"name": "x"}', encoding="utf-8")
        sym = tmp_path / "sym.json"
        os.symlink(real, sym)
        with pytest.raises(ValueError):
            load_bank(str(sym))


# -----------------------------------------------------------------------------
# Live-serve stub
# -----------------------------------------------------------------------------


class TestApplyBankToServe:
    def test_deferred_v0_67_1(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.vector_bank import VectorBank, apply_bank_to_serve

        bank = VectorBank(
            name="b",
            base_model="m",
            projection_seed=0,
            vector_dim=1,
            entries=(),
        )
        with pytest.raises(NotImplementedError) as exc_info:
            apply_bank_to_serve(bank)
        assert "v0.67.1" in str(exc_info.value)

    def test_non_bank_rejected(self) -> None:
        from soup_cli.utils.vector_bank import apply_bank_to_serve

        with pytest.raises(TypeError):
            apply_bank_to_serve("not-a-bank")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Source-grep regression guards
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "vector_bank.py").read_text(
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
        src = (root / "src" / "soup_cli" / "utils" / "vector_bank.py").read_text(
            encoding="utf-8"
        )
        assert "atomic_write_text" in src
