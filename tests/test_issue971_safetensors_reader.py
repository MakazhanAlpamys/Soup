"""#971 — parse a safetensors header without mapping the file."""

import json
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.safetensors_reader import TensorRange, read_header  # noqa: E402


def _mixed_shard(tmp_path: Path) -> str:
    """An NF4-shaped shard: packed nibbles are uint8, statistics are float32.

    ``::nested_offset`` is a SCALAR, and it is here because the shape production
    actually has is what this fixture is for. Under double quantisation — the
    default — every quantised weight carries one, 7 of the 30 tensors in a real
    decoder-layer shard. Without it the fixture was "NF4-shaped" only in dtype,
    and ``read_into`` shipped unable to fill a 0-dim destination at all.
    """
    path = tmp_path / "layer_000.safetensors"
    save_file(
        {
            "self_attn.q_proj.weight": torch.randint(0, 255, (64, 32), dtype=torch.uint8),
            "self_attn.q_proj.weight::absmax": torch.rand(16, dtype=torch.float32),
            "self_attn.q_proj.weight::nested_offset": torch.tensor(
                0.3125, dtype=torch.float32
            ),
            "input_layernorm.weight": torch.rand(64, dtype=torch.bfloat16),
        },
        str(path),
    )
    return str(path)


class TestHeaderMatchesSafetensors:
    def test_names_dtypes_and_shapes_match_safe_open(self, tmp_path):
        path = _mixed_shard(tmp_path)
        ours = read_header(path)
        with safe_open(path, framework="pt") as handle:
            assert set(ours) == set(handle.keys())
            for name in handle.keys():
                sliced = handle.get_slice(name)
                assert ours[name].shape == tuple(int(d) for d in sliced.get_shape())

    def test_dtypes_use_soups_spelling(self, tmp_path):
        ours = read_header(_mixed_shard(tmp_path))
        assert ours["self_attn.q_proj.weight"].dtype == "uint8"
        assert ours["self_attn.q_proj.weight::absmax"].dtype == "float32"
        assert ours["input_layernorm.weight"].dtype == "bfloat16"

    def test_byte_ranges_address_the_real_tensor_bytes(self, tmp_path):
        path = _mixed_shard(tmp_path)
        ours = read_header(path)
        with safe_open(path, framework="pt") as handle:
            expected = handle.get_tensor("self_attn.q_proj.weight")
        entry = ours["self_attn.q_proj.weight"]
        with open(path, "rb") as fh:
            fh.seek(entry.start)
            raw = fh.read(entry.end - entry.start)
        assert len(raw) == expected.numel() * expected.element_size()
        assert raw == bytes(expected.reshape(-1).view(torch.uint8).numpy())

    def test_metadata_key_is_not_a_tensor(self, tmp_path):
        path = tmp_path / "meta.safetensors"
        save_file({"w": torch.zeros(4)}, str(path), metadata={"format": "pt"})
        assert set(read_header(str(path))) == {"w"}


class TestHeaderRefusesRatherThanGuesses:
    def test_a_truncated_file_is_refused(self, tmp_path):
        path = tmp_path / "short.safetensors"
        path.write_bytes(struct.pack("<Q", 4096) + b"{}")
        with pytest.raises(ValueError, match="header is truncated"):
            read_header(str(path))

    def test_an_oversized_header_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import _MAX_HEADER_BYTES

        path = tmp_path / "huge.safetensors"
        path.write_bytes(struct.pack("<Q", _MAX_HEADER_BYTES + 1))
        with pytest.raises(ValueError, match="header claims"):
            read_header(str(path))

    def test_an_unsupported_dtype_is_refused_by_name(self, tmp_path):
        path = tmp_path / "odd.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00\x00")
        with pytest.raises(ValueError, match="F8_E4M3"):
            read_header(str(path))

    def test_offsets_outside_the_file_are_refused(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 4)
        with pytest.raises(ValueError, match="past the end of the file"):
            read_header(str(path))

    def test_a_shape_that_disagrees_with_its_byte_range_is_refused(self, tmp_path):
        path = tmp_path / "mismatch.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
        with pytest.raises(ValueError, match="byte range"):
            read_header(str(path))

    def test_a_negative_start_offset_is_refused(self, tmp_path):
        """A negative data_offsets[0] would otherwise address bytes inside the
        JSON header itself rather than tensor data — reproduced upstream as
        TensorRange(start=66, end=70) reading back the tail of the header
        text with no exception raised."""
        path = tmp_path / "negative_start.safetensors"
        body = json.dumps(
            {"w": {"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body)
        with pytest.raises(ValueError, match="tensor-data region"):
            read_header(str(path))

    @staticmethod
    def _shard_with_shape(tmp_path: Path, entry: dict) -> str:
        """A one-tensor shard whose header entry is exactly ``entry``.

        Everything except ``shape`` is well-formed and the tensor bytes are
        present, so the only refusal the file can earn is the shape one.
        """
        path = tmp_path / "shape.safetensors"
        body = json.dumps({"w": entry}).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 4)
        return str(path)

    def test_a_missing_shape_is_refused_rather_than_read_as_a_scalar(self, tmp_path):
        """No ``shape`` key at all. A default of ``[]`` would silently call this
        a 0-dim scalar; the byte-range check then only disagrees by luck."""
        path = self._shard_with_shape(tmp_path, {"dtype": "F32", "data_offsets": [0, 4]})
        with pytest.raises(ValueError, match="tensor 'w' has no valid shape"):
            read_header(path)

    def test_a_null_shape_is_refused_by_name(self, tmp_path):
        """``"shape": null`` — ``dict.get(key, default)`` does NOT substitute the
        default for a stored ``None``, so this is the case a default cannot save:
        without the type check it raises a bare TypeError naming neither the file
        nor the tensor."""
        path = self._shard_with_shape(
            tmp_path, {"dtype": "F32", "shape": None, "data_offsets": [0, 4]}
        )
        with pytest.raises(ValueError, match="tensor 'w' has no valid shape"):
            read_header(path)

    def test_a_non_list_shape_is_refused_by_name(self, tmp_path):
        path = self._shard_with_shape(
            tmp_path, {"dtype": "F32", "shape": 4, "data_offsets": [0, 4]}
        )
        with pytest.raises(ValueError, match="tensor 'w' has no valid shape"):
            read_header(path)

    def test_the_refusal_names_the_file_as_well_as_the_tensor(self, tmp_path):
        path = self._shard_with_shape(
            tmp_path, {"dtype": "F32", "shape": None, "data_offsets": [0, 4]}
        )
        with pytest.raises(ValueError) as excinfo:
            read_header(path)
        assert "shape.safetensors" in str(excinfo.value)


class TestTheIdentityOfTheFileTheRangesCameOff:
    """``read_header`` closes the file; a caller that keeps the ranges for a
    whole run needs to be able to prove, at every later open, that it is still
    addressing the file it parsed."""

    def test_it_agrees_with_a_plain_stat_of_the_same_file(self, tmp_path):
        import os

        from soup_cli.utils.safetensors_reader import read_header_with_identity

        path = _mixed_shard(tmp_path)
        entries, identity = read_header_with_identity(path)
        st = os.stat(path)
        assert identity.size == st.st_size
        assert identity.mtime_ns == st.st_mtime_ns
        assert identity.ino == st.st_ino
        assert identity.dev == st.st_dev
        # And the header half is unchanged: `read_header` is this function.
        assert entries == read_header(path)

    def test_a_same_size_rewrite_produces_a_different_identity(self, tmp_path):
        """Size alone is not identity — this is the case it misses, and the one
        that reads at stale offsets with no error anywhere."""
        import os

        from soup_cli.utils.safetensors_reader import read_header_with_identity

        path = Path(_mixed_shard(tmp_path))
        _, before = read_header_with_identity(str(path))
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        os.utime(path, ns=(before.mtime_ns + 10**9, before.mtime_ns + 10**9))
        _, after = read_header_with_identity(str(path))
        assert after.size == before.size, "the rewrite must not change the size"
        assert after != before

    def test_it_is_frozen(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_header_with_identity

        _, identity = read_header_with_identity(_mixed_shard(tmp_path))
        with pytest.raises(Exception):
            identity.size = 1


def test_tensor_range_is_frozen():
    entry = TensorRange(name="w", dtype="float32", shape=(2,), start=0, end=8)
    with pytest.raises(Exception):
        entry.start = 1


class TestReadIntoFillsPreallocatedTensors:
    def _entry_and_expected(self, tmp_path, name):
        from safetensors import safe_open

        path = _mixed_shard(tmp_path)
        with safe_open(path, framework="pt") as handle:
            expected = handle.get_tensor(name).clone()
        return path, read_header(path)[name], expected

    @pytest.mark.parametrize(
        "name",
        [
            "self_attn.q_proj.weight",
            "self_attn.q_proj.weight::absmax",
            "input_layernorm.weight",
        ],
    )
    def test_bytes_match_safetensors_for_every_dtype(self, tmp_path, name):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, expected = self._entry_and_expected(tmp_path, name)
        dst = torch.empty(entry.shape, dtype=getattr(torch, entry.dtype), device="cpu")
        with open(path, "rb") as handle:
            read_into(handle, entry, dst)
        assert dst.dtype == expected.dtype
        assert torch.equal(dst.view(torch.uint8), expected.view(torch.uint8))

    def test_a_tensor_of_the_wrong_size_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "input_layernorm.weight")
        dst = torch.empty((entry.shape[0] + 1,), dtype=torch.bfloat16)
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="destination holds"):
                read_into(handle, entry, dst)

    def test_a_non_contiguous_destination_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "self_attn.q_proj.weight")
        dst = torch.empty((entry.shape[0], entry.shape[1] * 2), dtype=torch.uint8)[:, ::2]
        assert not dst.is_contiguous()
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="contiguous"):
                read_into(handle, entry, dst)

    def test_a_non_cpu_destination_is_refused(self, tmp_path):
        """Pins the CPU-device guard. ``device="meta"`` needs no GPU: a meta
        tensor carries real shape/stride/device metadata with no storage, so
        the contiguity check ahead of it still passes and the device check is
        what actually fires."""
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "self_attn.q_proj.weight")
        dst = torch.empty(entry.shape, dtype=torch.uint8, device="meta")
        assert dst.is_contiguous()
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="must live on the CPU, got meta"):
                read_into(handle, entry, dst)

    def test_a_truncated_file_raises_and_leaves_the_destination_partially_filled(
        self, tmp_path
    ):
        """read_into's failure contract: the destination is left undefined,
        not rolled back or zeroed. A short read writes whatever prefix bytes
        DID arrive and leaves the rest exactly as the caller left it — pinned
        here with a sentinel (not zero) so the assertion cannot pass by
        coincidence with an already-zeroed buffer."""
        from soup_cli.utils.safetensors_reader import read_into

        path, entry, _ = self._entry_and_expected(tmp_path, "self_attn.q_proj.weight")
        on_disk = Path(path).read_bytes()
        surviving_bytes = 8
        truncated = tmp_path / "cut.safetensors"
        truncated.write_bytes(on_disk[: entry.start + surviving_bytes])
        sentinel = 0xAB
        dst = torch.full(entry.shape, sentinel, dtype=torch.uint8)
        with open(truncated, "rb") as handle:
            with pytest.raises(OSError, match="short read"):
                read_into(handle, entry, dst)
        flat = dst.view(torch.uint8).reshape(-1)
        written_prefix = on_disk[entry.start : entry.start + surviving_bytes]
        assert bytes(flat[:surviving_bytes].numpy()) == written_prefix
        untouched_tail = flat[surviving_bytes:]
        assert torch.equal(untouched_tail, torch.full_like(untouched_tail, sentinel))


class TestScalarTensorsAreReadable:
    """NF4 double quantisation stores a 0-dim ``::nested_offset`` per quantised
    weight, and ``read_into`` could not fill one: torch refuses a dtype-``view``
    on a 0-dimensional tensor, so ``tensor.view(torch.uint8)`` raised
    "self.dim() cannot be 0 to view Float as Byte". Nothing caught it because
    this file's fixture and the source's had uint8 and float32 but no scalar —
    dtype-shaped, not shape-shaped. It reached a user as a crash on the FIRST
    layer of any 4-bit disk-tier run.
    """

    _NAME = "self_attn.q_proj.weight::nested_offset"

    def test_the_header_reports_a_scalar_as_one_itemsize_at_shape_empty(self, tmp_path):
        path = _mixed_shard(tmp_path)
        entry = read_header(path)[self._NAME]
        assert entry.shape == ()
        assert entry.dtype == "float32"
        # A scalar is one element, so its range is exactly one itemsize — the
        # arithmetic `prod(()) == 1` has to hold for the size check in
        # read_into to agree with a 0-dim destination's numel().
        assert entry.nbytes == 4
        assert entry.end - entry.start == 4

    def test_safetensors_agrees_that_it_is_a_scalar(self, tmp_path):
        """Control: the fixture really does store a 0-dim tensor, rather than a
        1-element vector that would never exercise the defect."""
        path = _mixed_shard(tmp_path)
        with safe_open(path, framework="pt") as handle:
            assert handle.get_tensor(self._NAME).dim() == 0
        assert read_header(path)[self._NAME].shape == ()

    def test_read_into_fills_a_zero_dim_destination(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_into

        path = _mixed_shard(tmp_path)
        entry = read_header(path)[self._NAME]
        with safe_open(path, framework="pt") as handle:
            expected = handle.get_tensor(self._NAME).clone()

        dst = torch.empty((), dtype=torch.float32, device="cpu")
        assert dst.dim() == 0, "the destination must be 0-dim or this proves nothing"
        with open(path, "rb") as handle:
            read_into(handle, entry, dst)
        assert dst.dim() == 0, "reading must not reshape the caller's tensor"
        assert torch.equal(dst, expected)
        assert float(dst) == pytest.approx(0.3125)

    def test_the_read_writes_through_rather_than_into_a_copy(self, tmp_path):
        """The reshape must yield a VIEW. If it ever returned a copy the read
        would succeed, report the right byte count, and leave the caller's
        tensor untouched — the buffer pool would then stage stale data with no
        error anywhere."""
        from soup_cli.utils.safetensors_reader import read_into

        path = _mixed_shard(tmp_path)
        entry = read_header(path)[self._NAME]
        dst = torch.zeros((), dtype=torch.float32, device="cpu")
        before = dst.data_ptr()
        with open(path, "rb") as handle:
            read_into(handle, entry, dst)
        assert dst.data_ptr() == before, "the destination was reallocated"
        assert float(dst) != 0.0, "the bytes never reached the caller's storage"
