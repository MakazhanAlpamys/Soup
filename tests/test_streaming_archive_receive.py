"""Fault-injected disconnects must resume exact bytes rather than discard paid runs."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def receiver():
    path = Path(__file__).parents[1] / "benchmarks/harness/streaming_residency_controls/receive.py"
    spec = importlib.util.spec_from_file_location("cache_receive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.receive


@pytest.mark.parametrize("exception", [False, True])
def test_repeated_disconnects_resume_exact_offset(tmp_path, exception):
    raw = bytes(range(251)) * 1000
    path, offsets = tmp_path / "archive.tar", []

    def fetch(offset, sink):
        offsets.append(offset)
        length = [37, 5000, 0, 100001, len(raw)][len(offsets) - 1]
        sink.write(raw[offset:offset + length])
        if len(offsets) < 5:
            if exception:
                raise ConnectionResetError("injected disconnect")
            return 255
        return 0

    proof = receiver()(path, len(raw), hashlib.sha256(raw).hexdigest(), fetch)
    assert path.read_bytes() == raw
    assert proof["sha256_verified"]
    assert offsets == [0, 37, 5037, 5037, 105038]


def test_existing_partial_prefix_is_preserved(tmp_path):
    raw = b"all original bytes including tensor operands" * 100
    path = tmp_path / "archive.tar"
    path.write_bytes(raw[:123])

    def fetch(offset, sink):
        assert offset == 123
        sink.write(raw[offset:])
        return 0

    proof = receiver()(path, len(raw), hashlib.sha256(raw).hexdigest(), fetch)
    assert proof["attempt_offsets"] == [123]


def test_disconnect_after_complete_payload_is_accepted_only_by_hash(tmp_path):
    raw = b"authenticated complete payload"

    def fetch(_offset, sink):
        sink.write(raw)
        return 255

    proof = receiver()(tmp_path / "archive", len(raw), hashlib.sha256(raw).hexdigest(), fetch)
    assert proof["transport_codes"] == [255] and proof["sha256_verified"]


def test_corrupted_complete_archive_cannot_be_acknowledged(tmp_path):
    raw = b"exact tensor values"

    def fetch(_offset, sink):
        sink.write(b"x" * len(raw))
        return 0

    with pytest.raises(ValueError, match="SHA mismatch"):
        receiver()(tmp_path / "archive", len(raw), hashlib.sha256(raw).hexdigest(), fetch)


def test_no_progress_is_bounded_and_partial_is_kept(tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"prefix")
    with pytest.raises(RuntimeError, match="exhausted"):
        receiver()(path, 100, "0" * 64, lambda _offset, _sink: 255, attempts=3)
    assert path.read_bytes() == b"prefix"


def test_oversized_prefix_is_rejected_without_network(tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"too long")
    with pytest.raises(ValueError, match="exceeds"):
        receiver()(path, 1, "0" * 64, lambda *_args: pytest.fail("must not fetch"))
