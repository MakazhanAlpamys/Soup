"""Byte-exact artifact transport regressions, independent of tensor computation."""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import random
from pathlib import Path

import pytest

from benchmarks.harness import lossless_results as codec

ROOT = Path(__file__).parents[1] / "benchmarks/harness/streaming_residency_controls"


def load(name):
    spec = importlib.util.spec_from_file_location("range_test_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RANGES = load("archive_ranges")
REMOTE = load("remote_ranges")


@pytest.mark.parametrize("short_success", [False, True])
def test_reuse_and_progress_reset_preserve_every_original_byte(tmp_path, short_success):
    source = tmp_path / "source"
    source.mkdir()
    rng = random.Random(792)
    base, new = rng.randbytes(4 * 2**20), rng.randbytes(2 * 2**20)
    (source / "base.bin").write_bytes(base)
    prior = tmp_path / "prior.tar"
    codec.pack(source, ["base.bin"], prior)
    (source / "new.bin").write_bytes(new)
    canonical = tmp_path / "canonical.tar"
    index = codec.pack(source, ["base.bin", "new.bin"], canonical)
    layout = RANGES.layout_from_archive(canonical)
    virtual_time, offsets = [0.0], []

    def transfer(batch, offset, sink, _deadline):
        virtual_time[0] += 601
        offsets.append(offset)
        request = base64.b64encode(json.dumps(batch).encode()).decode()
        complete = io.BytesIO()
        REMOTE.copy_spans(tmp_path, "canonical", request, offset, complete)
        data = complete.getvalue()
        size = min(len(data), 512 * 1024)
        sink.write(data[:size])
        if size != len(data) and not short_success:
            raise TimeoutError("useful progress before connection timeout")

    result = RANGES.rebuild(
        layout, index, tmp_path / "received", [prior], transfer, lambda: 10**6, 10**8,
        clock=lambda: virtual_time[0], sleep=lambda _seconds: None,
    )
    proof = codec.unpack(tmp_path / "received/results.tar", tmp_path / "decoded")
    assert proof["all_original_files_verified"] == 2
    assert proof["archive_sha256"] == index["archive_sha256"]
    assert (tmp_path / "decoded/base.bin").read_bytes() == base
    assert (tmp_path / "decoded/new.bin").read_bytes() == new
    assert result["encoded_object_bytes_reused"] == len(base)
    assert result["new_network_object_bytes"] == len(new)
    assert offsets == [0, 512 * 1024, 1024 * 1024, 1536 * 1024]
    assert virtual_time[0] > 600


@pytest.mark.parametrize("empty_success", [False, True])
def test_true_absence_of_progress_is_bounded(tmp_path, empty_success):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.bin").write_bytes(b"must remain byte exact")
    archive = tmp_path / "canonical.tar"
    index = codec.pack(source, ["raw.bin"], archive)
    virtual_time = [0.0]

    def no_bytes(_batch, _offset, _sink, _deadline):
        virtual_time[0] += 301
        if not empty_success:
            raise TimeoutError("no durable bytes")

    with pytest.raises(RuntimeError, match="no new archive bytes"):
        RANGES.rebuild(
            RANGES.layout_from_archive(archive), index, tmp_path / "received", [], no_bytes,
            lambda: 10**6, 10**8, clock=lambda: virtual_time[0], sleep=lambda _seconds: None,
        )
    assert virtual_time[0] == 602
    assert not (tmp_path / "received/results.tar").exists()


def test_corrupt_prior_object_is_refetched_instead_of_reused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payload = random.Random(13).randbytes(4096)
    (source / "raw.bin").write_bytes(payload)
    canonical = tmp_path / "canonical.tar"
    index = codec.pack(source, ["raw.bin"], canonical)
    layout = RANGES.layout_from_archive(canonical)
    prior = tmp_path / "corrupt-prior.tar"
    damaged = bytearray(canonical.read_bytes())
    span = next(iter(layout["objects"].values()))
    damaged[span["offset"]] ^= 1
    prior.write_bytes(damaged)
    calls = []

    def transfer(batch, offset, sink, _deadline):
        calls.append(offset)
        encoded = base64.b64encode(json.dumps(batch).encode()).decode()
        REMOTE.copy_spans(tmp_path, "canonical", encoded, offset, sink)

    result = RANGES.rebuild(
        layout, index, tmp_path / "received", [prior], transfer, lambda: 10**6, 10**8,
    )
    proof = codec.unpack(tmp_path / "received/results.tar", tmp_path / "decoded")
    assert calls == [0]
    assert result["encoded_object_bytes_reused"] == 0
    assert proof["archive_sha256"] == index["archive_sha256"]
    assert (tmp_path / "decoded/raw.bin").read_bytes() == payload


def test_remote_reader_resumes_across_span_boundaries(tmp_path):
    payload = bytes(range(256))
    (tmp_path / "sample.tar").write_bytes(payload)
    spans = [{"offset": 2, "bytes": 7}, {"offset": 20, "bytes": 5}]
    encoded = base64.b64encode(json.dumps(spans).encode()).decode()
    sink = io.BytesIO()
    REMOTE.copy_spans(tmp_path, "sample", encoded, 8, sink)
    assert sink.getvalue() == payload[21:25]
    with pytest.raises(AssertionError):
        REMOTE.copy_spans(tmp_path, "../sample", encoded, 0, io.BytesIO())
