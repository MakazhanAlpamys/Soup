"""YAML documents are refused when their expanded size is too large."""

from __future__ import annotations

import io
import tarfile
import time

import pytest
import yaml

from soup_cli.utils.yaml_limits import (
    MAX_YAML_EXPANDED_NODES,
    check_yaml_expanded_size,
    expanded_node_count,
)


def _alias_doc(levels: int) -> str:
    doc = "_type: t\npredicateType: p\nl0: &a0 [" + ", ".join(["x"] * 10) + "]\n"
    for i in range(1, levels + 1):
        doc += f"l{i}: &a{i} [" + ", ".join([f"*a{i - 1}"] * 10) + "]\n"
    return doc


def _manifest_with_attestation(body: str) -> str:
    indented = body.rstrip("\n").replace("\n", "\n    ")
    return (
        "can_format_version: 1\nname: n\nauthor: a\ncreated_at: '2026-01-01'\n"
        "base_hash: x\nattestations:\n  - " + indented + "\n"
    )


def _write_can(path: str, name: str, text: str) -> None:
    payload = text.encode()
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))


def test_plain_document_counts():
    assert expanded_node_count({"a": [1, 2], "b": "c"}) == 1 + (1 + 3) + (1 + 1)


def test_scalar_counts_one():
    assert expanded_node_count("x") == 1
    assert expanded_node_count(None) == 1


def test_shared_alias_counted_by_expansion_not_identity():
    data = yaml.safe_load(_alias_doc(3))
    assert expanded_node_count(data, limit=10**9) > 10_000


def test_shared_alias_count_is_exact():
    data = yaml.safe_load(_alias_doc(2))
    # l0 = 1 + 10, l1 = 1 + 10 * 11, l2 = 1 + 10 * 111; root = 1 + 5 keys
    # + two scalar values.
    assert expanded_node_count(data) == 1 + 5 + 2 + 11 + 111 + 1111


def test_limit_boundary():
    data = [0] * 9  # 10 nodes
    assert expanded_node_count(data, limit=10) == 10
    assert expanded_node_count(data, limit=9) == 10


def test_deep_alias_document_is_counted_quickly_and_refused():
    data = yaml.safe_load(_alias_doc(9))  # would be ~58 GB as JSON
    start = time.perf_counter()
    with pytest.raises(ValueError, match="expands to more than"):
        check_yaml_expanded_size(data, "attestation")
    assert time.perf_counter() - start < 2.0


def test_recursive_alias_refused():
    data = yaml.safe_load("a: &x [1, *x]")
    assert expanded_node_count(data) > MAX_YAML_EXPANDED_NODES


def test_deep_nesting_is_over_limit_without_recursion_error():
    data: list = []
    for _ in range(5000):
        data = [data]
    assert expanded_node_count(data) > MAX_YAML_EXPANDED_NODES


def test_attestation_refused_before_serialising(monkeypatch):
    from soup_cli.cans import schema

    def _no_dumps(*args, **kwargs):
        raise AssertionError("json.dumps must not run on an over-limit statement")

    monkeypatch.setattr(schema.json, "dumps", _no_dumps)
    with pytest.raises(ValueError, match="expands to more than"):
        schema.validate_attestation_statement(yaml.safe_load(_alias_doc(9)))


def test_manifest_fixture_shape():
    data = yaml.safe_load(_manifest_with_attestation(_alias_doc(1)))
    assert isinstance(data["attestations"], list)
    assert len(data["attestations"]) == 1
    assert data["attestations"][0]["_type"] == "t"
    assert data["attestations"][0]["l1"][0] == ["x"] * 10


def test_manifest_with_alias_attestation_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import inspect_can

    _write_can("a.can", "manifest.yaml", _manifest_with_attestation(_alias_doc(9)))
    start = time.perf_counter()
    with pytest.raises(ValueError, match="expands to more than"):
        inspect_can("a.can")
    assert time.perf_counter() - start < 5.0


def test_config_with_aliases_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import read_config

    _write_can("c.can", "config.yaml", _alias_doc(9))
    with pytest.raises(ValueError, match="config.yaml expands to more than"):
        read_config("c.can")


def test_small_alias_manifest_still_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import inspect_can

    _write_can("s.can", "manifest.yaml", _manifest_with_attestation(_alias_doc(2)))
    manifest = inspect_can("s.can")
    assert manifest.attestations[0]["predicateType"] == "p"
