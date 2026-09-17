"""YAML documents are refused when they expand to more nodes than source bytes."""

from __future__ import annotations

import io
import tarfile
import time

import pytest
import yaml

from soup_cli.utils.yaml_limits import (
    ALIAS_EXPANSION_SLACK,
    check_expanded_nodes,
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


def _write_can(path: str, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, text in members.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


_PLAIN_MANIFEST = (
    "can_format_version: 1\nname: n\nauthor: a\ncreated_at: '2026-01-01'\n"
    "base_hash: x\n"
)


# ---------------------------------------------------------------------------
# The counter
# ---------------------------------------------------------------------------


def test_plain_document_counts():
    assert expanded_node_count({"a": [1, 2], "b": "c"}, limit=100) == 1 + (1 + 3) + (1 + 1)


def test_scalar_counts_one():
    assert expanded_node_count("x", limit=100) == 1
    assert expanded_node_count(None, limit=100) == 1


def test_shared_alias_counted_by_expansion_not_identity():
    data = yaml.safe_load(_alias_doc(3))
    assert expanded_node_count(data, limit=10**9) > 10_000


def test_shared_alias_count_is_exact():
    data = yaml.safe_load(_alias_doc(2))
    # l0 = 1 + 10, l1 = 1 + 10 * 11, l2 = 1 + 10 * 111; root = 1 + 5 keys
    # + two scalar values.
    assert expanded_node_count(data, limit=10**6) == 1 + 5 + 2 + 11 + 111 + 1111


def test_limit_boundary():
    data = [0] * 9  # 10 nodes
    assert expanded_node_count(data, limit=10) == 10
    assert expanded_node_count(data, limit=9) == 10


def test_recursive_alias_refused():
    data = yaml.safe_load("a: &x [1, *x]")
    assert expanded_node_count(data, limit=10**6) > 10**6


def test_deep_nesting_is_over_limit_without_recursion_error():
    data: list = []
    for _ in range(5000):
        data = [data]
    assert expanded_node_count(data, limit=10**6) > 10**6


# ---------------------------------------------------------------------------
# The rule: no more nodes than source bytes (plus slack)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["{a,b,c,d}", "[[],[],[]]", "-\n-\n-\n", "a:\nb:\nc:\n", "[a,b,c]", "[{},{},{}]"],
)
def test_alias_free_documents_have_no_more_nodes_than_bytes(text):
    data = yaml.safe_load(text)
    assert expanded_node_count(data, limit=10**6) <= len(text.encode())


def test_expansion_rule_boundary():
    at_limit = [0] * (ALIAS_EXPANSION_SLACK + 100 - 1)  # SLACK + 100 nodes
    check_yaml_expanded_size(at_limit, "doc", source_bytes=100)
    over = [0] * (ALIAS_EXPANSION_SLACK + 100)
    with pytest.raises(ValueError, match=r"doc expands to more than 10100 nodes from 100 bytes"):
        check_yaml_expanded_size(over, "doc", source_bytes=100)


def test_expansion_rule_scales_with_source_bytes():
    data = [0] * 50_000
    with pytest.raises(ValueError, match="expands to more than"):
        check_yaml_expanded_size(data, "doc", source_bytes=1_000)
    check_yaml_expanded_size(data, "doc", source_bytes=40_001)


def test_check_expanded_nodes_boundary():
    check_expanded_nodes([0] * 9, "doc", limit=10)
    with pytest.raises(ValueError, match="doc expands to more than 9 nodes"):
        check_expanded_nodes([0] * 9, "doc", limit=9)


def test_deep_alias_document_is_counted_quickly_and_refused():
    doc = _alias_doc(9)
    data = yaml.safe_load(doc)  # would be ~58 GB as JSON
    start = time.perf_counter()
    with pytest.raises(ValueError, match="expands to more than"):
        check_yaml_expanded_size(data, "attestation", source_bytes=len(doc))
    assert time.perf_counter() - start < 2.0


# ---------------------------------------------------------------------------
# Attestations
# ---------------------------------------------------------------------------


def test_attestation_refused_before_serialising(monkeypatch):
    from soup_cli.cans import schema

    def _no_dumps(*args, **kwargs):
        raise AssertionError("json.dumps must not run on an over-limit statement")

    monkeypatch.setattr(schema.json, "dumps", _no_dumps)
    with pytest.raises(ValueError, match="attestation expands to more than"):
        schema.validate_attestation_statement(yaml.safe_load(_alias_doc(9)))


# ---------------------------------------------------------------------------
# Can readers
# ---------------------------------------------------------------------------


def test_manifest_fixture_shape():
    data = yaml.safe_load(_manifest_with_attestation(_alias_doc(1)))
    assert isinstance(data["attestations"], list)
    assert len(data["attestations"]) == 1
    assert data["attestations"][0]["_type"] == "t"
    assert data["attestations"][0]["l1"][0] == ["x"] * 10


def test_large_alias_free_attestation_loads(tmp_path, monkeypatch):
    """205k nodes, more than any fixed 200k cap, but no amplification."""
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import inspect_can

    count = 205_000
    body = (
        "_type: t\npredicateType: p\nitems: [" + ",".join(["0"] * count) + "]\n"
    )
    _write_can("big.can", {"manifest.yaml": _manifest_with_attestation(body)})
    manifest = inspect_can("big.can")
    assert len(manifest.attestations[0]["items"]) == count


def test_five_level_alias_manifest_refused_by_the_manifest_rule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import inspect_can

    _write_can("a.can", {"manifest.yaml": _manifest_with_attestation(_alias_doc(5))})
    with pytest.raises(ValueError, match="manifest.yaml expands to more than"):
        inspect_can("a.can")


def test_manifest_with_alias_attestation_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import inspect_can

    _write_can("a.can", {"manifest.yaml": _manifest_with_attestation(_alias_doc(9))})
    start = time.perf_counter()
    with pytest.raises(ValueError, match="expands to more than"):
        inspect_can("a.can")
    assert time.perf_counter() - start < 5.0


def test_config_with_aliases_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import read_config

    _write_can("c.can", {"config.yaml": _alias_doc(9)})
    with pytest.raises(ValueError, match="config.yaml expands to more than"):
        read_config("c.can")


def test_small_alias_manifest_still_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans.unpack import inspect_can

    _write_can("s.can", {"manifest.yaml": _manifest_with_attestation(_alias_doc(2))})
    manifest = inspect_can("s.can")
    assert manifest.attestations[0]["predicateType"] == "p"


# ---------------------------------------------------------------------------
# soup can run reads the config through the capped reader before anything else
# ---------------------------------------------------------------------------


def _spawn_recorder(monkeypatch):
    from soup_cli.cans import run as run_mod

    calls: list = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        raise AssertionError("no subprocess may start for a refused can")

    monkeypatch.setattr(run_mod.subprocess, "run", _fake_run)
    return run_mod, calls


def test_run_refuses_alias_amplified_config_before_spawning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_mod, calls = _spawn_recorder(monkeypatch)
    _write_can("r.can", {"manifest.yaml": _PLAIN_MANIFEST, "config.yaml": _alias_doc(5)})

    with pytest.raises(ValueError, match="config.yaml expands to more than"):
        run_mod.run_can("r.can", yes=True, extract_dir=str(tmp_path / "out"))
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_run_refuses_oversized_config_before_spawning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_mod, calls = _spawn_recorder(monkeypatch)
    from soup_cli.cans import unpack

    monkeypatch.setattr(unpack, "MAX_CONFIG_BYTES", 64)
    _write_can("r.can", {"manifest.yaml": _PLAIN_MANIFEST, "config.yaml": "a: " + "x" * 100})

    with pytest.raises(ValueError, match="config.yaml"):
        run_mod.run_can("r.can", yes=True, extract_dir=str(tmp_path / "out"))
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_run_refuses_can_without_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_mod, calls = _spawn_recorder(monkeypatch)
    _write_can("r.can", {"manifest.yaml": _PLAIN_MANIFEST})

    with pytest.raises(ValueError, match="no member 'config.yaml'"):
        run_mod.run_can("r.can", yes=True, extract_dir=str(tmp_path / "out"))
    assert calls == []


def test_cli_run_yes_refuses_alias_amplified_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app

    _, calls = _spawn_recorder(monkeypatch)
    _write_can("r.can", {"manifest.yaml": _PLAIN_MANIFEST, "config.yaml": _alias_doc(5)})
    result = CliRunner().invoke(app, ["can", "run", "r.can", "--yes"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "expands to more than" in " ".join(result.output.split())
    assert calls == []
