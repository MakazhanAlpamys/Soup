"""Reading .can archives is bounded by member size and count."""

from __future__ import annotations

import copy
import io
import tarfile
import tracemalloc

import pytest


def _write_can(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


def test_oversized_manifest_refused_without_reading_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    payload = b"#" + b"x" * (64 << 20)
    _write_can("big.can", [("manifest.yaml", payload)])
    del payload
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="manifest.yaml"):
            unpack.inspect_can("big.can")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < unpack.MAX_MANIFEST_BYTES * 2, peak


def test_oversized_config_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    _write_can("c.can", [("config.yaml", b"a: " + b"x" * (unpack.MAX_CONFIG_BYTES + 10))])
    with pytest.raises(ValueError, match="config.yaml"):
        unpack.read_config("c.can")


def test_config_at_the_cap_is_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    monkeypatch.setattr(unpack, "MAX_CONFIG_BYTES", 64)
    payload = b"a: " + b"x" * 61
    assert len(payload) == 64
    _write_can("c.can", [("config.yaml", payload)])
    assert unpack.read_config("c.can") == {"a": "x" * 61}


def test_header_size_is_not_trusted(tmp_path, monkeypatch):
    """A member whose header under-reports its size is still capped by the read.

    tarfile itself never yields more bytes than the header states, so the
    under-reporting header is simulated: ``getmember`` hands back a copy whose
    ``size`` is tiny while ``extractfile`` still streams the real payload.
    That isolates the read cap from the header check. The refusal alone
    cannot tell a bounded read from a read of the whole member followed by a
    length check, so the peak allocation is asserted too.
    """
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    monkeypatch.setattr(unpack, "MAX_CONFIG_BYTES", 1024)
    payload = b"a: " + b"x" * (32 << 20)
    _write_can("c.can", [("config.yaml", payload)])
    del payload

    real_getmember = tarfile.TarFile.getmember
    real_extractfile = tarfile.TarFile.extractfile

    def _small_header(self, name):
        member = copy.copy(real_getmember(self, name))
        member.size = 3
        return member

    def _full_payload(self, member):
        name = member if isinstance(member, str) else member.name
        return real_extractfile(self, real_getmember(self, name))

    monkeypatch.setattr(tarfile.TarFile, "getmember", _small_header)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", _full_payload)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="config.yaml"):
            unpack.read_config("c.can")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 8 << 20, peak


def test_non_utf8_member_names_the_member(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    _write_can("u.can", [("config.yaml", b"a: \xff\xfe")])
    with pytest.raises(ValueError, match="config.yaml"):
        unpack.read_config("u.can")


def test_non_regular_manifest_member_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    with tarfile.open("l.can", "w:gz") as tf:
        info = tarfile.TarInfo("manifest.yaml")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ValueError, match="manifest.yaml"):
        unpack.inspect_can("l.can")


def test_extract_refuses_too_many_members(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    monkeypatch.setattr(unpack, "MAX_EXTRACT_MEMBERS", 5)
    _write_can("m.can", [(f"f{i}.txt", b"x") for i in range(6)])
    with pytest.raises(ValueError, match="members"):
        unpack.extract_can("m.can", str(tmp_path / "out"))
    assert not list((tmp_path / "out").glob("f*.txt"))


def test_extract_refuses_too_many_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    monkeypatch.setattr(unpack, "MAX_EXTRACT_BYTES", 100)
    _write_can("b.can", [("a.txt", b"x" * 60), ("b.txt", b"y" * 60)])
    with pytest.raises(ValueError, match="bytes"):
        unpack.extract_can("b.can", str(tmp_path / "out"))
    assert not (tmp_path / "out" / "a.txt").exists()


def test_extract_at_the_limits_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cans import unpack

    monkeypatch.setattr(unpack, "MAX_EXTRACT_MEMBERS", 2)
    monkeypatch.setattr(unpack, "MAX_EXTRACT_BYTES", 120)
    _write_can("b.can", [("a.txt", b"x" * 60), ("b.txt", b"y" * 60)])
    dest = unpack.extract_can("b.can", str(tmp_path / "out"))
    assert (dest / "a.txt").read_bytes() == b"x" * 60
    assert (dest / "b.txt").read_bytes() == b"y" * 60


def test_limits_cover_what_pack_can_write():
    from soup_cli.cans import pack, unpack

    assert unpack.MAX_EXTRACT_BYTES >= 10 * pack._MAX_CAN_SIZE_BYTES
    assert unpack.MAX_CONFIG_BYTES <= unpack.MAX_MANIFEST_BYTES <= unpack.MAX_EXTRACT_BYTES


def test_packed_can_round_trips_under_the_caps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    from soup_cli.cans import unpack
    from soup_cli.cans.pack import fork_can, pack_entry
    from soup_cli.registry.store import RegistryStore

    store = RegistryStore()
    try:
        eid = store.push(
            name="recipe", tag="v1", base_model="llama-3.1-8b",
            task="sft", run_id=None,
            config={"base": "llama-3.1-8b", "task": "sft", "training": {"lr": 2e-5}},
            notes="demo",
        )
    finally:
        store.close()
    attestation = {"_type": "https://in-toto.io/Statement/v1", "predicateType": "p"}
    out = pack_entry(
        entry_id=eid, out_path=str(tmp_path / "recipe.can"), attestations=[attestation],
    )

    manifest = unpack.inspect_can(str(out))
    assert manifest.name == "recipe"
    assert manifest.attestations == [attestation]
    assert unpack.read_config(str(out))["training"] == {"lr": 2e-5}
    dest = unpack.extract_can(str(out), str(tmp_path / "out"))
    assert {p.name for p in dest.iterdir()} == {
        "manifest.yaml", "config.yaml", "data_ref.yaml", "recipe.md",
    }

    forked = fork_can(
        source=str(out), out_path=str(tmp_path / "fork.can"),
        modifications=["training.lr=5e-5"],
    )
    assert unpack.inspect_can(str(forked)).name == "recipe-fork"
    assert unpack.read_config(str(forked))["training"] == {"lr": 5e-5}
