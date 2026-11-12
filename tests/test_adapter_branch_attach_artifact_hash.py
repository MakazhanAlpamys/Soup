"""#2727: ``--attach-to-registry`` records the branch pointer as it stays on disk.

``attach_branch_to_registry`` used to hash and size the pointer, then rewrite it
with ``registry_entry_id``. The ``branch_ref`` artifact therefore described bytes
that no longer existed, and ``soup registry show`` / ``soup card`` printed a
SHA-256 and size that never matched the file.
"""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.hashing import hash_file
from soup_cli.registry.store import RegistryStore

runner = CliRunner()


@pytest.fixture()
def entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    (tmp_path / "branches").mkdir()
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "branches"))
    (tmp_path / "soup.yaml").write_text("base: org/base\ntask: sft\n", encoding="utf-8")
    with RegistryStore() as store:
        return store.push(name="model", tag="v1", base_model="org/base", task="sft", run_id=None,
                          config={"lr": 0.0001})


def _branch_ref(entry_id):
    with RegistryStore() as store:
        rows = [a for a in store.get_artifacts(entry_id) if a["kind"] == "branch_ref"]
    assert len(rows) == 1, rows
    return rows[0]


def test_the_cli_records_the_hash_and_size_of_the_linked_pointer(entry, tmp_path):
    result = runner.invoke(app, ["adapters", "branch", "v18", "-c", "soup.yaml",
                                 "--base", "org/base", "--attach-to-registry", "model:v1"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    pointer = tmp_path / "branches" / "v18.json"
    assert json.loads(pointer.read_text(encoding="utf-8"))["registry_entry_id"] == entry
    row = _branch_ref(entry)
    assert row["sha256"] == hash_file(str(pointer))
    assert row["size_bytes"] == pointer.stat().st_size


def test_a_failed_artifact_write_leaves_the_pointer_unlinked(entry, tmp_path, monkeypatch):
    from soup_cli.utils import adapter_branch

    assert runner.invoke(app, ["adapters", "branch", "v19", "-c", "soup.yaml",
                               "--base", "org/base"]).exit_code == 0
    pointer = tmp_path / "branches" / "v19.json"
    before = pointer.read_bytes()

    def boom(self, **_kwargs):
        raise ValueError("artifact refused")

    monkeypatch.setattr(RegistryStore, "add_artifact", boom)
    with pytest.raises(ValueError, match="artifact refused"):
        adapter_branch.attach_branch_to_registry("v19", "model:v1")
    assert pointer.read_bytes() == before
    with RegistryStore() as store:
        assert store.get_artifacts(entry) == []
