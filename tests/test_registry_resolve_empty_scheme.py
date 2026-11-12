"""#2723: a bare ``registry://`` resolves to nothing, not to every entry.

``RegistryStore.resolve`` checked for an empty reference before stripping the
``registry://`` scheme, so ``registry://`` reached the prefix lookup as ``""``
(``LIKE '%'``). With one entry it resolved to that entry, and
``soup registry delete registry:// --yes`` deleted it.
"""

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture()
def one_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    with RegistryStore() as store:
        return store.push(name="model", tag="v1", base_model="org/base", task="sft", run_id=None,
                          config={"lr": 0.0001})


def _count() -> int:
    with RegistryStore() as store:
        return len(store.list())


@pytest.mark.parametrize(
    "ref", ["registry://", "", None, "   ", "\t\n", "registry://   ", "registry://\t"],
)
def test_an_empty_reference_resolves_to_nothing(one_entry, ref):
    with RegistryStore() as store:
        assert store.resolve(ref) is None


def test_the_scheme_still_resolves_a_real_id_and_prefix(one_entry):
    with RegistryStore() as store:
        assert store.resolve(one_entry) == one_entry
        assert store.resolve(f"registry://{one_entry}") == one_entry
        assert store.resolve(f"registry://{one_entry[:-4]}") == one_entry


@pytest.mark.parametrize("ref", ["model:v1", "registry://model:v1"])
def test_name_and_tag_still_resolve(one_entry, ref):
    with RegistryStore() as store:
        assert store.resolve(ref) == one_entry


def test_delete_of_a_bare_scheme_keeps_the_entry(one_entry):
    result = runner.invoke(app, ["registry", "delete", "registry://", "--yes"])
    assert result.exit_code != 0, strip_ansi(result.output)
    assert _count() == 1
    with RegistryStore() as store:
        assert store.get(one_entry) is not None


def test_show_of_a_bare_scheme_is_not_found(one_entry):
    result = runner.invoke(app, ["registry", "show", "registry://"])
    assert result.exit_code != 0
    assert one_entry not in strip_ansi(result.output)
