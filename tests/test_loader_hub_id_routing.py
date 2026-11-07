"""`load_dataset` routing of a single-string `data.train`: Hub id, local file or
remote URI.

A Hub dataset id with a dot in its name (``teknium/OpenHermes-2.5``,
``org/name.v1``) used to be read as a local file: the single-string branch asked
``Path(train_path).suffix`` ("is there any extension?"), and ``Path("org/name-2.5").suffix``
is ``".5"``. The list form (``data.train: [a, b]`` with ``data.interleave``) was
fixed in #468 by classifying each entry with ``loader._classify_train_entry``
(local only when the suffix is one of ``SUPPORTED_EXTENSIONS``); the single
string now goes through the same function, so the two spellings of one dataset
name cannot disagree. One check is added on top for the single string: a string
with a suffix that is an existing file is a local file whatever the suffix, so a
real ``data.tsv`` still gets the loader's "Unsupported file format" error rather
than a Hub lookup.

Each test records the branch ``load_dataset`` takes by replacing the three
loaders with stubs, so nothing touches the network or the disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from soup_cli.config.schema import DataConfig, _classify_data_train_entry
from soup_cli.data import loader


class _RoutedError(Exception):
    """Raised by a stub loader to stop load_dataset at the branch it took."""

    def __init__(self, route: str, target: str) -> None:
        super().__init__(route, target)
        self.route = route
        self.target = target


@pytest.fixture
def route_of(monkeypatch):
    """Return ``route(name) -> (branch, argument the loader received)``."""

    def fake_hub(name, *args, **kwargs):
        raise _RoutedError("hub", name)

    def fake_remote(name, *args, **kwargs):
        raise _RoutedError("remote", name)

    def fake_local(path):
        raise _RoutedError("local", str(path))

    monkeypatch.setattr(loader, "_load_hf_dataset", fake_hub)
    monkeypatch.setattr(loader, "_load_remote_dataset", fake_remote)
    monkeypatch.setattr(loader, "load_raw_data", fake_local)

    def route(name: str) -> tuple[str, str]:
        config = DataConfig(train=name, format="plaintext", val_split=0.0)
        with pytest.raises(_RoutedError) as caught:
            loader.load_dataset(config)
        return caught.value.route, caught.value.target

    return route


# ---------------------------------------------------------------------------
# The bug: a dot in a Hub id is not a file extension.
# ---------------------------------------------------------------------------

DOTTED_HUB_IDS = [
    "teknium/OpenHermes-2.5",
    "org/name-2.5",
    "org/name.v1",
    "org/model-3.1-data",
    "org/name-v1.1",
    "mlfoundations/dclm-baseline-1.0",
    "name-2.5",
]


@pytest.mark.parametrize("name", DOTTED_HUB_IDS)
def test_dotted_hub_id_routes_to_the_hub_loader(route_of, name):
    assert route_of(name) == ("hub", name)


@pytest.mark.parametrize("name", ["org/name", "name", "teknium/OpenHermes", "a/b_c-d"])
def test_undotted_hub_id_routes_to_the_hub_loader(route_of, name):
    assert route_of(name) == ("hub", name)


def test_dotted_hub_id_loads_rows_through_datasets_load_dataset(monkeypatch, tmp_path):
    """End to end through the real _load_hf_dataset with a stand-in `datasets`:
    the id reaches `datasets.load_dataset` unchanged and its rows come back."""
    asked: list[str] = []

    def fake_load_dataset(name, **kwargs):
        asked.append(name)
        return {"train": [{"text": f"row-{i}"} for i in range(3)]}

    fake_module = type(sys)("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    monkeypatch.chdir(tmp_path)

    config = DataConfig(train="teknium/OpenHermes-2.5", format="plaintext", val_split=0.0)
    result = loader.load_dataset(config)

    assert asked == ["teknium/OpenHermes-2.5"]
    assert [row["text"] for row in result["train"]] == ["row-0", "row-1", "row-2"]


# ---------------------------------------------------------------------------
# What must not move: local files with a real extension stay local.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extension",
    [".jsonl", ".json", ".csv", ".parquet", ".txt", ".JSONL", ".Json", ".CSV"],
)
def test_existing_local_file_with_a_supported_extension_routes_to_the_local_loader(
    route_of, tmp_path, extension
):
    data_file = tmp_path / f"train{extension}"
    data_file.write_text("x", encoding="utf-8")
    route, target = route_of(str(data_file))
    assert (route, Path(target)) == ("local", data_file)


@pytest.mark.parametrize("extension", [".jsonl", ".json", ".csv", ".parquet", ".txt"])
def test_dotted_local_filename_with_a_supported_extension_stays_local(
    route_of, tmp_path, extension
):
    # "v1.2.data.jsonl": the dots before the last one belong to the file name.
    data_file = tmp_path / f"v1.2.data{extension}"
    data_file.write_text("x", encoding="utf-8")
    route, target = route_of(str(data_file))
    assert (route, Path(target)) == ("local", data_file)


def test_local_file_inside_a_dotted_directory_stays_local(route_of, tmp_path):
    data_dir = tmp_path / "my.data-2.5"
    data_dir.mkdir()
    data_file = data_dir / "train.jsonl"
    data_file.write_text("x", encoding="utf-8")
    route, target = route_of(str(data_file))
    assert (route, Path(target)) == ("local", data_file)


def test_relative_local_file_stays_local(route_of, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.jsonl").write_text("x", encoding="utf-8")
    assert route_of("train.jsonl") == ("local", "train.jsonl")
    assert route_of("./data/train.parquet") == ("local", str(Path("./data/train.parquet")))


# ---------------------------------------------------------------------------
# What must not move: a path with no extension, and remote URIs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["./sub/dir", "plain_name", "some/dir/without_extension"])
def test_path_without_an_extension_keeps_routing_to_the_hub_loader(route_of, name):
    # On main a name with no suffix goes to the Hub loader, whether or not a
    # local entry of that name exists; that is unchanged.
    assert route_of(name) == ("hub", name)


def test_existing_local_directory_without_an_extension_still_routes_to_the_hub_loader(
    route_of, tmp_path
):
    local_dir = tmp_path / "saved_dataset"
    local_dir.mkdir()
    assert route_of(str(local_dir)) == ("hub", str(local_dir))


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/data.jsonl",
        "gs://bucket/data-2.5.jsonl",
        "az://container/data.v1",
        "s3://bucket/no_extension",
        "https://example.com/data.jsonl",
    ],
)
def test_remote_uri_keeps_routing_to_the_remote_loader(route_of, uri):
    # Checked before any suffix test, so a dotted or extension-less URI is
    # still remote (the scheme allowlist is enforced inside the remote loader).
    assert route_of(uri) == ("remote", uri)


# ---------------------------------------------------------------------------
# A name that also exists on disk. The classifier is lexical (the list form
# and the schema never ask the file system); `load_dataset` adds one check on
# top for the single string: a string with a non-empty suffix that IS a file
# is a local file, so a real file with an unsupported extension still gets
# load_raw_data's "Unsupported file format" error instead of a Hub lookup.
# ---------------------------------------------------------------------------

UNSUPPORTED_FILE_NAMES = ["data.tsv", "train.jsonl.gz", "data.xlsx"]


@pytest.mark.parametrize("filename", UNSUPPORTED_FILE_NAMES)
def test_existing_file_with_an_unsupported_extension_routes_to_the_local_loader(
    route_of, tmp_path, filename
):
    data_file = tmp_path / filename
    data_file.write_text("x", encoding="utf-8")
    route, target = route_of(str(data_file))
    assert (route, Path(target)) == ("local", data_file)


@pytest.mark.parametrize("filename", UNSUPPORTED_FILE_NAMES)
def test_existing_relative_file_with_an_unsupported_extension_routes_to_the_local_loader(
    route_of, tmp_path, monkeypatch, filename
):
    (tmp_path / filename).write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert route_of(filename) == ("local", filename)


@pytest.mark.parametrize(
    ("filename", "extension"),
    [("data.tsv", ".tsv"), ("train.jsonl.gz", ".gz"), ("data.xlsx", ".xlsx")],
)
def test_existing_file_with_an_unsupported_extension_ends_in_the_unsupported_format_error(
    monkeypatch, tmp_path, filename, extension
):
    # Through the real load_dataset and the real load_raw_data; the Hub and
    # remote loaders fail loudly if they are ever reached.
    def must_not_run(*args, **kwargs):
        raise AssertionError("an existing local file must not reach the Hub/remote loader")

    monkeypatch.setattr(loader, "_load_hf_dataset", must_not_run)
    monkeypatch.setattr(loader, "_load_remote_dataset", must_not_run)
    data_file = tmp_path / filename
    data_file.write_text("x", encoding="utf-8")

    config = DataConfig(train=str(data_file), format="plaintext", val_split=0.0)
    with pytest.raises(ValueError, match=rf"Unsupported file format: {extension}\. Supported:"):
        loader.load_dataset(config)


@pytest.mark.parametrize("filename", UNSUPPORTED_FILE_NAMES)
def test_missing_file_with_an_unsupported_extension_routes_to_the_hub_loader(
    route_of, tmp_path, monkeypatch, filename
):
    # No such file: the lexical rule applies, as in the list form.
    assert route_of(str(tmp_path / filename)) == ("hub", str(tmp_path / filename))
    monkeypatch.chdir(tmp_path)
    assert route_of(filename) == ("hub", filename)


def test_dotted_name_that_exists_as_a_file_is_a_local_file(route_of, tmp_path, monkeypatch):
    # A file literally called "org/name-2.5" is what the string names.
    (tmp_path / "org").mkdir()
    (tmp_path / "org" / "name-2.5").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert route_of("org/name-2.5") == ("local", str(Path("org/name-2.5")))


def test_dotted_name_that_exists_as_a_directory_is_still_a_hub_id(
    route_of, tmp_path, monkeypatch
):
    # A directory keeps the Hub route (datasets.load_dataset itself accepts a
    # local directory), so the id is not turned into a local file path.
    (tmp_path / "org" / "name-2.5").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert route_of("org/name-2.5") == ("hub", "org/name-2.5")
    absolute = str(tmp_path / "org" / "name-2.5")
    assert route_of(absolute) == ("hub", absolute)


def test_dotted_name_with_no_such_file_is_a_hub_id(route_of, tmp_path, monkeypatch):
    (tmp_path / "org").mkdir()
    monkeypatch.chdir(tmp_path)
    assert route_of("org/name-2.5") == ("hub", "org/name-2.5")


def test_existing_file_without_an_extension_keeps_routing_to_the_hub_loader(
    route_of, tmp_path, monkeypatch
):
    # Only a string WITH a suffix is checked against the disk: a Hub id such as
    # "imdb" must not turn into a local path because a file of that name sits
    # in the working directory. Same route as before.
    (tmp_path / "imdb").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert route_of("imdb") == ("hub", "imdb")
    absolute = str(tmp_path / "imdb")
    assert route_of(absolute) == ("hub", absolute)


# ---------------------------------------------------------------------------
# One rule for both spellings: single string == list entry == schema.
# ---------------------------------------------------------------------------

PARITY_NAMES = [
    "org/name-2.5",
    "org/name.v1",
    "teknium/OpenHermes-2.5",
    "org/name",
    "name",
    "train.jsonl",
    "train.JSON",
    "data.csv",
    "data.parquet",
    "notes.txt",
    "v1.2.data.jsonl",
    "./sub/dir",
    "data.tsv",
    "train.jsonl.gz",
    "s3://bucket/data.jsonl",
    "gs://bucket/data-2.5.jsonl",
    "https://example.com/data.jsonl",
]


@pytest.mark.parametrize("name", PARITY_NAMES)
def test_single_string_route_matches_the_list_form_and_the_schema(
    route_of, name, tmp_path, monkeypatch
):
    # In an empty directory, so no name here exists on disk: the file check in
    # load_dataset (tested above) is the only thing that may differ from the
    # classifier, and only for a string that is a real file.
    monkeypatch.chdir(tmp_path)
    kind = loader._classify_train_entry(name)
    assert route_of(name)[0] == kind
    assert _classify_data_train_entry(name) == kind
