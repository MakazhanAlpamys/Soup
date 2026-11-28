"""``soup data`` Hub and registry commands show outside text as typed (#2489).

Search queries, dataset ids, splits, formats, dataset names, paths, Hub
metadata and the errors raised about them keep ``:name:`` aliases, markup and
a backslash before a bracket as written, and lose their control and bidi
characters, in ``search``, ``preview``, ``download``, ``register``,
``unregister``, ``registry`` and ``push``. Display only: exit codes, the
registry file and the files written are unchanged. No test here reaches the
network -- every Hub, ModelScope and Modelers call is replaced -- and the
registry lives in a temporary home.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import data
from soup_cli.data.formats import CONVERTIBLE_FORMATS
from soup_cli.utils import hf, hubs
from soup_cli.utils.hf import validate_repo_id
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x\x9b\u202en", "x\x1b\x07n"]
#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")
#: A closing tag with nothing to close.
CLOSING_TAG = "a[/x]b"
ALL_VALUES = [*VALUES, CLOSING_TAG]
#: Dataset names the registry accepts: no ``/``, ``\`` or ``:``.
REGISTER_NAMES = [v for v in ALL_VALUES if not any(sep in v for sep in "/\\:")]
#: ``--hf-dataset`` values the repo-id validator refuses (all but ``plain``).
BAD_REPO_IDS = [v for v in ALL_VALUES if v != "plain"]
HUBS = "hf, modelers, modelscope"
ALPACA_ROW = {"instruction": "a", "output": "b"}

_NOT_A_WINDOWS_NAME = "{} cannot be part of a file name on Windows"
#: ``ALL_VALUES`` that a test writes as part of a real file or directory name.
FILE_VALUES = [
    "plain",
    "x[red]n",
    "[/]n",
    pytest.param("n:rocket:", marks=skip_colon_on_windows),
    pytest.param(
        "x\\[n",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason=_NOT_A_WINDOWS_NAME.format("'\\'")
        ),
    ),
    "x\x9b\u202en",
    pytest.param(
        "x\x1b\x07n",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason=_NOT_A_WINDOWS_NAME.format("A C0 control")
        ),
    ),
    CLOSING_TAG,
]


def _shown(value: object) -> str:
    text = str(value)
    for char in DROPPED:
        text = text.replace(char, "")
    return text


def _flat(text: str) -> str:
    return " ".join(text.split())


def _plain(result) -> str:
    """The output without colour, after checking how the command ended."""
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    output = strip_ansi(result.output)
    assert [char for char in DROPPED if char in output] == []
    return output


def _assert_lines(result, exit_code: int, *lines: str) -> None:
    """Each of ``lines`` is a whole line of the output; nothing dropped is anywhere."""
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    shown = [_flat(line) for line in _plain(result).splitlines()]
    for line in lines:
        assert _flat(line) in shown, (line, shown)


def _assert_row(result, exit_code: int, *cells: str) -> None:
    """A table row holds exactly ``cells``; nothing dropped is anywhere."""
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    rows = [
        [_flat(cell) for cell in line.split("│")[1:-1]]
        for line in _plain(result).splitlines()
        if line.startswith("│")
    ]
    assert [_flat(cell) for cell in cells] in rows, (cells, rows)


def _data(*args: str):
    return make_cli_runner().invoke(app, ["data", *args])


def _raising(error: Exception):
    def fail(*_args, **_kwargs):
        raise error

    return fail


def _registry() -> Path:
    return data._get_registry_path()


def _write_registry(entries: dict) -> None:
    path = _registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


def _read_registry() -> dict:
    return json.loads(_registry().read_text(encoding="utf-8"))


def _repo_id_error(value: str) -> str:
    with pytest.raises(ValueError) as caught:
        validate_repo_id(value)
    return str(caught.value)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """cwd is ``tmp_path/work``; home (and so ``~/.soup``) is ``tmp_path/home``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(data, "console", Console(width=1000, height=100))
    assert _registry() == home / ".soup" / "datasets.json"
    return work


def _search(monkeypatch, found=(), error=None) -> list:
    calls = []

    def search(search, sort="downloads", limit=20):
        calls.append(search)
        if error is not None:
            raise error
        return list(found)

    monkeypatch.setattr(data, "list_datasets", search)
    return calls


def _dataset_info(value: str, description: str = "") -> dict:
    return {
        "id": f"o/{value}",
        "description": description,
        "downloads": 2_000_000,
        "likes": 7,
        "size_bytes": 2048,
        "splits": {value: 1200},
        "features": [value, "text"],
        "tags": [value, "t"],
    }


def _downloaded(monkeypatch, rows=(ALPACA_ROW,), error=None) -> list:
    calls = []

    def download(dataset_id, split="train", samples=None, trust_remote_code=False):
        calls.append((dataset_id, split))
        if error is not None:
            raise error
        return [dict(row) for row in rows]

    monkeypatch.setattr(data, "_hf_download_dataset", download)
    return calls


def _modelscope(monkeypatch) -> list:
    calls = []
    package = ModuleType("modelscope")
    module = ModuleType("modelscope.msdatasets")

    class MsDataset:
        @staticmethod
        def load(dataset_id, split):
            calls.append((dataset_id, split))
            return object()

    module.MsDataset = MsDataset
    package.msdatasets = module
    monkeypatch.setitem(sys.modules, "modelscope", package)
    monkeypatch.setitem(sys.modules, "modelscope.msdatasets", module)
    return calls


def _modelers(monkeypatch, out_dir=None, error=None) -> list:
    calls = []

    def download(hub, repo_id, *, local_dir, repo_type="model", **_kwargs):
        calls.append((hub, repo_id, repo_type))
        if error is not None:
            raise error
        return out_dir

    monkeypatch.setattr(hubs, "download_repo", download)
    return calls


class _Api:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        if self.error is not None:
            raise self.error


def _hf_push(monkeypatch, api) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hf, "get_hf_api", lambda token=None, endpoint=None: api)


def _push(*args: str, hf_dataset: str = "owner/ds", input_path: str = "d.jsonl"):
    return _data("push", "--input", input_path, "--hf-dataset", hf_dataset, *args)


def _dataset_file(name: str = "d.jsonl") -> Path:
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ALPACA_ROW) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# soup data search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ALL_VALUES)
def test_search_names_a_bad_sort_as_typed(workdir, monkeypatch, value):
    calls = _search(monkeypatch)

    result = _data("search", "q", "--sort", value)

    _assert_lines(
        result,
        1,
        f"Invalid sort: {_shown(value)}",
        "Valid options: createdAt, downloads, lastModified, likes, trending",
    )
    assert calls == []


@pytest.mark.parametrize("value", ALL_VALUES)
def test_search_shows_a_hub_error_as_raised(workdir, monkeypatch, value):
    _search(monkeypatch, error=RuntimeError(f"hub said {value}"))

    result = _data("search", "q")

    _assert_lines(result, 1, f"Search failed: hub said {_shown(value)}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_search_with_no_results_names_the_query_as_typed(workdir, monkeypatch, value):
    calls = _search(monkeypatch)

    result = _data("search", value)

    _assert_lines(result, 0, f"No datasets found for '{_shown(value)}'.")
    assert calls == [value]


@pytest.mark.parametrize("value", ALL_VALUES)
def test_search_table_shows_query_ids_and_tags_as_typed(workdir, monkeypatch, value):
    found = [SimpleNamespace(id=f"owner/{value}", downloads=1500, likes=3, tags=[value, "t2"])]
    _search(monkeypatch, found=found)

    result = _data("search", value)

    _assert_lines(result, 0, f"HuggingFace Datasets: '{_shown(value)}'", "Showing 1 results.")
    _assert_row(result, 0, f"owner/{_shown(value)}", "1.5K", "3", f"{_shown(value)}, t2")


# ---------------------------------------------------------------------------
# soup data preview
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ALL_VALUES)
def test_preview_shows_a_lookup_error_as_raised(workdir, monkeypatch, value):
    error = ValueError(f"Dataset not found: {value} — boom")
    monkeypatch.setattr(data, "_hf_dataset_info", _raising(error))

    result = _data("preview", value)

    _assert_lines(result, 1, f"Dataset not found: {_shown(value)} — boom")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_preview_shows_hub_metadata_as_typed(workdir, monkeypatch, value):
    info = _dataset_info(value, description=f"about {value}")
    monkeypatch.setattr(data, "_hf_dataset_info", lambda _dataset_id: info)

    result = _data("preview", f"o/{value}")

    shown = _shown(value)
    _assert_lines(result, 0, f"Dataset: o/{shown}")
    _assert_row(result, 0, "ID", f"o/{shown}")
    _assert_row(result, 0, "Description", f"about {shown}")
    _assert_row(result, 0, "Downloads", "2.0M")
    _assert_row(result, 0, "Size", "2.0 KB")
    _assert_row(result, 0, "Splits", f"{shown} (1.2K)")
    _assert_row(result, 0, "Features", f"{shown}, text")
    _assert_row(result, 0, "Tags", f"{shown}, t")


def test_preview_without_a_description_says_so(workdir, monkeypatch):
    monkeypatch.setattr(data, "_hf_dataset_info", lambda _dataset_id: _dataset_info("d"))

    result = _data("preview", "o/d")

    _assert_row(result, 0, "Description", "No description")


# ---------------------------------------------------------------------------
# soup data download
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_names_an_unknown_hub_as_typed(workdir, value):
    result = _data("download", "o/d", "--hub", value)

    _assert_lines(result, 2, f"hub {value!r} not supported. Supported: {HUBS}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_from_modelscope_names_the_dataset_as_typed(workdir, monkeypatch, value):
    calls = _modelscope(monkeypatch)

    result = _data("download", value, "--hub", "modelscope")

    _assert_lines(
        result,
        0,
        f"ModelScope dataset {_shown(value)} loaded; use soup_cli.utils.hubs.download_repo "
        "for raw snapshot download.",
    )
    assert calls == [(value, "train")]


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_from_modelers_names_dataset_and_folder_as_typed(workdir, monkeypatch, value):
    calls = _modelers(monkeypatch, out_dir=f"cache/{value}")

    result = _data("download", value, "--hub", "modelers")

    shown = _shown(value)
    _assert_lines(result, 0, f"Downloaded {shown} from modelers → cache/{shown}")
    assert calls == [("modelers", value, "dataset")]


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_from_modelers_shows_a_missing_sdk_as_raised(workdir, monkeypatch, value):
    _modelers(monkeypatch, error=ImportError(f"install {value}"))

    result = _data("download", "o/d", "--hub", "modelers")

    _assert_lines(result, 1, f"install {_shown(value)}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_from_modelers_shows_a_refusal_as_raised(workdir, monkeypatch, value):
    _modelers(monkeypatch, error=ValueError(f"bad repo {value}"))

    result = _data("download", "o/d", "--hub", "modelers")

    _assert_lines(result, 2, f"bad repo {_shown(value)}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_names_dataset_and_split_as_typed(workdir, monkeypatch, value):
    error = ValueError(f"Failed to load dataset o/{value}: boom")
    calls = _downloaded(monkeypatch, error=error)

    result = _data("download", f"o/{value}", "--split", value, "--output", "out.jsonl")

    shown = _shown(value)
    _assert_lines(
        result,
        1,
        f"Downloading o/{shown} (split={shown})...",
        f"Failed to load dataset o/{shown}: boom",
    )
    assert calls == [(f"o/{value}", value)]
    assert not Path("out.jsonl").exists()


@pytest.mark.parametrize("value", ALL_VALUES)
def test_download_names_a_bad_format_as_typed(workdir, monkeypatch, value):
    _downloaded(monkeypatch)

    result = _data("download", "o/d", "--format", value, "--output", "out.jsonl")

    _assert_lines(
        result,
        1,
        f"Invalid format: {_shown(value)}",
        f"Supported: {', '.join(CONVERTIBLE_FORMATS)}",
    )
    assert not Path("out.jsonl").exists()


@pytest.mark.parametrize("value", FILE_VALUES)
def test_download_names_the_output_file_as_typed(workdir, monkeypatch, value):
    _downloaded(monkeypatch)
    name = f"{value}.jsonl"
    Path(name).parent.mkdir(parents=True, exist_ok=True)

    result = _data("download", "o/d", "--output", name)

    _assert_lines(result, 0, "Downloaded 1 rows.", f"Output: {_shown(Path(name).resolve())}")
    assert Path(name).read_text(encoding="utf-8") == json.dumps(ALPACA_ROW) + "\n"


def test_download_conversion_lines_keep_their_wording(workdir, monkeypatch):
    # The format named on these two lines is one of CONVERTIBLE_FORMATS: an
    # unknown --format is refused above, so only a closed set reaches them.
    _downloaded(monkeypatch, rows=(ALPACA_ROW, {"instruction": 5, "output": "x"}))

    result = _data("download", "o/d", "--format", "sharegpt", "--output", "out.jsonl")

    _assert_lines(
        result,
        0,
        "Converted 1 rows to sharegpt format.",
        "1 rows were left out because they do not convert to sharegpt.",
        "Downloaded 1 rows.",
    )


# ---------------------------------------------------------------------------
# soup data register / unregister / registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ALL_VALUES)
def test_register_names_a_refused_name_as_typed(workdir, value):
    name = f"{value}/"

    result = _data("register", name, "data.jsonl")

    _assert_lines(
        result,
        1,
        f"Dataset name '{_shown(name)}' must not contain path separators (/ \\ :) or null bytes",
    )
    assert not _registry().exists()


@pytest.mark.parametrize("value", REGISTER_NAMES)
def test_register_names_the_dataset_as_typed(workdir, value):
    result = _data("register", value, "data.jsonl")

    _assert_lines(
        result, 0, f"Registered dataset '{_shown(value)}'", "Path: data.jsonl", "Format: auto"
    )
    assert _read_registry()[value]["format"] == "auto"


@pytest.mark.parametrize("value", ALL_VALUES)
def test_register_names_the_format_as_typed(workdir, value):
    result = _data("register", "plain", "data.jsonl", "--format", value)

    _assert_lines(result, 0, "Registered dataset 'plain'", f"Format: {_shown(value)}")
    assert _read_registry() == {
        "plain": {"path": str(Path("data.jsonl").resolve()), "format": value}
    }


@pytest.mark.parametrize(
    "value",
    [pytest.param(v, marks=skip_colon_on_windows) if ":" in v else v for v in ALL_VALUES],
)
def test_register_names_the_path_as_typed(workdir, value):
    result = _data("register", "plain", f"data/{value}.jsonl")

    _assert_lines(result, 0, "Registered dataset 'plain'", f"Path: data/{_shown(value)}.jsonl")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_unregister_names_a_missing_dataset_as_typed(workdir, value):
    result = _data("unregister", value)

    _assert_lines(result, 1, f"Dataset '{_shown(value)}' not found in registry.")
    assert not _registry().exists()


@pytest.mark.parametrize("value", ALL_VALUES)
def test_unregister_names_the_removed_dataset_as_typed(workdir, value):
    kept = {"path": "k.jsonl", "format": "auto"}
    _write_registry({value: {"path": "p.jsonl", "format": "auto"}, "kept": kept})

    result = _data("unregister", value)

    _assert_lines(result, 0, f"Removed dataset '{_shown(value)}' from registry.")
    assert _read_registry() == {"kept": kept}


@pytest.mark.parametrize("value", ALL_VALUES)
def test_registry_table_shows_entries_as_stored(workdir, value):
    _write_registry({value: {"path": f"p/{value}", "format": value}})

    result = _data("registry")

    shown = _shown(value)
    _assert_lines(result, 0, "Registered Datasets")
    _assert_row(result, 0, shown, f"p/{shown}", shown)


# ---------------------------------------------------------------------------
# soup data push
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ALL_VALUES)
def test_push_names_an_unknown_hub_as_typed(workdir, value):
    result = _push("--hub", value)

    _assert_lines(result, 2, f"hub {value!r} not supported. Supported: {HUBS}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_push_names_a_missing_file_as_typed(workdir, value):
    name = f"{value}.jsonl"

    result = _push(input_path=name)

    _assert_lines(result, 1, f"Dataset file not found: {_shown(Path(name))}")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_push_names_a_directory_as_typed(workdir, value):
    Path(value).mkdir(parents=True)

    result = _push(input_path=value)

    _assert_lines(result, 1, f"Expected a file, got a directory: {_shown(Path(value))}")


@pytest.mark.parametrize("value", BAD_REPO_IDS)
def test_push_names_a_bad_repo_id_as_typed(workdir, value):
    _dataset_file()

    result = _push(hf_dataset=value)

    _assert_lines(result, 1, f"Invalid --hf-dataset repo id: {_shown(_repo_id_error(value))}")


def test_push_names_the_hub_in_a_bad_repo_id_refusal(workdir):
    _dataset_file()

    result = _push("--hub", "modelscope", hf_dataset="a[/x]b")

    _assert_lines(
        result, 1, f"Invalid --modelscope-dataset repo id: {_repo_id_error('a[/x]b')}"
    )


def test_push_names_a_bad_endpoint(workdir, monkeypatch):
    _dataset_file()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_ENDPOINT", "ftp://example.com")

    result = _push()

    _assert_lines(
        result, 1, "HF_ENDPOINT invalid: HF_ENDPOINT must use http/https scheme, got: ftp"
    )


@pytest.mark.parametrize("value", ALL_VALUES)
def test_push_shows_an_endpoint_error_as_raised(workdir, monkeypatch, value):
    # A real HF_ENDPOINT only ever puts its URL scheme in this error, and
    # urlparse keeps a scheme to [A-Za-z0-9+.-]; the error is replaced here.
    _dataset_file()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hf, "resolve_endpoint", _raising(ValueError(f"bad {value}")))

    result = _push()

    _assert_lines(result, 1, f"HF_ENDPOINT invalid: bad {_shown(value)}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_push_shows_a_missing_sdk_as_raised(workdir, monkeypatch, value):
    _dataset_file()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hf, "get_hf_api", _raising(ImportError(f"install {value}")))

    result = _push()

    _assert_lines(result, 1, f"install {_shown(value)}")


@pytest.mark.parametrize("value", ALL_VALUES)
def test_push_shows_an_upload_error_as_raised(workdir, monkeypatch, value):
    _dataset_file()
    api = _Api(error=RuntimeError(f"rejected {value}"))
    _hf_push(monkeypatch, api)

    result = _push()

    _assert_lines(result, 1, f"Upload failed: rejected {_shown(value)}")


def test_push_receipt_names_the_dataset_url(workdir, monkeypatch):
    # The repo id on this line passed validate_repo_id: [A-Za-z0-9._-] and one '/'.
    _dataset_file()
    api = _Api()
    _hf_push(monkeypatch, api)

    result = _push(hf_dataset="owner/data-set")

    _assert_lines(result, 0, "Uploaded to https://huggingface.co/datasets/owner/data-set")
    assert [call[1]["repo_id"] for call in api.calls] == ["owner/data-set"] * 2


@pytest.mark.parametrize(
    ("error", "exit_line"),
    [
        (ImportError, "install {}"),
        (ValueError, "Upload failed: install {}"),
        (RuntimeError, "Upload failed: install {}"),
    ],
)
@pytest.mark.parametrize("value", ALL_VALUES)
def test_push_to_modelscope_shows_an_upload_error_as_raised(
    workdir, monkeypatch, value, error, exit_line
):
    _dataset_file()
    monkeypatch.setattr(hubs, "upload_repo", _raising(error(f"install {value}")))

    result = _push("--hub", "modelscope")

    _assert_lines(result, 1, exit_line.format(_shown(value)))


@pytest.mark.parametrize("value", FILE_VALUES)
def test_push_to_modelscope_names_the_uploaded_file_as_typed(workdir, monkeypatch, value):
    name = f"{value}.jsonl"
    _dataset_file(name)
    calls = []
    monkeypatch.setattr(hubs, "upload_repo", lambda *args, **kwargs: calls.append(args))

    result = _push("--hub", "modelscope", input_path=name)

    _assert_lines(
        result, 0, f"Uploaded {_shown(Path(name).name)} to modelscope dataset owner/ds"
    )
    assert calls == [("modelscope", "owner/ds")]


# ---------------------------------------------------------------------------
# Colour: each line keeps its style
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(workdir, monkeypatch):
    monkeypatch.setattr(
        data,
        "console",
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
    )
    return workdir


def test_a_refusal_stays_red(coloured):
    result = _data("search", "q", "--sort", "bad")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "\x1b[31mInvalid sort: bad\x1b[0m\n" in result.output


def test_no_results_stays_yellow(coloured, monkeypatch):
    _search(monkeypatch)

    result = _data("search", "q")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[33mNo datasets found for 'q'.\x1b[0m\n" in result.output


def test_the_search_title_stays_italic(coloured, monkeypatch):
    _search(monkeypatch, found=[SimpleNamespace(id="o/d", downloads=1, likes=1, tags=[])])

    result = _data("search", "q")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    title = next(line for line in result.output.splitlines() if "HuggingFace Datasets" in line)
    assert title.startswith("\x1b[3m") and title.endswith("\x1b[0m"), repr(title)
    assert _flat(title[len("\x1b[3m") : -len("\x1b[0m")]) == "HuggingFace Datasets: 'q'"


def test_the_register_receipt_stays_green_with_a_bold_name(coloured):
    result = _data("register", "mine", "data.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        "\x1b[32mRegistered dataset '\x1b[0m\x1b[1;32mmine\x1b[0m\x1b[32m'\x1b[0m\n"
        in result.output
    )
    assert "  Path: data.jsonl\n  Format: auto\n" in result.output


def test_the_download_progress_line_stays_dim(coloured, monkeypatch):
    _downloaded(monkeypatch)

    result = _data("download", "o/d", "--output", "out.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[2mDownloading o/d (split=train)...\x1b[0m\n" in result.output


def test_no_description_stays_dim(coloured, monkeypatch):
    monkeypatch.setattr(data, "_hf_dataset_info", lambda _dataset_id: _dataset_info("d"))

    result = _data("preview", "o/d")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[2mNo description\x1b[0m" in result.output
