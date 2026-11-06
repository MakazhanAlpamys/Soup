"""soup steer prints typed paths, ids and errors as typed."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.registry.attach as registry_attach
import soup_cli.utils.steering as steering
from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)
# A backslash separates directories on Windows, so a name that holds one cannot be a file there.
_NO_BACKSLASH_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="a backslash separates directories on Windows"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately. A backslash before
# an unmatched bracket and a trailing pair of backslashes are what markup rewrites.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.steer.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    (workdir / "pairs.jsonl").write_text('{"positive": "a", "negative": "b"}\n', encoding="utf-8")


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _train(*args: str, base: str = "tiny-model", pairs: str = "pairs.jsonl"):
    return runner.invoke(
        app,
        ["steer", "train", "--base", base, "--name", "vec", "--pairs", pairs, *args],
    )


def _raising(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def _artifact(output_dir: str = "out"):
    return SimpleNamespace(
        method="caa", name="vec", layer=3, hidden_dim=8, intervention_point="residual",
        output_dir=output_dir, base="tiny-model", num_pairs=1,
    )


# --- train: input errors ---------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_method_is_shown_as_typed(value):
    result = _train("--method", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid steer-train input: unknown steering method {value!r}; supported:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_name_is_shown_as_typed(value):
    result = runner.invoke(
        app, ["steer", "train", "--base", "b", "--name", value, "--pairs", "pairs.jsonl"]
    )

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid steer-train input: steering name {value!r} must match" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_pairs_outside_cwd_are_shown_as_typed(name):
    result = _train(pairs=f"../{name}.jsonl")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}.jsonl").replace("\u202e", "")
    assert f"Invalid steer-train input: --pairs {quoted} must stay under cwd" in text
    assert _not_shown(text)


# --- train: plan and result ------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_shows_the_base_as_typed(value):
    result = _train("--plan-only", base=value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Base: " + re.escape(_shown(value)) + r" (?:\W )+Method: caa", text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_shows_the_pairs_path_as_typed(name):
    Path(f"{name}.jsonl").write_text('{"positive": "a", "negative": "b"}\n', encoding="utf-8")

    result = _train("--plan-only", pairs=f"{name}.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Pairs: \S*" + re.escape(_shown(name)) + r"\.jsonl (?:\W )+Layer:", text)
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError, FileNotFoundError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_training_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(steering, "build_steering_vector", _raising(error(f"train said {value}")))

    result = _train()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Steer-train failed: train said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_result_panel_shows_the_saved_directory_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        steering, "build_steering_vector", lambda **kwargs: _artifact(f"out-{value}")
    )

    result = _train()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Pairs: 1 (?:\W )+Saved: out-" + re.escape(_shown(value)), text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_attach_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(steering, "build_steering_vector", lambda **kwargs: _artifact())
    failing = _raising(RuntimeError(f"attach {value}"))
    monkeypatch.setattr(registry_attach, "attach_artifact", failing)

    result = _train("--registry-id", "entry1")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Error: could not attach to registry: attach {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_attached_entry_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        steering, "build_steering_vector", lambda **kwargs: _artifact(f"out-{value}")
    )
    monkeypatch.setattr(registry_attach, "attach_artifact", lambda *a, **k: None)

    result = _train("--registry-id", f"id-{value}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = re.escape(_shown(value))
    assert re.search(
        r"Attached steering_vector to registry entry id-" + shown + r" \(out-" + shown, text
    )
    assert _not_shown(text)


# --- apply -----------------------------------------------------------------


@pytest.mark.parametrize("value", _VALUES)
def test_apply_shows_a_refused_name_as_typed(value):
    result = runner.invoke(app, ["steer", "apply", "--name", value])

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Invalid steer-apply input: steering name {value!r} must match" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_apply_shows_a_failed_load_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(steering, "resolve_steering_dir", _raising(error(f"load said {value}")))

    result = runner.invoke(app, ["steer", "apply", "--name", "vec"])

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Cannot load steering vector: load said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_apply_shows_the_directory_as_typed(monkeypatch, value):
    loaded = SimpleNamespace(
        method="caa", layer=3, intervention_point="residual", vector=[0.0] * 8
    )
    monkeypatch.setattr(steering, "resolve_steering_dir", lambda name: f"dir-{value}")
    monkeypatch.setattr(steering, "load_steering_artifact", lambda path: loaded)

    result = runner.invoke(app, ["steer", "apply", "--name", "vec"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Dir: dir-" + re.escape(_shown(value)) + r" ", text)
    assert _not_shown(text)


# --- list ------------------------------------------------------------------


class _FakeStore:
    def __init__(self, name, path):
        self._name = name
        self._path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def list(self):
        return [{"id": "e1", "name": self._name}]

    def get_artifacts(self, entry_id):
        return [{"kind": "steering_vector", "path": self._path}]


@pytest.mark.parametrize("value", _VALUES)
def test_list_shows_the_entry_and_the_path_as_typed(monkeypatch, value):
    import soup_cli.registry.store as registry_store

    monkeypatch.setattr(
        registry_store, "RegistryStore", lambda: _FakeStore(f"entry-{value}", f"path-{value}")
    )

    result = runner.invoke(app, ["steer", "list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert f"\u2502 entry-{shown} \u2502 path-{shown} \u2502" in text
    assert _not_shown(text)


# --- styles and layout -----------------------------------------------------


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.steer.console",
        Console(
            force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
        ),
    )


def test_errors_start_with_a_red_label(monkeypatch):
    _coloured(monkeypatch)

    refused = _train("--method", "nope")
    monkeypatch.setattr(steering, "build_steering_vector", _raising(ValueError("train said no")))
    failed = _train()

    assert refused.exit_code == 2
    assert failed.exit_code == 2
    assert "\x1b[31mInvalid steer-train input:\x1b[0m unknown steering method" in refused.output
    assert "\x1b[31mSteer-train failed:\x1b[0m train said no" in failed.output


def test_the_result_panel_keeps_its_labels_bold(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(steering, "build_steering_vector", lambda **kwargs: _artifact())

    result = _train()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    for label in ("Method:", "Name:", "Layer:", "Intervention:", "Hidden dim:", "Pairs:", "Saved:"):
        assert f"\x1b[1m{label}\x1b[0m" in result.output, label


def test_the_attach_line_has_a_green_label_a_bold_id_and_a_dim_path(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(steering, "build_steering_vector", lambda **kwargs: _artifact())
    monkeypatch.setattr(registry_attach, "attach_artifact", lambda *a, **k: None)

    result = _train("--registry-id", "entry1")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[32mAttached\x1b[0m steering_vector to registry entry" in result.output
    assert "\x1b[1mentry1\x1b[0m" in result.output
    assert re.search(r"\x1b\[2m\(\S*vector\S*\)\x1b\[0m", result.output)


def test_a_failed_attach_starts_with_a_red_label(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(steering, "build_steering_vector", lambda **kwargs: _artifact())
    monkeypatch.setattr(registry_attach, "attach_artifact", _raising(RuntimeError("no")))

    result = _train("--registry-id", "entry1")

    assert result.exit_code == 1
    assert "\x1b[31mError:\x1b[0m could not attach to registry: no" in result.output


def test_apply_keeps_its_labels_bold(monkeypatch):
    _coloured(monkeypatch)
    loaded = SimpleNamespace(
        method="caa", layer=3, intervention_point="residual", vector=[0.0] * 8
    )
    monkeypatch.setattr(steering, "resolve_steering_dir", lambda name: "some-dir")
    monkeypatch.setattr(steering, "load_steering_artifact", lambda path: loaded)

    result = runner.invoke(app, ["steer", "apply", "--name", "vec"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    for label in ("Vector:", "Method:", "Layer:", "Intervention:", "Dim:", "Strength:", "Dir:"):
        assert f"\x1b[1m{label}\x1b[0m" in result.output, label


def _line_counts(output: str, labels) -> dict:
    lines = strip_ansi(output).splitlines()
    return {
        label: sum(1 for line in lines if re.match(r"\W*" + re.escape(label), line))
        for label in labels
    }


def test_every_detail_of_the_train_panels_has_its_own_line(monkeypatch):
    monkeypatch.setattr(steering, "build_steering_vector", lambda **kwargs: _artifact())

    result = _train()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # The plan panel and the result panel both name the method, the name, the layer and the pairs.
    expected = {
        "Base:": 1, "Method:": 2, "Name:": 2, "Pairs:": 2, "Layer:": 2, "Description:": 1,
        "Intervention:": 1, "Hidden dim:": 1, "Saved:": 1,
    }
    assert _line_counts(result.output, expected) == expected


def test_every_detail_of_the_apply_panel_has_its_own_line(monkeypatch):
    loaded = SimpleNamespace(
        method="caa", layer=3, intervention_point="residual", vector=[0.0] * 8
    )
    monkeypatch.setattr(steering, "resolve_steering_dir", lambda name: "some-dir")
    monkeypatch.setattr(steering, "load_steering_artifact", lambda path: loaded)

    result = runner.invoke(app, ["steer", "apply", "--name", "vec"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    expected = dict.fromkeys(
        ("Vector:", "Method:", "Layer:", "Intervention:", "Dim:", "Strength:", "Dir:"), 1
    )
    assert _line_counts(result.output, expected) == expected
