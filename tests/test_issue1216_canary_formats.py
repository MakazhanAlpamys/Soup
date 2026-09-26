"""#1216: `soup data canary insert` writes canaries the training loader keeps.

`insert` used to append `{"messages": [...]}` rows whatever the dataset's
format, so on every non-chatml dataset the loader dropped every canary and
`check` could only ever report OK. It now writes the canaries in the file's
own format (alpaca, sharegpt, chatml) and refuses every other format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import FORMAT_SIGNATURES, detect_format, format_to_messages
from soup_cli.data.loader import load_dataset, load_raw_data
from soup_cli.utils.canary import (
    CANARY_FORMATS,
    canary_rows,
    generate_canaries,
    load_manifest,
    write_manifest,
)
from tests.conftest import strip_ansi

_CHAT = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
_SHAREGPT = [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]

# One valid row per format `detect_format` can return.
_ROWS = {
    "alpaca": {"instruction": "q", "output": "a"},
    "sharegpt": {"conversations": _SHAREGPT},
    "chatml": {"messages": _CHAT},
    "dpo": {"prompt": "q", "chosen": "a", "rejected": "b"},
    "kto": {"prompt": "q", "completion": "a", "label": True},
    "llava": {"image": "img.png", "conversations": _SHAREGPT},
    "sharegpt4v": {"image": "img.png", "conversations": _SHAREGPT},
    "embedding": {"anchor": "q", "positive": "a"},
    "tool-calling": {"messages": _CHAT, "tools": [], "tool_calls": []},
    "audio": {"audio": "clip.wav", "messages": _CHAT},
    "asr": {"audio": "clip.wav", "text": "a"},
    "plaintext": {"text": "q a"},
}
_REFUSED = sorted(set(FORMAT_SIGNATURES) - set(CANARY_FORMATS))


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text)).strip()


def _write_dataset(path: Path, fmt: str, count: int = 3) -> None:
    path.write_text(
        "".join(json.dumps(_ROWS[fmt]) + "\n" for _ in range(count)),
        encoding="utf-8",
    )


def _insert(*extra: str):
    return CliRunner().invoke(
        app,
        ["data", "canary", "insert", "train.jsonl", "--manifest", "m.json",
         "--count", "4", *extra],
    )


def _secrets() -> list[str]:
    return [canary.secret.strip() for canary in load_manifest("m.json")]


def _loaded_rows(path: str, data_format: str) -> list[dict]:
    """The rows `soup train` would train on."""
    config = DataConfig(train=path, format=data_format, val_split=0.0)
    return list(load_dataset(config)["train"])


def test_every_detectable_format_has_a_fixture():
    """A new detect_format result (e.g. cross_encoder) must join the sweep."""
    assert set(_ROWS) == set(FORMAT_SIGNATURES)


@pytest.mark.parametrize("fmt", sorted(set(_ROWS) - {"sharegpt4v"}))
def test_fixture_rows_detect_as_their_format(fmt):
    # sharegpt4v has llava's signature and llava is checked first, so
    # detection never returns it; the explicit --format sweep still covers it.
    assert detect_format([_ROWS[fmt]]) == fmt


def test_alpaca_canaries_reach_the_training_rows(tmp_path, monkeypatch):
    """The issue's repro: 0 of 4 canaries on main."""
    monkeypatch.chdir(tmp_path)
    _write_dataset(Path("train.jsonl"), "alpaca")
    result = _insert("-o", "canaried.jsonl")
    assert result.exit_code == 0, (result.output, repr(result.exception))

    rows = _loaded_rows("canaried.jsonl", "auto")
    assert len(rows) == 7
    blob = json.dumps(rows)
    assert all(secret in blob for secret in _secrets())


@pytest.mark.parametrize("load_as", ["auto", "explicit"])
@pytest.mark.parametrize("insert_as", ["auto", "explicit"])
@pytest.mark.parametrize("fmt", CANARY_FORMATS)
def test_rendered_formats_load_every_canary(
    fmt, insert_as, load_as, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_dataset(Path("train.jsonl"), fmt)
    extra = ["--format", fmt] if insert_as == "explicit" else []
    result = _insert("-o", "canaried.jsonl", *extra)
    assert result.exit_code == 0, (result.output, repr(result.exception))

    rows = _loaded_rows("canaried.jsonl", fmt if load_as == "explicit" else "auto")
    assert len(rows) == 3 + 4
    blob = json.dumps(rows)
    missing = [secret for secret in _secrets() if secret not in blob]
    assert not missing, f"{fmt}: canaries dropped by the loader: {missing}"
    assert json.loads(Path("m.json").read_text())["format"] == fmt


@pytest.mark.parametrize("insert_as", ["auto", "explicit"])
@pytest.mark.parametrize("fmt", _REFUSED)
def test_other_formats_are_refused_and_nothing_is_written(
    fmt, insert_as, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_dataset(Path("train.jsonl"), fmt)
    extra = ["--format", fmt] if insert_as == "explicit" else []
    result = _insert("-o", "canaried.jsonl", *extra)

    assert result.exit_code == 1, (fmt, result.output)
    named = fmt if insert_as == "explicit" else detect_format([_ROWS[fmt]])
    output = _plain(result.output)
    assert f"'{named}'" in output
    assert "Supported: alpaca, sharegpt, chatml" in output
    assert not Path("canaried.jsonl").exists()
    assert not Path("m.json").exists()


def test_explicit_format_that_is_not_the_files_format_is_refused(
    tmp_path, monkeypatch
):
    """Under `data.format: auto` the file loads as chatml and drops them."""
    monkeypatch.chdir(tmp_path)
    _write_dataset(Path("train.jsonl"), "chatml")
    result = _insert("-o", "canaried.jsonl", "--format", "alpaca")

    assert result.exit_code == 1
    output = _plain(result.output)
    assert "--format alpaca" in output and "'chatml'" in output
    assert not Path("canaried.jsonl").exists()
    assert not Path("m.json").exists()


def test_empty_dataset_needs_an_explicit_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("train.jsonl").write_text("", encoding="utf-8")

    refused = _insert("-o", "canaried.jsonl")
    assert refused.exit_code == 1
    assert "--format" in _plain(refused.output)
    assert not Path("m.json").exists()

    result = _insert("-o", "canaried.jsonl", "--format", "sharegpt")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    rows = _loaded_rows("canaried.jsonl", "auto")
    assert len(rows) == 4
    assert all(secret in json.dumps(rows) for secret in _secrets())


def test_json_output_is_an_array_that_loads_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_dataset(Path("train.jsonl"), "alpaca")
    result = _insert("-o", "canaried.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))

    assert len(load_raw_data(Path("canaried.json"))) == 7
    rows = _loaded_rows("canaried.json", "auto")
    assert all(secret in json.dumps(rows) for secret in _secrets())


@pytest.mark.parametrize("name", ["canaried.txt", "canaried.csv", "canaried"])
def test_output_suffix_the_loader_would_misread_is_refused(
    name, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_dataset(Path("train.jsonl"), "alpaca")
    result = _insert("-o", name)

    assert result.exit_code == 1
    assert ".jsonl or .json" in _plain(result.output)
    assert not Path(name).exists()
    assert not Path("m.json").exists()


@pytest.mark.parametrize("fmt", CANARY_FORMATS)
def test_rendered_row_trains_on_the_pair_check_scores(fmt):
    """`check` scores (carrier as one user turn, secret as the target)."""
    canary = generate_canaries(count=1, seed=0)[0]
    (row,) = canary_rows([canary], fmt)
    assert format_to_messages(row, fmt)["messages"] == [
        {"role": "user", "content": canary.carrier},
        {"role": "assistant", "content": canary.secret.strip()},
    ]


@pytest.mark.parametrize("fmt", ["plaintext", "dpo", "bogus", None])
def test_canary_rows_refuses_formats_check_cannot_score(fmt):
    with pytest.raises(ValueError, match="unsupported canary format"):
        canary_rows(generate_canaries(count=1, seed=0), fmt)


class TestManifestFormat:
    def test_written_format_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        canaries = generate_canaries(count=2, seed=0)
        write_manifest(canaries, "m.json", "sharegpt")
        assert json.loads(Path("m.json").read_text())["format"] == "sharegpt"
        assert load_manifest("m.json") == canaries

    def test_manifest_without_format_loads_as_chatml(self, tmp_path, monkeypatch):
        """Every manifest written before #1216 has no format field."""
        monkeypatch.chdir(tmp_path)
        canaries = generate_canaries(count=2, seed=0)
        write_manifest(canaries, "m.json")
        payload = json.loads(Path("m.json").read_text())
        del payload["format"]
        Path("m.json").write_text(json.dumps(payload), encoding="utf-8")
        assert load_manifest("m.json") == canaries

    @pytest.mark.parametrize("value", ["plaintext", "dpo", "bogus", 7, None])
    def test_unknown_format_is_refused_not_scored_as_chat(
        self, value, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        write_manifest(generate_canaries(count=2, seed=0), "m.json")
        payload = json.loads(Path("m.json").read_text())
        payload["format"] = value
        Path("m.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported canary format"):
            load_manifest("m.json")


class TestCheckReadsTheFormat:
    def _check(self, monkeypatch):
        from soup_cli.commands import data_canary as cmd

        monkeypatch.setattr(
            cmd, "_load_pair", lambda *a, **k: (object(), object(), "cpu")
        )
        # Canaries mid-distribution -> OK.
        monkeypatch.setattr(
            cmd, "compute_pair_losses",
            lambda model, tok, pairs, **kw: (
                [5.0] * 2 + [float(i) for i in range(len(pairs) - 2)]
            ),
        )
        return CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "fake/model", "--controls", "16"],
        )

    def _manifest(self, **overrides):
        write_manifest(generate_canaries(count=2, seed=0), "m.json")
        payload = json.loads(Path("m.json").read_text())
        payload.update(overrides)
        for key in [k for k, v in overrides.items() if v is ...]:
            del payload[key]
        Path("m.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_old_manifest_without_format_still_checks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._manifest(format=...)
        result = self._check(monkeypatch)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Verdict: OK" in _plain(result.output)

    def test_unknown_manifest_format_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._manifest(format="plaintext")
        result = self._check(monkeypatch)
        assert result.exit_code == 1
        assert "unsupported canary format 'plaintext'" in _plain(result.output)
