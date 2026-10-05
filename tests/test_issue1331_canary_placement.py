"""#1331: `soup data canary insert` spreads the canaries through the file.

`insert` used to append the canaries, and the loader holds out the file's tail
as validation (`_split_val`, default `data.val_split: 0.1`). From 135 rows on,
all 16 canaries were in validation, no model trained on them, and
`soup data canary check` could only report OK.

The canaries are now spread evenly: the file is cut into K equal stretches and
canary i goes to a seeded random row of stretch i, so a held-out tail of
fraction v holds about v of them for every seed. The manifest records each
canary's row and the file's row count, so which canaries a split trains on can
be worked out from it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig
from soup_cli.data.loader import load_dataset
from soup_cli.utils.canary import canary_rows, generate_canaries, load_manifest
from tests.conftest import strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text)).strip()


def _write_alpaca(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps({"instruction": f"question {i}", "output": f"answer {i}"}) + "\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )


def _insert(*extra: str):
    return CliRunner().invoke(
        app,
        [
            "data",
            "canary",
            "insert",
            "train.jsonl",
            "-o",
            "canaried.jsonl",
            "--manifest",
            "m.json",
            *extra,
        ],
    )


def _secrets_in(rows) -> set[str]:
    blob = json.dumps(list(rows))
    return {c.secret.strip() for c in load_manifest("m.json") if c.secret.strip() in blob}


# --------------------------------------------------------------------------
# end to end: insert with its defaults, load at the default val_split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [100, 135, 200, 1000])
def test_the_default_split_trains_on_the_canaries(rows, tmp_path, monkeypatch):
    """The issue's table: tail placement put 12, 15, 16 and 16 of 16 in val."""
    monkeypatch.chdir(tmp_path)
    _write_alpaca(Path("train.jsonl"), rows)
    result = _insert()
    assert result.exit_code == 0, (result.output, repr(result.exception))

    manifest = json.loads(Path("m.json").read_text(encoding="utf-8"))
    assert len(manifest["canaries"]) == 16
    assert manifest["rows"] == rows + 16

    config = DataConfig(train="canaried.jsonl", format="auto")
    assert config.val_split == 0.1
    data = load_dataset(config)
    assert len(data["train"]) + len(data["val"]) == rows + 16
    in_train = _secrets_in(data["train"])
    in_val = _secrets_in(data["val"])
    assert len(in_train) + len(in_val) == 16

    # The manifest says which canaries the default split trains on.
    split_idx = len(data["train"])
    trainable = {
        entry["secret"].strip() for entry in manifest["canaries"] if entry["row"] < split_idx
    }
    assert in_train == trainable
    assert len(in_val) <= 3, sorted(in_val)
    # And `insert` said so.
    assert (
        f"At the default data.val_split (0.1), {len(in_train)} of 16 canaries are "
        "in the training split" in _plain(result.output)
    ), result.output


# --------------------------------------------------------------------------
# the manifest describes the file
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["0", "7"])
@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_each_recorded_row_holds_its_canary(suffix, seed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_alpaca(Path("train.jsonl"), 40)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "canary",
            "insert",
            "train.jsonl",
            "-o",
            f"canaried{suffix}",
            "--manifest",
            "m.json",
            "--count",
            "5",
            "--seed",
            seed,
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))

    text = Path(f"canaried{suffix}").read_text(encoding="utf-8")
    written = (
        json.loads(text)
        if suffix == ".json"
        else [json.loads(line) for line in text.splitlines() if line]
    )
    manifest = json.loads(Path("m.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == len(written) == 45
    rows = [entry["row"] for entry in manifest["canaries"]]
    assert rows == sorted(set(rows))
    for entry in manifest["canaries"]:
        assert written[entry["row"]] == {
            "instruction": entry["carrier"],
            "output": entry["secret"].strip(),
        }
    # Every other row is the dataset's, in its original order.
    others = [row for i, row in enumerate(written) if i not in set(rows)]
    assert others == [{"instruction": f"question {i}", "output": f"answer {i}"} for i in range(40)]
    # `check` still reads the manifest.
    assert len(load_manifest("m.json")) == 5


def test_the_printed_count_follows_the_loaders_split(tmp_path, monkeypatch):
    """20 rows at val_split 0.1 train on rows 0-17: a canary on row 17 is trained
    on and one on row 18, the first held-out row, is not."""
    from soup_cli.commands import data_canary as cmd

    monkeypatch.chdir(tmp_path)
    _write_alpaca(Path("train.jsonl"), 18)

    def _at_the_boundary(rows, inserted, *, seed):
        mixed = list(rows[:17]) + [inserted[0], inserted[1]] + list(rows[17:])
        return mixed, (17, 18)

    monkeypatch.setattr(cmd, "interleave_canary_rows", _at_the_boundary)
    result = _insert("--count", "2")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        "At the default data.val_split (0.1), 1 of 2 canaries are in the training "
        "split" in _plain(result.output)
    ), result.output

    data = load_dataset(DataConfig(train="canaried.jsonl", format="auto"))
    assert len(data["train"]) == 18
    assert len(_secrets_in(data["train"])) == 1


def test_insertion_is_deterministic_in_the_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_alpaca(Path("train.jsonl"), 300)

    def _run(seed: str) -> tuple[str, str]:
        result = _insert("--seed", seed)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        return (
            Path("canaried.jsonl").read_text(encoding="utf-8"),
            Path("m.json").read_text(encoding="utf-8"),
        )

    first = _run("3")
    assert _run("3") == first

    def _rows(manifest_text: str) -> list[int]:
        return [entry["row"] for entry in json.loads(manifest_text)["canaries"]]

    assert _rows(_run("4")[1]) != _rows(first[1])


# --------------------------------------------------------------------------
# the placement itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(
    "rows, count", [(0, 16), (1, 16), (100, 16), (135, 16), (1000, 16), (7, 3), (50, 1)]
)
def test_each_canary_lands_in_its_own_stretch_of_the_file(rows, count, seed):
    """Canary i sits in the i-th of `count` equal stretches of the mix, so no
    end of the file (the val tail included) can collect them."""
    from soup_cli.utils.canary import interleave_canary_rows

    data = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(rows)]
    inserted = canary_rows(generate_canaries(count=count, seed=seed), "alpaca")

    mixed, positions = interleave_canary_rows(data, inserted, seed=seed)

    total = rows + count
    assert len(mixed) == total
    assert [mixed[p] for p in positions] == inserted
    assert [row for i, row in enumerate(mixed) if i not in positions] == data
    for i, position in enumerate(positions):
        assert i * total // count <= position < (i + 1) * total // count, (i, position)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("val_split", [0.05, 0.1, 0.2, 0.5])
def test_a_held_out_tail_holds_about_its_share_for_every_seed(val_split, seed):
    from soup_cli.utils.canary import interleave_canary_rows

    rows, count = 500, 16
    data = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(rows)]
    inserted = canary_rows(generate_canaries(count=count, seed=0), "alpaca")
    _, positions = interleave_canary_rows(data, inserted, seed=seed)

    split_idx = int((rows + count) * (1 - val_split))
    held_out = sum(1 for p in positions if p >= split_idx)
    assert abs(held_out - val_split * count) <= 1, (held_out, positions)


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------


def test_the_canary_docs_say_how_canaries_meet_val_split():
    text = (REPO_ROOT / "docs" / "data.md").read_text(encoding="utf-8")
    section = text.split("## Canaries (`soup data canary insert|check`)", 1)[1]
    section = " ".join(section.split("\n## ", 1)[0].split())
    assert "data.val_split" in section, section
    assert "spread" in section, section
    assert '"row"' in section, section
