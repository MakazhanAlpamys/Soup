import json
from pathlib import Path

import pytest

from soup_cli.utils.best_of_n_checkpoint import (
    append_checkpoint,
    initialise_checkpoint,
    load_checkpoint,
)
from tests.conftest import strip_ansi


@pytest.mark.parametrize(
    ("invalid_part", "message"),
    [
        ("run-digest", "does not match"),
        ("total", "does not match"),
        ("version", "version is not supported"),
        ("entry-digest", "digest mismatch"),
    ],
)
def test_invalid_checkpoint_keeps_incomplete_tail_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_part: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = Path("run.checkpoint.jsonl")
    initialise_checkpoint(str(checkpoint), digest="expected-run", total=2)
    append_checkpoint(
        str(checkpoint),
        index=0,
        sft={"prompt": "Question", "output": "Answer"},
        dpo=None,
    )

    records = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    header = records[0]["_best_of_n_checkpoint"]
    if invalid_part == "run-digest":
        header["run_digest"] = "different-run"
    elif invalid_part == "total":
        header["total"] = 99
    elif invalid_part == "version":
        header["version"] = 999
    else:
        records[1]["entry_digest"] = "invalid-entry-digest"

    committed = "".join(
        json.dumps(record, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    original = committed + b'{"index":1,"sft":'
    checkpoint.write_bytes(original)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(
            str(checkpoint),
            digest="expected-run",
            total=2,
        )

    assert checkpoint.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        checkpoint.name
    ]


def test_valid_checkpoint_recovers_incomplete_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = Path("run.checkpoint.jsonl")
    sft = {"prompt": "Question", "output": "Answer"}
    initialise_checkpoint(str(checkpoint), digest="expected-run", total=2)
    append_checkpoint(
        str(checkpoint),
        index=0,
        sft=sft,
        dpo=None,
    )
    committed = checkpoint.read_bytes()
    checkpoint.write_bytes(committed + b'{"index":1,"sft":')

    entries = load_checkpoint(
        str(checkpoint),
        digest="expected-run",
        total=2,
    )

    assert entries == [(sft, None)]
    assert checkpoint.read_bytes() == committed

    next_sft = {"prompt": "Next question", "output": "Next answer"}
    append_checkpoint(
        str(checkpoint),
        index=1,
        sft=next_sft,
        dpo=None,
    )

    assert load_checkpoint(
        str(checkpoint),
        digest="expected-run",
        total=2,
    ) == [(sft, None), (next_sft, None)]

@pytest.mark.parametrize(
    "content",
    [
        b"",
        b'{"_best_of_n_checkpoint":',
        (
            b'{"_best_of_n_checkpoint":'
            b'{"version":1,"run_digest":"expected-run","total":2}}'
        ),
    ],
    ids=["empty", "partial-header", "header-without-newline"],
)
def test_checkpoint_without_committed_header_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = Path("run.checkpoint.jsonl")
    checkpoint.write_bytes(content)

    with pytest.raises(ValueError, match="empty|header"):
        load_checkpoint(
            str(checkpoint),
            digest="expected-run",
            total=2,
        )

    assert checkpoint.read_bytes() == content


@pytest.mark.parametrize(
    "committed_corruption",
    [False, True],
    ids=["incomplete-utf8-tail", "invalid-utf8-committed-record"],
)
def test_checkpoint_handles_interrupted_utf8_without_losing_committed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    committed_corruption: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = Path("run.checkpoint.jsonl")
    sft = {"prompt": "Otázka", "output": "Odpoveď"}
    initialise_checkpoint(str(checkpoint), digest="expected-run", total=2)
    append_checkpoint(
        str(checkpoint),
        index=0,
        sft=sft,
        dpo=None,
    )
    committed = checkpoint.read_bytes()
    fragment = b'{"index":1,"sft":{"prompt":"\xc3'
    if committed_corruption:
        fragment += b"\n"

    original = committed + fragment
    checkpoint.write_bytes(original)

    if committed_corruption:
        with pytest.raises(ValueError, match="invalid JSON on line 3"):
            load_checkpoint(
                str(checkpoint),
                digest="expected-run",
                total=2,
            )
        assert checkpoint.read_bytes() == original
    else:
        assert load_checkpoint(
            str(checkpoint),
            digest="expected-run",
            total=2,
        ) == [(sft, None)]
        assert checkpoint.read_bytes() == committed

@pytest.mark.parametrize(
    "existing_outputs",
    [False, True],
    ids=["new-outputs", "existing-outputs"],
)
def test_resume_rejects_invalid_checkpoint_without_changing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_outputs: bool,
) -> None:
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.eval import judge
    from soup_cli.utils import magpie

    monkeypatch.chdir(tmp_path)
    source = Path("prompts.jsonl")
    source.write_bytes(b'{"prompt":"Explain gravity."}\n')

    checkpoint = Path("run.checkpoint.jsonl")
    initialise_checkpoint(
        str(checkpoint),
        digest="different-run",
        total=1,
    )
    header = json.loads(checkpoint.read_text(encoding="utf-8"))
    header["_best_of_n_checkpoint"]["version"] = 999
    checkpoint.write_bytes(
        (json.dumps(header) + "\n").encode("utf-8")
        + b'{"index":0,"sft":{"prompt":"\xc3'
    )

    output_names = [
        "selected.jsonl",
        "pairs.jsonl",
        "run.manifest.json",
    ]
    if existing_outputs:
        for name in output_names:
            Path(name).write_bytes(
                f"Previous successful artifact: {name}\n".encode("utf-8")
            )

    generation_calls = []

    def unexpected_generation(*args, **kwargs):
        generation_calls.append((args, kwargs))
        pytest.fail("Generation started before rejecting an invalid checkpoint")

    def fake_sampler_factory(*args, **kwargs):
        return unexpected_generation

    monkeypatch.setattr(
        judge,
        "JudgeEvaluator",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        magpie,
        "make_magpie_generate_fn",
        fake_sampler_factory,
    )

    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    result = CliRunner().invoke(
        app,
        [
            "data",
            "best-of-n",
            "--prompts",
            str(source),
            "--provider",
            "ollama",
            "--model",
            "test-model",
            "--judge",
            "ollama://test-judge",
            "--output",
            "selected.jsonl",
            "--emit-pairs",
            "pairs.jsonl",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            "run.manifest.json",
            "--resume",
        ],
    )
    message = " ".join(strip_ansi(result.output).split())

    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "Invalid checkpoint" in message
    assert "version is not supported" in message
    assert generation_calls == []
    assert after == before
