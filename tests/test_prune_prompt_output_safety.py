import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.utils.prune_prompt import prune_traces
from tests.conftest import strip_ansi


class CharacterTokenizer:
    """Deterministic tokenizer double; no model downloads required."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.mark.parametrize("token_aware", [False, True], ids=["char", "token"])
@pytest.mark.parametrize("path_alias", [False, True], ids=["same-path", "dot-alias"])
def test_prune_rejects_input_as_output_without_changing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_aware: bool,
    path_alias: bool,
) -> None:
    monkeypatch.chdir(tmp_path)

    source = Path("traces.jsonl")
    original = (
        '{"prompt": "SYSTEM: alpha", "output": "one"}\n'
        '{"prompt": "SYSTEM: beta", "output": "two"}\n'
    ).encode("utf-8")
    source.write_bytes(original)

    output_path = "./traces.jsonl" if path_alias else "traces.jsonl"
    tokenizer = CharacterTokenizer() if token_aware else None

    try:
        with pytest.raises(ValueError):
            prune_traces(
                "traces.jsonl",
                output_path=output_path,
                min_frequency=1.0,
                tokenizer=tokenizer,
            )
    finally:
        assert source.read_bytes() == original, "Input data was modified"

@pytest.mark.parametrize("output_path", ["traces.jsonl", "./traces.jsonl"])
def test_prune_cli_rejects_input_as_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_path: str,
) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    source = Path("traces.jsonl")
    original = (
        '{"prompt": "SYSTEM: alpha", "output": "one"}\n'
        '{"prompt": "SYSTEM: beta", "output": "two"}\n'
    ).encode("utf-8")
    source.write_bytes(original)

    result = CliRunner().invoke(
        app,
        [
            "prune-prompt",
            "--input",
            "traces.jsonl",
            "--output",
            output_path,
        ],
    )
    message = " ".join(strip_ansi(result.output).split())

    assert source.read_bytes() == original
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "--input" in message
    assert "--output" in message
    assert "same file" in message

@pytest.mark.parametrize("token_aware", [False, True], ids=["char", "token"])
def test_prune_writes_distinct_output_without_changing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_aware: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("traces.jsonl")
    destination = Path("pruned.jsonl")
    rows = [
        {"prompt": "SYSTEM: alpha", "output": "one", "metadata": {"id": 1}},
        {"prompt": "SYSTEM: beta", "output": "two", "metadata": {"id": 2}},
    ]
    original = (
        "\n".join(json.dumps(row) for row in rows) + "\n"
    ).encode("utf-8")
    source.write_bytes(original)

    tokenizer = CharacterTokenizer() if token_aware else None
    report = prune_traces(
        str(source),
        output_path=str(destination),
        min_frequency=1.0,
        tokenizer=tokenizer,
    )

    assert source.read_bytes() == original
    assert destination.is_file()
    actual_rows = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert actual_rows == [
        {"prompt": "alpha", "output": "one", "metadata": {"id": 1}},
        {"prompt": "beta", "output": "two", "metadata": {"id": 2}},
    ]
    assert report.prefix == "SYSTEM: "
    assert report.prefix_chars == len("SYSTEM: ")
    assert report.rows_total == 2
    assert report.rows_pruned == 2
    assert report.min_frequency == 1.0

@pytest.mark.parametrize("token_aware", [False, True], ids=["char", "token"])
@pytest.mark.parametrize(
    "existing_output",
    [False, True],
    ids=["new-output", "existing-output"],
)
def test_prune_read_failure_does_not_publish_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_aware: bool,
    existing_output: bool,
) -> None:
    import builtins
    from contextlib import contextmanager

    from soup_cli.utils import prune_prompt

    monkeypatch.chdir(tmp_path)
    source = Path("traces.jsonl")
    destination = Path("pruned.jsonl")
    original_input = (
        b'{"prompt": "SYSTEM alpha", "output": "one"}\n'
        b'{"prompt": "SYSTEM beta", "output": "two"}\n'
    )
    original_output = b"Previous successful output\n"
    source.write_bytes(original_input)
    if existing_output:
        destination.write_bytes(original_output)

    expected_files = {path.name for path in tmp_path.iterdir()}
    real_open = builtins.open
    source_reads = 0

    def interrupted_lines(handle):
        yield next(handle)
        raise OSError("Simulated failure during second input pass")

    @contextmanager
    def open_with_read_failure(file, mode="r", *args, **kwargs):
        nonlocal source_reads
        with real_open(file, mode, *args, **kwargs) as handle:
            if Path(file).resolve() == source.resolve() and mode == "r":
                source_reads += 1
                if source_reads == 2:
                    yield interrupted_lines(handle)
                    return
            yield handle

    monkeypatch.setattr(
        prune_prompt,
        "open",
        open_with_read_failure,
        raising=False,
    )

    with pytest.raises(
        OSError,
        match="Simulated failure during second input pass",
    ):
        prune_prompt.prune_traces(
            str(source),
            output_path=str(destination),
            min_frequency=1.0,
            tokenizer=CharacterTokenizer() if token_aware else None,
        )

    assert source_reads == 2
    assert source.read_bytes() == original_input
    if existing_output:
        assert destination.read_bytes() == original_output
    else:
        assert not destination.exists()
    assert {path.name for path in tmp_path.iterdir()} == expected_files

@pytest.mark.parametrize("token_aware", [False, True], ids=["char", "token"])
@pytest.mark.parametrize(
    "existing_output",
    [False, True],
    ids=["new-output", "existing-output"],
)
def test_prune_replace_failure_preserves_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_aware: bool,
    existing_output: bool,
) -> None:
    from soup_cli.utils import paths

    monkeypatch.chdir(tmp_path)
    source = Path("traces.jsonl")
    destination = Path("pruned.jsonl")
    original_input = (
        b'{"prompt": "SYSTEM alpha", "output": "one"}\n'
        b'{"prompt": "SYSTEM beta", "output": "two"}\n'
    )
    original_output = b"Previous successful output\n"
    source.write_bytes(original_input)
    if existing_output:
        destination.write_bytes(original_output)

    expected_files = {path.name for path in tmp_path.iterdir()}
    replace_calls = []

    def fail_replace(src, dst):
        staged = Path(src)
        replace_calls.append((staged, Path(dst)))
        assert staged.resolve() != destination.resolve()
        assert staged.parent.resolve() == destination.parent.resolve()
        rows = [
            json.loads(line)
            for line in staged.read_text(encoding="utf-8").splitlines()
        ]
        assert rows == [
            {"prompt": "alpha", "output": "one"},
            {"prompt": "beta", "output": "two"},
        ]
        assert Path(dst).resolve() == destination.resolve()
        raise OSError("Simulated output replacement failure")

    with monkeypatch.context() as patch:
        patch.setattr(paths.os, "replace", fail_replace)
        with pytest.raises(
            OSError,
            match="Simulated output replacement failure",
        ):
            prune_traces(
                str(source),
                output_path=str(destination),
                min_frequency=1.0,
                tokenizer=CharacterTokenizer() if token_aware else None,
            )

    assert len(replace_calls) == 1
    assert source.read_bytes() == original_input
    if existing_output:
        assert destination.read_bytes() == original_output
    else:
        assert not destination.exists()
    assert {path.name for path in tmp_path.iterdir()} == expected_files
