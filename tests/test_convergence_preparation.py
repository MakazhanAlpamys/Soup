"""Behavioral contract for the STEP 11/19 data reconstruction."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.harness.prep_convergence import LABELS, prepare, verify
from soup_cli.eval.custom import load_eval_tasks


@pytest.fixture
def sources(tmp_path: Path) -> dict:
    paths = {}
    for split, count in (("train", 15020), ("test", 320)):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"text": f"{split} feeling {i}", "label": i % 6})
                                + "\n" for i in range(count)), encoding="utf-8")
        paths[split] = path
        paths[f"{split}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**paths, "revision": "a" * 40, "seed": 42}


def test_complete_protocol_and_consumers(sources: dict, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    manifest = prepare(output=output, **sources)
    assert manifest["historical_bytes_recovered"] is False
    assert manifest["source"]["revision"] == sources["revision"]
    assert manifest["seed"] == 42
    assert manifest["source"]["train_sha256"] == sources["train_sha256"]
    identities = []
    prompts = set()
    for i in range(5):
        data = (output / f"train_s{i}.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in data]
        assert len(rows) == 3000
        assert all([m["role"] for m in row["messages"]] == ["user", "assistant"] for row in rows)
        assert {row["messages"][1]["content"] for row in rows} == set(LABELS)
        for row, index in zip(rows, manifest["files"][f"train_s{i}.jsonl"]["source_rows"]):
            assert row["messages"][1]["content"] == (
                "sadness", "joy", "love", "anger", "fear", "surprise")[index % 6]
            assert f"train feeling {index}\nEmotion:" in row["messages"][0]["content"]
        prompts.update(row["messages"][0]["content"] for row in rows)
        identities.extend(manifest["files"][f"train_s{i}.jsonl"]["source_rows"])
    assert len(identities) == len(set(identities)) == 15000
    tasks = load_eval_tasks(str(output / "task_eval.jsonl"))
    assert len(tasks) == 300
    assert all(task.scoring == "exact" and task.expected in LABELS for task in tasks)
    for task, index in zip(tasks, manifest["files"]["task_eval.jsonl"]["source_rows"]):
        assert task.expected == ("sadness", "joy", "love", "anger", "fear", "surprise")[index % 6]
        assert f"test feeling {index}\nEmotion:" in task.prompt
    assert not prompts.intersection(task.prompt for task in tasks)
    assert verify(output) == manifest


def test_deterministic_bytes_and_seed_changes_selection(sources: dict, tmp_path: Path) -> None:
    for directory in ("a", "b", "c"):
        prepare(output=tmp_path / directory, **{**sources, "seed": 43 if directory == "c" else 42})
    for path in (tmp_path / "a").iterdir():
        assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes()
    assert (tmp_path / "a/train_s0.jsonl").read_bytes() != (
        tmp_path / "c/train_s0.jsonl").read_bytes()


@pytest.mark.parametrize("field,value", [("revision", "main"), ("seed", True),
                                        ("train_sha256", "0" * 64)])
def test_refuses_unpinned_or_changed_source(
    sources: dict, tmp_path: Path, field: str, value: object,
) -> None:
    with pytest.raises(ValueError):
        prepare(output=tmp_path / "bad", **{**sources, field: value})
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("row", [{"text": "x", "label": True}, {"text": "x", "label": 6},
                                 {"text": "", "label": 1}, {"label": 1}])
def test_invalid_rows_fail_before_publishing(sources: dict, tmp_path: Path, row: dict) -> None:
    sources["train"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    sources["train_sha256"] = hashlib.sha256(sources["train"].read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        prepare(output=tmp_path / "bad", **sources)
    assert not (tmp_path / "bad").exists()


def test_excludes_duplicates_and_heldout_text(sources: dict, tmp_path: Path) -> None:
    with sources["train"].open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"text": "test feeling 1", "label": 1}) + "\n")
        stream.write(json.dumps({"text": "train feeling 1", "label": 1}) + "\n")
    sources["train_sha256"] = hashlib.sha256(sources["train"].read_bytes()).hexdigest()
    manifest = prepare(output=tmp_path / "out", **sources)
    assert manifest["excluded_train_rows"] == 2
    assert verify(tmp_path / "out") == manifest


def test_short_source_fails(sources: dict, tmp_path: Path) -> None:
    sources["test"].write_text('{"text":"one","label":0}\n', encoding="utf-8")
    sources["test_sha256"] = hashlib.sha256(sources["test"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="300"):
        prepare(output=tmp_path / "bad", **sources)


def test_tamper_and_overwrite_fail(sources: dict, tmp_path: Path) -> None:
    output = tmp_path / "out"
    prepare(output=output, **sources)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(output=output, **sources)
    assert before == (output / "manifest.json").read_bytes()
    path = output / "train_s2.jsonl"
    path.write_bytes(path.read_bytes().replace(b"feeling", b"changed", 1))
    with pytest.raises(ValueError, match="hash"):
        verify(output)


def test_empty_existing_directory_is_also_preserved(sources: dict, tmp_path: Path) -> None:
    # POSIX rename can replace an empty directory; refusal must be explicit.
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare(output=output, **sources)
    assert output.is_dir() and list(output.iterdir()) == []


@pytest.mark.parametrize("corruption", ["overlap", "heldout", "label", "roles", "count"])
def test_verifier_checks_semantics_even_with_updated_hash(
    sources: dict, tmp_path: Path, corruption: str,
) -> None:
    output = tmp_path / "out"
    manifest = prepare(output=output, **sources)
    filename = "task_eval.jsonl" if corruption == "heldout" else "train_s1.jsonl"
    rows = [json.loads(line) for line in (output / filename).read_text().splitlines()]
    if corruption == "overlap":
        manifest["files"][filename]["source_rows"][0] = manifest["files"]["train_s0.jsonl"][
            "source_rows"][0]
    elif corruption == "heldout":
        first = json.loads((output / "train_s0.jsonl").read_text().splitlines()[0])
        rows[0]["prompt"] = first["messages"][0]["content"]
    elif corruption == "label":
        rows[0]["messages"][1]["content"] = "invalid"
    elif corruption == "roles":
        rows[0]["messages"][0]["role"] = "assistant"
    else:
        rows.pop()
    data = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    (output / filename).write_bytes(data)
    manifest["files"][filename]["sha256"] = hashlib.sha256(data).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(output)


def test_cli_default_fails_and_self_test_runs() -> None:
    script = Path(__file__).parents[1] / "benchmarks/harness/prep_convergence.py"
    default = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert default.returncode != 0
    self_test = subprocess.run([sys.executable, str(script), "--self-test"],
                               capture_output=True, text=True)
    assert self_test.returncode == 0, self_test.stderr + self_test.stdout
