"""Behavioral contract for the STEP 11/19 data reconstruction."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.harness.prep_convergence import LABELS, prepare, verify
from soup_cli.eval.custom import load_eval_tasks

FROM_RECORD = [
    "dair-ai/emotion dataset and exact-match metric scoring",
    "five disjoint 3000-row subsets of the 16000-row train split",
    "one fixed 300-row test sample, identical for every run",
    "file names train_s0.jsonl and task_eval.jsonl",
]
NEW_CHOICES = [
    "split config and caller-asserted revision",
    "SHA-256 selection rule and explicit seed",
    "exact-text deduplication, keeping the first label",
    "excluding all test text from train",
    "prompt template, label strings and row formats",
    "file names train_s1.jsonl through train_s4.jsonl",
]


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
    assert manifest["from_record"] == FROM_RECORD
    assert manifest["new_choices"] == NEW_CHOICES
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


@pytest.mark.parametrize("field,value,message", [
    ("revision", "main", "revision must be an immutable 40-character commit SHA"),
    ("seed", True, "seed must be an explicit integer"),
    ("train_sha256", "bad", "source SHA-256 must be lowercase hex"),
    ("train_sha256", "0" * 64, "source hash mismatch"),
    ("test_sha256", "0" * 64, "source hash mismatch"),
])
def test_refuses_unpinned_or_changed_source(
    sources: dict, tmp_path: Path, field: str, value: object, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare(output=tmp_path / "bad", **{**sources, field: value})
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("row,message", [
    ({"text": "x", "label": True}, "empty text or invalid emotion label"),
    ({"text": "x", "label": 6}, "empty text or invalid emotion label"),
    ({"text": "", "label": 1}, "empty text or invalid emotion label"),
    ({"label": 1}, "each source row needs text"),
])
def test_invalid_rows_fail_before_publishing(
    sources: dict, tmp_path: Path, row: dict, message: str,
) -> None:
    # Keep enough valid rows: a missing validation must not fail on row count instead.
    with sources["train"].open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    sources["train_sha256"] = hashlib.sha256(sources["train"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=message):
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


@pytest.mark.parametrize("corruption,message", [
    ("overlap", "training subsets overlap"),
    ("heldout", "invalid label or overlapping prompt"),
    ("label", "invalid label or overlapping prompt"),
    ("roles", "invalid training roles"),
    ("count", "invalid row counts or identities"),
    ("historical", "unsupported reconstruction manifest"),
    ("index", "invalid row counts or identities"),
    ("split", "invalid row counts or identities"),
])
def test_verifier_checks_semantics_even_with_updated_hash(
    sources: dict, tmp_path: Path, corruption: str, message: str,
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
    elif corruption == "historical":
        manifest["historical_bytes_recovered"] = True
    elif corruption == "index":
        manifest["files"][filename]["source_rows"][0] = manifest["source_counts"]["train"]
    elif corruption == "split":
        manifest["files"][filename]["split"] = "test"
    else:
        rows.pop()
    data = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    (output / filename).write_bytes(data)
    manifest["files"][filename]["sha256"] = hashlib.sha256(data).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        verify(output)


@pytest.mark.parametrize("field", ["from_record", "new_choices"])
@pytest.mark.parametrize("change", ["missing", "altered"])
def test_verifier_requires_provenance_distinction(
    sources: dict, tmp_path: Path, field: str, change: str,
) -> None:
    output = tmp_path / "out"
    manifest = prepare(output=output, **sources)
    if change == "missing":
        manifest.pop(field, None)
    else:
        manifest[field] = ["all choices are historical"]
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid reconstruction provenance"):
        verify(output)


def test_unicode_separators_inside_source_text_are_preserved(sources: dict, tmp_path: Path) -> None:
    separator_text = "before\u2028middle\u0085after"
    with sources["train"].open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"text": separator_text, "label": 2}, ensure_ascii=False) + "\n")
    sources["train_sha256"] = hashlib.sha256(sources["train"].read_bytes()).hexdigest()
    manifest = prepare(output=tmp_path / "out", **sources)
    selected = [name for name, entry in manifest["files"].items()
                if entry["split"] == "train" and 15020 in entry["source_rows"]]
    assert len(selected) == 1
    rows = [json.loads(line) for line in (tmp_path / "out" / selected[0]).read_bytes().splitlines()]
    matching = [row for row in rows if separator_text in row["messages"][0]["content"]]
    assert len(matching) == 1
    assert matching[0]["messages"][1]["content"] == "love"


def test_cli_default_fails_and_self_test_runs() -> None:
    script = Path(__file__).parents[1] / "benchmarks/harness/prep_convergence.py"
    default = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             timeout=60)
    assert default.returncode == 2, default.stderr + default.stdout
    assert "provide train/test JSONL" in default.stderr
    self_test = subprocess.run([sys.executable, str(script), "--self-test"],
                               capture_output=True, text=True, timeout=60)
    assert self_test.returncode == 0, self_test.stderr + self_test.stdout


def test_cli_verification_failure_reports_reason(sources: dict, tmp_path: Path) -> None:
    output = tmp_path / "out"
    prepare(output=output, **sources)
    (output / "train_s0.jsonl").write_bytes(b"tampered\n")
    script = Path(__file__).parents[1] / "benchmarks/harness/prep_convergence.py"
    result = subprocess.run([sys.executable, str(script), "--verify", str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stderr + result.stdout
    assert "Preparation failed: output hash mismatch" in result.stdout


def test_cli_existing_output_reports_reason(sources: dict, tmp_path: Path) -> None:
    output = tmp_path / "[existing]"
    output.mkdir()
    script = Path(__file__).parents[1] / "benchmarks/harness/prep_convergence.py"
    arguments = []
    for key, value in sources.items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run([sys.executable, str(script), *arguments, "--output", str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stderr + result.stdout
    assert "Preparation failed: output already exists" in result.stdout
    assert list(output.iterdir()) == []
