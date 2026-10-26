#!/usr/bin/env python3
"""Reconstruct the emotion data protocol in H100 STEP 11/19, not historical bytes.

Inputs are JSONL exports of dair-ai/emotion's split train/test columns text/label.
Their explicitly supplied SHA-256 values bind the actual bytes; the revision is
caller-asserted provenance, not authentication of their Hugging Face origin.
No ML package, network access, training or model evaluation is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console

LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
PROMPT = "Classify the emotion as sadness, joy, love, anger, fear, or surprise.\nText: {}\nEmotion:"
FILES = tuple(f"train_s{i}.jsonl" for i in range(5)) + ("task_eval.jsonl",)
FROM_RECORD = (
    "dair-ai/emotion dataset and exact-match metric scoring",
    "five disjoint 3000-row subsets of the 16000-row train split",
    "one fixed 300-row test sample, identical for every run",
    "file names train_s0.jsonl and task_eval.jsonl",
)
NEW_CHOICES = (
    "split config and caller-asserted revision",
    "SHA-256 selection rule and explicit seed",
    "exact-text deduplication, keeping the first label",
    "excluding all test text from train",
    "prompt template, label strings and row formats",
    "file names train_s1.jsonl through train_s4.jsonl",
)


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(path: Path, expected: str) -> list[dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("source SHA-256 must be lowercase hex")
    data = path.read_bytes()
    if _sha(data) != expected:
        raise ValueError("source hash mismatch")
    rows = []
    for line in data.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("each source row needs text and an integer label")
        if (not row["text"].strip() or type(row.get("label")) is not int
                or row["label"] not in range(6)):
            raise ValueError("empty text or invalid emotion label")
        rows.append({"text": row["text"], "label": row["label"]})
    return rows


def _select(rows: list[dict], split: str, seed: int, excluded: set[str]) -> list[int]:
    seen = set(excluded)
    indices = []
    for i, row in enumerate(rows):
        if row["text"] not in seen:
            seen.add(row["text"])
            indices.append(i)
    # SHA-256 ranking is stable across Python versions; no RNG-global state.
    return sorted(indices, key=lambda i: (_sha(_json([seed, split, i, rows[i]])), i))


def prepare(
    *, train: Path, test: Path, train_sha256: str, test_sha256: str,
    revision: str, seed: int, output: Path,
) -> dict:
    """Write five paired training files and held-out tasks without overwriting.

    Deduplicate exact source text and exclude the entire test split from train.
    Selection is a new deterministic reconstruction: the original seed, prompt
    template and indices were not published. Both training arms use these same
    five files. No adapter-init seed or historical accuracy claim follows.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable 40-character commit SHA")
    if type(seed) is not int:
        raise ValueError("seed must be an explicit integer")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists")
    train_rows = _source(train, train_sha256)
    test_rows = _source(test, test_sha256)
    train_ids = _select(train_rows, "train", seed, {row["text"] for row in test_rows})
    test_ids = _select(test_rows, "test", seed, set())
    if len(train_ids) < 15000 or len(test_ids) < 300:
        raise ValueError("need 15000 distinct train rows and 300 held-out rows")
    manifest: dict[str, Any] = {
        "schema_version": 1, "historical_bytes_recovered": False,
        "from_record": list(FROM_RECORD), "new_choices": list(NEW_CHOICES),
        "protocol": "h100-step11-step19-reconstruction-v1", "seed": seed,
        "selection": "sha256(seed,split,index,row); exact-text dedup; exclude all test text",
        "source": {"dataset": "dair-ai/emotion", "config": "split", "revision": revision,
                   "train_sha256": train_sha256, "test_sha256": test_sha256},
        "source_counts": {"train": len(train_rows), "test": len(test_rows)},
        "excluded_train_rows": len(train_rows) - len(train_ids),
        "labels": list(LABELS), "prompt_template": PROMPT, "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prep-convergence-", dir=output.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        for position, filename in enumerate(FILES):
            training = position < 5
            indices = (train_ids[position * 3000:(position + 1) * 3000]
                       if training else test_ids[:300])
            source_rows = train_rows if training else test_rows
            encoded = []
            for index in indices:
                row = source_rows[index]
                prompt = PROMPT.format(row["text"])
                label = LABELS[row["label"]]
                item = ({"messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": label}]} if training else
                        {"prompt": prompt, "expected": label, "scoring": "exact"})
                encoded.append(_json(item))
            data = b"".join(encoded)
            (stage / filename).write_bytes(data)
            manifest["files"][filename] = {
                "sha256": _sha(data), "rows": len(indices),
                "split": "train" if training else "test", "source_rows": indices,
            }
        (stage / "manifest.json").write_bytes(_json(manifest))
        verify(stage)
        stage.rename(output)
    return manifest


def verify(output: Path) -> dict:
    """Check protocol shape, byte hashes, row identities and text non-overlap.

    This checks self-consistency, not authenticity: preserve the manifest hash
    independently when transferring artifacts. Source origin is caller-asserted.
    """
    manifest = json.loads((output / "manifest.json").read_bytes())
    if (manifest.get("schema_version") != 1
            or manifest.get("historical_bytes_recovered") is not False):
        raise ValueError("unsupported reconstruction manifest")
    if (manifest.get("from_record") != list(FROM_RECORD)
            or manifest.get("new_choices") != list(NEW_CHOICES)):
        raise ValueError("invalid reconstruction provenance")
    if set(manifest["files"]) != set(FILES):
        raise ValueError("incomplete file set")
    seen_ids: set[int] = set()
    seen_prompts: set[str] = set()
    for i, filename in enumerate(FILES):
        entry = manifest["files"][filename]
        data = (output / filename).read_bytes()
        if _sha(data) != entry["sha256"]:
            raise ValueError("output hash mismatch")
        rows = [json.loads(line) for line in data.splitlines()]
        count = 3000 if i < 5 else 300
        indices = entry["source_rows"]
        split = "train" if i < 5 else "test"
        if (entry["rows"] != count or len(rows) != count or len(indices) != count
                or entry["split"] != split or len(set(indices)) != count
                or any(type(index) is not int or not 0 <= index < manifest["source_counts"][split]
                       for index in indices)):
            raise ValueError("invalid row counts or identities")
        if i < 5:
            if seen_ids.intersection(indices):
                raise ValueError("training subsets overlap")
            seen_ids.update(indices)
        for row in rows:
            if i < 5:
                messages = row["messages"]
                if [message["role"] for message in messages] != ["user", "assistant"]:
                    raise ValueError("invalid training roles")
                prompt, label = messages[0]["content"], messages[1]["content"]
            else:
                if row["scoring"] != "exact":
                    raise ValueError("held-out scoring must be exact")
                prompt, label = row["prompt"], row["expected"]
            if label not in LABELS or not isinstance(prompt, str) or prompt in seen_prompts:
                raise ValueError("invalid label or overlapping prompt")
            seen_prompts.add(prompt)
    return manifest


def run_self_test() -> None:
    """Exercise the full-size protocol with synthetic CPU-only source rows."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        options: dict[str, Any] = {"revision": "0" * 40, "seed": 17}
        for split, count in (("train", 16000), ("test", 2000)):
            path = root / f"{split}.jsonl"
            data = b"".join(_json({"text": f"{split} {i}", "label": i % 6}) for i in range(count))
            path.write_bytes(data)
            options[split] = path
            options[f"{split}_sha256"] = _sha(data)
        first = prepare(output=root / "a", **options)
        second = prepare(output=root / "b", **options)
        if first != second:
            raise ValueError("determinism control failed")
        (root / "a/train_s0.jsonl").write_bytes(b"tampered\n")
        try:
            verify(root / "a")
        except ValueError:
            return
        raise ValueError("tamper control failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", type=Path, metavar="OUTPUT")
    for name in ("train", "test", "output"):
        parser.add_argument(f"--{name}", type=Path)
    for name in ("train-sha256", "test-sha256", "revision"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    console = Console(highlight=False)
    try:
        if args.self_test:
            run_self_test()
        elif args.verify:
            verify(args.verify)
        else:
            options = vars(args).copy()
            del options["self_test"], options["verify"]
            if any(value is None for value in options.values()):
                parser.error("provide train/test JSONL, both SHA-256 values, "
                             "revision, seed and output")
            prepare(**options)
    except (ValueError, FileExistsError) as exc:
        console.print(f"Preparation failed: {exc}", markup=False, highlight=False)
        return 1
    except (OSError, KeyError, TypeError) as exc:
        console.print(f"Preparation failed: {type(exc).__name__}")
        return 1
    console.print("PASS: reconstruction data checks (no historical-quality claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
