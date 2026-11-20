#!/usr/bin/env python3
"""Write the three deterministic datasets of the ds_compare protocol (seed 42).

    ds_compare_rows_a.jsonl   64 alpaca rows, 32 tokens each      (Shape A, 256 steps)
    ds_compare_rows_b.jsonl   64 alpaca rows, 512 tokens each     (Shape B, 128 steps)
    ds_compare_rows_c.jsonl   64 alpaca rows, 2,048 tokens each   (Shape C, 64 steps)

"Tokens" are the length of ``input_ids`` after the chat template of the pinned model
(``ds_common.render_alpaca_row``, the same rendering ``soup train`` applies): the
template adds a begin token, a header per turn and a trailing assistant header, 15 tokens
for a user + assistant pair, so a Shape A row has 17 tokens of content.

The text is built from a fixed pool of 256 common English words, each one token (with and
without a capital and a leading space) in the Llama-3 tokenizer. Sentence lengths and word
choices come from ``random.Random(42)``; the instruction is one of 64 fixed sentences. The
output length is then fitted so that EVERY row has exactly the target length.

Without the tokenizer the files are still written (word counts from a constant per shape)
and the run says the token counts are UNMEASURED. The committed files were written with the
tokenizer present and ``--check`` re-measures them, so a run without a tokenizer leaves an
output directory that already holds measured files untouched unless ``--force`` is given.

    python make_rows.py                         # tokenizer from the local HF cache if present
    python make_rows.py --tokenizer <model dir> # tokenizer files of the pinned model revision
    python make_rows.py --check --tokenizer <dir>  # measure the committed files, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ds_common import (  # noqa: E402
    HERE,
    MODEL_ID,
    MODEL_REVISION,
    ROWS_PER_SHAPE,
    SEED,
    SHAPES,
    atomic_write_json,
    load_rows,
    render_alpaca_row,
)

# 256 words, each a single token in the Llama-3 tokenizer in all four spellings used here
# (" word", " Word", "Word", "word"); verified against the pinned revision by selftest.py.
WORD_POOL = (
    "the be to of and a in that have it for not on with he as you do at this but his by from "
    "they we say her she or an will my one all would there their what so up out if about who "
    "get which go me when make can like time no just know take people into year your good "
    "some could them see other than then now look only come its over think also back after "
    "use two how our work first well way even new want because any these give day most us is "
    "are was were been has had did does made each tell set three need house picture try again "
    "animal point mother world near build self earth father head stand own page should "
    "country found answer school grow study still learn plant cover food sun four between "
    "state keep eye never last let thought city tree cross farm hard start story far sea draw "
    "left late run while press close night real life few north open next white children begin "
    "got walk example ease paper group always music those both mark often letter until river "
    "car care second book science eat room friend fish stop once base cut sure watch color "
    "face wood main plain girl young ready above ever red list though feel talk bird soon "
    "body dog family direct pose leave song measure door product black short class wind "
    "question complete ship area half rock order fire south problem piece pass since top "
    "whole king space"
).split()

# 64 fixed topics; the instruction of row i is "Write a note about the <topic>.".
TOPICS = (
    "river mountain garden window bridge market forest island harbor village engine kitchen "
    "library station orchard meadow lantern compass blanket teacher farmer sailor painter "
    "baker miller weaver potter doctor pilot singer dancer writer builder runner climber "
    "swimmer thunder rainbow harvest winter summer autumn spring morning evening shadow "
    "mirror candle basket ladder hammer anchor saddle button ribbon feather pebble shell "
    "acorn cloud comet meteor canyon desert"
).split()

# Used only when no tokenizer is present: words in the output per shape (approximate).
UNMEASURED_WORDS = {"A": 9, "B": 438, "C": 1813}

MANIFEST_NAME = "ds_compare_rows_manifest.json"


def _sentence_boundaries(rng: random.Random, total_words: int) -> list[int]:
    """1-based word counts after which a sentence ends (5 to 12 words per sentence)."""
    boundaries: list[int] = []
    count = 0
    while count < total_words:
        count += rng.randint(5, 12)
        boundaries.append(count)
    return boundaries


def build_output(words: list[str], boundaries: set[int], count: int, final_period: bool) -> str:
    """The first ``count`` words, sentence breaks inside, optional period at the very end."""
    parts: list[str] = []
    capitalize = True
    for index in range(count):
        word = words[index]
        if capitalize:
            word = word.capitalize()
            capitalize = False
        parts.append(word)
        if (index + 1) in boundaries and index + 1 < count:
            parts[-1] += "."
            capitalize = True
    text = " ".join(parts)
    return text + "." if final_period else text


def _row(instruction: str, output: str) -> dict[str, str]:
    return {"instruction": instruction, "input": "", "output": output}


def _length(tokenizer: Any, row: dict[str, str]) -> int:
    return len(render_alpaca_row(tokenizer, row, max_length=1_000_000)["input_ids"])


def fit_row(
    tokenizer: Any,
    instruction: str,
    words: list[str],
    boundaries: set[int],
    target: int,
) -> dict[str, str]:
    """Largest word count whose row length is <= target, then a final period if one short.

    The row length is non-decreasing in the word count and grows by one token per word, or
    by two when the word is followed by a sentence break, so the largest count that fits
    lands on ``target`` or ``target - 1``; a final period closes the one-token gap.
    """
    low, high = 1, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        row = _row(instruction, build_output(words, boundaries, middle, False))
        if _length(tokenizer, row) <= target:
            low = middle
        else:
            high = middle - 1
    row = _row(instruction, build_output(words, boundaries, low, False))
    length = _length(tokenizer, row)
    if length == target - 1:
        row = _row(instruction, build_output(words, boundaries, low, True))
        length = _length(tokenizer, row)
    if length != target:
        raise ValueError(f"could not fit a row to {target} tokens (best {length})")
    return row


def make_shape(shape: str, tokenizer: Any | None) -> list[dict[str, str]]:
    spec = SHAPES[shape]
    rng = random.Random(f"ds_compare:{SEED}:{shape}")
    max_words = int(spec["target_tokens"]) + 64 if tokenizer is not None else 0
    rows: list[dict[str, str]] = []
    for index in range(ROWS_PER_SHAPE):
        instruction = f"Write a note about the {TOPICS[index]}."
        if tokenizer is None:
            count = UNMEASURED_WORDS[shape]
        else:
            count = max_words
        words = [rng.choice(WORD_POOL) for _ in range(count + 8)]
        boundaries = _sentence_boundaries(rng, len(words))
        boundary_set = set(boundaries)
        if tokenizer is None:
            rows.append(_row(instruction, build_output(words, boundary_set, count, True)))
        else:
            rows.append(fit_row(tokenizer, instruction, words, boundary_set, spec["target_tokens"]))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(location: str | None) -> tuple[Any | None, str]:
    """(tokenizer, note). Local files only: this script never downloads."""
    location = location or os.environ.get("DS_COMPARE_TOKENIZER")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, "transformers is not installed"
    try:
        if location:
            return AutoTokenizer.from_pretrained(location, local_files_only=True), location
        return (
            AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True),
            f"{MODEL_ID}@{MODEL_REVISION[:12]} (local cache)",
        )
    except Exception as exc:  # noqa: BLE001 - any load failure means "not available"
        return None, f"tokenizer not available locally ({type(exc).__name__})"


def measure(tokenizer: Any, path: Path, max_length: int) -> list[int]:
    return [
        len(render_alpaca_row(tokenizer, row, max_length)["input_ids"]) for row in load_rows(path)
    ]


def tokenizer_fingerprint(tokenizer: Any, location: str) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None) or ""
    info: dict[str, Any] = {
        "source": "model directory" if os.path.isdir(location) else location,
        "chat_template_sha256": hashlib.sha256(str(template).encode("utf-8")).hexdigest(),
        "vocab_size": len(tokenizer),
    }
    tokenizer_json = Path(location) / "tokenizer.json" if os.path.isdir(location) else None
    if tokenizer_json is not None and tokenizer_json.is_file():
        info["tokenizer_json_sha256"] = sha256_file(tokenizer_json)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", help="model directory (or repo id in the local cache)")
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--check", action="store_true", help="measure existing files, write none")
    parser.add_argument("--unmeasured", action="store_true", help="ignore any tokenizer")
    parser.add_argument(
        "--force", action="store_true", help="overwrite measured files with unmeasured ones"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    tokenizer: Any | None = None
    note = "tokenizer not requested"
    if not args.unmeasured:
        tokenizer, note = load_tokenizer(args.tokenizer)

    if args.check:
        if tokenizer is None:
            print(f"CHECK IMPOSSIBLE: {note}")
            return 2
        failed = False
        for shape, spec in SHAPES.items():
            path = out_dir / spec["rows_file"]
            lengths = measure(tokenizer, path, spec["max_length"])
            exact = len(lengths) == ROWS_PER_SHAPE and set(lengths) == {spec["target_tokens"]}
            print(
                f"shape {shape}: {path.name} rows={len(lengths)} tokens min={min(lengths)} "
                f"max={max(lengths)} target={spec['target_tokens']} -> "
                f"{'OK' if exact else 'MISMATCH'}"
            )
            failed = failed or not exact
        return 1 if failed else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / MANIFEST_NAME
    if tokenizer is None and existing.exists() and not args.force:
        try:
            was_measured = bool(json.loads(existing.read_text(encoding="utf-8")).get("measured"))
        except (OSError, ValueError):
            was_measured = False
        if was_measured:
            print(f"token counts of a regenerated text would be UNMEASURED ({note});")
            print(
                f"{out_dir} already holds measured files: left untouched (use --force to overwrite)"
            )
            return 0
    manifest: dict[str, Any] = {
        "generator": "benchmarks/harness/ds_compare/make_rows.py",
        "seed": SEED,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "measured": tokenizer is not None,
        "files": {},
    }
    if tokenizer is not None:
        manifest["tokenizer"] = tokenizer_fingerprint(tokenizer, note)
        print(f"tokenizer: {note}")
    else:
        print(f"token counts are UNMEASURED ({note}); word counts come from a constant per shape")

    for shape, spec in SHAPES.items():
        rows = make_shape(shape, tokenizer)
        path = out_dir / spec["rows_file"]
        write_jsonl(path, rows)
        entry: dict[str, Any] = {
            "shape": shape,
            "rows": len(rows),
            "sha256": sha256_file(path),
            "max_length": spec["max_length"],
            "target_tokens": spec["target_tokens"],
            "token_len": None,
        }
        if tokenizer is not None:
            lengths = measure(tokenizer, path, spec["max_length"])
            entry["token_len"] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
            }
            print(
                f"shape {shape}: {path.name} rows={len(rows)} tokens per row "
                f"min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.2f} "
                f"(target {spec['target_tokens']}, max_length {spec['max_length']}) "
                f"sha256={entry['sha256'][:16]}"
            )
        else:
            print(f"shape {shape}: {path.name} rows={len(rows)} tokens per row UNMEASURED")
        manifest["files"][path.name] = entry
    atomic_write_json(out_dir / MANIFEST_NAME, manifest)
    print(f"wrote {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
