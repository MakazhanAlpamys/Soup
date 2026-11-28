"""Read-only input-text overlap between training and evaluation rows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from soup_cli.utils.row_text import extract_row_text

_ASSISTANT = frozenset({"assistant", "gpt", "chatgpt", "bot", "model"})


NEAR_MAX_ROWS = 10_000
NEAR_MAX_FILE_BYTES = 16 * 1024**2
_NEAR_MAX_JACCARD_WORK = 50_000_000
_NEAR_LIMIT_HELP = "Use a smaller split or remove --near-duplicates to run the exact check."
_THRESHOLD_ERROR = "--threshold must be a finite number greater than 0 and at most 1"
_PERMUTATIONS_ERROR = "--permutations must be an integer from 16 to 256"
_SEED_ERROR = "--seed must be an integer from 0 to 4294967295"


@dataclass(frozen=True)
class NearDuplicateOptions:
    threshold: float = 0.8
    permutations: int = 128
    seed: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError(_THRESHOLD_ERROR)
        if not isinstance(self.permutations, int) or not 16 <= self.permutations <= 256:
            raise ValueError(_PERMUTATIONS_ERROR)
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError(_SEED_ERROR)

    @classmethod
    def from_cli(
        cls, threshold: str | None, permutations: str | None, seed: str | None,
    ) -> "NearDuplicateOptions":
        values: dict[str, Any] = {}
        for name, raw, convert, error in (
            ("threshold", threshold, float, _THRESHOLD_ERROR),
            ("permutations", permutations, int, _PERMUTATIONS_ERROR),
            ("seed", seed, int, _SEED_ERROR),
        ):
            if raw is not None:
                try:
                    values[name] = convert(raw)
                except ValueError as exc:
                    raise ValueError(error) from exc
        return cls(**values)


def load_near_rows(path: str, option: str) -> list[Mapping[str, Any]]:
    """Apply the tighter lexical limits before growing the index."""
    from soup_cli.utils.data_score import JsonlLimitError, load_jsonl_rows

    try:
        return load_jsonl_rows(
            path, strict=True, max_file_bytes=NEAR_MAX_FILE_BYTES, max_rows=NEAR_MAX_ROWS,
        )
    except JsonlLimitError as exc:
        raise ValueError(f"{option} in --near-duplicates mode: {exc}. {_NEAR_LIMIT_HELP}") from exc


def _word_shingles(text: str) -> set[bytes]:
    # Same lowercased whitespace tokens and short-text fallback as data dedup.
    words = text.lower().split()
    return {
        " ".join(words[i : i + 3]).encode("utf-8", errors="surrogatepass")
        for i in range(max(1, len(words) - 2))
    }


class _NearIndex:
    def __init__(self, index: Mapping[str, tuple[int, int]], options: NearDuplicateOptions):
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError as exc:
            raise ValueError(
                '--near-duplicates requires datasketch. '
                'Install it with pip install "soup-cli[data]"'
            ) from exc

        self.options = options
        self._empty_hash = MinHash(num_perm=options.permutations, seed=options.seed)
        # A more permissive lookup helps recall and admits at least two bands
        # even with 16 permutations. The exact Jaccard check decides all hits.
        self._lsh = MinHashLSH(
            threshold=min(options.threshold, 0.75), num_perm=options.permutations,
        )
        self._rows: dict[int, tuple[int, set[bytes]]] = {}
        self._remaining_work = _NEAR_MAX_JACCARD_WORK
        for text, (first, count) in index.items():
            shingles = _word_shingles(text)
            self._rows[first] = (count, shingles)
            self._lsh.insert(first, self._minhash(shingles))

    def _minhash(self, shingles: set[bytes]) -> Any:
        signature = self._empty_hash.copy()
        hashes = iter(shingles)
        # Bound the temporary shingle-by-permutation array for long records.
        while batch := list(islice(hashes, 256)):
            signature.update_batch(batch)
        return signature

    def find(self, text: str) -> tuple[int, int, float] | None:
        shingles = _word_shingles(text)
        best_row, matches, best_score = 0, 0, 0.0
        for row in self._lsh.query(self._minhash(shingles)):
            count, candidate = self._rows[row]
            self._remaining_work -= min(len(shingles), len(candidate))
            if self._remaining_work < 0:
                raise ValueError(
                    f"--near-duplicates exceeded {_NEAR_MAX_JACCARD_WORK} shingle comparisons. "
                    f"{_NEAR_LIMIT_HELP}"
                )
            shared = len(shingles & candidate)
            score = shared / (len(shingles) + len(candidate) - shared)
            if score >= self.options.threshold:
                matches += count
                if score > best_score or (score == best_score and row < best_row):
                    best_row, best_score = row, score
        if matches:
            return best_row, matches, best_score
        return None


def _without_response(messages: object) -> list:
    if not isinstance(messages, list) or any(not isinstance(x, Mapping) for x in messages):
        raise ValueError("conversation must be a list of message objects")
    if messages and (messages[-1].get("role") or messages[-1].get("from")) in _ASSISTANT:
        return messages[:-1]
    return messages


def _input_text(row: Mapping[str, Any]) -> str:
    """Reuse the shared text extractor on input fields, excluding the final answer."""
    if not isinstance(row, Mapping):
        raise TypeError("record must be a JSON object")
    if "prompt" in row:
        fields = {"prompt": row["prompt"]}
        if "system" in row:
            fields["system"] = row["system"]
    elif "instruction" in row:
        fields = {key: row[key] for key in ("instruction", "input", "system") if key in row}
    elif "messages" in row or "conversations" in row:
        key = "messages" if "messages" in row else "conversations"
        fields = {key: _without_response(row[key])}
    elif "chosen" in row and "rejected" in row:
        chosen = _without_response(row["chosen"])
        rejected = _without_response(row["rejected"])
        if chosen != rejected:
            raise ValueError("chosen/rejected conversations do not have the same prompt")
        fields = {"messages": chosen}
    else:
        fields = {key: row[key] for key in ("query", "text", "content") if key in row}
    text = extract_row_text(fields).strip()
    if not text:
        raise ValueError("record has no usable input text")
    return text


def check_leakage(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    *,
    samples: int = 5,
    near: NearDuplicateOptions | None = None,
) -> dict[str, Any]:
    """Count eval records matching any train input, with bounded example pairs.

    Record numbers are one-based and exclude blank JSONL lines. Each eval record
    counts once even if several train records match. Only outer whitespace is
    stripped for exact matches. The opt-in lexical index also checks near matches.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or not 0 <= samples <= 20:
        raise ValueError("samples must be an integer in [0, 20]")
    if not train or not evaluation:
        raise ValueError("train and eval must both contain records")
    index: dict[str, tuple[int, int]] = {}
    for number, row in enumerate(train, 1):
        try:
            text = _input_text(row)
        except (TypeError, ValueError) as error:
            raise ValueError(f"train record {number}: {error}") from error
        first, count = index.get(text, (number, 0))
        index[text] = (first, count + 1)
    near_index = _NearIndex(index, near) if near is not None else None
    matched = 0
    examples = []
    for number, row in enumerate(evaluation, 1):
        try:
            text = _input_text(row)
        except (TypeError, ValueError) as error:
            raise ValueError(f"eval record {number}: {error}") from error
        exact = index.get(text)
        if exact is not None:
            first, count = exact
            method, score = "exact", 1.0
        else:
            lexical = near_index.find(text) if near_index is not None else None
            if lexical is None:
                continue
            first, count, score = lexical
            method = "near"
        matched += 1
        if len(examples) < samples:
            pair = {
                "train_row": first,
                "eval_row": number,
                "train_matches": count,
                "prompt_preview": text[:160],
            }
            if near is not None:
                pair.update(method=method, score=score)
            examples.append(pair)
    report = {
        "schema_version": 1,
        "matching": "exact extracted input text; outer whitespace stripped",
        "row_numbers": "one-based records, excluding blank lines",
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "overlap_eval_rows": matched,
        "overlap_percent": 100.0 * matched / len(evaluation),
        "samples": examples,
    }
    if near is not None:
        report["matching"] = "exact or lexical near-duplicate (word shingles and Jaccard)"
        report["near_duplicates"] = {
            "threshold": near.threshold,
            "permutations": near.permutations,
            "seed": near.seed,
        }
    return report
