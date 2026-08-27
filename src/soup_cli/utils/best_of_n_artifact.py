"""Content-addressed artifacts for two-phase Best-of-N workflows."""

from __future__ import annotations

import hashlib
import json
import math
import ntpath
import os
import re
import stat
from typing import Any

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

_CANDIDATE_SCHEMA = "soup.best_of_n.candidates.v1"
_VERIFIER_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: Any) -> str:
    data = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(data).hexdigest()


def _validate_sampler(sampler: Any) -> dict:
    if not isinstance(sampler, dict):
        raise ValueError("candidate sampler specification must be an object")
    kind = sampler.get("kind")
    common = {"kind", "model", "n", "temperature", "max_new_tokens"}
    expected = (
        common | {"provider"}
        if kind == "provider"
        else common | {"revision", "device", "seed", "trust_remote_code"}
    )
    if kind not in {"provider", "local"} or set(sampler) != expected:
        raise ValueError("candidate sampler specification has unsupported fields")
    model = sampler.get("model")
    if (
        not isinstance(model, str)
        or not model
        or len(model) > 256
        or os.path.isabs(model)
        or ntpath.isabs(model)
        or "\\" in model
        or model.startswith(("./", "../", "~/"))
    ):
        raise ValueError("candidate sampler model must be a public identifier")
    n = sampler.get("n")
    temperature = sampler.get("temperature")
    max_new_tokens = sampler.get("max_new_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 64:
        raise ValueError("candidate sampler n is invalid")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("candidate sampler temperature is invalid")
    try:
        temperature_value = float(temperature)
    except (OverflowError, ValueError) as exc:
        raise ValueError("candidate sampler temperature is invalid") from exc
    if not math.isfinite(temperature_value) or not 0 <= temperature_value <= 2:
        raise ValueError("candidate sampler temperature is invalid")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 4096
    ):
        raise ValueError("candidate sampler max_new_tokens is invalid")
    if kind == "provider":
        if sampler.get("provider") not in {"ollama", "vllm"}:
            raise ValueError("candidate sampler provider is invalid")
    else:
        revision = sampler.get("revision")
        if (
            not isinstance(revision, str)
            or not revision
            or len(revision) > 256
            or os.path.isabs(revision)
            or ntpath.isabs(revision)
            or "\\" in revision
            or revision.startswith(("./", "../", "~/"))
        ):
            raise ValueError("candidate sampler revision is invalid")
        device = sampler.get("device")
        if not isinstance(device, str) or not re.fullmatch(
            r"(?:auto|cpu|mps|cuda(?::\d+)?)", device
        ):
            raise ValueError("candidate sampler device is invalid")
        seed = sampler.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("candidate sampler seed is invalid")
        if not isinstance(sampler.get("trust_remote_code"), bool):
            raise ValueError("candidate sampler trust flag is invalid")
    return sampler


def build_candidate_group(
    prompt: str,
    prompt_index: int,
    candidates: list[str],
    sampler: dict,
    *,
    source_line: int,
) -> dict:
    """Build one self-validating, ordered candidate group."""
    sampler = _validate_sampler(sampler)
    if (
        isinstance(source_line, bool)
        or not isinstance(source_line, int)
        or source_line < 1
    ):
        raise ValueError("candidate source line must be a positive integer")
    if len(candidates) != sampler["n"] or not all(
        isinstance(candidate, str) for candidate in candidates
    ):
        raise ValueError("sampled candidates do not match the sampler specification")
    prompt_id = _sha({"prompt_index": prompt_index, "prompt": prompt})
    candidate_records = [
        {"index": index, "text": text, "sha256": _sha(text.encode("utf-8"))}
        for index, text in enumerate(candidates)
    ]
    core = {
        "prompt_index": prompt_index,
        "source_line": source_line,
        "prompt_id": prompt_id,
        "prompt_sha256": _sha(prompt.encode("utf-8")),
        "prompt": prompt,
        "candidates": candidate_records,
        "sampler": sampler,
    }
    return {**core, "group_digest": _sha(core)}


def candidate_artifact_text(groups: list[dict], sampler: dict) -> str:
    sampler = _validate_sampler(sampler)
    header = {
        "_best_of_n_candidates": {
            "schema": _CANDIDATE_SCHEMA,
            "prompt_count": len(groups),
            "sampler": sampler,
        }
    }
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in [header, *groups]
    )


def _read_regular(path: str, field: str) -> bytes:
    enforce_under_cwd_and_no_symlink(path, field)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{field} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _jsonl(data: bytes, field: str) -> list[dict]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} must be UTF-8") from exc
    rows = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} has invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{field} line {line_number} must be an object")
        rows.append(row)
    return rows


def load_candidate_artifact(path: str) -> tuple[list[dict], dict, str]:
    """Authenticate every group and return groups, public sampler spec, file hash."""
    data = _read_regular(path, "--candidate-artifact path")
    rows = _jsonl(data, "candidate artifact")
    if not rows:
        raise ValueError("candidate artifact is empty")
    header = rows[0].get("_best_of_n_candidates")
    if not isinstance(header, dict) or header.get("schema") != _CANDIDATE_SCHEMA:
        raise ValueError("candidate artifact header or schema is invalid")
    sampler = _validate_sampler(header.get("sampler"))
    groups = rows[1:]
    if not isinstance(sampler, dict) or header.get("prompt_count") != len(groups):
        raise ValueError("candidate artifact header does not match its groups")
    if not groups:
        raise ValueError("candidate artifact contains no prompt groups")
    seen_ids: set[str] = set()
    for index, group in enumerate(groups):
        if group.get("prompt_index") != index or not isinstance(group.get("prompt"), str):
            raise ValueError("candidate groups must be sequential")
        source_line = group.get("source_line")
        if (
            isinstance(source_line, bool)
            or not isinstance(source_line, int)
            or source_line < 1
        ):
            raise ValueError(f"candidate group {index} has an invalid source line")
        prompt = group["prompt"]
        expected_id = _sha({"prompt_index": index, "prompt": prompt})
        if group.get("prompt_id") != expected_id or expected_id in seen_ids:
            raise ValueError(f"candidate group {index} has an invalid prompt id")
        seen_ids.add(expected_id)
        if group.get("prompt_sha256") != _sha(prompt.encode("utf-8")):
            raise ValueError(f"candidate group {index} has a prompt digest mismatch")
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != sampler["n"]:
            raise ValueError(f"candidate group {index} has the wrong candidate count")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or candidate.get("index") != candidate_index:
                raise ValueError(f"candidate group {index} has unordered candidates")
            text = candidate.get("text")
            if not isinstance(text, str) or candidate.get("sha256") != _sha(text.encode("utf-8")):
                raise ValueError(f"candidate group {index} has a candidate digest mismatch")
        core = {key: value for key, value in group.items() if key != "group_digest"}
        if group.get("sampler") != sampler or group.get("group_digest") != _sha(core):
            raise ValueError(f"candidate group {index} has a group digest mismatch")
    return groups, sampler, _sha(data)


def _verifier(value: Any, index: int) -> dict[str, str]:
    if not isinstance(value, dict) or "name" not in value:
        raise ValueError(f"judgment {index} needs verifier.name")
    if set(value) - {"name", "version", "method"}:
        raise ValueError(f"judgment {index} has unsupported verifier fields")
    clean: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str) or not _VERIFIER_VALUE.fullmatch(item):
            raise ValueError(f"judgment {index} has an invalid verifier {key}")
        clean[key] = item
    return clean


def load_judgments(path: str, groups: list[dict]) -> tuple[list[dict], str]:
    """Validate complete one-to-one judgments against exact candidate groups."""
    data = _read_regular(path, "--judgments path")
    rows = _jsonl(data, "judgments")
    by_prompt: dict[str, dict] = {}
    for index, row in enumerate(rows):
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id in by_prompt:
            raise ValueError("judgments contain a missing or duplicate prompt_id")
        by_prompt[prompt_id] = row
    if set(by_prompt) != {group["prompt_id"] for group in groups}:
        raise ValueError("judgments must cover every candidate group exactly once")

    validated = []
    for index, group in enumerate(groups):
        row = by_prompt[group["prompt_id"]]
        if row.get("group_digest") != group["group_digest"]:
            raise ValueError(f"judgment {index} has a candidate digest mismatch")
        winner_idx = row.get("winner_idx")
        candidates = group["candidates"]
        if isinstance(winner_idx, bool) or not isinstance(winner_idx, int):
            raise ValueError(f"judgment {index} winner_idx must be an integer")
        if not 0 <= winner_idx < len(candidates):
            raise ValueError(f"judgment {index} winner_idx is out of range")
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError(f"judgment {index} scores must match candidate count")
        clean_scores = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"judgment {index} scores must be numeric")
            try:
                number = float(score)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"judgment {index} scores must be finite") from exc
            if not math.isfinite(number):
                raise ValueError(f"judgment {index} scores must be finite")
            clean_scores.append(number)
        expected_winner = max(range(len(clean_scores)), key=clean_scores.__getitem__)
        if winner_idx != expected_winner:
            raise ValueError(f"judgment {index} winner_idx does not match its scores")
        validated.append(
            {
                "prompt_id": group["prompt_id"],
                "group_digest": group["group_digest"],
                "winner_idx": winner_idx,
                "scores": clean_scores,
                "verifier": _verifier(row.get("verifier"), index),
            }
        )
    return validated, _sha(data)


def materialize_rows(
    groups: list[dict],
    judgments: list[dict],
    *,
    sampler: dict,
    candidate_artifact_sha256: str,
    judgments_sha256: str,
) -> tuple[list[dict], list[dict]]:
    """Produce byte-stable SFT and DPO rows from authenticated offline inputs."""
    sft_rows = []
    dpo_rows = []
    for group, judgment in zip(groups, judgments):
        candidates = group["candidates"]
        winner_idx = judgment["winner_idx"]
        loser_idx = min(range(len(candidates)), key=judgment["scores"].__getitem__)
        provenance = {
            "mode": "offline",
            "n": len(candidates),
            "source_line": group["source_line"],
            "winner_idx": winner_idx,
            "scores": judgment["scores"],
            "prompt_id": group["prompt_id"],
            "candidate_group_digest": group["group_digest"],
            "candidate_artifact_sha256": candidate_artifact_sha256,
            "judgments_sha256": judgments_sha256,
            "sampler": sampler,
            "verifier": judgment["verifier"],
        }
        prompt = group["prompt"]
        winner = candidates[winner_idx]["text"]
        sft_rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": winner},
                ],
                "_best_of_n": provenance,
            }
        )
        if loser_idx != winner_idx:
            dpo_rows.append(
                {
                    "prompt": prompt,
                    "chosen": winner,
                    "rejected": candidates[loser_idx]["text"],
                    "_best_of_n": provenance,
                }
            )
    return sft_rows, dpo_rows


def stable_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
