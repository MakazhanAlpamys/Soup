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
_OFFLINE_MANIFEST_SCHEMA = "soup.best_of_n.offline_manifest.v1"
_VERIFIER_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_SHA256_VALUE = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: Any) -> str:
    data = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(data).hexdigest()


def sampler_identity_fingerprint(*parts: str) -> str:
    """Return a privacy-safe identity pin for a sampler source or endpoint."""
    if not parts or not all(isinstance(part, str) and part for part in parts):
        raise ValueError("sampler identity parts must be non-empty strings")
    return _sha({"identity": list(parts)})


def local_model_content_fingerprint(path: str) -> str:
    """Hash the exact regular-file content of a local model source.

    Logical relative names and bytes are included so replacing a file in the
    same directory cannot reuse an authenticated candidate checkpoint. Paths
    themselves are never returned or written to the public artifact.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("local model path must be a non-empty string")
    root = os.path.realpath(path)
    files: list[tuple[str, str]] = []
    if os.path.isfile(root):
        files.append((os.path.basename(root), root))
    elif os.path.isdir(root):
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            filenames.sort()
            for dirname in dirnames:
                if os.path.islink(os.path.join(directory, dirname)):
                    raise ValueError("local model path contains a symlinked directory")
            for filename in filenames:
                logical = os.path.relpath(os.path.join(directory, filename), root)
                files.append((logical.replace(os.sep, "/"), os.path.join(directory, filename)))
    else:
        raise ValueError("local model path must be a regular file or directory")

    digest = hashlib.sha256(b"soup.local-model-content.v1\0")
    for logical, source in files:
        resolved = os.path.realpath(source)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(resolved, flags)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("local model content must contain only regular files")
            name = logical.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(before.st_size.to_bytes(16, "big"))
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(fd)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if identity_before != identity_after:
                raise ValueError("local model content changed while it was fingerprinted")
        finally:
            os.close(fd)
    return digest.hexdigest()


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


def validate_sampler_spec(sampler: Any) -> dict:
    """Validate and return one public sampler specification."""
    return _validate_sampler(sampler)


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


def candidate_artifact_header(prompt_count: int, sampler: dict) -> dict:
    """Build the authenticated candidate-artifact header."""
    sampler = _validate_sampler(sampler)
    if isinstance(prompt_count, bool) or not isinstance(prompt_count, int) or prompt_count < 1:
        raise ValueError("candidate artifact prompt count is invalid")
    return {
        "_best_of_n_candidates": {
            "schema": _CANDIDATE_SCHEMA,
            "prompt_count": prompt_count,
            "sampler": sampler,
        }
    }


def stable_json_line(row: dict) -> bytes:
    """Encode one canonical JSONL record as exact UTF-8 bytes."""
    return _canonical(row) + b"\n"


def candidate_artifact_text(groups: list[dict], sampler: dict) -> str:
    header = candidate_artifact_header(len(groups), sampler)
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


def validate_candidate_group(
    group: Any,
    index: int,
    sampler: dict,
    *,
    expected_prompt: str | None = None,
    expected_source_line: int | None = None,
) -> str:
    """Authenticate one candidate group and return its prompt id."""
    if not isinstance(group, dict):
        raise ValueError(f"candidate group {index} must be an object")
    if group.get("prompt_index") != index or not isinstance(group.get("prompt"), str):
        raise ValueError("candidate groups must be sequential")
    source_line = group.get("source_line")
    if (
        isinstance(source_line, bool)
        or not isinstance(source_line, int)
        or source_line < 1
    ):
        raise ValueError(f"candidate group {index} has an invalid source line")
    if expected_source_line is not None and source_line != expected_source_line:
        raise ValueError(f"candidate group {index} does not match its source line")
    prompt = group["prompt"]
    if expected_prompt is not None and prompt != expected_prompt:
        raise ValueError(f"candidate group {index} does not match its source prompt")
    expected_id = _sha({"prompt_index": index, "prompt": prompt})
    if group.get("prompt_id") != expected_id:
        raise ValueError(f"candidate group {index} has an invalid prompt id")
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
    return expected_id


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
        prompt_id = validate_candidate_group(group, index, sampler)
        if prompt_id in seen_ids:
            raise ValueError(f"candidate group {index} has an invalid prompt id")
        seen_ids.add(prompt_id)
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


def validate_judgment(row: Any, group: dict, index: int) -> dict:
    """Validate one judgment against its exact candidate group."""
    if not isinstance(row, dict):
        raise ValueError(f"judgment {index} must be an object")
    if row.get("prompt_id") != group["prompt_id"]:
        raise ValueError(f"judgment {index} has a prompt id mismatch")
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
    return {
        "prompt_id": group["prompt_id"],
        "group_digest": group["group_digest"],
        "winner_idx": winner_idx,
        "scores": clean_scores,
        "verifier": _verifier(row.get("verifier"), index),
    }


def load_judgments(path: str, groups: list[dict]) -> tuple[list[dict], str]:
    """Validate complete one-to-one judgments against exact candidate groups."""
    data = _read_regular(path, "--judgments path")
    rows = _jsonl(data, "judgments")
    by_prompt: dict[str, dict] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id in by_prompt:
            raise ValueError("judgments contain a missing or duplicate prompt_id")
        by_prompt[prompt_id] = row
    if set(by_prompt) != {group["prompt_id"] for group in groups}:
        raise ValueError("judgments must cover every candidate group exactly once")

    validated = []
    for index, group in enumerate(groups):
        row = by_prompt[group["prompt_id"]]
        validated.append(validate_judgment(row, group, index))
    return validated, _sha(data)


def materialize_group(
    group: dict,
    judgment: dict,
    *,
    sampler: dict,
    candidate_artifact_sha256: str,
    judgments_sha256: str,
) -> tuple[dict, dict | None]:
    """Produce one SFT row and optional DPO row from authenticated records."""
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
    sft = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": winner},
        ],
        "_best_of_n": provenance,
    }
    dpo = None
    if loser_idx != winner_idx:
        dpo = {
            "prompt": prompt,
            "chosen": winner,
            "rejected": candidates[loser_idx]["text"],
            "_best_of_n": provenance,
        }
    return sft, dpo


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
        sft, dpo = materialize_group(
            group,
            judgment,
            sampler=sampler,
            candidate_artifact_sha256=candidate_artifact_sha256,
            judgments_sha256=judgments_sha256,
        )
        sft_rows.append(sft)
        if dpo is not None:
            dpo_rows.append(dpo)
    return sft_rows, dpo_rows


def stable_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def offline_manifest_text(
    *,
    candidate_artifact_sha256: str,
    judgments_sha256: str,
    sft_path: str,
    sft_bytes: bytes,
    sft_count: int,
    dpo_path: str,
    dpo_bytes: bytes,
    dpo_count: int,
) -> str:
    """Build the final commit marker for one offline materialization."""
    if not isinstance(sft_bytes, bytes) or not isinstance(dpo_bytes, bytes):
        raise TypeError("offline dataset payloads must be bytes")
    return offline_manifest_from_digests(
        candidate_artifact_sha256=candidate_artifact_sha256,
        judgments_sha256=judgments_sha256,
        sft_path=sft_path,
        sft_sha256=_sha(sft_bytes),
        sft_count=sft_count,
        dpo_path=dpo_path,
        dpo_sha256=_sha(dpo_bytes),
        dpo_count=dpo_count,
    )


def offline_manifest_from_digests(
    *,
    candidate_artifact_sha256: str,
    judgments_sha256: str,
    sft_path: str,
    sft_sha256: str,
    sft_count: int,
    dpo_path: str,
    dpo_sha256: str,
    dpo_count: int,
) -> str:
    """Build an offline commit marker from incrementally computed digests."""
    if not _SHA256_VALUE.fullmatch(candidate_artifact_sha256):
        raise ValueError("candidate artifact SHA-256 is invalid")
    if not _SHA256_VALUE.fullmatch(judgments_sha256):
        raise ValueError("judgments SHA-256 is invalid")
    if not _SHA256_VALUE.fullmatch(sft_sha256):
        raise ValueError("SFT SHA-256 is invalid")
    if not _SHA256_VALUE.fullmatch(dpo_sha256):
        raise ValueError("DPO SHA-256 is invalid")
    if isinstance(sft_count, bool) or not isinstance(sft_count, int) or sft_count < 0:
        raise ValueError("SFT row count is invalid")
    if isinstance(dpo_count, bool) or not isinstance(dpo_count, int) or dpo_count < 0:
        raise ValueError("DPO row count is invalid")
    dpo_requested = bool(dpo_path)
    manifest = {
        "schema": _OFFLINE_MANIFEST_SCHEMA,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "judgments_sha256": judgments_sha256,
        "dpo_requested": dpo_requested,
        "sft": {
            "file": os.path.basename(sft_path),
            "rows": sft_count,
            "sha256": sft_sha256,
        },
        "dpo": (
            {
                "file": os.path.basename(dpo_path),
                "rows": dpo_count,
                "sha256": dpo_sha256,
            }
            if dpo_requested
            else None
        ),
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _validate_manifest_dataset(
    record: Any, *, label: str, path: str, required: bool
) -> None:
    if not required:
        if record is not None or path:
            raise ValueError(f"offline manifest unexpectedly records {label}")
        return
    if not path:
        raise ValueError(f"offline manifest requires the {label} path")
    if not isinstance(record, dict) or set(record) != {"file", "rows", "sha256"}:
        raise ValueError(f"offline manifest {label} record is invalid")
    if record["file"] != os.path.basename(path):
        raise ValueError(f"offline manifest {label} filename does not match")
    rows = record["rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError(f"offline manifest {label} row count is invalid")
    if not isinstance(record["sha256"], str) or not _SHA256_VALUE.fullmatch(
        record["sha256"]
    ):
        raise ValueError(f"offline manifest {label} SHA-256 is invalid")
    actual_sha, actual_rows = _regular_sha_and_rows(path, f"{label} path")
    if record["sha256"] != actual_sha or rows != actual_rows:
        raise ValueError(f"offline manifest {label} content does not match")


def _regular_sha_and_rows(path: str, field: str) -> tuple[str, int]:
    """Hash and count non-empty JSONL records without buffering the dataset."""
    enforce_under_cwd_and_no_symlink(path, field)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"{field} could not be opened safely: {exc}") from exc
    digest = hashlib.sha256()
    rows = 0
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"{field} must be a regular file")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            for line in handle:
                digest.update(line)
                if line.strip():
                    rows += 1
    finally:
        if fd >= 0:
            os.close(fd)
    return digest.hexdigest(), rows


def _parse_offline_manifest(data: bytes) -> dict:
    """Parse and validate the bounded manifest envelope."""
    try:
        manifest = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("offline manifest is not valid UTF-8 JSON") from exc
    expected = {
        "schema",
        "candidate_artifact_sha256",
        "judgments_sha256",
        "dpo_requested",
        "sft",
        "dpo",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("offline manifest fields are invalid")
    if manifest["schema"] != _OFFLINE_MANIFEST_SCHEMA:
        raise ValueError("offline manifest schema is unsupported")
    for field in ("candidate_artifact_sha256", "judgments_sha256"):
        value = manifest[field]
        if not isinstance(value, str) or not _SHA256_VALUE.fullmatch(value):
            raise ValueError(f"offline manifest {field} is invalid")
    if not isinstance(manifest["dpo_requested"], bool):
        raise ValueError("offline manifest dpo_requested must be a bool")
    return manifest


def verify_offline_manifest(
    path: str, *, sft_path: str, dpo_path: str = ""
) -> dict:
    """Verify the final marker against exact SFT/DPO file bytes and counts."""
    manifest = _parse_offline_manifest(_read_regular(path, "--manifest path"))
    _validate_manifest_dataset(manifest["sft"], label="SFT", path=sft_path, required=True)
    _validate_manifest_dataset(
        manifest["dpo"],
        label="DPO",
        path=dpo_path,
        required=manifest["dpo_requested"],
    )
    return manifest


def find_committed_sibling_dpo(path: str, *, sft_path: str) -> str:
    """Return an authenticated prior DPO that sits beside ``path``.

    Offline manifests intentionally record public basenames rather than local
    absolute paths. A later SFT-only generation can therefore retire a prior
    DPO only when that exact, hash-bound file is beside the manifest. Outputs
    elsewhere remain unlisted and must not be inferred by consumers.
    """
    manifest = _parse_offline_manifest(_read_regular(path, "--manifest path"))
    if manifest.get("dpo_requested") is False:
        verify_offline_manifest(path, sft_path=sft_path)
        return ""
    record = manifest.get("dpo")
    if not isinstance(record, dict) or set(record) != {"file", "rows", "sha256"}:
        raise ValueError("offline manifest DPO record is invalid")
    filename = record.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or os.path.basename(filename) != filename
    ):
        raise ValueError("offline manifest DPO filename is invalid")
    sibling = os.path.join(os.path.dirname(os.path.abspath(path)), filename)
    if not os.path.lexists(sibling):
        return ""
    verify_offline_manifest(path, sft_path=sft_path, dpo_path=sibling)
    return sibling
