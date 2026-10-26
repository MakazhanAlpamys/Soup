"""Restore and audit the MLP measurement record without models, CUDA or lab access.

Shared tensors and dictionary-encoded profiler strings are lossless: every restored
file must match its original SHA256 before either independent auditor reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

FORMAT = 1
STUDIES = ("mlp-diagnostic-v1", "mlp-diagnostic-v1-recovery", "mlp-budget-b-v1")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def destination(root: Path, name: str) -> Path:
    if not name or "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError("invalid evidence path")
    base = os.path.normcase(os.path.realpath(root))
    target = root / name
    if os.path.commonpath([base, os.path.normcase(os.path.realpath(target))]) != base:
        raise ValueError("evidence path escapes root")
    return target


def encode_file(name: str, raw: bytes, steps: list) -> dict:
    result = dict(sha256=sha(raw), shared_steps=False, event_dictionary=False)
    text = raw.decode("utf-8")
    if name.endswith(".jsonl") and text.strip():
        rows = [json.loads(line) for line in text.splitlines()]
        fields = list(rows[0])
        newline = "\r\n" if b"\r\n" in raw else "\n"
        if (
            all(list(row) == fields for row in rows)
            and "".join(json.dumps(row) + newline for row in rows).encode() == raw
        ):
            values = [[row[key] for key in fields] for row in rows]
            strings, ids = [], {}
            if "gpus" in fields:
                column = fields.index("gpus")
                for row in values:
                    encoded = []
                    for gpu in row[column]:
                        indices = []
                        for value in gpu:
                            assert isinstance(value, str)
                            if value not in ids:
                                ids[value] = len(strings)
                                strings.append(value)
                            indices.append(ids[value])
                        encoded.append(indices)
                    row[column] = encoded
            result.update(
                kind="jsonl", fields=fields, rows=values, gpu_strings=strings, newline=newline
            )
            assert decode_file(result, steps) == raw
            return result
    if name.endswith(".json"):
        value = json.loads(text)
        layouts = [
            (indent, ending, nl)
            for indent in (None, 2)
            for ending in ("", "\n")
            for nl in ("\n", "\r\n")
        ]
        layout = next(
            (
                x
                for x in layouts
                if (json.dumps(value, indent=x[0]) + x[1]).replace("\n", x[2]).encode() == raw
            ),
            None,
        )
        if layout is not None:
            result.update(kind="json", indent=layout[0], ending=layout[1], newline=layout[2])
            if isinstance(value, dict) and isinstance(value.get("steps"), list):
                assert value["steps"] == steps, "tensor reference changed"
                value["steps"] = None
                result["shared_steps"] = True
            if isinstance(value, dict) and "events" in value:
                strings, indices, rows = [], {}, []

                def intern(string):
                    if string not in indices:
                        indices[string] = len(strings)
                        strings.append(string)
                    return indices[string]

                for event in value["events"]:
                    assert list(event) == [
                        "name",
                        "ancestors",
                        "children",
                        "device_us",
                        "self_cpu_us",
                    ]
                    rows.append(
                        [
                            intern(event["name"]),
                            [intern(s) for s in event["ancestors"]],
                            [intern(s) for s in event["children"]],
                            event["device_us"],
                            event["self_cpu_us"],
                        ]
                    )
                value["events"] = dict(strings=strings, rows=rows)
                result["event_dictionary"] = True
            result["value"] = value
            assert decode_file(result, steps) == raw
            return result
    result.update(kind="text", text=text)
    return result


def decode_file(entry: dict, steps: list) -> bytes:
    if entry["kind"] == "text":
        raw = entry["text"].encode()
    elif entry["kind"] == "jsonl":
        rows = []
        for values in entry["rows"]:
            row = dict(zip(entry["fields"], values, strict=True))
            if "gpus" in row:
                row["gpus"] = [[entry["gpu_strings"][i] for i in gpu] for gpu in row["gpus"]]
            rows.append(json.dumps(row) + entry["newline"])
        raw = "".join(rows).encode()
    else:
        assert entry["kind"] == "json"
        value = json.loads(json.dumps(entry["value"]))
        if entry["shared_steps"]:
            assert value["steps"] is None
            value["steps"] = steps
        if entry["event_dictionary"]:
            strings, rows = value["events"]["strings"], value["events"]["rows"]
            value["events"] = [
                dict(
                    name=strings[row[0]],
                    ancestors=[strings[i] for i in row[1]],
                    children=[strings[i] for i in row[2]],
                    device_us=row[3],
                    self_cpu_us=row[4],
                )
                for row in rows
            ]
        raw = json.dumps(value, indent=entry["indent"]) + entry["ending"]
        raw = raw.replace("\n", entry["newline"]).encode()
    assert sha(raw) == entry["sha256"], "restored original bytes differ"
    return raw


def verify_sources(root: Path, git_root: Path) -> int:
    bindings = json.loads((root / "source-blobs.json").read_text())
    blobs = sorted({v["git_blob"] for group in bindings.values() for v in group["files"].values()})
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=git_root,
        check=True,
        input=("\n".join(blobs) + "\n").encode(),
        capture_output=True,
    )
    stream, contents = io.BytesIO(result.stdout), {}
    for blob in blobs:
        header = stream.readline().decode().split()
        assert len(header) == 3 and header[:2] == [blob, "blob"], "fetch the recorded Git blobs"
        contents[blob] = stream.read(int(header[2]))
        assert stream.read(1) == b"\n"
    count = 0
    for group in bindings.values():
        for item in group["files"].values():
            raw = contents[item["git_blob"]]
            assert item["newlines"] in ("identity", "crlf")
            if item["newlines"] == "crlf":
                raw = raw.replace(b"\n", b"\r\n")
            assert sha(raw) == item["sha256"]
            count += 1
    return count


def core(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_record(root: Path, git_root: Path | None = None) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format"] == FORMAT
    for name, expected in manifest["files_sha256"].items():
        assert sha(destination(root, name).read_bytes().replace(b"\r\n", b"\n")) == expected
    reference_file = root.parent / "issue2162-rtx3090-v1/reference-updates.json"
    assert (
        sha(reference_file.read_bytes().replace(b"\r\n", b"\n"))
        == manifest["reference_file_sha256"]
    )
    steps = json.loads(reference_file.read_text())["2048"]
    objects = json.loads((root / "objects.json").read_text())
    count = 0
    with tempfile.TemporaryDirectory(prefix="soup2162-mlp-audit-") as directory:
        scratch = Path(directory)
        for study in STUDIES:
            mapping = json.loads((root / study / "files.json").read_text())
            for name, identity in mapping.items():
                entry = objects[identity]
                assert entry["sha256"] == identity
                path = destination(scratch / study, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(decode_file(entry, steps))
                count += 1
        diagnostic = core("issue2162_mlp_diagnostic_audit.py").audit(scratch / STUDIES[1])
        budget = core("issue2162_mlp_budget_audit.py").audit(scratch / STUDIES[2])
        diagnostic, budget = json.loads(json.dumps([diagnostic, budget]))
        for study, value in [(STUDIES[1], diagnostic), (STUDIES[2], budget)]:
            assert value == json.loads((scratch / study / "independent-audit.json").read_text())
        assert (
            diagnostic["total_updates"] == 14
            and not diagnostic["eligible_to_prepare_timing_protocol"]
        )
        assert budget["primary_passes"] and budget["total_optimizer_updates"] == 1812
    result = dict(verified=True, restored_files=count, diagnostic=diagnostic, budget=budget)
    if git_root is not None:
        result["git_source_bindings_verified"] = verify_sources(root, git_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--verify-git-sources", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_record(args.evidence, args.verify_git_sources)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print({k: v for k, v in result.items() if k not in ("diagnostic", "budget")})


if __name__ == "__main__":
    main()
