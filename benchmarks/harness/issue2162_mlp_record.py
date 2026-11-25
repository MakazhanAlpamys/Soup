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
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from issue2162_bulk import inspect, materialize
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


def extract_numerical_inputs(run: Path) -> dict:
    """Select the original B timing rows and memory peaks, without aggregation."""
    freeze = json.loads((run / "freeze.json").read_text())
    entries = freeze["timings"] + freeze["confirmation"]
    names = ([e["tag"] + ".json" for e in freeze["gates"]]
             + [e["tag"] + "-timing.json" for e in entries] + ["history-inputs.json"])
    return dict(
        format=1,
        original_files_sha256={name: sha((run / name).read_bytes()) for name in names},
        gate_peaks={
            e["tag"]: {key: json.loads((run / (e["tag"] + ".json")).read_text())[key]
                       for key in ("peak_allocated", "peak_reserved")}
            for e in freeze["gates"]
        },
        timings={e["tag"]: json.loads((run / (e["tag"] + "-timing.json")).read_text())
                 for e in entries},
        history=json.loads((run / "history-inputs.json").read_text()),
    )


def audit_numbers(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    reference = root.parent / "issue2162-rtx3090-v1/reference-updates.json"
    assert sha(reference.read_bytes().replace(b"\r\n", b"\n")) == manifest["reference_file_sha256"]
    data = json.loads((root / "numerical-inputs.json").read_text())
    freeze = json.loads((root / STUDIES[2] / "freeze.json").read_text())
    mapping = json.loads((root / STUDIES[2] / "files.json").read_text())
    original = json.loads((root / STUDIES[2] / "original-audit.json").read_text())
    assert data["format"] == 1 and freeze["family"] == 12
    assert freeze["warmup"] == 10 and freeze["measured_updates"] == 50
    names = ([e["tag"] + ".json" for e in freeze["gates"]]
             + [e["tag"] + "-timing.json" for e in freeze["timings"] + freeze["confirmation"]]
             + ["history-inputs.json"])
    assert set(data["original_files_sha256"]) == set(names)
    assert all(mapping[name] == value for name, value in data["original_files_sha256"].items())
    science = core("issue2162_mlp_budget_audit.py")
    import itertools
    import random

    orders = list(itertools.permutations(science.RAW))
    random.Random(42).shuffle(orders)
    assert freeze["gates"] == [
        dict(stage="gate", arm=arm, variant=variant, tag=f"gate-{arm}-{variant}")
        for variant in ("normal", "profile") for arm in science.RAW
    ]
    assert freeze["orders"] == [list(order) for order in orders]
    assert freeze["timings"] == [
        dict(stage="timing", arm=arm, block=i, tag=f"block{i}-{arm}")
        for i, order in enumerate(orders, 1) for arm in order
    ]
    assert freeze["confirmation"] == [
        dict(stage="confirmation", arm=arm, pair=i, tag=f"confirm{i}-{arm}")
        for i in range(1, 7)
        for arm in (("native", "mixed") if i % 2 else ("mixed", "native"))
    ]
    gates, timings = data["gate_peaks"], data["timings"]
    entries = freeze["timings"] + freeze["confirmation"]
    assert set(gates) == {e["tag"] for e in freeze["gates"]} and len(gates) == 6
    assert list(timings) == [e["tag"] for e in entries] and len(timings) == 30
    for entry in entries:
        timing = timings[entry["tag"]]
        assert timing["status"] == "ok" and timing["arm"] == entry["arm"]
        assert timing["initial_reference_sha256"] == freeze["reference_sha256"]
        assert timing["verified_initial_updates"] == 2
        assert timing["warmup_updates"] == 10 and timing["measured_updates"] == 50
        assert len(timing["step_times_s"]) == len(timing["losses"]) == 50
        assert all(math.isfinite(v) and v > 0 for v in timing["step_times_s"])
        assert all(math.isfinite(v) for v in timing["losses"])
        assert statistics.median(timing["step_times_s"]) == timing["median_step_s"]
    for value in [*gates.values(), *timings.values()]:
        assert 0 < value["peak_allocated"] <= value["peak_reserved"] <= science.CAP
    memories = []
    for variant in ("normal", "profile"):
        memories += science.memories({arm: gates[f"gate-{arm}-{variant}"]
                                      for arm in science.RAW})
    blocks = {}
    for i in range(1, 7):
        rows = {arm: timings[f"block{i}-{arm}"] for arm in science.RAW}
        memories += science.memories(rows)
        blocks[i] = {arm: row["median_step_s"] for arm, row in rows.items()}
    comparisons = {}
    for key in ("mixed_vs_native", "mixed_vs_qkvo", "qkvo_vs_native"):
        candidate, baseline = key.split("_vs_")
        comparisons[key] = science.summary([
            100 * (row[baseline] - row[candidate]) / row[baseline] for row in blocks.values()
        ])
    assert comparisons == original["stage_a"]
    eligible = (comparisons["mixed_vs_native"]["passes_two_percent"]
                and comparisons["mixed_vs_qkvo"]["positive_lower_bounds"]
                and all(row["passes"] for row in memories))
    values, checks = [], []
    for i in range(1, 7):
        rows = {arm: timings[f"confirm{i}-{arm}"] for arm in ("native", "mixed")}
        values.append(100 * (rows["native"]["median_step_s"] - rows["mixed"]["median_step_s"])
                      / rows["native"]["median_step_s"])
        checks += science.memories(rows)
    confirmation = science.summary(values, 73)
    confirmation["memory_checks"] = checks
    confirmation["passes_all"] = confirmation["passes_two_percent"] and all(
        row["passes"] for row in checks)
    assert confirmation == original["confirmation"] and eligible == original["stage_a_eligible"]
    memories += checks
    assert memories == original["memory_checks"]
    historical = []
    assert len(data["history"]) == 8
    for row in data["history"]:
        fresh = science.summary(row["reductions"])
        assert fresh["median_reduction_percent"] == row["original_median"]
        historical.append(dict(**row, new_family_summary=fresh))
    assert historical == original["history"]
    numerical_passes = bool(eligible and confirmation.pop("passes_all"))
    budget = dict(
        verified=False, gate_processes=6, timing_processes=30, timed_updates=1500,
        warmup_updates=300, total_optimizer_updates=1812, family=12,
        stage_a=comparisons, numerical_stage_a_eligible=eligible, stage_a_eligible=None,
        confirmation=confirmation, primary_passes=None, numerical_rule_passes=numerical_passes,
        memory_checks=memories, blocks=blocks, history=historical,
    )
    return dict(verified=False, restored_files=0, budget=budget,
                diagnostic=dict(verified=False,
                                status="requires objects.json; historical audit retained"))


def _audit_record_full(root: Path, git_root: Path | None = None) -> dict:
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
        assert extract_numerical_inputs(scratch / STUDIES[2]) == json.loads(
            (root / "numerical-inputs.json").read_text())
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


def audit_record(root: Path, git_root: Path | None = None, bulk: Path | None = None) -> dict:
    _, paths, missing = inspect(root, bulk, "files_sha256")
    if missing:
        result = audit_numbers(root)
    else:
        with materialize(root, paths) as complete:
            reference = complete.parent / "issue2162-rtx3090-v1/reference-updates.json"
            reference.parent.mkdir()
            shutil.copyfile(root.parent / "issue2162-rtx3090-v1/reference-updates.json", reference)
            result = _audit_record_full(complete, git_root)
    result.update(
        mode="numbers-only" if missing else "full", numbers_verified=True,
        missing_bulk_files=missing,
        unavailable_checks=[
            "Restore/hash all 281 original files and compare both full independent audit outputs.",
            "Bit-exact tensor replays, raw operators and diagnostic A mechanism/planning screen.",
            "Sample-by-sample admission, resource, thermal, CI and sampling gates.",
            "Full gate verdict, source inventories and Git blob bindings.",
        ] if missing else [],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--verify-git-sources", type=Path)
    parser.add_argument("--bulk", type=Path, help="extracted bulk archive root")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_record(args.evidence, args.verify_git_sources, args.bulk)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print({k: v for k, v in result.items() if k not in ("diagnostic", "budget")})


if __name__ == "__main__":
    main()
