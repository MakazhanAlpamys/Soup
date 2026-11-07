"""Compact measurement evidence must retain exactness and reject altered records."""

import copy
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "benchmarks/harness/issue2050_audit.py"
    spec = importlib.util.spec_from_file_location("issue2050_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt():
    names = [f"adapter.{index}" for index in range(128)]
    steps = [
        {
            "logits": "a" * 64,
            "loss": "b" * 64,
            "gradients": {name: "c" * 64 for name in names},
            "parameters": {name: "d" * 64 for name in names},
        }
        for _ in range(2)
    ]
    return {"steps": steps, "operators": {}, "status": "ok"}


def test_compact_digest_covers_every_tensor_name_value_and_update():
    module = _module()
    original = _receipt()
    compact = module.compact_gate(original)
    assert compact["tensor_hash_count"] == 516
    assert "gradients" not in compact["step_summaries"][0]
    for step, category, key in (
        (0, "gradients", "adapter.127"),
        (1, "parameters", "adapter.0"),
    ):
        changed = copy.deepcopy(original)
        changed["steps"][step][category][key] = "e" * 64
        assert module.compact_gate(changed)["tensor_hashes_sha256"] != compact[
            "tensor_hashes_sha256"
        ]
    reordered = copy.deepcopy(original)
    reordered["steps"][0]["gradients"] = dict(
        reversed(list(reordered["steps"][0]["gradients"].items()))
    )
    assert module.compact_gate(reordered) == compact


@pytest.mark.parametrize("mutation", ["count", "name", "hash"])
def test_compaction_refuses_incomplete_or_malformed_tensor_hashes(mutation):
    receipt = _receipt()
    if mutation == "count":
        receipt["steps"][0]["gradients"].pop("adapter.127")
    elif mutation == "name":
        receipt["steps"][0]["parameters"]["other"] = receipt["steps"][0]["parameters"].pop(
            "adapter.127"
        )
    else:
        receipt["steps"][1]["parameters"]["adapter.0"] = "x" * 64
    with pytest.raises(AssertionError):
        _module().compact_gate(receipt)


def test_source_binding_accepts_checkout_newlines_but_rejects_changed_code(tmp_path):
    module = _module()
    raw = b"value = 1\r\n"
    blob = raw.replace(b"\r\n", b"\n")
    name = "source.py"
    binding = {
        name: {
            "git_blob": hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest(),
            "frozen_newlines": "crlf",
        }
    }
    freeze = {name: hashlib.sha256(raw).hexdigest()}
    for content in (raw, blob):
        (tmp_path / name).write_bytes(content)
        module.verify_source(tmp_path, freeze, binding)
    (tmp_path / name).write_bytes(b"value = 2\n")
    with pytest.raises(AssertionError, match="source"):
        module.verify_source(tmp_path, freeze, binding)


def test_compact_record_reproduces_original_full_scientific_audit():
    evidence = Path(__file__).parents[1] / "benchmarks/results/issue2050-rtx3090-v1"
    expected = json.loads((evidence / "independent-audit.json").read_text())
    actual = _module().audit(evidence)

    def assert_same_record(left, right, path="audit"):
        assert type(left) is type(right), path
        if isinstance(left, dict):
            assert left.keys() == right.keys(), path
            for key in left:
                assert_same_record(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            assert len(left) == len(right), path
            for index, (item, reference) in enumerate(zip(left, right)):
                assert_same_record(item, reference, f"{path}[{index}]")
        elif isinstance(left, float):
            # Python 3.12 changed sum() rounding. The original audit was saved
            # on 3.12; 3.11 differs only at floating-point roundoff scale.
            assert math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-12), path
        else:
            assert left == right, path

    assert_same_record(actual, expected)


@pytest.mark.parametrize("mutation", ["digest", "comparison", "saved_bytes", "timing_row"])
def test_compact_audit_rejects_tampering(tmp_path, mutation):
    source = Path(__file__).parents[1] / "benchmarks/results/issue2050-rtx3090-v1"
    shutil.copytree(source, tmp_path / "evidence")
    evidence = tmp_path / "evidence"
    path = evidence / "process-evidence.json"
    record = json.loads(path.read_text())
    process = record["processes"]["s2048-qkvo"]
    if mutation == "digest":
        process["tensor_hashes_sha256"] = "0" * 64
    elif mutation == "comparison":
        process["matches_reference"] = False
    elif mutation == "saved_bytes":
        process["saved_bytes"] = 0
    else:
        row_path = evidence / "s2048-qkvo-pairs/p0-B.json"
        row = json.loads(row_path.read_text())
        row["step_times_s"][0] += 1
        row_path.write_text(json.dumps(row))
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        _module().audit(evidence)
