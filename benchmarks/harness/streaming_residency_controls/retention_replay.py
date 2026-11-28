"""Independent bounded-memory replay of original numerical predicates and paired gates."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from rich.console import Console


def compare_traces(first, second):
    """Replay every element at original dtype/tolerance; no tensor omission.

    Original full-file comparator already runs remotely before timing. This local
    replay limits temporary double-difference arrays, retaining identical per-element
    finite/equality/allclose predicates and the original three-decimal loss check.
    """
    import torch
    from safetensors import safe_open

    groups, losses = {}, []
    with safe_open(first, framework="pt", device="cpu") as left, \
            safe_open(second, framework="pt", device="cpu") as right:
        assert set(left.keys()) == set(right.keys()), "trace keys differ"
        for name in left.keys():
            group = name.split("/")[0]
            stats = groups.setdefault(group, {"tensors": 0, "failed_tensors": 0,
                                             "max_abs": 0., "failures": []})
            stats["tensors"] += 1
            first_slice, second_slice = left.get_slice(name), right.get_slice(name)
            shape = first_slice.get_shape()
            assert shape == second_slice.get_shape()
            exact = group in {"batch", "initial"}
            axis = next((index for index, size in enumerate(shape) if size > 1), 0)
            width = max(1, 2**20 // max(1, math.prod(shape[axis + 1:]))) if shape else 1
            ranges = range(0, shape[axis], width) if shape else [0]
            passed, bitwise, maximum = True, True, 0.
            for offset in ranges:
                selection = [slice(None)] * len(shape)
                if shape:
                    selection[axis] = slice(offset, min(shape[axis], offset + width))
                a = first_slice[tuple(selection)] if shape else left.get_tensor(name)
                b = second_slice[tuple(selection)] if shape else right.get_tensor(name)
                equal = torch.equal(a, b)
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                close = equal if exact else torch.allclose(a, b, rtol=.01, atol=1e-5)
                passed = passed and finite and close
                bitwise = bitwise and equal
                maximum = max(maximum, float((a.double() - b.double()).abs().max()))
            if not passed:
                stats["failed_tensors"] += 1
                stats["failures"].append(name)
            stats["max_abs"] = max(stats["max_abs"], maximum)
            stats.setdefault("bitwise_equal", True)
            stats["bitwise_equal"] = stats["bitwise_equal"] and bitwise
        losses = [left.get_tensor("loss").tolist(), right.get_tensor("loss").tolist()]
    loss_match = all(f"{a:.3f}" == f"{b:.3f}" for a, b in zip(*losses, strict=True))
    return {"passed": bool(loss_match and all(row["failed_tensors"] == 0
                                              for row in groups.values())),
            "loss_three_decimals": loss_match, "groups": groups, "rtol": .01, "atol": 1e-5,
            "bitwise_equal": all(row["bitwise_equal"] for row in groups.values()),
            "all_original_elements_checked": True, "temporary_chunk_elements": 2**20}


def main(root, output, reference=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    paths, receipts = {}, []
    for source in ([reference, root] if reference else [root]):
        for name in ("accuracy", "timing-resident", "timing-retained", "timing-both", "final"):
            receipt = source / ("bundle-" + name + "-verified.json")
            if not receipt.exists():
                continue
            proof = json.loads(receipt.read_text())
            folder = source / "bundles" / name / "retained"
            manifest = json.loads((folder / "original-files.json").read_text())
            for relative in manifest["files"]:
                paths[relative] = folder / relative
            receipts.append(proof)

    def read(name):
        return json.loads(paths[name].read_text())

    state = read("results/progress.json")
    qualification = (read("results/qualification-reference.json")
                     if "results/qualification-reference.json" in paths else None)
    if qualification:
        environment = read("results/environment.json")
        gpu = environment["gpu"].split(", ")
        assert gpu[0] == qualification["gpu_uuid"] and gpu[2] == qualification["driver_version"]
        assert environment["source_sha256"] == qualification["source_sha256"]
        assert environment["packages"] == qualification["packages"]
        assert read("results/model-8-manifest.json") == qualification["model_manifest"]
    result = {"status": state["status"], "local_replay": True, "local_torch": torch.__version__,
              "source": "d500ae0dca7c4da905f8d59c9866af2f8aaa824b", "receipts": receipts,
              "captures": {}, "semantics": {}, "performance": {}, "remote": state}
    initial = {}
    changed = nonzero_gradients = 0
    with safe_open(paths["results/capture-baseline/trace.safetensors"],
                   framework="pt", device="cpu") as capture:
        for name in capture.keys():
            if name.startswith("initial/"):
                original_name = name.removeprefix("initial/")
                initial[original_name.replace(".default.", ".")] = capture.get_tensor(name).clone()
                changed += not torch.equal(capture.get_tensor(name),
                                           capture.get_tensor("final/" + original_name))
            elif name.startswith("gradient/"):
                nonzero_gradients += bool(torch.count_nonzero(capture.get_tensor(name)))
    assert len(initial) == 504 and changed > 0
    result["baseline_training_evidence"] = {
        "captured_optimizer_updates": 2, "adapter_tensors_changed": changed,
        "nonzero_captured_gradient_tensors": nonzero_gradients,
        "captured_gradient_tensors": 1008, "initial_adapter_tensors": len(initial)}
    for arm, report in state["arms"].items():
        if "capture" in report:
            verdict = compare_traces(paths["results/capture-baseline/trace.safetensors"],
                                     paths[f"results/capture-{arm}/trace.safetensors"])
            assert verdict["passed"] == report["capture"]["passed"]
            result["captures"][arm] = verdict
        if "semantics" in report:
            verdict = compare_traces(paths["results/semantics-baseline/trace.safetensors"],
                                     paths[f"results/semantics-{arm}/trace.safetensors"])
            assert verdict["passed"] == report["semantics"]["passed"]
            result["semantics"][arm] = verdict
        if report["status"] != "TIMED_REQUIRES_INDEPENDENT_ANALYSIS":
            continue
        pairs = []
        for pair in range(4):
            metrics, provenance, tensors = {}, {}, {}
            for current in ("baseline", arm):
                prefix = f"results/timing-{arm}-{pair}-{current}/"
                metrics[current] = read(prefix + "metrics.json")
                provenance[current] = read(prefix + "cache-provenance.json")
                config = read(prefix + "resolved-config.json")
                training, data = config["training"], config["data"]
                assert training["quantization"] == "none" and training["seed"] == 792
                assert training["batch_size"] == 1 and data["max_length"] == 1024
                assert training["gradient_accumulation_steps"] == 2
                assert not training["fast_lora"] and not training["use_liger"]
                assert training["lora"]["r"] == training["lora"]["alpha"] == 4
                assert training["lora"]["dropout"] == 0
                assert set(training["lora"]["target_modules"]) == {
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
                tensors[current] = load_file(
                    str(paths[prefix + "adapter/adapter_model.safetensors"]))
                item, info = metrics[current], provenance[current]
                assert info["arm"] == current and info["layer_count"] == 36
                expected_indices = ([] if current == "baseline"
                                    else state["selection"]["indices"][arm])
                assert info["indices"] == expected_indices
                if qualification:
                    assert item["data_sha256"] == qualification["data_sha256"]
                    assert item["initial_adapter_sha256"] == qualification["initial_adapter_sha256"]
                assert item["amp"] == {"bf16": True, "fp16": False}
                assert item["decoder_dtype"] == "torch.bfloat16"
                assert item["warmup"] == 6 and item["steps"] == 8
                assert item["deterministic_algorithms"] and not item["diagnostic_instrumented"]
                assert len(item["seconds"]) == 8
                assert all(shape == [1, 1024] for shape in item["input_shapes"])
                assert info["allocator_budget_bytes"] == 7 * 2**30
                assert info["whole_run_peak_reserved_bytes"] <= info["allocator_budget_bytes"]
                assert 7.5 * 2**30 <= info["physical_gpu_bytes"] <= 8.1 * 2**30
            a, b = metrics["baseline"], metrics[arm]
            assert a["data_sha256"] == b["data_sha256"]
            assert a["initial_adapter_sha256"] == b["initial_adapter_sha256"]
            first, second = tensors["baseline"], tensors[arm]
            assert set(first) == set(second) and len(first) == 504
            assert all(torch.isfinite(value).all() and torch.isfinite(second[name]).all()
                       and torch.allclose(value, second[name], rtol=.01, atol=1e-5)
                       for name, value in first.items())
            losses = [[row["loss"] for row in item["training_log_history"] if "loss" in row]
                      for item in (a, b)]
            assert all(len(values) == 14 for values in losses)
            assert all(math.isfinite(value) for values in losses for value in values)
            assert all(f"{x:.3f}" == f"{y:.3f}" for x, y in zip(*losses, strict=True))
            entry = {"pair": pair, "order": ("AB", "BA", "BA", "AB")[pair],
                     "time_reduction_percent": (1 - b["median_seconds_per_optimizer_step"]
                                                / a["median_seconds_per_optimizer_step"]) * 100,
                     "final_bitwise_equal": all(torch.equal(value, second[name])
                                                for name, value in first.items()),
                     "losses_bitwise_equal": losses[0] == losses[1], "trajectory_gate_passed": True}
            for current, item in metrics.items():
                info = provenance[current]
                seconds = item["median_seconds_per_optimizer_step"]
                assert set(tensors[current]) == set(initial)
                updated = sum(not torch.equal(value, initial[name])
                              for name, value in tensors[current].items())
                assert updated > 0, "saved adapter executed no observable update"
                entry[current] = {
                    "seconds": item["seconds"], "median_step_seconds": seconds,
                    "tokens_per_second": 2048 / seconds,
                    "peak_allocated_bytes": max(item["peak_allocated_bytes"]),
                    "peak_reserved_bytes": max(item["peak_reserved_bytes"]),
                    "host_memory": info["host_memory_at_exit"],
                    "whole_run_peak_allocated_bytes": info["whole_run_peak_allocated_bytes"],
                    "whole_run_peak_reserved_bytes": info["whole_run_peak_reserved_bytes"],
                    "resident_bytes": info["residency"]["resident_bytes"],
                    "retention": info.get("retention"), "per_update": info["per_update"],
                    "setup_seconds": item["setup_seconds"],
                    "saved_adapter_tensors_changed_from_exact_initial": updated,
                    "benchmark_body_seconds_excluding_imports": info["whole_process_seconds"],
                    "process_wall_seconds_including_imports": next(
                        stage["seconds"] for stage in state["stages"]
                        if stage["name"] == f"timing-{arm}-{pair}-{current}"),
                    "training_wall_seconds_including_final_save":
                        item["training_wall_seconds_including_final_save"],
                }
                telemetry_name = f"results/gpu-telemetry-timing-{arm}.jsonl"
                samples = [json.loads(line)
                           for line in paths[telemetry_name].read_text().splitlines()]
                start = info["started_unix"]
                samples = [row for row in samples
                           if start <= row["unix"] <= start + info["whole_process_seconds"]]
                assert samples, "missing sampled device telemetry for timing process"
                fields = [row["gpu"].split(", ") for row in samples]
                entry[current]["sampled_device"] = {
                    "scope": "benchmark body; approximately five-second samples, not exact peaks",
                    "samples": len(samples),
                    "max_used_mib": max(int(row[6].split()[0]) for row in fields),
                    "max_temperature_celsius": max(int(row[4]) for row in fields),
                    "sm_clock_mhz_range": [min(int(row[2].split()[0]) for row in fields),
                                           max(int(row[2].split()[0]) for row in fields)],
                    "max_visible_compute_processes": max(
                        len(row["processes"].strip().splitlines())
                        if row["processes"].strip() else 0 for row in samples),
                }
            pairs.append(entry)
        reductions = [row["time_reduction_percent"] for row in pairs]
        generator = random.Random(792)
        draws = sorted(statistics.median(generator.choices(reductions, k=4)) for _ in range(20000))
        median = statistics.median(reductions)
        positive = sum(value > 0 for value in reductions) / 4
        result["performance"][arm] = {
            "pairs": pairs, "median_paired_time_reduction_percent": median,
            "positive_pair_fraction": positive, "bootstrap_95_percent": [draws[499], draws[19499]],
            "practical_gate_passed": median >= 2 and positive >= .75 and draws[499] > 0}
    output.write_text(json.dumps(result, indent=2))
    Console().print({"status": result["status"], "capture_arms": list(result["captures"]),
                     "performance": {arm: value["median_paired_time_reduction_percent"]
                                     for arm, value in result["performance"].items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    main(args.root, args.output, args.reference)
