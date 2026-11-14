"""Independent receipt audit; no experiment modules or GPU imports."""

import hashlib
import json
from pathlib import Path

CAP = 8 * 2**30


def digest(path):

    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run: Path):
    root = run.parent

    freeze = json.loads((run / "freeze.json").read_text())

    amendment = json.loads((run / "recovery-amendment.json").read_text())

    assert digest(run / "recovery-amendment.json") == freeze["recovery_amendment_sha256"]

    assert digest(run / "original-freeze.json") == amendment["original_freeze_sha256"]

    assert digest(run / "bridge-source.py") == amendment["bridge_sha256"]

    assert amendment["completed_processes_reused"] == 3 and amendment["failed_profile_retry"] == 1

    assert amendment["prior_failed_updates"] == 2

    assert amendment["accepted_updates_expected"] == 12

    assert amendment["total_updates_including_failed_expected"] == 14

    assert not amendment["model_math_changed"] and not amendment["frozen_original_source_changed"]

    original = root / "mlp-diagnostic-v1"

    for tag in ("native-normal", "qkvo-normal", "mixed-normal"):
        assert digest(run / (tag + ".json")) == digest(original / (tag + ".json"))

    failed = json.loads((run / "failed-native-profile.json").read_text())

    assert failed["status"] == "failed" and failed["optimizer_updates"] == 2

    assert failed["error"] == "ValueError: profile event has no model phase"

    assert failed["steps"] == json.loads((run / "reference.json").read_text())["steps"]

    for arm in ("native", "qkvo", "mixed"):
        sidecar = json.loads((run / (arm + "-profile.operator-events.json")).read_text())

        assert sidecar["bridge_sha256"] == amendment["bridge_sha256"]

        assert sidecar["events"]

    roster = [
        f"{arm}-{variant}"
        for variant in ("normal", "profile")
        for arm in ("native", "qkvo", "mixed")
    ]

    assert freeze["roster"] == roster and freeze["updates_per_process"] == 2

    assert freeze["timing_pairs"] == 0

    assert digest(run / "protocol.md") == freeze["protocol_sha256"]

    assert digest(run / "reference.json") == freeze["reference_sha256"]

    assert digest(run / "census.json") == freeze["census_sha256"]

    assert digest(run / "source-hashes.json") == freeze["sources_sha256"]

    assert json.loads((run / "transfer-check.json").read_text())["verified"]

    assert json.loads((run / "profiler-capability.json").read_text())["verified"]

    publication = json.loads((run / "publication.json").read_text())

    assert publication["verified"] and publication["protocol_sha256"] == freeze["protocol_sha256"]

    ledger = json.loads((run / "ledger.json").read_text())

    assert [row["tag"] for row in ledger] == ["profiler-capability"] + roster

    assert all(row["returncode"] == 0 and row["guard_failure"] is None for row in ledger)

    prior = json.loads((run / "material-preflight.json").read_text())

    post = json.loads((run / "material-preservation.json").read_text())

    assert prior == post and post["verified"] and len(post["files_sha256"]) == 41

    reference = json.loads((run / "reference.json").read_text())

    assert reference["status"] == "ok" and len(reference["steps"]) == 2

    rows = {}

    for tag in roster:
        arm, variant = tag.split("-")

        receipt = json.loads((run / (tag + ".json")).read_text())

        assert receipt["status"] == "ok" and receipt["arm"] == arm

        assert receipt["profile"] == (variant == "profile")

        assert receipt["optimizer_updates"] == 2 and receipt["steps"] == reference["steps"]

        assert all(
            len(step["gradients"]) == len(step["parameters"]) == 256 for step in receipt["steps"]
        )

        assert receipt["reference_sha256"] == freeze["reference_sha256"]

        assert receipt["cap_binding"]["verified"] and receipt["cap_bytes"] == CAP

        assert receipt["native"]["k"] == 11 and receipt["native"]["cap_bound"] == "override"

        assert receipt["native"]["full_layer_charge"] == 482015738

        assert receipt["native"]["budget_bytes"] == 5756682240

        assert any("training.stream_vram_override" in message for message in receipt["messages"])

        assert receipt["shape"] == [1, 2048] and receipt["cache_scope"] == "layer"

        assert receipt["attention_implementation"] == "sdpa"

        expected_k = 11 if arm == "native" else 8

        assert receipt["effective_k"] == expected_k

        assert receipt["plan"]["retained_indices"] == list(range(32 - expected_k, 32))

        saved = {"native": 0, "qkvo": 1006632960, "mixed": 1593835520}[arm]

        tensors = {"native": 0, "qkvo": 96, "mixed": 106}[arm]

        assert receipt["cache_bytes"] == saved and receipt["cache_tensors"] == tensors

        if arm != "native":
            assert receipt["plan"]["qkvo_indices"] == list(range(24))

            assert receipt["plan"]["mlp_indices"] == (list(range(19, 24)) if arm == "mixed" else [])

            assert receipt["plan"]["raw_bytes"] == saved

            expected_charge = 1829297278 if arm == "mixed" else 1155345648

            assert receipt["plan"]["charged_bytes"] == expected_charge

            assert receipt["plan"]["residual_bytes"] == 1900556336

            assert receipt["plan"]["slack_bytes"] == 1900556336 - expected_charge

        assert 0 < receipt["peak_allocated"] <= receipt["peak_reserved"] <= CAP

        if variant == "profile":
            assert "aten::_scaled_dot_product_efficient_attention" in receipt["attention_ops"]

            # Non-reentrant checkpoint can raise its early-stop sentinel while
            # aten::mm is saving inputs, before the down projection launches any
            # CUDA kernel. Preserve these CPU boundaries, but do not require a
            # device duration for work that never executed. All tested QKVO and
            # gate/up operations still require positive CUDA attribution.
            raw = json.loads((run / (tag + ".operator-events.json")).read_text())["events"]
            for row in receipt["projection_ops"]:
                if row["device_us"] > 0:
                    continue
                assert row["projection"] == "down_proj" and row["phase"] == "backward"
                label = f"soup2162|backward|{row['layer']}|down_proj"
                events = [event for event in raw if label in event["ancestors"]]
                assert len(events) == row["count"] and events
                assert all(event["device_us"] == 0 and not event["children"] for event in events)

        readiness = [
            json.loads(line) for line in (run / (tag + "-readiness.jsonl")).read_text().splitlines()
        ]

        assert [row["consecutive"] for row in readiness[-3:]] == [1, 2, 3]

        resources = [
            json.loads(line) for line in (run / (tag + "-resources.jsonl")).read_text().splitlines()
        ]

        assert resources

        for sample in resources:
            assert sample["host_free"] >= 32 * 2**30 and sample["rss"] <= 16 * 2**30

            assert not sample["windows_ci"] and not sample["linux_ci"]

            assert [float(gpu[6]) for gpu in sample["gpus"]] == [420, 350, 420, 420]

            assert all(float(gpu[5]) <= 75 for gpu in sample["gpus"])

            assert all(float(gpu[4]) <= 5 for gpu in sample["gpus"] if int(gpu[0]) != 0)

        rows[tag] = receipt

    memories = []

    for variant in ("normal", "profile"):
        baseline, candidate = rows["qkvo-" + variant], rows["mixed-" + variant]

        delta = candidate["peak_allocated"] - baseline["peak_allocated"]

        memories.append(
            dict(
                variant=variant,
                allocated_delta=delta,
                saved_delta=587202560,
                passes=delta <= 587202560 + 64 * 2**20,
            )
        )

    profiles = {
        arm: rows[arm + "-profile"]["projection_ops"] for arm in ("native", "qkvo", "mixed")
    }

    def count(arm, phase, layer, projection):

        return sum(
            row["count"]
            for row in profiles[arm]
            if row["phase"] == phase and row["layer"] == layer and row["projection"] == projection
        )

    attention = ("q_proj", "k_proj", "v_proj", "o_proj")

    mlp = ("gate_proj", "up_proj")

    expected_counts = []

    for arm in profiles:
        for layer in range(32):
            for projection in attention + mlp:
                expected_counts.append(count(arm, "forward", layer, projection) == 2)

                backward = 2 if (layer < 21 if arm == "native" else layer < 24) else 0

                if arm != "native" and projection in attention:
                    backward = 0

                if arm == "mixed" and projection in mlp and layer in range(19, 24):
                    backward = 0

                expected_counts.append(count(arm, "backward", layer, projection) == backward)

    avoided_us = (
        sum(
            row["device_us"]
            for row in profiles["qkvo"]
            if row["phase"] == "backward"
            and row["projection"] in mlp
            and row["layer"] in range(19, 24)
        )
        / 2
    )

    estimate = (
        100
        * (24 * 0.0024343118421280736 + avoided_us / 1e6 - 3 * 0.011611449999691104)
        / 1.83223659999976
    )

    mechanism = all(expected_counts)

    feasible = mechanism and all(row["passes"] for row in memories) and estimate >= 3

    return dict(
        verified=True,
        full_model_processes=6,
        exact_updates=12,
        adapter_tensors=256,
        reused_processes=3,
        failed_profile_attempts=1,
        failed_attempt_updates=2,
        total_updates=14,
        timing_pairs=0,
        measured_speedup=None,
        mechanism_passes=mechanism,
        memory_checks=memories,
        mlp_device_ms_per_update=avoided_us / 1000,
        planning_estimate_percent=estimate,
        eligible_to_prepare_timing_protocol=feasible,
        profiles=profiles,
        peaks={
            tag: dict(allocated=row["peak_allocated"], reserved=row["peak_reserved"])
            for tag, row in rows.items()
        },
        limitations=[
            "Kernel time is not step time.",
            "Old layer penalties are linearly extrapolated.",
            "No independent throughput confirmation has run.",
            "Capped 24 GiB RTX 3090; not a physical 8 GiB card.",
        ],
    )
