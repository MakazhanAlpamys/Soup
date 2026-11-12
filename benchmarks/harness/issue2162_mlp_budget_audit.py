"""Independent B audit from saved tensors, operation receipts and raw timing rows."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

CAP = 8 * 2**30
RAW = {"native": 0, "qkvo": 1090519040, "mixed": 2382364672}
TENSORS = {"native": 0, "qkvo": 104, "mixed": 126}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values, seed=42):
    assert len(values) in (5, 6) and all(math.isfinite(x) for x in values)
    draw = np.median(
        np.asarray(values)[
            np.random.default_rng(seed).integers(0, len(values), (10000, len(values)))
        ],
        axis=1,
    )
    ordinary = np.quantile(draw, [0.025, 0.975], method="linear").tolist()
    quantiles = [0.05 / 24, 1 - 0.05 / 24]
    adjusted = np.quantile(draw, quantiles, method="linear").tolist()
    median = statistics.median(values)
    return dict(
        median_reduction_percent=median,
        ordinary_95=ordinary,
        family_adjusted_ci=adjusted,
        adjusted_quantiles=quantiles,
        correction_family=12,
        reductions_percent=values,
        resamples=10000,
        seed=seed,
        count=len(values),
        positive_lower_bounds=ordinary[0] > 0 and adjusted[0] > 0,
        passes_two_percent=median >= 2 and ordinary[0] > 0 and adjusted[0] > 0,
    )


def memories(rows):
    result = []
    for candidate, baseline, saved in [
        ("mixed", "qkvo", 1291845632),
        ("mixed", "native", 2382364672),
        ("qkvo", "native", 1090519040),
    ]:
        if candidate not in rows or baseline not in rows:
            continue
        delta = rows[candidate]["peak_allocated"] - rows[baseline]["peak_allocated"]
        result.append(
            dict(
                contrast=candidate + "_vs_" + baseline,
                delta=delta,
                saved=saved,
                passes=delta <= saved + 64 * 2**20,
            )
        )
    return result


def raw_projection_summary(events):
    """Rebuild saved projection counters from persisted inner operation events."""
    operations = {"aten::mm", "aten::addmm", "bitsandbytes::gemm_4bit"}
    grouped = {}
    for e in events:
        if e["name"] not in operations or "PythonDispatchMode" in e["children"]:
            continue
        if any(c in operations for c in e["children"]):
            continue
        scopes = [a for a in e["ancestors"] if a.startswith("soup2162|")]
        if not scopes:
            continue
        _, phase, layer, projection = scopes[0].split("|")
        assert phase in ("forward", "backward")
        key = (phase, int(layer), projection, e["name"])
        row = grouped.setdefault(
            key,
            dict(
                phase=phase,
                layer=int(layer),
                projection=projection,
                op=e["name"],
                count=0,
                device_us=0.0,
                self_cpu_us=0.0,
            ),
        )
        row["count"] += 1
        row["device_us"] += e["device_us"]
        row["self_cpu_us"] += e["self_cpu_us"]
    return [grouped[k] for k in sorted(grouped)]


def audit(run: Path):
    freeze = json.loads((run / "freeze.json").read_text())
    reference = json.loads((run / "reference.json").read_text())
    assert freeze["family"] == 12 and freeze["warmup"] == 10 and freeze["measured_updates"] == 50
    for key, name in [
        ("protocol_sha256", "protocol.md"),
        ("reference_sha256", "reference.json"),
        ("census_sha256", "census.json"),
        ("sources_sha256", "source-hashes.json"),
    ]:
        assert digest(run / name) == freeze[key]
    transfer = json.loads((run / "transfer-check.json").read_text())
    assert transfer["verified"]
    for name, value in transfer["files_sha256"].items():
        assert digest(run / name) == value, name
    assert digest(run / "controller-source.py") == freeze["controller_sha256"]
    sources = json.loads((run / "source-hashes.json").read_text())
    assert digest(run / "history-inputs.json") == sources["history-inputs.json"]
    publication = json.loads((run / "publication.json").read_text())
    assert publication["verified"] and publication["protocol_sha256"] == freeze["protocol_sha256"]
    assert json.loads((run / "material-preservation.json").read_text()) == json.loads(
        (run / "material-preflight.json").read_text()
    )
    assert len(json.loads((run / "material-preservation.json").read_text())["files_sha256"]) == 41
    expected_gates = [
        dict(stage="gate", arm=arm, variant=v, tag=f"gate-{arm}-{v}")
        for v in ("normal", "profile")
        for arm in RAW
    ]
    assert (
        freeze["gates"] == expected_gates
        and len(freeze["timings"]) == 18
        and len(freeze["confirmation"]) == 12
    )
    import itertools
    import random

    orders = list(itertools.permutations(RAW))
    random.Random(42).shuffle(orders)
    assert freeze["orders"] == [list(x) for x in orders]
    assert freeze["timings"] == [
        dict(stage="timing", arm=arm, block=i, tag=f"block{i}-{arm}")
        for i, order in enumerate(orders, 1)
        for arm in order
    ]
    assert freeze["confirmation"] == [
        dict(stage="confirmation", arm=arm, pair=i, tag=f"confirm{i}-{arm}")
        for i in range(1, 7)
        for arm in (("native", "mixed") if i % 2 else ("mixed", "native"))
    ]
    stage = json.loads((run / "stage-a-decision.json").read_text())
    entries = (
        freeze["gates"] + freeze["timings"] + (freeze["confirmation"] if stage["eligible"] else [])
    )
    ledger = json.loads((run / "ledger.json").read_text())
    assert [row["tag"] for row in ledger] == [entry["tag"] for entry in entries]
    assert all(row["returncode"] == 0 and row["guard_failure"] is None for row in ledger)
    receipts = {}
    timings = {}
    peaks = {}
    sampling = {}
    for entry in entries:
        tag, arm = entry["tag"], entry["arm"]
        r = json.loads((run / (tag + ".json")).read_text())
        assert r["status"] == "ok" and r["arm"] == arm and r["steps"] == reference["steps"]
        assert r["optimizer_updates"] == 2 and all(
            len(s["gradients"]) == len(s["parameters"]) == 256 for s in r["steps"]
        )
        assert (
            r["reference_sha256"] == freeze["reference_sha256"]
            and r["census_sha256"] == freeze["census_sha256"]
        )
        assert r["allocation"] == "fixed-candidate-B" and r["cache_scope"] == "layer"
        assert r["cache_bytes"] == RAW[arm] and r["cache_tensors"] == TENSORS[arm]
        assert r["shape"] == [1, 2048] and r["attention_implementation"] == "sdpa"
        assert r["native"]["k"] == 11 and r["native"]["budget_bytes"] == 5756682240
        assert (
            r["native"]["full_layer_charge"] == 482015738 and r["native"]["cap_bound"] == "override"
        )
        assert r["cap_binding"]["verified"] and r["cap_bytes"] == CAP
        assert any("training.stream_vram_override" in s for s in r["messages"])
        k = 11 if arm == "native" else 6
        assert r["effective_k"] == k and r["plan"]["retained_indices"] == list(range(32 - k, 32))
        if arm != "native":
            assert r["plan"]["qkvo_indices"] == list(range(26))
            assert r["plan"]["mlp_indices"] == (list(range(15, 26)) if arm == "mixed" else [])
            assert r["plan"]["raw_bytes"] == RAW[arm] and r["plan"]["cache_tensors"] == TENSORS[arm]
            assert r["plan"]["residual_bytes"] == 2864587812
            charge = 2734318038 if arm == "mixed" else 1251624452
            assert (
                r["plan"]["charged_bytes"] == charge
                and r["plan"]["slack_bytes"] == 2864587812 - charge
            )
        assert 0 < r["peak_allocated"] <= r["peak_reserved"] <= CAP
        assert r["profile"] == (entry.get("variant") == "profile")
        if r["profile"]:
            assert "aten::_scaled_dot_product_efficient_attention" in r["attention_ops"]
            raw = json.loads((run / (tag + ".operator-events.json")).read_text())
            assert raw_projection_summary(raw["events"]) == r["projection_ops"]
            for layer in range(32):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"):
                    for phase in ("forward", "backward"):
                        rows = [
                            x
                            for x in r["projection_ops"]
                            if x["layer"] == layer
                            and x["projection"] == proj
                            and x["phase"] == phase
                        ]
                        expected = (
                            2
                            if phase == "forward" or layer < (21 if arm == "native" else 26)
                            else 0
                        )
                        if (
                            phase == "backward"
                            and arm != "native"
                            and proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                        ):
                            expected = 0
                        if (
                            phase == "backward"
                            and arm == "mixed"
                            and proj in ("gate_proj", "up_proj")
                            and layer >= 15
                        ):
                            expected = 0
                        assert sum(x["count"] for x in rows) == expected and all(
                            x["device_us"] > 0 for x in rows
                        )
        readiness = [
            json.loads(line) for line in (run / (tag + "-readiness.jsonl")).read_text().splitlines()
        ]
        assert [x["consecutive"] for x in readiness[-3:]] == [1, 2, 3]
        from datetime import datetime

        assert (
            datetime.fromisoformat(publication["created_at"].replace("Z", "+00:00")).timestamp()
            < readiness[0]["unix_s"]
        )
        assert datetime.fromisoformat(freeze["at"]).timestamp() < readiness[0]["unix_s"]
        for sample in readiness[-3:]:
            assert all(float(g[4]) <= 5 for g in sample["gpus"])
            selected = next(g for g in sample["gpus"] if int(g[0]) == 0)
            assert selected[1].strip() == "GPU-05919122-a78e-9af3-3b3c-f9a14fb4d2f4"
            assert float(selected[5]) <= 60 and float(selected[2]) - float(selected[3]) >= 10 * 1024
        resources = [
            json.loads(line) for line in (run / (tag + "-resources.jsonl")).read_text().splitlines()
        ]
        assert resources
        for sample in resources:
            assert sample["host_free"] >= 32 * 2**30 and sample["rss"] <= 16 * 2**30
            assert not sample["windows_ci"] and not sample["linux_ci"]
            assert [float(g[6]) for g in sample["gpus"]] == [420, 350, 420, 420]
            assert all(float(g[5]) <= 75 for g in sample["gpus"])
            assert all(float(g[4]) <= 5 for g in sample["gpus"] if int(g[0]) != 0)
        receipts[tag] = r
        if entry["stage"] != "gate":
            t = json.loads((run / (tag + "-timing.json")).read_text())
            assert t["status"] == "ok"
            assert (
                t["initial_reference_sha256"] == freeze["reference_sha256"]
                and t["verified_initial_updates"] == 2
            )
            assert t["warmup_updates"] == 10 and t["measured_updates"] == 50
            assert r["continuation_optimizer_updates"] == 58 and t["plan"] == r["plan"]
            assert len(t["step_times_s"]) == len(t["losses"]) == 50
            assert all(math.isfinite(x) and x > 0 for x in t["step_times_s"])
            assert all(math.isfinite(x) for x in t["losses"])
            assert t["median_step_s"] == statistics.median(t["step_times_s"])
            assert 0 < t["peak_allocated"] <= t["peak_reserved"] <= CAP
            start, end = t["window_start_unix_s"], t["window_end_unix_s"]
            assert end > start
            assert resources[0]["unix_s"] <= start
            times = [s["unix_s"] for s in resources]
            assert times == sorted(times)
            inside = [s for s in times if start <= s <= end]
            assert inside
            edges = [start, *inside, end]
            sampling[tag] = dict(
                samples_inside=len(inside), max_gap_s=max(b - a for a, b in zip(edges, edges[1:]))
            )
            timings[tag] = t
        peaks[tag] = dict(
            allocated=(timings.get(tag) or r)["peak_allocated"],
            reserved=(timings.get(tag) or r)["peak_reserved"],
        )
    memory_checks = []
    for v in ("normal", "profile"):
        memory_checks += memories({arm: receipts[f"gate-{arm}-{v}"] for arm in RAW})
    blocks = {}
    comparisons = {}
    for i in range(1, 7):
        rows = {arm: timings[f"block{i}-{arm}"] for arm in RAW}
        memory_checks += memories(rows)
        blocks[i] = {arm: row["median_step_s"] for arm, row in rows.items()}
    for key in ("mixed_vs_native", "mixed_vs_qkvo", "qkvo_vs_native"):
        candidate, baseline = key.split("_vs_")
        comparisons[key] = summary(
            [100 * (r[baseline] - r[candidate]) / r[baseline] for r in blocks.values()]
        )
    assert comparisons == stage["comparisons"]
    eligible = (
        comparisons["mixed_vs_native"]["passes_two_percent"]
        and comparisons["mixed_vs_qkvo"]["positive_lower_bounds"]
        and all(x["passes"] for x in memory_checks)
    )
    assert eligible == stage["eligible"]
    confirmation = None
    if eligible:
        values = []
        checks = []
        for i in range(1, 7):
            rows = {arm: timings[f"confirm{i}-{arm}"] for arm in ("native", "mixed")}
            values.append(
                100
                * (rows["native"]["median_step_s"] - rows["mixed"]["median_step_s"])
                / rows["native"]["median_step_s"]
            )
            checks += memories(rows)
        confirmation = summary(values, 73)
        confirmation["memory_checks"] = checks
        confirmation["passes_all"] = confirmation["passes_two_percent"] and all(
            x["passes"] for x in checks
        )
        assert confirmation == json.loads((run / "confirmation-decision.json").read_text())
        memory_checks += checks
    else:
        assert not any(run.glob("confirm*-timing.json"))
    history = json.loads((run / "history-inputs.json").read_text())
    assert len(history) == 8
    historical = []
    for h in history:
        fresh = summary(h["reductions"])
        assert fresh["median_reduction_percent"] == h["original_median"]
        historical.append(dict(**h, new_family_summary=fresh))
    return dict(
        verified=True,
        gate_processes=6,
        timing_processes=len(timings),
        exact_update_replays=2 * len(entries),
        timed_updates=50 * len(timings),
        warmup_updates=10 * len(timings),
        total_optimizer_updates=12 + 60 * len(timings),
        family=12,
        stage_a=comparisons,
        stage_a_eligible=eligible,
        confirmation=confirmation,
        primary_passes=bool(eligible and confirmation and confirmation["passes_all"]),
        memory_checks=memory_checks,
        blocks=blocks,
        peaks=peaks,
        history=historical,
        timing_sampling=sampling,
    )
