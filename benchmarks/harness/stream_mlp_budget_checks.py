"""Before timing, validate B's exactness, memory and executed-op roster."""
from __future__ import annotations

from typing import Any

CAP = 8 * 2**30
RAW = {"native": 0, "qkvo": 1090519040, "mixed": 2382364672}
TENSORS = {"native": 0, "qkvo": 104, "mixed": 126}


def validate_receipt(receipt: dict, arm: str, reference: dict) -> None:
    if receipt["status"] != "ok" or receipt["arm"] != arm:
        raise ValueError("failed or wrong-arm receipt")
    if receipt["steps"] != reference["steps"] or receipt["optimizer_updates"] != 2:
        raise ValueError("two exact replay updates required")
    if any(len(step["gradients"]) != 256 or len(step["parameters"]) != 256
           for step in receipt["steps"]):
        raise ValueError("adapter roster changed")
    if receipt["allocation"] != "fixed-candidate-B" or receipt["cache_scope"] != "layer":
        raise ValueError("allocation changed")
    if receipt["cache_bytes"] != RAW[arm] or receipt["cache_tensors"] != TENSORS[arm]:
        raise ValueError("saved storage mismatch")
    expected_k = 11 if arm == "native" else 6
    if (receipt["native"]["k"] != 11 or receipt["effective_k"] != expected_k
            or receipt["plan"]["retained_indices"] != list(range(32 - expected_k, 32))):
        raise ValueError("whole-layer roster changed")
    if arm != "native":
        if (receipt["plan"]["qkvo_indices"] != list(range(26))
                or receipt["plan"]["mlp_indices"] != (
                    list(range(15, 26)) if arm == "mixed" else [])):
            raise ValueError("selective roster changed")
    if not receipt["cap_binding"]["verified"] or receipt["cap_bytes"] != CAP:
        raise ValueError("native cap did not bind")
    if not 0 < receipt["peak_allocated"] <= receipt["peak_reserved"] <= CAP:
        raise ValueError("memory cap exceeded")


def memory_comparisons(rows: dict[str, Any]) -> list[dict]:
    result = []
    for candidate, baseline, saved in (("mixed", "qkvo", RAW["mixed"] - RAW["qkvo"]),
                                       ("mixed", "native", RAW["mixed"]),
                                       ("qkvo", "native", RAW["qkvo"])):
        if candidate not in rows or baseline not in rows:
            continue
        delta = rows[candidate]["peak_allocated"] - rows[baseline]["peak_allocated"]
        result.append(dict(contrast=f"{candidate}_vs_{baseline}", delta=delta,
                           saved=saved, passes=delta <= saved + 64 * 2**20))
    return result


def gate_decision(receipts: dict, reference: dict) -> dict:
    expected = {f"{arm}-{variant}" for arm in RAW for variant in ("normal", "profile")}
    if set(receipts) != expected:
        raise ValueError("six complete gate processes required")
    checks = []
    for tag, receipt in receipts.items():
        arm, variant = tag.split("-")
        validate_receipt(receipt, arm, reference)
        if receipt["profile"] != (variant == "profile"):
            raise ValueError("profiler roster changed")
        if variant == "normal":
            continue
        if "aten::_scaled_dot_product_efficient_attention" not in receipt["attention_ops"]:
            raise ValueError("selected attention backend changed")
        operators = receipt["projection_ops"]
        for layer in range(32):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"):
                for phase in ("forward", "backward"):
                    matching = [row for row in operators if row["layer"] == layer
                                and row["projection"] == projection and row["phase"] == phase]
                    count = sum(row["count"] for row in matching)
                    expected_count = 2
                    if phase == "backward":
                        expected_count = 2 if layer < (21 if arm == "native" else 26) else 0
                        if arm != "native" and projection in (
                            "q_proj", "k_proj", "v_proj", "o_proj"
                        ):
                            expected_count = 0
                        if (arm == "mixed" and projection in ("gate_proj", "up_proj")
                                and layer >= 15):
                            expected_count = 0
                    if count != expected_count or any(row["device_us"] <= 0 for row in matching):
                        detail = f"{tag}/{layer}/{projection}/{phase}"
                        raise ValueError(f"operator/attribution mismatch: {detail}")
    for variant in ("normal", "profile"):
        checks.extend(memory_comparisons({arm: receipts[f"{arm}-{variant}"] for arm in RAW}))
    return dict(verified=True, passes=all(row["passes"] for row in checks),
                gate_processes=6, exact_updates=12, removed_mlp_matmuls=44, memory_checks=checks)
