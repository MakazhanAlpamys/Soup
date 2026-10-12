"""Capture, replay, and audit a disjoint expert-profile holdout measurement.

Only collect imports model dependencies. Replay uses saved integer routing counters;
its all-layer means include layer zero. The random comparison is analytic, not a
sampled cache. These traffic fractions are not timing or SSD-read measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import moe_expert_coverage as coverage  # noqa: E402

MATERIAL_GAIN_MIN = 0.10
LOSS_MAX = 0.05
NUMERIC_TOLERANCE = 1e-12
REQUIRED_STEPS = 16
SCHEMA_VERSION = 1
_MAX_SHAPES = 8

console = Console()


def rule_metadata() -> dict[str, Any]:
    return {
        "material_gain_min": MATERIAL_GAIN_MIN,
        "loss_max": LOSS_MAX,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "required_steps": REQUIRED_STEPS,
        "default_split": 8,
        "random_hit": 0.25,
        "cache_fraction": 0.25,
        "layer_population": "all discovered layers, including layer zero",
        "transfer_loss": "signed: training in-sample hit minus evaluation heldout hit",
        "scope": "this captured contiguous split; no population or timing generalisation",
        "order": [
            "VOID",
            "HOTSET TRANSFERS",
            "GAIN WITH DRIFT",
            "DIRECTION-DEPENDENT",
            "NO MATERIAL GAIN",
        ],
    }


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def contiguous_split(n_steps: int, split: int | None = None) -> tuple[list[int], list[int]]:
    """Return disjoint chronological halves; an explicit split may be asymmetric."""
    if not _integer(n_steps, 2):
        raise ValueError("n_steps must be an integer of at least two")
    if split is None:
        if n_steps % 2:
            raise ValueError("the default split requires an even step count")
        split = n_steps // 2
    if not _integer(split, 1) or split >= n_steps:
        raise ValueError("split must be an integer strictly inside the step range")
    return list(range(split)), list(range(split, n_steps))


def verdict(directions: Sequence[Mapping[str, Any]], *, invariants_ok: bool = True) -> str:
    """Apply the frozen two-direction rule only after structural validity."""
    if not invariants_ok or len(directions) != 2:
        return "VOID"
    values = []
    for direction in directions:
        gain, loss = direction.get("gain"), direction.get("transfer_loss")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (gain, loss)
        ):
            return "VOID"
        values.append((gain, loss))
    material = [gain >= MATERIAL_GAIN_MIN - NUMERIC_TOLERANCE for gain, _ in values]
    if all(material):
        if all(loss <= LOSS_MAX + NUMERIC_TOLERANCE for _, loss in values):
            return "HOTSET TRANSFERS"
        return "GAIN WITH DRIFT"
    if any(material):
        return "DIRECTION-DEPENDENT"
    return "NO MATERIAL GAIN"


def _capture_errors(shape: Mapping[str, Any], n_experts: int, top_k: int) -> list[str]:
    errors = []
    if not _integer(n_experts, 4) or n_experts % 4:
        errors.append("n_experts must be positive and divisible by four")
    if not _integer(top_k, 1) or not _integer(n_experts, 1) or top_k > n_experts:
        errors.append("top_k must be an integer in [1, n_experts]")
    if not _integer(shape.get("steps"), 1) or shape["steps"] != REQUIRED_STEPS:
        errors.append("capture must contain exactly 16 completed steps")
    batch, seq, tokens = shape.get("batch"), shape.get("seq"), shape.get("tokens_per_step")
    if not _integer(batch, 1) or not _integer(seq, 1):
        errors.append("batch and seq must be positive integers")
    elif not _integer(tokens, 1) or tokens != batch * seq:
        errors.append("tokens_per_step must equal batch * seq")
    chunks = shape.get("chunks_available")
    if not _integer(chunks, 1) or (
        _integer(batch, 1) and chunks < REQUIRED_STEPS * batch
    ):
        errors.append("not enough distinct chunk indexes for all steps without wrapping")
    layers = shape.get("layers")
    if not isinstance(layers, list) or not layers:
        errors.append("no captured router layers")
        return errors
    seen_layers, seen_routers = set(), set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            errors.append(f"layer row {index} is not an object")
            continue
        layer_id, router = layer.get("layer"), layer.get("router")
        if not _integer(layer_id) or layer_id in seen_layers:
            errors.append(f"layer row {index} has an unknown or duplicate layer identity")
        else:
            seen_layers.add(layer_id)
        if not isinstance(router, str) or not router or router in seen_routers:
            errors.append(f"layer row {index} has a missing or duplicate router identity")
        else:
            seen_routers.add(router)
        counts = layer.get("per_step_counts")
        if not isinstance(counts, list) or len(counts) != REQUIRED_STEPS:
            errors.append(f"layer row {index} must contain exactly 16 step count vectors")
            continue
        for step, row in enumerate(counts):
            if not isinstance(row, list) or len(row) != n_experts:
                errors.append(f"layer row {index}, step {step}: wrong expert vector length")
            elif any(not _integer(value) for value in row):
                errors.append(f"layer row {index}, step {step}: counts must be nonnegative ints")
            elif _integer(tokens, 1) and _integer(top_k, 1) and sum(row) != tokens * top_k:
                errors.append(f"layer row {index}, step {step}: total != tokens_per_step * top_k")
    return errors


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sum_counts(rows: Sequence[Sequence[int]], n_experts: int) -> list[int]:
    return [sum(row[expert] for row in rows) for expert in range(n_experts)]


def _full_in_sample(rows: Sequence[Sequence[int]], n_experts: int) -> dict[str, Any]:
    counts = _sum_counts(rows, n_experts)
    hot = coverage.hot_set(counts)
    hit = coverage.traffic_hit_rate(counts, hot)
    share = coverage.top_share(counts, 0.25)
    delta = hit - share
    if abs(delta) > NUMERIC_TOLERANCE:
        raise ValueError("full-data in-sample/top25 identity failed")
    return {
        "counts": counts,
        "hot_ids": sorted(hot),
        "in_sample_hit": hit,
        "top25_share": share,
        "identity_delta": delta,
    }


def _score_direction(
    rows: Sequence[Sequence[int]], train: list[int], evaluation: list[int], n_experts: int
) -> dict[str, Any]:
    trained_counts = _sum_counts([rows[step] for step in train], n_experts)
    hot = coverage.hot_set(trained_counts)
    training_hits = [coverage.traffic_hit_rate(rows[step], hot) for step in train]
    heldout_hits = [coverage.traffic_hit_rate(rows[step], hot) for step in evaluation]
    oracle_sets = [coverage.hot_set(rows[step]) for step in evaluation]
    oracle_hits = [
        coverage.traffic_hit_rate(rows[step], oracle)
        for step, oracle in zip(evaluation, oracle_sets)
    ]
    if any(
        fixed > oracle + NUMERIC_TOLERANCE for fixed, oracle in zip(heldout_hits, oracle_hits)
    ):
        raise ValueError("per-step oracle does not bound the fixed cache")
    in_sample, heldout = _mean(training_hits), _mean(heldout_hits)
    return {
        "train_steps": train,
        "evaluation_steps": evaluation,
        "hot_ids": sorted(hot),
        "training_counts": trained_counts,
        "in_sample_step_hits": training_hits,
        "heldout_step_hits": heldout_hits,
        "oracle_step_hits": oracle_hits,
        "oracle_hot_ids": [sorted(hot_ids) for hot_ids in oracle_sets],
        "random_hit": 0.25,
        "heldout_hit": heldout,
        "in_sample_hit": in_sample,
        "oracle_hit": _mean(oracle_hits),
        "gain": heldout - 0.25,
        "relative_gain": heldout / 0.25 - 1.0,
        "transfer_loss": in_sample - heldout,
    }


def score_shape(
    shape: Mapping[str, Any], n_experts: int, top_k: int, split: int | None = None
) -> dict[str, Any]:
    """Validate raw counters, then score each half exclusively on the other half."""
    errors = _capture_errors(shape, n_experts, top_k)
    if errors:
        return {
            "verdict": "VOID", "invariants_ok": False, "errors": errors,
            "rule_applicable": False, "diagnostics_only": False,
        }
    try:
        left, right = contiguous_split(shape["steps"], split)
    except ValueError as exc:
        return {
            "verdict": "VOID", "invariants_ok": False, "errors": [str(exc)],
            "rule_applicable": False, "diagnostics_only": False,
        }
    layers = []
    try:
        for layer in shape["layers"]:
            rows = layer["per_step_counts"]
            layers.append({
                "layer": layer["layer"],
                "router": layer["router"],
                "full_in_sample": _full_in_sample(rows, n_experts),
                "directions": [
                    _score_direction(rows, left, right, n_experts),
                    _score_direction(rows, right, left, n_experts),
                ],
            })
    except ValueError as exc:
        return {
            "verdict": "VOID", "invariants_ok": False, "errors": [str(exc)],
            "rule_applicable": False, "diagnostics_only": False,
        }
    metric_names = (
        "heldout_hit", "in_sample_hit", "oracle_hit", "gain", "relative_gain", "transfer_loss"
    )
    directions = []
    for index, (train, evaluation) in enumerate(((left, right), (right, left))):
        metrics = {
            name: _mean([layer["directions"][index][name] for layer in layers])
            for name in metric_names
        }
        worst = min(
            layers,
            key=lambda layer: (layer["directions"][index]["heldout_hit"], layer["layer"]),
        )
        directions.append({
            "train_steps": train,
            "evaluation_steps": evaluation,
            "random_hit": 0.25,
            **metrics,
            "worst_layer": {
                "layer": worst["layer"], "router": worst["router"],
                **{name: worst["directions"][index][name] for name in metric_names},
            },
        })
    rule_applicable = len(left) == len(right) == 8
    return {
        "verdict": verdict(directions) if rule_applicable else None,
        "invariants_ok": True,
        "errors": [],
        "rule_applicable": rule_applicable,
        "diagnostics_only": not rule_applicable,
        "split": len(left),
        "random_hit": 0.25,
        "cache_experts": n_experts // 4,
        "layer_count": len(layers),
        "full_in_sample": {
            name: _mean([layer["full_in_sample"][name] for layer in layers])
            for name in ("in_sample_hit", "top25_share", "identity_delta")
        },
        "directions": directions,
        "layers": layers,
    }


def _digest(payload: bytes) -> dict[str, str]:
    sha = hashlib.sha256(payload).hexdigest()
    return {"sha256": sha, "sha256_16": sha[:16]}


def fingerprints() -> dict[str, Any]:
    own = pathlib.Path(__file__)
    imported = pathlib.Path(coverage.__file__)
    return {
        "holdout": {"file": own.name, **_digest(own.read_bytes())},
        "coverage": {"file": imported.name, **_digest(imported.read_bytes())},
    }


def runtime_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "datasets", "huggingface_hub", "numpy", "rich"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _score_recorded_shape(
    shape: Mapping[str, Any], n_experts: int, top_k: int,
    routers: Any, split: int | None = None,
) -> dict[str, Any]:
    """Require the entire saved discovery inventory before scoring a record."""
    errors = []
    if not isinstance(routers, list) or not routers:
        errors.append("a nonempty saved router discovery inventory is required")
    else:
        expected = set()
        seen_layers, seen_names = set(), set()
        for router in routers:
            if not isinstance(router, Mapping):
                errors.append("saved router inventory contains a non-object entry")
                continue
            layer_id, name = router.get("layer"), router.get("name")
            if (
                not _integer(layer_id) or not isinstance(name, str) or not name
                or layer_id in seen_layers or name in seen_names
            ):
                errors.append("saved router inventory has invalid or duplicate identities")
                continue
            seen_layers.add(layer_id)
            seen_names.add(name)
            expected.add((layer_id, name))
        layers = shape.get("layers")
        actual = set()
        if isinstance(layers, list):
            for layer in layers:
                if not isinstance(layer, Mapping):
                    continue
                layer_id, name = layer.get("layer"), layer.get("router")
                if _integer(layer_id) and isinstance(name, str) and name:
                    actual.add((layer_id, name))
        if (
            not isinstance(layers, list) or len(layers) != len(routers)
            or actual != expected
        ):
            errors.append("captured layer/router population differs from saved discovery inventory")
    if errors:
        return {
            "verdict": "VOID", "invariants_ok": False, "errors": errors,
            "rule_applicable": False, "diagnostics_only": False,
        }
    return score_shape(shape, n_experts, top_k, split)


def replay_results(results: Mapping[str, Any], split: int | None = None) -> dict[str, Any]:
    """Recalculate summaries without importing torch, transformers, or datasets."""
    original_meta = results.get("meta", {})
    n_experts, top_k = original_meta.get("n_experts"), original_meta.get("top_k")
    corpora = []
    for corpus in results["corpora"]:
        corpora.append({
            "label": corpus.get("label"),
            "spec": corpus.get("spec"),
            "shapes": {
                name: _score_recorded_shape(
                    shape, n_experts, top_k, original_meta.get("routers"), split
                )
                for name, shape in corpus["shapes"].items()
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "replay",
        "meta": {
            "rule": rule_metadata(),
            "fingerprints": fingerprints(),
            "runtime_versions": runtime_versions(),
            "n_experts": n_experts,
            "top_k": top_k,
            "requested_split": split,
        },
        "collection_meta": dict(original_meta),
        "corpora": corpora,
    }


def _audit_population(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["traffic_caught_by_corpus_hot25"] is not None]
    all_mean = _mean([row["top25_share"] for row in rows]) if rows else None
    scored_mean = _mean([row["top25_share"] for row in scored]) if scored else None
    heat_mean = _mean([row["traffic_caught_by_corpus_hot25"] for row in scored]) if scored else None
    return {
        "all_layer_rows": len(rows),
        "scored_layer_rows": len(scored),
        "top25_mean_all_layers": all_mean,
        "top25_mean_scored_layers": scored_mean,
        "hot25_mean_scored_layers": heat_mean,
        "mismatched_layer_summary_gap": all_mean - heat_mean if scored else None,
        "matched_layer_summary_gap": scored_mean - heat_mean if scored else None,
        "max_abs_identity_difference": max(
            (abs(row["identity_difference"]) for row in scored), default=None
        ),
        "max_abs_count_top25_difference": max(
            (abs(row["count_top25_difference"]) for row in rows), default=None
        ),
    }


def audit_published(paths: Sequence[pathlib.Path | str]) -> dict[str, Any]:
    """Recompute the published identity and the all-vs-scored population gap."""
    if len(paths) != 4:
        raise ValueError("audit-published requires exactly four JSON files")
    files, all_rows = [], []
    for input_path in paths:
        path = pathlib.Path(input_path)
        payload = path.read_bytes()
        results = json.loads(payload)
        shapes, file_rows = [], []
        for corpus in results["corpora"]:
            for name, shape in corpus["shapes"].items():
                rows = []
                for layer in shape["layers"]:
                    counts = layer["counts"]
                    if not isinstance(counts, list) or not counts or any(
                        not _integer(value) for value in counts
                    ):
                        raise ValueError(f"{path}: invalid published expert counts")
                    top25 = layer["top25_share"]
                    ahead = layer.get("one_layer_ahead")
                    heat = ahead["traffic_caught_by_corpus_hot25"] if ahead is not None else None
                    if any(
                        type(value) not in (int, float) or not math.isfinite(value)
                        for value in ([top25, heat] if ahead is not None else [top25])
                    ):
                        raise ValueError(f"{path}: nonfinite or invalid published traffic fraction")
                    recomputed = coverage.top_share(counts, 0.25)
                    rows.append({
                        "layer": layer["layer"],
                        "top25_share": top25,
                        "recomputed_top25_share": recomputed,
                        "traffic_caught_by_corpus_hot25": heat,
                        "identity_difference": top25 - heat if heat is not None else None,
                        "count_top25_difference": recomputed - top25,
                    })
                population = _audit_population(rows)
                stored_mean = shape.get("traffic_caught_by_corpus_hot25_mean")
                heat_mean = population["hot25_mean_scored_layers"]
                shapes.append({
                    "corpus": corpus.get("label"),
                    "shape": name,
                    **population,
                    "published_hot25_mean": stored_mean,
                    "published_summary_difference": (
                        stored_mean - heat_mean
                        if stored_mean is not None and heat_mean is not None else None
                    ),
                    "layers": rows,
                })
                file_rows.extend(rows)
        files.append({
            "filename": str(path),
            **_digest(payload),
            **_audit_population(file_rows),
            "shapes": shapes,
        })
        all_rows.extend(file_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "audit-published",
        "meta": {"fingerprints": fingerprints(), "runtime_versions": runtime_versions()},
        **_audit_population(all_rows),
        "files": files,
    }


def atomic_json(path: pathlib.Path | str, results: Mapping[str, Any]) -> None:
    """Replace a same-directory checkpoint only after flushing a complete JSON."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(results, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_source(directory: pathlib.Path | str) -> dict[str, Any]:
    """Record full source revision and dirtiness, or absence of a git checkout."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if revision.returncode or not re.fullmatch(r"[0-9a-fA-F]{40}", revision.stdout.strip()):
            return {"revision": None, "dirty": None}
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain", "--", "."],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return {
            "revision": revision.stdout.strip().lower(),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def resolve_revision(repo: str, requested: str, repo_type: str = "model") -> str:
    """Resolve once, then pass the exact full immutable SHA into actual reads."""
    from huggingface_hub import HfApi

    revision = HfApi().repo_info(repo, revision=requested, repo_type=repo_type).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(f"{repo}: Hub did not return a full immutable commit SHA")
    return revision.lower()


def load_pinned_hf_corpus(spec: str, revision: str, limit: int) -> list[str]:
    """Preserve the old accepted-nonempty-string selection with a pinned read."""
    from datasets import load_dataset

    parts = spec[3:].split(":")
    if not parts[0] or len(parts) > 4:
        raise ValueError("HF data spec must be hf:repo[:config][:split][:field]")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    field_name = parts[3] if len(parts) > 3 and parts[3] else None
    stream = load_dataset(name, config, split=split, streaming=True, revision=revision)
    texts = []
    for row in stream:
        value = row.get(field_name) if field_name is not None else next(
            (value for value in row.values() if isinstance(value, str)), None
        )
        if isinstance(value, str) and value.strip():
            texts.append(value)
        if len(texts) >= limit:
            break
    return texts


def _hardware(torch: Any, device: str) -> dict[str, Any]:
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_runtime": torch.version.cuda,
        "device": device,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return result


def _token_digest(chunks: Any) -> dict[str, str]:
    """SHA256 of row-major signed little-endian int64 token IDs."""
    values = chunks.numpy().astype("<i8", copy=False)
    return _digest(values.tobytes(order="C"))


def _capture_shape(
    model: Any, hooks: Sequence[Any], chunks: Any, shape: dict[str, Any],
    *, device: str, n_experts: int, top_k: int, routers: list[Mapping[str, Any]],
) -> None:
    """Use coverage's router capture only when its modulo indexing cannot wrap."""
    batch, seq = shape["batch"], shape["seq"]
    needed = REQUIRED_STEPS * batch
    if chunks.shape[0] < needed:
        raise ValueError(
            f"{batch}x{seq} needs {needed} distinct chunk indexes; only {chunks.shape[0]} available"
        )
    shape["packed_tokens"] = {
        "encoding": "row-major signed little-endian int64",
        "shape": [int(chunks.shape[0]), seq],
        **_token_digest(chunks),
    }
    shape["consumed_tokens"] = {
        "shape": [needed, seq], **_token_digest(chunks[:needed]),
    }
    shape["step_inputs"] = [
        {
            "step": step,
            "chunk_start": step * batch,
            "chunk_stop": (step + 1) * batch,
            **_token_digest(chunks[step * batch:(step + 1) * batch]),
        }
        for step in range(REQUIRED_STEPS)
    ]
    stats, steps, tokens = coverage.measure_shape(
        model, hooks, chunks, batch=batch, seq=seq, batches=REQUIRED_STEPS,
        device=device, offset=0,
    )
    shape.update({
        "steps": steps,
        "tokens_per_step": tokens,
        "chunks_available": int(chunks.shape[0]),
        "chunks_used": needed,
        "wraparound": False,
        "layers": [
            {"layer": layer.layer, "router": layer.name, "per_step_counts": layer.per_step_counts}
            for layer in stats
        ],
    })
    shape["summary"] = _score_recorded_shape(shape, n_experts, top_k, routers)
    if not shape["summary"]["invariants_ok"]:
        raise ValueError("; ".join(shape["summary"]["errors"]))
    shape["status"] = "COMPLETE"


def _load_model(
    args: argparse.Namespace, meta: dict[str, Any]
) -> tuple[Any, Any, list[Any]]:
    """Load one immutable unquantised model and its explicitly eager routers."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    meta.update({"device": device, "hardware": _hardware(torch, device)})
    model_sha = resolve_revision(args.model, args.revision)
    meta.update({
        "model_revision": model_sha,
        "tokenizer": args.model,
        "requested_tokenizer_revision": args.revision,
        "tokenizer_revision": model_sha,
        "config_revision": model_sha,
    })
    config = AutoConfig.from_pretrained(
        args.model, revision=model_sha, trust_remote_code=args.trust_remote_code,
    )
    n_experts = (
        getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    )
    top_k = getattr(config, "num_experts_per_tok", None)
    if not _integer(n_experts, 4) or n_experts % 4 or not _integer(top_k, 1):
        raise ValueError("model config must declare divisible-by-four experts and positive top_k")
    if top_k > n_experts:
        raise ValueError("model config top_k exceeds n_experts")
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("the new arm requires unquantised model weights")
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    meta.update({
        "n_experts": n_experts, "top_k": top_k,
        "model_type": getattr(config, "model_type", None),
        "hidden_layers": getattr(config, "num_hidden_layers", None),
    })
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=model_sha, trust_remote_code=args.trust_remote_code,
    )
    console.print(f"Loading {args.model}@{model_sha} on {device}/{args.dtype}", markup=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=model_sha, config=config, dtype=getattr(torch, args.dtype),
        attn_implementation="eager", trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    meta.update({
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "experts_implementation": getattr(model.config, "_experts_implementation", None),
        "training": bool(model.training),
        "use_cache": False,
    })
    if model.training or meta["attention_implementation"] != "eager" or (
        meta["experts_implementation"] != "eager"
    ):
        raise ValueError("model did not retain eval/eager attention/eager expert configuration")
    hooks = coverage.attach_routers(model, n_experts=n_experts, top_k=top_k)
    if any(hook.layer < 0 for hook in hooks) or len({hook.layer for hook in hooks}) != len(hooks):
        raise ValueError("router discovery returned unknown or duplicate layer identities")
    meta["routers"] = [
        {"layer": hook.layer, "name": hook.name, "class": type(hook.module).__name__}
        for hook in hooks
    ]
    meta["n_routers"] = len(hooks)
    return model, tokenizer, hooks


def _read_corpus(
    args: argparse.Namespace, corpus: dict[str, Any], requested: str | None
) -> list[str]:
    """Read one pinned corpus and record the digest of its actual selected text."""
    spec = corpus["spec"]
    if spec.startswith("hf:"):
        dataset_sha = resolve_revision(spec[3:].split(":")[0], requested, "dataset")
        corpus.update({
            "requested_dataset_revision": requested,
            "dataset_revision": dataset_sha,
        })
        texts = load_pinned_hf_corpus(spec, dataset_sha, args.limit_rows)
    else:
        texts = coverage.load_corpus(spec, args.fields.split(","), args.limit_rows)
        directory_revision = coverage.corpus_revision(spec)
        directory_source = git_source(spec) if os.path.isdir(spec) else None
        if os.path.isdir(spec) and directory_revision is None and args.source_revision:
            directory_revision = args.source_revision.lower()
            directory_source = {
                "revision": directory_revision, "dirty": None,
                "revision_origin": "supplied archive revision",
            }
        corpus.update({"git_revision": directory_revision, "source": directory_source})
    corpus.update({
        "documents": len(texts),
        "corpus_digest_encoding": "UTF-8 documents joined by NUL, in read order",
        **_digest(chr(0).join(texts).encode("utf-8")),
    })
    return texts


def collect(args: argparse.Namespace) -> int:
    """Collect unquantised eager forward-only routing with shape checkpoints."""
    source = git_source(pathlib.Path(__file__).parent)
    source["supplied_revision"] = args.source_revision
    if source["revision"] is None and args.source_revision:
        source["revision"] = args.source_revision.lower()
        source["revision_origin"] = "supplied archive revision"
    else:
        source["revision_origin"] = "git" if source["revision"] else None
    results = {
        "schema_version": SCHEMA_VERSION,
        "mode": "collect",
        "meta": {
            "status": "COLLECTING",
            "model": args.model,
            "requested_model_revision": args.revision,
            "source": source,
            "fingerprints": fingerprints(),
            "runtime_versions": runtime_versions(),
            "rule": rule_metadata(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seed": args.seed,
            "batches_per_shape": args.batches,
            "shapes": [f"{batch}x{seq}" for batch, seq in args.shape],
            "limit_rows": args.limit_rows,
            "fields": args.fields.split(","),
            "dtype": args.dtype,
            "load_4bit": False,
            "quantisation": None,
            "attention_implementation_requested": "eager",
            "experts_implementation_requested": "eager",
            "trust_remote_code": args.trust_remote_code,
            "host_before": coverage.host_state(),
        },
        "corpora": [],
    }
    atomic_json(args.out, results)
    active_shape = None
    try:
        model, tokenizer, hooks = _load_model(args, results["meta"])
        device = results["meta"]["device"]
        n_experts, top_k = results["meta"]["n_experts"], results["meta"]["top_k"]
        dataset_revisions = iter(args.dataset_revision)
        for spec, label in zip(args.data, args.data_label):
            corpus = {"label": label, "spec": spec, "shapes": {}}
            results["corpora"].append(corpus)
            requested = next(dataset_revisions) if spec.startswith("hf:") else None
            texts = _read_corpus(args, corpus, requested)
            console.print(
                f"Corpus {label!r}: {len(texts)} documents, SHA256 {corpus['sha256']}", markup=False
            )
            cached_seq, cached_chunks = None, None
            for batch, seq in args.shape:
                active_shape = {"batch": batch, "seq": seq, "status": "COLLECTING"}
                corpus["shapes"][f"{batch}x{seq}"] = active_shape
                if cached_seq != seq:
                    cached_chunks = coverage.pack_token_stream(texts, tokenizer, seq, args.seed)
                    cached_seq = seq
                _capture_shape(
                    model, hooks, cached_chunks, active_shape,
                    device=device, n_experts=n_experts, top_k=top_k,
                    routers=results["meta"]["routers"],
                )
                atomic_json(args.out, results)
                console.print(
                    f"  {batch}x{seq}: {active_shape['summary']['verdict']}", markup=False
                )
                active_shape = None
        results["meta"]["status"] = "COMPLETE"
    except (Exception, KeyboardInterrupt) as exc:
        results["meta"]["status"] = "VOID"
        results["meta"]["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if active_shape is not None:
            active_shape["status"] = "VOID"
            active_shape.setdefault("summary", {
                "verdict": "VOID", "invariants_ok": False, "errors": [str(exc)],
                "rule_applicable": False, "diagnostics_only": False,
            })
        console.print(f"VOID: {type(exc).__name__}: {exc}", style="red", markup=False)
    results["meta"]["host_after"] = coverage.host_state()
    results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(args.out, results)
    return 0 if results["meta"]["status"] == "COMPLETE" else 2


def _full_sha(text: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", text):
        raise argparse.ArgumentTypeError("source revision must be a full 40-character commit SHA")
    return text.lower()


def _protect_inputs(output: str, inputs: Sequence[str]) -> None:
    target = os.path.normcase(os.path.realpath(output))
    if any(target == os.path.normcase(os.path.realpath(path)) for path in inputs):
        raise ValueError("output must not overwrite an input JSON")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)
    capture_parser = subparsers.add_parser("collect", help="capture frozen-revision routing")
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--revision", required=True)
    capture_parser.add_argument("--source-revision", type=_full_sha)
    capture_parser.add_argument("--data", action="append", required=True)
    capture_parser.add_argument("--data-label", action="append", required=True)
    capture_parser.add_argument("--dataset-revision", action="append", default=[])
    capture_parser.add_argument("--fields", default="text,content,output,instruction,messages")
    capture_parser.add_argument("--shape", action="append", type=coverage.parse_shape)
    capture_parser.add_argument("--batches", type=int, default=REQUIRED_STEPS)
    capture_parser.add_argument("--limit-rows", type=int, default=4000)
    capture_parser.add_argument("--seed", type=int, default=17)
    capture_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    capture_parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    capture_parser.add_argument("--trust-remote-code", action="store_true")
    capture_parser.add_argument("--out", required=True)
    replay_parser = subparsers.add_parser("replay", help="recalculate without models or network")
    replay_parser.add_argument("results")
    replay_parser.add_argument("--split", type=int)
    replay_parser.add_argument("--out", required=True)
    audit_parser = subparsers.add_parser("audit-published", help="audit four original records")
    audit_parser.add_argument("results", nargs=4)
    audit_parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mode == "collect":
        args.shape = args.shape or [(1, 512), (4, 512), (1, 2048)]
        if args.batches != REQUIRED_STEPS:
            capture_parser.error("the frozen rule requires exactly --batches 16")
        if args.limit_rows < 1:
            capture_parser.error("--limit-rows must be positive")
        if len(args.shape) > _MAX_SHAPES or len(set(args.shape)) != len(args.shape):
            capture_parser.error("provide at most eight distinct shapes")
        if len(args.data_label) != len(args.data):
            capture_parser.error("provide exactly one --data-label per --data")
        if len(args.dataset_revision) != sum(spec.startswith("hf:") for spec in args.data):
            capture_parser.error("provide one --dataset-revision per HF --data, in data order")
        return collect(args)
    try:
        if args.mode == "replay":
            _protect_inputs(args.out, [args.results])
            path = pathlib.Path(args.results)
            payload = path.read_bytes()
            results = replay_results(json.loads(payload), args.split)
            results["source"] = {"filename": str(path), **_digest(payload)}
        else:
            _protect_inputs(args.out, args.results)
            results = audit_published(args.results)
        atomic_json(args.out, results)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        console.print(f"Refused: {exc}", style="red", markup=False)
        return 2
    console.print(f"Wrote {args.out}", markup=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
