"""G1 and G3 of ``benchmarks/gate-kimi-k2-admission.md`` on the real Kimi K2 files.

Reads ``config.json`` and ``model.safetensors.index.json`` as JSON data only: no K2 Python
file is imported or executed, and nothing is downloaded. CPU only.

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<tree>/src python benchmarks/harness/kimi_k2_gate.py \
        --k2-dir <folder holding base/ instruct/ thinking/> --report <out.json>

``--k2-dir`` holds one sub-folder per variant with that variant's text files. G1 loads the
Base and Instruct ``config.json`` (each copied alone into a fresh folder) through Soup's
loader and compares them field by field with the raw JSON, runs the controls, and runs the
guard on the Thinking file. G3 rebuilds the Base checkpoint's header map from the index
names and the config, checks it against the index and against the structure-derived map the
tests use, then runs Soup's disk pre-flight estimator over it in both expert layouts and holds
it against the shape-only oracle in ``tests/test_kimi_k2_streaming.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: The investigation's figures for the estimate's total (payload + header and index allowances).
INVESTIGATION_TOTALS = {"per_expert": 533_246_623_322, "fused": 2_039_152_149_798}
_SIZE = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _alone(config_path: str, scratch: str, name: str) -> str:
    """A fresh folder holding only a copy of ``config_path``."""
    folder = os.path.join(scratch, name)
    os.makedirs(folder)
    shutil.copyfile(config_path, os.path.join(folder, "config.json"))
    return folder


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(f"{record.levelname}: {record.getMessage()[:300]}")


def _field_by_field(config, raw: dict) -> list:
    """One row per raw key: the rule of G1a, with its two named normalisations."""
    rows = []
    for key, value in raw.items():
        if key in ("model_type", "auto_map"):
            continue
        if key == "torch_dtype":
            ok = str(config.torch_dtype) == "torch." + value
            rows.append({"key": key, "equal": ok, "note": f"str(torch_dtype)={config.torch_dtype}"})
            continue
        got = config.rope_parameters["rope_theta"] if key == "rope_theta" else getattr(
            config, key, "<absent>"
        )
        row = {"key": key, "equal": got == value}
        if key == "rope_scaling" and isinstance(got, dict):
            row["raw_keys_equal"] = {name: got.get(name) for name in value} == value
            row["added_keys"] = {name: got[name] for name in sorted(set(got) - set(value))}
        if not row["equal"]:
            row["got"] = repr(got)[:300]
        rows.append(row)
    return rows


def gate_g1(k2_dir: str, scratch: str) -> dict:
    import transformers
    import transformers.dynamic_module_utils as dynamic
    import transformers.models.auto.configuration_auto as configuration_auto
    from transformers import AutoConfig, DeepseekV3Config

    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        kimi_k2_guard_failures,
        load_stream_model_config,
    )
    from tests.test_kimi_k2_streaming import K2_BASE_CONFIG, K2_THINKING_CHANGES

    paths = {
        variant: os.path.join(k2_dir, variant, "config.json")
        for variant in ("base", "instruct", "thinking")
    }
    out: dict = {"sha256": {variant: _sha256(path) for variant, path in paths.items()}}
    out["instruct_bytes_equal_base"] = open(paths["base"], "rb").read() == open(
        paths["instruct"], "rb"
    ).read()
    raw = _load_json(paths["base"])
    out["test_constant_is_the_real_base_config"] = raw == K2_BASE_CONFIG

    calls = []
    real_dynamic = (dynamic.get_class_from_dynamic_module,
                    configuration_auto.get_class_from_dynamic_module)

    def _spy(*args, **kwargs):
        calls.append(args[:1])
        raise AssertionError("remote code was loaded")

    dynamic.get_class_from_dynamic_module = _spy
    configuration_auto.get_class_from_dynamic_module = _spy
    try:
        for variant in ("base", "instruct"):
            folder = _alone(paths[variant], scratch, f"g1-{variant}")
            before = open(os.path.join(folder, "config.json"), "rb").read()
            records = _Records()
            logger = logging.getLogger("transformers")
            logger.addHandler(records)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = load_stream_model_config(folder)
            logger.removeHandler(records)
            rows = _field_by_field(config, _load_json(paths[variant]))
            entry = {
                "type": type(config).__name__,
                "is_deepseek_v3_config": type(config) is DeepseekV3Config,
                "model_type": config.model_type,
                "auto_map": getattr(config, "auto_map", None),
                "rope_parameters": config.rope_parameters,
                "fields": len(rows),
                "unequal": [row for row in rows if not row["equal"]],
                "rope_scaling_row": next(row for row in rows if row["key"] == "rope_scaling"),
                "warnings": [str(item.message)[:300] for item in caught],
                "log_records_warning_or_above": records.messages,
                "config_json_unchanged": open(os.path.join(folder, "config.json"), "rb").read()
                == before,
            }
            try:
                AutoConfig.from_pretrained(folder, trust_remote_code=False)
                entry["autoconfig_alone"] = "loaded (control FAILED)"
            except ValueError as exc:
                entry["autoconfig_alone"] = "ValueError: " + str(exc)[:200]
            native_raw = {k: v for k, v in _load_json(paths[variant]).items() if k != "auto_map"}
            native_raw["model_type"] = "deepseek_v3"
            native_folder = os.path.join(scratch, f"g1-{variant}-native")
            os.makedirs(native_folder)
            with open(os.path.join(native_folder, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(native_raw, fh)
            left = config.to_dict()
            right = AutoConfig.from_pretrained(native_folder).to_dict()
            left.pop("_name_or_path")
            right.pop("_name_or_path")
            entry["autoconfig_equivalent"] = left == right
            if left != right:
                entry["autoconfig_differences"] = sorted(
                    key for key in set(left) | set(right) if left.get(key) != right.get(key)
                )
            out[variant] = entry
    finally:
        dynamic.get_class_from_dynamic_module = real_dynamic[0]
        configuration_auto.get_class_from_dynamic_module = real_dynamic[1]
    out["dynamic_module_calls"] = len(calls)

    thinking = _load_json(paths["thinking"])
    differs = sorted(
        key for key in set(thinking) | set(raw) if thinking.get(key, "<absent>") != raw.get(
            key, "<absent>"
        )
    )
    failures = kimi_k2_guard_failures(thinking)
    out["thinking"] = {
        "keys_differing_from_base": differs,
        "trimmed_constant_matches": all(
            thinking[key] == value for key, value in K2_THINKING_CHANGES.items()
        ),
        "guard_failures": [failure._asdict() for failure in failures],
        "first_failing_field": failures[0].field if failures else None,
    }
    folder = _alone(paths["thinking"], scratch, "g1-thinking")
    try:
        load_stream_model_config(folder)
        out["thinking"]["loader"] = "admitted (FAILED)"
    except UnsupportedKimiK2Error as exc:
        out["thinking"]["loader"] = "UnsupportedKimiK2Error: " + str(exc)[:600]
    out["transformers"] = transformers.__version__
    return out


def index_header_map(raw: dict, index: dict) -> dict:
    """name -> (file, dtype, shape) from the index NAMES and the config (dtype facts: G3)."""
    hidden, heads = raw["hidden_size"], raw["num_attention_heads"]
    q_head = raw["qk_nope_head_dim"] + raw["qk_rope_head_dim"]
    shared = raw["n_shared_experts"] * raw["moe_intermediate_size"]
    moe, dense, experts = raw["moe_intermediate_size"], raw["intermediate_size"], raw[
        "n_routed_experts"
    ]
    block = raw["quantization_config"]["weight_block_size"]
    linears = {
        "self_attn.q_a_proj": (raw["q_lora_rank"], hidden),
        "self_attn.q_b_proj": (heads * q_head, raw["q_lora_rank"]),
        "self_attn.kv_a_proj_with_mqa": (raw["kv_lora_rank"] + raw["qk_rope_head_dim"], hidden),
        "self_attn.kv_b_proj": (heads * (raw["qk_nope_head_dim"] + raw["v_head_dim"]),
                                raw["kv_lora_rank"]),
        "self_attn.o_proj": (hidden, heads * raw["v_head_dim"]),
        "mlp.gate_proj": (dense, hidden), "mlp.up_proj": (dense, hidden),
        "mlp.down_proj": (hidden, dense),
        "mlp.shared_experts.gate_proj": (shared, hidden),
        "mlp.shared_experts.up_proj": (shared, hidden),
        "mlp.shared_experts.down_proj": (hidden, shared),
    }
    norms = {
        "input_layernorm.weight": (hidden,), "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_a_layernorm.weight": (raw["q_lora_rank"],),
        "self_attn.kv_a_layernorm.weight": (raw["kv_lora_rank"],),
    }
    weight_map = index["weight_map"]
    out = {}
    for name, file in weight_map.items():
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            out[name] = (file, "BF16", (raw["vocab_size"], hidden))
            continue
        if name == "model.norm.weight":
            out[name] = (file, "BF16", (hidden,))
            continue
        short = re.match(r"model\.layers\.\d+\.(.+)$", name).group(1)
        if short in norms:
            out[name] = (file, "BF16", norms[short])
        elif short == "self_attn.rotary_emb.inv_freq":
            out[name] = (file, "BF16", (56,))
        elif short == "mlp.gate.weight":
            out[name] = (file, "BF16", (experts, hidden))
        elif short == "mlp.gate.e_score_correction_bias":
            out[name] = (file, "F32", (experts,))
        else:
            expert = re.match(r"mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(weight.*)$",
                              short)
            if expert is not None:
                shape = (moe, hidden) if expert.group(1) != "down_proj" else (hidden, moe)
                part = expert.group(2)
            else:
                module, part = short.rsplit(".", 1)
                shape = linears[module]
            if part == "weight":
                assert name + "_scale_inv" in weight_map, name
                out[name] = (file, "F8_E4M3", shape)
            else:
                assert part == "weight_scale_inv", name
                out[name] = (file, "F32", (-(-shape[0] // block[0]), -(-shape[1] // block[1])))
    return out


def gate_g3(k2_dir: str, scratch: str) -> dict:
    from soup_cli.utils import safetensors_reader
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from soup_cli.utils.safetensors_reader import _DTYPES, TensorRange
    from tests.test_kimi_k2_streaming import k2_cache_oracle, k2_header_map

    base = os.path.join(k2_dir, "base")
    raw = _load_json(os.path.join(base, "config.json"))
    index_path = os.path.join(base, "model.safetensors.index.json")
    index = _load_json(index_path)
    out: dict = {"index_sha256": _sha256(index_path),
                 "index_tensors": len(index["weight_map"]),
                 "index_total_size": index["metadata"]["total_size"]}
    from_index = index_header_map(raw, index)
    from_config = k2_header_map(raw)
    out["map_keys_equal_index"] = set(from_index) == set(index["weight_map"])
    out["map_from_index_equals_map_from_config"] = from_index == from_config
    total = sum(math.prod(shape) * _SIZE[dtype] for _f, dtype, shape in from_index.values())
    out["map_total_bytes"] = total
    out["map_total_equals_index_total_size"] = total == index["metadata"]["total_size"]
    by_dtype: dict = {}
    for _f, dtype, shape in from_index.values():
        by_dtype[dtype] = by_dtype.get(dtype, 0) + math.prod(shape) * _SIZE[dtype]
    out["map_bytes_by_dtype"] = by_dtype

    folder = _alone(os.path.join(base, "config.json"), scratch, "g3-config")
    config = load_stream_model_config(folder)
    arch = stream_arch_of(config)
    started = time.perf_counter()
    probe = build_meta_skeleton(folder, dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    out["meta_probe_seconds"] = round(time.perf_counter() - started, 2)
    del probe
    out["arch"] = arch
    out["quant_suffixes"] = sorted(suffixes)

    by_file: dict = {}
    for name, (file, dtype, shape) in from_index.items():
        by_file.setdefault(file, []).append((name, dtype, shape))

    def _read_header(path):
        entries, cursor = {}, 8
        for name, dtype, shape in by_file[os.path.basename(path)]:
            size = math.prod(shape) * _SIZE[dtype]
            entries[name] = TensorRange(name=name, dtype=_DTYPES[dtype], shape=tuple(shape),
                                        start=cursor, end=cursor + size)
            cursor += size
        return entries

    real_read_header = safetensors_reader.read_header
    safetensors_reader.read_header = _read_header
    try:
        paths = [os.path.join(scratch, "k2-headers", file) for file in sorted(by_file)]
        for label, layout in (("per_expert", "per_expert"), ("fused", "")):
            started = time.perf_counter()
            estimate = estimate_shard_cache_bytes(
                paths, dtype="bfloat16", arch=arch, quant="nf4", quant_suffixes=suffixes,
                double_quant=True, expert_layout=layout, num_hidden_layers=config.num_hidden_layers,
                quantization_config=config.quantization_config,
            )
            seconds = round(time.perf_counter() - started, 2)
            layer_bytes, other_bytes = k2_cache_oracle(from_index, suffixes, layout)
            out[label] = {
                "seconds": seconds,
                "layers": len(estimate.layer_bytes),
                "payload_layers": sum(estimate.layer_bytes),
                "payload_other": estimate.other_bytes,
                "header_bytes": estimate.header_bytes,
                "index_bytes": estimate.index_bytes,
                "expert_file_bytes": estimate.expert_file_bytes,
                "total_bytes": estimate.total_bytes,
                "ram_tier_bytes": estimate.ram_tier_bytes,
                "layer0_bytes": estimate.layer_bytes[0],
                "layer1_bytes": estimate.layer_bytes[1],
                "oracle_layers_equal": estimate.layer_bytes == layer_bytes,
                "oracle_other_equal": estimate.other_bytes == other_bytes,
                "investigation_total": INVESTIGATION_TOTALS[label],
                "delta_vs_investigation": estimate.total_bytes - INVESTIGATION_TOTALS[label],
            }
    finally:
        safetensors_reader.read_header = real_read_header
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k2-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    import psutil
    import torch

    report: dict = {
        "box": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "available_ram_bytes_at_start": psutil.virtual_memory().available,
        }
    }
    with tempfile.TemporaryDirectory(prefix="k2-gate-") as scratch:
        report["g1"] = gate_g1(args.k2_dir, scratch)
        report["g3"] = gate_g3(args.k2_dir, scratch)
    report["box"]["peak_rss_bytes"] = psutil.Process().memory_info().peak_wset if hasattr(
        psutil.Process().memory_info(), "peak_wset"
    ) else None
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
