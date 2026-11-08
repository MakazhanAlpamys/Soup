"""Untimed production-stream check for #842. Executed by nf4_stream_bench --model."""

from __future__ import annotations

import pathlib
import tempfile


def model_check(count_paths, difference, hardware) -> dict:
    import bitexact
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, LlamaConfig
    from variant2_gate import build_streamed_arm

    import soup_cli.utils.layer_stream_runtime as runtime
    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    layers, tokens = 4, 3
    torch.manual_seed(842)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=4096,
        intermediate_size=26624,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=layers,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        use_cache=False,
    )
    report = {
        "hardware": hardware(),
        "tokens": tokens,
        "shape": config.to_dict(),
        "buffers": 2,
        "scale_cache": False,
        "checkpoint_recomputation": True,
        "arms": {},
    }

    def run(model, ids, counts):
        model.train()
        counts.clear()
        logits = model(input_ids=ids, use_cache=False).logits
        forward = dict(counts)
        counts.clear()
        logits.float().square().mean().backward()
        grads = bitexact.collect_lora_grads(model)
        expected = {
            name for name, _ in runtime.canonical_named_parameters(model) if "lora_" in name
        }
        if not expected or set(grads) != expected:
            raise ValueError("missing adapter gradients")
        return logits.detach(), grads, {"forward": forward, "backward": dict(counts)}

    with tempfile.TemporaryDirectory() as tmp:
        weights = str(pathlib.Path(tmp) / "weights")
        shards = str(pathlib.Path(tmp) / "shards")
        with torch.device("cuda"):
            base = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        base.cpu().save_pretrained(weights)
        del base
        skeleton = runtime.build_meta_skeleton(weights, dtype="float16", quant="nf4")
        suffixes = runtime.quantised_layer_suffixes(skeleton)
        del skeleton
        index = shard_checkpoint(
            weights,
            shards,
            dtype="float16",
            arch="llama",
            quant="nf4",
            quant_suffixes=suffixes,
            quant_device="cuda",
        )
        sizes = [pathlib.Path(layer_shard_path(shards, i)).stat().st_size for i in range(layers)]
        if min(sizes) <= 165 * 1024**2:
            raise ValueError(f"sub-threshold layer shards: {sizes}")
        report["layer_file_bytes"] = sizes
        resident = AutoModelForCausalLM.from_pretrained(
            weights,
            quantization_config=runtime.build_nf4_config("float16"),
            dtype=torch.float16,
            device_map={"": 0},
        )
        resident.requires_grad_(False)
        resident = get_peft_model(resident, bitexact.lora_config())
        bitexact.make_non_vacuous_lora(resident)
        bitexact.enable_resident_checkpointing(resident)
        ids = torch.randint(0, config.vocab_size, (1, tokens), device="cuda")

        def build():
            return runtime.build_streamed_model(
                model_id=weights,
                shard_dir=shards,
                index=index,
                lora_config=bitexact.lora_config(),
                device="cuda",
                dtype="float16",
                buffers=2,
                pin=True,
                require_pin=True,
                seed=842,
                quant="nf4",
            )

        with count_paths() as counts:
            ref, ref_grads, ref_paths = run(resident, ids, counts)
            report["resident_paths"] = ref_paths
            for arm in ("unsafe", "repair"):
                model, state = build_streamed_arm(
                    repair_enabled=(arm == "repair"), runtime_module=runtime, build_model=build
                )
                try:
                    copied = bitexact.copy_lora(resident, model)
                    logits, grads, paths = run(model, ids, counts)
                    if copied != len(ref_grads) or grads.keys() != ref_grads.keys():
                        raise ValueError("resident and streamed adapter sets differ")
                    forward = difference(logits, ref)
                    backward = {
                        name: difference(grad, ref_grads[name]) for name, grad in grads.items()
                    }
                    all_exact = all(g["exact"] for g in backward.values())
                    report["arms"][arm] = {
                        "forward": forward,
                        "backward": backward,
                        "paths": paths,
                        "pinned": state.pinned,
                        "pass": state.pinned
                        and forward["exact"]
                        and all(g["finite"] for g in backward.values())
                        and all_exact == (arm == "repair")
                        and paths["forward"].get("kernel") == layers * len(suffixes),
                    }
                finally:
                    state.close()
                del model, state
    report["pass"] = ref_paths["forward"].get("kernel") == layers * len(suffixes) and all(
        row["pass"] for row in report["arms"].values()
    )
    return report
