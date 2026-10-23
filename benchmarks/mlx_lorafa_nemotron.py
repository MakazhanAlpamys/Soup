"""Offline, pinned Nemotron BF16 lifecycle comparison; run each arm in a fresh process.

Uses the same six rows/rank/dtype/token budget and measures complete CLI training,
first/remaining report intervals, Gram setup, Metal/RSS peaks and optimizer sizes.
No model download. Artifacts remain under --output; never overwrite the base.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pinned Nemotron BF16 LoRA/FA lifecycle measurement"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["lora", "lorafa"], required=True)
    arguments = parser.parse_args()

    import hashlib
    import importlib.metadata as md
    import importlib.util
    import json
    import os
    import resource
    import subprocess
    import sys
    import time
    from pathlib import Path

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from soup_cli.utils.mlx_generation import generate_native, load_native_model

    base = Path(arguments.model)
    fa = arguments.method == "lorafa"
    root = Path(arguments.output)
    root.mkdir(parents=True, exist_ok=False)
    root = root.absolute()
    source = root / "train.jsonl"
    source.write_text('{"text":"Question: What is 2 plus 2? Answer: 4."}\n' * 6)
    cfg = SoupConfig(
        base=str(base),
        backend="mlx",
        output=str(root / "adapter"),
        data={
            "train": str(source),
            "format": "plaintext",
            "val_split": 0,
            "max_length": 64,
            "train_on_responses_only": False,
        },
        training={
            "quantization": "none",
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "lr": 0.0001,
            "logging_steps": 1,
            "save_steps": 1,
            "scheduler": "constant",
            "warmup_ratio": 0,
            "weight_decay": 0,
            "use_lorafa": fa,
            "gradient_checkpointing": True,
            "rewind_log": False,
            "lora": {
                "r": 4,
                "alpha": 8,
                "dropout": 0,
                "target_modules": ["mixer.q_proj", "mixer.v_proj"],
                "use_rslora": False,
            },
        },
    )
    config = root / "config.json"
    config.write_text(cfg.model_dump_json())
    config_before = (base / "config.json").read_bytes()
    observed = {}
    setup = MLXSFTTrainerWrapper.setup
    apply = MLXSFTTrainerWrapper._apply_lora

    def capture_setup(self, data):
        setup(self, data)
        observed["wrapper"] = self
        observed["base"] = dict(tree_flatten(self.model.parameters()))
        mx.eval(observed["base"])

    def capture_apply(self, model):
        apply(self, model)
        observed["initial"] = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(observed["initial"])
        for name, module in model.named_modules():
            if name.endswith("mixer.q_proj"):
                assert module.lora_a.shape[1] == 4 and module.scale == 2
            elif name.endswith("mixer.v_proj"):
                assert module.lora_a.shape[1] == 4 and module.scale == 2

    MLXSFTTrainerWrapper.setup = capture_setup
    MLXSFTTrainerWrapper._apply_lora = capture_apply
    import mlx_lm.tuner.trainer as native
    import numpy as np

    import soup_cli.trainer.mlx_lorafa as fa_module

    np.random.seed(42)
    mx.random.seed(42)
    step_times = []
    loop_metrics = {}

    def measure_loop(function):
        def run(**kwargs):
            optimizer = kwargs["optimizer"]._inner
            loop_metrics["setup_gram_seconds"] = getattr(optimizer, "setup_seconds", 0.0)
            loop_metrics["trainable_tensor_count"] = len(
                dict(tree_flatten(kwargs["model"].trainable_parameters()))
            )
            loop_metrics["trainable_elements"] = sum(
                v.size for _, v in tree_flatten(kwargs["model"].trainable_parameters())
            )
            callback = kwargs["training_callback"]
            report = callback.on_train_loss_report
            last = [time.perf_counter()]

            def record(info):
                now = time.perf_counter()
                step_times.append(now - last[0])
                last[0] = now
                loop_metrics["trained_tokens"] = info["trained_tokens"]
                report(info)

            callback.on_train_loss_report = record
            beginning = time.perf_counter()
            result = function(**kwargs)
            loop_metrics["loop_seconds"] = time.perf_counter() - beginning
            moments = [v for n, v in tree_flatten(optimizer.state) if n.endswith((".m", ".v"))]
            loop_metrics["moment_tensor_count"] = len(moments)
            loop_metrics["moment_elements"] = sum(v.size for v in moments)
            return result

        return run

    native.train = measure_loop(native.train)
    fa_module.train_lorafa = measure_loop(fa_module.train_lorafa)
    start = time.monotonic()
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    (root / "train.log").write_text(result.output + "\n" + repr(result.exception))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    train_seconds = time.monotonic() - start
    train_peak = mx.get_peak_memory()
    train_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = observed["wrapper"].model
    model.eval()
    after = dict(tree_flatten(model.parameters()))
    for name, value in observed["base"].items():
        parent, leaf = name.rsplit(".", 1)
        actual = after.get(parent + ".linear." + leaf, after.get(name))
        assert mx.array_equal(actual, value).item(), name
    assert len(observed["base"]) == 263
    assert all(v.dtype == mx.bfloat16 for v in observed["base"].values())
    trainable = dict(tree_flatten(model.trainable_parameters()))
    changed = [
        n for n, v in trainable.items() if not mx.array_equal(v, observed["initial"][n]).item()
    ]
    assert len(changed) == len(trainable) == (8 if fa else 16)
    if fa:
        assert all(
            mx.array_equal(v, after[n]).item()
            for n, v in observed["initial"].items()
            if n.endswith(".lora_a")
        )
    saved = mx.load(str(root / "adapter/adapters.safetensors"))
    assert saved.keys() == observed["initial"].keys()
    assert all(mx.isfinite(v).all().item() for v in trainable.values())
    probe = mx.array([[1, 2, 3, 4]])
    expected = model(probe)
    mx.eval(expected)
    scales = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_a"):
            scales.append((module, module.scale))
            module.scale = 0
    unadapted = model(probe)
    mx.eval(unadapted)
    for module, scale in scales:
        module.scale = scale
    effect = mx.abs(expected - unadapted).max().item()
    assert effect > 0
    mx.save(str(root / "expected.npy"), expected)
    script = (
        "import sys;import mlx.core as mx;"
        "from soup_cli.utils.mlx_generation import load_native_model;"
        "m,_=load_native_model(sys.argv[1]);"
        "mx.save(sys.argv[2],m(mx.array([[1,2,3,4]])))"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(root / "adapter"), str(root / "fresh.npy")],
        check=True,
        timeout=120,
        env=os.environ | {"PYTHONPATH": str(Path.cwd() / "src")},
        capture_output=True,
        text=True,
    )
    error = mx.abs(mx.load(str(root / "fresh.npy")) - expected).max().item()
    assert error == 0, error
    loaded, tok = load_native_model(str(root / "adapter"))
    completion, count = generate_native(
        loaded,
        tok,
        [{"role": "user", "content": "What is 2 + 2? Reply briefly."}],
        temperature=0,
        max_tokens=16,
    )
    (root / "prompts.jsonl").write_text('{"prompt":"What is 2 + 2? Reply briefly."}\n')
    result = CliRunner().invoke(
        app,
        [
            "infer",
            "--backend",
            "mlx",
            "--model",
            str(root / "adapter"),
            "--input",
            str(root / "prompts.jsonl"),
            "--output",
            str(root / "inference.jsonl"),
            "--temperature",
            "0",
            "--max-tokens",
            "16",
        ],
    )
    (root / "infer.log").write_text(result.output + "\n" + repr(result.exception))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    row = json.loads((root / "inference.jsonl").read_text())
    assert row["response"] == completion and row["tokens_generated"] == count
    assert config_before == (base / "config.json").read_bytes()
    assert all(importlib.util.find_spec(n) is None for n in ("torch", "peft", "trl"))
    receipt = {
        "model": str(base),
        "dtype": "BF16",
        "base_tensor_count": 263,
        "all_base_unchanged": True,
        "changed": changed,
        "metadata": json.loads((root / "adapter/adapter_config.json").read_text()),
        "reload_max_error": error,
        "adapter_logit_effect": effect,
        "completion": completion,
        "tokens": count,
        "elapsed_seconds": time.monotonic() - start,
        "train_seconds": train_seconds,
        "training_metal_peak_bytes": train_peak,
        "training_rss_peak_bytes": train_rss,
        "measurement": loop_metrics,
        "report_interval_seconds": step_times,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "arm": "lorafa" if fa else "ordinary_lora",
        "peak_bytes": mx.get_peak_memory(),
        "versions": {n: md.version(n) for n in ["mlx", "mlx-lm", "transformers", "datasets"]},
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_before).hexdigest(),
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    from rich.console import Console

    Console().print_json(data=receipt)


if __name__ == "__main__":
    main()
