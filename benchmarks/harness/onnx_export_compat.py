"""Reproduce the bounded ONNX compatibility probes of issue #2676, without downloads.

The native export is a lab prototype, not a new Soup export backend. Run ``create``
in the training environment first. ``legacy-copy`` changes only a synthetic
fixture's generic tokenizer class label, in a separate copy, for the export-only
4.x environment; it is not a conversion tool for arbitrary real tokenizers.
See benchmarks/onnx-export-compatibility-2026-11-11.md for commands and limits.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console


def versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "torch",
        "transformers",
        "optimum",
        "optimum-onnx",
        "onnx",
        "onnxruntime",
        "numpy",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def create(root: Path) -> None:
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    root.mkdir(parents=True, exist_ok=False)
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "Hello": 2,
        "world": 3,
        "a": 4,
        "test": 5,
        "example": 6,
        "model": 7,
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
        model_input_names=["input_ids", "attention_mask"],
    )
    prompts = ["Hello", "Hello world", "a test example", "Hello world a test"]
    (root / "prompts.txt").write_text("\n".join(prompts) + "\n")
    configs = {
        "gpt2": (
            GPT2LMHeadModel,
            GPT2Config(
                vocab_size=8,
                n_positions=64,
                n_embd=32,
                n_layer=1,
                n_head=4,
                use_cache=False,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=1,
            ),
        ),
        "llama": (
            LlamaForCausalLM,
            LlamaConfig(
                vocab_size=8,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                rope_theta=10000.0,
                use_cache=False,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=1,
            ),
        ),
    }
    manifest = {"versions": versions(), "seed": 2676, "prompts": prompts, "models": {}}
    for name, (cls, cfg) in configs.items():
        path = root / name
        with torch.random.fork_rng():
            torch.manual_seed(2676)
            model = cls(cfg).float().eval()
        model.config._attn_implementation = "eager"
        model.save_pretrained(path)
        tok.save_pretrained(path)
        arrays = {}
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                tensors = tok(prompt, return_tensors="pt", padding=False, truncation=False)
                arrays[f"input_ids_{i}"] = tensors["input_ids"].numpy()
                arrays[f"attention_mask_{i}"] = tensors["attention_mask"].numpy()
                arrays[f"logits_{i}"] = model(**tensors, use_cache=False).logits.numpy()
        np.savez(root / f"{name}-reference.npz", **arrays)
        manifest["models"][name] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.iterdir()
            if p.is_file()
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    Console().print_json(json.dumps(manifest))


def inspect_graph(
    root: Path,
    name: str,
    graphdir: Path,
    report: Path,
    source: Path | None = None,
    *,
    backend: str = "existing graph",
) -> int:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from soup_cli.eval.onnx_check import compare_logits, validate_onnx_export

    prompts = (root / "prompts.txt").read_text().splitlines()
    same_stack = validate_onnx_export(source or root / name, graphdir, prompts, atol=1e-4)
    graph = next(graphdir.glob("*.onnx"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(graph), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    references = np.load(root / f"{name}-reference.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(source or root / name), local_files_only=True)
    pairs = []
    for i in range(len(prompts)):
        tokenized = tokenizer(prompts[i], return_tensors="np", padding=False, truncation=False)
        np.testing.assert_array_equal(tokenized["input_ids"], references[f"input_ids_{i}"])
        feed = {}
        for entry in session.get_inputs():
            if entry.name == "position_ids":
                feed[entry.name] = np.arange(references[f"input_ids_{i}"].shape[1], dtype=np.int64)[
                    None
                ]
            else:
                feed[entry.name] = references[f"{entry.name}_{i}"]
        pairs.append(
            (references[f"logits_{i}"], session.run(output_names=["logits"], input_feed=feed)[0])
        )
    cross = compare_logits(pairs, atol=1e-4)
    record = {
        "versions": versions(),
        "model": name,
        "reference_source": str(source or root / name),
        "backend": backend,
        "tokens_match_training_reference": True,
        "optimum_imported": any(x == "optimum" or x.startswith("optimum.") for x in sys.modules),
        "same_stack": vars(same_stack),
        "cross_train_stack": vars(cross),
        "pass": same_stack.passed and cross.passed,
        "inputs": [
            {"name": x.name, "type": x.type, "shape": x.shape} for x in session.get_inputs()
        ],
    }
    report.write_text(json.dumps(record, indent=2))
    Console().print_json(json.dumps(record))
    return 0 if record["pass"] else 2


def native_export(source: Path, output: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, position_ids):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits

    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    ids = torch.tensor([[2, 3, 4]])
    output.mkdir(parents=True, exist_ok=False)
    torch.onnx.export(
        Logits(model),
        (ids, torch.ones_like(ids), torch.arange(3)[None]),
        str(output / "model.onnx"),
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamo=False,
        opset_version=18,
        dynamic_axes={
            name: {1: "sequence"}
            for name in ("input_ids", "attention_mask", "position_ids", "logits")
        },
    )


def legacy_copy(root: Path, name: str, output: Path) -> None:
    """Prepare only this synthetic generic-tokenizer fixture for the 4.x loader."""
    source = root / name
    config = json.loads((source / "tokenizer_config.json").read_text())
    if config["tokenizer_class"] != "TokenizersBackend":
        raise ValueError("This fixture lane expects the Transformers 5 generic TokenizersBackend.")
    shutil.copytree(source, output)
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (output / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    assert (output / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()


def corrupt(graphdir: Path, output: Path) -> None:
    """Inject a real numerical error into a copied graph; retain the original."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    shutil.copytree(graphdir, output)
    path = next(output.glob("*.onnx"))
    graph = onnx.load(path)
    changed = []
    for tensor in graph.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            values = numpy_helper.to_array(tensor).copy() + np.float32(0.5)
            tensor.CopyFrom(numpy_helper.from_array(values, name=tensor.name))
            changed.append(tensor.name)
    assert changed
    onnx.save(graph, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["create", "inspect", "native", "legacy-copy", "corrupt"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", choices=["gpt2", "llama"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "create":
        create(args.root)
        return 0
    if args.name is None:
        parser.error("--name is required outside create")
    if args.mode in ("native", "inspect") and (args.graph is None or args.report is None):
        parser.error("--graph and --report are required for native/inspect")
    if args.mode in ("legacy-copy", "corrupt") and args.output is None:
        parser.error("--output is required for legacy-copy/corrupt")
    if args.mode == "legacy-copy":
        legacy_copy(args.root, args.name, args.output)
        return 0
    if args.mode == "corrupt":
        if args.graph is None:
            parser.error("--graph is required for corrupt")
        corrupt(args.graph, args.output)
        return 0
    if args.mode == "native":
        native_export(args.root / args.name, args.graph)
    return inspect_graph(
        args.root,
        args.name,
        args.graph,
        args.report,
        args.source,
        backend="torch.onnx.export" if args.mode == "native" else "existing graph",
    )


if __name__ == "__main__":
    raise SystemExit(main())
