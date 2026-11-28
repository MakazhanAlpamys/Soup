"""Isolated FlashAttention plus persistent-layer study; never provisions hardware."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def forced_attention_backend(backend: str):
    """Select attention at construction without editing upstream streaming source."""
    from transformers import AutoModelForCausalLM

    if backend not in {"sdpa", "flash_attention_2"}:
        raise ValueError("unsupported research backend")
    model_class = AutoModelForCausalLM
    had_local = "from_config" in model_class.__dict__
    descriptor = model_class.__dict__.get("from_config")
    original = model_class.from_config
    state = {"requested": backend, "constructed_backends": []}

    def construct(_cls: Any, config: Any, **kwargs: Any):
        if "attn_implementation" in kwargs:
            raise ValueError("attention override already present")
        model = original(config, attn_implementation=backend, **kwargs)
        selected = model.config._attn_implementation
        if selected != backend:
            raise ValueError("requested attention backend did not activate")
        state["constructed_backends"].append(selected)
        return model

    model_class.from_config = classmethod(construct)
    try:
        yield state
    finally:
        if had_local:
            model_class.from_config = descriptor
        else:
            delattr(model_class, "from_config")


def augment_semantics(module: Any) -> dict:
    """Add the same profiled unmasked forward to both arms; retain every old check."""
    source = inspect.getsource(module.attention_semantics)
    tree = ast.parse(source)

    def profile_block(value: Any):
        return next(node for node in ast.walk(value) if isinstance(node, ast.With)
                    and any(isinstance(item.optional_vars, ast.Name)
                            and item.optional_vars.id == "profile" for item in node.items))

    additions = ast.parse(
        'with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):\n'
        ' extra = wrapper.model(input_ids=inputs[:1],'
        ' attention_mask=torch.ones_like(mask[:1]), position_ids=positions[:1]).logits\n'
        'tensors["logits/unmasked_fastpath"] = extra.detach().cpu().clone()\n'
        'torch.cuda.synchronize()\n'
    ).body
    profile_block(tree).body.extend(additions)
    stripped = copy.deepcopy(tree)
    del profile_block(stripped).body[-len(additions):]
    assert ast.dump(stripped) == ast.dump(ast.parse(source))
    ast.fix_missing_locations(tree)
    namespace = module.__dict__
    exec(compile(tree, "<additional-unmasked-semantic-forward>", "exec"), namespace)
    module.attention_semantics = namespace["attention_semantics"]
    return {"original_semantic_AST_unchanged": True,
            "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "added_same_unmasked_forward_to_all_arms": True}


def main() -> None:
    parser = __import__("argparse").ArgumentParser(add_help=False)
    parser.add_argument("--attention", choices=["sdpa", "external", "hybrid"], required=True)
    selected, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    output = Path(remaining[remaining.index("--output") + 1])
    record = {"attention": selected.attention, "thresholds_changed": False, "status": "STARTED"}
    external = selected.attention != "sdpa"
    context = contextlib.nullcontext(None)
    if external:
        prefix = Path("/root/liger/fa274-nofast")
        sys.path.insert(0, str(prefix))
        import flash_attn
        import flash_attn_2_cuda

        assert flash_attn.__version__ == "2.7.4"
        for path in [flash_attn.__file__, flash_attn_2_cuda.__file__]:
            assert os.path.commonpath([os.path.realpath(prefix), os.path.realpath(path)]) == str(
                prefix
            )
        record.update(external_version=flash_attn.__version__,
                      external_binary_sha256=hashlib.sha256(
                          Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest())
        if selected.attention == "hybrid":
            from benchmarks.harness.streaming_attention_candidate import (
                native_masks_external_unmasked,
            )

            context = native_masks_external_unmasked()
    from benchmarks.harness import streaming_optimization_bench as semantics
    from benchmarks.harness.streaming_residency_bench import main as run

    if "--semantics" in remaining:
        record.update(augment_semantics(semantics))
    try:
        with forced_attention_backend("flash_attention_2" if external else "sdpa") as state:
            with context as counts:
                run()
                if counts is not None:
                    record["route_counts"] = dict(counts)
                    assert counts["external_unmasked_calls"] > 0, "external route never executed"
                    if "--semantics" in remaining:
                        assert counts["native_explicit_mask_calls"] > 0
            record["construction"] = state
            assert state["constructed_backends"]
        record["status"] = "EXECUTED_NOT_AN_ACCURACY_OR_SPEED_VERDICT"
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "attention-residency-provenance.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
