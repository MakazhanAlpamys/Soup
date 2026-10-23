"""Save a synthetic LoRA adapter (non-zero B) so the adapter path can be measured.

    python cuda_graph_decode_synthetic_lora.py --base BASE_DIR --output ADAPTER_DIR

soup infer loads ``ADAPTER_DIR`` like a training output: adapter_config.json names the
base, and the tokenizer is saved beside it. The base config.json is copied too,
because the decode harness fingerprints it; the loader ignores it, since
adapter_config.json takes precedence. lora_B is scaled down so the text stays close
to the base model and generation reaches the full token budget.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--b-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            init_lora_weights=False,
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.mul_(args.b_scale)
    output = Path(args.output)
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(output)
    shutil.copyfile(Path(args.base) / "config.json", output / "config.json")
    print(f"saved {output} over base {args.base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
