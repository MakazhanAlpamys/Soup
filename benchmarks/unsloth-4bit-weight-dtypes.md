# Unsloth 4-bit embedding and output-head dtype (#2301)

Measured on 2026-10-30, against Soup main
`7c546b22909ad8b4db4631331105306307256a57`.

## Hardware and software

- NVIDIA GeForce RTX 5090, compute capability 12.0, driver 570.211.01.
- Python 3.11.15, torch 2.8.0+cu128 (CUDA build 12.8), Triton 3.4.0.
- Unsloth 2026.10.10, unsloth-zoo 2026.10.10, transformers 5.17.0,
  peft 0.21.2, bitsandbytes 0.50.2.
- Model: `unsloth/Meta-Llama-3.1-8B-bnb-4bit`, revision
  `70d37f62f4f475a5a42dbbd5b7aad38f420e0960`, downloaded and passed as a local
  snapshot. The weight file's SHA-256 was verified as
  `aa838c03b42f11bdd06d74c5553bb294c2ad1624a5cf1152dbdf00eeebf42684`.

## Observation

`FastLanguageModel.from_pretrained` used `max_seq_length=2048`, `dtype=None`,
`load_in_4bit=True`. Then `FastLanguageModel.get_peft_model` used `r=16`,
`lora_alpha=16`, `lora_dropout=0.05`, the seven targets below, `bias="none"`,
`use_gradient_checkpointing="unsloth"`, and `use_rslora=False`, as Soup's loader does.

| weight | shape | after load | after `get_peft_model` | bytes per element |
|---|---|---|---|---|
| input embedding | `[128256, 4096]` | `torch.bfloat16` | `torch.bfloat16` | 2 |
| output head | `[128256, 4096]` | `torch.bfloat16` | `torch.bfloat16` | 2 |

The two weights did not share storage. All seven targets (`q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) received adapters in all
32 layers: 224 modules. The machine-readable observation is in
[`evidence/unsloth-4bit-weight-dtypes.json`](evidence/unsloth-4bit-weight-dtypes.json).

This is a real weight load and LoRA injection, not a training-peak measurement.
Unsloth warned about xformers 0.0.32.post2's SM 12.0 FA3 dispatch and disabled
xformers (`Xformers = None`); the load and injection completed successfully.
No forward or backward pass was run. The existing activation/logits coefficients
remain uncalibrated for Unsloth, which still warns outside the calibrated range.

The estimator pins 2 bytes for both of these weights under Unsloth 4-bit and
retains the Transformers fp32 figure. It imports no Unsloth or torch and performs
no runtime model load. FP16 also occupies 2 bytes; this record directly observes
BF16 on this model and stack, not every Unsloth model or future version.

## CPU regression and floor

`tests/test_unsloth_vram_estimate.py` uses only a local `config.json` with the
Llama-3.1-8B dimensions, no checkpoint and no network. With rank 16 and
`adamw_torch`, the fixed expectations are:

| backend | auto targets | weights (bytes) | adapters (bytes) | floor (GB, 1e9 bytes) |
|---|---|---:|---:|---:|
| transformers | q, v | 7,901,874,299 | 27,262,976 | 8.510926203 |
| unsloth | seven projections | 5,800,527,995 | 167,772,160 | 6.971616635 |

Both use 8,030,261,248 parameters from the config, of which 1,050,673,152 belong
to the separate embedding and head. The floor adds the 500,000,000-byte CUDA
context and adapter, gradient and AdamW state bytes (4 + 4 + 8 per trainable
parameter). Activations, logits, dequantization and allocator slack are excluded.

## Reproduce the dtype observation

Download the pinned snapshot above first, then pass its path as the first argument
to a script containing the following. Import Unsloth before transformers or peft.

```python
import sys

from rich.console import Console


def main() -> None:
    from unsloth import FastLanguageModel

    console = Console()
    model, _ = FastLanguageModel.from_pretrained(
        model_name=sys.argv[1], max_seq_length=2048, dtype=None, load_in_4bit=True
    )

    def observe(stage: str) -> None:
        embedding = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        console.print(stage, {
            "embedding": (tuple(embedding.shape), str(embedding.dtype), embedding.element_size()),
            "head": (tuple(head.shape), str(head.dtype), head.element_size()),
            "shared_storage": embedding.data_ptr() == head.data_ptr(),
        })

    observe("after_load")
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", use_rslora=False,
    )
    observe("after_get_peft_model")


if __name__ == "__main__":
    main()
```
