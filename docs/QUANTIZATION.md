# Quantization Menu (v0.38.0)

Soup supports 9 train-time quantization formats. Pick one with
`training.quantization` in your `soup.yaml`:

| Value | Format | Bits | Use case | Optional dep |
|---|---|---|---|---|
| `4bit` | BNB NF4 (QLoRA) | 4 | Default. Best general LoRA training. | bitsandbytes |
| `8bit` | BNB int8 | 8 | Larger memory budget, more accurate gradients. | bitsandbytes |
| `none` | Full precision | 16/32 | Full fine-tuning or DPO/PPO without quant. | — |
| `gptq` | Pre-quantized GPTQ | 2/3/4/8 | Train LoRA on top of an existing GPTQ checkpoint. | gptqmodel |
| `awq` | Pre-quantized AWQ | 4 | Train LoRA on top of an existing AWQ checkpoint. | autoawq |
| `hqq:Nbit` | HQQ | 1, 2, 3, 4, 5, 6, 8 | Wide bit range; compose with LoRA. | hqq |
| `aqlm` | AQLM | 2 | Extreme compression. | aqlm |
| `eetq` | EETQ | 8 | Fast 8-bit kernel for SM75+. | eetq |
| `mxfp4` | BNB MXFP4 | 4 | Newer 4-bit type with better activation distribution. | bitsandbytes ≥ 0.45 |
| `fp8` | FP8 dequant-on-load | — | Train fp16/bf16 on top of FP8-released checkpoints. | transformers ≥ 4.45 |

## Compatibility matrix — quant × multi-GPU

`soup train` runs `check_quant_distributed_compat()` at startup and
prints any incompatibilities. Hard rejections come from the upstream
implementations (LlamaFactory's `quantization.py` enforces the same
matrix).

| Format | DDP | FSDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| `4bit` | ✅ | ✅ (set `bnb_4bit_quant_storage`) | ✅ | ✅ | ✅ |
| `8bit` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `gptq` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `awq` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hqq:*` | ✅ | ❌ | ✅ | ✅ | ❌ |
| `aqlm` | ✅ | ❌ | ✅ | ✅ | ❌ |
| `eetq` | ✅ | ❌ | ✅ | ✅ | ❌ |
| `mxfp4` | ✅ | ✅ (set `bnb_4bit_quant_storage`) | ✅ | ✅ | ✅ |
| `fp8` | ✅ | ✅ | ✅ | ✅ | ✅ |

## FSDP + QLoRA — set `bnb_4bit_quant_storage`

When training with `quantization: 4bit` (or `mxfp4`) under FSDP, set
`bnb_4bit_quant_storage` to your compute dtype:

```yaml
training:
  quantization: 4bit
  bnb_4bit_quant_storage: bfloat16   # or float16
```

Without this, FSDP's all-gather upcasts the packed 4-bit codes to fp32 →
2-3x slowdown plus risk of silent NaNs. The setting is the same one
LlamaFactory calls "crucial for fsdp+qlora" (`quantization.py:178`).

## Pre-quantized formats need a pre-quantized base

`gptq` / `awq` / `aqlm` / `eetq` / `fp8` all expect the base model to
already be quantized. Soup runs a pre-flight check on local paths and
falls through for HF repo IDs (where HF surfaces the failure if the
referenced repo isn't actually quantized). To produce a quantized base
yourself, use `soup export --format gptq` (or `awq`).

## Combining with `quantization_aware`

`quantization_aware: true` (int8 QAT) and `quantization_aware: 'fp8'`
(FP8 training on Hopper+ GPUs) are **mutually exclusive** with every
pre-quantized format. The schema rejects the combination at config-load.

## Multi-trainer wiring (status)

v0.38.0 wires the quant menu into the SFT trainer only. Multi-trainer
expansion (DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO / RewardModel /
Pretrain / Embedding) is tracked for v0.38.1 — same stub-then-live
pattern as v0.27.0 MII and v0.37.0 multipack.
