# Gate record: #2049, resident peak VRAM for the shared estimate

**Hardware.** NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB (torch reports 6.442 GB), driver
610.60, WSL2 (kernel 6.18.33.1, 16,236,180 kB of RAM visible, the default limit) on
Windows 11. torch 2.14.0+cu130, transformers 5.17.0, trl 0.29.1, peft 0.20.0,
bitsandbytes 0.50.2, accelerate 1.15.0, Python 3.12. Soup main `7d355d36`. Every number
below was measured in that configuration; nothing is extrapolated to other hardware.

**What is measured.** `benchmarks/harness/vram_estimate_calibration.py` (sha256
`cd4687bf327086dd5efdd0442a8cd7315ceb37da013dfb3dc35455c95ce898fb`) runs `soup train`'s
own setup (`SFTTrainerWrapper.setup`) and two training steps per row, and reads
`torch.cuda.max_memory_allocated()` over them. Two steps, because AdamW creates its state
inside the first `optimizer.step()`, after that step's backward pass: only from the
second step on do the optimizer state, the gradients and the activations share the
card. Each row runs in its own process (#1693 showed rows built one after another in
one process do not start from the same state). Every row fills `max_length` (the rows
overshoot it three times and the collated length is read back), `val_split: 0`, chat
template `chatml`, LoRA alpha `2r` and the default dropout.

**Models.** Downloaded as pinned snapshots and passed as local directories:
SmolLM2-135M-Instruct `12fd25f7`, SmolLM2-360M-Instruct `a10cc151`,
Qwen2.5-0.5B-Instruct `7ae55760`, Llama-3.2-1B-Instruct (unsloth) `5a8abab4`,
TinyLlama-1.1B-Chat-v1.0 `fe8a4ea1`, Qwen2.5-1.5B-Instruct `989aa798`,
Qwen2.5-3B-Instruct `aa8e7253`. Vocabularies from 32,000 to 151,936.

The raw results are in
[`evidence/vram-2049-resident.json`](evidence/vram-2049-resident.json): the 39 result lines
of the two-step runs in the order printed, every field as the harness printed it, with one
change. The harness names each model by its local snapshot directory (`row.base` and
`model.base`); the file names it by its Hugging Face id (`model`) and full commit
(`revision`) instead, so `row` is renamed `run` and `model` is renamed `config`. No
two-step row was left out: none failed, was void or ran out of memory.

Not in the file: 22 lines from an earlier version of the harness that ran one step per
row. They are the first 22 rows of the table below, measured again with two steps,
because one step misses AdamW's state: the two-step peak is 1.000 to 1.288 times the
one-step one, and on the full fine-tuning row at 1024 x 2 it is 3.085 GB against 2.756.

---

## The rows

"before step" is `memory_allocated()` before the first step (the loaded model and
adapters). "peak" is the measured `max_memory_allocated()`. "estimate" is
`utils/vram_estimate.py` with the model's own `config.json` and safetensors count,
every term except `overhead` (the CUDA context and allocator slack, which
`max_memory_allocated()` does not see). GB is 1e9 bytes.

| # | model | load | adapters | seq x batch | ckpt | optimizer | before step | peak | estimate | est/peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SmolLM2-135M-Instruct | bf16 | r8 auto | 512 x 1 | off | adamw_torch | 0.271 | 0.914 | 0.908 | 0.994 |
| 2 | SmolLM2-135M-Instruct | bf16 | r8 auto | 2048 x 1 | off | adamw_torch | 0.271 | 2.795 | 2.803 | 1.003 |
| 3 | SmolLM2-135M-Instruct | bf16 | r8 auto | 2048 x 1 | on | adamw_torch | 0.271 | 1.576 | 1.599 | 1.014 |
| 4 | SmolLM2-135M-Instruct | bf16 | r8 auto | 1024 x 4 | off | adamw_torch | 0.271 | 5.252 | 5.330 | 1.015 |
| 5 | SmolLM2-135M-Instruct | bf16 | full FT | 512 x 1 | off | adamw_torch | 0.538 | 2.761 | 2.798 | 1.013 |
| 6 | SmolLM2-135M-Instruct | bf16 | full FT | 1024 x 2 | on | adamw_torch | 0.538 | 3.085 | 3.477 | 1.127 |
| 7 | SmolLM2-135M-Instruct | bf16 | full FT | 512 x 1 | off | adamw_bnb_8bit | 0.538 | 2.053 | 1.990 | 0.970 |
| 8 | SmolLM2-135M-Instruct | bf16 | r64 all-linear | 1024 x 2 | off | adamw_torch | 0.347 | 3.874 | 3.863 | 0.997 |
| 9 | SmolLM2-360M-Instruct | bf16 | r8 auto | 2048 x 1 | off | adamw_torch | 0.727 | 4.224 | 4.289 | 1.015 |
| 10 | SmolLM2-360M-Instruct | bf16 | r8 auto | 4096 x 1 | on | adamw_torch | 0.727 | 3.436 | 3.551 | 1.033 |
| 11 | SmolLM2-360M-Instruct | 4-bit | r8 auto | 2048 x 1 | off | adamw_torch | 0.354 | 4.943 | 5.335 | 1.079 |
| 12 | Qwen2.5-0.5B-Instruct | bf16 | r8 auto | 512 x 1 | off | adamw_torch | 0.990 | 2.525 | 2.553 | 1.011 |
| 13 | Qwen2.5-0.5B-Instruct | bf16 | r8 auto | 1024 x 1 | off | adamw_torch | 0.990 | 4.037 | 4.110 | 1.018 |
| 14 | Qwen2.5-0.5B-Instruct | bf16 | r8 auto | 512 x 2 | off | adamw_torch | 0.990 | 4.037 | 4.110 | 1.018 |
| 15 | Qwen2.5-0.5B-Instruct | bf16 | r8 auto | 1024 x 1 | on | adamw_torch | 0.990 | 2.928 | 2.960 | 1.011 |
| 16 | Qwen2.5-0.5B-Instruct | bf16 | r8 auto | 2048 x 1 | on | adamw_torch | 0.990 | 4.845 | 4.923 | 1.016 |
| 17 | Qwen2.5-0.5B-Instruct | 4-bit | r8 auto | 1024 x 1 | off | adamw_torch | 0.732 | 4.815 | 5.037 | 1.046 |
| 18 | Llama-3.2-1B-Instruct | bf16 | r8 auto | 512 x 1 | off | adamw_torch | 2.476 | 4.010 | 4.055 | 1.011 |
| 19 | Llama-3.2-1B-Instruct | bf16 | r8 auto | 1024 x 1 | on | adamw_torch | 2.476 | 4.155 | 4.226 | 1.017 |
| 20 | Llama-3.2-1B-Instruct | 4-bit | r8 auto | 1024 x 1 | off | adamw_torch | 1.556 | 5.929 | 5.982 | 1.009 |
| 21 | Llama-3.2-1B-Instruct | 4-bit | r8 auto | 2048 x 1 | on | adamw_torch | 1.556 | 5.541 | 5.741 | 1.036 |
| 22 | TinyLlama-1.1B-Chat-v1.0 | bf16 | r8 auto | 1024 x 1 | off | adamw_torch | 2.205 | 4.310 | 4.358 | 1.011 |
| 23 | TinyLlama-1.1B-Chat-v1.0 | bf16 | r8 auto | 2048 x 1 | on | adamw_torch | 2.205 | 3.219 | 3.348 | 1.040 |
| 24 | TinyLlama-1.1B-Chat-v1.0 | bf16 | r8 auto | 2048 x 2 | on | adamw_torch | 2.205 | 4.207 | 4.478 | 1.064 |
| 25 | TinyLlama-1.1B-Chat-v1.0 | 4-bit | r8 auto | 2048 x 1 | off | adamw_torch | 1.047 | 6.740 | 7.316 | 1.086 |
| 26 | TinyLlama-1.1B-Chat-v1.0 | bf16 | r16 all-linear | 1024 x 1 | off | adamw_torch | 2.251 | 5.416 | 5.538 | 1.023 |
| 27 | Qwen2.5-1.5B-Instruct | 4-bit | r8 auto | 512 x 1 | off | adamw_torch | 1.623 | 5.011 | 4.996 | 0.997 |
| 28 | Qwen2.5-1.5B-Instruct | 4-bit | r8 auto | 1024 x 1 | on | adamw_torch | 1.623 | 4.165 | 4.099 | 0.984 |
| 29 | Qwen2.5-1.5B-Instruct | bf16 | r8 auto | 512 x 1 | on | adamw_torch | 3.092 | 4.099 | 4.129 | 1.007 |
| 30 | SmolLM2-360M-Instruct | bf16 | r8 auto | 1024 x 2 | off | adamw_torch | 0.727 | 4.224 | 4.289 | 1.015 |
| 31 | SmolLM2-360M-Instruct | 4-bit | r8 auto | 4096 x 1 | on | adamw_torch | 0.354 | 3.407 | 3.666 | 1.076 |
| 32 | Qwen2.5-3B-Instruct | 4-bit | r8 auto | 512 x 1 | on | adamw_torch | 2.693 | 4.430 | 4.054 | 0.915 |
| 33 | Qwen2.5-3B-Instruct | 4-bit | r8 auto | 1024 x 1 | on | adamw_torch | 2.693 | 5.521 | 5.318 | 0.963 |
| 34 | Qwen2.5-3B-Instruct | 4-bit | r8 auto | 512 x 1 | off | adamw_torch | 2.693 | 7.458 | 7.462 | 1.001 |
| 35 | SmolLM2-360M-Instruct | 8-bit | r8 auto | 2048 x 1 | off | adamw_torch | 0.506 | 5.970 | 5.787 | 0.969 |
| 36 | Qwen2.5-0.5B-Instruct | 8-bit | r8 auto | 1024 x 1 | off | adamw_torch | 0.906 | 4.655 | 5.419 | 1.164 |
| 37 | Llama-3.2-1B-Instruct | 8-bit | r8 auto | 512 x 1 | off | adamw_torch | 2.027 | 4.383 | 4.362 | 0.995 |
| 38 | SmolLM2-135M-Instruct | bf16 | full FT | 1024 x 1 | off | adamw_torch | 0.538 | 3.482 | 3.443 | 0.989 |
| 39 | SmolLM2-135M-Instruct | bf16 | full FT | 2048 x 1 | on | adamw_torch | 0.538 | 3.085 | 3.477 | 1.127 |

Seven rows (20, 21, 25, 26, 33, 34 and 35) allocated more than the 5.37 GB the card had
free. Under WSL2 the driver falls back to shared system memory instead of raising, so
those peaks are real allocations, but on a native Linux driver those runs would have
run out of memory.

| rows | n | estimate / peak | allowed |
|---|---|---|---|
| bf16 LoRA | 20 | 0.994 to 1.064 | 0.98 to 1.10 |
| 4-bit | 11 | 0.915 to 1.086 | 0.91 to 1.10 |
| 8-bit | 3 | 0.969 to 1.164 | 0.95 to 1.20 |
| full fine-tuning | 5 | 0.970 to 1.127 | 0.96 to 1.15 |

`tests/test_vram_estimate_calibration.py` holds every row to the allowed band for its
kind of run, and checks that with the logits term removed every row with a vocabulary
of 100k or more falls below its band. The bands are as tight as each kind fits. Each of
these takes at least one row out of its band (checked one at a time): dropping the
checkpoint boundary, the adapters' saved inputs, the fp32 output head under 4-bit and
8-bit, or the extra inputs a trained base keeps; halving the per-hidden,
per-intermediate or adapted-input activation coefficients, the 4-bit and 8-bit extras,
the 4-bit, 8-bit, fp32 and full fine-tuning weight figures, the gradient bytes or the
AdamW state. Terms too small on these rows to move a peak by a band's width are not
pinned by the table: the fp32 adapters at r=8, the 4-bit dequantization copy, and the
checkpoint boundary at half its size.

## What the rows show

1. **The logits and loss chain costs 11.6 bytes per element on these resident runs,**
   fitted jointly with the activations on the 20 bf16 LoRA rows. That is the streaming
   planner's measured loss arithmetic (`layer_stream.LOGITS_LOSS_BYTES_PER_ELEMENT`, 12)
   without the 2-byte retained copy #327 measured on the streaming path
   (`LOGITS_RETENTION_BYTES_PER_ELEMENT`). The estimate imports the 12 and does not
   charge the retention. Checkpointing does not reduce the term.
2. **Activations.** Per token per layer, the bf16 LoRA rows fit 6.1 bf16 values per
   hidden unit, 3.5 per intermediate unit and 1.6 per input feature of each adapted
   module, which is what autograd keeps: RMSNorm's fp32 input, attention's q, k, v and
   output, SiLU's input and output and the up projection, and each adapter's dropout
   output and mask. With checkpointing, one hidden-sized tensor per layer plus one
   layer recomputed.
3. **Under 4-bit and 8-bit loading the parts that are not quantized are fp32.**
   `before step` matches the embeddings and output head at 4 bytes a parameter plus the
   linear layers at 0.51 to 0.53 (4-bit) or 1.00 (8-bit) bytes, within 1.5% on all 14
   rows; the old flat 0.5 and 1.0 under-priced a small model with a large vocabulary.
   The fp32 output head also makes the logits 2 bytes an element dearer, and
   bitsandbytes' linear layers keep their inputs for the backward pass: 3.6 more
   intermediate-sized values per layer at 4-bit, 4.5 at 8-bit.
4. **Full fine-tuning** keeps 0.5 more hidden-sized and 1.3 more intermediate-sized
   values per layer, the inputs of the trained linear layers.
5. **The worst row is 4-bit Qwen2.5-3B with checkpointing at 512 tokens, 0.915.** The
   same model at 1024 tokens is 0.963, so the shortfall does not grow with the tokens;
   it is left as measured rather than fitted away.
6. **The gate records named in #2049 do not calibrate the resident total.**
   `gate-v0.73.1-measured-vram-fit.md` and `gate-395-second-stack-vram.md` are streamed
   runs (`stream_buffers: 2`), which the resident pre-flight skips.

## What the rows do not cover

- **Only checkpointing off and on (`full`) was measured.** The `medium` tier (every
  other layer) is priced from those two: the layers it keeps at the off cost, the rest
  at the boundary, plus the one being recomputed. `selective` (attention only) is
  priced as no checkpointing, an upper bound. Neither tier has a row.
- **Seven dense decoders of 135M to 3B parameters,** at up to 4,096 tokens and batch 4,
  on one card and one stack. No model of 7B or more, and no mixture-of-experts model,
  which the estimate sizes as dense.
- **The coefficients are fitted on these same 39 rows,** so the bands above show how
  well they fit them, not the error on a run they were not fitted to.

## Reproduce

```bash
python benchmarks/harness/vram_estimate_calibration.py --rows rows.json --out out.jsonl
```

with each row `{"base": "<snapshot dir>", "quantization": ..., "lora_r": ...,
"target_modules": ..., "seq": ..., "batch": ..., "gradient_checkpointing": ...,
"optimizer": ...}` as listed in the evidence file.
