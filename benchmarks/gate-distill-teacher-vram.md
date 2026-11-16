# Gate record: #ISSUE, the distillation teacher and divergence in the VRAM estimate

**Hardware.** NVIDIA RTX PRO 5000 Blackwell (sm_120), 48 GB, driver 580.105.08, Linux. The card
was shared: other users' processes held about 38.7 GB throughout, so every row ran with
`torch.cuda.set_per_process_memory_fraction(0.19)` (a 9.0 GiB cap), one row at a time. torch
2.11.0+cu130, transformers 5.16.1, peft 0.20.0, trl 0.29.1, accelerate 1.14.0, Python 3.13.
Soup main `1ac7b400`; the trainer is unchanged by this fix, so the rows hold for it too.

**What is measured.** `benchmarks/harness/vram_estimate_distill.py` runs `soup train`'s own
setup (`DistillTrainerWrapper.setup`) and two steps per row in a fresh process, and reads
`torch.cuda.max_memory_allocated()` over them, the protocol of
[`gate-2049-resident-vram.md`](gate-2049-resident-vram.md). Every row fills `max_length`
(three times over, truncated), `val_split: 0`, chat template `chatml`, LoRA r8 alpha 16 on
`q_proj`/`v_proj`, `adamw_torch`, temperature 2.0. "before step" is the student, its adapters
and the teacher; "main estimate" is `utils/vram_estimate.py` on `main`, which read only `base`;
"estimate" is the same with the `teacher` and `distill` terms. Both leave out `overhead` (the
CUDA context and allocator slack, which `max_memory_allocated()` does not see). GB is 1e9 bytes.

**Models.** The student in every row is `tiny-qwen2`: a random Qwen2 built from config (hidden
256, intermediate 768, 2 layers, 4 heads, 2 KV heads, tied embeddings, 40,470,784 parameters)
with the Qwen2.5 tokenizer, so its vocabulary is Qwen2.5's 151,936, saved in bf16. A small
student keeps the student's own terms small next to the ones measured here. The teacher is
either the same model (rows that isolate the divergence) or `Qwen/Qwen2.5-1.5B-Instruct` at
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the teacher of the H200 record.

The raw results are in [`evidence/vram-distill-teacher.json`](evidence/vram-distill-teacher.json):
all 19 result lines, every field as the harness printed it, except that local model paths are
replaced by the names above.

## The rows

"KD bytes/element" is (peak - main estimate - teacher weights) / (batch x sequence x 151,936):
what the divergence holds on top of the student's logits and loss.

| # | teacher | kernel | divergence | seq x batch | ckpt | before step | peak | main estimate | main / peak | estimate | est / peak | KD bytes/element |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tiny-qwen2 | dense | forward_kl | 256 x 1 | on | 0.162 | 1.945 | 0.551 | 0.283 | 2.110 | 1.085 | 33.75 |
| 2 | tiny-qwen2 | dense | forward_kl | 512 x 1 | on | 0.162 | 3.739 | 1.020 | 0.273 | 4.057 | 1.085 | 33.90 |
| 3 | tiny-qwen2 | dense | forward_kl | 1024 x 1 | on | 0.162 | 7.315 | 1.960 | 0.268 | 7.953 | 1.087 | 33.90 |
| 4 | tiny-qwen2 | dense | forward_kl | 512 x 2 | on | 0.162 | 7.908 | 1.960 | 0.248 | 7.953 | 1.006 | 37.71 |
| 5 | tiny-qwen2 | dense | js | 512 x 1 | on | 0.162 | 4.663 | 1.020 | 0.219 | 4.991 | 1.070 | 45.79 |
| 6 | tiny-qwen2 | dense | reverse_kl | 512 x 1 | on | 0.162 | 3.739 | 1.020 | 0.273 | 4.057 | 1.085 | 33.90 |
| 7 | tiny-qwen2 | checkpoint | forward_kl | 512 x 1 | on | 0.162 | 2.586 | 1.020 | 0.395 | 2.735 | 1.058 | 19.09 |
| 8 | tiny-qwen2 | checkpoint | forward_kl | 1024 x 1 | on | 0.162 | 4.610 | 1.960 | 0.425 | 5.308 | 1.151 | 16.52 |
| 9 | tiny-qwen2 | checkpoint | forward_kl | 1024 x 2 | on | 0.162 | OOM | 3.838 | | 10.453 | | |
| 10 | Qwen2.5-1.5B | dense | forward_kl | 256 x 1 | on | 3.168 | 4.958 | 0.551 | 0.111 | 5.116 | 1.032 | 33.93 |
| 11 | Qwen2.5-1.5B | dense | forward_kl | 512 x 1 | on | 3.168 | 6.759 | 1.020 | 0.151 | 7.064 | 1.045 | 34.08 |
| 12 | Qwen2.5-1.5B | dense | forward_kl | 512 x 1 | off | 3.168 | 6.767 | 1.025 | 0.151 | 7.069 | 1.045 | 34.12 |
| 13 | Qwen2.5-1.5B | checkpoint | forward_kl | 1024 x 1 | on | 3.168 | 7.645 | 1.960 | 0.256 | 8.314 | 1.088 | 16.70 |
| 14 | tiny-qwen2 | dense | forward_kl | 256 x 2 | on | 0.162 | 4.020 | 1.020 | 0.254 | 4.057 | 1.009 | 37.51 |
| 15 | tiny-qwen2 | dense | js | 256 x 2 | on | 0.162 | 4.932 | 1.020 | 0.207 | 4.991 | 1.012 | 49.24 |
| 16 | tiny-qwen2 | checkpoint | js | 512 x 1 | on | 0.162 | 2.742 | 1.020 | 0.372 | 2.813 | 1.026 | 21.09 |
| 17 | tiny-qwen2 | checkpoint | forward_kl | 512 x 2 | on | 0.162 | 5.222 | 1.960 | 0.375 | 5.308 | 1.016 | 20.45 |
| 18 | tiny-qwen2 | checkpoint | js | 512 x 2 | on | 0.162 | 5.378 | 1.960 | 0.364 | 5.463 | 1.016 | 21.45 |
| 19 | tiny-qwen2 | checkpoint | reverse_kl | 512 x 1 | on | 0.162 | 2.586 | 1.020 | 0.395 | 2.735 | 1.058 | 19.09 |

Row 9 ran out of memory under the 9.0 GiB cap. `main` priced it at 3.8 GB, so its pre-flight
would have let it through on a card of that size; the new estimate is 10.5 GB.

The H200 record of #722 ([`distill-chunked-h200-2026-11-10.md`](distill-chunked-h200-2026-11-10.md),
Qwen2.5-0.5B-Instruct student, Qwen2.5-1.5B-Instruct teacher, B=2, S=2048, LoRA r8 on q/v,
checkpointing on) is the check on a larger student that was not used to fit anything:

| arm | peak | main estimate | main / peak | estimate | est / peak |
|---|---|---|---|---|---|
| dense (`run-01-dense`) | 33.990 | 8.849 | 0.260 | 35.584 | 1.047 |
| `distill_chunk_size: 128`, `distill_checkpoint: true` (`run-02-chunked`) | 22.978 | 8.849 | 0.385 | 25.004 | 1.088 |

## What the rows show

1. **The teacher is resident at its stored dtype.** Before the first step the Qwen2.5-1.5B
   rows hold 3.006 GB more than the tiny-teacher rows: 1,543,714,304 - 40,470,784 parameters
   at 2 bytes. The estimate charges `teacher_params x 2` (4 for a float32 `config.json`).
2. **The divergence costs a fixed number of bytes per element of the logits.** Dense forward
   and reverse KL hold 33.8 to 34.1 bytes an element with a batch of one and 37.5 to 37.7 with
   a batch of two, at every length measured; with a batch of two the student's shifted logits
   are no longer a contiguous view, so the CE path copies them in fp32 (4 bytes). JS keeps
   the mixture's log-probabilities and both probabilities: 45.8 and 49.2. The checkpointed
   kernel recomputes each chunk in the backward and keeps only its inputs: 16.5 to 21.5,
   highest at the shortest length, where one chunk's transients weigh most. The estimate
   charges the largest of each, rounded up: 38, 38 and 50 dense, 21, 21 and 22 checkpointed.
3. **The student's checkpointing does not move it** (rows 11 and 12): the divergence is outside
   the decoder layers.

## What the rows do not cover

- One student architecture and one real teacher, on one card and one stack. Cross-tokenizer
  `uld_strategy` and MiniLLM (`minillm_enabled`) were not measured and are priced as the plain
  divergence; `distill_mode: sequence` was not measured (its teacher is charged, its divergence
  is not, since it trains on the teacher's text).
- Every position is charged; the divergence covers only trained ones, so a run with long
  prompts and `train_on_responses_only` holds less.
- Because of that, a distillation run counts as outside the calibrated range: the pre-flight
  warns when it does not fit, and refuses only when its floor (the student's and the teacher's
  weights, the CUDA context and the trained parameters' state) is more than the card holds.

## Reproduce

```bash
python benchmarks/harness/vram_estimate_distill.py --rows rows.json --out out.jsonl
```

with each row `{"student": "<dir>", "teacher": "<dir>", "seq": ..., "batch": ..., "kernel":
"dense" | "checkpoint", "chunk": 128, "divergence": ..., "gradient_checkpointing": ...}` as
listed in the evidence file. `SOUP_HARNESS_MEMORY_FRACTION=0.19` applied the cap used here.
