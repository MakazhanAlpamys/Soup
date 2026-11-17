# Gate record: #2909, the distillation teacher and divergence in the VRAM estimate

**Hardware.** NVIDIA RTX PRO 5000 Blackwell (sm_120), 48 GB, driver 580.105.08, Linux. The card
was shared, so every row ran capped with `torch.cuda.set_per_process_memory_fraction`, one row
at a time: rows 1-19 at 0.19 (9.0 GiB) while other users' processes held about 38.7 GB, rows
20-39 at 0.17 (8.0 GiB) while they held about 33 GB. torch 2.11.0+cu130, transformers 5.16.1,
peft 0.20.0, trl 0.29.1, accelerate 1.14.0, Python 3.13. Rows 1-19 ran on Soup main `1ac7b400`,
rows 20-39 on this branch with main `1faef1c8` merged in; `trainer/distill.py` is the same in
both, and this fix does not change it.

**What is measured.** `benchmarks/harness/vram_estimate_distill.py` runs `soup train`'s own
setup (`DistillTrainerWrapper.setup`) and two steps per row in a fresh process, and reads
`torch.cuda.max_memory_allocated()` over the steps, the protocol of
[`gate-2049-resident-vram.md`](gate-2049-resident-vram.md). Rows 20-39 also read it over the
setup, which loads both models and, with `distill_mode: sequence`, runs the teacher's
generation. Token-level rows fill `max_length` (three times over, truncated); `val_split: 0`,
chat template `chatml`, LoRA r8 alpha 16 on `q_proj`/`v_proj`, `adamw_torch`, temperature 2.0.
"before step" is what is allocated before the first step; "main estimate" is
`utils/vram_estimate.py` on `main`, which read only `base`; "estimate" is the same with this
fix's terms. Both leave out `overhead` (the CUDA context and allocator slack, which
`max_memory_allocated()` does not see). GB is 1e9 bytes.

**Models.** The student in every row is `tiny-qwen2`: a random Qwen2 built from config (hidden
256, intermediate 768, 2 layers, 4 heads, 2 KV heads, tied embeddings, 40,470,784 parameters)
with the Qwen2.5 tokenizer, so its vocabulary is Qwen2.5's 151,936, saved in bf16. A small
student keeps the student's own terms small next to the ones measured here. The teacher is
either the same model (rows that isolate the divergence) or `Qwen/Qwen2.5-1.5B-Instruct` at
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the teacher of the H200 record.

The raw results are in [`evidence/vram-distill-teacher.json`](evidence/vram-distill-teacher.json):
all 39 result lines, every field as the harness printed it, except that local model paths are
replaced by the names above and the harness's own estimate by `estimate_on_main`, `main`'s
estimate of the same config. Rows 1-19 have no `chunk` field: their checkpointed rows used
`distill_chunk_size: 128`, the harness's default for them.

## Token-level rows

"chunk" is `distill_chunk_size`. "KD bytes/element" is (peak - main estimate - teacher weights)
/ (batch x sequence x 151,936): what the divergence holds on top of the student's logits and
loss.

| # | teacher | kernel | chunk | divergence | seq x batch | ckpt | before step | peak | main estimate | main / peak | estimate | est / peak | KD bytes/element |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tiny-qwen2 | dense | unset | forward_kl | 256 x 1 | on | 0.162 | 1.945 | 0.551 | 0.283 | 2.110 | 1.085 | 33.75 |
| 2 | tiny-qwen2 | dense | unset | forward_kl | 512 x 1 | on | 0.162 | 3.739 | 1.020 | 0.273 | 4.057 | 1.085 | 33.90 |
| 3 | tiny-qwen2 | dense | unset | forward_kl | 1024 x 1 | on | 0.162 | 7.315 | 1.960 | 0.268 | 7.953 | 1.087 | 33.90 |
| 4 | tiny-qwen2 | dense | unset | forward_kl | 512 x 2 | on | 0.162 | 7.908 | 1.960 | 0.248 | 7.953 | 1.006 | 37.71 |
| 5 | tiny-qwen2 | dense | unset | js | 512 x 1 | on | 0.162 | 4.663 | 1.020 | 0.219 | 4.991 | 1.070 | 45.79 |
| 6 | tiny-qwen2 | dense | unset | reverse_kl | 512 x 1 | on | 0.162 | 3.739 | 1.020 | 0.273 | 4.057 | 1.085 | 33.90 |
| 7 | tiny-qwen2 | checkpoint | 128 | forward_kl | 512 x 1 | on | 0.162 | 2.586 | 1.020 | 0.395 | 2.891 | 1.118 | 19.09 |
| 8 | tiny-qwen2 | checkpoint | 128 | forward_kl | 1024 x 1 | on | 0.162 | 4.610 | 1.960 | 0.425 | 5.230 | 1.134 | 16.52 |
| 9 | tiny-qwen2 | checkpoint | 128 | forward_kl | 1024 x 2 | on | 0.162 | OOM | 3.838 | | 9.909 | | |
| 10 | Qwen2.5-1.5B | dense | unset | forward_kl | 256 x 1 | on | 3.168 | 4.958 | 0.551 | 0.111 | 5.116 | 1.032 | 33.93 |
| 11 | Qwen2.5-1.5B | dense | unset | forward_kl | 512 x 1 | on | 3.168 | 6.759 | 1.020 | 0.151 | 7.064 | 1.045 | 34.08 |
| 12 | Qwen2.5-1.5B | dense | unset | forward_kl | 512 x 1 | off | 3.168 | 6.767 | 1.025 | 0.151 | 7.069 | 1.045 | 34.12 |
| 13 | Qwen2.5-1.5B | checkpoint | 128 | forward_kl | 1024 x 1 | on | 3.168 | 7.645 | 1.960 | 0.256 | 8.236 | 1.077 | 16.70 |
| 14 | tiny-qwen2 | dense | unset | forward_kl | 256 x 2 | on | 0.162 | 4.020 | 1.020 | 0.254 | 4.057 | 1.009 | 37.51 |
| 15 | tiny-qwen2 | dense | unset | js | 256 x 2 | on | 0.162 | 4.932 | 1.020 | 0.207 | 4.991 | 1.012 | 49.24 |
| 16 | tiny-qwen2 | checkpoint | 128 | js | 512 x 1 | on | 0.162 | 2.742 | 1.020 | 0.372 | 3.046 | 1.111 | 21.09 |
| 17 | tiny-qwen2 | checkpoint | 128 | forward_kl | 512 x 2 | on | 0.162 | 5.222 | 1.960 | 0.375 | 5.230 | 1.002 | 20.45 |
| 18 | tiny-qwen2 | checkpoint | 128 | js | 512 x 2 | on | 0.162 | 5.378 | 1.960 | 0.364 | 5.386 | 1.001 | 21.45 |
| 19 | tiny-qwen2 | checkpoint | 128 | reverse_kl | 512 x 1 | on | 0.162 | 2.586 | 1.020 | 0.395 | 2.891 | 1.118 | 19.09 |
| 20 | tiny-qwen2 | checkpoint | unset | forward_kl | 256 x 1 | on | 0.162 | 1.945 | 0.551 | 0.283 | 2.110 | 1.085 | 33.75 |
| 21 | tiny-qwen2 | checkpoint | unset | forward_kl | 512 x 1 | on | 0.162 | 3.739 | 1.020 | 0.273 | 4.057 | 1.085 | 33.90 |
| 22 | tiny-qwen2 | checkpoint | unset | forward_kl | 768 x 1 | on | 0.162 | 5.527 | 1.490 | 0.270 | 6.005 | 1.086 | 33.90 |
| 23 | tiny-qwen2 | checkpoint | unset | forward_kl | 256 x 2 | on | 0.162 | 4.020 | 1.020 | 0.254 | 4.057 | 1.009 | 37.51 |
| 24 | tiny-qwen2 | checkpoint | unset | js | 512 x 1 | on | 0.162 | 4.355 | 1.020 | 0.234 | 4.680 | 1.075 | 41.83 |
| 25 | tiny-qwen2 | checkpoint | unset | reverse_kl | 512 x 1 | on | 0.162 | 3.739 | 1.020 | 0.273 | 4.057 | 1.085 | 33.90 |
| 26 | tiny-qwen2 | checkpoint | unset | js | 256 x 2 | on | 0.162 | 4.628 | 1.020 | 0.220 | 4.680 | 1.011 | 45.33 |
| 27 | tiny-qwen2 | checkpoint | 512 | forward_kl | 1024 x 1 | on | 0.162 | 5.777 | 1.960 | 0.339 | 6.397 | 1.107 | 24.02 |
| 28 | tiny-qwen2 | checkpoint | 512 | js | 1024 x 1 | on | 0.162 | 6.399 | 1.960 | 0.306 | 7.019 | 1.097 | 28.02 |
| 29 | tiny-qwen2 | checkpoint | 32 | forward_kl | 512 x 1 | on | 0.162 | 2.294 | 1.020 | 0.445 | 2.599 | 1.133 | 15.34 |
| 30 | tiny-qwen2 | dense | 128 | forward_kl | 512 x 1 | on | 0.162 | 3.035 | 1.020 | 0.336 | 3.357 | 1.106 | 24.85 |
| 35 | tiny-qwen2 | dense | 512 | forward_kl | 1024 x 1 | on | 0.162 | 6.383 | 1.960 | 0.307 | 7.019 | 1.100 | 27.91 |
| 36 | tiny-qwen2 | dense | 32 | forward_kl | 512 x 1 | on | 0.162 | 2.862 | 1.020 | 0.357 | 3.182 | 1.112 | 22.63 |
| 37 | tiny-qwen2 | dense | 128 | js | 512 x 1 | on | 0.162 | 3.957 | 1.020 | 0.258 | 4.291 | 1.084 | 36.71 |
| 38 | tiny-qwen2 | dense | 128 | forward_kl | 256 x 2 | on | 0.162 | 3.320 | 1.020 | 0.307 | 3.357 | 1.011 | 28.52 |
| 39 | tiny-qwen2 | dense | 128 | js | 256 x 2 | on | 0.162 | 4.234 | 1.020 | 0.241 | 4.291 | 1.013 | 40.27 |

Row 9 ran out of memory under the 9.0 GiB (9.64 GB) cap. `main` priced it at 3.8 GB, so its
pre-flight would have let it through on a card of that size; the new estimate is 9.9 GB.

The H200 record of #722 ([`distill-chunked-h200-2026-11-10.md`](distill-chunked-h200-2026-11-10.md),
Qwen2.5-0.5B-Instruct student, Qwen2.5-1.5B-Instruct teacher, B=2, S=2048, LoRA r8 on q/v,
checkpointing on) is the check on a larger student that was not used to fit anything:

| arm | peak | main estimate | main / peak | estimate | est / peak |
|---|---|---|---|---|---|
| dense (`run-01-dense`) | 33.990 | 8.849 | 0.260 | 35.585 | 1.047 |
| `distill_chunk_size: 128`, `distill_checkpoint: true` (`run-02-chunked`) | 22.978 | 8.849 | 0.385 | 23.527 | 1.024 |

## `distill_mode: sequence` rows

The teacher answers "Write a detailed essay of 2,000 words on the history of the Roman Empire."
during setup, with at most `min(max_length, 256)` new tokens, and is freed; the two steps then
train the student on that text. Rows 33 and 34 put 400 words before the question, so prompt
and answer fill 512 tokens. "tokens" is the teacher's prompt plus the new tokens of each
`generate` call. "setup" and "steps" are `max_memory_allocated()` over each; the peak is the
larger. "generation estimate" is the student's weights and adapters plus
`teacher_generation_bytes`: the teacher's weights, its key/value cache for `max_length` prompt
tokens and the new ones, four intermediate-sized MLP tensors over the prompt, and the 9.125 MiB
of cuBLAS and cuBLASLt workspaces the first matmul allocates.

| # | teacher | seq x batch | tokens | setup | steps | generation estimate | gen / setup | main estimate | main / peak | estimate | est / peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | Qwen2.5-1.5B | 256 x 1 | 49 + 256 | 3.189 | 0.566 | 3.211 | 1.007 | 0.551 | 0.173 | 3.211 | 1.007 |
| 32 | tiny-qwen2 | 256 x 1 | 49 + 256 | 0.173 | 0.566 | 0.174 | 1.001 | 0.551 | 0.973 | 0.551 | 0.973 |
| 33 | tiny-qwen2 | 512 x 1 | 449 + 256 | 0.175 | 1.033 | 0.175 | 1.002 | 1.020 | 0.988 | 1.020 | 0.988 |
| 34 | Qwen2.5-1.5B | 512 x 1 | 449 + 256 | 3.221 | 1.033 | 3.237 | 1.005 | 1.020 | 0.317 | 3.237 | 1.005 |

## What the rows show

1. **The teacher is resident at its stored dtype.** Before the first step the Qwen2.5-1.5B
   rows hold 3.006 GB more than the tiny-teacher rows: 1,543,714,304 - 40,470,784 parameters
   at 2 bytes. The estimate charges `teacher_params x 2` (4 for a float32 `config.json`).
2. **In one chunk the divergence costs a fixed number of bytes per element of the logits.**
   Dense forward and reverse KL hold 33.8 to 34.1 bytes an element with a batch of one and
   37.5 to 37.7 with a batch of two, at every length measured. With a batch of two
   `student_logits[:, :-1, :].contiguous()` in the CE path of `trainer/distill.py` is a copy,
   not a view, which is the difference. JS keeps the mixture's log-probabilities and both
   probabilities: 45.8 and 49.2.
3. **`distill_checkpoint` with `distill_chunk_size` unset is one chunk, and saves nothing for
   the KL divergences.** Rows 20-26 (`distill.py` takes the checkpointed path with one chunk
   of every trained position) peak exactly where the dense rows 1, 2, 14 and 6 do, to the
   byte, for forward and reverse KL, and lower for JS (41.8 and 45.3 against 45.8 and 49.2):
   the backward recomputes the whole chunk, so it holds what the dense kernel saved.
4. **With a chunk smaller than the step, the divergence is linear in the chunk size.** Per
   element, KD = kept + (one chunk - kept) x chunk / trained positions, on every chunked row:
   with `distill_checkpoint`, rows 29, 7, 8, 13, 27 and 21 lie within 0.12 bytes an element of
   14.1 + 19.8 x the fraction for forward KL (32 to 512 tokens and one chunk), rows 16, 28 and
   24 on 14.1 + 27.8 x it for JS, and with a batch of two the kept part is 18.0 for both (rows
   17, 18, 23 and 26).
   Without checkpointing each chunk's saved tensors stay for the backward, so more is kept:
   21.9 + 12.0 x the fraction for forward KL (rows 36, 30, 35 and 2), 33.6 for JS, and 25.4
   and 37.2 with a batch of two (rows 38 and 39). The estimate charges vocabulary x (KEPT x
   positions + (ONE_CHUNK - KEPT) x min(chunk, positions)) with the largest of each, rounded
   up: ONE_CHUNK 38, 38 and 50 dense and 38, 38 and 46 checkpointed, KEPT 26, 26 and 38 dense
   and 18 checkpointed (forward KL, reverse KL, JS). It charges every position, where the
   chunks cover the trained ones.
5. **The student's checkpointing does not move it** (rows 11 and 12): the divergence is outside
   the decoder layers.
6. **A sequence-level teacher sets the peak while it writes, not in a step.** The setup of rows
   31 and 34 holds both models, the teacher's cache and its prefill (3.189 and 3.221 GB); their
   steps hold the student alone (0.566 and 1.033 GB): once the teacher is freed
   (`trainer/distill.py`), 0.091 GB stays allocated, the student, its adapters and 9.6 MB of
   cuBLAS workspaces.
   The estimate prices the larger phase: `teacher_generation` is how far writing exceeds
   training, and the teacher is in no step. With the tiny teacher (rows 32 and 33) training is
   larger, and the estimate is the student's own, as on `main`: 0.973 and 0.988 of the peak,
   since the student's terms (#2049) leave out about 13 MB this 40M-parameter student's step
   holds, the workspaces among it.

## What the rows do not cover

- One student architecture and one real teacher, on one card and one stack.
- Cross-tokenizer `uld_strategy` and MiniLLM (`minillm_enabled`, offline or on-policy) were not
  measured. They compute their own loss, which reads none of `distill_divergence`,
  `distill_chunk_size` and `distill_checkpoint` (the schema refuses the two chunk settings with
  them), so they are priced at 50 bytes an element on every position, the largest figure
  measured (dense JS in one chunk), and the estimate's notes say they are not measured. A ULD
  teacher with a larger vocabulary than the student's, or MiniLLM's rollouts, may hold more.
- Every position is charged; the divergence covers only trained ones, so a run with long
  prompts and `train_on_responses_only` holds less.
- A sequence-level teacher's prompt is generated from in full, whatever `max_length` is; the
  estimate prices a prompt of `max_length` tokens.
- Because of all that, a distillation step counts as outside the calibrated range: a run
  whose student fits the card but not with its teacher gets the estimate as a warning and
  starts. The student's own estimate is judged as on `main`, since a teacher only adds to
  it: a run whose student alone does not fit the card is still refused inside the
  calibrated range (`commands/train.py`, with `--allow-oom-attempt` to override), and a run
  whose student's floor (its weights, the CUDA context and the trained parameters' state)
  is more than the card holds is refused either way. The teacher's weights stay out of the
  floor: it is frozen and loaded with `device_map: auto`, which can keep part of it in CPU
  memory.

## Reproduce

```bash
python benchmarks/harness/vram_estimate_distill.py --rows rows.json --out out.jsonl
```

with each row as listed under `run` in the evidence file: `{"student": "<dir>", "teacher":
"<dir>", "seq": ..., "batch": ..., "kernel": "dense" | "checkpoint", "chunk": null | n,
"divergence": ..., "gradient_checkpointing": ...}`, or `{"mode": "sequence", "prompt_words":
n, ...}`. `SOUP_HARNESS_MEMORY_FRACTION` applies the cap.
