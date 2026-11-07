<!--
Measurement record, DRAFT: all three parts are measured; two items are marked <PENDING>
(a tokenizer_config.json diff and the invoice totals).
Published as measured; nothing here is rounded in the direction of a conclusion.

Hardware (part 1): 1x H100 80GB HBM3 (SXM), rented on vast.ai, instance 59547782, 2026-11-07.
  EPYC 9684X (2 sockets, 384 threads), 1,511 GB host RAM, Ubuntu 22.04.5, driver 615.71.09.
  Parts 2 and 3: the same machine, instance 59572480, one job (job_part23.sh, DEVIATIONS.md D-4).
Stack: Python 3.11.14, torch 2.9.1+cu128, transformers 5.22.2, peft 0.21.2, accelerate 1.15.0,
  deepspeed 0.19.9 (cpu_adam built), trl 0.29.1, datasets 5.1.0.
Soup: soup-cli 0.77.2 at 9ca9819747a624da4cbbff03cc8e1a7d2a81100d (not editable).
Harness: benchmarks/harness/ds_compare/ at 3c32b121a5b459be156145c50f584447ddc29100 (re-runnable).
Rule: benchmarks/harness/ds_compare/PROTOCOL.md, committed before the first rental and not edited
  since; sha256 bcb6ad4275aba456c1527ded0bb853f7ba1110a974620beaa171cc06249ab164 (committed LF blob;
  its CRLF working copy hashes to a40c70de..., see section 4).
Raw output: benchmarks/results/ds-compare/part1/ and part23/ (this record's numbers),
  part1-first-rental/ (the first rental, which ended early) and part2-first-attempt/ (a replicate of part 2);
  the last two are informational.
-->

# Gate record: layer streaming against DeepSpeed on one H100 (re-check of STEP 3)

**Status: all three parts MEASURED (2026-11-07, one H100 SXM host; Shape A at 80, 24 and 16 GB, Shapes B
and C at 80 GB).** **Layer streaming is slower than every DeepSpeed arm that keeps the weights on the GPU, at
every setting that was run, and it is faster than DeepSpeed ZeRO-3 with CPU parameter offload by only 1.4x to
1.7x, not by the 2.93x of `gate-h100-validation.md` STEP 3.** G1 = LOWER (S1 61.99 / D1 39.79 tok/s = 1.56x
against the pre-registered threshold of 2.64x); G2 = 0.23 against the best DeepSpeed arm at 80, 24 and 16 GB;
G4 = REVERSED at both realistic lengths (SB / D-best 0.20 at 512 tokens, 0.41 at 2,048). No run was
void and none ran out of memory.

- **Where STEP 3's 2.93x came from.** The streamed number reproduces: S1 61.99 tok/s against
  63.52 in STEP 3 (-2.4%). The DeepSpeed number does not. STEP 3's denominator, 21.65 tok/s with a
  38,135 MiB peak, came from DeepSpeed reached through Soup's own `--deepspeed` flag on a box where
  `cpu_adam` could not be built (so optimizer offload was never measured there; D1 does not use it).
  Configured from the DeepSpeed documentation in a plain Hugging Face
  Trainer + PEFT script with no Soup code, the same ZeRO-3 + CPU parameter offload setting runs at
  39.79 tok/s (1.84x STEP 3's value) with a 4,979 MiB peak, and the variant with optimizer offload and
  a built `cpu_adam` (D2) at 39.01 tok/s. This run did not separate the causes: the Soup path, the
  library versions and the host machine all changed together.
- **Against the best DeepSpeed arm (G2).** SB (the best streaming configuration) over D3 (ZeRO-2 with
  optimizer offload, parameters resident on the GPU) is **0.23** at 80 GB (64.36 / 280.72 tok/s),
  **0.23** under the 24 GB cap (63.88 / 281.41) and **0.23** under the 16 GB cap
  (64.02 / 282.72). The caps changed nothing: every arm, resident R1 and ZeRO-2 D3 included, fit under both
  (an interpretation of why is in section 11). Streaming is faster only than the two DeepSpeed arms that
  move parameters off the GPU: SB / D1 = 1.62, 1.68 and 1.69 at 80, 24 and 16 GB (the rise is a session
  difference in D1, section 8).
- **Realistic lengths (G4): REVERSED at both.** At 512 tokens SB / D3 = 0.20, SB / D1 = 1.42,
  R1 / SB = 7.14; at 2,048 tokens 0.41, 1.56 and 2.75. The 512-token SB value rests on two repeats,
  751.66 and 1,022.78 tok/s, the first being the first run on a cold instance; with the second alone the
  ratios are 0.23, 1.64 and 6.19, the same verdict. A first attempt at part 2 on the same
  host gave 0.22, 1.55, 6.47 and 0.42, 1.57, 2.75.
- **Resident reference (G3).** Resident bf16 LoRA (R1, 406.79 tok/s at Shape A) is **6.32x** SB; with gradient
  checkpointing (R2, 221.96 tok/s) it is 3.45x.
- **Ablation (S1 spread 0.75%).** Pinning the store is the one large effect: without it (S4)
  throughput falls 65.6%. `stream_buffers: 3` adds 2.5%; `stream_recompute` auto or off adds 0.8%
  (the edge of S1's spread, see section 8); `expandable_segments` off costs 1.2% (one slow start
  in three repeats, see section 8); the loss chunk changes nothing.
- **What streaming buys is VRAM, not speed on a card the model fits.** At Shape A, SB peaks at 3,487 MiB
  against 16,529 MiB for resident R1 and 17,515 MiB for ZeRO-2 D3. Against ZeRO-3 with parameter
  offload the gap is much smaller than STEP 3 reported: D1 peaks at 4,979 MiB, 1.70x S1's 2,921 MiB
  (STEP 3: 9.7x) and 1.43x SB's. At the longer rows the low-memory streaming configuration S1 peaks at
  3,861 MiB (512 tokens) and 6,921 MiB (2,048 tokens), against 9,333 and 22,577 MiB for D1 and
  19,885 and 30,061 MiB for R1, at 1.57x and 1.49x D1's throughput.
- **What was not reached.** The pre-registered caps (24 and 16 GiB) were too loose to leave only streaming
  and ZeRO-3 offload in the race, and an explicit 8 or 12 GB cap was not run. That is an interpretation of
  the figures, stated as such in section 11; no run below 16 GiB exists.

Unit convention: peak VRAM is in MiB, as `nvidia-smi` reports it (sampled every 0.5 s) or as
`torch.cuda.max_memory_reserved` divided by 2^20; "GB" in the text is decimal. tok/s is trained
tokens divided by the trainer's `train_runtime` (the STEP 3 metric). A spread is
(max - min) / median over the repeats of one arm.

---

## 1. Why this exists

`gate-h100-validation.md` STEP 3 reported bf16 layer streaming at 2.93x the throughput of DeepSpeed
ZeRO-3 with CPU parameter offload (63.52 against 21.65 tok/s, peak 3,935 against 38,135 MiB), one
H100, Llama-3.1-8B-Instruct, LoRA r=8, batch 1, 256 steps. Three things in that comparison were
open, and the pre-registered protocol (`PROTOCOL.md`, section 0) was written to close them:

1. **Replication on current code.** STEP 3 ran Soup `2c9a078` (2026-08-06); its run scripts and its
   64-row dataset were never committed. This run commits everything it uses.
2. **What each streaming optimization contributes** to the streamed number (the ablation).
3. **DeepSpeed configured from its documentation.** STEP 3 reached DeepSpeed only through Soup's
   `--deepspeed` flag and could not build `cpu_adam` (no `nvcc` on that box), so optimizer offload
   was never measured. Here DeepSpeed and the resident reference run in a plain Hugging Face Trainer +
   PEFT script with no Soup import, with the DeepSpeed JSONs written from the DeepSpeed configuration
   documentation, so the comparison does not depend on Soup's wrapper.

The protocol says that a ratio below 2.93x is stated in the first paragraph of the record. It is.

## 2. The box (part 1)

| item | value |
|---|---|
| rental | vast.ai instance 59547782, on demand, $2.4333/h |
| GPU | H100 80GB HBM3 (SXM), 81,559 MiB, 700 W limit, driver 615.71.09, CUDA UMD 13.4, PCIe link gen 5 x16 at idle |
| SM clock while busy | 1,980 MHz median in every one of the 47 runs (session median 1,980 MHz; 0 valid runs more than 5% below it) |
| CPU | EPYC 9684X, 2 sockets x 96 cores, 384 threads |
| host RAM | 1,511 GB total (1,426 GB available at the start), 7 GB swap |
| `ulimit -l` | **8,192 KB (8 MB)**, recorded and not raised |
| disk | NVMe (the guest reports non-rotational); 120 GB overlay root |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-147-generic, container image `pytorch/pytorch` 2.9.1 / CUDA 12.8 / cuDNN 9 devel |
| toolchain | Python 3.11.14, `nvcc` 12.8.93, g++ 11.4.0 |
| libraries | torch 2.9.1+cu128, transformers 5.22.2, peft 0.21.2, accelerate 1.15.0, deepspeed 0.19.9, trl 0.29.1, datasets 5.1.0 |
| Soup | soup-cli 0.77.2, installed from git at `9ca9819747a624da4cbbff03cc8e1a7d2a81100d` (not editable) |
| harness | `benchmarks/harness/ds_compare/` at `3c32b121a5b459be156145c50f584447ddc29100` (written to `payload_part1.txt` by the payload builder) |
| set-up time | 176 s (`setup_seconds.txt`); job window 09:26:28Z to 11:34:08Z |

**`cpu_adam`.** `setup.sh` built DeepSpeed 0.19.9 from source with `DS_BUILD_CPU_ADAM=1` (the full
build log is kept; the wheel build reports 45.3 s, and the `cpu_adam_op` extension was compiled with
g++), then constructed a `DeepSpeedCPUAdam` object and printed `deepspeed 0.19.9 cpu_adam OK`. Arm D2
(`offload_optimizer: cpu`) ran to completion in all three repeats; that is the configuration that
failed on STEP 3's box with `AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'`.

**Parts 2 and 3** ran on instance 59572480 (rate $2.4333/h; kit launch 13:10:10Z, stopped 14:19:35Z), the
same machine as part 1: the two hardware records differ only in host name, time, free memory and disk fill
(`part23/setup_part2/hardware.txt`), the libraries are the same, soup-cli is the same `9ca9819`, the harness
commit is the same `3c32b121a`, and both set-ups recorded the protocol hash `bcb6ad42...`. They ran back to
back in one job, `job_part23.sh` (DEVIATIONS.md D-4): part 2 (Shapes B and C at 80 GB) from 13:11:56Z to
13:36:56Z, 20 of 20 runs ok; part 3 (Shape A under the 24 and 16 GB caps) from 13:36:56Z to 14:14:27Z, 15 of
15 runs ok. Set-up took 193 s for part 2 and 65 s for part 3. The busy SM clock was 1,980 MHz in 31 of the 35
runs and 1,965 MHz in the four Shape C runs of D3 and R1 (0.8% below 1,980 MHz; the void
limit is 5%); the session median was 1,980 MHz and no valid run was more than 5% below it.

## 3. The model

`NousResearch/Meta-Llama-3.1-8B-Instruct` at revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`
(`d10aef7999a2`), the ungated copy the earlier records used (no token on the rented box). It is
downloaded once per rental into a local directory that every arm uses. Local sha256 of every file
(`setup_part1/model_hashes.txt`, written 2026-11-07T09:28:28Z):

| file | bytes | sha256 |
|---|---|---|
| `model-00001-of-00004.safetensors` | 4,976,698,672 | `2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668` |
| `model-00002-of-00004.safetensors` | 4,999,802,720 | `09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15` |
| `model-00003-of-00004.safetensors` | 4,915,916,176 | `fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa` |
| `model-00004-of-00004.safetensors` | 1,168,138,808 | `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b` |
| `model.safetensors.index.json` | 23,950 | `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b` |
| `config.json` | 855 | `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e` |
| `generation_config.json` | 184 | `189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd` |
| `special_tokens_map.json` | 296 | `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec` |
| `tokenizer.json` | 9,085,657 | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |
| `tokenizer_config.json` | 50,870 | `24e8a6dc2547164b7002e3125f10b415105644fcf02bf9ad8b674c87b1eaaed6` |

**The same weights as `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a1`.** Re-checked for
this record on 2026-11-07 from the Hub's file listings of both revisions: the four `*.safetensors`
shards have identical byte sizes, and the git object ids of their LFS pointer files are identical in
the two repositories (`a59dca288679`, `4ab0726caf5e`, `5864b1555756`, `b8babcaa5a88`). An LFS pointer
holds the file's sha256 and size, so equal pointer ids mean equal sha256. (The unauthenticated listing
masks the sha256 itself for the gated repository, which is why the comparison goes through the pointer
ids; the sha256 values of the NousResearch listing equal the local hashes above.)
`config.json`, `generation_config.json`, `model.safetensors.index.json`, `special_tokens_map.json`
and `tokenizer.json` have the same git object ids in both.

**One file differs.** `tokenizer_config.json` is not the same file in the two revisions: 50,870 bytes
(blob `421cda369d1e`) in the pinned copy against 55,351 bytes (blob `db88166e2bc4`) at the
meta-llama revision. The content was not diffed (the meta-llama file cannot be read without a token).
Every arm trains the same pinned copy and renders its rows with the same template, so the difference
cannot move a ratio between arms; it can move the token count of a row against a run on the meta-llama
copy. Shape A rows are exactly 32 tokens in the pinned copy (checked against its tokenizer by
`make_rows.py --check` during set-up). <PENDING: diff the two `tokenizer_config.json` files once a
token is available, and state what differs.>

## 4. The protocol and its frozen hash

`benchmarks/harness/ds_compare/PROTOCOL.md` was committed (`65afbf1d5`, 2026-11-06 21:29Z, "protocol
wording fixed before any run") before the first rental of 2026-11-07, and is unchanged since. The first
protocol commit (`a016cb04a`, sha256 `4f2a77b1...`) was superseded a minute later, before any run. The
same text has two sha256 values, depending on which file is hashed; both are recorded, with the file each
was computed from:

| sha256 | computed from |
|---|---|
| **`bcb6ad4275aba456c1527ded0bb853f7ba1110a974620beaa171cc06249ab164`** | the committed blob (`git show HEAD:benchmarks/harness/ds_compare/PROTOCOL.md`, LF line ends); also the shipped copy that the rented box hashed (`setup_part1/protocol.sha256`, made by `git archive`) |
| `a40c70debf72a448b6920033a54954d1e274390f6373d8eeed79ddf2b3542261` | the CRLF working copy of the same file on the Windows machine where the protocol was frozen (`sha256sum` of the checkout, `core.autocrlf=true`) |

Converting the LF text to CRLF reproduces the second value and converting back reproduces the first
(re-computed for this record), so the text is the same and only the line ends differ. The value that
identifies the committed and shipped text is the first.

Nothing in the thresholds moved after data existed. The gates are computed below as section 6 of the
protocol states them: G1 CONFIRMED if S1 / D1 >= 2.64 (2.93 - 10%); G2 and G4 are numbers to quote;
G3 is stated plainly whichever way it falls; the ablation reads a change inside S1's spread as "no
measurable effect"; a gate value within 2% of a threshold is reported as a near miss.

## 5. Deviations from the protocol

Four differences between the frozen text and what ran are listed in `DEVIATIONS.md` in the harness
folder. D-1 to D-3 were found on the first rental of part 1 (section 9), before any value entered a gate;
D-4 concerns how parts 2 and 3 were run. None moves a threshold.

**D-1. The streaming arms set `stream_pin: true` explicitly.** Protocol section 2 has S1 with
`stream_pin` automatic. In the run every streaming arm except S4 carries `training.stream_pin: true` in
its `soup.yaml` (S4 carries `false`; SB inherits `true`). With the key set, a failed page-lock makes the
run refuse instead of falling back to pageable memory, so a forced run cannot turn pageable silently.
*Effect on what ran: none that was measured.* Soup's own banner reads the same with the key unset and
with it set: on the first rental (key unset) `base store 16.06 GB across 32 layers (pinned)` and
`Layer streaming ready: 32 layers, 16.06 GB pinned RAM store (16.11 GB page-locked)`; in this run's S1
(key set) the same two lines; in S4 (`false`) `(pageable)` and `16.06 GB pageable RAM store`. At the
Soup commit that ran, the automatic decision attempts to pin (`pinned_limit_bytes=None` at
`src/soup_cli/trainer/stream_setup.py:1370`, so `decide_pinning` returns `pinned=True`) and falls back,
loudly, only if the allocation fails; no code under `src/` reads `RLIMIT_MEMLOCK` (the name occurs in
one comment). The container's limit was 8 MB on both rentals and did not prevent a 16.11 GB page-locked
store. S1 is therefore "current main with the pin requested explicitly", which, on the evidence of the
first rental, is the same run as the automatic choice.
`DEVIATIONS.md` D-1 states the same account (corrected after the harness commit `3c32b121a` that
shipped to the rented box).

**D-2. Arm S0 is not measured.** Protocol section 2 has S0 = Soup at `2c9a078` in a separate venv. On
the first rental S0's `soup train` failed twice (attempt 1 and its one repeat, about 10 s each, exit
code 1) before training started:

```
RuntimeError: non-layer weights left unmaterialised after tying: ['model.embed_tokens.weight', 'lm_head.weight']
```

The S0 venv held soup-cli `2c9a078883af6e826df433804da3368c70561d5c`, transformers 4.57.6, trl 0.26.2,
peft 0.20.0, accelerate 1.14.0, torch 2.9.1+cu128 (`part1-first-rental/env_s0.txt`). Its banner shows a
13.96 GB layer store against S1's 16.06 GB; the 2.10 GB difference equals the size of the embedding plus
the head (2 x 128,256 x 4,096 bf16 values), which fits the "untied embedding and head streaming" change
the protocol lists among the code-level changes since STEP 3. The failure was not diagnosed.
The part-1 job ran with `SKIP_S0=1`.
*Effect:* G1's secondary value S0 / D1 is not reported, and the sum of the code-level changes since
STEP 3 that have no switch (pinned arena packing, untied embedding and head streaming, head and
embedding prefetch, per-call overhead removals, the allocator hint) is not measured on this box. G1
itself (S1 / D1) and the ablation do not depend on S0. For orientation only, S1 here is 2.4% below
STEP 3's 63.52 tok/s and 0.7% below STEP 4's five-repeat bf16 median of 62.43, on different hardware.

**D-3. The Hugging Face Trainer owns the scheduler in the DeepSpeed arms.** The protocol allows
DeepSpeed's own AdamW / WarmupLR where DeepSpeed must own the optimizer and scheduler. In D1..D4 DeepSpeed
owns the optimizer (the JSON has an `optimizer` block: AdamW, lr 2e-4, weight decay 0) and the Trainer
owns the scheduler (`lr_scheduler_type` constant, no warm-up; the JSON has no `scheduler` block, and
each run's resolved DeepSpeed config and trace arguments say so). `DEVIATIONS.md` gives the reason: the
strict-zip failure of STEP 3 does not occur on this torch, and a DeepSpeed `WarmupLR` with
`warmup_num_steps: 0` is refused by the Trainer's config check, seen on an earlier rental whose log is
not among the pulled files. *Effect:* none on throughput; the first optimizer step of a DeepSpeed arm
runs at lr 2e-4 as in every other arm, instead of lr 0 under `WarmupLR`.

**D-4. Parts 2 and 3 ran in one job on one instance** (`job_part23.sh`, the same machine as part 1), not as two
rentals. A first attempt at part 2 on another rental (instance 59565501, 11:52Z to 12:20Z) finished all 20 runs,
but its files were pulled only at 14:21Z, after part 2 had been run again inside instance 59572480; the rerun
is the record's value and the first attempt is reported as a replicate in section 11 and enters no gate. See
`DEVIATIONS.md`.

## 6. Arms

Every arm is one fresh process. Streaming arms run `soup train --config soup.yaml --yes` on a generated
`soup.yaml`; the DeepSpeed and resident arms run `benchmarks/harness/ds_compare/hf_baseline.py`, a plain
Hugging Face Trainer + PEFT script that imports nothing from `soup_cli`. Common to every arm: bf16 base
weights (no NF4), LoRA r=8, alpha 16, dropout 0.05 on `q_proj` and `v_proj` (each Soup run's saved adapter
is checked against this list; all matched), AdamW lr 2e-4 constant with no warm-up, weight decay 0,
gradient clipping 1.0, batch 1, gradient accumulation 1, seed 42, `max_length` 256, no validation split.
The loss is computed on the assistant turn only (the HF script reproduces Soup's chat-template rendering
and loss mask). Shape A: `ds_compare_rows_a.jsonl`, 64 alpaca rows of exactly 32 tokens, 4 epochs, 256
steps, **8,192 trained tokens per run** (counted by the trainer in the Soup arms and by the collator in
the HF arms; both equal 8,192 in every run).

**Streaming arms (Soup).** The generated `soup.yaml` of S1 is: `task: sft`, alpaca data, `epochs: 4`,
`lr: 0.0002`, `batch_size: 1`, `quantization: none`, `lora: {r: 8, alpha: 16, dropout: 0.05}`,
`optimizer: adamw_torch`, `scheduler: constant`, `warmup_ratio: 0.0`, `weight_decay: 0.0`,
`max_grad_norm: 1.0`, `logging_steps: 1`, `seed: 42`, `stream_layers: true`, `stream_pin: true`. The
defaults for everything else apply: `stream_recompute: on`, `stream_buffers: 2`, no loss chunk,
`fast_lora: false`, and the allocator hint exported by Soup (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
recorded in each run JSON). The exact `soup.yaml` of every run is in `part1/A/80/<arm>/rep<k>.work/`.

| arm | what differs from S1 |
|---|---|
| S1 | nothing (current main, defaults, pin requested explicitly, D-1) |
| S2 | `stream_recompute: auto` (at 80 GB the planner keeps the activations of 32 of 32 layers, as `off` does) |
| S3 | `stream_recompute: 'off'` |
| S4 | `stream_pin: false` (leave-one-out: pinning) |
| S5 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in the child environment (leave-one-out: allocator hint) |
| S6 | `stream_buffers: 3` |
| S7 | `stream_loss_chunk: 128` |
| S8 | `fast_lora: true` and `lora.dropout: 0` (informational: dropout 0 changes the numerics, so it never enters a ratio or SB) |
| SB | S1 + `stream_recompute: auto` + `stream_buffers: 3`, derived by the protocol's rule from the Shape A data after repeat 3: every single change S2..S7 faster than S1 by more than S1's spread (0.75%) is combined; S2 (+0.85%) and S6 (+2.51%) qualified; S3 qualified too (+0.83%) but writes the same key as S2, and the rule kept S2 (`sb_overrides.json`) |
| S0 | not run (D-2) |

**DeepSpeed and resident arms.** Launch: `deepspeed --num_gpus 1 --master_port <port> hf_baseline.py
--model <dir> --rows ds_compare_rows_a.jsonl --max-length 256 --epochs 4 --arm <id> --deepspeed <json>`;
resident arms: `python hf_baseline.py ... --deepspeed none`. The `TrainingArguments(deepspeed=...)` object
is built before `from_pretrained`, so the model is created under `zero.Init`. Common DeepSpeed JSON
values (the JSONs are in the harness folder; "auto" is resolved by the Hugging Face integration and the
resolved value is in each run JSON): `bf16.enabled: true`; `optimizer` AdamW with `lr: 0.0002`,
`betas: auto`, `eps: auto`, `weight_decay: 0.0`; `gradient_clipping: auto` (1.0);
`train_batch_size: auto` (1); `train_micro_batch_size_per_gpu: 1`; `gradient_accumulation_steps: 1`;
`overlap_comm: true`; `contiguous_gradients: true`; `reduce_bucket_size: auto` (resolved 16,777,216);
`pin_memory: true` on every offload.

| arm | DeepSpeed / script setting |
|---|---|
| D1 | ZeRO stage 3; `offload_param: {device: cpu, pin_memory: true}`; `offload_optimizer: {device: none}`; `stage3_prefetch_bucket_size: auto` (15,099,494); `stage3_param_persistence_threshold: auto` (40,960); `stage3_max_live_parameters: 1e9`; `stage3_max_reuse_distance: 1e9`; `sub_group_size: 1e9` (the STEP 3 setting, as a JSON) |
| D2 | D1 + `offload_optimizer: {device: cpu, pin_memory: true}` (`cpu_adam` built) |
| D3 | ZeRO stage 2; `offload_optimizer: {device: cpu, pin_memory: true}`; `allgather_bucket_size: 5e8`; `allgather_partitions: true`; `reduce_scatter: true`; parameters on the GPU |
| D4 | ZeRO stage 3, `offload_param: {device: none}`, `offload_optimizer: {device: none}`, the D1 stage-3 values (single GPU: parameters on the GPU) |
| D1PF, D2PF | D1 / D2 with `stage3_prefetch_bucket_size: 5e8`; run only if the arm's GPU was idle more than 50%; **not run**: D1's mean GPU utilisation was 65.2% and D2's 65.3% (idle 34.8% and 34.7%), so the rule's condition was not met (6 slots skipped, listed in `DONE_part1`) |
| R1 | resident bf16 LoRA, no DeepSpeed, no gradient checkpointing |
| R2 | R1 + gradient checkpointing (`gradient_checkpointing=True`) |

Order of execution: untimed warm-ups of S1 and D1, then Shape A at 80 GB with the arms interleaved by
repeat (reversed on even repeats), then SB x 3, then S8 x 3. SB is derived from the first three repeats,
so its runs are not interleaved with the others.

## 7. Results, Shape A at 80 GB

Three timed repeats per arm, each a fresh process; the value of an arm is the median of its repeats.
Steady-state tok/s is tokens over time from step 33 to the end (per-step timestamps), which removes
warm-up and first-touch costs. "Final loss" is the loss logged at step 256 (one row); "mean loss" is the
mean over the 256 logged steps.

| arm | median tok/s | spread | steady-state tok/s | peak VRAM, nvidia-smi (MiB, median / max) | torch reserved (MiB, median) | final loss (step 256) | mean loss | runs valid |
|---|---|---|---|---|---|---|---|---|
| S1 | **61.99** | 0.75% | 62.27 | 2,921 | 2,066 | 3.349 | 3.963 | 3 of 3 |
| S2 | **62.52** | 2.27% | 62.73 | 3,295 / 3,307 | 2,452 | 3.349 | 3.963 | 3 of 3 |
| S3 | **62.50** | 0.16% | 62.79 | 3,065 | 2,210 | 3.349 | 3.963 | 3 of 3 |
| S4 | **21.30** | 1.85% | 21.42 | 2,891 | 2,066 | 3.349 | 3.963 | 3 of 3 |
| S5 | **61.26** | 11.24% | 62.44 | 2,911 | 2,056 | 3.349 | 3.963 | 3 of 3 |
| S6 | **63.55** | 0.33% | 63.87 | 3,341 | 2,486 | 3.349 | 3.963 | 3 of 3 |
| S7 | **61.99** | 0.67% | 62.29 | 2,881 | 2,026 | 3.349 | 3.963 | 3 of 3 |
| S8 (informational) | **61.99** | 0.20% | 62.36 | 2,921 | 2,066 | 3.428 | 3.960 | 3 of 3 |
| SB | **64.36** | 0.40% | 64.59 | 3,487 / 3,723 | 2,872 | 3.349 | 3.963 | 3 of 3 |
| D1 | **39.79** | 1.89% | 39.92 | 4,979 | 3,628 | 3.904 | 4.015 | 3 of 3 |
| D2 | **39.01** | 0.94% | 39.10 | 4,921 | 3,570 | 3.893 | 4.005 | 3 of 3 |
| D3 | **280.72** | 0.45% | 284.38 | 17,515 | 15,808 | 3.842 | 4.016 | 3 of 3 |
| D4 | **87.96** | 1.75% | 88.40 | 17,741 | 16,566 | 3.904 | 4.015 | 3 of 3 |
| R1 | **406.79** | 0.55% | 414.94 | 16,529 | 15,772 | 3.781 | 4.023 | 3 of 3 |
| R2 | **221.96** | 1.06% | 224.59 | 16,383 | 15,626 | 3.781 | 4.023 | 3 of 3 |

Everything else the run measured, per arm:

| arm | repeats, tok/s (rep 1, 2, 3) | train_runtime (s, median) | whole-process wall (s, median) | mean GPU utilisation | peak host RSS (GiB, median) |
|---|---|---|---|---|---|
| S1 | 61.57, 62.03, 61.99 | 132.1 | 155 | 95.3% | 17.0 |
| S2 | 61.14, 62.56, 62.52 | 131.0 | 156 | 96.4% | 17.2 |
| S3 | 62.52, 62.42, 62.50 | 131.1 | 154 | 96.8% | 17.0 |
| S4 | 21.04, 21.44, 21.30 | 384.6 | 397 | 39.7% | 16.9 |
| S5 | 61.26, 55.23, 62.11 | 133.7 | 156 | 94.8% | 17.1 |
| S6 | 63.55, 63.43, 63.64 | 128.9 | 152 | 95.3% | 17.0 |
| S7 | 61.99, 61.71, 62.12 | 132.2 | 155 | 95.8% | 17.1 |
| S8 (informational) | 62.07, 61.99, 61.94 | 132.2 | 155 | 96.3% | 17.1 |
| SB | 64.49, 64.36, 64.23 | 127.3 | 151 | 96.8% | 17.2 |
| D1 | 39.93, 39.79, 39.18 | 205.9 | 237 | 65.2% | 32.8 |
| D2 | 39.35, 39.01, 38.98 | 210.0 | 243 | 65.3% | 32.4 |
| D3 | 281.59, 280.33, 280.72 | 29.2 | 46 | 32.8% | 5.3 |
| D4 | 87.96, 88.47, 86.93 | 93.1 | 109 | 9.3% | 12.0 |
| R1 | 407.60, 405.38, 406.79 | 20.1 | 30 | 38.2% | 4.4 |
| R2 | 224.10, 221.96, 221.74 | 36.9 | 47 | 28.4% | 3.5 |

**Validity (protocol section 5).** All 47 runs (45 timed, 2 untimed warm-ups) ran at attempt 1 and
exited 0, and none was void: no foreign process on the GPU (the process ids were visible inside the
container and every pre-run check found the GPU idle); the busy SM clock median was 1,980 MHz in every
run; the final loss is identical across the repeats of each arm to every printed digit. One run stands
out without being void: S5 repeat 2 took 148.3 s of `train_runtime` against 133.7 and 131.9 s for its
other two repeats (55.23 tok/s whole-run, 62.60 tok/s steady-state), so the whole-run median of S5 sits on
a slow start. The GPU utilisation column is `nvidia-smi`'s `utilization.gpu` (the share of time a kernel was
running) averaged over the training window; it is not SM occupancy.

**Losses.** The seven non-informational Soup arms (S1..S7) and SB have bit-identical losses at steps
64, 128, 192 and 256 (4.249, 3.480, 3.258, 3.349): pinning, buffers, recompute and the loss chunk change
timing, not numerics. D1 and D4 are identical to each other (ZeRO-3 with and without parameter offload),
as are R1 and R2 (gradient checkpointing). The mean loss over the run is 3.96 in the Soup arms and
4.00 to 4.02 in the HF and DeepSpeed arms, within 1.5%; the single-row final losses differ more
(3.35 to 3.90); one row is a noisy sanity check that every arm trains the same task, not a quality
comparison, and the cause of the spread was not examined.

## 8. Gate verdicts, computed as the protocol states them

**G1, replication of the STEP 3 headline on current code: LOWER.**
S1 / D1 = 61.99 / 39.79 = **1.558**, against the threshold 2.64 (2.93 - 10%). The value is not within 2%
of the threshold, so the near-miss rule does not apply. S0 / D1 is not reported (D-2).

| | STEP 3 (Soup `2c9a078`, 2026-08-06) | this run (Soup `9ca9819`, 2026-11-07) |
|---|---|---|
| bf16 layer streaming | 63.52 tok/s, peak 3,935 MiB | S1: 61.99 tok/s, peak 2,921 MiB |
| ZeRO-3, CPU parameter offload | 21.65 tok/s, peak 38,135 MiB, via Soup's `--deepspeed`, DeepSpeed's WarmupLR | D1: 39.79 tok/s, peak 4,979 MiB, plain Trainer + PEFT script |
| throughput ratio | 2.93x | **1.56x** |
| peak VRAM ratio (DeepSpeed / streaming) | 9.7x | 1.70x (S1) or 1.43x (SB, 3,487 MiB) |

The numerator moved by 2.4%; the denominator moved by 1.84x. This run did not isolate why. Candidates
that changed together: DeepSpeed reached through Soup's wrapper against a standalone script; DeepSpeed
0.19.9, transformers 5.22.2 and torch 2.9.1 against 0.19.4, 4.57.6 and torch 2.13; the DeepSpeed-owned
scheduler against the Trainer-owned one (D-3); and the host machine. STEP 3's own caveat stands: ZeRO-3
is designed to shard across ranks, and at world size 1 it has no partner, so this is the regime it is
least suited to. (Separately, STEP 3 reports no peak below 35,997 MiB for DeepSpeed even after tightening
every memory knob; D1 here peaks at 4,979 MiB with `stage3_max_live_parameters: 1e9`.)

**G2, against the best DeepSpeed arm (no pass mark), at 80 GB: 0.23.**
SB 64.36 tok/s (spread 0.40%) over the fastest DeepSpeed arm, D3, 280.72 tok/s (spread 0.45%).
Peak VRAM pair SB / D3: 3,487 / 17,515 MiB. No prefetch variant ran, so the ratio with the variants
included is the same. The ratio against each DeepSpeed arm:

| DeepSpeed arm | median tok/s | spread | SB / arm | arm / SB | peak VRAM MiB (SB / arm) | torch reserved MiB (SB / arm) |
|---|---|---|---|---|---|---|
| D1 | 39.79 | 1.89% | 1.618 | 0.62 | 3,487 / 4,979 | 2,872 / 3,628 |
| D2 | 39.01 | 0.94% | 1.650 | 0.61 | 3,487 / 4,921 | 2,872 / 3,570 |
| D3 | 280.72 | 0.45% | **0.229** | 4.36 | 3,487 / 17,515 | 2,872 / 15,808 |
| D4 | 87.96 | 1.75% | 0.732 | 1.37 | 3,487 / 17,741 | 2,872 / 16,566 |

**G2 under the VRAM caps and at 80 GB (Shape A; no pass mark).** The caps are in GiB and limit PyTorch's
allocator; every arm ran to completion under both (section 11 says what that implies).

| budget | SB tok/s (repeats, spread) | arms that fit | best DeepSpeed arm that fits | tok/s (repeats, spread) | SB / best | peak VRAM MiB, nvidia-smi (SB / best) |
|---|---|---|---|---|---|---|
| 80 GB (part 1) | 64.36 (3, 0.40%) | all | D3 | 280.72 (3, 0.45%) | **0.229** | 3,487 / 17,515 |
| 24 GB cap | 63.88 (2, 0.17%) | SB, D1, D2, D3, R1: all five | D3 | 281.41 (2, 0.45%) | **0.227** | 3,509 / 17,515 |
| 16 GB cap | 64.02 (1) | SB, D1, D2, D3, R1: all five | D3 | 282.72 (1) | **0.226** | 3,727 / 17,515 |

No prefetch variant ran (part 1), so the ratios with the variants included are the same. SB / D1 at the same
three budgets: 1.618, 1.679, 1.695; D1 itself ran 4.3% and 5.0% slower in the part-3
session than in part 1 with its memory use unchanged (3,628 MiB reserved, far below either cap), while SB moved
by 0.7% and 0.5%; the rise of SB / D1 from 80 GB to the caps is that session difference, not a cap effect.

**G3, the resident reference, stated plainly: resident is faster.**
R1 / SB = 406.79 / 64.36 = **6.32** (peak VRAM 16,529 / 3,487 MiB); R2 / SB = 221.96 / 64.36 = **3.45**
(16,383 / 3,487 MiB). An 8B bf16 model fits an 80 GB card, nothing beats training it resident there, and
this run agrees: R1 is 6.56x S1 and 6.32x SB, at 4.74x SB's peak VRAM. The case for streaming is the
VRAM it needs, not speed on a card the model fits.

**G4, realistic lengths, 80 GB: REVERSED at both shapes** (SB / D-best below 1.0). D-best is D3. At 512
tokens SB / D3 = 0.196, SB / D1 = 1.420, R1 / SB = 7.140; at 2,048 tokens 0.406, 1.563 and 2.746; peak VRAM pairs and
the detail are in section 11. The Shape A headline (G1) is quoted here only next to these values.

**Ablation (protocol section 6).** Change of the median against S1 (61.99 tok/s, spread 0.75%):

| arm | change from S1 | median tok/s | change vs S1 | arm's own spread | steady-state change | peak VRAM MiB (change vs S1) | protocol reading (S1 spread 0.75%) |
|---|---|---|---|---|---|---|---|
| S2 | `stream_recompute: auto` | 62.52 | +0.85% | 2.27% | +0.73% | 3,295 (+374) | faster |
| S3 | `stream_recompute: off` | 62.50 | +0.83% | 0.16% | +0.84% | 3,065 (+144) | faster |
| S4 | `stream_pin: false` | 21.30 | -65.64% | 1.85% | -65.60% | 2,891 (-30) | slower |
| S5 | `expandable_segments:False` | 61.26 | -1.17% | 11.24% | +0.26% | 2,911 (-10) | slower |
| S6 | `stream_buffers: 3` | 63.55 | +2.51% | 0.33% | +2.57% | 3,341 (+420) | faster |
| S7 | `stream_loss_chunk: 128` | 61.99 | -0.01% | 0.67% | +0.02% | 2,881 (-40) | no measurable effect |
| S8 | `fast_lora: true`, `lora.dropout: 0` | 61.99 | -0.01% | 0.20% | +0.15% | 2,921 (+0) | no measurable effect [informational] |
| SB | S2 + S6 | 64.36 | +3.82% | 0.40% | +3.72% | 3,487 (+566) | derived arm |

- **Pinning is the one large effect.** Without it throughput falls to 21.30 tok/s, 2.91x slower, and the
  mean GPU utilisation to 39.7%. Peak VRAM is unchanged within 30 MiB.
- **`stream_buffers: 3`** is the one clear small gain (+2.51%, outside both arms' spreads) for 420 MiB.
- **`stream_recompute` auto and off** read "faster" by the protocol's rule, at the edge: +0.85% and +0.83%
  against an S1 spread of 0.75%, and S2's own spread (2.27%) is larger than its gain. At 80 GB the auto
  planner keeps the activations of all 32 layers, as `off` does, and the two arms' throughput is the same
  within noise (62.52 and 62.50; the 0.02-point difference between their gains decided which of the two
  the rule kept for SB). Their peak VRAM differs by 230 MiB (3,295 against 3,065), which this run did not
  explain. Read the throughput effect as "at most 1%".
- **`expandable_segments` off** reads "slower" by the rule (-1.17%), but S5's median sits on one slow start
  (repeat 2); its steady-state tok/s is +0.26% against S1. Read it as "no effect in the steady state".
- **`stream_loss_chunk: 128`** changes nothing (-0.005%), and so does `fast_lora` with dropout 0 (S8,
  informational).
- **SB** is +3.82% over S1 for +566 MiB of peak VRAM (3,487 against 2,921 MiB at the median; SB's first
  repeat peaked at 3,723 MiB).

## 9. Informational: the first rental of part 1 (no gate input)

The first rental of part 1 (instance 59539897, $2.2333/h) ended during S4 repeat 1, after three
Soup arms had finished one repeat each. It is a different physical machine from the one above, and
nothing from it enters a gate, a ratio or the SB derivation.

| | first rental (ended early) | this run (instance 59547782) |
|---|---|---|
| CPU | EPYC 7402P, 1 socket x 24 cores, 48 threads | EPYC 9684X, 2 sockets x 96 cores, 384 threads |
| host RAM | 157 GB | 1,511 GB |
| driver, PCIe link at idle | 570.207, gen 4 x16 | 615.71.09, gen 5 x16 |
| `ulimit -l` | 8,192 KB | 8,192 KB |
| SM clock while busy | 1,980 MHz | 1,980 MHz |
| soup-cli, torch, transformers, peft, accelerate, deepspeed | `9ca9819`, 2.9.1+cu128, 5.22.2, 0.21.2, 1.15.0, 0.19.9 | the same |
| S1, repeat 1 | **27.29 tok/s**, 333 s wall, peak 2,922 MiB | 61.57 tok/s (median of 3: 61.99), 156 s wall, 2,921 MiB |
| S2, repeat 1 | **27.39 tok/s**, 333 s wall, peak 3,304 MiB | 61.14 tok/s, 157 s, 3,307 MiB |
| S3, repeat 1 | **27.32 tok/s**, 333 s wall, peak 3,066 MiB | 62.52 tok/s, 154 s, 3,065 MiB |
| untimed warm-up S1 / D1 | 27.18 / 19.80 tok/s | 59.88 / 39.56 tok/s |

What the table says, and what it does not.

- **The slowdown is not a pinning effect.** In all three first-rental Soup arms the store was reported
  pinned, with the key unset: `store_pinned: true` in each run JSON, and for S1 the two banner lines
  quoted in D-1 (16.11 GB page-locked). The same S1 run on the other machine is about 2.3x faster.
- **DeepSpeed slowed down by about the same factor.** The untimed D1 warm-up ran 19.80 tok/s on the first
  rental against 39.56 here (0.50x), and S1's warm-up 27.18 against 59.88 (0.45x). Soup and DeepSpeed
  both lost about half their throughput on that machine.
- **The cause is not isolated.** The CPU, host memory, PCIe generation at idle and driver all differ; the
  rental ended before any one of them could be tested. The first rental's S1 / D1 pair of untimed
  warm-ups gives 1.37x (one run each), which is not a gate value.
- **What follows from it** is in section 10.

## 10. Host variance

Absolute throughput moved about 2.3x between two H100 SXM hosts with the same container image, the same
Soup commit, the same libraries and the same SM clock: S1 ran at 27.29 tok/s on the first host and 61.57
tok/s on the second at repeat 1 (the second host's median is 61.99, 2.27x), and DeepSpeed moved by about the
same factor (D1 warm-up 19.80 against 39.56 tok/s). Section 9 lists the two hosts. The pin state did not
differ: Soup's banner said pinned on both, with `ulimit -l` at 8,192 KB on both. Which property of the host
moved the numbers was not isolated.

So this record compares ratios between arms measured on one box in one session (interleaved by repeat inside
a part), and quotes an absolute tok/s only with its host named (part 1: the second host, EPYC 9684X, PCIe
gen 5 x16, instance 59547782; parts 2 and 3 name theirs). A ratio built from arms of two different hosts is
not reported; the comparisons with STEP 3's absolute numbers (sections 5 and 8) are across hosts and are
orientation only. The same machine on different rentals also moves: D1 ran 4.3% and 5.0% slower in the
part-3 session than in part 1 (section 8) at identical memory use, and SB at Shape B ran 751.66 tok/s on its
first run of part 2 against 1,022.78 on its second (section 11).

## 11. Parts 2 and 3: Shapes B and C, and the VRAM caps

Part 2 (Shapes B and C) and part 3 (the VRAM caps) ran in one job on the part-1 machine (D-4). Shape B is 64 rows of 512 tokens, 2 epochs, 128 steps, 65,536 trained tokens per run; Shape C is 64 rows
of 2,048 tokens, 1 epoch, 64 steps, 131,072 trained tokens per run. Arms SB, S1, D3 (D-best, the fastest
DeepSpeed arm of Shape A at 80 GB, `d_best.json`), D1 and R1; 2 repeats each. The caps run Shape A under a hard
cap from `torch.cuda.set_per_process_memory_fraction`, the cap in GiB (so 16 means 17,179,869,184 bytes);
arms SB, D1, D2, D3 and R1; 2 repeats at 24 GB, 1 at 16 GB. All 35 runs ran at attempt 1, exited 0, were not
void, saw no foreign process on the GPU, trained the expected token count, and (Soup arms) saved an adapter
with the expected targets; the cap line was present in every capped run's log. No arm ran out of memory, so no
Shape C streaming arm needed the `stream_recompute: on` rerun.

**G4 (protocol section 6). Read: HOLDS if SB / D-best >= 2.0, SMALLER if 1.0 <= SB / D-best < 2.0, REVERSED if
below 1.0. Both shapes read REVERSED.**

| shape | SB tok/s | D-best (D3) tok/s | D1 tok/s | R1 tok/s | SB / D-best | SB / D1 | R1 / SB | peak VRAM MiB, nvidia-smi (SB / D-best / D1 / R1) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B (512 tokens) | 887.22 | 4,529.65 | 624.70 | 6,334.31 | **0.20** | **1.42** | **7.14** | 6,693 / 20,807 / 9,333 / 19,885 | REVERSED |
| C (2,048 tokens) | 3,908.29 | 9,621.51 | 2,500.76 | 10,730.24 | **0.41** | **1.56** | **2.75** | 17,083 / 30,969 / 22,577 / 30,061 | REVERSED |

Every arm at the two shapes (two repeats each, the value is the median of the two, so the mean):

| arm | median tok/s | repeats, tok/s (rep 1, 2) | spread | steady-state tok/s | peak VRAM, nvidia-smi (MiB) | torch reserved (MiB) | mean loss | mean GPU utilisation | runs valid |
|---|---|---|---|---|---|---|---|---|---|
| **Shape B, 512 tokens** | | | | | | | | | |
| SB | **887.22** | 751.66, 1,022.78 | 30.56% | 949.26 | 6,693 | 5,838 | 5.376 | 84.5% | 2 of 2 |
| S1 | **978.03** | 979.15, 976.91 | 0.23% | 988.56 | 3,861 | 3,006 | 5.376 | 95.8% | 2 of 2 |
| D3 | **4,529.65** | 4,503.76, 4,555.54 | 1.14% | 4,639.73 | 20,807 | 19,100 | 5.366 | 50.8% | 2 of 2 |
| D1 | **624.70** | 614.75, 634.65 | 3.19% | 628.86 | 9,333 | 7,982 | 5.365 | 66.5% | 2 of 2 |
| R1 | **6,334.31** | 6,334.49, 6,334.12 | 0.01% | 6,588.03 | 19,885 | 19,128 | 5.365 | 60.8% | 2 of 2 |
| **Shape C, 2,048 tokens** | | | | | | | | | |
| SB | **3,908.29** | 3,916.23, 3,900.35 | 0.41% | 3,957.60 | 17,083 | 16,226 | 5.394 | 95.4% | 2 of 2 |
| S1 | **3,732.26** | 3,740.37, 3,724.14 | 0.43% | 3,805.62 | 6,921 | 6,066 | 5.394 | 95.2% | 2 of 2 |
| D3 | **9,621.51** | 9,606.36, 9,636.66 | 0.31% | 9,780.12 | 30,969 | 29,262 | 5.393 | 78.2% | 2 of 2 |
| D1 | **2,500.76** | 2,477.22, 2,524.30 | 1.88% | 2,493.44 | 22,577 | 21,226 | 5.392 | 75.5% | 2 of 2 |
| R1 | **10,730.24** | 10,728.57, 10,731.91 | 0.03% | 10,834.92 | 30,061 | 29,304 | 5.393 | 88.4% | 2 of 2 |

- **Cold start in Shape B's SB.** SB's first repeat at Shape B (751.66 tok/s, 87.2 s of `train_runtime`, mean GPU utilisation
  71%) was the first run of part 2 on a freshly set-up instance (the README notes that parts 2 and 3 have no untimed
  warm-up); its second repeat ran 1,022.78 tok/s in 64.1 s at 98% utilisation. The protocol's value is the median of the two,
  887.22, with a spread of 30.56%; the run was not void (clock, loss and process checks pass). Using the second repeat
  alone would give SB / D3 = 0.23, SB / D1 = 1.64, R1 / SB = 6.19, which is the same verdict at both
  shapes; it is shown as a sensitivity, not as a replacement. At the median SB is below S1 at Shape B (887.22 against
  978.03); on the second repeat it is +4.6% above it.
- **Streaming against DeepSpeed with parameter offload.** S1 / D1 = 1.57 at 512 tokens and 1.49 at 2,048; SB / D1 =
  1.42 (second repeat alone 1.64) and 1.56. Against the resident arms the gap narrows only at 2,048 tokens: R1 / SB is
  2.75 there, 7.14 at 512 tokens and 6.32 at 32 tokens (Shape A).
- **SB keeps activations when memory is free; S1 recomputes.** At Shape C, SB is 4.7% faster than S1 (3,908.29 against 3,732.26) and peaks at
  17,083 MiB against 6,921 MiB; at Shape B SB peaks at 6,693 MiB against 3,861 MiB. S1 is the
  low-memory streaming configuration at these lengths: 6,921 MiB against 22,577 MiB for D1 and 30,061 MiB for R1 at 2,048
  tokens, and 3,861 against 9,333 and 19,885 MiB at 512 tokens.
- **Losses.** Mean loss over the run is 5.376 for SB and 5.365 for R1 at Shape B, and
  5.394 and 5.393 at Shape C: every arm trains the same task.

**A replicate of part 2.** A first attempt at part 2 (instance 59565501, the same machine, 11:52Z to 12:20Z; D-4)
finished all 20 runs. Its files arrived after the rerun above had finished, so the rerun is the record's value and
this is a replicate; it enters no gate. Its own gate lines (`part2-first-attempt/gates.txt`) read the same
verdicts: Shape B SB / D-best 0.22, SB / D1 1.55, R1 / SB 6.47, REVERSED; Shape C 0.42, 1.57, 2.75, REVERSED.

| shape | arm | rerun median tok/s (record) | first attempt median tok/s | difference | rerun repeats | first-attempt repeats |
|---|---|---|---|---|---|---|
| B | SB | 887.22 | 976.16 | +10.0% | 751.66, 1,022.78 | 935.11, 1,017.21 |
| B | S1 | 978.03 | 979.55 | +0.2% | 979.15, 976.91 | 979.12, 979.98 |
| B | D3 | 4,529.65 | 4,505.62 | -0.5% | 4,503.76, 4,555.54 | 4,520.29, 4,490.95 |
| B | D1 | 624.70 | 630.86 | +1.0% | 614.75, 634.65 | 630.40, 631.32 |
| B | R1 | 6,334.31 | 6,317.02 | -0.3% | 6,334.49, 6,334.12 | 6,343.32, 6,290.71 |
| C | SB | 3,908.29 | 3,897.31 | -0.3% | 3,916.23, 3,900.35 | 3,892.22, 3,902.40 |
| C | S1 | 3,732.26 | 3,738.65 | +0.2% | 3,740.37, 3,724.14 | 3,737.78, 3,739.52 |
| C | D3 | 9,621.51 | 9,378.36 | -2.5% | 9,606.36, 9,636.66 | 9,376.22, 9,380.51 |
| C | D1 | 2,500.76 | 2,480.11 | -0.8% | 2,477.22, 2,524.30 | 2,485.42, 2,474.79 |
| C | R1 | 10,730.24 | 10,727.25 | -0.0% | 10,728.57, 10,731.91 | 10,731.47, 10,723.04 |

The first attempt's SB at Shape B was also slower on its first repeat (935.11 against 1,017.21), as the
first run of that session. Its Shape C D1 peaked at 22,353 MiB against 22,577 MiB in the rerun; no other arm's peak differs.

**The caps, Shape A (part 3).** Cap sizes: 24 GB = 24,576 MiB, 16 GB = 16,384 MiB of allocator memory. Every
run below ran to completion; the reserved column is the largest `torch.cuda.max_memory_reserved` of its
repeats.

| cap | arm | median tok/s | repeats, tok/s | spread | torch reserved (MiB, max) | nvidia-smi peak (MiB, max) | outcome |
|---|---|---|---|---|---|---|---|
| 24 GB | SB | **63.88** | 63.83, 63.94 | 0.17% | 2,872 | 3,531 | ran, exit 0 |
| 24 GB | D1 | **38.06** | 38.28, 37.83 | 1.19% | 3,628 | 4,979 | ran, exit 0 |
| 24 GB | D2 | **37.41** | 37.38, 37.45 | 0.18% | 3,570 | 4,921 | ran, exit 0 |
| 24 GB | D3 | **281.41** | 282.04, 280.78 | 0.45% | 15,808 | 17,515 | ran, exit 0 |
| 24 GB | R1 | **407.09** | 407.07, 407.10 | 0.01% | 15,772 | 16,529 | ran, exit 0 |
| 16 GB | SB | **64.02** | 64.02 | n/a (1 run) | 2,872 | 3,727 | ran, exit 0 |
| 16 GB | D1 | **37.78** | 37.78 | n/a (1 run) | 3,628 | 4,979 | ran, exit 0 |
| 16 GB | D2 | **37.48** | 37.48 | n/a (1 run) | 3,538 | 4,889 | ran, exit 0 |
| 16 GB | D3 | **282.72** | 282.72 | n/a (1 run) | 15,808 | 17,515 | ran, exit 0 |
| 16 GB | R1 | **409.78** | 409.78 | n/a (1 run) | 15,772 | 16,529 | ran, exit 0 |

**Interpretation (not a measurement).** The caps are in GiB, so the "16 GB" cap is 17,179,869,184 bytes of
allocator memory. At Shape A the resident R1 reserves 16,538,140,672 bytes (15,772 MiB) and D3 reserves
16,575,889,408 bytes (15,808 MiB), 576 and 612 MiB below the 16 GiB cap, so both fit under
it and under the 24 GiB cap, and no arm ran out of memory. Their `nvidia-smi` peaks, which include the CUDA context
that sits outside the cap, were 16,529 and 17,515 MiB, above the 16,384 MiB of a 16 GiB card; on a physical 16 GB card they would
be at or past the edge. So the pre-registered caps were not tight enough to reach the regime in which only streaming
and ZeRO-3 with parameter offload fit; the bf16 weights alone are 16.06 GB (14.96 GiB), so that regime begins below
about 16 GB for this model. In that regime the two arms left are SB and D1 (D2 behaves like D1). Where both were
measured and both used under 5,000 MiB (Shape A at 80, 24 and 16 GB; D1 4,979 MiB, SB 3,487 to 3,727 MiB), the ratio SB / D1
was 1.62, 1.68 and 1.69; at 512 and 2,048 tokens (80 GB, where neither arm is under 5,000 MiB) it was 1.42 and 1.56. The part-3 D1 was
about 4 to 5% slower than part 1's (section 8), so the spread of those ratios is a session effect. An explicit 8 or
12 GB cap was not run, so there is no measurement below the caps that were used; the figures above are what is
known about the arms that fit there.

## 12. Cost so far

Protocol section 8: budget USD 30 and at most 8 hours of H100 time for the series, through the self-stopping
rental kit only. The figures are the kit's own estimate, `est.cost` = hourly rate x (time the instance was seen
stopped, or destroyed if it never stopped, minus launch time), computed from the kit's ledger for every
`ds-compare` instance (`vk.py status` lists only live instances, and all of these are destroyed). They exclude
bandwidth charges and the storage charge of an instance while it sat stopped; the last column estimates the
storage from the ledger's storage rate. The invoice replaces them.

| instance | what | date (UTC) | rate | launch to stopped | hours | est.cost | storage while stopped (est.) |
|---|---|---|---|---|---|---|---|
| 59473650 | first test rental, destroyed before the job started | 2026-11-06 | $2.2333/h | 21:41:57Z to 21:43:37Z | 0.03 | $0.06 | $0.00 |
| 59473879 | canary; set-up failed (exit 2) | 2026-11-06 | $2.2333/h | 21:44:30Z to 21:48:30Z | 0.07 | $0.15 | $0.00 |
| 59519012 | canary | 2026-11-07 | $2.3000/h | 04:39:07Z to 04:56:27Z | 0.29 | $0.66 | $0.09 |
| 59539897 | first part-1 rental, ended during S4 repeat 1 (section 9) | 2026-11-07 | $2.2333/h | 07:51:35Z to 08:41:09Z | 0.83 | $1.85 | $0.00 |
| 59547782 | part 1 (the numbers of sections 7 and 8) | 2026-11-07 | $2.4333/h | 09:23:02Z to 11:43:25Z | 2.34 | $5.69 | $0.02 |
| 59565501 | first attempt at part 2 (a replicate, section 11) | 2026-11-07 | $2.4333/h | 11:52:35Z to 12:20:08Z | 0.46 | $1.12 | $0.34 |
| 59572480 | parts 2 and 3 (one job) | 2026-11-07 | $2.4333/h | 13:10:10Z to 14:19:35Z | 1.16 | $2.82 | $0.01 |
| **total** | | | | | **5.17 h** | **$12.35** | **$0.45** |

Total about $12.35 for 5.17 hours of instance time, or about $12.80 with the stopped-instance storage, against
the budget of USD 30 and 8 hours. <PENDING: invoice totals>

## 13. Files and how to repeat

Protocol and harness (committed before the run; the harness commit that ran is `3c32b121a`):
`benchmarks/harness/ds_compare/` holds `PROTOCOL.md`, `DEVIATIONS.md`, `README.md`, `setup.sh`,
`run_all.sh`, `orchestrate.py`, `soup_arm.py`, `hf_baseline.py`, `bench_trace.py`, `sampler.py`,
`sitecustomize.py`, `gates.py`, `make_rows.py`, the three row files and `ds_compare_rows_manifest.json`,
the six DeepSpeed JSONs and the job scripts. Their sha256 as recorded on the box (`setup_part1/env.txt`):

| file | sha256 |
|---|---|
| `ds_compare_rows_a.jsonl` | `f886b68766234799224135f2aae74f5405bf1e9b3dbc3cfcc33029666c897625` |
| `ds_compare_rows_b.jsonl` | `fe00eb5e51b097c6ffeb72b23f40e39b8a961baa0b7b6514818fefe9eb96117c` |
| `ds_compare_rows_c.jsonl` | `6bd4949e3206f7faedbd4b3c7a91a1f5b66251e79af1eabf44c179b5f0bd1f06` |
| `ds_compare_rows_manifest.json` | `fa77d9c1d1e9ffe731a4ef08911a140afd1d7ddd6ebafb1b024e60a3a208865d` |
| `ds_d1_z3_param_offload.json` | `2fa0a4cb925ec47b4b20452ca5f81da30029c9a807026aeac919f5f3c2258e8b` |
| `ds_d2_z3_param_opt_offload.json` | `224202582668b9b5aaec65b49c8acec14f201eb30054bb5231c8a9c6f6d46d5c` |
| `ds_d3_z2_opt_offload.json` | `2e4de2a64f048d0718aba1d68a2728262a05963139d6450e700392ddbea4fcdb` |
| `ds_d4_z3_no_offload.json` | `7b39aaa133285e34869f18e20294c487f4659bbdc460f5e6120955928a8e1b61` |
| `ds_d1_prefetch5e8.json` | `e2889e4698f70fd952bccae1bab39de98f4168346912210d825c786eada3903d` |
| `ds_d2_prefetch5e8.json` | `bcde046756a9217b4a308f148e0d29ae9a5b4b92ec0042b8f21dc7214643c115` |

Raw results of part 1, in `benchmarks/results/ds-compare/part1/`:

- `A/80/<arm>/rep<k>.json`: the run JSON of every run (status, void flags, argv, versions, tokens, tok/s,
  steady-state tok/s, peaks, sampler summary, loss curve, resolved DeepSpeed config, Soup stream log);
  next to each, `rep<k>.gpu.csv` (`nvidia-smi` every 0.5 s: time, memory, utilisation, SM clock),
  `rep<k>.rss.csv` (host RSS of the process tree at 1 Hz), `rep<k>.raw.trace.json` (per-step times and
  losses), `rep<k>.log` (console), and for Soup arms `rep<k>.work/soup.yaml` and the saved adapter's
  `adapter_config.json`; SB also `rep<k>.overrides.json`.
- `warmup/{S1,D1}/`: the two untimed warm-ups.
- `gates.json`, `gates.txt`, `runs.csv`, `sb_overrides.json`, `session_part1.json`, `progress_part1.json`,
  `DONE_part1`, `run_log.jsonl`, `orchestrate.log`, `job_part1.log`, `payload_part1.txt`.
- `setup_part1/`: `hardware.txt`, `env.txt`, `model_hashes.txt`, `protocol.sha256`, `setup.log`,
  `setup_seconds.txt`, `deepspeed_build.log`.

Raw results of parts 2 and 3, in `benchmarks/results/ds-compare/part23/`: the same layout as part 1 for
`B/80`, `C/80`, `A/24` and `A/16` (run JSON, `gpu.csv`, `rss.csv`, `raw.trace.json`, console log, and for Soup
arms `soup.yaml`, `adapter_config.json` and, for SB, `overrides.json`), plus `gates.json`, `gates.txt`,
`runs.csv` (recomputed over every run present, part 1's included), `d_best.json`, `session_part2.json`,
`session_part3.json`, `progress_part2.json`, `progress_part3.json`, `DONE_part2`, `DONE_part3`, `run_log.jsonl`,
`orchestrate.log`, `job_part2.log`, `job_part3.log`, `job23.txt`, `payload_part2.txt`, `payload_part3.txt`,
`VK_STATUS`, `vk_guard.log` and `setup_part2/`, `setup_part3/` (hardware, environment, model hashes, protocol
hash, set-up log; the DeepSpeed build logs of the two set-ups are left out, the build is the same as part 1's and
`setup.log` carries the `cpu_adam OK` line). The first part-2 attempt's evidence (a replicate, section 11) is in
`benchmarks/results/ds-compare/part2-first-attempt/`: its 20 run JSONs, `gates.txt`, `hardware.txt`,
`job_part2.log`, `VK_STATUS`. The job script of parts 2 and 3 is `benchmarks/harness/ds_compare/job_part23.sh`.

The first rental's evidence (section 9) is in `benchmarks/results/ds-compare/part1-first-rental/`:
`hardware.txt`, `env_s0.txt`, the S0 failure log, the S1, S2 and S3 run JSONs, S1's Soup console log, the
two warm-up JSONs, `job_part1.log`. Left out of every results folder: Soup's `experiments.db` files, each run's
`rep<k>.raw.json` (the child process's own output, merged into `rep<k>.json`), and Soup's console logs
(`rep<k>.raw.log`, mostly progress bars), except S1 and S4 repeat 1, which carry the pin lines quoted in
D-1.
All copied text files have LF line ends, no trailing spaces and one final newline; the originals had
CRLF or lone carriage returns in the CSVs and progress output. No value was changed.

**To repeat the whole series on another H100** (a CUDA devel image with `nvcc`, Python 3.11 to 3.13,
torch >= 2.6 installed, one idle GPU, no Hugging Face token):

```bash
cd benchmarks/harness/ds_compare
SKIP_S0=1 bash setup.sh && bash run_all.sh --all
```

`bash run_all.sh --all --dry-run` prints every command and every `soup.yaml` without running anything;
`bash run_all.sh --dry-run --part part1` prints one part's plan. The harness README ("Running in parts")
describes the split into parts; parts 2 and 3 ran in one job, `job_part23.sh` (D-4).

## 14. Known limits and open items

- **Single GPU.** ZeRO-3 is designed to shard across ranks; at world size 1 it is outside its intended
  regime, which is also streaming's home ground. No multi-GPU claim is made either way.
- **One host per part, and one session per part.** Ratios hold inside a part; section 9 shows how far absolute
  throughput moves between two hosts, and section 10 how far the same machine moves between sessions.
- **Two repeats at Shapes B and C and at the 24 GB cap, one at the 16 GB cap.** Two repeats give a median that is
  the mean of both, and a spread that one slow run can dominate (SB at Shape B, section 11).
- **No explicit 8 or 12 GB cap was run** (section 11): the regime where only streaming and ZeRO-3 with parameter
  offload fit was not measured.
- **Timing scope.** `train_runtime` is the trainer's: for the HF and DeepSpeed arms it includes engine
  start-up before the first step; for Soup arms the stream set-up is outside it. `steady_tok_per_s` removes
  both and the whole-process wall time is recorded. Soup arms log every step (`logging_steps: 1`, needed for
  per-step timing), which costs a database commit and a panel update per step that the plain Trainer
  does not have; both are part of what `soup train` costs and both bias against Soup, slightly.
- **Dataset.** STEP 3's own file was never committed. Shape A is a new dataset of the same shape (64 rows,
  8,192 tokens over 256 steps); the row text differs. The pinned copy's chat template has no system
  header and always appends an assistant header, identically in every arm.
- **Not isolated:** the cause of the 1.84x between STEP 3's DeepSpeed number and D1 (section 8), the cause of
  the first rental's slowdown (section 9), and the S0 failure (D-2).
- **Open before this draft is final:** <PENDING: `tokenizer_config.json` diff (section 3)>; <PENDING: invoice
  totals (section 12)>.
