# ds_compare: layer streaming against an independently configured DeepSpeed, 1x H100

The harness of a pre-registered comparison (the protocol is frozen by its sha256, recorded with
the run). It re-checks, on current `main`, the STEP 3 / STEP 4 result of
`benchmarks/gate-h100-validation.md` (bf16 layer streaming 2.93x the throughput of DeepSpeed
ZeRO-3 with CPU parameter offload, Llama-3.1-8B-Instruct, LoRA r=8, batch 1), adds an ablation
of what each streaming optimization contributes, and runs DeepSpeed the way an independent
engineer would set it up: a plain Hugging Face Trainer + PEFT script with no Soup code in the
path, configured from the DeepSpeed documentation.

## Repeat the whole series with one command

On a rented H100 (a CUDA *devel* image with `nvcc`, Python 3.11 to 3.13, torch >= 2.6 already
installed, one idle GPU, no Hugging Face token needed):

```bash
cd /workspace/ds_compare            # this folder
bash setup.sh && bash run_all.sh --all
```

`setup.sh` takes about an hour (model download, the DeepSpeed build). `run_all.sh --all` takes
about 5.3 h of GPU time without the conditional runs and about 6.5 h with all of them (the
plan is estimated, see below; `bash run_all.sh --all --dry-run` prints it). Everything lands
under `results/`; `results/DONE` is written last.

```bash
bash run_all.sh --all --dry-run          # every command, every soup.yaml, no run
bash run_all.sh --only S1,D1             # restrict to some arms (SB and DBEST are valid ids)
bash run_all.sh --all --phases A,SB      # restrict to some phases (see orchestrate.py)
DEADLINE_EPOCH=$(date -d '+7 hours' +%s) bash run_all.sh --all   # stop rule, see below
```

A finished run is not run again: restarting `--all` after an interruption resumes.

## Files

| file | what it is |
|---|---|
| `setup.sh` | hardware record, soup-cli from a git commit (`SOUP_COMMIT`, default the `main` commit written in the file), DeepSpeed built with `DS_BUILD_CPU_ADAM=1` (log kept) and `DeepSpeedCPUAdam` verified, second venv for S0, the model at its pinned revision with no token, `results/model_hashes.txt`, `results/env.txt` |
| `run_all.sh` | environment preparation, then `orchestrate.py` |
| `orchestrate.py` | the plan (protocol sections 2 to 5), the interleaved order, the SB derivation rule, the D-best choice, the void checks, the budget drop order, `gates.json` at the end |
| `soup_arm.py` | streaming arms S0..S8 and SB: writes the `soup.yaml`, runs `soup train --config soup.yaml --yes`, writes the run json |
| `hf_baseline.py` | plain Hugging Face Trainer + PEFT LoRA: arms D1..D4, the two prefetch variants, R1 and R2. Imports nothing from `soup_cli` |
| `bench_trace.py` | the per-step recorder (a `TrainerCallback`) and the metric derivation, shared by all arms |
| `sitecustomize.py` | on `PYTHONPATH` for every arm alike: the VRAM cap (`SOUP_BENCH_VRAM_CAP_GB`) and, for `soup train`, the passive recorder (`SOUP_BENCH_TRACE_OUT`). Neither imports torch at interpreter start |
| `sampler.py` | `nvidia-smi` every 0.5 s (memory, utilization, SM clock), compute-process list every 2.5 s, host RSS of the process tree at 1 Hz; CSVs next to the run json |
| `gates.py` | G1, G2, G3, G4, the ablation table, spreads, near-miss flags |
| `make_rows.py` | writes the three datasets (seed 42) and `ds_compare_rows_manifest.json` |
| `ds_compare_rows_a.jsonl`, `_b`, `_c` | 64 alpaca rows of exactly 32, 512 and 2,048 tokens |
| `ds_d1_z3_param_offload.json`, `ds_d2_z3_param_opt_offload.json`, `ds_d3_z2_opt_offload.json`, `ds_d4_z3_no_offload.json` | the DeepSpeed arms |
| `ds_d1_prefetch5e8.json`, `ds_d2_prefetch5e8.json` | the two allowed prefetch variants (run only when that arm's GPU was idle more than 50%) |
| `tiny_fixture.py`, `selftest.py` | CPU rehearsal fixtures and the checks of the harness logic |

## Arms

| id | what |
|---|---|
| S0 | Soup at `2c9a078` in a second venv with the earlier stack (transformers 4.57.6, trl 0.26.2, peft 0.20.0, accelerate 1.14.0); S1's yaml with the keys that did not exist then dropped (`training.seed`; the others are not set by any arm); the json lists them |
| S1 | current `main`, defaults: `stream_layers`, `stream_recompute: on`, `stream_buffers: 2`, pinning automatic, `expandable_segments` exported by Soup |
| S2 / S3 | S1 + `stream_recompute: auto` / `off` |
| S4 | S1 + `stream_pin: false` |
| S5 | S1 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in the child environment (Soup keeps a user value, `src/soup_cli/utils/gpu.py:239-253`) |
| S6 / S7 | S1 + `stream_buffers: 3` / `stream_loss_chunk: 128` |
| S8 | S1 + `fast_lora: true` + `lora.dropout: 0` (informational, never in a ratio) |
| SB | S1 plus every single change S2..S7 faster than S1 by more than S1's spread, derived after Shape A repeat 3 (`results/sb_overrides.json`) |
| D1 | ZeRO-3, `offload_param: cpu`, no optimizer offload (the STEP 3 setting) |
| D2 | ZeRO-3, `offload_param: cpu` + `offload_optimizer: cpu` (`cpu_adam`) |
| D3 | ZeRO-2, `offload_optimizer: cpu`, parameters on the GPU |
| D4 | ZeRO-3, no offload |
| D1PF / D2PF | D1 / D2 with `stage3_prefetch_bucket_size: 5e8` |
| R1 / R2 | resident bf16 LoRA / the same with gradient checkpointing |

Shapes: **A** = 64 rows of 32 tokens, 4 epochs, 256 steps, `max_length` 256; **B** = 64 rows of
512 tokens, 2 epochs, 128 steps; **C** = 64 rows of 2,048 tokens, 1 epoch, 64 steps. Every
row of a shape has exactly the same length after the chat template (verified by
`make_rows.py --check`, which `setup.sh` runs against the real tokenizer), so a step trains the
same number of tokens in every arm.

Order of execution: warm-up S1 and D1 (untimed); Shape A at 80 GB, 3 repeats interleaved by
repeat (the arm order is reversed on even repeats); SB x 3; the prefetch variants; Shapes B and
C repeat 1 (SB, S1, D-best, D1, R1); the 24 GB cap, 2 repeats (SB, D1, D2, D3, R1); Shapes B
and C repeat 2; the 16 GB cap, 1 repeat; S8 x 3. "80 GB" means the whole card, no cap.

Void rules (protocol section 5) are applied to every run and a void run is repeated once: another process on the
GPU, busy SM clock more than 5% below the session median, non-zero exit that is not an
out-of-memory under a cap, final loss more than 20% away from the arm's other repeats; also
operational voids: the VRAM cap line missing from the log, no train-end event recorded. An
out-of-memory under a cap is "does not fit", not repeated. A streaming arm that runs out of
memory at Shape C is rerun once with `stream_recompute: on` (json: `rerun_recompute_on`; the
first attempt is kept as `rep<k>.oom.json`).

Budget rule (protocol section 8): before every run, the estimated time of everything still
pending is compared with `DEADLINE_EPOCH` (and with `BUDGET_USD` at `RENTAL_RATE_USD_PER_HOUR`,
alias `VAST_RATE_USD_PER_HOUR`, when both are set); when it does not fit, whole groups are dropped in this order: S8, the
prefetch variants, the 16 GB cap, Shape B repeat 2, Shape C repeat 2, the 24 GB cap. Shape A's
S1 and D1 and repeat 1 of Shapes B and C are never dropped. `results/dropped.json` lists what was dropped.

## Results layout

```
results/
  <shape>/<budget_gb>/<arm>/rep<k>.json         the run (written the moment it ends)
                            rep<k>.gpu.csv      unix_time, memory_used_mib, utilization_gpu_pct, sm_clock_mhz
                            rep<k>.rss.csv      unix_time, rss_bytes, processes
                            rep<k>.log          console output (soup arms: rep<k>.raw.log is soup's own)
                            rep<k>.raw.trace.json   per-step times, losses, tokens
                            rep<k>.void1.json   a void first attempt
  warmup/<arm>/warmup.json
  sb_overrides.json  d_best.json  dropped.json  gates.json  gates.txt  runs.csv  run_log.jsonl
  hardware.txt  model_hashes.txt  env.txt  env_s0.txt  deepspeed_build.log  setup.log  DONE
```

A run json holds: arm id, argv, versions, status (`ok`, `oom`, `failed`), void flags and
reasons, trained tokens and how they were counted, `train_runtime_s`, `tok_per_s` (tokens /
`train_runtime`, the STEP 3 metric), `steady_tok_per_s` (steps 33..end from per-step
timestamps), `max_memory_reserved_bytes` (torch), the `nvidia-smi` peak and the mean utilization and median busy SM clock over the training window, peak host RSS, final loss and the loss at steps
64/128/192/256 (where the run has them), wall seconds, cost, and for Soup arms the yaml, the stream plan (tier,
store size, pinned or pageable), the `stream_recompute: auto` planner line and the adapter's target modules next to the expected ones.

## Software

* Pinned by the harness: the model `NousResearch/Meta-Llama-3.1-8B-Instruct` at revision
  `d10aef7999a2b5ba950ab3974312feeedbfe0b77` (a local copy, used by every arm; its four
  safetensors shards were compared with the published original's LFS sha256 values separately
  from this harness); soup-cli at `SOUP_COMMIT` (default `9ca9819747a624da4cbbff03cc8e1a7d2a81100d`) and, for S0, at `2c9a078`.
* Resolved at setup and written to `results/env.txt`: torch (the image's, >= 2.6), transformers
  (>= 5.16.1, < 6), peft (>= 0.20), trl (>= 0.29), accelerate, datasets, DeepSpeed (latest unless
  `DEEPSPEED_SPEC` pins it; the earlier record used 0.19.4). S0's venv pins the earlier record's stack.
* The CPU rehearsals of this harness ran on torch 2.14.0, transformers 5.17.0, peft 0.20.0, accelerate 1.15.0, trl 0.29.1, Python 3.12.
* Launch of the DeepSpeed arms: `deepspeed --num_gpus 1 --master_port <port> hf_baseline.py ...` (the launcher
  the Hugging Face DeepSpeed guide documents; it appends `--local_rank`, which the script accepts). Resident arms: `python hf_baseline.py`.
  `TrainingArguments(deepspeed=...)` is built before `from_pretrained`, so the model is created under
  `zero.Init`. Optimizer and scheduler are owned by DeepSpeed in the D arms (the json has the `optimizer` and `scheduler` blocks).

## What is the same in every arm

bf16 base weights; LoRA r=8, alpha=16, dropout 0.05 on `q_proj` and `v_proj` (Soup's resolved default for
Llama: `target_modules: auto` returns PEFT's table entry, `src/soup_cli/utils/peft_wiring.py:381-439`; the
adapter written by each Soup run is compared with this list); AdamW, lr 2e-4, weight decay 0, gradient
clipping 1.0, constant learning rate, batch 1, gradient accumulation 1, seed 42, `save_steps` beyond the
run, no validation split. The loss is computed on the assistant turn only: the HF script reproduces Soup's
chat-template rendering and loss mask (`ds_common.render_alpaca_row`, checked against
`soup_cli.data.loss_mask.build_assistant_only_labels` by `selftest.py`).

## Known limits

* STEP 3's own dataset and scripts were never committed. These files are a new dataset of the same shape
  (64 rows, 8,192 tokens over 256 steps = 32 tokens per step, derived from STEP 3's tok/s x runtime); the
  row text differs. The pinned model's chat template has no system header and always appends a
  `<|start_header_id|>assistant<|end_header_id|>` header; the header after the answer is part of the trained span,
  identically in every arm.
* "Linear warmup 0" is run as a constant 2e-4. DeepSpeed's `WarmupLR` raises `warmup_num_steps` to at least 2
  internally, so in D1..D4 the first optimizer step uses lr 0 (log warm-up); this changes the first loss values
  slightly and no throughput.
* `train_runtime` is the trainer's: for the HF and DeepSpeed arms it includes engine start-up that precedes
  the first step; for Soup arms the stream set-up is outside it. `steady_tok_per_s` removes both, and the
  whole-process wall time is recorded.
* Soup arms log every step (`logging_steps: 1`, needed for per-step timing); each step costs a SQLite commit and
  a panel update that the plain Trainer does not have, and TRL's `SFTTrainer` computes extra metrics per step.
  Both are part of what `soup train` costs and both bias against Soup, slightly.
* The VRAM caps limit the PyTorch caching allocator (reserved memory). The CUDA context (a few hundred MB) and
  pinned host memory are outside the cap, so the `nvidia-smi` peak sits that much above it.
* The SB runs happen after Shape A repeat 3 (SB is derived from it), so they are not interleaved with the other arms.
* The "another process on the GPU" check needs process ids that match inside the container; when they do not,
  the run json says `pid_namespace_visible: false` and only the idle-GPU check before each run applies.
* The ZeRO-3 / PEFT / HF Trainer path and the Soup run were not run on a GPU while this harness was written
  (CPU only, no DeepSpeed installed on the development box). The untimed warm-up of D1 is the first real test of
  that path; the orchestrator's run loop was exercised with fake child processes only (`selftest.py`).
* Estimates of the plan (about 5.3 h without the conditional runs, 6.5 h with them) are built from the STEP 3 and STEP 4
  timings for streaming (2.2 min) and D1 (6.5 min) and guesses for the other arms; they are replaced by the
  measured wall time of each arm as runs finish.

## Checks that run without a GPU

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/src python selftest.py     # DS_COMPARE_TOKENIZER=<model dir> adds the real-token checks
python make_rows.py --check --tokenizer <model dir>
bash run_all.sh --all --dry-run
python gates.py <results dir>
```
