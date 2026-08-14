# Changelog

All notable changes to **Soup CLI** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Detailed, per-release notes for every published version live on the
[GitHub Releases page](https://github.com/MakazhanAlpamys/Soup/releases). This
file tracks unreleased changes and links out for historical detail rather than
reproducing 70+ versions of notes.

## [Unreleased]

### Added

- **Semantic Stratified Splitting for `soup data split` command.** Adds `--stratify-semantic` and `--num-clusters` options to automatically partition dataset splits proportionally across semantic groups using TF-IDF + K-Means clustering. Includes validation checks, ValueError safety, a 50,000-row cap, and a warnings module.

### Fixed


- **The MLX `adapter_config.json` shipped `target_modules` unresolved, so a default
  MLX adapter loaded as a silent no-op (#392).** `_apply_lora` resolved
  `target_modules: auto` into a local variable and trained the resolved modules; the
  writer serialised the raw config value, so the shipped file carried
  `{"keys": ["auto"]}`. On load, `linear_to_lora_layers` matches no module against that
  and `load_weights(strict=False)` drops every LoRA tensor without a word — generation
  with the adapter is bit-identical to the base model. `"auto"` is the schema default,
  so this was every MLX run that did not name its modules by hand, and the file exists
  precisely to promise the output dir loads with
  `mlx_lm.load(..., adapter_path=...)`. Both callers now go through one
  `resolve_mlx_target_keys()`, because two copies of "which modules did we train?" is
  how they drifted. Reported with a root cause and a control by
  [@armanbot-jpg](https://github.com/armanbot-jpg): hand-editing `keys` in the saved
  file makes the very same `adapters.safetensors` produce the tuned behaviour.

## [0.73.1] - 2026-08-14

### Added

- **`training.stream_vram_probe` decides the layer-streaming VRAM pre-flight on a
  MEASUREMENT instead of the fitted formula (#349).** The pre-flight predicts peak
  VRAM from a formula fitted to 10 real runs, and its documented contract is that it
  never under-predicts. Measured through the real `soup train` on an RTX 3050 Laptop
  (4 GB, Windows, torch 2.5.1) with SmolLM2-135M streamed in bf16 at batch 1, that
  contract holds at short sequence and then fails:

  | seq | predicted | real peak | ratio |
  |---|---|---|---|
  | 4352 | 3.282 GB | 3.036 GB | 1.081x — over-predicts, safe |
  | 5120 | 3.844 GB | 4.118 GB | **0.934x — under-predicts** |
  | 6144 | 4.590 GB | 5.830 GB | **0.787x — under by 21%** |

  Under-prediction is the direction that does not announce itself: an OOM on Linux,
  and on Windows/WDDM a silent spill to host memory. The existing grid could not
  have caught it — all ten of its rows are at seq 256 or 512, so it varies batch and
  says nothing about sequence length, and `test_never_under_predicts` has been
  narrowed to state that scope rather than imply a global guarantee.

  With the flag on, one real forward+backward runs at the configured shape after the
  streamed model is built and its peak decides; the prediction is printed beside it
  so a divergence is visible. Measured cost: **1.0-5.3 s** (SmolLM2-135M at 1x1024
  and 2x2048; Llama-3.1-8B NF4 at 1x512), against a training run of minutes to hours.
  Off by default — it costs a step, and it can refuse a run the formula accepts.

  Scope is deliberately narrow. **`task: sft` only**: the probe runs a plain causal-LM
  step, which *is* the SFT step but is not a preference loss, so its agreement with one
  is not established. Measured at a single matching shape it is conservative there too
  (6.02 GB against a real DPO step's 5.30 GB, +13.5%) — but one point is not a
  validation, and a sign flip would mean a gate waving through over-budget runs. It also **cannot overrule a
  prediction more than 4x over budget**: the largest disagreement ever measured is 21%,
  so beyond a small multiple the config is simply too big and is refused by arithmetic
  without touching the GPU. The gate reads `max_memory_allocated`, not
  `max_memory_reserved` — reserved runs 1.08-1.41x allocated and overshoots what has to
  fit, and gating on it would refuse this feature's own flagship configuration
  (Llama-3.1-8B NF4, 3.70 GB reserved against 3.45 GB free, which runs).

  Two readings were tried during this work and **withdrawn as unsupported**, recorded
  because the tempting inference was wrong twice in one investigation: that preference
  losses are over-budgeted ~8.8x (it is 1.15x at the budgeted shape — the earlier figure
  came from rows that realised 142 of a budgeted 2048 tokens), and that the over-budget
  runs were silently spilling (`num_alloc_retries` was 0 on every shape measured). The
  mechanism behind the long-sequence divergence is likewise **not claimed**: `seq**2`
  from the attention score matrix is the obvious candidate and the numbers do not settle
  it.

### Fixed

- **The MLX `adapter_config.json` shipped `target_modules` unresolved, so a default
  MLX adapter loaded as a silent no-op (#392).** `_apply_lora` resolved
  `target_modules: auto` into a local variable and trained the resolved modules; the
  writer serialised the raw config value, so the shipped file carried
  `{"keys": ["auto"]}`. On load, `linear_to_lora_layers` matches no module against that
  and `load_weights(strict=False)` drops every LoRA tensor without a word — generation
  with the adapter is bit-identical to the base model. `"auto"` is the schema default,
  so this was every MLX run that did not name its modules by hand, and the file exists
  precisely to promise the output dir loads with
  `mlx_lm.load(..., adapter_path=...)`. Both callers now go through one
  `resolve_mlx_target_keys()`, because two copies of "which modules did we train?" is
  how they drifted. Reported with a root cause and a control by
  [@armanbot-jpg](https://github.com/armanbot-jpg): hand-editing `keys` in the saved
  file makes the very same `adapters.safetensors` produce the tuned behaviour.

- **`training.batch_size` accepted 0 and negative values.** `Union[int, Literal["auto"]]`
  carried no lower bound, so `batch_size: -4` loaded and then meant whatever each
  trainer's arithmetic did with it — including the streaming VRAM pre-flight, which
  multiplies by it. Now rejected at config load. Surfaced by the #349 security review.

### Fixed

- **Layer streaming's VRAM pre-flight now actually calls its own calibration hook (#348).**
  `calibrated_logits_bytes_per_element()` exists to raise the budget when a stack's loss
  path measures a heavier retention than the shipped constant assumes, guarding against a
  future stack silently under-budgeting by 12.5% with nothing to catch it. `_stream_budget_lines`
  called `estimate_stream_peak_vram()` without `logits_bytes_per_element=`, so the parameter
  was always `None` and the calibration never ran outside its own test. It is now forwarded to
  both the budget and the panel's `logits` figure; the value can only raise the prediction
  (floored at the shipped constant), and the panel prints an extra line naming both numbers
  when the calibration measures above it. This makes the probe unconditional rather than
  opt-in (see #327 below, whose wording is updated to match): every streamed run now pays
  one transient `14 * vocab_size * max(tokens)` allocation (96 MiB at the defaults) plus two
  `torch.cuda.synchronize()` calls before the fit decision is taken. On today's measured
  stacks this is a no-op in effect (`measured` is 12.0 with zero spread, `max(14, 14)` is
  14, no extra line prints), so the cost buys nothing yet, which is the point of a guard
  against a stack that hasn't shipped.
- **MLX backend now actually dispatches to the MLX trainer for `task: sft`.** Previously `backend: mlx` silently fell through to the transformers `SFTTrainerWrapper`, training on MPS/CUDA instead of MLX. The trainer was also rewritten for mlx-lm >= 0.31 (`create_dataset` + `CacheDataset`, `TrainingCallback`), with `model.freeze()` before LoRA — without it the saved "adapter" was a full fine-tune (172 tensors vs 24 LoRA tensors on a 1.2B model) — and an `adapter_config.json` is written so the output dir loads directly with `mlx_lm.load(..., adapter_path=dir)`. (#362)
- **`mlx-lm` floor raised to >= 0.31.3** (the version the MLX SFT path is built against).
- **`training.seed` reached the SFT wrapper and nothing else (#353).** #341 added the
  knob and wired it into `trainer/sft.py`. The other seventeen task wrappers each build
  their own `TrainingArguments` subclass (`GRPOConfig`, `DPOConfig`, `RewardConfig`, and
  so on) and none of them read the field, so `task: grpo` with `training.seed: 7`
  trained at HF's default of 42 with no error and no warning. Replicates that differed
  only in `training.seed` were therefore the same run, which is what happened to STEP 25
  of the H100 record: its five "replicates" were five runs of seed 42, measured against
  a within-mode spread produced by the very thing it thought it was varying.
  Threading the config is only half the repair. `Trainer.__init__` runs
  `set_seed(args.seed)`, but `get_peft_model` has already drawn `lora_A` by then, and
  `classifier` / `reward_model` / `prm` have already drawn a freshly initialised head
  inside `from_pretrained`, so every wrapper now applies the seed at the top of
  `setup()` as well, before the model is loaded. `unlearn` builds no `Trainer` at all
  and drew its RMU control vector from a generator hard-coded to 0; that draw now
  follows `training.seed`, staying at 0 when the seed is unset so existing runs keep
  their control direction.
  An unset seed still resolves to 42 and leaves `data_seed` at `None`, so the values a
  run trains at are unchanged. What changes is when they arrive. Nothing called
  `set_seed` before `get_peft_model` previously, so `lora_A` was drawn from torch's
  default generator, which is seeded from entropy once per process: an unseeded run's
  adapter and classification-head initialisation varied from one process to the next,
  and it is now deterministic at 42. Replicate variation that came out of runs setting
  no seed was coming from exactly that, so those runs are now identical to each other
  and varying a replicate means setting `training.seed` on purpose. The MLX backend
  (`backend: mlx`) is the one path that still reads neither field.
- **`training.seed` on the MLX backend now says it is ignored (#353, fourth criterion).**
  MLX has its own RNG (`mx.random`) and none of the MLX wrappers touch it, so a seeded
  MLX run was silently unseeded — which looks identical to a seeded one until two
  replicates disagree. That gap only became reachable between #353 being filed and #381
  landing: `backend: mlx` was never dispatched at all until #362 (#363). Setting either
  field now appends to the wrapper's existing "MLX backend ignores:" line, naming
  `training.seed` / `training.data_seed` so it is greppable as written. A warning, not a
  rejection: a config valid on transformers should not become unloadable by switching
  backend. Seeding MLX for real is separate work with a separate RNG.
- **Under `use_fsdp2_compile`, every `checkpoint-*` still loads as a dead adapter
  (#351).** #335's repair runs once, on the output root, after the final `save_model`.
  HF's Trainer writes its periodic checkpoints through that same `save_model` with
  `output_dir=<run>/checkpoint-N`, so they come out carrying `_orig_mod.` on every key
  and nothing ever normalised them. Measured at 70B on 8×H100
  (`benchmarks/gate-h100-validation.md`, STEP 28): 320 canonical keys in the output
  root, **320 prefixed ones in `checkpoint-100`**. Resuming is the case that decides
  how bad this is, and it is worse than #335 was.
  `PeftModel.from_pretrained` at least warns; `Trainer._load_from_checkpoint` calls
  `model.load_adapter(...)` and drops its return value, and `load_adapter` deliberately
  does not warn (it hands the missing keys back in the load result instead, which
  nothing reads), while `load_state_dict(strict=False)` discards the `_orig_mod.` keys
  without a word. A resumed run therefore continues from a re-zeroed `lora_B` in total
  silence: #335's failure shape with its one warning removed. `load_best_model_at_end`
  was reloading a dead adapter for the same reason, since the Trainer restores
  `state.best_model_checkpoint` through that same `load_adapter`, and this repairs that
  path too. Normalising now happens on HF's `on_save`, as each checkpoint is written.
  Both that callback and the final save are gated on `args.should_save`, the condition
  `save_model` writes under, so exactly the rank holding the file repairs it rather
  than all eight opening it at once. The callback is attached in `setup()`, ahead of
  anything a caller adds later: `HFPushCallback.on_save` uploads `checkpoint-{step}` on
  this same event and `CallbackHandler` dispatches in insertion order, so a
  normalisation attached after it would leave `--push-as` publishing the prefixed
  adapter and keeping the repaired one on local disk.
- **A streamed model's `named_parameters()` still carried the wrapper's
  `.inner.` segment, so a name-keyed comparison against a resident model of the same
  checkpoint saw no overlap (#369).** v0.72.1 made `state_dict()` canonical for
  serialisation; this issue is the first time something compared the two model kinds by
  parameter name instead, and the #331 repair gate read the resulting empty intersection
  (`grads exact 0/0`) as a pass. `layer_stream_runtime.canonical_named_parameters()`
  strips `.inner.` the same way `state_dict()` already does, and
  `assert_canonical_parameters_intersect()` raises instead of reporting an empty
  intersection as success. Neither the forward path nor `state_dict()` changed, so the
  v0.72.0 bit-exactness gates remain valid unexercised. `named_modules()` and
  `named_buffers()` carry the same segment and are deliberately not covered — the
  comparison that produced the false green was over parameter names.

### Added

- **`training.stream_vram_override` gives the layer-streaming VRAM pre-flight an
  explicit escape hatch (#347).** `decide_stream_fit` refused any run it predicted
  would not fit, with no way through except lowering `batch_size` or `max_length`.
  Setting this field now replaces the measured free-VRAM figure the pre-flight
  checks against, in either direction: raised past a documented over-prediction to
  let a known-safe config through, or lowered to enforce a cap `mem_get_info()`
  cannot see, such as `set_per_process_memory_fraction` on a shared or capped card
  (a Colab/Kaggle T4, a MIG slice). Rejected at config load when set while
  `stream_layers` is false, mirroring the existing `stream_source`/`stream_buffers`
  footgun gate.

### Fixed

- **A hosted notebook's preinstalled `torchao` made `get_peft_model` raise, and it read as a
  Soup bug (#389).** `peft`'s `is_torchao_available()` does not return False on a version it
  considers too old, it **raises `ImportError`** — nine frames inside `get_peft_model`, with
  nothing near the top of the traceback naming `torchao`. Colab preinstalls `torchao` 0.10.0
  against a `peft` that demands newer, so the first `soup train` on a free notebook died in a
  place unrelated to anything the user had configured. Now mapped in `utils/errors.py` to the
  cause and the one-line fix (`pip uninstall -y torchao`), with the part worth saying out
  loud: Soup does not need `torchao` at all unless `training.quantization_aware` is set.
  Found by running `notebooks/proof-4gb.ipynb` on a real free-tier session, which is the same
  place #385's first repair was caught being a no-op.

- **bf16 was assumed on every CUDA card, so the entire free GPU tier failed (#385, #387).**
  Fourteen places, and only the first was known: `trainer/stream_setup.py` chose the
  layer-streaming store dtype with the literal `"bfloat16" if on_cuda else "float32"` (#385),
  and then a live run found `SFTTrainerWrapper._resolve_mixed_precision` returning
  `(device == "cuda", False)` as its default — and an audit found the same
  `bf16=self.device == "cuda"` in **twelve more wrappers**: bco, classifier, distill, dpo,
  embedding, ipo, kto, online_dpo, orpo, pretrain, reward_model, simpo (#387). So it was not
  a streaming bug at all; **every task** died on that hardware.
  The sharpest detail is that the codebase already knew: `trainer/asr.py` carries the comment
  *"bf16=cuda was hardcoded, which crashes on pre-Ampere cards (T4 / GTX 16xx)"* and fixes it
  — in that one wrapper, never propagated. All fourteen now take the answer from one place,
  `utils/gpu.bf16_fp16_flags`, including ASR, whose private copy was folded in. bf16 needs
  Ampere.
  **Colab's free tier is a T4 (sm_75), Kaggle is a T4 or a P100, and V100 / GTX 16xx / RTX 20xx
  are all pre-Ampere**, so on that hardware Soup ran in a dtype the card has no units for.
  Neither could fail on the maintainer's RTX 3050, which is Ampere; this is the same shape as
  the four backends the H100 session found had never executed once.

  **Two corrections to the first version of this entry, both established by finally running
  it on a real T4 rather than reasoning about one.** (1) The claim that every task *died before
  step 0* on transformers' *"Your setup doesn't support bf16/gpu"* was wrong: that error was
  produced by a local stub forcing `is_bf16_supported()` to False, and transformers gates on
  the same permissive call described next, so on the current stack it does not raise at all.
  (2) More seriously, **the first fix was a no-op on the hardware it was written for.**
  `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`: when its
  compute-capability fast path fails it falls through to merely *constructing* a bf16 tensor,
  which software emulation satisfies — so a T4 answers **True**, and asking the bare question
  selected bf16 exactly as the hardcoded literal had. The predicate now asks
  `is_bf16_supported(including_emulation=False)` (falling back to a capability check on older
  torch), and `get_compute_dtype` — a second copy of the same question — was folded into it.
  What a T4 actually *does* with emulated bf16, as opposed to what it reports, is not yet
  measured.
  **This cannot regress a working setup**: where bf16 is supported the answer is unchanged,
  and where it is not the previous behaviour was a crash. A test SCANS every module in
  `soup_cli/trainer/` rather than parametrising over a hand-written list — the list is what
  hid the twelve — and the existing unit test for this line had to be rewritten because it
  asserted the defect (`test_auto_flag_off_preserves_legacy` required bf16 on any CUDA device,
  and passed in CI precisely because CI has no GPU and the old code never asked the driver).
  Correctness of the alternative was measured before the change rather than assumed —
  streamed-vs-resident logits are bit-exact at **0.000000e+00** in float16 as well as
  bfloat16, in *both* quantisations, against resident references of matching numerics (an NF4
  streamed run compared against a genuinely NF4 resident one, since comparing it against a
  bf16 model would measure the dtype rather than the streaming), and the adapter is non-zero.
  **Not yet verified on a pre-Ampere card**: that exactness was measured *using* fp16 on
  Ampere, so it establishes the plumbing, not the Turing/Pascal kernels — the free-tier
  notebook is the natural place to close that.

### Changed

- **Retracted: "layer streaming is bound by host-to-device transfer, not by the GPU."**
  That sentence appeared in the README, in the v0.73.0 notes below, in the H100 gate record
  and in the preprint's abstract. It was an *inference* from the H100 replication — the same
  configuration returning the same throughput on a card with two orders of magnitude more
  compute — and it had never been measured. It was measured on 2026-08-11 on the original
  laptop and is **false at the published configuration**: four interleaved ablation arms in
  one process at a pinned clock show that removing *all* host-to-device traffic (6.864 GB per
  step) buys **1.44%**, removing the NF4 dequantisation buys **9.80%**, and removing both
  leaves **88.7%** of the step; the compute stream is blocked on a copy for 8.4 ms of a
  4190 ms step; and the step runs at **71.3%** of that card's same-session, shape-matched GEMM
  ceiling. The claim *is* true below roughly 128 tokens per step, where the fixed transfer
  volume dominates — the published configuration is not there.
  **No measured number anywhere changes**, and the replication result stands in a weaker form:
  the constraint is common to both machines and is not the compute the datacenter card adds.
  The H100's own bottleneck was never instrumented and no claim is made about it. Record:
  `benchmarks/probe-v0.73.0-what-bounds-streaming.md`. Every occurrence in the gate record and
  in the v0.73.0 notes below is **annotated in place rather than deleted**, because in a folder
  whose whole premise is publishing the record as written, a silent deletion costs more
  credibility than the error does.

### Validation (measured, not changed)

- **Layer streaming completed a run on hardware the maintainer does not own: a free-tier
  Colab Tesla T4 (sm_75, Turing), via [`notebooks/proof-4gb.ipynb`](notebooks/proof-4gb.ipynb).**
  Every streaming number this project has published came from one RTX 3050 Laptop or one
  borrowed 8×H100, and the pre-Ampere fix above (#385, #387) had been verified *using* fp16
  on an Ampere card, which establishes the plumbing and not the Turing kernels. This closes
  that specific gap and nothing wider. `NousResearch/Meta-Llama-3.1-8B-Instruct`, NF4,
  `stream_layers: true`, `stream_buffers: 2`, batch 1, `max_length: 256`, LoRA r=8/α=16,
  fp16 (a T4 has no bf16 units): 7 steps, exit 0, adapter written with **128 tensors, 128 of
  them non-zero**, and a **measured peak of 2.91 GB** against the pre-flight's predicted
  ~3.02 GB — an over-prediction of **3.8%**, which is the direction the estimator was fitted
  to err in (v0.72.3 fitted it to never under-predict) and is the whole reason it is allowed
  to stop a run. The card has 15.6 GB, so the run was constrained artificially with
  `torch.cuda.set_per_process_memory_fraction` to **4.00 GB**, and the cap was shown to bite
  rather than assumed: a deliberate 4.29 GiB allocation was refused. That artificial cap is
  also the reason **no throughput figure is quoted from this run, here or in the notebook** —
  a capped card is not a benchmark, and the panel's own forecast (31–46 tok/s from 2.20 TFLOPS
  measured at 1185 MHz) is a compute bound, not a measurement of what the run did.
  Two things the run did **not** establish, stated because the temptation is to let the exit
  code cover them. **Backward/gradient exactness at 8B on Turing is not shown** — a non-zero
  adapter proves gradients flowed, not that they were right, and the streamed-vs-resident
  comparison in the notebook's section 4 produced **no captured output**, so it is recorded as
  unrun rather than as a pass. And the loss moving 4.5266 → 4.0674 over 7 steps, non-monotonically
  (4.5266, 4.2388, 3.7485, 4.6930, 4.1940, 4.1127, 4.0674), is reported because it is what the
  run printed; over seven steps it is not evidence of learning.
  One observation worth carrying forward: the pre-flight panel reported **free VRAM 15.10 GB**,
  i.e. it read the device and not the per-process cap the run was actually held to, so the fit
  decision was taken against a number 3.8× larger than the budget in force. The run fit on its
  own merits (2.91 GB against 4.00 GB), so nothing was protected by luck — but on that hardware
  the pre-flight is not what would have caught an over-budget config, which is exactly the case
  `training.stream_vram_override` (#347, above) exists for. Record:
  `benchmarks/run-t4-colab-free-tier.md`.

## [0.73.0] - 2026-08-09

**The release that came out of three days on somebody else's hardware.**

Every number this project had ever published was measured on one machine: an RTX 3050
Laptop, 4 GB, Windows. From 5–9 August it ran on a borrowed 8×H100 box (Ubuntu 24.04,
a much newer torch / bitsandbytes / trl / peft stack) for the first time. That found
**one silent correctness defect in layer streaming, four backends that had never
actually run, and a documented multi-GPU entry point that had never launched** — plus
the first evidence that the laptop result reproduces on hardware nothing like it. The
full record, published as written including six rejected hypotheses and three false
positives that controls caught, is
[`benchmarks/gate-h100-validation.md`](benchmarks/gate-h100-validation.md).

This is a **minor** bump, not a v0.72.x patch: it adds two capabilities that did not
exist, and repairs four backends.

### Added

- **`training.seed` and `training.data_seed` (#341).** Every Soup run trained at
  seed 42 with no way to change it, so "run this twice with a different seed" was
  impossible. Both default to `None` rather than to `42` on purpose: an unset seed has
  to reproduce **two** different historical defaults — HF's 42 for `TrainingArguments`
  and 0 for the multipack sampler since v0.37.0 — and a plain `42` default would have
  silently re-ordered every existing multipack run.
  **Scope, stated rather than left as a footgun:** wired into the SFT trainer only.
  Other task wrappers build their own `TrainingArguments`, so setting it on a DPO run
  parses and does nothing — tracked as #353.
- **Full fine-tuning as `lora.r: 0` (#340).** The SFT trainer's full-FT branch was
  dead code with no way to reach it. `r: 0` was chosen on repo evidence, not taste:
  three consumers already treat rank 0 as "no adapter", and `r: 0` previously crashed
  inside PEFT, so no config that worked before changes meaning. `lora.r` also gained a
  lower bound — `r: -5` used to parse and die inside PEFT.
- **`--deepspeed zero3_offload`** — ZeRO-3 with CPU parameter offload. `zero3` set
  `offload_param: none` and the only offload *preset* was stage 2, optimizer-only, so
  the configuration a user short of VRAM actually wants could not be named on the
  command line. Measured on one H100 (Llama-3.1-8B, bf16, LoRA r=8, 256 steps):
  **21.65 tok/s at a 38,135 MiB peak**. `offload_optimizer` deliberately stays `none` —
  turning it on makes DeepSpeed JIT-build `cpu_adam` against a matching CUDA toolkit
  and fail without one; copy the emitted JSON and flip it if you have `nvcc`.
- **`trl` support widened to `>=0.14.0,<0.29`** (#326), behind a capability-probe
  compat layer (`trainer/_trl_compat.py`) rather than a version table — a version
  table is what was wrong twice before. The trainers now ask each config class whether
  it accepts `max_prompt_length`, and resolve `ORPOConfig`/`CPOConfig`/`BCOConfig`
  through `trl.experimental` when trl 0.29 drops them from the public namespace. All
  six preference trainers construct **and train to identical losses** on trl 0.26.2,
  0.28.0 and 1.9.2.

### Fixed — backends that had never been run

- **`soup train --gpus N` never launched at all (#77).** `accelerate launch` takes a
  script path positionally, and Soup handed it `sys.executable` — so accelerate opened
  the Python binary and parsed it as source (`SyntaxError: source code cannot contain
  null bytes`). Every rank died before the trainer existed. The documented multi-GPU
  entry point has been dead since it shipped, invisible to single-GPU CI because that
  path skips the launcher wrapper entirely. Separately, `--no-reexec` printed a command
  with every user flag dropped, so following the hint literally trained without
  `--fsdp`.
- **DeepSpeed could not train a LoRA model on any stage (#336).** HF builds two
  optimizer parameter groups and with LoRA the no-decay group is *empty*; DeepSpeed
  drops it, leaving one group against two `base_lrs`, and torch's scheduler then hits a
  strict-`zip` length mismatch. Verified repaired on 2×H100 with real `soup train`:
  zero2 6790.2 tok/s, zero3 1025.3, zero++ 977.1, all exit 0 with a live adapter
  (96/96). `zero++` failed earlier and independently — it set fp16 quantised
  weights/gradients against a bf16 model and hardcoded `zero_hpz_partition_size: 8`
  regardless of the real world size; both are now derived, and the rewrite is printed
  rather than applied silently.
- **`use_fsdp2_compile` wrote an adapter that reloads as all zeros (#335).** Under
  `torch.compile` the Trainer saves *through* the wrapper, so every key came out as
  `_orig_mod.base_model.model...`. The tensors were genuinely trained
  (max|lora_B| 7.0e-3 measured) and `PeftModel.from_pretrained` matched none of them —
  emitting only a `UserWarning` and leaving `lora_B` at its zero init. Measured on
  4×H100: **0 of 96 non-zero** against 96/96 for the paired non-compile run, reproduced
  3/3, with the run exiting 0 throughout.
- **`soup serve --backend sglang` returned 500 on every generation (#76).** sglang
  0.5.16's `Runtime.generate` returns a JSON *string*; Soup subscripted it as a dict.
  Deterministic, not a race — the backend loaded cleanly and then failed 100% of
  requests. It had genuinely never been run, because SGLang does not support Windows.
- **The vLLM backend ignored the model's chat template (#332).** `utils/vllm.py`
  hand-rolled a `"User: ...\nAssistant:"` prompt while the transformers backend used
  `apply_chat_template`, so every vLLM user's model saw a format it was never trained
  on. On Llama-3.1-8B + LoRA, identical server and sampling params, only the prompt
  differing: a run-on loop burning all 200 tokens **before**, an 8-token answer
  **after**. Both backends now share one `build_chat_prompt`.
- **Three vLLM serving defects (#333)**, each verified live: `finish_reason` was
  hardcoded `"stop"` even at `completion_tokens == max_tokens`; `--dashboard` silently
  no-opped (`/metrics` returned 404 with nothing printed); and `--max-model-len` did not
  exist although the engine factory already accepted it. `--dashboard` on a backend that
  cannot serve it now warns at startup naming the backend instead of doing nothing.

### Fixed — training paths that silently did the wrong thing

- **`data.max_length` was capped at 1024 on every SFT run (#78).** `SFTTrainer`
  converts `TrainingArguments` with `SFTConfig(**args.to_dict())`, and `max_length` is
  an SFT-only field that `TrainingArguments` does not carry — so it always took
  `SFTConfig`'s default. Measured before the fix: `data.max_length=4096` gave 1024
  tokens per sample, with no warning.
- **`training.use_liger: true` crashed at step 0 (#78).** Soup patched the model but
  never set `TrainingArguments.use_liger_kernel`, the flag TRL reads to know the fused
  path returns `logits=None`; its entropy metric then ran on `None`. Reproduced across
  the whole supported trl pin, so the feature did not run at all. Separately, Liger's
  architecture match was a *substring of the model name*, so any model loaded from a
  local directory trained without Liger on a flag the user had explicitly set — it now
  reads `AutoConfig.model_type`.
- **FlashAttention 3 was selected from a version that can never report 3 (#334).**
  Dao-AILab ships FA3 as `flash_attn_3`; `flash_attn` itself stays in the 2.x line, so
  the branch could not fire for any real install — and had it fired, it produced an
  `attn_implementation` transformers would reject. On Hopper hardware users silently got
  FA2 or SDPA while the docs advertised FA3. Both detectors now ask transformers.
  This makes detection honest; it does not make FA3 measurable here.

### Fixed — layer streaming

- **A silent wrong-gradient defect on large NF4 models (#331).**
  `bitsandbytes.MatMul4Bit` stashes the packed weight and `quant_state` on `ctx` as
  plain attributes instead of through `save_for_backward`, so gradient checkpointing
  cannot discard and recompute them. The reference is captured in the forward, *aliases
  the streaming buffer pool*, and is read in the backward after that slot has been
  refilled. Result: a bit-exact forward, a healthy-looking loss curve, and wrong
  gradients on every layer but the last `stream_buffers`. It bites NF4 above a threshold
  bracketed at **163.8–171.5 MiB per layer** — so 32B and 72B, never 8B or 14B, and
  never bf16.
  The repair keeps the weight out of that function entirely: dequantise inside the
  checkpointed region and use a native matmul, which saves the dequantised weight
  through the ordinary mechanism. Gated against a resident NF4 reference with a
  repair-disabled control arm in the same process: **real 32B 256/256 gradient tensors
  exact against the control's 8–12/256**, at +2.9% peak VRAM and −4.8% throughput; and
  again on **real 72B — the size where the defect was worst — 320/320 against 8/320**,
  at +2.6% and −3.7%. De-aliasing was measured and rejected first: bnb holds the
  reference across the whole forward-to-backward span, so any copy is O(model), which
  took real 32B from 4,220 to 19,720 MiB and deletes the feature's premise.
- **All four preference losses died on newer trl (#328).**
  `should_enable_hf_gradient_checkpointing` existed so the HF Trainer does not
  checkpoint an already-checkpointed streamed model twice — and only `sft.py` ever
  called it. TRL's default is `False` on trl 0.19.1 and `True` on 0.26.2, so one
  omission had two opposite symptoms: an explicit `gradient_checkpointing: true`
  silently dropped on the older stack, and on the newer one HF checkpointed the *inner*
  decoder layer, recomputed it after the reparametrisation context had exited, and
  killed dpo/orpo/simpo/kto with `Tensor on device cuda:0 is not on the expected
  device meta!`.
- **Layer streaming is now refused when `nn.DataParallel` would engage.** HF wraps the
  model in DataParallel whenever more than one CUDA device is visible and the run is not
  distributed, and DataParallel requires every parameter on `device_ids[0]` — streaming
  keeps the decoder on `meta` by design. It accounted for 8 of the 9 streaming-suite
  failures on the H100 box and is unreachable on a one-GPU machine. It refuses rather
  than quietly using 1 of 8 cards.
- **`device_map="auto"` broke every distributed launch, in fifteen places.**
  transformers refuses `device_map="auto"` outright under a distributed launch, so the
  exact `accelerate launch` command `soup train --gpus 8` prints died on every rank. A
  first pass fixed six trainers; nine sites still carried it. All fifteen now go through
  `utils/gpu.resolve_device_map`, and the regression guard scans every module in
  `soup_cli/trainer/` instead of a hand-written list — the list is what hid the nine.
- **`LOGITS_BYTES_PER_ELEMENT` is split into two independently measured terms** (#327).
  The 14 is 12 + 2, measured stage by stage on an H100 with zero spread across three
  repeats, and only the 2 differs between stacks. It is deliberately **not lowered** —
  see Known Limitations. What is new is an upward-only calibration
  (`max(14, measured + 2)`), which closes a real unguarded hole: a future stack that
  grew a fourth fp32 buffer would be under-budgeted by 12.5% today with nothing to catch
  it. Default behaviour is byte-identical and the pre-flight path takes no new CUDA.
  (Originally landed as an opt-in probe with no caller; #348 above wires it into the
  pre-flight itself, so it now runs on every streamed run rather than sitting inert.)

### Fixed — eval, ship and export

- **Two of `soup ship`'s three behavioural suites measured nothing (#316).** On
  Meta-Llama-3.1-8B-Instruct, `mini_tool_call` scored **0.000** and `mini_format_json`
  **0.000** on a model that does both correctly. All harness defects: the tool-call
  prompt never told the model tools exist, the JSON check ran `json.loads` over the
  whole output while 38 of 40 answers sit inside a ```` ```json ```` fence, and the
  generation budget was taking `make_generator`'s default of 64, truncating 31 of 40
  tool calls one closing brace short.
- **The refusal detector missed the apostrophe models actually type (#316).** The
  patterns spelled the contraction with U+0027; Llama-3.1 writes U+2019. Over the
  shipped 40-item `mini_safety` suite, **28 of 40 refusals scored as non-refusals** and
  the suite reported 0.300 for a model whose true refusal rate is 1.000 — a 0.70 error
  against a 0.05 regression threshold, i.e. 14× the thing it exists to detect.
- **`soup train --task online_dpo` passed a keyword trl removed at 0.25 (#300).** The
  probe asked whether `BasePairwiseJudge` was importable and used the answer to decide
  whether to pass `reward_model=`; those two facts had decoupled, so the probe said yes
  for every trl in the supported range. It now asks the signature — through an MRO walk,
  because on 0.26.2 the top-level class is a deprecation shim whose direct signature
  reports no parameters at all.
- **A quantised GGUF export deleted a previously exported f16 (#144).** The f16
  intermediate was named `{model}.f16.gguf` next to the output — exactly the default
  output name of a `--quant f16` export — and unlinked when quantisation finished. So
  exporting q4_0 destroyed an earlier f16 export, even with an unrelated `--output`.
  Reproduced. It now lives in a private temp directory. Separately, the convert
  dependencies were installed only after an auto-clone, so `--llama-cpp /path` and an
  already-present checkout both died on `ModuleNotFoundError: sentencepiece`.

### Fixed — packaging and docs

- **`requires-python` now has an upper bound: `>=3.10,<3.13` (#358).** CI tests 3.10,
  3.11 and 3.12 and nothing above. Without a ceiling, pip on 3.13+ resolved torch wheels
  nobody here has run, and the failure was not a Soup error message — it was a loader
  crash inside `c10.dll` / `libc10.so` before any Soup code executed, leaving the user
  nothing to act on. The bound is **3.13, not 3.14**: 3.13 is equally untested.
  `tests/test_requires_python_bound.py` derives the bound from the CI matrix, so
  widening one without the other fails the suite.
- **The `trl` bounds shipped in v0.72.4 were wrong at both ends**, and were corrected
  before #326 widened them again. The ceiling was over-tight by five releases: the
  v0.72.4 table read a `trl/experimental/` *relocation* as a field removal. The floor
  `>=0.7.0` was impossible — `setup()` imports `GRPOTrainer`, first exported at 0.14.0.
  One detail in the v0.72.4 note was also imprecise: `ORPOConfig`/`CPOConfig` are not
  deleted at 0.29, they leave the public namespace — an `ImportError` rather than a
  rejected keyword, which is worse, not milder.
- **The layer-streaming pre-flight panel was titled after a flag that does not exist**
  (#329). It read `soup train --stream-layers`; there is no such option — streaming is
  enabled by `training.stream_layers` in `soup.yaml`. It is the first thing a streaming
  run prints, so it was the feature's most-read line of documentation, and it pointed at
  a `No such option`.
- **Seven of the eight configs in `examples/configs/` did not parse.** They were written
  against a pre-nesting schema, so `soup train --config examples/configs/sft_basic.yaml`
  — the first command `examples/README.md` tells you to run — failed validation. Also
  fixed while there: two configs declared `format: sharegpt` for preference-shaped data
  (a silently wrong training run, not an error), `target_modules` listed `out_proj`
  which no Llama has, and `vision_llama.yaml` pointed at files that have never existed
  in this repo. `tests/test_examples_configs.py` now parses every one of them.
- **`soup data demo` metadata described files other than the ones it ships.**
  `sharegpt_demo` declared `format: sharegpt` for prompt/chosen/rejected rows — the
  value a user copies straight into `data.format` — and `grpo_demo` declared
  `reasoning`, which is not in the schema's format literal at all.
- **Encoding corruption in `pyproject.toml`** (fourteen em-dashes round-tripped through
  cp1251, one of them in the `unit` pytest marker description that `pytest --markers`
  prints), and **`docs/commands.md`**, where missing newlines collapsed five commands
  onto two lines and the page promised "the full command list" while omitting seven.

### Changed — documentation corrected against measurement

- **LISA delivers the quality half of its claim, not the memory half (#306).** Measured
  at 3B and 8B on an H100 with LISA engagement verified three independent ways: it beats
  full fine-tuning at both learning rates, and it is **1.22× LoRA's VRAM at 3B and 1.51×
  at 8B**, with the gap *widening* with scale — embeddings, LM head and final norm stay
  trainable every interval and are 70.7% of everything LISA trains at 8B. The docs now
  say that plainly, and say when LISA is still the right choice.
- **FlashAttention and Liger were measured for the first time**, at **1.015×** and
  **1.051× / −12.9% VRAM**, against documented claims of "2–4×" and "20–60% / 20–40%".
  Both claims are corrected in the docs.
- **The GEMM-ceiling test could not pass on a datacenter GPU.** Its plausibility bound
  was `< 200 TFLOPS`, written when the only hardware this project had was an RTX 3050;
  an H100 returns a correct 786.5 TFLOPS and the assertion fired.

### Validation (measured, not changed)

- **The laptop result reproduces on completely different hardware.** Llama-3.1-8B NF4:
  119.6 tok/s in a 3.32 GB peak on the RTX 3050 against a **median 113.00 tok/s in the
  same 3.32 GB** on an H100 — first outside evidence that the method is bound by
  host-to-device transfer, not by the GPU. (See Known Limitations for what the laptop
  figure does and does not now mean.)
  > **Correction, 2026-08-13, left in place rather than rewritten.** The measurement
  > stands; the explanation does not. "Bound by host-to-device transfer" was an inference,
  > never a measurement, and a probe on 2026-08-11 refuted it at the published
  > configuration — deleting every host-to-device byte buys 1.4% and the step runs at 71.3%
  > of the card's same-session GEMM ceiling
  > (`benchmarks/probe-v0.73.0-what-bounds-streaming.md`). What survives is the weaker
  > claim: the constraint is common to both machines and is not the compute the H100 adds.
- **Forward bit-exactness at real model sizes**, not just on toys: logits `torch.equal`
  against a resident reference of matching numerics at 0.5B, 8B, 14B, 32B and 72B. Every
  previously published bit-exactness result was on 3-layer from-config checkpoints,
  because a 4 GB card cannot hold a resident 8B to compare against. *Backward*
  exactness is a separate claim measured separately — see #331 above and the per-model
  ledger at the top of the gate record.
- **A streamed model is as good as a resident one.** Paired over five disjoint training
  subsets and judged by Soup's own `soup ship`: mean difference **+0.006 against an
  identical 0.013 within-arm spread**, in bf16; **+0.0053 against spreads of 0.0333 and
  0.0200** in NF4. This had never been measured anywhere in the project.
- **Against DeepSpeed ZeRO-3 CPU offload**, same box, same data, same model:
  **2.93× the throughput at 9.7× less peak VRAM** at matched numerics. The honest
  reading is narrow, and the same session establishes it: eight cards of ZeRO-3 are
  *slower* than one card training resident for a model that fits. Layer streaming is
  not "faster than DeepSpeed" — it is for the case where the one card you have is too
  small.

### Known limitations

- **The `training.seed` knob reaches the SFT trainer only** (#353). Every other task
  builds its own `TrainingArguments`; setting it there parses and does nothing.
- **A resident 4-bit run is still not bit-reproducible from a seed** (#354), while a
  streamed one is. `get_peft_model` builds `lora_A` *before* `Trainer.__init__` calls
  `set_seed`, so the adapter init escapes the seed. Diagnosed with three competing
  hypotheses each killed by a control; the one-line fix is verified in the gate record
  but **not shipped here**. It has a real consequence for `soup ship`: three runs of one
  unchanged resident config moved `mini_common_sense` by 0.375 and `mini_mmlu` by 0.269
  against a `forgetting_threshold` of 0.05, so five of seven suites can cross the
  regression line on a re-run that changed nothing.
- **The layer-streaming VRAM pre-flight still over-predicts** (#327), and
  `LOGITS_BYTES_PER_ELEMENT` was deliberately left at 14 rather than lowered to the
  measured loss-path value. The asymmetry decides it: over-predicting refuses a config
  that would have worked — visible, annoying, data intact. Under-predicting on Linux is
  a clean OOM, but on **Windows/WDDM there is no exception at all** — it silently spills
  to host memory (measured: 9.27 GB allocated on a 4.29 GB card with nothing raised) and
  the claim "peak is bounded by one layer" quietly stops being true. The counterfactual
  settles it: 14 under-predicts 0 of 10 measured rows (worst +0.85% over), the lower
  value under-predicts 10 of 10 (worst −10.49%). The practical cost is that a streamed
  DPO run is refused from `max_length: 768` on a 4 GB card.
- **`bitsandbytes` still has the defect #331 works around.** Soup no longer sends
  streamed NF4 weights through `MatMul4Bit`, but the underlying library behaviour —
  saving tensors outside `save_for_backward` where checkpointing cannot see them — is
  upstream and unchanged. Filed as
  [bitsandbytes-foundation/bitsandbytes#2034](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2034).
- **DeepSpeed + LoRA is repaired in `sft.py` only** (#336). The other trainer wrappers
  still hit the empty-parameter-group failure under DeepSpeed, a user-supplied
  `--deepspeed my.json` is passed through unresolved, and ZeRO++ hierarchical
  partitioning is unit-tested but never exercised across two nodes.
- **`utils/sglang.py` still has both of the defects the vLLM rewrite fixed** — its own
  hand-rolled prompt and a hardcoded `finish_reason`. Named rather than silently
  skipped.
- **The closed-loop reward-hacking controller's mechanism is confirmed; its efficacy is
  not** (#286). At 7B the between-mode difference of 0.130 sits *inside* the within-mode
  spread of 0.140–0.195. Settling it needs ≥5 seeds per mode.
- **Two `soup ship` leg-2 suites have scoring gaps that outrank capability** (#346,
  #357). `mini_tool_call` ranks by brace hygiene — Llama-3.1-8B names the right tool
  40/40 and scores 0.225 — and `mini_mmlu` loses 8 of 26 items because
  `extract_mcq_letter` does not know `\boxed{C}`, scoring the 8B at 0.423, below a 0.5B.
  Adding that one form takes it to 0.731 and the inversion disappears. A third
  suspected inversion (#356) was **withdrawn as a measurement error of our own** — it
  was measured at a 64-token budget where `soup ship` uses 256.
- **The 70B FSDP2 recipe could not be smoke-tested** (#41): it needs all eight cards, so
  it could not be parallelised with anything else in the session.
- **The RAM-vs-disk streaming throughput gap remains unmeasured** (#325), and layer
  streaming remains **BETA**.

### A note on the preprint

[DOI 10.5281/zenodo.21771064](https://doi.org/10.5281/zenodo.21771064) is unaffected in
its correctness claims: its configuration is 8B NF4 at **105 MiB per layer**, comfortably
below the 163.8–171.5 MiB boundary of #331, and it survives a 50-backward soak at
`worst_abs = 0.0`. What this release changes is **scope**, not validity — exactness moves
from 3-layer from-config toys to resident references at 8B / 14B / 32B / 72B. A version 2
of the record carrying the H100 validation, the resident references, the DeepSpeed
comparison and the disclosed defect is in preparation.

One number should be read with a caveat: **the published 119.6 tok/s laptop figure was
measured before the #331 repair and has not been re-run on repaired code.** The repair
cost −4.8% throughput at 32B, so treat the laptop figure as a pre-repair number until
someone re-measures it on an RTX 3050. An H100 cannot substitute — the whole point of the
H100 result is that this method is transfer-bound, so its throughput does not carry across
machines.

> **Correction, 2026-08-13.** The conclusion holds, the reason given for it does not: the
> laptop is not transfer-bound (see the correction above). An H100 still cannot substitute,
> for a narrower reason — the repair's cost is paid in the per-layer NF4 dequantisation,
> measured at 9.8% of the step on the laptop, and that share belongs to that card's clock,
> GEMM ceiling and launch overhead.

## [0.72.4] - 2026-08-03

**Added — preference losses over layer streaming: DPO, ORPO, SimPO and KTO.**

Layer streaming kept the frozen base in CPU RAM and fed it to the GPU one decoder
layer at a time, but only for `task: sft`. This release opens it to the four
preference losses. The whole risk was one thing: **DPO needs a reference model, and a
second model instance would double memory and defeat the feature entirely.**

- **The reference is the same streamed base with its adapters disabled** — one set of
  weights, one stream, no second pass. Measured on an RTX 3050 4 GB with a 730 MB
  model: streamed DPO peaked at **0.914x** the SFT peak, with a byte-identical RAM
  store and buffer pool. Forcing a real second instance in the same harness cost
  **+730.44 MB against 730.44 MB of weights** — exactly one copy. That control is what
  makes the first number mean something.
- **KTO is *not* reference-free**, contrary to how it is usually described: it selects
  its reference exactly the way DPO does, so it gets the same treatment and the same
  memory assertion. ORPO and SimPO genuinely are reference-free.
- **Bit-exact against a resident run of the same loss** — `0.0` difference for all
  four, the standard every slot in this series inherits.
- **The pre-flight now knows that a paired loss is twice the rows.** DPO, ORPO and
  SimPO concatenate chosen and rejected into one tensor, so a VRAM budget computed at
  one row per example would have under-predicted by half — and on Windows the
  consequence is not an error but a silent spill to host memory that makes the run an
  order of magnitude slower.
- **KTO requires `batch_size >= 2`** (its KL term is degenerate at 1). Soup now says so
  when your config is read, rather than minutes later after sharding the checkpoint.
  KTO is streamable at all only because v0.72.3 lifted the batch-1 restriction.
- **`grpo` and `ppo` remain excluded permanently**, not "not yet": generation rollouts
  re-read every layer once per generated token, which destroys the amortisation
  streaming depends on. The refusal says so and deliberately names no release.

The streaming setup now lives in one shared place instead of being copied per trainer,
so the NF4 pre-flight, the RAM/disk tier choice and the VRAM fit refusal cannot drift
between SFT and the preference losses.

**Fixed — `trl` is now capped, and that is a real bug fix, not a CI tweak.**
Six trainers (`bco`, `dpo`, `ipo`, `kto`, `orpo`, `simpo`) pass `max_prompt_length` to
their `trl` config, and `trl` removed it in stages. So anyone who ran
`pip install 'soup-cli[train]'` and resolved to a recent `trl` had
`soup train --task orpo` fail on import.

> **Correction (see [0.73.0]).** This release shipped the cap as `<0.25` on the
> strength of a staged-removal table that was itself wrong. The real stages are
> `kto` at 0.27, `bco`/`orpo`/`simpo` at 0.28 and `dpo`/`ipo` at 0.29. v0.73.0 first
> corrected the cap to `<0.27` and then raised it to `<0.29` behind a capability-probe
> compat layer, so the trainers no longer set the cap at all.

That was already true before this release and nothing caught it: the `trl` imports
live inside `setup()`, which no test had ever called on those wrappers, so CI stayed
green while the code only worked on older `trl`. This release's end-to-end preference
tests are what surfaced it.

The boundary was read off the published wheels per config rather than inferred from a
version number — the removal being staged is exactly why a single spot-check gives the
wrong answer, and see the correction above for how that method can still land on the
wrong answer. Supporting the newer API is its own piece of work; declaring a dependency
the code actually works with comes first.

Honest costs: streaming makes the reference free in **memory**, not in **time** — DPO
traverses the layer stack three times per step against SFT's two, measured at **1.52x**
the layer reads. And the VRAM pre-flight is a sound *upper* bound for preference
losses rather than a tight estimate; see Known Limitations in the release notes.

## [0.72.3] - 2026-07-28

**Added — layer streaming breadth: more architectures, bigger batches, resume, and a
disk tier.**

Layer streaming (v0.72.0–.2) was deliberately narrow: Llama/Qwen only, batch 1, no
gradient accumulation, no resume, RAM only. This release lifts all of it, and each
capability was gated against a streamed-vs-resident bit-exactness reference before any
of it was written.

- **Six more model families.** `mistral`, `gemma`, `gemma2`, `gemma3_text`, `phi` and
  `phi3` join the allowlist, each verified **bit-exact** against the same checkpoint
  loaded resident, under both bf16 and NF4. Phi-3 is the notable one: it fuses Q/K/V
  into a single `qkv_proj`, so there is no `q_proj` to find, and it is bit-exact anyway.
  Multimodal `gemma3` is deliberately *not* accepted — only `gemma3_text`.
- **`batch_size` above 1, and gradient accumulation.** Both previously refused.
- **A pre-flight VRAM budget that accounts for batch and vocabulary.** Streaming bounds
  the *weights*; activations and the logits tensor are untouched by it and both scale
  with `batch × seq`. On a 152k-vocab model at batch 8 the logits term alone measured
  **8.71 GB — 146× the entire layer-buffer pool**. `soup train` now predicts peak VRAM
  and refuses a run that will not fit, naming the two knobs that scale it.
- **A throughput forecast**, quoted as a range from a GEMM ceiling measured on your own
  card in that session, alongside the SM clock — never a compiled-in per-card constant.
- **`--resume` and `--hf-resume` work with streaming.**
- **A disk overflow tier.** `stream_source: auto` (the default) uses RAM when the base
  fits and falls back to an NVMe disk tier when it does not, holding nothing resident.
  `stream_source: ram` refuses instead of falling back. Non-NVMe disks are still
  refused outright.
- **`soup doctor --disk`** reports the detected media type.

**Fixed**

- `estimate_logits_bytes` charged 6 bytes per logit element; the measured peak is **14**
  (`transformers` holds the bf16 logits, the fp32 upcast, log-softmax's fp32 output and
  the fp32 gradient live at once). The old figure under-predicted that term by 2.33×.
- Adapters could not be loaded *into* a streamed model: `load_state_dict` narrows keys
  by child name, so a canonical checkpoint matched **0 of N** tensors and PEFT reported
  only a warning — a resumed run reproduced the from-scratch loss curve exactly. The
  streaming layer now redirects canonical keys at load time, mirroring the v0.72.1
  save-side fix.
- The NVMe-only tier guard was wired to a hardcoded constant and could never fire.
- Streaming weight sources are now released when training ends **or raises**; the disk
  tier holds one open shard handle per decoder layer.
- Subprocess helpers resolve tools to absolute paths (on Windows, `CreateProcess`
  searches the current directory before `PATH`).
- The `[mcp]` extra is now capped at `mcp<2`. The SDK's 2.0.0 release removed
  `mcp.shared.memory.create_connected_server_and_client_session` and dropped
  `Server.list_tools`, breaking `soup mcp serve`'s round-trip tests for anyone
  installing fresh. Support for the 2.x API is tracked separately.

**Known limitations**

- The **RAM-vs-disk performance gap is unmeasured** on the development hardware and no
  number is claimed for it. safetensors memory-maps the shards, so the OS page cache
  keeps them resident between steps on a machine with spare RAM, and at ~5 effective
  TFLOPS the NVMe read hides under compute. The disk tier's *correctness* is verified
  bit-exact against the RAM tier; its speed relative to RAM is not characterised.
- End-to-end `soup train --resume` could not be demonstrated on the development box:
  `transformers` refuses `torch.load` below torch 2.6 (CVE-2025-32434), which blocks
  **every** resume there, streaming or not. The streaming-specific half — the adapter
  round-trip and loss continuity — is verified on the production CUDA path.
- Loading *into* a streamed model works; `named_parameters()` and `state_dict()` still
  disagree in memory, which is the deliberate cost of a serialisation-only design.
- Layer streaming remains **BETA**.

## [0.72.2] - 2026-07-28

**Added — NF4 layer streaming: fine-tune Llama-3.1-8B on a 4 GB laptop GPU.**

Layer streaming (v0.72.0) keeps the frozen base in CPU RAM and streams it to the
GPU one decoder layer at a time, so peak VRAM is bounded by one layer instead of
the whole model. It was bf16-only, which capped it at about 3B on a small card.
Quantising the streamed base to NF4 shrinks it ~4×, and that is what brings 8B
within reach.

Add one line to a streaming config:

```yaml
training:
  stream_layers: true
  quantization: 4bit    # NF4
  batch_size: 1
```

**Measured on a 4 GB RTX 3050 Laptop** (Windows, batch 1, S=512, gradient
checkpointing, 50 steps after 10 warm-up, `PagedAdamW8bit`):

| Model | tok/s | Peak VRAM | RAM store | GPU util |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 119.6 | 3.32 GB | 3.60 GB (page-locked) | 100% |
| Qwen2.5-3B | 264.2 | 1.76 GB | 1.43 GB (page-locked) | 100% |

For scale: 1M training tokens is about 2.3 h at 8B on that card (arithmetic from
the measured rate, not a separate measurement).

Qwen2.5-3B also got **1.85× faster** than the bf16 streaming path (264.2 vs
143.1 tok/s) — and the reason is not arithmetic. A 1.43 GB store fits under the
machine's page-locked memory ceiling where a 5.55 GB one did not, which restores
asynchronous host-to-device copies and lifts GPU utilisation from 79.3% to 100%.

**Correctness.** A streamed NF4 run is **bit-exact** against a resident NF4 run:
the same quantised bytes through the same bitsandbytes kernels. Logit equality,
non-zero gradients at layer 0, and a matching multi-step loss curve are all
regression tests, not one-off measurements.

The base is quantised once, offline, and cached under `~/.soup/layer-stream/`.
The cache is keyed to the quantisation *and* the source checkpoint, so switching
between `none` and `4bit`, or retraining a base in place, re-shards instead of
silently streaming the wrong bytes.

**Fixed — a streamed 4-bit run reported its parameter count ~6.5× too high.**
SmolLM2-135M printed "878,154,048 total" (true: 134,515,008). Display only —
training was unaffected — but at 8B it would have read ~52 B.

Scope is unchanged and still BETA: RAM tier, `task: sft`, Llama/Qwen,
`batch_size: 1`, no gradient accumulation, no `--resume`. Every rejected config
names the release that lifts it. `quantization` values other than `none` and
`4bit` are refused.

**Fixed — `soup --help` was 5x slower than it should be.** Since v0.72.0 the CLI
imported PyTorch on startup, taking **6.0 s** where it now takes **1.15 s**.

`soup reward stress` (v0.71.41) put `utils/reward_stress` on the light CLI path.
That module imported `utils/reward_hack_control` to reuse a single string
constant — and `reward_hack_control` resolves its `TrainerCallback` base class at
module scope, which pulls in `transformers` and `torch`. Importing a ~4.4 s
dependency for one constant made every `soup` invocation pay for the training
stack, including commands that never touch a model.

Nothing produced wrong results; this was purely startup latency. The light core
(`pip install soup-cli` without the `[train]` extra) was never broken — it fell
back cleanly when torch was absent, just slowly when it was present.

Added `tests/test_cli_startup_is_light.py`, which asserts the invariant at
runtime (`import soup_cli.cli` must not put `torch` in `sys.modules`) instead of
inspecting source text for `import torch`, which is what the previous guards did
and why this went unnoticed.

## [0.72.1] - 2026-07-27

**Fixed — layer-streaming adapters were saved in an unloadable form.** If you
trained with `stream_layers: true` on v0.72.0, the adapter that run wrote is
**inert**: every tensor was saved under a key containing an extra `.inner.`
segment, so `soup merge`, `soup serve`, `soup chat` and
`PeftModel.from_pretrained` all loaded **zero** adapter tensors and silently
returned the untuned base model. PEFT emitted only a `UserWarning`, so nothing
failed and nothing looked wrong.

The training itself was correct — the streamed run's numerics are unaffected,
and v0.72.0's bit-exactness results still stand. Only the saved file was
affected.

**If you have a v0.72.0 streamed adapter: re-save or re-run it on v0.72.1.**
There is no way to recover the original file's association with the base model
beyond renaming its keys; re-running is the reliable path. A quick check —
if `adapter_model.safetensors` contains keys with `.inner.` in them, it is
affected:

```bash
python -c "from safetensors.torch import load_file; \
print([k for k in load_file('adapter_model.safetensors') if '.inner.' in k][:3])"
```

Streamed adapters now save byte-for-byte in the same layout as an ordinary LoRA
run, and are portable to any tool that has never heard of layer streaming.

**Also fixed — `--hf-resume` bypassed the streaming resume refusal.** The guard
only tested `--resume`, but `--hf-resume` reaches `resume_from` through a
different branch. That combination previously appeared to work by accident
(checkpoint and live model shared the same key shape); once adapters are saved
canonically it would instead have matched *nothing* and continued with a
freshly initialised adapter, silently. Both flags are now refused for streaming
runs, naming v0.72.3.

Also in this release: every "this lands in vX.Y.Z" refusal message was corrected
after the v0.72.x roadmap was renumbered (NF4 streaming is now v0.72.2; the disk
tier, wider architectures, larger batches, gradient accumulation and
checkpoint/resume are v0.72.3; preference losses are v0.72.4).

**Known limitation:** in memory the streamed model's `named_parameters()` still
carries the wrapper segment, so loading *into* a streaming run (`--resume`)
remains unsupported and is refused with a message naming v0.72.3.

## [0.72.0] - 2026-07-26

> **Superseded by v0.72.1 — adapters saved by this version load as zero
> tensors.** The entry below is left as published; the defect and the fix are
> described under [0.72.1]. Version numbers named as "upcoming" below were also
> renumbered there (NF4 is v0.72.2, not v0.72.1).

**Layer streaming (BETA) — fine-tune models that don't fit in your card.** The
frozen base lives in CPU RAM and is streamed into two pre-allocated VRAM buffers
one decoder layer at a time, so peak VRAM is bounded by the size of *one layer*
instead of the whole model. Only the LoRA adapters, their gradients and
optimizer state stay resident. Slower than resident training — but these models
did not run on the card at all.

Measured on the development box (**RTX 3050 Laptop 4 GB**, Windows 11, 16.9 GB
RAM), batch 1, gradient checkpointing on, 50 steps after 10 warm-up:

| Model | S | tok/s | GPU util | Peak VRAM |
|---|---|---|---|---|
| Qwen2.5-0.5B | 512 | 978.6 | 91.4% | 1.47 GB |
| Qwen2.5-1.5B | 512 | 525.0 | 96.8% | 1.82 GB |
| Qwen2.5-1.5B | 1024 | 487.6 | 96.7% | 2.96 GB |
| Qwen2.5-3B | 512 | 143.1 | 79.3% | **2.15 GB** |

**Qwen2.5-3B trains in 2.15 GB on a 4 GB card where a resident run OOMs.** The
honest cost: **1.43× slower than resident**, measured at 0.5B — the only
apples-to-apples comparison available on this box, because 1.5B and above cannot
run resident here at all.

### Added

- **`training.stream_layers: true`** — stream the frozen base layer-by-layer
  from CPU RAM. `training.stream_source` (`auto`/`ram`/`disk`) and
  `training.stream_buffers` (2–8, default 2 = double buffering) tune it.
- `soup_cli/utils/layer_stream.py` — tier choice, pinned-vs-pageable decision,
  architecture allowlist, VRAM/throughput arithmetic (no torch import).
- `soup_cli/utils/layer_shard.py` — rewrites an HF checkpoint into one
  safetensors shard per decoder layer, one tensor at a time, so sharding a model
  that does not fit never needs it to fit.
- `soup_cli/utils/layer_stream_runtime.py` — buffer pool, CPU-RAM weight source,
  prefetch scheduler on a dedicated CUDA stream, and the streamed layer wrapper.
- Shards are cached under `~/.soup/layer-stream/` (override with
  `SOUP_LAYER_STREAM_CACHE_DIR`) and keyed to the source checkpoint's
  fingerprint, so a base retrained in place re-shards instead of silently
  training against stale weights.
- When the base cannot be page-locked, the RAM store falls back to pageable
  memory **and says so**, including the measured cost (GPU utilisation
  ~97% → ~79%).

### Changed

- The pre-flight hardware-fit gate is skipped for streaming runs: it models a
  resident run and would otherwise refuse exactly the runs streaming enables.
- Gradient checkpointing is handled per-layer by the streamer; the HF Trainer's
  own is left off so layers are not recomputed twice.

### Known limitations

- **BETA, and proof-of-mechanism at 3B.** Nothing above 3B was measured. No
  8B/14B claim is supported.
- Scope: RAM tier, bf16, `task: sft`, Llama/Qwen, batch size 1, no gradient
  accumulation, no `--resume`. Every refusal names the release that lifts it.
- 4-bit (NF4) streaming is **v0.72.1** — NF4 weights carry a quantisation state
  and cannot be byte-copied into a plain buffer.
- The disk overflow tier, larger batches, gradient accumulation and
  checkpoint/resume are **v0.72.2**.
- The 3B number used a **pageable** store (this box cannot page-lock 5.55 GB),
  so it is a lower bound.
- `expandable_segments:True` is silently ignored on Windows; Soup detects this
  and does not claim it is active.
- Numbers are Windows/WDDM and therefore systematically pessimistic vs Linux.

## [0.71.41] - 2026-07-19

**`soup reward stress`: is your reward verifier gameable?** Turn the reward-hacking
detector on the *verifier itself*. `soup reward synth` (v0.71.40) proves a verifier
separates your references from friendly perturbations; `stress` asks the adversarial
question a reward-hacking model asks at train time — *does the verifier pay out for
degenerate junk?* It feeds empty, length-padded, repetition, and sentinel-spam
completions and flags any the verifier accepts. Pure, offline, exit 0 = robust /
2 = gameable / 1 = error. Nothing in TRL / Unsloth / Axolotl / OpenRLHF tests a
verifier for gameability.

### Added

- **`soup reward stress <reward.py|builtin> [--references golds.jsonl]`** — adversarial
  verifier probe. Attacks (`--attacks empty,length,repetition,sentinel`, `--sentinel`)
  are scored against the real gold, so numeric / tool_call / json_schema verifiers get a
  valid target and still must reject the junk. Reports a per-attack accept-rate table +
  an overall gameability verdict (`--max-gameable`, `--threshold`, `--output-report`).
  Loads the target through the existing reward loader, so it probes a synthesized `.py`
  **and** a builtin (`accuracy` / `format` / `verifiable`). A gold-requiring verifier
  probed with no `--references` is a hard error, never a false "robust".

### Fixed

- Corrected the Telemetry section in the ops docs: Soup's telemetry primitives exist but
  are **not wired to any command** — no data is ever sent today (the previous wording
  implied a live opt-in sender). Wiring is deferred until a public privacy policy ships.

## [0.71.40] - 2026-07-19

**`soup reward synth`: auto-generate a deterministic reward verifier from your data.**
Point it at a JSONL of reference (gold) outputs and it infers a verifier, emits a
readable / committable `.py` reward function, and — the moat — *refuses* to emit one
that can't tell your references from auto-generated bad answers. Nothing in
TRL / Unsloth / Axolotl / OpenRLHF synthesizes a reward; every reward today is
hand-written, hand-picked, or a trained-weights artifact.

### Added

- **`soup reward synth <references.jsonl> -o reward.py`** — deterministic verifier
  synthesis. Four families, auto-detected (or pick with `--kind`): `numeric`
  (last-number / `\boxed{}` / `####` extraction, exact or `--tolerance`), `json_schema`
  (induced keys + types + required), `regex` (positional char-classes over
  equal-length golds), `tool_call` (per-tool `required`/`allowed` argument binding).
  The emitted file is self-contained and rides `load_reward_fn`'s existing `.py`
  path — no new trusted-exec surface; you read, edit, commit, and diff it.
- **Mandatory calibration report** — the synthesized verifier is loaded back and run
  against its own references (must accept ≥90%) and auto-perturbed negatives (must
  reject). A degenerate always-accept verifier is **refused** (`exit 2`), never
  silently emitted. `--plan-only` reports the induced spec without writing;
  `--output-report` persists the calibration JSON.
- **Comma-separated `reward_fn` (`"accuracy,format"`)** now trains — it resolves to a
  reward *ensemble* (`GRPOTrainer(reward_funcs=[...])`, and unlocks the `rm_ensemble`
  reward-hack detector which needs ≥2 rewards). GRPO-only, validated at config-parse
  time. Fixes a recipe (`deepseek-v3-reasoning`) that shipped exactly this and
  previously crashed with `Unknown reward function` (#311).

### Changed / Fixed

- `training.reward_fn` gains a field validator (null-byte / blank / oversize /
  empty-comma-segment rejection) — the oldest arbitrary-code field was the least
  guarded. Comma + `verifiable` without a `verifiable_domain` now fails at parse
  time like the bare `verifiable` form.
- `envs/calculator.py` / `envs/guess_number.py` docstrings corrected: the reward is
  `reward_fn: verifiable` + `verifiable_domain: math` (the bare `reward_fn='math'`
  they showed was never valid).

## [0.71.39] - 2026-07-19

**"CI for weights, not prompts": close the evidence loop.** `soup ship`'s verdict is
now something Soup can emit, commit, review, and bind to the exact model that
produced it — turning the `soup ci init` gate from "edit two numbers in a JSON file"
into a reproducible, provenance-bound check that renders on every PR.

### Added

- **`soup ship --emit-evidence <path>`** — re-serialises the verdict into the
  `--evidence` INPUT schema, so a run's output is replayable as input: feeding it
  back through `--evidence` (same `--forgetting-threshold`) reproduces an identical
  verdict. Output is finally input.
- **`ShipConfig` under `eval.ship` in `soup.yaml` + `soup ship --config soup.yaml`** —
  commit the gate policy (`task_eval` / `task_mode` / `general_suite` /
  `forgetting_threshold` / `judge_model` / `baseline`) so the verdict is reviewable in
  a PR diff and reproducible. An explicit CLI flag always wins (CLI > config > default).
- **`soup ship --push owner/repo#N`** — post the verdict as a GitHub PR comment
  (reuses the `soup adapters pr --push` `gh api` plumbing). Best-effort: a missing
  token / `gh` failure warns but never flips the SHIP / DON'T-SHIP exit code.
- **Evidence provenance + staleness gate.** With `--emit-evidence`, `--config` STAMPS
  a `provenance` block (`config_sha` — a semantic, order-insensitive recipe hash — plus
  `base_model` and a best-effort `data_sha`) onto the evidence. With `--evidence`
  alone, `--config` GATES: it refuses (exit 3) evidence whose `config_sha` drifted
  from the committed config, so a PR that changed `soup.yaml` but forgot to recompute
  its evidence is caught. The gate policy (`eval.ship`) is EXCLUDED from the hash, so
  tuning `forgetting_threshold` never falsely invalidates evidence about an unchanged
  model.
- **`soup ci init --config <soup.yaml>`** — binds the generated gate's `soup ship`
  step to the committed config (provenance/staleness enforcement in CI).

### Changed

- **Exit-code note:** `soup ship --config` usage / staleness errors are exit `3`
  (usage), preserving `0 = SHIP`, `2 = DON'T SHIP`, `1 = runtime` from v0.71.38.

### Security

- `provenance.config_sha` read from untrusted evidence is shape-validated as a hex
  digest before being echoed (a raw value could smuggle terminal ESC bytes past
  `rich.markup.escape`).
- `provenance.data_sha` hashes the training file through an `O_NOFOLLOW` fd with a
  symlink-rejecting containment check and an 8 GiB cap (was an unguarded `hash_file`).
- `soup ci init`'s path validation now rejects `#` and the YAML 1.1 line breaks
  (NEL / LS / PS), closing a plain-scalar comment-truncation of the generated `run:`
  step (also hardens the pre-existing `--data` / `--suite` / `--evidence` args).

## [0.71.38] - 2026-07-17

**`soup ship`'s regression leg now has teeth.** Leg 2 (the catastrophic-forgetting
/ regression gate that carries the whole SHIP / DON'T-SHIP claim) was 15
hand-written trivia prompts scored by case-insensitive **substring** containment
— it scored `"B"` for "**B**erlin", `"ok"` for "lo**ok**", `"3"` for "1**3**", and
had **zero** items for tool-calling, safety, or JSON validity. This release makes
the gate real: a fixed, extraction-based scorer + bundled, offline, zero-dep eval
suites that catch a regression the old gate waved through.

### Changed

- **Fixed answer scorer (breaking — verdicts can change).** `soup ship`'s leg-2
  MCQ / instruction / arithmetic answers are now scored by answer-**extraction**
  + a boundary-aware match, replacing the raw substring test. A spurious
  substring inside another word ("Berlin", "look", "13") no longer scores a
  correct answer, so an existing run's verdict may flip — intentionally, because
  the old gate was reporting false negatives.
- **Bundled, offline general suite.** The default `--general-suite` is now seven
  hand-authored suites shipped in the wheel: `mini_mmlu` / `mini_common_sense` /
  `mini_instruction` (expanded), a new `mini_arithmetic`, and three behavioural
  suites the old gate had no coverage for — `mini_tool_call` (function-calling),
  `mini_format_json` (JSON validity), and `mini_safety` (refusal-rate). Each is
  scored to a per-model absolute score by the pure scorers Soup already ships
  (`eval/custom`, `utils/diagnose`); no lm-eval, no network, no download. Every
  suite is large enough that a single-item flip (1/N < 0.05) trips the default
  threshold instead of being rounded away.
- **`soup ship` exit codes: usage errors moved 2 → 3.** Exit `2` now means only
  DON'T-SHIP; a typo'd flag or bad `--general-suite` exits `3` (mirroring
  `soup plan` / `soup env check`), so CI can tell a config error from a caught
  regression. Offline `--evidence` read/parse errors stay `1`. **Breaking** for
  anyone parsing exit `2` as "usage error".

### Fixed

- `soup ship` help + docstrings no longer describe `--task-mode pairwise` as
  "reserved for a later release" (it shipped in v0.71.31); the dead
  `SUPPORTED_TASK_MODES` gate is removed.
- `soup diagnose`'s package docstring said "Six" probes (there are seven —
  citation) and pointed live loading at an unshipped version; it now re-exports
  all seven `score_*` probe functions so callers need not reach into submodules.

## [0.71.37] - 2026-07-17

**Every `pip install soup-cli[extra]` command now works on Windows `cmd.exe`**, and
eval-gate benchmark tasks run instead of always failing.

### Fixed

- **Install hints are now quoted so they work in every shell.** Soup printed
  `pip install 'soup-cli[ui]'` — bash / zsh / PowerShell syntax. `cmd.exe` has no
  single-quote quoting, so it hands the quotes to pip verbatim and pip refuses:

  ```
  ERROR: Invalid requirement: "'soup-cli[train]'": Expected package name at the start of dependency specifier
  ```

  Every hint, README command, and docs example now uses `pip install
  "soup-cli[extra]"`, which works in cmd, PowerShell, bash, and zsh alike — the
  same spelling the repo already used for `pip install -e ".[dev]"`. Measured on
  Windows: single quotes fail only on `cmd.exe`; double quotes pass everywhere;
  dropping the quotes passes on Windows but breaks zsh, which globs the bracket.

  Nothing in Soup can rescue the command after it is typed — pip and the shell
  own it, and Soup is not installed yet when the README command runs — so the
  fix is the spelling we print. A regression test now scans the package and every
  docs code block for the single-quoted form.

  If you followed an older tutorial and hit `Invalid requirement`, swap the `'`
  for `"`; nothing is wrong with the package.

- **Eval-gate `type: benchmark` tasks now actually run.** `eval/gate.py` probed
  for a `forgetting.run_mini_benchmark` helper that never existed, so every
  `type: benchmark` task in a gate suite failed 100% of the time — while
  advising an `[eval]` extras install that could not fix it. The gate now calls
  `ForgettingDetector` directly (the same way `soup ship` already did), and an
  unknown benchmark name fails with the list of valid names. Thanks
  [@Sanjays2402](https://github.com/Sanjays2402)!
  ([#315](https://github.com/MakazhanAlpamys/Soup/pull/315), closes
  [#310](https://github.com/MakazhanAlpamys/Soup/issues/310))

## [0.71.36] - 2026-07-16

**Data Moat II** — a semantic layer over your training data, plus two tools for
what a fine-tune forgets and leaks.

### Added

- **`soup data dedup --semantic`** — near-duplicate removal over embedding
  cosine instead of MinHash shingling. Catches *reworded* duplicates that
  MinHash misses (measured: 0.88–0.91 cosine on rewordings MinHash scored as
  distinct) while correctly keeping distinct-but-similar instructions. Zero new
  dependencies: uses `transformers` from the `[train]` extra. **Read the
  known-limitation below before lowering `--threshold`.**
- **`soup data topics <data>`** — cluster a dataset and label each cluster with
  c-TF-IDF terms, plus a coverage table (`82% code · 6% math`) and a warning for
  thin topics. Labels are *emergent term clusters*, not a fixed taxonomy.
- **`soup data canary insert|check`** — Secret-Sharer memorization probe. Insert
  K high-entropy secrets, then check any model/adapter: each secret's loss is
  ranked against never-inserted controls drawn from the same space. Exit 2 on
  MAJOR so CI can gate. Measured on SmolLM2-135M: a memorized set lands at
  percentile 0.0 (loss 1.7–2.5) against a clean model's 4.1–6.2.
- **`soup train --replay old.jsonl --replay-ratio 0.1`** — continual-learning
  rehearsal. Interleaves a seeded sample of an old dataset into training so a
  new task does not erase the old one. `r` is the fraction of the **final**
  mixed set (`n_replay = round(r/(1-r) · n_new)`), rows are interleaved rather
  than appended, and an undersized pool reports the shortfall instead of
  repeating rows. Mixed into `train` only — validation stays pure new-task.

### Fixed

- **The hardware-fit gate refused to train any local checkpoint.** A merged
  model (`soup merge -o ./mymodel`) has no size marker in its name, so the size
  guesser returned its 7B default, predicted ~16 GB of VRAM and refused. Local
  checkpoints are now *measured* from their safetensors header (0.135B actual vs
  7.0B guessed) — this had blocked `soup merge` → train-from-merged entirely.
  Third instance of this class after the v0.71.32 (Whisper) and v0.71.33
  (`M` suffix) fixes.
- **`pip install 'soup-cli[extra]'` hints printed without the extra.** Rich ate
  the bracket, so every "install the missing dependency" message across 17 sites
  told users to run `pip install 'soup-cli'` — which succeeds and still leaves
  the feature broken. Affected `[eval]`, `[data]`, `[serve]`, `[ui]`, `[tui]`,
  `[compile]`, `[mcp]`, `[carbon]` and others, including Typer help text.
- Replay rows bypassed the image/audio path-traversal validation that the
  primary dataset receives.

### Known limitations

- **Semantic dedup is not a paraphrase detector.** Measured with
  all-MiniLM-L6-v2, paraphrase cosines (0.49–0.76) **overlap** with
  genuinely-distinct rows (0.54–0.76): "Add two numbers" vs "Multiply two
  numbers" scores 0.759, *higher* than the true paraphrase "reverse a string" /
  "invert the order of characters" at 0.491. **No threshold separates them**, so
  lowering `--threshold` to chase paraphrases deletes real training rows. The
  default (0.8) is deliberately conservative.
- **Replay is validated at proof-of-mechanism scale.** On SmolLM2-135M + LoRA,
  replay retained the old task 7% better than a no-replay control — the correct
  direction — but forgetting without it was only +4%, i.e. mild. The effect size
  at full fine-tuning or 7B+ is unproven on a 4 GB box.
- **Canary exposure is the sampled-control approximation**, not full-space rank
  enumeration. "No exposure" is not proof of no memorization.
- **`data topics` / `dedup --semantic` require `[train]`** (torch) and download
  an embedding model. Plain MinHash `dedup` stays on the light core.
- Replay v1 is `sft`/`pretrain` only and is incompatible with
  `packing`/`multipack`.

## [0.71.35] - 2026-07-15

### Added
- **Compliance templates — `soup init --template hipaa|soc2|eu-ai-act|sr-11-7`.**
  Four regulation-shaped starting configs. Soup's compliance controls are CLI
  flags/commands rather than config keys, so each template is a valid training
  config plus header comments naming the exact commands for that regime
  (PHI scrubbing + air-gap for HIPAA, BOM/attest/sign for SOC 2, Annex XI +
  energy tracking for the EU AI Act, repro-receipt + diagnose/ship for SR 11-7).
  Templates default to a license-clean Apache-2.0 base.
- **`soup card <registry-id> -o MODELCARD.md` — model-card autogen.** Turns a
  Local Model Registry entry into a publishable, provenance-carrying HF model
  card: base model, training config, eval scorecard, config/data hashes,
  lineage (ancestors) and a table of every registered artifact. Adapter vs
  full-model is inferred from registered artifacts, falling back to the training
  config (LoRA rank, with Spectrum/LISA full-FT correctly treated as dense), so
  the card sets the right `library_name` and never misreports the model type.
- **`soup push --card <registry-id>`** — render that registry-driven card and
  upload it as `README.md`, overriding the auto-generated one. A bad ref fails
  fast before any network call; HF hub only.
- **`soup ci init` — fine-tuning CI.** Writes `.github/workflows/soup-gate.yml`,
  a PR gate chaining `soup data validate` → `soup expect` →
  `soup ship --evidence` (exit 2 blocks the merge). Every interpolated path is
  validated to stay under the repo root and shell-quoted; the branch and Python
  version are regex-gated; the write is atomic, symlink-rejecting, and refuses
  to clobber an existing workflow without `--force`.
- **Compliance quickstart** — a new [docs/compliance.md](docs/compliance.md)
  walkthrough: template → PII scrub → train with receipt/Annex XI/energy →
  registry → BOM + attestation → scan/sign/verify → air-gap → model card → CI gate.

### Fixed
- **GGUF export now actually works on Windows** (validated end-to-end against a
  locally-built llama.cpp: SmolLM2-135M → q4_0 / q4_k_m / q8_0 / f16 → `soup deploy
  ollama` → live inference). Four real bugs, each of which independently broke the
  path:
  - **`soup export --format gguf` cloned llama.cpp into your current directory.**
    `SOUP_DIR` is the bare name `.soup`, but the lookup used it relatively rather
    than anchoring to `~` like the rest of the codebase — so the canonical
    `~/.soup/llama.cpp` was never found and a fresh ~200 MB checkout was dropped
    into whatever directory you ran from.
  - **The first GGUF export downgraded your PyTorch and broke CUDA.** The auto-clone
    ran `pip install -r <llama.cpp>/requirements.txt` into your interpreter, and
    llama.cpp pins `torch~=2.2.1` against the CPU wheel index (observed:
    torch 2.5.1+cu → 2.2.2+cpu, transformers 4.57 → 4.46). Soup now installs only
    the convert script's extra dependencies, unpinned, and never touches torch.
  - **A correctly-built llama.cpp was not found on Windows.** MSVC (like Xcode) is a
    multi-config generator and emits `build/bin/Release/llama-quantize.exe`; only the
    flat single-config layout was searched.
  - **`soup deploy ollama` failed on a relative GGUF path** with "pull model manifest:
    file does not exist" — Ollama resolves `FROM` against the Modelfile's directory,
    and Soup writes the Modelfile to a temp dir. The Modelfile now emits an absolute path.
- **Model-card injection hardening (affects the pre-existing `soup push` card too).**
  The `## Training` section interpolated `base` / `task` / `scheduler` / `recipe`
  unescaped. Since `SoupConfig.base` and `scheduler` have no charset validator, a
  crafted-but-valid config could smuggle raw HTML — or a backtick breaking out of
  the surrounding code span — into a card published to the Hub. All values now go
  through the markdown escaper, which additionally neutralises backticks and
  strips C0/ESC control bytes.

## [0.71.34] - 2026-07-15

### Added
- **`soup adapters arithmetic` — task-vector algebra over LoRA adapters (add / scale / negate).**
  Apply task arithmetic (arXiv:2212.04089) to LoRA deltas via an expression such as
  `"coder + 0.5*math - toxic"`, mapping names to adapter dirs with repeatable
  `--adapter name=path`. Produces one merged adapter you can serve or merge.
  - Signed, un-normalized element-wise combine over same-rank adapters; the effective
    delta `ΔW = B @ A` scales **linearly** with each coefficient (negation flips the
    delta, `0.5·` halves it) via a √|c| factor split — not the `c²` a naive sum gives.
    Mixed-rank inputs are refused with a clear "harmonize rank" message.
  - Reuses the backdoor-scan gate (refuses a FAIL-scanned input unless `--allow-unscanned`)
    and a same-base-model check (`--allow-cross-base` to override). Hand-written expression
    parser (no `eval`), cwd-contained/symlink-rejecting paths, exit 0 = ok / 1 = refusal.
- **LISA — Layerwise Importance Sampled AdamW (arXiv:2403.17919).** Full-fine-tuning
  quality at LoRA-like memory: every N steps LISA freezes all decoder layers except a
  small random set (embeddings + head always trainable). Enable with
  `training.lisa_enabled: true` (+ `lisa_num_layers`, `lisa_interval_steps`) on a
  `task: sft`, transformers, text, `quantization: none` run; mutually exclusive with
  LoRA features and the other freeze mechanisms. Live on a 4 GB GPU for small models.

## [0.71.33] - 2026-07-13

### Added
- **`soup draft` — train and, above all, MEASURE a speculative-decoding draft model.**
  - `soup draft measure --target <m> --draft <d> --prompts p.jsonl` reports a draft's
    **acceptance rate** (the fraction of the target's own greedy tokens the draft
    would have proposed correctly) plus **real plain-vs-assisted throughput**. This is
    the honest gate: it tells you whether speculative decoding is worth enabling
    *before* you ship it. Exit 0 / 2 (below `--min-acceptance`, for CI) / 1.
  - `soup draft distill --target <tuned> --draft-base <tiny> --data d.jsonl -o draft/`
    distils your target into the tiny base (logit KD via the existing `task: distill`
    trainer) and emits a **dense** draft model, ready to load as an `assistant_model`.
  - `soup draft list`, plus a local draft registry (`~/.soup/drafts.json`) that
    `soup serve --auto-spec` consults **before** the built-in pairing table — so a
    draft you trained yourself is picked up automatically.
  - Draft and target must share a tokenizer; a mismatched pair is refused up front
    (speculative decoding proposes draft token ids into the target's vocabulary, so a
    mismatch silently produces garbage rather than failing).

  **Read the measured results before you use this** — see *Known limitations* below.
  On the validated pair, distillation did **not** improve acceptance, and speculative
  decoding was a net **slowdown**. `soup draft measure` is what tells you that.

### Known limitations
- **Distilling a draft did not improve its acceptance rate on the validated pair.**
  Measured on `SmolLM2-360M-Instruct` (target) with a `SmolLM2-135M-Instruct` draft:
  the *stock* draft already scored **69.3%**, and distilling it moved that to 69.7%
  after 2 epochs and back to **69.3%** after 10 epochs — i.e. no gain beyond noise.
  A small same-family draft is already near its capacity ceiling for agreeing with
  the target, and logit KD cannot buy capacity it does not have. Whether distillation
  materially raises acceptance for a genuinely diverged fine-tune, or a larger
  target/draft pair, is **unproven** on a 4 GB box — tracked as a scale issue.
- **Speculative decoding was a net slowdown on that pair** (measured 0.55–0.64×): the
  draft's forward pass costs more than the tokens it saves at this size.
  `soup draft measure` reports this truthfully rather than assuming a speedup — which
  is precisely the point of shipping the measurement.
- **Acceptance is teacher-forced greedy agreement**, the metric the Medusa/EAGLE
  papers report. It is exact and deterministic, and it is the right number for
  comparing drafts — but it is not the accepted-token count of a *sampling* run, which
  also depends on the rejection-resample cascade.
- **Same-tokenizer only.** Cross-tokenizer drafts (ULD-aligned + universal assisted
  decoding) are deferred.

### Changed
- **PRM-guided GRPO scores completions in a single batched forward.**
  `PRMScorer.__call__` now right-pads all completions into one `[B, T]` tensor
  (+ attention mask) and runs one `output_hidden_states` pass instead of one
  forward per completion, cutting per-step reward latency on non-tiny models.
  Numerically identical to the per-completion path (parity + mixed-length tests).
  Closes #298
  ([#301](https://github.com/MakazhanAlpamys/Soup/pull/301) by [@Ekaanksh-dev](https://github.com/Ekaanksh-dev)).

### Fixed
- **The hardware-fit gate no longer refuses to train small models.**
  `model_size_from_name` did not understand an `M` (millions) suffix, so
  `SmolLM2-135M` fell through to the 7B default, was predicted to need ~14 GB of
  weights, and was blocked. Every draft-sized model hit this. `1.7B` was also being
  read as `7B` (the `7b` marker matched inside `1.7b`), while `Qwen2.5-7B-Instruct-1M`
  correctly stays 7B (1M is the *context*, not the parameter count).
- **`soup serve --backend vllm` no longer force-enables `trust_remote_code`.** The
  vLLM path now goes through the same `--trust-remote-code` default-deny gate (and
  warning panel) as the transformers backend, so serving an untrusted repo never
  executes its code silently.
- **Multi-adapter serving (`soup serve --adapters name=path`) now actually switches
  adapters.** The named adapters are loaded into the model and selected per request
  (via `POST /v1/adapters/activate/{name}` or the request `adapter` field);
  previously every request silently ran the startup model. The base model is served
  when no adapter is selected.
- **Vision datasets reject out-of-directory image paths.** `llava` / `sharegpt4v`
  rows are containment-checked against `image_dir` (mirroring the audio loader), so a
  crafted `{"image": "/etc/passwd"}` row can no longer read arbitrary local files.
- **`soup train --dry-run --gpus N` no longer launches a real multi-GPU run.** The
  accelerate re-exec is skipped under `--dry-run`. The re-exec also now forwards
  `--minillm-on-policy`, `--capture-activations`, and `--capture-prompts` (previously
  dropped on multi-GPU runs).
- **MLX SFT now builds a real optimizer** (`AdamW` from the configured LR) instead of
  passing `optimizer=None`, which left the model untrained.
- **`soup data inspect` / `preview` / `search` escape dataset- and Hub-derived text**
  so a stray `[/]` no longer crashes the command and a crafted `[link=…]` tag can't
  render a phishing hyperlink. `soup runs` list/show escape config-derived fields too.
- **`soup infer --task asr` hardening** — an oversized reference no longer crashes the
  whole batch after transcription (that row's metric is skipped); an all-skipped run
  exits non-zero instead of reporting success; `--asr-task` is validated upfront; and
  dataset-derived filenames are control-stripped before printing. ASR training now
  caps transcript labels to Whisper's decoder limit, warns on >30 s audio, and picks
  fp16 on pre-Ampere GPUs instead of hardcoding bf16.
- **Knowledge-distillation KD term aligns with the CE term.** The token-level
  divergence is now computed over causal-shifted positions, so the distillation signal
  covers exactly the trained tokens (previously off by one).
- **Miscellaneous robustness** — `load_config_from_string` raises `ValueError` (not
  `TypeError`) on a non-mapping YAML document; the Web UI Bearer-token check is
  constant-time; and `soup doctor --vscode` / the LR-finder report use the centralised
  atomic, symlink-rejecting writer.

## [0.71.32] - 2026-07-07

### Added
- **ASR fine-tuning (`task='asr'`, Whisper)** — fine-tune Whisper on your accent
  or domain, locally. whisper-tiny (39M) / base (74M) train on a 4 GB GPU.
  - New `AsrTrainerWrapper` (HF `Seq2SeqTrainer` + `WhisperProcessor`); data rows
    are `{"audio": <path>, "text": <transcript>}` under the new `data.format='asr'`.
    Audio decodes via the hardened `load_audio_mono` (16 kHz mono, soundfile
    pre-probe + `O_NOFOLLOW` + symlink/size guards); transcripts become decoder
    labels (pad → −100, decoder-start token stripped).
  - Optional LoRA on q/v attention projections via `training.asr_lora: true`
    (default full fine-tune); `training.asr_language` / `training.asr_task`
    (`transcribe`|`translate`) set the decoder prefix and are persisted to an
    `asr_generation.json` sidecar so inference restores them.
  - **`soup infer --task asr`** — transcribe an `{"audio": path[, "text": ref]}`
    JSONL; reports per-row and corpus WER/CER when references are present. Loads a
    full model or a PEFT/LoRA adapter dir. Flags: `--asr-language`, `--asr-task`,
    `--audio-dir` (audio paths are cwd/dir-contained; UNC/traversal rejected).
  - New pure-python `utils/asr_metrics.py` — WER / CER / `word_accuracy`
    (= 1 − WER, for a higher-is-better ship metric leg) / `corpus_wer`, with a
    light Whisper-style text normalizer (no new dependency).
  - Recipes: `whisper-tiny-asr`, `whisper-base-asr` (live-trainable),
    `whisper-large-v3-asr` (parse-only, needs a larger GPU), plus
    `smolvlm-256m-sft` (vision). Catalog 138 → 142.

### Fixed
- `model_size_from_name` now knows Whisper checkpoint sizes (tiny…large), so the
  hardware-fit gate no longer mistakes a 39M whisper-tiny for the 7B default and
  blocks ASR training on consumer GPUs.

## [0.71.31] - 2026-07-06

### Added
- **Judge-in-the-loop suite** — put an LLM judge in the loop across the workflow:
  - **`task='online_dpo'`** — Online DPO training (wraps TRL `OnlineDPOTrainer`):
    the model generates two completions per prompt on-policy each step and a
    *judge* (a pairwise LLM judge over the existing ollama/openai-compatible
    backend) OR a `reward_model` picks the winner. Config:
    `training.online_dpo_judge: "ollama://model"` (or set `reward_model` —
    exactly one), `online_dpo_loss_type: sigmoid|ipo`, `online_dpo_max_new_tokens`;
    `beta` reuses `dpo_beta`. Transformers + text only. Recipe:
    `online-dpo-smollm2-135m`. Adapts to the installed TRL: on trl 0.19.x the
    judge is a swap-debiased *pairwise* comparison; on trl 1.x (which removed
    pairwise judges) the same `JudgeEvaluator` is used as a *pointwise*
    reward function — a documented per-version behaviour difference.
  - **`soup data best-of-n`** — Best-of-N rejection sampling (BOND-lite): sample
    N completions from `--base` locally, a `--judge` scores each pointwise, and
    the winner is written as an SFT chat row (with provenance). `--emit-pairs`
    also writes winner-vs-loser DPO pairs.
  - **`soup data evolve`** — Evol-Instruct instruction evolution (WizardLM depth
    / breadth) over an ollama/vllm provider, completing the synthetic-data suite
    (Magpie / Forge / Persona / evolve).
  - **`soup ship --task-mode pairwise`** — a true pairwise judge win-rate as the
    ship leg-1 task-win (base = 0.5 coin-flip, tuned = its win-rate; swap-debiased),
    fusing with the catastrophic-forgetting guard into one SHIP / DON'T-SHIP verdict.

### Security
- `soup data best-of-n` / `evolve` write outputs via atomic `mkstemp` + `os.replace`
  (re-validated cwd containment), closing the TOCTOU symlink-swap window between the
  containment check and the write. All judge/provider URLs are SSRF-validated; model
  loads probe `trust_remote_code`.

## [0.71.30] - 2026-07-05

### Added
- **PRM-guided GRPO** — use a trained Process Reward Model as the *per-step*
  reward inside GRPO (the o1-era process-supervision signal). Set
  `training.prm_reward: <PRM dir|id>` (a model produced by `soup train`
  `task=prm`) and `training.prm_aggregate: min|prod|last`; the PRM splits each
  generated completion into reasoning steps, scores every step with its reward
  head, and folds the per-step scores into one scalar reward that GRPO
  optimises. The PRM reward *replaces* `reward_fn` and rides the existing
  reward-shaping + reward-hack-mitigation seam, so the v0.71.26 controller
  observes it. Cross-validators gate `task='grpo'` + `backend='transformers'` +
  `modality='text'`. Default aggregation is `min` (weakest-link); `prod`
  assumes calibrated `[0,1]` step scores.
- **Bundled rollout environments** — three pure-Python toy environments
  (`soup_cli.envs.calculator` / `retrieval_qa` / `guess_number`) exposing a
  `rollout(prompts)` entry point so the live `openenv` GRPO rollout path runs
  out-of-the-box: `training.rollout_backend=openenv` +
  `training.rollout_func=soup_cli.envs.calculator:rollout`. Three ready-made
  recipes added (`grpo-env-calculator` / `grpo-env-retrieval-qa` /
  `grpo-env-guess-number`); catalog 134 → 137.

### Fixed
- **`soup train task=prm` producer conformance** (surfaced by the v0.71.30 live
  smoke): the PRM trainer now casts its reward head to the base-model dtype
  (bf16 CUDA runs previously crashed on the first `compute_loss`), saves the
  tokenizer alongside the model (so a PRM checkpoint is loadable standalone),
  and returns the standard trainer-result shape (previously the CLI crashed with
  a `KeyError: 'initial_loss'` right after saving).

### Notes
- Proof-of-mechanism only: validated on a tiny model (SmolLM2-135M) with a tiny
  synthetic PRM and synthetic reward — not a production reward-model claim
  (scale ask tracked in #286). The bundled environments are deterministic
  single-shot *seeders*, not interactive multi-turn model-in-the-loop episodes
  (the live `openenv` contract passes only prompts). Step split is a newline
  heuristic; PRM completions are scored one forward pass each.

## [0.71.29] - 2026-07-05

### Added
- **`soup shrink`** — depth-prune a model + optional distill-heal
  ("The Unreasonable Ineffectiveness of the Deeper Layers", arXiv:2403.17887).
  Ranks decoder layers by the angular distance of the residual stream across a
  contiguous block over a calibration set, drops the least-important block
  (first and last layer always protected), optionally *heals* by distilling the
  original model into the pruned student, and emits a single dense smaller model
  plus a one-screen **SHIP / DON'T SHIP** perplexity verdict.
  `soup shrink --model <id|path> [--drop-ratio 0.25 | --drop-layers N] --calib
  <calib.jsonl> [--heal <heal.jsonl> --heal-steps N] [--tolerance 0.10]
  [-o <dir>] [--device cpu] [--attach-to-registry <id>] [--plan-only]`.
  Exit codes: 0 = SHIP, 2 = DON'T SHIP, 1 = error. The heal runs as an isolated
  `soup train` subprocess (LoRA-student logit distillation) and the adapter is
  fused back so the shipped artifact stays a single dense model. Arch allowlist
  v1: Llama / Qwen / SmolLM. Validated live on SmolLM2-135M (drop 25 %: 30 -> 22
  layers, ppl x2.98 unhealed; drop 4 + heal: ppl x1.35, recovered).

### Security
- `soup shrink` contains `--calib` / `--heal` / `--output-dir` (and every
  derived write path: `<out>/model`, `<out>/heal_adapter`, the fuse staging dir)
  under cwd with `os.path.realpath` + `commonpath` + O_NOFOLLOW + symlink
  rejection, re-validating derived paths right before each write (TOCTOU). The
  heal subprocess uses an argv list (no shell) with a timeout; its config is
  schema-validated before spawn; subprocess output is C0/ESC-stripped before it
  reaches the terminal. `--model` defaults `trust_remote_code=False` with a
  probe + warn.

## [0.71.28] - 2026-07-04

### Added
- **MCP server (`soup mcp serve`)** — drive Soup from any Model Context Protocol
  client (Claude Code / Cursor / Cline / Continue) over stdio. No fine-tuning
  CLI ships an MCP server. Exposes 14 read-only tools as JSON — `advise`,
  `data_inspect` / `data_validate` / `data_score` / `data_doctor`,
  `recipes_search` / `recipes_show`, `runs_list` / `runs_show`,
  `registry_list` / `registry_show`, `profile`, `diagnose_evidence`,
  `ship_evidence` — plus two **plan-only** mutating tools (`train_start`,
  `export`) gated behind `--allow-mutating` (they render the exact command that
  would run; they never execute). The official `mcp` SDK is behind a new
  `[mcp]` extra (`pip install 'soup-cli[mcp]'`), lazy-imported so the core CLI
  stays light. Security: stdio-only (no network listener); every path argument
  re-enters cwd-containment + symlink rejection; output is control-char
  sanitized; errors are path-free; string / size / int bounds enforced.

### Fixed
- The DPO / IPO / KTO / BCO trainers now apply configured vocabulary expansion
  (`data.add_new_tokens` / `data.new_special_tokens`) via the shared
  `apply_vocab_expansion()` helper during setup, consistent with the SFT path —
  they previously ignored it. Closes #292
  ([#293](https://github.com/MakazhanAlpamys/Soup/pull/293) by [@CODING-DARSH](https://github.com/CODING-DARSH)).
- The ORPO / SimPO / GRPO trainers now also apply configured vocabulary
  expansion via the shared `apply_vocab_expansion()` helper — completing
  consistent vocab-expansion behavior across every SFT/preference/RL trainer.
  Closes #294
  ([#295](https://github.com/MakazhanAlpamys/Soup/pull/295) by [@CODING-DARSH](https://github.com/CODING-DARSH)).

## [0.71.27] - 2026-07-03

### Added
- **Fine-tune Doctor** — kill the top *silent* fine-tune failures before a
  single training step; no competitor (Unsloth/Axolotl/LlamaFactory) ships
  any of these three:
  - `soup data doctor <data> --model <id|path>` — chat-template
    compatibility report over 8 checks: `chat_template` present,
    `template_render`s cleanly, has `{% generation %}` markers,
    `eos_in_labels` (the **#1 "model never stops generating" bug** — every
    assistant turn's trained span must actually contain an EOS/EOT token;
    checks every turn, not just the last), `bos_duplication` (template +
    tokenizer both prepending BOS), `system_role` support (Mistral-style
    templates reject a leading system turn), `unknown_roles`, and
    `truncation_risk` (p95 rendered length vs `max_length`). Same OK / MINOR
    / MAJOR taxonomy as `soup diagnose`; exit 0 = OK/MINOR, exit 2 = MAJOR.
    `--train-on-responses-only` / `--train-on-messages-with-train-field`
    select the same masking strategy `soup train` would use, so the report
    and `--show-mask` never disagree about what's actually trained.
  - `soup data doctor ... --show-mask N` — render N sample rows with
    per-token trained/masked colouring through the REAL collator path
    (answer-only / per-message-train-field / RAFT span-mask) — not a
    reimplementation — so an assistant-mask bug is visible instantly.
  - `soup data lint <data>` — preference-data linter for
    dpo/orpo/simpo/ipo/bco/kto: `length_bias` (chosen systematically longer
    than rejected — the **#1 silent DPO degradation**, reported as a
    Cohen's d effect size), `label_imbalance` (KTO desirable:undesirable
    ratio), `near_duplicates` (MinHash/LSH, reuses the `soup data dedup`
    kernel), `identical_pairs` (chosen == rejected — zero preference
    signal), and `prompt_leak` (the prompt echoed verbatim inside the
    completion — a common synthetic-data pipeline bug). Optional `--model`
    for exact token-length bias (default: word count).
  - Validated live against the real `HuggingFaceTB/SmolLM2-135M-Instruct`
    tokenizer on Windows + RTX 3050 — this smoke pass found and fixed two
    genuine bugs beyond what synthetic fixtures alone caught: an EOS check
    that required the EOS token to be the *literal last* trained token
    (real templates often have a trailing formatting token after the
    closing tag that stays inside the trained span), and two call sites
    that only caught `(ValueError, TypeError)` around a tokenizer's
    `apply_chat_template` when a real Jinja `raise_exception()`
    (Mistral-style no-system-role guard) raises
    `jinja2.exceptions.TemplateError`.

### Fixed
- Harden `commands/diagnose.py`'s `--evidence` loader against a TOCTOU
  symlink swap: opens with `O_NOFOLLOW` and size-checks the open fd via
  `os.fstat` instead of `os.path.getsize` on the path before the open —
  backports the hardened loader shipped for `soup ship` in v0.71.25 (closes
  v0.71.25 known-limitation (4)).
- Harden judge-model URL validation against a hostname prefix bypass
  (`http://localhost.attacker.com`) — `GateTask._valid_judge_url` /
  `_parse_judge_url` now use `urllib.parse.urlparse` + hostname checks instead
  of `startswith`. Closes #283
  ([#288](https://github.com/MakazhanAlpamys/Soup/pull/288) by [@CODING-DARSH](https://github.com/CODING-DARSH)).
- `SFTTrainerWrapper` now applies configured vocabulary expansion
  (`data.add_new_tokens` / `data.new_special_tokens`) and resizes the model
  embeddings during initialization — previously these fields were accepted by
  the schema but silently ignored. Closes #289
  ([#287](https://github.com/MakazhanAlpamys/Soup/pull/287) by [@CODING-DARSH](https://github.com/CODING-DARSH)).
- Vision and audio SFT paths now apply that same configured vocabulary
  expansion (`data.add_new_tokens` / `data.new_special_tokens`) via the shared
  `apply_vocab_expansion()` helper, consistent with the text SFT path — they
  previously ignored it. Closes #290
  ([#291](https://github.com/MakazhanAlpamys/Soup/pull/291) by [@CODING-DARSH](https://github.com/CODING-DARSH)).

### Security
- `soup data doctor` strips C0 control characters (keeping tab/newline/CR)
  from dataset-derived content before it reaches the terminal — Rich's
  `markup.escape()` only neutralises `[...]` tag syntax, not raw escape
  sequences, so an untrusted training row (e.g. an unknown `role` field, or
  `--show-mask`'s decoded token text on a byte-level BPE tokenizer) could
  otherwise carry a literal ESC byte through to the terminal (title-bar /
  OSC-8 link spoofing, or obscuring a MAJOR verdict via cursor tricks).
  `--output` JSON is unaffected (`json.dumps` already escapes control
  characters).

## [0.71.26] - 2026-07-01

### Added
- **Closed-loop reward-hacking auto-mitigation.** The trainer now *detects*
  reward hacking mid-run and *self-corrects* — instead of only halting. Set
  `training.reward_hack_mitigation` (or `soup train --reward-hack-mitigation`)
  to one of four modes on a GRPO/PPO run (requires `reward_hack_detector`):
  - `log_only` — instrument only: append a per-step `mitigation_log.jsonl`
    (drop_pct, verdict, reward mean/std, completion-length trend, repetition)
    and *never* touch training.
  - `kl_control` — a reversible **bang-bang + hysteresis** controller: when the
    hacking signal trips, raise the KL coefficient β (geometric, clamped to
    `[floor, ceil]`, never crossing 0); relax it when the signal recovers.
    Dwell + release-patience prevent flapping; a multi-signal vote combines the
    detector drop with a length-trend and repetition signal.
  - `pid_lagrangian` — a **PID-Lagrangian** controller (Stooke et al.) that
    holds the hacking signal at a target, plus an **escalation ladder**:
    raise β → roll back to the last-good RL checkpoint → early-stop.
  - Anti-gaming hardening: per-signal EMA/median smoothing,
    conservative-on-disagreement voting, a reward-distribution-drift guard, and
    optional bounded **reward shaping** on the gamed proxy (length / repetition
    / sentinel). A plain-English give-up explanation is logged on early-stop.
  - **Proof-of-mechanism only** (see Known Limitations): validated on
    SmolLM2-135M + a synthetic length-hacking task on a single RTX 3050 — all
    four stages pass live, including a real mid-run rollback. PPO ships **BETA**
    (mechanism unit-tested; the on-GPU proof is GRPO-only).
- Ready-made `qwen2.5-coder-7b-sft` recipe for `Qwen/Qwen2.5-Coder-7B-Instruct`
  (catalog 133 → 134)
  ([#285](https://github.com/MakazhanAlpamys/Soup/pull/285) by [@Deadpool2000](https://github.com/Deadpool2000)).

### Security
- `RLCheckpointCallback.restore_checkpoint` / `save_checkpoint` refuse a
  **symlinked** `optimizer.pt` — `torch.load(weights_only=False)` on an
  attacker-placed symlink in a shared checkpoint dir was an RCE vector.
- Bool-before-int/float guards on every new `reward_hack_*` numeric field;
  `reward_hack_signals` bounded (`max_length=4`); the mitigation log writer is
  cwd-contained with symlink-reject-on-rotate and secret redaction.

## [0.71.25] - 2026-06-27

### Added
- **`soup ship` — the SHIP / DON'T-SHIP verdict.** After fine-tuning, answer one
  question: did the model get better, or did I break it? `soup ship` fuses two
  checks into a single binary decision — **leg 1**: the task metric *strictly*
  improved (base → tuned); AND **leg 2**: no general benchmark regressed past a
  forgetting threshold (default 0.05 absolute points). It SHIPs only when both
  hold — otherwise DON'T SHIP, *even if the task metric looks great*. The output
  is a one-screen verdict + the reason, with CI-gateable exit codes
  (0 = SHIP, 2 = DON'T SHIP, 1 = runtime error).
  - Leg-1 modes: `--task-mode metric` (eval accuracy) or `judge_score`
    (LLM-as-a-judge); pairwise win-rate is planned for a later release.
  - Leg-2 suite: built-in mini benchmarks by default (offline, CPU), or
    `--general-suite <names>` to route lm-eval benchmarks; `--baseline
    registry://… | file.json` supplies recorded base scores.
  - `--evidence ev.json` decides offline from pre-computed scores (no model
    load); `--output verdict.json` persists the machine-readable verdict.
- Friendlier error messages: the CUDA-OOM hint now also suggests
  `gradient_checkpointing` and `4bit` quantization, plus new mappings for
  Hugging Face gated repos (`huggingface-cli login` / `HF_TOKEN`) and
  `trust_remote_code` errors. Closes #272
  ([#282](https://github.com/MakazhanAlpamys/Soup/pull/282) by [@Akshaya-reddy18](https://github.com/Akshaya-reddy18)).

### Security
- `soup ship` input hardening: `--evidence` is opened with `O_NOFOLLOW` + an
  fstat size cap (16 MiB) under cwd containment; `--task-eval` is cwd-contained
  and symlink-rejected; `--judge-model` is validated by scheme/host via
  `urlparse` (blocks the `http://localhost.attacker.com` prefix bypass);
  lm-eval model ids reject `,`/`=` injection; `--general-suite` is bounded
  (≤ 50 names, ≤ 256 chars each).

## [0.71.24] - 2026-06-21

### Added
- **2026 model-family recipe expansion (catalog 116 → 133).** 17 new ready-made
  SFT recipes for the open-weight models released Feb–Jun 2026, each with its
  Hugging Face repo-ID verified to resolve:
  - **Qwen 3.5 (Apache-2.0):** `qwen3.5-0.8b-sft`, `qwen3.5-2b-sft`,
    `qwen3.5-4b-sft`, `qwen3.5-9b-sft`, `qwen3.5-27b-sft`, and the
    `qwen3.5-35b-a3b-sft` / `qwen3.5-122b-a10b-sft` / `qwen3.5-397b-a17b-sft`
    MoE sizes.
  - **Qwen 3.6 (Apache-2.0):** `qwen3.6-27b-sft`, `qwen3.6-35b-a3b-sft`.
  - **DeepSeek-V4 (MIT):** `deepseek-v4-flash-sft`, `deepseek-v4-pro-sft`.
  - **GLM (MIT):** `glm-5.1-sft`.
  - **Kimi (Modified MIT):** `kimi-k2.5-sft`, `kimi-k2.6-sft`.
  - **MiniMax (MiniMax Community License — commercial use needs a separate
    agreement):** `minimax-m3-sft`.
  - **Mistral Large 3 (Apache-2.0, 675B/41B-active multimodal MoE):**
    `mistral-large-3-sft`.
- Unit-test coverage for the `warmup.py` auto-warmup-steps helper
  ([#274](https://github.com/MakazhanAlpamys/Soup/pull/274) by [@shatakshi-1404](https://github.com/shatakshi-1404)).

### Fixed
- **Stale recipe repo-ID:** `glm-5-sft` now points at `zai-org/GLM-5` (the org
  migrated from `THUDM`).

## [0.71.23] - 2026-06-12

### Added
- **Native Spectrum targeted training (closes #266).** A new `soup spectrum
  scan` reads a model's `.safetensors` shards **one tensor at a time** (no
  model load — peak RAM is the largest single weight matrix), computes a
  singular-value SNR per weight matrix with a Marchenko-Pastur noise threshold
  (arXiv:2406.06623), ranks layers within each module-type group and prints
  the top `--top-percent` as a ready-to-paste `training.unfrozen_parameters`
  YAML block. This lets you scan even a very large model's layer SNR on a CPU
  box and then full-fine-tune only the high-signal layers.
  - `soup spectrum scan --model <id|path> --top-percent 50 [--modules mlp,attn|all] [--output patch.yaml]`
    — SNR table + the YAML patch; results cache at `~/.soup/spectrum/<slug>.json`
    (override via `SOUP_SPECTRUM_CACHE_DIR`).
  - New schema field `training.unfrozen_parameters: list[str]` — regex patterns
    of parameter names to keep trainable; the SFT trainer freezes every
    parameter then unfreezes the matched set (full fine-tuning, LoRA off).
    Mutually exclusive with LoRA features / `freeze_layers` / `freeze_ratio` /
    `train_router_only` / `expand_layers`; requires `task=sft`,
    `backend=transformers`, `modality=text`, and `quantization=none`.
  - The SNR kernel is pure-numpy and transpose-invariant (singular values are
    identical for `W` and `W.T`); GPT-2 `Conv1D` naming
    (`c_attn`/`c_fc`/`c_proj`) is recognised alongside Llama-style names.
  - The existing `spectrum` trainer-plugin wrapper is untouched (back-compat).
    LISA (per-step layer sampling) is tracked separately in #267.

### Security
- `soup spectrum scan` validates `unfrozen_parameters` patterns at parse time:
  rejects nested-unbounded-quantifier regexes (ReDoS), null bytes, empties, and
  caps count (50k) and length (512). Hub downloads route through the
  SSRF-hardened, namespace-pinned `hubs.snapshot_download`; symlinked shards and
  matrices above a 2^31-element SVD cap are skipped; `--output` stays under cwd.

## [0.71.22] - 2026-06-10

### Added
- **Perf & measure polish** — a 4-issue patch tightening four live paths from
  the recent BETA lifts. Pure code, validated on Windows + RTX 3050.
  - **MiniLLM on-policy KV-cache (closes #263).** The on-policy distillation
    rollout (`soup train` with `training.minillm_on_policy: true`) now threads
    `past_key_values` so each step forwards only the new token instead of
    re-feeding the whole prefix — resolving the O(L²) per-step cost from
    v0.71.18. A LoRA student (the common distill case) activates the cache
    too: the new `_supports_kv_cache` probe unwraps the PEFT model via
    `get_base_model()` before deciding. The teacher is always cached; the
    student cache respects the retained autograd graph and degrades gracefully
    if a model returns no cache mid-loop.
  - **`soup serve --mole` KV-cache (closes #262).** Each of the N task adapters
    in a served MoLE now keeps its own KV cache in lockstep, created fresh per
    `generate()` call (never stored on the instance, so there is no
    cross-request leak). Top-k zero-weight adapters are still skipped, and the
    output is byte-identical to the no-cache path on a real MoLE.
  - **Deploy-autopilot live measure factories (closes #143).** `soup deploy
    autopilot --measure` ships a first-party transformers loader factory (lazy
    import, per-candidate quant config via the Quant Menu loader; `before` =
    base, `after` = quantised) replacing the inject-only test hooks. The
    baseline is now scored **once** and the whole candidate list is
    **pre-validated up front**, so a typo in `--measure-candidates` raises
    before any model load instead of burning N live loads or doubling peak
    VRAM.
  - **Live-codec TTS via SNAC, partial (#265-partial).** The live-codec
    encode path (`data.format='audio'`) is validated for **Orpheus**:
    `load_audio_mono` now probes `soundfile.info` (duration + byte cap)
    *before* `soundfile.read` (no multi-GB decode into RAM) and reads through
    an `O_NOFOLLOW` file descriptor; a real SNAC-backed encode of a 24 kHz wav
    produced 42 Orpheus codec tokens.

### Fixed
- MiniLLM on-policy KV-cache was silently disabled for LoRA students (the
  PEFT wrapper hid the base model's `past_key_values` support) — now probed
  via `get_base_model()`.
- Deploy-measure no longer re-scores the baseline once per candidate or burns
  live model loads on a bad candidate (per-candidate validation moved up front).
- `load_audio_mono` capped audio duration only *after* decoding into RAM —
  the cap is now checked from `soundfile.info` before reading.

### Known limitations
- KV-cache correctness is validated (cache == no-cache equality on real tiny
  artifacts) but large-model throughput gains were not measured on the 4 GB
  dev box.
- **#265 stays open** — the live-codec `data.format='audio'` SNAC encode path
  is validated for Orpheus only; the other four TTS families keep their
  per-family codec dependency gate.
- The deploy-measure first-party factory's real quantized (bitsandbytes 4-bit)
  load is CUDA + bitsandbytes-gated; on Windows / no-bnb the injected test
  seams are the validated path.
- The MoLE serve KV-cache assumes single-sequence (`B == 1`) decode.

## [0.71.21] - 2026-06-10

### Added
- **Precision & rollout lift (BETA, hw-gated)** — lifts five deferred
  `NotImplementedError` stubs to live code.
  - **FP8 attention + NVFP4 (closes #141).** `training.fp8_attention: true`
    now converts the model's attention projections (q/k/v/o + fused qkv
    variants) to FP8 training modules via torchao's
    `convert_to_float8_training` with an attention-only `module_filter_fn`
    (Hopper SM ≥ 9.0 gate); `training.nvfp4: true` quantises via torchao's
    `NVFP4Config` (Blackwell SM ≥ 10.0 gate). Both are wired into the v0.28
    speed/memory pipeline and degrade to a visible yellow advisory when the
    gate fires — a conversion failing partway raises an honest "model may be
    PARTIALLY converted" error rather than silently training on a
    half-converted model.
  - **vLLM sleep mode (closes #124).** `training.vllm_sleep_mode: true` is
    live: `create_vllm_engine(sleep_mode=True)` sets
    `AsyncEngineArgs.enable_sleep_mode` (vLLM ≥ 0.7 gate with a friendly
    upgrade message), the new `vllm_sleep_cycle(engine, level=1|2)` context
    manager wraps the optimisation step (wake in `finally`), and the GRPO
    trainer threads the flag into TRL's `GRPOConfig` when the installed TRL
    exposes the hook (advisory otherwise).
  - **Multi-turn agent rollout launchers (closes #125).** `soup train` with
    `task: grpo` + `training.rollout_backend: openenv` +
    `training.rollout_func: my_module:fn` now runs a LIVE rollout: the
    resolver imports the operator's callable (same trusted-code policy as
    `data.prompt_strategy`), feeds it the dataset prompts as seeds, and the
    returned `{prompt, answer?}` rows replace the prompt dataset. Rows are
    normalised (extra keys stripped, message-list prompts deep-copied,
    non-string answers rejected loudly). `art` / `ruler` / `nemo_gym` raise a
    friendly ImportError when the backend package is missing and an honest
    BETA gate when present (injectable `_EXTERNAL_ROLLOUT_RUNNERS` seam).
    Validated by a real GRPO + openenv rollout train on SmolLM2-135M.
  - **Apple-adapter conversion (closes #228).** `soup apple-adapter` is live
    for `hf-to-mlx` / `mlx-to-hf`: PEFT LoRA safetensors ↔ mlx-lm adapters
    with both matrices transposed (`lora_A [r,in]` ↔ `lora_a [in,r]`),
    bf16 sources upcast via the torch loader, `adapters.safetensors` +
    `num_layers` emitted for mlx-lm's `load_adapters`, rank/alpha/dropout
    carried through, legacy `adapters.npz` still read, optional v0.60
    Merkle-root signing. The `*-to-apple` directions stay upstream-gated
    (no published FoundationModels adapter spec). Validated by a real bf16
    PEFT adapter round-tripping with numeric equality.
  - **Llama-4 expert delinearization (closes #97).** `soup
    delinearize-llama4` now runs a live torch runtime: fused 2-D expert
    tensors `[E*dim_in, dim_out]` reshape to 3-D `[E, dim_in, dim_out]`
    (expert count from `config.json` or `--num-experts`), other tensors pass
    through, JSON sidecars are copied, writes are atomic. `--plan-only`
    keeps the old render-and-exit flow.

### Fixed
- `safetensors.numpy.save` silently mangles non-contiguous (transposed)
  arrays — the apple-adapter writer now makes every array C-contiguous
  first (caught by the new round-trip assertions).

### Known limitations
- fp8_attention / nvfp4 / vllm_sleep_mode are BETA hardware-gated — the
  converters and gates ship validated via capability probes and fake-module
  dispatch tests, but end-to-end runs need a Hopper/Blackwell GPU + torchao
  (or vLLM ≥ 0.7), none of which exist on the maintainer's RTX 3050 /
  Windows box. The `art` / `ruler` / `nemo_gym` rollout adapters are
  honestly BETA-gated until validated against the upstream packages.

## [0.71.20] - 2026-06-09

### Added
- **Modality II trainers — TTS / BitNet / MoE expert quant (BETA, hw-gated)**
  — lifts three v0.52.0 schema-only `NotImplementedError` stubs to real code.
  - **TTS fine-tuning** (closes #131). `soup train` with `task='tts'` +
    `modality='audio_out'` now routes to a live `TTSTrainerWrapper`. TTS
    families (Orpheus / Sesame-CSM / Llasa / Spark / Oute) are decoder
    language models, so a TTS fine-tune is next-token cross-entropy over
    interleaved `[text][audio-codec-token]` chat sequences — the wrapper
    reuses the SFT path and adds per-family emotion-control templating
    (Orpheus / Oute) and registration of operator-supplied codec special
    tokens (`data.new_special_tokens`) with an embedding resize. The
    **pre-encoded chat workflow** (codec tokens produced offline, then trained
    with `data.format=chat`) is the live, validated path; the **live-codec
    workflow** (`data.format='audio'`, encode raw audio at train time) needs
    the family's heavyweight codec dependency (SNAC / BiCodec / XCodec2 / …)
    and is hardware/dependency-gated with a friendly per-family `RuntimeError`.
    Verified end-to-end on SmolLM2-135M-Instruct.
  - **BitNet 1.58-bit** (closes #134). `build_bitnet_trainer` returns a live
    `BitNetTrainerWrapper` that gates on the upstream `onebitllms` package
    (absent → friendly `RuntimeError` naming it). `soup export --format
    bitnet | tq1_0` now runs a real llama.cpp TQ1_0 ternary export (reuses the
    v0.53.1 gguf convert→quantize pipeline) instead of the deferred panel; it
    requires a built llama.cpp toolchain (friendly `FileNotFoundError` when
    absent).
  - **MoE expert quant + router-only training** (closes #136).
    `apply_moe_expert_quant` detects fused-MoE expert `nn.Linear` blocks and
    replaces them with bitsandbytes `Linear4bit` (`nf4`) / `Linear8bitLt`
    (`int8_rowwise`), leaving attention + the router in full precision; it
    runs **before** `get_peft_model` (QLoRA-on-experts) so PEFT attaches to the
    quantized base. `train_router_only` freezes every expert and keeps the
    gating router trainable, applied after LoRA. CUDA-gated (friendly
    `RuntimeError` when bitsandbytes/CUDA absent). Validated live on an
    RTX 3050: 8 expert Linears → 8 `Linear4bit` with dequant error 0.0155 vs
    source (weights genuinely carried), router-only freeze, and device-aware
    placement.

### Known limitations
- The TTS live-codec workflow, BitNet 1.58 training (`onebitllms`), and BitNet
  GGUF export (llama.cpp) are hardware/dependency-gated — the friendly gates
  ship and the plumbing is validated, but the end-to-end runs against real TTS
  models + audio codecs / a BitNet base + onebitllms / a built llama.cpp
  toolchain stay open infra-blocked items on the maintainer's RTX 3050 / Windows
  box.

## [0.71.19] - 2026-06-09

### Added
- **Quant Menu for vision / audio modality** (closes #81). The Quant Menu
  (`gptq` / `awq` / `hqq:Nbit` / `aqlm` / `eetq` / `mxfp4` / `fp8`) was rejected
  by the config modality gate for `modality in {vision, audio}` — those paths
  carried inline `BitsAndBytesConfig` blocks that handled only `4bit` / `8bit`.
  v0.71.19 drops the gate (the mlx-backend gate is retained) and threads the
  unified `build_quantization_config_for_loader` through
  `_setup_vision_transformers` / `_setup_audio_transformers`, so multi-modal SFT
  can train a LoRA on top of any pre-quantized base. The `4bit` / `8bit` config
  shapes are byte-for-byte the same as the old inline blocks; `mxfp4` still
  routes through `prepare_model_for_kbit_training`. Verified: the unified loader
  returns the right config object for every format on both modalities, and
  `_setup_vision_transformers` threads a `GPTQConfig` into
  `AutoModelForVision2Seq.from_pretrained`.

### Fixed
- **Multipack DataLoader sharding under FSDP / DeepSpeed ZeRO / DDP** (closes
  #80). The multipack `get_train_dataloader` override built a raw `DataLoader`
  and returned it directly, so under distribution every rank trained on the
  **same** packed bins (no data sharding). It now routes the loader through
  `accelerator.prepare(...)` when `num_processes > 1` — exactly what HF Trainer's
  own `get_train_dataloader` does — so accelerate's `BatchSamplerShard`
  round-robins whole bins across ranks (preserving the FFD packing) and
  equalises per-rank batch counts. The single-process path is unchanged
  (byte-for-byte the validated v0.40.4 raw-DataLoader behaviour). Verified live:
  a single-GPU multipack SFT on SmolLM2-135M trains end-to-end (RTX 3050). Full
  multi-GPU validation remains a QA item (no multi-GPU box); the distributed
  routing is mocked-tested.

## [0.71.18] - 2026-06-08

### Added
- **MiniLLM true on-policy rollout** (closes #257). `training.minillm_on_policy:
  true` (with `minillm_enabled: true`) replaces the offline distribution blend
  with the real on-policy procedure of Gu et al. 2024 §3.1: each step samples a
  fresh autoregressive rollout from the per-token mixture
  `ratio·teacher + (1-ratio)·student`, then accumulates the length-normalised
  reverse-KL `KL(student || teacher)` on the full distributions (differentiable
  w.r.t. the student only; sampled tokens are detached). New
  `training.minillm_rollout_length` knob ([1, 512]; auto-derives
  `min(max_length, 32)` when unset — the loop re-forwards the full prefix each
  step, so keep it small). Verified live: on-policy distill on tiny-gpt2
  (student + frozen teacher), finite loss, end-to-end train.
- **Cross-tokenizer ULD with token-sequence alignment** (closes #258). New
  `training.uld_strategy: wasserstein_aligned` handles **fully-disjoint**
  tokenizers (not just a vocab-size mismatch): per batch element the student and
  teacher token sequences are aligned over their decoded character spans
  (offset-overlap when both decode to the same text, difflib Ratcliff-Obershelp
  char matching otherwise), the teacher logits are mean-pooled onto the student
  positions, and the existing sorted-Wasserstein-1 surrogate is applied.
  Verified live: aligned distill with a GPT-2 BPE student + a Llama SentencePiece
  teacher, finite loss, end-to-end train.
- **`soup agent eval --sandbox`** (closes #110). Each heuristic-passing tool-call
  prediction is now *executed* against a generated mock of the endpoint in the
  v0.25.0 RLVR `code_exec` sandbox and classified into ok / tool_error / timeout
  / arg_error. The endpoint path, its required path params, and the predicted
  arguments are base64-embedded as **data** (no code interpolation). Strong
  isolation (RLIMIT / namespaces / sandbox-exec) is POSIX-only; on Windows the
  subprocess + 5 s timeout + 10 KB output cap + network guard still apply (a
  friendly reduced-isolation advisory is printed). Verified live on Windows:
  4-prediction scorecard (ok=1 / tool_error=1 / arg_error=2 / timeout=0).
- **`soup train --cloud modal`** (closes #16). Render a self-contained Modal.com
  app from `soup.yaml` for serverless GPU training when you have no local GPU.
  The config YAML is base64-embedded as data (no interpolation, no secrets); the
  `--gpu` type (t4 / l4 / a10g / a100 / a100-80gb / l40s / h100) is validated
  against a closed allowlist. Default is **plan-only**: write the stub + print the
  `modal run` command. `--cloud-submit` attempts a live submit gated on a Modal
  token (`modal setup` / `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`). New
  `[modal]` extra (`pip install 'soup-cli[modal]'`; only needed for live submit —
  plan-only render needs no dependency). Verified live: real stub rendered, exit
  0.

## [0.71.17] - 2026-06-08

### Added
- **Serve-time MoLE** (closes #259). A `task='moe_lora_routing'` run now writes a
  self-describing `mole_manifest.json` next to `mole_gate.pt`, and
  `soup serve --mole <dir>` loads the base + N frozen task LoRAs + the trained
  gate and blends them **per token** at decode time (custom blend loop —
  non-streaming + streaming). `--mole` requires `--backend transformers` and is
  mutually exclusive with `--bank` / `--steer` / `--adapters` /
  `--speculative-decoding`. The base model comes from `--base` (or the manifest
  when unset). Verified live on SmolLM2-135M (2 task adapters, real generation +
  SSE streaming).
- **Per-request multi-tenant vector banks** (closes #260). `soup serve --bank`
  now resolves the active VeRA/VB-LoRA user per request via a
  `contextvars.ContextVar`, so concurrent requests on a threaded server never
  race on shared instance state. The streaming path re-selects the user inside
  the generator's own context. Verified live: two `X-User-Id` headers produce
  distinct steered outputs, an absent / unknown id self-clears to the clean
  baseline (no cross-request leak), and a repeated user is deterministic.
- **Epoch-aware RAFT document shuffle** (closes #253). `data.raft_epoch_shuffle:
  true` re-permutes the golden + distractor documents **each training epoch**
  (per-epoch salt) so the model can't latch onto one fixed citation slot.
  `epoch=0` reproduces the legacy single-permutation order exactly. Verified live
  on a 2-epoch SmolLM2-135M RAFT run.
- **`soup diagnose --citation-style` / `--shuffle-seed`** (closes #254). The live
  citation failure-mode probe now accepts the citation style (bracket / inline /
  footnote) and the RAFT shuffle seed so the golden `[doc-N]` ids line up with
  what the model saw at train time. Verified live (rows=6, mean_recall=1.000).

### Fixed
- MoLE `train()` now returns the `initial_loss` / `final_loss` / `total_steps` /
  `duration_secs` / `duration` keys the generic train handler reads, so
  `soup train task=moe_lora_routing` completes cleanly (previously raised
  `KeyError: 'initial_loss'` after writing the gate). Surfaced by the #259 smoke.

## [0.71.16] - 2026-06-07

### Added
- **Covariance-preconditioned ROME via `--cov-corpus`** (closes #250). `soup edit
  set --method rome --cov-corpus <jsonl|txt>` now estimates the key covariance
  `C = E[k kᵀ] + λI` over a stats corpus and uses the preconditioned update
  `u = C⁻¹ k*` instead of the covariance-free `C = I` path — the genuine ROME
  closed form, which spreads the rank-1 update mass to reduce collateral
  interference with other facts. Falls back to `C = I` when no corpus is given.
  The exact post-condition `down(k*) += delta` is preserved either way. The
  corpus loader is cwd-contained, symlink-rejected (O_NOFOLLOW + raw-path
  lstat), and size/line-capped; `--cov-corpus` is rejected (fail-loud) for any
  method other than `rome`. Verified on real `gpt2` (prob 0.005 → 0.9997) and
  SmolLM2-135M.
- **GPT-2 (`transformer.h` / `mlp.c_proj`) support in the edit kernels** (closes
  #251). ROME / MEMIT / AlphaEdit now edit GPT-2-family models, not just
  Llama-family. The `Conv1D` weight layout (`[in, out]`, transposed relative to
  `nn.Linear`'s `[out, in]`) gets a transpose-aware rank-1 update, AlphaEdit
  null-space projection, and MEMIT band dim-check. PEFT-wrapped GPT-2 / Llama
  models are unwrapped via `get_base_model`. Verified end-to-end on real `gpt2`.
- **Mixtral joins the LongLoRA architecture allowlist** (closes #147). A bare
  `mistral` token does not appear in `mixtral` (m-i-x vs m-i-s), so the existing
  `is_mistral_model` detector excluded the MoE variant. A dedicated
  `is_mixtral_model` helper + `MixtralAttention` entry in the S² forward-override
  regex + `_SEPARATE_QKV_FAMILIES` now cover Mixtral-8x7B / 8x22B (the attention
  is the standard separate-QKV shell; the MoE lives in the MLP).

### Fixed
- **Atomic `EditGovernor` edit-count increment** (closes #252). Two concurrent
  `soup edit set` runs on the same base model could lose an increment: each read
  the persisted count, added locally, and the last writer clobbered the first.
  `save_state` now re-reads the persisted count INSIDE the cross-process lock and
  merges this run's delta (`edit_count − persisted_baseline`), mirroring the
  v0.60.0 `namespace_pin` pattern. Verified: two governors recording 3 + 2 edits
  from the same baseline persist a merged 5 (not a clobbered 2 or a naive +1).

### Notes
- Test count: 13511 → 13595 (+84 net; +81 in `tests/test_v07116.py`).

## [0.71.15] - 2026-06-07

### Fixed
- **Iterative-DPO config render bug** (closes #261). `soup iterative-dpo`'s
  default per-round trainer rendered `output: {dir: ...}` (a mapping), which
  `SoupConfig.output` (a plain string) rejected — so the spawned `soup train`
  subprocess failed at config validation. Now renders `output: <str>`, mirroring
  the v0.71.13 #229 `local-rl` fix. A regression test captures the rendered YAML
  and validates it via `load_config_from_string`; verified end-to-end with a real
  `soup train` round on SmolLM2-135M.

### Changed
- **CMA-ES merge loads the base model once** (closes #246). `soup adapters merge
  --strategy cmaes` previously reloaded the (multi-GB) base model into a fresh
  PEFT wrapper on every candidate in the population. The default scorer now loads
  the base once and reuses it across the whole `population × generations` loop —
  each candidate only loads its small merged LoRA, applies it, generates, and
  unloads it. Verified on SmolLM2-135M: the base loads exactly once across N
  candidates.
- **`soup loop` budget gate now estimates real cost** (closes #245). The
  pre-wired loop's per-iteration cost estimate was a hard `0.0` placeholder, so
  the dollar budget gate never tripped. It now wires v0.34 `run_cost.
  estimate_run_cost_usd` off the most-recent completed run's GPU + duration (the
  best forward signal for a repeating loop). Falls back to `0.0` on the first
  iteration / a CPU / unpriced GPU; never crashes the daemon.
- **`--diagnose-gate` is multi-node aware** (closes #170). The post-training
  diagnose gate (and the `--annex-xi` / `--repro-receipt` / capture hooks) fired
  on `LOCAL_RANK==0`, so a shared-filesystem multi-node run ran them once per
  *node*. They now gate on the global chief (`RANK==0` when `RANK` is set, else
  `LOCAL_RANK==0`) — once per *cluster*.

### Added
- **`soup train --track-energy --energy-out <path>`** (closes #244) persists the
  measured energy/CO2 reading as JSON so `soup bom emit --energy <path>` (the
  v0.71.3 #256 consumer) can attach it to an ML-BOM. Atomic + cwd-contained +
  symlink-rejected. Completes the train → BOM energy hand-off.

## [0.71.14] - 2026-06-05

### Added
- **Live FSDP shard consolidation** (closes #96). `soup merge-sharded-fsdp-weights`
  lifts the v0.44.0 plan-only stub: it now streams each `pytorch_model_fsdp_*.bin`
  shard via `torch.load(weights_only=True)` (no arbitrary pickle exec), unions the
  per-rank parameter fragments into one state-dict, and writes a single
  `.safetensors` atomically. Memory-friendly (one shard loaded at a time). New
  `--plan-only` flag prints the plan without writing. Single-process — no
  multi-GPU needed to MERGE. (Per-rank disjoint-parameter / FULL_STATE_DICT
  shards; DCP sharded-tensor reconstruction is out of scope — use
  `accelerate merge-weights` for those.)
- **Live `kv_cache_type` wiring on the transformers serve backend** (closes #140).
  `soup serve --kv-cache-type q8_0 | bf16 | f16 | fp8` lifts the v0.53.1
  `apply_kv_cache_type` `NotImplementedError` stub: `bf16`/`f16` load the model in
  that dtype (the KV cache inherits it); `q8_0` routes an 8-bit HQQ quantized KV
  cache through `model.generate` (needs `pip install hqq`); `fp8` raises a friendly
  runtime error (vLLM + Hopper-only — the transformers backend has no fp8 KV
  path). vLLM / SGLang KV-cache-dtype routing stays in the infra-blocked tail.
- **ONNX export QA verified** (closes #71) — `soup export --format onnx` exercised
  end-to-end on a tiny model: export exits 0, `model.onnx` loads in ONNX Runtime
  with `input_ids` present, and a forward pass produces a real output. Recorded in
  `tests/qa/v07114_qa.md`.

### Notes
- GGUF export (#70), AWQ/GPTQ export (#72), the CUDA + llama.cpp QA doc (#144),
  HF Hub push/Spaces deploy (#74), and the Community-QA tracking meta-issue (#79)
  remain open with `infra-blocked` labels — they need a built llama.cpp toolchain,
  `autoawq`/`auto-gptq` Windows wheels, or HF credentials the QA box lacks. See
  `tests/qa/v07114_qa.md`.

## [0.71.13] - 2026-06-04

### Added
- **Prompt-compile family — live wiring** (closes #225, #226, #227, #229). Four
  `soup` commands that shipped as deferred-stub `NotImplementedError` in v0.68.0
  are now real, validated end-to-end (real DPO train on SmolLM2-135M + real
  Ollama teacher distillation on RTX 3050).
- **`soup local-rl train` runs a real nightly DPO/KTO/ORPO train** (#229).
  `--once` harvests the latest thumbs-up/down DPO pairs from the local-RL SQLite
  and trains them via a `soup train` subprocess (argv list, no shell); a `state`
  table tracks `last_train_at` so a re-run with no new feedback skips, and a run
  with fewer than `--min-pairs` (default 10) skips. Without `--once` it renders a
  systemd `.service`/`.timer` + launchd `.plist` scheduler scaffold into
  `--scheduler-dir` for the user to install. New flags: `--once`, `--min-pairs`,
  `--output/-o`, `--scheduler-dir`, `--hour`, `--minute`.
- **`soup distill-prompt` prepares a real distillation dataset** (#226). For
  each prompt in the traces JSONL the teacher is called once via the v0.20
  provider helpers (Ollama / Anthropic / vLLM); `sft`/`kl` emit
  `{messages:[user, assistant=teacher]}` and `preference` emits
  `{prompt, chosen=teacher, rejected=student}`. New flags: `--provider`,
  `--base-url`, `--temperature`, `--max-rows`.
- **`soup compile` runs DSPy / GEPA / TextGrad prompt-program optimisation** (#225)
  and **`soup compile-tools` runs the TextGrad / GEPA tool-schema optimiser** (#227),
  both lazy-importing the optimiser libraries behind the new `[compile]` extra
  (`pip install 'soup-cli[compile]'`) with a friendly `ImportError` naming the
  extra when absent. `--plan-only` still renders the plan and exits 0.

### Security
- **systemd / launchd injection defence** (#229). `local-rl` and the scheduler
  renderers reject `\n` / `\r` in the model id and shell-quote every `ExecStart`
  argument, so a crafted model id cannot inject extra unit directives.

### Fixed
- **`local-rl` train config rendered `output` as a mapping** (#229). The nightly
  `soup train` YAML now emits `output: <dir>` (a plain string the schema accepts)
  instead of `output: {dir: <dir>}`; a regression test validates the rendered
  config against `SoupConfig`.

## [0.71.12] - 2026-06-04

### Added
- **Architecture + distillation + adapter-training — live wiring** (closes #145,
  #146, #148, #158, #84, #221, #222). Seven surfaces that shipped schema-only in
  earlier releases are now real, validated end-to-end on tiny models
  (SmolLM2-135M / a locally-built tiny Llama).
- **Sequence-level knowledge distillation is live** (#145). `task: distill` now
  accepts `distill_mode: token|sequence`; sequence mode trains the student on the
  teacher's generated continuations (cross-tokenizer-friendly hard-label KD)
  instead of per-token logit matching. `sequence` mode is mutually exclusive with
  the v0.70 cross-tokenizer ULD logit path.
- **Classifier LoRA is live** (#146). `task: classifier|reranker|cross_encoder`
  now attaches a LoRA adapter to the sequence-classification head when `lora` is
  configured, so a frozen encoder + small adapter can be trained instead of the
  full model.
- **LLaMA Pro block expansion is per-architecture** (#148). `expand_layers`
  now interleaves zero-initialised identity blocks for Llama / Qwen / Mistral
  decoder stacks (was Llama-shaped only), with `freeze_trainable_layers`
  freezing the original blocks so only the new ones train.
- **LongLoRA S² shifted-sparse attention is live** (#158). `use_longlora: true`
  now installs the shifted-sparse-attention forward override on the Q/K
  projections (Llama / Mistral / Qwen / Phi), restoring the patched forwards on
  context exit.
- **Mixture-of-Depths is live** (#84). `use_mod: true` attaches a per-layer
  top-k token router (`mod_capacity_factor`) so only a subset of tokens receive
  each block's residual update. Architecture allowlist: Llama / Qwen / Mistral;
  unsupported bases warn and skip.
- **VeRA / VB-LoRA multi-tenant serving is live** (#221). `soup serve --bank
  <bank.json> [--bank-strength S]` reconstructs the shared projection + per-user
  scaling vectors and installs a decode-time forward hook; the active user is
  selected per request via the `X-User-Id` header (an unknown/absent id is a
  zero-delta no-op, so there is no cross-request leak). Serves N personas at
  ~KB-per-user instead of a full LoRA each.
- **MoLE per-token adapter routing is live** (#222). `task: moe_lora_routing`
  with `mole_task_adapters: [...]` trains a per-token gating network that blends
  N frozen task LoRAs (`mole_top_k` / `mole_temperature`); only the router
  trains. The gate is saved as `mole_gate.pt` alongside the run.

### Changed
- `apply_bank_to_serve` (#221) and `build_gating_kernel` (#222) now return live
  objects (a `LoadedVectorBank` and a `torch.nn.Module` router) instead of the
  v0.67.0 deferred-stub `NotImplementedError`.

## [0.71.11] - 2026-06-04

### Added
- **GRPO / RL callbacks — live wiring** (closes #235, #236, #237, #238, #239,
  #240, #159, #160). The reward-hacking, cross-tokenizer distillation, MiniLLM,
  mid-epoch RL checkpoint, iterative-DPO and echo-trap surfaces that shipped
  schema-only in v0.70.0 are now real, validated end-to-end on SmolLM2-135M.
- **Reward-hacking detector is live** (#235). `--reward-hack-detector
  info_rm|rm_ensemble` now installs a GRPO `TrainerCallback` that reads the
  per-step rewards (via a shared, thread-safe reward-fn capture buffer),
  computes an InfoRM cluster-separation drop (`info_rm`) or RM-ensemble
  divergence (`rm_ensemble`), classifies OK/WARN/HACK, logs the verdict to
  `state.log_history`, and halts training on HACK when `--reward-hack-halt` is
  set. `rm_ensemble` requires ≥2 reward functions.
- **Cross-tokenizer ULD distillation is live** (#236). `task: distill` with
  `--uld-strategy wasserstein|topk_align` now computes a real Wasserstein-1
  (sorted-CDF) or top-k-aligned distillation loss inside the distill trainer,
  handling student/teacher vocab-size mismatch by clamping teacher ids to the
  teacher vocab.
- **MiniLLM reverse-KL distillation is live** (#237). `--minillm-enabled` adds
  a teacher-mixed, length-normalised reverse-KL term plus an optional
  pretrain-anchor SFT term (`--minillm-pretrain-anchor-path` /
  `--minillm-pretrain-anchor-weight`) that keeps the student near coherent
  language. The anchor corpus reader is cwd-contained + symlink-rejecting with
  a per-line byte cap.
- **Mid-epoch RL checkpoint is live** (#238). `--rl-checkpoint-save-every-steps
  N` writes a real adapter + optimizer state + JSON manifest every N steps
  during PPO/GRPO and prunes to `--rl-checkpoint-keep-last`, so a long RL run
  survives a crash without losing the optimizer momentum.
- **`soup iterative-dpo` orchestrator is live** (#239). Runs the full
  sample → reward-score → build-pairs → DPO-train loop across rounds: each
  round samples completions from the previous round's adapter, the next round
  trains a fresh LoRA from the base on that round's harvested pairs.
  `--plan-only` still renders the plan without running.
- **Echo-trap detector is live** (#240). `--echo-trap-enabled` installs a GRPO
  callback that scores per-trajectory n-gram repetition, classifies
  OK/WARN/TRAP against `--echo-trap-threshold`, logs the verdict, and halts on
  TRAP when `--echo-trap-halt` is set (catches RAGEN-style degenerate
  repetition in multi-turn agent RL).
- **GRPO variant fallback now warns once** (#159). When a `--grpo-variant`
  custom `compute_loss` falls back to the base trainer (because the installed
  TRL renamed the loss inputs), the trainer logs a one-shot WARNING instead of
  silently degrading to the default objective.

### Changed
- **GRPO reference-model EMA no longer materialises full state dicts** (#160).
  `--ref-model-ema-alpha` now updates the reference model in place by iterating
  `named_parameters()` (`ref = (1-α)·ref + α·policy`), eliminating the three
  model-sized allocations per step the v0.53.11 path made. A total
  name/shape-mismatch (0 shared parameters) logs a one-shot WARNING so a
  misconfigured EMA can't silently no-op.

## [0.71.10] - 2026-06-03

### Added
- **RAG family — live wiring** (closes #199, #200, #201, #202). The four
  retrieval / steering surfaces that shipped schema-only in v0.62.0 are now
  real, validated on SmolLM2-135M.
- **RAFT span-mask training is live** (#199). `data.format: raft` rows
  (`{query, golden_doc, distractor_docs, answer}`) now train answer-only: the
  prompt span is masked to `-100` and each document is labelled `[doc-N]` so
  the model learns to cite the supporting document. Documents are shuffled
  reproducibly (`data.raft_shuffle_seed`). Rows whose prompt fills
  `max_length` (answer fully truncated) are dropped with a warning rather than
  silently shrinking the effective dataset.
- **`soup ra-dit` — one-shot two-stage orchestrator** (#200). Trains the
  retriever (stage 1, embedding/contrastive) then the generator (stage 2,
  RAFT-SFT) in a single command, recording the trained retriever as the
  generator's paired retriever. A `soup train` of a generator-stage config
  with no retriever model set now auto-links the most-recent RA-DIT retriever
  run from the Registry. `--plan-only` validates both configs without
  training; `--retriever-model` overrides the auto-link.
- **`soup steer train` / `apply` + `soup serve --steer` are live** (#201).
  Fit a CAA (contrastive activation addition), ITI (inference-time
  intervention) or RepE (representation-engineering PCA) control vector from
  `{positive, negative}` contrastive pairs, persist it as a safetensors +
  config artifact, and apply it at decode time via a forward hook
  (`soup serve --steer <name> --steer-strength <s>`).
- **`soup eval citation` + citation-span loss boost are live** (#202). Score
  citation precision / recall / F1 over `{predicted, expected_ids}` or RAFT
  rows (`--shuffle-seed` aligns the golden `[doc-N]` id with what the model
  saw at train time). When `citation_faithful: true`, bracketed `[doc-id]`
  spans in the answer get a boosted per-token loss weight. A new `citation`
  failure mode is available in `soup diagnose`.

## [0.71.9] - 2026-06-03

### Added
- **Knowledge edit + unlearn — live wiring** (closes #193, #194, #196, #197,
  #203). The v0.61.0 / v0.62.0 schema-only stubs are now live, validated on
  SmolLM2-135M.
- **`soup edit set` (ROME / MEMIT / AlphaEdit) is live** (#194). New
  `soup_cli/utils/edit_kernels.py` ships covariance-free rank-1 weight-edit
  kernels: ROME (single-layer `W += δ·kᵀ/‖k‖²`), MEMIT (residual distributed
  across a layer band), AlphaEdit (ROME update projected orthogonal to the
  down-proj's top singular direction). `apply_edit` loads the model, optimises
  the target residual, applies the rank-1 update, and optionally saves with
  cwd-containment + symlink rejection. `--output`, `--device`, `--governor/
  --no-governor` flags added. On a tiny model a ROME edit moved
  `P("Lyon" | "The capital of France is")` from 0.0016 → 0.96.
- **`soup edit diff` live before/after generation** (#194). Pass
  `--before-model` + `--after-model` (+ `--probes`) to generate completions
  through both models and surface the probes whose output changed.
- **EditGovernor SQLite persistence + cross-process locking** (#196). New
  `EditGovernorStore` (mirrors `namespace_pin.NamespacePinStore` —
  $HOME/$CWD/$TMPDIR containment, TOCTOU symlink rejection, WAL +
  busy_timeout, `fcntl`/`msvcrt` sidecar lock, POSIX 0600). `save_governor` /
  `load_governor` / `default_governor_db_path` (env override
  `SOUP_EDIT_GOVERNOR_DB`) persist per-base-model edit-count + verdict across
  separate `soup edit set` runs.
- **`apply_edit` consults the EditGovernor automatically** (#197). When a
  governor is supplied, `check_can_edit()` runs BEFORE the model load (refusing
  on norm blowup / edit cap) and `record_edit()` runs AFTER with the measured
  Frobenius delta.
- **Live GRACE codebook** (#203). `GraceCodebook` (epsilon-ball nearest-key
  lookup), `apply_grace_edit` (captures a residual key + optimises a value +
  appends to a codebook sidecar), `save_codebook` / `load_codebook` (atomic,
  cwd-contained, symlink-rejected), `install_grace_hook` (decode-time residual
  substitution). New `edited_model` / `grace_codebook` Registry artifact kinds.
- **`soup train --task unlearn` is live (NPO / SimNPO / RMU)** (#193). New
  `soup_cli/utils/unlearn_kernels.py` (NPO `(2/β)·mean(-logσ(-β(πlp-reflp)))`,
  length-normalised SimNPO, RMU representation steering) + a self-contained
  `UnlearnTrainerWrapper` loop loading a LoRA policy, a frozen reference
  (NPO/RMU), and forget/retain JSONL datasets. NPO/SimNPO forget loss
  decreased on the tiny-model smoke. Warns when run without a retain set.

### Security
- `_save_edited_model` / `UnlearnTrainerWrapper` output dirs + `save_codebook`
  / `load_codebook` + `_load_unlearn_rows` enforce cwd-containment, raw-path
  symlink rejection (TOCTOU), null-byte rejection, and file-size / per-line
  caps. `apply_grace_edit` honours the governor for direct callers.

## [0.71.8] - 2026-06-03

### Added
- **Probes & SAE — real weights + live downloads** (closes #215, #216, #217,
  #218, #219). A new shared `soup_cli/utils/probe_kernel.py` provides the
  linear-probe math (contrast-pair derivation, apply, flag-rate, verdict bands,
  operator-supplied weight loading, deterministic synthetic fallback); every
  heavy import (`numpy` / `torch` / `safetensors`) is lazy.
- **`soup probe sleeper --weights <w.npz|.npy|.safetensors>`** (#215) — load a
  real calibrated probe direction instead of the synthetic fallback. Weights are
  cwd-contained, symlink-rejected, `O_NOFOLLOW`-opened, `allow_pickle=False`,
  and size-capped. `compute_contrast_probe(positive, negative)` derives a probe
  from contrast-pair activations.
- **`soup probe sae-diff <repo> --auto-download`** (#216) — fetch an
  allowlisted SAE from the HF Hub into `~/.soup/sae-cache/` (validated against
  `HF_HUB_ALLOWLIST` BEFORE any network call) via a new SSRF-hardened
  `soup_cli.utils.hubs.snapshot_download` (repo-id shape + home/cwd/tmp cache
  containment + namespace-pin TOFU gate).
- **`soup probe truth` / `soup probe harm`** (#217) — TruthfulQA-style honesty
  and HarmBench-style misuse activation probes (6 bundled bases each, 5% / 20%
  verdict bands, `--weights` to skip the allowlist with a real probe). The
  probe pack now ships truth + harm entries per base.
- **`soup probe interference --measure <eval_suite> --base-model <m> --adapter
  name=path ...`** (#218) — auto-measure the N×N interference matrix by actually
  loading the base + each LoRA adapter (PEFT multi-adapter), measuring loss for
  each adapter alone (diagonal) and each co-loaded pair
  (`add_weighted_adapter(combination_type="cat")`, off-diagonal). Exit 2 on a
  MAJOR worst-pair.
- **`soup train --capture-activations <layer> --capture-prompts <jsonl>`** (#219)
  — a post-training hook writes an SAE-diff-ready per-token activation snapshot
  to `<output>/activations/activations.json`. `resolve_layer_module` resolves
  the same `model.layers.N` path whether or not a LoRA adapter is loaded
  (PEFT-wrapper fallback).

### Security
- Probe / SAE / capture file I/O is cwd-contained + `O_NOFOLLOW` (TOCTOU close)
  + size-capped; SAE weight loads use `allow_pickle=False`. SAE auto-download
  validates the allowlist before any network call and rejects a glob result
  that resolves outside the snapshot dir (symlink-escape guard).

### Notes
- #215 is partial: the operator-supplied / contrast-pair / synthetic paths ship
  now, but the 6 large-base Anthropic-calibrated probe vectors remain
  upstream-gated (no public calibrated artifact exists). Documented as a known
  limitation.

## [0.71.7] - 2026-06-02

### Added
- **Eval live runners** — six probe surfaces that previously emitted heuristic
  / neutral stubs now load a real model and run live (closes #161, #162, #208,
  #211, #212, #165). New shared `soup_cli/utils/live_eval.py` provides the
  model-loading primitives (generator / multi-generator closures, masked
  cross-entropy eval-loss, a short-LoRA probe, and held-out logit agreement);
  every heavy import (`torch` / `transformers` / `peft` / `lm_eval`) is lazy.
- **`soup advise --probe-model <id>`** — runs a LIVE ROI probe: zero/few-shot
  token-F1 baselines, a short LoRA probe (relative held-out-loss improvement +
  real wall-clock), and base-model proximity (held-out logit agreement) folded
  into the dataset profile. Without `--probe-model`, `--probe` stays the offline
  heuristic.
- **`soup tunability --live`** — replaces the offline heuristic with a real
  per-candidate LoRA probe (loads each `repo_id`, trains `--probe-steps` on a
  held-out-excluded slice, reports the held-out-loss drop).
- **`soup eval capability --live --model <id>`** — invokes lm-eval-harness per
  resolved task (or a `--tasks` override) with `--limit` / `--device`, isolating
  per-task failures and surfacing a no-metric result as an explicit error.
- **`soup eval behavior --base-model <id> [--adapter <path>]`** — generates
  pre/post responses on the bundled behaviour battery and scores the live diff.
- **`soup diagnose --base-model <id> [--adapter <path>] [--dataset <jsonl>]
  [--tokenizer <id>]`** — runs all six failure-mode probes (forgetting / refusal
  / format / mode_collapse / memorization / contamination) live via
  `soup_cli.utils.diagnose.live.run_live_diagnose`; falls back to neutral OK or
  `--evidence` JSON when no model is supplied.

### Security
- The two new JSONL dataset readers (`diagnose.live._load_dataset_rows`,
  `tunability._load_jsonl_rows`) open with `O_NOFOLLOW` after the cwd-containment
  check, closing the check→open TOCTOU window (matches the v0.65 / v0.67 reader
  policy).

## [0.71.6] - 2026-06-02

### Added
- **`soup build` live runner** — the dbt-for-SFT DAG (`soup build <manifest>`) now
  *materialises* datasets instead of only dry-running the plan. Five built-in
  transforms ship live (`identity`, `drop_empty`, `lowercase`, `strip`,
  `dedup_exact`); `table` rebuilds from scratch, `view` re-derives on every run,
  and `incremental` re-transforms only the rows whose content hash changed
  (tracked in a SQLite state store, keyed by row hash **and** the model's
  transform+config fingerprint so a transform change re-runs everything). Custom
  transforms are passed per-run via the Python API's `transforms=` map. Outputs
  are written atomically; the `--output-dir` is symlink-checked before any
  directory is created.
- **`soup data gen-magpie` live generator** — the Magpie synthetic generator
  (Xu et al. 2024) now actually generates. It feeds an aligned model its
  chat-template prefix (chatml / llama3 / gemma / mistral families auto-detected)
  and harvests the self-generated user instruction + assistant response via raw
  completion. Live providers: `ollama` (`/api/generate` raw) and `vllm`
  (`/v1/completions`) — both SSRF-hardened (loopback-only HTTP); `anthropic` is
  rejected (no raw-completion endpoint). Optional `--quality-filter` drops
  low-quality rows via the v0.47 toxicity/educational scorers; exact-duplicate
  instructions are de-duplicated.
- **`soup eval irt-subset --model {1pl,2pl,3pl}`** — the IRT eval-cost optimiser
  gained 2PL (per-item discrimination) and 3PL (+guessing floor) joint
  coordinate-ascent MLE fits alongside the existing 1PL Rasch. `1pl` keeps the
  closed-form path for back-compat; `2pl`/`3pl` route through the new `fit_irt`.
- **Tokenizer-aware memorization probe** — `score_memorization(..., tokenizer=...)`
  and `split_prefix(..., tokenizer=...)` (used by `soup diagnose`) now split the
  prefix/suffix on real token-id boundaries and measure echo-overlap over
  sub-word tokens when a tokenizer (HF id / path / duck-typed object) is supplied,
  catching BPE-level memorization that whitespace tokenisation misses. Default
  (no tokenizer) keeps the whitespace behaviour.

### Fixed
- **`soup data augment --provider ollama|vllm` no longer crashes** — the command
  imported a non-existent `OllamaProvider` symbol and raised `ImportError` on
  every non-OpenAI provider. It now routes through the shared, SSRF-hardened
  provider factory; `--model` / `--base-url` are honoured, the output path is
  containment- and symlink-checked, and the write is atomic.

### Security
- **Ollama / vLLM provider URLs reject `0.0.0.0`** — `validate_ollama_url` /
  `validate_vllm_url` dropped the bind-any wildcard from their loopback allow-set
  (now `localhost` / `127.0.0.1` / `::1` only), matching the newer
  `validate_hub_endpoint` / `validate_webhook_url` SSRF validators. Reachable now
  that Magpie threads a user-supplied `--base-url` through these providers.

## [0.71.5] - 2026-06-02

### Added
- **`soup eval against` now reads eval metrics** — `ExperimentTracker.get_metric_series`
  falls back to the `eval_results` table when the metric is not a per-step
  training column (`loss` / `lr` / `grad_norm` / `speed` / `gpu_mem`). So
  `soup eval against <base> --candidate <run> --metric task_accuracy` returns a
  real score series (benchmark scores live in `eval_results`, not `metrics`)
  instead of "Empty series". Per-step columns still read from `metrics` — no
  regression for existing callers.
- **`soup advise` learns from past project outcomes** — `soup advise` now reads
  this project's accepted-verdict history (`~/.soup/advise_history.jsonl`) and
  biases the rubric: 3+ successful SFT precedents flip a marginal RAG call to
  SFT; 3+ negative GRPO outcomes suppress GRPO in favour of SFT-on-traces; an
  encouraged choice gets a small confidence nudge. Scoped per-project (one
  project's record never biases another). No history → identical to before.
- **Slack/Discord webhooks on four more commands** — `--slack-url` / `--discord-url`
  (SSRF-hardened, loopback-only HTTP, RFC1918 rejected, never crashes the
  command) now ship on `soup ingest`, `soup prune-prompt`, `soup ab` (fires only
  on a `reject_h0` / `accept_h0` decision, not `continue`), and
  `soup data active-sample` — not just `soup drift-alarm`. The validator + sender
  moved to a shared `soup_cli/utils/webhooks.py`.
- **Tokenizer-aware `soup prune-prompt`** — `--tokenizer <model_or_path>` detects
  and strips the shared system-prompt prefix on **token** boundaries instead of
  characters, so a multi-byte UTF-8 prefix can never be split mid-code-point.
  Default (no `--tokenizer`) keeps the whitespace-character behaviour.
- **Curriculum bucketing by loss percentile** — `DynamicCurriculumCallback` now
  buckets samples by the percentile rank of the live loss (or perplexity)
  signal within a rolling window when `data.curriculum_metric` is `loss` /
  `perplexity`, so a consistently-hard sample is routed to the same difficulty
  bucket across recomputes. `length` and warm-up still use round-robin.
- **`--hub` on `soup data push` and `soup data forge`** — `soup data push
  --hub modelscope|modelers` uploads a dataset via the matching SDK
  (`repo_type=dataset`, commit message sanitised); `soup data forge --hub
  <non-hf> --teacher owner/name` pre-fetches the teacher model from that hub
  (and warns when the teacher is not a repo id so `--hub` is never silently
  ignored). HF stays the default.

### Notes
- Live SaaS *pull* adapters for `soup ingest` (Langfuse / LangSmith / Helicone /
  OpenPipe / OpenAI SDKs, issue #204) remain deferred: they need credentialed
  vendor accounts with populated trace data to validate honestly. Tracked as an
  open, `infra-blocked` (external-account) item. `soup ingest` continues to parse
  the JSONL export you pull from your dashboard.

## [0.71.4] - 2026-06-02

### Added
- **Live canary verdict for `soup adapters merge`** — `--canary <suite.json>`
  scores the merged adapter against the first input and classifies
  **OK / MINOR / MAJOR** using the Quant-Lobotomy taxonomy (drop <2% OK, <5%
  MINOR, else MAJOR). `--strict-verdict` exits 2 on MAJOR. Pre-scored
  `{"baseline_scores","candidate_scores"}` suites run with no model load; a
  `{"tasks":[...]}` suite uses an injectable scorer. Replaces the v0.57 `UNKNOWN`
  stub.
- **Live evolutionary merge** — `soup adapters merge --strategy cmaes --eval
  <suite> --budget <t>` now runs the full CMA-ES loop: each candidate is merged,
  materialised, scored against the eval suite, and the best-weighted merge is
  written to `--output`. Replaces the v0.67 plan-only stub.
- **Publish an adapter PR to GitHub** — `soup adapters pr <title> --base-sha
  <hex> --adapter <path> --push owner/repo#N` posts the rendered PR Markdown as a
  GitHub PR comment via `gh api` (argv-list, body over JSON stdin; no shell).
  Token resolves from `GITHUB_TOKEN` / `GH_TOKEN`.
- **Pre-wired `soup loop` production stages** — `soup loop init --pre-wired` (or
  `soup loop watch --pre-wired`) swaps the v0.58 no-op stage stubs for real
  harvest (traces → preference pairs) → DPO train → eval-gate → canary-deploy
  callables. `soup loop status` now shows the `pre_wired` flag.
- **Loop iterations as Soup Cans + Registry lineage** — `soup loop watch
  --pack-cans` packs each successful iteration as a v0.26 Soup Can and appends a
  Registry entry (tag `loop-iter`), chaining a real lineage DAG across
  iterations visible through `soup history`. `soup loop replay <id> --extract
  <dir>` unpacks a recorded iteration.
- **Branch pointers into the Registry** — `soup adapters branch <name>
  --attach-to-registry <id>` links a branch snapshot to a Registry entry (shown
  as a `branches` node in `soup history`); `soup adapters branch <name>
  --from-registry <id>` derives a fresh snapshot's config + base from an entry.

### Security
- The backdoor-scan gate (v0.71.2 #192) and license-conflict gate (v0.60 Part E)
  now run for **all** merge strategies, including `--strategy cmaes` (previously
  bypassed because cmaes returned before the gates).
- `soup loop` canary deploy restricts `SOUP_LOOP_SERVE_ENDPOINT` to loopback /
  RFC1918-private hosts (a serve endpoint is the operator's own box/LAN), beyond
  the general webhook SSRF policy which permits any HTTPS host.
- `soup adapters pr --push` builds the `gh` child environment from an allowlist
  so unrelated secrets (`HF_TOKEN` / `OPENAI_API_KEY` / …) never reach the
  subprocess.
- The canary-suite JSON read uses `O_NOFOLLOW` + `os.fstat` (size cap enforced on
  the same fd) to close the symlink/size-cap TOCTOU window.

## [0.71.3] - 2026-06-01

### Added
- **Energy & CO2 measurement for training** — `soup train --track-energy` wraps
  the training window in a codecarbon **offline** tracker (no IP-geolocation
  network call) and reports kWh / CO2 / grid intensity, feeding those numbers
  into `--annex-xi`. New `EnergyTracker` context manager; graceful no-op when
  codecarbon is absent (`pip install soup-cli[carbon]`). `--energy-country`
  picks the ISO-3166 alpha-3 grid for the CO2 estimate (default `USA`).
- **PDF Annex XI/XII documents** — `soup train --annex-xi report.pdf` now renders
  a reportlab PDF (a `.md` path still renders markdown). `pip install
  soup-cli[pdf]`.
- **Auto-populated training-corpus domains in Annex XI/XII** — the top crawled
  domains (with shares) are now extracted from the training JSONL and listed in
  the EU AI Act docs, replacing the previous empty placeholder.
- **Soup Can manifest v3 with embedded attestations** — `soup can pack --attest
  <statement.json>` (repeatable) embeds in-toto Statements into a v3 can
  manifest; v1/v2 cans still load. Each statement is shape- and size-validated.
- **Local audit log auto-instrumentation** — every `soup` command now appends one
  HIPAA/SOC2-shaped record to `~/.soup/audit.jsonl` (secrets redacted, args
  capped). Opt out per-invocation with `--no-audit-log` or globally with
  `SOUP_NO_AUDIT_LOG=1`. Tail/rotate with `soup audit-log`.
- **Reproducibility receipt in airgap bundles** — `soup airgap-bundle
  --repro-receipt <receipt.json>` embeds an SR 11-7 receipt as
  `repro-receipt.json`; auto-detected from `<model>/repro-receipt.json` when not
  supplied.

### Security
- `soup can pack --attest` now rejects oversize attestation files by their raw
  size *before* parsing them into memory (defence against memory-exhaustion).
- The new file-loading paths (attestation JSON, airgap receipt, training-corpus
  scan, PDF write) are all cwd-contained + TOCTOU symlink-rejected and
  size-capped; the audit auto-log redacts `hf_`/`sk-`/`Bearer` tokens and never
  crashes the CLI on a broken log.

## [0.71.2] - 2026-06-01

### Added
- **ed25519 signing for `soup adapters sign` / `soup attest`** — real detached
  signatures (over the adapter Merkle root / the in-toto statement) via a new
  `[sign]` extra (`pip install soup-cli[sign]`, pulling `cryptography`).
  `soup adapters sign --backend ed25519 --key <priv.pem>` (or `--generate-key
  <out.pem>`, or `SOUP_SIGNING_KEY`); `soup adapters verify [--public-key
  <trusted.pem>]` does a cryptographic verify and, with a trusted key, genuine
  authentication. `soup attest emit --sign ed25519 --key <priv.pem>` writes a
  `<output>.sig` sidecar; new `soup attest verify <statement> --signature <sig>`
  verifies it (canonical-JSON, so it's platform/newline-independent). Sigstore
  keyless signing stays infra-blocked (needs an OIDC provider + Fulcio/Rekor
  network — can't be honestly validated offline).
- **Anti-AI-Jacking namespace pin on Hub downloads** — HF model fetches now
  consult a trust-on-first-use pin store: a repo whose author changes (or whose
  `created_at` jumps backward) is refused unless the namespace shift is explicitly
  allowed. Fails open when repo metadata is unavailable.
- **License auto-detection at `soup adapters merge`** — when `--license` isn't
  given, the license is read from each adapter's `adapter_config.json` /
  `config.json` / model-card frontmatter (HF `llama3.1`-style ids mapped to
  canonical) and the conflict gate runs automatically.
- **Backdoor-scan gate at `soup adapters merge`** — refuses to merge any input
  whose `soup adapters scan` returns FAIL (or can't be scanned) unless
  `--allow-unscanned` is passed; WARN is advisory.

### Changed
- License-conflict overrides (`--license-override <reason>`) are now recorded to
  the audit log for legal review.
- The namespace-pin store now uses SQLite WAL + busy-timeout and a cross-process
  file lock around its get+insert, so concurrent writers don't lose the trust
  anchor.

### Security
- ed25519 verification fails closed (any tamper / wrong key / missing key ⇒
  invalid). Signing keys + trusted public keys are symlink-rejected and
  size-capped via a shared reader (no cwd-containment — keys live outside the
  project). `--generate-key` refuses to overwrite any existing path.

## [0.71.1] - 2026-06-01

### Added
- `soup env fix` — render a reproducible install plan from `soup-env.lock`.
  Emits copy/paste `uv pip install` commands (`--format uv-pip`, default) or a
  `requirements.txt` body (`--format requirements`); `--output` optionally writes
  a `requirements.txt` under cwd. Print-only by design — never shells out to a
  package manager.
- `soup lock write --env-lock <path>` — auto-derive `--env-hash` from a
  `soup-env.lock` so operators who ran `soup env lock` don't copy the hash by
  hand. `--env-hash` still wins when passed explicitly.
- `soup serve --record-thumbs <db>` — capture thumbs-up/down feedback into a
  local-RL SQLite at startup, plus a new `POST /v1/thumbs` endpoint (transformers
  backend). Returns 404 when the flag isn't set.
- Judge-calibration persistence: `JudgeCalibrationReport.to_dict`,
  `write_judge_calibration`, and `load_judge_calibration`, backed by a new
  `judge_calibration` registry artifact kind. Loading re-validates the report so
  a corrupt on-disk field is rejected.
- Bundled MUSE and WMDP unlearning eval fixtures so
  `soup eval unlearning --benchmark muse|wmdp` runs out of the box. WMDP
  forget-set probes ship **redacted** (placeholder prompts + `REFUSED` responses)
  — Soup never ships verbatim hazardous content.

### Changed
- `soup completions` now introspects a cached base model's actual LoRA target
  modules (config-only `AutoConfig` load, `local_files_only=True`, never networks
  or raises) and falls back to the canonical default shape when the base isn't
  cached locally.
- `build_dag` exposes a `validate_build_source` helper (cwd-containment +
  symlink rejection) for build-manifest source paths.

## [0.71.0] - 2026-06-01

### Changed
- **Breaking — install split.** The heavy training stack (`torch`,
  `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`) moved
  out of the core install into a new `[train]` extra. `pip install soup-cli` is
  now a light CLI + data-tools install with **no PyTorch**; run
  `pip install 'soup-cli[train]'` (or `[all]`) to fine-tune. Existing users who
  train must reinstall with `[train]`. Version pins are unchanged.
- Trimmed `README.md` to a ~238-line front door; the full feature reference now
  lives under `docs/` (one topic page per area, indexed from the README).
- Raised the pytest coverage gate from 50% to 77% (`--cov-fail-under=77`).
- Migrated to a `src/` layout (`src/soup_cli/`) for cleaner packaging and to
  stop tests accidentally importing the in-tree package.

### Added
- `[train]` and `[all]` optional-dependency extras (`[all]` pulls
  `train`, `serve`, `ui`, `data`). `[dev]` self-references `[train]` so CI and
  contributors still get the full stack from `pip install -e ".[dev]"`.
- Friendly error mapping: a missing heavy dependency (`torch`, `transformers`,
  `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`) now surfaces
  "Training needs the [train] extra. Run: pip install 'soup-cli[train]'".
- `py.typed` marker (PEP 561) so downstream type checkers pick up Soup's inline
  type hints.
- `.pre-commit-config.yaml` with ruff (lint + format) and standard file-hygiene
  hooks.
- Lenient `mypy` configuration and a non-blocking `type-check` CI job.
- This `CHANGELOG.md`.

### Removed
- The historical, per-version security-fix log that had grown inside
  `SECURITY.md` (~220 KB). `SECURITY.md` is now a concise security policy; the
  detailed hardening notes remain in git history and the GitHub Releases notes.

[Unreleased]: https://github.com/MakazhanAlpamys/Soup/compare/v0.73.1...HEAD
[0.73.1]: https://github.com/MakazhanAlpamys/Soup/compare/v0.73.0...v0.73.1
[0.71.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.70.0...v0.71.0
