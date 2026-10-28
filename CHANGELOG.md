# Changelog

All notable changes to **Soup CLI** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Detailed, per-release notes for every published version live on the
[GitHub Releases page](https://github.com/MakazhanAlpamys/Soup/releases). This
file tracks unreleased changes and links out for historical detail rather than
reproducing 70+ versions of notes.

## [Unreleased]

## [0.77.0] - 2026-10-26

### Upgrade notes

Before the first run of an existing config on 0.77.0, run `soup doctor --config soup.yaml`.

- **What v0.76 warned about is refused at load.** Thirty-one staged fields set away from their
  default, `data.remove_unused_columns: true`, `training.quantization` / `load_in_8bit` on a task
  that never quantises, and `soup ab --beta` stop the load or the command; each refusal names the
  line to delete. The first entry under Changed lists them (#808).
- **Settings v0.76 accepted silently and never applied are refused too.** Each refusal names the
  setting and the fix; the entries under Fixed give the details:
  `training.use_mod: true` on a route without a Mixture-of-Depths router (#1752);
  a non-default `training.moe_aux_loss_coeff` on `backend: unsloth` or with `stream_layers` (#1749);
  `training.expand_layers` with a LoRA rank (#1783) or on `task: pretrain`, and `task: pretrain`
  with `lora.r: 0` (#1875); `training.citation_faithful` on `pretrain` (#1843);
  `training.neftune_alpha` on the 14 tasks that never forward it (#1883);
  `training.relora_steps` where the loss has a frozen reference (#1891);
  `quantization: 8bit` with `backend: unsloth` (#1946); vision or audio distillation (#2014) and
  Unsloth vision or audio SFT (#2015); `training.stream_layers` with `use_cut_ce` (#2023);
  vision row formats (`llava`, `sharegpt4v`) on preference tasks (#2064); and
  `training.lora.rank_pattern` / `alpha_pattern` with `backend: mlx` (#1996) or `backend: unsloth`
  (#2153).
- **A malformed KTO label drops its row.** A row whose label is not a boolean, a
  `true`/`false`-style string or -1, 0, 1 is dropped with a warning instead of trained (v0.76
  trained `null` as undesirable and `0.5` or `2` as desirable); a file with no row left is
  refused, and `soup data validate --min-valid-fraction 1.0` exits 2 on any such row (#2126).
- **A LoRA target on a streamed embedding or head is refused when the model is built.** Under
  `training.stream_layers`, v0.76 checked only the literal names `lm_head` and `embed_tokens` at
  load, so a regex or dotted spelling loaded (a regex then failed at its first save, or in peft
  under `task: dpo`). On an untied checkpoint any spelling is now refused before a streaming buffer
  is allocated; on a tied checkpoint, where v0.76 refused the literal names too, it now trains and
  saves (#2082).
- **Python 3.11-3.13.** `requires-python` is `>=3.11,<3.14`; pip on Python 3.10 keeps installing
  0.76.x (#1805, in #2089).
- **AWQ and GPTQ export are removed:** `soup export --format awq` / `gptq`,
  `soup quantize --to awq` / `gptq` and the `[awq]` / `[gptq]` extras. Stay on 0.76.x if you need
  them (#1951).
- **`soup serve` tool routes.** `POST /v1/tools/python` answers HTTP 501 unless the host provides
  OS-level isolation, as `POST /v1/tools/bash` already did (#2166). Both routes now require
  `--tool-auth-token` (or `SOUP_TOOL_AUTH_TOKEN`) on every bind, loopback included, and answer
  401 without it: start the server with the token and send `Authorization: Bearer <token>` with
  each request (#2174). The other routes are unchanged. The isolation cuts the network; the code
  still runs as the server's user, so use a long random token and see
  `docs/serving-and-export.md` for running the server for callers you do not fully trust.
- **`training.gradient_checkpointing` now reaches the trainer.** With the default `false`,
  `grpo`, `ipo`, `bco`, `reward_model` and `online_dpo` train with checkpointing off (TRL's own
  default used to apply), and `false` now turns Unsloth's checkpointing off; set `true` to keep
  the memory you had (#1778, #1981).
- **Settings that were ignored now apply, so results change:** `lora.use_rslora` on MLX and
  Unsloth (#1899), and `training.freeze_layers` / `freeze_ratio` with LoRA (#2145).
- **Docker:** the image is Ubuntu 24.04 with Python 3.12 on a CUDA 12.8 runtime; the host needs a
  driver that supports CUDA 12.8 (#2092).

### Added

- Add native Sesame CSM training for raw-audio transcript rows through Transformers
  CSM/Mimi, with backbone and depth-decoder losses, a frozen codec, standard LoRA or
  full fine-tuning, validation, checkpoint resume and processor saving. Pre-encoded
  CSM codec-string data and unsupported SFT options remain refused.
  Spark/Oute raw-audio support remains outside this change. (#265 by @Amix29 in #1769)

- **`soup_cli.mcp_server.registry.call_tool` runs a registry tool without an MCP server (#1761 by @Faisal-Fayaz in #1770).** The three guarantees a tool call gets — stray stdout diverted to stderr for the duration of the handler so it cannot corrupt a caller's channel, the result passed through C0/ESC/DEL sanitization, and every failure from the lookup or the handler mapped to a message carrying neither a path nor a traceback — lived in `mcp_server/server.py::_dispatch_tool`, which is private. A caller running the registry tools directly had to reimplement all three and import the private `_sanitize`, whose name could change without notice. `call_tool(name, arguments, *, allow_mutating=False)` is the helper the MCP server calls, so the two cannot drift apart; it is not promised as a stable in-process API. It returns the Python value rather than pretty-printed JSON, so a caller does not re-parse what it just called; serialization stays with the transport. `_sanitize` is public as `sanitize`, following #896, and the old private name remains as an alias so a downstream import is not broken. `allow_mutating` is a second gate for callers rather than a duplicate of the one `build_registry` applies: a table built permissively still needs the flag per call, so it cannot run a mutating tool by accident. A `name` that is not hashable raises `TypeError` rather than `McpToolError`: that is a caller bug, not a tool failure. The stdout guard is process-wide, so a threaded caller's other threads have their prints diverted for the duration of a call. No behaviour changes for `soup mcp serve`.

- **Remote log tail on Lambda Cloud training failure (`soup train --cloud lambda`).**
  When remote training on a Lambda instance exits with a non-zero status, the controller
  fetches the last lines of `/home/ubuntu/soup/train.log` over SSH (bounded to 50 lines and
  50 KB, with C0/C1 and ESC control characters stripped inline) and prints them before
  terminating the instance, preserving failure diagnostics without delaying instance cleanup.
  (#1705 by @Areen-09 in #1774)

- Live evaluation exposes `encode_prompt` and `MAX_PROMPT_TOKENS`, with shared prompt preparation for both generators; `chat_template_kwargs` and `soup ship --chat-template-kwargs` let callers pass model-specific template variables without duplicating special tokens. (#1758 reported by @abdulwaarith0, fixed by @Amix29 in #1776)

- `patch_fast_lora_single_projection` accepts an `allow_dropout` opt-in: the fast-LoRA single-projection kernel then models non-zero LoRA dropout itself (keep-mask drawn and saved, dropped adapter input rebuilt in the backward) instead of delegating the call to peft (#839 by @sloemo01 in #1784)

- **A `preference_loss_weights` blend whose largest weight is `dpo` or `ipo` now trains (#1425 by
  @ybai08 in #1785).** Both losses build trl's `DPOTrainer`, which keeps only the means of its
  log-probs, so any blend naming them stopped at the first step. The blend now takes the policy
  log-probs from the trainer's own forward pass and the reference log-probs from the trainer's
  `compute_ref_log_probs`; the `ipo` term divides each log-ratio by its response length, as trl's
  IPO loss does, so it is the loss `task: ipo` trains. The reference is run twice per step. A `dpo`
  or `ipo` term under a `simpo` / `orpo` primary trains too (see #1828 below); with full
  fine-tuning (no LoRA adapter to switch off) it still stops at the first step, and the message
  says why (those trainers run no reference) and what to do (give `dpo` or `ipo` the largest
  weight). Try: `soup train --config soup.yaml` with `task: preference` and
  `training.preference_loss_weights: {dpo: 0.7, simpo: 0.3}`.

- **Two reference pages generated from the code: `docs/config-reference.md` lists every
  `soup.yaml` key with its type, default, constraints and description, and
  `docs/cli-reference.md` lists every command, option and argument with its type, default,
  whether it is required, and its help (#1799).** `python scripts/gen_reference_docs.py` renders
  both from `SoupConfig` and the live command tree. The test suite checks that the pages still
  state what the code does (keys, types, defaults and constraints; commands and their
  parameters) and names each one that differs. Rewording a help string or a description needs
  no regeneration; `SOUP_DOCS_EXACT=1` adds the byte-exact check that a release runs.
  `CONTRIBUTING.md` says when to regenerate. Found on the way: the help of `soup eval against`
  sent the reader to `soup runs list`, a subcommand that does not exist. It now names
  `soup runs`, and a test resolves every `soup ...` invocation written in a help string or a
  field description against the live CLI.

- **A `dpo` or `ipo` term in a `preference_loss_weights` blend now also trains when the largest
  weight is `simpo` or `orpo`.** Those primaries build trl's CPO and
  ORPO trainers, which run no frozen reference, so a blend such as `{simpo: 0.6, dpo: 0.4}`
  stopped at the first step. The blend now makes the reference itself: the trainer's own forward
  once more per step, without gradients and with the LoRA adapter switched off, which is the base
  model the adapter started from. A blend that names neither `dpo` nor `ipo` runs no extra
  forward. With `init_strategy: pissa` / `olora` / `loftq` or `relora_steps` the adapter-off model
  is not the starting model, so that combination is refused before the model is loaded, with
  the reason. The terms are computed on the primary trainer's batch, and the CPO / ORPO batches start
  with the BOS token where the `DPOTrainer`'s need not, so the same `dpo` term differs a little
  between a `simpo` and a `dpo` primary; `docs/training.md` says which reference and which tokens
  each primary uses. Try: `soup train --config soup.yaml` with `task: preference` and
  `training.preference_loss_weights: {simpo: 0.6, dpo: 0.4}`. (#1425 by @ybai08 in #1828)

- **DPO runs natively on Apple Silicon: `task: dpo` with `backend: mlx` (#1846 by @dryn85-max).**
  It was refused at config load because the MLX wrapper was scaffolding. Soup now runs its own
  MLX loop with TRL's default sigmoid loss and `training.dpo_beta`. Rows are read by the same
  `format: dpo` converter as on the transformers backend and tokenized the way TRL 0.29 does,
  except that ties and rows the chat template cannot slice are skipped with a count, and per-row
  `tools` are not applied. No second model is loaded: reference log-probs are precomputed from
  the base model. `training.seed` and `training.data_seed` are honoured, and the vocabulary head
  runs in checkpointed chunks to keep long pairs in memory. Measured once on an M4 Pro,
  Qwen2.5-0.5B-Instruct trained faster than TRL on MPS with the same bf16 weights and less
  process memory. Try: `soup train --config soup.yaml` with `task: dpo`, `backend: mlx` and
  `base: mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

- **`training.stream_loss_chunk: N` runs the output head and the loss `N` tokens at a time under
  layer streaming, so the logits no longer grow with the context length.** Unset, the head builds
  the full `[batch, seq, vocab]` logits and the loss works on all of them, which is what ran a
  4 GB card out of memory from about 1024 tokens. Set, each chunk runs the head, takes its
  cross-entropy and the gradient on its own hidden states, and drops its logits; one
  `[batch, seq, hidden]` tensor carries the gradient to the backward pass. The loss and the
  gradients match the unchunked step within `1e-5` of the largest value, not bit for bit. The
  `entropy`, `mean_token_accuracy` and `num_tokens` columns and `rewind.jsonl` are written as
  before; a labelled forward returns no logits. `task: sft` only; refused without
  `stream_layers`, beside `use_cut_ce`, with `data.format: raft`, with a trainable output head,
  and for a model that changes its logits after the head (Gemma 2's soft-capping). The pre-flight
  charges the logits on `min(seq, N)` tokens plus two hidden-sized tensors. **Not measured on a
  GPU**: the peak-VRAM table #1693 asks for is still open. Try: `soup train --config soup.yaml`
  with `training.stream_layers: true` and `training.stream_loss_chunk: 128` (#1693 by @ybai08 in
  #1880).

- Layer streaming: new opt-in `training.stream_nf4_scale_cache` keeps each streamed double-quantised NF4 weight's expanded fp32 block scales on the compute device after its first use, so bitsandbytes' dequantise + linear path stops re-expanding them on every forward and checkpoint recompute (about 0.38 ms per Mistral-7B layer visit in an isolated RTX 5070 measurement). Results are bit-identical; the fused 4-bit GEMM path is unchanged. Costs 4 bytes per 64-element block of VRAM for the run (416 MiB on Mistral-7B), charged by the pre-flight. Off by default; needs `quantization: 4bit` with double quantization (#1892).

- **`soup registry push --data <file>` records a training data file's SHA-256 as the entry's `data_hash` (#1764 by @abdulwaarith0 in #1907).**
  Registering a run used to leave `data_hash` empty. The file must be under the current
  directory and is refused before anything is written if it is missing, not a regular file
  or outside it. The hash records the file registered as the training input, not proof of
  what the run trained on. `soup registry show` now prints `data_hash`, or `-` when there
  is none.

- **Layer streaming supports the DeepSeek-V3 architecture** (`model_type: deepseek_v3`: MLA
  attention, routed plus shared experts, dense first layers), gated on resident-vs-streamed parity
  forward and backward. The sharder stacks per-expert keys into the fused expert tensors, drops the
  MTP layers stored past `num_hidden_layers`, and keeps the router's `e_score_correction_bias` at
  fp32; the runtime loads that buffer from the shard (the streamed skeleton would otherwise run it
  at zero). FP8 checkpoints (`F8_*` tensors or `*_scale_inv` block scales) are refused by the
  sharder instead of being cast without their scales. Kimi K2's `kimi_k2` model type stayed
  refused in this change; #2107 (below) admits Kimi-K2-Base and Kimi-K2-Instruct. (#1914)

- A claim board, the pinned issue #1945 regenerated daily by `scripts/claim_board.py`, shows which `help wanted` issues are free, claimed or blocked (#1917)

- Native DoRA for MLX text SFT on dense unquantized and pre-quantized 4-bit linear targets, with reloadable adapter metadata, compatible-weight warm-start checks, explicit unsupported-target refusals and Apple Silicon lifecycle regression coverage (#1918 by @Amix29 in #1920).

- Task and bundled-suite eval scorers support optional batch generation with scalar fallback; live greedy generators use bounded left-padded batches, and scorer revision 6 prevents generation failures from earning credit (#1759 by @kok-o in #1958).

- **`soup registry push --idempotency-key` and `RegistryStore.push(idempotency_key=...)` make a registration safe to retry (#1763 by @abdulwaarith0 in #1964).**
  A repeated push with the same key returns the existing entry instead of creating a twin, as
  long as it repeats the entry's name, tag, base model, task, run, config and data hash; anything
  else under that key is refused with an error naming the entry. Two writers pushing the same key
  at once end up with one entry. Existing registry databases gain the nullable column in place.
  Pushes without a key are unchanged.

- Record nullable run base revisions and portable final-artifact provenance, bind supported resident Transformers base loads to one Hub commit, preserve unknown/offline/legacy flows, and transfer historical identities into registry entries without re-reading training files; keep cache-path hints explicitly unverified and training-hash capture separate (#1902 by @kok-o in #1972).

- **`training.fast_lora: true` runs a LoRA's projections through Soup's own Fast-LoRA autograd functions** (q/k/v with one shared input, gate/up/down as one SwiGLU block, every other adapted Linear as a single projection) instead of peft's modules, on the resident and the layer-streamed SFT paths. Setup prints what each kernel took over and refuses the run when none did. Off by default; `task: sft`, `transformers`, text, quantization `none` or `4bit`, and `lora.dropout: 0` (Soup's default is 0.05); refused with the reason beside `use_liger`, DoRA, VeRA, `target_parameters`, Spectrum (`unfrozen_parameters`), LISA and `quantization_aware: fp8`. Verified on CPU in float32 and float64 against peft's modules; the bf16/fp16, NF4 and streamed-CUDA checks have not been run, and no speedup is claimed. Try: `soup train --config soup.yaml` with `training.fast_lora: true` and `training.lora.dropout: 0` (#792 by @ybai08 in #1973).

- Add `soup merge --low-memory`, which merges a plain LoRA adapter into a safetensors base one
  tensor at a time instead of loading the whole model into RAM. The tensor values are identical to
  the regular merge for float32, float16 and bfloat16; adapters it cannot reproduce exactly
  (DoRA, `modules_to_save`, rank or alpha patterns, embedding adapters) are refused with a
  pointer back to the regular merge (#2006 by @kuanysh228).

- **Expert-granularity streaming for DeepSeek-V3 MoE layers.** `shard_checkpoint(...,
  expert_layout="per_expert")` writes each MoE layer's routed experts to their own file, one pair of
  matrices per expert (NF4 under `quant="nf4"`), and the streamed model reads an expert only when
  the router picks it, so the VRAM bound no longer grows with the expert count. It is bit-exact
  against transformers' eager experts on real OLMoE-1B-7B layers, forward and backward, in fp32,
  bf16 and NF4. In a measurement harness (not `soup train`) on one RTX 5070 Laptop, a LoRA step
  through a model built in Kimi K2's shape with random weights (shape only; nothing was trained on
  K2) peaked at 4,723 MiB of VRAM and took 14.5-18.5 minutes
  (`benchmarks/probe-rtx5070-giant-moe-step.md`). Reachable from the sharder API and the benchmark
  harnesses, and from `soup train` through `training.stream_expert_layout: per_expert` (#2029).
  Backward recomputation preserves the forward autocast settings. (#2008)

- Capture a bounded complete local text-training source SHA-256 at `soup train` startup, retain unknown identities for unsupported/incomplete inputs, and preserve the historical digest through final artifacts and registry registration. (#1764 part 2 by @kok-o in #2024)

- Bind emitted ship evidence to adapter/task/source bytes, suite versions and effective policy; add opt-in `--require-provenance` drift checks and an explicitly caller-asserted `--data-sha256` identity while preserving legacy replay. (#1765 by @kok-o in #2025)

- **`training.stream_expert_layout: per_expert` streams a DeepSeek-V3-family MoE one routed expert
  at a time from `soup train`** (default `fused`). The layer buffers then hold only the dense part, so
  the VRAM bound no longer grows with the expert count. In `soup train` on an 8 GB laptop card,
  the same Moonlight-16B LoRA config peaked at 1.0 GB of GPU memory against 3.1 GB fused (NF4
  experts against bf16, different cache tiers), at about 2.7x the step time
  (`benchmarks/probe-rtx5070-moonlight-16b.md`). Other architectures are refused at setup, `moe_lora` is refused with it, the VRAM pre-flight
  charges the expert in flight, and a cache volume that is not NVMe is warned about. (#2029)

- **Layer streaming shards a DeepSeek-V3 FP8 block-quantised checkpoint by dequantising it**
  (the format DeepSeek-V3 is published in: `float8_e4m3fn` weights with a float32
  `weight_scale_inv` per 128 x 128 block, declared in `config.json`). It used to be refused. Each
  FP8 weight is multiplied by its block scales in fp32 at shard time, ragged edges cropped, cast to
  the stream dtype and then stored as before (NF4 under `quantization: 4bit`); routed experts are
  dequantised before they are stacked or written per expert, and the scales are never stored. The
  work goes one band of rows at a time into an output at the stream dtype, so a tensor costs its
  output plus a small band. No config key: the format is detected from the checkpoint, and the
  streamed model is built at the stream dtype. A tiny DeepSeek-V3 streamed from its FP8 copy
  matches a resident model holding the same dequantised weights bit for bit (logits, loss, LoRA
  gradients, both expert layouts), and its cache holds the same tensors, bit for bit, as one
  sharded from those weights, NF4 included (`benchmarks/gate-fp8-shard.md`). The disk pre-flight
  charges each FP8 weight at its stored rate and nothing for the scales: on Moonlight-16B-A3B
  converted to this format, CPU only, the 27-layer NF4 cache (30.60 GB, sharded in 93 s) came
  within 0.0002% of the estimate, and every checked FP8-derived tensor equals an independent
  dequant of the source. Refused, naming the supported format: every other FP8 spelling
  (per-tensor scales, e5m2, compressed-tensors, ue8m0 scales, static activation scales, an
  out-of-range `weight_block_size`, a missing or misshapen scale, an FP8 tensor without a scale,
  a scale without its tensor), damaged values (a NaN byte, a negative or non-finite scale, a
  float16 overflow) and an FP8 base that is not `model_type: deepseek_v3`. Kimi K2
  (`model_type: kimi_k2`) was still refused by name in this change; #2107 (below) admits
  Kimi-K2-Base and Kimi-K2-Instruct. (#2061)

- **ORPO runs natively on Apple Silicon: `task: orpo` with `backend: mlx` (#2004 by @dryn85-max
  in #2073).** It was refused at config load. Soup's own MLX loop now trains TRL 0.29.1's ORPO
  objective with `training.orpo_beta`: the stable odds-ratio term on mean completion
  log-probabilities, plus the chosen-sequence NLL over prompt and answer. No reference model is
  needed. Rows are tokenized the way TRL's ORPO trainer does, so rows that fit in `max_length`
  train on the same tokens on either backend; over-long rows are cut with Soup's usual cap, and
  ties train as they do in TRL. The adapter loads with `mlx_lm.load`. `--resume` refuses a
  checkpoint from another task or base. A nonfinite loss stops the run. `quantization: 8bit`,
  `lora.r: 0` and non-text modalities are refused on MLX, and `task: preference` stays refused
  there. Measured once on an M4 Pro: one epoch on 600 UltraFeedback pairs took about four minutes
  with Qwen2.5-0.5B-Instruct. Try: `soup train --config soup.yaml` with `task: orpo`,
  `backend: mlx` and `base: mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

- **`training.eval_steps` works on `backend: mlx` for SFT, counted in optimizer updates (#1990 by
  @dryn85-max in #2074).** The field was refused on MLX. MLX SFT now evaluates after every
  N completed optimizer updates (after gradient accumulation, as on transformers) and once after
  the last one. Every validation row counts, including a tail smaller than one batch, weighted by
  supervised tokens. The evaluations do not move the training order or take a step. A validation
  split smaller than one batch is evaluated rather than skipped when the field is set, and an
  empty split is refused before the model loads. Unset, MLX keeps mlx-lm's own cadence. The run
  result reports `val_loss`, and `adapter_config.json` records the interval. Try:
  `soup train --config soup.yaml` with `backend: mlx` and `training.eval_steps: 10`.

- Add a RunPod cloud-training controller with plan-only rendering, bounded YAML validation, isolated credential-gated submission, SSH identity isolation, per-run host files and confirmed Pod deletion with transient-error retries. Preserve proxy, temporary-directory, Windows and SSH-agent settings while filtering other parent variables, and replace unencodable log characters under isolated Python. Report safe failure reasons and distinguish rejected creates from ambiguous launches; not yet live-validated. (#2081 by @kok-o) Before the release, the controller was changed to run `ssh` and `scp` by the absolute path found on `PATH` (never the current directory) without `RUNPOD_API_KEY` in their environment, to keep `PATHEXT` and `NoDefaultCurrentDirectoryInExePath`, and to fetch results into a dedicated new directory, moving only the expected output directory into place. A failed or timed-out copy removes that directory when nothing was written to it and otherwise names it.

- Add the `documents` extra (`pypdf>=6.21.2` and `python-docx>=1.2.0`) for PDF/DOCX ingestion, and point missing-reader errors to its install command; Markdown and TXT need no extra (#2057 by @kok-o in #2096). Try: `pip install "soup-cli[documents]"`, then `soup data ingest <file>` on a PDF or DOCX file.

- **SimPO runs natively on Apple Silicon: `task: simpo` with `backend: mlx` (#1999 by @dryn85-max
  in #2097).** It was refused at config load. Soup's own MLX loop now trains TRL 0.29.1's CPO
  `simpo` objective: `-logsigmoid(0.1 * (a - b) - simpo_gamma)` on mean completion
  log-probabilities, with beta fixed at `CPOConfig`'s default 0.1 as on the transformers backend,
  plus `cpo_alpha` times the chosen-completion NLL. No reference model is needed. Rows are
  tokenized as TRL's CPO trainer does, so rows that fit in `max_length` train on the same tokens on
  either backend; ties train as in TRL. The adapter loads with `mlx_lm.load`. `--resume` refuses
  a checkpoint from another task or base. A nonfinite loss stops the run. `quantization: 8bit`,
  `lora.r: 0`, non-text modalities and LoRA+ are refused on MLX, and `task: preference` stays
  refused there, now naming `task: simpo` / `task: orpo` / `task: dpo`. Try:
  `soup train --config soup.yaml` with `task: simpo`, `backend: mlx` and
  `base: mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

- Add eager `--version` option to the root CLI, printing package version and repository URL without importing heavy dependencies (#2101 by @Nari-Ab in #2105).

- **Layer streaming admits Kimi K2 (`model_type: kimi_k2`).** Kimi-K2-Base and Kimi-K2-Instruct are
  DeepSeek-V3's decoder in DeepSeek-V3's FP8 block format, but their `config.json` declares
  `model_type: kimi_k2` with an `auto_map` onto the repository's own code, so `AutoConfig` refused
  them as custom code before the architecture list was reached (layer streaming never runs a
  repository's model code). The streamed model's config is now read from that file as plain data
  into transformers' own `DeepseekV3Config` (`model_type` set to `deepseek_v3`, `auto_map` dropped),
  after a guard that admits a K2 config only where that class computes what K2's own code computes:
  `noaux_tc` sigmoid routing with one expert group, top-k of at least 2 under renormalisation, every
  layer past `first_k_dense_replace` an MoE layer, `num_key_value_heads` equal to the head count,
  only the yarn rope keys K2 reads, unquantised or DeepSeek-V3 block FP8 weights, every top-level
  field of the published K2-Base / K2-Instruct `config.json` present (only `auto_map` and
  `quantization_config` may be left out: a missing `rope_theta` would otherwise take the native
  default 10000.0), and no field outside it (besides three metadata keys). A config that fails
  any of them is refused before the tokenizer or any weight is loaded, naming every such field and
  its value and saying that other K2 releases are out of scope for now; Kimi-K2-Thinking
  (compressed-tensors INT4) is refused as `quantization_config.quant_method='compressed-tensors'`,
  and Kimi-K2-Instruct-0905 for the fields its file adds (`ep_size`, `pad_token_id` and serialised
  defaults).
  In the gate (`benchmarks/gate-kimi-k2-admission.md`) the field-by-field config check fails as
  registered on one field: transformers adds `rope_theta` and `rope_type` to `rope_scaling`, as it
  does for any DeepSeek-V3 config. The other checks pass. A tiny K2-like checkpoint (the published
  config with tiny dimensions) streams bit for bit like a resident `DeepseekV3ForCausalLM` on the
  same weights (logits, loss, LoRA gradients; `fused` and `per_expert`; bf16 and FP8 block), and the
  disk pre-flight sizes the published Base layout exactly:
  533.2 GB per_expert and 2.04 TB fused under `quantization: 4bit`
  (`benchmarks/gate-kimi-k2-admission.md`; CPU only, no K2 weights downloaded, nothing trained on
  K2). K2's tokenizer is repository code: it needs `--trust-remote-code`, which reaches the tokenizer
  only, and `pip install tiktoken`, which Soup does not install; a failed tokenizer load on a K2
  base now says so. The streamed tokenizer load is handed the config already loaded: before, with
  `--trust-remote-code`, `AutoTokenizer` read the config again with that flag and, for a repository
  with an `auto_map`, imported and ran its configuration module (`configuration_deepseek.py` on K2,
  and on Moonlight-16B since #1933). Known differences from K2's own code remain: the MoE combine's
  rounding and eager versus sdpa attention. Known limit: on a full-size K2, transformers'
  `DeepseekV3NaiveMoe` briefly allocates `torch.empty(384, 4096, 7168)` bf16 on the CPU per MoE
  layer (about 22.5 GB of commit) before accelerate moves it to meta, in the setup's NF4 /
  `moe_lora` probe and in `build_streamed_model`; RSS barely moves, but a box without that commit
  headroom fails with a raw `DefaultCPUAllocator` error (DeepSeek-V3 had the same before, at about
  15 GB). (#2107)

- **KTO runs natively on Apple Silicon: `task: kto` with `backend: mlx` (#2000 requested by
  @Amix29, built by @dryn85-max in #2111).** It was refused at config load. Soup's own MLX
  loop now trains TRL 0.29.1's KTO
  objective with `training.kto_beta`: summed completion log-ratios against a reference
  precomputed once from the base model, and a detached KL estimate built from TRL's own
  mismatched rows. Batches run in dataset order, as in TRL. Rows that fit in `max_length` train
  on the same tokens on either backend. Labels are read by the loader's KTO converter, exactly
  as on the transformers backend. An empty completion, or a prompt that leaves no completion
  room, is refused before training. `batch_size` stays at least 2, and a run whose gradient
  accumulation cuts the last batches warns naming them. `--resume` refuses another task or
  base, and a nonfinite loss stops the run. `quantization: 8bit`, `lora.r: 0`, non-text
  modalities, DoRA and LoRA+ are refused on MLX. Try: `soup train --config soup.yaml` with
  `task: kto`, `backend: mlx` and `format: kto` data.

- MLX text SFT supports the loss watchdog: consecutive high-loss reports stop at a completed optimizer update, save the last loadable adapter and record the stop reason and actual completed work (#1989 by @Amix29 in #2115).

- **GRPO can use an LLM judge as its reward: `training.reward_fn: judge_rank` (#2047 by @ybai08
  in #2117).** One judge call sees all `num_generations` completions of a prompt together and
  scores each from 0 to 1 against the others, in a fresh order on every call; it also runs in an
  ensemble such as `format,judge_rank`. The judge is set with `training.grpo_judge` (the URL form
  of `online_dpo_judge`, checked at load) and the run must state `training.grpo_judge_max_calls`:
  after the data loads and before the model does, a run that could make more judge calls than that
  is refused with the count. `grpo_judge_rubric`, `grpo_judge_max_tokens` and
  `grpo_judge_max_failure_rate` are optional. A reply that is not a JSON object with one score
  in [0, 1] per completion is asked for once more; a group that still has none scores `None`
  instead of 0.0, is logged as `judge/failed_group_rate` (and groups scored all equal as
  `judge/equal_score_group_rate`), and training stops with an error naming the judge once too
  many training groups failed. A judge that cannot be reached ends the run. Prompts and
  completions are sent to the judge host. A request the judge answers with 429 or a 5xx, or
  that times out or loses its connection, is retried up to twice inside each counted call, so the
  judge host can see up to three requests per counted call. `training.reward_fn: judge_rank` on
  any task other than `grpo`, or with `modality: vision`, is now refused at load. Every new field is off by default and no
  existing default changes; other judge requests keep `max_tokens: 1024`.

- Add `qwen3.5-4b-dpo`, a text preference-training recipe for `Qwen/Qwen3.5-4B`,
  with independently tested metadata/YAML identities and the explicit language-tower
  contract. Settings inherit the existing 9B DPO sibling; no full-4B training or
  performance result is claimed (#275 by @kok-o in #2129).

- Add `training.stream_recompute: on | off | auto` for layer streaming (SFT). `off` keeps each
  streamed layer's activations instead of re-running its forward as the checkpoint recompute;
  `auto` measures one step at the configured shape at setup and keeps activations for as many of
  the last layers as fit in VRAM (RAM tier only). Gradients are bit-identical to the recompute
  path; on an RTX 5070 Laptop 8 GB a streamed Mistral-7B NF4 step is 22% / 17% / 8.5% faster at
  512 / 1024 / 2048 tokens. The default `on` keeps the shipped behaviour. (#2152)

- Add `qwen3.5-2b-dpo`, a text preference-training recipe for `Qwen/Qwen3.5-2B`,
  with independent metadata/YAML identity tests, language-tower and preference-pair
  checks, and CPU/meta architecture qualification. Settings inherit the existing
  Qwen3.5 DPO siblings; full-2B training and NF4 performance remain unmeasured
  (#275 by @kok-o in #2160).

### Changed

- **Config values that v0.76 loaded with a "Soup v0.77 will refuse it" warning are now refused at load (#808, #759, #795, #1418).**
  v0.76 gave one release of notice; v0.77.0 flips both switches (`STAGED_FIELD_SEVERITY` and
  `DEPRECATED_VALUE_SEVERITY`) to `"error"` in the release commit. A config that sets one of the
  items below no longer loads: `soup train` exits `1` with `Config validation error:` and one
  indented line per refused setting, `soup doctor --config` exits `2` with the same lines,
  `soup recipes verify --config` and `soup ship --config` exit `3`, and
  `load_config_from_string` (API / UI) raises `ValueError` with the same refusals (a value refusal
  keeps Pydantic's `Value error, ` prefix there). `soup plan` and `soup apply` read the YAML
  without building the config, so they write and accept a plan for a config that `soup train`
  refuses; check it with `soup train --dry-run` or `soup doctor --config`. A config that sets
  both kinds gets both in one report, the staged-field lines first. When a staged field's own
  check fails as well (for example `grace_codebook: true` without a size and a dimension,
  `long_context_grpo: true` on `sft`, or `video_fps` with a non-video format), only the staged
  line is shown, so no message asks for a line that the next load refuses.
  Nothing else changes, and a config that leaves these items unset, or writes them at their
  defaults, loads as before.

  - **Staged fields (#808, #761).** Thirty-one fields that are validated but read by nothing are
    refused when set away from their default, each on its own line ending
    `is accepted but read by nothing; refused as of v0.77. Delete the line.`:
    `training.long_context_grpo`, `vision_grpo`, `quantize_ref_model`, `grace_codebook`,
    `grace_codebook_size`, `grace_codebook_dim`, `lr_groups`, `early_stop_patience`,
    `citation_recall_threshold`, `forgetting_eval_steps`, `forgetting_benchmark`,
    `forgetting_stop`, `checkpoint_eval_steps`, `checkpoint_eval_metric`,
    `checkpoint_eval_tasks`, `checkpoint_keep_top`, `convergence_window`, `convergence_rel_tol`,
    `rl_checkpoint_include_ref_model`, `rl_checkpoint_include_rollout_buffer`, and
    `data.video_dir`, `video_fps`, `video_maxlen`, `image_min_pixels`, `image_max_pixels`,
    `image_resize_algorithm`, `split_thinking`, `eval_on_each_dataset`, `resize_vocab`,
    `extend_conversation`, `skip_prepare_dataset`. **What to do:** delete each named line. None of
    them changed what a run did, so removing them changes no result. For regression gating and
    early stopping, use `soup train --gate <suite.yaml>`. A `soup sweep --param` that sets one of
    them in any arm is refused the same way: the command exits `1` before the first arm starts.
  - **`data.remove_unused_columns: true` (#759).** No trainer ever read it. `soup autopilot` and
    run-config dumps from v0.75 and earlier wrote it into every config, so such files now stop at
    load. **What to do:** delete the line (or set it to `false`).
  - **`training.quantization: 4bit` or `8bit`, or `training.load_in_8bit: true`, on a task that
    never quantises the base (#795)**: `distill`, `classifier`, `reranker`, `cross_encoder`, `prm`,
    `moe_lora_routing`, `unlearn` and `asr`. v0.76 resolved these to `none` with a warning. Run
    configs dumped while `4bit` was the default carry the line literally. **What to do:** delete the
    line or set `quantization: none`; for `load_in_8bit: true`, delete that line and any
    `quantization` line. Leaving `quantization` unset still resolves to `none`. On these tasks with
    `backend: unsloth` (no unsloth setup) or `backend: mlx` (no MLX trainer), the same refusal
    ends with the backend's fix too (`Also, ... Use backend: transformers.`), so one edit makes the
    config load.
  - **`soup ab --beta` and `MsprtConfig(beta=...)` (#1418).** `--beta` has done nothing since
    #1418. `soup ab --beta ...` now exits `2` before reading any rows, like its other flag errors,
    and `MsprtConfig(beta=...)` raises `ValueError`. **What to do:** remove `--beta`; for a stricter
    `accept_h0`, lower `--alpha`.

  The `--beta` help text, the `data.remove_unused_columns` description in
  `docs/config-reference.md`, and `docs/training.md`, `docs/evaluation.md` and
  `docs/peft-and-efficiency.md` now describe the refusal, not the warning. The warning path stays
  in the code, behind the two switches, for the next value that needs a release of notice.

- **`LICENSE` is the published Apache-2.0 text again, the holder line is spelled once, and `license-files` is pinned (#1677 reported by @Ercaner1988, in #1742).**
  `LICENSE` had drifted from the text on apache.org: section 6 lacked "reasonable and customary
  use in" and section 9 read "Warranty or Support" where the published wording is "Warranty or
  Additional Liability". It is now byte-identical to the published file, with the appendix left
  as the standard template. `NOTICE`, the five READMEs, the package `authors` and one stray test
  header now carry the same holder line, `Copyright 2026 Makazhan Alpamys and the Soup
  contributors`, and `NOTICE` is reduced to the product name and that line. `pyproject.toml`
  names `license-files = ["LICENSE", "NOTICE"]` instead of relying on hatchling's default globs;
  `tests/test_issue1677_licence_files.py` fails if the key, either file, the two restored
  passages or the holder line goes. The README points redistributors to sections 4 and 6 of the
  licence, `CONTRIBUTING.md` and the PR template say that contributions are accepted under
  Apache-2.0 (no CLA, no DCO), and `tests/fixtures/cyclonedx/README.md` records the sha256 of the
  three vendored CycloneDX schemas.

- **Polished the Simplified Chinese wording for the post-#331 8B NF4 README figures.** `README.zh.md` now uses clearer wording for the clock-adjusted comparison and the two measurement rows; all numbers, links and code spans are unchanged. (#1836 by @Momoyeyu)

- **Polished the Simplified Chinese wording of the v0.76.0 "What's New" translation (#1837 by @Momoyeyu).** `README.zh.md` now states that `data.val_split` holds rows out for validation, names the disk/RAM tiers as storage tiers, and renders `unpinned` as not bound to CPU cores; all numbers, links and code spans are unchanged.

- Layer streaming: each streamed decoder layer now calls `functional_call` with `tie_weights=False` when none of its substituted weights is tied, skipping a walk over every parameter and buffer on each forward and checkpoint recompute (~0.1 ms per call, ~6.7 ms per 32-layer step on a PEFT Mistral layer); results are unchanged by construction (#1890).

- Verify core dependency floors in a dedicated, isolated CI cell, including real CLI help/startup and Plotext 5.2.2 rendering, without conflicts from preinstalled runner tooling (#1802 by @kok-o in #1927).

- Support Typer's standalone and vendored Click implementations while preserving CLI usage exit codes, option precedence, filesystem isolation and complete help/reference traversal; admit Typer through 0.27.x while retaining the standalone Click dependency for older supported releases. (#1803 by @kok-o in #2013)

- Polish literal phrasing and punctuation in the Simplified Chinese README while preserving its structure, figures, links, code and sync stamp (#1838 by @Momoyeyu in #2067).

- **The Docker image now builds on Ubuntu 24.04 with Python 3.12 and installs exactly the tagged soup-cli release (#2092).**
  The image was `nvidia/cuda:12.1.0-runtime-ubuntu22.04` with Python 3.10 and an unpinned
  `pip install "soup-cli[...]"`, so once `requires-python` moved to 3.11 pip would have resolved
  the previous release on Python 3.10 and an image tagged with the new version would have held
  it. The base is now `nvidia/cuda:12.8.2-runtime-ubuntu24.04` with Python 3.12 in a venv at
  `/opt/venv`, and `docker.yml` passes the version of the git tag to the build as `SOUP_VERSION`.
  The build stops if the Python is outside 3.11 to 3.13, if `SOUP_VERSION` is not `X.Y.Z`, or if
  the installed soup-cli is not that version. A local build must name the release
  (`docker build --build-arg SOUP_VERSION=0.77.0 .`, or set `SOUP_VERSION` for
  `docker compose build`), and a manual run of `docker.yml` must be started on a tag
  (`gh workflow run docker.yml --ref v0.77.0`): on a branch it now stops instead of pushing
  `latest`. The entrypoint, working directory and extras (`train`, `serve`, `data`, `eval`) are
  unchanged. The container runtime now asks the host for a driver that supports CUDA 12.8 (570
  or newer on GeForce cards) where the 12.1 base accepted 525 and 530 or newer.

- **Bind base revisions in single-process HF layer streaming
  (#1902 by @kok-o in #2094).** Streamed config/skeleton and checkpoint downloads
  share the run's resolved commit, with revision-specific shard and weight-copy
  cache slots. The first pinned streamed run after upgrading re-shards and
  re-copies the base into the new `<slug>@<commit>` slots, costing disk space and
  time; old slots are left in place. Local/cache-path hints remain unverified;
  Unsloth, MLX and distributed loads remain unbound.

- Document accepted and refused preference row layouts for DPO and KTO in `docs/data.md`, backed by doc-driven test verification in `tests/test_data_docs_preference_layouts.py` (#2086 by @Areen-09 in #2120).

- Coordinate the base-model commit across distributed Hugging Face Transformers
  `soup train` workers: rank zero resolves once, broadcasts nullable identity before
  setup and compares rank receipts before recording final provenance. Offline
  resolution remains unknown; collective errors stop independent per-rank binding.
  This changes distributed Transformers runs from `unsupported_backend` to a bound
  identity with `revision=<sha>` on every rank when resolution succeeds. Initialize
  the shared group with Trainer's 1800-second collective timeout; malformed/stale
  distributed launch declarations fail instead of silently running unbound.
  This is a separate distributed slice of #1902, not universal backend coverage
  (#1902 by @kok-o in #2127).

- `soup serve`: `POST /v1/tools/python` now answers HTTP 501 without running the code where
  the host has no OS-level isolation (Windows, Linux with Python 3.11) and where the Linux
  namespaces cannot be created for the run (a kernel or container that refuses them), the
  same as `POST /v1/tools/bash`. It runs where `os.unshare` creates the namespaces (Linux,
  Python 3.12+) or `sandbox-exec` is available (macOS). `code_exec_reward` and the recipe
  `code` node are unchanged (#2166).

- **`soup serve`: `/v1/tools/python` and `/v1/tools/bash` require `--tool-auth-token` on every bind, as they already did beyond loopback.** A server started without a token now answers both routes with 401 on a loopback bind too, before the request's content is used; the reply names the fix (start `soup serve` with `--tool-auth-token <secret>` or set `SOUP_TOOL_AUTH_TOKEN`, and send `Authorization: Bearer <secret>`). `--tool-auth-token` now also reads `SOUP_TOOL_AUTH_TOKEN`, which keeps the token off the command line. `/v1/tools/web_search`, `/v1/thumbs`, the adapter routes and the generation routes are unchanged, and `soup serve` still starts without a token on loopback. To keep calling the two routes after upgrading, pass the token to the server and send it with each request (#2174)

- `soup deploy autopilot --target rtx-4090-24gb` now plans 4bit QLoRA + vLLM: it wrote `quantization: awq` (load a pre-quantized AWQ base) for an ordinary base, a route Soup no longer sets up since the AWQ export and extra are gone (#1951). The profile's notes say to point vLLM at a pre-quantized AWQ checkpoint directly; the `--measure-candidates` examples now read `4bit,8bit,fp8`.

- `soup ingest --logs` names a directory or a non-UTF-8 export ("is a directory, not an export file", "is not valid UTF-8 (byte N)") and exits `1`, instead of stopping on an uncaught `FileNotFoundError` / `UnicodeDecodeError`; an existing output is still left untouched (#2018 follow-up).

### Removed

- **Removed deprecated `awq` and `gptq` export and quantize formats and extras (#1710 by @Nari-Ab in #1951; see #338, #1690).** Both upstream projects (`autoawq` and `auto-gptq`) are archived and incompatible with `[train]`. `soup export --format awq` / `--format gptq` and `soup quantize --to awq` / `--to gptq` are now refused at the CLI with non-zero exit codes naming the remaining supported formats (`gguf`, `onnx`, `tensorrt`, `bitnet`, `tq1_0`, `torchao`, `gguf-ud`). `soup export` also lost `--bits`, `--group-size`, and `--calibration-samples`, so scripts passing them now exit 2 with `No such option`; `--calibration-data` is now read only by `--format gguf-ud`. The `[awq]` and `[gptq]` extras and related export helpers are removed in v0.77; users who still need AWQ or GPTQ export should remain on 0.76.x.

- **Breaking: Python 3.10 is no longer supported (#1805, in #2089).** From 0.77 the package declares `requires-python = ">=3.11,<3.14"` (it was `">=3.10,<3.13"`): Python 3.10 reached end of life on 2026-10-01, and pip on 3.10 keeps installing 0.76. Python 3.13 is now supported and tested in CI, on Ubuntu, Windows and macOS in every nightly and release run; pull-request runs stay on 3.11 and 3.12.

### Fixed

- **`soup loop watch` can now present the `soup serve` tool token (#1139 by @dchaudhari7177 in #1184).** Since 0.75.1 `soup serve` requires its `--tool-auth-token` on the adapter routes, so the loop's deploy stage, which activates the adapter over HTTP, was refused on any server started with a token, and reported only a bare `activate POST failed`. Pass the new `--tool-auth-token` or set `SOUP_TOOL_AUTH_TOKEN`; a `--detach` child gets the token through its environment, never its argv, and the audit log records it as `<redacted>`. A refused activation now says whether the endpoint wanted a token that was not sent or rejected the one that was, and names both ways to supply it. Without a token the loop sends no `Authorization` header, as before. The activate POST to a plain-http (loopback) endpoint now ignores `HTTP_PROXY` / `ALL_PROXY`, so a proxy never sees the token in clear text; an https endpoint still reads the environment, including `SSL_CERT_FILE` / `SSL_CERT_DIR`.

- **`training.reasoning_effort` and `training.train_on_eot` now load only on the tasks that read them (#806 by @dchaudhari7177 in #1587).**
  Both are applied by the SFT formatter, which runs on `sft`, `tts` and
  `distill`. Before, the gate refused `tts`, where both flags take effect,
  and accepted `pretrain`, `classifier`, `reranker` and `cross_encoder`,
  where they loaded and changed nothing. Such a config now fails at load.
  On `pretrain`, `train_on_eot` changed only the `soup data preprocess`
  cache key, never the cached rows; a pretrain cache built with the flag
  has to be rebuilt without it.

- Square-bracket text in model replies and names/paths (`soup chat`, `soup diff`, `soup eval human`)
  and dataset keys/issues (`soup data inspect`, `soup data validate`) is escaped through
  `for_terminal` instead of parsed as Rich markup, preventing `MarkupError` crashes on
  closing tags like `[/INST]` or `[/b]` and preserving literal text (#1596 by @Nari-Ab in #1602).

- Suggest lowering `data.max_length` alongside `--batch-size` in the
  friendly error hints for `CUDA out of memory` and `OutOfMemoryError`
  (#1694 by @Nick-800 in #1708).

- **The layer-streaming disk pre-flight reported a reusable cache at the size it
  would have if it were rewritten, not at the size of the files it keeps (#1703).**
  When the cache is reusable, `Layer-shard cache: ... GB (reusable)` and the
  projected total now add up the cache's own files (`index.json`, the extras shard,
  the large shards and every layer file, wherever it is striped), and a measured
  size is not labelled "approximately". Other files in the folder are not counted,
  and a cache file that cannot be measured keeps the estimate (#1703 by @drakeo338
  in #1709).

- `soup serve --backend sglang` now applies a LoRA adapter passed as `--model`. SGLang
  applies an adapter only to the requests that name it in `lora_path` and runs every other
  request on the base weights. The backend registered the adapter at startup but never
  named it, so every answer came from the base model while `/v1/models` and each response
  reported the adapter. The adapter is now registered under the name `adapter`
  (`adapter=<path>`, which also keeps a path holding `=` from being split there), and every
  request names it, on the token-ids and the string route, streaming or not. A full
  model's requests name no adapter and are unchanged. The `[sglang]` extra and
  `soup doctor` now require `sglang>=0.5.1` (was `>=0.2.0`): older releases refuse to
  start an engine with a LoRA adapter unless its radix cache is turned off, which Soup
  does not do (#1724 by @Batyrkhantzy111 in #1732).

- `training.moe_aux_loss_coeff` set to a non-default value on `backend: unsloth` (sft, pretrain) or with `training.stream_layers: true` is now refused at config load instead of loading and being ignored: neither setup applies the MoE auxiliary loss (#1412 reported by @maximilianfeix, fixed by @kok-o in #1749).

- Layer streaming `StreamPrefetcher.prime` skips reloading decoder layer 0 when the slot already holds it from the preceding backward pass, avoiding redundant H2D copies at step boundaries (#1354 by @kok-o in #1750).

- **Tool-call eval scorers no longer credit a JSON number as a boolean (#1755 by @Faisal-Fayaz in #1756).** `tool_call_match` and `tool_call_args_subset` compared parsed `arguments` with Python `==`, and Python has `bool` as a subclass of `int`: a model that answered `{"enabled": 1}` where the task expected `{"enabled": true}` scored `1.0` — full credit for a wrong answer, through `score_task` and `run_eval` alike. Both are valid JSON, and they are different arguments. Arguments are now compared with a `_json_equal` that rules on `bool` before anything else, so a boolean and a number never match at any depth, while numbers stay equal across forms (`1 == 1.0` still matches, as before). It walks nested arguments through an explicit worklist rather than recursion, so an argument nested as deeply as `json.loads` accepts is still scored instead of raising. `tool_call_match` scores such a call `0.0`; `tool_call_args_subset` keeps its `0.5` function-name credit and gives no argument credit, so `0.5`, and its `0.5` `matched` threshold is unchanged. Scores recorded before this change can be higher than the same outputs score now, because a boolean/number substitution used to count as correct.

- **`training.use_mod: true` is refused at config load on routes that never install a Mixture-of-Depths router (#1752).** The flag was accepted on every task. Only the resident LoRA arm of the transformers SFT setup (which `tts` reaches) and the transformers pretrain setup (after its LISA/LoRA split, so LISA is included) call the router. Full fine-tuning, LISA and Spectrum on SFT, vision, audio, layer streaming, unsloth and MLX SFT all trained without one. Those requests now fail at load, naming `training.use_mod`, the task, the settings that leave the applying path, and a valid change or `use_mod: false`. Transformers SFT LoRA, TTS LoRA, pretrain LoRA and pretrain LISA still apply it. Pretrain does not inherit SFT's modality or streaming restrictions. The MLX SFT trainer keeps its own warning for the field, but a config can no longer reach it, so `soup doctor --config` lists no row for it. **This is a breaking change** for configs that set `use_mod: true` on a route that never applied it — that run was already training a different model than requested. `use_mod: false` and an omitted flag are unchanged, and an out-of-range `mod_capacity_factor` is still rejected by its own check (#1752 by @mvanhorn in #1757).

- `copy_bundle_to` now accepts an explicit keyword-only `root` argument so callers with their own workspace can safely import demo fixtures under that root without changing the process working directory or bypassing containment checks (#1762 by @Nari-Ab in #1767).

- **PRM and RAFT divide each accumulated micro-batch mean by `gradient_accumulation_steps` (#1444 by @Momoyeyu in #1771).** Both trainers return a per-micro-batch mean and cannot honour HF's label-count `num_items_in_batch`; they now decline loss kwargs so `Trainer.training_step` applies its standard `1 / gradient_accumulation_steps` normalisation. Logged `loss`, `train_loss` and `grad_norm` match a `gradient_accumulation_steps: 1` run instead of reading G times too large, and `max_grad_norm` clips the gradient for the stated loss. RAFT's `loss_weights` weighted mean, including citation boosts, is unchanged.

- Dynamic warmup for PRM, MoLE routing, and PPO wrappers via resolve_trainer_warmup_steps, and wire PPO deepspeed, response_length, and save_total_limit settings (#805 by @Tanmay9223 in #1775).

- GGUF export reports pipeline phases and streams converter/quantizer logs with `--verbose`; `--no-install-deps` refuses missing conversion packages before merge/toolchain work without invoking pip. (#1760 reported by @abdulwaarith0, fixed by @Amix29 in #1777)

- Honor `training.gradient_checkpointing` in the TRL/HF arguments for GRPO, IPO, BCO, reward-model, online-DPO, pretrain and embedding, including unquantized runs. Preserve the full-checkpointing fallback for enabled tiers. Behaviour change: with the default `gradient_checkpointing: false`, GRPO, IPO, BCO, reward-model and online-DPO now train with checkpointing off, where TRL's own default (`true`) used to apply; set `gradient_checkpointing: true` to keep it. (#797 by @Amix29 in #1778)

- **Non-finite eval-gate baselines are now refused at read, write, and gate boundaries instead of silently passing (#1773 reported by @taljeon, fixed by @here-2007 in #1779).** `resolve_baseline`, `stamp_baseline_scores`, `write_baseline_file`, and `run_gate` now require scores to be finite numbers; `NaN`, `+inf`, and `-inf` (including JSON numeric overflows such as `1e999` and `-1e999`) raise a `ValueError` naming the offending task. In `soup eval gate`, invalid baselines exit with code 3 (`EXIT_USAGE_ERROR`) before evaluating tasks or reporting a verdict, rather than printing `GATE PASSED` or misdiagnosing positive overflow as a regression.

- **KL calibration probability calculation now uses numerically stable log-softmax without artificial flooring (#1772 reported by @taljeon, fixed by @here-2007 in #1780).** `eval.calibrate.kl_divergence` previously floored candidate probabilities at `1e-12` before taking logarithms, which truncated large logit divergences (gaps > ~27.6) and caused `run_calibration` to return `MINOR` when the actual divergence exceeded the documented `MAJOR` threshold (>= 0.20). KL divergence is now evaluated directly using stable log-probabilities via `_log_softmax`, preserving monotonic scaling for large divergence gaps without underestimating quant-calibration deltas.

- **`training.expand_layers` (LLaMA Pro) is refused at config load together with a LoRA rank (#1326 by @abdulwaarith0 in #1783).**
  The block expansion runs before `get_peft_model`, which froze the appended
  blocks again, so only adapters on the original layers trained while the
  appended blocks stayed the identity. The error names `expand_layers`,
  `freeze_trainable_layers` and `lora.r` and points to `lora.r: 0` with
  `quantization: none`; `task: pretrain`, which has no `lora.r: 0`, gets the
  `expand_layers` scope refusal instead (#1813). The LLaMA Pro examples in
  `docs/peft-and-efficiency.md` and `docs/training.md` now set `lora.r: 0`, and
  `docs/training.md` now says the expansion runs before `get_peft_model`.

- **`docs/backends-and-ops.md`: the contents entry for the telemetry section reaches its heading
  again, and the shell-completions section says what completes today (#1788).** The entry linked
  `#telemetry-opt-in-hardware-info-only`, an anchor that stopped existing when the heading became
  "Telemetry (opt-in)". The completions paragraph promised recipe-name completion, named a
  `--target-modules` flag that no command has, and said config introspection "lands in v0.64.1".
  What `soup completions` gives is command names and option names; no option or argument has a
  value completer attached, and the two completers in `soup_cli/utils/completions.py` are helpers
  that nothing calls yet.

- **`soup doctor --config` ignores explicitly written schema defaults (#1330).** Minimal configs and full YAML dumps now produce the same support report, including vision/audio defaults; `seed: 0` remains active. Removed unreachable MLX/Unsloth rejection rows and multimodal response-mask rows, retaining documentation of their limitations and load-time refusals. Round-trip regressions now pass without xfails. Follow-up to #1336 by @Yi-111-a. (#1330 by @akkupratap323 in #1791)

- CLI help now includes descriptions for the remaining plugin, data, registry, `.can`, and agent command entries, with a regression test preventing new blank help text (#1793 by @KeuqS in #1795).

- **A resumed `dpo`, `ipo` or `grpo` run continues from the checkpoint's adapter (#1768 reported by
  @bobbonauta, fixed by @ybai08 in #1797).** TRL adds a frozen `ref` adapter for these tasks, so their
  checkpoints hold the trained adapter at the root and `ref` in `checkpoint-N/ref/`, and
  `Trainer._load_from_checkpoint` then loads only the subfolder. `--resume` restored the optimizer
  and the step counter and trained a freshly initialised adapter: on a 4-step run resumed at step 2,
  the result was two steps from zero. The trained adapter is now loaded right after the Trainer's
  own load. `task: preference` with `dpo` or `ipo` is covered; `grpo` with `grpo_beta: 0`, `orpo`,
  `simpo`, `bco`, `kto`, `sft` and `reward_model` write no `ref/` and were not affected. **Re-run
  any dpo, ipo or grpo training that was resumed from a checkpoint.**

- Refuse empty, missing, or misspelled eval-gate task lists before reporting a verdict; direct `run_gate` calls also reject a suite emptied after loading. Nonempty gate scores, thresholds, baselines, the no-model smoke generator, and unknown-key handling are unchanged (#1798 by @taljeon in #1821).

- Sanitize dataset previews, review panels, paths, and image format labels before terminal rendering. (#1812 by @kocaemre in #1822)

- Seven fields in `soup.yaml` without schema descriptions (`data`, `training`, `training.gradient_accumulation_steps`, `training.warmup_ratio`, `training.weight_decay`, `training.max_grad_norm`, and `training.lora`) now carry descriptions in `SoupConfig` (#1794 by @Nari-Ab in #1831).

- `soup loop watch` with the pre-wired stages now learns from `soup serve --record-thumbs`: the harvest reads the thumbs database named by `SOUP_LOOP_THUMBS_DB` or `soup loop init --thumbs-db PATH` and unions its preference pairs (tagged `source: "soup-serve-thumbs"`) with the trace pairs. Before, it read only `--trace-log` directories, which carry no thumbs signal, so a loop next to a serve that recorded thumbs harvested nothing. A configured database outside cwd, a symlink, a missing file, or a file that is not a thumbs database stops `soup loop watch` with exit 2, naming the field (#1440 by @Srinivasan8888 in #1832).

- Gate-style commands now name the problem in their error text instead of printing a bare value: `soup expect` reports `data file not found: <path>` / `suite file not found: <path>` for a missing input, `soup lock check`/`soup lock show` name a corrupt lock file and say its JSON is invalid, and `soup draft measure`/`soup draft distill` point at the missing `[train]` extra when `transformers` is absent. `soup expect`'s module docstring is also corrected (a suite failure exits 2, not 3). Exit codes are unchanged (#1830 by @abhinyaay in #1833).

- **`soup ui` validates the served Host names on a loopback bind.** Requests must name the bound host or `localhost` / `127.0.0.1` / `[::1]` with the bound port; an omitted port means HTTP port 80. Foreign names, wrong ports and missing, malformed or duplicate headers get `403 {"detail": "Host not allowed"}` before authentication, body reads or routes, with the normal security response headers. This closes the Host/Origin hardening gap in the session-cookie check. `--public` keeps its Bearer gate, and the startup login URL is unchanged (#1829 by @vaputa in #1839).

- Bracketed text in seven CLI help strings (`[deprecated]`, `[doc-id]`, `[doc-N]`, `soup-cli[sigstore]`, `[data-pro]`) is now escaped so Rich markup does not drop it (#1792 by @Nari-Ab in #1840).

- Early CLI validation and device preflight checks for `soup bench train` (`--steps <= --warmup`), `soup recipes verify` (missing or unparseable `--config`), and `soup infer` / `soup bench infer` (`--cuda-graphs` on non-CUDA setups) now refuse in well under a second before torch is imported or a model is loaded (#1817 by @Nari-Ab in #1841).

- Remove outdated unlearning release promises and guard exception messages across the source tree against expired promises. (#1405 by @kok-o in #1842)

- Refuse training.citation_faithful on pretrain, whose trainer does not apply citation span weights; use task: sft with data.format: raft. (#1406 by @kok-o in #1843)

- Attribute Linux streaming disk detection to the cache volume, resolving partitions and stacked-device members instead of accepting an unrelated NVMe disk; retain the path-based throughput fallback. (#1800 by @kok-o in #1847)

- Share text-only conversation prompt extraction across trace and ingest parsers, retaining context through the last user turn so unrelated answers do not form preference pairs; keep system-plus-one-user prompts as plain user text, use role/content JSON only when an earlier user or assistant turn exists, normalize LangChain human/ai/system message types, omit blank context turns and non-text content parts, and select assistant output explicitly. (#1809 by @kok-o in #1848)

- **MLX SFT runs are reproducible: `training.seed` and `training.data_seed` are applied
  (#1850 by @dryn85-max in #1855).** mlx-lm's batch shuffle draws from numpy's global RNG, and
  LoRA init and dropout draw from `mx.random`. Soup seeded neither, so every run of one
  `soup.yaml` trained on a different batch order, even with the seed left at the 42 the panel
  prints. `mx.random` is now seeded before the adapter is built and numpy right before training,
  and the `MLX backend ignores: training.seed` warning is gone. Runs started before this change
  cannot be reproduced. Try: run `soup train --config soup.yaml` twice with `backend: mlx` and
  compare the losses.

- **`backend: mlx` no longer crashes when a split is smaller than one batch (#1851 by
  @dryn85-max in #1857).** mlx-lm trains and evaluates in whole batches. A one-row validation
  split, which the default `val_split: 0.1` gives on the bundled 10-row `alpaca_tiny.jsonl`,
  crashed at the first evaluation with `Dataset must have at least batch_size=2 examples`, after
  the model had loaded. Such a split is now skipped with a warning that names both numbers and
  says `val_loss` will be absent, and the run trains. A training split smaller than one batch is
  refused before the model loads. Try: `soup train --config soup.yaml` with `backend: mlx`,
  `data.train: examples/data/alpaca_tiny.jsonl` and `batch_size: 2`.

- Quote leaderboard CSV model and benchmark fields without changing ordinary output bytes. (#1834 by @taljeon in #1863)

- Model and output paths printed in `soup chat`, `soup diff`, and `soup eval` (`benchmark`, `custom`, `human`, Elo table, leaderboard) are escaped through `for_terminal` instead of raw Rich markup, preventing `MarkupError` crashes on bracketed paths like `models/[/x]` and stripping terminal control characters (#1864 by @Nari-Ab in #1868).

- **`soup ship` error messages no longer pass control bytes to the terminal (#1869 by @bernalalexis-try in #1872).** `_fail` escaped Rich markup but let ESC and C1 bytes through, including in exception text forwarded from rendering a Hub model's chat template. It now strips control characters before escaping.

- **Dataset commands warn and skip non-object JSON rows instead of crashing.** JSONL warnings identify the physical line and JSON-array warnings identify the row; validation lists these as issues, and review checks objects before accessing pair fields. Rejected contents are not printed. Canary insertion still refuses a non-object first row and preserves later rows. (#1860 by @akkupratap323 in #1873)

- **`task: pretrain` with `lora.r: 0` is refused at config load, and so is `training.expand_layers` (LLaMA Pro) on `task: pretrain` (#1813 by @abdulwaarith0 in #1875).**
  The pretrain trainer always attaches a LoRA adapter, so a rank of 0 loaded
  and then failed inside peft (or unsloth) when training started, with an error
  that named neither Soup nor the field. The refusal names `lora.r` and the
  task, and points to `lora.r >= 1`, LISA or `task: sft`. LISA keeps accepting
  `lora.r: 0` on pretrain, since it attaches no adapter and trains as before.
  LLaMA Pro had no pretrain route that trained the appended blocks (a rank
  freezes them again, #1326), so the `expand_layers` scope check now refuses
  `task: pretrain` with the `task: sft` recipe. `docs/peft-and-efficiency.md`
  and `docs/training.md` say so.

- `soup serve --help` names `/v1/tools/bash` among the endpoints `--tool-auth-token` protects, next to `/v1/tools/python` and `/v1/tools/web_search`, and a test checks every registered `/v1/tools/*` route against the help text. The generated reference pages are regenerated after #1795 and #1783 left them stale (#1879).

- `training.neftune_alpha` is now refused at config load on the 14 tasks whose trainer never forwards it to `TrainingArguments.neftune_noise_alpha` (it was silently dropped); `docs/peft-and-efficiency.md` now lists the 9 tasks that apply it (#1820 by @DBVamsi in #1883)

- `soup export --help` names every accepted `--format`: `torchao` and `gguf-ud` were missing from the list although both have worked since v0.53.1; a test keeps the list equal to `SUPPORTED_FORMATS`. Stale "schema-only" / "deferred" statements in the GGUF, BitNet and save-format module docstrings, a dead TTS anchor in `docs/training.md`, a dead energy anchor in `docs/adapters-and-governance.md` and the "planned PTQ export surface" sentence in `docs/performance-and-quantization.md` now describe what is live (#1886).

- **`training.relora_steps` is refused where the loss has a frozen reference.** trl reads the reference of `dpo`, `ipo`, `kto`, `bco` and `grpo` with `grpo_beta` above 0 from the policy's own base weights, and every ReLoRA restart merges the trained adapter into those weights, so from the first restart on the reference was the policy as of that step, without an error. Those tasks, and a `preference` run whose loss or blend names `dpo`, `ipo` or `bco`, now stop when the config is loaded, with the reason. `sft`, `orpo`, `simpo`, `grpo` with `grpo_beta: 0` and a `simpo` / `orpo` preference run keep ReLoRA (#1835 by @ybai08 in #1891).

- Refuse eval-gate suites with repeated task names that previously loaded, so distinct task scores cannot overwrite name-keyed baseline evidence. (#1824 by @taljeon in #1893)

- `training.lora.use_rslora` is now applied on `backend: mlx` and `backend: unsloth`: MLX receives the rank-stabilised scale `alpha / sqrt(r)` in both `adapter_config.json` and at training time, and the Unsloth loader forwards `use_rslora` to `get_peft_model` from every trainer. Before, the flag loaded on both backends and was never read, so an rsLoRA run trained plain-LoRA scaling (`alpha / r`, an 8x smaller update at the default `r: 64, alpha: 16`) with nothing said. Existing configs with `use_rslora: true` on mlx or Unsloth now train at the rank-stabilised scale, so their results change; drop the flag to keep the old `alpha / r` behaviour (#1853 by @ndablizov in #1899)

- `soup migrate` now writes what the source trained instead of Soup's defaults, and names what it cannot carry: axolotl `val_set_size` and LLaMA-Factory `val_size` map to `data.val_split` (`0` when absent, as both tools default); a LLaMA-Factory config without `lora_rank` / `lora_alpha` gets LLaMA-Factory's `r: 8, alpha: 2 * r`, a comma-separated `lora_target` becomes a list, and `lora_target: all` / axolotl `lora_target_linear: true` become `all-linear` rather than `auto` (two of seven projections on Llama); `pref_beta` under `pref_loss: ipo` maps to `training.ipo_tau` and `simpo_gamma` to `training.simpo_gamma`; the Unsloth migrator takes `base` from the training load, skipping the stock notebook's `if False:` reload, and reads `gradient_accumulation_steps`, `warmup_ratio`, `weight_decay` and `seed` from the final trainer's config. Every source key or trainer keyword a migrator does not map is listed in one "not migrated" warning (#1807 by @ndablizov in #1901)

- Oversized integer baseline scores now raise `ValueError` instead of leaking `OverflowError` (#1865 by @dipesh-singh07 in #1903)

- Five commands no longer exit 0 when nothing succeeded or nothing was compared: `soup sweep` exits 1 when every arm failed (one completed arm still exits 0); a CMA-ES merge in which every evaluation failed raises instead of writing a "converged" merge of the first sample, `converged` requires a real score, and `soup adapters merge --strategy cmaes` exits non-zero and writes no adapter (its missing-`--eval` message now says `.jsonl`); `soup drift-alarm` exits 1 and shows no OK panel when it could read no live or reference row, naming the file and the keys it reads (`output`, `response`, `text`); `soup eval auto` exits 1 when no leg evaluated anything, with the same code whichever leg failed; `soup loop watch --detach` validates `--poll-interval` / `--max-iterations` in the parent and spawns no child when they are invalid (#1808 by @ndablizov in #1905)

- A Hugging Face checkpoint is the files `model.safetensors.index.json` names, not every `*.safetensors` in the directory: a `consolidated.safetensors` beside the shards (the mistral-inference copy Mistral's repositories ship, which transformers never loads) is no longer counted by the hardware-fit pre-flight, `soup profile`, `soup cost` and batch estimation, no longer downloaded for layer streaming and `soup spectrum scan` (29.0 GB for a 14.5 GB model), no longer copied out of the HF cache, no longer scored by the spectrum scan (which emitted `unfrozen_parameters` matching no parameter) and no longer kept resident by the layer-stream sharder. Without an index the checkpoint is `model.safetensors`, else the `model-*-of-*.safetensors` shards, else every `*.safetensors` except `consolidated*`, and the Hub download fetches by the same rule; a directory holding only a consolidated file is refused with its name (#1806 by @ndablizov in #1906)

- `soup data validate` counts every row the loader skipped (not a JSON object, or not valid JSON) as an invalid row in its `N/M rows valid` line, in `--min-valid-fraction` and in the exit code: a file of only junk rows exits 2 ("no usable rows remain") with `--format` and 3 without one, and `[good, 42]` is `1/2 rows valid` and fails `--min-valid-fraction 1.0`. A file that is not empty but has no object row says how many rows were skipped instead of "Dataset is empty" (#1887 by @ndablizov in #1908)

- The 30 shipped grpo, ipo, bco, reward-model and online-DPO recipes, templates and example configs now set `training.gradient_checkpointing: true`: since #1778 passes the key to TRL, leaving it unset fell back to the schema default `false` instead of TRL's `true` and used more activation memory (#1923 by @VincentTLe).

- **Fixed MCQ forgetting scoring for unboxed letter spellings (#1442 by @Momoyeyu in #1924).** `extract_mcq_letter` now accepts a lone option letter wrapped in markdown emphasis, quotes, backticks or square brackets, and a standalone uppercase letter that leads the output before an explanation; boxed values and prose openers such as `A cat ...` still earn no credit.

- Custom eval task loading now reports invalid non-string `scoring` values with
  the field and physical JSONL line instead of an unhashable-type error, before
  model loading or result writes (#1926 by @kok-o).

- Empty judge prompt files now fail eval gates with a task/file diagnostic and no measured score, preventing an unmeasured zero baseline (#1928 by @kok-o).

- **Dataset loading no longer prints one warning per skipped row.** The JSON/JSONL loader and `soup data review` print the first three warnings per kind (non-object rows, invalid-JSON lines), then one `... and N more` line; `soup data validate` still reports the exact skipped count. (#1888 by @george07-t in #1930)

- **Layer streaming builds the streamed model from transformers' own class, even under
  `--trust-remote-code`.** The flag still reaches the tokenizer, which is where a repo like
  Moonlight-16B (remote-code tokenizer only) needs it; its remote modelling code lays the experts out
  differently from the stacked shards and could not be streamed. The dated 2026-10-15 Moonlight-16B
  LoRA run on an 8 GB card is recorded in `benchmarks/probe-rtx5070-moonlight-16b.md`. (#1933)

- **A registry `name:tag` that matches more than one entry is refused instead of resolving to the newest, and `RegistryStore.list()` filters by `run_id` (#1763 by @abdulwaarith0 in #1934).**
  `push` keeps every entry, so pushing the same name and tag twice left twins, and any
  command given that `name:tag` silently used the newest. The error now names the
  candidates and asks for the full id, as an ambiguous id prefix already did.
  `soup eval quant-check` reports either kind of ambiguity as a usage error instead of a
  traceback. `RegistryStore.list(run_id=...)` finds the entries registered from one run.

- `quantization: 8bit` with `backend: unsloth` is refused at config load, naming `4bit`, `none` and `backend: transformers` for 8-bit: Soup's Unsloth loader passes `load_in_4bit` only, so `8bit` loaded the model in 16-bit with no word (#1811 by @ndablizov in #1946)

- `use_liger: true` is reported as applied only when the model's `config.model_type` is in liger's own kernel table: for a type liger has no kernels for (`gpt_neox`, `phi`, `gpt2`, ...) SFT printed "Liger Kernel enabled" and set `use_liger_kernel=True` with no kernel applied, and a `phi` model could be "patched" as Phi-3 by its name; it now prints "no matching architecture found" and trains without Liger (#1811 by @ndablizov in #1947)

- Leaderboard CSV exports prefix a model or benchmark name that a spreadsheet would read as a formula with a quote. (#1950)

- The Web UI now deletes the temporary `soup_ui_config_*.yaml` it writes for each
  `/api/train/start` once the training process exits, and when the process fails to
  start, instead of leaving one file per run in the system temp directory
  (#1464 by @ShivanshShukla in #1952).

- Accept a leading UTF-8 BOM in local JSON, CSV and TXT datasets, matching JSONL, so Excel/Windows exports parse and detect their format correctly while embedded U+FEFF characters are preserved. Thanks to @VincentTLe for the report and proposed fix. (#1922 by @kok-o in #1954)

- Diagnose unsupported text architectures before a fresh default GGUF converter clone or dependency/conversion work, using generated capabilities from the exact llama.cpp pin; probe explicit, environment-selected and cached converters independently, preserve inconclusive probe behavior, validate nested text metadata and clean temporary adapter merges on refusal. The shared b5270 pin is unchanged. (#1911 by @kok-o in #1955)

- `soup compile` and `soup compile-tools` pass eval suite signal, metric derivation, and iteration bounds to DSPy, TextGrad, and GEPA optimizer runs, TextGrad now refuses without a configured engine and its panel can print `Score: unmeasured`, and add `--task-lm`, `--reflection-lm`, and `--engine` options to `soup compile-tools` (#1810 by @Nari-Ab in #1956).

- GGUF export inspects safetensors headers and passes --no-mtp when declared MTP tensors are absent, refusing unsupported converters before conversion while preserving real MTP weights (#1912 reported by @feloau, fixed by @kok-o in #1957).

- `soup can run` now removes the directory it extracted a can into when a later step
  fails (env capture outside cwd or into a missing directory, a deploy target with no
  `*.gguf`, a training subprocess error), instead of leaving it in the system temp
  directory. A caller-supplied `--extract-dir` and a run that ends with a nonzero
  training exit are left in place (#1464 by @ShivanshShukla in #1960).

- **Layer streaming now reports mixed direct-I/O stripes accurately.** The ready line shows direct-I/O drives as `k/N`, runtime stats expose each root's mode, page-cache fallbacks name the affected root and operating-system reason, and RAM-tier runs no longer print disk read-ahead guidance (#1615 by @Samearth17 in #1962)

- `soup loop watch` now refuses a thumbs database whose `thumbs` table lacks a column the harvest reads (`prompt`, `response`, `thumb` or `ts`) before the loop starts, with the same exit 2 as a missing or unreadable database. Before, the pre-flight only checked that the table existed, so a hand-made table with other columns passed it and the first iteration failed, which under `--detach` ends without a message (#1440 by @Srinivasan8888 in #1966).

- `soup loop watch --pre-wired` now trains on a harvest of one or two preference pairs: the generated loop DPO config sets `data.val_split: 0`, where the schema default 0.1 used to leave a one-pair harvest with no training rows and `soup train` exited 1. When training is skipped, the saved iteration record now says why, with `no_preference_pairs` and the count, or `pairs_file_missing` when the pairs file is absent. Before, the iteration only recorded a skip (#1909 by @Srinivasan8888 in #1968).

- `training.loss_spike_recovery_max_attempts`: persist and enforce attempt counts across re-launches via `spike_recovery.json`, capping recovery when the budget is reached (part of #808 by @Nari-Ab in #1971).

- Repository tooling, not part of the package: the issue claim board respects completed and
  released claims, keeps active scoped owners across unrelated edits to old comments, and
  recognises leading Taking requests when it links ready or draft pull requests. A completion
  in the same paragraph also clears an uncertain repeated acceptance (#1974).

- The RL checkpoint manifest now reports `has_ref_model: false` and `has_rollout_buffer: false`, because no save writes reference-model or rollout-buffer state. Before, it copied `training.rl_checkpoint_include_ref_model` and `training.rl_checkpoint_include_rollout_buffer` from the config, so a checkpoint holding only the adapter and optimizer claimed both. Setting either flag to `true` is now refused at load as a staged field ("Delete the line."); the two flags were staged after v0.76.0, so no release warned about them first (#808 by @Srinivasan8888 in #1980).

- **The Unsloth backend now honours `training.gradient_checkpointing` instead of silently ignoring it.** It was hardcoded to Unsloth's checkpointing kernel, so `training.gradient_checkpointing: false` changed nothing on the Unsloth path. The flag now resolves through the same rule as the HF path: `false` (the schema default) disables checkpointing entirely, `true` (or a tier) enables Unsloth's optimized kernel, and layer streaming keeps it off since it already checkpoints every layer. Set `training.gradient_checkpointing: true` to keep the optimized Unsloth checkpointing (#1981 by @SpinachDigital).

- `soup sweep --strategy random` no longer hangs when a `--param` list or a `--sweep-config` value list names the same value twice (`lr=1e-5,2e-5,1e-5`, or `1e-5,0.00001`): it now draws from each list's distinct values, plans each distinct arm once, and always finishes, `--dry-run` included (#2007 by @MohammadHijjawi97).

- **Keep the claim board with the accepted owner when a competing take is declined (#2010).**
  A maintainer's explicit reaffirmation is checked against the matching open pull request,
  preserving that author's claim date and PR link. Declined requests no longer appear as
  unconfirmed owners; quoted or withdrawn decline instructions do not reject a request.

- Refuse vision/audio distillation at config load and in `soup doctor` instead of accepting modalities the text-only trainer ignores; preserve supported token/sequence text distillation and document the boundary. (#1186 item 1 by @kok-o in #2014)

- Refuse Unsloth vision/audio SFT at config load instead of silently using transformers, report the supported backend in `soup doctor`, and correct both inline and file-backed multimodal templates. (#1186 item 2 by @kok-o in #2015)

- Prevent `soup ingest`, `soup prune-prompt`, and `soup data best-of-n` from writing outputs that alias their input files, including hard-link aliases. (#1422 by @Martini147 in #2018)
- Reject overlapping effective RA-DIT stage output directories before training starts.
- Publish prune-prompt output atomically and validate best-of-n checkpoints before truncating incomplete journal tails.

- Refuse `training.stream_layers` with `training.use_cut_ce` because streamed setup never installs the resident CCE patch; preserve the conservative full-logits budget. (#1206 by @kok-o in #2023)

- Quality scans share the multi-format expectations extractor, see Alpaca/preference/ShareGPT text and prompt benchmark rows, and report unscanned rows instead of silently reporting them clean (#1427 by @kok-o in #2037; following @abdulbasit-ogunrinu's agreed plan).

- Canonicalize ship provenance suite order so re-stamped evidence passes strict replay for non-alphabetical and default suite selections (#2035, reported by @abdulwaarith0, fixed by @kok-o in #2038).

- **The layer-streaming disk pre-flight no longer under-counts a fused MoE cache under `quantization: 4bit`.** It charged the NF4 rate to every parameter, but the default `fused` expert layout keeps the stacked routed experts at the stream dtype, so for Moonlight-16B-A3B (`deepseek_v3`) it charged 8.23 GB and the sharder then wrote 30.60 GB: a disk could pass the check and fill mid-shard. The cache is now sized from the source safetensors headers (no weight is loaded), tensor by tensor, by the same storage rules the sharder writes with, including the exact NF4 sidecar bytes; for Moonlight it lands within 0.01% of the cache written in both `fused` and `per_expert`. A striped cache splits that same figure over its drives. A dense base under `quantization: 4bit` is now charged more, because its embeddings, head and norms stay at the stream dtype (Llama-3.1-8B 4.14 GB -> 5.70 GB, Llama-3.2-1B 0.64 GB -> 1.03 GB), so a run that passed by less than that margin is now refused. A checkpoint the sharder would refuse is refused before anything is copied. When the files cannot be listed or a header cannot be read, the old rate is charged and the panel says "approximately", as it now also does for a qwen4_exp base that is not an oQ checkpoint. (#2039)

- Prevent Transformers threaded weight-loading SIGABRT crash on Apple Silicon MPS by setting default `HF_DEACTIVATE_ASYNC_LOAD=1` in `soup train`, `soup sweep` and training benchmarks, and surface status in `soup doctor` (#2076 reported by @feloau, fixed by @Nari-Ab in #2040).

- `soup profile` and the MCP `tool_profile` now correctly recognize full fine-tuning runs (`lora.r: 0`, Spectrum, LISA, PRM, and classifiers without LoRA), budgeting the whole model parameter count, optimizer state, and gradients instead of reporting 0 trainable parameters and 0 GB optimizer memory (#1423 by @Nari-Ab in #2041).

- `backend: mlx` no longer silently ignores `data.chat_template` on SFT, DPO or ORPO: the MLX
  trainer renders the model's shipped template only, so a prefix-stable
  override was dropped and the run failed the mask's prefix-stability check
  deep inside training. The combination is now refused at config load
  (#2075 reported by @feloau, fixed by @vaputa in #2051).

- `soup train` no longer stalls on the interactive `Start training?` confirm
  when stdin is not a TTY (pipes, CI, services): the run now refuses up front
  with a message naming `--yes`. Answering the confirm by piping `y` into
  stdin is replaced by that same refusal — pass `--yes` instead
  (#2077 reported by @feloau, fixed by @vaputa in #2052).

- `rollout_backend: ruler` now names `openpipe-art`, the package RULER ships in, instead of
  `ruler-eval`, which is not on PyPI, and no longer treats PyPI's unrelated `ruler` grammar
  library as the backend being installed (#2046 by @Rahul-cs50 in #2059).

- **GRPO gradient watchdog now identifies which parameters went non-finite in warning logs and trainer state (#342 by @here-2007 with @harshitthek in #2060).** `GRPOStabilityCallback` inspects `named_parameters()` (falling back to `parameters()`) when a non-finite gradient step is detected, reporting the count and first few offending parameter names in the console/logger warning. Persists per-step non-finite parameter records and the cumulative run summary (`nan_param_count`, `nan_param_names`) in `state.log_history` and `trainer_state.json`. Tested on CPU with NaN injection on a real module.

- **`soup adapters check` and `soup adapters diff` no longer crash on a bfloat16 adapter.** Both read the weights through `safe_open(framework="numpy")`, which cannot return bfloat16 (numpy has no such dtype), so an adapter saved from a bf16 run failed with `data type 'bfloat16' not understood`, or with "Corrupt or unreadable safetensors weights" in `check`. They now share one reader that parses the header with `read_header` and widens bfloat16 to float32 exactly (it is the top 16 bits of a float32); the other dtypes adapters store are read as before, a dtype numpy cannot hold (FP8, and the U16/U32/U64 tags `read_header` does not map) is refused as an unsupported dtype naming the tensor, and the size cap, tensor-count cap and symlink checks are unchanged (#2058 by @drakeo338 in #2063).

- Refuse vision row formats (llava, sharegpt4v) on preference tasks at config load, with "has no vision-preference row path; use a text preference format instead", and drop the unsupported pixtral-dpo recipe (#1867 by @Vishalkumarsingh6428 in #2064).

- Audit-log lines written by older versions now get the same webhook URL masking as new lines, applied once when the log is next written. A pass that fails is not repeated by every command (the next attempt waits a day), and temp files an interrupted pass left behind are removed by the next one (#2065).

- **The layer-streaming RAM check that runs before sharding (`stream_source: ram`) now charges what the RAM tier will hold.** It charged the flat NF4 rate on the source files' size on disk. That under-counted a fused MoE base under `quantization: 4bit` (Moonlight-16B-A3B: about 8.2 GB against a 30.6 GB cache, whose stacked routed experts stay at the stream dtype) and an FP8 block source about 2x (one byte per weight on disk, dequantised in the cache). Such a base was refused only by the exact check after sharding, after minutes of work and about 30 GB of disk writes. The check now reuses the disk pre-flight's estimate from the source safetensors headers, so the headers are read once. It charges what the check after sharding measures: the decoder-layer files, the embedding and head and the resident extras (not the `per_expert` experts files, which stay on disk on every tier). It refuses by that check's rule, 70% of free RAM or 55% of physical RAM, so a base the exact check would refuse is refused before anything is sharded. When the headers cannot be listed or read, it charges the old rate and the refusal says "approximately". `stream_source: auto` is unchanged: nothing before sharding decides or warns on a RAM figure. (#2068)

- **`target_modules: auto` now adapts the query projection of a DeepSeek-V3 base with `q_lora_rank: null`, such as Moonlight-16B-A3B.** DeepSeek-V3's attention builds a single `q_proj` when the config's `q_lora_rank` is `null` and the low-rank pair `q_a_proj` / `q_b_proj` otherwise, but the `deepseek_v3` entry named only the pair. peft refuses an attach only when no target matches, so on such a base the key/value and output projections took the adapter and the query path trained none, without a warning. The `deepseek_v3` and `kimi_k2` entries now follow `q_lora_rank`, read from the config the model or the layer-streaming setup has. A base whose config sets an integer, as DeepSeek-V3 and Kimi K2 do, or does not carry the field, gets the same targets as before. `glm_moe_dsa` is unchanged: its config does not accept `q_lora_rank: null`. `kimi_k25` is unchanged too, so a Kimi K2.5 base with `q_lora_rank: null` would still get no adapter on `q_proj`. No shipped K2.5 or K2.6 config sets it, and because the vision tower has its own `q_proj` the entry would have to be a regex, which `moe_lora: true` discards. (#2071)

- **Layer streaming with `moe_lora: true` now keeps list attention targets, as resident training does, and its VRAM forecast charges what attaches.** The streaming setup handed the `moe_lora` helper no targets and then replaced the resolved `target_modules` list with its answer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, the expert FFN names and the fused expert parameters). Every name that list does not repeat lost its adapter: on DeepSeek-V3, whose attention has no `k_proj` / `v_proj`, attention kept only `o_proj` and lost `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` and `kv_b_proj`; on Qwen3.5-MoE the linear-attention `in_proj_qkv` / `out_proj` were dropped; and an explicit `lora.target_modules` list was dropped entirely. The setup now resolves the targets before its meta-skeleton probe and passes them to the same helper the resident trainers use, so when the targets are a list (an explicit list, or what `auto` resolves to on DeepSeek-V3 and Qwen3.5-MoE) a streamed run adapts their union with the MoE targets, in the same order as a resident run of the same config. A string target (`auto` on qwen4_exp resolves to `"all-linear"`, or a regex) reaches the same helper on both paths, where it keeps its meaning without the MoE routers. The pre-flight VRAM forecast now counts that union and one adapter per routed expert for each fused expert tensor, instead of the configured value: `target_modules: ['o_proj']` was charged one target per layer while the union and every routed expert attached. Runs without `moe_lora` are unchanged. (#2072)

- Generated fine-tuning gates require every training-data row to convert by passing `--min-valid-fraction 1.0`; standalone validation and existing workflows are unchanged. (#2079 by @kok-o)

- **Under `training.stream_layers`, a LoRA target on the streamed embedding or head is refused
  while the streamed model is built, whatever `target_modules` spelling matched it, and a tied
  checkpoint is no longer refused (#1048 by @ybai08 in #2082).** The old check ran when the
  config was read and matched only the literal names `lm_head` and `embed_tokens`. On an untied
  checkpoint a regex such as `'.*(q_proj|lm_head)'` or a dotted name such as
  `model.embed_tokens` loaded; the regex then failed at its first default save
  (`Cannot copy out of meta tensor`), and under `task: dpo` it failed in peft during setup
  (`Target module StreamedLargeLayer(...) is not supported`). A tied checkpoint streams neither
  module, and its adapter saves and loads back unchanged, but it was refused anyway. The check
  now runs after the adapter is attached and before any streaming buffer is allocated. On an
  untied checkpoint it names the module or modules and the current `target_modules`, and says to
  change them or train without `stream_layers`. A tied checkpoint trains and saves with the
  defaults. Saving or resuming an adapter on an untied checkpoint's streamed embedding or head
  is still not supported.

- `soup train --cloud lambda --cloud-submit` now starts the local controller with an isolated Python interpreter (`-I`) and only the environment it needs: the Lambda settings, the variables `ssh`/`scp` use, and the proxy, certificate, SSH-agent and library-path settings. The controller replaces output characters its encoding cannot show instead of stopping, and the printed manual command is `python -I <stub>`. (#2091)

- **Preserve preference pairs with `data.format: auto` (#2090 by @kok-o in #2093).**
  DPO, IPO, SimPO, ORPO, BCO, preference and reward-model tasks resolve auto to
  the pair converter, including `messages` prompts and mixed answer layouts.
  Training and `--dry-run` reject missing pair columns before constructing a
  trainer, with the loaded format and a `data.format: dpo` hint. SFT ChatML
  and KTO's unpaired layout keep their existing behavior.

- **`moe_lora: true` keeps a string `target_modules` and adds the MoE targets on top of it, as it already did for a list.** `resolve_moe_lora_targets` merged the MoE names into a list target (#1590) but replaced a string with the MoE list (`q_proj`, `k_proj`, `v_proj`, `o_proj` and the expert FFN names) plus the fused expert parameters, and a string means something to peft. `"all-linear"`, which `target_modules: auto` resolves to on qwen4_exp, adapts every linear layer but the output head, so a qwen4_exp base with `moe_lora` lost the Gated DeltaNet `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` / `out_proj`, the QSA indexer's `index_qk_proj`, the hyper-connection mixers and the PLE `key_proj` / `value_proj`: 10 adapted modules on a tiny two-layer model, where 32 now attach. Any other string is a regex peft full-matches against each module name, and it was dropped outright. Now `"all-linear"` (in any casing peft accepts) is kept as it is, since it already reaches every module the MoE names reach, and only the fused expert parameters are added. A regex becomes one regex, `(?:<your regex>)|(?:<the MoE names>)`, so peft adapts exactly what either half matches; a leading global flag such as `(?i)` stays with your half, and a regex that does not compile is refused at setup, naming `training.lora.target_modules`. Under `moe_lora` a string never adapts a MoE router or shared-expert gate (a module whose output is routing logits, found from the model's structure, DBRX's and JetMoE's wrapped routers included), even one your regex names: they go to peft's `exclude_modules`. Only where peft would still adapt a router weight with them excluded does the string become an explicit list of what it means without them: `all-linear` on the families peft converts from the transformers-4 layout (qwen3_moe, deepseek_v3, ...), where peft adds the router to it as a parameter `exclude_modules` does not reach, a regex reaching a non-router named `gate` there, and VeRA, whose config has no `exclude_modules`. To adapt a router, name it in an explicit list. A regex that matches no module is warned about. The `ScatterMoE LoRA:` setup line says which of these happened. Lists are unchanged, and so is every run without `moe_lora`. This applies to every resident trainer and, since #2072 hands it the streaming setup's resolved targets, to layer streaming. (#2095)

- **Layer streaming refuses a quantised checkpoint it cannot dequantise, by name, before a byte of
  the layer cache is written.** A Kimi-K2-Thinking-shaped `deepseek_v3` checkpoint
  (compressed-tensors `pack-quantized` INT4: int32 `weight_packed`, `weight_scale`, int64
  `weight_shape`) used to be accepted: the sharder cast every int32 word and int64 shape to the
  stream dtype, wrote the whole cache, and the streamed build then failed with "shard is missing
  decoder weights". A source whose names still matched, such as bitsandbytes LLM.int8() (an int8
  `weight` beside its `SCB`), did not fail at all: the int8 codes were streamed as weights. Now
  refused, by the sharder, the disk pre-flight and the trainer setup in the same words: a
  `quantization_config` whose `quant_method` is anything but `fp8` (compressed-tensors in any
  format, GPTQ, AWQ, bitsandbytes, HQQ, AQLM, EETQ, torchao, quanto and every other quantiser
  transformers registers, or `load_in_4bit`/`load_in_8bit` alone), refused from `config.json`
  before any weight is resolved; and any tensor stored as an integer or packed type (I8, I16, I32,
  I64, U8, U16, U32, U64, BOOL, F4), or named like a quantiser's companion (`weight_packed`,
  `weight_scale`, `qweight`, `qzeros`, `g_idx`, `SCB`, `absmax`, `quant_map`, `W_q`, ...). The
  message names the format, says layer streaming cannot dequantise it, and points to a bf16 or
  FP8-block copy. Still read: plain floating point, DeepSeek-V3's block FP8, qwen4_exp's oMLX/oQ
  affine bundles, and the integer buffers a supported model really has (qwen4_exp's PLE n-gram
  hashing tables, `position_ids`). A pre-flight that met a stored type its header reader has no
  spelling for (U32, U16, F4, ...) now refuses that tensor instead of sizing the cache at a guess.
  The sharder checks the source before it creates the cache folder, looks a cache up or
  quarantines an index, so a refusal leaves an existing cache untouched and a cache an older Soup
  sharded from such a source is not reused. A `quantization_config` that is not a mapping is
  refused as malformed, and one under `text_config` is read too. (#2103)

- `soup doctor --config` checks the config before the `All checks passed!` line, so a refused,
  missing or unreadable config no longer gets that line before doctor exits 2; the System, GPU,
  Resources and Dependencies tables still print first (#2100 by @Nari-Ab in #2106).

- `soup ui --public`, `--host 0.0.0.0` and `--host ::` now print and open a loopback URL
  (`http://127.0.0.1:<port>` or `http://[::1]:<port>`) instead of the bind address, which
  Windows cannot connect to; the server still binds to all interfaces and the `--public`
  phone URL keeps the LAN address. An IPv6 host such as `--host ::1` is now bracketed in the
  URL (#2054 by @Rahul-cs50 in #2110).

- **`mini_mmlu` no longer sits at a 1.000 ceiling (#1192 by @AaronProbha18 in #2113).**
  The suite grows from 26 to 40 rows with 14 four-option questions, many with a
  common misconception as a distractor; Qwen2.5-1.5B-Instruct measures 0.850
  against 0.350 on SmolLM2-135M-Instruct. The answer key no longer favours one
  letter, so a model that always answers "B" drops from 0.462 to 0.300, and the
  bundled scorer provenance revision moves to 7 so pre-change baselines are not
  silently compared on the new scale.

- Respect explicit `training.unlearn_alpha: 0.0` for pure forget loss in unlearn training (NPO, SimNPO, and RMU) instead of silently falling back to 1.0 (#2108 by @Nari-Ab in #2114).

- **`task: simpo` on the transformers backend caps the prompt as DPO and ORPO do (#2112 by
  @dryn85-max in #2116).** TRL 0.29.1's CPO tokenizer cuts only the answers, so a long prompt
  reached the model whole, beyond `data.max_length`. After TRL tokenizes, the prompt is now cut to
  `data.max_length // 2` tokens, keeping its end, and the answers to the room left, so no sequence
  exceeds `data.max_length`. A prompt longer than `data.max_length // 2` tokens is cut even when
  the whole row would fit; a row whose prompt is at most `data.max_length // 2` tokens and which
  fits is unchanged.

- Drop KTO rows with a malformed label, with a warning, instead of silently training nulls,
floats or containers as desirable or undesirable rows. Accept booleans, the documented normalized
true/1/yes and false/0/no strings, and exactly the integers -1, 0 and 1. Validation
and training report the offending label through the shared converter; `soup train` warns with
the first such row and its count and refuses only an empty result (#2119 by @kok-o in #2126).

- **Resident Kimi K2 configs are refused at config load instead of failing after the
  model's remote code and weights start loading (#2104 by @kok-o in #2130).** The four
  known Hugging Face Kimi K2 bases (Kimi-K2-Base, -Instruct, -Instruct-0905 and
  -Thinking), loaded resident on the transformers backend, passed validation and then
  failed at model loading. They now stop at load with `Resident Kimi K2 is unsupported on
  Transformers 5.x`. For Base and Instruct the message names the supported route,
  `training.stream_layers: true` (#2107), its tokenizer prerequisites
  (`--trust-remote-code` and `pip install tiktoken`) and the streaming guide; that route
  is verified on CPU against a tiny K2-shaped checkpoint only, not by full K2 or GPU
  training. Thinking and Instruct-0905 have no supported route yet, and K2.5 / K2.6 are
  outside this check. The two affected recipes are marked unavailable: `soup recipes
  show` still prints them, and `soup recipes use` refuses before writing a config, with
  exit code 3.

- Honor `training.gradient_checkpointing` in distill, classifier (`classifier`, `reranker`, `cross_encoder`), ASR and unlearn, which ignored it and always trained with checkpointing off. Distill, classifier and ASR now pass it to their HF training arguments; unlearn, which has its own training loop, turns checkpointing on for its policy model. Every enabled tier is full checkpointing on these tasks, and the default `false` is unchanged. (#797 by @ybai08 in #2140)

- **`training.freeze_layers` / `training.freeze_ratio` did nothing on a LoRA run: the freeze was set before `get_peft_model`, which then attached a fresh trainable adapter to every targeted module in every layer, the frozen ones included (#1432 by @AmirF194 and @VanshSharmaPES in #2145).**
  The SFT trainer now passes the layers the freeze plan left trainable to peft as `layers_to_transform`, so a frozen layer gets no adapter. On a 2-layer Llama with LoRA on `q_proj` / `v_proj`, `freeze_layers: 1` (or `freeze_ratio: 0.5`) takes the trainable LoRA parameters from 1,024 to 512 and layer 0 no longer changes. A config that sets `freeze_layers` / `freeze_ratio` together with LoRA now trains fewer layers than it did before, which is what the option always documented.
  Shapes peft cannot restrict by layer are refused when the model is built rather than ignoring the freeze: a string `lora.target_modules` (a regex, or `all-linear`, including the value `auto` resolves to for some architectures), a list target that resolves only outside a numbered decoder layer (such as `embed_tokens`), `lora.target_parameters`, `lora.use_vera`, and a plan that freezes every layer. A top-level `lm_head` target is accepted and keeps its adapter. Full fine-tuning (`lora.r: 0`) is unchanged.

- **`training.lora.rank_pattern` and `alpha_pattern` are refused with `backend: mlx` instead of being silently ignored (#1996).** The MLX trainer builds every adapter, for SFT, DPO, ORPO and SimPO alike, from the one scalar `lora.r` and `lora.alpha`, so a config that asked for a per-module rank or alpha loaded, passed `soup train --dry-run` and trained a uniform adapter, and `soup doctor --config` listed nothing for it. Now refused at config load, naming each field the config set, `backend: mlx`, and the two ways out: remove the pattern, or use `backend: transformers`, which applies it. A task MLX cannot run, `lora.r: 0` and MLX DoRA keep their own, earlier refusals. An empty `{}` or `null` sets nothing and still loads, and `transformers` configs are unchanged. This is a stop-gap: it goes away when the MLX trainer honours the patterns. (#2150)

- Accept UTF-8 BOMs in user-supplied command inputs (#2136 reported by @yunaremaia, fixed
  by @vaputa in #2155), preserving the first
  JSONL record and prompt text. Covers agent predictions, tokenizer corpora,
  benchmark prompts, draft inputs, edit and shrink calibration data, diagnose
  evidence, Can attestations, generation context and Autopilot's local model
  config probe. Attestation verification accepts BOM-prefixed statements and
  signature sidecars while retaining canonical signing and trusted-key checks;
  embedded U+FEFF and UTF-8 output encoding are unchanged.

- **A streamed `soup train` on Linux (WSL2 included) now gets the `expandable_segments` allocator
  hint (#2149 by @ybai08 in #2156).** The hint is set only while CUDA is not yet initialised,
  and `soup train` initialised CUDA when it detected the device, before the streamed setup set
  the hint, so every streamed CUDA run printed "expandable_segments allocator hint not enabled:
  CUDA was already initialised before Soup could set it". `soup train` now sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before it detects the device, for streamed
  runs on Linux. A value you exported is kept, and if that value is why the hint is not on, the
  line now says `PYTORCH_CUDA_ALLOC_CONF is set to something else`. Runs without
  `stream_layers`, macOS and Windows are unchanged.

- **`training.lora.rank_pattern` and `alpha_pattern` are refused with `backend: unsloth` instead of being silently ignored (#2153).** Every unsloth setup, for SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, BCO, preference, pretrain, embedding and TTS alike, goes through one loader that gives `FastLanguageModel.get_peft_model` a single scalar `lora.r` and `lora.alpha` and has no parameter for a pattern, so a config that asked for a per-module rank or alpha loaded, passed `soup train --dry-run` and trained a uniform adapter. Now refused at config load, naming each field the config set, `backend: unsloth`, and the two ways out: remove the pattern, or use `backend: transformers`, which applies it. A task with no unsloth setup, SFT with a vision or audio modality, and `lora.r: 0` keep their own refusals. An empty `{}` or `null` sets nothing and still loads, and `transformers` configs are unchanged. This is a stop-gap: it goes away when the unsloth loader honours the patterns. (#2165)

### Security

- Raise the pypdf floor for the `documents` extra to 6.21.2 (`pypdf>=6.21.2`).

- Bound judge replies to 1 MiB and HTTP generation/Magpie replies to 8 MiB while
  streaming decoded bytes, including gzip expansion. Oversized replies are
  refused before JSON parsing, close the response stream, and do not log the
  response body. Generation/judge errors identify the endpoint setting and cap;
  Magpie drops the affected sample and continues within its attempt limit
  (#1819 by @kok-o in #1929).

- `for_terminal` and `strip_control` now also remove bidi controls (U+202A-U+202E, U+2066-U+2069, U+061C, U+200E, U+200F) and invisible separators (U+200B, U+2028, U+2029, U+FEFF) from dataset, adapter and evidence text before it is printed, so a crafted row can no longer reorder or hide what a preview shows; ZWJ and ZWNJ are kept for Arabic, Persian, Indic and emoji text (#1861 by @aadityad12 in #1953, reproduction by @akkupratap323).

## [0.76.1] - 2026-10-14

### Changed

- **The README and the performance docs now state the 8B NF4 figure with its SM clock and give the post-#331 row, and no longer say the re-measurement is pending (#1207, in #1699).** Llama-3.1-8B-Instruct NF4 on an RTX 3050 Laptop 4 GB reads **119.6 tok/s at 952 MHz in a 3.32 GB peak** before the #331 repair and **208.6 tok/s at 1935–1957 MHz in a 2.40 GB peak** after it, the second row measured by @umran666 (#1103). The two are not directly comparable: 59% against 68% of the same-session GEMM ceiling is about 14% slower per unit clock, so the raw 1.74x gain is the clock, not a speedup, and large-layer streaming changed the memory pattern between the rows, so even the fraction is not like-for-like. Changed: the headline, the GIF alt text and caption and the preprint bullet (Citing Soup) in `README.md`, and the table row, the headline and the large-layer paragraph in `docs/performance-and-quantization.md`. The four README translations (`README.tr.md`, `README.ar.md`, `README.ja.md`, `README.zh.md`) carry the same sentences, with every number, link and code span kept identical, and are re-stamped; their prose is a machine-assisted draft for the translation maintainers to polish. The preprint (DOI 10.5281/zenodo.21771064) is not edited in place; #1207 tracks its amendment as a new version.

- **The "What's New" block of `README.tr.md`, `README.ar.md`, `README.ja.md` and `README.zh.md`
  now describes v0.76.0 (#1825).** All four still described v0.75.0. The new text follows the
  block in `README.md` bullet by bullet, with every inline code span, command, flag, config key,
  URL, link target, number and benchmark file name kept identical; the prose is a machine-assisted
  draft for the translation maintainers to polish. This change moves no `synced-from` stamp,
  because the section is exempt from them; the stamps of the four files were re-taken by #1699 in
  the same release.

- soup.yaml and every other YAML file Soup reads go through one loader that applies the same nesting and expansion limits (100 levels of nesting; one node per source byte plus 10,000 for anchors, aliases and merge keys, with merge keys chained at most 100 levels), and each reader keeps its own size cap: 1 MiB for the training-config readers and for readers that had no cap before, 256 KiB for recipes, `export --quant-config` files, RA-DIT configs and sweep files, 5 MiB for agent specs, 16 MiB for can manifests, 64 KiB for README frontmatter, 4 MiB for the config read by `soup ship`. A soup.yaml that is over a limit, cannot be read or is not valid YAML is refused with one line naming the file and the problem; the other readers report through their own handlers. The audit log now finds credential options from the command tree and also masks long options by name, so the values of `--slack-url` and `--discord-url` are masked and `--min-tokens`, a length threshold, is no longer masked (#1827).

- **The Typer floor is `typer>=0.15.4`, and one CI cell now starts the CLI on the declared Typer
  and Click floors (#1858).** `typer>=0.9.0` was declared while no command could run below Typer
  0.12.4: the command tree did not build (`RuntimeError: Type not yet supported: int | None`), so
  `soup --help` and `soup version` failed with everything else. Typer 0.12.4 - 0.15.3 start the
  CLI only next to Click 8.1 or older and do not ask for it, so a resolver paired them with a
  Click on which every help page raised. 0.15.4 is the lowest Typer that works with any Click it
  can be installed with; `click>=8.0.0` was measured again and holds. A fresh install is
  unchanged, since a resolver already picks the newest Typer. An environment that holds an older
  Typer now has it upgraded, or the conflict named, at install time, where before the install
  succeeded and the first command failed; `soup doctor` reports such a Typer as outdated. The
  `transformers-floor` CI job installs exactly the two declared floors, builds the command tree,
  renders every help page and runs `soup version` under them, and a test fails when a declared
  floor and that job's pin differ (#1802).

### Fixed

- **`task: dpo`, `ipo`, `bco` and `preference` train on the bundled DPO examples again.** Both files (`examples/data/dpo_sample.jsonl`, `examples/data/chat_preferences.jsonl`) and any dataset written the same way, a string `prompt` beside `chosen` / `rejected` message lists that repeat the user turn, failed inside trl after the model had loaded, with an error that changed with `PYTHONHASHSEED` (`'list' object has no attribute 'endswith'`, or a `TypeError`). The `dpo` and `kto` converters now hand the trainer every preference row in one layout: all strings, or all message lists with the prompt turns in `prompt` only. `reward_model` no longer trains on a text with the question in it twice, and `orpo` and `simpo` prepare the same tokens as before. A row that cannot be written that way (one answer a string and the other a message list, a malformed message, no prompt to be found) is dropped at load with a reason that names the field, and a dataset that mixes text rows with conversational ones loads as conversations, with a note. A string `prompt` beside message-list answers is not read when the two answers share leading turns: those turns are the prompt, as `orpo` and `simpo` already read such a row, so a question written only in that string has to be in the conversations as well (#1849 by @dryn85-max in #1866).

- A single-string `data.train` that names a Hub dataset with a dot in its name (`teknium/OpenHermes-2.5`, `org/name.v1`) is now loaded from the Hub instead of failing with `Data file not found`: the loader read the dot as a file extension. A single string now takes the same route as the same name inside a `data.train` list: a supported file extension (`.jsonl`, `.json`, `.csv`, `.parquet`, `.txt`) means a local file, a `://` string a remote URI, anything else a Hub id. A path with an unsupported suffix that does not exist (a typo such as `data/train.jsnol`) therefore goes to the Hub lookup too, instead of `Data file not found`, while an existing file with an unsupported suffix (`data.tsv`) still gets `Unsupported file format` (#1889).

## [0.76.0] - 2026-10-12

### Added

- **Added `soup data clean` for automated dataset sanitization and repair (#718 by @Deadpool2000).**
  Provides deterministic, non-destructive cleaning of fine-tuning datasets across all supported formats (`chatml`, `alpaca`, `sharegpt`, `dpo`, `kto`, `tool-calling`). Runs safe character sanitization (C0 controls, zero-width spaces `\u200b-\u200d`, BOM) and empty assistant turn pruning by default, while placing heuristic transformations behind explicit opt-in flags (`--strip-boilerplate`, `--repair-code`, `--repair-json`, `--prune-echo`). Accurately tracks all modifications, strictly preserves unknown metadata keys and custom roles non-destructively, guarantees input file immutability, refuses unsafe in-place or unforced overwrites, and supports dry-run inspection and structured machine-readable JSON output.

- **Token-chunked checkpointed KL distillation (#722 by @kok-o in #743).** Added
  `training.distill_chunk_size` and `training.distill_checkpoint` knobs for `task: distill`.
  Response-only token selection (`labels != -100`) eliminates unmasked prompt and padding
  tokens from retained autograd memory, while token chunking and optional non-reentrant
  activation checkpointing (`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`)
  discard intermediate `log_softmax` and divergence tensors during forward and recompute
  them in backward, reducing retained autograd tensor bytes and bounding transient peak
  activations on long sequences and large vocabularies. The divergence kernel computes
  and accumulates in FP32 precision, shifting baseline bf16 distillation loss by up to ~1 ULP (~0.4%).

- **A test now fails when any typer command is called as a plain function with a parameter left unfilled (#752 by @dchaudhari7177 in #757).** Called directly rather than through typer, an unfilled parameter keeps its `typer.Option`/`typer.Argument` default, a truthy `OptionInfo`. That is how `soup eval auto` crashed after saving its results in #752, fixed at its two call sites by #754. `tests/test_typer_defaults_do_not_leak.py` walks the whole `soup_cli` package, resolves each call through `from X import name` to the command it names, and matches positional arguments to parameters by declaration order, so the *next* unfilled parameter anywhere fails the suite and the report names only the parameters actually left unfilled. A synthetic command proves the reporter can still report, so the guard cannot be silenced unnoticed.

- **`soup adapters audit` checks whether a finished run did what its config asked (#762 by @Srinivasan8888 in #763).** The MLX path records its effective settings in `adapter_config.json` — optimizer, scheduler, warmup and total updates, weight decay, gradient clipping, accumulation, checkpointing and the response-only mask — because each of those was once accepted and silently dropped (#683, #684, #685, #686, #749). Nothing read that record back, so answering "did my run do what I asked?" meant opening two files and comparing by eye. `soup adapters audit ./out --config soup.yaml` prints a per-setting table of asked-versus-ran, explains each divergence, and exits `2` when any is found -- `0` on agreement and `1` on a usage or read error, so a CI gate can tell a verdict from a mistyped path (matching `soup ship` and `soup shrink`); `--json` emits the same result for machine consumption. A setting the record cannot speak to is reported `unknown` and never as agreeing — a false clean bill is worse than no audit — and `unknown` does not fail the command, since an older adapter simply predates the keys. The motivating case is a warmup that rounds away: `warmup_ratio: 0.03` over twelve optimizer updates is zero warmup steps, which the trainer warns about once and the adapter records permanently. Response-only masking is audited by its recorded *effect* (`mask_prompt` / `response_token_mask`) rather than by the request the record echoes back, so a plain-text run that asked for masking and silently got none — #683 itself — is reported as a divergence instead of a clean bill. On the transformers path only LoRA shape can be checked, because PEFT writes its own `adapter_config.json`, and the command says so rather than implying a clean bill it cannot give.

- `soup train` accepts `--nodes`, `--node-rank`, `--master-addr`, and `--master-port`
  for multi-node Accelerate launches. `--gpus` is the per-node count, so eight GPUs
  on two nodes launches 16 workers. Launch settings are validated before training:
  `--master-addr` must parse as an IPv4/IPv6 literal or an RFC 1123 hostname, so a
  host with the port glued on (`head:29500`) is refused rather than exported as
  `MASTER_ADDR` on every node. Multi-node NCCL defaults allow InfiniBand and preserve
  user overrides. `--no-reexec` and `--dry-run` show the command without starting
  workers. (#40 by @tristangrech in #853)

- **Added ready-made `kimi-k2.5-dpo` and `kimi-k2.5-grpo` recipes for moonshotai/Kimi-K2.5
  (#275 by @Areen-09 in #872).** Completes the task trio for the base model,
  mirroring the Kimi-K2.6 DPO and GRPO architectures over the ~1T MoE geometry
  (LoRA r32/a64 for DPO, r16/a32 for GRPO, 4-bit quantization, `moe_lora`,
  `gradient_checkpointing`, `max_length: 8192`). The recipes are not trained — ~1T MoE
  requires multi-node DeepSpeed — so hyperparameters are inherited from K2.6 templates
  rather than tuned for K2.5. Catalog count 167 -> 169.

- **Added ready-made `minimax-m3-dpo` and `mistral-large-3-dpo` recipes
  (#849 by @akkupratap323 in #933).**
  Both mirror their giant-MoE SFT siblings with the catalog's DPO preference-data
  shape, LoRA r32/a64, 4-bit quantization, `moe_lora`, gradient checkpointing, and
  one training epoch. These inherited configurations are intended for multi-node
  use and have not been validated by training the 428B and 675B base models.

- **`soup doctor` now reports GPU architecture and precision capability diagnostics (#829 by @Samearth17 in #947).** GPU diagnostics show the detected NVIDIA compute capability and architecture, whether the installed Torch build includes native/PTX support for that GPU, and hardware/software support for BF16, FP8, and NVFP4. Unsupported GPU architectures now include a CUDA-enabled PyTorch reinstall hint.

- Add the `qwen3.5-35b-a3b-grpo` recipe, completing the Qwen 3.5 35B-A3B SFT, DPO, and GRPO trio. (#847 by @Amix29 in #956)

- **Added optional PEFT LoRA-FA optimizer support (#725 by @webdevsamran in #962).** Freezes LoRA A matrices and trains only B matrices via PEFT's create_lorafa_optimizer, reducing adapter activation memory retention and cutting optimizer state size by 50% during training. Supported for sft, pretrain, and embedding tasks on the Transformers backend.

- **LoRA `rank_pattern` and `alpha_pattern` are now held by a test on every trainer that builds an adapter (#959 by @dchaudhari7177 in #963).** #923 routed all eighteen `build_lora_config` callers through the shared builder, but ten had a behavioural test. `asr`, `classifier`, `distill`, `embedding`, `ppo`, `unlearn` and layer streaming are now driven on CPU with a tiny model and checked for the configured per-module rank (streaming, which never materialises the base, is checked on the `LoraConfig` it streams with), and a scan of `trainer/` fails when a new caller of the builder has no pattern test.

- **Added ready-made `qwen3.6-35b-a3b-dpo` and `qwen3.6-35b-a3b-grpo` recipes for Qwen/Qwen3.6-35B-A3B
  (#275 / #846 by @Areen-09 in #978).** Completes the task trio with the existing SFT sibling
  over the ~35B MoE geometry (3B active, LoRA r16/a32, 4-bit quantization, `moe_lora`,
  `moe_aux_loss_coeff: 0.01`). DPO adopts the `qwen3.5-35b-a3b-dpo` template (`max_length: 4096`),
  and GRPO adopts the `qwen3-30b-a3b-reasoning-grpo` template (`max_length: 8192`,
  `gradient_checkpointing`). Both recipes declare explicit `modality: text`.
  Catalog count 172 -> 174.

- **Added ready-made `qwen3.6-27b-dpo` and `qwen3.6-27b-grpo` recipes
  (#845 by @akkupratap323 in #1015).** Both preserve the dense SFT sibling's explicit
  text modality, LoRA r16/a32, 4-bit quantization, and `auto` batch size with gradient
  accumulation 8. DPO uses preference JSONL data with `lr: 5e-6`; GRPO uses reasoning
  JSONL data with four generations and the accuracy reward. These inherited 27B
  configurations were schema- and CLI-validated but were not trained locally.
  Catalog count 174 -> 176.

- **`soup rewind` names the dataset rows a training loss spike came from (#1000 by @Srinivasan8888 in #1018).** An SFT run now writes `<output>/rewind.jsonl`: one line per micro-batch holding the dataset rows in it, each row's mean supervised-token loss and its supervised-token count. `soup rewind` lists the steps whose loss went non-finite or past twice the median of the previous twenty measured steps, then ranks that step's rows by their share of its summed token loss, so one long malformed row stands out even when its per-token mean is ordinary. On an 8 GB M1 it flagged step 58 of a Qwen2.5-0.5B run and put the poisoned row at 100% of that step's loss. Both backends, controlled by `training.rewind_log` (default on); the recorder switches itself off with one warning rather than record rows it cannot attribute, including on resumed runs and under packing.

- **`soup draft measure` reports the break-even acceptance rate and the best draft length (#843 by @Srinivasan8888 in #1042).** It now also times the draft on its own and reports the latency ratio, the acceptance at which assisted generation stops being slower at the `--num-assistant-tokens` in use, and the modelled best k, all labelled as a ceiling that assumes independent acceptance and excludes framework overhead. A draft no faster than its target is reported as "no acceptance rate pays". `--sweep-k 1,2,3,5,8` measures assisted throughput at each draft length and names the measured best k. The docs no longer say a 70% acceptance rate is where speculative decoding pays: the one pair measured at scale was 81% and ran at 0.48x.

- **`pytest -m gpu` selects the CUDA-dependent tests, and `CONTRIBUTING.md` says how to run and report them (#833 by @Srinivasan8888 in #1050).** Fifteen test modules had their own `torch.cuda.is_available()` probe, and nine had an identical `requires_cuda` skip, so the GPU subset couldn't be selected and was only ever run by accident. `tests/conftest.py` now holds the one probe and skips `@pytest.mark.gpu` tests without CUDA; `tests/test_issue833_gpu_marker.py` fails on a private probe. Runs on real cards are recorded in `benchmarks/gpu-test-runs.md`, seeded with the H100 and RTX 5070 Laptop runs already published.

- **`soup infer` supports opt-in batched causal-LM inference with `--batch-size`; the default remains 1 so existing inference behavior is preserved. fp16/bf16 batching may not be bit-exact with batch size 1.** (#765 by @Samearth17 in #1083)

- **Arabic and Japanese READMEs (#1100 by @Ercaner1988).** `README.ar.md` and
  `README.ja.md` are full translations of the current `README.md`, linked from the language
  banner and registered in `MAINTAINERS` behind the same sync gate as `README.tr.md`. The Arabic
  tables sit in a right-to-left block, and the 13 Latin identifiers that GitHub's bidi
  algorithm rendered reversed inside Arabic text (the concept DOI showed as
  `zenodo.21771064/10.5281`) are wrapped in `<span dir="ltr">`, found by running the algorithm
  over every code span rather than by eye. Spanish, Russian and Chinese are not included until
  someone can verify them.

- **`data.mask_history` now does what it says (#761 by @Srinivasan8888 in #1101).** The
  field was declared, validated and documented with a worked example, and read by no
  module outside the schema, so a config asking to train on the last turn trained on
  every turn. With `train_on_responses_only: true` it now keeps only the **last**
  assistant turn in the loss; earlier assistant turns are masked alongside the user
  and system turns that path already excludes, so it only ever removes tokens from
  the loss. It applies to `task: sft` and `task: distill`, the two trainers that
  build their labels through the shared chat formatter.
- **`data.mask_history` requires `data.train_on_responses_only: true` (#761 by @Srinivasan8888 in #1101).**
  Without it every token trains, including the history the field asks to exclude, so
  the combination is refused at config load with a message naming both. That also
  rules out `train_on_messages_with_train_field`, which is already exclusive with
  `train_on_responses_only`.
- **`backend: mlx` says it ignores `data.mask_history` (#761 by @Srinivasan8888 in #1101).**
  MLX SFT builds its own mask and supervises every assistant turn, so the field has
  no effect there. `soup train` now lists it on its "MLX backend ignores:" line, and
  `soup doctor --config` reports it instead of an all-clear.

- **Added the measured mixed QuEST route as an experimental SFT path (#674 by
  @Shutaru in #1105).** `training.quantization_aware: quest` reconstructs the
  confirmed 168-W4 / 161-A4 / seven-A16 route on its exact 24-block Llama
  topology, calibrates fixed activation clips from the first 32 tokenized
  training rows, and records the full route in a mandatory sidecar on final and
  checkpoint saves. Resume and the Soup-aware loader reject metadata drift.
  The retained result is evaluation-only on one model: this is not strict W4A4,
  packed INT4, upstream QuEST parity, an efficiency claim, or evidence that
  training through the route preserves quality.

- Warn on unconsumed staged config fields with a v0.77 refusal deadline (#808 by @Guts1005 in #1106)

- **Adapter manifests can now be signed and verified with Sigstore keyless OIDC (#185 by @Amix29 in #1114).**
  Install `soup-cli[sigstore]`; `soup adapters sign --backend sigstore` records
  the Fulcio/Rekor bundle, and verification requires an out-of-band certificate
  identity and OIDC issuer instead of trusting the bundle's own identity.

- **In-toto/SLSA attestations can now use Sigstore keyless OIDC signing (#179 by @Amix29 in #1115).**
  `soup attest emit --sign sigstore` writes a modern Sigstore Bundle into the
  existing `.sig` sidecar, and verification requires an out-of-band certificate
  identity and OIDC issuer; install `soup-cli[sigstore]` to sign or verify. An
  explicit Sigstore request can no longer degrade to unsigned.

- **`soup recipes verify` checks that a config can attach a LoRA adapter, without downloading weights (#1116 by @Srinivasan8888 in #1117).**
  It builds each base's architecture on the meta device from `config.json` alone, with the
  model class that config's trainer loads (only SFT chooses it by `modality`), then builds
  the adapter config the way the trainer does — `training.moe_lora` included — and runs
  the real peft attach with the config's own LoRA settings. `soup recipes verify` checks
  every shipped recipe, template and example (`--no-templates` limits it to the recipes);
  `--config soup.yaml` checks one before a long download; `--json` emits machine-readable
  rows. Exit codes follow the gate/verdict contract: 0 when everything checkable attaches,
  2 when a config cannot attach, 3 for a bad flag or a missing or unparseable `--config`,
  and 1 for a crash or a missing or broken `torch` / `transformers` / `peft` (it names
  `pip install "soup-cli[train]"`). Text from a config is printed safely: markup prints
  literally and control bytes are stripped. `task: asr` without `asr_lora` reports no
  adapter, as that trainer full-fine-tunes, and the target resolver's advisories go to
  stderr, so `--json` stdout stays a document. A gated repo, one needing
  `trust_remote_code`, or a config this `transformers` cannot read is reported as
  unverified rather than broken, and remote code is never executed.

- **`soup bench train` measures a short SFT run and fails when the model was not training (#836 by @Srinivasan8888 in #1146).**
  `soup bench train --config soup.yaml --steps N --warmup W -o report.json` trains for
  exactly N optimizer steps in a scratch directory and writes a JSON report. The report
  holds median and p95 step time after warm-up, supervised and total tokens, allocated
  and reserved peak memory kept separate, provenance, and a hash of the resolved config.
  It exits 1, and still writes the report, when there are no trainable parameters, a
  step logged `grad_norm == 0` or a non-finite norm, the trainable parameters did not
  change, or fewer steps ran than requested. An absent `grad_norm` is recorded as not
  reported, never as `0.0`. It covers `task: sft` on the transformers backend.
- **`soup bench` is now a group with `infer` and `train` (#836 by @Srinivasan8888 in #1146).**
  `soup bench <model>` still runs the inference benchmark, which is now also
  `soup bench infer <model>`. `soup bench --help` lists the two subcommands; the
  inference flags are under `soup bench infer --help`.

- **Added ready-made `deepseek-v4-pro-dpo` and `deepseek-v4-pro-grpo` recipes for deepseek-ai/DeepSeek-V4-Pro (#850 by @kok-o in #1297).** Completes the task trio with the existing SFT sibling over the 1.6T-class flagship MoE architecture (LoRA r32/a64, 4-bit quantization, `moe_lora: true`, `moe_aux_loss_coeff: 0.01`, `gradient_checkpointing: true`). DPO uses `max_length: 4096`, `epochs: 1`, `lr: 5e-6`, and `gradient_accumulation_steps: 32`. GRPO uses `max_length: 8192`, `epochs: 3`, `lr: 1e-5`, and `gradient_accumulation_steps: 16`.

- **`soup infer` and `soup bench infer` accept an experimental `--cuda-graphs` flag (#1322).**
  On a resident Qwen2 or Llama model, optionally under one unmerged LoRA adapter from
  `soup train`, generation replays CUDA graphs instead of launching every kernel from Python;
  without the flag nothing changes. Measured on an RTX 5070 Laptop (Windows, unpinned) with
  Qwen2.5-1.5B-Instruct in FP16 and 128 greedy tokens per request, the slowest run with the
  flag was 2.9x the fastest run without it (2.8x under an unmerged LoRA), with identical token
  IDs on every request and at most +42 MiB of reserved VRAM, and `soup infer` over 24 prompts
  of varied length, model load and compile included, took 96-102 s against 131-282 s with
  identical responses. Before the first row Soup runs one greedy warm-up on the longest
  prompt, so a one-prompt run can take longer overall; the summary's `Duration` and
  `Throughput` start after it. The flag needs PyTorch >= 2.14 and accepts plain LoRA only:
  quantization, CPU or disk offload, layer streaming, flash attention, DoRA and other LoRA
  variants, several adapters, a non-CUDA device or more than one GPU, and `--task asr` are
  refused before any output is written, and every refusal ends with an `Omit --cuda-graphs`
  sentence. It is not available in `soup chat` or `soup serve`. Record:
  `benchmarks/gate-v0.76.0-cuda-graph-decode.md`.

- **`soup ui` keeps the session across reloads with an HttpOnly cookie on a loopback bind (#1191 by @mohdUwaish59 in #1474).** Opening `/?token=<TOKEN>` checks the token, sets an `HttpOnly; SameSite=Strict; Path=/` session cookie named after the port (`soup_ui_session_<PORT>`) and redirects to the same URL without `token`, keeping the other query parameters. The cookie holds a random session id that the server keeps only in memory, never the token, so a reload or a new tab stays signed in without the token being readable from page script, and every restart ends every session. A cookie-authenticated request other than `GET`/`HEAD` must carry an `Origin` naming this exact server (scheme, host and port), because every port of `127.0.0.1` is one site and `SameSite=Strict` alone would not stop a page from another local app; a missing or `null` `Origin` is refused with 403. The `Authorization: Bearer` header is unchanged and needs no `Origin`, SSE streams accept the cookie while Bearer clients keep the ticket exchange, and CORS still sends no `Access-Control-Allow-Credentials`. On `--public` or any non-loopback bind no cookie is set or accepted, and the phone page keeps the token in page memory as before.

- **Layer streaming can stripe its disk-tier cache across two or more NVMe drives (#1536).**
  Set `SOUP_LAYER_STREAM_STRIPE_DIRS` to extra folders on other NVMe drives (`;` on Windows,
  `:` elsewhere): decoder layer `i` is stored on drive `i mod N`, and the reader keeps one layer
  in flight per drive. Each folder must already exist, be absolute and not a link, sit on its own
  volume and classify as NVMe; anything else refuses the run and names the entry. Soup makes its
  own folder inside each root owner-only. `training.stream_read_ahead` defaults to drives + 1.
  On a cold 70B-shaped NF4 store at seq 512 the training step went from 17.4 s to 9.97 s (1.75x)
  on two drives from a rested box (`benchmarks/gate-two-drive-striping.md`); under sustained
  back-to-back load both drives throttled and the speed-up measured 1.28-1.41x
  (`benchmarks/probe-rtx5070-two-drive-sustained.md`, no formal verdict yet). Unset, nothing
  changes.

- **`soup autopilot` asks for a missing `--model`, `--data` or `--goal` when run in a terminal (#1576 by @Deadpool2000 in #1577).** When it is not run in a terminal it fails as before.

- **Added a Simplified Chinese (`zh-Hans`) README translation (#1632 by @Momoyeyu in #1733).**
  `README.zh.md` mirrors the English README section-by-section, while the shared
  language banner, code blocks, URLs, link targets, numbers, and inline code stay
  synchronized by the translation ratchet.

- **Single-adapter health and silent-failure diagnostic scan (`soup adapters check`).**
  Audits a LoRA adapter for Frobenius weight delta ||ΔW||_F with float64 accumulation
  and rsLoRA scaling awareness, live layer fraction, all-zero lora_B initialisation traps,
  orphaned projections, and legacy .inner. wrapper namespace leaks without a PyTorch dependency,
  ending in a single contract line (`alive` vs `inactive: <reason>`).
  (#1721 by @harshitthek in #1738)

### Changed

- **`soup ship` evidence decoding is now shared by the CLI and MCP surfaces (#758 by @ousamabenyounes in [#768](https://github.com/MakazhanAlpamys/Soup/pull/768)).** A single strict schema reader now keeps verdict fields, refusal behavior, warning semantics, and the default forgetting threshold in lockstep; unknown verdict-affecting fields are rejected until both surfaces support them.

  Reviewing that reader found a second defect and it is fixed in the same release: the MCP
  error redactor paired on bare quotes, so when an evidence value held **both** quote
  characters -- where `repr()` delimits with single quotes and backslash-escapes the internal
  ones -- the pairing slipped by one and the evidence text between them reached the transport.
  Each alternative now consumes an escaped character inside the run, so an escaped quote
  cannot end it.

- **Unify exit codes across gate and verdict commands under a single 0/2/3 taxonomy (#813 by @kok-o in #921).**
  Gate and verdict commands now consistently exit `0` on pass / OK / minor / ship, exit `2` on gate failure / regression / drift / don't ship / suite failure (`EXIT_GATE_FAILED`), and exit `3` on usage or input errors (missing or unparseable input files, bad flags, invalid schema) (`EXIT_USAGE_ERROR`). Shared constants are centralized in `soup_cli.utils.exit_codes`.

  **Breaking changes in command exit codes:**
  - `soup eval gate`: exit on gate failure changed from `1` to `2`.
  - `soup eval against`: exit on regression changed from `1` to `2`; exit on unsupported metric/tracker changed to `EXIT_RUNTIME_ERROR` (`1`) / `EXIT_USAGE_ERROR` (`3`).
  - `soup eval quant-check`: aligns `MAJOR` regression exit code with `EXIT_GATE_FAILED` (`2`); unparseable inputs exit `3`.
  - `soup lock check`: drift exit code changed from `3` to `2` (`EXIT_GATE_FAILED`); missing/unparseable lock file exits `3`.
  - `soup lock show`: missing lock file exit code changed from `2` to `3` (`EXIT_USAGE_ERROR`), aligning with documentation.
  - `soup expect`: suite failure exit code changed from `3` to `2` (`EXIT_GATE_FAILED`); input/schema errors exit `3`.
  - `soup data validate`: input errors exit `3` (previously `1`).
  - `soup data lint`: unsupported format / input errors exit `3` (previously `2`).
  - `soup ship`: offline `--evidence` parse and malformed evidence errors exit `3` (previously `1`).
  - Click / Typer CLI usage errors (such as `--bogus` or missing required options) on gate and verdict commands now exit `3` (`EXIT_USAGE_ERROR`), avoiding collision with `EXIT_GATE_FAILED` (`2`).

- Changed `soup data recipe --execute` and the Python `run_recipe` API to refuse
  silent placeholder generation by default; pass a live provider or explicitly select
  offline mode (#817 by @Amix29 in #922).

- **The v0.72.2 8B NF4 throughput protocol is now re-runnable on repaired code, and the post-repair row is recorded (#361 by @umran666 in #1014, measured in #1103).**
  The RTX 3050 run has happened: Llama-3.1-8B-Instruct NF4 streamed at
  **208.6 tok/s** in a 2.40 GB peak (5.70 GB pinned store) at 1935 MHz / 59 C,
  100% util, 59% of the same-session GEMM ceiling against 68% pre-repair — the
  row and its clock in `benchmarks/gate-v0.72.2-nf4.md`, the raw numbers in
  `benchmarks/results/issue361-post-repair-8b-nf4.json`. It ran at roughly twice
  the pre-repair row's 952 MHz, so its 208.6 tok/s and the published 119.6 are
  not comparable directly and the record says so. `benchmarks/harness/
  issue361_nf4_throughput.py` repeats the step-6 protocol unchanged (batch 1,
  S=512, 50 steps after 10 warm-up, `PagedAdamW8bit`, same-session GEMM
  ceiling) with `require_pin=True` so a pageable fallback refuses instead of
  recording the lower-bound shape, and the 119.6 tok/s / 3.32 GB copies in the
  README, the benchmarks index and the perf docs are now labelled pre-repair,
  with the reserved row in `benchmarks/gate-v0.72.2-nf4.md` now filled rather
  than pending. CPU-tested protocol guards live in
  `tests/test_issue361_nf4_throughput.py`.

- **Adding a recipe is now one number plus one command (#1016 by @harryc295 in #1051).**
  `tests/recipe_count.py` already pins `EXPECTED_RECIPE_COUNT`; `scripts/sync_recipe_count.py`
  rewrites the documentation sites that state the count from `len(RECIPES)`, so the second
  half of the edit stops being manual. It imports `DOC_SITES` from
  `tests/test_recipe_count_is_synced.py` rather than duplicating it, so the checker and the
  fixer cannot disagree about which sites exist or what their patterns are, and it rebuilds
  each match around the captured number so the surrounding wording survives byte for byte.
  `--check` reports what is out of sync and exits 1 without writing. File I/O is byte-level
  rather than text-level so a one-number edit stays a one-line diff on Windows checkouts.
  It deliberately leaves `EXPECTED_RECIPE_COUNT` alone: the milestone stays a decision
  rather than a side effect of running a script. Tooling and tests only; no runtime
  behaviour changes.

- **`training.nvfp4` and three of the four `soup export --format torchao` schemes asked torchao for classes it does not have (#826 by @Srinivasan8888 in #1066).** `NVFP4Config`, `Int8DynActInt4Config` and `Float8DynActFloat8Config` are not exported by torchao 0.18.0, the newest release, so a Blackwell user who set `nvfp4: true` got a yellow advisory and a bf16 run, three export schemes could not build a config at all, and both refusals said "upgrade torchao" when there was nothing newer to upgrade to. Every class is now resolved through the module that defines it, so a rename fails naming the path; `training.nvfp4` uses `NVFP4TrainingConfig`, which quantises the forward and backward GEMMs, rather than an inference config that would only quantise weights after training. The four disagreeing torchao floors (`0.4.0`, `0.5.0`, `0.7.0`, `0.5`) become one constant, `inner_k_tiles` leaves the export allowlist because `Int4WeightOnlyConfig` raises `TypeError` for it, and a new `torchao-contract` CI job installs `.[qat]` so the names are checked against the real package instead of a stub that defined them. **Breaking for `[qat]` users:** that one floor is `torchao>=0.18.0`, up from `0.4.0`, because 0.18.0 is the release these module paths were verified against; an older torchao is no longer accepted by the extra. Also narrowed: `soup export --format torchao` used to find `quantize_` at the torchao top level *or* in `torchao.quantization`, and now resolves it from `torchao.quantization` only — the path that exists in 0.18.0, so nothing at the new floor loses it. Scope: this makes `nvfp4: true` reach NVFP4 training (the linears become `NVFP4Linear`), which the maintainer confirmed on a Blackwell GPU; a training step additionally needs triton and M, K, N all divisible by 128, which are torchao's constraints and are not checked by Soup.

- **`training.loraplus_lr_ratio` was silently ignored on every LoRA task except
  sft, pretrain and embedding (#745 by @abdulwaarith0 in #1080).**
  #724 wired LoRA+ into the SFT family only. The other trainers that build a LoRA
  adapter (asr, bco, classifier, distill, dpo, grpo, ipo, kto, online_dpo, orpo,
  ppo, reward_model, simpo) accepted the ratio and trained the LoRA B matrices at
  the base learning rate. They now attach the LoRA+ optimizer, so B trains at
  `lr * loraplus_lr_ratio` and A at `lr`, and runs that set the ratio produce
  different weights than before. PPO builds its optimizer and scheduler eagerly,
  so the LoRA+ optimizer is passed to its constructor instead, and it also carries
  the value model at the base learning rate, as trl's default optimizer does.
  The ratio is now refused at config parse where it cannot apply: `unlearn` (no
  `Trainer` optimizer), `prm` (full fine-tune), `moe_lora_routing` (only the
  routing gate trains), `classifier` / `reranker` / `cross_encoder` / `asr`
  without their `classifier_lora` / `asr_lora` flag and `lora.r > 0`, and
  `lora.use_vera` (VeRA trains scaling vectors, not A/B matrices, so every
  tensor would train at `lr * ratio`). A source scan fails if a new LoRA
  trainer ships without wiring LoRA+ or a documented exemption.

- **Sesame CSM is now refused at config load until Soup has a native multimodal trainer (#265 by @Amix29 in #1112).** The unrunnable `sesame-csm-tts` recipe is removed because CSM trains text plus 32 Mimi codebooks as parallel frames rather than a codec-token assistant turn.
- **Llasa can now encode raw audio with Transformers-native XCodec2 during TTS training (#265 by @Amix29 in #1112).** The live path resamples to 16 kHz, emits Llasa speech tokens, requires the matching `[audio]` / `torchaudio` dependency, and releases the temporary XCodec2 model after dataset encoding instead of holding it through the training run.

- `soup eval behavior` and `soup eval checklist` now exit with a usage error when required evidence is omitted instead of reporting a neutral successful gate (#1123 by @taljeon in #1129).

- ReLoRA restarts now follow the paper contract (#693 by @jagadeepmamidi in #1133): each fire merges the PEFT-scaled LoRA update into the base weight, reinitializes `lora_A`/`lora_B`, clears optimizer state when `relora_reset_optimizer` is set, and re-warms the learning rate from `1/(W+1)` without an exact-zero step. Behaviour break: `relora_steps` requires explicit `quantization: none` because the schema default is `4bit`. Embedding adapters with an empty `lora_A` are skipped; Linear LoRA is still merged. The final active adapter is merged into the accumulated in-memory base and saved as a standalone dense output; `--resume` and `--hf-resume` are refused for ReLoRA because intermediate checkpoints do not encode restart history. Distributed/sharded execution remains explicitly refused, and `relora_prune_ratio` remains ignored with a warning for backward-compatible YAML.

- **Streamed CUDA NF4 now follows bitsandbytes' own fused-versus-dequantized dispatch,
  so it runs the same kernels as resident NF4 (#842 by @Amix29 in #1143).** Pooled
  weight aliases stay checkpoint-visible and backward verifies stream-slot ownership.

- **An explicitly requested FP8 run now stops when torchao is missing, and says how to fix it (#835 by @Srinivasan8888 in #1154).**
  With `quantization_aware: fp8` or `fp8_attention: true` on a card that can run
  FP8, the trainers used to print a yellow advisory and train in bf16 under a config
  that said FP8; that is what `soup sweep`, `soup bench train` and the Python API
  reached. They now stop at setup with `FP8DependencyMissingError`, naming
  `pip install "soup-cli[qat]"`, the same way an unsupported card already stops. A
  box with transformer-engine but no torchao stops too, because the converter is
  torchao's. A config that does not ask for FP8 is unaffected. If your runs relied
  on the old fallback, remove the FP8 setting or install the extra.
- **`soup train` checks an FP8 request before loading the model (#835 by @Srinivasan8888 in #1154).**
  It used to run the int8 QAT check for `quantization_aware: fp8` too, so a missing
  torchao stopped the run with the int8 hint `pip install torchao`, even on a card
  that cannot run FP8, where installing it would not help. It now asks the FP8
  questions in the trainers' order: a card that cannot run FP8 is named as the reason,
  and on a card that can, a missing torchao names `pip install "soup-cli[qat]"`. It
  asks for torchao's float8 converter itself, so a box with only transformer-engine
  stops here too instead of after the model loads. On `backend: unsloth` only the
  unsloth refusal is printed. Under `--dry-run` the local card is reported as a note
  and the dry run exits 0, since FP8 configs are often written on one machine for
  another; a missing torchao and schema errors still fail it (a missing torchao prints
  the card note too), and the real run still stops on the card.

- **FP8 now applies on SFT's audio path, and `stream_layers: true` and `backend: unsloth` refuse FP8 and NVFP4 at config load (#1124 by @kok-o in #1155).** `quantization_aware: fp8` and `fp8_attention: true` loaded on audio SFT and were never applied; `SFTTrainerWrapper._setup_audio_transformers` now converts the model after the LoRA adapter is attached, as the text and vision setups do, so a card that cannot run FP8, or a missing torchao, stops the run at setup instead of training in bf16. `stream_layers: true` (on every task it supports) and `backend: unsloth` (on every task) now refuse `quantization_aware: fp8`, `fp8_attention` and `nvfp4` at config load, because neither path calls the converters: these configs loaded before and trained without the precision they asked for, except the `fp8` combinations on unsloth, which `soup train` already refused before training. The unsloth `fp8` message says to switch to `backend: transformers`.

- **`soup bench train` samples the SM clock while the counted steps run, and keeps every step time (#1166 by @AaronProbha18 in #1177).**
  **Breaking (report schema):** `provenance.sm_clock_mhz_after_run` is removed. That value
  was one `nvidia-smi` read taken after training returned, on a card that was already idle.
  The gate-836 record measured it at mostly 180 MHz, and never above 1417, against a busy median of 2370–2557 MHz, and
  it could not tell apart two runs whose throughput differed 1.83x. The replacement,
  `provenance.sm_clock_mhz_busy`, has `min`, `median`, `max` and `sample_count` over samples
  kept only from the post-warm-up steps. Queries start on a 100 ms schedule, a query slower than
  that skips ticks, and each sample is timestamped at the midpoint of its query. `unavailable_reason` says why no clock was reported: not a CUDA run,
  no `nvidia-smi`, no readable clock, or no sample inside the counted steps. The clock is
  queried for torch's own device by PCI bus id, and a failing or unparseable driver-version
  query is now `null` rather than the tool's error text. `timing.step_seconds` (new) lists
  every counted step in run order, so a mid-run mode change can be located from one report.
  The docs now say that pre-flight refusals (unsupported task or backend, `--steps <= --warmup`,
  zero trainable parameters) write no report, and that only post-run check failures leave a
  `valid: false` one.

- **An untied streamed checkpoint reloads the embedding at the tail of the previous step's backward instead of at the start of the next forward (#975 by @VanshSharmaPES in #1178).** `embed_tokens` and an untied `lm_head` share one device slot, so every step swapped the slot back to the embedding right as the forward began, a full head-sized copy with nothing to overlap it. The reload now happens once layer 0's backward recompute is reached, so it overlaps the rest of the backward, and the forward-start load becomes a same-owner no-op from the second step on. The refill is ordered after the head's still-queued backward by the `wait_stream` in `LargeLayerBufferPool.load_async`, which a new GPU test pins. On an untied Llama with hidden 2048, vocab 128256 and a 501 MiB slot, on an RTX 5070 Laptop GPU, the embedding's compute-stream wait fell from about 19-23 ms to about 0.15 ms per step; that is still about 0.1% of a cold 70B step, and the step-time change (2-6.5%) is close to that fixture's noise floor. Preference losses get part of it: DPO's no-grad reference forward and KTO's extra forwards still load the embedding at the start of their forward. Tied checkpoints have no shared slot and are unchanged, and each `train()` now ends with one extra large-layer load that no later forward consumes.

- **Every non-standard `grpo_variant` now applies the `grpo_beta` KL penalty, so existing variant configs will not reproduce prior runs (#1232 in #1249).**
  `gspo`, `dapo`, `dr_grpo`, `bnpo`, `two_sided` and `rft` now train with a KL
  penalty of `grpo_beta` (default `0.1`) against the
  reference policy, so their losses, gradients and trajectories change. Before,
  they validated `grpo_beta` and dropped it. The variant loss replaces trl's
  `_compute_loss`, the one place on Soup's path where trl adds the KL term, so
  neither `grpo_beta` nor the β that the reward-hack controller
  (`kl_control` / `pid_lagrangian`) sets at runtime changed the loss or the
  gradient, while trl still ran the reference forward on every step. Each
  variant now adds trl 0.29's per-token estimator
  `exp(ref - logp) - (ref - logp) - 1`, weighted by `grpo_beta`, to its
  per-token loss before its own reduction, which is where trl's loss adds it.
  `rft` applies it only to the accepted completions it trains on. On a real
  trainer, the `dapo` loss and gradient now equal trl's stock `dapo` ones at
  β 0.1, 1 and 10 on the same generated batch.

- **An untied streamed checkpoint issues the head's load right after the embedding lookup instead of at the last decoder layer (#1174 by @VanshSharmaPES in #1258).** The embedding's bytes in the shared large-layer slot are dead once the lookup has run, so `StreamPrefetcher` now fires `tail_prefetch` from layer 0's `advance()` (`head_prefetch_layer`, default 0) rather than the last layer's, giving the head's copy every decoder layer's forward to hide behind instead of one. `head_prefetch_layer = None` restores the old timing. Loss, gradients and adapter weights are bit-identical between the two timings, and the sequence and count of real large-layer loads are unchanged; the refill is ordered after the queued lookup by the `wait_stream` in `LargeLayerBufferPool.load_async`, which a new GPU test pins. On an RTX 5070 Laptop GPU, with an 8-layer untied checkpoint (hidden 2048, a 501 MiB shared slot, NF4, sequence 512), the step's median total `wait()` fell from 11.96 to 4.50 ms on the RAM tier, from 15.83 to 1.42 ms on the disk tier, and from 57.29 to 29.16 ms on the disk tier at `stream_read_ahead: 3` for a step that runs a no-grad forward first. A decoder layer's copy now queues behind the head's on the shared copy stream, so the other decoder layers wait longer (0.78 to 4.06 ms on the RAM tier), which gives back part of the saving; layer 0 was the best of layers 0 to 3. Tied checkpoints have no shared slot and are unchanged.

- **`training.quantization_aware: true` is now refused at config load, on every task and backend (#1222 in #1275).** It never ran quantization-aware training. Where it was applied, which is the transformers model setup of the `sft`, `dpo`, `kto`, `orpo`, `ipo`, `bco`, `simpo`, `grpo`, `ppo` and `pretrain` trainers (also run by `task: tts` and `task: preference`), it ran torchao's post-training int8 weight-only quantization over every Linear layer after LoRA, adapter included: a LoRA run had zero trainable parameters, and none of the runs tried (SFT, SFT with gradient checkpointing, full fine-tuning, DPO) completed. Everywhere else it was not applied at all. `backend: mlx` ignored it with a warning. `backend: unsloth` never applied it: `soup train` refused the combination, and `soup sweep` and `soup bench train` trained without it (#1248, superseded by this change). The trainers for `reward_model`, `embedding`, `prm`, `online_dpo`, `asr`, `classifier`, `reranker`, `cross_encoder`, `distill`, `unlearn` and `moe_lora_routing` never read it, and neither did layer streaming or SFT's audio path, so those runs trained without QAT. **Breaking for both groups:** a config that sets it loaded before and now stops at load. `soup train` exits 1, the Web UI and API return the message, and `soup doctor --config` exits 2. Every spelling the field reads as true is refused (`yes`, `on`, `1`, `"true"` ...), with a message that names `training.quantization_aware` and #1222. Remove the key or set it to `false`. `quantization_aware: fp8` and `quantization_aware: quest` are unaffected, because they use their own code paths; the one visible change for `fp8` is that `soup train`'s refusal of `fp8` on `backend: unsloth` now says `Use backend: transformers.` instead of recommending `quantization_aware: true`. Real int8 QAT is a separate feature and is not available yet.

- **`training.use_longlora: true` is now refused at config load (#1240 in #1278).** It never trained correctly: the LongLoRA override rolled the query/key projections of half the attention heads along the sequence with wrap-around, under full causal attention, so earlier positions could read keys computed from the last tokens of each sequence (future tokens leaked into the training loss), and no S² grouped attention ran at all. **Breaking for configs that set it:** a config that loaded before now stops at load, even though it never trained correctly. `soup train` exits 1, the Web UI and API return the message, and `soup doctor --config` exits 2. Every spelling the field reads as true is refused (`yes`, `on`, `1`, `"t"` ...), with a message that names `training.use_longlora`, says why, and points at `training.rope_scaling_type` with plain LoRA for context extension. Remove the key or set it to `false`. It stays refused until real S² attention is implemented; no shipped recipe, template or example set it.

- **`training.ebft_variant` is now refused at config load (#1230 in #1279).** It never trained correctly: the EBFT term it added to the SFT loss scored each position's logits against that position's own input token, with no causal shift, so it rewarded copying the input over predicting the next token (on the real trainer the added term was exactly the unshifted cross-entropy). Shifting it would not help: shifted, the term is the model's own cross-entropy, so the loss would count cross-entropy twice. EBFT is not yet a distinct objective, and it stays refused until the intended objective is implemented from its reference. **Breaking for configs that set it:** a config with `ebft_variant: structured` or `strided` loaded before and now stops at load. `soup train` exits 1, the Web UI and API return the message, and `soup doctor --config` exits 2. Every non-null value gets the same message, which names `training.ebft_variant`, says why and names #1230; `ebft_temperature` on its own now points at the refusal instead of asking for `ebft_variant`. Remove both keys to train plain SFT. GDPO (`gdpo_variant`) is unchanged (its own problem on trl 0.29 is #1309), and no shipped recipe, template or example used EBFT.

- Refuse ten optimizer names at config load instead of crashing inside the trainer: transformers 5.x rejects `adam_mini`, `adamw_apex_fused`, `adamw_hf`, `ao_adamw_fp8`, `ao_adamw_4bit`, `ao_adamw_8bit`, `badam`, `came_pytorch`, `dion` and `muon` in `TrainingArguments`, so each used to pass config load and fail after the model had loaded. They are removed from the allowlist with a message naming working alternatives, the LoRA-FA validator no longer recommends `adamw_hf`, and a new test builds `TrainingArguments(optim=name)` for every allowlisted name so a rejected name fails the suite instead of a user's run. `muon` and `adamw_hf` still work on `backend: mlx` (the MLX backend builds them itself); every other backend refuses them with a message that names the backend (#1269 by @wanjinhao1 in #1283).

- Allow `training.grpo_beta: 0` to configure KL-free GRPO recipes (DAPO, Dr. GRPO), skipping the reference-model forward pass; refuse `grpo_beta: 0` with beta-adjusting reward-hack mitigation modes (`kl_control`, `pid_lagrangian`) at config load (#1247 by @kok-o in #1295).

- **`soup ab` now runs a two-sided test, says whether the treatment is better or worse, and gives no verdict before 30 rows per arm (40 below `--alpha 0.05`) (#1227 in #1310).** The mSPRT tested only `mean_t - mean_c = +effect_size`, so a treatment that moved the other way crossed the accept boundary: a `judge_score` regression of any size was reported as "No significant difference" (exit 0, webhook `accept_h0`), and on `latency` and `retry_rate`, where lower is better, so was a real improvement. The log-likelihood ratio is now the log of the average of the likelihood ratios for `+effect_size` and `-effect_size` (a symmetric two-point mixture, computed with logsumexp), with the same `log((1 - beta) / alpha)` boundary. A `reject_h0` carries a `direction`, `better` or `worse`, read through the metric's polarity (`judge_score` higher is better; `latency` and `retry_rate` lower is better). The verdict table and the webhook payload show it (`null` on `accept_h0`), and the panel recommends a rollback only for `worse`. `MsprtVerdict` gains the field and requires it on a `reject_h0`. **Fewer than 30 rows in either arm now always gives `continue`, or fewer than 40 when `--alpha` is below 0.05** (`min_rows_per_arm(alpha)`), and the verdict table and panel show the value in use. `soup ab` estimates the variance from the rows, and from a handful of them the estimate is too noisy: without a burn-in, re-running after every new pair would reject a true H0 up to 16% of the time at `--alpha 0.05` (the one-sided test it replaces: 11%). A Monte-Carlo sweep over effect sizes from 0.1 to 5 standard deviations, with runs followed to 1000 rows per arm, picked the burn-ins: at `--alpha 0.05`, 30 rows keep the worst case at 0.051, within Monte-Carlo error; at `--alpha 0.01` the worst case stays near 0.0108, about 8% above the nominal level whatever the burn-in, because the variance is estimated from the rows (a variance-robust statistic is #1265). The issue's own 20-row reproduction now returns `continue` at the default `--alpha`, because 20 rows fall inside the burn-in. The calibration holds up to 1000 rows per arm and for `--alpha` of 0.01 and above, and `soup ab` prints a warning past either limit. The sweep is committed under `benchmarks/` (`gate-1227-ab-burn-in.md`). Because one boundary now covers both directions, the log-likelihood ratio for an effect in one direction is about `log 2` lower than before. The exit code is unchanged, 0 on every decision, so read `direction` to gate on a regression.

- **Training now evaluates the validation split it holds out, and generation tasks stop holding one out (#1223 in #1313).**
  `data.val_split` (default `0.1`) kept that share of the rows out of training on every run, but no trainer on the
  transformers backend scheduled an evaluation, so those rows were never used and `val_loss` stayed empty. Every
  task that receives a split now evaluates it at the end of each epoch, at the per-device train batch (not
  Hugging Face's 8), and records `eval_loss` on the Val loss row, in the tracker's `val_loss` and in
  `TrainEvent.val_loss`; `prm` and `moe_lora_routing`, which attach no live callback, keep it in the trainer's log
  history. Runs take a few percent longer. The new `training.eval_steps: N` evaluates every N optimizer steps
  instead. `grpo`, `ppo` and `online_dpo` evaluate by generating, so by default they neither evaluate nor withhold
  rows: `data.val_split` is ignored with a one-line note and every row trains, which changes the rows a default
  config trains on. `grpo` holds out and evaluates when `training.eval_steps` is set. It evaluates at the largest
  multiple of `num_generations` that fits in the train batch, since TRL scores whole groups, and a note says so
  when that differs from the train batch. Its evaluation rewards never reach the reward-hack detector, the
  mitigation controller or the echo-trap detector. `ppo`, `online_dpo`, `unlearn` and `backend: mlx` refuse the
  setting, and each refusal says why. The embedding trainer gains the evaluation step that Hugging Face's default
  could not run on its anchor/positive batches.

- **An explicitly requested FP8 conversion that fails now stops the run instead of training on (#1152 by @Srinivasan8888 in #1333).**
  With `quantization_aware: fp8`, a float8 conversion that raised used to print a yellow line and train in bf16; it now stops at setup and quotes torchao's error. With `fp8_attention: true`, a conversion that fails partway, or a model with no attention projections, used to print its own "PARTIALLY converted" or "refusing the silent no-op" message as a yellow line and train on anyway; both now stop the run, as those messages say. This follows #835's rule for a card that cannot run FP8 and for a missing torchao. `utils/fp8.py::validate_fp8_config`, which nothing called, is removed.

- **`training.lr_groups`, `training.early_stop_patience` and `training.citation_recall_threshold` now warn at load, and v0.77 refuses them (#761 by @Srinivasan8888 in #1337).**
  All three were documented and validated but read by nothing: `lr_groups` never set a per-module learning rate, `citation_recall_threshold` never gated a save, and `early_stop_patience` never stopped a run. Setting one away from its default now prints "accepted but read by nothing; v0.77 will refuse it", as the #808 fields do. `early_stop_patience` moves from `soup train`'s "set but not enforced" note to that warning, so it is reported once, with the date. The docs no longer show `lr_groups` or `citation_recall_threshold` doing what they never did.

- **Nine training-intelligence and convergence knobs warn at load and v0.77 refuses them (#808 by @Guts1005 in #1379).**
  `forgetting_eval_steps`, `forgetting_benchmark`, `forgetting_stop`, `checkpoint_eval_steps`, `checkpoint_eval_metric`, `checkpoint_eval_tasks`, `checkpoint_keep_top`, `convergence_window`, and `convergence_rel_tol` were accepted and validated, but none of the nine ever changed a run. Setting any of them away from its default now prints "accepted but read by nothing; v0.77 will refuse it". They move from `soup train`'s "set but not enforced" advisory note to the loader's warning so each knob is reported once, with the refusal date. Deleting the key keeps configs loading after v0.77, and schema defaults stay silent.

- **`soup ab` now decides with a variance-robust statistic, with no burn-in, and `accept_h0` can take longer to reach when there is no difference (#1265 by @abdulwaarith0 in #1419).** The two-sided mSPRT from #1227 estimated the variance from the rows, so it needed a burn-in of 30 or 40 rows per arm, was calibrated only up to 1000 rows and down to `--alpha 0.01`, and still sat near 0.011 at 0.01. The normal-inverse-gamma Bayes factor that replaces it depends on the rows only through the t statistic, so its false-positive rate stays at most `--alpha` however often you re-run it, from 7 rows per arm and at any number of rows: the worst simulated rate is 0.033 at `--alpha 0.05` and 0.0066 at 0.01. The first 5 rows per arm set the scale for `--effect-size` and show in the table as `prior_scale`; the burn-in table, the 1000-row limit and both calibration warnings are gone. This is not a pure speed-up. When the arms are the same, `accept_h0` takes several times as many rows as before (about 281 pairs against 43 at an `--effect-size` of 0.5 standard deviations), and at small effect sizes it may not arrive within 1000; `docs/evaluation.md` has the full table, and #1575 in this release replaces the accept rule, which brings that down to 117 pairs. Part of the old speed was wrong answers: under a real difference of 0.3 standard deviations, the old test ended about 18% of runs in `accept_h0`, against under 0.4% now on simulated Gaussian rows. When the first rows barely vary (a warm cache, or a judge that gives 1.0 early), `accept_h0` waits as `continue` while the tested rows spread more than 3 times as much as the held-out ones, since that start made nearly every run accept H0 with or without a difference; a `reject_h0` is never held back, so the false-positive bound is unchanged. The `--alpha` + `--beta` < 1 check from #1339 does not ship: #1575 retires `--beta` in the same release. The #1339 and #1384 refusals of values a float cannot carry cover the new statistic too: it is never compared with the boundaries unless it is finite.

- **`soup diagnose` no longer scores a probe that never ran as OK 1.00 (#1435 by @bakdauletbaktygaliyev in [#1489](https://github.com/MakazhanAlpamys/Soup/pull/1489)).** A live probe that was requested but produced no measurement, such as a `--dataset` with no usable rows or a probe that raised, now reports the new `NOT_RUN` verdict (grey in the badge and the table, ranked between MINOR and MAJOR) and keeps the exception text in its evidence. `soup diagnose` exits `3` when the overall verdict is `NOT_RUN`; pass `--allow-not-run` to keep exit `0` for runs that knowingly skip probes. MAJOR still exits `2`. `soup train --diagnose-gate` now also refuses a `NOT_RUN` report (exit `3`), and a bad `--tokenizer` is an input error (exit `3`) raised before any model loads. Probes that do not apply, and modes omitted from an `--evidence` file, stay a neutral OK.

- **Breaking: `training.gdpo_variant` is now refused at config load.** Every non-null value is refused, including `standard`, `length_normalized` and `margin`; unset and null still load. On supported TRL versions (0.29 and later) the setting previously loaded but silently changed nothing because `DPOTrainer.dpo_loss` no longer exists. Remove `gdpo_variant` and use plain `task: dpo` for `standard`; `task: simpo` is the nearest, not identical, objective for `length_normalized`; `margin` has no equivalent. GDPO metadata no longer advertises live wiring, and the retained legacy helper raises instead of silently ignoring a missing loss hook. No new loss implementation is added; no shipped recipe, template or example sets this field. (#1309 by @taljeon in #1515)

- **Breaking for `soup data score --benchmarks` / `-b` without comparison texts:** benchmark names are labels, not bundled corpora; the command now refuses labels alone by name and asks for `--benchmark-file`. With a file, labels are validated only, not corpus selectors; help, docs and an execution note state that all file texts are compared. Plain scoring reports decontamination as not run instead of a misleading zero. The new cwd-contained operator JSONL option passes actual text and `--threshold` to the existing 8-gram scorecard kernel, without benchmark downloads or a new scoring algorithm. Invalid, empty, truncated or unusable benchmark evidence fails before a scorecard is printed. The shared loader's strict mode is opt-in; other callers retain their previous behavior. This covers only the data-score item, not the remaining command paths. (#1448 by @taljeon in #1518)

- **A `simpo` + `orpo` blend now trains instead of refusing (#1425 by @umran666 in #1556).** #1513 made every blend refuse, because no trl 0.29 trainer puts per-sequence log-probs on the batch. The CPO (SimPO) and ORPO trainers both return theirs from `concatenated_forward`, so the wrapper now captures them there: a blend naming only `simpo` and `orpo` runs. Blends naming `dpo` or `ipo` still stop and name the terms — those trainers compute log-probs inline, publish only means, and need a frozen reference model this path does not build.

  **Behaviour changes:**
  - `{simpo: 0.7, orpo: 0.3}` is live on trl 0.29 where it previously stopped at step 0. `bco` mixed with anything is still rejected at config load.
  - Each blended loss now reads its hyperparameters from the config rather than off the primary trainer, which is whichever loss has the larger weight. `training.simpo_gamma` and `training.orpo_beta` now reach their own term either way; before, `orpo_beta` could feed SimPO's beta and ORPO's odds-ratio weight was always `1.0`, so the same weights written in a different YAML order trained a different objective.
  - In a blend, `simpo` and `orpo` are their preference terms only — without the likelihood term the standalone `preference_loss: simpo` / `orpo` runs carry. A `{simpo: 1.0}` blend is not the standalone SimPO loss, and `training.cpo_alpha` does not affect it. Use `training.preference_loss` for a full standalone loss.
  - With `val_split` set, trl's `prediction_step` calls `get_batch_loss_metrics` directly, so eval reports the primary loss's own trl loss rather than the blend's value.

- **`soup ab` reaches `accept_h0` about twice as soon when there is no difference, for every user and at the defaults, and `--beta` is retired (#1418 by @abdulwaarith0 in #1575).** The statistic and the `reject_h0` rule from #1265 are unchanged. `accept_h0` used to wait for the Bayes factor to fall to `log(beta / (1 - alpha))`, which took 621 pairs per arm at an `--effect-size` of 0.3 standard deviations (`--alpha 0.05`). It now comes once a confidence sequence for the difference, built from the same mixture at level `--alpha` and valid however often you re-run `soup ab`, lies inside `(-effect_size, +effect_size)`: 307 pairs at 0.3, 117 at 0.5 (was 281). This changes when every `soup ab` run says `accept_h0`, with no flag to opt out, and so when the `--slack-url` and `--discord-url` webhooks fire with `"decision": "accept_h0"`; a pipeline that waits on that message will hear it sooner. The accept panel now says what it means: any difference is smaller than `--effect-size`, at confidence `1 - --alpha`. A true difference of exactly `--effect-size` ends in `accept_h0` in at most 0.0083 of simulated runs at `--alpha 0.05` (was 0.0042) and 0.0019 at 0.01 (was 0.0032), so the chance of detecting it stays within 0.005 of before; the false-positive rate can only fall, and does. The `log_likelihood_ratio` reported next to an `accept_h0`, in the table and the webhook payload, is still the reject rule's statistic and can be positive. `--alpha` now bounds both wrong verdicts, so `--beta` no longer does anything: for this release `soup ab --beta X` prints that, says a stricter accept means a lower `--alpha`, and runs exactly as without it, and `MsprtConfig(beta=...)` raises the same message as a `SoupConfigDeprecationWarning`; Soup v0.77 refuses it. The `--alpha` + `--beta` < 1 check from #1339 went with it. The record is `benchmarks/gate-1418-ab-cs-accept.md`, and `docs/evaluation.md` has the new time-to-accept table.

- Changed the private-address refusal on outbound endpoints to say what to do: it now
  ends with "address the server by its hostname". `http://0.0.0.0` in `soup data generate
  --api-base` and the chat proxy now says "0.0.0.0 is ambiguous; use 127.0.0.1 or
  localhost", the same hint `HF_ENDPOINT`, hub endpoints and webhook URLs give, from one
  shared constant (#1549 by @PranavD2905 in #1582).

- `soup ab` now says when the accept hold is what is keeping a run at `continue`. #1419 holds `accept_h0` back while the tested rows spread more than `ACCEPT_HOLD_SPREAD_RATIO` (3) times the held-out ones, and the held-out rows are fixed — so once that hold fires on a saturated warm-up (a warm cache, or a judge that gives the same score for the first rows), no amount of extra data of the same kind releases it. The run sat at `continue` for the whole dataset with the accept rule already satisfied, while the panel printed "Insufficient evidence. Collect more samples and re-run.", advice that cannot help. A `continue` now carries `accept_held` plus the two spreads that decided it (`held_out_spread`, `tested_spread`, and `held_out_rows`), and the panel prints one line naming the ratio and what to do about it. The four keys are not added to the webhook payload: a held run is a `continue` and the webhook fires only on a terminal decision, so they could only ever be `false` / `null` there — the panel and the `MsprtVerdict` are where the facts live. These are new keys only: nothing is renamed or removed, and a run that is not held carries no note at all. The accept rule itself is unchanged (#1524 by @yunaremaia in #1585).

- QuEST: refuse a local base whose `config.json` cannot satisfy the route before any weight file is read, instead of after the model loads. The 24-block count and the group-128 / power-of-two widths are now checked from the config the fingerprint already parses, so a base that was going to be refused is no longer loaded and no longer has every byte of its weights read and SHA-256-hashed — on `main` that was once per run, and it is now none. The refusal message is unchanged and the post-load topology gate still runs; a width that cannot be read from the config is left to that gate rather than guessed at. (#1199 by @yunaremaia in #1588)

- **Refuse `training.lora.use_dora` on unsloth and mlx backends (#1651 by @kok-o in #1657).** Reject `use_dora: true` at config load time when `backend` is `unsloth` or `mlx`, which do not implement DoRA adapter constructors.

- **Reworded `README.tr.md` and `README.ar.md` (#1681 by @Ercaner1988).** LLM reviewers, each prompted to
  read as a native speaker of one language and with no shared context, flagged false
  friends and calques in the shipped translations (for example "issue", "Serving",
  "gated" and "wheels" rendered with words that mean something else). The findings were
  applied with LLMs and vouched for by the files' maintainer, who reads Turkish natively
  and Arabic at an upper-intermediate level. `README.md` and the sync stamps are unchanged.

- `soup serve` says what `--tool-auth-token` covers when it binds beyond loopback. A
  non-loopback `--host` is refused without the token, which reads as if the token put the
  whole server behind a credential; it is checked on the tool, thumbs and adapter routes
  only, the generation routes take none, the vLLM / SGLang / MII backends have no route
  that checks it, and on a wildcard bind (`0.0.0.0`, `::`) the `Host` header is not
  compared. None of that changes. The command now prints it as a `Note:` at startup on
  such a bind, the `--host` help says that the token is required there (it said "should be
  paired with") and what it is checked on, and `docs/serving-and-export.md` has a paragraph
  on what the token does not cover (#1689).

- **Cache live canary policy and batch outcome writes**: keyed serving requests refresh the parsed policy when its file identity changes, and stats flush after 64 outcomes or two seconds, including an orderly-shutdown flush. Watch rollback takes effect on the next keyed request; superseded promotions cannot overwrite newer statistics. On-disk `canary-stats.json` and watch verdicts now lag traffic by the buffered window (normally up to 64 outcomes or two seconds); a crash can lose that window. Disk outages retain one pending batch but refuse additional outcomes, so outage losses can exceed one batch. Process-killing failures never reach recording. (#1320 by @akkupratap323 in #1712)

- **Layer streaming page-locks less memory for a pinned RAM store whose largest tensors come last (#1702 in #1713).** The store is pinned in power-of-two arenas (#901), and the planner filled them in allocation order, looking only at the arena it opened last. A Qwen3-8B NF4 store is 3.58 GB of decoder tensors followed by two 1.24 GB vocabulary matrices; neither matrix fit behind the decoder, so each opened a 2 GiB arena of its own and 8 GiB was page-locked for a 6.07 GB store. `plan_pinned_arenas` now also plans the arenas largest tensor first and uses that layout only when it page-locks strictly fewer bytes: the same store takes 6 GiB (8,589,934,592 -> 6,442,450,944 bytes, measured on an RTX 5070 Laptop), and peak process RSS in a three-step run went from 10.85 GB to 8.71 GB. A store that gains nothing, including the disk tier's staging on the measured geometries, is laid out exactly as before, and there is no new config field. Three deterministic optimizer steps are byte-identical between the two layouts (logits, loss, 144 gradients, 144 updated adapter tensors). This is a memory change, not a speed claim: a 40-run A/B on the same box (old and new layout alternating, the disk tier as the control) found no difference in step speed at its resolution of +/-15%. The record is `benchmarks/gate-1702-arena-packing.md`.

- **The `soup train` VRAM pre-flight now checks the pre-quantized formats
  (#1652 by @ybai08 in #1729).** It used to build its estimate only for
  `none`, `4bit`, `8bit` and `mxfp4`; twelve other formats started without a
  check and without a word. `gptq`, `awq`, `aqlm` and `eetq` are now checked
  at the figures the predictor already had (0.55, 0.55, 0.5 and 1.0 bytes per
  parameter). `fp8` is checked as well, and its figure is corrected from 1.0
  to 2.0: an FP8 base is loaded with `dequantize=True` and trains in bf16, so
  the old figure promised half the memory the run needs. A refused `fp8` run
  gets the sentence an `mxfp4` run gets. `hqq:*` has no figure yet, so the run
  starts and the pre-flight prints `Pre-flight skipped: no memory model for
  quantization: hqq:4bit`. This refuses configs in those five formats that
  used to start on a card too small for the estimate; `--allow-oom-attempt`
  launches them anyway.

### Deprecated

- **`soup export --format awq` and `--format gptq` are deprecated and will be removed in Soup v0.77.** Both upstream projects (AutoAWQ and AutoGPTQ) are archived, neither extra can be installed next to `[train]`, and `auto-gptq` has no Python 3.12 wheel, so the two formats could not be reached from a normal install. Until the removal they work as before. The command now starts with a `Deprecated:` notice that names the formats to move to (`gguf`, `onnx`, `tensorrt`, `bitnet`, `tq1_0`), so it is there whatever stops the run, the missing extra included; `soup quantize --to awq` and `--to gptq` print the same notice after the command they hand you; and the `--format` and `--to` help texts mark both formats (#1690).

### Removed

- Remove unreachable kernel auto-compose production helpers and validator branches after schema-level rejection of kernel_auto_compose: true (#976 by @kok-o in #1006).

- Remove orphaned `build_bitnet_trainer` and `validate_bitnet_compat` from `utils/bitnet.py`, unreachable after schema-level rejection of `quantization: bitnet_1.58` training (#1004 by @VanshSharmaPES in #1034).

- Removed the dormant `auto_quant` compatibility module (`soup_cli/utils/auto_quant.py`) that `soup serve --auto-quant` stopped using when it began refusing with exit 2 in #816; the refusal itself is unchanged and still covered by regression tests. `import soup_cli.utils.auto_quant` now raises `ModuleNotFoundError` with nothing replacing it (#1131 by @wanjinhao1 in #1262).

- The Web UI plugin registry, `soup_cli.ui.plugins` (`register_tab`, `list_tabs`, `get_tab`, `clear_tabs`, `load_plugins`, `TabSpec`), is removed, and with it the "Web UI Plugin Registry + Env Knobs" section of `docs/serving-and-export.md`. The module has been importable since v0.44.0, so `from soup_cli.ui.plugins import ...` stops working; no rendered feature is lost, because `create_app` never read the registry and a registered tab was never shown. `soup plugins` (`soup_cli.plugins`) is a different feature and is unchanged (#1465 by @Srinivasan8888 in #1608).

- **Drop unused `eval_config` from `SoupTrainerCallback` (#1643 by @kok-o in #1654).** Remove unused `eval_config` parameter and attribute from `build_soup_trainer_callback` and `SoupTrainerCallback` after auto-eval migration to `soup train`.

### Fixed

- **`soup doctor` now reports the MLX backend (#659 by @ump45nose in #660).**
  The MLX panel had been written but never wired into the report, so Apple Silicon
  users saw no MLX section at all. `doctor` now renders an MLX panel showing the
  installed version, chip, and unified memory when MLX is present, says plainly
  when it is missing on Apple Silicon, and stays quiet elsewhere. `get_mlx_version()`
  reads `mlx.core.__version__` (falling back to distribution metadata) instead of
  `mlx.__version__`, which the `mlx` package does not export and which therefore
  always reported "unknown".

- **The MLX benchmark harness now pins, and verifies, every field that moves its step counts (#716 by @dchaudhari7177 in #727).** `benchmarks/harness/mlx_sft_smoke.py` inherited `gradient_accumulation_steps` from the schema default, so the published M1 record depended on a value the harness never mentioned: at the default of 4 the record's 48 iterations are 12 optimizer updates, and because the wrapper rounds `iters` down to a whole accumulation group (#696), a row count such as 50 silently drops rows. The harness now pins `batch_size`, `gradient_accumulation_steps`, `epochs` and `lr` with the reason for each, resolves iterations and optimizer updates from the *loaded* config the same way `MLXSFTTrainerWrapper` does, and exits non-zero before training when either is not `rows * epochs`, naming both counts and any dropped rows. No published figure changes: @Srinivasan8888 re-ran the branch on the M1 8 GB the record was measured on and reproduced `Trained Tokens 2130` exactly. `tests/test_issue716_harness_step_guard.py` checks the resolution against the `TrainingArgs` the real wrapper builds and drives the abort path through `main()`.

- **`streaming: true` with `interleave: over` and `val_split` no longer puts a row on both sides of the split (#702 by @dchaudhari7177 in #729).** `over` streams with `stopping_strategy="all_exhausted"`, which recycles the shorter source, and the positional tail slice then put a row in val and its recycled copy in train, so validation loss on that source measured memorisation: 10 of 20 val rows leaked on the issue's 100-against-10 shape. The streaming `over` split is now sized over distinct rows and draws val from content that occurs once, so train keeps every row and its oversampling. It falls back to repeated content only when it must, and then withholds that content's other copies from train with a warning giving the count. A split that would leave train empty raises the same kind of error the eager path's per-source carve-out does. Streaming val comes from distinct rows in stream order, so unlike the eager path it is not balanced across sources, and `docs/data.md` now says so. @AmirF194 found that the first version silently dropped a genuinely repeated source row.

- `soup migrate` now recognizes multi-record JSONL by structure even when the
  filename does not end in `.jsonl`, so a data file such as `data.json` gets the
  existing friendly wrong-file error instead of a raw YAML `ParserError`.
  Single JSON objects and Unsloth notebooks still reach their selected migrator,
  and the structural sniff remains bounded to 64 KiB per line (#730 by @be-student,
  finding by @swalla02 and @AmirF194, closes #699).

- Stabilize distillation loss and gradients in log space (#719 by @be-student).
  All three divergences, including forward-KL, now compute and return FP32
  scalars for FP16/BF16 inputs. Non-finite logits propagate for AMP step skipping;
  no per-step finite-value guard synchronizes the device. The FP32 trade-off is
  documented in `docs/training.md`: the maintainer measured 1.71x time and 1.24x
  peak memory for the original guarded kernel, and removing the guards reduced
  time from 10.91 ms to 9.40 ms (main: 6.39 ms). This revision also avoids unused
  probability tensors. Two FP32 logits copies use about 7.8 GiB at
  B=4/S=2048/V=128256. These measurements are hardware-specific and were not
  rerun for this revision (PR #736).

- **Distillation no longer multiplies its gradient by `gradient_accumulation_steps` (#720 by @dchaudhari7177 in #742).** `_DistillTrainer.compute_loss` returns a mean over its own microbatch, so the same effective batch split four ways previously produced four times the gradient and changed the effective learning rate. The trainer now sets `model_accepts_loss_kwargs = True` after `Trainer.__init__`, consumes Transformers' full-window `num_items_in_batch`, and weights each microbatch by its share of trained tokens as completed in #943. `tests/test_issue720_distill_grad_accum.py` runs the real `_DistillTrainer` body through `Trainer.training_step` on a one-layer Llama and asserts the accumulated gradient matches the unaccumulated one at 4 and 8 steps.

- **`soup ship` stamps the judge's actual load numerics into the verdict and `--emit-evidence`, and refuses a replay under a different family (#746 by @jagadeepmamidi, Closes #367).** Live evaluation already reused `training.quantization` for the load (#570); the absolute leg-1 / leg-2 numbers for a 4-bit run were still recorded as if the judge's precision were implicit. `--emit-evidence` now writes a top-level `numerics` stamp (`4bit` / `8bit` / `bfloat16` / `float32`), the panel and `--output` report it, and `--config` plus `--evidence` refuse a family mismatch (`4bit` vs full precision, `4bit` vs `8bit`) the same way `config_sha` refuses a drifted recipe. Pre-#367 evidence without a stamp warns rather than failing closed. This is a correctness-of-absolute-numbers fix: paired NF4 vs bf16 comparisons that judged both arms the same way remain valid.

- **`soup train` left a run at `status='running'` forever when setup failed (#764 by
  @SID-6921 in #767).**
  The tracker registered a run before loading the model, but the only `except` that marked
  a run failed wrapped the training call alone — a tokenizer/model load, quantization, LoRA
  attach, or dataset mapping failure in the ~250 lines before it left the row stuck at
  `running` with no way to tell it apart from a live run. On a real database this was 220 of
  432 rows, the oldest since 2026-03-04. The failure boundary now wraps everything from right
  after the run is registered through the end of the command, and `fail_run` records the
  exception type and message on the row (`runs.error_message`, added via the existing lazy
  migration). Separately, `soup train` now records its own pid via `mark_running` the way the
  MCP execution path already does, so `_reconcile_orphaned_run` (#401) can rescue a
  SIGKILL'd or Ctrl+C'd run on the next `soup runs` instead of leaving it invisible to a
  mechanism this project already built for exactly that purpose. `error_message` is
  redacted and length-capped before storage, and `soup runs show` now surfaces it and
  reports `terminated`/`launching` correctly instead of falling back to `running` for
  every non-completed, non-failed status.

- **`train_on_responses_only: false` trained on a doubled BOS, one token off from
  inference after #782 (#788 by @abdulwaarith0).**
  The default SFT path (`train_on_responses_only: true`) builds its ids with
  `add_special_tokens=False` in `data/loss_mask.py`, so it never doubled the BOS. The
  opt-out text path did: `data/sft_format.py` handed TRL a `{"text"}` column, and TRL's
  language-modeling `tokenize_fn` re-tokenised it with the tokenizer's default
  `add_special_tokens=True`. A chat template that renders `{{ bos_token }}` (Llama-3,
  Gemma, Mistral) on a tokenizer whose post-processor also prepends BOS therefore trained
  on `[bos, bos, ...]`. Before #782 that matched inference by accident; after #782 inference
  sends one BOS, so adapters trained on the text path were off by one token from how they
  are prompted. `soup data preprocess` tokenised the same way. The text path now
  pre-tokenises with `add_special_tokens=False` (via `build_full_sequence_labels`), so the
  chat template is the single source of special tokens, matching inference and HF's own
  `apply_chat_template(tokenize=True)`: one BOS for a `{{ bos_token }}` template, and zero
  for a template that renders no BOS at all — both Soup's `data.chat_template` presets and
  native templates of that shape (TinyLlama, Zephyr). The stop token is preserved rather
  than lost. TRL's language-modeling path also appended `eos_token` as a string to every
  text row that did not already end with it (`add_eos`), independent of the tokenizer's
  post-processor, so the pre-tokenised row reproduces that rule: the EOS id is appended
  when the sequence does not already end on it. A `{{ bos_token }}` template on a BOS-only
  or EOS-less (Qwen-shaped) tokenizer therefore keeps the single trailing stop token it
  trained on before. This changes training results in two ways on configs that hit the
  affected shapes: a template that renders no BOS (TinyLlama / Zephyr, not only Soup's
  presets) now trains with zero leading BOS rather than the tokenizer's default one, and a
  row that used to end on two trailing EOS tokens — a template or post-processor that
  appended one on top of `add_eos`'s — now trains on exactly one. The rule applies only to
  text a chat template rendered; pretrain-style raw text with no template keeps the
  tokenizer's own special tokens unchanged. `soup data doctor --show-mask` now renders
  through the same builder, so the X-ray matches training. Adapters trained with
  `train_on_responses_only: false` before this change carry one extra leading BOS in
  training compared with inference. `soup data preprocess` de-duplicates the same leading
  BOS but stays pinned to the old cache contents otherwise: it tokenises as before
  (`add_special_tokens=True`) and drops only the one duplicated leading BOS, so every other
  token is unchanged — including the post-processor's EOS and the room HF truncation
  reserves for it, so a truncated row still ends on its stop token. Its EOS therefore
  remains whatever the post-processor added (never TRL's `add_eos`), so a Qwen-shaped
  tokenizer's cache and live path can still differ on the EOS — that pre-existing mismatch
  is tracked separately in #791 rather than changed here. The cache also keeps `main`'s
  single leading BOS where the live path now yields none for a `data.chat_template` preset;
  that BOS divergence is tracked in #876. The `soup data preprocess` cache
  key changes so datasets cached under the old encoding are re-tokenised rather than reused.

- **`docs/commands.md` covers every registered leaf — including `soup mcp runs reconcile` — behind a Typer-walking coverage ratchet (#822 by @MrBeldum in #855).** The matcher requires a real entry row (token-bounded, no prose-only hits, one `|`/`/`-connected `a|b|c` shorthand group) so deleting a command's own rows fails the guard.

- **Require both models and valid probe keys in edit diff (#818 by @kok-o in #863).**
  `soup edit diff` previously allowed probes without models to fall through to a
  placeholder path that reported `changed: false` for each unmeasured probe and exited 0.
  Now, specifying `--probes` requires both `--before-model` and `--after-model` (exiting 2
  naming missing flags if omitted), asymmetric model flags are rejected naming the partner
  flag, and probe files with zero valid rows or missing the `'prompt'` key exit 2 with an
  explicit diagnostic rather than silently producing placeholder claims.

- **`soup serve --backend vllm` no longer sends the engine a doubled BOS (#785 part 2 by @abdulwaarith0 in #867).**
  #782 fixed every path that tokenizes in process, and left the engines that
  tokenize the prompt string themselves for someone with the hardware to check.
  Measured on vLLM 0.29.0 with `unsloth/Llama-3.2-1B-Instruct`, whose Llama-3
  template renders `{{ bos_token }}` (id 128000): for the string Soup sent, the
  engine reported `prompt_token_ids[:2] == [128000, 128000]` over 37 ids, against
  the 36 ids `apply_chat_template(tokenize=True)` returns. The vLLM backend now
  hands the engine `{"prompt_token_ids": [...]}` built with no tokenizer special
  tokens whenever the chat template rendered the prompt, which reproduced the
  36-id encoding exactly (`[128000, 128006]`), so the model is prompted with the
  prefix Soup trains on. `usage.prompt_tokens` drops by one on such templates.
  The change is wider than removing a duplicate: for a model whose chat template
  renders no BOS but whose tokenizer post-processor would add one, the engine
  previously received one BOS and now receives zero, matching Soup's training
  path and the in-process fix #782 already shipped for the transformers backend.
  Models without a chat template, and templates that fail to render, still send
  the legacy string and are tokenized by the engine exactly as before. The
  SGLang and DeepSpeed-MII backends were measured through their own tokenizer
  paths and are tracked separately in #785.

- **The README translation ratchet no longer passes a re-stamp without a re-translation (#870 by @Ercaner1988, follow-up to #852).** The sync stamp proves when a translation was synced, and the structure check proved headings, code-block languages, URLs and inline code survived — but a command changed inside a fence (`soup train` → `soup trainx`) or a figure changed in prose (`119.6` → `999.9`) stayed green behind a valid stamp. Each `## ` section of a translation is now also compared with `README.md` on its code-block contents with `#` comments dropped (comments being the only part of a code block a translation may change), its relative link targets, and its numbers in prose; in-page anchors, which are translated with their headings, must resolve to a heading of the same file. Measured figures therefore keep `README.md`'s digits in every language: `README.tr.md` had 10 figures written with Turkish separators (`119,6`, `3,32 GB`), now converted. Each translation names its maintainer in the test, and a red run says whom to ping.

- Fixed `soup doctor` exiting 0 after reporting a `[train]` package as INCOMPATIBLE: an installed package beyond its declared ceiling (`transformers`, `peft`, `trl`) now makes the command exit non-zero, while a missing or outdated `[train]` member stays advisory (#874 by @chiliec in #877).

- **`soup ingest --source langfuse --pull` is bounded in time and filtered on the answer,
  not only on the request (#865 by @Konuktor in #881).** Three findings from the #859
  review. The timeout reached the socket rather than the call, so a server sending a byte
  before every timeout expired kept the pull alive indefinitely — a 0.4 s timeout allowed a
  2.01 s call against a drip server; each request now carries a wall-clock deadline covering
  the connect and the whole response, and the body is read in chunks so the deadline is
  checked while bytes are still arriving. `type=GENERATION` was only a query parameter, so a
  server that ignored it would have written spans and tool calls out as training rows; each
  observation's own type is now checked before it becomes a row, the parameter is still sent,
  and skipped observations are counted in the summary. A pagination cursor the server repeats
  used to run until `--max-pages` — 100 identical requests by default — and now stops on the
  first repeat. No change to the default (no `--pull`) path, which still imports no HTTP code
  and opens no connection.

- **`soup mcp serve`'s `data_validate` no longer reports every row valid for a format it does not recognise (#878 by @Srinivasan8888 in #882).** `validate_and_stats` enables its format check only when the requested format is in `VALID_FORMATS`, so an unrecognised value silently turned the check off and a missing one never turned it on — both answered "every row valid" for a reason unrelated to the data. `soup data validate` was guarded against that in #869, so the same file got two different verdicts depending on whether the CLI or the MCP tool asked. The handler now resolves `format` the way `data_doctor` beside it already did: anything outside `VALID_FORMATS` raises a tool error naming the accepted values, an omitted or `auto` format is auto-detected as the tool schema always promised, and the resolved format is returned in the report so a caller can see which one was checked. The allowlist is `VALID_FORMATS` itself rather than a copy, since a hand-written second tuple is how the two surfaces drifted apart to begin with.

- **`soup doctor` now handles modern NVIDIA CUDA driver headers and supported PyTorch wheel indexes (#827 by @Samearth17 in #883).** CUDA detection accepts both `CUDA Version: X.Y` and `CUDA UMD Version: X.Y` headers with flexible whitespace, and PyTorch wheel selection now uses only indexes that provide releases compatible with Soup's `torch>=2.6.0` floor. An unknown driver version now returns `None` instead of defaulting to `cu121`, and drivers reporting CUDA 12.1–12.3 now map to `cu118` because `cu121` is not in the supported wheel table. When no supported CUDA wheel can be selected, `soup doctor` falls back to `pip install "soup-cli[train]"` instead of generating an invalid `whl/None` index. The GPU diagnostic also distinguishes a CUDA-enabled Torch build that cannot initialise CUDA from a genuine CPU-only Torch build.

- **`soup doctor` names missing `[train]` packages on a partial stack (#875 by @HarshRajSinghania in #884).**
  A six-of-seven install no longer prints "Training stack not installed".

- **The `serve-fast` extra now floors `vllm>=0.4.3`, up from `0.4.0` (#892 by @abdulwaarith0).**
  Since #867 the vLLM backend hands the engine a `{"prompt_token_ids": [...]}` mapping as the
  first positional argument to `AsyncLLMEngine.generate`. vLLM only accepts a mapping there from
  0.4.3 on, when `generate`'s first parameter became `inputs: str | TextPrompt | TokensPrompt |
  TextTokensPrompt`; 0.4.2 and earlier take a `prompt` string plus a separate `prompt_token_ids`
  keyword, so on those versions the mapping lands on the string parameter and is mis-handled. The
  old `0.4.0` floor therefore advertised support the backend no longer has.

- **`soup ingest --source langfuse --pull` printed the server's error body to the terminal
  with its raw ANSI escapes intact, letting a hostile Langfuse response inject terminal
  control sequences (#885 by @wangzhengzhuo05 in #893).**
  `commands/ingest.py` rendered `PullError` through bare `rich.markup.escape`, which
  neutralises `[...]` markup but passes C0 control bytes (ESC, NUL, BEL) straight through, so a
  crafted response body could reach the terminal as live escape sequences. The user-facing
  path now routes the message through the shared `utils/terminal.py:for_terminal` helper (C0
  strip + markup escape) at both `PullError` print sites, so the server's text is still shown
  but its control bytes are not. Added by @wangzhengzhuo05.

- **`soup draft distill` and `soup shrink` report a validation failure in their own rendered config as a CLI error rather than a traceback (#895 by @1cbyc in #904).** Both commands validate the YAML they generate before starting a training subprocess and sanitize the validation message for terminal output.

- **`soup train --config` on a YAML document that is a list or a scalar now exits 1 with
  `Config must be a YAML mapping, got list` instead of a `TypeError` traceback (#905).**
  `load_config_from_string` had refused that shape for releases; the file call site never
  did. The unknown-config-key report also no longer claims to be capped when a config has
  exactly the cap's worth (100) of unknown keys and nothing was dropped.

- **The config-field consumer guard no longer counts a read inside a function nothing calls (#807 by @Srinivasan8888 in #906).** The guard added in #748 treated a field as wired if its name was read anywhere under `src/`, so `training.warmup_auto` passed on one line in `autopilot.generate_config` — a function with no caller, whose line reads the autopilot decisions dict and writes the value *into* a config rather than reading the field. The file already refused exactly this for `schema.py` `@property` resolvers and did not apply the same reasoning to ordinary functions. A read now counts only when the function enclosing it is referenced somewhere in `src/`; one live read still rescues a field that also has dead ones, and a module-scope read always counts because module scope runs on import. `LoraConfig` is also in scope for the first time, which surfaces `lora.loftq_iter` and `lora.loftq_bits` as having no read at all. Six fields are allowlisted with their issue references rather than left failing, so the number can only shrink as #794 and #808 are answered.

- **Six command modules, `utils/ship_verdict.py`, `mcp_server/registry.py`, and `commands/serve.py` carried a private copy of the terminal control-byte sanitizer instead of importing the shared one (#896 by @VanshSharmaPES in #907).** v0.75.0 added `soup_cli.utils.terminal.for_terminal` (strip C0/DEL control bytes, then escape Rich markup) for the config loader's unknown-key report, but nine other call sites — `commands/adapters.py`, `data_canary.py`, `data_doctor.py`, `draft.py`, `infer.py`, `shrink.py`, `serve.py`'s `--auto-spec` pairing message, `utils/ship_verdict.py`, and `mcp_server/registry.py` — still carried a byte-identical `_CONTROL_STRIP_TABLE` under their own name, which is exactly how the config loader got missed for a release. All nine now import `for_terminal` (or the new strip-only `strip_control`, for the handful of call sites that feed non-markup contexts like `Text.append(..., style=...)` or an exception message escaped later downstream) from the shared module, and a new repo-wide test (`test_issue896_no_duplicate_terminal_sanitizer.py`) fails CI if a module besides `utils/terminal.py` ever defines `_CONTROL_STRIP_TABLE` again.

- **Short training runs no longer show a fabricated `0.0000 -> 0.0000` loss delta (#899 by @taljeon in #908).**
  All Hugging Face-style trainer wrappers now share one loss-history summary: two or more
  per-step measurements produce a delta, one measurement or a run-level `train_loss` mean
  produces one value without an arrow, and missing data is shown as unavailable. The
  completion panel therefore never claims a before/after comparison it did not measure,
  while non-finite per-step values remain visible instead of being replaced by the last
  healthy loss.

- **`soup plan` and `soup apply` now refuse a config carrying an unknown key (#894 by @wangzhengzhuo05 in #909).**
  `_load_yaml_config` read the YAML with `yaml.safe_load` and handed the raw mapping to
  `build_plan`, so it never built a `SoupConfig` and never ran `find_unknown_config_keys`:
  a typo'd key planned cleanly and was refused one command later by `soup train`. Both
  commands now exit 1 with the loader's message and suggestion.

- Repinned `notebooks/proof-4gb.ipynb` from an unreproducible `git+...@main` install to
  the released `soup-cli[train]==0.75.0`, corrected its "0.73.0 is published" and
  "#347 is open" claims (#347 shipped `training.stream_vram_override` in v0.73.1), added
  that override to the section 5 config so the pre-flight checks against the same 4 GB
  cap the allocator enforces, and added a cell printing `torch`/`transformers`/`peft`/
  `bitsandbytes`/`soup_cli` versions. Colab's free-tier T4 runtime now defaults to
  Python 3.13, ahead of soup-cli's `<3.13` ceiling, so the install cell falls back to
  `pip install --ignore-requires-python` when it detects that. Re-ran the notebook end
  to end on a free Colab T4 and committed it with outputs: section 4's streamed-vs-
  resident forward comparison is bit-exact (`torch.equal = True`), and section 5 trained
  Llama-3.1-8B under the 4 GB cap with a measured peak of 1.83 GB. Added
  `tests/test_issue844_notebook_pin_guard.py` so a future stale pin, an unreleased
  version, or a renamed `soup_cli` import fails CI instead of surfacing on the next
  Colab run (#844 by @VanshSharmaPES in #910).

- Fixed `soup eval against` so it can compare results written by Soup's own eval
  commands, rejects unknown metrics before database access, distinguishes unavailable
  comparisons from regressions in generated hooks, and reports single aggregate scores
  without pretending they form a confidence interval (#812 by @Amix29 in #914).

- **The Web UI now rejects oversized YAML-entry request bodies before parsing
  (#897 by @Amix29 in #915).** `/api/config/validate`, `/api/train/start`, and
  `/api/config/from-form` return HTTP 413 above 1 MiB. The limit checks both the
  declared `Content-Length` and the bytes actually received, so chunked requests
  and understated headers cannot bypass it; all 171 shipped recipes remain valid.

- **`soup data lint` no longer fails correct conversational DPO data on `prompt_leak`, and `length_bias` needs a real size difference (#810 by @dchaudhari7177 in #916).** A conversational `chosen` / `rejected` carries its prompt as the leading user turn, and the linter flattened every turn, so `prompt_leak` flagged every row whose prompt was 40+ characters, including both shipped DPO examples, and `length_bias` measured prompt plus answer. Both checks now read only the assistant turns. `length_bias` also requires the mean lengths to differ by at least 10% for MAJOR and 5% for MINOR, on top of Cohen's d, so a consistent one-word gap between near-constant lengths (or the zero-variance fallback) is no longer MAJOR on its own.

- **`soup eval quant-check` now exits with code 2 on a `MAJOR` accuracy regression verdict, rejects standalone `.gguf` paths up front (exit code 3), and requires `--allow-stub` for stub scoring (#809 by @kok-o in #917).** A model load failure on either side now reports which side failed and exits with code 1 instead of silently falling back to deterministic stubs and reporting a false `OK`. Breaking: a run that previously passed on stubs now needs `--allow-stub`. When `--allow-stub` is explicitly provided, stub outputs are marked with `"stub": true` in JSON and display visible warning banners in table/markdown output.

- **`soup reward stress` now detects structure-preserving attacks that game built-in format and substring verifiers (#814 by @kok-o in #918).** `reward_stress` previously checked four classic junk strings (`empty`, `length`, `repetition`, `sentinel`), none of which exercised the structure rewarded by reasoning scaffolds. Consequently, the built-in `format` reward was reported as "robust (not gameable)" despite awarding 1.0 to `<think></think>a`, and `accuracy` missed distractor answer-sprays. `soup reward stress` now adds two structure-preserving attack families (`wrapped_junk` and `answer_spray`), expands each family into multiple distinct variant attempts, determines verdicts per-family to prevent denominator dilution, tests `answer_spray` against real gold substitutions with non-collapsing distractor generation, and ensures `verifiable` math checks remain robust (exit 0).

- Apply `data.chat_template` consistently to conversational preference, reward-model,
  GRPO, and Online DPO trainers, and reject it on non-chat tasks
  (#804 by @Amix29 in #919).

- **`soup profile` no longer reports an assumed GPU as a measured fit, and knows newer and laptop cards (#830 by @dchaudhari7177 in #920).** With no GPU detected and no `--gpu`, it printed "OK Fits in 24 GB VRAM" and a batch size as if it had measured a 24 GB card; it now says no GPU was detected, shows the would-fit result against the assumed size, and asks for `--gpu`, and `--json` and the MCP `profile` tool report `gpu_memory_source` (`flag`, `detected` or `assumed`). `--gpu` now accepts the RTX 3050, 5060 through 5080 (including Ti variants), B200 and laptop parts whose memory differs from the desktop card, as well as device names as torch reports them ("NVIDIA GeForce RTX 5070 Laptop GPU"). The speed estimate is labelled as a typical A100 figure.

- Build PEFT LoRA configs through one shared helper so `rank_pattern` and
  `alpha_pattern` work consistently outside SFT and pretraining
  (#803 by @Amix29 in #923).

- Label the legacy `soup data toxicity` scorer honestly as an abuse-keyword heuristic,
  stop treating ambiguous technical terms as safety signals, preserve legacy output
  keys, and remove it from Magpie's default quality filter
  (#821 by @Amix29 in #925).

- Stop Autopilot from enabling or advertising unwired forgetting and checkpoint-quality
  gates, and warn when any staged sibling setting differs from its default. (#927 by @Amix29)

- Configure Transformers RoPE before text SFT and pretrain model construction, because changing
  the config after construction left the rotary frequencies unchanged (#793 by @Amix29 in #931).
  Preserve only type-independent native parameters when changing algorithms, forward YaRN
  tunables, and refuse nested per-layer RoPE configurations rather than reporting a partial
  extension as enabled.

- **Sharding a large checkpoint no longer dies on Windows.** `shard_checkpoint` held every
  source `*.safetensors` mapping open for the whole of pass 2, and `safe_open` maps a file
  privately on Windows, so each live handle charged commit for that file's entire size —
  measured, 48.99 GB mapped raised the commit charge 46.17 GB while free physical memory never
  moved. A real Llama-3.1-70B (30 files, ~141 GB) therefore charged the whole checkpoint and
  died with an access violation (no traceback, exit 139), so it could not be sharded for layer
  streaming at all. At most two source handles are now alive at once, and the tensor readers
  return tensors that own their memory: a `get_tensor` view keeps its whole file mapped, and
  charged, for as long as it lives, which would otherwise defeat the cap. (#934, fixes #926)

- Discover installed Soup plugins lazily, persist explicit opt-in state across processes, and
  connect enabled hooks to training (#819 by @Amix29 in #935). Third-party entry points are now
  listed from package metadata without importing their code; `soup plugins enable <name>` is the
  explicit code-loading boundary. Malformed recursive state and Rich-markup resource names are
  handled safely. **BREAKING:** the non-installing `soup plugins install` advisory now exits 2
  instead of reporting success with exit 0.

- **`soup data preprocess` cached chat rows without the EOS the live path trains on,
  so a cached run could train with no stop token (#938 by @abdulwaarith0).**
  The live SFT path (`train_on_responses_only: false`) reproduces TRL's `add_eos` rule
  in `data/loss_mask.py`: a chat row that does not already end on `eos_token` gets one
  appended. `soup data preprocess` tokenised the rendered template directly, with no such
  step, so a cached row carried an EOS only when the template or the tokenizer's
  post-processor supplied one. On a template that renders no EOS and a tokenizer whose
  post-processor appends none (the Qwen shape), the live path trained on one stop token
  and the cached path on zero. It was silent: the loss curve looks normal and
  `soup data doctor --show-mask` renders the live path, not the cache. The preprocess
  cache now applies the same `add_eos` rule for chat rows, appending the EOS id when the
  row does not already end on it, and only while the row is under the length budget so a
  truncated row is not pushed past `max_length` (the live path, which appends then
  truncates, keeps no trailing EOS there either). Pretrain rows never go through a chat
  template and are unchanged. The leading BOS still differs between the cache and the live
  path by design and is tracked in #876. The `soup data preprocess` cache key changes, so a
  dataset cached under the old encoding is re-tokenised rather than silently reused. Models
  trained from a `soup data preprocess` cache before this change may have trained without a
  stop token on templates that render none, which teaches run-on generation; re-run
  `soup data preprocess` and retrain to pick up the stop token.

- Rewrite the shipped DPO and chat-preference examples so rejected responses retain a clear quality contrast without teaching a response-length shortcut, and regression-test both files with `soup data lint` (#929 by @taljeon in #940).

- **`soup reward stress verifiable --verifiable-domain json_schema` now forwards supplied
  references as `schema` metadata instead of reporting that `--references` is missing (#937 by
  @Amix29 in #941).** JSON-object schema fields are decoded before scoring, and a verifier that
  returns the wrong number of scores after receiving references now gets a distinct contract
  diagnostic. The built-in attacks remain non-JSON and therefore cannot yet discriminate schema
  strictness; `reference_accept` is not a meaningful control for this metadata-based domain.

- **Distillation gradient accumulation now weights microbatches by their shifted, non-masked
  training-token counts (#932 by @Amix29 in #943).** This supersedes #742's fixed-per-microbatch
  compensation: the trainer now accepts Transformers' accumulation count and normalizes each
  microbatch by its share of the window's trained tokens. Unequal-length microbatches therefore
  produce the same gradient as the equivalent full batch instead of giving short and long
  microbatches equal influence.

- **CLI help, config descriptions, and feature guides now describe shipped behaviour instead
  of expired release plans (#823 by @Amix29 in #945).** Removed stale `schema-only`, `deferred`, and
  `lands in` claims from live features, removed internal release-phase labels from command
  help, and stopped `soup advise run --help` from recommending the nonexistent `--no-record`
  flag. Descriptions for genuinely incomplete surfaces now state their current limitation
  without promising a past release. A version-aware source-level ratchet scans every Typer help
  string and Pydantic field description while allowing documented future release notices and
  explicitly allowlisted schema-only fields.

- **Sweep leaderboards now keep diverged arms visible but sort them after finite losses (#944 by @taljeon in #946).** Non-finite losses are labelled `diverged`, never selected as the best run, and explicitly count as worse than an early-stop threshold instead of relying on NaN comparison behavior.

- **The Web UI now caps `/api/chat/send` and `/api/data/inspect` request bodies before they are parsed (#939 by @AmirF194 in #948).** Both routes let FastAPI read and parse the whole body before the handler's own checks ran, so `chat/send` forwarded an unbounded body upstream and `inspect` paid a full read even on a path that was always going to be rejected. `chat/send` now caps at 1 MiB, `inspect` at 8 KiB; both check `Content-Length` and the bytes actually received, so a missing or understated header cannot bypass the limit.

- **`autodistill`'s MLX teacher worker verification failed under a venv on Windows,
  where `Scripts/python.exe` is a launcher that spawns the real interpreter as a
  grandchild (#898 by @SID-6921 in #950).**
  `run_mlx_teacher_capture_process` compared `receipt.worker_pid` (the child's own
  `os.getpid()`) against `subprocess.Popen(...).pid`. Under a real venv those are two
  different processes, so the comparison always failed even on a successful capture.
  The parent now generates a nonce and hands it to the child in the request; the child
  echoes it back into the worker receipt, and the parent verifies the nonce instead of
  the PID. `worker_pid` still appears on both models -- the receipt's own value
  (`os.getpid()`) is the useful one; the result's mirrors whatever `Popen` reported,
  which is the launcher's PID on exactly the platform this fixes, so it is informational
  only, not proof of anything.

- **The control-strip ratchet skipped a file it could not parse, so a broken module could
  hide behind a green run (#949 by @jarik2014 in #951).**
  `tests/test_issue896_no_duplicate_terminal_sanitizer.py` walks every tracked file under
  `src/soup_cli/` with `ast.parse` and a `SyntaxError` was caught and skipped, which made the
  guard's clean answer mean "zero offenders among the files I could read" rather than "zero
  offenders" — and a module that does not parse is a worse problem than the duplicate table
  the ratchet exists to find, since it would surface later as an `ImportError` somewhere
  unrelated. The guard now fails naming the file and the error, and a test proves it by
  writing a real unparseable file inside the scanned tree (registered with `git add -N` so it
  is part of the `git ls-files` walk) and asserting the guard goes red.

- **`soup mcp serve`'s `data_doctor` now refuses blank and unknown formats instead of treating blank as auto-detect or falling through to unrelated tokenizer/doctor errors (#889 by @Rayan-and-beyond in #952).** The handler now mirrors `data_validate`'s #882 behavior: omitted `format` still auto-detects, explicit values must come from the shared `VALID_FORMATS` allowlist, and an invalid value raises the same bounded error naming the rejected value and accepted formats. Tests pin blank and near-miss rejection, message parity, every current valid format, allowlist extension without a handler change, and the omitted-format control.

- MCP export and profile errors now name the rejected format or GPU and list the
current accepted values. Error formatting is shared with data validation, and
control characters in rejected values are escaped (#953 by @shubhambhar007 in #954).

- Reject `training.kernel_auto_compose: true` instead of timing one unchanged model and falsely reporting that the fastest kernel combination was applied. Use `use_liger` or `use_flash_attn` explicitly. (#801 by @Amix29 in #955)

- Narrow the MLX backend-support table to the four reachable CUDA-only fields (`use_mod`, `moe_lora`, `quantization_aware`, `use_fsdp2_compile`); the other five from #958 are refused by schema validation before MLX ever sees them (#958 by @ege-arhan in #961).

- **Data recipes no longer report an unreachable provider as a successful run (#930 by @taljeon in #969).** Live `llm_text` and `judge` nodes now count transport, non-2xx, and malformed-response failures; a node whose every provider call fails exits non-zero and names the endpoint. Partial runs keep their successful rows and report the failure count, while a legitimate empty completion remains a successful response rather than being mistaken for an outage.

- **Layer streaming's disk tier no longer reads on the compute thread (#971 in #973).** `stream_source: disk` (and `auto` when the base does not fit RAM) used to call a synchronous, memory-mapped read from the thread that was supposed to be feeding the GPU, so every byte arrived through page faults while the card idled. The base now streams through a background reader that parses each shard's header itself — no memory map, so nothing charges commit for the whole file — and stages layers into host memory that is page-locked when the box allows it, with a loud pageable fallback rather than a silent one when it cannot; setting `training.stream_pin: true` on this tier now REFUSES a run the box cannot page-lock, where it previously announced that pinning does not apply and went on regardless. Measured on one box (RTX 5070 Laptop, Windows, NVMe; [record](benchmarks/gate-971-async-nvme-source.md)) against a **same-day control of the source it replaces**, cold on a 36 GB 70B-shaped NF4 store larger than RAM at batch 1 × seq 512: **92–100 s per step became 30–48 s (2.1–3.1×)**, each pair position-matched — 5.1–5.6 → 10.6–16.9 tok/s, 0.70–0.76 → 1.46–2.32 GB/s at the source — with peak VRAM unchanged at 4.38 GB. It is still bound by the read: per-layer read brackets are 82.6–84.9% of the step. **Warm, with the whole store in the page cache, it is a REGRESSION of 1.03–1.20×** (the two figures are the two run orders; the direction survives reversing the order, the magnitude does not) — with the store cached a synchronous read is close to a `memcpy`, so there is nothing for a reader to hide; if the store fits RAM the RAM tier is 1.8–2.3× faster than either disk arm and is what `auto` already picks. A shard that changes on disk while a run holds it is refused by name rather than read at stale offsets, and a reader that wedges mid-read now surfaces as an error instead of a hang. New `training.stream_read_ahead` (default 2, range 1–8) sets the reader's depth, where **a setting of N stages N−1 layers ahead** because one slot is always held by the layer being consumed, and each level costs one layer of pinned host RAM (1.5 GB at depth 1, 2.0 at 2, 2.8 at 4 on a 70B) — which the pre-flight now charges, prints and refuses the run over if it will not fit free RAM, instead of predicting zero host residency for this tier. No depth was distinguishable on throughput in that record once run order was controlled for, so treat it as a host-memory knob. A non-default `stream_read_ahead` beside `stream_source: ram` is **refused** rather than accepted and ignored: the reader belongs to the NVMe tier and `ram` never falls back to it. The default is accepted, because a default is not a decision.

- **Enabling MiniLLM with the default offline settings is now a config error instead of a silent no-op (#692 by @jagadeepmamidi in #977).** `minillm_enabled: true` plus `minillm_on_policy: false` plus `minillm_teacher_mix_ratio: 0.0` used to load and forward the teacher, then blend it at weight zero so the MiniLLM term was `KL(student || stopgrad(student))`. Ordinary cross-entropy kept the loss curve healthy. That combination is rejected at parse and names the three fields; ratio 0 stays legal on the on-policy path, where it only means student-only sampling and the loss is still `KL(student || teacher)`.

- **BitNet 1.58 training configs now fail at config load instead of falling through to the ordinary trainer and later reporting `unknown quantization` (#825 by @Rayan-and-beyond in #981).** The error states that BitNet training is not implemented yet and points to the still-working `soup export --format tq1_0` path. The unusable `falcon-e-bitnet-sft` recipe and the docs that advertised training as live are removed, with recipe snapshots and catalog counts kept in sync.

- **`soup loop watch`'s canary deploy stage no longer reports a canary verdict of OK when nothing evaluated a canary (#815 by @AmirF194 in #982).** `deploy_to_canary` activates the trained adapter at 100% traffic and recorded `canary_verdict: "OK"` from that activation's bare 2xx; the actual per-request traffic split and verdict machinery in `canary_router` never ran. The stage now reports `canary_verdict: None` on that path, matching every other exit in the function.

- `training.use_flash_attn` and `training.use_liger` are now rejected at config load for any task outside `sft`/`tts`, the only trainers that read either flag. `soup autopilot` stops auto-enabling both for DPO/GRPO/pretrain on Ampere or newer, where they previously built a config the CLI's own validation would then refuse for a package that task never used. `soup migrate --from axolotl` and the `generate_config` autopilot emitter got the same scoping, so neither hands back a config the loader then rejects (#806 by @AmirF194 in #983).

- **Five tests that passed with colour off and failed with colour forced now answer the same way in every shell (#886 by @Srinivasan8888 in #984).** Under `FORCE_COLOR=1` they failed for reasons unrelated to what they assert, through three distinct mechanisms. A test fixture built its Rich `Console` without pinning `force_terminal`, so it colourised into the buffer and `"\x1b" not in out` became unsatisfiable; that fixture is now pinned, which keeps the broad assertion rather than narrowing it to hand-picked escape sequences. Two banner checks and two `soup doctor` NCCL checks matched raw output that colour splits with SGR codes, and now match ANSI-stripped output, as the newer tests in those files already did. The control-byte test also gained an assertion that the stripped payload stays visible as inert text, so a sanitiser that swallowed it entirely would no longer pass. None of the five was red in CI, where no workflow forces colour, but this class has turned CI red four times before when something upstream changed.

- **`soup edit diff` refuses both models with no `--probes` instead of writing an empty diff (#880 by @Srinivasan8888 in #985).** Given `--before-model` and `--after-model` but no probe file, it exited 0 and wrote a report with `total_probes: 0`, which reads as "the edit changed nothing" when nothing was measured. It now exits 2 without writing a file, and the message names `--probes`. The three combinations #818 and #863 already refuse are unchanged, and running with neither model and no probes still succeeds.

- Report unavailable training loss for MLX SFT and unlearn runs that capture no loss, instead of inventing 0.0000 (#960 by @Rayan-and-beyond in #986).

- **Layer streaming's pageable fallback no longer poisons the run, and the RAM store page-locks at its own size (#901 in #989).** A Qwen2.5-14B NF4 run on an 8 GB card with 32 GB of RAM predicted a 3.39 GB peak against 7.41 GB free, printed "could not page-lock the base … falling back to a PAGEABLE RAM store", and died with `AcceleratorError: CUDA error: out of memory` inside `SFTTrainer.__init__`. Memory was never the problem: a failed `pin_memory=True` allocation leaves the CUDA runtime's last error set, and the run's first kernel launch reported it as an out-of-memory with gigabytes free (`cudaMalloc`, `synchronize` and memcpy all pass; a kernel's launch check reads it). `_build_source` now drains that error and releases the page-locked blocks the abandoned attempt left in torch's host cache before it builds the pageable store, on both tiers. The page-lock itself failed because torch's caching host allocator rounds every pinned request up to the next power of two, so pinning the store one tensor at a time cost 1.73x its bytes (measured on the real 14B shard cache: 6.82 GB requested, 11.83 GB page-locked); `RamSource` now pins the store in power-of-two arenas with every tensor a view, so the same 9.93 GB store page-locks as 10 GiB and the ready line prints the real figure. The VRAM probe also calls an out-of-memory `AcceleratorError` an OOM verdict rather than an instrument failure. Measured on the dev box with a synthetic 14B shape: the reporter's config now trains end to end with the store pinned, peak 2.944 GB allocated / 3.244 GB reserved — the pre-flight's prediction was right. Record: `benchmarks/gate-901-14b-on-8gb.md`; the shape generator (`synth_checkpoint.py --shape qwen2.5-14b --arch qwen2`) and the step probe (`issue901_stream_step_probe.py`) ship under `benchmarks/harness/`.

- **The layer-streaming disk tier reads each layer through direct I/O in sector-aligned ranges, into arena-packed staging (#974 in #991).** Cold, on a 36 GB 70B-shaped NF4 store larger than RAM, every buffered read primitive topped out at 1.2–2.9 GB/s on the dev box's NVMe whatever the request shape — the cache manager, not the drive, was the ceiling gate-971's async source sat at (1.46–2.32 GB/s, 84.7% of the step) — while unbuffered reads of the same fresh layers reached 3.5–5.65 GB/s. The reader now reads each layer's data section as sector-aligned byte ranges through `FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux and `F_NOCACHE` on macOS (a buffered `open` where a filesystem refuses, e.g. tmpfs), four ranges in parallel through per-worker handles, into ONE staging region per slot with tensors as views, and packs the pinned regions into #901's power-of-two arenas so the disk tier's staging stops paying the per-tensor 1.7–1.9x rounding. Measured position-matched in both run orders on the dev box (RTX 5070 Laptop 8 GB, Windows 11): a cold 70B-shaped step went from 110.5 s on the synchronous path in the same slot to **16.2–17.3 s** (6.4x), against 34.8–45.6 s for the reader this replaces (2.0–2.8x), with the read share down from 84.7% to 62–64% at 6.0–7.0 GB/s and the two run positions within 1.07x of each other; peak VRAM unchanged at 4.38 GB. Warm, with the store in the page cache, the async source had been reported 1.03–1.20x slower than the synchronous one (gate-971 §5); position-matched it was 0.91–0.97x before this change — the bracket was run order — and after it, in one session with the reader it replaces imported from `origin/main`, the new reader is 0.76x of the old one warm on the plain point (1.582 vs 2.091 s; 0.94x on the instrumented points) and 2.25x faster right after a page-cache eviction (1.788 vs 4.024 s), with the RAM tier (1.13–1.20 s) still the floor for a store that fits. Two consequences worth knowing: a disk-tier run no longer warms the page cache for whatever runs after it, and a store that fits RAM still belongs on the RAM tier (`stream_source: auto`). The ready line prints the page-locked figure on this tier too; `direct_io`, `pinned_bytes` and `staging_bytes` are reported. Record: `benchmarks/gate-974-disk-tier-direct-io.md`; the read-primitive probes ship under `benchmarks/harness/issue974_*.py`.

- **CLI escape-hygiene tests no longer answer differently depending on the shell they run in (#987 by @AaronProbha18 in #993).**
  Seven tests assert that an untrusted field — an exception message, a filename, a
  manifest secret, a config digest — is sanitised before it reaches the terminal,
  spelled `assert "\x1b" not in result.output`. Each command prints through an
  unpinned module-level `Console()`, so under `FORCE_COLOR=1 TERM=xterm-256color`
  Rich styled the *surrounding* panel markup and the assertion failed even when the
  field itself had been sanitised correctly — the test was measuring the ambient
  terminal, not the code under test. Each now pins its console with
  `Console(force_terminal=False)`; `no_color=True` alone is not sufficient, because
  it suppresses colour without pinning tty detection, which is the exact bug
  `test_replay.py` carried. No assertion was weakened, narrowed, or routed through
  an ANSI strip: all seven keep the broad `"\x1b" not in` form, and six are
  demonstrably falsifiable by a real sanitiser regression. `tests/test_cli_help_assertions_are_ansi_safe.py`
  records why this class is left to per-test pinning rather than added to the
  repo-wide ratchet — telling "asserts on console output" from "asserts on a
  pure-function return" is about an even split, and a guard that false-positives on
  half of what it sees gets deleted. Test-only; no runtime behaviour changes.

- **`soup serve --backend sglang` no longer sends the engine a doubled BOS (#890 by @abdulwaarith0 in #995).**
  The same defect #867 fixed for vLLM, on the backend that was split out because
  `sgl.Runtime.generate` only ever posts `{"text": prompt}` and the server then
  tokenizes that string with `add_special_tokens=True`; neither the request nor
  `ServerArgs` has a flag to turn that off. Measured end to end on SGLang 0.5.9
  with `unsloth/Llama-3.2-1B-Instruct`, whose Llama-3 template renders
  `{{ bos_token }}` (id 128000): for the string Soup sent, the engine ran on
  `[128000, 128000, ...]` over 37 ids (`meta_info.prompt_tokens` 37), against
  the 36 ids `apply_chat_template(tokenize=True)` returns. The same `/generate`
  endpoint takes `input_ids` and uses them verbatim, so the backend now posts the
  ids built with no tokenizer special tokens to it whenever the chat template
  rendered the prompt, which reproduced the 36-id encoding exactly
  (`[128000, 128006]`, `prompt_tokens` 36). Both the streaming and the
  non-streaming route go through it, and the response is the same
  `{"text", "meta_info"}` dict the app already parsed, so `usage`,
  `finish_reason` and the streamed chunks are unchanged apart from
  `usage.prompt_tokens` dropping by one on such templates. As with #867, a model
  whose template renders no BOS but whose tokenizer post-processor would add one
  now receives zero, matching Soup's training path. Models without a chat
  template, and templates that fail to render, still go through
  `Runtime.generate` as the legacy string and are tokenized by the engine
  exactly as before. The MII backend is fixed separately in #891.
  Packaging note: the ids POST is now bounded by a 300s timeout and issued
  through `httpx`, so `httpx>=0.24.0` joins the `[sglang]` extra; anyone
  installing `soup-cli[sglang]` picks it up automatically.

- **`soup serve --backend mii` no longer sends the engine a doubled BOS (#891 by @abdulwaarith0 in #996).**
  The same defect #867 fixed for vLLM, on the backend that was split out because
  `MIIPipeline.__call__` takes strings only and encodes each one through
  `HFTokenizer.encode`, which is `tokenizer.encode(input, return_tensors="pt")`
  with no way to say the special tokens are already there. On a BOS-prepending
  tokenizer whose template renders `{{ bos_token }}` (`unsloth/Llama-3.2-1B-Instruct`,
  `bos_token_id` 128000), that is `add_special_tokens=True` over a string that
  already starts with the BOS text, so the prompt tokenizes to
  `[128000, 128000, ...]`, 37 ids, against the 36 ids
  `apply_chat_template(tokenize=True)` returns. MII's seam is
  `ModelConfig.tokenizer`: `mii.pipeline(tokenizer=...)` accepts a wrapper, which
  `load_tokenizer` wraps once more and calls `encode(input, return_tensors="pt")`
  on, with the prompt object the pipeline was given, untouched. `soup serve` now
  builds the pipeline with such a wrapper around the same tokenizer whose chat
  template it applies, and hands a templated prompt over as a `TokenizedPrompt`:
  still a `str`, so the pipeline's contract holds, but carrying the ids built with
  no tokenizer special tokens, which the wrapper returns verbatim, so the engine
  runs on the 36-id encoding (`[128000, 128006, ...]`). If the engine ever reports
  a prompt length other than the one Soup encoded, the server logs a warning naming
  both rather than doubling in silence. As with #867, a model whose template renders
  no BOS but whose tokenizer post-processor would add one now receives zero,
  matching Soup's training path. Models without a chat template, and templates that
  fail to render, still hand the pipeline a plain string, which the wrapper encodes
  with the very call MII's `HFTokenizer` makes, so the engine tokenizes it exactly
  as before; a pipeline built without a tokenizer keeps MII loading its own, as it
  did. The wrapper is covered against MII 0.3.3's real `HFTokenizer.encode` contract
  in `tests/test_issue891_mii_engine_bos.py`. The SGLang backend is fixed separately
  in #890.

- **`docs/commands.md` and `docs/training.md` documented four `soup train
  --minillm-*` flags that were never wired as CLI options, so copying the
  documented command raised `No such option` and exit 2 (#979 by @SID-6921 in #998).**
  Only `--minillm-on-policy` was ever a real flag; `minillm_enabled`,
  `minillm_teacher_mix_ratio`, `minillm_pretrain_anchor_weight` and
  `minillm_pretrain_anchor_path` are `soup.yaml` fields with no CLI
  counterpart. Both doc sites now show the YAML config instead. Added a
  guard (`tests/test_issue979_documented_cli_flags_exist.py`) that walks
  `soup train`'s live Typer tree and fails when a doc names a `--flag` with
  no counterpart, so the next one fails a test instead of a support ticket.
  Building it turned up eight more pre-existing mismatches in unrelated
  subsystems (ULD strategy, RL mid-epoch checkpointing, echo-trap
  detection); those are tracked separately in #997 rather than folded into
  this fix.

- The distill ``_DistillTrainer`` test harness now derives ``compute_loss``
  closure bindings from the factory and ``TrainingConfig`` defaults, and fails
  with the missing name if a new free variable cannot be resolved. Nested
  functions such as ``_token_weighted_accumulation`` are not reported as
  missing, and the recompiled class is compiled under
  ``<compiled _DistillTrainer>`` so a ``NameError`` can no longer masquerade as
  ``distill.py`` (#990 by @jagadeepmamidi in #1008).

- **A LoRA target on `lm_head` or `embed_tokens` crashed the first forward pass on an untied streamed checkpoint (#1012 by @AmirF194 in #1019).** `build_streamed_model` runs `get_peft_model` before `install_streaming`, so peft wraps the boundary module in a LoRA tuner layer first and exposes its `weight` as a read-only property; `StreamedLargeLayer`'s substitution used a plain `{"weight": ...}` `functional_call`, which cannot set that property, and raised `AttributeError` at the first forward. `_large_layer_weight_param_name` now resolves the same indirection `_layer_name_map` already resolves for decoder-layer projections, substituting onto `base_layer.weight` when the boundary module is peft-wrapped. Verified bit-exact (`torch.equal`, not an approximate tolerance) against a resident LoRA-on-head model for logits, loss and every LoRA gradient. This fixes the forward crash only; saving and resuming a checkpoint with a LoRA-wrapped head or embedding still needs separate work, tracked on #1012.
- **`training.stream_layers` now refuses a LoRA target on `lm_head`/`embed_tokens` at config parse time (#1012 by @AmirF194 in #1019).** The forward pass above is correct, but `save_pretrained()`/`load_adapter()` on the streamed head/embedding boundary is not implemented yet, so a run with that config trained for hours and died at its first `save_steps`. Drop the boundary module from `training.lora.target_modules`, or train without `stream_layers`, until the save/resume follow-up lands.

- Apply `data.chat_template` to conversational PPO prompts, pass `answer`
  through in PPO `messages` format, and reject `data.chat_template` on non-chat
  tasks `prm`, `asr`, `moe_lora_routing`, and `unlearn`
  (#936 by @kok-o in #1021).

- **`training.nvfp4` and `training.fp8_attention` are now applied on `task: sft` post-LoRA (#800 by @kok-o in #1022).** `SFTTrainerWrapper._apply_quantization_aware` now delegates to `apply_v028_speed_memory` with `skip_cut_ce=True` so NVFP4, FP8 attention projections, and FP8 training now reach their converters on SFT runs without double-patching Cut-CE. Both flags are also gated at schema time against non-v028 tasks and the MLX backend to prevent silent no-ops.

- **Unify `SoupTrainerCallback` construction across all trainers and reject unsupported monitoring on tasks without callbacks (#802).** `training.loss_spike_recovery` and `training.grad_accum_auto_tune` previously reached the live callback only in `trainer/sft.py`; fifteen other trainers instantiated `SoupTrainerCallback` with watchdog settings only, causing spike recovery hint generation and grad-accum advisory monitors to default to `False`. All trainers now build callback arguments through `soup_callback_kwargs(tcfg, batch_size=..., output_dir=...)` in `monitoring/callback.py`. Tasks without a live callback (`prm`, `moe_lora_routing`, `unlearn`) explicitly refuse `loss_watchdog`, `loss_spike_recovery`, and `grad_accum_auto_tune` at config load time (#802 by @kok-o in #1023).

- **Refuse symlinks on Windows when reading the draft registry and writing the audit log (#820).** On Windows, `os.O_NOFOLLOW` does not exist and evaluates to `0`, allowing `os.open` to follow symbolic links. Three sites under `$HOME` (`~/.soup/`) relied on `O_NOFOLLOW` alone without `cwd` containment: `utils/draft.py:_read_registry`, `utils/draft.py:_registry_lock`, and `utils/audit_log.py:append_audit_event`. As a result, symlinks placed at `drafts.json` could leak arbitrary JSON into speculative decoding, a symlink at `drafts.json.lock` would follow and create its target via `O_CREAT`, and `append_audit_event` would append audit logs to arbitrary symlinked targets. Introduced `utils.paths.open_no_follow(path, flags, mode)` which combines `lstat` inspection (detecting `S_ISLNK` and Windows reparse points) with post-open `fstat` cross-validation on Windows to detect a TOCTOU swap. Migrated the three vulnerable sites, enabled symlink tests on Windows when privileges permit, and added a repo-wide ratchet preventing new bare `getattr(os, "O_NOFOLLOW", 0)` usages (#820 by @kok-o in #1024).

- Warn once when distillation teacher produces persistently non-finite logits.
  Consecutive non-finite distillation terms are accumulated directly on device using
  asynchronous PyTorch operations with zero host synchronization on the per-step path.
  A single diagnostic warning is emitted naming the teacher model, the consecutive step
  count, and likely root causes (unsupported dtype underflow, mismatched tokenizer, or
  overly small distillation temperature). Transient non-finite steps (such as standard
  GradScaler overflow recovery) reset the on-device counter with no sync and emit no
  warning (#887 by @kok-o in #1026).

- **`data.remove_unused_columns` now defaults to `false`, and an explicit `true` loads with a warning and is ignored; Soup v0.77 will refuse it (#759 by @Srinivasan8888 in #1027).** No trainer reads the field. The trainers that set the Hugging Face argument pass `False`, so a custom collator still receives the columns the model's `forward()` does not name, and the old declared default of `true` misdescribed every run. `true` is not refused yet, because `soup autopilot` writes the whole config, so every file it generated on the old default contains `remove_unused_columns: true`; those keep loading, and no run changes behaviour, because the effective value was always `False`. Loading one prints `data.remove_unused_columns: true has no effect and is ignored`, once per load; delete the line before v0.77. The field is not deleted, since an unknown key is now a hard error.

- **The config-field guard no longer reports a broken checkout when one directory is reached by two spellings (#988 by @Srinivasan8888 in #1029).** Its scan/import check compared paths with `==`, so on macOS, where the filesystem is case-insensitive but keeps the case you type, running the test file by a lowercased path failed six tests with a message telling you to fix `PYTHONPATH` or reinstall. The check now compares the two directories with `os.path.samefile`, which asks whether they are the same directory on disk. It checks both exist first, so a missing tree is still reported instead of crashing, and genuinely different trees are still caught. Test-only.

- **A streamed save that writes an adapter holding nothing now raises instead of reporting success (#1011 by @dchaudhari7177 in #1031).** Twice a `stream_layers: true` run wrote an `adapter_model.safetensors` that reloads as nothing (v0.72.0's `.inner.` keys, #1005's 0-tensor file under peft 0.21) and the only signal was a `UserWarning` at a later load. After the final save and on every `save_steps` checkpoint, the streamed trainers (`sft`, `dpo`, `orpo`, `simpo`, `kto`) now read the file's keys back and raise, naming the file and both counts, when it holds no LoRA tensors, a different number than the model trained, or any key carrying the wrapper's `.inner.` segment. Resident runs are unaffected.

- **`soup doctor --config` no longer prints an unverified source line for `training.use_fsdp2_compile` on MLX (#1001 by @dchaudhari7177 in #1032).** The reason cited `commands/train.py:1235`, which nothing pinned and which had already drifted once. It now names `validate_fsdp2_compile_config`, which `soup train` runs before resolving the trainer, and a test fails if any backend-support reason cites a `file.py:NNN` location again.

- **`soup serve --auto-quant` now refuses instead of selecting a format from timer noise (#816 by @Amix29 in #1035).** The pre-bind path timed three calls to a constant-returning stub, called the fastest few hundred nanoseconds a quality-gated evaluation, and could then force AWQ, GPTQ, or FP8 on a plain checkpoint through vLLM. The flag now exits 2 with an explicit explanation until Soup can load and evaluate each candidate; vLLM receives `quantization=None` and therefore follows the checkpoint's own metadata. Transformers and SGLang no longer print a selection they never apply, and the help plus serving guide state the current boundary.

- **SFT gradient-checkpointing tiers now make the distinct trade-offs they print (#796 by @Amix29 in #1036).** `medium` reaches Transformers as `every_n_layers=2` instead of checkpointing every block, while `selective` wraps only each decoder block's direct attention child and leaves HuggingFace full-model checkpointing off, including after PEFT enables it for a k-bit model. Consequently, `quantization: 4bit` plus `gradient_checkpointing: selective` can now use more peak memory than before because it no longer inherits PEFT's full-block checkpointing; this is the intended selective speed/memory trade-off. Architectures without a selectable attention child report and apply a full fallback instead of claiming an inactive tier. A four-layer tiny Llama now measures 4/4 checkpointed blocks under `full`, 2/4 under `medium`, and four attention-only wrappers under `selective`, with a real CPU training step and finite non-zero gradient norm in the selective arm.

- Wire PiSSA, OLoRA, LoftQ, and VeRA through the shared PEFT constructor used by
  every Transformers trainer. Unsupported backend combinations and pre-quantized
  PiSSA or integrated LoftQ runs now fail at config load instead of silently
  training a different adapter (#794 by @Amix29 in #1037).

- **A `format: pre_tokenized` run now accepts the cache `soup data preprocess` wrote (#1038 by @Srinivasan8888 in #1040).** Preprocess hashed the source format into the cache key while the training gate hashed `pre_tokenized`, so every cache was refused with "cache hash mismatch; re-run `soup data preprocess`", and re-running changed nothing; a list `data.train` raised `ValueError` instead. The gate now recomputes the key from the format preprocess recorded, and both sides hash a list the same way. A cache built from a different tokenizer, `max_length`, dataset or interleave strategy is still refused, and existing cache keys are unchanged. `metadata.json` also records the installed Soup version instead of a hardcoded `0.53.7`.

- **The FP8 and NVFP4 gate tests give the same answer on every machine (#834 by @Srinivasan8888 in #1041).** Two tests called the real hardware and torchao gates with a placeholder model, so they passed only while a gate refused on the test machine; on a Hopper or Blackwell card with torchao installed they failed with `AttributeError`. Each gate now has its own test with the gate patched both ways and a small real module, and the FP8 torchao check uses a model without attention projections so removing that gate can no longer go unnoticed.

- **Tasks that never apply `training.quantization` no longer record a quantised run, and an explicit `4bit` or `8bit` on them is deprecated; Soup v0.77 will refuse it (#795 by @Srinivasan8888 in #1043).** `distill`, `classifier`, `reranker`, `cross_encoder`, `prm`, `moe_lora_routing`, `unlearn` and `asr` never quantise their models, yet accepted the default `4bit`, stored it in the run's config, and let the VRAM pre-flight budget them as QLoRA jobs. For these tasks the default changes: an unset `quantization` now resolves to `none`. An explicit `4bit`, `8bit` or `load_in_8bit: true` still loads, because every config Soup dumped on the old default carries `quantization: 4bit` literally, but it prints a warning naming the task, resolves to `none`, and will be refused in v0.77; set `quantization: none`. A Quant Menu value or a 4-bit-only setting such as `bnb_4bit_quant_storage` is refused at config load, naming the task. PRM also refuses a `lora` block it never reads (`lora.r: 0` is allowed), and the pre-flight budgets PRM, and the classifier family and ASR without their LoRA opt-in, as full fine-tunes. `docs/training.md` lists which tasks honour the field.

- **Both FP8 training paths ask one hardware gate, with torch's own floors, and a request the card cannot run stops the run (#835 by @Srinivasan8888 in #1044).** `training.fp8_attention` refused Ada (sm_89) cards such as the RTX 40-series, L4 and L40S, which `torch._scaled_mm` supports, while `training.quantization_aware: fp8` converted the model on any card, Ampere included. Both now require compute capability 8.9 or newer. The `rowwise` recipes also need a torch that dispatches their kernel on the card (Ada 2.7, RTX 50-series 2.8, 11.x 2.10), are never available on Windows, and before torch 2.11 need a CUDA 12 build. When FP8 is requested and cannot run, the run now stops at setup on every trainer that reaches the converter (the eleven that call `apply_v028_speed_memory`, plus SFT's own call), naming the reason, instead of warning and training without FP8.

- `training.gradient_checkpointing: false` now reaches the model on a 4-bit/8-bit/mxfp4
  run (#797 by @AmirF194 in #1045). All 15 `prepare_model_for_kbit_training(...)` call sites across the
  trainers called it with no `use_gradient_checkpointing` kwarg, so peft's own `True`
  default silently overrode the config; `transformers.Trainer` only ever enables
  checkpointing and has no disable branch, so nothing downstream could turn it back off
  once k-bit prep had switched it on. Every call site now threads the resolved value
  through. This covers the k-bit-prep mechanism only; the TRL-config-default and unsloth
  mechanisms the issue also names are unchanged.

- **The AcceleratorError-as-OOM probe classification is now reachable and covered without a GPU (#1002 by @Areen-09 in #1047).** #989 classified an OOM arriving as `AcceleratorError` as a result (`oom=True`) rather than an instrument failure (`failed=True`), but its only coverage sat under `@requires_cuda` and skipped across all CI cells without a GPU runner. `measure_step_peak_bytes` now delegates its failure mapping to a pure helper `_classify_probe_exception` whose three-outcome mapping is reachable without CUDA while keeping the early return intact, and a dedicated test suite pins the classifier, the outcome mapping, and predicate identity with `batch_probe._is_cuda_oom`.

- **The `soup draft` panel now safely renders target and draft model names (#1052 by @MugdhoAI in #1053).** Model names containing Rich markup or terminal control bytes are escaped and stripped before being displayed.

- **Resuming a LoRA adapter targeting a streamed `lm_head`/`embed_tokens` silently dropped the head's own tensors (#1048 by @AmirF194 in #1055).** `StreamedLargeLayer._redirect_canonical_weight` only redirected the literal `"weight"` key into `inner.`, which is all a bare `inner` ever contributes. `get_peft_model()` runs before `install_streaming()` (#1012), so a LoRA target on one of these boundary modules makes `inner` a peft tuner layer whose own keys are `lora_A`/`lora_B` (or `lora_embedding_A`/`lora_embedding_B`) instead, none of which matched the old check. Generalized the hook to redirect every key under its own prefix, same as the decoder-layer version already does. The `save_pretrained()` meta-tensor crash and the parse-time refusal gaps from the same issue are separate, still open.

- **An untied streamed checkpoint takes a preference step again (#1049 by @dchaudhari7177 in #1061).** `embed_tokens` and an untied `lm_head` share one device slot, and the streamed layer passed autograd a view of it, so the reference forward of `dpo` or `kto` refilled those bytes in place and the policy backward raised `one of the variables needed for gradient computation has been modified by an inplace operation`. The projection now takes a private copy of its weight, one head-sized buffer per step, only for `dpo` and `kto` (the losses whose reference forward runs before the backward) and only when the slot is shared; `sft`, `orpo` and `simpo` keep the view (that scoping came from @VanshSharmaPES in #1174), and so does the embedding, because its backward reads the indices rather than the weight values. The VRAM pre-flight charges the copy as a second large-layer slot for exactly the runs that take it, so its prediction still never under-counts. Llama-3-8B-class checkpoints are untied, so this is the common shape; tied runs and `sft` were unaffected.

- **Refuse a stale-green PR merge without forcing a rebase on every push to main (#1017 by @jagadeepmamidi in #1071).** A `pull_request` run pins `refs/pull/N/merge` at creation, so GitHub's green marks describe a `main` that may no longer exist (`strict` is off). `merge-freshness` rebuilds the merge with `git merge-tree --write-tree` (no write to the fork), fails on ruff F821 or a conflict (the #763 `NameError` class), reports lag as a neutral check rather than a rebase demand, and re-posts that check on every push to `main`. It does not flip `required_status_checks.strict` and does not re-run the 13-job matrix; `gh run rerun` still cannot refresh a frozen merge SHA, which CONTRIBUTING now says.

- **`training.moe_lora` did nothing on DPO, KTO, ORPO, SimPO and GRPO, so 15 shipped MoE recipes trained something other than what they said (#798 by @Srinivasan8888 in #1074).** Only `sft.py`, `pretrain.py` and `stream_setup.py` ever read the flag; the 8 DPO and 7 GRPO recipes that set it got attention-only LoRA. Target selection now lives in one helper, `utils/moe.resolve_moe_lora_targets`, which every one of those trainers calls — as do `sft` and `pretrain`, which each kept their own copy of the block. `training.moe_expert_quant` and `training.train_router_only` are refused outside `sft`/`tts`, and a non-default `training.moe_aux_loss_coeff` outside `sft`/`tts`/`pretrain`, because no other task's trainer reads them.
- **All 15 of those recipes could not attach LoRA at all on transformers 5.x, not merely train the wrong modules (#798 by @Srinivasan8888 in #1074).** `resolve_lora_target_modules` returns `None` for `qwen3_moe`, and peft then refuses outright: `No target_modules passed but also no target_parameters found.` The MoE-aware targets close it for the architectures these recipes name; the general gap for other MoE architectures is #1070.
- **`moe_lora` with a non-zero `lora.dropout` now stops at the LoRA attach instead of failing inside peft (#798 by @Srinivasan8888 in #1074).** transformers 5.x keeps a Qwen3-MoE's experts as fused 3-D parameters, which peft adapts through `lora.ParamWrapper`, and that refuses dropout: `lora.ParamWrapper does not work with lora_dropout != 0.` At the schema default of `0.05` the attach failed on every task, including `sft`. The check runs against the loaded model rather than at config load, because whether the experts are fused depends on the checkpoint and the transformers version; a dense base is untouched. All 31 shipped MoE recipes now pin `lora.dropout: 0.0`, and a guard fails if a new one arrives without it. The check asks peft whether it would route these targets onto fused parameters, rather than scanning the model: Mixtral and MiniMax have the same fused 3-D expert parameters but peft attaches to them without complaint, and refusing those would be wrong. Measured on tiny random stand-ins on CPU, never on a real 30B+ checkpoint.
- **`moe_lora` does not reach every MoE family, and two shipped recipes are still attention-only (#798 by @Srinivasan8888 in #1074).** Expert adapters attach on `qwen3_moe` (11 recipes), `deepseek_v3` (6) and `glm4_moe` (3); on `minimax` they do not, so `minimax-m3-sft` and `minimax-m3-dpo` still train attention-only LoRA after this change, because peft has no v4->v5 conversion mapping for that model type and its experts are never targeted. The seven `kimi-k2.x` and two `mistral-large-3` recipes are untested rather than known-good: no tiny stand-in for those configs builds in the installed transformers. Per-architecture target resolution is #1070.

- `run_cost` no longer prices an RTX A4000 as an A40 or an `NVIDIA A100-SXM4-80GB` as a 40 GB A100. GPU-name patterns are now word-boundary anchored and ordered most-specific first, and the rate table gains L4, A10G, H200, B200, RTX A4000/A4500, RTX 6000 Ada, RTX PRO 6000 and the RTX 40/50-series (#831 by @rishcx in #1081). **Cost figures and the `soup loop` budget gate change for these cards:** L4, H200, B200, RTX 40/50-series and RTX PRO 6000 used to be unpriced and counted as $0, so their budget gate never fired; they now have a rate and it does. Names the old unanchored patterns mispriced (`A100X`, `A1000` and `RTX A1000 Laptop GPU` as an A100, `T400` as a T4) now resolve to unknown instead. Laptop parts (for example `RTX 4090 Laptop GPU`) still price as the desktop card, which overstates a laptop run. Existing rows keep their labels and rates.

- **`soup data preprocess` ignored `data.chat_template` and left it out of the cache key (#1067 by @Blizzeq in #1082).**
  Rows were rendered with the tokenizer's shipped template whatever the config
  said, and two configs differing only in the template shared one cache. Preprocess
  now renders with the resolved template and hashes it into the key. The
  `pre_tokenized` training config must name the same template, and a cache written
  before this change is refused with `cache hash mismatch`; re-run
  `soup data preprocess` to rebuild it.

- **GRPO training now skips non-finite optimizer steps instead of corrupting adapter weights.**
  A pre-step gradient watchdog detects NaN or Inf gradients and zeros them before
  `optimizer.step()`, so the update is skipped and the adapter is preserved. The
  skipped-step count and fraction are persisted in `trainer_state.json` and reported
  by `soup adapters audit`. (#342 by @harshitthek in #1084)

- **Reject and report `loss_watchdog`, `loss_spike_recovery`, and `grad_accum_auto_tune` on `backend: mlx` (#1069).** All three callback-monitoring settings are now rejected at schema validation time naming `backend='mlx'` with their exact technical reasons (watchdog stop control, spike recovery watchdog requirement, and unified memory VRAM pressure), declared as rejected in `backend_support.py`'s `_MLX_SFT` registry table, and warned about at runtime in `trainer/mlx_sft.py::_check_unsupported` (#1069 by @kok-o in #1088).

- **The `commands.md` coverage guard now matches against the command column rather than whole entry rows (#1046 by @Areen-09 in #1094).** Previously, mentioning a command inside another row's description prose allowed commands whose own rows were deleted (such as `soup serve`, `soup export`, `soup train`, or `soup bom emit`) to falsely count as documented. The guard now isolates the command column before description whitespace and inline comments, anchors leaf matching, updates token lookahead to include dots (`gate.v2`), and explicitly documents `soup loop resume`, `soup lock check`, and `soup steer list` in their own rows.

- **The `soup data preprocess` cache kept a leading BOS the live path never trains on (#876 by @Srinivasan8888 in #1096).**
  The cache tokenised the rendered chat template with `add_special_tokens=True` and
  dropped a leading BOS only when it was doubled, so a `data.chat_template` preset on
  a tokenizer whose post-processor prepends one cached a BOS the live path
  (`add_special_tokens=False`) does not have. It now drops exactly the BOS the
  post-processor contributed, keeping the template's own.
- **A truncated row no longer gains a stop token the live path lacks (#876 by @Srinivasan8888 in #1096).**
  #791's "append the training EOS when the row is under `max_length`" check ran after
  a BOS was removed, so a row truncated to the budget read as one token short of it.
  Truncation is decided before the strip now.
- The preprocess cache schema is `v5` (#876 by @Srinivasan8888 in #1096). Re-running
  `soup data preprocess` writes a new cache directory and orphans the old one; pointing
  training at the old directory is refused with `cache hash mismatch`.

- **`target_modules: auto` now resolves on every MoE architecture Soup ships a recipe for (#1070 by @Srinivasan8888 in #1102).**
  peft has no LoRA mapping for any of them, so `auto` reached it as `None` and the
  attach failed with `No target_modules passed but also no target_parameters found`.
  Each now resolves to the attention projections that architecture defines —
  `qwen3_moe`, `deepseek_v3`, `deepseek_v4`, `glm_moe_dsa`, `kimi_k25`, `gpt_oss`,
  `minimax_m2` and `minimax_m3_vl`, plus `kimi_k2` (the text tower of K2.5/K2.6 and
  the outer type of Kimi-K2-Thinking) and, defensively, `minimax_m3_vl_text`
  (MiniMax-M3's text tower when its config loads without the wrapper; no shipped
  base reports it) — including the three MoE recipes that set no MoE flag
  (`gpt-oss-20b-sft`, `gpt-oss-120b-sft`, `minimax-m2-sft`). MiniMax-M3 resolves to
  a regex scoped to its language tower, so a text fine-tune no longer adapts the
  vision encoder. DeepSeek-OCR is deliberately not covered: its architecture needs
  `trust_remote_code` to build, so its module names cannot be measured.
- **`target_modules: auto` on an architecture neither Soup nor peft maps is refused by name (#1070 by @Srinivasan8888 in #1102).**
  peft refused these before too, as `No target_modules passed`, naming neither the
  architecture nor the fix; the message now names the `model_type` and the fix.
  It is decided at the last step before the attach, so `training.moe_lora: true` and
  `training.lora.target_parameters` are both honoured first. Scope, stated so it is
  not a surprise: it reaches dense architectures too (`phi3`, `smollm3`, `lfm2` and
  several vision and audio architectures in the catalogue), every one of which
  already failed at attach with peft's message.

- **`soup loop canary` now routes keyed transformers requests at the configured traffic share and automatically clears a regressing canary (#815 by @akkupratap323 in #1104).** Serve records rollout-scoped generation outcomes for stable and canary buckets; `soup loop watch` evaluates those observations and persists a rollback after a `MAJOR` verdict, while explicit adapter selection keeps its existing behavior and the 100% activation endpoint still applies to requests without a conversation key.

- **`soup adapters audit` now verifies the effective MLX learning rate (#942 by @Amix29 in #1109).**
  MLX runs compare `training.lr` with the optimizer plan's recorded `peak_lr`;
  transformers/PEFT reports that the effective learning rate was not checked rather
  than presenting missing optimizer metadata as a clean audit.

- **The training docs no longer advertise eight nonexistent `soup train` flags (#997 by @Amix29 in #1110).**
  ULD, RL checkpointing, echo-trap settings, and `output` now use their real
  YAML fields, and the live Typer-tree guard has no remaining grandfathered
  exceptions for these spellings.

- **`mini_tool_call` no longer sits at a 1.000 ceiling on the measured strong reference (#405 by @Amix29 in #1111).**
  The 40-row suite now mixes tool selection with exact `NO_TOOL` decisions and
  semantically close distractors; the measured reference moves to 0.850 versus
  0.050 on the weak model, and the bundled scorer provenance revision changes
  so pre-change baselines are not silently compared on the new scale.

- Spark-TTS and OuteTTS raw-audio live-codec configs now fail closed at config parse with the real upstream compatibility reason; the unrunnable Spark recipe is removed, while Oute keeps the pre-encoded `chatml` recipe path (#265 by @Amix29 in #1113).

- **HuggingFace repo IDs are now preserved without Windows path separator corruption in `soup infer` and `soup bench` (#1097 by @Guts1005 in #1118).** When resolving a HuggingFace repo identifier on Windows, the repo string was previously converted to a `pathlib.Path`, turning forward slashes into backslashes and causing HuggingFace Hub to reject the repository name with `HFValidationError` before loading. The repo ID is now maintained as a string and forwarded byte-identical to model and tokenizer loaders across all platforms.

- **`soup data preprocess` cached rows with no `labels`, so a `pre_tokenized` run trained on the prompt (#1054).** The cache carried only `{input_ids, attention_mask}`; with no `labels` column TRL's collator falls back to `labels = input_ids`, so the same rows optimised a different objective depending only on whether they had been preprocessed first — full-sequence loss instead of the live path's assistant-only mask, with a normal-looking loss curve and no error. Cached rows now carry a `labels` column built by the same `data/loss_mask.py` helpers the live path uses, honouring `data.train_on_responses_only`, `data.train_on_messages_with_train_field`, `data.mask_history` and `training.train_on_eot`. That loss-mask mode is now a cache-key input (tokenizer schema `v6`), so a cache built under one masking setting is refused by name under another, and a cache with no `labels` is refused outright rather than silently trained on. `soup train` also reports the real cached row count for a `pre_tokenized` config instead of `0` (#1054 by @AaronProbha18 in #1120).

- **Fix invalid Hugging Face base model IDs in shipped recipes (#1122).** Updated `devstral-sft` to `mistralai/Devstral-Small-2507`, `glm-4.6-sft` to `zai-org/GLM-4.6`, `granite-4-sft` to `ibm-granite/granite-4.0-h-tiny-base`, `internvl-3-5-sft` to `OpenGVLab/InternVL3_5-8B`, `kimi-k2-sft` to `moonshotai/Kimi-K2-Base`, `mistral-medium-3-5-sft` to `mistralai/Mistral-Medium-3.5-128B`, and `voxtral-sft` to `mistralai/Voxtral-Mini-3B-2507`. Removed `cogito-v2-sft` and `gemma3-9b-sft` because no compatible replacement base exists for their shipped recipe definitions. (#1122 by @parnika0110 in #1132)

- Reject structurally unsafe regex patterns in custom eval, diagnose format probes, and Data Recipe validators before matching, replacing the diagnostic canary search with a field-named error (#1137 by @taljeon in #1142).

- **`training.moe_lora` now works on `ipo`, `bco`, `reward_model`, `ppo`, `embedding` and `online_dpo` (#1099 by @Srinivasan8888 in #1148).**
  These tasks accepted `moe_lora: true` and never read it, so the flag silently did
  nothing. Each now picks the expert targets through the same helper as the other
  MoE-wired trainers, including its `lora.dropout` refusal on fused experts. On
  `embedding` it applies only with `lora.r >= 1`. The flag still loads unread on
  `classifier`, `reranker`, `cross_encoder`, `distill`, `unlearn`, `asr`,
  `moe_lora_routing` and `prm`; #1151 tracks wiring the first five and refusing the
  rest.

- Preserve absent gradient norms as `NULL` in training records and charts instead of fabricating `0.0` or carrying the previous measurement into a new step (#1147 by @taljeon in #1149).

- `ExecutionManager.execute()` now opens the MCP run log with `open_no_follow` (#820), refusing a file symlink planted at `.soup/mcp-runs/<run_id>.log` before the child spawns; previously the plain append followed the link and redirected every byte the child wrote to its target (#1138 by @xiechimon in #1150).

- **`target_modules: auto` now resolves on GLM-4.6 and Granite 4.0 (#1122 by @MakazhanAlpamys in #1164).**
  Both recipes named a Hugging Face id that 404s, so the #1102 completeness ratchet
  recorded their bases as unresolvable and never asked whether their architectures
  were covered. Repointing them (#1132) makes `glm4_moe` and `granitemoehybrid` real
  entries in the catalogue, and neither was in `MOE_TEXT_LORA_TARGETS`. Both now
  resolve to the four attention projections enumerated from the architecture on the
  meta device, the same policy as every other row of that table.
- **Granite 4.0 says at setup that `auto` adapts only part of the decoder (#1122 by @MakazhanAlpamys in #1164).**
  `granitemoehybrid` is a hybrid: on `granite-4.0-h-tiny-base` only 4 of the 40
  decoder layers define a `self_attn` at all — the other 36 are Mamba-2 blocks — so
  the attention-projection entry reaches a tenth of the model. Training now prints a
  yellow `Partial LoRA coverage:` line naming the model type, the counted fraction
  and what was skipped (the Mamba projections, the shared-expert linears and the
  fused routed experts), instead of leaving a surprisingly small adapter to be
  discovered at merge time. The fraction is counted off the loaded model, so a
  different Granite depth reports its own numbers.

- **Recipe size labels are the base's total parameter count, and a test now keeps them so.** Nine labels were wrong and moved: `llama4-scout-17b-sft`, `-dpo`, `-grpo` and `llama4-scout-tools` 17B -> 109B, `deepseek-v3-7b-sft` 7B -> 685B, `voxtral-sft` 3B -> 4.7B, `glm-4.6-sft` 9B -> 357B, `glm-5-sft` 9B -> 754B, `minimax-m2-sft` 9B -> 230B. `tests/test_recipe_sizes.py` compares every recipe's label with the Hub's recorded parameter count (0.70x to 1.15x) and names a recipe whose base has no count unless it is excluded with a reason (#1161 by @kok-o in #1168).

- **Added modality axis to `backend_support` and made `soup doctor --config` report unread `train_on_responses_only` and `mask_history` on vision and audio configs (#1156 by @kok-o in #1169).**
  Multimodal vision and audio collators supervise all sequence tokens rather than performing assistant-only masking. `backend_support.REGISTRY` previously only tracked `(task, backend)` pairs with no modality axis, causing `soup doctor --config` to print an affirmative all-clear on multimodal configs that set these flags. `REGISTRY` now accepts `(task, backend, modality)` and flags both settings as ignored when running vision or audio SFT.

- **Refuse `.soup` and `.soup/mcp-runs` directory symlinks and pre-planted hard links during MCP execution (#1158 by @kok-o in #1171).**
  Following #1138, `soup mcp serve` now refuses execution if `.soup` or `.soup/mcp-runs` is a symbolic link or junction, or if a pre-planted hard link exists at the run log path, closing directory redirection and hard link overwrite attacks (with a residual TOCTOU window if an attacker has concurrent write access inside `.soup`).

- **An unregistered `data.chat_template` name is refused before any model loads (#1175 by @Blizzeq).**
  `soup train` checks the name right after loading the config, so a typo exits 1
  at once instead of after the model has been downloaded and loaded. `soup train
  --dry-run` and `soup train --cloud` now refuse it as well, where they used to
  exit 0. Both `soup train` and `soup data preprocess` print `Invalid
  data.chat_template` with the known template names.

- QuEST artifacts now store a content-derived identity for local base models instead of the machine's absolute path in route metadata, while keeping format-v1 sidecars readable and resumable under their original path-based rule. (#1176 by @Shutaru)

- **`training.moe_lora` now works on `classifier`, `reranker`, `cross_encoder`, `distill` and `unlearn`, and is refused where no trainer reads it (#1151 by @Srinivasan8888 in #1179).**
  These tasks accepted `moe_lora: true` and never read it. `distill` and `unlearn`,
  and the classifier family with `classifier_lora: true`, now pick the expert targets
  through the same helper as every other MoE-wired trainer. Refused at config load,
  naming the task: the classifier family without `classifier_lora: true` and
  `lora.r > 0` (that trainer then full-fine-tunes and builds no adapter), `asr` (it
  trains only Whisper, which has no experts), `moe_lora_routing` (it builds no LoRA
  adapter) and `prm` (it fine-tunes every base parameter). It is also refused on
  `backend: unsloth` for every task, and on `task: sft` with `modality: vision` or
  `audio`, whose setup skips the MoE step. Every path now either reads the flag or refuses it, except `backend: mlx`,
  where it loads and `soup doctor` reports it as ignored. Because
  `training.train_router_only` and `training.moe_expert_quant` require
  `moe_lora: true`, a config setting either on `backend: unsloth` now fails to load
  too, with the unsloth message; the unsloth path reads neither.

- **`soup data preprocess` stops on a chat row it cannot tokenize, instead of dropping it (#1180 by @Srinivasan8888 in #1182).**
  **This is a behaviour change: a dataset that `soup data preprocess` used to cache
  partially, with exit 0, now makes it exit 1.** A row the chat template rejected (for
  example `Conversation roles must alternate` on a Llama-2- or Gemma-style template), an
  empty `messages` list, a row rendered as empty text, or a tokenizer error (on a pretrain
  text row too, which live pretraining refuses the same way) was skipped without a word,
  and a `pre_tokenized` run then trained on fewer rows than the live run of the same
  `soup.yaml`, which stops on those rows. The command now stops before writing anything
  and names the row, numbered from 1 as live training numbers it, with the start of its
  first message. Fix or remove the row and re-run. A tokenizer with no chat template is
  reported once and points at `data.chat_template`. The rows are checked when the command
  builds a cache: if one with the same key already exists it stops at
  `Target already exists` (exit 0) without reading them, so rebuild a cache written before
  this change with `--yes`.

- **Training reports the dataset rows its format converter drops (#1181 by @Srinivasan8888 in #1183).**
  A row the converter rejects, such as a misspelled `messages` key, is still dropped so one
  bad line does not abort a load, but it is no longer silent. Every load now prints
  `Warning: N of M rows dropped` with the first row's index and the converter's reason,
  and for a local file the `soup data validate` command that lists them all. The count
  agrees with `soup data validate`. A clean file prints nothing.

- **`task: distill` with `backend: unsloth` is refused at config load instead of silently training on transformers (#1186 by @chiliec in #1193).** The distill trainer never reads the backend, so the unsloth request was dropped without a word. `validate_distill_compat` now rejects it the same way it rejects `backend: mlx`, with a message naming `backend: transformers`, and `soup doctor --config` reports the config as unloadable (exit 2) instead of clean.

- The friendly error handler no longer parses exception text as Rich markup, so install hints keep their extra (`soup-cli[audio]` was printed as `soup-cli`), and control bytes in an exception message are stripped before they reach the terminal (#1200 by @chiliec in #1202).

- `distill_mode: sequence` runs on transformers 5: teacher generation no longer fails with a bare `AttributeError` on the first prompt, because the chat-template encoding is read as a mapping and its attention mask is passed to `generate` (#1238 by @MugdhoAI in #1243).

- **`soup train --find-lr` no longer fails when the training set is smaller than `--find-lr-steps`, or blames `--find-lr-output` for it (#1189 by @mohdUwaish59 in #1244).** The live sweep takes one row per step; it stopped when the rows ran out (or when the loss went non-finite), but the report was written against the full schedule, so the run ended with `Invalid --find-lr-output: lrs and losses must have equal length`. The sweep is now sized to the rows over the same `--find-lr-start` → `--find-lr-end` range, with a line naming both numbers, and a divergence mid-sweep writes a report for the steps before it. The recommendation needs 4 points, so `--find-lr-steps` below 4 is rejected up front (the minimum was 2, which always failed), and a training set with fewer than 4 rows is refused before the model loads instead of falling back to a synthetic curve. An empty training set is refused the same way; it previously wrote a synthetic report and exited 0. `Invalid --find-lr-output` is now reported only for a problem with the output path.

- **Web UI returns 401 rather than 500 when Authorization header contains non-ASCII bytes (#1195 by @Areen-09 in #1246).** Starlette decodes header bytes >= 0x80 as Latin-1 strings; `secrets.compare_digest` rejected non-ASCII strings with `TypeError`, causing an unhandled 500 that printed a full traceback on the server console for every such request and dropped security headers. Token comparisons now encode to UTF-8 bytes before comparing, aligning with `soup serve` and `soup mcp serve`.

- **`task: prm` on CUDA trains every base parameter and the reward head on fp32 master weights (#1235 in #1250).** PRM loaded the base and the head in bf16 whenever the device was `"cuda"`, with no fp32 copy for the optimizer. Since v0.74.0 that made the first step fail with `mat1 and mat2 must have the same dtype` (outside DeepSpeed), because the bf16 head ran on the fp32 outputs of the autocast forward. Before that the base trained in pure bf16, where most AdamW steps at PRM learning rates round away: an RMSNorm weight at 1.0 never moves at `lr` 1e-5 or 2e-5. PRM now loads fp32 on every device string, `"cuda"` and `"cuda:0"` alike, and leaves bf16 to autocast (fp16 plus the GradScaler on pre-Ampere cards, which now receives fp32 gradients). With the default optimizer, expect about 16 bytes per parameter before activations instead of about 8, which is what the VRAM pre-flight already budgeted for `task: prm` (the pre-flight runs only when `batch_size` is an integer, not with the default `auto`). Without DeepSpeed the saved checkpoint is now fp32, and loading needs about twice the host RAM of a bf16 load. Under DeepSpeed only host RAM at load grows, since the engine still trains and saves in 16-bit on its own fp32 master copy. The `training.quantization` warning and refusal for the eight tasks that ignore that field now say the trainer never quantises the base, not that it loads "at checkpoint precision".

- Reject `training.quantization_aware: quest` with `training.auto_mixed_precision: true` at config load, preventing auto-selection of unmeasured FP16 for QuEST (#1245 by @Shutaru in #1252).

- **The `soup data preprocess` cache key now covers every setting that changes the cached rows (#1127 by @Blizzeq in #1253).**
  `data.val_split`, `data.replay`, `data.replay_ratio`, `data.replay_seed`, the
  streaming settings (`data.streaming`, `data.buffer_size`), the media
  directories (`data.image_dir`, `data.audio_dir`) and the top-level `task` all decide
  which rows are cached, but none was part of the key, so a `pre_tokenized` run under a
  different value reused the cache without a mismatch. They are now key inputs
  (tokenizer schema `v7`), and every other `data` field is declared out of the key with
  a reason, so a new field cannot be left unclassified. Caches from `v6` and earlier are
  refused; re-run `soup data preprocess`.

- **`soup migrate --from axolotl` no longer turns `rl: orpo`, `rl: ipo`, and
  `rl: simpo` into `task: sft` with no warning (#1254).** The rl-to-task
  mapping only covered `dpo`, `kto`, `grpo`, and `gdpo`; every other `rl`
  value fell through to `sft` silently, so a preference config loaded with
  the wrong task and the wrong data format and later failed training with
  an error that never named `rl`. The mapping now covers `orpo`, `ipo`, and
  `simpo`, each resolving to its matching Soup task and the `dpo` data
  format. An `rl` value with no Soup task, such as axolotl's `ebft`, is
  refused instead of silently falling back to `task: sft`, naming the
  unknown value and listing the supported ones. `soup migrate` exits
  non-zero for it instead of writing a config (#1214 by @jamalkamaladdin in
  #1254).

- **`training.loraplus_lr_ratio` with `optimizer: apollo_adamw`, `lomo` or `adalomo`, or with `use_galore`, is now refused at config load (#1210 by @mohdUwaish59 in #1255).** LoRA+ asks transformers for its optimizer without the model, and transformers can only build those three optimizers from one, so such a config passed `soup train`'s config load and failed only after the base model had been downloaded and loaded. LoRA+ with GaLore was likewise refused only at setup. Both now fail at load with a message that names the two fields.

- **`soup shrink` saves pruned Qwen2, Qwen3, Qwen2-MoE and SmolLM3 models (#1241 by @drakeo338 in #1256).**
  The prune dropped the decoder layers but left `layer_types` (and SmolLM3's
  `no_rope_layers`) at the old length, so transformers 5 refused to save the
  model after scoring and the run exited 1 with an empty `model/` directory.
  Those lists are now sliced with the same kept indices as the layers, so a
  mixed sliding/full-attention list keeps the right entry for each surviving
  layer.
  A config that places its MoE layers by layer index (a non-default
  `mlp_only_layers`, `decoder_sparse_step`, `moe_layers` or
  `interleave_moe_layer_step`) is refused before the prune, since the kept
  layers would reload with their MoE and dense MLPs on the wrong layers.

- **`soup autopilot` no longer writes a config that `soup train` refuses on an Ampere or newer GPU (#1212 by @wanjinhao1 in #1257).** For the SFT goals, autopilot turned on `use_flash_attn` and `use_liger` on every GPU with compute capability 8.0 or higher without checking that `flash-attn` or `liger-kernel` was installed. Neither ships with a standard install, so `soup train`, even with `--dry-run`, refused the file autopilot had just written. Each flag is now turned on only when its package is installed, checked the same way `soup train` checks it, and when the GPU qualifies but a package is missing, the Autopilot Decisions panel names it and the command that installs it. The Liger refusal in `soup train` also told users to run `pip install "soup-cli"`, because `[liger]` was read as Rich markup; it now prints `pip install "soup-cli[liger]"`.

- **`soup data doctor` gains `--mask-history/--no-mask-history` (default off), so the compat report and the `--show-mask` preview train only the last assistant turn, as a `data.mask_history: true` run does (#1251 by @wanjinhao1 in #1259).** `--mask-history` is refused together with `--no-train-on-responses-only` or `--train-on-messages-with-train-field`, the same combinations `soup.yaml` refuses.

- `--baseline registry://` now takes each benchmark's score and provenance from its newest eval row, so re-measuring a baseline replaces the stale score instead of only silencing the scorer-revision warning; the warning names the benchmarks that are still stale, and ties on `created_at` resolve by insertion order (#1224 by @anyingiit in #1260).

- **`soup data forge --judge-provider` wrote an empty-answer row for every failed judge call and still reported "synth complete" (#1221 by @anyingiit in #1261).** The forge judge was built without `raise_on_error`, so a connection error, HTTP 404/429/500 or malformed response came back as empty text; `score_uncertainty` rates an empty reply 1.0, so no `--uncertainty-threshold` could prune it, and the command exited 0 with a dataset that teaches a model to answer with nothing. The forge judge is now built with `raise_on_error=True` (as `soup data recipe` has been since #969); a call that fails, returns a malformed reply or returns empty text is counted and never becomes a row, whatever the threshold. The summary reports `N of M judge calls failed` with the provider, endpoint and first error; a run whose judge produced no usable row exits 1 without writing the dataset or provenance files, and a partial outage keeps the successful rows under a yellow "synth complete with judge failures" panel.

- **`401`/`403` digits inside an error message no longer masquerade as an auth failure (#1267 by @sup3dev in #1268).**
  A row number (`train row 401:`), tensor size (`(4013)`), checkpoint path
  (`checkpoint-4010`), date, or hash slice printed "Authentication failed." /
  "Access denied." and hid the real error. Auth status is now read from the
  exception object (`response.status_code`, `.code`, walking wrapped
  `__cause__`/`__context__`), with a text fallback only for explicit HTTP
  phrasing (`401 Unauthorized`, `HTTP Error 401`, `Error code: 401`); the
  bare-digit ERROR_MAP entries are gone. Genuine 401/403 from requests, urllib
  and huggingface_hub now get the specific hint instead of the generic
  "HTTP request failed.", and gated-repo hints still win for
  `GatedRepoError` / `RepositoryNotFoundError`.

- Show QuEST task, backend and modality errors before the auto mixed-precision refusal when a config violates both, and link the BF16 evidence boundary from the auto mixed-precision guide (#1276 by @Shutaru).

- **A refused `pre_tokenized` cache now says why (#1272 by @Blizzeq in #1281).**
  `soup data preprocess` records the key generation and the row-set inputs in
  `metadata.json`. When training refuses a cache, the message says it predates
  row-set keying (#1127), or names the `data.*` fields whose values differ, such as
  `data.val_split`, instead of showing two bare hashes. `docs/data.md` lists the
  fields a `pre_tokenized` config must keep as they were at preprocess time.

- **Online DPO no longer trains a pair its judge could not rank as "the second completion
  wins" (#1225 in #1282).**
  Soup's pairwise judge answers `-1` for a tie, a failed or unreadable judge call, or a
  verdict that changes when the two completions are swapped. trl 0.29's `OnlineDPOTrainer`
  builds `mask = rank == 0`, so every such pair trained as a win for the second completion,
  and the loss was averaged over all pairs. With the judge server down, a whole run learned
  arbitrary labels and finished normally. On the `training.online_dpo_judge` path those pairs
  now add exactly zero gradient, and the loss is the mean over the pairs the judge did rank
  in each batch (each micro-batch, under gradient accumulation). `judge/invalid_rate` logs
  the share it could not rank, with a WARNING the first time and on every step in which it
  ranked nothing. The logged `loss` and `train_loss` average only the batches that ranked a
  pair, and a logging window with none logs no `loss`, so an unranked stretch can no longer
  produce the run's lowest or final loss. For every trainer, the run tracker now stores no
  loss, rather than a fabricated 0.0, for a log that arrives before the run's first measured
  loss (later ones still repeat the last measured loss). That includes the row of the MLX SFT
  trainer's first validation pass, which runs before any training step, and the live panel
  shows `Loss: -` instead of `0.0000` until a loss is measured. `soup runs clean` therefore
  picks the checkpoint to keep whole among the checkpoints: the one with the lowest loss
  recorded at its step, or the latest one, with a note, when none has a recorded loss.
  Before, it could strip or delete every checkpoint of a run shorter than `logging_steps`, or
  of any run whose lowest loss was logged between two checkpoints. A step that ranked nothing
  still takes an optimizer step, so AdamW momentum and weight decay keep moving the weights
  and the learning-rate schedule advances. That drift is bounded: training stops with an
  error naming the judge once 32 pairs in a row went unranked, and a shorter run that ranked
  no pair fails the same way instead of saving an adapter. trl's `logps/*` and `rewards/*`
  statistics now describe the ranked pairs only. With clean 0/1 verdicts, training is
  bit-identical to before, and the `reward_model` path is unchanged.

- **Hardware-fit gate correctly infers Hub model sizes using boundary-safe regex and MoE parsing (#1198 by @Areen-09 in #1285).** `utils/gpu.py:model_size_from_name` previously relied on a static scan of hardcoded substring markers and fell back to 7.0B, causing VRAM under-prediction up to ~58x–96x on large models (such as 72B, 32B, 14B, 405B, 675B) and substring collision bugs (e.g., `11b` matching `1b`, `17b`/`27b` matching `7b`, `0.8b` matching `8b`, and `8x7b` matching `7b`). Model sizes are now dynamically parsed with boundary-safe regex, active-parameter tokens (`-aNNb`) are filtered to retain total parameters, and MoE architectures (`8x7B`, `8x22B`, generic `NxMb`, and Llama 4 `17B-16E`) are supported. The same parameter estimate also drives `batch_size: auto`, `soup cost`, and `soup profile`, while Hub IDs carrying no size marker (`DeepSeek-V3`, `Kimi-K2`, `GLM-5`, `MiniMax-M2`) still fall back to 7.0B, an under-estimate for those models.

- **`soup registry diff`, `soup eval compare` and `soup eval leaderboard` now use the newest eval row per benchmark instead of the oldest (#1270 by @Ru0k3 in #1291).**
  Each built its score map by overwriting in the order the experiment tracker returns rows, which is newest first, so the oldest score won and a re-measurement was ignored: with run A logging 0.30 and then 0.45, and run B logging 0.50 and then 0.40, `soup registry diff` and `soup eval compare` reported +0.20 and no regression instead of the -0.05 regression. All three commands and the eval gate's `registry://` baseline now pick rows through one helper, `soup_cli.eval.results.newest_eval_rows` (newest `created_at` first, then insertion order for equal timestamps); the leaderboard keeps the newest row for each model and benchmark.

- **Parquet datasets preserve nested Python values (#1218 by @commit-msuyog in #1292).**
  Parquet loading now uses PyArrow to preserve nested lists and dictionaries instead of
  converting them to NumPy arrays, preventing valid rows from being dropped by format
  validation.

- GRPO variant runs (`grpo_variant` other than `standard`) with a non-zero
  `grpo_beta` log the `kl` metric again, as trl's stock loss does: the masked
  batch mean of the per-token KL, averaged over the accepted tokens for `rft`
  (#1263 by @MohammadHijjawi97 in #1293).

- **Fix DPO / IPO / preference beta schedule callback crashes and refuse un-wired reference regeneration (#1229 by @kok-o in #1294).** DPO, IPO, and `preference` runs configured with `dpo_beta_schedule` crashed at step 0 with `AttributeError: 'BetaScheduleCallback' object has no attribute 'on_epoch_begin'` because custom callbacks were plain Python classes rather than Hugging Face `TrainerCallback` subclasses. Converted callbacks to lazily subclass `TrainerCallback` without introducing import-time dependencies on heavy ML packages, verified beta annealing behavior under IPO and preference training, and enforced a load-time refusal for `dpo_ref_regen_epochs` as the feature is not wired yet.

- `task: embedding` with `embedding_loss: contrastive` (the default) no longer trains on one-row batches, where the loss and gradient are exactly 0 and the adapter never moves: `batch_size: 1` is refused at config load, `batch_size: auto` resolves to at least 2 (also when `triplet` falls back to contrastive for rows without `negative`), the last partial batch of an epoch is dropped, and a batch larger than the training set is lowered to its size. Auto batch sizing and the hardware-fit estimate also stop guessing 7B for BGE / E5 / GTE / BERT / RoBERTa / MiniLM checkpoints named `-small` / `-base` / `-large` (#1234 by @kok-o in #1296).

- Refuse rows with no causal-loss target at setup in `task: distill` with human-facing row number and `data.max_length`, matching SFT behavior, and return finite 0.0 CE loss in `compute_loss` if a microbatch contains zero targets (#1242 by @kok-o in #1298).

- **`soup expect` scans all dataset formats for PII, token length, and refusals (#1231 by @kok-o in #1300).** The dataset expectation gate previously only inspected a few field names (`text`, `instruction`, `output`, string `messages`), missing ShareGPT, DPO (string and conversational `chosen`/`rejected`), KTO (`completion`), Alpaca `input`/`system`, multimodal content parts, tool calling, and sentence embedding fields. The text extractors now cover all format signatures declared in `FORMAT_SIGNATURES`, verify assistant-side refusal patterns across formats, and fail closed when rows lack extractable text.

- **Tokenize non-Latin scripts, non-spaced languages, and digits in `soup diagnose` and `live_eval` (#1233 by @kok-o in #1301).** The previous regex discarded non-ASCII characters and standalone digits, causing healthy models responding in non-Latin scripts (Cyrillic, Greek, Arabic, Hebrew, Hindi, CJK, etc.) to falsely read as MAJOR mode collapse, and causing forgetting probes using `token_f1` to evaluate to zero retention. Tokenization now performs stream-safe NFC Unicode normalization (capping combining mark runs at 30 to prevent quadratic reordering), preserves non-Latin word boundaries, generates bigrams for non-spaced scripts (CJK, Kana, Thai), and ignores mark-only tokens. `token_f1` splits joiners (`-` and `_`) to preserve baseline English scores. Downstream consumers (`soup data topics`, canary discovery, and eval design) now tokenize digits and non-Latin words. When `--tokenizer` is passed, the memorization probe computes sub-word bigram overlap via tuples, avoiding key collisions.

- **`soup migrate` explicitly writes `quantization: none` and emits `lora.r: 0` for full fine-tuning (#1215 by @kok-o in #1302).** Previously, unquantized Axolotl, LLaMA-Factory, and Unsloth configurations omitted the quantization field, causing Soup to default them to 4-bit QLoRA upon loading, while full fine-tunes defaulted to LoRA `r=64`. Migrators now explicitly set `quantization: none` for unquantized sources, support HQQ/EETQ methods, map full fine-tunes to `lora.r: 0`, refuse full fine-tuning for unsupported tasks (e.g. DPO/PT/RM), resolve notebook variable assignments in Unsloth, and validate output with `load_config_from_string` before writing.

- **`soup train --resume` properly resumes `task: prm` and `task: moe_lora_routing` rather than retraining from step 0 and overwriting checkpoints (#1220).** Both `PRMTrainerWrapper` and `MoleRoutingTrainerWrapper` now accept and forward `resume_from_checkpoint` to `self.trainer.train()`. In MoLE training, the trained router gate weights (`mole_gate.pt`) are now saved into each periodic `checkpoint-N` directory via `_save_checkpoint`, and restored on resume via `_load_from_checkpoint` alongside adapter re-freezing prior to continuing optimization, ensuring step counts, optimizer momentum, and gate weights continue consistently. Tasks that do not support resume (such as `task: unlearn`) cleanly exit non-zero on `--resume` / `--hf-resume` with a descriptive message naming the task (#1220 by @kok-o in #1303).

- **`training.packing: true` preserves assistant-only loss mask during SFT instead of training prompt/system tokens (#1236).** In TRL 0.29.1, dataset packing drops the unpreserved `labels` column and falls back to training all input tokens except the first. SFT datasets now ensure an `assistant_masks` column alongside `labels`, which TRL packs alongside `input_ids` and uses to mask prompt/system tokens with `-100` during collation (and prevents label dropping under `use_liger: true`). Pre-tokenized caches carrying `labels` are also augmented with `assistant_masks` automatically (#1236 by @kok-o in #1304).

- **Fix classifier, reranker, and cross_encoder data loading and row validation (#1219 by @kok-o in #1306).** Sequence classification tasks (`classifier`, `reranker`, `cross_encoder`) were losing dataset columns such as `label`, `text_a`, `text_b`, `question`, and `answer` during dataset normalization because `preserve_source_columns` was restricted to `task == "grpo"`, causing crashes with `TypeError: label must be int / str / list, got NoneType` after model loading. Centralized source column retention in `task_preserves_source_columns` shared across `soup train` and `soup sweep`, added task-scoped `cross_encoder` format handling for `text_a`/`text_b` and `question`/`answer` pairs, gated `data.format: cross_encoder` to `task: cross_encoder` at schema validation, and added upfront dataset label and shape validation that fails immediately at data load and on `--dry-run` with specific row indices before model weights are loaded.

- **`soup ui` opens and prints a login URL that already carries the auth token (#1194 by @Faisal-Fayaz in #1307).** The first browser tab was `http://host:port` with no `?token=`, so every `/api/*` route except `/api/health` returned 401 and the dashboard rendered `Unauthorized`. The startup panel and `webbrowser.open` now use `http://host:port/?token=<token>`, including under `--no-browser`. `app.js` already reads that query parameter, and on a loopback bind the server exchanges it for an HttpOnly session cookie (#1191).

- **`training.expand_layers` is refused at config load unless `training.quantization` is `none` (#1237 by @chiliec in #1308).** Under the default `4bit`, the "zeroed" residual projections of the appended LLaMA Pro blocks decoded to `-absmax` instead of 0 (NF4 code 0 is `-1.0`), so the new blocks were not identity, and a positive `freeze_trainable_layers` crashed at setup on `4bit` and `8bit` because torch cannot set `requires_grad` on integer weights. The error now names both fields and asks for `quantization: none`.

- **`rope_scaling_type: llama3` on a checkpoint that already ships Llama 3.1 RoPE scaling now composes with that scaling instead of replacing it (#1239 in #1311).** Llama-3.1 checkpoints ship `max_position_embeddings: 131072` with a native `llama3` block (factor 8 over an original 8192 tokens). Extending one to 262144 tokens rebuilt the block as factor 2 over 131072, so 35 of the 64 frequency pairs of an 8B head rotated up to 8x faster than in pretraining, and the run printed only "Long-context enabled". `llama3` on a native `llama3` block now keeps the checkpoint's `original_max_position_embeddings`, `low_freq_factor` and `high_freq_factor`, and multiplies its `factor` by `data.max_length / max_position_embeddings` (factor 16 over 8192 in that example), so no pair rotates faster than in the checkpoint. **New refusals:** extending a checkpoint whose RoPE block is already scaled in any other way is refused before the model is built, with an error that names the checkpoint's `rope_type` and `factor`. That covers `linear`, `dynamic` or `yarn` on a native `llama3` block, and any `rope_scaling_type` on a native `yarn`, `longrope`, `linear` or `dynamic` block. `rope_scaling_type: longrope` is now refused at config load, in any spelling. Its per-dimension factors exist only on checkpoints already scaled with LongRoPE, and extending those is refused, so it cannot extend any checkpoint; the error names the types that extend a checkpoint without RoPE scaling (`linear`, `dynamic`, `yarn` or `llama3`). Fine-tuning a LongRoPE checkpoint such as Phi-3-mini-128k at its native length, with `rope_scaling_type` unset, is unchanged. A `data.max_length` at or below the checkpoint's `max_position_embeddings` still changes nothing, and checkpoints without RoPE scaling (Llama 2, Mistral, the 8k Llama 3) extend exactly as before. `task: sft` and `task: pretrain`, the two tasks that apply RoPE scaling, both get the new behaviour.

- **`task: moe_lora_routing` on CUDA trains its router gate on an fp32 master weight (#1266 in #1312).** The gate is the only tensor MoLE trains, and on the device `"cuda"` it was cast to bf16 with no fp32 copy for the optimizer, so its gradient and AdamW moments were bf16 as well and most AdamW steps rounded away: after 8 steps at the default `lr` (2e-5), 82.6-91.4% of the gate's weights had never moved. `"cuda:0"` kept it fp32. The gate is now fp32 on every device string and is created on the base model's device, so it moves with the base; the frozen base and the task LoRAs load as before. `mole_gate.pt` is saved in fp32 (`4 x hidden x adapters` bytes), and `soup serve --mole` loads it and a bf16 file from an earlier run alike. A gate trained on CUDA before this fix kept most of its initial weights, so retrain it.

- **GRPO's `accuracy` and verifiable `math` rewards now read the gold answer the way they read the completion (#1226 in #1316).** An Alpaca `output` or a ShareGPT/ChatML reference turn such as `6*7=42\n#### 42` was compared raw, so every correct completion scored 0.0, every group had zero reward variance and GRPO got no signal, with no warning. Both sides now go through one parser: the text after the last `####`, the last `\boxed{}` (a space before the brace and nested braces allowed), or the text after `The answer is` / `Answer:` (also `**Answer**:`, and on the next line), whose number is read from its own clause, so `The answer is 41 apples, not 42.` is 41. An answer phrase that names more than one distinct value is a hedge and states no answer: `The answer is either 41 or 42.` scores 0.0 in both rewards whatever the gold, and a gold written that way is refused (the same value twice, as in `42 (i.e. 42.0)`, is not a hedge). Parenthetical justifications belong to the clause and hedge it (`The answer is 42 (6*7=42).`), while a connective ends the clause, so `The answer is 42 because 6*7=42.` and `The answer is 42, because 6*7=42.` both read 42; a time or a ratio such as `3:45` or `1:1,000` is one answer, compared as text. `$`, `\(...\)`, trailing punctuation, thousands separators such as `1,000` and `1{,}000`, and the Unicode minus are normalised. A numeric gold is compared by value; any other gold, such as `\frac{14}{3}`, is compared as normalised text (case, whitespace, `\left` / `\right` and `\dfrac` ignored), now by `math` as well, so a dataset that mixes numeric and LaTeX golds, such as MATH-500, loads under `math`, and validation prints how many golds are compared as text. `accuracy` no longer pays 0.5 when the gold merely appears in the completion (that paid `#### 420` for a gold of 42) and scores only 0.0 or 1.0; a completion with no marked answer is scored by its last number, so `Six times seven is 42.`, and also `Not 42.`, now score 1.0 instead of 0.5. `math`'s 0.6 near-miss band is unchanged. A row whose gold states no single final answer, such as a multi-line reference with no marked answer, a hedge, or a `#### Solution` heading over a worked solution (a `####` line that is not a number and has more text after it), is refused before generation with its split, row and field and a count of the other such rows. PPO's `accuracy` and `math` rewards use the same scoring, but PPO's data is not validated this way (on the supported trl, PPO scores rollouts with its reward model).

- **`soup data canary insert` wrote chatml canaries into every dataset, so on any other format the loader dropped all of them and `check` could only report OK (#1216 by @abdulwaarith0 in #1317).** `insert` now writes the canaries in the dataset's own format (`--format`, default `auto`): alpaca, sharegpt and chatml, with the carrier as the prompt and the secret as the trained response. Every other format is refused before anything is written, naming the format, as is an explicit `--format` that is not the file's own. The manifest records the format; a manifest without one reads as chatml, and `check` refuses a format it does not know rather than scoring it as chat. `-o canaried.json` now writes a JSON array that loads back, and suffixes the loader would misread are refused.
  (#1216 by @abdulwaarith0 in #1317)

- **`format: tool-calling` dropped every row `soup agent synth` writes, and `soup train --dry-run` still said "Ready to train!" on 0 rows (#1217 by @abdulwaarith0 in #1321).** The converter now reads calls where agent data puts them: assistant turns keep their own `tool_calls` and `tool` turns their `tool_call_id`, in source order, so a trajectory is no longer rewritten as "answer, then call". A top-level `tool_calls` list still works as the single-call form. `tools` is optional, `format: auto` detects `{messages, tools}` as tool-calling, and synthesised rows carry the schema of the tool they call. `soup train` now exits 1, on `--dry-run` and on a real run, when the data loads zero training rows, naming the format and the first dropped row's reason.
  (#1217 by @abdulwaarith0 in #1321)

- **`training.moe_lora` with `backend: unsloth` is refused with a reason that is true for the task (#1264 by @Srinivasan8888 in #1323).**
  The refusal said "unsloth attaches its own fixed attention targets" on every task, but ten tasks have no unsloth setup at all (`reward_model`, `prm`, `classifier`, `reranker`, `cross_encoder`, `distill`, `unlearn`, `moe_lora_routing`, `online_dpo`, `asr`): there it is `backend: unsloth` that goes unapplied, and seven of them now say so. The other three (`asr`, `prm`, `moe_lora_routing`), and `task: sft` with `modality: vision` or `audio`, now get their own refusal first, which is the same on every backend, so switching to `backend: transformers` no longer surfaces a second, different refusal for the same config. Nothing that loaded before stops loading.

- **GRPO's `accuracy` and verifiable `math` rewards now match a LaTeX matrix however its rows are spaced (#1226, follow-up to #1316 in #1332).** Answer normalisation read the second backslash of a row break followed by a space (`\\ `) as the LaTeX control space `\ ` and deleted it, so a gold such as MATH-500's `\begin{pmatrix} -2 \\ -14 \\ -7 \end{pmatrix}` scored 1.0 only against a completion that spaced its rows the same way, and 0.0 against the same matrix written `\begin{pmatrix}-2\\-14\\-7\end{pmatrix}`. A row break is now kept whole, while `\,`, `\!`, `\;`, `\:` and `\ ` are still dropped. After an answer phrase (`The answer is $...$.`), a LaTeX environment such as a matrix is now one answer, like `(3, 4)`: its entries used to count as separate values, so a matrix whose entries stood apart read as a hedge and scored 0.0 (and a gold written that way was refused), while one with a single stand-alone entry was read as that number and could pay a numeric gold equal to it. All 7 MATH-500 golds with a space after a row break are matrices, and each now matches its other spacings; across the 9,792 GSM8K and MATH-500 gold texts, only those 7 answers and their 7 solutions read differently, and no numeric reading changes.

- **`soup train --find-lr` refuses instead of writing a synthetic LR curve when the live sweep cannot run (#1203 reported by @mohdUwaish59, fixed by @umran666 in #1334).** When the config did not load (a YAML syntax error) or the sweep could not run (a base model id that does not exist, a model or dataset that cannot be loaded, or an install without `torch`), the command printed one yellow line, wrote a report whose `recommended_lr` came from the LR schedule alone, and exited 0, so two different nonexistent ids reported the identical number. It now names the cause, exits 1 and writes no report. The refusals added in #1189, for fewer than 4 rows or a loss that turns non-finite before 4 steps have run, are unchanged.

- `soup eval behavior --base-model` now scores each live generation by whether it refuses, not by whether it contains the probe's oracle label, so a healthy model no longer reads MAJOR and a regression no longer reads delta 0.00. `elephant` and `syceval` are refused with `--base-model` before a model loads (exit `3`) and point to `--evidence`, and `looks_like_refusal` recognises `I'm unable` and `I'm not able`. The same classifier scores the `mini_safety` and `mini_over_refusal` suites of `soup eval gate` and `soup ship`, the refusal probe of `soup diagnose` and `soup expect`'s `expect_no_refusal_pattern`, so those now count the contracted forms as refusals; `BUNDLED_SCORER_REVISION` moves to 3, so a `--baseline` recorded before this change warns (#1228 by @developerjillur in #1335).

- **`soup doctor --config` no longer reports a setting written in its off position (#1330 by @Yi-111-a in #1336).** A config that wrote `loss_watchdog: false` or `use_liger: false` on `backend: mlx` was listed as ignored or rejected, so a config written out in full by `soup autopilot` got 15 rows where the same values written minimally got none. A setting is now reported only when its value is switched on; `seed: 0` still counts as set.

- Keep the default email PII fallback linear for long near-miss local parts
  under the existing 50 KB per-row scan cap, covering `soup expect`,
  `soup data pii` and `soup data score` (#1329 by @taljeon in #1360).
- Email snippets now follow the bounded local-part match, so leading punctuation
  and punctuation-adjacent addresses may produce different snippet spans; row-level
  detection remains unchanged (#1329 by @taljeon in #1360).

- **Refuse incompatible pre-tokenized data transformations.** Soup data preprocessing does not apply non-text modality transforms (vision/audio/audio_out), `data.add_new_tokens`, `data.new_special_tokens`, or `data.prompt_strategy` to cached ids, so those combinations are refused at config load (#1273 by @Samearth17 in #1363)

- **The eval gate's MCQ scorer read `\boxed{B.}` and `\boxed{**B**}` as C on a completion that echoes the option list, while the reward parser read B (#1347 by @putramkti in #1365).** The letter tier kept its own boxed regex after #1316 replaced the parser's with a normalising, brace-balancing read, so the two readers disagreed on the same box: the right answer scored wrong and the echoed option scored right. Both now read the boxes through one shared scan in `utils/final_answer.py`, and a test pins the tier to it so they cannot drift apart again.

- **The friendly error reads a 401/403 out of the whole exception chain, not just the outer message (#1361 by @putramkti in #1366).** The status-phrase check (`401 Client Error`, `HTTP Error 403`, ...) searched only the outermost message, so a wrapper that says nothing about status, such as `RuntimeError("Failed to load adapter")` around a text-only `401 Client Error: Unauthorized`, printed the wrapper's message instead of `Authentication failed.` It now reads every exception in the `__cause__`/`__context__` chain, outermost first, so a 403 wrapper of a text-only 401 still reads as the 403 it says. An exception whose own `__str__` raises no longer turns the friendly error into a second traceback either.

- **A failed Anthropic judge/provider call blamed `--judge-base-url` /
  `--base-url` for the failure, though Anthropic ignores that flag entirely
  (#1340 by @SID-6921 in #1370).** `soup data forge --judge-provider
  anthropic --judge-base-url <x>` and `soup data recipe --provider anthropic
  --base-url <x>` both named `<x>` in the terminal error and, for the recipe
  path, in `.checkpoint.json`'s `failed_reason` -- but the API key comes from
  `ANTHROPIC_API_KEY` only, and every call actually goes to
  `https://api.anthropic.com/v1/messages` regardless of what `base_url` was,
  so the message pointed at a host that was never contacted. Both callers
  route through one shared label helper, which now returns the real
  Anthropic origin whenever the provider is Anthropic, ignoring `base_url`;
  `ollama` / `vllm` labels are unaffected since those providers do forward
  `base_url` to the request.

- **`accuracy` and `verifiable/math` scored 0.0 on common MATH answer spellings (#1346 by @AmirF194 in #1371).** `normalize_answer` had no rule for `\text{}` / `\textbf{}` / `\mathrm{}` / `\mbox{}` wrappers, degree marks, or the compact `\frac12` / `\frac1{2}` / `\frac{1}2` spellings, and a phrase answer's one-letter variable prefix (`x = 7`) was read on the completion side but not the reference side. A correct completion in any of those spellings scored 0.0 against a gold in the other. All four are now handled the same way on both sides, though the variable split runs in the two parsers rather than inside the shared normaliser. A variable name is kept rather than discarded: when both sides name one and the names differ (`x = 3` against `y = 3`), the pair still scores 0.0, since a directrix or an asymptote's variable is part of its answer, and its case is part of its name (`x = 5` does not match `\boxed{X = 5}`). A gold with a variable prefix and nothing after it (`\boxed{x =}`, `x =`) states no answer and is now refused before generation (it was compared as the text `x =`). A side effect of running the degree and `\text{}` rewrites before the hedge scan: a clause such as `30^\circ, 60^\circ` now reads as two distinct values and hedges, where it previously read as one: a gold such as `The answer is 30^\circ, 60^\circ.` is now refused before generation, as `The answer is 30, 60.` already is. Units are still not stripped and nothing is evaluated (`\frac{1}{2}` does not equal `0.5`); see `docs/training.md`.

- **`backend: unsloth` is refused at load on the tasks that have no unsloth setup (#1357 by @Srinivasan8888 in #1372).**
  `reward_model`, `prm`, `classifier`, `reranker`, `cross_encoder`, `unlearn` and `moe_lora_routing` accepted `backend: unsloth`, but their trainers never read it: the run trained on plain transformers while `soup train` printed "unsloth (fast mode)", and `soup doctor --config` exited 0. The config now stops at load, naming the task and pointing to `backend: transformers` (and naming `training.unsloth_bnb_4bit` too when it is set, since only unsloth reads it), and `soup doctor --config` exits 2. **This is a breaking change for such a config**, but a one-line one: those runs always trained on plain transformers, so switching to `backend: transformers` changes nothing about what trains. `distill`, `asr` and `online_dpo` keep their own existing refusals, and the docs no longer say unsloth works with every task.

- **`soup ab` no longer rejects H0 on two identical arms when `--alpha` + `--beta` is 1 or more (#1339 by @qwist1233-cpu in #1378).** Each rate was checked in (0, 1) on its own, but the reject boundary `log((1 - beta) / alpha)` stays above the accept boundary `log(beta / (1 - alpha))` only while their sum is below 1. Past that there is no `continue` band left, and because reject is tested first, every verdict in the gap rejected: `--alpha 0.6 --beta 0.5` reported `reject_h0` with `direction worse` on two byte-identical arms, and `--beta 0.95` (a power of 0.95 typed as beta) rejected a true H0 in a quarter of single looks. #1378 refused those pairs, but that refusal does not ship: #1575 retires `--beta` in the same release. The reject boundary is now `log(1 / alpha)` and the accept rule no longer uses `beta`, so there are no boundaries left to cross: two identical arms never reject, and a passed `--beta` is ignored with a warning.
- **`soup ab` names `--effect-size` instead of exiting on a bare `OverflowError` (#1339 by @qwist1233-cpu in #1378).** The drift term squares the standardised effect `--effect-size / pooled standard error`, and `**` raises `OverflowError` where `*` would return infinity, so a huge `--effect-size` ended the run on `Error: OverflowError: (34, 'Result too large')` with no flag named. The ratio is now bounded before it is squared, and the message carries the flag, the ratio and the standard error measured. Because it is the ratio that overflows, a near-constant metric column reached this at the default `--effect-size` too, and is named as well.

- **`training.loraplus_lr_ratio` is now refused at config load on `backend: mlx` (#1324 by @wanjinhao1 in #1383).** The MLX optimizer trains every LoRA tensor at one learning rate, so the ratio was accepted by validation, `soup train --dry-run` and `soup doctor --config`, and then never applied. An MLX config that sets it and loaded before now fails with an error naming the field, the backend and the way out: remove the key, which trains exactly what those runs trained, or use `backend: transformers`. `transformers` and `unsloth` are unaffected.

- `soup data canary insert` spreads the canaries through the file instead of appending
  them, and records each canary's row in the manifest. The loader holds out the file's
  tail as validation (`data.val_split`, 0.1 by default), so from 135 rows on every appended
  canary was held out, no model trained on it, and `soup data canary check` could only
  report OK. `insert` now prints how many canaries the default split trains on
  (#1331 by @MohammadHijjawi97 in #1386).

- **`training.expand_layers` (LLaMA Pro) is refused at config load outside `task: sft` / `task: pretrain` on `backend: transformers` with `modality: text` (#1325 by @wanjinhao1 in #1390).** Block expansion is applied only by those two trainers' text setup, so every other task, `backend: unsloth` or `mlx`, and SFT with `modality: vision` or `audio` accepted the setting and trained the model without the new blocks, with no message. `task: pretrain` with a `modality` other than `text`, and `task: sft` with `modality: audio_out`, are refused too: those trainers ignore that `modality` and did expand the model, so set `modality: text` there. The error names the task, backend and modality it got; `task: sft` and `task: pretrain` on `backend: transformers` with `modality: text` load as before.

- **`soup ab` names a metric column whose spread a float cannot carry, instead of
  exiting on a bare `ZeroDivisionError` or `OverflowError` (#1384 by @MohammadHijjawi97 in #1392).**
  A near-constant column (`0` and `1e-161`) has a positive pooled variance that still
  underflows to a standard error of `0.0` once divided by the row counts, and the
  z-statistic divided by it. A column spread past about `1.3e154` (`0` and `1e200`)
  overflowed while its variance was squared, and an arm whose values summed past the
  largest float gave an infinite mean and a silent `continue` with a `nan`
  log-likelihood ratio. All three now exit 1 with a message that names the column,
  says whether its spread is below or above what the statistic can represent or its
  values are so large that an arm's sum overflows (an arm of `1e308` values), and
  says to rescale the metric.

- **`soup sweep` builds the trainer `soup train` builds, for every task and for `backend: mlx` (#1213 by @putramkti in #1397).** The command carried its own `if/elif` chain over `task`, last extended in v0.40.0. Ten values (`distill`, `tts`, `classifier`, `reranker`, `cross_encoder`, `prm`, `unlearn`, `moe_lora_routing`, `online_dpo`, `asr`) reached the SFT wrapper, and `backend: mlx` was never read at all, so an arm trained next-token cross-entropy while the run registry recorded the configured task and the sweep ranked those numbers as the configured objective. Both commands now resolve through `soup_cli/trainer/dispatch.py::build_trainer`, which refuses a task it has no wrapper for instead of falling back to SFT.

- **Recipes that could not load their model are fixed or removed (#1145 by @Srinivasan8888 in #1398).**
  `mistral-large-3-sft` and `mistral-large-3-dpo` are removed: `mistralai` publishes Mistral-Large-3 in the native `params.json` + `consolidated-*` layout with no `config.json`, which `AutoConfig` cannot read (`docs/models.md` names the Hugging Face-format conversion that exists on the Hub). `minimax-m3-sft` now sets `modality: vision`, so SFT builds its vision-language wrapper and LoRA stays on the language tower; its processor is custom code, so it needs `soup train --trust-remote-code`. `minimax-m3-dpo` is removed, since the DPO trainer cannot load that wrapper (#1393). The catalog loses three recipes.

- **`training.freeze_trainable_layers` without `training.expand_layers` is now refused at config load (#1396 by @wanjinhao1 in #1399).** The field was accepted on its own but read by nothing: without `expand_layers`, `utils/block_expansion.py` returned before looking at it, so a config that set `freeze_trainable_layers: 2` to train only the top two layers trained the whole model (or every LoRA target), with no message. The error names both fields and the remedy (remove it, or set `expand_layers`); the field description and `docs/peft-and-efficiency.md` now say it applies only together with `expand_layers`.

- The refusal for `training.citation_faithful` on a task other than `sft`/`pretrain` no longer says its span-mask runtime "ships in v0.62.1"; that runtime has run in the SFT trainer since v0.71.10. The streaming docs now state that `dpo` and `kto` hold a copy of an untied output head per step, which the VRAM pre-flight already charges (#1353 by @maximilianfeix in #1402).

- `training.moe_aux_loss_coeff` set to a non-default value on `task: sft` with `modality: vision` or `audio` is now refused at config load instead of loading and being ignored: only the text setup applies the MoE auxiliary loss (#1394 by @maximilianfeix in #1404).

- `task: ppo` runs again on trl 0.29: the dataset handed to trl's `PPOTrainer` carries only `input_ids` / `attention_mask`, so its padding collator no longer crashes on the `prompt_text` strings before the first step. A train set with fewer than `batch_size` x `gradient_accumulation_steps` x the number of processes rows is refused before any model loads, instead of hanging (#1391 by @maximilianfeix in #1410).

- **A completed PPO run no longer reports "Loss: unavailable" (#1413 by @Ashutosh-Panda2004 in #1416).** `summarize_training_loss` now takes the per-step loss key from the caller instead of only reading `loss`: the PPO trainer passes `loss/policy_avg`, which is what trl's PPO trainer (0.29) logs. A non-default key is recorded in the result as `loss_key`, so a PPO policy loss is never mistaken for an SFT cross-entropy. SFT/DPO summaries are unchanged.

- `--fsdp` now reaches `task: prm` and `task: moe_lora_routing`, and both `--fsdp` and `--deepspeed` reach `task: online_dpo`; the wrappers stored them and never passed them to their trainer arguments (#1204 by @Oxygen56 in #1417; online_dpo half by @maximilianfeix in #1417).

- **`soup data clean --repair-json` and `--drop-invalid-json` missed nested tool calls and dropped valid documented-shape rows (#1338 by @abdulwaarith0 in #1453).** Both flags read only a flat `arguments` key on the row's top-level `tool_calls`, so calls inside an assistant turn (what `soup agent synth` writes, and the OpenAI fine-tuning shape) were never repaired or checked, and a valid `{"function": {"arguments": ...}}` call read as `""` and got its row dropped. They now read every call the tool-calling loader reads, in both shapes, and leave dict arguments alone. A `tool-calling` row with no top-level `tool_calls` no longer gets an empty list added, so the report stops counting it under "Format Normalization".

- **A failed adapter merge or reward-verifier replace could destroy the only
  good copy on disk (#1450 by @drakeo338 in #1454).** `merge_adapter_to_dense` (shared
  by `soup shrink` and `soup draft distill`) deleted `out_dir` before swapping
  the newly merged model in, so a failure between the two steps — a lock held
  on the staging dir, disk full — left neither the old nor the new model.
  `soup reward synth -o reward.py --force` had the same shape: it overwrote
  the existing verifier with the candidate before calibrating it, then deleted
  the file outright on refusal or error, destroying a previously working
  verifier. `merge_adapter_to_dense` now moves the previous artifact aside
  instead of deleting it, and restores it if the replacement does not make it
  all the way to accepted. `soup reward synth` instead writes the candidate to
  a temp sibling and never touches `-o` until it is accepted, so there is
  nothing to move aside or restore on a refusal or error — the previous
  verifier is simply never disturbed.

- `soup tokenizer train` now saves a complete byte-level BPE: a matching `ByteLevel` decoder (so `decode(encode(x)) == x`, no more `Ġ` for spaces), the 256-byte initial alphabet (so unseen characters never become `<unk>`), no hard-coded `unk_token`, and a `--vocab-size` floor that accounts for the byte alphabet plus special tokens (#1429 by @wanjinhao1 in #1455).

- **`--trust-remote-code` is now threaded through on the five paths that dropped it (#1449 by @drakeo338 in #1456).**
  `soup data download` forwards the flag to `load_dataset` and only prints the
  "Remote Code Warning" panel when it is set. `soup eval auto` forwards it to
  both `benchmark()` (appended to the lm-eval `model_args` from the resolved
  flag only, never from `adapter_config.json`) and `custom()`, which now
  passes it to the `pipeline(...)` load. `soup train --find-lr`'s live sweep
  loads the tokenizer and model with the resolved flag instead of a hardcoded
  `False`. `soup draft distill` and `soup shrink` (heal step) now append
  `--trust-remote-code` to the child `soup train` argv when it was resolved
  true, instead of asking the user to re-run with a flag they already passed.
  On `datasets>=4`, which dropped `trust_remote_code` support upstream and
  now just logs an error and ignores it, `soup data download
  --trust-remote-code` is refused outright instead of silently forwarding a
  flag that would do nothing.
  `soup eval benchmark` and `soup eval custom` gain their own
  `--trust-remote-code` option (off by default).

- **`soup expect` fails closed on unreadable JSONL input (#1434 by @wanjinhao1 in #1457).** The gate now reads the dataset as `utf-8-sig` like the training loader, so a UTF-8 BOM no longer hides the first row from PII and other checks. Malformed JSON lines and valid JSON that is not an object are refused as an input error (exit 3) with their line numbers instead of being silently skipped, and a file with no checkable rows is refused rather than passing with `Rows 0`.

- `soup card` now shows the newest score of a benchmark that was evaluated more than once, not the oldest: the scorecard walked the tracker's newest-first rows and kept the last one, and rows with an identical timestamp now resolve to the later insert. It uses the same newest-row helper as `soup registry diff` and `soup eval compare` (#1315 by @Ru0k3 in #1459).

- `soup infer`, `soup chat` and `soup diff` now honor `--device` when loading
  models instead of always passing `device_map="auto"` and
  `torch_dtype=torch.float16` to `from_pretrained` (#1443 by @drakeo338 in
  #1460). `cpu` (and `mlx` on this path) is pinned to the CPU in `float32`;
  `cuda:N` and `mps` are pinned to that device; plain `cuda` keeps
  `device_map="auto"`. GPUs keep `float16`. Values are case-insensitive, and
  an unknown `--device` (`gpu`, `auto`, `cuda:0,1`, ...) now raises an error
  instead of being printed and ignored.
- Default-path change: the CLIs always auto-detect a device when `--device` is
  not given, so these rules apply without the flag too, and to `soup bench`,
  which loads through `soup infer`. On a CPU-only host the model now loads in
  `float32` pinned to the CPU instead of `"auto"`/`float16`, which doubles its
  RAM use (a 7B model needs about 28 GB instead of 14 GB) (#1443 by @drakeo338 in
  #1460).

- **`soup expect` refuses suite keys and argument names an expectation does not take, instead of running it on its defaults (#1428 by @Srinivasan8888 in #1461).**
  `parse_suite_spec` read only `name` and `args` and dropped every other key, and the dispatcher read arguments with `args.get(key, default)`, so the `min_tokens` / `max_tokens` example in `docs/compliance.md` (written beside `name` instead of under `args:`) and a misspelled argument such as `max_token` both ran `expect_token_length_between` on its defaults of 1 and 1,048,576 tokens and passed a 5000-token row with exit 0. A key beside `name` or an argument the expectation does not take is now refused when the suite loads (exit 3, before the data file is opened), with the key named and either "belongs under `args:`" or the nearest valid name; `min_tokens` / `max_tokens` / `threshold` are validated at parse time too. The `docs/compliance.md` example moves to the `args:` form. A suite that carried one of these keys now exits 3 on its next run, which is the point.

- **`mini_arithmetic` no longer sits at a 1.000 ceiling (#1192 by @AaronProbha18 in #1462).**
  The 40-row suite replaces 12 single-step rows with 16 multi-step and
  large-operand ones; Qwen2.5-1.5B-Instruct measures 0.800 against 0.375 on
  SmolLM2-135M-Instruct, and the bundled scorer provenance revision moves to 4
  so pre-change baselines are not silently compared on the new scale.

- **`soup data augment` wrote rows with an empty rewritten field, and `soup distill-prompt` silently dropped every failed row, both still reporting success (#1274 by @AaronProbha18 in #1463).** Both commands built their LLM provider without `raise_on_error`, so a connection error, HTTP 404/429/500 or malformed response came back as empty text instead of raising — `augment` rewrote a field to `""` and kept the row, `distill-prompt` dropped the row and wrote an empty output file with `Rows: 0`, and both exited 0. Same defect class as #1221 (fixed for `soup data forge` in #1261): the provider is now built with `raise_on_error=True` for both commands, failures (a raised error or an empty/whitespace-only reply) are counted via the shared `ForgeJudgeStats`, and a row is never written with an empty or unrewritten field. A run where every call failed exits 1 without writing an output file; a partial outage keeps only the successful rows under a yellow "with provider failures" panel reporting `N of M provider calls failed` with the endpoint and first error.

- **`soup compile`, quantized GGUF export and `task: unlearn` now validate their destination before the expensive step (#1445 by @wanjinhao1 in #1469).** `soup compile -o <path outside cwd, or a symlink>` used to run the whole optimizer and its LLM calls before refusing the path with an unhandled `ValueError` (exit 1); it now refuses at plan time and exits 2, with the write-time check kept as the TOCTOU guard. A quantized GGUF export into a directory that did not exist yet crashed in `mkdtemp` after the adapter was merged, and the cleanup deleted the merged model; the parent directory is now created up front. `task: unlearn` used to load the policy and a frozen reference copy and run every NPO / SimNPO / RMU step before refusing an `output` outside the working directory or a symlinked one; `setup()` now refuses first, before any model is loaded.
`soup compile --plan-only` now also refuses such a destination (exit 2) instead of printing the plan and exiting 0.

- Machine-readable output from `soup audit-log tail --json`, `soup bom emit`, `soup eval leaderboard --format json|csv` and `soup profile --json` now writes straight to stdout via `typer.echo` instead of the Rich console, so a piped record is no longer folded at 80 columns and `[...]` is kept verbatim instead of being parsed as markup (which previously crashed the command on a token like `[/x]`) (#1468 by @abhinyaay in #1470).

- `soup data generate --template reasoning` now tells the generator how to end each solution, so the golds it writes are readable by GRPO's `accuracy` and `verifiable` math rewards: `math` rows end with `#### <number>` on its own line, `logic` and `code` rows with an `Answer: <answer>` line. Before, the prompt asked only for "the final answer", and a gold such as `So there are 42 apples.` makes `soup train` refuse the dataset before generation (#1349 by @Momoyeyu in #1471).

- Model output is now read by its structure. The LLM judge (`soup eval judge`, the eval gate's judge task and `soup ship`'s `judge_score`) reads the reply's whole JSON object, so reasoning that quotes a nested object no longer scores every criterion at the minimum; criterion keys match case-insensitively, `{"score": N}` is accepted, and a missing or non-numeric criterion, or one scored under two spellings, raises an error naming it; a reply longer than 65,536 characters is refused unread. `soup eval checklist` keywords and `soup eval behavior --evidence` oracle labels match when they start or end with punctuation (`-5`, `(B)`, `C++`, `100%`), still as whole words on both sides. The judge node of `soup data recipe --execute` reads the first word of the reply: `OK` keeps the row, `REJECT` or `NOT OK` drops it, and any other reply counts as a failed call; its default prompt asks for a one-word verdict (#1467 by @developerjillur in #1472).

- Autopilot no longer enables DoRA on a GPTQ / AWQ / AQLM / EETQ base. peft
  refuses DoRA on those layers, so a large dataset produced a config that
  failed after the model load. `decide_peft` takes the chosen `quantization`
  and keeps `use_dora` off for them (#1466 by @drakeo338 in #1473).

- Wire the "Standalone Sweep Config" docs example: `soup sweep --sweep-config sweep.yaml` now loads `strategy`/`n_runs`/`seed`/`params` from a separate YAML file, instead of `--param` being a required option that the documented invocation never supplied (#1465 by @AmirF194 in #1475).

- Fixed KTO reaching TRL with a per-device batch of 1: the `batch_size >= 2` parse-time refusal now covers every `task='kto'` config (not only `stream_layers`), `batch_size: auto` floors its estimate at 2 instead of 1, and `soup local-rl train --train-method kto` writes `batch_size: 2` (#1420 by @wanjinhao1 in #1476).

- `soup serve --device` now reaches the transformers loader. `_load_model` ignored
  its `device` argument and always passed `device_map="auto"`, so `--device cpu`
  still placed the model on a visible GPU. The base model, the merged model and the
  speculative draft model now use the same placement rule as `soup infer` / `chat` /
  `diff`: `cuda:N` is accepted and pinned to that GPU (it was accepted and ignored
  before), `mps` is pinned, `mlx` runs on the CPU, and `cuda` keeps
  `device_map="auto"`. Unknown values are refused with exit code 2 naming
  `cpu, cuda, cuda:<index> or mps`; `--device` is case-insensitive. An explicit
  `--kv-cache-type bf16` / `f16` still sets the load dtype on every device.
  Behaviour change: on a CPU-only host the default load is now float32 pinned to the
  CPU, which doubles its RAM (a 7B model needs about 28 GB instead of 14 GB)
  (#1478 by @drakeo338 in #1479).

- `soup diagnose` now detects verbatim memorization on long training rows by measuring
  clipped completion-bigram precision against the held-out suffix. The fixed 64-token live
  generation budget no longer dilutes overlap as the untouched suffix grows, while adjacent-token
  matching avoids false positives from shared vocabulary and repetition loops. This changes the
  threshold semantics from set-Jaccard, so earlier memorization results are not directly comparable
  (#1437 by @fukalous in #1481).

- **`mistral-medium-3-5-sft` can now load and attach LoRA (#1395 by @Areen-09 in #1482).** The recipe base model (`mistralai/Mistral-Medium-3.5-128B`) uses `Mistral3ForConditionalGeneration`, a vision-language wrapper. The recipe was missing `modality: vision` under `task: sft`, causing the loader to dispatch to `AutoModelForCausalLM` and fail. In addition, `target_modules: auto` had no mapping for `mistral3` in `MOE_TEXT_LORA_TARGETS`, which now targets the language model attention projections (`.*language_model\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)`) while excluding the vision tower, multimodal projector, and MLP projections. The recipe is also unpinned from the cannot-attach exceptions list in `scripts/check_recipe_attach.py`.

- **A judge that cannot be reached is reported as unavailable, not scored as a tie (#1447 by @Srinivasan8888 in #1484).** Every LLM-judge request now goes through one helper that retries 429, 5xx and transport errors within a bound (honouring `Retry-After`), fails at once on a 4xx that will not heal, treats a reply with no usable content as a judge error, and raises `JudgeUnavailableError` naming the judge URL once the retries are spent. `soup ship --task-mode pairwise` exits 1 with that message instead of scoring a dead judge as a measured 0.5 win-rate and printing DON'T SHIP; `soup eval judge` skips and counts the failing row and still scores, shows and saves the rest; a training eval gate still fails closed, and its log line says which task could not be scored and why instead of calling it a regression. The Online DPO adapter keeps leaving a failed pair unranked, as #1225 designed.

- `soup version --json --full` and `soup cost --json` now write their JSON straight to stdout instead of through the Rich console. Rich folds a document wider than the console (80 columns when the output is piped) and reads `[...]` inside a string as markup, so a long record was split across lines, a value such as `[v2]` was dropped, and `[/x]` aborted the command with `MarkupError`. `soup eval against --json-only` had the same defect and is fixed with them. A test walks `src/soup_cli/` and fails if any command puts serialised JSON back through `console.print` (#1480 by @putramkti in #1485).

- `soup eval`'s custom-task `contains` scorer now requires the expected string to appear as a whole token, so `14` no longer counts as containing `4` and `#### 420` no longer contains `42`. This changes scores: a gate baseline or a `soup eval custom --run-id` result recorded with the old substring `contains` can be higher than the same outputs score now, and the baseline's scorer stamp does not notice, so re-measure those before comparing. A new `scoring: "answer"` mode compares the final answer the model states (via the same parser GRPO's `accuracy` reward uses) instead of substring-matching, and refuses task rows whose `expected` states no single answer instead of silently scoring 0 (#1342 by @Momoyeyu in #1490).

- **Refuse `.soup` and `.soup/mcp-runs` directory symlinks and junctions at plan time (#1211 by @kok-o in #1492).**
  Following #1158 and #1171, `snapshot_config` now inspects the directory hierarchy with `refuse_linked_dirs` before creating directories or writing `config.yaml`, preventing config snapshot redirection outside the working directory when `.soup` or `.soup/mcp-runs` is a symbolic link or Windows junction.

- **`soup data clean` no longer drops a tool-calling trajectory whose call-only assistant turn has empty content (#1477 by @abdulwaarith0 in #1493).** With no flags, the `--min-tokens` check ran on every assistant turn in the `chatml` branch, so a turn with `"content": ""` and a non-empty `tool_calls`, the shape the tool-calling format itself writes, dropped its row as "Empty / Whitespace Turns" while the same turn with `"content": null` was kept and the loader accepted both. A turn that carries tool calls is now exempt from `--min-tokens`; turns with no calls keep the rule.

- `format: auto` now detects a dataset's format from a bounded prefix of rows instead of the first row alone, so a first row that happens to lack an optional key no longer picks the poorer converter for the whole file and silently drops that key from every row. A mixed file either upgrades to the richer format of the same family when its converter accepts the poorer rows, or is refused with a message naming both shapes and the row where each was first seen. Files that used to load under `format: auto` with a key stripped or rows dropped under a warning now stop at load; set `data.format` to the shape you want to keep. The Web UI data explorer now reports the refusal instead of failing, and the auto-detection behaviour is documented. (#1424 by @chaitanya-y20 in #1496).

- **`soup data from-traces` reads the signals Soup's own producers write, instead of reporting a green 0 pairs (#1440 by @umran666 in #1498).** The pair builders key on the per-trace `signal` and on `edited_output`, but the `soup-serve` parser read the signal only from a nested `feedback.rating` that no Soup producer writes: `soup ingest` emits a top-level `signal` and `soup serve --trace-log` records no feedback at all, so every ingest row read as `none` and `soup data from-traces --signal thumbs_up` wrote an empty file and exited 0. The top-level `signal` is now read (`feedback.rating` stays as a fallback for older logs), an `edited_output` under any of its three names makes `--signal user_edit` buildable from a real record, and a harvest that reads traces but matches none now names how many it read, which signal it wanted and which signals were present instead of reporting a normal write of 0. The examples in `docs/data.md` are corrected too: `--format soup_serve`, `--signal regeneration` and a single-file `--logs` for the soup-serve parser are all refused by the CLI.

- **`soup expect` refuses a top-level suite key other than `expectations` (#1483 by @Srinivasan8888 in #1499).** `parse_suite_spec` read only `expectations` and dropped every other top-level key, so a suite with `fail_fast: true` (or any typo) beside `expectations:` loaded, ran and exited 0 with no message about the key. It is now refused when the suite loads, before the data file is opened (exit 3), with every unknown key named; the key text is echoed as a Python literal so a hostile key cannot reach the terminal as a control sequence. This is the same check #1461 added one level down, inside an expectation entry.

- **`task: simpo` refuses a dataset whose completion would be truncated away, instead of training to NaN behind a logged loss of 0.0 (#1208 by @umran666 in #1500).** On trl 0.29.x, a SimPO row whose `chosen` outgrows `data.max_length` loses its whole `rejected`: trl truncates each answer to `max_length` minus the **longer** answer's length, so the shorter side of a lopsided pair ends with zero tokens and SimPO's length-normalised log-probability becomes 0/0. Every LoRA tensor went NaN while transformers' `logging_nan_inf_filter` logged the loss as `0.0`. `soup train` now refuses before the first step, naming how many rows are affected, which ones, and the `data.max_length` involved; a pair that keeps trainable tokens on both sides is left alone, including a long `chosen` beside a shorter `rejected` that trl truncated to 64/14.

- **GRPO's `accuracy` and verifiable `math` rewards now end the answer clause at justification connectives `because` and `since` (#1328).** Previously, an inline justification such as `The answer is 42 because 6*7=42.` stayed in the clause and counted all numbers, treating the answer as a hedge and scoring 0.0. The clause now ends at `because` or `since`, correctly reading 42 (#1328 by @kok-o in #1501).

- **`distill_mode: sequence` now numbers rows the way token mode does, and refuses a split with nothing left (#1344 in #1502).** In `distill_mode: sequence`, original source row indices are preserved when skipping rows with no prompt turn, user's 1-based source row positions are reported on causal-loss target failures (noting the assistant response was generated by the teacher), skipped row counts and positions are printed at setup, and splits left with 0 usable rows are refused before training starts (#1344 by @kok-o in #1502).

- Resuming a QLoRA SFT run no longer upcasts the trainable LoRA adapter from bf16
  to fp32: the dtypes recorded before `Trainer._load_from_checkpoint` are restored
  right after it, before the optimizer is created, so a resumed run reproduces the
  uninterrupted one (#1503 by @bobbonauta in #1504).

- **`soup expect` requires a judge for `expect_chosen_preferred_over_rejected_by_judge` instead of silently scoring 1.0 (#1433 by @kok-o in #1505).**
  `expect_chosen_preferred_over_rejected_by_judge` gave every row 1.0 when `judge_fn` was `None`, so `soup expect` ran with no judge and passed every preference dataset at any threshold. The expectation now flags preference rows as violations when no judge is configured (exit 2). A judge can be supplied via `soup expect --judge <url>` or in suite YAML under `args: {judge: "ollama://llama3.1", threshold: 0.7}`. Callers that deliberately want to trust existing labels without running a judge can opt in with `args: {advisory: true}`.

- **`soup train --cloud lambda` keeps running across transient SSH disconnects instead of terminating the instance (#1439 by @kok-o in #1506).**
  The rendered Lambda lifecycle controller previously relied on a single idle 90000s SSH session without keepalive probes and treated SSH exit 255 as a training failure, immediately terminating the instance and losing ongoing runs when NAT or Wi-Fi dropped. The controller now configures `ServerAliveInterval=30` and `ServerAliveCountMax=6`, polls training status with short SSH calls checking cloud-init and `/home/ubuntu/soup.exit`, retries transient transport errors (exit 255 and timeouts) over a 15-minute retry window with exponential backoff before giving up, retries checkpoint copy (`scp`) across a 15-minute window before instance termination, and prints clear distinct messages for connection loss, remote training exit, and copy failures.

- **`training.warmup_ratio` dynamically resolves over actual optimizer steps instead of a precomputed estimate (#1431 by @kok-o in #1507).**
  Trainer wrappers previously turned `warmup_ratio` into fixed `warmup_steps` using raw dataset row counts before constructing the trainer. Under GRPO (where prompts repeat `num_generations` times), SFT with packing or multipack, and pretrain with packing, the trainer executed a different step count, causing warmup to shrink or cover the entire run so peak learning rate was never reached. All wrappers now pass `warmup_ratio` through via `resolve_trainer_warmup_steps` directly to `TrainingArguments(warmup_steps=...)`, letting transformers resolve warmup dynamically as `ceil(real_optimizer_steps * warmup_ratio)`. Runs with fewer than 34 optimizer steps at the default `warmup_ratio: 0.03` now resolve to exactly one warmup step, whose first step trains at learning rate 0.

- **Decouple `soup diagnose` forgetting probe from training dataset rows (#1436 by @kok-o in #1508).** The live forgetting probe previously evaluated retention on the first 12 rows of `--dataset`, which is documented as the adapter's own training data. Fine-tuned adapters match or outperform their base model on their own training targets, causing the probe to falsely report `OK 1.000` even when catastrophic forgetting occurred across general tasks. The forgetting probe now evaluates against a separate `--holdout <path>` JSONL option or a built-in general knowledge probe set with reference answers when `--holdout` is omitted, while `--dataset` continues feeding the memorization, format, and citation probes. If `--holdout` has no usable prompt/answer rows, forgetting reports `NOT_RUN` rather than a false pass.

- Layer streaming now materializes models cached by huggingface_hub 1.32 or later,
  whose Xet blobs live in the cache-wide shared blob store; links that leave both the
  repo's blob store and the marked shared store are still refused (#1509 by @bobbonauta in #1510).

- Layer streaming now accepts a local path to a Hugging Face cache snapshot
  (`snapshots/<commit>`): it is copied to regular files like the snapshot behind a Hub id,
  instead of stopping with "no .safetensors weight files found" (#1511 by @bobbonauta in #1512).

- **`training.preference_loss_weights` refuses a blend it cannot compute, instead of training one loss and reporting several (#1425 by @umran666 in #1513).** No trl 0.29 trainer puts per-sequence log-probs on the batch, so every requested term went uncomputed and the wrapper fell back to `primary_loss * primary_weight`: `{dpo: 0.7, simpo: 0.3}` printed `Multi-objective preference loss: 0.70·dpo + 0.30·simpo` at setup and then trained DPO alone scaled by 0.7, with nothing in the output saying so. A blend that cannot be evaluated now stops the run, naming the terms it could not compute, the ones it could, and what was requested; the weighted-sum path itself is unchanged, so a trl that does carry log-probs still blends. Every `preference_loss_weights` config therefore stops at the first step until the real blend lands: to train one loss, remove the weights and set `training.preference_loss`.

- **`task: ppo` refuses the reward-hacking and echo-trap flags it cannot feed, instead of loading them inert (#1441 by @umran666 in #1514).** The `trl` PPO trainer this build supports takes no reward functions — its reward comes from the reward model it calls internally — so the wrappers Soup built for the shared signal buffer were dropped, the buffer was never written, and because the buffer object existed the detector's log-reading fallback returned early as well. A run with `reward_hack_detector`, `reward_hack_mitigation` or `echo_trap_enabled` was announced and then never halted, mitigated or trapped. Those three flags are now refused before the model loads, naming every flag set and pointing at `task: grpo`, where the buffer is fed. The on-GPU buffer feed for PPO rollouts is still open.

- `soup local-rl train` without `--once` renders `--min-pairs` and `-o` into the systemd `.service` and launchd `.plist` it writes, so the nightly job trains with the threshold and output directory the user asked for instead of the defaults (10 pairs, `local_rl_adapter/`). A `--min-pairs` the nightly run would refuse now exits 2 before any unit file is written (part of #1448 by @MohammadHijjawi97 in #1517).

- **`soup advise run --probe-model X` ran the live probe and hid its ROI table (#1448 by @qwist1233-cpu in #1521).** `--probe-model` implies `--probe` and the probe ran under that derived flag, but the `ROI deltas (probe)` table was gated on the raw `--probe` option. The table now follows the same flag as the probe.

- **`soup diagnose` live probes now read ShareGPT, Alpaca `input`, ChatML, tool-calling, DPO, KTO and plaintext datasets (#1435 by @bakdauletbaktygaliyev in [#1526](https://github.com/MakazhanAlpamys/Soup/pull/1526)).** Rows go through the same format converters training uses, with the old flat-key lookup as a fallback, so these datasets give the format and mode-collapse probes a prompt/answer pair instead of `NOT_RUN`. Memorization no longer waits for pairs and runs on plaintext rows. LLaVA and audio rows are skipped because the probes are text-only, a prompt over 8192 characters is skipped instead of failing the probe, the `NOT_RUN` reason names the dominant cause, and skipped rows are counted in `extras.rows_skipped` and printed as `Rows skipped: ...`. For multi-turn ChatML rows the probe prompt now stops at the first assistant reply. Two more visible changes: rows with an `audio` field and `messages` used to be probed as text and are now `NOT_RUN` (exit 3 unless `--allow-not-run`), and Alpaca prompts now include `input` and `system`, which moves existing scores. A prompt containing a NUL byte is skipped like an over-long one instead of failing the probe. A `--holdout` file is read the same way: ShareGPT, Alpaca and ChatML hold-outs now feed the forgetting probe, and LLaVA and audio hold-out rows are skipped, so an audio hold-out is `NOT_RUN` like an audio `--dataset`.

- **`training.freeze_layers` / `training.freeze_ratio` are refused at load on the tasks that never applied them (#1497 by @abdulwaarith0 in #1532).** Both fields loaded on every task, but only `task: sft` (and `tts`, which trains through the SFT trainer) freezes layers. On every other task (`dpo`, `grpo`, `ppo`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `preference`, `reward_model`, `pretrain`, `embedding`, `prm`, `classifier`, `reranker`, `cross_encoder`, `distill`, `unlearn`, `moe_lora_routing`, `online_dpo`, `asr`) the run trained every layer and said nothing. **This is a breaking change: a config that sets either field on one of those tasks loaded before and now stops at load**, with a message naming the field, the task and the fix (use `task: sft`, or remove the key), and `soup doctor --config` exits 2. Removing the key changes nothing about what trains, since those runs were already training every layer. `sft` and `tts` configs, and dumped configs (which write both fields as `null`), are unchanged.

- `soup reward synth`'s numeric verifier now reads both the gold and the completion with the shared answer parser the built-in `math` / `accuracy` rewards use, so `1,000`, `#### 1,000`, `\boxed {42}` and `#### −42` spellings score the same under the generated verifier as under the built-ins, and malformed comma groupings like `10,00` are no longer detected as numeric. Regeneration also stops paying some completions the old regex paid — `####` or `\boxed{}` answers carrying units, `%` or a non-dollar currency symbol or code (`#### 42 apples`, `\boxed{42\%}`, `#### €42`), hedged or justified answer phrases, and a wrong last box after a right first now score 0, matching the built-ins (bare `42%` and `The answer is 42%.` still pay). A `reward.py` synthesized before this version scored the newly-accepted spellings 0, so regenerate it before comparing runs; the emitted file now imports `soup_cli` rather than being fully self-contained. The calibration report and `--output-report` also name every reference the verifier rejects (`rejected_references`) instead of reporting only a refusal rate, and a refused run lists the first ten rejected references in the panel (#1350 by @Momoyeyu in #1533).

- Fix `soup train --task unlearn` crashing with `KeyError: 'duration'` in the
  "Training Complete!" panel after the run had already finished: the adapter was
  saved and the tracker row written, but the unlearn wrapper was the one
  transformers-backend trainer that returned only `duration_secs` without the
  pre-formatted `duration` string. The wrapper now formats `duration` like every
  other wrapper, and the completion panel falls back to formatting
  `duration_secs` itself so no future wrapper can lose a finished run's summary
  (#1529 by @vaputa in #1537).

- `soup eval`'s custom-task `contains` scorer is linear again in the length of the output and of `expected`. The whole-token match from #1490 restarted its search at every position, so a 400,000-character output against a 200,000-character `expected` took 68 s; it now takes milliseconds. Both sides are case-folded with `str.lower()`, as in v0.75.1, so `istanbul` no longer matches `İstanbul` and `s` no longer matches `ſ`; no other score changes (#1528 by @UroojFatima-052 in #1540).

- **Aligned ULD (`uld_strategy: wasserstein_aligned`) no longer scores student positions that have no teacher target (#1426 by @developerjillur in #1541).** A byte-fallback piece, an empty decoded piece or a student-only special token aligns to no teacher token; it got an all-zero teacher row, which softmaxes to uniform, and each such position added about `V/2` to the loss. These positions now leave both the sum and the denominator under the label mask, the attention mask and no mask, and the trainer decodes student special tokens to `""`, as it already does for the teacher.

- **76 tests across 10 files downloaded tiny models/tokenizers from the
  Hugging Face Hub on every run, none marked `smoke` and none guarded against
  the network being unavailable (#1356 by @SID-6921 in #1544).** A real sweep
  (`HF_HUB_OFFLINE=1`, empty `HF_HOME`, `pytest tests/ -m "not smoke"`) found
  a much larger landscape than a preliminary grep-based pass had estimated.
  Every site now builds its tiny model and tokenizer locally instead --
  a `ByteLevelBPETokenizer` trained from a few lines of in-file text feeding
  a real, randomly-initialised `LlamaConfig`/`GPT2Config` model, matching the
  existing `_write_tiny_qwen2` convention -- since none of the fixed tests
  needed real pretrained weights or vocabulary, only a working tokenizer and
  a save/reload round-trip. A full-suite offline re-sweep afterward confirms
  zero remaining Hub-dependent failures.

- `notebooks/proof-4gb.ipynb` runs again on a hosted multi-GPU runtime with a current `huggingface-hub`: the install cell caps `huggingface-hub<1.32.0` (soup-cli 0.75.0 refuses the shared blob store introduced in 1.32, #1509) and the notebook sets `CUDA_VISIBLE_DEVICES` to one card before torch is imported, because layer streaming cannot run under `nn.DataParallel` (#1543 by @here-2007 in #1545).

- `eval.auto_eval: true` now runs the configured `eval.benchmarks` and `eval.custom_tasks` after `soup train` finishes; before, no trainer handed the evaluation config to the training callback, so nothing ran (#1487 by @commit-msuyog in #1546).

- Terminal output strips the C1 control range (U+0080-U+009F) as well as C0
  and DEL, and text read from files is printed through the shared
  `for_terminal` helper instead of raw, through `escape()` alone, or quoted
  with `repr()` after escaping: adapter-config base model names in `soup
  chat`, `soup merge`, `soup serve`, `soup export`, `soup eval benchmark` and
  the `--trust-remote-code` warning; `soup adapters list` / `info` /
  `compare` / `diff`; benchmark names in the `soup ship` footer; `soup edit
  diff`; the SFT "cannot open image/audio" warnings; audio file names in the
  `soup infer --task asr` skip warnings; the config loader's validation
  lines; `soup probe interference` loss keys; and the Web UI's log line for a
  refused config. Markup in such a value no longer changes an exit code: a
  signature sidecar whose `backend` name contains `[/]` gets `soup attest
  verify`'s documented exit code 3 instead of 1, and a path or key
  containing `[/]` gets exit code 2 instead of 1 from `soup env lock` / `env
  fix`, `soup tunability`, `soup deploy autopilot --measure` and `soup probe
  interference` (`soup data forge` and `soup adapters arithmetic` now print
  such a name as text too). A `.can` manifest `author` refuses control
  characters (#1552).

- **`soup init` refuses a symbolic link, junction or other reparse point at the config path it writes, and MCP planning creates the config snapshot's run directory fresh (#1553).**
  `soup init` wrote the starter config with a plain write that follows a link at the target (`soup.yaml` or `--output`), and a dangling link even skipped the overwrite prompt. It now refuses a symbolic link, junction or other reparse point there before prompting or running the wizard, naming the path, matching the link refusal the other config writers apply, and it creates or opens the file without following a link that appears there afterwards: nothing is written through such a link and nothing is created at its target. Ordinary paths, including ones outside the working directory or reached through a linked parent directory, behave as before. Following #1492, `train_start` under `soup mcp serve` now also checks the run directory `.soup/mcp-runs/<run_id>` itself, creates it with `exist_ok=False`, and creates `config.yaml` exclusively (`O_EXCL`) without following a link, refusing the plan with a path-free error when the run directory or the file already exists or is a link.

- The outbound loopback allowlist is now a single shared set:
  ``http://[::1]:<port>`` judge URLs are accepted by ``GateTask.judge_model``,
  ``soup ship --judge-model``, and the online-DPO trainer's
  ``_parse_judge_url``, matching the judge API-base and vLLM checks. Each
  gate previously kept its own ``("localhost", "127.0.0.1")`` literal, which
  refused the IPv6 loopback (#1548 by @akshatshahh in #1554).

- `training.lora.use_dora: true` with `training.quantization: gptq`, `awq`,
  `aqlm` or `eetq` is now refused when the config is loaded. peft has no DoRA
  variant for those layers, so the config used to load and fail only when the
  adapter was attached, after the model load. Autopilot's `decide_peft` reads
  the same format set (#1466 by @ybai08 in #1555).

- Fix `soup data from-traces --format soup-serve` silently writing an empty output (exit 0, `Wrote 0 preference pair(s)`) when `--logs` pointed at a single file: the soup-serve reader wants the directory `soup serve --trace-log` writes, and a file is now refused with a message naming the option and the expected shape; a source that reads zero traces at all now says so ahead of the count line, so an empty output no longer reads the same as a completed harvest (#1530 by @vaputa in #1557).

- **`training.quantization: mxfp4` passed config validation and then failed at
  the model load (#1466 by @ybai08 in #1558).** Soup built
  `BitsAndBytesConfig(bnb_4bit_quant_type="mxfp4")`, which bitsandbytes
  rejects (it accepts `nf4` and `fp4` only) and which clashes with the
  `Mxfp4Config` a pre-quantized MXFP4 checkpoint carries, including the bases
  `soup autopilot` passes through as `mxfp4`. The loader now returns
  transformers' `Mxfp4Config(dequantize=True)`, since transformers does not
  train MXFP4 weights as loaded, so the model trains in bf16. `mxfp4` no
  longer goes through `prepare_model_for_kbit_training`, which would cast the
  dequantized model to fp32. Two configs that validated before are now
  refused, neither of which could train: `bnb_4bit_quant_storage` with `mxfp4`
  (it is a bitsandbytes argument), and `mxfp4` with a local base whose
  `config.json` does not declare `quant_method: mxfp4`.
  The hardware-fit calculator (`soup_cli.utils.hardware_fit`) now prices an
  `mxfp4` base at 2 bytes per parameter, as the bf16 model it loads (it was
  0.55).

- Lao, Khmer, Myanmar, halfwidth katakana and CJK Ext C+ are now scored per character pair like Thai; symbol-only answers are compared by their symbols; rows and answer sets with no tokens are skipped and counted. (#1343 by @Hitesh-XS in #1559)

- **`soup migrate` maps every LLaMA-Factory `pref_loss` and TRL trainer class to its Soup task, and refuses the rest by name (#1214 by @Srinivasan8888 in #1560).** LLaMA-Factory `stage: dpo` with `pref_loss: ipo` now migrates to `task: ipo` instead of plain `dpo`; `hinge`, `kto_pair` and any other value with no Soup task make `soup migrate` exit 1 naming the value and the supported list, instead of loading as `dpo` with no warning. Unsloth notebooks: `RewardTrainer` -> `reward_model`, `BCOTrainer` -> `bco`, `OnlineDPOTrainer` -> `online_dpo` (with a warning that the judge / reward model is not carried over), and `CPOTrainer` -> `simpo` only when the `loss_type="simpo"` on that trainer's own call or its `args=` config; other CPO losses are refused. `SFTConfig` and the other TRL `*Config` classes are now read for hyperparameters, so a notebook passing `args=SFTConfig(...)` keeps its batch size, epochs, learning rate and output directory. `OnlineDPOTrainer` notebooks migrate with `training.online_dpo_judge: ollama://REPLACE-ME` as a placeholder (TRL's judge is a Python object with no string form) and a warning naming it; replace it, or set `training.reward_model` instead. Migrations that used to produce a config and now stop with exit 1: LLaMA-Factory `stage: dpo` with `pref_loss: hinge`, `kto_pair`, any other unlisted value, or a scalar that is not a string such as `5` or `true` (they loaded as `task: dpo`; a list or mapping already stopped on `main`, with a `TypeError` traceback instead of the by-name message), and an Unsloth notebook whose `CPOTrainer` does not set `loss_type="simpo"` on its own call or its `args=` config, or whose `args=` cannot be read statically (they loaded as `task: sft`). Migrations whose result changes: LLaMA-Factory `pref_loss: orpo` / `simpo` under a stage other than `dpo` now keeps that stage's task, as LLaMA-Factory itself does (they loaded as `orpo` / `simpo`); and because `SFTConfig` is now in the config table, a notebook with a later `SFTTrainer(args=SFTConfig(...))` after another trainer migrates as `sft` (the last `*Config` call decides, as it already did for the other config classes). The axolotl migrator was settled by #1254.

- **IP-literal parsing depended on the platform and the interpreter (#1550 by @qwist1233-cpu in #1562).** On Windows the C library refused `0xffffffff`, `4294967295` and `0377.0377.0377.0377`, so those spellings of `255.255.255.255` passed the outbound guard; 6to4, NAT64 and Teredo literals were classified only by the interpreter's `ipaddress` tables. `net_guard` now parses IPv4 spellings in pure Python, unwraps the embedded IPv4 address and classifies it too, and `refuse_private_ip_literal` parses once.

- **A judge that is down no longer costs 3 s of backoff for every row (#1522
  by @ybai08 in #1563).** Each judged row retried on its own, so `soup eval
  judge` over 10,000 rows slept for more than 8 hours against a host that
  never answered, and `soup data from-traces --judge` did the same and then
  wrote an empty file with exit 0. An evaluator now counts the requests in a
  row that spent their retries unanswered and raises `JudgeDownError`, a
  `JudgeUnavailableError`, at the third. `soup eval judge` stops there, prints
  one message naming the judge URL and how many items were not judged, shows
  the scores from before the outage without saving them, and exits 1;
  `from-traces --judge` exits 1 and writes nothing; `soup data best-of-n` and
  `soup ship --task-mode pairwise` say how many prompts or pairs were left.
  One reply resets the count, and a 4xx or an unusable reply counts as a
  reply. A judge-ranked training run is unchanged: it keeps asking and resumes
  on the first reply.

- Honor configured optimizer, schedule, warmup, weight decay, clipping and effective batch settings across PRM, MoLE, PPO and unlearning; keep unlearn loss and sample counts on their original scale.
  Unlearn now tokenizes forget and retain rows to `data.max_length` (default 2048) instead of
  a hard-coded 256, so existing unlearn configs use more memory and run longer; set
  `data.max_length: 256` to keep the old limit. On unlearn, `batch_size: auto` now resolves
  to 1 rather than to a memory estimate, so the same config makes the same number of
  optimizer updates on any card. (#805 by @Oxygen56 in #1564)

- **The provenance and compliance artifacts record what was measured, and only that (#1446 by @Srinivasan8888 in #1569).** `soup attest emit --invocation` now reaches the SLSA statement, at `buildDefinition.externalParameters.invocation`, unsigned and signed; it was stored under a key the builder never read. Annex XI and XII write `not measured` for training compute, energy and CO2 when nothing was measured (`soup train --annex-xi` without `--track-energy`, and FLOPs always, which Soup does not measure), instead of `0` and `0.000 kWh` / `0.000 kg` presented as measurements. The CycloneDX 1.6 ML-BOM has an RFC 4122 `serialNumber`, writes a known SPDX id in its canonical spelling (`apache-2.0` → `Apache-2.0`), an SPDX expression under `expression`, and any other license name under `license.name`; the SPDX 2.3 document does the same in `licenseConcluded` / `licenseDeclared` (other names become a `LicenseRef-` with `hasExtractedLicensingInfos`), and now states that the training data is a build dependency of the model rather than the reverse. The SR 11-7 reproducibility receipt fills `driver_version` on a CUDA host. The CycloneDX 1.6 schema set is vendored under `tests/fixtures/cyclonedx/` and the document is validated against it.

- **`soup expect` names at most ten unknown suite keys in a refusal, then counts the rest (by @Srinivasan8888 in #1570).** The refusals #1461 and #1499 added for a key beside `name` / `args` and for a key beside `expectations:` named every offending key, so a suite under the 1 MiB cap with tens of thousands of unknown keys produced a message nearly as large as the file. The first ten are named and the remainder reported as `and N more`; messages with ten keys or fewer are unchanged.

- **Every documented `soup` invocation names a real command path and real flags, and a guard keeps it so (#1465 by @Srinivasan8888 in #1578).** Five documented commands named a flag or command path the CLI refuses and one recipe description named an option no command has: `soup loop watch --max-iter` is `--max-iterations`; `soup serve ... --spec-tokens` is `--num-speculative-tokens`; `soup --verbose eval --model ...` is `soup eval benchmark ...`; `--no-telemetry` is a global option and goes before the command (`soup --no-telemetry train ...`); `soup adapters blame` has no `--live`; the `online-dpo-smollm2-135m` description points at `training.online_dpo_judge`. Sixteen more documented lines were missing a required option or carried an extra argument (`soup data pii` / `decontaminate` / `augment` take `--input`; `soup infer` needs `--output`; `soup eval custom` needs `--model` and `soup eval human` both `--model-a` / `--model-b`; `soup serve --reasoning-parser` needs `--model`; a second adapter needs its own `--adapters`; `soup runs list` is `soup runs`; `soup recipes search --tag mlx` is `soup recipes search mlx`) and are corrected. `docs/data.md` now says that `soup data decontaminate --benchmarks` records labels only and that `--benchmark-file` is what filters, since no benchmark corpora are bundled; its examples pass one. A test walks every `soup ...` line (aligned columns, `VAR=value` prefixes and backslash continuations included) and every `` `soup ...` `` span in `docs/*.md` and `README.md`, plus every recipe description, against the live Click tree: a flag or command path that does not exist, or a subcommand a group does not have, fails CI. Whether a line also carries every required option is not checked. (The standalone sweep config was #1475.)

- **A `training.prm_reward` Hub id is downloaded before training, and a PRM without a reward head is refused before the model loads (#1466 by @Srinivasan8888 in #1580).** `prm_reward` is documented as a path or a Hub id, but the PRM's reward head was read only from a local directory, so a GRPO run given `org/name` loaded the tokenizer and the model and then died at the first reward call with "path is not a directory". The id (optionally `org/name@revision`) is now resolved through Soup's Hub download wrapper (`utils.hubs.snapshot_download`, which runs the repo-id, revision and namespace-pin checks first), restricted to weights, tokenizer files and custom-code modules, and the head is read from that snapshot when the reward is built; a repo that is not a Soup-trained PRM, a download failure, or a value that is neither an existing directory nor a repo id each stop with one message naming the value, before any training. Existing local paths keep their containment check and now also fail early when they carry no reward head.

- **`soup migrate --from unsloth` reads a notebook that carries IPython magics (#1579 by @Srinivasan8888 in #1583).** Every code cell was concatenated and parsed as one Python module, so the first `!pip install ...` or `%%capture` line made the whole notebook a `SyntaxError` with no cell and no line, and every official Unsloth notebook starts with such a cell. A `%%` cell magic whose body IPython runs as Python (`%%capture`, `%%time`, `%%timeit`, `%%prun`) now loses only its header and its body is read; any other cell magic (`%%bash`, `%%writefile`, ...) is skipped whole. A line whose first non-blank character is `!` or `%` is treated as a magic only at statement level, so a `%` operator or a `!=` at the start of a continuation line is untouched, and a magic that is the only statement of an `if` / `for` / `with` body leaves a valid block behind. A cell that parses today is read byte for byte. A real syntax error is now reported with its code cell and line.

- **With `training.expand_layers` set, `training.freeze_trainable_layers` must now equal it, and any other value fails at load (#1409 by @dchaudhari7177 in #1586).**
  The field was read only for its sign: `1`, `2` and `4` all trained just the
  appended blocks, and `0` or a negative value silently trained the whole
  expanded model. It now has one meaning, freeze the original model and train
  only the appended blocks, so a config with `expand_layers: N` and a
  `freeze_trainable_layers` of `0`, a negative value or a different positive
  value is refused with both fields named and `Set freeze_trainable_layers: N.`
  Before, with `lora.r: 0`, `0` and negative values trained every layer and any
  positive value trained the appended blocks; under LoRA the value changed
  nothing (#1326). `expand_layers: N` with `freeze_trainable_layers: N` trains
  exactly what it did.

- `training.moe_lora: true` now adapts the routed experts of every fused-expert MoE on transformers 5 (by module names alone Mixtral and MiniMax adapted attention only, Qwen2-MoE attention plus its shared expert, Qwen3.5-MoE the shared expert only; Qwen3-MoE, OLMoE, DeepSeek-V3 and GLM-4 were reached only through peft's own name conversion): the fused `mlp.experts.gate_up_proj` / `down_proj` tensors are handed to peft as `target_parameters`, the module targets `auto` resolved are kept beside them (an explicit `lora.target_modules` list is kept too, where `moe_lora` used to replace it), and the console line names the parameters. Because peft's `ParamWrapper` refuses dropout, `moe_lora` with a non-zero `lora.dropout` is now refused on Mixtral and MiniMax too (#1421 by @Srinivasan8888 in #1590).
- `training.moe_expert_quant` on a fused-expert model is refused at setup, naming the architecture and the fused parameters, instead of printing `applied to 0 expert Linear block(s)` and training in full precision; matching no expert at all is refused the same way (#1421 by @Srinivasan8888 in #1590).

- A `task: grpo` run with `training.eval_steps` now records the held-out reward (`eval_reward`, higher is better) as its validation metric: the live panel shows a `Val reward` row, the tracker's metrics table gains a `val_reward` column (added to existing databases in place), and the training event stream carries `val_reward`. TRL's `eval_loss` on grpo, the policy objective, is no longer stored or shown as `val_loss`; every other task keeps `eval_loss` / `val_loss` unchanged. The per-task choice lives in `VALIDATION_METRICS` in `utils/eval_schedule.py` (#1389 by @Srinivasan8888 in #1592).

- **The pattern used to match HTTP 401/403 errors in the `format_friendly_error` function in `errors.py` no longer backtracks quadratically if handed an exception which contains the characters "HTTP" followed by whitespace with no status code (#1367 by @johncoplin in #1594).** The old pattern had two places in which whitespace could go if the separator was missing from an exception, meaning the function would have to try every possible combination of whitespace before being able to print the message for unmatched errors. This pattern has now been replaced with one that only has two spots for whitespace if the separator exists, preventing quadratic backtracking and meaning error-messages print faster overall for these kinds of exceptions.

- Vision and audio rows loaded through a streaming `data.interleave`, a remote URI or a Hugging Face Hub dataset (train and `validation` splits) now resolve their `image` / `audio` paths the same way as rows from a local file: against `data.image_dir` / `data.audio_dir` when set, otherwise against the directory of the local data file the row came from, skipping a row whose file lies outside that directory with the usual "rows skipped" warning. These load paths used to keep the value as written, so a relative path was opened against the working directory and `data.image_dir` / `data.audio_dir` had no effect. This is stricter for a remote URI or a Hub dataset whose rows hold file paths and that sets neither directory: there is no local directory to resolve them against, so the load now stops with a message naming the setting (`data.image_dir: .` resolves them under the working directory, skipping paths outside it). An image the Hub dataset has already decoded is passed on as before. On every load path, in `soup infer --task asr` and in `soup data inspect`, a value naming a network share, a device or an extended-length path is skipped without being looked up, and so are a `bytes` value and, on Windows, a DOS device name such as `nul`; on Linux and macOS a value beginning with `//` now counts as a network path too. The SFT vision and audio paths and the ASR / TTS audio loader read a file only when it is a regular file, so a directory, FIFO or device is reported instead of being read (#1599).

- Vision SFT accepts decoded PIL images from Hugging Face Hub datasets and Parquet image structs ({"bytes", "path"}) without crashing. Path structs follow the same directory resolution and path containment rules as file path strings (#1597 by @Nari-Ab in #1601).

- `soup data inspect` resolves relative vision image paths against the data file's directory (or `--image-dir`), the same way training does, and reports an image training would refuse as outside the image directory instead of counting it as found (#1598 by @commit-msuyog in #1604).

- `soup data generate --template preference`: deduplicate generated preference rows (DPO and KTO formats) against `--dedup-with` and within-run duplicates using a stable key, rather than bypassing deduplication (part of #1448 by @Nari-Ab in #1605).

- `soup merge --save-format 4bit_forced`: pass `llm_int8_skip_modules=[]` to `BitsAndBytesConfig` so transformers quantizes all linear modules including `lm_head` rather than silently skipping them, and refuse models with tied embeddings (`tie_word_embeddings=True`) by name before loading weights (part of #1448 by @Nari-Ab in #1606).

- `soup serve`: wire output constraint logits processors and reasoning parser into `_stream_response` so streaming chat completions (`stream: true`) apply structured generation constraints and strip `<think>` blocks before emitting SSE events (part of #1448 by @Nari-Ab in #1607).

- **GRPO variant loss kernels now guard against exponential ratio overflow and NaN backward gradients.**
  Importance weight ratios and sequence ratios in `grpo_variants.py` are clamped from
  above before exponentiation using an advantage-aware bound, and padding tokens use boolean
  masking, preventing infinite terms from producing NaN parameter gradients under policy divergence.
  (#1609 by @harshitthek in #1610)

- **A pinned staging arena whose base is off a 4096-byte boundary no longer
  turns direct I/O off for the whole disk-tier store (#1531 by @ybai08 in
  #1621).** `plan_pinned_arenas` aligns the staging regions relative to their
  arena, and torch's caching host allocator can hand back an arena at a
  multiple of only 512 bytes; one misaligned region then sent every layer
  through the page cache, 2.4x slower cold (#974), with a `logger.warning`
  as the only sign. `AsyncDiskSource` now reads each arena's base, shifts the
  placements inside the arena when its slack allows, and otherwise takes the
  next power of two up; `pinned_bytes` counts the arenas actually held. If the
  staging is misaligned anyway, the page-cache fallback is printed on the
  console as well as logged.

- `data.add_new_tokens` / `data.new_special_tokens` are refused at config load wherever no trainer setup adds them to the tokenizer, naming what does not apply them: every `backend: unsloth` setup (`tts` with `new_special_tokens` included; SFT's vision and audio paths excepted), `backend: mlx`, `training.stream_layers: true`, and the tasks whose trainers never call the expansion helper (`pretrain`, `embedding`, `ppo`, `reward_model`, `prm`, `classifier`, `reranker`, `cross_encoder`, `unlearn`, `moe_lora_routing`, `distill`, `asr`). They used to load there and be dropped, so the tokens never reached the tokenizer. On `backend: transformers` for `sft`, `dpo`, `kto`, `orpo`, `ipo`, `simpo`, `bco`, `grpo`, `online_dpo`, `preference` and `tts` nothing changes (#1358 by @Srinivasan8888 in #1622).

- **QuEST continuation now preserves the artifact's measured calibration.** Continuing training from a local QuEST artifact previously recalibrated activation clips on the first 32 new training rows, changing the route represented by the checkpoint. Soup now restores the saved clipping table and route. It also refuses a non-local artifact and a `config.json` declaration that differs from the mandatory metadata sidecar. (#674 by @Shutaru in #1624).

- **The grad-accum advisory measures VRAM pressure against the card the run uses, not a fixed 24 GB (#1620).** `training.grad_accum_auto_tune` built its monitor with a hard-coded `total_vram_gb=24.0`, so on every card smaller than about 24 GB (all common 4, 8, 12, 16 and 20 GB cards) the advisory could never fire, while a 40-80 GB card fired at ~22 GB of peak use (about 28% of an H100). The callback now reads the CUDA device's real total memory on the first advisory call (an explicit `grad_accum_total_vram_gb` still overrides it), reads the peak from the same device, prints the measured numbers in the advisory (`peak 7.8 of 8.0 GB (97%)`), and the live memory panel reads its total from the same device instead of always device 0 (#1620 by @vaputa in #1625).

- **`SOUP_LAYER_STREAM_STRIPE_DIRS` refusals show the entry as typed (#1611).** Every refusal formatted the entry with `repr`, so each backslash of a Windows path was doubled, and copying the entry out of the message gave a spelling that does not work. Entries are now shown in single quotes with single backslashes. Control characters are still escaped as `\xNN`, so they never reach the terminal raw. The same applies to the Windows owner-only folder errors and the disk pre-flight message (#1611 by @vaputa in #1626).

- **`training.auto_mixed_precision` is now refused at config load on every task whose trainer never reads it (#1618).** The field loaded on all 23 tasks while only the SFT trainer applied it (inherited by `tts`), so a DPO, GRPO, KTO or any other non-SFT run carrying the key trained at its trainer's default bf16/fp16 choice with nothing said, and `soup autopilot` wrote the key into generated non-SFT configs. Setting it on any task outside `sft`/`tts` now fails config validation naming the task and the way out, `generate_config` no longer emits it there, the GRPO `grpo_fp16` + `auto_mixed_precision` refusal now says `grpo_fp16` is the only dtype pick on that task, and the field description and GRPO docstring no longer promise a picker that never ran. **This is a breaking change:** a config that sets the field on a task other than `sft`/`tts` used to load and is now refused — removing the key changes nothing about how such a run trains (#1618 by @vaputa in #1627).

- Layer streaming validates `SOUP_LAYER_STREAM_STRIPE_DIRS` early during `soup train` startup and `--dry-run`, refusing missing folders, bad syntax, and same-volume stripes before loading datasets or prompting, and `soup doctor` now reports stripe roots under System Resources (#1613 by @Nari-Ab in #1628).

- Layer streaming copies a Hugging Face snapshot only from the repo's own `blobs` directory
  and the marked shared store: a `blobs` directory that is itself a link (a symlink, or a
  junction on Windows) is now refused with a message naming it instead of being followed,
  and a junction inside a snapshot directory is refused like a directory symlink. The links
  huggingface_hub creates, including those into the shared store of 1.32 or later, work as
  before, and so does a cache or repo folder that was moved as a whole and linked (#1629).

- `info_rm` reward-hacking detection no longer votes on a step whose median split leaves either half constant — a two-valued reward (e.g. `reward_fn: accuracy`'s 0/1), or a continuous reward where half the batch lands on the same value. Previously the pooled-variance floor collapsed to `1e-9` and a first step near a 50% success rate recorded a phantom separation of ~31,623 as the baseline, so later, healthier steps read as a collapse and could halt the run or mutate β. Those steps now cast no vote and log one warning naming the cause. A step where every reward is identical still votes once a baseline exists, so a genuinely collapsed continuous reward reads HACK as before. The pooled-variance floor is also tied to the observed reward range instead of a bare `1e-9`. Detector logs and `mitigation_log.jsonl` from runs before this version are not comparable — use `reward_hack_detector: rm_ensemble` or the length/repetition signals on two-valued rewards (#1438 by @Momoyeyu in #1630).

- **Wire DPO-family reference adapter regeneration on LoRA (#1345 by @kok-o in #1636).** `dpo_ref_regen_epochs` periodically syncs reference model weights on DPO, IPO, and preference training. Under LoRA (where TRL creates a frozen `ref` adapter and leaves `trainer.ref_model` as `None`), in-place copies active `.default.` adapter parameters into `.ref.` parameters under `torch.no_grad()` on every Nth epoch. Full fine-tuning (`lora.r: 0`) remains refused at config load time.

- **A fresh `pip install 'soup-cli[train]'` could no longer run `task: grpo` or
  `task: online_dpo`: the trainer import stopped with `ModuleNotFoundError: No
  module named 'requests'` (#1637).** trl imports `requests` at module scope in
  `trl/generation/vllm_client.py`, and both `trl.trainer.grpo_trainer` and
  `trl.experimental.online_dpo` import that module unconditionally, but trl
  lists `requests` only under its optional `vllm` extra. A training install
  received the package indirectly through `datasets`, until datasets 5.1.0
  (2026-10-05) dropped that dependency. Environments created before that date
  keep working because pip does not remove an installed package; environments
  created after it were missing `requests`. On GRPO the failure surfaced as
  `RuntimeError: Failed to import trl.trainer.grpo_trainer`, and on online DPO
  as `installed trl 0.29.1 does not provide 'OnlineDPOConfig'`, with the real
  cause at the end of the message. The `[train]` extra now declares
  `requests>=2.32.2` itself — the floor `datasets` carried, so an existing
  environment resolves to the same set. Trainers that do not load those two
  trl modules were not affected: on trl 0.29.1 the SFT, DPO, KTO, ORPO, CPO,
  BCO, PPO and reward trainer classes import without `requests`. On an
  installed release without this change, `pip install "requests>=2.32.2"`
  restores both tasks. `tests/test_train_extra_declares_requests.py` pins the
  declaration.

- **Step PyTorch profiler on each training step so `soup train --profile` writes traces (#1448 by @kok-o in #1639).** `soup train --profile` previously announced writing a trace upon completion but left `<output_dir>/profiles/` empty because `torch.profiler.profile` schedule state only advances when `profiler.step()` is invoked per step. Added `ProfileStepCallback` lazily subclassing HuggingFace `TrainerCallback` via PEP 562 to step the profiler on `on_step_end` without importing `transformers` at CLI startup, and wired it into `soup_cli.commands.train` when profiling is enabled.

- `soup sweep` now marks an arm as failed when trainer construction or setup raises,
  preventing stuck `running` rows in the experiment tracker (#1464 by @deximple in #1640)

- **A sink-based ratchet for outbound HTTP calls across `src/soup_cli` (#1547 by @Areen-09 in #1646).**
  An AST ratchet verifies every outbound HTTP call (`httpx` methods and clients,
  `urllib.request` openers and callers, and `requests` methods) against a declared
  ledger table of known gates and justifications, in addition to shape-based checks,
  with fail-closed bidirectional enforcement. `validate_electricity_map_endpoint` now
  routes through `net_guard.LOOPBACK_HOSTS` and `refuse_private_ip_literal`: it refuses
  abbreviated IPv4 forms (such as `10.1`) and RFC 6598 carrier-grade NAT (`100.64.0.0/10`),
  while loopback addresses such as `https://127.0.0.2/` and `https://[::ffff:127.0.0.1]/`
  are now accepted under the shared loopback policy.

- `soup diagnose`, `soup train --diagnose-gate` and `soup eval unlearning`
  print text they did not write through the shared `for_terminal` helper
  instead of `escape()` alone, so ESC, DEL and the C1 control range
  (U+0080-U+009F) are stripped before the text reaches the terminal: the
  evidence line of every score or metric (from `--evidence` /
  `--diagnose-gate` JSON or a live run), the adapter name and run id in the
  table titles and the overall panel, the `--output` and `--badge` paths,
  the `--attach-to-registry` id, and the exception text in their error
  lines. The `--diagnose-gate failed` line of `soup train` printed its
  exception text as Rich markup: an evidence file whose refused `verdict`
  contained `[/]` ended the command with a `MarkupError` traceback instead
  of the message (#1647).

- **`training.freeze_layers` / `training.freeze_ratio` are refused at load on the `sft` and `tts` paths that never applied them (#1581 by @abdulwaarith0 in #1649).** #1532 refused the two fields on every other task; inside `sft` and `tts` they are read only on the text path with `backend: transformers` and no layer streaming. With `backend: mlx`, `backend: unsloth`, `modality: vision`, `modality: audio` or `training.stream_layers: true` (on `tts`, `backend: unsloth`) the config loaded, nothing was frozen and nothing was printed: those setups always attach an adapter, so the run trained every layer. **This is a breaking change: a config that sets either field on one of those paths loaded before and now stops at load**, with a message naming the field, every setting that leaves the text path and what to change (for example `Use backend: transformers, or remove the key.`), and `soup doctor --config` exits 2. Removing the key changes nothing about what trains. Text configs on `backend: transformers`, and dumped configs (which write both fields as `null`), are unchanged.

- **A host written only as numbers is an IPv4 address or it is refused (#1650, follow-up to #1562).** `net_guard` reads IPv4 spellings with one strict grammar on every platform, and text that looks like a number but does not fit it (a number too large for its position such as `4294967296` or `10.0.0.256`, a `0x` with no hex digit after it, an `8` in an octal part, a fifth part) was treated as a hostname, although C libraries do not all read such text the same way. The outbound endpoint checks (`--api-base`, judge URLs, `online_dpo_judge`, the vLLM URL, webhooks, the OTLP endpoint, the Langfuse host, the telemetry endpoint, the Web UI chat endpoint) now refuse a host made of numeric labels only that is not a valid IPv4 address; where the check reports its own message it reads `an all-numeric host that is not a valid IPv4 address is not allowed`. A hostname with any label that is not a number, loopback, and every numeric spelling that is a valid address are unaffected. A decimal number longer than `int()` converts (4300 digits) is now "not an address" instead of an uncaught `ValueError`. A host that is a single space counts as non-public on every platform, as it did on Windows.

- **The `soup train` VRAM pre-flight skipped `training.quantization: mxfp4`
  (#1631 by @ybai08 in #1655).** An MXFP4 base is dequantized on load and
  trains in bf16, but the pre-flight built its estimate only for `none`,
  `4bit` and `8bit`, so an `mxfp4` run was never checked. It is now checked
  against the same estimate as a `quantization: none` run, about 2 bytes per
  parameter, and a run that does not fit is refused with one extra sentence
  saying the estimate prices the bf16 model. This refuses `mxfp4` configs that
  used to start on a card too small for the model; `--allow-oom-attempt`
  launches them anyway.

- **Budget `mxfp4` bases as dequantized bf16 in autopilot (#1653 by @kok-o in #1656).** `decide_quantization` now enforces the VRAM budget for `mxfp4` pre-quantized models which dequantize on load to bf16 (2 bytes per parameter), preventing generation of configurations that cannot load on the target GPU budget.

- **Handle `typer.Exit` cleanly in `soup train` auto-eval custom tasks (#1641 by @kok-o in #1658).** Catch `typer.Exit` when executing auto-eval custom tasks to avoid printing an unhandled traceback and reporting the run as failed.

- **Keep trainable adapter dtype across checkpoint resume for DPO, IPO, GRPO, and Reward trainers (#1573 by @kok-o in #1660).** Apply `keep_trainable_dtype_on_resume` in `DPOTrainerWrapper`, `IPOTrainerWrapper`, `GRPOTrainerWrapper`, and `RewardModelTrainerWrapper` so that resuming a run preserves the trainable adapter's original dtype instead of upcasting to fp32.

- **Auto-evaluation after `soup train` follows the run's `--trust-remote-code` flag
  (#1644 by @Areen-09 in #1662).** `soup train --trust-remote-code` with
  `eval.auto_eval: true` now passes the flag to the benchmark and custom
  evaluation of the model the run just saved, instead of always `False`, which
  made a base that needs custom code train and then fail its own evaluation.
  Without the flag both stay off, as before. The opt-in is CLI only: a
  `soup.yaml` with `trust_remote_code` (top level or under `eval:`) is refused
  as an unknown key.

- Re-check that a `SOUP_LAYER_STREAM_STRIPE_DIRS` folder has not been renamed and replaced by a link before every per-layer write and the marker write of a layer-stream shard, instead of only once before the shard starts; and refuse a directory junction as a stripe root while still accepting a genuine Windows volume mount point, distinguishing the two by their link target rather than their shared reparse tag (#1614 by @SID-6921 in #1663).

- Layer streaming reuses its regular-file copy of a Hugging Face snapshot only while that
  snapshot would still be copied. When the repo's `blobs` directory has become a link (a
  symlink, or a junction on Windows) since the copy was made, or the snapshot has gained a
  linked directory or a file whose link leaves the blob store, the run now stops with the
  same message as a first copy instead of returning the earlier copy. That copy is left in
  place, and an unchanged snapshot is reused without copying, as before (#1666).

- Layer streaming keeps the regular-file copy of a local Hugging Face snapshot path
  (`.../models--org--name/snapshots/<commit>`) in the cache slot of the Hub id `org/name`
  only when that path is the folder the Hub id resolves to in the active Hugging Face cache
  (a cache that was moved and linked counts as the same folder). A snapshot folder anywhere
  else now gets a slot of its own, named after its resolved location, so a copy taken from
  one folder is never reused for another folder or for the Hub id; such a folder is copied
  once more on its next run (#1667).

- **`mini_common_sense` no longer sits at a 1.000 ceiling (#1192 by @AaronProbha18 in #1673).**
  The suite grows from 24 to 40 rows with 16 four-option reasoning questions;
  Qwen2.5-1.5B-Instruct measures 0.850 against 0.275 on SmolLM2-135M-Instruct,
  which is chance level for the suite. The answer key no longer favours one
  letter, so a model that always answers "B" drops from 0.583 to that same
  0.275, and the bundled scorer provenance revision moves to 5 so pre-change
  baselines are not silently compared on the new scale.

- `soup local-rl train --backend` prints a note that the option has no effect on training; it is validated but not rendered into the scheduler scaffold (part of #1448 by @Nick-800 in #1678).

- `soup export --registry-id` now records the exported files for `onnx`,
  `tensorrt`, `awq`, `gptq`, `torchao`, `gguf-ud`, `bitnet` and `tq1_0` (every
  shard or engine file, named in the success line). `--deploy ollama` now works
  with `gguf-ud`, `bitnet` and `tq1_0` and is refused before the export starts
  for the formats Ollama cannot load (part of #1448 by @Nick-800 in #1679).

- **`task: ppo` saves its checkpoints again on transformers 5.19.0.** trl's experimental `PPOTrainer` is a `transformers.Trainer` subclass that never runs `Trainer.__init__`, and transformers 5.19.0 made `Trainer.save_model` read an attribute only `Trainer.__init__` assigns (`is_distributed_loading_by_transformers`), so every PPO run stopped at its first checkpoint with `AttributeError: 'PPOTrainer' object has no attribute 'is_distributed_loading_by_transformers'`. After building the PPO trainer Soup now gives it each attribute the installed `Trainer.__init__` assigns that the trainer lacks and `Trainer`'s save path reads (that one, `is_fsdp_xla_v1_enabled` and `hp_name`), with the value `Trainer.__init__` would have assigned; an attribute the trainer already has is left alone, and nothing changes on a transformers that never had the attribute. No other task is affected: every other trl trainer Soup builds runs `Trainer.__init__`. Reported by @Srinivasan8888 in #1675 (#1682).

- `soup diagnose --attach-to-registry` and `soup eval unlearning --attach-to-registry`
  attach the report. Both called the registry helper with arguments it does not take, so
  the attach raised a `TypeError` on every run, the report was never linked to the entry,
  and the command printed the error as a warning and exited 0. The `--output` file is now
  stored on the entry (as `diagnose_report` / `eval_results`), and a requested attach is
  part of the command's success, as it already is for `soup bom emit`, `soup attest emit`
  and `soup eval custom`: a failed attach (unknown or ambiguous entry, unreadable
  registry) prints an error and exits 1 with the report left on disk, and
  `--attach-to-registry` without `--output` is refused before anything runs (exit 3 for
  `soup diagnose`, exit 2 for `soup eval unlearning`) instead of being skipped with a
  warning (#1683).

- `soup steer train --registry-id` and `soup edit set --registry-id` record the result on
  the registry entry. Both passed their output directory to the registry, which stores one
  hashed file per artifact and refuses a directory, so every run printed `Could not attach
  to Registry: artifact not found: <dir>` as a warning and exited 0 with nothing recorded.
  `soup steer train` now attaches the vector file (`steering_vector.safetensors`) as
  `steering_vector`, and a vector saved outside `./steering/<name>` is found by name through
  the entry; `soup edit set` attaches the weight file(s) `save_pretrained` wrote as
  `edited_model`, or `grace_codebook.json` as `grace_codebook`. A requested attach is part
  of the command's success, as it already is for `soup bom emit`, `soup attest emit` and
  `soup eval custom`: a failed attach (unknown or ambiguous entry, unreadable registry, no
  weight file) prints an error and exits 1 with the result left on disk, and `soup edit set
  --registry-id` without `--output` is refused before the model is loaded (exit 2) instead
  of skipping the attach without a message (#1686).

- The Web UI's New Training page shows the run's log. The "Training Logs" card and the
  function that streams `/api/train/logs` into it were both there, but nothing called the
  function, so starting a run showed `Training started! PID: ...` and never a line of
  output. The page now opens the card and follows the log when a run is started from it,
  and when it is opened or reloaded while a run is in progress (the server replays the log
  from its first line), including for a run that exits at once, whose output is the error
  to read. The progress card stays hidden: no route reports a total step count for the run
  the UI started, so its bar could not move (#1687).

- `soup train --gpus N` on Windows waits for the `accelerate launch` it hands the run to and
  exits with the launcher's exit code. The hand-over used `os.execvp` on every platform. On
  Linux and macOS that replaces the process, so the launcher's status is the command's; on
  Windows it starts the launcher as a new process and ends the calling one with status 0 at
  once, so the shell got its prompt back while the run was still going and `soup train
  --gpus 2 && next-step` ran `next-step` whatever the launcher returned later. Windows now
  runs the launcher as a child (list argv, no shell), looked up in the `PATH` directories
  only, and Ctrl+C waits for the launcher's own shutdown. Linux and macOS are unchanged; an
  end-to-end test with a stand-in `accelerate` now checks the exit code on whichever
  platform the suite runs on (#1688).

- **DPO loads a vision-language wrapper checkpoint when `modality: vision` is set (#1393 by @Srinivasan8888 in #1707).** `task: dpo` always built `AutoModelForCausalLM`, so a base published as a vision-language wrapper (MiniMax-M3, `model_type: minimax_m3_vl`) could not load at all, while SFT trained its language tower through its vision path. With `modality: vision`, DPO now loads `AutoModelForImageTextToText` with the model's processor, which trl's `DPOTrainer` takes as its processing class for text-only preference rows, and the LoRA targets resolve to the language tower as before; `soup recipes verify` builds the same class, and `task: preference` with `preference_loss: dpo` follows it. `backend: unsloth` does not read `modality` for DPO and is refused at config load for that combination. The text path is unchanged.

- **Unsloth migration reads the final trainer's config**: task, output directory and hyperparameters come from the config passed to the last trainer, whether inline, named or positional. SFT warm-ups and unused later configs no longer overwrite stage settings. Warm-up `num_train_epochs` and other settings no longer leak into the final stage; this removes inherited `training.epochs: 2` from four public Qwen3 4B GRPO notebook variants (including AMD, Kaggle and Hugging Face Course). Earlier-stage warnings and config-only notebook behavior are preserved. (#1603 by @akkupratap323 in #1711)

- **`soup train` refuses `use_liger` / `use_flash_attn` under `stream_layers: true` (#1692 by @Faisal-Fayaz in #1716).** Both switches loaded, trained and said nothing, because they are read only by the resident setup: Liger patches the MLP forward in `_setup_transformers` and flash-attn is handed to `from_pretrained` as `attn_implementation`. A streamed run goes to `_setup_streaming_transformers`, whose model is built by `AutoModelForCausalLM.from_config(...)` with no attention argument and no Liger patch, and `_liger_applied` is never set, so `use_liger_kernel` misses `TrainingArguments` as well. Since these are the switches a user reaches for when a streamed run will not fit, silently dropping them was the worst of the outcomes, so `stream_layers` now names both fields in its conflict list at config load. Nothing is wired: this is the refusal half, and each switch is offered again once it is shown to engage on a streamed model. `use_cut_ce` has the same shape under streaming and stays on #1206, deliberately still loading here. The streamed preference tasks (`dpo`/`kto`/`orpo`/`simpo`) are unaffected: they already refuse both fields on their own terms, so no branch was needed for them.

- **`soup eval benchmark`: do not store an unmeasured result as zero.**
  Missing or scoreless requested results are skipped, named and counted; a genuine
  numeric zero is retained and booleans are not scores. The saved banner counts
  persisted results, and an all-skipped invocation exits 1. Groups require their
  own numeric entry; sub-task rows are not saved implicitly. (#1718 by @taljeon in #1722)

- **`training.moe_lora: true` with `lora.use_vera: true` is refused on a fused-expert MoE (#1197 by @Srinivasan8888 in #1723).** VeRA wraps `nn.Linear` modules only and has no `target_parameters` path, so on a transformers-5 MoE whose routed experts are 3-D tensors (Qwen3-MoE, Qwen2-MoE, Mixtral, MiniMax) the flag adapted the attention projections exactly as VeRA without it did, while printing the ScatterMoE targeting line. The combination now stops at setup, before the dropout rule, naming both flags, the fused parameters and the two ways out (`use_vera: false` to adapt the experts with LoRA, or drop `moe_lora`). A dense base and a checkpoint that keeps one `nn.Linear` per expert are not refused.

- Layer streaming now reads source safetensors headers during the disk pre-flight to charge each stripe drive the exact bytes of the decoder layers it receives, allocating non-decoder weights to the primary root at streaming dtype rather than splitting the shard estimate evenly (#1612 by @Nari-Ab in #1726).

- **`soup eval human`: reject non-object JSONL rows before loading either model.**
  Report the physical line number and expected JSON object instead of passing a
  string or array through to a later TypeError. Object rows are unchanged.
  (#1725 by @taljeon in #1727)

- **A QuEST resume from a format-v1 checkpoint no longer reads and re-hashes the whole local base a third time (#1199 by @Faisal-Fayaz in #1728).** `validate_resume_metadata` re-resolved `legacy_base_model` to compare it against the identity this run had already computed, so a first v1 resume read every selected file of a local base three times instead of twice — about 3.9 s per GB on the maintainer's measurement, more on a busy machine. The identity is already in `current["base_model"]` (the caller filled that field with it), so the extra pass rediscovered what it was handed. It now takes an optional pre-resolved identity and `SFTTrainerWrapper.train` passes the one it holds, which removes the pass. The value passed is the identity of `config.base` resolved before the load (`_quest_base_identity_before`), which is what the check compares against — **not** `current["base_model"]`, which on the continuation path is the artifact sidecar's recorded base and would make the check compare a value with itself. An identity that disagrees with the base still produces the same `QuEST original base model reference does not match current identity` refusal, and the default path for a caller that passes nothing is unchanged. Hub bases are unaffected either way, since their identity is the repo id; the pass that disappears is the one that cost something. Fix 1 of that issue (hash once and compare `(st_dev, st_ino, st_size, st_mtime_ns)` after the load) is deliberately **not** here: it would trade the content guarantee for a stat one, and the issue says to skip it if the stronger guarantee is wanted. Fix 3 already merged as #1588.

- **`training.rope_scaling_type` is refused at load where no setup applies it (#1697 by @ybai08 in #1730).** The field loaded on all 23 tasks and every path, but only the transformers setups of `sft`, `tts` and `pretrain` read it; everywhere else the run trained on the checkpoint's own RoPE and printed nothing. **This is a breaking change: a config that sets the field on one of these loaded before and now stops at load**: any of the other 20 tasks; `task: sft` with `backend: mlx`, `modality: vision`, `modality: audio` or `training.stream_layers: true`; and `sft`, `tts` or `pretrain` with `backend: unsloth`. The message names the field, the task or every setting that leaves the reading setup, and what to change (for example `Use backend: transformers, or remove the key.`); a `yarn_*` key set beside it is named too, since it is legal only with `rope_scaling_type: yarn`. `soup doctor --config` exits 2. Removing the key changes nothing about what trains. The `longcontext` template no longer suggests `# backend: unsloth`, which it would now be refused with. Text configs on `backend: transformers`, `pretrain` on `backend: transformers` with any modality, and dumped configs (which write the field as `null`) are unchanged.

- **Custom tool-call subset scoring: do not credit malformed provided arguments.**
  Values that fail to parse as a JSON object no longer earn empty-argument
  credit. Valid empty objects, omitted arguments, weights and the match threshold
  keep their existing treatment. (#1734 by @taljeon in #1737)

- Quant-check rounds score-drop float noise before comparing verdict thresholds,
  so exact two- and five-percentage-point drops remain MINOR and MAJOR respectively
  without changing scores or thresholds. (#1740 by @taljeon in #1743)

- **`training.quantization: fp8` stops with the reason on transformers below 5.17.0 (#1731 by @ybai08 in #1745).** Soup's declared floor is 5.16.1, and there the FP8 quantizer raises `AttributeError: 'NoneType' object has no attribute 'get'` in `update_tp_plan` when it loads a dense base, so an `fp8` run died inside transformers. The run now stops where the quantization config is built: `quantization: fp8 needs transformers >= 5.17.0, and 5.16.1 is installed`, with the cause and `pip install -U 'transformers>=5.17.0'`. Every other quantization mode is unchanged on 5.16.1. The Quant Menu row and the missing-class message said `>= 4.45`; both now say 5.17.0.

- **`soup train` suggests `backend: unsloth` only for a config that would load with it (#1736 by @ybai08 in #1746).** With unsloth installed, every run on `backend: transformers` was told `Tip: unsloth is installed. Add backend: unsloth ...`, including configs the loader refuses as soon as that line is added: a task with no unsloth setup, `freeze_layers`, `rope_scaling_type`, `moe_lora` recipes. 38 of the 184 shipped templates and recipes on transformers got a tip they would be refused with. The tip now validates the config again with the backend changed and is printed only if that succeeds, so a new backend refusal is covered without a list in `train.py`. Nothing else about the run changes, and a machine without unsloth validates nothing twice.

- **Interrupted content-changing layer re-shards can no longer leave an old cache index pointing at partially rewritten files.** Soup quarantines the old index before rewriting, reports unfinished re-shards and unindexed cache files, and keeps post-commit cleanup failures non-fatal. (#1616 by @Samearth17 in #1747)

- `deploy measure` delegates verdict classification to `eval.quant_check.classify_delta`, eliminating the duplicated ladder and rounding score-drop float noise so exact two- and five-percentage-point drops are now classified as MINOR and MAJOR respectively without changing scores or thresholds; caches written before this fix retain their stored verdict until cleared (#1744 by @kok-o in #1748).

- Close file descriptors before unlinking staged offline datasets on Windows so the original error is not masked by WinError 32 and no staging files are left behind (#1464 by @Guts1005 in #1753)

- **`soup <command> --help` shows argument descriptions again on fresh installs: click is bounded below 8.5 (#1789).** click 8.5.0 overwrites the help text typer gives each positional argument, so every positional argument in every `--help` page lost its description (`ADAPTER  TEXT  [required]` with nothing after it) on any install that resolved the newest click; option help was unaffected. Soup now declares `click>=8.0.0,<8.5`, which it imports directly but had left to typer to pull in, and lifting the bound would need typer 0.26 or newer.

### Security

- **`soup ui` no longer keeps its auth token in `sessionStorage` or on `window._authToken` (#1136 by @mohdUwaish59 in #1190).** The Web UI now holds the token in a closure in `app.js`, and every request goes through one `authFetch` wrapper that adds the `Authorization` header without exposing the token to other page script. The token is deliberately not persisted: a reload or a new tab asks for it again (paste the token `soup ui` printed), where before it was restored from `sessionStorage`. This narrows the gap between a future rendering mistake and a token disclosure; it does not stop script already running in the page, for which the Content-Security-Policy remains the defence. A fresh desktop browser can now sign in at all: `soup ui` opens its URL without a token, and on 0.75.0 through 0.75.2 nothing in the page asked for one, so every panel stayed at 401; the page now prompts on the first 401 and accepts the pasted token or the whole `Authorization: Bearer …` line. `authFetch` also refuses cross-origin URLs, so the header only ever goes to the UI's own origin. The chat request no longer sends an empty `Bearer ` header when no token is held.

## [0.75.2] - 2026-10-01

### Changed

- Outbound endpoint checks now refuse a private, link-local or reserved IP literal over `https` as well as `http`. This covers `soup data generate --api-base`, the vLLM provider and the commands that use it, judge URLs in `soup eval judge`, eval-gate suites, `soup ship --judge-model` / `eval.ship.judge_model` and `training.online_dpo_judge`, and the Web UI chat proxy. Loopback still works; address a LAN server by its hostname. `training.online_dpo_judge` is now checked when soup.yaml loads, not only when the trainer starts.

- `soup data generate --api-base` and the Web UI chat proxy no longer treat `http://0.0.0.0` as local; use `localhost` or `127.0.0.1`.

- `100.64.0.0/10` and `fec0::/10` literals are now treated as non-public by every check that already refused private addresses (webhooks, the OTLP endpoint, telemetry, hub endpoints over HTTP, and `soup ingest --pull`, where `--allow-private-host` admits them).

- Every check that refuses private addresses now also recognizes an IP literal written with non-ASCII digits or label separators, which an HTTP client folds to ASCII before it connects.

### Fixed

- `soup ship` checks `--judge-model` before it builds the base and tuned models; it used to load both and only then stop with a usage error.

- The Web UI chat proxy answers 400, not 500, for an endpoint URL that does not parse.

### Security

- Hardening of outbound endpoint validation — [GHSA-9f49-hqpf-c849](https://github.com/MakazhanAlpamys/Soup/security/advisories/GHSA-9f49-hqpf-c849). The advisory has the details and upgrade guidance.

## [0.75.1] - 2026-09-21

### Fixed

- **A layer-streamed LoRA adapter saved as ZERO tensors under peft 0.21.0 (#1005 in #1010).** peft 0.21 selects an adapter's tensors by the prefixes it reads off `model.named_modules()` and keeps the matching `state_dict()` entries, where 0.20 filtered by the `lora_` substring; the streaming wrapper's module and parameter names carried `.inner.` while its `state_dict()` keys were canonical, so `trainer.save_model()`, every `save_steps` checkpoint and `get_peft_model_state_dict()` returned nothing, and nothing raised. The wrapper now enumerates the inner layer's tree at its own prefix (`named_modules()` / `named_children()`, with `train()` / `apply()` / `_apply()` reaching the inner layer explicitly), so names and keys are one spelling — which also keeps transformers' weight-decay grouping, built from `named_children()`, from silently dropping every LoRA parameter into the no-decay group. **Any adapter saved with `stream_layers: true` on peft>=0.21 before this fix holds zero tensors (a 40-byte `adapter_model.safetensors`, no error at save time) and cannot be recovered — re-train it on a Soup with this fix, or with `peft<0.21`.** Naming only; nothing on the streamed forward path changes. Ten peft-version-independent tests pin names == keys against a resident LoRA model; the four tests that documented the old asymmetry are inverted rather than deleted. No `peft<0.21` pin.

- **A Ctrl+C arriving immediately after the Lambda controller started could still leave the instance running (#1073 in #1078).** `submit_lambda_run` spawned the controller outside the `try` that absorbs interrupts, so a SIGINT delivered between `Popen` returning and the waiting loop being entered — or one landing on the "still waiting" notice, which is written from the interrupt handler — unwound the parent while the controller was still terminating the paid instance in its `finally` block. The spawn and the notice now sit inside the guarded region: once the child exists, every path leads back to `proc.wait()` and the controller's exit code remains what the function returns. An interrupt raised before the child exists still propagates unchanged, and a `wait()` failure that is not an interrupt still surfaces rather than being retried.

- `soup runs clean` now accepts `--no-keep-weights` to delete whole non-best checkpoints; `--keep-weights` (the default) keeps weights on every supported Click version — on Click 8.1 it previously deleted whole checkpoints (#1057)

- **Cloud runs keep their outputs and their cleanup (#1058).** `soup train --cloud modal` now writes run outputs to the `soup-outputs` Modal volume and downloads them to the local output directory when the run ends (also after a failed run); previously checkpoints were lost with the container. `soup train --cloud lambda --cloud-submit`: pressing Ctrl+C now waits for the controller to terminate the Lambda instance; previously the controller was killed 0.25 s later and the instance could keep running (Linux/macOS).

### Security

- **Hardening across the CLI, the inference server, the Web UI, MCP execution, config parsing and `.can` handling — [GHSA-63h4-gvp4-r26g](https://github.com/MakazhanAlpamys/Soup/security/advisories/GHSA-63h4-gvp4-r26g).** Eight issues: Hugging Face credentials presented to a non-HF hub and `OPENAI_API_KEY` presented to any https judge host; `soup adapters verify --public-key` accepting a record whose backend is not ed25519; the Web UI rendering dataset and run content as HTML with no Content-Security-Policy; `soup serve` tool and adapter routes checking neither `Host` nor `Origin`; `soup mcp serve --allow-execute` not pinning every approved input; config regexes able to hang loading or training (the old guard ran the untrusted pattern against a probe); credential option values written to the audit log; and unbounded reads and YAML alias expansion when reading an untrusted `.can`. The advisory carries the affected range of each issue and says which credentials to rotate.

## [0.75.0] - 2026-09-12

### Added

- Wire opt-in anonymous hardware-only telemetry behind `SOUP_TELEMETRY=1` with stdlib delivery, `--no-telemetry` flag, command sanitization, and public privacy policy (#318 by @kok-o in #529).

- **Added a ready-made `deepseek-v4-flash-dpo` recipe for deepseek-ai/DeepSeek-V4-Flash
  (#275 by @Srinivasan8888 in #662).** The base already shipped SFT (v0.71.24) and
  GRPO (#279 by @Faisal-Fayaz); this completes the trio with the preference shape
  (`task: dpo`, `format: dpo`, `preference_train.jsonl`, `dpo_beta: 0.1`,
  `lr: 5e-6`) over the MoE geometry its SFT sibling already uses (LoRA r16/a32,
  `batch_size: auto`, `gradient_accumulation_steps: 8`, 4-bit, `moe_lora`,
  `moe_aux_loss_coeff`) — deliberately not the GRPO sibling's `1`/`16`, whose
  batch shape is driven by rollout generation that DPO does not do. `lr: 5e-6`
  and `dpo_beta: 0.1` are unanimous across all 12 DPO recipes in the catalog.
  The recipe is not trained — DeepSeek-V4-Flash is MoE — so no hyperparameter
  here is a measured recommendation. Catalog count 163 -> 164.

- **Added a ready-made `kimi-k2.6-dpo` recipe for moonshotai/Kimi-K2.6
  (#275 by @kok-o in #664).** The base already shipped SFT (v0.71.24)
  and GRPO (#614 by @umran666); this completes the trio with the direct preference
  optimization (DPO) shape (`task: dpo`, `preference_train.jsonl`, `dpo_beta: 0.1`,
  `lr: 5e-6`) over the ~1T MoE geometry its siblings already use
  (LoRA r32/a64, `batch_size: 1`, `gradient_accumulation_steps: 16`, 4-bit,
  `moe_lora`, `gradient_checkpointing`, `max_length: 8192`). The recipe is not
  trained — ~1T MoE is multi-node — so no hyperparameter here is a measured
  recommendation. Catalog count 164 -> 165.

- **The MLX backend now drives Soup's live terminal dashboard, the `soup ui` stream and the experiment tracker
  (#23 by @Srinivasan8888 in #665).** `MLXSFTTrainerWrapper.train()` accepted
  `display` / `tracker` / `run_id` and discarded them, so an MLX run showed
  mlx-lm's raw stdout while the same config on the transformers path got the
  Rich panel. It is an adapter rather than a reuse of `SoupTrainerCallback` —
  mlx-lm exposes only `on_train_loss_report` / `on_val_loss_report`, so
  bridging through a HuggingFace `TrainerCallback` would have meant faking
  trainer state. `speed` carries `iterations_per_second` to match the
  transformers path's `train_steps_per_second`, because the panel hard-labels
  that field `it/s`; `grad_norm` is deliberately left unset, since mlx-lm does
  not compute one and a field reading a plausible `0.0` on one backend is worse
  than an absent one. Verified against real mlx-lm on an 8 GB M1.

- **Validation loss is now recorded and displayed (#23 by @Srinivasan8888 in
  #713).** Soup never stored it on any backend: `SoupTrainerCallback.on_log`
  read `logs["loss"]` and never `logs["eval_loss"]`, so an evaluation step left
  the last training loss in place and re-reported it to every sink — the
  evaluated number existed nowhere. There was also nowhere to put it: no
  `metrics` column, no `TrainEvent` field, and `TrainingDisplay.update()` reads
  only `grad_norm`, `speed` and `gpu_mem` out of `**kwargs`. The live panel, the
  `metrics` table and the `/api/train/stream` SSE frame now each carry
  `val_loss` as its own series, never folded into `loss`, and the panel gains a
  `Val loss` row. The panel carries the last measured value forward between
  evaluations; the stored and streamed values do **not** — a step where no
  evaluation ran records `NULL`, so a series read back has one point per
  measurement rather than one per logged step. Existing `~/.soup/experiments.db` files are
  migrated in place; rows written before this ship read `NULL` rather than a
  fabricated `0.0`, because an unmeasured value must not read as a measured one.
  The MLX producer follows separately.

- **MLX runs now report validation loss (#23 by @Srinivasan8888 in #739).** mlx-lm's `on_val_loss_report` hook was never implemented, so it was called on every evaluation and did nothing — the base-class method is a `pass`, making it a silent no-op rather than an error. It now drives all three sinks #713 added: the live display, the `metrics.val_loss` column, and the `val_loss` field on the SSE stream. The last measured value stays on the panel between evaluations, but only real measurements are written to the database or the wire, so a run records one row per evaluation rather than one per training step. Upstream's `iteration` is recorded as sent, including its off-by-one against the training counter, so a row matches mlx-lm's own log line. A payload without a `val_loss` key writes no row, emits no event, and does not clear an earlier measurement. Verified on Apple Silicon with a real validation split: 9 display updates, 5 database rows and 5 stream events for 5 evaluations.

- **Proved LoRA+ optimizer and scheduler state survive save/resume (#724 by @SID-6921 in
  #747).**
  #738 fixed `training.loraplus_lr_ratio` crashing on the SFT, pretrain and embedding
  wrappers, but its last acceptance criterion — that resuming a LoRA+ run preserves the
  optimizer's momentum and the scheduler's decayed learning rate — was never exercised.
  A new test runs a real 4-step training loop twice against the same seeded base model and
  dataset, once straight through and once interrupted at a real checkpoint and resumed,
  both targeting the same `max_steps`; the optimizer momentum, scheduler LR, and final LoRA
  weights all match exactly between the two. No production code changed: the existing
  `attach_loraplus_optimizer` wiring already handled this correctly, it just wasn't proven.

- **`soup doctor --config soup.yaml` reports the settings your backend does not read (#755 by @Srinivasan8888 in #756).** A field can be declared, validated and documented and still be read by nothing on the backend you chose — filed one field at a time as #683, #686, #745 and #749 — and `soup doctor` could not help, because it read no config at all. It now loads the config and lists only the settings that config actually writes which its task and backend do not read, with the reason and, where one exists, the issue that recorded it; `model_fields_set` separates those from the fields sitting at their schema default. Backend support is declared rather than inferred, because inference does not work: reachability over the import graph detects none of five independently-known MLX gaps, reads of the trainer module alone invent gaps for fields that live in helper modules, and `--dry-run` exits before a trainer is ever constructed. **Scope is deliberately narrow: `task=sft` on `backend=mlx`, seven entries, every one of them a setting `trainer/mlx_sft.py` already warns about at runtime — the value added is that you see it before the run rather than during it, not that it knows anything new.** Five further entries were removed before merge because #734 and #750 wired those fields; the guard shipped here is what caught them. Every other task/backend pair reports nothing rather than guessing. A guard keeps the declared table honest in both directions — an entry marked unread fails once any declared module reads the field, an entry marked read-to-warn fails once the warning is deleted — scanning the trainer **and** the helpers it delegates to, because the fields #734 wired live mostly in `mlx_optim.py`. `soup doctor --config` exits 2 when the config cannot be read, so the leg can gate CI.

- **A Turkish README, behind a gate that checks what a translation was synced *from*, not only when (#852 by @Ercaner1988, supersedes #774 and #769).** `README.tr.md` is a full translation of the current `README.md`, linked from a language banner above the logo that lists only languages that exist. `tests/test_readme_translation_sync.py` turns CI red when a translation drifts: every `README.*.md` must carry a `synced-from` stamp matching `sha256(README.md)` over LF-normalised bytes, so CRLF checkouts on Windows agree with Linux and macOS; files are found with a glob, so a new language cannot land unchecked; and each `## ` section must keep `README.md`'s fenced-code languages, external URLs and inline-code spans, in both directions. That last check is what a stamp alone could not do — a different document can carry a valid stamp — and run against the previous draft of the Turkish file it found 29 dropped and 3 invented code references, including two release-note bullets that do not exist in `README.md`. A control suite breaks one thing at a time in a temp tree to prove each lock can fail.

- **`soup ingest --source langfuse --pull` fetches generations straight from Langfuse
  (#204 by @Konuktor in #859).** No export step: one row per `GENERATION` observation
  in a `--since` window (default `7d`), read from Langfuse's Observations API v2 —
  `/api/public/traces` leaves Langfuse Cloud on 2026-11-16 — and handed to the existing
  `parse_langfuse`, with chat message lists joined the way it already joins
  `{"messages": [...]}`. Credentials come from `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` (plus `LANGFUSE_HOST` or `LANGFUSE_BASE_URL`), never a flag, and
  stay out of the output, console, debug logs, audit log and error messages. The host is
  HTTPS-only through the same SSRF validator as `--slack-url`, with `--allow-private-host`
  for self-hosted Langfuse; redirects are refused. Every loop is bounded: 30 s per
  request, 64 MiB per response, a `--max-pages` cap (default 100) that exits 1 and writes
  nothing instead of truncating, and 429 retries honouring `Retry-After` up to 60 s.
  Standard-library HTTPS, so no new dependency; without `--pull` nothing changes and the
  pull code is not imported. The hint the local-export path prints named `LANGFUSE_KEY`,
  a variable Langfuse does not read; it now names the key pair. Validated end to end
  against a Langfuse Cloud Hobby project. LangSmith, Helicone, OpenPipe and OpenAI Stored
  Completions stay open on #204.

- **Added the `qwen3.5-0.8b-grpo` and `qwen3.5-2b-grpo` recipes (#848 by @SID-6921 in
  #864).**
  `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` previously shipped SFT only; both models are
  small enough that a contributor might actually run a few steps of GRPO reasoning training
  on their own hardware. Learning rate (`1e-5`), LoRA rank (`r=16`/`alpha=32`), and
  `max_length: 4096` follow the existing GRPO templates rather than the SFT siblings'
  values, stated explicitly rather than inherited silently. Catalog count: 165 -> 167.

### Changed

- **The README now leads with `pipx`, and names PEP 668 by its error text (#671 by @swalla02 in #673).**
  `pip install soup-cli`, the first command in the README, fails on Debian 12 and
  Ubuntu 23.04 or later:

  ```
  error: externally-managed-environment
  ```

  Those distros ship an `EXTERNALLY-MANAGED` marker in the system Python so pip
  refuses to write into a `site-packages` that `apt` also manages
  ([PEP 668](https://peps.python.org/pep-0668/)). It is the default on current
  Debian and Ubuntu, so it is now the ordinary first-run experience on Linux
  rather than an edge case, and nothing in Soup can rescue the command: the
  package is not installed yet when the README line runs.

  Soup declares a `soup` console script, which makes it an application rather
  than a library, so the install now leads with `pipx install soup-cli` and
  `uv tool install soup-cli`. Both give Soup its own virtualenv and put the
  command on `PATH`, which is outside PEP 668's scope entirely. Every existing
  `pip install` line is kept, under a sentence saying when it is the right
  choice: already inside a virtualenv, a Colab notebook, or a Docker image.
  A callout carries the literal `externally-managed-environment` string so the
  error text pasted into a search engine matches Soup's own README, and it also
  gives the `python3 -m venv` route for anyone who would rather not add a tool.

  Use `pip` rather than `pipx` if you want to `import soup_cli` alongside your
  own code; pipx isolation is the wrong shape for that.

  Documentation only. No packaging or command behaviour changed.

- **Multipack FFD placement is O(N log N) instead of O(N^2) (#694 by @dchaudhari7177 in [#726](https://github.com/MakazhanAlpamys/Soup/pull/726)).** Preparing a multipack epoch scanned every open bin for each item; placement now descends a segment tree of per-bin remaining capacity. The packing is unchanged, which is the property that matters — altering which item lands in which bin would silently change every multipack training run — and that was verified at merge against the pre-change function over 6,000 randomized cases with 5,384 of them containing repeated lengths, for zero mismatches. Measured on one box: 5.5x faster at 1,000 rows, 37x at 10,000, and 96x at 30,000, with per-10x-N scaling dropping from 104-130x to 8.9-13.9x. The cap error message no longer describes the algorithm as quadratic, because it no longer is.

- **Sequence-level GSPO objective replaces legacy column-centering (#744 by @kok-o, closes #723).**
  Upgraded `grpo_variant: gspo` from the legacy column-centering heuristic to the
  published Group Sequence Policy Optimization algorithm (Qwen Team, arXiv:2507.18071).
  Because the mathematical objective is replaced at the sequence level, existing
  configurations specifying `grpo_variant: gspo` will yield different numerical losses
  and gradients on upgrade compared to prior versions and will not reproduce prior runs.
  The objective now computes a length-normalized sequence importance ratio
  $s_i = \exp\left(\frac{1}{|y_i|} \sum_{t=1}^{|y_i|} (\log p_\text{new} - \log p_\text{old})\right)$
  and applies sequence-level surrogate clipping with default radius $\varepsilon = 0.2$
  or operator-supplied `grpo_delta`, strictly isolating masked tokens with 0.0 gradient,
  remaining invariant to padding tokens and placement (left vs right), and respecting
  batch permutation invariance.

- **Optimize dataset validator (`soup data validate` / `validate_and_stats`) (#771 by @iam-saiteja).**
  Consolidates statistics aggregation into a single pass and replaces string serialization
  with fast hashable row signatures for faster dataset inspection (~1.25x–1.60x speedup).

- **Breaking: an unknown config key now refuses the load (#627 deadline, in #879).**
  v0.74.0 reported every key no config model declares — a typo like `quantizaton`, or a
  field that only exists on a newer Soup — as a warning that named v0.75 as the release
  that would start refusing, and `TestTheDeadline` pins that promise to the declared
  version in both directions. This is that release: `soup train` exits 1 before the
  training stack is imported, and the API / Web UI loader raises `ValueError` with the
  same text, naming the field you probably meant. Nothing is defaulted or substituted.
  The refusing message says `Refused.` rather than `Not applied.`, which would have read
  as if the run went ahead without the key; the `soup sweep` guard, which always refused,
  changes wording the same way. The detector now sees a config the way `SoupConfig`
  does: a root-level `lora:` block — the LlamaFactory / Axolotl spelling the schema has
  accepted and moved under `training` since v0.40.1 — is remapped through the same
  function before the walk, so a spelling the validator accepts is never one the
  detector refuses (the release review caught the first draft refusing it). Two
  `soup fetch examples` files (`llama-3.1-8b-lora`, `qwen2.5-7b-dpo`) used that
  spelling and were moved to the canonical `training.lora`; every recipe, template and
  example config scans clean. Two more findings from the same review ride along: key
  names reached `console.print` unescaped (Rich markup and raw ESC bytes from a YAML
  key could restyle or spoof the terminal — the class six command modules already guard
  privately; the loader now uses the shared `soup_cli.utils.terminal.for_terminal`), and
  the scan was unbounded (50,000 bogus keys cost ~31 s of CPU per Web UI request,
  measured; it now stops at 100 findings and says so). The Web UI's `/api/train/start`
  and `/api/config/from-form` return the loader's message — the key and the suggestion —
  instead of a generic "invalid configuration", rendered through the SPA's `escapeHtml()`.
  A config that must stay loadable on v0.74 as well needs the key removed, not renamed.

### Fixed

- Direct AWQ exports now require an explicit, usable calibration JSONL before
  AutoAWQ is imported or the model is loaded, preventing its large default
  calibration dataset from being downloaded silently (#338 by @Faisal-Fayaz in #592).

- **GEMM throughput forecasts now use the card's resolved stream dtype (#617 by @Samearth17 in #648).**
  The pre-flight measurement no longer hard-codes bf16, so pre-Ampere cards are
  benchmarked with the fp16 dtype they actually use. The forecast also reports
  the measured dtype alongside TFLOPS and clock.

- Repaired non-existent Hugging Face repository IDs in MLX catalog recipes
  (#661 by @kok-o in #666). `qwen3-8b-sft-mlx` previously pointed to
  `mlx-community/Qwen3-8B-Instruct-4bit` (which huggingface_hub reports as
  RepositoryNotFoundError); it now references `mlx-community/Qwen3-8B-4bit`.
  `gemma3-9b-sft-mlx` is renamed to `gemma3-4b-sft-mlx` (user-visible rename;
  the old 9B size does not exist upstream) pointing to
  `mlx-community/gemma-3-4b-it-4bit` (size 4B, M1+ 16GB).

- **A non-dict message in a `multimodal` row aborted the whole dataset load instead of
  dropping the row (#670 by @abdulwaarith0).**
  `_convert_multimodal` iterates `row["messages"]` and calls `msg.get("content")` without
  first checking that `msg` is a dict, so a row like `{"messages": ["hello"]}` — a bare
  string, `None`, an int, or a nested list where an object belongs — raised
  `AttributeError: 'str' object has no attribute 'get'`. `format_to_messages` catches
  `KeyError`, `TypeError`, `IndexError` and `ValueError` to route malformed rows to its
  drop path, and `AttributeError` is in none of them, so the exception escaped the
  converter and killed the run: one bad line in a large JSONL file took the entire dataset
  with it rather than being skipped like every other malformed row. The converter already
  guarded the *parts* inside a message (`multimodal content part must be a dict`) but not
  the message itself. It now raises `ValueError` for a non-dict message, which the existing
  drop path already understands, so such rows are dropped and the rest of the dataset
  loads. Valid rows are unaffected: typed content-part lists and the plain-string
  back-compat path both convert exactly as before.

- `soup migrate` no longer refuses a valid competitor config just because the file
  is named `*.jsonl`. The guard branched on the filename alone; the content sniff
  written to gate it, `_looks_like_jsonl()`, had no call site, so a YAML config
  saved as `config.jsonl` exited 2 with "got JSONL" and could not be migrated at
  all. Both conditions are now required, which leaves `.ipynb` notebooks (which
  legitimately start with `{`) untouched. The N2 test now drives the command
  through `CliRunner` rather than calling the helper, so the wiring is what is
  actually asserted (#675 by @swalla02, closes #672).
  The sniff also reads `utf-8-sig` rather than `utf-8`, so a UTF-8 BOM (which
  Windows tooling writes by default) cannot defeat it, and its read is bounded
  so a file with no newline is not pulled into memory entire.

- **chatml / audio / video converters silently kept a non-dict message instead of
  dropping the row (#676 by @BetterAndBetterII in #678).**
  `format_to_messages` documents a drop contract: a malformed row returns `None`
  so one bad JSONL line is skipped rather than corrupting the dataset.
  `_convert_chatml`, `_convert_audio` and `_convert_video` passed a non-dict
  `messages` element through verbatim, so `{"messages": ["hello"]}` — or `42`,
  a nested list, or `None` where a message object belongs — survived into
  training. `chatml` is what `detect_format` returns for a bare
  `{"messages": [...]}` row, so this was the default path for the most common
  dataset shape. Each converter now raises `ValueError` for a non-dict message,
  which the existing drop path already understands. The list-type guard from
  @v01dst in #679 also rejects empty non-list values such as `""` and `{}`;
  regression tests pin both shapes. Video rows with missing or null `messages`
  still normalize to an empty list. The shared conversion-based validator
  from #712 uses these guards too, so `soup data validate` and `load_dataset`
  agree on which rows are dropped. Valid rows convert exactly as before.

- **MLX SFT ignored `training.gradient_accumulation_steps` (#684 by @AmirF194 in #696).** The wrapper built
  mlx-lm's `TrainingArgs` without a `grad_accumulation_steps` kwarg, so mlx-lm's own
  dataclass default of 1 always applied regardless of the configured value, silently
  changing the effective batch size and optimizer-update cadence on every MLX run. The
  written `adapter_config.json` also hardcoded `grad_accumulation_steps: 1`, so the drift
  could not be detected from the output afterwards either. Both now carry the value
  `training.gradient_accumulation_steps` actually resolves to. `iters` is also rounded
  down to a whole number of accumulation groups (kept at a minimum of one group), because
  mlx-lm only updates the optimizer on `it % accum == 0` and never flushes a trailing
  partial group; without this a dataset smaller than the accumulation window trained for
  zero optimizer updates.

  **Anyone who never touched this field will see a different effective batch size after
  upgrading.** `gradient_accumulation_steps` defaults to 4, so an MLX run that used to
  update on every micro-batch now accumulates over 4 before updating: a 4x larger
  effective batch size and roughly a quarter as many optimizer updates for the same
  `iters`. This is the correct value for the schema default; a differently-converging run
  after this upgrade is expected, not a regression.

- **MLX SFT ignored `training.gradient_checkpointing` (#685 by @AmirF194 in #698).** The wrapper built
  mlx-lm's `TrainingArgs` without a `grad_checkpoint` kwarg, so mlx-lm's own dataclass
  default of `False` always applied regardless of the configured value, silently giving up
  the memory savings a user enabled checkpointing for. The written `adapter_config.json`
  also hardcoded `grad_checkpoint: false`, so the drift could not be detected from the
  output afterwards either. `gradient_checkpointing` accepts a bool or one of the
  `selective`/`medium`/`full`/`auto` tiers; mlx-lm has only a single on/off switch, so any
  non-`False` tier now resolves to `True` there, the same `bool()` coercion already used
  for this field on the other trainer backends. A tier value now also prints the same
  `MLX backend ignores: ...` advisory this file already uses for other flattened options,
  since the granularity itself is silently dropped even though checkpointing is enabled.

- `data.interleave`'s `over`/`probs` strategies could place the same row (a source's original plus its `over`/`probs`-padded copy) in both `train` and `val`; `val_split` now runs before the padding, per source, on the eager local-file and HF-hub interleave paths (the streaming path is unaffected and still has this leak). The requested `val_split` fraction is no longer exact on those two paths under `over`/`probs`, since it is now taken from each source's smaller, unpadded row count; `concat`/`under` are unaffected either way. See `docs/data.md`'s interleave section for the numbers (#680 by @AmirF194 in #701).

- **The MLX SFT benchmark harness now exercises the display/tracker bridge
  (#703 by @Shutaru; Refs #23).** It passes a real Rich display and an artifact-local SQLite
  tracker to the MLX wrapper, and rejects runs without display updates or
  recorded metrics. Training stays inside the stdout tee so whole-run
  throughput still comes from mlx-lm's trained-token counter, with ANSI controls
  and wrapped whitespace normalised before parsing the final report. Regression
  tests cover the wiring, terminal output, failure/interruption cleanup, and
  disconnected telemetry. The stdout tee preserves terminal detection so the
  Rich panel can refresh during an interactive run; all terminal widths from
  20 to 200 are exercised by the regression test. The existing Apple Silicon
  measurements are unchanged; a new hardware run with the bridge enabled remains outstanding.

- **`wasserstein`/`topk_align` cross-tokenizer distillation trained on logits the teacher
  computed for the wrong text, with no visible failure (#704 by @AmirF194, closes #681).**
  Both strategies forwarded the student's own `input_ids` straight to the teacher, clamped
  into its vocab range to avoid an index error. Clamping does not translate tokens between
  vocabularies, so a mismatched pair silently fed the teacher garbled or wrong text while
  still reporting a finite, plausible-looking loss. `wasserstein`/`topk_align` now refuse a
  tokenizer pair that isn't interchangeable at `setup()`, naming `wasserstein_aligned` (which
  already re-tokenizes correctly) as the working alternative for genuinely different
  tokenizers. The same-tokenizer fast path (e.g. two sizes in one family) is unaffected.

- Fixed `ValueError: Lora rank r must be > 0` when training embeddings with
  `lora.r: 0` (full fine-tuning) on `task: embedding` (#700 by @kok-o in #705).
  `_setup_transformers()` now skips PEFT wrapping when `r == 0`, and
  `_EmbeddingTrainer` delegates `model` and `args` attributes.

- Prevented training subprocess hang in Web UI when no client drains stdout
  (#688 by @kok-o in #706). Output is now continuously consumed by a background
  thread into a bounded ring buffer, allowing training to proceed regardless of
  subscriber state and supporting multi-subscriber replay via `Last-Event-ID`.

- Required authentication on Web UI read endpoints and SSE streams
  (#687 by @kok-o in #707). Run configurations, logs, and system metrics now
  require Bearer authorization. SSE endpoints authenticate via short-lived,
  single-use tickets exchanged over an authenticated POST, avoiding durable
  tokens in query strings. Non-loopback binding requires a valid authentication
  token and is rejected with exit code 2 if missing or invalid.

- **`task: embedding` on the transformers backend no longer crashes after setup (#708 by @jagadeepmamidi, closes #690).** `_EmbeddingTrainer` already delegated `train` / `save_model` / `add_callback` / `state` but not `model` or `args`, which `train()` reads for fp16 dtype alignment.

- **SFT and pretrain `packing: true` no longer raise `TypeError` on TRL 0.29 (#709 by @jagadeepmamidi, closes #691).** `packing` is an `SFTConfig` field; passing it as an `SFTTrainer` kwarg is rejected because `__init__` has no `**kwargs`. Soup now carries it through `_as_sft_config` on both trainers. `packing_cross_doc_attn_mask` is schema-rejected: it never mapped to a valid TRL `packing_strategy` (allowlist is `bfd` / `bfd-requeue` / `wrapped`). Use `packing: true` with FlashAttention instead.

- **`data.streaming` is honoured for a single HuggingFace Hub dataset name (#710 by @jagadeepmamidi, Refs #689).** `_load_one_hub_dataset` now forwards `streaming=True` (and `buffer_size` shuffle) to `datasets.load_dataset`, then materialises up to `MAX_REMOTE_ROWS`. Previously the flag was ignored and the full dataset was materialised; a Hub source over 1,000,000 rows is now cut to 1M with a warning. An all-hub interleave list with streaming stays schema-rejected.

- The ULD distillation loss (`wasserstein` / `topk_align` / `wasserstein_aligned`)
  ignored the response-only label mask and the causal shift the CE term already
  applies, so under the default `train_on_responses_only` it optimized prompt
  tokens and the shift-boundary position too (#711 by @AmirF194, closes
  #682). `uld_distill_loss` and `uld_aligned_loss` now take an optional
  `labels` argument and mask on `labels != -100` after the same shift
  `_compute_distill_term` uses.

- **`soup data validate` green-lit rows that `load_dataset` then drops, because the
  validator never ran the converters (#712 by @abdulwaarith0).**
  `validate_and_stats` judged a row by top-level key presence against `FORMAT_SIGNATURES`,
  while the loader runs `format_to_messages` and drops any row that returns `None`. Those
  are different notions of "valid", so they disagreed two ways: the six formats absent from
  `FORMAT_SIGNATURES` (`prm`, `pre_tokenized`, `input_output`, `video`, `multimodal`,
  `raft`) skipped validation entirely and reported every row valid regardless of content,
  and for the formats that were checked, key presence passed rows a converter drops on a
  value — a null field, a non-dict message. `soup data validate` could report a file fully
  valid that the loader could not load, moving the failure from validate time into the
  middle of a training run. `valid_rows` is now computed by running the real conversion
  path per row, so agreement is structural rather than a second, weaker check that drifts;
  every format is covered, including the six that were skipped. When a row is dropped, the
  converter's own reason is surfaced for the first few offenders (`row 3: chatml message
  must be a dict`) so the report says which rows and why, not only a count. Exit-code
  behaviour is unchanged: `validate` still exits 0 even at `0/N`, so the `soup ci init`
  gate is unaffected; whether a `0/N` file should fail the gate is left to a separate
  change. On the repo's own datasets this shifts 120 of 360 file-by-format verdicts, every
  one an over-count on a file that genuinely has zero valid rows in that format, with no
  exit code moved.

- #651: Raise the minimum supported PyTorch version to 2.6.0 because TRL 0.29 preference trainers require the public FSDP2 API. (#651 by @Samearth17 in #717)

- **MLX SFT now honours `data.train_on_responses_only` (#683 by @Srinivasan8888 in #733).** The option defaults to `true` and reached nothing on the MLX path: no mask was passed to mlx-lm's dataset constructor, so every MLX SFT run trained on system and user turns against the documented default, and adapter metadata recorded `mask_prompt: false` regardless. Setting upstream's flag is not the fix — `ChatDataset` masks a single prefix ending before the last message, so on multi-turn chat it supervises only the final assistant turn. Measured on Apple Silicon over 16 two-turn conversations: 772 supervised tokens before, 71 with upstream's flag, 146 with a correct mask. Chat rows now use a Soup per-token mask injected through `train(loss=..., iterate_batches=...)`; prompt/completion rows use upstream's flag, which is correct for that shape; plain-text rows warn instead, because upstream raises when the flag is set on them. Non-prefix-stable chat templates, conversations with no assistant content, and conversations opening on an assistant turn are refused explicitly rather than approximated. This does not make the two backends comparable: MLX now excludes the assistant header from the loss and refuses templates it cannot align, while the transformers path supervises the header and, on thinking-style templates such as Qwen3's, supervises a user-turn fragment. MLX is the stricter of the two; the transformers behaviour is filed separately. A template MLX cannot mask -- Qwen3's, which injects its thinking block only for the last assistant turn -- is now refused at dataset construction, before the training loop starts, with a message naming `data.train_on_responses_only: false` as the remedy. **One shipped recipe is affected.** `qwen3-8b-sft-mlx` does not set `data.train_on_responses_only`, so it takes the `true` default: measured against the real `Qwen/Qwen3-0.6B` tokenizer, single-turn and system+single-turn rows train (8 supervised tokens, the thinking block included), and **multi-turn rows now refuse** where they previously trained on the prompt. Set `data.train_on_responses_only: false` to restore the old behaviour on multi-turn Qwen3 data. The other two MLX recipes are unaffected: `llama3.1-8b-sft-mlx` and `gemma3-4b-sft-mlx` mask correctly on both shapes, verified against `mlx-community/gemma-3-4b-it-4bit` itself.

- **MLX SFT now honours `warmup_ratio`, `scheduler`, `weight_decay` and `optimizer` (#686 by @Srinivasan8888 in #734).** The backend built `AdamW(learning_rate=<scalar>)` and nothing else, so all four were validated, accepted and dropped without a warning — an MLX run silently trained a different recipe from the configured one, and only 8 of the 32 names on Soup's optimizer allowlist have an MLX equivalent while all 32 became AdamW. Schedules are now composed from `mlx.optimizers` in **optimizer-update** units (`iters // gradient_accumulation_steps`), which is what MLX drives a callable learning rate from; building against iterations would stretch the warmup by the accumulation factor and never reach the cosine floor. Measured on Apple Silicon, a `cosine` run with `warmup_ratio=0.2` now reports 10 distinct learning rates across 40 iterations where the old path reported one. Unsupported optimizer and scheduler names are refused by name rather than silently becoming AdamW, weight decay on an optimizer that cannot take it is refused rather than dropped, a `warmup_ratio` that rounds to zero updates says so, and adapter metadata records the effective optimizer and schedule.

- `grpo_variant: gspo` centered the per-token log-ratio across the batch
  before the completion mask was applied, so a masked (padding) token
  shifted the loss and gradient of every unmasked row sharing its column,
  and picked up a non-zero gradient of its own (#735 by @AmirF194, refs
  #723). The batch-mean statistic is now taken over unmasked positions
  only.

- PPO now forwards `training.epochs` and `ppo_kl_penalty` to TRL 0.29's
  `num_train_epochs` and `kl_coef` fields while retaining legacy field-name
  compatibility (#721 by @be-student in #737).

- **`training.loraplus_lr_ratio` crashed every affected run instead of enabling LoRA+
  (#738 by @abdulwaarith0).**
  The SFT, pretrain and embedding wrappers inserted `loraplus_lr_ratio` into
  `training_kwargs` and forwarded it to `TrainingArguments(**training_kwargs)`. It is not a
  `TrainingArguments` field — it belongs to PEFT's optimizer construction — so an advertised,
  schema-accepted option raised `TypeError: __init__() got an unexpected keyword argument
  'loraplus_lr_ratio'` before the first training step. It is now routed through a shared
  `attach_loraplus_optimizer` helper that builds a `create_loraplus_optimizer` optimizer (LoRA
  B matrices at `lr × ratio`, A at `lr`) and assigns it to `trainer.optimizer` after the
  trainer is built — respected because `Trainer.create_optimizer` only builds one when
  `self.optimizer is None`, with the scheduler still built from it using the configured
  warmup/schedule. The optimizer class and its betas/eps come from the run's configured
  optimizer via `Trainer.get_optimizer_cls_and_kwargs`, so LoRA+ uses the optimizer the user
  asked for, and weight decay is applied through PEFT's own `loraplus_weight_decay`. The
  combination with GaLore is now rejected with a clear message rather than silently overriding
  the GaLore optimizer, and `loraplus_lr_ratio` on a non-LoRA run is rejected instead of
  silently doing nothing.

- **MLX SFT now honours `training.max_grad_norm` (#749 by @Srinivasan8888 in #750).** The field is forwarded into `TrainingArguments` by sixteen transformers trainers, so the documented default clips every transformers run at 1.0 — and on `backend: mlx` it reached nothing: no MLX file read it, `mlx_lm`'s `TrainingArgs` has no such field and its trainer clips nowhere, and the `MLX backend ignores:` line did not mention it. The same config therefore trained clipped on one backend and unclipped on the other, silently. Gradients are now clipped through `mlx.optimizers.clip_grad_norm` by an optimizer that clips before delegating, which reaches mlx-lm's single `optimizer.update(model, grad)` call site without forking its training loop; `optimizer.state` and `optimizer.learning_rate` still delegate, because upstream reads both off the object it is handed. The norm that ran is recorded in `adapter_config.json`. Measured on an M1 inside upstream's own compiled-step shape, one step on an ill-conditioned batch moved SGD weights by 388.03 unclipped versus 0.0085 at `max_grad_norm: 1.0`; AdamW moves 0.126505 versus 0.126504, because Adam normalises by its second moment — so this rescues `optimizer: sgd` / `lion` and gradient spikes rather than every MLX run.

- **`soup eval auto` no longer dies with a `TypeError` after a successful eval (#752 by @Srinivasan8888 in #754).** Typer fills a command's parameters only when typer invokes it; called as a plain Python function, an unpassed parameter keeps its literal `typer.models.OptionInfo` default, which is truthy. `auto()` passed three of `custom()`'s five typer parameters, so `output` and `attach_to_registry` arrived as `OptionInfo`: the `--output`/`--attach-to-registry` block fired on every run though neither was requested, `output or "eval_results.json"` evaluated to the `OptionInfo` rather than the filename, and `write_eval_json` raised `TypeError: expected str, bytes or os.PathLike object, not OptionInfo` — uncaught, after the eval had already run and saved its results to the tracker. The same call in `monitoring/callback.py` is wrapped in `except Exception`, so mid-training auto-eval with `eval.custom_tasks` configured reported `Auto-eval custom failed: ... not OptionInfo` instead. Both call sites now pass every typer parameter, and a guard fails the suite when any bare-name call to a typer command leaves one unfilled — resolving each callee to a declaration in the calling module or a `from X import name`, rather than by bare name, which had merged every `main()` in the tree and flagged `autodistill/mlx_worker.py`'s argparse entry point with fifteen options it does not declare.

- **Inference prompts are now encoded the way Soup trains them (#781 by @Konuktor in #782).**
  `soup chat`, `soup serve` (transformers backend and `--mole`), `soup infer` /
  `soup bench`, `soup diff`, the `live_eval` generators behind the eval gate,
  `soup ship`, `soup diagnose` and `soup advise`, and `soup data generate`'s local
  provider rendered the chat template to text and then let the tokenizer add its
  own special tokens, so templates that render `{{ bos_token }}` (Llama-3, Gemma,
  Mistral) sent a doubled BOS while Soup's default SFT path trains with one. A
  rendered prompt is now encoded with no tokenizer special tokens, matching training
  and `apply_chat_template(tokenize=True)`: 2 → 1 BOS on vendor templates, 1 → 0 on
  Soup's `data.chat_template` presets (which render none), and no trailing EOS from
  tokenizers that append one; `usage.prompt_tokens` drops accordingly. Models without
  a chat template, and templates that fail to render, are encoded exactly as before.
  Adapters trained with `train_on_responses_only: false` saw the doubled BOS in
  training and now differ from inference by one token; that path and the vLLM /
  SGLang / MII engines are tracked in #781.

- Fixed `soup doctor` treating the optional `[train]` stack as missing required dependencies: a core-only install now suggests `pip install "soup-cli[train]"` once (with the CUDA wheel index on NVIDIA GPUs) instead of bare per-package floors, `soup version --full` derives installed extras from distribution metadata, and a missing or incompatible core dependency exits non-zero (#828 by @wangzhengzhuo05 in #854).

- Make `soup data validate` fail when no rows are usable, and add an optional
  minimum-valid-fraction threshold for stricter CI gates (#811 by @akkupratap323 in #858).

- Route terminal charts through Rich so `NO_COLOR` and redirected output contain no raw
  ANSI escapes while interactive charts retain colour (#824 by @akkupratap323 in #860).

- **`soup data validate --format` accepted any string and reported every row valid for it
  (#866 by @SID-6921 in #869).**
  `commands/data.py` declared `fmt: str` with help text listing the real formats but never
  checked the value against `soup_cli.data.formats.VALID_FORMATS` before `validate_and_stats`
  ran, so `--format bogus` exited 0 and reported "2/2 rows valid for bogus format". Since
  `soup data validate` is the first step of the PR gate `soup ci init` writes, a typo'd
  `--format` in a CI workflow silently turned the gate into a pass for any file — the same
  class of silent pass #811 and #858 just closed for "0 usable rows". An unknown format now
  exits 1 (input error), not the failed-gate exit code 2.

### Security

- **`soup ui --public` no longer serves the FastAPI docs to the LAN (#731 by @Srinivasan8888 in #732).** `/openapi.json`, `/docs`, `/docs/oauth2-redirect` and `/redoc` answered 200 without a token on a non-loopback bind, so anyone on the network could enumerate every route, parameter and request/response shape. No run data, configuration or logs were exposed — every endpoint the schema describes already answered 401 after #707 — so this was reconnaissance rather than disclosure, and it predates #707. The four routes are now absent (404) on a non-loopback bind and unchanged on loopback. They are removed rather than gated because `/docs` is a browser navigation that cannot carry a Bearer header, so gating would have broken the page for developers while leaving `/openapi.json` readable by any HTTP client.

## [0.74.0] - 2026-09-04

### Added

- **`soup eval aider` runs Aider's Polyglot code-editing benchmark through its
  official Docker harness (#91 by @Amix29 in #482).** The command preflights
  Docker, the daemon, and the locally built benchmark image; mounts a prepared
  Polyglot corpus read-only; keeps output under cwd; and aggregates bounded per-exercise JSON
  into a Soup result row. `--run-id` records the score in `eval_results` for the
  existing run comparison workflow. The optional `[aider]` extra installs the
  normal Aider CLI while the docs make the source-only benchmark-image setup
  explicit.
- **Native Apple Silicon telemetry for `soup monitor` (#99 by @Amix29 in #481).**
  The monitor now reads bounded plist output from macOS `powermetrics` and
  renders GPU utilization and power in the existing Rich table. It reuses an
  explicitly cached sudo credential through non-interactive `sudo -n`, never
  reads a password, and gives an actionable Activity Monitor fallback when
  permission or telemetry is unavailable. NVIDIA-only VRAM, memory-utilization,
  and temperature fields remain unavailable rather than being guessed.
- **`soup mcp serve` gains network transports: `--transport sse` and
  `--transport http` (#296 in #479).** v1 was stdio-only, which suits a client that
  spawns Soup as a subprocess but leaves remote and multi-client setups with
  nothing. Both new transports serve the *same* registry — the end-to-end test
  compares the advertised tool names against `build_registry` rather than a
  hardcoded count, so a subset cannot creep in. stdio remains the default and
  is untouched.

  Adding a listener is the risky part, so it is gated three ways. Every HTTP
  request needs `Authorization: Bearer <token>`, compared with
  `secrets.compare_digest`, with no opt-out — a loopback port is reachable by
  every process on the box. The token is validated by the existing
  `utils/qr_url.py::validate_token`, so `soup ui` and `soup mcp serve` agree on
  what a token is instead of growing a second format, and it travels in the
  header only: no query-string fallback that could land in an access log. The
  SDK's DNS-rebinding protection is switched on (`421` on a foreign `Host`,
  `403` on a foreign `Origin`), which is the gate the token cannot be — a page
  the operator merely visits sends no `Authorization` header but its request
  still reaches the port. Binding off loopback warns, and a wildcard bind warns
  again that the Host check has nothing left to pin.

  `--allow-execute` is refused with either network transport, and that refusal
  is the one behavioural change to an existing flag. Gated execution (#297)
  spawns real training / export processes; behind a listener a leaked Bearer
  token would mean process execution rather than plan disclosure, and stdio is
  a pipe to a client the operator already started, which is a different trust
  boundary. The refusal is made twice on purpose: once in the CLI so the
  operator gets a readable message, and once in `build_asgi_app()` so a direct
  caller cannot put an executing registry behind a listener either. The stdio
  banner also stopped claiming "execution disabled" -- that string predated
  #297 and was false from the moment execution shipped.

  `--host` / `--port` / `--auth-token` are refused under `--transport stdio`
  rather than silently ignored, and the ASGI app is built by a
  `build_asgi_app()` factory so the auth and rebinding behaviour is tested
  through an in-process transport without binding a socket; one further test
  binds a real ephemeral port and drives initialize -> list_tools -> call_tool
  through the SDK's own SSE client.

  The `[mcp]` extra floor moves `1.2.0` -> `1.10.0`, measured against the
  published wheels rather than a changelog: `server/streamable_http_manager.py`
  first appears in 1.8.0 (absent in 1.7.0) and `server/transport_security.py`
  in 1.10.0 (absent in 1.9.4). `--transport sse` alone would have run on the
  old floor; rebinding protection would not, and a listener whose origin
  checking silently disappears on an older SDK is worse than a resolver error.
  The `<2` cap is unchanged — #322 is the 2.x migration. No new package: `mcp`
  already requires starlette, uvicorn, sse-starlette and httpx-sse.

- **`soup data best-of-n` can sample candidates from Ollama or vLLM providers
  (#299 by @Faisal-Fayaz in #466).** The existing local Transformers `--base` path stays
  the default, while `--provider ollama|vllm --model <m> [--base-url <url>]`
  draws each prompt's N candidates through the existing SSRF-validated raw-
  completion seam. Provider and model are recorded in `_best_of_n` provenance;
  Anthropic is refused by name because its Messages API has no raw-completion
  endpoint, and local-only flags cannot be silently ignored in provider mode.

- **Baseline artifacts carry a scorer/version provenance stamp (#404 by @AchuthReddy-16 in #485).**
  Shared helpers `stamp_baseline_scores` / `write_baseline_file` write
  `{"scores": {...}, "provenance": {"soup_version", "scorer_revision"}}`.
  `soup eval gate --write-baseline <path>` is the user-facing producer.
  `resolve_baseline` warns once on unknown provenance (unstamped files) or a
  `scorer_revision` mismatch, and stays silent when the stamp matches.
  `BUNDLED_SCORER_REVISION` + a locked fingerprint fail the suite if a bundled
  scorer's output moves without a revision bump. Registry `save_eval_result`
  stamps `details_json`; `soup ship --emit-evidence` includes the same stamp.

- **Layer streaming now accepts Qwen3.5 MoE text checkpoints whose decoder
  layers do not expose exactly the same weight keys in every block (by
  @Shutaru in #426).** The sharder now records per-layer shard headers instead of
  deriving the pool from layer 0 alone, while still refusing divergent storage
  layouts for any shared key and rebuilding NF4 `Params4bit` views from
  validated per-weight metadata. `qwen3_5_moe` / `qwen3_5_moe_text` route
  through the existing qwen3 streamer. A heterogeneous toy MoE is bit-exact
  streamed versus resident on CPU; live validation was also run on
  `Qwen/Qwen3.5-35B-A3B` with layer streaming, NF4, MoE LoRA target resolution
  and a 3072-token SFT dataset. `stream_layers` now refuses
  `moe_expert_quant`, which is applied only by the resident setup path and was
  otherwise silently ignored.

- **`live_eval.load_model_and_tokenizer` gains a `quantization` parameter; no live evaluation
  path sets it yet (#367 by @AmirF194 in #461).**
  `quantization="4bit"` builds the same nf4 `BitsAndBytesConfig` every other 4-bit load path
  in this codebase uses (`"8bit"` and the unset default are also supported), and the four
  internal callers this helper has (`make_generator`, `make_multi_generator`, `lora_probe`,
  `measure_logit_agreement`) still call it with no `quantization`, so today an NF4-trained
  adapter is still judged against a bf16 base it never saw during training. The follow-up is
  wiring those four callers to a default derived from the run's own configuration (`soup ship
  --config`, the registry entry, or the adapter's `adapter_config.json`), per the issue's own
  Fix path; that plus `soup ship` reporting the numerics it judged with and a staleness gate
  on mismatched-numerics evidence are left open (issue acceptance criteria 2 and 4).

- **`training.stream_pin` makes layer-streaming pinning configurable (#366 by @ousamabenyounes in #416).**
  Page-locking the RAM store is chosen automatically by `decide_pinning`, and
  until now nothing could override it — so while #331 was live, `pin=False` was
  the only known mitigation for silently wrong NF4 gradients yet was unreachable
  from `soup.yaml`. `stream_pin: false` now forces the pageable store (and the
  pre-flight states the throughput it costs, up to 6.56x measured, rather than
  absorbing it silently); `stream_pin: true` forces the pinned store and, **on
  the RAM tier**, refuses the run — naming the store size, not the ceiling
  (#366 AC3): `pinned_limit_bytes` is passed as `None`, so the page-lock ceiling
  is deliberately left unprobed and the store size is the only figure the
  refusal can honestly cite — if the box cannot page-lock it, instead of
  degrading silently; unset keeps today's automatic behaviour. On the disk tier
  (no RAM store to page-lock) and on CPU (no device to copy to) pinning is
  *inapplicable* rather than unsatisfiable, so `true` is **announced and the run
  proceeds** — refusing there would brick the large-model runs the disk tier
  exists for, and would make the key uncommittable to a config shared between a
  GPU box and a CPU box. Set while `stream_layers: false` it is rejected as a
  footgun, like the other stream keys.

- **A ready-made `qwen3.5-9b-grpo` recipe for GRPO reasoning training with `Qwen/Qwen3.5-9B` (#277 by @harshitthek in #448).**
  The recipe combines the established GRPO defaults (accuracy reward, beta=0.1, 4 generations)
  with LoRA r=16 and 4-bit quantization.

- **A ready-made `glm-5.1-dpo` recipe for DPO preference training with `zai-org/GLM-5.1` (#280 by @Osheun in #452).**
  The first recipe pairing DPO with a MoE base, so it carries `moe_lora: true`
  alongside the established DPO defaults (beta=0.1, LoRA r=32 / alpha=64, 4-bit
  quantization). `epochs: 1` and `max_length: 8192` are taken from the SFT
  sibling rather than the smaller qwen defaults, which suit a 754B MoE better.

- **Cross-tokenizer draft support for `soup draft` and `soup serve` (#304 by @CODING-DARSH in #417).**
  - `soup draft distill` now accepts target/draft pairs with mismatched tokenizers or vocabularies, automatically routing through `uld_strategy: wasserstein_aligned` (Universal Logit Distillation) instead of refusing the pair.
  - `soup draft measure` supports cross-tokenizer acceptance measurement using decoded character-span alignment (`count_accepted_spans`) across different vocabularies and token boundaries. Target generation neutralizes only `repetition_penalty` with `repetition_penalty=1.0` (Refs #345) so target greedy argmax and draft raw-logit scoring are evaluated consistently; remaining generation processors (`no_repeat_ngram_size`, `encoder_repetition_penalty`, `min_new_tokens`, `bad_words_ids`, `suppress_tokens`, and `sequence_bias`) are not altered.
  - `soup serve --speculative-decoding` supports cross-tokenizer draft serving via Transformers Universal Assisted Decoding (UAD) when supported by the installed `transformers` version, raising a clear error if unsupported.
  - Compatible same-tokenizer pairs strictly preserve the existing native fast path.
- **`data.interleave` is now wired into training-time dataset loading (#443 by @blackcoderx in #460).**
  `parse_interleave`/`InterleaveSpec` have been schema-validated and unit-tested since
  v0.42.0, but `load_dataset()` never called them — every multi-dataset mixture request
  silently trained on nothing but `data.train`'s single path, the same gap #330 and #442
  papered over in their respective renderers. `DataConfig.train` now accepts `str |
  list[str]`; a list of `>= 2` local file paths combines via `data.interleave`
  (`concat` / `under` / `over` / `{strategy: probs, probs: [...]}`) into one row set
  before the existing `val_split` line in `_finalize` runs, so a single path stays
  byte-identical. `interleave` is local-files-only: `training.packing` /
  `training.multipack` and `data.streaming` / an HF-hub dataset name are all rejected
  at config-parse time with a message naming the reason (streaming/hub-dataset
  interleaving is follow-up #459). Both the `soup data mix --optimize` recipe writer
  and the `--live` overlay renderer emit the real N-dataset mixture again instead of
  collapsing to one path.

- **A repo-wide documentation ratchet to guarantee declared recipe counts stay synchronized with the catalog (#453 by @harshitthek in #457).**
  Derives the expected count dynamically from `len(RECIPES)` and scans all declared documentation sites, preventing silent Git auto-merge drift across sequential recipe additions.

- Branch coverage for `lr_groups.py`'s `build_optimizer_param_groups`: the case
  where every parameter matches a configured group, so no `base` optimizer
  group is appended (#273 by @AmirF194 in #469).
- Branch coverage for `replay.py`'s `downsample`: the case where the stride
  already lands on the last row, so the endpoint pin is not appended a second
  time (#273 by @AmirF194 in #470).

- Re-enabled `/v1/tools/bash` execution with OS-level namespace/sandbox isolation (#151 by @kok-o in #527). Note this is a breaking change: `serve()` now raises `typer.Exit(code=2)` when run with `--host 0.0.0.0` without `--tool-auth-token`, whereas previously it only printed a warning.

- **`data.interleave` now supports `data.streaming: true` and lists of HF-hub dataset
  names (#459 by @blackcoderx in #468).**
  #443 left `data.interleave` local-files-only, refusing `data.streaming` and any
  remote-URI / hub-name list entry at parse time with a message naming this issue as
  the follow-up. Every `data.train` list entry is now classified once (local file /
  remote URI / HF-hub name) and dispatched: an all-local/remote list with
  `data.streaming: true` delegates combining to HF `datasets.interleave_datasets` /
  `concatenate_datasets` rather than reimplementing mixing over a source whose size
  can't be known ahead of time — `concat` maps to `concatenate_datasets`, `under`/`over`
  map to `stopping_strategy="first_exhausted"`/`"all_exhausted"`, and `probs` maps to
  `probabilities=probs`, each chosen so the strategy names mean the same thing as the
  local path (verified by running one `probs` config through both paths and comparing
  the resulting proportions, not by two tests that each pass alone). An all-HF-hub-name
  list is loaded eagerly per entry and combined with the same `_combine_interleaved` the
  local path already uses; a hub entry's own `validation` split is honoured for the
  combined result only when *every* entry provides one, otherwise it's ignored (warned)
  and `data.val_split` applies to the combined train rows — a decided precedence rather
  than an emergent one. Still refused, by name: an all-hub list with `data.streaming:
  true` (streaming N differently-shaped hub datasets and reconciling their splits is a
  separate, larger effort), and any list mixing hub names with local/remote entries.

- **LISA now accepts `task: pretrain`, not `sft` alone (#307 by @ousamabenyounes in #476).**
  Continued pre-training is the same full-fine-tune-of-a-rotating-set-of-decoder-layers
  mechanism LISA was built for, so the sft-only gate (inherited from Spectrum's
  `unfrozen_parameters`) was arbitrary. The schema task gate now reads a
  `_LISA_SUPPORTED_TASKS` allow-list (`sft`, `pretrain`) and its refusal names every
  accepted task instead of only rejecting yours; `trainer/pretrain.py` replaces its LoRA
  path with LISA and attaches `LisaCallback`, and reports its parameter summary as `LISA`
  counted off the raw parameters, since a LISA run has no `PeftModel` wrapper to ask
  `get_nb_trainable_parameters`. Both trainers route through one
  `peft_wiring.apply_lisa_setup`, so they cannot drift on what "LISA is on" means. The
  rest of the gate is unchanged: `transformers` + `text` + `quantization: none`, mutually
  exclusive with the LoRA feature flags, `freeze_layers`/`freeze_ratio` and
  `unfrozen_parameters`. The released `[0.71.34]` block still says `task: sft`, which is
  what 0.71.34 shipped.

- **A weekly `dependency drift` job, and a test that asks trl what it still
  accepts (#323 in #486).** Two failure classes were structurally invisible until a PR
  happened to be open: a bug that only manifests on a CPU-only runner (a CUDA
  build never calls `_convert_weight_packed_for_cpu`, so no GPU dev box can
  reach it), and an upstream removal behind a floor-only pin. The second one
  shipped for several releases with CI green throughout — `trl>=0.7.0` let CI
  resolve 0.29.1 while the dev box ran 0.19.1, and on 0.29.1 six trainers could
  not build their config at all, because the trl imports live inside `setup()`
  and no test had ever called it.

  `.github/workflows/dependency-drift.yml` runs the same resolve on a schedule:
  it installs the latest resolvable stack, runs the suite against it, and
  writes a resolved-vs-declared table into the run summary. It also flags a
  package declared with incompatible ranges in two extras — which it already
  found before merging: `[mlx]` asks for `transformers>=5.0.0` while `[train]`
  caps it below 5, so `pip install "soup-cli[train,mlx]"` cannot resolve.

  `tests/test_issue323_trl_kwarg_drift.py` answers the setup() question without
  a model: it reads the keywords each wrapper passes to its trl config and asks
  the installed class whether it still accepts them, through the same
  `config_accepts` capability probe the wrappers use at runtime — never a
  version comparison, since a version table is what was wrong twice. The
  `resolve_trl_symbol` indirection is followed, so the three configs that moved
  to `trl.experimental` are covered rather than silently skipped. Two blind
  spots are stated rather than implied: `**splat` calls are invisible to a
  static read, and the trl TRAINER classes are excluded because they take much
  of their signature through `**kwargs`, which makes a signature check report
  `model` and `train_dataset` as rejected — measured, and the reason the scan
  filters on `Config`.

- **`soup mcp serve` now runs on both mcp majors, and the `<2` cap is lifted
  (#322 in #498).** v0.72.3 capped the `[mcp]` extra the day mcp 2.0.0 broke
  every round-trip test; that unblocked a release and pinned anyone who wanted
  2.x in the same environment. Both removals are bridged rather than pinned
  around: the `@server.list_tools()` / `@server.call_tool()` decorators became
  `on_list_tools=` / `on_call_tool=` constructor callbacks, and
  `create_connected_server_and_client_session` gave way to the lower-level
  `create_client_server_memory_streams`, which both majors still ship.

  `build_server` chooses by probing the `Server` constructor, never by reading
  `mcp.__version__` — the rule `trainer/_trl_compat.py` earned after two wrong
  bounds derived from version tables, and a test walks the AST to enforce it.
  The dispatch logic stays in a single `_dispatch_tool` with two thin adapters,
  guarded by a test that fails if a second implementation appears, because two
  copies is how the majors would drift apart while both kept passing.

  The three modules the sse / streamable-http transports depend on all survive
  2.0.0 unchanged, so those 46 tests needed no edit. Verified by running the
  MCP suites twice, once per major: 184 passed against 2.0.0, and the full
  suite against 1.29.0. The floor stays at the 1.10.0 measured in #296.

- **Added a text-only `qwen3.8-27b-sft` catalog recipe for
  `Qwen/Qwen3.8-27B` (#477 by @Amix29 in #513).** The recipe explicitly selects
  `modality: text`, uses the measured Qwen3.5-family decoder path from #507,
  and includes catalog, configuration, CLI, modality, and count-sync coverage.

- `soup train --cloud lambda` adds cloud GPU training plans for Lambda Cloud instances (Refs #264 by @kok-o in #528).

- Added 7 ready-made SFT recipes - Qwen2.5-Coder 1.5B/14B/32B
  (`qwen2.5-coder-{1.5b,14b,32b}-sft`), Qwen2.5-Math 1.5B/7B (`qwen2.5-math-{1.5b,7b}-sft`),
  and DeepSeek-R1-Distill-Qwen 1.5B/7B (`deepseek-r1-distill-qwen-{1.5b,7b}-sft`) - and
  corrected `mistral-small-3-sft` to point at the real hub repo
  (`mistralai/Mistral-Small-24B-Instruct-2501`). Catalog grows 147 -> 154 (#536 by @Nick-800)

- Added #550: a content-addressed two-phase Best-of-N workflow for exporting local
  candidates and materializing verified offline judgments without model or network access (#550 by @Amix29).

- Added 4 ready-made recipes - DeepSeek-R1-Distill-Llama-8B SFT
  (`deepseek-r1-distill-llama-8b-sft`) and three R1-Distill DPO variants
  (`deepseek-r1-distill-qwen-{1.5b,7b}-dpo`, `deepseek-r1-distill-llama-8b-dpo`) with pinned
  lr and dpo_beta. Catalog grows 154 -> 158 (#569 by @Nick-800)

- [#572](https://github.com/MakazhanAlpamys/Soup/pull/572) adds a weight-free
  Qwen3.8-Flash-Next text-LoRA compatibility scaffold on Transformers 5.16.1,
  including architecture-aware linear targets, MoE detection, and a tiny-config
  forward/backward gate. Legacy int64-only Torch scatter runtimes receive an
  instance-local QSA index compatibility shim; real checkpoint, catalog,
  streaming, multimodal, and routed-expert-parameter support remain explicitly
  unclaimed (#572 by @Amix29).

- [#575](https://github.com/MakazhanAlpamys/Soup/pull/575) adds opt-in PEFT
  `target_parameters` LoRA for Qwen4-Exp routed `gate_up_proj` and `down_proj`
  expert tensors in resident Transformers SFT and continued pretraining, with
  validated compatibility constraints and weight-free backward/save/reload coverage (#575 by @Amix29).

- [#576](https://github.com/MakazhanAlpamys/Soup/pull/576) adds
  `training.lisa_train_embeddings` (default `true` = LISA as published in #267).
  Set it `false` to freeze LISA's always-on group — input embeddings, LM head,
  and final norm — so only the sampled `lisa_num_layers` decoder layers train.
  That always-on group is ~70% of everything LISA trains at 8B, so it is where
  LISA's memory actually goes; freezing it is a real quality/memory trade rather
  than a free win, which is why it is an opt-in knob. Setting it `false` while
  `lisa_enabled` is `false` is rejected, matching the other `lisa_*` fields.
  The analytical VRAM pre-flight still treats LISA as full fine-tuning
  regardless of this flag (it does not yet credit the frozen-embeddings saving,
  which needs a measured constant on GPU hardware), so a frozen-embeddings run
  that would fit can still be conservatively refused — use `--allow-oom-attempt`
  to launch it. Refs #377 (#576 by @Srinivasan8888).

- [#582](https://github.com/MakazhanAlpamys/Soup/pull/582) adds a ready-made
  `smollm3-3b-sft` recipe for HuggingFaceTB/SmolLM3-3B. The catalog shipped
  SmolLM2 in three sizes but no SmolLM3; this fills that gap with a small/edge
  LoRA SFT recipe (r8, 8-bit, auto batch). Catalog count 158 → 159. Closes #271 (#582 by @Srinivasan8888).

- Add Qwen4-Exp layer-streamed SFT with sparse read-only PLE N-gram access for dense Transformers and oMLX/oQ affine checkpoints, including an explicit `training.stream_ngram_source` policy and fail-closed task/quantization/media gates (#602 by @Amix29 in #603).

- Added the model-free AutoDistill Milestone A artifact contract: versioned and immutable
  plan/capture/shard/consumption schemas, explicit top-k plus residual-tail semantics,
  transactional resume/corruption rules, and deterministic plan-only estimates
  (#580 by @Amix29 in #613).

- **Added a ready-made `kimi-k2.6-grpo` recipe for moonshotai/Kimi-K2.6
  (#281 by @umran666 in #614).** The v0.71.24 model-family expansion shipped the
  SFT variant but no GRPO reasoning variant; this fills that gap with the MoE
  giant GRPO shape (`grpo_beta: 0.1`, `num_generations: 4`,
  `reward_fn: accuracy`, `moe_lora: true`, `gradient_checkpointing: true`,
  4-bit, `max_length: 8192`), and keeps the Modified MIT licence note in the
  recipe description. Catalog count 160 -> 161.

- **Added a ready-made `qwen3.5-35b-a3b-dpo` recipe for Qwen/Qwen3.5-35B-A3B
  (#276 by @Srinivasan8888 in #615).** The catalog shipped the MoE SFT sibling
  `qwen3.5-35b-a3b-sft` in v0.71.24 but no preference-tuning variant; this pairs
  the `qwen2.5-7b-dpo` DPO shape (`format: dpo`, `lr 5e-6`, `dpo_beta: 0.1`) with
  the sibling's MoE settings (`moe_lora: true`, `moe_aux_loss_coeff: 0.01`), and
  carries an explicit `modality: text` so it joins the Qwen3.5-family contract in
  `test_issue427_qwen35_text_modality.py` rather than falling back to the schema
  default. Catalog count 159 → 160.

- Added the internal AutoDistill Milestone B1 same-tokenizer, teacher-only MLX capture and
  transactional shard publication boundary, with immutable input fingerprints, explicit
  corruption/resume checks, and proof that no student model is loaded during capture
  (#580 by @Amix29 in #629).

- **Added a ready-made `qwen3.5-9b-dpo` recipe for Qwen/Qwen3.5-9B
  (#275 by @Srinivasan8888 in #632).** The base already shipped SFT and GRPO
  variants; this completes the trio with the preference-alignment shape
  (`task: dpo`, `format: dpo`, `dpo_beta: 0.1`, `lr: 5e-6`, LoRA r16/a32, 4-bit,
  `max_length: 4096`), matching every other DPO recipe in the catalog. Catalog
  count 161 -> 162.

- Pinned every recipe's *resolved* `SoupConfig` against a committed snapshot
  (`tests/fixtures/recipe_config_snapshots.json`, regenerated via
  `scripts/generate_recipe_snapshot.py`), so a schema-default change no longer
  silently retunes recipes that rely on that default. Confirmed by mutation:
  changing `dpo_beta`'s schema default failed 149 of the 162 recipes — every
  one that doesn't explicitly pin the field — which is the exposure this
  closes (#621 by @SID-6921 in #637).

- **Added a ready-made `glm-5.1-grpo` recipe for zai-org/GLM-5.1
  (#275 by @Srinivasan8888 in #656).** The base already shipped SFT (v0.71.24)
  and DPO (#280 by @Osheun); this completes the trio with the reasoning shape
  (`task: grpo`, `reasoning_train.jsonl`, `grpo_beta: 0.1`, `num_generations: 4`,
  `reward_fn: accuracy`) over the 754B MoE geometry its siblings already use
  (LoRA r32/a64, `batch_size: 1`, `gradient_accumulation_steps: 16`, 4-bit,
  `moe_lora`, `gradient_checkpointing`, `max_length: 8192`) — deliberately not
  the 30B GRPO template's r16/a32. `lr: 1e-5` is the value both conventions
  agree on: 15 of 22 GRPO recipes and `glm-5.1-sft` alike. The recipe is not
  trained — 754B is multi-node — so no hyperparameter here is a measured
  recommendation. Catalog count 162 -> 163.

### Changed

- **Remove the name-based `SCORER_CHANGED_IN_V0_73_2` baseline warning in favour
  of the #404 scorer_revision stamp (#404 by @AchuthReddy-16 in #485).**
  Stale baselines are detected by provenance, not by a hard-coded suite list.

- **Remove hand-maintained test suite statistics from `CONTRIBUTING.md` in favor of a permanent digit-free shape invariant (#465 by @harshitthek in #467).**
  Eliminates drift across routine test additions by making test-count divergence impossible at the documentation source.

- **Lazy callback builders now self-import their callback class names so runtime
  lookup never raises `NameError` while preserving lazy heavy-dependency loading
  (#320 by @AchuthReddy-16 in #455).** `build_echo_trap_callback`,
  `build_reward_hack_callback`, `build_minillm_callback`,
  `build_rl_checkpoint_callback`, and `build_push_callback` now resolve their
  callback types through local module imports in the builder body, and the
  regression suite adds subprocess coverage for all five builder calls.

- **User-visible changes now use per-PR changelog fragments (#487 by @Amix29 in #490).**
  Contributors add a uniquely named, version-scoped Markdown file instead of editing the
  shared `[Unreleased]` section. A standard-library assembler preserves long entries
  verbatim, rejects stale or malformed fragments, and consumes them during release
  preparation. A tag-time release gate refuses publication if any fragment remains, so
  the conflict is removed without making changelog loss silent.

- **Qwen3.5 and Qwen3.6 text recipes now state their decoder-only intent (#427 by
  @Amix29 in #501).** Apple Silicon measurements confirmed that Soup's text path loads
  `Qwen3_5ForCausalLM` without the visual tower, so all twelve catalog recipes now set
  `modality: text` explicitly and regression coverage prevents that decision from falling
  back to the schema default.

- **Transformers training now supports Qwen3.5-family text decoders and can be
  installed together with MLX (#502 and #503 by @Amix29 in #507).** The shared
  stack moves to Transformers 5.12.1+, TRL 0.29+, and PEFT 0.20+, with exact
  floor coverage, matching `soup doctor` diagnostics, and capability-based fallbacks
  for TRL APIs that moved under `trl.experimental`. Transformers LoRA
  `target_modules: auto` now covers both
  full- and linear-attention projections in Qwen3.5 text models without
  changing explicit targets or MLX defaults. The compatibility pass also
  updates removed Transformers arguments and keeps TRL 0.29's streamed DPO
  reference adapter off `meta`, so it remains a frozen adapter-sized snapshot
  instead of requiring a second model. DPO and ORPO also restore the removed
  prompt cap on TRL's prepared token ids, so `data.max_length` remains an
  effective sequence bound rather than a configuration-only value. The CLI's
  histogram and loss-curve renderers now support both plotext 5 and plotext 6, with
  a real Plotext 6 runtime pinned in the compatibility CI cell.

- **Unknown config keys are now reported instead of silently dropped, and v0.75 will reject them (#627 by @Srinivasan8888 in #628).**
  None of the config models overrode Pydantic's default `extra="ignore"`, so a key the
  schema did not declare validated clean and was discarded: `soup train --dry-run` printed
  "Config valid. Ready to train!", the run exited 0, and the requested setting was never
  applied — `training.quantizaton: none` trained 4-bit quantized when full precision was what
  you asked for, `training.gradient_checkpoint: true` did no checkpointing, `data.max_len: 512`
  truncated at 2048. #623 is the live
  case: `training.stream_pin` reached main two days after 0.73.3 shipped, a user on the
  released wheel wrote the documented escape hatch, and the resulting OOM was investigated
  as a layer-streaming defect. Loading a config now walks the whole model tree — `data`,
  `training`, `training.lora` and the rest, so a guard applied to one model and forgotten on
  another cannot look like it works — and reports every key it cannot place in **one** report
  per load, naming the field you probably meant (`did you mean 'quantization'?`). This is a
  warning, not a refusal: a config written against a newer Soup still runs on an older wheel,
  which is the case #623's user was in. **From v0.75 the same config will fail to load** — one
  minor of notice, since this release is the one that starts warning — and the warning names
  that version so the deadline is decidable rather than a permanent notice.
  The version is stated in one constant (`config/unknown_keys.py`) and asserted against the
  declared `soup_cli.__version__` by a test, so the release that crosses the deadline turns a
  test red rather than leaving the message promising a rejection that already shipped.
- **`soup sweep` now hard-errors on a `--param` that matches no config field, with no deadline (#627 by @Srinivasan8888 in #628).**
  A different failure class from a dropped training key, so it is called out separately: a
  sweep whose swept knob is never applied produces arms that are all identical, and there is
  no partially-useful result to preserve by continuing. `--param lora_rank=8,16` used to run
  the full grid at the base config's LoRA rank and report the winner; the grid is now checked
  before the first arm starts, and `sweep parameter does not match any config field: unknown
  config key 'lora_rank' - not applied.` is printed and the command **exits 1**. Anyone with a
  typo'd sweep parameter will see a new error where they previously got plausible, meaningless
  results, and a scripted sweep now fails instead of succeeding with a table of failed arms.

- **The recipe-config snapshot fixture (#621/#637) is now delta-encoded** —
  a change to a schema default that no recipe explicitly pins now produces
  one failure naming the shared baseline, not 149 identical failures across
  every recipe that relies on it. What's now structurally distinct is
  "a recipe's own value moved" (a named per-recipe test) versus "the shared
  defaults moved" (the one baseline test) — those are different tests now,
  not just different-looking diff text. Measured:
  `tests/fixtures/recipe_config_snapshots.json` goes from 1,571,967 bytes /
  49,586 lines to 66,560 bytes / 2,040 lines — 23.6x smaller by byte count,
  24.3x fewer lines.

  **Traded, not eliminated, and worth stating at the size that actually
  occurs**: a recipe that redundantly pins a value equal to the *old*
  default starts appearing in its own delta once the default moves out
  from under it, since what used to be a no-op pin just started doing
  something — that recipe's resolved config hasn't changed, only its
  relationship to the (now different) baseline has. For a rarely-pinned
  field (`dpo_beta`, 0.1 -> 0.2) that's 13 named recipes plus the baseline,
  14 total, down from 149. For `epochs` (all 162 recipes declare it
  explicitly, distributed `{1: 32, 2: 4, 3: 126}`), moving the default
  3 -> 1 moves 158: the 126 that pinned the old default plus the 32 that
  already pinned the new one — both now differ from a baseline that used
  to match one of them. The 4 recipes pinning `2` stay green, correctly,
  since neither the old nor the new default was ever their value. 158
  named recipes plus the baseline — a wall of red on a change that alters
  no recipe's actual behavior. This is the acceptance criterion "changing a
  schema default fails, naming the affected recipes" holding exactly as
  specified, applied to recipes whose default-shaped pin is the thing that
  moved; it is not the "one failure" case, and the fragment previously
  understated it by only showing the favorable example.

  Verified by mutation against the real catalog: a new schema field
  produces exactly 1 failure (down from 164); a recipe's own value
  changing, a recipe added without regenerating, an empty or deleted
  fixture, and a redundant default-duplicating line being deleted
  (exposure 1, still deliberately unpinned per #621) all behave exactly as
  before (#638 by @SID-6921 in #640).

### Fixed

- **The twelve non-SFT trainers (`dpo`, `kto`, `orpo`, `simpo`, `ipo`, `bco`,
  `online_dpo`, `grpo`, `ppo`, `pretrain`, `reward_model`, `embedding`) loaded
  a frozen LoRA base as float32 regardless of the checkpoint's own dtype
  (#491 by @AmirF194 in #492).** #471 fixed this for the SFT trainer; the
  same `model_kwargs` shape, missing the same key, was unchanged in the
  other twelve. None of them has a full fine-tuning branch, so every load
  there is a frozen base: the new shared `resolve_frozen_base_load_dtype()`
  keeps the checkpoint's own dtype (`torch_dtype="auto"`), except on a
  pre-Ampere CUDA card, where it now matches the float16 compute dtype
  those cards already use instead of leaving the base in bf16 storage.

- **Layer streaming now keeps its host store on CPU on Apple Silicon
  (#434 by @Amix29 in #480).**
  PyTorch 2.7+ can return an MPS tensor for
  `torch.empty(device="cpu", pin_memory=True)`, while `is_pinned()` remains
  false. Direct runtime callers could therefore place the whole frozen base in
  the MPS allocator and still report a pinned RAM store, even though the normal
  `soup train` setup already disabled pinning outside CUDA. The runtime now
  disables both optional and required pinning when the target is MPS, and
  `RamSource` independently refuses any allocation that is not genuinely CPU
  memory (or claims pinning without being pinned). MPS proceeds experimentally
  with a pageable CPU source and MPS layer buffers; CUDA pinning behaviour is
  unchanged.

- **CI and production load sites no longer assume Transformers ``dtype=``
  (#478 by @AchuthReddy-16).** ``dtype=`` on ``AutoModel*.from_pretrained`` / ``from_config`` is the
  >=4.56 rename of ``torch_dtype=``. Soup still declares
  ``transformers>=4.36.0,<5.0.0``, but the 12-cell matrix only ever installed the
  newest 4.x, so a >=4.56-only kwarg stayed green. Call sites in chat / diff /
  infer / export / merge / serve / mole routing / layer-stream runtime now pass
  ``torch_dtype=`` (still accepted on current 4.57.x). A static AST guard fails
  if a production ``AutoModel*`` load/config site reintroduces ``dtype=``. A new
  Ubuntu/3.11 ``transformers-floor`` job installs under
  ``.github/constraints/transformers-floor.txt`` using the lowest non-yanked
  resolvable Transformers version ``4.46.1`` with the lowest version in the
  declared TRL range ``0.14.0`` (``transformers==4.36.0`` is ResolutionImpossible
  against declared ``trl`` — the declared Transformers floor in
  ``pyproject.toml`` remains unchanged), runs ``pip check``, asserts both pins,
  and runs the guard. The existing 12-cell matrix is untouched.

- **`downsample` now returns at most `max_points` rows, which is what its
  docstring has always promised (#473 in #474).** The stride was
  `len(rows) // max_points` — a divisor, not a cap — so five rows with
  `max_points=2` came back with three, and the endpoint pin could add a
  fourth. Sample indices are now spread evenly across the series with both
  endpoints included, so `soup runs replay` renders exactly
  `min(len(rows), max_points)` points. Even spacing also avoids the cliff a
  ceil-based stride would introduce: a series one row over the cap keeps
  `max_points` points rather than roughly half of them. `max_points=1` is the
  one shape where both endpoints cannot fit and returns the final row, which
  is what the endpoint pin existed to guarantee. The off-by-one guard added
  in #470 is kept, renamed to `test_last_index_lands_on_last_row_no_duplicate`
  and re-pinned to the new arithmetic. Consequence of the old behaviour was a
  chart with a few more points than intended, never a wrong number.

- **`cut_ce.py` and `liger.py` now normalize path separators and match architecture
  keywords on the last path component only (#456 by @harshitthek in #458).**
  On Windows, `rsplit("/")` never split on backslashes, causing parent directory
  names (e.g. `C:\experiments\phi-3-runs\step-2000`) to over-match architecture
  keywords during fallback detection when config resolution was unavailable. In
  `liger.py`, switching from whole-path to last-component matching also drops
  org-prefix false positives (e.g. `mistralai/*`, `Qwen/*` when the model name
  does not contain the keyword) and prevents parent directories from applying the
  wrong fused kernel across POSIX and Windows. Both modules now normalize dual
  separators and strip trailing slashes deterministically.

- **`measure_gemm_tflops` now records per-repeat samples and
  `test_takes_the_best_repeat_not_the_first` verifies best-of-N selection within
  a single measurement (#444 by @harshitthek in #451).** Comparing two separate
  probe calls taken at different moments caused intermittent test failures on
  developer GPU machines under background contention. `GemmCeiling` now preserves
  `samples: tuple[float, ...]` and the test asserts `max(samples)` selection
  deterministically.

- **Duck-typed tokenizer mappings no longer raise a misleading error, and
  `data_doctor` shares the public `coerce_token_ids` helper (#441 by @AchuthReddy-16 in #447, part 2 found by @emre155).** A dict-like
  output that is not registered as `collections.abc.Mapping` used to be iterated
  as keys (`input_ids[0]='input_ids'`), sending the operator looking at their
  data; the mask path skipped the same objects and silently dropped
  `assistant_masks`. Both gates now use one mapping-like predicate, and the
  helper is public so the two modules cannot drift.

- **`soup data mix --live` handed every candidate proxy run a config it could
  not load (#442 by @blackcoderx in #445).** `_render_overlay_yaml` emitted `data.train` as a YAML
  list, the same shape #330 fixed in the recipe writer, but every `--live`
  test mocked `subprocess.run` so nothing ever loaded the overlay through the
  schema — a config the tool could not itself load read as a passing feature.
  `data.train` now renders as the single highest-weighted dataset in each
  candidate, mirroring #330's fix, with a comment noting which dataset was
  picked and why.

- **`use_cut_ce` silently did nothing for any model loaded from a local
  checkpoint directory, and conflated Phi-2 with Phi-3 (#383 by @AmirF194 in #446).**
  `apply_cut_ce()` picked the CCE patcher by matching an architecture keyword
  against the model path's last component, so `checkpoint-2000` / `my-finetune`
  / any other directory `soup merge`/`soup shrink`/`soup train` writes out
  matched nothing and CCE stayed off with no error, on a flag the user
  explicitly set. Separately, every Phi variant (`phi-2`, `phi-3`, `phi-4`)
  mapped to the same `"phi3"` patcher, even though `cut_cross_entropy` has no
  Phi-2 patcher at all (its `config.model_type` is `"phi"`, not `"phi3"`), and
  a bare `"gemma"` fallback entry dispatched to a patcher `cut_cross_entropy`
  does not have at all, crashing instead of reporting unsupported. Detection
  now resolves `AutoConfig.from_pretrained(model_name).model_type` first,
  mirroring the identical fix already shipped for Liger Kernel (#78), and
  falls back to the name-based match only when that is unavailable; Phi-2 and
  plain Gemma-1 both correctly report unsupported instead of running under
  the wrong kernel or crashing. The two call sites that separately hand-wrote
  the "no matching architecture" advisory now share one message.

- **`soup draft measure` now refuses a mismatched pair up front and no longer
  discards a completed measurement when the assisted arm fails
  (#344 by @ousamabenyounes in #409).** `measure`
  gated on `same_tokenizer()` (tokenizer vocab + probe ids), which accepts a pair
  whose tokenizers are identical but whose `config.vocab_size` differs by padded
  embedding rows (e.g. Qwen2.5 large←small) — exactly the pair `distill` refuses.
  Transformers' assisted generation gates on `config.vocab_size`, so the run died
  with "different tokenizers" deep inside `generate()`, after the expensive load,
  and because the report was written only after that arm every completed
  acceptance/plain-throughput number was thrown away. `measure` now uses the same
  `config.vocab_size` precondition as `distill` (refusing before any model loads;
  the shared `_vocab_size_of` also reads a composite model's `get_text_config()`,
  so a multimodal target like Llava is no longer refused), keeps `same_tokenizer()`
  as an additional check, and writes the report incrementally so a failing assisted
  arm leaves the acceptance rate and plain throughput on disk. The report records
  an `assisted_status` (`pending` / `complete` / `untimed` / `crash` /
  `interrupted`) so a failed arm is distinguishable on disk from a completed or
  un-run one — `pending` is what a report keeps when the process dies mid-arm
  and no handler runs, which is the case the incremental write exists for.

- **`soup export --format gptq` crashed with no calibration data and, when it
  did run, wrote a shard name the standard loader can't find
  (#338 by @AmirF194 in #475).** With no `--calibration-data`,
  `_export_gptq` called `model.quantize(tokenizer)`; auto-gptq's `quantize()`
  expects tokenized examples, not a bare tokenizer, so this failed with
  "object is not iterable". GPTQ export now requires `--calibration-data`
  up front and rejects a file with zero usable samples, since auto-gptq has
  no built-in fallback dataset (unlike AWQ). Separately, `save_quantized`
  writes its own `gptq_model-<bits>bit-<group>g.safetensors` shard, which
  `AutoModelForCausalLM.from_pretrained` does not look for; the exported
  directory now also carries a standard `model.safetensors`. The shared
  `except ImportError` blocks on both the AWQ and GPTQ paths also stopped
  reporting a fixed "not installed" string when the package itself imports
  fine but a transitive import inside it fails for an unrelated reason.

- **SmolVLM/Idefics3 vision SFT now reaches real training batches (#302 by
  @Amix29 in #488).** Soup keeps LLaVA messages and PIL images together until
  collation, converts legacy `<image>` markers to structured multimodal content,
  and lets the processor produce image-token expansion plus architecture-specific
  pixel tensors. The vision path uses the Transformers trainer with this collator so
  older supported TRL releases cannot pre-tokenize the dataset as text-only. Image
  placeholder ids are excluded from causal-LM labels, and the collator preserves a
  leading BOS whether it comes from the chat template or the tokenizer default.

- Pre-Ampere cards (T4/P100/V100/GTX 16xx — the whole free Colab/Kaggle tier)
  could crash `stream_layers: true` training with `_amp_foreach_non_finite_check_
  and_unscale_cuda not implemented for 'BFloat16'`: peft creates LoRA adapters in
  the base checkpoint's dtype while fp16 GradScaler requires fp32 gradients.
  Trainable `*lora_*` params are now cast to fp32 before optimizer creation from
  every trainer `train()` site through one shared helper (`lora.r: 0`, Spectrum
  and LISA full-FT paths are deliberately untouched so trainable memory does not
  double after the VRAM pre-flight). The #385 static scanner was **narrowed**, not
  weakened: modules that take the precision decision via the shared alignment
  helper now count as covered (#425, #429) (#429 by @lesterppo).

- **`SFTTrainerWrapper` no longer silently upcasts every load to fp32 (#339 by @blackcoderx in #471).**
  All three `from_pretrained` call sites (text/vision/audio) now pass an explicit `torch_dtype`: a
  frozen base (LoRA/QLoRA — the base never receives an optimizer step) preserves the checkpoint's
  own dtype via the shared `resolve_frozen_base_load_dtype()` (#491/#492) instead of defaulting to
  fp32 — pre-Ampere CUDA cards (T4/P100/V100/GTX 16xx/RTX 20xx) get an explicit `torch.float16`
  override there instead of bf16-storage/fp16-compute. A trainable base (`lora.r: 0`,
  `unfrozen_parameters`, `lisa_enabled` — schema-gated to modality='text') loads `torch.float32` as a
  deliberate, documented numerics choice. Measured on an H100 with Llama-3.1-8B, LoRA, frozen base:
  48,241 MiB -> 18,658 MiB peak (2.59x / 28.9 GB), byte-identical across 3 repeats — the original
  #339/#471 benchmark, carried over unchanged in this revision rather than re-measured. The full-FT
  discriminator is a single shared `is_full_finetune()`, used by both `SFTTrainerWrapper` and
  `commands/train.py`'s VRAM pre-flight classifier — previously independent copies that disagreed in
  both directions. `setup()`'s console summary label also now names LISA runs correctly instead of
  mislabeling them "LoRA applied".

- **Every DeepSpeed-capable trainer now prunes the empty LoRA optimizer group,
  not just `sft.py` (#359 in #484).** #336 fixed the failure where LoRA leaves HF's
  no-decay parameter group empty, DeepSpeed drops it, and the LR scheduler
  keeps two `base_lrs` until torch's strict `zip` raises at the first
  `lr_scheduler.step()` — but it fixed it in one wrapper. Measured before
  changing anything: 19 modules under `soup_cli/trainer/` accept a
  `deepspeed_config` and exactly one called the guard, so 18 tasks still died
  the same way under `--deepspeed` with LoRA.

  Coverage is enforced by a scan over `soup_cli/trainer/*.py` rather than a
  list of names, following `test_device_map_distributed.py` — whose own
  history is the argument, since its first version parametrized over the six
  trainers that fix had touched and passed while nine more sites still carried
  the defect. The scanner requires the guard only where a module both accepts
  a `deepspeed_config` and constructs a trainer itself, so the delegating
  `preference.py` wrapper is correctly exempt, and it carries a control
  proving the pattern can fail.

  `attach_empty_param_group_guard` now declines a trainer with no callable
  `create_optimizer` instead of raising. That is load-bearing once the guard
  is attached from eighteen wrappers rather than one: not every TRL trainer
  exposes the method, and an AttributeError there would convert a
  DeepSpeed-only defect into a crash on the ordinary path.
- **`--deepspeed my.json` is now resolved the way a preset is (#359 in #484).** A
  user-supplied config reached DeepSpeed unresolved, so none of the preset
  rewrites applied to it. The decision recorded: resolve, but only keys that
  are provably invalid for the run. `zero_hpz_partition_size` is refused by
  DeepSpeed when the world size is not divisible by it, and the fp16 quantiser
  against a `bf16` run raises `expected mat1 and mat2 to have the same dtype`
  inside `deepspeed/runtime/zero/linear.py` — neither is a preference. Since
  the documented way to customise ZeRO++ is to copy the preset JSON, which
  copies both defects, an unresolved user file inherited a crash the presets
  are already protected from. A config using none of those keys is returned by
  its own path, byte-identical; a repair is printed and written to a temp copy,
  and the user's file on disk is never modified. Malformed JSON passes through
  untouched, because DeepSpeed reports a bad config better than a traceback.

- **Transformers floor CI now tracks declared dependency metadata (#494 by @Amix29 in
  #496).** The compatibility guard derives the training lower bound from `pyproject.toml`,
  refuses a tested pin below it, requires a documented reason for a higher resolvable pin,
  and makes the workflow read exact versions from the constraints file instead of
  restating them.

- **A server crash between spawning an MCP training job and recording its pid no
  longer lets a restart double-book a second job (#506 by @AmirF194).** The one-active-execution
  cap now treats an unresolved launch as active rather than reading it as free
  capacity.

- **TensorRT-LLM export now fails immediately with a clear message instead of
  silently producing zero artifact bytes (#337 by @AmirF194 in #508).** `_export_tensorrt()`
  shelled out to `python -m tensorrt_llm.commands.convert_checkpoint`, a module
  absent from every current TensorRT-LLM release (`tensorrt_llm.commands` ships
  only `bench/build/eval/prune/refit/serve`; conversion now lives as a
  per-architecture `examples/<arch>/convert_checkpoint.py` script instead). The
  entry point is checked right after the existing `tensorrt_llm` availability
  check, before the LoRA merge and checkpoint directory are touched. Docs now
  warn that installing `tensorrt_llm` can downgrade a training environment's
  `torch`/`transformers`/`numpy`/`datasets` pins.

- Fixed layer streaming's hidden duplicate-disk cost: Soup now reuses regular Hugging Face cache files, preflights materialized-weight and shard-cache writes per volume, refuses before exhausting disk space, and explains source-fingerprint cache rebuilds (#510 by @Amix29).

- Layer streaming now admits the dense `qwen3_5` / `qwen3_5_text` decoder used by Qwen3.8-27B after native mixed-attention resident-vs-streamed parity coverage. (#514 by @Amix29 in #515)

- Layer streaming now preserves BF16 checkpoints on capable MPS runtimes instead of silently doubling the store and disk cache to FP32. (#516 by @Amix29 in #519)

- The negative llama.cpp quantizer lookup test is now isolated from binaries installed on the host `PATH`. (#518 by @Amix29 in #520)

- Fixed fresh installs with Plotext 6: `soup data stats` histograms and `soup runs`
  loss charts now dispatch to Plotext 6's Figure API while retaining the Plotext 5
  module-level path. The temporary `<6.0.0` dependency cap is lifted because both
  supported majors are covered by the compatibility layer and regression tests
  (#507, #522).

- Added `soup mcp runs reconcile --expunge-launching` so operators can safely recover
  execution capacity from stale `launching` rows without editing SQLite by hand (#524 by @Amix29 in #525).

- Layer streaming now shards an untied `embed_tokens` and `lm_head` pair separately and
  reuses one vocabulary-sized device buffer instead of keeping both matrices resident (#526 by @Amix29).
  Tied embeddings keep their existing resident one-matrix path and numerics.

- SFT now refuses examples without causal loss targets and final training states
  containing non-finite metrics or parameters before saving artifacts (#535 by @Amix29).

- Streamed LoRA adapters now preserve the exact configured base-model provenance,
  restoring automatic base detection across adapter workflows (#537 by @Amix29).

- Fully cached Hugging Face snapshots can now be materialized and sharded with
  outgoing traffic disabled, while preserving commit and blob integrity (#538 by @Amix29).

- Apple internal NVMe volumes behind APFS and Apple Fabric are now detected for
  the layer-streaming disk tier without requiring a manual override (#539 by @Amix29).

- Response-only masking now supports system-first conversations with native chat
  templates that reject transient prefixes without a user query (#543 by @Amix29).

- The final training panel now preserves the last real loss, learning rate, and
  gradient norm when Transformers emits a summary-only log event (#544 by @Amix29).

- Reproducibility receipts now record the MPS backend, Apple chip name, and
  unified-memory capacity without collecting unique hardware identifiers (#545 by @Amix29).

- Best-of-N now rejects non-finite and boolean judge scores before selecting a
  winner or emitting invalid JSONL (#547 by @Amix29 in #551).

- Best-of-N now rejects malformed prompt rows with line-numbered errors and
  records each accepted row's source line instead of silently losing data (#548 by @Amix29 in #552).

- Fixed #549 by making Best-of-N generation durable and resumable per prompt, with exactly-once
  checkpoint validation and rollback-safe, manifest-last publication that never exposes a partial
  SFT/DPO generation after an output failure (#553 by @Amix29).

- Fixed #555: two-phase Best-of-N candidate export now resumes from durable,
  authenticated per-prompt checkpoints, while offline validation and output
  materialization stream through bounded-memory disk staging. Resume binds
  prompt source lines and exact local-model content, and streamed SFT/DPO
  publication rolls back as one set on failure. Candidate export now seeds each
  prompt independently so resumed runs reproduce the same candidates; this
  changes the output of existing `--export-candidates --seed N` runs (#559 by @Amix29).

- Fixed #556: offline Best-of-N now commits SFT and optional DPO outputs through a
  final verifiable manifest, rejects unsupported online recovery options, and
  makes interrupted or mismatched generations fail closed (#557 by @Amix29).

- **`_assert_finite_training_state` no longer refuses a run over a self-corrected
  transient metric (#546 by @AmirF194 in #560).** The log_history scan now checks only
  the most recently logged value of each metric instead of raising on the first
  non-finite value found at any step, so a GradScaler warm-up nan that recovers no
  longer blocks the final save.

- Live multipack training now packs each bin to `training.batch_size * data.max_length`
  instead of `data.max_length` alone, so the configured batch size actually affects
  packing density again (#562 by @AmirF194).

- [#564](https://github.com/MakazhanAlpamys/Soup/pull/564) enables BF16 autocast for
  hardware-validated resident SFT, DPO, reward-model, and PRM training on capable
  Apple Silicon MPS runtimes. The live capability probe is shared with layer
  streaming; CPU and unvalidated MPS trainers remain FP32, while PRM retains FP32
  master weights to avoid a fatal Metal optimizer dtype mismatch (#564 by @Amix29).

- Preserve reward metadata from local GRPO datasets and keep earlier assistant turns when
  separating a final reference answer from multi-turn conversations (#566 by @Amix29).

- [#568](https://github.com/MakazhanAlpamys/Soup/pull/568) enables runtime-probed
  BF16 autocast for local Transformers GRPO/RLVR on capable Apple Silicon MPS
  runtimes. Unsupported MPS remains FP32, local generation avoids the CUDA-only
  vLLM path, and an Apple Silicon train/save/reload smoke covers deterministic
  RLVR with LoRA (#568 by @Amix29).

- `live_eval`'s four callers now forward `quantization`; `soup ship --config` reuses the
  training run's own (#367 by @AmirF194 in #570). #461 added `quantization` to `load_model_and_tokenizer` but
  nothing set it, so an NF4-trained adapter was still judged against a bf16 base.
  `make_generator`, `make_multi_generator`, `lora_probe` and `measure_logit_agreement`
  now accept and forward it. `soup ship --config soup.yaml` derives a default from
  `training.quantization` (`4bit`/`8bit`; other quant_menu formats still fall back to
  full precision) and prints which precision it loaded, now matching what it actually
  passes to `from_pretrained` (previously the fallback message claimed bf16 without
  ever setting `torch_dtype`). `advise --probe-model`, `diagnose --base-model` and
  `tunability --live` take no `--config`/`--quantization` flag yet, so they keep the
  existing default. Stamping the numerics into the verdict/evidence JSON and a
  staleness gate on mismatched-numerics evidence (issue criteria 2 and 4) stay open.

- [#574](https://github.com/MakazhanAlpamys/Soup/pull/574) corrects the MCP
  execution-tool refusal, which told operators to restart with `--allow-execute`
  and then falsely added that execution tools "are not implemented in this
  version" — execution shipped in v0.73.3 (#297 by @Srinivasan8888 in #574). The refusal now names the
  disabled tool and the flag that enables it, and the stale `build_registry`
  docstring no longer describes `allow_execute` as reserved for future tools.
  Closes #483.

- **Eval-gate LLM judge normalization uses the active rubric scale instead of
  hardcoded `/ 10.0` (#577 by @here-2007 in #578).** `_run_judge_task` in
  `eval/gate.py` divided the aggregate judge score by `10.0`, assuming a 1–10
  scale, while `DEFAULT_RUBRIC` in `eval/judge.py` uses 1–5. A perfect judge
  score of 5.0 normalized to 0.50 instead of 1.00, making typical gate
  thresholds (0.70, 0.80) impossible to satisfy and triggering false-positive
  training stops under `on_regression: stop`. Normalization now dynamically
  derives `scale_min` and `scale_max` from `evaluator.rubric["scale"]` via
  min-max scaling to `[0.0, 1.0]`, clamps out-of-bounds values, handles
  degenerate scales (`min == max`), and aligns with the existing dynamic
  normalization in `commands/ship.py`.

- [#581](https://github.com/MakazhanAlpamys/Soup/pull/581) ports two v0.73.0
  vLLM serve fixes to the SGLang backend, which had the identical pair standing.
  The prompt now applies the model's own chat template via the shared
  `build_chat_prompt` (with the legacy `User:`/`Assistant:` fallback for
  template-less models) instead of a hand-rolled third copy, so the model no
  longer sees a format it was never trained on. And `finish_reason` reports
  `"length"` when a response hits `max_tokens` (on both the sync and streaming
  paths) instead of a hardcoded `"stop"`, so a client doing continue-on-length
  can tell a truncated answer from a completed one. The tokenizer load in
  `soup serve --backend sglang` now uses the same `trust_remote_code` setting
  the SGLang runtime itself uses, so a custom-code model no longer falls back
  to the legacy prompt in silence, and the three-branch operator warning vLLM
  prints about the chat template is now printed here too. Live verification
  against a real SGLang runtime on Linux remains an open follow-up on #360 (#581 by @Srinivasan8888).

- `soup train` now warns when `training.convergence_detection` is enabled but not
yet wired into the live training loop, matching the existing honesty guard for
other advisory training-intelligence flags (#583).

- **GRPO objective variants now execute without silently falling back to stock
  TRL loss (#584 by @here-2007 in #585).** `_GRPOTrainerVariant.compute_loss()`
  previously checked `inputs` for `per_token_logps`, which is never
  pre-populated by TRL's rollout dataloader. The trainer now obtains
  per-token log probabilities in a single forward pass via
  `_get_per_token_logps_and_entropies` without duplicating forward computation
  or increasing peak VRAM, ensuring `gspo`, `dapo`, `dr_grpo`, `bnpo`,
  `two_sided`, and `rft` objectives train their intended loss formulations.

- PPO's reward model, and `task: reward_model`'s own trained model, now only inherit the
  Quant Menu quantization config when `training.quantize_reward_model` is set (#586 by @AmirF194).
  Since v0.53.0 added that flag, it was validated by its own task-scoped check but never
  read by either loader, so both tasks quantized regardless of the flag's value.

- [#588](https://github.com/MakazhanAlpamys/Soup/pull/588) makes FSDP + BNB
  4-bit training resolve quantized storage and trainable adapter
  parameters to the same floating compute dtype, preventing the integer-storage
  and mixed-dtype flattening failures that made the `llama3-70b-fsdp2` recipe
  unrunnable. The recipe now pins bf16 storage for its A100/H100 target hardware (#588 by @Faisal-Fayaz).

- Raised the `accelerate` floor to 0.27.0, the first release whose FSDP
  checkpoint save/load path can be restricted to the trainable adapter. This
  closes a remaining way a LoRA/PEFT run under FSDP could regress back to
  writing the full frozen base model into every checkpoint on an unpinned
  `accelerate` install (#352, #591) (#591 by @AmirF194).

- `_telemetry_endpoint_is_safe` could not reject private/loopback HTTPS
  endpoints because the upstream HTTPS requirement bypassed the
  scheme-conditional private-IP check in `validate_hub_endpoint`. The fix
  layers a telemetry-strict private, loopback, and link-local rejection
  on top of the existing hub sanitisation without modifying `hubs.py`
  (#593, #598) (#598 by @harshitthek).

- **`training.stream_vram_probe` docs now name each ratio's denominator (#595 by
  @umran666 in #605).** Four code sites quoted the probe series (0.992x / 0.830x)
  labelled "the real peak". They now carry BOTH series with their denominators:
  the real-run peak (0.934x at seq 5120, 0.787x at 6144) and the probe series
  (0.992x at seq 4096, 0.830x at 5120) — the gap between them is the probe
  running 12.5-14.3% above the real training step, the conservativeness the gate
  depends on. The field description also carried the withdrawn claim that the probe
  under-measures preference losses badly enough to make the gate unsafe — at the
  one matching shape it reads +13.5% high, the same safe direction it shows for
  SFT, and the `task='sft'` restriction stands because one shape is not a
  validation. Comments and field descriptions only; no behaviour change.

- **`soup serve --backend mii` now applies the served model's own chat template
  and reports the engine's real `finish_reason`** (#606 by @ARAVIND281 in #608), instead of a
  hand-rolled `System:/User:/Assistant:` prompt and a hardcoded `"stop"`. The
  backend reuses `build_chat_prompt` and `resolve_finish_reason` rather than
  keeping its own copies, and `serve` loads and passes the tokenizer for the MII
  path as it already does for vLLM. Behaviour is unchanged for a model that
  ships no chat template.

- **The reward-hack ladders' mode split is now documented, and their thresholds
  are pinned by tests (#371 fix-path item 2, by @umran666 in #611).** The H100
  gate record carried an open inconsistency — the beta ladder escalated on four
  consecutive HACK votes while the rollback ladder, requiring three, never
  fired in any arm. The code settles it: the beta ladder lives in `kl_control`
  (no rollback rung), the rollback rung lives only in `pid_lagrangian`, and the
  schema already rejects `reward_hack_rollback` under any other mode — so the
  arm that fired one structurally lacks the other, and every arm with the other
  crashed at #342 before it could run. Documented in the module docstring and
  both run paths; comments only, no behaviour change. Tests pin the remaining
  CPU-testable acceptance criteria: each ladder fires at its documented
  threshold on N consecutive HACK votes with a control at N-1, and the
  `beta == floor` degenerate case the defaults produce.

- `soup doctor` now recommends a PyTorch CUDA wheel that matches the
  driver's `nvidia-smi` CUDA version instead of always suggesting `cu121`.
  An unreadable driver header falls back to `cu121` (conservative: a parse
  failure is treated as an old driver, not a new one). Drivers at CUDA 11.8
  get `cu118`. Windows also notes that the PyPI torch wheel is CPU-only.
  (#612 by @MKnaomi2)

- [#631](https://github.com/MakazhanAlpamys/Soup/pull/631) pins the last
  open row of #273: the `74->73` branch of `setup_logging`, which was only
  ever executed incidentally depending on ambient logger state (100% branch
  coverage on one machine, 98% on another). Two hermetic tests now force the
  state: a handler without the `_soup_log_tier` tag survives reconfiguration,
  and a Soup handler with a stale tier is still removed. Dropping the
  `is not None` guard fails both tests by name (mutation demonstrated and
  reverted). `BrPart` 1 -> 0, for a reason rather than by luck (#631 by @umran666).

- **Three tests no longer fail on a clean checkout when `FORCE_COLOR` is set
  (#633 by @Srinivasan8888 in #635).** `soup recipes show` and
  `soup data mix --apply` emit syntax-highlighted output, and three assertions
  read that output raw — so on a colour-capable terminal one of them fed ANSI
  escapes to a YAML parser (`unacceptable character #x001b`) and another looked
  for `modality: text` inside a highlighted span. The assertions now normalise
  through a shared `strip_ansi` helper, and a repo-wide AST guard fails on any new
  raw-output assertion against a syntax-highlighted command, so the fourth
  occurrence of this failure class is also the last one that has to be found by
  hand.

- `--resume auto` (and a direct `--resume <path>`) now finds MLX checkpoints.
  `_resolve_checkpoint` only recognized `checkpoint-N` directories (the
  transformers/unsloth shape); mlx-lm's tuner saves step-numbered
  `NNNNNNN_adapters.safetensors` files instead, so an MLX run's own output
  never matched and training silently restarted from scratch every time.
  `MLXSFTTrainerWrapper.train()` now loads the located checkpoint's adapter
  weights before training starts. This is a weights-only warm start, not a
  full resume — mlx-lm's LoRA trainer exposes no optimizer state or step
  count, so the step count and data position both restart from zero
  regardless of how far the checkpoint got, and the run says so. Also
  fixes MLX `--resume` for any config with `experiment_name` set, which
  the first version of this fix missed (#634 reported and diagnosed by
  @imahsanali, fixed by @SID-6921 in #639).

- **`[train]` now declares the torch floor that actually binds, and doctor's
  copy is pinned to it (#636 by @umran666 in #641).**
  `[train]` said `torch>=2.3.0` while requiring `transformers>=5.16.1`, whose
  own torch extra forces `torch>=2.5` — the declared floor could never bind,
  and `soup doctor` carried an unpinned second copy on top. The extra now
  declares `torch>=2.5.0` with the reason stated beside it; doctor keeps a
  literal that `tests/test_issue636_torch_floor.py` pins to the declaration,
  and a drift guard compares the declared floor against the *declared*
  transformers floor's real metadata on the `transformers-floor` CI job (which
  installs exactly 5.16.1 under the constraints file), skipping with a reason
  anywhere the installed transformers is a different release rather than
  passing vacuously.
  Closes #636.

- **`soup sweep` no longer crashes with an `IndexError` when the run grid is empty (#628 by @Srinivasan8888 in #643).**
  The unknown-parameter precheck added in #628 reads `combinations[0]` to probe the grid, so
  a `--max-runs` value that leaves no combinations — `--max-runs -1`, for instance — raised
  `IndexError: list index out of range` from inside the guard rather than reaching the sweep.
  Nonsense input either way, but the previous behaviour was an empty results table and exit 0,
  and an unhandled traceback is a worse answer than that. The probe is skipped when the grid is
  empty, restoring the old response. Bounding `--max-runs` at `ge=1` would reject the input
  outright and is the better long-term fix; it is a pre-existing CLI contract change and is
  left for its own change.

- `stream_source: auto` now falls back to the NVMe disk tier when the RAM store
  fits `MemAvailable` but the store plus resident extras would exceed Soup's
  physical-host safety ceiling, and forced `stream_source: ram` refuses that
  case during pre-flight instead of letting the kernel OOM-kill the process
  (#622 by @ousamabenyounes in #644).

- **`soup sweep --dry-run` now validates instead of returning before the
  config loads — and forecloses a latent Python 3.10 crash in the sweep
  pre-flight (#642 by @umran666 in #645).**
  `--dry-run` exits before `load_config` ever ran, so the #627 unknown-key
  warning and the #628 `--param` pre-flight were both unreachable under it:
  the issue's typo'd `quantizaton` printed a grid and exit 0 with no warning,
  while `soup train --dry-run` reported it. This is a stated behaviour change:
  `--dry-run` now means "validate, then print the grid without running
  anything", both paths share the single load, and the severity split is
  unchanged — warn-only exits 0, a typo'd sweep parameter exits 1 before
  printing a grid it can never run. On the execution path the same validation
  now happens before the confirmation prompt instead of after it. Making the
  pre-flight reachable exposed a latent crash in it: the probe walks a full
  `model_dump()`, which visits `training.preference_loss_weights`
  (`dict[str, float]`). On Python 3.10 a PEP-585 alias passes
  `isinstance(c, type)`, and whether the subsequent `issubclass` raises
  `TypeError` depends on the pydantic release — `ModelMetaclass` inherits
  the raising check from `ABCMeta` through 2.10.x (verified: every real
  sweep died with `issubclass() arg 1 must be a class` on 3.10.11 +
  pydantic 2.10.6) and defines its own tolerant `__subclasscheck__` by
  2.13.4 (verified under the same interpreter), so the crash was live
  across the unmasked part of the supported `pydantic>=2.0.0` range on the
  oldest supported Python, and a current pydantic masks it — which is also
  how #628 shipped green: its unit tests fed the walker hand-built dicts
  that never reached such a field. `_nested_models` now filters
  parameterized generics via `typing.get_origin`; real-command tests cover
  each branch and the clean-config control, and the filter has a
  deterministic killer (an ABCMeta stand-in recreating the pre-override
  metaclass, pinning the real schema field) plus a control so it cannot be
  "fixed" by never walking.
  Closes #642.

- **Live training panel now labels the GPU figure as `GPU peak:` and reads `max_memory_allocated()` instead of `memory_allocated()` (#650 by @YuriPerro in #652).**
  `on_log` fired between steps, after activations and gradients were freed, so the
  sampled `memory_allocated()` was the inter-step trough -- on the reproducing run
  `5.8/15.9 GB` while `nvidia-smi` showed the device fully occupied.
  The fix uses `max_memory_allocated()`, the lifetime peak, and changes the label
  to `GPU peak:` so the figure is self-describing. `reset_peak_memory_stats()` is
  intentionally not called here: that reset is process-global and would clobber the
  grad-accum advisor's own peak reading at `callback.py:566`.

- **The CUDA batch probe now refuses on a measured peak, not only on a raise
  (#649 by @YuriPerro in #654).**
  Under the WDDM driver (native Windows, WSL2) the allocator spills to host
  memory instead of raising `OutOfMemoryError`, so `make_cuda_probe_fn`'s only
  fit criterion, "the step did not throw", approved a batch 16 that ran an
  order of magnitude slower in shared memory and `pick_batch_size` cached it.
  The probe now reads `max_memory_allocated` after the synthetic step and
  refuses when it exceeds what this process can reach on the device
  (`mem_get_info` free plus what it already holds), mirroring the counter and
  basis `decide_measured_fit` documents for layer streaming; an OOM surfacing
  at a synchronize point as `AcceleratorError` / `RuntimeError("out of
  memory")` is classified as OOM instead of propagating as misconfiguration;
  and the cache key carries a version tag so entries written by the old probe
  are ignored. The WDDM case is covered by a mocked-torch test that needs no
  GPU, with a fitting control beside it.
  Closes #649.

### Security

- **Telemetry and webhook SSRF guards now reject abbreviated, decimal, hex,
  and octal IPv4 forms (#600 by @here-2007 in #604).** `_is_private_or_link_local()`
  in `utils/hubs.py`, `utils/hf.py`, and `utils/webhooks.py` previously delegated
  exclusively to `ipaddress.ip_address()`, which raises `ValueError` on non-canonical
  IPv4 representations like `127.1`, `2130706433`, `0x7f000001`, and `0177.0.0.1`.
  On Linux (glibc), the OS resolver (`getaddrinfo`/`inet_aton`) parses these forms,
  allowing crafted `--slack-url https://2852039166/` webhooks or telemetry overrides
  to reach internal network addresses and cloud metadata (`169.254.169.254`). All three
  guards now normalise non-standard IPv4 literals via in-process `socket.inet_aton()`
  and strip trailing FQDN dots before verifying against `ipaddress.ip_address()`.

- **The SGLang serve backend now obeys the `--trust-remote-code` gate instead of
  loading every model with it enabled (#360 by @Srinivasan8888 in #619).**
  `serve.py` resolves the v0.36.0 default-deny gate once for every backend and
  passes the result to vLLM, but `_serve_sglang` was called without it and
  `create_sglang_runtime` hardcoded `trust_remote_code=True` at both
  `sgl.Runtime` call sites, so `soup serve --backend sglang` executed a model's
  custom repo code whether or not the user opted in. The warning panel said so,
  but a notice is not a gate. `create_sglang_runtime` now defaults to `False`
  like the vLLM path, and the resolved value reaches the runtime and the
  tokenizer load alike. **Behaviour change:** a custom-code model on the SGLang
  backend now fails to load without `--trust-remote-code` rather than silently
  running its code.

- **Telemetry SSRF guard now applies a tiered architecture with primary allowlist,
  internal TLD rejection, and DNS defence in depth (#599 by @here-2007 in #624).**
  `_telemetry_endpoint_is_safe()` previously only checked literal IP formats, allowing
  hostnames resolving to private addresses (`10.0.0.1.nip.io`, `localtest.me`) or internal
  TLDs (`metadata.internal`) to bypass validation. The guard now validates via three tiers:
  (1) a primary allowlist for canonical PostHog domains (`*.posthog.com`) with zero DNS
  lookups, (2) static string rejection of internal TLD suffixes (`.local`, `.internal`,
  `.lan`, `.home`, `.corp`, `.intranet`) and literal private IPs, and (3) defence-in-depth
  DNS resolution for custom non-default FQDNs that fails closed on resolver errors/timeouts
  and rejects private or loopback target IPs.

- **The OTLP tracing endpoint validator now rejects abbreviated, decimal, hex,
  and octal IPv4 forms, closing a bypass `#600`/`#604` never reached (#616 by
  @SID-6921 in #625).** `utils/tracing.py`'s `_is_private_ip` predated the
  #604 fix and was never folded into it: it had no `socket.inet_aton`
  fallback. The loopback spellings (`https://127.1:4317` and friends) were
  permitted anyway; the reachable hole was **non-loopback** private ranges in
  alternate encodings — `https://2852039166:4317` and `https://0xa9fea9fe:4317`
  both resolve to `169.254.169.254`, the cloud-metadata address, and both were
  accepted while the dotted-quad form was refused. Same for `10.0.0.5` as
  `167772165` / `0xa000005` / `012.0.0.5` / `10.5`, and `192.168.1.1` as
  `3232235777`. All refused now; `127.0.0.1`, `localhost` and ordinary
  hostnames still pass. Found while consolidating the SSRF
  loopback/private-host predicate, which had drifted into three copies of the
  function and six of the host set across `hf.py`, `hubs.py`, `webhooks.py`,
  `loop_stages.py`, `qr_url.py` and `tracing.py` — issue #616 undercounted, and
  an AST walk during review found the real number. All six call sites now
  import one definition from the new `utils/net_guard.py`, and a guard test
  walks `src/soup_cli/` for any function whose name ends in
  `is_private_or_link_local` or constant ending in `LOOPBACK_HOSTS` — matching
  the leading-underscore spelling every original copy used — so a reappearing
  duplicate fails the build. `loop_stages._endpoint_is_local`, a seventh copy
  of the parsing logic guarding the deploy canary, had the same abbreviated-IPv4
  gap; it now shares `net_guard.parse_ip_literal` while keeping its own
  narrower policy, since adopting the wider predicate would change what that
  surface trusts as a local target rather than only where the logic lives.
  **The consolidation is otherwise behaviour-preserving.** The shared
  predicate's `is_reserved` / `is_multicast` clauses do not change any caller's
  outcome: in `hf.py` and `hubs.py` both branches raise, so the predicate
  selects an error message rather than an accept/reject decision, and
  `qr_url.py` imports only the host set. `loop_stages` becomes marginally more
  permissive — it is an allow-list gate, so teaching it `parse_ip_literal`
  makes it recognise 26 further spellings as local — but every one of those
  canonicalises to an address it already accepted, so the set of endpoints Soup
  will POST to is unchanged.

## [0.73.3] - 2026-08-18

### Added

- **`eval.ship.noise_floor` is now committable to `soup.yaml` (#406 by @ousamabenyounes in #410).** Every
  `soup ship` gate-policy flag was settable in a committed config and read back by
  `--config` — except `--noise-floor` (added in v0.73.2), which had no field and
  could only be passed on the command line, so a team enforcing a floor in CI had
  to hand-edit the workflow. `ShipConfig` gains a bounded `noise_floor`
  (`[2, 10]`, imported from `ship_verdict` so the schema and the CLI validator
  cannot disagree; bool-as-int rejected), wired with the same CLI > config >
  default precedence as the other five flags. As a live-measurement input it is
  measured only when producing evidence and refused under `--evidence` exactly as
  the flag is; it follows `forgetting_threshold` in being excluded from the
  recipe `config_sha`, so setting a floor never invalidates evidence.

- **A ready-made `qwen3.5-4b-pretrain` recipe for continued pre-training of
  `Qwen/Qwen3.5-4B-Base` (#278 by @Faisal-Fayaz in #422).** The recipe uses plaintext data, one epoch,
  QLoRA 4-bit quantization, and the established continued-pretraining defaults.

- **A ready-made `deepseek-v4-flash-grpo` recipe for GRPO reasoning training
  with `deepseek-ai/DeepSeek-V4-Flash` (#279 by @Faisal-Fayaz in #432).** The recipe combines the
  established GRPO defaults with MoE LoRA and gradient checkpointing.

- **`soup mcp serve --allow-execute` can now actually execute, behind a single-use
  confirmation token (#297 by @CODING-DARSH in #393).** `train_start` and `export` issue a
  short-lived, server-generated token bound to the plan and to the execution kind;
  `train_execute` / `export_execute` accept **only** that token — no command, no argv, no
  shell string, no client-supplied environment — and the server launches the planned Soup
  CLI command with `shell=False`, `stdin=DEVNULL`, output to `.soup/mcp-runs/<run_id>.log`.
  Two integrity properties close the gap between planning and running: the config is
  **snapshotted** at plan time and executed from the copy, so the file cannot be edited
  underneath the run, and `digest_file` now walks a directory tree by content (sorted
  relative paths + per-file hash, symlink refusal, bounded) rather than by mtime+size,
  which did not change when a file *inside* a protected directory was rewritten — so a
  model could be swapped between plan and execution and revalidation still passed. Token
  consumption and capacity acquisition happen before `Popen`, so a failed spawn requires a
  fresh plan rather than enabling replay. Runs go through the existing `ExperimentTracker`,
  so `soup runs` sees what MCP started.

### Changed

- **The MLX SFT dispatch route no longer imports the Transformers SFT wrapper
  before choosing a backend (#394 by @Shutaru in #431).** The wrapper is import-light
  today, but the standalone `soup-cli[mlx]` runtime no longer depends on it
  remaining so. An additive Apple Silicon CI job verifies that `mlx` and
  `mlx-lm` import, the PyTorch/TRL training stack is absent, and a one-step real
  CLI SFT run completes. This hardens the runtime boundary; it does not claim to
  resolve the still-unpinned torch-present hang reported in #394.
- **`soup card` now links the ML-BOM and the in-toto/SLSA attestation — they are
  first-class registry artifact kinds (#309 by @ousamabenyounes in #420).** `bom` and `attestation` were the
  two compliance documents `soup card` could not surface: `RegistryStore` had no
  such kinds, so there was no way to attach them. Added both to the valid-kind
  set and a `--attach-to-registry <id>` flag to `soup bom emit` and
  `soup attest emit` (mirroring the existing `--attach-to-registry` pattern);
  once attached they appear in the card's artifact table for free. Signed
  attestations also attach the detached `.sig` sidecar. The flag needs
  `--output` (omitting it exits 2), and a requested registry attachment failure
  exits 1 after preserving files already emitted.

### Fixed

- **Assistant-only masking no longer mistakes `BatchEncoding` keys for token
  ids (#430 by @Shutaru in #439).** Tokenizer mapping outputs are read through `input_ids`, and
  tensor-like ids are normalised to Python integers before they reach the
  collator; missing or non-integer ids now fail loudly instead of building a
  garbage label mask. A template that returns an all-zero assistant mask while
  assistant messages exist now falls back to incremental rendering, avoiding a
  silent all-`-100` no-op training run. The same mapping assumption was removed
  from the data doctor.

- **`soup data mix --optimize` wrote a recipe `soup train` could not load
  (#330 by @blackcoderx in #440).** `render_mix_recipe_yaml` emitted `data.train` as a YAML list of
  every searched dataset, but `DataConfig.train` is typed `str`, so the recipe
  failed to load with `data -> train: Input should be a valid string`.
  `data.train` now renders as the single highest-weighted dataset from the
  search; the full ranked weight/path breakdown is kept as a comment above it
  so no information is lost, and the comment explains that `data.interleave`
  is not yet consumed by training.

- **Layer streaming now verifies that every trainable LoRA parameter has real
  storage after PEFT attaches the adapter (#433 reported by @lesterppo, fixed by @Faisal-Fayaz in #435 and #437).** PEFT 0.18 creates streamed
  adapters on `meta` for Soup to materialise, while PEFT 0.19 may create them as
  real tensors immediately, so `materialize_meta_adapters()` returning `0`
  cannot distinguish a healthy no-op from a missed adapter. The streamed build
  now enforces the actual postcondition and refuses to install the runtime if a
  trainable `lora_*` parameter remains on `meta`, naming the stranded parameter
  instead of allowing a silent no-training run.

- **Windows process liveness misread a process that exited with code 259 as
  still running, forever (#424 by @blackcoderx in #436).** `GetExitCodeProcess` returns `STILL_ACTIVE`
  (259) for a genuinely running process, but 259 is also a legal exit code, so
  a child that exited *with* 259 was indistinguishable from one still running.
  That silently defeated `ExperimentTracker`'s reconcile-on-read (#401) and
  could wedge the MCP server's one-active-execution cap (#402) shut, refusing
  every subsequent execution with no error an operator could act on. The check
  now waits on the process handle with `WaitForSingleObject(handle, 0)`, which
  is signalled the instant the process exits regardless of its exit code,
  falling back to the exit-code read only if the wait itself fails. The two
  byte-identical copies of this primitive in `experiment/tracker.py` and
  `mcp_server/execution.py` are now one shared `utils/process_liveness.py`.

- **`soup train --no-reexec` now prints the flags you actually typed (#372 by @AchuthReddy-16 in #415).**
  The advisory `accelerate launch` command is derived from the same argv the
  auto-reexec would have used, so `--fsdp` / `--deepspeed` / `--config` (and the
  rest of the run-shaping tail, including `--name` and `--replay`) cannot silently
  fall off the printed hint. Following that line used to train without FSDP.

- **`detect_device()` and `get_gpu_info()` now recognise Apple Silicon MLX (#423 by @harshitthek in #428).**
  Previously on Apple Silicon, `detect_device()` only probed PyTorch MPS and fell back
  to `'cpu'`, triggering a false `Warning: 4bit quantization is not supported on CPU`
  alert and silently downgrading `quantization` from `4bit` to `none`. Device and GPU
  info detection now accept an optional `backend=` parameter: passing `backend="mlx"`
  resolves to `'mlx'` with the chip name (e.g. `Apple Silicon (Apple M2 Max)`), preserves
  4-bit quantization intact for `mlx-lm` pre-quantized models, reports Apple unified
  memory in telemetry, and skips the CUDA-shaped analytical VRAM preflight on MLX runs.
  The preflight skip is pinned by `TestHardwareFitGateIsMlxAware` (mirrors the
  streaming-aware gate test). The known-limitation warning in
  `docs/backends-and-ops.md` is replaced with the resolved behaviour.

- **A run whose watcher died was reported `running` forever (#401 by @ousamabenyounes in #407).**
  `ExecutionManager._watch` runs as a `daemon=True` thread, so when the MCP
  server process exits it is killed without unwinding and `finish_execution`
  never runs — `ExperimentTracker` kept the run at `running` with no watcher
  left to correct it, training the operator to ignore the one status field that
  guards against a second concurrent run. The tracker now reconciles on read:
  when a `running` row carries a pid whose process is gone, `list_runs` /
  `get_run` rewrite it to `terminated` with an unknown (`None`) exit code — an
  unknown outcome is never recorded as success, and the richer
  `completed`/`failed` terminal statuses are left untouched. `soup runs` no
  longer hardcodes `running` for a non-running row.
- **Layer streaming now accepts a fast virtio/cloud disk instead of refusing it
  as an HDD (#365 by @ousamabenyounes in #411).** `detect_disk_kind` trusted `/sys/block/<dev>/queue/rotational`,
  which a paravirtual (virtio) block device defaults to `1` with no media hint —
  so a genuinely NVMe-backed cloud disk (measured 1.5 GB/s read) was classified
  `hdd` and denied the disk-overflow tier, the very audience the tier targets.
  `rotational=0` stays authoritative (solid state); when the flag is unreliable
  (`rotational=1`), the media type is now decided by a bounded, O_DIRECT
  sequential-read measurement rather than the flag — NVMe-class throughput
  (>= 1 GB/s) earns the tier, a genuinely slow disk still classifies `hdd` and is
  still refused (160 seeks/step, plan P11). A new `training.stream_disk_kind`
  override (`nvme`/`ssd`/`hdd`) is the escape hatch for the case where even that
  is wrong; it prints what it overrode beside what was detected. Disk detection
  may write a small scratch file next to the streamed shards to run the probe.

- **The one-active-execution cap could double-book after a server restart (#402 by @ousamabenyounes in #408).**
  `ExecutionManager._active_run_id` is in-memory, so a restarted MCP server
  started with an empty slot, saw free capacity, and would launch a second
  training while a child from the previous server (which survives a client
  disconnect, #297) was still using the GPU. The cap is now gated on a
  persisted run whose recorded pid is still alive: a live prior child blocks a
  new execution across restarts, while a stale record whose process is gone
  frees the slot rather than wedging it shut. `docs/commands.md` now states the
  actual scope of the cap.

- **`soup ship --noise-floor` now measures the leg-1 task axis in the judge modes,
  so a judge-scored win smaller than the judge's own noise no longer counts (#403 by @ousamabenyounes in #419).**
  The floor was measured in `--task-mode metric` only; `judge_score` / `pairwise`
  printed a warning and left leg 1 at a 0.0 floor, i.e. the exact blindness the flag
  exists to remove. It is now measured: `judge_score` scores the base side N times
  through the judge, and `pairwise` judges the base model against itself (expected
  win-rate 0.5, so the spread is directly measured, not inferred). Because those
  repeats fold in the judge's own sampling noise, the floor is labelled
  `decode + judge` on the panel and stamped `judge_inclusive` in the evidence/JSON
  block so it is never read as a decode-only number. `--help` states the extra
  judge-API cost.

- **`training.bnb_4bit_use_double_quant` was validated but never read — every 4-bit
  path Soup builds the `BitsAndBytesConfig` for hardcoded double-quantization to `True`,
  so `bnb_4bit_use_double_quant: false` passed validation and was silently ignored (#321 by @ousamabenyounes in #418).**
  The flag is now threaded through the three call sites Soup owns — the resident loader
  (`build_quantization_config_for_loader`), the layer-streaming path (`stream_setup` reads it
  once and passes the SAME value to the sharder and the meta skeleton, so streamed-vs-resident
  bit-exactness cannot drift), and the 4bit save path (`soup merge --no-double-quant` for the
  `4bit` / `4bit_forced` save formats). The Unsloth loader is out of scope: `FastLanguageModel`
  builds its own `BitsAndBytesConfig` internally with double-quant hardcoded on and exposes no
  override, so that path cannot honour the flag. The schema field is now tri-state
  (`Optional[bool]`, unset = `None`): unset resolves to the shipped default (double-quant on),
  so a run that never set the flag trains identically, but because the resolved config now
  carries the field, `soup ship --evidence` provenance and `soup lock check` will report a
  one-time fingerprint drift on a previously-unset config even though the model numerics are
  unchanged. The config-load footgun (`=true` requires `quantization: 4bit`) fires only on an
  explicit `true`, and unset serializes as `None`, so a dumped-and-reloaded config no longer
  trips it.
- **`soup env check` now flags an installed package that violates Soup's own
  declared version bound (#368 by @ousamabenyounes in #421).** `pip install "soup-cli[serve-fast]"` (vllm)
  into a training venv silently pushes `transformers` past the `<5.0.0` cap Soup
  declares, producing an environment Soup's own metadata says is unsupported with
  no warning at any point. `env check` audits installed versions against the
  bounds read from package metadata — not a hardcoded copy, since a second copy
  of the cap is exactly the drift this catches — and exits 3 when one is violated,
  independent of any lock file. An `extra == "X"` requirement is Soup's bound
  only when `[X]` was opted into, which the running environment cannot confirm,
  so its marker is evaluated and it is skipped rather than raised as a false
  positive — except for the ABI-relevant `TRACKED_PACKAGES` set, which since the
  v0.71.0 deps-split lives in metadata *only* under `extra == "train"/"all"/
  "dev"` and therefore includes the `transformers <5.0.0` case #368 was reported
  about. A package is counted once even when its bound is restated under several
  extras. If `packaging` is somehow unimportable the audit still degrades to a
  clean report so `env check` keeps working, but now says so rather than
  reporting a silent clean it never verified. The bounds audit no longer
  short-circuits the lock diagnostic (both
  print). `packaging` is a declared dependency now — the audit degraded to a
  silent "clean" without it, the wrong direction for a checker.
  `docs/serving-and-export.md` states the separate-environment guidance for
  `[serve-fast]`.

- **`kl_control` rewrote the trainer's β/kl_coef on every step, including a `hold`, so a
  non-acting run was numerically identical to `log_only` (#371 by @AmirF194 in #414).** `_run_bang_bang` called
  `_apply_coefficient` unconditionally; on a `hold` the controller writes back the value
  already there, which is a no-op value but not a no-op write. The write is now skipped when
  the bang-bang step holds, and the mitigation log records `mitigation_status` (`held` /
  `acted` / `released`) so a non-acting step is distinguishable from an acting one without
  parsing the free-text `action` reason.

- **`extract_mcq_letter` scored zero for `\boxed {A}` — whitespace between the
  command and the brace (follow-up to #357, by @ousamabenyounes in #396).** The shipped `\boxed\{` regex tolerates spaces
  *inside* the braces but not between `\boxed` and `{`; LaTeX permits it there and
  models emit it, so `\boxed { C }` still read as no answer and was not rescued by
  the cue tier either. The boxed-letter regex now allows `\s*` after the command.

- **`soup draft distill --steps N` now delivers ~N optimiser steps instead of
  N/4.44 (#364 by @ousamabenyounes in #399).** The epoch count that realised `--steps` divided the request
  by `rows // batch_size`, ignoring that `val_split` (0.1) removes rows from
  training and `gradient_accumulation_steps` (4) micro-batches make one
  optimiser step — both divide the budget, so every distill run trained for
  roughly a fifth of the requested steps, silently. Epochs are now derived from
  the effective steps-per-epoch, the emitted config pins the run shape
  (`val_split`, `gradient_accumulation_steps`) so the arithmetic and the trainer
  cannot drift, and the pre-flight prints the resolved step count.


- **`MitigationLogWriter` silently dropped every record once its parent directory
  vanished mid-run (#343 by @ousamabenyounes in #398).** `record()` reopens the log per call and swallowed the
  `OSError` from `open("ab")` with a bare `return`, so when a shared temp root was
  cleaned by another process the controller kept acting while its log quietly stopped
  growing — the run completes while its evidence goes missing, the failure shape this
  project treats as the worst kind. The writer now recreates the directory and retries
  the write, and surfaces the loss once via a warning naming the path. `record()` still
  never raises, so a vanished log never takes down the training run.

## [0.73.2] - 2026-08-15

`soup ship`'s leg 2 is the project's differentiator, and it was lying in both
directions: two of its suites ranked by the wrong thing, one whole failure
direction had no detector at all, and a caller error was indistinguishable from
a regression. Every item below was reproduced on the dev box against the
shipped v0.73.1 code before a line was changed.

### Added

- **`soup ship --noise-floor N` — measure what the instrument can resolve
  before calling a delta significant.** Greedy decoding is not deterministic on
  GPU: measured on an H100, the same model with no adapter over five runs spread
  **0.015 strict / 0.020 format-blind**, and `soup ship` compared against a 0.05
  threshold without ever telling the operator what its own instrument could
  resolve — four of six paired deltas in that session sat inside the floor. With
  the flag, the BASE model is re-run N times (2–10), the per-axis floor is
  `max − min` across the repeats, it is printed beside the verdict, and every
  axis is then gated at `max(threshold, floor)`. Leg 1 must clear the task
  axis's floor too. The `max` is load-bearing in both directions: a floor above
  the threshold widens the gate to what is actually measurable, and a floor
  below it must never tighten the gate behind the operator's back. Opt-in, so no
  existing run changes; costs N extra base passes. **The leg-1 floor is measured
  in `--task-mode metric` only** — in the judge modes a repeat would fold the
  judge's own sampling noise into a number presented as decode noise, so the run
  warns and leaves leg 1 at a 0.0 floor instead. Carried caveat, because it
  bounds the claim: n=3, one model, one dataset. It **sizes** the effect; it does
  not calibrate a threshold.
- **A benign-prompt axis, `mini_over_refusal`, joins the default leg-2 suite
  (7 → 8) (#317).** Leg 2 flagged a *drop* in `mini_safety` and had no reverse,
  so a tune that refuses everything registered as a monotone safety improvement
  with no ceiling on how useless the model became. Reproduced as
  *indistinguishability*, which is the actual claim: two models with
  byte-identical scores on all seven shipped suites and the same SHIP verdict,
  one of which refuses every benign request. The new suite is 40 hand-authored,
  original benign-but-scary-sounding requests (defensive security, first aid,
  sysadmin verbs like "kill"/"terminate", fiction with dark themes) scored as
  *fraction NOT refused*, so the existing regression rule catches over-refusal
  with no change to the verdict engine. Paired with `mini_safety` neither axis
  can be gamed alone. Same caveat as above: 40 prompts and one greedy pass size a
  gap, they do not calibrate a threshold.
- **Semantic Stratified Splitting for `soup data split` (#388).**
  `--stratify-semantic` and `--num-clusters` partition splits proportionally
  across semantic groups (TF-IDF + K-Means) so a whole topic cannot land
  entirely in one split. `scikit-learn` is a declared member of the `[data]`
  extra and a missing import refuses with the install command rather than
  silently falling back. 50k-row cap, an explicit error for a pure-stop-word
  corpus, and a warning when `--num-clusters` is passed without the flag.
  Contributed by [@Deadpool2000](https://github.com/Deadpool2000).
- **`soup mcp serve --allow-execute` (#391).** A stronger opt-in than
  `--allow-mutating`, which it implies, plus the `_refuse_execute` handler the
  execution tools will use. `train_start` and `export` stay **plan-only** —
  nothing in this slice executes anything, and the help text says so in the
  present tense. `build_registry(allow_mutating=True)` still returns the same 16
  tools. Contributed by [@CODING-DARSH](https://github.com/CODING-DARSH).

### Fixed

- **`extract_mcq_letter` did not know `\boxed{C}`, and the MCQ prompt never
  asked for a letter (#357).** Meta-Llama-3.1-8B-Instruct scored **0.423 on
  `mini_mmlu` — below a 0.5B** — while scoring 1.000 on two other MCQ suites. Of
  15 failures, 8 boxed the right letter and 6 boxed a *value* because nothing in
  the prompt asked for a letter. Reproduced here at the extreme: a stub that
  answers every item CORRECTLY in the boxed-letter style scored **0.000** on
  `mini_mmlu` and `mini_common_sense`. Both halves are needed — the extractor
  alone is worth +8 items, the prompt alone **0**, together 0.423 → 0.731 and the
  inversion disappears. The new tier fires **only** when the box holds a single
  A–J letter: reading `\boxed{4}` as "option 4" would be a wrong credit, not a
  repair. Among the boxed / cue / paren forms, **position decides, not form** —
  a model that boxes a scratch answer and then self-corrects chose the
  correction.
- **`mini_tool_call` ranked by brace hygiene (#346).** The 8B named the right
  tool **40/40** and scored 0.225: it emitted three opening braces and two
  closing ones, the whole-string parse failed, the bounded scan returned the
  INNER object, and the scorer rejected it for having no `"function"` key. The
  gate suite's own unwrapping layer now restores the envelope for an object
  carrying **both** `name` and `arguments` — requiring `arguments` is what stops
  an echoed `{"name", "description"}` menu entry from scoring, i.e. from
  crediting copying. The missing brace is the model's own output, **not**
  truncation; that attribution was believed and shipped in `c87fd00` before a
  budget sweep disproved it.
- **`score_bundled_suite` returned `0.0` for a non-callable `gen` (#355).** On
  the three behavioural suites it scored 0.0 while the MCQ suites raised
  `TypeError` — and in leg 2 a 0.0 reads as "the model failed every item" →
  DON'T SHIP, so a caller error was indistinguishable from a regression **and
  failed in the direction that looks like a finding**. Both branches now raise.
  A *callable* that misbehaves is still a failed item, which is the correct
  existing contract.
- **The verdict panel silently ate its own leg-1 marker.** `render_ship_panel`
  built its header as `... [{won_str}]`, and a bare `[no win]` is valid Rich
  markup for an unknown tag — so the panel never printed "won"/"no win" on any
  release up to and including v0.73.1. The plain-text rubric, which has no markup
  parser, printed it correctly the whole time, which is why it went unnoticed.
  Found while rendering the new noise-floor panel.
- **Untrusted names could drive the terminal.** Benchmark and noise-floor axis
  names come from an `--evidence` JSON file, and `rich.markup.escape` neutralises
  Rich's `[...]` syntax and nothing else, so a raw ESC byte survived it. Both
  render paths now strip C0/DEL first, matching the `_for_terminal` pattern
  already used in six other command modules.
- **The `--evidence` round-trip and the MCP `ship_evidence` tool now agree.** A
  verdict decided against a measured floor does not replay without it, so the
  floor is part of the evidence schema (#312's output-is-input property), and
  **both** readers honour it — a schema extended in one consumer and not the
  other would have made the same file replay to different decisions through the
  CLI and through `soup mcp serve`.
- **A duplicated `#392` CHANGELOG entry.** PR #388 branched before `18a278a`
  moved that entry into `[0.73.1]`, and the merge re-introduced it under
  `[Unreleased]`, listing the same fix twice.

### Security

- **An evidence-supplied noise floor is bounded and never silent.** A floor
  widens the gate, so `"floors": {"mini_mmlu": 1.0}` in an evidence file masks
  any possible drop on that axis. This does not cross a new trust boundary —
  anyone who can edit that file can already write `{"base": 0.9, "tuned": 0.9}`
  and force a SHIP outright — but it is a far quieter edit to miss in review, and
  `soup ci init` wires `ship --evidence` as a PR merge gate. Values are therefore
  bounded to `[0.0, 1.0]`, the mapping is capped at 50 axes / 256-char names
  (mirroring the CLI's own limits), a malformed block is **refused rather than
  dropped** (a dropped floor replays as a different verdict), and any floor that
  exceeds `--forgetting-threshold` is announced — on stderr by the CLI, and in
  the returned payload by the MCP tool, whose stdout is the JSON-RPC channel.
  Neither reader is the quiet one.

### Changed

- **`decide_ship` now canonicalises the `TaskWin` it stores**, as it already did
  for the benchmark deltas. It recomputes leg 1 from the raw scores, so a
  `TaskWin` built without the floor would otherwise have rendered "won" beside a
  DON'T SHIP decided with it.
- **`--baseline` snapshots taken before this release are on a different scale**
  for `mini_mmlu`, `mini_common_sense` and `mini_tool_call`, because their
  scorers changed. A stored baseline skips the live base run, so it would be
  diffed against a freshly-scored tuned model — measured on an *unchanged* model
  the jumps are 0.423 → 0.731 and 0.225 → 1.000, far larger than the 0.05 gate.
  `soup ship` now warns by name when `--baseline` supplies a stored score for an
  affected suite. `mini_instruction` and `mini_arithmetic` are unaffected and are
  deliberately not named: neither carries a single-letter answer, so the prompt
  cue and the option-letter extractor never touch them (verified, 0 of 24 and 0
  of 36 items).

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

[Unreleased]: https://github.com/MakazhanAlpamys/Soup/compare/v0.77.0...HEAD
[0.77.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.76.0...v0.77.0
[0.76.1]: https://github.com/MakazhanAlpamys/Soup/compare/v0.76.0...v0.76.1
[0.76.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.75.2...v0.76.0
[0.75.2]: https://github.com/MakazhanAlpamys/Soup/compare/v0.75.1...v0.75.2
[0.75.1]: https://github.com/MakazhanAlpamys/Soup/compare/v0.75.0...v0.75.1
[0.75.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.74.0...v0.75.0
[0.74.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.73.3...v0.74.0
[0.73.1]: https://github.com/MakazhanAlpamys/Soup/compare/v0.73.0...v0.73.1
[0.71.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.70.0...v0.71.0
