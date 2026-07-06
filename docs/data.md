# Data Engineering

[← Back to the Soup README](../README.md)

> Data formats, the Axolotl/LF-parity pipeline, data tools, synthetic generation/forge, quality scorecards, trace tooling, remote datasets, mixing, recipe DAGs, and the v0.69 data-engineering surfaces.

**Contents:**

- [Data Engineering Pro](#data-engineering-pro)
- [Production Trace Ecosystem (`soup ingest`)](#production-trace-ecosystem-soup-ingest)
- [Prompt Mining (`soup prune-prompt`)](#prompt-mining-soup-prune-prompt)
- [Active-Learning Sampler (`soup data active-sample`)](#active-learning-sampler-soup-data-active-sample)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Data Augmentation](#data-augmentation)
- [Trace-to-Preference](#trace-to-preference)
- [Config Migration](#config-migration)
- [Data Formats](#data-formats)
- [Data Pipeline Pro](#data-pipeline-pro)
- [Data Tools](#data-tools)
- [Demo Datasets (`soup data demo`)](#demo-datasets-soup-data-demo)
- [Trace-to-Preference: LLM-Judge Filter](#trace-to-preference-llm-judge-filter)
- [Synthetic Data Forge](#synthetic-data-forge)
- [Data Quality Scorecard](#data-quality-scorecard)
- [Remote Datasets (S3 / GCS / Azure / OCI)](#remote-datasets-s3--gcs--azure--oci)
- [Data Recipe DAG](#data-recipe-dag)
- [Data Mixing Optimizer (BETA)](#data-mixing-optimizer-beta)
- [AOT Tokenization with `soup data preprocess`](#aot-tokenization-with-soup-data-preprocess)
- [Data Recipe DAG Runner (`soup data recipe --execute`)](#data-recipe-dag-runner-soup-data-recipe---execute)

---

## Data Engineering Pro

The v0.69.0 release ships 5 surfaces that turn dataset prep from "throw a JSONL at the trainer" into a first-class engineering workflow.

```bash
# dbt-for-SFT — DAG of dataset transforms with incremental materialization
cat > build.yaml << 'EOF'
models:
  - {name: raw, kind: incremental, source: data/raw.jsonl, transform: identity}
  - {name: filtered, kind: incremental, refs: [raw], transform: filter_low_quality}
  - {name: tokenized, kind: incremental, refs: [filtered], transform: tokenize}
EOF
soup build build.yaml --dry-run                 # validate topology + plan
soup build build.yaml --output-dir built/       # live materialise (v0.71.6)

# Expectations suite — Great Expectations for chat data
cat > suite.yaml << 'EOF'
expectations:
  - {name: expect_no_pii}
  - {name: expect_token_length_between, args: {min_tokens: 16, max_tokens: 4096}}
  - {name: expect_no_refusal_pattern}
EOF
soup expect data.jsonl suite.yaml   # exit 3 on suite failure

# Magpie synthetic data — chat-template-prefix harvest (live, v0.71.6)
soup data gen-magpie --base meta-llama/Llama-3.1-8B-Instruct \
    --provider ollama --target 1000 --output magpie.jsonl --quality-filter

# Persona-Hub diversity — prompt × persona × style matrix sampling
soup data persona-mix --prompts prompts.jsonl --n 500 --output mixed.jsonl

# Brain-rot detector (arXiv 2510.13928) — refuses to train on excessive slop
soup data brain-rot data.jsonl --strict --max-major-fraction 0.10

# Best-of-N rejection sampling (v0.71.31) — sample N locally, a judge picks the winner
soup data best-of-n --base HuggingFaceTB/SmolLM2-135M-Instruct \
    --prompts prompts.jsonl --n 8 --judge ollama://llama3.1 \
    -o best_of_n.jsonl --emit-pairs pairs.jsonl

# Evol-Instruct (WizardLM depth/breadth, v0.71.31) — grow instruction diversity
soup data evolve --input seeds.jsonl --provider ollama --model llama3.1 \
    --strategy depth --rounds 2 -o evolved.jsonl
```

Every command applies the project-wide TOCTOU policy (`os.lstat + S_ISLNK` symlink rejection before any open) and cwd containment via the shared `paths.enforce_under_cwd_and_no_symlink` helper. All five are LIVE: `soup build` materialises with five built-in transforms (`identity` / `drop_empty` / `lowercase` / `strip` / `dedup_exact`) and SQLite-tracked incremental re-transform (v0.71.6); `soup data gen-magpie` harvests via raw completion against `--provider ollama|vllm` (loopback-only; `anthropic` rejected — no raw-completion endpoint, v0.71.6).

### Custom Transforms

Use a dotted-path string (`module.path:function_name`) as the ``transform``
value to import a custom transform at build time:

```yaml
models:
  - {name: clean, kind: table, source: data/raw.jsonl, transform: my_pkg.transforms:clean_row}
  - {name: enriched, kind: table, refs: [clean], transform: my_pkg.transforms:enrich}
```

The target function must accept exactly two positional arguments (`row`, ``config``)
and return a ``dict`` or ``None``. Soup resolves the dotted path lazily (the module
is imported only when the build actually runs) and caches the result so repeated
references to the same path do not re-import.

**Trusted-input posture.** The dotted-path syntax causes Soup to import an
arbitrary Python module and call a function from it. Treat ``transform`` values
as *trusted input*: do not feed untrusted or operator-controlled YAML into ``soup
build`` on shared CI hosts. An attacker who controls the manifest can execute
arbitrary code during the build phase. If you must accept user-supplied manifests,
validate them against a allowlist of permitted transform paths before passing
them to the resolver.


## Production Trace Ecosystem (`soup ingest`)

Closing the data flywheel without leaving your existing observability stack. `soup ingest` parses JSONL exports from every major SaaS dashboard and emits a normalised trace stream that `soup data from-traces` (v0.26) consumes.

```bash
# Six supported sources — adapters for the major SaaS vendors + raw OTel
soup ingest --source langfuse     --logs ./langfuse-export.jsonl --output traces.jsonl
soup ingest --source langsmith    --logs ./langsmith-runs.jsonl
soup ingest --source helicone     --logs ./helicone-requests.jsonl
soup ingest --source openpipe     --logs ./openpipe-export.jsonl
soup ingest --source otel         --logs ./otel-spans.jsonl
soup ingest --source openai-stored --logs ./oai-stored-completions.jsonl
```

The CLI never makes the network call — operators export from their SaaS dashboard or vendor API, then point `soup ingest` at the local file. Auth env vars (`LANGFUSE_KEY` / `LANGSMITH_API_KEY` / `HELICONE_API_KEY` / `OPENPIPE_API_KEY` / `OPENAI_API_KEY` / `OTEL_EXPORTER_OTLP_HEADERS`) are advisory only — Soup surfaces which one is unset so operators wire creds before the SaaS-side export. A PII reminder fires on every ingest run (matches v0.26.0 Trace-to-Preference policy).


## Prompt Mining (`soup prune-prompt`)

Production LLM apps often pin a multi-paragraph system prompt to every request. Fine-tuning with that prefix wastes tokens (the model learns to copy what's already in context). `soup prune-prompt` finds the longest character prefix shared by ≥ 95% of rows and strips it, so the FT model internalises the behaviour instead.

```bash
soup prune-prompt --input traces.jsonl --output pruned.jsonl --min-frequency 0.95
```

Binary-search over up-to-32 candidate templates finds the longest qualifying prefix (a longer threshold-meeting prefix may exist beyond the universal one — Soup does not early-exit on the 100% match). Two-pass file read with a 100 000-row DoS cap.

**Tokenizer-aware mode (v0.71.5).** Pass `--tokenizer <id-or-path>` (a HuggingFace repo id, a local path, or anything `AutoTokenizer.from_pretrained` accepts) to detect the shared prefix in *token* space and decode only the remaining ids:

```bash
soup prune-prompt --input traces.jsonl --output pruned.jsonl --tokenizer Qwen/Qwen2.5-0.5B
```

Char-level stripping can cut a BPE multi-byte sequence in half when the shared prefix ends mid-token; token-aware pruning finds the longest shared *token-id* prefix and decodes the remainder, so the boundary always lands on a real token. Per-row encoding is capped at 50 000 tokens. Omit `--tokenizer` to keep the original character-level behaviour.


## Active-Learning Sampler (`soup data active-sample`)

Surface the most uncertain prod traces for human review. Two modes via the input data shape:

- **Single RM:** `rm_score: 0.5` → uncertainty 1.0 (peak); `rm_score: 0.0` or `1.0` → uncertainty 0.0.
- **Dual RM:** `rm_scores: [s1, s2]` → uncertainty = `|s1 - s2|` (pairwise disagreement).

```bash
soup data active-sample --input traces.jsonl --output for-review.jsonl --budget 100
```

The output JSONL is a drop-in prompt set for `soup eval human` (v0.19). Budget is bounded `[1, 100 000]`.

**Webhooks (v0.71.5).** `soup ingest`, `soup prune-prompt`, `soup ab`, and `soup data active-sample` all accept `--slack-url` / `--discord-url` and POST a one-line summary on completion through the same SSRF-hardened validator as `soup drift-alarm` (scheme allowlist, loopback-only HTTP, RFC1918 / link-local / reserved / multicast rejected; the post never raises, so a flaky webhook can't fail the command). `soup ab` only fires when the sequential test actually decides (`reject_h0` / `accept_h0`), not while it's still `continue`-ing.


## Synthetic Data Generation

Generate training data using LLMs:

```bash
# Generate using OpenAI API
soup data generate --prompt "Create math word problems" --count 100 --format alpaca

# Use a different model
soup data generate --prompt "Medical Q&A pairs" --model gpt-4o --count 500

# Deduplicate against existing data
soup data generate --prompt "..." --count 200 --dedup-with existing.jsonl

# Use seed examples to guide style
soup data generate --prompt "..." --seed examples.jsonl --count 100

# Use a local OpenAI-compatible server (soup serve, Ollama, etc.)
soup data generate --prompt "..." --provider server --api-base http://localhost:11434/v1
```

### Multi-Provider Support

```bash
# Generate via local Ollama instance
soup data generate --prompt "..." --provider ollama --model llama3.1
soup data generate --prompt "..." --ollama-model llama3.1  # shorthand

# Generate via Anthropic Claude API (set ANTHROPIC_API_KEY env var)
soup data generate --prompt "..." --provider anthropic --model claude-3-haiku-20240307

# Generate via local vLLM server
soup data generate --prompt "..." --provider vllm --model meta-llama/Llama-3.1-8B-Instruct
```

### Domain Templates

```bash
# Code instruction pairs (Python, JS, Go, Rust, Java)
soup data generate --prompt "..." --template code --language Python --task-type function

# Multi-turn conversations
soup data generate --prompt "..." --template conversation --turns 6 --topic "science"

# QA from context document
soup data generate --prompt "..." --template qa --context document.txt

# Preference data (DPO/KTO/ORPO)
soup data generate --prompt "..." --template preference --pref-task dpo

# Chain-of-thought reasoning (GRPO)
soup data generate --prompt "..." --template reasoning --domain math
```

### Quality Pipeline

```bash
# Auto-validate after generation (remove malformed entries)
soup data generate --prompt "..." --validate

# Auto-filter by quality (coherence scoring)
soup data generate --prompt "..." --filter

# Auto-dedup (MinHash, requires: pip install 'soup-cli[data]')
soup data generate --prompt "..." --dedup

# Full quality pipeline: validate + filter + dedup
soup data generate --prompt "..." --quality-pipeline
```


## Data Augmentation

Augment an existing dataset using an LLM — rephrase for diversity, translate for multilingual coverage, or apply a style transform.

```bash
# Rephrase each example N times for more diversity
soup data augment ./data/train.jsonl --strategy rephrase --count 3 \
  --output ./data/train_augmented.jsonl

# Translate into multiple languages
soup data augment ./data/train.jsonl --strategy translate --lang es,fr,de \
  --output ./data/train_multilingual.jsonl

# Style transfer (formal / casual / technical / etc.)
soup data augment ./data/train.jsonl --strategy style --styles formal,casual \
  --output ./data/train_styled.jsonl

# Local provider (Ollama / vLLM) — loopback-only, pick the model + base URL
soup data augment ./data/train.jsonl --strategy rephrase --count 2 \
  --provider ollama --model qwen2.5:0.5b --output ./data/train_local.jsonl
```

Works with any provider supported by `soup data generate` (OpenAI, Ollama, vLLM, local server). `--model` and `--base-url` select a specific local model/endpoint; the Ollama/vLLM paths are loopback-only (SSRF-hardened). `--count` is capped at 10; `--lang` and `--styles` each capped at 10 entries × 32 chars.


## Trace-to-Preference

Harvest DPO / KTO-ready preference pairs from your production inference logs — no manual labeling.

```bash
# LangChain logs + thumbs-up signal
soup data from-traces --logs ./logs/langchain.jsonl \
  --format langchain --signal thumbs_up --output prefs.jsonl

# OpenAI API logs + regeneration signal (second response wins)
soup data from-traces --logs ./logs/openai.jsonl \
  --format openai --signal regeneration --output prefs.jsonl

# Soup-serve logs + user-edit signal (edited response wins over original)
soup data from-traces --logs ./logs/soup-serve.jsonl \
  --format soup_serve --signal user_edit --output prefs.jsonl

# Preview generated pairs before training
soup data review prefs.jsonl --sample 10
```

**Supported log formats:** `langchain`, `openai`, `soup_serve`
**Supported signals:** `thumbs_up` (rating-based), `regeneration` (latest wins), `user_edit` (edited wins)

Trace files are capped at 100,000 lines to prevent OOM on production logs. A PII warning panel appears on every run — redact sensitive fields before harvesting.


## Config Migration

Switch from other tools with one command:

```bash
# Import from LLaMA-Factory
soup migrate --from llamafactory llama3_lora_sft.yaml

# Import from Axolotl
soup migrate --from axolotl axolotl_config.yml

# Import from Unsloth notebook
soup migrate --from unsloth finetune.ipynb

# Preview without writing
soup migrate --from llamafactory config.yaml --dry-run
```

Automatically maps model, LoRA, training params, quantization, and task type. Warns about unsupported features.


## Data Formats

Soup supports these formats (auto-detected). Files can be JSONL, JSON, CSV, Parquet, or TXT.

**Alpaca:**
```json
{"instruction": "Explain gravity", "input": "", "output": "Gravity is..."}
```

**ShareGPT:**
```json
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
```

**ChatML:**
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

**DPO / ORPO / SimPO / IPO (preference pairs):**
```json
{"prompt": "Explain gravity", "chosen": "Gravity is a force...", "rejected": "I don't know"}
```

**KTO (unpaired preferences):**
```json
{"prompt": "Explain gravity", "completion": "Gravity is a force...", "label": true}
```

**LLaVA (vision):**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this."}, {"from": "gpt", "value": "A cat."}]}
```

**ShareGPT4V (vision):**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nExplain this chart."}, {"from": "gpt", "value": "Revenue growth."}]}
```

**Plaintext (pre-training):**
```json
{"text": "Raw text document for continued pre-training..."}
```
Or use `.txt` files directly (one document per line).

**Embedding (sentence embedding pairs/triplets):**
```json
{"anchor": "What is Python?", "positive": "Python is a programming language."}
{"anchor": "What is Python?", "positive": "A programming language.", "negative": "A type of snake."}
```

**Audio (speech + conversation):**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe."}, {"role": "assistant", "content": "Hello world."}]}
```

**PRM (process reward, stepwise-supervised):**
```json
{"prompt": "Solve 2+2", "completions": ["First, add", "Result is 4"], "labels": [true, true]}
```

**Pre-tokenized (skip tokenize stage):**
```json
{"input_ids": [1, 2, 3, ...], "labels": [-100, 2, 3, ...], "attention_mask": [1, 1, 1, ...]}
```
Use with `data.format: pre_tokenized` and `data.tokenized_path: ./.soup-tokenized/<key>` after running `soup data preprocess`.

**Input/Output (template-free, segment-level loss control):**
```json
{"segments": [{"text": "Q: hi", "label": false}, {"text": "A: hello", "label": true}]}
```

**Video:**
```json
{"video": "clip.mp4", "messages": [{"role": "user", "content": "Describe this clip."}]}
```

**Multimodal (typed content parts — text / image / audio / video in one message):**
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What's in this?"}, {"type": "image", "url": "x.png"}]}]}
```


## Data Pipeline Pro

Soup speaks the same dataset surface as Axolotl + LlamaFactory + Unsloth — remote URIs, streaming, sharding, multi-dataset interleaving, vocab expansion, and document ingestion all live in one schema.

**Remote datasets** (schema gate live; fsspec backend wiring lands in v0.42.1):

```yaml
data:
  train: s3://my-bucket/datasets/train.jsonl   # also gs:// gcs:// az:// abfs:// abfss:// oci://
  streaming: true
  buffer_size: 8192
  shards: 4
```

**Multi-dataset interleave:**

```yaml
data:
  interleave: { strategy: probs, probs: [0.7, 0.3] }   # also: concat / under / over
  eval_on_each_dataset: true
```

**Vocab expansion + advanced masking:**

```yaml
data:
  add_new_tokens: ["<reasoning>", "</reasoning>"]
  new_special_tokens: ["<|tool_call|>"]
  resize_vocab: true
  mask_history: true
  split_thinking: true            # Qwen3-style <think> reasoning-block masking
  image_min_pixels: 256
  image_max_pixels: 4096
  image_resize_algorithm: bicubic
  video_fps: 24
  video_maxlen: 32
  video_dir: ./videos
```

**AOT preprocessing:**

```bash
# Tokenize once, reuse the cache across runs.
soup data preprocess soup.yaml --output ./.soup-tokenized

# Then in soup.yaml:
#   data:
#     format: pre_tokenized
#     tokenized_path: ./.soup-tokenized/<16-char-cache-key>
```

**Document ingestion (PDF / DOCX / MD / TXT → JSONL):**

```bash
soup data ingest report.pdf --output report.jsonl
soup data ingest README.md
soup data ingest notes.docx
```

**Custom prompt strategies (schema only — runtime invocation in v0.42.1):**

```yaml
data:
  prompt_strategy: my_pkg.transforms:rephrase
```


## Data Tools

```bash
# Inspect a dataset
soup data inspect ./data/train.jsonl

# Validate format (auto-detects if --format not specified)
soup data validate ./data/train.jsonl
soup data validate ./data/train.jsonl --format alpaca

# Convert between formats
soup data convert ./data/train.jsonl --to sharegpt --output converted.jsonl

# Merge multiple datasets
soup data merge data1.jsonl data2.jsonl --output merged.jsonl --shuffle

# Remove near-duplicates (requires: pip install 'soup-cli[data]')
soup data dedup ./data/train.jsonl --threshold 0.8

# Extended statistics (length distribution, token counts, languages)
soup data stats ./data/train.jsonl

# Filter by quality (perplexity + coherence scoring)
soup data filter ./data/train.jsonl --coherence 0.3
soup data filter ./data/train.jsonl --perplexity 500 --coherence 0.3
soup data filter ./data/train.jsonl --score-only  # add scores without filtering
```


## Demo Datasets (`soup data demo`)

Tiny JSONL fixtures bundled with Soup so you can warm up `soup train` without
hunting for data:

```bash
# List available bundles
soup data demo

# Copy one into the current directory
soup data demo alpaca_demo --output ./alpaca.jsonl
```

Bundles: `alpaca_demo`, `sharegpt_demo`, `dpo_demo`, `grpo_demo`. Output path
must stay under cwd; existing files are not overwritten.


## Trace-to-Preference: LLM-Judge Filter

`soup data from-traces --judge` filters harvested preference pairs through an LLM judge:

```bash
soup data from-traces \
  --logs ./prod-traces.jsonl --format langchain --signal thumbs_up \
  --output ./prefs.jsonl \
  --judge --judge-provider ollama --judge-model llama3 \
  --min-confidence 0.7
```

The judge scores `chosen` and `rejected` independently against its rubric (default helpfulness/accuracy/safety on a 1-5 scale). Pairs whose normalised `(chosen - rejected)` confidence falls below `--min-confidence` are dropped. Per-pair backend exceptions are counted (not crashed) and reported. Provider allowlist `{openai, server, ollama}` validated at the CLI boundary; SSRF protection on `--judge-api-base` carries over from `soup eval judge`.


## Synthetic Data Forge

Multi-stage synthetic data pipeline with full provenance — every synthetic row links back to the source document, the judge call, and the filter score:

```bash
# Pipeline: chunk docs → judge → active-prune → JSONL + provenance manifest
soup data forge \
    --docs ./my_docs/ \
    --task sft \
    --target-rows 1000 \
    --uncertainty-threshold 0.4 \
    --output forge_dataset.jsonl \
    --provenance forge_provenance.json
```

Three tasks supported: `sft` (Q&A pairs), `preference` (chosen/rejected), `tool` (tool-call hypotheses). Active learning prunes rows whose judge reply is too close to the source chunk (low Jaccard distance), keeping only uncertain / informative samples. The provenance manifest is a separate JSON file mapping every row id to `{source_doc, judge_id, chunk_id, filter_score}` so you have a complete audit trail for compliance.

Document discovery is one level deep over `.txt` / `.md` / `.json` / `.jsonl`; dotfiles + symlinked directories are skipped. All paths are cwd-contained, all writes are atomic via staged-tempfile + `os.replace`, and write targets are rejected if they're symlinks. **Judge providers are live**: `--judge-provider ollama` (localhost-only), `--judge-provider anthropic` (env-only API key), `--judge-provider vllm` (scheme-validated). Per-call judge exceptions logged at DEBUG.

**Alternative teacher hubs (v0.71.5).** `--hub modelscope|modelers` pre-fetches the `--teacher` from that hub when the teacher is a routable repo id (`owner/name`); `--hub hf` (default) is a no-op and leaves the teacher as a provenance label. If `--hub` is non-HF but `--teacher` is not a repo id (e.g. the default `local-judge`), Soup prints a loud yellow warning rather than silently dropping the flag.


## Data Quality Scorecard

Composite, lightweight data-quality triage — no GPU, no 200 MB Presidio model:

```bash
# Single-shot composite scorecard
soup data score --input training.jsonl

# Standalone subcommands — JSONL-in, enriched JSONL-out
soup data pii          --input training.jsonl --output pii_flagged.jsonl
soup data toxicity     --input training.jsonl --output tox_flagged.jsonl --threshold 0.1
soup data langdetect   --input training.jsonl --output tagged.jsonl
soup data educational  --input training.jsonl --output scored.jsonl
soup data decontaminate --input training.jsonl --benchmarks mmlu,gsm8k,humaneval --output clean.jsonl
```

The scorecard reports PII flagged, toxic flagged, language distribution, mean educational value, and decontamination removed. PII detection uses a narrow ReDoS-hardened regex set (email / phone / SSN / credit-card) with a 50 KB pre-cap on every input. Language detection is a stopword heuristic across six languages. Toxicity is a keyword baseline; the Llama-Guard-3-1B variant + FineWeb-Edu classifier ship behind `[data-pro]` extras. Decontamination uses n-gram containment against benchmark corpora: use `--benchmarks mmlu,gsm8k` for built-in allowlist, or `--benchmark-file custom_benchmark.jsonl` for your own corpus.


## Remote Datasets (S3 / GCS / Azure / OCI)

Point `data.train` at any object in the v0.42.0 fsspec allowlist and `soup train` will stream it through `fsspec.open` after running the URI through the same SSRF-hardened validator used everywhere else in Soup (bucket regex, no userinfo / query / fragment):

```yaml
data:
  train: s3://my-bucket/datasets/train.jsonl
  format: alpaca
  streaming: true       # opt-in HF datasets streaming with shuffle
  buffer_size: 10000    # shuffle buffer (requires streaming=true)
```

Recognised schemes: `s3://`, `gs://`, `gcs://`, `az://`, `abfs://`, `abfss://`, `oci://`. The matching backend SDK (`s3fs` / `gcsfs` / `adlfs` / `ocifs`) is lazy-imported — install only what you need or grab the convenience extra:

```bash
pip install soup-cli[remote]   # fsspec + s3fs + gcsfs + adlfs
```

Materialised rows are capped at 1M to defend against pathological remote objects; use a local split for larger jobs.


## Data Recipe DAG

```bash
soup data recipe my_recipe.yaml
```

```yaml
nodes:
  - name: seed1
    kind: seed
    config: {path: prompts.jsonl}
  - name: llm1
    kind: llm_text
  - name: judge1
    kind: judge
  - name: samp1
    kind: sampler
edges:
  - [seed1, llm1]
  - [llm1, judge1]
  - [judge1, samp1]
```

Closed node-kind allowlist (`seed` / `llm_text` / `code` / `judge` / `validator` / `sampler`); Kahn's topological sort via `collections.deque` (deterministic, O(N+E)); cycle / self-loop / duplicate-edge / dangling-edge / unknown-kind rejection. `_MAX_NODES=256`, `_MAX_EDGES=1024`, `_MAX_FILE_BYTES=1MiB`. The recipe file must stay under cwd and **must not be a symlink** (`os.lstat + S_ISLNK` TOCTOU defence). Live offline runner against a local model lands in v0.45.1.


## Data Mixing Optimizer (BETA)

Search for the dataset mixture weights that minimise eval loss on a short proxy run.

```bash
soup data mix --optimize --budget 1h \
    --datasets dolma.jsonl,wikipedia.jsonl,arxiv.jsonl \
    --num-probes 8 --output mix_recipe.yaml
```

Writes a YAML recipe with a `data.interleave` block you can splice into your `soup.yaml`. `--budget` accepts `60s` / `5m` / `1h` / `24h`. Per-candidate proxy failures are isolated (DEBUG-logged, sentinel high loss recorded) so a single OOM combo does not abort the whole search; `partial=True` is surfaced in the report when the budget cap trips mid-loop.

Re-apply a previously written recipe:

```bash
soup data mix --apply mix_recipe.yaml
```

Live wiring of the proxy training loop into a short `soup train` run is the v0.48.1 deliverable; v0.48.0 ships a synthetic offline proxy (quadratic penalty around the uniform simplex) so the budget tracker, optimiser surface, and recipe writer can be exercised without GPUs. `scikit-optimize` is opt-in via `OptimizerProtocol`; the default fallback is a deterministic Dirichlet sampler.


## AOT Tokenization with `soup data preprocess`

Pre-tokenize your dataset once and cache Arrow shards keyed by
`(dataset, tokenizer, max_length, format)`:

```bash
soup data preprocess soup.yaml --output ./tokenized_cache
```

SFT and Pretrain trainers short-circuit at schema validation when
`format: pre_tokenized` + `tokenized_path: ./tokenized_cache` is set, eliminating
the per-epoch tokenization tax. Cache keys ensure resume safety; partial runs pick
up from the last completed shard.


## Data Recipe DAG Runner (`soup data recipe --execute`)

Execute a Data Recipe DAG end-to-end:

```bash
soup data recipe path/to/recipe.yaml --execute --output ./out
```

Six node kinds now run live: **seed** (JSONL load), **llm_text** (LLM generation via
any provider), **code** (execution via RLVR sandbox), **judge** (binary scoring),
**validator** (regex or JSON schema), **sampler** (deterministic selection). Checkpoint
written per node; resume rehydrates from per-node sidecars. Failed rows logged with
redacted reasons (paths stripped, capped at 256 chars).


## Fine-tune Doctor (`soup data doctor`)

Chat-template compatibility report — catches the top *silent* fine-tuning failures
before a single training step:

```bash
soup data doctor ./data/train.jsonl --model meta-llama/Llama-3.1-8B-Instruct

# Render N sample rows with per-token trained/masked colouring, through the REAL
# collator path (answer-only / per-message-train-field / RAFT span-mask)
soup data doctor ./data/train.jsonl --model meta-llama/Llama-3.1-8B-Instruct --show-mask 5
```

Eight checks, same OK/MINOR/MAJOR taxonomy as `soup diagnose` (exit 0 on OK/MINOR,
exit 2 on MAJOR): `chat_template` (tokenizer has one), `template_render` (renders
cleanly on a sample), `generation_markers` (`{% generation %}` support),
`eos_in_labels` — the **#1 "model never stops generating" bug**: every trained
assistant turn must actually contain an EOS/EOT token, checked across the *whole*
trained span, not just the last turn — `bos_duplication` (template + tokenizer both
prepending BOS), `system_role` (Mistral-style templates that reject a leading system
turn), `unknown_roles`, and `truncation_risk` (p95 rendered length vs
`data.max_length`). `--train-on-responses-only` / `--train-on-messages-with-train-field`
select the same masking strategy `soup train` would use, so the report and
`--show-mask` preview can never disagree about what's actually trained.


## Preference-Data Linter (`soup data lint`)

Catches the top silent degradations in DPO/ORPO/SimPO/IPO/BCO/KTO preference data:

```bash
soup data lint ./data/prefs.jsonl
soup data lint ./data/prefs.jsonl --model meta-llama/Llama-3.1-8B-Instruct  # exact token-length bias, not word count
```

Five checks: `length_bias` — the **#1 silent DPO degradation**: `chosen`
systematically longer than `rejected`, reported as a Cohen's d effect size —
`label_imbalance` (KTO desirable:undesirable ratio), `near_duplicates`
(MinHash/LSH, reuses the `soup data dedup` kernel; requires
`pip install 'soup-cli[data]'`, degrades to an advisory skip otherwise),
`identical_pairs` (`chosen == rejected` — zero preference signal), and
`prompt_leak` (the prompt echoed verbatim inside the completion, a common
synthetic-data pipeline bug). Same OK/MINOR/MAJOR taxonomy and exit codes as
`soup data doctor`.


