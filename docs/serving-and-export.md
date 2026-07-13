# Serving, Inference & Export

[← Back to the Soup README](../README.md)

> The OpenAI-compatible inference server, batch inference, benchmarking, merge/export (GGUF/ONNX/TensorRT/AWQ/GPTQ), the Anthropic Messages endpoint, speculative decoding, deploy autopilot, the Web UI, and Agent Forge.

**Contents:**

- [Merge LoRA Adapter](#merge-lora-adapter)
- [Export to GGUF](#export-to-gguf)
- [Batch Inference](#batch-inference)
- [Inference Benchmarking](#inference-benchmarking)
- [Inference Server](#inference-server)
- [Web UI](#web-ui)
- [Inference Server Trace Log](#inference-server-trace-log)
- [Soup Quantize — Ergonomic Export Alias](#soup-quantize--ergonomic-export-alias)
- [Llama.cpp Proxy](#llamacpp-proxy)
- [Tail-Latency Stats + Tool-Call Timer](#tail-latency-stats--tool-call-timer)
- [Web UI Plugin Registry + Env Knobs](#web-ui-plugin-registry--env-knobs)
- [Deploy Autopilot](#deploy-autopilot)
- [Agent Forge](#agent-forge)
- [HF Space SDK Auto-Pick](#hf-space-sdk-auto-pick)
- [Anthropic Messages API Converter](#anthropic-messages-api-converter)
- [Server-Side Tools](#server-side-tools)
- [Anthropic Messages Endpoint](#anthropic-messages-endpoint)
- [Train + measure your own draft (`soup draft`)](#train--measure-your-own-draft-soup-draft)
- [N-gram Speculative Decoding](#n-gram-speculative-decoding)
- [Server-Side Tool Endpoints](#server-side-tool-endpoints)

---

## Merge LoRA Adapter

Merge a LoRA adapter with its base model into a standalone model:

```bash
# Auto-detect base model from adapter_config.json
soup merge --adapter ./output --output ./merged

# Specify base model and dtype
soup merge --adapter ./output --base meta-llama/Llama-3.1-8B --dtype bfloat16
```


## Export to GGUF

Export models to GGUF format for use with [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
# Export LoRA adapter (auto-merges with base, then converts)
soup export --model ./output --format gguf --quant q4_k_m

# Export with different quantizations
soup export --model ./output --format gguf --quant q8_0
soup export --model ./output --format gguf --quant f16

# Export a full (already merged) model
soup export --model ./merged --format gguf

# Specify llama.cpp path manually
soup export --model ./output --format gguf --llama-cpp /path/to/llama.cpp
```

Supported quantizations: `q4_0`, `q4_k_m`, `q5_k_m`, `q8_0`, `f16`, `f32`

### ONNX Export

Export models to ONNX format for use with [ONNX Runtime](https://onnxruntime.ai/):

```bash
pip install 'soup-cli[onnx]'
soup export --model ./output --format onnx
soup export --model ./output --format onnx --output ./model_onnx
```

### TensorRT-LLM Export

Export models to TensorRT-LLM format for high-throughput GPU inference:

```bash
pip install 'soup-cli[tensorrt]'
soup export --model ./output --format tensorrt
soup export --model ./output --format tensorrt --output ./model_trt
```

### BitNet 1.58 TQ1_0 GGUF Export (live in v0.71.20)

Export a BitNet 1.58-bit model as a `TQ1_0` (1.58-bit ternary) GGUF via
llama.cpp's convert→quantize pipeline. Both the `bitnet` alias and the explicit
`tq1_0` flavour map to the same `TQ1_0` quantization (no importance matrix is
needed — ternary weights export directly):

```bash
soup export --model ./output --format bitnet   # → TQ1_0 ternary GGUF
soup export --model ./output --format tq1_0     # same flavour
soup export --model ./output --format bitnet --llama-cpp /path/to/llama.cpp
```

A built llama.cpp toolchain is required; LoRA adapters are auto-merged before
export. See [Performance & Quantization → BitNet](performance-and-quantization.md)
for the training side.

After export, use with Ollama manually or auto-deploy:
```bash
# Manual (3-step)
echo 'FROM ./my-model.q4_k_m.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model

# Auto-deploy (1-step)
soup export --model ./output --format gguf --deploy ollama --deploy-name my-model
```

### Deploy to Ollama

Deploy a GGUF model directly to your local [Ollama](https://ollama.com/) instance:

```bash
# Deploy a GGUF model
soup deploy ollama --model ./output/model.q4_k_m.gguf --name soup-my-model

# Deploy with system prompt and parameters
soup deploy ollama --model ./model.gguf --name soup-chat \
  --system "You are a helpful assistant." \
  --template chatml \
  --parameter temperature=0.7 \
  --parameter top_p=0.9

# Export + deploy in one command
soup export --model ./output --format gguf --deploy ollama

# List Soup-deployed models
soup deploy ollama --list

# Remove a model
soup deploy ollama --remove soup-my-model
```

Auto-detected chat templates: `chatml`, `llama`, `mistral`, `vicuna`, `zephyr` (or `auto` to infer from soup.yaml).


## Batch Inference

Run a model on a list of prompts and save results:

```bash
# JSONL input (each line: {"prompt": "..."})
soup infer --model ./output --input prompts.jsonl --output results.jsonl

# Plain text input (one prompt per line)
soup infer --model ./output --input prompts.txt --output results.jsonl

# Custom generation settings
soup infer --model ./output --input prompts.jsonl --output results.jsonl \
  --max-tokens 512 --temperature 0.3
```

Output is JSONL with `prompt`, `response`, and `tokens_generated` fields. Shows a progress bar and throughput summary.


## Inference Benchmarking

Quickly measure your model's generation speed and memory footprint before deployment:

```bash
# Benchmark local speed and VRAM usage on 3 automatically generated prompts
soup bench ./output

# Customizing benchmarking parameters
soup bench ./output --num-prompts 5 --max-tokens 256

# Use custom prompts from a text file (one per line) or JSONL
soup bench ./output --prompts-file my_prompts.txt
soup bench ./output --prompts-file bench_suite.jsonl
```

This acts as a built-in "speedometer," outputting Tokens-Per-Second (TPS), Total Latency, and Peak VRAM allocations into a clean status table.


## Inference Server

Start a local OpenAI-compatible inference server:

```bash
# Install server dependencies
pip install 'soup-cli[serve]'

# Start server
soup serve --model ./output --port 8000

# With custom settings
soup serve --model ./output --port 8080 --host 127.0.0.1 --max-tokens 1024
```

Endpoints:
- `POST /v1/chat/completions` — chat completions (streaming supported)
- `GET /v1/models` — list available models
- `GET /health` — health check

Compatible with OpenAI SDK:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="output",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### KV Cache Quantization (`soup serve --kv-cache-type`)

Shrink the inference-time KV cache on the transformers backend:

```bash
soup serve --model ./output --kv-cache-type bf16   # cache in the model compute dtype
soup serve --model ./output --kv-cache-type q8_0   # 8-bit quantized cache (needs `hqq`)
```

`bf16`/`f16` need no extra dependency; `q8_0` requires a quant backend (`hqq` / `optimum-quanto`) or the CLI exits with an install hint; `fp8` is Hopper-only. Full detail and the vLLM/SGLang status live in [Performance & Quantization → KV Cache Types](performance-and-quantization.md#kv-cache-types-v0530).

### vLLM Backend (2-4x Faster Inference)

Use [vLLM](https://github.com/vllm-project/vllm) for significantly better throughput in production:

```bash
# Install vLLM support
pip install 'soup-cli[serve-fast]'

# Start with vLLM backend
soup serve --model ./output --backend vllm

# Multi-GPU with tensor parallelism
soup serve --model ./output --backend vllm --tensor-parallel 2

# Control GPU memory usage
soup serve --model ./output --backend vllm --gpu-memory 0.8
```

> **Tip:** Soup auto-detects vLLM. When installed, you'll see a hint during `soup serve` if you haven't enabled it yet.

### SGLang Backend

Use [SGLang](https://github.com/sgl-project/sglang) as an alternative high-throughput backend:

```bash
# Install SGLang support
pip install 'soup-cli[sglang]'

# Start with SGLang backend
soup serve --model ./output --backend sglang

# Multi-GPU with tensor parallelism
soup serve --model ./output --backend sglang --tensor-parallel 2
```

### Speculative Decoding

Use a smaller draft model to speed up generation. **Measure before you trust it** — see
[Train + measure your own draft](#train--measure-your-own-draft-soup-draft) below. Speculative
decoding is only a win when the draft agrees with the target often enough to pay for its own
forward pass; on a small target it is frequently a *slowdown*.

```bash
# Transformers backend — uses HF assisted generation
soup serve --model ./output --speculative-decoding small-draft-model --spec-tokens 5

# vLLM backend — uses vLLM native speculative decoding
soup serve --model ./output --backend vllm --speculative-decoding small-draft-model

# Auto-pair: Soup picks the draft for you based on the target family
soup serve --model meta-llama/Llama-3.1-70B-Instruct --backend vllm --auto-spec
# → auto-paired: meta-llama/Llama-3.2-1B-Instruct (target: Llama-3.1-70B-Instruct)
```

`--auto-spec` handles Llama 3.1/3.3/4, Qwen 2.5/3, Mistral Large, Mixtral, DeepSeek V3/R1, and Gemma 2/3. Models without a known draft pairing (e.g. 8B-or-smaller targets where draft+target overhead outweighs the gain) print a yellow "no draft" note and fall back to standard decoding. A draft you trained yourself with `soup draft distill` is picked up **before** this built-in table.

### Train + measure your own draft (`soup draft`)

The question nobody answers before enabling speculative decoding: *would the draft actually
propose the tokens my model is going to emit?* `soup draft measure` answers it.

```bash
# Would this draft pay off? Acceptance rate + REAL plain-vs-assisted throughput.
soup draft measure --target ./my-tuned-model \
  --draft HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts prod-prompts.jsonl

# Distil a draft from your own target, then serve it automatically.
soup draft distill --target ./my-tuned-model \
  --draft-base HuggingFaceTB/SmolLM2-135M-Instruct \
  --data traffic.jsonl -o draft/
soup serve --model ./my-tuned-model --auto-spec     # picks up ./draft
soup draft list
```

**Acceptance rate** is the fraction of the target's own greedy tokens the draft would have
proposed correctly (teacher-forced argmax agreement — the metric the Medusa/EAGLE papers report).
Higher is better; roughly, ≥70% is where speculative decoding starts paying for the draft's
forward pass on realistic hardware. `--min-acceptance 0.6` exits **2** below the floor, so CI can
gate on it (exit 0 = ok, 2 = below floor, 1 = error).

`distill` runs logit KD through the existing `task: distill` trainer and emits a **dense** model
(a PEFT adapter directory cannot be loaded as an `assistant_model`). Draft and target must share a
tokenizer — a mismatch is refused up front, because speculative decoding proposes *draft* token ids
into the *target's* vocabulary and a mismatched pair silently produces garbage instead of failing.
A `soup shrink` output makes a good draft base: same tokenizer by construction.

**Measured reality check (be sceptical of speedup claims, including ours).** On
`SmolLM2-360M-Instruct` with a `SmolLM2-135M-Instruct` draft, the *stock* draft already scored
**69.3%** acceptance, distilling it changed nothing (69.7% after 2 epochs, 69.3% after 10), and
assisted decoding came out **0.55–0.64×** — a net slowdown. A small same-family draft is already
near its ceiling for agreeing with the target, and the draft's forward pass costs more than the
tokens it saves. Whether distillation pays off on a larger or genuinely diverged target/draft pair
is unproven on a 4 GB box. Run `soup draft measure` on *your* pair rather than assuming.

### Prefix Caching

For RAG and agent workloads with a shared system prompt, enable vLLM's automatic prefix cache:

```bash
soup serve --model ./output --backend vllm --prefix-cache
```

The first request with a given prefix warms the cache; subsequent requests skip the shared prefix compute entirely. Big latency win when 100+ requests share the same system prompt.

### Dynamic LoRA Hot-Swap

Switch the active adapter at runtime without restarting the server:

```bash
soup serve --model base-model --adapters chat=./chat-adapter code=./code-adapter
```

```bash
# Activate an adapter
curl -X POST http://localhost:8000/v1/adapters/activate/chat
# → {"active": "chat", "status": "ok"}

# Return to base model
curl -X POST http://localhost:8000/v1/adapters/deactivate
# → {"active": null, "status": "ok"}

# List loaded adapters with active flag
curl http://localhost:8000/v1/adapters
# → {"adapters": [{"name": "chat", "active": true}, ...], "active": "chat"}
```

Names are validated against `^[a-zA-Z0-9][a-zA-Z0-9-]*$`; activate/deactivate calls are thread-safe behind a lock.

### Multi-Tenant Vector Bank (`soup serve --bank`)

Serve many per-user personas from one model at KB-per-user instead of a full LoRA each.
A VeRA / VB-LoRA bank stores a shared random projection (reconstructed deterministically
from a seed — never stored on disk) plus a small per-user scaling vector. The active user
is chosen **per request** via the `X-User-Id` header:

```bash
# bank.json carries {name, base_model, projection_seed, vector_dim, entries: [{user_id, scaling}, ...]}
soup serve --model base-model --bank ./bank.json --bank-strength 1.0
```

```bash
# Apply alice's persona to this request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-User-Id: alice" \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'

# No / unknown X-User-Id → zero-delta no-op (plain base model, no cross-request leak)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'
```

The per-token delta is `v_user ⊙ (x @ Pᵀ)`, added to the last decoder layer's residual by a
decode-time forward hook. `--bank-strength` scales the delta (magnitude capped at 100). The
`--bank` path must live under your cwd. The active user is resolved **per request** via a
`contextvars.ContextVar`, so concurrent requests on a threaded server never race on shared state
— each request (streaming and non-streaming) gets its own isolated active-user selection, and an
absent / unknown id self-clears to the clean baseline. (v0.71.12 / per-request v0.71.17)

### Serve a Trained MoLE (`soup serve --mole`)

Serve a Mixture-of-LoRA-Experts adapter trained with `task=moe_lora_routing`. The training run
writes a self-describing `mole_manifest.json` next to `mole_gate.pt`; `soup serve --mole <dir>`
loads the base model + the N frozen task LoRAs + the trained gate and blends them **per token**
at decode time (a custom blend loop — both non-streaming and SSE streaming):

```bash
# <dir> is the MoLE training output (contains mole_gate.pt + mole_manifest.json)
soup serve --model ./mole_out --mole ./mole_out --device cuda
```

The base model comes from `--base` if set, otherwise the base recorded in the manifest. `--mole`
requires `--backend transformers` and is mutually exclusive with `--bank` / `--steer` /
`--adapters` / `--speculative-decoding`. The manifest + gate are cwd-contained, symlink-rejected,
and size-capped, and the gate loads with `weights_only=True`. Because the blend recomputes the
sequence each step (no KV cache), this path is best for small / demo models. (v0.71.17)

### Structured Output (JSON Schema / Regex)

Constrain model output to a valid JSON schema or regex pattern:

```bash
# JSON schema (schema file must live under your cwd)
soup serve --model ./output --structured-output json --json-schema product.json

# Regex (length-capped at 2048 chars, null bytes rejected)
soup serve --model ./output --structured-output regex --regex-pattern '\d{3}-\d{4}'
```

The `validate_json_schema` helper caps serialised size at 64KB and requires a top-level `type` field so malformed schemas fail fast at server startup, not per-request.

### Continuous-Batching Dashboard + `/metrics`

Track live server health:

```bash
soup serve --model ./output --dashboard
```

```bash
curl http://localhost:8000/metrics
# → {
#   "requests_total": 1234,
#   "tokens_generated_total": 456789,
#   "active_requests": 3,
#   "latency_p50_ms": 185.2,
#   "latency_p95_ms": 720.0,
#   "latency_samples": 1000
# }
```

Latency percentiles are computed from the last 1000 requests; counters include failure paths so the dashboard shows true reliability, not just success rate.

### OpenTelemetry Request Tracing

Emit per-request spans to your OTLP collector:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp

soup serve --model ./output \
  --trace \
  --trace-endpoint http://localhost:4317
```

The OTLP endpoint is SSRF-hardened: only http/https schemes, plain HTTP only for loopback (`localhost`/`127.0.0.1`/`::1`), and RFC1918 / link-local / `0.0.0.0` all rejected via `ipaddress.ip_address`. When the SDK is missing the flag is a no-op with a warning — the server starts fine without spans.

> **Note:** `max_tokens` is capped at 16,384 per request. Error details are never exposed in HTTP responses.


## Web UI

Launch a local web interface to manage experiments, start training, explore data, and chat with models — all from your browser.

```bash
pip install 'soup-cli[ui]'
soup ui
# -> opens http://127.0.0.1:7860 in your browser
# -> prints auth token to console
```

**Pages:**
- **Dashboard** — view all experiment runs, loss charts, system info, multi-run comparison
- **New Training** — create configs from templates or 142 ready-made recipes, validate, start training with live SSE log streaming and progress bar
- **Data Explorer** — browse and inspect datasets (JSONL, JSON, CSV, Parquet)
- **Model Chat** — chat with streaming responses, configurable temperature/top_p/max_tokens, system prompt, adapter selection, markdown rendering, chat export

**Live monitoring + enhanced UX:**
- **Training Live Monitor** — real-time SSE log streaming, live metrics, progress bar with ETA
- **Enhanced Metrics** — 2x2 chart grid (loss, LR, grad_norm, throughput) + GPU memory chart, eval results table
- **Multi-Run Compare** — overlay loss curves from up to 5 runs side-by-side
- **Chat Upgrade** — SSE streaming via proxy, typing indicator, cancel button, markdown renderer (bold, italic, code blocks), chat export as JSON
- **Config Builder** — recipe dropdown (142 recipes), config schema API for dynamic form generation

**Security:** The Web UI generates a random auth token at startup (printed to console). All mutating endpoints (start/stop training, delete runs, inspect data, validate config) require `Authorization: Bearer <token>` header. CORS is restricted to the served origin. Data inspection is sandboxed to the working directory.

```bash
# Custom port, don't auto-open browser
soup ui --port 8080 --no-browser
```


## Inference Server Trace Log

`soup serve --trace-log <path>` writes a passive append-only JSONL log per chat completion:

```bash
soup serve --model ./out --trace-log ./serve-trace.jsonl --trace-log-cap-mb 100
```

Each line: `{"ts": ..., "prompt": ..., "response": ..., "latency_ms": ..., "tokens": ...}`. Path-containment validated, hard rotation cap (default 100 MB, one backup retained), symlink-reject on the backup path (TOCTOU defence), and `hf_*` / `sk-*` / `Bearer …` token shapes redacted to `<redacted>` before write. Failures (disk full, serialisation errors) never crash the request handler.


## Soup Quantize — Ergonomic Export Alias

```bash
soup quantize ./out --to gguf --bits 4
soup quantize ./out --to gptq --bits 4 -o ./out-gptq
```

Prints the equivalent `soup export …` invocation (escaped via `shlex.quote`) for copy-paste. Intentionally does NOT in-process call `soup export` — Typer commands aren't safe to re-enter.


## Llama.cpp Proxy

```bash
soup llama --help                  # list supported subcommands
soup llama cli -m model.gguf -p "Hello"
soup llama gguf-split --merge a.gguf b.gguf out.gguf
soup llama server -m model.gguf
```

Closed allowlist: `cli` / `mtmd-cli` / `gguf-split` / `server` / `quantize`. Forwards to `llama-*` binary on PATH (`shutil.which`) with **filtered child env** — `HF_TOKEN` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and other secrets are dropped before exec; only `PATH` / `HOME` / `USER` / locale + llama.cpp-recognised `LLAMA_CPP_HOME` / `GGML_*` / `OMP_NUM_THREADS` are forwarded.


## Tail-Latency Stats + Tool-Call Timer

```python
from soup_cli.utils.tail_latency import summarise_latency
from soup_cli.utils.tool_outputs import ToolOutputsBuffer, ToolCallTimer

stats = summarise_latency([12.3, 14.1, 9.7, 18.8, 11.2])
# TailLatencySummary(count=5, mean=..., p50=..., p95=..., p99=..., ema=...)

buffer = ToolOutputsBuffer()
with ToolCallTimer(buffer, name="fetch_url") as timer:
    timer.set_output("...")
```

Pure-Python EMA + linear-interp percentiles (DoS cap: `MAX_SAMPLES=1_000_000`). `ToolOutputsBuffer` is a thread-safe `collections.deque(maxlen=1000)` ring with truncated previews; `ToolCallTimer` records duration / output / error per invocation for tool-calling SFT runs.


## Web UI Plugin Registry + Env Knobs

```python
# src/soup_cli/ui/plugins/my_tab.py
from soup_cli.ui.plugins import register_tab

def render_my_tab(request) -> str:
    return "<div>my tab body</div>"

register_tab(name="my-tab", title="My Tab", render=render_my_tab)
```

Drop-in plugin registry with kebab-case name allowlist, 32-tab cap, idempotent re-register. Plus `API_HOST` / `API_PORT` / `API_KEY` / `GRADIO_HOST` / `GRADIO_PORT` env knobs for FastAPI + Gradio surfaces.


## Deploy Autopilot

Pick the optimal PEFT + quantisation + speculative-decoding combo for your hardware target in one command:

```bash
soup deploy autopilot --target rtx-4090-24gb --base meta-llama/Llama-3.2-1B
# Writes:
#   deploy_autopilot.yaml  — ready-to-train soup.yaml recipe
#   deploy_autopilot.sh    — planned deploy shell script
```

Profiles ship out of the box for Apple Silicon (`mac-m3`, `mac-m4-pro`), consumer NVIDIA (`rtx-3060-12gb`, `rtx-4090-24gb`), mobile (`iphone-16`, `pixel-9`), local runtimes (`ollama-local`, `lm-studio`), and cloud (`runpod-a100`, `hf-jobs-h100`). `--list` shows the full table. Every profile is a frozen dataclass with closed allowlists on runtime / quant / PEFT — bad config values fail at import time. The generated bash uses `shlex.quote` on the model path and writes are protected by cwd containment + `os.lstat + S_ISLNK` TOCTOU rejection.


## Agent Forge

Turn an OpenAPI 3.x, MCP server manifest, or GraphQL introspection JSON straight into a tool-calling SFT dataset — no manual labelling, no scaffolding:

```bash
# 1. Parse spec + synthesise a tool-calling dataset
soup agent synth --spec api.yaml --output ds.jsonl --examples-per-endpoint 4

# 2. Plan the training run (prints the soup train invocation)
soup agent train --spec api.yaml --base meta-llama/Llama-3.2-1B

# 3. Score model predictions against the spec catalog
soup agent eval --spec api.yaml --predictions preds.jsonl
```

Each row of the synthesised dataset is `{messages: [user, assistant_with_tool_call], tool: <name>, source_endpoint: <path>}`. `$ref` strings in OpenAPI are left opaque (no external resolution — defends against file-read SSRF), `yaml.safe_load` only, 5 MiB spec cap, 10 000-endpoint cap, atomic JSONL write via staged-tempfile + `os.replace`. `eval` enforces a 1 000 000-line cap on predictions and rejects symlinks at every read/write boundary.


## HF Space SDK Auto-Pick

When you deploy a custom Space template directory, Soup now picks `space_sdk="streamlit"` / `"gradio"` from the rendered `requirements.txt`:

```bash
soup deploy hf-space --space my-org/my-app --model my-org/my-model --template-dir ./my-template
```

If `requirements.txt` lists `streamlit`, the Space is created with the Streamlit SDK. Otherwise (no requirements, gradio listed, etc.), Soup falls back to the Gradio default. The HF Hub allows `docker` and `static` SDKs too, but those cannot be inferred from `requirements.txt` alone — use the built-in templates or supply a custom one with an explicit `--sdk` override.


## Anthropic Messages API Converter

Pure-Python converters between OpenAI chat-completions and Anthropic Messages payload shapes:

```python
from soup_cli.utils.anthropic_messages import to_anthropic, from_anthropic

anthropic_payload = to_anthropic({
    "model": "claude-3-5-sonnet",
    "messages": [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ],
    "max_tokens": 256,
})
```

Multiple `system` messages join with `\n\n`. `tool` role with structured (list) content is concatenated into a single `tool_result` text block, never silently dropped. `max_tokens` capped at 16384, `temperature` bounded `[0.0, 2.0]`. Live `/v1/messages` endpoint inside `soup serve` lands in v0.45.1.


## Server-Side Tools

```python
from soup_cli.utils.server_tools import (
    SUPPORTED_TOOLS, WebSearchConfig, is_domain_allowed, validate_web_search_config,
)

# SUPPORTED_TOOLS == frozenset({"python", "bash", "web_search"})
config = WebSearchConfig(
    domain_allowlist=("example.com", ".docs.example.com"),
    rate_limit_per_minute=30,
)
validate_web_search_config(config)
is_domain_allowed("a.docs.example.com:443", config.domain_allowlist)  # True
is_domain_allowed("[::1]", config.domain_allowlist)                   # False
```

`python` and `bash` reuse the v0.25.0 RLVR sandbox; `web_search` is gated by an explicit domain allowlist (default empty = deny all). `is_domain_allowed` strips `:port` suffixes before matching and rejects IPv6 literals so `Host: api.example.com:443` matches `api.example.com`. Live HTTP tool endpoints in v0.45.1.


## Anthropic Messages Endpoint

Both `soup serve --backend transformers` and `soup serve --backend vllm` now expose
a POST `/v1/messages` route that accepts Anthropic Messages-shaped payloads:

```bash
curl http://localhost:8000/v1/messages -H "Content-Type: application/json" -d '{
  "model": "my-model",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 64
}'
```

Non-streaming requests return Anthropic-shaped envelopes. Streaming (`stream: true`)
returns Anthropic event-shape SSE with `message_start` → `content_block_delta` →
`message_delta` + `message_stop` events. Validation errors return a generic
`"Invalid request"` 400 body with details logged server-side at DEBUG. CORS restricted
to loopback-only (`localhost` / `127.0.0.1`) on both backends.


## N-gram Speculative Decoding

When a server is configured with an `NgramSpecConfig`, every chat completion forwards
`prompt_lookup_num_tokens=N` into `model.generate(...)` (HF Transformers ≥ 4.38
prompt-lookup decoding — no draft model required). Mutually exclusive with a real
`assistant_model`; if both are set, the real draft model wins.


## Server-Side Tool Endpoints

Three POST routes are now available on `soup serve`:

- **`POST /v1/tools/python`** — Sandboxed Python execution. Requires Bearer token auth.
  Wraps the RLVR sandbox with 5-second timeout and 64KB code cap. Returns 200 with
  `stdout` / `stderr` / `return_value`; 400 on validation error; 401 on bad auth.

- **`POST /v1/tools/web_search`** — Domain-allowlisted web search. Requires Bearer token
  auth. Uses httpx backend with hard 5-second timeout and 5-result cap. Returns results
  as `[{url, title, snippet}]` with snippets sanitized (null bytes stripped).
  Deny-by-default via `WebSearchConfig.domain_allowlist`.

- **`POST /v1/tools/bash`** — Deferred to v0.53.8. Current child-process isolation
  insufficient for `/bin/sh -c` (subprocess escapes the RLVR sandbox). Returns 501 with
  v0.53.8 marker pending container/namespace work.


