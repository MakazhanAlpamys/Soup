# Native MLX HTTP serving: qualification and route review

Scope: #2001, confirmed in the issue on 2026-11-02. One dense causal text model,
one optional ordinary native LoRA, three existing HTTP routes, serialized decoding,
a bounded queue and incremental SSE. This does not qualify quantized serving,
continuous batching, distributed serving, tools or model-quality improvements.

## Reproduction

Use an isolated Apple Silicon Python 3.12 environment. The current run uses MLX
0.32.4 / MLX-LM 0.32.2. The floor run uses MLX 0.31.2, MLX-LM 0.31.3,
Transformers 5.16.1, Pydantic 2.6.0, FastAPI 0.104.0, uvicorn 0.24.0 and
httpx 0.27.2; the HTTP client pin accommodates the old Starlette
TestClient API. Neither environment installs Torch, TRL, PEFT, datasets, accelerate
or bitsandbytes. MLX 0.20.0 is not asserted as a resolvable combined dependency
floor: MLX-LM 0.31.3 requires MLX >=0.31.2 on Darwin, and that effective floor
is the one exercised here.

```bash
python -m venv .venv-native
. .venv-native/bin/activate
pip install -e '.[mlx,serve]' 'mlx==0.32.4' 'mlx-lm==0.32.2' pytest httpx \
  -c .github/constraints/pinned-upstream.txt
pytest -o addopts= tests/test_mlx_serve.py tests/test_mlx_streaming.py \
  tests/test_mlx_generation.py tests/test_mlx_serve_native.py -q

# WORK must be a real, nonsymlink directory containing the generated fixtures.
WORK=/absolute/path/to/native-serve-evidence
mkdir -p "$WORK"
python -m benchmarks.qualify_mlx_serve --fixture-dir "$WORK/fixture" \
  --workdir "$WORK" --report "$WORK/tiny.json"
```

`benchmarks/harness/mlx_serve_fixture.py` creates a seeded one-layer, width-32 local
Llama with a WordLevel tokenizer, then performs 16 real Adam LoRA updates. It checks
frozen tensors exactly and saves a nonzero adapter. The qualification process loads
the saved artifact; a **fresh child process** launches the public `soup serve` CLI.
The harness waits for an actual bound `/health` response before requests. For two
prompts and a repeat of the first, nonstreamed and streamed content, finish reason
and usage must equal direct `mlx_lm.stream_generate`, exactly. A base-only control
must differ from the adapter on at least one prompt or qualification fails.

The separate pretrained run uses the revision and file hashes in its JSON record.
Download that revision into a real local directory, then run the same harness with
`--model`, `--model-identity` and `--model-revision` instead of `--fixture-dir`.
All serving runs are offline after preparation. These are lifecycle/greedy equality
checks, not a retrieval, general capability or throughput benchmark. Observed HTTP
arrival times alone do not prove incremental decoding: the portable ASGI test
requires the first content frame to arrive while the generator is still blocked.

## Route and lifecycle security review

| Boundary | Implementation and verification |
|---|---|
| Shared application | `_create_app` retains its existing CORS and browser-Origin dependency. Native routes return before tools, adapter mutation, messages, legacy completions or metrics are registered. No second application factory or all-backend refactor. |
| Bind/auth | Existing loopback default and non-loopback token requirement remain. As on existing generation routes, bearer tokens do **not** authenticate inference. Reverse-proxy Host names are allowed; foreign browser Origins are refused. Operators must protect exposed inference outside this route. |
| Model/adapter paths | CLI validates local directories and their immediate files under cwd using realpath/commonpath, rejecting symlink components/files. Native metadata binds an optional `--base` to the saved base identity. HTTP accepts only the loaded model ID, never a filesystem path or per-request adapter. Hub base identities in operator-provided metadata use the existing shared loader. |
| Remote code | The same shared default-deny loader policy checks the actual base. Explicit operator `--trust-remote-code` retains the existing opt-in; no HTTP parameter enables it. Native loader validates every adapter key, shape and finite floating tensor. Corrupt adapters fail startup rather than becoming base-only generation. |
| Input bounds | Body size is checked before extending the buffer (1 MiB). Strict schemas bound 128 messages, 65536 chars/message, 16384 input tokens and 1–16384 generated tokens. Numeric NaN/Inf, unknown fields, unsupported roles/content/controls and another model ID fail. Stop strings are bounded separately. |
| Admission/backpressure | One worker owns model initialization, decode and cleanup. At most eight jobs wait; overload is 503 with Retry-After. Each consumer buffers at most eight increments. Tests hold an active decode, fill the queue and prove cancelled queued work releases admission capacity. |
| Request isolation | New messages, native generator, default fresh KV cache and random seed per job; no per-request mutation of the active model or adapter. Repeated native HTTP results match independent direct generation after an unrelated request. |
| Cancellation | Queued jobs are removed immediately. Active cancellation is latched and checked before the next iterator advance; closing the native iterator releases its local cache. The response wrapper also cancels if a client disconnects before its body iterator starts. A running prefill/kernel cannot be interrupted mid-call. |
| Startup/shutdown | Readiness follows successful loading, with no per-request reload. Startup exceptions and normal/exceptional response exits close the worker. Shutdown cancels pending/active jobs and joins at a decode boundary; a worker that cannot stop within ten seconds is reported as a shutdown failure. |
| Errors and SSE | Before headers, generation errors return sanitized HTTP errors. After headers, a sanitized SSE error and DONE terminate the stream. No internal exception or filesystem path is returned. Stop prefixes are retained so cross-increment delimiters cannot leak into content. |

## Validation records

- Current and minimum stacks: **166 passed each**, including real native model
  tests. Minimum-stack HTTPX emits 52 deprecation warnings about the older
  TestClient shortcut; these are not failures.
- Seven isolated mutants fail as intended: whole-response buffering, lost active
  cancellation, unbounded admission, leaked stop prefixes, ignored adapters,
  enabled unqualified quantized serving and a retained startup traceback. Restored
  controls pass. Separate red/green tests verify typed responses/OpenAPI and reject
  invalid engine usage instead of returning a malformed successful response.
- Real public CLI records on an **Apple M4 Pro / 48 GiB**, with Metal available
  and the PyTorch training stack absent:
  [trained tiny LoRA/current](evidence/mlx-serve-2001-tiny.json),
  [trained tiny LoRA/minimum](evidence/mlx-serve-2001-floor.json),
  [SmolLM2-135M-Instruct/current](evidence/mlx-serve-2001-pretrained.json).
  The pretrained revision is `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
  Both tiny runs make 16 optimizer updates, preserve all 11 frozen base tensors
  exactly, and produce adapter output distinct from base-only controls. Each of
  the three CLI runs completes six real HTTP generation calls, checks direct
  engine equality and exits successfully on SIGINT.

Each JSON records the upstream base commit, SHA256 of the changed source files,
the sorted whole-source manifest digest, fixture/benchmark hashes, model files,
dependency versions, commands and exact content/count comparisons. The source
digest identifies the qualified implementation even before its publishing commit.

Default tests remain offline and deterministic; real-native tests are explicitly
selected in the existing `mlx-smoke` job, installing the serving extras and HTTP
test client there. Portable streaming/cancellation/queue tests use real ASGI handlers
and controlled generators; they are kept separate from Metal model qualification.

The final full serial run (`python -m pytest tests/ -v --tb=short`) passed on
2026-11-03: **48,986 passed, 590 skipped, 10 deselected**, **87.99% coverage**
(77% required), 1490.56 seconds, exit 0. The functional source digest was unchanged.
Loopback traffic bypassed the macOS system proxy while external traffic retained
its proxy. Ruff, generated-reference structure and all three dependency checks
passed. Incomplete or failing development attempts are not counted as passing runs.
