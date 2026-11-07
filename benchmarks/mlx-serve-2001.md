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
| Admission/backpressure | One worker owns model initialization, decode and cleanup. At most eight jobs wait; overload is 503 with Retry-After. Each consumer buffers at most eight increments. Every increment has a 30-second put deadline (`OUTPUT_PUT_TIMEOUT_SECONDS`): expiry cancels the job and closes the iterator so the worker serves the next job. A timeout monitor also cancels any blocked streamed ASGI send. Tests keep a client connected with a permanently blocked send and verify the next request and cleanup within bounded waits; separate controls pin cancellation and the timeout signal. Admission tests fill the waiting queue and prove cancelled work releases capacity. |
| Request isolation | New messages, native generator, default fresh KV cache and random seed per job; no per-request mutation of the active model or adapter. Repeated native HTTP results match independent direct generation after an unrelated request. |
| Cancellation | Queued jobs are removed immediately. Active cancellation is latched and checked before the next iterator advance; closing the native iterator releases its local cache. The response wrapper also cancels if a client disconnects before its body iterator starts. A running prefill/kernel cannot be interrupted mid-call. |
| Startup/shutdown | Readiness follows successful loading, with no per-request reload. Startup exceptions and normal/exceptional response exits close the worker. Shutdown cancels pending/active jobs and joins at a decode boundary; a worker that cannot stop within ten seconds is reported as a shutdown failure. |
| Errors and SSE | Validation details contain only loc/msg/type using the Pydantic 2.x API. Empty/over-limit prompts are 422; other engine failures are 500. Client departures use a private handled exception rather than task cancellation. Before headers, generation errors return sanitized HTTP errors. After headers, a sanitized SSE error and DONE terminate the stream. No internal exception or filesystem path is returned. Stop prefixes are retained so cross-increment delimiters cannot leak into content. |

## Validation records

- Current and minimum stacks after the item 13 follow-up: **184 passed each**, including real native model
  tests. Minimum-stack HTTPX emits 55 deprecation warnings about the older
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

The original full serial run (`python -m pytest tests/ -v --tb=short`) passed on
2026-11-03: **48,986 passed, 590 skipped, 10 deselected**, **87.99% coverage**
(77% required), 1490.56 seconds, exit 0. The functional source digest was unchanged.
Loopback traffic bypassed the macOS system proxy while external traffic retained
its proxy. Ruff, generated-reference structure and all three dependency checks
passed. Incomplete or failing development attempts are not counted as passing runs.

## First review regressions

The response-start failure case proves that a stream whose body never starts still
cancels its decode and releases the next request. Additional regressions verify the
MLX seed call, missing-dependency preflight, forced-color token diagnostics, quiet
nonstreamed disconnects, validation-detail compatibility, no iterator advance after
cancellation, the portable prompt cap, engine-error classification and simultaneous
distinguishable HTTP streams. Ten individual review mutations fail; restored
controls pass. The ANSI case deliberately retains actual escape sequences.

Independent core/serve-only environments with FastAPI 0.104.0, uvicorn 0.24.0 and
Pydantic **2.1.1** and **2.3.0** each pass **172 portable tests**. These use a replaced
native boundary and are not additional real-MLX dependency-floor claims. The
separate current/minimum native environments and refreshed public CLI JSON records
qualify the actual native path. The final revision serial result is recorded
separately after completion; the original serial result above describes the initial
qualified implementation.

The reviewed revision also passed the full serial suite on 2026-11-03:
**48,995 passed, 590 skipped, 10 deselected**, **88.00% coverage**, 1657.05 seconds,
exit 0, with its functional source digest unchanged throughout. Those were the first-review results; the clarified follow-up is recorded below.

## Clarified follow-up (items 5, 7, 8 and 11)

The output queue has a **30-second per-increment put deadline**, not a total
generation timeout. Expiry cancels the job, closes native decoding, releases the
worker and cancels a blocked streamed ASGI send. The regression leaves the client
connected and blocks its first body send forever; a second request must complete,
the first iterator must close, and the blocked send must end within bounded waits.
The stalled-reader test uses a 0.5-second budget to allow scheduling margin; the
separate direct queue test uses 0.01 seconds.

Additional regressions make only the route poll see a streamed departure and
assert no error frame, fabricated DONE or error log; reject symlinked files inside
both model and adapter directories and an adapter metadata base outside cwd; and
require NativePromptError at both source-level prompt refusal sites. Each path
test first validates an otherwise-valid artifact. Ten isolated follow-up mutations
fail, including both path guards, both error types, deadline/cancellation/timeout
signal/send cleanup, quiet departure and the pre-existing response-start cancel.
Current and minimum real-native stacks each pass **183 tests**. Pydantic 2.1.1
and 2.3.0 core/serve stacks each pass **180 portable tests**, as does the forced-color
portable run. All three real public CLI JSON records match the revised source.
The final follow-up serial suite passed on 2026-11-03: **49,003 passed, 590 skipped,
10 deselected**, **88.00% coverage**, 1587.54 seconds, exit 0; its functional source
digest was unchanged throughout.

## Healthy slow-reader contract (item 13)

A portable HTTP regression pins the documented 30.0-second default, then uses a
0.5-second test budget and reads 40 content frames at 25 ms per body frame. It
asserts that elapsed stream time exceeds one budget while all 40 characters, the
terminal reason/usage and DONE are received. A default of 0.05 or a deadline shared
by the whole job each fails this case. The permanently stalled-reader test uses
0.5 seconds without suppressing missing-attribute errors. Deadline expiry emits
one warning naming the configured limit; cancellation prevents repeated warnings.
Removing that warning fails the direct queue case. The three isolated counterfactuals
fail and restored controls pass.

Current and minimum real-native stacks each pass **184 tests**. Independent
Pydantic 2.1.1 and 2.3.0 core/serve stacks each pass **181 portable tests**, as does
the forced-color portable run. All three public CLI records were refreshed and
match the final source and qualification scripts. The final serial suite passed
on 2026-11-03: **49,004 passed, 590 skipped, 10 deselected**, **88.00% coverage**
(77% required), 1591.77 seconds, exit 0; the functional source digest was unchanged.
Ruff, dependency checks and generated-reference structure pass.
