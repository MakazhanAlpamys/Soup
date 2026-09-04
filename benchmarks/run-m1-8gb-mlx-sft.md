<!--
Working record, published as written.

Five completed runs and one failure, captured from the printed output of the
harness in `harness/mlx_sft_smoke.py` on an 8 GB M1. The failure and the two
corrections I made while measuring are left in, in the order they happened.

Hardware: Apple M1, 8 GB unified memory, 8 cores, macOS 26.6.2, arm64.
Software: Python 3.12.14, mlx 0.32.2, mlx-lm 0.31.3.
-->

# MLX SFT on an 8 GB M1 — where the ceiling actually is

**Status: FIVE RUNS COMPLETED, ONE FIXTURE FAILED. This is not a gate.** It
establishes that `MLXSFTTrainerWrapper`'s training loop runs end to end against a
real MLX runtime, and it measures peak memory and throughput on the low-memory
end of Apple Silicon. It does **not** establish correctness of the resulting
adapters beyond "they load and the loss falls" — see *What this does not
establish*, which is longer than the results on purpose.

Filed against [#23](https://github.com/MakazhanAlpamys/Soup/issues/23), which
asked for exactly this and had been open since v0.25.0 because CI has no Apple
Silicon and the maintainer has no Mac.

## Why this box

`benchmarks/gate-qwen4-ple-m4-max.md` is already an Apple Silicon record, so this
is **not** the first non-CUDA row — a framing worth correcting before it
propagates. That record is PyTorch/MPS, covers Qwen4 decoder mapping, and states
plainly that it "must not be read as a throughput, peak-memory, or production-
trainability claim."

What is new here is narrower and worth stating exactly:

- the first measurement of the **MLX backend's training loop** rather than the
  transformers/MPS path;
- the first **throughput and peak-memory** figures from Apple Silicon;
- **8 GB**, against the 128 GiB M4 Max of that record and the M4 of
  [#362](https://github.com/MakazhanAlpamys/Soup/pull/362). A footprint problem
  shows up here or nowhere.

Unit convention: **GB as MLX reports them** (`mx.get_peak_memory()`), which is
GiB-based. Host figures come from `memory_pressure` and `sysctl vm.swapusage`.

---

## Configuration

| | |
|---|---|
| Machine | Apple M1, 8 GB unified memory, 8 cores |
| OS / Python | macOS 26.6.2, arm64, Python 3.12.14 |
| MLX | `mlx` 0.32.2, `mlx-lm` 0.31.3, Metal available |
| Config | `backend: mlx`, `task: sft`, LoRA r=8 α=16, `batch_size: 1`, `max_length: 512`, `lr: 1e-4` |
| Data | 48 synthetic chatml rows, 1 epoch -> 48 iterations |
| Harness | [`harness/mlx_sft_smoke.py`](harness/mlx_sft_smoke.py) |

**Dispatch was asserted, not assumed.** Every run checks
`resolve_trainer(cfg)` returns `MLXSFTTrainerWrapper` *before* the timer starts
and aborts otherwise. This is not ceremony:
[#363](https://github.com/MakazhanAlpamys/Soup/issues/363) was `backend: mlx`
never reaching the MLX trainer at all, so a plausible-looking number here could
have been the transformers path. @nicolasramos flagged exactly this when handing
the issue over, and the check is in the harness because of it.

---

## Results

All five completed runs, in the order they ran. `peak` is MLX's own
`get_peak_memory()`; the per-iteration `Peak mem` mlx-lm prints is given where it
differs.

| model (4-bit) | load | peak after load | train, 48 it | **peak in train** | mlx-lm peak | **host free before -> after** | swap total -> | tok/s | loss | adapter | reload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct` | 31.7 s | 0.262 GB | 19.7 s | **0.497 GB** | 0.497 | **54% -> 39%** | 2048 M | ~240 | 3.639 -> 0.107 | 2,122 KB | 0.9 s |
| `Llama-3.2-3B-Instruct` | 123.3 s | 1.683 GB | 29.5 s | **2.083 GB** | 2.236 | **54% -> 32%** | 2048 -> 3072 M | ~82 | 4.527 -> 0.330 | 8,972 KB | 2.1 s |
| `Qwen2.5-7B-Instruct` | 268.3 s | 3.993 GB | 67.6 s | **4.584 GB** | 4.922 | **69% -> 14%** | 3072 -> 9216 M | ~33 | 4.514 -> 0.127 | 9,868 KB | 15.8 s |
| **`Llama-3.1-8B-Instruct`** | 317.8 s | 4.207 GB | 71.0 s | **4.800 GB** | 5.154 | **69% -> 14%** | 3072 -> 9216 M | ~36 | 3.454 -> 0.142 | 13,326 KB | 23.8 s |

**Peak and free unified memory are the load-bearing columns here**, not tok/s: on
8 GB shared with the OS the ceiling is the interesting variable. Both 7B and 8B
end at **14% host free**, and the swap file more than triples across those two
rungs — the machine absorbs them rather than refusing, which is the mechanism
section below.

`load` includes the first-time Hub download, so it is a network figure, not a
model-load figure.

### The headline

**`llama3.1-8b-sft-mlx` — the shipped recipe — trains on an 8 GB M1.** Peak
5.154 GB, 48 iterations in 71 s, adapter written and reloadable.

I predicted before running that it would not fit, and said so on the issue. It
fits. The prediction was wrong and the measurement is the reason this row exists.

### Two things I cannot explain and am not going to smooth over

**7B is slower than 8B.** `Qwen2.5-7B` runs at ~33 tok/s against `Llama-3.1-8B`'s
~36, despite being the smaller model, and its `It/sec` is lower across every
reported iteration. Candidate explanations I did **not** test: Qwen2.5's larger
vocabulary (151,936 vs 128,256) making the output projection dominate at this
tiny sequence length; different host memory pressure between the two rungs (the
7B rung ran with 14% host memory free at its end, the 8B rung also 14%, but swap
had grown between them). Stated as an observation, not a mechanism.

**Adapter reload time scales far worse than size.** 0.9 s -> 2.1 s -> 15.8 s ->
23.8 s across a 6x span of adapter size. The reload re-reads the full base model,
so this is dominated by base weights, not the adapter — but 15.8 s for 7B against
23.8 s for 8B is steeper than the parameter ratio, and I did not isolate it.

### Host behaviour

The interesting mechanism is that **a model too big for RAM does not fail here —
it pages.** macOS grew the swap file through the run (2048 -> 3072 -> 4096 ->
8192 -> 9216 MB) and free memory sat at 5-6% for long stretches while `ls /`
still returned in 16 ms. MLX memory-maps weights, so most of the model is
file-backed and evictable; low "free memory" is the normal shape of this
workload, not distress.

That has a direct consequence for anyone writing a pre-flight for this backend:
**an allocation-failure check would not fire.** It is the same trap as the WDDM
spill in [#649](https://github.com/MakazhanAlpamys/Soup/issues/649) — "the step
did not raise" is not "the step fits" — arrived at from the opposite platform.

---

## The failure: the fixture #23 recommends does not load

Issue #23 names `mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit` as the tiny test
model. It cannot be used:

```
FileNotFoundError: No safetensors found in
  ~/.cache/huggingface/hub/models--mlx-community--TinyLlama-1.1B-Chat-v1.0-4bit/snapshots/01a7088...
```

The repo exists and resolves. Its file list, last modified **2024-01-05**:

```
config.json  special_tokens_map.json  tokenizer.json  tokenizer_config.json
weights.00.safetensors
```

`mlx_lm` 0.31.3 fetches `allow_patterns=["model*.safetensors", ...]`
(`utils.py:237-239`) and `load_model` globs `model*.safetensors`
(`utils.py:316`). The legacy `weights.NN.safetensors` naming matches neither, so
the weights are never downloaded and the load fails on the tokenizer-only
snapshot. Nothing is wrong with the repo; the naming convention predates the one
mlx-lm now requires.

**Anyone writing the CI job in #23 should not use that model id.**
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` is the smallest fixture verified
working here (282 MB, 19.7 s for 48 iterations).

---

## What this does not establish

- **No correctness claim.** Loss falls and the adapter reloads. Nothing here
  compares MLX output against a transformers reference, checks gradient
  exactness, or evaluates the tuned model. "It trains" is the claim; "it trains
  *correctly*" is not.
- **The data is synthetic and trivial** — 48 rows of short Q/A, deliberately
  repetitive. A loss falling from 3.6 to 0.1 on 48 repeated rows is memorisation,
  not evidence of learning. It is a smoke signal, nothing more.
- **`target_modules: auto` means Q/V only here.** Every run printed
  `auto cannot be resolved for all architectures; defaulting to
  ['self_attn.q_proj', 'self_attn.v_proj']`. This is deliberate — see
  [#392](https://github.com/MakazhanAlpamys/Soup/issues/392), where shipping
  `{"keys": ["auto"]}` silently dropped every LoRA tensor — but it means these
  loss curves are Q/V-only LoRA, not all-linear, and are not comparable with
  CUDA-path numbers that resolved `auto` differently.
- **One box, one run per rung. No repeats, so there is no spread** — and a
  later re-run shows the spread is large. Re-running `Qwen2.5-0.5B` from a warm
  cache on a quieter machine gave **355-480 tok/s against the ~240 in the table**,
  the same model under the same config. Part of that is the 16-row re-run being
  shorter, part is host pressure, and I did not separate them. Treat every tok/s
  figure here as an order of magnitude that moves by ~2x with conditions, not as
  a benchmark. The T4 record set the precedent for not quoting throughput off a
  single constrained run; this row is bound by the same limit.
- **`load` is polluted by download time.** First-time Hub fetches dominate it.
  A warm-cache load was not measured separately.
- **Host pressure varied between rungs** and was not controlled. Swap grew
  monotonically across the session.
- **DPO and GRPO are untouched.** `_validate_mlx_task` rejects them at config
  load; only SFT was exercised.

## Corrections made while measuring

Both are mine, both were caught before they reached a claim, and both are the
reason the numbers above are worth anything.

1. **A bare HTTP 401 from the Hub proves nothing.** I initially read 401 as
   "model does not exist" and asserted on that basis that
   `SmolLM2-135M-Instruct-4bit` was missing. A repo id I invented returns the
   same 401. The correct instrument is `huggingface_hub.model_info`, which
   distinguishes `RepositoryNotFoundError` from a gated-but-present repo. This
   mattered beyond the fixture choice — it is what put
   [#661](https://github.com/MakazhanAlpamys/Soup/issues/661) on evidence rather
   than on a status code.
2. **`target_modules: auto` degrading to Q/V is not a defect.** I recorded it as
   one, then found #392 and the docstring explaining why it is deliberate. It
   belongs in this record as context for reading the loss curves, not as a bug.

Also lost and re-run: the TinyLlama rung's output was truncated by a `tail -120`
in my own ladder invocation, so the failure above was re-measured from scratch
rather than reconstructed.

## Reproducing

```bash
pip install -e ".[mlx]"
python benchmarks/harness/mlx_sft_smoke.py mlx-community/Qwen2.5-0.5B-Instruct-4bit 48 1
```

Requires Apple Silicon. The harness asserts MLX dispatch before measuring and
prints host memory before and after, so a run on another box is directly
comparable to the table above.
