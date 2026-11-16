# IPO layer streaming on a Tesla T4

This record follows the [qualification conditions for #2766](https://github.com/MakazhanAlpamys/Soup/issues/2766#issuecomment-6915635433).
The rows cover one local fixture on this card at this shape.
These runs measure finite updates, scaler recovery and adapter persistence.
Throughput, large-model capacity and CUDA numerical parity were not measured. The gates require
the four scale-1024 streamed cells and their resident controls to apply an update,
and both default-setting runs to apply an update within 12 attempts.

The run used upstream commit `693a928f838c147e412cacf74c7bdb00cdb8f6fc` plus the
[qualification patch](evidence/ipo-streaming-t4/qualification.patch), SHA-256
`4334662f352b1ed43cece34b10b426fdd1f07b2b03989602a796524c12a8e638`. The [environment record](evidence/ipo-streaming-t4/environment.json)
contains source file hashes. The four original failed cells used base
`a6f5e72466991aabbce10b303047c997cedd71bc` and the patch recorded in
[their environment](evidence/ipo-streaming-t4/original-environment.json).

Kaggle Linux, one visible Tesla T4 (SM 7.5), FP16, driver
`580.178.04`. Python 3.13.16, Torch
`2.11.0+cu130`, CUDA 13.0, Transformers 5.20.2, TRL 0.29.1,
PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, datasets 4.8.5.
[Full package freeze](evidence/ipo-streaming-t4/pip-freeze.txt).

The two-layer untied Llama fixture has vocabulary 64, hidden size 64, intermediate
size 128, max length 64, seed 7, batch 1, and LoRA rank 4 on q/v. Runs use zero
warmup, `save_steps: 1`, and `ipo_tau: 0.1`. Streamed runs use the RAM tier.
Fixture SHA-256 `d81c1a44a04d2c1a8f539dddfee997db26d460721d8d27362509c6c8f2b37f80`.

These Tesla T4 figures establish a new baseline for this two-layer, vocabulary-64
model; they are not compared with the RTX 3050 or RTX 5070 records.
The JSON field `peak_allocated_bytes` records PyTorch allocator statistics through
adapter reload, including 234898432 bytes streamed and 17396224 bytes resident
for direct IPO without quantization at scale 1024; it is not a peak-VRAM
measurement or a memory-saving claim.

## Original default-scale one-step failures

All four original cells below failed with loss 25.0 and NaN gradient norm.
With the initial policy/reference log-ratio of zero, IPO's loss is
`(1 / (2 * ipo_tau))^2`. The separate direct-task diagnostic measured the skipped
update and unchanged adapter at scale 65536 in both modes and quantizations,
with backoff to 32768.

| Task | Quantization | Loss | Gradient norm | Original log |
|---|---|---:|---|---|
| `ipo` | none | 25.0 | NaN | [log](evidence/ipo-streaming-t4/original-ipo-legacy-none.log) |
| `ipo` | NF4 | 25.0 | NaN | [log](evidence/ipo-streaming-t4/original-ipo-legacy-4bit.log) |
| `preference`, loss `ipo` | none | 25.0 | NaN | [log](evidence/ipo-streaming-t4/original-ipo-preference-none.log) |
| `preference`, loss `ipo` | NF4 | 25.0 | NaN | [log](evidence/ipo-streaming-t4/original-ipo-preference-4bit.log) |

[Resident none](evidence/ipo-streaming-t4/ipo-diagnostic-False-None-none.json) and
[resident NF4](evidence/ipo-streaming-t4/ipo-diagnostic-False-None-4bit.json) controls recorded a skipped
first step with adapter delta zero, as did their
[streamed none](evidence/ipo-streaming-t4/ipo-diagnostic-True-None-none.json) and
[streamed NF4](evidence/ipo-streaming-t4/ipo-diagnostic-True-None-4bit.json) counterparts.

## Initial scale 1024, set in the harness only

Each cell applied its first update at step one. Both checkpoint and final
outputs contain eight policy and eight frozen reference tensors. The harness
checks exact tensor equality between checkpoint, final output, and reload on
the matching base. The reload comparison promotes saved and reloaded tensors to
FP32 so PEFT's lossless BF16 promotion can pass while any change in values fails.

The previous matrix applied all eight scale-1024 updates and both default runs
applied step five. Its two streamed NF4 checks failed at reload because the
harness also required the original BF16 dtype. The
[first-retry status](evidence/ipo-streaming-t4/first-retry-status.json) and
[first-retry patch](evidence/ipo-streaming-t4/first-retry-qualification.patch) preserve that attempt.
The tables below come from the rerun with the value comparison.

| Mode | Task | Quantization | Loss | Gradient norm | Max adapter delta |
|---|---|---|---:|---:|---:|
| streamed | `ipo` | none | 25.0 | 15.559629440307617 | 1.99999976757681e-05 |
| streamed | `ipo` | 4bit | 25.0 | 13.991411209106445 | 1.9999999494757503e-05 |
| streamed | `preference` | none | 25.0 | 15.559629440307617 | 1.99999976757681e-05 |
| streamed | `preference` | 4bit | 25.0 | 13.991411209106445 | 1.9999999494757503e-05 |
| resident | `ipo` | none | 25.0 | 26.783069610595703 | 1.99999976757681e-05 |
| resident | `ipo` | 4bit | 25.0 | 25.859407424926758 | 1.99999976757681e-05 |
| resident | `preference` | none | 25.0 | 26.783069610595703 | 1.99999976757681e-05 |
| resident | `preference` | 4bit | 25.0 | 25.859407424926758 | 1.99999976757681e-05 |

In this table and the default-settings table below, loss 25.0 is the analytic
initial value: zero-initialized LoRA B makes policy and reference equal, so that
column alone does not check the reference pass or streaming.
The streamed and resident gradient-norm columns are not claimed equal; the CUDA
cells do not synchronize or compare initial adapter weights, while the
[nonzero-adapter CPU test](../tests/test_ipo_streaming.py) copies them before
checking loss and gradients.

## Default settings, cap 12 attempts

The direct `ipo` spelling with quantization `none` ran at the default scale
65536 and stopped after the first applied update. The JSON rows record the scale,
skip flag, loss, gradient norm, and adapter delta at every attempt.

| Mode | Skipped | First applied | Scale at update | Loss | Gradient norm | Max adapter delta |
|---|---:|---:|---:|---:|---:|---:|
| streamed | 4 | 5 | 4096 | 25.0 | 15.559629440307617 | 1.99999976757681e-05 |
| resident | 4 | 5 | 4096 | 25.0 | 26.783069610595703 | 1.99999976757681e-05 |

## Reproduce

Check out the recorded upstream commit, apply the qualification patch, and use
the recorded package versions. Expose one Tesla T4 with
`CUDA_VISIBLE_DEVICES=0`. With `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
and `PYTHONPATH=src`, collect the ten GPU nodes with
`pytest tests/test_ipo_streaming.py -m gpu --collect-only --no-cov -q`. Run each
node in a separate process with a 600-second timeout. Set
`SOUP_TEST_2766_EVIDENCE_DIR` to an existing directory to retain the JSON rows.
The archived logs use placeholders for machine-local paths. Private notebook
addresses are omitted. Test IDs, counts, timings and numeric results are retained.

CPU tests check one shared store, materialized frozen reference weights, loss
and gradients on an untied head with nonzero adapters, checkpoint resume, and
runtime/save guards. [Seven negative controls](evidence/ipo-streaming-t4/cpu-mutations.json) fail the
intended tests when the corresponding protection is removed.

Local validation passed 390 targeted checks, 139 checks on the Transformers
5.16.1 / TRL 0.29.0 floor, 1,291 existing-caller checks, and 253 marker/docs/license
checks. CUDA cases are skipped in those CPU runs. Ruff, generated reference
structure and recipe snapshots passed. The full local suite was not run.
The [verification record](evidence/ipo-streaming-t4/verification.json) lists the
results. [Publication correspondence](evidence/ipo-streaming-t4/publication-correspondence.json)
records the qualified wrapper and test harness as byte-identical to the initial
PR head `c5ac634f`, along with the descriptions, diagnostic suggestions and fixture
registration changes at that head.

After merging upstream `bcd1e6be`, 391 integration checks passed with 28 skips.
At that head, the qualified wrapper and harness matched byte for byte. Wheel and sdist
builds passed, archive paths are unique, and the wheel includes 16 bundled text
guides. The wheel includes the existing UI logo and excludes benchmark/test data.

The review follow-up adds a CPU step at batch size 2 and gradient accumulation 2
through both IPO spellings, requiring finite loss, a changed adapter, nonzero
saved LoRA B and one accumulation substep. The helper accepts those settings
with the original defaults of 1 and 1. The IPO wrapper, CUDA test body and
fixture functions are unchanged; the recorded CUDA run still uses the patch
above. [Follow-up checks](evidence/ipo-streaming-t4/review-follow-up.json) record
the test-only delta and its verification.
