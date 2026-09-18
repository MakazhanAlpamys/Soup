# GPU test runs

CI has no GPU, so the CUDA-dependent tests (`pytest -m gpu`, see
[CONTRIBUTING.md](../CONTRIBUTING.md#gpu-tests)) only run when someone runs them on
a card. This file records every such run, so a regression on real hardware has a
last-known-good row to compare against. One row per card and date. Figures are
copied from the record each row links, not re-derived.

**Both recorded runs happened before the `gpu` marker existed (#833)**, so neither
is a `pytest -m gpu` run. The scope column says what was actually run.

| Date | Card (compute capability) | Driver / torch / bitsandbytes | Commit | Scope | Result | Found |
|---|---|---|---|---|---|---|
| 2026-08-06 (the session's setup date; STEP 1 itself is undated) | 8x NVIDIA H100 80GB HBM3 | 590.48.01 / 2.13.0+cu130 / 0.50.0 | not recorded | `tests/test_v07200.py`, `test_v07202.py`, `test_v07203.py`, `test_v07204.py`, `--no-cov` | `9 failed, 419 passed`; with one card visible (`CUDA_VISIBLE_DEVICES=0`): `6 failed, 422 passed` | three defects: streaming vs `nn.DataParallel` on a multi-GPU box, the #328 meta leak, a laptop-calibrated GEMM-ceiling bound ([`gate-h100-validation.md`](gate-h100-validation.md), STEP 1, FINDINGS 1–3) |
| 2026-09-10 | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), 8151 MiB | 616.92 (CUDA 13.4) / 2.14.0+cu130 / 0.50.2 | `bb0ae6b6` | full suite | `20709 passed, 2 failed, 186 skipped` | streamed NF4 not bit-exact vs resident NF4 on Blackwell: `test_issue385_stream_dtype.py` `[nf4]` in float16 and bfloat16 ([#776](https://github.com/MakazhanAlpamys/Soup/issues/776)) |

## Adding a run

Run `pytest tests/ -m gpu --no-cov -v` on a clean checkout, then send the details
listed in [CONTRIBUTING.md](../CONTRIBUTING.md#gpu-tests). A failure is as useful
as a pass: file it, and link the issue in the **Found** column.
