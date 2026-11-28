# Gate record - Fast-LoRA operator qualification and NF4 rounding

**Verdict (2026-10-12) — Positive for the corrected numerical fixtures.** Both NF4 wrapper matrices passed. The timing arms yielded no verdict.

Measured by @kok-o.

## 1. Setup

Can Fast-LoRA match the original PEFT projection outputs and gradients when an NF4 base is wrapped by either regular PEFT Linear or native PEFT Linear4bit?

- NVIDIA GeForce RTX 3090, 24 GB, compute capability 8.6, Linux and Python 3.12.3. The pilot used driver 595.71.05 and the corrected run 580.167.08. torch 2.11.0+cu126, PEFT 0.20.0, Transformers 5.17.0, bitsandbytes 0.50.1 and Triton 3.6.0. The pilot also recorded Accelerate 1.14.0. Host RAM and the Linux release are unspecified in the reports.
- Synthetic single, QKV and MLP projections, with input gradients on and off. Dense probes use FP32, FP16 or BF16. NF4 probes use FP16 or BF16 compute, FP32 adapters, nonzero LoRA B and checkpointing on and off. The separate streamed control uses four synthetic layers and two slots. The tested NF4 support for `training.fast_lora` was pending in the CLI at the publication base.

## 2. Method and results

The two PEFT wrappers round at different points. Regular Linear adds the FP32 update before narrowing the result. Native Linear4bit narrows the scaled update before adding it. The earlier Fast-LoRA path selected native rounding whenever it saw NF4 quantization state. The corrected implementation records the actual wrapper and preserves its cast/add order. Both low-precision arms used identical packed bases and quantization state. The float64 oracle dequantized the same weights.

Replay required every expected output and gradient, finite values and matching shapes and dtypes. The FP16/BF16 rule was `fast_max_abs_error <= 2 * peft_max_abs_error + 1e-8` against float64. The FP32 checks kept the original `assert_close` rule.

| Corrected fixture | Result |
|---|---|
| Dense FP32 / FP16 / BF16 | All 123 expected quantities retained and passed |
| 50-update loss probe, seed 792 | 50/50 comparisons match to three decimals in each dtype |
| NF4 with regular Linear, tiny and aligned | 48/48 cases pass |
| NF4 with native Linear4bit, tiny and aligned | 48/48 cases pass |
| Four-layer NF4 streaming | 56 adapter gradients plus logits and loss per dtype pass |
| Changed-adapter streamed controls | Both rejected, 20 observed custom execution sites each |
| Selected GPU regressions | 15 passed, zero failures, errors or skips |

| Replay dtype | Exact forward / checked | Exact backward / checked |
|---|---:|---:|
| FP32 | 10 / 10 | 29 / 31 |
| FP16 | 92 / 92 | 333 / 335 |
| BF16 | 92 / 92 | 333 / 335 |

Independent CPU replay checked 895 positive quantities from 116 files with zero failures. All 772 NF4 and streamed positive quantities were exact. The six numerical tolerance passes were dense input-gradient comparisons.

## 3. Commands

The harness lives at the evidence commit. The pilot's GPU harnesses were `benchmarks/harness/d2_cloud_run.py` and `benchmarks/harness/d2_nf4_probe.py`. The archived handoff preserves GPU commands and source snapshots. Extract the corrected run archive, set `BASE` to its root and run the recorded CPU replay.

```bash
python benchmarks/harness/d2_audit_evidence.py "$BASE/run2-acceptance" \
  --aligned-dir "$BASE/run2-acceptance/nf4-dense-bf16-nf4-aligned" \
  --aligned-dir "$BASE/run2-acceptance/nf4-bnb-bf16-tiny" \
  --aligned-dir "$BASE/run2-acceptance/nf4-bnb-bf16-nf4-aligned" \
  --output /path/to/new-audit.json
```

## 4. Superseded runs and limits

The corrected matrix superseded the pilot's numerical verdict. The pilot replay covered 68 files and 567 quantities, with three failures in tiny BF16 QKV, tiny FP16 MLP and aligned BF16 MLP. Layout, contiguous-input and linear-versus-matmul controls left the failures in place. A same-host rounding-policy prototype passed 48 cases before the full wrapper matrix was repeated.

All 240 raw timing arms per dtype in the corrected run failed the peer-process guard. The pilot timing remains descriptive. The operator fixtures leave pretrained-model quality, full-training performance and standalone memory savings unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-rtx3090-20261012-v7
