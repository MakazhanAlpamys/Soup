# Fast-LoRA parity for RAM and disk streamed SFT

**Verdict (2026-10-18): positive.** The final CUDA fixtures passed the streamed PEFT comparison for dense and nested NF4 weights in FP16 and BF16 on both storage tiers.

Measured by @kok-o.

## Question and setup

Can the Fast-LoRA MLP, QKV and single-projection backward kernels run through the real layer-streamed SFT trainer, complete optimizer updates and export ordinary PEFT adapters?

- Linux RTX 3090 (sm86, 24 GB). Recorded stack was torch 2.11.0+cu126, Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1 and bitsandbytes 0.50.1. The main report identifies the CUDA build and leaves host RAM, driver and Linux release to the archived hardware evidence.
- CPU controls ran on Windows with Python 3.10 and torch 2.13.0+cpu with the same recorded Transformers, PEFT, TRL and bitsandbytes versions.
- The fixture is a local random four-layer Llama. Dense and nested NF4 bases use native FP16 and BF16 execution. The streamed trainer with two slots alternates RAM and async disk. The reference uses streamed PEFT without the patches, with the same initialization, placement and weight format. Layer-streamed support for `training.fast_lora` was pending in the CLI at the publication base.

## Method and results

With nonzero initial LoRA B, each update accumulates two microbatches. Compare rounded loss, logits, all 56 adapter tensors and gradients across two optimizer steps. Inspect actual custom backward execution and checkpoint recomputation. Validate periodic and final canonical adapters by loading them with ordinary PEFT.

| Evidence | Coverage | Outcome |
|---|---|---|
| Final CUDA integration | 8 cases, dense/NF4 × FP16/BF16 × RAM/disk | All passed |
| Independent CPU replay of saved CUDA tensors | 1,832 paired comparisons | Exact in the retained finite fixtures |
| Supplementary NF4 oracle replay | 116 quantities | Exact Fast-LoRA/PEFT tensors in the retained finite fixtures |

The CUDA gate kept the three-decimal loss rule and used rtol=1e-2 and atol=1e-5 for low-precision tensors. The supplementary oracle used the predeclared twice-reference-error bound.

## Reproduction

The test and the executed source snapshot live at the evidence commit. Use the recorded package freeze and inspect the GPU JUnit records alongside the saved tensors.

```bash
python -m pytest tests/test_fast_lora_streaming.py -m gpu --no-cov -v
```

## Limits and superseded runs

The first CUDA pilot passed 6 cases and failed 2 NF4/FP16 cases in mixed-dtype backward. The second pilot passed 16 tests, but the supplementary check refused because its decision rules were missing from the archive. The final run used the corrected AMP handling and byte-checked rules. It supersedes both pilots for qualification.

CPU replay uses stored tensors. The fixture establishes wiring, update and save/reload behavior. It leaves real-data task quality, streamed peak-memory savings and speed unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-fast-lora-streaming-20261018
