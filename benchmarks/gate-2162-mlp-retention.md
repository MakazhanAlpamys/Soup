# QKVO and MLP retention against native auto

Refs #2162. **The fixed B allocation passed its two-stage measurement rule:** 3.625068% less time per step in the six-block primary and 3.422014% in six fresh confirmation pairs. Both family-adjusted intervals have positive lower bounds. This record covers the measurement and benchmark code; a production option still needs a separate scope and review.

## Results

Each process contributes its median of 50 synchronized step durations. Each block/pair contributes 100 × (baseline − candidate) / baseline; the reported gain is the median of those six reductions. Positive means less time per step.

| Comparison | Median reduction | Ordinary 95% | Adjusted 99.583333% |
|---|---:|---|---|
| mixed_vs_native | 3.625068% | [2.984510%, 3.850795%] | [2.896977%, 3.882214%] |
| mixed_vs_qkvo | 3.127229% | [2.759756%, 3.598178%] | [2.637896%, 3.735478%] |
| qkvo_vs_native | 0.249797% | [0.174252%, 0.582551%] | [0.152429%, 0.793752%] |
| independent confirmation | 3.422014% | [2.686110%, 3.761645%] | [2.610860%, 3.764280%] |

The primary compares mixed retention with the original native auto K11. The same-K QKVO-only control isolates the added gate/up retention: mixed improves 3.127229% over that control, while the control improves 0.249797% over native. This supports saving gate/up outputs in this allocation. It does not make the earlier QKVO-only or projection-scoped results pass.

The frozen rule requires primary median ≥2% and positive ordinary and corrected lower bounds, plus positive mixed-versus-QKVO lower bounds and every correctness, memory and resource gate. Only then may six new AB/BA confirmation pairs run. Confirmation must independently meet median ≥2% and both positive lower bounds.

Bootstrap settings: 10,000 whole-block/whole-pair resamples, NumPy default_rng, seed 42 for the primary and 73 for confirmation, linear quantiles. Family 12 includes eight earlier contrasts, these three contrasts and confirmation; adjusted quantiles are 0.05/24 and 1−0.05/24. Six observations make the bootstrap discrete. These are small experiments on one card, not a guarantee of population coverage.

## Protocol and source identity

The [public B protocol](https://github.com/MakazhanAlpamys/Soup/issues/2162#issuecomment-6426785053) was posted before any B model process. The original wording is preserved in [RULE.md](results/issue2162-mlp-v1/mlp-budget-b-v1/RULE.md).

- Frozen: `2026-10-26T20:35:30.652760+00:00`; publication: `2026-10-26T20:39:11Z`.
- Protocol SHA256: `73c6a3b7d64a02e20da94e0d5e4353e31413999d5384e06e99072664c1215a5b`.
- Source ZIP SHA256: `cc6177df580e7801e6e652a136452696b41df37238c0d594fae1c020289344c5`.
- Base: `094644e30fd714190562a1dcf3b38f8680a091c5`, plus the frozen benchmark helpers/tests. The helpers were uncommitted at measurement time. Their exact bytes are now bound to Git blobs in `source-blobs.json`, with explicit LF/CRLF reconstruction. The source ZIPs are retained locally and in the lab, rather than checked in.

## Allocation, recipe and exactness

| Arm | Whole retained layers | Cached frozen base outputs | Raw cache bytes |
|---|---|---|---:|
| native | 21..31 (K11) | none | 0 |
| qkvo | 26..31 (K6) | Q/K/V/O in 0..25 | 1,090,519,040 |
| mixed | 26..31 (K6) | same QKVO; gate/up in 15..25 | 2,382,364,672 |

Mistral-7B NF4/double quantization, bf16, batch 1 × sequence 2048, pinned RAM and two stream buffers. The original model/shard identity is preserved. LoRA targets q/k/v/o, rank 8, alpha 16, dropout 0; 32 layers × four targets × A/B gives 256 trainable tensors. Model seed 42, nonzero B seed 7, input seed 1234; PagedAdamW8bit at 2e-4, deterministic algorithms, TF32 off, OMP/MKL threads 1. The observed backend is SDPA efficient attention.

Six initial processes checked normal/profile execution for all three arms. Thirty timing processes each began with two exact reference updates, then finished ten warmups and 50 measured updates. Totals: **72 exact reference-update replays, 1,500 measured updates, 300 warmups and 1,812 optimizer updates**. Logits, loss, all 256 gradients and all 256 updated tensors matched bit for bit. The profiler confirmed removal of 44 selected gate/up recomputation matmuls over two mixed updates.

## Memory

```text
[cyan]training.stream_recompute='auto':[/] the last 11 of 32 streamed layers keep their activations, the rest recompute (measured at batch 1 x seq 2048: one layer 420.0 MB retained, charged 482.0 MB at reserved/allocated 1.15; budget 5.76 GB = training.stream_vram_override 8.59 GB minus the reserved peak 2.83 GB)
```

Both the allocator and `training.stream_vram_override` enforce 8,589,934,592 bytes. The census reproduces charge 482,015,738 bytes per whole layer and a 5,756,682,240-byte budget. B charges 2,734,318,038 cache bytes within the K6 residual 2,864,587,812 bytes, leaving 130,269,774 charged bytes.

Peak allocated/reserved memory: **6.439509 / 6.759766 GiB**. All 30 allocated-increase checks passed: mixed−QKVO, mixed−native and QKVO−native in each gate variant and primary block, plus mixed−native in every confirmation pair. The allowance is saved raw bytes + 64 MiB. Mixed−native increases allocated memory by 366,390,272 bytes, net of the five whole layers no longer retained. Mixed−QKVO adds exactly 1,291,845,632 bytes for the MLP outputs.

## Hardware and activity

Environment snapshot recorded during the gate phase: `Windows-11-10.0.26200-SP0`; Python `3.13.7`. Packages: torch `2.14.1+cu130`, transformers `5.19.0`, bitsandbytes `0.50.2`, numpy `2.5.4`.

```text
0, GPU-05919122-a78e-9af3-3b3c-f9a14fb4d2f4, 610.88, NVIDIA GeForce RTX 3090, 420.00
1, GPU-7b0178c9-5f01-96cd-6911-25d9e5fa4126, 610.88, NVIDIA GeForce RTX 3090, 350.00
2, GPU-7c1f2b70-2fd5-9552-b36f-8bb959f66fd6, 610.88, NVIDIA GeForce RTX 3090, 420.00
3, GPU-928172b9-95ec-7dce-c9aa-5bcbe966c3f7, 610.88, NVIDIA GeForce RTX 3090, 420.00
```

Measured-window samples: selected GPU 64–70 °C, SM clocks 1290–1965 MHz; other GPUs at most 2% utilization. Maximum sample gap: 1.228918 s. No sampled Windows/Linux CI worker was active. Sampling does not rule out activity between probes or host contention. The 24 GiB RTX 3090 was capped at 8 GiB; this does not establish performance on a physical 8 GiB card.

## Every timing process

Executed order, median seconds per step, and allocated/reserved bytes including warmups. All 50 durations and losses per process are reconstructible from the evidence.

| Process | Median seconds | Allocated bytes | Reserved bytes |
|---|---:|---:|---:|
| block1-qkvo | 1.811098300 | 5,622,524,416 | 5,966,397,440 |
| block1-mixed | 1.756258050 | 6,914,370,048 | 7,258,243,072 |
| block1-native | 1.825588950 | 6,547,979,776 | 7,182,745,600 |
| block2-native | 1.831274850 | 6,547,979,776 | 7,182,745,600 |
| block2-mixed | 1.760180850 | 6,914,370,048 | 7,258,243,072 |
| block2-qkvo | 1.828483450 | 5,622,524,416 | 5,966,397,440 |
| block3-qkvo | 1.824421900 | 5,622,524,416 | 5,966,397,440 |
| block3-native | 1.829289650 | 6,547,979,776 | 7,182,745,600 |
| block3-mixed | 1.776295550 | 6,914,370,048 | 7,258,243,072 |
| block4-mixed | 1.760634300 | 6,914,370,048 | 7,258,243,072 |
| block4-native | 1.830549900 | 6,547,979,776 | 7,182,745,600 |
| block4-qkvo | 1.823752150 | 5,622,524,416 | 5,966,397,440 |
| block5-native | 1.834803750 | 6,547,979,776 | 7,182,745,600 |
| block5-qkvo | 1.830519600 | 5,622,524,416 | 5,966,397,440 |
| block5-mixed | 1.771458850 | 6,914,370,048 | 7,258,243,072 |
| block6-mixed | 1.775544650 | 6,914,370,048 | 7,258,243,072 |
| block6-qkvo | 1.828227150 | 5,622,524,416 | 5,966,397,440 |
| block6-native | 1.831818900 | 6,547,979,776 | 7,182,745,600 |
| confirm1-native | 1.830035050 | 6,547,979,776 | 7,182,745,600 |
| confirm1-mixed | 1.782255400 | 6,914,370,048 | 7,258,243,072 |
| confirm2-mixed | 1.775671900 | 6,914,370,048 | 7,258,243,072 |
| confirm2-native | 1.826097000 | 6,547,979,776 | 7,182,745,600 |
| confirm3-native | 1.833008450 | 6,547,979,776 | 7,182,745,600 |
| confirm3-mixed | 1.766125250 | 6,914,370,048 | 7,258,243,072 |
| confirm4-mixed | 1.774789050 | 6,914,370,048 | 7,258,243,072 |
| confirm4-native | 1.833369000 | 6,547,979,776 | 7,182,745,600 |
| confirm5-native | 1.834978900 | 6,547,979,776 | 7,182,745,600 |
| confirm5-mixed | 1.765905150 | 6,914,370,048 | 7,258,243,072 |
| confirm6-mixed | 1.766276050 | 6,914,370,048 | 7,258,243,072 |
| confirm6-native | 1.835263800 | 6,547,979,776 | 7,182,745,600 |

## Earlier diagnostic and historical comparisons

Candidate A removed the intended gate/up work, but its analytical planning estimate was 2.971433%, below the fixed 3% screen. It had no timing pairs and remains ineligible under that protocol. B was a separately declared allocation with an analytical 3.990470% estimate. A's failed screen remains failed.

A's first profiler process failed because ContextVar phase labels did not reach the autograd thread. Its two exact updates remain recorded. Recovery reused the three successful normal gates and retried only the failed profiler, using a thread-visible observer that saved raw events. A therefore has 12 accepted updates plus two failed-profile updates, 14 total. The corrected parser distinguishes zero-device down-projection early-stop boundaries from executed QKVO/gate/up kernels. A generated pytest metadata file absent from the source ZIP is explicitly identified in the source bindings; no missing source is silently skipped.

The original QKVO outcomes and verdicts remain in [the earlier report](gate-2162-auto-qkvo.md). Below are their descriptive intervals recomputed with family 12; these do not replace the original decisions.

| Prior comparison | Median reduction | Family-12 interval |
|---|---:|---|
| timing-v1 / 2048 | 1.421695% | [0.656295%, 1.597937%] |
| timing-v1 / 4096 | 0.607885% | [-0.452782%, 2.099459%] |
| reallocation-v1 / combined_vs_native | 1.860896% | [1.739993%, 2.647195%] |
| reallocation-v1 / combined_vs_reduced | 2.509463% | [2.148977%, 2.680395%] |
| reallocation-v1 / reduced_vs_native | -0.633966% | [-0.850357%, 0.509160%] |
| scoped-v2-resume1 / scoped_vs_native | 1.997942% | [1.437839%, 2.594954%] |
| scoped-v2-resume1 / scoped_vs_legacy | 0.045541% | [-1.264484%, 0.342097%] |
| scoped-v2-resume1 / legacy_vs_native | 2.034504% | [1.423084%, 3.156867%] |

## Reproduce without CUDA

The compact record reconstructs 281 original files byte for byte, including raw profiler events and telemetry. Tensor hashes are shared with the earlier record; repeated profiler/GPU strings use dictionaries. Every reconstructed file is checked against its original SHA256 before the two independent auditors run. Their complete outputs must equal the original closure audits.

```sh
python benchmarks/harness/issue2162_mlp_record.py --evidence benchmarks/results/issue2162-mlp-v1 --verify-git-sources . --output audit-2162-mlp.json
python benchmarks/harness/issue2162_mlp_report.py --evidence benchmarks/results/issue2162-mlp-v1 --output report-2162-mlp.md
```

The closure verified four services, two real inference requests, three containers, original power limits and 41 model/shard hashes. Full original ZIPs, closure receipts and reports remain preserved locally and in the lab. This submission does not rerun any model or change production code.
