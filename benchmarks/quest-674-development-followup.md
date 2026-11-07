# Soup #674 development follow-up

The closest retained W4A16 reconstruction reaches response-NLL ratios of 1.016327 and 1.016500 on DEV2048. Neither replica passes the frozen zero-margin parity gate. The public mixed QuEST route and strict native W4A4 remain separate measurements, with larger gaps. This record requests a decision about future acceptance criteria; it does not change a runtime default or validate production readiness.

The previous [W4A4 feasibility](gate-674-quest-w4a4-sft.md) and [post-fit](quest-674-postfit-development-results.md) failures remain unchanged. Issue [#674](https://github.com/MakazhanAlpamys/Soup/issues/674) stays open.

## Measurement contract

Model: `ahxt/LiteLlama-460M-1T`. The follow-ups used RTX 3090 GPUs, FP32 master parameters and BF16 matrix operations. The preserved native-study stack includes PyTorch 2.11.0+cu128 and Transformers 5.16.1. The labels W4/A4 describe the simulated projection quantization, not packed storage or integer kernels.

DEV704, DEV1024 and DEV2048 are all development-exposed. Each new fit used the same frozen 512 TRAIN rows; TRAIN/DEV row, instruction and source-group exclusion checks passed. DEV2048 contains 2,048 rows and 127,395 response targets. FINAL256 and OASST256 remain unopened; DEV4096 was not evaluated.

Ratios below use the strongest fixed FP on each panel. A value of 1.000000 is point NLL parity; 1.016327 means about 1.63% more NLL, not 1.63% less accuracy and not a perplexity ratio. The selected FP may differ between panels. The earlier families retain all 17 FP comparators; the latest mixed studies also retain the newly measured legacy FP as an 18th.

Each frozen study used 100,000 paired-row bootstrap draws, seed 674, target-token weighting and linear quantiles. For a family of m comparisons, the one-sided corrected upper is quantile `1 - 0.025/m`. Zero-margin parity requires every point gap and corrected upper to be <=0, across both replicas and all three panels. Improvement gates require negative gaps and uppers. These are within-campaign corrections; they do not remove bias from the wider adaptive search.

## W4A16 reconstruction

The 4B label is an update-budget label, not a model size: it doubles the retained 2B per-block alpha and scale schedules to 4,096 updates each, for 196,608 optimizer updates per new model over 24 blocks. Each fit starts from the same original dqscale-v2 initialization. It reconstructs fixed adjacent UINT4 pairs and scales at group size 128, with integer zero points, 168 W4 projections and 51 fixed non-W4 tensors. Activations remain A16. TRAIN-only hardening and rollback select the terminal block state; DEV does not select checkpoints.

This is affine reconstruction with dequantized FP32 storage, not native QuEST W4A4 training or a deployable packed-INT4 backend. Seeds 42/7 vary reconstruction order within one pretrained lineage. Primary `wide7` was fixed before scoring; `wide42` is only the descriptive best on DEV2048.

| Panel | 2B seed42 | 4B seed42 | 2B seed7 | 4B seed7 (primary) |
|---|---:|---:|---:|---:|
| spent704 | 1.024161549 | 1.023401119 | 1.023623541 | 1.023568303 |
| new1024 | 1.018285958 | 1.017583947 | 1.017395575 | 1.017186842 |
| dev2048 | 1.017033818 | 1.016326642 | 1.016802834 | 1.016500479 |

Only 2/6 corrected 4B-minus-2B improvements passed. Both were seed42 on DEV1024/2048; seed7 did not replicate them. Budget advancement and replicated parity are false. The 102-comparison parity family failed for both arms. The local independent audit recomputed all 108 contrasts, checked 48 checkpoints and nine exact panel replays. This did not justify another budget increase.

## Native strict W4A4 follow-ups

Two reconstruction fits used the stronger FP initialization directly, native 168 W4 / 168 A4 projections, fixed TRAIN calibration, and 24 blocks x512 updates per fit. All 18 corrected reconstruction-improvement comparisons passed, which permitted the preregistered short global refinement.

That refinement made 256 updates per replica on 512 TRAIN rows, with inherited clips fixed and response-only teacher KL plus normalized hidden-state MSE. Its primary `refine7` was fixed before scoring.

| Panel | Reconstructed42 | Refined42 | Reconstructed7 | Refined7 (primary) |
|---|---:|---:|---:|---:|
| spent704 | 1.065106136 | 1.062802455 | 1.066378514 | 1.067113806 |
| new1024 | 1.059014148 | 1.055740377 | 1.062741012 | 1.057564764 |
| dev2048 | 1.061039703 | 1.057020196 | 1.063569503 | 1.059345883 |

The short refinement passed 6/12 corrected improvement comparisons and failed the advancement gate. Both strict parity gates remain false. The separate local audits covered 120 reconstruction and 114 refinement contrasts. These native research forwards and export/reload checks do not establish strict support through the public training command.

## Retained public mixed route and transfer experiment

The unchanged public mixed route is 168 W4 / 161 A4 +7 A16. Its retained historical masters were measured prospectively on the current instrument. Four matched reconstruction fits then compared weight-only controls with joint weights/positive row amplitudes, seeds 42/7. Each fit ran 12,288 updates; temporary amplitudes were materialized into ordinary FP32 masters before export. Activation clips, the public native grid and all 51 non-W4 tensors stayed fixed. Primary `joint7` was fixed before fitting.

| Panel | Retained mixed | Weights42 | Weights7 | Joint42 | Joint7 (primary) |
|---|---:|---:|---:|---:|---:|
| spent704 | 1.040378418 | 1.049477246 | 1.048262907 | 1.046618698 | 1.049218205 |
| new1024 | 1.123946968 | 1.072270888 | 1.066718099 | 1.068694310 | 1.071285300 |
| dev2048 | 1.122956701 | 1.073863672 | 1.068899394 | 1.071794713 | 1.072082102 |

Reconstruction improved DEV1024/2048 against the retained mixed checkpoint but regressed DEV704. Only 6/18 corrected joint-improvement comparisons passed; the family required both joint arms to beat both controls and the baseline on every panel. The additional amplitudes did not demonstrate a consistent benefit. The 216 parity comparisons failed to establish parity for any arm. The local independent audit recomputed all 240 contrasts and verified 96 full block checkpoints, four discarded smoke checkpoints, eight native reloads and four native/cache equivalences. No further fit followed this failed advancement gate.

The preceding mixed-baseline audit covered 60 contrasts, 12 exact current-instrument panel replays, and 128 fresh-process replay rows. The retained W4A16 rows replayed exactly. The older published legacy FP losses did not: matching artifact hashes, inputs and tested loader variants did not attribute the discrepancy. That baseline-v2 attempt remains failed. Baseline-v3 is a new prospective current-instrument measurement, not an approximate historical replay, and does not replace the older published numbers or gate.

## Acceptance question and remaining production checks

Is zero-margin FP parity a hard requirement for #674, or would a separately scoped, opt-in W4A16 feature with a declared bounded quality loss be worth considering? If so, which prospective quality margin and practical task tests would the project require, together with measured memory/runtime benefits, before calling that feature production-ready?

A different acceptance policy would apply to a new validation protocol. It would not turn these failed parity gates into passes. W4A16 would also require its own implementation scope: the best reconstruction artifact is not the merged QuEST mixed route. Public-command training quality, seeds 42/7/123, a frozen final-panel protocol and actual efficiency measurements remain outstanding. FP16 remains refused under the existing QuEST decision; this record supplies no FP16 evidence.

## Evidence and limits

The accompanying [JSON evidence](evidence/quest-674-development-followup-v1.json) retains all 642 aggregate contrasts from these five audited studies, their NLLs and ratios, fixed gate decisions, and hashes of the preserved source protocols/results/archives. It contains no private infrastructure or raw training text. Source checkpoints, per-row losses and full runners remain local, so this PR alone cannot reproduce training or recompute bootstrap intervals. The hashes provide traceability to the preserved evidence, not a claim of independent public reproduction.

All source studies have audited closure, exact required native/cache or replay checks, and preserved model/optimizer evidence. No model was run again for this publication. One pretrained lineage, two order seeds and repeated development selection limit generalization. There is no production, cross-model, packed-INT4 or efficiency claim.

Evidence file SHA-256: `807794c8d587ed26ce32600ae23e03ff6ac16ca3d8804d0287aeb962cf7bee05`.
