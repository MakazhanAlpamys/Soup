# Incremental QKVO retention over native auto

Refs #2162. **INCONCLUSIVE:** the final primary measured 1.997942% less time per step, below its predeclared 2% threshold. Projection-scoped dispatch also failed to improve reliably over layer-scoped dispatch. The conditional confirmation did not run. This record adds no production field or default.

## Results

Each percentage is the median of paired percentage reductions computed from each process's median step time. Positive means less time. The control result 2.034504% does not replace the primary.

| Experiment / comparison | Median reduction | Ordinary 95% | Adjusted interval |
|---|---:|---|---|
| timing-v1 / seq2048 | 1.421695% | [0.656295%, 1.597937%] | [0.656295%, 1.597937%] |
| timing-v1 / seq4096 | 0.607885% | [-0.452782%, 2.099459%] | [-0.452782%, 2.099459%] |
| reallocation-v1 / combined_vs_native | 1.860896% | [1.783880%, 2.280525%] | [1.739993%, 2.280525%] |
| reallocation-v1 / combined_vs_reduced | 2.509463% | [2.233214%, 2.661054%] | [2.233214%, 2.661054%] |
| reallocation-v1 / reduced_vs_native | -0.633966% | [-0.843195%, -0.040999%] | [-0.843195%, -0.040999%] |
| scoped-v2-resume1 / scoped_vs_native | 1.997942% | [1.685070%, 2.377408%] | [1.437839%, 2.594954%] |
| scoped-v2-resume1 / scoped_vs_legacy | 0.045541% | [-0.733307%, 0.244185%] | [-0.733307%, 0.244185%] |
| scoped-v2-resume1 / legacy_vs_native | 2.034504% | [1.613962%, 2.838812%] | [1.423084%, 2.838812%] |

The first experiment uses a two-cell family (97.5% adjusted intervals). Each later experiment uses three contrasts (98.3333%). All use 10,000 whole-pair or whole-block bootstrap resamples, NumPy default_rng(42), and linear quantiles. Five pairs or six blocks produce discrete bootstrap distributions; these intervals are descriptive of small experiments, not a guarantee of population coverage. They are the original per-experiment corrections, not a correction across this adaptive sequence of experiments.

The final family interval spans 2%. The result establishes neither that the population gain is above 2% nor that it is below 2%. Its numerical acceptance rule still failed. The earlier 1.421695% and 1.860896% failed rules remain failed.

## Recipe and exactness

- Model: `unsloth/mistral-7b-instruct-v0.3`, revision `60b99dcbc7f853b7cd18472a607e8d045b96f03e`; existing NF4 shards, bf16 compute, pinned RAM and two stream buffers.
- LoRA: rank 8, alpha 16, dropout 0, bias none, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`. There are 32 layers x 4 target modules x 2 trainable matrices (A/B) = **256** gradient tensors and 256 updated parameter tensors. #2161 used two targets (q/v), hence 128.
- Model seed 42, nonzero B seed 7, synthetic token seed 1234, batch 1, sequence 2048 (4096 in the first secondary cell). PagedAdamW8bit, lr 2e-4; OMP/MKL threads 1, deterministic algorithms, TF32 disabled.
- Actual selected backend: SDPA efficient attention, verified by profiler operators. Torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2. #2161 used forced SDPA math, so its 3.14% comparison against full recompute is not this comparison against auto.
- All 84 successful full-model processes checked two complete optimizer updates, including logits, loss, every gradient and every updated parameter bit-for-bit. The 56 timing processes replayed the reference in their first two warmups, then completed 10 warmups total and 50 measured updates each (2,800 measured updates; 560 warmups). The 28 untimed gates add 56 updates.

## Changes between the frozen experiments

The initial public plan kept native K fixed. Later local protocols changed the implementation and design after seeing earlier results. They were frozen locally before their own runs, not publicly preregistered. The JSON times and resource timestamps below are local provenance, not an external timestamp attestation. Earlier experiments are retained rather than pooled or relabelled.

| Experiment | Locally frozen UTC | First timing process UTC | Design |
|---|---|---|---|
| timing-v1 | 2026-10-26T12:39:51.346420+00:00 | 2026-10-26T12:39:53.517617+00:00 | Native K retained: seq2048 K11 + QKVO9; seq4096 K3 + QKVO8; five AB/BA pairs per cell. |
| reallocation-v1 | 2026-10-26T14:33:19.507437+00:00 | 2026-10-26T14:36:14.483247+00:00 | Seq2048: native K11, reduced K10 without QKVO, K10 + QKVO19; six three-arm blocks. |
| scoped-v2-resume1 | 2026-10-26T16:24:33.864396+00:00 | 2026-10-26T16:31:03.432718+00:00 | Seq2048: native K11, layer-scoped K10 + QKVO19, projection-scoped K10 + QKVO19; six three-arm blocks. Conditional six-pair confirmation was ineligible. |

Both six-block orders include all permutations, shuffled once with Python random.Random(42). Their frozen rule requires primary median >=2% and positive ordinary and adjusted lower bounds. The last experiment additionally requires positive lower bounds for scoped_vs_legacy. Only then may the frozen stage B run six new native/scoped pairs (seed 73, ordinary 95% lower bound >0 and median >=2%). Stage B cannot rescue a failed stage A.

The source commits and all Python file hashes are in each `source-blobs.json` and `source-hashes.json`. The original freeze and protocol hashes are retained. The measurements refer to those commits, not later changes on main.

## Memory and the original auto decision

Both the allocator limit and `training.stream_vram_override` were **8,589,934,592 bytes (8 GiB)**. The native seq2048 message, preserved in `census-v1/logs.jsonl` with its original wrapping, is:

```text
training.stream_recompute='auto': the last 11 of 32 streamed layers keep their
activations, the rest recompute (measured at batch 1 x seq 2048: one layer
420.0 MB retained, charged 482.0 MB at reserved/allocated 1.15; budget 5.76 GB
= training.stream_vram_override 8.59 GB minus the reserved peak 2.83 GB)
```

The underlying integers are 419,972,096 retained bytes per whole layer, 482,015,738 charged bytes, a 5,756,682,240-byte budget and 2,833,252,352-byte probe reserved peak. Native K11 leaves 454,509,122 bytes; K10 leaves 936,524,860. The latter holds 19 QKVO caches (layers 3..21), 796,917,760 raw bytes / 914,648,638 charged bytes, alongside full retention of layers 22..31.

The owner-requested #2161 memory check is evaluated here as an additional publication check: candidate allocated peak minus comparator allocated peak <= saved bytes + 64 MiB in every pair/block. The original #2162 protocols required exact cache accounting and both peaks <=8 GiB. These retrospective memory checks do not change their timing verdicts. Original peaks are exact bytes, so no decimal-GB rounding allowance is needed.

| Experiment / contrast | Max allocated increase (bytes) | Saved bytes | Memory check |
|---|---:|---:|---|
| timing-v1 / 2048 | 377,487,360 | 377,487,360 | PASS |
| timing-v1 / 4096 | 671,088,640 | 671,088,640 | PASS |
| reallocation-v1 / combined_vs_native | 393,722,880 | 796,917,760 | PASS |
| reallocation-v1 / combined_vs_reduced | 796,917,760 | 796,917,760 | PASS |
| reallocation-v1 / reduced_vs_native | -403,194,880 | 0 | PASS |
| scoped-v2-resume1 / scoped_vs_native | 393,722,880 | 796,917,760 | PASS |
| scoped-v2-resume1 / scoped_vs_legacy | 0 | 796,917,760 | PASS |
| scoped-v2-resume1 / legacy_vs_native | 393,722,880 | 796,917,760 | PASS |

For the K10 QKVO arms, the increase versus native K11 is net of the whole layer no longer retained. The reduced-K10 control in the reallocation experiment isolates the cache cost: exactly 796,917,760 additional allocated bytes. Final candidate allocated/reserved peaks are 6,941,702,656 / 7,291,797,504 bytes (6.464964 / 6.791016 GiB). Native K11 peaks are 6,547,979,776 / 7,182,745,600 bytes.

## Hardware and competing activity

Selected card: RTX 3090 24 GiB, UUID `GPU-05919122-a78e-9af3-3b3c-f9a14fb4d2f4`, unchanged 420 W power limit. This is a capped 24 GiB card, not a physical 8 GiB card. Other cards kept 350/420/420 W and existing services.

The read-only inventory at 2026-10-26 19:02 UTC reports NVIDIA driver 610.88 and Windows 11 Pro for Workstations 10.0.26200, last boot 12:07:23.5 UTC. **OS and driver were captured after the experiments**; the timing receipts do not independently attest those versions throughout the campaign. The following temperatures, SM clocks and other-GPU utilization are contemporaneous samples inside measured windows:

| Experiment | Temperature C | SM clock MHz | Other-GPU max utilization | Max sampling gap s |
|---|---|---|---:|---:|
| timing-v1 | 63..71 | 870..1980 | 2% | 1.381981 |
| reallocation-v1 | 62..70 | 1335..1965 | 2% | 1.226223 |
| scoped-v2-resume1 | 63..70 | 1350..1980 | 2% | 1.369663 |

The other GPUs were not assumed idle: their sampled utilization reached 2%, below the fixed 5% admission/timing limit. Windows CI workers were checked each sample, Linux CI workers every fifth sample; none was reported active. Sampling cannot exclude activity between probes or host contention. All raw samples, including RSS, free RAM, host CPU and all four GPUs, remain in the record.

## Every timing block

These tables preserve the executed order. Each value is median seconds per step, followed by peak allocated / reserved **bytes**, including warmups. All 50 individual durations and losses per process are in each `timings.jsonl`.

### timing-v1

| Pair/block | Order and measurements |
|---|---|
| seq2048-p1 | seq2048-p1-auto: **1.803038450 s**, 6547979776 / 7182745600; seq2048-p1-qkvo: **1.791205200 s**, 6925467136 / 7291797504 |
| seq2048-p2 | seq2048-p2-qkvo: **1.804668450 s**, 6925467136 / 7291797504; seq2048-p2-auto: **1.833974200 s**, 6547979776 / 7182745600 |
| seq2048-p3 | seq2048-p3-auto: **1.829062500 s**, 6547979776 / 7182745600; seq2048-p3-qkvo: **1.808394500 s**, 6925467136 / 7291797504 |
| seq2048-p4 | seq2048-p4-qkvo: **1.810208450 s**, 6925467136 / 7291797504; seq2048-p4-auto: **1.837045950 s**, 6547979776 / 7182745600 |
| seq2048-p5 | seq2048-p5-auto: **1.835410900 s**, 6547979776 / 7182745600; seq2048-p5-qkvo: **1.809316950 s**, 6925467136 / 7291797504 |
| seq4096-p1 | seq4096-p1-auto: **4.395995950 s**, 6043057664 / 6840909824; seq4096-p1-qkvo: **4.303703800 s**, 6714146304 / 8015314944 |
| seq4096-p2 | seq4096-p2-qkvo: **4.316226850 s**, 6714146304 / 8015314944; seq4096-p2-auto: **4.352844200 s**, 6043057664 / 6840909824 |
| seq4096-p3 | seq4096-p3-auto: **4.366560250 s**, 6043057664 / 6840909824; seq4096-p3-qkvo: **4.340838400 s**, 6714146304 / 8015314944 |
| seq4096-p4 | seq4096-p4-qkvo: **4.353400600 s**, 6714146304 / 8015314944; seq4096-p4-auto: **4.380026100 s**, 6043057664 / 6840909824 |
| seq4096-p5 | seq4096-p5-auto: **4.370068000 s**, 6043057664 / 6840909824; seq4096-p5-qkvo: **4.389854900 s**, 6714146304 / 8015314944 |

### reallocation-v1

| Pair/block | Order and measurements |
|---|---|
| block1 | block1-reduced: **1.822730100 s**, 6144784896 / 6771703808; block1-combined: **1.783560050 s**, 6941702656 / 7291797504; block1-native: **1.832058200 s**, 6547979776 / 7182745600 |
| block2 | block2-native: **1.833154750 s**, 6547979776 / 7182745600; block2-combined: **1.798893800 s**, 6941702656 / 7291797504; block2-reduced: **1.845281200 s**, 6144784896 / 6771703808 |
| block3 | block3-reduced: **1.843247450 s**, 6144784896 / 6771703808; block3-native: **1.832415000 s**, 6547979776 / 7182745600; block3-combined: **1.800531100 s**, 6941702656 / 7291797504 |
| block4 | block4-combined: **1.794795300 s**, 6941702656 / 7291797504; block4-native: **1.829815300 s**, 6547979776 / 7182745600; block4-reduced: **1.840911750 s**, 6144784896 / 6771703808 |
| block5 | block5-native: **1.830672050 s**, 6547979776 / 7182745600; block5-reduced: **1.846239300 s**, 6144784896 / 6771703808; block5-combined: **1.796752800 s**, 6941702656 / 7291797504 |
| block6 | block6-combined: **1.800420350 s**, 6941702656 / 7291797504; block6-reduced: **1.849272850 s**, 6144784896 / 6771703808; block6-native: **1.833940500 s**, 6547979776 / 7182745600 |

### scoped-v2-resume1

| Pair/block | Order and measurements |
|---|---|
| block1 | block1-legacy: **1.768721000 s**, 6941702656 / 7291797504; block1-scoped: **1.791086200 s**, 6941702656 / 7291797504; block1-native: **1.826377300 s**, 6547979776 / 7182745600 |
| block2 | block2-native: **1.832815500 s**, 6547979776 / 7182745600; block2-scoped: **1.793229200 s**, 6941702656 / 7291797504; block2-legacy: **1.799384850 s**, 6941702656 / 7291797504 |
| block3 | block3-legacy: **1.797931500 s**, 6941702656 / 7291797504; block3-native: **1.823886950 s**, 6547979776 / 7182745600; block3-scoped: **1.797662400 s**, 6941702656 / 7291797504 |
| block4 | block4-scoped: **1.795585900 s**, 6941702656 / 7291797504; block4-native: **1.833117350 s**, 6547979776 / 7182745600; block4-legacy: **1.791963800 s**, 6941702656 / 7291797504 |
| block5 | block5-native: **1.827580400 s**, 6547979776 / 7182745600; block5-legacy: **1.794595500 s**, 6941702656 / 7291797504; block5-scoped: **1.791970500 s**, 6941702656 / 7291797504 |
| block6 | block6-scoped: **1.792966100 s**, 6941702656 / 7291797504; block6-legacy: **1.794331850 s**, 6941702656 / 7291797504; block6-native: **1.840732250 s**, 6547979776 / 7182745600 |

## Preserved unsuccessful preparation and recovery

The first scoped implementation failed CPU preflight before any model gate or timing. PyTorch 2.14 resets the replay cursor when a SAC mode is re-entered; the corrected implementation gives each projection call its own unmodified mode pair and tests single entry. Its failed CPU log is retained.

The next attempt completed six gates, then refused candidate admission before torch import/model loading because selected-GPU residual utilization was 8%. Recovery waited for three consecutive readings under the same limits, reused those six gates byte-for-byte, and ran the remaining frozen roster. The refusal, previous ledger and file hashes are retained; no completed full-model gate or timing was repeated.

## Reproduce the audit

From a checkout with Python, NumPy and Rich:

```sh
python benchmarks/harness/issue2162_audit.py \
  --evidence benchmarks/results/issue2162-rtx3090-v1 \
  --verify-git-sources . --output audit-2162.json
python benchmarks/harness/issue2162_report.py \
  --evidence benchmarks/results/issue2162-rtx3090-v1 \
  --audit audit-2162.json --output report-2162.md
```

Source verification requires the original commits in Git history (fetch the PR branch without a shallow limit). Without `--verify-git-sources`, the data audit still runs but reports source verification as unavailable. It never runs a model. Published Git blob bindings replace redundant source ZIPs and preserve their original hashes and newline encodings.

Evidence: [manifest](results/issue2162-rtx3090-v1/manifest.json), [protocol guide](protocol-2162-auto-qkvo.md), [independent audit](results/issue2162-rtx3090-v1/independent-audit.json). Shared reference updates reconstruct all 84 original receipts byte-for-byte; timing rows and telemetry are retained without aggregation. The original independent audits and all earlier failed thresholds remain accessible.

Any further experiment needs a different, justified cell with a clearly larger predicted benefit and a public protocol before execution. Its correction family must include the previous 1.421695%, 1.860896%, 1.997942% and 2.034504% results, as requested by the maintainer. No further round is started by this record.
