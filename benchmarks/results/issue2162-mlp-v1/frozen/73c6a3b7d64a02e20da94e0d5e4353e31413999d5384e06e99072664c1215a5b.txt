# Fixed MLP budget B experiment for #2162

The candidate A diagnostic confirmed exactness and removed the 20 intended
gate/up matmuls over two profiled updates. It measured 30.8545415 ms of avoided
CUDA work per update. Substituting that into the previous layer-cost model gave
2.9714326%, below A's fixed 3% planning screen. That outcome is preserved.

This separate experiment tests the previously described allocation B. It spends
more of the same memory budget on gate/up outputs: six whole layers, QKVO on
26 other layers, and gate/up on eleven of them. Using the measured gate/up cost
and old layer penalties predicts 3.9904699%. That is a planning estimate under
linear assumptions, not an observed speedup. The user explicitly authorized
continuing to this next allocation; A is not rerun or reclassified.

The cell remains the existing Mistral-7B NF4/double-quant model and shards,
bf16, batch 1, sequence 2048, pinned RAM, two buffers, q/k/v/o LoRA rank 8,
alpha 16, dropout 0, model seed 42, nonzero B seed 7, synthetic input seed 1234,
PagedAdamW8bit lr 2e-4, deterministic algorithms and TF32 off. Use the actual
selected SDPA efficient-attention backend. OMP/MKL threads are 1. The RTX 3090
stays at 420 W. Native override and allocator both enforce 8,589,934,592 bytes.
This is a capped 24 GiB card, not evidence from a physical 8 GiB card.

The original native census must reproduce K11, 482,015,738 bytes charged per
whole layer, budget 5,756,682,240 bytes and the original probe peaks. Refuse a
changed census rather than adapt the candidate. Layer-wide SAC and the native
mutation checks stay in place. Save only frozen base matmul outputs; LoRA,
dequantization and other operations keep their existing arithmetic.

| Arm | Whole retained layers | Selective outputs |
|---|---|---|
| native | 21..31, K11 | none |
| qkvo | 26..31, K6 | Q/K/V/O in layers 0..25 |
| mixed, primary candidate | 26..31, K6 | same QKVO plus gate/up in layers 15..25 |

QKVO costs 1,090,519,040 raw bytes and 104 tensors. Mixed costs 2,382,364,672
raw bytes and 126 tensors, charged 2,734,318,038 bytes within the residual
2,864,587,812 bytes. Charged slack is 130,269,774 bytes. The additional 22 MLP
outputs cost 1,291,845,632 raw bytes. No roster change follows observed scores.

First run six fresh exactness processes in fixed order: native-normal,
qkvo-normal, mixed-normal, native-profile, qkvo-profile, mixed-profile. Each
performs two updates and must reproduce the original native logits, loss,
all 256 adapter gradients and updated tensors bit for bit. Profiling uses the
thread-visible phase observer validated in A's recovery and saves raw operator
events before parsing. Profile CPU/CUDA; distinguish actual inner matmuls from
SAC interception and down-projection early-stop boundaries. Require the actual
efficient-attention operations and removal of 44 selected gate/up matmuls over
two mixed updates, with unchanged forward/nonselected counts. Missing QKVO or
gate/up CUDA attribution refuses timing.

Every process must match predicted saved storage and tensor counts. Allocated
and reserved peaks must both stay at most 8 GiB. The mixed-minus-qkvo allocated
increase must be at most the extra MLP bytes plus 64 MiB. Also check each cache
arm against native using its total saved bytes plus 64 MiB. Apply these memory
comparisons in the gates and every complete timing block/pair.

Stage A fixes six balanced blocks, one of each permutation of native/qkvo/mixed,
shuffled once with Python random.Random(42): 18 fresh processes. Each has ten
warmup updates, including two exact replays of the old reference, then exactly
50 measured updates. Use synchronized perf_counter step durations and retain
all raw durations, losses, ordering, telemetry and memory peaks. No profiler
or tensor hashing runs inside the measured window.

For each block compute 100*(baseline median time-candidate median time)/baseline
median time. Report the median of the six paired reductions for mixed_vs_native
(primary), mixed_vs_qkvo (incremental mechanism) and qkvo_vs_native (control).
Use 10,000 whole-block bootstrap resamples, NumPy default_rng(42), linear
quantiles. Ordinary interval: [.025,.975]. The Bonferroni family is fixed at
12: all eight published contrasts from timing-v1, reallocation-v1 and
scoped-v2-resume1, plus these three and the confirmation contrast. The adjusted
interval uses [.05/24,1-.05/24], or 99.5833333% coverage. This includes the
previous 1.421695%, 1.860896%, 1.997942% and 2.034504% explicitly. Recompute their
descriptive intervals with this family, while retaining every original verdict.
The diagnostic A estimate is not a throughput comparison or additional test.

Stage A passes only if mixed_vs_native has median >=2% with positive ordinary
and adjusted lower bounds, and mixed_vs_qkvo has both lower bounds positive,
and all exactness/operator/memory/resource gates pass. No minimum percentage is
imposed on the incremental diagnostic contrast. No partial timing interpretation.

Only then run stage B: six fresh native/mixed pairs, alternating AB/BA (native
first on odd pairs), 12 processes with the same 10 warmups and 50 measurements.
The candidate, code, layer roster and recipe remain unchanged. Use the median
paired reduction, 10,000 whole-pair bootstrap resamples, seed 73 and the same
ordinary and family-12 adjusted intervals. Confirmation requires median >=2%
and positive lower bounds in both intervals, plus every operational gate.
It cannot rescue a failed stage A. No extra pairs or candidate selection.

Freeze protocol, all source bytes, controllers, model/shard hashes, orders and
decision code after CPU tests and before full-model execution. Publish this
protocol first. Successful complete processes are not repeated. Stop and retain
any failure. Any instrumentation recovery must preserve its failed attempt and
identify all repeated updates; it cannot change math, criteria or the candidate.

Require three consecutive idle admission readings before each worker, with
healthy inventory, no Windows/Linux CI worker, other GPUs at most 5% busy,
selected GPU at most 5% busy and 60 C and at least 10 GiB free. During workers:
all GPUs at most 75 C, owned process-tree RSS at most 16 GiB, host free RAM at
least 32 GiB, unchanged power, no competing GPU/CI work, 15-minute worker limit.
Only terminate the owned worker on a failure. Services, permissions, credentials,
host/WSL/Docker and #674 data remain unchanged. No downloads or expenses.

After the fixed eligible roster, check four services, two real inferences,
three healthy containers and original powers. Rehash the 41 model/shard files
and frozen sources. Transfer only the evidence ZIP with SHA256, CRC, unique
entries, path containment and an 80 MiB expanded limit. Independently recompute
all statistics and gates from raw rows, then preserve report and audit locally
and in the lab. Both stages passing supports this measured configuration only;
it does not establish production acceptance or a public option.
