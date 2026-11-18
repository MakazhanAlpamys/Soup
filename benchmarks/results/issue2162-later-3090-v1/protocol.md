# Fresh same-rule qualification on the ASUS RTX 3090

The 27 October GPU1 campaign ended after six gates and ten timings because CI
started during block4-native. Its three attempted segments, unknown partial
updates, negative/incomplete verdict and ZIPs remain intact. No receipt or timing
from that campaign is reused. This is a fresh run of the same frozen
uniform-base-flops-v1 layout on the same physical GPU1 after a controlled Windows
reboot. The prior same-card campaign's four contrasts consume reserved family
slots. With fifteen observed older contrasts and five reserved slots, this
study uses family 24. No candidate was selected from the interrupted timings.

The GPU0 375 W limit is applied by an existing scheduled task; it remains
unchanged. GPU1 is still at 350 W, GPU2/3 at 420 W. Current service/container
identities are frozen before models. The two inference services are queried
through WSL because Windows localhost forwarding is not available; their slot
task counters remain fixed throughout all benchmark processes. Idle CI runner
tasks are paused for this controlled window and restored at closure or by a
separate deadline fallback. No user workload or active job is terminated.

The new cell evaluates the existing `uniform-base-flops-v1` rule on the lab's
ASUS RTX 3090, GPU 1, at its unchanged 350 W limit. Previous 3090 studies used
GPU 0 at 420 W. This is another physical card on the same host and architecture,
with 24 GiB capped at 8 GiB. We have no physical 8 GiB GPU.

This study keeps the allocation rule evaluated on the laptop. Its Performance
run completed eighteen timings but failed: mixed/native median reduction
8.166451%, ordinary 95% interval [-5.595217%, 27.472056%], family-16 interval
[-15.799741%, 37.033011%]. The incremental contrast also crossed zero; confirmation
was ineligible. Earlier laptop attempts were interrupted, and Performance had
pauses, different admission regimes and thermal throttling. All records and
verdicts remain separate. We are changing hardware conditions, not selecting
a new layout from those times.

The prior manually allocated 3090 result remains 3.625068% in Stage A and
3.422014% in confirmation. That result does not establish the automatic rule's
performance. MLP retention remains outside the issue's original scope pending
the owner's review and restatement. This experiment changes no production option,
schema or default, and success would not by itself establish production acceptance.

## Fixed cell and oracle

Use the same 41 hashed model/shard files, Mistral-7B NF4/double-quant, BF16,
batch 1, sequence 2048, pinned host RAM and two stream buffers. q/k/v/o LoRA:
rank 8, alpha 16, dropout 0, model seed 42, nonzero B seed 7, synthetic input
seed 1234, PagedAdamW8bit lr 2e-4. Deterministic algorithms, TF32 disabled,
CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL threads 1. Both allocator fraction and
native override impose 8,589,934,592 bytes. Keep default SDPA dispatch and
record the executed attention kernels; all arms must use the same backend.

Windows 11 build 26200, driver 610.88, Python 3.13.7, torch 2.14.1+cu130,
transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, numpy 2.5.4. Record the
remaining package versions and validate them before every process. Target UUID:
`<TARGET_GPU_UUID>`. No driver, clock, fan or power changes.

The first native-normal process creates a fresh two-update oracle on this card.
Every later process must reproduce logits, loss, all 256 adapter gradients and
all 256 updated tensors bit for bit. Every process repeats the native census
and must reproduce K, whole-layer charge, budget, allocated/reserved probe peaks
and retained indices exactly. No old GPU receipt or timing is reused.

## Allocation and controls

The rule and model execution are unchanged from the published laptop protocol.
Validate uniform dimensions across every decoder projection. Let N be layers,
H hidden width, I MLP width, V K/V output width, and T=batch*sequence. Native
census supplies K_native, whole-layer charge C, budget B and probe peaks.
QKVO raw storage is T*(2H+2V)*2 bytes per layer. Each gate/up output uses T*I*2
bytes. Apply the native retained_layer_charge function to each using the probe
peaks, giving Qc and Mc. Reserve 64 MiB and enumerate integers satisfying:

    0 <= K <= K_native; 0 <= Q <= N-K; 0 <= M <= Q
    K*C + Q*Qc + 2*M*Mc + 64 MiB <= B

Maximize K*(Qf+Mf)+Q*Qf+M*Mf, where Qf=2*T*(2*H*H+2*H*V) and
Mf=2*T*(2*H*I). Ties choose more K, fewer charged bytes, more Q, then more M.
Keep the last K layers whole, cache QKVO on the first Q remaining layers,
and gate/up on the last M of those Q. No timing, GPU-name lookup or empirical
weight enters this rule. If there is no feasible allocation with M>0, stop.

Arms are native auto, the rule's mixed layout, and a QKVO-only control using
exactly mixed's K and Q while dropping only gate/up retention. Keep layer-wide
selective checkpointing, mutation checks, arithmetic and buffer ownership.
The FLOP score omits attention, dequantization, dispatch and transfers. It gives
down-projection no credit because previous profiles found checkpoint early-stop
before that recompute kernel. Positive down recompute CUDA time invalidates
this premise and stops the experiment before timing.

## Gates and measurements

Six fresh processes, in this order: native-normal, qkvo-normal, mixed-normal,
native-profile, qkvo-profile, mixed-profile. Each makes two updates. The first
creates the oracle; the next ten updates replay it. Keep raw profiler events
and thread-visible phase attribution. Forward counts remain unchanged. Cached
or whole-retained projections have no backward recompute matmul; other QKVO,
gate and up projections have two over the two updates, with positive CUDA
attribution. Mixed must remove exactly 4*M gate/up matmuls relative to QKVO.

Saved tensor counts and bytes must equal the rule. Allocated and reserved peaks
must stay below 8 GiB. For mixed/QKVO, mixed/native and QKVO/native, allocated
peak growth must not exceed additional raw cache bytes plus 64 MiB. Check both
gate variants, every timing block and every eligible confirmation pair.

Stage A: six blocks, all six permutations of native/QKVO/mixed shuffled with
random.Random(42), eighteen fresh processes. Each has ten warmup updates,
including two exact oracle replays, followed by fifty measured updates.
Synchronize CUDA and measure with perf_counter. Preserve every duration and
loss. No tensor hashing or profiler enters the measured window.

For each block compute 100*(baseline_median-candidate_median)/baseline_median.
The estimand is the median of the six block reductions. Report mixed/native
(primary), mixed/QKVO (incremental), and QKVO/native. Bootstrap 10,000 whole
blocks with NumPy default_rng(42), linear quantiles. Ordinary interval is
[.025,.975]; Bonferroni family 24 uses [.0010416667,.9989583333], or about
99.7917% coverage.
The family retains twenty prior slots, including five reserved but unrun
contrasts, and adds these three contrasts plus confirmation. Recompute prior descriptive
intervals at 24 without revising any old verdict. The family does not correct
historical layout selection. With six observations adjusted bounds can equal
the observed extrema; neither six blocks nor a favorable outcome is guaranteed
to produce a narrow interval.

Stage A passes only if mixed/native median >=2%, its ordinary and adjusted lower
bounds are positive, both mixed/QKVO lower bounds are positive, and every gate
passes. Only then run six fresh native/mixed confirmation pairs, AB on odd pairs,
BA on even pairs, with the same ten warmups and fifty measurements. Confirmation
uses bootstrap seed 73 and family 24; median >=2% and both lower bounds must
be positive. It cannot rescue failed Stage A. No additional pair or layout.

Totals: six gates plus eighteen timings, 2 reference updates, 46 exact replay
updates, 180 warmups, 900 measured updates, 1,092 updates overall. If confirmation
is eligible: thirty timings, 70 exact replays, 300 warmups, 1,500 measurements,
1,812 updates overall. These are disposable benchmark updates, not a model fit.

## Lab controls and closure

Services remain resident. GPU 1 has about 14 GiB free; GPU 0/2/3 remain at their
frozen 375/420/420 W, GPU 1 at 350 W. Before each process require three readings
at least five seconds apart: target utilization <=5%, target free memory >=10 GiB,
other GPUs <=5%, all temperatures <=75 C, host free RAM >=32 GiB, and no Windows
or AuthGuard-CI Linux Runner.Worker. Wait at most thirty minutes. Pause idle CI
runner tasks reversibly before the roster and restore them at closure.

The existing WDDM GPU process identities are recorded during preparation.
Only those same PID/creation-time identities plus owned workers are allowed.
Windows exposes WSL compute as PID 4, so this alone cannot prove GPU exclusivity.
Before any model, after two health-check inferences, record both
inference servers' idle slot IDs and task counters. Require them unchanged across
all benchmark processes, detecting even completed requests between samples.
Other GPU activity is also checked throughout. No service or desktop app is stopped.

Sample all four GPUs' utilization, memory, temperature, power, clocks and thermal/
power flags throughout, with gaps <=5 seconds. Keep every sample, including
throttling. No counter duration is inferred. Enforce unchanged identities, driver,
power scheme and power limits, temperatures <=75 C, owned process-tree RSS <=16 GiB,
host free RAM >=32 GiB, fifteen minutes per worker and six hours from campaign
preflight. Inspect thermal and first-ten/last-ten timing drift descriptively at closure;
never discard observations on that basis. The user-facing laptop cooling changes
are not carried into this new lab study.

Freeze source, helpers, protocol, roster and criteria locally before any model.
The user explicitly instructed immediate execution without a new public post or
waiting period. The owner's request for prior publication and a reading window
is therefore not followed by this local experiment. Disclose this in any later
submission, including a positive one. Scientific gates and the roster are unchanged.
CPU tests do not allocate GPU models. On failure,
terminate only our worker tree and preserve the attempt; no automatic retry.

Rehash all 41 materials and scientific sources after the roster. Independently
audit exactness, census/rule, operators, memory, statistics, process order, update
counts, admission and telemetry. Check four WSL and four Windows services, the
exact current container set, and two real inferences before and after. Do not
start containers for this study. Preserve the full evidence, report, audit,
hashes and history on the lab,
checking ZIP SHA256, CRC, path containment and an 80 MiB expanded limit.

A passing result supports this rule on this measured card. Same architecture,
shared host, resident services, WDDM, a synthetic cell and a cap on a 24 GiB card
limit generalization. The owner still decides scope and production integration.
