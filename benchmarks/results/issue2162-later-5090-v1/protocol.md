# Census-derived retention on RTX 5090 Laptop

The next cell tests whether a rule based on the native census can produce a useful
retention layout on another GPU. The available card is an RTX 5090 Laptop with
24,463 MiB physical memory, capped at 8,589,934,592 bytes. This addresses the
different-card and rule-derived-layout questions; it does not supply physical
8 GiB evidence or resolve whether MLP retention belongs in Soup's public scope.
No production field or default changes are included.

This is a full separate repeat, directly requested by the operator on 27 October
2026 in the current Performance power scheme. Six completed Turbo gates and one
Turbo timing remain preserved and are not pooled or reused. The earlier thermal
failure and two Chrome interruptions remain failures with unknown partial updates.
All six gates and every planned timing start fresh, including a new two-update oracle.
The source, allocation, exactness, statistical gates and roster below are unchanged.
This operating amendment is local and has not been published or owner-approved.
No partial performance scores were interpreted to choose this run or its layout.

## Cell and reference

Keep the existing 41 audited model/shard files: Mistral-7B NF4/double-quant,
bf16, batch 1, sequence 2048, pinned host RAM, two stream buffers, q/k/v/o LoRA
rank 8, alpha 16, dropout 0, model seed 42, nonzero B seed 7, synthetic input
seed 1234, PagedAdamW8bit lr 2e-4. Deterministic algorithms, TF32 off,
CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL threads 1. Native override and allocator
both enforce the same 8 GiB cap. Use the runtime's default SDPA dispatch and
record its actual kernels; all arms must use the same backend. Do not force the
3090's efficient-attention kernel onto this architecture.

Environment: Windows 11 build 26300, NVIDIA driver 617.42, SM120, Python 3.13.14,
torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2 and
numpy 2.5.4. The lab used Python 3.13.7; this is a same-card paired comparison,
not a direct comparison of absolute step times between the two systems.

The first native-normal process measures its native census and records two
optimizer updates as the new-card oracle. Later processes must reproduce its
logits, loss, all 256 adapter gradients and all 256 updated tensors bit for bit.
Cross-architecture equality with the 3090 oracle is not assumed. Every fresh
process repeats the native census and must reproduce its K, whole-layer charge,
budget, reserved/allocated peaks and retained indices exactly. A drift stops
the study. Model files remain offline and are hashed before and after execution.

## Allocation rule: uniform-base-flops-v1

Validate uniform dimensions against every decoder projection. Let N be the
layer count, H the hidden width, I the MLP width, V the K/V output width, and
T=batch*sequence. Each bf16 element costs two bytes. Native census supplies
K_native, whole-layer charge C, budget B and reserved/allocated probe peaks.

QKVO storage per layer is T*(2H+2V)*2 bytes. Each gate/up output costs T*I*2
bytes. Charge each using the existing retained_layer_charge function and the
native probe peaks; call the resulting charges Qc and Mc. Reserve 64 MiB.

Enumerate every integer (K,Q,M) satisfying:

    0 <= K <= K_native; 0 <= Q <= N-K; 0 <= M <= Q
    K*C + Q*Qc + 2*M*Mc + 64 MiB <= B

Maximize the following avoided frozen-base matmul FLOP score:

    Qf = 2*T*(2*H*H + 2*H*V)
    Mf = 2*T*(2*H*I)
    score = K*(Qf+Mf) + Q*Qf + M*Mf

Resolve ties by more K, fewer charged bytes, more Q, then more M. Keep the last
K layers whole, save QKVO on the first Q remaining layers, and save gate/up on
the last M of those Q. These index rules are fixed now, not selected from new
timings. There is no GPU-name lookup, measured-time input or empirical weight.
If no feasible allocation with M>0 exists, stop without timing or a fallback.

The score omits attention, dequantization, dispatch and transfers. Earlier
profiles show checkpoint early-stop before the down-projection recompute
kernel, so down-projection earns no FLOP credit. A positive down-projection
recompute CUDA time in the new gates invalidates that premise and stops timing.
This is a testable allocation heuristic, not a claim of optimal step latency.

Arms: original native auto; the rule's mixed layout; a QKVO-only control with
exactly the mixed layout's K and Q, dropping only its MLP outputs. Do not
reoptimize that control. Keep layer-wide SAC, native mutation checks, checkpoint
and buffer lifetimes. Only frozen-base matmul outputs are retained; arithmetic
and LoRA remain unchanged.

## Gates and fixed process roster

Run six fresh processes in order: native-normal, qkvo-normal, mixed-normal,
native-profile, qkvo-profile, mixed-profile. Each performs two updates. The
first creates the oracle; the remaining ten updates replay it. Profile CPU and
CUDA with the validated thread-visible phase observer. Preserve raw events,
distinguish inner executed matmuls from interception, and require positive
CUDA attribution for executed QKVO/gate/up operations. Forward counts remain
unchanged. Cached or whole-retained projections must have zero recompute
matmuls; all other QKVO/gate/up projections have two across the two updates.
The mixed arm must remove exactly 4*M gate/up matmuls versus its QKVO control.

Saved tensor counts and bytes must match the rule. Every process must stay
within the cap for both allocated and reserved peaks. For mixed-vs-QKVO,
mixed-vs-native and QKVO-vs-native, the allocated-peak increase must not exceed
the additional raw saved bytes plus 64 MiB. Check both gate variants, every
timing block and every eligible confirmation pair: 30 comparisons if complete.

Stage A has six blocks containing all six permutations of native/qkvo/mixed,
shuffled once with random.Random(42): 18 fresh processes. Each has ten warmup
updates (the first two are exact oracle replays) and exactly 50 measured
updates. Synchronize CUDA and use perf_counter step durations. Preserve every
duration and loss; no profiler or tensor hashing enters the measured window.

For each block compute 100*(baseline_median-candidate_median)/baseline_median.
The estimand is the median of six block reductions. Report mixed_vs_native
(primary), mixed_vs_qkvo (incremental) and qkvo_vs_native. Use 10,000 whole-block
bootstrap resamples, NumPy default_rng(42), linear quantiles. Ordinary interval
is [.025,.975]; Bonferroni family 16 uses [.05/32,1-.05/32] (99.6875% coverage).
The family includes the previous twelve contrasts, these three and confirmation.
Recompute prior descriptive intervals at 16 while preserving original verdicts.
This count does not adjust for historical layout selection; with six blocks
the adjusted percentile bounds can collapse to the observed extrema.

Stage A requires mixed_vs_native median >=2%, with positive ordinary and
adjusted lower bounds, plus positive lower bounds for mixed_vs_qkvo and all
exactness, operator, memory and resource gates. No intermediate scores are
interpreted. Only after A passes, run six new native/mixed pairs, native first
on odd pairs and mixed first on even pairs: twelve fresh processes with the
same ten warmups and fifty measured updates. Confirmation uses seed 73 and
the same family 16, requiring median >=2% and both lower bounds positive.
It cannot rescue failed A. There is no additional pair, candidate or retuning.

If both stages run, totals are six gate processes, thirty timing processes,
two reference updates, seventy exact replay updates, 300 timing warmups and
1,500 measured updates: 1,812 optimizer updates overall. Otherwise A closure
has eighteen timings, 46 replay updates, 180 warmups, 900 measured updates and
1,092 total updates. All are disposable benchmark updates.

## Laptop controls and closure

Keep the current Performance power scheme and AC power. The programmatic power-limit
query reports N/A; record that limitation and sample power draw, SM/memory
clocks, temperature, utilization and memory. Do not claim a fixed wattage.
Require three consecutive readings five seconds apart before each process:
GPU utilization <=5%, temperature <=75 C, GPU free memory >=10 GiB, host free
RAM >=20 GiB, healthy driver and no Windows Runner.Worker. Admission expires
after thirty minutes. At the operator's explicit instruction, execution temperature
is managed by the stock driver/firmware; the former 82 C software stop is removed.
No thermal protection, clock, voltage, power limit or driver setting is changed.
Record SW/HW thermal and power-brake flags plus cumulative counters, clocks,
power and temperature. Keep all samples including throttled samples; do not
filter, retry, or select results by throttling. Enforce process-tree RSS <=16 GiB,
host free RAM >=16 GiB, unchanged driver/device/scheme,
AC power, fifteen minutes per worker and six hours for the model campaign.
Retain telemetry with no gap above five seconds.

This is a WDDM desktop GPU with Standard/MSHybrid graphics. Preserve the current
desktop instead of stopping applications. Freeze current GPU identities and the
PID, creation time and executable of Chrome browser roots before models. Permit
their Chrome descendants with matching executable paths, recording ancestor
identities in every GPU sample; reject other unknown GPU processes. This explicitly
replaces the old empty-PID-baseline rule for this new run only. Active browser work
can affect both CPU and GPU timings; balanced ordering cannot prove absence of bias.
Do not claim exclusive GPU use or a clean isolated hardware replication. No later
desktop whitelist or retrospective removal of slow/throttled samples is allowed.

On a failure terminate only the owned worker, preserve its partial evidence,
and stop. Do not repeat completed models or alter frozen code, gates or layout
to obtain a favorable result. Any justified helper recovery must preserve the
failed attempt and identify repeated updates before further execution.

At closure independently recompute raw statistics, exactness, operator and
memory gates, census/rule consistency, source/material hashes, update counts
and telemetry coverage. Preserve every process and ordering, report all three
contrasts and confirmation if eligible, and archive with SHA256/CRC and an
80 MiB expanded limit. Copy the verified evidence to the lab as well. A passing
cell supports this measured rule on this card; it is not production acceptance.
