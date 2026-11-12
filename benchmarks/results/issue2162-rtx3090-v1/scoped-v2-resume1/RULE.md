# #2162 projection-scoped dispatch v2 and conditional confirmation

The user authorized continued engineering on 26 October 2026 to reach the 2%
step-time reduction goal. The earlier residual-budget (1.421695%) and one-layer
reallocation (1.860896%) experiments remain closed with failed numerical rules.
This is a changed implementation, not a rerun to change either conclusion.

The candidate narrows selective-checkpoint dispatch to the frozen Q/K/V/O base
forward calls. The layer checkpoint, stream lifetime, saved outputs, mutation
checks, mathematical operations and cache allocation remain unchanged. Attention,
MLP and LoRA operations no longer enter the selective-cache dispatch mode.

Freeze source, recipe, protocol, allocation and both stages before execution.
Use the same existing Mistral-7B NF4 checkpoint/shards, synthetic input seed1234,
LoRA rank8/alpha16/qkvo, nonzero B seed7, model seed42, batch1, sequence2048,
bf16, pinned RAM, two buffers, PagedAdamW8bit lr2e-4, deterministic algorithms,
TF32 disabled and selected SDPA efficient attention. No new downloads or data.
GPU0 remains at420W; both the allocator and native override are8589934592 bytes.
The original native auto probe must reproduce K11 and its byte charges.

Stage A fixes three arms:

- native: K11, no selective cache.
- legacy: K10 plus nineteen QKVO layers, layer-wide SAC, as in reallocation-v1.
- scoped: K10 plus the same nineteen QKVO layers, projection-scoped SAC.

The K10 arms retain layers22..31 and cache layers3..21. Cache storage is
796917760 bytes, charged914648638, within residual936524860 bytes. Neither the
selection nor the cache scope may adapt after observing scores.

Before timing, run normal/repeat/profile fresh processes for each arm: nine
processes, two optimizer updates each. Every logit/loss, all256 gradients and
all256 updated tensors must match the retained native seq2048 reference exactly.
Require unchanged actual cache bytes/counts, complete parameter roster, applied
K, native override binding and selected efficient-attention operations. Reject
approximate replay, backend changes, over-cap allocated/reserved peaks or errors.

Stage A timing uses all six permutations of three arms, shuffled once by Python
random.Random(42), six whole blocks/eighteen fresh processes. Each uses10 warmup
updates and50 measured updates; the first two warmups replay the old native
reference exactly. Report scoped_vs_native(primary), scoped_vs_legacy and
legacy_vs_native. Each block reduction is100*(comparator median step time minus
candidate median step time)/comparator median step time. Estimate the median of
six block reductions. Use10000 whole-block NumPy default_rng(42) bootstrap
resamples, linear quantiles, ordinary .025/.975 and three-contrast corrected
.05/6,1-.05/6. The primary must have median>=2% and both lower bounds>0. In
addition, scoped_vs_legacy must have positive ordinary and corrected lower
bounds to attribute an improvement to the changed implementation. No2% minimum
is imposed on that diagnostic contrast. Only then is stage B eligible.

Stage B is a predeclared fresh confirmation of exactly the same scoped candidate
against the original native K11. Six pairs, alternating native/scoped then
scoped/native, three of each order; twelve fresh processes. Each has the same
10warmup+50measured updates and first-two-update exact replay. No further tuning,
candidate choice, extra pair or changed source is permitted. Apply the median
of six paired reductions,10000 whole-pair resamples with NumPy default_rng(73),
linear .025/.975 quantiles. Confirmation requires median>=2% and the lower95%
bound>0, as well as every correctness/memory/resource gate. There is one fixed
confirmation contrast. Stage B cannot rescue a failed stage A and is not run
when stage A fails. A pass in both stages supports this measured configuration,
not general GPU performance or production acceptance.

All four GPUs, host RAM/RSS/CPU and CI worker activity are sampled. Every worker
must stay below75C,16GiB RSS, with at least32GiB host RAM available and healthy
driver inventory; selected GPU power remains420W. Timing additionally refuses
other-GPU utilization above5% or active Windows/Linux CI workers. Per-worker
limit15minutes; terminate only this experiment's worker on failure. No service,
power, permissions, credential, host/WSL/Docker restart or expense is authorized
by this protocol. The existing service workloads remain in place.

Finish every fixed stage before interpreting its results. On failure preserve
the attempt; never retry models silently. The controller does not add candidates.
Closure checks existing services and real inference, rehashes models/shards and
all frozen sources, packages evidence, checks ZIP SHA256/CRC/path containment and
independently audits the complete executed roster. Report both successes and
failures; preserve local/lab copies. No public messages, push or production
option are part of this experiment. Subsequent engineering needs a separate
predeclared protocol; the user's authorization to continue does not change old
criteria or justify repeated testing of an unchanged implementation.

Preparation history: scoped-v1 failed CPU preflight on the lab before freeze or
any full-model process. The PyTorch2.14 replay mode resets its operation cursor
on entry. Version2 gives each projection call its own unmodified native SAC
mode pair, entered once. A regression test enforces single entry on older
PyTorch too. Both preparations and logs remain preserved; the statistics,
rosters, recipes and acceptance thresholds above are unchanged.
