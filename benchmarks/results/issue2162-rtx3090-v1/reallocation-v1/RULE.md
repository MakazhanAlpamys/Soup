# #2162 one-layer reallocation follow-up

This is a new experiment prompted by the user's request to pursue the 2% goal.
The earlier native-K result remains 1.421695% at sequence 2048 and did not pass
its rule. None of its sources, observations, intervals or decisions is changed.

The hypothesis is that releasing exactly one fully retained layer can pay for
enough additional QKVO outputs to improve the incremental gain over native auto.
The tradeoff includes the recomputation of the released layer; it is not free.

Scope is sequence 2048 only, Mistral-7B NF4, bf16, pinned RAM, batch 1, two weight
buffers, the same model/shards and rank-8 q/k/v/o adapters, and the same 8 GiB
allocator cap and native byte override. GPU0 keeps its existing 420 W limit.
No service is stopped. The selected SDPA backend is preserved; no math fallback.

Three arms are fixed before execution:

- native: the original native auto K=11, no QKVO.
- reduced: K=10, no QKVO, to expose the cost of releasing one layer.
- combined: K=10 plus the residual-budget QKVO selection (expected 19 layers).

The native auto probe must still reproduce the original K and budget-derived
selection. The one-layer release is then applied explicitly and recorded as a
separate allocation, preserving the original native measurement. Frozen expected
plans are checked against actual model widths and applied layer flags.

Before timing, each arm has a normal, repeat and CPU-profiler process, two
optimizer updates each. All nine receipts must match every logit, loss, all 256
adapter gradients and all 256 updated tensors from the preserved native reference
exactly. Cache storage/counts, active retention, selected attention operations and
combined peak memory must pass. No approximate replay, changed K or smaller cache
is accepted after a failure.

Timing uses six fresh-process blocks, all six permutations of three arms, in a
seed-42 shuffled order fixed in freeze.json. This balances each arm's position
twice and each pair's relative order three times. There are no optional extra
blocks. Every arm uses 10 warmup and 50 measured updates; its first two warmups
must replay the original exactness reference before proceeding.

Three contrasts are reported: combined versus native (primary), combined versus
reduced, and reduced versus native. Each block's reduction is 100 times the
difference of the arms' median step times divided by the comparator's median.
The estimate is the median of six block reductions. Bootstrap resamples entire
blocks 10,000 times, NumPy default_rng seed42, linear quantiles. Ordinary intervals
use .025/.975; the three-contrast Bonferroni family uses .05/6 and 1-.05/6.
Only the primary determines advancement: median reduction >=2% with both lower
bounds >0, plus all correctness, memory and resource gates. Diagnostics cannot
replace the primary. This is a follow-up selected after the earlier exploratory
result, not a reinterpretation of that result or a claim of project acceptance.

All four GPUs, CPU load, memory and CI activity are sampled. Timing refuses other
GPU utilization above 5%, active CI workers, GPU temperatures above 75 C, process
RSS above 16 GiB, available host RAM below 32 GiB, or a driver error. Each worker
has a 15-minute limit. Only this campaign's worker can be terminated by its guard.
No score is interpreted until the complete roster is available and audited.

The controller stops on failure; it never repeats an arm, changes the allocation
or starts another candidate. Sources and all attempts remain preserved. Closure
revalidates model/shard hashes, copies only a checksum-verified evidence archive,
audits all receipts independently, and checks the existing lab services. No
public configuration, production option, commit push or external message is part
of this experiment.
