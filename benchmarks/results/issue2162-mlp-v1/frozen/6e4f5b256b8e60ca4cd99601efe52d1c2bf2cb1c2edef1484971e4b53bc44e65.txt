# MLP retention diagnostic for #2162

The next question is whether saving MLP gate/up outputs removes enough additional
recomputation to justify a new timing experiment. This extends the QKVO-only
scope. The record in #2218 and all earlier failed rules remain unchanged.

This protocol runs a fixed mechanistic diagnostic, with zero timing pairs and
no end-to-end speed verdict. There is one candidate, A. Candidate B is not run.

The cell stays Mistral-7B NF4/bf16, batch 1, sequence 2048, pinned RAM, two stream
buffers, rank-8/alpha-16 q/k/v/o LoRA, nonzero B seed 7, model seed 42 and input
seed 1234. It uses PagedAdamW8bit at 2e-4, deterministic algorithms, TF32 off,
and the actually selected SDPA efficient-attention backend. The existing model
and shards are read offline. Both native override and allocator enforce
8,589,934,592 bytes on the same RTX 3090, at its original 420 W.

The original native census must reproduce K11, charge 482,015,738 bytes per
whole layer, budget 5,756,682,240 bytes and the original reserved/allocated
probe peaks. A changed census stops this protocol rather than selecting a
different layout.

| Arm | Whole retained layers | Selective outputs |
|---|---|---|
| native | 21..31, K11 | none |
| qkvo | 24..31, K8 | Q/K/V/O in layers 0..23 |
| mixed | 24..31, K8 | same QKVO, plus gate/up in layers 19..23 |

The qkvo cache is 1,006,632,960 raw bytes. Mixed is 1,593,835,520 raw bytes,
charged 1,829,297,278 bytes, leaving 71,259,058 bytes in the selective budget.
This is a prediction from the existing census; actual saved storage and peaks
must be checked. The mixed arm adds exactly ten outputs, 587,202,560 raw bytes.
Keep the simpler layer-wide SAC, frozen base-linear matmul outputs only,
unchanged checkpoint/buffer lifetimes and the native cache mutation checks.

After local and remote CPU tests, freeze source bytes, archive SHA256, protocol,
model/shard hashes and the complete process roster. Before model loading, one
tiny CUDA-profiler capability probe must show device time for a matrix multiply.
If it fails, stop without starting full-model processes.

Six fresh model processes run once in this order: native-normal, qkvo-normal,
mixed-normal, native-profile, qkvo-profile, mixed-profile. Each performs exactly
two optimizer updates, discarded after validation. All twelve updates must
reproduce the original native logits, loss, all 256 adapter gradients and all
256 updated tensors bit for bit. Normal and profiled runs therefore also test
whether instrumentation changed the arithmetic. No extra warmups, fitting,
downloads, completed-model reruns or throughput measurements are added.

Profile CPU and CUDA with scopes around frozen base projections and explicit
forward/backward phases. Count innermost redispatched matmuls: an outer
PythonDispatchMode event on a cache hit is not executed compute. Attribute only
ops inside projection-forward scopes, excluding the separate gradient matmuls.
Record per-layer/op counts and device time, cache tensors, actual attention
dispatch and allocated/reserved peaks. Missing CUDA attribution stops the test.

The mechanism must remove exactly 20 gate/up recomputation matmuls across the
two profiled mixed updates relative to qkvo (ten per update), with identical
forward counts and unchanged nonselected MLP recomputation. All QKVO base
recomputation is absent in both K8 arms. Raw saved bytes/counts must match the
roster. Every allocated/reserved peak must be at most 8 GiB; the mixed-minus-qkvo
allocated increase must be at most the extra saved bytes plus 64 MiB.

The diagnostic only supports preparing a timing protocol if correctness,
memory, resource and operator gates pass and the following planning estimate
is at least 3%:

    100 * (24 * 0.0024343118421280736 + M - 3 * 0.011611449999691104)
        / 1.83223659999976

Here M is the qkvo profile's summed CUDA time of the selected gate/up
recomputation divided by two updates, in seconds; the mixed arm must remove
those operations. The other constants come from all six earlier reallocation
blocks. This estimate assumes kernel-time savings translate to step savings
and linearly extrapolates the old whole-layer penalty. It is neither a new
measured speedup nor a confidence interval. Report the estimate even if it
fails. A failed diagnostic does not trigger candidate B.

A later throughput study needs its own public frozen protocol, fixed pair
count and independent confirmation before execution. Its correction family
must include 1.421695%, 1.860896%, 1.997942% and 2.034504%, plus the new contrasts.
This diagnostic cannot pass the 2% acceptance rule or authorize an opt-in field.

Require three consecutive idle admission readings before each worker: all
four GPUs healthy, other GPUs at most 5% busy, no Windows/Linux CI worker,
selected GPU at most 5% busy and 60 C, at least 10 GiB free on it. During each
worker enforce temperatures at most 75 C, process-tree RSS at most 16 GiB,
free host RAM at least 32 GiB, unchanged power limits, no competing CI/GPU
activity and a 15-minute limit. Stop only the owned worker on a guard failure;
preserve the attempt without repeating completed models. Services stay running.

Closure checks four services, two real inference requests, three existing
healthy containers and original power limits; rehashes the 41 model/shard
files and frozen sources; transfers only evidence ZIP with SHA256/CRC/path
containment and an 80 MiB expanded limit; independently audits all six receipts.
Failures stay in the record. Nothing changes the #674 datasets or evidence.
