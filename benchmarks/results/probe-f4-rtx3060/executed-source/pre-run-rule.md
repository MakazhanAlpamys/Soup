# Probe — the MoE small-GEMM factor (F4)

Status: NOT RUN. The rule below was committed 2026-10-09, before the first run
and before I had access to a GPU. There are no results yet.

Hardware: not assigned yet. The card will be stamped in §3 when the first run
happens.

## 2. The rule, written before the run

### 2.1 What I am measuring

A MoE expert gets only a few of the step's tokens, so its matrix multiply has
few rows. The weight matrix has to be read in full either way. The plan assumes
this costs 0.5–0.7 of dense throughput. Nobody measured it.

So I measure one ratio:

```
factor(M) = TFLOPS(M, K, N) / TFLOPS(512, K, N)
```

Only M changes. K and N stay the same, same card, same session. M = 512 is the
denominator because that is the number of tokens in one step. I also report the
ratio against M = 8192, to see whether M = 512 is itself already slow.

I also run three arms that do the same amount of arithmetic:

- A: one GEMM with M = 512, K = 2048, N = 16384.
- B: 64 separate GEMMs with M = 64, K = 2048, N = 2048, times added up.
- C: the same 64, but in a Python loop that picks out the rows, like the real
  code does.

All three are 34.36 GFLOP, which I checked by hand before the run. A to B shows
the cost of the small shape. B to C shows the cost of the loop.

### 2.2 Shapes

From each model's `config.json`:

| model | hidden | intermediate | experts | top-k | GEMM | K | N |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B-0924 | 2048 | 1024 | 64 | 8 | gate_up | 2048 | 2048 |
| OLMoE-1B-7B-0924 | | | | | down | 1024 | 2048 |
| granite-3.0-1b-a400m-base | 1024 | 512 | 32 | 8 | gate_up | 1024 | 1024 |
| granite-3.0-1b-a400m-base | | | | | down | 512 | 1024 |

`gate_up` is one fused matrix: transformers stores gate and up together and
splits the result with `.chunk(2, dim=-1)`. That is why N is
`2 × intermediate` and not `intermediate`.

### 2.3 Which M values

```
M = 1, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192
```

- 64 is OLMoE at a 512-token step: 512 × 8 / 64.
- 128 is granite at a 512-token step: 512 × 8 / 32.
- 256 is OLMoE at a 2048-token step, 512 is granite at 2048.
- 16 is a quiet expert, 1 is a sanity check, 8192 is the large-M limit.

The plan says "~200 tokens per expert". That is OLMoE at a 2048-token step. At
512 tokens the same model gives 64, so the step size matters and I state it.

### 2.4 How one point is measured

Three warm-up calls, then 20 timed iterations, three repeats, keep the best.
This is the same protocol as `stream_probe.py`, so the numbers can sit next to
the project's other records.

Time is taken with CUDA events, not `time.time()`. GPU work runs in the
background, so a normal timer would measure how fast Python sent the commands
instead of how long the GPU took.

dtype is bfloat16 if the card supports it. If it does not, I use float16 and say
so, and that run is not comparable with the bf16 records.

### 2.5 Order of the runs

I do not sweep M from small to large. The card heats up and slows down, so the
later points would look worse for the wrong reason.

Order is 512 → M → M → 512 for each M. Both arms then sit in the middle of the
run on average, so the heating affects them equally and cancels in the ratio.

Inside a pair I average the two readings. The "keep the best" above applies only
to the three repeats of one point.

Three rounds, each one a separate run of the script.

### 2.6 Sanity check

`factor(1)` must be below 0.25. With one row the GPU is doing almost no work per
byte it reads, so it has to be far from full speed. If the script reports
something near 1.0 there, the script is wrong and none of its other numbers mean
anything.

The script also checks that the result of `a @ b` has shape `(M, N)`, so if I
mix up K and N it fails with a clear message instead of measuring the wrong
thing.

### 2.7 When an arm does not count

An arm is void, kept in the record under a `_void` name, and re-run once if:

- V1: the laptop was on battery at any point;
- V2: another process was using the GPU;
- V3: the SM clock moved more than 10% from the round's median;
- V4: the machine went to sleep;
- V5: the arm ended without writing its result.

A second void in the same place ends the probe with no verdict.

The 10% in V3 is a guess. The proper way is to measure how much the card's clock
varies on its own first, and I have no card yet. If V3 fires every time, the
answer is "no verdict" plus this note — I will not move the threshold after
seeing the numbers.

### 2.8 Verdict

Read `factor` at each model's real M: 64 for OLMoE, 128 for granite, both at a
512-token step. Different rows for the two models.

| measured | what it means |
|---|---|
| 0.70 or less | the plan's assumption holds; the curve replaces the 0.5–0.7 guess |
| 0.70 to 0.90 | the plan is too pessimistic; the giant table's rows go up |
| 0.90 or more | the small GEMM is not the MoE bottleneck; look at the loop and at NF4 instead |
| any of V1–V5 fired | no verdict; numbers published as description only |

Before running, I expect granite to come out closer to 1 than OLMoE, because
granite gives 128 tokens per expert where OLMoE gives 64. If it comes out the
other way, I do not understand something and that is worth more than a
confirmed guess.

### 2.9 What this does not measure

This is a pure matrix multiply, which is only one part of a real step. Not
measured: NF4 dequantisation, the router, the backward pass, a full training
step, attention, and models with more than 64 experts.

Layer streaming is not used here. `SUPPORTED_STREAM_ARCHS` in
`src/soup_cli/utils/layer_stream.py` does not list `olmoe` or `granitemoe`, so
these two models are measured resident.

Numbers from different cards are never compared. This record applies only to the
card stamped in §3.
