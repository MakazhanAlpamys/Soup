<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any model was downloaded, which is the whole point
of it: a rule written after the numbers are in is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe, Windows 11 Pro 26100. Stack: Python
3.12.10, torch 2.14.0+cu130, transformers 5.17.0, bitsandbytes 0.50.2.
Soup: branch probe/moe-expert-coverage cut from origin/main a08ae74f, worktree
C:\Users\user\projects\Soup-moe.
Harness: benchmarks/harness/moe_expert_coverage.py (committed with this file).
-->

# Probe — does a training step touch every expert? (MoE step 0)

**Status: measured 2026-09-15. A training step touches essentially every
expert of every layer — C = 0.967 to 1.000 across two models, three corpora and
three batch shapes — so expert-granularity streaming saves memory and NOT reads,
and the rule committed here before the run says to ship it, if at all, as a
capacity tier with no throughput promise.**

**The rule's provenance, stated precisely because a reviewer checked it and the
first version of this paragraph was not.** §2 was first committed in
**4908d52b** (16:44) and **amended in 8d0ff678** (18:48), which rewrote "that
predictability is NOT measured in step 0" into "is measured in step 0 after all"
and added the two prefetch predictors and the fourth rule row. The first model
was downloaded at **19:53**, so both commits predate every number here and the
pre-registration survives - but the record has to say the rule was edited rather
than let "committed in 4908d52b" imply it never was. Nothing in the rule has been
touched since the numbers arrived, which is the property that matters and which
`git log` can check.

---

## 0. The question, and why it decides a feature

Layer streaming today bounds VRAM by ONE decoder layer times `stream_buffers`.
On a 744B-class MoE one NF4 layer is ~5 GB, so two buffers do not fit an 8 GB
card and the model is refused at the pre-flight. Streaming an EXPERT at a time
instead would bound it by "the dense part plus a few experts", which on paper
admits any size.

Whether that is worth building turns on one number, and `.claude/plan.md` states
the trap plainly: if a step's token batch routes to nearly every expert anyway,
**the read volume does not fall** and only the memory bound improves. Our step
reads the stack twice (forward, then the backward recompute), so on a 744B store
that is ~600 GB per step; at the ~4.3 GB/s the disk tier averages on a COLD 70B
step after #974 - the 4.5-5.1 GB/s figure is the warm 7B rate, corrected here
after review - that is around two minutes per step even if nothing else got
worse.

So: **union coverage** — the fraction of a layer's experts that at least one
token in the step routes to — measured per layer, on a real model, on real text,
at the batch shapes Soup actually trains at.

## 1. Why a coverage number alone means nothing

With `E` experts, `k` picked per token and `T` tokens routed independently and
uniformly, the expected coverage is

```
1 - (1 - k/E)^T
```

For OLMoE's 64 experts / top-8 at T = 512 tokens that is `1 - (7/8)^512`, which
is `1 - 10^-29.7`. **Chance alone predicts coverage indistinguishable from
100%.** Even a single sequence of 512 tokens is far more than enough to touch
every expert if routing is anywhere near uniform.

A measured coverage of 0.95 is therefore not "most experts are used"; it is a
*large* departure from chance in the direction that helps. Every coverage figure
in §5 is printed beside this baseline, and the harness computes it.

This also sets the bar honestly: for the read to fall by half, routing must be
concentrated enough that half the experts see **no token at all** out of 512 —
roughly 4,096 assignments landing on at most 32 of 64 experts.

### The baseline across the whole family, computed before any model was loaded

Pure arithmetic from the formula above, so it needs no hardware and no download,
and it is what makes the bar concrete. The last two columns are the token budget
at which uniform routing would reach 50% and 99% coverage.

| model | E | k | baseline at T=512 | at T=2048 | T for 50% | T for 99% |
|---|---|---|---|---|---|---|
| `OLMoE-1B-7B` (measured here) | 64 | 8 | 1 - 10^-29.7 | 1 - 10^-119 | 5.2 | 34 |
| `granite-3.0-1b-a400m-base` (measured here) | 32 | 8 | 1 - 10^-64.0 | 1 - 10^-256 | 2.4 | 16 |
| `Qwen3-30B-A3B` (plan; not measured, §7) | 128 | 8 | 1 - 10^-14.4 | 1 - 10^-57 | 10.7 | 71 |
| `Mixtral-8x7B` | 8 | 2 | 1 - 10^-64.0 | 1 - 10^-256 | 2.4 | 16 |
| DeepSeek-V3 class | 256 | 8 | 1 - 10^-7.1 | 1 - 10^-28 | 21.8 | 145 |

**Read the last column.** Under uniform routing, even a 256-expert model needs
only **145 tokens** to touch 99% of its experts, and Soup's smallest streaming
step is 512. Expert-granularity streaming can only save READS if real routing is
concentrated enough that a 512-token step behaves like roughly five tokens'
worth of routing diversity. That is the size of the departure from chance the
feature needs, and stating it before the measurement is what keeps a coverage of
0.9 from being reported as encouraging.

A larger expert count does move the baseline in the helpful direction — the
whole column shifts right as E/k grows — which is precisely why §7 records that
the largest model measured here has 64 experts, and that the plan's own target
class (744B, 19,456 routed experts across the stack) is far outside it.

## 2. The decision rule, written before the run

Taken from `.claude/plan.md` and made specific. `C` is the mean union coverage
over layers at the shape Soup trains at (batch x seq = 512 and 2048); `S` is the
skew, reported as the share of assignments taken by the busiest 25% of experts.

| measured | verdict | what ships |
|---|---|---|
| **C <= 0.50** | both memory AND read win | schedule the feature: per-expert sharding, an expert buffer pool, the planner arithmetic in expert units. Size: the whole async-NVMe project again |
| **0.50 < C < 0.85** | ambiguous | do not schedule on this evidence. Re-measure at the shapes and models the decision would actually apply to, and state the read saving as `1 - C` with its spread, not as a headline |
| **C >= 0.85** | memory only | build it, if at all, as **"a capacity tier for MoE"** with an explicit promise of **no throughput gain** — the same honesty v0.72.3's 70B demonstration was given |
| **S >= 0.60** (busiest quartile takes 60%+ of traffic) | hot/cold split is real | a **hybrid RAM+disk tier** — hot experts resident, cold ones streamed — is the cheaper first deliverable and should precede full expert streaming, whatever C says |
| **hot-set Jaccard < 0.7 step to step** | the hot set is not stable | a persisted per-dataset heat file is NOT the input it looks like; the hybrid tier would have to re-learn continuously, which changes its cost |

**A third condition, which the plan does not state and which I am adding before
seeing any number.** Even at low coverage, a read saving only materialises if
the expert set can be known EARLY ENOUGH to prefetch. Today's prefetcher is a
perfect predictor because the walk is deterministic — forward 0..L-1, backward
L-1..0 — and `stream_read_ahead` exploits exactly that. Which *experts* a layer
needs is not known until its router has run, which is after the previous layer's
output exists. Expert-granularity therefore trades a perfect prefetch for a
conditional one, and a reader that must wait for the router is a reader that
stalls. Colibrì's answer is a lookahead that predicts the next layer's experts
from the current one (they report 71.6% one layer ahead). **That predictability
is measured in step 0 after all**, and until the numbers are in, a "C <= 0.50"
result licenses scheduling the feature's *design*, not a throughput claim.


**So step 0 measures it, because it is free once the router indices are
captured.** Two candidate predictors, each reported as the share of layer K+1's
*traffic* it would have caught - weighted by assignments, not by expert, since a
prefetch that catches the busiest experts and misses three idle ones has done
its job:

- **the set layer K itself just used** - the cheapest possible lookahead, no
  model, no training, available the moment layer K's router has run;
- **layer K+1's corpus-wide busiest quartile** - what a persisted per-dataset
  heat file would hold, i.e. the hybrid tier's own predictor.

A fourth row of the rule follows: **if neither predictor catches most of the
traffic, the read-win branch is not reachable even at low coverage**, because
the reader would have to wait for each router before it could fetch. What is
still NOT measured is a *learned* predictor of the kind the C engine reports;
these two are the free lower bound on what one could be worth.

**These numbers are only interpretable where coverage is well below 1.0.** If
nearly every expert is touched, "layer K's used set" is nearly every expert and
a hit rate near 1.0 says only that prefetching everything works - which is what
the layer tier already does. The harness says so at the point of use.

## 3. What is measured, and on what

Every identifier below was resolved against the Hub before the run rather than
written from memory — configs only, a few KB each — and two of the four I first
wrote down were wrong. The corrections are in the table.

| model | model_type | layers | E / k | hidden | why it is here |
|---|---|---|---|---|---|
| `allenai/OLMoE-1B-7B-0924` | `olmoe` | 16 | 64 / 8 | 2048 | the plan names it; 6.9B total / 1.3B active, the smallest fully-trained MoE with a realistic expert count. No shared experts |
| `ibm-granite/granite-3.0-1b-a400m-base` | `granitemoe` | 24 | 32 / 8 | 1024 | a second (E, k) point at half the experts and a different architecture, 1.3B total so it fits the card in bf16 — which makes it the arm that needs no quantisation at all |

`ibm-granite/granite-3.0-1b-a400m` (without `-base`) **does not exist**; the Hub
carries `-base` and `-instruct`. `Qwen/Qwen3-30B-A3B` resolves and is
`qwen3_moe`, 48 layers, 128 experts, top-8 — confirming the numbers §1's table
uses for it, and it is still skipped for the reason §7 gives.

Three corpora, because routing skew is a property of the text and a single
domain would overstate it:

- **prose** — `hf:Salesforce/wikitext:wikitext-2-raw-v1:train:text`;
- **code** — this repository's own `src/soup_cli/`, which is real Python and
  needs no download;
- **math** — `hf:openai/gsm8k:main:train:question`.

**The canonical owner prefixes are load-bearing on `datasets` 5.0.1**: the bare
ids `wikitext` and `gsm8k`, which most documentation still shows, both fail with
`HfUriError: Invalid HF URI ... Repository id`. Checked by streaming one row from
each.

Shapes: `1x512`, `4x512`, `1x2048` — the plan asks for batch x seq of 512 and
2048, and the two ways of reaching 2048 are measured separately because routing
is context-dependent and a 4x512 step is not a 1x2048 step.

## 4. The instrument, and what was done to trust it

`benchmarks/harness/moe_expert_coverage.py`, standalone (no Soup import, so it
runs against a stock transformers install). Per layer it reports union
coverage, routing skew (Gini plus the share taken by the busiest 10/25/50% of
experts), hot-set stability (Jaccard of the busiest quartile between steps and
against the corpus), and the two one-layer-ahead prefetch hit rates above.

Router discovery is by SHAPE rather than an architecture table: in transformers
5.17 every MoE decoder — olmoe, qwen3_moe, mixtral, granitemoe, deepseek\* —
exposes a `*TopKRouter` module returning `(router_logits, router_scores,
router_indices)` with integer indices of shape `(tokens, top_k)`. The harness
hooks anything of that shape, falls back to taking `topk` of a float
`(tokens, num_experts)` output itself, and **refuses by name** rather than
guessing if neither holds. The order of the tuple deliberately does not matter,
and that is not a hypothetical: `granitemoe` puts the indices first where the
other three put them last.

Tokens are **packed, not padded**: the corpus is tokenised into one stream and
sliced into exact `seq`-length chunks. A padded batch routes its PAD positions
too, and those assignments would land in the counts as if they were traffic.

Validated 2026-09-15 before any real model was downloaded
(`validate_moe_harness.py`, kept in the session scratchpad; it builds a 6.62M
-param OLMoE with random weights and drives the harness end to end):

- the statistics against hand-computed values — `uniform_coverage(8,2,4) =
  1-0.75^4`, `gini([1,1,1,1]) = 0`, `gini([0,0,0,4]) = 0.75`, `top_share`,
  `hot_set`, `jaccard`;
- index extraction: given `(logits, scores, indices)` it takes the integer
  tensor, given logits alone it takes the right top-k itself, and given neither
  shape it returns nothing rather than inventing an answer;
- **the arithmetic identity that is the real check**: the per-expert counts must
  sum to exactly `tokens x top_k x steps` at every layer. They do, at 3 layers x
  2 shapes. That is what proves no token was dropped or double-counted;
- the predictability numbers exist for every layer but the first, are bounded in
  [0, 1], and - the discriminating one - the corpus-hot-quartile hit rate never
  exceeds that layer's own top-25% traffic share, which it cannot by the
  definition of the hot set and would only do if the arithmetic were wrong;
- **and the same end-to-end run against a tiny `granitemoe`**, which is the
  check that shape-based discovery is not just a nicer way of writing an
  architecture table. `granitemoe`'s router returns
  `(top_k_index, top_k_weights, router_logits)` - the indices FIRST - where
  olmoe, qwen3_moe and mixtral all return `(router_logits, router_scores,
  router_indices)` with the indices last. A table keyed on position would have
  read granite's indices as logits; scanning for the integer tensor whose last
  dimension is `top_k` reads both correctly, and the counts add up on both.

**One correction, kept because it is the point.** A first version of the
validation asserted that 512 tokens over 8 experts "must reach coverage 1.0, as
chance predicts". It measured 0.979, and the **assertion** was wrong, not the
harness: a randomly-initialised router is a fixed random map rather than a
uniform one, and the validation corpus is one sentence repeated, so the hidden
states barely vary and the routing concentrates (gini 0.43). The check now
asserts that shape — coverage below the baseline, with visible skew — which is
the behaviour the harness exists to detect.

### The NF4 caveat, and how it is bounded rather than assumed

OLMoE in bf16 is 13.8 GB and does not fit 8 GB of VRAM, so the primary arm loads
it as NF4 (`--load-4bit`). The router itself is **not** quantised: in
transformers 5.17 a top-k router holds its weight as a bare `nn.Parameter` used
through `F.linear`, not an `nn.Linear`, so `replace_with_bnb_linear` never sees
it. What quantisation can still move is the hidden state the router reads. That
is why a **bf16 control on CPU** — one corpus, one shape, few steps — is part of
the run rather than a footnote: the difference between the two arms bounds the
caveat with a number.

## 5. Results

Measured 2026-09-15 19:53-20:19 local, on a box with nothing else running: a
peer session's cold 70B series had finished and its author confirmed the machine
free. Raw JSON under `benchmarks/results/probe-rtx5070/moe/`.

**Box state, and where each number comes from** - a reviewer asked, and the
answer was not in the record. Free physical RAM 18.78-20.56 GB, commit charge
21.38-24.01 GB of a 53.45 GB limit, **2 Python processes throughout**, sampled
either side of all three blocks. Those came from a shell wrapper that is
deliberately NOT committed (§8) and were therefore, as originally written,
unverifiable by a reader.

Two things about that, one narrower than the review assumed and one wider. The
**narrower**: the wrapper's broken query and its box stamp are different
mechanisms. What could not fire was its neighbour check, a
`Get-CimInstance Win32_Process -Filter` whose nested quotes were malformed; the
stamp used `Get-CimInstance Win32_OperatingSystem` plus
`Get-Process python | Measure-Object`, neither of which errored. The stamp is
also positively controlled rather than merely un-errored: run again at 20:27:10
while a peer's ablation was on the card it returned 4 Python processes, and at
20:20:40 with nothing running it returned 2. It discriminates.

The **wider**: none of that was in the results, so a reader still had to take my
word. Fixed forward rather than argued - `host_state()` and
`harness_fingerprint()` now record available RAM, swap, the Python process count
and a SHA-256 of the harness itself into `meta.host_before` / `meta.host_after`
of every run, so a future result carries its own box state and names the code
that produced it. **`granite_bf16_cuda_run2.json` is the first block that
carries one** — available RAM 17.78 GB before and 16.86 GB after, 3 Python
processes, harness `61b9a4be` — and run 1 does not, which is why both are
published.

**And it should be said plainly that none of this can move a number here.** A
routing count is deterministic given the weights and the tokens: the top-k of a
router does not depend on how much RAM was free. Memory pressure could have made
the run slower, and there are no timing claims in this record. This paragraph is
about whether the record can be checked, not about whether the result is right.

Every figure below is a mean over the model's layers, over 16 steps per shape.
**The chance baseline is 1.000 to three decimals for every row in both tables**,
which is the point of §1: these coverages are *below* chance, and that is the
only sense in which they are interesting.

### granite-3.0-1b-a400m-base — 24 layers, 32 experts, top-8, bf16, no quantisation

| corpus | shape | coverage (vs chance) | min layer | max layer | Gini | top-25% share | hot-25 Jaccard step-to-step | caught by prev layer's set | caught by corpus hot-25 |
|---|---|---|---|---|---|---|---|---|---|
| prose | 1x512 | 0.997 (chance 1.000) | 0.971 | 1.000 | 0.290 | 0.432 | 0.581 | 0.994 | 0.434 |
| prose | 4x512 | 0.999 (chance 1.000) | 0.982 | 1.000 | 0.285 | 0.430 | 0.705 | 0.997 | 0.432 |
| prose | 1x2048 | 0.999 (chance 1.000) | 0.975 | 1.000 | 0.292 | 0.435 | 0.656 | 0.996 | 0.437 |
| code | 1x512 | 0.997 (chance 1.000) | 0.986 | 1.000 | 0.405 | 0.509 | 0.747 | 0.997 | 0.513 |
| code | 4x512 | 1.000 (chance 1.000) | 1.000 | 1.000 | 0.409 | 0.511 | 0.827 | 1.000 | 0.515 |
| code | 1x2048 | 1.000 (chance 1.000) | 0.996 | 1.000 | 0.406 | 0.509 | 0.805 | 1.000 | 0.513 |
| math | 1x512 | 0.998 (chance 1.000) | 0.969 | 1.000 | 0.312 | 0.446 | 0.738 | 0.995 | 0.449 |
| math | 4x512 | 0.999 (chance 1.000) | 0.984 | 1.000 | 0.314 | 0.448 | 0.889 | 0.998 | 0.450 |
| math | 1x2048 | 0.998 (chance 1.000) | 0.979 | 1.000 | 0.312 | 0.446 | 0.858 | 0.996 | 0.449 |

### OLMoE-1B-7B-0924 — 16 layers, 64 experts, top-8, NF4

| corpus | shape | coverage (vs chance) | min layer | max layer | Gini | top-25% share | hot-25 Jaccard step-to-step | caught by prev layer's set | caught by corpus hot-25 |
|---|---|---|---|---|---|---|---|---|---|
| prose | 1x512 | 0.980 (chance 1.000) | 0.958 | 0.998 | 0.289 | 0.428 | 0.372 | 0.982 | 0.428 |
| prose | 4x512 | 1.000 (chance 1.000) | 0.998 | 1.000 | 0.271 | 0.416 | 0.496 | 1.000 | 0.415 |
| prose | 1x2048 | 0.996 (chance 1.000) | 0.990 | 1.000 | 0.271 | 0.413 | 0.397 | 0.997 | 0.413 |
| code | 1x512 | 0.967 (chance 1.000) | 0.909 | 1.000 | 0.552 | 0.663 | 0.686 | 0.963 | 0.675 |
| code | 4x512 | 0.990 (chance 1.000) | 0.961 | 1.000 | 0.558 | 0.668 | 0.806 | 0.992 | 0.681 |
| code | 1x2048 | 0.985 (chance 1.000) | 0.942 | 1.000 | 0.535 | 0.642 | 0.718 | 0.987 | 0.654 |
| math | 1x512 | 0.984 (chance 1.000) | 0.955 | 1.000 | 0.412 | 0.520 | 0.676 | 0.981 | 0.524 |
| math | 4x512 | 0.997 (chance 1.000) | 0.988 | 1.000 | 0.414 | 0.520 | 0.803 | 0.997 | 0.525 |
| math | 1x2048 | 0.995 (chance 1.000) | 0.985 | 1.000 | 0.422 | 0.526 | 0.789 | 0.994 | 0.532 |

Per layer, corpus `code`, shape `1x512` (chance baseline 1.000000):

| layer | coverage | experts touched | Gini | top-10% | top-25% | caught by prev layer | caught by corpus hot-25 |
|---|---|---|---|---|---|---|---|
| 0 | 0.996 | 63.8 / 64 | 0.354 | 0.277 | 0.487 | - | - |
| 1 | 1.000 | 64.0 / 64 | 0.360 | 0.313 | 0.504 | 0.997 | 0.504 |
| 2 | 0.999 | 63.9 / 64 | 0.407 | 0.343 | 0.546 | 1.000 | 0.546 |
| 3 | 0.993 | 63.6 / 64 | 0.495 | 0.345 | 0.611 | 0.999 | 0.611 |
| 4 | 0.987 | 63.2 / 64 | 0.590 | 0.493 | 0.710 | 0.985 | 0.710 |
| 5 | 0.979 | 62.6 / 64 | 0.611 | 0.476 | 0.721 | 0.964 | 0.721 |
| 6 | 0.978 | 62.6 / 64 | 0.573 | 0.422 | 0.672 | 0.982 | 0.672 |
| 7 | 0.966 | 61.8 / 64 | 0.597 | 0.465 | 0.693 | 0.968 | 0.693 |
| 8 | 0.971 | 62.1 / 64 | 0.577 | 0.429 | 0.679 | 0.928 | 0.679 |
| 9 | 0.980 | 62.8 / 64 | 0.595 | 0.453 | 0.698 | 0.946 | 0.698 |
| 10 | 0.968 | 61.9 / 64 | 0.532 | 0.391 | 0.635 | 0.957 | 0.635 |
| 11 | 0.957 | 61.2 / 64 | 0.627 | 0.480 | 0.737 | 0.980 | 0.737 |
| 12 | 0.926 | 59.2 / 64 | 0.617 | 0.477 | 0.722 | 0.957 | 0.722 |
| 13 | 0.953 | 61.0 / 64 | 0.639 | 0.514 | 0.739 | 0.836 | 0.739 |
| 14 | 0.912 | 58.4 / 64 | 0.617 | 0.444 | 0.718 | 0.979 | 0.718 |
| 15 | 0.909 | 58.2 / 64 | 0.635 | 0.477 | 0.734 | 0.963 | 0.734 |

Verdict inputs: **C = 0.967 to 1.000** over 9 corpus/shape combinations; **S = 0.413 to 0.668**.

### Finding 0 — the measurement reproduces exactly, and finding out cost a corpus

granite was run a second time at 20:36, 41 minutes after the first, with the
identical configuration — same model, same three corpora, same three shapes,
same 16 steps, same seed 17 — published as
`granite_bf16_cuda_run2.json` beside run 1 rather than replacing it. Routing is
deterministic given the weights and the tokens, so two runs of one configuration
must agree EXACTLY, and a difference is a finding rather than noise. Suggested
in review, and it earned its keep immediately.

| corpus | same text? | per-expert counts | largest difference in any statistic |
|---|---|---|---|
| prose | yes (`399831ef40995321`) | **identical at every layer and shape** | **0.000e+00** |
| math | yes (`21b099a4d0676f55`) | **identical at every layer and shape** | **0.000e+00** |
| code | **NO** (`111769fb…` -> `e2213959…`) | differ | 6.741e-03 |

**Where the input was identical, the output is identical to the last bit** —
per-expert counts, not merely rounded summaries. That is the strongest form the
reproducibility check can take, and it is what licenses reading a 0.997 here as
a measurement rather than a sample.

**Where it was not identical, the cause is mine and it is a design fault worth
publishing.** The code corpus is not a fixture: it is this repository's own
`src/soup_cli`, read live. Between the two runs I merged `origin/main` into the
branch, which changed 13 files under that directory (+799 / -212 lines,
including #989's layer-stream work). So the corpus moved, and the statistics
moved with it by 6.7e-03. **Nothing about the model changed; the text did.**

Three things follow, and the first two are already in the harness:

- the corpus digest is what caught it, and it caught it on first use — a
  comparison that could not fail would have been worth nothing;
- a directory corpus now records its **git revision** (`meta.corpora[].git_revision`,
  with a `-dirty` suffix when the tree is modified), so the next such difference
  is explicable rather than merely visible;
- `os.walk` was yielding directories in filesystem order, which on this box is
  **not** sorted (`data/providers` before `data/_fixtures`), so the same
  directory would concatenate differently on another machine. Now sorted in
  place. **This changes the code corpus's order relative to both published
  runs**, so a third run will produce a third digest — which is exactly why each
  result carries its own.

The honest reading of the code-corpus rows in the tables below is therefore:
they describe a real corpus of real Python, at a stated tree, and they are not
reproducible from the repository name alone. Prose and math are, and they
reproduced.

### Finding 1 — a training step touches essentially every expert, at every shape, on both models

**C = 0.967 to 1.000** over 18 corpus/shape combinations across the two models.
The lowest single layer anywhere is **0.909** — OLMoE, code, layer 15, at the
smallest shape (1 x 512), which is still 58.2 of 64 experts. The one place a
layer is saturated across the board is **granite on code at 2048 tokens**,
where all 24 layers touch all 32 experts; prose and math on the same model and
shape still leave a layer at 0.982 and 0.984. So even "every expert, every
layer" is corpus-dependent.

More tokens means more coverage, monotonically and as chance predicts: OLMoE on
code goes 0.967 at 512 tokens to 0.990 at 2048. There is no shape in the
measured range where a step reads a usefully smaller set.

### Finding 2 — concentration is real, corpus-dependent, and grows with depth

Coverage says nothing about how the traffic is distributed, and the two answers
are different. Gini over the same runs is **0.271 (OLMoE, prose) to 0.558
(OLMoE, code)**, and the busiest quartile of experts takes **41% to 67%** of all
assignments depending on the corpus.

Code concentrates hardest on both models (granite 0.41 Gini vs 0.29 on prose;
OLMoE 0.55 vs 0.27), which is why three domains were measured rather than one:
a single-domain corpus would have reported this as a property of the model.

It also **grows with depth**, and the per-layer tables are where that shows.
OLMoE on code, 1 x 512: layer 0 has coverage 0.996 at Gini 0.354, layer 15 has
coverage 0.909 at Gini 0.635. Prose and math on the same model move much less
(L0 0.998/0.263 to L15 0.974/0.353, and 0.999/0.303 to 0.975/0.436). A mean over
layers hides this, which is why both tables carry a per-layer breakdown.

### Finding 3 — the prefetch numbers are trivial here, exactly as pre-warned

"Traffic caught by the previous layer's used set" comes out at **0.963 to
1.000** everywhere. That is not a result. §2 said so before the measurement: at
coverage near 1.0 the previous layer's used set is nearly every expert, so the
hit rate is near 1.0 by construction and means only that prefetching everything
works — which is what the layer tier already does.

The other predictor is meaningful and says something different: prefetching a
layer's **corpus-wide busiest quartile** catches **0.413 to 0.681** of its
traffic, matching that layer's top-25% share to within 0.005 on granite and 0.013 on
OLMoE. That
agreement is itself a result: per-step traffic is close to the corpus average,
i.e. **steps are homogeneous** and a heat file computed once would not be
chasing a moving target.

Whether the hot SET is stable is a separate question and the answer splits by
corpus. Step-to-step Jaccard of the busiest quartile is **0.372-0.496 on OLMoE
prose** — below the rule's 0.7 — and **0.676-0.806 on OLMoE code and math**. On
granite it is 0.581-0.889. So the hot set is stable where the traffic is
concentrated and unstable where it is not, which is coherent but means a
persisted heat file is a code-and-math device, not a general one.

### Finding 4 — NF4 did not move the answer

The control re-ran OLMoE in **bf16 on the CPU** over math at 4 x 512, which is
the one arm that needs no quantisation at all. Against the NF4 arm's same corpus
and shape:

| | NF4 / cuda | bf16 / cpu | delta |
|---|---|---|---|
| coverage (mean over layers) | 0.9974 | 0.9966 | -0.0008 |
| coverage (worst layer) | 0.9883 | 0.9883 | 0.0000 |
| Gini | 0.4136 | 0.4173 | +0.0037 |
| busiest quartile's share | 0.5200 | 0.5235 | +0.0034 |

The largest per-layer coverage difference anywhere is **0.0068**, at layer 7.

**The check is stronger than a single-variable comparison would be, because the
two arms differ in four ways at once**: precision (NF4 against bf16), device
(cuda against cpu), step count (16 against 4) and corpus slice (4000 rows
against 1500, and the record's own sha256 of the two differs). So the ~0.004
agreement constrains **all four together**, which is the useful direction:
whatever NF4 does to routing here is two orders of magnitude smaller than the
corpus-to-corpus spread the same model shows (Gini 0.271 on prose against 0.558
on code).

**"Bound" was the wrong word and the review was right to press on it.** Four
differences agreeing to 0.004 is not a bound unless they cannot cancel, and
nothing here proves they cannot. What makes cancellation implausible rather than
excluded is that **four different metrics agree simultaneously** - coverage,
worst layer, Gini and the quartile share - so a cancellation would have to hold
across all of them at once. The tighter control exists and was not run: bf16 on
CPU over the same 4000-row slice at 16 steps leaves only precision and device,
at ~25 minutes of CPU. The verdict does not turn on it, since the means are
0.967-1.000 against a 0.85 threshold.

The router itself was never quantised — in transformers 5.17 a top-k router
holds its weight as a bare `nn.Parameter` used through `F.linear`, so
`replace_with_bnb_linear` does not see it — and this measures the remaining
question, which is what NF4 does to the hidden state the router reads.

### What follows arithmetically, and is NOT a measurement

With union coverage at ~1.0, **the read volume of an expert-streamed step equals
the read volume of a layer-streamed one**: every expert is demanded at least
once, so every expert is fetched, and how often each is used afterwards changes
nothing about the bytes. That is the plan's own trap paragraph, confirmed.

It also fixes what a hot/cold split would be worth, and the answer is not the
skew figure. If a fraction `f` of a layer's experts are kept resident, the
streamed bytes fall to `coverage - f`, i.e. **by `f`**. Keeping the busiest
quartile of OLMoE's experts resident saves **25% of the read volume**, not the
67% of *traffic* those experts carry. The traffic share decides whether a hot
set can be picked at all and whether it is stable; it does not size the saving.

Two consequences, both raised in review of this record and both arithmetic
rather than measurement:

- **There is one regime where the traffic share WOULD size a saving, and it does
  not rescue the feature**: micro-batching within a step without caching experts
  across micro-batches, so a hot expert is re-fetched per micro-batch. That is
  strictly worse than the batch-union measured here, so it cannot turn a
  non-saving into a saving; it can only make the un-cached case cost more.
- **On the class the plan is aimed at, the hybrid tier's saving is about 5%.** A
  744B-class MoE carries on the order of 600 GB of routed experts, and the
  fraction an 8 GB card with 32 GB of host RAM can hold resident is roughly 5%.
  By the arithmetic above the read saving is that fraction. This strengthens the
  negative rather than softening it.

## 6. Verdict

**Against the rule committed in §2 before any of this was measured:**

| rule input | measured | branch |
|---|---|---|
| `C` | **0.967-1.000** (worst single layer 0.909) | **`C >= 0.85` — memory only** |
| `S` | 0.413-0.681; `>= 0.60` only on OLMoE + code | hybrid tier indicated **for code-like data on 64 experts**, not generally |
| hot-set Jaccard | 0.372-0.889; `< 0.7` on OLMoE prose | a persisted heat file is not a general device |
| one-layer-ahead | 0.963-1.000, uninterpretable at this coverage | no throughput claim is licensed |

**So: expert-granularity streaming is a capacity tier for MoE, and it must be
shipped with an explicit promise of no throughput gain.** It would let a model
whose single NF4 layer does not fit two VRAM buffers train at all, which is a
real and useful thing — the same kind of thing v0.72.3's 70B demonstration was,
and it deserves the same honesty about what it does not do.

**The plan's step 1 should not be scheduled on this evidence.** It is sized at
"the whole async-NVMe project again"; what the measurement licenses is a
capacity tier, and the read-bound problem it was hoped to solve is untouched.

**The cheaper deliverable the rule points at is narrower than the rule assumed.**
A hybrid RAM+disk tier is defensible only where the traffic concentrates AND the
hot set is stable, which here is code and math on the 64-expert model, not prose
and not the 32-expert one. And by the arithmetic above its saving is the
resident fraction, so it is a memory-for-bandwidth trade with a known exchange
rate rather than a throughput feature.

## 7. What this did NOT measure

- **`Qwen3-30B-A3B`, which the plan names.** 60 GB in bf16 against 31.7 GB of
  RAM and 8 GB of VRAM; NF4 is ~15 GB, which fits neither the card nor a
  comfortable share of host RAM alongside a 60 GB download. It is skipped, and
  the two models in §3 are what this box can honestly carry. The consequence is
  stated rather than hidden: **the largest expert count measured here is 64**,
  and a 128- or 256-expert model at the same token budget would have a LOWER
  chance baseline and could behave differently. Note which way that cuts: at
  128 experts chance still predicts 99% coverage within 71 tokens, so a bigger
  model makes the verdict LESS likely to change, not more - but it is not
  measured, and the two models here agreed rather than diverged, which is one
  data point about the trend and not two.
- **A learned one-layer-ahead predictor.** Step 0 measures the two FREE
  predictors instead: the previous layer's own used set, and a corpus-wide heat
  quartile. They bound from below what a learned one could be worth, without
  training anything.
- **Any timing.** The forward ran with a hook on every router and, in the CPU
  control, at no throughput anybody should quote. The wall-clock figures in §8
  are for planning a re-run, not performance evidence.
- **Backward routing.** Only the forward's router decisions are observed; the
  recompute re-runs the same routers on the same hidden states, so the expert
  SET is the same, but that is an argument rather than a measurement.
- **Training-time drift.** All routing here is from a trained checkpoint at rest.
  Whether a fine-tune moves the hot set is a different question.

## 8. Reproducing

Both blocks, as run, from the repository root. No `soup_cli` import and no
`PYTHONPATH`; the harness runs against a stock transformers install.

```bash
python benchmarks/harness/moe_expert_coverage.py \
  --model ibm-granite/granite-3.0-1b-a400m-base \
  --data hf:Salesforce/wikitext:wikitext-2-raw-v1:train:text --data-label prose \
  --data src/soup_cli --data-label code \
  --data hf:openai/gsm8k:main:train:question --data-label math \
  --shape 1x512 --shape 4x512 --shape 1x2048 --batches 16 --limit-rows 4000 \
  --device cuda --dtype bfloat16 \
  --out benchmarks/results/probe-rtx5070/moe/granite_bf16_cuda.json

python benchmarks/harness/moe_expert_coverage.py \
  --model allenai/OLMoE-1B-7B-0924 \
  ... same --data / --shape / --batches ... \
  --device cuda --dtype bfloat16 --load-4bit \
  --out benchmarks/results/probe-rtx5070/moe/olmoe_nf4_cuda.json

python benchmarks/harness/moe_expert_coverage.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --data hf:openai/gsm8k:main:train:question --data-label math \
  --shape 4x512 --batches 4 --limit-rows 1500 --device cpu --dtype bfloat16 \
  --out benchmarks/results/probe-rtx5070/moe/olmoe_bf16_cpu_control.json
```

Tables from the JSON, without re-running anything:

```bash
python benchmarks/harness/moe_coverage_report.py \
  benchmarks/results/probe-rtx5070/moe/*.json --layer-corpus code --layer-shape 1x512
```

Timings on this box, for planning rather than as a claim: granite 2m26s
including its download, OLMoE NF4 16m28s (the 13.8 GB fetch dominates), the CPU
control 6m42s. Free physical RAM never fell below 18.78 GB and the Python
process count stayed at 2.

**One thing the runner used is deliberately NOT committed.** A shell wrapper
stamped the host baseline and refused to start when a neighbour's probe was
alive; its process-counting query was malformed and returned 0 whether or not
anything was running, i.e. it was a guard that could not fire. The box was
verified free independently, so no number here is affected, but a broken guard
is worse than none and shipping it would have been worse still. The commands
above are what ran; the box-state discipline is in the prose, where it cannot
silently fail.
