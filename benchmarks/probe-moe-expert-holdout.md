# Expert heat profiles: correction to Finding 3 and a held-out check

**Status: measured 2026-10-09; rule recorded 2026-10-08.** This is a
separate record. Neither [`probe-moe-expert-coverage.md`](probe-moe-expert-coverage.md)
nor its [`moe_expert_coverage.py`](harness/moe_expert_coverage.py) is changed.
The new [`moe_expert_holdout.py`](harness/moe_expert_holdout.py) imports that
harness and records both source fingerprints.

## 1. Correction to Finding 3

The published record at [`9aa43bd7`](https://github.com/SoupTeam/soup-rnd/blob/9aa43bd7/benchmarks/probe-moe-expert-coverage.md#L437-L443), lines 437-443, says:

> The other predictor is meaningful and says something different: prefetching a
> layer's **corpus-wide busiest quartile** catches **0.413 to 0.681** of its
> traffic, matching that layer's top-25% share to within 0.005 on granite and 0.013 on
> OLMoE. That
> agreement is itself a result: per-step traffic is close to the corpus average,
> i.e. **steps are homogeneous** and a heat file computed once would not be
> chasing a moving target.

That agreement is an identity, not evidence of homogeneous steps [CODE].
`global_hot` is selected from `current.counts`, the sum of the same steps on
which `traffic_caught_by_corpus_hot25` is then scored
(`moe_expert_coverage.py:474-479`). Each step has the same assignment count
`tokens × top_k`; the original record's validation checks that accounting
(lines 220-222) [DOC]. For step counts $C_{t,e}$, $A = T k$ assignments per
step, $n$ steps and the hottest set $H$ of the aggregate counts:

$$
\frac{1}{n}\sum_t\frac{\sum_{e\in H}C_{t,e}}{A}
=\frac{\sum_{e\in H}\sum_t C_{t,e}}{nA}
=\operatorname{top25\_share}\!\left(\sum_t C_{t,e}\right).
$$

The right-hand side is exactly what `top25_share` computes at line 505 [CODE].
Steps with different hot experts can still satisfy this identity. The original
record describes the same-set score as an upper bound (lines 223-226), but
does not say that this construction always attains it [DOC]. This correction
changes the interpretation of Finding 3, not the original union-coverage
measurement or its read-volume arithmetic.

Recalculation from all four published JSON files [RUN]:

| Published file in `results/probe-rtx5070/moe/` | Scored layer rows | Maximum absolute layer difference | Maximum summary gap from different layer populations |
|---|---:|---:|---:|
| `granite_bf16_cuda.json` | 207 | 0.0 | 0.0044128100 |
| `granite_bf16_cuda_run2.json` | 207 | 0.0 | 0.0044065006 |
| `olmoe_nf4_cuda.json` | 135 | 0.0 | 0.0123450438 |
| `olmoe_bf16_cpu_control.json` | 15 | 0.0 | 0.0050322215 |
| **Total** | **564** | **0.0** | — |

The text's 0.005/0.013 tolerances describe averaging different sets of layers:
`top25_share` includes layer zero, while the corpus-hot predictor is only
reported for layers 1..L-1. Averaging both over the same scored layers gives
exact equality. These JSON files do not contain per-step counters; a held-out
split cannot be reconstructed from them. Section 2 therefore defines a new
measurement, not a reproduction of the RTX 5070 figures.

## 2. Decision rule, committed before the new measurement

**Question.** Does a hottest-quarter profile selected on one contiguous half
of the measured steps retain a material traffic-hit advantage on the other
half, in both directions? This is checkpoint-at-rest routing, not adapter
training drift, batch-1 decode, SSD timing or measured read savings.

### Configuration and provenance

- Models: `ibm-granite/granite-3.0-1b-a400m-base` and
  `allenai/OLMoE-1B-7B-0924`.
- Corpora: prose (`Salesforce/wikitext`, `wikitext-2-raw-v1`, `train`, `text`),
  code (`src/soup_cli` at the measured source commit), and math (`openai/gsm8k`,
  `main`, `train`, `question`). Use the first 4000 accepted nonempty strings
  (`--limit-rows 4000`), as the original harness does. The directory corpus's
  deterministic walk is imported unchanged.
- Shapes: `1x512`, `4x512`, `1x2048`; exactly 16 steps per shape; seed 17.
  Packing and seeded chunk permutation are imported unchanged. Enough distinct
  chunks for all 16 steps are required; no wraparound or silent step reduction.
- **New arm:** CUDA, bfloat16 base weights, no quantisation, eager attention and
  eager expert execution. No heavy local model run. Cloud types, tried in order:
  `hyperstack_A6000` (48 GB VRAM, listed at $0.60/h),
  `massedcompute_A6000_plus` (48 GB, $0.68/h),
  `g2-standard-8:nvidia-l4:1` (24 GB, $1.02/h). Record the actual device, RAM,
  storage, OS, Python and package versions; no equivalence to the dev box is
  assumed. Types were checked with `brev create --dry-run` [RUN].
- Requested immutable pins, resolved again on the box and used for the actual
  model, tokenizer and dataset reads:

  | Source | Commit |
  |---|---|
  | [Granite model and tokenizer](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base/tree/d91cbed802d85eb1b32374623331f6ab2b37403a) | `d91cbed802d85eb1b32374623331f6ab2b37403a` |
  | [OLMoE model and tokenizer](https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/6d84c48581ece794365f2b8e9cfb043c68ade9c5) | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
  | [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3) | `b08601e04326c79dfdd32d625aee71d232d685c3` |
  | [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) | `740312add88f781978c0658806c59bc2815b9866` |

  Pins obtained from the first-party Hugging Face model/dataset APIs [DOC].
  Results must name the actually resolved pins and text/token-stream digests;
  a requested pin alone is not proof of the downloaded revision. The code
  corpus is the committed bundle's tree; its source commit is supplied
  explicitly because `git archive` carries no `.git` directory.
- Pinned runtime planned for the cloud: Python 3.12, torch `2.14.1` CUDA 13.0,
  transformers `5.19.0`, datasets `5.1.0`, huggingface-hub `1.33.0`, rich `15.0.0`,
  psutil `7.2.2`. If a pin is unavailable or incompatible, record and commit a
  rule amendment before collecting any routing counts; do not silently substitute.
- Save integer counters for every `[layer][step][expert]`, not just aggregate
  statistics. Record SHA-256 of the new harness and imported coverage module,
  both full and first 16 hex digits. Replay and alternative contiguous splits
  must not need a model or network.

### Split and contrasts

Each of the 18 model/corpus/shape arms has the same split: select the hottest
`c = E/4` experts using steps `[0,8)`, score only `[8,16)`; then reverse.
Ties use ascending expert ID, as in the imported `hot_set`. These are contiguous
halves of the **seed-permuted measured step sequence**, not claims of document
chronology. Within an arm the halves use disjoint packed chunks; different
shapes are separate arms and can share underlying corpus tokens.

Report per layer, including layer zero, and means over that same complete
layer population:

1. **In-sample identity:** select and score the full 16-step aggregate; this
   equals its top-quarter share. Also report the training half's own in-sample
   hit for each direction.
2. **Held-out hit $h$:** the training-half hot set's mean traffic share on the
   evaluation half, with selected IDs and the individual step scores saved.
3. **Per-step oracle:** each evaluation step's own hottest `c` experts. This is
   an upper bound for the assignment hit of any `c`-expert cache on that step;
   it is not a realizable predictive policy or a latency bound.
4. **Random-set expectation:** `c/E = 0.25`, analytically, not a sampled random
   cache experiment. Absolute gain $g=h-0.25$ and relative gain $h/0.25-1$.
5. **Transfer loss:** training-half in-sample hit minus held-out hit, signed;
   an increase on the held-out half is a negative loss. Also show the worst
   layer's held-out hit as a diagnostic, not as a hidden extra gate.

### Invariants and named outcomes

Structural failure gives **VOID**: exactly 16 counter rows at every discovered
layer, `E` nonnegative integer counters each, sum exactly `tokens × top_k` at
every step, `E` divisible by four, full in-sample identity error at most
`1e-12`, and every fixed-set step hit at most its per-step oracle (same
`1e-12` arithmetic tolerance). Missing arms remain missing, not negative.

For a valid arm, use both directions' **all-layer means**. Material gain means
at least `0.10` absolute above random (`h >= 0.35`); acceptable transfer loss
means at most `0.05` absolute. Boundaries are inclusive, with only `1e-12`
roundoff tolerance. Apply these named outcomes in order:

| Outcome | Rule |
|---|---|
| **HOTSET TRANSFERS** | Both gains ≥ 0.10 and both signed transfer losses ≤ 0.05 |
| **GAIN WITH DRIFT** | Both gains ≥ 0.10, but at least one transfer loss > 0.05 |
| **DIRECTION-DEPENDENT** | Exactly one gain ≥ 0.10 |
| **NO MATERIAL GAIN** | Neither gain ≥ 0.10; does not mean no gain at all |

No claim that all corpora generalise follows from one arm. Report all 18
outcomes separately; no majority vote or post-run threshold change. Replays
with a non-half split are diagnostics, not this rule's verdict.

### Cloud safety and budget

The remaining stage budget is about $8.1. At most one instance at once, with a
Windows scheduled deadline at creation +120 minutes (`WakeToRun`,
`StartWhenAvailable`) and instance-side shutdown at creation +130 minutes,
armed immediately after the first successful `brev exec`, as in Part B run 2.
The scheduled deadline retrieves completed or partial output, verifies its
SHA-256 against the box's expected hash, then deletes; copy and verification
are joined by `&&`. A failed copy or checksum must never authorize deletion.
The shutdown is the independent fallback if copying or the workstation fails;
recover the stopped box before final deletion if necessary. Log every attempt,
including VOID arms, and finish by checking `brev ls` is empty.

Before provisioning, `logged.sh` runs inner commands with
`bash -o pipefail -c`. Intentionally copying a nonexistent path must return
nonzero through its logging pipeline, and a chained delete must not execute.
The proof is retained with the results. A stale local archive is not a
successful copy. Normal completion also deletes only after matching the
expected remote SHA-256. No adapter files or internal correspondence are part
of this measurement's result bundle.

**Preflight proof [RUN, SYNTHETIC transport].** The nonexistent copy returned
1 through `logged.sh`, including when followed by a successful `cat` in the
inner pipeline. The normal completion wrapper also returned 1 and never
called delete, even with a stale local archive whose hash matched the expected
remote hash. No cloud command ran in this control. Output:
[`copy-guard.log`](results/probe-moe-expert-holdout/copy-guard.log).
The tested external wrappers' SHA-256 values are `a524b645211ad1a9…`
(`logged.sh`) and `0af40f2713a23647…` (copy → verify → delete wrapper).

The measured wrapper discovers the SSH user's actual `$HOME` and waits up to
ten minutes for resource disappearance after verified deletion. Its SYNTHETIC
failed-copy control passed; tested SHA-256 is `66adaa595fd2b839…`, while
`logged.sh` is unchanged. Source revision and checksum transport files are
LF-only and checked before a new create.

**Provisioning-only exception, added 2026-10-08 before a replacement run.**
If provisioning fails before **any** successful `brev exec` or `brev copy`,
there can be no uploaded code or measurement output from this run. That failed
resource may be deleted without a remote checksum. Log `brev ls` immediately
before deletion, wait until the resource disappears from `brev ls`, and stop
for operator intervention if delete fails or it remains for more than ten
minutes: a FAILURE status does not prove provider-side compute has stopped
([brev-cli #384](https://github.com/brevdev/brev-cli/issues/384)).
After any successful exec or copy, the checksum gate remains mandatory.
For each replacement create, rearm the workstation task for the actual new
instance name at its new +120-minute deadline and verify `NextRunTime` and
the action's target before setup. Do not retain the first attempt's deadline.
After the failed Hyperstack attempt, try only `massedcompute_A6000_plus`,
then `g2-standard-8:nvidia-l4:1`; do not include Hyperstack again or add other
types. If all declared types fail, stop for operator choice. Failed attempts'
create-to-delete lifetimes and listed rates belong in the cost table.

**Repeat authorization, 2026-10-09; committed before another create.**
One repeat on `massedcompute_A6000_plus` is authorized. The prior A6000 exposed
the required GPU and RAM, but ended before setup and collected no counters;
it does not establish a model or capacity failure. Do not retry Hyperstack
or GCP and do not add another type. Models, precision, corpora, shapes,
16 steps, seed 17, split and verdict thresholds above are unchanged.

Before renting, run every final shell script through `bash -n` separately.
In WSL, build the archive and LF checksum/revision files from one committed
snapshot, verify its SHA, extract it into a temporary directory, and check
required sources plus a revision marker of exactly 40 hexadecimal characters.
Check all uploaded text files for CR bytes. Discover the SSH user's actual
`$HOME`, rather than fixing a username. Preserve or repeat the failed-copy
guard against the final wrapper and logger versions before any GPU time.

Refresh the advertised rate and available storage information before create.
The 2026-10-09 Brev catalog advertises **$0.684/h**, 64 GiB RAM, 12 vCPUs and
a fixed 256 GB disk for this type [RUN]; it exposes no separate storage fee
or invoice. At that rate 130 minutes is **$1.482**; allowing ten additional
minutes through deletion is **$1.596**, plus a **$2 reserve** for unitemized
charges [ESTIMATE]. This fits the approximately $7.60 compute-based remainder,
but does not assert the actual remaining balance or that storage is free.

Both deadlines stay anchored to the new original create: Windows +120 minutes
with actual target and `NextRunTime` checked, instance shutdown +130 minutes
armed on first successful exec and checked again before setup. Verify the
uploaded archive remotely before setup. After any successful exec/copy, even
an empty or partial run still requires downloaded output and the expected
remote SHA match before delete, resource disappearance, and deadline removal.
If this repeat is blocked, preserve evidence and finish safely; another
create or a change to this rule requires a new operator decision.

**Final pre-repeat copy control [RUN, SYNTHETIC transport].** The logger
records each command's exit status without changing it. On the deployed
logger (`b8fe7677f934c311…`) and copy/delete wrapper (`66adaa595fd2b839…`),
the nonexistent source returned 1 and delete was never called, including with
a matching stale local archive. No model or cloud command ran in this control.
The output is retained in `copy-guard.log`.

## 3. Results and limitations

### Held-out measurement, 2026-10-09

All **18 arms completed with 16 steps**, on one `massedcompute_A6000_plus`
RTX A6000. The unchanged rule gives **17 HOTSET TRANSFERS and 1 GAIN WITH
DRIFT** [RUN]. Granite has nine HOTSET TRANSFERS; OLMoE has eight, with prose
`1x512` the GAIN WITH DRIFT arm. There are no missing or VOID arms in this
repeat. The three earlier VOID attempts below remain provisioning/transport
failures, not negative routing measurements.

Local `replay` used only the saved counters: both replay summaries equal the
collected summaries. An independent arithmetic check recomputed the hot sets,
assignment totals, held-out hits, per-step oracles, gains and signed losses
from the integers, agreeing within `1e-12` on all 18 arms and 36 directions.
It checked all 360 layer/shape populations, including layer 0, all **5,760
step rows / 258,048 integer counters**, and non-overlapping fit/evaluation
chunk ranges. Full in-sample identity error is **0.0** throughout [RUN].
The [verification artifact](results/probe-moe-expert-holdout/holdout-verification.json)
records these checks; the [Granite replay](results/probe-moe-expert-holdout/granite-replay.json)
and [OLMoE replay](results/probe-moe-expert-holdout/olmoe-replay.json) retain
per-layer sets, per-step scores and both directions.

**Reading the tables.** A fits steps `[0,8)` and evaluates `[8,16)`; B reverses
those halves. Every entry is the mean over the same complete layer population
(24 layers for Granite, 16 for OLMoE). Random hit is **0.25 in every arm and
direction**. Gain is held-out hit minus 0.25; loss is training in-sample hit
minus held-out hit. The full in-sample column equals top25 share exactly,
with identity delta 0.0; that column is not transfer evidence. Values below
are rounded to six decimals; named outcomes use the unrounded values.

| Model | Corpus | Shape | Full in-sample = top25 | Train hit A/B | Held-out A/B | Oracle A/B | Gain A/B | Loss A/B | Outcome |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Granite | prose | 1x512 | 0.432226 | 0.434759 / 0.433589 | 0.427120 / 0.427636 | 0.460309 / 0.457588 | 0.177120 / 0.177636 | 0.007638 / 0.005952 | HOTSET TRANSFERS |
| Granite | prose | 4x512 | 0.429969 | 0.430144 / 0.430021 | 0.429576 / 0.429521 | 0.436612 / 0.437543 | 0.179576 / 0.179521 | 0.000567 / 0.000500 | HOTSET TRANSFERS |
| Granite | prose | 1x2048 | 0.434686 | 0.436931 / 0.433778 | 0.430852 / 0.433780 | 0.444008 / 0.445645 | 0.180852 / 0.183780 | 0.006079 / -0.000002 | HOTSET TRANSFERS |
| Granite | code | 1x512 | 0.508173 | 0.500952 / 0.515990 | 0.515177 / 0.499369 | 0.522744 / 0.507033 | 0.265177 / 0.249369 | -0.014225 / 0.016621 | HOTSET TRANSFERS |
| Granite | code | 4x512 | 0.509791 | 0.509200 / 0.510828 | 0.509983 / 0.508200 | 0.512075 / 0.510584 | 0.259983 / 0.258200 | -0.000783 / 0.002628 | HOTSET TRANSFERS |
| Granite | code | 1x2048 | 0.514917 | 0.505822 / 0.524815 | 0.523400 / 0.504656 | 0.529182 / 0.507491 | 0.273400 / 0.254656 | -0.017579 / 0.020158 | HOTSET TRANSFERS |
| Granite | math | 1x512 | 0.446065 | 0.444471 / 0.447929 | 0.447467 / 0.444026 | 0.451177 / 0.449174 | 0.197467 / 0.194026 | -0.002996 / 0.003904 | HOTSET TRANSFERS |
| Granite | math | 4x512 | 0.447592 | 0.446810 / 0.448392 | 0.448374 / 0.446737 | 0.449002 / 0.447476 | 0.198374 / 0.196737 | -0.001564 / 0.001655 | HOTSET TRANSFERS |
| Granite | math | 1x2048 | 0.446360 | 0.446692 / 0.446176 | 0.445930 / 0.446434 | 0.446867 / 0.447776 | 0.195930 / 0.196434 | 0.000762 / -0.000258 | HOTSET TRANSFERS |
| OLMoE | prose | 1x512 | 0.427061 | 0.446253 / 0.432335 | 0.395660 / 0.395197 | 0.502819 / 0.534824 | 0.145660 / 0.145197 | 0.050592 / 0.037138 | GAIN WITH DRIFT |
| OLMoE | prose | 4x512 | 0.415401 | 0.412684 / 0.421409 | 0.414101 / 0.407262 | 0.447837 / 0.446611 | 0.164101 / 0.157262 | -0.001417 / 0.014146 | HOTSET TRANSFERS |
| OLMoE | prose | 1x2048 | 0.413068 | 0.424578 / 0.412570 | 0.389009 / 0.400302 | 0.484521 / 0.489595 | 0.139009 / 0.150302 | 0.035569 / 0.012267 | HOTSET TRANSFERS |
| OLMoE | code | 1x512 | 0.666986 | 0.676464 / 0.659687 | 0.655087 / 0.672146 | 0.673069 / 0.693413 | 0.405087 / 0.422146 | 0.021378 / -0.012459 | HOTSET TRANSFERS |
| OLMoE | code | 4x512 | 0.663602 | 0.663642 / 0.663865 | 0.663252 / 0.663145 | 0.667232 / 0.666907 | 0.413252 / 0.413145 | 0.000391 / 0.000720 | HOTSET TRANSFERS |
| OLMoE | code | 1x2048 | 0.639050 | 0.634111 / 0.646135 | 0.642074 / 0.630634 | 0.652888 / 0.641994 | 0.392074 / 0.380634 | -0.007963 / 0.015501 | HOTSET TRANSFERS |
| OLMoE | math | 1x512 | 0.520847 | 0.518827 / 0.525482 | 0.520155 / 0.513803 | 0.534374 / 0.529474 | 0.270155 / 0.263803 | -0.001328 / 0.011679 | HOTSET TRANSFERS |
| OLMoE | math | 4x512 | 0.521500 | 0.521215 / 0.522207 | 0.521557 / 0.520352 | 0.525074 / 0.523842 | 0.271557 / 0.270352 | -0.000342 / 0.001855 | HOTSET TRANSFERS |
| OLMoE | math | 1x2048 | 0.527290 | 0.526114 / 0.528907 | 0.527698 / 0.525638 | 0.531630 / 0.529386 | 0.277698 / 0.275638 | -0.001584 / 0.003270 | HOTSET TRANSFERS |

OLMoE prose `1x512` retains material gain in both directions, but A's loss is
**0.05059242248535156**, just above the frozen 0.05 boundary. Its held-out hits
are 0.395660400390625 and 0.39519691467285156; gains are 0.145660400390625 and
0.14519691467285156. B's loss is 0.03713798522949219. The rule therefore gives
GAIN WITH DRIFT, not HOTSET TRANSFERS and not NO MATERIAL GAIN [RUN]. This is
variation between sampled data halves, not evidence of adapter-training drift.
Across all directions, held-out hit is 0.389009-0.672146, gain is
0.139009-0.422146 and signed loss is -0.017579 to 0.050592.

Worst-layer values are diagnostics, not extra verdict gates. For example,
Granite prose `1x512` passes on the all-layer mean, while layer 9's held-out
hit is below 0.35 in both directions. Relative gain is `held-out / 0.25 - 1`;
it is shown as a fraction, not percentage points.

| Model | Corpus | Shape | Worst layer A/B | Its held-out hit A/B | Relative gain A/B (all-layer mean) |
|---|---|---|---:|---:|---:|
| Granite | prose | 1x512 | 9 / 9 | 0.341919 / 0.347931 | 0.708481 / 0.710546 |
| Granite | prose | 4x512 | 9 / 9 | 0.357193 / 0.355186 | 0.718305 / 0.718085 |
| Granite | prose | 1x2048 | 9 / 20 | 0.361427 / 0.365570 | 0.723408 / 0.735120 |
| Granite | code | 1x512 | 0 / 0 | 0.418945 / 0.405365 | 1.060710 / 0.997477 |
| Granite | code | 4x512 | 0 / 0 | 0.411369 / 0.411034 | 1.039932 / 1.032800 |
| Granite | code | 1x2048 | 0 / 0 | 0.410416 / 0.403221 | 1.093601 / 1.018626 |
| Granite | math | 1x512 | 18 / 18 | 0.359192 / 0.356567 | 0.789866 / 0.776103 |
| Granite | math | 4x512 | 18 / 18 | 0.362335 / 0.360023 | 0.793498 / 0.786949 |
| Granite | math | 1x2048 | 18 / 18 | 0.360901 / 0.361069 | 0.783718 / 0.785737 |
| OLMoE | prose | 1x512 | 5 / 5 | 0.370880 / 0.362610 | 0.582642 / 0.580788 |
| OLMoE | prose | 4x512 | 1 / 3 | 0.386925 / 0.386063 | 0.656403 / 0.629049 |
| OLMoE | prose | 1x2048 | 7 / 7 | 0.363472 / 0.380592 | 0.556036 / 0.601210 |
| OLMoE | code | 1x512 | 0 / 0 | 0.477234 / 0.480316 | 1.620346 / 1.688583 |
| OLMoE | code | 4x512 | 0 / 0 | 0.482048 / 0.476479 | 1.653008 / 1.652578 |
| OLMoE | code | 1x2048 | 0 / 0 | 0.453354 / 0.454956 | 1.568295 / 1.522535 |
| OLMoE | math | 1x512 | 1 / 1 | 0.423676 / 0.414001 | 1.080620 / 1.055214 |
| OLMoE | math | 4x512 | 1 / 1 | 0.421234 / 0.420082 | 1.086229 / 1.081409 |
| OLMoE | math | 1x2048 | 1 / 1 | 0.424240 / 0.423523 | 1.110792 / 1.102551 |

### Capture provenance and verified teardown

The measured archive is commit **`011309792e989fcdb37cdf12eb7e8dc661e62ebd`**,
which committed the repeat authorization before create; the statistical rule
was already committed in `be3d23b2`. Local WSL preflight checked each of the
12 final shell files separately with `bash -n`, LF-only uploaded files,
`sha256sum -c`, extraction, required paths, the exact 40-hex source marker,
and both harnesses against their committed blobs [RUN]. See
[`preflight.log`](results/probe-moe-expert-holdout/preflight.log) and the final
matching-stale-archive failed-copy control in
[`copy-guard.log`](results/probe-moe-expert-holdout/copy-guard.log).

- Uploaded source archive SHA-256:
  `06fa17b23ebef9e7e11afd29f249c576221d2b57b5d685ccd02dee4a23853680`.
  The remote checksum passed before setup.
- Collector SHA-256:
  `366ace0102ef73b18da4c3cfaa7912d25a830f4103f85ae0cc5e477b5e4c9f9a`.
- Imported coverage harness SHA-256 in the committed LF archive:
  `728c9720aa2bb60358d41ed10286c0571304654033889d9fcf2c75ab73365f7f`.
  This differs from the untouched Windows CRLF file's `dc00b5dcd5746d05`
  prefix only because of line endings. The preflight's Git-blob comparison
  proves source equivalence; local replay retains the original collection
  fingerprints separately from its Windows fingerprints. Neither original
  coverage file was edited.
- Actual model, tokenizer and config revisions equal the requested immutable
  pins in section 2 for both models. Actual WikiText and GSM8K revisions also
  equal their requested pins. Code is **real Soup source**, not SYNTHETIC code,
  at the measured archive commit. Accepted documents: prose 4000, code 541,
  math 4000; respective corpus SHA-256 prefixes `399831ef40995321`,
  `c00ceccb47511dff`, `21b099a4d0676f55`, identical across models. Full corpus,
  packed-token, consumed-token and per-step input digests are in each capture.
- Actual runtime: Python **3.12.15**, torch **2.14.1+cu130**, CUDA **13.0**,
  transformers **5.19.0**, datasets **5.1.0**, huggingface-hub **1.33.0**,
  numpy **2.5.3**, rich **15.0.0**, psutil **7.2.2**. Eager attention and expert
  execution, bfloat16, no quantisation, no adapter, `use_cache=False`.
  [`runtime-freeze.txt`](results/probe-moe-expert-holdout/runtime-freeze.txt)
  lists the installed packages. The host reported 12 CPUs, about 62 GiB RAM,
  49,140 MiB GPU memory, driver 580.126.09 and a 251 GiB root partition.
- Granite collected at 02:34:49-02:36:27 UTC; OLMoE at 02:36:29-02:38:48 UTC.
  Both collector exits and remote setup/measurement exit were **0**.
  [`collection-commands.log`](results/probe-moe-expert-holdout/collection-commands.log)
  preserves the deployed collector script and flags;
  [`setup-commands.log`](results/probe-moe-expert-holdout/setup-commands.log)
  preserves the deployed setup script;
  [`setup evidence`](results/probe-moe-expert-holdout/setup.evidence.log),
  [`Granite log`](results/probe-moe-expert-holdout/granite-bf16-cuda.log),
  [`OLMoE log`](results/probe-moe-expert-holdout/olmoe-bf16-cuda.log) and
  [`run-status.txt`](results/probe-moe-expert-holdout/run-status.txt) retain
  actual output and exit codes.

The controller's original creation anchor was **02:27:30 UTC**, 4.29 seconds
before the logged create command. The Windows deadline was registered before
create and reverified before setup at **04:27:30 UTC** (09:27:30 +05:00), with
the actual instance `moe-holdout-a6000` in the action and wake/start-when-available
enabled. On first successful exec the server armed shutdown relative to that
same anchor, not to SSH readiness: planned +130 minutes, **04:37:30 UTC**;
actual scheduled shutdown **04:36:38.106436 UTC**, within the one-minute
round-down allowance. Both deadlines and the remote source archive were
verified before setup [RUN].

The output archive's expected remote SHA-256 was
`dd5c491a2e996324bab0f45fe60c168e730acd7b711afdb8d39a6680b4625c7c`.
The copy succeeded and its local SHA matched **before delete**. `brev ls`
then showed DELETING twice and **no instances**; absence was logged at
**02:40:37 UTC**. The Windows task was removed and a fresh local query at
02:41:24 UTC reported `HOLDOUT-DEADLINE-ABSENT` [RUN]. No provisioning-only
exception was used. See
[`cloud evidence`](results/probe-moe-expert-holdout/cloud-repeat.evidence.log).

Raw capture files were copied byte-for-byte and never hand-edited:

| Raw capture | SHA-256 |
|---|---|
| [Granite](results/probe-moe-expert-holdout/granite-bf16-cuda.json) | `03efbc13845902e55b4d5a3da95e9f3f7ff6b9828373b1a86a0782f50de6b528` |
| [OLMoE](results/probe-moe-expert-holdout/olmoe-bf16-cuda.json) | `0f81247afc7f6796ab1645bc83e44d7a163581a62c30bb3f48a84c1676c89f5a` |

### Interpretation boundary

This answers only **base-profile transfer between these seed-permuted halves
of the same corpus** for two real small MoE checkpoints and long forward
passes. It does not establish natural-order stationarity, cross-corpus
transfer, repeat-seed stability, adapter-induced routing drift, autoregressive
batch-1 decode, DeepSeek-V3/K2 routing, NF4/BF16 equivalence, a cache capacity
other than one quarter, SSD bytes saved or serving throughput. There is no
adapter, SSD-offload implementation or serving-speed measurement in this run.
The one drift outcome prevents a blanket claim that every measured profile
transfers within the loss limit. The positive gains support measuring a base
hot-set policy; they do not replace P6's real-adapter and decode checks.

### Attempt costs and retained VOID evidence

The Hyperstack resource failed during provisioning. `create` ended with
`instance ... failed`; the first exec was refused at FAILURE / SHELL NOT
READY. No exec or copy succeeded, no setup ran and no counters were collected
[RUN]. It was removed under the provisioning-only exception, not under the
normal verified-output deletion path. `brev ls` was logged before deletion
and then showed DELETING, DELETING, STARTING, and finally no instances at
16:54:46 UTC. The transition through STARTING is why a successful delete
command alone was not taken as cleanup proof. Source-lined lifecycle evidence:
[`cloud-run.evidence.log`](results/probe-moe-expert-holdout/cloud-run.evidence.log).

The initial MassedCompute A6000 exposed 49,140 MiB GPU memory, driver
580.126.09 / CUDA 13.0, 12 CPUs, about 62 GiB RAM and a 251 GiB root partition
[RUN]. This is a disk observation, not proof of total provisioned storage or
its price. The Windows deadline targeted the actual instance at
23:59:53 +05:00 (18:59:53 UTC); the server-side deadline was 19:08:56 UTC,
within the original creation +130 minutes.

This attempt ended at transport validation, **before setup**. The saved
bundle has an empty results directory and empty setup log: no model,
dataset or routing counters were collected [RUN]. Because exec and uploads
had succeeded, this resource used the normal gate: remote archive checksum
`6a906a75437d7f424b45fcc9d7b703a22ee10ffbf2f54a03921eea9687c1258b`,
successful copy, matching local checksum, then delete. It disappeared after
DELETING, DELETING and DEPLOYING; no absence was inferred from delete alone.

The GCP L4 reached Brev's reported Ready status, but the first exec exhausted
20 SSH-readiness attempts with a port-22 connection timeout. No exec or copy
succeeded and instance-side shutdown could not be armed. Its Windows
deadline was freshly registered for 00:24:32 +05:00 on 2026-10-09.
It was removed under the provisioning-only exception; the pre-delete
`brev ls` still said RUNNING / COMPLETED / READY. After delete it showed
RUNNING, DELETING, STOPPING, then no instances. All resources were confirmed
absent by 17:38:00 UTC and the workstation deadline was removed [RUN].
All three attempts are **VOID before routing counters**, not negative
holdout outcomes. The subsequent repeat used only the authorized MassedCompute
type and unchanged rule; no undeclared machine type was substituted.

| Attempt, 2026-10-08 UTC | Outcome | Create command → delete command; confirmed absent | Listed rate | Compute estimate |
|---|---|---|---:|---:|
| `hyperstack_A6000` | VOID, provisioning before shell | 16:41:57 → 16:53:22 (11m25s); absent 16:54:46 (12m49s total) | $0.60/h | $0.114 to delete; **$0.128 conservatively through confirmed absence** |
| `massedcompute_A6000_plus` | VOID, transport validation before setup | 17:00:05 → 17:13:42 (13m37s); absent 17:15:15 (15m10s total) | $0.68/h | $0.154 to delete; **$0.172 through confirmed absence** |
| `g2-standard-8:nvidia-l4:1` | VOID, SSH unreachable before any exec/copy | 17:24:34 → 17:36:34 (12m00s); absent 17:38:00 (13m26s total) | $1.02/h | $0.204 to delete; **$0.228 through confirmed absence** |

These are list-price estimates, not an invoice or evidence that provider
billing stopped at the delete command. Every failed lifetime is included
even though no measurement ran. Historical rates above retain the price
quotes used for those attempts.

| Successful repeat, 2026-10-09 UTC | Create command → delete command; confirmed absent | Listed rate | Compute estimate |
|---|---|---:|---:|
| `massedcompute_A6000_plus`, all 18 arms | 02:27:34.291 → 02:39:31.521 (11m57.23s); absent confirmed 02:40:37.972 (13m03.68s total) | $0.684/h | $0.136274 to delete; **$0.148899 conservatively through confirmed absence** |

The fresh [catalog](results/probe-moe-expert-holdout/price-catalog-2026-10-09.json)
reported fixed minimum/maximum/target disk **256 GB**. The 251 GiB root
partition was observed on the box, not inferred from the catalog. No separate
storage rate or final invoice was exposed; the preregistered $2 reserve
was a budget margin, not observed spending.

The first three conservative compute estimates total **$0.528343**. Adding
the successful repeat gives **$0.677243** for this holdout work; with the prior
serving-engine probe's approximately $1.87, stage compute at this capture's
close was **$2.547243 ($2.55)** [ESTIMATE]. The later
[registered real-Qwen repeat](probe-lora-hook-engines.md#65-real-model-repeat-with-complete-controls)
brings stage compute to **~$4.08**, leaving **~$5.92 before non-compute charges**
against the $10 ceiling. Separate storage charges remain unobserved;
these figures are not an invoice.

### Repository verification

The historical pre-fix Windows/Python 3.12.10 invocation ran
`pytest tests/ -n logical --dist loadfile` without `-x`, with shared
`PYTHONHASHSEED=0` and the repository's default non-smoke selection:
**34,159 passed, 11 failed, 428 skipped, 2 xfailed**, 1,195.63 s [RUN].
Coverage was **86.30% (75,667 / 87,676 lines)**, above the 77% gate; that suite
invocation did not pass.

Five lazy callback-builder subprocesses exceeded 60 s and `doctor` exceeded
120 s. Two existing performance assertions missed their limits (2.4698 s
against 2 s; boxed-scan growth 4.4716 against 3); the local Langfuse `/ok`
control missed its 0.4 s read deadline. All nine passed in a sequential
repeat. The tenth repeated case, plugin discovery, still failed: Windows
`cp1251` decoding rejected byte `0x98` in the subprocess reader, leaving
`stdout=None`; the same test passed with `PYTHONUTF8=1`. The eleventh
full-run failure was the existing transformers 5.19.0 / TRL 0.29.1
`PPOTrainer.is_distributed_loading_by_transformers` checkpoint error.
Controlled baseline comparison reproduced the PPO and plugin failures on
both baseline and track, with the other nine passing in both arms.
The publication branch restores the PPO policy's load-time sharding flag
for native checkpoint saving and decodes the plugin subprocess as UTF-8.
The same eleven cases subsequently passed, including after the benchmark
tests in one invocation; see the [verification record](probe-lora-hook-engines.md#64-real-model-repeat-preregistration).

The sequential repeat was **9 passed, 1 failed**, 194.34 s; the UTF-8 check
was **1 passed**, 7.70 s. Neither replaces the full-run counts or coverage.
Targeted holdout/LoRA regression checks were **77 passed**, with model-free
audit and default/custom-split replay CLI smoke. Retained verification evidence:
[`full xdist failures and summary`](results/probe-moe-expert-holdout/pytest-full-xdist.evidence.log),
[`sequential`](results/probe-moe-expert-holdout/pytest-sequential.log),
[`UTF-8 check`](results/probe-moe-expert-holdout/pytest-plugin-utf8.log).
The [artifact manifest](results/probe-lora-hook-engines/artifact-provenance.json)
records source and published hashes. Routing-result JSONs and the original
coverage record/harness remain byte-identical; log extracts identify their
source captures and selected line ranges.

Fresh `replay` CLI runs reproduce both complete routing summaries exactly.
A fresh `audit-published` CLI run reproduces all four captures' numerical
results; only its recorded holdout-harness fingerprint differs from the
earlier audit. The [comparison record](results/probe-moe-expert-holdout/replay-final-verification.json)
retains input/output identities and every difference. No raw routing capture
or historical verdict was changed.

The later full-suite and test-isolation results are recorded in
[§6.6 of the adapter probe](probe-lora-hook-engines.md#66-repository-verification-and-test-isolation);
they are separate from the historical counts above.

## 4. Reproducing

Run from the repository root. `collect` uses the pinned real checkpoints;
`replay` and `audit-published` need no torch, model weights or network.

```bash
python benchmarks/harness/moe_expert_holdout.py audit-published \
  benchmarks/results/probe-rtx5070/moe/granite_bf16_cuda.json \
  benchmarks/results/probe-rtx5070/moe/granite_bf16_cuda_run2.json \
  benchmarks/results/probe-rtx5070/moe/olmoe_nf4_cuda.json \
  benchmarks/results/probe-rtx5070/moe/olmoe_bf16_cpu_control.json \
  --out benchmarks/results/probe-moe-expert-holdout/published-audit.json

python benchmarks/harness/moe_expert_holdout.py collect \
  --model ibm-granite/granite-3.0-1b-a400m-base \
  --revision d91cbed802d85eb1b32374623331f6ab2b37403a \
  --source-revision <source-commit> \
  --data hf:Salesforce/wikitext:wikitext-2-raw-v1:train:text --data-label prose \
  --data src/soup_cli --data-label code \
  --data hf:openai/gsm8k:main:train:question --data-label math \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --dataset-revision 740312add88f781978c0658806c59bc2815b9866 \
  --shape 1x512 --shape 4x512 --shape 1x2048 \
  --batches 16 --seed 17 --limit-rows 4000 --device cuda --dtype bfloat16 \
  --out benchmarks/results/probe-moe-expert-holdout/granite-bf16-cuda.json

# Same data, shapes and flags for OLMoE; no --load-4bit:
# --model allenai/OLMoE-1B-7B-0924
# --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5
# --out benchmarks/results/probe-moe-expert-holdout/olmoe-bf16-cuda.json

python benchmarks/harness/moe_expert_holdout.py replay \
  benchmarks/results/probe-moe-expert-holdout/granite-bf16-cuda.json \
  --out <replayed-summary.json>
# Add --split <index> for a diagnostic alternative contiguous split.

python -m pytest tests/test_moe_expert_holdout.py -q --no-cov
```
