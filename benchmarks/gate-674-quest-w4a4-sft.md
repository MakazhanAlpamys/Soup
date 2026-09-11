# QuEST W4A4 SFT: gate failed (#674)

**Gate failed: 0.114 nat [0.099, 0.130], one training seed, one model, fake
quantization. Not parity, not an integration, not an efficiency claim.**

Measured by [@Shutaru](https://github.com/Shutaru). The main experiment completed
on 2026-09-09; the last bounded diagnostic completed on 2026-09-11. This record
expands the [research update](https://github.com/MakazhanAlpamys/Soup/issues/674#issuecomment-5636030904)
with the protocol and provenance requested in the
[maintainer's reply](https://github.com/MakazhanAlpamys/Soup/issues/674#issuecomment-5636239673).
It leaves [#674](https://github.com/MakazhanAlpamys/Soup/issues/674) open.

## Result on the same held-out panel

All eleven rows use the same 192 Dolly examples and 7,819 supervised response
targets (CHECK192). Lower response-target negative log-likelihood (NLL) is better.
`RDxx` names identify retained local experiments, not Soup releases.

| Model / retained endpoint | CHECK NLL, nat per target |
| --- | ---: |
| Original base | 2.619283 |
| FP02, the common parent | 2.246091 |
| RD13 old-data FP | 2.247933 |
| RD13 fresh-data FP, strongest retained FP | 2.235367 |
| RD15 FP control | 2.236810 |
| Earlier calibrated/distilled W4A4 (Recipe06) | 2.377740 |
| Full-width W4A4 initialization, without group128 | 2.520581 |
| Group128 W4A4 initialization | 2.514601 |
| Full-width W4A4 terminal control, without group128 | 2.357655 |
| Group128 W4A4 terminal candidate | 2.349768 |
| Concurrent FP terminal control | 2.236143 |

The predeclared group128 candidate has NLL **2.349768174976808**, against
**2.235367338223475** for the strongest retained FP on these same rows. The gap
is **0.11440083675333268** nat, with paired 95% interval
**[0.0991910814834233, 0.12992046340524593]**.

Our experimental quality gate required both the gap and its 95% upper bound to
be at most 0.1 nat. Both fail. The 0.1 number was our preregistered experimental
criterion, not a numerical requirement imposed by the issue author.

Group128 improved over Recipe06 by 0.027972 nat; the signed candidate-minus-control
interval is [-0.042689, -0.013470]. Against the non-group128 terminal control,
the difference is -0.007887 nat with interval [-0.023544, 0.007761]. That interval
crosses zero, so the separate requirement for a demonstrated improvement over
that control also fails. No alternate arm or initialization was promoted.

Intervals use 2,000 paired sequence bootstrap resamples, seed 17. Each resample
recomputes the token-weighted loss difference; tokens within a sequence are not
sampled independently. They are conditional on the observed sample and one
training seed, not estimates of training-seed variation. The FP comparator was
chosen as the lowest CHECK NLL among the predeclared retained FP endpoints; the
reported interval is for that selected pair, not a selection-adjusted confidence
statement. Historical training compute across those endpoints is not equal.

## Instrument card

| Item | Recorded setup / provenance |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB reported VRAM (24 GB class); one local GPU for RD21 |
| NVIDIA driver | **Not captured in the historical RD21 records.** A documentation-time query on 2026-09-11 returned `616.64`; this is not evidence of the driver used on 2026-09-09 |
| OS / Python | Windows; Python `3.12.10`, as recorded in the original feasibility environment |
| PyTorch / CUDA | `2.11.0+cu128` / `12.8`, also recorded in each RD21 fit; CUDA here is the PyTorch runtime/build version, not the driver's advertised maximum |
| Transformers | `5.16.1` in the original feasibility environment |
| Model and tokenizer | `ahxt/LiteLlama-460M-1T`, revision `77b8a976440e7d1ea5a890eaf1e0175b1cac0078` |
| Dataset | `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, file `databricks-dolly-15k.jsonl` |
| Research checkout | Soup base `64d029e`; local experimental scripts identified below, not a released backend |

The missing historical driver is a provenance limitation. It has not been filled
in from today's installation. The data in this record do not establish behavior
on another GPU, software stack or model.

RD21 recorded deterministic algorithms enabled with errors rather than warnings,
cuDNN deterministic enabled, cuDNN benchmark disabled, TF32 disabled, BF16/FP16
reduced-precision reduction disabled, and one CPU thread. Its
`CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED` fields were unset. These are recorded
settings, not a guarantee of cross-stack bit-exactness.

## Fixed protocol

The scientific protocol was registered as RD18 before execution; RD21 completed
it after operational recovery, without changing the recipe or decision rules.

### Format and training

This is an **experimental QuEST variant**. Both operands receive a full-input-width
Hadamard transform (width 1,024 or 4,096), then contiguous groups of 128 for RMS
scaling and 4-bit quantization. Activation clip scales come from the fixed
TRAIN-calibrated Recipe06 table, reused without recalibration. The full-row
control uses the same transform and quantization arithmetic without group128.

All 168 transformer-block linears remain W4A4. Embeddings, normalization,
attention operations and the output head keep their declared exemptions. Master
parameters are FP32, with BF16 matrix execution and fake quantization.
Quantization intermediates retain the incoming rotated dtype under autocast;
this is not an entirely FP32-grid computation. It is not an unmodified upstream
QuEST result, packed INT4 storage, an INT4 inference kernel, or `soup train`
integration. There is no extra full-precision residual or inference ensemble.

The native W4A4, group128 W4A4 and concurrent FP arms each start from an
independent copy of the same retained FP02 checkpoint. An independent frozen
FP02 copy is the teacher. Each arm uses a fresh AdamW optimizer and:

- 1,024 fixed training examples, batch size 2, four passes in the saved order
  without shuffling: exactly 2,048 updates and 169,460 response targets per arm.
- Peak learning rate `5e-6`, cosine decay, 205 warmup updates, AdamW beta2 `0.95`,
  and zero weight decay.
- Objective `0.25 * response CE + 0.75 * KL(teacher || student)`, temperature 1,
  normalized on the same supervised response positions for every arm.
- Python, NumPy and Torch seed 42. No intermediate CHECK evaluation, quality-based
  early stopping or post-result extension.

Each student processed 343,776 unpadded input tokens over the full fit; the
teacher processed the same number. With batch padding this was 444,216 input
tokens per model, separate from the 169,460 supervised response targets. These
are short SFT runs, not pretraining-scale token budgets.

### Tokenization and measurement

The pinned GPT-2 tokenizer uses `<|endoftext|>` (ID 50256) explicitly for BOS, EOS
and padding to resolve conflicting special-token metadata in the model repo.
Prompts contain `### Instruction:`, optional `### Context:`, and `### Response:`.
Rows with empty instruction/response or a prompt longer than 192 tokens are
excluded. Responses are encoded with a leading space, capped at 64 tokens, and
receive EOS only when shorter than that cap. Total length is at most 256.
Prompt and padding labels are `-100`; only response targets contribute to loss.

The primary NLL is the model's batch-size-two loss, weighted by supervised target
count. Separate per-target FP32 cross-entropy values are summed in FP64 by
sequence for bootstrap uncertainty. Agreement with the primary metric is checked
within `5e-6` nat; this numerical check does not relax the 0.1 quality gate.
Rounded table entries must not be used to reconstruct the full-precision gap.

After all fits and checkpoint replays passed, the eleven fixed endpoints above
were evaluated once on CHECK192. The strongest retained FP had to improve over
base, as did the concurrent FP. Group128 also had to improve over both Recipe06
and the non-group128 control by at least 0.005 nat, with each paired interval's
upper bound below zero, in addition to the FP-gap gate. Engineering checks passed;
the full acceptance conjunction failed.

### Checkpoints

All three two-update smokes and their fresh-process replays passed before full
fits. The full fits each saved the pre-probe weights, optimizer and format
metadata. Fresh-process reload checks matched native logits, followed by a
controlled positive update at learning rate `5e-7` that changed the parameters
and reproduced the objective, parameter digest and optimizer state. The teacher
remained unchanged. Probe-mutated copies did not enter CHECK evaluation.

These checks cover the recorded inputs and explicit experimental wrapper
reconstruction. They do not establish a public tokenizer/generation package or
a complete Soup save/load API contract.

## Panel-spending history

Panels are identified by their manifests, not merely by their sizes. Several
different studies used 192 examples; they are not interchangeable.

| Panel | Exposure and current status |
| --- | --- |
| Original pilot validation (32) and four-epoch test (64) | Evaluated in the early failed recipes; spent, excluded from later fresh panels |
| Original R&D TRAIN1024 / DEV128 | TRAIN used for fitting/calibration and DEV repeatedly used for development; neither is a fresh test |
| RD08 FIT64 / ID128 / SHIFT64 | Used for the response-objective rotation study; its evaluation panels are spent |
| RD11 CHECK192, 7,399 targets | Evaluated for global calibration; spent |
| RD13 CHECK192, 6,612 targets | Evaluated for the diversity study; spent |
| RD15 FIT1024 / CHECK192, 7,323 CHECK targets | FIT reused for RD18/RD21 and later TRAIN diagnostics; CHECK evaluated and spent |
| RD18/RD21 CHECK192, 7,819 targets | Fresh for the frozen RD21 experiment; now spent, including when the candidate failed |
| Later TRAIN windows, including RD54/RD55 | Previously exposed fitting data; diagnostic evidence only, not new generalization results |
| Reserved FINAL256 | **Never evaluated.** Still reserved for a single final confirmation; the failed candidate did not proceed to it |

The original R&D split used zero-based eligible-row indices `[32:160]` plus
`[224:1120]` for TRAIN, `[1120:1248]` for DEV, and `[1248:1504]` for FINAL256.
Eligible rows were sorted by normalized row hash after normalized-instruction
and document-group deduplication and the fixed encoding filters.

RD21 reused RD15's exact FIT1024 (42,365 targets per pass). Its new CHECK took
the first 192 eligible rows from the fixed 8,192-row candidate prefix after
excluding every row, normalized instruction and document-group identity in the
original, RD08, RD11, RD13 and RD15 manifests, including their fitting and final
partitions. This is identity-based exclusion, not a claim of semantic or
pretraining decontamination. Original DEV and FINAL256 had no RD21 evaluation
path. No panel was reselected based on its loss.

## Failed directions retained

Different rows, formats and budgets were used across these studies. Their
absolute NLLs must not be combined into a leaderboard or subtracted from RD21.

The original 128-example one-epoch pilot failed all three seed pairs. Repeating
the same 128 examples for four epochs also failed, and its FP control became
worse than base on that study's separate test. More data, clipping, distillation
and grouping led to the RD21 candidate, but not parity.

Local reconstruction screens tried paired Givens rotations, response-gradient-
weighted scales, teacher/student clipping tables and correlated rounding. Some
reduced local reconstruction error without the required response-NLL gain. One
rotation reduced local error by about 66% while its response-NLL point estimate
increased by 0.0035 nat. Four representative screens received fresh replays;
that is not a replay claim for all sixteen screens. A subsequent single-layer,
64-step response-objective rotation study did not establish a gain on its
separate evaluation panel; it does not rule out rotation methods generally.

Gradient-direction interventions did not pass their fixed finite-step checks.
One later replay timed out during final restoration. Eighteen matching partial
trial records were retained, not counted as a completed replay.

RD54 output calibration had exact first/replay results across six controls.
On TRAIN rows 32:64 (1,218 targets), Q improved by 0.007951 nat but FP improved
by 0.023545, widening the fair gap from 0.251284 to 0.266877. Both TRAIN windows
failed the continuation filter. A CPU FP32 cross-entropy discrepancy found
during this work was isolated and corrected scoring checked against fresh GPU
loss. That was a measurement correction, not a model improvement; the earlier
failed run was not relabelled as successful.

RD55 tested direct W4A4 conversion of the later concurrent FP checkpoint, without
recovery training. Its first run and fresh replay matched. The new initialization
still lost to the best retained trained Q endpoint on both TRAIN windows, failing
the predeclared investment filter of at least 0.01 nat improvement on both.
No recovery run followed. However, a retrospective, same-row comparison on
32 TRAIN examples showed that it improved over the **old untrained initialization**
by 0.403683 nat. Failing the conservative trained-endpoint comparison does not
prove that the new starting point lacks value or that recovery would fail.

## Costs and limits

Matched new 2,048-update runs, including fit/save/probe work:

| Arm | Elapsed seconds | Peak allocated GPU bytes | GiB (bytes / 2^30) |
| --- | ---: | ---: | ---: |
| Concurrent FP | 495.062 | 10089582592 | 9.397 |
| Full-width W4A4 without group128 | 1077.563 | 13320890880 | 12.406 |
| Group128 W4A4 | 1150.453 | 14721952768 | 13.711 |

The reference QAT implementation is slower and uses more memory. These are
descriptive desktop timings, not controlled INT4-kernel or deployment benchmarks.
PyTorch allocated-memory peaks are not total device usage and exclude driver
allocations. Historical training of the parent and retained controls is not
included in this matched new-run cost table.

No application-level utility, cross-model behavior, genuine three-seed
confirmation or irreducible quantization floor has been established. Changing
the outer seed alone would be inadequate: the retained loader resets Torch to
42 and reuses fixed parent weights and batch order. A future seed study must
define real paired training variation before claiming seeds 42/7/123.

## Provenance and reproduction boundary

The scripts, protocols and full records remain local. This contribution publishes
descriptions and SHA-256 bindings, as requested, rather than a runnable integration.
It contains no raw corpus rows, token IDs, checkpoints or credentials. Hashes
identify retained bytes; they do not make unavailable files independently
reproducible or prove when a protocol was registered. CI for this documentation
change does not rerun the GPU experiments.

Before publication, 55 retained source/evidence/manifest bindings from the RD21
audit were rechecked, including the installed Torch gradient-clipping source.
The saved eleven-model evaluation retains per-row target counts and NLL sums.
The audit binds those results to the sources and fit/replay records. Principal
file bindings are below; paths are relative to the local research collection,
not links to files shipped by this PR.

<details>
<summary>SHA-256 file bindings</summary>

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `RD18_GROUP_RETRAIN_PROTOCOL.md` | Fixed science, panels, budget and gates | `1e5fb278328a15e38576a66fa9a8c4c80b3af8e21ea189b5f0dca15ee21faa9c` |
| `RD21_SCHEDULED_RECOVERY_PROTOCOL.md` | Operational recovery without a recipe change | `d1660457329d6e4ab3268e96e69bedd408ff45822d6dffcf0cdae55f7c9d8ee2` |
| `quest_sft_feasibility.py` | Pinned assets, tokenizer and response-only encoding | `5a623830c8c9e5213559cc37f986dcee2aa8bcfc6f4e59480888bbd92a1d5d56` |
| `quest_rd_group_data.py` | Panel construction and exclusion checks | `511f67b7c95139d0038dda82d294a28efb6e54958c7dc8f3c831350114c48ccd` |
| `quest_rd_group_train_ops.py` | Full-width/group128 operators and coverage | `e2e87352b69c72b0279f6cf4282539d39ce9af4b695a97fdb921425b02788b29` |
| `quest_rd_group_run.py` | Fixed fit, evaluation and replay entry points | `724287317edf5b675ec7b0a7119cf9eef5771ab3452523afc6d7d33110a327d5` |
| `quest_rd_group_scheduled.py` | RD21 execution wrapper | `b5b9c63f6aa9636da78a1a69122fac62e3864d47214cbf59c19ee7a7f410a898` |
| `quest_rd_measurement.py` | Primary NLL and per-sequence decomposition | `ebe9d279b9b08843f1c763d0c997eaf7778ca66d26e197a6928b2dd7a55748f1` |
| `quest_rd_statistics.py` | Paired bootstrap | `5b7928cfba54d1bd3d481fa2484d39323e9a75d1e4d9e9a4ebc533019b064655` |
| `quest_rd_group_audit.py` | Independent artifact and decision checks | `632199693d10ba4d1b51f25bb8920139a6457a25363b081c7005a92ad5d15832` |
| `quest_rd_group_scheduled_audit.py` | RD21 audit entry point | `734f16520c9aee50c4d6476a450711481259a71bff844e49e9252003853cd677` |
| `rd18-panels.json` | FIT/CHECK manifest and upstream panel bindings | `f62d600e9c1fc5dc3c1a1b0290dc2f9e2bf79f3a1419a9060771ae1b2638d9e3` |
| `rd-clipping-calibration.calibration.json` | Frozen TRAIN-derived activation clips | `2670fe8bfece03a06fcf9fd3d80fe72437dcd3f3da8c101a5c9959d332c5670d` |
| `rd21-screen-01.native.json` | Non-group128 full-fit record | `009a6a91930179749c0c55ef28bd9d10cc34d0e0d9aadf3561feecf37db0dfa2` |
| `rd21-screen-01.native.replay.json` | Fresh-process replay | `d00535ff22bdf3a5068c7c89f7eb5ebdd99594df94f45030f6afbf922e2ea0bf` |
| `rd21-screen-01.group128.json` | Candidate full-fit record | `f8b79f538df91c940dbb5a33c381c87684b64615bd161232b87e9cfbe87a888f` |
| `rd21-screen-01.group128.replay.json` | Candidate fresh-process replay | `824a3c72823472d3b4d367c5afd0b56666953382055d6d70d494ccf698ff477f` |
| `rd21-screen-01.fp.json` | Concurrent FP full-fit record | `9c9de60c4dc418ce3914d5dd358397eb94577b2ff73ff19789f1c12df65416f2` |
| `rd21-screen-01.fp.replay.json` | Concurrent FP fresh-process replay | `58564974aa417f1fb8e92e1bb3eb340b8d7c98fb1b26255a033f15c738aad0ab` |
| `rd21-screen-01.evaluation.json` | All eleven same-panel measurements | `2c6e73b1e0302739d5dea74a39a5044e4aff7c83e586a89a5b9df53b9106a407` |
| `rd21-audit-01.json` | Completed audit and failed acceptance decision | `46345c958fe5d6e2320f5e5170bdee8342d445419e849a92930f55ce618a6e4a` |
| `rd55-initialization-01.first.json` | Last bounded initialization diagnostic | `401255f7e4eb2a2a86974633400f240d6b75d4d7dfd612d50ba9e0d0847f64bf` |
| `rd55-initialization-01.replay.json` | Complete scientific replay of that diagnostic | `2b562784f0a43dfe2c4c74a48034205128bad13014859449eb86a7b40b8b9719` |

</details>

## Status and possible follow-up

Larger sweeps are paused. The maintainer suggested weight-only/activation-only
and per-block ablations, evaluation-time dynamic clipping, gradient comparisons,
and budget/stronger-FP-initialization recovery probes. These are possible new
experiments, not results or commitments in this record. Older recipe ablations
do not settle those questions for this candidate. A flat budget curve would not
by itself prove an inherent method limit, nor would high gradient cosine rule
out every surrogate-gradient problem.

Any follow-up needs a separately predeclared small development panel and a
bounded decision rule. The spent CHECK192 cannot validate a changed recipe.
FINAL256 stays reserved for one locked confirmation with genuine seeds 42/7/123.
The record can stand as a negative result without another training run.
