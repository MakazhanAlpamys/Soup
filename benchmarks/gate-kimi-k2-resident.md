# Gate: the resident Kimi K2 route reproduces K2's own pinned remote code

Status: **PASS** (2026-11-12, CPU only). Sections 1-3 are the rule, committed before the
run; sections 4-6 are filled from the run. No GPU, no real K2 weights (a seeded tiny
K2-shaped checkpoint stands in, as the admission gate's G2 does).

## 1. Question

#2104 lifts Soup's resident (non-streamed) route to load Kimi-K2-Base and -Instruct
natively: the raw `config.json` goes through the field-by-field guard into transformers'
native `DeepseekV3Config`, the safetensors are read tensor by tensor with each FP8 block
weight dequantised against its float32 `_scale_inv` (the admission gate's G2 pinned that
dequantisation bit-exact against an independent per-block reference), and the state dict is
assigned onto a meta skeleton. The admission gate (#2107) proved the *streamed* model
bit-exact against a *resident native model on the same weights* — but both sides of that
comparison were transformers' own classes. This gate closes the remaining loop: does the
resident route compute what **K2's own published modelling code** computes on the same
weights? That is the acceptance comparison named in the issue.

K2's remote code does not import or run on modern transformers (the admission gate traced
this on 5.17), and Soup never executes repository code in production. So the comparison is
fixture-based: K2's own `modeling_deepseek.py` — pinned by the admission gate at
moonshotai/Kimi-K2-Base revision `ce72df012259dcc55d945e890f815fe7ef69159c`, sha256
`ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4` — is run ONCE, in an
isolated venv under the system temp dir, on a pinned old transformers selected by the
recorder, against the shared seeded tiny K2 checkpoint (`tests/kimi_k2_tiny.py`, the exact
builder the admission gate's G2 registers). Its final-position logits and losses for three
fixed seeded inputs are committed as fixtures with a manifest pinning every hash and the
validated versions. The gate test then loads the FP8 tiny checkpoint through
`load_resident_model` on the current stack and compares.

## 2. Decision rules (written before any run)

### Fixtures provenance

Every hash in this chain is asserted, not assumed:

| item | pin |
|---|---|
| `modeling_deepseek.py` | revision `ce72df012259dcc55d945e890f815fe7ef69159c`, sha256 `ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4` (the admission gate's pin; the recorder refuses to run on any other file) |
| `configuration_deepseek.py` | same revision; sha256 recorded in the manifest at record time |
| tiny checkpoint `config.json` | sha256 recorded in the manifest at record time (the bytes `tests/kimi_k2_tiny.py::_tiny_k2_raw()` writes) |
| transformers | the recorder installs candidate 4.x pins (starting at 4.57.6, the stack in `benchmarks/README.md`'s history) into the isolated venv until one not only imports the pinned modelling file but runs it to produce the fixtures; the validated pin is recorded in the manifest |
| fixtures npz | sha256 recorded in the manifest; `--check` and the gate test re-verify it |

`trust_remote_code=True` appears nowhere outside the recorder's isolated venv. The gate
test monkeypatches transformers' dynamic-module loaders to fail the test if called.

### G, the parity cell

Model: the FP8 copy of the shared tiny K2 checkpoint (block 16 x 32, written by
`benchmarks/harness/fp8_block_quantize.py`), loaded through `load_resident_model` at
`dtype=float32` on CPU. Reference: the recorded fixtures (K2's own code, fp32, CPU, three
fixed inputs of lengths 8, 16 and 33, seeded generator over the tiny vocab; the exact ids
are in the manifest and the npz). The npz holds **two** references, both recorded from K2's
own code on the same inputs:

- `logits_i`/`loss_i` — K2's code on the pristine bf16 checkpoint. Reported below against
  transformers' own native class on the current stack (the class the resident route
  builds): it must agree to fp32 roundoff, evidencing that K2's code and the native class
  compute identically on this gate's tolerance.
- `logits_fp8_i`/`loss_fp8_i` — K2's code on the dequantised FP8 round trip, i.e. the
  exact weights the FP8 cell materialises (the checkpoint whose tensors are
  `float8_e4m3fn` + `_scale_inv` dequantises to these values; the FP8 cell cannot be
  referenced against the pristine recording because the e4m3 round trip moves the final
  logits by ~1e-2 on this tiny model — measured in section 5 — three orders of magnitude
  over the gate tolerance; the round trip itself is gated bit-exactly by
  `tests/test_fp8_block_streaming.py`).

The FP8 cell references the second recording:

- **PASS**: for every input, the final-position logits satisfy
  `torch.allclose(rtol=1e-5, atol=1e-6)` (pinned here, before the run; the comparison runs
  at fp32) against `logits_fp8_i`, and the loss matches the same tolerance.
- **FAIL**: any check false, reported with its maximum absolute difference.
- **Fixtures integrity**: the manifest's modelling sha256 must equal the admission gate's
  pin; the fixtures file's sha256 must equal the manifest's; the two references must
  differ (an identical pair would mean the round trip was never recorded). Checked by the
  gate test on every run, and cheaply by the recorder's `--check`.

### Controls (each must hold, or the cell pins nothing)

- **yarn vs default rope**: the same FP8 checkpoint with `rope_scaling: null` (admitted by
  the guard; computes the default rope) must land OUTSIDE the gate tolerance against the
  fixtures.
- **perturbed router bias**: adding 1.0 to every `e_score_correction_bias` after the load
  must land OUTSIDE the gate tolerance against the fixtures.

### Mutation checks

The gate test's docstring names the production changes each guard is for; after the run,
each is applied in the source, a named test must fail, and the file is restored (sha256
checked).

## 3. Not shown here

- **The FP8 round trip.** This gate compares the FP8-dequantised resident load against
  fixtures recorded from K2's own code on the same dequantised weights (section 2 explains
  why not against the pristine bf16 recording); the quantize→dequant round trip itself
  (per-block exactness, ragged edges) is gated bit-exactly by
  `tests/test_fp8_block_streaming.py`. The pristine recording is kept as evidence (section
  5) that K2's code and the native class agree at the gate tolerance.
- **A second transformers pin.** The fixtures are recorded once, on one validated pin; the
  gate does not claim K2's code computes identically across transformers versions (its
  config declares 4.48.3).
- **The MoE-combine rounding difference** the admission gate lists as known-not-gated
  (K2 sums weighted expert outputs in fp32; the native class rounds each to the model
  dtype): at the gate's fp32 tolerance it must be invisible, and the cell below measures
  exactly how invisible.
- No GPU, no real K2 weights, no K2 training run, no throughput.

## 4. Box state

- WSL (Ubuntu on the Windows 11 box; the GPU host's Linux side), CPU only. Python 3.12.3.
- Gate/test env (the maintainer's audit venv): torch 2.15.0+cpu, transformers 5.22.2,
  numpy 2.5.4.
- Recorder venv (isolated, system temp dir): created by the recorder; pip bootstrapped via
  get-pip.py (the host python ships no ensurepip). Validated pin **transformers 4.53.3**,
  torch 2.15.0+cpu, numpy 2.5.4.
- Code: this branch (`issue-2104-resident-k2`, Tasks 1-3 merged at `78f7b441`) plus this
  task's commit (below). Gate test `tests/test_kimi_k2_resident_parity.py` (5 tests) and
  the untouched admission suite `tests/test_kimi_k2_streaming.py` (186 tests) ran in one
  pytest process: 191 passed.
- Fixtures: `tests/fixtures/kimi_k2/resident_parity_fixtures.npz` (7,300 B; sha256
  `a8679f7f73559b40590abf947efaa2e8932b99516b6155038668daf9e9f9a304`) +
  `manifest.json`. `python benchmarks/harness/kimi_k2_resident_parity_record.py --check`
  re-verifies fixtures sha256, manifest pins and npz/manifest id agreement without the venv.
- The pinned `modeling_deepseek.py` (sha256 `ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4`)
  and `configuration_deepseek.py` (sha256 `54f29dd0d56e9a13f639c703e524daa71794e360050012aea3788ad7cfc38b17`)
  were fetched at revision `ce72df012259dcc55d945e890f815fe7ef69159c`; the recorder refused
  to run until the modelling file's full sha256 equalled the admission gate's pin.
- `trust_remote_code=True` executed only inside the recorder's isolated venv. In the gate
  test both of transformers' dynamic-module loaders were monkeypatched to fail the test if
  called; they were called 0 times.

## 5. Results

### The pin walk (why 4.53.3)

The recorder starts at 4.57.6 (the stack in `benchmarks/README.md`'s history) and walks
the 4.x line until a candidate runs the pinned file. Importing the file is not sufficient:
**4.57.6 through 4.54.0 all import it and then fail at runtime** — K2's code calls
`DynamicCache.get_usable_length`, which transformers' newer caches no longer have. The
walk recorded the failures and stopped at **4.53.3**, the first candidate that both
imports and runs the pinned `modeling_deepseek.py`. (The K2 repo's own config declares
`transformers_version` 4.48.3, consistent with a pre-cache-refactor era.)

### G, the parity cell: PASS

`load_resident_model` on the FP8 tiny checkpoint (`dtype=float32`, CPU) against the
fixtures recorded from K2's own code on the dequantised FP8 round trip, three inputs of
lengths 8, 16 and 33; tolerance pinned at `rtol=1e-5, atol=1e-6` (section 2):

| input | len | parity max abs diff (logits) | loss diff |
|---|---|---|---|
| 0 | 8 | 8.9e-08 | 0.0 |
| 1 | 16 | 8.9e-08 | 0.0 |
| 2 | 33 | 6.7e-08 | 0.0 |

The resident route reproduces K2's own published code to fp32 roundoff; the loss is exact.

**The pristine bf16 cell: PASS.** `load_resident_model` on the unquantised bf16 tiny
checkpoint (the K2-spelled folder, stale `inv_freq` keys and all) against the pristine
fixtures (`logits_i`) at the same pinned tolerance: max abs diff 8.9e-08 / 8.9e-08 /
6.0e-08, loss exact (inputs 0/1/2). This cell commits the pristine fixtures to a test —
they are no longer evidenced only by the integrity difference check.

**Pristine-reference evidence.** transformers' own native `DeepseekV3ForCausalLM` on the
current stack (5.22.2), fp32 on the pristine bf16 tiny checkpoint, reproduces the
bf16-source fixtures (`logits_i`) to max abs diff 8.9e-08 / 5.9e-08 across the three
inputs: K2's own code and the native class compute identically at this gate's tolerance,
which is exactly what the resident route relies on.

**Why the FP8 cell references the round-trip fixtures.** The e4m3 quantise→dequant round
trip (block 16 x 32, the admission gate's block) moves the final logits well outside any
honest implementation tolerance, measured pristine-vs-round-trip on the same K2 run:
1.13e-02 / 1.04e-02 / 9.3e-03 for inputs 0/1/2. That error belongs to the quantisation,
not to any load path; it is gated bit-exactly by `tests/test_fp8_block_streaming.py`
(per-block dequant exactness, both layouts) and the admission gate's G2 (streamed vs
resident on the same FP8 source). Referencing the round-trip fixtures keeps this gate
measuring the resident route against K2's own computation on the weights it actually
materialises. The pristine fixtures are committed alongside and checked for integrity
(the two references must differ).

### Controls: PASS

- **default rope** (`rope_scaling: null`, admitted by the guard): max abs diff against the
  FP8 fixtures 1.5e-03 / 1.1e-03 / 1.6e-03 (inputs 0/1/2) — outside the gate tolerance.
- **zeroed router bias** (`e_score_correction_bias.zero_()`): max abs diff 1.3e-01 /
  7.7e-02 / 1.0e-01 — outside the gate tolerance. (A mere +1.0 constant shift was tried
  first and does NOT move the logits: sigmoid saturation leaves the top-k and weights
  unchanged; the gate pins the zeroing control, matching the admission gate's.)

### Fixtures integrity: PASS

Manifest modelling sha256 equals the admission gate's pin; the npz sha256
(`a8679f7f…a304`) matches the manifest; manifest↔npz id agreement checked; the pristine
and round-trip references differ on every input.

## 6. Mutation checks

Each mutation was applied to the source, the parity test ran, the file was restored byte
for byte (git-verified clean after each round); all three mutants caught, each by
`test_the_resident_fp8_load_matches_k2s_own_remote_code`:

| mutation | the production change it stands for | result |
|---|---|---|
| M1 | bypassing the per-tensor FP8 dequant in `load_resident_model` (raw FP8 bytes cast instead of block-dequantised) | FAIL as required |
| M2 | dropping `routed_scaling_factor` from the native config mapping | FAIL as required |
| M3 | loading the rope with the class-default theta (10000.0) instead of K2's published 50000.0 | FAIL as required |

After the mutations: `tests/test_kimi_k2_resident_parity.py` + the admission suite,
191 passed in one process.
