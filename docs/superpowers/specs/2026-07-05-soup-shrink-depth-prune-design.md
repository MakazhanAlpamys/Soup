# v0.71.29 — `soup shrink`: depth-prune + distill-heal

**Status:** approved 2026-07-05 · **Target release:** v0.71.29
**Paper:** "The Unreasonable Ineffectiveness of the Deeper Layers" (Gromov et al., arXiv:2403.17887); heal is Minitron-style logit distillation rather than the paper's LoRA-finetune.

## Pitch

"Make your model ~25 % smaller and faster, locally." Rank layers by importance
(angular distance of the residual stream across a contiguous block over a
calibration set), drop the least-important contiguous block, then optionally
**heal** by distilling the original model into the pruned student. One dense,
ship-ready smaller model comes out the other end, with a before/after
perplexity verdict.

## Command

```
soup shrink --model <id|path>
            [--drop-ratio 0.25 | --drop-layers N]
            --calib <calib.jsonl>
            [--heal <heal.jsonl> --heal-steps N]
            [--tolerance 0.10]
            [-o <dir>]
            [--device cpu|cuda]
            [--attach-to-registry <id>]
            [--plan-only]
```

- `--drop-ratio` / `--drop-layers` are mutually exclusive; exactly one required.
  Both specify only the block **count** `n`; the block **position** is always
  chosen by argmin over the importance scan (users do not pin a start index in
  v1). `drop_layers = round(drop_ratio * num_layers)`.
- `--calib` is required (the importance pass needs prompts).
- `--heal` optional; when set, `--heal-steps` (default 200) drives the distill.
- `--tolerance` — perplexity-regression tolerance for the verdict (default 0.10
  = 10 %).
- `-o` default `./shrunk`.
- `--plan-only` — print the importance table + chosen block + would-run heal
  command and exit 0, without writing weights or training.

## Flow

1. **Load** the model + tokenizer via `live_eval.load_model_and_tokenizer`
   (`trust_remote_code=False` probe + warn on `--model`, mirroring
   `chat.py`/`diff.py`).
2. **Score importance** — one forward per calib prompt with
   `output_hidden_states=True`; accumulate per-token angular distance for every
   candidate block (see *Importance metric*).
3. **Select block** — argmin angular distance over the **position-bounded**
   candidate set (see *Position bounds*).
4. **Prune** — slice `model.model.layers` to remove the block, patch
   `config.num_hidden_layers`, `save_pretrained(<out>/model)`.
5. **Measure ppl** — original ppl (step 1 model) and pruned ppl measured on the
   model **reloaded from `<out>/model`** (see *Reload-before-measure*).
6. **Heal** (optional) — subprocess `soup train` with `task='distill'`,
   `teacher_model=<original>`, `base=<out>/model`, LoRA student → adapter at
   `<out>/heal_adapter`; then fuse the adapter back into `<out>/model` so the
   final artifact stays a single dense model; re-measure final ppl on the
   reloaded fused model.
7. **Verdict** — `decide_shrink(ppl_original, ppl_final, tolerance)` → render a
   one-screen SHIP / DON'T-SHIP panel + params-saved %.
8. **Registry attach** (optional) — attach the shrink report JSON as an artifact
   on `--attach-to-registry <id>`.

## Importance metric (paper-faithful; off-by-one pinned)

`output_hidden_states=True` returns a tuple of length `num_layers + 1`.
**Index 0 is the embedding output; index `i` (i ≥ 1) is the output of decoder
layer `i-1`.** For a contiguous block of length `n` whose first dropped layer is
`L` (0-indexed decoder layer), the residual stream **entering** the block is
`hidden_states[L]` and the stream **leaving** it is `hidden_states[L + n]`.

Per the paper, the block importance is the **angular distance** between those two
residual vectors, per token, averaged over all non-pad tokens across the whole
calibration set:

```
d = (1/π) · arccos( <h_L, h_{L+n}> / (||h_L|| · ||h_{L+n}||) )
```

Reduction: **per-token angular distance, then mean over every non-pad token**
(over sequence positions and over the calib set) — NOT last-token-only. Lower
`d` ⇒ the block transforms the residual stream the least ⇒ safest to drop.

**Off-by-one is the top silent-bug risk.** A unit test on a tiny known-shape
model (e.g. a 4-layer toy config) asserts the exact boundary indices
(`hidden_states[L]` / `hidden_states[L+n]`) and that
`len(hidden_states) == num_layers + 1`, not merely that the pass runs.

## Position bounds

`select_drop_block` enforces both a **count** and a **position** bound:

- count: `1 ≤ n < num_layers`;
- position: the dropped block must exclude **both the first and the last decoder
  layer** — the paper shows layer 0 and the final layer carry the most
  transformation. Valid start indices are `L ∈ [1, num_layers − n − 1]`
  (inclusive), so the dropped set `[L, L+n−1] ⊆ [1, num_layers−2]`.
- If no valid block exists (`n > num_layers − 2`), raise a friendly error naming
  the max droppable count for this model.

The `--drop-layers` manual path is guarded by the same position bound (argmin is
taken only over position-valid blocks), so a user cannot force-drop the first or
last layer.

## Reload-before-measure

Pruned ppl (and, when healing, final ppl) is measured on the model **reloaded
from the saved dir**, never the in-memory sliced module. Slicing the
`ModuleList` leaves each surviving layer's `self_attn.layer_idx` stale (still its
original index) — wrong for KV cache during generation. `from_pretrained`
reconstructs layers with correct contiguous `layer_idx`, so measuring on the
reloaded artifact (a) fixes this, (b) validates exactly what the user ships, and
(c) surfaces any config/state-dict mismatch immediately. Cost: one extra load —
worth it.

## Heal (Minitron-style logit distillation)

Verified in-repo: `DistillTrainerWrapper._distill_term` (distill.py:47–118)
computes **logit-level** KL/JS divergence over the vocab dimension only — no
hidden-state / feature matching. Therefore a pruned (fewer-layer) student vs the
full-depth teacher is dimensionally safe (both emit `[B, S, vocab]` logits); no
layer mapping is required.

- Heal runs as a **subprocess `soup train`** (mirrors
  `ra_dit_run._run_train_subprocess`): a generated `soup.yaml` with
  `task: distill`, `teacher_model: <original>`, `base: <out>/model`, LoRA
  student, `data.train: <heal.jsonl>`, epochs/steps from `--heal-steps`.
- **Memory:** distill keeps the teacher resident ⇒ heal is ~2× model memory.
  The student trains as **LoRA** (not full-FT) to stay within the consumer-GPU
  budget. Documented guideline: heal validated ≤ 3 B on the 4 GB dev box; larger
  is works-but-unvalidated advisory (same honesty class as the v0.71.24 giants).
- After the distill subprocess, the LoRA adapter is fused back into
  `<out>/model` (reusing the existing merge path) so the shipped artifact is a
  single dense smaller model. `<out>/heal_adapter` is retained for transparency.

## Verdict (`decide_shrink` — dedicated, mirrors ship_verdict)

Pure, no top-level torch. Mirrors `ship_verdict`'s frozen-dataclass +
`render_*_panel` + `*_to_dict` shape but with its own rule — `decide_ship` would
trivially reject every shrink (pruning always raises ppl).

```python
@dataclass(frozen=True)
class ShrinkVerdict:
    decision: str          # SHIP | DON'T SHIP
    ppl_original: float
    ppl_final: float
    ppl_ratio: float       # ppl_final / ppl_original
    tolerance: float
    layers_before: int
    layers_after: int
    params_saved_pct: float
    healed: bool
    soup_version: str
```

Rule: **SHIP iff `ppl_final / ppl_original − 1 ≤ tolerance`**, else DON'T SHIP.
Exit code: 0 = SHIP, 2 = DON'T SHIP, 1 = runtime error (mirrors
`soup diagnose` / `soup ship`). Reject non-finite / non-positive ppl and
`tolerance ∉ [0, 5]` with clear errors.

## Files

- `src/soup_cli/utils/shrink.py` (NEW):
  - torch-lazy: `compute_layer_importance`, `select_drop_block`,
    `prune_model_layers`, `SUPPORTED_SHRINK_ARCHS` allowlist + `_shrink_arch_of`.
  - pure: frozen `LayerImportance` + `ShrinkVerdict`, `decide_shrink`,
    `render_shrink_panel`, `shrink_verdict_to_dict`.
- `src/soup_cli/commands/shrink.py` (NEW Typer command) — orchestration, path
  guards, subprocess heal, registry attach.
- `src/soup_cli/cli.py` — register `shrink`.
- `tests/test_v07129.py`.
- Docs: `docs/peft-and-efficiency.md` (or `docs/training.md`), `docs/commands.md`,
  README `## What's New`, `CHANGELOG.md`, `.claude/CLAUDE.md`.

## Arch allowlist (v1)

Llama / Qwen / SmolLM (all expose `model.model.layers` as an `nn.ModuleList` and
`config.num_hidden_layers`). Anything else → friendly reject naming the supported
families. Detection mirrors the `is_*_model` regex-word-boundary style used in
`longlora.py` / `peft_patches.py`.

## Security

- `--calib`, `--heal`, `-o` — cwd-contained (`os.path.realpath` +
  `os.path.commonpath`) + O_NOFOLLOW open + size cap on the JSONL reads, mirroring
  `commands/diagnose.py::_load_evidence` and the v0.71.28 MCP guards.
- `--model` unconstrained (mirrors every other model-loading command);
  `trust_remote_code=False` probe + warn.
- Drop bounds validated (count + position); arch allowlist friendly-reject.
- Generated heal `soup.yaml` written to a fixed path under `-o`; config values
  are schema-validated via `load_config_from_string` before the subprocess runs
  (no shell; argv list).
- `--drop-layers` / `--heal-steps` / `--tolerance` bounded (reject bool, negative,
  out-of-range).

## Honesty / known limitations (for release notes)

1. **Importance pass loads the model** → live-validated ≤ 3 B (SmolLM2-360M
   dev-box smoke); larger models are works-but-unvalidated advisory.
2. **Heal is ~2× model memory** (teacher resident); LoRA student keeps it
   tractable; ≤ 3 B validated on 4 GB.
3. **Arch allowlist v1 = Llama/Qwen/SmolLM**; other families are a friendly
   reject, not silently mishandled (follow-up issue for MoE / GQA-exotic arches).
4. **Angular-distance importance is a heuristic** (per the paper) — a pre-flight
   estimate, not a guarantee; the ppl verdict is the ground truth.

## Smoke (RTX 3050, step 6)

- SmolLM2-360M, `--drop-ratio 0.25`, tiny calib → importance table + chosen block
  → ppl before/after on the reloaded pruned dir.
- `--heal <tiny.jsonl> --heal-steps 200` → distill subprocess → fuse → final ppl
  → verdict panel (SHIP or DON'T with the ratio).
- Failure modes: `--drop-ratio` + `--drop-layers` together (reject); drop count
  ≥ num_layers-1 (reject with max); non-allowlisted arch (friendly reject);
  calib outside cwd (reject); bad tolerance (reject).
```
