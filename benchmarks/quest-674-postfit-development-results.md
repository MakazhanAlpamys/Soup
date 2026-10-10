# Issue 674: best retained post-fit development checkpoint

4 October 2026. The best retained integrated QuEST checkpoint has response-NLL
ratio **1.0383673005** against the strongest retained FP checkpoint on DEV704.
It remains above the zero-margin parity target. This record preserves the best
checkpoint selection, its last-stage recipe, numeric evidence, and nine negative
or inconclusive follow-ups. It does not authorize final confirmation.

## Best retained result

Both endpoints use the same 704 sequences and 25,017 response targets from
DEV192B + CONFIRM512. This panel has already been used for candidate selection.

| Endpoint or contrast | Result |
| --- | ---: |
| QuEST: `readout-adaptation-v1/runs/fit-quest-full` | NLL 2.125037937785311 |
| FP: `prompt-addition-v1/runs/fit-fp` | NLL 2.046518545766202 |
| QuEST / FP | 1.0383673004974954 |
| QuEST minus FP | +0.07851939201910918 nat/target |
| Paired 95% gap interval | [+0.07149328368143801, +0.08602618686052922] |

The interval uses 2,000 paired sequence bootstrap draws with NumPy RNG seed 674
and target-weighted ratios of loss sums. It is descriptive after adaptive
selection and conditional on these selected checkpoints. The zero-margin rule
requires ratio <= 1.000 and paired 95% upper gap <= 0; neither passes here.

The [published evaluation-only mixed-route result](gate-674-quest-mixed-route.md)
had a gap of +0.086344 against its fixed FP control. This selected post-fit result
has a smaller gap and lower absolute Q loss on the same pooled panel, but the
FP control and training histories differ. It is not a controlled estimate of the
effect of the continuation code change, and the historical three-seed average
must not be relabelled as this selected one-seed measurement.

## Preserved last-stage recipe

The winner is the **full-parameter continuation arm** of the readout experiment,
not its norm/head-only arm. It retains 219 FP32 master parameter tensors
(461,685,760 parameters) and uses BF16 execution. The integrated route remains
168 fake-W4 linears, 161 A4 activations and seven A16 activations in block 23;
group 128 and the full-width normalized Hadamard are unchanged. The inherited
clipping table is restored from the parent's QuEST metadata.

- Student parent: `feature-continuation-v1/runs/fit-quest-control`, master
  parameter digest `900ef3147a35e17738e0e4cb4b2a3b82672c82375916c7d8cba4b32ebb5bd6a0`.
- Frozen teacher: `teacher-followup-v1/runs/fit-fp`, digest
  `83b4bfedd4644f05e54f3e0982eeba43c8d1bde9dc8cdf29aff413c7f7e97968`.
- Objective: 0.25 response cross-entropy + 0.75 response forward KL, temperature 1;
  feature loss weight 0. This is a research adapter around native SFT. The public
  Soup CLI does not expose this combined QuEST/teacher recipe.
- One epoch of the retained 2,037-row TRAIN pool: 1,019 updates, batch 2 with a final
  singleton, 92,357 response targets; training-row source hash
  `7d5f832f5b2eac4ad7a45b7819eea26aaf6328db658839062d018f18c1c22e0c`.
- Fresh AdamW optimizer, beta1 = .9, beta2 = .95, epsilon = 1e-8, weight decay 0,
  gradient norm cap 1; peak LR 5e-6, cosine horizon 1,019, warmup 102 updates.
- Model seed 42, data seed 674, gradient accumulation 1, gradient checkpointing off.

The Q winner and strongest FP have different parent histories. Their complete
last-stage receipts each record 1,019 student forwards, teacher forwards,
backwards and optimizer updates, 2,037 examples and 92,357 response targets. The
portable evidence records the lineage protocol hashes; reproduction requires
the separately retained parents, encoded TRAIN pool, and research workers.
Starting from the original Hub model and using only these last-stage settings
does not reproduce this checkpoint.

## Follow-ups that did not justify replacing the winner

These are results of the specified tests, not universal exclusions of the methods.
TRAIN diagnostics do not constitute held-out Q/FP parity evidence.

| Follow-up | Scope and observation | Decision |
| --- | --- | --- |
| Dolly/OASST1/OASST2 continuation | Completed matched fits; DEV704 Q/strongest-FP ratio 1.040610 | Worse than retained 1.038367 |
| UltraChat continuation | 1,024 separate training pairs and 512 updates per arm; eight development scores. DEV704 ratio 1.041713; UltraChat ratio 1.047574 | Gap change only -0.002257 nat, CI [-0.008683, +0.004379]; missed 0.03-nat gate and exceeded Dolly regression tolerance |
| Stronger teacher target | Small fitted-TRAIN screen; treatment minus control +0.001086 nat, CI [-0.012057, +0.014800] | No supported improvement; gate failed |
| Intermediate feature distillation | Small fitted-TRAIN screen; treatment minus control -0.009348 nat, CI [-0.023029, +0.003732] | Point improvement uncertain; gate failed |
| FP32 Hadamard arithmetic | 16-row fitted-TRAIN diagnostic; treatment minus control +0.013716 nat, CI [-0.008544, +0.036046] | No supported improvement; gate failed |
| Scalar logit temperature | Q TRAIN loss improved; Q/FP gap change -0.002887 nat, CI [-0.008448, +0.002355] | Gap-narrowing gate failed |
| Group size 64 | 16 fitted-TRAIN rows; group 64 minus native 128 +0.138453 nat, CI [+0.088862, +0.189135] | Worse; not promoted |
| Block-128 Hadamard on trained Q | 64 fitted-TRAIN rows; block minus native +0.185829 nat, CI [+0.151520, +0.218562] | Worse; native restoration bit-identical |
| Common W/A group permutation | 16 fitted-TRAIN rows; cross-group minus native +0.161242 nat; whole-group symmetry control exceeded frozen tolerances | Inconclusive mechanism test; failed control prevents a clean regrouping claim |

The UltraChat trained Q and FP each completed 512 updates, 1,024 examples and 63,579
response targets. Their example order, batch-size and response-mask hashes match.
Both improved in absolute UltraChat loss, while the Q/FP ratio changed from
1.047491 to 1.047574. The OASST2 development ratio was 1.043632 against the new FP.
All eight scoring actions used fresh model processes and zero training updates.

## Evidence and verification

[Portable evidence](evidence/quest-674-postfit-v1.json) includes paired per-sequence
loss sums and identities, checkpoint-file hashes, recipe/protocol bindings and
source hashes for the negative records. It excludes raw tokenized data, model
weights and machine-specific paths. The large checkpoints and original research
workers remain separately preserved local artifacts, not Git objects.

Verify the reported statistic without loading a model or accessing a final panel:

```bash
python benchmarks/harness/quest_parity_evidence.py
```

Optionally verify all 27 files of the retained Q/FP artifacts in the preserved
artifact tree. This reads checkpoint files for hashing but performs no inference:

```bash
python benchmarks/harness/quest_parity_evidence.py --artifact-root .quest-validation/parity-plan-v1
```

The verifier rejects changed pairing, invalid losses, changed target weighting,
altered summaries, unsupported confirmation claims, and artifact paths outside
the declared root. Targeted continuation/integration/evidence checks: 122 passed
with two Click deprecation warnings. Repository Ruff checks passed.

`training_quality_validated` stays false. Three-seed confirmation remains pending.
FINAL256 and OASST256 remain unopened and reserved for a single separately frozen
confirmation only after an eligible candidate. These measurements do not establish
cross-model generality, pure W4A4, upstream QuEST numerical parity, packed INT4 or
efficiency. No compute was purchased or credentials/permissions changed.
