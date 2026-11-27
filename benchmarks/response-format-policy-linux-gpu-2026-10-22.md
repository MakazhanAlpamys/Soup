# Response-format policies and decoder stopping on Mistral-7B

## Question

At a fixed supervised-token budget, does filtering or ordering training examples
improve held-out cross-entropy (CE) or training time? In separate inference
experiments, does stopping at the next role marker improve exact answers, and
do fixed examples of the requested output format add any benefit?

## Environment and training method

The training and boundary measurements ran on CUDA Linux guests from
2026-10-20 to 2026-10-22. The published comparison files omit the GPU and CPU
models, installed VRAM and host RAM, and the training guest's distro and driver
versions. The boundary recovery pinned NVIDIA driver 580.178.04. Shared runtime
pins were Python 3.12.3, torch 2.14.1, CUDA toolkit 13.0.3.0 and runtime wheel
13.0.96, Transformers 5.19.0, PEFT 0.21.2, safetensors 0.8.0, tokenizers
0.23.2 and Accelerate 1.15.0.

The study used `mistralai/Mistral-7B-v0.3` at
`caa1feb0e54d415e2df31207e5f4e273e33509b1` and
`HuggingFaceH4/ultrachat_200k` at `8049631c405ae6576f93f445c6b8166f76f5505a`.
Each policy had one full run from the same FP16 base with a fresh FP32 LoRA
adapter and AdamW optimizer. Settings were seed 901, FP16 autocast, LoRA rank 8
and alpha 16 on `q_proj` and `v_proj`, dropout 0, learning rate `1e-4`,
maximum sequence length 2048 and packing disabled. The full arms ran in table
order. Each consumed 5,000,000 useful loss targets in 611 updates, with 8,192
targets per update and 2,880 in the final update. Separate pilots preceded
the full study. The training-phase time has no separate timing warm-up specified
and excludes data preparation, model load, evaluation and checkpoint reload.

Baseline shuffles the admissible pool. The coherence filter applies a heuristic
threshold of 0.7 before shuffling. The two length curricula use the baseline's
token multiset, sorted shortest first or longest first. The shared held-out
set has 256 examples and 322,332 targets. Screening checked normalized
conversations, prompts, source IDs and shared 13-word spans. Every full arm
passed frozen-base, AMP, exact raw and PEFT reload, and guest run-validity and
clock checks. The run-validity checks track suspend, power and other disk reads.

The `full_study.py` harness uses `check` for the offline checks, `full` once
for each arm with the frozen specification and prepared data, and `review`
with the four completed run roots. `FULL_STUDY_PROTOCOL.txt` specifies the
inputs and arguments.

## Four-policy results

| Policy | Useful targets | Non-padding input tokens | Processed slots | Padding slots |
|---|---:|---:|---:|---:|
| Baseline | 5,000,000 | 5,459,986 | 5,473,632 | 13,646 |
| Coherence filter | 5,000,000 | 5,456,023 | 5,470,224 | 14,201 |
| Shortest first | 5,000,000 | 5,440,667 | 5,454,152 | 13,485 |
| Longest first | 5,000,000 | 5,340,190 | 5,354,064 | 13,874 |

| Policy | Held-out CE | Relative CE change (fraction) | Train seconds | Useful targets/s | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9839257797862864 | 0.0 | 5609.586099249998 | 891.3313587732436 | 17582362624 | 19868418048 |
| Coherence filter | 0.9841728593419583 | 0.00025111605036465257 | 5597.937483947004 | 893.18610905147 | 17582297088 | 20539506688 |
| Shortest first | 0.9929982158818574 | 0.00922065086813939 | 5597.310593071001 | 893.286144633385 | 17581314048 | 24792530944 |
| Longest first | 0.9942416858139351 | 0.010484435147018356 | 5533.245572255 | 903.6287897777718 | 17580265472 | 18398314496 |

Baseline had the lowest CE. The coherence filter and shortest-first ordering
were within the prespecified 1% CE margin. Longest first exceeded it.
The original decoder scored 0/12 on the frozen task diagnostics for every
trained adapter.

## Response-boundary confirmation

The `boundary_confirmation.py` harness paired both decoders for each frozen
model state, one exclusive supervised process per state, with decoder order
alternating across states. Prompts, weights, strict exact/JSON scoring and
greedy decoding with a 128-token cap were fixed. The candidate stopped at the
first generated newline role marker (`user:`, `assistant:` or `system:`)
and scored only the text before it. Its raw tokens had to match the original
generation prefix on the same prompt. The fresh suite contained 48 tasks in
12 families. The old task prompts were also replayed as diagnostics. An earlier
attempt stopped at an NVML driver/library mismatch before generation. The
separately declared recovery produced the accepted measurements below.

| Model state | Old exact, original | Old exact, boundary | Fresh exact, original | Fresh exact, boundary | Fresh paired losses | Fresh tokens, original | Fresh tokens, boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0/12 | 2/12 | 0/48 | 7/48 | 0 | 6144 | 679 |
| Baseline | 0/12 | 4/12 | 0/48 | 14/48 | 0 | 6105 | 657 |
| Coherence filter | 0/12 | 4/12 | 0/48 | 15/48 | 0 | 5945 | 621 |
| Shortest first | 0/12 | 3/12 | 0/48 | 15/48 | 0 | 6144 | 758 |
| Longest first | 0/12 | 3/12 | 0/48 | 14/48 | 0 | 5229 | 724 |

The fixed descriptive gate passed. It required a trained-baseline gain of at
least 4/48, zero losses for every trained adapter, nonnegative gains for the
other trained adapters, and complete validity, prefix and state checks.

## Format-prompt confirmation

The prospective `format_prompt_confirmation.py` harness kept the boundary
decoder fixed and paired the original prompt with a prompt that prepended two
fixed format demonstrations. It specified the initial and trained baseline
states, 96 fresh tasks across 12 families, and 12 safety prompts for separate
blinded review. Preflight stopped the attempt before generation. The attempt
produced 0 responses, so there are no format-prompt scores.

**Verdict: MIXED.** The boundary decoder improved paired exact answers. The single-seed policy comparison established neither a quality winner nor a speedup, and the format-prompt attempt produced no results.

## Limitations

The CE and timing observations are exploratory. One run per policy leaves
between-run variation unknown, and filtering changes the training mix. The
boundary test measures a decoding change on a small synthetic suite, where
absolute exact scores remained low. Lexical screening leaves semantic overlap
and pretraining exposure unknown. Guest checks leave physical-host isolation
unverified. Independent blinded safety review is pending, so overall quality
has no verdict.

Measured by @nurseit14

Evidence: https://github.com/SoupTeam/soup-rnd/tree/db0287a173ad8d8762b48679bf04b82460ff1f0a/benchmarks/g6
