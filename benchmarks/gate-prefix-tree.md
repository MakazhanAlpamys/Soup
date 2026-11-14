<!--
Measurement record, published as measured, in the order things happened.

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 581.15; 15.8 GB host RAM;
Windows 10 (10.0.19045). One machine, 2026-11-13 and 2026-11-14.
Stack: Python 3.11.5, torch 2.14.1+cu130, transformers 5.22.2, peft 0.21.2, trl 0.29.1,
bitsandbytes 0.50.2.
Models: unsloth/mistral-7b-instruct-v0.3 at revision 60b99dcbc7f8 (the revision the other
records pin) and Qwen/Qwen2.5-0.5B-Instruct (model.safetensors 988,097,824 bytes, downloaded
2026-11-13, revision not pinned). NF4 with double quantisation, layer-streamed, RAM tier,
LoRA r 8 on q_proj / v_proj, dropout 0, lr 1e-4, seed 3, batch_size 1.
Data: glaiveai/glaive-function-calling-v2 (e7f4b64560) and databricks/databricks-dolly-15k
(bdd27f4d94), first turn of each conversation, rows in file order that fit max_length; and
Soup's own tool-calling fixture, src/soup_cli/data/_fixtures/gate/tool_call.jsonl (40 rows).
Harness: benchmarks/harness/prefix_tree_ab.py. Raw output: results/prefix-tree-rtx3050-v1/.
-->

# Gate record — `training.prefix_tree`: tokens several examples share are computed once

**Status: MEASURED on one laptop (RTX 3050 Laptop 4 GB), a 7B and a 0.5B model, three
datasets.** Through the shipped trainer, layer-streamed NF4:

| | data | examples/s, one per step -> prefix tree | held-out loss, same wall time |
|---|---|---|---|
| Mistral-7B | tool calling, one 4-function schema | 0.400 -> 1.404 (3.5x) | 0.1488 -> **0.1340** in 66 % of the time |
| Mistral-7B | the same, rows of 1024 / 2048 tokens | 0.392 -> 3.109 (7.9x) / 0.375 -> 3.768 (10.0x) | not run |
| Mistral-7B | tool calling as published | 0.479 -> 1.130 (2.4x) | not run |
| Mistral-7B | plain instructions (dolly) | 0.563 -> 1.235 (2.2x) | not run |
| Qwen2.5-0.5B | tool calling, one 4-function schema | 2.42 -> 10.72 (4.4x) | 0.2556 -> **0.2126** |
| Qwen2.5-0.5B | tool calling as published | 2.63 -> 5.03 (1.9x) | 0.2078 -> **0.1950** |
| Qwen2.5-0.5B | plain instructions (dolly) | 2.69 -> 7.34 (2.7x) | 1.4806 -> 1.4848 (no gain, section 4) |

Peak VRAM stays where batch 1 has it (1.1 to 1.3 GB at 512-token rows on both models). Loss
agreement with the ordinary batch is exact in fp32 and inside each model's own layout noise in
bf16 (section 5). **Not measured: any other card, any model above 7B, generation quality (only
held-out loss), more than one seed.**

Unit convention: **decimal GB**, as in the other records.

---

## 1. What it is

In a causal language model a token's hidden state depends only on the tokens before it. Rows
that begin with the same tokens have bit-identical states over that beginning, and an ordinary
batch computes them once per row. A prefix-tree row holds the shared beginning once and the
continuations after it, with a 4D attention mask (a token attends to itself and its ancestors)
and `position_ids` equal to the depth in the tree. Every token sees exactly the context it had
in its own example, so the loss and its gradient are those of the ordinary batch holding the
same examples. `src/soup_cli/utils/prefix_tree.py` has the three rules that keep that exact
with the stock shifted loss (only untrained tokens are shared; a trained token's parent is
repeated in front of it when it is not its row neighbour; a root's label is ignored).

A step is still at most `batch_size` rows of `data.max_length` tokens, so the layer-streaming
VRAM pre-flight stays valid. That is why `prefix_tree` is allowed beside `stream_layers`, where
`packing` and `multipack` are refused.

Two things make a step cheaper per example, and the tables below keep them apart where they
can: fewer tokens (the shared beginning) and more examples per row (a row is filled up to
`max_length` even when nothing is shared, and a streamed step pays a per-sweep cost that is
then paid once for several examples).

## 2. Speed: the same examples, one per step against prefix-tree rows

`prefix_tree_ab.py` builds the model, tokenizer, dataset and collator through
`SFTTrainerWrapper.setup` and runs each planned group (one prefix-tree row's examples) as
`one` (each example its own step, what a card that fits batch 1 does today) and as `tree`.
Forward + backward through `trainer.training_step`, no optimizer step, so both arms see the
same weights. Arms interleaved, order rotated, medians. Token counts are for all 600 training
rows; the timed part is the first 30 groups on the 0.5B model (3 rounds) and the first 6 on the
7B model (2 rounds).

| model | data | `max_length` | tokens, ordinary -> tree | examples per row | `one` | `tree` | | peak VRAM, one -> tree |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B | glaive, one schema | 512 | 227,027 -> 82,472 (2.75x) | 3.55 | 0.400 ex/s | **1.404** | 3.51x | 1.14 -> 1.19 GB |
| Mistral-7B | glaive, one schema | 1024 | 227,027 -> 38,618 (5.88x) | 15.4 | 0.392 | **3.109** | 7.93x | 1.15 -> 1.45 GB |
| Mistral-7B | glaive, one schema | 2048 | 227,027 -> 30,297 (7.49x) | 40.0 | 0.375 | **3.768** | 10.0x | 1.15 -> 1.94 GB |
| Mistral-7B | glaive as published | 512 | 146,495 -> 116,957 (1.25x) | 2.11 | 0.479 | **1.130** | 2.36x | 1.11 -> 1.19 GB |
| Mistral-7B | dolly | 512 | 89,586 -> 88,647 (1.01x) | 2.70 | 0.563 | **1.235** | 2.19x | 1.12 -> 1.20 GB |
| Qwen2.5-0.5B | glaive, one schema | 512 | 194,853 -> 60,867 (3.20x) | 4.80 | 2.42 | **10.72** | 4.43x | 1.09 -> 1.26 GB |
| Qwen2.5-0.5B | glaive as published | 512 | 128,672 -> 100,542 (1.28x) | 2.48 | 2.63 | **5.03** | 1.92x | 1.12 -> 1.24 GB |
| Qwen2.5-0.5B | dolly | 512 | 98,247 -> 88,875 (1.11x) | 2.79 | 2.69 | **7.34** | 2.73x | 1.15 -> 1.26 GB |

- **"One schema"** is glaive-function-calling-v2 as an agent with a fixed tool set: the four
  most frequent functions (compact JSON, 256 tokens for Qwen, 315 for Mistral) given to every
  row about one of them. **"As published"** keeps each row's own function definitions; rows
  about the same function still share them. **dolly** has no system prompt; Qwen's template
  adds its short default one, Mistral's adds none.
- Mistral's instruct template renders no system role, so for that model the harness puts the
  schema at the start of the user turn (`--system-in-user`), which is that format's own
  convention.
- **dolly on Mistral shares nothing (1.01x fewer tokens) and is still 2.19x faster per
  example**: a streamed 7B step costs about 1.8 s for a 150-token example and most of that is
  per sweep, not per token, so 2.7 examples per sweep is the whole gain. The same effect is
  why the other rows beat their token ratio.
- **Longer rows**: the 7B model at `max_length` 512 uses 1.19 GB of this card, so rows of 1024
  and 2048 tokens fit (1.45 and 1.94 GB). A batch-1 run gains nothing from a longer
  `max_length` (its examples are about 380 tokens); a prefix-tree row holds 15 and 40 of them.
- **The ordinary batch of the same examples** was run on the 0.5B model only: 2.37 ex/s at a
  5.73 GB peak (one schema; over the card, the times spread 50 % between rounds), 3.77 ex/s at
  2.08 GB (as published), 2.65 ex/s at 3.08 GB (dolly). It is the same gradient as the tree
  row at 1.7x to 4.5x its memory.
- Round-to-round spread of the `tree` arm: 0.0 to 1.1 % on the 7B model, 2.6 to 10.9 % on the
  0.5B model.
- The first runs, on the 40-row fixture (streamed 0.5B): one schema for all rows 2.10 -> 19.92
  ex/s (9,147 -> 1,975 tokens); as shipped, with 27 different tool lists, 2.23 -> 5.72 ex/s
  (8,782 -> 5,887 tokens).

## 3. The training loop, and loss on rows it never trained on

`prefix_tree_ab.py --train` calls `wrapper.train()`; 100 held-out rows (the next rows of the
same file) are scored before and after. Three runs per model and dataset on the same training
rows: **A** ordinary, one epoch; **B** prefix tree, one epoch (the same examples, fewer
optimizer steps); **C** prefix tree for as many epochs as fit in about A's time. "Training" is
`train_runtime` minus the end-of-epoch evaluations the trainer runs inside it.

| model, data | | epochs | optimizer steps | training | held-out loss |
|---|---|---|---|---|---|
| Mistral-7B, glaive one schema (300 rows) | before | | | | 2.8378 |
| | A ordinary | 1 | 300 | 794.5 s | 0.1488 |
| | B prefix tree | 1 | 86 | **263.0 s** | 0.1833 |
| | C prefix tree | 2 | 172 | 524.7 s | **0.1340** |
| Qwen2.5-0.5B, glaive one schema (600 rows) | before | | | | 1.9687 |
| | A ordinary | 1 | 600 | 245.2 s | 0.2556 |
| | B prefix tree | 1 | 125 | **57.7 s** | 0.3555 |
| | C prefix tree | 4 | 500 | 229.2 s | **0.2126** |
| Qwen2.5-0.5B, glaive as published (600 rows) | before | | | | 1.4274 |
| | A ordinary | 1 | 600 | 254.9 s | 0.2078 |
| | B prefix tree | 1 | 242 | **117.6 s** | 0.2408 |
| | C prefix tree | 2 | 484 | 234.7 s | **0.1950** |
| Qwen2.5-0.5B, dolly (600 rows) | before | | | | 2.1400 |
| | A ordinary | 1 | 600 | 258.2 s | **1.4806** |
| | B prefix tree | 1 | 215 | 154.9 s | 1.5006 |
| | C prefix tree | 2 | 430 | 306.7 s | 1.4848 |

- **One epoch is faster and ends slightly worse.** A prefix-tree step holds several examples,
  so an epoch is a quarter to a half of the optimizer steps: 3.0x, 4.3x, 2.2x and 1.7x less
  training time, and a held-out loss 0.02 to 0.10 higher at the same learning rate.
- **The same time ends better on the tool-calling data**: on all three such runs, C reaches a
  lower held-out loss than A in no more time (the 7B run in 66 % of it).
- **On dolly it does not**: C took longer than A and ended 0.004 higher. See section 4.
- Single runs, one seed, held-out LOSS on the next 100 rows of the same file. Nothing here
  measures generated tool calls or answers.

## 4. What went wrong, and what is not explained

- **dolly on the 0.5B model trains slower than the A/B predicts** (0.72 s per prefix-tree step
  against 0.42 s in section 2; reproduced on an idle card). Per step: rows up to 468 tokens
  take about 0.44 s, rows of 470 tokens and more about 1.2 s, and dolly's rows are the ones
  that vary in width (106 to 512). The same threshold appeared for ORDINARY batches of random
  tokens in that process (468 tokens: 380 ms, 470: 869 ms) and did not appear at all in a
  process set up on the glaive data, or in a repeat with memory figures printed, where the
  allocator's reserved pool reached 3.60 GB with the driver reporting 0 free while only
  1.26 GB was allocated. bitsandbytes' dispatch rule has no threshold there. **The cause is
  not established**; the dolly rows above are as measured, and section 2's dolly figure for
  this model is its first 30 rows.
- **The first model check refused a correct model.** It compared a prefix-tree row with the
  ordinary batch on random tokens under a fixed tolerance per dtype. Pretrained Qwen2.5-0.5B on
  random tokens in bf16 disagrees with ITSELF by 0.10 to 1.05 of its logit range (fp16: 0.005
  to 0.017; fp32: 1e-5), so no fixed tolerance separates a correct row from a wrong one. The
  shipped check uses two or three real training examples, measures the model's own layout
  noise (the batch against the same examples one at a time) and requires the row to stay
  within four times it and under a quarter of two deliberately wrong rows.
- **The held-out evaluation after `wrapper.train()` failed** on a streamed model ("buffer slot
  0 holds layer 22, but layer 0 was requested"): the layer stream is closed when `train()`
  returns. The harness evaluates in `on_train_end` instead. Not a prefix-tree matter; recorded
  because the first quality series was lost to it.
- **A first 7B attempt was refused by the trainer** ("no causal-loss target remains"): rows
  with a system message through Mistral's template, which renders none. Hence
  `--system-in-user`.
- One quality run on dolly was repeated on an idle card after a leftover process of this
  session was found on the GPU; the repeat agreed with it (154.9 s against 158.5 s).

## 5. Does it compute the same thing

**fp32, pretrained Qwen2.5-0.5B, GPU:** the row's logits differ from the ordinary batch's by
7e-6 to 1e-5 of the logit range (three seeds of random tokens), against 0.27 to 0.37 for
positions left wrong on purpose and 0.65 to 1.13 for a plain causal mask.

**CPU, fp64 / fp32, tiny random models** (`tests/test_prefix_tree.py`): logits bit-identical
on Llama, Mistral and Qwen2 and to 3e-16 on GPT-2; the float64 loss equal to 1e-12; gradients
to float rounding; the same through a layer-streamed model.

**bf16, the per-group loss computed more than one way** (worst relative difference):

| run | two ordinary layouts (`batch` against `one`) | `tree` against `batch` | `tree` against `one` |
|---|---|---|---|
| Mistral-7B, all five runs | not run | not run | 0.13 % to 0.38 % |
| Qwen2.5-0.5B, glaive one schema | 4.7 % | 2.5 % | 4.2 % |
| Qwen2.5-0.5B, glaive as published | 3.5 % | 2.3 % | 2.5 % |
| Qwen2.5-0.5B, dolly | 1.0 % | 1.6 % | 1.4 % |

On the 0.5B model two ordinary layouts of the same examples disagree by as much as the tree
disagrees with either; on the 7B model everything agrees to a few tenths of a percent.

**The model check as printed by the trainer** (row against batch / the model's own noise /
plain causal mask / wrong positions, in logit units): Mistral-7B, glaive one schema: 0.68 /
0.81 / 17.6 / 8.66 of a range of 31.5. Qwen2.5-0.5B, dolly: 1.25 / 1.66 / 28.1 / 22.6 of 30.8.

## 6. Limits

- One card. One seed. Held-out loss, not generation quality. The largest model is 7B.
- The gain is the data's and the row's: shared untrained tokens (the prompt, with
  `data.train_on_responses_only`, the default) and how many examples fit `max_length`. Rows
  that share nothing still fill a row.
- A step covers more examples, so an epoch has fewer optimizer steps; compare runs by wall
  time or by examples seen, and expect to revisit the learning rate or the epochs.
- Examples that share a beginning meet in one step, as with any grouping sampler.
- `task: sft`, text, single process. Refused beside `packing`, `multipack`,
  `stream_loss_chunk`, `data.format: raft`, `data.replay` and `data.interleave`.
- The 4D mask keeps scaled-dot-product attention off its flash path; a row of `max_length`
  tokens holds a `max_length` x `max_length` boolean mask.
