<!--
Measurement record, published as measured, in the order things happened.

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 581.15; 15.8 GB host RAM;
Windows 10 (10.0.19045). One machine, one session, 2026-11-13.
Stack: Python 3.11.5, torch 2.14.1+cu130, transformers 5.22.2, peft 0.21.2, trl 0.29.1,
bitsandbytes 0.50.2.
Model: Qwen/Qwen2.5-0.5B-Instruct (model.safetensors 988,097,824 bytes, downloaded
2026-11-13, revision not pinned), NF4 with double quantisation, LoRA r 8 on q_proj / v_proj,
dropout 0.
Data: Soup's own tool-calling fixture, src/soup_cli/data/_fixtures/gate/tool_call.jsonl
(40 rows), turned into system / user / assistant rows by the harness.
Harness: benchmarks/harness/prefix_tree_ab.py. Raw output: results/prefix-tree-rtx3050-v1/.
-->

# Gate record — `training.prefix_tree`: tokens several examples share are computed once

**Status: MEASURED on one laptop (RTX 3050 Laptop 4 GB), one small model.** Through the
shipped trainer, layer-streamed NF4, on 40 tool-calling examples that share one tool schema: a
pass over the data takes **2.01 s as prefix trees against 19.07 s one example per step**
(19.9 against 2.10 examples/s, 9.5x), at a peak of 1.26 GB. The ordinary batch holding the
same examples per step needs 5.85 GB, over the card. On the fixture as shipped (27 different
tool lists in 40 rows) it is 6.99 s against 17.97 s (2.6x). `soup`'s own training loop: three
epochs in 7.06 s against 58.05 s. Loss agreement with the ordinary batch is exact in fp32
(1e-5 of the logit range) and inside this model's own layout noise in bf16, which is large
(section 4). **Not measured: an 8B model, any other card, model quality on held-out data.**

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

## 2. Three ways to run the same examples (the A/B)

`prefix_tree_ab.py` builds the model, tokenizer, dataset and collator through
`SFTTrainerWrapper.setup` and runs every planned group (one prefix-tree row's examples) three
ways: `one` (each example as its own step, what a card that fits batch 1 does today),
`batch` (the ordinary batch of the group) and `tree`. Forward + backward through
`trainer.training_step`, no optimizer step, so the three arms see the same weights. Three
rounds, arm order rotated, medians. `max_length` 512, `batch_size` 1.

| run | tokens, ordinary -> tree | arm | steps | pass | examples/s | peak VRAM |
|---|---|---|---|---|---|---|
| streamed NF4, one schema | 9,147 -> 1,975 (4.63x), 4 rows | one | 40 | 19.07 s | 2.10 | 0.77 GB |
| | | batch | 4 | 14.54 s | 2.75 | **5.85 GB** (over the card) |
| | | tree | 4 | **2.01 s** | **19.92** | 1.26 GB |
| streamed NF4, fixture as shipped | 8,782 -> 5,887 (1.49x), 14 rows | one | 40 | 17.97 s | 2.23 | 0.77 GB |
| | | batch | 14 | 14.75 s | 2.71 | **4.88 GB** (over the card) |
| | | tree | 14 | **6.99 s** | **5.72** | 1.18 GB |
| resident NF4, one schema | 9,147 -> 1,975 (4.63x), 4 rows | one | 40 | 17.14 s | 2.33 | 1.75 GB |
| | | batch | 4 | out of memory | | |
| | | tree | 4 | **5.89 s** | **6.79** | 2.73 GB |

- "One schema" gives every row the fixture's longest tool list as its system message: the
  shape of a dataset written against one tool schema. "As shipped" keeps each row's own list.
- The streamed tree arm is faster than the token ratio (9.5x for 4.63x fewer tokens, 2.6x for
  1.49x) because a streamed step also pays a per-sweep cost, and 4 sweeps replace 40.
- The `batch` arm is the fair statement of what the tree replaces: the same examples per step
  and the same gradient. On this card it does not fit (Windows lets the allocation spill past
  4 GB into shared memory and the arm's times spread by 23 to 150 % between rounds; resident,
  it raised out-of-memory), so its times are not a baseline.
- The `one` arm of the first run spread 98 % between rounds (a cold first round); the other
  `one` and all `tree` arms spread 0.8 to 6 %.
- The resident `one` and `tree` arms are slower per token than the streamed ones here; that is
  the resident 4-bit path, not this change, and is not investigated in this record.

## 3. The training loop a user runs

`prefix_tree_ab.py --train` calls `wrapper.train()`. Streamed NF4, one schema, lr 1e-4, seed 3.

| | epochs | optimizer steps | train runtime | logged loss, first -> last step | peak VRAM |
|---|---|---|---|---|---|
| `prefix_tree: false` | 3 | 120 | 58.05 s | 2.382 -> 0.463 | 0.86 GB |
| `prefix_tree: true` | 3 | 12 | **7.06 s** | 2.027 -> 0.967 | 1.26 GB |
| `prefix_tree: true` | 24 | 96 | 52.56 s | 2.027 -> 0.158 | 1.26 GB |

Read this with care. With `prefix_tree: true` a step holds ten examples here, so an epoch is
4 optimizer steps instead of 40: the same data in an eighth of the time, and a tenth of the
updates. At the same learning rate, three epochs therefore end at a higher loss (0.967 against
0.463). Given the same wall time instead (24 epochs, 52.6 s against 58.0 s) the logged loss
ends lower (0.158). These are single runs, the loss is the last step's training loss on 40
memorisable rows, and nothing here measures quality on held-out data. What the table does
show is that the shipped loop runs on prefix-tree batches at the measured rate.

## 4. Does it compute the same thing

**fp32, pretrained Qwen2.5-0.5B, GPU:** the row's logits differ from the ordinary batch's by
7e-6 to 1e-5 of the logit range (three seeds of random tokens), against 0.27 to 0.37 for
positions left wrong on purpose and 0.65 to 1.13 for a plain causal mask.

**CPU, fp64 / fp32, tiny random models** (`tests/test_prefix_tree.py`): logits bit-identical
on Llama, Mistral and Qwen2 and to 3e-16 on GPT-2; the float64 loss equal to 1e-12; gradients
to float rounding; the same through a layer-streamed model.

**bf16 is not reproducible across layouts on this model, with or without a tree.** Per group,
the loss computed three ways (first round):

| run | `batch` against `one` (two ordinary layouts) | `tree` against `batch` | `tree` against `one` |
|---|---|---|---|
| streamed NF4, one schema | up to 2.2 % | up to 3.0 % | up to 1.1 % |
| streamed NF4, fixture as shipped | up to 5.5 % | up to 4.0 % | up to 4.0 % |
| resident NF4, one schema | (batch out of memory) | | up to 0.9 % |

On the fixture as shipped the per-group differences have both signs for the tree (-1.9 % to
+4.0 % against `one`) and for the ordinary batch (-4.2 % to +5.5 %). The tree disagrees with
an ordinary layout by no more than two ordinary layouts disagree with each other.

**The model check, and why it uses real examples.** Before the first step the trainer runs two
or three training examples as an ordinary batch and as one prefix-tree row and compares the
logits, with the model's own layout noise measured the same way (the batch against the examples
one at a time) and two deliberately wrong rows as controls. A first version used random tokens
and a fixed tolerance per dtype. It refused this model: on random tokens, in bf16, pretrained
Qwen2.5-0.5B disagreed with ITSELF by 0.10 to 1.05 of its logit range (fp16: 0.005 to 0.017;
fp32: 1e-5), so no fixed tolerance could separate a correct row from a wrong one. On real
training examples, the shipped check printed for the streamed run: row against batch 1.34 of a
range of 34, the model's own noise 1.28, plain causal mask 21.4, wrong positions 18.3.

## 5. Limits

- One card, one model of 0.5B parameters, one dataset of 40 rows. No 8B run: the gain is a
  token count, and the streamed step is close to linear in tokens, but that is an argument,
  not a measurement.
- The gain is the data's. Rows that share no untrained beginning get nothing from sharing (a
  row can still hold several short unrelated examples). It needs prompt tokens that are not
  trained on, which is `data.train_on_responses_only`, the default.
- Examples that share a beginning meet in one step, as with any grouping sampler; and a step
  covers more examples, so an epoch has fewer optimizer steps (section 3).
- `task: sft`, text, single process. Refused beside `packing`, `multipack`,
  `stream_loss_chunk`, `data.format: raft`, `data.replay` and `data.interleave`.
- The 4D mask keeps scaled-dot-product attention off its flash path; a row of `max_length`
  tokens holds a `max_length` x `max_length` boolean mask.
