# Note: can a 24-32 GB card stream a whole Kimi K2 layer without expert streaming?

Arithmetic only: nothing here was built or measured. Shapes from
`moonshotai/Kimi-K2-Instruct`'s `config.json` (hidden 7168, 384 routed experts of
`moe_intermediate_size` 2048, one shared expert, MLA attention, 61 layers).

## One MoE layer

| part | parameters | bf16 | NF4 (+ fp32 absmax per 64) |
|---|---|---|---|
| routed experts, 384 x 3 x 7168 x 2048 | 16.91 B | 33.8 GB | 8.5 GB + 1.06 GB = 9.5 GB |
| attention (q_a, q_b, kv_a, kv_b, o) + shared expert + router | ~0.15 B | 0.29 GB | ~0.09 GB |

## Layer streaming as it is today (`expert_layout` fused)

The routed experts are 3-D parameters, not `nn.Linear` modules, so layer
streaming's NF4 path does not touch them: a K2 layer streams at bf16, **33.8 GB**,
and the pool holds at least two layer buffers (`stream_buffers` >= 2). That is
68 GB of buffers: **no on 24 GB, no on 32 GB**, and no on an 80 GB card either
once activations, the vocabulary slot (163840 x 7168 bf16 = 2.35 GB) and the
transformers expert kernel's own working set are added.

## If the fused experts were stored NF4 (not built)

Two NF4 layer buffers are 19 GB, plus the vocabulary slot (2.35 GB), activations
and LoRA state (a few GB at seq 512). The compute still needs bf16 expert
weights: dequantising the whole layer at once is the 33.8 GB above again, so it
would have to dequantise one expert at a time, which is the compute half of
expert streaming anyway.

- 32 GB card: plausible, ~23-25 GB peak (estimate).
- 24 GB card: at the edge; one buffer less would be needed, and the current
  floor is two.
- Host side: the disk tier's staging is `stream_read_ahead` layers, 2 x 9.5 GB
  page-locked, on top of the resident extras.

## What expert streaming changes

The streaming unit becomes one expert (24.8 MB at NF4); the pool holds the dense
part of a layer (~0.1 GB) and one expert at a time. The VRAM bound no longer
depends on the expert count. The bytes read per step do not shrink: a training
step routes to ~97-100% of the experts (`probe-moe-expert-coverage.md`), and the
backward reads them again. L2L (Pudipeddi et al. 2020) is the published method
for amortising those reads across micro-batches; it is not used here.

**Answer:** without expert streaming, no card from 24 to 32 GB holds a K2 layer
as layer streaming stores it today (bf16 experts, 33.8 GB). With NF4 experts and
per-expert dequantisation, a 32 GB card plausibly could; a 24 GB card would be at
the edge. Both figures are estimates.
