# Gate: expert-granularity streaming is bit-exact against the resident MoE block

Status: **PASS**: first run 2026-10-14 18:36Z (harness at 06fe28f16), re-run at the reviewed tip f5281f96f (2026-10-14T19:04:45Z, after the reader gained its size checks), PASS again. The rule in section 2 was committed in 74f0a08b, before the first run.

## 1. Question

`utils/expert_stream.py` replaces a MoE layer's routed-experts module with
`StreamedExperts`, which reads one expert at a time from a per-layer experts file
and fetches it again in the backward. Does it give the same numbers as the
resident experts module on a REAL MoE checkpoint, not only on the random tiny
models of `tests/test_expert_stream.py`?

## 2. Decision rule (written before any run)

Model: `allenai/OLMoE-1B-7B-0924` from the local Hugging Face cache (64 experts,
top-8, 16 layers; already on disk since the #992 coverage probe). Layers 0, 7
and 15. For each layer:

- the resident control is transformers' `OlmoeExperts` holding that layer's real
  expert weights, run with `experts_implementation="eager"`;
- the streamed arm is `StreamedExperts` over an experts file written by
  `write_expert_shard` from the same tensors;
- the routing is the layer's real router (`mlp.gate.weight`) applied to the
  input: softmax then top-8, as `OlmoeTopKRouter` does;
- the input is 512 tokens of N(0, 1) hidden states (seed 1234), scaled by the
  layer's `post_attention_layernorm` weight (the norm the MoE block reads), and
  an N(0, 1) output gradient (seed 4321).

Harness: `benchmarks/harness/expert_stream_gate.py`; raw JSON under
`benchmarks/results/probe-rtx5070/expert-stream/`.

Cells: (a) CPU fp32, unquantised; (b) CUDA bf16, unquantised; (c) CUDA bf16 with
every expert matrix NF4-quantised by the sharder, the control holding the
dequantised weights. Each cell compares, with `torch.equal`: the output, the
gradient of the input, and the gradient of the routing weights.

- **PASS**: `torch.equal` holds for all 3 layers x 3 cells x 3 tensors (27 checks),
  and the streamed arm read each routed-to expert exactly twice (forward +
  backward) and no other expert.
- **FAIL**: any check false. The record then states the max absolute difference
  per failing tensor; no tolerance is substituted after the fact.

Not measured here: speed (expert streaming promises none; the #992 probe found a
step routes to 96.7-100% of experts), a whole-model step, any DeepSeek-V3 weights.

## 3. Box state

- 2026-10-14 18:36:59Z; AC power on; 10.8 GB RAM free before the run.
- GPU: 0 MiB in use before the run; the harness saw 7,013 of 8,123 MiB free once CUDA was up.
- Foreign activity: a 32 GB model download writing to D: (this lane's own). No other session ran pytest or used the GPU; the run is a correctness check, so timing noise does not affect the verdict.
- Software: Python 3.12.10, Windows 11 26200, torch 2.14.0+cu130, transformers 5.17.0, bitsandbytes 0.50.2, safetensors 0.8.0.
- Harness commit 06fe28f16 for the first run; f5281f96f for the re-run, whose JSON is the one committed (`benchmarks/results/probe-rtx5070/expert-stream/gate.json`). Both runs: 27/27 equal, every fetch count 2 x routed.

## 4. Results

Every one of the 27 checks is `torch.equal`, max abs difference 0.0:
`output`, `grad_input` and `grad_routing` x cells `cpu-fp32`, `cuda-bf16` and `cuda-bf16-nf4` x layers 0, 7 and 15.

| layer | cell | experts routed to | expert reads | reads == 2 x routed |
|---|---|---|---|---|
| 0 | cpu-fp32 | 64 / 64 | 128 | yes |
| 0 | cuda-bf16 | 64 / 64 | 128 | yes |
| 0 | cuda-bf16-nf4 | 64 / 64 | 128 | yes |
| 7 | cpu-fp32 | 64 / 64 | 128 | yes |
| 7 | cuda-bf16 | 64 / 64 | 128 | yes |
| 7 | cuda-bf16-nf4 | 64 / 64 | 128 | yes |
| 15 | cpu-fp32 | 64 / 64 | 128 | yes |
| 15 | cuda-bf16 | 64 / 64 | 128 | yes |
| 15 | cuda-bf16-nf4 | 64 / 64 | 128 | yes |

512 tokens routed to all 64 experts in every layer, which matches the #992 coverage
probe (96.7-100%). The JSON also holds the wall seconds of each arm. They are NOT a
timing measurement: one run each, first-call warm-up included (layer 0's resident
CUDA arm, 6.1 s, is CUDA start-up). The rule asks for no speed figure.

## 5. Verdict

**PASS.** On real OLMoE-1B-7B weights, `StreamedExperts` equals the resident
eager experts block bit for bit:

- forward output, input gradient and routing-weight gradient;
- fp32 on CPU, bf16 on CUDA, and NF4 against the dequantised control;
- each routed-to expert read exactly twice per forward + backward, and no other expert read.

Not shown here: whole-model steps on real DeepSeek-V3-family weights (the tiny-model
parity tests in `tests/test_expert_stream.py` cover the whole-model path), and speed.
