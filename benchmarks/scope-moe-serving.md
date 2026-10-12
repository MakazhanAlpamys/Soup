# Scope decision: who serves a giant MoE with a Soup adapter?

**FINAL scope study, 2026-10-09 — not product readiness.**
[Evidence and assumptions](scope-moe-serving-evidence.md) ·
[Prototype tasks](scope-moe-serving-prototype.md) ·
[Adapter measurements](probe-lora-hook-engines.md) ·
[Routing holdout](probe-moe-expert-holdout.md)

## Decision and ownership

**Variant 2: llama.cpp serves; Soup trains and exports the adapter.**
The engine owns execution, expert placement and SSD reads. Contribute the MLA
LoRA hook upstream; do not add an inference runtime to Soup.

Soup's layer streaming is training-only: no KV cache or expert-granular fetch,
and generation is deliberately refused. Variant 1 would require a new runtime.
llama.cpp already has native Windows, grouped DeepSeek-V3 routing, memory-mapped
GGUF and unmerged LoRA [CODE]. Alternatives fail the required OS, architecture,
SSD or adapter constraints; the [engine matrix](scope-moe-serving-evidence.md#3-engines-that-could-serve-deepseek-v3--kimi-k2-with-experts-on-ssd)
retains the exclusions and Linux-only fallbacks.

## Intended user workflow

On **native Windows 11, RTX 5070 Laptop 8 GB, 32 GB RAM and two NVMe drives**,
a user trains an attention adapter for DeepSeek-V3 671B or Kimi K2 ~1T,
optionally including the shared expert, then exports it separately and loads it
unmerged in llama.cpp. Routed experts stay frozen and may exceed RAM;
attention uses the GPU only where it fits.

The format is PEFT safetensors with HF DeepseekV3 targets `q_a_proj`,
`q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, optionally
`shared_experts.{gate,up,down}_proj`. **This workflow is not established:**
Soup still needs DeepSeek-V3 streamed training, FP8-checkpoint support and
adapter-only GGUF export. Use llama.cpp at or after `b11476` with the hook;
Soup's `b5270` export pin must move.

## Evidence and correctness gate

- A draft hook, **18 added / 13 removed lines**, closes four missing MLA LoRA
  paths on tiny SYNTHETIC DeepSeek-V3-shaped models. Real-config-only exports
  check rank and the hook's `k_b`/`v_b` shapes, not execution on real MLA weights.
- Real **Qwen3.5-35B-A3B has no MLA**. The
  [registered f32 CPU repeat](probe-lora-hook-engines.md#65-real-model-repeat-with-complete-controls)
  on stock `b11476` gives both SYNTHETIC variants APPLIED:
  `r = 1.4406e-5 / 8.2624e-6`, `n_base = 1.3040e-6`, top-1 agreement 1.0.
  All base/adapted repeats, tokens, disabled-reference/base and non-routed
  target controls pass. Historical bf16 remains VOID; historical f32 lacked
  adapted repeats. The [full result](results/probe-lora-hook-engines/part-b-real-model-run3.json)
  and [verification](results/probe-lora-hook-engines/part-b-run3-verification.json)
  establish transport on this CPU path, not trained-adapter quality or real MLA.
- Tiny f32 gate: `r ≤ 0.01`; dropped-effect threshold `ρ ≤ 0.01`.
  Part B gate: **`r ≤ t = 3f + 0.02`, `t ≤ 0.5`**, where
  `f = ‖z_eng⁰ − z_ref⁰‖ / ‖Δ_ref‖`, not the base-relative gap `n_base`.
  Require matching tokens, top-1 agreement ≥ 0.9, `n_base ≤ 0.05`,
  `s_ref ≥ 0.01`, exact disabled-adapter/base agreement, frozen routed experts,
  deterministic base and adapted logits, and successful conversion/loading.
  **Generation speed is not a criterion.**

The separate CUDA/BF16 routing holdout gives **17 HOTSET TRANSFERS and
1 GAIN WITH DRIFT** across 18 small-model arms [RUN]. This measures base-profile
transfer between data halves—not adapter drift, batch-1 decode reuse or SSD
savings. It does not complete P6.

## Conditions and unresolved work

Soup invokes llama.cpp as a separate **MIT-licensed program**; no engine code
enters Soup. The hook follows upstream's contribution process. Strata remains
**ideas only, no implementation code copied**. Model weights keep their own licences.

P3 must guard exact PEFT target paths: tested PEFT 0.21.2 can retarget shared
experts onto routed experts. Real MLA parity, quantised/GPU paths, upstream
acceptance and fit on the 8/32 GB box remain unproven. P4/P5 test correctness
and resources; P6 tests adapter-induced routing drift. None is a product promise.
