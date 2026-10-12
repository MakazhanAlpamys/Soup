# Prototype tasks after the MoE serving-scope decision

**Status: FINAL scope-study plan, 2026-10-09; not an implemented prototype.** These follow from
[`scope-moe-serving.md`](scope-moe-serving.md) (variant 2: llama.cpp serves, Soup
trains and exports, a small hook goes upstream). Each task writes its decision
rule before its first run, as every record in this directory does. None of them
promises generation speed.

The completed [real-Qwen f32 CPU repeat](probe-lora-hook-engines.md#65-real-model-repeat-with-complete-controls)
gives both SYNTHETIC adapter variants APPLIED on stock `b11476`, with all
base/adapted repeat, token, disabled-reference/base and non-routed-target
controls passed. This closes the real-Qwen adapted-determinism gap, not P1–P7:
Qwen has no MLA. Trained-adapter quality, real MLA, quantised/GPU paths,
memory/SSD fit on the target box and the full user workflow remain unproven.

| # | Task | Gate (rule written before the run) | Hardware | Depends on |
|---|---|---|---|---|
| P1 | **Upstream the MLA LoRA hook to llama.cpp.** Start from the measured draft ([`mla-lora-hook-draft.patch`](results/probe-lora-hook-engines/mla-lora-hook-draft.patch), 18+/13−): the batched factor transpose in `convert_lora_to_gguf.py`; `build_lora_mm` at `wq_a`, `wq_b`, `wkv_a_mqa`, the absorbed `wk_b` (main and MTP graphs) and both `wv_b` sites. Add llama.cpp's own tests, and settle with the maintainers whether the DeepSeek-V2-Lite `wq` and legacy `wkv_b` paths are in scope. Process: an issue first, AI use disclosed, every line understood by its author, PR text written by a human. | [`lora_hook_parity.py`](harness/lora_hook_parity.py) with the patched tag: every `dsv3-tiny` variant APPLIED, Qwen3.5 control unchanged. The same with a quantised base (Q8_0, Q4_K) and on the CUDA backend. | CPU for conversion and the CPU backend; one CUDA card for the GPU backend | — |
| P2 | **Soup: export the adapter alone, as GGUF LoRA.** A `soup export` path that calls `convert_lora_to_gguf.py --base` from a pinned llama.cpp (no base weights are loaded), and moves the auto-clone pin off `b5270`. Refuse by name any module the pinned engine does not apply, so that no adapter is dropped silently: today that is the four MLA projections on `deepseek2`, until P1 lands. | The probe harness driven through the Soup command instead of the raw converter: same verdicts as P1. The real-config export check ([`lora_export_real_configs.py`](harness/lora_export_real_configs.py), record §4d) driven the same way: PASS for all three configs from `config.json` alone. A tiny DeepSeek-V3 adapter on a pre-hook pin is refused with the module names in the message. | CPU | P1 for the full module set |
| P3 | **Soup: guard exact adapter target paths before allocation.** At configuration load, build a meta-model from `config.json`, apply Soup's normal target resolution, and compare the original requested tensor paths with those selected after PEFT conversion on a copy of `LoraConfig`. Refuse mismatches with requested and actual names before checkpoint loading and `get_peft_model`. After [PEFT #3715](https://github.com/huggingface/peft/pull/3715) is released, raise the PEFT floor to that actual fixed release; keep the guard as a safeguard. Details and limits below. | On the real DeepSeek-V3 config on meta: peft 0.21.2 rejects `shared_experts.{gate,up,down}_proj`, naming requested shared-expert and actual routed-expert tensors; attention-only passes; `gate` is identified as routers, not routed experts. With PEFT containing #3715, the same shared-expert list passes and selects only shared-expert paths. | CPU, config only; no checkpoint load or adapter allocation | — |
| P4 | **Correctness on real MLA weights.** No 671B reference fits a laptop or this stage's budget. Use the first few layers of DeepSeek-V3 and Kimi K2 with real weights dequantised from FP8, plus the embedding and head. Run Part B through patched llama.cpp in f32 and Q4_K. The registered full Qwen3.5-35B-A3B f32 repeat passes both adapters with all controls (§6.5), but Qwen has no MLA; historical bf16 remains VOID. In Q4_K, use reference weights dequantised from the same GGUF: comparing against original weights adds quantisation error to `f` and may make `t > 0.5`, not necessarily so. SYNTHETIC adapters provide controls, not a completed trained workflow. A real Soup-trained DeepSeek-V3/K2 adapter still requires streamed training and FP8-checkpoint support. | Full Part B rule and controls: matching tokens, exact disabled-adapter/base agreement, frozen routed experts, deterministic base and adapted logits, successful conversion/loading, `s_ref ≥ 0.01`, `n_base ≤ 0.05`, top-1 agreement ≥ 0.9, `r ≤ t = 3f + 0.02 ≤ 0.5`. State whether the adapter is SYNTHETIC or real Soup-trained. | cloud CPU instance with ≥ 128 GB RAM | P1; separately planned DeepSeek-V3 streamed training and FP8-checkpoint support for a real Soup-trained adapter, P3 for its shared-expert variant |
| P5 | **SSD serving smoke on the dev box with the adapter, in two arms.** (a) **Full model from SSD, load and resources only:** DeepSeek-V3 or K2 as a low-bit GGUF memory-mapped from NVMe, attention partly on the 8 GB GPU, adapter loaded unmerged. Its adapter effect is not compared with P4's: P4's truncated checkpoint is a different model, and the layers it drops change both the base logits and the adapter's effect. (b) **Numerical match on the dev box:** P4's checkpoint, adapter and tokens, on the same llama.cpp build and backends as arm (a), against P4's reference. Record adapter provenance in both arms: SYNTHETIC controls are allowed, but the full user workflow requires a real Soup-trained DeepSeek-V3/K2 adapter and the separately planned streamed-training and FP8-checkpoint support. No speed promise. | (a) Runs to completion without spilling into shared GPU memory; the adapter is not lost: a non-zero effect for each module on its own, against a negative control; peak VRAM and RAM, the read rate and the versions recorded, and the box state as in the house rules. (b) P4's rule, with `f` measured on this build: `r` ≤ 3f + 0.02, `t` ≤ 0.5, top-1 agreement ≥ 0.9. | the dev box (RTX 5070 Laptop 8 GB, 32 GB RAM, NVMe, native Windows) | P1, P2; P4 for arm (b); separately planned DeepSeek-V3 streamed training and FP8-checkpoint support for a real Soup-trained adapter, P3 for its shared-expert variant |
| P6 | **Does a trained adapter shift routing enough to stale an engine's hot-expert profile?** Engines place or cache experts from a routing profile: Strata's VRAM cache, KTransformers' frequency placement, llama.cpp RFC #24528. Compare routed top-k on identical teacher-forced tokens under the base and a real Soup-trained attention adapter on Qwen3.5-35B-A3B: per-layer set overlap and the held-out hit-rate loss of a base-model profile. Fit profiles on one half of the measured steps and score only the other half, in both directions, following the [separate holdout record](probe-moe-expert-holdout.md); fitting and scoring on all the same steps gives the old Finding 3 identity, not stability evidence. On held-out steps, compare the base profile on base versus adapted routes and the adapter-refitted profile. Cache capacities both as fractions of E and as multiples of k: `{k, 2k, 4k, 8k} ∪ {E/8, E/4, E/2}`, below E, plus the share of DeepSeek-V3's and K2's routed experts that the dev box's RAM holds: 3-5% if about 18 GB of its 32 GB is left for them (18/368 GB = 4.9% for DeepSeek-V3, 18/571 GB = 3.2% for K2), 8.7% and 5.6% if all 32 GB were [ESTIMATE]. Report hit, gain over random `c/E`, per-step oracle, and transfer loss on matched layer populations including layer 0; distinguish ordinary profile/sampling drift from adapter-induced loss. | Written before the run, with these rules: frozen splits and thresholds, saved per-step/layer/expert counters; a stated outcome if a model fails to run; a fixed-seed per-token random permutation or SYNTHETIC-route negative control whose expected profile hit is `c/E`; paired precision comparisons on the same token ids (generated sequences replayed from the reference precision); a verdict on an NF4 model only for NF4 unless bf16↔NF4 was paired on it; each predictor's tensor and normalisation stated, including d = 0. The completed base-only small-model holdout gives 17 HOTSET TRANSFERS and 1 GAIN WITH DRIFT; it does not establish adapter drift or giant-model batch-1 decode reuse and does not complete P6. | one GPU for the adapter's training; the routing measurement itself runs on CPU | P3 for a shared-expert variant |
| P7 | **Expert prefetch for llama.cpp's SSD path on Windows**, only if P5's full-model arm shows page faults on routed experts dominate. Prefetch the next layer's likely experts with `PrefetchVirtualMemory`, predicted from the current hidden state by the next layer's router, as Colibrì and HOBBIT do. Proposed upstream, not built in Soup. | Interleaved arms on the dev box (with and without), rule written before the run; a correct prefetch hides about 5% on GLM-5.2 per SSD-LLaMA, so a small effect is the expected one. | the dev box | P5 |

## P3: exact paths before weights or adapters allocate

The failure is tracked in [PEFT #3711](https://github.com/huggingface/peft/issues/3711).
As of 2026-10-08, maintainer BenjaminBossan's
[PR #3715](https://github.com/huggingface/peft/pull/3715) is open and unmerged
[DOC]. Its inspected source preserves a target if it matches at least one
module and **all** matching modules are `nn.Linear`; shared-expert tests cover
that case, while mixed/broad, legacy layer-specific and fused-selection tests
remain xfailed [CODE]. Do not treat a version bump as proof of arbitrary
target-path correctness.

1. At configuration load, **before checkpoint loading and before
   `get_peft_model`**, instantiate the base architecture on `meta` from
   `config.json`, following
   [`meta_model`](harness/lora_export_real_configs.py). No base weights are
   needed for names, types and shapes.
2. Apply the same target resolution as Soup's training path, including
   `auto`, explicit lists or regexes, exclusions, layer restrictions and
   configured parameter targets. Resolve the original requested selection to
   exact base tensor paths, not just target suffixes.
3. Preserve that request and run `convert_peft_config_for_transformers` on a
   **copy of `LoraConfig`**: the function mutates it in place. Resolve the
   converted final `target_modules` and `target_parameters` against the same
   meta-model to exact base tensor paths.
4. Compare requested and final path sets. Reject changes with both sets of
   names, including `mlp.shared_experts.{gate,up,down}_proj.weight` or dense
   `mlp.{gate,up,down}_proj.weight` silently redirected to
   `mlp.experts.{gate_up_proj,down_proj}`. Do **not** refuse merely because
   `target_parameters` is nonempty: conversion also turns `gate` into
   `gate.weight`, a router weight, not a routed-expert tensor. Keep module
   targets and their corresponding tensor paths distinct in the diagnostic.
5. Once a release containing #3715 is published, raise Soup's
   `peft>=0.20.0,<1.0.0` floor to that actual fixed release, without predicting
   its version. Retain the path guard afterwards for unresolved conversion
   cases and future changes.

An inspection after `get_peft_model` can be too late on a giant model: all
unintended routed-expert adapters can allocate first. Memory-budget checking
alone cannot prove the adapter is on the intended tensors. The correctness
guard may share one meta-model traversal with budget checking; neither check
needs to load the checkpoint [INFERENCE].

The P3 gate is config-only on the **real DeepSeek-V3 config**, not a published
meta timing or a real trained adapter: peft 0.21.2 must reject a qualified
shared-expert list with the requested and actual tensor names; attention-only
must pass; `gate.weight` must be classified as routers. With PEFT containing
#3715, that same shared-expert list must pass and select only shared-expert
paths. Broad regexes and layer-specific fused requests still require path
comparison; the upstream PR does not guarantee them.

Order: P1 and P3 can start at once and run in parallel. P2 waits on P1 for the
DeepSeek modules but can ship the Qwen3.5 path earlier. P4 gates P5's numerical
arm, and P7 is conditional on P5's full-model arm.
