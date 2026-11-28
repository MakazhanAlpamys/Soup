# Measurement record - giant MoE serving scope and adapter repeat

Measured by @Batyrkhantzy111 on 2026-10-09.

## 1. Question and setup

Who should own batch-1 serving when DeepSeek-V3 671B or Kimi K2 (~1T) routed
experts may exceed RAM and a separately exported Soup LoRA must stay unmerged?
The design target was native Windows 11, an RTX 5070 Laptop GPU with 8 GB VRAM,
32 GB host RAM and two NVMe drives. The study did not run either giant model on
that laptop. Its GPU driver and CUDA version were unrecorded.

The measured repeat used real `Qwen/Qwen3.5-35B-A3B` weights at
`59d61f3ce65a6d9863b86d2e96597125219dc754`. The BF16 checkpoint was upcast to
FP32 in Transformers and the base GGUF. Qwen3.5-35B-A3B has no multi-head latent
attention (MLA). Both adapters were synthetic PEFT LoRA, seed 17, rank 8,
alpha 16 and initial B standard deviation 0.02. The variants used Soup's
`lora.target_modules: auto` and the same targets with the shared expert added.

The repeat ran CPU-only with 48 threads on an AMD EPYC 7B13 host (48 vCPUs,
377 GiB RAM, 200 GiB tmpfs), Ubuntu 22.04 Linux (kernel 6.8.0 build 1069,
glibc 2.35). The reference used Python 3.12.15, PyTorch 2.14.1+cpu,
Transformers 5.19.0, PEFT 0.21.2 and NumPy 2.5.3. The converter used PyTorch
2.11.0+cpu, Transformers 4.57.6, NumPy 2.2.6 and GGUF 0.19.0. The engine was
stock llama.cpp `b11476` (`988190680d5a89fce97de3c20df2c2813731fd61`),
built with GNU 11.4.0. GPU driver and CUDA were inapplicable to the repeat.

## 2. Method and commands

The scope review compared a Soup-native runtime with delegation to an engine.
It screened pinned engine versions for native Windows, grouped routing,
disk-backed experts and unmerged LoRA. A native runtime would need a KV cache,
router-driven expert fetch and eviction, and decode kernels. At the reviewed
llama.cpp revision, memory-mapped GGUF and Windows paging provided a candidate
disk path, with placement and reuse still to measure. The reviewed
ik_llama.cpp and KTransformers disk paths required Linux. The SSD-specific
alternatives reviewed had model, OS or adapter gaps.

The study estimated that the DeepSeek-V3 attention, shared-expert, router and
embedding weights would occupy about 9.6 GB at 4.5 bits per weight. This
exceeded the target VRAM even before cache and workspace allocations.

The real-checkpoint repeat gave both runtimes the same prompt tokens. The rule
required matching tokens, bitwise repeats of base and adapted logits within
each runtime, equality of disabled-adapter and base logits in the reference,
zero routed-expert adapter targets, and successful conversion and loading.
The numeric gates required base top-1 agreement ≥0.9, a base-relative gap
`n_base` ≤0.05, and a reference effect `s_ref` ≥0.01.

For the adapter-effect vectors `Δ`, error `r` is
`norm(Δ_engine - Δ_reference) / norm(Δ_reference)`. The floor `f` is the
base-logit gap divided by `norm(Δ_reference)`. The passing rule was
`r ≤ t = 3f + 0.02` with `t ≤ 0.5`. The protocol allowed one redraw at four
times the initial B scale for a weak reference effect. Neither variant needed
it. Generation speed was outside the rule.

The harness lives at the evidence commit and matches the measured source
revision `3e75c3af1d7aab7431c5ca9785dbf1fe0808bb30`. This command uses the
captured precision and thread settings with paths and the box label
substituted. The harness converts the base with `--outtype f32 --no-mtp`
and evaluates with `-fa off -ctk f32 -ctv f32`.

```bash
HF_XET_CHUNK_CACHE_SIZE_BYTES=0 <ref-venv>/bin/python \
  benchmarks/harness/lora_hook_real_model.py \
  --llama-bin <build>/bin --llama-src <llama-src> \
  --convert-python <convert-venv>/bin/python \
  --model-dir <checkpoint-dir> --work-dir /dev/shm/<scratch> --threads 48 \
  --box "<box>" --precision f32 \
  --out part-b-real-model-run3.json --log part-b-real-model-run3.log
```

## 3. Results and serving decision

| Adapter variant | Modules | `r` | `f` | `t` | `s_ref` | Result |
|---|---:|---:|---:|---:|---:|---|
| Soup auto | 80 | 1.4406289065096857e-5 | 1.1735580656080632e-5 | 0.020035206741968244 | 0.11111299846204006 | APPLIED |
| Auto plus shared expert | 200 | 8.262431261704828e-6 | 5.4927185512028305e-6 | 0.02001647815565361 | 0.23740075942989677 | APPLIED |

Base top-1 agreement was 1.0 across 22 tokens, with
`n_base = 1.3039755553902343e-6`. All registered controls passed. The saved
verification reports six finite engine arrays of shape `[22, 248320]` and
matching engine repeats. It records 36 audited artifacts, including the
adapter conversions and engine captures. The reference controls are preserved
as digests and verdicts in the result.

The study chose llama.cpp to own decode, expert placement and SSD reads.
Soup's role is training and exporting a separate PEFT adapter. The design
uses the engine as a separate MIT-licensed program. The proposed DeepSeek-family
targets are `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` and
`o_proj`, with optional `shared_experts.{gate,up,down}_proj`. The target lists
need exact path validation because the tested PEFT version could retarget
shared-expert requests onto routed experts.

**Verdict — mixed. Both synthetic adapter variants passed FP32 CPU transport on Qwen3.5, and the giant-model serving workflow remained unverified.**

## 4. Limitations and superseded runs

The completed repeat supersedes the earlier Qwen FP32 capture for the
transport verdict because it includes the adapted-logit repeats and exact
target controls. The earlier BF16 attempt remains void at the base-parity
gate. Earlier tiny-model MLA and real-configuration export checks cover
different questions. The real Qwen run uses one prompt and synthetic adapters.
Raw reference logits were absent from the archive.

The earlier tiny-model work established the need for an MLA hook on the
tested engine. The scope review treated upstream integration of that hook
and Soup's adapter-only GGUF export as pending. Real DeepSeek-V3 and Kimi K2
execution, quantized and GPU parity, and target-laptop fit remain outside
this measurement. A separate routing holdout measured cross-split transfer
on unadapted base models.
Adapter-induced routing drift, batch-1 decode reuse and SSD-read savings
remain open.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
