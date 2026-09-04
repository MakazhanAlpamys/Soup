# Supported Models & Optional Extras

[← Back to the Soup README](../README.md)

> Recommended model families, the VRAM size guide, and the pip extras matrix.

## Supported Models

Soup works with **any** of the **340,000+** text-generation models on [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). If a model supports `AutoModelForCausalLM`, it works with Soup — zero config changes needed.

### Recommended Models

| Model Family | Models | Sizes | Best For |
|---|---|---|---|
| **Llama 4** | Llama-4-Scout-17B, Llama-4-Maverick-17B | 17B | General, multilingual |
| **Llama 3.x** | Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct | 1B–70B | Chat, instruction following |
| **Llama 3.2 Vision** | Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision | 11B–90B | Image understanding |
| **Gemma 3** | Gemma-3-4B-IT, Gemma-3-9B-IT, Gemma-3-27B-IT | 4B–27B | Efficient, multilingual |
| **Qwen 3.5 / 3.6 / 3.8** | Qwen3.5-0.8B…397B-A17B, Qwen3.6-27B/35B-A3B, Qwen3.8-27B | 0.8B–397B | 262K context, native vision, MoE |
| **Qwen 3** | Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B | 0.6B–235B | Reasoning, code, MoE |
| **Qwen 2.5** | Qwen2.5-7B-Instruct, Qwen2.5-Coder-32B-Instruct, Qwen2.5-Math-7B-Instruct | 0.5B–72B | Code, math |
| **DeepSeek** | DeepSeek-R1-Distill-Llama-8B, DeepSeek-V3-0324, DeepSeek-V4-Flash/Pro | 1.5B–1.6T | Reasoning (GRPO), code, MoE |
| **GLM** | GLM-5, GLM-5.1 | 9B–754B | Chinese + English, MoE |
| **Kimi** | Kimi-K2, Kimi-K2.5, Kimi-K2.6 | ~1T (MoE) | Long-context agentic, MoE |
| **MiniMax** | MiniMax-M2, MiniMax-M3 | 230B–428B | Agentic, MoE (community license) |
| **Phi-4** | Phi-4-14B, Phi-4-mini-reasoning | 3.8B–14B | Compact reasoning |
| **Mistral** | Mistral-7B-Instruct-v0.3, Mistral-Small-24B, Mistral-Large-3 | 7B–675B | Fast, efficient, MoE |
| **Mixtral** | Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B | 47B–141B | MoE architecture |
| **CodeLlama** | CodeLlama-7b-Instruct-hf, CodeLlama-34b-Instruct | 7B–34B | Code generation |
| **StarCoder 2** | StarCoder2-15B, StarCoder2-7B | 3B–15B | Code completion |
| **Yi** | Yi-1.5-34B-Chat, Yi-1.5-9B-Chat | 6B–34B | Multilingual chat |
| **InternLM 3** | InternLM3-8B-Instruct | 8B | Chinese + English |
| **Falcon** | Falcon-11B, Falcon-40B-Instruct | 7B–180B | Open-weight |

Qwen3.5, Qwen3.6, and Qwen3.8 checkpoints advertise a multimodal conditional-generation
architecture on the Hub, but Soup's catalog recipes are deliberately text-only.
Their explicit `modality: text` selects `AutoModelForCausalLM`, which instantiates the
language decoder without the visual tower. This is appropriate for text-only SFT,
pre-training, and GRPO data; it is not a full multimodal fine-tune. The Transformers
backend requires Transformers 5.16.1 or newer, which also supplies the
`qwen4_exp` text decoder used by Qwen3.8-Flash-Next (#571). On Torch runtimes
whose `scatter` operator still requires `int64` indices, Soup widens only the
QSA indexer's local index tensors before LoRA injection; Torch and the upstream
Transformers class remain unchanged.

With `training.stream_layers: true`, the Qwen4-Exp text decoder is admitted by
its own parity-tested path. Its very large frozen PLE N-gram table is excluded
from the decoder-layer shard and can be served from the original safetensors
with `training.stream_ngram_source: disk`. Reads are sparse and read-only; Soup
does not create a second dense PLE copy in its shard cache.

Dense Transformers checkpoints and oMLX/oQ affine Qwen4 bundles are accepted by
this layer-streamed SFT path. For oQ, Soup dequantizes each frozen decoder layer
into the reusable stream cache and maps the fused Switch-MLP expert tensors to
the Transformers text decoder. The packed PLE table remains in the original
checkpoint and only selected rows are dequantized, so oQ requires
`stream_ngram_source: disk` (or `auto`). Vision-tower and MTP weights are not
part of the text-only CausalLM and are ignored. `training.quantization` must
remain `none`; streamed NF4 parity is separate and is not validated for Qwen4.

The Qwen4 resident-versus-streamed parity oracle currently covers float32 CPU.
BF16 parity on CUDA has not been measured, so CUDA BF16 production readiness is
still pending even though the runtime selects BF16 there. The 176.9B-parameter
production oQ checkpoint completed cache construction and training setup on an
M4 Max but not a full optimizer step; it is not validated as trainable on a
128 GiB Mac. See the [M4 Max gate record](../benchmarks/gate-qwen4-ple-m4-max.md).

### Vision Models (with `modality: vision`)

| Model | Size | Supported Formats |
|---|---|---|
| LLaMA-3.2-11B-Vision-Instruct | 11B | LLaVA, ShareGPT4V |
| Qwen2-VL-7B-Instruct | 7B | LLaVA, ShareGPT4V |
| Pixtral-12B-2409 | 12B | LLaVA, ShareGPT4V |

### ASR Models (`task: asr`, Whisper — v0.71.32)

| Recipe | Base | Size | Status |
|---|---|---|---|
| `whisper-tiny-asr` | openai/whisper-tiny | 39M | Live on 4 GB |
| `whisper-base-asr` | openai/whisper-base | 74M | Live on 4 GB |
| `whisper-large-v3-asr` | openai/whisper-large-v3 | 1.5B | Parse-only (larger GPU) |

Rows are `{"audio": <path>, "text": <transcript>}` with `data.format: asr`. See
[Training → ASR fine-tuning](training.md).

### Quick Size Guide

| VRAM | Max Model (QLoRA 4-bit) | Example |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

> **Note:** Soup auto-detects your GPU and estimates the optimal batch size. Use `soup doctor` to check your setup.

## Optional Extras

> **Python 3.10, 3.11 or 3.12.** Since v0.73.0 the package declares
> `requires-python = ">=3.10,<3.13"`. Those are exactly the versions CI tests. Without the
> upper bound, pip on 3.13+ resolved PyTorch wheels nobody had validated, and the failure was
> not a Soup error message — it was a loader crash inside `c10.dll` / `libc10.so` before any
> Soup code ran. If you are on 3.13+, create a 3.12 environment; support widens when CI does.

### Quoting the extra

**Use double quotes.** `pip install "soup-cli[train]"` is the only spelling that works in every
shell — `cmd.exe`, PowerShell, bash, and zsh. Every command in the table below uses it.

Older tutorials and videos (including some of ours) show the single-quoted
`pip install 'soup-cli[train]'`. That is bash / zsh / PowerShell syntax, and it fails on Windows
`cmd.exe`, which has no single-quote quoting and hands the quotes straight to pip:

```
ERROR: Invalid requirement: "'soup-cli[train]'": Expected package name at the start of dependency specifier
```

If you hit that, swap the `'` for `"` — pip is rejecting a literal quote character, nothing is
wrong with the package. (Dropping the quotes entirely works on Windows too, but zsh then reads
`[train]` as a glob and fails.)

### The extras table

The core `pip install soup-cli` is a light install — the CLI, config system, and data tools, with
no PyTorch. Add `[train]` to fine-tune, or install other extras only when you need them:

Every row below is written with `pip`, and works verbatim inside a virtualenv, a Colab notebook
or a Docker image. Outside one, on Debian 12 / Ubuntu 23.04 or later, `pip` refuses with
`error: externally-managed-environment`; substitute `pipx` or `uv tool` for `pip` and the extras
spelling is unchanged. See [the README's install section](../README.md#1-install).

| Extra | Install | What it adds |
|---|---|---|
| `train` | `pip install "soup-cli[train]"` | Training stack: torch, transformers, peft, trl, datasets, bitsandbytes, accelerate |
| `all` | `pip install "soup-cli[all]"` | `train` + `serve` + `ui` + `data` in one shot |
| `fast` | `pip install "soup-cli[fast]"` | Unsloth backend (2-5x faster, lower VRAM) |
| `vision` | `pip install "soup-cli[vision]"` | Vision / multimodal fine-tuning (Pillow) |
| `audio` | `pip install "soup-cli[audio]"` | Audio / speech fine-tuning (librosa, soundfile) |
| `mlx` | `pip install "soup-cli[mlx]"` | Standalone Apple Silicon SFT backend for local data; `[train]` is not required |
| `qat` | `pip install "soup-cli[qat]"` | Quantization-Aware Training (torchao) |
| `serve` | `pip install "soup-cli[serve]"` | Inference server (FastAPI + uvicorn) |
| `serve-fast` | `pip install "soup-cli[serve-fast]"` | vLLM inference backend (2-4x throughput) |
| `sglang` | `pip install "soup-cli[sglang]"` | SGLang inference backend |
| `ui` | `pip install "soup-cli[ui]"` | Web UI + inference server |
| `tui` | `pip install "soup-cli[tui]"` | Full-screen Textual dashboard (`soup tui`) |
| `eval` | `pip install "soup-cli[eval]"` | Benchmark evaluation (lm-evaluation-harness) |
| `aider` | `pip install "soup-cli[aider]"` | Aider CLI; Polyglot evaluation also needs Aider's source-built Docker image |
| `data` | `pip install "soup-cli[data]"` | Deduplication (MinHash via datasketch) |
| `data-pro` | `pip install "soup-cli[data-pro]"` | Language detection + PII (langdetect, presidio) |
| `deepspeed` | `pip install "soup-cli[deepspeed]"` | Multi-GPU training (DeepSpeed ZeRO) |
| `liger` | `pip install "soup-cli[liger]"` | Liger Kernel fused ops |
| `ring-attn` | `pip install "soup-cli[ring-attn]"` | Ring FlashAttention (sequence parallelism) |
| `onnx` / `tensorrt` | `pip install "soup-cli[onnx]"` | ONNX / TensorRT-LLM export |
| `awq` / `gptq` | `pip install "soup-cli[awq]"` | AWQ / GPTQ quantized export |
| `trackers` | `pip install "soup-cli[trackers]"` | MLflow / SwanLab / Trackio logging |
| `remote` | `pip install "soup-cli[remote]"` | Remote datasets (s3 / gs / az / oci) |
| `dev` | `pip install "soup-cli[dev]"` | Tests + lint + types (pytest, ruff, mypy, pre-commit) |

The complete, authoritative extras list is in [`pyproject.toml`](../pyproject.toml).
