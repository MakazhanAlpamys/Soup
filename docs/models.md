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
| **Qwen 3** | Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B | 0.6B–235B | Reasoning, code, MoE |
| **Qwen 2.5** | Qwen2.5-7B-Instruct, Qwen2.5-Coder-32B-Instruct | 0.5B–72B | Code, math |
| **DeepSeek** | DeepSeek-R1-Distill-Llama-8B, DeepSeek-V3-0324 | 1.5B–671B | Reasoning (GRPO), code |
| **Phi-4** | Phi-4-14B, Phi-4-mini-reasoning | 3.8B–14B | Compact reasoning |
| **Mistral** | Mistral-7B-Instruct-v0.3, Mistral-Small-24B-Instruct | 7B–24B | Fast, efficient |
| **Mixtral** | Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B | 47B–141B | MoE architecture |
| **CodeLlama** | CodeLlama-7b-Instruct-hf, CodeLlama-34b-Instruct | 7B–34B | Code generation |
| **StarCoder 2** | StarCoder2-15B, StarCoder2-7B | 3B–15B | Code completion |
| **Yi** | Yi-1.5-34B-Chat, Yi-1.5-9B-Chat | 6B–34B | Multilingual chat |
| **InternLM 3** | InternLM3-8B-Instruct | 8B | Chinese + English |
| **Falcon** | Falcon-11B, Falcon-40B-Instruct | 7B–180B | Open-weight |

### Vision Models (with `modality: vision`)

| Model | Size | Supported Formats |
|---|---|---|
| LLaMA-3.2-11B-Vision-Instruct | 11B | LLaVA, ShareGPT4V |
| Qwen2-VL-7B-Instruct | 7B | LLaVA, ShareGPT4V |
| Pixtral-12B-2409 | 12B | LLaVA, ShareGPT4V |

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

The core `pip install soup-cli` is a light install — the CLI, config system, and data tools, with
no PyTorch. Add `[train]` to fine-tune, or install other extras only when you need them:

| Extra | Install | What it adds |
|---|---|---|
| `train` | `pip install 'soup-cli[train]'` | Training stack: torch, transformers, peft, trl, datasets, bitsandbytes, accelerate |
| `all` | `pip install 'soup-cli[all]'` | `train` + `serve` + `ui` + `data` in one shot |
| `fast` | `pip install 'soup-cli[fast]'` | Unsloth backend (2-5x faster, lower VRAM) |
| `vision` | `pip install 'soup-cli[vision]'` | Vision / multimodal fine-tuning (Pillow) |
| `audio` | `pip install 'soup-cli[audio]'` | Audio / speech fine-tuning (librosa, soundfile) |
| `mlx` | `pip install 'soup-cli[mlx]'` | Apple Silicon backend (mlx, mlx-lm) |
| `qat` | `pip install 'soup-cli[qat]'` | Quantization-Aware Training (torchao) |
| `serve` | `pip install 'soup-cli[serve]'` | Inference server (FastAPI + uvicorn) |
| `serve-fast` | `pip install 'soup-cli[serve-fast]'` | vLLM inference backend (2-4x throughput) |
| `sglang` | `pip install 'soup-cli[sglang]'` | SGLang inference backend |
| `ui` | `pip install 'soup-cli[ui]'` | Web UI + inference server |
| `tui` | `pip install 'soup-cli[tui]'` | Full-screen Textual dashboard (`soup tui`) |
| `eval` | `pip install 'soup-cli[eval]'` | Benchmark evaluation (lm-evaluation-harness) |
| `data` | `pip install 'soup-cli[data]'` | Deduplication (MinHash via datasketch) |
| `data-pro` | `pip install 'soup-cli[data-pro]'` | Language detection + PII (langdetect, presidio) |
| `deepspeed` | `pip install 'soup-cli[deepspeed]'` | Multi-GPU training (DeepSpeed ZeRO) |
| `liger` | `pip install 'soup-cli[liger]'` | Liger Kernel fused ops |
| `ring-attn` | `pip install 'soup-cli[ring-attn]'` | Ring FlashAttention (sequence parallelism) |
| `onnx` / `tensorrt` | `pip install 'soup-cli[onnx]'` | ONNX / TensorRT-LLM export |
| `awq` / `gptq` | `pip install 'soup-cli[awq]'` | AWQ / GPTQ quantized export |
| `trackers` | `pip install 'soup-cli[trackers]'` | MLflow / SwanLab / Trackio logging |
| `remote` | `pip install 'soup-cli[remote]'` | Remote datasets (s3 / gs / az / oci) |
| `dev` | `pip install 'soup-cli[dev]'` | Tests + lint + types (pytest, ruff, mypy, pre-commit) |

The complete, authoritative extras list is in [`pyproject.toml`](../pyproject.toml).


