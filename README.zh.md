<!-- synced-from: README.md sha256:1cdf85f67b56a053d993850d6d8ee27e161e3efd9d2bf6d3ec10e68691fcc3b6 -->
<p align="center"><strong>🌍 <a href="README.tr.md">Türkçe</a> | <a href="README.md">English</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.ru.md">Русский</a> | <a href="README.es.md">Español</a></strong></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>一键微调和后训练LLM。无需SSH，无配置地狱。</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">网站</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#web界面">Web界面</a> &middot;
  <a href="#配置">配置</a> &middot;
  <a href="docs/commands.md">命令</a> &middot;
  <a href="docs/models.md">模型</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="下载量"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0许可证">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="测试"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
</p>

---

Soup将LLM微调的痛苦转化为简单的工作流程。一个配置，一条命令，完成。

```bash
pip install "soup-cli[train]"   # 添加[train]用于微调；单独的soup-cli是轻量级CLI
soup init --template chat
soup train
```

**在4GB笔记本GPU上微调8B模型。** 层流式传输将冻结的基础模型保持在VRAM之外，一次向GPU提供一个解码器层。在RTX 3050 Laptop 4GB上测量：Llama-3.1-8B-Instruct + NF4，**119.6 tok/s，峰值3.32GB** — 与正常驻留运行位精确，在H100上以相同的3.32GB独立重现了113.00 tok/s。可选（`stream_layers: true`）且仍为BETA —
[工作原理](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[所有测量](benchmarks/) ·
[论文](https://doi.org/10.5281/zenodo.21771064) ·
**[在免费的Colab T4上自己验证](notebooks/proof-4gb.ipynb)**

---

## 🎯 为什么选择Soup？

LLM训练仍然痛苦。即使是经验丰富的团队也会将30-50%的时间花费在与基础设施斗争上，而不是改进模型。Soup解决了这个问题。

- **零SSH。** 再也不用SSH进入损坏的GPU盒子。
- **一个配置。** 只需要一个简单的YAML文件。
- **一切自动化。** 批量大小、GPU检测、量化 — 全部处理。
- **本地工作。** 使用QLoRA在自己的GPU上训练。无需云服务。

### 新功能 — v0.74.0

**v0.74.0 — 冻结基础一直在fp32中加载。** 仅修复这一点就在未更改的配置上将峰值VRAM减少了2.59倍。**此版本中120个合并的拉取请求中有116个来自维护者之外的25个人。**

- 每次SFT加载都会静默地将冻结基础上转换为fp32。在H100上使用Llama-3.1-8B + LoRA测量：**48,241 MiB → 18,658 MiB峰值 — 2.59倍，28.9 GB**。
- **Transformers 5.x，TRL 0.29，PEFT 0.20。**
- 修复了四个相同形状的SSRF绕过（缩写、十进制、十六进制和八进制IPv4）。
- **`soup serve`在绑定到非回环主机而没有`--tool-auth-token`时现在退出2。**
- **`soup train --cloud lambda`** — 默认仅计划，`--cloud-submit`实时提交。

---

## 🏗️ 架构和模块

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70多个CLI命令（train, eval, data, serve, agent, …）
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — 所有配置字段的单一真实来源
│   ├── autodistill/     # 自动蒸馏（数据捕获、发布、MLX worker）
│   ├── autopilot/       # 零配置任务/量化/学习率选择
│   ├── cans/            # Soup Cans：模型打包、发布、验证
│   ├── cloud/           # Lambda Labs / Modal云控制器
│   └── utils/           # 共享助手（350多个模块）
├── docs/                # 主题指南 + 完整命令参考
├── templates/           # chat.yaml、code.yaml、medical.yaml起始模板
├── benchmarks/          # 测量记录（每个数字的原始数据）
├── tests/               # 460个测试文件
└── pyproject.toml       # 依赖项、额外项、构建系统
```

**核心依赖项（轻量级安装 — 无PyTorch）：**
`typer`, `rich`, `pydantic>=2`, `pyyaml`, `huggingface-hub`, `plotext`, `packaging`

**`[train]`额外项：** `torch>=2.6.0`, `transformers>=5.16.1,<6`, `peft>=0.20,<1`, `trl>=0.29`, `datasets`, `bitsandbytes`, `accelerate`

**注意：** 此存储库包含Python源代码以及安装脚本、CI工作流和打包工具；我们不声称"%100纯Python"。

---

## 🚀 安装

### 要求

- Python 3.10、3.11或3.12（这些是CI测试的版本；3.13+尚未支持）
- 带CUDA的GPU（推荐）、Apple Silicon（MPS）或CPU（实验性 — 非常慢）
- 使用QLoRA的7B模型需要8GB以上VRAM

### 安装步骤

```bash
# 轻量级核心：CLI + 配置 + 数据工具，无PyTorch
pipx install soup-cli
uv tool install soup-cli          # 如果你已经使用uv，同样的想法

# 添加训练栈（torch、transformers、peft、trl、datasets、…）
pipx install "soup-cli[train]"

# 一次性安装所有内容（train + serve + ui + data）
pipx install "soup-cli[all]"

# 或从GitHub（最新开发版本）
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

已经在virtualenv、Colab notebook或Docker镜像内？直接使用`pip`：

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **`error: externally-managed-environment`？** 这是[PEP 668](https://peps.python.org/pep-0668/)，不是Soup问题。使用`pipx`或`uv tool`。

> **双引号，不是单引号。** `"soup-cli[train]"`是在所有shell中都有效的唯一拼写。

完整额外项表格（`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, `liger`, `onnx`, `tensorrt`, …）：[`docs/models.md`](docs/models.md#optional-extras)

### 从分叉安装

此存储库是`Ercaner1988/Soup`分叉。上游：`MakazhanAlpamys/Soup`。

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

---

## 📖 使用方法

### 快速开始

```bash
# 1. 创建配置
soup init                       # 交互式向导
soup init --template chat       # 或从模板开始

# 2. 训练、测试、发布
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo user/my-model

# 3. 合并和导出
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**模板：** `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`

### 常用命令

```bash
soup train  --config soup.yaml                         # 训练（SFT/DPO/GRPO/PPO/KTO/ORPO/…）
soup infer  --model ./output --input prompts.jsonl    # 批量推理
soup chat   --model ./output                           # 交互式聊天
soup serve  --model ./output                           # OpenAI兼容API服务器
soup ui                                                # 本地浏览器仪表板
soup merge  --adapter ./output                         # 将LoRA合并到基础模型
soup export --model ./output --format gguf             # 导出用于部署
soup eval   benchmark --model ./output                 # 评估
soup data   inspect ./data/train.jsonl                 # 数据集统计
soup recipes list                                      # 100多个现成的模型配方
soup autopilot --model <id> --data d.jsonl --goal chat # 零配置
soup doctor                                            # GPU / 依赖 / 环境检查
```

### 配置

完整的`soup.yaml`：

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 快2-5倍，pip install "soup-cli[fast]"

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
  quantization: 4bit

output: ./output
```

`config/schema.py`是每个字段的单一真实来源。

> **未知配置键今天会警告，在v0.75中将被拒绝。**

### 数据格式

Alpaca、ShareGPT、ChatML、偏好对（DPO / ORPO / SimPO / IPO / KTO）、视觉、音频、ASR、纯文本、嵌入、RAFT等 — 从JSONL、JSON、CSV、Parquet或TXT自动检测。请参阅[`docs/data.md`](docs/data.md)。

### 支持的模型

Soup适用于[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation)上的**任何**文本生成模型。Llama 3.x/4、Qwen 2.5/3、Gemma 3、Mistral、Mixtral、DeepSeek R1/V3、Phi-4和100多个其他模型作为现成配方提供（`soup recipes list`）。

| VRAM | 最大模型（QLoRA 4-bit） | 示例 |
|------|------------------------|------|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+（完整）或MoE | Mixtral-8x22B, DeepSeek-V3 |

### Web界面

```bash
pip install "soup-cli[ui]"
soup ui
# 打开 http://127.0.0.1:7860
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ 测试和质量门

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — 任何提交前必须清洁
pytest tests/ -v --tb=short        # 单元测试（快速，无GPU；460个测试文件）
pytest tests/ -m smoke -v          # 冒烟测试（下载小模型，训练）
pre-commit install                 # 可选：提交时ruff lint+格式化
```

**CI矩阵：** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**安全性：** `soup doctor`在一个地方检查GPU、系统资源、依赖项和版本。遥测严格选择加入（`SOUP_TELEMETRY=1`，默认关闭）。

---

## 📚 文档

| 指南 | 覆盖范围 |
|------|---------|
| [训练任务和方法](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO，工具调用，PRM，预训练，蒸馏 |
| [PEFT和效率](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA |
| [性能和量化](docs/performance-and-quantization.md) | QAT, FP8, KV缓存，层流式传输，多GPU / DeepSpeed / FSDP |
| [数据工程](docs/data.md) | 格式，数据工具，合成生成，配方DAG |
| [评估和探针](docs/evaluation.md) | 评估设计，基准测试，NLG指标，A/B |
| [服务和导出](docs/serving-and-export.md) | OpenAI兼容服务器，合并/导出，Web UI |
| [适配器和治理](docs/adapters-and-governance.md) | 适配器生命周期，模型注册表，Soup Cans |
| [合规性](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7模板，审计日志 |
| [后端和操作](docs/backends-and-ops.md) | MLX/Unsloth后端，HF Hub，自动驾驶 |
| [命令参考](docs/commands.md) | 完整的`soup`命令列表 |
| [模型和额外项](docs/models.md) | 模型家族，VRAM指南，pip额外项矩阵 |

---

## 👥 贡献者

上游存储库（`MakazhanAlpamys/Soup`）由社区构建。此分叉（`Ercaner1988/Soup`）包含多语言文档。

| 贡献者 | 提交数 | 角色 |
|--------|--------|------|
| Alpamys (Alpamys Makazhan) | 912 | 项目所有者和首席开发者 |
| AmixDigital | 37 | Web UI前门部分 |
| Srinivasan R | 24 | MLX响应专用掩码，验证损失，配方ID保护 |
| Amir Fathi | 24 | 贡献 |
| Ben Younes | 15 | 贡献 |
| Salil Mhatre | 11 | 贡献 |
| Darsh | 9 | 贡献 |
| Nurkhan Esenbek | 9 | 贡献 |
| Harshit Sharma | 7 | 贡献 |
| UmranPros | 6 | 贡献 |
| （其他24位贡献者） | 5–1 | 见[CONTRIBUTORS.md](CONTRIBUTORS.md) |
| **Ercan Er** | **2** | **多语言文档和分支配置** |

> 指标来自`git shortlog -sn HEAD`输出（ercan1分支，2026-09-09）。

---

## 📄 许可证

[Apache-2.0](LICENSE)。版权所有 © Soup贡献者。

此分叉（`Ercaner1988/Soup`）基于原始项目：[MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

### 引用

```bibtex
@misc{makazhan2026exact,
  title     = {Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU},
  author    = {Makazhan, Alpamys},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v3},
  doi       = {10.5281/zenodo.21918325},
  url       = {https://doi.org/10.5281/zenodo.21918325}
}
```
