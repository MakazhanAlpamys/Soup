<!-- synced-from: README.md sha256:f4beaf4fb00428833a8f4a6541981266464c7b2b301c4ee492276477b27f7727 -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <strong>中文</strong> | <a href="README.ru.md">Русский</a></p>
<!-- synced-from: README.md sha256:4675902b8ea66e77e0822a60fdc9d3205a11eb02fb70cdf3b5670bd9ef1af391 -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <strong>中文</strong></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>一条命令完成 LLM 微调与后训练。无需 SSH，告别配置灾难。</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">官网</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="#配置">配置</a> &middot;
  <a href="#文档">文档</a> &middot;
  <a href="docs/commands.md">命令</a> &middot;
  <a href="docs/models.md">模型</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="下载量"></a>
  <img src="https://img.shields.io/badge/python-3.11--3.13-blue" alt="Python 3.11-3.13">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 许可证">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="测试"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="官网"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - 在 4 GB 笔记本 GPU 上微调 8B LLM | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup 把 LLM 微调的痛点变成一条简单的工作流。一份配置、一条命令，搞定。

```bash
pip install "soup-cli[train]"   # 加 [train] 才能微调；裸 `soup-cli` 是轻量 CLI
soup init --template chat
soup train
```

**在 4 GB 笔记本 GPU 上微调 8B 模型。** Layer streaming 让冻结的基座模型不占
显存，按解码层逐层送入 GPU。在 RTX 3050 Laptop 4 GB 上实测：Llama-3.1-8B-Instruct +
NF4 达到 **952 MHz 下 119.6 tok/s，峰值 3.32 GB**（修复前）——与常规常驻显存的运行逐位一致，并在 H100 上
以同样的 3.32 GB 独立复现出 113.00 tok/s。（两个数字都测于 v0.72.2，早于 v0.73.0 那次
在 32B 上付出 −4.8% 代价的正确性修复。修复后的复测在另一台 RTX 3050 Laptop 4 GB
上完成（@umran666，#1103），结果是 **1935–1957 MHz 下 208.6 tok/s，峰值 2.40 GB**。这相当于
同场 GEMM 上限的 59%，而修复前是 68%：按单位时钟频率折算大约慢 14%，所以表面上 1.74x 的提升来自
时钟频率，而不是加速。large-layer streaming 也改变了两组数据之间的内存占用方式，因此连这个比例也不是
同口径的对比——[完整记录](benchmarks/gate-v0.72.2-nf4.md)。）需显式开启
（`stream_layers: true`），仍是 BETA——
[工作原理](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[全部实测数据](benchmarks/) · [论文](https://doi.org/10.5281/zenodo.21771064) ·
**[在免费 Colab T4 上亲自验证](notebooks/proof-4gb.ipynb)**（把进程限制在 4 GB，
然后断言流式模型与常驻模型逐位一致）

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="在 4 GB 卡上对 Llama-3.1-8B 执行 soup train 的预检：3.60 GB 基座固定在内存中跨 32 层流转，配两个 113 MB 显存缓冲，实测峰值 3.32 GB、119.6 tok/s，停在 4 GB 线以内（修复前的那组数据，测于 v0.72.2，952 MHz；#331 修复后的那组数据见下方说明）"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4、LoRA、batch 1、seq 512，RTX 3050 Laptop 4 GB——<b>峰值 3.32 GB，119.6 tok/s</b>，952 MHz（修复前，v0.72.2；#331 修复后：1935 MHz 下 208.6 tok/s，2.40 GB，不可直接比较，@umran666，#1103）。<a href="https://youtu.be/T1LCErE943E">完整视频（90 秒）</a></sub>
</p>

## 为什么选择 Soup

训练 LLM 至今仍然令人痛苦。即使是有经验的团队，也有 30-50% 的时间耗在和基础设施搏斗上，
而不是改进模型。Soup 解决的就是这件事。

- **零 SSH：** 再也不用 SSH 进一台坏掉的 GPU 机器。
- **一份配置：** 一个简单的 YAML 文件就是全部所需。
- **全部自动：** batch size、GPU 检测、量化，都替你处理。
- **本地可跑：** 用自己的 GPU 跑 QLoRA，不需要云。

## 最新动态

**v0.76.0——Soup 接受却从未应用的设置，现在会直接中止加载。**
那些经过校验、写进文档，却没有被它们声称要配置的路径上的任何代码读取的选项，现在会在加载时被拒绝，
并指出设置名以及该怎么改；其余的会发出警告，并指明将拒绝它们的版本。以前能加载的配置现在可能
无法加载：请先运行 `soup doctor --config soup.yaml`。

- **破坏性变更：从未生效的设置在加载时被拒绝。** 其中包括
  `training.quantization_aware: true`（它从未真正运行过量化感知训练）、`training.use_longlora`、
  `training.ebft_variant`、`training.gdpo_variant`、没有任何初始化流程读取它时的
  `training.rope_scaling_type`、`freeze_layers` / `freeze_ratio`（transformers 后端上
  文本 `sft` / `tts` 路径之外）、`auto_mixed_precision`（`sft` / `tts` 之外），以及另外七个
  没有 unsloth 初始化流程的任务上的 `backend: unsloth`。每条报错都会指出字段名以及该怎么改。
  `lr_groups`、`early_stop_patience`、`forgetting_eval_steps` 及其他无人读取的配置项，现在加载时发出警告，
  从 v0.77 起被拒绝。
- **验证集损失现在会被评估。** `data.val_split` 在每次运行中都会把部分行留作验证、排除在训练之外，
  但 transformers 后端上没有任何训练器去评估它们。现在，每个拿到划分的任务都会在每个 epoch 结束时评估，或通过
  `training.eval_steps` 每 N 步评估一次，并记录验证集损失；运行时间会增加百分之几。
  **对 `grpo`、`ppo` 和 `online_dpo` 是破坏性变更：** 默认不再把行留作验证（`data.val_split`
  会被忽略并给出提示），因此每一行都参与训练。
- **破坏性变更：每个非标准的 `grpo_variant` 都会应用 `grpo_beta` KL 惩罚。** `gspo`、`dapo`、
  `dr_grpo`、`bnpo`、`two_sided` 和 `rft` 以前会校验 `grpo_beta` 然后把它丢掉，因此已有的 variant
  配置无法复现此前的运行结果。`grpo_beta: 0` 现在无 KL 运行，并跳过参考模型的前向传播。
- **Layer streaming 在后台线程中通过直接 I/O 读取其磁盘存储层，并且可以跨 NVMe 驱动器做条带化。**
  在一个比内存还大的 36 GB、70B 形态的 NF4 存储上（一台 RTX 5070 Laptop 8 GB 机器），冷启动的一步
  从同步读取的 110.5 s 降到 16.2-17.3 s，而借助 `SOUP_LAYER_STREAM_STRIPE_DIRS` 跨两块驱动器后，
  从 17.4 s 降到 9.97 s（持续负载探测：两轮，1.28x 和 1.41x，没有正式判定；
  [记录](benchmarks/probe-rtx5070-two-drive-sustained.md)）。能装进内存的存储在内存层上仍然更快，
  `stream_source: auto` 会选择它。需显式开启（`stream_layers: true`），仍是 BETA；
  记录：[直接 I/O](benchmarks/gate-974-disk-tier-direct-io.md)、
  [条带化](benchmarks/gate-two-drive-striping.md)。
- **新命令和新选项。** `soup adapters check` 扫描 LoRA adapter 是否存在静默失效；
  `soup bench train` 训练固定步数，若模型没有在训练则以退出码 1 退出
  （它覆盖 transformers 后端上的 `task: sft`；`soup bench` 现在是一个命令组，`soup bench infer`
  就是原来的推理基准测试）；`soup data clean` 清洗并修复数据集，且不会改动输入文件；
  `soup recipes verify` 在 meta 设备上检查（不下载任何权重）每个随附配方都能挂载 LoRA adapter；
  `soup rewind` 指出造成 loss 尖峰的数据集行。`soup infer` 新增需显式开启的 `--batch-size` 和实验性的
  `--cuda-graphs`（需要 PyTorch 2.14 或更新版本；在一台 RTX 5070 Laptop 上，未绑定 CPU 核心，
  Qwen2.5-1.5B 以 FP16 运行，开启后最慢的一次运行，其解码速度也比未开启时最快的一次快 2.9x）。
  `soup ab` 运行双侧、对方差稳健的检验，`soup adapters sign --backend sigstore` 和
  `soup attest emit --sign sigstore` 使用 Sigstore 免密钥 OIDC 签名
  （`pip install "soup-cli[sigstore]"`）。
- **已弃用：** `soup export --format awq` 和 `--format gptq` 将在 v0.77 移除；这两个上游项目都已归档，
  两个 extra 也都无法与 `[train]` 一起安装。

> 仅支持 Python **3.11–3.13**。在 3.14+ 上，pip 过去会解析到未测试过的 PyTorch
> wheel，在 Soup 运行之前就崩在原生扩展里。

更早的更新记录在 [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) 页面。

## 快速开始

### 1. 安装

Soup 是一个命令行应用，最干净的安装方式是给它一个独立的环境，并把 `soup`
放进你的 `PATH`：

```bash
# 轻量核心：CLI + 配置 + 数据工具，不含 PyTorch
pipx install soup-cli
uv tool install soup-cli          # 同理，如果你已经在用 uv

# 加训练栈（torch、transformers、peft、trl、datasets、…）
pipx install "soup-cli[train]"

# 一次装全（train + serve + ui + data）
pipx install "soup-cli[all]"

# 或者从 GitHub 装（最新开发版）
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

已经在 virtualenv、Colab notebook 或 Docker 镜像里了？直接用 `pip`，
名字和 extras 都一样：

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

如果你还想在自己的代码里 `import soup_cli`，就用 `pip` 而不是 `pipx`，
因为 pipx 会刻意把应用和其他一切隔离开。

完整的 extras 表（`fast`、`mlx`、`serve`、`eval`、`ui`、`vision`、`audio`、…）在
[`docs/models.md`](docs/models.md#optional-extras)。

> **`error: externally-managed-environment`？** 那是
> [PEP 668](https://peps.python.org/pep-0668/)，不是 Soup 的问题。Debian 12、
> Ubuntu 23.04 及更新版本禁止 `pip` 写入系统 Python，因为那些文件也归 `apt` 管。
> `pipx` 和 `uv tool` 通过给 Soup 独立环境绕开了它，所以它们列在最上面。
> `python3 -m venv .venv && source .venv/bin/activate` 之后再用普通 `pip` 也可以。

> **用双引号，不是单引号。** `"soup-cli[train]"` 是唯一在每种 shell 都能用的写法——
> `cmd.exe`、PowerShell、bash 和 zsh。如果你从旧教程复制了 `'soup-cli[train]'` 而被
> pip 拒绝，原因就是这个：
> [为什么，以及确切报错](docs/models.md#quoting-the-extra)。

`soup init`、`soup data …` 和其他数据/检查命令在轻量安装下就能用。
微调（`soup train`）需要 `[train]` extra。

### 2. 创建配置

```bash
soup init                       # 交互式向导
soup init --template chat       # 或从模板开始
```

模板：`chat`、`code`、`tool-calling`、`medical`、`reasoning`、`vision`、`kto`、`orpo`、
`simpo`、`ipo`、`bco`、`rlhf`、`pretrain`、`moe`、`longcontext`、`embedding`、`audio`。

### 3. 训练、测试、发布

```bash
soup train --config soup.yaml                 # LoRA、量化、batching——全部代劳
soup chat  --model ./output                    # 和你的模型对话
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # 把 LoRA 合进基座
soup export --model ./output --format gguf --quant q4_k_m   # 导出 GGUF 给 Ollama / llama.cpp
```

更多导出目标（ONNX、TensorRT、AWQ、GPTQ、BitNet）和部署选项在
[`docs/serving-and-export.md`](docs/serving-and-export.md)。

## Web UI

更喜欢浏览器？`soup ui` 提供一个本地仪表盘，可以做实验、配置训练、
看实时指标、探索数据集、和模型聊天。

```bash
pip install "soup-cli[ui]"
soup ui
# 打开 http://127.0.0.1:7860/?token=<token>，终端也会显示同一 URL
```

![Soup Web UI——新建训练](docs/assets/web-ui-new-training.png)

[Web UI 文档](docs/serving-and-export.md#web-ui)

## 配置

一份完整的 `soup.yaml`：

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 快 2-5 倍，pip install "soup-cli[fast]"

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

`config/schema.py` 是每个字段的唯一权威来源。高级数据、训练和 PEFT 选项的文档在
[文档](#文档)一节。

> **自 v0.75 起，未知配置键会被拒绝。** 没有任何模型声明过的键——比如
> `quantizaton` 这样的笔误，或只存在于更新版 Soup 的字段——以前会通过校验然后被
> 丢弃，任务带着“设置没生效”的状态继续跑。v0.74 会在加载时报出来并提示你可能想写的
> 字段；从 **v0.75** 起同样的配置直接加载失败，所以请修正或删掉这个键，别再指望它被
> 忽略。见 [未知配置键](docs/backends-and-ops.md#unknown-config-keys)。

## 文档

完整的功能参考在 [`docs/`](docs/)。从这里开始：

| 指南 | 内容 |
|---|---|
| [训练任务与方法](docs/training.md) | SFT、DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO、tool-calling、PRM、预训练、蒸馏、分类、视觉/音频/TTS、机器遗忘、RAFT/RA-DIT、训练循环加固检测器 |
| [PEFT、长上下文与效率](docs/peft-and-efficiency.md) | DoRA、LoRA+、rsLoRA、VeRA、OLoRA、NEFTune、PiSSA、ReLoRA、optimizer 与 PEFT 工具集、LLaMA Pro、GaLore、YaRN/LongLoRA、packing、课程学习、自动调参 |
| [性能与量化](docs/performance-and-quantization.md) | QAT、FP8、量化菜单（I + II）、KV-cache、NVFP4、保存格式、Cut Cross-Entropy、gradient checkpointing、kernel、activation offloading、layer streaming、多 GPU / DeepSpeed / FSDP |
| [数据工程](docs/data.md) | 格式、对齐 Axolotl/LF 的数据管线、数据工具、合成生成与 forge、质量记分卡、trace 工具、远程数据集、混合、recipe DAG |
| [评测与探针](docs/evaluation.md) | eval design/gate、eval-gated 训练、benchmark、NLG 指标、校准、Elo 竞技场、诊断、后训练 X-ray 探针、A/B、漂移、可调性、`soup advise` |
| [服务与导出](docs/serving-and-export.md) | OpenAI 兼容服务器、批量推理、benchmark、merge/export、Anthropic Messages 端点、投机解码（训练并测量你自己的 draft）、deploy autopilot、Web UI、Agent Forge |
| [Adapter、registry 与治理](docs/adapters-and-governance.md) | Adapter 生命周期/管理、model registry、Soup Cans、数据飞轮（`soup loop`）、知识编辑、steering、供应链管控（scan/sign/BOM/attest/audit/airgap） |
| [合规与治理快速上手](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` 模板、来源追溯（BOM/attest/repro-receipt）、审计日志、air-gap、模型卡自动生成（`soup card`）、CI gate（`soup ci init`） |
| [后端、平台与运维](docs/backends-and-ops.md) | MLX/Unsloth 后端、其他 hub、HF Hub 集成、autopilot、experiment tracking、plan/apply、env lockfile、硬件适配、补全、插件、实用命令 |
| [命令参考](docs/commands.md) | 完整的 `soup` 命令列表 |
| [支持的模型与 extras](docs/models.md) | 推荐模型家族、VRAM 容量指南、pip extras 矩阵 |

## 数据格式

Alpaca、ShareGPT、ChatML、偏好对（DPO / ORPO / SimPO / IPO / KTO）、视觉、音频、
ASR、纯文本、embedding、RAFT 等等——全部从 JSONL、JSON、CSV、Parquet 或 TXT 自动检测，
所以多数情况下把 `data.train` 指向文件就行，其他什么都不用改。每种格式的 schema 和
完整示例，加上数据管线（远程 URI、流式、分片、交错、词表扩展、文档导入），都在
[`docs/data.md`](docs/data.md#data-formats)，而 [Sesame CSM 原生音频训练](docs/training.md#native-sesame-csm) 则另有说明。

## 常用命令

```bash
soup train  --config soup.yaml        # 训练（SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...）
soup infer  --model ./output --input prompts.jsonl --output results.jsonl   # 批量推理
soup chat   --model ./output          # 交互式对话
soup serve  --model ./output          # OpenAI 兼容 API 服务
soup ui                               # 本地浏览器仪表盘
soup merge  --adapter ./output        # 把 LoRA 合进基座模型
soup export --model ./output --format gguf           # 导出用于部署
soup eval   benchmark --model ./output               # 评测
soup data   inspect ./data/train.jsonl               # 数据集统计
soup recipes list                     # 100+ 个现成模型配方
soup autopilot --model <id> --data d.jsonl --goal chat  # 零配置
soup doctor                           # 检查 GPU / 依赖 / 环境
```

完整的命令列表在 [`docs/commands.md`](docs/commands.md)。

## 支持的模型

Soup 支持 [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation)
上**任意**文本生成模型——只要能被 `AutoModelForCausalLM` 加载就能用，零配置改动。
Llama 3.x/4、Qwen 2.5/3、Gemma 3、Mistral、Mixtral、DeepSeek R1/V3、Phi-4 等 100+
模型都有现成配方（`soup recipes list`）。

| VRAM | 最大模型（QLoRA 4-bit） | 示例 |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+（全量）或 MoE | Mixtral-8x22B, DeepSeek-V3 |

完整的模型表 + 视觉模型表格和可选 extras 矩阵在 [`docs/models.md`](docs/models.md)。

## Docker

不在本地装 CUDA 或 PyTorch 也能跑 Soup（每个 release 都会发布镜像到 GHCR）：

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # 或在本地构建
```

## 环境要求

- Python 3.11、3.12 或 3.13（这是 CI 测试的版本；3.14+ 暂不支持，因为 PyTorch
  技术栈还没在那上面验证过）
- 带 CUDA 的 GPU（推荐）、Apple Silicon（MPS）、或 CPU（实验性——非常慢）
- 7B 模型跑 QLoRA 需要 8 GB+ 显存

所有训练任务都能在 CPU 上跑测试（量化自动关闭）。可选 extras（`train`、`all`、
`fast`、`vision`、`qat`、`serve`、`serve-fast`、`ui`、`eval`、`deepspeed`、`liger`、
`mlx`、`onnx`、`tensorrt`、…）列在
[`docs/models.md`](docs/models.md#optional-extras)。

## 故障排查

```bash
soup doctor    # GPU、系统资源、依赖、版本，一处看全
```

CUDA wheel、版本不匹配：[`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting)。

## 开发

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # 单元测试（快，无需 GPU）
pytest tests/ -m smoke -v          # 冒烟测试（下载一个小模型并训练）
pytest tests/ -m gpu --no-cov -v   # GPU 测试（需要 CUDA 卡；请反馈结果，见 CONTRIBUTING.md）

pre-commit install                 # 可选：提交时跑 ruff lint+format
```

完整工作流见 [CONTRIBUTING.md](CONTRIBUTING.md)，报告漏洞见 [SECURITY.md](SECURITY.md)。
遥测严格 opt-in（`SOUP_TELEMETRY=1`，默认关闭；见[隐私政策](docs/backends-and-ops.md#privacy-policy)）。

## 支持 Soup

Soup 是 Apache-2.0，免费——并且会一直免费。它在一台 4 GB 笔记本上公开构建和维护，
所以这些文档里的每个性能数字都是实测而非声称。

如果 Soup 帮你省下了一次训练，[给仓库点 star](https://github.com/MakazhanAlpamys/Soup)
帮助最大，也不花一分钱。如果你想直接资助这项工作：

**[❤️ 捐赠](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)**——一次性，金额随意
（在结账页用 *Change amount*）。付款由 Stripe 在维护者的注册企业 **MePlay, Inc.**
名下处理——结账页和信用卡账单上显示的是这个名字，不是 "Soup"。

捐款用于购买 GPU 时间，做那些受硬件门槛限制的工作——多卡、8B+ 验证、Apple
Silicon——一台 4 GB 笔记本够不到的那些。

另一条路就是**硬件本身**。这些工作项都挂着诚实的 "requires \<hardware\>" gate 而不是
未验证的声明，所以如果你能用到更大的机器——或者有闲置的 GPU 额度——跑一个
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
issue 并把数字贴出来，和资助 GPU 时间的帮助一样大。那些 issue 写明了目前哪些工作
正卡在硬件上。

## 贡献者

由社区共建 ❤️——感谢每一位贡献者。见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

[![Contributors](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## 联系方式

Bug 和功能请求请发到
[Issues](https://github.com/MakazhanAlpamys/Soup/issues)，问题讨论去
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)——两处都会更快得到
回复，也能帮到后来遇到同样问题的人。

即时聊天、安装帮助、以及一切适合对话形式的事情，欢迎加入
[Discord](https://discord.gg/dgd2pJcjwP) 或 [Telegram 社区](https://t.me/souptasters)。
任何六个月后还应该能被搜到的东西都应该留在 Issues 或 Discussions——Discord 上的
一个回答只帮到一个人，issue 帮到每一个撞上同一问题的人。
[行为准则](CODE_OF_CONDUCT.md)在那里同样适用。

不适合公开的事情——安全报告（见 [SECURITY.md](SECURITY.md)）、行为准则事务或媒体
联络——请发邮件到 **team@trysoup.dev**。这是项目邮箱，Soup 相关事务的正确地址。
**makazanalpamys@gmail.com** 是维护者的个人邮箱；也能找到同一个人，可以作为备选。

## 引用 Soup

Layer streaming——把冻结的基座模型放在主机内存里、按解码层逐层送入 GPU，从而在
4 GB 笔记本 GPU 上训练 8B 模型——在一篇预印本中有完整描述，其中还包括验证流式
运行与常驻运行一致性的正确性协议（前向和反向分开陈述，因为那是两个主张而不是
一个）。

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**Version 3（2026 年 8 月 13 日）是当前版本。** 标题和主张没变——8B 跑在 4 GB 上——
v1 以来的任何实测数字也没有变。v3 做的是**撤回一个我们曾发布的解释**，这也是描述
这篇论文用途的最短方式：

- **v3 撤回的结论："layer streaming 的瓶颈在主机到设备的传输，而不在 GPU。"**
  那其实是从下面的 H100 复现得出的*推断*，从来没有实测过。我们在 8 月 11 日做了
  测量，发现在已发布的配置下它不成立：把主机到设备的字节传输全部省掉也只换来
  **1.4%** 的提升，
  计算流等待拷贝的时间只占 step 的 **0.20%**，而该 step 跑到了这张卡同场 GEMM
  上限的 **71.3%**。最大的流式专属开销是逐层 NF4 反量化，占 9.8%
  （[完整记录](benchmarks/probe-v0.73.0-what-bounds-streaming.md)）。每个测量都仍然
  成立；复现结论以更弱的形式仍然成立——约束对两台机器是共同的，而不是 GPU 的算力。
- **在完全不同硬件上的复现**（v2 新增）：RTX 3050 上 119.6 tok/s（952 MHz 下），对 H100 上
  113.00 的中位数，同为 3.32 GB 峰值。两者都早于 #331 修复，也是论文报告的数字。#331 修复后的
  那组数据（@umran666，#1103）是 1935–1957 MHz 下 208.6 tok/s，峰值 2.40 GB，为同场 GEMM 上限的
  59%，修复前为 68%：表面上 1.74x 的提升来自时钟频率，而不是加速；两组数据之间的内存占用方式也变了，
  因此连这个比例也不是同口径的对比
  （[完整记录](benchmarks/gate-v0.72.2-nf4.md)）。
- **发现并修复了一个静默的错误梯度缺陷。** NF4 单层超过 ~165 MiB 时，前向仍然
  逐位一致、loss 曲线看着健康，但梯度是错的。原因已在上游库中定位并上报；修复
  已通过真实 32B 和 72B 上的对照实验验证。
- **真实模型规模的逐位一致性**，而不是三层玩具：前向覆盖 0.5B 到 72B，反向覆盖
  8B 和 14B。
- **首次实测了训练后模型的质量**，与常驻运行无区别。
- **与 DeepSpeed 的对比**——包括一个不利于我们的结果：八卡的 ZeRO-3 比单卡常驻
  训练还慢。
- **局限性章节重写**：v1 的十条里一条已解决、四条收窄，并新增七条。

引用你使用的版本。`10.5281/zenodo.21771064` 是概念 DOI，始终解析到最新版本
（今天是 v3）；v1 和 v2 仍可在各自的版本 DOI 下引用且不会被修改——上面的撤回
之所以发成一个新版本，正是为了让我们声称过什么、什么时候声称的记录保持完整。

文中每个数字背后的测量记录都在 [`benchmarks/`](benchmarks/)，按原样发布——
包括失败、被证伪的假设，以及测出来又被弃用的数字。

```bibtex
@misc{makazhan2026exact,
  title        = {Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU},
  author       = {Makazhan, Alpamys},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3},
  doi          = {10.5281/zenodo.21918325},
  url          = {https://doi.org/10.5281/zenodo.21918325}
}
```

## 许可证

[Apache-2.0](LICENSE)。Copyright 2026 Makazhan Alpamys and the Soup contributors。如果你再分发 Soup 或其修改版本，许可证的第 4 条和第 6 条对你适用：每份副本都要保留 `LICENSE` 和 `NOTICE`，在你改动过的文件上注明改动，并且除说明作品来源之外不要使用 Soup 这一名称。
