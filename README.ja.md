<p align="center">**🌍 [Türkçe](README.md) | [English](README.en.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)**

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>LLMのファインチューニングと事後学習をコマンド 하나로。SSHなし、設定の地獄なし。</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">ウェブサイト</a> &middot;
  <a href="#クイックスタート">クイックスタート</a> &middot;
  <a href="#ウェブui">ウェブUI</a> &middot;
  <a href="#設定">設定</a> &middot;
  <a href="docs/commands.md">コマンド</a> &middot;
  <a href="docs/models.md">モデル</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="ダウンロード数"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0ライセンス">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="テスト"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
</p>

---

SoupはLLMファインチューニングの苦痛をシンプルなワークフローに変えます。設定ひとつ、コマンドひとつ、完了。

```bash
pip install "soup-cli[train]"   # ファインチューニングには[train]を追加；soup-cliのみは軽量CLI
soup init --template chat
soup train
```

**4GBラップトップGPUで8Bモデルをファインチューニング。** レイヤーストリーミングはフローズンベースをVRAM外に保ち、GPUに一度に1デコーダレイヤーを供給します。RTX 3050 Laptop 4GBで測定：Llama-3.1-8B-Instruct + NF4、**119.6 tok/s、ピーク3.32GB** — 通常のレジデンット実行とビット一致、H100で同じ3.32GBで113.00 tok/sに独立して再現。オプトイン（`stream_layers: true`）でまだBETA —
[仕組み](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[全測定値](benchmarks/) ·
[論文](https://doi.org/10.5281/zenodo.21771064) ·
**[無料でColab T4で自分で確認](notebooks/proof-4gb.ipynb)**

---

## 🎯 なぜSoupなのか？

LLMのトレーニングはまだ苦痛です。経験豊富なチームでも時間の30-50%をモデル改善ではなくインフラとの戦いに費やします。Soupが解決します。

- **SSHゼロ。** 故障したGPUボックスに二度とSSHしない。
- **設定ひとつ。** シンプルなYAMLファイルだけが必要。
- **すべて自動。** バッチサイズ、GPU検出、量子化 — 対応済み。
- **ローカルで動作。** QLoRAで自分のGPUでトレーニング。クラウド不要。

### 新機能 — v0.74.0

**v0.74.0 — フローズンベースがずっとfp32でロードされていた。** これらを修正するだけでも変更のない設定でピークVRAMを2.59倍削減。**このリリースでマージされたプルリクエスト120件中116件はメンテナー以外から、25人によって。**

- すべてのSFTロードがフローズンベースを静かにfp32にキャストアップ。H100でLlama-3.1-8B + LoRAを測定：**48,241 MiB → 18,658 MiBピーク — 2.59倍、28.9 GB**。
- **Transformers 5.x、TRL 0.29、PEFT 0.20。**
- 同じ形の4つのSSRFバイパス（省略形、10進、16進、8進IPv4）を修正。
- **`soup serve`は`--tool-auth-token`なしで非ループバックホストにバインドされた場合、exit 2を返すように。**
- **`soup train --cloud lambda`** — デフォルトでプランのみ、`--cloud-submit`でライブ送信。

---

## 🏗️ アーキテクチャとモジュール

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70以上のCLIコマンド（train、eval、data、serve、agent、…）
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — 全設定フィールドの単一情報源
│   ├── autodistill/     # 自動蒸留（データキャプチャ、公開、MLXワーカー）
│   ├── autopilot/       # ゼロコンフィグタスク/量子化/LR選択
│   ├── cans/            # Soup Cans：モデルパッケージ化、公開、検証
│   ├── cloud/           # Lambda Labs / Modalクラウドコントローラー
│   └── utils/           # 共有ヘルパー（350以上のモジュール）
├── graft/               # Graft MCP設定（ユーザーレベル配線）
├── docs/                # トピックガイド + 完全コマンドリファレンス
├── templates/           # chat.yaml、code.yaml、medical.yamlスタートテンプレート
├── benchmarks/          # 測定レコード（全数値の生データ）
├── tests/               # 460テストファイル
└── pyproject.toml       # 依存関係、エクストラ、ビルドシステム
```

**コア依存関係（軽量インストール — PyTorchなし）：**
`typer`、`rich`、`pydantic>=2`、`pyyaml`、`huggingface-hub`、`plotext`、`packaging`

**`[train]`エクストラ：** `torch>=2.6.0`、`transformers>=5.16.1,<6`、`peft>=0.20,<1`、`trl>=0.29`、`datasets`、`bitsandbytes`、`accelerate`

**注意：** このリポジトリにはPythonソースコードに加えてセットアップスクリプト、CIワークフロー、パッケージングツールが含まれています；"%100ピュアPython"を主張していません。

---

## 🚀 インストール

### 必要条件

- Python 3.10、3.11、または3.12（これらがCIでテストされているバージョン；3.13+はまだ未サポート）
- CUDA搭載GPU（推奨）、Apple Silicon（MPS）、またはCPU（実験的 — 非常に遅い）
- QLoRAで7Bモデルの場合8GB以上のVRAM

### インストール手順

```bash
# 軽量コア：CLI + 設定 + データツール、PyTorchなし
pipx install soup-cli
uv tool install soup-cli          # すでにuvを使っている場合の同じアイデア

# トレーニングスタックを追加（torch、transformers、peft、trl、datasets、…）
pipx install "soup-cli[train]"

# すべて（train + serve + ui + data）を一度に
pipx install "soup-cli[all]"

# またはGitHubから（最新開発版）
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

すでにvirtualenv、Colabノートブック、またはDockerイメージ内にある場合は、`pip`を直接使用：

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **`error: externally-managed-environment？** これは[PEP 668](https://peps.python.org/pep-0668/)、Soupの問題ではありません。`pipx`または`uv toolを使用`。

> **ダブルクォート、単一クォートではなく。** `"soup-cli[train]"`はすべてのシェルで動作する唯一のつづり。

完全エクストラテーブル（`fast`、`mlx`、`serve`、`eval`、`ui`、`vision`、`audio`、`liger`、`onnx`、`tensorrt`、…）：[`docs/models.md`](docs/models.md#optional-extras)

### フォークからのインストール

このリポジトリは`Ercaner1988/Soup`フォークです。上流： `MakazhanAlpamys/Soup`。

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

---

## 📖 使用方法

### クイックスタート

```bash
# 1. 設定を作成
soup init                       # インタラクティブなウィザード
soup init --template chat       # またはテンプレートから開始

# 2. トレーニング、テスト、出荷
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo user/my-model

# 3. マージしてエクスポート
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**テンプレート：** `chat`、`code`、`tool-calling`、`medical`、`reasoning`、`vision`、`kto`、`orpo`、`simpo`、`ipo`、`bco`、`rlhf`、`pretrain`、`moe`、`longcontext`、`embedding`、`audio`

### 一般的なコマンド

```bash
soup train  --config soup.yaml                         # トレーニング（SFT/DPO/GRPO/PPO/KTO/ORPO/…）
soup infer  --model ./output --input prompts.jsonl    # バッチ推論
soup chat   --model ./output                           # インタラクティブチャット
soup serve  --model ./output                           # OpenAI互換APIサーバー
soup ui                                                # ローカルブラウザダッシュボード
soup merge  --adapter ./output                         # LoRAをベースモデルにマージ
soup export --model ./output --format gguf             # デプロイ用にエクスポート
soup eval   benchmark --model ./output                 # 評価
soup data   inspect ./data/train.jsonl                 # データセット統計
soup recipes list                                      # 100以上のレディーメイドモデルレシピ
soup autopilot --model <id> --data d.jsonl --goal chat # ゼロコンフィグ
soup doctor                                            # GPU / 依存関係 / 環境チェック
```

### 設定

完全な`soup.yaml`：

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5倍高速、pip install "soup-cli[fast]"

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

`config/schema.py`は各フィールドの単一情報源です。

> **未知の設定キーは今日警告し、v0.75で拒否されます。**

### データ形式

Alpaca、ShareGPT、ChatML、優先度ペア（DPO / ORPO / SimPO / IPO / KTO）、ビジョン、オーディオ、ASR、プレーンテキスト、エンベディング、RAFTなど — JSONL、JSON、CSV、Parquet、またはTXTから自動検出。[`docs/data.md`](docs/data.md)をご覧ください。

### サポートされるモデル

Soupは[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation)の**あらゆる**テキスト生成モデルで動作します。Llama 3.x/4、Qwen 2.5/3、Gemma 3、Mistral、Mixtral、DeepSeek R1/V3、Phi-4など100以上がレディメイドレシピとして提供（`soup recipes list`）。

| VRAM | 最大モデル（QLoRA 4-bit） | 例 |
|------|--------------------------|------|
| 8 GB | ~7B | Llama-3.1-8B、Mistral-7B |
| 16 GB | ~14B | Phi-4-14B、Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B、Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB以上 | 70B以上（フル）またはMoE | Mixtral-8x22B、DeepSeek-V3 |

### ウェブUI

```bash
pip install "soup-cli[ui]"
soup ui
# http://127.0.0.1:7860 を開く
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ テストと品質ゲート

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — コミット前にクリーンであること
pytest tests/ -v --tb=short        # ユニットテスト（高速、GPU不要；460テストファイル）
pytest tests/ -m smoke -v          # スモークテスト（小さいモデルダウンロード、トレーニング）
pre-commit install                 # 任意：コミット時のruff lint+フォーマット
```

**CIマトリクス：** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**セキュリティ：** `soup doctor`はGPU、システムリソース、依存関係、バージョンを一か所でチェックします。テレメトリーは厳密にオプトイン（`SOUP_TELEMETRY=1`、デフォルトオフ）。

---

## 📚 ドキュメント

| ガイド | カバー範囲 |
|--------|-----------|
| [トレーニングタスクとメソッド](docs/training.md) | SFT、DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO、ツール呼び出し、PRM、事前トレーニング、蒸留 |
| [PEFTと効率](docs/peft-and-efficiency.md) | DoRA、LoRA+、rsLoRA、VeRA、OLoRA、NEFTune、PiSSA、ReLoRA |
| [パフォーマンスと量子化](docs/performance-and-quantization.md) | QAT、FP8、KVキャッシュ、レイヤーストリーミング、マルチGPU / DeepSpeed / FSDP |
| [データエンジニアリング](docs/data.md) | 形式、データツール、合成生成、レシピDAG |
| [評価とプローブ](docs/evaluation.md) | 評価設計 benchマーク、NLGメトリクス、A/B |
| [ servingとエクスポート](docs/serving-and-export.md) | OpenAI互換サーバー、マージ/エクスポート、ウェブUI |
| [アダプターとガバナンス](docs/adapters-and-governance.md) | アダプターライフサイクル、モデルレジストリ、Soup Cans |
| [コンプライアンス](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7テンプレート、監査ログ |
| [バックエンドと運用](docs/backends-and-ops.md) | MLX/Unslothバックエンド、HF Hub、オートパイロット |
| [コマンドリファレンス](docs/commands.md) | 完全な`soup`コマンドリスト |
| [モデルとエクストラ](docs/models.md) | モデルファミリー、VRAMガイド、pipエクストラマトリクス |

---

## 👥 コントリビューター

上流リポジトリ（`MakazhanAlpamys/Soup`）はコミュニティによって構築されています。このフォーク（`Ercaner1988/Soup`）はGraft MCP統合用にカスタマイズされています。

| コントリビューター | コミット数 | 役割 |
|-------------------|-----------|------|
| Alpamys (Alpamys Makazhan) | 912 | プロジェクトオーナー兼リードデベロッパー |
| AmixDigital | 37 | ウェブUIフロントドアセクション |
| Srinivasan R | 24 | MLXレスポンンマスク、検証損失、レシペIDガード |
| Amir Fathi | 24 | コントリビューション |
| Ben Younes | 15 | コントリビューション |
| Salil Mhatre | 11 | コントリビューション |
| Darsh | 9 | コントリビューション |
| Nurkhan Esenbek | 9 | コントリビューション |
| Harshit Sharma | 7 | コントリビューション |
| UmranPros | 6 | コントリビューション |
| （その他のコントリビューター24名） | 5–1 | [CONTRIBUTORS.md](CONTRIBUTORS.md)参照 |
| **Ercan Er** | **2** | **Graft MCP設定（ユーザーレベル配線）** |

> 指標は`git shortlog -sn HEAD`出力から取得（ercan1ブランチ、2026-09-09）。

---

## 📄 ライセンス

[Apache-2.0](LICENSE)。著作権 © Soupコントリビューター。

このフォーク（`Ercaner1988/Soup`）は元のプロジェクトに基づいています： [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

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