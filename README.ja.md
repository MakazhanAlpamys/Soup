<!-- synced-from: README.md sha256:a6789c2877515d5ac707160a55ab94cd84b390e2e31168b69d4f954a47c5562a -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <strong>日本語</strong> | <a href="README.zh.md">中文</a></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>コマンド一つで LLM のファインチューニングとポストトレーニングを。SSH も、設定地獄もありません。</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">ウェブサイト</a> &middot;
  <a href="#クイックスタート">クイックスタート</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="#設定">設定</a> &middot;
  <a href="#ドキュメント">ドキュメント</a> &middot;
  <a href="docs/commands.md">コマンド</a> &middot;
  <a href="docs/models.md">モデル</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="ダウンロード数"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 ライセンス">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="テスト"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="ウェブサイト"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - 4 GB のノート PC 向け GPU で 8B の LLM をファインチューニング | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup は、LLM ファインチューニングの手間をシンプルなワークフローに変えます。設定一つ、コマンド一つで完了です。

```bash
pip install "soup-cli[train]"   # ファインチューニングには [train] を追加。素の `soup-cli` は軽量 CLI です
soup init --template chat
soup train
```

**4 GB のノート PC 向け GPU で 8B モデルをファインチューニングできます。** レイヤーストリーミングは、凍結したベースモデルを
VRAM の外に置き、デコーダ層を一層ずつ GPU に供給します。RTX 3050 Laptop 4 GB での計測値：
Llama-3.1-8B-Instruct + NF4 で **119.6 tok/s、ピーク 3.32 GB** です。すべてをメモリに常駐させた通常の実行と
ビット単位で一致し、H100 上でも同じ 3.32 GB で 113.00 tok/s として独立に再現されました。
（どちらの数値も、32B で −4.8% のコストを伴った v0.73.0 の正確性修正より前の v0.72.2 で計測されたもので、
それ以降どちらも 4 GB のカードでは再実行されていません。再計測は
Issue [#361](https://github.com/MakazhanAlpamys/Soup/issues/361) で保留中です。）
オプトイン方式（`stream_layers: true`）で、現在も BETA です —
[仕組み](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[全計測結果](benchmarks/) · [論文](https://doi.org/10.5281/zenodo.21771064) ·
**[無料の Colab T4 で自分で確かめる](notebooks/proof-4gb.ipynb)**（プロセスを 4 GB に制限したうえで、
ストリーミングしたモデルが通常のモデルとビット単位で同一であることを検証します）

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="4 GB のカードで Llama-3.1-8B を対象にした soup train の事前チェック：32 層にわたって RAM に固定された 3.60 GB のベースストアと、113 MB の VRAM バッファ二つ。続いて 119.6 tok/s でピーク 3.32 GB を計測し、4 GB のラインを下回っています（v0.72.2 で計測、#331 の修正前。再計測は Issue #361 で保留中）"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4、LoRA、バッチ 1、シーケンス長 512、RTX 3050 Laptop 4 GB — <b>ピーク 3.32 GB、119.6 tok/s</b>（v0.72.2 で計測、#331 の修正前。再計測は Issue #361 で保留中）。<a href="https://youtu.be/T1LCErE943E">フル動画（90 秒）</a></sub>
</p>

## なぜ Soup なのか？

LLM の学習は今なお骨の折れる作業です。経験豊富なチームでさえ、時間の 30-50% をモデルの改善ではなく
インフラとの格闘に費やしています。Soup はこれを解決します。

- **SSH 不要。** 壊れた GPU マシンに SSH で接続する必要は二度とありません。
- **設定は一つ。** 必要なのはシンプルな YAML ファイル一つだけです。
- **すべて自動。** バッチサイズ、GPU の検出、量子化 — すべて処理されます。
- **ローカルで動作。** QLoRA を使って自分の GPU で学習できます。クラウドは不要です。

## 新着情報

**v0.75.0 — 同じ `soup.yaml` が、MLX では transformers とは異なるレシピで学習していました。しかも何の警告もなく。**
六つの学習オプションが検証され、ドキュメント化され、受け入れられていながら、そのバックエンドでは何にも
読まれていませんでした。**このリリースの 60 件のプルリクエストはすべてメンテナ以外から寄せられたもの**で、
22 人によるものです。

- **破壊的変更：未知の設定キーがあると、読み込みが拒否されるようになりました。**
  v0.74 で警告を出し、このリリースを期限として明示していました。`quantizaton` のようなタイプミスや、
  より新しい Soup にしか存在しないキーは、以前は無視され、その設定が適用されないまま実行が続いていました。
  現在は CLI（終了コード 1）でも API（`ValueError`）でも失敗し、おそらく意図していたフィールド名を示します。
  検出器は、スキーマが v0.40.1 からサポートしているルートレベルの `lora:` の再マッピングを適用するため、
  この書き方は拒否されずに受け入れられます。この書き方を使っていた二つの `soup fetch examples`
  ファイルは、正規の `training.lora` に移行しました。すべてのレシピとテンプレートは問題なく読み込まれ、
  キー名は端末に届く前にエスケープされ、走査には上限が設けられています。
- **MLX は、受け入れた設定に従うようになりました。** `train_on_responses_only`、`warmup_ratio` /
  `scheduler` / `weight_decay` / `optimizer`、`max_grad_norm`、`gradient_accumulation_steps`
  と `gradient_checkpointing` は、`backend: mlx` ではそれぞれ検証された後に破棄されていました。
  32 個のオプティマイザ名のうち MLX に対応するものは 8 個だけで、残りの 24 個は黙って AdamW になる代わりに
  名前を示して拒否されます。MLX はさらにライブダッシュボード、トラッカー、`soup ui` も駆動し、
  `soup doctor --config` はバックエンドが読まない設定を一覧表示します。
- **検証損失はどこにも存在していませんでした。** すべてのバックエンドで計算されては捨てられており、
  メトリクスの列もイベントのフィールドもなく、パネルにも何も表示されませんでした。現在は記録され、
  ストリーミングされ、表示されます。
- **破壊的変更：`grpo_variant: gspo` は、公開されている系列レベルの目的関数になりました**
  （arXiv:2507.18071）。パディングトークンが、同じ列を共有するすべての行の勾配までずらしてしまう
  列中心化のヒューリスティックを置き換えるものです。既存の gspo 設定では、以前の実行は再現されません。
- **Web UI の読み取りエンドポイントと SSE に認証が必要になりました**。クエリ文字列に含めるトークンではなく、
  短命の使い捨てチケットを使います。`--public` は `/docs` と `/openapi.json` を LAN に公開しなくなり、
  出力を誰も読まない場合に学習サブプロセスがハングすることもなくなりました。
- **`torch>=2.6.0`** により、v0.74.0 の既知の制限が解消されます。2.5.1 では `trl>=0.29` をインポートできず、
  すべての選好学習トレーナーが動作しませんでした。あわせて修正：`training.loraplus_lr_ratio` を設定した
  実行はすべてクラッシュしており、`packing: true` は TRL 0.29 でエラーになっていました。

> 対応は Python **3.10–3.12** のみです。3.13 以降では、Soup が動き出す前からネイティブ拡張でクラッシュする、
> テストされていない PyTorch の wheel を pip が解決してしまっていました。

過去のハイライトは [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) ページにあります。

## クイックスタート

### 1. インストール

Soup はコマンドラインアプリケーションです。そのため、最もクリーンなインストール方法は、Soup に専用の環境を与え、
`soup` を `PATH` に追加するものです：

```bash
# 軽量コア：CLI + 設定 + データツール、PyTorch なし
pipx install soup-cli
uv tool install soup-cli          # 同じ考え方です（すでに uv を使っている場合）

# 学習スタックを追加（torch, transformers, peft, trl, datasets, …）
pipx install "soup-cli[train]"

# すべて（train + serve + ui + data）を一度に
pipx install "soup-cli[all]"

# または GitHub から（最新の開発版）
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

すでに virtualenv、Colab ノートブック、Docker イメージの中にいますか？ その場合は、同じ名前とエクストラで
`pip` を直接使ってください：

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

自分のコードからも `import soup_cli` したい場合は、`pipx` ではなく `pip` を使ってください。
pipx は意図的に、アプリケーションをそれ以外のすべてから隔離するためです。

エクストラの完全な一覧（`fast`、`mlx`、`serve`、`eval`、`ui`、`vision`、`audio`、…）は
[`docs/models.md`](docs/models.md#optional-extras) にあります。

> **`error: externally-managed-environment` が出ましたか？** これは
> [PEP 668](https://peps.python.org/pep-0668/) によるもので、Soup の問題ではありません。Debian 12、
> Ubuntu 23.04 以降では、これらのファイルを `apt` も管理しているため、`pip` がシステムの Python に
> 書き込むことを防いでいます。`pipx` と `uv tool` は Soup に専用の環境を与えることでこれを回避します。
> 上で最初に挙げているのはそのためです。`python3 -m venv .venv && source .venv/bin/activate`
> の後に通常の `pip` を使っても、同じように動作します。

> **シングルクォートではなく、ダブルクォートを。** `"soup-cli[train]"` は、すべてのシェル —
> `cmd.exe`、PowerShell、bash、zsh — で動作する唯一の書き方です。古いチュートリアルから
> `'soup-cli[train]'` をコピーして pip に拒否された場合、それが原因です：
> [理由と正確なエラー内容](docs/models.md#quoting-the-extra)。

`soup init`、`soup data …`、その他のデータ／検査用コマンドは、軽量インストールで動作します。
ファインチューニング（`soup train`）には `[train]` エクストラが必要です。

### 2. 設定を作成する

```bash
soup init                       # 対話型ウィザード
soup init --template chat       # またはテンプレートから始める
```

テンプレート：`chat`、`code`、`tool-calling`、`medical`、`reasoning`、`vision`、`kto`、`orpo`、
`simpo`、`ipo`、`bco`、`rlhf`、`pretrain`、`moe`、`longcontext`、`embedding`、`audio`。

### 3. 学習、テスト、公開

```bash
soup train --config soup.yaml                 # LoRA、量子化、バッチ処理 — すべて自動で処理
soup chat  --model ./output                    # モデルと会話する
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # LoRA をベースにマージする
soup export --model ./output --format gguf --quant q4_k_m   # Ollama / llama.cpp 向けの GGUF
```

その他のエクスポート先（ONNX、TensorRT、AWQ、GPTQ、BitNet）とデプロイの選択肢は
[`docs/serving-and-export.md`](docs/serving-and-export.md) にあります。

## Web UI

ブラウザのほうがお好みですか？ `soup ui` は、実験、学習のセットアップ、ライブメトリクス、データセットの探索、
モデルとのチャットのためのローカルダッシュボードを提供します。

```bash
pip install "soup-cli[ui]"
soup ui
# http://127.0.0.1:7860/?token=<token> を開きます
```

![Soup Web UI — 新規トレーニング](docs/assets/web-ui-new-training.png)

[Web UI のドキュメント](docs/serving-and-export.md#web-ui)

## 設定

完全な `soup.yaml` の例：

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2〜5 倍高速、pip install "soup-cli[fast]"

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

`config/schema.py` は、すべてのフィールドについての唯一の信頼できる情報源です。高度なデータ・学習・PEFT
オプションは [ドキュメント](#ドキュメント) で解説しています。

> **未知の設定キーは v0.75 以降拒否されます。** どのスキーマモデルも宣言していないキー — `quantizaton`
> のようなタイプミスや、より新しい Soup にしか存在しないフィールド — は、以前は検証を問題なく通過したうえで
> 破棄されていました。つまり、その設定がまったく適用されないまま実行が進んでいました。v0.74 では、
> 読み込み時に、おそらく意図していたフィールド名とともにこれを報告していました。**v0.75** からは同じ設定の
> 読み込みが失敗するため、キーが無視されることに頼らず、キーを修正するか削除してください。
> [未知の設定キー](docs/backends-and-ops.md#unknown-config-keys) を参照してください。

## ドキュメント

機能の完全なリファレンスは [`docs/`](docs/) にあります。まずはここから：

| ガイド | 内容 |
|---|---|
| [学習タスクと手法](docs/training.md) | SFT、DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO、ツール呼び出し、PRM、事前学習、蒸留、分類、ビジョン／音声／TTS、アンラーニング、RAFT/RA-DIT、学習ループ堅牢化（ループハードニング）の検出器 |
| [PEFT、ロングコンテキスト、効率化](docs/peft-and-efficiency.md) | DoRA、LoRA+、rsLoRA、VeRA、OLoRA、NEFTune、PiSSA、ReLoRA、オプティマイザと PEFT のコレクション、LLaMA Pro、GaLore、YaRN/LongLoRA、パッキング、カリキュラム学習、自動チューニング |
| [パフォーマンスと量子化](docs/performance-and-quantization.md) | QAT、FP8、Quant Menu（I + II）、KV キャッシュ、NVFP4、保存形式、Cut Cross-Entropy、勾配チェックポイント、カーネル、アクティベーションのオフロード、レイヤーストリーミング、マルチ GPU / DeepSpeed / FSDP |
| [データエンジニアリング](docs/data.md) | フォーマット、Axolotl/LF と同等のパイプライン、データツール、合成データ生成と forge、品質スコアカード、トレース（trace）ツール、リモートデータセット、ミキシング、レシピ DAG |
| [評価とプローブ](docs/evaluation.md) | 評価の設計／ゲート、評価ゲート付き学習、ベンチマーク、NLG 指標、キャリブレーション、Elo アリーナ、診断、学習後の X 線プローブ、A/B、ドリフト、チューニング可能性、`soup advise` |
| [サービングとエクスポート](docs/serving-and-export.md) | OpenAI 互換サーバー、バッチ推論、ベンチマーク、マージ／エクスポート、Anthropic Messages エンドポイント、投機的デコーディング（独自のドラフトモデルを学習 + 計測）、デプロイのオートパイロット、Web UI、Agent Forge |
| [アダプター、レジストリ、ガバナンス](docs/adapters-and-governance.md) | アダプターのライフサイクル／管理、モデルレジストリ、Soup Cans、データフライホイール（`soup loop`）、知識編集、ステアリング（steering）、サプライチェーンのセキュリティ対策（scan/sign/BOM/attest/audit/airgap） |
| [コンプライアンスとガバナンスのクイックスタート](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 向けの `init` テンプレート、来歴情報（BOM/attest/repro-receipt）、監査ログ、エアギャップ、モデルカードの自動生成（`soup card`）、CI ゲート（`soup ci init`） |
| [バックエンド、プラットフォーム、運用](docs/backends-and-ops.md) | MLX/Unsloth バックエンド、代替ハブ、HF Hub 連携、オートパイロット、実験トラッキング、plan/apply、環境ロックファイル、ハードウェア適合性、シェル補完、プラグイン、ユーティリティコマンド |
| [コマンドリファレンス](docs/commands.md) | `soup` コマンドの完全な一覧 |
| [対応モデルとエクストラ](docs/models.md) | 推奨モデルファミリー、VRAM サイズガイド、pip エクストラのマトリクス |

## データ形式

Alpaca、ShareGPT、ChatML、選好ペア（DPO / ORPO / SimPO / IPO / KTO）、ビジョン、音声、ASR、プレーンテキスト、
埋め込み（embedding）、RAFT など — すべて JSONL、JSON、CSV、Parquet、TXT から自動検出されます。
そのため、ほとんどの場合は `data.train` をファイルに向けるだけで、他に変えるものは何もありません。
形式ごとの実例付きスキーマと、データパイプライン（リモート URI、ストリーミング、シャーディング、
インターリーブ、語彙拡張、ドキュメントの取り込み）は [`docs/data.md`](docs/data.md#data-formats) にあります。

## よく使うコマンド

```bash
soup train  --config soup.yaml        # 学習（SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...）
soup infer  --model ./output --input prompts.jsonl --output results.jsonl   # バッチ推論
soup chat   --model ./output          # 対話型チャット
soup serve  --model ./output          # OpenAI 互換 API サーバー
soup ui                               # ローカルのブラウザダッシュボード
soup merge  --adapter ./output        # LoRA をベースモデルにマージする
soup export --model ./output --format gguf           # デプロイ用にエクスポートする
soup eval   benchmark --model ./output               # 評価する
soup data   inspect ./data/train.jsonl               # データセットの統計
soup recipes list                     # 100 以上の既製モデルレシピ
soup autopilot --model <id> --data d.jsonl --goal chat  # 設定不要
soup doctor                           # GPU / 依存関係 / 環境をチェックする
```

完全なコマンド一覧は [`docs/commands.md`](docs/commands.md) にあります。

## 対応モデル

Soup は [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) 上の
**あらゆる**テキスト生成モデルで動作します — `AutoModelForCausalLM` で読み込めるなら、設定を一切変えずに
動作します。Llama 3.x/4、Qwen 2.5/3、Gemma 3、Mistral、Mixtral、DeepSeek R1/V3、Phi-4 をはじめ、
100 以上のモデルが既製レシピとして提供されています（`soup recipes list`）。

| VRAM | 最大モデル（QLoRA 4-bit） | 例 |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+（フル）または MoE | Mixtral-8x22B, DeepSeek-V3 |

モデルとビジョンの完全な表、およびオプションのエクストラのマトリクスは [`docs/models.md`](docs/models.md) にあります。

## Docker

CUDA や PyTorch をローカルにインストールせずに Soup を実行できます（イメージはリリースごとに GHCR に公開されます）：

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # またはローカルでビルドする
```

## 動作要件

- Python 3.10、3.11、3.12（CI でテストしているのはこれらのバージョンです。PyTorch スタックがまだ検証されて
  いないため、3.13 以降は現時点では対応していません）
- CUDA 対応 GPU（推奨）、Apple Silicon（MPS）、または CPU（実験的 — 非常に低速）
- QLoRA で 7B モデルを扱うには 8 GB 以上の VRAM

すべての学習タスクは、テスト用に CPU 上でも実行できます（量子化は自動的に無効になります）。オプションの
エクストラ（`train`、`all`、`fast`、`vision`、`qat`、`serve`、`serve-fast`、`ui`、`eval`、`deepspeed`、
`liger`、`mlx`、`onnx`、`tensorrt`、…）の一覧は [`docs/models.md`](docs/models.md#optional-extras) にあります。

## トラブルシューティング

```bash
soup doctor    # GPU、システムリソース、依存関係、バージョンを一か所で確認
```

CUDA の wheel、バージョンの不一致：[`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting)。

## 開発

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # ユニットテスト（高速、GPU 不要）
pytest tests/ -m smoke -v          # スモークテスト（小さなモデルをダウンロードして学習する）
pytest tests/ -m gpu --no-cov -v   # GPU テスト（CUDA カードが必要。結果を報告してください。CONTRIBUTING.md を参照）

pre-commit install                 # 任意：コミット時に ruff で lint + format
```

全体のワークフローは [CONTRIBUTING.md](CONTRIBUTING.md) を、脆弱性の報告については [SECURITY.md](SECURITY.md)
を参照してください。テレメトリーは完全にオプトインです（`SOUP_TELEMETRY=1`、デフォルトはオフ。
[プライバシーポリシー](docs/backends-and-ops.md#privacy-policy) を参照）。

## Soup を支援する

Soup は Apache-2.0 ライセンスの無料ソフトウェアであり、今後もそうあり続けます。4 GB のノート PC 一台で
オープンに開発・保守されており、このドキュメントの性能数値がすべて、主張ではなく計測によるものなのはそのためです。

Soup のおかげで学習の実行を一回でも節約できたなら、[リポジトリにスターを付けていただく](https://github.com/MakazhanAlpamys/Soup)
ことが最も助けになり、費用もかかりません。作業に直接資金を提供したい場合は：

**[❤️ 寄付する](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — 一回限りで、金額は自由です（チェックアウト
ページで *Change amount*（金額を変更）を使ってください）。決済は、メンテナの登録事業者である **MePlay, Inc.**
の名義で Stripe によって処理されます — チェックアウトページやカードの利用明細に表示されるのは「Soup」ではなく、
この名前です。

寄付は、ハードウェアに依存する作業 — マルチ GPU、8B 以上での検証、Apple Silicon — のための GPU 時間に充てられます。
いずれも、4 GB のノート PC 一台では手の届かない作業です。

まさにこれらの項目を前に進めるもう一つの方法は、**ハードウェアそのもの**です。これらは検証されていない主張としてではなく、
「\<ハードウェア\>が必要」と動作要件を明記したゲートの背後で公開されています。そのため、より大きなマシン —
あるいは使われていない GPU クレジット — を利用できるなら、
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
の Issue のいずれかに記載された実験を実行して数値を投稿していただくことは、GPU 時間に資金を提供するのと同じくらい
助けになります。これらの Issue には、現在ハードウェアが理由で何が止まっているのかが正確に書かれています。

## コントリビューター

コミュニティによって作られています ❤️ — 貢献してくださったすべての方に感謝します。
[CONTRIBUTORS.md](CONTRIBUTORS.md) をご覧ください。

[![コントリビューター](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## お問い合わせ

バグ報告や機能リクエストは [Issue トラッカー](https://github.com/MakazhanAlpamys/Soup/issues) へ、質問は
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions) へお寄せください — どちらもより早く回答が得られ、
同じ問題を抱える次の人の助けにもなります。

ライブチャット、セットアップの手助け、そして会話のほうが向いているあらゆることについては、
[Discord](https://discord.gg/dgd2pJcjwP) または [Telegram コミュニティ](https://t.me/souptasters) に参加してください。
六か月後にも見つけられるべき内容は、Issues か Discussions へお寄せください — Discord での回答は一人の助けにしか
なりませんが、Issue は同じことに直面するすべての人の助けになります。そこでも [行動規範](CODE_OF_CONDUCT.md) が適用されます。

公開の場にふさわしくない事柄 — セキュリティ報告（[SECURITY.md](SECURITY.md) を参照）、行動規範に関する問題、
報道関係 — については、**team@trysoup.dev** までメールでご連絡ください。これはプロジェクトのアドレスで、Soup に
関するあらゆる用件に適した連絡先です。**makazanalpamys@gmail.com** はメンテナの個人アドレスで、同じ人物に届くため、
代わりの連絡先として使っていただいても問題ありません。

## Soup の引用

レイヤーストリーミング — 凍結したベースモデルをホストの RAM からデコーダ層一層ずつストリーミングすることで、
4 GB のノート PC 向け GPU で 8B モデルを学習する手法 — は、ストリーミングによる実行をメモリ常駐の実行と照合して
検証する正確性プロトコルとともに、プレプリントで解説されています（順伝播と逆伝播は別々に記述しています。
これらは一つではなく、二つの主張だからです）。

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**バージョン 3（2026 年 8 月 13 日）が最新版です。** タイトルと主張 — 4 GB で 8B — は変わっておらず、v1 以降、
計測値は一つも変わっていません。v3 で行ったのは、**私たちが公開していた説明の撤回**です。これは同時に、
本論文が何のためにあるのかを説明する最も簡潔な方法でもあります：

- **v3 で撤回：「レイヤーストリーミングのボトルネックは GPU ではなく、ホストからデバイスへの転送である」。**
  これは下記の H100 での再現から導いた*推論*であり、一度も計測されていませんでした。8 月 11 日に計測したところ、
  公開した構成ではこれは誤りでした。ホストからデバイスへのバイトをすべて削除して得られる改善は **1.4%**、
  計算ストリームがコピーを待っているのはステップの **0.20%** で、ステップはそのカードの同一セッションにおける
  GEMM 上限の **71.3%** で動作しています。ストリーミング固有のコストで最も大きいのは、層ごとの NF4 逆量子化で、
  9.8% です（[記録](benchmarks/probe-v0.73.0-what-bounds-streaming.md)）。すべての計測結果は有効なままです。
  再現はより弱い形で成り立ちます — 制約は両方のマシンに共通しており、GPU の計算能力ではありません。
- **元の環境とはまったく異なるハードウェアでの再現**（v2 で追加）：RTX 3050 での 119.6 tok/s に対し、H100 では
  中央値 113.00 で、ピークは同じ 3.32 GB でした。どちらも #331 の修正より前のもので、4 GB での再計測は
  Issue #361 で保留中です。
- **表に現れない、誤った勾配の不具合を発見し、修正しました。** 層あたり約 165 MiB を超える NF4 では、順伝播は
  ビット単位で一致したままで、損失曲線も健全に見えていましたが、勾配は誤っていました。原因は上流ライブラリに
  あると特定し、そちらに報告済みです。修正は、実際の 32B と 72B で対照と照合するゲートを通しています。
- **実際のモデルサイズでのビット単位の一致**（三層のおもちゃのモデルではなく）：順伝播は 0.5B から 72B まで、
  逆伝播は 8B と 14B で確認しています。
- **学習済みモデルの品質を初めて計測**し、メモリ常駐の実行と区別できないことを確認しました。
- **DeepSpeed との比較** — 私たちに不利な結果も含みます：八枚のカードによる ZeRO-3 は、メモリ常駐で学習する
  一枚のカードよりも遅いという結果です。
- **制限事項の章を書き直しました**：v1 の十項目のうち一つが解消し、さらに四つの範囲が狭まり、新たに七つが加わりました。

使用したバージョンを引用してください。`10.5281/zenodo.21771064` はコンセプト DOI で、常に最新版（現在は v3）に
解決されます。v1 と v2 はそれぞれのバージョン DOI で引き続き引用でき、編集されることはありません — 上記の撤回を
新しいバージョンとして出したのは、まさに、私たちがいつ何を主張したかの記録をそのまま残すためです。

本論文のすべての数値の裏付けとなる計測記録は、書かれたままの形で [`benchmarks/`](benchmarks/) に公開されています —
失敗、誤りだとわかった仮定、計測した後に破棄した数値も含めてです。

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

## ライセンス

[Apache-2.0](LICENSE)。Copyright © Soup コントリビューター。
