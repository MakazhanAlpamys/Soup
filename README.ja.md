<!-- synced-from: README.md sha256:aa2170df59636760e0235b87cddf967a14e932605ed3abb16480115d99db6e40 -->
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
  <img src="https://img.shields.io/badge/python-3.11--3.13-blue" alt="Python 3.11-3.13">
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
**4 GB のノート PC 用 GPU で、8B モデルをファインチューニングできます。** レイヤーストリーミングは、凍結したベースモデルを
VRAM の外に置き、デコーダー層を一層ずつ GPU に送り込みます。RTX 3050 Laptop 4 GB での実測値:
Llama-3.1-8B-Instruct + NF4 で **952 MHz 時に 119.6 tok/s、ピーク 3.32 GB**（修正前）— 全体をメモリに常駐させた通常の実行とビット単位で
完全に一致し、H100 でも同じ 3.32 GB で 113.00 tok/s と独立に再現されました。
（どちらの数値も v0.72.2 で、v0.73.0 の正確性修正（32B で −4.8% のコスト）より前に測定したものです。
修正後の実行は、別の RTX 3050 Laptop 4 GB（@umran666、#1103）で
**1935–1957 MHz 時に 208.6 tok/s、ピーク 2.40 GB** でした。これは同一セッションでの GEMM 上限の 59% にあたり、
修正前の 68% に対して、クロックあたりでは約 14% 遅いことになります。つまり、見かけの 1.74x の向上は高速化ではなく
クロックの差です。大きなレイヤーのストリーミングによって両行のあいだでメモリの使われ方も変わっているため、この割合でさえ
同じ条件での比較ではありません — [記録](benchmarks/gate-v0.72.2-nf4.md)。）オプトイン方式（`stream_layers: true`）で、
まだ BETA です —
[仕組み](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[全計測結果](benchmarks/) · [論文](https://doi.org/10.5281/zenodo.21771064) ·
**[無料の Colab T4 で自分で確かめる](notebooks/proof-4gb.ipynb)**（プロセスを 4 GB に制限したうえで、
ストリーミングしたモデルが通常のモデルとビット単位で同一であることを検証します）

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="4 GB のカードで Llama-3.1-8B を対象にした soup train の事前チェック：32 層にわたって RAM に固定された 3.60 GB のベースストアと、113 MB の VRAM バッファ二つ。続いて 119.6 tok/s でピーク 3.32 GB を計測し、4 GB のラインを下回っています（v0.72.2 で計測、#331 の修正前。再計測は Issue #361 で保留中）"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4、LoRA、バッチ 1、シーケンス長 512、RTX 3050 Laptop 4 GB — <b>ピーク 3.32 GB、119.6 tok/s</b>（v0.72.2 で計測、#331 の修正前。再計測は Issue #361 で保留中）。<a href="https://youtu.be/T1LCErE943E">フル動画（90 秒）</a></sub>
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="4 GB のカードでの Llama-3.1-8B 向け soup train の事前チェック: 32 層にわたって RAM 上に固定された 3.60 GB のベースストアと、113 MB の VRAM バッファ 2 つ。その後、119.6 tok/s で測定されたピークは 3.32 GB で、4 GB のラインの手前に収まっています（修正前の行。v0.72.2、952 MHz で測定。#331 修正後の行は下のキャプションを参照）"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4、LoRA、バッチサイズ 1、シーケンス長 512、RTX 3050 Laptop 4 GB — <b>ピーク 3.32 GB、119.6 tok/s</b>、952 MHz 時（修正前、v0.72.2。#331 修正後: 1935 MHz 時に 208.6 tok/s、2.40 GB。直接の比較はできません。@umran666、#1103）。 <a href="https://youtu.be/T1LCErE943E">動画（90 秒）</a></sub>
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
**v0.76.0 — Soup が受け入れながら一度も適用していなかった設定は、読み込みを止めるようになりました。**
検証され、文書化されていながら、設定対象と称する経路のどこからも読み取られていなかったオプションは、
読み込み時に拒否され、設定名と変更すべき内容が示されます。それ以外は警告を出し、拒否することになる
リリースを明示します。以前は読み込めた設定が読み込めなくなることがあります。まず
`soup doctor --config soup.yaml` を実行してください。

- **破壊的変更: 一度も効果のなかった設定は、読み込み時に拒否されます。** 対象には、
  `training.quantization_aware: true`（量子化対応学習は実際には一度も実行されていませんでした）、
  `training.use_longlora`、`training.ebft_variant`、`training.gdpo_variant`、どのセットアップも読み取らない
  場合の `training.rope_scaling_type`、`freeze_layers` / `freeze_ratio`（transformers バックエンドの
  テキスト `sft` / `tts` 経路以外）、`auto_mixed_precision`（`sft` / `tts` 以外）、そして unsloth の
  セットアップを持たないさらに七つのタスクでの `backend: unsloth` が含まれます。各メッセージは、該当する
  フィールドと変更すべき内容を示します。`lr_groups`、`early_stop_patience`、`forgetting_eval_steps` など、
  読み取られない他のつまみは、今は読み込み時に警告を出し、v0.77 から拒否されます。
- **検証損失が評価されるようになりました。** `data.val_split` はすべての実行で行を学習から除外していましたが、
  transformers バックエンドのどのトレーナーもそれらを評価していませんでした。今は、分割を受け取るすべての
  タスクが、各エポックの終わりに、または `training.eval_steps` を指定した場合は N ステップごとにそれを評価し、
  検証損失を記録します。実行時間は数パーセント長くなります。**`grpo`、`ppo`、`online_dpo` では破壊的変更です。**
  既定では行を取り分けなくなり（`data.val_split` は注記付きで無視されます）、すべての行が学習に使われます。
- **破壊的変更: 標準でないすべての `grpo_variant` が `grpo_beta` の KL ペナルティを適用します。**
  `gspo`、`dapo`、`dr_grpo`、`bnpo`、`two_sided`、`rft` は `grpo_beta` を検証しながら捨てていたため、
  既存のバリアント設定は以前の実行を再現できません。`grpo_beta: 0` は KL なしで動作し、参照モデルの
  フォワードパスを省略するようになりました。
- **レイヤーストリーミングは、ディスク階層をバックグラウンドスレッドでダイレクト I/O により読み込み、
  NVMe ドライブにまたがってストライプ化できます。** RAM より大きい 36 GB の 70B 相当の NF4 ストア
  （RTX 5070 Laptop 8 GB 搭載の一台）で、コールドステップは同期読み込みの 110.5 s から 16.2-17.3 s に短縮され、
  `SOUP_LAYER_STREAM_STRIPE_DIRS` による 二台のドライブへの分散では 17.4 s から 9.97 s になりました
  （持続負荷プローブ: 二ラウンド、1.28x と 1.41x、正式な判定なし。
  [記録](benchmarks/probe-rtx5070-two-drive-sustained.md)）。RAM に収まるストアは、引き続き RAM 階層のほうが
  高速で、`stream_source: auto` はそちらを選びます。オプトイン方式（`stream_layers: true`）で、まだ BETA です。
  記録: [ダイレクト I/O](benchmarks/gate-974-disk-tier-direct-io.md)、
  [ストライピング](benchmarks/gate-two-drive-striping.md)。
- **新しいコマンドとオプション。** `soup adapters check` は LoRA アダプターをスキャンして、サイレントな失敗を検出します。
  `soup bench train` は決まったステップ数だけ学習し、モデルが学習していなかった場合は終了コード 1 で終了します
  （transformers バックエンドの `task: sft` が対象です。`soup bench` はグループになり、`soup bench infer` が
  従来の推論ベンチマークです）。`soup data clean` は、入力ファイルに触れずにデータセットをサニタイズして
  修復します。`soup recipes verify` は、重みをダウンロードせずに meta デバイス上で、同梱のすべてのレシピが
  LoRA アダプターを取り付けられることを確認します。`soup rewind` は、損失スパイクの原因となったデータセットの
  行を特定します。`soup infer` には、オプトインの `--batch-size` と実験的な `--cuda-graphs` が加わりました
  （PyTorch 2.14 以降が必要。RTX 5070 Laptop 一台、ピン留めなし、FP16 の Qwen2.5-1.5B で、これを使った最も遅い実行でも、
  使わなかった最も速い実行より 2.9x 速くデコードできました）。`soup ab` は、分散に頑健な両側検定を実行します。
  `soup adapters sign --backend sigstore` と `soup attest emit --sign sigstore` は、Sigstore のキーレス OIDC で
  署名します（`pip install "soup-cli[sigstore]"`）。
- **非推奨:** `soup export --format awq` と `--format gptq` は v0.77 で削除されます。どちらの
  上流プロジェクトもアーカイブ済みで、どちらの extra も `[train]` と一緒にはインストールできません。

> 対応は Python **3.10–3.12** のみです。3.13 以降では、Soup が動き出す前からネイティブ拡張でクラッシュする、
> テストされていない PyTorch の wheel を pip が解決してしまっていました。
**v0.77.0 — v0.76 が警告していたものは、拒否されるようになりました。** 何からも読み取られない
31 個の設定フィールドと、非推奨の三つの値は、警告付きで読み込まれることがなくなりました。読み込みは、
設定ごとに一行で、その設定名と削除すべきものを示して停止します。v0.76 では警告付きで、あるいは黙って読み込めた設定が、
v0.77 では読み込めなくなることがあります。まず `soup doctor --config soup.yaml` を実行してください。

- **破壊的変更: ステージされたフィールドと非推奨の値は拒否されます。** 対象のフィールドには、
  `training.lr_groups`、`early_stop_patience`、`forgetting_eval_steps`、`checkpoint_eval_*`、
  `quantize_ref_model`、`long_context_grpo`、`vision_grpo`、`data.split_thinking`、そして
  `data.video_*` キーが含まれます。`training.rl_checkpoint_include_ref_model` と
  `rl_checkpoint_include_rollout_buffer` は v0.76.0 より後にステージされたため、事前に警告した
  リリースはありません。値は、`data.remove_unused_columns: true`、量子化を一切行わないタスクでの
  `training.quantization: 4bit` / `8bit`（または `load_in_8bit: true`）、そして `soup ab --beta` です。
  v0.76 が何も言わずに無視していた十四個の設定も拒否されます。たとえば何も適用しない場所での `use_mod` や
  `neftune_alpha`、LoRA ランク付きの `expand_layers`、Unsloth での `8bit` やビジョン SFT、MLX または
  Unsloth での `lora.rank_pattern` です。CHANGELOG のアップグレードノートを参照してください。
- **破壊的変更: Python 3.11–3.13、AWQ と GPTQ のエクスポートは廃止、`soup serve` の二つのルートに
  トークンが必要に。** Python 3.10 はサポート対象外になりました（2026-10-01 にサポート終了）。3.10 では
  pip は引き続き v0.76.x を解決します。`soup export --format awq` / `gptq` と
  `soup quantize --to awq` / `gptq` は削除されました。どちらの上流プロジェクトもアーカイブ済みのためです。
  `gguf`、`onnx`、`tensorrt`、`bitnet`、`tq1_0` を使うか、v0.76.x にとどまってください。
  `/v1/tools/python` と `/v1/tools/bash` は、サーバーが `--tool-auth-token`（または
  `SOUP_TOOL_AUTH_TOKEN`）付きで起動されていなければ、どのバインドでも 401 を返します。また
  `/v1/tools/python` は、ホストに OS レベルの分離がない環境では、bash と同じく 501 を返します。
- **DPO、ORPO、SimPO、KTO が Apple Silicon でネイティブに動作します。** `backend: mlx` での
  `task: dpo`、`orpo`、`simpo`、`kto` は、Soup 独自の MLX ループで TRL の目的関数に従って学習します
  （ORPO と SimPO は参照モデル不要）。MLX のテキスト SFT は、ネイティブ DoRA、`training.eval_steps`、
  `training.seed`、損失ウォッチドッグに対応しました。`lora.use_rslora` は MLX と Unsloth で適用される
  ようになったため、そこでの rsLoRA 実行は要求したスケールで学習し、結果が変わります。
- **レイヤーストリーミングが DeepSeek-V3 系の MoE に対応しました。** Soup のレイヤーストリーミング
  （L2L の実装、Pudipeddi et al. 2020）は、`model_type: deepseek_v3` を、常駐とストリーミングで
  フォワードとバックワードの一致（パリティ）を確認したうえでストリーミングし、FP8 ブロック量子化
  チェックポイントをシャード時に逆量子化します。Kimi-K2-Base と -Instruct は、フィールドごとの検査を
  通ったうえで受け入れられます（K2 形状の小さなチェックポイントが CPU 上でビット単位で一致して
  ストリーミングされます。K2 そのものでは何も学習していません）。
  `training.stream_expert_layout: per_expert` は、ルーティングされたエキスパートを一つずつストリーミング
  します。RTX 5070 Laptop 8 GB 一台で、同じ Moonlight-16B の LoRA 設定は、fused の 3.1 GB に対して
  GPU メモリのピークが 1.0 GB（エキスパートは bf16 に対して NF4、キャッシュ階層も異なる）で、ステップ時間は約 2.7x でした
  （[記録](benchmarks/probe-rtx5070-moonlight-16b.md)）。`training.stream_recompute: off` / `auto` は、
  ストリーミングされるレイヤーのアクティベーションを再計算せずに保持し、勾配はビット単位で一致します。
  同じカードの RAM 階層では、Mistral-7B NF4 のステップが 512 トークンで 22%、2048 で 8.5% 速くなります
  （[記録](benchmarks/gate-stream-recompute.md)）。オプトイン方式（`stream_layers: true`）で、BETA です。
- **検証できる来歴（provenance）。** `soup registry push --data <file>` は学習ファイルの SHA-256 を
  記録し、`--idempotency-key` によりプッシュを安全に再試行できます。transformers バックエンドでは、常駐、
  レイヤーストリーミング、マルチ GPU の各実行が、読み込んだベースモデルの Hub コミットを記録します。
  `soup ship --require-provenance` は、エビデンスを、それに刻まれたアダプター・タスク・ソースのバイト列、
  スイートのバージョン、判定ポリシーと照合します。
- **その他の新機能。** `soup --version`。`soup data ingest` で PDF と DOCX を扱うための
  `pip install "soup-cli[documents]"`。`soup merge --low-memory`。GRPO が LLM ジャッジで採点できる
  ようになりました（`training.reward_fn: judge_rank`）。そして
  [`docs/config-reference.md`](docs/config-reference.md) と
  [`docs/cli-reference.md`](docs/cli-reference.md) はコードから生成され、どちらかがコードの動作を正しく
  述べなくなると、テストスイートが失敗します。

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
その他のエクスポート先（ONNX、TensorRT、BitNet）とデプロイのオプションは、
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
埋め込み、RAFT など — すべて JSONL、JSON、CSV、Parquet、TXT から自動検出されるため、ほとんどの場合は
`data.train` をファイルに向けるだけで、他は何も変わりません。各形式の実例つきスキーマと、データパイプライン
（リモート URI、ストリーミング、シャーディング、インターリーブ、語彙拡張、ドキュメント取り込み）は、
[`docs/data.md`](docs/data.md#data-formats) にあり、[Sesame CSM のネイティブ音声学習](docs/training.md#native-sesame-csm)は別途説明されています。

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
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml --yes
docker compose up   # またはローカルでビルド
```

## 動作要件

- Python 3.11、3.12、または 3.13（CI がテストしているバージョンです。PyTorch スタックがまだ検証されて
  いないため、3.14 以降は未サポートです）
- CUDA 対応 GPU（推奨）、Apple Silicon（MPS）、または CPU（実験的 — 非常に遅い）
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
**バージョン 3（2026 年 8 月 13 日）が最新版です。** タイトルと主張 — 4 GB で 8B — は変わらず、v1 以降に
測定値が変わったものはありません。v3 が行っているのは、**私たちが公開していた説明の撤回**であり、これは
この論文が何のためにあるのかを最も短く言い表す方法でもあります。

- **v3 で撤回: 「レイヤーストリーミングの律速は GPU ではなく、ホストからデバイスへの転送である」という説明。**
  これは、下記の H100 での再現からの*推論*であり、実際に測定されたことはありませんでした。8 月 11 日に測定
  したところ、公開された構成ではこれは誤りでした。ホストからデバイスへのバイト転送をすべて取り除いても
  得られるのは **1.4%** にすぎず、計算ストリームがコピーを待つ時間はステップの **0.20%**、ステップはその
  カードの同一セッションでの GEMM 上限の **71.3%** で動作しています。ストリーミング固有の最大のコストは、
  レイヤーごとの NF4 逆量子化で、9.8% です
  （[記録](benchmarks/probe-v0.73.0-what-bounds-streaming.md)）。すべての測定値は有効なままです。再現は弱い形で
  残ります — この制約は両方のマシンに共通であり、GPU の計算能力ではありません。
- **元とはまったく異なるハードウェアでの再現**（v2 で追加）: RTX 3050 で 119.6 tok/s（952 MHz 時）、H100 で中央値
  113.00 tok/s、ピークはどちらも 3.32 GB です。どちらも #331 の修正より前のもので、論文が報告しているのはこの数値です。
  #331 修正後の行（@umran666、#1103）は、1935–1957 MHz 時に 208.6 tok/s、ピーク 2.40 GB で、同一セッションでの
  GEMM 上限に対する割合は 59%（修正前は 68%）です。見かけの 1.74x の向上は高速化ではなくクロックの差であり、
  両行のあいだでメモリの使われ方も変わっているため、この割合でさえ同じ条件での比較ではありません
  （[記録](benchmarks/gate-v0.72.2-nf4.md)）。
- **サイレントな勾配誤りの欠陥を発見し、修正しました。** レイヤーあたり約 165 MiB を超える NF4 では、
  順伝播はビット単位で一致したままで、損失曲線も健全に見えましたが、勾配が誤っていました。原因は上流
  ライブラリで特定され、そこに報告されています。修正は、実際の 32B と 72B でのコントロールに対して
  ゲートされています。
- **三層のおもちゃではなく、実際のモデルサイズでのビット単位の一致**: 順伝播は 0.5B から 72B まで、
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

[Apache-2.0](LICENSE)。Copyright 2026 Makazhan Alpamys and the Soup contributors。Soup またはその改変コピーを再配布する場合は、ライセンスの第4条と第6条が適用されます。コピーごとに `LICENSE` と `NOTICE` を保持し、変更したファイルにその旨を明記し、作品の出所を述べる範囲を超えて Soup の名称を使用しないでください。
