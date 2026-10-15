<!-- synced-from: README.md sha256:fa009c2fcc2c94ea06e25ea8620c2fbed9bfb8ac6611bf71a2669a06f7c50b46 -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <strong>العربية</strong> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center" dir="rtl">
  <strong>اضبط نماذج اللغة الكبيرة (LLMs) ضبطًا دقيقًا وأجرِ لها تدريبًا لاحقًا بأمر واحد. بلا SSH، وبلا جحيم الإعدادات.</strong>
</p>

<p align="center" dir="rtl">
  <a href="https://trysoup.dev">الموقع</a> &middot;
  <a href="#البدء-السريع">البدء السريع</a> &middot;
  <a href="#واجهة-الويب">واجهة الويب</a> &middot;
  <a href="#الإعدادات">الإعدادات</a> &middot;
  <a href="#التوثيق">التوثيق</a> &middot;
  <a href="docs/commands.md">الأوامر</a> &middot;
  <a href="docs/models.md">النماذج</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="التنزيلات"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="ترخيص Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="الاختبارات"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="الموقع"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - ضبط نموذج لغوي كبير بحجم 8B على وحدة معالجة رسومية لحاسوب محمول بسعة 4 GB | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

يحوّل Soup عناء الضبط الدقيق لنماذج اللغة الكبيرة إلى سير عمل بسيط. ملف إعدادات واحد، وأمر واحد، وانتهى الأمر.

```bash
pip install "soup-cli[train]"   # أضف [train] للضبط الدقيق؛ أما soup-cli وحدها فهي واجهة سطر الأوامر الخفيفة
soup init --template chat
soup train
```

**اضبط نموذجًا بحجم 8B ضبطًا دقيقًا على وحدة معالجة رسومية لحاسوب محمول بسعة 4 GB.** يُبقي بثّ الطبقات
النموذجَ الأساسي المجمَّد خارج VRAM ويغذّي به وحدة المعالجة الرسومية طبقةَ فك ترميز واحدة في كل مرة.
قِيس الأداء على RTX 3050 Laptop بسعة 4 GB: Llama-3.1-8B-Instruct + NF4 عند **119.6 tok/s بتردد 952 MHz وبذروة 3.32 GB** (قبل الإصلاح) —
مطابق بتًّا ببتّ لتشغيل عادي مقيم بالكامل في الذاكرة، وأُعيد إنتاجه بصورة مستقلة على H100 عند
113.00 tok/s وفي 3.32 GB نفسها. (قِيس الرقمان على الإصدار v0.72.2، قبل إصلاح صحة النتائج في v0.73.0 الذي كلّف
انخفاضًا قدره 4.8% في الأداء عند 32B. أما التشغيل بعد الإصلاح، على حاسوب محمول آخر ببطاقة RTX 3050 Laptop بسعة 4 GB (@umran666، #1103)،
فيسجّل **208.6 tok/s بتردد 1935–1957 MHz وبذروة 2.40 GB**. وهذا يعادل 59% من سقف GEMM في الجلسة نفسها
مقابل 68% قبل الإصلاح: أي أبطأ بنحو 14% لكل وحدة من تردد الساعة، فالكسب الخام البالغ 1.74x مردّه تردد الساعة
لا تسريع فعلي. وقد غيّر بثّ الطبقات الكبيرة أيضًا نمط استخدام الذاكرة بين الصفّين، فحتى هذه النسبة ليست مقارنة
متكافئة — [السجل](benchmarks/gate-v0.72.2-nf4.md).) وهو لا يُفعَّل إلا بطلب صريح (`stream_layers: true`)
ولا يزال BETA —
[كيف يعمل](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[جميع القياسات](benchmarks/) · [الورقة البحثية](https://doi.org/10.5281/zenodo.21771064) ·
**[تحقّق بنفسك على Colab T4 مجاني](notebooks/proof-4gb.ipynb)** (يحدّ العملية بسعة 4 GB، ثم يتحقق من أن
النموذج المبثوث مطابق بتًّا ببتّ لنموذج عادي)

<p align="center" dir="rtl">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="الفحص المسبق لأمر soup train لنموذج Llama-3.1-8B على بطاقة 4 GB: مخزن أساسي بحجم 3.60 GB مثبَّت في RAM عبر 32 طبقة ومخزنان مؤقتان في VRAM حجم كلٍّ منهما 113 MB، ثم ذروة مقيسة قدرها 3.32 GB عند 119.6 tok/s، متوقفةً دون خط الـ 4 GB (صفّ ما قبل الإصلاح، قِيس على v0.72.2 بتردد 952 MHz؛ وصفّ ما بعد #331 في الشرح أدناه)"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4 وLoRA وحجم دفعة 1 وطول تسلسل 512 على RTX 3050 Laptop بسعة 4 GB — <b>ذروة 3.32 GB، و119.6 tok/s</b> بتردد 952 MHz (قبل الإصلاح، v0.72.2؛ وبعد #331: 208.6 tok/s بتردد 1935 MHz في 2.40 GB، ولا تصح مقارنتهما مباشرة، @umran666، #1103). <a href="https://youtu.be/T1LCErE943E">الفيديو كاملًا (90 ثانية)</a></sub>
</p>

## لماذا Soup؟

لا يزال تدريب نماذج اللغة الكبيرة مؤلمًا. حتى الفرق ذات الخبرة تقضي ما بين 30 و50% من وقتها في مصارعة
البنية التحتية بدلًا من تحسين النماذج. ويأتي Soup ليعالج ذلك.

- **بلا SSH.** لن تحتاج بعد اليوم إلى الدخول عبر SSH إلى خادم GPU معطَّل.
- **ملف إعدادات واحد.** كل ما تحتاجه ملف YAML بسيط.
- **كل شيء تلقائي.** حجم الدفعة، واكتشاف وحدة GPU، والتكميم — كلها تُدار عنك.
- **يعمل محليًا.** درّب على وحدة GPU الخاصة بك باستخدام QLoRA. لا حاجة إلى السحابة.

## ما الجديد

**v0.76.0 — الإعدادات التي كان Soup يقبلها ولا يطبّقها أبدًا باتت توقف التحميل.** الخيارات التي كان يُتحقَّق
من صحتها وتُوثَّق ولا يقرؤها شيء في المسار الذي تزعم أنها تضبطه، صارت تُرفَض عند التحميل، مع تسمية
الإعداد وما ينبغي تغييره؛ أما البقية فتُصدر تحذيرًا وتسمّي الإصدار الذي سيرفضها. وقد يتوقف تحميل إعداد
كان يُحمَّل من قبل: شغّل `soup doctor --config soup.yaml` أولًا.

- **تغيير غير متوافق مع الإصدارات السابقة: يُرفَض عند التحميل أي إعداد لم يكن له أي أثر.** ومن ذلك
  `training.quantization_aware: true` (فلم يكن يشغّل تدريبًا واعيًا بالتكميم قط)، و`training.use_longlora`،
  و`training.ebft_variant`، و`training.gdpo_variant`، و`training.rope_scaling_type` حيث لا تقرؤه أي خطوة
  تهيئة، و`freeze_layers` / `freeze_ratio` خارج مسار `sft` / `tts` النصي على الواجهة الخلفية transformers،
  و`auto_mixed_precision` خارج `sft` / `tts`، و`backend: unsloth` في سبع مهام أخرى ليست لها خطوة تهيئة
  لـ unsloth. وتسمّي كل رسالة الحقل وما ينبغي تغييره. أما `lr_groups` و`early_stop_patience`
  و`forgetting_eval_steps` وغيرها من الإعدادات التي لا يقرؤها شيء فتُصدر تحذيرًا عند التحميل الآن، وستُرفَض
  بدءًا من v0.77.
- **صارت خسارة التحقق تُقيَّم الآن.** كان `data.val_split` يستبعد صفوفًا من التدريب في كل تشغيل، ولم يكن
  أي مدرّب على الواجهة الخلفية transformers يقيّمها. أما الآن فكل مهمة تحصل على قسم للتحقق تقيّمه في نهاية
  كل حقبة (epoch)، أو كل N خطوة عبر `training.eval_steps`، وتسجّل خسارة التحقق؛ وتستغرق التشغيلات
  أطول ببضعة في المئة. **تغيير غير متوافق مع الإصدارات السابقة بالنسبة إلى `grpo` و`ppo` و`online_dpo`:**
  فهي افتراضيًا لم تعد تستبعد صفوفًا (يُتجاهَل `data.val_split` مع ملاحظة)، ولذلك يُدرَّب النموذج على كل صف.
- **تغيير غير متوافق مع الإصدارات السابقة: كل `grpo_variant` غير قياسي يطبّق عقوبة KL الخاصة بـ `grpo_beta`.**
  كانت `gspo` و`dapo` و`dr_grpo` و`bnpo` و`two_sided` و`rft` تتحقق من `grpo_beta` ثم تُسقطه، ولذلك لن تعيد
  إعدادات المتغيّرات الموجودة إنتاج التشغيلات السابقة. وصار `grpo_beta: 0` يعمل بلا KL ويتخطى التمريرة
  الأمامية للنموذج المرجعي.
- **يقرأ بثّ الطبقات طبقته القرصية على خيط في الخلفية، عبر الإدخال/الإخراج المباشر (direct I/O)، ويمكنه توزيعها
  على عدة أقراص NVMe.** على مخزن NF4 بحجم 36 GB وبهيئة نموذج 70B أكبر من ذاكرة RAM (جهاز واحد بوحدة
  RTX 5070 Laptop سعة 8 GB)، انخفضت الخطوة الباردة من 110.5 s مع القراءة المتزامنة إلى 16.2-17.3 s، ومن
  17.4 s إلى 9.97 s عبر قرصين باستخدام `SOUP_LAYER_STREAM_STRIPE_DIRS` (مسبار تحت حمل متواصل: جولتان،
  1.28x و1.41x، بلا حكم رسمي؛ [السجل](benchmarks/probe-rtx5070-two-drive-sustained.md)). أما المخزن الذي
  يتسع في ذاكرة RAM فيظل أسرع على طبقة RAM، وهي ما يختاره `stream_source: auto`. وهو لا يُفعَّل إلا بطلب
  صريح (`stream_layers: true`) ولا يزال BETA؛ والسجلات:
  [الإدخال/الإخراج المباشر](benchmarks/gate-974-disk-tier-direct-io.md)،
  [التوزيع على الأقراص](benchmarks/gate-two-drive-striping.md).
- **أوامر وخيارات جديدة.** يفحص `soup adapters check` مهايئ LoRA بحثًا عن الإخفاق الصامت؛ ويدرّب
  `soup bench train` عددًا ثابتًا من الخطوات ويخرج برمز 1 إذا لم يكن النموذج يتدرّب (ويغطي `task: sft`
  على الواجهة الخلفية transformers؛ وصار `soup bench` مجموعة أوامر، و`soup bench infer` هو معيار
  الاستدلال القديم)؛ ويطهّر `soup data clean` مجموعة بيانات ويصلحها دون المساس بملف الإدخال؛
  ويتحقق `soup recipes verify` على جهاز meta، ودون تنزيل أي أوزان، من أن كل وصفة مشحونة تستطيع إرفاق
  مهايئ LoRA؛ ويسمّي `soup rewind` صفوف مجموعة البيانات التي تقف وراء قفزة الخسارة. ويكتسب `soup infer`
  الخيار الاختياري <span dir="ltr">`--batch-size`</span> والخيار التجريبي <span dir="ltr">`--cuda-graphs`</span>
  (يتطلب PyTorch 2.14 أو أحدث؛ على جهاز RTX 5070 Laptop واحد، دون تثبيت، مع Qwen2.5-1.5B بدقة FP16،
  كان أبطأ تشغيل مع هذا الخيار أسرع بمقدار 2.9x في فك الترميز من أسرع تشغيل بدونه). ويُجري `soup ab` اختبارًا
  ثنائي الجانب متينًا إزاء التباين، ويوقّع `soup adapters sign --backend sigstore`
  و`soup attest emit --sign sigstore` باستخدام Sigstore OIDC بلا مفاتيح
  (<span dir="ltr">`pip install "soup-cli[sigstore]"`</span>).
- **قيد الإيقاف التدريجي (Deprecated):** سيُزال `soup export --format awq` و<span dir="ltr">`--format gptq`</span>
  في v0.77؛ فكلا المشروعين الأصليين مؤرشف ولا يُثبَّت أي من الملحقين (extras) إلى جانب
  <span dir="ltr">`[train]`</span>.

> Python **3.10–3.12** فقط. وعلى الإصدار 3.13 وما بعده كان pip يختار حزم PyTorch (wheels) غير مختبرة تنهار في
> الامتداد الأصلي قبل أن يعمل Soup أصلًا.

تجد أبرز ما في الإصدارات الأقدم في صفحة [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

## البدء السريع

### 1. التثبيت

Soup تطبيق سطر أوامر، ولذلك فإن أنظف تثبيت هو أن تمنحه بيئته الخاصة وأن يضع `soup` على `PATH` لديك:

```bash
# النواة الخفيفة: CLI + الإعدادات + أدوات البيانات، بلا PyTorch
pipx install soup-cli
uv tool install soup-cli          # الفكرة نفسها، إن كنت تستخدم uv أصلًا

# أضف حزمة التدريب (torch وtransformers وpeft وtrl وdatasets وغيرها …)
pipx install "soup-cli[train]"

# كل شيء (train + serve + ui + data) دفعة واحدة
pipx install "soup-cli[all]"

# أو من GitHub (أحدث نسخة تطوير)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

هل أنت داخل بيئة virtualenv أو دفتر Colab أو صورة Docker أصلًا؟ استخدم `pip` مباشرةً بالأسماء والحزم
الإضافية نفسها:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

استخدم `pip` بدلًا من `pipx` إن كنت تريد أيضًا تنفيذ `import soup_cli` من شيفرتك الخاصة، لأن pipx يعزل
التطبيق عمدًا عن كل ما عداه.

يوجد جدول الحزم الإضافية الكامل (`fast` و`mlx` و`serve` و`eval` و`ui` و`vision` و`audio` و…) في
[`docs/models.md`](docs/models.md#optional-extras).

> **`error: externally-managed-environment`؟** هذه هي
> [PEP 668](https://peps.python.org/pep-0668/)، وليست مشكلة في Soup. فمنذ Debian 12 وUbuntu 23.04 وما
> بعدهما يمنع النظام `pip` من الكتابة في Python الخاصة بالنظام، لأن `apt` يدير هذه الملفات أيضًا. ويتجاوز
> `pipx` و`uv tool` ذلك بمنح Soup بيئته الخاصة، وهذا سبب ذكرهما أولًا أعلاه. كما يفي بالغرض تنفيذ
> `python3 -m venv .venv && source .venv/bin/activate` ثم استخدام `pip` العادي.

> **علامتا اقتباس مزدوجتان لا مفردتان.** الصيغة <span dir="ltr">`"soup-cli[train]"`</span> هي الوحيدة التي تعمل في كل الأصداف —
> `cmd.exe` وPowerShell وbash وzsh. وإن كنت قد نسخت <span dir="ltr">`'soup-cli[train]'`</span> من برنامج تعليمي أقدم ورفضه pip،
> فهذا هو السبب:
> [السبب ونص الخطأ كاملًا](docs/models.md#quoting-the-extra).

تعمل الأوامر `soup init` و<span dir="ltr">`soup data …`</span> وسائر أوامر البيانات والفحص على التثبيت الخفيف.
أما الضبط الدقيق (`soup train`) فيحتاج إلى الحزمة الإضافية `[train]`.

### 2. أنشئ ملف الإعدادات

```bash
soup init                       # معالج تفاعلي
soup init --template chat       # أو ابدأ من قالب
```

القوالب: `chat` و`code` و`tool-calling` و`medical` و`reasoning` و`vision` و`kto` و`orpo` و`simpo` و`ipo`
و`bco` و`rlhf` و`pretrain` و`moe` و`longcontext` و`embedding` و`audio`.

### 3. درّب، واختبر، وانشر

```bash
soup train --config soup.yaml                 # LoRA والتكميم وتقسيم الدفعات — كل ذلك يُدار عنك
soup chat  --model ./output                    # تحدَّث إلى نموذجك
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # ادمج LoRA في النموذج الأساسي
soup export --model ./output --format gguf --quant q4_k_m   # GGUF لـ Ollama / llama.cpp
```

وتوجد أهداف تصدير أخرى (ONNX وTensorRT وAWQ وGPTQ وBitNet) وخيارات النشر في
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## واجهة الويب

تفضّل المتصفح؟ يقدّم `soup ui` لوحة محلية للتجارب وإعداد التدريب والمقاييس الحية واستكشاف مجموعات
البيانات والدردشة مع النموذج.

```bash
pip install "soup-cli[ui]"
soup ui
# يفتح http://127.0.0.1:7860/?token=<token>
```

![واجهة ويب Soup — تدريب جديد](docs/assets/web-ui-new-training.png)

[توثيق واجهة الويب](docs/serving-and-export.md#web-ui)

## الإعدادات

ملف `soup.yaml` كامل:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # أسرع بمقدار 2-5 أضعاف، pip install "soup-cli[fast]"

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

`config/schema.py` هو المصدر الوحيد للحقيقة لكل حقل. أما خيارات البيانات والتدريب وPEFT المتقدمة
فموثَّقة تحت [التوثيق](#التوثيق).

> **مفاتيح الإعداد غير المعروفة تُرفَض منذ v0.75.** المفتاح الذي لا يعرّفه أي نموذج في المخطط (Pydantic) — كخطأ إملائي مثل
> `quantizaton`، أو حقل لا يوجد إلا في إصدار أحدث من Soup — كان يجتاز التحقق نظيفًا ثم يُهمَل، فيستمر
> التشغيل دون تطبيق الإعداد ببساطة. وكان v0.74 يبلّغ عنه عند التحميل مع الحقل الذي ربما قصدته؛ أما بدءًا
> من **v0.75** فيفشل تحميل الإعداد نفسه، فصحِّح المفتاح أو احذفه بدلًا من الاعتماد على إهماله. راجع
> [مفاتيح الإعداد غير المعروفة](docs/backends-and-ops.md#unknown-config-keys).

## التوثيق

المرجع الكامل للميزات موجود في [<span dir="ltr">`docs/`</span>](docs/). ابدأ من هنا:

<div dir="rtl">

| الدليل | ما يغطيه |
|---|---|
| [مهام التدريب وأساليبه](docs/training.md) | SFT وDPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO، واستدعاء الأدوات، وPRM، والتدريب المسبق، والتقطير، والتصنيف، والرؤية/الصوت/TTS، وإلغاء التعلّم، وRAFT/RA-DIT، وكواشف تحصين الحلقات التكرارية (loops) ضد الأعطال |
| [PEFT والسياق الطويل والكفاءة](docs/peft-and-efficiency.md) | DoRA و<span dir="ltr">LoRA+</span> وrsLoRA وVeRA وOLoRA وNEFTune وPiSSA وReLoRA، ومجموعة المُحسِّنات وطرائق PEFT، وLLaMA Pro وGaLore وYaRN/LongLoRA، والتعبئة (packing)، والمنهج التدريجي (curriculum)، والضبط التلقائي |
| [الأداء والتكميم](docs/performance-and-quantization.md) | QAT وFP8 وقائمة التكميم (I + II) وذاكرة KV المؤقتة وNVFP4 وصيغ الحفظ وCut Cross-Entropy ونقاط فحص التدرّج (gradient checkpointing) والنوى (kernels) ونقل التنشيطات (activation offloading) وبثّ الطبقات وتعدد GPU / DeepSpeed / FSDP |
| [هندسة البيانات](docs/data.md) | الصيغ، وخط أنابيب مكافئ لـ Axolotl/LF، وأدوات البيانات، والتوليد الاصطناعي وforge، وبطاقات تقييم الجودة، وأدوات التتبّع (trace)، ومجموعات البيانات البعيدة، والمزج، ورسوم الوصفات الموجَّهة (DAG) |
| [التقييم والمجسّات](docs/evaluation.md) | تصميم التقييم/بوابته، والتدريب المشروط ببوابة التقييم، والمقاييس المعيارية، ومقاييس NLG، والمعايرة، وساحة Elo، والتشخيص، ومجسّات X-ray بعد التدريب، وA/B، والانحراف (drift)، وقابلية الضبط، و`soup advise` |
| [الخدمة والتصدير](docs/serving-and-export.md) | خادم متوافق مع OpenAI، والاستدلال الدُفعي، وقياس الأداء، والدمج/التصدير، ونقطة نهاية Anthropic Messages، وفك الترميز التخميني (درّب مسودتك وقِسها)، والطيار الآلي للنشر، وواجهة الويب، وAgent Forge |
| [المهايئات (Adapters) والسجل والحوكمة](docs/adapters-and-governance.md) | دورة حياة المهايئات وإدارتها، وسجل النماذج، وSoup Cans، ودولاب البيانات (`soup loop`)، وتحرير المعرفة، والتوجيه (steering)، وضوابط سلسلة التوريد (scan/sign/BOM/attest/audit/airgap) |
| [البدء السريع للامتثال والحوكمة](docs/compliance.md) | قوالب `init` لـ HIPAA/SOC2/EU-AI-Act/SR-11-7، والمنشأ (BOM/attest/repro-receipt)، وسجل التدقيق، والعزل الشبكي (air-gap)، والتوليد التلقائي لبطاقة النموذج (`soup card`)، وبوابة CI (`soup ci init`) |
| [الواجهات الخلفية والمنصة والعمليات](docs/backends-and-ops.md) | واجهتا MLX/Unsloth الخلفيتان، والمراكز البديلة (hubs)، وتكامل HF Hub، والطيار الآلي، وتتبّع التجارب، وplan/apply، وملفات قفل البيئة، وملاءمة العتاد، وإكمالات الأوامر، والإضافات، وأوامر الأدوات المساعدة |
| [مرجع الأوامر](docs/commands.md) | قائمة أوامر `soup` الكاملة |
| [النماذج المدعومة والحزم الإضافية](docs/models.md) | عائلات النماذج الموصى بها، ودليل أحجام VRAM، ومصفوفة الحزم الإضافية في pip |

</div>

## صيغ البيانات

Alpaca وShareGPT وChatML وأزواج التفضيلات (DPO / ORPO / SimPO / IPO / KTO) والرؤية والصوت وASR والنص
العادي والتضمين (embedding) وRAFT وغيرها — كلها تُكتشف تلقائيًا من JSONL أو JSON أو CSV أو Parquet أو TXT،
ولذلك تكتفي في معظم الحالات بتوجيه `data.train` إلى ملف دون أن يتغير شيء آخر. وتوجد المخططات مع مثال
عملي لكل صيغة، إضافةً إلى خط أنابيب البيانات (معرّفات URI البعيدة، والبثّ، والتقسيم إلى شظايا (sharding)، والدمج المتناوب (interleaving)، وتوسيع
المفردات، واستيعاب المستندات)، في [`docs/data.md`](docs/data.md#data-formats).

## الأوامر الشائعة

```bash
soup train  --config soup.yaml        # درّب (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl --output results.jsonl   # استدلال دُفعي
soup chat   --model ./output          # دردشة تفاعلية
soup serve  --model ./output          # خادم API متوافق مع OpenAI
soup ui                               # لوحة متصفح محلية
soup merge  --adapter ./output        # ادمج LoRA في النموذج الأساسي
soup export --model ./output --format gguf           # صدِّر للنشر
soup eval   benchmark --model ./output               # قيِّم
soup data   inspect ./data/train.jsonl               # إحصاءات مجموعة البيانات
soup recipes list                     # أكثر من 100 وصفة نموذج جاهزة
soup autopilot --model <id> --data d.jsonl --goal chat  # بلا إعدادات
soup doctor                           # افحص GPU / الاعتماديات / البيئة
```

قائمة الأوامر الكاملة موجودة في [`docs/commands.md`](docs/commands.md).

## النماذج المدعومة

يعمل Soup مع **أي** نموذج لتوليد النصوص على
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) — فإن كان يُحمَّل عبر
`AutoModelForCausalLM` فهو يعمل دون أي تغيير في الإعدادات. وتأتي Llama 3.x/4 وQwen 2.5/3 وGemma 3
وMistral وMixtral وDeepSeek R1/V3 وPhi-4 وأكثر من 100 نموذج آخر كوصفات جاهزة (`soup recipes list`).

<div dir="rtl">

| VRAM | أكبر نموذج (QLoRA بدقة <span dir="ltr">4-bit</span>) | مثال |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | <span dir="ltr">70B+</span> (كامل) أو MoE | Mixtral-8x22B, DeepSeek-V3 |

</div>

جداول النماذج الكاملة ونماذج الرؤية ومصفوفة الحزم الإضافية موجودة في [`docs/models.md`](docs/models.md).

## Docker

شغّل Soup دون تثبيت CUDA أو PyTorch محليًا (تُنشر الصورة على GHCR مع كل إصدار):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # أو ابنِ الصورة محليًا
```

## المتطلبات

- Python 3.10 أو 3.11 أو 3.12 (هذه هي الإصدارات التي يختبرها CI؛ أما 3.13 وما بعده فغير مدعوم بعد لأن
  حزمة PyTorch لم يُتحقَّق منها هناك)
- GPU يدعم CUDA (موصى به) أو Apple Silicon (MPS) أو CPU (تجريبي — بطيء جدًا)
- 8 GB فأكثر من VRAM للنماذج بحجم 7B مع QLoRA

تعمل جميع مهام التدريب على CPU لأغراض الاختبار (ويُعطَّل التكميم تلقائيًا). والحزم الإضافية الاختيارية
(`train` و`all` و`fast` و`vision` و`qat` و`serve` و`serve-fast` و`ui` و`eval` و`deepspeed` و`liger` و`mlx`
و`onnx` و`tensorrt` و…) مذكورة في [`docs/models.md`](docs/models.md#optional-extras).

## استكشاف الأخطاء وإصلاحها

```bash
soup doctor    # GPU وموارد النظام والاعتماديات والإصدار في مكان واحد
```

حزم CUDA الجاهزة (wheels) وعدم تطابق الإصدارات: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## التطوير

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # فحص الشيفرة (lint)
pytest tests/ -v                   # اختبارات الوحدة (سريعة، بلا GPU)
pytest tests/ -m smoke -v          # اختبارات الدخان (تنزّل نموذجًا صغيرًا وتدرّبه)
pytest tests/ -m gpu --no-cov -v   # اختبارات GPU (تحتاج بطاقة CUDA؛ بلّغ عن النتائج، راجع CONTRIBUTING.md)

pre-commit install                 # اختياري: ruff lint+format عند كل commit
```

راجع [CONTRIBUTING.md](CONTRIBUTING.md) لمعرفة سير العمل الكامل، و[SECURITY.md](SECURITY.md) للإبلاغ عن
ثغرة أمنية. والقياس عن بُعد (Telemetry) لا يعمل إلا بموافقتك الصريحة (`SOUP_TELEMETRY=1`، وهو معطَّل افتراضيًا؛ راجع
[سياسة الخصوصية](docs/backends-and-ops.md#privacy-policy)).

## ادعم Soup

Soup مرخَّص بـ Apache-2.0 ومجاني — وسيبقى كذلك. يُبنى ويُصان علنًا على حاسوب محمول واحد بسعة 4 GB،
ولهذا فإن كل رقم أداء في هذا التوثيق مقيس لا مدَّعى.

إن وفّر عليك Soup جولة تدريب، فإن [وضع نجمة على المستودع](https://github.com/MakazhanAlpamys/Soup)
هو أكثر ما يفيد، ولا يكلّفك شيئًا. وإن أردت تمويل العمل مباشرةً:

**[❤️ تبرَّع](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — مرة واحدة وبأي مبلغ (استخدم خيار
*Change amount* / *تغيير المبلغ* في صفحة الدفع). تُعالَج المدفوعات عبر Stripe باسم النشاط التجاري
المسجَّل للمشرف، **MePlay, Inc.** — وهذا الاسم، لا «Soup»، هو الذي يظهر في صفحة الدفع وفي كشف حساب بطاقتك.

تشتري التبرعات وقت GPU للأعمال المقيَّدة بالعتاد — تعدد وحدات GPU، والتحقق على نماذج 8B فما فوق،
وApple Silicon — التي لا يستطيع حاسوب محمول واحد بسعة 4 GB بلوغها.

والطريقة الأخرى لتحريك هذه البنود نفسها هي **العتاد ذاته**. فهي تُطرح خلف بوابات صادقة من نوع
«يتطلب \<العتاد\>» بدلًا من ادعاءات غير مُتحقَّق منها؛ ولذلك إن كان لديك وصول إلى جهاز أكبر — أو رصيد GPU
غير مستخدم — فإن تشغيل إحدى مسائل
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
ونشر الأرقام يفيد بقدر ما يفيد تمويل وقت GPU. وتذكر تلك المسائل بالضبط ما هو معطَّل بسبب العتاد اليوم.

## المساهمون

بناه المجتمع ❤️ — شكرًا لكل من ساهم. راجع [CONTRIBUTORS.md](CONTRIBUTORS.md).

[![المساهمون](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## التواصل

مكان الأخطاء وطلبات الميزات هو [متتبّع المسائل](https://github.com/MakazhanAlpamys/Soup/issues)،
ومكان الأسئلة هو [Discussions](https://github.com/MakazhanAlpamys/Soup/discussions) — وكلاهما يُجاب عنه
أسرع ويساعد الشخص التالي الذي يواجه المشكلة نفسها.

للدردشة المباشرة والمساعدة في الإعداد وكل ما يُفضَّل أن يكون حوارًا مباشرًا، انضم إلى
[Discord](https://discord.gg/dgd2pJcjwP) أو إلى [مجتمع Telegram](https://t.me/souptasters).
أما كل ما ينبغي أن يظل قابلًا للعثور عليه بعد ستة أشهر فمكانه Issues أو Discussions — فالإجابة في Discord
تفيد شخصًا واحدًا، بينما يفيد البلاغ (Issue) كل من يصادف الأمر نفسه.
وتنطبق [مدونة قواعد السلوك](CODE_OF_CONDUCT.md) هناك أيضًا.

أما لكل ما لا يصلح للعلن — بلاغات الأمان (راجع [SECURITY.md](SECURITY.md)) أو مسائل مدونة قواعد السلوك
أو الصحافة — فراسل **team@trysoup.dev**. فهذا هو عنوان المشروع والعنوان الصحيح لكل ما يتعلق بـ Soup.
أما **makazanalpamys@gmail.com** فهو العنوان الشخصي للمشرف؛ وهو يصل إلى الشخص نفسه، ويصلح بديلًا احتياطيًا عند الحاجة.

## الاستشهاد بـ Soup

يوصَف بثّ الطبقات — أي تدريب نموذج بحجم 8B على وحدة معالجة رسومية لحاسوب محمول بسعة 4 GB عبر بثّ
النموذج الأساسي المجمَّد من ذاكرة RAM المضيفة طبقةَ فك ترميز واحدة في كل مرة — في ورقة ما قبل النشر
(preprint)، مع بروتوكول صحة النتائج الذي يتحقق من تشغيلٍ مبثوث مقارنةً بتشغيل مقيم بالكامل في الذاكرة
(ويُذكر التمرير الأمامي والخلفي كلٌّ على حدة، لأنهما ادعاءان اثنان لا ادعاء واحد).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**الإصدار 3 (13 أغسطس 2026) هو الحالي.** لم يتغير العنوان ولا الادعاء — 8B على 4 GB — ولم يتغير أي رقم
مقيس منذ v1. وما يفعله v3 هو **سحب تفسير كنا قد نشرناه**، وهذا أيضًا أقصر وصف لغاية الورقة:

- **سُحب في v3: «بثّ الطبقات مقيَّد بالنقل من المضيف إلى الجهاز، لا بوحدة GPU».**
  كان ذلك *استنتاجًا* من إعادة الإنتاج على H100 المذكورة أدناه، ولم يُقَس قط. وقد قسناه في 11 أغسطس
  فتبيّن أنه خاطئ عند الإعداد المنشور: فحذف كل بايت منقول من المضيف إلى الجهاز لا يوفّر سوى **1.4%**،
  ويظل مسار الحساب (compute stream) ينتظر انتهاء عملية نسخ خلال **0.20%** من زمن الخطوة، وتعمل الخطوة عند **71.3%** من سقف GEMM لتلك
  البطاقة في الجلسة نفسها. وأكبر تكلفة خاصة بالبثّ هي إزالة تكميم NF4 لكل طبقة، بنسبة 9.8%
  ([السجل](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). وتبقى كل القياسات قائمة؛ أما إعادة
  الإنتاج فتصمد بصورة أضعف — فالقيد مشترك بين الجهازين وليس قدرة GPU الحسابية.
- **إعادة إنتاج على عتاد لا يشبه الأصلي في شيء** (أُضيفت في v2): 119.6 tok/s على RTX 3050 (بتردد 952 MHz) مقابل وسيط
  113.00 على H100، عند الذروة نفسها 3.32 GB. وكلاهما سابق لإصلاح #331، وهما الرقمان
  اللذان توردهما الورقة. أما صفّ ما بعد #331 (@umran666، #1103) فهو 208.6 tok/s بتردد 1935–1957 MHz وبذروة 2.40 GB،
  أي 59% مقابل 68% من سقف GEMM في الجلسة نفسها: فالكسب الخام البالغ 1.74x مردّه تردد الساعة لا تسريع فعلي، وقد تغيّر
  نمط استخدام الذاكرة بين الصفّين، فحتى هذه النسبة ليست مقارنة متكافئة
  ([السجل](benchmarks/gate-v0.72.2-nf4.md)).
- **عيب صامت يُنتج تدرّجات خاطئة، اكتُشف وأُصلح.** على NF4 فوق نحو 165 MiB لكل طبقة بقي التمرير الأمامي
  مطابقًا بتًّا ببتّ وبدا منحنى الخسارة سليمًا بينما كانت التدرّجات خاطئة. وقد حُدِّد السبب في المكتبة
  المصدرية (upstream) وأُبلغ عنه هناك؛ والإصلاح مشروط بضوابط على نماذج حقيقية بحجم 32B و72B.
- **مطابقة بتًّا ببتّ عند أحجام نماذج حقيقية** بدلًا من نماذج تجريبية مبسّطة من ثلاث طبقات: التمرير الأمامي من
  0.5B إلى 72B، والخلفي عند 8B و14B.
- **جودة النموذج المدرَّب، مقيسةً للمرة الأولى**، ولا يمكن تمييزها عن تشغيل مقيم بالكامل في الذاكرة.
- **مقارنة مع DeepSpeed** — بما في ذلك النتيجة التي لا تصبّ في صالحنا: ثماني بطاقات بـ ZeRO-3 أبطأ من
  بطاقة واحدة تدرّب تشغيلًا مقيمًا.
- **أُعيدت كتابة قسم القيود**: من بنود v1 العشرة أُغلق بند واحد وضاق نطاق أربعة أخرى، وأُضيفت سبعة بنود
  جديدة.

استشهد بالإصدار الذي استخدمته. إن <span dir="ltr">`10.5281/zenodo.21771064`</span> هو معرّف DOI الشامل (Concept DOI) ويحيل دائمًا إلى أحدث
إصدار (v3 اليوم)؛ ويبقى v1 وv2 قابلين للاستشهاد بمعرّفَي DOI الخاصَّين بكل منهما ولا يُعدَّلان — فالسحب
المذكور أعلاه إصدار جديد بالتحديد كي يبقى سجل ما ادّعيناه ومتى ادّعيناه سليمًا.

وسجلات القياس التي تقف خلف كل رقم فيها موجودة في [<span dir="ltr">`benchmarks/`</span>](benchmarks/)، منشورةً كما كُتبت — بما في
ذلك الإخفاقات والافتراضات التي تبيّن خطؤها والأرقام التي قِيست ثم استُبعدت.

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

## الترخيص

[Apache-2.0](LICENSE). حقوق النشر © مساهمو Soup.
