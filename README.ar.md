<<<<<<< Updated upstream
<p align="center">**🌍 [Türkçe](README.md) | [English](README.en.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)**
=======
<p align="center"><strong>🌍 <a href="README.md">Türkçe</a> | <a href="README.en.md">English</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.ru.md">Русский</a> | <a href="README.es.md">Español</a></strong></p>

<!-- synced-from: README.md sha256:73b53c056cd7db0f63c60a2e04e194606a6aa34cbae9b69ef29b732d85b336d3 -->
>>>>>>> Stashed changes

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
<<<<<<< Updated upstream
  <strong>ضبط نماذج اللغة الكبيرة والتدريب اللاحق بأمر واحد. لا SSH، لا تعقيدات تكوين.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">الموقع الإلكتروني</a> &middot;
  <a href="#البداية-السريعة">البداية السريعة</a> &middot;
  <a href="#واجهة-الويب">واجهة الويب</a> &middot;
  <a href="#التكوين">التكوين</a> &middot;
  <a href="docs/commands.md">الأوامر</a> &middot;
  <a href="docs/models.md">النماذج</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="التحميلات"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="رخصة Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="الاختبارات"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
=======
  <strong>ضبط النماذج اللغوية الكبيرة بأمر واحد. لا SSH، ولا黄河الإعدادات.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">الموقع</a> &middot;
  <a href="#quick-start">البدء</a> &middot;
  <a href="#web-ui">الواجهة</a> &middot;
  <a href="#configuration">الإعدادات</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a>
>>>>>>> Stashed changes
</p>

---

<<<<<<< Updated upstream
يحول Soup عذاب ضبط نماذج اللغة الكبيرة إلى سير عمل بسيط. تكوين واحد، أمر واحد، انتهى.

```bash
pip install "soup-cli[train]"   # إضافة [train] للضبط الدقيق؛ soup-cli وحده هو CLI الخفيف
=======
Soup يحول صعوبة ضبط النماذج اللغوية إلى سير عمل بسيط.

```bash
pip install "soup-cli[train]"
>>>>>>> Stashed changes
soup init --template chat
soup train
```

<<<<<<< Updated upstream
**ضبط دقيق لنموذج 8B على GPU محمول 4 GB.** تدفق الطبقات يحافظ على القاعدة المجمدة خارج VRAM ويغذي GPU بطبقة واحدة من فك التشفير في كل مرة. مقاس على RTX 3050 Laptop 4 GB: Llama-3.1-8B-Instruct + NF4 بـ **119.6 tok/s، ذروة 3.32 GB** — متطابق بت مع تشغيل عادي مقيم، وتم إعادة إنتاجه بشكل مستقل على H100 بـ 113.00 tok/s في نفس 3.32 GB. اختياري (`stream_layers: true`) ولا يزال BETA — [كيف يعمل](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) · [جميع القياسات](benchmarks/) · [الورقة البحثية](https://doi.org/10.5281/zenodo.21771064) · **[تحقق بنفسك على Colab T4 المجاني](notebooks/proof-4gb.ipynb)**

---

## 🎯 لماذا Soup؟

تدريب نماذج اللغة الكبيرة لا يزال مؤلماً. حتى الفرق ذات الخبرة تقضي 30-50% من وقتها في محاربة البنية التحتية بدلاً من تحسين النماذج. Soup يحل هذا.

- **صفر SSH.** لا تدخل SSH أبداً إلى صندوق GPU مكسور مرة أخرى.
- **تكوين واحد.** ملف YAML بسيط هو كل ما تحتاجه.
- **كل شيء تلقائي.** حجم الدفعة، كشف GPU، التكميم — تم التعامل معه.
- **يعمل محلياً.** تدريب على GPU الخاص بك مع QLoRA. لا حاجة للسحابة.

### الجديد — v0.74.0

**v0.74.0 — القاعدة المجمدة كانت تُحمل في fp32 طوال الوقت.** إصلاح هذا وحده يقلل ذروة VRAM بمقدار 2.59x على تكوين غير متغير. **116 من 120 طلب سحب مدمج في هذا الإصدار جاءت من خارج المصون**, من 25 شخصاً.

- كل تحميل SFT رفع القاعدة المجمدة صامتاً إلى fp32. مقاس على H100 مع Llama-3.1-8B + LoRA: **48,241 MiB → 18,658 MiB ذروة — 2.59x، 28.9 GB**.
- **Transformers 5.x، TRL 0.29، PEFT 0.20.**
- أربعة تجاوزات SSRF من نفس الشكل (IPv4 مختصر، عشري، سادس عشر، ثماني) تم إصلاحها.
- **`soup serve` الآن يخرج بـ 2** عندما يرتبط بمضيف غير loopback بدون `--tool-auth-token`.
- **`soup train --cloud lambda`** — خطة فقط افتراضياً، `--cloud-submit` يرسل مباشرة.

---

## 🏗️ الهيكل المعماري والوحدات

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70+ أوامر CLI (train, eval, data, serve, agent, …)
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — مصدر واحد للحقيقة لجميع حقول التكوين
│   ├── autodistill/     # التقطير التلقائي (التقاط البيانات، النشر، عامل MLX)
│   ├── autopilot/       # اختيار المهمة/التكميم/LR بدون تكوين
│   ├── cans/            # Soup Cans: تغليف النماذج، النشر، التحقق
│   ├── cloud/           # متحكمات سحابة Lambda Labs / Modal
│   └── utils/           # مساعدات مشتركة (350+ وحدة)
├── graft/               # تكوين Graft MCP (الأسلاك على مستوى المستخدم)
├── docs/                # أدلة المواضيع + مرجع الأوامر الكامل
├── templates/           # قوالب البداية chat.yaml، code.yaml، medical.yaml
├── benchmarks/          # سجلات القياس (بيانات خام لكل رقم)
├── tests/               # 460 ملف اختبار
└── pyproject.toml       # التبعيات، الإضافات، نظام البناء
```

**التبعيات الأساسية (تثبيت خفيف — بدون PyTorch):**
`typer`, `rich`, `pydantic>=2`, `pyyaml`, `huggingface-hub`, `plotext`, `packaging`

**إضافات `[train]`:** `torch>=2.6.0`, `transformers>=5.16.1,<6`, `peft>=0.20,<1`, `trl>=0.29`, `datasets`, `bitsandbytes`, `accelerate`

**ملاحظة:** يحتوي المستودع على كود مصدر Python إلى جانب نصوص الإعداد، سير عمل CI، وأدوات التغليف؛ لا ندعي "%100 Python خالص".

---

## 🚀 التثبيت

### المتطلبات

- Python 3.10، 3.11 أو 3.12 (هذه الإصدارات التي يختبرها CI؛ 3.13+ غير مدعوم بعد)
- GPU مع CUDA (مُوصى)، Apple Silicon (MPS)، أو CPU (تجريبي — بطيء جداً)
- 8 GB+ VRAM لنماذج 7B مع QLoRA

### خطوات التثبيت

```bash
# النواة الخفيفة: CLI + التكوين + أدوات البيانات، بدون PyTorch
pipx install soup-cli
uv tool install soup-cli          # نفس الفكرة، إذا كنت تستخدم uv بالفعل

# إضافة مكدس التدريب (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# كل شيء (train + serve + ui + data) دفعة واحدة
pipx install "soup-cli[all]"

# أو من GitHub (أحدث تطوير)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

بالفعل داخل virtualenv، دفتر ملاحظات Colab، أو صورة Docker؟ استخدم `pip` مباشرة:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **`error: externally-managed-environment`؟** هذا [PEP 668](https://peps.python.org/pep-0668/)، ليس مشكلة Soup. استخدم `pipx` أو `uv tool`.

> **علامات اقتباس مزدوجة، ليس مفردة.** `"soup-cli[train]"` هو الهجاء الوحيد الذي يعمل في كل shell.

جدول الإضافات الكامل (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, `liger`, `onnx`, `tensorrt`, …): [`docs/models.md`](docs/models.md#optional-extras)

### التثبيت من الشوكة

هذا المستودع هو شوكة `Ercaner1988/Soup`. المنبع: `MakazhanAlpamys/Soup`.

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

---

## 📖 الاستخدام

### البداية السريعة

```bash
# 1. إنشاء تكوين
soup init                       # معالج تفاعلي
soup init --template chat       # أو ابدأ من قالب

# 2. تدريب، اختبار، إرسال
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo user/my-model

# 3. دمج وتصدير
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**القوالب:** `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`

### الأوامر الشائعة

```bash
soup train  --config soup.yaml                         # تدريب (SFT/DPO/GRPO/PPO/KTO/ORPO/…)
soup infer  --model ./output --input prompts.jsonl    # استنتاج دفعي
soup chat   --model ./output                           # دردشة تفاعلية
soup serve  --model ./output                           # خادم API متوافق مع OpenAI
soup ui                                                # لوحة متصفح محلية
soup merge  --adapter ./output                         # دمج LoRA في النموذج الأساسي
soup export --model ./output --format gguf             # تصدير للنشر
soup eval   benchmark --model ./output                 # تقييم
soup data   inspect ./data/train.jsonl                 # إحصائيات مجموعة البيانات
soup recipes list                                      # 100+ وصفة نموذج جاهزة
soup autopilot --model <id> --data d.jsonl --goal chat # بدون تكوين
soup doctor                                            # فحص GPU / التبعيات / البيئة
```

### التكوين

`soup.yaml` كامل:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x أسرع، pip install "soup-cli[fast]"

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

`config/schema.py` هو المصدر الوحيد للحقيقة لكل حقل.

> **مفاتيح التكوين المجهولة تحذر اليوم وسترفض في v0.75.**

### تنسيقات البيانات

Alpaca، ShareGPT، ChatML، أزواج التفضيل (DPO / ORPO / SimPO / IPO / KTO)، رؤية، صوت، ASR، نص عادي، تضمين، RAFT والمزيد — كشف تلقائي من JSONL، JSON، CSV، Parquet أو TXT. انظر [`docs/data.md`](docs/data.md).

### النماذج المدعومة

Soup يعمل مع **أي** نموذج توليد نص على [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). Llama 3.x/4، Qwen 2.5/3، Gemma 3، Mistral، Mixtral، DeepSeek R1/V3، Phi-4 و100+ آخرون يأتون كوصفات جاهزة (`soup recipes list`).

| VRAM | أقصى نموذج (QLoRA 4-bit) | مثال |
|------|-------------------------|-------|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (كامل) أو MoE | Mixtral-8x22B, DeepSeek-V3 |

### واجهة الويب

```bash
pip install "soup-cli[ui]"
soup ui
# يفتح http://127.0.0.1:7860
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ الاختبار وبوابات الجودة

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — يجب أن يكون نظيفاً قبل أي commit
pytest tests/ -v --tb=short        # اختبارات الوحدة (سريع، بدون GPU؛ 460 ملف اختبار)
pytest tests/ -m smoke -v          # اختبارات الدخان (يحمل نموذج صغير، يدرب)
pre-commit install                 # اختياري: ruff lint+تنسيق على commit
```

**مصفوفة CI:** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**الأمان:** `soup doctor` يفحص GPU، موارد النظام، التبعيات، والإصدار في مكان واحد. التيلييمتري اختياري بصرامة (`SOUP_TELEMETRY=1`، افتراضي مغلق).

---

## 📚 التوثيق

| الدليل | يغطي |
|-------|--------|
| [مهام ووسائل التدريب](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO، استدعاء الأدوات، PRM، التدريب المسبق، التقطير |
| [PEFT والكفاءة](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA |
| [الأداء والتكميم](docs/performance-and-quantization.md) | QAT, FP8, KV-cache، تدفق الطبقات، متعدد GPU / DeepSpeed / FSDP |
| [هندسة البيانات](docs/data.md) | التنسيقات، أدوات البيانات، التوليد الاصطناعي، DAGs الوصفة |
| [التقييم والمجسات](docs/evaluation.md) | تصميم التقييم، المعايير، مقاييس NLG، A/B |
| [الخدمة والتصدير](docs/serving-and-export.md) | خادم متوافق مع OpenAI، الدمج/التصدير، واجهة الويب |
| [المحولات والحوكمة](docs/adapters-and-governance.md) | دورة حياة المحول، سجل النماذج، Soup Cans |
| [الامتثال](docs/compliance.md) | قوالب HIPAA/SOC2/EU-AI-Act/SR-11-7، سجل التدقيق |
| [الخلفيات والعمليات](docs/backends-and-ops.md) | خلفيات MLX/Unsloth، HF Hub، الطيار الآلي |
| [مرجع الأوامر](docs/commands.md) | قائمة أوامر `soup` الكاملة |
| [النماذج والإضافات](docs/models.md) | عائلات النماذج، دليل VRAM، مصفوفة إضافات pip |

---

## 👥 المساهمون

المستودع المنبع (`MakazhanAlpamys/Soup`) مبني من قبل المجتمع. هذه الشوكة (`Ercaner1988/Soup`) مخصصة لتكامل Graft MCP.

| المساهم | Commits | الدور |
|---------|---------|-------|
| Alpamys (Alpamys Makazhan) | 912 | مالك المشروع والمطور الرئيسي |
| AmixDigital | 37 | قسم الباب الأمامي لواجهة الويب |
| Srinivasan R | 24 | قناع الاستجابة MLX، خسائر التحقق، حارس معرف الوصفة |
| Amir Fathi | 24 | مساهمات |
| Ben Younes | 15 | مساهمات |
| Salil Mhatre | 11 | مساهمات |
| Darsh | 9 | مساهمات |
| Nurkhan Esenbek | 9 | مساهمات |
| Harshit Sharma | 7 | مساهمات |
| UmranPros | 6 | مساهمات |
| (24 مساهم آخر) | 5–1 | انظر [CONTRIBUTORS.md](CONTRIBUTORS.md) |
| **Ercan Er** | **2** | **تكوين Graft MCP (الأسلاك على مستوى المستخدم)** |

> المقاييس من مخرجات `git shortlog -sn HEAD` (فرع ercan1، 2026-09-09).

---

## 📄 الرخصة

[Apache-2.0](LICENSE). حقوق الطبع والنشر © مساهمي Soup.

هذه الشوكة (`Ercaner1988/Soup`) مبنية على المشروع الأصلي: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

### الاستشهاد

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
``````
=======
## لماذا Soup؟

- **لا SSH**
- **إعداد واحد**
- **كل شيء تلقائي**
- **يعمل محلياً**

## البدء السريع

```bash
pip install "soup-cli[train]"
soup init --template chat
soup train
```

## Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup-cli:latest
```

## المتطلبات

- Python 3.10-3.12
- GPU من NVIDIA أو Apple Silicon

## الرخصة

Apache-2.0
>>>>>>> Stashed changes
