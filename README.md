<p align="center"><strong>🌍 <a href="README.md">Türkçe</a> | <a href="README.en.md">English</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.ru.md">Русский</a> | <a href="README.es.md">Español</a></strong></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>LLM ince ayarını ve son eğitimini tek komutla yapın. SSH yok, yapılandırma karmaşası yok.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Web Sitesi</a> &middot;
  <a href="#hızlı-başlangıç">Hızlı Başlangıç</a> &middot;
  <a href="#web-arayüzü">Web Arayüzü</a> &middot;
  <a href="#yapılandırma">Yapılandırma</a> &middot;
  <a href="docs/commands.md">Komutlar</a> &middot;
  <a href="docs/models.md">Modeller</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="İndirmeler"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/lisans-Apache--2.0-blue" alt="Apache-2.0 Lisansı">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Testler"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Web sitesi"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-katıl-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&theme=light" alt="Soup CLI - 4 GB laptop GPU'sunda 8B LLM ince ayarı | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&utm_medium=badge&utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup, LLM ince ayarının zahmetini basit bir iş akışına dönüştürür. Bir yapılandırma, bir komut, bitti.

```bash
pip install "soup-cli[train]"   # [train] ince ayar için; yalnız soup-cli hafif CLI'dir
soup init --template chat
soup train
```

**4 GB laptop GPU'sunda 8B modeli ince ayar yapın.** Katman akışı (layer streaming), donmuş tabanı VRAM dışında tutar ve GPU'ya bir kod çözücü katmanı aynı anda besler. RTX 3050 Laptop 4 GB'de ölçülen: Llama-3.1-8B-Instruct + NF4, **119.6 tok/s, 3.32 GB tepe** — normal bir yerleşik çalışmaya karşı bit-özdeş, aynı 3.32 GB'de H100'de bağımsız olarak 113.00 tok/s ile doğrulandı. (tok/s sayısı, −4.8% maliyetine neden olan v0.73.0 doğruluk onarımından önce v0.72.2'de ölçülmüştür; 4 GB kart üzerinde yeniden çalıştırılmamıştır.) Katılım (`stream_layers: true`) isteğe bağlı ve hâlâ BETA —
[nasıl çalışır](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[tüm ölçümler](benchmarks/) · [makale](https://doi.org/10.5281/zenodo.21771064) ·
**[ücretsiz Colab T4'te kendiniz kontrol edin](notebooks/proof-4gb.ipynb)** (işlemi 4 GB ile sınırlar, ardından akışan modelin normal bir modelle bit-özdeş olduğunu doğrular)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="soup train pre-flight for Llama-3.1-8B on a 4 GB card: a 3.60 GB base store pinned in RAM across 32 layers and two 113 MB VRAM buffers, then a measured peak of 3.32 GB at 119.6 tok/s, stopping short of the 4 GB line"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, batch 1, seq 512, RTX 3050 Laptop 4 GB — <b>3.32 GB tepe, 119.6 tok/s</b>. <a href="https://youtu.be/T1LCErE943E">Tam video (90s)</a></sub>
</p>

---

## 🎯 Neden Soup?

LLM eğitimi hâlâ zahmetli. Deneyimli ekipler bile zamanlarının %30–50'sini modelleri iyileştirmek yerine altyapıyla boğuşarak geçirir. Soup bunu çözer.

- **Sıfır SSH.** Bozuk bir GPU kutusuna asla SSH açmayın.
- **Tek yapılandırma.** Basit bir YAML dosyası yeterli.
- **Her şey otomatik.** Toplu iş boyutu, GPU algılama, nicemleme — halloldu.
- **Yerel çalışır.** QLoRA ile kendi GPU'nuzda eğitin. Bulut gerekmez.

### Yeni Neler Var — v0.74.0

**v0.74.0 — donmuş taban baştan beri fp32'de yükleniyordu.** Bunu düzeltmek tek başına değiştirilmemiş bir yapılandırmada tepe VRAM'ı 2.59 kat azaltır. **Bu sürümdeki 120 birleştirilen çekme isteğinin 116'sı dışarıdan, 25 kişiden geldi.**

- Her SFT yüklemesi donmuş tabanı sessizce fp32'ye yükseltti. H100'de Llama-3.1-8B + LoRA ile ölçülen: **48.241 MiB → 18.658 MiB tepe — 2.59x, 28.9 GB**.
- **Transformers 5.x, TRL 0.29, PEFT 0.20.**
- Dört aynı biçimde SSRF atlatma (kısaltılmış, ondalık, onaltılık, sekizli IPv4) kapatıldı.
- **`soup serve` artık exit 2 döndürür** `--tool-auth-token` olmadan loopback dışı bir ana bilgisayara bağlandığında.
- **`soup train --cloud lambda`** — varsayılan olarak yalnızca plan, `--cloud-submit` ile canlı gönderim.

> **Bilinen sınırlama:** Bildirilen `torch>=2.5.0` tabanı `trl>=0.29` ile çalışmıyor — torch 2.5.1'de trl içe aktarılamaz. Temiz kurulum daha yeni bir torch çözer ve etkilenmez; sabitlenmiş 2.5.x ortamı değildir. Bkz. [#651](https://github.com/MakazhanAlpamys/Soup/issues/651).

> Python **3.10–3.12** yalnızca. 3.13+ üzerinde pip, Soup çalışmadan önce çöken test edilmemiş PyTorch tekerleklerini çözerdi.

<details>
<summary>Önceki sürüm — v0.73.3, her çekme isteği bakımcının dışından geldi</summary>

**v0.73.3 — bu sürümdeki her çekme isteği bakımcıdan başka birinden geldi.** Tamamı 8 kişiden, beşi ilk kez görünüyor. Buldukları ilginç: doğrulanan, belgelenen ve sonra hiçbir şey tarafından okunmayan dört ayrı bayrak.
- **Sıfır belirteçte eğitilmiş yalnızca asistan masking, normal bir kayıp eğrisiyle.** `BatchEncoding` döndüren bir belirteçleyici — bu bir `dict` değil — korumayı aştı, böylece etiket maskesi eşlemenin **anahtar dizelerinden** inşa edildi. İstisna yok, uyarı yok, eğitim gibi görünen bir kayıp eğrisi. Hata ile değil, türü okuyarak bulundu.
- **Apple Silicon'da `quantization: 4bit` sessizce `none` olarak yeniden yazıldı.**
</details>

---

## 🏗️ Mimari ve Modüller

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70+ CLI komutu (train, eval, data, serve, agent, …)
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — tüm yapılandırma alanları için tek kaynak
│   ├── autodistill/     # Otomatik damıtma (veri yakalama, yayımlama, MLX çalışanı)
│   ├── autopilot/       # Sıfır-yapılandırma görev/nicemleme/LR seçimi
│   ├── cans/            # Soup Cans: model paketleme, yayımlama, doğrulama
│   ├── cloud/           # Lambda Labs / Modal bulut denetleyicileri
│   └── utils/           # Paylaşılan yardımcılar (350+ modül)
├── graft/               # Graft MCP yapılandırması (kullanıcı düzeyi kablo bağlantısı)
├── docs/                # Konu başına rehberler + tam komut referansı
├── templates/           # chat.yaml, code.yaml, medical.yaml başlangıç şablonları
├── benchmarks/          # Ölçüm kayıtları (her sayı için ham veri)
├── tests/               # 460 test dosyası
└── pyproject.toml       # Bağımlılıklar, extras, yapı sistemi
```

**Temel bağımlılıklar (çekirdek kurulum — PyTorch yok):**
`typer`, `rich`, `pydantic>=2`, `pyyaml`, `huggingface-hub`, `plotext`, `packaging`

**`[train]` ekstrası:** `torch>=2.6.0`, `transformers>=5.16.1,<6`, `peft>=0.20,<1`, `trl>=0.29`, `datasets`, `bitsandbytes`, `accelerate`

**Dikkat:** Depo, Python kaynak kodu yanı sıra kurulum betikleri, CI iş akışları ve paket araçları içerir; bu nedenle "%100 saf Python" iddiası yapılmaz.

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10, 3.11 veya 3.12 (CI testi yapılan sürümler; 3.13+ henüz desteklenmiyor)
- GPU ile CUDA (önerilen), Apple Silicon (MPS) veya CPU (deneysel — çok yavaş)
- QLoRA ile 7B modeller için 8 GB+ VRAM

### Kurulum Adımları

```bash
# Hafif çekirdek: CLI + yapılandırma + veri araçları, PyTorch yok
pipx install soup-cli
uv tool install soup-cli          # zaten uv kullanıyorsanız

# Eğitim yığınını ekleyin (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Her şey bir arada (train + serve + ui + data)
pipx install "soup-cli[all]"

# GitHub'dan (en güncel geliştirme)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

Zaten bir virtualenv, Colab not defteri veya Docker görüntüsünün içindeyseniz `pip` kullanın:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **`error: externally-managed-environment`?** Bu [PEP 668](https://peps.python.org/pep-0668/), Soup sorunu değil. `pipx` veya `uv tool` kullanın.

> **Çift tırnak, tek tırnak değil.** `"soup-cli[train]"` her kabukta çalışan tek yazımdır.

Tam extras tablosu (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, `liger`, `onnx`, `tensorrt`, …): [`docs/models.md`](docs/models.md#optional-extras)

### Çataldan Kurulum

Bu depo `Ercaner1988/Soup` çatalıdır. Upstream: `MakazhanAlpamys/Soup`.

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

### Sorun Giderme

```bash
soup doctor    # GPU, sistem kaynakları, bağımlılıklar ve sürüm tek yerde kontrol eder
```

CUDA tekerlekleri, sürüm uyuşmazlıkları: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#sorun-giderme)

---

## 📖 Kullanım

### Hızlı Başlangıç

```bash
# 1. Yapılandırma oluşturun
soup init                       # etkileşimli sihirbaz
soup init --template chat       # şablondan başlayın

# 2. Eğitin, test edin, gönderin
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo kullanici/modelim

# 3. Birleştirin ve dışa aktarın
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**Şablonlar:** `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`

### Temel Komutlar

```bash
soup train  --config soup.yaml                         # eğitim (SFT/DPO/GRPO/PPO/KTO/ORPO/…)
soup infer  --model ./output --input istemler.jsonl    # toplu çıkarım
soup chat   --model ./output                           # etkileşimli sohbet
soup serve  --model ./output                           # OpenAI uyumlu API sunucusu
soup ui                                                # yerel tarayıcı panosu
soup merge  --adapter ./output                         # LoRA'yı taban modele birleştir
soup export --model ./output --format gguf             # dağıtım için dışa aktar
soup eval   benchmark --model ./output                 # değerlendir
soup data   inspect ./data/train.jsonl                 # veri kümesi istatistikleri
soup recipes list                                      # 100+ hazır model tarifi
soup autopilot --model <id> --data d.jsonl --goal chat # sıfır-yapılandırma
soup doctor                                            # GPU / bağımlılıklar / ortam kontrolü
```

### Yapılandırma

Tam bir `soup.yaml`:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x daha hızlı, pip install "soup-cli[fast]"

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

`config/schema.py` her alan için tek kaynaktır.

> **Bilinmeyen yapılandırma anahtarları bugün uyarı verir, v0.75'te reddedilecek.**

### Veri Biçimleri

Alpaca, ShareGPT, ChatML, tercih çiftleri (DPO / ORPO / SimPO / IPO / KTO), görsel, ses, ASR, düz metin, gömme, RAFT ve daha fazlası — JSONL, JSON, CSV, Parquet veya TXT'den otomatik algılama. Bkz. [`docs/data.md`](docs/data.md).

### Desteklenen Modeller

Soup, [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation)'daki **herhangi** bir metin üretim modeliyle çalışır. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral, Mixtral, DeepSeek R1/V3, Phi-4 ve 100+ diğerleri hazır tarifler olarak gelir (`soup recipes list`).

| VRAM | Maksimum model (QLoRA 4-bit) | Örnek |
|------|-------------------------------|-------|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (tam) veya MoE | Mixtral-8x22B, DeepSeek-V3 |

### Web Arayüzü

```bash
pip install "soup-cli[ui]"
soup ui
# http://127.0.0.1:7860 adresini açar
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ Test ve Kalite Kapıları

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — her commit öncesi temiz olmalı
pytest tests/ -v --tb=short        # birim testleri (hızlı, GPU gerekmez; 460 test dosyası)
pytest tests/ -m smoke -v          # duman testleri (küçük model indirir, eğitir)
pre-commit install                 # isteğe bağlı: committe ruff lint+biçimlendirme
```

**CI matrisi:** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**Güvenlik:** `soup doctor` GPU, sistem kaynakları, bağımlılıklar ve sürümü tek yerde kontrol eder. Telemetri kesinlikle katılım bazlıdır (`SOUP_TELEMETRY=1`, varsayılan kapalı).

---

## 📚 Belgeler

| Rehber | Kapsam |
|--------|--------|
| [Eğitim görevleri ve yöntemler](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, araç çağırma, PRM, ön eğitim, damıtma |
| [PEFT, uzun bağlam ve verimlilik](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA |
| [Performans ve nicemleme](docs/performance-and-quantization.md) | QAT, FP8, KV önbellek, katman akışı, çoklu GPU / DeepSpeed / FSDP |
| [Veri mühendisliği](docs/data.md) | Biçimler, veri araçları, yapay üretim, tarif DAG'ları |
| [Değerlendirme ve sondalar](docs/evaluation.md) | Eval tasarımı, kıyaslamalar, NLG metrikleri, A/B |
| [Sunma ve dışa aktarma](docs/serving-and-export.md) | OpenAI uyumlu sunucu, birleştirme/dışa aktarma, Web Arayüzü |
| [Adaptörler ve yönetim](docs/adapters-and-governance.md) | Adaptör yaşam döngüsü, model kayıt defteri, Soup Cans |
| [Uyumluluk](docs/compliance.md) | HIPAA/SOC2/AB-YZ-Yasası/SR-11-7 şablonları, denetim günlüğü |
| [Arka uçlar ve işlemler](docs/backends-and-ops.md) | MLX/Unsloth arka uçları, HF Hub, otopilot |
| [Komut referansı](docs/commands.md) | Tam `soup` komut listesi |
| [Desteklenen modeller ve extras](docs/models.md) | Model aileleri, VRAM rehberi, pip extras matrisi |

### Geliştirme

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # birim testleri (hızlı, GPU gerekmez)
pytest tests/ -m smoke -v          # duman testleri (küçük model indirir, eğitir)

pre-commit install                 # isteğe bağlı: committe ruff lint+biçimlendirme
```

Tam iş akışı için [CONTRIBUTING.md](CONTRIBUTING.md) ve güvenlik açığı bildirmek için [SECURITY.md](SECURITY.md) dosyalarına bakın. Telemetri kesinlikle katılım bazlıdır (`SOUP_TELEMETRY=1`, varsayılan kapalı; bkz. [Gizlilik İlkesi](docs/backends-and-ops.md#gizlilik-politikası)).

### Soup'u Destekleyin

Soup Apache-2.0 ve ücretsiz — ve öyle kalacak. Tek bir 4 GB laptop üzerinde açık kaynak olarak inşa ediliyor ve bakımı yapılıyor, bu nedenle bu belgelerdeki her performans numarası iddia değil ölçümdür.

Soup bir eğitim çalıştırması kurtardıysa, [depoyu yıldızlamak](https://github.com/MakazhanAlpamys/Soup) en çok yardımcı olur ve hiçbir maliyeti yoktur. Doğrudan çalışmayı finanse etmek isterseniz:

**[❤️ Bağış yapın](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — tek seferlik, herhangi bir tutar (ödeme sayfasında *Tutarı değiştir* kullanın). Ödemeler bakımcının kayıtlı işletmesi **MePlay, Inc.** tarafından Stripe aracılığıyla işlenir — bu ad, "Soup" değil, ödeme sayfasında ve kart ekstresinde görünen şeydir.

Bağışlar, tek bir 4 GB laptop'un ulaşamayacağı donanım kapılı çalışmalar için GPU süresi satın alır — çoklu GPU, 8B+ doğrulama, Apple Silicon.

Diğer tam olarak bu öğeleri taşımanın yolu **donanımın kendisidir**. Daha büry box'unuz varsa — veya kullanılmayan GPU kredileri — [`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) sorunlarından birini çalıştırıp numaraları göndermek, GPU süresini finanse etmek kadar yardımcı olur. Bu sorunlar bugün donanım tarafından tam olarak neyin engellendiğini söyler.

### İletişim

Hata raporları ve özellik istekleri için [sorun izleyicisi](https://github.com/MakazhanAlpamys/Soup/issues), sorular için [Tartışmalar](https://github.com/MakazhanAlpamys/Soup/discussions) — her ikisi de daha hızlı yanıt alır ve aynı sorunu yaşayan bir sonraki kişiye yardımcı olur.

Canlı sohbet, kurulum yardımı ve konuşma olarak daha iyi okunan her şey için [Discord](https://discord.gg/dgd2pJcjwP) katılın. Altı ay sonra hâlâ bulunabilir olması gereken her şey Sorunlar veya Tartışmalarda yer almalı — Discord yanıtı bir kişiye yardımcı olur, bir sorun aynı şeyle karşılaşan herkese yardımcı olur. [Davranış Kuralları](CODE_OF_CONDUCT.md) orada da geçerlidir.

Herkese açık olmayan her şey için — güvenlik raporları (bkz. [SECURITY.md](SECURITY.md)), Davranış Kuralları konuları veya basın — **team@trysoup.dev** adresine e-posta gönderin. Bu proje adresidir ve Soup ile ilgili her şey için doğru olanıdır. **makazanalpamys@gmail.com** bakımcının kişisel adresidir; aynı kişiye ulaşır ve iyi bir yedek seçenektir.

---

## 👥 Katkıda Bulunanlar

Upstream depo (`MakazhanAlpamys/Soup`) topluluğun katkısıyla inşa edilmiştir. Bu çatal (`Ercaner1988/Soup`) Graft MCP bütünleşmesi için özelleştirilmiştir.

| Katkıda Bulunan | Commitler | Rol |
|----------------|-----------|-----|
| Alpamys (Alpamys Makazhan) | 912 | Proje sahibi ve baş geliştirici |
| AmixDigital | 37 | Web Arayüzü ön kapı bölümü |
| Srinivasan R | 24 | MLX yanıt maskesi, doğrulama kayıpları, tarif kimliği koruma |
| Amir Fathi | 24 | Katkılar |
| Ben Younes | 15 | Katkılar |
| Salil Mhatre | 11 | Katkılar |
| Darsh | 9 | Katkılar |
| Nurkhan Esenbek | 9 | Katkılar |
| Harshit Sharma | 7 | Katkılar |
| UmranPros | 6 | Katkılar |
| (diğer 24 katkıda bulunan) | 5–1 | Bkz. [CONTRIBUTORS.md](CONTRIBUTORS.md) |
| **Ercan Er** | **2** | **Graft MCP yapılandırması (kullanıcı düzeyi kablo bağlantısı)** |

> Ölçümler `git shortlog -sn HEAD` çıktısından alınmıştır (ercan1 dalı, 2026-09-09).

---

## 📄 Lisans

[Apache-2.0](LICENSE). Telif hakkı © Soup katkıda bulunanları.

Bu çatal (`Ercaner1988/Soup`) özgün projeyi temel alır: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

### Alıntı

Katman akışı — donmuş tabanı ana bilgisayar RAM'inden bir kod çözücü katmanı aynı anda akıştırarak 4 GB laptop GPU'da 8B model eğitimi — bir ön baskıda, akışan bir çalıştırmanın yerleşik bir karşı doğru protokolü ile birlikte açıklanmıştır (ileri ve geri ayrı ayrı belirtilir, çünkü bunlar iki iddia değil birincidir).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**Sürüm 3 (13 Ağustos 2026) günceldir.** Başlık ve iddia değişmedi — 8B 4 GB üzerinde — ve v1'den bu yana ölçülen hiçbir sayı değişmedi. Sürüm 3'ün yaptığı, **yayınladığımız bir açıklamayı geri çekmektir**, bu da makalenin ne olduğunu tanımlamanın en kısa yoludur:

- **Sürüm 3'te geri çekilen: "katman akışı, ana bilgisayardan cihaza aktarıma bağlıdır, GPU'ya değil."** Bu, aşağıdaki H100 replikasyonundan bir *çıkarımdı* ve hiç ölçülmemişti. 11 Ağustos'ta ölçtük ve yayınlanan yapılandırmada yanlıştır: her ana bilgisayardan cihaza baytı silmek **%1.4** satın alır, hesaplama akışı bir kopyayı **%0.20** adım bekler ve adım o kartın aynı oturum GEMM tavanının **%71.3**'ünde çalışır. En büyük akışa özgü maliyet katman başına NF4 ters nicemlemedir, **%9.8** ([kayıt](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). Her ölçüm geçerli; replikasyon daha zayıf bir biçimde hayatta kalır — kısıtlama her iki makinede ortaktır ve GPU'nun hesabı değildir.
- **Orijinalden çok farklı donanımda replikasyon** (v2'de eklendi): RTX 3050'de 119.6 tok/s, H100'de medyan 113.00'a karşı, aynı 3.32 GB tepe değerinde.
- **Sessiz bir yanlış gradyan kusuru, bulundu ve onarıldı.** Katman başına ~165 MiB üzerinde NF4'te ileri bit-özdeş kaldı ve kayıp eğrisi sağlıklı görünürken gradyanlar yanlıştı. Neden akış kütüphanesinde adlandırılmış ve orada bildirilmiştir; onarım gerçek 32B ve 72B üzerindeki kontrollere karşı korumalıdır.
- **Gerçek model boyutlarında bit-özdeşlik** üç katmanlı oyuncaklar yerine: 0.5B'den 72B'ye ileri, 8B ve 14B'de geri.
- **İlk kez ölçülen eğitilmiş model kalitesi** ve yerleşik bir çalıştırmadan ayırt edilemez.
- **DeepSpeed karşılaştırması** — bizi pohpohlamayan sonuç dahil: ZeRO-3'ün sekiz kartı, bir kart yerleşik eğitimden daha yavaş.
- **Sınırlar bölümü yeniden yazıldı:** v1'in on maddesinden biri kapandı ve dördü daha daraltıldı, yedi yeni eklendi.

Kullandığınız sürümü alıntılayın. `10.5281/zenodo.21771064` kavram DOI'sidir ve her zaman en son sürüme çözümlenir (bugün v3); v1 ve v2 kendi sürüm DOI'lerinde alıntılanabilir ve düzenlenmez — yukarıdaki geri çekme, tam olarak kaydın neyi, ne zaman iddia ettiğimizi bozulmadan tutmak için yeni bir sürümdür.

Arkasındaki ölçüm kayıtları [`benchmarks/`](benchmarks/) içindedir, yazıldığı gibi yayınlanmış — başarısızlıklar, yanlış çıkan varsayımlar ve ölçülüp atılan numaralar dahil.

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
