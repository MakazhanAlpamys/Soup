**🌍 [Türkçe](README.md) | [English](README.en.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)**

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
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
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
**[ücretsiz Colab T4'te kendiniz kontrol edin](notebooks/proof-4gb.ipynb)**

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
