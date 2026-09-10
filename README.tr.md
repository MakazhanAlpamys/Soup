<!-- synced-from: README.md sha256:1cdf85f67b56a053d993850d6d8ee27e161e3efd9d2bf6d3ec10e68691fcc3b6 -->
<!-- Bu dosya README.md'nin Türkçe çevirisidir. Stamp eşleşmiyorsa çeviriyi güncelleyin. -->
<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>LLM'leri tek komutla ince ayarlayın ve sonradan eğitin. SSH yok, yapılandırma cehennemi yok.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Web Sitesi</a> &middot;
  <a href="#hızlı-başlangıç">Hızlı Başlangıç</a> &middot;
  <a href="#web-arayüzü">Web Arayüzü</a> &middot;
  <a href="#yapılandırma">Yapılandırma</a> &middot;
  <a href="#belgeler">Belgeler</a> &middot;
  <a href="docs/commands.md">Komutlar</a> &middot;
  <a href="docs/models.md">Modeller</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="İndirmeler"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 Lisansı">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Testler"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Web Sitesi"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-katıl-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&theme=light" alt="Soup CLI - 4 GB laptop GPU'da 8B LLM ince ayarı | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&utm_medium=badge&utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

**🌍 Çeviriler:** [English](README.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)

---

Soup, LLM ince ayarının karmaşıklığını basit bir iş akışına dönüştürür. Tek yapılandırma, tek komut, tamam.

```bash
pip install "soup-cli[train]"   # [train] ince ayar için; yalın `soup-cli` hafif CLI'dır
soup init --template chat
soup train
```

**4 GB laptop GPU'da 8B model ince ayarlayın.** Katman akışı, dondurulmuş tabanı VRAM dışında tutarak GPU'ya tek seferde bir kod çözücü katman besler. RTX 3050 Laptop 4 GB üzerinde ölçüldü: Llama-3.1-8B-Instruct + NF4 ile **119,6 tok/s, 3,32 GB tepe** — normal yerleşik çalışmayla bit düzeyinde özdeş ve H100 üzerinde bağımsız olarak 113,00 tok/s ile aynı 3,32 GB'de yeniden üretildi.
(tok/s değeri, v0.73.0 doğruluk düzeltmesinden önce v0.72.2'de ölçüldü; 4 GB kartta o tarihten bu yana yeniden çalıştırılmadı.) Katıl (`stream_layers: true`) ve hâlâ BETA —
[nasıl çalışır](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[tüm ölçümler](benchmarks/) · [makale](https://doi.org/10.5281/zenodo.21771064) ·
**[ücretsiz Colab T4'te kendiniz deneyin](notebooks/proof-4gb.ipynb)** (işlemi 4 GB ile sınırlayıp akışlı modelin normal çalışmayla bit düzeyinde özdeş olduğunu doğrular)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="soup train ön uçuşu: 4 GB kartta Llama-3.1-8B, 32 katmanda RAM'e sabitlenmiş 3,60 GB taban deposu ve iki 113 MB VRAM tamponu, ardından 119,6 tok/s ile ölçülen 3,32 GB tepe"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, toplu iş 1, dizi 512, RTX 3050 Laptop 4 GB — <b>3,32 GB tepe, 119,6 tok/s</b>. <a href="https://youtu.be/T1LCErE943E">Tam video (90 sn)</a></sub>
</p>

## Neden Soup?

LLM eğitimi hâlâ acı verici. Deneyimli ekipler bile modellerini iyileştirmek yerine zamanlarının %30-50'sini altyapıyla boğuşmaya harcar. Soup bunu çözer.

- **SSH yok.** Bozuk GPU kutusuna bir daha SSH atmayın.
- **Tek yapılandırma.** Basit bir YAML dosyası yeterli.
- **Her şey otomatik.** Toplu iş boyutu, GPU algılama, niceleme — halledildi.
- **Yerel çalışır.** QLoRA ile kendi GPU'nuzda eğitin. Bulut gerekmez.

## Yenilikler

**v0.74.0 — dondurulmuş taban sürekli fp32'de yükleniyordu.** Bunu düzeltmek tek başına değişmemiş bir yapılandırmada tepe VRAM'i 2,59 kat düşürdü. **Bu sürümdeki 120 birleştirilmiş pull request'in 116'sı bakıcı dışından**, 25 kişiden geldi.

- **Her SFT yüklemesi dondurulmuş tabanı sessizce fp32'ye yükseltti.** Hiçbir zaman optimizer adımı almayan bir taban, üç yükleme yolunun tamamında iki kat denetim noktası hassasiyetinde somutlaştırıldı. H100 üzerinde Llama-3.1-8B + LoRA ile ölçüldü: **48.241 MiB → 18.658 MiB tepe — 2,59 kat, 28,9 GB**, üç tekrarda bayt özdeş. Eğitilebilir bir taban hâlâ kasıtlı olarak fp32 yüklenir.
- **Transformers 5.x, TRL 0.29, PEFT 0.20.** Qwen3.5 ailesi metin kod çözücüleri Transformers yolunda eğitilir ve `pip install "soup-cli[train,mlx]"` artık çözümlenir — iki ek daha önce birlikte karşılanamayan aralıklar bildiriyordu.
- **Ücretsiz Colab/Kaggle katmanı hiç akış yapamıyordu.** T4 / P100 / V100 / GTX 16xx katman akışını çökertiyordu; çünkü peft, LoRA bağdaştırıcılarını denetim noktasının veri türünde oluştururken fp16 GradScaler, fp32 gradyanlarına ihtiyaç duyar.
- **Aynı biçimdeki dört SSRF atlatması.** Kısaltılmış, ondalık, onaltılık ve sekizli IPv4 yazımları (`127.1`, `2130706433`, `0x7f000001`, `0177.0.0.1`) telemetri ve değerlendirme HTTP istemcilerine ulaşıyordu.
- **Tercih kaybı akışı** (DPO/ORPO/SimPO/BCO/KTO). İnce ayar çalıştırmasında `loss: dpo` yapılandırıldığında katman akışı artık düzgün çalışır.
- **`soup diff` ve `soup eval` komutları** tam katman akışı desteğiyle çalışır.

## Hızlı Başlangıç

### 1. Kurulum

Soup bir komut satırı uygulamasıdır; en temiz kurulum ona kendi ortamını verir ve `soup`'u `PATH`'e ekler:

```bash
# Hafif çekirdek: CLI + yapılandırma + veri araçları, PyTorch yok
pipx install soup-cli
uv tool install soup-cli          # uv kullanıyorsanız aynı fikir

# Eğitim yığınını ekle (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Her şey bir arada (train + serve + ui + data)
pipx install "soup-cli[all]"

# Ya da GitHub'dan (son geliştirme sürümü)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

Zaten bir sanal ortam, Colab not defteri veya Docker görüntüsü içindeyseniz `pip` kullanın:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

Kendi kodunuzdan `import soup_cli` yapmak istiyorsanız `pipx` yerine `pip` kullanın; pipx uygulamayı kasıtlı olarak izole eder.

Tam ek tablosu (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) [`docs/models.md`](docs/models.md#optional-extras) adresindedir.

> **`error: externally-managed-environment`?** Bu [PEP 668](https://peps.python.org/pep-0668/), Soup sorunu değil. Debian 12, Ubuntu 23.04 ve sonrası `pip`'in sistem Python'una yazmasını engeller. `pipx` ve `uv tool` Soup'a kendi ortamını vererek bunu aşar. `python3 -m venv .venv && source .venv/bin/activate` ardından düz `pip` de işe yarar.

> **Çift tırnak, tek tırnak değil.** `"soup-cli[train]"` her kabukta çalışan tek yazımdır — `cmd.exe`, PowerShell, bash ve zsh. Eski bir eğiticiden `'soup-cli[train]'` kopyaladıysanız ve pip reddettiyse, neden olduğu burada: [açıklama ve tam hata](docs/models.md#quoting-the-extra).

### 2. Yapılandırma oluşturma

```bash
soup init                       # etkileşimli sihirbaz
soup init --template chat       # ya da şablondan başla
```

Şablonlar: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Eğit, test et, yayınla

```bash
soup train --config soup.yaml                 # LoRA, niceleme, toplu işleme — hepsi halledildi
soup chat  --model ./output                    # modelinizle konuşun
soup push  --model ./output --repo siz/modelim

soup merge  --adapter ./output                              # LoRA'yı tabana birleştir
soup export --model ./output --format gguf --quant q4_k_m   # Ollama / llama.cpp için GGUF
```

Diğer dışa aktarma hedefleri (ONNX, TensorRT, AWQ, GPTQ, BitNet) ve dağıtım seçenekleri [`docs/serving-and-export.md`](docs/serving-and-export.md) adresindedir.

## Web Arayüzü

Tarayıcıyı mı tercih ediyorsunuz? `soup ui` denemeler, eğitim kurulumu, canlı ölçütler, veri kümesi keşfi ve model sohbeti için yerel bir pano sunar.

```bash
pip install "soup-cli[ui]"
soup ui
# http://127.0.0.1:7860 adresini açar
```

![Soup Web Arayüzü — Yeni Eğitim](docs/assets/web-ui-new-training.png)

[Web Arayüzü belgeleri](docs/serving-and-export.md#web-ui)

## Yapılandırma

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

`config/schema.py` her alan için tek doğru kaynaktır. Gelişmiş veri, eğitim ve PEFT seçenekleri [Belgeler](#belgeler) altında belgelenmiştir.

> **Bilinmeyen yapılandırma anahtarları bugün uyarı verir, v0.75'te reddedilecek.** Hiçbir modelin bildirmediği bir anahtar — `quantizaton` gibi bir yazım hatası veya yalnızca daha yeni bir Soup sürümünde bulunan bir alan — temiz olarak doğrulanıp atılıyordu. Artık yükleme sırasında muhtemelen kastettiğiniz alan ile birlikte raporlanır. **v0.75**'ten itibaren aynı yapılandırma uyarı yerine yüklenemeyecek; anahtarı düzeltin veya kaldırın. Bkz. [Bilinmeyen yapılandırma anahtarları](docs/backends-and-ops.md#unknown-config-keys).

## Belgeler

Tam özellik referansı [`docs/`](docs/) dizinindedir. Buradan başlayın:

| Rehber | Kapsam |
|---|---|
| [Eğitim görevleri ve yöntemleri](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, araç çağrısı, PRM, ön eğitim, damıtma, sınıflandırma, görü/ses/TTS, unutturma, RAFT/RA-DIT, döngü sertleştirme dedektörleri |
| [PEFT, uzun bağlam ve verimlilik](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, optimizer ve PEFT hayvanat bahçesi, LLaMA Pro, GaLore, YaRN/LongLoRA, paketleme, müfredat, otomatik ayarlama |
| [Performans ve niceleme](docs/performance-and-quantization.md) | QAT, FP8, Niceleme Menüsü (I+II), KV-önbellek, NVFP4, kayıt biçimleri, Kesim Çapraz Entropi, gradyan denetim noktası, çekirdekler, etkinleştirme boşaltma, katman akışı, çok GPU / DeepSpeed / FSDP |
| [Veri mühendisliği](docs/data.md) | Biçimler, Axolotl/LF eşdeğeri ardışık düzen, veri araçları, yapay üretim ve forge, kalite puan kartları, izleme araçları, uzak veri kümeleri, karıştırma, tarif DAG'ları |
| [Değerlendirme ve sondalar](docs/evaluation.md) | Değerlendirme tasarımı/kapı, değerlendirme geçitli eğitim, kıyaslamalar, NLG ölçütleri, kalibrasyon, Elo arenası, teşhis, eğitim sonrası X-ışını sondaları, A/B, kayma, ayarlanabilirlik, `soup advise` |
| [Sunum ve dışa aktarma](docs/serving-and-export.md) | OpenAI uyumlu sunucu, toplu çıkarım, kıyaslama, birleştirme/dışa aktarma, Anthropic Messages uç noktası, spekülatif kod çözme, dağıtım otomatik pilotu, Web Arayüzü, Etmen Forge |
| [Bağdaştırıcılar, kayıt defteri ve yönetişim](docs/adapters-and-governance.md) | Bağdaştırıcı yaşam döngüsü/yönetimi, model kayıt defteri, Soup Kutuları, veri volan (`soup loop`), bilgi düzenleme, yönlendirme, tedarik zinciri kontrolleri |
| [Uyumluluk ve yönetişim hızlı başlangıç](docs/compliance.md) | HIPAA/SOC2/AB-YZ-Yasası/SR-11-7 şablonları, köken (BOM/tasdik/yeniden üretim makbuzu), denetim günlüğü, hava boşluğu, model kartı otomatik üretimi (`soup card`), CI kapısı (`soup ci init`) |
| [Arka uçlar, platform ve operasyonlar](docs/backends-and-ops.md) | MLX/Unsloth arka uçları, alternatif hub'lar, HF Hub bütünleşmesi, otomatik pilot, deney izleme, plan/uygula, env kilit dosyaları, donanım uyumu, tamamlamalar, eklentiler, yardımcı komutlar |
| [Komut referansı](docs/commands.md) | Tam `soup` komut listesi |
| [Desteklenen modeller ve ekstralar](docs/models.md) | Önerilen model aileleri, VRAM boyut rehberi, pip ekstraları matrisi |

## Veri Biçimleri

Alpaca, ShareGPT, ChatML, tercih çiftleri (DPO / ORPO / SimPO / IPO / KTO), görü, ses, ASR, düz metin, gömme, RAFT ve daha fazlası — hepsi JSONL, JSON, CSV, Parquet veya TXT'den otomatik algılanır; çoğu durumda `data.train`'i bir dosyaya yönlendirmeniz yeterlidir. Biçim başına çalışma örneğiyle şemalar ve veri ardışık düzeni [`docs/data.md`](docs/data.md#data-formats) adresindedir.

## Yaygın Komutlar

```bash
soup train  --config soup.yaml        # eğit (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl   # toplu çıkarım
soup chat   --model ./output          # etkileşimli sohbet
soup serve  --model ./output          # OpenAI uyumlu API sunucusu
soup ui                               # yerel tarayıcı panosu
soup merge  --adapter ./output        # LoRA'yı taban modele birleştir
soup export --model ./output --format gguf           # dağıtım için dışa aktar
soup eval   benchmark --model ./output               # değerlendir
soup data   inspect ./data/train.jsonl               # veri kümesi istatistikleri
soup recipes list                     # 100+ hazır model tarifi
soup autopilot --model <id> --data d.jsonl --goal chat  # sıfır yapılandırma
soup doctor                           # GPU / bağımlılıklar / ortamı denetle
```

Tam komut listesi [`docs/commands.md`](docs/commands.md) adresindedir.

## Desteklenen Modeller

Soup, [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation)'daki **herhangi** bir metin üretim modeliyle çalışır — `AutoModelForCausalLM` ile yükleniyorsa, sıfır yapılandırma değişikliğiyle çalışır. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral, Mixtral, DeepSeek R1/V3, Phi-4 ve 100'den fazlası hazır tarif olarak gelir (`soup recipes list`).

| VRAM | Maks model (QLoRA 4-bit) | Örnek |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (tam) veya MoE | Mixtral-8x22B, DeepSeek-V3 |

Tam model + görü tabloları ve isteğe bağlı ek matrisi [`docs/models.md`](docs/models.md) adresindedir.

## Docker

CUDA veya PyTorch'u yerel olarak kurmadan Soup'u çalıştırın (görüntü her sürümde GHCR'ye yayınlanır):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # ya da yerel olarak derleyin
```

## Gereksinimler

- Python 3.10, 3.11 veya 3.12 (CI'nin test ettiği sürümler; 3.13+ henüz PyTorch yığını doğrulanmadığından desteklenmiyor)
- CUDA'lı GPU (önerilen), Apple Silicon (MPS) veya CPU (deneysel — çok yavaş)
- 7B modeller için QLoRA ile 8 GB+ VRAM

Tüm eğitim görevleri test için CPU'da çalışır (niceleme otomatik devre dışı). İsteğe bağlı ekstralar
(`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) [`docs/models.md`](docs/models.md#optional-extras) adresindedir.

## Sorun Giderme

```bash
soup doctor    # GPU, sistem kaynakları, bağımlılıklar ve sürüm tek yerde
```

CUDA tekerlekleri, sürüm uyumsuzlukları: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Geliştirme

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # birim testleri (hızlı, GPU yok)
pytest tests/ -m smoke -v          # duman testleri (küçük model indirir, eğitir)

pre-commit install                 # isteğe bağlı: commit'te ruff lint+format
```

Tam iş akışı için [CONTRIBUTING.md](CONTRIBUTING.md), güvenlik açığı bildirmek için [SECURITY.md](SECURITY.md) adresine bakın. Telemetri kesinlikle isteğe bağlıdır (`SOUP_TELEMETRY=1`, varsayılan kapalı; bkz. [Gizlilik Politikası](docs/backends-and-ops.md#privacy-policy)).

## Soup'u Destekleyin

Soup Apache-2.0 ve ücretsizdir — ve öyle kalacak. Tek bir 4 GB laptop'ta açık olarak geliştirilip sürdürülmektedir; bu yüzden bu belgelerdeki her performans rakamı iddia değil ölçümdür.

Soup size bir eğitim çalıştırması kazandırdıysa, [depoyu yıldızlamak](https://github.com/MakazhanAlpamys/Soup) en çok yardımcı olur ve hiçbir maliyeti yoktur. Çalışmayı doğrudan desteklemek isterseniz:

**[❤️ Bağış Yapın](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — tek seferlik, herhangi bir miktar (ödeme sayfasında *Tutarı değiştir* kullanın). Ödemeler, bakıcının kayıtlı işletmesi **MePlay, Inc.** adına Stripe tarafından işlenir — ödeme sayfasında ve kart ekstrenizdeki isim "Soup" değil budur.

Bağışlar çok GPU, 8B+ doğrulama, Apple Silicon gibi donanım gereksinimli çalışmalar için GPU süresi satın alır — tek bir 4 GB laptop'ın ulaşamadığı işler.

Bu öğeleri taşımanın diğer yolu **donanımın kendisidir**. Bunlar dürüst "donanım gerektirir" kapılarının arkasında gösterilir. Daha büyük bir kutuya veya kullanılmayan GPU kredilerine erişiminiz varsa, [`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) sorunlarından birini çalıştırıp sayıları göndermek, GPU süresini finanse etmek kadar yardımcı olur.

## Katkıda Bulunanlar

Topluluk tarafından inşa edildi ❤️ — katkıda bulunan herkese teşekkürler. Bkz. [CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Katkıda Bulunanlar](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

Bu depo [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup) deposunun çatalıdır. Çatala özgü katkılar:

| Katkıda Bulunan | Katkı |
|-----------------|-------|
| Ercan Er | Graft MCP yapılandırması (kullanıcı düzeyine taşındı), 7 dilli çeviri altyapısı + senkronizasyon testi |

## İletişim

Hatalar ve özellik istekleri [sorun takipçisine](https://github.com/MakazhanAlpamys/Soup/issues), sorular [Discussions](https://github.com/MakazhanAlpamys/Soup/discussions) kısmına gidin — her ikisi de daha hızlı yanıt alır ve aynı sorunla karşılaşan bir sonraki kişiye yardımcı olur.

Canlı sohbet, kurulum yardımı ve konuşma olarak daha iyi okunan her şey için [Discord](https://discord.gg/dgd2pJcjwP)'a veya [Telegram topluluğuna](https://t.me/souptasters) katılın. Altı ay sonra hâlâ bulunabilir olması gereken her şey Issues veya Discussions'a aittir. [Davranış Kuralları](CODE_OF_CONDUCT.md) orada da geçerlidir.

Kamuya açık olmayan şeyler için — güvenlik raporları ([SECURITY.md](SECURITY.md)), Davranış Kuralları konuları veya basın — **team@trysoup.dev** adresine e-posta gönderin. Bu proje adresidir. **makazanalpamys@gmail.com** bakıcının kişisel adresidir; aynı kişiye ulaşır ve iyi bir yedektir.

## Soup'u Atıfta Bulunma

Katman akışı — dondurulmuş tabanı ana RAM'den tek seferde bir kod çözücü katman aktararak 4 GB laptop GPU'da 8B model eğitimi — bir ön baskıda doğruluk protokolüyle birlikte açıklanmıştır.

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**Sürüm 3 (13 Ağustos 2026) günceldir.** Başlık ve iddia değişmedi — 4 GB'da 8B — ve v1'den bu yana hiçbir ölçülen sayı değişmedi. v3'ün yaptığı şey **yayımladığımız bir açıklamayı geri çekmektir**:

- **v3'te geri çekildi: "katman akışı GPU değil, ana bilgisayardan cihaza aktarımla sınırlıdır."** Bu, aşağıdaki H100 çoğaltmasından yapılan bir *çıkarımdı* ve hiç ölçülmemişti. 11 Ağustos'ta ölçtük ve yayımlanan yapılandırmada yanlış: tüm ana bilgisayardan cihaza baytları silmek **%1,4** kazandırıyor, hesaplama akışı **%0,20** oranında bir kopyayı bekliyor ve adım o kartın aynı oturumdaki GEMM tavanının **%71,3**'ünde çalışıyor. En büyük akışa özgü maliyet katman başına NF4 ters nicelemesi, **%9,8** ([kayıt](benchmarks/probe-v0.73.0-what-bounds-streaming.md)).
- **Orijinalden çok farklı donanımda çoğaltma** (v2'de eklendi): RTX 3050'de 119,6 tok/s, H100'de ortalama 113,00 tok/s, aynı 3,32 GB tepe.
- **Sessiz yanlış gradyan hatası, bulundu ve düzeltildi.**
- **Gerçek model boyutlarında bit düzeyinde özdeşlik** — üç katmanlı oyuncaklar yerine: 0,5B'den 72B'ye ileri, 8B ve 14B'de geri.
- **Eğitilmiş model kalitesi, ilk kez ölçüldü** ve yerleşik çalışmadan ayırt edilemez.
- **DeepSpeed ile karşılaştırma** — bizi pohpohlamayan sonuç dahil: ZeRO-3'ün sekiz kartı, yerleşik eğiten tek karttan daha yavaş.

Kullandığınız sürümü atıfta bulunun. `10.5281/zenodo.21771064` kavram DOI'sidir ve her zaman en son sürüme çözümlenir; v1 ve v2 kendi sürüm DOI'leriyle atıfta bulunulabilir ve düzenlenmez.

Arkasındaki ölçüm kayıtları [`benchmarks/`](benchmarks/) dizinindedir — başarısızlıklar, yanlış çıkan varsayımlar ve ölçülüp sonra atılan sayılar dahil.

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

## Lisans

[Apache-2.0](LICENSE). Telif hakkı © Soup katkıda bulunanları.
