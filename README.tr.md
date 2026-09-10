<p align="center"><strong>🌍 <a href="README.md">Türkçe</a> | <a href="README.en.md">English</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.ru.md">Русский</a> | <a href="README.es.md">Español</a></strong></p>

<!-- synced-from: README.md sha256:73b53c056cd7db0f63c60a2e04e194606a6aa34cbae9b69ef29b732d85b336d3 -->

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>LLM'leri tek komutla ince ayar yapın ve son eğitimden geçirin. SSH yok, yapılandırma cehennemi yok.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Web Sitesi</a> &middot;
  <a href="#quick-start">Hızlı Başlangıç</a> &middot;
  <a href="#web-ui">Web Arayüzü</a> &middot;
  <a href="#configuration">Yapılandırma</a> &middot;
  <a href="#documentation">Dokümantasyon</a> &middot;
  <a href="docs/commands.md">Komutlar</a> &middot;
  <a href="docs/models.md">Modeller</a> &middot;
  <a href="https://discord.gg/dgd2pjcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="İndirmeler"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/lisans-Apache--2.0-blue" alt="Apache-2.0 Lisansı">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Testler"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/web-sitesi-trysoup.dev-mavi" alt="Web Sitesi"></a>
  <a href="https://discord.gg/dgd2pjcjwP"><img src="https://img.shields.io/badge/Discord-katıl-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-katıl-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp&utm_source=badge-featured&amp&utm_medium=badge&amp&utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - Fine-tune an 8B LLM on a 4 GB laptop GPU | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp&utm_medium=badge&amp&utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup, LLM ince ayar yapma zahmetini basit bir iş akışına dönüştürür. Bir yapılandırma, bir komut, tamamlandı.

```bash
pip install "soup-cli[train]"   # ince ayar için [train] ekleyin; bare `soup-cli` hafif CLI'dir
soup init --template chat
soup train
```

**4 GB'lık bir dizüstü bilgisayarda 8B model ince ayar yapın.** Katman akışı, dondurulmuş tabanı VRAM dışında tutar ve bir seferde bir kod çözücü katmanını GPU'ya besler.

## Neden Soup?

LLM eğitimi hâlâ acı verici. Deneyimli ekipler bile zamanlarının %30-50'sini altyapıyla boğuşarak geçiriyor, modelleri geliştirmek yerine. Soup bunu düzeltir.

- **Sıfır SSH.** Bir daha asla bozulmuş bir GPU kutusuna SSH ile bağlanmak zorunda kalmayın.
- **Bir yapılandırma.** Basit bir YAML dosyası ihtiyacınız olan tek şey.
- **Her şeyi otomatik.** Toplu iş boyutu, GPU algılama, nicemleme — hallolur.
- **Yerel çalışır.** QLoRA ile kendi GPU'nuzda eğitin. Bulut gerekmez.

## Yenilikler

**v0.74.0 — dondurulmuş taban tüm bu süre boyunca fp32 olarak yükleniyordu.** Bunu düzeltmek tek başına değişmemiş bir yapılandırmada tepe VRAM'ı 2.59x keser.

## Hızlı Başlangıç

```bash
pip install "soup-cli[train]"
soup init --template chat
soup train
```

## Web Arayüzü

```bash
soup web
```

## Yapılandırma

`config.yaml` dosyasında tüm eğitim seçenekleri tanımlanır.

## Dokümantasyon

Tam dokümantasyon: `docs/` klasöründe.

## Veri Formatları

- Chat formatı, Completion formatı, Tercih formatı desteklenir.

## Yaygın Komutlar

- `soup init` - Yeni proje başlat
- `soup train` - Eğitimi başlat
- `soup export` - Modeli dışa aktar
- `soup web` - Web arayüzünü başlat

## Desteklenen Modeller

Llama 3.1, Qwen2.5, Mistral ve diğerleri desteklenir.

## Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup-cli:latest
```

## Gereksinimler

- Python 3.10-3.12
- NVIDIA GPU (CUDA) veya Apple Silicon (MLX)

## Sorun Giderme

VRAM yetersizse `load_in_4bit: true` kullanın veya `batch_size` değerini düşürün.

## Geliştirme

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Katkıcılar

Topluluk tarafından inşa edildi.

## İletişim

[GitHub Issues](https://github.com/MakazhanAlpamys/Soup/issues) üzerinden bildirin.

## Lisans

Apache-2.0. Telif hakkı © Soup katkıcıları.
