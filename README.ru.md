<p align="center">**🌍 [Türkçe](README.md) | [English](README.en.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)**

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Дообучение и пост-тренинг LLM одной командой. Никакого SSH, никакого ада конфигураций.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Сайт</a> &middot;
  <a href="#быстрый-старт">Быстрый старт</a> &middot;
  <a href="#веб-интерфейс">Веб-интерфейс</a> &middot;
  <a href="#конфигурация">Конфигурация</a> &middot;
  <a href="docs/commands.md">Команды</a> &middot;
  <a href="docs/models.md">Модели</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Загрузки"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Лицензия Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Тесты"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
</p>

---

Soup превращает боль дообучения LLM в простой рабочий процесс. Один конфиг, одна команда, готово.

```bash
pip install "soup-cli[train]"   # добавьте [train] для дообучения; просто soup-cli — это легкий CLI
soup init --template chat
soup train
```

**Дообучите 8B модель на 4GB GPU ноутбука.** Потоковая передача слоёв сохраняет замороженную базу вне VRAM и подаёт GPU по одному слою декодера за раз. Измерено на RTX 3050 Laptop 4GB: Llama-3.1-8B-Instruct + NF4 при **119.6 tok/s, пик 3.32 GB** — побитно идентично с обычным резидентным запуском и независимо воспроизведено на H100 при 113.00 tok/s с теми же 3.32 GB. Опционально (`stream_layers: true`) и всё ещё BETA —
[как работает](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[все измерения](benchmarks/) ·
[статья](https://doi.org/10.5281/zenodo.21771064) ·
**[проверьте сами на бесплатном Colab T4](notebooks/proof-4gb.ipynb)**

---

## 🎯 Почему Soup?

Обучение LLM всё ещё болезненно. Даже опытные команды тратят 30-50% времени на борьбу с инфраструктурой вместо улучшения моделей. Soup это исправляет.

- **Никакого SSH.** Никогда больше не заходите по SSH в сломанный GPU-бокс.
- **Один конфиг.** Нужен только простой YAML-файл.
- **Всё автоматически.** Размер батча, обнаружение GPU, квантизация — обработано.
- **Работает локально.** Обучайте на своём GPU с QLoRA. Облако не нужно.

### Что нового — v0.74.0

**v0.74.0 — замороженная база всё время загружалась в fp32.** Исправление только этого сокращает пиковый VRAM в 2.59 раза при неизменённом конфиге. **116 из 120 объединённых pull request в этом релизе пришли извне от мейнтейнера**, от 25 человек.

- Каждая загрузка SFT молча повышала замороженную базу до fp32. Измерено на H100 с Llama-3.1-8B + LoRA: **48,241 MiB → 18,658 MiB пик — 2.59x, 28.9 GB**.
- **Transformers 5.x, TRL 0.29, PEFT 0.20.**
- Исправлены четыре SSRF-обхода одинакового типа (сокращённые, десятичные, шестнадцатеричные и восьмеричные IPv4).
- **`soup serve` теперь возвращает код 2** при привязке к не-loopback хосту без `--tool-auth-token`.
- **`soup train --cloud lambda`** — по умолчанию только план, `--cloud-submit` отправляет вживую.

---

## 🏗️ Архитектура и модули

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70+ CLI команд (train, eval, data, serve, agent, …)
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — единственный источник истины для всех полей конфигурации
│   ├── autodistill/     # Автодистилляция (захват данных, публикация, MLX worker)
│   ├── autopilot/       # Выбор задач/квантизации/LR без конфигурации
│   ├── cans/            # Soup Cans: упаковка моделей, публикация, верификация
│   ├── cloud/           # Контроллеры облаков Lambda Labs / Modal
│   └── utils/           # Общие помощники (350+ модулей)
├── graft/               # Конфигурация Graft MCP (пользовательская проводка)
├── docs/                # Тематические руководства + полный справочник команд
├── templates/           # Стартовые шаблоны chat.yaml, code.yaml, medical.yaml
├── benchmarks/          # Записи измерений (сырые данные для каждого числа)
├── tests/               # 460 тестовых файлов
└── pyproject.toml       # Зависимости, дополнения, система сборки
```

**Основные зависимости (лёгкая установка — без PyTorch):**
`typer`, `rich`, `pydantic>=2`, `pyyaml`, `huggingface-hub`, `plotext`, `packaging`

**Дополнения `[train]`:** `torch>=2.6.0`, `transformers>=5.16.1,<6`, `peft>=0.20,<1`, `trl>=0.29`, `datasets`, `bitsandbytes`, `accelerate`

**Примечание:** Этот репозиторий содержит исходный код Python наряду со скриптами установки, CI-процессами и инструментами упаковки; мы не заявляем о "%100 чистом Python".

---

## 🚀 Установка

### Требования

- Python 3.10, 3.11 или 3.12 (это версии, которые тестирует CI; 3.13+ пока не поддерживается)
- GPU с CUDA (рекомендуется), Apple Silicon (MPS) или CPU (экспериментально — очень медленно)
- 8GB+ VRAM для 7B моделей с QLoRA

### Шаги установки

```bash
# Лёгкое ядро: CLI + конфиг + инструменты данных, без PyTorch
pipx install soup-cli
uv tool install soup-cli          # та же идея, если вы уже используете uv

# Добавить стек обучения (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Всё сразу (train + serve + ui + data)
pipx install "soup-cli[all]"

# Или с GitHub (последняя разработка)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

Уже внутри virtualenv, Colab notebook или Docker образа? Используйте `pip` напрямую:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **`error: externally-managed-environment`?** Это [PEP 668](https://peps.python.org/pep-0668/), не проблема Soup. Используйте `pipx` или `uv tool`.

> **Двойные кавычки, не одинарные.** `"soup-cli[train]"` — единственное написание, которое работает во всех оболочках.

Полная таблица дополнений (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, `liger`, `onnx`, `tensorrt`, …): [`docs/models.md`](docs/models.md#optional-extras)

### Установка из форка

Этот репозиторий — форк `Ercaner1988/Soup`. Upstream: `MakazhanAlpamys/Soup`.

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

---

## 📖 Использование

### Быстрый старт

```bash
# 1. Создать конфиг
soup init                       # интерактивный мастер
soup init --template chat       # или начать с шаблона

# 2. Обучить, протестировать, отправить
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo user/my-model

# 3. Объединить и экспортировать
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**Шаблоны:** `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`

### Основные команды

```bash
soup train  --config soup.yaml                         # обучение (SFT/DPO/GRPO/PPO/KTO/ORPO/…)
soup infer  --model ./output --input prompts.jsonl    # пакетный инференс
soup chat   --model ./output                           # интерактивный чат
soup serve  --model ./output                           # OpenAI-совместимый API сервер
soup ui                                                # локальная браузерная панель
soup merge  --adapter ./output                         # объединить LoRA в базовую модель
soup export --model ./output --format gguf             # экспорт для развёртывания
soup eval   benchmark --model ./output                 # оценка
soup data   inspect ./data/train.jsonl                 # статистика датасета
soup recipes list                                      # 100+ готовых рецептов моделей
soup autopilot --model <id> --data d.jsonl --goal chat # без конфигурации
soup doctor                                            # проверка GPU / зависимостей / окружения
```

### Конфигурация

Полный `soup.yaml`:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # в 2-5 раз быстрее, pip install "soup-cli[fast]"

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

`config/schema.py` — единственный источник истины для каждого поля.

> **Неизвестные ключи конфигурации предупреждают сегодня и будут отклонены в v0.75.**

### Форматы данных

Alpaca, ShareGPT, ChatML, пары предпочтений (DPO / ORPO / SimPO / IPO / KTO), видение, аудио, ASR, простой текст, эмбеддинги, RAFT и др. — автоопределение из JSONL, JSON, CSV, Parquet или TXT. См. [`docs/data.md`](docs/data.md).

### Поддерживаемые модели

Soup работает с **любой** моделью генерации текста на [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral, Mixtral, DeepSeek R1/V3, Phi-4 и 100+ других поставляются как готовые рецепты (`soup recipes list`).

| VRAM | Максимальная модель (QLoRA 4-bit) | Пример |
|------|-----------------------------------|--------|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (полная) или MoE | Mixtral-8x22B, DeepSeek-V3 |

### Веб-интерфейс

```bash
pip install "soup-cli[ui]"
soup ui
# Открывает http://127.0.0.1:7860
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ Тестирование и контроль качества

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — должно быть чисто перед любым коммитом
pytest tests/ -v --tb=short        # unit тесты (быстрые, без GPU; 460 тестовых файлов)
pytest tests/ -m smoke -v          # smoke тесты (загружают маленькую модель, обучают)
pre-commit install                 # опционально: ruff lint+формат при коммите
```

**CI матрица:** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**Безопасность:** `soup doctor` проверяет GPU, системные ресурсы, зависимости и версию в одном месте. Телеметрия строго opt-in (`SOUP_TELEMETRY=1`, по умолчанию выключена).

---

## 📚 Документация

| Руководство | Охватывает |
|-------------|-----------|
| [Задачи и методы обучения](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, вызов инструментов, PRM, предобучение, дистилляция |
| [PEFT и эффективность](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA |
| [Производительность и квантизация](docs/performance-and-quantization.md) | QAT, FP8, KV-кэш, потоковая передача слоёв, мульти-GPU / DeepSpeed / FSDP |
| [Инженерия данных](docs/data.md) | Форматы, инструменты данных, синтетическая генерация, DAG рецептов |
| [Оценка и зонды](docs/evaluation.md) | Дизайн оценки, бенчмарки, NLG метрики, A/B |
| [Сервинг и экспорт](docs/serving-and-export.md) | OpenAI-совместимый сервер, слияние/экспорт, веб-UI |
| [Адаптеры и управление](docs/adapters-and-governance.md) | Жизненный цикл адаптеров, реестр моделей, Soup Cans |
| [Соответствие требованиям](docs/compliance.md) | Шаблоны HIPAA/SOC2/EU-AI-Act/SR-11-7, журнал аудита |
| [Бэкенды и операции](docs/backends-and-ops.md) | Бэкенды MLX/Unsloth, HF Hub, автопилот |
| [Справочник команд](docs/commands.md) | Полный список команд `soup` |
| [Модели и дополнения](docs/models.md) | Семейства моделей, руководство по VRAM, матрица pip дополнений |

---

## 👥 Участники

Upstream репозиторий (`MakazhanAlpamys/Soup`) построен сообществом. Этот форк (`Ercaner1988/Soup`) настроен для интеграции Graft MCP.

| Участник | Коммиты | Роль |
|----------|---------|------|
| Alpamys (Alpamys Makazhan) | 912 | Владелец проекта и ведущий разработчик |
| AmixDigital | 37 | Секция входной двери веб-UI |
| Srinivasan R | 24 | MLX маска только ответов, потери валидации, защита recipe-id |
| Amir Fathi | 24 | Вклады |
| Ben Younes | 15 | Вклады |
| Salil Mhatre | 11 | Вклады |
| Darsh | 9 | Вклады |
| Nurkhan Esenbek | 9 | Вклады |
| Harshit Sharma | 7 | Вклады |
| UmranPros | 6 | Вклады |
| (ещё 24 участника) | 5–1 | См. [CONTRIBUTORS.md](CONTRIBUTORS.md) |
| **Ercan Er** | **2** | **Конфигурация Graft MCP (пользовательская проводка)** |

> Метрики из вывода `git shortlog -sn HEAD` (ветка ercan1, 2026-09-09).

---

## 📄 Лицензия

[Apache-2.0](LICENSE). Авторские права © участники Soup.

Этот форк (`Ercaner1988/Soup`) основан на оригинальном проекте: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

### Цитирование

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