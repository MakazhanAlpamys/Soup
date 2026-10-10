<!-- synced-from: README.md sha256:ec1dafd943f745bda3caed2f8e2719be574711b084c5e99c91ca62715ab7d2da -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <strong>Русский</strong></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Дообучайте LLM и проводите их пост-обучение одной командой. Без SSH и без ада конфигураций.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Сайт</a> &middot;
  <a href="#быстрый-старт">Быстрый старт</a> &middot;
  <a href="#веб-интерфейс">Веб-интерфейс</a> &middot;
  <a href="#конфигурация">Конфигурация</a> &middot;
  <a href="#документация">Документация</a> &middot;
  <a href="docs/commands.md">Команды</a> &middot;
  <a href="docs/models.md">Модели</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Загрузки"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Лицензия Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Тесты"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Сайт"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - дообучение 8B-модели на видеокарте ноутбука с 4 GB VRAM | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup превращает мучительное дообучение LLM в простой рабочий процесс. Один конфиг, одна команда — и готово.

```bash
pip install "soup-cli[train]"   # add [train] to fine-tune; bare `soup-cli` is the light CLI
soup init --template chat
soup train
```

**Дообучите модель 8B на видеокарте ноутбука с 4 GB VRAM.** Потоковая подгрузка слоёв (layer streaming) держит
замороженную базовую модель вне VRAM и подаёт её на GPU по одному слою декодера за раз. Замеры на
RTX 3050 Laptop 4 GB: Llama-3.1-8B-Instruct + NF4 показывает **119.6 tok/s и пик 3.32 GB** — результат
побитово совпадает с обычным запуском, при котором модель целиком находится в видеопамяти, и независимо
воспроизведён на H100: 113.00 tok/s при тех же 3.32 GB.
(Обе цифры измерены на v0.72.2, до исправления корректности в v0.73.0, которое стоило
−4.8% на 32B; ни одна из них с тех пор не перепроверялась на карте с 4 GB — повторный замер
ожидается в issue [#361](https://github.com/MakazhanAlpamys/Soup/issues/361).) Функция включается
явно (`stream_layers: true`) и пока остаётся в статусе BETA —
[как это работает](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[все замеры](benchmarks/) · [статья](https://doi.org/10.5281/zenodo.21771064) ·
**[проверьте сами на бесплатном Colab T4](notebooks/proof-4gb.ipynb)** (ограничивает процесс
4 GB, а затем проверяет, что модель в потоковом режиме побитово совпадает с обычной)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="Предварительная проверка soup train для Llama-3.1-8B на карте с 4 GB: хранилище базовой модели объёмом 3.60 GB, закреплённое в RAM и разбитое на 32 слоя, и два буфера VRAM по 113 MB, затем измеренный пик 3.32 GB при 119.6 tok/s, не доходящий до отметки 4 GB (измерено на v0.72.2, до исправления #331; повторный замер ожидается в issue #361)"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, батч 1, последовательность 512, RTX 3050 Laptop 4 GB — <b>пик 3.32 GB, 119.6 tok/s</b> (измерено на v0.72.2, до исправления #331; повторный замер ожидается в issue #361). <a href="https://youtu.be/T1LCErE943E">Полное видео (90 с)</a></sub>
</p>

## Зачем нужен Soup?

Обучать LLM по-прежнему мучительно. Даже опытные команды тратят 30–50% времени на борьбу
с инфраструктурой вместо улучшения моделей. Soup решает эту проблему.

- **Никакого SSH.** Больше никогда не подключайтесь по SSH к сломанному GPU-серверу.
- **Один конфиг.** Достаточно простого YAML-файла.
- **Всё автоматически.** Размер батча, определение GPU, квантизация — всё берёт на себя Soup.
- **Работает локально.** Обучайте на собственной GPU с помощью QLoRA. Облако не нужно.

## Что нового

**v0.75.0 — один и тот же `soup.yaml` молча обучал на MLX не тот рецепт, что на transformers.**
Шесть обучающих опций проходили проверку, были описаны в документации, принимались — но этот бэкенд их попросту не
читал. **Все 60 pull request'ов этого релиза пришли не от мейнтейнера**,
а от 22 человек.

- **Несовместимое изменение: неизвестный ключ конфигурации теперь прерывает загрузку.**
  v0.74 выдавал предупреждение и называл этот релиз крайним сроком. Опечатка вроде `quantizaton`
  или ключ, существующий только в более новой версии Soup, раньше молча отбрасывался, а запуск
  продолжался без применения настройки; теперь загрузка конфигурации завершается ошибкой в CLI (код выхода 1)
  и в API (`ValueError`) и называет поле, которое вы, скорее всего, имели в виду. Детектор учитывает
  переназначение корневого `lora:`, которое схема поддерживает с v0.40.1, так что такая запись
  принимается, а не отвергается; два файла `soup fetch examples`, где она использовалась, переведены
  на каноническую форму `training.lora`. Все рецепты и шаблоны загружаются без замечаний, имена
  ключей экранируются перед выводом в терминал, а сам проход по ключам ограничен по объёму.
- **MLX учитывает конфигурацию, которую принимает.** `train_on_responses_only`, `warmup_ratio` /
  `scheduler` / `weight_decay` / `optimizer`, `max_grad_norm`, `gradient_accumulation_steps`
  и `gradient_checkpointing` проходили проверку, а затем отбрасывались при `backend: mlx`. Только
  8 из 32 названий оптимизаторов имеют аналог в MLX; остальные 24 отклоняются по имени, а не
  молча превращаются в AdamW. MLX теперь также обслуживает живую панель, трекер
  и `soup ui`, а `soup doctor --config` перечисляет настройки, которые бэкенд не читает.
- **Валидационный лосс нигде не сохранялся.** Он вычислялся на каждом бэкенде и выбрасывался:
  ни столбца в метриках, ни поля в событии, ничего на панели. Теперь он записывается,
  передаётся потоком и отображается.
- **Несовместимое изменение: `grpo_variant: gspo` — это опубликованная целевая функция уровня
  последовательности** (arXiv:2507.18071), заменившая эвристику центрирования по столбцам, в которой
  паддинг-токен тоже сдвигал градиент каждой строки, делящей с ним столбец. Существующие конфигурации
  gspo не воспроизведут прежние запуски.
- **Эндпоинты чтения веб-интерфейса и SSE требуют аутентификации**, причём с короткоживущими
  одноразовыми тикетами вместо токена в строке запроса; `--public` больше не отдаёт `/docs`
  и `/openapi.json` в локальную сеть; а подпроцесс обучения больше не зависает, когда никто не
  читает его вывод.
- **`torch>=2.6.0`** снимает известное ограничение v0.74.0: на 2.5.1 `trl>=0.29` не импортировался,
  и все тренеры предпочтений не работали. Также исправлено: `training.loraplus_lr_ratio`
  приводил к падению каждого запуска, где он был задан, а `packing: true` вызывал ошибку на TRL 0.29.

> Только Python **3.10–3.12**. На 3.13+ pip раньше подбирал непротестированные сборки PyTorch,
> которые падали в нативном расширении ещё до запуска Soup.

Прежние главные изменения собраны на странице [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

## Быстрый старт

### 1. Установка

Soup — приложение командной строки, поэтому самая чистая установка даёт ему собственное
окружение и добавляет `soup` в ваш `PATH`:

```bash
# Light core: CLI + config + data tools, no PyTorch
pipx install soup-cli
uv tool install soup-cli          # same idea, if you already use uv

# Add the training stack (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Everything (train + serve + ui + data) in one shot
pipx install "soup-cli[all]"

# Or from GitHub (latest dev)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

Уже работаете внутри виртуального окружения, блокнота Colab или образа Docker? Используйте `pip`
напрямую, с теми же именами и extras:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

Берите `pip`, а не `pipx`, если вам нужно ещё и делать `import soup_cli` из собственного
кода: pipx намеренно изолирует приложение от всего остального.

Полная таблица extras (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) находится в
[`docs/models.md`](docs/models.md#optional-extras).

> **Видите `error: externally-managed-environment`?** Это
> [PEP 668](https://peps.python.org/pep-0668/), а не проблема Soup. Debian 12,
> Ubuntu 23.04 и новее не позволяют `pip` писать в системный Python, потому что этими файлами
> управляет и `apt`. `pipx` и `uv tool` обходят это ограничение, выделяя Soup собственное окружение,
> поэтому они и стоят в списке выше первыми. `python3 -m venv
> .venv && source .venv/bin/activate`, а затем обычный `pip` подойдут не хуже.

> **Двойные кавычки, а не одинарные.** `"soup-cli[train]"` — единственная запись, которая работает во
> всех оболочках: `cmd.exe`, PowerShell, bash и zsh. Если вы скопировали `'soup-cli[train]'` из
> устаревшего руководства и pip отверг её, причина именно в этом:
> [почему так и как выглядит точная ошибка](docs/models.md#quoting-the-extra).

`soup init`, `soup data …` и другие команды для работы с данными и диагностики работают и в лёгкой установке.
Для дообучения (`soup train`) нужен extra `[train]`.

### 2. Создайте конфигурацию

```bash
soup init                       # interactive wizard
soup init --template chat       # or start from a template
```

Шаблоны: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Обучайте, проверяйте, публикуйте

```bash
soup train --config soup.yaml                 # LoRA, quantization, batching — all handled
soup chat  --model ./output                    # talk to your model
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # merge LoRA into the base
soup export --model ./output --format gguf --quant q4_k_m   # GGUF for Ollama / llama.cpp
```

Другие форматы экспорта (ONNX, TensorRT, AWQ, GPTQ, BitNet) и варианты развёртывания описаны в
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Веб-интерфейс

Предпочитаете браузер? `soup ui` поднимает локальную панель для экспериментов,
настройки обучения, живых метрик, просмотра датасетов и чата с моделью.

```bash
pip install "soup-cli[ui]"
soup ui
# Opens http://127.0.0.1:7860
```

![Веб-интерфейс Soup — новое обучение](docs/assets/web-ui-new-training.png)

[Документация по веб-интерфейсу](docs/serving-and-export.md#web-ui)

## Конфигурация

Полный `soup.yaml`:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install "soup-cli[fast]"

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

`config/schema.py` — единственный источник истины для каждого поля. Расширенные параметры данных,
обучения и PEFT описаны в разделе [Документация](#документация).

> **Неизвестные ключи конфигурации отвергаются начиная с v0.75.** Ключ, который не объявлен ни в одной
> модели схемы, — опечатка вроде `quantizaton` или поле, существующее только в более новой версии Soup, —
> раньше проходил проверку и молча отбрасывался: запуск продолжался, а настройка
> просто не применялась. v0.74 сообщал об этом при загрузке и называл поле, которое вы, скорее всего,
> имели в виду; начиная с **v0.75** такая же конфигурация не загружается, поэтому исправьте или удалите
> ключ, а не рассчитывайте, что он будет проигнорирован. См.
> [Неизвестные ключи конфигурации](docs/backends-and-ops.md#unknown-config-keys).

## Документация

Полный справочник по возможностям находится в [`docs/`](docs/). Начните отсюда:

| Руководство | Что описано |
|---|---|
| [Задачи и методы обучения](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, вызов инструментов, PRM, предобучение, дистилляция, классификация, изображения (vision)/аудио/TTS, разобучение (unlearning), RAFT/RA-DIT, детекторы для защиты цикла обучения (loop hardening) |
| [PEFT, длинный контекст и эффективность](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, зоопарк оптимизаторов и PEFT, LLaMA Pro, GaLore, YaRN/LongLoRA, упаковка последовательностей, curriculum-обучение, автонастройка |
| [Производительность и квантизация](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-кэш, NVFP4, форматы сохранения, Cut Cross-Entropy, градиентное чекпойнтирование (gradient checkpointing), ядра, выгрузка активаций, потоковая подгрузка слоёв, мульти-GPU / DeepSpeed / FSDP |
| [Инженерия данных](docs/data.md) | Форматы, конвейер с паритетом возможностей относительно Axolotl/LF, инструменты для данных, синтетическая генерация и forge, оценочные карточки качества, инструменты для трейсов, удалённые датасеты, смешивание, DAG рецептов |
| [Оценка и зондирование (probes)](docs/evaluation.md) | Проектирование оценки и гейт, обучение с гейтом по метрикам оценки, бенчмарки, метрики NLG, калибровка, арена Elo, диагностика, «рентгеновское» зондирование модели после обучения, A/B, дрейф, дообучаемость (tunability), `soup advise` |
| [Сервинг и экспорт](docs/serving-and-export.md) | OpenAI-совместимый сервер, пакетный инференс, бенчмаркинг, слияние/экспорт, эндпоинт Anthropic Messages, спекулятивное декодирование (обучите и измерьте собственную черновую модель), автопилот развёртывания, веб-интерфейс, Agent Forge |
| [Адаптеры, реестр и управление (governance)](docs/adapters-and-governance.md) | Жизненный цикл и управление адаптерами, реестр моделей, Soup Cans, маховик данных (`soup loop`), редактирование знаний, управление активациями (steering), контроль цепочки поставок (scan/sign/BOM/attest/audit/airgap) |
| [Быстрый старт по комплаенсу и управлению (governance)](docs/compliance.md) | Шаблоны `init` для HIPAA/SOC2/EU-AI-Act/SR-11-7, происхождение (BOM/attest/repro-receipt), журнал аудита, air-gap, автогенерация карточки модели (`soup card`), гейт CI (`soup ci init`) |
| [Бэкенды, платформа и эксплуатация](docs/backends-and-ops.md) | Бэкенды MLX/Unsloth, альтернативные хабы, интеграция с HF Hub, автопилот, трекинг экспериментов, plan/apply, lock-файлы окружения, подбор под железо, автодополнение, плагины, служебные команды |
| [Справочник команд](docs/commands.md) | Полный список команд `soup` |
| [Поддерживаемые модели и extras](docs/models.md) | Рекомендуемые семейства моделей, руководство по размеру VRAM, матрица pip extras |

## Форматы данных

Alpaca, ShareGPT, ChatML, пары предпочтений (DPO / ORPO / SimPO / IPO / KTO), изображения, аудио,
ASR, обычный текст, эмбеддинги, RAFT и другие — все они определяются автоматически в файлах JSONL, JSON, CSV, Parquet или
TXT, поэтому в большинстве случаев достаточно указать в `data.train` путь к файлу, и больше ничего менять не нужно. Схемы с
готовым примером для каждого формата, а также конвейер данных (удалённые URI, потоковая обработка, шардирование,
чередование, расширение словаря, приём документов) описаны в
[`docs/data.md`](docs/data.md#data-formats).

## Часто используемые команды

```bash
soup train  --config soup.yaml        # train (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl --output results.jsonl   # batch inference
soup chat   --model ./output          # interactive chat
soup serve  --model ./output          # OpenAI-compatible API server
soup ui                               # local browser dashboard
soup merge  --adapter ./output        # merge LoRA into the base model
soup export --model ./output --format gguf           # export for deployment
soup eval   benchmark --model ./output               # evaluate
soup data   inspect ./data/train.jsonl               # dataset stats
soup recipes list                     # 100+ ready-made model recipes
soup autopilot --model <id> --data d.jsonl --goal chat  # zero-config
soup doctor                           # check GPU / deps / environment
```

Полный список команд приведён в [`docs/commands.md`](docs/commands.md).

## Поддерживаемые модели

Soup работает с **любой** моделью генерации текста на
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation): если она загружается через
`AutoModelForCausalLM`, то заработает без каких-либо изменений конфигурации. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4 и более 100 других поставляются как готовые рецепты (`soup recipes list`).

| VRAM | Максимальная модель (QLoRA 4-bit) | Пример |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) или MoE | Mixtral-8x22B, DeepSeek-V3 |

Полные таблицы моделей и vision-моделей, а также матрица необязательных extras находятся в [`docs/models.md`](docs/models.md).

## Docker

Запускайте Soup без локальной установки CUDA и PyTorch (образ публикуется в GHCR при каждом релизе):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # or build locally
```

## Требования

- Python 3.10, 3.11 или 3.12 (именно эти версии тестируются в CI; 3.13+ пока не поддерживается,
  потому что стек PyTorch на них ещё не проверен)
- GPU с CUDA (рекомендуется), Apple Silicon (MPS) или CPU (экспериментально — очень медленно)
- 8 GB+ VRAM для моделей на 7B с QLoRA

Для тестирования все задачи обучения можно запускать на CPU (квантизация при этом отключается автоматически). Необязательные extras
(`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) перечислены в
[`docs/models.md`](docs/models.md#optional-extras).

## Устранение неполадок

```bash
soup doctor    # GPU, system resources, dependencies, and version in one place
```

Wheel-пакеты с CUDA, несовместимость версий: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Разработка

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # unit tests (fast, no GPU)
pytest tests/ -m smoke -v          # smoke tests (downloads a tiny model, trains)
pytest tests/ -m gpu --no-cov -v   # GPU tests (need a CUDA card; report results, see CONTRIBUTING.md)

pre-commit install                 # optional: ruff lint+format on commit
```

Полный рабочий процесс описан в [CONTRIBUTING.md](CONTRIBUTING.md), а о найденной уязвимости сообщайте
по инструкции из [SECURITY.md](SECURITY.md). Телеметрия строго добровольная (`SOUP_TELEMETRY=1`, по умолчанию выключена; см. [Политику конфиденциальности](docs/backends-and-ops.md#privacy-policy)).

## Поддержите Soup

Soup распространяется под Apache-2.0, бесплатен и останется таким. Он создаётся и поддерживается открыто,
на единственном ноутбуке с 4 GB, и именно поэтому каждая цифра производительности в этой документации
измерена, а не заявлена.

Если Soup сэкономил вам запуск обучения, больше всего поможет [звезда на репозитории](https://github.com/MakazhanAlpamys/Soup),
а она ничего не стоит. Если хотите профинансировать работу напрямую:

**[❤️ Пожертвовать](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — разово, в любом размере (на странице
оплаты воспользуйтесь пунктом *Change amount*). Платежи обрабатывает Stripe от имени зарегистрированной
компании мейнтейнера, **MePlay, Inc.** — именно это название, а не «Soup», вы увидите на странице оплаты
и в выписке по карте.

Пожертвования идут на оплату GPU-времени для задач, зависящих от железа, — мульти-GPU, проверка на 8B+,
Apple Silicon, — до которых одному ноутбуку с 4 GB не дотянуться.

Другой способ сдвинуть именно эти задачи — **само железо**. Эти задачи честно помечены
ограничением «требуется \<оборудование\>» и не сопровождаются непроверенными заявлениями, так что если у вас есть доступ к машине
помощнее или неиспользуемые кредиты на GPU, то запуск одного из issue с меткой
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
и публикация полученных цифр помогут не меньше, чем оплата GPU-времени. В этих issue прямо сказано,
что сегодня упирается в железо.

## Участники

Создано сообществом ❤️ — спасибо всем, кто внёс свой вклад. См.
[CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Участники](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## Контакты

Сообщения об ошибках и пожелания по функциям — в
[issue tracker](https://github.com/MakazhanAlpamys/Soup/issues), вопросы — в
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions): так вам ответят быстрее,
а следующему человеку с той же проблемой будет проще.

Для живого общения, помощи с установкой и всего, что лучше обсуждать в разговоре, заходите в
[Discord](https://discord.gg/dgd2pJcjwP) или в [сообщество в Telegram](https://t.me/souptasters).
Всё, что должно остаться доступным для поиска и через полгода,
оставляйте в Issues или Discussions: ответ в Discord помогает одному человеку, а issue — всем,
кто столкнётся с тем же. [Кодекс поведения](CODE_OF_CONDUCT.md) действует и там.

По вопросам, которые не подходят для публичного обсуждения, — сообщения о безопасности (см. [SECURITY.md](SECURITY.md)),
вопросы Кодекса поведения или обращения прессы — пишите на **team@trysoup.dev**. Это адрес проекта
и правильный адрес для всего, что связано с Soup. **makazanalpamys@gmail.com** — личный адрес
мейнтейнера; письмо дойдёт до того же человека, так что это приемлемый запасной вариант.

## Как цитировать Soup

Потоковая подгрузка слоёв — обучение модели на 8B на видеокарте ноутбука с 4 GB VRAM за счёт подачи замороженной базовой модели
из оперативной памяти хоста по одному слою декодера за раз — описана в препринте вместе с протоколом проверки
корректности, который сверяет запуск в потоковом режиме с запуском, где модель целиком находится в видеопамяти (прямой и обратный
проходы описаны по отдельности, потому что это два утверждения, а не одно).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**Актуальна версия 3 (13 августа 2026).** Название и главное утверждение не изменились — 8B на 4 GB, —
и ни одно измеренное число не менялось со времён v1. Вместо этого v3 **отзывает объяснение, которое мы публиковали
раньше**, и это же самый короткий способ сказать, зачем нужна эта статья:

- **Отозвано в v3: «потоковая подгрузка слоёв упирается в передачу с хоста на устройство, а не в GPU».**
  Это был *вывод* из описанного ниже воспроизведения на H100, и его никогда не измеряли. Мы измерили его 11 августа,
  и для опубликованной конфигурации он неверен: если убрать все байты, передаваемые с хоста на устройство, выигрыш составит
  **1.4%**, вычислительный поток ждёт копирования лишь **0.20%** шага, а сам шаг достигает **71.3%**
  потолка GEMM этой же карты в той же сессии. Крупнейшие затраты, присущие именно потоковому режиму, — это деквантизация NF4 на каждом слое: 9.8%
  ([протокол измерения](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). Все измерения остаются в силе;
  воспроизведение сохраняется в ослабленном виде: ограничение общее для обеих машин
  и дело не в вычислительной мощности GPU.
- **Воспроизведение на железе, совсем не похожем на исходное** (добавлено в v2): 119.6 tok/s на RTX
  3050 против медианных 113.00 на H100 при том же пике 3.32 GB. Оба замера сделаны до
  исправления #331; повторный замер на 4 GB ожидается в issue #361.
- **Найден и исправлен скрытый дефект с неверными градиентами.** На NF4 при размере слоя выше ~165 MiB
  прямой проход оставался побитово точным, а кривая потерь выглядела здоровой, хотя градиенты были неверны. Причина названа в вышестоящей (upstream) библиотеке, и о ней там сообщено;
  исправление защищено проверкой на контрольных запусках на реальных моделях 32B и 72B.
- **Побитовое совпадение на реальных размерах моделей**, а не на игрушечных трёхслойных: прямой проход — от 0.5B до 72B,
  обратный — на 8B и 14B.
- **Качество обученной модели измерено впервые** и неотличимо от запуска с моделью целиком в видеопамяти.
- **Сравнение с DeepSpeed** — включая результат, который нам не льстит: восемь карт с ZeRO-3
  медленнее одной карты, обучающей модель целиком в видеопамяти.
- **Раздел об ограничениях переписан**: из десяти пунктов v1 один закрыт, ещё четыре сужены,
  и добавлено семь новых.

Цитируйте ту версию, которой вы пользовались. `10.5281/zenodo.21771064` — это концептуальный DOI, он всегда ведёт на
последнюю версию (сегодня это v3); v1 и v2 остаются доступными для цитирования по собственным DOI версий и не
редактируются — отзыв выше оформлен как новая версия именно для того, чтобы запись о том, что мы заявляли
и когда, осталась нетронутой.

Записи измерений, лежащие за каждым числом в статье, опубликованы в [`benchmarks/`](benchmarks/)
как есть — включая неудачи, предположения, оказавшиеся неверными, и числа, которые были измерены, а затем
отброшены.

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

## Лицензия

[Apache-2.0](LICENSE). Copyright © участники проекта Soup.
