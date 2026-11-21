<!-- synced-from: README.md sha256:5b33b108c15916db92b052ddf457ca8000e2647ea124924cb08287eac083f19f -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.es.md">Español</a> | <strong>Português</strong></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Faça fine-tuning e pós-treinamento de LLMs com um único comando. Sem SSH, sem inferno de configuração.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Site</a> &middot;
  <a href="#início-rápido">Início Rápido</a> &middot;
  <a href="#interface-web">Interface Web</a> &middot;
  <a href="#configuração">Configuração</a> &middot;
  <a href="#documentação">Documentação</a> &middot;
  <a href="docs/commands.md">Comandos</a> &middot;
  <a href="docs/models.md">Modelos</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Licença Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Testes"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Site"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - Fine-tuning de um LLM de 8B em uma GPU de notebook de 4 GB | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

O Soup transforma a dor do fine-tuning de LLMs em um fluxo de trabalho simples. Uma configuração, um comando, pronto.

```bash
pip install "soup-cli[train]"   # add [train] to fine-tune; bare `soup-cli` is the light CLI
soup init --template chat
soup train
```

**Faça fine-tuning de um modelo de 8B em uma GPU de notebook de 4 GB.** O layer streaming mantém a base
congelada fora da VRAM e a envia para a GPU uma camada do decoder por vez. Medido em uma RTX 3050
Laptop de 4 GB: Llama-3.1-8B-Instruct + NF4 a **119.6 tok/s, pico de 3.32 GB** — idêntico bit a bit a
uma execução residente normal, e reproduzido de forma independente em uma H100 a 113.00 tok/s nos
mesmos 3.32 GB. (Ambos os números foram medidos na v0.72.2, antes da correção de corretude da v0.73.0,
que custou −4.8% em 32B; nenhum deles foi medido novamente em uma placa de 4 GB desde então — a nova
medição está pendente na issue [#361](https://github.com/MakazhanAlpamys/Soup/issues/361).) É opcional
(`stream_layers: true`) e ainda está em BETA —
[como funciona](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[todas as medições](benchmarks/) · [artigo](https://doi.org/10.5281/zenodo.21771064) ·
**[confira você mesmo em um Colab T4 gratuito](notebooks/proof-4gb.ipynb)** (limita o processo a
4 GB e depois verifica que um modelo com streaming é idêntico bit a bit a um modelo normal)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="Pré-verificação do soup train para o Llama-3.1-8B em uma placa de 4 GB: um armazenamento de base de 3.60 GB fixado na RAM ao longo de 32 camadas e dois buffers de VRAM de 113 MB, depois um pico medido de 3.32 GB a 119.6 tok/s, ficando abaixo da linha de 4 GB (medido na v0.72.2, antes do reparo #331; nova medição pendente na issue #361)"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, batch 1, seq 512 em uma RTX 3050 Laptop de 4 GB — <b>pico de 3.32 GB, 119.6 tok/s</b> (medido na v0.72.2, antes do reparo #331; nova medição pendente na issue #361). <a href="https://youtu.be/T1LCErE943E">Vídeo completo (90 s)</a></sub>
</p>

## Por que usar o Soup?

Treinar LLMs ainda é doloroso. Mesmo equipes experientes gastam 30-50% do tempo lutando contra a
infraestrutura em vez de melhorar os modelos. O Soup resolve isso.

- **Zero SSH.** Nunca mais se conecte por SSH a uma máquina com GPU quebrada.
- **Uma configuração.** Basta um arquivo YAML simples.
- **Tudo automático.** Tamanho de batch, detecção de GPU, quantização — tudo resolvido.
- **Funciona localmente.** Treine na sua própria GPU com QLoRA. Sem precisar de nuvem.

## Novidades

**v0.75.0 — o mesmo `soup.yaml` treinava uma receita diferente no MLX e no transformers,
em silêncio.** Seis opções de treinamento eram validadas, documentadas, aceitas — e não eram lidas por
nada nesse backend. **Todos os 60 pull requests desta versão vieram de fora, não do
mantenedor**, de 22 pessoas.

- **Mudança incompatível: uma chave de configuração desconhecida agora recusa o carregamento.**
  A v0.74 emitia um aviso e apontava esta versão como o prazo final. Um erro de digitação como
  `quantizaton`, ou uma chave que só existe em um Soup mais novo, era descartado e a execução seguia
  sem que a configuração fosse aplicada; agora o carregamento falha na CLI (código de saída 1) e na
  API (`ValueError`), indicando o campo que você provavelmente quis dizer. O detector aplica o
  remapeamento de `lora:` no nível raiz que o schema respeita desde a v0.40.1, de modo que essa grafia
  é aceita, não recusada; os dois arquivos de `soup fetch examples` que a usavam passaram para o
  `training.lora` canônico. Todas as receitas e templates carregam sem avisos, os nomes das chaves
  são escapados antes de chegar ao terminal e a varredura é limitada.
- **O MLX respeita a configuração que aceitou.** `train_on_responses_only`, `warmup_ratio` /
  `scheduler` / `weight_decay` / `optimizer`, `max_grad_norm`, `gradient_accumulation_steps`
  e `gradient_checkpointing` eram validados e depois descartados em `backend: mlx`. Apenas
  8 dos 32 nomes de otimizadores têm equivalente no MLX; os outros 24 são recusados pelo nome,
  em vez de virarem AdamW em silêncio. O MLX também alimenta o painel ao vivo, o tracker e
  o `soup ui`, e o `soup doctor --config` lista as configurações que um backend não lê.
- **A loss de validação não ficava registrada em lugar nenhum.** Era calculada em todos os backends e
  descartada: sem coluna de métricas, sem campo de evento, nada no painel. Agora ela é registrada,
  transmitida e exibida.
- **Mudança incompatível: `grpo_variant: gspo` agora é o objetivo em nível de sequência publicado**
  (arXiv:2507.18071), substituindo uma heurística de centralização por coluna em que um token de
  padding também deslocava o gradiente de todas as linhas que compartilhavam sua coluna. As
  configurações gspo existentes não reproduzirão execuções anteriores.
- **Os endpoints de leitura da Web UI e o SSE exigem autenticação**, com tickets de uso único e curta
  duração em vez de um token na query string; `--public` não serve mais `/docs` e
  `/openapi.json` para a rede local; e um subprocesso de treinamento não trava mais quando nada
  lê a sua saída.
- **`torch>=2.6.0`** resolve a limitação conhecida da v0.74.0: no 2.5.1, o `trl>=0.29` não
  conseguia ser importado e todos os trainers de preferência estavam inoperantes. Também foram
  corrigidos: `training.loraplus_lr_ratio` derrubava todas as execuções que o definiam, e
  `packing: true` gerava erro no TRL 0.29.

> Somente Python **3.10–3.12**. No 3.13+, o pip costumava resolver wheels do PyTorch não testados,
> que quebram na extensão nativa antes mesmo de o Soup rodar.

Os destaques de versões mais antigas estão na página de [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

## Início Rápido

### 1. Instalação

O Soup é um aplicativo de linha de comando, então a instalação mais limpa é dar a ele o seu próprio
ambiente e colocar o `soup` no seu `PATH`:

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

Já está dentro de um virtualenv, de um notebook do Colab ou de uma imagem Docker? Use o `pip`
diretamente, com os mesmos nomes e extras:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

Use o `pip` em vez do `pipx` se você também quiser fazer `import soup_cli` no seu próprio
código, já que o pipx isola o aplicativo de todo o resto de propósito.

A tabela completa de extras (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) está em
[`docs/models.md`](docs/models.md#optional-extras).

> **Recebeu `error: externally-managed-environment`?** Isso é o
> [PEP 668](https://peps.python.org/pep-0668/), não um problema do Soup. No Debian 12 e no
> Ubuntu 23.04 em diante, o `pip` não pode gravar no Python do sistema, porque o
> `apt` também gerencia esses arquivos. O `pipx` e o `uv tool` contornam isso dando ao Soup
> o seu próprio ambiente, e é por isso que aparecem primeiro acima. Rodar `python3 -m venv
> .venv && source .venv/bin/activate` e depois o `pip` comum funciona igualmente bem.

> **Aspas duplas, não simples.** `"soup-cli[train]"` é a única grafia que funciona em todos os
> shells — `cmd.exe`, PowerShell, bash e zsh. Se você copiou `'soup-cli[train]'` de um tutorial
> mais antigo e o pip o rejeitou, é por esse motivo:
> [por quê, e o erro exato](docs/models.md#quoting-the-extra).

O `soup init`, o `soup data …` e os demais comandos de dados/inspeção funcionam na instalação leve.
O fine-tuning (`soup train`) precisa do extra `[train]`.

### 2. Crie uma configuração

```bash
soup init                       # interactive wizard
soup init --template chat       # or start from a template
```

Templates: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Treine, teste, entregue

```bash
soup train --config soup.yaml                 # LoRA, quantization, batching — all handled
soup chat  --model ./output                    # talk to your model
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # merge LoRA into the base
soup export --model ./output --format gguf --quant q4_k_m   # GGUF for Ollama / llama.cpp
```

Mais destinos de exportação (ONNX, TensorRT, AWQ, GPTQ, BitNet) e opções de deploy estão em
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Interface Web

Prefere o navegador? O `soup ui` sobe um painel local para experimentos,
configuração de treinamento, métricas ao vivo, exploração de datasets e chat com o modelo.

```bash
pip install "soup-cli[ui]"
soup ui
# Opens http://127.0.0.1:7860
```

![Soup Web UI — New Training](docs/assets/web-ui-new-training.png)

[Documentação da interface web](docs/serving-and-export.md#web-ui)

## Configuração

Um `soup.yaml` completo:

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

O `config/schema.py` é a fonte única da verdade para cada campo. As opções avançadas de dados, treinamento
e PEFT estão documentadas em [Documentação](#documentação).

> **Chaves de configuração desconhecidas são rejeitadas desde a v0.75.** Uma chave que nenhum modelo do schema
> declara — um erro de digitação como `quantizaton`, ou um campo que só existe em um Soup mais novo —
> antes passava na validação e era descartada, e a execução seguia com a configuração simplesmente
> não aplicada. A v0.74 a reportava no carregamento, com o campo que você provavelmente quis dizer; a
> partir da **v0.75**, a mesma configuração falha ao carregar, então corrija ou remova a chave em vez
> de contar com o fato de ela ser ignorada. Veja [Chaves de configuração desconhecidas](docs/backends-and-ops.md#unknown-config-keys).

## Documentação

A referência completa de recursos está em [`docs/`](docs/). Comece por aqui:

| Guia | Conteúdo |
|---|---|
| [Tarefas e métodos de treinamento](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pré-treinamento, destilação, classificação, visão/áudio/TTS, unlearning, RAFT/RA-DIT, detectores de proteção contra loops de geração |
| [PEFT, contexto longo e eficiência](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, catálogo de otimizadores e métodos PEFT, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, curriculum, ajuste automático |
| [Desempenho e quantização](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, formatos de salvamento, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Engenharia de dados](docs/data.md) | Formatos, o pipeline com paridade em relação ao Axolotl/LF, ferramentas de dados, geração sintética e forge, scorecards de qualidade, ferramentas de traces, datasets remotos, mistura, DAGs de receitas |
| [Avaliação e probes](docs/evaluation.md) | Design/gate de avaliação, treinamento com gate de avaliação, benchmarks, métricas de NLG, calibração, arena Elo, diagnóstico, probes de raio-X pós-treinamento, A/B, drift, capacidade de ajuste (tunability), `soup advise` |
| [Serving e exportação](docs/serving-and-export.md) | Servidor compatível com a OpenAI, inferência em lote, benchmarking, merge/exportação, endpoint Anthropic Messages, decodificação especulativa (treine e meça seu próprio draft), piloto automático de deploy, Web UI, Agent Forge |
| [Adapters, registry e governança](docs/adapters-and-governance.md) | Ciclo de vida/gerenciamento de adapters, registry de modelos, Soup Cans, o data flywheel (`soup loop`), edição de conhecimento, steering, controles da cadeia de suprimentos (scan/sign/BOM/attest/audit/airgap) |
| [Início rápido de conformidade e governança](docs/compliance.md) | Templates de `init` para HIPAA/SOC2/EU-AI-Act/SR-11-7, proveniência (BOM/attest/repro-receipt), log de auditoria, air-gap, geração automática de model card (`soup card`), gate de CI (`soup ci init`) |
| [Backends, plataforma e operações](docs/backends-and-ops.md) | Backends MLX/Unsloth, hubs alternativos, integração com o HF Hub, piloto automático, rastreamento de experimentos, plan/apply, lockfiles de ambiente, adequação ao hardware, completions, plugins, comandos utilitários |
| [Referência de comandos](docs/commands.md) | A lista completa de comandos do `soup` |
| [Modelos compatíveis e extras](docs/models.md) | Famílias de modelos recomendadas, o guia de tamanho por VRAM, a matriz de extras do pip |

## Formatos de Dados

Alpaca, ShareGPT, ChatML, pares de preferência (DPO / ORPO / SimPO / IPO / KTO), visão, áudio,
ASR, texto puro, embedding, RAFT e mais — tudo detectado automaticamente a partir de JSONL, JSON, CSV, Parquet ou
TXT, de modo que, na maioria dos casos, você aponta `data.train` para um arquivo e nada mais muda. Os
schemas, com um exemplo prático para cada formato, e o pipeline de dados (URIs remotas, streaming, sharding,
interleaving, expansão de vocabulário, ingestão de documentos) estão em
[`docs/data.md`](docs/data.md#data-formats).

## Comandos Comuns

```bash
soup train  --config soup.yaml        # train (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl   # batch inference
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

A lista completa de comandos está em [`docs/commands.md`](docs/commands.md).

## Modelos Compatíveis

O Soup funciona com **qualquer** modelo de geração de texto no
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) — se ele carrega com
`AutoModelForCausalLM`, funciona, sem nenhuma alteração de configuração. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4 e mais de 100 outros já vêm incluídos como receitas prontas (`soup recipes list`).

| VRAM | Modelo máximo (QLoRA 4-bit) | Exemplo |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (completo) ou MoE | Mixtral-8x22B, DeepSeek-V3 |

As tabelas completas de modelos + visão e a matriz de extras opcionais estão em [`docs/models.md`](docs/models.md).

## Docker

Rode o Soup sem instalar CUDA ou PyTorch localmente (imagem publicada no GHCR a cada release):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # or build locally
```

## Requisitos

- Python 3.10, 3.11 ou 3.12 (essas são as versões que o CI testa; o 3.13+ ainda não é compatível
  porque a stack do PyTorch não foi validada nele)
- GPU com CUDA (recomendado), Apple Silicon (MPS) ou CPU (experimental — muito lento)
- 8 GB+ de VRAM para modelos de 7B com QLoRA

Todas as tarefas de treinamento rodam em CPU para fins de teste (a quantização é desativada automaticamente). Os extras
opcionais (`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) estão listados em
[`docs/models.md`](docs/models.md#optional-extras).

## Solução de Problemas

```bash
soup doctor    # GPU, system resources, dependencies, and version in one place
```

Wheels de CUDA, incompatibilidades de versão: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Desenvolvimento

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

Veja o [CONTRIBUTING.md](CONTRIBUTING.md) para o fluxo de trabalho completo e o [SECURITY.md](SECURITY.md) para
reportar uma vulnerabilidade. A telemetria é estritamente opt-in (`SOUP_TELEMETRY=1`, desativada por padrão; veja a [Política de Privacidade](docs/backends-and-ops.md#privacy-policy)).

## Apoie o Soup

O Soup é Apache-2.0 e gratuito — e continuará assim. Ele é construído e mantido abertamente em
um único notebook de 4 GB, e é por isso que todo número de desempenho nesta documentação é medido, e não
apenas afirmado.

Se o Soup poupou uma execução de treinamento para você, [dar uma estrela no repositório](https://github.com/MakazhanAlpamys/Soup)
é o que mais ajuda, e não custa nada. Se você quiser financiar o trabalho diretamente:

**[❤️ Doe](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — uma vez só, em qualquer valor (use
*Change amount* na página de checkout). Os pagamentos são processados pela Stripe em nome da empresa
registrada do mantenedor, a **MePlay, Inc.** — é esse nome, e não "Soup", que aparece na página de
checkout e na fatura do seu cartão.

As doações compram tempo de GPU para o trabalho que depende de hardware — multi-GPU, validação em 8B+, Apple Silicon —
que um único notebook de 4 GB não consegue alcançar.

A outra forma de destravar exatamente esses itens é o **próprio hardware**. Eles são lançados com
avisos honestos de "requer \<hardware\>", em vez de afirmações não verificadas; então, se você tem acesso a uma
máquina maior — ou a créditos de GPU sem uso —, executar o que uma das issues
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
pede e publicar os números ajuda tanto quanto financiar o tempo de GPU. Essas issues dizem
exatamente o que hoje está bloqueado por falta de hardware.

## Colaboradores

Construído pela comunidade ❤️ — obrigado a todos que contribuíram. Veja
[CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Colaboradores](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## Contato

Bugs e sugestões de funcionalidades devem ir para o
[rastreador de issues](https://github.com/MakazhanAlpamys/Soup/issues), e dúvidas para as
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions) — em ambos os casos a resposta vem mais rápido
e ajuda a próxima pessoa com o mesmo problema.

Para bate-papo ao vivo, ajuda com a instalação e tudo o que funciona melhor como conversa, entre no
[Discord](https://discord.gg/dgd2pJcjwP) ou na [comunidade do Telegram](https://t.me/souptasters).
Tudo o que ainda deva ser encontrável daqui a seis meses
pertence às Issues ou às Discussions — uma resposta no Discord ajuda uma pessoa, uma issue ajuda todos
os que esbarrarem na mesma coisa. O [Código de Conduta](CODE_OF_CONDUCT.md) vale lá também.

Para o que não couber em um canal público — relatos de segurança (veja o [SECURITY.md](SECURITY.md)),
questões do Código de Conduta ou imprensa — escreva para **team@trysoup.dev**. Esse é o endereço do projeto
e o correto para qualquer assunto relacionado ao Soup. **makazanalpamys@gmail.com** é o endereço
pessoal do mantenedor; ele chega à mesma pessoa e serve bem como alternativa.

## Como citar o Soup

O layer streaming — treinar um modelo de 8B em uma GPU de notebook de 4 GB transmitindo a base congelada da
RAM do host, uma camada do decoder por vez — é descrito em um preprint, junto com o protocolo de
correção que verifica uma execução com streaming contra uma residente (forward e backward declarados
separadamente, porque são duas afirmações e não uma).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**A versão 3 (13 de agosto de 2026) é a atual.** O título e a afirmação continuam os mesmos — 8B em 4 GB —
e nenhum número medido mudou desde a v1. O que a v3 faz é **retirar uma explicação que
havíamos publicado**, que também é a forma mais curta de descrever para que serve o artigo:

- **Retirada na v3: "o layer streaming é limitado pela transferência de host para dispositivo, e não pela GPU."**
  Isso era uma *inferência* a partir da replicação em H100 abaixo, e nunca havia sido medido. Medimos
  em 11 de agosto e é falso na configuração publicada: eliminar todos os bytes de host para dispositivo
  rende **1.4%**, o stream de computação espera por uma cópia durante **0.20%** do
  passo, e o passo roda a **71.3%** do teto de GEMM da mesma sessão daquela placa. O maior
  custo específico do streaming é a dequantização NF4 por camada, com 9.8%
  ([o registro](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). Todas as medições continuam
  válidas; a replicação sobrevive em uma forma mais fraca — a restrição é comum às duas máquinas e
  não é a capacidade de computação da GPU.
- **Replicação em um hardware completamente diferente do original** (adicionada na v2): 119.6 tok/s na RTX
  3050 contra uma mediana de 113.00 em uma H100, no mesmo pico de 3.32 GB. Ambas são anteriores ao reparo
  #331; a nova medição em 4 GB está pendente na issue #361.
- **Um defeito silencioso que gerava gradientes errados, encontrado e corrigido.** Em NF4 acima de ~165 MiB por camada, o
  forward continuava idêntico bit a bit e a curva de loss parecia saudável, enquanto os gradientes estavam errados. A
  causa está identificada na biblioteca upstream e foi reportada lá; a correção só é liberada após passar por controles
  em modelos reais de 32B e 72B.
- **Igualdade bit a bit em tamanhos reais de modelo**, em vez de modelos de brinquedo de três camadas: forward de 0.5B a 72B,
  backward em 8B e 14B.
- **Qualidade do modelo treinado, medida pela primeira vez**, e indistinguível de uma execução residente.
- **Uma comparação com o DeepSpeed** — incluindo o resultado que não nos favorece: oito placas
  de ZeRO-3 são mais lentas do que uma placa treinando de forma residente.
- **A seção de limitações reescrita**: dos dez itens da v1, um foi resolvido e outros quatro tiveram o escopo
  reduzido, e sete novos foram adicionados.

Cite a versão que você usou. `10.5281/zenodo.21771064` é o DOI de conceito e sempre aponta para
a versão mais recente (a v3 hoje); a v1 e a v2 continuam citáveis nos seus próprios DOIs de versão e não são
editadas — a retratação acima é uma nova versão justamente para que o registro do que afirmamos,
e de quando, permaneça intacto.

Os registros de medição por trás de cada número do artigo estão em [`benchmarks/`](benchmarks/), publicados
como foram escritos — incluindo as falhas, as suposições que se mostraram erradas e os números que
foram medidos e depois descartados.

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

## Licença

[Apache-2.0](LICENSE). Copyright © Colaboradores do Soup.
