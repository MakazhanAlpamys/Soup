# Ingeniería de datos

> 🇪🇸 Traducción al español de [`data.md`](data.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> Formatos de datos, el pipeline con paridad Axolotl/LF, herramientas de datos, generación/forja sintética, tarjetas de puntuación de calidad, herramientas de trazas, datasets remotos, mezcla, DAG de recetas y las superficies de ingeniería de datos de la v0.69.

**Contenido:**

- [Ingeniería de datos Pro](#ingeniería-de-datos-pro)
- [Ecosistema de trazas de producción (`soup ingest`)](#ecosistema-de-trazas-de-producción-soup-ingest)
- [Minería de prompts (`soup prune-prompt`)](#minería-de-prompts-soup-prune-prompt)
- [Muestreador de aprendizaje activo (`soup data active-sample`)](#muestreador-de-aprendizaje-activo-soup-data-active-sample)
- [Generación de datos sintéticos](#generación-de-datos-sintéticos)
- [Aumento de datos](#aumento-de-datos)
- [Trace-to-Preference](#trace-to-preference)
- [Migración de configuración](#migración-de-configuración)
- [Formatos de datos](#formatos-de-datos)
- [Pipeline de datos Pro](#pipeline-de-datos-pro)
- [Herramientas de datos](#herramientas-de-datos)
- [Datasets de demostración (`soup data demo`)](#datasets-de-demostración-soup-data-demo)
- [Trace-to-Preference: filtro LLM-Judge](#trace-to-preference-filtro-llm-judge)
- [Forja de datos sintéticos](#forja-de-datos-sintéticos)
- [Tarjeta de puntuación de calidad de datos](#tarjeta-de-puntuación-de-calidad-de-datos)
- [Datasets remotos (S3 / GCS / Azure / OCI)](#datasets-remotos-s3--gcs--azure--oci)
- [Deduplicación semántica (`soup data dedup --semantic`)](#deduplicación-semántica-soup-data-dedup---semantic)
- [Saneamiento y reparación de datasets (`soup data clean`)](#saneamiento-y-reparación-de-datasets-soup-data-clean)
- [Mapa de temas (`soup data topics`)](#mapa-de-temas-soup-data-topics)
- [Canarios (`soup data canary insert|check`)](#canarios-soup-data-canary-insertcheck)
- [DAG de recetas de datos](#dag-de-recetas-de-datos)
- [Optimizador de mezcla de datos (BETA)](#optimizador-de-mezcla-de-datos-beta)
- [Tokenización AOT con `soup data preprocess`](#tokenización-aot-con-soup-data-preprocess)
- [Ejecutor de DAG de recetas de datos (`soup data recipe --execute`)](#ejecutor-de-dag-de-recetas-de-datos-soup-data-recipe---execute)

---

## Deduplicación semántica (`soup data dedup --semantic`)

`soup data dedup` elimina los casi duplicados con MinHash por defecto — rápido, sin
torch, pero **léxico**: compara shingles de tokens compartidos, así que dos filas que dicen
lo mismo con otras palabras le parecen no relacionadas.

`--semantic` compara la similitud coseno de los embeddings:

```bash
soup data dedup train.jsonl --semantic -o clean.jsonl
soup data dedup train.jsonl --semantic --threshold 0.85 --field text -o clean.jsonl
soup data dedup train.jsonl --semantic --embed-model sentence-transformers/all-mpnet-base-v2
```

Requiere el extra `[train]` (reutiliza `transformers`; no hay una
dependencia nueva) y descarga un modelo de embedding pequeño en el primer uso. El `dedup` con MinHash simple
se queda en el núcleo ligero.

**Qué te aporta.** Medido contra MinHash sobre las mismas filas
(all-MiniLM-L6-v2):

| par | coseno | MinHash | `--semantic` |
|---|---|---|---|
| duplicado exacto | 1.000 | detectado | detectado |
| "ordena **una lista** de enteros" / "ordena **un arreglo** de enteros" | 0.908 | **omitido** | detectado |
| "que ordena una lista de **ints**" (reformulado) | 0.880 | **omitido** | detectado |
| "Suma dos números" / "Multiplica dos números" | 0.759 | conservado | conservado (correcto) |

Así, `--semantic` detecta **reformulaciones** que el shingling de MinHash puntúa como distintas.

### No es un detector de paráfrasis — y por qué el valor por defecto es 0.8

Las paráfrasis más marcadas **no** se pueden separar de forma confiable. Medido, los cosenos de paráfrasis
(0.49–0.76) *se solapan* con filas realmente distintas (0.54–0.76):

- "invierte una cadena" / "invierte el orden de los caracteres" — una **paráfrasis real** — puntúa **0.491**
- "Suma dos números" / "Multiplica dos números" — **dos filas que debes conservar** — puntúa **0.759**

Una paráfrasis real puede puntuar *más bajo* que dos filas que deben sobrevivir ambas, así que **ningún
umbral las separa limpiamente**. Bajar `--threshold` para perseguir el recall de paráfrasis
elimina filas de entrenamiento reales — pérdida silenciosa de datos, que es peor que conservar
un duplicado. El 0.8 por defecto es deliberadamente conservador. Súbelo o bájalo solo
contra tus propios datos, y revisa qué se descartó.

`--threshold` significa Jaccard para MinHash y coseno para `--semantic`. Son
escalas distintas; un valor ajustado para una no es significativo para la otra.

## Saneamiento y reparación de datasets (`soup data clean`)

`soup data clean` aplica reglas deterministas de higiene para reparar datasets de ajuste fino corruptos, mal formados o ruidosos sin modificar nunca el archivo de entrada en su lugar:

```bash
# Clean dataset with safe non-destructive defaults -> writes to <input>_cleaned.jsonl
soup data clean raw_data.jsonl

# Specify custom output path
soup data clean raw_data.jsonl -o clean_data.jsonl

# Preview modifications and statistics without writing any files
soup data clean raw_data.jsonl --dry-run

# Output machine-readable JSON for CI/CD pipelines
soup data clean raw_data.jsonl --json

# Enable optional heuristic repairs (AI disclaimers, code fences, tool-call JSON, echo pruning)
soup data clean raw_data.jsonl --strip-boilerplate --repair-code --repair-json --prune-echo
```

### Reglas de limpieza y valores por defecto:
- **Por defecto (seguro y no destructivo):**
  1. **Caracteres de control y espacios en blanco:** elimina los controles C0 (`\x00-\x1f`), los espacios de ancho cero (`\u200b-\u200d`, `\ufeff`) y normaliza CRLF/CR a LF de Unix.
  2. **Turnos vacíos y degenerados:** descarta las filas cuyo turno del asistente esté vacío o sea más corto que `--min-tokens`. Un turno que lleva tool calls queda exento: su carga útil son las llamadas, y `"content": ""` es lo que escribe el formato tool-calling en un turno que solo contiene llamadas.
- **Reparaciones heurísticas de activación opcional (banderas):**
  1. `--strip-boilerplate`: elimina preámbulos prefabricados (*"¡Claro! Como modelo de lenguaje de IA..."*) y despedidas (*"¡Espero que esto ayude!"*) en varias pasadas.
  2. `--repair-code`: cierra automáticamente las vallas de código de triple backtick (```` ``` ````) sin cerrar en las respuestas del asistente.
  3. `--repair-json`: desenvuelve bloques de código markdown de los argumentos JSON y repara las comas finales en las tool calls.
  4. `--prune-echo`: descarta las filas donde el asistente solo repite literalmente el prompt del usuario.
  5. `--drop-invalid-json`: descarta las filas con una tool call cuyos argumentos todavía no se pueden parsear como JSON (después de `--repair-json`, cuando ambas están activadas).

  `--repair-json` y `--drop-invalid-json` leen todas las llamadas que lee el cargador de tool-calling: el `tool_calls` de nivel superior de la fila y el `tool_calls` de cada turno del asistente, en la forma documentada `{"function": {"name": ..., "arguments": ...}}` o plana. Los argumentos dados como objeto JSON se dejan tal cual.

Admite todos los formatos estándar: `chatml`, `alpaca`, `sharegpt`, `dpo`, `kto` y `tool-calling`.

## Mapa de temas (`soup data topics`)

Mira qué estás entrenando en realidad:

```bash
soup data topics train.jsonl                       # 'auto' picks the cluster count
soup data topics train.jsonl --clusters 8 -o topics.json
```

Genera el embedding de cada fila, agrupa con k-means y etiqueta cada cluster con términos
c-TF-IDF — términos frecuentes en *ese* cluster y raros en el resto, así que palabras de relleno como
"the" nunca se convierten en etiqueta. Imprime una tabla de cobertura más una advertencia para cualquier tema
por debajo del 2% de los datos:

```
        Topic map — 4200 rows, 6 clusters
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━┓
┃ Topic                    ┃ Rows ┃ Coverage ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━┩
│ function / python / code │ 3444 │    82.0% │
│ theorem / proof / math   │  252 │     6.0% │
│ refuse / harmful / safe  │   42 │     1.0% │
└──────────────────────────┴──────┴──────────┘
topic 'refuse / harmful / safe' is thin: 1.0% of rows (42/4200)
```

Las etiquetas son **clusters de términos emergentes**, no una clasificación contra una
taxonomía fija: "82% code" significa que el 82% de las filas cayó en un cluster cuyos términos principales
parecen código. Requiere `[train]`.

## Canarios (`soup data canary insert|check`)

Demuestra si un modelo memorizó tus datos — para detección de fugas y procedencia.

```bash
# 1. insert K unique secrets (keep the manifest OUT of your repo)
soup data canary insert train.jsonl -o canaried.jsonl --count 16 --manifest secrets.json

# 2. train on canaried.jsonl as usual, then:
soup data canary check --manifest secrets.json --base ./my-model --adapter ./lora
```

`insert` escribe los canarios en el propio formato del dataset, así que el cargador los conserva:
`--format auto` (el valor por defecto) lo detecta a partir de las primeras 100 filas, igual que lo hace
`data.format: auto` (un archivo que mezcla formas se rechaza, consulta Formatos de datos). Se admiten Alpaca, sharegpt y chatml, cada uno con el carrier como prompt
y el secreto como respuesta entrenada. Cualquier otro formato se rechaza: dpo, kto y
embedding no tienen una única respuesta supervisada, plaintext entrena con texto sin procesar en lugar
del turno de chat que puntúa `check`, tool-calling coloca un turno de sistema con el esquema de herramientas
antes del prompt, que `check` no renderiza, y los formatos multimodales necesitan un
archivo real de imagen o audio por fila. `-o` acepta `.jsonl`, o `.json` para un arreglo JSON.

`insert` distribuye los canarios por el archivo en lugar de añadirlos al final: el archivo se
corta en un tramo igual por canario, y cada canario va a una fila aleatoria de su propio
tramo (`--seed` fija las filas). Las filas propias del dataset conservan su orden. El
cargador reserva las últimas filas del archivo como validación (`data.val_split`, 0.1 por defecto),
y un canario ahí nunca se entrena, así que añadirlos al final dejaba a todos los canarios fuera de alcance
a partir de 135 filas. Distribuidos, una cola reservada de
`data.val_split` contiene más o menos esa proporción de ellos: `insert` imprime cuántos de ellos
entrena el split por defecto, y el manifest registra la fila de cada canario en el archivo escrito
(`"row"`, contada desde 0) y el número de filas del archivo (`"rows"`). Define `data.val_split: 0`
para una corrida en la que todos los canarios deban entrenarse.

`check` mide la loss del modelo en cada secreto insertado y la clasifica frente a
**controles** nunca insertados, tomados del mismo espacio de secretos y que comparten el mismo
prompt carrier — así que una loss baja significa que el *secreto* es inusualmente probable, no el
prompt. Sale con **2** en MAJOR, para que la CI pueda bloquear ante una fuga.

Esto es loss frente a controles (Carlini et al., *The Secret Sharer*), no "pregúntale al modelo
y mira si dice el secreto": un modelo puede memorizar un canario y aun así no emitirlo
con decodificación greedy, así que "no volvió nada" sería una falsa tranquilidad.

El veredicto pregunta si **más** canarios parecen memorizados de lo que explica el azar
(cola binomial, α=0.05) en lugar de si alguno bajó mucho — con 16
canarios, una regla de "cualquiera de ellos" se dispara en un modelo **limpio** cerca del 15% de las
veces, lo que haría inútil un bloqueo de CI.

Medido en SmolLM2-135M:

| modelo | loss | percentiles | veredicto |
|---|---|---|---|
| entrenado con los canarios | 1.7–2.5 | todos 0.0% | **MAJOR** (exit 2) |
| nunca los vio | 4.1–6.2 | 1.6%–93% | OK (exit 0) |

**El manifest es el artefacto sensible**, no el dataset: cualquiera que lo tenga puede
reproducir los secretos. Se escribe con permisos `0600` en POSIX y no debe incluirse en un commit
junto a los datos que protege. `check --output` incluye los mismos secretos.

La exposición es una **aproximación por controles muestreados**, no una enumeración de rangos en todo el espacio —
"sin exposición" no es prueba de que no haya memorización.

---

## Ingeniería de datos Pro

La versión v0.69.0 incluye 5 superficies que convierten la preparación de datasets de "tirar un JSONL al trainer" en un flujo de trabajo de ingeniería de primera clase.

```bash
# dbt-for-SFT — DAG of dataset transforms with incremental materialization
cat > build.yaml << 'EOF'
models:
  - {name: raw, kind: incremental, source: data/raw.jsonl, transform: identity}
  - {name: filtered, kind: incremental, refs: [raw], transform: filter_low_quality}
  - {name: tokenized, kind: incremental, refs: [filtered], transform: tokenize}
EOF
soup build build.yaml --dry-run                 # validate topology + plan
soup build build.yaml --output-dir built/       # live materialise (v0.71.6)

# Expectations suite — Great Expectations for chat data
cat > suite.yaml << 'EOF'
expectations:
  - {name: expect_no_pii}
  - {name: expect_token_length_between, args: {min_tokens: 16, max_tokens: 4096}}
  - {name: expect_no_refusal_pattern}
EOF
soup expect data.jsonl suite.yaml   # exit 2 on suite failure
# Inspects ChatML, ShareGPT, DPO (chosen/rejected), KTO (completion), Alpaca,
# and sentence embedding formats. Fails closed if any row yields zero extractable text.
# Malformed or non-object lines and files with no checkable rows are refused with exit 3.

# Magpie synthetic data — chat-template-prefix harvest (live, v0.71.6)
soup data gen-magpie --base meta-llama/Llama-3.1-8B-Instruct \
    --provider ollama --target 1000 --output magpie.jsonl --quality-filter

# Persona-Hub diversity — prompt × persona × style matrix sampling
soup data persona-mix --prompts prompts.jsonl --n 500 --output mixed.jsonl

# Brain-rot detector (arXiv 2510.13928) — refuses to train on excessive slop
soup data brain-rot data.jsonl --strict --max-major-fraction 0.10

# Best-of-N rejection sampling — local sampling stays the default
soup data best-of-n --base HuggingFaceTB/SmolLM2-135M-Instruct \
    --prompts prompts.jsonl --n 8 --judge ollama://llama3.1 \
    -o best_of_n.jsonl --emit-pairs pairs.jsonl

# Or draw the N candidates from a running Ollama / vLLM raw-completion endpoint
soup data best-of-n --provider ollama --model qwen2.5:7b \
    --base-url http://localhost:11434 --prompts prompts.jsonl --n 8 \
    --judge ollama://llama3.1 -o best_of_n.jsonl

# Every non-blank prompt row is validated; accepted SFT rows record source_line
# in their _best_of_n provenance so input/output completeness can be checked.

# If sampling or judging stops, continue from the last fsynced prompt group.
soup data best-of-n --provider ollama --model qwen2.5:7b \
    --prompts prompts.jsonl --n 8 --judge ollama://llama3.1 \
    -o best_of_n.jsonl --resume

# Two-phase / air-gapped workflow: sample first, without constructing a judge.
soup data best-of-n --base Qwen/Qwen3.8-27B --revision <commit> \
    --prompts prompts.jsonl --n 8 --export-candidates candidates.jsonl

# An offline human, deterministic program, CI job, or Codex writes one judgment
# per candidate group, copying prompt_id and group_digest from candidates.jsonl:
# {"prompt_id":"...","group_digest":"...","winner_idx":2,
#  "scores":[0.1,0.4,0.9,...],"verifier":{"name":"Codex","version":"offline-v1"}}
soup data best-of-n --candidate-artifact candidates.jsonl \
    --judgments judgments.jsonl -o best_of_n.jsonl --emit-pairs pairs.jsonl

# Evol-Instruct (WizardLM depth/breadth, v0.71.31) — grow instruction diversity
soup data evolve --input seeds.jsonl --provider ollama --model llama3.1 \
    --strategy depth --rounds 2 -o evolved.jsonl
```

Every command applies the project-wide TOCTOU policy (`os.lstat + S_ISLNK` symlink rejection before any open) and cwd containment via the shared `paths.enforce_under_cwd_and_no_symlink` helper. All five are LIVE: `soup build` materialises with five built-in transforms (`identity` / `drop_empty` / `lowercase` / `strip` / `dedup_exact`) and SQLite-tracked incremental re-transform (v0.71.6); `soup data gen-magpie` and provider-backed `best-of-n` harvest via raw completion against `--provider ollama|vllm` (SSRF-validated; `anthropic` rejected because the Messages API has no raw-completion endpoint). Provider-backed `best-of-n` records the sampler provider and model in each row's `_best_of_n` provenance; omit `--provider` to retain the local Transformers `--base` path.

`best-of-n` fsyncs each completed prompt group to a private recovery journal
(`<output>.checkpoint.jsonl` by default). `--resume` reuses only a sequential
prefix whose prompt and run-configuration digest matches exactly, so completed
prompts are not sampled or judged twice. Before final publication, Soup snapshots
any prior SFT, DPO, and manifest targets. Each file remains an atomic replacement,
the manifest is written last, and any failed replacement restores the complete old
generation or removes the newly created set. The manifest binds the exact SHA-256
hashes and row counts as one generation. Keep or archive the checkpoint after
success if reproducible rematerialization is useful; it contains dataset content
and should be protected like the generated dataset.

The candidate artifact preserves every ordered candidate with prompt, candidate,
group, and whole-artifact SHA-256 bindings plus a public sampler specification.
The offline phase validates complete one-to-one coverage before writing anything:
missing or duplicate prompt ids, changed group digests, invalid winner indexes,
score-count drift, non-finite scores, and winner/score disagreement all fail
closed. It does not construct a sampler or judge. SFT and DPO rows retain the
candidate-artifact and judgment-file digests, group id, public sampler settings,
and bounded verifier identity. Endpoint URLs and local model paths are never
copied into those artifacts. Reusing the same two input files produces identical
training-row bytes. The offline command writes `<output>.manifest.json` last (or
the explicit `--manifest` path) and binds the exact SFT/DPO hashes, row counts,
candidate artifact, judgment file, and whether DPO output was requested. Treat
the manifest as the commit marker: missing or mismatched manifests identify an
interrupted or replaced generation.

The candidate artifact and verified judgment file are the durable recovery
boundary for offline materialization. This phase performs no sampling, so it has
no progress checkpoint: `--resume` and `--checkpoint` are rejected. After any
late write failure, keep those two inputs and rerun the exact offline command.
Soup removes the previous manifest before replacing outputs and publishes the
new manifest last. A prior, manifest-authenticated DPO beside the manifest is
removed when the replacement run requests SFT only.

Consumers must verify the final manifest and open exactly the SFT/DPO files it
lists. They must never discover training inputs by globbing neighboring JSONL
files: an unlisted sidecar, including an older DPO stored elsewhere, is not part
of the committed generation.

Candidate export durably checkpoints each completed prompt group at
`<artifact>.checkpoint.jsonl`. If sampling stops, rerun the same command with
`--resume`; Soup authenticates the checkpoint against the prompts and sampler
before continuing at the first incomplete group. Candidate and judgment inputs
are validated through a temporary disk index, and final SFT/DPO files are staged
incrementally, so memory does not grow with the complete artifact size.
For a local model directory, the checkpoint binds the exact regular-file names,
sizes, and contents through a privacy-safe fingerprint; replacing weights at the
same path therefore invalidates resume before the model is loaded. Prompt source
lines and provider endpoints are bound as well without exposing private paths or
URLs. Streamed SFT/DPO replacements are committed as one rollback-protected set,
and an SFT-only replacement retires a prior manifest-bound DPO in that same
transaction.

### Transformaciones personalizadas

Use a dotted-path string (`module.path:function_name`) as the ``transform``
value to import a custom transform at build time:

```yaml
models:
  - {name: clean, kind: table, source: data/raw.jsonl, transform: my_pkg.transforms:clean_row}
  - {name: enriched, kind: table, refs: [clean], transform: my_pkg.transforms:enrich}
```

The target function must accept exactly two positional arguments (`row`, ``config``)
and return a ``dict`` or ``None``. Soup resolves the dotted path lazily (the module
is imported only when the build actually runs) and caches the result so repeated
references to the same path do not re-import.

**Trusted-input posture.** The dotted-path syntax causes Soup to import an
arbitrary Python module and call a function from it. Treat ``transform`` values
as *trusted input*: do not feed untrusted or operator-controlled YAML into ``soup
build`` on shared CI hosts. An attacker who controls the manifest can execute
arbitrary code during the build phase. If you must accept user-supplied manifests,
validate them against a allowlist of permitted transform paths before passing
them to the resolver.


## Ecosistema de trazas de producción (`soup ingest`)

Closing the data flywheel without leaving your existing observability stack. `soup ingest` parses JSONL exports from every major SaaS dashboard and emits a normalised trace stream that `soup data from-traces` (v0.26) consumes.

```bash
# Six supported sources — adapters for the major SaaS vendors + raw OTel
soup ingest --source langfuse     --logs ./langfuse-export.jsonl --output traces.jsonl
soup ingest --source langsmith    --logs ./langsmith-runs.jsonl
soup ingest --source helicone     --logs ./helicone-requests.jsonl
soup ingest --source openpipe     --logs ./openpipe-export.jsonl
soup ingest --source otel         --logs ./otel-spans.jsonl
soup ingest --source openai-stored --logs ./oai-stored-completions.jsonl
```

With `--logs` the CLI never makes a network call — operators export from their SaaS dashboard or vendor API, then point `soup ingest` at the local file. Auth env vars (`LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` / `LANGSMITH_API_KEY` / `HELICONE_API_KEY` / `OPENPIPE_API_KEY` / `OPENAI_API_KEY` / `OTEL_EXPORTER_OTLP_HEADERS`) are advisory on that path — Soup surfaces which ones authenticate the source. A PII reminder fires on every ingest run (matches v0.26.0 Trace-to-Preference policy).

### Live pull from Langfuse (`--pull`, #204)

Langfuse is the one source Soup can fetch directly, so there is no export step:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...   # Project Settings -> API Keys
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://us.cloud.langfuse.com   # optional: default is https://cloud.langfuse.com
soup ingest --source langfuse --pull --since 7d --output traces.jsonl
```

- **What one row is.** One output row per `GENERATION` observation in the window — the unit that carries a model, the exact input it was given and the output it produced — read from Langfuse's Observations API v2 (`/api/public/traces` is removed from Langfuse Cloud on 2026-11-16) and checked again on each observation, so a server that ignores the `type` filter cannot turn spans or tool calls into rows — they are counted as skipped in the summary. A row's `trace_id` is the observation id. The API returns plain-text input and output as-is but structured values (chat message lists, objects) as JSON inside a string; those are decoded, and a chat message list becomes a `prompt` of every message's content joined by newlines (system prompt included), the same flattening `parse_langfuse` applies to a `{"messages": [...]}` export. An agent trace therefore yields one row per LLM call it made; its spans and tool calls yield none. Generations with no input or no output are skipped and counted in the summary line, so a pull that matched nothing usable says so instead of writing an empty file silently.
- **Credentials.** Read from the environment only, never from a flag, so they never reach the audit log's argv. `LANGFUSE_BASE_URL` is honoured before `LANGFUSE_HOST`, the same precedence as the Langfuse SDK. The key pair is not written to the output, the console, debug logs or error messages.
- **Host checks.** HTTPS only. The host goes through the same SSRF validator as `--slack-url`; a private, link-local, shared (`100.64.0.0/10`), site-local (`fec0::/10`) or loopback address (self-hosted Langfuse) additionally needs `--allow-private-host`. Redirects are refused rather than followed with credentials attached.
- **Bounds.** `--since` accepts `30m` / `24h` / `7d` up to `365d` (default `7d`). Each request is bounded by a 30 s wall-clock deadline covering the connect and the whole response — a server that drip-feeds bytes cannot outlast it — and a response is capped at 64 MiB. Pages hold 100 generations; if results are still pending after `--max-pages` pages (default 100, max 10 000), the command stops with exit 1 and writes nothing — the output streams to a staging file, so an earlier file at `--output` is left untouched. HTTP 429 is retried up to 5 times, honouring `Retry-After` with a 60 s ceiling, and a pagination cursor the server repeats stops the pull instead of spending the rest of the page budget.
- **Without `--pull`** nothing changes: the pull code is not imported and no connection is opened.

The other sources have no live pull yet — export them and pass `--logs`.


## Minería de prompts (`soup prune-prompt`)

Production LLM apps often pin a multi-paragraph system prompt to every request. Fine-tuning with that prefix wastes tokens (the model learns to copy what's already in context). `soup prune-prompt` finds the longest character prefix shared by ≥ 95% of rows and strips it, so the FT model internalises the behaviour instead.

```bash
soup prune-prompt --input traces.jsonl --output pruned.jsonl --min-frequency 0.95
```

Binary-search over up-to-32 candidate templates finds the longest qualifying prefix (a longer threshold-meeting prefix may exist beyond the universal one — Soup does not early-exit on the 100% match). Two-pass file read with a 100 000-row DoS cap.

**Tokenizer-aware mode (v0.71.5).** Pass `--tokenizer <id-or-path>` (a HuggingFace repo id, a local path, or anything `AutoTokenizer.from_pretrained` accepts) to detect the shared prefix in *token* space and decode only the remaining ids:

```bash
soup prune-prompt --input traces.jsonl --output pruned.jsonl --tokenizer Qwen/Qwen2.5-0.5B
```

Char-level stripping can cut a BPE multi-byte sequence in half when the shared prefix ends mid-token; token-aware pruning finds the longest shared *token-id* prefix and decodes the remainder, so the boundary always lands on a real token. Per-row encoding is capped at 50 000 tokens. Omit `--tokenizer` to keep the original character-level behaviour.


## Muestreador de aprendizaje activo (`soup data active-sample`)

Surface the most uncertain prod traces for human review. Two modes via the input data shape:

- **Single RM:** `rm_score: 0.5` → uncertainty 1.0 (peak); `rm_score: 0.0` or `1.0` → uncertainty 0.0.
- **Dual RM:** `rm_scores: [s1, s2]` → uncertainty = `|s1 - s2|` (pairwise disagreement).

```bash
soup data active-sample --input traces.jsonl --output for-review.jsonl --budget 100
```

The output JSONL is a drop-in prompt set for `soup eval human` (v0.19). Budget is bounded `[1, 100 000]`.

**Webhooks (v0.71.5).** `soup ingest`, `soup prune-prompt`, `soup ab`, and `soup data active-sample` all accept `--slack-url` / `--discord-url` and POST a one-line summary on completion through the same SSRF-hardened validator as `soup drift-alarm` (scheme allowlist, loopback-only HTTP, RFC1918 / link-local / shared `100.64.0.0/10` / site-local `fec0::/10` / reserved / multicast rejected, as is a host written only as numbers that is not a valid IPv4 address; the post never raises, so a flaky webhook can't fail the command). `soup ab` only fires when the sequential test actually decides (`reject_h0` / `accept_h0`), not while it's still `continue`-ing.


## Generación de datos sintéticos

Generate training data using LLMs:

```bash
# Generate using OpenAI API
soup data generate --prompt "Create math word problems" --count 100 --format alpaca

# Use a different model
soup data generate --prompt "Medical Q&A pairs" --model gpt-4o --count 500

# Deduplicate against existing data
soup data generate --prompt "..." --count 200 --dedup-with existing.jsonl

# Use seed examples to guide style
soup data generate --prompt "..." --seed examples.jsonl --count 100

# Use a local OpenAI-compatible server (soup serve, Ollama, etc.)
soup data generate --prompt "..." --provider server --api-base http://localhost:11434/v1
```

With `--provider server`, `openai` or `vllm`, `--api-base` takes plain HTTP only for loopback
(`localhost`, `127.0.0.1`, `::1`) and HTTPS for any other host; `--provider ollama` stays
loopback-only. A private, link-local or reserved IP literal is refused on either scheme, so
address a server on your network by its hostname. A host written only as numbers that is not a
valid IPv4 address (`10.0.0.256`, `4294967296`) is refused as well.

### Multi-Provider Support

```bash
# Generate via local Ollama instance
soup data generate --prompt "..." --provider ollama --model llama3.1
soup data generate --prompt "..." --ollama-model llama3.1  # shorthand

# Generate via Anthropic Claude API (set ANTHROPIC_API_KEY env var)
soup data generate --prompt "..." --provider anthropic --model claude-3-haiku-20240307

# Generate via local vLLM server
soup data generate --prompt "..." --provider vllm --model meta-llama/Llama-3.1-8B-Instruct
```

### Domain Templates

```bash
# Code instruction pairs (Python, JS, Go, Rust, Java)
soup data generate --prompt "..." --template code --language Python --task-type function

# Multi-turn conversations
soup data generate --prompt "..." --template conversation --turns 6 --topic "science"

# QA from context document
soup data generate --prompt "..." --template qa --context document.txt

# Preference data (DPO/KTO/ORPO)
soup data generate --prompt "..." --template preference --pref-task dpo

# Chain-of-thought reasoning (GRPO). `math` rows end with `#### <number>`, so the
# `accuracy` / `verifiable` math rewards can read the gold; `logic` / `code` rows
# end with an `Answer: <answer>` line.
soup data generate --prompt "..." --template reasoning --domain math
```

### Quality Pipeline

```bash
# Auto-validate after generation (remove malformed entries)
soup data generate --prompt "..." --validate

# Auto-filter by quality (coherence scoring)
soup data generate --prompt "..." --filter

# Auto-dedup (MinHash, requires: pip install "soup-cli[data]")
soup data generate --prompt "..." --dedup

# Full quality pipeline: validate + filter + dedup
soup data generate --prompt "..." --quality-pipeline
```


## Aumento de datos

Augment an existing dataset using an LLM — rephrase for diversity, translate for multilingual coverage, or apply a style transform.

```bash
# Rephrase each example N times for more diversity
soup data augment --input ./data/train.jsonl --strategy rephrase --count 3 \
  --output ./data/train_augmented.jsonl

# Translate into multiple languages
soup data augment --input ./data/train.jsonl --strategy translate --lang es,fr,de \
  --output ./data/train_multilingual.jsonl

# Style transfer (formal / casual / technical / etc.)
soup data augment --input ./data/train.jsonl --strategy style --styles formal,casual \
  --output ./data/train_styled.jsonl

# Local provider (Ollama, loopback-only) — pick the model + base URL
soup data augment --input ./data/train.jsonl --strategy rephrase --count 2 \
  --provider ollama --model qwen2.5:0.5b --output ./data/train_local.jsonl
```

Works with `--provider ollama` (the default), `anthropic` or `vllm`. `--model` and `--base-url` select a specific model/endpoint. The Ollama path is loopback-only; the vLLM path takes plain HTTP only for loopback and HTTPS for a remote server, and refuses a private, link-local or reserved IP literal and a host written only as numbers that is not a valid IPv4 address (SSRF-hardened). `--count` is capped at 10; `--lang` and `--styles` each capped at 10 entries × 32 chars.

A provider call that fails (transport error, non-200 status, malformed response) or returns an empty reply never becomes a row: that variant is dropped. The summary reports `N of M provider calls failed` with the first error, and the command exits 1 without writing the output file when no call produced a usable row.


## De trazas a preferencias

Harvest DPO / KTO-ready preference pairs from your production inference logs — no manual labeling.

```bash
# LangChain logs + thumbs-up signal
soup data from-traces --logs ./logs/langchain.jsonl \
  --format langchain --signal thumbs_up --output prefs.jsonl

# OpenAI API logs + regeneration signal (last response wins). The signal is
# `regenerations`; `regeneration` is refused by the CLI (#1440).
soup data from-traces --logs ./logs/openai.jsonl \
  --format openai --signal regenerations --output prefs.jsonl

# Soup-serve logs + user-edit signal (edited response wins over original).
# `--logs` is a DIRECTORY of *.jsonl (one per `soup serve` session): the
# soup-serve format reads a directory; a single file is refused at the
# CLI — read a single JSONL with --format langchain or openai instead (#1530).
soup data from-traces --logs ./traces \
  --format soup-serve --signal user_edit --output prefs.jsonl

# Preview generated pairs before training
soup data review prefs.jsonl --sample 10
```

**Supported log formats:** `langchain`, `openai`, `soup-serve`
**Supported signals:** `thumbs_up` (rating-based), `regenerations` (latest wins), `user_edit` (edited wins)

**The `soup-serve` record shape.** The parser reads a top-level `signal` in the
canonical vocabulary, falling back to a nested `feedback.rating` (`up` / `down`)
for older logs:

```json
{"prompt": "Q", "output": "Good", "signal": "thumbs_up"}
{"prompt": "Q", "output": "Bad",  "signal": "thumbs_down"}
{"prompt": "Q", "output": "Raw",  "signal": "user_edit", "edited_output": "Polished"}
```

Who writes what:

- `soup ingest` is a producer. It writes `thumbs_up`, `thumbs_down` or `none`, under
  the field name `trace_id` rather than `id`, so its output feeds this command
  directly.
- `user_edit` and `regenerated` rows come from **your own** pipeline. `signal:
  user_edit` has to accompany the edit field: a record with `edited_output` but no
  `signal` reads as "no trace carried a signal". The edit itself is read from
  `edited_output`, `edited_response`, or a nested `feedback.edited_output`.
- `soup serve --trace-log` records carry **no signal yet** (`ts`, `prompt`,
  `response`, `latency_ms`, `tokens`), so a harvest of that directory reports the
  diagnostic below rather than a silent 0.

When traces are read but none pair, the command prints how many it read, which
signal it wanted and which signals were present, instead of reporting a normal
write of 0 pairs.

Trace files are capped at 100,000 lines to prevent OOM on production logs. A PII warning panel appears on every run — redact sensitive fields before harvesting.


## Migración de configuración

Switch from other tools with one command:

```bash
# Import from LLaMA-Factory
soup migrate --from llamafactory llama3_lora_sft.yaml

# Import from Axolotl
soup migrate --from axolotl axolotl_config.yml

# Import from Unsloth notebook
soup migrate --from unsloth finetune.ipynb

# Preview without writing
soup migrate --from llamafactory config.yaml --dry-run
```

Automatically maps model, LoRA, training params, quantization, and task type. Warns about unsupported features.
IPython lines in a notebook are skipped, so an unmodified Unsloth notebook reads as the Python it
contains: `!pip install ...` and `%env ...` lines (a `!` line continued with a trailing backslash counts
as one line), a `%%capture` header (its body is still read), and a `%%bash` cell whole; a real syntax
error is reported with its code cell and line.

An axolotl `rl:` value with no Soup task (for example `rl: ebft`) stops the migration: `soup migrate` exits 1 and names the value instead of writing a `task: sft` config.

The same rule holds for the other two sources. A LLaMA-Factory `stage: dpo` with
a `pref_loss` that has no Soup task (`hinge`, `kto_pair`, anything else) stops
the migration with exit 1 naming the value; `ipo` migrates to `task: ipo`, and
`pref_loss` is read only under `stage: dpo`, as LLaMA-Factory does. In an Unsloth
notebook `RewardTrainer` migrates to `reward_model`, `BCOTrainer` to `bco`,
`OnlineDPOTrainer` to `online_dpo` (with a placeholder `training.online_dpo_judge`
to replace), and `CPOTrainer` to `simpo` only when `loss_type="simpo"` is set on
its own call or its `args=` config; any other CPO loss, or an `args=` that cannot
be read statically, stops the migration naming the value. `SFTConfig` and the
other TRL `*Config` classes are read for hyperparameters.
The final trainer determines the task; its inline or named config, passed as `args=`
or positionally, supplies output and hyperparameters using the latest module-level
binding before that trainer (including annotated assignments).
Earlier stages retain warnings but cannot supply settings, and unused configs are ignored.
Aliases, conditional rebindings and unresolved `**kwargs` are not evaluated: migration
reports "Could not read the final trainer's args statically" and uses Soup defaults.


## Formatos de datos

Soup supports these formats (auto-detected). Files can be JSONL, JSON, CSV, Parquet, or TXT.

**Auto-detection.** It reads the first 100 rows, not just the first. Each row resolves
to the richest format its keys fit. If every row agrees, that format is used. A file
that mixes `chatml` and `tool-calling` rows loads as `tool-calling` (the plain rows
convert as ordinary chat turns). Every other mix (`sharegpt` with `llava`, `chatml`
with `audio`, `plaintext` with `asr`, or two unrelated shapes) is refused with a message
naming both shapes and the row where each first appears, because converting the file
with either one would drop the other's keys or rows; set `data.format` to choose. Rows
after the first 100 are not inspected; `soup data validate` lists every row the chosen
format drops.

**Alpaca:**
```json
{"instruction": "Explain gravity", "input": "", "output": "Gravity is..."}
```

**ShareGPT:**
```json
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
```

**ChatML:**
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

**DPO / ORPO / SimPO / IPO (preference pairs):**
```json
{"prompt": "Explain gravity", "chosen": "Gravity is a force...", "rejected": "I don't know"}
```

**KTO (unpaired preferences):**
```json
{"prompt": "Explain gravity", "completion": "Gravity is a force...", "label": true}
```

**LLaVA (vision):**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this."}, {"from": "gpt", "value": "A cat."}]}
```

**ShareGPT4V (vision):**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nExplain this chart."}, {"from": "gpt", "value": "Revenue growth."}]}
```

**Plaintext (pre-training):**
```json
{"text": "Raw text document for continued pre-training..."}
```
Or use `.txt` files directly (one document per line).

**Embedding (sentence embedding pairs/triplets):**
```json
{"anchor": "What is Python?", "positive": "Python is a programming language."}
{"anchor": "What is Python?", "positive": "A programming language.", "negative": "A type of snake."}
```

With `embedding_loss: contrastive` (the default), each row's negatives are the other rows' positives in the same batch, so a batch needs at least two rows: `batch_size: 1` is refused at config load, `batch_size: auto` resolves to at least 2, and the last partial batch of each epoch is dropped. This also applies when `triplet` falls back to contrastive because the rows have no `negative`. Use `triplet` with a `negative` on every row, or `cosine`, to train at batch size 1.

**Cross-Encoder (paired sequence classification - `data.format: cross_encoder`):**
```json
{"text_a": "What is Python?", "text_b": "Python is a programming language.", "label": 1}
{"question": "What is Python?", "answer": "Python is a programming language.", "label": 1}
```
Reads paired text columns (`text_a`/`text_b` or `question`/`answer`) carrying a `label` field. Gated strictly to `task: cross_encoder` (`format: auto` resolves to `cross_encoder` for this task) to preserve global format detection for other tasks.

**Audio (speech + conversation):**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe."}, {"role": "assistant", "content": "Hello world."}]}
```

**ASR (Whisper transcription — `data.format: asr`, v0.71.32):**
```json
{"audio": "clip.wav", "text": "hello world"}
```
Audio paths resolve under `data.audio_dir` (containment-checked). Used by
`task: asr` and `soup infer --task asr`. See [Training → ASR](training.md).

**PRM (process reward, stepwise-supervised):**
```json
{"prompt": "Solve 2+2", "completions": ["First, add", "Result is 4"], "labels": [true, true]}
```

**Pre-tokenized (skip tokenize stage):**
```json
{"input_ids": [1, 2, 3, ...], "labels": [-100, 2, 3, ...], "attention_mask": [1, 1, 1, ...]}
```
Use with `data.format: pre_tokenized` and `data.tokenized_path: ./.soup-tokenized/<key>` after running `soup data preprocess`.

**Input/Output (template-free, segment-level loss control):**
```json
{"segments": [{"text": "Q: hi", "label": false}, {"text": "A: hello", "label": true}]}
```

**Video:**
```json
{"video": "clip.mp4", "messages": [{"role": "user", "content": "Describe this clip."}]}
```

**Multimodal (typed content parts — text / image / audio / video in one message):**
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What's in this?"}, {"type": "image", "url": "x.png"}]}]}
```


## Pipeline de datos Pro

Soup speaks the same dataset surface as Axolotl + LlamaFactory + Unsloth — remote URIs, streaming, sharding, multi-dataset interleaving, vocab expansion, and document ingestion all live in one schema.

**Remote datasets** are loaded through the matching fsspec backend:

```yaml
data:
  train: s3://my-bucket/datasets/train.jsonl   # also gs:// gcs:// az:// abfs:// abfss:// oci://
  streaming: true
  buffer_size: 8192
  shards: 4
```

**HuggingFace Hub names** (a single `data.train` like `org/dataset`): `data.streaming: true`
is forwarded to `datasets.load_dataset(..., streaming=True)` and `buffer_size` shuffles
that stream, then Soup materialises up to 1M rows — the same shape as remote (#689).
An all-hub *list* with `streaming: true` is still refused (#459). `buffer_size` shuffles the train split only; a capped validation split takes the first N rows unshuffled.

**Image and audio paths** in `llava` / `sharegpt4v` / `audio` / `asr` rows resolve the same
way on every load path: a local file, a local or streaming interleave, a remote URI, a Hub
dataset (both splits) and the `data.replay` file. A relative path resolves against
`data.image_dir` / `data.audio_dir` when it is set, otherwise against the directory of the
local file the row came from, and a row whose file lies outside that directory is skipped
with a warning. A remote URI or a Hub dataset has no such directory, so when its rows hold
file paths the setting is required: once the rows are read, the load stops with a message
naming it. `data.image_dir: .` resolves them under the working directory. An image that a
Hub dataset has already decoded carries no path and is passed on as it is.

A value that names a network share, a device or an extended-length path (`\\host\share\...`,
`//host/share/...`, `\\?\...`, `\\.\...`, `\??\...`), or that holds a NUL byte, is skipped
without being looked up, and `soup data inspect` does not look it up either. That includes
a value beginning with two slashes on Linux and macOS, where `//srv/x.png` would otherwise
read as `/srv/x.png`, and, on Windows, a DOS device name such as `nul` or `com1`. Write such
rows relative to the media directory instead, even when the directory itself is on a share.

Before reading a file, the SFT vision and audio paths and the ASR / TTS audio loader check
that it is a regular file. A directory, FIFO or device gets the SFT path's "cannot open"
warning, and stops the ASR / TTS loader with an error, as a symlink does. The check is made
before the file is opened, so a file replaced in between is not caught.

**Multi-dataset interleave** (v0.42.0 schema, wired into training-time loading in #443;
extended to streaming and HF-hub dataset names in #459):

```yaml
data:
  train:
    - dolma.jsonl
    - wikipedia.jsonl
  interleave: { strategy: probs, probs: [0.7, 0.3] }   # also: concat / under / over
  # eval_on_each_dataset: true                         # staged; refused as of v0.77 — #808
```

`data.train` as a list requires `data.interleave` (and vice versa). `training.packing` /
`training.multipack` must be off. With the `probs` strategy, `len(data.train)` must equal
`len(probs)`.

Every list entry is classified once — **local** file path, **remote** URI, or **HF-hub**
dataset name (no *recognised* file suffix — `.jsonl` / `.json` / `.csv` / `.parquet` /
`.txt` only count as local; a hub name with a version number like `teknium/OpenHermes-2.5`
still classifies as hub) — and the classes may not mix within one list. Any entry
containing `"://"` classifies as remote regardless of scheme — a scheme outside the
allowlist (e.g. `https://`, `http://`, `ftp://`) is refused by name at load time rather
than silently falling through to local/hub classification:

- **All entries local files and/or remote URIs:** with `data.streaming: false` (default),
  entries must be local files only (the original #443 path, eager-loaded and combined
  in-process). With `data.streaming: true`, entries may also be remote URIs, and combining
  delegates to HF `datasets.interleave_datasets` / `concatenate_datasets` instead — a
  remote URI entry always requires `data.streaming: true` (there is no non-streaming
  multi-remote-file loader). Each remote entry is canonicalised through the same
  SSRF-hardened `validate_remote_uri` allowlist used everywhere else in Soup (bucket
  regex, no userinfo / query / fragment) before it reaches the streaming loader. The
  streaming path supports the same file types as local loading — `.jsonl` / `.json` /
  `.csv` / `.parquet` / `.txt` — chosen per entry by suffix, so flipping only
  `data.streaming: true` keeps reading the same file format instead of misparsing it as
  JSON; an unrecognised suffix refuses by name rather than reaching the HF loader.
- **All entries HF-hub dataset names** (e.g. `teknium/OpenHermes-2.5`): each name's own
  `train` split is loaded and combined the same way as the local-file path. Always eager —
  `data.streaming: true` is not yet supported for an all-hub-name list (streaming several
  differently-shaped hub datasets through their own split negotiation is unimplemented and
  refuses at parse time, by name). A hub entry's own `validation` split is used for the
  combined val set **only when every entry provides one** (combined the same way); if only
  some entries provide one it is ignored (warned) and `data.val_split` is derived instead:
  per source before `over`/`probs` pad it, same as the local-file path below, or from the
  combined train rows for `concat`/`under`. A partial hub split is not a decided mixture.
- **A mix of hub names with local/remote entries in the same list** always refuses — there
  is no decided answer for how a hub split and a local file's row count should reconcile.

The strategy names mean the same thing on the streaming path as on the local path, though
not byte-identically (a streaming source's size generally can't be known ahead of time):

| strategy | local (eager)                                   | streaming (delegated)                                                        |
|----------|--------------------------------------------------|-------------------------------------------------------------------------------|
| `concat` | every source's rows, in order                     | `concatenate_datasets(streams)`                                              |
| `under`  | truncate every source to the smallest source's size | `interleave_datasets(streams, stopping_strategy="first_exhausted")`       |
| `over`   | upsample every source to the largest source's size (cycled) | `interleave_datasets(streams, stopping_strategy="all_exhausted")` |
| `probs`  | exact apportionment to the requested ratio        | `interleave_datasets(streams, probabilities=probs, stopping_strategy="first_exhausted")` — converges to the same ratio, sampled rather than exact |

On the local (eager) and all-hub-name paths, `data.val_split` is applied per source before
`over`/`probs` pad it with copies of its own rows, so a padded row can never land on both
sides of the split; `concat`/`under` never duplicate rows and still split the combined
result as before.

The streaming path reaches the same guarantee by a different route (#702). A stream is not
countable ahead of time, so there is nothing to take a fraction of before interleaving
starts; instead, once `over` has been materialised, the split is sized over the *distinct*
rows and val is taken from the end of the stream, preferring rows whose content occurs only
once, so train keeps every row and all of its oversampling. Only if there are too few such
rows is repeated content moved to val, and then its other copies are withheld from train
and the number withheld is printed as a warning. A split that would leave train empty
raises instead. Because val comes from distinct rows in stream order rather than from each
source in turn, **it is not balanced across sources**: with 100 rows against 10 under
`val_split: 0.1`, every val row comes from the larger source, since the smaller one's rows
are all recycled. The eager path's per-source carve-out is mixture-representative; this one
is not.

`concat`/`under`/`probs` do not *add* duplicates on the streaming path (only `over` uses
`stopping_strategy="all_exhausted"`), so they keep the ordinary positional split. That is a
statement about interleaving, not about your data: rows that are already duplicated in a
source can still land on both sides of the split under any strategy, on either path.

Splitting before padding also means the requested `val_split` fraction is no longer exact
under `over`/`probs`: it is taken from each source's own (smaller, unpadded) row count, so
the held-out share of the final, padded total comes out lower than requested. For example,
two sources of 1000 and 100 rows with `over` and `val_split: 0.1` yield 110 val rows out of
1910 total (5.8%), not the 200/2000 (10%) a single-source split would give. `concat`/`under`
are unaffected (they never pad). This is the trade-off for closing the duplicate-row leak,
not a separate bug: holding out an exact 10% of the padded total would mean some val rows
are copies of val rows already counted, or of train rows.

Train and val also end up with different source mixtures once `over`/`probs` pads: val is
carved from each source's original, unpadded rows, while train sees the padded, rebalanced
mix. Anyone who oversampled specifically to correct a source imbalance gets a validation set
that still reflects the original, un-rebalanced skew, not the mixture train now trains on.

**Vocab expansion + advanced masking:**

```yaml
data:
  add_new_tokens: ["<reasoning>", "</reasoning>"]
  new_special_tokens: ["<|tool_call|>"]
  mask_history: true              # train only on the LAST assistant turn
  # Staged fields below warn in v0.76 and are refused as of v0.77 (#808):
  # resize_vocab: true
  # split_thinking: true            # Qwen3-style <think> reasoning-block masking
  # image_min_pixels: 256
  # image_max_pixels: 4096
  # image_resize_algorithm: bicubic
  # video_fps: 24
  # video_maxlen: 32
  # video_dir: ./videos
```

`add_new_tokens` / `new_special_tokens` are added to the tokenizer, with the embeddings
resized, on `backend: transformers` for `sft` (text, vision and audio), `dpo`, `kto`,
`orpo`, `ipo`, `simpo`, `bco`, `grpo`, `online_dpo`, `preference` and `tts`, and on SFT's
vision and audio paths whatever the backend (they run their transformers setup before the
backend is read). Any other task, `backend: unsloth` (text SFT and every other task) or
`mlx`, and `training.stream_layers` never add them, so a config setting them there is
refused at load, naming which of those applies (#1358).

`mask_history: true` keeps only the **last** assistant turn in the loss: every
earlier assistant turn is masked alongside the user and system turns the
assistant-only path already excludes. It never adds tokens to the loss.

It only means something for a **multi-turn chat shape** — `chatml`, `sharegpt`
and the other message-list formats — where turns exist to mask. A single-turn
conversation trains identically with it on or off, and a flat format such as
`alpaca` or `plaintext` has no turns at all.

It requires `train_on_responses_only: true`, which is the path that marks
tramos asistentes; con `false` cada token entrena, incluida la historia de este
El campo solicita excluir, por lo que la combinación se rechaza en la carga de configuración. Eso también
descarta `train_on_messages_with_train_field`, que en sí mismo es exclusivo de
`train_on_responses_only`: el campo por mensaje `train` y `mask_history` pueden
Nunca ambos deciden una carrera.

En texto SFT es respetado por los transformadores y los backends unsloth, que ejecutan
el mismo generador de hileras `SFTTrainerWrapper.setup()`, y `task: distill` lo honra
también, ya que distill construye cada fila de esa manera. En `task: sft` con
`modality: vision` o `audio` se acepta pero no se aplica: esas filas son
construido por los preparadores de visión y audio, que nunca lo leyeron (#1156).

**`backend: mlx` lo ignora:** MLX SFT construye su propia máscara y supervisa cada
turno asistente, por lo que la misma configuración entrena el último giro en los transformadores y cada
Enciende MLX. `soup train` lo dice en su línea "MLX backend ignora:", y
`soup doctor --config` lo informa.

**Visión multimodal y audio ignórelo:** Para `modality: vision` y `modality: audio`,
cada token de texto está supervisado. El clasificador de visión de Soup enmascara solo el relleno y la imagen.
tokens y la ruta de audio solo relleno. El enmascaramiento exclusivo del asistente tendría que localizar al
El asistente se extiende después de que el procesador expande los tokens de imagen o audio, que ninguna ruta
lo hace todavía. Tanto `mask_history` como `train_on_responses_only` no están leídos en estos
modalidades, por lo que cada token de texto se entrena. `soup doctor --config` informa un explícito
`mask_history: true` como ignorado. No incluye `train_on_responses_only`: su
El esquema predeterminado es `true`, y escribir un valor predeterminado cuenta como dejarlo sin escribir.
La ausencia de una fila de médicos no implica enmascaramiento multimodal solo para asistentes.


Preprocesamiento **AOT:**
```bash
# Tokenize once, reuse the cache across runs.
soup data preprocess soup.yaml --output ./.soup-tokenized

# Then in soup.yaml:
#   data:
#     format: pre_tokenized
#     tokenized_path: ./.soup-tokenized/<16-char-cache-key>
```

Los cachés pre-tokenizados contienen los identificadores producidos por el preprocesamiento de datos de sopa.
Debido a que el preprocesamiento no aplica una modalidad que no sea de texto (`vision`, `audio`,
o `audio_out`), `data.add_new_tokens`, `data.new_special_tokens`, o
`data.prompt_strategy`, esas combinaciones se rechazan en la carga de configuración
de ser ignorado en silencio.

**Ingestión de documentos (PDF/DOCX/MD/TXT → JSONL):**

```bash
soup data ingest report.pdf --output report.jsonl
soup data ingest README.md
soup data ingest notes.docx
```

**Estrategias de solicitud personalizadas (solo esquema: invocación en tiempo de ejecución en v0.42.1):**

```yaml
data:
  prompt_strategy: my_pkg.transforms:rephrase
```


## Herramientas de datos

```bash
# Inspect a dataset
soup data inspect ./data/train.jsonl

# Validate format (auto-detects if --format not specified)
soup data validate ./data/train.jsonl
soup data validate ./data/train.jsonl --format alpaca

# Require at least 90% of rows to be usable
soup data validate ./data/train.jsonl --min-valid-fraction 0.9

# Convert between formats
soup data convert ./data/train.jsonl --to sharegpt --output converted.jsonl

# Merge multiple datasets
soup data merge data1.jsonl data2.jsonl --output merged.jsonl --shuffle

# Remove near-duplicates (requires: pip install "soup-cli[data]")
soup data dedup ./data/train.jsonl --threshold 0.8

# Extended statistics (length distribution, token counts, languages)
soup data stats ./data/train.jsonl

# Filter by quality (perplexity + coherence scoring)
soup data filter ./data/train.jsonl --coherence 0.3
soup data filter ./data/train.jsonl --perplexity 500 --coherence 0.3
soup data filter ./data/train.jsonl --score-only  # add scores without filtering

# Clean dataset (control chars, zero-width spaces, empty turns; opt-in heuristics)
soup data clean ./data/train.jsonl
soup data clean ./data/train.jsonl -o ./data/clean.jsonl --dry-run
```

`soup data validate` sale con el código `0` cuando al menos una fila es utilizable y el
Se cumple la fracción mínima válida opcional. Sale con el código `3` para errores de entrada,
como un archivo faltante o un formato indetectable, y codifique `2` cuando no esté vacío.
el conjunto de datos no tiene filas utilizables o está por debajo de `--min-valid-fraction`. Un parcialmente válido
El conjunto de datos todavía existe con el código `0` cuando no se especifica ningún mínimo.

El entrenamiento carga un conjunto de datos a través de los mismos convertidores, y una fila que rechazan es
cayó en lugar de detener la ejecución, por lo que una línea defectuosa no aborta la carga. El
Se informa la caída: `soup train`, y cualquier otro comando que cargue un conjunto de datos,
imprime `Warning: N of M rows dropped` con el índice de la primera fila y el
Razón del convertidor. Para un archivo local también imprime el `soup data validate`
comando que los enumera todos. El
El recuento coincide con `soup data validate` para el mismo archivo. Antes del n.° 1181 las filas
fueron eliminados sin decir palabra. Si una carga termina con cero filas de entrenamiento, `soup train`
se detiene en la salida 1 antes de cargar el modelo, incluido `--dry-run`, nombrando el formato
las filas se leyeron como y el motivo de eliminación de la primera fila (#1217).


## Conjuntos de datos de demostración (`soup data demo`)

Pequeños dispositivos JSONL incluidos con Soup para que puedas calentar `soup train` sin
buscando datos:

```bash
# List available bundles
soup data demo

# Copy one into the current directory
soup data demo alpaca_demo --output ./alpaca.jsonl
```

Paquetes: `alpaca_demo`, `sharegpt_demo`, `dpo_demo`, `grpo_demo`. Ruta de salida
debe permanecer bajo control de tracción; Los archivos existentes no se sobrescriben.


## Seguimiento a preferencia: filtro de juez LLM

`soup data from-traces --judge` filtra los pares de preferencias recopilados a través de un juez de LLM:

```bash
soup data from-traces \
  --logs ./prod-traces.jsonl --format langchain --signal thumbs_up \
  --output ./prefs.jsonl \
  --judge --judge-provider ollama --judge-model llama3 \
  --min-confidence 0.7
```

El juez califica `chosen` y `rejected` de forma independiente según su rúbrica (utilidad/precisión/seguridad predeterminada en una escala de 1 a 5). Los pares cuya confianza normalizada de `(chosen - rejected)` cae por debajo de `--min-confidence` se eliminan. Las excepciones de backend por par se cuentan (no fallan) y se informan. Un juez que deja de responder no se cuenta de esa manera: una vez que 3 solicitudes seguidas han pasado sus reintentos sin respuesta, el comando se detiene con la salida 1, nombra al juez y cuántos pares no fueron juzgados, y no escribe ningún archivo de salida (#1522). Lista de permitidos de proveedores `{openai, server, ollama}` validada en el límite de CLI; La protección SSRF en `--judge-api-base` se transfiere desde `soup eval judge`.


## Forja de datos sintéticos

Canalización de datos sintéticos de varias etapas con procedencia completa: cada fila sintética enlaza con el documento fuente, la decisión del juez y la puntuación del filtro:

```bash
# Pipeline: chunk docs → judge → active-prune → JSONL + provenance manifest
soup data forge \
    --docs ./my_docs/ \
    --task sft \
    --target-rows 1000 \
    --uncertainty-threshold 0.4 \
    --output forge_dataset.jsonl \
    --provenance forge_provenance.json
```

Se admiten tres tareas: `sft` (pares de preguntas y respuestas), `preference` (elegida/rechazada), `tool` (hipótesis de llamada de herramienta). El aprendizaje activo elimina las filas cuya respuesta del juez está demasiado cerca del fragmento de origen (baja distancia de Jaccard), manteniendo solo muestras inciertas/informativas. El manifiesto de procedencia es un archivo JSON independiente que asigna cada identificación de fila a `{source_doc, judge_id, chunk_id, filter_score}` para que tenga un seguimiento de auditoría completo para el cumplimiento.

El descubrimiento de documentos tiene un nivel de profundidad sobre `.txt` / `.md` / `.json` / `.jsonl`; Se omiten los archivos de puntos + directorios con enlaces simbólicos. Todas las rutas están contenidas en cwd, todas las escrituras son atómicas a través del archivo temporal preparado + `os.replace` y los destinos de escritura se rechazan si son enlaces simbólicos. **Los proveedores de jueces están en vivo**: `--judge-provider ollama` (solo host local), `--judge-provider anthropic` (clave API solo env), `--judge-provider vllm` (validado por esquema). Una llamada de juez que falla (error de transporte, estado distinto de 200, respuesta con formato incorrecto) o que devuelve una respuesta vacía nunca se convierte en una fila, sea lo que sea `--uncertainty-threshold`. El resumen informa `N of M judge calls failed` con el primer error y el comando sale 1 sin escribir archivos cuando no se produjo ninguna fila utilizable.

**Acentros de profesores alternativos (v0.71.5).** `--hub modelscope|modelers` busca previamente el `--teacher` de ese centro cuando el profesor es un ID de repositorio enrutable (`owner/name`); `--hub hf` (predeterminado) no es operativo y deja al maestro como una etiqueta de procedencia. Si `--hub` no es HF pero `--teacher` no es un ID de repositorio (por ejemplo, el `local-judge` predeterminado), Soup imprime una advertencia amarilla fuerte en lugar de soltar la bandera en silencio.


## Cuadro de mando de calidad de datos

Clasificación de calidad de datos liviana y compuesta: sin GPU, sin modelo Presidio de 200 MB:

```bash
# Single-shot composite scorecard
soup data score --input training.jsonl

# Include real operator-supplied comparison texts (no benchmark corpora are bundled)
soup data score --input training.jsonl --benchmark-file benchmark.jsonl --threshold 0.8

# Standalone subcommands — JSONL-in, enriched JSONL-out
soup data pii          --input training.jsonl --output pii_flagged.jsonl
soup data toxicity     --input training.jsonl --output tox_flagged.jsonl --threshold 0.1
soup data langdetect   --input training.jsonl --output tagged.jsonl
soup data educational  --input training.jsonl --output scored.jsonl
soup data decontaminate --input training.jsonl --benchmark-file benchmarks.jsonl --benchmarks mmlu,gsm8k,humaneval --output clean.jsonl
```

`data score` muestra la descontaminación como **no ejecuta**, no es un cero medido, a menos que
`--benchmark-file` proporciona textos comparativos reales. `--benchmarks` / `-b`
contiene únicamente etiquetas permitidas y ahora requiere ese archivo; los nombres solos no
descargar o agrupar corpus de referencia. Esta es una negativa demoledora para el ex
ruta silenciosa `data score -b ...` (#1448). Utilice `--benchmark-file` con o sin
etiquetas. Con un archivo, las etiquetas ** solo se validan **: no seleccionan ni filtran
sus filas, y el comando imprime una nota informándolo. Todos los textos del archivo suministrado entran
el mismo corpus de comparación, por lo que omitir etiquetas o cambiar una etiqueta válida a
otro no cambia el recuento de eliminaciones. El JSONL del archivo está delimitado y
cwd contenido por el cargador compartido, con estricto
análisis: filas con formato incorrecto/sin objeto, archivos vacíos y filas sin 8 gramos utilizables
El texto se rechaza antes de imprimir el cuadro de mando. El texto proviene de `text`,
`content` o `messages[].content`, a través del extractor compartido. El fijo de 8 gramos.
usos heurísticos `--threshold`; un cero resultante describe sólo los suministrados
textos, no la ausencia de contaminación frente a todos los parámetros públicos. El otro
Las rutas de entrada/cargador independiente mantienen su comportamiento tolerante anterior.

El cuadro de mando informa coincidencias de PII, coincidencias de palabras clave abusivas, distribución del idioma, valor educativo heurístico medio y eliminaciones de descontaminación. La detección de PII utiliza un conjunto reducido de expresiones regulares reforzadas con ReDoS (correo electrónico/teléfono/SSN/tarjeta de crédito) con un límite previo de 50 KB en cada entrada. La detección de idioma es una heurística de palabras vacías en seis idiomas. `soup data toxicity` se mantiene como nombre de comando compatible, pero su salida es explícitamente una heurística de palabras clave de abuso, no un clasificador de toxicidad. Los términos técnicos y médicos ambiguos como proceso `kill`, hilo `die` y corazón `attack` no se tratan como señales de seguridad independientes. Esto cambia un modo de falla conocido por limitaciones explícitas: en la revisión del mantenedor, 9 de 10 ejemplos abusivos retenidos obtuvieron una puntuación de cero y 10 de 12 ejemplos técnicos o editoriales benignos fueron marcados en el umbral predeterminado. Úselo solo para clasificar palabras clave, nunca como una decisión de seguridad. Por lo tanto, el filtro de calidad predeterminado de Magpie aplica solo heurísticas educativas y no vacías; proporcionar una política explícita respaldada por modelos fuera de Soup cuando se requiere una clasificación de seguridad. El extra `[data-pro]` actualmente agrega `langdetect` y Presidio únicamente; no instala Llama Guard ni FineWeb-Edu. La descontaminación utiliza contención de n-gramas contra los mensajes de texto proporcionados por las personas que llaman: `--benchmarks mmlu,gsm8k` selecciona solo etiquetas permitidas; utilice `--benchmark-file custom_benchmark.jsonl` para datos de comparación reales.


## Conjuntos de datos remotos (S3/GCS/Azure/OCI)

Apunte `data.train` a cualquier objeto en la lista de permitidos de fsspec v0.42.0 y `soup train` lo transmitirá a través de `fsspec.open` después de ejecutar el URI a través del mismo validador reforzado con SSRF que se usa en todas partes de Soup (expresión regular del depósito, sin información de usuario/consulta/fragmento):

```yaml
data:
  train: s3://my-bucket/datasets/train.jsonl
  format: alpaca
  streaming: true       # opt-in HF datasets streaming with shuffle
  buffer_size: 10000    # shuffle buffer (requires streaming=true)
```

Esquemas reconocidos: `s3://`, `gs://`, `gcs://`, `az://`, `abfs://`, `abfss://`, `oci://`. El SDK de backend correspondiente (`s3fs` / `gcsfs` / `adlfs` / `ocifs`) se importa de forma diferida: instale solo lo que necesita o aproveche la comodidad adicional:

```bash
pip install soup-cli[remote]   # fsspec + s3fs + gcsfs + adlfs
```

Las filas materializadas tienen un límite de 1M para defenderse de objetos patológicos remotos; utilice una división local para trabajos más grandes.


## Receta de datos DAG

```bash
soup data recipe my_recipe.yaml
```

```yaml
nodes:
  - name: seed1
    kind: seed
    config: {path: prompts.jsonl}
  - name: llm1
    kind: llm_text
    config: {prompt: "Answer the request: {text}"}
  - name: judge1
    kind: judge
  - name: samp1
    kind: sampler
edges:
  - [seed1, llm1]
  - [llm1, judge1]
  - [judge1, samp1]
```

Lista de permitidos de tipo de nodo cerrado (`seed` / `llm_text` / `code` / `judge` / `validator` / `sampler`); Clasificación topológica de Kahn mediante `collections.deque` (determinista, O(N+E)); ciclo / bucle automático / borde duplicado / borde colgante / rechazo de tipo desconocido. `_MAX_NODES=256`, `_MAX_EDGES=1024`, `_MAX_FILE_BYTES=1MiB`. El archivo de receta debe permanecer en cwd y ** no debe ser un enlace simbólico** (defensa `os.lstat + S_ISLNK` TOCTOU).


## Optimizador de mezcla de datos (BETA)

Busque pesos de mezcla de conjuntos de datos que minimicen la pérdida de evaluación en una ejecución corta de proxy.

```bash
soup data mix --optimize --budget 1h \
    --datasets dolma.jsonl,wikipedia.jsonl,arxiv.jsonl \
    --num-probes 8 --output mix_recipe.yaml
```

Escribe una receta YAML que puedes insertar en tu `soup.yaml`: `data.train` se representa como la lista completa de conjuntos de datos clasificados (alineados con el índice con `data.interleave.probs`) y `data.interleave` lleva los pesos de la mezcla buscada; a partir del n.° 443, `data.interleave` está completamente conectado a la carga del conjunto de datos en tiempo de entrenamiento, por lo que `soup train` consume la mezcla real de N conjuntos de datos que encontró esta búsqueda en lugar de colapsando en un solo camino. `--budget` acepta `60s` / `5m` / `1h` / `24h`. Las fallas de proxy por candidato se aíslan (se registran DEBUG, se registran altas pérdidas centinela) por lo que un solo combo OOM no cancela toda la búsqueda; `partial=True` aparece en el informe cuando el límite presupuestario se dispara a mitad del ciclo.

Vuelva a aplicar una receta previamente escrita:

```bash
soup data mix --apply mix_recipe.yaml
```

Pase `--live --base-yaml soup.yaml` para calificar a cada candidato con una breve ejecución de proxy `soup train`. Sin `--live`, Soup utiliza un proxy sintético fuera de línea (penalización cuadrática alrededor del simplex uniforme) para que el rastreador de presupuesto, la superficie optimizadora y el escritor de recetas se puedan ejercer sin GPU. `scikit-optimize` se puede inscribir a través de `OptimizerProtocol`; el recurso alternativo predeterminado es un muestreador determinista de Dirichlet.


## Tokenización AOT con `soup data preprocess`

Pre-tokenice su conjunto de datos una vez y almacene en caché los fragmentos de flecha codificados por
`(dataset, tokenizer, max_length, format, chat_template, loss-mask mode, task)`:

```bash
soup data preprocess soup.yaml --output ./tokenized_cache
```

Los entrenadores SFT y Pretrain sufren un cortocircuito en la validación del esquema cuando
Se establece `format: pre_tokenized` + `tokenized_path: ./tokenized_cache`, eliminando
el impuesto de tokenización por época. Las claves de caché garantizan la seguridad del currículum; selección de carreras parciales
desde el último fragmento completado.

Las filas se representan con `data.chat_template` cuando está configurado, igual que en vivo
capacitación. La configuración de entrenamiento `pre_tokenized` debe nombrar la misma plantilla, ya que
el entrenamiento guarda el tokenizador con él; uno diferente es rechazado con
`cache hash mismatch`. Un caché escrito antes de que la plantilla se uniera a la clave es
se negó de la misma manera: vuelva a ejecutar `soup data preprocess` para reconstruirlo.

Una fila que el comando no puede tokenizar lo detiene y no se escribe nada. Para preentrenamiento
esa es una fila de texto que el tokenizador rechaza (un sustituto solitario, por ejemplo), que
el preentrenamiento en vivo también se niega; El cargador deja caer una fila `text` vacía en ambos
caminos. Eso cubre una conversación que la plantilla rechaza (por ejemplo
`Conversation roles must alternate` en una plantilla estilo Llama-2 o Gemma), un
lista `messages` vacía, una fila que la plantilla representa como texto vacío y un tokenizador
error. El mensaje numera la fila desde 1, como lo hace el entrenamiento en vivo, contando los
filas que sobrevivieron a la carga y cita el inicio de su primer mensaje (o de un
preentrenar el texto de la fila), que es lo que lo encuentra cuando se elimina una fila anterior o
Los archivos estaban entrelazados. El entrenamiento en vivo se detiene ante una conversación rechazada, una
uno vacío y un render vacío también, por lo que un caché que los omitiera se entrenaría en
menos filas que el mismo `soup.yaml` ejecutado en vivo. Antes del n.° 1180, fueron eliminados.
sin una palabra y el comando salió de 0. Corrija o elimine la fila y vuelva a ejecutarla. A
El tokenizador sin plantilla de chat se informa una vez, antes de cualquier fila, y apunta a
`data.chat_template`. Las filas se verifican cuando el comando crea un caché: si uno
con la misma clave ya existe se detiene en `Target already exists` (salida 0)
sin leerlos, así que reconstruya un caché escrito antes de este cambio con `--yes`.

Las filas almacenadas en caché llevan una columna `labels` enmascarada exactamente como la ejecución en vivo equivalente
lo enmascararía (`data.train_on_responses_only` /
`data.train_on_messages_with_train_field`, más `data.mask_history` y
`training.train_on_eot`). Ese modo de máscara también es parte de la clave de caché, por lo que un caché
construido bajo una configuración de enmascaramiento es rechazado - con el mismo
Error `cache hash mismatch`: cuando se carga bajo uno diferente. cachés escritos
antes de esta solución (esquema de tokenizador `v5` y anteriores) no tienen `labels` y son
rechazado; vuelva a ejecutar `soup data preprocess`. Con `task: sft`, un `pre_tokenized`
El conjunto de datos que usted mismo creó debe llevar su propia columna `labels` (`-100` en cada
ficha para no entrenar); Se rechaza un tren o un split de validación sin uno.
en lugar de entrenar en cada ficha. `task: pretrain` no tiene tal verificación: a
conjunto de datos sin trenes `labels` en cada token, porque el clasificador de TRL copia
`input_ids` en `labels`. Ese es el objetivo del preentrenamiento, y un
La caché `soup data preprocess` creada para `task: pretrain` registra las mismas etiquetas.

La clave completa, como `PREPROCESS_KEY_FIELDS` en `src/soup_cli/utils/data_pipeline.py`
lo declara:

| Entrada clave | Campos de configuración |
|---|---|
| conjunto de datos | `data.train`, `data.interleave`, `data.val_split`, `data.replay`, `data.replay_ratio`, `data.replay_seed`, `data.streaming`, `data.buffer_size`, `data.image_dir`, `data.audio_dir` |
| tokenizador | `base` |
| longitud_max | `data.max_length` |
| formato | `data.format` (la lectura previa al proceso del formato fuente, registrada en `metadata.json`) |
| plantilla_chat | `data.chat_template`, resuelto al Jinja que renderiza |
| modo de máscara de pérdida | `data.train_on_responses_only`, `data.train_on_messages_with_train_field`, `data.mask_history`, `training.train_on_eot` |
| tarea | `task` |

La entrada del conjunto de datos cubre todas las configuraciones que deciden qué filas se almacenan en caché: solo
la división del tren se almacena en caché, las filas de reproducción se mezclan primero y la transmisión
Los cargadores eligen filas y su orden. Todos los demás campos `data` se enumeran en
`NOT_PREPROCESS_KEY_FIELDS` por el motivo de que no puede cambiar una fila almacenada en caché y un
Se debe agregar un nuevo campo a una de las dos tablas. Cachés escritos antes de esto
(esquema de tokenizador `v6` y anteriores) son rechazados; vuelva a ejecutar `soup data preprocess`.

La configuración de entrenamiento `pre_tokenized` debe mantener `data.val_split`, `data.replay`,
`data.replay_ratio`, `data.replay_seed`, `data.streaming`, `data.buffer_size`,
`data.image_dir` y `data.audio_dir` como estaban cuando se creó el caché, tal como
Debe conservar la plantilla de chat. Se rechaza una caché creada con valores diferentes y
el mensaje nombra los campos que cambiaron. Un caché escrito antes de esta clave.
lo dice en lugar de mostrar dos hashes desnudos.


## Ejecutor DAG de receta de datos (`soup data recipe --execute`)

Ejecute un DAG de receta de datos de un extremo a otro:

```bash
soup data recipe path/to/recipe.yaml --execute --output ./out \
    --provider ollama --model llama3.1
```

Los nodos `llm_text` y `judge` admiten `ollama`, `anthropic` y `vllm`; usar
`--base-url` para anular el punto final de loopback para Ollama o vLLM. Corriendo cualquiera
El tipo de nodo sin `--provider` se rechaza, por lo que los datos del marcador de posición no pueden confundirse.
para generaciones vivas. Solo para pruebas deterministas, `--offline` habilita explícitamente
Marcadores de posición `llm_text(offline): ...` y hace que los nodos jueces acepten cada fila; el
El comando imprime una advertencia cada vez que este modo está activo.

Se cuentan las fallas de llamadas de proveedores en vivo: si cada intento de llamada para un `llm_text`
o el nodo `judge` falla, el comando nombra el punto final y sale 1. Fallos parciales
mantener filas utilizables e informar su recuento en el resumen de finalización, mientras que un `llm_text`
El proveedor que devuelve legítimamente una finalización vacía todavía cuenta como una llamada exitosa.
Un nodo activo `judge` lee la primera palabra de cada respuesta: `OK` mantiene la fila, `REJECT`
o `NOT OK` lo descarta, y una respuesta que no incluye ninguno de los dos, incluida una vacía, se descarta y
cuenta como una llamada fallida.

Ahora se ejecutan seis tipos de nodos en vivo: **seed** (carga JSONL), **llm_text** (generación LLM mediante
Ollama, Anthropic o vLLM), **code** (ejecución a través de RLVR sandbox), **judge** (puntuación binaria),
**validator** (esquema regex o JSON), **sampler** (selección determinista). Control
escrito por nodo; reanudar la rehidratación de los sidecares por nodo. Filas fallidas registradas con
motivos redactados (rutas eliminadas, con un límite de 256 caracteres).
Los nodos validadores de expresiones regulares rechazan patrones estructuralmente inseguros antes de hacer coincidir filas;
el error identifica el campo `config.regex` del nodo. Quedan simples alternancias
válido.


## Doctor afinado (`soup data doctor`)

Informe de compatibilidad de plantillas de chat: detecta los principales errores de ajuste de *silent*
antes de un solo paso de entrenamiento:

```bash
soup data doctor ./data/train.jsonl --model meta-llama/Llama-3.1-8B-Instruct

# Render N sample rows with per-token trained/masked colouring, through the REAL
# collator path (answer-only / per-message-train-field / RAFT span-mask)
soup data doctor ./data/train.jsonl --model meta-llama/Llama-3.1-8B-Instruct --show-mask 5
```

Ocho comprobaciones, la misma taxonomía OK/MENOR/MAJOR que `soup diagnose` (salida 0 en OK/MENOR,
salida 2 en MAYOR): `chat_template` (el tokenizador tiene uno), `template_render` (renderiza
limpiamente en una muestra), `generation_markers` (soporte `{% generation %}`),
`eos_in_labels` — el error **#1 "el modelo nunca deja de generar"**: todos los entrenados
El turno del asistente debe contener en realidad un token EOS/EOT, marcado en todo el *whole*
lapso entrenado, no solo el último turno: `bos_duplication` (plantilla + tokenizador ambos
anteponiendo BOS), `system_role` (plantillas estilo Mistral que rechazan un sistema líder
giro), `unknown_roles` y `truncation_risk` (p95 longitud renderizada vs.
`data.max_length`). `--train-on-responses-only` / `--train-on-messages-with-train-field`
seleccione la misma estrategia de enmascaramiento que usaría `soup train`, para que el informe y
La vista previa de `--show-mask` nunca puede estar en desacuerdo sobre lo que realmente se entrenó.
`--mask-history` (desactivado por defecto, coincide con `data.mask_history`) reduce el
máscara de solo asistente al turno de asistente **last**, exactamente como el
sopa.yaml bandera del mismo nombre; se rechaza con
`--no-train-on-responses-only` o `--train-on-messages-with-train-field`,
Las mismas combinaciones `soup.yaml` se niegan.


## Linter de datos de preferencia (`soup data lint`)

Capta las principales degradaciones silenciosas en los datos de preferencias de DPO/ORPO/SimPO/IPO/BCO/KTO:

```bash
soup data lint ./data/prefs.jsonl
soup data lint ./data/prefs.jsonl --model meta-llama/Llama-3.1-8B-Instruct  # exact token-length bias, not word count
```

Cinco comprobaciones: `length_bias`: la degradación silenciosa del DPO **#1**: `chosen`
sistemáticamente más largo que `rejected`, informado como tamaño del efecto d de Cohen; MAJOR
requiere |d| >= 0.8 y longitudes medias con al menos un 10% de diferencia; MENOR requiere |d| >= 0.3 y 5%, por lo que
la brecha constante de una palabra entre longitudes casi constantes no está marcada -
`label_imbalance` (relación KTO deseable:indeseable), `near_duplicates`
(MinHash/LSH, reutiliza el kernel `soup data dedup`; requiere
`pip install "soup-cli[data]"`, de lo contrario se degrada a un aviso de omisión),
`identical_pairs` (`chosen == rejected` - señal de preferencia cero), y
`prompt_leak` (el mensaje se repitió palabra por palabra dentro de la finalización, un
error en la canalización de datos sintéticos). Para conversacional `chosen` / `rejected` (mensaje
listas), `length_bias` y `prompt_leak` leen sólo los turnos del asistente, ya que el
El turno principal del usuario es el mensaje en sí. La misma taxonomía y códigos de salida OK/MENOR/MAJOR que
`soup data doctor`.
