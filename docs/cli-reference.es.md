# Referencia de la CLI

> 🇪🇸 Traducción al español de [`cli-reference.md`](cli-reference.md). Ante cualquier duda, manda el original en inglés.

[← Volver al índice de la documentación](README.md)

> **Archivo generado — no lo edites a mano.**
> Se renderiza desde el árbol de comandos en vivo de Typer (`soup_cli.cli:app`). Después de agregar
> o quitar un comando, una opción o un argumento, o de cambiar su tipo, su valor por defecto o
> si es obligatorio, ejecuta `python scripts/gen_reference_docs.py` y haz commit del
> resultado; la suite de pruebas falla hasta que la página coincida. El texto de ayuda se actualiza
> cuando se corta una release, así que entre releases el texto de aquí puede ir por detrás del código.

Todos los comandos de `soup` con todas sus opciones y argumentos. Para un listado de una línea
por tarea consulta la [Referencia de comandos](commands.md); para las claves de `soup.yaml` consulta
la [Referencia de configuración](config-reference.md).

Cómo leer una tabla:

- **Name** es una opción con su forma corta, o un argumento en mayúsculas (`PATH`;
  `PATH...` toma cualquier número de valores). `--flag` / `--no-flag` es un solo interruptor con
  ambas formas.
- **Type**: un `flag` no toma valor; `text`, `integer`, `float` y `path` toman uno;
  una opción `repeatable` puede indicarse más de una vez.
- **Default** es el valor que se usa cuando se omite la opción (`on` / `off` para un flag);
  una celda vacía significa que no hay ninguno. *(set at runtime)* marca un valor por defecto calculado en
  la máquina, que esta página no imprime.
- **Required** dice `yes` cuando el comando se niega a correr sin esa opción.

`--help` funciona en todos los comandos y no se lista. Las opciones globales van antes del
nombre del comando: `soup --verbose train --config soup.yaml`.

## Opciones globales

Uso: `soup [OPTIONS] <command> [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--verbose`, `-V` | flag | off |  | Muestra el traceback completo en los errores |
| `--log-level` | text | `normal` |  | Nivel de logging: quiet \| normal \| verbose \| debug |
| `--no-audit-log` | flag | off |  | Desactiva el registro de auditoría HIPAA/SOC2 local para esta invocación (también vía SOUP_NO_AUDIT_LOG=1). Por defecto: un registro de una línea por comando en \~/.soup/audit.jsonl. v0.71.3. |
| `--no-telemetry` | flag | off |  | Excluye esta ejecución de la telemetría anónima de solo hardware. |
| `--install-completion` | flag |  |  | Instala el autocompletado de shell para el shell actual. |
| `--show-completion` | flag |  |  | Imprime el script de autocompletado del shell, para copiarlo o personalizarlo. |

## Comandos

| Command | Summary |
|---|---|
| [`soup ab`](#soup-ab) | Arnés A/B secuencial mSPRT de dos caras sobre latency / judge_score / retry_rate: informa si el tratamiento es mejor o peor, válido bajo re-ejecución tras cada fila nueva (v0.63.0). |
| [`soup adapters`](#soup-adapters) | Gestión de adaptadores: list, info, compare de adaptadores LoRA. |
| [`soup advise`](#soup-advise) | Decisión previa al vuelo: PROMPT_ENG / RAG / SFT / DPO / GRPO. Ejecútalo ANTES de gastar 8 horas en una GPU (v0.54.0). |
| [`soup agent`](#soup-agent) | Agent Forge: spec -> dataset de tool-calling / train / eval (v0.46.0). |
| [`soup airgap-bundle`](#soup-airgap-bundle) | Construye un tarball firmado con modelo + datasets + wheels + kernels (v0.60.0). |
| [`soup apple-adapter`](#soup-apple-adapter) | Convierte / firma adaptadores HF / MLX / Apple FoundationModels (v0.68.0). |
| [`soup apply`](#soup-apply) | Ejecuta la corrida de entrenamiento planificada, negándose ante una deriva entre soup.yaml y soup.tfstate (v0.64.0). |
| [`soup attest`](#soup-attest) | Atestaciones In-toto + SLSA-3 por etapa de Soup Can (v0.59.0). |
| [`soup audit-log`](#soup-audit-log) | Registro de auditoría JSONL con forma HIPAA/SOC2: tail + rotate (v0.59.0). |
| [`soup autopilot`](#soup-autopilot) | Autopilot: das modelo+datos+objetivo y Soup elige los hiperparámetros óptimos. |
| [`soup bench`](#soup-bench) | Benchmark de un modelo: velocidad de inferencia (infer, el valor por defecto) o entrenamiento (train). |
| [`soup bom`](#soup-bom) | Emisor de ML-BOM CycloneDX + lista de materiales de IA SPDX (v0.59.0). |
| [`soup build`](#soup-build) | DAG estilo dbt para SFT: valida, planifica y materializa las transformaciones del dataset. |
| [`soup can`](#soup-can) | Soup Cans: empaqueta/inspecciona/verifica/bifurca artefactos .can compartibles. |
| [`soup card`](#soup-card) | Genera una model card de HuggingFace desde una entrada del registro. |
| [`soup chat`](#soup-chat) | Chatea en la terminal con un modelo ajustado. |
| [`soup ci`](#soup-ci) | CI de fine-tuning: inicializa un workflow de compuerta de PR. |
| [`soup compile`](#soup-compile) | Compila un programa de prompts DSPy / GEPA contra una suite de evaluación (v0.68.0). |
| [`soup compile-tools`](#soup-compile-tools) | Compila / optimiza esquemas + descripciones de herramientas (v0.68.0). |
| [`soup completions`](#soup-completions) | Emite un script de autocompletado para bash / zsh / fish. Úsalo con `eval "$(soup completions bash)"` (v0.64.0). |
| [`soup cost`](#soup-cost) | Estima el costo de entrenamiento en USD entre proveedores de nube. |
| [`soup data`](#soup-data) | Herramientas de datasets: inspect, convert, merge, dedup, validate, stats. |
| [`soup delinearize-llama4`](#soup-delinearize-llama4) | Delineariza los pesos de expertos fusionados de Llama 4 a forma 3-D. |
| [`soup deploy`](#soup-deploy) | Despliega modelos: integración con Ollama (deploy, list, remove). |
| [`soup diagnose`](#soup-diagnose) | Boleta de resultados post-entrenamiento: forgetting / refusal / format / mode_collapse / memorization / contamination (v0.56.0). |
| [`soup diff`](#soup-diff) | Compara lado a lado las salidas de dos modelos sobre los mismos prompts. |
| [`soup distill-prompt`](#soup-distill-prompt) | Destila trazas cargadas de prompts en un plan de FT pequeño (v0.68.0). |
| [`soup doctor`](#soup-doctor) | Revisa las dependencias del sistema, la GPU y la compatibilidad. |
| [`soup draft`](#soup-draft) | Entrena + mide un modelo draft de decodificación especulativa: destila tu modelo ajustado en un draft diminuto y luego sírvelo con --auto-spec (v0.71.33). |
| [`soup drift-alarm`](#soup-drift-alarm) | Alarma de deriva de KL móvil sobre la distribución de tokens de salida con webhook opcional de Slack/Discord (v0.63.0). |
| [`soup edit`](#soup-edit) | Edición de conocimiento (ROME / MEMIT / AlphaEdit) - parchea hechos sin reentrenar (v0.61.0). |
| [`soup env`](#soup-env) | Lockfile de entorno hermético + detección de deriva de ABI (v0.64.0). |
| [`soup eval`](#soup-eval) | Evalúa modelos: benchmarks, evaluaciones personalizadas, juez LLM, leaderboard. |
| [`soup expect`](#soup-expect) | Corre una suite de expectativas contra un dataset JSONL. |
| [`soup export`](#soup-export) | Exporta un modelo a formato GGUF, ONNX, TensorRT-LLM, AWQ, GPTQ o TorchAO. |
| [`soup fetch`](#soup-fetch) | Descarga una configuración de ejemplo lista para editar desde el catálogo incluido. |
| [`soup history`](#soup-history) | Muestra el árbol de linaje de todas las entradas con un nombre dado. |
| [`soup infer`](#soup-infer) | Corre inferencia por lotes sobre un archivo JSONL de prompts. |
| [`soup ingest`](#soup-ingest) | Importador universal de trazas: Langfuse / LangSmith / Helicone / OpenPipe / OTel / OpenAI Stored Completions (v0.63.0). |
| [`soup init`](#soup-init) | Crea una configuración soup.yaml nueva de forma interactiva o desde una plantilla. |
| [`soup iterative-dpo`](#soup-iterative-dpo) | Bucle iterativo de DPO: sample, score, pair y train. |
| [`soup license-advisor`](#soup-license-advisor) | Recomienda una base limpia en licencia para un destino de despliegue (b2c / defense / embedded) + marca el riesgo aguas abajo (v0.64.0). |
| [`soup llama`](#soup-llama) | Proxy a los binarios de llama.cpp (cli / mtmd-cli / gguf-split / server). |
| [`soup local-rl`](#soup-local-rl) | Daemon de flywheel de LLM personal (init / status / record / harvest / train) (v0.68.0). |
| [`soup lock`](#soup-lock) | Lockfile de corrida compartido (write / show / check). |
| [`soup loop`](#soup-loop) | Flywheel de datos: traces -> preference pairs -> DPO -> gate -> despliegue canary -> rollback, todo desde la CLI (v0.58.0). |
| [`soup mcp`](#soup-mcp) | Servidor Model Context Protocol - maneja Soup desde cualquier cliente MCP (v0.71.28). |
| [`soup merge`](#soup-merge) | Fusiona un adaptador LoRA con su modelo base en un modelo completo. |
| [`soup merge-sharded-fsdp-weights`](#soup-merge-sharded-fsdp-weights) | Consolida los archivos de shard de FSDP en un único archivo safetensors. |
| [`soup migrate`](#soup-migrate) | Importa una configuración desde LLaMA-Factory, Axolotl o un notebook de Unsloth. |
| [`soup monitor`](#soup-monitor) | Monitor de GPU en vivo: Util / Temp / VRAM / Power por GPU. |
| [`soup plan`](#soup-plan) | Renderiza un plan de entrenamiento previo al vuelo (cost / ETA / SHA / VRAM) y escribe soup.tfstate para que lo consulte `soup apply` (v0.64.0). |
| [`soup plugins`](#soup-plugins) | Descubre, lista, habilita y deshabilita plugins de Soup. |
| [`soup probe`](#soup-probe) | Sondas de activación: defección de sleeper-agent / honesty / misuse / interferencia por pares / dif de características SAE / probe pack (v0.66.0, truth+harm v0.71.8). |
| [`soup profile`](#soup-profile) | Estima los requisitos de memoria, velocidad y GPU ANTES de entrenar. |
| [`soup prune-prompt`](#soup-prune-prompt) | Detecta + quita un prefijo de system-prompt compartido en los datos de entrenamiento para que el modelo FT lo internalice (v0.63.0). |
| [`soup push`](#soup-push) | Sube un modelo entrenado a HuggingFace Hub (u otro hub). |
| [`soup quantize`](#soup-quantize) | Cuantiza un modelo - alias ergonómico de `soup export --format <to>`. |
| [`soup quickstart`](#soup-quickstart) | Corre una demo completa: crea datos de muestra, configuración y entrena. |
| [`soup ra-dit`](#soup-ra-dit) | Orquestador de dos etapas RA-DIT: retriever -> generator (v0.71.10). |
| [`soup recipes`](#soup-recipes) | Configuraciones listas para usar: list, show, use, search de recetas para modelos populares. |
| [`soup registry`](#soup-registry) | Model Registry: push, list, show, diff, search, promote, delete. |
| [`soup reward`](#soup-reward) | Sintetiza un verificador de reward determinista a partir de salidas de referencia, con un informe de calibración que rechaza verificadores degenerados (v0.71.40). |
| [`soup rewind`](#soup-rewind) | Nombra las filas del dataset detrás de un pico de loss, desde la caja negra de vuelo del entrenamiento. |
| [`soup runs`](#soup-runs) | Seguimiento de experimentos: list, show, compare de corridas. |
| [`soup serve`](#soup-serve) | Inicia un servidor de inferencia local con API compatible con OpenAI. |
| [`soup ship`](#soup-ship) | Veredicto SHIP / DON'T SHIP tras el fine-tuning: victoria en la tarea Y sin olvido catastrófico, fundidos en una sola decisión (v0.71.25). |
| [`soup shrink`](#soup-shrink) | Poda en profundidad de un modelo (elimina el bloque de capas contiguo menos importante por distancia angular residual) + distill-heal opcional (v0.71.29). |
| [`soup spectrum`](#soup-spectrum) | Escaneo de SNR de Spectrum para entrenamiento dirigido (v0.71.23). |
| [`soup steer`](#soup-steer) | Steering de activaciones (CAA / ITI / RepE) - intervención en tiempo de inferencia sin reentrenar (v0.62.0). |
| [`soup sweep`](#soup-sweep) | Corre un barrido de hiperparámetros: búsqueda en grid o aleatoria sobre los parámetros de entrenamiento. |
| [`soup tokenizer`](#soup-tokenizer) | Herramientas de tokenizer: entrena un tokenizer BPE desde JSONL (v0.53.9). |
| [`soup train`](#soup-train) | Inicia el entrenamiento desde una configuración soup.yaml. |
| [`soup tui`](#soup-tui) | Panel de terminal a pantalla completa: lista corridas, inspecciona métricas, sigue los logs. |
| [`soup tunability`](#soup-tunability) | Entrena con sonda de 6-10 bases pequeñas sobre una porción reservada + reporta la frontera de Pareto de (delta de evaluación, costo de entrenamiento, licencia) (v0.64.0). |
| [`soup ui`](#soup-ui) | Lanza la interfaz web de Soup. |
| [`soup version`](#soup-version) | Muestra la versión de la CLI de Soup. |
| [`soup why`](#soup-why) | Explica en lenguaje llano las anomalías de entrenamiento de una corrida. |

## `soup ab`

Arnés A/B secuencial mSPRT de dos caras sobre latency / judge_score / retry_rate: informa si el tratamiento es mejor o peor, válido bajo re-ejecución tras cada fila nueva (v0.63.0).

Uso: `soup ab [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | JSONL con {arm, &lt;metric>} por fila. |
| `--metric` | text |  | yes | Métrica: latency \| judge_score \| retry_rate. |
| `--alpha` | float | `0.05` |  | Tasa de error (0, 1) de ambos veredictos, mantenida para una re-ejecución de la prueba tras cada fila nueva, con cualquier número de filas: sin diferencia la prueba termina en reject_h0, y con una diferencia real de --effect-size o más en accept_h0, cada uno en como máximo esta proporción de las corridas. |
| `--beta` | float |  |  | Retirada (#1418): se ignora, con una advertencia. Ahora --alpha también acota los veredictos accept_h0 incorrectos. |
| `--effect-size` | float | `0.1` |  | Diferencia mínima detectable entre medias. |
| `--slack-url` | text |  |  | URL de webhook de Slack opcional — se envía por POST en una decisión reject_h0 / accept_h0 (no en continue). Validada contra SSRF. |
| `--discord-url` | text |  |  | URL de webhook de Discord opcional — se envía por POST en una decisión reject_h0 / accept_h0 (no en continue). Validada contra SSRF. |

## `soup adapters`

Gestión de adaptadores: list, info, compare de adaptadores LoRA.

Uso: `soup adapters <command> [ARGS]...`

Subcomandos:

- [`arithmetic`](#soup-adapters-arithmetic)
- [`audit`](#soup-adapters-audit)
- [`bisect`](#soup-adapters-bisect)
- [`blame`](#soup-adapters-blame)
- [`branch`](#soup-adapters-branch)
- [`branches`](#soup-adapters-branches)
- [`check`](#soup-adapters-check)
- [`check-safetensors`](#soup-adapters-check-safetensors)
- [`checkout`](#soup-adapters-checkout)
- [`compare`](#soup-adapters-compare)
- [`diff`](#soup-adapters-diff)
- [`info`](#soup-adapters-info)
- [`list`](#soup-adapters-list)
- [`merge`](#soup-adapters-merge)
- [`pr`](#soup-adapters-pr)
- [`scan`](#soup-adapters-scan)
- [`sign`](#soup-adapters-sign)
- [`verify`](#soup-adapters-verify)

### `soup adapters arithmetic`

Aritmética de vectores de tarea sobre adaptadores LoRA (add / scale / negate) (v0.71.34).

Uso: `soup adapters arithmetic [OPTIONS] EXPRESSION`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `EXPRESSION` | text |  | yes | Expresión de vector de tarea, p. ej. "coder + 0.5\*math - toxic". |
| `--adapter` | text, repeatable |  | yes | mapeo name=path para cada adaptador referenciado en la expresión (repeatable). Los nombres usan \[A-Za-z0-9\_.-]. |
| `--output`, `-o` | text |  | yes | Directorio de salida para el adaptador fusionado |
| `--rank` | integer, at least 1 |  |  | Límite de rango por SVD truncada. Si se indica, enruta la fusión por la ruta de concatenación exacta y limita el rango de salida a N vía SVD (funciona para entradas del mismo rango Y de rango mixto). Sin él, las entradas de rango mixto conservan el rango concatenado completo (Σ rᵢ) y las entradas del mismo rango usan la ruta rápida elemento a elemento. |
| `--allow-unscanned` | flag | off |  | Omite la compuerta de escaneo de backdoors (v0.71.2 #192). Por defecto rechaza una entrada cuyo `adapters scan` devuelve FAIL o que no se puede escanear. |
| `--allow-cross-base` | flag | off |  | Permite combinar adaptadores entrenados sobre modelos base distintos. Por defecto se rechaza una discrepancia de modelo base (los vectores de tarea de bases distintas no son comparables). |

### `soup adapters audit`

Comprueba si una corrida terminada hizo lo que pedía su configuración (#762).

Uso: `soup adapters audit [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador |
| `--config`, `-c` | text |  | yes | soup.yaml con el que se inició la corrida |
| `--json` | flag | off |  | Emite JSON legible por máquina |

### `soup adapters bisect`

Búsqueda binaria en un historial de entrenamiento para encontrar el primer checkpoint que falla.

Uso: `soup adapters bisect [OPTIONS] HISTORY...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `HISTORY...` | text, any number |  | yes | Identificadores de checkpoint ordenados (el más antiguo primero). Al menos 2. Ejemplo: `soup adapters bisect ckpt-100 ckpt-200 ckpt-300 ckpt-400` |
| `--eval-command` | text |  | yes | Plantilla de comando de shell; '{ckpt}' se reemplaza con cada id de checkpoint. El código de salida 0 significa PASS, distinto de cero significa FAIL. Ejemplo: 'soup eval custom --model {ckpt} --tasks tasks.jsonl' |
| `--plan-only` | flag | off |  | Imprime el plan y sale sin correr la búsqueda binaria |
| `--output`, `-o` | text |  |  | Escribe el resultado JSON en la ruta |

### `soup adapters blame`

Atribuye el movimiento de pesos a filas del dataset mediante influencia DataInf (v0.66.0).

Uso: `soup adapters blame [OPTIONS] ADAPTER_DIR`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER_DIR` | text |  | yes | Ruta al adaptador entrenado |
| `--dataset` | text |  | yes | JSONL de entrenamiento sobre el que se construyó el adaptador |
| `--layer` | text |  | yes | Capa a atribuir (p. ej. q_proj.7) |
| `--budget` | text | `4h` |  | Presupuesto de tiempo de reloj (p. ej. 4h, 30m) |
| `--shards` | integer, 2 to 100 | `10` |  | Número de shards del dataset para leave-one-out |
| `--top-k` | integer, 1 to 10000 | `50` |  | Número de filas influyentes principales a reportar (v0.66.0) |
| `--plan-only` | flag | off |  | Imprime el plan y sale sin correr |
| `--output`, `-o` | text |  |  | Escribe el resultado JSON de blame en la ruta (v0.66.0) |

### `soup adapters branch`

Captura el estado de un entorno de entrenamiento como una rama comparable (v0.57.0).

Uso: `soup adapters branch [OPTIONS] NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre de la rama (alfanumérico + .\_-) |
| `--config`, `-c` | text |  |  | Ruta a soup.yaml (obligatorio salvo con --from-registry) |
| `--base` | text |  |  | Id del modelo base (obligatorio salvo con --from-registry) |
| `--dataset` | text |  |  | Ruta del dataset de entrenamiento (opcional) |
| `--attach-to-registry` | text |  |  | Id/ref de entrada del registro al que adjuntar este puntero de rama como artefacto branch_ref + vinculación vía registry_entry_id (v0.71.4). |
| `--from-registry` | text |  |  | Deriva config + base_model + hash del dataset desde una entrada del registro en lugar de --config/--base (v0.71.4). |

### `soup adapters branches`

Lista todas las ramas capturadas (v0.57.0).

Uso: `soup adapters branches`

No toma ninguna opción ni argumento.

### `soup adapters check`

Audita si un adaptador está entrenado: delta de Frobenius, B en cero, claves internas (#1721).

Uso: `soup adapters check [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador |
| `--json` | flag | off |  | Emite un documento JSON legible por máquina |

### `soup adapters check-safetensors`

Rechaza pesos pickle / PyTorch-classic en la frontera (v0.60.0).

Uso: `soup adapters check-safetensors [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador / modelo |
| `--strict` | flag | off |  | Sale con 3 ante cualquier archivo de pesos inseguro (pickle / PyTorch-classic) |

### `soup adapters checkout`

Restaura la configuración de una rama capturada en el directorio actual (v0.57.0).

Uso: `soup adapters checkout [OPTIONS] NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre de la rama a pasar a |
| `--output`, `-o` | text | `soup.yaml` |  | Dónde escribir la configuración restaurada |

### `soup adapters compare`

Compara dos adaptadores LoRA lado a lado.

Uso: `soup adapters compare ADAPTER1 ADAPTER2`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER1` | text |  | yes | Ruta al primer adaptador |
| `ADAPTER2` | text |  | yes | Ruta al segundo adaptador |

### `soup adapters diff`

Dif de Frobenius de ΔW por capa + deriva del rango efectivo (v0.57.0).

Uso: `soup adapters diff [OPTIONS] ADAPTER_A ADAPTER_B`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER_A` | text |  | yes | Ruta al primer adaptador |
| `ADAPTER_B` | text |  | yes | Ruta al segundo adaptador |
| `--top-k` | integer, 1 to 200 | `10` |  | Número de proyecciones más cambiadas a reportar |
| `--format` | text | `table` |  | Formato de salida: table \| json \| markdown |
| `--output`, `-o` | text |  |  | Escribe el informe en un archivo (solo json/markdown) |

### `soup adapters info`

Muestra metadatos detallados de un adaptador LoRA.

Uso: `soup adapters info ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador |

### `soup adapters list`

Escanea un directorio en busca de adaptadores LoRA y los lista.

Uso: `soup adapters list [DIRECTORY]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DIRECTORY` | text | `.` |  | Directorio donde buscar adaptadores (recursivo) |

### `soup adapters merge`

Fusiona adaptadores LoRA vía linear / ties / dare / svd (v0.57.0, compuerta de licencia v0.60.0).

Uso: `soup adapters merge [OPTIONS] ADAPTERS...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTERS...` | text, any number |  | yes | Dos o más rutas de adaptadores a fusionar |
| `--output`, `-o` | text |  | yes | Directorio de salida para el adaptador fusionado |
| `--strategy` | text | `linear` |  | linear \| ties \| dare \| svd \| cmaes |
| `--weights` | text |  |  | Pesos separados por comas (por defecto: iguales) |
| `--density` | float | `0.2` |  | Densidad de recorte para ties/dare en (0, 1] |
| `--seed` | integer | `0` |  | Semilla aleatoria para dare / cmaes |
| `--rank` | integer |  |  | Rango SVD (solo estrategia svd) |
| `--eval` | text |  |  | Ruta a la suite de evaluación (obligatoria para --strategy cmaes). La usa el bucle evolutivo para puntuar las fusiones candidatas. |
| `--budget` | text | `1h` |  | Presupuesto de tiempo de reloj para cmaes (60s..24h, p. ej. 1h, 30m) |
| `--population` | integer, 2 to 256 | `8` |  | Tamaño de población de cmaes por generación |
| `--max-generations` | integer, 1 to 10000 | `20` |  | Límite de generaciones de cmaes |
| `--license` | text, repeatable |  |  | Id de licencia SPDX por adaptador (repeatable; mismo orden que las entradas). v0.60.0 rechaza fusiones con licencias en conflicto salvo que se indique --license-override &lt;reason>. |
| `--license-override` | text |  |  | Justificación de texto libre (>=8 caracteres) para fusionar a pesar de un conflicto de licencia. Se registra en el registro de auditoría para revisión legal (v0.71.2). |
| `--allow-unscanned` | flag | off |  | Omite la compuerta de escaneo de backdoors v0.71.2. Por defecto `adapters merge` se niega a fusionar una entrada cuyo `adapters scan` devuelve FAIL (o que no se puede escanear). |
| `--canary` | text |  |  | JSON de suite canary para calcular un veredicto en vivo OK/MINOR/MAJOR del adaptador fusionado frente a la primera entrada (v0.71.4 #172). Forma: {"baseline_scores": \[...], "candidate_scores": \[...]}. |
| `--strict-verdict` | flag | off |  | Sale con 2 cuando el veredicto canary es MAJOR (compuerta de CI). |

### `soup adapters pr`

Renderiza un PR al estilo de GitHub para un adaptador.

Uso: `soup adapters pr [OPTIONS] TITLE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `TITLE` | text |  | yes | Título corto del PR (p. ej. 'add-support-tone') |
| `--base-sha` | text |  | yes | SHA de 64 hex del modelo base |
| `--adapter` | text |  | yes | Ruta al adaptador candidato |
| `--eval` | text |  |  | Ruta a los deltas de evaluación en JSON: lista de {"metric": ..., "baseline": ..., "candidate": ...} |
| `--samples` | text |  |  | Ruta a los difs de muestras en JSON: lista de {"prompt": ..., "baseline_output": ..., "candidate_output": ...} |
| `--dataset-diff` | text |  |  | Ruta a un archivo de texto que contiene el dif del dataset |
| `--output`, `-o` | text |  |  | Escribe el Markdown renderizado en la ruta (por defecto: stdout) |
| `--format`, `-f` | text | `markdown` |  | markdown \| json |
| `--push` | text |  |  | Publica el Markdown renderizado como comentario en un PR de GitHub: owner/repo#N (p. ej. MakazhanAlpamys/Soup#42). Autenticación vía GITHUB_TOKEN / GH_TOKEN (v0.71.4). |

### `soup adapters scan`

Escaneo espectral de backdoors sobre los pesos del adaptador LoRA (v0.60.0).

Uso: `soup adapters scan [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador |
| `--format` | text | `text` |  | Formato de salida: text \| json |

### `soup adapters sign`

Calcula el manifiesto + escribe ``.soup-signature.json`` (v0.60.0, ed25519 v0.71.2).

Uso: `soup adapters sign [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador |
| `--backend` | text | `unsigned` |  | Backend de firma: unsigned \| ed25519 \| sigstore (OIDC sin clave) |
| `--key` | text |  |  | Ruta PEM de la clave privada ed25519 (o define SOUP_SIGNING_KEY). |
| `--generate-key` | text |  |  | Genera un par de claves ed25519 nuevo, persiste la clave privada aquí (PEM, 0600) y firma con ella. |
| `--interactive-oidc` | flag | off |  | Permite el OIDC del navegador de Sigstore cuando no existe una credencial ambiental. Desactivado por defecto para que los runners headless fallen en lugar de colgarse. |

### `soup adapters verify`

Verifica ``.soup-signature.json`` contra los archivos actuales (v0.60.0, ed25519 v0.71.2).

Uso: `soup adapters verify [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Ruta al directorio del adaptador |
| `--strict` | flag | off |  | Sale con 3 ante cualquier fallo de verificación (apto para CI) |
| `--public-key` | text |  |  | PEM de la clave pública ed25519 de confianza. Cuando se define, la clave de firma embebida debe coincidir con ella y exige una firma ed25519 (autenticación genuina, no solo consistencia). |
| `--cert-identity` | text |  |  | Identidad (SAN) del certificado de Sigstore de confianza. Requiere --cert-oidc-issuer y es obligatoria para adaptadores firmados con sigstore. |
| `--cert-oidc-issuer` | text |  |  | Emisor OIDC de confianza. Obligatorio junto con --cert-identity. |

## `soup advise`

Decisión previa al vuelo: PROMPT_ENG / RAG / SFT / DPO / GRPO. Ejecútalo ANTES de gastar 8 horas en una GPU (v0.54.0).

Uso: `soup advise <command> [ARGS]...`

Subcomandos:

- [`compare`](#soup-advise-compare)
- [`explain`](#soup-advise-explain)
- [`run`](#soup-advise-run)

### `soup advise compare`

Muestra veredictos anteriores de advise_history.jsonl.

Uso: `soup advise compare [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--limit` | integer, 1 to 1000 | `20` |  | Cuántos veredictos recientes mostrar. |

### `soup advise explain`

Imprime la rúbrica, los pesos y el rastro de evidencia del último veredicto.

Uso: `soup advise explain`

No toma ninguna opción ni argumento.

### `soup advise run`

Renderiza un Veredicto para el dataset proporcionado.

Uso: `soup advise run [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Ruta al dataset JSONL sobre el que aconsejar (debe permanecer dentro del cwd). |
| `--goal`, `-g` | text |  |  | Declaración de objetivo en una línea: p. ej. "haz más conciso nuestro chatbot". Afina la clasificación de tareas. |
| `--probe` | flag | off |  | También corre una sonda de ROI (zero/few-shot + baseline RAG + LoRA de N pasos). Heurística por defecto; indica --probe-model &lt;id> para una sonda EN VIVO que carga el modelo + entrena LoRA sobre una porción reservada diminuta. |
| `--probe-model` | text |  |  | Id del modelo base para una sonda EN VIVO (p. ej. HuggingFaceTB/SmolLM2-135M). Cuando se define, la sonda carga este modelo, puntúa zero/few-shot, entrena LoRA y mide la proximidad al modelo base. Implica --probe. |
| `--probe-device` | text |  |  | Dispositivo para la sonda en vivo (cuda / cpu). Se detecta automáticamente si se omite. |
| `--record` | flag | off |  | Añade este veredicto a advise_history.jsonl con accepted=True. Omite --record para saltarlo. Se guarda en \~/.soup/. |
| `--notes` | text | `""` |  | Notas opcionales adjuntas al veredicto registrado (&lt;=4096 caracteres). |

## `soup agent`

Agent Forge: spec -> dataset de tool-calling / train / eval (v0.46.0).

Uso: `soup agent <command> [ARGS]...`

Subcomandos:

- [`eval`](#soup-agent-eval)
- [`synth`](#soup-agent-synth)
- [`train`](#soup-agent-train)

### `soup agent eval`

Puntúa las tool-calls predichas contra el catálogo de herramientas de la spec.

Uso: `soup agent eval [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--spec`, `-s` | text |  | yes | Spec de API (dentro del cwd). |
| `--predictions`, `-p` | text |  | yes | JSONL de salidas del modelo, una por línea, con como mínimo {tool: &lt;name>, arguments: {...}}. |
| `--sandbox` | flag | off |  | Ejecuta cada predicción que pasa la heurística contra un mock en sandbox del endpoint (v0.71.18 #110): clasifica ok / tool_error / timeout / arg_error. El aislamiento fuerte es solo POSIX (aislamiento reducido en Windows; se imprime un aviso amable). |

### `soup agent synth`

Analiza una spec de API y sintetiza un dataset SFT de tool-calling.

Uso: `soup agent synth [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--spec`, `-s` | text |  | yes | Ruta a la spec de OpenAPI / MCP / GraphQL (YAML o JSON, dentro del cwd). |
| `--output`, `-o` | text | `agent_dataset.jsonl` |  | Dónde escribir el dataset JSONL sintetizado (dentro del cwd). |
| `--kind`, `-k` | text |  |  | Sobrescritura del tipo de spec: openapi \| mcp \| graphql. Se detecta automáticamente si se omite. |
| `--examples-per-endpoint`, `-n` | integer, 1 to 32 | `1` |  | Número de filas sintéticas a emitir por endpoint. |

### `soup agent train`

Wrapper de un solo disparo: synth + invocación planificada de soup train (impresa).

Uso: `soup agent train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--spec`, `-s` | text |  | yes | Spec de API (dentro del cwd). |
| `--base`, `-b` | text |  | yes | Id de repo HF del modelo base a ajustar. |
| `--dataset-out` | text | `agent_dataset.jsonl` |  | Dónde escribe su dataset el paso de synth (dentro del cwd). |
| `--output-dir` | text | `./agent_train_output` |  | Dónde guardará los checkpoints la corrida planificada de soup train. |
| `--examples-per-endpoint`, `-n` | integer, 1 to 32 | `4` |  |  |

## `soup airgap-bundle`

Construye un tarball firmado con modelo + datasets + wheels + kernels (v0.60.0).

Uso: `soup airgap-bundle [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--output`, `-o` | text |  | yes | Ruta del tarball de salida (contenida en el cwd) |
| `--model` | text |  | yes | Ruta al directorio del modelo |
| `--dataset` | text, repeatable |  |  | Directorio de datasets (repeatable) |
| `--wheel` | text, repeatable |  |  | Directorio de wheels (repeatable) |
| `--kernel` | text, repeatable |  |  | Directorio de CUDA / kernels (repeatable) |
| `--bundle-size-cap` | float | `100.0` |  | Límite en GiB (por defecto 100). La construcción se aborta cuando se supera. |
| `--repro-receipt` | text |  |  | Incrusta un JSON de recibo de reproducibilidad (de `soup train --repro-receipt`) en el bundle. Se detecta automáticamente desde &lt;model>/repro-receipt.json cuando no se indica. v0.71.3 #188. |

## `soup apple-adapter`

Convierte / firma adaptadores HF / MLX / Apple FoundationModels (v0.68.0).

Uso: `soup apple-adapter [OPTIONS] SOURCE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SOURCE` | text |  | yes | Directorio del adaptador de origen |
| `--direction` | text | `hf-to-mlx` |  | Permitidos: hf-to-apple, hf-to-mlx, mlx-to-apple, mlx-to-hf |
| `--output`, `-o` | text |  | yes | Directorio de salida |
| `--sign` / `--no-sign` | flag | off |  | Firma el adaptador convertido vía firma con raíz Merkle de v0.60. |
| `--plan-only` | flag | off |  | Renderiza el plan + sale con 0 (sin conversión en vivo). |

## `soup apply`

Ejecuta la corrida de entrenamiento planificada, negándose ante una deriva entre soup.yaml y soup.tfstate (v0.64.0).

Uso: `soup apply [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta a soup.yaml. |
| `--state` | text | `soup.tfstate` |  | Ruta al archivo de estado escrito por `soup plan`. |
| `--dry-run` | flag | off |  | Verifica la deriva y sale sin invocar el entrenamiento. |

## `soup attest`

Atestaciones In-toto + SLSA-3 por etapa de Soup Can (v0.59.0).

Uso: `soup attest <command> [ARGS]...`

Subcomandos:

- [`emit`](#soup-attest-emit)
- [`verify`](#soup-attest-verify)

### `soup attest emit`

Emite una atestación in-toto/SLSA-3 por etapa.

Uso: `soup attest emit [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--stage` | text |  | yes | Etapa: extract / train / eval / export / publish. |
| `--subject` | text |  | yes | Nombre del artefacto. |
| `--sha` | text |  | yes | SHA-256 de 64 hex del artefacto. |
| `--builder` | text | `soup-cli` |  | Identidad del builder (por defecto: soup-cli). |
| `--invocation` | text |  |  | Marcador de invocación de forma libre (p. ej. línea de comandos). |
| `--sign` | text | `unsigned` |  | Backend de firma: unsigned (por defecto) \| ed25519 \| sigstore (OIDC sin clave + Fulcio/Rekor). |
| `--key` | text |  |  | Ruta PEM de la clave privada ed25519 (o define SOUP_SIGNING_KEY). |
| `--output`, `-o` | text |  |  | Ruta del archivo de salida (contenida en el cwd). |
| `--attach-to-registry` | text |  |  | Adjunta la declaración de atestación emitida a un id de entrada del registro (necesita --output). |
| `--interactive-oidc` | flag | off |  | Permite el OIDC del navegador de Sigstore cuando no existe una credencial ambiental. Desactivado por defecto para que los runners headless fallen en lugar de colgarse. |

### `soup attest verify`

Verifica una atestación firmada con ed25519 o Sigstore.

Uso: `soup attest verify [OPTIONS] STATEMENT`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `STATEMENT` | text |  | yes | Ruta al JSON de la declaración in-toto. |
| `--signature`, `-s` | text |  | yes | Ruta al sidecar JSON .sig escrito por `attest emit --sign`. |
| `--public-key` | text |  |  | PEM de la clave pública ed25519 de confianza. Cuando se define, la clave embebida debe coincidir con ella (autenticación genuina). |
| `--cert-identity` | text |  |  | Identidad (SAN) del certificado de Sigstore de confianza. Requiere --cert-oidc-issuer y es obligatoria para atestaciones firmadas con sigstore. |
| `--cert-oidc-issuer` | text |  |  | Emisor OIDC de confianza. Obligatorio junto con --cert-identity. |

## `soup audit-log`

Registro de auditoría JSONL con forma HIPAA/SOC2: tail + rotate (v0.59.0).

Uso: `soup audit-log <command> [ARGS]...`

Subcomandos:

- [`rotate`](#soup-audit-log-rotate)
- [`tail`](#soup-audit-log-tail)

### `soup audit-log rotate`

Fuerza una pasada de rotación en el límite actual.

Uso: `soup audit-log rotate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--path` | text |  |  | Sobrescribe la ruta del registro de auditoría. |
| `--cap-mb` | integer, 1 to 10000 | `100` |  | Límite de rotación en MiB. |

### `soup audit-log tail`

Muestra los registros de auditoría más recientes.

Uso: `soup audit-log tail [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--limit` | integer | `50` |  | Máximo de registros a mostrar (1-100000). |
| `--path` | text |  |  | Sobrescribe la ruta del registro de auditoría (por defecto: \$SOUP_AUDIT_LOG_PATH o \~/.soup/audit.jsonl). |
| `--json` | flag | off |  | Emite JSONL crudo en lugar de una tabla. |

## `soup autopilot`

Autopilot: das modelo+datos+objetivo y Soup elige los hiperparámetros óptimos.

Uso: `soup autopilot [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  |  | Modelo base (id de modelo HF) |
| `--data`, `-d` | text |  |  | Ruta del dataset (JSONL) |
| `--goal`, `-g` | text |  |  | Objetivo: chat \| reasoning \| code \| classification \| tool-calling \| alignment \| domain-adapt |
| `--gpu-budget` | text | `""` |  | Presupuesto de VRAM, p. ej. 24GB (por defecto: detección automática) |
| `--output`, `-o` | text | `soup.yaml` |  | Ruta de la configuración de salida |
| `--dry-run` | flag | off |  | Muestra las decisiones pero no escribe la configuración |
| `--yes`, `-y` | flag | off |  | Salta los avisos de confirmación |

## `soup bench`

Benchmark de un modelo: velocidad de inferencia (infer, el valor por defecto) o entrenamiento (train).

Uso: `soup bench <command> [ARGS]...`

Subcomandos:

- [`infer`](#soup-bench-infer)
- [`train`](#soup-bench-train)

### `soup bench infer`

Corre un benchmark de inferencia (velocidad y memoria) sobre un modelo cargado.

Uso: `soup bench infer [OPTIONS] MODEL`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `MODEL` | text |  | yes | Ruta al modelo (adaptador LoRA o modelo completo) a medir |
| `--base`, `-b` | text |  |  | Modelo base para el adaptador LoRA (se detecta automáticamente si no se define) |
| `--max-tokens` | integer | `128` |  | Máximo de tokens a generar por prompt |
| `--num-prompts`, `-n` | integer | `3` |  | Número de prompts a correr para promediar (se ignora cuando --prompts-file está definido) |
| `--prompts-file` | text |  |  | Ruta a un archivo de prompts personalizado (.txt o .jsonl) |
| `--p50` | flag | off |  | Imprime la latencia p50 (mediana) por prompt. v0.53.9 #26. |
| `--p95` | flag | off |  | Imprime la latencia p95 por prompt. v0.53.9 #26. |
| `--backend` | text | `auto` |  | Sugerencia de backend de inferencia: auto (por defecto) \| transformers \| mlx. v0.53.9 #28. |
| `--cuda-graphs` | flag | off |  | Decodificación experimental con CUDA graphs para modelos Qwen2/Llama residentes y sin cuantizar. |

### `soup bench train`

Entrena durante un número fijo de pasos e informa tiempos, memoria y tokens -- fallando cuando el modelo no estaba entrenando de verdad.

Uso: `soup bench train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Configuración a entrenar |
| `--steps` | integer, at least 1 | `20` |  | Pasos del optimizador a correr |
| `--warmup` | integer, at least 0 | `3` |  | Pasos iniciales medidos pero excluidos de la medición de tiempos |
| `--output`, `-o` | text | `bench-train.json` |  | Dónde escribir el informe JSON |

## `soup bom`

Emisor de ML-BOM CycloneDX + lista de materiales de IA SPDX (v0.59.0).

Uso: `soup bom <command> [ARGS]...`

Subcomandos:

- [`emit`](#soup-bom-emit)

### `soup bom emit`

Emite un BOM CycloneDX + SPDX a partir de los SHAs proporcionados por la CLI.

Uso: `soup bom emit [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--name` | text |  | yes | Nombre del modelo / adaptador. |
| `--version` | text | `0.1.0` |  | Cadena de versión del modelo. |
| `--base-model` | text |  | yes | Id de repo HF del modelo base. |
| `--base-sha` | text |  | yes | SHA-256 del modelo base. |
| `--config-sha` | text |  | yes | SHA-256 de la configuración soup.yaml resuelta. |
| `--data-sha` | text |  |  | SHA-256 del dataset de entrenamiento. |
| `--task` | text | `sft` |  | Tarea de entrenamiento (sft / dpo / grpo / ...). |
| `--license` | text |  |  | Licencia: un id SPDX (p. ej. Apache-2.0, MIT; sin distinguir mayúsculas), una expresión SPDX (Apache-2.0 OR MIT) o cualquier otro nombre. |
| `--format`, `-f` | text | `cyclonedx` |  | Formato del BOM de salida: cyclonedx \| spdx \| both. |
| `--output`, `-o` | text |  |  | Ruta del archivo de salida (contenida en el cwd). Cuando --format=both, esto es el prefijo y Soup escribe &lt;prefix>.cdx.json + &lt;prefix>.spdx.json. |
| `--energy`, `-e` | text |  |  | Ruta al JSON de medición de energía. |
| `--attach-to-registry` | text |  |  | Adjunta el/los archivo(s) BOM emitidos a un id de entrada del registro (necesita --output). |

## `soup build`

DAG estilo dbt para SFT: valida, planifica y materializa las transformaciones del dataset.

Uso: `soup build [OPTIONS] CONFIG`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CONFIG` | text |  | yes | Ruta al manifiesto de build (YAML) |
| `--output-dir`, `-o` | text | `build_out` |  | Directorio (dentro del cwd) para el JSONL del modelo materializado + el almacén de estado. |
| `--dry-run` | flag | off |  | Valida el manifiesto + imprime el plan topológico; no ejecuta. |

## `soup can`

Soup Cans: empaqueta, inspecciona, verifica y bifurca artefactos .can compartibles.

Uso: `soup can <command> [ARGS]...`

Subcomandos:

- [`fork`](#soup-can-fork)
- [`inspect`](#soup-can-inspect)
- [`pack`](#soup-can-pack)
- [`publish`](#soup-can-publish)
- [`run`](#soup-can-run)
- [`verify`](#soup-can-verify)

### `soup can fork`

Bifurca un can con modificaciones de configuración y vuelve a empaquetarlo.

Uso: `soup can fork [OPTIONS] SOURCE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SOURCE` | text |  | yes | Ruta al archivo .can de origen |
| `--out`, `-o` | text |  | yes | Ruta de salida del .can bifurcado |
| `--modify` | text, repeatable |  |  | Modificación como 'training.lr=5e-5' (repetible) |
| `--author` | text | `unknown` |  |  |

### `soup can inspect`

Previsualiza el manifiesto de un archivo .can sin extraerlo.

Uso: `soup can inspect PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo .can |

### `soup can pack`

Empaqueta una entrada del registro en un archivo .can compartible.

Uso: `soup can pack [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--entry-id` | text |  | yes | Id, prefijo o name:tag de la entrada del registro |
| `--out`, `-o` | text |  | yes | Ruta del .can de salida |
| `--author` | text | `unknown` |  | Identificador del autor |
| `--description` | text | `""` |  | Descripción de formato libre |
| `--attest` | text, repeatable |  |  | Inserta un JSON de in-toto Statement en el manifiesto del can (v3, repetible). Se valida la estructura de cada archivo. v0.71.3. |

### `soup can publish`

Publica un archivo .can en HF Hub como dataset.

Uso: `soup can publish [OPTIONS] CAN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CAN_PATH` | text |  | yes | Ruta al archivo .can |
| `--hf-hub` | text |  | yes | Repo de dataset de HF Hub (p. ej., 'me/my-recipe-can') |

| `--private` | flag | off |  |  |
| `--message`, `-m` | text |  |  | Mensaje de commit (solo la primera línea, limitado a 200 caracteres) |

### `soup can run`

Ejecuta un can de extremo a extremo: extrae -> entrena (-> implementación opcional).

Uso: `soup can run [OPTIONS] CAN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CAN_PATH` | text |  | yes | Ruta al archivo .can |
| `--yes`, `-y` | flag | off |  | Omite el panel de confirmación de seguridad (entrena y obtiene datos automáticamente) |
| `--deploy` | flag | off |  | Ejecuta los deploy_targets del can después de un entrenamiento exitoso |
| `--extract-dir` | text |  |  | Dónde extraer (por defecto: directorio temporal nuevo) |
| `--env-capture` | text |  |  | Escribe el resumen del entorno (pip freeze + información de GPU) en esta ruta |

### `soup can verify`

Verifica el esquema del archivo .can y que se pueda analizar la configuración.

Uso: `soup can verify PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo .can |

## `soup card`

Genera una ficha de modelo de HuggingFace a partir de una entrada del registro.

Uso: `soup card [OPTIONS] REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Id, prefijo o name:tag de una entrada del registro |
| `-o`, `--output` | text | `MODELCARD.md` |  | Ruta del Markdown de salida (debe permanecer bajo el directorio actual) |

## `soup chat`

Conversa en la terminal con un modelo ajustado.

Uso: `soup chat [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Ruta al directorio del adaptador LoRA o al modelo completo |
| `--base`, `-b` | text |  |  | Id del modelo base. Se detecta automáticamente desde adapter_config.json si no se define. |
| `--device` | text |  |  | Dispositivo: cuda, mps, cpu. Se detecta automáticamente si no se define. |
| `--max-tokens` | integer | `512` |  | Número máximo de tokens que se generarán |
| `--temperature`, `-t` | float | `0.7` |  | Temperatura de muestreo (0.0 = determinista, 1.0 = creativo) |
| `--system`, `-s` | text |  |  | Prompt del sistema para la conversación |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--hub` | text | `hf` |  | Hub de origen del modelo base: hf (por defecto) / modelscope / modelers. Los hubs que no son HF requieren el SDK correspondiente; se crea una instantánea del modelo base en una caché dentro del cwd antes de iniciar el chat (v0.53.10 #152). |

## `soup ci`

CI de ajuste fino: inicializa un flujo de trabajo para validar PR.

Uso: `soup ci <command> [ARGS]...`

Subcomandos:

- [`init`](#soup-ci-init)

### `soup ci init`

Escribe un flujo de trabajo de GitHub Actions para validar el ajuste fino.

Uso: `soup ci init [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--data` | text | `./data/train.jsonl` |  | Ruta a los datos de entrenamiento (relativa al repo) |
| `--suite` | text | `expectations.yaml` |  | YAML de la suite de expectativas (relativa al repo) |
| `--evidence` | text | `ship_evidence.json` |  | JSON de evidencia de publicación (relativa al repo) |
| `--python` | text | `3.11` |  | Versión de Python para el ejecutor de CI (p. ej., 3.11) |
| `--branch` | text | `main` |  | Rama que activa el flujo de trabajo |
| `--config` | text |  |  | Vincula la validación de publicación a un soup.yaml confirmado (rechaza evidencia desactualizada) |
| `-o`, `--output` | text | `.github/workflows/soup-gate.yml` |  | Ruta de salida del flujo de trabajo |
| `--force`, `-f` | flag | off |  | Sobrescribe un flujo de trabajo existente |

## `soup compile`

Compila un programa de prompt DSPy / GEPA con una suite de evaluación (v0.68.0).

Uso: `soup compile [OPTIONS] PROGRAM`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PROGRAM` | text |  | yes | Ruta al programa DSPy / GEPA (.py) |
| `--eval` | text |  | yes | Ruta al archivo de la suite de evaluación JSON / JSONL |
| `--optimizer` | text | `mipro` |  | Nombre del optimizador. Permitidos: bootstrap_fewshot, copro, gepa, mipro, textgrad |
| `--max-iters` | integer | `10` |  | Número máximo de iteraciones del optimizador |
| `--output`, `-o` | text | `compiled_program.py` |  | Ruta del programa de salida |
| `--plan-only` | flag | off |  | Muestra el plan resuelto y termina con código 0 (sin compilación activa). |

## `soup compile-tools`

Compila / optimiza esquemas y descripciones de herramientas (v0.68.0).

Uso: `soup compile-tools [OPTIONS] SPEC`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SPEC` | text |  | yes | Ruta a la especificación OpenAPI / MCP / GraphQL |
| `--eval` | text |  | yes | Ruta al JSONL de evaluación de llamadas a herramientas |
| `--optimizer` | text | `textgrad` |  | Permitidos: gepa, textgrad |
| `--output`, `-o` | text | `compiled_tools.json` |  | Ruta del esquema de salida |
| `--plan-only` | flag | off |  | Muestra el plan y termina con código 0 (sin optimización activa). |

## `soup completions`

Emite un script de autocompletado para bash / zsh / fish. Úsalo con `eval "$(soup completions bash)"` (v0.64.0).

Uso: `soup completions SHELL`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SHELL` | text |  | yes | Uno de: bash, zsh, fish. |

## `soup cost`

Calcula el costo del entrenamiento en USD entre proveedores de nube.

Uso: `soup cost [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta al archivo de configuración soup.yaml |
| `--gpu`, `-g` | text |  |  | Filtra por una GPU específica (p. ej., A100, H100, RTX 4090) |
| `--json` | flag | off |  | Emite JSON para usar en scripts |

## `soup data`

Herramientas para datasets: inspección, conversión, fusión, deduplicación, validación y estadísticas.

Uso: `soup data <command> [ARGS]...`

Subcomandos:

- [`active-sample`](#soup-data-active-sample)
- [`augment`](#soup-data-augment)
- [`best-of-n`](#soup-data-best-of-n)
- [`brain-rot`](#soup-data-brain-rot)
- [`canary`](#soup-data-canary)
- [`clean`](#soup-data-clean)
- [`convert`](#soup-data-convert)
- [`decontaminate`](#soup-data-decontaminate)
- [`dedup`](#soup-data-dedup)
- [`demo`](#soup-data-demo)
- [`doctor`](#soup-data-doctor)
- [`download`](#soup-data-download)
- [`educational`](#soup-data-educational)
- [`evolve`](#soup-data-evolve)
- [`filter`](#soup-data-filter)
- [`forge`](#soup-data-forge)
- [`from-traces`](#soup-data-from-traces)
- [`gen-magpie`](#soup-data-gen-magpie)
- [`generate`](#soup-data-generate)
- [`ingest`](#soup-data-ingest)
- [`inspect`](#soup-data-inspect)
- [`langdetect`](#soup-data-langdetect)
- [`lint`](#soup-data-lint)
- [`merge`](#soup-data-merge)
- [`mix`](#soup-data-mix)
- [`persona-mix`](#soup-data-persona-mix)
- [`pii`](#soup-data-pii)
- [`preprocess`](#soup-data-preprocess)
- [`preview`](#soup-data-preview)
- [`push`](#soup-data-push)
- [`recipe`](#soup-data-recipe)
- [`register`](#soup-data-register)
- [`registry`](#soup-data-registry)
- [`review`](#soup-data-review)
- [`sample`](#soup-data-sample)
- [`score`](#soup-data-score)
- [`search`](#soup-data-search)
- [`split`](#soup-data-split)
- [`stats`](#soup-data-stats)
- [`topics`](#soup-data-topics)
- [`toxicity`](#soup-data-toxicity)
- [`unregister`](#soup-data-unregister)
- [`validate`](#soup-data-validate)

### `soup data active-sample`

Muestra los traces de producción más inciertos para revisión humana.

Uso: `soup data active-sample [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | JSONL de filas de traces con campos rm_score o rm_scores. |
| `--output`, `-o` | text | `active_samples.jsonl` |  | JSONL de salida con las filas de mayor incertidumbre (por defecto: active_samples.jsonl). |
| `--budget` | integer | `100` |  | Máximo de filas que se mostrarán para revisión humana (1 - 100_000). |
| `--slack-url` | text |  |  | URL opcional del webhook de Slack: se envía un POST al terminar. Validada contra SSRF. |
| `--discord-url` | text |  |  | URL opcional del webhook de Discord: se envía un POST al terminar. Validada contra SSRF. |

### `soup data augment`

Aumenta un dataset mediante un LLM (reformular / traducir / estilo).

Uso: `soup data augment [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Archivo JSONL de origen |
| `--output`, `-o` | text | `augmented.jsonl` |  | Ruta del JSONL de salida |
| `--strategy`, `-s` | text | `rephrase` |  | Estrategia de aumento: rephrase, translate, style |
| `--provider`, `-p` | text | `ollama` |  | Proveedor de LLM: ollama, anthropic, vllm |
| `--model` | text | `""` |  | Id del modelo del proveedor (por defecto, según el proveedor) |
| `--base-url` | text | `""` |  | URL base del proveedor (ollama/vllm; por defecto, loopback) |
| `--count`, `-c` | integer, 1 to 10 | `2` |  | Multiplicador de aumento (1-10) |
| `--lang` | text | `""` |  | Idiomas de destino separados por comas para translate |
| `--styles` | text | `""` |  | Estilos separados por comas para la estrategia style |
| `--requests-per-minute` | integer, 1 to 600 | `60` |  | Límite de solicitudes por minuto al proveedor |
| `--dedup` | flag | off |  | Deduplica los datos aumentados y los originales |

### `soup data best-of-n`

Muestreo de rechazo Best-of-N: toma N muestras por prompt y un evaluador elige la mejor.

Uso: `soup data best-of-n [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base` | text | `""` |  | Id/ruta del modelo base local del que se tomarán muestras |
| `--prompts` | text | `""` |  | JSONL of {prompt\|messages} rows |
| `--judge` | text | `""` |  | Judge URL (ollama://\|https://\|http://localhost) |
| `--provider` | text | `""` |  | Provider sampler: ollama \| vllm |
| `--model` | text | `""` |  | Id del modelo del muestreador del proveedor |
| `--base-url` | text | `""` |  | URL base del muestreador del proveedor |
| `--n` | integer | `8` |  | Candidates per prompt \[2, 64] |
| `--output`, `-o` | text | `""` |  | JSONL SFT de salida (obligatorio salvo con --plan-only) |
| `--emit-pairs` | text | `""` |  | También escribe pares DPO de ganador/perdedor en este JSONL |
| `--checkpoint` | text | `""` |  | Registro de recuperación (por defecto: &lt;output-or-artifact>.checkpoint.jsonl) |
| `--manifest` | text | `""` |  | Manifiesto de commit (por defecto: &lt;output>.manifest.json) |
| `--resume` | flag | off |  | Reanuda un checkpoint coincidente |
| `--export-candidates` | text | `""` |  | Artefacto JSONL de candidatos solo para muestreo |
| `--candidate-artifact` | text | `""` |  | Artefacto de candidatos sin conexión para materializar |
| `--judgments` | text | `""` |  | JSONL de evaluaciones verificadas sin conexión |
| `--temperature` | float | `1.0` |  | Sampling temp \[0, 2] |
| `--max-new-tokens` | integer | `256` |  | Max new tokens per candidate \[1, 4096] |
| `--device` | text | `""` |  | Local sampler device: cuda \| cpu |
| `--seed` | integer | `0` |  | Semilla de muestreo local |
| `--revision` | text | `""` |  | Revisión fijada del modelo local |
| `--trust-remote-code` | flag | off |  | Permite código de modelo personalizado en --base local |
| `--plan-only` | flag | off |  | Valida e imprime el plan |

### `soup data brain-rot`

Evalúa un dataset para detectar brain-rot según arXiv 2510.13928.

Uso: `soup data brain-rot [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Ruta al dataset JSONL |
| `--strict` | flag | off |  | Termina con código 3 ante un veredicto MAJOR (modo de validación CI) |
| `--max-major-fraction` | float | `0.25` |  | Strict-mode MAJOR-row fraction cap \[0, 1] |
| `--lang` | text | `en` |  | Per-language heuristic bundle: en \| es \| fr \| de \| ru \| auto. Closes issue #234. |

### `soup data canary`

Canarios de dataset (prueba de memorización).

Uso: `soup data canary <command> [ARGS]...`

Subcomandos:

- [`check`](#soup-data-canary-check)
- [`insert`](#soup-data-canary-insert)

#### `soup data canary check`

Comprueba si un modelo memorizó los canarios del manifiesto.

Uso: `soup data canary check [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--manifest` | text |  | yes | Manifiesto de canarios |
| `--base` | text |  | yes | Id o ruta del modelo base |
| `--adapter` | text |  |  | Adaptador LoRA que se comprobará |
| `--controls` | integer | `128` |  | Controles nunca insertados con los que comparar la clasificación |
| `--device` | text | `auto` |  | auto/cpu/cuda |
| `--output`, `-o` | text |  |  | Escribe el informe como JSON |

#### `soup data canary insert`

Inserta K canarios únicos en un dataset y los registra.

Uso: `soup data canary insert [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |
| `--output`, `-o` | text |  | yes | Archivo de salida: .jsonl o .json para un arreglo JSON |
| `--manifest` | text |  | yes | Dónde escribir el manifiesto de canarios (CONTIENE LOS SECRETOS) |
| `--count`, `-k` | integer | `16` |  | Número de canarios |
| `--seed` | integer | `0` |  | Semilla para los canarios y las filas donde se insertan |
| `--format` | text | `auto` |  | Formato del dataset para escribir los canarios: auto (detectar), alpaca, sharegpt, chatml |

### `soup data clean`

Limpia y repara un dataset de ajuste fino: caracteres de control, espacios en blanco y coherencia del formato.

Uso: `soup data clean [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset de entrada (JSONL) |
| `--output`, `-o` | text |  |  | Path to write cleaned dataset (default: &lt;input>\_cleaned.jsonl) |
| `--force`, `-f` | flag | off |  | Sobrescribe el archivo de salida si ya existe |
| `--dry-run` | flag | off |  | Previsualiza los cambios sin escribir el archivo de salida |
| `--min-tokens` | integer | `1` |  | Longitud mínima en caracteres/tokens para los turnos del asistente (se descartan los más cortos) |
| `--prune-echo` / `--no-prune-echo` | flag | off |  | Elimina turnos del asistente que solo repiten el prompt del usuario (opcional) |
| `--strip-boilerplate` / `--no-strip-boilerplate` | flag | off |  | Elimina avisos genéricos de IA ('As an AI...', 'Certainly!...') |
| `--repair-code` / `--no-repair-code` | flag | off |  | Auto-close unclosed markdown \`\`\` code fences in assistant completions |
| `--repair-json` / `--no-repair-json` | flag | off |  | Repara comas finales y vallas markdown en argumentos de llamadas a herramientas |
| `--drop-invalid-json` | flag | off |  | Descarta filas de llamadas a herramientas con errores de sintaxis JSON no reparables |
| `--json` | flag | off |  | Emite un resumen JSON legible por máquinas |

### `soup data convert`

Convierte un dataset entre formatos (alpaca, sharegpt, chatml).

Uso: `soup data convert [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Archivo del dataset de entrada |
| `--to`, `-t` | text |  | yes | Formato de destino: alpaca, sharegpt, chatml |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_&lt;format>.jsonl) |

### `soup data decontaminate`

Descarta filas que se superponen con benchmarks públicos (heurística de n-gramas).

Uso: `soup data decontaminate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes |  |
| `--benchmarks`, `-b` | text |  |  | Nombres de benchmarks separados por comas (solo etiquetas; no se incluyen corpus). |
| `--benchmark-file` | text |  |  | v0.53.7: JSONL de benchmark proporcionado por el operador (dentro del cwd). El campo ``text``/``prompt``/``content`` de cada fila se agrega al conjunto de contaminación por n-gramas. Compatible con --benchmarks (lo amplía). |
| `--output`, `-o` | text | `clean.jsonl` |  | JSONL de salida con las filas que superaron la descontaminación. |
| `--threshold` | float, 0.0 to 1.0 | `0.8` |  |  |
| `--n` | integer, 1 to 32 | `8` |  |  |

### `soup data dedup`

Elimina filas casi duplicadas: MinHash (por defecto) o embeddings (--semantic).

Uso: `soup data dedup [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_deduped.jsonl) |
| `--threshold` | float | `0.8` |  | Umbral de similitud (0.0-1.0): Jaccard de MinHash por defecto, coseno de embeddings con --semantic. |
| `--field`, `-f` | text |  |  | Campo para calcular hash/embedding (por defecto: todos los campos de texto concatenados) |
| `--semantic` | flag | off |  | Use embedding cosine (SemDeDup) instead of MinHash. Catches paraphrases MinHash misses. Requires soup-cli\[train]. |
| `--embed-model` | text | `sentence-transformers/all-MiniLM-L6-v2` |  | Modelo de embeddings usado por --semantic. |
| `--device` | text | `auto` |  | Dispositivo para --semantic (auto/cpu/cuda). |

### `soup data demo`

Enumera o copia un dataset de demostración incluido (v0.43.0).

Uso: `soup data demo [OPTIONS] [NAME]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  |  | Nombre del paquete (alpaca_demo / sharegpt_demo / dpo_demo / grpo_demo). Omítelo para enumerar los paquetes disponibles. |
| `--output`, `-o` | text |  |  | Ruta del JSONL de destino (por defecto: ./&lt;name>.jsonl). |

### `soup data doctor`

Informe de compatibilidad de plantillas de chat + radiografía de máscaras de loss.

Uso: `soup data doctor [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset (jsonl/json/csv/parquet). |
| `--model`, `-m` | text |  | yes | Id del tokenizador o ruta local para comprobar la compatibilidad de la plantilla de chat. |
| `--format`, `-f` | text | `auto` |  | Formato del dataset (se detecta automáticamente por defecto). |
| `--max-length` | integer | `2048` |  | max_length de entrenamiento; se usa para comprobar el riesgo de truncamiento. |
| `--sample` | integer | `200` |  | Número de filas de muestra para las comprobaciones. |
| `--show-mask` | integer |  |  | Muestra N filas de ejemplo con colores por token para entrenado/enmascarado (la ruta real del collator). |
| `--train-on-responses-only` / `--no-train-on-responses-only` | flag | on |  | Enmascara tokens que no sean del asistente (refleja data.train_on_responses_only). |
| `--train-on-messages-with-train-field` | flag | off |  | Enmascaramiento por campo train:bool de cada mensaje (refleja la misma bandera de soup.yaml). |
| `--mask-history` / `--no-mask-history` | flag | off |  | Entrena solo el ÚLTIMO turno del asistente (refleja data.mask_history). |
| `--train-on-eot` | flag | off |  | Extiende el tramo entrenado hasta el token EOS/EOT final. |
| `--trust-remote-code` | flag | off |  | Permite cargar un tokenizador que incluya código personalizado. |
| `--output`, `-o` | text |  |  | Escribe el informe JSON en esta ruta. |

### `soup data download`

Descarga un dataset de HuggingFace y lo guarda como JSONL.

Uso: `soup data download [OPTIONS] DATASET_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATASET_ID` | text |  | yes | Id del dataset de HuggingFace (p. ej., teknium/OpenHermes-2.5) |
| `--output`, `-o` | text |  |  | Ruta del archivo de salida (por defecto: &lt;dataset-name>.jsonl) |
| `--split` | text | `train` |  | Dataset split to download (e.g. train, test, train\[:1000]) |
| `--samples`, `-n` | integer |  |  | Número máximo de muestras para descargar (usa streaming, no descarga todo) |
| `--format`, `-f` | text |  |  | Convierte al formato Soup después de descargar: alpaca, sharegpt, chatml |
| `--trust-remote-code` | flag | off |  | Permite datasets que incluyen cargadores Python personalizados. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--hub` | text | `hf` |  | Hub de origen: hf (por defecto) / modelscope / modelers. Los hubs que no son HF requieren el SDK correspondiente (v0.53.8 #130). |

### `soup data educational`

Etiqueta cada fila con una puntuación de valor educativo

Uso: `soup data educational [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes |  |
| `--output`, `-o` | text | `educational.jsonl` |  |  |

### `soup data evolve`

Evol-Instruct: profundiza (depth) o diversifica (breadth) instrucciones semilla.

Uso: `soup data evolve [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input` | text |  | yes | Seed JSONL ({prompt\|instruction\|messages} per row) |
| `--provider` | text |  | yes | ollama \| vllm |
| `--model` | text |  | yes | Id del modelo del proveedor |
| `--strategy` | text | `depth` |  | depth \| breadth |
| `--rounds` | integer | `1` |  | Evolution rounds \[1, 5] |
| `--output`, `-o` | text | `""` |  | JSONL de salida (obligatorio salvo con --plan-only) |
| `--base-url` | text | `""` |  | URL base del proveedor |
| `--temperature` | float | `1.0` |  | Sampling temp \[0, 2] |
| `--max-tokens` | integer | `512` |  | Max tokens \[1, 16384] |
| `--plan-only` | flag | off |  | Valida e imprime el plan |

### `soup data filter`

Filtra el dataset por calidad: puntuación de perplexity y/o coherencia.

Uso: `soup data filter [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_filtered.jsonl) |
| `--perplexity`, `--ppl` | float |  |  | Umbral máximo de perplexity (se eliminan las filas que lo superan) |
| `--coherence`, `--min-coherence` | float |  |  | Umbral mínimo de coherencia 0.0-1.0 (se eliminan las filas por debajo) |
| `--ppl-model` | text | `gpt2` |  | Modelo para puntuar perplexity (por defecto: gpt2) |
| `--field`, `-f` | text |  |  | Campo que se puntuará (por defecto: todos los campos de texto concatenados) |
| `--score-only` | flag | off |  | Add scores to data without filtering (writes \_scored.jsonl) |

### `soup data forge`

Ejecuta el pipeline de datos sintéticos de varias etapas con procedencia.

Uso: `soup data forge [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--docs`, `-d` | text |  | yes | Directorio de documentos (txt/md/json/jsonl, dentro del cwd). |
| `--task`, `-t` | text | `sft` |  | Forge task: sft \| preference \| tool. |
| `--target-rows`, `-r` | integer, 1 to 1000000 | `100` |  | Número máximo de filas sintetizadas. |
| `--teacher` | text | `local-judge` |  | Etiqueta registrada en la procedencia para el backend del evaluador. |
| `--output`, `-o` | text | `forge_dataset.jsonl` |  | Ruta JSONL de salida (dentro del cwd). |
| `--provenance` | text | `forge_provenance.json` |  | Ruta de salida del manifiesto de procedencia (dentro del cwd). |
| `--uncertainty-threshold` | float, 0.0 to 1.0 | `0.0` |  | Puntuación mínima de distancia de Jaccard necesaria para conservar una fila sintetizada. |
| `--max-chunk-chars` | integer, 1 to 64000 | `1000` |  | Máximo de caracteres por fragmento de documento antes de llamar al evaluador. |
| `--judge-provider` | text |  |  | v0.53.7 #111: backend activo del evaluador. Uno de ollama / anthropic / vllm. Omítelo para usar el stub determinista sin conexión. |
| `--judge-model` | text | `llama3.1` |  | Nombre del modelo para el proveedor activo del evaluador (por defecto llama3.1). |
| `--judge-base-url` | text |  |  | Anula la URL base para Ollama (solo localhost) / vLLM (lista de esquemas permitidos + loopback). Se ignora para Anthropic. |
| `--hub` | text | `hf` |  | Hub del teacher: hf (por defecto) / modelscope / modelers. Si no es HF y --teacher es un id de repo (owner/name), el teacher se descarga previamente desde ese hub (v0.71.5 #157). |

### `soup data from-traces`

Recopila pares de preferencias de traces de producción.

Uso: `soup data from-traces [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--logs` | text |  | yes | Origen de traces: un archivo JSONL para langchain/openai o el directorio donde soup-serve escribe con `soup serve --trace-log` (se rechaza un solo archivo) |
| `--format` | text |  | yes | Trace format: langchain \| openai \| soup-serve |
| `--signal` | text | `thumbs_up` |  | Signal to extract pairs from: thumbs_up \| regenerations \| user_edit |
| `--output`, `-o` | text | `prefs.jsonl` |  | Ruta de salida de los pares de preferencias (JSONL) |
| `--judge` | flag | off |  | Filtra pares según la confianza de un evaluador LLM (v0.40.3 #33). |
| `--judge-provider` | text | `openai` |  | Judge backend: openai \| server \| ollama. Used with --judge. |
| `--judge-model` | text | `gpt-4o-mini` |  | Id del modelo evaluador (p. ej., 'gpt-4o-mini', 'llama3', 'qwen2.5'). Se usa con --judge. |
| `--judge-api-base` | text |  |  | URL base de la API del evaluador. Protegida contra SSRF. Se usa con --judge. |
| `--min-confidence` | float | `0.7` |  | Descarta pares con confianza del evaluador inferior a este umbral (0.0 - 1.0). |

### `soup data gen-magpie`

Genera datos sintéticos a partir de un prefijo de plantilla de chat.

Uso: `soup data gen-magpie [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base` | text |  | yes | Id del modelo base alineado y ajustado para chat |
| `--provider` | text |  | yes | ollama \| anthropic \| vllm |
| `--target` | integer |  | yes | Target row count \[1, 1_000_000] |
| `--output`, `-o` | text | `""` |  | Ruta JSONL de salida (obligatoria salvo con --plan-only) |
| `--base-url` | text | `""` |  | URL base del proveedor (ollama/vllm; por defecto, loopback) |
| `--max-tokens` | integer | `512` |  | Max tokens per generation \[1, 16384] |
| `--quality-filter` / `--no-quality-filter` | flag | on |  | Aplica heurísticas de contenido no vacío y educativo (no es un clasificador de toxicidad) |
| `--plan-only` | flag | off |  | Valida e imprime el plan; no genera datos. |

### `soup data generate`

Genera datos de entrenamiento sintéticos mediante un LLM.

Uso: `soup data generate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--prompt`, `-p` | text |  | yes | Prompt del sistema que describe qué tipo de datos se generarán |
| `--count`, `-n` | integer | `100` |  | Número de ejemplos que se generarán |
| `--output`, `-o` | text | `generated.jsonl` |  | Ruta del archivo de salida |
| `--format`, `-f` | text | `alpaca` |  | Formato de salida: alpaca, sharegpt, chatml |
| `--provider` | text | `openai` |  | Proveedor de LLM: openai, local, server, ollama, anthropic, vllm |
| `--model`, `-m` | text | `gpt-4o-mini` |  | Nombre del modelo (id de modelo OpenAI, modelo Ollama o ruta de modelo local) |
| `--api-key` | text |  |  | \[deprecated] Use OPENAI_API_KEY env var instead. Env: `OPENAI_API_KEY`. |
| `--api-base` | text |  |  | URL base personalizada de la API (debe usar HTTPS para API remotas) |
| `--batch-size` | integer | `5` |  | Número de ejemplos por llamada a la API |
| `--temperature`, `-t` | float | `0.8` |  | Temperatura de muestreo para la generación |
| `--dedup-with` | text |  |  | Ruta al dataset existente contra el que se deduplicará |
| `--seed` | text |  |  | Ruta al archivo de ejemplos semilla (JSONL) que guiará la generación |
| `--template` | text |  |  | Plantilla de dominio: code, conversation, qa, preference, reasoning |
| `--language` | text | `Python` |  | Lenguaje para la plantilla de código (Python, JavaScript, Go, Rust, Java) |
| `--task-type` | text | `function` |  | Tipo de tarea para la plantilla de código (function, debug, explain, refactor, test) |
| `--turns` | integer | `4` |  | Número de turnos para la plantilla de conversación (2-10) |
| `--topic` | text | `general knowledge` |  | Tema para la plantilla de conversación |
| `--context` | text |  |  | Ruta al documento de contexto para la plantilla QA |
| `--pref-task` | text | `dpo` |  | Formato de tarea de preferencias: dpo, kto, orpo |
| `--domain` | text | `math` |  | Dominio para la plantilla de razonamiento (math, logic, code) |
| `--ollama-model` | text |  |  | Nombre del modelo Ollama (p. ej., llama3.1). Atajo para --provider ollama --model X |
| `--validate` | flag | off |  | Valida automáticamente los datos generados después de la generación |
| `--filter` | flag | off |  | Filtra automáticamente los datos generados por calidad (perplexity + coherencia) |
| `--dedup` | flag | off |  | Deduplica automáticamente los datos generados (MinHash) |
| `--quality-pipeline` | flag | off |  | Ejecuta el pipeline de calidad completo: validación + filtrado + deduplicación |
| `--requests-per-minute`, `--rpm` | integer | `60` |  | Límite de solicitudes por minuto para la API (por defecto: 60) |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos del proveedor local que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). |

### `soup data ingest`

Incorpora un documento en JSONL con una fila por página / encabezado.

Uso: `soup data ingest [OPTIONS] FILE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `FILE` | text |  | yes | Archivo PDF / DOCX / MD / TXT |
| `--output`, `-o` | text | `./ingested.jsonl` |  | Ruta JSONL de salida |

### `soup data inspect`

Inspecciona un dataset: muestra estadísticas y filas de ejemplo.

Uso: `soup data inspect [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset (jsonl, csv, parquet) |
| `--rows`, `-r` | integer | `5` |  | Número de filas de muestra que se mostrarán |
| `--image-dir` | text |  |  | Directorio que contiene las imágenes referenciadas |

### `soup data langdetect`

Etiqueta cada fila con un código de idioma de 2 letras (heurístico).

Uso: `soup data langdetect [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes |  |
| `--output`, `-o` | text | `langdetect.jsonl` |  |  |

### `soup data lint`

Linter de datos de preferencias (dpo/orpo/simpo/ipo/bco/kto).

Uso: `soup data lint [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset de preferencias (dpo/kto). |
| `--format`, `-f` | text | `auto` |  | Formato del dataset: auto, dpo, kto. |
| `--model`, `-m` | text |  |  | Tokenizador opcional para calcular el sesgo con longitud exacta en tokens (por defecto: conteo de palabras). |
| `--sample` | integer | `2000` |  | Número de filas de muestra para las comprobaciones. |
| `--trust-remote-code` | flag | off |  | Permite cargar un tokenizador que incluya código personalizado. |
| `--output`, `-o` | text |  |  | Escribe el informe JSON en esta ruta. |

### `soup data merge`

Fusiona varios datasets en un solo archivo.

Uso: `soup data merge [OPTIONS] FILES...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `FILES...` | text, any number |  | yes | Rutas a los archivos de dataset que se fusionarán |
| `--output`, `-o` | text | `merged.jsonl` |  | Ruta del archivo de salida |
| `--shuffle` | flag | off |  | Mezcla los datos después de la fusión |

### `soup data mix`

BETA: optimiza los pesos de mezcla por dataset frente a una ejecución proxy.

Uso: `soup data mix [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--optimize` | flag | off |  | Ejecuta una búsqueda bayesiana de los pesos de mezcla de datasets. |
| `--apply` | text |  |  | Vuelve a imprimir una mix-recipe escrita anteriormente (ruta dentro del cwd). |
| `--datasets` | text |  |  | Lista separada por comas de rutas JSONL de datasets (>= 2, todas dentro del cwd). |
| `--budget` | text | `1h` |  | Límite de tiempo transcurrido: dígitos + sufijo opcional s/m/h (p. ej., 1h, 30m). |
| `--num-probes`, `-n` | integer, 1 to 256 | `8` |  | Número máximo de ejecuciones proxy. |
| `--seed` | integer, 0 to 2147483647 | `42` |  | Semilla RNG del optimizador. |
| `--output`, `-o` | text | `mix_recipe.yaml` |  | Ruta de salida de la receta YAML (dentro del cwd). |
| `--overwrite` | flag | off |  | Sobrescribe la ruta de salida si existe. |
| `--live` | flag | off |  | v0.53.5 #116: ejecuta en vivo una ejecución proxy corta de `soup train` por candidato (requiere --base-yaml). Por defecto usa el proxy sintético sin conexión. |
| `--base-yaml` | text |  |  | Ruta a un soup.yaml base (dentro del cwd) que proporciona base/task/training/output. Obligatorio con --live. |

### `soup data persona-mix`

Muestrea una matriz de prompt × persona × estilo.

Uso: `soup data persona-mix [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--prompts` | text |  | yes | Archivo JSONL con un prompt por fila |
| `--n` | integer |  | yes | Number of rows to emit \[1, 1_000_000] |
| `--output` | text |  | yes | Ruta JSONL de salida |
| `--personas` | text | `""` |  | JSONL opcional de personas (por defecto usa el conjunto incluido) |
| `--styles` | text | `""` |  | JSONL opcional de estilos (por defecto usa el conjunto incluido) |
| `--seed` | integer | `0` |  | Semilla determinista |

### `soup data pii`

Marca filas que contienen patrones de correo electrónico / teléfono / SSN / tarjeta de crédito.

Uso: `soup data pii [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes |  |
| `--output`, `-o` | text | `pii.jsonl` |  |  |

### `soup data preprocess`

Tokeniza un dataset AOT y lo guarda en caché en disco para reutilizarlo entre ejecuciones.

Uso: `soup data preprocess [OPTIONS] CONFIG_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CONFIG_PATH` | text |  | yes | Ruta a soup.yaml: usa data.train + tokenizer + max_length |
| `--output`, `-o` | text | `./.soup-tokenized` |  | Directorio de caché dentro del cwd |
| `--yes` | flag | off |  | Omite el mensaje de confirmación |

### `soup data preview`

Previsualiza un dataset remoto de HuggingFace: metadatos, particiones y características.

Uso: `soup data preview DATASET_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATASET_ID` | text |  | yes | Id del dataset de HuggingFace (p. ej., teknium/OpenHermes-2.5) |

### `soup data push`

Sube un dataset JSONL local a un hub de modelos como repo de dataset.

Uso: `soup data push [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Ruta al archivo JSONL del dataset local |
| `--hf-dataset` | text |  | yes | Id del repo de dataset de HuggingFace (p. ej., user/my-dataset) |
| `--private` | flag | off |  | Hace privado el repo del dataset HF |
| `--message` | text | `Upload dataset with Soup CLI` |  | Mensaje de commit para la carga del dataset |
| `--hub` | text | `hf` |  | Hub de destino: hf (por defecto) / modelscope / modelers. Los hubs que no son HF cargan mediante el SDK correspondiente (v0.71.5 #157): repo_type=dataset, mensaje de commit sanitizado a la primera línea + 200 caracteres. |

### `soup data recipe`

Valida un DAG de Data Recipe y, de forma opcional, ejecuta todos sus nodos.

Uso: `soup data recipe [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta a recipe.yaml dentro del cwd |
| `--execute` | flag | off |  | Ejecuta el DAG validado de extremo a extremo (v0.53.7 #106: ejecución activa por nodo: seed / llm_text / code / judge / validator / sampler). |
| `--output`, `-o` | text |  |  | Directorio de salida para el nodo sampler (obligatorio con --execute). |
| `--provider` | text |  |  | Proveedor LLM para nodos llm_text/judge: ollama, anthropic o vllm. |
| `--model` | text |  |  | Nombre del modelo del proveedor (por defecto llama3.1). |
| `--base-url` | text |  |  | URL base del proveedor (Ollama/vLLM; por defecto, loopback). |
| `--offline` | flag | off |  | Permite explícitamente la generación de marcadores de posición y la evaluación que acepta todo. |

### `soup data register`

Registra un dataset local por nombre para usarlo en soup.yaml.

Uso: `soup data register [OPTIONS] [NAME_ARG] [PATH_ARG]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME_ARG` | text |  |  | Nombre del dataset (alternativa posicional a --name) |
| `PATH_ARG` | text |  |  | Ruta al archivo del dataset (alternativa posicional a --path) |
| `--name`, `-n` | text |  |  | Nombre del dataset |
| `--path`, `-p` | text |  |  | Ruta al archivo del dataset |
| `--format`, `-f` | text | `auto` |  | Formato del dataset: alpaca, sharegpt, chatml, dpo, kto, auto |

### `soup data registry`

Enumera todos los datasets registrados.

Uso: `soup data registry`

No acepta opciones ni argumentos.

### `soup data review`

Previsualiza pares de preferencias para revisión manual.

Uso: `soup data review [OPTIONS] INPUT`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `INPUT` | text |  | yes | Ruta al JSONL de preferencias (chosen/rejected) |
| `--sample`, `-s` | integer | `10` |  | Cantidad de pares que se previsualizarán (1-100) |

### `soup data sample`

Toma una muestra de un subconjunto del dataset mediante distintas estrategias.

Uso: `soup data sample [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_sampled.jsonl) |
| `--n`, `-n` | integer |  |  | Número de muestras que se seleccionarán |
| `--pct` | float |  |  | Porcentaje del dataset que se muestreará (0-100) |
| `--strategy`, `-s` | text | `random` |  | Estrategia de muestreo: random, diverse (TF-IDF + clusters), hard (por longitud) |
| `--seed` | integer |  |  | Semilla aleatoria para reproducibilidad |

### `soup data score`

Tarjeta compuesta de calidad de datos (PII + clasificación por palabras clave + idioma + educación + descontaminación).

Uso: `soup data score [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Entrada JSONL dentro del cwd. |
| `--benchmarks`, `-b` | text |  |  | Etiquetas de benchmark (p. ej., mmlu,gsm8k), solo validadas; no seleccionan ni filtran los textos de --benchmark-file. Requiere --benchmark-file; no se incluyen corpus. |
| `--benchmark-file` | text |  |  | JSONL de benchmark proporcionado por el operador dentro del cwd, con texto utilizable de 8-gramas en cada fila. |
| `--threshold` | float, 0.0 to 1.0 | `0.8` |  | Umbral de superposición para descontaminación. |

### `soup data search`

Busca datasets en HuggingFace Hub.

Uso: `soup data search [OPTIONS] QUERY`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `QUERY` | text |  | yes | Consulta para buscar datasets de HuggingFace |
| `--limit`, `-l` | integer | `20` |  | Número máximo de resultados que se mostrarán |
| `--sort`, `-s` | text | `downloads` |  | Ordenar por: downloads, likes, lastModified, trending, createdAt |

### `soup data split`

Divide el dataset en archivos de train/val/test.

Uso: `soup data split [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |
| `--val` | integer |  |  | División de validación: porcentaje (por defecto) o cantidad absoluta (con --absolute) |
| `--test` | integer |  |  | División de prueba: porcentaje (por defecto) o cantidad absoluta (con --absolute) |
| `--train` | integer |  |  | División de entrenamiento (informativa; el resto de train se deduce de --val + --test). Se acepta para mantener paridad del comando. |
| `--absolute` | flag | off |  | Interpreta --val/--test como cantidades absolutas de muestras, no como porcentajes |
| `--seed` | integer |  |  | Semilla aleatoria para divisiones reproducibles |
| `--stratify` | text |  |  | Nombre del campo para la división estratificada (preserva la distribución de categorías) |
| `--stratify-semantic` | flag | off |  | Usa agrupamiento semántico (TF-IDF + K-Means) para dividir de forma estratificada sin requerir un campo de categoría |
| `--num-clusters` | integer |  |  | Número de clusters semánticos para la división estratificada semántica (por defecto: 5) |

### `soup data stats`

Estadísticas ampliadas del dataset: distribución de longitudes, conteo de tokens e idiomas.

Uso: `soup data stats PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |

### `soup data topics`

Agrupa un dataset por temas y muestra una tabla de cobertura.

Uso: `soup data topics [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset (jsonl/json/csv) |
| `--clusters`, `-k` | text | `auto` |  | Número de clusters temáticos o 'auto' (heurística de raíz cuadrada). |
| `--embed-model` | text | `sentence-transformers/all-MiniLM-L6-v2` |  | Modelo de embeddings. |
| `--device` | text | `auto` |  | auto/cpu/cuda |
| `--seed` | integer | `0` |  | Semilla del agrupamiento. |
| `--output`, `-o` | text |  |  | Escribe el informe como JSON. |

### `soup data toxicity`

Ejecuta una heurística de palabras clave de abuso (no es un clasificador de toxicidad).

Uso: `soup data toxicity [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes |  |
| `--output`, `-o` | text | `toxicity.jsonl` |  |  |
| `--threshold` | float, 0.0 to 1.0 | `0.05` |  | Se conservan las filas que alcanzan el umbral de palabras clave de abuso. |

### `soup data unregister`

Elimina un dataset del registro local.

Uso: `soup data unregister [OPTIONS] [NAME_ARG]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME_ARG` | text |  |  | Nombre del dataset (alternativa posicional a --name) |
| `--name`, `-n` | text |  |  | Nombre del dataset que se eliminará |

### `soup data validate`

Valida un dataset; devuelve el código de salida 3 para errores de entrada y 2 para datos inutilizables.

Uso: `soup data validate [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Ruta al archivo del dataset |
| `--format`, `-f` | text | `auto` |  | Formato esperado: auto, alpaca, sharegpt, chatml, dpo, kto, plaintext |
| `--min-valid-fraction` | float, 0.0 to 1.0 | `0.0` |  | Termina con código 2 cuando la fracción de filas válidas es inferior a este valor |

## `soup delinearize-llama4`

Delineariza los pesos de expertos fusionados de Llama 4 a formato 3-D.

Uso: `soup delinearize-llama4 [OPTIONS] SOURCE_DIR`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SOURCE_DIR` | text |  | yes | Llama 4 checkpoint directory (containing \*.safetensors). |
| `--target`, `-o` | text |  | yes | Directorio de destino para los pesos delinearizados (dentro del cwd). |
| `--model-id` | text |  |  | Id opcional del modelo; advierte si no parece ser un modelo Llama 4. |
| `--num-experts` | integer |  |  | Cantidad de expertos para el cambio de forma. Por defecto, usa num_local_experts de config.json. |
| `--plan-only` | flag | off |  | Muestra el plan y termina sin cambiar la forma (flujo v0.44.0). |

## `soup deploy`

Implementa modelos: integración con Ollama (implementación, lista y eliminación).

Uso: `soup deploy <command> [ARGS]...`

Subcomandos:

- [`autopilot`](#soup-deploy-autopilot)
- [`hf-space`](#soup-deploy-hf-space)
- [`ollama`](#soup-deploy-ollama)

### `soup deploy autopilot`

Elige una combinación de PEFT + cuantización + speculative decoding para un destino de hardware.

Uso: `soup deploy autopilot [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target`, `-t` | text |  |  | Nombre del perfil (p. ej., mac-m3, rtx-4090-24gb, ollama-local). |
| `--base`, `-b` | text | `meta-llama/Llama-3.2-1B` |  | Id de repo HF del modelo base o ruta local que se incluirá en la receta. |
| `--recipe-out` | text | `deploy_autopilot.yaml` |  | Dónde escribir la receta soup.yaml generada (dentro del cwd). |
| `--script-out` | text | `deploy_autopilot.sh` |  | Dónde escribir el script de implementación planificado (dentro del cwd). |
| `--output-dir` | text | `./output` |  | Directorio de salida del entrenamiento de la receta. |
| `--list`, `-l` | flag | off |  | Enumera todos los perfiles de implementación conocidos. |
| `--measure` | flag | off |  | Run Quant-Lobotomy measurement on candidate quants. Requires --tasks. Results cached at \~/.soup/deploy_autopilot_cache.json (v0.53.1 #109). |
| `--tasks` | text |  |  | Tareas de evaluación JSONL para --measure (un prompt + expected por línea). |
| `--measure-candidates` | text |  |  | Candidatos de cuantización separados por comas que se medirán (por defecto: cuantización principal del perfil). Ejemplo: 4bit,gptq,awq |

### `soup deploy hf-space`

Crea un Space de HuggingFace que envuelve un modelo ajustado.

Uso: `soup deploy hf-space [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Id del repo de modelo de HuggingFace que se incluirá en el Space (p. ej., user/my-model) |
| `--space`, `-s` | text |  | yes | Id del repo de Space de HuggingFace que se creará (p. ej., user/my-space) |
| `--template`, `-t` | text | `gradio-chat` |  | Plantilla de Space: gradio-chat, streamlit-chat |
| `--template-dir` | text |  |  | Directorio de plantilla personalizado (anula --template). Debe contener app.py + README.md y, de forma opcional, requirements.txt. Usa el marcador {MODEL_REPO} para sustituirlo. |
| `--private` | flag | off |  | Crea el Space como privado |
| `--yes`, `-y` | flag | off |  | Omite la confirmación |

### `soup deploy ollama`

Implementa un modelo GGUF en una instancia local de Ollama.

Uso: `soup deploy ollama [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  |  | Ruta al archivo del modelo GGUF |
| `--name`, `-n` | text |  |  | Nombre del modelo Ollama (p. ej., soup-my-model) |
| `--system`, `-s` | text |  |  | Prompt del sistema para el modelo |
| `--template`, `-t` | text | `auto` |  | Plantilla de chat: auto, chatml, llama, mistral, vicuna, zephyr |
| `--parameter`, `-p` | text, repeatable |  |  | Parámetro de Ollama (repetible): temperature=0.7, top_p=0.9, etc. |
| `--list`, `-l` | flag | off |  | Enumera los modelos implementados por Soup en Ollama |
| `--remove`, `-r` | text |  |  | Elimina un modelo de Ollama por nombre |
| `--yes`, `-y` | flag | off |  | Omite el mensaje de confirmación |

## `soup diagnose`

Informe de post-entrenamiento: olvido / rechazo / formato / mode_collapse / memorización / contaminación (v0.56.0).

Uso: `soup diagnose [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Id de ejecución del registro (o cualquier etiqueta opaca). |
| `--base` | text | `""` |  | Nombre del modelo base (informativo). |
| `--adapter` | text | `""` |  | Nombre del adaptador (informativo). |
| `--evidence` | text |  |  | Archivo JSON con puntuaciones de sondeos precalculadas (consulta el README). |
| `--output`, `-o` | text |  |  | Escribe el informe JSON en esta ruta. |
| `--badge` | text |  |  | Escribe una insignia SVG en esta ruta. |
| `--attach-to-registry` | text |  |  | Adjunta el informe --output a un id de entrada del registro (requiere --output). Termina con código 1 si falla la adjunción. |
| `--base-model` | text |  |  | Id del modelo base que se CARGARÁ para un diagnóstico activo (p. ej., HuggingFaceTB/SmolLM2-135M). Si se define, los 6 sondeos se ejecutan contra el modelo en lugar de emitir puntuaciones neutrales. |
| `--dataset` | text |  |  | JSONL de entrenamiento para sondeos activos de formato / memorización / citas (debe permanecer dentro del cwd). |
| `--holdout` | text |  |  | JSONL de evaluación reservado para el sondeo activo de olvido (debe permanecer dentro del cwd; si no se define, usa prompts generales integrados). |
| `--tokenizer` | text |  |  | Id/ruta del tokenizador para un sondeo activo de memorización de subpalabras. |
| `--device` | text |  |  | Dispositivo para la ejecución activa (cuda / cpu). |
| `--citation-style` | text | `bracket` |  | Citation style for the live RAFT citation probe: bracket (\[doc-id]) / inline ((doc-id)) / footnote (\[^id]). Must match the style the model was trained to emit. (v0.71.17 #254) |
| `--allow-not-run` | flag | off |  | Termina con código 0 incluso si no se ejecutó un sondeo solicitado (veredicto NOT_RUN). Sin esta opción, esa ejecución termina con código 3. |
| `--shuffle-seed` | integer |  |  | RAFT document-shuffle seed for the live citation probe (composes the \[doc-N] ids the model is scored against). (v0.71.17 #254) |

## `soup diff`

Compara lado a lado las salidas de dos modelos con los mismos prompts.

Uso: `soup diff [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model-a`, `-a` | text |  | yes | Ruta al primer modelo (adaptador LoRA o modelo completo) |
| `--model-b`, `-b` | text |  | yes | Ruta al segundo modelo (adaptador LoRA o modelo completo) |
| `--prompts`, `-p` | text |  |  | Ruta al archivo de prompts (JSONL con campo 'prompt' o un prompt por línea) |
| `--prompt` | text, repeatable |  |  | Prompt individual que se comparará (se puede repetir) |
| `--base-a` | text |  |  | Modelo base de model A (se detecta automáticamente para adaptadores LoRA) |
| `--base-b` | text |  |  | Modelo base de model B (se detecta automáticamente para adaptadores LoRA) |
| `--max-tokens` | integer | `256` |  | Máximo de tokens que se generarán por respuesta |
| `--temperature`, `-t` | float | `0.7` |  | Temperatura de muestreo |
| `--device` | text |  |  | Dispositivo: cuda, mps, cpu. Se detecta automáticamente si no se define. |
| `--output`, `-o` | text |  |  | Guarda los resultados en un archivo JSONL |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |

## `soup distill-prompt`

Destila traces con muchos prompts en un plan pequeño de FT (v0.68.0).

Uso: `soup distill-prompt [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--traces` | text |  | yes | Ruta al JSONL de traces |
| `--teacher` | text |  | yes | Id del modelo teacher |
| `--student` | text |  | yes | Id del modelo student |
| `--strategy` | text | `sft` |  | Estrategia de destilación. Permitidas: kl, preference, sft |
| `--provider` | text | `ollama` |  | Proveedor de teacher/student: ollama / anthropic / vllm. |
| `--base-url` | text |  |  | URL base del proveedor (ollama / vllm). |
| `--temperature` | float | `0.0` |  | Temperatura de muestreo para teacher/student. |
| `--max-rows` | integer |  |  | Limita el número de filas destiladas. |
| `--output`, `-o` | text | `distilled.jsonl` |  | Ruta JSONL de salida |
| `--plan-only` | flag | off |  | Muestra el plan y termina con código 0 (sin preparación activa). |

## `soup doctor`

Comprueba las dependencias del sistema, la GPU y la compatibilidad.

Uso: `soup doctor [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--nccl` | flag | off |  | Mide el ancho de banda NCCL y lo compara con la tabla de referencia. |
| `--disk` | flag | off |  | Detect the disk media type (NVMe / SSD / HDD). Layer streaming's disk overflow tier needs NVMe. Off by default: the probe costs \~9 s on Windows, and on Linux it WRITES a \~64 MiB scratch file (.soup-diskprobe-\*, git-ignored) into the current directory to measure sequential read throughput. |
| `--config`, `-c` | text |  |  | También comprueba un soup.yaml: informa qué ajustes definidos no se leen para su tarea y backend (#755). |

## `soup draft`

Entrena y mide un modelo draft para speculative decoding: destila tu modelo ajustado en un draft diminuto y luego sírvelo con --auto-spec (v0.71.33).

Uso: `soup draft <command> [ARGS]...`

Subcomandos:

- [`distill`](#soup-draft-distill)
- [`list`](#soup-draft-list)
- [`measure`](#soup-draft-measure)

### `soup draft distill`

Destila un modelo de destino en un draft diminuto y denso para speculative decoding.

Uso: `soup draft distill [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target` | text |  | yes | Modelo ajustado que se acelerará (el teacher). |
| `--draft-base` | text |  | yes | Modelo diminuto que recibirá la destilación (el student). Debe compartir el tokenizador del modelo de destino. Una salida de `soup shrink` cumple este requisito por construcción. |
| `--data` | text |  | yes | Conjunto JSONL de destilación (filas de chat / instrucciones). |
| `-o`, `--output` | text | `draft` |  | Directorio para el modelo draft denso. |
| `--steps` | integer, 1 to 1000000 | `500` |  | Pasos de entrenamiento aproximados. |
| `--device` | text |  |  | cpu \| cuda (default: auto-detect). |
| `--no-register` | flag | off |  | Do not record the draft in \~/.soup/drafts.json (it then won't be picked up by `soup serve --auto-spec`). |
| `--trust-remote-code` | flag | off |  | Permite código de modelado personalizado del target / draft-base (necesario para arquitecturas que incluyen auto_map). |
| `--force` | flag | off |  | Sobrescribe -o incluso si ya contiene un directorio que no es de draft. |
| `--plan-only` | flag | off |  | Imprime la configuración de destilación y termina; no escribe nada. |

### `soup draft list`

Enumera los drafts entrenados localmente que puede usar `soup serve --auto-spec`.

Uso: `soup draft list`

No acepta opciones ni argumentos.

### `soup draft measure`

Informa la tasa de aceptación y el throughput de un draft frente a su modelo de destino.

Uso: `soup draft measure [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target` | text |  | yes | Modelo que se está sirviendo. |
| `--draft` | text |  | yes | Modelo draft (una salida de `soup draft distill`). |
| `--prompts` | text |  | yes | Prompts JSONL representativos del tráfico de producción. |
| `--max-new-tokens` | integer, 1 to 4096 | `64` |  | Tokens que se generarán por prompt. |
| `--num-assistant-tokens` | integer, 1 to 64 | `5` |  | Tokens que propone el draft por paso (el parámetro de generación asistida). |
| `--device` | text |  |  | cpu \| cuda. |
| `--min-acceptance` | float, 0.0 to 1.0 |  |  | Termina con código 2 si la tasa de aceptación cae por debajo de este valor (para validación CI). |
| `--trust-remote-code` | flag | off |  | Permite código de modelado personalizado del target / draft (necesario para arquitecturas que incluyen auto_map). |
| `-o`, `--output` | text |  |  | Escribe el informe como JSON. |
| `--sweep-k` | text |  |  | También mide la generación asistida en cada una de estas longitudes del draft; p. ej., 1,2,3,5,8 (máximo 8, cada una en 1..64). |

## `soup drift-alarm`

Alarma de deriva Rolling-KL en la distribución de tokens de salida con webhook opcional de Slack/Discord (v0.63.0).

Uso: `soup drift-alarm [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--reference` | text |  | yes | JSONL de salidas de referencia del momento del FT. |
| `--live` | text |  | yes | JSONL de salidas de producción en vivo. |
| `--threshold` | float | `0.2` |  | Umbral de deriva (divergencia KL). Por defecto: 0.2. |
| `--slack-url` | text |  |  | URL opcional del webhook de Slack: se envía un POST cuando hay deriva. Validada contra SSRF. |
| `--discord-url` | text |  |  | URL opcional del webhook de Discord: se envía un POST cuando hay deriva. Validada contra SSRF. |

## `soup edit`

Edición de conocimiento (ROME / MEMIT / AlphaEdit): aplica cambios a hechos sin volver a entrenar (v0.61.0).

Uso: `soup edit <command> [ARGS]...`

Subcomandos:

- [`diff`](#soup-edit-diff)
- [`set`](#soup-edit-set)

### `soup edit diff`

Diferencia de inyección de conocimiento: hechos cambiados entre antes / después.

Uso: `soup edit diff [OPTIONS] BEFORE AFTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BEFORE` | text |  | yes | Etiqueta o id de ejecución del modelo ANTES de la edición (metadatos). |
| `AFTER` | text |  | yes | Etiqueta o id de ejecución del modelo DESPUÉS de la edición (metadatos). |
| `--probes` | text |  |  | Archivo JSONL con prompts de sondeo; obligatorio si se definen tanto --before-model como --after-model. Cada fila debe tener un campo 'prompt'. Máximo de 1000 filas. |
| `--output`, `-o` | text |  |  | Ruta donde escribir el JSON de diferencias generado. |
| `--top-k`, `-k` | integer | `10` |  | Número de hechos modificados que se mostrarán (1-100). |
| `--before-model` | text |  |  | Ruta/id HF del modelo ANTES (generación activa junto con --after-model y --probes). |
| `--after-model` | text |  |  | Ruta/id HF del modelo DESPUÉS (generación activa junto con --before-model y --probes). |
| `--device` | text |  |  | Dispositivo de torch (cpu / cuda). Por defecto, CUDA cuando está disponible. |

### `soup edit set`

Aplica una única edición quirúrgica de conocimiento.

Uso: `soup edit set [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base`, `-b` | text |  | yes | Id HF o ruta local del modelo base. |
| `--method`, `-m` | text | `rome` |  | Método de edición: rome / memit / alphaedit. |
| `--subject`, `-s` | text |  | yes | Oración de prefijo (p. ej., "Paris is the capital of France"). |
| `--target`, `-t` | text |  | yes | Nuevo objetivo de completado (p. ej., "Lyon"). |
| `--layer`, `-l` | integer |  |  | Índice de capa MLP que se editará (por defecto, capa recomendada específica del método). |
| `--output`, `-o` | text |  |  | Directorio para guardar el modelo editado / codebook GRACE (dentro del cwd). |
| `--device` | text |  |  | Dispositivo de torch (cpu / cuda). Por defecto, CUDA cuando está disponible. |
| `--governor` / `--no-governor` | flag | on |  | Consulta el governor de edición secuencial (rechaza cambios si las normas se disparan). |
| `--cov-corpus` | text |  |  | Solo ROME: corpus JSONL/texto para estimar la covarianza de clave C para una actualización precondicionada por C^{-1} (reduce daños colaterales). |
| `--plan-only` | flag | off |  | Imprime el EditPlan resuelto y termina sin aplicarlo. |
| `--registry-id` | text |  |  | Id de entrada del registro al que adjuntar el resultado guardado (requiere --output): archivo(s) de pesos como edited_model o codebook GRACE como grace_codebook. Termina con código 1 si falla la adjunción. |

## `soup env`

Detección de deriva ABI + archivo de bloqueo de entorno hermético (v0.64.0).

Uso: `soup env <command> [ARGS]...`

Subcomandos:

- [`check`](#soup-env-check)
- [`fix`](#soup-env-fix)
- [`lock`](#soup-env-lock)
- [`status`](#soup-env-status)

### `soup env check`

Compara el entorno actual con el lock e informa las diferencias.

Uso: `soup env check [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--lock` | text | `soup-env.lock` |  | Ruta al archivo lock con el que se comparará. |

### `soup env fix`

Genera un plan de instalación reproducible desde ``soup-env.lock``.

Uso: `soup env fix [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--lock` | text | `soup-env.lock` |  | Ruta al archivo lock desde el que se generará el plan de instalación. |
| `--format` | text | `uv-pip` |  | Install-plan format: uv-pip (copy/paste uv commands) \| requirements. |
| `--output`, `-o` | text |  |  | También puede escribir un requirements.txt en esta ruta (dentro del cwd). |

### `soup env lock`

Guarda una instantánea del entorno actual en un archivo lock.

Uso: `soup env lock [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--output`, `-o` | text | `soup-env.lock` |  | Ruta donde escribir el archivo lock (por defecto: ./soup-env.lock). |

### `soup env status`

Imprime el resumen del entorno bloqueado actualmente.

Uso: `soup env status [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--lock` | text | `soup-env.lock` |  | Ruta al archivo lock (por defecto: ./soup-env.lock). |

## `soup eval`

Evalúa modelos: benchmarks, evaluaciones personalizadas, evaluador LLM y tabla de clasificación.

Uso: `soup eval <command> [ARGS]...`

Subcomandos:

- [`against`](#soup-eval-against)
- [`aider`](#soup-eval-aider)
- [`auto`](#soup-eval-auto)
- [`behavior`](#soup-eval-behavior)
- [`benchmark`](#soup-eval-benchmark)
- [`capability`](#soup-eval-capability)
- [`checklist`](#soup-eval-checklist)
- [`citation`](#soup-eval-citation)
- [`compare`](#soup-eval-compare)
- [`coverage`](#soup-eval-coverage)
- [`custom`](#soup-eval-custom)
- [`design`](#soup-eval-design)
- [`discover`](#soup-eval-discover)
- [`gate`](#soup-eval-gate)
- [`gate-install`](#soup-eval-gate-install)
- [`human`](#soup-eval-human)
- [`irt-subset`](#soup-eval-irt-subset)
- [`judge`](#soup-eval-judge)
- [`leaderboard`](#soup-eval-leaderboard)
- [`lock`](#soup-eval-lock)
- [`quant-check`](#soup-eval-quant-check)
- [`unlearning`](#soup-eval-unlearning)

### `soup eval against`

Comprobación de regresión entre ejecuciones (IC de bootstrap pareado).

Uso: `soup eval against [OPTIONS] BASELINE_RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASELINE_RUN_ID` | text |  | yes | Id de ejecución de referencia (de `soup runs`). |
| `--candidate` | text |  | yes | Id de ejecución candidata cuyas métricas se comparan con la referencia. |
| `--metric` | text | `task_accuracy` |  | Metric to check: task_accuracy \| refusal_rate \| format_validity \| p95_latency_ms \| custom \| aider_polyglot \| judge:&lt;model> \| benchmark:&lt;lm-eval-task>. |
| `--n-samples` | integer | `1000` |  | Muestras de bootstrap pareado (100-100000). |
| `--seed` | integer | `0` |  | Semilla determinista. |
| `--json-only` | flag | off |  | Suprime la salida Rich; emite una sola línea de veredicto JSON. |
| `--suite` | text |  |  | Locked eval suite (from `soup eval lock`) to validate as a gate precondition — a missing / unparseable suite BLOCKS the check (exit 3). cwd-contained. The generated pre-push hook passes \$GATE_SUITE here so the locked suite is actually enforced. |

### `soup eval aider`

Ejecuta el benchmark de programación Polyglot de Aider en su imagen oficial de Docker.

Uso: `soup eval aider [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Identificador de modelo de Aider (por ejemplo, openai/gpt-4.1) |
| `--output`, `-o` | text |  | yes | Directorio contenido para los archivos de resultados de upstream y Soup |
| `--exercises-dir` | text | `tmp.benchmarks/polyglot-benchmark` |  | Checkout preparado de Aider-AI/polyglot-benchmark |
| `--image` | text | `aider-benchmark` |  | Imagen local del benchmark upstream de Aider, compilada |
| `--threads`, `-t` | integer, 1 to 64 | `1` |  | Número de ejercicios de benchmark que se ejecutarán en paralelo |
| `--num-tests`, `-n` | integer | `-1` |  | Número de ejercicios que se ejecutarán (-1 ejecuta todos) |
| `--timeout` | integer, 60 to 86400 | `86400` |  | Tiempo máximo de ejecución del benchmark en segundos |
| `--run-id` | text |  |  | Guarda la puntuación agregada en un id de ejecución Soup existente |
| `--allow-host-services` | flag | off |  | Permite que el contenedor del benchmark acceda a servicios del host |

### `soup eval auto`

Ejecuta una evaluación automática usando la configuración de soup.yaml.

Uso: `soup eval auto [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta al archivo de configuración soup.yaml |
| `--benchmarks`, `-b` | text |  |  | Benchmarks separados por comas (anulan la configuración) |
| `--tasks`, `-t` | text |  |  | Ruta al JSONL de evaluación personalizada (anula la configuración) |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |

### `soup eval behavior`

Puntúa una ejecución con una batería de comportamiento integrada (diferencia antes/después).

Uso: `soup eval behavior [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Identificador de ejecución. |
| `--battery`, `-b` | text | `xstest` |  | Batería: xstest / harmbench / jailbreakbench / elephant / syceval. |
| `--evidence`, `-e` | text |  |  | Ruta a un archivo JSON con arreglos {pre_responses, post_responses, oracle}. Cada entrada oracle es una palabra que debe incluir la respuesta correspondiente; --base-model evalúa el comportamiento de rechazo en su lugar. |
| `--output`, `-o` | text |  |  | Ruta donde escribir el JSON BehaviorDiffReport generado. |
| `--base-model` | text |  |  | Id del modelo base para una diferencia EN VIVO (genera respuestas antes/después). |
| `--adapter` | text |  |  | Ruta del adaptador LoRA para el modelo 'post' en una diferencia en vivo. |
| `--device` | text |  |  | Dispositivo para la diferencia en vivo (cuda / cpu). |

### `soup eval benchmark`

Evalúa con benchmarks estándar (envuelve lm-evaluation-harness).

Uso: `soup eval benchmark [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Ruta al modelo o al directorio del adaptador LoRA |
| `--benchmarks`, `-b` | text | `mmlu` |  | Nombres de benchmark separados por comas (mmlu, gsm8k, hellaswag, etc.) |
| `--fewshot`, `-f` | integer |  |  | Número de ejemplos few-shot (por defecto, el del benchmark si no se define) |
| `--batch-size` | integer | `8` |  | Tamaño del batch para la evaluación |
| `--run-id` | text |  |  | Vincula los resultados a una ejecución de entrenamiento existente |
| `--device` | text |  |  | Dispositivo: cuda, mps, cpu. Se detecta automáticamente si no se define. |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |

### `soup eval capability`

Ejecuta un perfil de capacidades integrado (MMLU-Pro / GPQA / AIME / ...).

Uso: `soup eval capability [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Identificador de ejecución. |
| `--suite`, `-s` | text | `fast` |  | Perfil: full / fast / math / code. |
| `--output`, `-o` | text |  |  | Ruta donde escribir el JSON CapabilityReport generado. |
| `--live` | flag | off |  | Ejecuta la suite EN VIVO mediante lm-eval-harness contra --model. |
| `--model`, `-m` | text |  |  | Id de modelo HF para una ejecución en vivo (obligatorio con --live). |
| `--tasks` | text |  |  | Anulación de tareas lm-eval separadas por comas (en vivo; omite --suite). |
| `--limit` | integer |  |  | Limita los ejemplos de evaluación por tarea (en vivo; usa 1-5 para una prueba rápida). |
| `--device` | text |  |  | Dispositivo para la ejecución en vivo (cuda / cpu). |

### `soup eval checklist`

Ejecuta pruebas de comportamiento CheckList MFT / INV / DIR.

Uso: `soup eval checklist [OPTIONS] SPEC_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SPEC_PATH` | text |  | yes | Ruta al YAML DSL de CheckList. |
| `--evidence`, `-e` | text |  |  | JSON obligatorio con respuestas por prueba proporcionadas por el operador. |
| `--output`, `-o` | text |  |  | Ruta donde escribir el JSON CheckListReport generado. |

### `soup eval citation`

Puntúa precisión / recall / F1 de citas en un JSONL (#202).

Uso: `soup eval citation [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | JSONL de filas {predicted, expected_ids} O filas RAFT {query, golden_doc, distractor_docs, answer}. |
| `--style` | text | `bracket` |  | Estilo de cita: bracket / inline / footnote. |
| `--shuffle-seed` | integer |  |  | For RAFT-shaped rows: the data.raft_shuffle_seed used at train time, so the golden \[doc-N] id matches what the model saw. Ignored for {predicted, expected_ids} rows. |
| `--output`, `-o` | text |  |  | Escribe aquí el JSON CitationScore por fila + agregado. |

### `soup eval compare`

Compara los resultados de evaluación de dos ejecuciones de entrenamiento.

Uso: `soup eval compare RUN1 RUN2`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN1` | text |  | yes | Id de la primera ejecución |
| `RUN2` | text |  | yes | Id de la segunda ejecución |

### `soup eval coverage`

Análisis heurístico de cobertura / brechas de una suite bloqueada o en borrador.

Uso: `soup eval coverage [OPTIONS] DESIGN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DESIGN_PATH` | text |  | yes | Ruta a un JSON EvalDesign |
| `--task` | text |  | yes | Task category from v0.54.0 taxonomy: factual_lookup \| style_shaping \| format_conversion \| reasoning \| tool_use \| summarization \| classification. |

### `soup eval custom`

Ejecuta tareas de evaluación personalizadas desde un archivo JSONL.

Uso: `soup eval custom [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--tasks`, `-t` | text |  | yes | Ruta al archivo JSONL de tareas de evaluación |
| `--model`, `-m` | text |  | yes | Ruta al directorio del modelo |
| `--run-id` | text |  |  | Vincula los resultados a una ejecución de entrenamiento existente |
| `--attach-to-registry` | text |  |  | Adjunta el JSON de evaluación a una entrada del registro como kind=eval_results |
| `--output`, `-o` | text |  |  | Ruta para escribir el JSON de evaluación. Se aplica independientemente de --attach-to-registry (v0.40.1 / G10). |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |

### `soup eval design`

Crea un borrador de suite de evaluación a partir de datos de entrenamiento + un objetivo de una línea.

Uso: `soup eval design [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Ruta al JSONL de datos de entrenamiento |
| `--goal`, `-g` | text |  | yes | Descripción del objetivo en una línea (p. ej., 'mejor en SQL'). |
| `--num-dimensions`, `-n` | integer | `5` |  | Número de dimensiones de evaluación que se propondrán (1-20). |
| `--output`, `-o` | text | `evals/design.json` |  | Ruta donde escribir el JSON EvalDesign generado. |

### `soup eval discover`

Descubre un conjunto de canarios reservado para detectar regresiones.

Uso: `soup eval discover [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Ruta al JSONL de datos de entrenamiento |
| `--base` | text |  |  | Id del modelo base (se registra; lo usa `soup diagnose`). |
| `--num-clusters` | integer | `5` |  | Número de clusters de comportamiento que se descubrirán (1-64). |
| `--per-cluster` | integer | `3` |  | Canarios reservados que se tomarán por cluster (1-64). |
| `--seed` | integer | `0` |  | Semilla determinista. |
| `--output`, `-o` | text | `evals/canaries.json` |  | Ruta donde escribir el JSON CanarySet generado. |

### `soup eval gate`

Ejecuta una suite eval-gate de forma independiente (veredicto posterior).

Uso: `soup eval gate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--suite`, `-s` | text |  | yes | Ruta al YAML de la suite de evaluación (consulta el ejemplo evals/gate.yaml) |
| `--baseline`, `-b` | text |  |  | Referencia: registry://&lt;id> o ruta a un JSON de puntuación plano o con sello |
| `--regression-threshold` | float | `0.05` |  | Caída absoluta máxima frente a la referencia antes de activar una regresión (0.0-1.0) |
| `--model`, `-m` | text |  |  | Ruta del modelo o id HF que se evaluará (obligatorio para puntuación en vivo) |
| `--write-baseline` | text |  |  | Escribe las puntuaciones de tareas de esta ejecución como JSON de referencia con sello ({'scores', 'provenance'}) que puede usar --baseline (#404). |

### `soup eval gate-install`

Instala una compuerta de regresión pre-push para la referencia seleccionada.

Uso: `soup eval gate-install [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--baseline` | text |  | yes | Id de ejecución de referencia con el que se compara el hook pre-push. |
| `--suite` | text | `evals/locked.json` |  | Ruta a la suite de evaluación bloqueada (dentro del cwd). |
| `--hook-path` | text | `.git/hooks/pre-push` |  | Destino del hook; normalmente .git/hooks/pre-push. |
| `--force` | flag | off |  | Sobrescribe un hook existente. |

### `soup eval human`

Ejecuta una evaluación A/B humana entre dos modelos.

Uso: `soup eval human [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Ruta al archivo JSONL con prompts de evaluación |
| `--model-a`, `-a` | text |  | yes | Ruta al primer modelo |
| `--model-b`, `-b` | text |  | yes | Ruta al segundo modelo |
| `--output`, `-o` | text | `human_eval_results.json` |  | Archivo de salida para los resultados |

### `soup eval irt-subset`

Elige un subconjunto de evaluación de costo mínimo que conserve el poder de clasificación.

Uso: `soup eval irt-subset [OPTIONS] RESPONSES_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RESPONSES_PATH` | text |  | yes | Ruta al JSONL con filas {item_id, correct(bool), respondent_id? (obligatorio para 2pl/3pl)}. |
| `--size`, `-z` | text | `small` |  | Perfil del subconjunto: full / small / tiny. |
| `--model`, `-m` | text | `1pl` |  | Modelo IRT: 1pl (Rasch) / 2pl (discriminación) / 3pl (+ adivinación). |
| `--output`, `-o` | text |  |  | Ruta donde escribir el JSON IrtSubsetPlan generado. |

### `soup eval judge`

Evalúa salidas del modelo mediante un evaluador LLM.

Uso: `soup eval judge [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target` | text |  | yes | Ruta al JSONL con pares de prompt+respuesta que se evaluarán |
| `--model`, `-m` | text | `gpt-4o-mini` |  | Nombre del modelo evaluador |
| `--provider` | text | `openai` |  | Proveedor del evaluador: openai, server, ollama |
| `--rubric`, `-r` | text |  |  | Ruta al archivo YAML de rúbrica |
| `--api-base` | text |  |  | URL base de la API para el modelo evaluador |
| `--run-id` | text |  |  | Vincula los resultados a una ejecución de entrenamiento existente |

### `soup eval leaderboard`

Muestra la tabla de clasificación local de todos los modelos evaluados.

Uso: `soup eval leaderboard [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--sort-by`, `-s` | text |  |  | Ordena por un benchmark específico (por defecto: promedio) |
| `--format`, `-f` | text | `table` |  | Formato de salida: table, json, csv |

### `soup eval lock`

Congela un EvalDesign como artefacto eval_suite con checksum.

Uso: `soup eval lock [OPTIONS] DESIGN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DESIGN_PATH` | text |  | yes | Ruta a un JSON EvalDesign (de `soup eval design`) |
| `--output`, `-o` | text | `evals/locked.json` |  | Ruta donde escribir la suite bloqueada canonizada. |
| `--attach-to-registry` | text |  |  | Id o nombre de entrada del registro a la que se adjuntará la suite bloqueada. |

### `soup eval quant-check`

Compara la precisión antes y después de la cuantización con la misma suite de evaluación.
Uso: `soup eval quant-check [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--before` | text |  | yes | Ruta del modelo anterior a la cuantización o registry://&lt;id> |
| `--after` | text |  | yes | Ruta del modelo posterior a la cuantización o registry://&lt;id> |
| `--tasks` | text |  | yes | Archivo JSONL de tareas de evaluación (consulta 'soup eval custom') |
| `--format` | text | `table` |  | Output format: table \| json \| markdown |
| `--allow-stub` | flag | off |  | Permite generadores stub deterministas si falla la carga del modelo activo. |

### `soup eval unlearning`

Puntúa una ejecución de unlearning en TOFU / MUSE / WMDP.

Uso: `soup eval unlearning [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Identificador de ejecución (p. ej., id de registro de la ejecución de unlearning). |
| `--benchmark`, `-b` | text | `tofu` |  | Benchmark: tofu / muse / wmdp. |
| `--evidence`, `-e` | text |  |  | Ruta a un archivo JSON con evidencia precalculada. Esquema: {forget_quality: {pre_loss, post_loss}, model_utility: {pre_acc, post_acc}, priv_leak: {mia_auc}}. Las claves ausentes reciben una puntuación neutral OK. |
| `--output`, `-o` | text |  |  | Ruta donde escribir el JSON UnlearnReport generado. |
| `--attach-to-registry` | text |  |  | Id de entrada del registro a la que adjuntar el informe --output como artefacto eval_results (requiere --output). Termina con código 1 si falla la adjunción. |

## `soup expect`

Ejecuta una suite de expectativas sobre un dataset JSONL.

Uso: `soup expect [OPTIONS] DATA SUITE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Ruta al dataset JSONL |
| `SUITE` | text |  | yes | Ruta al YAML de la suite de expectativas |
| `--judge` | text |  |  | URL del modelo evaluador (p. ej., ollama://llama3.1, https://api.openai.com/gpt-4o-mini) |

## `soup export`

Exporta un modelo a formato GGUF, ONNX, TensorRT-LLM, AWQ, GPTQ o TorchAO.

Uso: `soup export [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Ruta al directorio del modelo (modelo completo o adaptador LoRA) |
| `--format`, `-f` | text | `gguf` |  | Formato de exportación: gguf, onnx, tensorrt, awq, gptq, bitnet, tq1_0 (awq y gptq están obsoletos y se eliminarán en Soup v0.77) |
| `--quant`, `-q` | text | `q4_k_m` |  | Tipo de cuantización: q4_0, q4_k_m, q5_k_m, q8_0, f16, f32 |
| `--output`, `-o` | text |  |  | Ruta del archivo de salida. Por defecto: &lt;model-name>.&lt;quant>.gguf |
| `--base`, `-b` | text |  |  | Id del modelo base (para adaptadores LoRA). Se detecta automáticamente si no se define. |
| `--llama-cpp` | text |  |  | Path to llama.cpp directory. Auto-detected or cloned to \~/.soup/llama.cpp |
| `--onnx-task` | text | `text-generation` |  | Tarea de exportación ONNX: text-generation (LM causal) o feature-extraction (embedding) |
| `--deploy` | text |  |  | Implementa automáticamente después de exportar. Actualmente admite: ollama |
| `--deploy-name` | text |  |  | Nombre del modelo para la implementación (se usa con --deploy) |
| `--bits` | integer | `4` |  | Bits de cuantización para AWQ/GPTQ: 4 u 8 |
| `--group-size` | integer | `128` |  | Tamaño de grupo para la cuantización AWQ/GPTQ |
| `--calibration-data` | text |  |  | Ruta al JSONL de calibración. Obligatoria para AWQ y GPTQ. |
| `--calibration-samples` | integer | `128` |  | Número de muestras de calibración para AWQ/GPTQ |
| `--registry-id` | text |  |  | Adjunta el artefacto exportado a esta entrada del registro (por defecto: coincidencia automática según el directorio de salida del modelo --model de origen) |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--quant-config` | text |  |  | Ruta al YAML para exportación PTQ de torchao (v0.53.1 #142). Obligatoria cuando --format=torchao. |
| `--gguf-flavour` | text |  |  | Advanced GGUF format flag — UD-Q\*\_K_XL / IQ\*\_M / Q4_0_4_4 / etc. Required when --format=gguf-ud (v0.53.1 #139). |
| `--hub` | text | `hf` |  | Hub de origen del modelo base cuando --model es un adaptador LoRA: hf (por defecto) / modelscope / modelers (v0.53.10 #152). |

## `soup fetch`

Obtiene del catálogo incluido una configuración de ejemplo lista para editar.

Uso: `soup fetch [OPTIONS] NAMESPACE [NAME]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAMESPACE` | text |  | yes | Uno de: examples, configs, deepspeed_configs. |
| `NAME` | text |  |  | Nombre de la entrada del catálogo (omítelo para enumerarlas). |
| `--output`, `-o` | text |  |  | Ruta de destino (por defecto: ./&lt;filename> en el cwd). |
| `--force` | flag | off |  | Sobrescribe si --output ya existe. |

## `soup history`

Muestra el árbol de linaje de todas las entradas con un nombre determinado.

Uso: `soup history NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre de entrada del registro que se rastreará (p. ej., 'medical-chat') |

## `soup infer`

Ejecuta inferencia por lotes sobre un archivo JSONL de prompts.

Uso: `soup infer [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Ruta al modelo (adaptador LoRA o modelo completo) |
| `--input`, `-i` | text |  | yes | Ruta al archivo JSONL de entrada (cada línea: {"prompt": "..."}) |
| `--output`, `-o` | text |  | yes | Ruta al archivo JSONL de salida para los resultados |
| `--base`, `-b` | text |  |  | Modelo base para el adaptador LoRA (se detecta automáticamente si no se define) |
| `--max-tokens` | integer, 1 to 16384 | `256` |  | Máximo de tokens que se generarán por respuesta (1-16384) |
| `--temperature`, `-t` | float | `0.7` |  | Temperatura de muestreo (0 = codicioso) |
| `--batch-size` | integer, at least 1 | `1` |  | Número de prompts de LM causal que se generarán a la vez (por defecto: 1; el procesamiento por lotes es opcional; el procesamiento fp16/bf16 puede no ser idéntico bit a bit al tamaño de lote 1) |
| `--device` | text |  |  | Dispositivo: cuda, mps, cpu. Se detecta automáticamente si no se define. |
| `--task` | text | `text` |  | Inference task: 'text' (default, chat generation) or 'asr' (Whisper transcription; input rows are {"audio": path\[, "text": reference]}). |
| `--asr-language` | text |  |  | Idioma de decodificación ASR (anula el sidecar de entrenamiento). Solo con --task asr. |
| `--asr-task` | text |  |  | ASR decode objective: transcribe \| translate. --task asr only. |
| `--audio-dir` | text |  |  | Directorio base: las rutas de audio en --input deben permanecer dentro de este directorio (--task asr; por defecto, el directorio del archivo --input). Se rechazan rutas UNC o de recorrido. |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--hub` | text | `hf` |  | Hub de origen del modelo base: hf (por defecto) / modelscope / modelers. Los hubs que no son HF requieren el SDK correspondiente (v0.53.10 #152). |
| `--cuda-graphs` | flag | off |  | Decodificación experimental con CUDA graph para modelos Qwen2/Llama residentes y sin cuantizar. |

## `soup ingest`

Importador universal de traces: Langfuse / LangSmith / Helicone / OpenPipe / OTel / OpenAI Stored Completions (v0.63.0).

Uso: `soup ingest [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--source` | text |  | yes | Trace source: langfuse \| langsmith \| helicone \| openpipe \| otel \| openai-stored |
| `--logs` | text |  |  | Ruta a la exportación de traces JSONL (un evento por línea). Obligatoria salvo que se use --pull. |
| `--output`, `-o` | text |  |  | JSONL de salida (por defecto: traces.jsonl en cwd). |
| `--pull` | flag | off |  | Obtiene generaciones en vivo desde la API de Langfuse en lugar de leer --logs (solo --source langfuse). Lee LANGFUSE_PUBLIC_KEY y LANGFUSE_SECRET_KEY, además de LANGFUSE_HOST para otra región o instancia autoalojada. |
| `--since` | text |  |  | Con --pull: cuánto tiempo atrás buscar, p. ej., 30m, 24h, 7d (por defecto 7d, máximo 365d). |
| `--max-pages` | integer, 1 to 10000 |  |  | Con --pull: termina con error y no escribe nada si aún hay resultados pendientes después de estas páginas de 100 generaciones (por defecto 100). |
| `--allow-private-host` | flag | off |  | Con --pull: permite que LANGFUSE_HOST sea una dirección privada o loopback (Langfuse autoalojado). Se sigue exigiendo HTTPS. |
| `--slack-url` | text |  |  | URL opcional del webhook de Slack: se envía un POST al terminar. Validada contra SSRF. |
| `--discord-url` | text |  |  | URL opcional del webhook de Discord: se envía un POST al terminar. Validada contra SSRF. |

## `soup init`

Crea una configuración soup.yaml nueva de forma interactiva o a partir de una plantilla.

Uso: `soup init [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--template`, `-t` | text |  |  | Plantilla: audio, bco, chat, code, embedding, eu-ai-act, hipaa, ipo, kto, longcontext, medical, moe, orpo, pretrain, reasoning, rlhf, simpo, soc2, sr-11-7, tool-calling, vision |
| `--output`, `-o` | text | `soup.yaml` |  | Ruta del archivo de configuración de salida |
| `--force`, `-f` | flag | off |  | Sobrescribe la configuración existente sin preguntar (v0.40.1 / M2). |

## `soup iterative-dpo`

Ciclo iterativo de DPO para muestrear, puntuar, emparejar y entrenar.

Uso: `soup iterative-dpo [OPTIONS] [<command>] [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base-model` | text |  | yes | Id HF / ruta local |
| `--reward-model` | text |  | yes | Id HF / ruta del RM |
| `--prompts` | text |  | yes | JSONL de prompts de origen |
| `--output-dir` | text |  | yes | Directorio de salida |
| `--rounds` | integer | `3` |  | Número de rondas de DPO iterativo |
| `--pairs-per-round` | integer | `500` |  | Número de pares (chosen, rejected) por ronda |
| `--plan-only` | flag | off |  | Imprime el plan resuelto y termina (sin entrenamiento). |

## `soup license-advisor`

Recomienda un modelo base compatible con licencias para un destino de implementación (b2c / defense / embedded) + señala riesgos posteriores (v0.64.0).

Uso: `soup license-advisor [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target`, `-t` | text |  | yes | Deploy target: b2c \| defense \| embedded. |
| `--license` | text |  |  | Opcional: comprobación de riesgos por licencia (p. ej., apache-2.0, llama-3). |
| `--monthly-active-users`, `--mau` | integer | `0` |  | MAU esperado para la comprobación de riesgos por licencia (por defecto 0). |

## `soup llama`

Proxy para los binarios de llama.cpp (cli / mtmd-cli / gguf-split / server).

Uso: `soup llama [<command>] [ARGS]...`

Subcomandos:

- [`cli`](#soup-llama-cli)
- [`gguf-split`](#soup-llama-gguf-split)
- [`mtmd-cli`](#soup-llama-mtmd-cli)
- [`quantize`](#soup-llama-quantize)
- [`server`](#soup-llama-server)

### `soup llama cli`

Reenvía argumentos al binario llama-cli de llama.cpp.

Uso: `soup llama cli [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Argumentos reenviados a llama-cli. |

### `soup llama gguf-split`

Reenvía argumentos al binario llama-gguf-split de llama.cpp.

Uso: `soup llama gguf-split [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Argumentos reenviados a llama-gguf-split. |

### `soup llama mtmd-cli`

Reenvía argumentos al binario llama-mtmd-cli de llama.cpp.

Uso: `soup llama mtmd-cli [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Argumentos reenviados a llama-mtmd-cli. |

### `soup llama quantize`

Reenvía argumentos al binario llama-quantize de llama.cpp.

Uso: `soup llama quantize [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Argumentos reenviados a llama-quantize. |

### `soup llama server`

Reenvía argumentos al binario llama-server de llama.cpp.

Uso: `soup llama server [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Argumentos reenviados a llama-server. |

## `soup local-rl`

Daemon de ciclo de mejora de LLM personal (init / status / record / harvest / train) (v0.68.0).

Uso: `soup local-rl <command> [ARGS]...`

Subcomandos:

- [`harvest`](#soup-local-rl-harvest)
- [`init`](#soup-local-rl-init)
- [`record`](#soup-local-rl-record)
- [`status`](#soup-local-rl-status)
- [`train`](#soup-local-rl-train)

### `soup local-rl harvest`

Recopila pares DPO de las valoraciones con pulgar y los guarda en un archivo JSONL.

Uso: `soup local-rl harvest [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Ruta a la base SQLite de local-RL |
| `--output`, `-o` | text | `dpo_pairs.jsonl` |  | Ruta JSONL de salida |

### `soup local-rl init`

Crea el esquema SQLite de local-RL.

Uso: `soup local-rl init [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Ruta a la base SQLite de local-RL |

### `soup local-rl record`

Agrega un registro de valoración positiva / negativa.

Uso: `soup local-rl record [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Ruta a la base SQLite de local-RL |
| `--prompt` | text |  | yes | Texto del prompt |
| `--response` | text |  | yes | Respuesta del modelo |
| `--thumb` | text |  | yes | up / down |

### `soup local-rl status`

Imprime los contadores de la base de datos local-RL.

Uso: `soup local-rl status [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Ruta a la base SQLite de local-RL |

### `soup local-rl train`

Ejecuta (``--once``) o programa el entrenamiento nocturno DPO/KTO/ORPO.

Uso: `soup local-rl train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Ruta a la base SQLite de local-RL |
| `--backend` | text |  |  | Se acepta por compatibilidad; no afecta al entrenamiento. Permitidos: mlx, ollama |
| `--model` | text |  | yes | Base de entrenamiento: id de repo HF o ruta local (NO la etiqueta de Ollama) |
| `--train-method` | text | `dpo` |  | Permitidos: dpo, kto, orpo |
| `--once` | flag | off |  | Ejecuta el entrenamiento ahora (puntual); omite la instalación del scheduler. |
| `--min-pairs` | integer | `10` |  | Omite el entrenamiento si se recopilan menos pares. |
| `--output`, `-o` | text | `local_rl_adapter` |  | Directorio de salida del adaptador |
| `--scheduler-dir` | text | `local-rl-scheduler` |  | Directorio donde generar la estructura de systemd / launchd. |
| `--hour` | integer | `3` |  | Hora diaria del entrenamiento (hora local). |
| `--minute` | integer | `0` |  | Minuto diario del entrenamiento. |

## `soup lock`

Archivo lock compartido de ejecuciones (escribir / mostrar / comprobar).

Uso: `soup lock <command> [ARGS]...`

Subcomandos:

- [`check`](#soup-lock-check)
- [`show`](#soup-lock-show)
- [`write`](#soup-lock-write)

### `soup lock check`

Rechaza con código de salida 2 si el lock se desvió del estado actual.

Uso: `soup lock check [OPTIONS] [PATH]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text | `soup.lock` |  | Ruta al soup.lock seguido |
| `--base-sha` | text |  | yes | SHA de 64 dígitos hexadecimales del modelo base actual |
| `--dataset-sha` | text |  | yes | SHA de 64 dígitos hexadecimales del dataset actual |
| `--env-hash` | text |  | yes | Hash de entorno actual de 64 dígitos hexadecimales |
| `--base-model` | text |  | yes | Id del modelo base actual |

### `soup lock show`

Imprime un archivo lock seguido.

Uso: `soup lock show [PATH]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text | `soup.lock` |  | Ruta a soup.lock |

### `soup lock write`

Genera un ``soup.lock`` a partir de los hashes base/dataset/entorno.

Uso: `soup lock write [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base-model` | text |  | yes | Id / ruta de modelo HF |
| `--base-sha` | text |  | yes | SHA de 64 dígitos hexadecimales del modelo base |
| `--dataset-sha` | text |  | yes | SHA de 64 dígitos hexadecimales del dataset |
| `--env-hash` | text |  |  | Hash de entorno de 64 dígitos hexadecimales. Si se omite, se deriva automáticamente de --env-lock (soup-env.lock) mediante `soup env lock`. |
| `--env-lock` | text | `soup-env.lock` |  | Ruta a soup-env.lock desde donde derivar --env-hash automáticamente (solo se usa cuando se omite --env-hash). |
| `--output`, `-o` | text | `soup.lock` |  | Ruta de salida |

## `soup loop`

Ciclo de mejora de datos: traces -> pares de preferencias -> DPO -> compuerta -> implementación canary -> reversión, todo desde la CLI (v0.58.0).

Uso: `soup loop <command> [ARGS]...`

Subcomandos:

- [`canary`](#soup-loop-canary)
- [`init`](#soup-loop-init)
- [`pause`](#soup-loop-pause)
- [`replay`](#soup-loop-replay)
- [`resume`](#soup-loop-resume)
- [`status`](#soup-loop-status)
- [`watch`](#soup-loop-watch)

### `soup loop canary`

Promueve ``new_adapter`` como canary con una división de tráfico.

Uso: `soup loop canary [OPTIONS] NEW_ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NEW_ADAPTER` | text |  | yes | Id/ruta del adaptador que será canary. |
| `--traffic` | text | `5%` |  | Porcentaje de tráfico, p. ej., "5%" o "5". |
| `--autoroll-on-regress` / `--no-autoroll-on-regress` | flag | on |  | Revierte automáticamente ante un veredicto MAJOR. |

### `soup loop init`

Crea el archivo de control .soup/loop.yaml (configuración inicial única).

Uso: `soup loop init [OPTIONS] SERVED_MODEL`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SERVED_MODEL` | text |  | yes | Id del modelo servido (p. ej., registry://abc12). |
| `--eval` | text |  | yes | Ruta de suite de evaluación o referencia del registro. |
| `--baseline` | text |  | yes | Id del registro o archivo de referencia. |
| `--monthly-budget` | text |  |  | Límite mensual en USD (p. ej., 50usd, 100). |
| `--max-runs-per-day` | integer |  |  | Límite de inicios de iteración por día UTC. |
| `--pre-wired` | flag | off |  | Usa las etapas harvest/train/gate/deploy preconfiguradas (utils/loop_stages.py) durante watch en vez de stubs vacíos (v0.71.4). |
| `--force` | flag | off |  | Sobrescribe el loop.yaml existente. |

### `soup loop pause`

Pausa el daemon watch en el siguiente límite de iteración.

Uso: `soup loop pause`

No acepta opciones ni argumentos.

### `soup loop replay`

Reproduce un manifiesto de iteración del ciclo registrado.

Uso: `soup loop replay [OPTIONS] [ITERATION_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ITERATION_ID` | text |  |  | Id de iteración (omítelo para enumerarlas todas). |
| `--extract` | text |  |  | Extrae el .can de la iteración en este directorio para volver a ejecutarlo como prueba hipotética (v0.71.4). |

### `soup loop resume`

Reanuda un ciclo pausado (la siguiente iteración watch se reanudará automáticamente).

Uso: `soup loop resume`

No acepta opciones ni argumentos.

### `soup loop status`

Muestra los contadores y el estado actual.

Uso: `soup loop status`

No acepta opciones ni argumentos.

### `soup loop watch`

Ejecuta el daemon harvest → train → gate → deploy.

Uso: `soup loop watch [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--foreground` | flag | off |  | Ejecuta en primer plano (por defecto). |
| `--detach` | flag | off |  | Inicia un subproceso en segundo plano que ejecuta --foreground. |
| `--max-iterations` | integer |  |  | Se detiene después de N iteraciones (para pruebas/demos). |
| `--poll-interval` | float | `60.0` |  | Seconds between iterations \[1, 3600]. |
| `--pre-wired` | flag | off |  | Fuerza las etapas de producción preconfiguradas aunque loop.yaml no las haya definido. |
| `--pack-cans` | flag | off |  | Empaqueta cada iteración como Soup Can v0.26 + entrada del registro (v0.71.4). |

## `soup mcp`

Servidor Model Context Protocol: controla Soup desde cualquier cliente MCP (v0.71.28).

Uso: `soup mcp <command> [ARGS]...`

Subcomandos:

- [`runs`](#soup-mcp-runs)
- [`serve`](#soup-mcp-serve)

### `soup mcp runs`

Recupera ejecuciones MCP persistidas.

Uso: `soup mcp runs <command> [ARGS]...`

Subcomandos:

- [`reconcile`](#soup-mcp-runs-reconcile)

#### `soup mcp runs reconcile`

Recupera capacidad de ejecución bloqueada por un servidor MCP bloqueado.

Uso: `soup mcp runs reconcile [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--expunge-launching` | flag | off |  | Elimina filas launching obsoletas después de comprobar que ningún PID registrado siga activo. |
| `--older-than-seconds` | integer, at least 1 | `300` |  | Solo considera filas launching con al menos esta antigüedad en segundos. |

### `soup mcp serve`

Inicia el servidor MCP (herramientas de solo lectura por defecto).

Uso: `soup mcp serve [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--allow-mutating` | flag | off |  | Habilita las herramientas de modificación solo de planificación (train_start / export). Incluso habilitadas, solo muestran el comando que se ejecutaría: v1 nunca ejecuta entrenamiento ni exportación. Desactivadas por defecto: esas herramientas se rechazan. |
| `--allow-execute` | flag | off |  | Reserva la compuerta de ejecución para futuras herramientas MCP. Implica --allow-mutating, pero train_start y export siguen siendo solo de planificación y nunca se ejecutan en esta versión. |
| `--transport` | text | `stdio` |  | stdio (por defecto), sse o http. stdio usa JSON-RPC por las tuberías del proceso y es lo que necesita un cliente que inicia Soup. sse y http abren un listener local en su lugar, para configuraciones remotas o con varios clientes, y requieren un token Bearer. |
| `--host` | text |  |  | Interfaz a la que enlazarse con --transport sse/http. Por defecto es 127.0.0.1. Cualquier otro enlace expone el servidor a la red, donde el token Bearer es la única barrera. |
| `--port` | integer |  |  | Puerto de enlace para --transport sse/http. Por defecto es 8765. |
| `--auth-token` | text |  |  | Anula el token Bearer generado automáticamente (16-128 caracteres base64 seguros para URL) para --transport sse/http. Si se omite, se genera e imprime un token nuevo en cada inicio. |

## `soup merge`

Fusiona un adaptador LoRA con su modelo base para crear un modelo completo.

Uso: `soup merge [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--adapter`, `-a` | text |  | yes | Ruta al directorio del adaptador LoRA |
| `--base`, `-b` | text |  |  | Id del modelo base. Se detecta automáticamente desde adapter_config.json si no se define. |
| `--output`, `-o` | text | `./merged` |  | Directorio de salida para el modelo fusionado |
| `--dtype` | text | `float16` |  | Tipo de datos del modelo fusionado: float16, bfloat16, float32 |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--save-format` | text | `fp16` |  | Formato para guardar el checkpoint fusionado. fp16 (por defecto) escribe una fusión fp16 estándar. 4bit / 4bit_forced escriben una sola fusión cuantizada con BNB-4bit, sin el ciclo descuantizar-fusionar-recuantizar (v0.53.1 #142). |
| `--double-quant` / `--no-double-quant` | flag | on |  | Doble cuantización BNB para los formatos 4bit / 4bit_forced (#321). Activada por defecto, igual que los cargadores de entrenamiento; pasa --no-double-quant para desactivarla. Se ignora con el formato fp16. |
| `--hub` | text | `hf` |  | Hub de origen del modelo base: hf (por defecto) / modelscope / modelers. Los hubs que no son HF requieren el SDK correspondiente (v0.53.10 #152). |

## `soup merge-sharded-fsdp-weights`

Consolida archivos de shards FSDP en un único archivo safetensors.

Uso: `soup merge-sharded-fsdp-weights [OPTIONS] SHARD_DIR`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SHARD_DIR` | text |  | yes | Directory containing pytorch_model_fsdp\_\*.bin shard files. |
| `--output`, `-o` | text |  | yes | Ruta del archivo .safetensors de destino (dentro del cwd). |
| `--plan-only` | flag | off |  | Imprime el plan de consolidación y termina sin escribir nada. |

## `soup migrate`

Importa una configuración de LLaMA-Factory, Axolotl o un notebook Unsloth.

Uso: `soup migrate [OPTIONS] CONFIG_FILE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--from` | text |  | yes | Herramienta de origen: llamafactory, axolotl o unsloth |
| `CONFIG_FILE` | text |  | yes | Ruta al archivo de configuración de origen (.yaml o .ipynb) |
| `--output`, `-o` | text | `soup.yaml` |  | Ruta de salida para el soup.yaml generado |
| `--dry-run` | flag | off |  | Imprime la configuración generada sin escribir en el archivo |
| `--yes`, `-y` | flag | off |  | Omite los mensajes de confirmación |

## `soup monitor`

Monitor de GPU en vivo: utilización / temperatura / VRAM / energía por GPU.

Uso: `soup monitor [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--refresh`, `-r` | float | `2.0` |  | Intervalo de actualización en segundos (0.25 a 30). |
| `--once` | flag | off |  | Imprime una instantánea y termina (omite el panel en vivo). |

## `soup plan`

Genera un plan previo al entrenamiento (costo / ETA / SHA / VRAM) y escribe soup.tfstate para que lo consulte `soup apply` (v0.64.0).

Uso: `soup plan [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta a soup.yaml. |
| `--state` | text | `soup.tfstate` |  | Ruta donde escribir el archivo de estado (por defecto: ./soup.tfstate). |

## `soup plugins`

Descubre, enumera, habilita y deshabilita plugins de Soup.

Uso: `soup plugins [<command>] [ARGS]...`

Subcomandos:

- [`disable`](#soup-plugins-disable)
- [`enable`](#soup-plugins-enable)
- [`install`](#soup-plugins-install)
- [`list`](#soup-plugins-list)

### `soup plugins disable`

Uso: `soup plugins disable NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre del plugin |

### `soup plugins enable`

Uso: `soup plugins enable NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre del plugin |

### `soup plugins install`

Explica cómo instalar plugins sin fingir que la instalación ya ocurrió.

Uso: `soup plugins install NAME`
| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre del plugin |

### `soup plugins list`

Enumera todos los plugins registrados.

Uso: `soup plugins list`

No acepta opciones ni argumentos.

## `soup probe`

Sondeos de activación: deserción de sleeper-agent / honestidad / uso indebido / interferencia por pares / diferencia de características SAE / paquete de sondeos (v0.66.0, truth+harm v0.71.8).

Uso: `soup probe <command> [ARGS]...`

Subcomandos:

- [`harm`](#soup-probe-harm)
- [`interference`](#soup-probe-interference)
- [`pack`](#soup-probe-pack)
- [`sae-diff`](#soup-probe-sae-diff)
- [`sleeper`](#soup-probe-sleeper)
- [`truth`](#soup-probe-truth)

### `soup probe harm`

Aplica a las activaciones el sondeo de uso indebido al estilo HarmBench (v0.71.8 #217).

Uso: `soup probe harm [OPTIONS] BASE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  | yes | Id del modelo base (debe tener un sondeo incluido) |
| `--evidence` | text |  |  | JSON with 'activations': \[\[...], ...] (2D float matrix) |
| `--weights` | text |  |  | Pesos de sondeo calibrados proporcionados por el operador (.npz / .npy / .safetensors) |
| `--output`, `-o` | text |  |  | Escribe el JSON de resultados del sondeo en esta ruta |

### `soup probe interference`

Matriz de interferencia de adaptadores por pares N×N (v0.66.0; --measure v0.71.8 #218).

Uso: `soup probe interference [OPTIONS] [LOSSES]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `LOSSES` | text |  |  | JSON: {adapters: \[...], losses: {'a\|b': loss_value, ...}} |
| `--measure` | text |  |  | Mide automáticamente las pérdidas en esta suite de evaluación JSONL en lugar de proporcionar un JSON de pérdidas. Requiere --base-model + ≥2 --adapter (v0.71.8 #218). |
| `--base-model` | text |  |  | Id del modelo base para --measure |
| `--adapter` | text, repeatable |  |  | Especificación de adaptador name=path para --measure (repetible) |
| `--device` | text |  |  | Dispositivo para --measure (cpu / cuda) |
| `--output`, `-o` | text |  |  | Escribe el JSON de la matriz en esta ruta |

### `soup probe pack`

Ensambla o enumera paquetes de sondeos calibrados para un modelo base (v0.66.0).

Uso: `soup probe pack [OPTIONS] [BASE]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  |  | Id del modelo base (p. ej., meta-llama/Llama-3-8B) |
| `--list` | flag | off |  | Enumera todos los modelos base que incluyen un paquete de sondeos |
| `--output`, `-o` | text |  |  | Escribe el JSON del manifiesto del paquete en esta ruta |

### `soup probe sae-diff`

Diferencia de características SAE entre activaciones previas y posteriores al FT (v0.66.0).

Uso: `soup probe sae-diff [OPTIONS] SAE PRE_ACTS POST_ACTS`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SAE` | text |  | yes | Ruta al checkpoint safetensors de SAE (o id de repo con --auto-download) |
| `PRE_ACTS` | text |  | yes | JSON: {'activations': \[\[...]]} |
| `POST_ACTS` | text |  | yes | JSON: {'activations': \[\[...]]} |
| `--top-k` | integer, 1 to 10000 | `20` |  | Máximo de características modificadas que se informarán |
| `--auto-download` | flag | off |  | Treat &lt;sae> as an HF Hub repo id and download it into \~/.soup/sae-cache/ first (must be in HF_HUB_ALLOWLIST; v0.71.8 #216). |
| `--output`, `-o` | text |  |  | Escribe el JSON del informe de diferencias en esta ruta |

### `soup probe sleeper`

Aplica a las activaciones el sondeo de deserción de sleeper-agent (v0.66.0).

Uso: `soup probe sleeper [OPTIONS] BASE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  | yes | Id del modelo base (debe tener un sondeo incluido) |
| `--evidence` | text |  |  | JSON with 'activations': \[\[...], ...] (2D float matrix) |
| `--weights` | text |  |  | Pesos de sondeo calibrados proporcionados por el operador (.npz / .npy / .safetensors). Si se define, reemplaza el fallback de semilla sintética y acepta cualquier id de modelo base (v0.71.8 #215). |
| `--output`, `-o` | text |  |  | Escribe el JSON de resultados del sondeo en esta ruta |

### `soup probe truth`

Aplica a las activaciones el sondeo de honestidad al estilo TruthfulQA (v0.71.8 #217).

Uso: `soup probe truth [OPTIONS] BASE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  | yes | Id del modelo base (debe tener un sondeo incluido) |
| `--evidence` | text |  |  | JSON with 'activations': \[\[...], ...] (2D float matrix) |
| `--weights` | text |  |  | Pesos de sondeo calibrados proporcionados por el operador (.npz / .npy / .safetensors) |
| `--output`, `-o` | text |  |  | Escribe el JSON de resultados del sondeo en esta ruta |

## `soup profile`

Calcula memoria, velocidad y requisitos de GPU ANTES del entrenamiento.

Uso: `soup profile [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta al archivo de configuración soup.yaml |
| `--gpu`, `-g` | text |  |  | GPU de destino para recomendaciones (p. ej., rtx3090, rtx4090, a100, h100). Se detecta automáticamente si no se define. |
| `--json` | flag | off |  | Emite JSON para usar en scripts |

## `soup prune-prompt`

Detecta y elimina un prefijo compartido de prompt del sistema en los datos de entrenamiento para que el modelo FT lo internalice (v0.63.0).

Uso: `soup prune-prompt [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | JSONL de entrada con filas {prompt, output} (p. ej., de `soup ingest`). |
| `--output`, `-o` | text |  | yes | JSONL de salida con el prefijo compartido eliminado de `prompt`. |
| `--min-frequency` | float | `0.95` |  | El prefijo debe aparecer en al menos esta fracción de filas para eliminarse (0.0 - 1.0). |
| `--tokenizer` | text |  |  | Tokenizador HF opcional (id del modelo o ruta local) para detectar prefijos teniendo en cuenta tokens. Por defecto: nivel de caracteres de espacios en blanco. |
| `--slack-url` | text |  |  | URL opcional del webhook de Slack: se envía un POST al terminar. Validada contra SSRF. |
| `--discord-url` | text |  |  | URL opcional del webhook de Discord: se envía un POST al terminar. Validada contra SSRF. |

## `soup push`

Publica un modelo entrenado en HuggingFace Hub (o en otro hub).

Uso: `soup push [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Ruta al modelo entrenado / directorio del adaptador LoRA |
| `--repo`, `-r` | text |  | yes | Id del repo de HuggingFace, p. ej., username/my-model |
| `--private` | flag | off |  | Hace privado el repo de HuggingFace |
| `--token`, `-t` | text |  |  | Upload token. For --hub hf: \[deprecated] use HF_TOKEN or cached login. For other hubs: that hub's token (or MODELSCOPE_API_TOKEN / MODELERS_TOKEN). Env: `HF_TOKEN`. |
| `--message` | text | `Upload model trained with Soup CLI` |  | Mensaje de commit para la carga |
| `--collection` | text |  |  | Agrega el repo publicado a una Collection de HF existente (slug: 'owner/title-hash') |
| `--hub` | text | `hf` |  | Hub de destino: hf (por defecto) / modelscope / modelers. Los hubs que no son HF requieren el SDK correspondiente y omiten la ruta de Collections específicas de HF y generación automática de fichas del modelo (v0.53.10 #152). |
| `--card` | text |  |  | Id, prefijo o name:tag de una entrada del registro: genera su ficha de modelo con datos de procedencia (mediante `soup card`) y la carga como README.md, reemplazando la ficha generada automáticamente (v0.71.35). Solo hub HF. |

## `soup quantize`

Cuantiza un modelo: alias sencillo de `soup export --format <to>`.

Uso: `soup quantize [OPTIONS] MODEL_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `MODEL_PATH` | text |  | yes | Checkpoint de origen (directorio safetensors o archivo .safetensors único). |
| `--to` | text | `gguf` |  | Target format: gguf \| gptq \| awq \| onnx \| tensorrt (awq and gptq are deprecated and will be removed in Soup v0.77). |
| `--bits` | integer | `4` |  | Bits de cuantización (1-16; se aplica a gguf/gptq/awq). |
| `--output`, `-o` | text |  |  | Directorio de destino (por defecto: &lt;model_path>-&lt;to>). |

## `soup quickstart`

Ejecuta una demostración completa: crea datos de muestra, configuración y entrena.

Uso: `soup quickstart [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--yes`, `-y` | flag | off |  | Omite el mensaje de confirmación |
| `--dry-run` | flag | off |  | Crea solo los datos y la configuración; no entrena |
| `--output`, `-o` | text |  |  | Directorio de salida para datos, configuración y artefactos de ejecución (por defecto: directorio actual). |

## `soup ra-dit`

Orquestador RA-DIT de dos etapas: recuperador -> generador (v0.71.10).

Uso: `soup ra-dit [OPTIONS] [<command>] [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--retriever-config`, `-r` | text |  | yes | Receta de embedding/contraste de etapa 1 (dentro del cwd). |
| `--generator-config`, `-g` | text |  | yes | Receta RAFT-SFT de etapa 2 (dentro del cwd). |
| `--retriever-model` | text |  |  | Anulación manual del modelo recuperador; omite el enlace automático del registro. |
| `--timeout` | integer | `21600` |  | Tiempo límite estricto por etapa, en segundos (60..21600). |
| `--plan-only` | flag | off |  | Valida las rutas de configuración y genera el plan; omite el entrenamiento. |

## `soup recipes`

Configuraciones listas para usar: enumera, muestra, usa y busca recetas para modelos populares.

Uso: `soup recipes <command> [ARGS]...`

Subcomandos:

- [`list`](#soup-recipes-list)
- [`search`](#soup-recipes-search)
- [`show`](#soup-recipes-show)
- [`use`](#soup-recipes-use)
- [`verify`](#soup-recipes-verify)

### `soup recipes list`

Enumera todas las recetas disponibles.

Uso: `soup recipes list`

No acepta opciones ni argumentos.

### `soup recipes search`

Busca recetas por palabra clave, tarea o tamaño del modelo.

Uso: `soup recipes search [OPTIONS] [QUERY]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `QUERY` | text |  |  | Palabra clave de búsqueda |
| `--task` | text |  |  | Filtra por tarea |
| `--size` | text |  |  | Filtra por tamaño del modelo |

### `soup recipes show`

Muestra la configuración de una receta (imprime YAML en stdout).

Uso: `soup recipes show NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre de la receta |

### `soup recipes use`

Copia una receta a soup.yaml (o a una ruta personalizada).

Uso: `soup recipes use [OPTIONS] NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Nombre de la receta |
| `--output`, `-o` | text | `soup.yaml` |  | Ruta de salida del archivo de configuración |
| `--yes`, `-y` | flag | off |  | Omite los mensajes de confirmación |

### `soup recipes verify`

Comprueba que una configuración pueda conectar un adaptador LoRA, sin descargar pesos.

Uso: `soup recipes verify [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text |  |  | Verifica una sola configuración en lugar de todo el catálogo. |
| `--json` | flag | off |  | Filas legibles por máquinas en stdout. |
| `--templates` / `--no-templates` | flag | on |  | También verifica src/soup_cli/templates y examples/ (modo catálogo). |

## `soup registry`

Registro de modelos: publica, enumera, muestra, compara, busca, promueve y elimina.

Uso: `soup registry <command> [ARGS]...`

Subcomandos:

- [`delete`](#soup-registry-delete)
- [`diff`](#soup-registry-diff)
- [`list`](#soup-registry-list)
- [`promote`](#soup-registry-promote)
- [`push`](#soup-registry-push)
- [`search`](#soup-registry-search)
- [`show`](#soup-registry-show)

### `soup registry delete`

Elimina una entrada del registro y sus filas de linaje / artefactos.

Uso: `soup registry delete [OPTIONS] REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Referencia de la entrada que se eliminará |
| `--yes`, `-y` | flag | off |  | Omite la confirmación |

### `soup registry diff`

Diferencia lado a lado de configuración + evaluación entre dos entradas.

Uso: `soup registry diff LEFT RIGHT`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `LEFT` | text |  | yes | Referencia de la entrada izquierda |
| `RIGHT` | text |  | yes | Referencia de la entrada derecha |

### `soup registry list`
Enumera las entradas del registro.

Uso: `soup registry list [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--name`, `-n` | text |  |  |  |
| `--tag`, `-t` | text |  |  |  |
| `--base`, `-b` | text |  |  |  |
| `--task` | text |  |  |  |
| `--limit`, `-l` | integer | `100` |  |  |

### `soup registry promote`

Agrega una etiqueta a una entrada existente (p. ej., promoción a 'prod').

Uso: `soup registry promote [OPTIONS] REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Referencia de la entrada que se promoverá |
| `--tag`, `-t` | text |  | yes | Etiqueta que se asignará |

### `soup registry push`

Registra una ejecución de entrenamiento completada como entrada del registro.

Uso: `soup registry push [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--run-id` | text |  | yes | Id de ejecución de entrenamiento que se registrará |
| `--name`, `-n` | text |  | yes | Nombre de la entrada del registro |
| `--tag`, `-t` | text | `v1` |  | Etiqueta (p. ej., v1, prod-2024) |
| `--notes` | text | `""` |  | Notas de formato libre |

### `soup registry search`

Busca entradas del registro por nombre/base/tarea/notas.

Uso: `soup registry search [OPTIONS] QUERY`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `QUERY` | text |  | yes | Palabra clave que se buscará |
| `--limit`, `-l` | integer | `50` |  |  |

### `soup registry show`

Muestra todos los detalles de una entrada del registro.

Uso: `soup registry show REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Id, prefijo o name:tag de la entrada |

## `soup reward`

Sintetiza un verificador de recompensas determinista a partir de salidas de referencia, con un informe de calibración que rechaza verificadores degenerados (v0.71.40).

Uso: `soup reward <command> [ARGS]...`

Subcomandos:

- [`stress`](#soup-reward-stress)
- [`synth`](#soup-reward-synth)

### `soup reward stress`

Sondea adversarialmente si un verificador de recompensas se puede manipular (termina con código 0 si es robusto / 2 si es manipulable).

Uso: `soup reward stress [OPTIONS] REWARD_TARGET`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REWARD_TARGET` | text |  | yes | Verificador que se sondeará: una ruta .py, un nombre integrado o 'verifiable'. |
| `--references` | text |  |  | JSONL de salidas de referencia (habilita sondeos conscientes de la respuesta correcta). |
| `--field` | text | `answer` |  | Campo de respuesta correcta (por defecto: answer). |
| `--verifiable-domain` | text |  |  | Dominio para un destino 'verifiable'. |
| `--sentinel` | text | `GOLD` |  | Token de spam centinela. |
| `--threshold` | float | `0.5` |  | Recompensa >= este valor = aceptar. |
| `--max-gameable` | float | `0.0` |  | Tasa máxima permitida de aceptación de basura antes de que el veredicto cambie a manipulable. |
| `--attacks` | text | `empty,length,repetition,sentinel,wrapped_junk,answer_spray` |  | Lista separada por comas: empty,length,repetition,sentinel,wrapped_junk,answer_spray. |
| `--output-report` | text |  |  | También escribe el informe de estrés como JSON. |

### `soup reward synth`

Sintetiza un verificador de recompensas determinista a partir de salidas de referencia.

Uso: `soup reward synth [OPTIONS] REFERENCES`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REFERENCES` | text |  | yes | JSONL de salidas de referencia (correctas): campo gold o mensajes de chat. |
| `-o`, `--output` | text |  |  | Ruta donde escribir el verificador .py. |
| `--kind` | text | `auto` |  | Verifier family: auto \| numeric \| json_schema \| regex \| tool_call. |
| `--field` | text | `answer` |  | Campo de salida correcta (por defecto: answer). |
| `--tolerance` | float |  |  | Tolerancia de coincidencia numérica (solo para kind numeric). |
| `--min-discrimination` | float | `0.5` |  | Se niega a generar el resultado salvo que (accept_rate refs - accept_rate negatives) >= este valor. |
| `--force` | flag | off |  | Sobrescribe una salida existente. |
| `--plan-only` | flag | off |  | Detecta y muestra la especificación inducida; no escribe nada. |
| `--output-report` | text |  |  | También escribe el informe de calibración como JSON. |

## `soup rewind`

Identifica las filas del dataset detrás de un pico de loss, a partir del registrador de entrenamiento.

Uso: `soup rewind [OPTIONS] [RUN_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  |  | Id de ejecución (o prefijo). Por defecto, la ejecución más reciente. |
| `--step` | integer |  |  | Detalla este paso en lugar del peor pico. |
| `--top` | integer, at least 1 | `10` |  | Número de filas que se enumerarán para el paso detallado (al menos 1). |
| `--json` | text |  |  | Escribe en esta ruta como JSON los picos y las filas clasificadas (debe permanecer bajo el directorio actual). |
| `--no-preview` | flag | off |  | Omite las previsualizaciones de filas (no carga el dataset). |

## `soup runs`

Seguimiento de experimentos: enumera, muestra y compara ejecuciones.

Uso: `soup runs [OPTIONS] [<command>] [ARGS]...`

Subcomandos:

- [`clean`](#soup-runs-clean)
- [`compare`](#soup-runs-compare)
- [`curriculum-curve`](#soup-runs-curriculum-curve)
- [`delete`](#soup-runs-delete)
- [`replay`](#soup-runs-replay)
- [`show`](#soup-runs-show)

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--limit`, `-l` | integer | `20` |  | Máximo de ejecuciones que se mostrarán |
| `--cwd-only` | flag | off |  | Only show runs whose output_dir is under the current working directory (default: show all in \~/.soup/experiments.db). |

### `soup runs clean`

Limpia inteligentemente archivos de checkpoint redundantes para recuperar espacio en disco.

Uso: `soup runs clean [OPTIONS] [RUN_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  |  | Id de ejecución (o prefijo) que se limpiará. Omítelo si se usa --all. |
| `--all` | flag | off |  | Limpia todas las ejecuciones históricas. |
| `--dry-run` | flag | off |  | Calcula el espacio que se recuperaría sin eliminar archivos. |
| `--force`, `-f` | flag | off |  | Omite la confirmación |
| `--keep-weights` / `--no-keep-weights` | flag | on |  | Conserva los pesos intermedios del modelo y elimina solo los estados del optimizador y scheduler (por defecto). --no-keep-weights elimina todos los checkpoints que no sean el mejor. |

### `soup runs compare`

Compara lado a lado dos ejecuciones de entrenamiento.

Uso: `soup runs compare RUN_1 RUN_2`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_1` | text |  | yes | Id de la primera ejecución (o prefijo) |
| `RUN_2` | text |  | yes | Id de la segunda ejecución (o prefijo) |

### `soup runs curriculum-curve`

v0.48.0 (BETA): visualiza los pesos dinámicos de los grupos del curriculum.

Uso: `soup runs curriculum-curve [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Id de ejecución (o prefijo). |
| `--history` | text |  |  | Anula la ruta a curriculum_history.jsonl (por defecto, dentro del output_dir de la ejecución). |
| `--width` | integer, 4 to 200 | `10` |  | Ancho de columna por grupo. |

### `soup runs delete`

Elimina una ejecución de entrenamiento y sus métricas.

Uso: `soup runs delete [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Id de ejecución (o prefijo) que se eliminará |
| `--force`, `-f` | flag | off |  | Omite la confirmación |

### `soup runs replay`

Reproduce las métricas de una ejecución completada: vuelve a generar el resumen + curva de loss.

Uso: `soup runs replay [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Id de ejecución (o prefijo) que se reproducirá. |
| `--plot` / `--no-plot` | flag | on |  | Genera la curva de loss |

### `soup runs show`

Muestra información detallada de una ejecución específica, incluida la curva de loss.

Uso: `soup runs show [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Id de ejecución (o prefijo) que se mostrará |
| `--plot` / `--no-plot` | flag | on |  | Muestra la curva de loss |

## `soup serve`

Inicia un servidor de inferencia local con API compatible con OpenAI.

Uso: `soup serve [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Ruta al directorio del adaptador LoRA o al modelo completo |
| `--base`, `-b` | text |  |  | Id del modelo base. Se detecta automáticamente desde adapter_config.json si no se define. |
| `--port`, `-p` | integer | `8000` |  | Puerto en el que se servirá |
| `--host` | text | `127.0.0.1` |  | Host al que enlazarse. Por defecto es loopback (127.0.0.1). Cualquier otro host requiere --tool-auth-token, que solo se comprueba en las rutas de herramientas, valoraciones y adaptadores: las rutas de generación no requieren token. |
| `--device` | text |  |  | Dispositivo: cuda, cuda:&lt;index>, mps o cpu. Se detecta automáticamente si no se define. Con el backend transformers, cpu carga en CPU en float32 (salvo que --kv-cache-type bf16/f16 seleccione el tipo); cuda:&lt;index> y mps se fijan a ese dispositivo; cuda usa device_map=auto, todo en float16. |
| `--max-tokens` | integer | `512` |  | Máximo de tokens por defecto para la generación |
| `--backend` | text | `transformers` |  | Backend de inferencia: transformers (por defecto), vllm, sglang o mii |
| `--tensor-parallel`, `--tp` | integer | `1` |  | Número de GPU para paralelismo tensorial (solo vLLM) |
| `--gpu-memory` | float | `0.9` |  | Fracción de memoria de GPU que se usará (solo vLLM, 0.0-1.0) |
| `--max-model-len` | integer |  |  | Longitud máxima de secuencia para el motor vLLM (solo vLLM). Redúcela si el motor no inicia porque la caché KV no cabe. Por defecto: máximo del propio modelo. |
| `--speculative-decoding` | text |  |  | Modelo draft para speculative decoding (id o ruta de un modelo más pequeño/rápido) |
| `--num-speculative-tokens` | integer | `5` |  | Número de tokens que genera el modelo draft por paso (speculative decoding) |
| `--adapters` | text, repeatable |  |  | Adaptadores LoRA como pares name=path (repetible). P. ej., chat=./chat-adapter |
| `--prefix-cache` | flag | off |  | Habilita el almacenamiento en caché de prefijos de vLLM para prompts del sistema compartidos (cargas de trabajo RAG/agentes). |
| `--auto-spec` | flag | off |  | Empareja automáticamente un modelo draft para speculative decoding según el modelo de destino. |
| `--structured-output` | text | `off` |  | Constrain generation: off (default) \| json \| regex. |
| `--json-schema` | text |  |  | Ruta al archivo de esquema JSON (se usa con --structured-output json). |
| `--regex-pattern` | text |  |  | Patrón regex (se usa con --structured-output regex). |
| `--dashboard` | flag | off |  | Habilita el dashboard de continuous-batching en vivo + endpoint /metrics. |
| `--trace` | flag | off |  | Habilita el trazado de solicitudes OpenTelemetry (requiere opentelemetry-sdk). |
| `--trace-endpoint` | text |  |  | URL del endpoint OTLP (por defecto: http://localhost:4317). |
| `--auto-quant` | flag | off |  | No está disponible hasta que Soup pueda medir cada candidato cargado; el comando se niega en lugar de adivinar un formato. |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--trace-log` | text |  |  | Agrega por solicitud {prompt, response, latency_ms, tokens, ts} al JSONL de esta ruta. La ruta debe permanecer dentro del cwd. Rotación a 100 MB (se conserva una copia de respaldo). Agregado en v0.40.3 (#33). |
| `--trace-log-cap-mb` | integer | `100` |  | Límite de rotación en MB para --trace-log (1 - 10000). Por defecto 100. |
| `--record-thumbs` | text |  |  | Captura automáticamente comentarios positivos/negativos en una base SQLite local-rl en esta ruta mediante POST /v1/thumbs. La ruta debe permanecer dentro del cwd. Solo backend transformers. v0.71.1 (#230). |
| `--reasoning-parser` | text |  |  | Strip reasoning-trace blocks from responses. One of: deepseek-r1 \| qwen3 \| phi4 \| openthinker. v0.53.9 #98. |
| `--steer` | text |  |  | Aplica un vector de dirección de activación guardado durante la decodificación (CAA / ITI / RepE). Pasa el nombre registrado mediante `soup steer train`; el backend transformers instala el hook de decodificación después de cargar el modelo. |
| `--steer-strength` | float | `1.0` |  | Steering strength multiplier (\|s\| &lt;= 10.0). Ignored when --steer is unset. |
| `--hub` | text | `hf` |  | Hub de origen del modelo base: hf (por defecto) / modelscope / modelers. Los hubs que no son HF requieren el SDK correspondiente (v0.53.10 #152). |
| `--bank` | text |  |  | Ruta a un banco de vectores VeRA / VB-LoRA (JSON). Servicio LoRA multiusuario a MB por usuario: el delta por token es v_u ⊙ Px, enrutado por el encabezado de solicitud X-User-Id. Requiere --backend transformers. (v0.71.12 #221) |
| `--bank-strength` | float | `1.0` |  | Multiplicador de intensidad del delta del banco de vectores. Se ignora si no se define --bank. |
| `--mole` | text |  |  | Ruta al directorio de salida de entrenamiento MoLE (mole_gate.pt + mole_manifest.json). Sirve el modelo base + N LoRA de tareas congelados, con mezcla de compuertas por token durante la decodificación. Requiere --backend transformers; no se combina con --bank / --steer / --adapters. (v0.71.17 #259) |
| `--kv-cache-type` | text |  |  | Tipo de caché KV para decodificación: q8_0 (cuantizado a 8 bits, requiere hqq) / bf16 / f16 (tipo de caché) / fp8 (solo vLLM+Hopper). Solo backend transformers; el enrutamiento vLLM / SGLang está en la parte bloqueada. v0.71.14 (#140). |
| `--tool-auth-token` | text |  |  | Requiere 'Authorization: Bearer ***' en los endpoints de herramientas de ejecución de código (/v1/tools/python, /v1/tools/web_search). Muy recomendado cuando --host no es loopback, ya que esos endpoints ejecutan Python proporcionado por quien llama en un sandbox de mejores esfuerzos. |

## `soup ship`

Veredicto SHIP / DON'T SHIP después del ajuste fino: éxito en la tarea Y sin olvido catastrófico, combinado en una sola decisión (v0.71.25).

Uso: `soup ship [OPTIONS] [<command>] [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base` | text |  |  | Id/ruta del modelo base (el "antes"). |
| `--tuned` | text |  |  | Id/ruta del modelo ajustado (un modelo "después" independiente). |
| `--adapter` | text |  |  | Ruta del adaptador LoRA (tuned = base + este adaptador). |
| `--task-eval` | text |  |  | JSONL de tareas de evaluación para el éxito en tarea de etapa 1. |
| `--task-mode` | text | `metric` |  | Leg-1 mode: metric \| judge_score \| pairwise (judge win-rate). |
| `--noise-floor` | integer |  |  | Vuelve a ejecutar el modelo BASE N veces (2-10) para medir la resolución de este instrumento, muestra el resultado junto al veredicto y no considera significativo ningún delta inferior al umbral medido. Cuesta N pasadas base adicionales. El umbral de tarea de etapa 1 se mide en cada --task-mode; en judge_score / pairwise cada repetición requiere N pasadas adicionales del EVALUADOR (N x las llamadas a la API del evaluador), y el umbral se etiqueta como decode + judge. |
| `--judge-model` | text |  |  | URL del modelo evaluador para --task-mode judge_score o pairwise. |
| `--general-suite` | text |  |  | Lista separada por comas de benchmarks de etapa 2 (por defecto: suite sin conexión incluida: MCQ/arithmetic + tool-call/JSON/safety; los nombres no incluidos se envían a lm-eval). |
| `--baseline` | text |  |  | registry://&lt;id> o JSON con sello de puntuaciones de etapa 2 del modelo base (omite la ejecución base). Las referencias sin sello o cuyo scorer_revision no coincide generan una advertencia una vez; vuelve a medir para que ambos lados compartan la misma escala de puntuación (#404). |
| `--forgetting-threshold` | float | `0.05` |  | Caída máxima permitida de etapa 2 en puntos absolutos antes de DON'T SHIP. |
| `--evidence` | text |  |  | JSON de puntuaciones precalculadas (sin conexión, no carga modelos). |
| `--output`, `-o` | text |  |  | Escribe el JSON del veredicto en esta ruta. |
| `--emit-evidence` | text |  |  | Escribe las puntuaciones con el esquema de entrada --evidence para que esta ejecución pueda reproducirse sin conexión (la salida sirve de entrada). Con --config también SELLA la procedencia de la configuración en la evidencia generada. |
| `--config` | text |  |  | soup.yaml cuyo bloque eval.ship proporciona valores por defecto (las banderas CLI tienen prioridad). Con solo --evidence, ACTÚA COMO COMPUERTA (rechaza evidencia cuyo config_sha se haya desviado de esta configuración o cuya familia numerics no coincida); con --emit-evidence, SELLA la procedencia de esta configuración. |
| `--push` | text |  |  | Publica el veredicto como comentario de GitHub PR (owner/repo#N). Autenticación mediante GITHUB_TOKEN / GH_TOKEN; requiere la CLI `gh`. |
| `--device` | text |  |  | Dispositivo para la ejecución en vivo (cuda / cpu). |

## `soup shrink`

Reduce la profundidad del modelo (elimina el bloque contiguo de capas menos importantes según la distancia angular residual) + destilación reparadora opcional (v0.71.29).

Uso: `soup shrink [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model` | text |  | yes | Id o ruta local del modelo que se reducirá. |
| `--drop-ratio` | float |  |  | Fracción de capas que se eliminarán (0-1); p. ej., 0.25. |
| `--drop-layers` | integer |  |  | Número explícito de capas contiguas que se eliminarán. |
| `--calib` | text |  | yes | JSONL de calibración (prompts): debe permanecer dentro del cwd. |
| `--heal` | text |  |  | JSONL de reparación (filas de chat): destila el modelo original en el modelo podado. |
| `--heal-steps` | integer | `200` |  | Pasos aproximados del optimizador para la reparación por destilación. |
| `--tolerance` | float | `0.1` |  | Tolerancia de regresión de perplexity para el veredicto. |
| `--output-dir`, `-o` | text | `./shrunk` |  | Directorio del modelo reducido + informe. |
| `--device` | text |  |  | Dispositivo para los pases de importancia/perplexity (cuda / cpu). |
| `--trust-remote-code` | flag | off |  | Permite código de modelado personalizado (auto_map). |
| `--attach-to-registry` | text |  |  | Adjunta el informe de reducción a un id de entrada del registro. |
| `--plan-only` | flag | off |  | Imprime la tabla de importancia + el bloque elegido y termina. |
## `soup spectrum`

Analiza la SNR del espectro para entrenamiento dirigido (v0.71.23).

Uso: `soup spectrum <command> [ARGS]...`

Subcomandos:

- [`scan`](#soup-spectrum-scan)

### `soup spectrum scan`

Analiza la SNR de las capas de un modelo y emite un parche ``training.unfrozen_parameters``.

Uso: `soup spectrum scan [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Id de HF Hub o directorio local del modelo. |
| `--top-percent` | float | `50.0` |  | Conserva el N% superior de capas (según SNR) dentro de cada tipo de módulo. |
| `--modules` | text | `all` |  | Lista de tipos de módulo separados por comas que se analizarán: p. ej., 'mlp,attn' o 'all'. |
| `--output`, `-o` | text |  |  | Escribe el parche YAML en esta ruta (debe permanecer dentro del cwd). |
| `--no-cache` | flag | off |  | Skip the \~/.soup/spectrum scan cache. |

## `soup steer`

Dirección de activación (CAA / ITI / RepE): intervención durante inferencia sin volver a entrenar (v0.62.0).

Uso: `soup steer <command> [ARGS]...`

Subcomandos:

- [`apply`](#soup-steer-apply)
- [`list`](#soup-steer-list)
- [`train`](#soup-steer-train)

### `soup steer apply`

Resuelve y carga un vector de dirección guardado e imprime sus metadatos.

Uso: `soup steer apply [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--name`, `-n` | text |  | yes | Identificador de un vector de dirección guardado. |
| `--strength`, `-s` | float | `1.0` |  | Steering strength multiplier (\|s\| &lt;= 10.0). |

### `soup steer list`

Enumera los vectores de dirección almacenados localmente (tipo de artefacto del registro ``steering_vector``).

Uso: `soup steer list`

No acepta opciones ni argumentos.

### `soup steer train`

Entrena un vector de dirección a partir de pares de prompts contrastivos.

Uso: `soup steer train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base`, `-b` | text |  | yes | Id HF o ruta local del modelo base. |
| `--method`, `-m` | text | `caa` |  | Método de dirección: caa / iti / repe. |
| `--name`, `-n` | text |  | yes | Identificador del vector entrenado (p. ej., 'safety-v1'). |
| `--pairs`, `-p` | text |  | yes | Ruta al JSONL de pares de prompts contrastivos (positivo, negativo). |
| `--layer`, `-l` | integer |  |  | Índice de capa del decodificador de la que se extraerá el vector del flujo residual (por defecto: capa intermedia). |
| `--output`, `-o` | text |  |  | Directorio donde escribir el vector de dirección (por defecto: ./steering/&lt;name>). Debe estar dentro del cwd. |
| `--device` | text |  |  | Dispositivo de torch (cpu / cuda). Por defecto, CUDA cuando está disponible. |
| `--top-k`, `-k` | integer | `8` |  | Solo ITI: número de cabezas de atención en las que se intervendrá (1-256). |
| `--plan-only` | flag | off |  | Valida las entradas e imprime el plan resuelto; omite el ajuste activo. |
| `--registry-id` | text |  |  | Id de entrada del registro a la que adjuntar el archivo del vector entrenado como artefacto steering_vector. Termina con código 1 si falla la adjunción. |

## `soup sweep`

Ejecuta una búsqueda de hiperparámetros: búsqueda de cuadrícula o aleatoria sobre parámetros de entrenamiento.

Uso: `soup sweep [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta al archivo de configuración soup.yaml base |
| `--param`, `-p` | text, repeatable |  |  | Parámetro que se explorará: key=val1,val2,val3 (p. ej., lr=1e-5,2e-5,5e-5) |
| `--sweep-config` | text |  |  | Ruta a un YAML de búsqueda independiente (strategy/n_runs/seed/params) en lugar de --param; strategy, n_runs y seed tienen prioridad sobre --strategy/--max-runs |
| `--strategy`, `-s` | text | `grid` |  | Estrategia de búsqueda: grid, random |
| `--max-runs` | integer |  |  | Número máximo de ejecuciones (útil para la estrategia random) |
| `--name`, `-n` | text |  |  | Prefijo del nombre del experimento de búsqueda |
| `--dry-run` | flag | off |  | Muestra las ejecuciones planificadas sin ejecutarlas |
| `--early-stop` | float |  |  | Se detiene antes si el loss de una ejecución supera el mejor loss por este factor (p. ej., 1.5 = 50% peor) |
| `--yes`, `-y` | flag | off |  | Omite el mensaje de confirmación |

## `soup tokenizer`

Herramientas de tokenizador: entrena un tokenizador BPE a partir de JSONL (v0.53.9).

Uso: `soup tokenizer <command> [ARGS]...`

Subcomandos:

- [`train`](#soup-tokenizer-train)

### `soup tokenizer train`

Entrena un tokenizador BPE a partir de un corpus local.

Uso: `soup tokenizer train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Ruta al corpus JSONL o .txt (debe permanecer dentro del cwd). |
| `--vocab-size`, `-v` | integer | `32000` |  | Tamaño objetivo del vocabulario (256-200000). |
| `--output`, `-o` | text | `tokenizer_out` |  | Directorio de salida para tokenizer.json + vocab.json (dentro del cwd). |
| `--min-frequency` | integer | `2` |  | Frecuencia mínima de pares (>=1). |
| `--special-token` | text, repeatable |  |  | Agrega un token especial (repetible). Por defecto: &lt;pad>, &lt;s>, &lt;/s>, &lt;unk>. |

## `soup train`

Inicia el entrenamiento desde una configuración soup.yaml.

Uso: `soup train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Ruta al archivo de configuración soup.yaml |
| `--name`, `-n` | text |  |  | Nombre del experimento (se genera automáticamente si no se define) |
| `--dry-run` | flag | off |  | Valida la configuración y los datos sin entrenar |
| `--resume`, `-r` | text |  |  | Reanuda desde un checkpoint: ruta al directorio de checkpoint ('auto' para el más reciente); en el backend MLX, en su lugar una ruta a un archivo de adaptador .safetensors. No compatible si se configura training.relora_steps |
| `--wandb` | flag | off |  | Habilita el registro de Weights &amp; Biases |
| `--tensorboard` | flag | off |  | Habilita el registro de TensorBoard (registra en output_dir/runs/) |
| `--tracker` | text |  |  | Rastreador de experimentos: mlflow / swanlab / trackio (v0.43.0). Mutuamente excluyente con --wandb / --tensorboard. |
| `--deepspeed` | text |  |  | Habilita DeepSpeed: zero2, zero3, zero2_offload, zero3_offload (etapa 3 + descarga de parámetros a CPU), zero++ (ZeRO++) o ruta al JSON de configuración |
| `--fsdp` | text |  |  | Habilita FSDP2: full_shard, shard_grad o full_offload |
| `--gpus` | text |  |  | GPU por nodo para entrenamiento distribuido ('auto' o un entero) |
| `--nodes` | integer | `1` |  | Número de máquinas; cada una debe usar el mismo conteo de --gpus |
| `--node-rank` | integer |  |  | Rango de esta máquina, de 0 a nodes menos 1 (por defecto 0) |
| `--master-addr` | text |  |  | Nombre de host o IP del rango 0, accesible desde todos los nodos; obligatorio con --nodes > 1 |
| `--master-port` | integer |  |  | Puerto de coordinador compartido entre varios nodos (por defecto 29500) |
| `--no-reexec` | flag | off |  | Imprime el comando de inicio distribuido en lugar de ejecutarlo |
| `--gate` | text |  |  | Habilita el entrenamiento con compuerta de evaluación usando un archivo de suite (atajo para training.eval_gate.enabled=true + suite=&lt;path>) |
| `--push-as` | text |  |  | Publica automáticamente cada checkpoint save_steps en HF Hub como rama 'checkpoint-&lt;step>' del repo indicado (p. ej., user/my-model) |
| `--hf-resume` | flag | off |  | Descarga la rama de checkpoint más reciente del repo --push-as y reanuda desde ella. Requiere --push-as; no es compatible si se configura training.relora_steps. |
| `--find-lr` | flag | off |  | Buscador de rango de LR (v0.32.0): ejecuta una búsqueda geométrica breve de LR, escribe un informe JSON con el LR recomendado y termina sin entrenar. |
| `--find-lr-start` | float | `1e-07` |  | Buscador de rango de LR: LR inicial (por defecto 1e-7) |
| `--find-lr-end` | float | `0.1` |  | Buscador de rango de LR: LR final (por defecto 1e-1) |
| `--find-lr-steps` | integer | `100` |  | Buscador de rango de LR: número de pasos de búsqueda (por defecto 100) |
| `--find-lr-output` | text | `lr_finder.json` |  | Buscador de rango de LR: ruta del informe JSON (por defecto ./lr_finder.json) |
| `--yes`, `-y` | flag | off |  | Omite el mensaje de confirmación |
| `--trust-remote-code` | flag | off |  | Permite cargar modelos que incluyen Python personalizado mediante auto_map. Se deniega por defecto (v0.36.0). Actívalo solo si confías en el origen. |
| `--echo-trap-tokenizer-aware` | flag | off |  | Usa n-gramas de id de tokenizador para puntuar la trampa de eco. Requiere training.echo_trap_enabled=true en grpo/ppo. |
| `--reward-hack-detector` | text |  |  | Reward-hacking detector for GRPO/PPO: info_rm \| rm_ensemble. Overrides training.reward_hack_detector. (v0.71.26) |
| `--reward-hack-halt` | flag | off |  | Detiene automáticamente el entrenamiento ante un veredicto HACK. Requiere --reward-hack-detector (o training.reward_hack_detector). (v0.71.26) |
| `--replay` | text |  |  | Dataset anterior que se intercalará como repaso de aprendizaje continuo, para que entrenar la tarea nueva no borre la anterior. Solo sft/pretrain; incompatible con packing/multipack. Anula data.replay. (v0.71.36) |
| `--replay-ratio` | float |  |  | Fracción del conjunto de entrenamiento mezclado FINAL que corresponde a filas replay (por defecto 0.1). Anula data.replay_ratio. (v0.71.36) |
| `--replay-seed` | integer |  |  | Semilla para la muestra e intercalado replay. Anula data.replay_seed. (v0.71.36) |
| `--reward-hack-mitigation` | text |  |  | Closed-loop reward-hacking mitigation mode: off \| log_only \| kl_control \| pid_lagrangian. Requires training.reward_hack_detector on grpo/ppo. Overrides training.reward_hack_mitigation. (v0.71.26) |
| `--minillm-on-policy` | flag | off |  | Usa el rollout MiniLLM teacher-mixed VERDADERAMENTE on-policy (v0.71.18 #257) en lugar de la mezcla de distribuciones sin conexión. Requiere training.minillm_enabled=true con task='distill'. |
| `--profile` | flag | off |  | Registra una traza de torch.profiler (JSON de Chrome trace) durante los primeros pasos de entrenamiento. Salida: &lt;output>/profiles/&lt;run_id>.trace.json |
| `--allow-oom-attempt` | flag | off |  | Omite la compuerta analítica de ajuste de hardware VRAM e inicia incluso si se prevé que la ejecución agotará la memoria de GPU (exclusión voluntaria). |
| `--diagnose-gate` | text |  |  | Después del entrenamiento, ejecuta `soup diagnose` con el JSON de evidencia proporcionado (o evidencia temporal). Se niega a marcar la ejecución como exitosa si algún modo de fallo devuelve MAJOR (código 2) o NOT_RUN (código 3). |
| `--annex-xi` | text |  |  | Después del entrenamiento, genera un documento automático del Anexo XI/XII de la Ley de IA de la UE en la ruta indicada (dentro del cwd). Por ahora, cuerpo Markdown; PDF en v0.59.1. |
| `--repro-receipt` | text |  |  | Después del entrenamiento, escribe en la ruta indicada un recibo de reproducibilidad al estilo SR 11-7 (semillas + versiones de kernel + GPU + SO). v0.59.0. |
| `--capture-activations` | text |  |  | Después del entrenamiento, captura activaciones del flujo residual en la capa de decodificador indicada (p. ej., model.layers.5) con --capture-prompts y las escribe en &lt;output>/activations/activations.json para soup probe sae-diff / sleeper. v0.71.8 #219. |
| `--capture-prompts` | text |  |  | JSONL (o .txt) de prompts para ejecutar con --capture-activations (un prompt por línea; campo 'prompt'/'text' o texto sin procesar). |
| `--track-energy` | flag | off |  | Mide la energía + CO2 de la ventana de entrenamiento mediante codecarbon (sin conexión; requiere `pip install soup-cli[carbon]`). Envía los kWh / CO2 a --annex-xi. v0.71.3. |
| `--energy-country` | text | `USA` |  | Código de país ISO 3166-1 alpha-3 para la estimación de intensidad de red de CO2 usada por --track-energy (por defecto USA). |
| `--energy-out` | text |  |  | Escribe la medición de --track-energy en este archivo JSON (dentro del cwd) para que `soup bom emit --energy <path>` pueda usarla. v0.71.15. |
| `--cloud` | text |  |  | Entrena en una GPU en la nube en lugar de localmente (v0.71.18 #16). Compatible: modal, lambda (runpod está planificado). Genera un stub de app en la nube a partir de la configuración (solo planificación); usa --cloud-submit para enviarla en vivo. |
| `--gpu` | text | `a100` |  | Tipo de GPU en la nube para --cloud (t4 / l4 / a10 / a10g / a100 / a100-80gb / l40s / h100 / a6000). Por defecto a100. Se aplican listas permitidas específicas del proveedor. |
| `--cloud-submit` | flag | off |  | Con --cloud, envía la ejecución generada en vivo mediante el SDK o API de la nube (requiere el token/clave API del proveedor correspondiente). Por defecto, solo planificación (genera e imprime el comando). Lambda requiere una clave SSH registrada. |

## `soup tui`

Dashboard de terminal a pantalla completa: enumera ejecuciones, inspecciona métricas y sigue logs.

Uso: `soup tui [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--refresh` | float | `1.0` |  | Intervalo de actualización en segundos (0.25 - 10.0). |
| `--limit` | integer | `50` |  | Número máximo de ejecuciones que se cargarán en el dashboard. |

## `soup tunability`

Sondea y entrena 6-10 modelos base pequeños en una parte reservada + informa la frontera de Pareto (delta de evaluación, costo de entrenamiento, licencia) (v0.64.0).

Uso: `soup tunability [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--dataset`, `-d` | text |  |  | Ruta a la parte reservada JSONL usada para el sondeo. |
| `--candidates` | text |  |  | Subconjunto de candidatos por defecto, separado por comas y por nombre. |
| `--probe-steps` | integer | `100` |  | Pasos por sondeo de candidato (10-10000). |
| `--holdout-size` | integer | `64` |  | Número de filas reservadas para puntuar el delta (10-100000). |
| `--output`, `-o` | text |  |  | Ruta opcional para el informe JSON. |
| `--plan-only` | flag | off |  | Imprime los candidatos planificados + estimación de costo y termina sin sondear. |
| `--list` | flag | off |  | Enumera el catálogo integrado de candidatos y termina. |
| `--live` | flag | off |  | Ejecuta un sondeo LoRA EN VIVO por candidato (carga cada repo + entrena --probe-steps en una parte reservada pequeña). Si no se usa, se aplica una heurística determinista sin conexión. |
| `--device` | text |  |  | Dispositivo para el sondeo en vivo (cuda / cpu). Se detecta automáticamente si se omite. |

## `soup ui`

Inicia la interfaz web de Soup.

Uso: `soup ui [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--port`, `-p` | integer | `7860` |  | Puerto en el que se servirá |
| `--host` | text | `127.0.0.1` |  | Host al que enlazarse |
| `--no-browser` | flag | off |  | No abre el navegador automáticamente |
| `--show-token` | flag | off |  | Imprime el token de autenticación y termina (para scripts) |
| `--public` | flag | off |  | Se enlaza a 0.0.0.0 para que la interfaz sea accesible desde un teléfono en la misma LAN. Al inicio, imprime una URL escaneable desde el teléfono + código QR ASCII. El HTTP loopback se actualiza automáticamente para requerir un token Bearer en la cadena de consulta. v0.53.9 #95. |
| `--auth-token` | text |  |  | Anula el token Bearer generado automáticamente (16-128 caracteres base64 seguros para URL). Útil para marcadores estables en el teléfono. Si se omite, se genera un token nuevo en cada inicio. |

## `soup version`

Muestra la versión de Soup CLI.

Uso: `soup version [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--full`, `-f` | flag | off |  | Muestra información del sistema y extras |
| `--json` | flag | off |  | Emite en formato JSON |

## `soup why`

Explica en lenguaje sencillo las anomalías de entrenamiento de una ejecución.

Uso: `soup why [RUN_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  |  | Id de ejecución (o prefijo). Por defecto, la ejecución más reciente. |
