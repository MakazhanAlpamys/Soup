# Contribuir a Soup

> 🇪🇸 Traducción al español de [`CONTRIBUTING.md`](CONTRIBUTING.md). Ante cualquier duda, manda el original en inglés.

¡Gracias por tu interés en contribuir a Soup! Agradecemos los informes de errores, las solicitudes de funciones y los pull requests de la comunidad.

## Primeros pasos

### 1. Fork y clonar

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
```

### 2. Configurar el entorno de desarrollo

**Requisitos:** Python 3.10, 3.11 o 3.12 (`requires-python = ">=3.10,<3.13"`). CI ejecuta
exactamente esas tres; `tests/test_requires_python_bound.py` deriva el límite declarado de
la matriz de CI, así que ampliar una sin la otra hace fallar la suite.

Trabaja dentro de un virtualenv y luego instala el proyecto en modo editable con las
dependencias de desarrollo:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

El paso de `venv` no es opcional en Debian 12, Ubuntu 23.04 o posterior: sin él
`pip` se niega con `error: externally-managed-environment`, porque esas distros
impiden que pip escriba en el Python del sistema que administra `apt`
([PEP 668](https://peps.python.org/pep-0668/)). `pipx` es la respuesta para
*instalar* Soup y la herramienta equivocada aquí, ya que un checkout editable necesita ser
importable por la suite de pruebas en lugar de quedar aislado de ella.

Esto instala:
- `pytest` para pruebas
- `ruff` para linting + formateo
- `pytest-cov` para cobertura
- `httpx` para pruebas HTTP
- `mypy` para verificación de tipos
- `pre-commit` para los hooks de git de lint/formato
- el stack completo de entrenamiento — `[dev]` se autorreferencia a `[train]`, así que `torch`,
  `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate` también se
  incorporan (desde v0.71.0 son un extra opcional, no dependencias centrales)
- `cryptography` — `[dev]` también lo incorpora (es el extra `[sign]`) para que las
  pruebas de firma ed25519 (`soup adapters sign` / `soup attest`) se ejecuten en CI
- `reportlab` — `[dev]` también lo incorpora (es el extra `[pdf]`) para que las
  pruebas de PDF del Anexo XI/XII (`soup train --annex-xi *.pdf`) se ejecuten en CI

### 3. Verificar la configuración

Ejecuta la suite de pruebas para confirmar que todo funciona:

```bash
pytest tests/ -v --tb=short
```

Ejecuta el linter:

```bash
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

Si ambos pasan, ¡estás listo para contribuir!

## Estilo de código

Usamos **ruff** para todo el estilo de código y el linting. Antes de hacer commit, ejecuta:

```bash
# Buscar problemas
ruff check src/soup_cli/ scripts/ tests/ benchmarks/

# Corregir problemas automáticamente
ruff check --fix src/soup_cli/ scripts/ tests/
```

### Guías de estilo

- **Longitud de línea:** 100 caracteres (aplicada por ruff)
- **Imports:** ordenados y organizados (regla I de ruff)
- **Nomenclatura:** nada de nombres de variable de una sola letra (ruff E741) — usa `entry`, `part`, `length` en lugar de `l`, `p`, etc.
- **Imports diferidos:** las dependencias pesadas (torch, transformers, peft, trl, etc.) deben importarse dentro de las funciones, no a nivel de módulo, para mantener la CLI ágil
- **Nada de encabezados de licencia de terceros.** Este proyecto es Apache-2.0 y se publica en PyPI, así que un `SPDX-License-Identifier` o una línea de copyright que nombre a cualquier otro es una declaración de licencia que no podemos hacer. Si generas archivos desde una plantilla —incluso con herramientas automatizadas, que es como ha ocurrido tres veces— elimina su encabezado antes de hacer push. `tests/test_no_foreign_license_headers.py` hace fallar la compilación por esto en lugar de depender de que tú o yo lo detectemos en un diff.
- **Validación de configuración:** usa siempre Pydantic v2 con `BaseModel` y `Field`
- **Salida:** usa `rich.console.Console` para toda la salida — nunca `print()` sin más
- **Anotaciones de tipo:** incluye siempre anotaciones de tipo para los parámetros y valores de retorno de las funciones

Ejemplo:

```python
# INCORRECTO
from torch import cuda
import transformers

def train():
    print("Starting training")
    model = transformers.AutoModel.from_pretrained("llama-7b")

# CORRECTO
def train():
    from torch import cuda
    import transformers

    console = Console()
    console.print("Starting training")
    model = transformers.AutoModel.from_pretrained("llama-7b")
```

## Estructura del proyecto

Directorios clave:

```
src/soup_cli/
  cli.py              - Main entry point, command routing
  commands/           - Command implementations (train, chat, eval, deploy, etc.)
  config/             - Config schema (schema.py) and loader (loader.py)
  data/               - Data loading, format conversion, providers, templates
  trainer/            - Training wrappers (SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, Pretrain, Reward Model, Embedding, BCO, Preference, Distill, Classifier, PRM, Unlearn, MoLE Routing)
  monitoring/         - Callbacks and live dashboard
  experiment/         - SQLite experiment tracking
  eval/               - Eval platform (custom tasks, LLM judge, human eval, leaderboard)
  migrate/            - Config migration (LLaMA-Factory, Axolotl, Unsloth)
  recipes/            - Ready-made configs for popular models (170 recipes)
  autopilot/          - Zero-config decision engine (v0.25.0)
  registry/           - Model Registry (hashing, store, diff, attach) (v0.26.0 + v0.33.0)
  cans/               - Shareable .can artifact format + run/publish orchestrator (v0.26.0 + v0.33.0)
  data/traces/        - Trace-to-Preference harvester (v0.26.0)
  data/collators.py   - CrossDocCollator for sample packing (v0.33.0)
  utils/              - GPU, errors, MoE, GaLore, QAT, Unsloth, vLLM, SGLang, Liger, FlashAttn, FSDP, Ring Attention, long-context, quality, curriculum, freeze, dataset-registry, mlx, peft_builder, paths, topology, launcher, mii, pipeline, cut_ce, fp8, gradient_ckpt, kernel_picker, cross_doc_attn, activation_offload, hf, spec_pairing, structured_output, metrics, tracing, lr_finder, grad_accum, mixed_precision, warmup, spike_recovery, convergence, v028_features, multipack_sampler, multipack, neat_packing, jinja_analyzer, quant_menu, relora, peft_patches, peft_wiring, dpo_variants, optimizer_zoo, lr_groups, loftq_init, block_expansion, tts, classifier, distill, bitnet, ebft_gdpo, moe_quant, reasoning_effort, gguf_quant, kv_cache, advanced_precision, save_formats, deploy_measure, advise, advise_history, adapter_diff, adapter_merge, blame, adapter_branch, unlearning, unlearning_eval, knowledge_edit, edit_governor, edit_diff, ra_dit, steering, citation_faithful, grace_codebook, ingest_sources, prune_prompt, active_sampler, ab_test, drift_alarm, tunability, terraform_plan, env_lock, hardware_fit, completions, license_advisor, behavior_battery, capability_suite, checklist_dsl, irt, sae_diff, sleeper_probe, interference, probe_pack, cmaes_merge, vector_bank, mole_routing, adapter_pr, soup_lock, adapter_bisect, prompt_compile, prompt_distill, compile_tools, apple_adapter, local_rl, build_dag, expectations, magpie, persona_hub, brain_rot, reward_hacking, uld, minillm, rl_checkpoint, iterative_dpo, echo_trap, mod, local_rl_scheduler, spectrum_scan, ship_verdict, reward_hack_control, data_doctor, data_lint, shrink, best_of_n, evolve, reward_synth, reward_stress, layer_stream, layer_shard, layer_stream_runtime
  templates/          - 21 built-in soup.yaml templates (YAML + manifest.json) with load_template loader (v0.39.0, +bco v0.40.0, +4 compliance v0.71.35)
  ui/                 - Web UI (FastAPI + HTML/JS SPA)

tests/                - Test suite covering all CLI commands, trainers, and model architectures
examples/             - Real-world config examples and datasets
```

## Ejecutar las pruebas

### Todas las pruebas

```bash
pytest tests/ -v --tb=short
```

### Un solo archivo de prueba

```bash
pytest tests/test_config.py -v
```

### Una sola prueba

```bash
pytest tests/test_data.py::test_detect_alpaca_format -v
```

### Con cobertura

```bash
pytest tests/ --cov=soup_cli --cov-report=html
```

### Pruebas de GPU

Todos los ejecutores de CI no tienen GPU, así que las pruebas que necesitan un dispositivo CUDA se omiten allí y
solo significan algo en una tarjeta real. Llevan el marcador `gpu` (#833):

```bash
pytest tests/ -m gpu --no-cov -v
```

`--no-cov` importa: la puerta `--cov-fail-under` por defecto se calcula sobre todo el
paquete y hace fallar una ejecución de un subconjunto. Para condicionar una prueba nueva a CUDA, decórala con
`@pytest.mark.gpu` (o `@pytest.mark.gpu(reason="...")`); no escribas tu propia
comprobación `torch.cuda.is_available()`. `tests/conftest.py` contiene la única sonda, y
`tests/test_issue833_gpu_marker.py` falla ante una privada. Usa
`tests.conftest.cuda_available()` solo para elegir un dispositivo, nunca para omitir.

Si tienes una tarjeta, una ejecución es una contribución real: las dos ejecuciones registradas hasta ahora encontraron
defectos que CI no pudo ver. Abre un issue (o comenta en el #833) con:

- la tarjeta, su capacidad de cómputo (`torch.cuda.get_device_capability()`) y la VRAM
- la versión del driver y la compilación de torch (`torch.__version__`, `torch.version.cuda`)
- la versión de `bitsandbytes`, si está instalada
- el commit que ejecutaste (`git rev-parse HEAD`)
- la línea de resumen (`N passed, N failed, N skipped`), y el nombre y la primera línea
  de error de cada fallo

Las ejecuciones aceptadas se registran en [`benchmarks/gpu-test-runs.md`](benchmarks/gpu-test-runs.md).

### Archivos de prueba

> Una muestra representativa de la suite a continuación. Ejecuta `pytest tests/ -v` para la lista completa.

Nombra un archivo de prueba nuevo según el módulo o la función que cubre (`test_<module>.py`), no según
un número de issue o una versión. Los archivos de prueba existentes conservan sus nombres.

| File | Cubre |
|------|--------|
| test_config.py | Carga, validación y valores por defecto de la configuración |
| test_data.py | Detección de formato, conversión, validación |
| test_gpu.py | Detección de GPU, estimación del tamaño de batch |
| test_cli.py | Comandos de la CLI, version --full |
| test_tracker.py | Seguimiento de experimentos en SQLite |
| test_runs.py | Comandos de la CLI `soup runs` |
| test_data_tools.py | Comandos de datos convert/merge/dedup/stats |
| test_eval.py | Comando eval |
| test_smoke_train.py | Pruebas de humo del pipeline completo (GPU) |
| test_chat.py | Comando chat, `_detect_base_model` |
| test_push.py | Comando push, `_format_size`, `_generate_model_card` |
| test_init.py | Comando init, plantillas, lógica de sobrescritura |
| test_callback.py | `SoupTrainerCallback` (basado en mocks) |
| test_display.py | Renderizado de `TrainingDisplay` |
| test_loader.py | Carga de datos (JSONL/JSON/CSV, casos límite) |
| test_validator.py | `validate_and_stats`, `extended_stats`, `_percentile` |
| test_formats.py | Conversión inversa, ida y vuelta, casos límite |
| test_merge.py | Comando merge, detección de adaptadores, validación |
| test_export.py | Comando export, tipos de cuantización GGUF, validación |
| test_resume.py | Resolución de checkpoint al reanudar, bandera de W&B |
| test_serve.py | Comando serve, app FastAPI, endpoints, streaming |
| test_generate.py | Data generate, parseo de JSON, validación, prompts |
| test_sweep.py | Parseo de parámetros de sweep, combinaciones, configuración anidada |
| test_diff.py | Recolección de prompts de diff, métricas, CLI |
| test_deepspeed.py | Configuraciones de DeepSpeed, detección multi-GPU, integración con el trainer |
| test_errors.py | Mensajes de error amigables, bandera --verbose, mapeo de errores |
| test_doctor.py | Comando `soup doctor`, verificación de versión, recursos del sistema, tabla de dependencias |
| test_quickstart.py | Demo de `soup quickstart`, creación de datos/config, --dry-run |
| test_grpo.py | Config GRPO, recompensas, preparación de datos, plantilla, atajos de sweep |
| test_progress.py | Barra de progreso de descarga de Rich, `_enable_hf_transfer_progress` |
| test_unsloth.py | Config del backend Unsloth, detección, integración con el trainer, plantillas |
| test_vision.py | Config de la modalidad de visión, formatos LLaVA/ShareGPT4V, loader, trainer, plantillas |
| test_qat.py | Config de QAT, validación, integración con el trainer, compatibilidad de exportación |
| test_ui.py | Comando de la UI web, endpoints FastAPI, autenticación, archivos estáticos, validación de config |
| test_vllm_serve.py | Detección del backend vLLM, creación del engine, bandera serve --backend, app FastAPI |
| test_ppo.py | Config PPO, config del modelo de recompensa, preparación de datos, plantilla RLHF, enrutamiento, sweep |
| test_kto.py | Config KTO, formato de datos, plantilla, enrutamiento, sweep, guard de train, asistente |
| test_orpo.py | Config ORPO, plantilla, enrutamiento, sweep, guard de train, asistente |
| test_simpo.py | Config SimPO, plantilla, enrutamiento, sweep, guard de train |
| test_ipo.py | Config IPO, plantilla, enrutamiento, sweep, guard de train |
| test_advanced_peft.py | Config de DoRA, LoRA+, GaLore, validación, atajos de sweep |
| test_infer.py | Comando de inferencia por lotes, lectura de prompts, validación de la CLI |
| test_tensorboard.py | Bandera de TensorBoard, conflicto con wandb, enrutamiento de report_to |
| test_pretrain.py | Tarea pretrain, formato de texto plano, config MoE, plantillas, enrutamiento |
| test_moe.py | Detección de MoE, objetivos LoRA de ScatterMoE, extracción de información de MoE |
| test_bugfixes.py | Correcciones de regresión de v0.10.1-v0.14.3 |
| test_cli_subprocess.py | Pruebas de la CLI por subproceso: punto de entrada, codificación, rutas, regresiones de plataforma |
| test_performance.py | Liger Kernel, FlashAttention, FSDP2, Ring Attention, contexto largo, escalado RoPE |
| test_embedding.py | Config de la tarea embedding, formato, plantilla, enrutamiento, sweep, pooling |
| test_onnx_tensorrt_export.py | Exportación ONNX, exportación TensorRT-LLM, compatibilidad con formatos |
| test_speculative_decoding.py | CLI de decodificación especulativa, modelo borrador, integración con vLLM |
| test_server_generate.py | Proveedor de servidor para data generate, validación de SSRF |
| test_quality_filter.py | Puntuación de perplejidad + coherencia, `soup data filter` |
| test_audio.py | Config de la modalidad de audio, formato, plantilla, enrutamiento, loader |
| test_sglang_serve.py | Detección del backend SGLang, creación del runtime, serve --backend |
| test_deploy_ollama.py | Deploy con Ollama, generación de Modelfile, mapeo de plantillas, validación de seguridad |
| test_eval_platform.py | Eval personalizado, judge, evaluación humana (Elo), leaderboard, compare, auto-eval, seguridad |
| test_synth_data_pro.py | Proveedores (Ollama, Anthropic, vLLM), plantillas, pipeline de calidad, SSRF |
| test_migrate.py | Migración de LLaMA-Factory/Axolotl/Unsloth, path traversal, validación de ida y vuelta |
| test_recipes.py | Catálogo de recetas, búsqueda, CLI (list/show/use), path traversal |
| test_neftune_rslora.py | Config/validación/sweep de NEFTune, config/validación/sweep de rsLoRA |
| test_profile.py | Profiler de entrenamiento: estimación de memoria, velocidad, recomendaciones de GPU, CLI |
| test_multi_adapter.py | Servicio multi-adaptador: validación, parseo, endpoints FastAPI, CLI |
| test_data_sample.py | Muestreo de datos: estrategias random/diverse/hard, CLI, casos límite |
| test_adapters.py | Gestión de adaptadores: list/info/compare, descubrimiento, metadatos |
| test_awq_gptq_export.py | Exportación AWQ/GPTQ: compatibilidad con formatos, CLI, mocks de quantize, calibración, seguridad |
| test_packing.py | Empaquetado de muestras: config, YAML, integración con el trainer, sweep |
| test_data_split.py | División de datos: divisiones por ratio/absoluta/estratificada, seed, casos límite |
| test_curriculum.py | Aprendizaje por currículo: config, ordenamiento por longitud, buckets, sweep |
| test_dataset_hub.py | Búsqueda de datasets de HF, vista previa, descarga, conversión de formato, seguridad |
| test_freeze_training.py | Entrenamiento con congelación: config, congelación de capas, nomenclatura de GPT-2, sweep |
| test_loss_watchdog.py | Watchdog de loss: config, comportamiento del callback, paciencia, sweep |
| test_dataset_registry.py | Registro de datasets: CRUD, CLI, validación de nombres, manejo de errores |
| test_tool_calling.py | Detección del formato de tool-calling, normalización, puntuación de eval, recetas (v0.25.0) |
| test_rlvr.py | Recompensas verificables de RLVR: math_verify, sandbox de code_exec, json_schema (v0.25.0) |
| test_peft_methods.py | LoraConfig de VeRA + OLoRA, peft_builder, integración con sweep (v0.25.0) |
| test_mlx_backend.py | Backend MLX de Apple Silicon: detección, trainers, enrutamiento (v0.25.0) |
| test_data_augment.py | Aumento de datos: estrategias rephrase/translate/style, CLI, seguridad (v0.25.0) |
| test_training_intelligence.py | Detección de olvido + inteligencia de checkpoints + SQLite (v0.25.0) |
| test_autopilot.py | Autopilot: analizadores, motor de decisión, CLI (v0.25.0) |
| test_registry.py | Model Registry: hashing, CRUD del store, artefactos, DAG de linaje, diff, CLI, historial (v0.26.0) |
| test_eval_gate.py | Entrenamiento con puerta de eval: config, carga de suites, línea base, callback, CLI (v0.26.0) |
| test_trace_to_pref.py | Trace-to-Preference: parsers de LangChain/OpenAI/Soup-serve, constructor de pares, CLI (v0.26.0) |
| test_quant_check.py | Quant-Lobotomy: classify_delta, resolve_model_ref, formatos de render, CLI (v0.26.0) |
| test_cans.py | Soup Cans: esquema de manifest, pack/unpack, tar traversal, seguridad de fork, CLI (v0.26.0) |
| test_multi_gpu.py | Dominio multi-GPU: topología, --gpus, lanzador de accelerate, ZeRO++, FSDP2+compile, pipeline (v0.27.0) |
| test_training_speed.py | Velocidad y memoria de entrenamiento: CCE, FP8, niveles de grad-ckpt, kernel picker, cross-doc attn, activation offload (v0.28.0) |
| test_hf_integration.py | Integración profunda con HF Hub: token/endpoint/repo_id, callback de auto-push, model card v2, colecciones, data push, HF Spaces, SSRF de IP privada (v0.29.0) |
| test_inference_advanced.py | Excelencia en inferencia: prefix caching, auto-pairing de decodificación especulativa, hot-swap de LoRA, salida estructurada, dashboard + /metrics, trazado OpenTelemetry (v0.30.0) |
| test_recipes_v031.py | Amplitud de modelos y recetas: 34 recetas nuevas (vision/audio/reasoning/edge/domain/multimodal); invariantes de todo el catálogo; validación del flujo de trabajo de CI (v0.31.0) |
| test_auto_tuning.py | Estabilidad de entrenamiento y auto-ajuste: buscador de rango de LR, monitor de grad-accum, precisión mixta automática, warmup automático, recuperación de picos, detector de convergencia, cableado de autopilot (v0.32.0) |
| test_part_f_hardening.py | Live Wire Parte F: aislamiento de sandbox a nivel de SO de RLVR + TOCTOU de prune_checkpoints (v0.33.0) |
| test_part_a_wave1.py | Live Wire Parte A: puntuación en vivo de la puerta de eval + attach al registry (v0.33.0) |
| test_part_a_wave2.py | Live Wire Parte A: soup can run / publish + esquema DeployTarget (v0.33.0) |
| test_part_e.py | Live Wire Parte E: bucle en vivo de --find-lr + aviso de recuperación de picos + push de precisión mixta automática + aviso de grad-accum (v0.33.0) |
| test_part_d.py | Primitivas históricas de la Parte D: LogitsProcessor de salida estructurada + humo de integración de push a HF (v0.33.0) |
| test_part_c.py | Live Wire Parte C: funciones multi-trainer de v0.28.0 + hooks selectivos de ckpt + CrossDocCollator (v0.33.0) |
| test_part_b.py | Live Wire Parte B: auto-reexec bajo accelerate launch + serve en vivo de DeepSpeed-MII (v0.33.0) |
| test_log_level.py | Niveles de logging inteligentes `--log-level quiet/normal/verbose/debug` (v0.34.0 Parte A) |
| test_run_cost.py | Costo por ejecución: búsqueda de tarifa de GPU + estimación + formato + integración con el tracker (v0.34.0 Parte B) |
| test_why.py | Explicador heurístico de `soup why`: NaN / plateau / divergencia / grad-norm / límites de LR (v0.34.0 Parte C) |
| test_crash_reporter.py | Bundle `.crash`: clasificación, censura de secretos, escritura bajo el cwd, truncado por tamaño excesivo (v0.34.0 Parte D) |
| test_replay.py | `soup runs replay`: resumen + downsample + renderizado en la CLI (v0.34.0 Parte E) |
| test_profiling.py | Auto-profiling: ProfilerSchedule + contención de rutas + degradación sin torch (v0.34.0 Parte F) |
| test_tui.py | `soup tui`: límites de la CLI + error de textual ausente + constructores de filas (v0.34.0 Parte G) |
| test_trainer_coverage_v035.py | Matriz de humo del cableado multi-trainer de v0.28.0: cada trainer × cada función de velocidad/memoria + benchmark_kernel_combos + elevación de la puerta de esquema (v0.35.0 Partes A / B / C / D — #60, #61, #45) |
| test_v0470_part_a.py | Synthetic Data Forge: dataclasses congeladas ForgePlan / ProvenanceRecord / ForgeRow + allowlist VALID_TASKS + chunk_document + score_uncertainty + discover_documents (contenido en el cwd, rechaza symlinks, allowlist de extensiones) + validadores de build_forge_plan (task / target_rows / teacher / umbral NaN+Inf) + synthesise_forge_rows (traga excepciones del judge en DEBUG) + write_forge_dataset + write_provenance (atómico, seguro ante TOCTOU) + humo de CLI (v0.47.0) |
| test_v0470_part_b.py | Foso de calidad de datos: MappingProxyType de BENCHMARKS + ScoreReport congelado + ngram_set / ngram_overlap_ratio / decontaminate_rows (ratio de contención) + detect_pii (regexes endurecidas contra ReDoS + precap de 50 KB) + detect_language (heurística de palabras vacías en 6 idiomas) + score_toxicity (línea base por palabras clave) + score_educational_value + compute_scorecard + load_jsonl_rows / write_jsonl_rows (contenido en el cwd, rechaza symlinks, atómico) + humo de CLI por subcomando (v0.47.0) |
| test_v0480_part_a.py | Trainer consciente del currículo (BETA): DynamicCurriculumPolicy congelado + límites; compute_bucket_weights softmax + water-fill (invariante estricto del piso); validate_distributed_curriculum como puerta DDP; render_curve + parse_history_jsonl con tope de DoS de 100k filas; validadores cruzados de SoupConfig (requires-curriculum / mlx-rejected / non-SFT-rejected / floor ≤ 1/buckets); CLI `soup runs curriculum-curve` (rechazo de symlink por TOCTOU + tope de 50 MB + salida 2 con JSONL corrupto) (v0.48.0) |
| test_v0490.py | v0.49.0 Contexto largo y arquitectura: kernels matemáticos de YaRN (yarn_find_correction_dim / yarn_find_correction_range / yarn_linear_ramp_mask / yarn_get_mscale — rechazo de bool/NaN/Inf, low==high desambiguado automáticamente); campos de esquema de YaRN (yarn_factor / yarn_attn_factor / yarn_beta_fast / yarn_beta_slow) + validador cruzado que rechaza campos de yarn sin rope_scaling_type=yarn; endurecimiento y verificación de Dynamic NTK; puerta de esquema de LongLoRA (familia Llama + sft + transformers + !ring_attn); regex de límite de palabra de `is_llama_model` (refleja la política de `is_gemma4_model` de v0.39.0 / `is_llama4_model` de v0.44.0); Llama 3.1 consciente de NTK (`scale_inv_freq_llama3` con transición suave / bool rechazado en cada parámetro / `old_context_len` cero rechazado; `detect_llama3_rope_in_config` con alias `rope_type` + comprobación explícita de `is None`); validación de entrada en el límite público de `get_rope_scaling_config` (target_length / original_length / yarn_factor — corrección de revisión de seguridad) (v0.49.0) |
| test_v0480_part_b.py | Optimizador de mezcla de datos (BETA): parse_budget (dígitos + sufijo s/m/h [60s, 24h]) + validate_datasets (contención + symlink + dedup + tope de 32) + MixCandidate (simplex + finito + bool rechazado) + BudgetTracker (reloj inyectable) + run_mix_optimizer (fallos de proxy aislados + propagación de KeyboardInterrupt + omisión de NaN + corte parcial de presupuesto) + render_mix_recipe_yaml (defensa contra inyección de YAML) + write_mix_recipe / load_mix_recipe (atómico + TOCTOU + tope de 256 KB) + CLI `soup data mix --optimize / --apply` (v0.48.0) |

| test_v0500_part_a.py | v0.50.0 Parte A — variantes del objetivo GRPO (gspo / dapo / dr_grpo / bnpo / two_sided / rft / standard); allowlist cerrada + `GRPOVariantSpec` congelado + inmutabilidad de `MappingProxyType`; `validate_grpo_variant` (bool / byte nulo / tamaño excesivo / desconocido rechazados); `validate_grpo_delta` (bool / NaN / Inf / fuera de rango rechazados); stub diferido-en-vivo de `apply_variant_loss`; integración con TrainingConfig (acepta Literal + two_sided requiere validador cruzado de delta + field_validator de NaN/Inf) (v0.50.0 Parte A) |
| test_v0500_part_b.py | v0.50.0 Parte B — GRPO de contexto largo + sleep de vLLM; `validate_long_context_grpo_compat` (exclusividad de task / backend / ring-attention + byte nulo + guards de bool); `validate_vllm_sleep_mode_compat` (allowlist de transformers / unsloth + byte nulo); stub diferido de `apply_vllm_sleep_mode` con marcador v0.50.1; integración con SoupConfig (vllm_sleep_mode requiere task='grpo' — corrección HIGH de revisión de código) (v0.50.0 Parte B) |
| test_v0500_part_c.py | v0.50.0 Parte C — allowlist de backends de rollout para agentes multiturno (art / ruler / nemo_gym / openenv); `RolloutBackendSpec` congelado + inmutabilidad de `MappingProxyType`; `validate_rollout_backend` (bool rechazado); mapeo por entrada de `required_rollout_package`; stub diferido de `launch_rollout`; puerta de tarea de SoupConfig + rechazo de mlx (v0.50.0 Parte C) |
| test_v0500_part_d.py | v0.50.0 Parte D — 7 perillas de estabilidad/eficiencia (`ref_model_ema_alpha` / `replay_buffer_size` / `async_grpo_prefetch` / `tis_threshold` / `mask_truncated_completions` / `defer_rerolling` / `skip_zero_advantage` / `off_policy_mask_threshold`); field_validator explícito que rechaza bool en todos los campos numéricos (corrección HIGH de la guía tdd); `mask_truncated_completions` requiere validador cruzado de `tis_threshold`; puerta de tarea de SoupConfig que nombra cada campo infractor; puerta de tarea de `grpo_fp16` (corrección HIGH de revisión de código) (v0.50.0 Parte D) |
| test_v0500_part_e.py | v0.50.0 Parte E — `task='prm'` (Process Reward Model) + bandera `vision_grpo`; `validate_prm_compat` (puertas de data.format / modality / mlx); `validate_vision_grpo_compat` (task ∈ {grpo, ppo} / modality='vision' / no-mlx); stub diferido de `build_prm_trainer`; integración con SoupConfig con todas las rutas de rechazo ejercitadas (v0.50.0 Parte E) |
| test_v0520.py | v0.52.0 Modalidad II — TTS / classifier / distill / BitNet / EBFT-GDPO / MoE quant / reasoning_effort: allowlist de familias TTS + allowlists de emoción por familia (Orpheus + Oute) + validate_tts_compat; tareas classifier / reranker / cross_encoder + num_labels (con guard de bool en field_validator) + dedup de label_names + puertas de campos solo para classifier; divergencia de distill (alias kl canonicalizado, Literal excluye el alias) + teacher_model + límites de distill_temperature; cuantización BitNet 1.58 + stubs de formato de exportación bitnet/tq1_0 + receta Falcon-E + detección por prefijo de organización is_bitnet_model; allowlists de variantes EBFT (structured/strided) + GDPO (standard/length_normalized/margin) + puertas de tarea; cuantización de expertos MoE (nf4/int8_rowwise) + train_router_only que requiere moe_lora=true; reasoning_effort + train_on_eot con puerta de tarea de la familia SFT; 6 recetas nuevas (5 TTS + Falcon-E BitNet); cobertura de correcciones de revisión (guard de bool en num_labels, allowlist de emociones de Oute, lazy-import en el validador de classifier, puertas de tarea, matrices de tamaño excesivo / NaN / Inf). Número de pruebas: 272 (v0.52.0) |
| test_v0531_82.py | v0.53.1 #82 detección de pre-cuantizado en autopilot: `detect_prequantized_format` + `decide_quantization(prequantized=...)` + `detect_prequantized_format_from_path` con contención al cwd + rechazo de symlink en config.json + alias de nombre/config + regex de límite de palabra (v0.53.1) |
| test_v0531_142.py | v0.53.1 #142 cableado en vivo de merge_4bit + export_torchao: fusión de una sola etapa BNB-4bit + PTQ de TorchAO con allowlist de kwargs por esquema + CLI `--save-format` + `--quant-config` + `load_quant_config` (yaml.safe_load + tope de 256 KB + allowlist de extensiones) + TOCTOU de rutas (v0.53.1) |
| test_v0531_139.py | v0.53.1 #139 export_advanced_gguf en vivo: pipeline de llama.cpp en 3 etapas (convert → imatrix → quantize) + eliminación del prefijo UD + forma de argv del subproceso + alternativa de alias JSONL de `_prepare_calibration_text` + eliminación de bytes nulos + tope de 50 MB + POSIX O_NOFOLLOW + escape Rich de `_safe_stderr` (v0.53.1) |
| test_v0531_109.py | v0.53.1 #109 deploy autopilot --measure: `compute_cache_key` + `sha_of_file` + bandas OK/MINOR/MAJOR de `measure_candidate` + alternativa suave de `pick_best` (max-por-delta) + ida y vuelta de caché con rechazo de symlink al cargar Y al guardar + integración con la CLI + tope `_MAX_CANDIDATES=32` + regresión de escape de marcado de `render_measure_table` (v0.53.1) |
| test_v0530.py | v0.53.0 Quant Menu II — GGUFs UD + caché KV + NVFP4 + paridad LF + formatos de guardado: Partes A+B GGUF (escalera UD de 14 entradas + IQ 12 + frozensets de Apple/ARM 10 + invariante de no solapamiento + `validate_*` sin distinguir mayúsculas + matriz de rechazo + unión `is_advanced_gguf_format` + búsqueda O(1) de `_LOWER_INDEX` + inmutabilidad de MappingProxyType + rechazo de forma de `validate_calibration_data_path` + límite de 4096 + stub diferido v0.53.1 de `export_advanced_gguf`); Parte C caché KV (frozenset `KV_CACHE_TYPES` + mayúsculas de `validate_kv_cache_type` + rechazo de bool/null/tamaño excesivo/no-string + `requires_hopper` delega en la especificación + `get_kv_cache_spec` congelado + esquema rechaza fp8-en-mlx con mensaje específico + q8_0-en-mlx permitido); Parte D precisión avanzada (`fp8_attention` requiere `quantization_aware='fp8'` ANTES del ordenamiento de la puerta mlx + guards de bool en cada parámetro string + esquema rechaza-sin-fp8-qat; rechazo mlx + vision de `nvfp4` + guards de bool; matriz de rechazo de `unsloth_bnb_4bit` backend='unsloth' + quantization='4bit'; `apply_*` diferido); Parte E paridad LF (`bnb_4bit_use_double_quant` rechaza parametrize none/8bit/gptq; `llm_int8` rechaza default-none + 4bit; `quantize_ref_model` pasa en dpo/grpo/kto + rechaza sft/pretrain; `quantize_reward_model` pasa en ppo/reward_model + rechaza dpo; `TypeError("v0.53.0 flag must be bool")` explícito desde `_validate_v053_bool_fields`; null explícito aflora como ValidationError `valid boolean`); Parte F formatos de guardado (normalización a minúsculas de `MERGE_SAVE_FORMATS` + matriz de rechazo; `TORCHAO_PTQ_SCHEMES` SENSIBLE A MAYÚSCULAS — `int4weightonly` rechazado; límite de 4096 de `validate_quant_config_path`; `MergeSaveSpec` + `TorchAOPTQSpec` congelados + inmutabilidad de MappingProxyType; `merge_4bit` + `export_torchao` diferidos); Transversal (ida y vuelta completa de YAML de 5 campos + invariante de cardinalidad + assert tautológico reemplazado por allowlist + re-validación idempotente + `get_gguf_spec` desconocido lanza + guards de bool en backend/modality/quantization en cada validador de la Parte D). Número de pruebas: 154 (v0.53.0) |
| test_v0570_part_a.py | v0.57.0 Parte A `soup adapters diff` — entropía SVD de `effective_rank` (identidad / concentrado / vacío / 1D / cero / eps-bool / eps no finito); `compute_layer_diffs` (idéntico-cero / norma-conocida-5 / forma-no-coincide-omitido / solo-intersección / no-mapping / tope de tamaño excesivo); `render_report_json` + `render_report_markdown` (ida y vuelta / TypeError de no-reporte / sección solo-listas); `LayerDiff` congelado vía FrozenInstanceError; `compute_adapter_diff` de extremo a extremo (fixture de safetensors + rechazo fuera del cwd + top_k bool + límite 1/201 + safetensors faltante + rechazo de .bin + symlink POSIX en el archivo de pesos); CLI (humo de table/json/markdown + --format desconocido + --output requiere no-table + fuera del cwd + escritura atómica); guard de source-grep sin torch de nivel superior. Número de pruebas: 36 (v0.57.0 Parte A) |
| test_v0570_part_b.py | v0.57.0 Parte B `soup adapters merge` — 4 estrategias en numpy puro: `merge_linear` (average / weighted / intersection / forma-no-coincide / único-rechazado / 17-demasiados / peso bool / negativo / rechazo de NaN / inf / suma-cero / longitud-no-coincide); `merge_ties` (density mitad-superior / elección de signo mayoritario / corrección de revisión **tied-sign-defaults-positive** / límites de density / density=1.0 inclusivo / density-bool); `merge_dare` (determinista por seed / seeds distintas divergen / seed bool / seed negativa / density=1 igual a linear); `merge_svd` (sin-rank igual a linear / rank reduce / no-2d pasa directo / clamp de rank / rank inválido); `merge_adapters` e2e linear + estrategia desconocida + salida-fuera-del-cwd + **rechazo de symlink POSIX en el safetensors de salida**; invariantes de frozenset `SUPPORTED_STRATEGIES` + tupla `STRATEGY_ORDER`; stub `predict_merged_verdict` + no-reporte + tipo canary_suite; FrozenInstanceError de `MergeReport`; ayuda de la CLI / linear / estrategia desconocida / pesos inválidos / rechazo de adaptador único; source-grep sin torch de nivel superior. Número de pruebas: 43 (v0.57.0 Parte B) |
| test_v0570_part_c.py | v0.57.0 Parte C `soup adapters blame` — `parse_budget` (60/60s/5m/2h / bajo-el-piso / sobre-el-tope / formato inválido / vacío / bool / byte nulo / no-string); `plan_blame` (feliz / presupuesto inviable / offsets de shard / división desigual / dataset vacío / adaptador fuera del cwd / dataset fuera del cwd / capa inválida / límites de shard / **shards bool=True AND False** / **budget_seconds bool**); el stub `run_blame` lanza NotImplementedError v0.57.1 + TypeError de no-plan; FrozenInstanceError de `BlamePlan` + `BlameShardWork`; ayuda de la CLI + humo solo-plan + presupuesto inválido + aviso de runner en vivo con salida 0. Número de pruebas: 33 (v0.57.0 Parte C) |
| test_v0580.py | v0.58.0 `soup loop` volante de datos centrado en la CLI — `LoopState` congelado + 3 allowlist cerradas + rechazo de bool/null/tamaño excesivo/no-string; `with_status` / `bumped` (contador desconocido + delta bool rechazados); E/S del archivo de estado atómica + contención al cwd + rechazo de symlink POSIX + tope de 1 MiB + descarte de campos desconocidos compatible hacia adelante; `init_state` se niega a sobrescribir sin `--force`; `CanaryPolicy` (entre campos, NaN/Inf, congelado); enrutamiento SHA-256 determinista (solo-estable / 0% / 100% / determinismo / división del 25% aproximada / clave vacía / clave NUL / no-política); `rollback` (limpia el canary / enrutar-tras-rollback todo-estable / motivo / no-política); `BucketStats` (registro + veredicto OK/MAJOR/UNKNOWN + límites + bucket inválido + snapshot MappingProxyType); `parse_budget_string` (5 felices + vacío/basura/NUL/negativo/desbordamiento); `check_budget` (dentro/permitido-presupuesto/permitido-tope-diario/None=ilimitado + guards de negativo/NaN/bool); `reset_daily_counter_if_new_day` (mismo-día/día-nuevo/None-previo/negativo); `IterationRecord` congelado + allowlist de veredicto + id con separador de ruta; escritura/lectura de iteración ida y vuelta + fuera del cwd + faltante + listado ordenado + no-registro + manifiesto inválido; `new_iteration_id` único × 20; `run_once` (por defecto + omisión por presupuesto + callbacks personalizados + tipos inválidos); `WatchConfig` (por defecto + poll_interval/max_iterations/callable incorrectos); `watch` ejecuciones finitas + on_iteration cambia el estado a stop; `maybe_rollback` (OK no-op / MAJOR limpia / UNKNOWN no-op / no-política / no-str); humo de CLI (loop --help / init / rechaza sobrescribir / --force / presupuesto inválido / status sin init / status / ciclo pause+resume / pause-stopped / resume-running / watch --max-iterations / mutex --foreground/--detach / canary / canary tráfico inválido / canary igual-a-estable / replay list-empty / replay show / replay desconocido); source-grep (cli.py registra loop / `__init__.py` == 0.58.0 / sin torch de nivel superior en 5 módulos de loop / typer.Typer). Cobertura de correcciones de revisión: 1 CRITICAL + 3 HIGH + 3 MEDIUM + 1 LOW. Número de pruebas: 160 (188 pasan + 1 omitida en POSIX). (v0.58.0) |
| test_v0570_part_d.py | v0.57.0 Parte D `soup adapters branch/checkout/branches` — `create_branch` feliz + con-dataset + nombre inválido + nombre con byte nulo + nombre bool + config fuera del cwd + config faltante + base vacío + base con byte nulo + **base_model bool rechazado** + config de tamaño excesivo 1MiB + escritura atómica; `list_branches` vacío + ordenado; `load_branch` ida y vuelta + faltante + nombre inválido + **rechazo de symlink POSIX**; `delete_branch` true-cuando-presente + false-faltante + **rechazo de symlink POSIX** + rechazo de traversal; `write_checkout` escribe el destino + detección de drift + rechazo fuera del cwd + no-rama + fuente faltante; FrozenInstanceError de `Branch`; override de entorno SOUP_BRANCHES_DIR (válido-honrado + byte nulo recurre al respaldo vía os.environ.get mockeado + CRLF recurre al respaldo); humo de CLI create / list / checkout / nombre inválido / config faltante / rama faltante / lista vacía. Número de pruebas: 37 (v0.57.0 Parte D, 4 omitidas en POSIX bajo Windows) |
| test_issue349_measured_fit.py | v0.73.1 #349 — ajuste de VRAM de stream medido: `training.stream_vram_probe` (bool, por defecto false, solo sft) decide la verificación previa de VRAM del layer-streaming MIDIENDO un forward+backward real con la forma configurada en lugar de usar la fórmula ajustada; justificado en RTX 3050 donde la fórmula subestima 0.934x con seq 5120 / 0.787x con seq 6144. Validación de config + límites de esquema (puerta de tarea sft, rechaza stream_layers=false). Además: el field_validator de `training.batch_size` ahora rechaza 0 y valores negativos (antes aceptaba `-4`). Número de pruebas: 33 (v0.73.1) |
| test_issue372_no_reexec_hint.py | #372 — la pista de `--no-reexec` se deriva del mismo argv que el auto-reexec (`build_train_reexec_argv` + `hint_argv_from_reexec`); el comando impreso tokenizado es un superconjunto de las banderas del usuario parametrizado sobre `--fsdp` / `--deepspeed` / `--config` / `--trust-remote-code`; el control sin banderas extra sigue imprimiendo un comando mínimo válido; el humo de CLI no se omite. |
| test_issue392_mlx_target_keys.py | v0.73.1 #392 — el `adapter_config.json` de MLX serializaba `target_modules` SIN RESOLVER, así que un adaptador MLX por defecto se cargaba como un no-op silencioso. `_apply_lora` resolvía `auto` en una variable LOCAL y entrenaba los módulos resueltos; el escritor serializaba el valor crudo de la config, así que el archivo llevaba `{"keys": ["auto"]}`, `linear_to_lora_layers` no coincidía con ningún módulo y `load_weights(strict=False)` descartaba los 96 tensores LoRA SIN UNA PALABRA — la generación con el adaptador era bit-idéntica a la base. `auto` es el valor por defecto del esquema, así que esto afectaba a cada ejecución de MLX que no nombrara sus módulos a mano, y `adapter_config.json` existe específicamente para prometer que el directorio de salida carga con `mlx_lm.load(..., adapter_path=...)`. Reportado por @armanbot-jpg con el control que lo aísla: editar `keys` a mano hace que el mismísimo `adapters.safetensors` produzca el comportamiento ajustado, es decir, los pesos estaban bien y la config no. **Tercera vez en este proyecto que una curva de loss sana envía un artefacto muerto** (las claves `.inner.` de v0.72.0, el adaptador de ajuste fino completo de #362, esto) — la curva de loss estructuralmente no puede verlo, solo el artefacto puede. Reparado como UNA única `resolve_mlx_target_keys()` compartida en lugar de enseñar al escritor a resolver también, ya que dos copias de "¿qué módulos entrenamos?" es como se desviaron. 10 en rojo antes, 10 en verde después; dos son controles (el valor resuelto nunca debe contener `auto` cualquiera que sea la entrada, y rank/scale/dropout deben seguir viniendo de la config, así que una reparación que codificara el bloque en duro fallaría en vez de pasar). (v0.73.1) |
| test_issue441_coerce_token_ids.py | #441 — `coerce_token_ids` es el helper público compartido (`data_doctor` debe importar ese nombre, no `_coerce_token_ids`, y no debe definir su propia copia); un mapping duck-typed (tiene `get`, no un `collections.abc.Mapping`) devuelve `[1, 2, 3]` en lugar de lanzar `input_ids[0]='input_ids'`; el mismo objeto que lleva `assistant_masks` toma la ruta de máscara (las labels conservan solo el token enmascarado — falla si solo se ensancha la puerta de coerce); `input_ids` faltante sigue lanzando el error existente; ambos puntos de llamada comparten `_is_mapping_like` (`isinstance(out, Mapping)` vive en exactamente un lugar). |
| test_v07302.py | v0.73.2 mejoras de la suite leg-2 de `soup ship` + piso de ruido: `--noise-floor N` (N en [2,10], solo en modo métrica) reejecuta la base N veces, el piso por eje = dispersión max-min, condiciona cada uno a max(umbral, piso) — `VramFit.noise_floor_block` congelado + `measure_base_spreads` (solo base, N pases forward) + agregación `compute_noise_floors`; esquema de JSON de evidencia ampliado con un bloque `noise_floor` opcional (dict de runs + floors); 8.ª suite integrada `mini_over_refusal` (puerta de seguridad inversa, benigno-no-rechazado) + recuento de DEFAULT_GENERAL_SUITE 8; correcciones del scorer: `extract_mcq_letter` ahora entiende el formato de una sola letra `\boxed{C}`, `tool_call_name_match` tolera el sobre externo faltante `{"function": ...}` (lleva AMBOS name + arguments), `score_bundled_suite` lanza TypeError para un generador no invocable (antes devolvía 0.0 en silencio); aviso de escala de línea base para las tres suites afectadas (mini_mmlu / mini_common_sense / mini_tool_call); la herramienta `ship_evidence` de MCP respeta noise_floor en el esquema de evidencia. (v0.73.2) |
| test_v0560.py | v0.56.0 `soup diagnose` tarjeta de informe de modos de fallo tras el entrenamiento — Parte A 6 sondas (forgetting Δ-accuracy + banda de tolerancia; delta de refusal advbench/xstest + tope `_MAX_REFUSAL_SCAN=8192`; formato JSON/regex/tool_call con sonda ReDoS + frozenset `_VALID_KINDS`; mode_collapse Jaccard por pares de n-gramas sobre K completions; memorization eco del prefijo de entrenamiento vía `split_prefix`; contamination reutilización de solapamiento de n-gramas de v0.47 + tope de complejidad combinada N×M>1e9); Parte B `FailureReport` + `FailureScore` dataclasses congeladas con taxonomía OK/MINOR/MAJOR (umbrales ≥0.85/≥0.60) + SDK `compose_report` / `build_report` + `write_report` atómico (contención realpath + rechazo de symlink) + `render_badge_svg` SVG de 6 celdas con escape HTML + humo de CLI (--evidence/--output/--badge/--attach-to-registry); Parte C tipo de artefacto `diagnose_report` + helper de rechazo MAJOR de `soup train --diagnose-gate`. Cobertura de correcciones de revisión: escritura de insignia atómica + segura ante TOCTOU, typer.Exit (no sys.exit), tope de tamaño de evidencia de 16 MiB, centralización de `extract_row_text`, `tokenize` delega en `_eval_text`, saneamiento de byte nulo de los extras, guards de regresión de source-grep. Número de pruebas: 123 (v0.56.0) |
| test_v0540.py | v0.54.0 decisión previa al vuelo de `soup advise` — Parte A motor de Veredictos (allowlists TASK_CATEGORIES + CHOICES; Verdict / DatasetProfile / ROIEstimate congelados; `classify_task` señales de palabra clave + tool_calls + traza de razonamiento + goal-steers; `compute_dataset_profile` forma + diversidad + elegidos/rechazados + detección de razonamiento; `build_verdict` rúbrica de 5 ramas con `_MIN_ROWS_FOR_GRPO=500`; `load_advise_dataset` contención al cwd + rechazo de symlink + eliminación de BOM + rechazo de JSON malformado); Parte B runner de sondas (`synth_probe_baselines` + `synth_probe_lora_delta` stubs heurísticos con kwargs compatibles hacia adelante `model`/`device`/`lr`/`timeout_seconds`; `format_verdict_rubric` + handoff `next_command_for`); Parte C aprendizaje entre proyectos (`record_verdict` + `load_history` + `_append_with_lock` con bloqueo entre procesos fcntl/msvcrt; `~/.soup/advise_history.jsonl` + sidecar `.lock` en Windows; contención de la anulación de entorno `history_path`; tope de 64 KB por línea en las lecturas del historial); humo de CLI (subcomandos run / explain / compare + `_rewrite_advise_argv` limitado a argv[1]); cobertura de correcciones de revisión (escritura atómica en scratch + rechazo de symlink al leer; estrés de registro concurrente de 8 hilos; límite 49↔50 / 499↔500 / 4096↔4097). Número de pruebas: 136 (v0.54.0) |
| test_v0510.py | v0.51.0 Expansión del catálogo de modelos + hubs de modelos alternativos: Parte E hubs.py (`SUPPORTED_HUBS` + `validate_hub_name` + `validate_hub_endpoint` paridad de SSRF / rechazo de CRLF / IPv6 mapeada privada rechazada / IPv6 loopback ok / caracteres de control; anulación por variable de entorno de `resolve_endpoint`; `default_endpoint` + `endpoint_env_var` + `required_hub_package` + `is_hf` con guards de bool; inmutabilidad de MappingProxyType); campo `hub` de TrainingConfig (por defecto + acepta Literal + rechaza None + normalización sin distinguir mayúsculas + ida y vuelta de YAML) + `_validate_hub_supported` de SoupConfig (mlx + no-hf rechazado; mlx + hf aceptado; modelers + transformers aceptado); Parte D extensión de MULTIPACK_ARCHITECTURES (parametrize de 20 arcos nuevos + legado preservado + recuento exacto=38 + inmutabilidad de frozenset); Partes A/B/C 26 recetas nuevas (parametrize sobre cada nombre × {get_recipe / RecipeMeta / carga de SoupConfig / yaml.safe_load / id de modelo sin partes nulas/en blanco/vacías / límites de max_length / campos requeridos de GRPO}); baichuan-sft usa `hub: modelscope`; recuento total de recetas >= 105 (v0.51.0) |

(Nota: la tabla de archivos de prueba de arriba es un subconjunto representativo, no exhaustivo — cada archivo `tests/test_*.py` corresponde a un área de función o a una versión.)

## Reservar un issue

Comenta en él diciendo que lo tomas. GitHub solo nos permite poner la insignia
formal de asignado a los colaboradores del repositorio, así que el comentario **es** la reserva — se verá
como no asignado de todas formas, y tratamos el hilo como el registro.

Dos compromisos a cambio:

- **Si un maintainer termina haciendo el trabajo en tu lugar, se te avisa, en el hilo,
  antes de que aterrice o cuando lo haga.** Esto se ha incumplido una vez ([#313](https://github.com/MakazhanAlpamys/Soup/issues/313)):
  a un colaborador se le dio luz verde, definió un protocolo cuidadoso, y luego la
  medición se ejecutó en hardware prestado y el issue se cerró sin una palabra
  para él. Fue un fallo de proceso de nuestro lado, y este párrafo existe
  por eso.
- **Una reserva que queda en silencio se libera con un comentario, no en silencio.** Primero
  preguntamos, y "la vida se interpuso" no necesita explicación — mira
  [#280](https://github.com/MakazhanAlpamys/Soup/issues/280) para ver la forma.

Si un issue necesita hardware que no tienes, dilo en la reserva. Varios issues
abiertos están etiquetados `infra-blocked` exactamente por eso, y saberlo temprano
es más útil que una rama estancada.

### Cómo se encola el trabajo

El tiempo de revisión y la CI alojada son los recursos escasos aquí, así que unas pocas reglas mantienen la fila
avanzando. Se aplican con criterio, no mecánicamente, y nada se cierra ni se
libera como castigo.

- **Las revisiones van en orden de número de issue.** Un pull request que está en verde y cuyas rondas
  de revisión están respondidas avanza; uno que espera a su autor no frena a los demás.
- **Pull requests abiertos por autor.** Un colaborador con menos de 3 pull requests fusionados
  puede tener 2 abiertos a la vez; con 3 a 9 fusionados, 3; con 10 o más, 5. Los pull
  requests de más no se cierran: un comentario dice detrás de qué esperan, y pueden
  convertirse en borradores. Marca uno como listo para revisión cuando aterrice otro.
- **Reservas.** Mantén como máximo 2 reservas cuyo pull request todavía no esté abierto. Un comentario sin
  un plan se lee como una pregunta, no como una reserva.
- **Una reserva en silencio.** Sin pull request y sin noticias tuyas durante 7 días, recibes un
  recordatorio amable en el hilo. Si pasan otros 7 días sin respuesta, la reserva se
  libera con un comentario. Nada de lo que escribiste se pierde, y puedes retomarla.
- **Un pull request en silencio.** Un pull request que espera a su autor durante 14 días recibe un ping
  en el hilo. Si pasan otros 7 días, lo cerramos con una nota de que puede reabrirse
  en cualquier momento.
- **CI completa.** Las celdas de prueba requeridas se ejecutan cuando un maintainer añade la etiqueta `ci:full` a un
  pull request listo para fusionar, no en cada push; los demás pushes ejecutan un subconjunto rápido.
  Un pull request en borrador no ejecuta ninguna CI hasta que lo marques como listo para revisión.

### Dificultad del issue

Los maintainers puntúan cada issue triado de 1 a 10 con una etiqueta `difficulty:N`, para que puedas elegir
algo que se ajuste a tu tiempo y experiencia. Cómo se elige el número y qué significa cada nivel
está en [docs/issue-difficulty.md](docs/issue-difficulty.md). Cuando abres un issue
puedes sugerir una dificultad en el formulario; el maintainer fija la definitiva.

## Hacer cambios

### 1. Crear una rama

```bash
git checkout -b feature/your-feature-name
# o
git checkout -b fix/your-bug-fix
```

### 2. Haz tus cambios

- Escribe código siguiendo las guías de estilo anteriores
- **Escribe primero las pruebas** (TDD) — y luego implementa para que pasen
- Mantén los commits enfocados y lógicos

Para cada pull request visible al usuario, añade su entrada a
`changelog.d/<latest-release>/<number>.<category>.md` en lugar de editar `CHANGELOG.md`;
usa la versión publicada más reciente en `CHANGELOG.md`, prefiere el número del PR en lugar del
issue que cierra, y elige `added`, `changed`, `deprecated`, `removed`, `fixed` o
`security`.

Si añades o quitas un comando, opción o argumento de la CLI, o un campo de `soup.yaml`, o cambias su
tipo, valor por defecto o límites, ejecuta `python scripts/gen_reference_docs.py` y haz commit de las páginas
que reescribe (`docs/cli-reference.md`, `docs/config-reference.md`); las pruebas fallan hasta
que lo hagas y nombran lo que difiere. Reformular una cadena de ayuda o una descripción no necesita nada:
los maintainers regeneran el texto en la publicación. Ante un conflicto de fusión en cualquiera de las páginas
generadas, conserva cualquiera de los dos lados y vuelve a ejecutar `python scripts/gen_reference_docs.py`.

### 3. Ejecuta pruebas y lint

Antes de hacer push, asegúrate de que todo pasa:

```bash
# Lint primero
ruff check --fix src/soup_cli/ scripts/ tests/

# Luego ejecuta las pruebas
pytest tests/ -v --tb=short
```

### 4. Commit

Escribe mensajes de commit claros y descriptivos siguiendo [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add <specific-files>
git commit -m "feat: add support for X"
# o
git commit -m "fix: resolve Y when Z"
```

Tipos: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

### 5. Haz push y abre un PR

```bash
git push origin feature/your-feature-name
```

Luego abre un pull request en GitHub con:
- Título claro que describa el cambio
- Descripción de qué y por qué
- Referencia cualquier issue relacionado (p. ej., "Closes #123")
- Resultados de las pruebas

## Lista de verificación del pull request

Cuando abres un PR, la plantilla de GitHub mostrará esta lista de verificación:

- [ ] `ruff check src/soup_cli/ scripts/ tests/ benchmarks/` pasa
- [ ] `pytest tests/ -v` pasa
- [ ] Documentación relevante actualizada (`README.md` y la página correspondiente en `docs/`) si es necesario
- [ ] Pruebas nuevas añadidas para la funcionalidad nueva
- [ ] Sin cambios incompatibles (o documentados en la descripción del PR)

## Arquitectura y decisiones de diseño

### Imports diferidos para la velocidad

Los imports pesados de ML (torch, transformers, trl) se importan dentro de los manejadores de comandos para que la CLI se mantenga rápida. Los usuarios pueden ejecutar `soup version` o `soup --help` al instante sin esperar a que cargue PyTorch.

### Pydantic para la validación de configuración

Todas las configuraciones YAML se validan con modelos de Pydantic v2. Estos modelos son la única fuente de verdad para los campos válidos y los valores por defecto. Consulta `config/schema.py`.

### Trainers como wrappers

`trainer/sft.py`, `trainer/dpo.py`, `trainer/grpo.py`, `trainer/ppo.py` envuelven los trainers de HuggingFace TRL con:
- Cuantización automática (BitsAndBytes, torchao QAT)
- Configuración automática de LoRA (PEFT)
- Estimación automática del tamaño de batch
- Integración con la barra de progreso

### El seguimiento de experimentos es SQLite

No se necesitan dependencias externas. Todas las ejecuciones, métricas y resultados de eval van a `~/.soup/experiments.db`.

### Normalización del formato de datos

Varios formatos (Alpaca, ShareGPT, ChatML, LLaVA, ShareGPT4V) se normalizan a una estructura unificada `{"messages": [...]}` en `data/formats.py`.

## Añadir una función nueva

### 1. Nuevo tipo de tarea de entrenamiento

Si añades un algoritmo de entrenamiento nuevo:

1. Crea `trainer/your_trainer.py` que envuelva el trainer de TRL adecuado
2. Añade campos de configuración a `config/schema.py` (Pydantic v2)
3. Añade una plantilla YAML en `src/soup_cli/templates/` + una entrada en `manifest.json` (mira las 21 plantillas existentes)
4. Actualiza `commands/train.py` para enrutar a tu trainer
5. Añade 30 o más pruebas en `tests/test_your_trainer.py`
6. Actualiza `README.md` + la página relevante en `docs/`

### 2. Formato de datos nuevo

1. Añade lógica de detección y conversión a `data/formats.py`
2. Añade pruebas en `tests/test_formats.py`
3. Actualiza `data/loader.py` si es necesario
4. Documenta en el `docs/*.md` correspondiente

### 3. Comando nuevo

1. Crea `commands/your_command.py` con una función manejadora
2. Registra en `src/soup_cli/cli.py` con `@app.command()`
3. Añade pruebas en `tests/test_your_command.py`
4. Actualiza el texto de ayuda y el README

### 4. Receta nueva

1. Añade una entrada `RecipeMeta` en `recipes/catalog.py`
2. Añade pruebas en `tests/test_recipes.py`
3. Actualiza la sección de recetas de `README.md`
4. Regenera `tests/fixtures/recipe_config_snapshots.json` con
   `python scripts/generate_recipe_snapshot.py` y revisa el diff — la configuración
   resuelta de la receta nueva debe estar en la instantánea confirmada, o
   `test_fixture_and_catalog_have_the_same_recipe_names` falla (#621)

## Issues de buen primer paso

Busca issues etiquetados como [`good first issue`](https://github.com/MakazhanAlpamys/Soup/labels/good%20first%20issue) en GitHub. Son tareas para principiantes que te ayudan a familiarizarte con el código.

Buenas áreas para tus primeras contribuciones:
- **Recetas nuevas** — añade una configuración lista para un modelo popular (mira `recipes/catalog.py`)
- **Documentación** — mejora los docstrings, los ejemplos del README o las configuraciones de ejemplo
- **Pruebas** — aumenta la cobertura de los comandos existentes
- **Correcciones de errores** — revisa los [issues abiertos](https://github.com/MakazhanAlpamys/Soup/issues) etiquetados como `bug`

## CI/CD

GitHub Actions se ejecuta en cada push y PR:
- **ruff** linting en Python 3.11 (debe pasar)
- **pytest** en Python 3.10, 3.11, 3.12 sobre Ubuntu, Windows, macOS (debe pasar)
- **transformers-floor** en Ubuntu / Python 3.11 (#478): instala
  `.[dev]` bajo `.github/constraints/transformers-floor.txt`, verifica la
  versión fijada de `transformers` y ejecuta el guard estático de sitios de carga
  `dtype=` / `torch_dtype=`. El piso declarado de `[train]` sigue siendo
  `transformers>=4.36.0,<5.0.0` en `pyproject.toml`; `transformers==4.36.0`
  no puede resolverse contra el rango declarado de `trl`, así que este trabajo no reescribe
  ese pin.
- **benchmark** (`.github/workflows/codspeed.yml`, #1065): un benchmark de regresión
  de CodSpeed, con su propio flujo de trabajo y fuera de la matriz anterior.
  Está **apagado**: el flujo de trabajo solo arranca por despacho manual, así que no añade ninguna
  verificación a tu pull request. Activarlo requiere la GitHub App de CodSpeed, la
  variable de repositorio `CODSPEED_ENABLED` puesta en `true`, y los disparadores
  `push` / `pull_request` restaurados en el archivo del flujo de trabajo (su encabezado es la
  lista de verificación). Aun así nunca se ejecuta en un PR de un fork — GitHub no concede a una
  ejecución de `pull_request` desde un fork un token OIDC, así que la subida no podría
  autenticarse. No es una verificación requerida y no bloquea nada.

**Los pull requests ejecutan primero un subconjunto rápido.** Un push a un pull request ejecuta `lint`
y las pruebas solo en Ubuntu / Python 3.12. Las celdas requeridas (Ubuntu en Python
3.10 y 3.12, Windows y macOS en 3.12, más los trabajos de humo) se ejecutan cuando un
maintainer añade la etiqueta `ci:full` al aprobar, y en cada push posterior mientras
la etiqueta permanezca. Hasta entonces las demás verificaciones `test (...)` aparecen como *Expected* y
los trabajos de humo como *skipped*; ninguna de las dos es un fallo por tu parte. Un PR se fusiona
solo cuando esas verificaciones están en verde en su head actual. Tras la fusión, `main`
ejecuta el mismo conjunto rápido en cada commit y la matriz completa (todos los Python de
3.10 a 3.12, con 3.11 solo en Ubuntu) una vez por noche; un push a una rama `release/**`
ejecuta cada celda.

**Un pull request en borrador no ejecuta ninguna CI.** Nada arranca mientras un pull request es un
borrador, sin importar lo que se le haga push. Márcalo como listo para revisión y el conjunto rápido
arranca en su head actual.

**La CI de los pull requests instala un stack upstream fijado.** En un pull request y en un
push a `main`, todos los trabajos que instalan torch o transformers (las celdas de prueba,
los trabajos de humo, `Recipe Validation` y `recipe attach`) los toman a ellos y a los
paquetes que resuelven con las versiones exactas de
`.github/constraints/pinned-upstream.txt`, para que una publicación upstream no pueda poner
en rojo todos los pull requests abiertos a la vez. La única excepción es `transformers-floor`,
que instala su propio piso exacto, y los paquetes fuera de ese archivo (numpy,
pydantic, mlx, torchao, ...) siguen resolviendo al más reciente en cada ejecución. La ejecución nocturna
en `main` instala sin restricciones, y ahí es donde aparece primero una publicación nueva de
un paquete fijado. Si tu cambio mueve un
límite en `pyproject.toml` más allá de uno de esos pins, mueve el pin en el mismo pull
request: `tests/test_ci_pinned_upstream.py` nombra el pin, y el encabezado del
archivo de restricciones dice cómo elegir la versión nueva.

**La CI ejecuta las pruebas en paralelo.** Cada celda `test (...)` ejecuta la suite con
`pytest-xdist`, un archivo de prueba por proceso trabajador (`-n logical --dist loadfile`).
Las pruebas de un archivo siguen ejecutándose en orden en un proceso, pero dos archivos pueden ejecutarse
al mismo tiempo y en cualquier orden. Así que una prueba no debe depender de que otro archivo
se haya ejecutado antes, del directorio de trabajo que dejó otro test, ni del
ancho de terminal del proceso: dale a la consola un ancho explícito en la prueba
en lugar de definir `COLUMNS`. Para reproducir un fallo que solo aparece en CI,
instala el plugin (`pip install "pytest-xdist[psutil]"`) y ejecuta
`pytest tests/ -n logical --dist loadfile`.

Consulta `.github/workflows/ci.yml`.

### Marcas de CI obsoletas (#1017)

Una ejecución del flujo de trabajo `pull_request` fija `refs/pull/N/merge` en el **momento de creación de la ejecución**.
`actions/checkout` obtiene ese SHA; el movimiento posterior de `main` no se vuelve a resolver.
`gh run rerun` reproduce la fusión congelada y no puede producir una respuesta distinta.
Por eso una marca verde puede describir un `main` que ya no existe (el
`NameError` de #763 en `_for_terminal`) y por eso una X roja puede describir una prueba que `main`
ya ha corregido. Ninguna es culpa del colaborador, y ninguna se limpia
esperando.

**Si tu PR está en rojo por fallos que no tocas:** fusiona o haz rebase sobre el `origin/main`
actual y haz push. Es la única forma de reconstruir el SHA de fusión para la
matriz de pruebas. Los maintainers no harán `gh pr update-branch` sobre un fork sin
preguntar.

**Si tu PR está en verde y `main` se ha movido:** la verificación `merge-freshness` se
vuelve a publicar en cada push a `main`. Reconstruye la fusión con
`git merge-tree --write-tree` (sin escribir en tu rama) y falla ante ruff F821
o un conflicto. El retraso respecto a `main` se reporta como una verificación **neutral**, no como un
fallo — un tope de 10 commits a la tasa de fusión de este repositorio es `strict: true` con pasos
extra. No vuelve a ejecutar la matriz de pruebas de 13 trabajos.

`required_status_checks.strict` se mantiene apagado: activarlo obligaría a un rebase en
cada fusión. Un procedimiento pre-fusión solo de documentación es lo que los maintainers ya hacen a
mano y no es un control — GitHub fusionará igualmente un PR CLEAN sobre marcas obsoletas.
Pide a un maintainer que añada `merge-freshness` a las verificaciones requeridas en `main` para que
el botón de fusión de GitHub lo respete. El trabajo `gate` en un evento pull_request ejecuta
la propia copia del PR de `scripts/merge_freshness.py`, así que se autoevalúa. Solo la
verificación `merge-freshness` publicada por el trabajo `reeval` desde la copia de `main` es
fiable.

Antes de fusionar cualquier cosa, que `gh pr checks` devuelva todo en verde es necesario y no
suficiente. Compara el `created_at` de la ejecución más reciente con los commits de `main` desde entonces,
o mira `merge-freshness`.

## Publicaciones

El proyecto sigue versionado semántico: `MAJOR.MINOR.PATCH`

### Proceso de incremento de versión

0. **Actualiza el stack de desarrollo, o lee la última ejecución de `dependency drift`** (#323).
   Una suite local en verde sobre un entorno obsoleto es una señal más débil de lo que parece:
   durante v0.72.x la máquina de desarrollo ejecutaba torch 2.5.1 / trl 0.19.1 mientras CI resolvía
   torch 2.13.0 / trl 0.29.1, y seis trainers no podían construir su configuración en
   absoluto con lo que un `pip install` nuevo le da realmente a un usuario. O bien
   `pip install -U -e ".[dev]"` antes de probar, o bien revisa la última ejecución de
   [`dependency drift`](.github/workflows/dependency-drift.yml) — instala
   el stack actual semanalmente, ejecuta la suite contra él y escribe la
   tabla resuelto-vs-declarado en el resumen de su ejecución.
1. Ejecuta `python scripts/assemble_changelog.py` y revisa el `CHANGELOG.md` ensamblado
2. Actualiza la versión en `pyproject.toml` y `src/soup_cli/__init__.py`
3. Ejecuta la suite completa de pruebas y el linting — esto incluye
   `tests/test_issue621_recipe_snapshot.py`, que falla ruidosamente si un valor por defecto
   del esquema cambió desde la última publicación sin regenerar
   `tests/fixtures/recipe_config_snapshots.json`
   (`python scripts/generate_recipe_snapshot.py`); una ejecución en rojo aquí significa que el
   fixture está obsoleto, no que la publicación sea insegura de cortar una vez regenerado
   y revisado
4. Actualiza `README.md`, la página relevante en `docs/` y `SECURITY.md` (si está relacionado con seguridad)
5. Haz commit con el mensaje: `Release v0.X.0`
6. Etiqueta: `git tag v0.X.0 && git push --tags`
7. GitHub Actions verifica que no quedó ningún fragmento atrás y luego publica en PyPI
   con la etiqueta `v*` (Trusted Publisher / OIDC)

## Reconocimiento

Acreditamos a cada colaborador — el reconocimiento es gratis y te lo ganaste.

- **[CONTRIBUTORS.md](CONTRIBUTORS.md):** cada PR fusionado (o adoptado en el árbol) te añade aquí, con tu trabajo y el número de PR.
- **CHANGELOG.md:** desde la publicación que incluye tu cambio, la entrada se etiqueta `(#NNN by @you)`.
- **Trailers de commit:** los PR aterrizan con una línea `Co-Authored-By:` para ti, así que el grafo
  de colaboradores de GitHub refleja tu trabajo incluso tras una fusión squash.

Si contribuiste y no apareces acreditado en algún sitio, eso es un bug — abre un PR o un issue y lo arreglaremos.

## Comunidad

- **Issues:** reporta errores y solicita funciones en [GitHub Issues](https://github.com/MakazhanAlpamys/Soup/issues)
- **Discussions:** haz preguntas en [GitHub Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)
- **Discord:** conversa con los maintainers y otros usuarios en [Discord](https://discord.gg/dgd2pJcjwP) — va bien para
  preguntas rápidas y para hacer pairing en un PR; todo lo que valga la pena encontrar después sigue perteneciendo a Issues o Discussions
- **Código de conducta:** lee [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — también aplica en Discord
- **Seguridad:** reporta problemas de seguridad mediante [SECURITY.md](SECURITY.md) — en privado, nunca en Discord

## ¿Preguntas?

- Consulta el [README](README.md) para el inicio rápido y las funciones
- Consulta la [documentación](docs/README.md) para la referencia completa de funciones y la arquitectura
- Abre una GitHub Discussion para tus preguntas

¡Gracias por contribuir!
