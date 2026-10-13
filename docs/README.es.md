# Documentación de Soup

> 🇪🇸 Traducción al español de [`README.md`](README.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README principal](../README.md)

El [README](../README.md) principal es la puerta de entrada de 5 minutos. Este directorio contiene la referencia
completa de funciones — cada capacidad de `soup`, agrupada por área.

| Guía | Cubre |
|---|---|
| [Tareas y métodos de entrenamiento](training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-entrenamiento, destilación, clasificación, visión/audio/TTS, desaprendizaje, RAFT/RA-DIT, detectores de endurecimiento de bucles, síntesis de verificador de recompensa |
| [Contrato de artefactos de AutoDistill](autodistill.md) | Esquemas del hito A del issue #580, huellas digitales inmutables, política de probabilidad/replay, reanudación transaccional, estimaciones solo de plan y modelo de amenazas |
| [PEFT, contexto largo y eficiencia](peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, zoológico de optimizadores y PEFT, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, currículo, auto-tuning, poda de profundidad + distill-heal (`soup shrink`) |
| [Rendimiento y cuantización](performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, formatos de guardado, Cut Cross-Entropy, gradient checkpointing, kernels, offloading de activaciones, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Ingeniería de datos](data.md) | Formatos, el pipeline con paridad Axolotl/LF, herramientas de datos, generación sintética y forge, tarjetas de calidad, herramientas de trazas, datasets remotos, mezcla, DAGs de recetas |
| [Evaluación y sondas](evaluation.md) | Diseño y compuerta de evaluación, entrenamiento con compuerta de evaluación, benchmarks, métricas NLG, calibración, arena Elo, diagnose, veredicto de `soup ship`, sondas de rayos X post-entrenamiento, A/B, drift, tuneabilidad, `soup advise` |
| [Servicio y exportación](serving-and-export.md) | Servidor compatible con OpenAI, inferencia por lotes, benchmarking, merge/export, endpoint de Anthropic Messages, decodificación especulativa (entrena y mide tu propio draft), autopiloto de despliegue, interfaz web, Agent Forge |
| [Adaptadores, registro y gobernanza](adapters-and-governance.md) | Ciclo de vida y gestión de adaptadores, registro de modelos, Soup Cans, el volante de datos (`soup loop`), edición de conocimiento, steering, controles de cadena de suministro |
| [Inicio rápido de cumplimiento y gobernanza](compliance.md) | Plantillas de `init` para HIPAA/SOC2/EU-AI-Act/SR-11-7, procedencia (BOM/attest/repro-receipt), registro de auditoría, air-gap, autogeneración de model card (`soup card`), compuerta de CI (`soup ci init`) |
| [Backends, plataforma y operación](backends-and-ops.md) | Backends MLX/Unsloth, entrenamiento en GPU en la nube con Modal, hubs alternativos, integración con HF Hub, autopiloto, seguimiento de experimentos, plan/apply, lockfiles de entorno, ajuste a hardware, completions, plugins, comandos de utilidad |
| [Referencia de comandos](commands.md) | La lista completa de comandos de `soup` |
| [Referencia de la CLI](cli-reference.md) | Cada comando con cada opción y argumento: tipo, valor por defecto, si es obligatorio — generado desde el código |
| [Referencia de configuración](config-reference.md) | Cada clave de `soup.yaml`: tipo, valor por defecto, límites y valores permitidos — generado desde el esquema |
| [Modelos compatibles y extras](models.md) | Familias de modelos recomendadas, la guía de tamaño según VRAM, la matriz de extras de pip |
| [Dificultad de los issues](issue-difficulty.md) | Cómo se califican los issues de 1 a 10 y qué significa cada nivel |

> Las notas de cada versión están en la página de [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases);
> consulta también el [CHANGELOG.md](../CHANGELOG.md) en la raíz del repositorio.
