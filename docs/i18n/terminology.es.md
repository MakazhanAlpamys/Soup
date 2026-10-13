# Terminología español — repositorio Soup

Glosario **normativo** de la traducción al español de Soup. Si un término aparece aquí, se usa
exactamente así en todos los archivos. Añadir una fila antes de introducir una traducción nueva.

## Se quedan en inglés (nombres, API, formatos, unidades)

| Término | Motivo |
|---|---|
| `soup`, nombres de subcomandos y banderas | Es el producto y su CLI |
| modelos (`Llama`, `Qwen`, `Mistral`, `Gemma`, `Phi`), datasets y repos de Hugging Face | Nombres propios |
| siglas y métodos: `LoRA`, `QLoRA`, `DoRA`, `LoRA+`, `rsLoRA`, `VeRA`, `OLoRA`, `PiSSA`, `ReLoRA`, `NEFTune`, `GaLore`, `YaRN`, `LongLoRA`, `GaLore`, `SFT`, `DPO`, `GRPO`, `PPO`, `KTO`, `ORPO`, `SimPO`, `IPO`, `BCO`, `PRM`, `RAFT`, `RA-DIT`, `QAT`, `FP8`, `NF4`, `NVFP4`, `W4A4`, `MXFP4`, `MoE`, `KV-cache`, `FSDP`, `DeepSpeed`, `ZeRO`, `vLLM`, `MLX`, `Unsloth`, `PEFT`, `GGUF`, `GGML`, `AWQ`, `GPTQ`, `bitsandbytes`, `flash-attn`, `SDPA`, `Cut Cross-Entropy` | Nombres propios de métodos/formatos |
| `token`, `tokens`, `tokenizador`, `embedding`, `checkpoint`, `dataset`, `batch`, `prompt`, `prompts`, `fine-tune`/`fine-tuning`, `post-train`/`post-training`, `GPU`, `VRAM`, `CPU`, `RAM`, `TFLOPS`, `throughput`, `benchmark`, `epoch`, `step`, `gradient`, `loss`, `learning rate`, `reward`, `rollout`, `pipeline`, `runtime`, `kernel`, `plugin`, `hook`, `log`, `logs`, `server`, `deploy`, `host`, `cluster`, `node`, `driver`, `slot`, `scheduler`, `cache`, `wrapper`, `patch`, `diff`, `commit`, `fork`, `pull request`, `issue`, `CI` | Vocabulario técnico de uso corriente en inglés entre profesionales |
| `YAML`, `JSON`, `JSONL`, `TOML`, `CSV`, `Parquet`, `Markdown`, `Docker`, `CUDA`, `ROCm`, `MPS` | Formatos y plataformas |
| nombres de archivos, rutas, claves de configuración y variables de entorno | Se copian literalmente |

## Se traducen

| Inglés | Español |
|---|---|
| training / to train | entrenamiento / entrenar |
| to fine-tune (verbo conjugado) | ajustar (con `fine-tune` cuando se cita la función) |
| inference | inferencia |
| evaluation / eval | evaluación / evaluación |
| serving | servicio / servidor (según contexto) |
| export | exportación / exportar |
| merge | fusión / fusionar |
| quantization | cuantización |
| distillation | destilación |
| dataset card | ficha del dataset |
| recipe | receta |
| runbook | runbook (se queda) |
| overview | resumen / visión general |
| quick start | inicio rápido |
| getting started | primeros pasos |
| requirements | requisitos |
| troubleshooting | solución de problemas |
| caveats / gotchas | advertencias / trampas |
| usage | uso |
| workflow | flujo de trabajo |
| pipeline (de datos propio de Soup) | pipeline (se queda) |
| fallback | alternativa / respaldo |
| default | por defecto |
| deprecation | obsolescencia / obsoleto |
| breaking change | cambio incompatible |
| release notes | notas de la versión |
| changelog | registro de cambios |
| memory footprint | consumo de memoria |
| peak memory | memoria máxima |
| budget | presupuesto |
| throughput | rendimiento (sólo si el original no usa la palabra técnica) |
| end to end | de extremo a extremo |
| out of the box | sin configuración adicional |
| maintains / supports | admite / da soporte a |
| must | debe |
| should | debería |
| failure / to fail | fallo / fallar |
| warning | advertencia |
| tip | consejo |
| note | nota |
| example | ejemplo |
| step 1, step 2 | paso 1, paso 2 |

## Criterios de duda

1. Si el original lo usa como **nombre** (de comando, bandera, formato, archivo, modelo,
   servicio, método), se queda igual.
2. Si es **prosa descriptiva**, se traduce.
3. Si el original mezcla ambos (p. ej. "the LoRA adapter"), se traduce lo descriptivo y se
   conserva el nombre: "el adaptador LoRA".
4. Ante la duda, se conserva el inglés entre paréntesis la primera vez:
   "ajuste fino (fine-tuning)".
