# Rendimiento y cuantización

> 🇪🇸 Traducción al español de [`performance-and-quantization.md`](performance-and-quantization.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> Entrenamiento consciente de cuantización (FP8 y la ruta experimental QuEST), el menú de cuantización (I + II), KV-cache, NVFP4, formatos de guardado, Cut Cross-Entropy, gradient checkpointing, autocomposición de kernels, descarga de activaciones y multi-GPU / DeepSpeed / FSDP.

**Contenido:**

- [Decodificación experimental con CUDA graphs](#decodificación-experimental-con-cuda-graphs)
- [Entrenamiento consciente de cuantización (QAT)](#entrenamiento-consciente-de-cuantización-qat)
- [Ruta experimental mixta W4/A4+A16 de QuEST](#ruta-experimental-mixta-w4a4a16-de-quest)
- [Entrenamiento FP8 (Ada+)](#entrenamiento-fp8-ada)
- [Cut Cross-Entropy (modelos de vocabulario grande)](#cut-cross-entropy-modelos-de-vocabulario-grande)
- [Niveles de gradient checkpointing](#niveles-de-gradient-checkpointing)
- [Autocomposición de kernels](#autocomposición-de-kernels)
- [Enmascaramiento de atención entre documentos](#enmascaramiento-de-atención-entre-documentos)
- [Menú de cuantización — 9 formatos de cuantización](#menú-de-cuantización--9-formatos-de-cuantización)
- [Descarga de activaciones (VRAM pequeña, batch grande)](#descarga-de-activaciones-vram-pequeña-batch-grande)
- [Layer streaming (BETA, v0.72.0; NF4 v0.72.2; disco + arquitecturas más amplias v0.72.3; pérdidas de preferencia v0.72.4)](#layer-streaming-beta-v0720-nf4-v0722-disco--arquitecturas-más-amplias-v0723-pérdidas-de-preferencia-v0724)
- [La corrección primero (v0.36.0)](#la-corrección-primero-v0360)
- [Multi-GPU / DeepSpeed / FSDP](#multi-gpu--deepspeed--fsdp)
- [Rendimiento + contexto largo](#rendimiento--contexto-largo)
- [Sonda en vivo del tamaño de batch en CUDA](#sonda-en-vivo-del-tamaño-de-batch-en-cuda)
- [Consolidación de shards de FSDP](#consolidación-de-shards-de-fsdp)
- [Exportación BitNet de 1.58 bits](#exportación-bitnet-de-158-bits)
- [Cuantización de expertos MoE + entrenamiento sólo del router (activo en v0.71.20)](#cuantización-de-expertos-moe--entrenamiento-sólo-del-router-activo-en-v07120)
- [Escalera GGUF de Unsloth Dynamic 2.0 (v0.53.0)](#escalera-gguf-de-unsloth-dynamic-20-v0530)
- [Tipos de KV cache (v0.53.0)](#tipos-de-kv-cache-v0530)
- [Atención FP8 + NVFP4 + `unsloth_bnb_4bit` nativo](#atención-fp8--nvfp4--unsloth_bnb_4bit-nativo)
- [Paridad de cuantización de LF / Axolotl (v0.53.0)](#paridad-de-cuantización-de-lf--axolotl-v0530)
- [Formatos de guardado avanzados (v0.53.0)](#formatos-de-guardado-avanzados-v0530)
- [Menú de cuantización II + pipeline de exportación (v0.53.1)](#menú-de-cuantización-ii--pipeline-de-exportación-v0531)

---

## Decodificación experimental con CUDA graphs

`soup infer` y `soup bench infer` aceptan `--cuda-graphs`. Sobre un modelo Qwen2 o Llama residente, opcionalmente con un adaptador LoRA de `soup train`, la generación reproduce CUDA graphs en lugar de lanzar cada kernel desde Python. La ruta por defecto no cambia.

```bash
soup infer --model ./output --input prompts.jsonl --output answers.jsonl --cuda-graphs
soup bench infer ./model --backend transformers --max-tokens 128 --cuda-graphs
```

Medido en una RTX 5070 Laptop (Windows, sin afinidad fija, procesos nuevos, A-B-B-A-B-A) con Qwen2.5-1.5B-Instruct en FP16, 128 tokens greedy por solicitud: la ejecución más lenta con la bandera fue **2.9x** la más rápida sin ella (60.1 vs 20.7 tok/s), y **2.8x** con un LoRA sin fusionar (49.2 vs 17.8 tok/s), con IDs de token idénticos en cada solicitud y +42 MiB de VRAM reservada. `soup infer` sobre 24 prompts de longitud variada, incluidos la carga del modelo y la compilación, tardó 96-102 s con la bandera frente a 131-282 s sin ella. Sin la bandera la dispersión entre ejecuciones es grande (13.5-20.7 tok/s): la decodificación está limitada por el host, que lanza ~1,200 kernels por token, y Windows mueve ese hilo entre los núcleos rápidos y lentos de la CPU híbrida; con la bandera la dispersión es del 4%. El registro está en [`benchmarks/gate-v0.76.0-cuda-graph-decode.md`](../benchmarks/gate-v0.76.0-cuda-graph-decode.md).

- **El arranque en frío es más lento.** Antes de la primera fila, Soup ejecuta un calentamiento greedy sobre tu prompt más largo, que compila y captura el grafo (unos 8-10 s en esa tarjeta) y dimensiona la caché estática una sola vez para todo el batch. Una ejecución de un solo prompt puede tardar más en total. `Duration` y `Throughput` del panel de resumen empiezan después del calentamiento.
- **Las salidas son las mismas.** La bandera no cambia ninguna regla de precisión, muestreo o parada; las filas muestreadas usan la temperatura que pediste, y el calentamiento es greedy, así que no consume RNG de muestreo. Un adaptador LoRA permanece sin fusionar, exactamente igual que sin la bandera.
- **Compatible:** clases stock de Qwen2/Llama, una GPU CUDA, pesos sin cuantizar y como máximo un adaptador LoRA plano activo. Cualquier otra cosa (cuantización, offload a CPU/disco, layer streaming, DoRA u otras variantes de LoRA, varios adaptadores, flash attention, ASR) se rechaza antes de escribir ninguna salida, y cada rechazo termina con `Omit --cuda-graphs`.
- **Requiere PyTorch >= 2.14.** Usa API privadas del compilador de PyTorch validadas en 2.14.0 (corrige un defecto de contabilidad del backend `cudagraphs` stock que, si no, clona cada peso en el pool del grafo). Las versiones anteriores, o una API cambiada, se rechazan con un mensaje de una línea.
- **No está en `soup chat` ni en `soup serve`.** Un historial de chat crece en cada turno, y una caché estática que crece se recompila en cada turno.
- **Sin medir:** `--max-tokens` muy por encima de 128 (cada actualización K/V cuesta una copia extra por paso, que crece con prompt + max_tokens), Linux/macOS, tamaños de batch mayores que 1 y arquitecturas distintas de Qwen2. Sobre un batch de 1024 prompts, el árbol de grafos se vuelve a registrar aproximadamente una vez cada tres filas, sin ningún fallback eager ni crecimiento de memoria.

Si la generación falla con la bandera activada, Soup se detiene con `Generation failed with --cuda-graphs: ...`, conserva las filas ya escritas y nombra la salida. Ejecuta el mismo comando sin `--cuda-graphs`.

## Entrenamiento consciente de cuantización (QAT)

`training.quantization_aware` selecciona una de dos rutas de entrenamiento consciente de cuantización, cada una descrita en su propia sección:

- `quantization_aware: fp8`: [entrenamiento FP8](#entrenamiento-fp8-ada) en GPU Ada o posteriores.
- `quantization_aware: quest`: la [ruta experimental mixta W4/A4+A16 de QuEST](#ruta-experimental-mixta-w4a4a16-de-quest).

**`quantization_aware: true` (QAT int8) se rechaza al cargar la config, en toda tarea y backend ([#1222](https://github.com/MakazhanAlpamys/Soup/issues/1222)).** Nunca ejecutó entrenamiento consciente de cuantización:

- Donde se aplicaba (la configuración del modelo de transformers de los entrenadores de SFT, DPO, KTO, ORPO, IPO, BCO, SimPO, GRPO, PPO y preentrenamiento, que `task: tts` y `task: preference` también ejecutan), aplicaba la cuantización int8 post-entrenamiento sólo de pesos de torchao sobre cada capa Linear después de LoRA, incluidas las capas del propio adaptador, y las congelaba todas. Una ejecución de LoRA tenía cero parámetros entrenables, y cada ejecución probada (SFT, SFT con gradient checkpointing, ajuste completo, DPO) terminó con un error.
- En el resto no se aplicaba en absoluto. `backend: mlx` la ignoraba con una advertencia y entrenaba sin ella. `backend: unsloth` nunca la aplicaba: `soup train` rechazaba la combinación, y `soup sweep` y `soup bench train` entrenaban sin ella. Los entrenadores de las otras tareas nunca la leían, y tampoco lo hacían layer streaming ni la ruta de audio de SFT, así que esas ejecuciones también entrenaban sin ella.

Una config que la activa ahora se detiene al cargar, sea cual sea el grupo en el que estuviera, con un mensaje que nombra `training.quantization_aware`; elimina la clave o ponla en `false`. Las rutas `fp8` y `quest` no se ven afectadas, porque usan sus propias rutas de código. El QAT int8 real (cuantización falsa durante el entrenamiento, más una exportación que cuantiza la base del mismo modo) es una función aparte y todavía no está disponible.

Para entregar hoy un modelo más pequeño, usa la cuantización posterior al entrenamiento, la opción por defecto: entrena con normalidad y luego cuantiza al exportar, por ejemplo con `soup export --quant q4_k_m`.


## Ruta experimental mixta W4/A4+A16 de QuEST

`quantization_aware: quest` habilita la ruta experimental exacta conservada para [#674](https://github.com/MakazhanAlpamys/Soup/issues/674): los 168 pesos lineales del transformer usan W4 falso con grupo de 128; 161 activaciones usan A4 falso con grupo de 128; y los siete lineales de atención/MLP del bloque decodificador 23 conservan activaciones A16. Ambos operandos usan una transformada de Hadamard normalizada de ancho completo. Los recortes de activación se seleccionan entre cinco candidatos fijos sólo en las primeras 32 filas **de entrenamiento** tokenizadas.

```yaml
base: ahxt/LiteLlama-460M-1T
task: sft
backend: transformers
modality: text

data:
  train: ./data/train.jsonl

training:
  quantization_aware: quest
  quantization: none
  batch_size: 2
  lora:
    r: 0

output: ./output
```

Este primer fragmento falla de forma cerrada a menos que el modelo cargado tenga la topología Llama de 24 bloques medida, con exactamente esos 168 lineales, anchos de entrada compatibles en potencias de dos y pesos maestros FP32. **Esa compuerta es sólo topológica:** no se comprueba la identidad del modelo, así que cualquier Llama compatible de 24 bloques / 168 lineales recibe la ruta aunque la medición de calidad conservada usó únicamente `ahxt/LiteLlama-460M-1T`. También requiere una GPU CUDA Ampere o posterior visible. DDP, DataParallel, DeepSpeed, FSDP, layer streaming, LoRA, descarga de activaciones, NVFP4, otros backends/tareas/modalidades y la carga pre-cuantizada no se aceptan. En un host multi-GPU, expón una tarjeta al proceso, por ejemplo:

```bash
CUDA_VISIBLE_DEVICES=0 soup train --config soup.yaml
```

`use_cut_ce: true` sigue siendo compatible: Cut Cross-Entropy se parchea antes de cargar el modelo y no depende de la ruta posterior a la carga de v0.28 que QuEST omite.

El artefacto final y cada checkpoint periódico contienen `quest_mixed_precision.json`. El sidecar cerrado y versionado registra cada ruta A4 y A16, la escala de recorte, el digest de las filas de calibración, la transformada, la grilla, el sustituto (surrogate) y la procedencia de la ruta. La procedencia explica por qué se seleccionó esta topología; no hace ninguna afirmación sobre la calidad de entrenamiento del artefacto que la acompaña. La reanudación se rechaza si los metadatos de la ruta ejecutable difieren.
Para modelos de Hub, `base_model` sigue siendo el ID del repo. Para una base local, los nuevos metadatos de formato v2 registran una identidad `local-sha256:` derivada del `config.json`, del `generation_config.json` opcional, de los recursos estándar del tokenizador (incluidos `additional_chat_templates/*.jinja`) y del archivo de pesos que Transformers selecciona (o su índice y los shards referenciados). La identidad usa nombres de archivo relativos y contenido, así que mover la misma base mantiene válida la reanudación; cambiar un archivo seleccionado la rechaza. Soup comprueba la identidad antes y después de cargar el modelo. Una primera reanudación desde un checkpoint de formato v1 reutiliza la identidad que la ejecución ya calculó en lugar de leer la base por tercera vez (#1199).
Una base local cuyo `config.json` declare otro número de capas o un ancho incompatible se rechaza antes de calcular el hash de cualquier archivo; un valor ausente o que no sea un entero se deja a la compuerta posterior a la carga. El sidecar y `config.json["soup_quest"]` contienen la identidad, no el directorio de origen.
Los modelos o tokenizadores locales que usan código `auto_map` personalizado se rechazan en esta ruta de huella porque el código cargado desde otro lugar no quedaría cubierto. Los tokenizadores locales que declaran `fast_tokenizer_files` también se rechazan, porque esta huella no cubre sus archivos JSON de tokenizador versionados.

Para continuar el entrenamiento desde un artefacto QuEST local, define `base` como su directorio y mantén `training.quantization_aware: quest`. Soup restaura la tabla de recorte y la ruta del artefacto en lugar de recalibrar con las nuevas filas de entrenamiento. El sidecar debe coincidir con la declaración en `config.json`; los metadatos ausentes o inconsistentes se rechazan. Esto conserva el formato de metadatos y la procedencia heredados.

Los sidecars de formato v1 siguen siendo legibles. Como no registraban un hash de contenido, reanudar desde la base original sigue comparando su cadena de ruta; esa ruta escribe metadatos v2 para los checkpoints posteriores. Un artefacto v1 no puede demostrar que una base reubicada sea la misma, y los sidecars v1 existentes pueden revelar ya la ruta local antigua.
Como Transformers genérico no puede inferir la ejecución de cuantización falsa a partir de los pesos maestros, carga explícitamente la ruta ejecutable:

```python
from soup_cli.utils.quest import load_mixed_quest_artifact

model = load_mixed_quest_artifact("./output")
```

### Límite de la evidencia

Esta es una integración de ingeniería de un resultado **sólo de evaluación**, no una receta de entrenamiento validada. El [registro de la ruta mixta](../benchmarks/gate-674-quest-mixed-route.md) usó un modelo y ningún paso hacia atrás ni actualización del optimizador. Su ruta de siete A16 midió una brecha de 0.086344 nat/target frente a FP fijo, con un intervalo pareado del 95% de [0.080241, 0.093190], sobre 704 ejemplos / 25,017 targets. Por lo tanto:

- no es W4A4 puro;
- no es evidencia de la calidad de entrenamiento de la ruta mixta ni de generalidad entre modelos;
- no es paridad numérica con QuEST upstream;
- no es INT4 empaquetado, y no es una afirmación de velocidad ni de eficiencia de memoria.

El [registro de desarrollo posterior al ajuste](../benchmarks/quest-674-postfit-development-results.md) conserva una continuación de investigación seleccionada con una razón de NLL de respuesta Q/FP más fuerte de 1.038367 y una brecha pareada de +0.078519 nat/target sobre el mismo panel de 704 filas, ahora con la selección ya gastada. Usa un adaptador CE/KL de investigación alrededor del SFT nativo; este objetivo no se expone como receta pública de la CLI de QuEST. El intervalo pareado del 95% de la brecha sigue por encima de cero, así que `training_quality_validated` permanece en false. El registro también conserva los seguimientos fallidos y las vinculaciones de hash de checkpoints.

La implementación mantiene maestros FP32. La aritmética de la grilla de Hadamard y de cuantización falsa corre bajo el autocast de CUDA del entrenador: BF16 por defecto en las GPU Ampere o posteriores que esta ruta requiere. `training.auto_mixed_precision: true` se rechaza para QuEST porque puede seleccionar FP16, que la compuerta #674 no midió. La calibración de activaciones siempre corre bajo BF16. Trata esto como una ruta de investigación reproducible, no como un formato de despliegue más barato.

## Entrenamiento FP8 (Ada+)

En GPU Ada, Hopper y Blackwell (RTX serie 40/50, L4, L40S, RTX 6000 Ada, H100 / H200, B100 / B200), entrena con multiplicaciones de matrices float8 para obtener una aceleración de ~2x frente a bf16 con calidad comparable. Esto amplía la infraestructura de QAT mediante `torchao.float8`:

```bash
pip install "soup-cli[qat]"   # torchao (floor: TORCHAO_MIN_VERSION in utils/torchao_compat.py)
```

```yaml
training:
  quantization_aware: fp8   # ← string 'fp8', not bool true
  quantization: none        # FP8 converts linears directly; no bnb 4bit needed
```

### Recetas de escalado FP8 (v0.28.1)

Elige una receta de escalado para equilibrar velocidad y precisión:

```yaml
training:
  quantization_aware: fp8
  fp8_recipe: rowwise      # tensorwise | rowwise | rowwise_with_gw_hp
```

| Receta | Kernel | Escalado | Compromiso |
|---|---|---|---|
| `tensorwise` (predeterminada) | cuBLAS | Una escala por tensor | La más rápida, buena precisión |
| `rowwise` | CUTLASS | Escala por fila, e4m3, escalas potencia de 2 | Más lenta, más precisa |
| `rowwise_with_gw_hp` | CUTLASS | Por fila + grad_weight en alta precisión | La más lenta y precisa |

Si omites `fp8_recipe`, se usa `tensorwise` (idéntico al comportamiento de v0.28.0).

El booleano `true` no es otra forma de escribir FP8: seleccionaba la ruta int8, que se rechaza al cargar la configuración ([#1222](https://github.com/MakazhanAlpamys/Soup/issues/1222)). FP8 requiere CUDA y una GPU Ada o posterior (capacidad de cómputo ≥ 8.9), y se rechaza al cargar la configuración en los backends mlx y unsloth. `soup train --dry-run` informa la tarjeta de este equipo como nota y termina con código 0, ya que las configuraciones FP8 suelen prepararse para otra máquina; si falta torchao, igualmente falla. Las recetas `rowwise` y `rowwise_with_gw_hp` ejecutan un kernel torch aparte con sus propios límites: requiere una versión de torch que lo despache en la tarjeta (Ada 8.9: torch ≥ 2.7; Hopper 9.x y Blackwell de centro de datos 10.x: cualquier torch compatible; RTX serie 50 12.x: torch ≥ 2.8; 11.x: torch ≥ 2.10; ninguna versión hasta la 2.14 lo ejecuta en 13.x), **nunca se compila en Windows** y, antes de torch 2.11, solo se compila con CUDA 12 o posterior. `tensorwise` no tiene estas restricciones. Si se solicita FP8 y esta tarjeta, el sistema operativo o la compilación de torch no pueden ejecutarlo, **el entrenamiento se detiene durante la preparación** con el motivo (`FP8HardwareUnsupportedError`), antes de convertir ninguna capa, en todos los entrenadores que llegan al convertidor; nunca continúa sin FP8. **Si falta torchao, el entrenamiento se detiene del mismo modo** (`FP8DependencyMissingError`, que indica `pip install "soup-cli[qat]"`), incluso en una tarjeta compatible con FP8. Lo mismo ocurre si falla la conversión float8: muestra el error de torchao y se detiene en lugar de entrenar en bf16 (#1152). Antes de esta versión, los entrenadores mostraban una advertencia amarilla y entrenaban en bf16 aunque la configuración indicara FP8 (#835). `soup train` comprueba ambas condiciones —primero la tarjeta— antes de cargar el modelo, por lo que se detiene con el mismo motivo y no descarga nada. Las configuraciones de texto, visión y audio de SFT convierten el modelo después de conectar el adaptador LoRA. `stream_layers: true` rechaza `quantization_aware: fp8`, `fp8_attention` y `nvfp4` al cargar la configuración en todas las tareas; `backend: unsloth` también los rechaza: ninguna de esas rutas llama a los convertidores (#1124). `fp8_attention` aplica la misma comprobación (#835). Está conectado en todos los entrenadores del backend transformers (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).

## Cut Cross-Entropy (modelos con vocabulario grande)

Los modelos con vocabularios de 128k o más (Llama 3.1, Qwen2) materializan un tensor de logits `(batch, seq, vocab)` enorme que domina la VRAM. Cut Cross-Entropy calcula la pérdida por fragmentos:

```bash
pip install "soup-cli[cce]"    # or: pip install cut-cross-entropy
```

```yaml
training:
  use_cut_ce: true   # Patches the CE kernel before model load
```

La detección de arquitectura primero lee `config.model_type`, así que un directorio de checkpoint local (salida de `soup merge`/`soup shrink`/`soup draft distill`) recibe el mismo parche que un ID de Hub; si no se puede resolver la configuración, se usa como alternativa una coincidencia por nombre en el último componente de la ruta (`meta-llama/Llama-3.1-8B` → parche llama). Ahorra entre 8 y 24 GB de VRAM en formas habituales de batch × secuencia. No es compatible con unsloth (usa su propio kernel CE) ni con mlx. Está conectado en todos los entrenadores del backend transformers (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain); ten en cuenta que PPO tiene su propio bucle forward, así que allí cut_ce no hace nada de forma controlada.

## Niveles de Gradient Checkpointing

En lugar de un booleano, `gradient_checkpointing` ahora admite niveles que permiten equilibrar con más precisión el cómputo y la memoria:

```yaml
training:
  # One of: false | true | "selective" | "medium" | "full" | "auto"
  gradient_checkpointing: auto
```

- **`full`** / `true`: cada bloque transformer (~30% más lento, mayor ahorro).
- **`medium`**: cada dos bloques (equilibrado).
- **`selective`**: solo atención (~10% más lento, ahorro moderado).
- **`auto`**: elige según la VRAM detectada: < 24 GB → full, 24-80 GB → medium, > 80 GB → selective.

En SFT, `medium` usa la ruta nativa de Transformers `every_n_layers=2`, mientras que `selective` envuelve el módulo de atención directo de cada bloque decoder y deja desactivado el checkpointing del modelo completo de HF para evitar recomputación doble. Si una arquitectura no expone un hijo de atención directo, Soup informa y aplica una alternativa `full` en vez de afirmar que está activo un nivel selectivo que no funciona. Por ahora, los demás wrappers de tareas tratan cualquier nivel habilitado como checkpointing completo. Las configuraciones booleanas heredadas siguen funcionando sin cambios.

GRPO, IPO, BCO, reward-model, online-DPO, pretrain y embedding pasan explícitamente este ajuste a sus argumentos de entrenamiento de TRL/HF: `false` lo desactiva y `true` lo activa, incluso en ejecuciones sin cuantización (`quantization: none`). Estos wrappers usan checkpointing completo para cualquier nivel habilitado.

## Composición automática de kernels

`training.kernel_auto_compose: true` se rechaza al cargar la configuración. La implementación original medía el mismo modelo ya cargado una vez por candidato, así que no comparaba distintas configuraciones de kernel ni aplicaba el nombre que informaba.

En su lugar, elige explícitamente la optimización compatible:

```yaml
training:
  use_liger: true
  # or, on a compatible CUDA setup:
  use_flash_attn: true
```

En Apple Silicon, mantén desactivadas ambas opciones y usa `backend: mlx`; MLX administra sus propios kernels.

## Enmascaramiento de atención entre documentos

`training.packing_cross_doc_attn_mask` se rechaza al cargar la configuración. Antes, Soup establecía `packing_strategy="attention_free"` en TRL, que nunca ha sido una estrategia válida en ninguna versión publicada de trl (la lista permitida es `bfd` / `bfd-requeue` / `wrapped`); por eso la opción siempre provocaba un `TypeError` durante la preparación en vez de crear una máscara funcional.

El aislamiento de documentos empaquetados es la estrategia predeterminada `bfd` de TRL cuando FlashAttention es `attn_implementation`. Usa:

```yaml
training:
  packing: true
```

No configures `packing_cross_doc_attn_mask`.

## Menú de cuantización: 9 formatos

Elige el formato de cuantización adecuado para el modelo base y el hardware. Soup carga el `quantization_config` correspondiente y entrena LoRA encima:

```yaml
# Train LoRA on top of a pre-quantized GPTQ checkpoint:
base: TheBloke/Llama-2-7B-Chat-GPTQ
training:
  quantization: gptq        # or: awq, hqq:4bit, aqlm, eetq, mxfp4, fp8

# FSDP + QLoRA — Soup resolves this to the compute dtype automatically;
# pin it explicitly when the recipe targets known bf16 hardware:
training:
  quantization: 4bit
  bnb_4bit_quant_storage: bfloat16
```

| Formato | Bits | Caso de uso | Dependencia opcional |
|---|---|---|---|
| `4bit` | 4 | Predeterminado. Mejor opción general para entrenar LoRA. | bitsandbytes |
| `8bit` | 8 | Mayor presupuesto de memoria, gradientes más precisos. | bitsandbytes |
| `none` | 16/32 | Ajuste fino completo o DPO/PPO sin cuantización. | — |
| `gptq` | 2/3/4/8 | Entrenar LoRA sobre un checkpoint GPTQ existente. | gptqmodel |
| `awq` | 4 | Entrenar LoRA sobre un checkpoint AWQ existente. | autoawq |
| `hqq:Nbit` | 1, 2, 3, 4, 5, 6, 8 | Amplio rango de bits; se combina con LoRA. | hqq |
| `aqlm` | 2 | Compresión extrema. | aqlm |
| `eetq` | 8 | Kernel rápido de 8 bits para SM75+. | eetq |
| `mxfp4` | 4 (stored) | Entrena LoRA sobre un checkpoint pre-cuantizado MXFP4 (por ejemplo GPT-OSS). Se carga con `Mxfp4Config(dequantize=True)`, por lo que entrena en bf16: Transformers no entrena los pesos MXFP4 tal como se cargan. Presupuesta memoria para el modelo bf16, un estimado de 2 bytes por parámetro (aproximadamente 42 GB para una base 20B). La comprobación previa de VRAM de `soup train` compara una ejecución `mxfp4` con ese estimado, igual que una ejecución `quantization: none`, y `--allow-oom-attempt` inicia una ejecución que de otro modo se rechazaría. | — |
| `fp8` | — | Entrena LoRA sobre un checkpoint publicado en FP8. Se carga con `FineGrainedFP8Config(dequantize=True)`, por lo que entrena en bf16: presupuesta memoria para el modelo bf16, un estimado de 2 bytes por parámetro, no para el archivo FP8. | transformers ≥ 5.17.0 |

**Comprobación previa de VRAM.** Antes de cargar el modelo, `soup train` calcula el costo de los pesos del formato elegido y rechaza una ejecución que no quepa en la tarjeta (`--allow-oom-attempt` la inicia de todos modos). Las cifras son estimados en bytes por parámetro: `4bit` 0.5, `8bit` 1.0, `none` 2.0, `gptq` y `awq` 0.55, `aqlm` 0.5, `eetq` 1.0, y 2.0 para `mxfp4` y `fp8`, que se descuantizan al cargar. `hqq:*` todavía no tiene una cifra, así que la ejecución comienza y la comprobación previa muestra `Pre-flight skipped: no memory model for quantization: hqq:4bit` (con el formato que configuraste). Que no se rechace no significa que la ejecución quepa.

**Matriz de compatibilidad.** `soup train` ejecuta `check_quant_distributed_compat()` al inicio. HQQ / EETQ / AQLM fallan de forma explícita con FSDP y ZeRO-3 (según la matriz de LlamaFactory en `quantization.py:199/211`). BNB 4-bit + FSDP resuelve `bnb_4bit_quant_storage` al tipo de cómputo flotante efectivo antes de cargar el modelo y luego alinea los parámetros entrenables del adaptador con ese tipo antes de envolver el modelo en FSDP. Esto evita los dos modos de fallo de FSDP: almacenamiento entero y tipos mixtos entre adaptador y almacenamiento.

**Precuantización + QAT.** `gptq` / `awq` / `hqq:*` / `aqlm` / `eetq` / `mxfp4` / `fp8` ya incluyen su propia escala; combinarlos con `quantization_aware` (`'fp8'` o la ruta experimental `'quest'`) se rechaza al cargar la configuración. `true` también se rechaza por sí solo, sin importar el formato ([#1222](https://github.com/MakazhanAlpamys/Soup/issues/1222)).

**Compatibilidad entre entrenadores.** Quant Menu está conectado en los 12 entrenadores del backend transformers (SFT / DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO / RewardModel / Pretrain / Embedding / BCO). El modelo de recompensa de PPO también se carga con la misma configuración de Quant Menu que la política cuando se pasa `tcfg`, así que una ejecución con política GPTQ + recompensa GPTQ no provoca silenciosamente OOM en fp16. El backend MLX se rechaza con un mensaje de error distinto; las modalidades de visión y audio ahora usan el mismo cargador unificado de Quant Menu (la restricción `modality: text` se eliminó en v0.71.19), por lo que el menú completo —`gptq` / `awq` / `hqq:*` / `aqlm` / `eetq` / `mxfp4` / `fp8`— también se aplica a SFT multimodal (cada checkpoint de visión/audio sigue necesitando una clase y un kernel compatibles con el formato elegido, por ejemplo `autoawq` para awq).

**Tipo de datos de módulos no cuantizados (#339/#471/#492).** El argumento `torch_dtype` propio de `from_pretrained` —establecido en `"auto"` (o, en una tarjeta CUDA anterior a Ampere, una sobrescritura explícita `torch.float16`; consulta la nota «Load dtype» de la sección Full fine-tuning de `docs/training.md`— ahora también se aplica a una carga QLoRA `4bit`/`8bit` y determina el tipo de los módulos que `quantization_config` no cuantiza (`embed_tokens`, norms, `lm_head`). Esto coincide con `bnb_4bit_compute_dtype`, que ya resuelve el mismo `get_compute_dtype()` según la tarjeta, en vez de dejar esos módulos con el valor predeterminado sin especificar de `from_pretrained`.

## Offloading de activaciones (poca VRAM, batch grande)

Envía las activaciones guardadas a RAM o disco durante el paso backward para admitir tamaños de batch efectivos mayores en GPU pequeñas:

```yaml
training:
  activation_offloading: cpu    # or "disk"
```

`cpu` mueve los tensores guardados a RAM (rápido, limitado por la RAM del sistema); `disk` los escribe en un directorio temporal dentro del directorio de salida del entrenamiento (más lento, limitado por el espacio libre en disco). Las rutas temporales se comprueban para que estén contenidas en el directorio de trabajo actual, `torch.load(weights_only=True)` impide la deserialización arbitraria de Python al recargar y el administrador de contexto intenta limpiar los archivos temporales tanto al salir normalmente **como** después de un fallo.

No es compatible con unsloth (usa su propio administrador de memoria) ni con mlx. Está conectado en todos los entrenadores del backend transformers (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).

## Layer streaming (BETA, v0.72.0; NF4 v0.72.2; disco + arquitecturas más amplias v0.72.3; pérdidas de preferencia v0.72.4)

Transmite las capas decoder congeladas del modelo base **una por una** desde la RAM de CPU a buffers pequeños de VRAM, en vez de mantener residente toda la base. El pico de VRAM queda limitado al tamaño de una capa, no al modelo completo; así, ahora es posible entrenar modelos que no caben de forma residente en la GPU.

```yaml
training:
  stream_layers: true          # Enable layer streaming
  stream_source: auto          # 'auto' (same-host RAM), 'ram', 'disk' (v0.72.3)
  stream_ngram_source: auto    # Qwen4 PLE only: 'auto', 'ram', or read-only 'disk'
  stream_buffers: 2            # Double-buffering; range [2, 8]
  stream_read_ahead: 2         # Disk tier only; range [1, 8]. N stages N-1 layers ahead, and each level costs one layer of pinned host RAM
  # stream_pin: false          # Force the pinned RAM store off (escape hatch) or on; unset = automatic. See below
  # stream_vram_override: 4_000_000_000   # Bytes to assume free (v0.73.x); see below
  # stream_vram_probe: true    # Decide the fit by MEASURING one step (sft only); see below
  # stream_disk_kind: nvme     # Override auto disk-kind detection: nvme/ssd/hdd; see below
```

```bash
# Layer streaming is a CONFIG key, not a CLI flag — just train normally:
soup train --config soup.yaml
```

**Cómo funciona.** Los adaptadores LoRA, sus gradientes y el estado del optimizador permanecen en VRAM (son pequeños). La base congelada vive en la RAM de CPU, fijada en páginas cuando el equipo lo permite, y se transmite: cada capa decoder se copia a uno de dos buffers VRAM preasignados en un stream CUDA dedicado mientras todavía se calcula la capa anterior, de manera que la carga se superpone al cómputo. `embed_tokens`, cuyo tamaño depende del vocabulario, y un `lm_head` sin pesos compartidos utilizan un slot compartido adicional: se carga el embedding para la entrada del modelo; luego se reutiliza la misma asignación para la cabeza de salida. Su copia se inicia justo después de consultar el embedding (`StreamPrefetcher.head_prefetch_layer`, predeterminado `0`), no en la última capa decoder. Al final del backward de ese paso, el embedding se vuelve a cargar en el slot para que la copia se superponga al cómputo restante en vez de bloquear la primera consulta del paso siguiente. Cada capa decoder se lee **dos veces** por paso: una durante forward y otra cuando backward la vuelve a calcular, porque `dL/dx = Wᵀ · dL/dy` requiere que los pesos lleguen a las capas inferiores. Es una consecuencia física, no un detalle de implementación, y explica por qué streaming consume tiempo.

Para NF4 transmitido en CUDA, Soup sigue el despacho 4-bit propio de bitsandbytes. Si bitsandbytes selecciona su GEMM fusionado personalizado, Soup lo ejecuta mediante una Function autograd visible para checkpointing, de modo que los pesos empaquetados del pool no sobrevivan a la propiedad de su slot. Si bitsandbytes selecciona descuantización seguida de `F.linear` para formas de entrenamiento mayores, Soup conserva esa ruta y evita añadir una tercera descuantización a un paso con checkpointing. La rama fusionada reduce la VRAM de pesos guardados, pero su backward sigue descuantizando porque bitsandbytes no ofrece un GEMM 4-bit transpuesto; por tanto, es un compromiso acotado entre memoria y tiempo, no una afirmación de que se haya alcanzado el techo completo de kernels de #842.

**Qwen3.8-Flash-Next / PLE de Qwen4-Exp.** Para fines de almacenamiento, la tabla N-gram congelada de PLE no es un peso de capa decoder: guardarla en el shard de la capa PLE haría que el buffer de capa compartido tuviera el tamaño de toda la tabla. Soup deja intactos los tensores `ngram_embedding.shard_*` del checkpoint, contiguos por fila. `stream_ngram_source: disk` abre esos safetensors originales en modo solo lectura y obtiene únicamente las filas solicitadas por los tokens actuales; nunca reescribe ni copia la tabla a la caché de shards de Soup. En checkpoints densos de Transformers, `ram` carga las mismas partes en la RAM de CPU sin concatenar una segunda tabla completa, y `auto` selecciona RAM solo si la tabla medida más el nivel de base elegido caben en el margen de memoria del host.

Este recorrido limitado exclusivamente a texto admite bundles Qwen4 afines oMLX/oQ. Soup descuantiza cada capa decoder congelada una vez en la caché reutilizable de streaming y asigna los pesos expertos Switch-MLP fusionados al decoder de Transformers. La tabla PLE empaquetada, mucho mayor, permanece en el checkpoint original: solo se descuantizan las filas solicitadas, por lo que oQ requiere `stream_ngram_source: disk` (o `auto`). Se ignoran la torre de visión y el componente MTP porque el modelo instanciado es `AutoModelForCausalLM`, no el wrapper multimodal o de decodificación especulativa. La comprobación de paridad en CPU cubre filas exactas, logits, pérdida, gradientes LoRA, que no se modifique el archivo fuente y la limpieza de mapeos. La misma comprobación con checkpoint pequeño float32 pasa en M4 Max dentro de la tolerancia MPS publicada; CPU continúa siendo el oráculo exacto bit a bit. No se han medido el rendimiento ni la memoria máxima con checkpoints de producción. La comprobación inicial tiene un alcance deliberadamente reducido: `task: sft` y `quantization: none` (esta última es la transformación NF4 opcional de Soup, no la codificación oQ de origen admitida). La paridad de pérdidas de preferencia y NF4 en streaming sigue pendiente; esas combinaciones Qwen4 fallan antes de fragmentar. La paridad Qwen4 residente frente a streaming solo se ha ejecutado en float32; CUDA selecciona BF16, pero no se ha medido una comprobación de paridad CUDA BF16 y sigue pendiente la preparación para producción por esa ruta. El checkpoint oQ 176.9B de producción terminó de construir la caché y preparar el entrenamiento en M4 Max, pero se detuvo la prueba de un paso antes de completar uno del optimizador porque la carga desestabilizaba el host. No está validado como entrenable en una Mac de 128 GiB; consulta el [registro de la comprobación M4 Max](../benchmarks/gate-qwen4-ple-m4-max.md).

**Apple Silicon es experimental.** Con `backend: transformers`, MPS usa una fuente CPU pageable y buffers de capa MPS; el pinning del host se desactiva. PyTorch 2.7+ podría convertir de otro modo `torch.empty(device="cpu", pin_memory=True)` en un tensor MPS, cargando toda la base al asignador MPS aunque `is_pinned()` siga siendo false (#434). Soup rechaza ese estado en el límite de la fuente y también desactiva el pinning antes de asignar memoria. Apple Silicon tiene memoria física unificada, así que las cifras de capacidad y rendimiento CUDA de más abajo no aplican: solo los pesos decoder y de vocabulario transmitidos al asignador MPS están limitados por sus pools de buffers, mientras que la fuente CPU sigue consumiendo memoria unificada. En macOS 14+ el almacén y el tipo de cómputo son bfloat16 tras una prueba MPS en vivo con un elemento; un runtime anterior recurre explícitamente a float32. Un decoder de prueba float32 sin pesos compartidos coincide bit a bit con su control MPS residente en Apple Silicon, incluidas dos cargas de pesos de borde a través de un solo slot. Aún no se afirma que streaming permita entrenar un modelo más grande ni que sea más rápido que el entrenamiento MPS residente. `backend: mlx` sigue siendo una ruta de carga de modelo separada e incompatible y se rechaza junto con `stream_layers`.

El entrenamiento residente de Transformers comparte la misma prueba activa de capacidad BF16 de MPS. SFT, DPO, GRPO/RLVR, modelado de recompensas y PRM usan autocast BF16 en un runtime compatible; los demás entrenadores conservan la política cautelosa FP32 hasta contar con validación específica de Apple Silicon. GRPO usa generación local de Transformers en vez de vLLM en MPS. PRM conserva pesos maestros FP32 incluso con autocast BF16 porque una base entrenable BF16 actualmente provoca una incompatibilidad fatal de tipo de datos en el optimizador Metal.

El compromiso: **1.43× más lento que el entrenamiento residente**, medido en 0.5B, el único tamaño para el que hay una comparación equivalente en el equipo de referencia, ya que allí 1.5B y superiores no pueden ejecutarse de forma residente.

### Streaming NF4 (`quantization: 4bit`)

Cuantizar la base transmitida a NF4 reduce el almacén RAM aproximadamente **4×**. Esto importa por dos motivos; el segundo es el más importante:

1. Un modelo mayor puede caber en la RAM del host: una base 8B ocupa ~3.6 GB en NF4 en vez de ~16 GB en bf16.
2. **El almacén cabe por debajo del límite de memoria fijada en páginas del equipo.** La memoria host fijada permite que `copy_(non_blocking=True)` realmente se superponga al cómputo. El equipo de referencia alcanzaba ~7.1 GB de memoria fijada; una base bf16 3B de 5.55 GB recurría a memoria pageable y perdía superposición, mientras que el almacén NF4 de 1.43 GB sí se fija y el uso pasa de 79.3% a 100%. **Ese límite se debía en gran parte a la contabilidad de Soup, no al equipo** (#901): el asignador host con caché de torch redondea cada solicitud fijada *a la siguiente potencia de dos*, y antes el almacén se fijaba tensor por tensor, así que el costo en memoria fijada era 1.7–1.9× su tamaño (medición: 6.82 GB de tensores decoder NF4 Qwen2.5-14B ocupaban 11.83 GB). Ahora el almacén se fija en unos pocos arenas de potencia de dos y cada tensor es una vista; el almacén NF4 14B de 9.93 GB fija exactamente 10 GiB. La línea de listo muestra la cifra real (`9.93 GB pinned RAM store (10.74 GB page-locked)`). Desde #1702, las arenas se llenan empezando por el tensor más grande cuando así se fijan menos bytes: un almacén NF4 Qwen3-8B de 6.07 GB, cuyas dos matrices de vocabulario de 1.24 GB abrían antes cada una su propio arena de 2 GiB detrás del decoder, ahora fija 6 GiB en vez de 8 GiB. Si no hay ganancia, el almacén conserva exactamente la disposición anterior.

La base se cuantiza **una sola vez, sin conexión**, un tensor a la vez, y se guarda en caché. La clave de la caché de shards incluye cuantización, tipo de datos, dispositivo de cuantización y una huella del checkpoint de origen; así, cambiar entre `none` y `4bit` —o volver a entrenar una base en el mismo lugar— vuelve a fragmentar en vez de transmitir silenciosamente bytes incorrectos.

La corrección tampoco implica un compromiso: una ejecución NF4 en streaming es **idéntica bit a bit** a una ejecución NF4 *residente* (los mismos bytes cuantizados pasan por los mismos kernels de bitsandbytes); esto es una prueba de regresión, no una medición aislada.

**Alcance de esa comparación.** El control residente es un modelo NF4 simple, *sin* `prepare_model_for_kbit_training` de PEFT. La ruta residente predeterminada de SFT 4-bit/8-bit lo llama (`trainer/sft.py`) y convierte a float32 todos los parámetros no cuantizados: `embed_tokens`, las normas y un `lm_head` sin pesos compartidos (verificado en `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`: todos en bfloat16 antes y todos en float32 después). El streaming conserva esos tensores en el tipo de datos del almacenamiento (bf16 en Ampere y posteriores, fp16 antes), por lo que una ejecución residente predeterminada y una en streaming parten de configuraciones numéricas distintas aunque los bloques cuantizados sean idénticos. Un caso informado por un contribuidor (`Qwen/Qwen3-8B` en `b968826d`, RTX 4070 SUPER, bitsandbytes 0.50.2, transformers 5.17.0, peft 0.21.1, torch 2.14.0+cu126; no forma parte del registro de benchmarks), con Qwen3-8B NF4, un texto de 73 tokens y un forward en el paso 0: el control residente sin conversión ascendente coincide bit a bit con streaming, mientras que la ruta residente predeterminada difiere (máx. |Δlogit| 0.52, media 0.054). Es un forward de un texto: no dice nada sobre backward, reanudación ni la trayectoria de entrenamiento completa; además, por construcción, la inicialización de `A` de LoRA también difiere (streaming materializa los adaptadores desde un generador CPU con semilla, `materialize_meta_adapters`; la ruta residente usa la inicialización propia de PEFT desde el RNG global).

**Cifras medidas (RTX 3050 Laptop 4 GB, Windows 11, LoRA, batch 1, 50 pasos después de 10 de calentamiento):**

| Modelo | Cuant. | Sec. | Rendimiento | Uso de GPU | Pico de VRAM | Almacén RAM |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **NF4** | 512 | **119.6 tok/s (pre-reparación, [#361](https://github.com/MakazhanAlpamys/Soup/issues/361))** | 100% | **3.32 GB** | 3.60 GB pinned |
| Qwen2.5-3B | NF4 | 512 | 264.2 tok/s | 100% | 1.76 GB | 1.43 GB pinned |
| Qwen2.5-3B | bf16 | 512 | 143.1 tok/s | 79.3% | 2.15 GB | 5.55 GB pageable |
| Qwen2.5-1.5B | bf16 | 512 | 525.0 tok/s | 96.8% | 1.82 GB | pinned |
| Qwen2.5-1.5B | bf16 | 1024 | 487.6 tok/s | 96.7% | 2.96 GB | pinned |
| Qwen2.5-0.5B | bf16 | 512 | 978.6 tok/s | 91.4% | 1.47 GB | pinned |

**Dato principal:** **Llama-3.1-8B se ajusta en una tarjeta de 4 GB a 119.6 tok/s con 3.32 GB (pre-reparación; medición repetida pendiente en [#361](https://github.com/MakazhanAlpamys/Soup/issues/361)).** Como referencia, 1M de tokens de entrenamiento son ~2.3 h a 8B (cálculo a partir del ritmo medido, no una medición independiente).

Las filas NF4 frente a bf16 de 3B difieren por 1.85×, pero atribuye esto al **pinning, no a la aritmética** (consulta el punto 2 anterior). Además, las dos filas se midieron en sesiones distintas y el reloj boost de esta tarjeta varía ~13% entre sesiones; considera indicativo el factor y toma el mecanismo como la afirmación principal.

La fila de 8B y 3.32 GB anterior es anterior al streaming de capas grandes: `embed_tokens` y `lm_head`, sin pesos compartidos ni cuantizar, permanecían residentes y ocupaban 2.10 GB. El código actual los guarda como shards separados de capa grande y reutiliza un slot de dispositivo del tamaño de la matriz mayor; por tanto, un par sin pesos compartidos de la misma forma debería recuperar el espacio de una matriz, mientras que un modelo con pesos compartidos mantiene el requisito de una matriz. En `dpo` y `kto` ese ahorro vuelve a ocuparse: el forward de la cabeza de salida conserva una copia privada de la matriz para su backward en cada paso (aproximadamente 1.05 GB para Llama-3.1-8B en bf16, contabilizados por la comprobación previa, #1049). CI en CPU fija logits idénticos bit a bit para ambos controles. Aún falta medir el nuevo pico CUDA en la RTX 3050 de referencia; la cifra histórica de 3.32 GB no se presenta como una medición nueva.

**Alcance con transparencia:**
- **Nivel RAM + desbordamiento a disco (v0.72.3).** `stream_source: auto` selecciona RAM si el almacén cabe en el margen dinámico de RAM libre y el límite de memoria física del host; si no, recurre a disco NVMe. SATA/HDD se rechaza. Corrección verificada. **La lectura no ocurre en el hilo de cómputo**: un lector en segundo plano interpreta por sí mismo el encabezado de cada shard (sin mapeo de memoria) y prepara en RAM de host `training.stream_read_ahead` capas (con memoria fijada cuando el equipo lo permite; consulta `stream_pin` más adelante), para que la GPU reciba datos mientras llega la capa siguiente. **Medición en frío y en caliente frente a un control del mismo día de la fuente que reemplaza** (RTX 5070 Laptop, 2026-09-14, [registro](../benchmarks/gate-971-async-nvme-source.md)).
  - **En frío —un almacén mayor que la RAM, que es el objetivo de este nivel— es 2.1–3.1x más rápido.** Con un almacén NF4 de forma 70B y 36 GB, batch 1 x secuencia 512, un paso pasó de 92–100 s a **30–48 s**, con cada par en la misma posición (5.1–5.6 → **10.6–16.9 tok/s**; 0.70–0.76 → **1.46–2.32 GB/s** en la fuente), frente a 124.5 s / 4.1 tok/s / 0.57 GB/s de la ruta síncrona anterior, medida el 2026-09-12 ([registro anterior](../benchmarks/probe-rtx5070-what-bounds-streaming.md)). El rango corresponde a la posición en la ejecución, no a la profundidad (consulta lo siguiente). El pico de VRAM sigue en 4.38 GB.
  - **Desde #974, el lector evita la caché de páginas.** La sección de datos de cada capa se lee como rangos de bytes alineados a sectores mediante E/S directa (`FILE_FLAG_NO_BUFFERING` en Windows, `O_DIRECT` en Linux, `F_NOCACHE` en macOS; se usa `open` con búfer si el sistema de archivos lo rechaza, por ejemplo tmpfs; el registro de comprobación previa indica cuál), cuatro rangos en paralelo, hacia **una región de preparación por slot** ubicada en los mismos arenas fijados de potencia de dos que el almacén RAM. El motivo surge de una medición, no de una preferencia: en ese mismo almacén de 36 GB en frío, todas las primitivas de lectura *con búfer* —`readinto` por tensor (la ruta reemplazada), un `readinto` por rango de bytes, cualquier cantidad de hilos— alcanzaron como máximo **1.2–2.9 GB/s** en el NVMe del equipo de desarrollo, mientras que las lecturas sin búfer de las mismas capas recién creadas llegaron a **3.5–5.65 GB/s**, con mejores resultados entre 2 y 4 rangos ([registro](../benchmarks/gate-974-disk-tier-direct-io.md)). Los 1.46–2.32 GB/s anteriores eran ese límite de lectura con búfer. Hay dos consecuencias importantes: la preparación ya no paga el redondeo por tensor a potencia de dos (mecanismo de #901, medido en 1.7–1.9×: los cuatro slots del almacén 70B fijan un único arena de 2 GiB para 1.96 GB; la línea de listo también muestra la cifra en este nivel), y una ejecución en nivel disco ya no calienta la caché de páginas para lo que se ejecute después. Con el mismo protocolo en frío del punto anterior (almacén NF4 con forma 70B de 36 GB, batch 1 x secuencia 512), el paso ahora tarda **16.2–17.3 s en ambas posiciones**: 6.4x la ruta síncrona medida en el mismo lugar (110.5 s), 2.0–2.8x más rápido que el lector reemplazado (34.8–45.6 s); la lectura ocupa 62–64% del paso a 6.0–7.0 GB/s. Las dos posiciones del nuevo lector difieren 1.07x, frente a 1.31x en el anterior, porque al evitar la caché ofrece el mismo resultado dondequiera que se ejecute ([registro](../benchmarks/gate-974-disk-tier-direct-io.md) §8).
  - **En caliente —todo el almacén en la caché de páginas— se informó como una REGRESIÓN de 1.03–1.20x** frente a la fuente síncrona (gate-971 §5). Al comparar la misma posición en ambos órdenes, queda en **0.91–0.97x**, es decir, no es más rápido: el intervalo se debía al orden de ejecución que el registro ya había señalado para el conjunto frío (#974). Tras el cambio, en una sesión frente al lector que reemplaza, el nuevo lector alcanza 0.76x del anterior en caliente (1.58 frente a 2.09 s) y es 2.25x más rápido justo después de desalojar la caché de páginas (1.79 frente a 4.02 s). **Si el almacén cabe en RAM, usa el nivel RAM**: 1.1 s frente a 1.8–2.0 s para cualquiera de las rutas de disco en la misma sesión; el nivel disco existe para un almacén que no cabe.
  - **Sigue limitado por la lectura, aunque menos desde #974**: en gate-971 la lectura por capa representaba 82.6–84.9% del paso en frío; con E/S directa representa 62–64%. No existe una base medida sin costo de lectura para esta secuencia con la cual compararlo (la cifra ~14 s es para una secuencia más corta), así que no se informa una razón de margen disponible.
  - **`stream_read_ahead` no cambió el rendimiento** en ese registro. Las profundidades 1, 2 y 4 no se distinguieron una vez controlado el orden de ejecución: la misma configuración midió 48.09 s al principio y 30.28 s al final, al calentarse la caché de páginas entre bloques; esa diferencia es mayor que toda la variación entre profundidades. Considéralo un **ajuste de memoria del host**: 1.5 GB de preparación fijada con profundidad 1, 2.0 GB con 2, 2.8 GB con 4 en un 70B. La comprobación previa muestra esa cifra en el nivel disco (`host staging read_ahead N -> X MB`) y **rechaza la ejecución** si no cabe junto con los extras residentes en el margen de RAM libre, indicando `stream_read_ahead` como el ajuste que debe reducirse. El embedding y un `lm_head` sin pesos compartidos ocupan un slot cada uno con cualquier profundidad, ya que cada cual constituye una capa.
  - **Detección del tipo de disco.** Un disco paravirtualizado (virtio) informa `rotational=1` sin indicación del medio, por lo que un disco de nube realmente respaldado por NVMe se identificaba erróneamente como HDD y se rechazaba (#365); ahora la detección mide una lectura secuencial O_DIRECT acotada cuando el indicador de rotación no es fiable y acepta rendimiento de clase NVMe (>= 1 GB/s), mientras que un disco realmente lento sigue rechazado. Configura `training.stream_disk_kind: nvme` (o `ssd`/`hdd`) para sobrescribirlo si la detección sigue equivocada; se muestra el valor resuelto junto al detectado.
- **Detección de discos APFS de Apple.** En macOS, un volumen APFS puede informar `Apple Fabric` aunque su almacén físico sea el NVMe interno de Apple. Soup resuelve el volumen de destino a su almacén físico APFS y lo acepta solo si ese dispositivo exacto aparece en `SPNVMeDataType`; un dispositivo de estado sólido que no coincide se clasifica como `ssd`, y el hardware desconocido sigue rechazándose. Si se configura explícitamente, `training.stream_disk_kind` mantiene la autoridad final.
- **Llama / Qwen / Qwen3.5 denso y MoE texto / Qwen4-Exp texto / Mistral / Gemma / Gemma2 / Gemma3-Text / Phi / Phi3** (`qwen3_5`, `qwen3_5_text`, `qwen3_5_moe` y `qwen3_5_moe_text` usan el streamer de qwen3; `qwen4_exp_text` usa el streamer de Qwen4-Exp), `task: sft`, `backend: transformers`, `modality: text`. La lista original está verificada bit a bit en bf16 y NF4. Las rutas decoder heterogéneas dense y MoE de Qwen3.5 están verificadas bit a bit frente a controles residentes en CPU; la ruta MoE también tiene validación de entrenamiento en streaming sobre `Qwen/Qwen3.5-35B-A3B`, cuya ejecución real 35B no cuenta con control residente porque el hardware disponible no pudo cargarlo de forma residente. Qwen4-Exp tiene actualmente una prueba de paridad exacta con un modelo diminuto float32, incluida su tabla externa PLE; siguen pendientes la validación con checkpoints reales y NF4.
- **Las claves heterogéneas de capa solo se permiten respecto a presencia o ausencia.** El sharder lee el encabezado safetensors de cada capa y el runtime crea la fuente RAM/disco a partir de esas especificaciones por capa; después las combina en un único pool de buffers VRAM. Una clave presente en varias capas debe conservar la misma forma y tipo de almacenamiento en todas. Los pesos NF4 también mantienen un `NF4WeightSpec` por clave corta, validan cada sidecar empaquetado con el encabezado del shard y comparten solo las tablas de códigos pequeñas después de comprobar que son iguales.
- **Los tamaños de batch, la acumulación de gradientes, `--resume` / `--hf-resume`** ya funcionan (v0.72.3).
- **Las tarjetas anteriores a Ampere (T4, P100, V100, GTX 16xx, RTX 20xx) ahora usan fp16 en streaming en vez de bf16.** Antes de esta corrección, el tipo de almacenamiento estaba fijado en bf16 para todos los dispositivos CUDA: toda la oferta gratuita de notebooks transmitía un tipo sin unidades de hardware compatibles, sin indicarlo, y no podía fallar en la tarjeta Ampere donde se midieron todas las cifras anteriores. fp16 coincide bit a bit con una referencia residente de la misma configuración numérica: `0.000000e+00` en ambas cuantizaciones, igual que bf16.
  **La capacidad se consulta mediante `torch.cuda.is_bf16_supported(including_emulation=False)`, y el argumento es esencial.** La llamada sin argumentos incluye emulación por defecto: si falla la ruta rápida de capacidad de cómputo, intenta crear un tensor bf16, que la emulación por software satisface; por eso **una T4 responde True**. La primera versión de esta corrección hacía la consulta sin argumentos y no cambiaba nada en el hardware al que se dirigía; se descubrió al ejecutar el [notebook de prueba](../notebooks/proof-4gb.ipynb) en una T4 real, no mediante razonamiento. `get_compute_dtype` duplicaba la misma consulta y ahora delega en el mismo helper.
  **Aún no se ha medido en una tarjeta anterior a Ampere**: la exactitud de fp16 indicada se midió *usando* fp16 en Ampere, así que valida el cableado, no los kernels de Turing/Pascal; en particular, bitsandbytes NF4 en sm_75.
- **Los adaptadores LoRA se convierten a fp32 cuando el streaming entrena en fp16.** peft crea los pesos del adaptador con el tipo de datos del checkpoint base (bf16 para Llama-3.1); en una tarjeta anterior a Ampere ese tipo no tiene unidades bf16 y GradScaler fp16 produce `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'` (#425). `align_trainable_dtype_for_fp16` convierte a fp32 los parámetros entrenables `*lora_*` antes de crear el optimizador, y todos los wrappers de entrenadores lo llaman; esto lo exige una prueba de escaneo, no una lista escrita a mano.
- **Una ejecución 8B en streaming ya termina en una tarjeta Turing —Colab gratuito, Tesla T4 (sm_75)— y eso es todo lo que demuestra.** `NousResearch/Meta-Llama-3.1-8B-Instruct`, NF4, `stream_buffers: 2`, batch 1, `max_length: 256`, LoRA r=8, fp16: 7 pasos; el adaptador guardado tenía 128 de 128 tensores distintos de cero en ambas ejecuciones (la segunda se ejecutó en el proceso, según el notebook, así que no hay código de salida que informar). La primera ejecución midió un pico de **2.91 GB** frente a ~3.02 GB previstos (la comprobación previa se excede en 3.8%, en la dirección segura para la que se calibró), con **VRAM libre 15.10 GB** según la comprobación: la cifra del dispositivo, no el límite por proceso. La T4 tiene 15.6 GB, así que el proceso se limitó a **4.00 GB** mediante `torch.cuda.set_per_process_memory_fraction`, y se comprobó que el límite actuaba: se rechazó una asignación de 4.29 GiB. El [notebook](../notebooks/proof-4gb.ipynb) guardado ahora establece `training.stream_vram_override` en esos mismos 4.00 GB, reemplazando la referencia para decidir si cabe; después la comprobación informa **VRAM libre 4.00 GB** en vez de toda la tarjeta, una barrera más estricta que la de la primera ejecución. Frente a ella: ~1.97 GB previstos, **pico medido 1.83 GB** (sobreestimación de +7.7%, todavía en dirección segura). La reducción de 2.91 GB es coherente, no contradictoria: entre las ejecuciones, los embeddings y `lm_head` pasaron de la asignación residente a un slot de capa grande en streaming. **No se cita rendimiento de ninguna de las dos ejecuciones**: una tarjeta bajo un límite artificial no es un benchmark y el notebook tampoco informa ninguno. **Lo que ninguna de las dos demuestra**: exactitud del backward/gradiente en 8B sobre Turing (un adaptador no nulo demuestra que hubo gradientes, no que fueran correctos). La comparación streaming frente a residente del notebook ahora está registrada y no pendiente: idéntica bit a bit, máx. `|streamed - resident|` = 0.0, `torch.equal` = True; pero con `SmolLM2-135M`, fp16, sin cuantizar, no con la configuración 8B NF4 de este punto. Registro: [`benchmarks/run-t4-colab-free-tier.md`](../benchmarks/run-t4-colab-free-tier.md).


- **El rendimiento bf16 de 3B indicado arriba es un LÍMITE INFERIOR.** El equipo de referencia no pudo fijar en memoria los 5.55 GB de la base (el límite medido de memoria fijada era 7.65 GB; #901 determinó después que el pinning por tensor se redondeaba a potencias de dos, así que ese almacén en realidad solicitaba ~10 GB; hoy sí se fijaría). Por eso la ejecución recurrió a un almacén pageable. La memoria pageable hace síncrona la copia del host al dispositivo, lo que reduce la superposición; se ve en la caída del uso de GPU de 96.8% (1.5B, fijado) a 79.3% (3B, pageable). Soup aplica esta alternativa automáticamente **y muestra su costo** en lugar de ocultarlo. NF4 resuelve esto en 3B: el almacén queda por debajo del límite y se fija.
- Las cifras son de Windows/WDDM y, por tanto, sistemáticamente pesimistas frente a Linux. Windows ignora silenciosamente `expandable_segments:True`; Soup lo detecta y no afirma que esté activo.

### Forzar el pinning (`training.stream_pin`)

El pinning se selecciona automáticamente; `stream_pin` permite que la configuración lo sobrescriba. **Desde
#971, el indicador cubre ambos niveles**: en el nivel RAM decide si se fija el almacén base; en el nivel disco, si se fija la *preparación* del host del lector asíncrono.

- **Sin configurar (predeterminado)** intenta usar memoria de host fijada en un destino CUDA y **recurre a otra opción con un aviso claro** si el equipo no puede proporcionarla: el almacén base en el nivel RAM, la preparación del lector en el nivel disco. El aviso indica el costo: las copias host→dispositivo pasan a ser síncronas y el uso medido de GPU cae de **~97% a ~79%**. En el nivel disco también indica `stream_read_ahead`, porque la profundidad determina cuánto se fija. Desde #974 esa preparación se empaqueta en los mismos arenas de potencia de dos que el almacén RAM (una región por slot, tensores como vistas), así que se fija una cantidad cercana a su propio tamaño en vez de 1.7–1.9x por tensor; la línea de listo también muestra esa cifra en este nivel.
- **`stream_pin: false`** fuerza memoria host pageable en ambos niveles: el almacén base en RAM y la preparación del lector en disco. La comprobación previa indica el rendimiento que cuesta en vez de ocultarlo: hasta **6.56×** medido (Qwen2.5-32B NF4), **7.41×** en un caso sintético. Esas dos cifras se midieron en el **almacén RAM** ([registro](../benchmarks/gate-h100-validation.md)); el mismo mecanismo se aplica a la preparación del lector, pero su magnitud allí no se ha medido. También es la salida de emergencia: fue la única mitigación conocida mientras #331 seguía abierto.
- **`stream_pin: true`** fuerza memoria de host fijada y **RECHAZA la ejecución en un destino CUDA** si el equipo no puede proporcionarla, en lugar de degradar a pageable y consumir todo el margen que el pinning debía aportar. El rechazo indica el tamaño del almacén en el nivel RAM y `stream_read_ahead` en el nivel disco.

**Cuándo `true` avisa en vez de rechazar.** Fijar páginas requiere un dispositivo CUDA al cual copiar. En un destino que no sea CUDA no hay nada que forzar, por lo que la solicitud es **inaplicable, no imposible**, y la ejecución *continúa con un aviso*:

| Nivel / dispositivo | Qué hace `stream_pin: true` |
|---|---|
| Nivel RAM en CUDA | fija el almacén base o **rechaza** e indica su tamaño |
| Nivel disco en CUDA (la base no cabe en RAM, los pesos llegan desde NVMe) | fija la preparación del lector o **rechaza** e indica `stream_read_ahead`, la profundidad que determina cuánto queda fijado |
| Destino distinto de CUDA (CPU o MPS) | avisa que el pinning de memoria host CUDA no aplica y continúa con una fuente CPU pageable |

Rechazar un destino que no sea CUDA haría imposible compartir `stream_pin: true` en un `soup.yaml` entre una máquina GPU y otra que no sea CUDA. **La fila del nivel disco cambió en #971**: antes avisaba que el pinning no aplicaba y seguía, porque ese nivel no tenía nada que fijar. Ahora el lector asíncrono prepara capas completas en RAM del host, así que el ajuste tiene efecto real allí; un `soup.yaml` con `stream_pin: true` que antes solo advertía en el nivel disco ahora puede provocar un rechazo.

Si se establece mientras `stream_layers: false`, la clave se rechaza como configuración peligrosa, igual que las demás claves `stream_*`.

### Dimensionar una ejecución en streaming (v0.72.3)

El streaming limita los **pesos**. No afecta las activaciones ni el tensor de logits, y ambos crecen con `batch × seq`. En modelos con vocabulario grande, este segundo término domina todo lo demás: medido en Qwen2.5-0.5B (vocab 151 936), batch 8, S=512, solo los logits ocupan **8.71 GB —146× todo el pool de buffers de capas (0.060 GB)**. Una comprobación previa que presupuestara únicamente pesos y buffers dejaría pasar esa configuración.

El mismo término determina lo que necesita una base 8B en una tarjeta de 4 GB. Llama-3.1-8B NF4 (vocab 128 256), en streaming, LoRA r=8: **3.448 GB con batch 1, S=512**, y luego **1.063 GB adicionales por cada bloque de 512 tokens en cada paso**: 4.512 GB con 1024, ya por encima de la tarjeta de 4.294 GB ([prueba v0.73.0, sección 9](../benchmarks/probe-v0.73.0-what-bounds-streaming.md#9-peak-vram-and-the-microbatch-sweep)). `data.max_length` tiene 2048 por defecto, así que la configuración «8B en 4 GB» necesita `data.max_length: 512`.

Por eso `soup train` predice el pico de VRAM antes de construir el modelo y **rechaza una ejecución que prevé que no cabrá**:

```
peak VRAM    ~0.48 GB at batch 2 x seq 256 (logits 0.35 GB)
free VRAM    3.46 GB
forecast     5685-8361 tok/s — a compute-bound bound, not a promise
             (from 6.75 TFLOPS measured on this card now using bfloat16 @ 862 MHz)
```

La predicción se ajustó con diez ejecuciones reales en dos modelos, una diferencia de vocabulario de 3.1×, batch 1–8 y dos longitudes de secuencia: **error máximo de 0.85%, sin subestimar nunca** —la única dirección segura para una cifra que puede detener una ejecución. El rechazo indica los dos ajustes que realmente la escalan (`training.batch_size`, `data.max_length`).

El rechazo en vez de una advertencia es deliberado. En Linux, un paso que excede el presupuesto produce un OOM duro. En Windows es peor: WDDM vuelca datos silenciosamente a la memoria del host y la ejecución solo se vuelve un orden de magnitud más lenta; aquí se midió un pico de 9.27 GB en una tarjeta de 4.29 GB **sin lanzar excepción alguna**. Interpretarlo como «el streaming es lento» sería exactamente la conclusión equivocada.

La línea de rendimiento es un **límite, no una promesa**. Proviene de un benchmark GEMM con el tipo de streaming resuelto para la tarjeta durante esa sesión: `bfloat16` en tarjetas con soporte BF16 nativo y `float16` en las demás. Se muestra junto con el reloj SM al que se midió porque esta misma tarjeta produjo 3.5 y 7.6 TFLOPS en dos sesiones con el mismo reloj informado. Incluir en Soup una constante por tarjeta sería inventar una cifra. Las ejecuciones reales en streaming alcanzaron 68–100% del límite medido.

### Tamaño de batch frente a acumulación de gradientes

Ambos funcionan desde v0.72.3 y no son intercambiables. Medición en Qwen2.5-0.5B bf16, S=256, almacén fijado, 50 pasos después de 10 de calentamiento:

| batch | accum | batch efectivo | rendimiento | pico VRAM |
|---|---|---|---|---|
| 1 | 1 | 1 | 556.6 tok/s | 0.842 GB |
| 1 | 4 | 4 | 540.1 tok/s | 0.846 GB |
| 4 | 1 | 4 | **1378.0 tok/s** | 2.28 GB |

La acumulación es **neutra respecto a E/S por token**: las lecturas de capas por cada 1000 tokens se mantuvieron constantes con accum 1, 2 y 4, porque `accum=N` vuelve a leer la base N veces *y* procesa N veces más tokens. Su costo es de oportunidad: con el **mismo batch efectivo de 4**, aumentar `batch_size` fue **2.52× más rápido**, porque una lectura de pesos se amortiza sobre cuatro veces más tokens.

La acumulación permite obtener un batch efectivo mayor con **VRAM constante** (0.842 → 0.846 GB al pasar accum de 1 a 4; aumentar batch, en cambio, costó 0.842 → 2.28 GB). Regla práctica: **aumenta `batch_size` hasta que la comprobación previa de VRAM lo rechace; luego acumula para el resto**. Soup muestra esta recomendación cuando detecta acumulación.

**Rechazado al cargar la configuración (cada punto indica la versión que lo habilita):**
- `batch_size: "auto"`: sondea un modelo residente que el streaming nunca carga; se permiten tamaños de batch explícitos (v0.72.3).
- `quantization` distinto de `none` o `4bit`: los demás formatos no se pueden transmitir a un buffer compartido.
- `backend: unsloth` / `backend: mlx`: streaming reemplaza la ruta de carga del modelo que administran esos backends.
- `task` distinto de `sft` / `dpo` / `orpo` / `simpo` / `kto`: se indica explícitamente. `grpo` y `ppo` se rechazan **de forma permanente**, no están pendientes: los rollouts de generación vuelven a leer todas las capas por cada token generado, lo que destruye la amortización necesaria para streaming.
- `task: kto` con `batch_size: 1`: el término KL de TRL es degenerado con batch 1; se rechaza al leer la configuración (esto aplica a toda ejecución KTO, no solo a streaming), en vez de minutos después de fragmentar el checkpoint.
- `lora.use_dora` / `lora.use_vera` / `lora.init_strategy` distinto de `random`: la inicialización requiere los pesos base reales, que están en el dispositivo meta en streaming.
- `moe_expert_quant`: la cuantización de expertos solo se ejecuta en la ruta de construcción residente y, de otro modo, se ignoraría sin aviso.
- `unfrozen_parameters`, `lisa_enabled`, `packing`, `multipack`, `use_fsdp2_compile`, `train_router_only`, `expand_layers`: cada una modifica o vuelve a congelar las mismas capas por separado.
- `use_liger` / `use_flash_attn`: **no se alcanzan en absoluto** en streaming (#1692). Ambas opciones se leen en la preparación residente: Liger parchea allí el forward de MLP y flash-attn se pasa a `from_pretrained` como `attn_implementation`. El modelo en streaming se construye con `from_config`, sin argumento de atención ni parche Liger, y tampoco se establece `_liger_applied`, por lo que `use_liger_kernel` tampoco llega a `TrainingArguments`. Son los ajustes a los que se recurre cuando una ejecución en streaming no cabe; por eso se rechazan en vez de descartarse. Se ofrecerán de nuevo cuando estén conectados y se demuestre que se activan en un modelo en streaming. En todas las tareas fuera de `{sft, tts}` ya se rechazan al cargar la configuración (#806).
- `stream_source` / `stream_ngram_source` / `stream_buffers` / `stream_read_ahead` / `stream_vram_override` / `stream_vram_probe` / `stream_disk_kind` / `stream_pin` configurados mientras `stream_layers: false`: se rechazan para evitar errores de configuración.
- `stream_read_ahead` distinto del valor predeterminado junto con `stream_source: ram`: se rechaza. El lector read-ahead pertenece al nivel de disco NVMe y `ram` nunca recurre a él, de otro modo, el ajuste se validaría y documentaría sin tener efecto. Se acepta el valor predeterminado porque no supone una decisión explícita.
- Ejecución en nivel disco cuyo área de preparación del host (capas `stream_read_ahead`, más un slot para el embedding y otro para un `lm_head` sin pesos compartidos), junto con los extras residentes, no cabe en el margen de RAM libre: se rechaza con el nombre del ajuste de profundidad que debe reducirse. Se rechaza en vez de limitarlo automáticamente: la profundidad configurada es una decisión y reducirla en silencio produciría una ejecución más lenta que la solicitada.
- Más de 8 formas de capa distintas en un índice de shards: se rechaza. La preparación se asigna por forma y una forma con un solo miembro consume un slot completo con cualquier profundidad; un índice donde cada capa fuera distinta mantendría en preparación gran parte del modelo al mismo tiempo. La memoria estaría fijada si el equipo lo permite y `stream_pin` no es false, o habría una advertencia de alternativa pageable; solo se rechazaría específicamente con `stream_pin: true` en CUDA. Es justamente lo que el nivel disco busca evitar. Un modelo real tiene entre una y tres formas.
- `stream_vram_probe` en una tarea que no sea `sft`: la prueba ejecuta un paso causal-LM simple, que *sí* corresponde al paso SFT pero no a una pérdida de preferencias. En una forma coincidente medida también es conservadora (6.02 GB frente a un paso DPO real de 5.30 GB, +13.5%), pero una sola forma no constituye validación; aún no se ofrece para `dpo`/`orpo`/`simpo`/`kto`.
- Arquitectura fuera de la lista compatible (llama / qwen2 / qwen3, incluidas las variantes de texto qwen3_5_moe / qwen4_exp / mistral / gemma / gemma2 / gemma3_text / phi / phi3): se indica explícitamente.

**Ejemplo de configuración:**

```yaml
base: Qwen/Qwen2.5-3B
task: sft
backend: transformers

data:
  train: ./data.jsonl
  format: alpaca
  max_length: 512
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: 1           # explicit sizes allowed; "auto" rejected
  gradient_accumulation_steps: 1   # values > 1 now allowed (v0.72.3)
  quantization: 4bit      # NF4 — ~4x smaller RAM store than bf16 (or `none`)
  gradient_checkpointing: true     # handled per-layer by the streamer
  stream_layers: true     # Enable layer streaming
  stream_source: auto     # RAM with auto-fallback to NVMe disk (v0.72.3)
  stream_buffers: 2       # double-buffering
  lora:
    r: 64
    alpha: 16

output: ./output
```

**Notas de rendimiento:**
- 1.43× más lento que el entrenamiento residente, medido en 0.5B (único tamaño de referencia que realmente cabe de forma residente en 4 GB, por lo que es una comparación justa).
- Las ejecuciones 1.5B alcanzan ~97% de uso de GPU, es decir, están limitadas por cómputo: con almacén fijado, las cargas de capas quedan casi completamente ocultas por el cómputo. El 79.3% de la ejecución 3B **no se debe al tamaño del modelo**: es el costo de recurrir al almacén pageable en ese equipo específico.
- La corrección no es un compromiso: los forwards en streaming y residentes se verificaron **idénticos bit a bit**, y una curva de pérdida de 100 pasos en streaming coincidió exactamente con la residente. Streaming sustituye los mismos bytes de pesos en los mismos kernels. (El control residente es el NF4 simple descrito en la nota de alcance bajo *Corrección*.)

> **Los adaptadores de v0.72.0 no se pueden cargar: vuelve a ejecutarlos en v0.72.1.** En v0.72.0, una ejecución en streaming guardaba cada tensor del adaptador con una clave que incluía un segmento `.inner.` adicional, así que `soup merge`, `soup serve`, `soup chat` y `PeftModel.from_pretrained` cargaban **cero** tensores y devolvían silenciosamente la base sin ajustar (PEFT solo emitía un `UserWarning`). El entrenamiento en sí era correcto: solo se afectó el archivo guardado. Compruébalo con:
>
> ```bash
> python -c "from safetensors.torch import load_file; \
> print([k for k in load_file('adapter_model.safetensors') if '.inner.' in k][:3])"
> ```
>
> Si aparece algo, el adaptador está afectado. Desde v0.72.1, un adaptador en streaming tiene exactamente el mismo formato de bytes que una ejecución LoRA normal.

> **peft 0.21.0 (publicado el 2026-09-15) guardaba vacío un adaptador en streaming en todas las versiones de Soup anteriores a la corrección #1005: vuelve a ejecutar esos entrenamientos.** peft 0.21 selecciona los tensores del adaptador que debe guardar usando los prefijos de `model.named_modules()`; los nombres de módulos del wrapper de streaming incluían `.inner.`, segmento que no aparecía en las claves guardadas. Por eso `trainer.save_model()`, cada checkpoint `save_steps` y `PeftModel.save_pretrained()` escribían un `adapter_model.safetensors` de 40 bytes con **cero** tensores; el entrenamiento informaba éxito y `--resume` no cargaba nada. El entrenamiento era correcto: solo estaba vacío el archivo y no se puede recuperar nada de él. Compruébalo con:
>
> ```bash
> python -c "from safetensors.torch import load_file; \
> print(len(load_file('adapter_model.safetensors')))"
> ```
>
> `0` significa que el adaptador se perdió. Se corrigió en `main` mediante PR #1010 (el wrapper informa ahora nombres canónicos, por lo que peft 0.20 y 0.21 guardan todos los tensores; no se requiere fijar `peft<0.21`); v0.76.0 incluye la corrección. Hasta usar una versión de Soup con ese arreglo, la alternativa es `pip install "peft<0.21"`.
>
> Desde #1011, una ejecución en streaming también comprueba cada checkpoint y el guardado final: si falta `adapter_model.safetensors`, conserva claves `.inner.` del wrapper o contiene una cantidad de tensores LoRA distinta a la entrenada, la ejecución se detiene con `RuntimeError` en vez de informar éxito. La comprobación manual anterior solo es necesaria para adaptadores guardados antes de ese cambio.

**Solución de problemas:**
- **«trainable LoRA parameters remain on the meta device»**: PEFT conectó un adaptador sin almacenamiento real y Soup rechazó la ejecución antes de instalar el runtime de streaming. Esta protección se basa deliberadamente en el estado final de los parámetros y no en cuántos tensores materializó Soup: algunas versiones de PEFT crean adaptadores reales por sí mismas. Al informar el problema, incluye la versión de PEFT y el parámetro indicado por el error.
- **Un adaptador en streaming solicita `--base` al abrirlo**: revisa `adapter_config.json`. Un artefacto correcto registra la referencia exacta del modelo configurado en `base_model_name_or_path`; un valor vacío indica que se produjo con una ruta de streaming anterior que perdió el origen del esqueleto meta. Pasar `--base` sigue siendo una alternativa válida para ese artefacto existente.
- **«layer streaming needs the base to fit in RAM»**: la base es mayor que la RAM libre. Establece `stream_source: auto` para recurrir al nivel disco NVMe, libera RAM o elige una base más pequeña.
- **«base exceeds the physical RAM safety ceiling»**: `stream_source: auto` recurrió al nivel disco NVMe porque el nivel RAM mantendría el almacén más los extras residentes por encima del límite de memoria física del host de Soup. Configura `stream_source: ram` solo si quieres que ese caso se rechace en vez de recurrir a disco.
- **«layer streaming needs NVMe or more RAM … the detected disk is 'hdd'» en un disco de nube rápido**: un dispositivo virtio informa `rotational=1` sin indicar el medio. La detección ahora mide el disco cuando ese indicador no es fiable; si aún lo clasifica mal, establece `training.stream_disk_kind: nvme` para activar el nivel ( `ssd`/`hdd` lo desactivan).
- **«could not page-lock the base … falling back to a PAGEABLE RAM store»**: es esperable en un equipo ocupado. El entrenamiento continúa, pero más lento. Cierra otras aplicaciones para conservar el almacén fijado. Desde #901, la alternativa también indica `after the failed page-lock: cleared the stale CUDA out-of-memory error … and released N GB of page-locked memory it left cached`: un pinning fallido deja en el runtime CUDA un error obsoleto de «out of memory» que, de otro modo, el primer lanzamiento de kernel informaría como un OOM real (el informe muestra `AcceleratorError: CUDA error: out of memory` dentro de `SFTTrainer.__init__`, aunque haya gigabytes de VRAM libres); además, los bloques fijados antes del fallo quedan en la caché host de torch hasta que se liberan. Ambos casos se gestionan antes de construir el almacén pageable. El almacén ahora también se fija en arenas de potencia de dos, por lo que esta alternativa es menos frecuente: un almacén NF4 14B de 9.93 GB se fija como 10 GiB; el pinning por tensor solicitaba ~17 GB.
- **«layer streaming does not support model_type=…»**: la lista compatible es llama / qwen2 / qwen3, incluidas las variantes de texto `qwen3_5_moe` / qwen4_exp / mistral / gemma / gemma2 / gemma3_text / phi / phi3. La variante multimodal `gemma3` está excluida a propósito; usa `gemma3_text`.
- **«predicted peak … exceeds free VRAM» cuando crees que es incorrecto**: primero reduce `batch_size` o `data.max_length`. De lo contrario, hay dos alternativas y no son intercambiables. `training.stream_vram_probe: true` (solo `sft`) **mide** un forward+backward real con la forma configurada y decide a partir de eso, mostrando la predicción junto a la medición; cuesta un paso (1–5 s medidos) y también puede rechazar una ejecución que la fórmula habría aceptado. En cambio, `training.stream_vram_override: <bytes>` **reemplaza** la cifra de VRAM libre que usa la comprobación: es una afirmación tuya, no una medición; aumentarla por encima de un límite real provoca OOM en Linux y desbordamiento silencioso en Windows. Prefiere la prueba si quieres una medición real; usa la sobrescritura cuando conoces algo que el controlador no puede informar.
- **La predicción no es igual de fiable para todas las longitudes de secuencia.** En una RTX 3050 de 4 GB con SmolLM2-135M en streaming bf16, batch 1, la fórmula sobreestima 8% en seq 4352 (seguro), pero luego **subestima: 0.934x del pico real en seq 5120 y 0.787x en 6144**. La cuadrícula de ajuste solo varió tamaño de batch, en seq 256 y 512; por eso el streaming de contexto largo es justo donde tiene menos evidencia. Si usas secuencias de varios miles de tokens, activa `stream_vram_probe`. Registro: [`benchmarks/gate-v0.73.1-measured-vram-fit.md`](../benchmarks/gate-v0.73.1-measured-vram-fit.md).
- **La comprobación previa muestra toda la tarjeta en una GPU limitada o compartida**: `torch.cuda.mem_get_info()` consulta el controlador a nivel de dispositivo y no detecta `set_per_process_memory_fraction`, una partición MIG ni otros procesos en la misma tarjeta. Establece `training.stream_vram_override` con lo que realmente puede usar tu proceso; la comprobación rechazará configuraciones que superen ese valor. También sirve para ensayar una tarjeta de 4 GB en una de 16 GB.
- **Más lento de lo esperado**: el streaming de capas intercambia tiempo por memoria. Si el modelo ya cabe de forma residente en tu tarjeta, no lo actives.

### Pérdidas de preferencia en streaming (v0.72.4)

`dpo`, `orpo`, `simpo` y `kto` usan streaming igual que `sft`: las mismas claves de configuración, la misma comprobación previa y los mismos rechazos. El aspecto interesante es el modelo de referencia de DPO.

**DPO compara el modelo entrenado con una referencia congelada.** Si se implementara como una segunda instancia del modelo, duplicaría la memoria y no tendría sentido usar streaming. En su lugar, Soup usa *la misma base en streaming con los adaptadores LoRA desactivados*, de modo que la referencia no requiere pesos adicionales. En un checkpoint sin pesos compartidos, `dpo` y `kto` también mantienen una copia de la cabeza de salida por paso (aproximadamente 1.05 GB para Llama-3.1-8B en bf16), contabilizada por la comprobación previa de VRAM (#1049); los checkpoints con pesos compartidos y `sft`/`orpo`/`simpo` no pagan ese costo. Medido en una RTX 3050 de 4 GB con un modelo de 730 MB:

| Ruta | Pico de VRAM | Frente a SFT |
|---|---|---|
| SFT en streaming | 89.53 MB | — |
| **DPO en streaming** | **81.87 MB** | **0.914×** |
| La misma ejecución forzada a construir un segundo modelo real | 812.32 MB | 9.92× |

La tercera fila es el control: una segunda instancia cuesta **+730.44 MB frente a 730.44 MB de pesos**, exactamente una copia. El almacén RAM y el pool de buffers VRAM son idénticos byte a byte entre las rutas SFT y DPO.

**KTO sí usa una referencia**, aunque a menudo se describa como si no; la selecciona igual que DPO y recibe el mismo tratamiento. ORPO y SimPO sí son realmente independientes de una referencia. Las cuatro rutas están verificadas **bit a bit** frente a una ejecución residente de la misma pérdida en un checkpoint con pesos compartidos; para uno sin pesos compartidos, esa comprobación cubre `dpo` (`tests/test_issue1049_untied_streamed_preference.py`). El alcance del control residente es el mismo que se describe en la nota bajo *Corrección*.

**El costo está en el tiempo, no en la memoria.** DPO ejecuta la pila de capas tres veces por paso (forward de política, forward de referencia, recomputación del checkpoint), frente a dos veces en SFT: se midieron **1.52×** las lecturas de capas en un modelo de 24 capas. Streaming no añade una segunda copia de pesos para la referencia (en un checkpoint sin pesos compartidos, `dpo` y `kto` siguen manteniendo una copia de la cabeza de salida; consulta arriba); la referencia no es gratuita.

**Dos puntos que debes conocer antes de configurarlo:**

- **`kto` requiere `batch_size: 2` o mayor.** El término KL de TRL es degenerado con batch 1, así que la ejecución no puede funcionar; Soup lo rechaza al leer la configuración, no después de fragmentar el checkpoint. (KTO admite streaming porque v0.72.3 eliminó la restricción de streaming a batch 1.)
- **La comprobación previa de VRAM es deliberadamente conservadora para pérdidas pareadas.** DPO, ORPO y SimPO pasan chosen y rejected al modelo en un único tensor, así que el presupuesto cuenta el doble de filas; es correcto y nunca subestima. Pero usa la tasa por elemento medida para la pérdida supervisada, mientras que las pérdidas de preferencia de TRL usan una ruta más económica, por lo que la estimación es un límite superior, no un ajuste exacto. Por ejemplo, en una tarjeta de 4 GB con un modelo 1B de vocabulario 128k: se permite DPO con `max_length: 512`, pero se rechaza desde `max_length: 768` aunque probablemente cabría. Reduce `max_length` si te ocurre. (Está registrado como seguimiento; subestimar sería mucho peor porque en Windows no genera un error, sino un desbordamiento silencioso a la memoria del host.)

**Hoja de ruta:**
- Un benchmark de referencia publicado para 14B en 8 GB: la parte de **memoria** está lista. Una ejecución NF4 con forma Qwen2.5-14B, batch 1 x seq 384, entrenó de principio a fin en una RTX 5070 Laptop 8 GB / equipo de 32 GB con el almacén fijado; el pico medido fue 2.94 GB frente a una predicción de 3.39 GB ([registro](../benchmarks/gate-901-14b-on-8gb.md), #901). Usa una forma sintética con pesos aleatorios, así que no se informan cifras de rendimiento ni calidad. Una ejecución real de Qwen2.5-14B-Instruct en una RTX 3070 de 8 GB, informada por @hasheng en #901, fijó el almacén de 9.93 GB (10.74 GB fijados), midió un pico de 2.92 GB frente a 3.39 GB previstos y completó 18 de 18 pasos.
- GRPO y PPO **no están planeados** explícitamente: los rollouts requieren generación, que vuelve a leer el modelo por token.

**Comprobación previa de disco y caché de shards.** Antes de materializar o fragmentar un checkpoint, Soup informa la huella proyectada completa: la fuente HF/local, cualquier copia de archivo regular que todavía necesite y la caché de shards por capa. Agrupa las escrituras requeridas por volumen de destino y rechaza la ejecución antes de escribir si falta espacio libre en ese volumen. Si se configura `SOUP_LAYER_STREAM_STRIPE_DIRS`, cada raíz se calcula según los bytes que recibirá, leídos de los encabezados safetensors de origen (la raíz primaria también almacena embedding, cabeza y extras); si no se puede leer un encabezado o el modelo es qwen4_exp, el reparto es uniforme y el panel indica que es aproximado. Sobrescribe las dos raíces de caché con `SOUP_SPECTRUM_CACHE_DIR` y `SOUP_LAYER_STREAM_CACHE_DIR`; ambas conservan la política de contención de Soup para home/cwd/tmp.

**Dos o más unidades NVMe.** Con una caché fría mayor que la RAM, el paso del nivel disco puede esperar a la lectura (medido con un almacén de forma 70B), y una sola unidad se vuelve el límite. Configura `SOUP_LAYER_STREAM_STRIPE_DIRS` con carpetas en otras unidades NVMe (`os.pathsep` separa entradas: `;` en Windows, `:` en otros sistemas) y la caché de capas se distribuye entre N raíces, contando la raíz primaria y las carpetas indicadas: la capa decoder `i` vive en la raíz `i mod N`; el índice, extras y archivos de embedding/cabeza permanecen en la raíz primaria; el lector mantiene una capa en vuelo por unidad. La caché se organiza así sin importar el nivel elegido, pero solo el nivel disco lee las unidades en paralelo. Cada entrada se valida; cualquier fallo rechaza la ejecución e indica la entrada y regla: debe ser una ruta absoluta (en Windows, con letra de unidad o recurso compartido UNC; no se permiten formatos `\\?\` ni `\\.\` ni caracteres de control), existir ya como carpeta (una unidad desmontada nunca se confunde con una carpeta vacía), no ser symlink ni junction de directorio (se acepta un punto de montaje de volumen Windows genuino: junction y punto de montaje comparten la etiqueta reparse, pero solo el destino de una junction señala una carpeta bajo una letra de unidad, UNC o incluso GUID de volumen que puede cambiarse; el punto de montaje señala un volumen completo), no estar dentro de la raíz primaria ni contenerla ni contener/estar dentro de otra entrada, residir en un volumen distinto al de la raíz primaria y al de las demás entradas, y clasificarse como NVMe (`training.stream_disk_kind` sobrescribe una detección incorrecta). Se admiten como máximo 6 carpetas. Las comprobaciones que no requieren sondeo del disco se ejecutan al iniciar `soup train` o `--dry-run`; `soup doctor` enumera cada entrada y usa `--disk` para la regla NVMe. Los bytes son exactamente los mismos que en la caché de una sola unidad, por lo que distribuirlos cambia la velocidad, no el resultado.

Por diseño, las carpetas de distribución no están sujetas a la contención home/cwd/tmp de la raíz primaria: una segunda unidad nunca está bajo `$HOME`. En su lugar se aplica directamente la protección que esa contención proporcionaba. Dentro de cada entrada, Soup solo escribe en su propia carpeta, nombrada con el modelo y un hash de la ruta de la caché primaria, de modo que dos cachés primarias (dos usuarios o dos valores de `SOUP_LAYER_STREAM_CACHE_DIR`) que compartan una unidad no toquen los archivos de la otra. Esa carpeta queda privada para la cuenta que ejecuta Soup (modo 0700 en POSIX; en Windows, ACL protegida para esa cuenta y SYSTEM, que reemplaza la heredada —en una unidad de datos suele permitir que todas las cuentas conectadas modifiquen archivos—); se vuelve a comprobar en cada reutilización, antes de cada escritura por capa y antes de escribir el marcador de un shard, y nunca se sigue si es un enlace o junction. Si Soup no puede privatizar una carpeta, rechaza la ejecución e indica cuál. Si se renombra una raíz mientras se crea un shard y se reemplaza por un enlace, se rechaza en la siguiente escritura, no solo al inicio; esto reduce la ventana a una escritura, pero no la elimina, lo que requeriría escribir respecto a un descriptor de carpeta abierto (la biblioteca estándar no ofrece esto en Windows). Soup nunca borra nada dentro de una carpeta distribuida. Soup solo deja su propia carpeta accesible al propietario, así que también restringe la escritura de la raíz de la unidad distribuida a tu cuenta (en una unidad de datos Windows predeterminada, otras cuentas pueden modificarla).

Cada unidad adicional requiere una capa más de preparación en el host: `stream_read_ahead` toma N + 1 por defecto. El valor configurado 2 se considera el predeterminado y se eleva del mismo modo; cualquier otro valor se conserva (con una advertencia si deja unidades sin usar), y un rechazo que solicite reducir la profundidad indica un valor que no vuelve a elevarse. Cambiar la lista vuelve a fragmentar la caché; quitarla vuelve a fragmentar en una sola raíz e indica la carpeta que queda atrás. Medido en dos unidades PM9B1, sin veredicto formal: 7.65-9.15 GB/s en conjunto frente a ~4 en una ([registro](../benchmarks/probe-rtx5070-two-drive-read.md)). Con aprobación: en un almacén NF4 frío de forma 70B y seq 512, el paso de entrenamiento bajó de 17.4 s a 9.97 s, 1.75x ([gate](../benchmarks/gate-two-drive-striping.md)). Es un pico de rendimiento de un equipo recién enfriado. En lecturas consecutivas sostenidas, ambas unidades de esta laptop reducen velocidad a unos 2.5 GB/s cada una; la mejora medida en la ventana tardía fue 1.28x y 1.41x. Una sola unidad reduce velocidad más tarde y en menor medida ([prueba sostenida](../benchmarks/probe-rtx5070-two-drive-sustained.md), [verificación por unidad](../benchmarks/probe-rtx5070-drive-throttle.md); ninguna tiene aún veredicto formal). Para ejecuciones largas, planifica alrededor de 1.3-1.4x y usa unidades con buena refrigeración.

Los snapshots de Hugging Face suelen exponer symlinks hacia su caché de blobs, que el sharder deliberadamente no sigue. Soup materializa esos pesos dentro de su caché Spectrum. Si la caché HF ya expone archivos reales, Soup ahora los lee directamente en vez de crear una segunda copia. Los shards por capa permanecen en `~/.soup/layer-stream/`. Su índice registra nombre de archivo, tamaño y `mtime_ns` de cada origen, así que un re-fragmentado necesario indica qué componente cambió en vez de gastar minutos reconstruyendo la caché sin aviso.

`base:` también puede ser una ruta local a un snapshot de caché Hugging Face (`.../models--org--name/snapshots/<commit>`). Soup lo copia a archivos regulares y reutiliza esa copia mientras coincidan el commit y los IDs de blob. Cuando la ruta es la carpeta a la que resuelve el ID Hub `org/name` en la caché Hugging Face activa (`HF_HUB_CACHE` / `HF_HOME`; una caché movida y enlazada sigue siendo la misma carpeta), la copia se ubica en el slot de caché Spectrum de ese ID Hub, `weights/org__name`, de modo que ambas formas de nombrar el modelo comparten una sola copia. Un directorio snapshot ubicado en otro lugar recibe su propio slot, `weights/<folder name>@<digest of its resolved location>`: una copia tomada de una carpeta nunca se usa para otra carpeta ni para el ID Hub, y esa carpeta cuesta una copia adicional en disco. Se rechaza un directorio cuyos archivos `.safetensors` sean symlinks pero que no sea uno de esos snapshots (incluido un directorio normal que también tenga un alias symlink a un shard); el mensaje indica los formatos aceptados: ID Hub, directorio snapshot o directorio de archivos regulares.

Esta materialización también funciona con `HF_HUB_OFFLINE=1` si el snapshot Hugging Face estándar está completo. Soup fija el commit resuelto en la consulta inicial de caché y copia solo archivos verificados del almacén blob de ese commit; no hace una segunda solicitud de metadatos al Hub para el directorio de archivos regulares. Si falta un blob o un symlink escapa del destino, se aborta antes de publicar el destino, en vez de dejar un checkpoint parcial que el sharder pudiera consumir. Con huggingface_hub 1.32 o posterior, también se siguen los enlaces al almacén compartido de caché (`<cache>/blobs`); cualquier otro destino se rechaza. El directorio `blobs` propio del repositorio se trata igual que ese almacén y debe ser un directorio real. Si es un enlace (symlink o junction en Windows), Soup se detiene e indica su nombre en vez de seguirlo, así que nunca sigue un enlace de caché que apunte fuera de la caché. Para mantener los archivos del modelo en otro disco, mueve la caché completa o la carpeta completa del repositorio y enlázala: primero se resuelve la ruta snapshot. Una junction dentro de un directorio snapshot se rechaza igual que un symlink de directorio. Una copia previa solo se reutiliza mientras el snapshot siga pasando estas comprobaciones; si el formato cambia después a uno rechazado, la siguiente ejecución se detiene con el mismo mensaje y deja la copia en su lugar.

## Primero, la corrección (v0.36.0)

Cuatro modos de fallo silencioso de Soup pasaron a ser errores explícitos.


### Enmascaramiento de pérdida solo para el asistente

De forma predeterminada, Soup enmascara con `-100` cada token que no sea del asistente para que la pérdida SFT solo refleje lo que el modelo debe *generar*. Se controla con `data.train_on_responses_only` (predeterminado `true`):

```yaml
data:
  train: data.jsonl
  train_on_responses_only: true   # default
  # OR per-message control:
  # train_on_messages_with_train_field: true
```

Cuando el tokenizador incluye una plantilla de chat con marcadores `{% generation %}`, la máscara es exacta. Sin esos marcadores, Soup recurre a recorrer los incrementos de tokenización y documenta que el resultado es menos preciso.

**El enmascaramiento del backend MLX es distinto y los dos backends no son comparables** ([#683](https://github.com/MakazhanAlpamys/Soup/issues/683)). MLX supervisa cada turno del asistente mediante una máscara por token y **excluye el encabezado del asistente**, mientras que la ruta transformers anterior sí lo incluye. MLX también **rechaza** una plantilla de chat cuyas representaciones parciales no sean prefijos de la representación completa, en vez de emitir una máscara que solo parezca plausible. El rechazo ocurre al construir el dataset, antes del bucle de entrenamiento, e indica `data.train_on_responses_only: false` como solución.

En la práctica esto afecta las plantillas de estilo thinking: `Qwen3` inserta su bloque thinking vacío solo para el *último* mensaje del asistente, por lo que se rechazan filas Qwen3 con varios turnos (las de un solo turno sí funcionan). Llama 3.1 y Gemma 3 enmascaran correctamente ambos formatos. `data.train_on_messages_with_train_field` y `data.train_on_prompt` no tienen equivalente en MLX y se informan en la línea `MLX backend ignores:` en vez de descartarse silenciosamente.

Después de tokenizar y truncar, cada fila response-only debe conservar al menos un objetivo de pérdida causal desplazado. Si `data.max_length` trunca la respuesta completa del asistente, Soup rechaza la fila e indica split y número de fila, en vez de entrenar con una secuencia totalmente enmascarada o guardar un adaptador no finito.

### `--trust-remote-code` opt-in (todos los comandos y entrenadores)

Todo comando que carga un modelo ahora requiere `--trust-remote-code` para ejecutar código Python personalizado de un repositorio de modelo (`auto_map` en `config.json`). Las organizaciones de primera parte (Meta, Mistral, Qwen, Google, etc.) no muestran el panel de advertencia; el resto muestra un panel `REMOTE CODE WARNING` antes de cargar. Los checkpoints locales de organizaciones desconocidas que contienen `auto_map` generan un `ValueError` claro durante la construcción, en vez de ejecutar código silenciosamente dentro de `from_pretrained`.

Cobertura:
- `soup train` (todas las tareas: SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward Model, Pretrain, Embedding, BCO y el dispatcher unificado Preference; la evaluación automática posterior al entrenamiento sigue el indicador de la ejecución; consulta `docs/evaluation.md`).
- `soup chat`, `soup serve`, `soup data download`, `soup eval auto`.
- `soup diff`, `soup export`, `soup merge`, `soup infer`, `soup data generate`.
- `soup eval benchmark`, `soup eval custom`, `soup train --find-lr`, `soup draft distill`, `soup shrink`.

`datasets` versión 4 o posterior ya no admite código remoto de datasets y descartaría silenciosamente el indicador, así que se rechaza `soup data download --trust-remote-code` con esas versiones instaladas.

```bash
soup train --config soup.yaml --trust-remote-code
soup infer --model my-org/custom-arch-model --input prompts.jsonl --output out.jsonl --trust-remote-code
soup export --model ./adapter --format gguf --trust-remote-code
```

`soup serve` resuelve esta comprobación **una sola vez** y entrega el resultado al backend que se ejecute. Así funcionaba desde el principio en transformers y vLLM, y desde #619 también en SGLang. Antes, el runtime y el tokenizador de este último cargaban con `trust_remote_code` activado de forma fija, así que lo anterior no se cumplía con `--backend sglang`. Un modelo con código personalizado en ese backend ahora falla al cargar si no se pasa el indicador, cuando antes cargaba y se ejecutaba silenciosamente.

### Fortalecimiento de plantillas de chat

Los tokenizadores sin plantilla de chat ahora generan `ValueError` con una sugerencia de corrección, en vez de construir silenciosamente cadenas basura `f"{role}: {content}"`.

```yaml
data:
  train: data.jsonl
  chat_template: chatml   # or: llama3, qwen2.5, mistral, gemma3, phi4, deepseek-r1, or a raw Jinja string
```

La sobrescritura se instala antes de renderizar los datos conversacionales de SFT, preference, reward-model, PPO, GRPO u Online DPO. El tokenizador guardado conserva la misma plantilla para inferencia. Las tareas que no renderizan chat (`pretrain`, `embedding`, `classifier`, `reranker`, `cross_encoder`, `prm`, `asr`, `moe_lora_routing` y `unlearn`) rechazan `data.chat_template` en vez de ignorarlo silenciosamente. Si se proporciona un nombre de plantilla no registrado, `soup train` y `soup data preprocess` lo rechazan antes de cargar el modelo y el mensaje enumera los nombres conocidos.

Las cadenas Jinja sin procesar se validan: se rechazan bytes nulos, más de 64KB y directivas que acceden al sistema de archivos (`{% include %}`, `{% import %}`, `{% from %}`, `{% macro %}`, `{% extends %}`) al cargar la configuración.

### Tamaño de batch automático con prueba OOM

```yaml
training:
  batch_size: auto                  # unchanged
  auto_batch_size_strategy: probe   # NEW: 'static' | 'probe' | 'auto' (default)
```

Reemplaza la fórmula estática de memoria con un ciclo real que prueba, reduce a la mitad y luego duplica hasta el límite. El tamaño elegido se guarda en caché en `~/.soup/batch_cache.json`, con clave `(model, max_length, quantization, lora_r, gpu_name, gpu_memory_gb)`, para que las ejecuciones repetidas omitan la prueba.

Se rechaza un candidato si el paso genera un error de memoria insuficiente **o** si termina con un pico medido (`max_memory_allocated`) superior a la VRAM que este proceso puede usar. La segunda comprobación cubre el controlador WDDM (Windows nativo y WSL2), donde el asignador vuelca datos al host en vez de generar un error: el paso termina, pero un orden de magnitud más lento. Sin la medición, la prueba aprobaría y almacenaría en caché un batch que no cabe (#649). Las entradas escritas antes de esa comprobación tienen una clave distinta y se ignoran.

## Multi-GPU / DeepSpeed / FSDP

Entrena en varias GPU con DeepSpeed o PyTorch FSDP2:

```bash
# DeepSpeed ZeRO Stage 2 (recommended for most cases)
soup train --config soup.yaml --deepspeed zero2

# DeepSpeed ZeRO Stage 3 (for very large models)
soup train --config soup.yaml --deepspeed zero3

# DeepSpeed ZeRO Stage 2 with CPU offload (optimizer states -> CPU)
soup train --config soup.yaml --deepspeed zero2_offload

# DeepSpeed ZeRO Stage 3 with CPU offload (parameters -> CPU; not enough VRAM)
soup train --config soup.yaml --deepspeed zero3_offload

# DeepSpeed ZeRO++ — quantized weights + gradients, hierarchical partitioning
soup train --config soup.yaml --deepspeed zero++

# FSDP2 Full Shard (native PyTorch, like ZeRO-3)
soup train --config soup.yaml --fsdp full_shard

# FSDP2 Shard Grad Op (like ZeRO-2)
soup train --config soup.yaml --fsdp shard_grad

# FSDP2 Full Shard with CPU offload
soup train --config soup.yaml --fsdp full_offload
```

`zero3_offload` conserva `offload_optimizer: none`: descargar el optimizador hace que DeepSpeed compile con JIT su operación `cpu_adam`, que requiere un CUDA toolkit (`nvcc`) compatible en el equipo. Si lo tienes, copia el JSON generado y cambia esa opción; o parte de `soup fetch deepspeed_configs zero3-cpu-offload`, variante que sí descarga el optimizador y por tanto necesita ese toolkit. Medición en una H100 con Llama-3.1-8B (bf16, LoRA r=8, 256 pasos): 21.65 tok/s con un pico de 38,135 MiB. Consulta [benchmarks/gate-h100-validation.md](../benchmarks/gate-h100-validation.md), STEP 3, que también lo compara con streaming de capas en el mismo equipo, datos y modelo.

### `--deepspeed <file>`: tu propio JSON

`--deepspeed` también acepta la ruta a una configuración JSON en vez del nombre de un preset. Ese archivo es tuyo: llega a DeepSpeed **idéntico byte a byte y por la misma ruta**, salvo que contenga una clave inválida para la ejecución que va a iniciar.

Se reescriben dos claves, y ambas se corrigen porque son errores, no preferencias (#359):

| clave | motivo de la corrección |
|---|---|
| `zero_hpz_partition_size` | DeepSpeed rechaza un valor por el que el tamaño del mundo no sea divisible; por eso el marcador `8` del preset ZeRO++ es inválido en cualquier equipo cuyo total no sea múltiplo de 8 |
| `zero_quantized_weights` / `zero_quantized_gradients` | el cuantizador CUDA fp16 junto con `bf16`, habilitado en el mismo archivo, hace que el all-gather descuantizado retorne `c10::Half` y se encuentre con una activación `c10::BFloat16`, lo que genera `expected mat1 and mat2 to have the same dtype` |

La forma documentada de personalizar ZeRO++ consiste en copiar el JSON del preset, que copia ambos defectos; sin resolverlos, el archivo del usuario heredaría un fallo del que los presets ya están protegidos. Cuando se reescribe una clave, el cambio se **imprime**, la configuración corregida se guarda en una copia temporal y **el archivo original en disco nunca se modifica**. Si la configuración no usa esas claves, no se toca.

Un JSON malformado se pasa tal cual: DeepSpeed diagnostica mejor que Soup una configuración incorrecta, y rechazarla aquí también descartaría archivos que DeepSpeed acepta.

### DeepSpeed + LoRA

Todo entrenador que pueda iniciarse con `--deepspeed` elimina el grupo vacío de optimizador sin decaimiento de HF antes de crear el scheduler de tasa de aprendizaje (#336, ampliado a todos los wrappers en #359). Sin esto, las ejecuciones LoRA fallan en el primer `lr_scheduler.step()`: todos los tensores LoRA entrenables son 2D, así que el grupo sin decaimiento queda vacío; DeepSpeed lo elimina, mientras el scheduler conserva dos `base_lrs`, y el `zip` estricto de torch genera un error. El ajuste fino completo llena ambos grupos, por lo que no se elimina ninguno.

### Indicador `--gpus`: lanzamiento según la topología

```bash
# Auto-detect local GPU count and launch under Accelerate
soup train --config soup.yaml --gpus auto

# Explicit local GPU count
soup train --config soup.yaml --gpus 4

# Print the launch command without starting training
soup train --config soup.yaml --gpus 4 --no-reexec
```

Con más de una GPU, Soup detecta la topología local NVLink / PCIe y se reemplaza por `accelerate launch`. Usa `--no-reexec` para imprimir el comando para ejecución manual; este modo informativo termina con estado 1. `--dry-run` valida la configuración y los datos sin iniciar el entrenamiento.

### Lanzamiento multinodo

Ejecuta Soup en **cada nodo**, con un `--node-rank` distinto. Para dos máquinas con ocho GPU cada una:

```bash
# On rank 0 (replace 10.0.0.10 with its reachable private hostname or IP)
soup train --config soup.yaml --gpus 8 --nodes 2 \
  --node-rank 0 --master-addr 10.0.0.10 --master-port 29500

# On rank 1
soup train --config soup.yaml --gpus 8 --nodes 2 \
  --node-rank 1 --master-addr 10.0.0.10 --master-port 29500
```

`--gpus` indica el número de GPU **por nodo**. Ambos comandos pasan `--num_processes 16 --num_machines 2` a Accelerate, que inicia ocho workers en cada máquina. También se admite una GPU por nodo. `--gpus auto` detecta la cantidad local; úsalo solo si todos los nodos tienen la misma cantidad de GPU visibles.

`--nodes` tiene valor predeterminado 1. Con varios nodos, `--master-addr` es obligatorio; `--node-rank` tiene valor predeterminado 0 y `--master-port`, 29500. Los rangos deben ser distintos y estar entre 0 y `nodes - 1`. Usa el mismo número de nodos, dirección del coordinador y puerto en todas las máquinas. Soup usa rendezvous estático: no aprovisiona máquinas, copia archivos ni ejecuta el comando en otros nodos.

`--master-addr` solo acepta una dirección: literal IPv4/IPv6 o nombre de host. El puerto va en `--master-port`; por eso se rechaza `--master-addr head:29500` en vez de exportarlo como un `MASTER_ADDR` inaccesible en todos los nodos.

Instala las mismas dependencias de Soup y entrenamiento en todos los nodos, y deja disponibles el modelo, la configuración y los datos en las rutas que usa cada comando. Todos los nodos necesitan acceso de red al puerto coordinador del rango 0 y a las conexiones entre pares de NCCL. Configura la red privada y el firewall del clúster; puede que no baste con abrir solo el puerto coordinador.

Los lanzamientos multinodo usan `NCCL_IB_DISABLE=0` por defecto para permitir InfiniBand. En un clúster solo TCP, establece `NCCL_IB_DISABLE=1` antes de iniciar cada nodo. Soup conserva los ajustes NCCL existentes. `--no-reexec` imprime el comando de lanzamiento y esta recomendación sin establecer variables de entorno. `--dry-run` también imprime el comando multinodo y luego valida localmente sin iniciar workers ni cambiar ajustes NCCL. Ninguno de los dos modos prueba la conectividad entre pares.

Las opciones multinodo no se pueden combinar con `--cloud` ni con `--find-lr`.

### FSDP2 + `torch.compile`

Combina `torch.compile` con cualquier preset FSDP para obtener +20-30% de rendimiento:

```yaml
# soup.yaml
training:
  use_fsdp2_compile: true
```

Requiere `--fsdp`, CUDA y `backend: transformers`.

### Configuración de paralelismo de pipeline (solo conexión en v0.27.0)

```yaml
training:
  parallelism: pipeline
  pipeline_stages: 4
```

La validación de configuración llega en v0.27.0; la ejecución activa, en v0.27.1. Consulta `recipes/deepseek-v3-pipeline` para un scaffold completo.

## Rendimiento y contexto largo

Optimiza el rendimiento del entrenamiento y amplía las ventanas de contexto:

`use_liger` y `use_flash_attn` solo los lee el entrenador de la familia SFT (`task: sft` o `task: tts`); si se configura cualquiera en otra tarea, se rechaza al cargar la configuración.

```yaml
# soup.yaml — performance options
training:
  use_liger: true            # Liger Kernel fused ops (measured 12.9% memory, 5.1% throughput); sft/tts only
  use_flash_attn: true       # FlashAttention v2/v3 auto-detection; sft/tts only
  gradient_checkpointing: true  # Required for long sequences

  # Long-context (128k+ tokens)
  rope_scaling_type: dynamic  # RoPE scaling: linear, dynamic, yarn, llama3; sft/tts/pretrain on backend: transformers only
  # use_ring_attention: true  # Sequence parallelism across GPUs

data:
  max_length: 131072          # Up to 1M tokens supported
```

Instala los paquetes opcionales de rendimiento:

```bash
pip install "soup-cli[liger]"     # Liger Kernel fused operations
pip install flash-attn --no-build-isolation  # FlashAttention
pip install "soup-cli[ring-attn]" # Ring FlashAttention (sequence parallelism)
```

## Prueba en vivo del tamaño de batch CUDA

Configura `auto_batch_size_strategy: probe` en `training:` y Soup ejecutará una prueba OOM real antes del entrenamiento:

```yaml
training:
  batch_size: auto
  auto_batch_size_strategy: probe
```

Para cada tamaño candidato `B`, la prueba ejecuta UN forward + backward + step en un batch sintético de `B` secuencias con longitud `max_length`. Si ocurre un error de memoria insuficiente o el pico medido supera lo que este proceso puede usar en el dispositivo (el caso de desbordamiento WDDM, #649), reduce a la mitad; de lo contrario duplica hasta `4 × static_estimate`. El tamaño elegido se guarda en caché por tupla `(model, max_length, quantization, lora_r, gpu)` en `~/.soup/batch_cache.json` para omitir la prueba en ejecuciones posteriores.

Las sesiones CPU y `auto_batch_size_strategy: static` omiten la prueba. Los tensores del batch sintético se liberan antes del backward para que el pico de VRAM refleje un paso de entrenamiento realista. En esta versión solo se admite SFT; otros entrenadores usan el estimado estático.

## Consolidación de shards FSDP

```bash
# Preview the plan (which shards, total size) without writing
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors --plan-only

# Consolidate for real
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors
```

Consolida archivos shard `pytorch_model_fsdp_*.bin` en un solo `.safetensors`. Carga cada shard por separado (en streaming, no todos a la vez) con `torch.load(weights_only=True)`, valida las formas de los tensores (rechaza claves duplicadas con formas incompatibles; si una clave duplicada tiene la misma forma, conserva la primera y muestra una advertencia) y escribe atómicamente el diccionario combinado. La ruta de salida y cada shard están sujetos a contención dentro de cwd y rechazo de symlinks; límite de 16 GiB por shard; `_MAX_SHARDS=1024`. `--plan-only` imprime el plan y termina con código 0. La consolidación activa en el lado de torch llegó en v0.71.14.

## Exportación BitNet de 1.58 bits

El entrenamiento BitNet 1.58 todavía no está implementado. Configurar `training.quantization: bitnet_1.58` se rechaza al cargar la configuración con un error práctico, en vez de continuar por la ruta SFT normal.

Sigue disponible la exportación de un checkpoint BitNet existente mediante el pipeline GGUF ternario TQ1_0 de llama.cpp:

```bash
soup export --model ./output --format bitnet   # → TQ1_0 ternary GGUF
soup export --model ./output --format tq1_0     # same flavour, explicit name
```

La exportación requiere una cadena de herramientas llama.cpp compilada; si faltan, los binarios convert/quantize generan un `FileNotFoundError` claro.

## Cuantización de expertos MoE + entrenamiento solo del router (activo en v0.71.20)

Para modelos fused-MoE entrenados con `moe_lora: true`, hay dos opciones activas:

- `training.moe_expert_quant: nf4 | int8_rowwise`: cuantiza **solo las capas expert `nn.Linear` del fused-MoE** con bitsandbytes (`Linear4bit` para `nf4`, `Linear8bitLt` para `int8_rowwise`), y deja atención y router de gating en precisión completa. El reemplazo ocurre **antes de `get_peft_model` (QLoRA sobre expertos)**, así que PEFT conecta sus adaptadores a la base cuantizada. Los pesos de origen se transfieren realmente a la capa cuantizada (error de descuantización validado de 0.0155 frente al origen en una RTX 3050). Se requieren CUDA y bitsandbytes; en CPU o sin bnb se genera un `RuntimeError` claro. Requiere una disposición de una `nn.Linear` por experto: en un modelo transformers-5 cuyos expertos enrutados son parámetros 3D fusionados (`mlp.experts.gate_up_proj`, como ocurre con todas las familias MoE incluidas en la pila declarada), no hay una `nn.Linear` de experto que envolver; la ejecución se rechaza durante la preparación e indica la arquitectura y los parámetros fusionados, antes de comprobar CUDA (#1421; antes imprimía `applied to 0 expert Linear block(s)` y entrenaba en precisión completa). Si no se encuentra ningún experto, se rechaza del mismo modo.
- `training.train_router_only: true`: congela todos los parámetros expertos y entrena solo el router de gating (se aplica después de LoRA, al conjunto final de parámetros).

Ambas opciones rechazan combinaciones que no tendrían efecto: configurar cualquiera sin `moe_lora=true` falla al cargar la configuración con un mensaje práctico.

**Qué tareas leen cada indicador (#798).** Antes se aceptaban en todas las tareas y solo algunas los aplicaban, así que la configuración guardada afirmaba que se había realizado un entrenamiento que nunca ocurrió:

| indicador | tareas donde se aplica | en otras tareas |
|---|---|---|
| `moe_lora` | `sft`, `pretrain`, `tts`, (desde #798) `dpo`, `kto`, `orpo`, `simpo`, `grpo`, (desde #1099) `ipo`, `bco`, `reward_model`, `ppo`, `embedding`, `online_dpo` y (desde #1151) `distill`, `unlearn`, además de `classifier`, `reranker`, `cross_encoder` con `classifier_lora: true` | se rechaza al cargar la configuración e indica el motivo en `asr`, `moe_lora_routing`, `prm`, la familia classifier sin `classifier_lora: true` y `lora.r > 0`, `backend: unsloth` (todas las tareas) y `task: sft` con `modality: vision` o `audio` |
| `moe_expert_quant`, `train_router_only` | `sft`, `tts` | se rechaza al cargar la configuración e indica la tarea |
| `moe_aux_loss_coeff` | `sft` (texto), `tts`, `pretrain` | un valor **distinto del predeterminado** se rechaza, incluso en `task: sft` con `modality: vision` o `audio` (#1394), en `backend: unsloth` y con `training.stream_layers: true` (#1412), cuyas configuraciones nunca aplican la pérdida auxiliar; el valor predeterminado `0.01` todavía se carga porque todas las configuraciones guardadas y recetas MoE publicadas de esas tareas lo especifican |

**`moe_lora` en las demás tareas LoRA (#1099).** #798 permitía cargarlo, pero no lo leía en `ipo`, `bco`, `reward_model`, `ppo` y `embedding`; `online_dpo` tenía la misma brecha. Las seis crean su adaptador por la misma ruta `build_lora_config`, así que se conectaron al mismo helper en vez de rechazarse. En `embedding` se aplica solo si `lora.r >= 1`; con `r: 0`, el entrenador hace ajuste fino completo y no crea un adaptador que pueda seleccionar el indicador. #1151 cerró los casos restantes: `distill` y `unlearn` crean el adaptador del mismo modo y se conectaron al mismo helper; lo mismo sucede con `classifier` / `reranker` / `cross_encoder` (un solo entrenador) por su ruta optativa de adaptador. Sin `classifier_lora: true` y `lora.r > 0`, ese entrenador hace ajuste fino completo y no crea adaptador, así que allí el indicador se rechaza al cargar. También se rechaza entonces en `asr`, que entrena solo Whisper (sin expertos); `moe_lora_routing`, que no crea adaptador LoRA; y `prm`, que ajusta cada parámetro base. Dos rutas lo rechazan cualquiera que sea la tarea: `task: sft` con `modality: vision` o `audio`, cuya configuración crea el adaptador sin el paso MoE, y `backend: unsloth`. En unsloth el motivo depende de la tarea (#1264): las trece tareas con configuración unsloth conectan su lista fija de atención y nunca leen el indicador, mientras que las demás no tienen configuración unsloth; en ese caso es el backend el que no lo aplica. Ambos mensajes remiten a `backend: transformers`, donde sí se lee. Las demás denegaciones de `moe_lora` de la misma comprobación ocurren primero, así que `asr`, `prm`, `moe_lora_routing` y SFT visión/audio reciben una sola denegación en cualquier backend. Ahora todas las rutas leen `moe_lora` o lo rechazan, salvo `backend: mlx`, donde se carga y `soup doctor --config` informa que se ignora; una prueba de código evita que un nuevo entrenador que cree adaptadores lo omita. `train_router_only` y `moe_expert_quant` requieren `moe_lora: true`, así que en `backend: unsloth` ahora fallan al cargar con el mismo mensaje. `preference` queda cubierto por los entrenadores a los que se deriva, y `tts` por el entrenador SFT del que hereda.

**`moe_lora` requiere `lora.dropout: 0.0` en un MoE con expertos fusionados.** transformers 5.x mantiene los expertos de Qwen3-MoE como parámetros 3D fusionados (`mlp.experts.gate_up_proj`), que peft adapta mediante `lora.ParamWrapper`; ese wrapper genera `lora.ParamWrapper does not work with lora_dropout != 0.`. Con el valor predeterminado del esquema `0.05`, conectar LoRA fallaba por completo, de modo que `moe_lora` no funcionaba en ninguna tarea, ni siquiera `sft`. Soup ahora se detiene al conectar el adaptador con un mensaje que identifica el indicador, en vez de dejar que peft muestre el fallo, y las 29 recetas publicadas que configuran `moe_lora` fijan `lora.dropout: 0.0`. La comprobación se realiza contra el modelo cargado, no al cargar la configuración: que los expertos estén fusionados depende del checkpoint y la versión de transformers; un modelo con un módulo por experto acepta dropout normalmente. Una base densa no se modifica: allí el indicador no tiene efecto. Desde #1421, el rechazo cubre todas las familias con expertos fusionados, incluidos Mixtral y MiniMax: sus expertos ahora son objetivos (párrafo siguiente), así que el wrapper de peft también rechaza dropout; antes conectaba silenciosamente el adaptador sin adaptar expertos.

**`moe_lora` alcanza los expertos fusionados en todas las familias (#1421).** En transformers 5, una capa MoE conserva todos los expertos enrutados como dos parámetros 3D en un módulo (`mlp.experts.gate_up_proj` y `down_proj`), no como una `nn.Linear` por experto; por eso los nombres de módulos no bastan para seleccionarlos. Antes de #1421, `moe_lora` solo enumeraba módulos y que esos nombres llegaran a los expertos dependía de la conversión de nombres propia de peft entre v4 y v5, disponible para algunas arquitecturas pero no otras. Ahora `resolve_moe_lora_targets` encuentra los parámetros fusionados en el modelo cargado y se los entrega a peft como `target_parameters`, junto con los objetivos de módulo ya resueltos (se conserva una lista `auto`, así que `in_proj_qkv` / `out_proj` de Qwen3.5 siguen adaptándose; no se hace con `stream_layers`, cuya preparación resuelve los objetivos en un esqueleto meta y reemplaza la lista). Mediciones en modelos de prueba pequeños con transformers 5.16.1 / peft 0.20.0, adaptadores de expertos enrutados por familia, solo por nombres de módulo y por la ruta del entrenador:

| familia | solo nombres de módulo (antes de #1421) | ruta del entrenador (ahora) |
|---|---|---|
| `qwen3_moe` | sí | sí |
| `deepseek_v3` | sí | sí |
| `glm4_moe` | sí | sí |
| `olmoe` | sí | sí |
| `qwen2_moe` | **no — atención + experto compartido** | sí |
| `qwen3_5_moe_text` | **no — solo experto compartido** | sí |
| `minimax` | **no — solo atención** | sí |
| `mixtral` | **no — solo atención** | sí |
| `kimi_k2` | **no medido** (no se construyen modelos de prueba aquí) | no medido |

`minimax-m3-sft` sigue entrenando LoRA solo en atención: no configura `moe_lora` porque se carga por la ruta de visión de SFT, que nunca ejecuta el paso MoE; se eliminó `minimax-m3-dpo` porque DPO no puede construir el wrapper `minimax_m3_vl` (#1145). Las siete recetas `kimi-k2.x` que configuran `moe_lora` no están probadas, no se consideran correctas: transformers instalado no dispone de un modelo de prueba diminuto para esas configuraciones. Un checkpoint que todavía conserva una `nn.Linear` por experto (disposición de transformers-4) se maneja como antes: las proyecciones se nombran como módulos y no se añade `target_parameters`. En un modelo con expertos fusionados se rechaza combinar `moe_lora` con `lora.use_vera: true` (#1197): VeRA solo envuelve módulos `nn.Linear` y no admite `target_parameters`, así que el indicador adaptaría atención (y un experto compartido) exactamente igual que VeRA sin él. Configura `use_vera: false` para adaptar expertos con LoRA, o elimina `moe_lora`. La disposición con una Linear por experto no se rechaza; VeRA sí puede envolver esas capas.

**`target_modules: auto` con una base MoE.** Antes de #1070, `resolve_lora_target_modules` no tenía mapeos para ninguna arquitectura MoE distribuida por Soup, por lo que `auto` resolvía a `None` y peft rechazaba con `No target_modules passed but also no target_parameters found`. Ahora esas arquitecturas resuelven a sus proyecciones de atención (consulta `docs/peft-and-efficiency.md`); si Soup y peft no conocen una arquitectura MoE, se rechaza durante la preparación y se la identifica. Con `moe_lora: true`, los objetivos provienen del escaneo del modelo, que se realiza *antes* de decidir el rechazo; así, `moe_lora` sigue funcionando en una arquitectura MoE sin mapeo, como `qwen2_moe`.

## Escalera GGUF Unsloth Dynamic 2.0 (v0.53.0)

La interfaz de despacho prevista para la escalera UD de 14 entradas (`UD-Q8_K_XL` … `UD-IQ1_M`) es `soup export --format gguf-ud --calibration-data <calib.jsonl>`. v0.53.0 incluye validadores de lista permitida cerrada, metadatos envueltos en `MappingProxyType` y una comprobación de la forma de la ruta de datos de calibración; la invocación activa de `imatrix` de llama.cpp llega en v0.53.1. Las variantes GGUF IQ compatibles con Apple/ARM (`IQ4_NL`, `Q4_0_4_4`, `Q5_K_M`, etc.) se incluyen como frozensets separados para que el despacho futuro de exportación por CLI pueda seleccionar por familia.

## Tipos de caché KV (v0.53.0)

`training.kv_cache_type: q8_0 | bf16 | f16 | fp8` controla el tipo de elemento de la caché KV durante inferencia. `fp8` solo funciona en Hopper; el backend MLX se rechaza al cargar la configuración.

**El runtime activo de serve llegó en v0.71.14** para el backend transformers:

```bash
soup serve --model ./output --kv-cache-type bf16     # cache stored in the model compute dtype
soup serve --model ./output --kv-cache-type q8_0     # 8-bit quantized KV cache (needs `hqq`)
```

- `bf16` / `f16` resuelven el tipo de cómputo del modelo para el `DynamicCache` predeterminado (sin dependencia adicional).
- `q8_0` conecta la caché KV cuantizada de transformers (`cache_implementation="quantized"`, backend hqq). Si no está instalado ningún backend de cuantización (`hqq` / `optimum-quanto`), la CLI termina con código 2 y una indicación de instalación, en vez de fallar abruptamente.
- `fp8` se rechaza en GPU anteriores a Hopper (capacidad de cómputo < 9.0) con un error de runtime claro que indica vLLM como alternativa en Ampere/Ada.
- La conexión de serve con vLLM / SGLang sigue registrada en [#140](https://github.com/MakazhanAlpamys/Soup/issues/140) (`infra-blocked`).

## Atención FP8 + NVFP4 + `unsloth_bnb_4bit` nativo

Tres booleanos de TrainingConfig amplían el menú FP8 de v0.28.0. `fp8_attention` y `nvfp4` son convertidores torchao activos desde v0.71.21, sujetos a compatibilidad de hardware:

- `fp8_attention: true`: requiere `quantization_aware: fp8` y un backend distinto de MLX y Unsloth (incompatible con `stream_layers: true`). Convierte las proyecciones de atención (q/k/v/o y variantes fusionadas) para entrenar en float8 con torchao en GPU Ada o posteriores (la misma comprobación que `quantization_aware: fp8`). Si la comprobación rechaza la tarjeta, el SO o la compilación de torch, o falta torchao, la ejecución se detiene durante la preparación (#835); si falla la conversión, muestra un error explícito «model may be PARTIALLY converted» en vez de entrenar con un modelo convertido a medias. También se detiene si el modelo no tiene proyecciones de atención, en vez de continuar como si la opción hubiera tenido efecto (#1152).
- `nvfp4: true`: entrenamiento FP4 exclusivo de Blackwell mediante torchao `NVFP4TrainingConfig` + `quantize_`; reemplaza `nn.Linear` con `NVFP4Linear` y cuantiza los GEMM forward **y backward**. (Hasta v0.75.0, Soup solicitaba `NVFP4Config`, nombre que torchao nunca ha exportado, así que el indicador se degradaba a una advertencia amarilla y una ejecución bf16; #826). Está limitado a tareas compatibles (se rechaza en `task: distill` y tareas sin conexión de rendimiento/memoria de v0.28), backend distinto de MLX/Unsloth y `modality: text` (incompatible con `stream_layers: true`); la comprobación de runtime SM ≥ 10 ocurre al construir el entrenador. La corrección hace que el indicador **llegue** al entrenamiento NVFP4: las capas lineales realmente se reemplazan con `NVFP4Linear`. El paso de entrenamiento también debe cumplir los requisitos propios de torchao, que Soup no comprueba: los kernels usan triton y `nvfp4_mm_triton` genera `ValueError: requires M, K, N all divisible by 128` si la forma no es compatible (`torchao/prototype/moe_training/nvfp4_training/nvfp4_linear.py`); por eso un modelo cuyas dimensiones hidden/intermediate no sean múltiplos de 128 fallará allí en vez de entrenar lentamente.
- `unsloth_bnb_4bit: true`: convierte «Unsloth Dynamic 4-bit» de una combinación implícita `backend=unsloth + quantization=4bit` en un indicador con nombre propio. Las combinaciones incoherentes se rechazan mutuamente al cargar la configuración.

El orden de los validadores cruzados elige el error más práctico: el requisito previo `quantization_aware='fp8'` se comprueba antes del rechazo de MLX para `fp8_attention`, y las comprobaciones de compatibilidad de tareas ocurren al cargar la configuración, antes de construir el entrenador en runtime.

En `backend: unsloth`, los ajustes activos `quantization_aware`, `fp8_attention` y `nvfp4` se rechazan al cargar la configuración. No aparecen como filas `rejected` en `soup doctor --config`: una configuración que se puede cargar no puede llegar a generar tal fila. Para estas funciones de precisión, selecciona una ruta transformers compatible.

## Paridad de cuantización LF / Axolotl (v0.53.0)

- `bnb_4bit_use_double_quant`: controla la doble cuantización de BNB. **Su valor predeterminado es `true`** (coincide con todas las rutas de carga 4-bit: residente, streaming de capas y formatos de guardado 4bit de `soup merge`), y ahora se respeta en todas partes (#321): configura `false` para desactivarla y el valor llega realmente a BNB. Configurarla explícitamente requiere `quantization: 4bit`; se rechaza al cargar la configuración si se combina con formatos del menú Quant (gptq / awq / hqq:Nbit / aqlm / eetq / mxfp4 / fp8).
- `quantize_reward_model: true`: activo para los entrenamientos `ppo` y `reward_model`; amplía a modelos de recompensa la conexión del menú Quant de v0.40.5 (`trainer/ppo.py`, `trainer/reward_model.py`).
- `quantize_ref_model: true`: se acepta en tareas con política de referencia (`dpo / ipo / simpo / orpo / bco / kto / preference / grpo / ppo`), pero ningún código lo lee. Campo por etapas: advierte en v0.76 y se rechaza a partir de v0.77 (#808).

## Formatos de guardado avanzados (v0.53.0)

`soup merge --save-format 4bit` y `--save-format 4bit_forced` escribirán un único checkpoint combinado cuantizado BNB-4bit sin el ciclo ineficiente de descuantizar → combinar → volver a cuantizar (receta unsloth `merged_4bit`). `4bit_forced` cuantiza todas las capas lineales, incluido `lm_head`; los modelos con embeddings compartidos (`tie_word_embeddings: true`) no pueden mantener un `lm_head` 4-bit que comparta pesos con los embeddings, así que se rechazan explícitamente indicando el nombre mediante `ValueError`. v0.53.0 incluye la lista permitida cerrada y los metadatos de especificación; el escritor activo llega en v0.53.1.

La interfaz de exportación PTQ prevista para `torchao.quantize_` + `save_pretrained` es `soup export --format torchao --quant-config <yaml>`. Se permiten cuatro esquemas: `Int4WeightOnly`, `Int8DynActInt4`, `Float8DynActFloat8`, `NVFP4`. **Distinguen mayúsculas y minúsculas**: estos son nombres de esquemas de Soup, vinculados a la clase torchao que implementa cada uno en `utils/torchao_compat.py` (tres de los cuatro nombres que Soup antes intentaba buscar con `hasattr` no existen en torchao; #826). `Int4WeightOnly` acepta `group_size`; se eliminó `inner_k_tiles` porque `Int4WeightOnlyConfig` genera `TypeError` con esa opción. A propósito, difiere de `--save-format` (normalizado a minúsculas); esto está documentado en ambos validadores.

## Menú Quant II + Pipeline de exportación (v0.53.1)

v0.53.1 convierte los esqueletos de esquema únicamente de v0.53.0 en conexiones activas:

```bash
# Single-stage BNB-4bit merged checkpoint (no dequant/merge/requant)
soup merge -a ./adapter -o ./merged_4bit --save-format 4bit

# TorchAO PTQ export — closed per-scheme kwarg allowlist
cat > q.yaml <<EOF
scheme: Int4WeightOnly
group_size: 32
EOF
soup export --model ./merged --format torchao --quant-config ./q.yaml --output ./out

# AWQ/GPTQ export (deprecated, removed in Soup v0.77) — an explicit local calibration set is required
soup export --model ./merged --format awq \
    --calibration-data ./calib.jsonl --output ./out-awq
soup export --model ./merged --format gptq \
    --calibration-data ./calib.jsonl --output ./out-gptq

# Unsloth Dynamic 2.0 / IQ / Apple-ARM GGUF via llama.cpp imatrix
soup export --model ./merged --format gguf-ud \
    --gguf-flavour UD-Q4_K_XL \
    --calibration-data ./calib.jsonl \
    --output ./out/model.UD-Q4_K_XL.gguf

# Deploy autopilot with live Quant-Lobotomy measurement
soup deploy autopilot --target rtx-4090-24gb \
    --base meta-llama/Llama-3.2-1B \
    --measure --tasks ./eval_tasks.jsonl \
    --measure-candidates 4bit,gptq,awq
```

Autopilot también detecta automáticamente bases precuantizadas: para `TheBloke/Llama-2-7B-Chat-GPTQ` recomienda `gptq` en lugar de apilar 4-bit encima. La detección consulta la expresión regular del nombre de la base Y `quantization_config.quant_method` en cualquier `config.json` local. Las rutas de modelo fuera de cwd se omiten silenciosamente (semántica de sondeo flexible). Una base `mxfp4` detectada se presupuesta como el modelo bf16 que carga (2 bytes por parámetro), y Autopilot rechaza cualquier `--gpu-budget` inferior a ese valor.

La exportación AWQ y GPTQ está obsoleta y se eliminará en Soup v0.77: ambos proyectos upstream (AutoAWQ y AutoGPTQ) están archivados y ninguno de los extras se puede instalar junto a `[train]`. Hasta entonces, `soup export --format awq` y `--format gptq` siguen funcionando como antes y empiezan con un aviso `Deprecated:` que indica los formatos a los que migrar (`gguf`, `onnx`, `tensorrt`, `bitnet`, `tq1_0`).

Las exportaciones directas AWQ y GPTQ requieren `--calibration-data`. Soup rechaza un JSONL ausente o inutilizable antes de importar el cuantizador o cargar el modelo. Así los datos de calibración se mantienen explícitos y se evita que AutoAWQ descargue silenciosamente su dataset predeterminado, de gran tamaño. Usa `--calibration-samples` para limitar la cantidad de filas JSONL utilizables (predeterminado: 128).

El pipeline GGUF avanzado utiliza POSIX `O_NOFOLLOW` para evitar la carrera TOCTOU entre la comprobación del symlink durante el despacho y la apertura real de los datos de calibración: un entorno preparado no puede sustituir el archivo entre la validación y la lectura.

`soup deploy autopilot --measure` guarda resultados en caché en `~/.soup/deploy_autopilot_cache.json`, con clave `(base, profile, eval-tasks)`. Las invocaciones repetidas terminan de inmediato; pasa `SOUP_DEPLOY_AUTOPILOT_CACHE=<path>` para redirigirla (limitado a home / cwd / tempdir). El candidato recomendado usa una alternativa flexible: primero `OK` según el orden de inserción; si ninguno, el candidato con el delta más pequeño (la menor caída respecto de su propia línea de base).
