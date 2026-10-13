# Servicio, inferencia y exportación

> 🇪🇸 Traducción al español de [`serving-and-export.md`](serving-and-export.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> El servidor de inferencia compatible con OpenAI, la inferencia por lotes, el benchmarking, la fusión/exportación (GGUF/ONNX/TensorRT/AWQ/GPTQ), el endpoint de Anthropic Messages, la decodificación especulativa, el autopiloto de despliegue, la interfaz web y Agent Forge.

**Contenido:**

- [Fusionar un adaptador LoRA](#fusionar-un-adaptador-lora)
- [Exportar a GGUF](#exportar-a-gguf)
- [Inferencia por lotes](#inferencia-por-lotes)
- [Benchmarking de inferencia](#benchmarking-de-inferencia)
- [Servidor de inferencia](#servidor-de-inferencia)
- [Interfaz web](#interfaz-web)
- [Log de trazas del servidor de inferencia](#log-de-trazas-del-servidor-de-inferencia)
- [Soup Quantize — alias ergonómico de exportación](#soup-quantize--alias-ergonómico-de-exportación)
- [Proxy de Llama.cpp](#proxy-de-llamacpp)
- [Estadísticas de latencia de cola + temporizador de llamadas a herramientas](#estadísticas-de-latencia-de-cola--temporizador-de-llamadas-a-herramientas)
- [Autopiloto de despliegue](#autopiloto-de-despliegue)
- [Agent Forge](#agent-forge)
- [Selección automática del SDK de HF Space](#selección-automática-del-sdk-de-hf-space)
- [Convertidor de la API Anthropic Messages](#convertidor-de-la-api-anthropic-messages)
- [Herramientas del lado del servidor](#herramientas-del-lado-del-servidor)
- [Endpoint de Anthropic Messages](#endpoint-de-anthropic-messages)
- [Entrena y mide tu propio draft (`soup draft`)](#entrena-y-mide-tu-propio-draft-soup-draft)
- [Decodificación especulativa por N-gram](#decodificación-especulativa-por-n-gram)
- [Endpoints de herramientas del lado del servidor](#endpoints-de-herramientas-del-lado-del-servidor)

---

## Fusionar un adaptador LoRA

Fusiona un adaptador LoRA con su modelo base en un modelo independiente:

```bash
# Detecta automáticamente el modelo base desde adapter_config.json
soup merge --adapter ./output --output ./merged

# Especifica el modelo base y el dtype
soup merge --adapter ./output --base meta-llama/Llama-3.1-8B --dtype bfloat16
```

Las salidas de entrenamiento de ReLoRA ya son modelos densos independientes: Soup fusiona el adaptador activo final dentro de la base acumulada en memoria antes de escribir la salida del entrenamiento. Carga o exporta ese directorio directamente; `soup merge` es solo para salidas que contienen únicamente el adaptador y no debe usarse para reconstruir una ejecución de ReLoRA.


## Exportar a GGUF

Exporta modelos al formato GGUF para usarlos con [Ollama](https://ollama.com/) y [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
# Exporta el adaptador LoRA (se fusiona automáticamente con la base y luego convierte)
soup export --model ./output --format gguf --quant q4_k_m

# Exporta con distintas cuantizaciones
soup export --model ./output --format gguf --quant q8_0
soup export --model ./output --format gguf --quant f16

# Exporta un modelo completo (ya fusionado)
soup export --model ./merged --format gguf

# Especifica manualmente la ruta de llama.cpp
soup export --model ./output --format gguf --llama-cpp /path/to/llama.cpp
```

Cuantizaciones admitidas: `q4_0`, `q4_k_m`, `q5_k_m`, `q8_0`, `f16`, `f32`

### Compilar llama.cpp (obligatorio para GGUF cuantizado)

Soup clona automáticamente llama.cpp (tag fijado `b5270`) en `~/.soup/llama.cpp` la primera vez que se usa, pero **no** lo compila. El paso de conversión (`f16` / `f32`) funciona solo con el script de Python; cada tipo *cuantizado* necesita además el binario `llama-quantize`, así que compílalo una vez:

```bash
cd ~/.soup/llama.cpp
cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-quantize -j 4
```

Soup encuentra el binario tanto en diseños de configuración única (`build/bin/llama-quantize`, Make/Ninja) como de configuración múltiple (`build/bin/Release/llama-quantize.exe`, MSVC/Xcode), o en el `PATH`. Apunta a un checkout existente con `--llama-cpp /path/to/llama.cpp` o con la variable de entorno `LLAMA_CPP_PATH`.

**Toolchain validada en Windows** (las seis cuantizaciones, luego `soup deploy ollama` → inferencia): Visual Studio 2022 Build Tools con la carga de trabajo *Desktop development with C++* (`Microsoft.VisualStudio.Component.VC.Tools.x86.x64`) + CMake ≥ 3.14, solo CPU. Linux/macOS solo necesitan un toolchain de C++ + CMake. Las compilaciones de llama.cpp con CUDA no están probadas (ver [#144](https://github.com/MakazhanAlpamys/Soup/issues/144)).

> **No** ejecutes `pip install -r ~/.soup/llama.cpp/requirements.txt`: fija `torch~=2.2.1` contra el índice de wheels de CPU y degradará un PyTorch con CUDA, rompiendo el entrenamiento. Soup instala deliberadamente solo las dependencias extra del script de conversión (`gguf`, `sentencepiece`, `protobuf`), sin fijar versiones.

### Exportación a ONNX

Exporta modelos al formato ONNX para usarlos con [ONNX Runtime](https://onnxruntime.ai/):

```bash
pip install "soup-cli[onnx]"
soup export --model ./output --format onnx
soup export --model ./output --format onnx --output ./model_onnx
```

### Exportación a TensorRT-LLM

Exporta modelos al formato TensorRT-LLM para inferencia en GPU de alto rendimiento:

```bash
pip install "soup-cli[tensorrt]"
soup export --model ./output --format tensorrt
soup export --model ./output --format tensorrt --output ./model_trt
```

`pip install tensorrt_llm` fija sus propias versiones de `torch`/`transformers`/`numpy`/`datasets`, lo que puede degradar el stack de un entorno de entrenamiento. Instálalo en un entorno separado del que usaste para entrenar y exporta allí.

### Exportación a GGUF BitNet 1.58 TQ1_0 (disponible en v0.71.20)

Exporta un modelo BitNet de 1.58 bits como un GGUF `TQ1_0` (ternario de 1.58 bits) mediante el pipeline convertir→cuantizar de llama.cpp. Tanto el alias `bitnet` como el sabor explícito `tq1_0` se asignan a la misma cuantización `TQ1_0` (no se necesita matriz de importancia: los pesos ternarios se exportan directamente):

```bash
soup export --model ./output --format bitnet   # → GGUF ternario TQ1_0
soup export --model ./output --format tq1_0     # el mismo sabor
soup export --model ./output --format bitnet --llama-cpp /path/to/llama.cpp
```

Se requiere un toolchain de llama.cpp compilado; los adaptadores LoRA se fusionan automáticamente antes de la exportación. Consulta [Rendimiento y cuantización → BitNet](performance-and-quantization.md) para el lado del entrenamiento.

Tras la exportación, úsalo con Ollama manualmente o despliega automáticamente:
```bash
# Manual (3 pasos)
echo 'FROM ./my-model.q4_k_m.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model

# Despliegue automático (1 paso)
soup export --model ./output --format gguf --deploy ollama --deploy-name my-model
```

### Desplegar en Ollama

Despliega un modelo GGUF directamente en tu instancia local de [Ollama](https://ollama.com/):

```bash
# Despliega un modelo GGUF
soup deploy ollama --model ./output/model.q4_k_m.gguf --name soup-my-model

# Despliega con system prompt y parámetros
soup deploy ollama --model ./model.gguf --name soup-chat \
  --system "You are a helpful assistant." \
  --template chatml \
  --parameter temperature=0.7 \
  --parameter top_p=0.9

# Exporta + despliega en un solo comando
soup export --model ./output --format gguf --deploy ollama

# Lista los modelos desplegados por Soup
soup deploy ollama --list

# Elimina un modelo
soup deploy ollama --remove soup-my-model
```

Plantillas de chat autodetectadas: `chatml`, `llama`, `mistral`, `vicuna`, `zephyr` (o `auto` para inferirlas desde soup.yaml).

### Asociación al registro y opciones de despliegue

`soup export` se integra con el registro local de artefactos de Soup y con los flujos de trabajo de despliegue:

- **Desplegar al exportar (`--deploy`)**: `--deploy ollama` es compatible con los formatos de la familia GGUF (`gguf`, `gguf-ud`, `bitnet`, `tq1_0`). Para formatos que no son GGUF (`onnx`, `tensorrt`, `awq`, `gptq`, `torchao`), `--deploy` falla de inmediato con un error informativo porque Ollama no puede cargar archivos de pesos que no sean GGUF. Usa `--deploy-name` para asignar un nombre de modelo personalizado en Ollama.
- **Registrar exportaciones (`--registry-id`)**: Pasa `--registry-id <id>` para registrar y adjuntar los artefactos exportados a una entrada del registro local en todos los formatos de exportación (`gguf`, `gguf-ud`, `bitnet`, `tq1_0`, `onnx`, `tensorrt`, `awq`, `gptq`, `torchao`). Cuando una exportación genera un directorio (como safetensors con varios shards, datos ONNX o archivos de motor TensorRT en `<output>/engine/*.engine`), todos los archivos de pesos y de motor producidos se descubren y se adjuntan a la entrada del registro.


## Inferencia por lotes

Ejecuta un modelo sobre una lista de prompts y guarda los resultados:

```bash
# Entrada JSONL (cada línea: {"prompt": "..."})
soup infer --model ./output --input prompts.jsonl --output results.jsonl

# Entrada de texto plano (un prompt por línea)
soup infer --model ./output --input prompts.txt --output results.jsonl

# Ajustes de generación personalizados
soup infer --model ./output --input prompts.jsonl --output results.jsonl \
  --max-tokens 512 --temperature 0.3
```

La salida es JSONL con los campos `prompt`, `response` y `tokens_generated`. Muestra una barra de progreso y un resumen de throughput.


## Benchmarking de inferencia

Mide rápidamente la velocidad de generación y el consumo de memoria de tu modelo antes del despliegue:

```bash
# Mide la velocidad local y el uso de VRAM en 3 prompts generados automáticamente
soup bench ./output

# Personaliza los parámetros del benchmarking
soup bench ./output --num-prompts 5 --max-tokens 256

# Usa prompts personalizados desde un archivo de texto (uno por línea) o JSONL
soup bench ./output --prompts-file my_prompts.txt
soup bench ./output --prompts-file bench_suite.jsonl
```

Esto funciona como un "velocímetro" integrado que muestra los tokens por segundo (TPS), la latencia total y las asignaciones máximas de VRAM en una tabla de estado limpia.

`soup bench <model>` es una forma abreviada de `soup bench infer <model>`; ambas aceptan las mismas banderas.

### Benchmark de entrenamiento

`soup bench train` ejecuta un trabajo SFT corto y de longitud fija a partir de tu configuración y escribe un informe JSON. Existe para que una cifra de throughput lleve la evidencia de que el modelo estaba entrenando mientras se medía (#836):

```bash
soup bench train --config soup.yaml --steps 20 --warmup 3 -o bench-train.json
```

La ejecución entrena exactamente `--steps` pasos del optimizador, registra cada paso, no guarda nada y escribe en un directorio temporal (scratch), no en el `output` de la configuración. Los primeros `--warmup` pasos se miden pero se excluyen de la medición de tiempos. Mide `task: sft` en el backend transformers y rechaza explícitamente cualquier otra tarea o backend.

Sale con **1** de dos formas distintas, y solo una de ellas deja un informe:

- **Los rechazos previos al vuelo no escriben informe.** Una tarea o un backend que no mide, un `--steps` que no es mayor que `--warmup`, y un modelo con cero parámetros entrenables se rechazan antes de que empiece el entrenamiento. No se escribe nada en `-o`; se imprime el motivo.
- **Los fallos en las comprobaciones posteriores sí escriben el informe**, con `valid: false` y una entrada por cada comprobación fallida en `failures`. Una ejecución que no entrenó es un resultado, y el archivo conserva la evidencia.

Las comprobaciones, todas ejecutadas sobre el informe final (`trainable_parameters` también es el rechazo previo al vuelo anterior, así que en el uso normal se dispara allí y no escribe nada):

| comprobación | falla cuando |
|---|---|
| `trainable_parameters` | ningún tensor con `requires_grad` tiene almacenamiento real (los `meta` se cuentan aparte). También se comprueba antes del entrenamiento, porque de lo contrario el Trainer moriría en autograd sin indicar la causa |
| `grad_norm` | un paso contado registró `grad_norm == 0.0` o una norma no finita. Un backend que no registra norma es `"not reported by this backend"`, nunca `0.0` |
| `parameters_changed` | los parámetros entrenables son idénticos bit a bit antes del primer paso y después del último. Esta comprobación no necesita `grad_norm`, así que también cubre los backends que no registran ninguna |
| `step_count` | se ejecutaron menos pasos del optimizador de los solicitados |

Solo se comprueban los pasos posteriores al warm-up para `grad_norm`. Bajo fp16 el GradScaler puede saltarse sus primeros pasos por overflow y registrar una norma no finita; sube `--warmup` por encima de ellos en lugar de leer eso como divergencia.

Campos del informe:

| campo | significado |
|---|---|
| `valid`, `failures` | el veredicto, y un `{check, message}` por cada comprobación fallida |
| `checks` | `trainable_parameters` (recuento), `grad_norm` (estado), `parameters_changed` (bool) |
| `timing` | `median_seconds`, `p95_seconds` (rango más cercano), `counted_steps`, `warmup_steps_discarded`, `total_seconds` y `step_seconds`: el tiempo de cada paso contado en orden de ejecución, de modo que una ejecución que cambió de modo a mitad de camino muestra dónde |
| `tokens` | `useful` (supervisados: `labels != -100`), `total` y `utilisation` (`useful / total`), contados a partir de los batches que recibió `training_step` |
| `throughput` | `useful_tokens_per_second` y `total_tokens_per_second` |
| `memory` | `max_memory_allocated_bytes` y `max_memory_reserved_bytes`, mantenidos por separado y leídos después de `reset_peak_memory_stats`. Ambos son `null` fuera de CUDA. Nunca se leen desde `nvidia-smi` |
| `provenance` | dispositivo, tarjeta, runtime de CUDA, compute capability, versión del driver, reloj de SM durante los pasos contados (`sm_clock_mhz_busy`: `min`, `median`, `max` y `sample_count` de las muestras de `nvidia-smi` conservadas solo de los pasos posteriores al warm-up. Las consultas empiezan con una cadencia de 100 ms, y una consulta más lenta se salta ticks. Cada una lleva marca de tiempo en el punto medio de su consulta, tomada para el propio dispositivo de torch por PCI bus id (la primera GPU de nvidia-smi cuando torch no informa ninguna); la memoria nunca se lee de esta forma. Sin ninguna muestra en la ventana, los números son `null`, y `unavailable_reason` dice por qué: `"not a CUDA run"`, `"no nvidia-smi tool found"`, `"nvidia-smi returned no readable SM clock"` o `"no sample fell inside the counted steps"` (una ejecución demasiado corta para la cadencia). Es `null` siempre que haya números), plataforma, Python, versiones de paquetes (torch, transformers, peft, trl, bitsandbytes, accelerate), dtype, optimizador, seed, seed de datos |
| `config_hash`, `resolved_config` | sha256 de la configuración totalmente resuelta, y la propia configuración, para que un cambio en un valor por defecto del esquema que altere una ejecución quede visible (#716) |
| `steps_requested`, `steps_measured` | lo que se pidió y lo que se ejecutó |

El tiempo de un paso va desde el final del paso anterior hasta su propio final, así que la carga de datos cuenta. El primer paso va desde su propio inicio. En CUDA cada límite se lee después de `torch.cuda.synchronize()`. `parameters_changed` demuestra que algo se movió, no que el modelo aprendiera algo útil. Es un mínimo, no una puerta de calidad.


## Servidor de inferencia

`--auto-quant` actualmente se rechaza con código de salida 2. Soup no puede comparar GGUF, AWQ, GPTQ, FP8 y una base sin cuantizar antes de que un motor de servicio los haya cargado; medir los tiempos de un stub sin evaluar haría que el resultado dependiera del ruido del temporizador y podría forzar un formato que el checkpoint no contiene. Cuantiza el checkpoint explícitamente y luego sirve ese checkpoint sin `--auto-quant`.

Inicia un servidor de inferencia local compatible con OpenAI:

```bash
# Instala las dependencias del servidor
pip install "soup-cli[serve]"

# Inicia el servidor
soup serve --model ./output --port 8000

# Con ajustes personalizados
soup serve --model ./output --port 8080 --host 127.0.0.1 --max-tokens 1024
```

Endpoints:
- `POST /v1/chat/completions` — chat completions (con soporte de streaming)
- `GET /v1/models` — lista los modelos disponibles
- `GET /health` — health check

Compatible con el SDK de OpenAI:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="output",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Cuantización de KV Cache (`soup serve --kv-cache-type`)

Reduce la KV cache en tiempo de inferencia en el backend transformers:

```bash
soup serve --model ./output --kv-cache-type bf16   # caché en el dtype de cómputo del modelo
soup serve --model ./output --kv-cache-type q8_0   # caché cuantizada de 8 bits (necesita `hqq`)
```

`bf16`/`f16` no necesitan ninguna dependencia extra; `q8_0` requiere un backend de cuantización (`hqq` / `optimum-quanto`) o la CLI sale con una sugerencia de instalación; `fp8` es solo para Hopper. El detalle completo y el estado de vLLM/SGLang están en [Rendimiento y cuantización → Tipos de KV Cache](performance-and-quantization.md#kv-cache-types-v0530).

### Backend vLLM (inferencia 2-4x más rápida)

Usa [vLLM](https://github.com/vllm-project/vllm) para obtener un throughput notablemente mejor en producción:

> **Instala `[serve-fast]` en un entorno separado de `[train]`.** Sus stacks de acelerador todavía pueden resolver builds de `torch` distintas. `soup env check` señala un paquete instalado que viola los límites declarados de esta distribución (código de salida 3), para que puedas detectar un venv contaminado antes de que una ejecución dependa de él.

```bash
# Instala el soporte de vLLM (en su propio entorno, no en el de entrenamiento)
pip install "soup-cli[serve-fast]"

# Inicia con el backend vLLM
soup serve --model ./output --backend vllm

# Multi-GPU con paralelismo de tensores
soup serve --model ./output --backend vllm --tensor-parallel 2

# Controla el uso de memoria de GPU
soup serve --model ./output --backend vllm --gpu-memory 0.8

# Limita la longitud de secuencia cuando la KV cache no cabe
soup serve --model ./output --backend vllm --max-model-len 8192
```

> **Consejo:** Soup detecta vLLM automáticamente. Cuando está instalado, verás una sugerencia durante `soup serve` si aún no lo has habilitado.

El backend vLLM aplica la **propia plantilla de chat del modelo**, exactamente igual que el backend transformers, y codifica él mismo el prompt renderizado para que el motor reciba los mismos token ids con los que entrena Soup, en lugar de volver a tokenizar la cadena. (El backend MII también tokeniza mediante un wrapper sobre el mismo tokenizador, así que un prompt con plantilla llega a su motor con los ids que implica la plantilla, #891.) Un modelo que no trae plantilla de chat recurre a un prompt genérico `User:` / `Assistant:`, y el servidor lo indica al arrancar. `finish_reason` informa `"length"` cuando una respuesta alcanza `max_tokens` y `"stop"` en caso contrario (`/v1/messages` los asigna a `max_tokens` / `end_turn`).

### Backend SGLang

Usa [SGLang](https://github.com/sgl-project/sglang) como backend alternativo de alto throughput:

```bash
# Instala el soporte de SGLang
pip install "soup-cli[sglang]"

# Inicia con el backend SGLang
soup serve --model ./output --backend sglang

# Multi-GPU con paralelismo de tensores
soup serve --model ./output --backend sglang --tensor-parallel 2
```

Como los backends transformers y vLLM, el backend SGLang aplica la **propia plantilla de chat del modelo** mediante el mismo constructor de prompts compartido (con alternativa a un prompt genérico `User:` / `Assistant:` para modelos sin plantilla). Cuando la plantilla renderizó el prompt, Soup lo codifica él mismo y envía los token ids al endpoint `/generate` del runtime, de modo que el motor no puede añadir un segundo BOS al que la plantilla ya renderizó (#890); la alternativa sin plantilla se sigue enviando como cadena y la tokeniza el motor como antes. `finish_reason` informa `"length"` cuando una respuesta alcanza `max_tokens` y `"stop"` en caso contrario, así que un cliente que hace continue-on-length puede distinguir una respuesta truncada de una completa (#360).

También respeta `--trust-remote-code` como cualquier otro backend. **Esto cambió:** antes, el runtime de SGLang y su tokenizador se cargaban con `trust_remote_code` activado de forma fija, así que el código personalizado del repo de un modelo se ejecutaba tanto si lo aceptabas como si no; el panel previo al vuelo lo anunciaba pero no ofrecía forma de rechazarlo. Un modelo que necesita código personalizado ahora **falla al cargar** en este backend a menos que pases la bandera:

```bash
soup serve --model ./output --backend sglang --trust-remote-code
```

### Decodificación especulativa

Usa un modelo draft más pequeño para acelerar la generación. **Mídelo antes de confiar en él**; consulta [Entrena y mide tu propio draft](#entrena-y-mide-tu-propio-draft-soup-draft) más abajo. La decodificación especulativa solo vale la pena cuando el draft coincide con el target lo suficiente como para pagar su propio forward pass; en un target pequeño es con frecuencia una *ralentización*.

```bash
# Backend transformers — usa la generación asistida de HF
soup serve --model ./output --speculative-decoding small-draft-model --num-speculative-tokens 5

# Backend vLLM — usa la decodificación especulativa nativa de vLLM
soup serve --model ./output --backend vllm --speculative-decoding small-draft-model

# Emparejamiento automático: Soup elige el draft por ti según la familia del target
soup serve --model meta-llama/Llama-3.1-70B-Instruct --backend vllm --auto-spec
# → emparejado automáticamente: meta-llama/Llama-3.2-1B-Instruct (target: Llama-3.1-70B-Instruct)
```

`--auto-spec` cubre Llama 3.1/3.3/4, Qwen 2.5/3, Mistral Large, Mixtral, DeepSeek V3/R1 y Gemma 2/3. Los modelos sin un emparejamiento de draft conocido (por ejemplo, targets de 8B o menos, donde la sobrecarga de draft+target supera la ganancia) imprimen una nota amarilla de "no draft" y recurren a la decodificación estándar. Un draft que entrenaste tú mismo con `soup draft distill` se toma **antes** que esta tabla integrada.

La decodificación especulativa con tokenizadores distintos es compatible con el backend Transformers mediante Universal Assisted Decoding (UAD). Cuando los modelos target y draft usan tokenizadores diferentes, Soup carga el tokenizador del draft y enruta a través del generador de candidatos entre tokenizadores de Transformers. Esto requiere una versión de `transformers` que admita la decodificación asistida entre tokenizadores (Soup lanza un error claro si la versión instalada carece de esta capacidad). Los pares con el mismo tokenizador siguen usando la ruta nativa de generación asistida de un solo tokenizador.

### Entrena y mide tu propio draft (`soup draft`)

La pregunta que nadie responde antes de habilitar la decodificación especulativa: *¿propondría realmente el draft los tokens que mi modelo va a emitir?* `soup draft measure` la responde.

```bash
# ¿Compensaría este draft? Tasa de aceptación + throughput REAL de plain frente a assisted.
soup draft measure --target ./my-tuned-model \
  --draft HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts prod-prompts.jsonl

# Destila un draft a partir de tu propio target y luego sírvelo automáticamente.
soup draft distill --target ./my-tuned-model \
  --draft-base HuggingFaceTB/SmolLM2-135M-Instruct \
  --data traffic.jsonl -o draft/
soup serve --model ./my-tuned-model --auto-spec     # toma ./draft
soup draft list
```

La **tasa de aceptación** es la fracción de los propios tokens greedy del target que el draft habría propuesto correctamente (coincidencia de argmax con teacher forcing: la métrica que reportan los artículos de Medusa/EAGLE). Cuanto más alta, mejor, y la banda STRONG / MODERATE / WEAK (≥70% / ≥50%) califica solo la tasa. **No dice si el par es más rápido.** Eso depende de cuánto más rápido sea el draft que su target: en el único par medido a escala (target Llama-3.1-8B, draft Llama-3.2-1B, una H100), una aceptación del 81.3% era STRONG y la generación asistida corrió a 0.48x de plain. `--min-acceptance 0.6` sale con **2** por debajo del umbral, de modo que CI puede usarlo como puerta (salida 0 = ok, 2 = por debajo del umbral, 1 = error).

**Punto de equilibrio y longitud del draft.** `measure` también mide el tiempo de la decodificación del draft por sí sola e informa:

- **Cociente de latencia** `c` = tok/s de plain ÷ tok/s del draft solo: lo que cuesta un token del draft en pasos del target.
- **Aceptación de equilibrio** en el `--num-assistant-tokens` en uso: la tasa a la que la generación asistida dejaría de ser más lenta. "None" significa que ninguna tasa compensa, lo que siempre ocurre cuando el draft no es más rápido que el target.
- **Mejor k**: la longitud del draft en 1..64 que maximiza la aceleración modelada con tu aceptación medida.

Estos tres son **modelados, no medidos**. Usan el modelo estándar de tokens esperados: aceptación por posición `a` independiente entre posiciones, `E = (1 − a^(k+1)) / (1 − a)` tokens por paso a un coste de `k·c + 1` pasos del target. Así que son un techo que excluye la sobrecarga del framework. En el par de H100 anterior, el modelo daba 0.955x en k=5 frente a un 0.481x medido. Para medir en su lugar, añade `--sweep-k 1,2,3,5,8` (como máximo 8 valores). Mide el tiempo de la generación asistida en cada k e informa el mejor k medido junto al modelado. Los números son de una sola ejecución greedy, y `soup serve` muestrea con temperatura 0.7 por defecto, así que trata el mejor k como un punto de partida.

*Nota: para los drafts con tokenizadores distintos, la tasa de aceptación medida es un límite inferior estricto. Una fusión de frontera de tokens entre el prompt y el primer token generado puede hacer que la puntuación se lea hasta `1/n_gen` por debajo de su valor real, pero nunca informará de más.*

`distill` ejecuta KD de logits mediante el entrenador `task: distill` existente y emite un modelo **denso** (un directorio de adaptador PEFT no se puede cargar como `assistant_model`). Cuando el draft y el target comparten tokenizador, se usa la destilación estándar. Cuando los tokenizadores o los vocabularios difieren, `soup draft distill` enruta automáticamente a través de Universal Logit Distillation entre tokenizadores (`uld_strategy: wasserstein_aligned`). Una salida de `soup shrink` es una buena base de draft: mismo tokenizador por construcción.

**Medición entre tokenizadores.** Cuando los tokenizadores difieren, los IDs de token en bruto no se pueden comparar directamente. `soup draft measure` detecta automáticamente tokenizadores no coincidentes y evalúa la aceptación alineando los tramos de caracteres decodificados (`count_accepted_spans`). Tanto para pares con el mismo tokenizador como entre tokenizadores, la generación del target neutraliza solo `repetition_penalty` con `repetition_penalty=1.0` (Refs #345) para que las propuestas del draft y las continuaciones del target se evalúen bajo reglas de argmax consistentes; los demás procesadores de generación (`no_repeat_ngram_size`, `encoder_repetition_penalty`, `min_new_tokens`, `bad_words_ids`, `suppress_tokens` y `sequence_bias`) se dejan intactos. El soporte entre tokenizadores amplía la elección de modelos draft, pero la decodificación especulativa entre tokenizadores añade sobrecarga de alineación de tokenización y no garantiza una aceleración; usa `soup draft measure` para verificar el throughput real en tu par concreto.

**Comprobación de la realidad medida (sé escéptico con las afirmaciones de aceleración, incluidas las nuestras).** En `SmolLM2-360M-Instruct` con un draft `SmolLM2-135M-Instruct`, el draft *de serie* ya puntuaba **69.3%** de aceptación, destilarlo no cambió nada (69.7% tras 2 epochs, 69.3% tras 10), y la decodificación asistida resultó **0.55–0.64×**: una ralentización neta. Un draft pequeño de la misma familia ya está cerca de su techo para coincidir con el target, y el forward pass del draft cuesta más que los tokens que ahorra. Que la destilación compense en un par target/draft más grande o realmente divergente no está probado en una máquina de 4 GB. Ejecuta `soup draft measure` en *tu* par en lugar de suponerlo.

### Caché de prefijos

Para cargas de trabajo de RAG y de agentes con un system prompt compartido, habilita la caché de prefijos automática de vLLM:

```bash
soup serve --model ./output --backend vllm --prefix-cache
```

La primera petición con un prefijo dado calienta la caché; las siguientes se saltan por completo el cómputo del prefijo compartido. Gran mejora de latencia cuando más de 100 peticiones comparten el mismo system prompt.

### Hot-swap dinámico de LoRA

Cambia el adaptador activo en tiempo de ejecución sin reiniciar el servidor:

```bash
soup serve --model base-model --adapters chat=./chat-adapter --adapters code=./code-adapter
```

```bash
# Activa un adaptador
curl -X POST http://localhost:8000/v1/adapters/activate/chat
# → {"active": "chat", "status": "ok"}

# Vuelve al modelo base
curl -X POST http://localhost:8000/v1/adapters/deactivate
# → {"active": null, "status": "ok"}

# Lista los adaptadores cargados con su indicador de activo
curl http://localhost:8000/v1/adapters
# → {"adapters": [{"name": "chat", "active": true}, ...], "active": "chat"}
```

Los nombres se validan contra `^[a-zA-Z0-9][a-zA-Z0-9-]*$`; las llamadas de activar/desactivar son thread-safe detrás de un lock. Activar/desactivar también comprueban las cabeceras `Host` y `Origin` (consulta [Endpoints de herramientas del lado del servidor](#endpoints-de-herramientas-del-lado-del-servidor)).

### Banco de vectores multi-tenant (`soup serve --bank`)

Sirve muchas personas por usuario desde un solo modelo con un coste de KB por usuario en lugar de un LoRA completo para cada una. Un banco VeRA / VB-LoRA guarda una proyección aleatoria compartida (reconstruida de forma determinista a partir de una seed, nunca almacenada en disco) más un pequeño vector de escalado por usuario. El usuario activo se elige **por petición** mediante la cabecera `X-User-Id`:

```bash
# bank.json contiene {name, base_model, projection_seed, vector_dim, entries: [{user_id, scaling}, ...]}
soup serve --model base-model --bank ./bank.json --bank-strength 1.0
```

```bash
# Aplica la persona de alice a esta petición
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-User-Id: alice" \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'

# Sin X-User-Id o desconocido → no-op de delta cero (modelo base puro, sin filtraciones entre peticiones)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'
```

El delta por token es `v_user ⊙ (x @ Pᵀ)`, que se suma al residual de la última capa del decoder mediante un forward hook en tiempo de decodificación. `--bank-strength` escala el delta (magnitud limitada a 100). La ruta de `--bank` debe estar bajo tu cwd. El usuario activo se resuelve **por petición** mediante un `contextvars.ContextVar`, así que las peticiones concurrentes en un servidor con hilos nunca compiten por estado compartido: cada petición (streaming y no streaming) obtiene su propia selección aislada de usuario activo, y un id ausente o desconocido se limpia solo a la línea base limpia. (v0.71.12 / por petición v0.71.17)

### Servir un MoLE entrenado (`soup serve --mole`)

Sirve un adaptador Mixture-of-LoRA-Experts entrenado con `task=moe_lora_routing`. La ejecución de entrenamiento escribe un `mole_manifest.json` autodescriptivo junto a `mole_gate.pt`; `soup serve --mole <dir>` carga el modelo base + los N LoRA de tareas congelados + la puerta entrenada y los combina **por token** en tiempo de decodificación (un bucle de mezcla personalizado, tanto en no streaming como en streaming SSE):

```bash
# <dir> es la salida del entrenamiento MoLE (contiene mole_gate.pt + mole_manifest.json)
soup serve --model ./mole_out --mole ./mole_out --device cuda
```

El modelo base proviene de `--base` si se define; si no, de la base registrada en el manifiesto. `--mole` requiere `--backend transformers` y es mutuamente excluyente con `--bank` / `--steer` / `--adapters` / `--speculative-decoding`. El manifiesto + la puerta están contenidos en el cwd, se rechazan symlinks y se limita su tamaño, y la puerta se carga con `weights_only=True`. Como la mezcla recalcula la secuencia en cada paso (sin KV cache), esta ruta es mejor para modelos pequeños o de demo. (v0.71.17)

### Salida estructurada (JSON Schema / Regex)

Restringe la salida del modelo a un JSON schema válido o a un patrón regex:

```bash
# JSON schema (el archivo de schema debe estar bajo tu cwd)
soup serve --model ./output --structured-output json --json-schema product.json

# Regex (longitud máxima 2048 caracteres, se rechazan bytes nulos)
soup serve --model ./output --structured-output regex --regex-pattern '\d{3}-\d{4}'
```

El helper `validate_json_schema` limita el tamaño serializado a 64KB y exige un campo `type` de nivel superior, para que los schemas mal formados fallen rápido al arrancar el servidor, no por petición.

### Dashboard de batching continuo + `/metrics`

Sigue el estado del servidor en vivo:

```bash
soup serve --model ./output --dashboard
```

```bash
curl http://localhost:8000/metrics
# → {
#   "requests_total": 1234,
#   "tokens_generated_total": 456789,
#   "active_requests": 3,
#   "latency_p50_ms": 185.2,
#   "latency_p95_ms": 720.0,
#   "latency_samples": 1000
# }
```

`/metrics` lo sirven los backends **transformers** y **vLLM** (en ambos, se pase o no `--dashboard`; la bandera solo registra la intención). Los backends **SGLang** y **MII** no lo exponen; pasar `--dashboard` allí imprime una advertencia al arrancar en lugar de no recolectar nada en silencio.

Los percentiles de latencia se calculan a partir de las últimas 1000 peticiones; los contadores incluyen las rutas de fallo, para que el dashboard muestre la fiabilidad real, no solo la tasa de éxito.

### Trazado de peticiones con OpenTelemetry

Emite spans por petición a tu colector OTLP:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp

soup serve --model ./output \
  --trace \
  --trace-endpoint http://localhost:4317
```

El endpoint OTLP está endurecido contra SSRF: solo esquemas http/https, HTTP simple solo para loopback (`localhost`/`127.0.0.1`/`::1`), y RFC1918 / link-local / el compartido `100.64.0.0/10` / site-local `fec0::/10` / `0.0.0.0` se rechazan todos mediante `ipaddress.ip_address`; un host escrito solo con números que no sea una dirección IPv4 válida también se rechaza. Cuando falta el SDK, la bandera es un no-op con una advertencia: el servidor arranca sin problemas sin spans.

> **Nota:** `max_tokens` está limitado a 16,384 por petición. Los detalles de error nunca se exponen en las respuestas HTTP.


## Interfaz web

Lanza una interfaz web local para gestionar experimentos, iniciar entrenamientos, explorar datos y chatear con modelos, todo desde tu navegador.

```bash
pip install "soup-cli[ui]"
soup ui
# -> abre http://127.0.0.1:7860/?token=<token> en tu navegador
# -> imprime la misma URL y el token en el panel de arranque (también con --no-browser)
```

La URL que `soup ui` abre e imprime ya lleva el token, así que la primera pestaña se autentica sola. En un bind de loopback, el servidor cambia ese token por una cookie de sesión HttpOnly y redirige, como se describe más abajo, de modo que la barra de direcciones queda limpia y una recarga o una pestaña nueva siguen con la sesión iniciada. El historial de tu navegador igualmente registra la URL que se le dio.

> **v0.75.0 hasta v0.75.2: añade el token a la URL tú mismo.** En estas versiones, la pestaña que abre `soup ui` no lleva token, así que muestra `Error loading dashboard: Unauthorized` y ninguna página puede cargar datos (#1194). Abre `http://127.0.0.1:<PORT>/?token=<TOKEN>` en su lugar, con el puerto de la línea `URL:` del panel de arranque y el valor de su línea `Token:`. La página conserva el token para esa pestaña, así que una recarga funciona, y lo quita de la barra de direcciones; el historial de tu navegador igualmente registra la URL. Una pestaña nueva u otro navegador necesitan la URL de nuevo, y el token cambia cada vez que arranca `soup ui`, a menos que pases `--auth-token`. Las versiones con la cookie de sesión también mantienen una pestaña nueva con la sesión iniciada; consulta "Cookie de sesión en un bind de loopback" más abajo. Las versiones posteriores abren e imprimen esa URL por sí mismas.

**Páginas:**
- **Dashboard** — consulta todas las ejecuciones de experimentos, gráficos de loss, información del sistema y comparación de varias ejecuciones
- **New Training** — crea configuraciones desde plantillas o 170 recetas ya hechas, valídalas e inicia el entrenamiento con streaming de logs SSE en vivo y barra de progreso
- **Data Explorer** — explora e inspecciona datasets (JSONL, JSON, CSV, Parquet)
- **Model Chat** — chatea con respuestas en streaming, temperatura/top_p/max_tokens configurables, system prompt, selección de adaptador, renderizado de markdown y exportación del chat

**Monitoreo en vivo + UX mejorada:**
- **Training Live Monitor** — streaming de logs SSE en tiempo real, métricas en vivo, barra de progreso con ETA
- **Enhanced Metrics** — cuadrícula de gráficos 2x2 (loss, LR, grad_norm, throughput) + gráfico de memoria de GPU, tabla de resultados de evaluación
- **Multi-Run Compare** — superpone curvas de loss de hasta 5 ejecuciones lado a lado
- **Chat Upgrade** — streaming SSE vía proxy, indicador de escritura, botón de cancelar, renderizador de markdown (negrita, cursiva, bloques de código), exportación del chat como JSON
- **Config Builder** — menú desplegable de recetas (170 recetas), API de esquema de configuración para generar formularios dinámicos

La norma del gradiente admite nulos: los backends o pasos que no la informan almacenan y transmiten `null`, y el gráfico de la interfaz web deja un hueco en lugar de dibujar un cero falso. Un `0.0` realmente registrado sigue siendo un valor medido y aparece en el panel de la terminal.

**Seguridad:** La interfaz web genera un token de autenticación aleatorio al arrancar (se imprime en la consola). Cada endpoint privado, tanto los que mutan (iniciar/detener entrenamiento, eliminar ejecuciones, inspeccionar datos, validar configuración) como los de lectura (ejecuciones, métricas, sistema, recetas, streams SSE), requiere una cabecera `Authorization: Bearer <token>` header. `/` y `/api/health` permanecen abiertos para que el dashboard pueda cargar. CORS está restringido al origen servido. La inspección de datos está aislada (sandbox) al directorio de trabajo.

**La página nunca almacena el token.** La página lee el token de `?token=…` —la URL que `soup ui` abre e imprime, y la URL telefónica de `--public`— siempre que un bind que no sea loopback deje ese parámetro en la URL; en un bind de loopback, la cookie de más abajo ya lo ha consumido. En caso contrario, pide el token la primera vez que se rechaza una petición y luego lo mantiene solo en la memoria de la página: nunca en `sessionStorage`, `localStorage`, una cookie ni una propiedad de `window`. Sin la cookie de sesión de más abajo, una recarga o una pestaña nueva por tanto lo vuelven a pedir; pega el token que imprimió `soup ui` (toda la línea `Authorization: Bearer …` también funciona). Si cancelas el aviso, la página queda sin sesión hasta tu siguiente clic, que lo vuelve a pedir. Eso es deliberado: un token persistido donde el script de la página pueda leerlo convertiría un futuro error de renderizado en una filtración del token. No protege contra el script que ya se ejecuta en la página; la política de contenido de más abajo es la defensa en ese caso.

**Cookie de sesión en un bind de loopback (#1191).** Abrir `http://127.0.0.1:<PORT>/?token=<TOKEN>` comprueba el token, establece una cookie `HttpOnly; SameSite=Strict; Path=/` llamada `soup_ui_session_<PORT>` y redirige a la misma URL sin `token`. La cookie contiene un id de sesión aleatorio, nunca el token, así que el script de la página no puede leer una credencial, y una recarga o una pestaña nueva en ese navegador siguen con la sesión iniciada. La cookie pertenece al nombre de host que abriste: iniciar sesión en `127.0.0.1` no inicia sesión en `localhost`, así que sigue usando el mismo. No tiene caducidad, así que termina cuando termina la sesión del navegador, y el servidor guarda los ids solo en memoria, así que cada reinicio de `soup ui` termina todas las sesiones; abre de nuevo la URL `?token=` para volver a iniciar sesión. Una petición autenticada por la cookie que no sea `GET` o `HEAD` debe llevar un `Origin` que nombre exactamente este servidor (esquema, host y puerto) o se rechaza con 403: todos los puertos de `127.0.0.1` cuentan como un solo sitio, así que `SameSite=Strict` por sí solo dejaría que una página servida por otra app local en un puerto distinto use la cookie. La cabecera `Authorization: Bearer` funciona exactamente igual que antes y no necesita `Origin`, así que los scripts y los clientes de API no cambian. Un token incorrecto no establece nada y carga la página como antes. Un navegador que bloquea las cookies para `127.0.0.1` recibe el aviso del token tras la redirección, porque el token ya ha salido de la URL; pega allí el token. Con `--public` (o cualquier `--host` que no sea loopback) no se establece ni se acepta ninguna cookie: la página del teléfono mantiene el token en la memoria de la página, y una recarga lo vuelve a pedir o vuelves a escanear el código QR. **Lo que esto no cubre:** las cookies no están aisladas por puerto, así que cualquier otro servidor en `127.0.0.1` al que tu navegador envíe una petición del mismo sitio también recibe la cookie de sesión. Un servidor así puede reproducirla fuera de un navegador con cualquier `Origin`, ya que la comprobación de `Origin` solo detiene a los navegadores, y actuar como tú en la interfaz, incluido iniciar entrenamientos, hasta que `soup ui` se reinicie.

Los cuerpos de petición con entrada YAML están limitados a 1 MiB en `/api/config/validate`, `/api/train/start` y `/api/config/from-form`. Los cuerpos más grandes devuelven HTTP 413 antes de la validación JSON/YAML. El servidor comprueba tanto `Content-Length` como los bytes realmente recibidos, así que las peticiones por chunks y las cabeceras que declaran menos de lo real no pueden saltarse el límite.

**Política de contenido.** Cada respuesta de la interfaz web lleva una `Content-Security-Policy` que permite scripts solo desde el propio origen de la interfaz y el archivo Chart.js fijado (sin script inline, sin `eval`), además de `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` y `X-Frame-Options: DENY`. Chart.js se carga con un hash de integridad de subrecurso (Subresource Integrity), así que el navegador lo rechaza si el CDN sirve bytes distintos. El marcado de la interfaz no lleva manejadores de eventos inline, y cada valor derivado del servidor o del dataset se escapa antes de renderizarse. Las páginas `/docs`, `/docs/oauth2-redirect` y `/redoc`, accesibles solo por loopback, son la única excepción a la cabecera de política, porque la documentación interactiva de FastAPI arranca desde un script inline.

**La documentación interactiva de la API es solo por loopback.** `/openapi.json`, `/docs`, `/docs/oauth2-redirect` y `/redoc` se sirven en un bind de loopback y están **ausentes** (404) en cualquier otro, incluido `soup ui --public`. Esto es deliberado y no incidental: el esquema no expone datos de ejecuciones, configuración ni logs, pero sí describe cada ruta, parámetro y forma de petición/respuesta, y en un bind de LAN eso es reconocimiento gratis. Ponerlos detrás del token se rechazó: `/docs` es una navegación del navegador y Swagger no puede adjuntarle una cabecera Bearer, así que protegerlos rompería la página para un desarrollador y aun así dejaría `/openapi.json` legible por cualquier cliente HTTP. Si necesitas el esquema con un bind público, léelo desde una instancia por loopback de la misma versión.

```bash
# Puerto personalizado, no abrir el navegador automáticamente
soup ui --port 8080 --no-browser
```


## Log de trazas del servidor de inferencia

`soup serve --trace-log <path>` escribe un log JSONL pasivo de solo anexado por cada chat completion:

```bash
soup serve --model ./out --trace-log ./serve-trace.jsonl --trace-log-cap-mb 100
```

Cada línea: `{"ts": ..., "prompt": ..., "response": ..., "latency_ms": ..., "tokens": ...}`. Se valida la contención de rutas, hay un tope de rotación fijo (por defecto 100 MB, se conserva una copia de respaldo), se rechazan symlinks en la ruta de respaldo (defensa contra TOCTOU), y las formas de token `hf_*` / `sk-*` / `Bearer …` se censuran a `<redacted>` antes de escribir. Los fallos (disco lleno, errores de serialización) nunca tumban el manejador de peticiones.


## Soup Quantize — alias ergonómico de exportación

```bash
soup quantize ./out --to gguf --bits 4
soup quantize ./out --to onnx -o ./out-onnx
```

Imprime la invocación equivalente de `soup export …` (escapada mediante `shlex.quote`) para copiar y pegar. Deliberadamente NO llama a `soup export` en proceso: los comandos de Typer no son seguros para reentrar. `--to awq` y `--to gptq` todavía imprimen su comando, seguido de un aviso de que los dos formatos están obsoletos y se eliminarán en Soup v0.77.


## Proxy de Llama.cpp

```bash
soup llama --help                  # lista los subcomandos admitidos
soup llama cli -m model.gguf -p "Hello"
soup llama gguf-split --merge a.gguf b.gguf out.gguf
soup llama server -m model.gguf
soup llama quantize model.gguf model-q4_k_m.gguf q4_K_M
```

### `soup llama quantize`

Reenvía cada argumento final al binario `llama-quantize` que esté en el `PATH` (las mismas reglas de allowlist cerrada / entorno filtrado que los demás subcomandos de `soup llama`). Úsalo cuando ya tengas un GGUF y quieras el cuantizador de llama.cpp directamente, en lugar de pasar por `soup export --format gguf` / `soup quantize`.

Allowlist cerrada: `cli` / `mtmd-cli` / `gguf-split` / `server` / `quantize`. Reenvía al binario `llama-*` del PATH (`shutil.which`) con un **entorno hijo filtrado**: `HF_TOKEN` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` y otros secretos se descartan antes del exec; solo se reenvían `PATH` / `HOME` / `USER` / locale + las variables reconocidas por llama.cpp `LLAMA_CPP_HOME` / `GGML_*` / `OMP_NUM_THREADS`.


## Estadísticas de latencia de cola + temporizador de llamadas a herramientas

```python
from soup_cli.utils.tail_latency import summarise_latency
from soup_cli.utils.tool_outputs import ToolOutputsBuffer, ToolCallTimer

stats = summarise_latency([12.3, 14.1, 9.7, 18.8, 11.2])
# TailLatencySummary(count=5, mean=..., p50=..., p95=..., p99=..., ema=...)

buffer = ToolOutputsBuffer()
with ToolCallTimer(buffer, name="fetch_url") as timer:
    timer.set_output("...")
```

Percentiles EMA + interpolación lineal en Python puro (tope anti-DoS: `MAX_SAMPLES=1_000_000`). `ToolOutputsBuffer` es un anillo `collections.deque(maxlen=1000)` thread-safe con vistas previas truncadas; `ToolCallTimer` registra duración / salida / error por invocación para las ejecuciones de SFT con tool-calling.


## Autopiloto de despliegue

Elige la combinación óptima de PEFT + cuantización + decodificación especulativa para tu hardware objetivo en un solo comando:

```bash
soup deploy autopilot --target rtx-4090-24gb --base meta-llama/Llama-3.2-1B
# Escribe:
#   deploy_autopilot.yaml  — receta soup.yaml lista para entrenar
#   deploy_autopilot.sh    — script de shell de despliegue planificado
```

Los perfiles vienen incluidos sin configuración adicional para Apple Silicon (`mac-m3`, `mac-m4-pro`), NVIDIA de consumo (`rtx-3060-12gb`, `rtx-4090-24gb`), móvil (`iphone-16`, `pixel-9`), runtimes locales (`ollama-local`, `lm-studio`) y nube (`runpod-a100`, `hf-jobs-h100`). `--list` muestra la tabla completa. Cada perfil es una dataclass congelada con allowlists cerradas en runtime / quant / PEFT: los valores de configuración incorrectos fallan en tiempo de importación. El bash generado usa `shlex.quote` en la ruta del modelo y las escrituras están protegidas por contención al cwd + rechazo de TOCTOU con `os.lstat + S_ISLNK`.


## Agent Forge

Convierte directamente un OpenAPI 3.x, un manifiesto de servidor MCP o un JSON de introspección de GraphQL en un dataset SFT de tool-calling, sin etiquetado manual ni andamiaje:

```bash
# 1. Analiza la spec + sintetiza un dataset de tool-calling
soup agent synth --spec api.yaml --output ds.jsonl --examples-per-endpoint 4

# 2. Planifica la ejecución de entrenamiento (imprime la invocación de soup train)
soup agent train --spec api.yaml --base meta-llama/Llama-3.2-1B

# 3. Puntúa las predicciones del modelo contra el catálogo de la spec
soup agent eval --spec api.yaml --predictions preds.jsonl
```

Cada fila del dataset sintetizado es `{messages: [user, assistant_with_tool_call], tool: <name>, source_endpoint: <path>}`. Las cadenas `$ref` en OpenAPI se dejan opacas (sin resolución externa; defensa contra SSRF de lectura de archivos), solo `yaml.safe_load`, tope de 5 MiB para la spec, tope de 10 000 endpoints, escritura JSONL atómica mediante tempfile preparado + `os.replace`. `eval` impone un tope de 1 000 000 de líneas en las predicciones y rechaza symlinks en cada límite de lectura/escritura.


## Selección automática del SDK de HF Space

Cuando despliegas un directorio de plantilla de Space personalizado, Soup ahora elige `space_sdk="streamlit"` / `"gradio"` a partir del `requirements.txt` renderizado:

```bash
soup deploy hf-space --space my-org/my-app --model my-org/my-model --template-dir ./my-template
```

Si `requirements.txt` lista `streamlit`, el Space se crea con el SDK de Streamlit. En caso contrario (sin requirements, gradio listado, etc.), Soup recurre al valor por defecto de Gradio. El HF Hub también admite los SDK `docker` y `static`, pero esos no se pueden inferir solo desde `requirements.txt`: usa las plantillas integradas o proporciona una personalizada con una anulación `--sdk` explícita.


## Convertidor de la API Anthropic Messages

Convertidores en Python puro entre las formas de payload de OpenAI chat-completions y de Anthropic Messages:

```python
from soup_cli.utils.anthropic_messages import to_anthropic, from_anthropic

anthropic_payload = to_anthropic({
    "model": "claude-3-5-sonnet",
    "messages": [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ],
    "max_tokens": 256,
})
```

Varios mensajes `system` se unen con `\n\n`. El rol `tool` con contenido estructurado (lista) se concatena en un único bloque de texto `tool_result`, nunca se descarta en silencio. `max_tokens` limitado a 16384, `temperature` acotada a `[0.0, 2.0]`. El endpoint `/v1/messages` en vivo dentro de `soup serve` llega en v0.45.1.


## Herramientas del lado del servidor

```python
from soup_cli.utils.server_tools import (
    SUPPORTED_TOOLS, WebSearchConfig, is_domain_allowed, validate_web_search_config,
)

# SUPPORTED_TOOLS == frozenset({"python", "bash", "web_search"})
config = WebSearchConfig(
    domain_allowlist=("example.com", ".docs.example.com"),
    rate_limit_per_minute=30,
)
validate_web_search_config(config)
is_domain_allowed("a.docs.example.com:443", config.domain_allowlist)  # True
is_domain_allowed("[::1]", config.domain_allowlist)                   # False
```

`python` y `bash` reutilizan el sandbox RLVR de v0.25.0; `web_search` está protegido por una allowlist de dominios explícita (por defecto vacía = denegar todo). `is_domain_allowed` quita los sufijos `:port` antes de comparar y rechaza literales IPv6, de modo que `Host: api.example.com:443` coincide con `api.example.com`. Endpoints HTTP de herramientas en vivo en v0.45.1.


## Endpoint de Anthropic Messages

`soup serve --backend transformers` y `soup serve --backend vllm` exponen una ruta POST `/v1/messages` que acepta payloads con la forma de Anthropic Messages (el backend **SGLang** no lo hace: solo sirve `/v1/chat/completions`, `/v1/models` y `/health`):

```bash
curl http://localhost:8000/v1/messages -H "Content-Type: application/json" -d '{
  "model": "my-model",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 64
}'
```

Las peticiones sin streaming devuelven sobres con la forma de Anthropic. El streaming (`stream: true`) devuelve SSE con la forma de eventos de Anthropic con eventos `message_start` → `content_block_delta` → `message_delta` + `message_stop`. Los errores de validación devuelven un cuerpo 400 genérico `"Invalid request"` con los detalles registrados del lado del servidor en DEBUG. CORS restringido solo a loopback (`localhost` / `127.0.0.1`) en ambos backends.


## Decodificación especulativa por N-gram

Cuando un servidor se configura con un `NgramSpecConfig`, cada chat completion reenvía `prompt_lookup_num_tokens=N` a `model.generate(...)` (decodificación prompt-lookup de HF Transformers ≥ 4.38; no se necesita modelo draft). Es mutuamente excluyente con un `assistant_model` real; si ambos están definidos, gana el modelo draft real.


## Endpoints de herramientas del lado del servidor

v0.53.7 incorpora llamadas a herramientas en vivo del lado del servidor en `soup serve`. Ahora hay tres rutas POST disponibles en `soup serve`:

- **`POST /v1/tools/python`** — ejecuta código Python proporcionado por el llamador en un subproceso aislado. Limitado a 512 MB de memoria, 5 segundos de tiempo de reloj de pared y un tope de concurrencia de un solo worker.
- **`POST /v1/tools/web_search`** — busca en la web y devuelve resultados filtrados por dominio como `[{url, title, snippet}]` con los snippets saneados (se eliminan los bytes nulos). Denegación por defecto mediante `WebSearchConfig.domain_allowlist`.

- **`POST /v1/tools/bash`** — ejecuta un comando bash con estricto aislamiento de red a nivel de SO (`unshare` en los runtimes Linux compatibles, `sandbox-exec` en macOS). Exige autenticación obligatoria con token Bearer cuando se enlaza a un host que no sea loopback, un tiempo de reloj de pared de 5 segundos, un entorno mínimo depurado de secretos y un límite combinado de 10KB de stdout/stderr en streaming antes de matar el proceso. Devuelve el stdout, el stderr, el código de salida y el estado de timeout reales. Esto no es un sandbox de sistema de archivos: el proceso hijo conserva el acceso de lectura del proceso del servidor a los archivos del sistema legibles por todos. El endpoint falla en cerrado con HTTP 501 cuando el aislamiento estricto del SO no está disponible (incluidos Windows o contenedores Linux restringidos).

Las rutas de herramientas, `/v1/thumbs` y activar/desactivar adaptadores aceptan peticiones solo cuando la cabecera `Host` nombra la dirección enlazada (cualquier nombre de loopback para un bind de loopback) y cualquier cabecera `Origin` nombra la misma; de lo contrario responden 421 o 403.

Las rutas de generación (`/v1/chat/completions`, `/v1/messages`) y el listado de adaptadores (`GET /v1/adapters`) llevan la mitad más estrecha de esa comprobación: una petición cuyo `Origin` nombre otro sitio se rechaza con 403, de modo que una página que el operador solo visita no puede dirigir la generación en su servidor ni enumerar los adaptadores cargados; CORS por sí solo solo detendría a esa página *leyendo* la respuesta, no enviando la petición. Las peticiones que no llevan `Origin` en absoluto (curl, los SDK de OpenAI/Anthropic, un proxy inverso) no se ven afectadas, y `Host` no se comprueba en estas rutas, así que un proxy todavía puede ponerlas por delante bajo su propio hostname.

**Qué no cubre `--tool-auth-token`.** `soup serve` rechaza un `--host` que no sea loopback sin `--tool-auth-token`, pero el token solo se comprueba en las rutas de herramientas, `/v1/thumbs` y las rutas de adaptadores. `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/health` y `/metrics` no aceptan token en ningún bind, así que responden a todo cliente que pueda alcanzar el puerto; pon delante del servidor un proxy que autentique si eso no es lo que quieres. Con `--backend vllm`, `sglang` o `mii` no hay rutas de herramientas ni de adaptadores, así que ninguna ruta comprueba el token. En un bind comodín (`0.0.0.0`, `::`) no hay un único nombre contra el que comparar `Host`, así que las rutas que en otros casos lo comparan solo comprueban que una cabecera `Origin`, cuando se envía, nombre el mismo host que `Host`. `soup serve` imprime estos puntos al arrancar siempre que se enlace más allá de loopback.
