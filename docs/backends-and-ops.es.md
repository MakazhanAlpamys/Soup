# Backends, plataforma y operaciones

> 🇪🇸 Traducción al español de [`backends-and-ops.md`](backends-and-ops.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> Backends MLX/Unsloth, hubs alternativos, integración con HF Hub, autopilot, seguimiento de experimentos, plan/apply, lockfiles de entorno, ajuste de hardware, completado de shell, el sistema de plugins y los comandos de utilidad independientes.

**Contenido:**

- [Autopilot (sin configuración)](#autopilot-sin-configuración)
- [Apple Silicon (backend MLX)](#apple-silicon-backend-mlx)
- [Backend Unsloth (entrenamiento 2-5x más rápido)](#backend-unsloth-entrenamiento-2-5x-más-rápido)
- [Chatea con tu modelo](#chatea-con-tu-modelo)
- [Sube a HuggingFace](#sube-a-huggingface)
- [Integración profunda con HuggingFace Hub](#integración-profunda-con-huggingface-hub)
- [Reanudar el entrenamiento](#reanudar-el-entrenamiento)
- [Gestión y limpieza de ejecuciones](#gestión-y-limpieza-de-ejecuciones)
- [Hubs de modelos alternativos](#hubs-de-modelos-alternativos)
- [Integración con TensorBoard](#integración-con-tensorboard)
- [Integración con Weights & Biases](#integración-con-weights--biases)
- [Recetas listas para usar](#recetas-listas-para-usar)
- [Búsqueda de hiperparámetros](#búsqueda-de-hiperparámetros)
- [Comparación de modelos](#comparación-de-modelos)
- [Demo de inicio rápido](#demo-de-inicio-rápido)
- [Verificación de estado](#verificación-de-estado)
- [Información de versión](#información-de-versión)
- [Manejo de errores](#manejo-de-errores)
- [Seguimiento de experimentos](#seguimiento-de-experimentos)
- [Extras de profiling](#extras-de-profiling)
- [Configuración de VS Code (`.vscode/launch.json`)](#configuración-de-vs-code-vscodelaunchjson)
- [Observabilidad y UX de desarrollo](#observabilidad-y-ux-de-desarrollo)
- [Monitor de GPU en vivo](#monitor-de-gpu-en-vivo)
- [Soup Fetch — ejemplos incluidos](#soup-fetch--ejemplos-incluidos)
- [Delinearizador de Llama 4](#delinearizador-de-llama-4)
- [Guardado ordenado con Ctrl+C](#guardado-ordenado-con-ctrlc)
- [Archivo de activación Checkpoint-Now](#archivo-de-activación-checkpoint-now)
- [Asistente del wizard de onboarding](#asistente-del-wizard-de-onboarding)
- [Configuración de sweep independiente](#configuración-de-sweep-independiente)
- [Hubs de modelos alternativos (ModelScope / Modelers)](#hubs-de-modelos-alternativos-modelscope--modelers)
- [Trackers de experimentos (MLflow / SwanLab / Trackio)](#trackers-de-experimentos-mlflow--swanlab--trackio)
- [Telemetría (opt-in)](#telemetría-opt-in)
- [Sistema de plugins](#sistema-de-plugins)
- [Catálogo de integraciones externas](#catálogo-de-integraciones-externas)
- [Plugins avanzados para el trainer](#plugins-avanzados-para-el-trainer)
- [Callbacks de plugins de Soup](#callbacks-de-plugins-de-soup)
- [Plan y apply estilo Terraform (`soup plan` / `soup apply`)](#plan-y-apply-estilo-terraform-soup-plan--soup-apply)
- [Lockfile de entorno hermético (`soup env`)](#lockfile-de-entorno-hermético-soup-env)
- [Calculadora de ajuste de hardware](#calculadora-de-ajuste-de-hardware)
- [Completado de shell (`soup completions`)](#completado-de-shell-soup-completions)
- [Asesor de licencias (`soup license-advisor`)](#asesor-de-licencias-soup-license-advisor)

---

## Autopilot (sin configuración)

Omite el YAML por completo. Dale a Autopilot un modelo base, un dataset y un objetivo: analiza tus datos, el modelo y el hardware, y luego elige por ti la tarea, la cuantización, el rango de LoRA, el learning rate, los epochs y las banderas de rendimiento.

```bash
# Sin configuración: elige todo automáticamente
soup autopilot --model meta-llama/Llama-3.1-8B-Instruct \
               --data ./data/train.jsonl \
               --goal chat

# Otros objetivos: chat | code | reasoning | instruct | vision
soup autopilot --model Qwen/Qwen2.5-7B --data ./data/math.jsonl --goal reasoning

# Limita a un presupuesto de GPU (1GB a 1TB)
soup autopilot --model <id> --data d.jsonl --goal chat --gpu-budget 24GB

# Previsualiza la configuración generada sin ejecutar
soup autopilot --model <id> --data d.jsonl --goal chat --dry-run
```

Autopilot escribe un `soup.yaml` listo para ejecutar. Edítalo a mano si hace falta y luego `soup train`.

Para los objetivos que entrenan con SFT, Autopilot activa FlashAttention y Liger solo cuando la GPU tiene compute capability 8.0 o superior y el paquete está instalado. Cuando la tarjeta califica pero falta un paquete, el panel Autopilot Decisions nombra el paquete y el comando que lo instala.


## Apple Silicon (backend MLX)

Ajusta en Macs M1-M4 mediante el framework [MLX](https://github.com/ml-explore/mlx) de Apple, sin CUDA y sin emulación.

```bash
# Instala el soporte de MLX
pip install "soup-cli[mlx]"
```

Para SFT con datos JSONL, JSON o CSV locales, `[mlx]` es suficiente por sí solo.
También se puede instalar con `[train]` cuando el mismo entorno necesita el
backend PyTorch/TRL Transformers: ambos extras ahora comparten el rango
`transformers>=5.16.1,<6` admitido. Una fuente de `datasets` de Hugging Face o
un dataset en streaming sigue necesitando `datasets`, porque esa fuente de datos
es dueña de la dependencia; la ruta de archivo local de abajo no.

`detect_device()` y `get_gpu_info()` reconocen Apple Silicon cuando se define
`backend: mlx`, conservando `training.quantization: 4bit` para checkpoints
precuantizados de `mlx-community` en lugar de degradarlos silenciosamente a
`none` ([#423](https://github.com/MakazhanAlpamys/Soup/issues/423)). El preflight
analítico de VRAM con forma de CUDA se omite en la ruta MLX porque la memoria
unificada de Apple la administra Metal, no un pool de VRAM CUDA fijo.

```yaml
base: mlx-community/Llama-3.2-3B-Instruct-4bit
task: sft
backend: mlx  # solo Apple Silicon

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  lora:
    r: 16
    alpha: 32
```

El backend MLX admite SFT. `backend: mlx` con `task: dpo` o `task: grpo` se rechaza al cargar la configuración, con un error que nombra la tarea: el `mlx-lm` upstream no trae ningún helper de entrenamiento DPO/GRPO, así que esos wrappers existen solo como respaldo para quienes llaman saltándose la validación de la configuración. Requiere `mlx-lm >= 0.31.3`. Usa `soup recipes search mlx` para configuraciones listas para Apple Silicon.

#### Optimizadores y planificadores

`training.optimizer`, `training.scheduler`, `training.warmup_ratio` y
`training.weight_decay` se respetan en el backend MLX
([#686](https://github.com/MakazhanAlpamys/Soup/issues/686)). Antes de eso se
validaban, se aceptaban y se descartaban: cada ejecución construía un AdamW
pelado a learning rate constante, sin importar lo que dijera la receta.

Como ahora se respetan en lugar de ignorarse, un ajuste que MLX no puede expresar
**se rechaza al construir el optimizador**, en vez de sustituirse en silencio:

| ajuste | MLX acepta | en caso contrario |
|---|---|---|
| `optimizer` | `adamw_torch`, `adamw_hf`, `adamw_torch_fused`, `sgd`, `adafactor`, `adagrad`, `rmsprop`, `muon` | se rechaza por nombre, listando lo disponible |
| `scheduler` | `cosine`, `linear`, `constant`, `constant_with_warmup` | se rechaza, nombrando el antiguo comportamiento de tasa constante |
| `weight_decay` | cualquier valor en todos los optimizadores anteriores excepto `adagrad` y `rmsprop` | un valor distinto de cero en `adagrad` / `rmsprop` se rechaza: esos constructores de MLX no toman `weight_decay`, y descartarlo en silencio es el defecto anterior |

La allowlist de optimizadores de Soup (`utils.optimizer_zoo`) es mucho más amplia
que cualquier cosa que traiga MLX, así que la mayoría de los valores válidos no tiene
equivalente en MLX. Ejecuta esas recetas en el backend transformers.

El planificador cuenta **actualizaciones del optimizador**, no iteraciones: mlx-lm llama
a `optimizer.update()` una vez cada `gradient_accumulation_steps`, así que un warmup de
`warmup_ratio × (iters // gradient_accumulation_steps)` es lo que realmente se ejecuta.
El plan efectivo se escribe en `adapter_config.json` (`optimizer`,
`scheduler`, `warmup_updates`, `total_updates`, `weight_decay`, `peak_lr`), así que
lo que se ejecutó se puede recuperar del directorio de salida.

`--resume auto` encuentra los checkpoints `NNNNNNN_adapters.safetensors` numerados por step de mlx-lm y arranca en caliente los pesos LoRA desde ellos ([#634](https://github.com/MakazhanAlpamys/Soup/issues/634)). Esto restaura solo los pesos del adaptador, no el estado del entrenamiento: el trainer LoRA de mlx-lm no expone estado del optimizador ni conteo de steps, así que el entrenamiento reinicia desde el step 0 sin importar hasta dónde llegó el checkpoint. Consulta [Reanudar el entrenamiento](#reanudar-el-entrenamiento) más abajo para la forma específica de los checkpoints de MLX.

### Transformers en MPS

La ruta regular `backend: transformers` puede ejecutar más que la superficie solo-SFT
de MLX. En un runtime MPS en vivo que acepta bfloat16, Soup activa autocast BF16
para los trainers de texto validados por hardware: SFT, DPO, GRPO/RLVR, modelado de recompensa
y PRM.
La decisión proviene de una sonda de asignación MPS en vivo, no de una suposición
sobre la versión de macOS. CPU y un runtime MPS no disponible o más antiguo se quedan en FP32; FP16 no se
selecciona en MPS.

Para checkpoints BF16, las ejecuciones residentes de SFT, DPO, GRPO y modelo de recompensa conservan
los pesos base congelados en BF16 mientras mantienen los parámetros LoRA en FP32. PRM usa
autocast BF16 pero conserva deliberadamente pesos maestros FP32: cargar su
base entrenable en BF16 hace que el optimizador de Metal aborte porque sus dtypes de acumulador
y matriz de destino difieren. Esta política se validó con ejecuciones de un
solo step en Apple Silicon para las cinco tareas. La validación de GRPO usa generación local de
Transformers (`use_vllm: false`) y RLVR determinista; vLLM sigue siendo
una ruta opcional orientada a CUDA. Los demás trainers de Transformers se quedan en FP32 en MPS
hasta que sus kernels específicos de tarea reciban cobertura de hardware equivalente.


## Backend Unsloth (entrenamiento 2-5x más rápido)

Usa el backend [Unsloth](https://github.com/unslothai/unsloth) para un entrenamiento mucho más rápido y hasta 80% menos VRAM:

```bash
# Instala el soporte de unsloth
pip install "soup-cli[fast]"
```

Luego agrega una línea a tu configuración:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
backend: unsloth  # 2-5x más rápido, -80% VRAM

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

Funciona con las tareas que tienen setup de unsloth: SFT (texto; SFT de visión y audio siguen ejecutando el setup de transformers), DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, BCO, preference, Pretrain, Embedding y TTS. Las demás (`reward_model`, `prm`, `classifier`, `reranker`, `cross_encoder`, `distill`, `unlearn`, `moe_lora_routing`, `online_dpo` y `asr`) no tienen setup de unsloth, así que `backend: unsloth` se rechaza al cargar la configuración para ellas (#1357). Si unsloth está instalado pero no habilitado, Soup lo sugiere, pero solo para una configuración que igual cargaría con `backend: unsloth` (#1736): una configuración que cualquiera de estos rechazos detendría no recibe ninguna sugerencia.

> **Consejo:** Soup detecta unsloth automáticamente. Cuando está instalado, `soup train` imprime una pista si aún no lo habilitaste y tu configuración cargaría con él.


## Entrenamiento en GPU en la nube

¿Sin GPU local? `soup train --cloud modal|lambda` genera un controlador específico del
proveedor a partir de tu `soup.yaml`. El YAML de configuración se incrusta en base64 como **datos**; las
credenciales se leen del entorno solo cuando empieza un envío en vivo.

```bash
pip install "soup-cli[modal]"   # solo se necesita para el envío en vivo

# Solo plan (por defecto): escribe el stub + imprime el comando `modal run`.
soup train --config soup.yaml --cloud modal --gpu a100

# Envía en vivo (autentícate una vez con `modal setup`, o define
# MODAL_TOKEN_ID + MODAL_TOKEN_SECRET).
soup train --config soup.yaml --cloud modal --gpu a100 --cloud-submit
```

`--gpu` acepta: `t4` / `l4` / `a10g` / `a100` / `a100-80gb` / `l40s` / `h100`. El
`soup_modal_app.py` generado construye una imagen con `soup-cli[train]` fijado a tu versión en ejecución, escribe
la configuración incrustada dentro del contenedor y ejecuta `soup train` en la GPU elegida.

Las salidas de la ejecución se escriben en el volumen `soup-outputs` de Modal (creado en el primer uso), en un directorio
con el nombre de la ejecución (`soup-<12 hex>`), así los checkpoints sobreviven al contenedor. Cuando la ejecución termina,
incluso después de una ejecución fallida, la app descarga cada archivo de ese directorio en tu directorio
de salida local e imprime un comando `modal volume get soup-outputs /<run name> <output dir> --force`
que reintenta la descarga; ese comando coloca los archivos bajo `<output dir>/<run name>`.

### RunPod (planificado)

El soporte de RunPod está actualmente en desarrollo y quedó fuera del despacho en vivo del CLI a la espera de
salvaguardas automatizadas de ciclo de vida y terminación. Ejecutar `soup train --cloud runpod` informa al operador que
RunPod todavía no está activo y apunta a los backends de nube activos (`--cloud modal` y `--cloud lambda`).

### Lambda Cloud

Lambda usa una instancia en lugar de una función serverless. El controlador local generado envía un
script cloud-init libre de secretos como `user_data` de la API, lo espera por SSH, copia la salida
configurada de vuelta y solicita la terminación de la instancia en un bloque `finally`. El controlador consulta el estado
del entrenamiento por conexiones SSH cortas con sondas de keepalive en lugar de mantener una única sesión
persistente abierta. Tolera caídas de red transitorias y desconexiones hasta por 15 minutos con
backoff exponencial antes de terminar la instancia y evitar facturación descontrolada. Mantén el controlador corriendo
hasta que informe que la terminación tuvo éxito; apagar la máquina invitada no termina la facturación.
Presionar Ctrl+C durante `--cloud-submit` interrumpe el controlador, que igual ejecuta ese bloque
`finally`, y `soup` espera a que termine la instancia y salga en lugar de matarlo; la única
forma de saltarse esa limpieza es matar el propio proceso del controlador (por ejemplo con `kill` desde
otra terminal), lo que puede dejar la instancia corriendo.

Registra primero la mitad pública de una clave SSH en Lambda, luego define:

```bash
export LAMBDA_API_KEY=...
export LAMBDA_SSH_KEY_NAME=my-lambda-key
export LAMBDA_SSH_PRIVATE_KEY=/path/to/private-key
export LAMBDA_REGION=us-tx-1  # opcional; por defecto us-tx-1
soup train --config soup.yaml --cloud lambda --gpu a100 --cloud-submit
```

`--gpu` acepta: `a10` / `a100` / `a6000` / `h100`. Las rutas de salida de Lambda deben ser relativas para que el
controlador pueda copiar el artefacto de vuelta de forma segura. La clave de API se queda en el llamador y nunca se incrusta
en cloud-init ni en los logs de la instancia.

La ruta de envío a Lambda todavía requiere la checklist de validación en vivo pagada en el #264 antes de
que se pueda describir como validada por el proveedor. El renderizado solo-plan y los límites del ciclo de vida están
cubiertos por pruebas offline.


## Chatea con tu modelo

```bash
# Chatea con un adaptador LoRA (detecta el modelo base automáticamente)
soup chat --model ./output

# Especifica el modelo base de forma explícita
soup chat --model ./output --base meta-llama/Llama-3.1-8B-Instruct

# Ajusta la generación
soup chat --model ./output --temperature 0.3 --max-tokens 256
```


## Sube a HuggingFace

```bash
# Sube el modelo al HF Hub
soup push --model ./output --repo your-username/my-model

# Hazlo privado
soup push --model ./output --repo your-username/my-model --private

# Agrupa en una Collection
soup push --model ./output --repo your-username/my-model \
    --collection your-username/my-collection-abc123
```


## Integración profunda con HuggingFace Hub

Soup trata al HF Hub como un backend de artefactos de primera clase. Una variable de entorno, una bandera,
sin banderas de token que cablear: todas las operaciones respetan las credenciales de
`huggingface-cli login` por defecto.

```bash
# Hub autoalojado: se define una vez, todos los comandos van allí.
export HF_ENDPOINT=https://hf.internal.example.com

# Sube automáticamente cada checkpoint de save_steps a HF como una rama 'checkpoint-<N>'.
soup train -c soup.yaml --push-as your-username/my-model

# Reanuda desde la última rama subida arriba.
soup train -c soup.yaml --push-as your-username/my-model --hf-resume

# Sube un archivo JSONL local como repo de dataset de HF.
soup data push --input train.jsonl --hf-dataset your-username/my-dataset

# Envuelve tu modelo ajustado en un Space de chat de Gradio con un solo comando.
soup deploy hf-space \
    --model your-username/my-model \
    --space your-username/my-chat-space \
    --template gradio-chat

# O una app de Streamlit:
soup deploy hf-space \
    --model your-username/my-model \
    --space your-username/my-chat-space \
    --template streamlit-chat
```

**Flujo de autorreanudación:** si el entrenamiento se cae, la siguiente llamada a `soup train ... --push-as
... --hf-resume` toma la última rama `checkpoint-<N>` de tu repo de HF
y la descarga de vuelta a `output_dir`, luego reanuda, sin copiar /
pegar rutas de checkpoint a mano. La contención en el cwd y `local_dir_use_symlinks=False`
evitan que un repo manipulado se escape del sistema de archivos.

La **autenticación** sigue las convenciones estándar de HF: variable de entorno `HF_TOKEN` > `HUGGINGFACE_HUB_TOKEN`
> `~/.cache/huggingface/token` (la define `huggingface-cli login`) > `~/.huggingface/token`.
Sin banderas de token personalizadas. El `--token` obsoleto de `soup push` todavía funciona, pero emite
una advertencia.

La **ficha del modelo v2** se autogenera en el primer push: lee los sidecar
`training_config.yaml` / `soup.yaml` para mostrar `task` / `base` / `lr` /
`optimizer`, y acepta un scorecard de evaluación opcional (tabla markdown).
Los caracteres activos en Markdown dentro de nombres de tareas y puntajes se neutralizan para un
renderizado seguro en HF Hub.


## Reanudar el entrenamiento

Reanuda una ejecución de entrenamiento desde un checkpoint:

```bash
# Detecta automáticamente el último checkpoint del directorio de salida
soup train --config soup.yaml --resume auto

# Reanuda desde un checkpoint específico
soup train --config soup.yaml --resume ./output/checkpoint-500
```

`backend: mlx` escribe y reanuda una forma de checkpoint distinta: un
archivo `NNNNNNN_adapters.safetensors` numerado por step (o el
`adapters.safetensors` final) directamente bajo `output`, no un directorio
`checkpoint-N`. `--resume auto` y `--resume ./output/0011800_adapters.safetensors`
funcionan ambos; `--resume ./output/checkpoint-500` no, porque MLX nunca
escribe esa forma. Este es un arranque en caliente solo de pesos: el trainer LoRA de mlx-lm
no expone estado del optimizador ni conteo de steps, así que la ejecución reanudada empieza a contar
desde el step 0 sin importar hasta dónde llegó el checkpoint.

`task: unlearn` rechaza `--resume` / `--hf-resume`, porque no escribe ningún checkpoint.


## Gestión y limpieza de ejecuciones

El entrenamiento de LLM genera archivos de checkpoint enormes. Soup administra automáticamente una base de datos SQLite de la pérdida y las métricas de tu entrenamiento, lo que te permite recuperar espacio en disco de forma segura una vez que el entrenamiento termina.

```bash
# Lista todas las ejecuciones de entrenamiento históricas
soup runs

# Compara dos experimentos distintos lado a lado
soup runs compare run_202611... run_202612...

# Limpia de forma inteligente los checkpoints redundantes
# (Conserva el modelo final y el checkpoint con la pérdida más baja)
soup runs clean run_202611...

# Previsualiza el espacio que se recuperaría en TODOS los experimentos
soup runs clean --all --dry-run
```

Por defecto, el comando `clean` opera en "modo quirúrgico" (`--keep-weights`), borrando enormes archivos de estado del optimizador (`optimizer.pt`) de los checkpoints menores para ahorrar gigabytes, pero conservando sus pesos de evaluación livianos por si quieres cargarlos después. Pasa `--no-keep-weights` para borrar por completo los checkpoints que no son el mejor (el checkpoint con la pérdida más baja siempre se conserva); combínalo con `--dry-run` para ver qué se borraría primero. El checkpoint que se conserva entero es el que tiene la pérdida más baja registrada en su step. Cuando ningún checkpoint tiene una pérdida registrada, por ejemplo en una ejecución más corta que `logging_steps`, que no registra ninguna, es el último checkpoint, y `clean` lo dice.


## Hubs de modelos alternativos

Define `training.hub` en tu `soup.yaml` para descargar de / subir a un hub que no sea HuggingFace. Útil en regiones donde el HF Hub no es accesible o está bloqueado.

```yaml
training:
  hub: modelscope   # o 'modelers' (Openmind), por defecto 'hf'
```

Sobrescribe el endpoint con una variable de entorno:

```bash
export MODELSCOPE_ENDPOINT=https://my-mirror.example.com
export MODELERS_ENDPOINT=https://corp-modelers.internal   # HTTPS solo para direcciones que no sean loopback
soup train --config soup.yaml
```

El validador de endpoints sigue las mismas reglas de SSRF que `HF_ENDPOINT`: solo esquemas `http`/`https`; HTTP plano permitido solo para `localhost` / `127.0.0.1` / `::1`; IPs privadas y link-local (RFC1918, 169.254/16, etc.) rechazadas en HTTP plano. `backend: mlx` es incompatible con hubs que no sean HF (`mlx-lm` solo descarga del HF Hub).

Las descargas y subidas de ModelScope y Modelers pasan por adaptadores SDK en vivo con importación diferida. La
ruta de HF sigue siendo la predeterminada, y MLX sigue siendo solo-HF.

`soup push --hub modelscope` se autentica con `--token` o `MODELSCOPE_API_TOKEN`; `--hub modelers`
con `--token` o `MODELERS_TOKEN`. Ninguno lee `HF_TOKEN`.


## Integración con TensorBoard

Registra las métricas del entrenamiento en TensorBoard para visualización local:

```bash
# Activa el registro en TensorBoard (requiere: pip install tensorboard)
soup train --config soup.yaml --tensorboard

# Consulta los logs
tensorboard --logdir ./output/runs/
```

> **Nota:** `--tensorboard` y `--wandb` no se pueden usar juntos. Elige uno.


## Integración con Weights & Biases

Envía las métricas del entrenamiento a [W&B](https://wandb.ai/) para seguimiento de experimentos en la nube:

```bash
# Activa el registro en W&B (requiere: pip install wandb)
soup train --config soup.yaml --wandb
```

Asegúrate de que `WANDB_API_KEY` esté definida o ejecuta `wandb login` primero.


## Recetas listas para usar

80 configuraciones ya armadas para modelos populares, sin adivinar hiperparámetros:

```bash
# Lista todas las recetas
soup recipes list

# Previsualiza una receta
soup recipes show llama3.1-8b-sft

# Usa una receta (escribe soup.yaml)
soup recipes use llama3.1-8b-sft

# Busca por tarea o palabra clave
soup recipes search --task grpo
soup recipes search "reasoning"
soup recipes search --size 7b
soup recipes search "medical"
soup recipes search "vision"

# Verifica que cada configuración incluida pueda acoplar su adaptador LoRA (no se descargan pesos)
soup recipes verify
soup recipes verify --config soup.yaml
soup recipes verify --json    # una fila por configuración en stdout, para CI
```

**Qué cubre:**

| Categoría | Modelos |
|---|---|
| **SFT general / DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO / Embedding / Pretrain** | Llama 3.1 / 3.2 / 4, Qwen 2.5 / 3, Mistral, Gemma 3, Phi-4, DeepSeek R1 / V3 |
| **Visión (multimodal)** | Llama-3.2-Vision (11B + 90B), Pixtral-12B, Qwen2-VL (7B + 72B), InternVL 2.5, MiniCPM-V 2.6 |
| **Audio (voz)** | Qwen2-Audio, SeamlessM4T v2 (traducción), Whisper-large-v3 (ASR) |
| **Razonamiento** | Los 6 tamaños de DeepSeek-R1-Distill (Qwen 1.5B / 7B / 14B / 32B + Llama 8B / 70B), Qwen3-Coder 30B, razonamiento de Qwen3-30B-A3B, razonamiento de Phi-4 |
| **Pequeños / edge / móvil** | SmolLM2 (135M / 360M / 1.7B), Qwen2.5 (0.5B / 1.5B / 3B), Gemma 2 2B, Phi-3.5-mini, Llama-3.2 (1B / 3B) |
| **Especialistas de dominio** | BioMistral 7B, Meditron 7B (médico) — CodeLlama (13B / 70B), Magicoder 6.7B (código) — Mathstral 7B (matemáticas) — Llama-2-13b-finance (base estilo FinGPT) — Nemotron-4 340B |
| **Razonamiento multimodal** | Llama-3.2-Vision GRPO, Pixtral DPO |
| **Multi-GPU** | llama3-70b-fsdp2, qwen3-32b-zeropp, deepseek-v3-pipeline |
| **Apple Silicon (MLX)** | llama3.1-8b / qwen3-8b / gemma3-9b SFT-MLX |
| **Tool-calling / agentic** | qwen3-8b-tools, llama4-scout-tools |


## Búsqueda de hiperparámetros

Busca los mejores hiperparámetros:

```bash
# Búsqueda en grilla sobre learning rate y rango de LoRA
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --param lora_r=8,16,32

# Búsqueda aleatoria con un máximo de ejecuciones
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --strategy random --max-runs 5

# Previsualiza sin ejecutar — valida primero (#642): una clave de configuración desconocida
# se rechaza igual que la rechaza `train --dry-run`, y un --param que no nombra ningún
# campo de configuración sale con código distinto de cero antes de imprimir cualquier grilla
soup sweep --config soup.yaml --param lr=1e-5,2e-5 --param epochs=2,3 --dry-run

# Parada temprana: omite las ejecuciones restantes si la pérdida supera 1.5x la mejor
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --early-stop 1.5
```

Cada brazo construye el trainer que `soup train` construye para el `task` y el
`backend` de la configuración, así que un brazo sobre una configuración `task: distill` destila, y un brazo sobre
`backend: mlx` usa el wrapper de MLX.


## Comparación de modelos

Compara las salidas de dos modelos lado a lado:

```bash
# Compara con prompts en línea
soup diff --model-a ./model_v1 --model-b ./model_v2 --prompt "Explain gravity"

# Compara con un archivo de prompts
soup diff --model-a ./base --model-b ./finetuned --prompts test_prompts.jsonl

# Guarda los resultados
soup diff --model-a ./a --model-b ./b --prompts prompts.txt --output results.jsonl
```


## Demo de inicio rápido

Ejecuta una demo completa con un solo comando: crea datos de ejemplo, la configuración y entrena un modelo diminuto:

```bash
# Demo completa (crea datos + configuración + entrena TinyLlama)
soup quickstart

# Solo crea los archivos sin entrenar
soup quickstart --dry-run

# Omite la confirmación
soup quickstart --yes
```


## Verificación de estado

Revisa tu entorno en busca de problemas de compatibilidad:

```bash
soup doctor [--nccl]
```

Muestra: versión de Python, disponibilidad de GPU, recursos del sistema (RAM/Disco), todas las versiones de dependencias y sugerencias de arreglo. Usa `--nccl` para medir y verificar el ancho de banda de comunicación multi-GPU contra los techos esperados del hardware.

Con `soup doctor --config soup.yaml`, el informe de soporte declarado lista solo los
ajustes activos escritos explícitamente que difieren de sus valores por defecto del esquema.
Escribir un valor por defecto cuenta como dejarlo sin escribir, así que un volcado YAML completo y su
fuente mínima tienen el mismo informe; `seed: 0` igual cuenta porque el
valor por defecto de seed no está definido. Los ajustes rechazados al cargar la configuración no se listan como filas del doctor:
sus valores activos fallan antes de que el informe pueda ejecutarse. Este informe describe las
brechas revisadas, no una garantía de que todo ajuste no listado esté admitido.

### Diagnósticos de GPU

Cuando hay una GPU NVIDIA disponible, `soup doctor` informa su compute capability
y su familia de arquitectura, si la build de PyTorch instalada contiene soporte nativo
o PTX para esa arquitectura, y los umbrales de hardware/software para
BF16, FP8 y NVFP4. Una advertencia de arquitectura faltante incluye una pista de reinstalación
de PyTorch con CUDA habilitado, basada en la versión de CUDA que reporta el driver. Cuando
la versión del driver no se puede mapear a una rueda admitida, la pista sigue
siendo accionable sin construir una URL `whl/None`.


## Información de versión

```bash
# Versión básica
soup version

# Salida legible por máquina
soup version --json
# -> {"version": "0.26.0", "python": "3.11.5", "platform": "linux"}

# Información completa del sistema (útil para reportes de bugs)
soup version --full
# -> soup v0.26.0 | Python 3.11.5 | CUDA 12.1 | extras: serve, data

# Información completa del sistema en JSON
soup version --full --json
# -> {"version": "0.26.0", "python": "3.11.5", "platform": "linux", "torch": "2.2.0", ...}
```


## Manejo de errores

Soup muestra mensajes de error amigables por defecto (2-3 líneas con una sugerencia de arreglo). Para tracebacks completos:

```bash
# La bandera global va ANTES del comando
soup --verbose train --config soup.yaml

# Funciona con cualquier comando
soup --verbose eval benchmark --model ./output --benchmarks mmlu
```

> **Nota:** `--verbose` es una bandera global: debe ir **antes** del nombre del comando, no después.


## Claves de configuración desconocidas

Todos los modelos de configuración solían ejecutarse con el `extra="ignore"` por defecto de Pydantic, así que una clave que el
esquema no declaraba se descartaba sin decir nada. `soup train --dry-run` imprimía
"Config valid. Ready to train!", la ejecución salía con 0 y el ajuste solicitado simplemente
nunca se aplicaba: `quantizaton: none` entrenaba en 4 bits cuantizado cuando la precisión completa era
lo que pediste, `gradient_checkpoint: true` no hacía ningún checkpointing y `max_len: 512`
truncaba en 2048.

Cargar una configuración ahora rechaza cada clave que no puede ubicar, en **un** informe por carga,
con el campo que probablemente querías decir:

```
Config validation error:

  unknown config key 'data.max_len' - did you mean 'max_length' or 'video_maxlen'? Refused.
unknown config key 'training.quantizaton' - did you mean 'quantization' or 'quantization_aware'? Refused.
```

**Desde la v0.75 una clave de configuración desconocida rechaza la carga.** La v0.74 incluía el mismo informe
como una advertencia que nombraba esta fecha límite, así que hubo exactamente una versión de aviso:
a propósito, porque una configuración escrita contra un Soup más nuevo tiene que seguir corriendo en una
rueda más antigua por al menos una versión. El rechazo es igual en todos los lugares donde se construye un `SoupConfig`
a partir de un archivo o una cadena: `soup train` sale con 1 antes de que se importe la pila de entrenamiento,
`soup sweep` / `soup doctor --config` / `soup ship --config` / `soup plan` /
`soup apply` rechazan de la misma forma, y el cargador de la Web UI / API lanza `ValueError` con
el mismo texto (la Web UI lo muestra). Nada se pone por defecto y nada se adivina:
la sugerencia es una pista para ti, no una sustitución que haga el cargador. Un bloque `lora:`
de nivel raíz no es una clave desconocida: el esquema ha aceptado esa grafía y la ha movido bajo
`training` desde la v0.40.1, y el detector aplica el mismo remapeo antes de mirar.

Una configuración que nombra una clave que tu Soup instalado no tiene normalmente significa una de dos
cosas: un error de tipeo (toma la sugerencia), o un campo agregado después de que salió tu versión
(`soup version` contra el [registro de cambios](../CHANGELOG.md) dirá cuál). Una configuración
que también deba seguir siendo cargable en la v0.74 necesita que se quite la clave, no que se renombre: la v0.74
la advierte y la ignora, la v0.75 la rechaza, y ninguna la aplica.

**`soup sweep` nunca tuvo el periodo de advertencia.** Un `--param` que no coincide con ningún campo
de configuración es un error duro desde la v0.74:

```bash
soup sweep --config soup.yaml --param lora_rank=8,16   # el campo es training.lora.r
# sweep parameter does not match any config field: unknown config key 'lora_rank' - refused.
echo $?   # 1
```

Todo el sweep se rechaza antes de que arranque el primer brazo, y el comando sale con código distinto de cero, así que
un sweep por script o por CI falla en lugar de reportar una grilla de brazos que cada uno falló
por la misma razón. Un sweep cuyo knob barrido nunca se aplica produce brazos que son todos
idénticos, así que no hay ningún resultado parcialmente útil que preservar continuando.


## Seguimiento de experimentos

Cada ejecución de `soup train` se registra automáticamente en una base de datos SQLite local (`~/.soup/experiments.db`).

```bash
# Lista todas las ejecuciones de entrenamiento
soup runs

# Muestra información detallada + la curva de pérdida de una ejecución
soup runs show run_20260223_143052_a1b2

# Compara dos ejecuciones lado a lado
soup runs compare run_1 run_2

# Borra una ejecución
soup runs delete run_1

# Reproduce el resumen + la curva de pérdida de una ejecución antigua desde el historial
soup runs replay run_1
```

Cada ejecución completada también guarda un costo estimado (`$` por ejecución) calculado a partir del
nombre del dispositivo GPU capturado y la duración. `soup runs show` renderiza `—` para CPU /
MPS / GPUs desconocidas (nada de ceros fabricados).

A partir de la v0.71.5, la búsqueda de series de métricas que alimenta el replay (`ExperimentTracker.get_metric_series`)
cae de forma transparente a la tabla `eval_results` cuando una métrica no tiene filas
por step, así que puedes graficar una curva de puntaje de benchmark (p. ej. `mmlu`, `gsm8k`) igual que
graficas `loss`, sin que te importe qué tabla contiene la serie.

### Integraciones de trackers (--tracker mlflow / swanlab / trackio)

```bash
# Transmite las métricas a MLflow (define MLFLOW_TRACKING_URI con la URL de tu servidor)
soup train --config soup.yaml --tracker mlflow

# O SwanLab (nube o local)
soup train --config soup.yaml --tracker swanlab

# O Trackio (subida por lotes, apta para uso offline)
soup train --config soup.yaml --tracker trackio
```

`--tracker` es mutuamente excluyente con `--wandb` y `--tensorboard`. Soup
valida el nombre del tracker contra una allowlist cerrada (`mlflow` / `swanlab` /
`trackio` / `wandb` / `tensorboard` / `none`); el paquete upstream en sí lo
carga HF Trainer en tiempo de ejecución, así que instala por separado el que necesites:

```bash
pip install mlflow      # o: swanlab / trackio
```

### Telemetría (opt-in)

Soup incluye un payload de telemetría solo con información de hardware (versión de Soup + comando +
Python major.minor + SO + arquitectura + duración + ID anónimo distinto). Está **desactivada por defecto** y nunca
envía nombres de modelos, rutas de datasets ni contenidos de configuración.

Para aceptarla, define la variable de entorno:

```bash
SOUP_TELEMETRY=1 soup train --config soup.yaml
```

Cuando `SOUP_TELEMETRY` no está definida, es `0`, o tiene cualquier valor distinto de `1`/`true`/`yes`/`on`, Soup realiza
cero solicitudes de red de telemetría.

También puedes desactivar la telemetría explícitamente para una invocación específica con la bandera `--no-telemetry`:

```bash
soup --no-telemetry train --config soup.yaml
```

Cuando está habilitada, la telemetría hace un POST HTTP síncrono de tipo fire-and-forget con 1 segundo de timeout de conexión y lectura (sin contar la resolución DNS) al salir el comando. El identificador anónimo se guarda en `~/.soup/telemetry_id`; borrar `~/.soup/telemetry_id` lo regenera en la siguiente aceptación. Consulta la [Política de privacidad](#política-de-privacidad) para más detalles.


## Extras de profiling

Snapshots de memoria CUDA, rastreo de anomalías y una tabla de referencia de ancho de banda NCCL:

```python
from soup_cli.utils.profiling_v0_43 import (
    memory_snapshot_context, detect_anomaly_context, nccl_bandwidth_check,
)

with memory_snapshot_context("run-123") as path:
    train_step()
    # En CUDA, vuelca profiles/run-123.snapshot.pickle al salir.

with detect_anomaly_context():
    train_step()
    # torch.autograd.set_detect_anomaly(True)

result = nccl_bandwidth_check(
    gpu="h100", link="nvlink", measured_gb_per_sec=400.0,
)
# {'expected_gb_per_sec': 450.0, 'measured_gb_per_sec': 400.0,
#  'ratio': 0.8889, 'status': 'OK'}
```


## Configuración de VS Code (`.vscode/launch.json`)

Escritor de una sola pasada para una configuración de depurador sensata:

```python
from soup_cli.utils.vscode_setup import write_vscode_launch
write_vscode_launch(config_path="soup.yaml")
# Escribe ./.vscode/launch.json con entradas de `soup train` + pytest.
```

Se rechaza si es un symlink en la ruta de destino, sin importar `force=True`, para defender
contra symlinks precolocados que redirijan la escritura fuera del cwd.


## Observabilidad y UX de desarrollo

Herramientas que explican *por qué* una ejecución se portó mal en lugar de volcar un stack trace.

### `soup why`

Explicador heurístico: lee la ejecución más reciente (o la nombrada) y muestra
diagnósticos en lenguaje claro con próximos pasos concretos.

```bash
soup why                 # la ejecución más reciente
soup why run_2026_abc    # id de ejecución específico (o prefijo)
```

Detecta: pérdida NaN/Inf, meseta (≥30 steps con <0.5% de cambio), divergencia
(pérdida > 3× la inicial), norma de gradiente persistentemente alta, learning rate fuera de la
banda típica `[1e-6, 5e-3]`. Puramente basado en reglas, sin llamadas a modelos.

### `soup tui`

Panel Textual a pantalla completa. Dos paneles: lista de ejecuciones (izquierda) + detalle de la ejecución
seleccionada (derecha). `r` refresca, `q` sale.

```bash
pip install "soup-cli[tui]"
soup tui --refresh 1.0 --limit 50
```

### Auto-profiling — `soup train --profile`

Registra un Chrome-trace de `torch.profiler` sobre una ventana de steps tempranos (por defecto
`wait=1, warmup=1, active=5, repeat=1`). Salida: `<output>/profiles/<run_id>.trace.json`.
Ábrelo en `chrome://tracing` o Perfetto.

### Paquetes de crash — archivos `.crash`

Cuando el entrenamiento falla, Soup escribe automáticamente un JSON `.crash` autocontenido en
`./.soup-crashes/crash_<utc>_<hex>.crash` que contiene: trace del error redactado,
tipo de fallo clasificado (`oom` / `nan` / `cuda` / `dataloader` / `nccl` /
`other`), estado de la GPU en el momento del crash, resumen del entorno, últimas 50 filas de métricas y la
configuración (redactada recursivamente de tokens `hf_*` / `sk-*` / `Bearer …`). El
output_dir se reduce a `os.path.basename` para que `$HOME` no se filtre.

### `--log-level quiet|normal|verbose|debug`

Bandera global en el comando raíz `soup`. Conecta un logger con formato Rich en el
namespace `soup`; `debug` habilita marcas de tiempo + rutas de módulo.

```bash
soup --log-level verbose train --config soup.yaml
soup --log-level debug runs show <id>
```


## Monitor de GPU en vivo

```bash
soup monitor                # actualización cada 2s, Util / Mem / VRAM / Temp / Power por GPU
soup monitor --refresh 0.5  # sondeo más rápido
soup monitor --once         # una sola captura, sin panel Live
```

En sistemas NVIDIA, Soup llama a `nvidia-smi` mediante un subproceso con lista de argumentos (sin shell)
con un timeout de 5 segundos. En Apple Silicon, lee la utilización de GPU y el consumo
de `/usr/bin/powermetrics --samplers gpu_power --format plist`. Ejecuta `sudo -v`
en una terminal antes de iniciar el monitor: Soup usa `sudo -n`, así que puede reutilizar
la credencial en caché sin pedir nunca ni leer una contraseña. Si la
credencial o la utilidad no están disponibles, el comando sale con una alternativa de Activity
Monitor en lugar de reportar un error de NVIDIA.

macOS no expone campos dedicados de VRAM, utilización de memoria ni temperatura de GPU
al estilo NVIDIA a través de este sampler. Esas columnas por lo tanto se quedan como `—`
en lugar de adivinar valores a partir de la memoria unificada o de sensores térmicos no relacionados.


## Soup Fetch — ejemplos incluidos

```bash
soup fetch examples                          # lista las entradas incluidas
soup fetch examples llama-3.1-8b-lora        # escribe en ./llama-3.1-8b-lora.yaml
soup fetch examples qwen2.5-7b-dpo -o ./my-config.yaml --force
soup fetch deepspeed_configs zero3-cpu-offload
```

Catálogo cerrado (`MappingProxyType`) de YAML / JSON listos para editar. Ruta de salida contenida en el cwd, verificación `os.path.commonpath` de la fuente incluida (defiende contra escapes del catálogo), rechazo de symlinks con `os.lstat + S_ISLNK` en el destino de escritura.


## Delinearizador de Llama 4

```bash
soup delinearize-llama4 ./llama4-checkpoint --target ./out-delinearized [--num-experts N] [--plan-only]
```

LIVE (v0.71.21): reestructura los pesos fusionados de expertos de Llama-4 `[E*din, dout]` → `[E, din, dout]` shard por shard (escrituras atómicas, tope de 16 GiB por shard, contención en el cwd) y copia los sidecars JSON para que el destino siga siendo cargable. El número de expertos se toma por defecto de `config.json` (`text_config.num_local_experts`); pasa `--num-experts` cuando la configuración no lo traiga (si no, sale con 2). `--plan-only` conserva el flujo de previsualización original y no escribe nada. `is_llama4_model` usa una regex con límites de palabra que hace juego con el patrón de `is_gemma4_model`: `ungemma-llama-4ish` se rechaza.


## Guardado ordenado con Ctrl+C

El primer SIGINT → el trainer escribe un checkpoint y continúa. El segundo SIGINT → el trainer se detiene limpiamente después del siguiente guardado. La alternativa sin estado lanza `KeyboardInterrupt` para que el usuario nunca se quede trabado. `GracefulSaveHandler.install()` es idempotente y se traga los fallos de `signal.signal` en hilos que no son el principal.


## Archivo de activación Checkpoint-Now

```bash
touch ./out/.checkpoint_now    # el trainer guarda en el siguiente step y luego borra el activador
```

Contención de ruta vía `is_under_cwd`; rechazo con `os.lstat + S_ISLNK` en el destino del activador para que un symlink precolocado no pueda redirigir la escritura.


## Asistente del wizard de onboarding

```python
from soup_cli.utils.onboarding import render_onboarding_yaml

text = render_onboarding_yaml({
    "base": "meta-llama/Llama-3.2-1B",
    "dataset": "./train.jsonl",
    "task": "sft",
    "quantization": "4bit",
    "epochs": 3,
})
```

Entrada de un wizard de cinco preguntas → un `soup.yaml` totalmente validado. Allowlists literales en `task` (`sft` / `dpo` / `kto` / `orpo` / `simpo` / `ipo` / `bco` / `preference`) y `quantization` (`4bit` / `8bit` / `none`); `epochs ∈ [1, 10]`; `output` contenido en el cwd; rechazo de bytes nulos en cada cadena.


## Configuración de sweep independiente

`--config` sigue siendo el `soup.yaml` base contra el que entrenar; `--sweep-config` apunta a un
archivo aparte que contiene la estrategia, el número de ejecuciones, la semilla y los parámetros barridos, en lugar de
repetir `--param` en la línea de comandos.

```bash
soup sweep --config soup.yaml --sweep-config sweep.yaml
```

```yaml
# sweep.yaml
strategy: random
n_runs: 20
seed: 42
params:
  lr: [0.0001, 0.0005, 0.001]
  epochs: [1, 3, 5]
```

Allowlist estricta de escalares en los valores (`str` / `int` / `float` / `bool`); `_MAX_FILE_BYTES=256KB`, `_MAX_PARAM_KEYS=32`, `_MAX_VALUES_PER_KEY=64`; `SweepSpec.params` es `MappingProxyType[str, Tuple[Any, ...]]` para inmutabilidad genuina.

`--param` y `--sweep-config` son mutuamente excluyentes. Con `--sweep-config`:

- El `strategy` y el `seed` del archivo ganan sobre `--strategy`.
- Un `n_runs` distinto de cero gana sobre `--max-runs`; `0` o ausente cae a `--max-runs`.
- `n_runs` solo limita el número de ejecuciones, así que una grilla más pequeña que `n_runs` ejecuta cada combinación
  (el ejemplo de arriba es 3 x 3 = 9, así que corre 9 pruebas, no 20).
- `seed` da la semilla al muestreador aleatorio.

Un archivo que omite `strategy`, combinado con `--strategy random --max-runs 2` en la línea
de comandos, ejecuta grid en silencio, porque el `grid` por defecto del archivo gana sobre ambas banderas.


## Hubs de modelos alternativos (ModelScope / Modelers)

Define `training.hub` para obtener el modelo base de un Hub que no sea HF:

```yaml
base: baichuan-inc/Baichuan2-7B
task: sft
training:
  hub: modelscope     # o "modelers"
```

`soup train` precarga el modelo en `./.soup_hub_cache/<sanitized-slug>/` mediante el SDK correspondiente (`modelscope.snapshot_download` / `openmind_hub.snapshot_download`) y reescribe `cfg.base` a la instantánea local. Las reejecuciones reutilizan la instantánea en caché. Tanto `huggingface-hub`, `modelscope` como `openmind-hub` se importan de forma diferida: instala solo lo que necesites.

API programática:

```python
from soup_cli.utils.hubs import download_repo, upload_repo

local_path = download_repo("modelscope", "baichuan-inc/Baichuan2-7B", local_dir="./snap")
upload_repo("modelers", "my-org/my-model", folder_path="./output", commit_message="Soup v0.53.8")
```

El despachador impone validación de forma en cada entrada (bool / byte nulo / barra inicial / segmentos `..` / caracteres de control / tamaño excesivo, todo rechazado) y ejecuta contención en el cwd sobre `local_dir` / `folder_path`.


## Trackers de experimentos (MLflow / SwanLab / Trackio)

Elige un tracker en el CLI; Soup lo pasa al `report_to` de HF Trainer:

```bash
soup train --tracker mlflow
soup train --tracker swanlab
soup train --tracker trackio
```

Si el paquete no está instalado, Soup ahora muestra un aviso amigable antes de que empiece el entrenamiento en lugar de un ImportError a mitad de ejecución:

```
--tracker mlflow requires the 'mlflow' package. Install with: pip install soup-cli[trackers] (or pip install mlflow)
```

```bash
pip install soup-cli[trackers]   # mlflow + swanlab + trackio
```


## Telemetría y política de privacidad

Soup contiene telemetría opcional, solo con información de hardware, en `utils/trackers.py` (`build_telemetry_payload` / `send_telemetry_payload`).

### Política de privacidad

La telemetría de Soup es estrictamente anónima y enfocada al hardware. Cuando la aceptas con `SOUP_TELEMETRY=1`, recopilamos solo los siguientes campos para entender qué entornos necesitamos admitir:

- `soup_version`: la versión de Soup que se ejecuta
- `command`: el comando de nivel superior ejecutado (p. ej. `train`, `data`, validado contra comandos conocidos; los comandos o rutas desconocidos se enmascaran como `(unknown)`)
- `python`: versión major.minor de Python
- `os`: nombre de la plataforma del SO (`platform.system()`)
- `arch`: arquitectura del sistema (`platform.machine()`)
- `duration_seconds`: duración de la ejecución del comando en segundos
- `distinct_id`: UUID4 anónimo generado localmente en la primera ejecución y guardado en `~/.soup/telemetry_id` para deduplicar eventos. Borrar `~/.soup/telemetry_id` lo regenera en la siguiente aceptación.

**NUNCA** recopilamos:
- Rutas o contenidos de datasets
- Nombres o arquitecturas de modelos
- Contenidos de archivos de configuración o hiperparámetros
- Nombres de usuario, rutas de archivos locales o nombres de directorios
- Direcciones IP, tokens o credenciales

Todas las subidas usan HTTPS, un timeout de 1 segundo de conexión y lectura (sin contar la resolución DNS) y validación SSRF defensiva. Cualquier excepción de red o de sistema de archivos se traga en silencio, así que la telemetría nunca puede fallar ni interrumpir tu trabajo.


## Sistema de plugins

Soup descubre módulos incluidos bajo `soup_cli.plugins` y distribuciones de Python instaladas
que publican el grupo de entry-points `soup_cli.plugins`. Un plugin externo
expone una función de registro sin argumentos:

```python
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        ...
    def post_train(self, ctx):
        ...

def register():
    register_plugin(
        name="my-plugin",
        version="1.0.0",
        plugin=MyPlugin(),
        description="Hooks into pre/post-train",
        templates=["my-template"],         # metadatos opcionales
        model_groups=["my-arch-family"],   # metadatos opcionales
    )
```

Decláralo en el `pyproject.toml` de la distribución del plugin:

```toml
[project.entry-points."soup_cli.plugins"]
my-plugin = "my_package.soup_plugin:register"
```

Los módulos de plugins incluidos en Soup están habilitados por defecto. Los entry points de terceros instalados
están **deshabilitados por defecto**: el descubrimiento lee sus nombres y versiones de los metadatos
del paquete sin importar ni ejecutar sus módulos. `soup plugins enable <name>` es
el límite explícito que carga el entry point seleccionado; su nombre de entry-point debe
coincidir con el nombre del plugin que registra. La elección se guarda de forma atómica en
`~/.soup/plugins.json` y la reutilizan los procesos posteriores de Soup.

Define `SOUP_PLUGIN_STATE_PATH` para usar un archivo de estado local confiable distinto, por ejemplo en
un entorno de prueba aislado. Esta ruta explícita no está confinada al espacio de trabajo
actual; Soup rechaza NULs, rutas demasiado largas, archivos de estado que son symlinks, contenido demasiado grande
y JSON malformado.

```bash
soup plugins                       # descubre y lista los plugins
soup plugins enable my-plugin      # activa de forma persistente
soup plugins disable my-plugin
```

`soup plugins install` sale deliberadamente con estado 2: Soup no ejecuta un instalador
de paquetes en nombre del usuario. Instala la distribución con tu flujo de trabajo de paquetes de Python
de confianza y luego habilítala explícitamente.

Los nombres de plugins son kebab-case (`^[a-z0-9][a-z0-9-]{0,39}$`); las versiones son semver-ish (`MAJOR.MINOR.PATCH`); el registro limita `_MAX_PLUGINS=64`, `_MAX_TEMPLATES_PER_PLUGIN=32`, `_MAX_MODEL_GROUPS_PER_PLUGIN=32`. Volver a registrar el mismo `(name, version, plugin, templates, model_groups, description)` es idempotente; cualquier campo que no coincida se rechaza con un error claro. `templates` y `model_groups` son metadatos descriptivos que muestra `soup plugins`; Soup no los aplica automáticamente a la configuración del modelo.


## Catálogo de integraciones externas

```python
from soup_cli.utils.integrations import list_integrations, get_integration

list_integrations()                       # 15 entradas
get_integration("lm-studio").target_artifacts   # ("gguf",)
```

15 objetivos del ecosistema cubiertos: `lm-studio`, `comfyui`, `stable-diffusion-cpp`, `open-webui`, `ollama`, `tei`, `pgvector`, `faiss`, `weaviate`, `sentence-transformers`, `claude-code`, `cursor`, `continue`, `cline`, `sillytavern`. El auto-descubrimiento + el cableado de lanzamiento llegan con Deploy Autopilot de la v0.46.0.


## Plugins avanzados para el trainer

```python
from soup_cli.utils.trainer_plugins import validate_trainer_plugin_list

validate_trainer_plugin_list(["grokfast", "spectrum"])
# devuelve ("grokfast", "spectrum") — minúsculas canónicas, sin duplicados, ≤ 8 entradas
```

Allowlist de 6 entradas (`cce_plugin`, `grokfast`, `spectrum`, `llmcompressor`, `sonicmoe`, `math_verify`) para que un futuro campo de esquema `training.trainer_plugins: [...]` tenga una superficie estable. Callbacks en vivo en la v0.45.1.


## Callbacks de plugins de Soup

Registra un plugin una vez mediante la API del registro de la v0.45.0; la v0.53.6 lo conecta a cada
trainer de backend transformer como un `TrainerCallback` real de HF:

```python
# my_package/soup_plugin.py — solo lo descubren los comandos/training que reconocen plugins
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        print("training about to start, args =", ctx["args"])

    def post_step(self, ctx):
        if ctx["state"].global_step % 100 == 0:
            print(f"step {ctx['state'].global_step}")

def register():
    register_plugin(name="my-plugin", version="0.1.0", plugin=MyPlugin())
```

Un hook de plugin que se porta mal se traga en nivel WARNING: un plugin malo nunca debe tumbar
una ejecución de entrenamiento de varias horas. La instantánea de hooks se toma al construir el callback,
así que un plugin registrado a mitad de la ejecución no recibe eventos retroactivamente. El descubrimiento es
diferido, así que importar el CLI o ejecutar un comando ligero no relacionado no carga los entry points
de plugins.


## Plan y apply estilo Terraform (`soup plan` / `soup apply`)

Una ejecución de entrenamiento es una operación con forma de infraestructura y de una sola vez: precio spot, costo esperado, SHA base, SHA del dataset, VRAM máxima. La v0.64 toma prestada la división plan-apply de Terraform para que puedas revisar los números antes de comprometer dinero.

```bash
# Genera un resumen previo + escribe soup.tfstate
soup plan --config soup.yaml

# Aplica — rechaza si hay drift (salida 3) y el YAML cambió desde `plan`
soup apply --config soup.yaml

# Valida sin ejecutar nada realmente
soup apply --config soup.yaml --dry-run
```

El archivo de estado es un sobre JSON delgado; el entrenamiento real sigue siendo manejado por `soup train`. El portero evita la sorpresa de "espera, ¿por qué gasté otros $0.50 en la configuración equivocada".


## Lockfile de entorno hermético (`soup env`)

El problema del "infierno de CUDA": un ajuste que funcionaba el viernes se rompe el lunes porque PyPI actualizó `transformers` en silencio más allá de la banda de compatibilidad del trainer. La v0.34 `soup doctor` muestra parte de esto; la v0.64 lo hace bloqueable.

```bash
# Toma una instantánea del entorno actual en soup-env.lock
soup env lock

# Imprime el resumen del entorno bloqueado
soup env status

# Compara el entorno actual con el lock — salida 3 si hay drift sensible a ABI
soup env check
```

`soup-env.lock` captura Python + plataforma + CUDA + 15 paquetes sensibles a ABI (torch / transformers / peft / trl / accelerate / bitsandbytes / flash-attn / xformers / deepspeed / unsloth / vllm / sentencepiece / tokenizers / datasets / huggingface-hub). Conecta `soup env check` a tu CI para rechazar rupturas silenciosas de ABI.


## Calculadora de ajuste de hardware

Dados (params, seq_len, batch_size, optimizer, quant, peft, gradient_checkpointing), el predictor analítico devuelve un desglose de VRAM máxima en 5 cubos (pesos / optimizador / gradientes / activaciones / overhead) y un veredicto OK/OOM con un margen de seguridad del 10%.

Para `soup train`, `params_b` se deriva con `model_size_from_name`:
- Una ruta de checkpoint local se mide directamente de los encabezados de metadatos de safetensors.
- Un id de Hub se parsea: p. ej. `72B`, `8x7B` → `46.7` en total, `17B-16E` → `109` en total, ignorando los marcadores de parámetros activos (`-A22B`).
- Un id sin token de tamaño (p. ej. `DeepSeek-V3`, `Kimi-K2`, `GLM-5`, `MiniMax-M2`) igual cae a `7B`, lo que subestima estos modelos (tienen de 229B a 1T de parámetros). Apunta `base:` a una descarga local para que se mida el checkpoint en su lugar.

```python
from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

inp = HardwareFitInput(
    params_b=7.0, seq_len=2048, batch_size=4,
    optimizer="adamw_torch", quant="4bit", peft="lora",
    gradient_checkpointing=True,
)
report = decide_hardware_fit(inp, available_vram_gb=24.0)
print(report.ok, report.reason)
# True | 'fits: peak 7.76 GB + 10% margin <= 24.00 GB available'
```

Cuando no cabe, el informe nombra perillas accionables: `--batch-size halve`, `--quantization 4bit`, `--gradient-checkpointing auto`. Se compone con la sonda de OOM en vivo de CUDA de la v0.40.3 (`make_cuda_probe_fn`), que igual se ejecuta cuando `auto_batch_size_strategy: probe`.

El cubo de pesos asume 2 bytes/param bajo `quant="none"` (una base congelada ahora sí carga con el dtype propio del checkpoint, típicamente bf16/fp16 — #339), salvo `peft="full"` (ajuste completo), que carga explícitamente pesos maestros fp32 y por eso asume 4 bytes/param en su lugar.

`mxfp4` y `fp8` se desquantizan al cargar, así que toman los mismos 2 bytes/param. `gptq`, `awq`, `aqlm` y `eetq` se quedan empaquetados y se cotizan con sus propias cifras. `soup train` no tiene cifra para `hqq:*` y dice que se saltó la verificación ([Menú de cuantización](performance-and-quantization.md#quant-menu--9-quantization-formats)).


## Completado de shell (`soup completions`)

Autocompletado con Tab para `soup` + cada subcomando. El script generado se apoya en Click/Typer, así que los comandos nuevos se detectan automáticamente.

```bash
# Bash
eval "$(soup completions bash)"        # shell actual
soup completions bash >> ~/.bashrc     # permanente

# Zsh
soup completions zsh > "${fpath[1]}/_soup"

# Fish
soup completions fish > ~/.config/fish/completions/soup.fish
```

Los nombres de comandos y de opciones se completan, directamente del árbol de comandos en vivo: `rec` + Tab da `recipes`, y después de `soup train`, `--res` + Tab da `--resume`. Los valores todavía no se completan —nombres de recetas después de `soup recipes show`, por ejemplo— porque ninguna opción ni argumento está cableado a un completador de valores. Hay dos listos para ese cableado en `soup_cli/utils/completions.py`: `complete_recipe_name`, y `complete_target_modules`, que desde la v0.71.1 busca la arquitectura del modelo base en su configuración de Hugging Face cuando esa configuración ya está en la caché local (solo la configuración, nunca una descarga) y ofrece los nombres de módulos de esa arquitectura, cayendo a los valores por defecto canónicos con forma de Llama (`q_proj` / `k_proj` / `v_proj` / etc.).


## Asesor de licencias (`soup license-advisor`)

Elegir una base limpia en licencias para un objetivo de despliegue específico es un dolor de cabeza recurrente de revisión legal. La v0.64 captura los tres contextos de despliegue más comunes como una allowlist cerrada y muestra el riesgo posterior por licencia.

```bash
# ¿Qué licencias son seguras para un producto de consumo B2C?
soup license-advisor --target b2c

# Defensa — se prohíben las licencias comunitarias de uso restringido
soup license-advisor --target defense

# Verificación por licencia: licencia comunitaria de Llama + 800M MAU = bloqueo (salida 3)
soup license-advisor --target b2c --license llama-3 --monthly-active-users 800000000
```

La allowlist de la familia Llama es estricta (sin coincidencias excesivas por `.startswith`), así que un hipotético futuro `llama-permissive-2030` no disparará por error el umbral de 700M MAU. Se compone con `soup adapters merge --license <id>` de la v0.60 para el portero de conflictos en el momento de la fusión.

## Solución de problemas

```bash
soup doctor    # GPU, recursos del sistema, dependencias y versión en un solo lugar
```

- **`ImportError: DLL load failed while importing _C` (Windows).** La rueda torch de PyPI
  es solo-CPU. Reinstala una build con CUDA; `soup doctor` imprime el
  comando `pip install` para la rueda que tu driver puede ejecutar.
- **`soup version` ≠ `pip show soup-cli`** — varias instalaciones de Python; usa un virtualenv.
