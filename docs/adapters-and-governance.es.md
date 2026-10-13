# Adaptadores, registro y gobernanza

> 🇪🇸 Traducción al español de [`adapters-and-governance.md`](adapters-and-governance.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> Ciclo de vida y gestión de adaptadores, el registro de modelos, Soup Cans, el volante de datos (`soup loop`), edición de conocimiento, direccionamiento de activaciones y las superficies de cadena de suministro / gobernanza (scan, sign, BOM, attest, audit, airgap).

**Contenido:**

- [Ciclo de vida de adaptadores (`soup adapters {merge,pr,bisect}`, `soup lock`)](#ciclo-de-vida-de-adaptadores-soup-adapters-mergeprbisect-soup-lock)
- [Volante de datos (`soup loop`)](#volante-de-datos-soup-loop)
- [Edición de conocimiento (`soup edit set`, ROME / MEMIT / AlphaEdit)](#edición-de-conocimiento-soup-edit-set-rome--memit--alphaedit)
- [Direccionamiento de activaciones (`soup steer`)](#direccionamiento-de-activaciones-soup-steer)
- [Codebook GRACE — ediciones de conocimiento de por vida](#codebook-grace--ediciones-de-conocimiento-de-por-vida)
- [Registro de modelos y linaje](#registro-de-modelos-y-linaje)
- [Gestión de adaptadores (git para LoRA)](#gestión-de-adaptadores-git-para-lora)
- [Soup Cans (recetas compartibles)](#soup-cans-recetas-compartibles)
- [Lista de materiales (`soup bom emit`)](#lista-de-materiales-soup-bom-emit)
- [Atestaciones de procedencia (`soup attest emit`)](#atestaciones-de-procedencia-soup-attest-emit)
- [Documentación automática de los Anexos XI/XII del EU AI Act (`soup train --annex-xi`)](#documentación-automática-de-los-anexos-xixii-del-eu-ai-act-soup-train---annex-xi)
- [Registro de auditoría (`soup audit-log`)](#registro-de-auditoría-soup-audit-log)
- [Recibo de reproducibilidad (`soup train --repro-receipt`)](#recibo-de-reproducibilidad-soup-train---repro-receipt)
- [Escáner de puertas traseras en adaptadores (`soup adapters scan`)](#escáner-de-puertas-traseras-en-adaptadores-soup-adapters-scan)
- [Firma y verificación de adaptadores (`soup adapters sign` / `verify`)](#firma-y-verificación-de-adaptadores-soup-adapters-sign--verify)
- [Modo estricto de safetensors (`soup adapters check-safetensors`)](#modo-estricto-de-safetensors-soup-adapters-check-safetensors)
- [Auditoría de salud del adaptador previa al vuelo (`soup adapters check`)](#auditoría-de-salud-del-adaptador-previa-al-vuelo-soup-adapters-check)
- [Fijación de namespace (anti-AI-jacking)](#fijación-de-namespace-anti-ai-jacking)
- [Matriz de conflictos de licencias en la fusión](#matriz-de-conflictos-de-licencias-en-la-fusión)
- [Paquete airgap (`soup airgap-bundle`)](#paquete-airgap-soup-airgap-bundle)

---

## Ciclo de vida de adaptadores (`soup adapters {merge,pr,bisect}`, `soup lock`)

v0.57 incluyó `adapters diff / merge / blame / branch`. v0.67 completa el ciclo de vida: fusión evolutiva guiada por tu evaluación, PRs con forma de GitHub para la revisión de adaptadores, un `soup.lock` compartido para la reproducibilidad del equipo y una bisección por búsqueda binaria sobre el historial de entrenamiento.

```bash
# 1. Fusión evolutiva: busca el simplex de pesos de fusión mediante CMA-ES.
#    v0.71.4 convierte esto en un bucle EN VIVO: cada candidato se fusiona, se puntúa contra
#    --eval y la mejor mezcla se escribe en --output.
soup adapters merge \
  adapter-finance/ adapter-medical/ adapter-legal/ \
  --strategy cmaes \
  --eval evals/domain_mix.yaml \
  --budget 1h \
  --population 8 \
  --max-generations 20 \
  --output merged/

# 1b. Fusión de una sola pasada con veredicto canary en vivo (v0.71.4). Puntúa el adaptador
#     fusionado frente a la primera entrada → OK / MINOR / MAJOR; --strict-verdict sale con 2
#     ante una regresión MAJOR. Una suite canary ya puntuada no necesita cargar el modelo.
soup adapters merge adapter-a/ adapter-b/ -o merged/ \
  --canary evals/canary.json --strict-verdict

# 2. Renderiza la fusión como un PR de GitHub para revisión (deltas de evaluación + diffs de muestras).
soup adapters pr "merge: 3-domain blend" \
  --base-sha $(git rev-parse HEAD) \
  --adapter merged/ \
  --eval evals/deltas.json \
  --samples evals/samples.json \
  --dataset-diff data/diff.txt \
  --format markdown -o pr.md

# 2b. Publícalo directamente en un comentario de PR de GitHub (v0.71.4). Autenticación vía GITHUB_TOKEN
#     / GH_TOKEN; usa `gh api` por debajo (sin filtrar secretos al proceso hijo).
soup adapters pr "merge: 3-domain blend" \
  --base-sha $(git rev-parse HEAD) --adapter merged/ \
  --eval evals/deltas.json --push your-org/your-repo#42

# 3. Fija un estado de ejecución reproducible. Closure = sha(base + dataset + env).
soup env lock                                     # v0.64 — captura el hash del entorno
soup lock write \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha $BASE_SHA \
  --dataset-sha $DATA_SHA \
  --env-hash $(jq -r .closure soup-env.lock) \
  -o soup.lock
# Los compañeros de equipo vuelven a verificar el lock; sale con 2 si hay drift.
soup lock check soup.lock \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha $BASE_SHA --dataset-sha $DATA_SHA --env-hash $ENV_HASH

# 4. Bisecciona un historial de entrenamiento para encontrar el step que rompió una evaluación.
soup adapters bisect \
  ckpt-step-100 ckpt-step-200 ckpt-step-400 ckpt-step-800 \
  --eval-command "soup eval custom --model {ckpt} --tasks eval.jsonl" \
  -o bisect.json
# Sale con 3 en BROKEN_AT: canalízalo a `soup adapters blame` para atribución.
```

CMA-ES es Python puro (sin dependencia de `cma`); la evaluación la aporta el operador mediante un closure, así funciona cualquier código de puntuación. El renderizado de PR escapa las celdas de las tablas Markdown, de modo que nombres de métricas manipulados no puedan inyectar filas de tabla ni enlaces. El lockfile se compone con `soup env lock` de v0.64: un drift en cualquiera de `{base_model, base_model_sha, dataset_sha, env_hash, closure_sha}` sale con 3 (`soup_version` y `created_at` son sólo informativos). La bisección usa `shlex.split` + `shlex.quote(ckpt)` en modo de lista de argv (sin `shell=True`), así que los ids de checkpoint no pueden inyectar metacaracteres de shell.

El almacenamiento en banco de VeRA / VB-LoRA (`soup_cli.utils.vector_bank`) da soporte al servicio multi-tenant en vivo,
y `task='moe_lora_routing'` entrena el kernel de gating MoLE disperso por token en vivo.


## Volante de datos (`soup loop`)

El bucle completo *trazas de producción → pares de preferencia → DPO con compuerta de evaluación → despliegue canary → rollback*, controlado desde un solo CLI. Conecta Trace-to-Preference de v0.26 + entrenamiento con compuerta de evaluación + linaje del registro + veredictos Quant-Lobotomy + Soup Cans + Autopilot de v0.25 + Advise de v0.54 + Eval Design de v0.55 + Diagnose de v0.56.

```bash
# Configuración inicial
soup loop init registry://abc12 --eval evals/lock.json --baseline registry://prod \
    --monthly-budget 50usd --max-runs-per-day 3

# Inspecciona los contadores + el estado
soup loop status

# Ejecuta el daemon (en primer plano)
soup loop watch --poll-interval 300

# Etapas de producción preconfiguradas (v0.71.4): trazas reales → DPO → compuerta de evaluación → canary,
# en lugar de los stubs sin efecto de v0.58. Actívalas una sola vez con `loop init --pre-wired` o
# por ejecución con --pre-wired. Empaqueta cada iteración como un Soup Can compartible con
# linaje del registro mediante --pack-cans.
soup loop init registry://abc12 --eval evals/lock.json --baseline registry://prod --pre-wired
soup loop watch --pre-wired --pack-cans --poll-interval 300

# Desempaqueta el Soup Can de una iteración registrada para análisis hipotético (v0.71.4)
soup loop replay iter-20260515T120000-abcdef01 --extract ./iter-dump

# Subproceso en segundo plano (escribe el PID, sin shell)
soup loop watch --detach

# Inicia serve con el candidato cargado y luego promuévelo al 5% del tráfico.
soup serve --model ./base --adapters candidate=./candidate
soup loop canary candidate --traffic 5% --autoroll-on-regress

# Pausa/reanuda el daemon entre iteraciones (cambio de estado atómico)
soup loop pause
soup loop resume

# Reproduce cualquier iteración registrada
soup loop replay iter-20260515T120000-abcdef01
```

El estado vive en `.soup/loop.yaml` (escritura atómica, contenido en el cwd, con rechazo de symlinks). Los manifiestos por iteración en `.soup-loops/<iter-id>/iteration.json` están dispuestos de modo que un Soup Can de v0.26 pueda envolverlos directamente.

El enrutamiento canary está disponible en el backend de transformers cuando el nombre del candidato
se cargó con `soup serve --adapters`. Envía una clave estable en el
campo `conversation_id` o en el encabezado `X-Conversation-Id` de
`POST /v1/chat/completions`; las peticiones sin clave conservan el comportamiento normal del
adaptador activo. El enrutador aplica hash a la clave, así que cada petición de una misma conversación se queda en
el mismo bucket estable o canary. Un campo `adapter` explícito sigue teniendo prioridad.

Las generaciones completadas y los errores de generación se acumulan por separado para los dos
buckets en `.soup/canary-stats.json`. `soup loop watch` evalúa esas observaciones
tras al menos 30 muestras canary. Si la tasa de éxito del canary va por debajo de la tasa estable en
más de cinco puntos porcentuales, el veredicto es `MAJOR`; con
`--autoroll-on-regress`, el watcher limpia de forma atómica el canary y su cuota de tráfico
de `loop.yaml`. `/v1/adapters/activate/<name>` sigue siendo un cambio explícito al 100% para
las peticiones sin clave de conversación y no cambia la política canary. Las peticiones con clave
siguen el reparto canary; su bucket estable usa el modelo base servido,
incluso cuando se activó otro adaptador de forma manual. Inicia `soup serve` desde el
directorio que contiene `.soup/`, porque las rutas del estado del loop y de las estadísticas canary se
resuelven desde el directorio de trabajo. Los clientes eligen su propia clave estable, así que un cliente
puede seleccionar a propósito una clave que haga hash dentro del bucket canary.

El proceso de servicio guarda en caché la política ya analizada usando la identidad
`(st_mtime_ns, st_size, st_ino)` del archivo de estado. Un reemplazo atómico la invalida,
así que un rollback escrito por `soup loop watch` afecta la siguiente petición con clave.
Los fallos de la ruta de estado se advierten una vez por episodio de fallo, no en cada petición;
una lectura exitosa de la política permite que un fallo posterior vuelva a advertir.

Los contadores de resultados se almacenan en búfer por aplicación de servicio y se persisten tras 64
resultados completados o dos segundos desde el primer resultado pendiente, lo que ocurra primero.
Un apagado ordenado vacía los contadores restantes. Con un sistema de archivos saludable y escribible,
una caída abrupta puede perder la ventana sin vaciar (como máximo un lote de 64 resultados, incluido
un volcado en curso, normalmente dos segundos como máximo); los retrasos de planificación o los errores de disco
pueden demorar la persistencia. Una escritura fallida conserva los contadores pendientes para reintentar.
Mientras ese lote esté lleno, se rechazan resultados adicionales hasta que la persistencia se recupere;
por eso una caída del disco puede perder más resultados que la ventana normal de caída de un lote.
El `canary-stats.json` en disco y los veredictos del watch van por detrás del tráfico en la ventana del búfer
(normalmente hasta 64 resultados o dos segundos). Los IDs de rollout evitan que las promociones compartan
muestras. Las muestras pendientes y las finalizaciones tardías de promociones reemplazadas se descartan
en lugar de reemplazar el archivo del nuevo rollout. El lock de estadísticas existente protege hilos
dentro de un proceso, no a varios procesos de servidor que escriban el mismo archivo.

Un canary que mata el proceso de servicio nunca llega al registro de resultados: el registro
ocurre sólo después de que la generación retorne o lance una excepción. Esas caídas permanecen invisibles
al veredicto canary salvo que un monitoreo externo de reinicios las detecte.


## Edición de conocimiento (`soup edit set`, ROME / MEMIT / AlphaEdit)

Parches factuales quirúrgicos SIN un bucle de ajuste fino completo. Un equipo de datos hospitalario corrigiendo una interacción farmacológica mal atribuida, un laboratorio arreglando una fecha histórica errónea, un equipo de seguridad respondiendo a un CVE alucinado: todo en una sola invocación del CLI.

```bash
# Edición en vivo: optimiza un residuo + aplica una actualización de pesos de rango 1 y luego guarda.
soup edit set \
  --base HuggingFaceTB/SmolLM2-135M \
  --method rome \
  --subject "The capital of France is" \
  --target "Lyon" \
  --output ./edited --device cuda

# ROME precondicionado por covarianza (actualización genuina C^-1 k*) a partir de un corpus de estadísticas.
soup edit set --base ./model --method rome \
  --subject "..." --target "..." \
  --cov-corpus ./stats.jsonl --output ./edited

# El modo sólo plan valida la petición + imprime el EditPlan resuelto + sale con 0.
soup edit set --base ./model --method rome --subject "..." --target "..." --plan-only

# Compara lo que dice el modelo antes y después de la edición (generación en vivo).
soup edit diff before after \
  --before-model ./base --after-model ./edited \
  --probes probes.jsonl --output diff.json
```

Los kernels ejecutan una actualización real de pesos de rango 1 en una proyección descendente del MLP. Admiten tanto la **familia Llama** (`model.model.layers[L].mlp.down_proj`, una `nn.Linear` con pesos `[out, in]`) como la **familia GPT-2** (`model.transformer.h[L].mlp.c_proj`, una `Conv1D` con el layout transpuesto `[in, out]`): la actualización tiene en cuenta la transposición, así que la post-condición `down(k*) += delta` se cumple exactamente en cualquiera de los dos layouts (validado en `gpt2` real: `P(target)` 0.005 → 0.9996, y en SmolLM2-135M). Los modelos envueltos con PEFT se desenvuelven de forma automática. **ROME** edita una sola capa; **MEMIT** distribuye el residuo a lo largo de una banda de capas; **AlphaEdit** proyecta la actualización de forma ortogonal a la principal dirección singular del down-proj.

Por defecto ROME usa la forma sin covarianza `C = I`. Pasa `--cov-corpus <jsonl|txt>` para estimar la covarianza de claves `C = E[k kᵀ] + λI` sobre un corpus de estadísticas y aplicar la forma cerrada genuina de ROME `u = C⁻¹ k*`: así se reparte la masa de la actualización de rango 1 según la forma cerrada y se reduce la interferencia colateral con otros hechos (la post-condición exacta se conserva de cualquier manera). El cargador del corpus está contenido en el cwd, rechaza symlinks y tiene límite de tamaño y de líneas; `--cov-corpus` se rechaza para cualquier método que no sea `rome`.

El gobernador de ediciones secuenciales es persistente (SQLite, con lock entre procesos, anulable con `SOUP_EDIT_GOVERNOR_DB`), así que el recuento de ediciones por modelo base + el veredicto de explosión de norma sobreviven entre ejecuciones separadas de `soup edit set`. El incremento del contador es atómico: dos ejecuciones concurrentes de `soup edit set` sobre la misma base se combinan bajo el lock entre procesos (sin incrementos perdidos). `soup edit set` consulta el gobernador de forma automática: rechaza ANTES de cargar el modelo si se supera el tope por modelo base o si hay un veredicto BLOWUP, y registra el `||ΔW||_F` medido después de cada edición. Pasa `--no-governor` para no usarlo. `--registry-id <id>` (necesita `--output`) registra el resultado guardado en esa entrada del registro: el o los archivos de pesos que escribió `save_pretrained` (`model.safetensors`, sus shards numerados o `pytorch_model*.bin`) como artefactos `edited_model`, o `grace_codebook.json` como artefacto `grace_codebook`. El registro guarda un archivo con hash por artefacto, nunca un directorio. Un attach solicitado es parte del éxito del comando: `soup edit set` sale con 1 cuando falla (entrada desconocida o ambigua, registro ilegible, sin archivo de pesos), dejando el modelo editado en disco, y sale con 2 antes de cargar el modelo cuando la bandera se da sin `--output`.


## Direccionamiento de activaciones (`soup steer`)

A veces no quieres reentrenar: quieres *empujar* el modelo a lo largo de una dirección aprendida en el momento de la decodificación. Soup incluye tres backends de vectores de control:

- **CAA** (Contrastive Activation Addition): suma un vector contrastivo al flujo residual.
- **ITI** (Inference-Time Intervention): desplaza cabezas de atención concretas a lo largo de una dirección aprendida.
- **RepE** (Representation Engineering): dirección basada en PCA en el flujo residual.

```bash
# Entrena un vector de direccionamiento a partir de pares de prompts contrastivos (positivos, negativos)
soup steer train --base meta-llama/Llama-3.1-8B-Instruct \
                 --method caa --name safety-v1 \
                 --pairs ./data/pairs.jsonl

# Aplícalo en el momento de decodificar mediante soup serve
soup serve --model ./adapter --steer safety-v1 --steer-strength 1.5

# Lista los vectores de direccionamiento guardados localmente
soup steer list
```

Los nombres de direccionamiento se validan contra una expresión regular estricta (`^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$`, sin separadores de ruta ni metacaracteres de shell); la fuerza está acotada por `|s| <= 10.0`. Con `--registry-id <id>` el archivo del vector entrenado (`steering_vector.safetensors`) se registra en esa entrada del registro como artefacto `steering_vector`, así se conserva el linaje y `soup steer list` lo muestra; `<id>` es un id de entrada, un prefijo de id único, `name:tag` o `registry://<id>`. Un vector guardado fuera de `./steering/<name>` con `--output` se encuentra después por nombre a través de una entrada llamada `<name>`. Un attach solicitado es parte del éxito del comando: `soup steer train` sale con 1 cuando falla, dejando el vector en disco. Sin la bandera no se escribe nada en el registro.

A partir de v0.71.10 el ajuste y el hook de decodificación están **en vivo** (validado en SmolLM2-135M): `soup steer train` captura activaciones del flujo residual (CAA / RepE) o activaciones de entrada de `o_proj` por cabeza (ITI) sobre los pares contrastivos, calcula el vector de control y persiste `steering_vector.safetensors` + `steering_config.json`. `soup serve --steer <name>` instala un hook forward en el modelo cargado que suma `strength × vector` en el momento de decodificar (backend de transformers; `--steer` se rechaza con un error claro en vLLM/SGLang). RepE / ITI necesitan al menos dos pares contrastivos; CAA funciona con uno.


## Codebook GRACE — ediciones de conocimiento de por vida

ROME / MEMIT en su versión básica se degradan tras docenas de ediciones secuenciales: las normas del modelo se disparan. GRACE (Hartvigsen et al., 2023) guarda cada edición en un codebook latente discreto para que sobrevivan miles de parches secuenciales:

```bash
soup edit set --base ./model --method grace \
              --subject "The CEO of Acme is" --target "Jane Doe"
```

```yaml
# Previsto para soup.yaml al entrenar con lookups conscientes de GRACE (por etapas; rechazado desde v0.77 — #808):
# training:
#   grace_codebook: true
#   grace_codebook_size: 1024    # entradas del codebook (máx. 100k)
#   grace_codebook_dim: 768      # ancho del flujo residual
```

`grace` se suma a la lista permitida existente de `rome` / `memit` / `alphaedit` en `soup edit set`; el gobernador de ediciones secuenciales sigue controlando la llamada cuando se dispara el recuento de ediciones por modelo base o el veredicto de explosión de norma. GRACE está en vivo para editar: `soup edit set --method grace --output ./ckpt` captura la clave residual en el último token del sujeto, optimiza un valor de reemplazo y añade una terna `(key, value)` a un sidecar `grace_codebook.json` (atómico, contenido en el cwd). En inferencia el codebook se aplica mediante un hook forward que sustituye el residuo siempre que caiga dentro de una bola épsilon de una clave almacenada, así los pesos base nunca se modifican y miles de ediciones sobreviven sin explosión de norma. La integración de lookups en tiempo de entrenamiento sigue por etapas.


## Registro de modelos y linaje

Todo ajuste fino que publiques debería ser reproducible. El registro local de Soup (`~/.soup/registry.db`) rastrea cada entrada mediante un hash de contenido de su configuración + datos + modelo base, además de punteros de linaje a las entradas padre.

```bash
# Registra una ejecución completada
soup registry push --run-id run_202611_abc123 --name llama31-chat --tag v1

# Lista entradas (filtra por nombre, tag, modelo base, tarea)
soup registry list
soup registry list --name llama31-chat --tag prod

# Muestra los detalles completos: configuración, línea base de evaluación, artefactos, ancestros
soup registry show llama31-chat-v1

# Diff de configuración lado a lado + delta de evaluación entre dos entradas
soup registry diff llama31-chat-v1 llama31-chat-v2

# Búsqueda de texto completo en nombre / modelo base / tarea / notas
soup registry search "medical reasoning"

# Promueve una entrada (añade un tag, p. ej. "prod")
soup registry promote llama31-chat-v1 --tag prod

# Elimina (en cascada a artefactos + enlaces de linaje)
soup registry delete llama31-chat-v1 --yes
```

**DAG de linaje** — cada entrada puede apuntar a un padre (su ejecución ancestro). Recorre el DAG de cualquier nombre con:

```bash
soup history llama31-chat
```

**Las referencias se resuelven de forma flexible** — puedes usar un ID de registro, un nombre (el más reciente) o `name:tag`. Los prefijos ambiguos lanzan un error en lugar de elegir en silencio la entrada equivocada. Los archivos del registro se guardan con permisos `600` en POSIX; cambia la ruta con `SOUP_REGISTRY_DB_PATH`.


## Gestión de adaptadores (git para LoRA)

`soup adapters` es la superficie de git para LoRA: diff consciente de los pesos, cuatro estrategias de fusión, blame leave-one-out y snapshots de rama con SHA-256. Todos los comandos operan sobre directorios `adapter_model.safetensors` (compatibles con peft).

```bash
# Diff Frobenius de ΔW por capa + drift del rango efectivo + top-K proyecciones cambiadas
soup adapters diff ./run-v17 ./run-v18

# JSON legible por máquina para CI
soup adapters diff ./run-v17 ./run-v18 --format json --output diff.json

# Fusión ponderada con estrategias linear / ties / dare / svd
soup adapters merge ./run-v17 ./run-v18 ./run-v19 -o ./merged --strategy ties \
  --weights 0.5,0.3,0.2 --density 0.2

# Fusión DARE (determinista vía --seed)
soup adapters merge ./run-v17 ./run-v18 -o ./merged --strategy dare \
  --density 0.5 --seed 42

# Aritmética de vectores de tarea (v0.71.34): suma / escala / NIEGA comportamientos entrenados
# hacia un solo adaptador (arXiv:2212.04089). Los nombres se asignan a directorios con --adapter name=path.
soup adapters arithmetic "coder + 0.5*math - toxic" \
  --adapter coder=./coder-lora --adapter math=./math-lora \
  --adapter toxic=./toxic-lora -o ./blended

# Plan de ablación leave-one-out con un presupuesto de 4 horas de reloj
soup adapters blame ./run-v18 --dataset train.jsonl --layer q_proj.7 \
  --budget 4h --shards 10 --plan-only

# Guarda un snapshot de un entorno de entrenamiento como rama comparable
soup adapters branch v18 --config soup.yaml --base meta-llama/Llama-3.1-8B \
  --dataset train.jsonl

# Enlaza una rama en el DAG de linaje del registro (v0.71.4): aparece como un
# nodo `branches` bajo la entrada en `soup history`.
soup adapters branch v18 --config soup.yaml --base meta-llama/Llama-3.1-8B \
  --attach-to-registry reg_20260601_abc123

# O deriva la configuración + base de un snapshot nuevo directamente de una entrada del registro (v0.71.4)
soup adapters branch v18-from-reg --from-registry reg_20260601_abc123

# Restaura la configuración del snapshot (se niega si el SHA de origen cambió)
soup adapters checkout v18 --output soup.yaml

# Lista todas las ramas con snapshot
soup adapters branches
```

**Cuatro estrategias de fusión (numpy puro, sin importar torch a nivel de módulo):**

| Estrategia | Matemáticas | Caso de uso |
|----------|------|----------|
| `linear` | Promedio ponderado por capa | Línea base; las tareas comparten una base |
| `ties` | Recorte por densidad → elección del signo mayoritario → promedio disjunto | Adaptadores de tareas en conflicto (Yadav et al. 2023) |
| `dare` | Descarte aleatorio con `density` + reescalado `1/density`, luego promedio | Fusión dispersa; reduce la interferencia entre parámetros (Yu et al. 2024) |
| `svd` | Fusión lineal → reconstrucción de bajo rango vía SVD (`--rank`) | Restringe el rango efectivo del delta fusionado |

**Valores por defecto y seguridad:**

- Las rutas de salida se verifican como contenidas bajo el cwd y rechazan symlinks preexistentes (defensa TOCTOU).
- Las escrituras de safetensors son atómicas vía `tempfile.mkstemp` + `os.replace`: una caída a mitad de escritura nunca deja un adaptador parcial en la ruta destino.
- El formato de adaptador `.bin` (pickle de PyTorch) se rechaza con un mensaje explícito de "vuelve a guardar como safetensors".
- Los punteros de rama viven bajo `~/.soup/branches/` (se cambian con `SOUP_BRANCHES_DIR`, restringido a `$HOME` / `$CWD` / `$TMPDIR`).
- `soup adapters checkout` verifica por SHA la configuración de origen: se niega a restaurar cuando el origen se ha desviado del snapshot, así la reproducibilidad nunca miente en silencio.

**v0.66.0:** `soup adapters blame` ahora está EN VIVO: el stub `NotImplementedError` de v0.57 (#171) se levanta mediante una aproximación de función de influencia al estilo DataInf. Pasa `--top-k 50` para controlar el número de principales influyentes que se informa; pasa un `probe_fn` real (API de Python) para alimentar gradientes reales, o usa la sonda sintética determinista por defecto para planificación sin conexión.

**v0.71.4:** el veredicto de fusión ahora está EN VIVO: `soup adapters merge … --canary suite.json` levanta el stub `UNKNOWN` de `MergeReport.verdict`. Una suite pre-puntuada `{"baseline_scores","candidate_scores"}` clasifica la mezcla como OK / MINOR / MAJOR sin cargar el modelo; una suite `{"tasks":[...]}` usa un scorer inyectable. `--strict-verdict` sale con 2 en MAJOR. Las compuertas de escaneo de puertas traseras y de conflicto de licencias ahora se ejecutan para **todas** las estrategias, incluida `cmaes`. `soup adapters branch <name> --from-registry <id>` / `--attach-to-registry <id>` enlazan snapshots del entorno de entrenamiento en el DAG de linaje del registro (se muestran como un nodo `branches` en `soup history`).

**v0.71.34 — aritmética de vectores de tarea.** `soup adapters arithmetic "<expr>"` aplica aritmética de tareas (arXiv:2212.04089) a los deltas de LoRA: **suma** (combina dos habilidades), **escala** (`2*coder`) y —el diferenciador— **niega** (`- toxic` elimina un comportamiento). Los nombres de la expresión se asignan a directorios de adaptadores mediante `--adapter name=path`, repetible. La matemática se hace a nivel del delta efectivo: un LoRA aporta `ΔW = B·A`, así que el coeficiente se aplica de modo que `ΔW` escale de forma **lineal** (restar un adaptador niega de verdad su delta: una suma ingenua elemento a elemento escalaría por `c²`, lo que convertiría la negación en un no-op). Los adaptadores deben compartir el modelo base (`--allow-cross-base` para anularlo) y el rango (los rangos mixtos se rechazan, no se aproximan en silencio). Cada entrada pasa la compuerta de escaneo de puertas traseras de v0.71.2 (`--allow-unscanned` para omitirla). La salida es un único adaptador cargable; exit 0 = ok, 1 = rechazo.


## Soup Cans (recetas compartibles)

Comparte una receta reproducible como un solo archivo `.can`: un tarball del manifiesto, la configuración completa y una referencia a los datos de entrenamiento (URL o dataset de HF). No los pesos ni los bytes del dataset: sólo lo suficiente para que otra persona vuelva a ejecutar el mismo entrenamiento.

```bash
# Empaqueta una entrada del registro en un .can
soup can pack --entry-id llama31-chat-v1 --out ./llama31-chat.can

# Incrusta atestaciones in-toto en el manifiesto (cans v3; repetible)
soup can pack --entry-id llama31-chat-v1 --out ./signed.can --attest att.json

# Previsualiza el manifiesto sin extraer
soup can inspect ./llama31-chat.can

# Verifica el esquema + que la configuración se pueda analizar
soup can verify ./llama31-chat.can

# Haz un fork con modificaciones (overrides de rutas con puntos) y vuelve a empaquetar
soup can fork ./llama31-chat.can --out ./llama31-chat-hot.can \
  --modify training.lr=5e-5 --modify training.epochs=5

# Ejecuta un .can de extremo a extremo: extrae → entrena (→ deploy opcional)
soup can run ./llama31-chat.can --yes
soup can run ./llama31-chat.can --yes --deploy --env-capture ./env.txt

# Publica un .can en HF Hub como dataset
soup can publish ./llama31-chat.can --hf-hub me/llama31-chat-recipe
```

**Seguridad** — la extracción de tar usa `filter="data"` en Python 3.12+ con rechazo de symlinks/hardlinks como alternativa en runtimes más antiguos. Límite de tamaño: 100 MB. `DataRef.url` debe ser HTTPS. Los overrides de fork rechazan claves dunder (`__class__`, `__init__`) y bytes nulos. La versión del formato del manifiesto admite `1` y `2` (el incremento aditivo de v0.33.0 añadió `deploy_targets`). `soup can run` requiere `--yes` (consentimiento obligatorio: descarga datos + entrena automáticamente). `soup can publish` valida `repo_id` y resuelve el token de HF vía variables de entorno / archivos de caché; los mensajes de commit son la primera línea + recortados a 200 caracteres.



## Lista de materiales (`soup bom emit`)

Emite una lista de materiales (BOM) **CycloneDX 1.6 ML-BOM** o **SPDX 2.3 + perfil de IA** desde cualquier
ejecución de entrenamiento. Los equipos de compras y los auditores de cumplimiento pueden ingerir el BOM directamente en
sus herramientas existentes: no hace falta un parser personalizado.

```bash
soup bom emit \
  --name adapter-v1 --version 0.1.0 \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha aaaa...64hex \
  --config-sha bbbb...64hex \
  --task sft --license Apache-2.0 \
  --format both --output bom
# escribe bom.cdx.json + bom.spdx.json
```

El componente raíz es `type=machine-learning-model` (según la extensión ML-BOM de CycloneDX). El modelo
base + los adaptadores padre + los archivos por artefacto aparecen como componentes con hashes SHA-256.
`--license` se escribe por lo que es: un id SPDX de la lista (en cualquier capitalización, `apache-2.0` pasa a
`Apache-2.0`) va en `license.id` de CycloneDX y en `licenseConcluded` /
`licenseDeclared` de SPDX; una expresión SPDX válida (`Apache-2.0 OR MIT`, operadores en mayúsculas,
cada operando un id de la lista o un `LicenseRef-`) va en `expression` de CycloneDX y en esos
campos de SPDX con sus ids canonizados, y cada operando `LicenseRef-` se define en
`hasExtractedLicensingInfos` (un operando calificado con `DocumentRef-` convierte el valor en un nombre, ya que
el documento no lleva referencias externas); cualquier otra cosa, incluido un nombre que sólo contenga "and" o
"with" (`Gemma Terms of Use and Prohibited Use Policy`), es `license.name` de CycloneDX y un
`LicenseRef-` de SPDX con su texto en `hasExtractedLicensingInfos`. El
`serialNumber` de CycloneDX es un `urn:uuid:` de RFC 4122 (#1446).

### Adjuntar energía + CO₂ (`--energy`)

Pasa un JSON de medición para registrar la energía y el carbono en el BOM:

```bash
# 1. Entrena y persiste la energía medida (necesita `pip install soup-cli[carbon]`):
soup train --config soup.yaml --track-energy --energy-out energy.json

# 2. Adjúntala al BOM:
soup bom emit ... --format both --output bom --energy energy.json
```

`soup train --track-energy --energy-out <path>` escribe la lectura medida como JSON en
exactamente el esquema de energía (`energy_kwh` / `co2_kg` / `pue` / `grid_intensity_g_per_kwh` /
`source`); `soup bom emit --energy <path>` luego la consume. (También puedes escribir el
JSON a mano.) La ruta se valida (se mantiene bajo el cwd, se rechazan los symlinks) tanto en la escritura como en la
lectura. La energía aparece en **ambas** salidas: bajo `metadata.properties` (`soup:energy_kwh`,
`soup:co2_kg`, …) en CycloneDX, y como anotaciones `OTHER` en el paquete del modelo en SPDX.


## Atestaciones de procedencia (`soup attest emit`)

Emite un **Statement in-toto v1** que envuelve un **predicado de procedencia SLSA-3 v1** para cada
etapa del ciclo de vida de un Soup Can:

```bash
soup attest emit \
  --stage train \
  --subject adapter-v1 \
  --sha aaaa...64hex \
  --builder soup-cli@0.59.0 \
  --output att.json
```

Las etapas son una lista permitida cerrada: `extract` / `train` / `eval` / `export` / `publish`.
El SHA del sujeto debe ser de 64 hex (sha256). `--invocation "<command line>"` se registra en
`predicate.buildDefinition.externalParameters.invocation` (hasta 4096 caracteres; uno
más largo se rechaza, no se recorta), junto a la etapa; el statement que se imprime sin
`--output` es el mismo JSON, palabra por palabra (#1446). El backend por defecto `--sign unsigned`
sigue siendo metadatos de manipulación sólo sin conexión. El **backend `ed25519` está en vivo** con
`pip install soup-cli[sign]`; **Sigstore está en vivo** con
`pip install soup-cli[sigstore]`:

```bash
# firma con clave sin conexión
soup attest emit --stage train --subject adapter-v1 --sha aaaa...64hex \
  --sign ed25519 --key signing.pem --output att.json
soup attest verify att.json --signature att.json.sig --public-key trusted.pub

# OIDC / Fulcio / Rekor sin clave
soup attest emit --stage train --subject adapter-v1 --sha aaaa...64hex \
  --sign sigstore --output att.json  # añade --interactive-oidc sólo para autenticación por navegador
soup attest verify att.json --signature att.json.sig \
  --cert-identity 'https://github.com/acme/repo/.github/workflows/release.yml@refs/heads/main' \
  --cert-oidc-issuer 'https://token.actions.githubusercontent.com'
```

Ambos backends de firma usan el mismo sidecar JSON `<output>.sig`. Para Sigstore
contiene el Bundle moderno completo (certificado, firma y prueba de
transparencia). `verify` vuelve a canonicalizar el JSON del statement y exige tanto la
identidad del certificado de Sigstore como el emisor OIDC fuera de banda; cualquiera de los dos valores sin
el otro se rechaza. La emisión con Sigstore requiere `--output` para que la entrada pública de Rekor
nunca sobreviva a un bundle local descartado; el OIDC por navegador es una opción explícita.
La verificación sale con 3 ante manipulación o incumplimiento de política, y sale con 1 cuando la propia
verificación de Sigstore no está disponible (por ejemplo, fallo de TUF/red), de modo que una
caída operativa no se reporte como manipulación. Los errores de uso/entrada salen con 2.
Una petición explícita de firma con Sigstore falla en cerrado si OIDC/Fulcio/Rekor no está
disponible; nunca se degrada en silencio a unsigned. Una firma válida prueba que el firmante
*afirmó* el statement; no vuelve a verificar el digest del sujeto contra un
artefacto.


## Documentación automática de los Anexos XI/XII del EU AI Act (`soup train --annex-xi`)

Genera un Anexo XI del EU AI Act (documentación técnica, Secciones 1+2) o un Anexo XII
(resumen público del entrenamiento del Artículo 53(1)(d)) directamente desde una ejecución de entrenamiento:

```bash
soup train --config soup.yaml --annex-xi annex.md           # markdown
soup train --config soup.yaml --annex-xi annex.pdf           # PDF (pip install soup-cli[pdf])
soup train --config soup.yaml --track-energy --annex-xi annex.md  # + kWh/CO2 medidos
```

Los 10 dominios principales por proporción, desglose por modalidad, cómputo de entrenamiento / kWh / CO₂,
descripción del modelo, modelo base, id de ejecución. Una ruta de salida `.pdf` genera un PDF de reportlab (una ruta
`.md` genera markdown). Los **dominios rastreados principales** se extraen automáticamente del JSONL de
entrenamiento (`cfg.data.train`). Con `--track-energy`, la energía medida se registra en el
documento; sin él, las filas de energía y CO₂ dicen "no medido" y no 0, y la
fila de cómputo de entrenamiento (FLOPs) siempre dice "no medido", porque nada la estima
todavía (#1446). Los campos que controla el operador se neutralizan para escapes (`|[](){}!<>` + salto de línea / CR / tabulador)
para que un nombre de modelo malicioso no pueda inyectar un encabezado falsificado en los renderizadores posteriores.

### Medición de energía y CO₂ (`--track-energy`)

`soup train --track-energy` envuelve la ventana de entrenamiento en un tracker **offline** de codecarbon
(sin llamada de red para geolocalización por IP). Informa kWh / CO₂ / intensidad de la red y los alimenta a
`--annex-xi`. `--energy-country <ISO3>` elige la red para la estimación de CO₂ (por defecto `USA`;
la cifra de kWh en sí es independiente del país). Añade `--energy-out <path>` para persistir la
medición como JSON para `soup bom emit --energy <path>` (el traspaso a un ML-BOM;
consulta [Adjuntar energía + CO₂](#adjuntar-energía--co2---energy)). Requiere `pip install
soup-cli[carbon]`; sin él, `--track-energy` es un no-op inofensivo.


## Registro de auditoría (`soup audit-log`)

**Cada comando `soup`** ahora añade un registro con forma HIPAA/SOC2 al log de auditoría JSONL en
`~/.soup/audit.jsonl` (se cambia con `SOUP_AUDIT_LOG_PATH`, verificado como contenido en
`$HOME / $CWD / $TMPDIR`). Omítelo por invocación con `soup --no-audit-log <cmd>` o
globalmente con `SOUP_NO_AUDIT_LOG=1`. Consulta el final o rota:

```bash
soup audit-log tail --limit 50          # vista de tabla de Rich
soup audit-log tail --json              # JSONL sin procesar para ingesta en SIEM
soup audit-log rotate --cap-mb 100      # fuerza una pasada de rotación
soup --no-audit-log version             # omite la línea de auditoría para este comando
```

La redacción de PII se aplica en **todos** los campos de cadena (`hf_*` / `sk-*` / `Bearer …` → `<redacted>`)
mediante la política `_SECRET_RE` de v0.40.3. `O_NOFOLLOW` de POSIX + permisos `0o600`, append atómico,
rotación a los 100 MiB con rechazo de symlinks en la ruta de respaldo. La auto-instrumentación es
de mejor esfuerzo: un log de auditoría roto nunca hace fallar el CLI.


## Recibo de reproducibilidad (`soup train --repro-receipt`)

El recibo de reproducibilidad al estilo SR 11-7 captura las semillas (torch + numpy + python), las versiones
de los kernels (CUDA + cuDNN + NCCL), el backend del acelerador, el modelo de GPU + el driver, el SO + la arquitectura.
En Apple Silicon registra de forma segura para la privacidad el nombre del chip y la capacidad de memoria unificada, pero
nunca un número de serie, un UUID de hardware ni una ruta específica del usuario:

```bash
soup train --config soup.yaml --repro-receipt repro.json
```

Los equipos de riesgo de modelo de un banco y los auditores de organizaciones reguladas obtienen un solo archivo JSON que identifica
el entorno exacto en el que se ejecutó la corrida. Escritura atómica, contenido en el cwd.


## Auditoría de ejecución frente a la configuración (`soup adapters audit`)

¿El adaptador terminado coincide con la configuración que lo pidió? Cinco
arreglos ya fusionados le enseñaron a la ruta de MLX a registrar lo que en realidad hizo en
`adapter_config.json` — #683 (masking), #684 (accumulation), #685 (gradient
checkpointing), #686 (optimizer and schedule), #749 (gradient clipping). Cada uno
existía porque un ajuste se aceptaba y se descartaba en silencio. Nada volvía a leer ese
registro hasta este comando.

```bash
soup adapters audit ./output --config soup.yaml
soup adapters audit ./output --config soup.yaml --json   # machine-readable
```

```
Audit: output
│ optimizer                    │ adamw_torch │ AdamW  │ ok      │
│ learning_rate                │ 0.0001      │ 0.0001 │ ok      │
│ warmup_ratio                 │ 0.03        │ —      │ unknown │
│ weight_decay                 │ 0.01        │ 0.01   │ ok      │
│ lora.r                       │ 8           │ 8      │ ok      │
```

**`unknown` nunca es `ok`.** Un ajuste del que el registro no puede hablar se informa
`unknown`, nunca como coincidencia — un visto bueno falso convierte "no lo sé" en "lo
verifiqué", que es exactamente la sustitución que este comando existe para deshacer. Tampoco
es un fallo: un adaptador más antiguo es anterior a las claves, y eso no es
culpa del usuario.

La configuración se lee a través del esquema, así que un ajuste que omitiste se audita
contra el valor por defecto que Soup habría usado realmente, no contra una segunda copia
de los valores por defecto que guarde la auditoría.

**El comparando de la tasa de aprendizaje en MLX es `peak_lr`.** `training.lr` es el valor
objetivo/base de pico que se pasa al plan del optimizador ya resuelto. El warmup empieza por debajo
de él y controla cuándo el plan lo alcanza; la caída coseno o lineal controla
lo que viene después. Comparar `training.lr` con una muestra temprana del plan
marcaría por tanto una ejecución de warmup correcta. La auditoría lo compara en cambio con el
`peak_lr` registrado en el plan del optimizador. Deliberadamente no usa el
`learning_rate` del registro, porque ese campo es sólo el valor de configuración repetido cuando se
escribe el archivo, no evidencia de lo que llegó al optimizador.

El `adapter_config.json` de PEFT en el backend de transformers no registra la
tasa de aprendizaje efectiva del optimizador. Esa fila es por tanto `unknown`, y la
salida dice explícitamente que la tasa de aprendizaje efectiva **no se verificó**;
nunca le da a un adaptador de transformers un visto bueno falso.

**El masking se audita por efecto, no por petición.** El
`train_on_responses_only` del registro es la propia petición de la configuración repetida; el efecto
es `mask_prompt` (el único prefijo enmascarado de upstream) y `response_token_mask`
(la máscara por token de Soup). Las filas de texto plano no llevan límites de rol, así que una ejecución
que pidió masking sólo de respuesta sobre ellas advierte una vez y entrena sobre la
secuencia completa — dejando un registro que dice `train_on_responses_only: true` junto a
dos claves de efecto en falso. La auditoría compara contra el efecto, así que esa ejecución se
informa como `DIVERGED`.

La evidencia se pesa igual en todo el resto de este comando: una clave verdadera
prueba que el masking *ocurrió*, diga lo que diga la otra, pero probar que **no**
ocurrió necesita ambas claves. Un registro que no lleve ninguna (escrito antes del
#683) — o que lleve una clave en falso con la otra ausente — es `unknown`, no una
divergencia, porque una clave ausente leída como `False` reclamaría una verificación que el
registro no puede respaldar.

`--json` emite `checked_count` junto a `diverged_count` y `unknown_count`,
así que un job de CI puede distinguir "todo coincidía" de "nada era verificable" — ambos
con salida 0 — además de `unknown_reason`, la misma explicación que imprime
la tabla. Las advertencias de configuración (una clave desconocida, por ejemplo) van a stderr, así que stdout
con `--json` es la carga útil y nada más.

**Códigos de salida: 0 = coincidencia, 2 = DIVERGED, 1 = error de uso o de lectura.** El
veredicto se mantiene fuera del `1` para que una compuerta de CI pueda distinguir "la ejecución no hizo lo que
pedía la configuración" de "la ruta estaba mal" -- un `adapter_config.json` ausente, un
`--config` ausente y una ruta fuera del directorio de trabajo salen todos con `1`.
Esto sigue a `soup ship` / `soup shrink` (0 pasa / 2 compuerta fallida / 1 error)
y no al más antiguo `adapters scan`, que es anterior a esa convención.
Las filas `unknown` salen con `0`.  En ejecuciones GRPO, cualquier `nan_skip_fraction` distinto de cero
se lee como `DIVERGED` (salida 2); incluso un solo step omitido de miles se
expone, para que el usuario decida si importa.

Las cadenas de `adapter_config.json` no son de confianza -- un adaptador puede descargarse
-- así que el texto derivado del registro se limpia de bytes de control ANSI/OSC y se escapa
contra el marcado de Rich antes de imprimirlo. `--json` no se sanea: un
consumidor automático recibe los bytes que el registro realmente contiene.

## Escáner de puertas traseras en adaptadores (`soup adapters scan`)

Análisis espectral de los pesos del adaptador LoRA antes de cargarlo. Marca el predominio de rango 1
(el patrón canónico de troyano en el espacio de pesos), la concentración de energía en el
vector singular top-1, NaN/Inf en los pesos y valores atípicos de norma de Frobenius mediante
agrupación robusta por mediana + MAD. Numpy puro, sin torch. Códigos de salida 0=OK / 1=WARN /
3=FAIL para que CI pueda buscar específicamente fallos de seguridad. Reutiliza el
cargador `adapter_diff` de v0.57.0 para que la superficie en disco siga teniendo una sola fuente.


## Firma y verificación de adaptadores (`soup adapters sign` / `verify`)

Manifiesto determinista de raíz Merkle sobre cada archivo del directorio del adaptador
(incluidos los subdirectorios anidados `tokenizer/` / `processor/`). Manipula cualquier archivo
y `verify` falla. El backend `unsigned` ofrece detección de manipulación sin conexión
(el hash de raíz Merkle es el ancla de confianza). El **backend `ed25519` está en vivo**
(`pip install soup-cli[sign]`): añade una firma separada real sobre la
raíz Merkle para la *autenticación* — prueba de que el firmante tiene la clave privada.

```bash
# genera un par de claves + firma de una sola vez
soup adapters sign ./adapter --backend ed25519 --generate-key signing.pem
# o firma con una clave existente (o define SOUP_SIGNING_KEY)
soup adapters sign ./adapter --backend ed25519 --key signing.pem
# verificación de auto-consistencia (clave pública incrustada)
soup adapters verify ./adapter
# autenticación genuina: exige que la clave del firmante coincida con una de confianza
soup adapters verify ./adapter --public-key trusted.pub
```

`verify` falla en cerrado: cualquier manipulación, una clave incorrecta/ausente/ilegible o una
firma de una clave no confiable marca el adaptador como inválido. Con `--public-key`,
un adaptador cuyo registro de firma no sea `ed25519` también es inválido. Las claves de firma y
las claves públicas de confianza se rechazan si son symlinks y tienen límite de tamaño, pero **no**
están contenidas en el cwd (las claves son secretos que viven fuera del proyecto). La firma
persiste como `.soup-signature.json` (escritura atómica).

El **backend `sigstore` también está en vivo** con `pip install soup-cli[sigstore]`.
Usa una identidad OIDC ambiente cuando hay una disponible (por ejemplo GitHub
Actions), pide un certificado a Fulcio, lo envía a Rekor y guarda el
bundle completo de Sigstore junto al registro de firma del adaptador. El OIDC por navegador
**no** se abre de forma implícita en ejecuciones sin interfaz o por defecto; actívalo explícitamente con
`--interactive-oidc`. La verificación es deliberadamente de fallo en cerrado y exige
tanto la identidad del certificado como el emisor OIDC suministrados fuera de banda:

```bash
soup adapters sign ./adapter --backend sigstore
soup adapters verify ./adapter \
  --cert-identity 'https://github.com/acme/repo/.github/workflows/release.yml@refs/heads/main' \
  --cert-oidc-issuer 'https://token.actions.githubusercontent.com'
```

La identidad incrustada en el bundle **no** se confía automáticamente; hacerlo
reduciría la autenticación a auto-consistencia. La identidad y el emisor son una sola
política de certificado: cualquiera de los dos valores sin el otro se rechaza. El modo `--strict`
sale con 3 ante cualquier fallo de verificación
(código de compuerta de CI distinto de los errores genéricos).


## Modo estricto de safetensors (`soup adapters check-safetensors`)

Rechaza los pesos pickle / clásicos de PyTorch en la frontera — lista permitida cerrada de 8
extensiones inseguras (`.bin` / `.pt` / `.pth` / `.ckpt` / `.pkl` /
`.pickle` / `.joblib` / `.msgpack`). Picklemod le da a cada cargador el
derecho de ejecutar código arbitrario al cargar; rechazar el archivo en la
frontera es la única mitigación sólida. Un aviso amable nombra el
archivo infractor y la receta canónica `from safetensors.torch import save_file`.
Código de salida 3 con `--strict` para la compuerta de CI.


## Auditoría de salud del adaptador previa al vuelo (`soup adapters check`)

Escaneo previo al vuelo que audita adaptadores LoRA individuales en busca de fallos de entrenamiento silenciosos (#1721):
- **||dW||_F total y por capa**: norma de Frobenius de las actualizaciones de los parámetros LoRA (dW = scaling * (B @ A), con alpha / r para LoRA estándar y alpha / sqrt(r) para rsLoRA) con acumulación en float64 mediante contracción de la traza de la matriz.
- **Fracción viva**: proporción de capas del adaptador con actualizaciones activas y distintas de cero.
- **Capas lora_B totalmente en cero**: marca capas sin entrenar o inicializadas en cero que no se movieron durante el entrenamiento.
- **Pares LoRA incompletos**: marca proyecciones huérfanas o guardadas parcialmente a las que les falta A o B.
- **Claves .inner. filtradas**: marca namespaces de wrapper heredados en adaptadores guardados antes del guard de guardado del #1011.

```bash
soup adapters check path/to/adapter/
# Salida legible por máquina
soup adapters check path/to/adapter/ --json
```

Muestra diagnósticos pensados para humanos que terminan en una sola línea de veredicto: `alive` (código de salida 0) o `inactive: <reason>` (código de salida 2 para inactividad/fallo silencioso, distinto del error de uso 1). Entre los posibles motivos de inactividad están:
- `all lora_B layers are zero (n/n)`
- `lora_B layers are zero (k/n)`
- `total ||ΔW||_F is 0.0`
- `leaked .inner. keys detected (<n> tensors)`
- `incomplete LoRA pairs detected (<n> orphans)`
- `shape mismatch in LoRA projections (<n> layers)`
- `non-finite weights detected (NaN/Inf)`
- `no lora weights found`


## Fijación de namespace (anti-AI-jacking)

Caché SQLite de confianza en el primer uso: registra `(repo_id, author, created_at)`
la primera vez que Soup ve un repositorio de HuggingFace. Las cargas posteriores comparan
la huella actual del Hub con el pin registrado y se niegan ante un cambio de autor
o un salto hacia atrás de `created_at`, salvo que el operador pase
`--allow-namespace-shift <new-author>` (coincidencia de autor sin distinguir mayúsculas;
se rechaza un bool para que quien llame no pueda colar un bypass para todo). Modelo de
amenaza: un atacante vigila un repositorio popular, espera a que el dueño original
lo borre o expire y vuelve a crear el mismo `owner/name` con
pesos maliciosos. Sin este control, cualquiera con Soup fijado a ese
namespace descarga veneno en silencio en la siguiente ejecución. **La compuerta está conectada a
las descargas de modelos del Hub**: consulta el pin antes de descargar y rechaza una
discrepancia (lanza antes de que aterrice ningún peso). Falla **abierto** cuando no se pueden
obtener los metadatos del repositorio (sin conexión / limitado por tasa), para que un hipo transitorio
nunca bloquee una descarga legítima. El almacén de pines usa WAL de SQLite + un
lock de archivo entre procesos para que las descargas concurrentes no pierdan el ancla de confianza.


## Matriz de conflictos de licencias en la fusión

Tabla de compatibilidad cerrada sobre 33 licencias al estilo SPDX que abarcan Apache /
MIT / BSD / LGPL / MPL / GPL / AGPL / CC-BY / CC-BY-NC / Llama-2/3.x /
Gemma / Qwen-research / Mistral-research / OpenRAIL / OpenAI-ToS /
Anthropic-AUP. `soup adapters merge --license apache-2.0 --license
cc-by-nc-4.0 -o merged` se niega (una no comercial no puede combinarse con
una permisiva). Cuando no pasas `--license`, la licencia se
**detecta automáticamente** desde el `adapter_config.json` / `config.json`
/ el frontmatter de la ficha del modelo de cada adaptador (los ids al estilo `llama3.1` de HF se asignan a los canónicos) y la
compuerta se ejecuta sola cuando todos los adaptadores resuelven a una licencia conocida. Para
pasar por encima de un conflicto marcado, pasa `--license-override "legal-cleared
2026-05-19 by alice"` (motivo de 8 caracteres mínimo y 4096 máximo) — la decisión queda
**registrada en el log de auditoría** para una revisión legal posterior.

La fusión además pasa por la compuerta del escáner de puertas traseras: cualquier entrada que
`soup adapters scan` marque como FAIL (o que no se pueda escanear) rechaza la
fusión con salida 3, salvo que pases `--allow-unscanned`.


## Paquete airgap (`soup airgap-bundle`)

Un solo tarball firmado con el modelo + los datasets + los wheels + los kernels de CUDA +
un `manifest.json` incrustado que lista el SHA-256 por archivo. Dimensionado para transferencia
unidireccional por medio físico a través de un diodo de datos. Un recibo de reproducibilidad se
incrusta como `repro-receipt.json` de nivel superior cuando pasas
`--repro-receipt <receipt.json>` (o se detecta automáticamente en
`<model>/repro-receipt.json`):

```bash
soup airgap-bundle --model ./out --output bundle.tar --repro-receipt repro.json
```

Tope de 100 GiB por defecto;
se niega ante tamaños mayores. Etiquetado determinista de datasets por nombre base ordenado
(NO por el orden de argv), para que las mismas entradas en distinto orden de argv produzcan
manifiestos idénticos. Defensa TOCTOU: nueva comprobación con `os.lstat + S_ISLNK` en
el directorio padre + la ruta de salida final antes de `mkstemp`; `os.replace` atómico desde un
archivo temporal hermano. `tarfile.data_filter` se ajusta en Python 3.12+ para que cualquier
llamador futuro que añada `tar.extractall` obtenga automáticamente el filtro de
extracción en modo seguro. `soup airgap-bundle` es a propósito un comando de nivel
superior (no `soup deploy airgap-bundle`): es una operación de exportación,
no un destino de deploy.
