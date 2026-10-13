# Contrato de artefactos de AutoDistill (Hito A)

> 🇪🇸 Traducción al español de [`autodistill.md`](autodistill.md). Ante cualquier duda, manda el original en inglés.

> Estado: corte de diseño y contrato para el [issue #580](https://github.com/MakazhanAlpamys/Soup/issues/580).
> Este documento es normativo para la versión 1 del artefacto. **No** agrega una tarea,
> campos de configuración, carga de modelos, captura, entrenamiento, asignación adaptativa ni un ejecutor de benchmark.

## 1. Objetivo y límite duro

AutoDistill es un bucle acotado futuro que descubre las debilidades del estudiante, le pide a un
docente local material dirigido, lo verifica, captura probabilidades reutilizables del docente,
entrena al estudiante y repite sólo mientras las ganancias medidas justifiquen el costo.

El invariante que motiva el formato offline es:

> Un proceso exclusivo del docente debe publicar y liberar toda la memoria del docente antes de
> que un proceso exclusivo del estudiante pueda consumir sus artefactos.

La versión 1 tiene un segundo límite duro: el docente y el estudiante usan el **mismo fingerprint de
tokenizador**. Un nombre de repositorio, un tamaño de vocabulario igual o cadenas de tokens
similares no bastan. La alineación de spans visibles entre tokenizadores, ULD y la KD de
secuencias quedan para más adelante y son objetivos explícitamente distintos.

## 2. No objetivos y límite de afirmaciones

El Hito A no elige un `top_k` por defecto, ni un método de compresión, ni un dtype de probabilidad,
ni cuantización del docente, ni tamaño de muestra de benchmark, ni umbral de calidad. No afirma
que la KD dispersa reproduzca la KD densa cuando `k < vocab_size`, ni que una pérdida de entrenamiento menor
signifique un modelo mejor.

El código de la v1 es intencionalmente independiente de `SoupConfig` y no importa ningún runtime de ML.
`task: autodistill` sigue siendo un nombre ilustrativo hasta que una revisión posterior acepte una CLI y una
superficie de configuración.

## 3. Arquitectura de bucle cerrado

El flujo de trabajo final tiene procesos separados y acotados:

1. Congelar un fingerprint del estudiante, una sonda de capacidad privada, los manifiestos
   de benchmarks públicos seleccionados, los presupuestos y las reglas de parada.
2. Sondear al docente y al estudiante con tareas generadas sin exponer prompts ni respuestas
   de la evaluación privada.
3. Admitir sólo las filas donde la verificación establezca una ventaja útil del docente.
4. Ejecutar la captura exclusiva del docente y publicar de forma transaccional shards inmutables.
5. Salir del proceso del docente y verificar que los fingerprints y hashes de los artefactos coincidan.
6. Ejecutar el entrenamiento exclusivo del estudiante contra shards de sólo lectura y confirmar el consumo sólo con
   un hash durable de checkpoint.
7. Reevaluar la sonda congelada y los benchmarks públicos seleccionados con los mismos IDs
   de ítem, harness, scaffold, presupuesto de tokens y versión.
8. Asignar un nuevo presupuesto acotado de generación o detenerse.

El Hito A implementa sólo los contratos necesarios alrededor de los pasos 4-6 y la aritmética
necesaria para planificar el paso 4.

## 4. Familia de artefactos versionada

Cada objeto JSON tiene un discriminador `schema` exacto. Los lectores rechazan versiones mayores
desconocidas y campos desconocidos. El JSON canónico es UTF-8, ordenado por clave, no tiene espacios
en blanco insignificantes, no permite valores NaN/Infinity y se hashea con SHA-256.

| Artefacto | Esquema | Propósito |
|---|---|---|
| Plan | `soup.autodistill.plan.v1` | Identidades inmutables, política de captura, política de consumo y estimación |
| Fila de token | `soup.autodistill.capture-token.v1` | IDs de prefijo/objetivo exactos y probabilidades seleccionada/cola del docente |
| Manifiesto de shard | `soup.autodistill.shard-manifest.v1` | Estado de la transacción y compromisos exactos de payload |
| Evento de consumo | `soup.autodistill.consumption-event.v1` | Libro mayor append-only de reserva, liberación, commit y replay explícito |

La implementación está en `soup_cli.autodistill.contract`. Los modelos de Pydantic son frozen
y rechazan campos extra. Los alias de JSON conservan la clave serializada `schema` aunque el
atributo de Python sea `schema_id`.

### 4.1 Plan

El plan vincula todas las entradas que pueden alterar los bytes capturados:

- identificadores de modelo del docente y del estudiante, revisiones hex inmutables de 40-64,
  hashes de `config.json` y la lista ordenada de ruta/tamaño/hash de los archivos de pesos;
- un identificador de tokenizador compartido, revisión inmutable, tamaño de vocabulario, lista de
  ruta/tamaño/hash del archivo de tokenizador, hash de la plantilla de chat y nombre/versión del renderer;
- lista ordenada de ruta/tamaño/hash de los datos de origen, número de filas, algoritmo de normalización y
  hash del contenido normalizado;
- backend y versión, dtype, etiqueta de cuantización, límite de secuencia, dirección de truncamiento,
  número de tokens planificado y máximo de IDs forzados adicionales por posición;
- las políticas completas de probabilidad y consumo;
- un perfil de throughput medido previamente, opcional, y su hash;
- la estimación exacta de almacenamiento/runtime sin modelo.

El SHA-256 del JSON canónico del plan es el `plan_sha256` incrustado en cada shard.
Cambiar cualquier identidad o política crea una ejecución distinta; nunca es una reanudación.

### 4.2 Normalización de datos

`soup-jsonl-c14n-v1` es determinista:

1. decodificar los bytes como UTF-8, aceptando y eliminando un BOM UTF-8 inicial;
2. dividir las formas de salto de línea universales preservando el orden de las filas;
3. rechazar filas en blanco, claves de objeto duplicadas, JSON malformado y raíces que no sean objetos;
4. serializar cada objeto como JSON canónico sin cambiar los puntos de código Unicode;
5. terminar cada fila con un byte LF.

Se conservan tanto los hashes de los bytes de origen como el hash normalizado. Los hashes de origen detectan un
cambio exacto de archivo; el hash normalizado permite que variantes de CRLF o de sólo formato identifiquen las
mismas filas lógicas ordenadas.

### 4.3 Token de captura

Cada fila de token de la v1 vincula:

- ID de ejemplo, tipo de trayectoria, posición, tamaño de vocabulario;
- IDs de token de contexto exactos;
- ID de token objetivo del docente o ID de token muestreado del estudiante, según la trayectoria;
- IDs top-k, IDs forzados, su unión ordenada y un log-probabilidad del docente por cada ID seleccionado;
- masa de cola residual, entropía completa del docente y temperatura.

Los shards de producción pueden usar más adelante un diseño de columna binaria más denso, pero debe preservar
estas semánticas exactamente y quedar identificado por un nuevo schema si su interpretación cambia.

### 4.4 Manifiesto de shard

Un manifiesto de shard contiene un ID portátil de shard/transacción, estado, hash del plan, hash del manifiesto
precedente, recuentos lógicos de fila/token y una lista ordenada de compromisos de payload de ruta/byte/fila/token/
SHA-256. Un manifiesto `staging` no tiene predecesor. Cada estado posterior
debe nombrar el hash canónico de su manifiesto precedente para que una transición no pueda reemplazar el historial
de forma silenciosa.

### 4.5 Evento de consumo

El consumo es un libro mayor append-only sobre un artefacto de origen inmutable. Cada evento
vincula el hash del artefacto, la vista de trayectoria, el estado previo/siguiente, los IDs de ejecución/reserva, la secuencia,
el hash de checkpoint opcional y la referencia de replay opcional. La secuencia comienza en cero,
es contigua, nunca mezcla artefactos ni vistas y debe formar una cadena de estados válida.

## 5. Política de probabilidad faltante

Para una distribución del docente `P` en una posición, define:

```text
S = teacher_top_k_ids union forced_token_ids
```

Los IDs forzados son el objetivo de referencia (ground-truth) en las trayectorias del docente experto y el token
muestreado del estudiante en las trayectorias de rollout del estudiante. Si tal ID ya está en top-k, se
almacena una sola vez.

El artefacto almacena, de forma explícita y sin valores por defecto:

- `log p_i` exacto del docente para cada `i in S`;
- `p_tail = 1 - sum(exp(log p_i) for i in S)`;
- entropía completa del docente, temperatura, ancho de ID, ancho de probabilidad, ancho de cola y ancho de entropía;
- `renormalize_selected = false`.

Las probabilidades top-k nunca se renormalizan a uno. Hacerlo descarta si el docente
asignó 1%, 20% u 80% al vocabulario omitido y cambia el objetivo de forma silenciosa.

Dadas las probabilidades del estudiante sobre los mismos IDs seleccionados y su masa residual, la KL
forward de cola gruesa de la v1 es:

```text
D_tail(P || Q)
  = sum(i in S) p_i * log(p_i / q_i)
  + p_tail * log(p_tail / q_tail)
```

Cuando `S` es el vocabulario completo, ambas colas son cero y esto equivale a la KL forward
densa. Cuando `S` es menor, la cola es un único bucket grueso: esto es una aproximación de
menor resolución de procesamiento de datos, no una equivalencia densa ni una reconstrucción de la
distribución dentro de la cola.

No puede introducirse ningún `top_k` por defecto, compresión polinómica/residual ni afirmación de “pérdida de calidad mínima”
hasta que la reconstrucción densa-vs-dispersa y las ablaciones posteriores cubran el
par de modelos, el dtype, el backend y la combinación de tareas reales.

## 6. Niveles de fidelidad del docente

La inferencia del docente en BF16 es preferible cuando cabe, pero no puede exigirse de forma universal.
El futuro controlador de captura debe clasificar la evidencia antes de elegir una pérdida:

| Nivel | Evidencia | Uso permitido |
|---|---|---|
| Distribuciones confiables | BF16/FP16 o logits cuantizados calibrados pasan una sonda A/B de fidelidad diminuta | KD dispersa más CE/preferencias verificadas |
| Sólo salidas confiables | Las respuestas se verifican, pero la fidelidad de los logits cuantizados no | Datos de SFT verificado y de preferencia/reparación; sin afirmación de KD de logits |
| Sin ventaja verificada | El docente no supera de forma fiable al estudiante/verificador | Sólo propuesta de tarea o deshabilitar el docente para ese estrato |

Un docente cuantizado, por lo tanto, se degrada según la evidencia, no por una prohibición fija. Un piloto debe
comparar SFT verificado solo contra SFT verificado más KD con los mismos tokens de entrenamiento del estudiante.

## 7. Máquinas de estado

### 7.1 Admisión de ejemplos

```text
proposed -> probed -> captured -> verified -> admitted
                    |            |          -> rejected
                    |            -> quarantined
                    -> rejected / quarantined
```

Las transiciones omitidas fallan. `admitted`, `rejected` y `quarantined` son registros terminales;
la reconsideración crea un nuevo ejemplo/versión en lugar de reescribir la decisión anterior.

El futuro generador adaptativo debería interpretar los resultados de la sonda así:

| Estudiante | Docente | Decisión |
|---|---|---|
| pass | pass | dominado; reducir el muestreo o aumentar la dificultad |
| fail | pass | muestra de frontera ideal |
| fail | fail | simplificar, cambiar docente/verificador o poner en cuarentena |
| pass | fail | no destilar la respuesta del docente |

### 7.2 Publicación de shards

```text
staging -> complete -> verified -> available
    |          |           |           |
    +----------+-----------+-----------+-> quarantined
```

Sólo un estudiante puede reservar shards `available`. La corrupción descubierta después de la
publicación mueve la vista del registro a `quarantined`; los bytes del payload no se reparan
en el lugar.

### 7.3 Consumo y replay

```text
available -> reserved -> committed
               |
               +-> available   (release after crash before durable checkpoint)
```

- Un commit requiere el SHA-256 exacto del checkpoint durable del estudiante.
- Una reserva por sí sola nunca prueba el consumo.
- Las trayectorias de rollout del estudiante son señales de una ronda y de consumo único. El replay después del commit
  está prohibido porque los prefijos quedan obsoletos con respecto al estudiante actualizado.
- Los artefactos del docente experto pueden reproducirse (replay) sólo mediante una nueva reserva cuyo
  `replay_of` sea igual al hash del evento confirmado anterior. El artefacto de origen permanece inmutable.
- La política de replay es explícita en cada plan; no se infiere de un nombre de archivo ni de una tarea.

### 7.4 Frontera del docente MLX en el Hito B1

El primer worker de captura para Apple Silicon es una superficie de implementación interna y opcional; no
agrega una tarea, un comando de CLI ni un campo de `SoupConfig`. El controlador inicia un proceso de Python nuevo
cuya solicitud acepta raíces de docente, tokenizador, dataset y publicación, pero ninguna raíz
de estudiante. El controlador escribe su recibo final sólo después de que ese proceso hijo haya salido con éxito y
se haya reabierto el manifiesto `available` exacto. El plan aún contiene el fingerprint inmutable del estudiante
como metadato del consumidor futuro; nunca se resuelve ni se carga durante la captura.

Para esta frontera, MLX-LM carga el modelo del docente desde su raíz de checkpoint local inmutable y
carga el tokenizador compartido canónico por separado desde una raíz de sólo tokenizador. Esa raíz de tokenizador
puede derivarse de los bytes del tokenizador del estudiante, pero no contiene pesos del estudiante y el
worker nunca resuelve un modelo de estudiante. `capture.backend_version` es la versión exacta de la distribución
`mlx-lm` instalada y el renderer del tokenizador es `mlx-lm@<version>`. Ambos deben coincidir con el plan
antes de la publicación. MLX/MLX-LM siguen siendo importaciones perezosas dentro del worker.

El worker también verifica los dtypes de los parámetros flotantes cargados contra `capture.dtype`. Un
checkpoint sin cuantizar debe exponer exactamente ese dtype y usa la identidad literal de cuantización
`none`. Un checkpoint cuantizado debe exponer el dtype base declarado; se permiten parámetros auxiliares float32
mientras que cualquier otro dtype flotante no declarado falla de forma cerrada. Si `config.json` contiene
`quantization` o `quantization_config`, su identidad es
`config-sha256:<sha256(canonical JSON of those active fields)>`. Esto vincula el plan a la receta
de cuantización exacta sin pretender que nombres como `4-bit` identifiquen de forma única un runtime.
El dtype de inferencia declarado, todos los dtypes de parámetros flotantes observados y la identidad
de cuantización se repiten en el recibo del proceso hijo.

El dataset vinculado es JSONL canónico con filas versionadas y ya tokenizadas:

```json
{"schema":"soup.autodistill.tokenized-teacher-example.v1","example_id":"ex-1","prompt_token_ids":[1,2],"target_token_ids":[3,4]}
```

Los IDs de prompt no están vacíos porque una distribución causal del siguiente token necesita una posición previa.
Cada byte de origen, fila normalizada, archivo de tokenizador, config del docente y peso del docente listado se
verifica antes de cargar el modelo. Una solicitud del worker selecciona un rango de ejemplos semiabierto, pero el fingerprint
completo del dataset y el total planificado de tokens objetivo se verifican antes de aceptar ese subconjunto de
shard. Los rangos nunca dividen la trayectoria de un ejemplo.

Sin truncamiento, un forward causal captura todas las posiciones objetivo de un ejemplo. Una vez que el
límite de secuencia declarado exige truncamiento, el worker evalúa el contexto exacto registrado para
cada posición. Sólo la fila final del vocabulario se convierte a valores host float32; los IDs seleccionados no
se renormalizan. La salida del proceso, y no la mera reclamación de cache, es la barrera de aislamiento de memoria.

## 8. Publicación transaccional, reanudación y corrupción

El escritor de captura sigue un protocolo de manifiesto escritura-al-final sobre un solo sistema de archivos:

1. Crear un directorio de staging específico de la transacción que los consumidores nunca escanean.
2. Escribir los payloads en archivos vecinos temporales; hacer flush y sync antes de reemplazar.
3. Calcular los hashes y los recuentos lógicos a partir de los bytes exactos en disco.
4. Escribir el manifiesto `complete` de forma atómica y encadenarlo al hash del manifiesto de staging.
5. Reabrir cada payload, verificar pertenencia, recuento de bytes, SHA-256, parseabilidad, recuento de filas,
   recuento de tokens, hash del plan y fingerprints de tokenizador/datos.
6. Escribir el manifiesto `verified` de forma atómica.
7. Publicar los nombres de los payloads como grupo y luego escribir de forma atómica el manifiesto `available` al final.

Si la publicación agrupada falla, se restaura el grupo available anterior o no queda visible ningún
grupo. Un manifiesto sin todos los payloads coincidentes es corrupción, nunca un éxito parcial.

Las decisiones de reanudación son cerradas y deterministas:

| Estado observado | Integridad/fingerprints | Acción |
|---|---|---|
| `staging` | válido/coincidente | reanudar la transacción de staging |
| `complete` | válido/coincidente | verificar y luego publicar |
| `verified` | válido/coincidente | publicar |
| `available` | válido/coincidente | reutilizar sin recapturar |
| `quarantined` | cualquiera | rechazar |
| cualquier otro estado | corrupto o no coincidente | poner en cuarentena e iniciar una nueva transacción |

Un fragmento final no confirmado puede truncarse sólo cuando sus bytes quedan fuera de un límite
de registro confirmado. Una fila modificada parseable, una posición duplicada/reordenada, un plan no coincidente,
una revisión de modelo cambiada, una falta de coincidencia de tokenizador o un hash de payload no coincidente falla de forma cerrada. La reanudación
nunca debe mezclar filas de dos planes ni duplicar un prefijo ya confirmado.

## 9. Estimación sólo del plan

La estimación sólo del plan acepta metadatos y throughput en caché opcional; no resuelve un hub,
ni instancia un tokenizador, ni inspecciona un dispositivo, ni importa Torch, Transformers ni MLX.

Para:

- `N`: número planificado de tokens capturados;
- `V`: tamaño del vocabulario;
- `K`: top-k explícito;
- `F`: máximo de IDs forzados adicionales por posición;
- `B_id`, `B_logp`, `B_tail`, `B_entropy`: anchos de almacenamiento explícitos;

los bytes exactos de log-probabilidad densa sin procesar son:

```text
dense_bytes = N * V * B_logp
```

La cota superior dispersa sin procesar de la v1 es:

```text
S_max = min(V, K + F)
sparse_upper_bytes
  = N * (S_max * (B_id + B_logp) + B_tail + B_entropy)
```

Esto excluye deliberadamente los metadatos de contenedor/índice/JSON y declara esa exclusión.
La salida real debe informar tanto los bytes/token planificados como los observados.

El runtime es `unknown` cuando no se proporciona un perfil medido compatible. Dado un rango de
extremo a extremo medido previamente `[throughput_min, throughput_max]`, la estimación sólo del plan informa:

```text
seconds_min = N / throughput_max
seconds_max = N / throughput_min
```

El hash del perfil, el backend, el fingerprint del modelo, el dtype/cuantización, la banda de longitud de secuencia
y la clase de hardware deben acompañar a un perfil real futuro. La estimación sólo del plan no debe medir un
modelo para que la estimación parezca completa.

## 10. Currículo adaptativo (hito posterior)

Las tareas no deben venir de una única lista permanente. Un asignador posterior debería mantener estratos por
dominio, subhabilidad, dificultad, idioma, formato y tipo de verificador. Debería combinar:

- debilidad e incertidumbre normalizadas del estudiante;
- proximidad a la frontera docente-estudiante;
- ventaja verificada del docente;
- fiabilidad del verificador;
- déficit de novedad/cobertura;
- ganancia medida por token de entrenamiento del estudiante;
- costo de generación del docente, almacenamiento y entrenamiento;
- un piso de exploración acotado.

La dificultad aumenta sólo después de que pasan los estratos más fáciles. Las filas repetidas de doble fallo o de
desacuerdo del verificador se simplifican o se ponen en cuarentena en lugar de etiquetarse a la fuerza. Cada bucle tiene
topes obligatorios de tokens del docente, tokens del estudiante, almacenamiento, tiempo de reloj, energía e iteraciones,
además de paradas por meseta y por regresión general/de seguridad.

La futura familia de objetivos puede combinar CE verificada, KD forward dispersa sobre trayectorias del docente,
KD dispersa consciente de la entropía sobre prefijos nuevos del estudiante, pérdida de preferencia/reparación y replay.
Esos pesos siguen siendo experimentales y no forman parte de los valores por defecto de los artefactos de la v1.

## 11. Cortafuegos de evaluación y registro de benchmarks (hito posterior)

Como mínimo, cada ejecución congela una pequeña sonda de capacidad privada antes de la generación. Sus prompts
y respuestas en bruto nunca llegan al docente, al generador de tareas, al asignador ni a los datos de entrenamiento;
sólo las bandas agregadas de debilidad pueden cruzar esa frontera.

La ejecución de benchmarks públicos es seleccionable porque las suites agénticas completas pueden dominar el runtime.
El registro planificado expone muestras `quick`, `standard`, `full` y `custom` deterministas.
El muestreo es estratificado y se selecciona mediante un hash estable de `(benchmark version, seed, item id)`;
se almacenan los IDs de ítem ordenados exactos. Las comparaciones previas/posteriores requieren la misma versión de benchmark,
los mismos ítems, harness, scaffold, herramientas, presupuesto de tokens/tiempo y scorer.

Las familias candidatas del catálogo incluyen:

- pequeños/generales: ARC, HellaSwag, PIQA, WinoGrande, GSM8K, MBPP y HumanEval;
- generales/razonamiento: MMLU-Pro, GPQA Diamond, MATH-500/AIME y LiveCodeBench;
- ingeniería de software: SWE-bench Verified/Pro y evaluaciones al estilo DeepSWE;
- agénticas: Terminal-Bench (fijar v4.0.0 cuando se seleccione), Agents' Last Exam y otras
  suites de tareas reproducibles con harness.

Los nombres anteriores son candidatos de la hoja de ruta, no claves de schema de la v1. Cada integración necesita su propia
revisión de licencia, disponibilidad, harness, versión, contaminación y costo. Las puntuaciones agénticas
miden el modelo más el harness/scaffold fijado, no el modelo de forma aislada.

Un usuario puede omitir un pre-baseline público costoso, pero entonces Soup puede informar sólo una puntuación
posterior absoluta, no una ganancia antes/después. La comparación científica requiere al menos:

1. estudiante sin tocar;
2. SFT verificado fijo;
3. SFT verificado adaptativo;
4. KD dispersa adaptativa;

con los mismos tokens de entrenamiento del estudiante. Informa los deltas absolutos y el cierre de la brecha con el docente, los pisos de
confianza/ruido, tres semillas cuando sea práctico, la memoria máxima, el tiempo de captura/entrenamiento, la energía, los bytes/token
y el error de reconstrucción/KL densa-vs-dispersa. La reducción de la pérdida por sí sola no es un criterio de aceptación.

## 12. Modelo de amenazas y fallos

| Amenaza/fallo | Detección | Respuesta requerida |
|---|---|---|
| Publicación interrumpida o parcial | Estado/payload faltante, falta de coincidencia de recuento/hash | Ignorar como reutilizable; reanudar una transacción coincidente o poner en cuarentena |
| Manipulación/bit rot de un payload parseable | SHA-256 sobre los bytes exactos | Poner en cuarentena; nunca reparar ni sobrescribir en el lugar |
| Mezcla de ejecución/modelo/tokenizador/datos | Falta de coincidencia del plan y del fingerprint inmutable | Rechazar la reanudación/el consumo y crear una nueva ejecución |
| Revisión de hub en movimiento | La revisión no es inmutable o los archivos ya no coinciden | Rechazar la captura/reutilización |
| Colisión de ID de token entre tokenizadores | El fingerprint del tokenizador difiere | Fallar de forma clara; sin coincidencia silenciosa de IDs ni alternativa ULD |
| Prefijos on-policy obsoletos del estudiante | El rollout del estudiante ya se confirmó o cambió de ronda | Prohibir el replay; recapturar desde el nuevo estudiante |
| Crash después de reservar, antes del checkpoint | Sin hash de checkpoint confirmado | Devolver la reserva a available |
| Crash después del checkpoint, antes del commit del libro mayor | El checkpoint existe pero el evento está ausente | Reconciliar el hash exacto y luego agregar un commit; nunca reentrenar a ciegas |
| Fuga de la evaluación privada | La procedencia/linaje se cruza con IDs de evaluación congelados | Poner los datos en cuarentena e invalidar la ejecución |
| Contaminación de benchmark | Comprobaciones de linaje de n-gramas/exacto contra corpus públicos fijados | Eliminar o poner en cuarentena e informar los estratos afectados |
| Manipulación del verificador/reward | Comprobaciones deterministas, controles adversariales, tasas de desacuerdo | Poner en cuarentena el desacuerdo; detenerse tras fallos repetidos |
| Autoconfirmación del docente | El docente es el único generador y juez | Exigir evidencia determinista o independiente cuando sea posible |
| Deriva de logits cuantizados | Fidelidad diminuta BF16/cuantizada y piloto A/B posterior | Recurrir a SFT/preferencias sólo de salida |
| Presupuesto desbocado | Topes duros obligatorios y estimación previa | Detenerse antes de exceder el primer presupuesto agotado |
| Path traversal/intercambio de symlinks | Rutas relativas normalizadas, `realpath` + `commonpath`, lecturas sin seguir enlaces | Rechazar el acceso y poner la transacción en cuarentena |
| Probabilidades no finitas/inválidas | Schema estricto, comprobaciones de masa/entropía | Rechazar la fila antes de completar el shard |
| Registros duplicados/reordenados | IDs/posiciones contiguos y recuentos confirmados | Fallar de forma cerrada; reanudar sólo el prefijo verificado |

El sistema no se defiende contra un kernel malicioso o un host de modelo comprometido. Los pesos locales del modelo,
el código generado y la ejecución del verificador siguen siendo entradas no confiables y requieren el mismo
sandbox y los mismos topes de ruta/tamaño que las superficies existentes de ejecución de código y datos de Soup.

## 13. Fronteras de los hitos

- **A (este corte):** schemas de la v1, fingerprints, contratos de probabilidad/replay, estimación
  determinista, máquinas de estado, semántica de integridad/reanudación, fixtures y modelo de amenazas.
- **B:** captura exclusiva del docente con el mismo tokenizador más CE+KL dispersa exclusiva del estudiante, separación
  de memoria entre procesos, equivalencia de ejecución interrumpida y una ruta CUDA más una de Apple Silicon.
- **C:** generación adaptativa, verificación, mapa de capacidades, asignador acotado, replay y paradas.
- **D:** ULD explícita de spans visibles entre tokenizadores con ataques de Unicode/control/plantilla de chat.
- **E:** benchmark reproducible con los mismos tokens e informe de ablación.

El Hito B debe primero demostrar la equivalencia densa-vs-offline sobre un fixture pequeño, determinista y de vocabulario completo
y el comportamiento exacto de reanudación. Sólo después de eso debería intentarse un smoke científico real entre
docente y estudiante.

## 14. Fixtures deterministas

`tests/fixtures/autodistill/v1/` contiene:

- un plan totalmente explícito sin perfil de runtime;
- filas de captura del docente experto y del estudiante con rollout sobre un vocabulario de tres tokens, donde la masa
  seleccionada es 0.9 y la cola residual es 0.1;
- un shard JSONL contado por bytes y comprometido con SHA-256, reconstruido a partir de datos de captura canónicos de modo que
  la conversión de saltos de línea del checkout no pueda alterar sus bytes comprometidos;
- un libro mayor de reserva/commit del rollout del estudiante.

Las pruebas cubren modelos frozen/strict, campos de política faltantes, estimaciones manipuladas, ausencia de importaciones pesadas,
canonicalización del dataset, requisitos del rollout del estudiante, unión top-k/forzada y masa de cola residual,
equivalencia densa de `k = vocab`, path traversal, corrupción, independencia de los saltos de línea del checkout, saltos de
estado, consumo contiguo exactamente una vez, replay encadenado del experto y cada decisión de reanudación.
