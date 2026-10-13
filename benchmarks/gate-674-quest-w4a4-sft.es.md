# SFT QuEST W4A4: control fallido (#674)
> 🇪🇸 Traducción al español de [`gate-674-quest-w4a4-sft.md`](gate-674-quest-w4a4-sft.md). Ante cualquier duda, manda el original en inglés.

**Control fallido: 0.114 nat [0.099, 0.130], una semilla de entrenamiento, un modelo, cuantización
simulada. No es paridad, no es una integración ni una afirmación de eficiencia.**

Medido por [@Shutaru](https://github.com/Shutaru). El experimento principal terminó
el 2026-09-09; el último diagnóstico acotado terminó el 2026-09-11. Este registro
amplía la [actualización de investigación](https://github.com/MakazhanAlpamys/Soup/issues/674#issuecomment-5636030904)
con el protocolo y la procedencia solicitados en la
[respuesta del mantenedor](https://github.com/MakazhanAlpamys/Soup/issues/674#issuecomment-5636239673).
Deja abierto [#674](https://github.com/MakazhanAlpamys/Soup/issues/674).

## Resultado en el mismo panel reservado

Las once filas usan los mismos 192 ejemplos Dolly y 7,819 targets de respuesta supervisados
(CHECK192). Un NLL (log-verosimilitud negativa) de respuesta-target menor es mejor.
`RDxx` identifica experimentos locales retenidos, no versiones de Soup.

| Modelo / endpoint retenido | NLL CHECK, nat por target |
| --- | ---: |
| Base original | 2.619283 |
| FP02, padre común | 2.246091 |
| FP de datos antiguos RD13 | 2.247933 |
| FP de datos nuevos RD13, FP retenido más sólido | 2.235367 |
| Control FP RD15 | 2.236810 |
| W4A4 calibrado/destilado anterior (Recipe06) | 2.377740 |
| Inicialización W4A4 de ancho completo, sin group128 | 2.520581 |
| Inicialización W4A4 group128 | 2.514601 |
| Control terminal W4A4 de ancho completo, sin group128 | 2.357655 |
| Candidato terminal W4A4 group128 | 2.349768 |
| Control FP concurrente | 2.236143 |

El candidato group128 predeclarado tiene NLL **2.349768174976808**, frente a
**2.235367338223475** del FP retenido más sólido en estas mismas filas. La brecha
es **0.11440083675333268** nat, con intervalo pareado del 95%
**[0.0991910814834233, 0.12992046340524593]**.

Nuestro control experimental de calidad exigía que tanto la brecha como su límite superior del 95%
fueran como máximo 0.1 nat. Ambos fallan. El valor 0.1 era nuestro criterio experimental
prerregistrado, no un requisito numérico impuesto por quien abrió el issue.

Group128 mejoró frente a Recipe06 en 0.027972 nat; el intervalo candidato menos control
con signo es [-0.042689, -0.013470]. Frente al control terminal sin group128,
la diferencia es -0.007887 nat, con intervalo [-0.023544, 0.007761]. Ese intervalo
cruza cero, así que también falla el requisito separado de demostrar una mejora frente a
ese control. No se promovió ningún brazo ni inicialización alternativos.

Los intervalos usan 2,000 remuestreos bootstrap pareados de secuencias, semilla 17. Cada remuestreo
vuelve a calcular la diferencia de pérdida ponderada por tokens; no se muestrean tokens de una
secuencia de manera independiente. Son condicionales a la muestra observada y a una
semilla de entrenamiento, no estimaciones de la variación entre semillas de entrenamiento. El
comparador FP es el NLL CHECK más bajo entre los endpoints FP retenidos predeclarados; el
intervalo informado corresponde a ese par seleccionado, no es una afirmación de confianza
ajustada por selección. El cómputo histórico de entrenamiento entre esos endpoints no es igual.

## Ficha del instrumento

| Elemento | Configuración / procedencia registrada |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, VRAM informada de 24,463 MiB (clase de 24 GB); una GPU local para RD21 |
| Controlador NVIDIA | **No capturado en los registros históricos RD21.** Una consulta al momento de documentar, el 2026-09-11, devolvió `616.64`; esto no demuestra cuál era el controlador usado el 2026-09-09 |
| SO / Python | Windows; Python `3.12.10`, según el registro del entorno de viabilidad original |
| PyTorch / CUDA | `2.11.0+cu128` / `12.8`, también registrados en cada ajuste RD21; aquí CUDA es la versión de runtime/compilación de PyTorch, no el máximo anunciado por el controlador |
| Transformers | `5.16.1` en el entorno de viabilidad original |
| Modelo y tokenizador | `ahxt/LiteLlama-460M-1T`, revisión `77b8a976440e7d1ea5a890eaf1e0175b1cac0078` |
| Dataset | `databricks/databricks-dolly-15k`, revisión `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, archivo `databricks-dolly-15k.jsonl` |
| Checkout de investigación | Base Soup `64d029e`; los scripts experimentales locales se identifican abajo, no es un backend publicado |

La ausencia del controlador histórico es una limitación de procedencia. No se completó
con los datos de la instalación de hoy. Los datos de este registro no establecen el comportamiento
en otra GPU, pila de software o modelo.

RD21 registró algoritmos deterministas activados con errores en vez de advertencias,
determinismo de cuDNN activado, benchmark de cuDNN desactivado, TF32 desactivado, reducción
con precisión reducida BF16/FP16 desactivada y un hilo de CPU. Los campos
`CUBLAS_WORKSPACE_CONFIG` y `PYTHONHASHSEED` estaban sin configurar. Son ajustes registrados,
no una garantía de igualdad bit a bit entre pilas.

## Protocolo fijo

El protocolo científico se registró como RD18 antes de la ejecución; RD21 lo completó
después de una recuperación operativa, sin cambiar la receta ni las reglas de decisión.

### Formato y entrenamiento

Esta es una **variante experimental de QuEST**. Ambos operandos reciben una transformación
Hadamard de ancho de entrada completo (ancho 1,024 o 4,096), y luego grupos contiguos de 128
para escalado RMS y cuantización de 4 bits. Las escalas de clipping de activaciones provienen
de la tabla Recipe06 calibrada con TRAIN, que se reutiliza sin recalibración. El
control de fila completa usa la misma transformación y aritmética de cuantización sin group128.

Las 168 capas lineales del bloque transformer permanecen W4A4. Las incrustaciones, la normalización,
las operaciones de atención y la cabeza de salida mantienen sus exenciones declaradas. Los
parámetros maestros son FP32, con ejecución de matrices BF16 y cuantización simulada.
Los intermedios de cuantización conservan el dtype rotado de entrada bajo autocast;
no es un cálculo íntegramente en una cuadrícula FP32. No es un resultado QuEST upstream sin
modificar, almacenamiento INT4 empaquetado, un kernel de inferencia INT4 ni una integración de
`soup train`. No hay residuo adicional de precisión completa ni ensemble de inferencia.

Los brazos W4A4 nativo, W4A4 group128 y FP concurrente parten cada uno de una copia
independiente del mismo checkpoint FP02 retenido. Una copia FP02 independiente y congelada
es el teacher. Cada brazo usa un optimizador AdamW nuevo y:

- 1,024 ejemplos de entrenamiento fijos, tamaño de lote 2, cuatro pasadas en el orden guardado
  sin barajar: exactamente 2,048 actualizaciones y 169,460 targets de respuesta por brazo.
- Tasa de aprendizaje máxima `5e-6`, decaimiento cosenoidal, 205 actualizaciones de calentamiento,
  beta2 de AdamW `0.95` y weight decay cero.
- Objetivo `0.25 * response CE + 0.75 * KL(teacher || student)`, temperatura 1,
  normalizado en las mismas posiciones de respuesta supervisadas para cada brazo.
- Semilla 42 para Python, NumPy y Torch. Sin evaluación CHECK intermedia, parada temprana
  basada en calidad ni extensión posterior al resultado.

Cada student procesó 343,776 tokens de entrada sin padding durante todo el ajuste; el
teacher procesó la misma cantidad. Con el padding de lotes, fueron 444,216 tokens de entrada
por modelo, aparte de los 169,460 targets de respuesta supervisados. Son ejecuciones SFT breves,
no presupuestos de tokens a escala de preentrenamiento.

### Tokenización y medición

El tokenizador GPT-2 fijado usa explícitamente `<|endoftext|>` (ID 50256) para BOS, EOS
y padding, para resolver metadatos especiales contradictorios en el repositorio del modelo.
Los prompts contienen `### Instruction:`, `### Context:` opcional y `### Response:`.
Se excluyen filas con instrucción/respuesta vacía o un prompt de más de 192 tokens.
Las respuestas se codifican con un espacio inicial, se limitan a 64 tokens y reciben EOS solo
si son más cortas que ese límite. La longitud total es como máximo 256.
Las etiquetas del prompt y del padding son `-100`; solo los targets de respuesta contribuyen a la pérdida.

El NLL principal es la pérdida del modelo con tamaño de lote dos, ponderada por el número de targets
supervisados. Los valores separados de entropía cruzada FP32 por target se suman en FP64 por
secuencia para estimar la incertidumbre bootstrap. La concordancia con la métrica principal se comprueba
dentro de `5e-6` nat; esta comprobación numérica no flexibiliza el control de calidad de 0.1.
No deben usarse las cifras redondeadas de la tabla para reconstruir la brecha de precisión completa.

Después de que todos los ajustes y reproducciones de checkpoints aprobaran, los once endpoints
fijos de arriba se evaluaron una vez en CHECK192. El FP retenido más sólido debía mejorar frente a
la base, al igual que el FP concurrente. Group128 también debía mejorar frente a Recipe06
y frente al control sin group128 en al menos 0.005 nat, con el límite superior de cada intervalo
pareado por debajo de cero, además del control de brecha frente a FP. Las comprobaciones de ingeniería
aprobaron; falló la conjunción completa de aceptación.

### Checkpoints

Las tres pruebas rápidas de dos actualizaciones y sus reproducciones en procesos nuevos aprobaron
antes de los ajustes completos. Cada ajuste completo guardó los pesos previos a la prueba, el optimizador
y los metadatos de formato. Las comprobaciones de recarga en procesos nuevos coincidieron con los logits
nativos, seguidas de una actualización positiva controlada con tasa de aprendizaje `5e-7` que cambió
los parámetros y reprodujo el objetivo, el digest de parámetros y el estado del optimizador. El teacher
permaneció sin cambios. Las copias modificadas por la prueba no se usaron en la evaluación CHECK.

Estas comprobaciones cubren las entradas registradas y la reconstrucción explícita del wrapper
experimental. No establecen un paquete público de tokenizador/generación ni un contrato completo
de API de guardado/carga de Soup.

## Historial de consumo de paneles

Los paneles se identifican por sus manifiestos, no solo por su tamaño. Varios estudios distintos
usaron 192 ejemplos; no son intercambiables.

| Panel | Exposición y estado actual |
| --- | --- |
| Validación piloto original (32) y prueba de cuatro épocas (64) | Evaluados en las primeras recetas fallidas; consumidos, excluidos de paneles nuevos posteriores |
| TRAIN1024 / DEV128 de I+D originales | TRAIN se usó para ajuste/calibración y DEV repetidamente para desarrollo; ninguno es una prueba nueva |
| RD08 FIT64 / ID128 / SHIFT64 | Usados en el estudio de rotación del objetivo de respuesta; sus paneles de evaluación están consumidos |
| RD11 CHECK192, 7,399 targets | Evaluado para calibración global; consumido |
| RD13 CHECK192, 6,612 targets | Evaluado para el estudio de diversidad; consumido |
| RD15 FIT1024 / CHECK192, 7,323 targets CHECK | FIT se reutilizó para RD18/RD21 y diagnósticos TRAIN posteriores; CHECK se evaluó y está consumido |
| RD18/RD21 CHECK192, 7,819 targets | Nuevo para el experimento RD21 congelado; ahora consumido, incluso tras fallar el candidato |
| Ventanas TRAIN posteriores, incluidas RD54/RD55 | Datos de ajuste previamente expuestos; solo evidencia diagnóstica, no resultados nuevos de generalización |
| FINAL256 reservado | **Nunca evaluado.** Sigue reservado para una única confirmación final; el candidato fallido no llegó a él |

La partición original de I+D usó índices de filas elegibles con base cero `[32:160]` más
`[224:1120]` para TRAIN, `[1120:1248]` para DEV y `[1248:1504]` para FINAL256.
Las filas elegibles se ordenaron por hash de fila normalizado, después de deduplicar por instrucción
normalizada y grupo de documentos, y aplicar los filtros de codificación fijos.

RD21 reutilizó el FIT1024 exacto de RD15 (42,365 targets por pasada). Su nuevo CHECK tomó
las primeras 192 filas elegibles del prefijo fijo de 8,192 filas candidatas, tras excluir cada fila,
instrucción normalizada e identidad de grupo de documentos de los manifiestos original, RD08, RD11,
RD13 y RD15, incluidas sus particiones de ajuste y final. Esta es una exclusión por identidad, no una
afirmación de descontaminación semántica o de preentrenamiento. DEV original y FINAL256 no tenían
ruta de evaluación RD21. No se volvió a seleccionar ningún panel según su pérdida.

## Direcciones fallidas conservadas

En estos estudios se usaron filas, formatos y presupuestos distintos. Sus NLL absolutos no deben
combinarse en una tabla de posiciones ni restarse de RD21.

El piloto original de una época con 128 ejemplos falló en los tres pares de semillas. Repetir
los mismos 128 ejemplos durante cuatro épocas también falló, y su control FP quedó peor que la base
en la prueba separada de ese estudio. Más datos, clipping, destilación y agrupamiento llevaron al
candidato RD21, pero no a la paridad.

Las evaluaciones locales de reconstrucción probaron rotaciones Givens pareadas, escalas ponderadas
por gradiente de respuesta, tablas de clipping teacher/student y redondeo correlacionado. Algunas
redujeron el error de reconstrucción local sin la mejora requerida del NLL de respuesta. Una
rotación redujo el error local cerca del 66% mientras su estimación puntual de NLL de respuesta
aumentó 0.0035 nat. Cuatro evaluaciones representativas recibieron reproducciones nuevas;
esto no afirma que se reprodujeran las dieciséis evaluaciones. Un estudio posterior de rotación
con objetivo de respuesta de una sola capa y 64 pasos no estableció una ganancia en su panel de
evaluación separado; no descarta en general los métodos de rotación.

Las intervenciones de dirección del gradiente no aprobaron sus comprobaciones fijas de pasos finitos.
Una reproducción posterior agotó el tiempo durante la restauración final. Se conservaron dieciocho
registros de pruebas parciales coincidentes; no se contaron como una reproducción completada.

La calibración de salida RD54 obtuvo resultados exactos iniciales/de reproducción en seis controles.
En las filas TRAIN 32:64 (1,218 targets), Q mejoró 0.007951 nat, pero FP mejoró 0.023545,
ampliando la brecha justa de 0.251284 a 0.266877. Ambas ventanas TRAIN fallaron el filtro de
continuación. Se aisló una discrepancia de entropía cruzada FP32 en CPU detectada durante este trabajo
y se comprobó la puntuación corregida frente a la pérdida nueva de GPU. Fue una corrección de medición,
no una mejora del modelo; la ejecución fallida anterior no se reclasificó como exitosa.

RD55 probó la conversión directa W4A4 del checkpoint FP concurrente posterior, sin entrenamiento
de recuperación. La primera ejecución y la reproducción nueva coincidieron. Aun así, la nueva
inicialización quedó por debajo del mejor endpoint Q entrenado retenido en ambas ventanas TRAIN,
y falló el filtro de inversión predeclarado de al menos 0.01 nat de mejora en ambas.
No hubo una ejecución de recuperación posterior. Sin embargo, una comparación retrospectiva con las
mismas filas en 32 ejemplos TRAIN mostró que mejoró 0.403683 nat frente a la **inicialización antigua
sin entrenar**. Fallar la comparación conservadora con el endpoint entrenado no demuestra que el
nuevo punto de partida carezca de valor ni que la recuperación fallaría.

## Costos y límites

Ejecuciones nuevas emparejadas de 2,048 actualizaciones, incluido el trabajo de ajuste/guardado/prueba:

| Brazo | Segundos transcurridos | Bytes máximos asignados en GPU | GiB (bytes / 2^30) |
| --- | ---: | ---: | ---: |
| FP concurrente | 495.062 | 10089582592 | 9.397 |
| W4A4 de ancho completo sin group128 | 1077.563 | 13320890880 | 12.406 |
| W4A4 group128 | 1150.453 | 14721952768 | 13.711 |

La implementación QAT de referencia es más lenta y usa más memoria. Son tiempos descriptivos de
escritorio, no benchmarks controlados de kernels INT4 ni de despliegue. Los máximos de memoria
asignada por PyTorch no son el uso total del dispositivo y excluyen las asignaciones del controlador.
El entrenamiento histórico del padre y los controles retenidos no se incluye en esta tabla de costos
emparejados de ejecuciones nuevas.

No se ha establecido utilidad a nivel de aplicación, comportamiento entre modelos, confirmación
real con tres semillas ni un límite irreducible de cuantización. Cambiar solo la semilla externa sería
insuficiente: el loader retenido restablece Torch a 42 y reutiliza pesos fijos del padre y el orden fijo
de lotes. Un estudio futuro de semillas debe definir variación real de entrenamiento pareado antes de
afirmar resultados con semillas 42/7/123.

## Procedencia y límites de reproducción

Los scripts, protocolos y registros completos permanecen locales. Esta contribución publica
las descripciones y los vínculos SHA-256 solicitados, no una integración ejecutable. No contiene
filas de corpus sin procesar, ID de tokens, checkpoints ni credenciales. Los hashes identifican
bytes retenidos; no hacen reproducibles de forma independiente los archivos no disponibles ni
prueban cuándo se registró un protocolo. CI de este cambio de documentación no vuelve a ejecutar
los experimentos de GPU.

Antes de publicar, se volvieron a comprobar 55 vínculos de fuentes/evidencias/manifiestos retenidos
de la auditoría RD21, incluido el código fuente instalado de recorte de gradiente de Torch. La
evaluación guardada de once modelos conserva los conteos de targets por fila y las sumas de NLL.
La auditoría vincula esos resultados con las fuentes y los registros de ajuste/reproducción. Los
vínculos principales de archivos aparecen abajo; las rutas son relativas a la colección de investigación
local, no enlaces a archivos incluidos en este PR.

<details>
<summary>Vínculos de archivos SHA-256</summary>

| Archivo | Propósito | SHA-256 |
| --- | --- | --- |
| `RD18_GROUP_RETRAIN_PROTOCOL.md` | Ciencia, paneles, presupuesto y controles fijos | `1e5fb278328a15e38576a66fa9a8c4c80b3af8e21ea189b5f0dca15ee21faa9c` |
| `RD21_SCHEDULED_RECOVERY_PROTOCOL.md` | Recuperación operativa sin cambio de receta | `d1660457329d6e4ab3268e96e69bedd408ff45822d6dffcf0cdae55f7c9d8ee2` |
| `quest_sft_feasibility.py` | Recursos fijados, tokenizador y codificación solo de respuestas | `5a623830c8c9e5213559cc37f986dcee2aa8bcfc6f4e59480888bbd92a1d5d56` |
| `quest_rd_group_data.py` | Construcción de paneles y comprobaciones de exclusión | `511f67b7c95139d0038dda82d294a28efb6e54958c7dc8f3c831350114c48ccd` |
| `quest_rd_group_train_ops.py` | Operadores de ancho completo/group128 y cobertura | `e2e87352b69c72b0279f6cf4282539d39ce9af4b695a97fdb921425b02788b29` |
| `quest_rd_group_run.py` | Entradas de ajuste, evaluación y reproducción fijas | `724287317edf5b675ec7b0a7119cf9eef5771ab3452523afc6d7d33110a327d5` |
| `quest_rd_group_scheduled.py` | Wrapper de ejecución RD21 | `b5b9c63f6aa9636da78a1a69122fac62e3864d47214cbf59c19ee7a7f410a898` |
| `quest_rd_measurement.py` | NLL principal y descomposición por secuencia | `ebe9d279b9b08843f1c763d0c997eaf7778ca66d26e197a6928b2dd7a55748f1` |
| `quest_rd_statistics.py` | Bootstrap pareado | `5b7928cfba54d1bd3d481fa2484d39323e9a75d1e4d9e9a4ebc533019b064655` |
| `quest_rd_group_audit.py` | Comprobaciones independientes de artefactos y decisiones | `632199693d10ba4d1b51f25bb8920139a6457a25363b081c7005a92ad5d15832` |
| `quest_rd_group_scheduled_audit.py` | Entrada de auditoría RD21 | `734f16520c9aee50c4d6476a450711481259a71bff844e49e9252003853cd677` |
| `rd18-panels.json` | Manifiesto FIT/CHECK y vínculos a paneles upstream | `f62d600e9c1fc5dc3c1a1b0290dc2f9e2bf79f3a1419a9060771ae1b2638d9e3` |
| `rd-clipping-calibration.calibration.json` | Clipping de activaciones congelado, derivado de TRAIN | `2670fe8bfece03a06fcf9fd3d80fe72437dcd3f3da8c101a5c9959d332c5670d` |
| `rd21-screen-01.native.json` | Registro de ajuste completo sin group128 | `009a6a91930179749c0c55ef28bd9d10cc34d0e0d9aadf3561feecf37db0dfa2` |
| `rd21-screen-01.native.replay.json` | Reproducción en proceso nuevo | `d00535ff22bdf3a5068c7c89f7eb5ebdd99594df94f45030f6afbf922e2ea0bf` |
| `rd21-screen-01.group128.json` | Registro de ajuste completo del candidato | `f8b79f538df91c940dbb5a33c381c87684b64615bd161232b87e9cfbe87a888f` |
| `rd21-screen-01.group128.replay.json` | Reproducción del candidato en proceso nuevo | `824a3c72823472d3b4d367c5afd0b56666953382055d6d70d494ccf698ff477f` |
| `rd21-screen-01.fp.json` | Registro de ajuste completo FP concurrente | `9c9de60c4dc418ce3914d5dd358397eb94577b2ff73ff19789f1c12df65416f2` |
| `rd21-screen-01.fp.replay.json` | Reproducción FP concurrente en proceso nuevo | `58564974aa417f1fb8e92e1bb3eb340b8d7c98fb1b26255a033f15c738aad0ab` |
| `rd21-screen-01.evaluation.json` | Once mediciones en el mismo panel | `2c6e73b1e0302739d5dea74a39a5044e4aff7c83e586a89a5b9df53b9106a407` |
| `rd21-audit-01.json` | Auditoría completada y decisión de aceptación fallida | `46345c958fe5d6e2320f5e5170bdee8342d445419e849a92930f55ce618a6e4a` |
| `rd55-initialization-01.first.json` | Último diagnóstico de inicialización acotado | `401255f7e4eb2a2a86974633400f240d6b75d4d7dfd612d50ba9e0d0847f64bf` |
| `rd55-initialization-01.replay.json` | Reproducción científica completa de ese diagnóstico | `2b562784f0a43dfe2c4c74a48034205128bad13014859449eb86a7b40b8b9719` |

</details>

## Estado y posible seguimiento

Las exploraciones más amplias están pausadas. El mantenedor sugirió ablaciones de solo pesos/solo
activaciones y por bloque, clipping dinámico al evaluar, comparaciones de gradiente y pruebas de
recuperación con presupuesto/inicialización FP más sólida. Son posibles experimentos nuevos, no
resultados ni compromisos de este registro. Las ablaciones de recetas anteriores no resuelven esas
preguntas para este candidato. Una curva de presupuesto plana no demostraría por sí sola un límite
inherente del método, ni un coseno de gradiente alto descartaría todos los problemas de gradiente sustituto.

Todo seguimiento necesita un panel de desarrollo pequeño, predeclarado por separado, y una regla
de decisión acotada. El CHECK192 consumido no puede validar una receta modificada.
FINAL256 sigue reservado para una confirmación bloqueada con semillas genuinas 42/7/123.
El registro puede mantenerse como resultado negativo sin otra ejecución de entrenamiento.
