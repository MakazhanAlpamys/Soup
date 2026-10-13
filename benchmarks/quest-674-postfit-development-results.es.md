# Incidente 674: mejor checkpoint de desarrollo retenido tras el ajuste
> 🇪🇸 Traducción al español de [`quest-674-postfit-development-results.md`](quest-674-postfit-development-results.md). Ante cualquier duda, manda el original en inglés.

4 de octubre de 2026. El mejor checkpoint integrado QuEST retenido tiene una
proporción de response-NLL de **1.0383673005** frente al checkpoint FP retenido más sólido en DEV704.
Sigue por encima del objetivo de paridad con margen cero. Este registro conserva la selección
del mejor checkpoint, la receta de su última etapa, la evidencia numérica y nueve seguimientos
negativos o no concluyentes. No autoriza la confirmación final.

## Mejor resultado retenido

Ambos extremos usan las mismas 704 secuencias y 25,017 objetivos de respuesta de
DEV192B + CONFIRM512. Este panel ya se utilizó para seleccionar candidatos.

| Extremo o contraste | Resultado |
| --- | ---: |
| QuEST: `readout-adaptation-v1/runs/fit-quest-full` | NLL 2.125037937785311 |
| FP: `prompt-addition-v1/runs/fit-fp` | NLL 2.046518545766202 |
| QuEST / FP | 1.0383673004974954 |
| QuEST menos FP | +0.07851939201910918 nat/target |
| Intervalo de diferencia pareada al 95% | [+0.07149328368143801, +0.08602618686052922] |

El intervalo usa 2,000 remuestreos pareados de secuencias con semilla RNG de NumPy 674
y proporciones ponderadas por objetivo de las sumas de loss. Es descriptivo después de la
selección adaptativa y está condicionado a estos checkpoints seleccionados. La regla de margen
cero exige ratio <= 1.000 y límite superior de la diferencia pareada al 95% <= 0; aquí no se
cumple ninguna de las dos condiciones.

El [resultado publicado de rutas mixtas sólo para evaluación](gate-674-quest-mixed-route.md)
tenía una diferencia de +0.086344 frente a su control FP fijo. Este resultado post-ajuste
seleccionado tiene una diferencia menor y una loss Q absoluta inferior en el mismo panel
agrupado, pero el control FP y los historiales de entrenamiento difieren. No es una estimación
controlada del efecto del cambio de código de continuación, y el promedio histórico de tres
semillas no debe rebautizarse como esta medición seleccionada de una sola semilla.

## Receta preservada de la última etapa

El ganador es la variante de **continuación de todos los parámetros** del experimento de
readout, no la variante que sólo ajusta normas/cabeza. Conserva 219 tensores de parámetros
maestros FP32 (461,685,760 parámetros) y usa ejecución BF16. La ruta integrada mantiene
168 lineales fake-W4, 161 activaciones A4 y siete activaciones A16 en el bloque 23; el grupo
128 y la Hadamard normalizada de ancho completo no cambian. La tabla de clipping heredada se
restaura desde los metadatos QuEST del padre.

- Padre del estudiante: `feature-continuation-v1/runs/fit-quest-control`, digest del parámetro
  maestro `900ef3147a35e17738e0e4cb4b2a3b82672c82375916c7d8cba4b32ebb5bd6a0`.
- Teacher congelado: `teacher-followup-v1/runs/fit-fp`, digest
  `83b4bfedd4644f05e54f3e0982eeba43c8d1bde9dc8cdf29aff413c7f7e97968`.
- Objetivo: 0.25 de entropía cruzada de respuesta + 0.75 de KL forward de respuesta, temperatura 1;
  peso de loss de características 0. Este es un adaptador de investigación alrededor de SFT nativo.
  La CLI pública de Soup no expone esta receta combinada QuEST/teacher.
- Una epoch del conjunto TRAIN retenido de 2,037 filas: 1,019 actualizaciones, batch 2 con un
  singleton final, 92,357 objetivos de respuesta; hash fuente de las filas de entrenamiento
  `7d5f832f5b2eac4ad7a45b7819eea26aaf6328db658839062d018f18c1c22e0c`.
- Optimizador AdamW nuevo, beta1 = .9, beta2 = .95, epsilon = 1e-8, weight decay 0,
  límite de norma del gradiente 1; LR máxima 5e-6, horizonte cosine 1,019, warmup de 102 actualizaciones.
- Semilla del modelo 42, semilla de datos 674, acumulación de gradiente 1, gradient checkpointing desactivado.

El ganador Q y el FP más sólido tienen historiales de padres distintos. Los recibos completos
de su última etapa registran cada uno 1,019 forwards del estudiante, forwards del teacher,
backwards y actualizaciones del optimizador, 2,037 ejemplos y 92,357 objetivos de respuesta.
La evidencia portable registra los hashes de protocolos de linaje; para reproducirlo se necesitan
los padres retenidos por separado, el conjunto TRAIN codificado y los workers de investigación.
Empezar desde el modelo original de Hub y usar sólo estos ajustes de última etapa no reproduce
este checkpoint.

## Seguimientos que no justificaron reemplazar al ganador

Estos son resultados de las pruebas especificadas, no exclusiones universales de los métodos.
Los diagnósticos TRAIN no constituyen evidencia de paridad Q/FP en datos reservados.

| Seguimiento | Alcance y observación | Decisión |
| --- | --- | --- |
| Continuación Dolly/OASST1/OASST2 | Ajustes emparejados completados; ratio Q/FP más sólido en DEV704: 1.040610 | Peor que el 1.038367 retenido |
| Continuación UltraChat | 1,024 pares de entrenamiento separados y 512 actualizaciones por variante; ocho puntuaciones de desarrollo. Ratio DEV704 1.041713; ratio UltraChat 1.047574 | Cambio de diferencia de sólo -0.002257 nat, IC [-0.008683, +0.004379]; no superó el gate de 0.03 nat y excedió la tolerancia de regresión de Dolly |
| Objetivo de teacher más sólido | Evaluación pequeña en TRAIN ajustado; tratamiento menos control +0.001086 nat, IC [-0.012057, +0.014800] | Sin mejora respaldada; gate fallido |
| Destilación intermedia de características | Evaluación pequeña en TRAIN ajustado; tratamiento menos control -0.009348 nat, IC [-0.023029, +0.003732] | Mejora puntual incierta; gate fallido |
| Aritmética Hadamard FP32 | Diagnóstico de 16 filas en TRAIN ajustado; tratamiento menos control +0.013716 nat, IC [-0.008544, +0.036046] | Sin mejora respaldada; gate fallido |
| Temperatura escalar de logits | La loss Q TRAIN mejoró; cambio en la diferencia Q/FP -0.002887 nat, IC [-0.008448, +0.002355] | Gate de reducción de diferencia fallido |
| Tamaño de grupo 64 | 16 filas de TRAIN ajustado; grupo 64 menos nativo 128 +0.138453 nat, IC [+0.088862, +0.189135] | Peor; no promovido |
| Hadamard de bloque 128 en Q entrenado | 64 filas de TRAIN ajustado; bloque menos nativo +0.185829 nat, IC [+0.151520, +0.218562] | Peor; restauración nativa idéntica bit a bit |
| Permutación común de grupos W/A | 16 filas de TRAIN ajustado; entre grupos menos nativo +0.161242 nat; el control de simetría de grupos completos excedió las tolerancias congeladas | Prueba del mecanismo no concluyente; el control fallido impide afirmar un reagrupamiento limpio |

El Q y el FP entrenados con UltraChat completaron cada uno 512 actualizaciones, 1,024 ejemplos
y 63,579 objetivos de respuesta. Los hashes del orden de ejemplos, el tamaño de batch y las
máscaras de respuesta coinciden. Ambos mejoraron la loss absoluta en UltraChat, mientras que
el ratio Q/FP cambió de 1.047491 a 1.047574. El ratio de desarrollo OASST2 fue 1.043632 frente
al nuevo FP. Las ocho acciones de puntuación usaron procesos de modelo nuevos y cero
actualizaciones de entrenamiento.

## Evidencia y verificación

La [evidencia portable](evidence/quest-674-postfit-v1.json) incluye sumas de loss e identidades
por secuencia pareadas, hashes de archivos de checkpoint, vinculaciones de receta/protocolo y
hashes fuente de los registros negativos. Excluye datos tokenizados sin procesar, pesos del
modelo y rutas específicas de la máquina. Los checkpoints grandes y los workers originales de
investigación siguen preservados por separado como artefactos locales, no como objetos Git.

Verifica la estadística informada sin cargar un modelo ni acceder a un panel final:

```bash
python benchmarks/harness/quest_parity_evidence.py
```

Opcionalmente, verifica los 27 archivos de los artefactos retenidos Q/FP en el árbol de artefactos
preservado. Esto lee archivos de checkpoint para calcular hashes, pero no hace inferencia:

```bash
python benchmarks/harness/quest_parity_evidence.py --artifact-root .quest-validation/parity-plan-v1
```

El verificador rechaza emparejamientos modificados, losses no válidas, cambios en la ponderación
de objetivos, resúmenes alterados, afirmaciones de confirmación no respaldadas y rutas de
artefactos fuera de la raíz declarada. Comprobaciones específicas de continuación/integración/
evidencia: 122 aprobadas con dos advertencias de obsolescencia de Click. Las comprobaciones
Ruff del repositorio se aprobaron.

`training_quality_validated` sigue siendo false. La confirmación con tres semillas sigue pendiente.
FINAL256 y OASST256 continúan sin abrirse y reservados para una única confirmación separada sólo
después de que haya un candidato elegible. Estas mediciones no establecen generalidad entre
modelos, W4A4 puro, paridad numérica con QuEST upstream, INT4 empaquetado ni eficiencia. No se
compró cómputo ni se cambiaron credenciales/permisos.
