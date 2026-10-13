# Ruta mixta QuEST W4/A4+A16: se superó la evaluación (#674)
> 🇪🇸 Traducción al español de [`gate-674-quest-mixed-route.md`](gate-674-quest-mixed-route.md). Ante cualquier duda, manda el original en inglés.

**El rescate de siete A16, predeclarado, superó el control de calidad solo de evaluación:
0.086344 nat/target [0.080241, 0.093190] por encima de FP fijo en 704 ejemplos y
25,017 targets. Es un solo modelo, con cuantización simulada y sin entrenamiento en esta
confirmación; no es W4A4 estricto, paridad con QuEST upstream, un resultado con INT4
empaquetado ni una afirmación de eficiencia.**

Medido por [@Shutaru](https://github.com/Shutaru). Este registro sigue al
[fallo del control SFT de W4A4 estricto](gate-674-quest-w4a4-sft.md); no revisa ni
reemplaza ese resultado. Los diagnósticos acotados posteriores localizaron suficiente
parte de la brecha de calidad en el bloque 23 como para justificar una confirmación
mixta congelada.

## Jerarquía predeclarada

Las dos rutas se seleccionaron en el panel RD70 DEV192 ya consumido y se congelaron
antes de puntuar cualquiera de ellas en DEV192B o CONFIRM512:

1. **Principal — un A16:** los 168 pesos lineales del transformer permanecen W4;
   `model.layers.23.mlp.down_proj` usa una activación A16 y los otros 167 usan
   A4.
2. **Rescate — siete A16:** los 168 pesos permanecen W4; las siete capas lineales de
   atención y MLP del bloque 23 usan activaciones A16 y las otras 161 usan A4. Este
   brazo solo era elegible si fallaba el principal.

Ambos usan tamaño de grupo 128 y la ruta Hadamard normalizada de ancho completo congelada.
En esta confirmación no se permitió entrenamiento, clipping, calibración, selección de ruta
ni actualización de parámetros aprendidos.

El análisis principal agrupa dos paneles disjuntos:

- DEV192B: 192 ejemplos / 7,033 targets de respuesta supervisados, puntuados aquí por primera vez;
- CONFIRM512: 512 ejemplos / 17,984 targets; los controles FP fijo y W4A4 nativo
  ya estaban congelados, pero ninguna de las rutas mixtas se había puntuado allí;
- agrupados: 704 ejemplos / 25,017 targets.

Cada estadística es una razón pareada de sumas a nivel de secuencia, promediada por igual sobre
las semillas de entrenamiento 42, 7 y 123. Los intervalos usan 2,000 remuestreos bootstrap
percentiles pareados con semilla bootstrap 76.

Un candidato aprueba solo si:

1. el límite superior pareado del 95% de la brecha agrupada frente a FP fijo es igual o
   inferior a 0.1 nat/target;
2. el límite superior pareado del 95% del cambio agrupado frente a W4A4 nativo es menor que cero;
3. el cambio puntual frente a W4A4 nativo es negativo para cada semilla de entrenamiento en
   ambos paneles.

## Resultado

Una brecha menor es mejor. Un cambio negativo frente a W4A4 nativo es una mejora.

| Panel | Candidato | Brecha vs. FP fijo | Intervalo pareado del 95% | Cambio vs. W4A4 nativo |
| --- | --- | ---: | ---: | ---: |
| DEV192B | un A16 | 0.092837 | [0.078875, 0.106514] | -0.007607 |
| DEV192B | siete A16 | 0.083981 | [0.070769, 0.096989] | -0.016463 |
| CONFIRM512 | un A16 | 0.096218 | [0.088824, 0.103862] | -0.006107 |
| CONFIRM512 | siete A16 | 0.087268 | [0.080281, 0.094832] | -0.015057 |
| Agrupados 704 | un A16 | **0.095267** | **[0.088739, 0.102443]** | **-0.006529** |
| Agrupados 704 | siete A16 | **0.086344** | **[0.080241, 0.093190]** | **-0.015452** |

El candidato principal de un A16 mejoró frente a W4A4 nativo con cada semilla y en ambos paneles,
pero su límite superior agrupado fue 0.102443. Por eso falló, sin redondear.

El rescate de siete A16 aprobó. Su cambio agrupado frente a W4A4 nativo fue
**-0.015452 [-0.017552, -0.013528]**, y su cambio puntual fue negativo para cada una
de las semillas 42, 7 y 123 en cada panel. La auditoría independiente volvió a calcular las
estadísticas por panel, agrupadas y por semilla, y seleccionó `seven-a16-block23` de la
jerarquía declarada.

## Ficha del instrumento y contabilidad de ejecución

| Elemento | Configuración registrada |
| --- | --- |
| SO / Python | Windows 11 `10.0.26200`; Python `3.12.10` |
| GPU | 2 x NVIDIA GeForce RTX 3090, 24,576 MiB cada una |
| Controlador NVIDIA | `591.86` |
| PyTorch / compilación CUDA | `2.11.0+cu128` / `12.8` |
| Transformers / NumPy | `5.16.1` / `2.5.3` |
| Otros paquetes capturados | safetensors `0.8.0`; huggingface-hub `1.30.0`; tokenizers `0.23.2` |
| Modelo | `ahxt/LiteLlama-460M-1T`, revisión `77b8a976440e7d1ea5a890eaf1e0175b1cac0078` |
| Dataset | `databricks/databricks-dolly-15k`, revisión `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |
| Trabajo realizado | 2,496 pasadas forward; **cero pasadas backward y cero actualizaciones del optimizador** |
| Worker | completado y recolectado; 721.312 s; RSS máximo muestreado 5,426,659,328 bytes |

Ambas fichas fueron visibles para la evaluación. Estas cifras describen
la ejecución de confirmación; no son benchmarks controlados de latencia, throughput ni memoria.

## Historial de consumo de paneles

| Panel | Estado después de RD76 |
| --- | --- |
| RD70 DEV192 | Ya consumido para seleccionar la ruta; excluido de esta confirmación |
| RD73 DEV192B | Evaluado aquí por primera vez; ahora consumido |
| RD75 CONFIRM512 | Existían controles; ambas rutas mixtas se evaluaron aquí por primera vez; ahora consumido |
| Agrupados 704 | Análisis principal sobre DEV192B + CONFIRM512; sin ejemplos adicionales |
| FINAL256 reservado | **Nunca evaluado. Sigue reservado.** |

Por lo tanto, la aprobación respalda la ruta mixta congelada en un modelo. FINAL256 no se
consumió para tomar la decisión de ingeniería.

## Vínculos de procedencia

Los archivos retenidos permanecen locales; los hashes vinculan este registro público con el
protocolo congelado, el código, los paneles, los endpoints, el resultado, el comprobante del worker
y la auditoría independiente.

| Artefacto | SHA-256 |
| --- | --- |
| Manifiesto de congelamiento | `52af1943044d7b874546bd2727c742db1b14a3477d336bb71942a1106ee00c91` |
| Resultado | `94fee10542da29281f7753cbf221a3421ad5acf67f2b290e52d65acece359cdf` |
| Comprobante del worker | `f9f4527c9e7a376c08e41f0041b5e7121b524183c60199346242d39cd50a492e` |
| Auditoría independiente | `09676b7efed3dcdf1951ebd6ad03fe4ddbd492904a4f3d624476b22dbe1f6eb7` |
| Texto del protocolo | `df888abfc9a9bcd4e01f1f17fa39747d546f8d866d24692d4570688aa5dfe4ef` |
| Manifiesto DEV192B | `9fe9b2b1edafd739eefa5dcfc341a6500aae98b80f656d21116dabf6d896b9a0` |
| Panel DEV192B | `3c5e438f9dd7282396721da81423314a7111ea297feac3883d6a68cbfe641961` |
| Manifiesto CONFIRM512 | `879c7d080dae1aa4c3c04c580197370f101eca5656612588045792d7a6b763eb` |
| Panel CONFIRM512 | `6b981b44622ead9ca5a90f2475a3546615a9513d448a23938297386e9b095a8d` |

Hashes congelados de endpoints:

| Semilla de entrenamiento | SHA-256 |
| ---: | --- |
| 42 | `a2f92e94ba8364709acd924797ab9732df9e0205ffb4c4128807d65dca81bda4` |
| 7 | `15226c16dcac195fca285f4e3da52d1c675c82d1f963c03e58f0549687c0a801` |
| 123 | `fdd52fa24cebdb76343e18c7b0b5c3d8252a0388f548322c5d5d1ac831c5acdb` |

Hashes del código RD76 congelado:

| Archivo | SHA-256 |
| --- | --- |
| `rd76/__init__.py` | `d73f91d66bc04dc65a8d1fd284402eaac7623d0034152309e0fa7bb64671e965` |
| `rd76/confirm_mixed.py` | `c53c16616057bcef3c32af26aeea28fc765cdbc73e5ad27c2f5f9887723abec4` |
| `rd76/test_confirm_mixed.py` | `0854f9bceb34dc667e0bfdbe9f6211524b0f52697fa5b3104cc63ebd7cb86f35` |

La auditoría independiente marcó como verdaderas todas las comprobaciones declaradas, incluidos
los hashes de código y dependencias, la contabilidad del grafo de cómputo, los vínculos de paneles
congelados, las estadísticas exactas por panel y agrupadas, la selección jerárquica, la finalización
del worker y el vínculo con su comprobante. Registra `final_evaluated: false`.

## Alcance y límites

Este resultado tiene un alcance deliberadamente acotado:

- **no es W4A4 estricto**: siete de las 168 rutas de activación son A16;
- **no es paridad numérica con QuEST upstream**;
- usa cuantización simulada densa, no kernels INT4 empaquetados;
- no hace afirmaciones sobre latencia, throughput, VRAM ni eficiencia de despliegue;
- no contiene pasada backward ni valida la calidad del entrenamiento con ruta mixta;
- cubre una arquitectura de modelo y no establece generalidad entre modelos;
- no es por sí mismo una integración de Soup ni un contrato genérico de guardado/carga.

La integración de ingeniería debe preservar exactamente esta ruta, hacer explícitos sus metadatos
y rechazar topologías no admitidas en vez de generalizar silenciosamente el resultado.
