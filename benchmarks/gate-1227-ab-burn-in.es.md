# Calibración de burn-in de `soup ab` (#1227)
> 🇪🇸 Traducción al español de [`gate-1227-ab-burn-in.md`](gate-1227-ab-burn-in.md). Ante cualquier duda, manda el original en inglés.

Medido el 2026-09-25 para [#1227](https://github.com/MakazhanAlpamys/Soup/issues/1227), en
CPU. Este es el registro que respalda `BURN_IN_ROWS_BY_ALPHA` y `CALIBRATED_HORIZON_ROWS` en
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py).

**Reemplazado por [#1265](gate-1265-ab-nig.md):** `soup ab` ahora usa un estadístico robusto a la varianza
sin burn-in, y ya no existen `BURN_IN_ROWS_BY_ALPHA` / `CALIBRATED_HORIZON_ROWS`. Este registro
y su harness se mantienen como la línea base con la que se compara el barrido.

## Qué controla

`soup ab` ejecuta un mSPRT bilateral. Promedia las razones de verosimilitud de Wald para una
diferencia tratamiento menos control de `+effect_size` y de `-effect_size`, y estima
la varianza a partir de las filas. Con unas pocas filas, esa estimación puede resultar demasiado pequeña.
Un operador que vuelve a ejecutar el comando después de cada nuevo par de filas podría entonces rechazar un H0 verdadero
con mucha más frecuencia que `--alpha`. La simulación de abajo sitúa ese valor hasta en **0.164** con alpha
0.05 sin burn-in; la prueba unilateral que reemplazó la bilateral llegó a **0.109**.

Por eso `soup ab` no da un veredicto, solo `continue`, hasta que cada brazo tenga
`min_rows_per_arm(alpha)` filas. Este registro explica cómo se eligieron esos valores y hasta dónde son válidos.

## Resumen

| `--alpha` | burn-in (filas por brazo) | peor error Tipo I, hasta 1000 filas | peor, hasta 200 filas | límite (alpha + 3 SE) |
|---|---|---|---|---|
| 0.05 y más | **30** | alpha 0.05: **0.0512** (ratio 0.35) · alpha 0.10: 0.0994 | 0.0492 · 0.0934 | 0.0521 · 0.1029 |
| 0.01 a menos de 0.05 | **40** | alpha 0.01: **0.0109** (0.01088, ratio 0.4) · alpha 0.025: 0.0261 | 0.0105 · 0.0244 | 0.0109 (0.01094) · 0.0265 |
| menos de 0.01 | 40, **sin calibrar** | alpha 0.005: 0.0058, por encima de su límite | 0.0056 | 0.0057 |

"Ratio" es effect_size / sigma, el único valor del que depende la decisión. La calibración es válida
hasta **1000 filas por brazo**; después de eso, `soup ab` imprime una advertencia. Para alpha menor que 0.01 imprime
otra.

## Procedimiento

- **Datos.** H0: filas de control y tratamiento con distribución N(0, sigma²), con effect_size 0.1 y
  sigma = 0.1 / ratio. Hay 22 ratios: 0.1, 0.15, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35,
  0.375, 0.4, 0.425, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1, 1.5, 2, 3 y 5.
- **Procedimiento.** Se observa después de cada nuevo par. El primer veredicto terminal (`reject_h0` o
  `accept_h0`) detiene la ejecución y no hay veredicto antes del burn-in. Una ejecución que sigue indecisa
  al llegar al horizonte se detiene allí. Los horizontes son 1000 y 200 filas por brazo.
- **Estadístico.** Se calcula exactamente como lo hace `soup ab`: la varianza se estima
  a partir de las filas (corregida con Bessel, combinada), con beta 0.20.
  - Cada ejecución del harness primero compara su estadístico vectorizado con el código de decisión incluido,
    `_verdict_from_summary`: diferencia relativa máxima 0 en 300 puntos,
    en las cuatro partes.
- **Ejecuciones.** 100,000 por ratio, con semilla por ratio.
- **Criterio.** Para cada nivel de alpha, el menor burn-in de {20, 30, 40, 50, 60} cuya tasa
  se mantiene en o por debajo de alpha + 3·sqrt(alpha(1−alpha)/100000) en **cada ratio**, en **ambos
  horizontes** y para **cada alpha medido en el nivel**: 0.05 y 0.10 para el primer nivel,
  0.01 y 0.025 para el segundo.

## Decisión

Peor tasa entre los 22 ratios. ✗ indica un valor por encima del límite.

| nivel | alpha | horizonte | N = 20 | N = 30 | N = 40 | N = 50 | N = 60 |
|---|---|---|---|---|---|---|---|
| ≥ 0.05 | 0.05 | 1000 | 0.05342 ✗ | **0.05119** | 0.04977 | 0.04965 | 0.04937 |
| ≥ 0.05 | 0.05 | 200 | 0.05206 | 0.04919 | 0.04672 | 0.04409 | 0.04210 |
| ≥ 0.05 | 0.10 | 1000 | 0.10073 | 0.09935 | 0.09817 | 0.09757 | 0.09674 |
| ≥ 0.05 | 0.10 | 200 | 0.09948 | 0.09335 | 0.08784 | 0.08419 | 0.08060 |
| 0.01–0.05 | 0.01 | 1000 | 0.01224 ✗ | 0.01135 ✗ | **0.01088** | 0.01077 | 0.01075 |
| 0.01–0.05 | 0.01 | 200 | 0.01224 ✗ | 0.01111 ✗ | 0.01045 | 0.00988 | 0.00934 |
| 0.01–0.05 | 0.025 | 1000 | 0.02764 ✗ | 0.02686 ✗ | 0.02611 | 0.02511 | 0.02496 |
| 0.01–0.05 | 0.025 | 200 | 0.02719 ✗ | 0.02550 | 0.02440 | 0.02305 | 0.02185 |
| < 0.01 | 0.005 | 1000 | 0.00721 ✗ | 0.00601 ✗ | 0.00577 ✗ | 0.00563 | 0.00560 |
| < 0.01 | 0.005 | 200 | 0.00721 ✗ | 0.00601 ✗ | 0.00558 | 0.00539 | 0.00501 |

Los límites son 0.05207 para alpha 0.05, 0.10285 para 0.10, 0.01094 para 0.01, 0.02648 para 0.025 y
0.00567 para 0.005.

**Elegido:** 30 filas con alpha ≥ 0.05 y 40 filas con 0.01 ≤ alpha < 0.05. Por debajo de 0.01 el valor
se mantiene en 40 con una advertencia, porque 40 filas no pasan para alpha 0.005 en 1000 filas.

## Qué muestra esto

- **El horizonte importa.** Una calibración anterior detenía cada ejecución en 200 filas. Eligió 20
  filas con alpha 0.05, y aquí 20 todavía pasa con 200 filas (0.05206). Si se prolongan hasta 1000 filas,
  las ejecuciones que siguen indecisas a las 200 continúan produciendo rechazos falsos, y 20 filas llega a
  0.05342. Por eso el burn-in está calibrado para ejecuciones de hasta 1000 filas por brazo.
- **La cola de ratios pequeños no depende del burn-in.** Con ratio 0.1, 30 filas dan 0.0003
  a 200 filas y 0.0278 a 1000 (alpha 0.05), porque esos rechazos ocurren tarde, mucho después
de cualquier burn-in. Lo mismo ocurre con alpha 0.01: después de 40 filas, el peor caso solo pasa de
  0.01088 a 0.01075 (N = 60), un piso alrededor de ratio 0.25.
- **El nivel alpha 0.01 solo pasa con este conjunto de semillas:** 0.01088 frente a 0.01094, cerca de 0.2
  errores estándar. Una revisión independiente midió 0.0113 (ratio 0.30) y 0.0116 (ratio 0.50)
  en dos de otros tres conjuntos de semillas, con un peor caso combinado cercano a 0.0108. Ese piso lo fija
  la varianza estimada, no el burn-in: 50 y 60 filas no lo reducen, y las mismas ejecuciones
  con la varianza verdadera dan 0.0095. Así que con alpha 0.01 la tasa de falsos positivos al observar repetidamente es
  cerca de 0.0108, aproximadamente 8% por encima del nivel nominal; la tolerancia del límite es +9.4%, no +10%.
  Se mantienen 40 filas porque más filas no aportan nada, y #1265 elimina el piso.
- **Con la varianza verdadera**, el mismo estadístico se mantiene por debajo de alpha en todas partes: peor caso
  0.0468 con alpha 0.05 y 0.0095 con alpha 0.01, hasta 1000 filas. El efecto que queda tras el burn-in es el de
  la estimación de varianza, no el de la mezcla.
- **Las alternativas, varianza estimada, peor caso entre ratios con 1000 filas:**

  | estadístico | sin burn-in, alpha 0.05 | 30 filas, alpha 0.05 | sin burn-in, alpha 0.01 | 40 filas, alpha 0.01 |
  |---|---|---|---|---|
  | mezcla de dos puntos (incluida) | 0.1637 | 0.0512 | 0.1009 | 0.0109 |
  | unilateral (la prueba reemplazada) | 0.1091 | 0.0512 | 0.0603 | 0.0109 |
  | el atajo `\|z\|` | 0.2232 | 0.1074 | 0.1216 | 0.0221 |

  `|z|` en la fórmula unilateral casi duplica la tasa con cualquier burn-in. Por eso el
  estadístico promedia las dos razones en su lugar.

## Qué no muestra esto

- **Solo beta 0.20.** Una comprobación independiente del código anterior, con las mismas ejecuciones con beta
  0.05, 0.10 y 0.30, encontró que un beta menor es más conservador y que 0.30 está dentro del ruido de 0.20.
- **Solo filas gaussianas.** La documentación de `soup ab` indica que las métricas binarias deben agregarse previamente por
  prompt.
- **Nada más allá de 1000 filas por brazo.** Los ratios pequeños siguen acumulándose allí, por lo que
  `soup ab` muestra una advertencia.
- **No es un estadístico robusto a la varianza.** El estadístico que no necesitaría burn-in está en
  [#1265](https://github.com/MakazhanAlpamys/Soup/issues/1265).

## Entorno

Windows 11, Python 3.12.10, numpy 2.5.3, solo CPU. Cerca de 55 s por ratio con 100,000 ejecuciones sobre
1000 filas, con `--known-variance`.

## Reproducir

El harness es [`harness/ab_burn_in_sweep.py`](harness/ab_burn_in_sweep.py). Ejecútalo en partes
y luego combínalas; las partes se dividen para ajustarse a una sesión de 10 minutos:

```bash
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 0.1,0.15,0.2,0.25,0.275,0.3,0.325 --out part1.json
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 0.35,0.375,0.4,0.425,0.45,0.5,0.55 --out part2.json
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 0.6,0.7,0.8,1.0 --out part3.json
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 1.5,2.0,3.0,5.0 --out part4.json
python benchmarks/harness/ab_burn_in_sweep.py report part1.json part2.json part3.json part4.json \
    --ratios-detail 0.25,0.4,0.425,0.55,1.0,2.0
```

Las partes y el informe combinado de esta ejecución están en
[`results/gate-1227-ab-burn-in/`](results/gate-1227-ab-burn-in/): `part1.json` a `part4.json`
y `sweep.log`. Las semillas están fijadas por ratio, así que la misma versión de numpy reproduce los números anteriores.
