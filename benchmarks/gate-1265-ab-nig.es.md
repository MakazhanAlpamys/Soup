# Mezcla normal-inversa-gamma de `soup ab` (#1265)
> 🇪🇸 Traducción al español de [`gate-1265-ab-nig.md`](gate-1265-ab-nig.md). Ante cualquier duda, manda el original en inglés.

Medido el 2026-09-27 para [#1265](https://github.com/MakazhanAlpamys/Soup/issues/1265), en
CPU. Este es el registro que respalda el estadístico y `PRIOR_SCALE_ROWS` en
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py). Reemplaza el burn-in
de [#1227](gate-1227-ab-burn-in.md).

**Regla de aceptación reemplazada por [#1418](gate-1418-ab-cs-accept.md):** `soup ab` ahora acepta H0
cuando una secuencia de confianza de la misma mezcla queda dentro de +-effect_size, no en
`log(beta/(1-alpha))`. El estadístico, `PRIOR_SCALE_ROWS` y la regla de rechazo registrados aquí
no cambian, y las ejecuciones de este registro son la línea base que reproduce el barrido.

## Qué controla

El estadístico de #1227 sustituyó la varianza estimada en una razón de verosimilitud construida para una varianza conocida.
Con unas pocas filas, esa estimación tiene demasiado ruido, así que `soup ab` retrasaba todos los veredictos
hasta tener 30 filas por brazo (40 por debajo de alpha 0.05). Aun así, con alpha 0.01 la tasa de error Tipo I al observar repetidamente
se mantuvo cerca de 0.011, los burn-in más largos no la redujeron, y por debajo de alpha 0.01 o después de
1000 filas por brazo no estaba calibrada en absoluto.

Ahora el estadístico es el factor de Bayes normal-inversa-gamma en su límite de Haar derecho: un prior plano
sobre la media común, 1/sigma sobre sigma y N(0, g) sobre la diferencia estandarizada
d = delta / sigma. Tiene una forma cerrada en el estadístico t combinado de dos muestras (Gönen, Johnson,
Lu y Westfall, 2005), con nu = n_c + n_t - 2 y n_eff = n_c n_t / (n_c + n_t):

```text
log BF = -0.5 log(1 + n_eff g) - (nu + 1)/2 [log1p(t² / (nu (1 + n_eff g))) - log1p(t² / nu)]
```

Es invariante a escala, así que bajo H0 su distribución no depende de sigma y, como razón de
verosimilitudes marginales, es una martingala bajo H0. La desigualdad de Ville limita a alpha la probabilidad de que
alguna vez alcance 1/alpha: desde las primeras filas, con cualquier horizonte y cualquier alpha. `soup ab`
rechaza en `log(1/alpha)` y acepta en `log(beta/(1-alpha))`.

`--effect-size` usa las unidades de la métrica y g necesita ese valor en desviaciones estándar. Las primeras
`PRIOR_SCALE_ROWS` filas de cada brazo se reservan; su desviación estándar combinada s0 determina
g = (effect_size / s0)² y el factor de Bayes se ejecuta sobre las filas posteriores. Como el prior
se fija antes de observar esas filas, la garantía sigue siendo exacta. Este registro compara el
diseño con el de burn-in y elige `PRIOR_SCALE_ROWS`.

## Resumen

Peor tasa de error Tipo I entre 22 valores de effect_size / sigma, 100,000 ejecuciones cada uno (el límite es
alpha + 3 SE; * = por encima del límite):

| `--alpha` | NIG, k = 5, hasta 1000 filas | hasta 200 filas | burn-in (#1227), hasta 1000 filas | límite |
|---|---|---|---|---|
| 0.10 | **0.0676** | 0.0576 | 0.0999 | 0.1029 |
| 0.05 | **0.0328** | 0.0287 | 0.0511 | 0.0521 |
| 0.025 | **0.0164** | 0.0139 | 0.0257 | 0.0265 |
| 0.01 | **0.0066** | 0.0055 | 0.0110 * | 0.0109 |
| 0.005 | **0.0034** | 0.0027 | 0.0058 * | 0.0057 |

La mezcla se mantiene muy por debajo de alpha en todos los casos, sin burn-in y sin límite de alpha ni de horizonte,
así que desaparecen el piso de 0.01 y ambas advertencias de `soup ab`. **`PRIOR_SCALE_ROWS = 5`**: los veredictos pueden
emitirse desde 7 filas por brazo.

## Procedimiento

- **Datos.** effect_size 0.1, sigma = 0.1 / ratio, 22 ratios: 0.1, 0.15, 0.2, 0.25, 0.275,
  0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1, 1.5, 2, 3 y 5.
  H0: ambos brazos N(0, sigma²), 100,000 ejecuciones por ratio. H1: tratamiento desplazado
  +effect_size, la diferencia más pequeña que el operador pidió detectar, 20,000 ejecuciones por ratio
  (17 de los ratios).
- **Procedimiento.** Se observa después de cada nuevo par; el primer veredicto terminal detiene la ejecución; una ejecución
  que sigue indecisa al llegar al horizonte se detiene allí. Horizontes de 1000 y 200 filas por brazo. beta 0.20.
- **Diseños.** NIG con k = 3, 5 y 8 filas reservadas por brazo (veredictos desde el par k + 2), y
  el diseño de burn-in de #1227 (30 filas con alpha >= 0.05, 40 por debajo, rechazo en
  `log((1-beta)/alpha)`), calculado con las propias funciones del harness de #1227.
- **Autocomprobación.** Cada ejecución del harness primero compara su factor de Bayes vectorizado con
  `_nig_log_bayes_factor` incluido (diferencia relativa máxima 0 en 150 puntos) y reproduce
  las ejecuciones simuladas mediante el propio `msprt_step` (0 de 64 observaciones difieren), en las 11 partes.
- **Forma cerrada.** `tests/test_issue1265_ab_nig.py` también compara la forma cerrada con
  una cuadratura directa de las dos verosimilitudes marginales (con precisión de 2e-4, incluso para brazos desiguales y
  nu = 2).

## Potencia frente al efecto solicitado

Proporción de ejecuciones H1 que terminan en `reject_h0` con la dirección correcta y promedio de pares al detenerse,
hasta 1000 filas por brazo. Los rechazos en la dirección incorrecta fueron como máximo 0.0002 en los diseños mostrados.

| ratio | burn-in, alpha 0.05 | NIG k=5, alpha 0.05 | burn-in, alpha 0.01 | NIG k=5, alpha 0.01 |
|---|---|---|---|---|
| 0.1 | 0.446 / 800 | 0.311 / 885 | 0.176 / 922 | 0.130 / 960 |
| 0.15 | 0.768 / 450 | 0.724 / 652 | 0.637 / 628 | 0.513 / 805 |
| 0.2 | 0.811 / 261 | **0.946** / 431 | 0.790 / 389 | **0.859** / 583 |
| 0.3 | 0.817 / 119 | **0.996** / 201 | 0.818 / 177 | **0.997** / 279 |
| 0.5 | 0.840 / 49 | **0.996** / 79 | 0.864 / 71 | **0.997** / 107 |
| 1 | 0.977 / 31 | **0.997** / 26 | 0.989 / 41 | **0.997** / 33 |
| 2 | 1.000 / 30 | 0.998 / 12 | 1.000 / 40 | 0.999 / 15 |
| 5 | 1.000 / 30 | 1.000 / 8 | 1.000 / 40 | 1.000 / 9 |

A partir de ratio 0.2, la mezcla detecta el efecto con más frecuencia, y desde ratio 1 también
llega a una decisión más rápido. La potencia del diseño con burn-in se estanca cerca de 0.82 entre ratios 0.2 y 0.5
porque termina incorrectamente alrededor del 18% de esas ejecuciones en `accept_h0`. Por eso también se detiene
antes allí. Por debajo de ratio 0.2 ninguno de los diseños es confiable dentro de 1000 filas (una prueba de n fijo
necesita cerca de 1570 por brazo con ratio 0.1), y el diseño con burn-in detecta más, aunque está en
su límite de error Tipo I. Con un horizonte de 200 filas se mantiene el mismo patrón, desplazado: el diseño con burn-in
detecta más hasta ratio 0.35 con alpha 0.05 (0.4 con alpha 0.01) y la mezcla por encima de ese valor
(tablas completas en `report.txt`).

## Costo: tiempo hasta `accept_h0`

Bajo H0, la mezcla necesita más filas para aceptar. Promedio de pares al detenerse, ejecuciones hasta 1000 filas (sección
"H0: time to accept_h0" de `report.txt`):

| ratio | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| burn-in, alpha 0.05 | 235 | 107 | 62 | 43 | 34 | 31 | 30 | 30 | 30 |
| NIG k=5, alpha 0.05 | 889 | 621 | 406 | 281 | 155 | 82 | 41 | 26 | 10 |

Un factor de Bayes de mezcla reúne evidencia a favor de H0 solo como -0.5 log(n_eff g), así que alcanzar
`log(beta/(1-alpha))` requiere filas. Las aceptaciones tempranas del diseño con burn-in provienen de la misma
varianza de sustitución que le resta potencia bajo H1. Hasta entonces, `continue` sigue siendo la respuesta honesta:
`soup ab` nunca declara `accept_h0` antes de tiempo.

## Elegir `PRIOR_SCALE_ROWS`

El error Tipo I está dentro de su límite para todo k, como debe ser. Potencia con ratio >= 0.3, alpha 0.05,
1000 filas: cerca de 0.980 para k = 3, 0.996 para k = 5, 0.999 para k = 8. Cada fila reservada agrega
cerca de un par a cada decisión, y k = 3 pierde 1.5 puntos de potencia debido a una escala de prior con más ruido.
k = 5 conserva casi toda la potencia de k = 8 con 3 filas menos.

## Límites

- **Filas gaussianas.** Como antes, las métricas binarias deben agregarse previamente por prompt.
- **Solo beta 0.20** en las ejecuciones H1; beta ahora solo determina el límite de aceptación.
- **El orden de las filas reservadas importa.** Las primeras k filas de cada brazo, en el orden del archivo,
  determinan la escala. Si las primeras filas de un brazo son todas idénticas, se reservan más filas hasta que alguna
  difiera; esto depende únicamente de las filas reservadas, así que la garantía no cambia.
- **Estos barridos son anteriores a la retención de aceptación.** `ACCEPT_HOLD_SPREAD_RATIO` (#1419) se agregó después
  de escribir este registro, así que todos los números anteriores proceden de una ejecución con la retención DESACTIVADA. En filas
  gaussianas las dos dispersiones coinciden, por lo que la retención casi no cambia los veredictos: 2 de las 600 ejecuciones con semilla fija en
  `tests/test_issue1265_ab_nig.py`; en una solo cambia el número de filas que tomó y en otra rechaza
  en vez de aceptar, que es la dirección segura. Las conclusiones de aquí se mantienen. Las cifras de tiempos
  en particular no se midieron con la retención activada.
- **Una retención que se activa durante un calentamiento saturado nunca se libera con más filas.** Las filas reservadas
  son fijas, así que una ejecución puede quedarse en `continue` durante todo el dataset mientras la secuencia de confianza
  para la diferencia ya está dentro de `+-effect_size`. `soup ab` ahora informa ese estado (`accept_held` más las dos dispersiones)
  en el panel y en `MsprtVerdict` que devuelven `run_msprt` / `msprt_step`, en vez de solo recomendar más muestras (#1524). No tiene
  salida JSON y una ejecución retenida no envía webhook. #1418 reemplazó la regla de aceptación y mantuvo
  esta retención, así que sigue aplicándose.

## Entorno

Windows 11, Python 3.12.13, numpy 2.5.2, solo CPU. Cerca de 350 s por ratio para una parte H0 con
100,000 ejecuciones sobre 1000 filas (8 partes en paralelo), y cerca de 70 s por ratio para H1 con 20,000.

## Reproducir

El harness es [`harness/ab_nig_sweep.py`](harness/ab_nig_sweep.py); importa el diseño de burn-in
de [`harness/ab_burn_in_sweep.py`](harness/ab_burn_in_sweep.py). Ejecútalo en partes
y luego combina los resultados:

```bash
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.1,0.15,0.2 --out h0-part1.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.25,0.275,0.3 --out h0-part2.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.325,0.35,0.375 --out h0-part3.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.4,0.425,0.45 --out h0-part4.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.5,0.55,0.6 --out h0-part5.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.7,0.8,1.0 --out h0-part6.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 1.5,2.0 --out h0-part7.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 3.0,5.0 --out h0-part8.json
python benchmarks/harness/ab_nig_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.1,0.15,0.2,0.25,0.3,0.35 --out h1-part1.json
python benchmarks/harness/ab_nig_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.4,0.45,0.5,0.6,0.7,0.8 --out h1-part2.json
python benchmarks/harness/ab_nig_sweep.py run --mode h1 --runs 20000 \
    --ratios 1.0,1.5,2.0,3.0,5.0 --out h1-part3.json
python benchmarks/harness/ab_nig_sweep.py report h0-part*.json h1-part*.json
```

Las partes, el informe combinado y el registro de ejecución están en
[`results/gate-1265-ab-nig/`](results/gate-1265-ab-nig/): `h0-part1.json` a `h0-part8.json`,
`h1-part1.json` a `h1-part3.json`, `report.txt` y `sweep.log`. Las semillas están fijadas por
modo y ratio, así que la misma versión de numpy reproduce los números anteriores.
