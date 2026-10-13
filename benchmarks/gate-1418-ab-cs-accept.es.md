# Regla de aceptación por secuencia de confianza de `soup ab` (#1418)
> 🇪🇸 Traducción al español de [`gate-1418-ab-cs-accept.md`](gate-1418-ab-cs-accept.md). Ante cualquier duda, manda el original en inglés.

Medido el 2026-10-02 para [#1418](https://github.com/MakazhanAlpamys/Soup/issues/1418), en
CPU. Este es el registro que respalda la regla de aceptación en
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py). Solo cambia cómo
`soup ab` llega a `accept_h0`; el estadístico y la regla de rechazo son los de
[#1265](gate-1265-ab-nig.md). `--beta` queda obsoleto.

## Qué controla

Con #1265, `soup ab` aceptaba H0 cuando el factor de Bayes normal-inversa-gamma bajaba a
`log(beta / (1 - alpha))`. Un factor de Bayes de mezcla solo acumula evidencia a favor de H0 como
-0.5 log(n_eff g), así que, sin diferencia, tardaba 621 pares en aceptar con effect_size / sigma
0.3 (alpha 0.05, beta 0.20, ejecuciones hasta 1000 filas), frente a 107 para el diseño con burn-in de #1227.
Parte de la rapidez del diseño con burn-in era incorrecta: terminaba cerca del 18% de las ejecuciones con una
 diferencia real de effect_size en `accept_h0`. El objetivo era aceptar más rápido sin perder esa ventaja
y sin cambiar la tasa de error Tipo I ni la potencia de `reject_h0`.

La regla de rechazo no cambia: `reject_h0` cuando el factor de Bayes alcanza `1/alpha`. La
regla de aceptación ahora lee la misma mezcla de la otra forma. Desplazar el tratamiento por una
diferencia candidata delta0 deja intacta la varianza combinada y mueve t a
(mean difference - delta0) / se; así, el factor de Bayes en ese valor de t prueba «la diferencia es
delta0» y es una martingala cuando delta0 es la diferencia verdadera. Las diferencias que no
ha rechazado en `1/level` forman una secuencia de confianza siempre válida, mean difference +- h, y,
por la desigualdad de Ville, la diferencia verdadera permanece dentro de ella en cada observación con probabilidad de al
menos 1 - level. El factor de Bayes crece con t², así que h tiene una forma cerrada. Con
spread = 1 + n_eff g y nu, n_eff como en #1265:

```text
c  = exp(-(2 log(1/level) + log(spread)) / (nu + 1))
t² = nu (1 - c) / (c - 1/spread)          (h = inf when c <= 1/spread)
h  = sqrt(t² * pooled_variance / n_eff)
```

`soup ab` acepta H0 cuando |mean difference| + h < effect_size: toda diferencia que las filas
aún permiten es menor que la que el operador preguntó. **El nivel es `--alpha`.** Una
diferencia de effect_size o mayor termina entonces en `accept_h0` en, como máximo, una proporción alpha de
las ejecuciones, sin importar cuántas veces observe el operador; por tanto, `--alpha` limita ambos veredictos incorrectos y
`--beta` ya no tiene nada que configurar. Más abajo se explica por qué alpha y no beta; se conservan los números de beta.

## Resumen

Promedio de pares al detenerse cuando los dos brazos son iguales, ejecuciones hasta 1000 filas por brazo, 100,000 ejecuciones
por ratio:

| effect_size / sd | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| #1265, alpha 0.05, beta 0.20 | 890 | 621 | 407 | 281 | 155 | 83 | 42 | 27 | 10 |
| **#1418, alpha 0.05** | **661** | **307** | **177** | **117** | **64** | **35** | **20** | **15** | **9** |
| #1265, alpha 0.01, beta 0.20 | 921 | 671 | 448 | 314 | 175 | 94 | 47 | 30 | 11 |
| **#1418, alpha 0.01** | **840** | **411** | **236** | **154** | **83** | **45** | **25** | **18** | **10** |
| #1227 burn-in, alpha 0.05 | 235 | 107 | 62 | 43 | 34 | 31 | 30 | 30 | 30 |

Aceptar H0 es de 2.0 a 2.4 veces más rápido con alpha 0.05 desde ratio 0.3 hasta 1 (de 1.6 a 2.1 con
alpha 0.01), y a partir de ratio 1.5 también es más rápido que el diseño con burn-in.

Frente a una diferencia verdadera exactamente igual a effect_size (20,000 ejecuciones por ratio, 17 ratios de 0.1
a 5, ejecuciones hasta 1000 filas):

| `--alpha` | peor `accept_h0` incorrecto, #1265 (beta 0.20) | peor `accept_h0` incorrecto, #1418 | potencia desde ratio 0.3, #1265 | potencia desde ratio 0.3, #1418 |
|---|---|---|---|---|
| 0.10 | 0.0063 | 0.0158 | 0.99360 | 0.98420 |
| 0.05 | 0.0042 | 0.0083 | 0.99585 | 0.99165 |
| 0.025 | 0.0036 | 0.0042 | 0.99650 | 0.99585 |
| 0.01 | 0.0032 | 0.0019 | 0.99685 | 0.99805 |
| 0.005 | 0.0031 | 0.0012 | 0.99665 | 0.99815 |

La potencia es la menor proporción de ejecuciones que rechazan en la dirección correcta entre ratios 0.3 y 5
(exacta con 20,000 ejecuciones, así que no está redondeada).
El diseño de burn-in de #1227 tenía cerca de 0.18 de aceptaciones incorrectas y 0.82 de potencia entre ratios 0.2
y 0.5.

## Procedimiento

- **Datos y observaciones.** Como en el registro de #1265: effect_size 0.1, sigma = 0.1 / ratio, los mismos
  22 ratios; H0: ambos brazos N(0, sigma²), 100,000 ejecuciones por ratio; H1: tratamiento desplazado por
  +effect_size, 20,000 ejecuciones por ratio en 17 de los ratios; una observación después de cada par desde el par 7
  en adelante, el primer veredicto terminal detiene la ejecución, horizontes de 1000 y 200 filas por brazo;
  `PRIOR_SCALE_ROWS = 5`; alphas 0.1, 0.05, 0.025, 0.01 y 0.005; betas 0.20 y 0.10 para
  los diseños que usan uno.
- **Mismas ejecuciones.** Las semillas y el orden de las extracciones son los de `ab_nig_sweep.py`, así que cada diseño
  ve exactamente las mismas ejecuciones que el registro de #1265.
- **Diseños.** `gate-1265`: la regla de #1265 sin la retención de aceptación, tal como se ejecutó en su registro.
  `nig`: la regla de #1265 con la retención incluida, la línea base. `cs-alpha`: la secuencia con
  nivel alpha y retención (incluida). `cs`: la secuencia con nivel beta y retención (la
  opción no elegida). `cs-nohold`: `cs` sin retención.
- **Autocomprobación.** Cada parte del harness primero compara su semiancho vectorizado con el incluido
  `_nig_confidence_half_width` (diferencia relativa máxima 0 en 45 puntos) y reproduce
  las ejecuciones simuladas mediante el propio `msprt_step` (0 de 32 ejecuciones difieren). El barrido se ejecutó cuando
  la rama incluía nivel beta, así que las partes comprobaron `cs`; tras el cambio a nivel alpha, el
  harness comprueba `cs-alpha` y `check` arroja los mismos 0 y 0 (al final de `sweep.log`). Si se apunta
  al nivel anterior, informa que 21 de 32 ejecuciones difieren, lo que permite distinguir ambos casos.
  Ningún conteo depende del diseño incluido, así que no se volvió a ejecutar el barrido.
- **Comprobación de reproducción.** Cada parte también compara sus conteos de `gate-1265` con
  `results/gate-1265-ab-nig/`: difieren 0 de 390 celdas en las 11 partes, así que la línea base es
  la registrada, no una reimplementación.

## `reject_h0` no cambia

La regla de rechazo usa la misma función con el mismo límite, así que la decisión de rechazo en cualquier observación
es idéntica. Lo que puede cambiar es la frecuencia con que una ejecución llega a una observación que rechaza, porque
una aceptación ahora termina algunas ejecuciones antes.

- **El error Tipo I solo puede bajar**, y así ocurre. Peor tasa entre los 22 ratios, ejecuciones hasta 1000 filas
  (límite alpha + 3 SE, como en #1265):

| `--alpha` | 0.10 | 0.05 | 0.025 | 0.01 | 0.005 |
|---|---|---|---|---|---|
| #1265 | 0.0676 | 0.0328 | 0.0164 | 0.0066 | 0.0034 |
| #1418 | 0.0594 | 0.0289 | 0.0145 | 0.0058 | 0.0030 |

  Con horizonte de 200 filas ocurre lo mismo (0.0287 a 0.0258 con alpha 0.05); ninguno supera su
  límite (`report.txt`).
- **La potencia para una diferencia exactamente igual a effect_size cambia según las aceptaciones incorrectas.** Proporción de ejecuciones H1 que rechazan
  en la dirección correcta, hasta 1000 filas:

| ratio | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|
| #1265, alpha 0.05 | 0.311 | 0.724 | 0.946 | 0.997 | 0.997 | 0.997 | 0.999 | 1.000 |
| #1418, alpha 0.05 | 0.311 | 0.724 | 0.947 | 0.993 | 0.993 | 0.992 | 0.994 | 1.000 |
| #1265, alpha 0.01 | 0.130 | 0.513 | 0.859 | 0.997 | 0.997 | 0.998 | 0.999 | 1.000 |
| #1418, alpha 0.01 | 0.130 | 0.513 | 0.860 | 0.998 | 0.998 | 0.999 | 0.999 | 1.000 |

  Desde ratio 0.3, con alpha 0.05 la potencia es de 0.004 a 0.005 menor; con alpha 0.01 o menos es igual o
  un poco mayor, porque las aceptaciones incorrectas de #1418 son menos frecuentes que las de #1265. Cada
  ejecución que termina en una aceptación `accept_h0` incorrecta habría seguido casi siempre hasta
  rechazar. Los rechazos en la dirección incorrecta fueron como máximo 0.0004 en cada diseño.

## Por qué nivel alpha y qué pasó con `--beta`

Se midieron y presentaron dos interpretaciones en el PR para que decidiera una persona responsable del mantenimiento:

- **Nivel beta (`cs`, no elegido).** `--beta` habría conservado su nombre y limitado las aceptaciones incorrectas a
  beta. Con el beta predeterminado de 0.20, acepta más rápido (220 pares con ratio 0.3 y alpha 0.05),
  pero termina hasta el 0.0389 de las ejecuciones con effect_size exacto en un `accept_h0` incorrecto (0.0430 en
  todos los alpha), así que la potencia cae de 0.997 a cerca de 0.965. Con beta 0.10: hasta 0.0175,
  potencia cerca de 0.985. Tablas completas en `report.txt`.
- **Nivel alpha (`cs-alpha`, incluido).** La potencia con effect_size se mantiene dentro de 0.005 de la actual con
  alpha 0.05 (0.0094 menos con 0.1) y se mantiene igual o por encima desde alpha 0.01 hacia abajo; ese es el criterio de #1418: sin cambios en el error Tipo I ni en
  la potencia de `reject_h0`.

Por lo tanto, `--beta` queda obsoleto. Se sigue analizando durante una versión: `soup ab --beta X` indica que
`--beta` ya no hace nada, que `--alpha` ahora también limita las aceptaciones incorrectas y que para una aceptación
más estricta se debe reducir `--alpha`; después se ejecuta exactamente igual que sin esa opción.
`MsprtConfig(beta=...)` lanza el mismo mensaje que un `SoupConfigDeprecationWarning`. Ambos dejan de admitirse
con la versión de lanzamiento indicada por `DEPRECATED_VALUE_REJECTION_VERSION`, siguiendo la misma transición de una versión que Soup usa para valores de configuración obsoletos (#759). La comprobación de #1339
`alpha + beta < 1` protegía el límite de aceptación anterior y desaparece junto con él.

## Retención de aceptación

#1265 incluyó una retención: mientras la desviación estándar de las filas evaluadas sea más de 3 veces la de las
filas reservadas, `accept_h0` espera y devuelve `continue`. Aquí no cambia; la pregunta de si debe mantenerse está en
[#1524](https://github.com/MakazhanAlpamys/Soup/issues/1524). Para ese issue:
la cobertura de la secuencia se mantiene con cualquier escala de prior fijada antes de las filas evaluadas, así que un
inicio saturado cambia el ancho de la secuencia, no su cobertura. En el barrido, la retención no produce cambios medibles (`cs`
frente a `cs-nohold`: menos de 1 par y como máximo 0.001 en la proporción de aceptación). En las filas de inicio saturado de
`tests/test_issue1265_ab_nig.py` sin retención, ambos desplazamientos se aceptan con 12 y 19 filas por brazo,
y ambas aceptaciones son correctas: los desplazamientos, 0 y -0.03, son menores que `--effect-size` 0.1.

## Qué indica `log_likelihood_ratio` al aceptar

El `log_likelihood_ratio` del veredicto (en la tabla y el payload del webhook) sigue siendo el
factor de Bayes frente a «sin diferencia», el estadístico de la regla de rechazo. No es lo que
acepta, así que en un `accept_h0` puede ser cualquier valor inferior a `log(1/alpha)`, incluso positivo: de
511 aceptaciones simuladas con alpha 0.05 (effect sizes de 0.5 a 3 sd, diferencias verdaderas de 0 a
0.8 effect_size), 82 tuvieron un valor positivo; el mayor fue 2.93 frente al límite de rechazo 3.00.
Un valor positivo allí significa que las filas se inclinan a favor de una diferencia menor que
effect_size.

## Límites

- **Filas gaussianas**, como antes. Las métricas binarias deben agregarse previamente por prompt.
- **H1 exactamente con effect_size.** Es el caso más difícil para la garantía de aceptación; una diferencia
  mayor se acepta con menor frecuencia.
- **Ejecuciones cortas con effect sizes pequeños.** Con un límite de 200 filas por brazo y ratio 0.3 o
  menos, #1418 llega a `accept_h0` con menos frecuencia que #1265 (0.017 frente a 0.041 de las ejecuciones con
  ratio 0.3, alpha 0.05); desde ratio 0.35 ocurre mucho más (0.426 frente a 0.096 con
  0.35). Ninguna regla acepta mucho en ese rango dentro de 200 filas.
- **La garantía es holgada en este caso.** Las aceptaciones incorrectas medidas están muy por debajo de alpha.

## Entorno

Windows 11, Python 3.12.13, numpy 2.5.2, solo CPU. Cerca de 430 a 600 s por ratio para una parte H0
con 100,000 ejecuciones sobre 1000 filas (8 partes en paralelo), y cerca de 130 s por ratio para H1 con
20,000.

## Reproducir

El harness es [`harness/ab_cs_accept_sweep.py`](harness/ab_cs_accept_sweep.py); importa
el [`harness/ab_nig_sweep.py`](harness/ab_nig_sweep.py) de #1265. Ejecútalo en partes y luego combina los resultados:

```bash
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.1,0.15,0.2 --out h0-part1.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.25,0.275,0.3 --out h0-part2.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.325,0.35,0.375 --out h0-part3.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.4,0.425,0.45 --out h0-part4.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.5,0.55,0.6 --out h0-part5.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.7,0.8,1.0 --out h0-part6.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 1.5,2.0 --out h0-part7.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 3.0,5.0 --out h0-part8.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.1,0.15,0.2,0.25,0.3,0.35 --out h1-part1.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.4,0.45,0.5,0.6,0.7,0.8 --out h1-part2.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h1 --runs 20000 \
    --ratios 1.0,1.5,2.0,3.0,5.0 --out h1-part3.json
python benchmarks/harness/ab_cs_accept_sweep.py report h0-part*.json h1-part*.json
python benchmarks/harness/ab_cs_accept_sweep.py check
```

Las partes, el informe combinado y el registro de ejecución están en
[`results/gate-1418-ab-cs-accept/`](results/gate-1418-ab-cs-accept/): `h0-part1.json` a
`h0-part8.json`, `h1-part1.json` a `h1-part3.json`, `report.txt` y `sweep.log`. Las semillas
están fijadas por modo y ratio, así que la misma versión de numpy reproduce los números anteriores.
