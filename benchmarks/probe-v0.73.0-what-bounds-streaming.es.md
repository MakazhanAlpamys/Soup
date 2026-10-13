<!--
Registro de mediciones de trabajo, publicado literalmente.

Al igual que los registros de gate que lo acompañan, este es el log que se llevó
mientras ocurría el trabajo, no un informe elaborado después: la instrumentación
fallida, la premisa corregida y las cifras descartadas permanecen, en el orden en
que ocurrieron.

Hardware: RTX 3050 Laptop (4 GB, 4.29 GB utilizables), 16.9 GB de RAM del host, NVMe,
Windows 11. torch 2.5.1+cu121 · bitsandbytes 0.49.2 · transformers 4.57.6 ·
peft 0.18.1 · Python 3.10.8. Soup v0.73.0 (árbol de trabajo en 57c18e5).
-->

# Qué limita realmente el streaming por capas y ¿puede Cut Cross-Entropy superarlo?
> 🇪🇸 Traducción al español de [`probe-v0.73.0-what-bounds-streaming.md`](probe-v0.73.0-what-bounds-streaming.md). Ante cualquier duda, manda el original en inglés.

**Estado: TAREA 1 RESPONDIDA: la ejecución está limitada por cómputo, no por transferencia.
TAREA 2 RESPONDIDA para el kernel, BLOQUEADA para la bandera distribuida: Cut Cross-Entropy
triplica el microbatch utilizable (1 -> 3) con +9.6% de rendimiento, pero el propio
`training.use_cut_ce` de Soup no puede activarse en ninguna pila instalada con pip (tres
bloqueos independientes, todos medidos en §7).**

Dos preguntas, planteadas en este orden porque la segunda sólo tiene sentido después de la
primera:

1. Soup afirma públicamente que el streaming por capas está **limitado por la
   transferencia del host al dispositivo**. Esa afirmación nunca se había medido. ¿Es cierta?
2. Cut Cross-Entropy se distribuye desde v0.28.0 y el streaming por capas desde
   v0.72.0. Nunca se habían activado ambos a la vez. ¿Qué ocurre?

Convención de unidades: **GB decimales**, igual que en los registros de gate.

---

## 0. La premisa era errónea antes de la primera medición, y en la dirección segura

El encargo que inició este trabajo calculó la tasa de transferencia lograda como
**~6.7 GB/s** —«aproximadamente la mitad del límite Gen4 x8»— y razonó que, si la mitad del
bus ya está en uso, es plausible que el bus sea la restricción.

Esa cifra se reproduce exactamente si se supone que el paso es de **128 tokens**:

```
7.2 GB / (128 tok / 119.6 tok/s) = 6.73 GB/s
```

La fila publicada de 119.6 tok/s corresponde a **S=512**, no 128. Con la longitud
real de secuencia, el mismo cálculo da:

```
6.864 GB per step / 4.281 s = 1.60 GB/s
```

Una segunda corrección, menor: el conteo ingenuo es de 64 cargas de capa por paso
(32 en forward + 32 al recalcular el backward). El conteo medido es **61.0**: el
prefetcher omite una carga cuando el slot ya contiene la capa que está por necesitar.
Así, el volumen transferido es 61 x 112.529 MB = **6.864 GB**, no 7.2 GB.

Esto no debilita la conclusión del encargo; la fortalece: la ejecución usa
**~20% de lo que esta máquina realmente puede hacer**, no ~50%. Se registra porque
una premisa que llega hasta la respuesta es la forma en que se publica una respuesta errónea.

---

## 1. Lo que esta máquina realmente puede hacer (T1a)

`scratchpad/t1_pcie.py`. Buffers del host fijados, copiados al dispositivo en un flujo
secundario, medidos con eventos CUDA, mejor de 30. La variante «patrón real» asigna los
30 tensores exactos de un fragmento de capa Llama-3.1-8B NF4, para poder separar el
sobrecosto por llamada del ancho de banda.

| # | Qué | GB/s | Nota |
|---|---|---|---|
| A | patrón real de 30 tensores de capa, **fijados** | **7.77** | mejor; mediana 7.50 |
| B | un bloque contiguo del mismo tamaño, 112.529 MB, fijado | 7.42 | |
| C | el mismo patrón de 30 tensores, **paginable** | 4.91 | ruta alternativa |
| D | bloque de 512 MB, fijado | 6.86 | |

Enlace negociado, comprobado inactivo y de nuevo **bajo carga**:
`PCIe Generation Current: 4`, `Link Width Current: 8x` (máx. 16x), es decir, Gen4 x8,
~15.75 GB/s teóricos.

**Hallazgo 1a: dividir una capa en 30 tensores no cuesta nada.** A no es más lenta que B;
es marginalmente más rápida, dentro del ruido. No hay sobrecosto por lanzamiento de copia
que valga la pena eliminar, así que «agrupar el fragmento en un solo bloque» no es una
optimización. Valía la pena comprobarlo antes de darlo por sentado.

**Hallazgo 1b: la tasa alcanzable es ~7.8 GB/s en reposo y la ejecución real logra más.**
Durante el entrenamiento se midieron **9.38 GB/s** en el flujo de copia (§3), por encima
de la prueba en reposo. La prueba inactiva subestima la tasa porque el enlace queda en un
estado de menor consumo entre ráfagas sincronizadas. Por tanto, aquí «alcanzable» significa
7.8–9.4 GB/s según el estado del enlace, frente a 15.75 teóricos.

---

## 2. El control: ¿el arnés reproduce la fila publicada? (T1b)

`scratchpad/t1_step.py --mode baseline`. Modelo creado mediante la ruta **distribuida**
(`read_shard_index` + `build_streamed_model`), NF4 + cuantización doble, LoRA r=8 en q/k/v/o,
batch 1, S=512, `PagedAdamW8bit`, 2 buffers, nivel RAM, fijado.

| | publicado v0.72.2 paso-6 | medido aquí |
|---|---|---|
| tok/s | 119.6 | **122.1 – 122.5** |
| VRAM máxima | 3.32 GB | **3.448 GB** |
| almacenamiento fijado | 3.60 GB | **3.601 GB** |
| reloj SM | 952 MHz | 952–960 MHz |

El almacenamiento coincide hasta tres decimales. El rendimiento es 2% mayor y la VRAM
máxima 0.13 GB mayor; ambos se explican porque la configuración de LoRA no quedó registrada
en la fila publicada (este arnés usa r=8 en cuatro proyecciones). **El control se sostiene**;
todo lo que sigue se mide en una ejecución que reproduce la publicada.

---

## 3. En qué se invierte realmente el paso (T1c)

Instrumentación con eventos CUDA, `--mode events`. Importan dos cifras y no son la misma:

- **tiempo de copia**: cuánto tardan las transferencias en su propio flujo. Puede solaparse.
- **espera**: cuánto tiempo permanece bloqueado el flujo de *cómputo* por el evento de copia
  de una capa. Esta es la única parte de la transferencia que **no** queda oculta.

Medido al registrar un evento en el flujo de cómputo inmediatamente antes y después de
`wait_event`, de modo que la diferencia entre ambos es exactamente el intervalo bloqueado.

| magnitud | por paso | proporción del paso de 4.190 s |
|---|---|---|
| tiempo del paso | 4.190 s | 100% |
| bytes transferidos | 6.864 GB | |
| tiempo de copia en el flujo de prefetch | 0.732 s | 17.5% |
| **flujo de cómputo detenido por una copia** | **0.0084 s** | **0.20%** |
| tasa del flujo de copia | 9.38 GB/s | |
| tasa promedio durante el paso | 1.638 GB/s | |

tok/s con instrumentación: **122.20**, frente a 122.14 sin instrumentación: la medición no
cuesta nada.

**Hallazgo 3: las transferencias quedan prácticamente ocultas a la perfección.** El flujo de
cómputo espera 8.4 ms de los 4190 ms. El doble búfer está haciendo su trabajo.

---

## 4. Ablaciones intercaladas (T1d)

Un perfil muestra a qué se atribuye el tiempo. Una ablación muestra qué se gana realmente
al eliminar un término, que es una afirmación más sólida. Cuatro variantes, activadas mediante
una bandera **dentro de un mismo proceso y sobre un mismo modelo**, e **intercaladas** A/B/C/D
por ronda para que la deriva monotónica del reloj no favorezca a ninguna: el método que usó
el gate de acumulación v0.72.3 por el mismo motivo.

Ambas variantes de ablación producen **matemáticas basura** y sólo sirven para medir tiempos:
`nocopy` deja bytes obsoletos en los buffers, `nodequant` multiplica por un peso almacenado en
caché que vale cero.

Llama-3.1-8B NF4, batch 1, S=512, 8 pasos x 2 rondas, todo a **960 MHz**:

| variante | paso | tok/s | frente a la base |
|---|---|---|---|
| A base | 4.211 s | 121.6 | — |
| B **sin transferencias H2D** | 4.150 s | 123.4 | **−1.44%** |
| C sin des-cuantización NF4 | 3.799 s | 134.9 | −9.80% |
| D ninguna de las dos | 3.743 s | 136.8 | −11.3% |

Variación entre rondas: 1.4% (A), 0.35% (B), 0.31% (C).

**Hallazgo 4: eliminar todos los bytes de tráfico del host al dispositivo acelera el paso 1.4%.**
Se eliminan 6.864 GB por paso para ganar 1.4%. Con esta configuración, el streaming por capas
no está limitado por la transferencia, y ningún ancho de banda PCIe cambiaría la cifra principal.

**Hallazgo 4b: al eliminar tanto la transferencia como la des-cuantización, queda el 88.7% del paso.**
Lo que limita esta ejecución no es ninguna de las dos cosas que añade el streaming.

---

## 5. Entonces, ¿qué lo limita? (T1e)

### 5a. El paso se separa claramente en un costo fijo y uno por token

Barrido de secuencia con batch 1, una creación, 8 pasos por punto:

| tokens | paso | tok/s | VRAM máxima |
|---|---|---|---|
| 16 | 0.983 s | 16.3 | 2.860 GB |
| 32 | 1.029 s | 31.1 | 2.869 GB |
| 64 | 1.043 s | 61.4 | 2.888 GB |
| 128 | 1.417 s | 90.3 | 2.931 GB |
| 256 | 2.269 s | 112.8 | 3.040 GB |
| 384 | 3.186 s | 120.5 | 3.183 GB |
| 512 | 4.179 s | 122.5 | 3.448 GB |

Mínimos cuadrados entre 128–512:

```
step(S) = 0.462 s + 7.190 ms/token
```

En S=512 eso equivale a **11.1% fijo y 88.1% proporcional a los tokens**. La parte fija
es donde están las transferencias y el sobrecosto de lanzamiento; la parte proporcional es
el cálculo propio del modelo.

**El ajuste se limita deliberadamente a 128–512, y los tres puntos inferiores explican por qué.**
Extrapolado a S=16, predice 0.577 s, mientras que se midieron 0.983 s, y S=16/32/64 se
mantienen casi planos en ~1.0 s: así que el término «fijo» no es una constante en todo el
rango. No es un mal ajuste; es la física de §5c: el sobrecosto de CPU por capa para los
lanzamientos y Python es **del lado de la CPU**, por lo que en secuencias largas la mayor
parte se emite mientras la GPU sigue ocupada y desaparece del tiempo de pared, mientras que
en secuencias cortas la CPU no puede adelantarse y el mismo sobrecosto queda completamente
expuesto. Por tanto, 0.462 s es la parte que sigue siendo inevitablemente serial en S=512,
y ~1.0 s en S=16 es el mismo sobrecosto sin nada detrás de lo cual ocultarse. Ambas cifras
son reales: son las vistas oculta y expuesta de un mismo costo, por eso §5a y §5c no se
contradicen.

### 5b. La parte proporcional es trabajo GEMM real, al 71% del límite de esta tarjeta

Medido **en la misma sesión y al mismo reloj**, porque el gate v0.72.2 estableció que citar
una fracción del límite entre sesiones no tiene sentido en esta tarjeta (13% de variación del
reloj boost). `scratchpad/t1_ceiling.py` ejecuta primero el paso en streaming, luego mide el
límite e informa el reloj de ambos.

| forma | M x K x N | TFLOPS |
|---|---|---|
| q_proj / o_proj | 512 x 4096 x 4096 | 7.67 |
| k_proj / v_proj | 512 x 4096 x 1024 | 7.51 |
| gate / up_proj | 512 x 4096 x 14336 | 7.47 |
| down_proj | 512 x 14336 x 4096 | 7.67 |
| **ponderado por FLOP** | | **7.55** @ 952 MHz |

Paso en streaming en la misma sesión: 4.182 s -> 122.4 tok/s @ 952 MHz, es decir,
22.52 TFLOP/paso según el conteo C=6 del gate -> **5.38 TFLOPS efectivos =
71.3% del límite ajustado a las formas y medido en la misma sesión.**

Con los dos términos específicos del streaming eliminados (variante D), el mismo cálculo da
6.02 TFLOPS = **79.7% del límite**.

### 5c. Hay un sobrecosto por capa, y sólo se ve cuando el cómputo es pequeño

Las mismas cuatro variantes en **S=16**, donde el cálculo es insignificante:

| variante | paso |
|---|---|
| A base | 0.974 – 1.187 s |
| B sin transferencias | 0.914 – 0.999 s |
| C sin des-cuantización | 0.814 – 0.868 s |
| D ninguna de las dos | **0.626 – 0.716 s** |

Con 16 tokens, el trabajo GEMM es ~0.09 s, pero la variante D aún cuesta ~0.67 s. Ese
remanente —aproximadamente 10 ms por visita de capa a lo largo de 64 visitas— es el
sobrecosto **de lanzamiento y Python** por capa: 30 tensores agrupados, una vista
`Params4bit` reconstruida por peso, una `functional_call` y ~50 lanzamientos de kernel, todo
repetido 64 veces.

En S=512 esto queda oculto detrás del cómputo y no cuesta nada. Por eso la curva de tok/s
del barrido es pronunciada por debajo de ~128 tokens: con secuencias cortas, la ejecución
**sí** está limitada por el costo fijo, y ahí las transferencias importan.

### 5d. «@100% util» nunca estuvo en tensión con nada

Muestreado bajo carga: `utilization.gpu` **100%**, utilización del controlador de memoria
44–51%, reloj fijo en 952 MHz, enlace estable en Gen4 x8. `utilization.gpu` es la fracción
del tiempo en que hay al menos un kernel residente; no es ocupación ni eficiencia. 100% de
utilización con 71% del límite GEMM es la señal habitual de un bucle eager limitado por
cómputo, y concuerda con que las transferencias estén ocultas; no lo contradice.

---

## 6. Tarea 1, respondida en una frase

**Estamos limitados por cómputo, no por transferencia: el paso en streaming se ejecuta al
71.3% del límite GEMM bf16 de esta tarjeta medido en la misma sesión; eliminar todos los bytes
del host al dispositivo aporta 1.4%, y el flujo de cómputo pasa 0.20% del paso esperando una
copia. El mayor sobrecosto específico del streaming no es el bus, sino la des-cuantización
NF4 por capa, con 9.8%, seguida de ~10 ms por visita de capa en lanzamientos de kernel y
overhead de Python, que sólo se convierte en la restricción principal por debajo de ~128
tokens por paso.**

### Qué implica esto para la afirmación pública

«Limitado por la transferencia del host al dispositivo» es **incorrecto en la configuración
publicada** y no debería repetirse. Lo que realmente demuestra el resultado de H100 es la
afirmación más moderada, pero aún interesante, hacia la que razonaba el encargo: el cuello de
botella es **común a ambas máquinas y no es el bus**. En la H100, eso concuerda con el mismo
hallazgo de aquí: un bucle eager, por capa, de des-cuantización seguida de GEMM no está
alimentado por la interconexión.

Dos salvedades honestas:

- La afirmación sobre la transferencia **sí** es cierta con secuencias cortas o pocos tokens
  por paso, cuando los 6.864 GB fijos dominan. El punto de cruce está en el barrido de §5a.
- Nada de esto afirma que la ejecución en H100 estuviera limitada por cómputo *por el mismo
  motivo*. Esa máquina ya no está disponible y este registro no afirma nada sobre ella.

### Qué lo haría realmente más rápido, según el margen medido

1. **La brecha de 28.7% hasta el límite GEMM** (variante D al 79.7%): atención,
   operaciones elemento a elemento y sobrecosto de lanzamiento. Es el mayor conjunto individual.
2. **La des-cuantización NF4, 9.8%.** Cada peso en streaming se des-cuantiza a un tensor bf16
   denso en cada visita: 448 llamadas por paso, ~437 MB de escrituras en VRAM por visita de
   capa. Este es el costo de la reparación #331, que sacó deliberadamente el peso de
   `MatMul4Bit` para que el checkpointing de gradientes pudiera verlo. Una operación fusionada
   de des-cuantización y multiplicación, que siga siendo visible para el checkpointing, es el
   objetivo obvio y el **único** elemento de esta lista específico del streaming.
3. **Transferencias: 1.4%.** No vale la pena tocarlas.

---

# TAREA 2 — Cut Cross-Entropy sobre el streaming por capas

## 7. La propia bandera `use_cut_ce` de Soup no puede activarse. Tres bloqueos, todos medidos

El esquema permite la combinación: esto se analiza correctamente y ningún validador cruzado
lo rechaza:

```yaml
training:
  stream_layers: true
  quantization: 4bit
  use_cut_ce: true
```

`training.use_cut_ce` enruta a `utils/cut_ce.py::apply_cut_ce`, que ejecuta
`from cut_cross_entropy.transformers import cce_patch`. Todos los intentos de lograr que
esa importación funcionara en esta pila fallaron, cada vez por un motivo *distinto*:

| # | Qué se intentó | Resultado |
|---|---|---|
| 1 | `pip install cut-cross-entropy` (lo que indica la docstring del módulo) | La rueda **25.1.1 no incluye ningún submódulo `transformers`**: 31 archivos, ninguno coincide con `transformer`. La importación nunca puede funcionar. |
| 2 | `pip install cut-cross-entropy[transformers]` | El extra sólo añade una *dependencia* `transformers>=4.44.2`; no añade código. Los mismos 31 archivos. |
| 3 | `pip install git+https://github.com/apple/ml-cross-entropy` (25.9.3, que **sí** incluye el submódulo) | `ImportError: cannot import name '_CONFIG_FOR_DOC' from transformers.models.gemma2.modeling_gemma2`. Al adaptar ese símbolo apareció el siguiente: `GEMMA2_INPUTS_DOCSTRING`. |
| 4 | (con 25.9.3) llamar directamente al kernel | `PackageNotFoundError: No package metadata was found for triton`: `is_triton_3_2()` busca la distribución `triton`, pero Windows instala `triton_windows`, que proporciona el *módulo* `triton` 3.1.0 bajo otro nombre de *distribución*. |

El bloqueo 3 es el decisivo y **no es específico de Windows**: la integración de CCE está
escrita para una generación de transformers que ya no exporta esos símbolos privados, y
`patch.py` importa anticipadamente el adaptador gemma2, así que un modelo **llama** tropieza
con una incompatibilidad de **gemma2**. Soup fija `transformers<5.0.0`, por lo que 4.57.6
está plenamente dentro del rango admitido.

**Cómo se presenta esto al usuario.** `apply_cut_ce` captura
`(ImportError, AttributeError, NotImplementedError)` y devuelve `False`. El trainer sí
comprueba el valor devuelto e imprime una línea amarilla; así que **no** es una operación vacía
silenciosa, a diferencia del defecto de Liger (#78), donde la bandera se establecía pero nunca
llegaba a `TrainingArguments`. Pero el mensaje no menciona ninguna de las causas:

> `Cut Cross-Entropy: no matching architecture found or cut_cross_entropy not installed`

Ambas partes son falsas aquí. El paquete **sí** está instalado y llama **sí** es una arquitectura
compatible; lo que falló es una importación upstream. Al leer esa línea, el usuario volvería a
instalar un paquete que ya tiene.

**Consecuencia: `training.use_cut_ce` está en el árbol desde v0.28.0 y, en cualquier pila
instalada con pip, nunca ha podido hacer nada.** Ninguna medición de este registro lo utiliza.

### Qué se midió en su lugar y cómo interpretar los resultados

Aquí CCE se conecta **manualmente**, como muestra el propio README de CCE:

```python
hidden = decoder(input_ids=ids).last_hidden_state
loss = linear_cross_entropy(hidden[:, :-1], lm_head.weight, labels[:, 1:])
```

con **cut-cross-entropy 25.1.1** + **triton-windows 3.1.0.post17**. Ambas variantes comparten
el mismo forward del decoder y sólo difieren en la loss. Por tanto, las cifras siguientes miden
**el kernel**, no la bandera de Soup.

---

## 8. Corrección: y obliga a cambiar lo que aquí puede significar «correcto»

La referencia debe usar **el mismo kernel de loss**: CCE no es bit a bit idéntica a torch CE,
así que comparar streaming+CCE con residente+CE normal produciría divergencias por un motivo
ajeno al streaming. Medido en el mismo modelo en streaming, CCE frente a `ForCausalLMLoss`:
**7.901e-03**. La salvedad es real, no teórica.

Modelo **SmolLM2-135M NF4**: el tamaño que usó GATE 1 del registro v0.72.2, y la única clase
para la que una referencia NF4 residente realmente cabe en esta tarjeta (ese registro midió
que NF4 residente **3B** se desbordaba a 6.07 GB de memoria del host en una tarjeta de 4 GB,
así que aquí no hay una referencia residente de 8B con ningún batch). Batch 1, S=256,
240 tensores del adaptador sincronizados de streaming a residente y **verificados como
idénticos antes de la comparación**, en lugar de asumirlo.

| variante | loss, streaming frente a residente | diferencia abs. máxima de gradiente, 240 tensores |
|---|---|---|
| **control** — ForCausalLMLoss | **0.000e+00** | **0.000e+00** |
| **prueba** — Cut Cross-Entropy | **0.000e+00** | 9.328e-03 / 1.178e-02 (dos ejecuciones) |

El control es lo que permite interpretar la fila de prueba: con la loss habitual este arnés ve
el streaming como bit a bit idéntico en ambas mitades, así que *sí* puede detectar igualdad.

### El control que determina cómo leer la fila de CCE

Ejecutar **dos veces la misma loss en el mismo modelo**, sin streaming en absoluto en la
comparación:

| repetición, mismo modelo, misma entrada | diferencia abs. máxima de gradiente |
|---|---|
| streaming, ForCausalLMLoss | **0.000e+00** |
| residente, ForCausalLMLoss | **0.000e+00** |
| streaming, **CCE** | 8.858e-03 |
| residente, **CCE** | 9.461e-03 |

**Hallazgo 8: la diferencia de gradiente entre streaming y residente con CCE (0.93–1.18e-02)
queda dentro del ruido entre ejecuciones de CCE en cualquiera de los modelos (0.89–0.95e-02).**
Se debe al carácter no determinista de CCE —su backward acumula el gradiente de pesos con
operaciones atómicas—, no a un defecto del streaming. El forward no se ve afectado: la loss es
bit a bit idéntica en streaming y residente, y la ruta HF se repite exactamente con 0.0 en ambos
modelos, así que el ruido es específico del backward de CCE.

**Hallazgo 8b: esta parte sí tiene un costo.** La afirmación central de Soup, y el criterio que
impone su CI, es la exactitud bit a bit frente a una referencia residente. Activar CCE **destruye
ese criterio**, no por el streaming: una ejecución residente ni siquiera es reproducible frente
a *sí misma*. Cualquier versión futura que distribuya `use_cut_ce` con streaming debe sustituir
«bit a bit idéntico» por «dentro del piso de ruido medido del kernel de loss» y medir ese piso
por modelo y forma. Es una garantía estrictamente más débil que la que ofrece actualmente el
proyecto y debe ser una decisión deliberada, no un efecto secundario de activar una bandera.

---

## 9. VRAM máxima y barrido de microbatch

Llama-3.1-8B NF4, en streaming, S=512, LoRA r=8 en q/k/v/o, `PagedAdamW8bit`, 2 buffers,
4 pasos medidos después de 2 de calentamiento. `hf` = `ForCausalLMLoss`, `cce` =
`linear_cross_entropy` conectado manualmente. Total de la tarjeta: **4.294 GB**.

| variante | microbatch | tokens/paso | tok/s | VRAM máxima | veredicto |
|---|---|---|---|---|---|
| hf | 1 | 512 | 121.52 | **3.448 GB** | cabe |
| hf | 2 | 1024 | 125.14 | 4.512 GB | **desbordamiento** (por encima de la tarjeta) |
| hf | 3 | 1536 | 40.95 | 5.573 GB | **desbordamiento**, caída de 3.0x |
| hf | 4 | 2048 | — | — | `CUDA error: out of memory` grave |
| cce | 1 | 512 | 120.52 | **3.144 GB** | cabe |
| cce | 2 | 1024 | 129.01 | 3.463 GB | cabe |
| cce | 3 | 1536 | **132.03** | 3.782 GB | cabe |
| cce | 4 | 2048 | 19.20 | 4.100 GB | **desbordamiento**, caída de 6.9x |

`cce b=3` se midió dos veces, en procesos distintos: 131.69 y 132.03
tok/s, VRAM máxima de **3.782 GB en ambas**.

### 9a. El dato principal: memoria, no velocidad

- **Con el mismo batch, CCE ahorra 0.304 GB (−8.8%) en b=1.** Esa es toda la respuesta a
  «medir la VRAM máxima en ambos casos» y, por sí sola, no es muy interesante.
- **La pendiente es lo interesante, y es perfectamente lineal.** La VRAM máxima aumenta
  **1.063 GB por cada microbatch adicional** sin CCE (1.064, 1.061) y **0.319 GB** con CCE
  (0.319, 0.319): una reducción de **3.33x** en el costo marginal de un microbatch.
- La diferencia, **0.744 GB por 512 tokens**, es el pipeline de entropía cruzada:
  **11.32 bytes por elemento de logit**. La comprobación previa de VRAM máxima de Soup cobra
  `LOGITS_BYTES_PER_ELEMENT = 14`. Es una medición independiente de la misma cantidad desde
otra perspectiva, y queda **19% por debajo** de la constante distribuida: un dato para el
  #327 abierto («la comprobación previa sobreestima»), medido aquí como costo marginal en
  lugar de ajustarse dentro de un total.
- **El microbatch utilizable pasa de 1 -> 3.** Sin CCE, esta configuración sólo cabe con batch
  1 en esta tarjeta; con CCE, cabe con 3.

### 9b. Dónde se aplana el crecimiento lineal, y no es donde se acaba la memoria

tok/s frente al microbatch, variante CCE: **120.52 -> 129.01 -> 132.03**, es decir, +7.0%
y luego +2.3%. El barrido de secuencia de la Tarea 1 ajustó independientemente
`step(S) = 0.462 s + 7.190 ms/token`, cuya asíntota es **139.1 tok/s**. Frente a ella:

| microbatch | tok/s | % de la asíntota de 139.1 |
|---|---|---|
| 1 | 120.52 | 86.6% |
| 2 | 129.01 | 92.7% |
| 3 | 132.03 | 94.9% |

**La curva se aplana entre los microbatches 2 y 3, antes de agotar la memoria y no por causa
de ella.** Este es el hallazgo de la Tarea 1 visto desde el otro lado: la ejecución ya está
limitada por cómputo con batch 1, así que aumentar el batch no puede mejorar el rendimiento;
sólo puede amortizar el costo fijo de 0.462 s/paso entre más tokens. Ese costo fijo equivale
al 11% del paso en b=1 y a ~4% en b=3; una vez amortizado, no queda nada más que recuperar
con el batching.

Así que el resumen honesto de la Tarea 2 es: **CCE aporta margen de memoria (batch marginal
3.3x más barato, microbatch utilizable de 1 -> 3), y ese margen se convierte en +9.6% de
rendimiento, tras lo cual se detiene.** Quien espere que CCE libere una gran aceleración en
esta configuración debería leer §6 primero.

### 9c. Tres cosas salieron mal al medir esto; todas quedan registradas

1. **El primer barrido duró ~20 minutos y no dejó nada.** El proceso murió (código de salida
   4) con un log vacío; los resultados sólo se escribían al final. Se corrigió guardando los
   datos después de cada punto, única razón por la que sobrevivieron al segundo fallo.
2. **Un `CUDA error: out of memory` grave no es `torch.cuda.OutOfMemoryError` del asignador.**
   Envenena el contexto CUDA, por lo que todas las llamadas posteriores del proceso fallan;
   la fila `hf b=4` terminó la ejecución y hubo que volver a medir los puntos restantes en un
   proceso nuevo. Un barrido que aumenta el tamaño del batch más allá del punto de rechazo debe
   prever que tendrá que reiniciarse, no que podrá continuar.
3. **Mi propio umbral de desbordamiento era incorrecto, dos veces.** `peak <= card total (4.294 GB)`
   habría clasificado `cce b=4` (4.100 GB) como válido, pero su rendimiento cayó 6.9x.
   `peak <= free-at-idle (3.460 GB)` habría clasificado `cce b=2` y `b=3` como desbordamientos,
   aunque se ejecutaron a 129 y 132 tok/s sin ninguna caída. Ninguno de los dos umbrales predice
   el comportamiento de esta máquina.

   Lo que realmente sustentan los datos: **un pico por encima del total de la tarjeta demuestra
   un desbordamiento** (`hf` b=2 y b=3; `cce` nunca lo supera); **un pico inferior no demuestra
   que quepa**, y la única señal fiable es conductual: una caída del rendimiento frente al punto
   vecino. Observa que `hf b=2` se desbordó por 0.2 GB y fue *más rápido* (125.14 frente a
   121.52 tok/s), así que la ausencia de caída tampoco demuestra que quepa. Es la misma
   propiedad de Windows/WDDM registrada por el gate v0.72.3 cuando midió 9.27 GB asignados en
   esta tarjeta de 4.29 GB sin que nada lo señalara.

---

## 10. Lo que NO se midió

Se explicita, en lugar de dejarlo implícito, porque varias de estas cosas parecerían estar
incluidas en el alcance:

- **`training.use_cut_ce` de Soup de extremo a extremo mediante `soup train`.** No puede
  activarse (§7). Todas las cifras de CCE aquí vienen de una llamada manual al kernel y ninguna
  demuestra que funcione la bandera distribuida.
- **Una referencia residente 8B de cualquier tipo.** No cabe en esta tarjeta, así que la
  comprobación de corrección de §8 se ejecuta con 135M. Nada aquí vuelve a verificar la
  corrección del streaming en 8B; el registro de H100 ya lo hace y este registro no amplía
  esa verificación.
- **Nsight Systems.** Se usó `torch.profiler` en su lugar. Su capa CUPTI **descartó
  silenciosamente todos los eventos del dispositivo** durante una ventana de 28 segundos y
  5 pasos: un trace de 82 MB con 340,425 eventos, ninguno de ellos un kernel. Funciona con un
  solo paso. El profiler también aumenta el paso ~31% (5.48 s frente a 4.19 s), así que §3–§5
  se apoyan en eventos CUDA y ablaciones, y el profiler sólo se usa para el desglose relativo
  de operaciones.
- **El binario propio `bandwidthTest` de CUDA.** No está instalado en esta máquina; §1 usa su
  equivalente en torch (buffers del host fijados, medidos con eventos CUDA). No se garantiza que
  ambos coincidan.
- **Si la ejecución en H100 estaba limitada por cómputo por el mismo motivo.** Esa máquina ya
  no está disponible. Este registro no afirma nada al respecto.
- **Streaming bf16 (sin NF4) con la misma descomposición**, ni **secuencias por encima de 512**
  ni el nivel de disco.
- **Si una operación fusionada de des-cuantización y multiplicación recuperaría realmente el
  9.8%.** §4 mide cuánto vale eliminar la des-cuantización, no cuánto costaría un kernel real
  fusionado.
- **CCE con otras longitudes de secuencia o vocabularios.** Los 11.32 bytes/elemento de §9a
  corresponden a una forma (S=512, vocab 128256), a partir de dos segmentos de pendiente.
- **cce b=5 y b=6.** Se detuvieron deliberadamente: b=4 ya se había desbordado, así que esos
  puntos habrían medido paginación de Windows durante ~30 minutos.

---

## 11. Seguimientos que justifica este registro

1. **`training.use_cut_ce` nace inoperante**: hay tres bloqueos en §7 y el texto de advertencia
   no menciona ninguno. Hay que fijar una integración funcional, incluir las ~40 líneas del
   parche o rechazar claramente la bandera al analizar la configuración, en vez de imprimir una
   línea amarilla a mitad de la ejecución que envía al usuario a reinstalar un paquete que ya
   tiene.
2. **La afirmación pública «limitado por la transferencia del host al dispositivo» es
   incorrecta** en la configuración publicada y hay que corregirla dondequiera que aparezca (§6).
3. **La des-cuantización NF4 vale 9.8%** y es el único término específico del streaming con
   margen real (§4).
4. **#327 obtiene una medición independiente**: el costo marginal del pipeline de CE aquí es
   11.32 B/elemento frente a la constante distribuida de 14 (§9a).
5. **Si alguna vez se distribuye CCE con streaming, debe cambiar el gate de exactitud bit a bit**:
   CCE no es reproducible ni consigo misma, en residente o streaming (§8b).
