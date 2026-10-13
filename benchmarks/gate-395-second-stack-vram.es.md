# Registro del gate — #395: ajuste de VRAM por streaming en una segunda GPU/stack
> 🇪🇸 Traducción al español de [`gate-395-second-stack-vram.md`](gate-395-second-stack-vram.md). Ante cualquier duda, manda el original en inglés.

**Hardware.** NVIDIA A10G, 23.0 GB (AWS `g5.xlarge`), Ubuntu 22.04, driver
580.126.09. torch 2.13.0+cu130, transformers 5.16.1, trl 0.29.1, peft 0.20.0,
Python 3.10. Todas las cifras siguientes se midieron con esa configuración. Nada aquí
se midió en otro entorno, ni se extrapola a otro hardware.

Este es el «segundo GPU/stack» que solicita #395. El registro al que responde,
[`gate-v0.73.1-measured-vram-fit.md`](gate-v0.73.1-measured-vram-fit.md), se
obtuvo con una RTX 3050 Laptop 4 GB / Windows 11 / torch 2.5.1 / **transformers
4.57.6**. Aquí cambian tanto la GPU como el stack, y la diferencia de transformers es de una
versión principal.

---

## El resultado principal

**La subestimación no se reproduce.** Mismo protocolo: configuración real de `soup train`
más un paso, bf16, batch 1, `quantization: none`, LoRA r=8,
`stream_buffers: 2`, «pico real» es `torch.cuda.max_memory_allocated()`.

SmolLM2-135M:

| seq | predicho | pico real | pred/real | veredicto |
|---|---|---|---|---|
| 2048 | 1.596 GB | 1.365 GB | 1.169x | sobrepredice — seguro |
| 3072 | 2.345 GB | 2.017 GB | 1.163x | sobrepredice — seguro |
| 4096 | 3.094 GB | 2.658 GB | 1.164x | sobrepredice — seguro |
| 4352 | 3.281 GB | 2.818 GB | 1.164x | sobrepredice — seguro |
| 5120 | 3.842 GB | 3.303 GB | 1.163x | sobrepredice — seguro |
| 6144 | 4.591 GB | 3.943 GB | 1.164x | sobrepredice — seguro |

Qwen2.5-0.5B (3.1x el vocabulario, donde el término de logits domina más):

| seq | predicho | pico real | pred/real | veredicto |
|---|---|---|---|---|
| 2048 | 4.854 GB | 4.177 GB | 1.162x | sobrepredice — seguro |
| 4096 | 9.346 GB | 8.012 GB | 1.166x | sobrepredice — seguro |
| 5120 | 11.592 GB | 9.929 GB | 1.167x | sobrepredice — seguro |
| 6144 | 13.837 GB | 11.848 GB | 1.168x | sobrepredice — seguro |

Frente a la RTX 3050 con las mismas formas: 1.081x en 4352, **0.934x** en 5120,
**0.787x** en 6144. Aquí la razón se mantiene plana en 1.16x hasta 6144. En este stack
no hay cruce, y la propiedad de dirección se cumple en todas las mediciones.

---

## Qué es realmente la diferencia — medido, no deducido a la inversa

Una revisión anterior de este registro destacaba **11.83 B/elemento**, deducidos al
restar del pico real los términos no-logits modelados por la fórmula. Se retira esa
cifra. Era una de dos interpretaciones que los datos no permitían distinguir,
y el instrumento para resolverlo (`measure_logits_loss_bytes_per_element`) ya estaba
en el árbol y no se había ejecutado. Ahora se ejecutó en el equipo adecuado.

### 1. El término de pérdida es el 12 documentado, también en este stack

```
stack: torch 2.13.0+cu130 | NVIDIA A10G | trl 0.29.1 | transformers 5.16.1

measure_logits_loss_bytes_per_element(), 3 repeats at the default shape:
  12.0
  12.0
  12.0
  spread: 0.00e+00

across vocabularies:
  vocab   8192: 12.0
  vocab  32768: 12.0
  vocab  49152: 12.0
  vocab 151936: 12.01010951979781   (allocator rounding)
```

La afirmación de invariancia de `LOGITS_LOSS_BYTES_PER_ELEMENT` —*«idéntico byte a byte en torch
2.5.1 y 2.13.0»*— **se cumple aquí**. Es un tercer stack que coincide, no un contraejemplo.
El 11.83 nunca fue una medición de este término.

### 2. La copia bf16 retenida `+2` NO ESTÁ en este stack

Este es el hallazgo. El 14 enviado es `12` (pérdida) + `2` (un tensor adicional con forma de logits bf16,
cargado incondicionalmente, retenido por algo que #327 no ha identificado). Si esa copia se retuviera aquí,
el pico real completo tendría que superar el término de logits calculado con 14. No ocurre —en ninguna fila:

| modelo | seq | 14 x E | pico real |
|---|---|---|---|
| SmolLM2-135M | 2048 | 1.409 GB | **1.365 GB** |
| SmolLM2-135M | 4096 | 2.819 GB | **2.658 GB** |
| SmolLM2-135M | 5120 | 3.523 GB | **3.303 GB** |
| SmolLM2-135M | 6144 | 4.228 GB | **3.943 GB** |
| Qwen2.5-0.5B | 2048 | 4.356 GB | **4.177 GB** |
| Qwen2.5-0.5B | 4096 | 8.713 GB | **8.012 GB** |
| Qwen2.5-0.5B | 5120 | 10.891 GB | **9.929 GB** |
| Qwen2.5-0.5B | 6144 | 13.069 GB | **11.848 GB** |

`14 x E` por sí solo supera todo el pico medido, sin dejar espacio para los
otros términos, y mucho menos para la copia retenida. Diez de diez filas, dos modelos, contraste de
vocabulario de 3.1x. La retención que #327 no pudo explicar **no ocurre en este
stack**, una restricción para cualquier explicación.

### 3. Los términos no-logits están sobreestimados en ~15%

Manteniendo fijo el 12 medido y examinando el resto del modelo:

| modelo | seq | no-logits modelado | verdadero implícito | sobreestimación |
|---|---|---|---|---|
| SmolLM2-135M | 2048 | 0.187 GB | 0.157 GB | 1.190x |
| SmolLM2-135M | 4096 | 0.275 GB | 0.242 GB | 1.138x |
| SmolLM2-135M | 6144 | 0.363 GB | 0.319 GB | 1.138x |
| Qwen2.5-0.5B | 2048 | 0.498 GB | 0.443 GB | 1.124x |
| Qwen2.5-0.5B | 6144 | 0.769 GB | 0.646 GB | 1.189x |

**media 1.152x, desviación estándar poblacional 0.0249** sobre las diez filas.

Así se descompone la sobrepredicción de 1.16x: la ausencia de la copia `+2` (la parte mayor y escalada por vocabulario) más una sobreestimación de ~15% de los términos no-logits. No es «el multiplicador de logits es distinto aquí», que era lo que sugería el 11.83 retirado.

## Qué dice esto sobre el mecanismo que #395 no pudo identificar

Se plantea con cuidado, porque aquí solo se midió una parte.

Medido: en este stack el término es **plano** respecto a la longitud de secuencia. No se midió
aquí, pero los propios números del registro de la 3050 implican: 14 / 0.934 = 15.0 B/elemento
en seq 5120 y 14 / 0.787 = 17.8 en 6144. En ese stack, el multiplicador **creció
con la longitud de secuencia**; aquí no cambia.

Si esa interpretación es correcta, la razón por la que nunca encajó un coeficiente es estructural. Un
término que varía entre stacks y según la longitud de secuencia no es una constante, por lo que ninguna
constante puede garantizar que nunca se subestime en todos los stacks —que es el argumento que ya expone
`training.stream_vram_probe`. Este registro respalda esa decisión, no la contradice.

### La hipótesis principal no está respaldada aquí

#395 la formula así: «un cambio de una ruta eficiente en memoria a la ruta matemática materializaría
la matriz de atención completa y es la hipótesis principal que conviene descartar o confirmar primero».

Al forzar cada backend SDPA en las formas en cuestión (1 cabeza, 64 dim, bf16,
causal), asignación adicional frente al nivel base:

| seq | flash | eficiente en memoria | math |
|---|---|---|---|
| 2048 | 2.4 MB | 2.4 MB | 383.9 MB |
| 3072 | 3.7 MB | 3.5 MB | 831.5 MB |
| 4096 | 4.9 MB | 4.7 MB | 1463.8 MB |
| 5120 | 6.1 MB | 5.9 MB | 2279.7 MB |
| 6144 | 7.3 MB | 7.1 MB | 3267.4 MB |

La ruta matemática es tan costosa como requiere la hipótesis: 3.27 GB en
seq 6144 bastarían para agotar una tarjeta de 4 GB. **Pero SDPA no la selecciona aquí.**
Sin un contexto de backend explícito, el valor predeterminado asigna 2.4-7.3 MB en ese mismo
rango, igual que flash y eficiente en memoria, y muy lejos de la cifra de una cabeza `seq^2`
(8.4 MB en 2048, que aumenta a 75.5 MB en 6144).

Así que en este stack nunca se materializa la matriz de atención, y la hipótesis no explica nada de lo
medido aquí. No queda **refutada** para la RTX 3050: ese equipo ejecutaba Windows/WDDM con torch 2.5.1,
donde la cobertura de formas de flash-attention era más limitada y es plausible que hubiera una alternativa math.
Confirmarlo o descartarlo allí requiere ese hardware.

---

## Determinismo

Cercano, pero no idéntico bit a bit, según lo reportado en la 3050. Se repitieron dos formas en un
proceso nuevo:

| seq | ejecución 1 | ejecución 2 | delta |
|---|---|---|---|
| 4096 | 2.6578 GB | 2.6493 GB | 0.32% |
| 6144 | 3.9430 GB | 3.9439 GB | 0.02% |

Es irrelevante frente a una diferencia de 16%, pero se registra en lugar de ocultarlo mediante redondeo.

---

## Lo que este registro no establece

- **Cuál** de los cuatro búferes fp32/bf16 dejó de estar activo. Se infiere de los
  cálculos, no se observó en una instantánea del asignador.
- Nada sobre la RTX 3050. Las cifras 15.0 / 17.8 B/elemento anteriores son
  cálculos sobre las razones publicadas en ese registro, no mediciones nuevas.
- **Por qué** se retiene la copia `+2` en la RTX 3050 y no aquí. Este registro
  acota #327 en un stack; no explica la retención. torch, trl y transformers ya se habían cambiado
  por separado en el equipo H100 sin reproducirlo, así que la causa no es ninguno de esos tres.
- Si la sobreestimación de ~15% de los términos no-logits corresponde a un solo término o varios. Es plana
  en un contraste de vocabulario de 3.1x y un rango de secuencia de 3x, lo cual apunta a un
  efecto de fracción fija y no a deriva, pero este registro no lo descompone.
- El escalado de batch en secuencias largas. Todas las filas aquí son batch 1.

---

## Cómo reproducirlo

`benchmarks/harness/issue395_second_stack_vram.py`, con una versión publicada de Soup:

```
python benchmarks/harness/issue395_second_stack_vram.py            # the sweep
python benchmarks/harness/issue395_second_stack_vram.py --measure  # section 1
python benchmarks/harness/issue395_second_stack_vram.py --sdpa     # section 4
```

Soup 0.73.3, A10G 23 GB (AWS `g5.xlarge`), Ubuntu 22.04, torch 2.13.0+cu130,
transformers 5.16.1, trl 0.29.1, peft 0.20.0, Python 3.10. Requiere
`jinja2>=3.1.0`: el extra `[dev]` puede resolver una versión inferior y la ruta de plantilla de chat
falla sin ese paquete. Basta con una tarjeta de 24 GB; el brazo más grande alcanza un pico de
11.85 GB.

## Nota sobre el método

El primer intento de esta prueba estaba mal y vale la pena registrarlo porque el fallo es silencioso. `data.max_length` **trunca**; no rellena. Las filas que
se tokenizan con poca longitud se ejecutan con su propia longitud, sin importar el máximo configurado, así que el «pico real» deja de responder a `seq` mientras la predicción sigue
creciendo; parece una sobrepredicción grande y limpia que aumenta con la
secuencia, pero es completamente un artefacto del arnés.

El arnés corregido excede en 3x el contenido de la fila para que el truncamiento llegue al
objetivo, y lee la longitud real de la secuencia del batch agrupado
(`input_ids.shape[-1]`) en su propia columna. Todas las filas anteriores se comprobaron así:
la longitud de secuencia solicitada y la realizada coinciden en las diez.
