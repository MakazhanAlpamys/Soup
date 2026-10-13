# Puerta Qwen4-Exp PLE / oQ en M4 Max: aprobado parcial, prueba de producción detenida

Fecha: 2026-08-31

Hardware: MacBook Pro M4 Max, 128 GiB de memoria unificada, SSD externo con lectura
secuencial medida de aproximadamente 946 MB/s. Software: macOS, Python 3.12, PyTorch
2.13.0, nivel de entrenamiento de Transformers del entorno de desarrollo de Soup.

Este registro separa la corrección de la arquitectura de la viabilidad de entrenar
el checkpoint completo de producción. No debe interpretarse como una afirmación de
rendimiento, memoria máxima ni posibilidad de entrenar en producción.

## Checkpoint

La fuente local fue `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`, un paquete oMLX/oQ afín
de Qwen4:

- `model_type=qwen4_exp`, 48 capas decoder;
- 106.29 GB de safetensors de origen;
- matrices afines mixtas de 4/5/6/8 bits, con escalas y sesgos BF16;
- una tabla PLE de n-gramas empaquetada de 320,001,536 x 160, dividida en 128 partes;
- componentes de torre de visión y MTP fuera del CausalLM solo de texto.

## Puertas aprobadas

1. El mapa de shards de oQ a Transformers se comprobó contra el esqueleto meta
   real del decoder de texto: 1,167 tensores esperados, 1,167 tensores en caché,
   cero entradas faltantes, inesperadas o con formas incompatibles.
2. La caché reutilizable del decoder se completó con 48 capas y 251.47 GB en disco.
   La fuente PLE empaquetada permaneció externa y de solo lectura; la tabla densa
   proyectada de 102.40 GB no se copió a esa caché.
3. Los vectores afines oQ del decoder se compararon de forma independiente con MLX
   para 4, 5, 6 y 8 bits.
4. La pequeña puerta nativa de Qwen4 residente frente a streaming en CPU siguió
   dando resultados idénticos bit a bit para filas, logits, loss y gradientes LoRA.
   Ambas variantes MPS pasaron localmente:

   ```text
   2 passed, 21 deselected in 5.94s
   ```

5. El checkpoint de producción llegó a cargar el tokenizer, reutilizar la caché,
   hacer mmap de PLE, construir el modelo solo de texto, inyectar LoRA, preparar
   el conjunto de datos y mostrar `Training started!`, sin errores de clave, forma,
   tipo de datos ni enrutamiento de módulos.

## Resultado de la prueba de producción: DETENIDA / no validada

El ajuste SFT MPS BF16 de un paso no se completó. Se detuvo repetidamente durante
el primer forward en streaming, sin traceback de Python ni salida del adaptador.
La persona responsable del equipo indicó que esta carga de trabajo puede
inestabilizar y apagar la Mac, y pidió que no se intentara de nuevo el entrenamiento
con el checkpoint completo.

En el último punto de preparación, Soup informó:

```text
48 layers, 251.47 GB disk stream cache
2 x 5357 MB decoder buffers + 1 x 1271 MB large-layer slot
21,214,912 trainable / 176,943,899,555 total parameters
```

La memoria del sistema muestreada después de la detención estaba libre en un 96%,
sin actividad de intercambio; por lo tanto, este registro no atribuye la detención
a falta de RAM del equipo. No se generó ningún informe de diagnóstico de macOS ni
un traceback fatal de Python, así que la causa precisa de bajo nivel sigue sin
resolverse.

## Veredicto

- **APROBADO:** detección del checkpoint, decodificación afín oQ, mapeo de expertos
  fusionados, filas PLE externas de solo lectura, construcción de caché, paridad
  pequeña en CPU/MPS y preparación del checkpoint de producción.
- **NO VALIDADO:** un paso completo del optimizador para el checkpoint de producción
  de 176.9B parámetros en M4 Max.
- **PENDIENTE:** paridad BF16 de Qwen4 residente frente a streaming en CUDA.

No promocionar este registro como prueba de que el modelo completo se puede entrenar
en una Mac con 128 GiB. Los checkpoints Qwen4 más pequeños y la paridad BF16 en CUDA
siguen siendo los objetivos seguros para la siguiente validación.
