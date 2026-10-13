# Flujo de trabajo con datos sintéticos (de extremo a extremo)

> 🇪🇸 Traducción al español de [`synthetic_workflow.md`](synthetic_workflow.md). Ante cualquier duda, manda el original en inglés.

Genera, filtra, puntúa y entrena con datos sintéticos — todo con `soup`. Se combina
con la receta incluida `synthetic_workflow.yaml`.

## 1. Generar

Levanta un modelo local de Ollama y pídele 200 pares de instrucción/respuesta
sobre un tema:

```bash
soup data generate \
  --prompt "You write concise Python instruction/response pairs for developers." \
  --provider ollama \
  --model llama3.2:3b \
  --topic "Python error handling" \
  --count 200 \
  --output ./synth_raw.jsonl
```

Para proveedores Anthropic / vLLM / server, cambia `--provider` y sigue la
matriz de `soup data generate --help`.

## 2. Filtrar por calidad

Descarta filas de baja perplejidad / baja coherencia:

```bash
soup data filter ./synth_raw.jsonl \
  --output ./synth_filtered.jsonl \
  --min-coherence 0.5
```

## 3. Puntuar por seguridad + diversidad

Toma la huella de PII / toxicidad / idioma / valor educativo, luego descontamina
contra tus evaluaciones posteriores. `soup data score` informa una tarjeta de puntuación en la
terminal y no escribe un archivo, así que el paso de descontaminación lee el
conjunto filtrado:

```bash
soup data score --input ./synth_filtered.jsonl
soup data decontaminate \
  --input ./synth_filtered.jsonl \
  --output ./synth_clean.jsonl \
  --benchmarks mmlu,gsm8k
```

## 4. Entrenar

Apunta `soup train` a `synthetic_workflow.yaml`:

```bash
soup train --config examples/synthetic_workflow.yaml --yes
```

Esa receta referencia `./synth_clean.jsonl`, elige `TinyLlama-1.1B-Chat`
como base, y ejecuta un trabajo de SFT con LoRA r=8.

## 5. Ver el progreso en vivo (v0.53.9)

En otra terminal:

```bash
soup ui --public --no-browser
```

Escanea el código QR impreso desde tu teléfono para monitorear la curva de loss y
el stream de entrenamiento SSE en vivo mientras corre el trabajo.

---

Esta es una guía breve — para más detalle consulta
[examples/README.md](README.md) y [examples/configs/](configs/).
