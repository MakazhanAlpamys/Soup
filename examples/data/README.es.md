# Datos de muestra

> 🇪🇸 Traducción al español de [`README.md`](README.md). Ante cualquier duda, manda el original en inglés.

**Estos son ejemplos de formato y fixtures de prueba — no datos de entrenamiento.** Cada archivo
contiene un puñado de filas, para mostrar cómo se ve un formato y darle algo que masticar al
parser y a la suite de tests. No se entrenará nada útil con este
volumen. Trátalos como una forma que copiar, no como un corpus inicial.

| Archivo | Formato | Filas | Demuestra |
|------|--------|-----:|--------------|
| `alpaca_tiny.jsonl` | alpaca | 10 | pares de instrucción `instruction` / `input` / `output` |
| `chat_preferences.jsonl` | dpo | 5 | `prompt` / `chosen` / `rejected`, mensajes como turnos de chat |
| `dpo_sample.jsonl` | dpo | 8 | La misma forma, respuestas más largas — la usa el ejemplo de DPO |
| `reasoning_math.jsonl` | alpaca | 5 | Soluciones desarrolladas paso a paso para GRPO/reasoning |

`soup data inspect <file>` imprime el formato detectado y las estadísticas de cualquiera de ellos.

## Conseguir datos reales

```bash
soup data search "instruction tuning"      # find datasets on the HF Hub
soup data preview <dataset-id>             # splits + features, before downloading
soup data download <dataset-id> -o out.jsonl
soup data demo                             # list bundled demo fixtures (same files as these)
```

## ¿Cuánto necesitas realmente?

- **Un formato o un estilo** (salida JSON, un tono de marca, una forma de respuesta): cientos de filas.
- **Una tarea que el modelo ya conoce a medias** (tu variante de resumen, clasificación): miles.
- **Datos nuevos** que el modelo nunca ha visto: el fine-tuning suele ser la herramienta equivocada — la recuperación (RAG)
  pone los datos en el prompt, y se mantiene correcta cuando cambian.

`soup advise <data> --goal "..."` lee tus datos y dice en cuál de estos casos estás.

Más sobre formatos, conversión y el pipeline de datos: [docs/data.md](../../docs/data.md).
