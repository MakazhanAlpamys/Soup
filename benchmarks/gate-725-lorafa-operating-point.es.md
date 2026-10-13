# Registro de medición del punto operativo LoRA-FA (#725)
> 🇪🇸 Traducción al español de [`gate-725-lorafa-operating-point.md`](gate-725-lorafa-operating-point.md). Ante cualquier duda, manda el original en inglés.

Medido por [@webdevsamran](https://github.com/webdevsamran) el 2026-09-14 UTC.

Para [#725](https://github.com/MakazhanAlpamys/Soup/issues/725), este registro documenta el punto operativo medido y la verificación de microbenchmark de la compatibilidad opcional con el optimizador PEFT LoRA-FA (Frozen-A LoRA) en Soup.

## Alcance y resultado principal

- **Parámetros entrenables:** exactamente **50.0%** menos parámetros del adaptador por proyección adaptada (solo entrena $B$; congela $A$ en la inicialización aleatoria).
- **Estado del optimizador:** exactamente **50.0%** menos elementos de tensores del estado del optimizador AdamW (`exp_avg_B`, `exp_avg_sq_B`) para proyecciones lineales cuadradas adaptadas.
- **Consumo de memoria de activaciones:** proporción analítica de retención de activaciones del adaptador de $r / d_{in}$ durante la retropropagación a través de $A$.
- **Latencia por paso y finitud:** la progresión de pérdida en 5 pasos muestra una disminución suave y finita de la pérdida, momentos primero y segundo del optimizador distintos de cero en $B$, y gradientes cero en $A$.

> [!IMPORTANT]
> **Alcance y advertencias:** estas mediciones representan un punto operativo de microbenchmark controlado y una proporción analítica de tensores guardados para las matrices de proyección adaptadas ($r / d_{in}$).
> **NO son ahorros totales ni máximos de VRAM del LLM, un resultado de throughput de extremo a extremo ni una afirmación sobre la calidad del modelo posterior al entrenamiento.**
> En el ajuste fino completo de un LLM, la memoria máxima está dominada por las activaciones del modelo base, las cachés KV y los pesos, por lo que los ahorros de memoria totales de extremo a extremo son sustancialmente menores. La calidad de tareas de extremo a extremo a lo largo de varias épocas y el throughput de entrenamiento en tiempo real aún deben evaluarse para cargas de trabajo posteriores específicas.

---

## Entorno

- **Host:** Windows 11 (AMD64)
- **PyTorch:** `2.7.0` (o `2.14.0+cu130` en el control de GPU)
- **PEFT:** `0.20.0`
- **Transformers:** `5.17.0`
- **Objetivo de prueba:** modelo LLaMA de 2 capas, `hidden_size=256`, `intermediate_size=512`, `heads=4`, `vocab_size=1000`, adaptador `r=16`, `alpha=32`.
- **Arnés:** [`benchmarks/harness/issue725_lorafa_operating_point.py`](harness/issue725_lorafa_operating_point.py)

---

## Punto operativo medido

### 1. Contabilidad de parámetros y estado del optimizador

| Métrica | LoRA estándar | LoRA-FA | Relativo (% de LoRA) |
|---|---:|---:|---:|
| Parámetros totales del modelo base | 1,856,768 | 1,856,768 | 100.0% |
| Parámetros entrenables del adaptador | 32,768 | 16,384 | **50.0%** |
| Elementos del estado del optimizador AdamW | 65,544 | 32,768 | **50.0%** |

### 2. Pérdida en varios pasos y tiempo por paso (5 pasos)

| Paso | Pérdida de LoRA estándar | Pérdida de LoRA-FA | Paso de LoRA estándar (ms) | Paso de LoRA-FA (ms) |
|---:|---:|---:|---:|---:|
| 1 | 6.9638 | 6.9638 | 32.27 | 23.84 |
| 2 | 7.0036 | 7.0043 | 23.27 | 21.62 |
| 3 | 6.9437 | 6.9426 | 20.97 | 21.21 |
| 4 | 6.9893 | 6.9900 | 19.99 | 21.76 |
| 5 | 7.0022 | 7.0043 | 19.92 | 21.29 |

- **Finitud de la pérdida:** ambos brazos mantienen una pérdida estrictamente finita en todos los pasos.
- **Verificación de gradientes:** en LoRA-FA, todos los parámetros `lora_A` tienen `requires_grad=False` y `grad=None`. Todos los parámetros `lora_B` tienen `requires_grad=True` y gradientes no nulos poblados.
- **Momentos del optimizador:** `exp_avg_B` y `exp_avg_sq_B` contienen valores no nulos en el estado del optimizador LoRA-FA.

### 3. Retención analítica de memoria de activaciones

Para una capa lineal adaptada $h = W_0 x + B A x$, donde $x \in \mathbb{R}^{B \times L \times d_{in}}$ y $u = A x \in \mathbb{R}^{B \times L \times r}$:
- LoRA estándar debe retener $x$ durante la pasada forward para calcular $\nabla_A \mathcal{L} = x^\top (B^\top \nabla_h \mathcal{L})$ durante la pasada backward, lo que requiere almacenamiento proporcional a $B \times L \times d_{in}$.
- LoRA-FA congela $A$, así que la retropropagación a través de $A$ solo calcula $\nabla_x \mathcal{L} = A^\top (B^\top \nabla_h \mathcal{L})$ usando los pesos fijos de $A$, sin calcular ni guardar gradientes para $A$. Por lo tanto, solo debe retenerse la proyección intermedia $u \in \mathbb{R}^{B \times L \times r}$ para $\nabla_B \mathcal{L}$.
- La proporción analítica del almacenamiento retenido de activaciones del adaptador es:
  $$\text{Ratio} = \frac{r}{d_{in}}$$
  Para $r=16$ con $d_{in}=256$: la proporción es 0.0625 ($\approx 16.0\times$ de reducción en la retención de activaciones del adaptador).
  Para $r=16$ con $d_{in}=4096$: la proporción es $\approx 0.0039$ ($\approx 256.0\times$ de reducción en la retención de activaciones del adaptador).

De nuevo, esta reducción se aplica estrictamente a la retención del tensor de activación de entrada del adaptador, no a la memoria total de activaciones del LLM.

---

## Reproducción

Ejecuta el arnés del benchmark:
```bash
python benchmarks/harness/issue725_lorafa_operating_point.py
```
