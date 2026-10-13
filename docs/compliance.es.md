# Inicio rápido de cumplimiento y gobernanza

> 🇪🇸 Traducción al español de [`compliance.md`](compliance.md). Ante cualquier duda, manda el original en inglés.

[← Volver al índice de la documentación](README.md)

Soup incluye un flujo de trabajo de cumplimiento de extremo a extremo: empieza con una
configuración con forma de regulación, entrena con captura de procedencia, firma + certifica el
artefacto, mantén un rastro de auditoría, y muévelo a través de un air gap — luego publica una
ficha de modelo documentada y controla los cambios futuros en CI.

- **Configuraciones preconfiguradas:** `soup init --template hipaa|soc2|eu-ai-act|sr-11-7`
- **Procedencia:** `soup train --repro-receipt` · `soup bom emit` · `soup attest emit`
- **Integridad:** `soup adapters sign` / `verify` · `soup adapters scan`
- **Rastro de auditoría:** el log de auditoría HIPAA/SOC2 está activado por defecto (`soup audit-log tail`)
- **Air gap:** `soup airgap-bundle`
- **Publicación:** `soup card <registry-id>` → `MODELCARD.md` (o `soup push --card`)
- **Puerta de CI:** `soup ci init` → un flujo de trabajo de PR que ejecuta validate → expect → ship

---

## 1. Empieza desde una plantilla de cumplimiento

Cada plantilla es una configuración de entrenamiento normal más comentarios de encabezado que
enumeran los comandos de cumplimiento exactos que se deben ejecutar en torno a ella. Elige el
régimen bajo el que operas:

```bash
soup init --template hipaa      # Protected Health Information
soup init --template soc2       # SOC 2 Trust Services Criteria
soup init --template eu-ai-act  # EU AI Act Annex XI/XII
soup init --template sr-11-7    # SR 11-7 Model Risk Management
```

Los controles de cumplimiento son **banderas/comandos de la CLI de Soup, no claves de
configuración** — el encabezado de la plantilla documenta cuáles aplican. Los pasos siguientes
son el camino común.

## 2. Limpia los datos antes de entrenar

```bash
soup data pii --input ./data/train.jsonl    # flag emails / phones / SSNs / MRNs
soup data decontaminate --input ./data/train.jsonl --benchmark-file benchmarks.jsonl  # drop benchmark overlap
```

## 3. Entrena con un recibo de reproducibilidad (+ Anexo XI / energía para la UE)

```bash
# SR 11-7 / SOC 2 / HIPAA: capture seeds, kernels, GPU, OS
soup train --config soup.yaml --repro-receipt receipt.json

# EU AI Act: auto-generate the Annex XI/XII documentation + measure energy
soup train --config soup.yaml \
    --annex-xi annex_xi.md \
    --track-energy --energy-country DEU --energy-out energy.json
```

El log de auditoría registra cada comando automáticamente:

```bash
soup audit-log tail          # review the trail
soup audit-log rotate        # force a rotation pass
```

## 4. Registra la ejecución, luego emite BOM + atestación

```bash
soup registry push --run-id <run-id> --name my-model --tag v1

soup bom emit --name my-model --base-model <model-id> \
    --base-sha <hex> --config-sha <hex> \
    --energy energy.json --format both -o my-model.bom \
    --attach-to-registry my-model:v1         # CycloneDX + SPDX, linked to the entry
soup attest emit --stage train --subject my-model --sha <hex> \
    --sign ed25519 --key key.pem -o my-model.attest.json \
    --attach-to-registry my-model:v1         # in-toto + SLSA-3, linked to the entry
```

`--attach-to-registry` registra el BOM / la atestación emitidos como artefactos `bom` /
`attestation` en la entrada. Las atestaciones firmadas incluyen su archivo auxiliar `.sig`,
así que la ficha de modelo (paso 7) enlaza cada archivo necesario para la verificación.
La opción necesita `--output`; omitirla es un error de uso. Un fallo de búsqueda o de
adjunción en el registry sale con un código distinto de cero mientras deja los archivos
emitidos en disco.

## 5. Firma, escanea y verifica el artefacto

```bash
soup adapters scan ./output                              # weight-space backdoor scan
soup adapters sign ./output --backend ed25519 --generate-key key.pem
soup adapters verify ./output --strict --public-key key.pub.pem
```

## 6. Transferencia por air gap (opcional)

```bash
soup airgap-bundle --model ./output --output my-model.tar --repro-receipt receipt.json
```

## 7. Genera una ficha de modelo documentada

Convierte la entrada del registry en un `MODELCARD.md` rico en procedencia — modelo base,
configuración de entrenamiento, scorecard de evaluación, hashes de configuración/datos, linaje,
y cada artefacto registrado — incluidos el BOM y la atestación adjuntos en el paso 4:

```bash
soup card my-model:v1 -o MODELCARD.md
# or, when uploading to the Hub, override the auto-generated card:
soup push --model ./output --repo you/my-model --card my-model:v1
```

## 8. Controla los cambios futuros en CI

Escribe un flujo de trabajo de GitHub Actions que bloquee un PR a menos que los datos se validen,
la suite de expectativas pase y el veredicto SHIP sea verde:

```bash
soup ci init --data data/train.jsonl --suite expectations.yaml --evidence ship_evidence.json
# writes .github/workflows/soup-gate.yml
```

El job generado ejecuta, en orden:

```
soup data validate <data>       # dataset format compliance
soup expect <data> <suite>      # PII / token-length / refusal / judge expectations
soup ship --evidence <ev.json>  # SHIP / DON'T-SHIP (exit 2 blocks the merge)
```

Una `expectations.yaml` mínima para el segundo paso:

```yaml
expectations:
  - name: expect_no_pii
  - name: expect_token_length_between
    args:
      min_tokens: 1
      max_tokens: 512
```

Nombres admitidos: `expect_no_pii`, `expect_token_length_between`,
`expect_no_refusal_pattern`, `expect_chosen_preferred_over_rejected_by_judge`.
Los argumentos van bajo `args:`. Una clave de entrada distinta de `name` / `args`, o un
argumento que la expectativa no acepta (`max_token`), se rechaza cuando la suite carga
(exit 3), así que un límite mal escrito no puede ejecutarse en silencio sobre los
valores por defecto. Una clave de nivel superior distinta de `expectations` (una `fail_fast:` o
una etiqueta `name:` al lado) se rechaza de la misma manera, nombrando la clave.
`expect_chosen_preferred_over_rejected_by_judge` requiere un modelo juez configurado
bajo `args: {judge: "ollama://llama3.1", threshold: 0.7}` o pasado por CLI con
`soup expect --judge <url>`. Cuando ambos están definidos, `--judge` tiene precedencia sobre
el `args.judge` de la suite. Sin un juez configurado, la puerta falla (o pasa
`args: {advisory: true}` para confiar explícitamente en las etiquetas existentes en modo advisory).

Campos escaneados por formato:
- ChatML / tool-calling / audio: cada `messages[].content` (cadena o partes de texto), argumentos de tool-call (por mensaje y `tool_calls` de nivel superior), y `tools[].function.description`.
- ShareGPT / LLaVA / ShareGPT4V: cada `conversations[].value`.
- Alpaca: `instruction`, `input`, `output`, `system`, `response`.
- DPO: `prompt`, `chosen`, `rejected` (cadena o lista de mensajes). KTO: `prompt`, `completion`.
- Embedding: `anchor`, `positive`, `negative` (cadena o lista).
- Plaintext / ASR: `text`. RAFT: `query`, `golden_doc`, `distractor_docs`, `answer`. PRM: `prompt`, `completions`. input_output: `segments[].text`.

`expect_no_refusal_pattern` lee solo el lado del asistente: turnos de assistant/gpt, `chosen` de DPO, `completion` de KTO, `output`/`response` de Alpaca, `answer` de RAFT, `completions` de PRM, `segments` con `label: true`, y `text` para las filas de texto plano. Una fila con texto pero sin lado de asistente lo pasa.

Salvaguarda de texto vacío: las filas que no contienen texto extraíble fallan de forma cerrada en todas las puertas de expectativas para impedir que las filas no analizables o en blanco pasen el CI. Las líneas JSON mal formadas y el JSON válido que no es un objeto se rechazan como error de entrada (exit 3) con sus números de línea, un BOM UTF-8 se elimina tal como lo elimina el cargador de entrenamiento, y un archivo que no produce ninguna fila verificable se rechaza en lugar de aceptarse.

Cada ruta se cita para el shell y se valida para mantenerse bajo la raíz del repositorio, de modo
que el flujo de trabajo generado es seguro contra inyección. Edita las rutas para que coincidan
con tu repositorio.

---

Ver también: [Adaptadores, registry y gobernanza](adapters-and-governance.md) para el
conjunto completo de comandos de la cadena de suministro, y [Evaluación y sondas](evaluation.md)
para el motor de veredictos de `soup ship`.
