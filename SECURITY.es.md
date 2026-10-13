# Política de seguridad

> 🇪🇸 Traducción al español de [`SECURITY.md`](SECURITY.md). Ante cualquier duda, manda el original en inglés.

## Versiones compatibles

Las correcciones de seguridad se publican para la última versión publicada. Recomendamos
ejecutar siempre la versión más reciente de PyPI (`pip install -U soup-cli`).

| Versión | Compatible         |
| ------- | ------------------ |
| 0.76.x  | :white_check_mark: |
| < 0.76  | :x:                |

## Reportar una vulnerabilidad

Reporta los problemas de seguridad **de forma privada** — no abras un issue público de GitHub
para nada relacionado con la seguridad.

- Preferido: abre un reporte privado mediante
  [GitHub Security Advisories](https://github.com/MakazhanAlpamys/Soup/security/advisories/new).
- Si no puedes usar GitHub, escribe a **team@trysoup.dev** (la dirección del proyecto), o a
  **makazanalpamys@gmail.com** (la dirección personal del mantenedor) si esa rebota.
  No reportes problemas de seguridad en Discord — es un canal público.

Nuestro objetivo es acusar recibo de los reportes en un plazo de 5 días hábiles y publicar una
corrección o mitigación para los problemas confirmados y dentro del alcance lo antes posible.
Al reportar, por favor incluye:

- la(s) versión(es) y plataforma afectadas,
- una reproducción mínima o prueba de concepto,
- el impacto que observaste.

## Alcance

Soup es una CLI local-first para el ajuste fino de LLM. El modelo de amenazas asume que el
operador ejecuta Soup en su propia máquina con sus propios datos. Problemas representativos
dentro del alcance:

- path traversal o lectura/escritura arbitraria de archivos desde rutas de configuración,
  dataset o artefactos que proporciona el usuario;
- SSRF en los proveedores de datos sintéticos, el servidor de inferencia o los validadores de
  hub/endpoint;
- inyección de comandos, Modelfile, plantillas de chat Jinja o unidades systemd/launchd;
- filtración de secretos en los logs, paquetes de fallos o artefactos generados;
- escape del sandbox en la ruta de recompensa de ejecución de código de RLVR.

Fuera de alcance:

- vulnerabilidades en pesos de modelos o datasets de terceros que elijas cargar;
- problemas que requieren un host ya comprometido;
- la configuración de DNS y correo de `trysoup.dev` — un registro DMARC / SPF / DKIM ausente o
  permisivo, y cualquier otra cosa que se establezca mediante una consulta DNS pública. Vale la
  pena corregirlos y los corregimos, pero no son hallazgos en Soup ni son elegibles para nada.

### Telemetría y salida de red

Soup es local-first. La telemetría es estrictamente **opt-in** (deshabilitada por defecto) mediante `SOUP_TELEMETRY=1` y se puede sobrescribir por invocación con `--no-telemetry`. Cuando está habilitada, la telemetría transmite metadatos de entorno anónimos, únicamente de hardware (SO, arquitectura, versión de Python, nombre del comando saneado contra los comandos registrados) por HTTPS. Nunca recopila pesos de modelos, contenidos de datasets, rutas de archivos, configuraciones ni secretos. Consulta la [Política de privacidad](docs/backends-and-ops.md#privacy-policy) para obtener todos los detalles.

**No hay bug bounty ni recompensa monetaria.** Acreditamos a quienes reportan por su nombre
en las notas de la versión, que es todo lo que ofrecemos. Los reportes que empiezan pidiendo un
pago reciben este párrafo como respuesta.

## Divulgación

Practicamos la divulgación coordinada. Una vez publicada la corrección, acreditamos a quien
reporta en las notas de la versión, salvo que se solicite el anonimato.

> Un registro detallado, por versión, del endurecimiento de seguridad histórico vivía antes
> en este archivo. Ahora vive en el historial de git del proyecto y en las notas de
> [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).
