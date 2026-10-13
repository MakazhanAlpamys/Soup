# Traducción al español de Soup

Este repositorio mantiene una traducción al español **junto al original en inglés**, nunca en lugar
de él. El inglés sigue siendo la fuente de verdad: si algo no coincide, manda el original.

**Mantenedor de esta traducción:** [@yunyminaya](https://github.com/yunyminaya) (rama `espanol`).

## Cómo está organizada

| Archivo | Contenido |
|---|---|
| `translation-rules.es.md` | Las reglas obligatorias que cumple cada archivo traducido |
| `terminology.es.md` | Glosario: qué se traduce y qué se queda en inglés |

- `F.md` → `F.es.md`, **en el mismo directorio**. No hay carpeta de idioma.
- El `README.md` de la raíz sigue la convención del repositorio: banner de idiomas + sello
  `synced-from` con el `sha256` del README, y `tests/test_readme_translation_sync.py` vigila que la
  traducción no se desincronice.

## Qué está traducido

- `README.es.md` y las guías de `docs/`
- `CONTRIBUTING.es.md`, `AGENTS.es.md`, `SECURITY.es.md`, `CODE_OF_CONDUCT.es.md`
- `examples/`, `tests/qa/` y los registros de `benchmarks/`

## Qué NO se traduce (y por qué)

| Ruta | Motivo |
|---|---|
| `CHANGELOG.md`, `CONTRIBUTORS.md` | Los genera una herramienta; se regeneran en cada versión |
| `changelog.d/**` | `scripts/assemble_changelog.py` valida el nombre de cada fragmento (`<número>.<sección>.md`); un `.es.md` rompería la herramienta |
| `LICENSE`, `NOTICE` | Texto legal: no se traduce |
| `src/`, `tests/`, `scripts/`, workflows, JSON/YAML | Código: los identificadores y los mensajes son parte del programa |

## Cómo se verifica

Cada archivo traducido se comprueba contra su original:

```bash
# las vallas de código coinciden
test "$(grep -c '^```' F.md)" = "$(grep -c '^```' F.es.md)"
# los encabezados no disminuyen
test "$(grep -c '^#' F.md)" -le "$(grep -c '^#' F.es.md)"
# y el tamaño no se desploma
test "$(wc -c < F.es.md)" -ge "$(( $(wc -c < F.md) * 80 / 100 ))"
```

El código no se traduce: los comandos, banderas, rutas, claves de configuración y nombres de API se
copian byte a byte. Lo único que puede cambiar dentro de un bloque de código es el texto de los
comentarios que empiezan con `#`.

Y el gate del propio repositorio, que es el que manda para el README:

```bash
pytest tests/test_readme_translation_sync.py -q --no-cov
```

## Cómo contribuir a la traducción

1. Lee `translation-rules.es.md` y `terminology.es.md` antes de tocar nada.
2. Si `README.md` cambia, actualiza `README.es.md` y sustituye su sello por la línea que imprime el
   test cuando falla (el propio test la imprime).
3. Un término nuevo se agrega al glosario en el mismo cambio que lo usa.
