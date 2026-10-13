# Reglas de traducción al español — repositorio Soup

Estándar **obligatorio** para toda traducción al español de este repositorio (rama `espanol`,
mantenida por [@yunyminaya](https://github.com/yunyminaya)). Todo traductor (humano o agente)
lee este archivo primero y lo cumple al pie de la letra.

## 1. Convención de nombres

- Cada archivo `F.md` se traduce a `F.es.md` **en el mismo directorio**. No hay subcarpeta de idioma.
- Nunca se sobrescribe ni se edita el original en inglés, ni las traducciones existentes
  (`README.tr.md`, `README.ar.md`, `README.ja.md`, `README.zh.md`).
- Si `F.es.md` ya existe, **se salta** (el proceso es idempotente y reanudable).

## 2. Estructura: idéntica al original

- Mismos encabezados, mismos niveles, mismo orden y mismo número de secciones.
- Mismas tablas: mismo número de columnas y de filas; sólo se traduce el texto de las celdas.
- Mismas listas: mismo número de elementos, mismo anidamiento, misma numeración.
- Un párrafo = una línea física cuando el original lo es (no re-envuelvas líneas).
- El archivo termina con exactamente un salto de línea.

## 3. Código: se copia, no se traduce

- **Bloques de código**: byte a byte idénticos, incluidos los comentarios `#`.
- **Código en línea** (`` `...` ``): intacto. Comandos, subcomandos, banderas (`--kv-cache-type`),
  rutas, claves de `soup.yaml`, nombres de modelos, variables de entorno, versiones, nombres de
  archivos/módulos/clases/PR, nombres de API y de eventos: **nunca** se traducen ni se reformatean.
- La prosa que rodea al código sí se traduce.

## 4. Enlaces: sólo cambia el texto visible

- Los **destinos** no cambian: ni URLs externas, ni enlaces relativos (`docs/models.md`,
  `../benchmarks/README.md`), ni anclas de otros archivos. Se traduce el texto entre corchetes.
- **Anclas internas** (`](#seccion)`): se traducen junto con su encabezado y deben resolver al
  slug de ese encabezado traducido (minúsculas, sin puntuación, espacios → guiones, los acentos
  se conservan). Ejemplo: `## Uso rápido` → `](#uso-rápido)`.

## 5. Números: se copian tal cual

- Las cifras del original se mantienen dígito a dígito: `119.6` sigue siendo `119.6`, **nunca**
  `119,6`. Nada de separador de miles añadido, nada de cifras "redondeadas" ni convertidas.
- Un número escrito con palabras en inglés se escribe con palabras en español.

## 6. Encabezado de traducción

En **todos** los archivos salvo `README.es.md` (que usa el banner de idiomas del repositorio),
justo después del primer encabezado —o en la primera línea, si el archivo no tiene encabezado—:

```markdown
> 🇪🇸 Traducción al español de [`F.md`](F.md). Ante cualquier duda, manda el original en inglés.
```

## 7. Tono y estilo

- Español neutro latinoamericano (es-US), profesional y directo.
- Trato de **tú**, imperativo neutro: "ejecuta", "usa", "define", "consulta".
- Nada de voseo, nada de "vosotros".
- Nada de calcos: *support* → "admite/compatible" (no "soporta"), *default* → "por defecto",
  *learn more* → "más información", *out of the box* → "sin configuración adicional".
- Se conserva el formato del original: negritas, cursivas, `> ` de cita, avisos, emojis, badges.

## 8. Terminología

Consulta y respeta `docs/i18n/terminology.es.md`. Cuando ese archivo no cubra un término:
si es un nombre propio, una API, una bandera o un formato → se deja en inglés; si es vocabulario
de uso general → se traduce.

## 9. Archivos grandes (> 25 KB)

1. Lee el original por secciones: `read_file(<ruta>, offset=N, limit=800)`.
2. El primer trozo se escribe con `write_file(<ruta>.es.md, ...)`.
3. Los siguientes se **añaden** con
   `python3 -c "import sys;open('<ruta>.es.md','a').write(sys.stdin.read())" <<'EOF' ... EOF`
4. No se resume, no se omiten secciones, no se deja ningún `[...]` pendiente.

## 10. Fuera de alcance (no se traduce)

- `CHANGELOG.md` y `CONTRIBUTORS.md` (generados por herramientas).
- `changelog.d/**` (fragmentos que consume `scripts/assemble_changelog.py`; un nombre válido
  debe ser `<número>.<sección>.md` y un `.es.md` rompería la herramienta).
- `LICENSE`, `NOTICE`, `CODEOWNERS`, `.mailmap` (texto legal/técnico).
- Todo el código: `src/`, `tests/*.py`, `scripts/`, `pyproject.toml`, workflows, JSON, YAML.

## 11. Verificación antes de informar (obligatoria)

Para cada archivo traducido:

```bash
test "$(grep -c '^```' ORIGEN)" = "$(grep -c '^```' F.es.md)"   # vallas iguales
test "$(grep -c '^#' ORIGEN)" -le "$(grep -c '^#' F.es.md)"      # encabezados >= originales
test "$(wc -c < F.es.md)" -ge "$(( $(wc -c < ORIGEN) * 80 / 100 ))"  # >= 80% del tamaño
```

Si algo no cuadra, se corrige antes de cerrar. Nada de "ya está" sin números.

## 12. Informe final del traductor

Una línea por archivo traducido:

```
docs/models.md  -> docs/models.es.md  12345 bytes (origen 13002)  vallas 12/12 OK
```

más una línea de incidencias (archivo: problema: decisión). Si un archivo no se pudo traducir,
se dice explícitamente y por qué.
