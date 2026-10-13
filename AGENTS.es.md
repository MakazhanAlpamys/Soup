# AGENTS.md

> 🇪🇸 Traducción al español de [`AGENTS.md`](AGENTS.md). Ante cualquier duda, manda el original en inglés.

Punto de entrada agnóstico de herramienta para agentes de código con IA (Codex, Cursor, Aider, Claude Code, etc.).

**Soup** es una herramienta de ajuste fino de LLM con la CLI como interfaz principal. Python 3.10+, Apache-2.0.

## Compilación y pruebas

```bash
pip install -e ".[dev]"          # Editable install + test deps
pytest tests/ -v --tb=short      # Run the suite (smoke tests are excluded by default)
ruff check src/soup_cli/ tests/  # Lint — must be clean before any commit
```

- `pytest -m smoke` ejecuta las pruebas lentas que descargan modelos y entrenan (se omiten por defecto).
- Matriz de CI: Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS.

## Convenciones (obligatorias)

- La **configuración** es Pydantic v2 en `src/soup_cli/config/schema.py` — única fuente de verdad.
- Las **dependencias pesadas** (`torch`, `transformers`, `peft`, `trl`, `mlx`) se importan de forma diferida dentro de las funciones, nunca en el nivel superior del módulo.
- La **salida** es mediante `rich.console.Console`, nunca `print()` sin más.
- **Contención de rutas**: usa `os.path.realpath` + `os.path.commonpath`, no `Path.resolve() + relative_to()` (falla con los nombres cortos de Windows).
- **Longitud de línea** 100, reglas de ruff `E, F, I, N, W`.

## Instrucciones completas

- La **referencia de funciones** está en [`docs/`](docs/README.md) — guías por tema más la lista completa de comandos de `soup` ([`docs/commands.md`](docs/commands.md)). Lee la página correspondiente antes de cambiar una función.
- El **flujo de trabajo de contribución, la estructura del proyecto y las notas de arquitectura** están en [`CONTRIBUTING.md`](CONTRIBUTING.md). Léelo antes de hacer cambios no triviales.
- El esquema de configuración de Pydantic en [`src/soup_cli/config/schema.py`](src/soup_cli/config/schema.py) es la única fuente de verdad para cada campo de configuración.
