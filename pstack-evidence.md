# pstack-evidence — Soup#2219 r2 (2026-10-30 PT06)

## Goal
Address MakazhanAlpamys CHANGES_REQUESTED on #2219 (mutation report + ANSI/docs/tests).

## Change
- ship_provenance: off-table drift (e.g. schema_version) → `_GENERIC_STALE`; whitespace before `check_binding`
- tests: drop raw ANSI assert; pin data-sha256 / noise-floor / schema+adapter generic / suite dedup+full recovery sentence / non-mapping suites path
- docs/evaluation.md: full recovery message example; note off-table generic line

## Verification
`.venv/bin/pytest tests/test_ship_provenance.py -q -o addopts=` → 65 passed
`FORCE_COLOR=1 COLUMNS=80` hostile scorer test → 1 passed
`uvx ruff check` on changed files → All checks passed
