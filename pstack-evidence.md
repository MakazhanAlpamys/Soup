# pstack-evidence — Soup#2255 r2 (2026-10-30 PT06)

## Goal
Address MakazhanAlpamys CHANGES_REQUESTED on #2255: pin decimals with literals, rank 0 guard, and unknown+None for bad rank.

## Change
tests/test_adapters_audit_rslora.py only: tighten negative-rank asserts; add zero-rank and nine-decimal near-miss cases; near-miss expectation is a literal.

## Verification
`.venv/bin/pytest tests/test_adapters_audit_rslora.py -q -o addopts=` → 13 passed
`uvx ruff check tests/test_adapters_audit_rslora.py` → All checks passed
