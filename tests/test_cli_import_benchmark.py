"""CodSpeed regression gate for the CLI's own import cost (#780, #1065).

`soup version` and `soup --help` spend roughly half their cold-start cost --
240-300ms of a 410-680ms total, measured across three CI platforms in #780 --
importing `soup_cli.config.schema`, the dominant leaf in that profile. This
benchmark deletes that one module from `sys.modules` and re-imports it on
every CodSpeed iteration, so a PR that makes the schema tree more expensive
to build shows up as a percentage change on the PR instead of as an
unwritten cost, the way #780 itself was.

This is a different instrument than #780's own measurement, not a
replacement for it (per #1065): instruction counting inside one long-lived
interpreter, not a cold subprocess start. It will not reproduce #780's
absolute milliseconds -- OS and import-metadata caches stay warm across
iterations here, and interpreter startup itself isn't part of what's timed.
What it tracks is the same import graph's cost moving, relatively, PR to PR.

The original module is put back after every iteration, and that is load
bearing: a re-import builds a *new* `SoupConfig` class, while
`soup_cli.config.loader` keeps the old one, so leaving the new module in
`sys.modules` makes every later `isinstance(load_config(...), SoupConfig)` in
the session false (first CI run of #1086: `test_v07112.py`, 9 of 9 cells).
"""

from __future__ import annotations

import importlib
import sys

_TARGET = "soup_cli.config.schema"


def _reimport_schema() -> None:
    package = importlib.import_module("soup_cli.config")
    original = importlib.import_module(_TARGET)
    del sys.modules[_TARGET]
    try:
        importlib.import_module(_TARGET)
    finally:
        sys.modules[_TARGET] = original
        package.schema = original


def test_config_schema_import_cost(benchmark) -> None:
    benchmark(_reimport_schema)


def test_the_benchmark_leaves_the_original_schema_module_in_place() -> None:
    """Fails if the restore above is dropped: the rest of the session would see
    a different `SoupConfig` than the one `soup_cli.config.loader` builds."""
    before = importlib.import_module(_TARGET)
    _reimport_schema()
    assert sys.modules[_TARGET] is before
    assert importlib.import_module("soup_cli.config").schema is before
