"""CodSpeed regression gate for the CLI's own import cost (#780, #1065).

`soup version` and `soup --help` spend roughly half their cold-start cost --
240-300ms of a 410-680ms total, measured across three CI platforms in #780 --
importing `soup_cli.config.schema`, the dominant leaf in that profile. This
benchmark deletes that one module from `sys.modules` and re-imports it on
every CodSpeed iteration, so a PR that makes the schema tree more expensive
to build shows up as a percentage change on the PR instead of as an
unwritten cost, the way #780 itself was.

What it measures, exactly: **a warm-process re-import, not #780's cold
start.** Instruction counting inside one long-lived interpreter, not a fresh
subprocess (per #1065). It will not reproduce #780's absolute milliseconds --
OS and import-metadata caches stay warm across iterations, and interpreter
startup is not timed -- so do not compare the two sets of numbers. What it
tracks is the cost of executing the schema module's own body (building its
pydantic model tree) moving, relatively, PR to PR.

What it does NOT catch: only `soup_cli.config.schema` itself is evicted, so
everything it imports stays cached after the first iteration. A new
*transitive* import added to that module is paid once during warm-up and
barely moves the number (planting `import transformers` in it did move it, but
not by its real cost -- found in review of #1086). `test_cli_startup_is_light`
remains the guard for transitive imports.

The original module is put back after every iteration, and that is load
bearing: a re-import builds a *new* `SoupConfig` class, while
`soup_cli.config.loader` keeps the old one, so leaving the new module in
`sys.modules` makes every later `isinstance(load_config(...), SoupConfig)` in
the session false (first CI run of #1086: `test_v07112.py`, 9 of 9 cells).

`pytest-codspeed` is deliberately not in `[dev]` (#1086 review): it is an
unbounded third-party pytest plugin and `[dev]` feeds all nine required cells.
`codspeed.yml` installs it itself; without it the benchmark test is skipped
and the isolation test below still runs everywhere.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

import pytest

_TARGET = "soup_cli.config.schema"

# `pytest.importorskip` inside the test body cannot do this job: the `benchmark`
# fixture is resolved before the body runs, so without the plugin the test would
# ERROR ("fixture 'benchmark' not found") instead of skipping. A skipif mark is
# evaluated first, and the test is still collected as a benchmark under --codspeed.
_HAS_CODSPEED = importlib.util.find_spec("pytest_codspeed") is not None


def _reimport_schema() -> ModuleType:
    package = importlib.import_module("soup_cli.config")
    original = importlib.import_module(_TARGET)
    del sys.modules[_TARGET]
    try:
        return importlib.import_module(_TARGET)
    finally:
        sys.modules[_TARGET] = original
        package.schema = original


@pytest.mark.skipif(
    not _HAS_CODSPEED, reason="pytest-codspeed is installed by codspeed.yml, not by [dev]"
)
def test_config_schema_import_cost(benchmark) -> None:
    benchmark(_reimport_schema)


def test_the_benchmark_really_re_imports_and_then_restores() -> None:
    """Two failure modes, each of which used to be invisible:

    * the re-import does nothing -- the benchmark would then pass while measuring
      0 ns (found in review of #1086: replacing the import with `pass` kept every
      test green), so the module returned must be a NEW object that built new
      classes;
    * the restore is dropped -- the rest of the session would see a different
      `SoupConfig` than the one `soup_cli.config.loader` builds.
    """
    before = importlib.import_module(_TARGET)
    fresh = _reimport_schema()
    assert fresh is not before
    assert hasattr(fresh, "SoupConfig")
    assert fresh.SoupConfig is not before.SoupConfig
    assert sys.modules[_TARGET] is before
    assert importlib.import_module("soup_cli.config").schema is before
