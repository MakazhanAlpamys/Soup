"""Canonical exit codes for Soup gate and verdict commands.

Taxonomy (project-wide standard):
- 0 (EXIT_OK): Gate passed (SHIP / OK / MINOR / valid data).
- 1 (EXIT_RUNTIME_ERROR): Unexpected execution failure (internal error, crash).
- 2 (EXIT_GATE_FAILED): Model or artifact failed the gate threshold
  (DON'T SHIP / MAJOR / regression / drift / unusable data).
- 3 (EXIT_USAGE_ERROR): User invocation, flag, or input-data error
  (missing file, invalid format, unparseable input, bad flag).
"""

from __future__ import annotations

EXIT_OK: int = 0
EXIT_RUNTIME_ERROR: int = 1
EXIT_GATE_FAILED: int = 2
EXIT_USAGE_ERROR: int = 3

# Aliases for domain clarity across gates
EXIT_PASS: int = EXIT_OK
EXIT_SHIP: int = EXIT_OK
EXIT_DONT_SHIP: int = EXIT_GATE_FAILED
EXIT_REGRESSION: int = EXIT_GATE_FAILED
EXIT_DRIFT: int = EXIT_GATE_FAILED
EXIT_MAJOR: int = EXIT_GATE_FAILED
