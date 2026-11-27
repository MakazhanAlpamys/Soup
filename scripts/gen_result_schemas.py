"""Generate the packaged command result and run status JSON Schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_DIR = _ROOT / "src" / "soup_cli" / "schemas"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Refuse a missing or stale schema.")
    args = parser.parse_args()

    sys.path.insert(0, str(_ROOT / "src"))
    from soup_cli.utils.command_result import CommandResult
    from soup_cli.utils.run_status import RunStatus

    stale = []
    for filename, model in (
        ("command-result-v1.json", CommandResult),
        ("run-status-v1.json", RunStatus),
    ):
        path = _SCHEMA_DIR / filename
        generated = json.dumps(model.model_json_schema(by_alias=True), indent=2, sort_keys=True)
        generated += "\n"
        if path.is_file() and path.read_text(encoding="utf-8") == generated:
            continue
        if args.check:
            stale.append(filename)
        else:
            path.write_text(generated, encoding="utf-8", newline="\n")

    if stale:
        names = ", ".join(stale)
        Console(stderr=True).print(f"Stale JSON Schema files: {names}. Regenerate without --check.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
