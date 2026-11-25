"""Generate the packaged JSON Schema for the command result preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = _ROOT / "src" / "soup_cli" / "schemas" / "command-result-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Refuse a missing or stale schema.")
    args = parser.parse_args()

    sys.path.insert(0, str(_ROOT / "src"))
    from soup_cli.utils.command_result import CommandResult

    generated = json.dumps(CommandResult.model_json_schema(by_alias=True), indent=2, sort_keys=True)
    generated += "\n"
    if args.check:
        if not _SCHEMA.is_file() or _SCHEMA.read_text(encoding="utf-8") != generated:
            Console(stderr=True).print(
                "Command result schema is stale. Regenerate it without --check."
            )
            return 1
        return 0

    _SCHEMA.write_text(generated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
