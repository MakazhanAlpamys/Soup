#!/usr/bin/env python3
"""The attach ratchet over every shipped config (#1116, second half).

``soup recipes verify`` builds each shipped recipe, template and example on the
meta device and runs the trainer's own target resolution and the real peft attach.
It exits 2 while any config cannot attach, and some shipped configs knowingly
cannot yet. So this script pins those, each with a reason, and fails only on a
change: a config that newly cannot attach, or a pinned one that now attaches and
should leave the list. Either way a change is a deliberate edit here.

It prints one count line so a run without a token is never read as full coverage
(ruled on #1116, 2026-09-21): gated bases are *skipped*, never passed or failed.
``.github/workflows/recipe-attach.yml`` runs it on pull requests without a token
and on push to main with ``HF_TOKEN``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field

_NO_AUTO = (
    "target_modules: auto has no mapping for this model_type; the #1070 table may "
    "grow one family per PR, measured on the meta device (ruled on #1116)"
)
_NO_PEFT_DEFAULT = (
    "peft has no default targets for this wrapper model_type, so it needs "
    "tower-scoped target_modules (#1116)"
)
_VL_WRAPPER = (
    "the base is a vision-language wrapper that the task's text model class "
    "cannot build"
)

#: Configs that cannot attach today, measured on the meta device without a token.
#: A name leaves this list when it attaches; a new one needs a reason to join.
EXCEPTIONS: dict[str, str] = {
    "lfm2-sft": _NO_AUTO,
    "ministral-sft": _NO_AUTO,
    "paddle-ocr-sft": _NO_AUTO,
    "phi3.5-mini-sft": _NO_AUTO,
    "phi4-14b-sft": _NO_AUTO,
    "phi4-reasoning-grpo": _NO_AUTO,
    "qvq-72b-sft": _NO_AUTO,
    "qwen2-vl-72b-sft": _NO_AUTO,
    "qwen2-vl-7b-sft": _NO_AUTO,
    "ra-dit-retriever": _NO_AUTO,
    "seamlessm4t-v2-sft": _NO_AUTO,
    "smollm3-3b-sft": _NO_AUTO,
    "whisper-large-v3-ft": _NO_AUTO,
    "llava-next-sft": _NO_PEFT_DEFAULT,
    "qwen2-audio-7b-sft": _NO_PEFT_DEFAULT,
    "templates/audio.yaml": _NO_PEFT_DEFAULT,
    "voxtral-sft": _NO_PEFT_DEFAULT,
    "mistral-medium-3-5-sft": _VL_WRAPPER + " (mistral3)",
}

_VERIFIED = frozenset({"attaches", "no_adapter"})


@dataclass
class Ratchet:
    verified: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    new_failures: list[tuple[str, str]] = field(default_factory=list)
    now_attach: list[str] = field(default_factory=list)
    not_reported: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.new_failures and not self.now_attach and not self.not_reported


def compare(rows: list[dict], exceptions: dict[str, str]) -> Ratchet:
    """Sort ``soup recipes verify --json`` rows against the pinned exceptions."""
    result = Ratchet()
    for row in rows:
        name, verdict = row["name"], row["verdict"]
        if verdict == "unverified":
            # Gated, remote code, or unreadable here: not a pass, not a failure.
            result.skipped.append((name, row.get("detail", "")))
        elif verdict == "cannot_attach":
            if name in exceptions:
                result.pinned.append(name)
            else:
                result.new_failures.append((name, row.get("detail", "")))
        elif verdict in _VERIFIED:
            result.verified.append(name)
            if name in exceptions:
                result.now_attach.append(name)
        else:
            raise ValueError(f"unknown verdict {verdict!r} for {name}")
    # verify leaves out a config that does not parse; a pinned one must not vanish.
    seen = {row["name"] for row in rows}
    result.not_reported = sorted(set(exceptions) - seen)
    return result


def count_line(result: Ratchet) -> str:
    return (
        f"{len(result.verified)} verified, {len(result.skipped)} skipped (gated, "
        f"remote code or unreadable here), {len(result.pinned)} pinned cannot-attach, "
        f"{len(result.new_failures)} new cannot-attach, "
        f"{len(result.now_attach)} pinned that now attach"
    )


def report(result: Ratchet) -> list[str]:
    lines = [count_line(result)]
    # The ruling asks for SKIPPED *with the reason*: a token only fixes the gated ones.
    by_reason = Counter(detail[:120] for _name, detail in result.skipped)
    for reason, count in by_reason.most_common():
        lines.append(f"  skipped {count}: {reason}")
    for name, detail in result.new_failures:
        lines.append(
            f"NEW cannot attach: {name}: {detail[:200]} -- fix it, or pin it in "
            "EXCEPTIONS in scripts/check_recipe_attach.py with a reason"
        )
    for name in result.now_attach:
        lines.append(f"now attaches, remove from EXCEPTIONS: {name}")
    for name in result.not_reported:
        lines.append(f"pinned but not reported by verify: {name}")
    return lines


def main() -> int:
    run = subprocess.run(
        [sys.executable, "-m", "soup_cli", "recipes", "verify", "--json"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    # 0: everything checkable attaches; 2: something cannot. Anything else is the
    # command itself failing (a missing library is 1), not a verdict.
    # Always forwarded: verify's advisories (a partial-coverage note, a config that
    # does not parse) belong in the CI log whatever the verdict.
    if run.stderr:
        print(run.stderr, file=sys.stderr, end="" if run.stderr.endswith("\n") else "\n")
    if run.returncode not in (0, 2):
        print(f"soup recipes verify exited {run.returncode}", file=sys.stderr)
        return 1
    result = compare(json.loads(run.stdout), EXCEPTIONS)
    print("\n".join(report(result)))
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
