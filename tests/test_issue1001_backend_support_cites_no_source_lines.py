"""#1001 — ``soup doctor --config`` printed ``commands/train.py:1235`` from a
``backend_support`` reason string. Nothing pinned it, and it had already drifted
once (it read ``:1186`` earlier in the PR that added it). A citation by name
survives edits above it; a line number does not.
"""

from __future__ import annotations

import re
from pathlib import Path

from soup_cli.config.backend_support import REGISTRY

#: ``some/file.py:123`` — a source location that silently rots when the file moves.
_LINE_CITATION = re.compile(r"[\w./-]+\.py:\d+")

_SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"


def _entries():
    for pair, entries in REGISTRY.items():
        for entry in entries:
            yield pair, entry


def test_no_backend_support_reason_cites_a_source_line():
    offenders = [
        f"{pair} {entry.field}: {match}"
        for pair, entry in _entries()
        for match in _LINE_CITATION.findall(entry.reason)
    ]
    assert offenders == [], (
        "a user-facing backend_support reason cites a source line, which drifts "
        f"silently when that file changes; name the function instead: {offenders}"
    )


def test_the_fsdp2_compile_reason_names_the_refusal_that_soup_train_runs():
    """The replacement citation is itself checked: the named function exists and
    ``soup train`` calls it, so renaming it turns this red instead of the doctor
    output going stale."""
    reason = next(
        entry.reason
        for _, entry in _entries()
        if entry.field == "training.use_fsdp2_compile"
    )
    assert "validate_fsdp2_compile_config" in reason

    from soup_cli.utils.fsdp import validate_fsdp2_compile_config

    errors = validate_fsdp2_compile_config(
        use_compile=True,
        fsdp_preset="full_shard",
        backend="mlx",
        device="cuda",
        deepspeed_config=None,
    )
    assert any("backend=transformers" in error for error in errors), errors

    train_source = (_SRC / "commands" / "train.py").read_text(encoding="utf-8")
    assert "validate_fsdp2_compile_config(" in train_source
    assert train_source.index("validate_fsdp2_compile_config(") < train_source.index(
        "resolve_trainer("
    )
