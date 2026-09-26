"""Regression guards for stale release-planning text in user-facing help."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from typer.testing import CliRunner

from soup_cli import __version__
from soup_cli.cli import app

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "src" / "soup_cli" / "config" / "schema.py"
COMMAND_PATHS = (
    ROOT / "src" / "soup_cli" / "cli.py",
    *(ROOT / "src" / "soup_cli" / "commands").glob("*.py"),
)

SCHEMA_ONLY = re.compile(r"schema[- ]only", re.IGNORECASE)
VERSIONED_RELEASE_PROMISE = re.compile(
    r"(?:deferred\s+to|lands\s+in|ships\s+in)\s+v"
    r"(?P<version>\d+(?:\.\d+){1,2})",
    re.IGNORECASE,
)
INTERNAL_RELEASE_PHASE = re.compile(
    r"\bv\d+(?:\.\d+){1,2}\s+part\s+[a-z]\b",
    re.IGNORECASE,
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Add a field here only when its schema-only status is intentional and the
# accompanying description states the current limitation. Versioned promises
# remain checked even for allowlisted fields.
ALLOWED_SCHEMA_ONLY_FIELDS: frozenset[str] = frozenset()


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split(".")]
    padded = parts + [0, 0]
    return padded[0], padded[1], padded[2]


CURRENT_RELEASE = _version_tuple(__version__)


def _has_expired_version_promise(text: str) -> bool:
    return any(
        _version_tuple(match.group("version")) <= CURRENT_RELEASE
        for match in VERSIONED_RELEASE_PROMISE.finditer(text)
    )


def _schema_description_is_stale(
    name: str,
    description: str,
    *,
    allowed_schema_only_fields: frozenset[str] = ALLOWED_SCHEMA_ONLY_FIELDS,
) -> bool:
    return _has_expired_version_promise(description) or (
        SCHEMA_ONLY.search(description) is not None
        and name not in allowed_schema_only_fields
    )


def _help_text_is_stale(text: str) -> bool:
    return _has_expired_version_promise(text) or SCHEMA_ONLY.search(text) is not None


def _literal_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _plain(text: str) -> str:
    """Strip Rich styling and collapse wrapping before matching CLI help."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _schema_descriptions(path: Path = SCHEMA_PATH) -> list[tuple[str, int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    descriptions: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for keyword in node.value.keywords:
            if keyword.arg != "description":
                continue
            description = _literal_string(keyword.value)
            if description is not None:
                descriptions.append((node.target.id, node.lineno, description))
    return descriptions


def _cli_help_strings() -> list[tuple[Path, int, str]]:
    strings: list[tuple[Path, int, str]] = []
    for path in COMMAND_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg != "help":
                        continue
                    value = _literal_string(keyword.value)
                    if value is not None:
                        strings.append((path, node.lineno, value))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_command = any(
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in {"command", "callback"}
                    for decorator in node.decorator_list
                )
                if is_command and (docstring := ast.get_docstring(node)):
                    strings.append((path, node.lineno, docstring))
    return strings


def test_schema_descriptions_have_no_expired_release_promises() -> None:
    descriptions = _schema_descriptions()
    assert len(descriptions) >= 300
    offenders = [
        f"{name}:{line}: {description}"
        for name, line, description in descriptions
        if _schema_description_is_stale(name, description)
    ]
    assert offenders == []


def test_cli_help_has_no_expired_promises_or_internal_phase_labels() -> None:
    help_strings = _cli_help_strings()
    assert len(help_strings) >= 1_000
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: {text}"
        for path, line, text in help_strings
        if _help_text_is_stale(text) or INTERNAL_RELEASE_PHASE.search(text)
    ]
    assert offenders == []


def test_stale_copy_scanner_can_fail_without_banning_future_notices() -> None:
    current = ".".join(str(part) for part in CURRENT_RELEASE)
    future = f"{CURRENT_RELEASE[0]}.{CURRENT_RELEASE[1] + 1}.0"
    assert _has_expired_version_promise(f"Runtime lands in v{current}")
    assert _has_expired_version_promise("Runtime was deferred to v0.74.1")
    assert not _has_expired_version_promise(f"Removal ships in v{future}")
    assert not _help_text_is_stale("Applies the post-v2 attention kernel")
    assert _schema_description_is_stale("reserved", "Reserved (schema-only).")
    assert not _schema_description_is_stale(
        "reserved",
        "Reserved (schema-only).",
        allowed_schema_only_fields=frozenset({"reserved"}),
    )


def test_the_schema_scanner_can_actually_fail(tmp_path: Path) -> None:
    sample = tmp_path / "schema.py"
    sample.write_text(
        'field: bool = Field(default=False, description="Runtime lands in v0.75.0")\n',
        encoding="utf-8",
    )

    descriptions = _schema_descriptions(sample)

    assert len(descriptions) == 1
    name, line, description = descriptions[0]
    assert (name, line) == ("field", 1)
    assert _schema_description_is_stale(name, description)


def test_advise_record_help_does_not_name_an_unavailable_inverse_flag() -> None:
    result = CliRunner().invoke(app, ["advise", "run", "--help"])

    assert result.exit_code == 0, result.output
    plain = _plain(result.output)
    assert "--record" in plain
    assert "--no-record" not in plain


def test_named_docs_no_longer_promise_past_follow_up_releases() -> None:
    stale_fragments = {
        "docs/data.md": (
            "fsspec backend wiring lands in v0.42.1",
            "Live offline runner against a local model lands in v0.45.1",
            "Live wiring of the proxy training loop into a short `soup train` run is the "
            "v0.48.1 deliverable",
        ),
        "docs/backends-and-ops.md": (
            "The hub adapter is schema-only in this release",
            "Trainer-callback wiring of `pre_train` / `post_train` / `pre_step` / "
            "`post_step` lands in v0.45.1",
        ),
        "docs/peft-and-efficiency.md": ("LongLoRA S² (schema-only this release)",),
        "docs/adapters-and-governance.md": ("ship as schema-only in v0.67.0",),
    }
    offenders = []
    for relative_path, fragments in stale_fragments.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        offenders.extend(
            f"{relative_path}: {fragment}" for fragment in fragments if fragment in text
        )
    assert offenders == []
