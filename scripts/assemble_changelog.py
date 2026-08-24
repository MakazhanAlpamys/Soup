#!/usr/bin/env python3
"""Validate and assemble conflict-free changelog fragments."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SECTIONS = ("added", "changed", "deprecated", "removed", "fixed", "security")
SECTION_TITLES = {section: section.title() for section in SECTIONS}
FRAGMENT_NAME = re.compile(
    rf"^(?P<number>[1-9]\d*)\.(?P<section>{'|'.join(SECTIONS)})\.md$"
)
RELEASE_HEADING = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\](?: - [^\r\n]*)?\r?$", re.MULTILINE
)
UNRELEASED_HEADING = re.compile(r"^## \[Unreleased\][^\r\n]*(?:\r?\n|$)", re.MULTILINE)
SECTION_HEADING = re.compile(
    r"^### (?P<title>[^\r\n]+?)[^\S\r\n]*(?:\r?\n|$)", re.MULTILINE
)


class ChangelogError(RuntimeError):
    """A changelog fragment cannot be safely assembled."""


@dataclass(frozen=True)
class Fragment:
    """A validated changelog fragment."""

    path: Path
    number: int
    section: str
    content: str


def _read_utf8(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChangelogError(f"{path}: changelog files must be valid UTF-8") from exc


def _latest_release(changelog: str, path: Path) -> str:
    match = RELEASE_HEADING.search(changelog)
    if match is None:
        raise ChangelogError(f"{path}: no released version heading was found")
    return match.group("version")


def _fragment_files(fragment_root: Path) -> list[Path]:
    if not fragment_root.exists():
        return []
    if not fragment_root.is_dir():
        raise ChangelogError(f"{fragment_root}: expected a directory")

    files: list[Path] = []
    for path in sorted(fragment_root.rglob("*")):
        if path.is_symlink():
            raise ChangelogError(f"{path}: symlinks are not allowed in changelog.d")
        if path.is_file():
            files.append(path)
    return files


def validate_fragments(root: Path | str = ".") -> list[Fragment]:
    """Return every fragment after validating layout, version, name, and content."""
    root = Path(root)
    changelog_path = root / "CHANGELOG.md"
    if not changelog_path.is_file():
        raise ChangelogError(f"{changelog_path}: file not found")

    changelog = _read_utf8(changelog_path)
    latest_release = _latest_release(changelog, changelog_path)
    fragment_root = root / "changelog.d"
    fragments: list[Fragment] = []

    for path in _fragment_files(fragment_root):
        relative = path.relative_to(fragment_root)
        if relative == Path("README.md"):
            continue
        if len(relative.parts) == 1:
            raise ChangelogError(
                f"{path}: fragment must be inside a release-version directory"
            )
        if len(relative.parts) != 2:
            raise ChangelogError(
                f"{path}: fragment must be exactly one directory below changelog.d"
            )

        baseline = relative.parts[0]
        if baseline != latest_release:
            raise ChangelogError(
                f"{path}: fragment baseline is {baseline}, but the latest release is "
                f"{latest_release}; rebase and move the fragment before merging"
            )

        name_match = FRAGMENT_NAME.fullmatch(path.name)
        if name_match is None:
            allowed = "|".join(SECTIONS)
            raise ChangelogError(
                f"{path}: invalid fragment filename; expected "
                f"<number>.({allowed}).md"
            )

        number = int(name_match.group("number"))
        content = _read_utf8(path)
        if not content:
            raise ChangelogError(f"{path}: fragment must not be empty")
        if not content.startswith("- "):
            raise ChangelogError(f"{path}: fragment must start with a Markdown list item (`- `)")
        if re.search(rf"(?<!\d)#{number}(?!\d)", content) is None:
            raise ChangelogError(f"{path}: fragment must reference #{number}")

        fragments.append(
            Fragment(
                path=path,
                number=number,
                section=name_match.group("section"),
                content=content,
            )
        )

    seen: dict[int, Path] = {}
    for fragment in fragments:
        previous = seen.setdefault(fragment.number, fragment.path)
        if previous != fragment.path:
            raise ChangelogError(
                f"{fragment.path}: #{fragment.number} already has fragment {previous}; "
                "use one fragment per issue or pull request"
            )
    return sorted(
        fragments,
        key=lambda fragment: (SECTIONS.index(fragment.section), fragment.number),
    )


def _unreleased_bounds(changelog: str) -> tuple[int, int]:
    heading = UNRELEASED_HEADING.search(changelog)
    if heading is None:
        raise ChangelogError("CHANGELOG.md: missing `## [Unreleased]` heading")
    next_release = re.search(r"^## \[", changelog[heading.end() :], re.MULTILINE)
    end = len(changelog) if next_release is None else heading.end() + next_release.start()
    return heading.end(), end


def _section_bounds(changelog: str, section: str) -> tuple[int, int] | None:
    unreleased_start, unreleased_end = _unreleased_bounds(changelog)
    unreleased = changelog[unreleased_start:unreleased_end]
    wanted = SECTION_TITLES[section]
    headings = list(SECTION_HEADING.finditer(unreleased))
    for index, heading in enumerate(headings):
        if heading.group("title") != wanted:
            continue
        start = unreleased_start + heading.end()
        end = (
            unreleased_start + headings[index + 1].start()
            if index + 1 < len(headings)
            else unreleased_end
        )
        return start, end
    return None


def _normalise_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _line_ending(content: str) -> str:
    return "\r\n" if content.endswith("\r\n") else "\n"


def _separate_before(prefix: str) -> str:
    normalised = _normalise_newlines(prefix)
    if normalised.endswith("\n\n"):
        return ""
    newline = _line_ending(prefix)
    if normalised.endswith("\n"):
        return newline
    return newline * 2


def _separate_after(payload: str) -> str:
    normalised = _normalise_newlines(payload)
    if normalised.endswith("\n\n"):
        return ""
    newline = _line_ending(payload)
    if normalised.endswith("\n"):
        return newline
    return newline * 2


def _join_verbatim(contents: list[str]) -> str:
    payload = ""
    for content in contents:
        if payload:
            payload += _separate_after(payload)
        payload += content
    return payload


def _append_to_section(changelog: str, section: str, contents: list[str]) -> str:
    bounds = _section_bounds(changelog, section)
    if bounds is None:
        raise AssertionError(f"missing section {section}")
    _, end = bounds
    payload = _join_verbatim(contents)
    prefix = changelog[:end]
    insertion = _separate_before(prefix) + payload + _separate_after(payload)
    return prefix + insertion + changelog[end:]


def _insert_section(changelog: str, section: str, contents: list[str]) -> str:
    unreleased_start, unreleased_end = _unreleased_bounds(changelog)
    unreleased = changelog[unreleased_start:unreleased_end]
    target_order = SECTIONS.index(section)
    insertion_at = unreleased_end

    for heading in SECTION_HEADING.finditer(unreleased):
        title = heading.group("title")
        known_section = next(
            (key for key, known_title in SECTION_TITLES.items() if known_title == title), None
        )
        if known_section is not None and SECTIONS.index(known_section) > target_order:
            insertion_at = unreleased_start + heading.start()
            break

    payload = _join_verbatim(contents)
    prefix = changelog[:insertion_at]
    heading = f"### {SECTION_TITLES[section]}\n\n"
    insertion = _separate_before(prefix) + heading + payload + _separate_after(payload)
    return prefix + insertion + changelog[insertion_at:]


def _apply_fragments(changelog: str, fragments: list[Fragment]) -> str:
    for section in SECTIONS:
        pending: list[str] = []
        for fragment in (item for item in fragments if item.section == section):
            bounds = _section_bounds(changelog, section)
            if bounds is not None:
                section_text = changelog[bounds[0] : bounds[1]]
                if _normalise_newlines(fragment.content) in _normalise_newlines(section_text):
                    continue
                if re.search(rf"(?<!\d)#{fragment.number}(?!\d)", section_text):
                    raise ChangelogError(
                        f"CHANGELOG.md: #{fragment.number} already exists in "
                        f"{SECTION_TITLES[section]} with different text"
                    )
            pending.append(fragment.content)

        if not pending:
            continue
        if _section_bounds(changelog, section) is None:
            changelog = _insert_section(changelog, section, pending)
        else:
            changelog = _append_to_section(changelog, section, pending)
    return changelog


def _atomic_write(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def assemble(root: Path | str = ".") -> list[Path]:
    """Merge every validated fragment into Unreleased, then consume the fragments."""
    root = Path(root)
    changelog_path = root / "CHANGELOG.md"
    fragments = validate_fragments(root)
    if not fragments:
        return []

    original = _read_utf8(changelog_path)
    assembled = _apply_fragments(original, fragments)
    if assembled != original:
        _atomic_write(changelog_path, assembled)

    consumed = [fragment.path for fragment in fragments]
    for path in consumed:
        path.unlink()
    for directory in sorted({path.parent for path in consumed}, reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return consumed


def check_empty(root: Path | str = ".") -> None:
    """Fail a release if any changelog fragment remains unassembled."""
    fragments = validate_fragments(root)
    if fragments:
        paths = ", ".join(str(fragment.path) for fragment in fragments)
        raise ChangelogError(
            f"unassembled changelog fragments remain ({paths}); "
            "assemble them before tagging with `python scripts/assemble_changelog.py`"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--check-empty",
        action="store_true",
        help="fail when a release would leave any fragment behind",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_empty:
            check_empty(args.root)
            sys.stdout.write("No unassembled changelog fragments remain.\n")
        else:
            consumed = assemble(args.root)
            sys.stdout.write(f"Assembled {len(consumed)} changelog fragment(s).\n")
    except ChangelogError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
