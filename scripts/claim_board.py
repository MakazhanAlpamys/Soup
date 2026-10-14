"""Render docs/claims.md: which open issues are free, claimed, or blocked.

Reads the open issues labelled ``help wanted``, ``claimed`` or ``infra-blocked`` through
``gh api`` (stdlib only) and writes one plain-markdown table. The claim rules are in
CONTRIBUTING.md ("How work is queued"); this page only reports them.

    python scripts/claim_board.py                         # print to stdout
    python scripts/claim_board.py --write docs/claims.md  # rewrite the page
    python scripts/claim_board.py --check docs/claims.md  # exit 1 when it differs

The generation-time line is ignored by ``--check``. Exit codes: 0 ok, 1 stale page, 2 an
API or usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = "MakazhanAlpamys/Soup"
MAINTAINER = "MakazhanAlpamys"
LABELS = ("help wanted", "claimed", "infra-blocked")
DEFAULT_PAGE = "docs/claims.md"
TITLE_WIDTH = 60
GENERATED_PREFIX = "Generated: "

_CONFIRM_RE = re.compile(r"\b(confirmed|is yours|yours)\b", re.IGNORECASE)
_TAKE_RE = re.compile(r"take this|i'd like to take|i would like to take|claim", re.IGNORECASE)
_AFTER_RE = re.compile(r"\bafter\s+#(\d+)", re.IGNORECASE)
_REF_RE = re.compile(r"#(\d+)")
_DIFFICULTY_RE = re.compile(r"^difficulty:(\d+)$")


class BoardError(Exception):
    """An API or usage problem; main() turns it into exit code 2."""


def label_names(issue: dict) -> list[str]:
    return [lab["name"] if isinstance(lab, dict) else lab for lab in issue.get("labels", [])]


def difficulty(issue: dict) -> int | None:
    for name in label_names(issue):
        match = _DIFFICULTY_RE.match(name)
        if match:
            return int(match.group(1))
    return None


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _login(comment: dict) -> str:
    return (comment.get("user") or {}).get("login", "")


def find_claimant(comments: list[dict]) -> tuple[str | None, str | None, bool]:
    """Return (login, since date, confirmed).

    Confirmed: the author of the comment a maintainer answered with a confirmation.
    Otherwise the author of the first non-maintainer comment containing a take phrase
    (unconfirmed). (None, None, False) when nobody can be named.
    """
    ordered = sorted(comments, key=lambda item: item["created_at"])
    for index, comment in enumerate(ordered):
        if _login(comment) != MAINTAINER or not _CONFIRM_RE.search(comment.get("body", "")):
            continue
        for earlier in reversed(ordered[:index]):
            if _login(earlier) not in (MAINTAINER, ""):
                return _login(earlier), earlier["created_at"][:10], True
    for comment in ordered:
        if _login(comment) not in (MAINTAINER, "") and _TAKE_RE.search(comment.get("body", "")):
            return _login(comment), comment["created_at"][:10], False
    return None, None, False


def blocker(issue: dict, comments: list[dict]) -> str | None:
    """`infra-blocked` label, or 'after #N' from the last maintainer comment."""
    if "infra-blocked" in label_names(issue):
        return "infra-blocked"
    for comment in sorted(comments, key=lambda item: item["created_at"], reverse=True):
        if _login(comment) == MAINTAINER:
            match = _AFTER_RE.search(comment.get("body", ""))
            return f"after #{match.group(1)}" if match else None
    return None


def quiet_days(comments: list[dict], login: str | None, now: datetime) -> int | None:
    if not login:
        return None
    times = [parse_time(c["created_at"]) for c in comments if _login(c) == login]
    if not times:
        return None
    return max((now - max(times)).days, 0)


def find_pr(prs: list[dict], issue_number: int, login: str | None) -> int | None:
    """Lowest open PR number by `login` whose title or body references the issue."""
    if not login:
        return None
    hits = []
    for pr in prs:
        if (pr.get("user") or {}).get("login") != login:
            continue
        text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
        if issue_number in {int(num) for num in _REF_RE.findall(text)}:
            hits.append(pr["number"])
    return min(hits) if hits else None


def classify(issue: dict, comments: list[dict], prs: list[dict], now: datetime) -> dict:
    labels = label_names(issue)
    login, since, confirmed = find_claimant(comments)
    block = blocker(issue, comments)
    claimed = "claimed" in labels or (login is not None and confirmed)
    if block:
        kind, state = "blocked", f"blocked: {block}"
    elif claimed and login:
        kind = "claimed"
        state = f"claimed by @{login} since {since}" + ("" if confirmed else " (unconfirmed)")
    elif claimed:
        kind, state = "claimed", "claimed (see thread)"
    else:
        kind, state = "free", "free"
    return {
        "number": issue["number"],
        "title": issue["title"],
        "difficulty": difficulty(issue),
        "kind": kind,
        "state": state,
        "pr": find_pr(prs, issue["number"], login) if kind != "free" else None,
        "quiet": quiet_days(comments, login, now) if kind != "free" else None,
    }


def sort_rows(rows: list[dict]) -> list[dict]:
    order = {"free": 0, "claimed": 1, "blocked": 2}

    def key(row: dict) -> tuple:
        diff = row["difficulty"] if row["difficulty"] is not None else 99
        quiet = -(row["quiet"] if row["quiet"] is not None else -1)
        if row["kind"] == "free":
            return (0, diff, row["number"])
        return (order[row["kind"]], quiet, row["number"])

    return sorted(rows, key=key)


def short_title(title: str) -> str:
    title = " ".join(title.replace("|", "/").split())
    return title if len(title) <= TITLE_WIDTH else title[: TITLE_WIDTH - 3] + "..."


def render(rows: list[dict], now: datetime) -> str:
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Claim board",
        "",
        f"{GENERATED_PREFIX}{stamp}",
        "",
        "Open issues labelled `help wanted`, `claimed` or `infra-blocked`. A comment on the "
        "issue is the claim and a maintainer confirms it. A claim with no pull request and no "
        "word from the claimant for 7 days gets a check-in; 7 more quiet days release it. "
        "A pull request waiting on its author for 14 days gets a ping, and 7 more days close "
        "it. See [How work is queued](../CONTRIBUTING.md#how-work-is-queued). This page is "
        "generated by `scripts/claim_board.py`; do not edit it by hand.",
        "",
        "| Issue | Title | Difficulty | State | PR | Quiet |",
        "|-------|-------|------------|-------|----|-------|",
    ]
    for row in sort_rows(rows):
        diff = str(row["difficulty"]) if row["difficulty"] is not None else "-"
        pr = f"#{row['pr']}" if row["pr"] else "-"
        quiet = f"quiet {row['quiet']} d" if row["quiet"] is not None else "-"
        lines.append(
            f"| #{row['number']} | {short_title(row['title'])} | {diff} | "
            f"{row['state']} | {pr} | {quiet} |"
        )
    if not rows:
        lines.append("| - | No open issues carry these labels | - | - | - | - |")
    return "\n".join(lines) + "\n"


def strip_stamp(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith(GENERATED_PREFIX))


def gh_json(args: list[str]) -> list[dict]:
    try:
        proc = subprocess.run(
            ["gh", "api", *args], capture_output=True, text=True, check=False, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BoardError(f"could not run gh: {exc}") from exc
    if proc.returncode != 0:
        raise BoardError(f"gh api {args[0]} failed: {proc.stderr.strip() or proc.returncode}")
    text = proc.stdout.strip()
    items: list[dict] = []
    try:
        # --paginate prints one JSON array per page, back to back.
        decoder, pos = json.JSONDecoder(), 0
        while pos < len(text):
            value, pos = decoder.raw_decode(text, pos)
            items.extend(value if isinstance(value, list) else [value])
            while pos < len(text) and text[pos].isspace():
                pos += 1
    except ValueError as exc:
        raise BoardError(f"gh api {args[0]} returned invalid JSON: {exc}") from exc
    return items


def fetch(now: datetime) -> list[dict]:
    issues: dict[int, dict] = {}
    for label in LABELS:
        query = label.replace(" ", "%20")
        found = gh_json(
            [f"repos/{REPO}/issues?state=open&per_page=100&labels={query}", "--paginate"]
        )
        for item in found:
            if "pull_request" not in item:
                issues[item["number"]] = item
    prs = gh_json([f"repos/{REPO}/pulls?state=open&per_page=100", "--paginate"])
    rows = []
    for number in sorted(issues):
        comments = gh_json([f"repos/{REPO}/issues/{number}/comments?per_page=100", "--paginate"])
        rows.append(classify(issues[number], comments, prs, now))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the claim board page.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", metavar="PATH", help=f"write the page (usually {DEFAULT_PAGE})")
    mode.add_argument(
        "--check",
        nargs="?",
        const=DEFAULT_PAGE,
        metavar="PATH",
        help="write nothing; exit 1 when the committed page differs (time line ignored)",
    )
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    try:
        page = render(fetch(now), now)
    except BoardError as exc:
        print(f"claim_board: {exc}", file=sys.stderr)
        return 2
    if args.write:
        Path(args.write).write_text(page, encoding="utf-8", newline="\n")
        print(f"wrote {args.write}")
        return 0
    if args.check:
        path = Path(args.check)
        committed = path.read_text(encoding="utf-8") if path.exists() else None
        if committed is None or strip_stamp(committed) != strip_stamp(page):
            print(
                f"{args.check} is stale. Run `python scripts/claim_board.py --write "
                f"{args.check}` and commit the result."
            )
            return 1
        print(f"{args.check} is up to date.")
        return 0
    sys.stdout.write(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
