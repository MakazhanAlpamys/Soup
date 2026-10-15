"""Render the claim board: which open issues are free, claimed, or blocked.

Reads the open issues labelled ``help wanted``, ``claimed`` or ``infra-blocked`` through
``gh api`` (stdlib only) and writes one plain-markdown table. The board is not committed:
it is the body of the pinned, locked issue #1945, which a scheduled job rewrites with
``gh issue edit 1945 --body-file <file>``. The claim rules are in CONTRIBUTING.md ("How
work is queued"); this page only reports them.

    python scripts/claim_board.py                    # print to stdout
    python scripts/claim_board.py --write board.md   # write the page to a file
    python scripts/claim_board.py --check board.md   # exit 1 when the file differs

The generation-time line is ignored by ``--check``. Exit codes: 0 ok, 1 stale file, 2 an
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
BOARD_ISSUE = 1945
GUIDE_URL = f"https://github.com/{REPO}/blob/main/CONTRIBUTING.md#how-work-is-queued"
TITLE_WIDTH = 60
GENERATED_PREFIX = "Generated: "

_CONFIRM_RE = re.compile(
    r"^\s*confirmed\b|\b(?:is|are) yours\b", re.IGNORECASE | re.MULTILINE
)
_TAKE_RE = re.compile(
    r"\b(?:i'd like to take|i would like to take|i'll take|i will take|take this"
    r"|claim this|i claim)\b",
    re.IGNORECASE,
)
_LINK_WORDS = r"(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?|refs?|part of)"
_AFTER_RE = re.compile(r"\bafter\s+#(\d+)", re.IGNORECASE)
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


def _always_open(number: int) -> bool:
    return True


def blocker(issue: dict, comments: list[dict], is_open=_always_open) -> str | None:
    """`infra-blocked` label, or 'after #N' (N still open) in the last maintainer comment."""
    if "infra-blocked" in label_names(issue):
        return "infra-blocked"
    for comment in sorted(comments, key=lambda item: item["created_at"], reverse=True):
        if _login(comment) == MAINTAINER:
            match = _AFTER_RE.search(comment.get("body", ""))
            if match and is_open(int(match.group(1))):
                return f"after #{match.group(1)}"
            return None
    return None


def quiet_days(comments: list[dict], login: str | None, now: datetime) -> int | None:
    if not login:
        return None
    times = [parse_time(c["created_at"]) for c in comments if _login(c) == login]
    if not times:
        return None
    return max((now - max(times)).days, 0)


def find_pr(prs: list[dict], issue_number: int, login: str | None) -> int | None:
    """Lowest open PR by `login` whose body closes or refs the issue (not "see also")."""
    if not login:
        return None
    hits = []
    for pr in prs:
        if (pr.get("user") or {}).get("login") != login:
            continue
        link = re.compile(rf"\b{_LINK_WORDS}:?\s+#{issue_number}\b", re.IGNORECASE)
        if link.search(pr.get("body") or ""):
            hits.append(pr["number"])
    return min(hits) if hits else None


def classify(
    issue: dict, comments: list[dict], prs: list[dict], now: datetime, is_open=_always_open
) -> dict:
    labels = label_names(issue)
    login, since, confirmed = find_claimant(comments)
    block = blocker(issue, comments, is_open)
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
    """Untrusted text: one backtick code span, so no link, tag, mention or issue ref renders."""
    title = " ".join(title.replace("|", "/").replace("`", "'").split())
    if len(title) > TITLE_WIDTH:
        title = title[: TITLE_WIDTH - 3] + "..."
    return f"`{title}`"


def render(rows: list[dict], now: datetime) -> str:
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Claim board",
        "",
        f"{GENERATED_PREFIX}{stamp}",
        "",
        "Open issues labelled `help wanted`, `claimed` or `infra-blocked`. To claim one, "
        "comment on that issue itself; this issue is locked. A maintainer confirms the claim. "
        "A claim with no pull request and no word from the claimant for 7 days gets a "
        "check-in; 7 more quiet days release it. A pull request waiting on its author for 14 "
        f"days gets a ping, and 7 more days close it. See [How work is queued]({GUIDE_URL}). "
        "This issue body is regenerated daily by `scripts/claim_board.py`; an edit made here "
        "is overwritten.",
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
    states: dict[int, bool] = {}

    def is_open(number: int) -> bool:
        if number not in states:
            found_one = gh_json([f"repos/{REPO}/issues/{number}"])
            states[number] = bool(found_one) and found_one[0].get("state") == "open"
        return states[number]

    rows = []
    for number in sorted(issues):
        comments = gh_json([f"repos/{REPO}/issues/{number}/comments?per_page=100", "--paginate"])
        rows.append(classify(issues[number], comments, prs, now, is_open))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the claim board page.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", metavar="PATH", help="write the page to this file")
    mode.add_argument(
        "--check",
        metavar="PATH",
        help="write nothing; exit 1 when this file differs from the live board "
        "(time line ignored)",
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
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing is None or strip_stamp(existing) != strip_stamp(page):
            print(
                f"{args.check} is stale. Run `python scripts/claim_board.py --write "
                f"{args.check}` to rewrite it."
            )
            return 1
        print(f"{args.check} is up to date.")
        return 0
    sys.stdout.write(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
