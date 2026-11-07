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
    r"^\s*confirmed\b|\b(?:is|are|it['’]s) yours\b", re.IGNORECASE | re.MULTILINE
)
_NAMED_CONFIRM_RE = re.compile(r"^\s*confirmed\s*[,!:]?\s*@([a-z0-9-]+)\b", re.IGNORECASE)
_LEADING_LOGIN_RE = re.compile(r"^\s*@([a-z0-9-]+)\b([^@\n]*)", re.IGNORECASE)
_DIRECT_CONFIRM_RE = re.compile(
    r"^\s*(?:confirmed(?:[,.!]|$)|(?:it(?: is|['’]s)|this is) yours\b)", re.IGNORECASE
)
_CLASSIC_TAKE_RE = re.compile(
    r"\b(?:i'd like to take|i would like to take|i'll take|i will take|take this"
    r"|claim this|i claim)\b",
    re.IGNORECASE,
)
_TAKE_RE = re.compile(_CLASSIC_TAKE_RE.pattern + r"|^\s*(?:i am )?taking\b", re.IGNORECASE)
_SCOPE_RE = re.compile(r"\b(item|fix)[ -](\d+)\b", re.IGNORECASE)
_TERMINAL_RE = re.compile(
    r"\b(?:is|was|has been) (?:completed|fulfilled|released)\b|\bis now done\b",
    re.IGNORECASE,
)
_TERMINAL_SUBJECT_RE = re.compile(
    r"^(?:(?:status|claim metadata) update:\s*)?(?:the\s+)?(?:accepted\s+)?"
    r"(?:claim\b|(?:item|fix)[ -]\d+\b|[a-z-]+\s+slice\b)",
    re.IGNORECASE,
)
_CLEAR_ALL_RE = re.compile(
    r"^(?:this issue is free to take|no remaining slice has an accepted claim)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(r"\b(?:if|unless|once|when|only after|as long as)\b", re.IGNORECASE)
_LINK_WORDS = r"(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?|refs?|part of)"
_AFTER_RE = re.compile(r"\bafter\s+#(\d+)", re.IGNORECASE)
_DIFFICULTY_RE = re.compile(r"^difficulty:(\d+)$")
_ClaimTarget = tuple[str | None, tuple[str, str] | None]
_ACCEPT_EVENT = 0
_UNCERTAIN_ACCEPT_EVENT = 1
_TERMINAL_EVENT = 2
_DECLINE_EVENT = 3


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


def _named_confirmation(body: str, issue_number: int) -> tuple[str, bool] | None:
    """Return the addressee and whether their acceptance states ownership."""
    named = _NAMED_CONFIRM_RE.match(body)
    if named:
        acceptance = body[named.end() :].split("@", 1)[0].split("\n", 1)[0]
        owns = bool(re.search(r"\b(?:is|are|it['’]s) yours\b", acceptance, re.IGNORECASE))
        return named.group(1), owns
    leading = _LEADING_LOGIN_RE.match(body)
    if leading and re.search(
        rf"(?:(?:#{issue_number}\b|\bthis issue)\s+is|it(?:['’]s|\s+is)) yours\b",
        leading.group(2), re.IGNORECASE,
    ):
        return leading.group(1), True
    thanks = re.match(r"^\s*thanks\s*,?\s+@([a-z0-9-]+)\b([^@]*)", body, re.IGNORECASE)
    if thanks and re.search(r"\b(?:item|fix) \d+ is yours\b", thanks.group(2), re.IGNORECASE):
        return thanks.group(1), True
    return None


def _claim_paragraphs(body: str) -> list[str]:
    """Ignore quoted examples; only the author's own prose can change a claim."""
    lines, fenced = [], False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and not line.lstrip().startswith(">"):
            lines.append(re.sub(r'"[^"\n]*"', "", line))
    parts = re.split(
        r"\n\s*\n|(?<=[.!?])\s+(?=confirmed\s*[,!:]?\s*@)",
        "\n".join(lines), flags=re.IGNORECASE,
    )
    return [part.strip() for part in parts if part.strip()]


def _scope(body: str) -> tuple[str, str] | None:
    match = _SCOPE_RE.search(body)
    return (match.group(1).lower(), match.group(2)) if match else None


def _acceptance(
    body: str, earlier: list[dict], number: int
) -> tuple[str | None, str | None, tuple[str, str] | None] | None:
    facts = re.split(r"(?<=[.!?;])\s+|\n", body)
    if not any(
        _CONFIRM_RE.search(fact) and not fact.endswith("?") and not _CONDITIONAL_RE.search(fact)
        for fact in facts
    ):
        return None
    named = _named_confirmation(body, number)
    if named is not None:
        login, owns = named
        matches = [item for item in earlier if _login(item).casefold() == login.casefold()]
        requests = [item for item in matches if _TAKE_RE.search(item.get("body", ""))]
        if not owns and not requests:
            return None
        if matches:
            item = (requests or matches)[-1]
            return _login(item), item["created_at"][:10], _scope(body) or _scope(item["body"])
        return None, None, _scope(body)
    if _DIRECT_CONFIRM_RE.match(body) and earlier:
        previous = earlier[-1]
        if _TAKE_RE.search(previous.get("body", "")):
            return _login(previous), previous["created_at"][:10], _scope(previous["body"])
    if any(_TAKE_RE.search(item.get("body", "")) for item in earlier):
        return None, None, _scope(body)
    return None


def _terminal_targets(body: str) -> list[_ClaimTarget]:
    targets: list[_ClaimTarget] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", body):
        if sentence.endswith("?") or _CONDITIONAL_RE.search(sentence):
            continue
        if _CLEAR_ALL_RE.search(sentence):
            targets.append((None, None))
        elif _TERMINAL_SUBJECT_RE.match(sentence) and _TERMINAL_RE.search(sentence):
            named = re.search(r"@([a-z0-9-]+)\b", sentence, re.IGNORECASE)
            targets.append((named.group(1).casefold() if named else None, _scope(sentence)))
    return targets


def _declined_requests(body: str, number: int) -> list[_ClaimTarget]:
    body = re.sub(r"`+[^`\n]*`+", "", body)
    addressed = re.match(
        r"^\s*(?:(?:thank you|thanks|confirmed)\s*[,!:]?\s*)?@([a-z0-9-]+)\b",
        body, re.IGNORECASE,
    )
    if addressed is None:
        return []
    login = addressed.group(1).casefold()
    for fact in re.split(r"(?<=[.!?;])\s+|\n", body[addressed.end():].lstrip(" ,:!")):
        fact = fact.strip()
        if fact.endswith("?") or _CONDITIONAL_RE.search(fact):
            continue
        if "@" in fact and not re.search(rf"@{re.escape(login)}\b", fact, re.IGNORECASE):
            continue
        if re.match(
            rf"(?:\b(?:this issue|it)|#{number}\b)\s+is not yours\b|"
            rf"(?:so\s+)?(?:please\s+)?do not open (?:a )?(?:separate|competing) "
            rf"pull request for #{number}\b",
            fact, re.IGNORECASE,
        ):
            return [(login, None)]
    return []


def _retained_pr(body: str, prs: list[dict], number: int, when: str) -> dict | None:
    body = re.sub(r"`+[^`\n]*`+", "", body)
    # A later clause can withdraw the apparent assertion; keep the whole paragraph.
    context = re.sub(r"@[a-z0-9-]+", "", body, flags=re.IGNORECASE)
    if re.search(
        r"\b(?:not|no longer|old|earlier|previous|withdrawn|superseded|incorrect)\b",
        context, re.IGNORECASE,
    ):
        return None
    for fact in re.split(r"(?<=[.!?;])\s+|\n", body):
        if fact.endswith("?") or _CONDITIONAL_RE.search(fact):
            continue
        retained = re.search(
            rf"(?:^|,\s+so\s+)(?:#{number}\b|this issue)\s+(?:stays|remains) with "
            r"@([a-z0-9-]+)\b", fact, re.IGNORECASE,
        )
        if retained is None:
            continue
        login = retained.group(1)
        matches = [
            pr for pr in prs
            if _login(pr).casefold() == login.casefold() and pr.get("state") == "open"
            and pr.get("created_at") and pr["created_at"] <= when
            and re.search(
                rf"#{pr['number']}\b\s+by\s+@{re.escape(login)}\b", body, re.IGNORECASE
            )
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _claim_state(
    comments: list[dict], number: int, prs: list[dict] | None = None
) -> tuple[str | None, str | None, bool, bool, int | None]:
    """Reduce trusted owner facts, keeping scoped claims until explicitly completed/released."""
    events: list[tuple[str, int, int, int, str | dict | list[_ClaimTarget]]] = []
    for index, item in enumerate(sorted(comments, key=lambda entry: entry["created_at"])):
        if _login(item) == MAINTAINER:
            created, updated = item["created_at"], item.get("updated_at", item["created_at"])
            terminal_at = created
            seen = set()
            paragraphs = _claim_paragraphs(item.get("body", ""))
            declined = _declined_requests("\n\n".join(paragraphs), number)
            if declined:
                events.append((created, index, -1, _DECLINE_EVENT, declined))
            for part, body in enumerate(paragraphs):
                named = _named_confirmation(body, number)
                requests = [
                    reply for reply in comments
                    if named and _login(reply).casefold() == named[0].casefold()
                    and _TAKE_RE.search(reply.get("body", ""))
                    and (_scope(body) is None or _scope(reply["body"]) == _scope(body))
                    and reply["created_at"] <= updated
                ]
                # Only a request that postdates creation establishes an appended acceptance.
                # An unrelated edit must not replay an old acceptance, in any paragraph.
                appended = bool(requests) and not any(
                    reply["created_at"] <= created for reply in requests
                )
                key = (named[0].casefold(), _scope(body)) if named else None
                uncertain = key in seen and any(
                    reply["created_at"] > created for reply in requests
                )
                accepted_at = updated if appended or uncertain else created
                events.append((accepted_at,
                               index, part,
                               _UNCERTAIN_ACCEPT_EVENT if uncertain else _ACCEPT_EVENT, body))
                seen.add(key)
                # An edit timestamp cannot date an old release. Keep its creation time,
                # except when it follows an appended acceptance in this same comment.
                terminal_at = max(terminal_at, accepted_at)
                targets = _terminal_targets(body)
                if targets:
                    events.append((terminal_at, index, part, _TERMINAL_EVENT, targets))
        elif _login(item):
            events.append((item["created_at"], index, -1, _ACCEPT_EVENT, item))
    replies: list[dict] = []
    active: dict[_ClaimTarget, tuple[str | None, str | None, int | None]] = {}
    terminal = False
    for when, _, part, action, value in sorted(events):
        if isinstance(value, dict):
            replies.append(value)
            continue
        if isinstance(value, str):
            retained = _retained_pr(value, prs or [], number, when)
            if retained is not None:
                login = _login(retained)
                active[(login.casefold(), None)] = (
                    login, retained["created_at"][:10], retained["number"]
                )
                continue
            if _declined_requests(value, number):
                continue
            if action == _UNCERTAIN_ACCEPT_EVENT and terminal:
                active[(None, _scope(value))] = (None, None, None)
                continue
            acceptance = _acceptance(value, replies, number)
            if acceptance is not None:
                login, since, scope = acceptance
                active[(login.casefold() if login else None, scope)] = (login, since, None)
            continue
        for login, scope in value:
            def matches(author: str | None, subject: tuple[str, str] | None) -> bool:
                return (login is None or author == login) and (scope is None or subject == scope)

            if action != _DECLINE_EVENT:
                terminal = True
                active = {key: claim for key, claim in active.items() if not matches(*key)}
            replies = [
                item for item in replies
                if not matches(_login(item).casefold(), _scope(item.get("body", "")))
            ]
    owners = {login for login, _ in active}
    if active:
        if None in owners or len(owners) != 1:
            return None, None, True, terminal, None
        login, since, pr = min(active.values(), key=lambda claim: claim[1] or "")
        return login, since, True, terminal, pr
    for item in replies:
        if _TAKE_RE.search(item.get("body", "")):
            return _login(item), item["created_at"][:10], False, terminal, None
    return None, None, False, terminal, None


def find_claimant(comments: list[dict], issue_number: int) -> tuple[str | None, str | None, bool]:
    """Return (login, since date, confirmed); several active owners need "see thread"."""
    login, since, confirmed, _, _ = _claim_state(comments, issue_number)
    return login, since, confirmed


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
    login, since, confirmed, terminal, claimed_pr = _claim_state(comments, issue["number"], prs)
    block = blocker(issue, comments, is_open)
    claimed = confirmed or ("claimed" in labels and not terminal)
    if block:
        kind, state = "blocked", f"blocked: {block}"
    elif claimed and login:
        kind = "claimed"
        # The login sits in a code span: the board names the claimant without a mention, so
        # a daily rewrite of the issue never notifies anyone.
        state = f"claimed by `@{login}` since {since}" + ("" if confirmed else " (unconfirmed)")
    elif claimed:
        kind, state = "claimed", "claimed (see thread)"
    else:
        kind, state = "free", "free"
    metrics = claimed or (not terminal and any(
        _login(item) == login and _CLASSIC_TAKE_RE.search(item.get("body", ""))
        for item in comments
    ))
    return {
        "number": issue["number"],
        "title": issue["title"],
        "difficulty": difficulty(issue),
        "kind": kind,
        "state": state,
        "pr": (claimed_pr or find_pr(prs, issue["number"], login))
        if kind != "free" and metrics else None,
        "quiet": quiet_days(comments, login, now) if kind != "free" and metrics else None,
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
        help="write nothing; exit 1 when this file differs from the live board (time line ignored)",
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
