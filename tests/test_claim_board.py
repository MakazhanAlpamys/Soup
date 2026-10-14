"""scripts/claim_board.py: classification, ordering and rendering with fake API data."""

import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "claim_board.py"
_SPEC = importlib.util.spec_from_file_location("claim_board", _PATH)
board = importlib.util.module_from_spec(_SPEC)
sys.modules["claim_board"] = board
_SPEC.loader.exec_module(board)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def issue(number, labels=("help wanted",), title="Some title"):
    return {"number": number, "title": title, "labels": [{"name": n} for n in labels]}


def comment(login, body, created):
    return {"user": {"login": login}, "body": body, "created_at": created}


def pull(number, login, body):
    return {"number": number, "user": {"login": login}, "title": "x", "body": body}


def test_free():
    row = board.classify(issue(5, ("help wanted", "difficulty:3")), [], [], NOW)
    assert row["kind"] == "free" and row["state"] == "free" and row["difficulty"] == 3
    assert row["pr"] is None and row["quiet"] is None


def test_claimed_confirmed_with_pr_and_quiet_days():
    comments = [
        comment("alice", "I'd like to take this", "2026-10-01T10:00:00Z"),
        comment("MakazhanAlpamys", "Confirmed, it is yours", "2026-10-02T10:00:00Z"),
        comment("alice", "working on it", "2026-10-04T12:00:00Z"),
    ]
    prs = [pull(90, "bob", "Fixes #7"), pull(91, "alice", "Refs #7")]
    row = board.classify(issue(7, ("claimed",)), comments, prs, NOW)
    assert row["state"] == "claimed by @alice since 2026-10-01"
    assert row["pr"] == 91
    assert row["quiet"] == 10


def test_claimed_unconfirmed_is_labelled():
    comments = [comment("carol", "Can I claim this?", "2026-10-10T00:00:00Z")]
    row = board.classify(issue(8, ("claimed",)), comments, [], NOW)
    assert row["state"] == "claimed by @carol since 2026-10-10 (unconfirmed)"


def test_claimed_label_without_nameable_claimant():
    row = board.classify(issue(9, ("claimed",)), [], [], NOW)
    assert row["state"] == "claimed (see thread)"


def test_blocked_by_label():
    row = board.classify(issue(10, ("help wanted", "infra-blocked")), [], [], NOW)
    assert row["kind"] == "blocked" and row["state"] == "blocked: infra-blocked"


def test_blocked_after_issue_uses_last_maintainer_comment_only():
    comments = [
        comment("MakazhanAlpamys", "Start after #11", "2026-10-01T00:00:00Z"),
        comment("MakazhanAlpamys", "Unblocked now", "2026-10-05T00:00:00Z"),
    ]
    assert board.classify(issue(12), comments, [], NOW)["kind"] == "free"
    comments[1]["body"] = "Please wait, after #13 lands"
    assert board.classify(issue(12), comments, [], NOW)["state"] == "blocked: after #13"


def test_quiet_days_fixed_now():
    comments = [comment("dan", "hi", "2026-10-14T11:00:00Z")]
    assert board.quiet_days(comments, "dan", NOW) == 0
    assert board.quiet_days(comments, "eve", NOW) is None
    assert board.quiet_days(comments, None, NOW) is None


def test_ordering_is_deterministic():
    rows = [
        {"number": 4, "difficulty": 5, "kind": "blocked", "quiet": None},
        {"number": 3, "difficulty": 2, "kind": "claimed", "quiet": 3},
        {"number": 2, "difficulty": 4, "kind": "claimed", "quiet": 9},
        {"number": 6, "difficulty": 4, "kind": "free", "quiet": None},
        {"number": 1, "difficulty": 1, "kind": "free", "quiet": None},
    ]
    assert [r["number"] for r in board.sort_rows(rows)] == [1, 6, 2, 3, 4]
    assert board.sort_rows(rows) == board.sort_rows(list(reversed(rows)))


def test_render_is_plain_markdown_and_truncates():
    row = board.classify(issue(1, title="x" * 100 + "|"), [], [], NOW)
    page = board.render([row], NOW)
    assert "\x1b" not in page
    assert "Generated: 2026-10-14 12:00 UTC" in page
    assert "CONTRIBUTING.md#how-work-is-queued" in page
    cell = page.splitlines()[-1].split("|")[2].strip()
    assert len(cell) == board.TITLE_WIDTH and cell.endswith("...")
    assert not re.search("[\\u0400-\\u04FF]", page)


def test_strip_stamp_ignores_generation_time_only():
    first = board.render([], NOW)
    later = board.render([], datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert first != later and board.strip_stamp(first) == board.strip_stamp(later)


def test_api_error_exits_2(monkeypatch, capsys):
    def boom(args):
        raise board.BoardError("nope")

    monkeypatch.setattr(board, "gh_json", boom)
    assert board.main([]) == 2
    assert "claim_board: nope" in capsys.readouterr().err
