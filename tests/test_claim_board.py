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


def _cell(page):
    return page.splitlines()[-1].split("|")[2].strip()


def test_render_is_plain_markdown_and_truncates():
    row = board.classify(issue(1, title="x" * 100), [], [], NOW)
    page = board.render([row], NOW)
    assert "\x1b" not in page
    assert "Generated: 2026-10-14 12:00 UTC" in page
    assert "CONTRIBUTING.md#how-work-is-queued" in page
    cell = _cell(page)
    assert len(cell) == board.TITLE_WIDTH + 2 and cell.endswith("...`")
    assert not re.search("[\\u0400-\\u04FF]", page)


def test_pipe_in_title_becomes_slash():
    page = board.render([board.classify(issue(1, title="a|b"), [], [], NOW)], NOW)
    assert _cell(page) == "`a/b`"
    assert len(page.splitlines()[-1].split("|")) == 8


def test_title_cannot_form_markdown_or_mention():
    title = "[link](x) <b> @user #1 back`tick \\ end"
    page = board.render([board.classify(issue(1, title=title), [], [], NOW)], NOW)
    cell = _cell(page)
    assert cell == "`[link](x) <b> @user #1 back'tick \\ end`"
    assert cell.count("`") == 2


def test_blocked_outranks_claimed():
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment("MakazhanAlpamys", "Confirmed", "2026-10-02T10:00:00Z"),
    ]
    row = board.classify(issue(3, ("claimed", "infra-blocked")), comments, [], NOW)
    assert row["kind"] == "blocked" and row["state"] == "blocked: infra-blocked"
    later = comments + [comment("MakazhanAlpamys", "after #99 lands", "2026-10-03T10:00:00Z")]
    row = board.classify(issue(3, ("claimed",)), later, [], NOW)
    assert row["kind"] == "blocked" and row["state"] == "blocked: after #99"


def test_confirm_regex_cases():
    assert not board._CONFIRM_RE.search("CI confirmed green")
    assert not board._CONFIRM_RE.search("this is not yours")
    assert board._CONFIRM_RE.search("Confirmed, go ahead")
    assert board._CONFIRM_RE.search("ok\nConfirmed")
    assert board._CONFIRM_RE.search("it is yours")
    assert board._CONFIRM_RE.search("they are yours")


def test_take_regex_cases():
    for text in ("unclaimed", "claimed by bob", "I claimed it"):
        assert not board._TAKE_RE.search(text), text
    for text in ("I'd like to take", "I'll take it", "take this", "claim this", "I claim"):
        assert board._TAKE_RE.search(text), text


def test_after_blocker_only_when_target_open():
    comments = [comment("MakazhanAlpamys", "after #20", "2026-10-01T00:00:00Z")]
    closed = board.classify(issue(4), comments, [], NOW, lambda n: False)
    assert closed["kind"] == "free"
    opened = board.classify(issue(4), comments, [], NOW, lambda n: True)
    assert opened["state"] == "blocked: after #20"


def test_pr_linking_needs_closing_keyword_and_claimant():
    comments = [comment("alice", "I'll take this", "2026-10-01T00:00:00Z")]
    prs = [
        pull(50, "alice", "see also #6"),
        pull(51, "bob", "Closes #6"),
        pull(52, "alice", "Part of #6"),
        pull(53, "alice", "Resolves #66"),
    ]
    assert board.classify(issue(6, ("claimed",)), comments, prs, NOW)["pr"] == 52
    assert board.classify(issue(6, ("claimed",)), comments, prs[:2], NOW)["pr"] is None


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


def _stub_fetch(monkeypatch):
    rows = [board.classify(issue(1), [], [], NOW)]
    monkeypatch.setattr(board, "fetch", lambda now: rows)
    return board.render(rows, NOW)


def test_check_detects_drift_and_match(monkeypatch, tmp_path, capsys):
    page = _stub_fetch(monkeypatch)
    path = tmp_path / "claims.md"
    path.write_text(page, encoding="utf-8")
    assert board.main(["--check", str(path)]) == 0
    path.write_text(page.replace("Some title", "Other"), encoding="utf-8")
    assert board.main(["--check", str(path)]) == 1
    assert "is stale" in capsys.readouterr().out


def test_write_keeps_existing_file_when_gh_fails(monkeypatch, tmp_path):
    def boom(now):
        raise board.BoardError("gh down")

    monkeypatch.setattr(board, "fetch", boom)
    path = tmp_path / "claims.md"
    path.write_bytes(b"old page\n")
    assert board.main(["--write", str(path)]) == 2
    assert path.read_bytes() == b"old page\n"
