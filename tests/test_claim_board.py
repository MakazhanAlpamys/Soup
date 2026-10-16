"""scripts/claim_board.py: classification, ordering and rendering with fake API data."""

import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "scripts" / "claim_board.py"
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
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91
    assert row["quiet"] == 10


def test_claimed_unconfirmed_is_labelled():
    comments = [comment("carol", "Can I claim this?", "2026-10-10T00:00:00Z")]
    row = board.classify(issue(8, ("claimed",)), comments, [], NOW)
    assert row["state"] == "claimed by `@carol` since 2026-10-10 (unconfirmed)"


def test_the_rendered_board_mentions_nobody():
    # The board is an issue body rewritten every day: a bare @login would notify the claimant.
    comments = [
        comment("alice", "I'd like to take this", "2026-10-01T10:00:00Z"),
        comment("MakazhanAlpamys", "Confirmed, it is yours", "2026-10-02T10:00:00Z"),
    ]
    page = board.render([board.classify(issue(7, ("claimed",)), comments, [], NOW)], NOW)
    assert "`@alice`" in page
    assert "@" not in re.sub(r"`[^`]*`", "", page)


def test_claimed_label_without_nameable_claimant():
    row = board.classify(issue(9, ("claimed",)), [], [], NOW)
    assert row["state"] == "claimed (see thread)"


@pytest.mark.parametrize("labels", [("help wanted",), ("help wanted", "claimed")])
def test_an_offer_on_other_issues_does_not_confirm_the_fallback(labels):
    """#1878: #1881 carries the issue; #1895's author is offered other work."""
    comments = [
        comment("Nari-Ab", "I'd like to claim this issue.", "2026-10-14T11:02:12Z"),
        comment(
            board.MAINTAINER,
            "Thanks @Nari-Ab. @Fire162 opened #1881 for this issue; it carries the issue. "
            "Your #1895 waits behind it as the fallback.",
            "2026-10-14T12:33:25Z",
        ),
        comment(
            "Nari-Ab",
            "#1895 is a draft fallback. Could you suggest a different issue?",
            "2026-10-14T12:40:23Z",
        ),
        comment(
            board.MAINTAINER,
            "Thanks @Nari-Ab. Three free issues: #1887, #1884, #1888. "
            "Post your plan on the issue you choose and the claim is yours now.",
            "2026-10-14T14:11:01Z",
        ),
    ]
    prs = [pull(1881, "Fire162", "Fixes #1878"), pull(1895, "Nari-Ab", "Fixes #1878")]
    row = board.classify(issue(1878, labels), comments, prs, NOW)
    assert row["kind"] == "claimed"
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


def test_latest_named_confirmation_replaces_a_completed_claim():
    """#1199: fix 3 was completed/released before fix 2 was confirmed again."""
    comments = [
        comment("yunaremaia", "I am taking proposed fix 3.", "2026-10-03T10:45:32Z"),
        comment(
            board.MAINTAINER,
            "Confirmed, @yunaremaia: proposed fix 3 is yours.",
            "2026-10-04T15:52:21Z",
        ),
        comment(
            board.MAINTAINER,
            "Fix 3 merged. The claim covered fix 3, which is now done, so I have taken "
            "`claimed` off: the two remaining fixes are free to take.",
            "2026-10-07T20:35:31Z",
        ),
        comment("Faisal-Fayaz", "Taking fix 2.", "2026-10-08T06:34:19Z"),
        comment(
            board.MAINTAINER,
            "Confirmed, @Faisal-Fayaz: fix 2 is yours. Marked the issue claimed again. "
            "Fix 1 stays unclaimed.",
            "2026-10-08T06:37:04Z",
        ),
    ]
    prs = [pull(1600, "yunaremaia", "Refs #1199"), pull(1700, "Faisal-Fayaz", "Refs #1199")]
    row = board.classify(issue(1199, ("help wanted", "claimed")), comments, prs, NOW)
    assert row["state"] == "claimed by `@Faisal-Fayaz` since 2026-10-08"
    assert row["pr"] == 1700 and row["quiet"] == 6


@pytest.mark.parametrize(
    "confirmation",
    [
        "Confirmed, @alice: it is yours.",
        "Confirmed, @alice: go ahead.",
    ],
)
def test_named_confirmation_binds_its_author_despite_an_interleaved_reply(confirmation):
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment("bob", "I can provide a fallback if needed.", "2026-10-02T09:00:00Z"),
        comment(
            board.MAINTAINER,
            confirmation,
            "2026-10-02T10:00:00Z",
        ),
    ]
    row = board.classify(issue(7), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91


@pytest.mark.parametrize("subject", ["#7", "this issue"])
def test_leading_named_same_issue_acceptance_excludes_a_rejected_fallback(subject):
    """The ordinary #1915 form addresses the winner before mentioning a fallback."""
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment("bob", "I can provide a fallback if needed.", "2026-10-02T09:00:00Z"),
        comment(
            board.MAINTAINER,
            f"@alice, thanks for the plan: {subject} is yours. @bob, please wait.",
            "2026-10-02T10:00:00Z",
        ),
    ]
    row = board.classify(issue(7), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91


@pytest.mark.parametrize("confirmation", ["Confirmed", "Confirmed, go ahead", "it is yours"])
def test_direct_confirmation_after_a_same_issue_take_request_still_names_it(confirmation):
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, confirmation, "2026-10-02T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [], NOW)
    assert row["state"] == "claimed by `@alice` since 2026-10-01"


def test_generic_confirmation_after_a_side_question_does_not_name_its_author():
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment("bob", "Which other issue is free?", "2026-10-02T09:00:00Z"),
        comment(board.MAINTAINER, "Confirmed", "2026-10-02T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [], NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


def test_a_named_confirmation_does_not_invent_an_author_from_public_text():
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment(
            board.MAINTAINER,
            "Confirmed, @someone-else: it is yours.",
            "2026-10-02T10:00:00Z",
        ),
    ]
    row = board.classify(issue(7), comments, [], NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize("with_reporter", [False, True])
@pytest.mark.parametrize(
    "confirmation",
    [
        "Confirmed on main: reproduced the bug.",
        "Confirmed, @alice: reproduced the bug.",
    ],
)
def test_confirming_bug_evidence_without_a_take_request_keeps_the_issue_free(
    with_reporter,
    confirmation,
):
    comments = (
        [comment("alice", "Here is the failing traceback.", "2026-10-01T10:00:00Z")]
        if with_reporter
        else []
    )
    comments.append(
        comment(
            board.MAINTAINER,
            confirmation,
            "2026-10-02T10:00:00Z",
        )
    )
    row = board.classify(issue(7), comments, [], NOW)
    assert row["kind"] == "free" and row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


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
    assert board.GUIDE_URL in page and "](../" not in page
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


def test_newlines_in_a_title_cannot_start_a_new_row_or_block():
    title = "a\n| x | y |\n# H\r\n- item\u2028z\u2029y\x85w"
    page = board.render([board.classify(issue(1, title=title), [], [], NOW)], NOW)
    assert page.count("\n") == board.render([], NOW).count("\n")
    assert _cell(page) == "`a / x / y / # H - item z y w`"
    assert len(page.splitlines()[-1].split("|")) == 8


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


def test_pr_link_needs_the_whole_issue_number_and_picks_the_lowest():
    comments = [comment("alice", "I'll take this", "2026-10-01T00:00:00Z")]
    prs = [
        pull(49, "alice", "Resolves #66"),
        pull(54, "alice", "Fixes #6"),
        pull(52, "alice", "Part of #6"),
    ]
    assert board.classify(issue(6, ("claimed",)), comments, prs, NOW)["pr"] == 52


def test_fetch_asks_for_open_issues_of_each_label_and_skips_pull_requests(monkeypatch):
    asked = []

    def fake(args):
        path = args[0]
        asked.append(path)
        if "/issues?" in path:
            label = parse_qs(urlsplit(path).query)["labels"][0]
            return {
                "help wanted": [issue(1), dict(issue(2), pull_request={})],
                "claimed": [issue(1), issue(3, ("claimed",))],
                "infra-blocked": [issue(4, ("infra-blocked",))],
            }[label]
        if "/pulls?" in path or path.endswith("/comments?per_page=100"):
            return []
        raise AssertionError(path)

    monkeypatch.setattr(board, "gh_json", fake)
    rows = board.fetch(NOW)
    assert [row["number"] for row in rows] == [1, 3, 4]
    assert [row["kind"] for row in rows] == ["free", "claimed", "blocked"]
    queries = [parse_qs(urlsplit(path).query) for path in asked if "/issues?" in path]
    assert len(queries) == 3
    assert {query["labels"][0] for query in queries} == {"help wanted", "claimed", "infra-blocked"}
    assert all(query["state"] == ["open"] for query in queries)


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
    path = tmp_path / "board.md"
    path.write_text(page, encoding="utf-8")
    assert board.main(["--check", str(path)]) == 0
    path.write_text(page.replace("Some title", "Other"), encoding="utf-8")
    assert board.main(["--check", str(path)]) == 1
    assert "is stale" in capsys.readouterr().out


def test_write_keeps_existing_file_when_gh_fails(monkeypatch, tmp_path):
    def boom(now):
        raise board.BoardError("gh down")

    monkeypatch.setattr(board, "fetch", boom)
    path = tmp_path / "board.md"
    path.write_bytes(b"old page\n")
    assert board.main(["--write", str(path)]) == 2
    assert path.read_bytes() == b"old page\n"


def test_page_is_an_issue_body_regenerated_daily():
    page = board.render([], NOW)
    assert "regenerated daily by `scripts/claim_board.py`" in page
    assert "comment on that issue itself" in page
    assert "docs/" not in page and "by hand" not in page


def test_write_puts_the_page_at_the_given_path(monkeypatch, tmp_path, capsys):
    page = _stub_fetch(monkeypatch)
    path = tmp_path / "board.md"
    assert board.main(["--write", str(path)]) == 0
    written = path.read_bytes()
    assert b"\r" not in written and written.startswith(b"# Claim board\n")
    assert board.strip_stamp(written.decode("utf-8")) == board.strip_stamp(page)
    assert f"wrote {path}" in capsys.readouterr().out


def test_check_calls_a_missing_file_stale(monkeypatch, tmp_path, capsys):
    _stub_fetch(monkeypatch)
    assert board.main(["--check", str(tmp_path / "board.md")]) == 1
    assert "is stale" in capsys.readouterr().out


def test_check_needs_a_path(capsys):
    with pytest.raises(SystemExit) as stopped:
        board.main(["--check"])
    assert stopped.value.code == 2
    assert "--check: expected one argument" in capsys.readouterr().err


def test_contributing_points_at_the_pinned_issue_not_a_docs_page():
    flat = " ".join((_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
    link = f"[#{board.BOARD_ISSUE}](https://github.com/{board.REPO}/issues/{board.BOARD_ISSUE})"
    assert link in flat
    assert "claims.md" not in flat
    assert not (_ROOT / "docs" / "claims.md").exists(), "the board is an issue body, not a file"
