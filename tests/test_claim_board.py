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


def comment(login, body, created, updated=None):
    return {
        "user": {"login": login},
        "body": body,
        "created_at": created,
        "updated_at": updated or created,
    }


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


@pytest.mark.parametrize("labels", [("help wanted",), ("help wanted", "claimed")])
@pytest.mark.parametrize(
    "number, login, scope, acceptance, terminal",
    [
        (
            1199,
            "Faisal-Fayaz",
            "fix 2",
            "Confirmed, @Faisal-Fayaz: fix 2 is yours.",
            "Status update: fix 2 by @Faisal-Fayaz is completed by merged #1728; "
            "fix 3 was completed by #1588. Thank you. The accepted fix-2 claim is "
            "fulfilled, so the stale `claimed` label is removed. This issue remains "
            "open for the explicitly unclaimed fix 1.",
        ),
        (
            797,
            "Amix29",
            "item 2",
            "Confirmed, @Amix29: item 2 of the 2026-10-05 scope is yours.",
            "Claim metadata update: the accepted item-2 slice is completed by #1778, "
            "as recorded above, so the stale `claimed` label is removed. Items 1, 3 "
            "and 4 remain open and unclaimed. This issue stays open.",
        ),
        (
            839,
            "sloemo01",
            "dropout item",
            "Welcome back, @sloemo01: the dropout item is yours again, the `claimed` "
            "label and the assignment are restored. Your plan matches the three points "
            "of the 2026-10-02 check-in, and #1784 is in review.",
            "Status update: the accepted dropout slice by @sloemo01 is completed by "
            "#1784 (merged as `1f9e1515`). Thank you. Its `claimed` label and completed "
            "assignment are cleared; this issue stays open for the separately listed "
            "CUDA/NF4 streaming qualification and micro-benchmark work. No remaining "
            "slice has an accepted claim in this thread.",
        ),
        (
            1186,
            "chiliec",
            "item 3",
            "Thanks, @chiliec: item 3 is yours again and the claimed label is restored; "
            "#1782 is in review.",
            "Claim metadata update: item 3 is completed by #1746 and #1782, as "
            "recorded above, so the completed slice no longer reserves this aggregate "
            "issue. The stale `claimed` label is removed. Items 1 and 2 remain open "
            "and unclaimed; this is not a closure of those modality defects.",
        ),
    ],
)
def test_completed_claim_slices_are_free_despite_historical_acceptance(
    labels, number, login, scope, acceptance, terminal
):
    """The four live aggregate issues kept old claims after their accepted slices merged."""
    comments = [
        comment(login, f"I'll take {scope}.", "2026-10-01T10:00:00Z"),
        comment(
            board.MAINTAINER,
            acceptance,
            "2026-10-02T10:00:00Z",
        ),
        comment(board.MAINTAINER, terminal, "2026-10-03T10:00:00Z"),
    ]
    row = board.classify(
        issue(number, labels), comments, [pull(1990, login, f"Refs #{number}")], NOW
    )
    assert row["kind"] == "free" and row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


def test_appended_completion_overrides_acceptance_in_the_same_edited_comment():
    comments = [
        comment("alice", "I'll take fix 2.", "2026-10-01T10:00:00Z"),
        comment(
            board.MAINTAINER,
            "Confirmed, @alice: fix 2 is yours.\n\n"
            "Status update: fix 2 by @alice is completed by merged #90. "
            "The accepted fix-2 claim is fulfilled, so the stale `claimed` label is removed. "
            "Fix 1 remains open and unclaimed.",
            "2026-10-02T10:00:00Z",
            "2026-10-10T10:00:00Z",
        ),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize("separator", ["\n\n", " "])
def test_appended_acceptance_does_not_outlive_its_later_completion_in_the_same_comment(separator):
    comments = [
        comment(
            board.MAINTAINER,
            "Thanks for the detailed plan.\n\nConfirmed, @bob: item 2 is yours."
            + separator + "Item 2 is completed by merged #92.",
            "2026-10-02T10:00:00Z",
            "2026-10-10T10:00:00Z",
        ),
        comment("bob", "Taking item 2.", "2026-10-07T10:00:00Z"),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [pull(92, "bob", "Refs #7")], NOW)
    assert row["kind"] == "free" and row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


def test_issue_release_discards_old_requests_before_a_new_take_and_confirmation():
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed", "2026-10-02T10:00:00Z"),
        comment(board.MAINTAINER, "The claim is released. This issue is free to take.",
                "2026-10-03T10:00:00Z"),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [], NOW)
    assert row["state"] == "free" and row["quiet"] is None
    comments.append(comment("bob", "I'll take this", "2026-10-05T10:00:00Z"))
    assert board.classify(issue(7, ("claimed",)), comments, [], NOW)["state"] == "free"
    comments.append(comment(board.MAINTAINER, "Confirmed", "2026-10-06T10:00:00Z"))
    row = board.classify(issue(7, ("claimed",)), comments, [pull(91, "bob", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@bob` since 2026-10-05"
    assert row["pr"] == 91 and row["quiet"] == 9


@pytest.mark.parametrize("take_text", ["Taking item 2.", "I am taking proposed fix 2."])
@pytest.mark.parametrize("draft", [False, True])
def test_leading_taking_request_binds_generic_acceptance_and_open_refs_pr(take_text, draft):
    comments = [
        comment("alice", take_text, "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, go ahead.", "2026-10-02T10:00:00Z"),
    ]
    pr = dict(pull(91, "alice", "Refs #7"), draft=draft)
    row = board.classify(issue(7), comments, [pr], NOW)
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91 and row["quiet"] == 13


def test_late_competitor_request_does_not_replace_the_confirmed_incumbent():
    comments = [
        comment("alice", "I'll take item 2.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @alice: item 2 is yours.",
                "2026-10-02T10:00:00Z"),
        comment("bob", "Taking item 2 too.", "2026-10-03T10:00:00Z"),
    ]
    prs = [pull(91, "alice", "Refs #7"), pull(92, "bob", "Refs #7")]
    row = board.classify(issue(7), comments, prs, NOW)
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91 and row["quiet"] == 13


def test_reaffirmed_open_pr_outranks_a_declined_competing_take():
    """#1432's owner claimed through a PR before the competing issue comment."""
    comments = [
        comment("bob", "I'd like to take this one. Here is my plan.",
                "2026-10-03T10:33:31Z"),
        comment(
            board.MAINTAINER,
            "Thank you, @bob. The plan is sound.\n\n"
            "#7 already has an open pull request: #91 by @alice, opened on 2026-09-30. "
            "An open pull request counts as a claim and the first claim stands, "
            "so #7 stays with @alice.\n\n"
            "So please do not open a separate pull request for #7.\n\n"
            "If #91 is released, this thread will say so.",
            "2026-10-04T15:51:20Z",
        ),
    ]
    prs = [dict(pull(91, "alice", "Thread the freeze plan into layer selection."),
                title="Respect the freeze plan (#7)", state="open",
                created_at="2026-09-30T15:33:36Z")]
    row = board.classify(issue(7, ("claimed",)), comments, prs, NOW)
    assert row["state"] == "claimed by `@alice` since 2026-09-30"
    assert row["pr"] == 91 and row["quiet"] is None
    assert board.render([row], NOW).splitlines()[-1] == (
        "| #7 | `Some title` | - | claimed by `@alice` since 2026-09-30 | #91 | - |"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "Thank you, @bob. The plan is sound.\n\n"
        "Please do not open a separate pull request for #7.",
        "Confirmed, @bob: this issue is not yours.",
        "Confirmed, @bob: #7 is not yours.",
    ],
)
def test_an_explicitly_declined_take_is_not_an_unconfirmed_owner(reply):
    comments = [
        comment("bob", "I'll take this.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, reply, "2026-10-04T10:00:00Z"),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [pull(92, "bob", "Refs #7")], NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


def test_an_edited_old_decline_cannot_become_a_positive_confirmation():
    comments = [
        comment(board.MAINTAINER, "Confirmed, @bob: this issue is not yours.",
                "2026-10-01T10:00:00Z", "2026-10-04T10:00:00Z"),
        comment("bob", "I'll take this.", "2026-10-03T10:00:00Z"),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [pull(92, "bob", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@bob` since 2026-10-03 (unconfirmed)"
    assert row["pr"] == 92 and row["quiet"] == 11


def test_a_declined_reacceptance_cannot_create_an_uncertain_accepted_owner():
    comments = [
        comment("bob", "Taking item 2.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @bob: item 2 is yours.\n\n"
                "Confirmed, @bob: this issue is not yours; item 2 remains unclaimed.",
                "2026-10-02T10:00:00Z", "2026-10-08T10:00:00Z"),
        comment(board.MAINTAINER, "The claim is released.", "2026-10-05T10:00:00Z"),
        comment("bob", "Taking item 2 again.", "2026-10-07T10:00:00Z"),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [pull(92, "bob", "Refs #7")], NOW)
    assert row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"user": {"login": "carol"}},
        {"state": "closed"},
        {"created_at": "2026-10-05T10:00:00Z"},
        {"number": 92},
    ],
)
def test_reaffirmed_pr_owner_needs_matching_open_prior_pr_metadata(metadata):
    comments = [
        comment("bob", "I'll take this.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, "Thank you, @bob.\n\n#7 stays with @alice. "
                "#91 by @alice already carries it.\n\n"
                "Please do not open a competing pull request for #7.",
                "2026-10-04T10:00:00Z"),
    ]
    pr = dict(pull(91, "alice", "Fixes #7"), state="open",
              created_at="2026-09-30T10:00:00Z")
    pr.update(metadata)
    row = board.classify(issue(7, ("claimed",)), comments, [pr], NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


def test_a_declined_request_can_be_accepted_after_a_new_take():
    comments = [
        comment("bob", "I'll take this.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @bob: this issue is not yours.",
                "2026-10-02T10:00:00Z"),
        comment("bob", "I'll take this again.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed.", "2026-10-04T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [pull(92, "bob", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@bob` since 2026-10-03"
    assert row["pr"] == 92 and row["quiet"] == 11


def test_declining_a_competitor_and_a_conditional_release_preserve_the_active_owner():
    comments = [
        comment("alice", "I'll take this.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @alice: it is yours.",
                "2026-10-02T10:00:00Z"),
        comment("bob", "I'll take this too.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, "Thank you, @bob. Please do not open a separate "
                "pull request for #7. If the claim is released, this thread will say so.",
                "2026-10-04T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91 and row["quiet"] == 13


@pytest.mark.parametrize(
    "reply",
    [
        "Thank you, @bob. The earlier instruction, "
        "`do not open a separate pull request for #7`, no longer applies.",
        "Thank you, @bob. The earlier instruction, "
        "do not open a separate pull request for #7, no longer applies.",
        "Thank you, @bob. The old note was "
        "`#91 by @alice, so #7 stays with @alice`; it is no longer correct.",
        "Thank you, @bob. It is not true that #7 stays with @alice. "
        "#91 by @alice is the old pull request.",
        "Thank you, @bob. The old note said #91 by @alice carries it, "
        "so #7 stays with @alice; that is no longer correct.",
        "Thank you, @bob. #91 by @alice carries it, so #7 stays with @alice. "
        "That ruling is no longer correct.",
    ],
)
def test_quoted_or_negated_claim_instructions_do_not_change_a_take_request(reply):
    comments = [
        comment("bob", "I'll take this.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, reply, "2026-10-04T10:00:00Z"),
    ]
    prs = [dict(pull(91, "alice", "Refs #7"), state="open",
                created_at="2026-09-30T10:00:00Z"), pull(92, "bob", "Refs #7")]
    row = board.classify(issue(7, ("claimed",)), comments, prs, NOW)
    assert row["state"] == "claimed by `@bob` since 2026-10-03 (unconfirmed)"
    assert row["pr"] == 92 and row["quiet"] == 11


@pytest.mark.parametrize("verb", ["stays", "remains"])
def test_direct_current_retention_names_the_verified_open_pr_owner(verb):
    comments = [comment(board.MAINTAINER, f"#7 {verb} with @alice. "
                        "#91 by @alice is the current open pull request.",
                        "2026-10-04T10:00:00Z")]
    prs = [dict(pull(91, "alice", "Fix the layer selection."), state="open",
                created_at="2026-09-30T10:00:00Z")]
    row = board.classify(issue(7, ("claimed",)), comments, prs, NOW)
    assert row["state"] == "claimed by `@alice` since 2026-09-30"
    assert row["pr"] == 91 and row["quiet"] is None


def test_accepting_an_overlapping_second_owner_does_not_silently_pick_a_winner():
    comments = [
        comment("alice", "I'll take item 2.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @alice: item 2 is yours.",
                "2026-10-02T10:00:00Z"),
        comment("bob", "Taking item 2.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @bob: item 2 is yours.",
                "2026-10-04T10:00:00Z"),
    ]
    prs = [pull(91, "alice", "Refs #7"), pull(92, "bob", "Refs #7")]
    row = board.classify(issue(7), comments, prs, NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


def test_completing_item_two_preserves_the_accepted_item_three_owner():
    comments = [
        comment("alice", "I'll take item 2.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @alice: item 2 is yours.",
                "2026-10-02T10:00:00Z"),
        comment("bob", "Taking item 3.", "2026-10-03T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @bob: item 3 is yours.",
                "2026-10-04T10:00:00Z"),
        comment(board.MAINTAINER, "Item 2 is completed by merged #90. "
                "The completed slice's `claimed` label is removed.",
                "2026-10-05T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [pull(92, "bob", "Refs #7")], NOW)
    assert row["state"] == "claimed by `@bob` since 2026-10-03"
    assert row["pr"] == 92 and row["quiet"] == 11


def test_two_acceptances_in_one_edited_owner_comment_report_see_thread():
    """#808 has two accepted slices, with the second appended after a later take."""
    comments = [
        comment("alice", "I'll take item 2.", "2026-10-01T10:00:00Z"),
        comment(
            board.MAINTAINER,
            "Thanks @alice, and sorry for the wait. Item 2 is yours: the checkpoint flags.\n\n"
            "Confirmed, @bob: item 3 is yours, distinct from @alice's accepted item 2.",
            "2026-10-02T10:00:00Z",
            "2026-10-10T10:00:00Z",
        ),
        comment("bob", "Taking item 3.", "2026-10-05T10:00:00Z"),
    ]
    prs = [pull(91, "alice", "Refs #808"), pull(92, "bob", "Refs #808")]
    row = board.classify(issue(808), list(reversed(comments)), prs, NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize(
    "login, body",
    [
        ("bob", "The claim is released. The `claimed` label is removed."),
        (board.MAINTAINER, "> The claim is released.\nThe claim is not released."),
        (board.MAINTAINER, "```text\nThe claim is released.\n```\nThe claim remains active."),
        (board.MAINTAINER, "Do not remove the `claimed` label. Item 2 is not completed."),
        (board.MAINTAINER, 'The contributor wrote "The claim is released."; that is incorrect.'),
        (board.MAINTAINER, "The claim is released? Please confirm before we free it."),
        (board.MAINTAINER, "The claim is released only after #99 lands."),
        (board.MAINTAINER, "Do not write `This issue is free to take` until it is released."),
        (board.MAINTAINER, "Do not write `The stale claimed label is removed` yet."),
        (board.MAINTAINER, "It is not true that this issue is free to take."),
        (board.MAINTAINER, "The stale claimed label is removed because other slices are open; "
                "@alice retains item 2."),
    ],
)
def test_third_party_quoted_or_negated_terminal_text_does_not_clear_claim(login, body):
    comments = [
        comment("alice", "I'll take item 2.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "Confirmed, @alice: item 2 is yours.",
                "2026-10-02T10:00:00Z"),
        comment(login, body, "2026-10-03T10:00:00Z"),
    ]
    row = board.classify(
        issue(7), comments, [pull(91, "alice", "Refs #7")], NOW, lambda number: False
    )
    assert row["state"] == "claimed by `@alice` since 2026-10-01"
    assert row["pr"] == 91


@pytest.mark.parametrize("prefix", ["", "Thanks for the detailed plan.\n\n"])
def test_unrelated_edit_of_old_acceptance_does_not_resurrect_a_released_claim(prefix):
    comments = [
        comment("alice", "I'll take this", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, prefix + "Confirmed, @alice: it is yours.\n\nTypo corrected.",
                "2026-10-02T10:00:00Z", "2026-10-10T10:00:00Z"),
        comment(board.MAINTAINER, "The claim is released. This issue is free to take.",
                "2026-10-05T10:00:00Z"),
    ]
    row = board.classify(issue(7, ("claimed",)), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize("prefix", ["", "Thanks for the detailed plan.\n\n"])
@pytest.mark.parametrize(
    "take, acceptance, release",
    [
        ("I'll take this.", "it is yours.", "The claim is released. This issue is free to take."),
        ("Taking item 2.", "item 2 is yours.", "Item 2 is released."),
    ],
)
def test_unrelated_edit_of_old_release_preserves_a_later_confirmed_owner(
    prefix, take, acceptance, release
):
    comments = [
        comment("alice", take, "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, f"Confirmed, @alice: {acceptance}",
                "2026-10-02T10:00:00Z"),
        comment(board.MAINTAINER, prefix + release + "\n\nTypo corrected.",
                "2026-10-05T10:00:00Z", "2026-10-10T10:00:00Z"),
        comment("bob", take, "2026-10-07T10:00:00Z"),
        comment(board.MAINTAINER, f"Confirmed, @bob: {acceptance}",
                "2026-10-08T10:00:00Z"),
    ]
    prs = [pull(91, "alice", "Refs #7"), pull(92, "bob", "Refs #7")]
    row = board.classify(issue(7, ("claimed",)), comments, prs, NOW)
    assert row["kind"] == "claimed"
    assert row["state"] == "claimed by `@bob` since 2026-10-07"
    assert row["pr"] == 92 and row["quiet"] == 7
    assert board.render([row], NOW).splitlines()[-1] == (
        "| #7 | `Some title` | - | claimed by `@bob` since 2026-10-07 | #92 | quiet 7 d |"
    )


@pytest.mark.parametrize("separator", ["\n\n", " "])
def test_appended_same_scope_reacceptance_reports_uncertain_edit_chronology(separator):
    comments = [
        comment("alice", "Taking item 2.", "2026-10-01T10:00:00Z"),
        comment(
            board.MAINTAINER,
            "Confirmed, @alice: item 2 is yours." + separator
            + "Confirmed, @alice: item 2 is yours again.",
            "2026-10-02T10:00:00Z",
            "2026-10-08T10:00:00Z",
        ),
        comment(board.MAINTAINER, "The claim is released.", "2026-10-05T10:00:00Z"),
        comment("alice", "Taking item 2 again.", "2026-10-07T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize("separator", ["\n\n", " "])
def test_explicit_completion_clears_an_uncertain_reacceptance_in_the_same_comment(separator):
    comments = [
        comment("alice", "Taking item 2.", "2026-10-01T10:00:00Z"),
        comment(
            board.MAINTAINER,
            "Confirmed, @alice: item 2 is yours.\n\n"
            "Confirmed, @alice: item 2 is yours again." + separator
            + "Item 2 is completed by merged #92.",
            "2026-10-02T10:00:00Z",
            "2026-10-08T10:00:00Z",
        ),
        comment(board.MAINTAINER, "The claim is released.", "2026-10-05T10:00:00Z"),
        comment("alice", "Taking item 2 again.", "2026-10-07T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [pull(92, "alice", "Refs #7")], NOW)
    assert row["kind"] == "free" and row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize(
    "offer",
    [
        "@alice, the vision and audio sites are yours if you want them.",
        "If you want the other two afterwards, say so and they are yours.",
    ],
)
def test_a_conditional_offer_does_not_accept_an_earlier_take(offer):
    comments = [
        comment("alice", "Taking this one.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, offer, "2026-10-02T10:00:00Z"),
    ]
    row = board.classify(issue(7), comments, [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "free"
    assert row["pr"] is None and row["quiet"] is None


@pytest.mark.parametrize("ownership", ["it's yours", "it’s yours", "it is yours"])
def test_named_remainder_acceptance_keeps_the_active_owner_over_an_older_taking_request(ownership):
    """#806's explicit remainder acceptance must outrank the completed earlier slice."""
    comments = [
        comment("AmirF194", "Taking this one. Scoping the fix to the flash-attn/liger half.",
                "2026-09-14T18:33:58Z"),
        comment("dchaudhari7177", "I'd like to take the reasoning_effort remainder.",
                "2026-10-03T10:28:10Z"),
        comment(board.MAINTAINER, f"@dchaudhari7177, {ownership}. Nobody else claimed the "
                "remainder after I opened it to anyone on 09-24, so I've labelled the issue "
                "`claimed`. Thanks for tracing it first.", "2026-10-04T15:51:53Z"),
    ]
    row = board.classify(issue(806, ("claimed",)), comments,
                         [pull(1587, "dchaudhari7177", "Refs #806")], NOW)
    assert row["state"] == "claimed by `@dchaudhari7177` since 2026-10-03"
    assert row["pr"] == 1587 and row["quiet"] == 11


@pytest.mark.parametrize("take_text, expected_pr, expected_quiet", [
    ("Taking item 2.", None, None),
    ("I'll take this.", 91, 13),
])
def test_unconfirmed_blocked_claim_metrics_preserve_the_existing_fallback(
    take_text, expected_pr, expected_quiet
):
    comments = [comment("alice", take_text, "2026-10-01T10:00:00Z")]
    row = board.classify(issue(7, ("infra-blocked",)), comments,
                         [pull(91, "alice", "Refs #7")], NOW)
    assert row["state"] == "blocked: infra-blocked"
    assert row["pr"] == expected_pr and row["quiet"] == expected_quiet


def test_arbitrary_slice_acceptance_after_taking_stays_conservatively_claimed():
    """#275's old recipe prose cannot safely identify or clear a numbered claim slice."""
    comments = [
        comment("alice", "Taking `some-recipe`.", "2026-10-01T10:00:00Z"),
        comment(board.MAINTAINER, "@alice: `some-recipe` is yours.",
                "2026-10-02T10:00:00Z"),
    ]
    row = board.classify(issue(275), comments, [], NOW)
    assert row["state"] == "claimed (see thread)"
    assert row["pr"] is None and row["quiet"] is None


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
    for text in (
        "unclaimed", "claimed by bob", "I claimed it", "Thanks for taking this", "I am not taking"
    ):
        assert not board._TAKE_RE.search(text), text
    for text in (
        "I'd like to take", "I'll take it", "take this", "claim this", "I claim",
        "Taking item 2.", "I am taking proposed fix 2.",
    ):
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
