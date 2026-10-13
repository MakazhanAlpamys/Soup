"""E4 part 1 - the double-quant prevalence probe (benchmarks/harness/e4_double_quant_prevalence.py).

The record ``benchmarks/probe-e4-double-quant-prevalence.md`` carries the rule that
was committed in ``df3dae1`` before any run. Section 1.1 shows the rule body with
its headings demoted two levels so that they nest under the subsection; nothing
else may differ from the committed bytes. ``TestRuleBodyIsTheCommittedOne`` pins
that.
"""

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD = "benchmarks/probe-e4-double-quant-prevalence.md"
RULE_COMMIT = "df3dae149315dbc5168261d57446cf289075422d"
# SHA-256 of the rule body ("## What B3 established" through the last line of
# "## Deliverable") as committed in RULE_COMMIT. CI checks out one commit, so
# the bytes are pinned here and compared with git itself wherever the commit
# object is present.
RULE_BODY_SHA256 = "6a0053d36568a025112a451ad875db6faba737047e98b6622577596b39ff6aa5"
BODY_START = b"## What B3 established"
BODY_STOP = b"\n### 1.2 Definitions"
BEGIN = b"<!-- e4-rule-body:begin -->\n"
END = b"<!-- e4-rule-body:end -->\n"


def _promoted_body() -> bytes:
    """The rule body from the record, headings promoted back two levels."""
    record = (REPO_ROOT / RECORD).read_bytes()
    assert record.count(BEGIN) == 1 and record.count(END) == 1
    shown = record[record.index(BEGIN) + len(BEGIN):record.index(END)]
    lines = shown.splitlines(keepends=True)
    # Every heading inside the markers is a demoted rule heading; anything else
    # would mean the transform was not "two levels, nothing else".
    assert all(line.startswith(b"#### ") for line in lines if line.startswith(b"#"))
    return b"".join(line[2:] if line.startswith(b"#### ") else line for line in lines)


class TestRuleBodyIsTheCommittedOne:
    def test_promoted_body_matches_the_pinned_hash(self):
        body = _promoted_body()
        assert body.startswith(BODY_START)
        assert hashlib.sha256(body).hexdigest() == RULE_BODY_SHA256

    def test_promoted_body_is_byte_identical_to_the_rule_commit(self):
        shown = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{RULE_COMMIT}^{{commit}}"],
            capture_output=True, check=False,
        )
        if shown.returncode != 0:
            # A depth-1 checkout has no rule commit; the pinned hash above is
            # the check there. The pin itself is verified wherever history is.
            return
        committed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{RULE_COMMIT}:{RECORD}"],
            capture_output=True, check=True,
        ).stdout
        body = committed[committed.index(BODY_START):committed.index(BODY_STOP)]
        assert hashlib.sha256(body).hexdigest() == RULE_BODY_SHA256
        assert _promoted_body() == body

    def test_the_rule_body_is_nested_under_section_1_1(self):
        record = (REPO_ROOT / RECORD).read_bytes()
        section = record[record.index(b"### 1.1 Agreed rule"):record.index(BODY_STOP)]
        assert b"\n## " not in section
        assert section.count(b"\n#### ") == 6
