"""The #808 silence checks must not read raw captured output (#1352).

Rich wraps the loader's warning at the render width, so
``"read by nothing" not in capsys.readouterr().out`` passes while the warning
is printed whenever the line break lands inside the phrase (all four ``warn``
cases at ``COLUMNS=50``). Those checks go through ``_plain(...)``; this scan
fails if either one goes back to raw output, without editing any source file.
"""

from __future__ import annotations

import re
from pathlib import Path

_GUARDED = Path(__file__).with_name("test_issue808_staged_config_fields.py")
_RAW_NEGATIVE = re.compile(r"\bnot\s+in\s+capsys\.readouterr\(\)\.(?:out|err)\b")
_PLAIN_CHECK = '"read by nothing" not in _plain(capsys.readouterr().out)'


def test_808_negative_checks_do_not_read_raw_output() -> None:
    lines = _GUARDED.read_text(encoding="utf-8").splitlines()
    offenders = [
        f"{_GUARDED.name}:{number}: {line.strip()}"
        for number, line in enumerate(lines, start=1)
        if _RAW_NEGATIVE.search(line)
    ]
    assert not offenders, (
        "a negative substring check on raw captured output passes when Rich wraps "
        "the phrase; route it through _plain(...):\n" + "\n".join(offenders)
    )


def test_808_negative_checks_are_still_there() -> None:
    # The scan above is vacuous if the checks are deleted rather than reverted.
    assert _GUARDED.read_text(encoding="utf-8").count(_PLAIN_CHECK) >= 2
