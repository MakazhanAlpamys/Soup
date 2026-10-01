"""Follow-up to #1461 / #1499: the unknown-key messages name at most ten keys.

A suite under the 1 MiB cap can carry tens of thousands of top-level or entry keys;
naming every one of them produced a message close to a megabyte. Ten keys and
"and N more" is the same cap shape the config loader uses for its unknown-key report.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.expectations import parse_suite_yaml

_KEYS = [f"k{i:02d}" for i in range(50)]


def _top_level(n: int) -> str:
    return "".join(f"{k}: 1\n" for k in _KEYS[:n]) + "expectations:\n  - name: expect_no_pii\n"


def _entry_level(n: int) -> str:
    return "expectations:\n  - name: expect_no_pii\n" + "".join(f"    {k}: 1\n" for k in _KEYS[:n])


@pytest.mark.parametrize("suite", [_top_level, _entry_level], ids=["top-level", "entry"])
def test_fifty_unknown_keys_name_ten_and_count_the_rest(suite) -> None:
    with pytest.raises(ValueError) as info:
        parse_suite_yaml(suite(50))
    message = str(info.value)
    assert all(f"'{k}'" in message for k in _KEYS[:10])
    assert "'k10'" not in message
    assert "and 40 more" in message
    assert len(message) < 400


@pytest.mark.parametrize("suite", [_top_level, _entry_level], ids=["top-level", "entry"])
def test_control_ten_or_fewer_keys_are_all_named(suite) -> None:
    with pytest.raises(ValueError) as info:
        parse_suite_yaml(suite(10))
    message = str(info.value)
    assert all(f"'{k}'" in message for k in _KEYS[:10])
    assert "more" not in message
