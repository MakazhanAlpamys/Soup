"""`soup data demo` metadata must describe the fixture it actually ships.

Three of the four bundles carried metadata that disagreed with their file:

- ``alpaca_demo`` advertised "20-row" against a 10-row fixture.
- ``sharegpt_demo`` declared ``format="sharegpt"`` for a file that is
  ``prompt``/``chosen``/``rejected`` — ``detect_format()`` calls it ``dpo``.
- ``grpo_demo`` declared ``format="reasoning"``, which is not a format Soup
  accepts anywhere.

The row count was dropped rather than corrected: a number describing a file,
hardcoded in another file, rots again the moment the fixture changes. The
``format`` field cannot be dropped — it is what a user copies into
``data.format`` — so it is pinned here against the detector instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import detect_format
from soup_cli.utils.demo_bundles import (
    DEMO_BUNDLE_NAMES,
    _bundle_source_path,
    get_bundle,
    list_bundles,
)

BUNDLE_NAMES = sorted(DEMO_BUNDLE_NAMES)


def _rows(name: str) -> list[dict]:
    src = Path(_bundle_source_path(get_bundle(name)))
    with open(src, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_bundles_were_actually_found():
    """Guard the parametrization: an empty registry would vacuously pass."""
    assert len(BUNDLE_NAMES) >= 4, BUNDLE_NAMES


@pytest.mark.parametrize("name", BUNDLE_NAMES)
def test_declared_format_matches_the_fixture(name: str):
    """The advertised format is what the loader will detect.

    This is the field a user copies into ``data.format``. Declaring
    ``sharegpt`` for preference rows sends them to the wrong normalizer, which
    is a silently wrong training run rather than an error.
    """
    declared = get_bundle(name).format
    assert declared == detect_format(_rows(name)), (
        f"{name} declares format={declared!r} but its fixture detects as "
        f"{detect_format(_rows(name))!r}"
    )


@pytest.mark.parametrize("name", BUNDLE_NAMES)
def test_declared_format_is_one_the_schema_accepts(name: str):
    """``format='reasoning'`` was advertised and would be rejected on use."""
    declared = get_bundle(name).format
    DataConfig(train="x.jsonl", format=declared)  # raises if not a valid literal


@pytest.mark.parametrize("name", BUNDLE_NAMES)
def test_description_claims_no_row_count(name: str):
    """No hardcoded counts in descriptions — that is what rotted.

    ``soup data inspect`` reports the real number from the file, which cannot
    disagree with it.
    """
    description = get_bundle(name).description
    assert not re.search(r"\d+[\s-]*row", description, flags=re.IGNORECASE), (
        f"{name} description hardcodes a row count: {description!r}. "
        "Counts belong in `soup data inspect`, not in metadata."
    )


def test_every_bundle_resolves_to_a_real_non_empty_fixture():
    """A bundle pointing at a missing or empty file is unusable."""
    for bundle in list_bundles():
        rows = _rows(bundle.name)
        assert rows, f"{bundle.name} resolves to an empty fixture"
