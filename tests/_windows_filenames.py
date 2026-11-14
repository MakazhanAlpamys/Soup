"""Shared filename restrictions for tests, independent of CI (#2492).

Use the colon marker only when a case needs a colon in a real filename or
relies on path handling that interprets a leading ``n:`` as a Windows drive.
A colon that is only displayed as text does not require this skip.
"""

from __future__ import annotations

import sys

import pytest


def _windows_colon_filename() -> bool:
    """Colon filenames are invalid on both local Windows and Windows CI."""
    return sys.platform == "win32"


skip_colon_on_windows = pytest.mark.skipif(
    _windows_colon_filename(), reason="':' is not valid in a Windows name"
)
