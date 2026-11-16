"""Send a non-finite float as JSON ``null`` (#2750).

JSON has no NaN or Infinity. Starlette renders with ``allow_nan=False`` and
raises on one, and ``json.dumps`` with its defaults writes bare ``NaN`` /
``Infinity`` tokens that a browser's ``JSON.parse`` refuses. ``null`` is what
the Web UI charts already draw as a gap.
"""

from __future__ import annotations

import math
from typing import Any


def finite_json(value: Any) -> Any:
    """Return ``value`` with every non-finite float replaced by ``None``.

    Dicts, lists and tuples are walked; a tuple comes back as a list, as it
    would from any JSON encoder. Every other value is returned unchanged.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value
