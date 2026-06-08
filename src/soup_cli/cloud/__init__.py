"""Cloud GPU backends for ``soup train --cloud`` (v0.71.18 #16).

Currently ships the Modal.com backend (:mod:`soup_cli.cloud.modal`):
generate a Modal app stub from the user's ``soup.yaml`` and either render
the planned ``modal run`` invocation (default) or submit it live (gated on a
Modal token). Extensible to ``runpod`` / ``lambda`` later.
"""

from __future__ import annotations

__all__ = ["modal"]
