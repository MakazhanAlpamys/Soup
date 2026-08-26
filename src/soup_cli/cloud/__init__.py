"""Cloud GPU backends for ``soup train --cloud`` (v0.71.18 #16).

Ships Modal, RunPod, and Lambda Cloud backends. Each backend renders a plan
from ``soup.yaml`` by default and gates live submission on provider-specific
credentials.
"""

from __future__ import annotations

__all__ = ["lambda_labs", "modal", "runpod"]
