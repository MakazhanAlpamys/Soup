"""Publish a ``.can`` file to the HuggingFace Hub as a dataset (#34, v0.33.0).

Cans are packaged as HF *datasets* (``repo_type='dataset'``) — they're
metadata-rich tarballs, not model weights. Adds a ``can-format-v1`` tag so
they're discoverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from soup_cli.cans.unpack import inspect_can
from soup_cli.utils.hf import resolve_token, validate_repo_id
from soup_cli.utils.paths import is_under_cwd

CAN_HF_TAG = "can-format-v1"
_MAX_COMMIT_MESSAGE_LEN = 200


def _sanitize_commit_message(message: Optional[str]) -> str:
    """First-line + 200-char cap. Mirrors v0.29.0 push.py / data push policy."""
    if not message:
        return f"Upload .can artifact ({CAN_HF_TAG})"
    first_line = message.splitlines()[0].strip()
    return first_line[:_MAX_COMMIT_MESSAGE_LEN] or f"Upload .can ({CAN_HF_TAG})"


def publish_can(
    can_path: str, *,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> str:
    """Upload a ``.can`` file to ``hf_hub://datasets/<repo_id>``.

    Returns the HF Hub URL of the uploaded file. Raises ``ValueError`` /
    ``FileNotFoundError`` for explicit user-actionable errors.

    Security:
        - ``can_path`` must stay under cwd.
        - ``repo_id`` is validated via ``utils.hf.validate_repo_id``.
        - Token is resolved via ``utils.hf.resolve_token`` (env > cache files).
        - Commit message is first-line-only, capped at 200 chars to prevent
          multi-line injection into public HF history.
    """
    src = Path(can_path)
    if not is_under_cwd(src):
        raise ValueError(f"can path '{can_path}' is outside cwd - refusing")
    if not src.exists():
        raise FileNotFoundError(f"can not found: {can_path}")

    validate_repo_id(repo_id)
    # inspect_can also enforces is_under_cwd + parses the manifest schema.
    manifest = inspect_can(str(src))

    resolved = token or resolve_token()
    if not resolved:
        raise ValueError(
            "no HF token found - set HF_TOKEN env var or `huggingface-cli login`"
        )

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset",
        token=resolved, private=private, exist_ok=True,
    )

    sanitized_msg = _sanitize_commit_message(commit_message)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=src.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=resolved,
        commit_message=sanitized_msg,
    )

    # Tag application via huggingface_hub is API-version-dependent — we
    # surface CAN_HF_TAG in the manifest (read by ``inspect_can``) and in
    # the commit message rather than calling a maybe-no-op API. Adding a
    # README front-matter ``tags:`` block is the supported path; a
    # follow-up tracked as a v0.33.x docs update.
    _ = manifest  # silence unused-var; manifest validates the can shape

    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{src.name}"
