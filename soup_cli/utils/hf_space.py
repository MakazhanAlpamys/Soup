"""HF Space custom template directory renderer (v0.40.2 #51).

Lets users supply their own ``app.py`` / ``README.md`` (+ optional
``requirements.txt``) for ``soup deploy hf-space`` instead of the built-in
``gradio-chat`` / ``streamlit-chat`` templates.

Security model — mirrors v0.29.0 Part F policy:

- Template directory must stay under the current working directory
  (``utils/paths.is_under_cwd``).
- ``model_repo`` is validated via :func:`utils.hf.validate_repo_id` BEFORE
  substitution, so a crafted repo id cannot inject Python source into the
  rendered ``app.py`` uploaded to HF Hub.
- Per-file size cap of 256 KB — defends against pathological templates.
- Only ``app.py``, ``README.md``, and ``requirements.txt`` are read; any
  other files in the directory are ignored.
"""
from __future__ import annotations

import os
import stat as stat_module
from pathlib import Path

from soup_cli.utils.hf import validate_repo_id
from soup_cli.utils.paths import is_under_cwd

_MAX_TEMPLATE_FILE_BYTES = 256 * 1024  # 256 KB
_KNOWN_FILES = ("app.py", "README.md", "requirements.txt")
_REQUIRED_FILES = ("app.py", "README.md")


def render_custom_template_dir(template_dir: str, model_repo: str) -> dict[str, str]:
    """Render a custom Space template directory.

    Returns a dict mapping in-repo filename → rendered content.
    ``{MODEL_REPO}`` placeholder is replaced with the validated ``model_repo``.

    Raises:
        ValueError: containment violation, invalid ``model_repo``, oversized file.
        FileNotFoundError: missing required file (``app.py`` / ``README.md``).
    """
    validate_repo_id(model_repo)

    if not is_under_cwd(template_dir):
        raise ValueError(
            "template-dir must stay under the current working directory; "
            f"got: {template_dir!r}"
        )

    base = Path(template_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"template-dir does not exist or is not a directory: {template_dir}"
        )

    rendered: dict[str, str] = {}
    for fname in _KNOWN_FILES:
        fpath = base / fname
        # ``lstat`` does NOT follow symlinks — defence-in-depth against a
        # crafted symlink at <template_dir>/app.py -> /etc/passwd reading
        # outside the containment-checked directory (mirrors v0.33.0
        # ``prune_checkpoints`` TOCTOU policy).
        try:
            st = os.lstat(fpath)
        except OSError:
            if fname in _REQUIRED_FILES:
                raise FileNotFoundError(
                    f"template-dir is missing required file: {fname}"
                ) from None
            continue
        if stat_module.S_ISLNK(st.st_mode):
            raise ValueError(
                f"template file {fname} is a symlink; refusing to render"
            )
        if not stat_module.S_ISREG(st.st_mode):
            if fname in _REQUIRED_FILES:
                raise FileNotFoundError(
                    f"template-dir is missing required file: {fname}"
                )
            continue
        if st.st_size > _MAX_TEMPLATE_FILE_BYTES:
            raise ValueError(
                f"template file {fname} exceeds 256 KB cap "
                f"({st.st_size} bytes); refusing to render"
            )
        try:
            content = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"template file {fname} is not valid UTF-8"
            ) from exc
        rendered[fname] = content.replace("{MODEL_REPO}", model_repo)

    return rendered
