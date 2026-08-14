"""v0.43.0 Part C — `soup doctor --vscode` writer for `.vscode/launch.json`.

Generates a minimal but useful Python debug config for `soup train` plus a
pytest config. Path containment via shared `is_under_cwd`; refuses to
overwrite an existing launch.json without `force=True` (matches v0.40.2
register_data policy).
"""
from __future__ import annotations

import json
import os

from soup_cli.utils.paths import atomic_write_text, is_under_cwd


def build_launch_json(*, config_path: str = "soup.yaml") -> dict:
    """Build the launch.json contents.

    `config_path` is the YAML config to pass to `soup train --config`.
    Validates the path so a crafted argument cannot inject arbitrary
    args into the generated JSON.
    """
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("config_path must be a non-empty string")
    if "\x00" in config_path:
        raise ValueError("config_path must not contain null bytes")
    if "\n" in config_path or "\r" in config_path:
        raise ValueError("config_path must not contain newlines")
    if len(config_path) > 512:
        raise ValueError("config_path too long (>512 chars)")
    return {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "soup train",
                "type": "python",
                "request": "launch",
                "module": "soup_cli.cli",
                "args": ["train", "--config", config_path],
                "console": "integratedTerminal",
                "justMyCode": False,
                "env": {"PYTHONUTF8": "1"},
            },
            {
                "name": "pytest (current file)",
                "type": "python",
                "request": "launch",
                "module": "pytest",
                "args": ["${file}", "-v", "--no-cov"],
                "console": "integratedTerminal",
                "justMyCode": False,
            },
        ],
    }


def write_vscode_launch(
    *,
    config_path: str = "soup.yaml",
    target_dir: str = ".vscode",
    force: bool = False,
) -> str:
    """Write `<cwd>/<target_dir>/launch.json`.

    Returns the absolute path written.
    Raises ValueError on containment violation.
    Raises FileExistsError when launch.json already exists and force=False.
    """
    if not isinstance(target_dir, str) or not target_dir:
        raise ValueError("target_dir must be a non-empty string")
    if "\x00" in target_dir:
        raise ValueError("target_dir must not contain null bytes")
    if not isinstance(force, bool):
        raise ValueError("force must be a bool")

    real = os.path.realpath(target_dir)
    if not is_under_cwd(real):
        raise ValueError("target_dir must stay under cwd")

    payload = build_launch_json(config_path=config_path)
    os.makedirs(real, exist_ok=True)
    out_path = os.path.join(real, "launch.json")
    # Refuse to clobber an existing non-symlink file unless force. A symlink at
    # the target (regardless of force) is rejected by atomic_write_text's
    # enforce_under_cwd_and_no_symlink, which also closes the lstat-then-open
    # TOCTOU window: bytes go to a fresh mkstemp file + os.replace, never
    # through a swapped-in symlink.
    if not force and os.path.lexists(out_path) and not os.path.islink(out_path):
        raise FileExistsError(
            f"{out_path} already exists; pass force=True to overwrite"
        )
    body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(body, out_path, field="launch.json")
    return out_path
