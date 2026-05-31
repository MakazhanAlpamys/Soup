"""v0.43.0 Part D — `soup data demo` bundle registry.

Single source of truth for the small JSONL fixtures bundled under
`examples/data/`. The CLI command resolves a name to a path, copies the
bundle into the user-supplied output (containment-checked), and prints
a short summary. No network access. Path containment via shared
`is_under_cwd` (mirrors v0.42.0 ingest policy).
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping

from soup_cli.utils.paths import is_under_cwd

_MAX_NAME_LEN = 32
_MAX_OUTPUT_BYTES = 50 * 1024 * 1024  # 50 MB defence


@dataclass(frozen=True)
class DemoBundle:
    name: str
    fixture: str  # filename under examples/data/
    description: str
    format: str   # alpaca / sharegpt / dpo / reasoning


_BUNDLES: Mapping[str, DemoBundle] = MappingProxyType({
    "alpaca_demo": DemoBundle(
        name="alpaca_demo",
        fixture="alpaca_tiny.jsonl",
        description="20-row Alpaca-style instruction tuning fixture",
        format="alpaca",
    ),
    "sharegpt_demo": DemoBundle(
        name="sharegpt_demo",
        fixture="chat_preferences.jsonl",
        description="ShareGPT-style multi-turn chat fixture",
        format="sharegpt",
    ),
    "dpo_demo": DemoBundle(
        name="dpo_demo",
        fixture="dpo_sample.jsonl",
        description="Preference (prompt/chosen/rejected) DPO fixture",
        format="dpo",
    ),
    "grpo_demo": DemoBundle(
        name="grpo_demo",
        fixture="reasoning_math.jsonl",
        description="Math reasoning fixture for GRPO/RLVR",
        format="reasoning",
    ),
})

DEMO_BUNDLE_NAMES = frozenset(_BUNDLES.keys())


def _validate_name(name: object) -> str:
    if not isinstance(name, str):
        raise ValueError("bundle name must be a string")
    if not name:
        raise ValueError("bundle name must not be empty")
    if "\x00" in name:
        raise ValueError("bundle name must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"bundle name length {len(name)} exceeds max {_MAX_NAME_LEN}")
    if name not in _BUNDLES:
        supported = ", ".join(sorted(_BUNDLES))
        raise ValueError(f"unknown bundle '{name}'. Supported: {supported}")
    return name


def list_bundles() -> list[DemoBundle]:
    """Return all registered demo bundles, sorted by name."""
    return [_BUNDLES[name] for name in sorted(_BUNDLES)]


def get_bundle(name: str) -> DemoBundle:
    """Return the bundle metadata for `name`, or raise ValueError."""
    canonical = _validate_name(name)
    return _BUNDLES[canonical]


def _bundle_source_path(bundle: DemoBundle) -> str:
    """Resolve the on-disk path for a bundle's fixture.

    v0.53.8 #93 — fixtures live under ``soup_cli/data/_fixtures/`` (package
    data), with a back-compat fallback to ``examples/data/`` for editable
    installs / repo-root invocations. The package-data location is
    zipapp / namespace-package safe; the legacy location is kept so
    contributors editing fixtures via ``examples/data/`` still see their
    changes.
    """
    # Filenames are baked-in constants (no user input), so direct join
    # is safe; we still defensively reject path separators.
    if "/" in bundle.fixture or "\\" in bundle.fixture:
        raise ValueError(
            f"bundle fixture name has separator: {bundle.fixture!r}"
        )
    # 1) Preferred — package data at soup_cli/data/_fixtures/.
    pkg_root = files("soup_cli")
    pkg_candidate = os.path.realpath(
        os.path.join(str(pkg_root), "data", "_fixtures", bundle.fixture)
    )
    if os.path.isfile(pkg_candidate):
        return pkg_candidate
    # 2) Fallback — legacy examples/data/ at repo root.
    repo_root = os.path.dirname(str(pkg_root))
    legacy = os.path.realpath(
        os.path.join(repo_root, "examples", "data", bundle.fixture)
    )
    if os.path.isfile(legacy):
        return legacy
    raise FileNotFoundError(
        f"bundle fixture missing: {bundle.fixture}"
    )


def copy_bundle_to(name: str, output_path: str) -> str:
    """Copy bundle's JSONL into `output_path`. Containment-checked.

    Returns the absolute path written. Refuses to overwrite existing files
    (caller must remove first). Validates that every line of the bundle is
    valid JSON to avoid silently shipping a malformed fixture.
    """
    bundle = get_bundle(name)
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string")
    if "\x00" in output_path:
        raise ValueError("output_path must not contain null bytes")
    real_out = os.path.realpath(output_path)
    if not is_under_cwd(real_out):
        raise ValueError("output_path must stay under cwd")
    if os.path.exists(real_out):
        raise FileExistsError(
            f"{output_path} already exists; remove it before re-running"
        )
    src = _bundle_source_path(bundle)
    os.makedirs(os.path.dirname(real_out) or ".", exist_ok=True)
    # Stage writes to a sibling temp file so a mid-stream cap rejection
    # never leaves a partial file at the user-visible target path.
    tmp_path = real_out + ".tmp"
    # TOCTOU defence: reject pre-placed symlinks at the temp path
    # (mirrors v0.33.0 #22 / v0.40.2 #51 / v0.42.0 ingest policy).
    try:
        if stat.S_ISLNK(os.lstat(tmp_path).st_mode):
            raise ValueError(
                "staging temp path is a symlink; aborting"
            )
    except FileNotFoundError:
        pass
    total = 0
    try:
        with open(src, encoding="utf-8") as f_in, open(
            tmp_path, "w", encoding="utf-8"
        ) as f_out:
            for lineno, raw in enumerate(f_in, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"bundle {name} line {lineno} is not valid JSON: {exc}"
                    ) from exc
                total += len(raw.encode("utf-8"))
                if total > _MAX_OUTPUT_BYTES:
                    raise ValueError(
                        f"bundle {name} exceeds {_MAX_OUTPUT_BYTES} byte cap"
                    )
                f_out.write(raw)
                if not raw.endswith("\n"):
                    f_out.write("\n")
        os.replace(tmp_path, real_out)
    except BaseException:
        # Best-effort cleanup of the staged temp file on any failure.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    return real_out
