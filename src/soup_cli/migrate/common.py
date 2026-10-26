"""Common utilities for config migration."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Max input file size for migration (10 MB — no legitimate config exceeds this)
MAX_CONFIG_FILE_SIZE = 10 * 1024 * 1024


def validate_input_path(path: Path) -> Path:
    """Validate that input path exists and is under cwd (path traversal protection)."""
    # realpath + commonpath containment (is_under_cwd) — Path.resolve() +
    # relative_to() breaks on Windows 8.3 short names.
    from soup_cli.utils.paths import is_under_cwd

    resolved = path.resolve()
    # Check path traversal first (before checking existence)
    if not is_under_cwd(path):
        raise ValueError(
            f"Input path is outside the current directory: {path}"
        )
    if not resolved.exists():
        raise ValueError(f"Input file not found: {path}")
    if resolved.stat().st_size > MAX_CONFIG_FILE_SIZE:
        raise ValueError(
            f"Input file too large ({resolved.stat().st_size} bytes). "
            f"Max: {MAX_CONFIG_FILE_SIZE} bytes."
        )
    return resolved


def validate_output_path(path: Path) -> Path:
    """Validate that output path is under cwd (path traversal protection)."""
    from soup_cli.utils.paths import is_under_cwd

    resolved = (Path.cwd() / path).resolve()
    if not is_under_cwd(resolved):
        raise ValueError(
            f"Output path is outside the current directory: {path}"
        )
    return resolved


#: ``data.val_split`` is a ratio and the schema caps it at 0.5 (``le=0.5``).
_MAX_VAL_SPLIT = 0.5


def validation_split(value: Any, key: str, warnings: List[str]) -> Optional[float]:
    """Map a source tool's validation-split setting to ``data.val_split`` (#1807).

    axolotl (``val_set_size``) and LLaMA-Factory (``val_size``) default to 0 and
    train on every row, so an absent key maps to ``0.0`` rather than to Soup's
    default of 0.1. Both tools also accept an absolute row count (a value of 1
    or more); ``data.val_split`` is a ratio capped at 0.5, so a value Soup cannot
    express is named in ``warnings`` and ``None`` is returned, which leaves the
    key out of the written config and Soup's default in force, with the warning
    saying so.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings.append(
            f"{key}: {value!r} could not be read as a number. Soup's default "
            f"data.val_split 0.1 applies until you set it."
        )
        return None
    if 0 <= value <= _MAX_VAL_SPLIT:
        return float(value)
    what = "an absolute row count" if value >= 1 else f"above {_MAX_VAL_SPLIT}"
    warnings.append(
        f"{key}: {value} is {what}; data.val_split is a ratio of at most "
        f"{_MAX_VAL_SPLIT}, so it was not written. Soup's default 0.1 applies until "
        f"you set data.val_split by hand."
    )
    return None


def unmapped_keys_warning(
    raw: Dict[str, Any], handled: Iterable[str], source: str
) -> Optional[str]:
    """One warning naming every top-level source key the migrator did not read (#1807).

    ``handled`` is every key the migrator maps or already warns about by name;
    anything else in ``raw`` would otherwise be dropped in silence. Returns
    ``None`` when nothing was dropped.
    """
    handled_set = set(handled)
    dropped = sorted(str(key) for key in raw if key not in handled_set)
    if not dropped:
        return None
    return (
        f"{len(dropped)} {source} key(s) not migrated (no Soup equivalent, or not read "
        f"by soup migrate): {', '.join(dropped)}"
    )


def split_module_list(value: Any) -> Any:
    """Normalise a LoRA target spelling to the list Soup and peft expect (#1807).

    LLaMA-Factory takes ``lora_target`` as one comma-separated string
    (``q_proj,v_proj``); peft matches a plain string against the whole module
    name, so the string had to become a list. A single name becomes a one-item
    list, a list is returned with its items stripped, and anything else is
    returned unchanged for the schema to judge.
    """
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value]
    return value


def to_number(value: Any) -> Any:
    """Convert string numeric values (e.g. '2e-4') to float/int.

    PyYAML safe_load treats scientific notation like '2e-4' as strings.
    This helper ensures they become proper numeric values.
    """
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def config_to_yaml(config: Dict[str, Any]) -> str:
    """Convert a migration result dict to soup.yaml YAML string.

    Strips internal keys like _warnings before serialization.
    """
    import yaml

    # Remove internal keys
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    return yaml.dump(clean, default_flow_style=False, sort_keys=False, allow_unicode=True)
