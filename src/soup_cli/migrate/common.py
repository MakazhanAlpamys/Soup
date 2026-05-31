"""Common utilities for config migration."""

from pathlib import Path
from typing import Any, Dict

# Max input file size for migration (10 MB — no legitimate config exceeds this)
MAX_CONFIG_FILE_SIZE = 10 * 1024 * 1024


def validate_input_path(path: Path) -> Path:
    """Validate that input path exists and is under cwd (path traversal protection)."""
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    # Check path traversal first (before checking existence)
    try:
        resolved.relative_to(cwd)
    except ValueError:
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
    resolved = (Path.cwd() / path).resolve()
    cwd = Path.cwd().resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ValueError(
            f"Output path is outside the current directory: {path}"
        )
    return resolved


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
