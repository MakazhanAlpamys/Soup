"""Data loading from local files and HuggingFace."""

import json
from pathlib import Path

from rich.console import Console

from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import detect_format, format_to_messages

console = Console()

# File extensions we support
SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet"}


def load_raw_data(path: Path) -> list[dict]:
    """Load raw data from a file into list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".jsonl":
        return _load_jsonl(path)
    elif ext == ".json":
        return _load_json(path)
    elif ext == ".csv":
        return _load_csv(path)
    elif ext == ".parquet":
        return _load_parquet(path)

    raise ValueError(f"Unsupported format: {ext}")


def _load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Warning: invalid JSON on line {i + 1}: {e}[/]")
    return data


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    raise ValueError("JSON file must contain a list of objects")


def _load_csv(path: Path) -> list[dict]:
    import csv

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas to read parquet files: pip install pandas pyarrow")
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def load_dataset(data_config: DataConfig) -> dict:
    """Load dataset for training. Returns dict with 'train' and optionally 'val' keys.

    Supports:
    - Local files (.jsonl, .json, .csv, .parquet)
    - HuggingFace dataset names (auto-detected if no file extension)
    """
    train_path = data_config.train

    # Check if it's a HuggingFace dataset
    if not Path(train_path).suffix:
        return _load_hf_dataset(train_path, data_config)

    # Local file
    path = Path(train_path)
    raw_data = load_raw_data(path)

    # Detect or use specified format
    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)
        console.print(f"[dim]Auto-detected format: {fmt}[/]")

    # Convert to standard message format
    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]  # filter failed rows

    # Split into train/val
    if data_config.val_split > 0:
        split_idx = int(len(formatted) * (1 - data_config.val_split))
        return {
            "train": formatted[:split_idx],
            "val": formatted[split_idx:],
        }

    return {"train": formatted}


def _load_hf_dataset(name: str, data_config: DataConfig) -> dict:
    """Load a dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    console.print(f"[dim]Loading from HuggingFace: {name}[/]")
    ds = hf_load(name)

    if "train" not in ds:
        raise ValueError(f"Dataset {name} has no 'train' split")

    raw_data = [dict(row) for row in ds["train"]]
    fmt = data_config.format
    if fmt == "auto":
        fmt = detect_format(raw_data)

    formatted = [format_to_messages(row, fmt) for row in raw_data]
    formatted = [r for r in formatted if r is not None]

    if data_config.val_split > 0 and "validation" not in ds:
        split_idx = int(len(formatted) * (1 - data_config.val_split))
        return {"train": formatted[:split_idx], "val": formatted[split_idx:]}

    result = {"train": formatted}
    if "validation" in ds:
        val_data = [dict(row) for row in ds["validation"]]
        val_formatted = [format_to_messages(row, fmt) for row in val_data]
        result["val"] = [r for r in val_formatted if r is not None]

    return result
