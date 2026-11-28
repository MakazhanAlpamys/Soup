"""Plan the giant MoE cache from pinned source headers and the actual writer layout."""

import argparse
import gc
import hashlib
import io
import json
import os
import re
import urllib.request
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console

from soup_cli.utils.checkpoint_files import checkpoint_files
from soup_cli.utils.expert_cache_budget import (
    expert_cache_budget_from_headers,
    new_expert_cache_budget,
    read_source_header,
)
from soup_cli.utils.fp8_block import fp8_block_size

MODEL_ID = "deepseek-ai/DeepSeek-V3"
MODEL_TYPE = "deepseek_v3"
DECODER_LAYERS = 61
SOURCE_SHARD_COUNT = 163
SOURCE_BYTES = 688_586_727_753


def source_file(source: Path, name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"model-\d+-of-\d+\.safetensors", name):
        raise ValueError("source plan requires top-level model shard names")
    root = os.path.realpath(source)
    entry = os.path.join(root, name)
    path = os.path.realpath(entry)
    try:
        contained = os.path.commonpath([root, path]) == root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError("planned source escapes checkpoint directory")
    if os.path.islink(entry):
        raise ValueError("source plan requires regular checkpoint shards, not symlinks")
    return Path(path)


def source_shards(source: Path) -> list[Path]:
    """Use the same shard selection as the writer and require the entire source."""
    selected = checkpoint_files(os.path.realpath(source))
    actual = sorted(path.name for path in source.glob("*.safetensors"))
    if selected.missing or not actual or actual != sorted(selected.weights):
        raise ValueError("source plan shard selection mismatch")
    return [source_file(source, name) for name in actual]


def require_complete_source(sizes: Sequence[int]) -> None:
    if (
        len(sizes) != SOURCE_SHARD_COUNT
        or any(type(size) is not int or size <= 0 for size in sizes)
        or sum(sizes) != SOURCE_BYTES
    ):
        raise ValueError("giant MoE requires the complete pinned 163-file DeepSeek-V3 source")


def validate_plan(source: Path, path: Path, revision: str) -> dict[str, Any]:
    """Bind a plan to downloaded source headers before trusting its disk bound."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(plan.get("version")) is not int
        or plan["version"] != 1
        or plan.get("source_revision") != revision
        or hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
        != plan.get("config_sha256")
    ):
        raise ValueError("source plan revision/config mismatch")
    paths = source_shards(source)
    records = plan["files"]
    if [row["name"] for row in records] != [file.name for file in paths]:
        raise ValueError("source plan shard selection mismatch")
    for row, file in zip(records, paths, strict=True):
        size = file.stat().st_size
        with file.open("rb") as handle:
            raw, _ = read_source_header(handle, size)
        if size != row["size"] or hashlib.sha256(raw).hexdigest() != row["header_sha256"]:
            raise ValueError("downloaded source differs from planned header/size")
    if sum(row["size"] for row in records) != plan["source_bytes"]:
        raise ValueError("source plan byte total mismatch")
    filenames = [str(file) for file in paths]
    budget = new_expert_cache_budget(
        filenames,
        quant_suffixes=plan["quant_suffixes"],
        n_layers=plan["n_layers"],
    )
    if (
        type(plan["cache_budget_bytes"]) is not int
        or budget != plan["cache_budget_bytes"]
        or type(plan["generic_cache_budget_bytes"]) is not int
        or new_expert_cache_budget(filenames) != plan["generic_cache_budget_bytes"]
    ):
        raise ValueError("source plan cache bound mismatch")
    return plan


def validate_writer_plan(
    plan: dict[str, Any],
    suffixes: Sequence[str],
    n_layers: int,
    dtype: str,
) -> None:
    if (
        sorted(suffixes) != plan["quant_suffixes"]
        or n_layers != plan["n_layers"]
        or dtype != plan["dtype"]
    ):
        raise ValueError("source plan quantization/layout differs from the actual writer")


def read_remote_header(url: str, size: int) -> tuple[bytes, dict[str, Any]]:
    """Refuse an ignored Range before reading any weight payload into memory."""

    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        )
        expected = end - start + 1
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
            ):
                raise ValueError("source server did not honor the bounded header range")
            raw = response.read(expected + 1)
        if len(raw) != expected:
            raise ValueError("source server returned a truncated or oversized header range")
        return raw

    prefix = fetch(0, 7)
    length = int.from_bytes(prefix, "little")
    if not 2 <= length <= min(64 * 2**20, size - 8):
        raise ValueError("invalid remote safetensors header length")
    raw = fetch(8, 7 + length)
    return read_source_header(io.BytesIO(prefix + raw), size)


@contextmanager
def meta_construction() -> Iterator[None]:
    import torch

    # The stream cache replaces the parameters. Skip deterministic poison fills on
    # temporary CPU allocations that Accelerate redirects to meta.
    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def writer_geometry(source: Path, dtype: str) -> tuple[list[str], int]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != MODEL_TYPE
        or fp8_block_size(config.get("quantization_config"), arch=MODEL_TYPE) is None
    ):
        raise ValueError("this pilot accepts DeepSeek-V3 FP8 only")
    n_layers = config.get("num_hidden_layers")
    if type(n_layers) is not int or n_layers != DECODER_LAYERS:
        raise ValueError("giant MoE requires the 61-layer DeepSeek-V3 decoder")

    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
    )

    with meta_construction():
        skeleton = build_meta_skeleton(str(source), dtype=dtype, quant="nf4", double_quant=False)
    suffixes = sorted(quantised_layer_suffixes(skeleton))
    del skeleton
    gc.collect()
    return suffixes, n_layers


def collect_plan(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

    info = HfApi(token=False).model_info(args.model, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Hub metadata does not match the pinned source commit")
    files = sorted(
        (row for row in info.siblings if row.rfilename.endswith(".safetensors")),
        key=lambda row: row.rfilename,
    )
    require_complete_source([row.size for row in files])
    output = Path(args.output)
    for row in files:
        source_file(output, row.rfilename)
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / "config"
    hf_hub_download(
        args.model,
        "config.json",
        revision=args.revision,
        local_dir=config_dir,
        token=False,
    )
    dtype = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float32"
    suffixes, n_layers = writer_geometry(config_dir, dtype)

    def fetch(row: Any) -> tuple[dict[str, Any], tuple[int, int, dict[str, Any]]]:
        remote = hf_hub_url(args.model, row.rfilename, revision=args.revision)
        raw, header = read_remote_header(remote, row.size)
        (output / (row.rfilename + ".header.json")).write_bytes(raw)
        record = {
            "name": row.rfilename,
            "size": row.size,
            "header_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return record, (row.size, len(raw), header)

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, files))
    headers = [header for _, header in rows]
    plan = {
        "version": 1,
        "model": args.model,
        "source_revision": args.revision,
        "config_sha256": hashlib.sha256((config_dir / "config.json").read_bytes()).hexdigest(),
        "source_bytes": sum(row.size for row in files),
        "files": [record for record, _ in rows],
        "dtype": dtype,
        "n_layers": n_layers,
        "quant_suffixes": suffixes,
        "cache_budget_bytes": expert_cache_budget_from_headers(
            headers,
            quant_suffixes=suffixes,
            n_layers=n_layers,
        ),
        "generic_cache_budget_bytes": expert_cache_budget_from_headers(headers),
        "scope": "header-only planning; all source payloads must still be downloaded; "
        "the runtime must independently match this actual writer layout",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {key: value for key, value in plan.items() if key != "files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    Console().print(collect_plan(parser.parse_args()))
