"""Prepare immutable inputs for streaming; optionally fetch the frozen public snapshot.

The text is a controlled causal-LM fixture, not a quality evaluation dataset.
Prepare once before the isolated arms; hash every source and derived shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.harness.streaming_controls.compare import save

REVISION = "e9a141a2091ea561b96483212645a2a05e6f99fc"
CORPUS = (
    "A research team compares two ways of reading the same model weights. "
    "The first method streams each layer from a local solid-state drive. "
    "The second keeps selected complete layers in host memory and streams the rest. "
    "Both methods use identical examples, parameters and optimizer settings. "
    "Before measuring throughput, the team compares losses, gradients and adapter updates. "
    "A memory budget includes staging buffers, resident tensors and a safety reserve. "
    "Asynchronous transfers must finish before their source buffers are released. "
    "Physical device counters include unrelated activity, so calibration and ownership matter. "
    "Warmup and model preparation are reported separately from training. "
    "A failed control stops the experiment. A faster run alone does not prove correctness.\n"
)


def tensor_digest(tensors: dict[str, Any]) -> str:
    """Hash raw tensor bytes, including bf16 which NumPy cannot represent directly."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_snapshot(destination: Path, deadline_unix_s: float) -> None:
    """One bounded download per frozen file. No credentials, resume or retries."""
    if (
        type(deadline_unix_s) not in (int, float)
        or not math.isfinite(deadline_unix_s)
        or deadline_unix_s <= time.time()
    ):
        raise TimeoutError("snapshot deadline already expired")
    expected = json.loads(Path(__file__).with_name("representative_model.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {"status": "DOWNLOADING", "files": {}}
    try:
        for name, row in expected["files"].items():
            if Path(name).name != name:
                raise ValueError("snapshot manifest must contain simple filenames")
            if time.time() >= deadline_unix_s:
                raise TimeoutError("snapshot deadline expired")
            url = f"https://huggingface.co/{expected['repo']}/resolve/{REVISION}/{name}"
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(
                url, timeout=min(30, deadline_unix_s - time.time())
            ) as response:
                if response.status != 200:
                    raise RuntimeError("full snapshot download requires HTTP 200")
                with (destination / name).open("xb") as handle:
                    while block := response.read(1 << 20):
                        if time.time() >= deadline_unix_s:
                            raise TimeoutError("snapshot deadline expired")
                        count += len(block)
                        if count > row["bytes"]:
                            raise ValueError(f"oversized snapshot file: {name}")
                        digest.update(block)
                        handle.write(block)
            if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"download hash or size mismatch: {name}")
            receipt["files"][name] = {"bytes": count, "sha256": digest.hexdigest()}
            save(destination / "download_receipt.json", receipt)
        receipt["status"] = "PASS_DOWNLOAD_SHA256"
    except BaseException as error:
        receipt.update(status="FAILED", error=repr(error))
        raise
    finally:
        save(destination / "download_receipt.json", receipt)


def write_prepared(
    weights: Path, output: Path, tokens: list[list[int]], identity: dict[str, Any], *, smoke: bool
) -> Path:
    """Shared preparation path; smoke inputs can never be used for CUDA acceptance."""
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = weights.absolute()
    config = json.loads((weights / "config.json").read_text())
    length = 8 if smoke else 256
    if len(tokens) != 6 or any(
        len(row) != length
        or any(type(x) is not int or not 0 <= x < config["vocab_size"] for x in row)
        for row in tokens
    ):
        raise ValueError("expected six complete fixed-length token batches without padding")
    if not smoke and (
        config.get("model_type") != "llama"
        or config.get("hidden_size") != 4096
        or config.get("num_hidden_layers") != 32
        or config.get("vocab_size") != 128256
        or identity.get("revision") != REVISION
    ):
        raise ValueError("the representative workload requires the frozen Llama 3.1 8B snapshot")
    output.mkdir(parents=True, exist_ok=False)
    directory = output / "shards"
    dtype = "float32" if smoke else "bfloat16"
    shard_checkpoint(str(weights), str(directory), dtype=dtype, arch="llama")
    files = {
        path.relative_to(output).as_posix(): {
            "sha256": file_sha(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    payload = {
        "kind": "cpu-fixture" if smoke else "llama31-8b",
        "weights": str(weights),
        "config_sha256": file_sha(weights / "config.json"),
        "identity": identity,
        "dtype": dtype,
        "tokens": tokens,
        "effective_tokens_per_step": length - 1,
        "steps": 6,
        "warmup": 2,
        "limit_bytes": 10_000_000 if smoke else 16 << 30,
        "reserved_bytes": 0 if smoke else 6 << 30,
        "cache_bytes": 45_000 if smoke else 4 << 30,
        "files": files,
        "scope": "pretrained dense model on controlled text; no quality or speed verdict",
    }
    save(output / "prepared.json", payload)
    return output / "prepared.json"


def prepare(snapshot: Path, output: Path) -> Path:
    """Verify against the in-repo model manifest before sharding or tokenizer use."""
    expected = json.loads(Path(__file__).with_name("representative_model.json").read_text())
    for name, row in expected["files"].items():
        path = snapshot / name
        if (
            path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or file_sha(path) != row["sha256"]
        ):
            raise ValueError(f"snapshot identity mismatch: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    stream = tokenizer.encode(CORPUS * 32, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    if bos is None or len(stream) < 6 * 255:
        raise ValueError("tokenizer did not produce the fixed workload")
    tokens = [[bos] + stream[step * 255 : (step + 1) * 255] for step in range(6)]
    frozen = json.loads(Path(__file__).with_name("representative_tokens.json").read_text())
    if (
        tokens != frozen["tokens"]
        or hashlib.sha256(CORPUS.encode()).hexdigest() != frozen["corpus_sha256"]
    ):
        raise ValueError("tokenizer output or corpus differs from frozen local preparation")
    return write_prepared(snapshot, output, tokens, expected, smoke=False)


def load_prepared(path: Path, *, cuda: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("kind") != ("llama31-8b" if cuda else "cpu-fixture"):
        raise ValueError("prepared workload kind does not match execution mode")
    recipe = {
        "dtype": "bfloat16" if cuda else "float32",
        "steps": 6,
        "warmup": 2,
        "effective_tokens_per_step": 255 if cuda else 7,
        "limit_bytes": 16 << 30 if cuda else 10_000_000,
        "reserved_bytes": 6 << 30 if cuda else 0,
        "cache_bytes": 4 << 30 if cuda else 45_000,
    }
    if any(
        type(payload.get(key)) is not type(value) or payload[key] != value
        for key, value in recipe.items()
    ):
        raise ValueError("prepared recipe differs from the fixed comparison")
    if cuda:
        expected = json.loads(Path(__file__).with_name("representative_model.json").read_text())
        if payload["identity"] != expected:
            raise ValueError("representative model identity differs")
        if payload["config_sha256"] != expected["files"]["config.json"]["sha256"]:
            raise ValueError("representative model configuration differs")
        frozen = json.loads(Path(__file__).with_name("representative_tokens.json").read_text())
        if payload["tokens"] != frozen["tokens"]:
            raise ValueError("representative tokens differ from frozen local preparation")
    weights = Path(payload["weights"])
    if file_sha(weights / "config.json") != payload["config_sha256"]:
        raise ValueError("model configuration changed after preparation")
    config = json.loads((weights / "config.json").read_text())
    tokens = payload.get("tokens", [])
    if len(tokens) != 6 or any(
        len(row) != recipe["effective_tokens_per_step"] + 1
        or any(type(x) is not int or not 0 <= x < config["vocab_size"] for x in row)
        for row in tokens
    ):
        raise ValueError("prepared token batch count, length or token IDs invalid")
    from soup_cli.utils.paths import is_under

    members = payload["files"]
    if "shards/index.json" not in members or not any(x.endswith(".safetensors") for x in members):
        raise ValueError("prepared shard manifest incomplete")
    for name, row in members.items():
        member = Path(name)
        item = path.parent / member
        if member.is_absolute() or ".." in member.parts or not is_under(item, path.parent):
            raise ValueError("invalid prepared file path")
        if (
            item.is_symlink()
            or item.stat().st_size != row["bytes"]
            or file_sha(item) != row["sha256"]
        ):
            raise ValueError(f"prepared shard changed: {name}")
    return payload


def validate_preparation_destination(output: Path) -> None:
    """Read-only admission before download; shard_checkpoint still enforces its own guard."""
    from soup_cli.utils.paths import is_under

    raw = str(output)
    if any(ord(char) < 0x20 for char in raw) or output.is_symlink():
        raise ValueError("invalid preparation output path")
    if output.exists():
        raise FileExistsError("prepared output must be new; no retry or overwrite")
    resolved = os.path.realpath(output / "shards")
    bounds = [
        os.path.realpath(os.path.expanduser("~")),
        os.path.realpath(os.getcwd()),
        os.path.realpath(tempfile.gettempdir()),
    ]
    if not any(is_under(resolved, bound) for bound in bounds):
        raise ValueError("prepared shards must be under HOME, CWD or TMPDIR")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-deadline-unix-s",
        type=float,
        help="Download the fixed snapshot once into a NEW snapshot directory",
    )
    args = parser.parse_args()
    from rich.console import Console

    # Reject a bad launch layout before paying for the model download.
    validate_preparation_destination(args.output)

    if args.download_deadline_unix_s is not None:
        fetch_snapshot(args.snapshot, args.download_deadline_unix_s)
    Console().print(str(prepare(args.snapshot, args.output)))


if __name__ == "__main__":
    main()
