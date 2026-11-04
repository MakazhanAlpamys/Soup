"""Versioned, local-only reward bundles with immutable payload publication.

No MLX or tokenizer imports at module import time. Bundle files are untrusted;
HF cache base symlinks are allowed separately, and their resolved bytes hashed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

MAX_JSON = 4 * 1024 * 1024
ARTIFACT = "mlx_reward_model"
DTYPES = {"F32": ("float32", 4), "F16": ("float16", 2), "BF16": ("bfloat16", 2)}
POLICY = {
    "join": "prompt_space_completion",
    "special_tokens": True,
    "terminal_eos": True,
    "pool": "final_norm_terminal_eos",
    "training": "right_truncate_reserve_eos",
    "scoring": "refuse_overlength",
}


def label(value: Any) -> str:
    """Bound and escape untrusted names; no control codes or rich markup."""
    from soup_cli.utils.terminal import for_terminal

    return for_terminal(ascii(str(value)))[:260]


def bad(member: Any, reason: str) -> ValueError:
    return ValueError(
        f"MLX reward: bundle member {label(member)}: {reason}. Restore an intact bundle."
    )


def _object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(raw: bytes) -> dict:
    def nonfinite(_value: str) -> None:
        raise ValueError("nonfinite JSON number")

    value = json.loads(raw, object_pairs_hook=_object, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")

    # json.loads also accepts overflowing finite decimal syntax such as 1e999.
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for entry in item.values():
                check(entry)
        elif isinstance(item, list):
            for entry in item:
                check(entry)

    check(value)
    return value


@contextmanager
def checked_file(path: Path) -> Iterator[Any]:
    from soup_cli.utils.paths import open_no_follow

    try:
        fd = open_no_follow(
            path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0),
            check_parent=True,
            refuse_hardlink=True,
        )
    except OSError as exc:
        raise bad(path.name, "unavailable, linked or unreadable file") from exc
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise bad(path.name, "expected a regular non-linked file")
        yield stream


def read_json(path: Path) -> dict:
    try:
        with checked_file(path) as stream:
            raw = stream.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            raise ValueError("JSON exceeds 4 MiB")
        return _json(raw)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise bad(path.name, "unreadable or invalid bounded JSON") from exc


def digest(stream: Any) -> str:
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def record(path: Path) -> dict:
    with checked_file(path) as stream:
        return {"size": os.fstat(stream.fileno()).st_size, "sha256": digest(stream)}


def _version(value: dict, member: str) -> None:
    version = value.get("format_version")
    if type(version) is not int or version != 1:
        raise ValueError(
            f"MLX reward: {member} format_version: expected 1, found {label(version)}. "
            "Use a compatible intact bundle."
        )
    if value.get("artifact_type") != ARTIFACT:
        raise bad(member, "wrong artifact_type")


def contained(root: Path, name: str) -> Path:
    from soup_cli.utils.paths import refuse_linked_dirs

    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise bad(name, "invalid relative member name")
    path = root / name
    anchor, target = os.path.realpath(root), os.path.realpath(path)
    if os.path.commonpath([anchor, target]) != anchor:
        raise bad(name, "member escapes bundle")
    try:
        refuse_linked_dirs(path.parent)
    except OSError as exc:
        raise bad(name, "linked parent directory") from exc
    return path


def inventory(root: Path) -> set[str]:
    """Walk without following links; every directory and file is checked."""
    from soup_cli.utils.paths import refuse_linked_dirs

    result = set()
    refuse_linked_dirs(root)
    for parent, dirs, files in os.walk(root, followlinks=False):
        for directory in dirs:
            refuse_linked_dirs(Path(parent) / directory)
        for filename in files:
            path = Path(parent) / filename
            with checked_file(path):
                pass
            result.add(path.relative_to(root).as_posix())
    return result


def read_bundle(root: Path) -> tuple[Path, dict]:
    pointer = read_json(root / "reward_model.json")
    _version(pointer, "reward_model.json")
    if set(pointer) != {"format_version", "artifact_type", "bundle", "manifest_sha256"}:
        raise bad("reward_model.json", "unexpected or missing pointer fields")
    name = pointer.get("bundle")
    if not isinstance(name, str) or not re.fullmatch(r"bundles/[0-9a-f]{32}", name):
        raise bad("reward_model.json", "bundle must be bundles/<32 lowercase hex characters>")
    payload = contained(root, name)
    # Check the payload directory itself, not only parents of the pointer.
    manifest_path = contained(payload, "manifest.json")
    with checked_file(manifest_path) as stream:
        raw = stream.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON or hashlib.sha256(raw).hexdigest() != pointer["manifest_sha256"]:
        raise bad("manifest.json", "size bound or SHA-256 mismatch")
    try:
        manifest = _json(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise bad("manifest.json", "invalid bounded JSON") from exc
    _version(manifest, "manifest.json")
    expected_fields = {
        "format_version",
        "artifact_type",
        "files",
        "base",
        "lora",
        "tokenizer",
        "max_length",
        "policy",
        "head",
        "training",
    }
    if set(manifest) != expected_fields:
        raise bad("manifest.json", "unexpected or missing manifest fields")
    members = manifest.get("files")
    if not isinstance(members, dict) or "reward.safetensors" not in members:
        raise bad("manifest.json", "missing payload inventory")
    if not any(name.startswith("tokenizer/") for name in members):
        raise bad("tokenizer/", "missing tokenizer inventory")
    for name, expected in members.items():
        if name != "reward.safetensors" and not name.startswith("tokenizer/"):
            raise bad(name, "unexpected payload member")
        path = contained(payload, name)
        if (
            not isinstance(expected, dict)
            or set(expected) != {"size", "sha256"}
            or type(expected["size"]) is not int
            or expected["size"] < 0
            or not isinstance(expected["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", expected["sha256"])
        ):
            raise bad(name, "invalid inventory record")
        try:
            found = record(path)
        except OSError as exc:
            raise bad(name, "unavailable or linked file") from exc
        if found != expected:
            raise bad(name, "size or SHA-256 mismatch")
    if inventory(payload) != set(members) | {"manifest.json"}:
        raise bad("manifest.json", "actual member inventory differs")
    validate_contract(manifest)
    return payload, manifest


def _finite(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_contract(manifest: dict) -> None:
    base = manifest.get("base")
    if (
        not isinstance(base, dict)
        or set(base) != {"path", "files"}
        or not isinstance(base["path"], str)
        or not base["path"]
        or not isinstance(base["files"], dict)
        or "config.json" not in base["files"]
    ):
        raise bad("manifest.json", "invalid base identity")
    for name, entry in base["files"].items():
        if (
            not isinstance(name, str)
            or "/" in name
            or "\\" in name
            or not isinstance(entry, dict)
            or set(entry) != {"size", "sha256"}
            or type(entry["size"]) is not int
            or entry["size"] < 0
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])
        ):
            raise bad("manifest.json", "invalid base member record")
    if not isinstance(manifest.get("training"), dict):
        raise bad("manifest.json", "invalid training metadata")
    if type(manifest.get("max_length")) is not int or manifest["max_length"] < 2:
        raise bad("manifest.json", "invalid max_length")
    if manifest.get("policy") != POLICY:
        raise bad("manifest.json", "unsupported tokenization or pooling policy")
    head = manifest.get("head")
    if (
        not isinstance(head, dict)
        or set(head) != {"seed", "std", "dtype", "bias"}
        or type(head["seed"]) is not int
        or head["std"] != 0.02
        or head["dtype"] != "float32"
        or head["bias"] is not False
    ):
        raise bad("manifest.json", "invalid head initialization contract")
    lora = manifest.get("lora")
    if not isinstance(lora, dict) or set(lora) != {"rank", "scale", "dropout", "keys", "layers"}:
        raise bad("manifest.json", "invalid LoRA contract")
    keys = lora["keys"]
    if (
        type(lora["rank"]) is not int
        or not 1 <= lora["rank"] <= 4096
        or type(lora["layers"]) is not int
        or not 1 <= lora["layers"] <= 2**16
        or not isinstance(keys, list)
        or not 1 <= len(keys) <= 1024
        or any(not isinstance(key, str) or not 1 <= len(key) <= 256 for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise bad("manifest.json", "LoRA metadata exceeds bounds")
    for field in ("scale", "dropout"):
        value = lora[field]
        if type(value) not in (int, float) or not _finite(value):
            raise bad("manifest.json", f"invalid LoRA {field}")
    if lora["scale"] <= 0 or not 0 <= lora["dropout"] < 1:
        raise bad("manifest.json", "invalid LoRA scale or dropout")
    tok = manifest.get("tokenizer")
    if (
        not isinstance(tok, dict)
        or set(tok) != {"eos_id", "vocab_size"}
        or type(tok["eos_id"]) is not int
        or type(tok["vocab_size"]) is not int
        or not 0 <= tok["eos_id"] < tok["vocab_size"]
    ):
        raise bad("manifest.json", "invalid tokenizer identity")


def tensor_header(stream: Any, member: str) -> dict:
    """Check layout and exact byte sizes before any tensor allocation."""
    stream.seek(0)
    length_bytes = stream.read(8)
    length = int.from_bytes(length_bytes, "little")
    if len(length_bytes) != 8 or not 0 < length <= MAX_JSON:
        raise bad(member, "invalid safetensors header length (limit 4 MiB)")
    raw = stream.read(length)
    if len(raw) != length:
        raise bad(member, "truncated safetensors header")
    try:
        header = _json(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise bad(member, "invalid safetensors JSON") from exc
    end = 0
    intervals = []
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(meta, dict) or set(meta) != {"dtype", "shape", "data_offsets"}:
            raise bad(key, "invalid tensor metadata")
        shape, offsets, dtype = meta["shape"], meta["data_offsets"], meta["dtype"]
        if (
            not isinstance(dtype, str)
            or dtype not in DTYPES
            or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * DTYPES[dtype][1]
        ):
            raise bad(key, "invalid tensor shape, dtype or byte range")
        intervals.append(tuple(offsets))
    for start, stop in sorted(intervals):
        if start != end:
            raise bad(member, "overlapping or non-contiguous tensor ranges")
        end = stop
    if 8 + length + end != os.fstat(stream.fileno()).st_size:
        raise bad(member, "tensor byte ranges differ from file size")
    return {key: value for key, value in header.items() if key != "__metadata__"}


def base_identity(root: Path) -> dict:
    """Hash the full MLX-LM glob, not just index-referenced shards."""
    # Base snapshots can contain Hub-cache symlinks. Resolve only at this
    # separate trusted-local-base boundary; bundle reads never do this.
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "MLX reward: base member config.json is unavailable. "
            "Provide the identical local/cached base with base=."
        )
    config = read_json(Path(os.path.realpath(config_path)))
    if (
        config.get("model_type") != "llama"
        or "model_file" in config
        or "auto_map" in config
        or "quantization" in config
        or "quantization_config" in config
    ):
        raise ValueError("MLX reward: dense Llama backbone required.")
    for name in ("tokenizer_config.json", "config.json"):
        candidate = root / name
        if candidate.is_file():
            obj = read_json(Path(os.path.realpath(candidate)))
            if "auto_map" in obj or "model_file" in obj:
                raise bad(name, "custom code is not supported")
    unsupported = [root / "hf_quant_config.json", *sorted(root.glob("*.bin")),
                   *sorted(root.glob("*.npz"))]
    for member in unsupported:
        if os.path.lexists(member):
            raise ValueError(
                f"MLX reward: base member {label(member.name)}: unsupported weight format "
                "or quantization sidecar. Provide a clean dense MLX safetensors base."
            )
    weights = sorted(root.glob("model*.safetensors"))
    if not weights:
        raise ValueError(
            "MLX reward: base member model*.safetensors is unavailable. "
            "Provide the identical local/cached base with base=."
        )
    tensors = {}
    for path in weights:
        with open(os.path.realpath(path), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise bad(path.name, "base weight is not regular")
            header = tensor_header(stream, path.name)
        if tensors.keys() & header.keys():
            raise bad(path.name, "duplicate base tensor keys across shards")
        tensors.update({key: path.name for key in header})
    members = [config_path, *weights]
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(Path(os.path.realpath(index))).get("weight_map")
        if mapping != tensors:
            raise bad(index.name, "index and actual globbed shard tensors differ")
        members.append(index)
    result = {}
    for path in sorted(members):
        with open(os.path.realpath(path), "rb") as stream:
            result[path.name] = {
                "size": os.fstat(stream.fileno()).st_size,
                "sha256": digest(stream),
            }
    return {"path": str(root.absolute()), "files": result}


def verify_base(root: Path, expected: dict) -> None:
    if not isinstance(expected, dict) or set(expected) != {"path", "files"}:
        raise bad("manifest.json", "invalid base identity")
    for member in expected["files"]:
        if not (root / member).is_file():
            raise ValueError(
                f"MLX reward: base member {label(member)} is unavailable. "
                "Provide the identical local/cached base with base=."
            )
    found = base_identity(root)
    if set(found["files"]) != set(expected["files"]):
        raise bad("manifest.json", "base member inventory differs; provide matching base=")
    for member, identity in expected["files"].items():
        if found["files"][member] != identity:
            raise ValueError(
                f"MLX reward: base member {label(member)}: expected SHA-256 "
                f"{label(identity.get('sha256'))}, found {found['files'][member]['sha256']}. "
                "Provide the matching base with base=."
            )


def load_tokenizer(root: Path, *, allow_cache_links: bool = False) -> Any:
    from transformers import AutoTokenizer

    from soup_cli.utils.mlx_reward_data import check_tokenizer

    for name in ("tokenizer_config.json", "config.json"):
        if (root / name).exists():
            path = root / name
            config = read_json(Path(os.path.realpath(path)) if allow_cache_links else path)
            if "auto_map" in config or "model_file" in config:
                raise bad("tokenizer/" + name, "custom code is not supported")
    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    check_tokenizer(tokenizer)
    return tokenizer


def read_weights(path: Path, expected: dict, identity: dict) -> dict:
    import mlx.core as mx

    with checked_file(path) as stream:
        if (
            os.fstat(stream.fileno()).st_size != identity["size"]
            or digest(stream) != identity["sha256"]
        ):
            raise bad(path.name, "size or SHA-256 mismatch")
        header = tensor_header(stream, path.name)
        for key in sorted(expected.keys() | header.keys()):
            if key not in header or key not in expected:
                reason = "missing" if key not in header else "unexpected"
                raise ValueError(
                    f"MLX reward: tensor {label(key)}: {reason}. Restore an intact bundle."
                )
            meta, tensor = header[key], expected[key]
            dtype = DTYPES[meta["dtype"]][0]
            expected_dtype = str(tensor.dtype).removeprefix("mlx.core.")
            if meta["shape"] != list(tensor.shape) or dtype != expected_dtype:
                raise ValueError(
                    f"MLX reward: tensor {label(key)}: expected shape={list(tensor.shape)}, "
                    f"dtype={expected_dtype}; found shape={meta['shape']}, dtype={dtype}. "
                    "Restore an intact bundle."
                )
        # The verified descriptor is deliberately reused, never its pathname.
        stream.seek(0)
        weights = mx.load(stream, format="safetensors")
        mx.eval(weights)
    for key, tensor in weights.items():
        if not bool(mx.all(mx.isfinite(tensor)).item()):
            raise ValueError(
                f"MLX reward: tensor {label(key)}: non-finite values. Restore an intact bundle."
            )
    return weights


def load_scorer(artifact: str | Path, *, base: str | Path | None = None) -> Any:
    from soup_cli.utils.mlx_reward_native import require_mlx

    payload, manifest = read_bundle(Path(artifact))
    require_mlx()
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load_model

    from soup_cli.utils.mlx_reward import NativeRewardScorer
    from soup_cli.utils.mlx_reward_native import check_dense_llama, make_reward_model

    root = Path(base if base is not None else manifest["base"]["path"])
    verify_base(root, manifest["base"])
    kwargs = (
        {"trust_remote_code": False}
        if "trust_remote_code" in inspect.signature(load_model).parameters
        else {}
    )
    backbone, _ = load_model(root, **kwargs)
    check_dense_llama(backbone)
    lora = manifest["lora"]
    allowed = {f"self_attn.{name}_proj" for name in ("q", "k", "v", "o")} | {
        f"mlp.{name}_proj" for name in ("gate", "up", "down")
    }
    if lora["layers"] != len(backbone.layers) or not set(lora["keys"]) <= allowed:
        raise bad("manifest.json", "LoRA targets/layers differ from dense Llama contract")
    backbone.freeze()
    linear_to_lora_layers(
        backbone, lora["layers"], {key: value for key, value in lora.items() if key != "layers"}
    )
    model = make_reward_model(backbone, manifest["head"]["seed"])
    expected = dict(tree_flatten(model.trainable_parameters()))
    weights = read_weights(
        payload / "reward.safetensors", expected, manifest["files"]["reward.safetensors"]
    )
    model.load_weights(list(weights.items()), strict=False)
    tokenizer = load_tokenizer(payload / "tokenizer")
    if {"eos_id": tokenizer.eos_token_id, "vocab_size": len(tokenizer)} != manifest["tokenizer"]:
        raise bad("tokenizer/", "saved EOS/vocabulary contract differs")
    if len(tokenizer) > backbone.args.vocab_size:
        raise bad("tokenizer/", "vocabulary exceeds base embeddings")
    model.eval()
    return NativeRewardScorer(model, tokenizer, manifest["max_length"])


def validate_output(root: Path) -> None:
    from soup_cli.utils.paths import refuse_linked_dirs

    refuse_linked_dirs(root)
    legacy = {
        "adapter_config.json",
        "adapters.safetensors",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "pytorch_model.bin",
    }
    if root.exists():
        for path in root.iterdir():
            if path.name in legacy or re.fullmatch(r"model.*\.(safetensors|bin|npz)", path.name):
                raise ValueError(
                    f"MLX reward: output contains a model or adapter file {label(path.name)}. "
                    "Choose a new output directory."
                )
    if os.path.lexists(root / "reward_model.json"):
        # The writer already owns a loaded base. Check the previous publication
        # without rehashing/reloading that base at every save interval.
        read_bundle(root)


def _write_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_JSON:
        raise bad(path.name, "JSON exceeds 4 MiB")
    with open(path, "xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def publish_bundle(root: Path, model: Any, tokenizer: Any, contract: dict) -> dict:
    """Save without moving the training RNG, including validation of an old bundle."""
    import mlx.core as mx

    state = list(mx.random.state)
    try:
        return _publish_bundle(root, model, tokenizer, contract)
    finally:
        mx.random.state = state


def _publish_bundle(root: Path, model: Any, tokenizer: Any, contract: dict) -> dict:
    """Publish last, retain all earlier generations; one training writer per root."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from soup_cli.utils.paths import refuse_linked_dirs

    validate_output(root)
    root.mkdir(parents=True, exist_ok=True)
    bundles = root / "bundles"
    bundles.mkdir(exist_ok=True)
    refuse_linked_dirs(bundles)
    ident = uuid.uuid4().hex
    stage, final = bundles / (".staging-" + ident), bundles / ident
    stage.mkdir()
    pointer_temp = root / (".reward-model-" + ident + ".json")
    published = False
    try:
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(weights)
        mx.save_safetensors(str(stage / "reward.safetensors"), weights)
        tokenizer.save_pretrained(stage / "tokenizer")
        files = {name: record(stage / name) for name in sorted(inventory(stage))}
        manifest = dict(contract, format_version=1, artifact_type=ARTIFACT, files=files)
        validate_contract(manifest)
        # Validate the serialized tensor bytes, not only in-memory parameters.
        read_weights(stage / "reward.safetensors", weights, files["reward.safetensors"])
        saved_tokenizer = load_tokenizer(stage / "tokenizer")
        if {"eos_id": saved_tokenizer.eos_token_id, "vocab_size": len(saved_tokenizer)} != manifest[
            "tokenizer"
        ]:
            raise bad("tokenizer/", "saved tokenizer identity differs")
        _write_json(stage / "manifest.json", manifest)
        sha = record(stage / "manifest.json")["sha256"]
        stage.rename(final)
        _write_json(
            pointer_temp,
            {
                "format_version": 1,
                "artifact_type": ARTIFACT,
                "bundle": "bundles/" + ident,
                "manifest_sha256": sha,
            },
        )
        os.replace(pointer_temp, root / "reward_model.json")
        published = True
        return {
            "bundle": "bundles/" + ident,
            "bytes": sum(record(final / name)["size"] for name in inventory(final)),
        }
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(final, ignore_errors=True)
            pointer_temp.unlink(missing_ok=True)
