"""Versioned, atomic MLX SFT continuation, using the native trainer hooks (#1977).

No backend imports at module scope. Bundle discovery/validation works on CPU-only
hosts. Staged directories are never checkpoints; a corrupt committed directory
is an error, never a reason to silently choose a legacy adapter.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import tempfile
from pathlib import Path

_BUNDLE = re.compile(r"^mlx-state-(\d+)$")
_MEMBERS = {
    "adapters.safetensors",
    "optimizer.safetensors",
    "random.safetensors",
    "state.json",
    "adapter_config.json",
}
_MAX_JSON = 16 * 1024 * 1024


def validate_config(cfg):
    """Refuse unsupported full-state requests without changing fresh-training scope."""
    tcfg, lora = cfg.training, cfg.training.lora
    if cfg.backend != "mlx" or cfg.task != "sft" or cfg.modality != "text":
        raise ValueError("mlx_full_state requires MLX text SFT")
    if tcfg.optimizer not in {"adamw_torch", "adamw_hf", "adamw_torch_fused"}:
        raise ValueError("mlx_full_state requires a declared AdamW alias")
    if (
        lora.r < 1
        or lora.use_dora
        or lora.use_vera
        or lora.use_rslora
        or lora.init_strategy != "random"
        or lora.rank_pattern
        or lora.alpha_pattern
        or tcfg.use_lorafa
        or tcfg.loraplus_lr_ratio is not None
    ):
        raise ValueError("mlx_full_state requires ordinary linear LoRA without variants/patterns")
    if tcfg.quantization not in {"none", "4bit"}:
        raise ValueError("mlx_full_state supports dense or genuine 4bit bases")


def is_bundle(path):
    path = Path(path)
    return bool(_BUNDLE.fullmatch(path.name)) or (
        path.is_dir() and (path / "manifest.json").exists()
    )


def discover_bundle(directory):
    candidates = [
        (int(m.group(1)), p) for p in Path(directory).iterdir() if (m := _BUNDLE.fullmatch(p.name))
    ]
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _json(path):
    try:
        if Path(path).stat().st_size > _MAX_JSON:
            raise ValueError("JSON member exceeds size limit")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid MLX checkpoint {Path(path).name}: {exc}") from exc


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _dataset_digest(dataset):
    # Stream rows instead of duplicating the complete tokenized corpus as one JSON string.
    digest = hashlib.sha256()
    if dataset is not None:
        for index in range(len(dataset)):
            digest.update(_canonical(dataset[index]).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _member(directory, name):
    path = directory / name
    root, actual = os.path.realpath(directory), os.path.realpath(path)
    if path.is_symlink() or os.path.commonpath([root, actual]) != root or not path.is_file():
        raise ValueError(f"Invalid checkpoint member path: {name}")
    return path


def read_bundle(path, *, staged=False):
    directory = Path(path)
    if directory.name.startswith(".mlx-state-") and not staged:
        raise ValueError("Uncommitted MLX staging directory is not a checkpoint")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Invalid MLX checkpoint directory (symlinks are refused)")
    manifest = _json(_member(directory, "manifest.json"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "soup-mlx-state"
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
    ):
        raise ValueError("Unsupported MLX checkpoint format/version")
    for field in ("update", "microbatch", "total_updates", "accumulation"):
        if type(manifest.get(field)) is not int or manifest[field] < 1:
            raise ValueError(f"Invalid MLX checkpoint progress: {field}")
    if (
        manifest["microbatch"] != manifest["update"] * manifest["accumulation"]
        or manifest["update"] > manifest["total_updates"]
    ):
        raise ValueError("Inconsistent MLX checkpoint progress/budget")
    match = _BUNDLE.fullmatch(directory.name)
    if match and int(match.group(1)) != manifest["update"]:
        raise ValueError("MLX checkpoint directory/update mismatch")
    members = manifest.get("members")
    if not isinstance(members, dict) or set(members) != _MEMBERS:
        raise ValueError("Invalid MLX checkpoint member set")
    if not isinstance(manifest.get("contract"), dict):
        raise ValueError("Invalid MLX checkpoint compatibility contract")
    for name, info in members.items():
        member = _member(directory, name)
        if (
            not isinstance(info, dict)
            or type(info.get("size")) is not int
            or info["size"] < 1
            or member.stat().st_size != info["size"]
            or not isinstance(info.get("sha256"), str)
            or _hash(member) != info["sha256"]
        ):
            raise ValueError(f"MLX checkpoint integrity failure: {name}")
    return manifest


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, list) else value


def _python_state(state):
    if (
        not isinstance(state, list)
        or len(state) != 3
        or type(state[0]) is not int
        or state[0] != 3
        or not isinstance(state[1], list)
        or len(state[1]) != 625
        or any(type(v) is not int or not 0 <= v < 2**32 for v in state[1][:-1])
        or type(state[1][-1]) is not int
        or not 0 <= state[1][-1] <= 624
        or (
            state[2] is not None
            and (type(state[2]) not in (int, float) or not math.isfinite(state[2]))
        )
    ):
        raise ValueError("Invalid Python random state")
    return _tuple(state)


def _numpy_state(state):
    import numpy as np

    if (
        not isinstance(state, list)
        or len(state) != 5
        or state[0] != "MT19937"
        or not isinstance(state[1], list)
        or len(state[1]) != 624
        or any(type(v) is not int or not 0 <= v < 2**32 for v in state[1])
        or type(state[2]) is not int
        or not 0 <= state[2] <= 624
        or type(state[3]) is not int
        or state[3] not in (0, 1)
        or type(state[4]) not in (int, float)
        or not math.isfinite(state[4])
    ):
        raise ValueError("Invalid NumPy random state")
    return (state[0], np.array(state[1], dtype=np.uint32), *state[2:])


def _numpy_json(state):
    return [state[0], state[1].tolist(), *state[2:]]


def _restore_random(keys):
    import mlx.core as mx

    # Current MLX exposes a read-only state container, but its key arrays support
    # in-place assignment. This also works with the floor's list container.
    for index, key in enumerate(keys):
        mx.random.state[index][:] = key


@contextlib.contextmanager
def preserve_random():
    """Instrumentation/evaluation must not move the training random streams."""
    import mlx.core as mx
    import numpy as np

    state = (list(mx.random.state), np.random.get_state(), random.getstate())
    mx.eval(state[0])
    try:
        yield
    finally:
        _restore_random(state[0])
        np.random.set_state(state[1])
        random.setstate(state[2])


class BatchCursor:
    """Checkpointable length-bucket permutation, matching native single-device batches."""

    def __init__(self, dataset, batch_size, seed, state=None):
        import numpy as np

        indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
        self.batches = [
            indices[i : i + batch_size] for i in range(0, len(indices) - batch_size + 1, batch_size)
        ]
        if not self.batches:
            raise ValueError("MLX full-state dataset must contain a complete batch")
        self.rng = np.random.RandomState(seed)
        self.order, self.position, self.epoch = [], 0, 0
        if state is not None:
            order, position, epoch = state.get("order"), state.get("position"), state.get("epoch")
            if (
                not isinstance(order, list)
                or any(type(v) is not int for v in order)
                or sorted(order) != list(range(len(self.batches)))
                or type(position) is not int
                or not 0 <= position <= len(order)
                or type(epoch) is not int
                or epoch < 1
            ):
                raise ValueError("Invalid MLX sampler permutation/cursor")
            self.order, self.position, self.epoch = order, position, epoch
            self.rng.set_state(_numpy_state(state.get("random")))

    def next(self):
        if self.position == len(self.order):
            self.order = self.rng.permutation(len(self.batches)).tolist()
            self.position = 0
            self.epoch += 1
        batch = self.batches[self.order[self.position]]
        self.position += 1
        return batch

    def state(self):
        return {
            "order": self.order,
            "position": self.position,
            "epoch": self.epoch,
            "random": _numpy_json(self.rng.get_state()),
        }


class _Subset:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def _schema(tensors):
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in tensors.items()
    }


def _check_tensors(actual, expected, label):
    import mlx.core as mx

    if _schema(actual) != _schema(expected):
        raise ValueError(f"Incompatible MLX checkpoint {label} tensor schema")
    if any(not mx.isfinite(value).all().item() for value in actual.values()):
        raise ValueError(f"Nonfinite MLX checkpoint {label} tensors")


def _load_checked(path, expected, label):
    """Check the bounded tensor header against live shapes before allocating payloads."""
    import mlx.core as mx

    codes = {
        "float32": "F32",
        "float16": "F16",
        "bfloat16": "BF16",
        "uint32": "U32",
        "uint64": "U64",
        "int32": "I32",
        "int64": "I64",
        "bool_": "BOOL",
    }
    try:
        size = path.stat().st_size
        with open(path, "rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if not 0 < length <= min(_MAX_JSON, size - 8):
                raise ValueError("invalid header length")
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError("invalid header object")
        header.pop("__metadata__", None)
        if set(header) != set(expected):
            raise ValueError("unexpected tensor keys")
        ranges = []
        for name, value in expected.items():
            item = header[name]
            if (
                not isinstance(item, dict)
                or item.get("shape") != list(value.shape)
                or any(type(v) is not int for v in item.get("shape", []))
                or item.get("dtype") != codes[str(value.dtype).split(".")[-1]]
            ):
                raise ValueError("incompatible tensor shape/dtype")
            offsets = item.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(v) is not int for v in offsets)
                or offsets[0] < 0
                or offsets[1] - offsets[0] != value.nbytes
            ):
                raise ValueError("invalid tensor data offsets")
            ranges.append(offsets)
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError("overlapping or gapped tensors")
            cursor = end
        if cursor != size - 8 - length:
            raise ValueError("truncated or trailing tensor payload")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid MLX checkpoint {label} tensor header: {exc}") from exc
    actual = mx.load(str(path))
    _check_tensors(actual, expected, label)
    return actual


class FullStateSession:
    """Bind native callback/iterator hooks to one verified original training plan."""

    def __init__(
        self,
        *,
        cfg,
        model,
        optimizer,
        plan,
        train_dataset,
        val_dataset,
        args,
        iterate,
        metadata,
        resume,
    ):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        validate_config(cfg)
        if mx.distributed.init().size() != 1:
            raise ValueError("MLX full-state continuation requires one device/process")
        base = Path(cfg.base)
        if not base.is_dir():
            raise ValueError("MLX full-state checkpoints require a local pinned base snapshot")
        self.cfg, self.model, self.optimizer, self.args = cfg, model, optimizer, args
        self.iterate_base, self.dataset, self.metadata = iterate, train_dataset, metadata
        self.accum, self.total = args.grad_accumulation_steps, plan.total_updates
        self.start_micro, self.cursor = 0, None
        self.output = Path(cfg.output)
        self.every = cfg.training.save_steps if cfg.training.save_steps > 0 else self.total
        self.seed = (
            cfg.training.data_seed
            if cfg.training.data_seed is not None
            else (cfg.training.seed if cfg.training.seed is not None else 42)
        )
        if any(
            not key.endswith((".lora_a", ".lora_b"))
            for key, _ in tree_flatten(model.trainable_parameters())
        ):
            raise ValueError("MLX full-state requires ordinary linear LoRA targets")
        config = cfg.model_dump(mode="json")
        # These presentation/recording/path values cannot change the next training update.
        for name in ("output", "experiment_name", "name", "description"):
            config.pop(name, None)
        for name in ("logging_steps", "rewind_log", "save_steps", "mlx_full_state"):
            config["training"].pop(name, None)
        versions = {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "numpy", "transformers", "tokenizers")
        }
        files = {
            str(p.relative_to(base)): _hash(p)
            for p in sorted(base.rglob("*"))
            if p.is_file()
            and (
                p.suffix in {".json", ".safetensors", ".model", ".jinja"}
                or p.name in {"merges.txt", "vocab.txt"}
            )
        }
        import mlx.nn as nn
        from mlx_lm.tuner.lora import LoRALinear

        adapted = [
            module
            for _, module in model.named_modules()
            if hasattr(module, "lora_a") or hasattr(module, "lora_b")
        ]
        if not adapted or any(
            not isinstance(module, LoRALinear)
            or not (
                isinstance(module.linear, nn.Linear)
                or (isinstance(module.linear, nn.QuantizedLinear) and module.linear.bits == 4)
            )
            for module in adapted
        ):
            raise ValueError("MLX full-state requires dense/4bit ordinary linear LoRA targets")
        self.contract = {
            "recipe": config,
            "base_files": files,
            "train": _dataset_digest(train_dataset),
            "validation": _dataset_digest(val_dataset),
            "optimizer": plan.as_metadata(),
            "adapter": metadata["lora_parameters"],
            "runtime": {
                **versions,
                "python": platform.python_version(),
                "os": platform.platform(),
                "device": mx.device_info(),
            },
        }
        self.bundle = Path(resume) if resume and is_bundle(resume) else None
        latest = discover_bundle(self.output)
        if latest is not None:
            if self.bundle is None:
                raise ValueError(
                    "Existing MLX full-state checkpoints: use --resume auto or fresh output"
                )
            if os.path.realpath(self.bundle) != os.path.realpath(latest):
                raise ValueError(
                    "MLX populated output requires its latest checkpoint; "
                    "use fresh output to resume an external or earlier checkpoint"
                )
        self.published = {}

    def prepare(self):
        """Validate everything first, then restore after setup and before the next draw."""
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten

        state = None
        if self.bundle is not None:
            manifest = read_bundle(self.bundle)
            if manifest["contract"] != self.contract:
                changed = [
                    key
                    for key in self.contract
                    if manifest["contract"].get(key) != self.contract[key]
                ]
                raise ValueError(
                    f"Incompatible MLX full-state checkpoint: {', '.join(changed)}; "
                    "restore the original recipe/runtime or use a fresh output"
                )
            state = _json(self.bundle / "state.json")
            if not isinstance(state, dict) or set(state) != {"sampler", "numpy", "python"}:
                raise ValueError("Invalid MLX checkpoint random/sampler state")
            if not isinstance(state["sampler"], dict):
                raise ValueError("Invalid MLX sampler state")
            self.start_micro = manifest["microbatch"]
            if manifest["total_updates"] != self.total or manifest["accumulation"] != self.accum:
                raise ValueError("Incompatible MLX checkpoint original budget/accumulation")
            cursor = BatchCursor(self.dataset, self.args.batch_size, self.seed, state["sampler"])
            if (cursor.epoch - 1) * len(cursor.order) + cursor.position != self.start_micro:
                raise ValueError("Inconsistent MLX sampler/progress")
            numpy_state = _numpy_state(state["numpy"])
            python_state = _python_state(state["python"])
            adapter_metadata = _json(self.bundle / "adapter_config.json")
            expected_metadata = {k: v for k, v in self.metadata.items() if k != "adapter_path"}
            if (
                not isinstance(adapter_metadata, dict)
                or {k: v for k, v in adapter_metadata.items() if k != "adapter_path"}
                != expected_metadata
            ):
                raise ValueError("Incompatible MLX checkpoint adapter metadata")
            adapters = _load_checked(
                self.bundle / "adapters.safetensors",
                dict(tree_flatten(self.model.trainable_parameters())),
                "adapter",
            )
            self.optimizer.init(self.model.trainable_parameters())
            saved_optimizer = _load_checked(
                self.bundle / "optimizer.safetensors",
                dict(tree_flatten(self.optimizer.state)),
                "optimizer",
            )
            if int(saved_optimizer["step"].item()) != manifest["update"]:
                raise ValueError("MLX optimizer counter/progress mismatch")
            saved_random = _load_checked(
                self.bundle / "random.safetensors",
                {str(i): key for i, key in enumerate(mx.random.state)},
                "random",
            )
            # No state is applied until all members, identities and schemas passed.
            self.model.load_weights(list(adapters.items()), strict=False)
            self.optimizer.state = tree_unflatten(list(saved_optimizer.items()))
            _restore_random([saved_random[str(i)] for i in range(len(saved_random))])
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            self.cursor = cursor
            self.published[os.path.realpath(self.bundle)] = manifest
        else:
            random.seed(self.cfg.training.seed if self.cfg.training.seed is not None else 42)
            self.cursor = BatchCursor(self.dataset, self.args.batch_size, self.seed)
        self.args.iters -= self.start_micro
        self.args.steps_per_report = self.accum
        # The native saver counts local iterations; session publication counts global updates.
        self.args.steps_per_save = self.args.iters + 1

    def iterate(self, **kwargs):
        if not kwargs.get("loop"):
            with preserve_random():
                yield from self.iterate_base(**kwargs)
            return
        if self.cursor is None:
            raise RuntimeError("Full-state sampler must be prepared before training")
        while True:
            indices = self.cursor.next()
            with preserve_random():
                # Reuse native padding/truncation/masking on one already sorted complete batch.
                batch = next(
                    self.iterate_base(
                        **{
                            **kwargs,
                            "dataset": _Subset(kwargs["dataset"], indices),
                            "loop": False,
                            "comm_group": None,
                        }
                    )
                )
            yield batch

    def save(self, microbatch):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        update = microbatch // self.accum
        if microbatch % self.accum or int(self.optimizer.step.item()) != update:
            raise ValueError("MLX full-state save requires a completed optimizer update")
        target = self.output / f"mlx-state-{update}"
        if target.exists():
            if read_bundle(target) != self.published.get(os.path.realpath(target)):
                raise ValueError("MLX checkpoint output collision; use fresh output")
            return
        stage = Path(tempfile.mkdtemp(prefix=".mlx-state-staging-", dir=self.output))
        try:
            with preserve_random():
                tensors = dict(tree_flatten(self.model.trainable_parameters()))
                optimizer = dict(tree_flatten(self.optimizer.state))
                random_keys = {str(i): key for i, key in enumerate(mx.random.state)}
                mx.eval(tensors, optimizer, random_keys)
                for label, values in [("adapter", tensors), ("optimizer", optimizer)]:
                    _check_tensors(values, values, label)
                mx.save_safetensors(str(stage / "adapters.safetensors"), tensors)
                mx.save_safetensors(str(stage / "optimizer.safetensors"), optimizer)
                mx.save_safetensors(str(stage / "random.safetensors"), random_keys)
                import numpy as np

                state = {
                    "sampler": self.cursor.state(),
                    "numpy": _numpy_json(np.random.get_state()),
                    "python": random.getstate(),
                }
                (stage / "state.json").write_text(_canonical(state))
                (stage / "adapter_config.json").write_text(_canonical(self.metadata))
                manifest = {
                    "format": "soup-mlx-state",
                    "version": 1,
                    "update": update,
                    "microbatch": microbatch,
                    "total_updates": self.total,
                    "accumulation": self.accum,
                    "contract": self.contract,
                    "members": {
                        name: {"size": (stage / name).stat().st_size, "sha256": _hash(stage / name)}
                        for name in _MEMBERS
                    },
                }
                (stage / "manifest.json").write_text(_canonical(manifest))
                read_bundle(stage, staged=True)
                for member in stage.iterdir():
                    with open(member, "rb") as stream:
                        os.fsync(stream.fileno())
                os.rename(stage, target)
                self.published[os.path.realpath(target)] = manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def callback(self, inner):
        session = self

        class Callback:
            def on_train_loss_report(self, info):
                info = {**info, "iteration": info["iteration"] + session.start_micro}
                micro = info["iteration"]
                with preserve_random():
                    if micro % max(1, session.cfg.training.logging_steps) == 0 or (
                        micro == session.total * session.accum
                    ):
                        inner.on_train_loss_report(info)
                if micro // session.accum % session.every == 0 or (
                    micro == session.total * session.accum
                ):
                    session.save(micro)

            def on_val_loss_report(self, info):
                offset = 0 if info.get("full_state_global") else session.start_micro
                with preserve_random():
                    inner.on_val_loss_report({**info, "iteration": info["iteration"] + offset})

        return Callback()
