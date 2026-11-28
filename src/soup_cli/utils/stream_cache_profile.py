"""Opt-in reuse of measured cache costs, including cold startup and job length.

Profiles are performance hints. They cannot increase configured cache ceilings
or replace current RAM admission and the full-shape GPU candidate probe.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROFILE_VERSION = 1
PROFILE_MAX_BYTES = 2**20
PROFILE_MAX_AGE = 30 * 86400
PROFILE_WARMUP = 2
PROFILE_SAMPLES = 6


def _is_finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class CacheMeasurement:
    ram_mib: int
    resident_count: int
    startup_seconds: float
    step_seconds: float
    ram_indices: tuple[int, ...] = ()
    gpu_indices: tuple[int, ...] = ()
    baseline_allocated: int = 0
    baseline_reserved: int = 0
    retained_layers: int = 0
    record_only_seconds: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.ram_mib, self.resident_count, self.baseline_allocated,
                      self.baseline_reserved, self.retained_layers):
            if type(value) is not int or value < 0:
                raise ValueError("cache profile counts must be non-negative integers")
        if (not _is_finite_number(self.startup_seconds) or self.startup_seconds < 0
                or not _is_finite_number(self.step_seconds) or self.step_seconds <= 0):
            raise ValueError("cache profile timings must be finite and positive")
        if (type(self.record_only_seconds) not in (int, float)
                or not 0 <= self.record_only_seconds <= self.startup_seconds):
            raise ValueError("record-only time must be within the measured startup")
        for indices in (self.ram_indices, self.gpu_indices):
            if (any(type(idx) is not int or idx < 0 for idx in indices)
                    or tuple(sorted(set(indices))) != tuple(indices)):
                raise ValueError("invalid cache profile layer indices")
        if self.resident_count != len(self.gpu_indices):
            raise ValueError("cache profile resident count differs from indices")
        if not self.ram_mib and self.ram_indices:
            raise ValueError("disabled RAM cache has layer indices")
        if self.resident_count and not 0 < self.baseline_allocated <= self.baseline_reserved:
            raise ValueError("resident profile requires the original baseline footprint")
        if set(self.ram_indices) & set(self.gpu_indices):
            raise ValueError("profile RAM and GPU caches must be disjoint")

    @property
    def ordinary(self) -> bool:
        return self.ram_mib == self.resident_count == self.retained_layers == 0

    @property
    def placement(self) -> tuple[int, int, int]:
        return self.ram_mib, self.resident_count, self.retained_layers

    def estimate(self, updates: int) -> float:
        return self.startup_seconds - self.record_only_seconds + updates * self.step_seconds


def profile_identity(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def choose_cache(
    rows: list[CacheMeasurement], updates: int, *, ram_mib: int,
    resident_limit: int | str, allow_retention: bool = False,
) -> CacheMeasurement | None:
    """Only measured, permitted candidates; ordinary streaming wins close calls."""
    if type(updates) is not int or updates < 1 or type(ram_mib) is not int or ram_mib < 0:
        raise ValueError("invalid cache profile workload or RAM ceiling")
    if resident_limit != "auto" and (type(resident_limit) is not int or resident_limit < 0):
        raise ValueError("invalid cache profile GPU ceiling")
    baseline = next((row for row in rows if row.ordinary), None)
    if baseline is None or updates < PROFILE_WARMUP:
        return baseline
    allowed = [row for row in rows if row.ram_mib <= ram_mib
               and (resident_limit == "auto" or row.resident_count <= resident_limit)
               and (allow_retention or not row.retained_layers)]
    if not allowed:
        return baseline
    selected = min(allowed, key=lambda row: row.estimate(updates))
    return selected if selected.estimate(updates) < .98 * baseline.estimate(updates) else baseline


def _profile_path(path: str | Path) -> Path:
    target = Path(path)
    if not profile_path_allowed(target):
        raise ValueError(
            "cache profile must stay under the home, working or temp directory"
        )
    if target.is_symlink():
        raise ValueError("cache profile must not be a symlink")
    parent = os.path.realpath(target.parent)
    resolved = os.path.realpath(target)
    if os.path.normcase(os.path.commonpath((parent, resolved))) != os.path.normcase(parent):
        raise ValueError("cache profile escapes its parent")
    return target


def profile_path_allowed(path: str | Path) -> bool:
    """True when ``path`` resolves under the home directory, the working directory or the
    temp directory (realpath + commonpath, so a symlinked parent cannot widen the set)."""
    try:
        candidate = os.path.realpath(path)
        bases = [os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.getcwd()),
                 os.path.realpath(tempfile.gettempdir())]
    except (OSError, ValueError):
        return False
    for base in bases:
        try:
            if os.path.normcase(os.path.commonpath((candidate, base))) == os.path.normcase(base):
                return True
        except ValueError:
            continue
    return False


#: A profile file nests about five levels deep. The bound is checked on the raw bytes before
#: parsing, because the C JSON parser's own recursion limit differs between platforms and
#: Python versions.
PROFILE_MAX_JSON_DEPTH = 32


def _json_nesting_exceeds(raw: bytes, limit: int) -> bool:
    """True when ``[``/``{`` nesting outside JSON strings goes deeper than ``limit``."""
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ {
            depth += 1
            if depth > limit:
                return True
        elif byte in (0x5D, 0x7D):  # ] }
            depth -= 1
    return False


def _read_profiles(path: Path) -> dict[str, Any]:
    from soup_cli.utils.paths import open_no_follow

    try:
        _profile_path(path)
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("must be a regular file")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
            with os.fdopen(open_no_follow(path, flags), "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("must be a regular file")
                raw = source.read(PROFILE_MAX_BYTES + 1)
        except FileNotFoundError:
            return {"version": PROFILE_VERSION, "profiles": {}}
        if len(raw) > PROFILE_MAX_BYTES:
            raise ValueError("exceeds the size limit")
        if _json_nesting_exceeds(raw, PROFILE_MAX_JSON_DEPTH):
            raise ValueError("JSON nesting is too deep")
        try:
            data = json.loads(raw)
        except RecursionError as error:
            raise ValueError("JSON nesting is too deep") from error
        if (not isinstance(data, dict) or data.get("version") != PROFILE_VERSION
                or not isinstance(data.get("profiles"), dict)
                or len(data["profiles"]) > 64):
            raise ValueError("unsupported format")
    except (ValueError, OSError) as error:
        raise ValueError(f"cache profile {path}: {error}") from error
    return data


def _parse_profile_entry(
    path: str | Path, identity: str, entry: Any,
) -> tuple[float, list[CacheMeasurement]]:
    try:
        if not isinstance(entry, dict) or "updated" not in entry or "rows" not in entry:
            raise ValueError("invalid cache profile entry")
        updated = entry["updated"]
        if not _is_finite_number(updated):
            raise ValueError("cache profile timestamp must be finite")
        if not isinstance(entry["rows"], list) or len(entry["rows"]) > 16:
            raise ValueError("invalid cache profile candidate list")
        try:
            rows = [CacheMeasurement(**{
                **row, "ram_indices": tuple(row["ram_indices"]),
                "gpu_indices": tuple(row["gpu_indices"]),
            }) for row in entry["rows"]]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid cache profile candidate ({type(error).__name__})") from error
        if len({row.placement for row in rows}) != len(rows):
            raise ValueError("duplicate cache profile candidate")
    except ValueError as error:
        raise ValueError(f"cache profile {path}: entry {identity[:12]}: {error}") from error
    return updated, rows


def validate_profile_file(path: str | Path) -> None:
    """Refuse a malformed existing profile before any model is loaded.

    A missing file is fine (nothing recorded yet). Every refusal names the file.
    """
    for identity, entry in _read_profiles(Path(path))["profiles"].items():
        _parse_profile_entry(path, identity, entry)


def load_measurements(path: str | Path, identity: str) -> list[CacheMeasurement]:
    profiles = _read_profiles(Path(path))["profiles"]
    if identity not in profiles:
        return []
    updated, rows = _parse_profile_entry(path, identity, profiles[identity])
    age = time.time() - updated
    return rows if -300 <= age <= PROFILE_MAX_AGE else []


def save_measurement(path: str | Path, identity: str, row: CacheMeasurement) -> None:
    """Atomic bounded JSON; concurrent recorders leave the existing file intact."""
    target = _profile_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary = None
    try:
        data = _read_profiles(target)
        rows = load_measurements(target, identity)
        rows = [item for item in rows if item.placement != row.placement]
        rows.append(row)
        if len(rows) > 16:
            raise ValueError("too many cache profile candidates")
        serialized = []
        for item in rows:
            fields = asdict(item)
            if not item.retained_layers:
                del fields["retained_layers"]
            if not item.record_only_seconds:
                del fields["record_only_seconds"]
            serialized.append(fields)
        data["profiles"][identity] = {"updated": time.time(), "rows": serialized}
        if len(data["profiles"]) > 64:
            oldest = min(data["profiles"], key=lambda key: data["profiles"][key]["updated"])
            del data["profiles"][oldest]
        payload = json.dumps(data, sort_keys=True, allow_nan=False).encode()
        if len(payload) > PROFILE_MAX_BYTES:
            raise ValueError("cache profile exceeds size limit")
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".stream-profile-",
                                         delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_driver_version(uuid: str) -> str | None:
    from soup_cli.utils.layer_stream import _resolve_tool

    tool = _resolve_tool("nvidia-smi")
    if tool is None:
        return None
    try:
        driver = subprocess.check_output(
            [tool, f"--id={uuid}", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return None
    return driver if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver) else None


def _runtime_identity(
    owner: Any, cfg: Any, args: Any, index_path: Path, optimizer: Any,
) -> str:
    import torch

    dataset = owner.trainer.train_dataset
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("cache profiles require a fingerprinted finite training dataset")
    if getattr(args, "world_size", 1) != 1:
        raise ValueError("cache profiles require single-device training")
    properties = torch.cuda.get_device_properties(owner._stream_runtime.pool.device)
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid or uuid == "None":
        raise ValueError("cache profiles require a stable GPU UUID")
    if not uuid.startswith(("GPU-", "MIG-")):
        uuid = "GPU-" + uuid
    driver = _read_driver_version(uuid)
    if driver is None:
        raise ValueError("cache profiles require the actual GPU driver identity")
    from soup_cli.utils.stream_ram_cache import available_memory

    ram = available_memory()
    training = cfg.training.model_dump(mode="json")
    for name in ("epochs", "stream_ram_cache_mb", "stream_resident_layers",
                 "stream_cache_profile", "stream_cache_profile_mode"):
        training.pop(name, None)
    # Retention is a placement choice. Keep the original key for recompute-on
    # runs so existing measurements can compete with new retention candidates.
    training["stream_recompute"] = "on"
    source = Path(__file__).parent
    return profile_identity({
        "implementation": PROFILE_VERSION,
        "source": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                   for name in ("layer_stream.py", "layer_stream_runtime.py",
                                "stream_cache_recompute.py", "stream_residency.py",
                                "stream_ram_cache.py", "async_disk_source.py")},
        "model": cfg.base, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "dataset": fingerprint, "training": training, "data": cfg.data.model_dump(mode="json"),
        "hardware": {"name": properties.name, "uuid": uuid, "driver": driver,
                     "vram": properties.total_memory, "capability": [properties.major,
                                                                         properties.minor]},
        "ram_limits": sorted(x["limit_bytes"] for x in ram["constraints"]),
        "runtime": {name: importlib.metadata.version(name)
                    for name in ("torch", "bitsandbytes", "transformers", "peft")},
        "cuda": torch.version.cuda,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized": torch.utils.deterministic.fill_uninitialized_memory,
        "optimizer": {"class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                      "defaults": optimizer.defaults},
        "batch": args.per_device_train_batch_size,
        "accumulation": args.gradient_accumulation_steps,
    })


def _clear_profile_caches(owner: Any, model: Any) -> None:
    import torch

    from soup_cli.utils.layer_stream_runtime import clear_stream_substitution_cache

    runtime = owner._stream_runtime
    torch.cuda.synchronize(runtime.pool.device)
    clear_stream_substitution_cache(model)
    runtime.source.configure_ram_cache(0)
    runtime.source.set_resident_layers(())
    runtime.pool.clear_resident_layers()
    torch.cuda.empty_cache()


def build_stream_cache_callback(
    owner: Any, cfg: Any, *, index_path: Path, setup_started: float, notify: Any,
) -> Any:
    """Record eight real updates or reuse history without a calibration sweep."""
    from transformers import TrainerCallback

    from soup_cli.utils.terminal import for_terminal

    profile_file = cfg.training.stream_cache_profile

    class CacheProfileCallback(TrainerCallback):
        def __init__(self) -> None:
            self.recording = False
            self.identity = None
            self.samples: list[float] = []
            self.first_step_at: float | None = None
            self.step_started = 0.0
            self.signature: dict[str, Any] = {}
            self.plan: Any = None
            self.record_only_seconds = 0.0

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            import torch

            from soup_cli.utils.stream_residency import ResidencyHint, configure_stream_residency

            runtime, model = owner._stream_runtime, kwargs["model"]
            if runtime.tier != "disk" or not runtime.pool.is_cuda:
                notify("Stream cache profile: ordinary streaming; requires CUDA async disk tier.")
                return control
            if getattr(state, "global_step", 0):
                notify("Stream cache profile: ordinary streaming on checkpoint resume.")
                return control
            try:
                self.identity = _runtime_identity(owner, cfg, args, index_path, kwargs["optimizer"])
            except (
                ValueError, TypeError, OSError, importlib.metadata.PackageNotFoundError,
            ) as error:
                notify(f"Stream cache profile: ordinary streaming ({for_terminal(error)}).")
                return control
            tcfg = cfg.training
            count, ram_mib, hint = tcfg.stream_resident_layers, tcfg.stream_ram_cache_mb, None
            allow_retention = tcfg.stream_recompute == "auto"
            selected = None
            if tcfg.stream_cache_profile_mode == "reuse":
                try:
                    selected = choose_cache(
                        load_measurements(tcfg.stream_cache_profile, self.identity),
                        state.max_steps, ram_mib=ram_mib, resident_limit=count,
                        allow_retention=allow_retention,
                    )
                except (ValueError, KeyError, TypeError, OSError) as error:
                    notify(f"Stream cache profile: invalid history ({for_terminal(error)}); "
                           "ordinary streaming.")
                    return control
                if selected is None or selected.ordinary:
                    notify("Stream cache profile: ordinary streaming; no amortized matching cache.")
                    return control
                indices = (*selected.ram_indices, *selected.gpu_indices)
                if (selected.retained_layers > runtime.n_layers
                        or any(idx >= runtime.n_layers for idx in indices)):
                    notify("Stream cache profile: invalid layer indices; ordinary streaming.")
                    return control
                count, ram_mib = selected.resident_count, selected.ram_mib
                if count:
                    hint = ResidencyHint(count, selected.baseline_allocated,
                                         selected.baseline_reserved)
            try:
                rows, seq_len, vocab = owner._stream_step_shape(cfg, tcfg, model.config)
                self.record_only_seconds = 0.0
                if count:
                    self.plan = configure_stream_residency(
                        model, runtime, "auto" if hint else count, rows=rows, seq_len=seq_len,
                        vocab_size=vocab, override_bytes=tcfg.stream_vram_override,
                        notify=notify, **({"hint": hint} if hint else {}),
                    )
                    self.record_only_seconds += getattr(self.plan, "record_only_seconds", 0.0)
                    if selected is not None and not self.plan.enabled:
                        notify("Stream cache profile: current GPU probe refused the candidate; "
                               "ordinary streaming.")
                        return control
                if ram_mib:
                    runtime.source.configure_ram_cache(ram_mib * 2**20)
                host = runtime.source.ram_cache_stats()
                gpu = tuple(sorted(runtime.pool.resident_slots))
                if selected is not None and (tuple(host["indices"]) != selected.ram_indices
                                             or gpu != selected.gpu_indices):
                    _clear_profile_caches(owner, model)
                    notify("Stream cache profile: current memory changed the plan; "
                           "ordinary streaming.")
                    return control
                retained = 0
                if allow_retention and (selected is None or selected.retained_layers):
                    from soup_cli.utils.stream_cache_recompute import (
                        configure_cached_stream_recompute,
                    )

                    retention = configure_cached_stream_recompute(
                        model, runtime, rows=rows, seq_len=seq_len, vocab_size=vocab,
                        override_bytes=tcfg.stream_vram_override, notify=notify,
                        retain=selected.retained_layers if selected is not None else None,
                    )
                    retained = retention.retained_layers
                    self.record_only_seconds += getattr(retention, "record_only_seconds", 0.0)
                    if selected is not None and retained != selected.retained_layers:
                        _clear_profile_caches(owner, model)
                        notify("Stream cache profile: current VRAM refused retained activations; "
                               "ordinary streaming.")
                        return control
                self.signature = dict(ram_mib=ram_mib, resident_count=len(gpu),
                    ram_indices=tuple(host["indices"]), gpu_indices=gpu,
                    baseline_allocated=getattr(self.plan, "baseline_peak_allocated_bytes", 0),
                    baseline_reserved=getattr(self.plan, "baseline_peak_reserved_bytes", 0),
                    retained_layers=retained)
                self.recording = selected is None and (not allow_retention or retained > 0)
                owner._stream_cache_profile_selection = dict(self.signature)
                torch.cuda.synchronize(runtime.pool.device)
                if selected is None and not self.recording:
                    notify("Stream cache profile: retention did not fit. Training continues "
                           "with recomputation and the history is unchanged.")
                    return control
                notify(f"Stream cache profile: {len(gpu)} GPU layers, RAM ceiling {ram_mib} MiB, "
                       f"{retained} layers retaining activations; "
                       f"{'recording real updates' if self.recording else 'reused; no sweep'}.")
            except BaseException:
                owner._close_stream_runtime()
                raise
            return control

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if self.recording:
                import torch

                torch.cuda.synchronize(owner._stream_runtime.pool.device)
                self.step_started = time.perf_counter()
                if self.first_step_at is None:
                    self.first_step_at = self.step_started
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if not self.recording:
                return control
            import torch

            torch.cuda.synchronize(owner._stream_runtime.pool.device)
            self.samples.append(time.perf_counter() - self.step_started)
            if len(self.samples) == PROFILE_WARMUP + PROFILE_SAMPLES:
                self.recording = False
                step = sum(self.samples[PROFILE_WARMUP:]) / PROFILE_SAMPLES
                startup = max(0., self.first_step_at - setup_started
                              + sum(self.samples[:PROFILE_WARMUP]) - PROFILE_WARMUP * step)
                try:
                    save_measurement(profile_file, self.identity, CacheMeasurement(
                        **self.signature, startup_seconds=startup, step_seconds=step,
                        record_only_seconds=min(startup, self.record_only_seconds),
                    ))
                    notify(f"Stream cache profile saved to {for_terminal(profile_file)}; "
                           "actual training continued.")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    notify("Stream cache profile could not be saved "
                           f"({for_terminal(error)}); training continues.")
            return control

    return CacheProfileCallback()
