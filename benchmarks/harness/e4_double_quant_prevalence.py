"""E4 part 1: how widespread is the bitsandbytes double-quant absmax defect?

Rule: benchmarks/probe-e4-double-quant-prevalence.md section 1, committed before any run.

    python benchmarks/harness/e4_double_quant_prevalence.py select --out DIR
    python benchmarks/harness/e4_double_quant_prevalence.py run --model deepseek-v3 --out DIR
    python benchmarks/harness/e4_double_quant_prevalence.py summarize --out DIR

Tensors are read with HTTP range requests from pinned repo revisions: only the
bytes of each selected tensor are downloaded. ``run`` appends one JSON line per
tensor to ``DIR/<model>.jsonl`` and skips keys already there, so an interrupted
run resumes instead of restarting. Failed tensors go to ``DIR/<model>.void.jsonl``.

Rule V4 ("a failed fetch, OOM or killed run is kept as void and reported, not
silently retried"):

* One HTTP request per fetch, no transport retry. Every request is journaled in
  the append-only ``DIR/<model>.attempts.jsonl`` (flushed and fsynced per line):
  a ``start`` event before it and an ``ok`` or ``fail`` event after it.
* A failed request voids its tensor with reason ``fetch_failed``. The other
  reasons are ``v1``, ``v2``, ``oom_or_killed`` (a MemoryError the process
  survived) and ``error``.
* On resume a key with a row is skipped, and a key with a void is reported and
  NOT measured again. A key with journal events but neither a row nor a void was
  in flight when the run stopped (interrupted fetch, fetched one ahead, or the
  process killed while computing, which is all a hard OOM kill leaves behind):
  it gets a void with reason ``interrupted`` and is not measured again either.
* ``run --remeasure-voids`` measures voided keys again. The new requests carry
  ``supersedes_attempt`` (the void's attempt id) and so does the new row; the
  void is kept. ``summarize`` refuses a void and a row for one key without
  that link.
* ``summarize`` writes ``summary.json`` (measurement aggregates) and
  ``attempts-summary.json`` (void counts by reason, remeasured keys, request
  counts). Results written before the journal existed have no attempt file and
  are reported as ``provenance: no per-attempt log``, never as zero failures.

The FP8 reference decoder here is independent of soup_cli (bit-field table, fp32
multiply, one bf16 rounding); each decode is cross-checked against B3's
``dequantize_fp8_blockwise`` and a mismatch voids the tensor (rule V2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUND = 0.17
NF4_BLOCK = 64
GROUP = 256
SEED = 20261008
SPREAD_EDGES = (1.0, 10.0, 100.0, 1000.0, float("inf"))
HTTP_TIMEOUT = 120
VOID_REASONS = ("fetch_failed", "interrupted", "v1", "v2", "oom_or_killed", "error")

MODELS = {
    "deepseek-v3": {
        "repo": "deepseek-ai/DeepSeek-V3",
        "rev": "e815299b0bcbac849fa540c768ef21845365c9eb",
        "fp8": True,
        "n_experts": 256,
        "moe_layers": list(range(3, 62)),
        "expected": {"attention": 310, "dense_mlp": 9, "shared_expert": 177, "routed_expert": 354},
    },
    "kimi-k2": {
        "repo": "moonshotai/Kimi-K2-Instruct",
        "rev": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
        "fp8": True,
        "n_experts": 384,
        "moe_layers": list(range(1, 61)),
        "expected": {"attention": 305, "dense_mlp": 3, "shared_expert": 180, "routed_expert": 360},
    },
    "mistral-7b": {
        "repo": "mistralai/Mistral-7B-v0.1",
        "rev": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
        "fp8": False,
    },
    "qwen2.5-7b": {
        "repo": "Qwen/Qwen2.5-7B",
        "rev": "d149729398750b98c0af14eb82c78cfe92750796",
        "fp8": False,
    },
    "olmoe-1b-7b": {
        "repo": "allenai/OLMoE-1B-7B-0924",
        "rev": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "fp8": False,
    },
}

_DENSE_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")
_ROUTED_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
_ST_DTYPES = {"F8_E4M3": "float8_e4m3fn", "BF16": "bfloat16", "F16": "float16", "F32": "float32"}


# --------------------------------------------------------------------------
# HTTP range reads
# --------------------------------------------------------------------------
def _url(model: dict, filename: str) -> str:
    return f"https://huggingface.co/{model['repo']}/resolve/{model['rev']}/{filename}"


class FetchFailed(Exception):  # noqa: N818 - the name the review of PR #10 asked for
    """One HTTP request failed. Carries the cause; rule V4 forbids retrying it here."""

    def __init__(self, url: str, exc: BaseException) -> None:
        super().__init__(f"GET {url} failed: {exc!r}")
        self.exc = exc
        self.attempt_id: str | None = None


class RuleVoidError(ValueError):
    """A validity check of the rule (V1 or V2) failed for one tensor."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _get(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    """GET ``url`` (bytes ``start..stop-1`` if given). One request, never retried."""
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{stop - 1}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return response.read()
    except OSError as exc:  # URLError, timeouts and resets are all OSError
        raise FetchFailed(url, exc) from exc


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Journal:
    """Append-only log of every HTTP request of one model's runs (rule V4)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()  # the prefetch thread and the main thread both write
        self._seq = sum(event["event"] == "start" for event in _read_jsonl(path))
        self.supersedes: dict = {}  # key -> attempt id of the void being remeasured
        self.last_attempt: dict = {}  # key -> id of its latest request in this process

    def start(self, key: str | None, filename: str, byte_range: list | None) -> str:
        attempt_id = uuid.uuid4().hex
        with self._lock:
            event = {
                "event": "start", "attempt_id": attempt_id, "key": key, "seq": self._seq,
                "file": filename, "range": byte_range, "ts": _now(),
            }
            if key in self.supersedes:
                event["supersedes_attempt"] = self.supersedes[key]
            self._seq += 1
            self.last_attempt[key] = attempt_id
            _append_jsonl(self.path, event)
        return attempt_id

    def end(self, attempt_id: str, ok: bool, size: int | None, exc: BaseException | None) -> None:
        with self._lock:
            _append_jsonl(self.path, {
                "event": "ok" if ok else "fail", "attempt_id": attempt_id, "bytes": size,
                "exc": None if exc is None else repr(exc), "ts": _now(),
            })


class Checkpoint:
    """Lazily fetched safetensors headers for one pinned repo revision."""

    def __init__(self, name: str, journal: Journal | None = None) -> None:
        self.name = name
        self.model = MODELS[name]
        self.journal = journal
        index = json.loads(self._request(None, "model.safetensors.index.json"))
        self.weight_map: dict = index["weight_map"]
        self._headers: dict = {}

    def _request(self, key: str | None, filename: str, start: int | None = None,
                 stop: int | None = None) -> bytes:
        """One journaled request made on behalf of the selected tensor ``key``."""
        url = _url(self.model, filename)
        if self.journal is None:
            return _get(url, start, stop)
        attempt_id = self.journal.start(key, filename, None if start is None else [start, stop])
        try:
            raw = _get(url, start, stop)
        except FetchFailed as failed:
            failed.attempt_id = attempt_id
            self.journal.end(attempt_id, False, None, failed.exc)
            raise
        # Anything else (KeyboardInterrupt, a kill) leaves the start without an end.
        self.journal.end(attempt_id, True, len(raw), None)
        return raw

    def header(self, filename: str, owner: str | None = None) -> tuple[dict, int]:
        if filename not in self._headers:
            size = int.from_bytes(self._request(owner, filename, 0, 8), "little")
            header = json.loads(self._request(owner, filename, 8, 8 + size))
            header.pop("__metadata__", None)
            self._headers[filename] = (header, 8 + size)
        return self._headers[filename]

    def entry(self, key: str) -> dict:
        header, _ = self.header(self.weight_map[key])
        return header[key]

    def fetch(self, key: str, owner: str | None = None):
        """The tensor ``key`` as an owned torch tensor. Rule V1: byte count checked.

        ``owner`` is the selected key the request is journaled under (an FP8
        weight owns the request for its ``_scale_inv`` tensor).
        """
        import torch

        owner = owner or key
        filename = self.weight_map[key]
        header, base = self.header(filename, owner)
        entry = header[key]
        start, stop = entry["data_offsets"]
        raw = self._request(owner, filename, base + start, base + stop)
        if len(raw) != stop - start:
            raise RuleVoidError(
                "v1", f"V1: {key} fetched {len(raw)} bytes, header says {stop - start}"
            )
        dtype = getattr(torch, _ST_DTYPES[entry["dtype"]])
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"])


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def _fp8_kind(key: str) -> str | None:
    if ".self_attn." in key:
        return "attention"
    if ".mlp.shared_experts." in key:
        return "shared_expert"
    if _ROUTED_RE.match(key):
        return "routed_expert"
    if _DENSE_RE.match(key):
        return "dense_mlp"
    return None


def select_fp8(ckpt: Checkpoint) -> tuple[list[dict], dict]:
    """The rule's sample: all attention / dense / shared FP8 weights, 2 drawn experts per layer."""
    model = ckpt.model
    weight_map = ckpt.weight_map
    rng = random.Random(SEED)
    drawn = {
        layer: sorted(rng.sample(range(model["n_experts"]), 2)) for layer in model["moe_layers"]
    }
    selected = []
    for key in sorted(weight_map):
        if not key.endswith(".weight") or key + "_scale_inv" not in weight_map:
            continue
        kind = _fp8_kind(key)
        if kind is None:
            continue
        if kind == "routed_expert":
            match = _ROUTED_RE.match(key)
            if int(match.group(2)) not in drawn.get(int(match.group(1)), ()):
                continue
        selected.append({"key": key, "kind": kind})
    counts = {kind: sum(s["kind"] == kind for s in selected) for kind in model["expected"]}
    if counts != model["expected"]:
        raise SystemExit(
            f"{ckpt.name}: sample counts {counts} differ from the rule {model['expected']}"
        )
    return selected, {str(layer): experts for layer, experts in drawn.items()}


def select_candidate(ckpt: Checkpoint) -> list[dict]:
    """Every 2-D weight in model.layers.*, minus norms and MoE router gates (rule 1.2)."""
    selected = []
    for key in sorted(ckpt.weight_map):
        if not key.startswith("model.layers.") or not key.endswith(".weight"):
            continue
        if "norm" in key or key.endswith(".mlp.gate.weight"):
            continue
        if len(ckpt.entry(key)["shape"]) != 2:
            continue
        if ".self_attn." in key:
            kind = "attention"
        elif ".mlp.experts." in key:
            kind = "routed_expert"
        else:
            kind = "dense_mlp"
        selected.append({"key": key, "kind": kind})
    return selected


# --------------------------------------------------------------------------
# reference decode (independent of soup_cli) and the B3 cross-check
# --------------------------------------------------------------------------
def _e4m3_table():
    import torch

    values = []
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent, mantissa = (byte >> 3) & 0xF, byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values.append(float("nan"))
        elif exponent == 0:
            values.append(sign * (mantissa / 8.0) * 2.0**-6)
        else:
            values.append(sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return torch.tensor(values, dtype=torch.float32)


def reference_decode(q, scale, table, block=(128, 128), chunk=1024):
    import torch

    rows, cols = q.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    as_bytes = q.view(torch.uint8)
    scale32 = scale.to(torch.float32)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        values = table[as_bytes[start:stop].long()]
        expanded = scale32[torch.arange(start, stop) // block[0]].repeat_interleave(block[1], dim=1)
        out[start:stop] = (values * expanded[:, :cols]).to(torch.bfloat16)
    return out


class _Slice:
    """The safetensors-slice surface dequantize_fp8_blockwise reads."""

    def __init__(self, tensor) -> None:
        self._tensor = tensor

    def get_dtype(self) -> str:
        return "F8_E4M3"

    def get_shape(self) -> list:
        return list(self._tensor.shape)

    def __getitem__(self, index):
        return self._tensor[index]


def b3_decode(q, scale):
    try:
        from soup_cli.utils.fp8_source import dequantize_fp8_blockwise
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from soup_cli.utils.fp8_source import dequantize_fp8_blockwise
    return dequantize_fp8_blockwise(
        _Slice(q), scale, block=(128, 128), dtype="bfloat16", key="e4-cross-check"
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _blocks(flat, block):
    import torch

    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat((flat, flat.new_zeros(pad)))
    return flat.reshape(-1, block)


def nf4_metrics(ref, double_quant: bool) -> dict:
    """Rule section 1.1 metrics for one tensor and one double-quant setting."""
    import torch
    from bitsandbytes.functional import dequantize_4bit, dequantize_blockwise, quantize_4bit

    packed, state = quantize_4bit(
        ref, blocksize=NF4_BLOCK, compress_statistics=double_quant, quant_type="nf4"
    )
    back = dequantize_4bit(packed, state)
    finite = bool(torch.isfinite(back).all())
    r = _blocks(ref.float().reshape(-1), NF4_BLOCK)
    d = _blocks(back.float().reshape(-1), NF4_BLOCK)
    del back, packed
    diff = d - r
    err = diff.abs().amax(dim=1)
    rel_fro = (diff.norm() / r.norm()).item()
    del diff, d
    absmax = r.abs().amax(dim=1)
    del r
    zero = absmax == 0
    over = torch.where(zero, err > 0, err > BOUND * absmax)
    ratio = torch.where(zero, torch.zeros_like(err), err / absmax.clamp_min(1e-38))
    if double_quant:
        stored = dequantize_blockwise(state.absmax, state.state2) + state.offset
    else:
        stored = state.absmax
    stored = stored.float().reshape(-1)[: absmax.numel()]
    flipped = (stored <= 0) & ~zero
    # Descriptive only, added after the 3-tensor smoke test: stored absmax more
    # than 10% off its true value, sign flips included. Feeds no rule decision.
    off10 = ((stored - absmax).abs() > 0.1 * absmax) & ~zero
    return {
        "frac_over": over.float().mean().item(),
        "worst": ratio.max().item(),
        "frac_flipped": flipped.float().mean().item(),
        "frac_absmax_off10": off10.float().mean().item(),
        "rel_fro": rel_fro,
        "finite": finite,
        "_over": over,
        "_flipped": flipped,
    }


def group_spread(absmax_blocks) -> tuple:
    """Per 256-block group: max/min nonzero true absmax. Returns (spread per group, median, max)."""
    import torch

    groups = _blocks(absmax_blocks, GROUP)
    nonzero = groups > 0
    high = torch.where(nonzero, groups, torch.zeros_like(groups)).amax(dim=1)
    low = torch.where(nonzero, groups, torch.full_like(groups, float("inf"))).amin(dim=1)
    spread = torch.where(torch.isfinite(low) & (low > 0), high / low, torch.ones_like(high))
    return spread, float(spread.median()), float(spread.max())


def spread_buckets(spread, over, flipped) -> list:
    """Q2 table rows: blocks, flipped blocks and over blocks per group-spread bucket."""
    per_block = spread.repeat_interleave(GROUP)[: over.numel()]
    rows = []
    for low, high in zip(SPREAD_EDGES[:-1], SPREAD_EDGES[1:]):
        mask = (per_block >= low) & (per_block < high)
        rows.append({
            "spread_lo": low,
            "spread_hi": None if math.isinf(high) else high,
            "blocks": int(mask.sum()),
            "flipped": int((flipped & mask).sum()),
            "over": int((over & mask).sum()),
        })
    return rows


def fetch_item(ckpt: Checkpoint, item: dict) -> tuple:
    """(weight, scale or None, seconds). Runs one tensor ahead in a worker thread."""
    t0 = time.perf_counter()
    weight = ckpt.fetch(item["key"])
    scale = None
    if ckpt.model["fp8"]:
        scale = ckpt.fetch(item["key"] + "_scale_inv", owner=item["key"])
    return weight, scale, time.perf_counter() - t0


def measure(ckpt: Checkpoint, item: dict, table, fetched: tuple) -> dict:
    import torch

    key = item["key"]
    weight, scale, fetch_s = fetched
    t0 = time.perf_counter()
    if ckpt.model["fp8"]:
        ref = reference_decode(weight, scale, table)
        if not torch.equal(ref, b3_decode(weight, scale)):
            raise RuleVoidError("v2", f"V2: {key} reference decode differs from B3's decoder")
        del scale
    else:
        ref = weight.to(torch.bfloat16)
    del weight
    absmax_blocks = _blocks(ref.float().reshape(-1), NF4_BLOCK).abs().amax(dim=1)
    spread, spread_median, spread_max = group_spread(absmax_blocks)
    row = {
        "key": key,
        "kind": item["kind"],
        "shape": list(ref.shape),
        "numel": ref.numel(),
        "zero_blocks": int((absmax_blocks == 0).sum()),
        "group_spread_median": spread_median,
        "group_spread_max": spread_max,
    }
    for double_quant in (False, True):
        m = nf4_metrics(ref, double_quant)
        tag = "dq_on" if double_quant else "dq_off"
        row[tag] = {k: v for k, v in m.items() if not k.startswith("_")}
        if double_quant:
            row["buckets_dq_on"] = spread_buckets(spread, m["_over"], m["_flipped"])
        del m
    row["fetch_s"] = round(fetch_s, 2)
    row["compute_s"] = round(time.perf_counter() - t0, 2)
    return row


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def box() -> dict:
    import bitsandbytes
    import torch

    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        tree = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
    except OSError:
        tree = "unknown"
    return {
        "when_utc": _now(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "torch_threads": torch.get_num_threads(),
        "tree": tree,
    }


def cmd_select(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"seed": SEED, "models": {}}
    for name, model in MODELS.items():
        ckpt = Checkpoint(name)
        if model["fp8"]:
            selected, drawn = select_fp8(ckpt)
            result["models"][name] = {"rev": model["rev"], "drawn_experts": drawn, "keys": selected}
        else:
            selected = select_candidate(ckpt)
            result["models"][name] = {"rev": model["rev"], "keys": selected}
        print(f"{name}: {len(selected)} tensors")
    (out / "selected-keys.json").write_text(json.dumps(result, indent=1))
    return 0


def _void_reason(exc: BaseException) -> str:
    if isinstance(exc, FetchFailed):
        return "fetch_failed"
    if isinstance(exc, RuleVoidError):
        return exc.reason
    if isinstance(exc, MemoryError):
        return "oom_or_killed"
    return "error"


def _void_interrupted(journal_path: Path, void_path: Path, done: set, voids: dict) -> list:
    """Void every key that was in flight when an earlier run stopped (rule V4).

    In flight: the journal has requests for the key but there is neither a row
    nor a void for it. Returns the newly voided keys.
    """
    events = _read_jsonl(journal_path)
    ended = {e["attempt_id"] for e in events if e["event"] in ("ok", "fail")}
    latest: dict = {}  # key -> id of its last request
    for event in events:
        if event["event"] == "start" and event["key"] is not None:
            latest[event["key"]] = event["attempt_id"]
    # A void written by a run names the key's last request, so a key whose last
    # request is not named by any void (and that has no row) was in flight.
    voided_attempts = {v.get("attempt_id") for rows in voids.values() for v in rows}
    new = []
    for key, attempt_id in latest.items():
        if key in done or attempt_id in voided_attempts:
            continue
        void = {
            "key": key, "reason": "interrupted", "attempt_id": attempt_id,
            "error": "run stopped with this tensor in flight"
            + ("" if attempt_id not in ended else " (requests ended, never measured)"),
        }
        _append_jsonl(void_path, void)
        voids.setdefault(key, []).append(void)
        new.append(key)
    return new


def cmd_run(args) -> int:
    out = Path(args.out)
    selection = json.loads((out / "selected-keys.json").read_text())["models"][args.model]
    done_path, void_path = out / f"{args.model}.jsonl", out / f"{args.model}.void.jsonl"
    journal_path = out / f"{args.model}.attempts.jsonl"
    done = {row["key"] for row in _read_jsonl(done_path)}
    voids: dict = {}
    for void in _read_jsonl(void_path):
        voids.setdefault(void["key"], []).append(void)
    for key in _void_interrupted(journal_path, void_path, done, voids):
        print(f"VOID {key}: interrupted in an earlier run", flush=True)

    journal = Journal(journal_path)
    todo = []
    for item in selection["keys"]:
        key = item["key"]
        if key in done:
            continue
        if key in voids:
            if not args.remeasure_voids:
                continue
            superseded = voids[key][-1].get("attempt_id")
            if superseded is None:
                raise SystemExit(f"{key}: its void has no attempt_id to supersede")
            journal.supersedes[key] = superseded
        todo.append(item)
    if args.limit:
        todo = todo[: args.limit]
    kept = sorted(key for key in voids if key not in done and key not in journal.supersedes)
    if kept:
        print(
            f"{args.model}: {len(kept)} voided tensors are NOT measured again "
            "(pass --remeasure-voids to do that, linked to the void)", flush=True,
        )
    ckpt = Checkpoint(args.model, journal)
    (out / f"{args.model}.box.json").write_text(json.dumps(box(), indent=1))
    table = _e4m3_table()
    print(f"{args.model}: {len(done)} done, {len(todo)} to go", flush=True)

    # One fetch ahead: network I/O overlaps the CPU-bound metrics.
    with ThreadPoolExecutor(max_workers=1) as pool:
        def prefetch(item):
            return pool.submit(fetch_item, ckpt, item)

        pending = prefetch(todo[0]) if todo else None
        for i, item in enumerate(todo):
            future = pending
            pending = prefetch(todo[i + 1]) if i + 1 < len(todo) else None
            key = item["key"]
            try:
                row = measure(ckpt, item, table, future.result())
            except Exception as exc:  # rule V4: kept and reported, never silently retried
                attempt_id = getattr(exc, "attempt_id", None) or journal.last_attempt.get(key)
                _append_jsonl(void_path, {
                    "key": key, "reason": _void_reason(exc), "error": repr(exc),
                    "attempt_id": attempt_id,
                })
                print(f"VOID {key}: {exc}", flush=True)
                continue
            if key in journal.supersedes:
                row["supersedes_attempt"] = journal.supersedes[key]
            with done_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"{args.model}: {len(done) + i + 1}/{len(selection['keys'])}", flush=True)
            del row
    return 0


def _quantiles(values: list) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {"n": len(values), "median": statistics.median(ordered), "p90": p90, "max": ordered[-1]}


def _attempts_entry(name: str, rows: list, voids: list, journal_path: Path) -> dict:
    """Void and request accounting for one model. Refuses an unlinked void + row."""
    void_attempts: dict = {}
    for void in voids:
        void_attempts.setdefault(void["key"], set()).add(void.get("attempt_id"))
    remeasured = set()
    for row in rows:
        if row["key"] not in void_attempts:
            continue
        link = row.get("supersedes_attempt")
        if link is None or link not in void_attempts[row["key"]]:
            raise SystemExit(
                f"{name}: {row['key']} has both a void and a result row that does not "
                "supersede it (rule V4). Refusing to summarize."
            )
        remeasured.add(row["key"])
    by_reason = {reason: 0 for reason in VOID_REASONS}
    for void in voids:
        reason = void.get("reason", "error")
        by_reason[reason if reason in by_reason else "error"] += 1
    entry = {
        "voids": len(voids),
        "voids_by_reason": by_reason,
        "void_keys": sorted(void_attempts),
        "remeasured_keys": len(remeasured),
    }
    if journal_path.exists():
        events = _read_jsonl(journal_path)
        counts = {kind: sum(e["event"] == kind for e in events) for kind in ("start", "ok", "fail")}
        counts["unfinished"] = counts["start"] - counts["ok"] - counts["fail"]
        entry.update(provenance="per-attempt log", attempts=counts)
    else:
        # Written before the journal existed: failed requests may have been
        # retried without a trace, so no count of them is given at all.
        entry.update(provenance="no per-attempt log", attempts=None)
    return entry


def cmd_summarize(args) -> int:
    out = Path(args.out)
    summary = {}
    attempts = {}
    for name in MODELS:
        path = out / f"{name}.jsonl"
        void_path = out / f"{name}.void.jsonl"
        journal_path = out / f"{name}.attempts.jsonl"
        if not (path.exists() or void_path.exists() or journal_path.exists()):
            continue
        rows = _read_jsonl(path)
        voids = _read_jsonl(void_path)
        attempts[name] = _attempts_entry(name, rows, voids, journal_path)
        per_kind = {}
        for kind in sorted({r["kind"] for r in rows}) + ["all"]:
            subset = [r for r in rows if kind in ("all", r["kind"])]
            per_kind[kind] = {
                f"{metric}_{tag}": _quantiles([r[tag][metric] for r in subset])
                for tag in ("dq_off", "dq_on")
                for metric in ("frac_over", "frac_flipped", "frac_absmax_off10", "worst", "rel_fro")
            }
        buckets = {}
        for r in rows:
            for b in r["buckets_dq_on"]:
                slot = buckets.setdefault(b["spread_lo"], {"spread_hi": b["spread_hi"], "blocks": 0,
                                                          "flipped": 0, "over": 0})
                for field in ("blocks", "flipped", "over"):
                    slot[field] += b[field]
        summary[name] = {
            "tensors": len(rows),
            "voids": len(voids),
            "nonfinite_dq_on": sum(not r["dq_on"]["finite"] for r in rows),
            "per_kind": per_kind,
            "spread_buckets_dq_on": buckets,
        }
    # summary.json keeps its schema: re-running this on committed rows reproduces
    # the committed file. Void and request accounting goes to its own file.
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "attempts-summary.json").write_text(json.dumps(attempts, indent=1))
    brief = {
        name: {
            "tensors": v["tensors"],
            "voids": v["voids"],
            "voids_by_reason": attempts[name]["voids_by_reason"],
            "remeasured_keys": attempts[name]["remeasured_keys"],
            "provenance": attempts[name]["provenance"],
            "median_frac_over_dq_on": v["per_kind"]["all"]["frac_over_dq_on"].get("median"),
        }
        for name, v in summary.items()
    }
    print(json.dumps(brief, indent=1))
    for name, entry in attempts.items():
        if entry["attempts"] is None:
            print(f"{name}: provenance: no per-attempt log")
        else:
            print(f"{name}: provenance: per-attempt log, requests {entry['attempts']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--out", required=True)
    p.add_argument(
        "--limit", type=int, default=0, help="smoke test: only the first N pending tensors"
    )
    p.add_argument(
        "--remeasure-voids", action="store_true",
        help="measure voided tensors again; the new row supersedes the void, which is kept",
    )
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"select": cmd_select, "run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
