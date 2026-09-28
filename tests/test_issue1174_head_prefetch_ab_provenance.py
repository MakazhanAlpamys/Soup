"""#1174: the head-prefetch A/B driver's JSON must say which card, driver and tree it measured.

Two things I broke, both silent because nothing exercised the driver's JSON before:
the payload's own ``"driver"`` key (the script path) was overwriting
``gpu_facts()``'s ``"driver"`` key (the NVIDIA version) when I spread the dict at
the top level, and ``_source_sha()`` resolved the commit from this file's own
location, not from the ``soup_cli`` tree it actually measured.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _REPO_ROOT / "benchmarks" / "harness"
_HARNESS = _HARNESS_DIR / "head_prefetch_ab.py"


def _load_driver():
    sys.path.insert(0, str(_HARNESS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("head_prefetch_ab", _HARNESS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_HARNESS_DIR))


_driver = _load_driver()


@pytest.mark.parametrize("status", ["", " M src/soup_cli/utils/layer_stream_runtime.py\n"])
def test_git_sha_names_the_imported_tree_and_marks_it_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    """The driver and the ``soup_cli`` it measures can come from different checkouts,
    so I resolve the commit from ``soup_cli.__file__``, not from this driver's file."""
    import soup_cli

    package_file = soup_cli.__file__
    assert package_file is not None
    package_dir = Path(package_file).resolve().parent
    expected_sha = "a" * 40
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        calls.append((command, cwd))
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=f"{tmp_path}\n")
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=f"{expected_sha}\n")
        return SimpleNamespace(stdout=status)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_driver.subprocess, "run", fake_run)

    assert _driver._source_sha() == expected_sha + ("-dirty" if status else "")
    assert calls == [
        (["git", "rev-parse", "--show-toplevel"], package_dir),
        (["git", "rev-parse", "HEAD"], tmp_path),
        (["git", "status", "--porcelain", "--untracked-files=normal", "--", "src"], tmp_path),
    ]


def test_source_sha_is_unknown_without_a_git_toplevel(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        return SimpleNamespace(stdout="\n")

    monkeypatch.setattr(_driver.subprocess, "run", fake_run)
    assert _driver._source_sha() == "unknown"


def test_source_sha_is_unknown_when_git_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        raise FileNotFoundError("no git")

    monkeypatch.setattr(_driver.subprocess, "run", fake_run)
    assert _driver._source_sha() == "unknown"


def test_source_sha_is_unknown_when_git_exits_non_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tree with no ``.git`` (a ``git archive`` extract) makes ``git rev-parse
    --show-toplevel`` exit 128; with ``check=True`` that is a ``CalledProcessError``."""

    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        raise _driver.subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(_driver.subprocess, "run", fake_run)
    assert _driver._source_sha() == "unknown"


def test_source_sha_does_not_record_a_head_that_is_not_a_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=f"{tmp_path}\n")
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="not-a-commit\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(_driver.subprocess, "run", fake_run)
    assert _driver._source_sha() == "unknown"


def test_the_gpu_facts_stay_nested_so_the_nvidia_driver_survives() -> None:
    """``gpu_facts`` returns its own ``"driver"`` key (the NVIDIA version); spreading
    it into a payload that also sets ``"driver"`` to the script path overwrites it."""
    text = _HARNESS.read_text(encoding="utf-8")
    assert '"gpu": stream_probe.gpu_facts(device)' in text
    assert "**stream_probe.gpu_facts(" not in text


class _FakePrefetcher:
    head_prefetch_layer = None


class _FakeLoss:
    def backward(self) -> None:
        pass


class _FakeModel:
    def __call__(self, *, input_ids: object = None, labels: object = None) -> Any:
        return SimpleNamespace(loss=_FakeLoss())

    def parameters(self) -> list[Any]:
        return [SimpleNamespace(requires_grad=True)]


class _FakeOptimizer:
    def __init__(self, *a: object, **k: object) -> None:
        pass

    def step(self) -> None:
        pass

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        pass


class _FakeInstruments:
    def __init__(self, *a: object, **k: object) -> None:
        self.events_on = False
        self._stall_pairs: list[Any] = []

    def reset(self) -> None:
        pass


class _FakeRuntime:
    n_layers = 4

    def __init__(self) -> None:
        self.prefetcher = _FakePrefetcher()
        self.large_pool = SimpleNamespace(nbytes=1024)
        self.source = object()

    def stats(self) -> dict[str, Any]:
        return {"tier": "ram", "pinned": True, "read_ahead": 2}

    def close(self) -> None:
        pass


def test_main_writes_git_sha_and_shards_into_the_real_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The payload is built inside ``main()``, behind the CUDA check, so nothing
    exercised it before: a hard-coded ``git_sha`` or a ``shards`` dropped back to
    ``None`` would both go unnoticed. I fake only the GPU-bound pieces — the model,
    the runtime, ``stream_probe``, ``layer0_wait`` and the bitsandbytes optimizer —
    and run the driver's real ``main()`` over them."""
    import bitsandbytes
    import torch

    out_path = tmp_path / "out.json"
    cli = SimpleNamespace(
        weights="fake-weights",
        shards="fake-shards-dir",
        tier="ram",
        quant="nf4",
        seq=8,
        batch=1,
        read_ahead=2,
        buffers=2,
        no_pin=False,
        lora_r=8,
        lora_targets="q_proj,k_proj,v_proj,o_proj",
        seed=3,
        input_seed=17,
        arms="last,0",
        step_shape="sft",
        warmup=0,
        blocks=1,
        transition=0,
        measure=1,
        out=str(out_path),
        label="test",
    )
    monkeypatch.setattr(_driver, "parse_args", lambda: cli)

    runtime = _FakeRuntime()
    model = _FakeModel()
    config = SimpleNamespace(vocab_size=32000)
    build_args_seen: list[Any] = []

    fake_gpu_facts = {
        "gpu": "simulated CUDA",
        "torch": "0.0.0",
        "vram_total_gb": 1.0,
        "soup_cli_file": "/fake/soup_cli/__init__.py",
        "driver": "616.92",
    }

    fake_stream_probe = SimpleNamespace(
        cuda_available=lambda: True,
        build=lambda args, device, dtype: (
            build_args_seen.append(args) or model,
            runtime,
            config,
            None,
            None,
            "fake-shard-dir",
            0.1,
            0.2,
        ),
        gpu_facts=lambda device: dict(fake_gpu_facts),
        Instruments=_FakeInstruments,
    )
    fake_layer0_wait = SimpleNamespace(_label_wrappers=lambda inst, copy, stall: None)
    monkeypatch.setattr(_driver, "_harness", lambda: (fake_stream_probe, fake_layer0_wait))
    monkeypatch.setattr(_driver, "_source_sha", lambda: "b" * 40)
    monkeypatch.setattr(bitsandbytes.optim, "PagedAdamW8bit", _FakeOptimizer)
    monkeypatch.setattr(
        torch, "Generator", lambda **k: SimpleNamespace(manual_seed=lambda *_: object())
    )
    monkeypatch.setattr(torch, "randint", lambda *a, **k: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    assert _driver.main() == 0
    assert build_args_seen[0].shards == "fake-shards-dir"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["git_sha"] == "b" * 40
    assert payload["gpu"] == fake_gpu_facts
    assert payload["gpu"]["driver"] == "616.92"
    assert payload["args"]["shards"] == "fake-shards-dir"
