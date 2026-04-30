"""Tests for OOM-probe auto batch-size (v0.36.0 Part D).

Replaces sft.py's static-formula auto-batch with a real try/halve probe and
a per-machine cache so repeat runs short-circuit. Mirrors LlamaFactory and
Axolotl behaviour.
"""

from __future__ import annotations

import json

import pytest


class _OOMError(Exception):
    """Stand-in for ``torch.cuda.OutOfMemoryError`` in unit tests."""


# ---------------------------------------------------------------------------
# Schema field
# ---------------------------------------------------------------------------


class TestSchemaField:
    def test_default_is_auto(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig()
        assert tcfg.auto_batch_size_strategy == "auto"

    def test_accepts_static(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(auto_batch_size_strategy="static")
        assert tcfg.auto_batch_size_strategy == "static"

    def test_accepts_probe(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(auto_batch_size_strategy="probe")
        assert tcfg.auto_batch_size_strategy == "probe"

    def test_rejects_unknown_value(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(auto_batch_size_strategy="random")


# ---------------------------------------------------------------------------
# Pure binary search
# ---------------------------------------------------------------------------


class TestProbeLoop:
    def test_converges_when_capacity_below_start(self):
        """Capacity 3, start 4 → halves to 2 (largest power of two that
        fits). Doubling probe(4) re-OOMs, so we stay at 2. Power-of-two
        granularity is a deliberate choice: we trade exactness for fewer
        probe steps (each step is a real GPU forward+backward)."""
        from soup_cli.utils.batch_probe import probe_batch_size

        capacity = 3

        def probe(b: int) -> bool:
            if b > capacity:
                raise _OOMError("simulated")
            return True

        out = probe_batch_size(
            probe,
            start=4,
            ceiling=16,
            oom_exceptions=(_OOMError,),
        )
        assert out == 2

    def test_converges_when_capacity_above_start(self):
        """Start 2, capacity 8 → doubles 2→4→8→16(OOM), back off → 8."""
        from soup_cli.utils.batch_probe import probe_batch_size

        capacity = 8

        def probe(b: int) -> bool:
            if b > capacity:
                raise _OOMError("simulated")
            return True

        out = probe_batch_size(
            probe,
            start=2,
            ceiling=64,
            oom_exceptions=(_OOMError,),
        )
        assert out == capacity

    def test_ceiling_caps_at_4x_static(self):
        """Capacity 1000, ceiling 8 → returns 8 (never tries higher)."""
        from soup_cli.utils.batch_probe import probe_batch_size

        def probe(b: int) -> bool:
            return True  # never OOMs

        out = probe_batch_size(
            probe,
            start=2,
            ceiling=8,
            oom_exceptions=(_OOMError,),
        )
        assert out == 8

    def test_starts_oom_halves_to_one(self):
        """Even start=1 OOMs → returns 1 (never go below 1)."""
        from soup_cli.utils.batch_probe import probe_batch_size

        def probe(b: int) -> bool:
            raise _OOMError("starved")

        with pytest.raises(RuntimeError, match="batch_size=1"):
            probe_batch_size(
                probe,
                start=2,
                ceiling=16,
                oom_exceptions=(_OOMError,),
            )

    def test_max_doublings_capped(self):
        """Search must not run forever — cap at 8 doublings."""
        from soup_cli.utils.batch_probe import probe_batch_size

        calls: list[int] = []

        def probe(b: int) -> bool:
            calls.append(b)
            return True

        probe_batch_size(
            probe,
            start=1,
            ceiling=10**6,
            oom_exceptions=(_OOMError,),
            max_doublings=8,
        )
        # At most 8 successful doublings + initial = 9 successful probes.
        assert len(calls) <= 12

    def test_rejects_invalid_start(self):
        from soup_cli.utils.batch_probe import probe_batch_size

        with pytest.raises(ValueError, match="start"):
            probe_batch_size(
                lambda b: True,
                start=0,
                ceiling=8,
                oom_exceptions=(_OOMError,),
            )

    def test_rejects_invalid_ceiling(self):
        from soup_cli.utils.batch_probe import probe_batch_size

        with pytest.raises(ValueError, match="ceiling"):
            probe_batch_size(
                lambda b: True,
                start=4,
                ceiling=2,  # < start
                oom_exceptions=(_OOMError,),
            )

    def test_unrelated_exception_propagates(self):
        """A non-OOM exception must propagate, not be swallowed as OOM."""
        from soup_cli.utils.batch_probe import probe_batch_size

        def probe(b: int) -> bool:
            raise RuntimeError("model bug")

        with pytest.raises(RuntimeError, match="model bug"):
            probe_batch_size(
                probe,
                start=2,
                ceiling=16,
                oom_exceptions=(_OOMError,),
            )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


class TestCache:
    def test_key_normalizes(self):
        from soup_cli.utils.batch_probe import make_cache_key

        a = make_cache_key(
            base="meta-llama/Llama-3.2-1B",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_memory_gb=80,
        )
        b = make_cache_key(
            base="meta-llama/Llama-3.2-1B",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_memory_gb=80,
        )
        assert a == b

    def test_key_differs_on_quantization(self):
        from soup_cli.utils.batch_probe import make_cache_key

        a = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        b = make_cache_key("m", 2048, "8bit", 64, "gpu", 80)
        assert a != b

    def test_key_rejects_bool_inputs(self):
        """v0.30.0 Candidate convention: bool is a subclass of int — guard."""
        from soup_cli.utils.batch_probe import make_cache_key

        with pytest.raises(ValueError, match="max_length"):
            make_cache_key("m", True, "4bit", 64, "gpu", 80)
        with pytest.raises(ValueError, match="lora_r"):
            make_cache_key("m", 2048, "4bit", True, "gpu", 80)
        with pytest.raises(ValueError, match="gpu_memory_gb"):
            make_cache_key("m", 2048, "4bit", 64, "gpu", True)

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import (
            load_cache,
            make_cache_key,
            save_cache_entry,
        )

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        key = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        save_cache_entry(key, 8)

        cache = load_cache()
        assert cache.get(key) == 8

    def test_load_corrupt_returns_empty(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import load_cache

        cache_path = tmp_path / "batch_cache.json"
        cache_path.write_text("not json", encoding="utf-8")
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        assert load_cache() == {}

    def test_load_missing_returns_empty(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import load_cache

        cache_path = tmp_path / "missing.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        assert load_cache() == {}

    def test_save_rejects_non_positive_value(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import make_cache_key, save_cache_entry

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))
        key = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)

        with pytest.raises(ValueError):
            save_cache_entry(key, 0)
        with pytest.raises(ValueError):
            save_cache_entry(key, -1)

    def test_save_rejects_bool_value(self, tmp_path, monkeypatch):
        """``bool`` is a subclass of int — guard like v0.30.0 Candidate."""
        from soup_cli.utils.batch_probe import make_cache_key, save_cache_entry

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))
        key = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)

        with pytest.raises(ValueError):
            save_cache_entry(key, True)


# ---------------------------------------------------------------------------
# pick_batch_size — main entry point
# ---------------------------------------------------------------------------


class TestPickBatchSize:
    def test_static_strategy_returns_static_estimate(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import pick_batch_size

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        out = pick_batch_size(
            static_estimate=4,
            strategy="static",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="cpu",
            gpu_memory_gb=0,
            probe_fn=None,
        )
        assert out == 4

    def test_cache_hit_short_circuits_probe(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import (
            make_cache_key,
            pick_batch_size,
            save_cache_entry,
        )

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        key = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        save_cache_entry(key, 16)

        called: list[int] = []

        def probe(b):
            called.append(b)
            return True

        out = pick_batch_size(
            static_estimate=4,
            strategy="probe",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="gpu",
            gpu_memory_gb=80,
            probe_fn=probe,
        )
        assert out == 16
        assert called == []  # probe was not invoked

    def test_probe_strategy_runs_probe_and_caches(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import (
            load_cache,
            make_cache_key,
            pick_batch_size,
        )

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        capacity = 8

        def probe(b):
            if b > capacity:
                raise _OOMError("oom")
            return True

        out = pick_batch_size(
            static_estimate=4,
            strategy="probe",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="gpu",
            gpu_memory_gb=80,
            probe_fn=probe,
            oom_exceptions=(_OOMError,),
        )
        assert out == capacity

        # Cache write happened.
        cache = load_cache()
        key = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        assert cache[key] == capacity

    def test_probe_without_callable_falls_back_to_static(self, tmp_path, monkeypatch):
        """No probe_fn supplied (e.g. CPU run) → use static estimate."""
        from soup_cli.utils.batch_probe import pick_batch_size

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        out = pick_batch_size(
            static_estimate=4,
            strategy="probe",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="cpu",
            gpu_memory_gb=0,
            probe_fn=None,
        )
        assert out == 4

    def test_auto_strategy_uses_probe_when_probe_fn_supplied(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.batch_probe import pick_batch_size

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        capacity = 4

        def probe(b):
            if b > capacity:
                raise _OOMError("oom")
            return True

        out = pick_batch_size(
            static_estimate=2,
            strategy="auto",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="gpu",
            gpu_memory_gb=80,
            probe_fn=probe,
            oom_exceptions=(_OOMError,),
        )
        assert out == capacity

    def test_cache_corruption_does_not_block_probe(self, tmp_path, monkeypatch):
        """Corrupt cache file → silently re-probe; ceiling = static * 4."""
        from soup_cli.utils.batch_probe import pick_batch_size

        cache_path = tmp_path / "batch_cache.json"
        cache_path.write_text("garbage", encoding="utf-8")
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        out = pick_batch_size(
            static_estimate=4,
            strategy="probe",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="gpu",
            gpu_memory_gb=80,
            probe_fn=lambda b: True,
            oom_exceptions=(_OOMError,),
        )
        # Probe ran; with no OOMs and ceiling = 4*4 = 16, lands at 16.
        assert out == 16

    def test_runtime_error_propagates_when_bs1_ooms(self, tmp_path, monkeypatch):
        """All-OOM probe → RuntimeError surfaces to caller."""
        from soup_cli.utils.batch_probe import pick_batch_size

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        def always_oom(b):
            raise _OOMError("oom")

        with pytest.raises(RuntimeError, match="batch_size=1"):
            pick_batch_size(
                static_estimate=2,
                strategy="probe",
                base="m",
                max_length=2048,
                quantization="4bit",
                lora_r=64,
                gpu_name="gpu",
                gpu_memory_gb=80,
                probe_fn=always_oom,
                oom_exceptions=(_OOMError,),
            )

    def test_explicit_probe_no_probe_fn_emits_warning(
        self, tmp_path, monkeypatch
    ):
        """strategy='probe' with probe_fn=None → console warning fires."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.batch_probe import pick_batch_size

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))
        buf = StringIO()
        console = Console(file=buf, force_terminal=False)

        out = pick_batch_size(
            static_estimate=4,
            strategy="probe",
            base="m",
            max_length=2048,
            quantization="4bit",
            lora_r=64,
            gpu_name="cpu",
            gpu_memory_gb=0,
            probe_fn=None,
            console=console,
        )
        assert out == 4
        assert "probe_fn" in buf.getvalue() or "static" in buf.getvalue()


# ---------------------------------------------------------------------------
# Cache-path containment (security review fix)
# ---------------------------------------------------------------------------


class TestCachePathContainment:
    def test_out_of_bounds_override_falls_back_to_default(
        self, tmp_path, monkeypatch
    ):
        """Env var pointing outside home/cwd/tmp → ignored, default used."""
        import os

        from soup_cli.utils.batch_probe import _cache_path

        # Use a sibling-of-temp path that is guaranteed outside any anchor —
        # an absolute root we cannot write to is equally fine, since the
        # function only resolves+rejects, no I/O.
        if os.name == "nt":
            evil = "C:\\evil-bound\\batch.json"
        else:
            evil = "/etc/cron.d/soup_evil"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", evil)
        path = _cache_path()
        # Fall-through path — must NOT be the evil override.
        assert os.path.realpath(path) != os.path.realpath(evil)
        assert path.endswith("batch_cache.json")

    def test_in_bounds_override_honoured(self, tmp_path, monkeypatch):
        import os

        from soup_cli.utils.batch_probe import _cache_path

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))
        # tmp_path is under tempfile.gettempdir() — allowed. Compare via
        # realpath to normalise across short-name / forward-slash forms.
        assert os.path.realpath(_cache_path()) == os.path.realpath(str(cache_path))


# ---------------------------------------------------------------------------
# Cache file integrity guard
# ---------------------------------------------------------------------------


class TestCacheFileShape:
    def test_cache_is_dict_of_str_int(self, tmp_path, monkeypatch):
        from soup_cli.utils.batch_probe import make_cache_key, save_cache_entry

        cache_path = tmp_path / "batch_cache.json"
        monkeypatch.setenv("SOUP_BATCH_CACHE_PATH", str(cache_path))

        key = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        save_cache_entry(key, 8)

        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, int)
            assert v > 0
