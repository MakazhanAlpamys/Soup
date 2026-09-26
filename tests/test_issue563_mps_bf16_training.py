"""Regression coverage for #563: BF16-capable MPS text trainers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def fake_mps(monkeypatch):
    import torch

    probes = []

    class _MPS:
        available = True

        def is_available(self):
            return self.available

    backend = _MPS()
    monkeypatch.setattr(torch.backends, "mps", backend)

    def apply(*, available=True, accepts_bf16=True):
        backend.available = available
        probes.clear()

        def empty(size, *, dtype, device):
            probes.append((size, dtype, device))
            if not accepts_bf16:
                raise TypeError("MPS runtime rejected bfloat16")
            return object()

        monkeypatch.setattr(torch, "empty", empty)
        return probes

    return apply


class TestMPSBF16Capability:
    def test_available_runtime_that_accepts_bf16_is_supported(self, fake_mps):
        import torch

        from soup_cli.utils.gpu import mps_supports_bf16

        probes = fake_mps(available=True, accepts_bf16=True)

        assert mps_supports_bf16() is True
        assert probes == [(1, torch.bfloat16, "mps")]

    def test_unavailable_runtime_does_not_probe_or_claim_bf16(self, fake_mps):
        from soup_cli.utils.gpu import mps_supports_bf16

        probes = fake_mps(available=False, accepts_bf16=True)

        assert mps_supports_bf16() is False
        assert probes == []

    def test_runtime_rejection_falls_back_to_fp32(self, fake_mps):
        from soup_cli.utils.gpu import mps_supports_bf16

        fake_mps(available=True, accepts_bf16=False)

        assert mps_supports_bf16() is False


class TestPrecisionPolicy:
    def test_verified_text_trainer_can_opt_into_mps_bf16(self, fake_mps):
        from soup_cli.utils.gpu import bf16_fp16_flags

        fake_mps(available=True, accepts_bf16=True)

        assert bf16_fp16_flags("mps", allow_mps_bf16=True) == (True, False)

    def test_mps_stays_fp32_without_per_trainer_opt_in(self, fake_mps):
        from soup_cli.utils.gpu import bf16_fp16_flags

        fake_mps(available=True, accepts_bf16=True)

        assert bf16_fp16_flags("mps") == (False, False)

    def test_cpu_stays_fp32_even_when_mps_is_available(self, fake_mps):
        from soup_cli.utils.gpu import bf16_fp16_flags

        fake_mps(available=True, accepts_bf16=True)

        assert bf16_fp16_flags("cpu", allow_mps_bf16=True) == (False, False)

    @pytest.mark.parametrize("auto_mixed_precision", [False, True])
    def test_sft_requests_the_verified_mps_policy(
        self, monkeypatch, auto_mixed_precision
    ):
        from soup_cli.trainer import sft

        calls = []

        def resolve(device, *, allow_mps_bf16=False):
            calls.append((device, allow_mps_bf16))
            return (True, False)

        monkeypatch.setattr(sft, "bf16_fp16_flags", resolve)
        wrapper = object.__new__(sft.SFTTrainerWrapper)
        wrapper.device = "mps"

        assert wrapper._resolve_mixed_precision(
            SimpleNamespace(auto_mixed_precision=auto_mixed_precision), "model"
        ) == (True, False)
        assert calls == [("mps", True)]

    def test_only_hardware_validated_text_trainers_opt_in(self):
        import soup_cli.trainer as trainer_package

        root = Path(trainer_package.__file__).parent
        expected = {"sft.py", "dpo.py", "grpo.py", "reward_model.py", "prm.py"}
        opted_in = {
            path.name
            for path in root.glob("*.py")
            if "allow_mps_bf16=True" in path.read_text(encoding="utf-8")
        }

        assert opted_in == expected


def test_layer_streaming_uses_the_same_mps_capability(monkeypatch):
    from soup_cli.utils import gpu
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    monkeypatch.setattr(gpu, "mps_supports_bf16", lambda: True)
    assert resolve_stream_dtype("mps") == "bfloat16"

    monkeypatch.setattr(gpu, "mps_supports_bf16", lambda: False)
    assert resolve_stream_dtype("mps") == "float32"
