"""Layer streaming must refuse a multi-GPU box instead of dying inside torch.

Found on 8x H100 (benchmarks/gate-h100-validation.md, FINDING 1): eight of the
nine streaming-suite failures there were one error,

    RuntimeError: module must have its parameters and buffers on device cuda:0
    (device_ids[0]) but found one of them on device: meta

`TrainingArguments` sets ``_n_gpu = torch.cuda.device_count()`` for a
non-distributed run and ``Trainer._wrap_model`` then applies
``nn.DataParallel`` whenever ``n_gpu > 1``. DataParallel requires every
parameter on ``device_ids[0]``; layer streaming keeps the decoder on ``meta``.
Incompatible by construction — and the message names nothing the user set.

The dev box that built this feature had one GPU, so the branch was never taken.

These tests run EVERYWHERE: the device count is monkeypatched, so a CPU-only CI
runner exercises the same code path as a datacenter node. The one test that
genuinely needs several cards is skipped by hardware, not left to fail.
"""

import os

import pytest

from soup_cli.trainer.stream_setup import _distributed_launch, refuse_if_data_parallel


def _fake_cuda(monkeypatch, count, available=True):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


class TestRefusesWhenDataParallelWouldEngage:
    def test_refuses_with_several_visible_gpus(self, monkeypatch):
        _fake_cuda(monkeypatch, 8)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        with pytest.raises(ValueError) as excinfo:
            refuse_if_data_parallel("cuda")
        assert "8 CUDA devices are visible" in str(excinfo.value)

    def test_message_names_the_flag_the_cause_and_the_fix(self, monkeypatch):
        """A guard that refuses without naming `stream_layers` is no better than
        torch's own error, which is the entire complaint being fixed."""
        _fake_cuda(monkeypatch, 2)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        with pytest.raises(ValueError) as excinfo:
            refuse_if_data_parallel("cuda:0")
        message = str(excinfo.value)
        assert "stream_layers" in message
        assert "DataParallel" in message
        assert "CUDA_VISIBLE_DEVICES=0" in message


class TestDoesNotRefuseWhenDataParallelWouldNot:
    """Controls. Without these the guard above is satisfied by one that always
    raises, which would break every single-GPU run in existence."""

    def test_single_gpu_is_allowed(self, monkeypatch):
        _fake_cuda(monkeypatch, 1)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        refuse_if_data_parallel("cuda")

    def test_cpu_is_allowed_even_with_many_gpus_visible(self, monkeypatch):
        """`--device cpu` never reaches DataParallel; refusing it would block
        the CPU path the streaming tests themselves run on."""
        _fake_cuda(monkeypatch, 8)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        refuse_if_data_parallel("cpu")

    def test_no_cuda_available_is_allowed(self, monkeypatch):
        _fake_cuda(monkeypatch, 8, available=False)
        refuse_if_data_parallel("cuda")

    def test_distributed_launch_is_allowed(self, monkeypatch):
        """torchrun / accelerate give each process one GPU, HF reports n_gpu=1
        and never wraps in DataParallel. Refusing there would ban a
        configuration that works."""
        _fake_cuda(monkeypatch, 8)
        monkeypatch.setenv("WORLD_SIZE", "8")
        refuse_if_data_parallel("cuda")


class TestDistributedDetection:
    @pytest.mark.parametrize(
        "value,expected",
        [("1", False), ("2", True), ("8", True), ("", False), ("0", False)],
    )
    def test_world_size_parsing(self, monkeypatch, value, expected):
        monkeypatch.setenv("WORLD_SIZE", value)
        assert _distributed_launch() is expected

    def test_unset_world_size_is_not_distributed(self, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert _distributed_launch() is False

    def test_malformed_world_size_is_not_distributed(self, monkeypatch):
        """A junk value must not silently disable the guard -- refusing with a
        clear message beats proceeding into torch's `meta` error."""
        monkeypatch.setenv("WORLD_SIZE", "not-a-number")
        assert _distributed_launch() is False


@pytest.mark.skipif(
    not os.environ.get("SOUP_TEST_MULTI_GPU"),
    reason="needs >1 real CUDA device; set SOUP_TEST_MULTI_GPU=1 to run",
)
class TestOnRealMultiGpuHardware:
    """Verified on 8x H100. Skipped by hardware rather than left failing, per the
    rule that the suite must be green on both classes of machine."""

    def test_real_device_count_triggers_the_guard(self):
        import torch

        if torch.cuda.device_count() < 2:
            pytest.skip("fewer than 2 CUDA devices visible")
        with pytest.raises(ValueError) as excinfo:
            refuse_if_data_parallel("cuda")
        assert "stream_layers" in str(excinfo.value)
