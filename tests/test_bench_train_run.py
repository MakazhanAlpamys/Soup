"""``bench/train_run.py``: the device synchronize it picks, and the precision it records."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.bench.train_run import _device_synchronize
from tests.test_issue836_bench_train_cli import workdir  # noqa: F401  (fixture)


def _fake_torch(*, cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, synchronize="cuda-sync"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(synchronize="mps-sync"),
    )


class TestDeviceSynchronize:
    @pytest.mark.parametrize("device", ["mps", "mps:0"])
    def test_mps_synchronizes_with_torch_mps(self, device):
        assert _device_synchronize(_fake_torch(mps=True), device) == "mps-sync"

    @pytest.mark.parametrize("device", ["cuda", "cuda:1"])
    def test_cuda_synchronizes_with_torch_cuda(self, device):
        assert _device_synchronize(_fake_torch(cuda=True), device) == "cuda-sync"

    def test_mps_without_a_backend_has_no_synchronize(self):
        assert _device_synchronize(_fake_torch(mps=False), "mps") is None

    def test_cuda_without_a_card_has_no_synchronize(self):
        assert _device_synchronize(_fake_torch(cuda=False), "cuda") is None

    def test_cpu_has_no_synchronize(self):
        assert _device_synchronize(_fake_torch(cuda=True, mps=True), "cpu") is None

    def test_a_cuda_run_does_not_pick_the_mps_call(self):
        assert _device_synchronize(_fake_torch(cuda=True, mps=True), "cuda") == "cuda-sync"


def test_the_report_records_the_training_precision(workdir):  # noqa: F811
    from soup_cli.bench.train_run import run_bench_train
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset

    report = run_bench_train(
        load_config("soup.yaml"), steps=3, warmup=1, device="cpu",
        load_dataset=lambda cfg: load_dataset(cfg.data),
    )
    assert report["provenance"]["training_precision"] == {"bf16": False, "fp16": False}
