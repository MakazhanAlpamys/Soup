"""Run only the streaming CUDA controls in an owned local context and preserve its identity."""

import argparse
import json
from pathlib import Path


def pytest_arguments(repo: Path, test_paths: list[str], receipt: Path) -> list[str]:
    """Keep CUDA fixtures on the repository's admitted data filesystem and within CWD."""
    return [
        "-o",
        "addopts=",
        "-m",
        "gpu",
        "-q",
        f"--junitxml={receipt}",
        f"--basetemp={repo / '.streaming_cuda_test_tmp'}",
        *test_paths,
    ]


def main() -> int:
    from benchmarks.harness.arm_validity import confirm_gpu_owner, gpu_compute_pids

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("test_paths", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = gpu_compute_pids()
    if before != []:
        raise RuntimeError(f"GPU not observably idle: {before!r}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    owner = confirm_gpu_owner(before, context_active=True, single_device=True)
    del probe
    (args.output / "gpu_owner.json").write_text(json.dumps(owner, indent=2))
    if owner["method"] != "exclusive-local-CUDA-initialization":
        raise RuntimeError("GPU ownership unresolved")
    import pytest

    return pytest.main(pytest_arguments(Path.cwd(), args.test_paths, args.output / "tests.xml"))


if __name__ == "__main__":
    raise SystemExit(main())
