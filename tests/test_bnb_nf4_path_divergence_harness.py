"""benchmarks/harness/bnb_nf4_path_divergence.py (#776), the parts that need no GPU.

The measurement itself needs a CUDA card (the numbers are in benchmarks/nf4-on-blackwell.md).
What can be pinned here: that the harness reports WHICH bitsandbytes it measured (a
``--bnb-path`` that silently fell back to the installed copy would publish the wrong build's
numbers), that it asks bitsandbytes' own dispatch which kernel ran and says "unknown" instead
of guessing, and that a machine without CUDA is a skip, not a failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

_HARNESS = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "bnb_nf4_path_divergence.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("bnb_nf4_path_divergence", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _ops(use_custom, max_rows=1536):
    return SimpleNamespace(
        _gemm_4bit_use_custom_fn=use_custom, _gemm_4bit_custom_max_m=max_rows
    )


class TestSelectedCustom:
    """The harness records bitsandbytes' own kernel choice for every row."""

    def test_asks_the_private_dispatch_with_the_row_shape(self):
        seen = []

        def use_custom(device_index, dtype, rows, out_features, in_features):
            seen.append((device_index, dtype, rows, out_features, in_features))
            return True

        assert harness._selected_custom(_ops(use_custom), 0, "dt", 12, 14336, 4096, 64) is True
        assert seen == [(0, "dt", 12, 14336, 4096)]

    def test_a_false_answer_is_reported_as_false(self):
        result = harness._selected_custom(_ops(lambda *args: False), 0, "dt", 12, 4096, 4096, 64)
        assert result is False

    def test_above_the_row_cap_bitsandbytes_never_asks_so_neither_do_we(self):
        def must_not_run(*args):
            raise AssertionError("bitsandbytes does not ask past its row cap")

        assert harness._selected_custom(_ops(must_not_run), 0, "dt", 1537, 4096, 4096, 64) is False

    def test_an_inner_dimension_that_is_not_a_blocksize_multiple_is_the_fallback(self):
        def must_not_run(*args):
            raise AssertionError("misaligned K never reaches the custom kernel")

        assert harness._selected_custom(_ops(must_not_run), 0, "dt", 4, 4096, 4100, 64) is False

    def test_a_build_without_the_private_symbol_is_unknown_not_guessed(self):
        assert harness._selected_custom(SimpleNamespace(), 0, "dt", 4, 4096, 4096, 64) is None
        assert harness._selected_custom(None, 0, "dt", 4, 4096, 4096, 64) is None

    def test_a_private_symbol_that_changed_signature_is_unknown_not_a_crash(self):
        def changed(device_index):
            return True

        assert harness._selected_custom(_ops(changed), 0, "dt", 4, 4096, 4096, 64) is None


class TestStats:
    def test_identical_tensors_report_zero(self):
        import torch

        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        assert harness._stats(tensor, tensor.clone()) == {
            "max_abs": 0.0,
            "n_diff": 0,
            "n_total": 6,
        }

    def test_counts_the_elements_that_differ_and_the_largest_gap(self):
        import torch

        left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = torch.tensor([[1.0, 2.5, 3.0], [4.0, 5.0, 3.0]])
        assert harness._stats(left, right) == {"max_abs": 3.0, "n_diff": 2, "n_total": 6}

    def test_half_precision_inputs_are_compared_in_float32(self):
        import torch

        left = torch.tensor([60000.0], dtype=torch.float16)
        right = torch.tensor([-60000.0], dtype=torch.float16)  # a float16 difference would overflow
        assert harness._stats(left, right)["max_abs"] == 120000.0


class TestDistVersion:
    """The recorded version comes from the build that was put on ``--bnb-path``."""

    def _fake_target(self, tmp_path, version):
        dist = tmp_path / f"bitsandbytes-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: bitsandbytes\nVersion: {version}\n", encoding="utf-8"
        )
        return str(tmp_path)

    def test_reads_the_version_from_the_given_directory(self, tmp_path):
        target = self._fake_target(tmp_path, "9.8.7.dev0")
        assert harness._dist_version(target) == "9.8.7.dev0"

    def test_without_a_path_it_reports_the_installed_distribution(self):
        import importlib.metadata as metadata

        try:
            installed = metadata.version("bitsandbytes")
        except metadata.PackageNotFoundError:
            assert harness._dist_version(None) == "unknown"
        else:
            assert harness._dist_version(None) == installed


class TestMainWithoutCuda:
    def test_no_cuda_is_a_skip_that_exits_zero(self, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(sys, "argv", ["bnb_nf4_path_divergence.py"])

        assert harness.main() == 0
        assert "SKIP: CUDA is required" in capsys.readouterr().out

    def test_bnb_path_goes_first_on_sys_path_as_an_absolute_path(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.chdir(tmp_path.parent)  # a relative path, without a cross-drive relpath
        relative = tmp_path.name
        monkeypatch.setattr(
            sys, "argv", ["bnb_nf4_path_divergence.py", "--bnb-path", relative]
        )

        assert harness.main() == 0
        assert sys.path[0] == os.path.abspath(relative)
        assert os.path.isabs(sys.path[0])


class TestArguments:
    def test_the_default_sweep_straddles_the_row_cap_and_the_small_row_range(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["bnb_nf4_path_divergence.py"])
        args = harness._parse_args()

        rows = [int(item) for item in args.rows.split(",")]
        assert 1 in rows and 12 in rows  # the decode / tiny-batch range of #776
        assert 1536 in rows and 1537 in rows  # either side of bitsandbytes' own row cap
        shapes = args.shapes.split(",")
        assert shapes == ["4096x4096", "4096x14336", "14336x4096"]
        assert args.dtypes.split(",") == ["float16", "bfloat16"]
        assert args.bnb_path is None and args.json is None
