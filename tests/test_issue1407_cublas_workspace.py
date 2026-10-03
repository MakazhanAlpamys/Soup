"""#1407 -- the pre-flight charged no cuBLAS workspace.

A streamed training step holds one cuBLAS workspace per (handle, stream) pair.
The forward runs on the caller's thread; the backward runs on autograd's CUDA
thread, which gets its own handle from PyTorch's ``thread_local`` pool
(``aten/src/ATen/cuda/CublasHandlePool.cpp``). Both are allocated through the
caching allocator, so ``torch.cuda.max_memory_allocated()`` -- the quantity
``estimate_stream_peak_vram`` documents itself as predicting -- counts them.
The formula named no term for them.

Measured on an RTX 5070 Laptop (cc 12.0, 8151 MiB, driver 616.92, torch
2.14.0+cu130), the 2-layer streamed fixture of ``test_v07204`` at 2 rows x seq 64:
the pre-flight predicted 13,934,816 B against a real DPO peak of 67,734,016 B
(4.86x low). With ``CUBLAS_WORKSPACE_CONFIG=:4096:1`` the same step peaked at
9,013,760 B, so 58,720,256 B of the miss -- exactly 2 x (32 - 4) MiB -- was the
workspace and nothing else.

The term is the FULL size on the device, not a delta over the legacy one. A delta
was tried and rejected: it leaves the GATE 2 grid untouched only because the grid
card's 16.3 MB of workspace is subtracted from a baseline that already contained
it, which makes the charge read 0 for ``CUBLAS_WORKSPACE_CONFIG=:4096:1`` while
the allocator really does move 58,720,256 B. Criterion 3 of #1407 asks the charge
to MATCH the allocator, and only the full size does.
``TestTheGridIsUnmoved`` records what the full size does to that grid.
"""

import pytest

from soup_cli.utils.layer_stream import (
    CUBLAS_WORKSPACE_BYTES_BY_MAJOR,
    CUBLAS_WORKSPACE_BYTES_LEGACY,
    CUBLAS_WORKSPACE_BYTES_PRE_HOPPER,
    CUBLAS_WORKSPACE_CONFIG_ENV,
    CUBLAS_WORKSPACE_HANDLES_PER_STREAMED_STEP,
    STREAM_FIXED_SLACK_BYTES,
    cublas_workspace_size_bytes,
    estimate_cublas_workspace_bytes,
    measure_cublas_workspace_bytes,
    parse_cublas_workspace_config,
)

#: Compute capability of the cards the workspace rule distinguishes. 8.6 is the
#: RTX 3050 the GATE 2 grid was measured on; 12.0 is the RTX 5070 the
#: under-prediction was reported on. #1407.
CC_LEGACY = 8
CC_HOPPER = 9
CC_BLACKWELL = 12


class TestParseCublasWorkspaceConfig:
    """PyTorch parses ``CUBLAS_WORKSPACE_CONFIG`` with
    ``std::regex exp(":([0-9]+):([0-9]+)")`` and sums ``SIZE * 1024 * COUNT``
    over every match (``parseChosenWorkspaceSize``, aten/src/ATen/cuda/
    CublasHandlePool.cpp). These pin that arithmetic exactly -- an estimate that
    parsed the variable differently would charge the wrong bytes while looking
    correct."""

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (":4096:8", 4096 * 8 * 1024),
            (":4096:2:16:8", (4096 * 2 + 16 * 8) * 1024),
            (":4096:1", 4096 * 1024),
            (":0:0", 0),
        ],
    )
    def test_sums_size_times_count_in_kib(self, config, expected):
        assert parse_cublas_workspace_config(config) == expected

    def test_the_documented_default_parses_to_the_device_default(self):
        """``:4096:2:16:8`` is the default PyTorch documents, and it must land on
        the same number the device rule produces -- otherwise an operator who sets
        it to the documented value silently changes the charge."""
        assert (
            parse_cublas_workspace_config(":4096:2:16:8")
            == CUBLAS_WORKSPACE_BYTES_PRE_HOPPER
            == CUBLAS_WORKSPACE_BYTES_LEGACY
        )

    @pytest.mark.parametrize("config", ["", "4096", ":abc", ":4096", ":", "nonsense"])
    def test_unparseable_is_none_so_the_device_default_applies(self, config):
        """No match at all: PyTorch ``TORCH_WARN``s and returns ``default_size``,
        and so does this. A pre-flight must not die here, and must not charge a
        size invented out of a string it could not parse."""
        assert parse_cublas_workspace_config(config) is None

    def test_a_pair_anywhere_in_the_string_counts_because_torch_searches(self):
        """CONTROL for the case above, and the reason it is not one. torch uses
        ``std::sregex_iterator``, which SCANS for matches rather than anchoring
        at the start, so ``garbage:4096:2`` is 8 MiB to PyTorch. An anchored
        ``match`` here would charge a different size than the allocator does."""
        assert parse_cublas_workspace_config("garbage:4096:2") == 4096 * 2 * 1024

    def test_unset_is_none_so_the_device_default_applies(self):
        assert parse_cublas_workspace_config(None) is None


class TestCublasWorkspaceSizeForTheDevice:
    """``parseChosenWorkspaceSize`` picks 32 MiB when the compute capability major
    is 9, 10, 11 or 12, and ``4096 * 1024 * 2 + 16 * 1024 * 8`` (8,320 KiB)
    otherwise."""

    @pytest.mark.parametrize("major", [9, 10, 11, 12])
    def test_hopper_and_blackwell_get_32_mib(self, major):
        assert cublas_workspace_size_bytes(major) == 32 * 1024 * 1024

    @pytest.mark.parametrize("major", [6, 7, 8])
    def test_anything_below_hopper_gets_the_8320_kib_default(self, major):
        assert cublas_workspace_size_bytes(major) == CUBLAS_WORKSPACE_BYTES_PRE_HOPPER

    def test_the_legacy_size_is_8320_kib_not_8_mib(self):
        """``4096 * 1024 * 2 + 16 * 1024 * 8`` is 8,320 KiB, not 8,192: the trailing
        ``:16:8`` pair is part of the default, and a fit that used 8 MiB would
        under-charge by 320 KiB per handle on every pre-Hopper card."""
        assert CUBLAS_WORKSPACE_BYTES_PRE_HOPPER == 8_320 * 1024
        assert CUBLAS_WORKSPACE_BYTES_PRE_HOPPER == 8_519_680

    def test_the_rtx_3050_the_grid_was_measured_on_is_below_hopper(self):
        # cc 8.6. If this ever moved, the grid's measured peaks would stop
        # containing what TestTheGridIsUnmoved assumes they contain.
        assert cublas_workspace_size_bytes(CC_LEGACY) == CUBLAS_WORKSPACE_BYTES_LEGACY

    def test_the_env_var_overrides_the_device_default(self):
        assert cublas_workspace_size_bytes(CC_BLACKWELL, config=":4096:1") == 4 * 1024 * 1024

    def test_the_env_var_can_remove_the_workspaces_entirely(self):
        assert cublas_workspace_size_bytes(CC_BLACKWELL, config=":0:0") == 0

    def test_a_zero_major_falls_back_to_the_default_rather_than_reading_props(self):
        """A device whose properties could not be read must not be treated as
        compute capability 0 and given a made-up size silently -- but it also must
        not refuse the run. It takes the below-Hopper size, which is what PyTorch
        uses on every card whose capability it could not determine."""
        assert cublas_workspace_size_bytes(0) == CUBLAS_WORKSPACE_BYTES_PRE_HOPPER

    def test_the_majors_that_get_32_mib_are_pinned_not_derived(self):
        """The rule is a list in PyTorch's source, not a threshold. A card newer
        than Blackwell must not be silently excluded by a comparison, and must not
        be charged 32 MiB until PyTorch's own rule says so."""
        assert sorted(CUBLAS_WORKSPACE_BYTES_BY_MAJOR) == [9, 10, 11, 12]
        for major, size in CUBLAS_WORKSPACE_BYTES_BY_MAJOR.items():
            assert cublas_workspace_size_bytes(major) == size == 32 * 1024 * 1024


class TestTheHandleCount:
    """One workspace per (handle, stream) pair. A streamed step uses two: the
    forward's thread and autograd's CUDA thread, which draws its own handle from
    a ``thread_local`` pool. The reporter's step held exactly two 32 MiB blocks
    and nothing else, so the count is measured, not assumed."""

    def test_a_step_is_charged_two_workspaces(self):
        assert CUBLAS_WORKSPACE_HANDLES_PER_STREAMED_STEP == 2

    @pytest.mark.parametrize("major", [CC_LEGACY, CC_HOPPER, CC_BLACKWELL])
    def test_two_handles_is_the_size_times_two(self, major):
        assert estimate_cublas_workspace_bytes(major) == 2 * cublas_workspace_size_bytes(major)

    def test_a_blackwell_step_holds_64_mib(self):
        assert estimate_cublas_workspace_bytes(CC_BLACKWELL) == 64 * 1024 * 1024

    def test_a_legacy_step_holds_16_3_mib(self):
        assert estimate_cublas_workspace_bytes(CC_LEGACY) == 17_039_360

    def test_cublas_lt_does_not_add_a_third(self):
        """PyTorch sizes a cuBLASLt workspace separately, but on CUDA
        ``getCurrentCUDABlasLtHandle`` ALIASES ``getCurrentCUDABlasHandle``
        rather than drawing another handle, so the two share one allocation.
        Counting both would over-charge every step by 32 MiB."""
        assert estimate_cublas_workspace_bytes(CC_BLACKWELL) == 2 * 32 * 1024 * 1024

    def test_rejects_a_negative_handle_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            estimate_cublas_workspace_bytes(CC_BLACKWELL, handles=-1)


class TestTheFormulaChargesIt:
    """The gap itself, and the acceptance criterion that the charge must match
    what the allocator does rather than merely grow."""

    #: The 2-layer fixture of ``test_v07204`` at 2 rows x seq 64 -- the shape whose
    #: prediction was 13,934,816 against a measured 67,734,016.
    _FIXTURE = dict(
        layer_bytes=14_160_384 // 2,
        buffers=2,
        extras_bytes=0,
        adapter_params=0,
        vocab_size=64,
        hidden_size=64,
        intermediate_size=160,
        n_layers=2,
        seq_len=64,
        batch_size=2,
    )

    #: Measured on an RTX 5070 (driver 616.92, torch 2.14.0+cu130), 2 rows at
    #: seq 64, one DPO step forward and backward.
    MEASURED_PEAK = 67_734_016
    MEASURED_PEAK_4MIB = 9_013_760
    #: The step's own workspace contribution, as the reporter measured it:
    #: 58,720,256 B = 2 x (32 - 4) MiB.
    MEASURED_WORKSPACE_DELTA = 58_720_256

    def _peak(self, **kwargs):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        return estimate_stream_peak_vram(**{**self._FIXTURE, **kwargs})

    def test_the_term_moves_the_prediction_by_exactly_the_workspaces(self):
        without = self._peak(cublas_workspace_bytes=0)
        charged = self._peak(cublas_workspace_bytes=estimate_cublas_workspace_bytes(CC_BLACKWELL))
        assert charged - without == 64 * 1024 * 1024

    def test_the_prediction_clears_the_reported_peak(self):
        """The measured DPO peak on the RTX 5070 was 67,734,016 B; the prediction
        for this shape before the term was 13,934,816 B."""
        predicted = self._peak(
            cublas_workspace_bytes=estimate_cublas_workspace_bytes(CC_BLACKWELL)
        )
        assert predicted >= self.MEASURED_PEAK, (
            f"predicted {predicted} is still below the measured {self.MEASURED_PEAK} "
            f"peak reported on an RTX 5070 (cc 12.0)"
        )

    def test_the_charge_matches_the_allocator_when_the_operator_shrinks_it(self):
        """Acceptance criterion 3. ``CUBLAS_WORKSPACE_CONFIG`` is read once per
        process by PyTorch, and the estimate must move by exactly what the
        allocator moved: 58,720,256 B, i.e. 2 x (32 - 4) MiB. A delta-over-fitted
        baseline cannot do this -- it reads 0 here, because 2 x 4 MiB is below the
        legacy size the constant already carries."""
        default = self._peak(
            cublas_workspace_bytes=estimate_cublas_workspace_bytes(CC_BLACKWELL)
        )
        shrunk = self._peak(
            cublas_workspace_bytes=estimate_cublas_workspace_bytes(
                CC_BLACKWELL, config=":4096:1"
            )
        )
        assert default - shrunk == self.MEASURED_WORKSPACE_DELTA

    def test_it_still_covers_the_step_with_the_workspaces_removed(self):
        """``:0:0`` removes the workspaces outright. The fix must not depend on
        them being large."""
        assert (
            self._peak(
                cublas_workspace_bytes=estimate_cublas_workspace_bytes(
                    CC_BLACKWELL, config=":0:0"
                )
            )
            >= self.MEASURED_PEAK_4MIB
        )

    def test_the_backward_only_peak_is_also_cleared(self):
        """With the backward removed the step held ONE workspace: 34,060,288 B
        (32 MiB + 505,856). A single handle's worth plus the modelled terms must
        cover it, or the term is sized for the wrong step."""
        assert self._peak(cublas_workspace_bytes=32 * 1024 * 1024) >= 34_060_288

    def test_the_default_is_the_device_charge_not_zero(self):
        """A caller that omits the term must not silently get the
        under-predicting formula this issue is about."""
        assert self._peak() == self._peak(
            cublas_workspace_bytes=measure_cublas_workspace_bytes()
        )

    def test_a_negative_charge_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._peak(cublas_workspace_bytes=-1)


class TestTheGridIsUnmoved:
    """``tests/test_v07203.py`` carries the ``<1%`` claim, and it was measured on an
    RTX 3050 (cc 8.6) whose ``max_memory_allocated`` peaks ALREADY contain two
    8,320 KiB workspaces. Naming the term therefore double counts 16.3 MB there.
    These tests state that consequence rather than leaving it to be discovered."""

    def test_the_grid_card_held_16_3_mib_of_workspace(self):
        assert 2 * CUBLAS_WORKSPACE_BYTES_LEGACY == 17_039_360

    def test_the_constant_could_never_have_covered_one_blackwell_workspace(self):
        """The arithmetic that makes a missing term necessary at all: if the 13.5 MB
        constant could hold a 32 MiB workspace, nothing would be missing."""
        assert STREAM_FIXED_SLACK_BYTES < 32 * 1024 * 1024
        assert STREAM_FIXED_SLACK_BYTES < estimate_cublas_workspace_bytes(CC_BLACKWELL)

    def test_the_grid_moves_by_exactly_the_workspace_it_already_held(self):
        """Bounded re-scope: the re-widened band is arithmetic, not a re-fit."""
        from tests.test_v07203 import MEASURED_VRAM_GRID, _predict

        for row in MEASURED_VRAM_GRID:
            assert (
                _predict(row, compute_capability_major=CC_LEGACY) - _predict(row)
                == 2 * CUBLAS_WORKSPACE_BYTES_LEGACY
            )

    def test_every_grid_row_stays_an_over_prediction(self):
        """The safe direction for a gate that refuses runs. If this ever goes
        negative, the fix has started UNDER-predicting on the only card the grid
        covers -- which is the failure this issue exists to remove."""
        from tests.test_v07203 import MEASURED_VRAM_GRID, _predict

        for row in MEASURED_VRAM_GRID:
            assert _predict(row, compute_capability_major=CC_LEGACY) > row["peak"], (
                row["label"]
            )

    def test_the_same_row_needs_46_8_mib_more_on_blackwell(self):
        """What the grid cannot see: on cc 12.0 the same shape carries 46.8 MiB more
        workspace than the constant was fitted against."""
        from tests.test_v07203 import MEASURED_VRAM_GRID, _predict

        row = MEASURED_VRAM_GRID[0]
        delta = _predict(row, compute_capability_major=CC_BLACKWELL) - _predict(
            row, compute_capability_major=CC_LEGACY
        )
        assert delta == 64 * 1024 * 1024 - 2 * CUBLAS_WORKSPACE_BYTES_LEGACY


class TestTheEnvVarIsReadFromTheProcess:
    """PyTorch reads ``CUBLAS_WORKSPACE_CONFIG`` once per process, so the estimate
    reads the same variable rather than carrying a second knob that could disagree
    with the library that allocates the memory."""

    def test_the_name_is_the_one_pytorch_reads(self):
        assert CUBLAS_WORKSPACE_CONFIG_ENV == "CUBLAS_WORKSPACE_CONFIG"

    def _pretend_device(self, monkeypatch, major):
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream, "_device_compute_capability_major", lambda device: major
        )

    def test_the_probe_follows_the_variable(self, monkeypatch):
        monkeypatch.setenv(CUBLAS_WORKSPACE_CONFIG_ENV, ":4096:1")
        self._pretend_device(monkeypatch, CC_BLACKWELL)
        assert measure_cublas_workspace_bytes() == 2 * 4 * 1024 * 1024

    def test_unset_leaves_the_device_rule_in_charge(self, monkeypatch):
        monkeypatch.delenv(CUBLAS_WORKSPACE_CONFIG_ENV, raising=False)
        self._pretend_device(monkeypatch, CC_BLACKWELL)
        assert measure_cublas_workspace_bytes() == 64 * 1024 * 1024

    def test_no_cuda_device_means_no_charge_at_all(self, monkeypatch):
        """A CPU-only machine has no cuBLAS workspace. Billing one would refuse runs
        over memory that does not exist, and CI runs without a GPU."""
        monkeypatch.delenv(CUBLAS_WORKSPACE_CONFIG_ENV, raising=False)
        self._pretend_device(monkeypatch, None)
        assert measure_cublas_workspace_bytes() == 0


@pytest.mark.gpu(reason="needs a CUDA device to read the real workspace size")
class TestTheRealDeviceIsAsked:
    """``torch.cuda.get_device_properties`` is authoritative; the mirrored rule
    above is only the fallback for a card whose properties cannot be read."""

    def test_the_charge_matches_the_live_device(self):
        import torch

        major = torch.cuda.get_device_properties(torch.cuda.current_device()).major
        assert measure_cublas_workspace_bytes() == estimate_cublas_workspace_bytes(major)

    def test_the_workspaces_the_charge_names_are_real_allocations(self):
        """Asserted against the allocator rather than the mirrored rule, so a stale
        table in the rule above cannot make it pass: a matmul on this thread and a
        backward on autograd's CUDA thread are the two (handle, stream) pairs a
        streamed step uses, and the allocator must grow by the workspace."""
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (a @ a).sum().backward()
        grew = torch.cuda.memory_allocated() - before

        major = torch.cuda.get_device_properties(torch.cuda.current_device()).major
        assert grew >= estimate_cublas_workspace_bytes(major), (
            f"charged {estimate_cublas_workspace_bytes(major)} but the allocator "
            f"only grew by {grew}"
        )
