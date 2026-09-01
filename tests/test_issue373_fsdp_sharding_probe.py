"""Regression coverage for the FSDP per-rank sharding probe (#373).

The probe these tests pin is a measurement harness, not shipped code. It exists
because the STEP 20 probe that established "local params = total / world_size
exactly" at 0.5B raised
``TypeError('FullyShardedDataParallel does not support len()')`` on all eight
ranks at 70B, so no sharding claim could be made there.

The mechanism recorded in the issue body was wrong, and the correction is what
these tests defend. That message is **not** raised by FSDP -- FSDP defines no
``__len__`` at all, and a bare one produces Python's own
``object of type 'X' has no len()``. The recorded text comes from
``torch._dynamo.eval_frame.OptimizedModule.__len__``, which reports the name of
the *wrapped* class. The failing object was therefore a ``torch.compile``
wrapper around FSDP, one indirection deeper than the 0.5B arm.

So "stop calling len()" is necessary but not sufficient: a probe that walks the
tree without unwrapping ``_orig_mod`` never raises and still mis-attributes what
it finds -- it sees ``OptimizedModule`` where the 0.5B record says
``FullyShardedDataParallel``, and concludes FSDP was never engaged. Silent wrong
numbers are worse than a crash, so the unwrapping is pinned here directly.

No GPU, no distributed init and no torch import: the probe's accounting is duck
typed over anything exposing ``parameters()`` / ``named_children()``, and these
doubles reproduce the ``OptimizedModule`` contract including its ``__len__``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PROBE_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "harness" / "fsdp_sharding_probe.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("fsdp_sharding_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves sys.modules[cls.__module__]; register before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


# --------------------------------------------------------------------------
# Doubles. These stand in for torch objects the CPU test environment cannot
# build: FSDP needs a process group, and OptimizedModule needs a compile.
# --------------------------------------------------------------------------


class FakeParameter:
    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class FakeModule:
    """Minimal nn.Module surface the probe actually consumes."""

    def __init__(self, params=(), children=None) -> None:
        self._params = list(params)
        self._children = dict(children or {})

    def parameters(self):
        yield from self._params
        for child in self._children.values():
            yield from child.parameters()

    def named_children(self):
        yield from self._children.items()


class FullyShardedDataParallel(FakeModule):
    """Name matters: the probe's histogram is compared against the 0.5B record."""


class OptimizedModule(FakeModule):
    """Reproduces torch._dynamo.eval_frame.OptimizedModule, __len__ included."""

    def __init__(self, orig_mod: FakeModule) -> None:
        super().__init__(children={"_orig_mod": orig_mod})
        self._orig_mod = orig_mod

    def __len__(self) -> int:
        # Byte-identical to eval_frame.py:476 -- the behaviour that broke STEP 20.
        if hasattr(self._orig_mod, "__len__"):
            return len(self._orig_mod)
        raise TypeError(f"{type(self._orig_mod).__name__} does not support len()")

    def parameters(self):
        yield from self._orig_mod.parameters()

    def named_children(self):
        yield from self._orig_mod.named_children()


def _sharded_rank_model(local_numel: int = 124_048_864):
    """A compiled-FSDP tree shaped like the 70B arm that failed."""
    inner = FullyShardedDataParallel(params=[FakeParameter(local_numel)])
    return OptimizedModule(inner)


class TestTheRecordedFailureIsReproduced:
    """The double must actually reproduce the defect, or it pins nothing."""

    def test_the_double_raises_the_recorded_message(self) -> None:
        model = _sharded_rank_model()
        with pytest.raises(TypeError) as excinfo:
            len(model)
        assert str(excinfo.value) == "FullyShardedDataParallel does not support len()"

    def test_a_bare_fsdp_module_raises_a_different_message(self) -> None:
        """FSDP is not the thing that raised; that is the whole correction."""
        with pytest.raises(TypeError) as excinfo:
            len(FullyShardedDataParallel())
        assert "does not support len()" not in str(excinfo.value)


class TestProbeSurvivesTheFailure:
    def test_local_parameter_numel_does_not_call_len(self) -> None:
        model = _sharded_rank_model(local_numel=124_048_864)
        assert probe.local_parameter_numel(model) == 124_048_864

    def test_walking_the_tree_does_not_call_len(self) -> None:
        model = _sharded_rank_model()
        # Raises TypeError if the walk measures length anywhere.
        assert probe.module_class_histogram(model)


class TestUnwrappingIsPinned:
    """The maintainer's request: pin the unwrapping, not just the absence of len().

    A probe that stops at the compile wrapper never raises and still reports the
    wrong thing. These tests fail for that probe.
    """

    def test_histogram_reports_the_wrapped_class_not_the_wrapper(self) -> None:
        model = _sharded_rank_model()
        histogram = probe.module_class_histogram(model)
        assert histogram["FullyShardedDataParallel"] == 1
        assert "OptimizedModule" not in histogram

    def test_fsdp_wrappers_are_counted_through_the_compile_wrapper(self) -> None:
        """The 0.5B record's evidence was '217x FullyShardedDataParallel'."""
        leaves = {
            f"layer{i}": FullyShardedDataParallel(params=[FakeParameter(10)])
            for i in range(217)
        }
        model = OptimizedModule(FullyShardedDataParallel(children=leaves))
        histogram = probe.module_class_histogram(model)
        assert histogram["FullyShardedDataParallel"] == 218  # 217 leaves + the root

    def test_unwrap_compiled_is_idempotent_on_an_uncompiled_module(self) -> None:
        bare = FullyShardedDataParallel()
        assert probe.unwrap_compiled(bare) is bare

    def test_unwrap_compiled_handles_nested_wrappers(self) -> None:
        inner = FullyShardedDataParallel()
        assert probe.unwrap_compiled(OptimizedModule(OptimizedModule(inner))) is inner


class TestShardingVerdict:
    """Acceptance criterion 2: asserted, not eyeballed."""

    def test_exact_division_is_sharded(self) -> None:
        verdict = probe.sharding_verdict(local=124_048_864, total=496_195_456, world_size=4)
        assert verdict.is_sharded is True
        assert verdict.expected_local == 124_048_864

    def test_inexact_division_is_not_sharded(self) -> None:
        verdict = probe.sharding_verdict(local=124_048_865, total=496_195_456, world_size=4)
        assert verdict.is_sharded is False

    def test_single_gpu_control_reports_the_full_count(self) -> None:
        """Acceptance criterion 3: the probe cannot be satisfied by always dividing."""
        verdict = probe.sharding_verdict(local=496_195_456, total=496_195_456, world_size=1)
        assert verdict.is_sharded is True
        assert verdict.expected_local == 496_195_456

    def test_unsharded_run_on_multiple_ranks_is_caught(self) -> None:
        """The flag was accepted but FSDP never engaged: local == total, world > 1."""
        verdict = probe.sharding_verdict(local=496_195_456, total=496_195_456, world_size=4)
        assert verdict.is_sharded is False

    def test_under_count_is_not_sharded(self) -> None:
        """A rank holding *less* than its share is not a pass.

        Caught by mutation: loosening the check to ``local <= expected_local``
        survived the rest of this class. An under-count is a real failure shape
        -- a rank whose parameters were never materialised reports a small
        number, and "less than expected" must not read as "sharded".
        """
        verdict = probe.sharding_verdict(local=1, total=496_195_456, world_size=4)
        assert verdict.is_sharded is False

    def test_world_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            probe.sharding_verdict(local=1, total=1, world_size=0)
