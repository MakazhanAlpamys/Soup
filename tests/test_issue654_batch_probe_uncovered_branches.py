"""#654 follow-ups — the two branches its merge comment recorded as uncovered.

#654 fixed the CUDA batch probe to gate on a measured peak rather than on the
absence of an ``OutOfMemoryError``, and recorded two things it did **not**
cover. Both are closed here.

**1. The graceful-degradation branch.** ``_probe_budget_bytes`` returns ``None``
when the driver cannot answer ``mem_get_info``, and ``_probe`` then returns
``True`` — deliberately keeping the pre-#649 exception-only criterion instead of
inventing a budget. Nothing executed that path: the fake in
``test_issue649_batch_probe_peak_gate.py`` builds ``mem_get_info`` as a lambda
that cannot fail. Flipping the branch to ``False`` would make the probe refuse
**every** batch on such a stack, down to ``_MIN_BATCH``, and no test would move.

**2. The ``AcceleratorError`` contract.** ``_is_cuda_oom`` classifies torch's
async-OOM spelling through ``isinstance(exc, RuntimeError)`` — it never names
``AcceleratorError``. That is correct upstream, but the existing test's
``_FakeAcceleratorError`` *also* subclasses ``RuntimeError``, so it passes
whether or not the real class does, and proves nothing about torch.

The merge comment reasoned that no CI cell could check the real assumption,
because the author's box had torch 2.5.1. That does not follow: ``[dev]`` pulls
``[train]``, which declares ``torch>=2.5.0`` — a floor, not a pin — so CI
resolves the newest wheel. Measured in this repo's ``.venv``: torch 2.13.0,
``AcceleratorError`` present, ``issubclass(..., RuntimeError)`` True. So the
guard below **executes** rather than skips. Where the symbol is genuinely
absent it skips with a stated reason rather than passing vacuously — the
convention ``tests/test_issue636_torch_floor.py`` sets for environment-dependent
checks.
"""

from __future__ import annotations

import types

import pytest

from tests.test_issue649_batch_probe_peak_gate import (
    GB,
    _Model,
    _probe_with,
    _Tensor,
)


def _torch_without_mem_get_info(*, peak_allocated: int, raises: bool = True):
    """A ``torch`` whose driver cannot answer ``mem_get_info``.

    ``peak_allocated`` is deliberately set far ABOVE anything a budget could
    be, so ``peak <= budget`` would be ``False`` if a budget were ever
    computed. A ``True`` from the probe can then only have come from the
    ``budget is None`` path — the test cannot pass for the wrong reason.
    """
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")

    class _FakeOOMError(Exception):
        pass

    cuda.OutOfMemoryError = _FakeOOMError
    cuda.is_available = lambda: True
    cuda.synchronize = lambda *a, **k: None
    cuda.empty_cache = lambda: None
    cuda.memory_allocated = lambda *a, **k: 0

    if raises:
        def mem_get_info(*a, **k):
            # What a driver that cannot answer actually does.
            raise RuntimeError("CUDA driver error: cannot query memory info")
    else:
        # The control: a driver that CAN answer, same peak, tiny budget.
        def mem_get_info(*a, **k):
            return (1 * GB, 4 * GB)

    cuda.mem_get_info = mem_get_info
    cuda.reset_peak_memory_stats = lambda *a, **k: None
    cuda.max_memory_allocated = lambda *a, **k: peak_allocated
    cuda.max_memory_reserved = lambda *a, **k: peak_allocated

    torch.cuda = cuda
    torch.long = "long"
    torch.full = lambda *a, **k: _Tensor()
    torch.ones_like = lambda *a, **k: _Tensor()
    return torch


class TestBudgetUnavailableKeepsTheExceptionOnlyCriterion:
    """#654 follow-up 1: ``if budget is None: return True``."""

    def test_probe_approves_when_the_driver_cannot_report_memory(self, monkeypatch):
        """A driver that cannot answer must not turn into a refusal of everything.

        The step completes and raises nothing, so the pre-#649 criterion says
        "fits". The peak here is 999 GB — larger than any budget — so if the
        gate were reached at all this would return False.
        """
        torch = _torch_without_mem_get_info(peak_allocated=999 * GB)
        probe = _probe_with(monkeypatch, torch, model=_Model())

        assert probe(8) is True

    def test_control_same_peak_is_refused_when_the_driver_can_answer(self, monkeypatch):
        """The control that makes the test above mean something.

        Identical 999 GB peak; the only change is that ``mem_get_info`` works.
        If this also returned True the first test would prove nothing about the
        ``budget is None`` path.
        """
        torch = _torch_without_mem_get_info(peak_allocated=999 * GB, raises=False)
        probe = _probe_with(monkeypatch, torch, model=_Model())

        assert probe(8) is False


class TestAcceleratorErrorInheritanceContract:
    """#654 follow-up 2: the assumption ``_is_cuda_oom`` actually rides on."""

    def test_real_torch_accelerator_error_subclasses_runtime_error(self):
        """Pin the upstream contract against the INSTALLED torch, not a fake.

        ``_is_cuda_oom`` never names ``AcceleratorError``; it catches it via
        ``isinstance(exc, RuntimeError)``. If upstream ever reparents the class
        the probe would propagate an async OOM instead of shrinking the batch,
        and every fake-torch test would still pass. This is the only check in
        the suite that would notice.
        """
        torch = pytest.importorskip("torch", reason="torch is not installed in this env")
        acc = getattr(torch, "AcceleratorError", None)
        if acc is None:
            pytest.skip(
                f"torch {torch.__version__} predates torch.AcceleratorError "
                "(added in 2.8); nothing to pin here"
            )

        assert issubclass(acc, RuntimeError), (
            f"torch {torch.__version__} AcceleratorError no longer subclasses "
            f"RuntimeError (mro: {[c.__name__ for c in acc.__mro__]}). "
            "batch_probe._is_cuda_oom classifies it by inheritance, so an async "
            "CUDA OOM would now propagate out of the probe instead of being "
            "reported as 'does not fit'."
        )

    def test_an_accelerator_error_outside_runtimeerror_would_propagate(self, monkeypatch):
        """Demonstrate the consequence the guard above protects against.

        This is what makes that guard load-bearing rather than decorative: with
        a class that does NOT subclass ``RuntimeError``, the same 'out of
        memory' text is not classified, and the probe raises instead of
        returning False.
        """
        from soup_cli.utils.batch_probe import _is_cuda_oom

        class _DetachedAcceleratorError(Exception):
            """An AcceleratorError that is not a RuntimeError."""

        torch = _torch_without_mem_get_info(peak_allocated=1 * GB, raises=False)
        torch.AcceleratorError = _DetachedAcceleratorError

        exc = _DetachedAcceleratorError("CUDA error: out of memory")
        assert _is_cuda_oom(exc, torch) is False

        # And end to end: the probe propagates rather than refusing the batch.
        def step():
            raise exc

        probe = _probe_with(monkeypatch, torch, model=_Model(step))
        with pytest.raises(_DetachedAcceleratorError, match="out of memory"):
            probe(8)

    def test_the_runtimeerror_spelling_is_still_classified(self, monkeypatch):
        """Reject-everything control: the classifier has not simply stopped working."""
        from soup_cli.utils.batch_probe import _is_cuda_oom

        torch = _torch_without_mem_get_info(peak_allocated=1 * GB, raises=False)
        assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"), torch) is True
        assert _is_cuda_oom(RuntimeError("an illegal memory access"), torch) is False
