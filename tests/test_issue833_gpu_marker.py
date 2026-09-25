"""#833 — one CUDA probe, one ``gpu`` marker, so ``pytest -m gpu`` selects the GPU subset.

Every CI runner is GPU-less, so a CUDA-dependent test only ever means something
when someone runs the suite on a card. Before this, fifteen modules had their own
probe and nine an identical ``requires_cuda`` skipif: there was nothing to select.

The criterion "``-m gpu`` selects every test that skips without CUDA" is held
structurally, because computing that set needs a CUDA box to diff against:

1. the hook in ``tests/conftest.py`` is the only thing that skips on missing CUDA,
   and it skips exactly the ``gpu``-marked tests (``TestTheHook``);
2. no other module under ``tests/`` may probe CUDA itself, or gate a skip on the
   shared probe (``TestNoPrivateProbe``), with controls built from every probe
   style that existed on ``main`` so the scan cannot pass by seeing nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest

TESTS = Path(__file__).resolve().parent

#: Skips gated on the shared probe that are allowed, each with why. Both skip
#: WITH CUDA: they assert the no-CUDA fallback, so ``-m gpu`` must not select them.
ALLOWED_SKIP_GATES = {
    (
        "test_issue349_measured_fit.py",
        "test_the_probe_returns_none_without_cuda_instead_of_raising",
    ): "skips when CUDA IS present: it asserts the no-CUDA fallback",
    ("test_v07120.py", "test_gates_without_bnb_or_cuda"):
        "skips when CUDA + bitsandbytes ARE present: it asserts the refusal without them",
}


def _calls_named(node: ast.AST, name: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called == name:
                return True
    return False


def _is_cuda_is_available_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "is_available"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    )


def _is_attr_call(node: ast.AST, owner: str, attr: str) -> bool:
    """``pytest.skip(...)`` / ``pytest.mark.skipif(...)`` style calls."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != attr:
        return False
    value = node.func.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return isinstance(value, ast.Name) and value.id == owner


def violations(source: str, filename: str = "<snippet>", allowed=None) -> list[str]:
    """Every private CUDA probe or CUDA-gated skip in ``source``."""
    allowed = ALLOWED_SKIP_GATES if allowed is None else allowed
    tree = ast.parse(source)
    found: list[str] = []

    for node in ast.walk(tree):
        if _is_cuda_is_available_call(node):
            found.append(f"{filename}:{node.lineno}: calls cuda.is_available() (use gpu marker)")
        if _is_attr_call(node, "pytest", "skipif") and node.args:
            if _calls_named(node.args[0], "cuda_available"):
                found.append(f"{filename}:{node.lineno}: skipif on cuda_available() (use gpu)")

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.If) and _calls_named(node.test, "cuda_available"):
            skips = [
                sub for stmt in node.body + node.orelse for sub in ast.walk(stmt)
                if _is_attr_call(sub, "pytest", "skip")
            ]
            if skips and (filename, function) not in allowed:
                found.append(
                    f"{filename}:{node.lineno}: pytest.skip gated on cuda_available() "
                    f"in {function} (use the gpu marker)"
                )
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


class TestNoPrivateProbe:
    def test_no_test_module_probes_cuda_itself(self):
        exempt = {"conftest.py", Path(__file__).name}
        found: list[str] = []
        scanned = 0
        for path in sorted(TESTS.rglob("*.py")):
            if path.name in exempt:
                continue
            scanned += 1
            found += violations(path.read_text(encoding="utf-8"), path.name)
        assert scanned > 500, f"the scan saw only {scanned} test modules"
        assert found == [], "\n".join(found)

    @pytest.mark.parametrize(
        "snippet",
        [
            # Every probe style that existed on main (5ee1278), trimmed.
            "def _cuda() -> bool:\n    return torch.cuda.is_available()\n",
            "def _cuda_available():\n    try:\n        import torch\n\n"
            "        return torch.cuda.is_available()\n"
            "    except Exception:\n        return False\n",
            "def _cuda():\n    return bool(torch.cuda.is_available())\n",
            "_NO_CUDA = not torch.cuda.is_available()\n",
            "@pytest.mark.skipif(not torch.cuda.is_available(), reason='x')\n"
            "def test_a():\n    pass\n",
            "def test_a():\n    if not torch.cuda.is_available():\n        pytest.skip('x')\n",
            "def test_a():\n    if not (torch.cuda.is_available() and bnb):\n"
            "        pytest.skip('x')\n",
            # The shared probe, used to gate a skip instead of the marker.
            "requires_cuda = pytest.mark.skipif(not cuda_available(), reason='x')\n",
            "def test_a():\n    if not conftest.cuda_available():\n        pytest.skip('x')\n",
        ],
    )
    def test_the_scan_rejects_every_probe_style_main_had(self, snippet):
        assert violations(snippet), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            "@pytest.mark.gpu\ndef test_a():\n    pass\n",
            "device = 'cuda' if cuda_available() else 'cpu'\n",
            "fake_torch.cuda.is_available.return_value = True\n",
            "with patch('torch.cuda.is_available', return_value=False):\n    pass\n",
            "cuda.is_available = lambda: True\n",
        ],
    )
    def test_the_scan_accepts_what_is_not_a_probe(self, snippet):
        assert violations(snippet) == [], snippet

    def test_the_allowlist_is_exact(self):
        """Each entry must still be a CUDA-gated skip the scan would otherwise
        flag, or it is a hole waiting for the next probe to hide in."""
        for (filename, function), _why in ALLOWED_SKIP_GATES.items():
            source = (TESTS / filename).read_text(encoding="utf-8")
            unallowed = violations(source, filename, allowed={})
            assert any(f" in {function} " in v for v in unallowed), (filename, function)


def _item(*markers):
    by_name = {m.name: m for m in markers}
    return SimpleNamespace(get_closest_marker=lambda name: by_name.get(name))


def _run_setup(item) -> None:
    """Call the hook; a skip here is the regression under test, so report it as a failure."""
    try:
        conftest.pytest_runtest_setup(item)
    except pytest.skip.Exception as exc:
        pytest.fail(f"the hook skipped a test it should have run: {exc}")


class TestTheHook:
    def test_a_gpu_test_skips_without_cuda(self, monkeypatch):
        monkeypatch.setattr(conftest, "_cuda_device_available", lambda: False)
        with pytest.raises(pytest.skip.Exception, match="pytest -m gpu"):
            conftest.pytest_runtest_setup(_item(pytest.mark.gpu.mark))

    def test_the_reason_is_kept(self, monkeypatch):
        monkeypatch.setattr(conftest, "_cuda_device_available", lambda: False)
        with pytest.raises(pytest.skip.Exception, match="peak VRAM needs CUDA"):
            conftest.pytest_runtest_setup(
                _item(pytest.mark.gpu(reason="peak VRAM needs CUDA").mark)
            )

    def test_a_gpu_test_runs_with_cuda(self, monkeypatch):
        monkeypatch.setattr(conftest, "_cuda_device_available", lambda: True)
        _run_setup(_item(pytest.mark.gpu.mark))

    def test_an_unmarked_test_runs_without_cuda(self, monkeypatch):
        """Control: the hook skips the marker, not everything."""
        monkeypatch.setattr(conftest, "_cuda_device_available", lambda: False)
        _run_setup(_item())
        _run_setup(_item(pytest.mark.smoke.mark))

    def test_the_marker_is_registered(self, request):
        assert any(line.startswith("gpu:") for line in request.config.getini("markers"))

    def test_the_probe_is_a_bool(self):
        assert isinstance(conftest.cuda_available(), bool)


class TestDeviceCountGating:
    """The gpu marker must gate on initial visible device count > 0 (#1128).

    Under CUDA_VISIBLE_DEVICES="", PyTorch may report device_count() == 0 pre-init
    but report 1 if a CUDA tensor is allocated post-init. Pinning the count at
    session start in conftest keeps the gpu marker deterministic across the suite,
    while leaving exported conftest.cuda_available() answering is_available() so
    other test callers do not regress.
    """

    def test_probe_initial_device_count_returns_zero_when_no_device_visible(
        self, monkeypatch
    ):
        """Under CUDA_VISIBLE_DEVICES='', is_available is True but device_count is 0."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        assert conftest._probe_initial_device_count() == 0

    def test_probe_initial_device_count_returns_count_when_device_visible(
        self, monkeypatch
    ):
        """When device is visible, probe returns positive count."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        assert conftest._probe_initial_device_count() == 1

    def test_probe_initial_device_count_fails_safe_on_exception(self, monkeypatch):
        """Broken driver or torch import error returns 0 safely."""
        import torch

        def _boom():
            raise RuntimeError("CUDA initialization error")

        monkeypatch.setattr(torch.cuda, "is_available", _boom)
        assert conftest._probe_initial_device_count() == 0

    def test_predicate_returns_false_when_initial_device_count_is_zero(
        self, monkeypatch
    ):
        """_cuda_device_available() returns False when initial count is 0 (#1128)."""
        monkeypatch.setattr(conftest, "_INITIAL_CUDA_DEVICE_COUNT", 0)
        assert conftest._cuda_device_available() is False

    def test_predicate_returns_true_when_device_is_available_and_visible(
        self, monkeypatch
    ):
        """Prove both directions: when card was visible at start, probe returns True (#1128)."""
        monkeypatch.setattr(conftest, "_INITIAL_CUDA_DEVICE_COUNT", 1)
        assert conftest._cuda_device_available() is True

    def test_exported_cuda_available_answers_is_available(self, monkeypatch):
        """Exported cuda_available() continues answering is_available() (#1128 review)."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(conftest, "_INITIAL_CUDA_DEVICE_COUNT", 0)

        conftest.cuda_available.cache_clear()
        try:
            assert conftest.cuda_available() is True
        finally:
            conftest.cuda_available.cache_clear()

    def test_gpu_test_skips_when_initial_device_count_is_zero(self, monkeypatch):
        """End-to-end hook check: zero visible devices causes the gpu test to skip (#1128)."""
        monkeypatch.setattr(conftest, "_INITIAL_CUDA_DEVICE_COUNT", 0)
        with pytest.raises(pytest.skip.Exception, match="needs a CUDA device"):
            conftest.pytest_runtest_setup(_item(pytest.mark.gpu.mark))

    def test_gpu_test_runs_when_initial_device_count_is_positive(self, monkeypatch):
        """End-to-end hook check: positive visible devices allows test to run (#1128)."""
        monkeypatch.setattr(conftest, "_INITIAL_CUDA_DEVICE_COUNT", 1)
        _run_setup(_item(pytest.mark.gpu.mark))

    def test_subprocess_initial_count_matches_fresh_probe_under_env(self):
        """Reload conftest in child processes and assert _INITIAL_CUDA_DEVICE_COUNT matches probe.

        Verifies the pinned constant reflects a fresh probe (#1128).
        """
        import os
        import subprocess
        import sys

        code = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd()))\n"
            "import tests.conftest as c\n"
            "fresh = c._probe_initial_device_count()\n"
            "assert c._INITIAL_CUDA_DEVICE_COUNT == fresh, "
            "f'mismatch: pinned {c._INITIAL_CUDA_DEVICE_COUNT} vs fresh {fresh}'\n"
        )
        for env_val in ["", "-1"]:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = env_val
            res = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(TESTS.parent),
            )
            assert res.returncode == 0, f"Subprocess failed with CVD={env_val!r}:\n{res.stderr}"

    @pytest.mark.parametrize(
        ("available", "device_count", "expected"),
        [
            (True, 0, 0),
            (True, 2, 2),
            (False, 3, 0),
        ],
    )
    def test_subprocess_pinned_count_faked_cuda(
        self, available: bool, device_count: int, expected: int
    ) -> None:
        """Assert _INITIAL_CUDA_DEVICE_COUNT in fresh child with faked torch.cuda (#1128).

        Catches both fail-open (hard-wired 1) and skip-everywhere (hard-wired 0)
        mutations without requiring a physical GPU on CI.
        """
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd()))\n"
            "import torch\n"
            f"torch.cuda.is_available = lambda: {available}\n"
            f"torch.cuda.device_count = lambda: {device_count}\n"
            "import tests.conftest as c\n"
            "print(c._INITIAL_CUDA_DEVICE_COUNT)\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(TESTS.parent),
        )
        assert res.returncode == 0, f"Subprocess failed:\n{res.stderr}"
        assert int(res.stdout.strip()) == expected
