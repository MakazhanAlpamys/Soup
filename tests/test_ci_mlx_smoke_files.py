"""Every `tests/test_mlx_*.py` that needs the real MLX library runs in the `mlx-smoke` job.

`mlx` is not in `[dev]`, so the 3x3 test matrix cannot import it and every test that
`importorskip`s `mlx.core` / `mlx_lm` skips there. `mlx-smoke` (macOS, `.[mlx]` and `pytest`
only) is the one job that can execute them, and it runs an explicit list of files. A new MLX
test file that nobody adds to that list therefore runs on no job at all, and the suite stays
green whatever its tests would say. That is how the DPO, ORPO and eval-steps files (#1846,
#2073, #2074) went unexecuted: their loss, gradient and training-loop tests skip wherever MLX
is absent, and the list named only the flight-recorder and DoRA files.

The contract pinned here: a `tests/test_mlx_*.py` that gates a test on the real library with
`pytest.importorskip("mlx...")` is listed in a `pytest` step of the `mlx-smoke` job, or sits in
`EXEMPT` with the reason it is not. A listed file that no longer exists, and an exemption that
no longer applies, fail too, so neither list can rot.
"""

from __future__ import annotations

import ast
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._source_scan import parse_source, walk_nodes

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
TESTS = ROOT / "tests"
JOB = "mlx-smoke"

#: Files named `tests/test_mlx_*.py` that need the real library and are deliberately not run by
#: the job, with the reason. Adding an entry is a decision a reviewer should see, so it carries
#: a sentence, not a flag.
EXEMPT: dict[str, str] = {
    "test_mlx_max_grad_norm.py": (
        "three of its tests read the installed mlx-lm / mlx surface (TrainingArgs has no clipping "
        "field, mlx.optimizers.clip_grad_norm exists) as a tripwire for an upstream release; "
        "mlx and mlx-lm are not in .github/constraints/pinned-upstream.txt, so listing the file "
        "would turn a required job red for every pull request the day mlx-lm adds such a field. "
        "Where those three should run is a separate decision."
    ),
}


def _needs_real_mlx(source: str) -> bool:
    """True when some test is gated on a real MLX package: `pytest.importorskip("mlx...")`.

    `importorskip` is the only gate this suite uses for MLX (a test that imported it without a
    gate would fail collection in the 3x3 matrix). The call may sit in a fixture, a method, a
    module-level line or an assignment, be imported bare, or pass the name as `modname=`.
    """
    for node in walk_nodes(parse_source(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "importorskip":
            continue
        arguments = list(node.args) + [kw.value for kw in node.keywords if kw.arg == "modname"]
        if not arguments:
            continue
        module = arguments[0]
        if isinstance(module, ast.Constant) and isinstance(module.value, str):
            if module.value == "mlx" or module.value.startswith(("mlx.", "mlx_")):
                return True
    return False


def _steps(workflow_text: str, job: str = JOB) -> list[dict[str, Any]]:
    return yaml.safe_load(workflow_text)["jobs"][job]["steps"]


def _pytest_commands(steps: list[dict[str, Any]]) -> list[list[str]]:
    """The argv of every `pytest ...` line the steps run (a step may run several)."""
    commands = []
    for step in steps:
        for line in str(step.get("run", "")).splitlines():
            if line.strip().split(" ")[0] == "pytest":
                commands.append(shlex.split(line))
    return commands


def _listed_files(workflow_text: str) -> set[str]:
    """Basenames of the `tests/*.py` files the job's `pytest` commands name."""
    listed = set()
    for words in _pytest_commands(_steps(workflow_text)):
        for word in words:
            if word.startswith("tests/") and word.endswith(".py"):
                listed.add(word[len("tests/"):])
    return listed


def _mlx_test_files() -> list[Path]:
    return sorted(TESTS.glob("test_mlx_*.py"))


def _needing_the_real_library() -> list[str]:
    return [
        path.name
        for path in _mlx_test_files()
        if _needs_real_mlx(path.read_text(encoding="utf-8"))
    ]


class TestEveryRealMlxTestFileRunsInTheJob:
    def test_each_one_is_listed_or_exempted_with_a_reason(self) -> None:
        listed = _listed_files(CI.read_text(encoding="utf-8"))
        missing = [
            name for name in _needing_the_real_library()
            if name not in listed and name not in EXEMPT
        ]
        assert not missing, (
            f"{missing} gate tests on the real MLX library, which only the `{JOB}` job installs, "
            f"and no step of that job runs them: they execute on no CI job. Add each to the "
            f"`pytest` line of the step 'Run the MLX tests that need the real library' in "
            f".github/workflows/ci.yml, or add it to EXEMPT in {Path(__file__).name} with the "
            f"reason it should not run there."
        )

    def test_a_listed_file_exists(self) -> None:
        gone = sorted(
            name for name in _listed_files(CI.read_text(encoding="utf-8"))
            if not (TESTS / name).is_file()
        )
        assert not gone, f"the {JOB} job names test files that do not exist: {gone}"

    def test_the_job_overrides_addopts_in_every_pytest_command(self) -> None:
        """`addopts` carries `--cov`, which the job's bare `pytest` install cannot parse."""
        commands = _pytest_commands(_steps(CI.read_text(encoding="utf-8")))
        assert commands, "the job has no pytest step"
        for words in commands:
            assert "addopts=" in words, words


class TestEveryExemptionIsStillTrueAndExplained:
    @pytest.mark.parametrize("name", sorted(EXEMPT))
    def test_the_file_exists_needs_the_library_and_is_not_listed(self, name: str) -> None:
        path = TESTS / name
        assert path.is_file(), f"{name} is exempted but no longer exists: drop the entry"
        assert name.startswith("test_mlx_"), name
        assert _needs_real_mlx(path.read_text(encoding="utf-8")), (
            f"{name} no longer gates a test on the real MLX library: drop the entry"
        )
        assert name not in _listed_files(CI.read_text(encoding="utf-8")), (
            f"{name} is both listed in the {JOB} job and exempted: drop the entry"
        )

    @pytest.mark.parametrize("name", sorted(EXEMPT))
    def test_the_reason_is_a_sentence(self, name: str) -> None:
        assert len(EXEMPT[name].split()) >= 12, EXEMPT[name]


class TestTheScanIsNotVacuous:
    """Controls: the contract above would pass if the scan saw nothing, so show what it sees."""

    def test_it_finds_the_files_the_job_was_missing_and_the_ones_it_had(self) -> None:
        found = set(_needing_the_real_library())
        assert {
            "test_mlx_dpo.py", "test_mlx_orpo.py", "test_mlx_eval_steps.py",
            "test_mlx_dora_native.py",
        } <= found, sorted(found)

    def test_it_ignores_a_file_that_only_fakes_mlx(self) -> None:
        """These install stand-ins in `sys.modules` and run on every runner."""
        found = set(_needing_the_real_library())
        assert "test_mlx_backend.py" not in found
        assert "test_mlx_dispatch.py" not in found

    @pytest.mark.parametrize(
        "source",
        [
            'import pytest\npytest.importorskip("mlx.core")\n',
            'import pytest\npytest.importorskip("mlx_lm")\n',
            'import pytest\npytest.importorskip("mlx_lm.tuner.trainer")\n',
            'import pytest\npytest.importorskip("mlx.optimizers")\n',
            'import pytest\npytest.importorskip("mlx")\n',
            'import pytest\ndef test_a():\n    mx = pytest.importorskip("mlx.core")\n',
            'import pytest\nclass T:\n    def t(self):\n        pytest.importorskip("mlx.nn")\n',
            'import pytest\n@pytest.fixture\ndef native():\n    pytest.importorskip("mlx_lm")\n',
            'from pytest import importorskip\nimportorskip("mlx.core")\n',
            'import pytest\npytest.importorskip(modname="mlx.core")\n',
            'import pytest\npytest.importorskip(\n    "mlx.core",\n    reason="x",\n)\n',
        ],
    )
    def test_it_sees_every_spelling_of_the_gate(self, source: str) -> None:
        assert _needs_real_mlx(source), source

    @pytest.mark.parametrize(
        "source",
        [
            'import pytest\npytest.importorskip("trl")\n',
            'import pytest\npytest.importorskip("numpy")\n',
            'import pytest\npytest.importorskip("mlxfoo")\n',
            'import pytest\npytest.importorskip(name)\n',
            'import pytest\npytest.importorskip()\n',
            'import sys\nsys.modules["mlx.core"] = object()\n',
            'text = "pytest.importorskip(\'mlx.core\')"\n',
            "# pytest.importorskip('mlx.core')\n",
            'import mlx.core as mx\n',
        ],
    )
    def test_it_does_not_count_what_is_not_a_gate(self, source: str) -> None:
        assert not _needs_real_mlx(source), source


class TestTheListParser:
    WORKFLOW = """
jobs:
  mlx-smoke:
    steps:
      - name: Verify
        run: |
          python -c "import mlx.core; print('tests/not_a_test_file.py')"
      - name: One
        run: pytest -o addopts= -m smoke -k mlx_sft_smoke -q tests/test_smoke_train.py
      - name: Two
        run: pytest -o addopts= -q tests/test_a.py tests/test_b.py
      - name: Three
        run: |
          pytest -o addopts= -q tests/test_c.py
          echo done
  other:
    steps:
      - run: pytest tests/test_elsewhere.py
"""

    def test_it_reads_every_pytest_line_of_this_job_only(self) -> None:
        assert _listed_files(self.WORKFLOW) == {
            "test_smoke_train.py", "test_a.py", "test_b.py", "test_c.py",
        }

    def test_it_sees_the_real_job_names(self) -> None:
        listed = _listed_files(CI.read_text(encoding="utf-8"))
        assert {
            "test_rewind_mlx.py", "test_mlx_dora_native.py", "test_mlx_dpo.py",
            "test_mlx_orpo.py", "test_mlx_eval_steps.py",
        } <= listed, sorted(listed)
