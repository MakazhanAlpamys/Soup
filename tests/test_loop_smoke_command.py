"""Every place that says how to run the loop's real-model smoke test says ``-m smoke`` (N3).

(Review 5b.) ``python -m pytest tests/test_loop_end_to_end_smoke.py -q --no-cov -p no:cacheprovider
-x`` exits 5 with ``1 deselected``: ``pyproject.toml`` has ``addopts = "-m 'not smoke' ..."``, and
the smoke file carries the ``smoke`` marker. A review of the loop ran that command, got a
green-looking nothing, and had to rediscover ``-m smoke``. These tests pin the cause (the default
deselects it, ``-m smoke`` selects it) and every place that documents the command.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = "tests/test_loop_end_to_end_smoke.py"
NAME = "test_loop_end_to_end_smoke"


def _collect(*extra: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", SMOKE, "--collect-only", "-q", "--no-cov",
         "-p", "no:cacheprovider", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=240, check=False,
    )
    return result.stdout + result.stderr


class TestWhyTheMarkerIsNeeded:
    def test_the_repo_default_deselects_the_smoke_file(self):
        out = _collect()

        assert "1 deselected" in out and "test_the_loop_on_a_real_model" not in out

    def test_dash_m_smoke_selects_it(self):
        out = _collect("-m", "smoke")

        assert "test_the_loop_on_a_real_model" in out
        assert "deselected" not in out

    def test_the_default_and_the_marker_are_what_this_file_assumes(self):
        config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        smoke = (ROOT / SMOKE).read_text(encoding="utf-8")

        assert "-m 'not smoke'" in config
        assert "pytest.mark.smoke" in smoke


class TestEveryPlaceThatNamesTheFileSaysHowToRunIt:
    def _places(self):
        files = sorted((ROOT / "docs").glob("*.md")) + sorted(ROOT.glob("tests/test_loop_end*.py"))
        return [path for path in files if NAME in path.read_text(encoding="utf-8")
                and path.name != Path(__file__).name]

    def test_there_are_places_to_check(self):
        names = {path.name for path in self._places()}

        assert {"adapters-and-governance.md", "test_loop_end_to_end.py"} <= names
        assert "test_loop_end_to_end_smoke.py" in names  # its own docstring

    def test_each_one_says_dash_m_smoke(self):
        offenders = [
            path.name for path in self._places()
            if not re.search(r"-m smoke", " ".join(path.read_text(encoding="utf-8").split()))
        ]

        assert offenders == [], f"{offenders} name the smoke test without saying -m smoke"

    def test_the_smoke_docstring_gives_the_whole_command_and_the_exit_code_without_it(self):
        text = " ".join((ROOT / SMOKE).read_text(encoding="utf-8").split())

        assert (
            "python -m pytest -m smoke tests/test_loop_end_to_end_smoke.py -q --no-cov "
            "-p no:cacheprovider"
        ) in text
        assert "exits 5 with ``1 deselected``" in text

    def test_the_docs_say_what_happens_without_it(self):
        text = " ".join(
            (ROOT / "docs" / "adapters-and-governance.md").read_text(encoding="utf-8").split()
        )

        assert "exits 5 with `1 deselected`" in text
        assert "python -m pytest -m smoke tests/test_loop_end_to_end_smoke.py" in text
