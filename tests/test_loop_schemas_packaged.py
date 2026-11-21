"""Every schema and example of the loop formats ships in the wheel.

``soup_cli.utils.loop_formats`` loads ``schemas/loop/*.schema.json`` (and the examples) from
the installed package. If the build configuration missed them, a tree checkout would pass
every test and a ``pip install`` would fail on the first ``soup loop`` write. The build is
hatchling with ``packages = ["src/soup_cli"]``: the package directory is walked whole, so a
file ships unless something excludes it (``exclude``, ``only-include``, a git-ignore rule).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "src" / "soup_cli" / "schemas" / "loop"


def _shipped_files() -> list:
    return sorted(
        path.relative_to(ROOT / "src").as_posix()
        for path in SCHEMA_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestTheBuildConfigurationIncludesThem:
    def test_the_directory_holds_six_schemas_and_six_examples(self):
        names = [Path(name).name for name in _shipped_files()]

        assert sum(name.endswith(".schema.json") for name in names) == 6
        assert sum(name.endswith(".jsonl") for name in names) == 6
        assert "__init__.py" in names  # a package, so importlib.resources can open it

    def test_the_wheel_walks_the_whole_package_directory(self):
        wheel = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]

        assert "src/soup_cli" in wheel["packages"]

    def test_nothing_in_the_build_configuration_excludes_them(self):
        build = _pyproject()["tool"]["hatch"]["build"]
        targets = build.get("targets", {})

        for section in (build, targets.get("wheel", {}), targets.get("sdist", {})):
            assert "exclude" not in section, section
            assert "only-include" not in section, section

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
    def test_no_git_ignore_rule_hides_them_from_the_build(self):
        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        names = _shipped_files()

        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--", *names],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

        assert result.returncode == 1, f"ignored by git, so hatchling skips them: {result.stdout}"

    def test_the_package_can_open_every_file_through_importlib_resources(self):
        from importlib import resources

        root = resources.files("soup_cli.schemas.loop")
        for name in _shipped_files():
            relative = name.removeprefix("soup_cli/schemas/loop/")
            assert root.joinpath(relative).is_file(), relative


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("hatchling") is None,
    reason="hatchling is not installed here (the build backend); the tests above cover the "
    "configuration it reads",
)
def test_a_built_wheel_contains_every_file(tmp_path):
    out = tmp_path / "wheel"

    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=600,
    )

    assert result.returncode == 0, result.stderr[-800:]
    (wheel,) = list(out.glob("soup_cli-*.whl"))
    in_wheel = set(zipfile.ZipFile(wheel).namelist())
    missing = [name for name in _shipped_files() if name not in in_wheel]
    assert missing == []
