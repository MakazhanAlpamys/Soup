"""Older release lines keep version tags without replacing the newest Docker image."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "docker.yml"
SCRIPT = ROOT / "scripts" / "docker_release_tags.py"


def _steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["build-and-push-image"]["steps"]


def _metadata() -> dict:
    return next(step["with"] for step in _steps() if step.get("id") == "meta")


def test_semver_cannot_implicitly_restore_latest() -> None:
    assert "latest=false" in _metadata().get("flavor", "").splitlines(), (
        "metadata-action automatically adds latest for semver unless explicitly disabled"
    )


def test_latest_tag_is_gated_without_gating_version_tags() -> None:
    tags = _metadata()["tags"].splitlines()
    assert "latest" not in tags, "an unconditional latest rolls the image back on an older release"
    assert "type=semver,pattern={{version}}" in tags
    assert "type=semver,pattern={{major}}.{{minor}}" in tags
    assert "type=raw,value=latest,enable=${{ steps.latest.outputs.latest == 'true' }}" in tags


def test_checkout_fetches_all_release_tags() -> None:
    checkout = next(step for step in _steps() if "actions/checkout@" in step.get("uses", ""))
    assert checkout.get("with", {}).get("fetch-depth") == 0


def test_latest_decision_uses_the_release_ref_before_registry_login() -> None:
    steps = _steps()
    latest_index = next(index for index, step in enumerate(steps) if step.get("id") == "latest")
    version_index = next(index for index, step in enumerate(steps) if step.get("id") == "version")
    login_index = next(
        index for index, step in enumerate(steps) if "docker/login-action@" in step.get("uses", "")
    )
    assert version_index < latest_index < login_index
    latest = steps[latest_index]
    assert latest["env"]["REF_NAME"] == "${{ github.ref_name }}"
    assert latest["run"].strip() == 'python scripts/docker_release_tags.py "${REF_NAME}"'
    assert "${{" not in latest["run"]


@pytest.fixture
def isolated_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient repository settings from redirecting the test's Git commands."""
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def tagged_repo(tmp_path: Path, isolated_git_environment: None) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Soup tests",
            "-c",
            "user.email=tests@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "seed",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


@pytest.mark.parametrize(
    ("tags", "release", "expected"),
    [
        (["v0.77.0", "v0.76.2"], "v0.76.2", "false"),
        (["v0.77.0", "v0.76.2"], "v0.77.0", "true"),
        (["v0.9.0", "v0.10.0"], "v0.9.0", "false"),
        (["v0.9.0", "v0.10.0"], "v0.10.0", "true"),
        (["v0.77.9", "v0.77.10"], "v0.77.9", "false"),
        (["v0.77.9", "v0.77.10"], "v0.77.10", "true"),
        (["v9.0.0", "v10.0.0"], "v10.0.0", "true"),
        (["v0.77.0", "v1.0.0-rc.1", "v2.0.0.1", "nightly"], "v0.77.0", "true"),
        (["v0.77.0"], "v0.76.2", "false"),
        (["v1.0.0-rc.1", "nightly"], "v1.0.0-rc.1", "false"),
        ([], "v0.77.0", "false"),
    ],
    ids=[
        "older-line",
        "newest-line",
        "minor-lexical-trap",
        "numeric-minor",
        "patch-lexical-trap",
        "numeric-patch",
        "numeric-major",
        "ignore-non-release-tags",
        "release-not-fetched",
        "no-stable-tags",
        "no-tags",
    ],
)
def test_script_decides_from_real_git_tags(
    tagged_repo: Path,
    tags: list[str],
    release: str,
    expected: str,
) -> None:
    for tag in tags:
        subprocess.run(["git", "tag", tag], cwd=tagged_repo, check=True)
    output = tagged_repo / "step-output"
    output.write_text("existing=preserved\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), release],
        cwd=tagged_repo,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == f"existing=preserved\nlatest={expected}\n"


def test_failed_git_lookup_does_not_enable_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_git_environment: None
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    output = tmp_path / "step-output"
    output.write_text("existing=preserved\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "v0.77.0"],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "existing=preserved\n"
