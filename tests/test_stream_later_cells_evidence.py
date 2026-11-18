"""The later #2162 record must not claim a full audit without its raw archive."""

import importlib.util
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).parents[1]
HELPER = ROOT / "benchmarks/harness/issue2162_later_cells.py"


def verifier():
    spec = importlib.util.spec_from_file_location("later_cells", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_git_numbers_are_recomputed_without_claiming_a_full_verdict():
    result = verifier().verify(None)
    assert result["mode"] == "numbers-only"
    assert result["snapshots_verified"] == 14
    assert result["verified"] is False
    assert result["numbers_verified"] is True
    assert (
        result["numbers"]["rtx5090"]["comparisons"]["mixed_vs_native"]["median_reduction_percent"]
        == 8.166451049472066
    )
    assert (
        result["numbers"]["rtx3090"]["confirmation"]["median_reduction_percent"]
        == 6.968552971461994
    )
    assert result["unavailable_checks"]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "a\\b"])
def test_archive_member_cannot_escape_extraction_root(tmp_path, name):
    with pytest.raises(ValueError):
        verifier().destination(tmp_path, name)


def test_present_wrong_bundle_is_rejected_instead_of_degraded(tmp_path):
    wrong = tmp_path / "wrong.zip"
    wrong.write_bytes(b"not the recorded archive")
    with pytest.raises(ValueError, match="bundle differs"):
        verifier().verify(wrong)


def test_bundle_inventory_matches_manifest_when_supplied():
    bundle = os.environ.get("SOUP2162_LATER_CELLS")
    if not bundle:
        pytest.skip("set SOUP2162_LATER_CELLS to the verified archive")
    module = verifier()
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    with ZipFile(bundle) as archive:
        module.bundle_checks(archive, manifest)
    result = module.verify(Path(bundle))
    assert result["verified"] is True
    assert result["rtx5090"]["stage_a_passes"] is False
    assert result["rtx3090"]["stage_a_passes"] is True
    assert result["rtx3090"]["confirmation_passes"] is True
