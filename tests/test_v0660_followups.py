"""v0.66.0 review-fix follow-up tests.

This file collects the TDD-wave regression guards for every fix landed in
the 3 review waves (python-review / code+security / tdd-guide). Keeping
them in one file makes the v0.66.x patches easier to find later.
"""
from __future__ import annotations

import inspect
import os
import sys
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# H1 fix: TypeError on bool/non-string verdict
# ---------------------------------------------------------------------------


def test_sleeper_verdict_bool_raises_type_error():
    from soup_cli.utils.sleeper_probe import SleeperProbeResult

    with pytest.raises(TypeError, match="verdict must be str"):
        SleeperProbeResult(
            base="b", num_tokens=10, defection_rate=0.0,
            max_score=0.0, verdict=True,
        )


def test_sleeper_verdict_non_string_raises_type_error():
    from soup_cli.utils.sleeper_probe import SleeperProbeResult

    with pytest.raises(TypeError, match="verdict must be str"):
        SleeperProbeResult(
            base="b", num_tokens=10, defection_rate=0.0,
            max_score=0.0, verdict=42,
        )


def test_interference_cell_verdict_bool_raises_type_error():
    from soup_cli.utils.interference import InterferenceCell

    with pytest.raises(TypeError, match="verdict must be str"):
        InterferenceCell(adapter_a="a", adapter_b="b", score=0.0, verdict=True)


# ---------------------------------------------------------------------------
# Wave 1 H1: FrozenInstanceError on mutation (not generic Exception)
# ---------------------------------------------------------------------------


def test_sae_feature_change_raises_frozen_instance_error():
    from soup_cli.utils.sae_diff import SaeFeatureChange

    c = SaeFeatureChange(feature_id=0, delta=0.0, pre_mean=0.0, post_mean=0.0)
    with pytest.raises(FrozenInstanceError):
        c.delta = 1.0  # type: ignore[misc]
    # Original value preserved (defence-in-depth)
    assert c.delta == 0.0


def test_sae_diff_report_raises_frozen_instance_error():
    from soup_cli.utils.sae_diff import SaeFeatureDiffReport

    r = SaeFeatureDiffReport(
        num_features=0, num_tokens=0, l2_drift=0.0, changes=tuple()
    )
    with pytest.raises(FrozenInstanceError):
        r.num_features = 99  # type: ignore[misc]


def test_row_influence_raises_frozen_instance_error():
    from soup_cli.utils.blame import RowInfluence

    r = RowInfluence(row_id=0, score=0.0, shard_id=0)
    with pytest.raises(FrozenInstanceError):
        r.score = 1.0  # type: ignore[misc]


def test_blame_result_raises_frozen_instance_error():
    from soup_cli.utils.blame import BlameResult

    r = BlameResult(
        adapter_dir="x", dataset_path="y", layer="l",
        top_influencers=tuple(), num_rows_scored=0, elapsed_seconds=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        r.num_rows_scored = 5  # type: ignore[misc]


def test_sleeper_probe_spec_raises_frozen_instance_error():
    from soup_cli.utils.sleeper_probe import SleeperProbeSpec

    s = SleeperProbeSpec(base="b", hidden_dim=4, threshold=0.5, description="d")
    with pytest.raises(FrozenInstanceError):
        s.threshold = 1.0  # type: ignore[misc]


def test_sleeper_probe_result_raises_frozen_instance_error():
    from soup_cli.utils.sleeper_probe import SleeperProbeResult

    r = SleeperProbeResult(
        base="b", num_tokens=0, defection_rate=0.0, max_score=0.0, verdict="OK"
    )
    with pytest.raises(FrozenInstanceError):
        r.verdict = "MAJOR"  # type: ignore[misc]


def test_interference_cell_raises_frozen_instance_error():
    from soup_cli.utils.interference import InterferenceCell

    c = InterferenceCell(adapter_a="a", adapter_b="b", score=0.0, verdict="OK")
    with pytest.raises(FrozenInstanceError):
        c.score = 1.0  # type: ignore[misc]


def test_interference_matrix_raises_frozen_instance_error():
    from soup_cli.utils.interference import InterferenceMatrix

    m = InterferenceMatrix(
        adapters=("a", "b"), cells=tuple(), worst_pair=None, worst_score=0.0
    )
    with pytest.raises(FrozenInstanceError):
        m.worst_score = 1.0  # type: ignore[misc]


def test_probe_entry_raises_frozen_instance_error():
    from soup_cli.utils.probe_pack import ProbeEntry

    e = ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d")
    with pytest.raises(FrozenInstanceError):
        e.kind = "sae"  # type: ignore[misc]


def test_probe_pack_raises_frozen_instance_error():
    from soup_cli.utils.probe_pack import ProbeEntry, ProbePack

    p = ProbePack(
        base="b",
        probes=(
            ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d"),
        ),
        soup_version="0.66.0",
    )
    with pytest.raises(FrozenInstanceError):
        p.base = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# H2 fix: Rich-markup escape in render_*_markdown
# ---------------------------------------------------------------------------


def test_interference_render_markdown_escapes_adapter_name():
    """A crafted adapter name with Rich markup must be escaped."""
    from soup_cli.utils.interference import (
        InterferenceCell,
        InterferenceMatrix,
        render_matrix_markdown,
    )

    # Adapter names are validated (no null bytes, no oversize) but
    # `[`/`]` characters ARE permitted, so the render must escape them.
    crafted = "[link=evil]X[/]"
    cell = InterferenceCell(
        adapter_a=crafted, adapter_b="b", score=0.0, verdict="OK"
    )
    m = InterferenceMatrix(
        adapters=(crafted, "b"),
        cells=(cell,),
        worst_pair=(crafted, "b"),
        worst_score=0.10,
    )
    text = render_matrix_markdown(m)
    # Escaped form present
    assert "\\[link=evil\\]X\\[/\\]" in text
    # Raw `[link=evil]X[/]` substring must NOT appear unescaped in a way
    # Rich would interpret (verify by counting escape backslashes)
    assert text.count("\\[link=evil") >= 1


def test_blame_render_markdown_escapes_layer():
    """The `layer` field is rendered verbatim (no basename) so it's the
    canonical Rich-markup-injection vector to test on this surface."""
    from soup_cli.utils.blame import BlameResult, render_blame_markdown

    r = BlameResult(
        adapter_dir="adp",
        dataset_path="data.jsonl",
        layer="[red]lm_head[/]",
        top_influencers=tuple(),
        num_rows_scored=0,
        elapsed_seconds=0.0,
    )
    text = render_blame_markdown(r)
    # Layer is escaped
    assert "\\[red\\]lm_head\\[/\\]" in text


def test_probe_pack_render_markdown_escapes_description():
    from soup_cli.utils.probe_pack import (
        ProbeEntry,
        ProbePack,
        render_pack_markdown,
    )

    pack = ProbePack(
        base="meta-llama/Llama-3-8B",
        probes=(
            ProbeEntry(
                name="sleeper-1",
                kind="sleeper",
                hidden_dim=4096,
                description="[link=evil]click[/]",
            ),
        ),
        soup_version="0.66.0",
    )
    text = render_pack_markdown(pack)
    assert "\\[link=evil" in text


# ---------------------------------------------------------------------------
# H1 (security): TOCTOU O_NOFOLLOW symlink rejection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink test")
def test_count_dataset_rows_rejects_symlink_via_o_nofollow(tmp_path, monkeypatch):
    """O_NOFOLLOW path open rejects symlink targets (TOCTOU defence)."""
    from soup_cli.utils.blame import plan_blame

    monkeypatch.chdir(tmp_path)
    # Create a real adapter dir
    adapter = tmp_path / "adp"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{}', encoding="utf-8")
    # Real dataset
    real = tmp_path / "real.jsonl"
    real.write_text('{"text":"row"}\n', encoding="utf-8")
    # Symlinked dataset
    sym = tmp_path / "data.jsonl"
    os.symlink(str(real), str(sym))

    with pytest.raises(ValueError, match="symlink"):
        plan_blame(
            adapter.name, sym.name,
            layer="x", budget_seconds=600, num_shards=2,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink test")
def test_load_sae_weights_rejects_symlink_via_o_nofollow(tmp_path, monkeypatch):
    from soup_cli.utils.sae_diff import load_sae_weights

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.safetensors"
    real.write_bytes(b"x")
    sym = tmp_path / "sae.safetensors"
    os.symlink(str(real), str(sym))
    with pytest.raises(ValueError, match="symlink"):
        load_sae_weights("sae.safetensors")


# ---------------------------------------------------------------------------
# M4 fix: _count_dataset_rows raises on >10M rows (no silent truncate)
# ---------------------------------------------------------------------------


def test_count_dataset_rows_raises_on_oversize(tmp_path, monkeypatch):
    """Synthetic dataset with >10M rows must raise (not silently truncate)."""
    from soup_cli.utils import blame

    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adp"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{}', encoding="utf-8")
    dataset = tmp_path / "huge.jsonl"
    dataset.write_text("a\nb\n", encoding="utf-8")

    # Patch os.fdopen to return an iterator that yields > 10M rows so we
    # don't need a 10M-line fixture on disk
    real_fdopen = os.fdopen

    class _Fake:
        def __iter__(self):
            for _ in range(10_000_002):
                yield b"x\n"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_fdopen(fd, mode):
        try:
            os.close(fd)
        except OSError:
            pass
        return _Fake()

    monkeypatch.setattr(os, "fdopen", _fake_fdopen)
    try:
        with pytest.raises(ValueError, match="10M"):
            blame._count_dataset_rows(str(dataset))
    finally:
        monkeypatch.setattr(os, "fdopen", real_fdopen)


# ---------------------------------------------------------------------------
# M2 fix: 64-bit seed (16 hex chars)
# ---------------------------------------------------------------------------


def test_sleeper_probe_seed_is_64_bit():
    """Source-grep regression: _probe_weights uses digest[:16] (64 bits)."""
    from soup_cli.utils import sleeper_probe

    source = inspect.getsource(sleeper_probe._probe_weights)
    assert "digest[:16]" in source, "M2 fix: seed must be 16 hex chars"


def test_blame_synthetic_probe_seed_is_64_bit():
    from soup_cli.utils import blame

    source = inspect.getsource(blame._default_synthetic_probe)
    assert "[:16]" in source, "M2 fix: blame seed must be 16 hex chars"


# ---------------------------------------------------------------------------
# M5 fix: probe_pack description length cap
# ---------------------------------------------------------------------------


def test_probe_entry_oversize_description_rejected():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(ValueError, match="4096"):
        ProbeEntry(
            name="x", kind="sleeper", hidden_dim=4,
            description="a" * 4097,
        )


def test_probe_entry_at_max_description_accepted():
    """Boundary test — exactly 4096 chars must accept."""
    from soup_cli.utils.probe_pack import ProbeEntry

    e = ProbeEntry(
        name="x", kind="sleeper", hidden_dim=4,
        description="a" * 4096,
    )
    assert len(e.description) == 4096


def test_probe_entry_null_byte_description_rejected():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(ValueError, match="null"):
        ProbeEntry(
            name="x", kind="sleeper", hidden_dim=4, description="a\x00b"
        )


# ---------------------------------------------------------------------------
# M6 fix: synthetic-probe row count capped at 100k
# ---------------------------------------------------------------------------


def test_default_synthetic_probe_caps_at_100k(tmp_path, monkeypatch):
    """Without probe_fn, runner caps at _DEFAULT_SYNTH_PROBE_CAP (100k)."""
    from soup_cli.utils.blame import (
        BlamePlan,
        BlameShardWork,
        run_blame,
    )

    monkeypatch.chdir(tmp_path)
    # Build a plan with 1M holdout — synthetic probe must cap to 100k
    plan = BlamePlan(
        adapter_dir="a", dataset_path="d", layer="x",
        budget_seconds=3600, num_shards=2, per_shard_seconds=1800,
        shards=(
            BlameShardWork(0, 0, 500_000, 1800),
            BlameShardWork(1, 500_000, 500_000, 1800),
        ),
        feasible=True, reason="ok",
    )
    result = run_blame(plan)  # no probe_fn -> synthetic
    assert result.num_rows_scored <= 100_000


# ---------------------------------------------------------------------------
# L3 fix: _LOWER_INDEX MappingProxyType immutability
# ---------------------------------------------------------------------------


def test_sleeper_lower_index_immutable():
    from soup_cli.utils.sleeper_probe import _LOWER_INDEX

    assert isinstance(_LOWER_INDEX, MappingProxyType)
    with pytest.raises(TypeError):
        _LOWER_INDEX["x"] = "y"  # type: ignore[index]


# ---------------------------------------------------------------------------
# H3 fix: probe interference CLI rejects non-numeric loss
# ---------------------------------------------------------------------------


def test_probe_interference_rejects_string_loss(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from soup_cli.cli import app as soup_app

    runner_ = CliRunner()
    monkeypatch.chdir(tmp_path)
    payload = {
        "adapters": ["a", "b"],
        "losses": {"a|a": "not-a-number", "b|b": 1.0},
    }
    p = tmp_path / "losses.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    result = runner_.invoke(soup_app, ["probe", "interference", p.name])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "must be numeric" in result.output


def test_probe_interference_rejects_bool_loss(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from soup_cli.cli import app as soup_app

    runner_ = CliRunner()
    monkeypatch.chdir(tmp_path)
    # Note: JSON booleans deserialise to Python bool, which is a subclass
    # of int — without the H3 guard, this would silently pass
    payload = {
        "adapters": ["a", "b"],
        "losses": {"a|a": True, "b|b": 1.0},
    }
    p = tmp_path / "losses.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    result = runner_.invoke(soup_app, ["probe", "interference", p.name])
    assert result.exit_code == 2, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# Boundary tests at exact thresholds
# ---------------------------------------------------------------------------


def test_classify_sleeper_score_boundary_minus_epsilon_ok():
    from soup_cli.utils.sleeper_probe import classify_sleeper_score

    # Just below 1%
    assert classify_sleeper_score(0.01 - 1e-9) == "OK"


def test_classify_sleeper_score_boundary_minus_epsilon_minor():
    from soup_cli.utils.sleeper_probe import classify_sleeper_score

    # Just below 5%
    assert classify_sleeper_score(0.05 - 1e-9) == "MINOR"


def test_classify_interference_boundary_minus_epsilon_ok():
    from soup_cli.utils.interference import classify_interference

    assert classify_interference(0.05 - 1e-9) == "OK"


def test_classify_interference_boundary_minus_epsilon_minor():
    from soup_cli.utils.interference import classify_interference

    assert classify_interference(0.20 - 1e-9) == "MINOR"


# ---------------------------------------------------------------------------
# Source-grep regression: project rules
# ---------------------------------------------------------------------------


def test_no_path_resolve_used_in_v0_66_modules():
    """Project rule: use os.path.realpath, not Path.resolve()."""
    from soup_cli.utils import blame, interference, probe_pack, sae_diff, sleeper_probe

    for mod in (sae_diff, blame, sleeper_probe, interference, probe_pack):
        src = inspect.getsource(mod)
        assert ".resolve()" not in src, (
            f"{mod.__name__}: use os.path.realpath, not Path.resolve()"
        )


def test_no_top_level_torch_in_v0_66_modules():
    """Heavy deps must be lazy-imported inside function bodies."""
    from soup_cli.utils import blame, interference, probe_pack, sae_diff, sleeper_probe

    for mod in (sae_diff, blame, sleeper_probe, interference, probe_pack):
        src = inspect.getsource(mod)
        top_level_imports = [
            line for line in src.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        for line in top_level_imports:
            for bad in ("torch", "transformers", "peft", "safetensors"):
                assert bad not in line, (
                    f"{mod.__name__}: top-level {bad} import: {line!r}"
                )


# ---------------------------------------------------------------------------
# Top-K rejection — Part A
# ---------------------------------------------------------------------------


def test_compute_feature_diff_rejects_negative_top_k():
    from soup_cli.utils.sae_diff import compute_feature_diff

    pre = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="≥1"):
        compute_feature_diff(pre, pre, top_k=-1)
