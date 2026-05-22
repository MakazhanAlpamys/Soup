"""v0.66.0 CLI integration tests.

Verify the `soup probe` group is registered + all four sub-commands work
end-to-end, plus the `soup adapters blame --live` runner is wired.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture(autouse=True)
def _chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# soup probe --help / pack / sleeper / interference / sae-diff
# ---------------------------------------------------------------------------


def test_probe_group_registered():
    result = runner.invoke(soup_app, ["probe", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = _strip_ansi(result.output)
    assert "pack" in out
    assert "sleeper" in out
    assert "interference" in out
    assert "sae-diff" in out


def test_probe_pack_list():
    result = runner.invoke(soup_app, ["probe", "pack", "--list"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Should list at least Llama
    assert "Llama" in _strip_ansi(result.output)


def test_probe_pack_unknown_base():
    result = runner.invoke(soup_app, ["probe", "pack", "unknown/model"])
    assert result.exit_code == 2
    assert "no probe pack" in _strip_ansi(result.output)


def test_probe_pack_happy(tmp_path):
    out_path = tmp_path / "pack.json"
    result = runner.invoke(soup_app, [
        "probe", "pack", "meta-llama/Llama-3-8B",
        "--output", str(out_path.name),
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["base"] == "meta-llama/Llama-3-8B"
    assert len(payload["probes"]) >= 1


def test_probe_pack_no_args_exits_2():
    result = runner.invoke(soup_app, ["probe", "pack"])
    assert result.exit_code == 2
    assert "list" in _strip_ansi(result.output)


def test_probe_sleeper_no_evidence_renders_metadata():
    result = runner.invoke(
        soup_app, ["probe", "sleeper", "meta-llama/Llama-3-8B"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = _strip_ansi(result.output)
    assert "Sleeper probe" in out


def test_probe_sleeper_unknown_base_exits_2():
    result = runner.invoke(soup_app, ["probe", "sleeper", "unknown/model"])
    assert result.exit_code == 2


def test_probe_sleeper_with_evidence(tmp_path):
    # Build evidence: 50 tokens × 4096-dim Llama-3 activations
    activations = np.random.RandomState(0).randn(50, 4096).astype(np.float32)
    evidence_path = tmp_path / "ev.json"
    evidence_path.write_text(
        json.dumps({"activations": activations.tolist()}), encoding="utf-8"
    )
    result = runner.invoke(soup_app, [
        "probe", "sleeper", "meta-llama/Llama-3-8B",
        "--evidence", evidence_path.name,
    ])
    # Returns 0 unless MAJOR; deterministic synthetic probe + random activations
    # gives a low rate, so likely OK or MINOR
    out = _strip_ansi(result.output)
    assert "Sleeper probe" in out
    assert result.exit_code in (0, 2)  # 2 only if MAJOR


def test_probe_interference_happy(tmp_path):
    payload = {
        "adapters": ["a", "b", "c"],
        "losses": {
            "a|a": 1.0, "b|b": 1.0, "c|c": 1.0,
            "a|b": 1.05, "a|c": 1.10,
            "b|a": 1.02, "b|c": 1.03,
            "c|a": 1.04, "c|b": 1.01,
        },
    }
    p = tmp_path / "losses.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(soup_app, ["probe", "interference", p.name])
    # All small (< 20%) -> exit 0
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "interference matrix" in _strip_ansi(result.output).lower()


def test_probe_interference_major_exits_2(tmp_path):
    payload = {
        "adapters": ["a", "b"],
        "losses": {
            "a|a": 1.0, "b|b": 1.0,
            "a|b": 2.0,  # 100% increase -> MAJOR
            "b|a": 1.0,
        },
    }
    p = tmp_path / "losses.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(soup_app, ["probe", "interference", p.name])
    assert result.exit_code == 2
    assert "MAJOR" in _strip_ansi(result.output)


def test_probe_interference_bad_key_format(tmp_path):
    payload = {
        "adapters": ["a", "b"],
        "losses": {"a_a": 1.0},  # no pipe separator
    }
    p = tmp_path / "losses.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(soup_app, ["probe", "interference", p.name])
    assert result.exit_code == 2


def test_probe_interference_missing_evidence_file(tmp_path):
    result = runner.invoke(soup_app, ["probe", "interference", "nonexistent.json"])
    assert result.exit_code == 2


def test_probe_sae_diff_help():
    result = runner.invoke(soup_app, ["probe", "sae-diff", "--help"])
    assert result.exit_code == 0
    assert "--top-k" in _strip_ansi(result.output)


def test_probe_sae_diff_missing_sae(tmp_path):
    pre = tmp_path / "pre.json"
    pre.write_text(json.dumps({"activations": [[1.0]]}), encoding="utf-8")
    post = tmp_path / "post.json"
    post.write_text(json.dumps({"activations": [[1.5]]}), encoding="utf-8")
    result = runner.invoke(soup_app, [
        "probe", "sae-diff", "nonexistent.safetensors", pre.name, post.name,
    ])
    assert result.exit_code in (1, 2)


# ---------------------------------------------------------------------------
# soup adapters blame --live (Part B closes #171)
# ---------------------------------------------------------------------------


def test_adapters_blame_live_runs_default_probe(tmp_path):
    """v0.66.0 lifted the v0.57 stub — without --plan-only the runner actually runs."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "tiny"}', encoding="utf-8"
    )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join('{"text": "row %d"}' % i for i in range(20)),
        encoding="utf-8",
    )
    result = runner.invoke(soup_app, [
        "adapters", "blame", adapter.name,
        "--dataset", dataset.name,
        "--layer", "q_proj.7",
        "--budget", "1h",
        "--shards", "5",
        "--top-k", "5",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = _strip_ansi(result.output)
    # No deferred advisory
    assert "v0.57.1" not in out
    # Live blame output
    assert "Blame" in out


def test_adapters_blame_top_k_flag_present():
    result = runner.invoke(soup_app, ["adapters", "blame", "--help"])
    assert result.exit_code == 0
    assert "--top-k" in _strip_ansi(result.output)
