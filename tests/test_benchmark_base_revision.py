"""Benchmark consumers reuse the adapter's recorded base identity (#1902)."""

import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

SHA = "a" * 40
BASE = "example/base"


@pytest.fixture
def benchmark_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": BASE}), encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "run_id": "run-test",
        "base_model": BASE,
        "base_revision": SHA,
        "base_revision_source": "hub_commit",
        "training_data_sha256": None,
    }
    path = adapter / "soup_run.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"mmlu": {"acc,none": 1.0}}}

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=evaluate))
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    return adapter, path, metadata, calls


def invoke(adapter):
    return CliRunner().invoke(
        app, ["eval", "benchmark", "--model", str(adapter), "--device", "cpu"]
    )


def test_matching_record_pins_harness_and_reports_commit(benchmark_case):
    adapter, _path, _metadata, calls = benchmark_case
    result = invoke(adapter)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls[0]["model"] == "hf"
    assert calls[0]["model_args"] == f"pretrained={BASE},peft={adapter},revision={SHA}"
    output = " ".join(strip_ansi(result.output).split())
    assert f"Base revision: {SHA} (recorded run)" in output


@pytest.mark.parametrize(
    "case", ["legacy", "mismatch", "override", "local", "invalid", "equals", "null"]
)
def test_unknown_identity_keeps_unpinned_load(benchmark_case, case):
    adapter, path, metadata, calls = benchmark_case
    selected_base = BASE
    if case == "legacy":
        path.unlink()
    else:
        if case == "mismatch":
            metadata["base_model"] = "another/base"
        elif case == "override":
            selected_base = "override/base"
        elif case == "local":
            selected_base = str(adapter.parent / "local")
            (adapter.parent / "local").mkdir()
            metadata["base_model"] = selected_base
        elif case == "invalid":
            metadata["base_revision"] = SHA + ",trust_remote_code=True"
        elif case == "equals":
            metadata["base_revision"] = SHA + "=bad"
        elif case == "null":
            metadata["base_revision"] = None
            metadata["base_revision_source"] = "unknown"
        path.write_text(json.dumps(metadata), encoding="utf-8")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": selected_base}), encoding="utf-8"
    )
    result = invoke(adapter)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls[0]["model_args"] == f"pretrained={selected_base},peft={adapter}"
    output = " ".join(strip_ansi(result.output).split())
    assert "Base revision: unknown (legacy, local source or base override)" in output


@pytest.mark.parametrize("base", ["bad,trust_remote_code=True", "bad=base"])
def test_base_injection_is_refused_before_harness(benchmark_case, base):
    adapter, _path, _metadata, calls = benchmark_case
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base}), encoding="utf-8"
    )
    result = invoke(adapter)
    assert result.exit_code == 1
    assert not calls


def test_unavailable_recorded_commit_propagates_without_retry(benchmark_case, monkeypatch):
    adapter, _path, _metadata, calls = benchmark_case

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise OSError("recorded commit is unavailable")

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=unavailable))
    result = invoke(adapter)
    assert result.exit_code != 0
    assert isinstance(result.exception, OSError)
    assert len(calls) == 1
    assert f"revision={SHA}" in calls[0]["model_args"]


@pytest.mark.parametrize(
    ("adapter_revision", "expected_revision"),
    [("a" * 40, SHA), ("b" * 40, None)],
)
def test_adapter_revision_must_match_recorded_commit(
    benchmark_case, adapter_revision, expected_revision
):
    adapter, _path, _metadata, calls = benchmark_case
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": BASE,
                "revision": adapter_revision,
            }
        ),
        encoding="utf-8",
    )

    result = invoke(adapter)
    assert result.exit_code == 0, (result.output, repr(result.exception))

    expected_args = f"pretrained={BASE},peft={adapter}"
    if expected_revision:
        expected_args += f",revision={expected_revision}"

    assert calls[0]["model_args"] == expected_args
    output = " ".join(strip_ansi(result.output).split())
    if expected_revision:
        assert f"Base revision: {SHA} (recorded run)" in output
    else:
        assert "Base revision: unknown (legacy, local source or base override)" in output
