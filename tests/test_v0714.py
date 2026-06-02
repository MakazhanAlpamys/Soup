"""v0.71.4 — Adapter lifecycle + loop wiring.

Closes #172, #173, #176, #177, #220, #223.

- #223  soup adapters pr --push  (GitHub PR publisher via `gh api`)
- #173  branch pointers into the v0.26 Registry lineage DAG
- #172  live canary verdict for soup adapters merge (replace UNKNOWN stub)
- #220  live eval-suite auto-wiring for soup adapters merge --strategy cmaes
- #176  pre-wired stage callbacks for soup loop watch
- #177  pack each soup loop iteration as a Soup Can + Registry lineage
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest
from typer.testing import CliRunner

runner = CliRunner()

POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink rejection is POSIX-only here"
)


@pytest.fixture()
def temp_registry(tmp_path, monkeypatch):
    """Isolated registry + branches dir under cwd, returns a push helper."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "registry.db"
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db))
    branches = tmp_path / "branches"
    branches.mkdir()
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(branches))

    from soup_cli.registry.store import RegistryStore

    def _push(name="mymodel", tag="v1", base="meta-llama/x", task="dpo",
              config=None, data_path=None):
        with RegistryStore() as store:
            return store.push(
                name=name, tag=tag, base_model=base, task=task,
                run_id="run-1", config=config or {"base": base, "task": task},
                data_path=data_path,
            )

    return _push


# ===========================================================================
# #173 — branch pointers into the v0.26 Registry lineage DAG
# ===========================================================================


def _write_cfg(tmp_path, name="soup.yaml", body="base: meta-llama/x\ntask: dpo\n"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


class TestBranchRefKind:
    def test_branch_ref_in_valid_kinds(self):
        from soup_cli.registry.store import _VALID_KINDS

        assert "branch_ref" in _VALID_KINDS

    def test_attach_artifact_branch_ref(self, temp_registry, tmp_path):
        from soup_cli.registry.attach import attach_artifact

        entry_id = temp_registry()
        ptr = tmp_path / "ptr.json"
        ptr.write_text('{"k": "v"}', encoding="utf-8")
        rowid = attach_artifact(
            entry_id, path=str(ptr), kind="branch_ref", enforce_cwd=False
        )
        assert isinstance(rowid, int)


class TestBranchSchema:
    def test_registry_entry_id_field_default_none(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import create_branch, load_branch

        cfg = _write_cfg(tmp_path)
        snap = create_branch("b1", config_path=cfg, base_model="meta-llama/x")
        assert snap.registry_entry_id is None
        reloaded = load_branch("b1")
        assert reloaded.registry_entry_id is None

    def test_create_with_registry_entry_id_roundtrips(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import create_branch, load_branch

        cfg = _write_cfg(tmp_path)
        create_branch(
            "b2", config_path=cfg, base_model="meta-llama/x",
            registry_entry_id="reg_20260601_abc123",
        )
        assert load_branch("b2").registry_entry_id == "reg_20260601_abc123"

    def test_create_with_dataset_sha256_direct(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import create_branch, load_branch

        cfg = _write_cfg(tmp_path)
        sha = "c" * 64
        create_branch(
            "b3", config_path=cfg, base_model="meta-llama/x", dataset_sha256=sha,
        )
        assert load_branch("b3").dataset_sha256 == sha

    def test_dataset_path_and_sha_mutually_exclusive(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import create_branch

        cfg = _write_cfg(tmp_path)
        ds = tmp_path / "data.jsonl"
        ds.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            create_branch(
                "b4", config_path=cfg, base_model="meta-llama/x",
                dataset_path=str(ds), dataset_sha256="d" * 64,
            )

    def test_registry_entry_id_null_byte_rejected(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import create_branch

        cfg = _write_cfg(tmp_path)
        with pytest.raises(ValueError):
            create_branch(
                "b5", config_path=cfg, base_model="meta-llama/x",
                registry_entry_id="reg\x00bad",
            )

    def test_back_compat_v057_pointer(self, temp_registry, tmp_path):
        """A v0.57.0 pointer with no registry_entry_id key loads cleanly."""
        from soup_cli.utils.adapter_branch import _branches_dir, load_branch

        legacy = {
            "name": "legacy",
            "config_path": str(tmp_path / "soup.yaml"),
            "config_sha256": "a" * 64,
            "dataset_sha256": None,
            "base_model": "meta-llama/x",
            "created_at": 1.0,
            "soup_version": "0.57.0",
        }
        (_branches_dir() / "legacy.json").write_text(
            json.dumps(legacy), encoding="utf-8"
        )
        snap = load_branch("legacy")
        assert snap.registry_entry_id is None
        assert snap.base_model == "meta-llama/x"


class TestAttachBranchToRegistry:
    def test_happy(self, temp_registry, tmp_path):
        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.adapter_branch import (
            attach_branch_to_registry,
            create_branch,
            load_branch,
        )

        entry_id = temp_registry()
        cfg = _write_cfg(tmp_path)
        create_branch("attached", config_path=cfg, base_model="meta-llama/x")
        rowid = attach_branch_to_registry("attached", entry_id)
        assert isinstance(rowid, int)
        assert load_branch("attached").registry_entry_id == entry_id
        with RegistryStore() as store:
            arts = store.get_artifacts(entry_id)
        kinds = {a["kind"] for a in arts}
        assert "branch_ref" in kinds

    def test_missing_entry_friendly(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import (
            attach_branch_to_registry,
            create_branch,
        )

        cfg = _write_cfg(tmp_path)
        create_branch("x", config_path=cfg, base_model="meta-llama/x")
        with pytest.raises(ValueError, match="not found"):
            attach_branch_to_registry("x", "no-such-entry")


class TestBranchFromRegistry:
    def test_derives_all_fields(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import branch_from_registry

        entry_id = temp_registry(base="org/big-model")
        snap = branch_from_registry("derived", entry_id)
        assert snap.base_model == "org/big-model"
        assert snap.registry_entry_id == entry_id
        assert os.path.isfile(snap.config_path)
        assert len(snap.config_sha256) == 64

    def test_with_data_hash(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import branch_from_registry

        ds = tmp_path / "data.jsonl"
        ds.write_text('{"x": 1}\n', encoding="utf-8")
        entry_id = temp_registry(data_path=str(ds))
        snap = branch_from_registry("derived2", entry_id)
        assert snap.dataset_sha256 is not None
        assert len(snap.dataset_sha256) == 64

    def test_missing_entry(self, temp_registry, tmp_path):
        from soup_cli.utils.adapter_branch import branch_from_registry

        with pytest.raises(ValueError, match="not found"):
            branch_from_registry("nope", "no-such-id")


class TestBranchCli:
    def _app(self):
        from soup_cli.commands.adapters import app

        return app

    def test_attach_to_registry_cli(self, temp_registry, tmp_path):
        entry_id = temp_registry()
        cfg = _write_cfg(tmp_path)
        result = runner.invoke(
            self._app(),
            [
                "branch", "clibranch",
                "-c", cfg, "--base", "meta-llama/x",
                "--attach-to-registry", entry_id,
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        from soup_cli.utils.adapter_branch import load_branch

        assert load_branch("clibranch").registry_entry_id == entry_id

    def test_from_registry_cli(self, temp_registry, tmp_path):
        entry_id = temp_registry(base="org/from-reg")
        result = runner.invoke(
            self._app(),
            ["branch", "fromreg", "--from-registry", entry_id],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        from soup_cli.utils.adapter_branch import load_branch

        snap = load_branch("fromreg")
        assert snap.base_model == "org/from-reg"
        assert snap.registry_entry_id == entry_id

    def test_from_registry_missing_friendly(self, temp_registry, tmp_path):
        result = runner.invoke(
            self._app(),
            ["branch", "fr2", "--from-registry", "nope"],
        )
        assert result.exit_code == 2
        assert "not found" in result.output.lower()

    def test_requires_config_when_not_from_registry(self, temp_registry, tmp_path):
        result = runner.invoke(
            self._app(),
            ["branch", "needsconfig", "--base", "meta-llama/x"],
        )
        assert result.exit_code == 2
        assert "config" in result.output.lower()


class TestHistoryBranchEdges:
    def test_history_renders_branch_edge(self, temp_registry, tmp_path):
        import typer

        from soup_cli.commands.history import history as history_cmd
        from soup_cli.utils.adapter_branch import (
            attach_branch_to_registry,
            create_branch,
        )

        entry_id = temp_registry(name="histmodel")
        cfg = _write_cfg(tmp_path)
        create_branch("hb", config_path=cfg, base_model="meta-llama/x")
        attach_branch_to_registry("hb", entry_id)

        app = typer.Typer()
        app.command()(history_cmd)
        result = runner.invoke(app, ["histmodel"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "branch" in result.output.lower()


# ===========================================================================
# #172 — live canary verdict for soup adapters merge (replace UNKNOWN stub)
# ===========================================================================


def _make_adapter(dir_path, *, scale=1.0):
    """Write a minimal LoRA-shaped adapter (safetensors + config)."""
    import numpy as np
    from safetensors.numpy import save_file

    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
            (rng.standard_normal((8, 16)) * scale).astype("float32"),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight":
            (rng.standard_normal((16, 8)) * scale).astype("float32"),
    }
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(
        json.dumps({
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "base_model_name_or_path": "hf-internal-testing/tiny-random-gpt2",
        }),
        encoding="utf-8",
    )
    return str(dir_path)


def _make_report():
    from soup_cli.utils.adapter_merge import MergeReport

    return MergeReport(
        strategy="linear", adapters=("a", "b"), weights=(0.5, 0.5),
        merged_layers=1, skipped_layers=(), output_dir="out", verdict="UNKNOWN",
    )


class TestPredictMergedVerdict:
    def test_none_returns_unknown(self):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        assert predict_merged_verdict(_make_report()) == "UNKNOWN"

    def test_no_drop_is_ok(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": [0.9, 0.8, 0.85, 0.9],
            "candidate_scores": [0.9, 0.81, 0.86, 0.9],
        }), encoding="utf-8")
        assert predict_merged_verdict(_make_report(), str(suite)) == "OK"

    def test_minor_drop(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        # ~3% drop on average → MINOR
        suite.write_text(json.dumps({
            "baseline_scores": [1.0, 1.0, 1.0, 1.0],
            "candidate_scores": [0.97, 0.97, 0.97, 0.97],
        }), encoding="utf-8")
        assert predict_merged_verdict(_make_report(), str(suite)) == "MINOR"

    def test_major_drop(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": [1.0, 1.0, 1.0, 1.0],
            "candidate_scores": [0.85, 0.85, 0.85, 0.85],
        }), encoding="utf-8")
        assert predict_merged_verdict(_make_report(), str(suite)) == "MAJOR"

    def test_length_mismatch(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": [1.0, 1.0],
            "candidate_scores": [1.0],
        }), encoding="utf-8")
        with pytest.raises(ValueError):
            predict_merged_verdict(_make_report(), str(suite))

    def test_empty_scores(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": [], "candidate_scores": [],
        }), encoding="utf-8")
        with pytest.raises(ValueError):
            predict_merged_verdict(_make_report(), str(suite))

    def test_non_numeric_score_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(
            '{"baseline_scores": [1.0], "candidate_scores": ["x"]}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            predict_merged_verdict(_make_report(), str(suite))

    def test_infinity_literal_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        # `json.loads` parses the non-standard `Infinity` / `NaN` literals by
        # default, so the finite-check in _require_score_list is load-bearing.
        suite.write_text(
            '{"baseline_scores": [1.0], "candidate_scores": [Infinity]}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="finite"):
            predict_merged_verdict(_make_report(), str(suite))

    def test_tasks_without_scorer(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "tasks": [{"prompt": "2+2", "expected": "4"}],
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="scorer"):
            predict_merged_verdict(_make_report(), str(suite))

    def test_tasks_with_scorer(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "tasks": [{"prompt": "p", "expected": "e"}, {"prompt": "p2", "expected": "e2"}],
        }), encoding="utf-8")

        def scorer(role, tasks):
            assert role in ("baseline", "candidate")
            # 3% drop → MINOR (5% is the MAJOR boundary).
            return [1.0 for _ in tasks] if role == "baseline" else [0.97 for _ in tasks]

        assert predict_merged_verdict(
            _make_report(), str(suite), scorer=scorer
        ) == "MINOR"

    def test_canary_must_be_str(self):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        with pytest.raises(TypeError):
            predict_merged_verdict(_make_report(), 123)  # type: ignore[arg-type]


class TestMergeCanaryCli:
    def _app(self):
        from soup_cli.commands.adapters import app

        return app

    def test_merge_with_canary_renders_verdict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": [0.9, 0.9], "candidate_scores": [0.9, 0.9],
        }), encoding="utf-8")
        result = runner.invoke(self._app(), [
            "merge", a, b, "-o", str(tmp_path / "out"),
            "--allow-unscanned", "--canary", str(suite),
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "OK" in result.output

    def test_strict_verdict_major_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": [1.0, 1.0], "candidate_scores": [0.8, 0.8],
        }), encoding="utf-8")
        result = runner.invoke(self._app(), [
            "merge", a, b, "-o", str(tmp_path / "out"),
            "--allow-unscanned", "--canary", str(suite), "--strict-verdict",
        ])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "MAJOR" in result.output

    def test_missing_canary_unknown_advisory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        result = runner.invoke(self._app(), [
            "merge", a, b, "-o", str(tmp_path / "out"), "--allow-unscanned",
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "UNKNOWN" in result.output

    def test_canary_outside_cwd_rejected(self, tmp_path, monkeypatch):
        sub = tmp_path / "work"
        sub.mkdir()
        monkeypatch.chdir(sub)
        a = _make_adapter(sub / "a")
        b = _make_adapter(sub / "b")
        outside = tmp_path / "canary.json"
        outside.write_text("{}", encoding="utf-8")
        result = runner.invoke(self._app(), [
            "merge", a, b, "-o", str(sub / "out"),
            "--allow-unscanned", "--canary", str(outside),
        ])
        assert result.exit_code == 2
        assert "cwd" in result.output.lower()


# ===========================================================================
# #220 — live eval-suite auto-wiring for soup adapters merge --strategy cmaes
# ===========================================================================


def _cmaes_plan(tmp_path, adapters, *, pop=4, gens=6):
    from soup_cli.utils.cmaes_merge import build_cmaes_plan

    suite = tmp_path / "eval.jsonl"
    suite.write_text('{"prompt": "p", "expected": "e"}\n', encoding="utf-8")
    return build_cmaes_plan(
        adapters=adapters, eval_suite=str(suite), budget_spec="60s",
        population_size=pop, max_generations=gens,
    )


class TestBuildCmaesEvalFn:
    def test_eval_fn_merges_and_scores(self, tmp_path, monkeypatch):
        from soup_cli.utils.cmaes_merge import build_cmaes_eval_fn

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a", scale=1.0)
        b = _make_adapter(tmp_path / "b", scale=2.0)
        plan = _cmaes_plan(tmp_path, [a, b])

        seen = {}

        def scorer(merged_dir, eval_suite):
            seen["dir"] = merged_dir
            seen["suite"] = eval_suite
            # The merged dir must contain a materialised safetensors file.
            assert os.path.isfile(os.path.join(merged_dir, "adapter_model.safetensors"))
            return 0.42

        fn = build_cmaes_eval_fn(plan, scorer=scorer)
        score = fn((0.5, 0.5))
        assert score == 0.42
        assert seen["suite"] == plan.eval_suite

    def test_eval_fn_cleans_up_temp(self, tmp_path, monkeypatch):
        from soup_cli.utils.cmaes_merge import build_cmaes_eval_fn

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        plan = _cmaes_plan(tmp_path, [a, b])
        captured = []

        def scorer(merged_dir, eval_suite):
            captured.append(merged_dir)
            return 0.5

        fn = build_cmaes_eval_fn(plan, scorer=scorer)
        fn((0.5, 0.5))
        # The per-generation temp dir is removed after scoring.
        assert not os.path.exists(captured[0])

    def test_run_cmaes_with_eval_fn_converges(self, tmp_path, monkeypatch):
        from soup_cli.utils.cmaes_merge import build_cmaes_eval_fn, run_cmaes_merge

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        plan = _cmaes_plan(tmp_path, [a, b], pop=4, gens=20)

        # Constant scorer → plateau → converged after 3 flat generations.
        fn = build_cmaes_eval_fn(plan, scorer=lambda d, s: 0.7)
        result = run_cmaes_merge(plan, eval_fn=fn)
        assert result.converged is True
        assert result.generations_run < plan.max_generations
        assert abs(sum(result.best_weights) - 1.0) < 1e-6

    def test_eval_fn_isolates_scorer_failure(self, tmp_path, monkeypatch):
        from soup_cli.utils.cmaes_merge import (
            _FAILED_EVAL_SENTINEL,
            _eval_safely,
            build_cmaes_eval_fn,
        )

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        plan = _cmaes_plan(tmp_path, [a, b])

        def boom(merged_dir, eval_suite):
            raise RuntimeError("scorer crashed")

        fn = build_cmaes_eval_fn(plan, scorer=boom)
        # The eval_fn itself surfaces the error; run_cmaes_merge wraps in
        # _eval_safely → sentinel. Confirm the isolation path.
        assert _eval_safely(fn, (0.5, 0.5)) == _FAILED_EVAL_SENTINEL


class TestCmaesCliLive:
    def _app(self):
        from soup_cli.commands.adapters import app

        return app

    def test_cmaes_runs_loop_and_writes_output(self, tmp_path, monkeypatch):
        import soup_cli.utils.cmaes_merge as cm

        monkeypatch.chdir(tmp_path)
        # Inject a synthetic scorer so the CLI never loads a model.
        monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", lambda d, s: 0.6)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        suite = tmp_path / "eval.jsonl"
        suite.write_text('{"prompt": "p", "expected": "e"}\n', encoding="utf-8")
        out = tmp_path / "merged"
        result = runner.invoke(self._app(), [
            "merge", a, b, "--strategy", "cmaes",
            "--eval", str(suite), "--budget", "60s",
            "--population", "4", "--max-generations", "6",
            "-o", str(out),
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # The best merge was materialised to --output.
        assert (out / "adapter_model.safetensors").is_file()
        # No more plan-only "deferred to v0.67.1" advisory.
        assert "deferred to" not in result.output.lower()

    def test_cmaes_still_requires_eval(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Real adapters + --allow-unscanned so the input gates pass and we
        # reach the cmaes arg-check (the scan gate now runs before dispatch).
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        result = runner.invoke(self._app(), [
            "merge", a, b, "--strategy", "cmaes",
            "-o", str(tmp_path / "out"), "--allow-unscanned",
        ])
        assert result.exit_code == 2
        assert "eval" in result.output.lower()


# ===========================================================================
# #176 — pre-wired stage callbacks for soup loop watch
# ===========================================================================


@pytest.fixture()
def loop_env(tmp_path, monkeypatch):
    """Isolated cwd + registry DB for loop tests; returns a state factory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.delenv("SOUP_LOOP_TRACE_DIR", raising=False)
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)

    from soup_cli.utils.loop_state import LoopState

    def _state(served="mymodel", eval_suite="evals/gate.yaml", baseline="base-ref",
               pre_wired=False):
        return LoopState(
            served_model=served, eval_suite=eval_suite, baseline=baseline,
            status="running", pre_wired=pre_wired,
        )

    return _state


def _write_trace_dir(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    rows = [
        {"id": "1", "prompt": "Q", "response": "good", "feedback": {"rating": "up"}},
        {"id": "2", "prompt": "Q", "response": "bad", "feedback": {"rating": "down"}},
    ]
    (trace_dir / "serve.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return str(trace_dir)


class TestLoopStatePreWired:
    def test_default_false(self, loop_env):
        assert loop_env().pre_wired is False

    def test_non_bool_rejected(self):
        from soup_cli.utils.loop_state import LoopState

        with pytest.raises(ValueError):
            LoopState(
                served_model="m", eval_suite="e", baseline="b",
                pre_wired="yes",  # type: ignore[arg-type]
            )

    def test_init_state_persists(self, loop_env, tmp_path):
        from soup_cli.utils.loop_state import init_state, read_state

        state, _path = init_state("mymodel", "evals/gate.yaml", "base-ref", pre_wired=True)
        assert state.pre_wired is True
        assert read_state().pre_wired is True

    def test_to_dict_includes_pre_wired(self, loop_env):
        from soup_cli.utils.loop_state import LoopState

        keys = set(loop_env().to_dict().keys())
        assert keys == set(LoopState.__dataclass_fields__.keys())
        assert "pre_wired" in keys


class TestHarvestStage:
    def test_produces_pairs(self, loop_env, tmp_path, monkeypatch):
        from soup_cli.utils.loop_stages import harvest_from_traces

        td = _write_trace_dir(tmp_path)
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", td)
        out = harvest_from_traces(loop_env())
        assert out["pairs_harvested"] == 1
        assert out["pairs_path"] is not None
        assert os.path.isfile(out["pairs_path"])

    def test_no_trace_dir_zero(self, loop_env):
        from soup_cli.utils.loop_stages import harvest_from_traces

        out = harvest_from_traces(loop_env())
        assert out["pairs_harvested"] == 0
        assert out["pairs_path"] is None

    def test_traces_present_but_no_pairs(self, loop_env, tmp_path, monkeypatch):
        """The common steady state: traces exist but no down-vote → no pairs.

        Distinct from the no-dir case: traces_collected > 0 but
        pairs_harvested == 0 and pairs_path is None (no file written)."""
        from soup_cli.utils.loop_stages import harvest_from_traces

        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        # All thumbs-up, no down → build_pairs(signal="thumbs_up") yields none.
        rows = [
            {"id": "1", "prompt": "Q", "response": "a",
             "feedback": {"rating": "up"}},
            {"id": "2", "prompt": "Q2", "response": "b",
             "feedback": {"rating": "up"}},
        ]
        (trace_dir / "s.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", str(trace_dir))
        out = harvest_from_traces(loop_env())
        assert out["pairs_harvested"] == 0
        assert out["pairs_path"] is None
        assert out["traces_collected"] >= 1


class TestTrainStage:
    def test_skips_without_pairs(self, loop_env):
        from soup_cli.utils.loop_stages import train_dpo_from_pairs

        out = train_dpo_from_pairs(
            loop_env(), {"pairs_harvested": 0, "pairs_path": None}
        )
        assert out["skipped"] is True
        assert out["run_id"] is None

    def test_invokes_runner(self, loop_env, tmp_path, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        captured = {}

        class _R:
            returncode = 0

        def fake_runner(argv, **kwargs):
            captured["argv"] = argv
            return _R()

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", fake_runner)
        pairs = tmp_path / "p.jsonl"
        pairs.write_text('{"prompt":"q","chosen":"a","rejected":"b"}\n', encoding="utf-8")
        out = ls.train_dpo_from_pairs(
            loop_env(), {"pairs_harvested": 1, "pairs_path": str(pairs)}
        )
        assert out["skipped"] is False
        assert out["run_id"].startswith("loop-train-")
        argv = captured["argv"]
        assert "train" in argv and "--yes" in argv

    def test_runner_failure_skips(self, loop_env, tmp_path, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        class _R:
            returncode = 1

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", lambda *a, **k: _R())
        pairs = tmp_path / "p.jsonl"
        pairs.write_text('{"prompt":"q","chosen":"a","rejected":"b"}\n', encoding="utf-8")
        out = ls.train_dpo_from_pairs(
            loop_env(), {"pairs_harvested": 1, "pairs_path": str(pairs)}
        )
        assert out["skipped"] is True


class TestGateStage:
    def test_skipped_when_train_skipped(self, loop_env):
        from soup_cli.utils.loop_stages import gate_against_baseline

        out = gate_against_baseline(loop_env(), {"skipped": True})
        assert out["gate_verdict"] == "SKIPPED"

    def test_skipped_when_adapter_missing(self, loop_env):
        from soup_cli.utils.loop_stages import gate_against_baseline

        out = gate_against_baseline(
            loop_env(), {"skipped": False, "adapter_path": "nope/dir"}
        )
        assert out["gate_verdict"] == "SKIPPED"

    def test_ok_verdict(self, loop_env, tmp_path, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        # Real eval suite + tasks; injected generate factory returns expected.
        evals = tmp_path / "evals"
        evals.mkdir()
        tasks = evals / "tasks.jsonl"
        tasks.write_text(
            '{"prompt": "2+2", "expected": "4", "scoring": "exact"}\n',
            encoding="utf-8",
        )
        suite = evals / "gate.yaml"
        suite.write_text(
            "suite: loop\n"
            "tasks:\n"
            "  - type: custom\n"
            "    name: math\n"
            "    threshold: 0.5\n"
            f"    tasks: {tasks.as_posix()}\n"
            "    scorer: exact\n",
            encoding="utf-8",
        )
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        monkeypatch.setattr(
            ls, "_GATE_GENERATE_FACTORY", lambda d: (lambda prompt: "4")
        )
        state = loop_env(eval_suite=str(suite))
        out = ls.gate_against_baseline(
            state, {"skipped": False, "adapter_path": str(adapter)}
        )
        assert out["gate_verdict"] == "OK"


class TestDeployStage:
    def test_noop_when_not_ok(self, loop_env):
        from soup_cli.utils.loop_stages import deploy_to_canary

        out = deploy_to_canary(loop_env(), {"gate_verdict": "MAJOR"})
        assert out["deployed"] is False
        assert out["canary_verdict"] is None

    def test_posts_when_ok(self, loop_env, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        seen = {}

        def poster(endpoint, name):
            seen["endpoint"] = endpoint
            seen["name"] = name
            return True

        monkeypatch.setattr(ls, "_DEPLOY_POSTER", poster)
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://localhost:8000")
        out = ls.deploy_to_canary(
            loop_env(), {"gate_verdict": "OK", "adapter_path": "adapters/run-1"}
        )
        assert out["deployed"] is True
        assert out["canary_verdict"] == "OK"
        assert seen["name"] == "run-1"


class TestPrewiredConfig:
    def test_build_returns_wired_config(self):
        from soup_cli.utils.loop_stages import (
            build_prewired_watch_config,
            deploy_to_canary,
            gate_against_baseline,
            harvest_from_traces,
            train_dpo_from_pairs,
        )

        cfg = build_prewired_watch_config(max_iterations=1)
        assert cfg.harvest_fn is harvest_from_traces
        assert cfg.train_fn is train_dpo_from_pairs
        assert cfg.gate_fn is gate_against_baseline
        assert cfg.deploy_fn is deploy_to_canary


class TestPrewiredE2E:
    def test_watch_harvests_and_trains(self, loop_env, tmp_path, monkeypatch):
        import soup_cli.utils.loop_stages as ls
        from soup_cli.utils.loop_iteration import list_iterations, read_iteration
        from soup_cli.utils.loop_state import write_state

        td = _write_trace_dir(tmp_path)
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", td)

        class _R:
            returncode = 0

        train_calls = []
        monkeypatch.setattr(
            ls, "_TRAIN_RUNNER", lambda argv, **k: train_calls.append(argv) or _R()
        )
        write_state(loop_env(pre_wired=True))
        from soup_cli.utils.loop_stages import build_prewired_watch_config

        cfg = build_prewired_watch_config(
            max_iterations=1, poll_interval_sec=1.0,
        )
        from soup_cli.utils.loop_daemon import watch

        _final, ran = watch(cfg)
        assert ran == 1
        # The DPO subprocess was invoked.
        assert len(train_calls) == 1
        # The iteration recorded the harvested pairs.
        ids = list_iterations()
        assert len(ids) == 1
        rec = read_iteration(ids[0])
        assert rec.pairs_harvested == 1


class TestLoopStagesSourceWiring:
    def test_no_top_level_heavy_imports(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (
            root / "src" / "soup_cli" / "utils" / "loop_stages.py"
        ).read_text(encoding="utf-8")
        head = "\n".join(
            line for line in src.splitlines()[:60]
            if line.strip() and not line.strip().startswith("#")
        )
        for forbidden in (
            "import torch", "import transformers", "import peft", "import trl",
        ):
            assert forbidden not in head, f"top-level {forbidden!r} in loop_stages"


class TestLoopCliPrewired:
    def _app(self):
        from soup_cli.commands.loop import app

        return app

    def test_init_pre_wired_flag(self, loop_env, tmp_path):
        result = runner.invoke(self._app(), [
            "init", "mymodel", "--eval", "evals/gate.yaml",
            "--baseline", "base-ref", "--pre-wired",
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        from soup_cli.utils.loop_state import read_state

        assert read_state().pre_wired is True

    def test_watch_help_lists_pack_cans(self):
        result = runner.invoke(self._app(), ["watch", "--help"])
        assert result.exit_code == 0
        out = result.stdout.replace("\n", " ")
        assert "--pack-cans" in out
        assert "--pre-wired" in out

    def test_status_shows_pre_wired(self, loop_env):
        runner.invoke(self._app(), [
            "init", "mymodel", "--eval", "evals/gate.yaml",
            "--baseline", "base-ref", "--pre-wired",
        ])
        result = runner.invoke(self._app(), ["status"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = result.stdout.replace("\n", " ")
        assert "pre_wired" in out
        assert "yes" in out


# ===========================================================================
# #177 — pack each loop iteration as a v0.26 Soup Can + Registry lineage
# ===========================================================================


def _write_iteration(iteration_id, *, served="mymodel"):
    from soup_cli.utils.loop_iteration import IterationRecord, write_iteration

    rec = IterationRecord(
        iteration_id=iteration_id,
        started_at="2026-06-01T00:00:00+00:00",
        finished_at="2026-06-01T00:01:00+00:00",
        pairs_harvested=3,
        run_id="run-x",
        gate_verdict="OK",
        canary_verdict=None,
        shipped=True,
        rolled_back=False,
        estimated_cost_usd=0.0,
        notes="",
    )
    write_iteration(rec)
    return rec


class TestRegistryNameFrom:
    @pytest.mark.parametrize("raw,expected_start", [
        ("mymodel", "mymodel"),
        ("registry://abc12", "abc12"),
        ("org/big-model", "org-big-model"),
        ("//weird", "weird"),
    ])
    def test_sanitises(self, raw, expected_start):
        from soup_cli.utils.loop_iteration import registry_name_from

        out = registry_name_from(raw)
        assert out[0].isalnum()
        assert out.startswith(expected_start) or out == "loop"

    def test_empty_falls_back(self):
        from soup_cli.utils.loop_iteration import registry_name_from

        assert registry_name_from("") == "loop"
        assert registry_name_from("///") == "loop"


class TestPackIterationAsCan:
    def test_writes_can_and_entry(self, loop_env, tmp_path):
        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.loop_iteration import pack_iteration_as_can

        _write_iteration("iter-aaa")
        can_path, entry_id = pack_iteration_as_can(
            "iter-aaa", served_model="mymodel"
        )
        assert os.path.isfile(can_path)
        assert can_path.endswith("iteration.can")
        with RegistryStore() as store:
            entry = store.get(entry_id)
        assert entry is not None
        assert entry["name"] == "mymodel"

    def test_can_verifies(self, loop_env, tmp_path):
        from soup_cli.cans.unpack import inspect_can
        from soup_cli.utils.loop_iteration import pack_iteration_as_can

        _write_iteration("iter-bbb")
        can_path, _ = pack_iteration_as_can("iter-bbb", served_model="mymodel")
        manifest = inspect_can(can_path)
        assert manifest.name == "mymodel"

    def test_parent_lineage(self, loop_env, tmp_path):
        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.loop_iteration import pack_iteration_as_can

        _write_iteration("iter-p1")
        _write_iteration("iter-p2")
        _can1, parent_id = pack_iteration_as_can("iter-p1", served_model="mymodel")
        _can2, child_id = pack_iteration_as_can(
            "iter-p2", served_model="mymodel", parent_registry_id=parent_id
        )
        with RegistryStore() as store:
            ancestors = store.get_ancestors(child_id)
        assert any(a["id"] == parent_id for a in ancestors)

    def test_missing_parent_still_creates(self, loop_env, tmp_path):
        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.loop_iteration import pack_iteration_as_can

        _write_iteration("iter-mp")
        _can, entry_id = pack_iteration_as_can(
            "iter-mp", served_model="mymodel", parent_registry_id="no-such-parent"
        )
        with RegistryStore() as store:
            assert store.get(entry_id) is not None


class TestWatchPackCans:
    def test_three_iterations_chain(self, loop_env, tmp_path, monkeypatch):
        import soup_cli.utils.loop_stages as ls
        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.loop_daemon import watch
        from soup_cli.utils.loop_state import write_state

        td = _write_trace_dir(tmp_path)
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", td)

        class _R:
            returncode = 0

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", lambda *a, **k: _R())
        write_state(loop_env(pre_wired=True))
        from soup_cli.utils.loop_stages import build_prewired_watch_config

        cfg = build_prewired_watch_config(
            max_iterations=3, poll_interval_sec=1.0,
            pack_iterations=True, served_model="mymodel", base_model="mymodel",
        )
        _final, ran = watch(cfg)
        assert ran == 3
        # 3 registry entries named mymodel, each with a loop-iter tag.
        with RegistryStore() as store:
            entries = store.list_by_name("mymodel")
        assert len(entries) == 3
        # Cans exist + verify.
        import glob

        from soup_cli.cans.unpack import inspect_can
        cans = glob.glob(os.path.join(".soup-loops", "*", "iteration.can"))
        assert len(cans) == 3
        for c in cans:
            inspect_can(c)  # raises if invalid

    def test_replay_extract(self, loop_env, tmp_path):
        from soup_cli.utils.loop_iteration import pack_iteration_as_can

        _write_iteration("iter-ext")
        pack_iteration_as_can("iter-ext", served_model="mymodel")

        from soup_cli.commands.loop import app

        result = runner.invoke(
            app, ["replay", "iter-ext", "--extract", "extracted"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile(os.path.join("extracted", "manifest.yaml"))

    def test_replay_extract_missing_can(self, loop_env, tmp_path):
        _write_iteration("iter-nocan")
        from soup_cli.commands.loop import app

        result = runner.invoke(
            app, ["replay", "iter-nocan", "--extract", "extracted"]
        )
        assert result.exit_code == 1
        assert "can" in result.output.lower()


# ===========================================================================
# #223 — soup adapters pr --push (GitHub PR publisher)
# ===========================================================================


class TestParsePrTarget:
    def test_happy(self):
        from soup_cli.utils.adapter_pr import parse_pr_target

        assert parse_pr_target("MakazhanAlpamys/Soup#42") == (
            "MakazhanAlpamys",
            "Soup",
            42,
        )

    def test_strips_whitespace(self):
        from soup_cli.utils.adapter_pr import parse_pr_target

        assert parse_pr_target("  owner/repo#7  ") == ("owner", "repo", 7)

    @pytest.mark.parametrize(
        "bad",
        [
            "owner/repo",  # no #N
            "owner#42",  # no repo
            "owner/repo#0",  # PR 0
            "owner/repo#-1",
            "owner/repo#abc",
            "/repo#1",
            "owner/#1",
            "owner repo#1",
        ],
    )
    def test_rejects_bad(self, bad):
        from soup_cli.utils.adapter_pr import parse_pr_target

        with pytest.raises(ValueError):
            parse_pr_target(bad)

    def test_non_string(self):
        from soup_cli.utils.adapter_pr import parse_pr_target

        with pytest.raises(TypeError):
            parse_pr_target(42)  # type: ignore[arg-type]

    def test_bool_rejected(self):
        from soup_cli.utils.adapter_pr import parse_pr_target

        with pytest.raises(TypeError):
            parse_pr_target(True)  # type: ignore[arg-type]


class TestResolveGithubToken:
    def test_github_token(self):
        from soup_cli.utils.adapter_pr import resolve_github_token

        assert resolve_github_token({"GITHUB_TOKEN": "ghp_abc"}) == "ghp_abc"

    def test_gh_token_fallback(self):
        from soup_cli.utils.adapter_pr import resolve_github_token

        assert resolve_github_token({"GH_TOKEN": "ghp_xyz"}) == "ghp_xyz"

    def test_strips_whitespace(self):
        from soup_cli.utils.adapter_pr import resolve_github_token

        assert resolve_github_token({"GITHUB_TOKEN": "  tok  "}) == "tok"

    def test_missing_fails_fast(self):
        from soup_cli.utils.adapter_pr import resolve_github_token

        with pytest.raises(RuntimeError, match="token"):
            resolve_github_token({})

    def test_blank_treated_as_missing(self):
        from soup_cli.utils.adapter_pr import resolve_github_token

        with pytest.raises(RuntimeError):
            resolve_github_token({"GITHUB_TOKEN": "   "})


class TestPostPrComment:
    def test_happy_path_invokes_gh(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        captured = {}

        class _Result:
            returncode = 0
            stdout = '{"html_url": "https://github.com/o/r/pull/1#issuecomment-9"}'
            stderr = ""

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["input"] = kwargs.get("input")
            return _Result()

        url = post_pr_comment(
            "o/r#1", "## Hello\nbody", env={"GITHUB_TOKEN": "tok"}, runner=fake_run
        )
        assert url == "https://github.com/o/r/pull/1#issuecomment-9"
        argv = captured["argv"]
        assert argv[0] == "gh"
        assert "api" in argv
        assert "repos/o/r/issues/1/comments" in argv
        assert "--method" in argv and "POST" in argv
        # body posted via JSON stdin (handles multiline safely)
        assert json.loads(captured["input"])["body"] == "## Hello\nbody"

    def test_missing_token_fails_before_subprocess(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        def fake_run(argv, **kwargs):  # pragma: no cover — must not be reached
            raise AssertionError("subprocess should not run without a token")

        with pytest.raises(RuntimeError, match="token"):
            post_pr_comment("o/r#1", "body", env={}, runner=fake_run)

    def test_nonzero_exit_raises(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        class _Result:
            returncode = 1
            stdout = ""
            stderr = "HTTP 404: Not Found"

        with pytest.raises(RuntimeError, match="404"):
            post_pr_comment(
                "o/r#1", "body", env={"GH_TOKEN": "t"}, runner=lambda *a, **k: _Result()
            )

    def test_empty_body_rejected(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        with pytest.raises(ValueError):
            post_pr_comment(
                "o/r#1", "   ", env={"GITHUB_TOKEN": "t"}, runner=lambda *a, **k: None
            )

    def test_null_byte_body_rejected(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        with pytest.raises(ValueError):
            post_pr_comment(
                "o/r#1",
                "a\x00b",
                env={"GITHUB_TOKEN": "t"},
                runner=lambda *a, **k: None,
            )

    def test_bad_target_rejected(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        with pytest.raises(ValueError):
            post_pr_comment(
                "not-a-target", "body", env={"GITHUB_TOKEN": "t"},
                runner=lambda *a, **k: None,
            )


class TestPrCliPush:
    def _adapters_app(self):
        from soup_cli.commands.adapters import app

        return app

    def test_push_flag_in_help(self):
        result = runner.invoke(self._adapters_app(), ["pr", "--help"])
        assert result.exit_code == 0
        out = result.stdout.replace("\n", " ")
        assert "--push" in out

    def test_push_missing_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        sha = "a" * 64
        result = runner.invoke(
            self._adapters_app(),
            [
                "pr",
                "my-title",
                "--base-sha",
                sha,
                "--adapter",
                "adapters/x",
                "--push",
                "owner/repo#42",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "token" in result.output.lower()

    def test_push_happy(self, monkeypatch):
        # Mock the post so we never touch the network.
        import soup_cli.utils.adapter_pr as ap

        monkeypatch.setattr(
            ap, "post_pr_comment",
            lambda target, body, **kw: "https://github.com/o/r/pull/42#c-1",
        )
        sha = "b" * 64
        result = runner.invoke(
            self._adapters_app(),
            [
                "pr",
                "my-title",
                "--base-sha",
                sha,
                "--adapter",
                "adapters/x",
                "--push",
                "owner/repo#42",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "github.com/o/r/pull/42" in result.output


# ===========================================================================
# Review follow-ups (Step 5 — fix every finding CRITICAL→LOW)
# ===========================================================================


class TestVerdictBoundaries:
    """Strict `<` boundaries: drop==0.02 → MINOR, drop==0.05 → MAJOR."""

    def _verdict(self, tmp_path, monkeypatch, baseline, candidate):
        from soup_cli.utils.adapter_merge import predict_merged_verdict

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "canary.json"
        suite.write_text(json.dumps({
            "baseline_scores": baseline, "candidate_scores": candidate,
        }), encoding="utf-8")
        return predict_merged_verdict(_make_report(), str(suite))

    def test_drop_just_under_two_pct_is_ok(self, tmp_path, monkeypatch):
        # 1.9% drop → still OK (boundary is strict `< 0.02`).
        assert self._verdict(
            tmp_path, monkeypatch, [1.0, 1.0], [0.981, 0.981]
        ) == "OK"

    def test_drop_exactly_two_pct_is_minor(self, tmp_path, monkeypatch):
        # Exactly 2.0% drop → MINOR (OK requires drop strictly < 0.02).
        assert self._verdict(
            tmp_path, monkeypatch, [1.0, 1.0], [0.98, 0.98]
        ) == "MINOR"

    def test_drop_just_under_five_pct_is_minor(self, tmp_path, monkeypatch):
        # 4.9% drop → still MINOR.
        assert self._verdict(
            tmp_path, monkeypatch, [1.0, 1.0], [0.951, 0.951]
        ) == "MINOR"

    def test_drop_exactly_five_pct_is_major(self, tmp_path, monkeypatch):
        # Exactly 5.0% drop → MAJOR (MINOR requires drop strictly < 0.05).
        assert self._verdict(
            tmp_path, monkeypatch, [1.0, 1.0], [0.95, 0.95]
        ) == "MAJOR"

    def test_improvement_is_ok(self, tmp_path, monkeypatch):
        # Candidate better than baseline → negative drop → OK.
        assert self._verdict(
            tmp_path, monkeypatch, [0.8, 0.8], [0.9, 0.9]
        ) == "OK"


class TestPackIterationSafely:
    def test_pack_failure_never_crashes_watch(self, loop_env, tmp_path, monkeypatch):
        """A pack failure mid-watch must not kill the daemon; the iteration
        manifest gets a ``pack-failed:`` note and the loop survives."""
        import soup_cli.utils.loop_iteration as li
        from soup_cli.utils.loop_daemon import WatchConfig, watch
        from soup_cli.utils.loop_state import init_state

        init_state("mymodel", "evals/gate.yaml", "base-ref")

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(li, "pack_iteration_as_can", boom)

        cfg = WatchConfig(
            poll_interval_sec=1.0, max_iterations=1, pack_iterations=True,
            served_model="mymodel",
        )
        final_state, n = watch(cfg)
        assert n == 1  # the daemon completed its single iteration
        # The manifest was re-written with the pack-failed note.
        from soup_cli.utils.loop_iteration import list_iterations, read_iteration
        ids = list_iterations()
        assert len(ids) == 1
        rec = read_iteration(ids[0])
        assert "pack-failed" in rec.notes

    def test_pack_iterations_false_skips_packing(self, loop_env, tmp_path):
        import glob

        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.loop_daemon import WatchConfig, watch
        from soup_cli.utils.loop_state import init_state

        init_state("mymodel", "evals/gate.yaml", "base-ref")
        cfg = WatchConfig(
            poll_interval_sec=1.0, max_iterations=1, pack_iterations=False,
        )
        watch(cfg)
        # No cans written, no registry rows created.
        assert glob.glob(os.path.join(".soup-loops", "*", "iteration.can")) == []
        with RegistryStore() as store:
            assert store.list_by_name("mymodel") == []


class TestGhArgvAdjacency:
    def test_method_and_input_flags_adjacent(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        captured = {}

        class _R:
            returncode = 0
            stdout = '{"html_url": "u"}'
            stderr = ""

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return _R()

        post_pr_comment(
            "o/r#1", "body", env={"GITHUB_TOKEN": "t"}, runner=fake_run
        )
        argv = captured["argv"]
        # `--method POST` and `--input -` must be value-adjacent.
        assert argv[argv.index("--method") + 1] == "POST"
        assert argv[argv.index("--input") + 1] == "-"
        # No `shell=True` could ever apply — argv[0] is the bare binary.
        assert argv[0] == "gh"


class TestPostPrCommentEnvAllowlist:
    def test_secrets_not_leaked_to_child(self, monkeypatch):
        """When ``env`` is None, the gh child env is built from an allowlist
        so HF_TOKEN / OPENAI_API_KEY never reach the subprocess."""
        from soup_cli.utils.adapter_pr import post_pr_comment

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_real")
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("PATH", "/usr/bin")

        captured = {}

        class _R:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(argv, **kw):
            captured["env"] = kw.get("env")
            return _R()

        post_pr_comment("o/r#1", "body", runner=fake_run)  # env defaults to None
        child_env = captured["env"]
        assert child_env is not None
        assert "HF_TOKEN" not in child_env
        assert "OPENAI_API_KEY" not in child_env
        assert child_env.get("GH_TOKEN") == "ghp_real"
        assert "PATH" in child_env

    def test_body_over_cap_rejected(self):
        from soup_cli.utils.adapter_pr import post_pr_comment

        big = "x" * 60_001
        with pytest.raises(ValueError, match="cap"):
            post_pr_comment(
                "o/r#1", big, env={"GITHUB_TOKEN": "t"},
                runner=lambda *a, **k: None,
            )


class TestCmaesTempCleanup:
    def test_temp_removed_even_when_scorer_raises(self, tmp_path, monkeypatch):
        from soup_cli.utils.cmaes_merge import build_cmaes_eval_fn

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        plan = _cmaes_plan(tmp_path, [a, b])
        seen = {}

        def scorer(merged_dir, eval_suite):
            seen["dir"] = merged_dir
            assert os.path.isdir(merged_dir)
            raise RuntimeError("scorer blew up")

        fn = build_cmaes_eval_fn(plan, scorer=scorer)
        with pytest.raises(RuntimeError):
            fn((0.5, 0.5))
        # `finally: shutil.rmtree(..., ignore_errors=True)` removed the temp.
        assert not os.path.exists(seen["dir"])


class TestRegistryNameFromCap:
    def test_caps_at_128_chars(self):
        from soup_cli.utils.loop_iteration import registry_name_from

        out = registry_name_from("a" * 500)
        assert len(out) == 128
        assert out[0].isalnum()


class TestDeploySsrfGuard:
    def test_public_http_endpoint_rejected(self, loop_env, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        # Plain HTTP to a public IP is rejected by validate_webhook_url.
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://8.8.8.8:8000")
        out = ls.deploy_to_canary(
            loop_env(),
            {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/x"},
        )
        assert out["deployed"] is False
        assert "SSRF" in str(out.get("notes", ""))

    def test_public_https_endpoint_rejected(self, loop_env, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        # HTTPS to a PUBLIC host passes the webhook validator but the deploy
        # surface tightens to loopback/LAN only (review MEDIUM-4).
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "https://8.8.8.8:8000")
        out = ls.deploy_to_canary(
            loop_env(),
            {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/x"},
        )
        assert out["deployed"] is False
        assert "loopback/LAN" in str(out.get("notes", ""))

    def test_public_hostname_rejected(self, loop_env, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        # A non-IP hostname can't be verified private without DNS → rejected.
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "https://evil.example.com")
        out = ls.deploy_to_canary(
            loop_env(),
            {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/x"},
        )
        assert out["deployed"] is False
        assert "loopback/LAN" in str(out.get("notes", ""))

    def test_loopback_endpoint_allowed_to_post(self, loop_env, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
        # Inject the poster so no real network call happens.
        monkeypatch.setattr(ls, "_DEPLOY_POSTER", lambda ep, name: True)
        out = ls.deploy_to_canary(
            loop_env(),
            {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/win"},
        )
        assert out["deployed"] is True
        assert out["canary_verdict"] == "OK"

    def test_private_lan_endpoint_allowed(self, loop_env, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        # RFC1918 over HTTPS is a legitimate LAN serve endpoint.
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "https://192.168.1.5:8000")
        monkeypatch.setattr(ls, "_DEPLOY_POSTER", lambda ep, name: True)
        out = ls.deploy_to_canary(
            loop_env(),
            {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/win"},
        )
        assert out["deployed"] is True


class TestNoHeavyTopLevelImports:
    """Heavy deps must be lazy-imported inside functions (cold-start policy)."""

    @pytest.mark.parametrize("module", [
        "soup_cli.utils.cmaes_merge",
        "soup_cli.utils.loop_stages",
        "soup_cli.utils.loop_iteration",
        "soup_cli.utils.adapter_pr",
    ])
    def test_no_top_level_heavy_imports(self, module):
        import importlib

        path = importlib.import_module(module).__file__
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        # Strip everything after the first `def`/`class` so we only inspect
        # the module-top import block.
        head = re.split(r"\n(?:def |class )", src, maxsplit=1)[0]
        for heavy in ("torch", "transformers", "peft", "trl", "numpy", "httpx"):
            assert f"\nimport {heavy}" not in head, (module, heavy)
            assert f"\nfrom {heavy}" not in head, (module, heavy)


class TestLoopStagesAtomicWrites:
    def test_uses_atomic_write_text(self):
        import soup_cli.utils.loop_stages as ls

        path = ls.__file__
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        # Pairs + train-yaml writes go through the shared atomic helper.
        assert "atomic_write_text(" in src
        # No bare open(..., "w") for the artifact writes.
        assert 'open(pairs_path, "w")' not in src


class TestMergeAdaptersCmaesGuard:
    def test_cmaes_strategy_rejected_with_path_hint(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import merge_adapters

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        with pytest.raises(ValueError, match="run_cmaes_merge"):
            merge_adapters([a, b], str(tmp_path / "out"), strategy="cmaes")


class TestWriteMergedAdapterPublic:
    def test_public_symbol_and_alias(self):
        from soup_cli.utils import adapter_merge

        assert hasattr(adapter_merge, "write_merged_adapter")
        # back-compat alias preserved for any external caller.
        assert (
            adapter_merge._write_merged_adapter
            is adapter_merge.write_merged_adapter
        )

    def test_cmaes_imports_public_name(self):
        import soup_cli.utils.cmaes_merge as cm

        with open(cm.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "write_merged_adapter" in src
        assert "_write_merged_adapter" not in src


class TestPackEntryRollback:
    def test_pack_failure_rolls_back_registry_entry(
        self, loop_env, tmp_path, monkeypatch
    ):
        """If pack_entry fails after the Registry push, the entry is rolled
        back so the watch-loop lineage chain never points at an orphan."""
        import soup_cli.cans.pack as pack_mod
        from soup_cli.registry.store import RegistryStore
        from soup_cli.utils.loop_iteration import pack_iteration_as_can

        _write_iteration("iter-rb")

        def boom(**kw):
            raise OSError("disk full")

        monkeypatch.setattr(pack_mod, "pack_entry", boom)
        with pytest.raises(OSError):
            pack_iteration_as_can("iter-rb", served_model="mymodel")
        # No orphaned entry left behind.
        with RegistryStore() as store:
            assert store.list_by_name("mymodel") == []


class TestWatchConfigValidation:
    def test_pack_iterations_must_be_bool(self):
        from soup_cli.utils.loop_daemon import WatchConfig

        with pytest.raises(ValueError):
            WatchConfig(pack_iterations="yes")  # type: ignore[arg-type]

    def test_base_model_null_byte_rejected(self):
        from soup_cli.utils.loop_daemon import WatchConfig

        with pytest.raises(ValueError):
            WatchConfig(base_model="a\x00b")

    def test_base_model_empty_rejected(self):
        from soup_cli.utils.loop_daemon import WatchConfig

        with pytest.raises(ValueError):
            WatchConfig(base_model="")

    def test_served_model_oversize_rejected(self):
        from soup_cli.utils.loop_daemon import WatchConfig

        with pytest.raises(ValueError):
            WatchConfig(served_model="a" * 513)

    def test_served_model_none_ok(self):
        from soup_cli.utils.loop_daemon import WatchConfig

        cfg = WatchConfig(served_model=None)
        assert cfg.served_model is None


class TestCmaesScanGate:
    def _app(self):
        from soup_cli.commands.adapters import app

        return app

    def test_cmaes_respects_backdoor_scan_gate(self, tmp_path, monkeypatch):
        """A FAIL scan must block --strategy cmaes too (review MEDIUM-5)."""
        import soup_cli.utils.adapter_scan as scan_mod

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path / "a")
        b = _make_adapter(tmp_path / "b")
        suite = tmp_path / "eval.jsonl"
        suite.write_text('{"prompt": "p", "expected": "e"}\n', encoding="utf-8")

        class _Rep:
            overall = "FAIL"
            summary = "rank-1 dominance"

        monkeypatch.setattr(scan_mod, "scan_adapter", lambda p: _Rep())
        result = runner.invoke(self._app(), [
            "merge", a, b, "--strategy", "cmaes",
            "--eval", str(suite), "-o", str(tmp_path / "out"),
        ])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "FAIL" in result.output
