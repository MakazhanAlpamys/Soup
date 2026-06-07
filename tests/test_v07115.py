"""v0.71.15 — "Loop & lifecycle polish".

Closes #261 (iterative-DPO ``_default_train_fn`` rendered ``output: {dir}`` which
SoupConfig rejects — mirrors the v0.71.13 #229 ``local_rl`` fix), #246
(CMA-ES default scorer reloaded the base per candidate — now load once and
reuse across the population loop), #245 (``soup loop`` ``estimate_cost`` was a
hard ``0.0`` placeholder — now wires v0.34 ``run_cost.estimate_run_cost_usd``
off the last completed run), #244 (``soup train --track-energy --energy-out``
persists the measurement so ``soup bom emit --energy`` can consume it), and
#170 (``--diagnose-gate`` fired on ``LOCAL_RANK==0`` — now ``RANK``-aware so a
multi-node run gates once per cluster, not once per node).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner


def _clean(text: str) -> str:
    """Strip ANSI + collapse whitespace so Rich line-wrapping doesn't break
    substring asserts (matches the v0.71.1 CI-fix policy)."""
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", text))


# ---------------------------------------------------------------------------
# #261 — iterative_dpo._default_train_fn renders a SoupConfig-valid YAML
# ---------------------------------------------------------------------------


class TestIterativeDpoDefaultTrainFn:
    def test_renders_schema_valid_config(self, tmp_path, monkeypatch):
        """The rendered YAML must round-trip through ``load_config_from_string``.

        The bug: ``output: {dir: ...}`` (a mapping) — SoupConfig.output is a
        plain ``str``, so config load raised. The fix mirrors #229's
        ``local_rl`` flat-string rendering.
        """
        import subprocess

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils import iterative_dpo as idpo

        monkeypatch.chdir(tmp_path)
        (tmp_path / "pairs.jsonl").write_text("{}\n", encoding="utf-8")
        captured: dict = {}

        def _fake_run(argv, **kw):
            cfg_path = argv[argv.index("--config") + 1]
            captured["yaml"] = Path(cfg_path).read_text(encoding="utf-8")

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        idpo._default_train_fn(
            base_model="hf-internal-testing/tiny-random-gpt2",
            pairs_path=str(tmp_path / "pairs.jsonl"),
            adapter_path="adapter_out",
        )
        cfg = load_config_from_string(captured["yaml"])
        assert cfg.output == "adapter_out"
        assert cfg.task == "dpo"
        assert cfg.base == "hf-internal-testing/tiny-random-gpt2"

    def test_output_is_a_plain_string_in_yaml(self, tmp_path, monkeypatch):
        """Regression guard: the rendered YAML must NOT nest output under a
        ``dir`` key (the exact shape of the #261 bug)."""
        import subprocess

        from soup_cli.utils import iterative_dpo as idpo

        monkeypatch.chdir(tmp_path)
        (tmp_path / "pairs.jsonl").write_text("{}\n", encoding="utf-8")
        captured: dict = {}

        def _fake_run(argv, **kw):
            cfg_path = argv[argv.index("--config") + 1]
            captured["doc"] = yaml.safe_load(
                Path(cfg_path).read_text(encoding="utf-8")
            )

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        idpo._default_train_fn(
            base_model="b",
            pairs_path=str(tmp_path / "pairs.jsonl"),
            adapter_path="adapter_out",
        )
        assert captured["doc"]["output"] == "adapter_out"
        assert not isinstance(captured["doc"]["output"], dict)


# ---------------------------------------------------------------------------
# #246 — CMA-ES default scorer loads the base model once, reuses it
# ---------------------------------------------------------------------------


class _FakeBase:
    pass


class _FakeTok:
    eos_token_id = 0


class _FakePeft:
    def __init__(self, base):
        self._base = base

    def eval(self):
        return self

    def unload(self):
        return self._base


def _write_merged_dir(path, base_name="tiny"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_name}), encoding="utf-8"
    )
    return str(path)


def _patch_cmaes_libs(monkeypatch, base_loads):
    """Patch the lazy transformers/peft/eval imports the scorer resolves."""
    import peft
    import transformers

    import soup_cli.eval.custom as ev
    import soup_cli.utils.cmaes_merge as cm

    def _fake_model_load(name, **kw):
        base_loads.append(name)
        return _FakeBase()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", _fake_model_load
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda name, **kw: _FakeTok()
    )
    monkeypatch.setattr(
        peft.PeftModel, "from_pretrained", lambda base, d, **kw: _FakePeft(base)
    )
    monkeypatch.setattr(cm, "_generate", lambda model, tok, prompt: "out")

    class _Task:
        prompt = "hi"

    class _Score:
        score = 0.5

    monkeypatch.setattr(ev, "load_eval_tasks", lambda suite: [_Task(), _Task()])
    monkeypatch.setattr(ev, "score_task", lambda task, out: _Score())
    return cm


class TestCachedBaseScorer:
    def test_loads_base_once_across_candidates(self, tmp_path, monkeypatch):
        base_loads: list = []
        cm = _patch_cmaes_libs(monkeypatch, base_loads)
        dirs = [_write_merged_dir(tmp_path / f"m{i}") for i in range(3)]
        scorer = cm._CachedBaseScorer()
        scores = [scorer(d, "suite.jsonl") for d in dirs]
        assert scores == [0.5, 0.5, 0.5]
        # The expensive base model is loaded exactly once and reused.
        assert base_loads == ["tiny"]

    def test_base_mismatch_raises(self, tmp_path, monkeypatch):
        base_loads: list = []
        cm = _patch_cmaes_libs(monkeypatch, base_loads)
        scorer = cm._CachedBaseScorer()
        scorer(_write_merged_dir(tmp_path / "a", "tiny"), "s.jsonl")
        with pytest.raises(ValueError, match="base_model mismatch"):
            scorer(_write_merged_dir(tmp_path / "b", "other"), "s.jsonl")

    def test_empty_tasks_returns_zero(self, tmp_path, monkeypatch):
        import soup_cli.eval.custom as ev

        base_loads: list = []
        cm = _patch_cmaes_libs(monkeypatch, base_loads)
        monkeypatch.setattr(ev, "load_eval_tasks", lambda suite: [])
        scorer = cm._CachedBaseScorer()
        assert scorer(_write_merged_dir(tmp_path / "a"), "s.jsonl") == 0.0

    def test_resolve_returns_fresh_cached_scorer(self, monkeypatch):
        import soup_cli.utils.cmaes_merge as cm

        monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", None)
        s1 = cm._resolve_cmaes_scorer(None)
        s2 = cm._resolve_cmaes_scorer(None)
        assert isinstance(s1, cm._CachedBaseScorer)
        assert isinstance(s2, cm._CachedBaseScorer)
        assert s1 is not s2  # per-run cache isolation

    def test_resolve_honours_override_and_injection(self, monkeypatch):
        import soup_cli.utils.cmaes_merge as cm

        def injected(d, s):
            return 0.9

        assert cm._resolve_cmaes_scorer(injected) is injected
        monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", lambda d, s: 0.6)
        out = cm._resolve_cmaes_scorer(None)
        assert out is cm._CMAES_SCORER_OVERRIDE

    def test_unload_clears_lingering_peft_config(self, tmp_path, monkeypatch):
        """The ``finally`` block must strip a lingering ``peft_config`` left
        by ``unload()`` so the next candidate re-wraps a clean base (no
        "multiple adapters" warning / accumulation)."""
        import peft

        base_loads: list = []
        cm = _patch_cmaes_libs(monkeypatch, base_loads)

        class _BaseWithCfg:
            def __init__(self):
                self.peft_config = {"default": object()}

        class _PeftLeavesCfg:
            def __init__(self, base):
                self._base = base

            def eval(self):
                return self

            def unload(self):
                # real PEFT often leaves peft_config on the unwrapped base
                return self._base

        monkeypatch.setattr(
            __import__("transformers").AutoModelForCausalLM,
            "from_pretrained",
            lambda name, **kw: (base_loads.append(name) or _BaseWithCfg()),
        )
        monkeypatch.setattr(
            peft.PeftModel, "from_pretrained", lambda base, d, **kw: _PeftLeavesCfg(base)
        )
        scorer = cm._CachedBaseScorer()
        scorer(_write_merged_dir(tmp_path / "a"), "s.jsonl")
        # peft_config must have been delattr'd off the cached base.
        assert not hasattr(scorer._base, "peft_config")

    def test_default_scorer_still_stateless(self, tmp_path, monkeypatch):
        """The back-compat ``_default_cmaes_scorer`` reloads per call."""
        base_loads: list = []
        cm = _patch_cmaes_libs(monkeypatch, base_loads)
        d = _write_merged_dir(tmp_path / "a")
        cm._default_cmaes_scorer(d, "s.jsonl")
        cm._default_cmaes_scorer(d, "s.jsonl")
        assert base_loads == ["tiny", "tiny"]  # one load per call (stateless)


# ---------------------------------------------------------------------------
# #245 — loop estimate_cost wires run_cost off the last completed run
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def _state(self):
        from soup_cli.utils.loop_state import LoopState

        return LoopState(served_model="m", eval_suite="e.jsonl", baseline="b")

    def _patch_tracker(self, monkeypatch, runs):
        import soup_cli.experiment.tracker as trk

        monkeypatch.setattr(trk.ExperimentTracker, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(
            trk.ExperimentTracker, "list_runs", lambda self, limit=50: runs
        )

    def test_uses_last_completed_priced_run(self, monkeypatch):
        from soup_cli.utils import loop_stages

        self._patch_tracker(
            monkeypatch,
            [{"status": "completed", "device_name": "NVIDIA A100", "duration_secs": 3600.0}],
        )
        cost = loop_stages.estimate_cost(self._state())
        # A100 = $1.10/hr × 1 hour
        assert cost == pytest.approx(1.10, abs=0.01)

    def test_skips_running_finds_completed(self, monkeypatch):
        from soup_cli.utils import loop_stages

        self._patch_tracker(
            monkeypatch,
            [
                {"status": "running", "device_name": "NVIDIA A100", "duration_secs": 3600.0},
                {"status": "completed", "device_name": "T4", "duration_secs": 3600.0},
            ],
        )
        cost = loop_stages.estimate_cost(self._state())
        assert cost == pytest.approx(0.20, abs=0.01)  # T4 = $0.20/hr

    def test_unpriced_gpu_returns_zero(self, monkeypatch):
        from soup_cli.utils import loop_stages

        self._patch_tracker(
            monkeypatch,
            [{"status": "completed", "device_name": "cpu", "duration_secs": 3600.0}],
        )
        assert loop_stages.estimate_cost(self._state()) == 0.0

    def test_no_completed_run_returns_zero(self, monkeypatch):
        from soup_cli.utils import loop_stages

        self._patch_tracker(monkeypatch, [{"status": "running", "device_name": "NVIDIA A100"}])
        assert loop_stages.estimate_cost(self._state()) == 0.0

    def test_empty_tracker_returns_zero(self, monkeypatch):
        from soup_cli.utils import loop_stages

        self._patch_tracker(monkeypatch, [])
        assert loop_stages.estimate_cost(self._state()) == 0.0

    def test_completed_run_missing_duration_returns_zero(self, monkeypatch):
        from soup_cli.utils import loop_stages

        # completed but no duration_secs -> skip the field check -> 0.0
        self._patch_tracker(
            monkeypatch, [{"status": "completed", "device_name": "NVIDIA A100"}]
        )
        assert loop_stages.estimate_cost(self._state()) == 0.0

    def test_completed_run_missing_device_returns_zero(self, monkeypatch):
        from soup_cli.utils import loop_stages

        self._patch_tracker(
            monkeypatch, [{"status": "completed", "duration_secs": 3600.0}]
        )
        assert loop_stages.estimate_cost(self._state()) == 0.0

    def test_estimate_raises_is_swallowed(self, monkeypatch):
        from soup_cli.utils import loop_stages, run_cost

        self._patch_tracker(
            monkeypatch,
            [
                {"status": "completed", "device_name": "A100", "duration_secs": 3600.0},
            ],
        )

        def _boom(*a, **k):
            raise ValueError("bad")

        monkeypatch.setattr(run_cost, "estimate_run_cost_usd", _boom)
        # The per-run estimate raising is caught (continue) -> no more rows -> 0.0
        assert loop_stages.estimate_cost(self._state()) == 0.0

    def test_tracker_error_never_crashes(self, monkeypatch):
        import soup_cli.experiment.tracker as trk
        from soup_cli.utils import loop_stages

        def _boom(self, *a, **k):
            raise RuntimeError("db locked")

        monkeypatch.setattr(trk.ExperimentTracker, "__init__", _boom)
        # cost estimation must never crash the daemon — returns 0.0
        assert loop_stages.estimate_cost(self._state()) == 0.0


# ---------------------------------------------------------------------------
# #244 — train --track-energy --energy-out → bom emit --energy handoff
# ---------------------------------------------------------------------------


class TestEnergyOutHandoff:
    def _measurement(self):
        from soup_cli.utils.energy import EnergyMeasurement

        return EnergyMeasurement(
            energy_kwh=0.5,
            co2_kg=0.2,
            pue=1.1,
            grid_intensity_g_per_kwh=400.0,
            source="codecarbon-offline",
        )

    def test_write_energy_json_roundtrips_into_bom_consumer(self, tmp_path, monkeypatch):
        """The keys written must be EXACTLY what ``bom emit --energy`` reads
        via ``EnergyMeasurement(**parsed)``."""
        from soup_cli.commands.train import _write_energy_json
        from soup_cli.utils.energy import EnergyMeasurement

        monkeypatch.chdir(tmp_path)
        m = self._measurement()
        _write_energy_json("energy.json", m)
        parsed = json.loads((tmp_path / "energy.json").read_text(encoding="utf-8"))
        assert set(parsed) == {
            "energy_kwh",
            "co2_kg",
            "pue",
            "grid_intensity_g_per_kwh",
            "source",
        }
        assert EnergyMeasurement(**parsed) == m

    def test_write_energy_json_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _write_energy_json

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            _write_energy_json("../escape.json", self._measurement())

    @pytest.mark.skipif(
        sys.platform == "win32", reason="symlink needs elevation on Windows"
    )
    def test_write_energy_json_rejects_symlink(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _write_energy_json

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real.json").write_text("{}", encoding="utf-8")
        os.symlink(tmp_path / "real.json", tmp_path / "energy.json")
        with pytest.raises(ValueError):
            _write_energy_json("energy.json", self._measurement())

    def test_train_help_lists_energy_out(self):
        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--energy-out" in _clean(result.output)

    def test_full_handoff_into_bom_emit(self, tmp_path, monkeypatch):
        """End-to-end: write energy.json via the producer helper, then feed it
        to ``soup bom emit --energy`` (the #256 consumer) and confirm the BOM
        carries the energy properties."""
        from soup_cli.cli import app
        from soup_cli.commands.train import _write_energy_json

        monkeypatch.chdir(tmp_path)
        _write_energy_json("energy.json", self._measurement())
        sha = "a" * 64
        result = CliRunner().invoke(
            app,
            [
                "bom", "emit",
                "--name", "m",
                "--base-model", "b",
                "--base-sha", sha,
                "--config-sha", sha,
                "--energy", "energy.json",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "energy_kwh" in result.output


# ---------------------------------------------------------------------------
# #170 — diagnose-gate RANK-aware multi-node guard
# ---------------------------------------------------------------------------


class TestDiagnoseGateRankGuard:
    def _guard(self):
        from soup_cli.commands.train import _should_run_diagnose_gate_on_rank

        return _should_run_diagnose_gate_on_rank

    def test_multinode_global_chief_runs(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert self._guard()() is True

    def test_multinode_node1_chief_skips(self, monkeypatch):
        """node-1 chief: LOCAL_RANK=0 but global RANK=8 — must SKIP so a
        shared-FS run gates once per cluster, not once per node (#170)."""
        monkeypatch.setenv("RANK", "8")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert self._guard()() is False

    def test_single_node_falls_back_to_local_rank_zero(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert self._guard()() is True

    def test_single_node_nonzero_local_rank_skips(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert self._guard()() is False

    def test_nonzero_global_rank_short_circuits_local_rank(self, monkeypatch):
        """A non-chief worker (RANK=5) skips regardless of LOCAL_RANK -- the
        global-RANK branch must short-circuit, ignoring LOCAL_RANK entirely."""
        monkeypatch.setenv("RANK", "5")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert self._guard()() is False

    def test_empty_rank_falls_back_to_local(self, monkeypatch):
        monkeypatch.setenv("RANK", "")
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert self._guard()() is False

    def test_whitespace_rank_is_truthy_and_overruns(self, monkeypatch):
        # " " is non-empty -> RANK branch -> int(" ") raises -> over-run (True)
        monkeypatch.setenv("RANK", " ")
        assert self._guard()() is True

    def test_malformed_rank_overruns(self, monkeypatch):
        monkeypatch.setenv("RANK", "not-an-int")
        assert self._guard()() is True

    def test_malformed_local_rank_overruns(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "garbage")
        assert self._guard()() is True


# ---------------------------------------------------------------------------
# Patch invariants
# ---------------------------------------------------------------------------


class TestPatchInvariants:
    def test_version_at_least_71_15(self):
        import soup_cli

        parts = tuple(int(p) for p in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 15)

    def test_loop_stages_no_heavy_top_level_imports(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "loop_stages.py"
        ).read_text(encoding="utf-8")
        for bad in ("\nimport torch", "\nimport transformers", "\nimport numpy"):
            assert bad not in src

    def test_cmaes_no_heavy_top_level_imports(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "cmaes_merge.py"
        ).read_text(encoding="utf-8")
        for bad in ("\nimport torch", "\nimport transformers", "\nimport peft"):
            assert bad not in src
