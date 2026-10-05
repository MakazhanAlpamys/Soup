"""Tests for issue #1436: diagnose forgetting probe inputs.

Ensures the live forgetting probe evaluates held-out / general prompts rather
than the adapter's own training rows from --dataset, catching catastrophic
forgetting when an adapter only performs on its fine-tuning data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

import soup_cli.utils.diagnose.live as live


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestForgettingProbeHoldout:
    def test_adapter_empty_outside_training_is_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 1: An adapter perfect on training data and empty elsewhere is flagged."""
        monkeypatch.chdir(tmp_path)
        train_file = tmp_path / "train.jsonl"
        rows = [
            {
                "prompt": f"Question {i}: what is code {i}?",
                "completion": f"The internal code {i} is ZETA{i}",
            }
            for i in range(12)
        ]
        _write_jsonl(train_file, rows)
        known = {r["prompt"]: r["completion"] for r in rows}

        asked: List[str] = []

        def adapter_gen(prompt: str) -> str:
            asked.append(prompt)
            return known.get(prompt, "")

        def base_gen(prompt: str) -> str:
            # Base model knows general knowledge benchmark answers,
            # but does not know private training codes.
            for p, ref in live._DEFAULT_FORGETTING_PROBES:
                if p == prompt:
                    return ref
            return "I do not know that code"

        monkeypatch.setattr(
            live,
            "load_adapter_pair",
            lambda base, adapter=None, **kw: {
                "base_gen": base_gen,
                "adapter_gen": adapter_gen,
                "base_multi": lambda p, k: [f"base {i}" for i in range(k)],
                "adapter_multi": lambda p, k: [f"adapter {i}" for i in range(k)],
            },
        )

        report = live.run_live_diagnose(
            run_id="test-run",
            base="base-model",
            adapter="adapter-lora",
            dataset_path=str(train_file),
        )

        forgetting_score = report.scores["forgetting"]
        # Must NOT report OK for an adapter that forgot general knowledge.
        assert forgetting_score.verdict != "OK"
        assert forgetting_score.verdict in ("MINOR", "MAJOR")
        assert forgetting_score.score < 0.85

    def test_forgetting_prompts_do_not_overlap_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 2: Prompts sent to the forgetting probe contain none of the --dataset rows."""
        monkeypatch.chdir(tmp_path)
        train_file = tmp_path / "train.jsonl"
        rows = [
            {"prompt": f"Task training prompt {i}", "response": f"Task target {i}"}
            for i in range(12)
        ]
        _write_jsonl(train_file, rows)
        dataset_prompts = {r["prompt"] for r in rows}

        forgetting_prompts_asked: List[str] = []

        def tracking_base_gen(p: str) -> str:
            return "base answer"

        def tracking_adapter_gen(p: str) -> str:
            forgetting_prompts_asked.append(p)
            return "adapter answer"

        monkeypatch.setattr(
            live,
            "load_adapter_pair",
            lambda base, adapter=None, **kw: {
                "base_gen": tracking_base_gen,
                "adapter_gen": tracking_adapter_gen,
                "base_multi": lambda p, k: [f"b {i}" for i in range(k)],
                "adapter_multi": lambda p, k: [f"a {i}" for i in range(k)],
            },
        )

        report = live.run_live_diagnose(
            run_id="test-run",
            base="base-model",
            adapter="adapter-lora",
            dataset_path=str(train_file),
        )

        # Assert on recorded prompts: none come from dataset_prompts
        assert len(forgetting_prompts_asked) > 0
        assert not any(p in dataset_prompts for p in forgetting_prompts_asked)
        assert report.scores["forgetting"] is not None

    def test_control_holdout_file_matching_base_is_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 3: With a hold-out file where adapter answers as well as base, forgetting is OK."""
        monkeypatch.chdir(tmp_path)
        train_file = tmp_path / "train.jsonl"
        holdout_file = tmp_path / "holdout.jsonl"

        train_rows = [{"prompt": f"Train {i}", "completion": f"Ans {i}"} for i in range(12)]
        holdout_rows = [{"prompt": f"Holdout {i}", "completion": f"Ref {i}"} for i in range(8)]
        _write_jsonl(train_file, train_rows)
        _write_jsonl(holdout_file, holdout_rows)

        # Both base and adapter answer holdout questions equally well
        def gen(prompt: str) -> str:
            return f"Answer for {prompt}"

        monkeypatch.setattr(
            live,
            "load_adapter_pair",
            lambda base, adapter=None, **kw: {
                "base_gen": gen,
                "adapter_gen": gen,
                "base_multi": lambda p, k: [f"b {i}" for i in range(k)],
                "adapter_multi": lambda p, k: [f"a {i}" for i in range(k)],
            },
        )

        report = live.run_live_diagnose(
            run_id="test-run",
            base="base-model",
            adapter="adapter-lora",
            dataset_path=str(train_file),
            holdout_path=str(holdout_file),
        )

        assert report.scores["forgetting"].verdict == "OK"
        assert report.scores["forgetting"].score >= 0.85

    def test_memorization_still_probes_dataset_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 4: Memorization still probes the --dataset (training) rows."""
        monkeypatch.chdir(tmp_path)
        train_file = tmp_path / "train.jsonl"
        train_rows = [
            {"prompt": f"UniquePromptKey_{i}", "completion": f"UniqueCompletionVal_{i}"}
            for i in range(12)
        ]
        _write_jsonl(train_file, train_rows)

        memorization_rows_probed: List[dict] = []

        import soup_cli.utils.diagnose.memorization as mem_mod

        orig_score_memorization = mem_mod.score_memorization

        def spy_score_memorization(training_rows, generator, **kw):
            memorization_rows_probed.extend(training_rows)
            return orig_score_memorization(training_rows, generator, **kw)

        monkeypatch.setattr(mem_mod, "score_memorization", spy_score_memorization)

        monkeypatch.setattr(
            live,
            "load_adapter_pair",
            lambda base, adapter=None, **kw: {
                "base_gen": lambda p: "base",
                "adapter_gen": lambda p: "adapter",
                "base_multi": lambda p, k: [f"b {i}" for i in range(k)],
                "adapter_multi": lambda p, k: [f"a {i}" for i in range(k)],
            },
        )

        live.run_live_diagnose(
            run_id="test-run",
            base="base-model",
            adapter="adapter-lora",
            dataset_path=str(train_file),
        )

        # Probed rows must match the training file
        assert len(memorization_rows_probed) == 12
        assert memorization_rows_probed[0]["prompt"] == "UniquePromptKey_0"

    @staticmethod
    def _stub_pair(monkeypatch: pytest.MonkeyPatch, adapter_gen, base_gen=None) -> None:
        monkeypatch.setattr(
            live,
            "load_adapter_pair",
            lambda base, adapter=None, **kw: {
                "base_gen": base_gen or (lambda p: "base"),
                "adapter_gen": adapter_gen,
                "base_multi": lambda p, k: [f"b {i}" for i in range(k)],
                "adapter_multi": lambda p, k: [f"a {i}" for i in range(k)],
            },
        )

    def test_bad_holdout_path_fails_before_the_model_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_holdout_1436.jsonl"
        _write_jsonl(outside, [{"prompt": "q", "response": "a"}])

        def boom(*args, **kwargs):
            raise AssertionError("the model was loaded before --holdout was validated")

        monkeypatch.setattr(live, "load_adapter_pair", boom)
        with pytest.raises(ValueError, match="holdout path"):
            live.run_live_diagnose(run_id="r", base="b", holdout_path=str(outside))

    def test_the_adapter_is_asked_the_holdout_rows_and_not_the_builtin_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        holdout = tmp_path / "holdout.jsonl"
        rows = [{"prompt": f"Held out {i}", "completion": f"Ref {i}"} for i in range(5)]
        _write_jsonl(holdout, rows)
        asked: list = []

        def adapter_gen(prompt):
            asked.append(prompt)
            return "x"

        self._stub_pair(monkeypatch, adapter_gen)
        live.run_live_diagnose(run_id="r", base="b", adapter="a", holdout_path=str(holdout))
        # the refusal probe reuses one of the built-in questions, so subtract its prompts
        refusal = set(live._REFUSAL_HARMFUL) | set(live._REFUSAL_BENIGN)
        builtin = {p for p, _ in live._DEFAULT_FORGETTING_PROBES} - refusal
        assert {r["prompt"] for r in rows} <= set(asked)
        assert not builtin & set(asked)

    @pytest.mark.parametrize("content", [[], [{"foo": "bar"}]])
    def test_unusable_holdout_is_not_run_not_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: list
    ) -> None:
        monkeypatch.chdir(tmp_path)
        holdout = tmp_path / "holdout.jsonl"
        _write_jsonl(holdout, content)
        self._stub_pair(monkeypatch, lambda p: "x")
        report = live.run_live_diagnose(
            run_id="r", base="b", adapter="a", holdout_path=str(holdout)
        )
        assert report.scores["forgetting"].verdict == "NOT_RUN", report.scores["forgetting"]

    def test_unusable_dataset_does_not_overwrite_the_measured_forgetting_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        dataset = tmp_path / "train.jsonl"
        _write_jsonl(dataset, [{"foo": "bar"}])
        known = dict(live._DEFAULT_FORGETTING_PROBES)
        self._stub_pair(monkeypatch, lambda p: "", base_gen=lambda p: known.get(p, "x"))
        report = live.run_live_diagnose(
            run_id="r", base="b", adapter="a", dataset_path=str(dataset)
        )
        assert report.scores["forgetting"].verdict == "MAJOR", report.scores["forgetting"]
        assert report.scores["format"].verdict == "NOT_RUN"

    def test_a_forgetting_probe_that_raises_is_not_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        def adapter_gen(prompt):
            raise ValueError("generation failed")

        self._stub_pair(monkeypatch, adapter_gen)
        report = live.run_live_diagnose(run_id="r", base="b", adapter="a")
        assert report.scores["forgetting"].verdict == "NOT_RUN", report.scores["forgetting"]

    def test_holdout_capped_at_probe_prompts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With 15 holdout rows, adapter is asked exactly the first 12."""
        monkeypatch.chdir(tmp_path)
        holdout_file = tmp_path / "holdout.jsonl"
        rows = [{"prompt": f"holdout_prompt_{i}", "completion": f"c_{i}"} for i in range(15)]
        _write_jsonl(holdout_file, rows)

        asked: list[str] = []

        def adapter_gen(prompt: str) -> str:
            asked.append(prompt)
            return "x"

        self._stub_pair(monkeypatch, adapter_gen)
        live.run_live_diagnose(
            run_id="r", base="b", adapter="a", holdout_path=str(holdout_file)
        )
        holdout_asked = [p for p in asked if p.startswith("holdout_prompt_")]
        assert holdout_asked == [f"holdout_prompt_{i}" for i in range(12)]

    def test_cli_accepts_holdout_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI accepts --holdout and passes it to run_live_diagnose."""
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        holdout_file = tmp_path / "holdout.jsonl"
        _write_jsonl(holdout_file, [{"prompt": "hp", "response": "ha"}])

        captured_kwargs = {}

        def mock_run_live(**kwargs):
            captured_kwargs.update(kwargs)
            from soup_cli.utils.diagnose.report import FailureReport
            return FailureReport(
                run_id=kwargs["run_id"],
                base=kwargs["base"],
                adapter=kwargs.get("adapter") or "",
                scores={},
                overall="OK",
            )

        monkeypatch.setattr("soup_cli.utils.diagnose.live.run_live_diagnose", mock_run_live)

        result = runner.invoke(
            app,
            [
                "diagnose",
                "run-123",
                "--base-model",
                "test-base",
                "--holdout",
                str(holdout_file),
            ],
        )

        assert result.exit_code == 0
        assert captured_kwargs.get("holdout_path") == str(holdout_file)
