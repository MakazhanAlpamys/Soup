"""Regression tests for #1909: a one-pair `soup loop watch` harvest could not
train because the generated DPO config kept the schema-default
``data.val_split: 0.1``, and ``_split_val`` left 0 training rows — the train
stage then only logged ``soup train exited rc=…`` with no hint that the
harvest was too small to split.

After the fix ``_render_dpo_yaml`` writes an explicit ``val_split`` sized to
the harvest (0.1 for ≥2 pairs, 0.0 for a single pair — the generated config
carries no ``training.eval_steps``, so an empty val side never reaches the
eval scheduler's validation-rows check), and a failed train subprocess
surfaces the stderr tail as a ``skip_reason``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.data.loader import _split_val
from soup_cli.utils import loop_stages
from soup_cli.utils.loop_state import LoopState


def _state() -> LoopState:
    return LoopState(
        served_model="http://127.0.0.1:8000",
        eval_suite="suite.yaml",
        baseline="baseline.json",
    )


class TestRenderDpoValSplit:
    def test_single_pair_uses_val_split_zero(self) -> None:
        text = loop_stages._render_dpo_yaml(_state(), "pairs.jsonl", "out", 1)
        config = yaml.safe_load(text)
        assert config["data"]["val_split"] == 0.0
        train, val = _split_val([{"row": i} for i in range(1)], config["data"]["val_split"])
        assert (len(train), len(val)) == (1, 0)

    def test_two_pairs_keep_one_train_row(self) -> None:
        text = loop_stages._render_dpo_yaml(_state(), "pairs.jsonl", "out", 2)
        config = yaml.safe_load(text)
        assert config["data"]["val_split"] == 0.1
        train, val = _split_val([{"row": i} for i in range(2)], config["data"]["val_split"])
        assert (len(train), len(val)) == (1, 1)

    @pytest.mark.parametrize("pairs,train,val", [(5, 4, 1), (10, 9, 1), (20, 18, 2)])
    def test_larger_harvests_keep_the_default_fraction(
        self, pairs: int, train: int, val: int,
    ) -> None:
        text = loop_stages._render_dpo_yaml(_state(), "pairs.jsonl", "out", pairs)
        config = yaml.safe_load(text)
        assert config["data"]["val_split"] == 0.1
        got_train, got_val = _split_val(
            [{"row": i} for i in range(pairs)], config["data"]["val_split"],
        )
        assert (len(got_train), len(got_val)) == (train, val)

    def test_val_split_is_written_explicitly(self) -> None:
        # The pre-fix renderer emitted no val_split key at all, so the schema
        # default applied. The key must be present in the rendered YAML.
        text = loop_stages._render_dpo_yaml(_state(), "pairs.jsonl", "out", 1)
        assert "val_split" in yaml.safe_load(text)["data"]

    def test_no_eval_steps_in_the_generated_config(self) -> None:
        # What makes val_split=0 safe: the eval scheduler only demands
        # validation rows when training.eval_steps asks for evaluation.
        text = loop_stages._render_dpo_yaml(_state(), "pairs.jsonl", "out", 1)
        config = yaml.safe_load(text)
        assert "eval_steps" not in config.get("training", {})


class TestTrainFailureDiagnostic:
    def _run_train(self, runner_result, monkeypatch, tmp_path: Path) -> dict:
        # The train stage writes its YAML under cwd-relative .soup-loops/;
        # keep that out of the repository root.
        monkeypatch.chdir(tmp_path)
        calls: list[list[str]] = []

        def _runner(argv, capture_output=True, timeout=None, check=False):
            calls.append(argv)
            return runner_result

        saved = loop_stages._TRAIN_RUNNER
        loop_stages._TRAIN_RUNNER = _runner
        try:
            state = _state()
            ctx = {"pairs_path": "pairs.jsonl", "pairs_harvested": 3}
            return loop_stages.train_dpo_from_pairs(state, ctx)
        finally:
            loop_stages._TRAIN_RUNNER = saved

    def test_failed_train_surfaces_stderr_tail_as_skip_reason(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        result = self._run_train(
            SimpleNamespace(
                returncode=1,
                stderr=(
                    "Traceback (most recent call last):\n  loading model\n"
                    "No training rows: loaded 0 train samples\n"
                ),
            ),
            monkeypatch,
            tmp_path,
        )
        assert result["skipped"] is True
        assert result["run_id"] is None
        reason = result["skip_reason"]
        assert "No training rows: loaded 0 train samples" in reason

    def test_stderr_tail_is_capped_at_three_lines(self, monkeypatch, tmp_path: Path) -> None:
        stderr = "\n".join(f"noise line {i}" for i in range(10)) + "\nfinal line\n"
        result = self._run_train(
            SimpleNamespace(returncode=2, stderr=stderr), monkeypatch, tmp_path,
        )
        reason = result["skip_reason"]
        assert reason.endswith("final line)")
        assert "noise line 0" not in reason

    def test_failure_without_stderr_still_names_the_exit_code(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        result = self._run_train(
            SimpleNamespace(returncode=137, stderr=""), monkeypatch, tmp_path,
        )
        assert result["skip_reason"] == "soup train failed (exit code 137)"

    def test_harvested_pair_count_reaches_the_renderer(self, monkeypatch, tmp_path: Path) -> None:
        # pairs_harvested=1 must produce a val_split of 0.0 in the written YAML.
        captured: dict = {}

        def _runner(argv, capture_output=True, timeout=None, check=False):
            captured["yaml_path"] = argv[argv.index("--config") + 1]
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        saved = loop_stages._TRAIN_RUNNER
        loop_stages._TRAIN_RUNNER = _runner
        try:
            loop_stages.train_dpo_from_pairs(
                _state(), {"pairs_path": "pairs.jsonl", "pairs_harvested": 1},
            )
        finally:
            loop_stages._TRAIN_RUNNER = saved
        config = yaml.safe_load(Path(captured["yaml_path"]).read_text(encoding="utf-8"))
        assert config["data"]["val_split"] == 0.0
