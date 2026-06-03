"""v0.62.0 review-fix follow-up tests.

Closes coverage gaps surfaced by the TDD review:

* H1 — citation_faithful + non-SFT-family task gate.
* H2 — RAFT _MAX_RAFT_FIELD_LEN per-field oversize boundary.
* H3 — RAFT null-byte rejection on golden_doc / answer / distractors.
* H5 — `soup steer train --plan-only` deferred-marker stripping regression guard.
* H6 — `apply_edit` still raises with v0.61.1 marker for rome/memit/alphaedit.
* M1 — steering name regex 128-char boundary acceptance.
* M3/M4 — soup steer train `--base` over-cap + null-byte rejection.
* M5 — extract_citation_ids public API coverage.
* M8 — GRACE codebook flag-without-both-knobs cross-validator.
* M9 — validate_ra_dit_compat direct-caller null-byte / bool defence-in-depth.
* L1 — list_steers context-manager source-grep regression guard.
* L4 — GRACE codebook size/dim max-boundary acceptance.
* L5 — RA-DIT retriever model 512-char boundary.
* L7 — CitationScore predicted_count / expected_count field values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---------- H1 — citation_faithful task-gate ----------


class TestCitationFaithfulTaskGate:
    def test_citation_faithful_on_grpo_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo

data:
  train: ./data/raft.jsonl
  format: raft

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  citation_faithful: true
  reward_fn: accuracy

output: ./output
"""
        with pytest.raises(Exception, match="sft|pretrain"):
            load_config_from_string(yaml_text)

    def test_citation_faithful_on_pretrain_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: pretrain

data:
  train: ./data/raft.jsonl
  format: raft

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  citation_faithful: true

output: ./output
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.citation_faithful is True

    def test_citation_faithful_on_dpo_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: dpo

data:
  train: ./data/raft.jsonl
  format: raft

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  citation_faithful: true

output: ./output
"""
        with pytest.raises(Exception, match="sft|pretrain"):
            load_config_from_string(yaml_text)


# ---------- H2 — RAFT per-field oversize boundary ----------


class TestRaftFieldOversizeBoundary:
    def test_query_at_cap_accepted(self):
        from soup_cli.data.formats import _MAX_RAFT_FIELD_LEN, _convert_raft

        out = _convert_raft({
            "query": "x" * _MAX_RAFT_FIELD_LEN,
            "golden_doc": "g",
            "distractor_docs": [],
            "answer": "a",
        })
        assert len(out["query"]) == _MAX_RAFT_FIELD_LEN

    def test_query_overcap_rejected(self):
        from soup_cli.data.formats import _MAX_RAFT_FIELD_LEN, _convert_raft

        with pytest.raises(ValueError, match="query"):
            _convert_raft({
                "query": "x" * (_MAX_RAFT_FIELD_LEN + 1),
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_golden_doc_overcap_rejected(self):
        from soup_cli.data.formats import _MAX_RAFT_FIELD_LEN, _convert_raft

        with pytest.raises(ValueError, match="golden_doc"):
            _convert_raft({
                "query": "q",
                "golden_doc": "x" * (_MAX_RAFT_FIELD_LEN + 1),
                "distractor_docs": [],
                "answer": "a",
            })

    def test_answer_overcap_rejected(self):
        from soup_cli.data.formats import _MAX_RAFT_FIELD_LEN, _convert_raft

        with pytest.raises(ValueError, match="answer"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "x" * (_MAX_RAFT_FIELD_LEN + 1),
            })

    def test_distractor_overcap_rejected(self):
        from soup_cli.data.formats import _MAX_RAFT_FIELD_LEN, _convert_raft

        with pytest.raises(ValueError, match="distractor"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": ["x" * (_MAX_RAFT_FIELD_LEN + 1)],
                "answer": "a",
            })


# ---------- H3 — RAFT null-byte rejection on every field ----------


class TestRaftNullByteRejection:
    def test_null_byte_golden_doc_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="null"):
            _convert_raft({
                "query": "q",
                "golden_doc": "bad\x00",
                "distractor_docs": [],
                "answer": "a",
            })

    def test_null_byte_answer_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="null"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": [],
                "answer": "bad\x00",
            })

    def test_null_byte_distractor_rejected(self):
        from soup_cli.data.formats import _convert_raft

        with pytest.raises(ValueError, match="null"):
            _convert_raft({
                "query": "q",
                "golden_doc": "g",
                "distractor_docs": ["bad\x00doc"],
                "answer": "a",
            })


# ---------- H5 — `soup steer train --plan-only` deferred-marker assertion ----------


class TestSteerPlanOnlyMarker:
    def test_plan_only_emits_v0621_marker(self, tmp_path, monkeypatch):
        from soup_cli.commands.steer import app

        monkeypatch.chdir(tmp_path)
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text(
            '{"positive": "x", "negative": "y"}\n', encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(app, [
            "train",
            "--base", "meta-llama/Llama-3.1-8B-Instruct",
            "--method", "caa",
            "--name", "safety-v1",
            "--pairs", "pairs.jsonl",
            "--plan-only",
        ])
        assert result.exit_code == 0, (
            result.output, repr(result.exception)
        )
        # Defence: the panel MUST surface the deferred-version marker so
        # operators know live training ships in v0.62.1.
        assert "v0.62.1" in result.output


# ---------- H6 — apply_edit v0.61.1 marker not regressed for legacy methods ----------


class TestEditMarkerRegressionGuard:
    # v0.71.9 #194/#203 — apply_edit is now LIVE. These guard that the
    # ROME-family methods route through run_edit_kernel and grace routes
    # through apply_grace_edit (mocked so no model load happens).
    @pytest.mark.parametrize("method", ["rome", "memit", "alphaedit"])
    def test_legacy_methods_dispatch_to_kernel(self, method: str, monkeypatch):
        import soup_cli.utils.edit_kernels as ek
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        plan = build_edit_plan(
            base="sshleifer/tiny-gpt2",
            method=method,
            subject="The capital of France is",
            target="Lyon",
        )
        seen = {}

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: ("MODEL", "TOK", "cpu"),
        )
        monkeypatch.setattr(ek, "measure_target_prob", lambda *a, **k: 0.1)

        def _fake_kernel(model, tok, *, method, **kwargs):
            seen["method"] = method
            return ek.EditKernelResult(
                method=method, layer=5, norm_delta=0.5, layers_edited=(5,),
            )

        monkeypatch.setattr(ek, "run_edit_kernel", _fake_kernel)
        result = apply_edit(plan)
        assert seen["method"] == method
        assert result.method == method
        assert result.norm_delta == 0.5

    def test_grace_dispatches_to_grace_edit(self, monkeypatch):
        import soup_cli.utils.grace_codebook as gc
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        plan = build_edit_plan(
            base="sshleifer/tiny-gpt2",
            method="grace",
            subject="The capital of France is",
            target="Lyon",
        )
        monkeypatch.setattr(gc, "apply_grace_edit", lambda p, **k: "GRACE_RESULT")
        assert apply_edit(plan) == "GRACE_RESULT"


# ---------- M1 — steering name 128-char boundary ----------


class TestSteeringNameBoundary:
    def test_at_max_length_accepted(self):
        from soup_cli.utils.steering import validate_steering_name

        name = "a" + "b" * 127  # exactly 128 chars
        assert validate_steering_name(name) == name

    def test_one_over_max_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        with pytest.raises(ValueError):
            validate_steering_name("a" + "b" * 128)  # 129 chars


# ---------- M3/M4 — `--base` over-cap + null-byte rejection ----------


class TestSteerTrainBaseValidation:
    def _make_runner(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pairs.jsonl").write_text(
            '{"positive": "x", "negative": "y"}\n', encoding="utf-8"
        )
        return CliRunner()

    def test_base_overcap_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.steer import app

        runner = self._make_runner(tmp_path, monkeypatch)
        result = runner.invoke(app, [
            "train",
            "--base", "x" * 513,
            "--method", "caa",
            "--name", "safety-v1",
            "--pairs", "pairs.jsonl",
            "--plan-only",
        ])
        assert result.exit_code == 2
        assert "base" in result.output.lower()

    def test_base_null_byte_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.steer import app

        runner = self._make_runner(tmp_path, monkeypatch)
        result = runner.invoke(app, [
            "train",
            "--base", "bad\x00name",
            "--method", "caa",
            "--name", "safety-v1",
            "--pairs", "pairs.jsonl",
            "--plan-only",
        ])
        assert result.exit_code == 2
        assert "null" in result.output.lower() or "base" in result.output.lower()

    def test_base_at_max_length_accepted(self, tmp_path, monkeypatch):
        from soup_cli.commands.steer import app

        runner = self._make_runner(tmp_path, monkeypatch)
        result = runner.invoke(app, [
            "train",
            "--base", "x" * 512,
            "--method", "caa",
            "--name", "safety-v1",
            "--pairs", "pairs.jsonl",
            "--plan-only",
        ])
        assert result.exit_code == 0, (
            result.output, repr(result.exception)
        )


# ---------- M5 — extract_citation_ids public-API coverage ----------


class TestExtractCitationIds:
    def test_extracts_bracketed(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        assert extract_citation_ids("See [doc-1] and [doc-2].") == (
            "doc-1", "doc-2",
        )

    def test_preserves_duplicates(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        assert extract_citation_ids("[doc-1] [doc-1]") == ("doc-1", "doc-1")

    def test_empty_returns_empty_tuple(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        assert extract_citation_ids("plain text without citations") == ()

    def test_non_string_raises(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        with pytest.raises(TypeError):
            extract_citation_ids(42)

    def test_oversize_raises(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        with pytest.raises(ValueError):
            extract_citation_ids("x" * 2_000_001)

    def test_regex_requires_leading_alnum(self):
        from soup_cli.utils.citation_faithful import extract_citation_ids

        # `_CITATION_RE` requires `[A-Za-z0-9]` as the first char inside the
        # brackets; a leading underscore/dash should NOT match.
        assert extract_citation_ids("[_bad-id]") == ()
        assert extract_citation_ids("[-bad]") == ()


# ---------- M5 supplement — score_citations expected_ids per-entry validation ----------


class TestScoreCitationsExpectedIds:
    def test_non_string_in_expected_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(TypeError):
            score_citations(predicted="[doc-1]", expected_ids=(42,))

    def test_bool_in_expected_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(TypeError):
            score_citations(predicted="[doc-1]", expected_ids=(True,))

    def test_empty_string_in_expected_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(ValueError, match="non-empty"):
            score_citations(predicted="[doc-1]", expected_ids=("",))

    def test_null_byte_in_expected_rejected(self):
        from soup_cli.utils.citation_faithful import score_citations

        with pytest.raises(ValueError, match="null"):
            score_citations(predicted="[doc-1]", expected_ids=("bad\x00id",))


# ---------- L7 — CitationScore field values ----------


class TestCitationScoreFields:
    def test_predicted_count_and_expected_count_set(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(
            predicted="[doc-1] [doc-2]",
            expected_ids=("doc-1",),
        )
        assert score.predicted_count == 2
        assert score.expected_count == 1

    def test_predicted_count_includes_duplicates(self):
        from soup_cli.utils.citation_faithful import score_citations

        score = score_citations(
            predicted="[doc-1] [doc-1] [doc-1]",
            expected_ids=("doc-1",),
        )
        assert score.predicted_count == 3


# ---------- M8 — GRACE codebook flag without both knobs ----------


class TestGraceCodebookPartialConfig:
    def test_flag_alone_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  grace_codebook: true

output: ./output
"""
        with pytest.raises(Exception, match="grace_codebook"):
            load_config_from_string(yaml_text)

    def test_flag_with_dim_missing_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  grace_codebook: true
  grace_codebook_size: 128

output: ./output
"""
        with pytest.raises(Exception, match="grace_codebook"):
            load_config_from_string(yaml_text)

    def test_grace_in_default_edit_layer(self):
        from soup_cli.utils.knowledge_edit import _DEFAULT_EDIT_LAYER  # type: ignore

        assert "grace" in _DEFAULT_EDIT_LAYER
        assert isinstance(_DEFAULT_EDIT_LAYER["grace"], int)
        assert _DEFAULT_EDIT_LAYER["grace"] >= 0

    def test_grace_in_edit_metadata_mapping(self):
        from soup_cli.utils.knowledge_edit import _EDIT_METHOD_METADATA  # type: ignore

        assert "grace" in _EDIT_METHOD_METADATA


# ---------- M9 — validate_ra_dit_compat direct-caller defence-in-depth ----------


class TestValidateRaDitCompatDirect:
    def test_null_byte_task_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_compat

        with pytest.raises(ValueError, match="null"):
            validate_ra_dit_compat(stage="retriever", task="sft\x00")

    def test_null_byte_stage_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_compat

        with pytest.raises(ValueError, match="null"):
            validate_ra_dit_compat(stage="retriever\x00", task="embedding")

    def test_bool_task_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_compat

        with pytest.raises(TypeError):
            validate_ra_dit_compat(stage="retriever", task=True)

    def test_bool_stage_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_compat

        with pytest.raises(TypeError):
            validate_ra_dit_compat(stage=True, task="embedding")

    def test_empty_task_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_compat

        with pytest.raises(ValueError, match="non-empty"):
            validate_ra_dit_compat(stage="retriever", task="")


# ---------- L4 — GRACE size/dim max-boundary acceptance ----------


class TestGraceMaxBoundary:
    def test_max_codebook_size_accepted(self):
        from soup_cli.utils.grace_codebook import (
            MAX_CODEBOOK_SIZE,
            validate_grace_codebook_size,
        )

        assert validate_grace_codebook_size(MAX_CODEBOOK_SIZE) == MAX_CODEBOOK_SIZE

    def test_max_codebook_dim_accepted(self):
        from soup_cli.utils.grace_codebook import (
            MAX_CODEBOOK_DIM,
            validate_grace_codebook_dim,
        )

        assert validate_grace_codebook_dim(MAX_CODEBOOK_DIM) == MAX_CODEBOOK_DIM


# ---------- L5 — RA-DIT retriever model 512-char boundary ----------


class TestRaDitRetrieverBoundary:
    def test_at_max_accepted(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        v = "x" * 512
        assert validate_ra_dit_retriever_model(v) == v

    def test_one_over_max_rejected(self):
        from soup_cli.utils.ra_dit import validate_ra_dit_retriever_model

        with pytest.raises(ValueError):
            validate_ra_dit_retriever_model("x" * 513)


# ---------- L1 — list_steers context-manager source-grep regression guard ----------


class TestSourceWiring:
    def test_list_steers_uses_context_manager(self):
        src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "commands" / "steer.py"
        text = src.read_text(encoding="utf-8")
        assert "with RegistryStore() as store:" in text

    def test_steer_uses_shared_path_helper(self):
        """Review M1 fix: `_validate_pairs_path` delegates to the shared helper."""
        src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "commands" / "steer.py"
        text = src.read_text(encoding="utf-8")
        assert "enforce_under_cwd_and_no_symlink" in text

    def test_version_string_is_at_least_v0620(self):
        """Floor-check: v0.62.0 features must remain present.

        Widened from exact-match in v0.63.0 (matches v0.56.0 / v0.51.0
        floor-check pattern so future releases don't regress this guard).
        """
        import soup_cli

        parts = soup_cli.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
        assert (major, minor) >= (0, 62), soup_cli.__version__

    def test_pyproject_version_is_at_least_v0620(self):
        """Floor-check the pyproject version against v0.62.0."""
        import re

        proj = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = proj.read_text(encoding="utf-8")
        m = re.search(r'^version = "(\d+)\.(\d+)\.(\d+)"', text, flags=re.MULTILINE)
        assert m is not None, "version line not found in pyproject.toml"
        major, minor = int(m.group(1)), int(m.group(2))
        assert (major, minor) >= (0, 62), (major, minor)
