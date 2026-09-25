"""Re-measurement protocol for the post-#331 8B NF4 laptop row (#361).

The published headline — Llama-3.1-8B-Instruct NF4 streamed at 119.6 tok/s
in a 3.32 GB peak on an RTX 3050 Laptop 4 GB — predates the #331 repair, and
no post-repair number exists yet. These pin the parts of the protocol that
are checkable without the card: arg validation order, the GFLOP/token
numerator convention, the sampler middle index, the seven protocol
constants themselves (steps, warm-up, seq, batch, buffers, LoRA r, targets),
and the fidelity guards on the one ``build_streamed_model`` call —
``pin=True`` and ``require_pin=True`` must both be present, and the
``runtime.source.pinned`` re-check must survive. A wrong number here is
worse than no number, so a protocol that can be silently edited into the
v0.72.0 pageable lower-bound shape is a protocol that will be.
No GPU, no model download: the GFLOP/token arm builds a small synthetic
safetensors checkpoint on CPU.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "benchmarks" / "harness" / "issue361_nf4_throughput.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("issue361_nf4_throughput", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_harness = _load_harness()


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """One decoder layer + embed + head with known element counts.

    decoder = q_proj (4x4) + k_proj (2x2) = 20 elements.
    vocab*hidden from config = 8*4 = 32, so GFLOP/token = 6*20 + 4*32 = 248.
    A tied config must give the same answer: lm_head-in-file is ignored.
    """

    pytest.importorskip("safetensors")
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text(
        json.dumps({"vocab_size": 8, "hidden_size": 4, "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    blob = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 2),
        "model.embed_tokens.weight": torch.zeros(8, 4),
        "lm_head.weight": torch.zeros(8, 4),
    }
    save_file(blob, str(weights / "model.safetensors"))
    return weights


class TestArgValidationBeatsTheCudaSkip:
    """A bad protocol flag must fail loudly, not pass as a skip."""

    def test_bad_seq_fails_even_without_cuda(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y", "--seq", "0"],
        )
        assert _harness.main() == 2
        assert "--seq" in capsys.readouterr().out

    def test_bad_buffers_fails_even_without_cuda(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y", "--buffers", "1"],
        )
        assert _harness.main() == 2
        assert "--buffers" in capsys.readouterr().out

    def test_valid_args_skip_without_cuda(self, monkeypatch, capsys) -> None:
        """CONTROL: the skip still works once validation passes."""
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y"],
        )
        assert _harness.main() == 0
        assert "intentional skip" in capsys.readouterr().out

    def test_warmup_boundary_zero_accepted_negative_refused(self, monkeypatch, capsys) -> None:
        """--warmup 0 is a legal protocol choice (measure immediately); -1
        must be refused as a bad flag, never reach the card. The labelled
        warm-up banner itself is card-side, so only the boundary shows here."""
        argv = ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y"]
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(sys, "argv", [*argv, "--warmup", "-1"])
        assert _harness.main() == 2
        assert "--warmup" in capsys.readouterr().out
        monkeypatch.setattr(sys, "argv", [*argv, "--warmup", "0"])
        assert _harness.main() == 0
        assert "intentional skip" in capsys.readouterr().out

    def test_zero_lora_r_is_refused_before_the_card(self, monkeypatch, capsys) -> None:
        """Left unchecked it fails after the sharding — minutes into a run on
        the 4 GB laptop the protocol exists for."""
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y", "--lora-r", "0"],
        )
        assert _harness.main() == 2
        assert "--lora-r" in capsys.readouterr().out

    def test_empty_targets_is_refused_before_the_card(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y", "--targets", ""],
        )
        assert _harness.main() == 2
        assert "--targets" in capsys.readouterr().out

    def test_a_nonempty_target_list_is_accepted(self, monkeypatch, capsys) -> None:
        """CONTROL: the guard must reject empty, not 'not the default'."""
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y", "--targets", " "],
        )
        assert _harness.main() == 2
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "issue361_nf4_throughput.py",
                "--weights",
                "x",
                "--shards",
                "y",
                "--targets",
                "q_proj",
            ],
        )
        assert _harness.main() == 0
        assert "intentional skip" in capsys.readouterr().out


class TestTheRowNamesItsTree:
    """A release version does not move between commits, so a row without the
    commit cannot say which tree produced it — and the tree matters here: three
    commits after this branch's base changed the path being measured."""

    def test_source_sha_is_a_commit_or_an_honest_unknown(self) -> None:
        sha = _harness._source_sha()
        assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)

    def test_the_versions_blob_carries_it(self) -> None:
        assert _harness._versions()["commit"] == _harness._source_sha()

    def test_the_printed_row_and_the_json_blob_both_name_it(self) -> None:
        text = _HARNESS.read_text(encoding="utf-8")
        assert "source commit:" in text
        assert '"versions": versions' in text


class TestNumeratorConvention:
    def test_tied_head_is_not_double_counted(self, tmp_path: Path) -> None:
        weights = _tiny_checkpoint(tmp_path)
        split = _harness.read_param_split(str(weights))
        assert split["decoder"] == 20
        assert split["embed"] == 32
        assert split["tied"] is True
        # The numerator is 6*20 + 4*32 = 248 GFLOP*1e-9 exactly; every operand
        # is a small integer, so there is no rounding to absorb and the
        # equality is meant literally.
        assert _harness.gflop_per_token(split) == 248 / 1e9

    def test_lm_head_effective_uses_config_not_file(self, tmp_path: Path) -> None:
        """A checkpoint whose file head differs from vocab*hidden still uses config."""
        weights = _tiny_checkpoint(tmp_path)
        split = _harness.read_param_split(str(weights))
        assert split["lm_head_file"] == 32
        assert split["lm_head_effective"] == 8 * 4


class TestSamplerMiddleIndex:
    def test_even_sample_set_reports_the_lower_median(self) -> None:
        sampler = _harness.GpuSampler()
        sampler.samples = [(100, 900, 60), (100, 950, 65)]
        summary = sampler.summary()
        assert (summary["util"], summary["clock"], summary["temp_max"]) == (100, 900, 65)

    def test_empty_sample_set_reports_no_reading(self) -> None:
        assert _harness.GpuSampler().summary()["n"] == 0


class TestJsonEvidence:
    """A 60-step 8B run on a 4 GB card is exactly the run that dies midway,
    so the row (or the partial evidence) must land on disk. The skip writes
    nothing: an intentional skip is not a row of zeros."""

    def test_dump_json_creates_parents_and_writes_the_payload(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "row.json"
        _harness._dump_json(str(target), {"status": "ok", "issue": 361})
        assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok", "issue": 361}

    def test_the_skip_writes_no_json_even_when_asked(self, monkeypatch, tmp_path: Path) -> None:
        target = tmp_path / "row.json"
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "issue361_nf4_throughput.py",
                "--weights",
                "x",
                "--shards",
                "y",
                "--json",
                str(target),
            ],
        )
        assert _harness.main() == 0
        assert not target.exists()

    def test_a_dead_run_leaves_what_it_got(self) -> None:
        row = _harness._partial_row("measuring", [0.5, 0.25], [3.0, 2.5], {"commit": "abc"})
        assert row["status"] == "measuring"
        assert row["steps_completed"] == 2
        assert row["step_times_s"] == [0.5, 0.25]
        assert row["losses"] == [3.0, 2.5]
        assert row["versions"] == {"commit": "abc"}

    def test_the_measured_loop_writes_it_after_every_step(self) -> None:
        """A hard kill (OOM, power cut) runs no handler, so the flush has to be
        inside the loop rather than only after it."""
        assert (
            '_partial_row("measuring", step_times, losses, versions)'
            in _HARNESS.read_text(encoding="utf-8")
        )


_PROTOCOL_CONSTANTS = {
    "DEFAULT_SEQ": 512,
    "DEFAULT_BATCH": 1,
    "DEFAULT_WARMUP": 10,
    "DEFAULT_STEPS": 50,
    "DEFAULT_BUFFERS": 2,
    "DEFAULT_LORA_R": 16,
    "DEFAULT_TARGETS": "q_proj,v_proj",
}


class TestProtocolConstantsAreThePublishedProtocol:
    """The gate's step-6 protocol, as module constants.

    These are intentionally compared to literals, not to the module's own
    values: the point is that flipping the constant fails here, so the
    defaults in --help can never quietly stop being the published protocol.
    A deliberate deviation is a diff to this table plus a note in the gate
    file — visible, not silent.
    """

    @pytest.mark.parametrize(("name", "expected"), sorted(_PROTOCOL_CONSTANTS.items()))
    def test_constant_matches_the_published_protocol(self, name: str, expected) -> None:
        assert getattr(_harness, name) == expected

    def test_seed_constants_are_stable(self) -> None:
        assert _harness.SEED == 3
        assert _harness.INPUT_SEED == 17

    def test_defaults_are_what_the_parser_serves(self, monkeypatch) -> None:
        """The parser must serve the constants, not a second copy of them."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["issue361_nf4_throughput.py", "--weights", "x", "--shards", "y"],
        )
        args = _harness.parse_args()
        assert args.seq == _harness.DEFAULT_SEQ
        assert args.batch == _harness.DEFAULT_BATCH
        assert args.warmup == _harness.DEFAULT_WARMUP
        assert args.steps == _harness.DEFAULT_STEPS
        assert args.buffers == _harness.DEFAULT_BUFFERS
        assert args.lora_r == _harness.DEFAULT_LORA_R
        assert args.targets == _harness.DEFAULT_TARGETS


_HARNESS_AST = ast.parse(_HARNESS.read_text(encoding="utf-8"))

# Sentinel for a keyword whose value is computed rather than written as a
# literal in the source. The pin tests demand `is True`, so an expression
# standing in for pin/require_pin fails them even though the key exists.
_NON_LITERAL = object()


def _kwarg_values_for(call: ast.Call, func_name: str) -> dict:
    assert isinstance(call.func, ast.Name) and call.func.id == func_name, (
        f"expected a call to {func_name}, found {ast.dump(call.func)[:80]}"
    )
    values = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue  # **unpacking: nothing to read statically
        try:
            values[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            values[kw.arg] = _NON_LITERAL
    return values


def _build_streamed_model_kwargs() -> dict:
    for node in ast.walk(_HARNESS_AST):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_streamed_model"
        ):
            values = _kwarg_values_for(node, "build_streamed_model")
            if values:
                return values
    raise AssertionError("no build_streamed_model(...) call with keyword args found")


class TestThePinGuardIsWired:
    """A silently pageable store IS the v0.72.0 lower-bound shape.

    The run is only comparable to the published row if the store page-locked
    or the harness refused. Deleting either kwarg turns the headline guard
    into prose, so the source itself is pinned here — the runtime refuses a
    required pin only if the kwarg reaches it.
    """

    def test_the_guard_is_a_source_literal_not_a_flag_expression(self) -> None:
        """``pin``/``require_pin`` must be written as literals in the call.

        ``_kwarg_values_for`` marks a computed value with a sentinel instead
        of a literal, so ``is True`` fails for any expression — no CLI flag
        or computed boolean can stand in for the guard at the call site.
        """
        kwargs = _build_streamed_model_kwargs()
        assert kwargs["pin"] is True
        assert kwargs["require_pin"] is True

    def test_the_source_recheck_survives(self) -> None:
        """`assert runtime.source.pinned` must still stand behind the kwarg."""
        assert "assert runtime.source.pinned" in _HARNESS.read_text(encoding="utf-8")


class TestGateRecordCarriesThePendingLabel:
    """The gate file is the row's future home: it must keep saying the
    headline predates the #331 repair and name this harness as the protocol,
    so a number pasted there before the card run has something to contradict.

    Since 2026-09-20 the reservation is satisfied, so this class now pins
    the pre-registration as history: the pending wording stays verbatim
    (marked satisfied) beside the measured row rather than being rewritten.
    """

    def test_the_pending_label_and_harness_pointer_are_present(self) -> None:
        gate = _REPO_ROOT / "benchmarks" / "gate-v0.72.2-nf4.md"
        text = gate.read_text(encoding="utf-8")
        assert "predates the #331 repair" in text
        assert "Remaining row pending" in text
        assert "harness/issue361_nf4_throughput.py" in text

    def test_the_pending_row_names_owner_date_and_direction(self) -> None:
        """A pending row with a stated expectation is a prediction; without one
        it is a placeholder. The three markers below are what make it the
        former, so dropping any of them is the failure this catches."""
        gate = (_REPO_ROOT / "benchmarks" / "gate-v0.72.2-nf4.md").read_text(encoding="utf-8")
        assert "Owed by @umran666" in gate
        assert "2026-09-22" in gate
        assert "Expected direction:" in gate
