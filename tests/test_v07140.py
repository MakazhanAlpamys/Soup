"""Tests for v0.71.40 — `soup reward synth` + comma-split reward loader (#311).

Covers:
- utils/reward_synth.py: extract_golds / detect_kind / inducers / render / calibrate / synthesize.
- commands/reward.py: `soup reward synth` CLI (happy / refuse / plan-only / guards).
- trainer/rewards.load_reward_fns + grpo._select_reward_fn comma-split (#311).
- config reward_fn validator; envs docstring fix.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from soup_cli.utils import reward_synth as rs


# ---------------------------------------------------------------------------
# extract_golds
# ---------------------------------------------------------------------------
class TestExtractGolds:
    def test_answer_field(self):
        rows = [{"prompt": "q", "answer": "5"}, {"prompt": "q2", "answer": 7}]
        assert rs.extract_golds(rows, field="answer") == ["5", "7"]

    def test_chat_last_assistant(self):
        rows = [
            {"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "gold-out"}]}
        ]
        assert rs.extract_golds(rows, field="answer") == ["gold-out"]

    def test_dict_gold_is_json_serialised(self):
        rows = [{"answer": {"k": 1}}]
        assert rs.extract_golds(rows, field="answer") == ['{"k": 1}']

    def test_no_resolvable_gold_raises(self):
        with pytest.raises(ValueError, match="no gold"):
            rs.extract_golds([{"prompt": "q"}], field="answer")

    def test_rows_must_be_sequence(self):
        with pytest.raises(TypeError):
            rs.extract_golds("not-a-list", field="answer")


# ---------------------------------------------------------------------------
# detect_kind
# ---------------------------------------------------------------------------
class TestDetectKind:
    def test_numeric(self):
        assert rs.detect_kind(["5", "12", "-3"]) == "numeric"

    def test_json_schema(self):
        assert rs.detect_kind(['{"a": 1}', '{"a": 2, "b": 3}']) == "json_schema"

    def test_tool_call_precedes_json_schema(self):
        golds = ['{"name": "search", "arguments": {"q": "x"}}',
                 '{"name": "lookup", "arguments": {"id": 3}}']
        assert rs.detect_kind(golds) == "tool_call"

    def test_regex_last_resort(self):
        # Same-length structured strings, not numbers, not JSON.
        assert rs.detect_kind(["2031-01-02", "1999-12-31"]) == "regex"

    def test_uninferrable_returns_none(self):
        assert rs.detect_kind(["hello world", "a completely different sentence here"]) is None


# ---------------------------------------------------------------------------
# inducers
# ---------------------------------------------------------------------------
class TestInduceNumeric:
    def test_int(self):
        spec = rs.induce_numeric(["1", "2", "3"])
        assert spec.is_float is False and spec.tolerance == 0.0

    def test_float(self):
        spec = rs.induce_numeric(["1.5", "2"])
        assert spec.is_float is True and spec.tolerance == rs.DEFAULT_NUMERIC_TOLERANCE

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="finite"):
            rs.induce_numeric(["1.0"], tolerance=-1)


class TestInduceJsonSchema:
    def test_keys_types_required(self):
        schema = rs.induce_json_schema(['{"a": 1, "b": "x"}', '{"a": 2}'])
        assert schema["type"] == "object"
        assert set(schema["properties"]) == {"a", "b"}
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["required"] == ["a"]  # b missing from row 2

    def test_array(self):
        schema = rs.induce_json_schema(["[1, 2]", "[3]"])
        assert schema["type"] == "array"


class TestInduceToolCall:
    def test_names_and_arg_keys(self):
        golds = ['{"name": "search", "arguments": {"q": "x"}}',
                 '{"name": "lookup", "arguments": {"id": 3}}']
        spec = rs.induce_tool_call(golds)
        assert set(spec.names) == {"search", "lookup"}
        assert set(spec.arg_keys) == {"q", "id"}

    def test_per_tool_required_and_allowed(self):
        # search always has q (required); lookup varies id/limit (id required only if
        # in every lookup call). allowed = union per name.
        golds = ['{"name": "search", "arguments": {"q": "a"}}',
                 '{"name": "search", "arguments": {"q": "b", "page": 2}}',
                 '{"name": "lookup", "arguments": {"id": 1}}']
        spec = rs.induce_tool_call(golds)
        assert spec.tools["search"]["required"] == ("q",)
        assert set(spec.tools["search"]["allowed"]) == {"q", "page"}
        assert spec.tools["lookup"]["required"] == ("id",)

    def test_no_tool_calls_raises(self):
        with pytest.raises(ValueError, match="no tool-call"):
            rs.induce_tool_call(["just some text", "5"])


class TestInduceRegex:
    def test_confident(self):
        pat = rs.induce_regex(["2031-01-02", "1999-12-31"])
        import re
        assert pat is not None and re.fullmatch(pat, "2000-05-06")

    def test_not_confident_returns_none(self):
        assert rs.induce_regex(["ab", "abcde", "x"]) is None


# ---------------------------------------------------------------------------
# render + round-trip
# ---------------------------------------------------------------------------
def _load_reward_fn(source: str, tmp_path: Path):
    p = tmp_path / "gen_reward.py"
    p.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gen_reward", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reward_fn


def _completions(*contents):
    return [[{"role": "assistant", "content": c}] for c in contents]


class TestRenderVerifier:
    def test_compiles_and_defines_reward_fn(self, tmp_path):
        src = rs.render_verifier_py("numeric", rs.NumericSpec(False, 0.0), meta={"n_refs": 3})
        compile(src, "<gen>", "exec")  # syntactically valid
        fn = _load_reward_fn(src, tmp_path)
        assert callable(fn)

    def test_numeric_roundtrip(self, tmp_path):
        src = rs.render_verifier_py("numeric", rs.NumericSpec(False, 0.0), meta={"n_refs": 2})
        fn = _load_reward_fn(src, tmp_path)
        out = fn(_completions("The answer is 5", "42"), answer=["5", "7"])
        assert out == [1.0, 0.0]

    def test_json_schema_roundtrip(self, tmp_path):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        src = rs.render_verifier_py("json_schema", schema, meta={"n_refs": 2})
        fn = _load_reward_fn(src, tmp_path)
        out = fn(_completions('{"a": 1}', "not json"), answer=["", ""])
        assert out == [1.0, 0.0]

    def test_tool_call_roundtrip(self, tmp_path):
        spec = rs.ToolCallSpec(tools={"search": {"required": ("q",), "allowed": ("q",)}})
        src = rs.render_verifier_py("tool_call", spec, meta={"n_refs": 1})
        fn = _load_reward_fn(src, tmp_path)
        out = fn(_completions('{"name": "search", "arguments": {"q": "x"}}',
                              '{"name": "evil", "arguments": {}}'), answer=["", ""])
        assert out == [1.0, 0.0]

    def test_tool_call_binding_rejects_hacks(self, tmp_path):
        # The code-review H1: two tools; a call must not borrow the other's arg key
        # or drop a required one. Union-of-all-keys + subset-check would accept both.
        spec = rs.induce_tool_call(['{"name": "search", "arguments": {"q": "x"}}',
                                    '{"name": "lookup", "arguments": {"id": 3}}'])
        fn = _load_reward_fn(rs.render_verifier_py("tool_call", spec, meta={"n_refs": 2}),
                             tmp_path)
        out = fn(_completions(
            '{"name": "search", "arguments": {"q": "y"}}',   # ok
            '{"name": "search", "arguments": {}}',           # missing required q
            '{"name": "search", "arguments": {"id": 3}}',    # borrowed lookup's key
        ), answer=["", "", ""])
        assert out == [1.0, 0.0, 0.0]

    def test_numeric_boxed_beats_trailing_number(self, tmp_path):
        # M1: a \boxed answer wins over an incidental trailing number.
        src = rs.render_verifier_py("numeric", rs.NumericSpec(False, 0.0), meta={"n_refs": 1})
        fn = _load_reward_fn(src, tmp_path)
        out = fn(_completions(r"So \boxed{4}. This took 12 seconds."), answer=["4"])
        assert out == [1.0]

    def test_numeric_bigint_exact(self, tmp_path):
        # M1: two 17-digit ints that collide under float() must NOT match at tol 0.
        src = rs.render_verifier_py("numeric", rs.NumericSpec(False, 0.0), meta={"n_refs": 1})
        fn = _load_reward_fn(src, tmp_path)
        out = fn(_completions("90071992547409915"), answer=["90071992547409916"])
        assert out == [0.0]

    def test_json_schema_int_float_symmetric(self, tmp_path):
        # H2: gold key induced as integer; a completion with 5.0 must still match.
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        fn = _load_reward_fn(rs.render_verifier_py("json_schema", schema, meta={"n_refs": 1}),
                             tmp_path)
        out = fn(_completions('{"x": 5.0}'), answer=[""])
        assert out == [1.0]

    def test_json_schema_typeless_key_presence_only(self, tmp_path):
        # H2: a key with no agreed type (varied across refs) is presence-only.
        schema = {"type": "object", "properties": {"a": {}}, "required": ["a"]}
        fn = _load_reward_fn(rs.render_verifier_py("json_schema", schema, meta={"n_refs": 1}),
                             tmp_path)
        assert fn(_completions('{"a": "text"}', '{"a": 9}'), answer=["", ""]) == [1.0, 1.0]

    def test_baked_constant_reflects_spec(self):
        # Mutation-guard: a different tolerance must change the emitted source.
        a = rs.render_verifier_py("numeric", rs.NumericSpec(True, 0.001), meta={"n_refs": 1})
        b = rs.render_verifier_py("numeric", rs.NumericSpec(True, 0.999), meta={"n_refs": 1})
        assert "0.001" in a and "0.001" not in b

    def test_rel_hint_cannot_inject_code(self):
        # security: rel_hint (the -o path) is .format'd raw into the header
        # docstring, and the file is later exec'd — a triple-quote break must not
        # survive into the emitted source.
        evil = 'x\n"""\nimport os\nopen("PWNED", "w").close()\n"""'
        src = rs.render_verifier_py("numeric", rs.NumericSpec(False, 0.0),
                                    meta={"n_refs": 1, "rel_hint": evil})
        benign = rs.render_verifier_py("numeric", rs.NumericSpec(False, 0.0),
                                       meta={"n_refs": 1, "rel_hint": "reward.py"})
        compile(src, "<g>", "exec")  # still valid Python (no docstring break-out)
        # No EXTRA triple-quote injected (the hint can't close the docstring early)
        # and the quotes needed to make the payload executable are stripped.
        assert src.count('"""') == benign.count('"""')
        assert 'open("PWNED"' not in src

    def test_hostile_gold_stays_repr_safe(self):
        # Untrusted tool names/keys are baked via repr(); they cannot break out.
        spec = rs.ToolCallSpec(tools={
            'ev"l\n"""x': {"required": (), "allowed": ('a"\nb',)}})
        src = rs.render_verifier_py("tool_call", spec, meta={"n_refs": 1})
        compile(src, "<g>", "exec")


class TestInduceJsonSchemaEdges:
    def test_mixed_shapes_refused(self):
        with pytest.raises(ValueError, match="mix"):
            rs.induce_json_schema(['{"a": 1}', "[1, 2]"])

    def test_no_containers_refused(self):
        with pytest.raises(ValueError, match="no JSON"):
            rs.induce_json_schema(["hello", "world"])

    def test_inconsistent_key_type_is_typeless(self):
        schema = rs.induce_json_schema(['{"a": 1}', '{"a": "x"}'])
        assert schema["properties"]["a"] == {}  # no type constraint


class TestInduceNumericGuard:
    def test_non_numeric_refused(self):
        with pytest.raises(ValueError, match="not numeric"):
            rs.induce_numeric(["hello", "world"])


class TestCalibrateFloors:
    def test_hard_floor_refuses_at_zero_threshold(self):
        # M2: an always-accept verifier is refused even with min_discrimination=0.
        def always_one(completions, **kw):
            return [1.0] * len(completions)
        rep = rs.calibrate(always_one, ["a", "b"], ["x", "y"], min_discrimination=0.0)
        assert rep.refused is True and rep.discrimination <= 0.0

    def test_self_accept_floor(self):
        # H2: a verifier accepting only half its own refs is refused (floor 0.9).
        state = {"i": 0}
        def half(completions, **kw):
            out = []
            for _ in completions:
                out.append(1.0 if state["i"] % 2 == 0 else 0.0)
                state["i"] += 1
            return out
        rep = rs.calibrate(half, ["a", "b", "c", "d"], ["x"], min_discrimination=0.1)
        assert rep.refused is True and "references" in rep.reason


class TestLoadRewardFnsGuards:
    def test_none_friendly(self):
        from soup_cli.trainer.rewards import load_reward_fns
        with pytest.raises(ValueError, match="must be a string"):
            load_reward_fns(None)

    def test_blank_rejected(self):
        from soup_cli.trainer.rewards import load_reward_fns
        with pytest.raises(ValueError, match="blank"):
            load_reward_fns("   ")

    def test_duplicate_rejected(self):
        from soup_cli.trainer.rewards import load_reward_fns
        with pytest.raises(ValueError, match="twice"):
            load_reward_fns("accuracy,accuracy")


# ---------------------------------------------------------------------------
# perturb + calibrate
# ---------------------------------------------------------------------------
class TestPerturbNegatives:
    def test_negatives_differ_from_golds(self):
        golds = ["5", "12"]
        negs = rs.perturb_negatives(golds, "numeric")
        assert negs and all(n not in golds for n in negs)


class TestCalibrate:
    def test_always_one_verifier_refused(self):
        def always_one(completions, **kw):
            return [1.0] * len(completions)
        rep = rs.calibrate(always_one, ["a", "b"], ["x", "y"], min_discrimination=0.5)
        assert rep.refused is True and rep.discrimination == 0.0

    def test_good_verifier_passes(self):
        def exact(completions, **kw):
            answers = kw.get("answer", [])
            return [1.0 if c[-1]["content"] == a else 0.0
                    for c, a in zip(completions, answers)]
        rep = rs.calibrate(exact, ["a", "b"], ["neg1", "neg2"], min_discrimination=0.5)
        assert rep.refused is False and rep.pos_accept == 1.0 and rep.neg_accept == 0.0

    def test_discrimination_boundary_exact(self):
        # Accept ALL positives (clears the 0.9 self-accept floor) but half the
        # negatives → discrimination 0.5 == threshold → pass (not < threshold).
        phase = {"neg": False}

        def v(completions, **kw):
            if not phase["neg"]:
                phase["neg"] = True
                return [1.0] * len(completions)  # positives: all accepted
            return [1.0 if i % 2 == 0 else 0.0 for i in range(len(completions))]
        rep = rs.calibrate(v, ["a", "b"], ["x", "y"], min_discrimination=0.5)
        assert rep.pos_accept == 1.0 and rep.neg_accept == 0.5
        assert rep.discrimination == 0.5 and rep.refused is False


# ---------------------------------------------------------------------------
# synthesize (end-to-end pure)
# ---------------------------------------------------------------------------
class TestSynthesize:
    def test_numeric_from_calculator_shape(self):
        rows = [{"prompt": "2+2", "answer": "4"}, {"prompt": "3+3", "answer": "6"}]
        res = rs.synthesize(rows, field="answer", kind="auto")
        assert res.kind == "numeric"
        assert "def reward_fn" in res.source

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            rs.synthesize([{"answer": "1"}], field="answer", kind="bogus")


# ---------------------------------------------------------------------------
# CLI — soup reward synth
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, rows):
    import json as _json
    path.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")


class TestRewardSynthCli:
    def _runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    def test_help(self):
        from soup_cli.commands.reward import app
        res = self._runner().invoke(app, ["synth", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "reward" in res.output.lower()

    def test_numeric_happy(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"),
                     [{"prompt": "2+2", "answer": "4"}, {"prompt": "3+3", "answer": "6"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert Path("reward.py").exists()
        # The emitted verifier agrees with the golds.
        fn = _load_reward_fn(Path("reward.py").read_text(encoding="utf-8"), tmp_path)
        assert fn(_completions("4", "99"), answer=["4", "6"]) == [1.0, 0.0]

    def test_cant_induce_is_error_exit_1(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        # All golds empty → regex can't be induced → INDUCTION ERROR (exit 1),
        # distinct from a calibration refusal (exit 2). No file written.
        _write_jsonl(Path("refs.jsonl"),
                     [{"answer": ""}, {"answer": ""}, {"answer": ""}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--kind", "regex"])
        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert "regex" in res.output.lower()
        assert not Path("reward.py").exists()

    def test_refusal_deletes_file(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        # A numeric verifier with a huge tolerance accepts the perturbed (+9999)
        # negatives too → discrimination below the default 0.5 → REFUSED (exit 2),
        # and the emitted file is removed (no partial artifact left behind).
        _write_jsonl(Path("refs.jsonl"), [{"answer": str(i)} for i in range(8)])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--kind", "numeric", "--tolerance", "1e9"])
        assert res.exit_code == 2, (res.output, repr(res.exception))
        assert "discrimination" in res.output.lower() and "refus" in res.output.lower()
        assert not Path("reward.py").exists()

    def test_plan_only_writes_nothing(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--plan-only"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert not Path("reward.py").exists()
        # Spec-specific fields, not just the kind name in the panel title.
        assert "numeric" in res.output.lower() and "tolerance" in res.output.lower()
        assert "float=false" in res.output.lower().replace(" ", "")

    def test_output_must_be_py(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.txt"])
        assert res.exit_code == 1
        assert ".py" in res.output

    def test_overwrite_without_force_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        Path("reward.py").write_text("# existing\n", encoding="utf-8")
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py"])
        assert res.exit_code == 1
        assert "force" in res.output.lower()
        assert Path("reward.py").read_text(encoding="utf-8") == "# existing\n"

    def test_bad_kind_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--kind", "bogus"])
        assert res.exit_code == 1
        assert "kind" in res.output.lower()


# ---------------------------------------------------------------------------
# Blocking riders — #311 comma-split loader + reward_fn validator + envs docstring
# ---------------------------------------------------------------------------
class TestCommaSplitLoader:
    def test_two_names_two_callables(self):
        from soup_cli.trainer.rewards import load_reward_fns
        fns = load_reward_fns("accuracy,format")
        assert isinstance(fns, list) and len(fns) == 2
        assert {f.__name__ for f in fns} == {"accuracy_reward", "format_reward"}

    def test_single_name_single_element_list(self):
        from soup_cli.trainer.rewards import load_reward_fns
        fns = load_reward_fns("accuracy")
        assert isinstance(fns, list) and len(fns) == 1
        assert fns[0].__name__ == "accuracy_reward"

    def test_whitespace_around_segments(self):
        from soup_cli.trainer.rewards import load_reward_fns
        fns = load_reward_fns(" accuracy , format ")
        assert len(fns) == 2

    def test_empty_segment_raises(self):
        from soup_cli.trainer.rewards import load_reward_fns
        with pytest.raises(ValueError, match="empty"):
            load_reward_fns("accuracy,")

    def test_verifiable_with_domain(self):
        from soup_cli.trainer.rewards import load_reward_fns
        fns = load_reward_fns("verifiable,format", verifiable_domain="math")
        assert len(fns) == 2


class TestSelectRewardFn:
    def test_comma_split_returns_list(self):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.trainer.grpo import _select_reward_fn
        tcfg = TrainingConfig(reward_fn="accuracy,format")
        out = _select_reward_fn(tcfg, "cpu", False)
        assert isinstance(out, list) and len(out) == 2

    def test_single_returns_callable(self):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.trainer.grpo import _select_reward_fn
        tcfg = TrainingConfig(reward_fn="accuracy")
        out = _select_reward_fn(tcfg, "cpu", False)
        assert callable(out) and not isinstance(out, list)


class TestRewardFnValidator:
    def test_null_byte_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig
        with pytest.raises(ValidationError, match="null"):
            TrainingConfig(reward_fn="acc\x00uracy")

    def test_empty_comma_segment_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig
        with pytest.raises(ValidationError, match="empty"):
            TrainingConfig(reward_fn="accuracy,,format")

    def test_overlong_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig
        with pytest.raises(ValidationError, match="512"):
            TrainingConfig(reward_fn="a" * 600)

    def test_comma_combo_accepted(self):
        from soup_cli.config.schema import TrainingConfig
        assert TrainingConfig(reward_fn="accuracy,format").reward_fn == "accuracy,format"


class TestRewardFnTaskGate:
    def _cfg(self, task, reward_fn):
        from soup_cli.config.loader import load_config_from_string
        return load_config_from_string(
            f"base: sshleifer/tiny-gpt2\ntask: {task}\n"
            f"data:\n  train: d.jsonl\ntraining:\n  reward_fn: {reward_fn}\n"
        )

    def test_grpo_allows_comma(self):
        cfg = self._cfg("grpo", "accuracy,format")
        assert cfg.training.reward_fn == "accuracy,format"

    def test_ppo_rejects_comma(self):
        # load_config_from_string re-wraps pydantic's ValidationError as ValueError.
        with pytest.raises(ValueError, match="grpo"):
            self._cfg("ppo", "accuracy,format")

    def test_sft_rejects_comma(self):
        with pytest.raises(ValueError, match="grpo"):
            self._cfg("sft", "accuracy,format")

    def test_ppo_single_name_ok(self):
        cfg = self._cfg("ppo", "accuracy")
        assert cfg.training.reward_fn == "accuracy"


class TestReviewFixes:
    def test_null_gold_falls_through_to_messages(self):
        rows = [{"answer": None,
                 "messages": [{"role": "assistant", "content": "gold"}]}]
        assert rs.extract_golds(rows, field="answer") == ["gold"]

    def test_null_gold_only_skipped(self):
        with pytest.raises(ValueError, match="no gold"):
            rs.extract_golds([{"answer": None}], field="answer")

    def test_non_finite_tolerance_raises(self):
        with pytest.raises(ValueError, match="finite"):
            rs.induce_numeric(["1.0"], tolerance=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            rs.induce_numeric(["1.0"], tolerance=float("inf"))

    def test_cli_bad_min_discrimination(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}])
        res = CliRunner().invoke(app, ["synth", "refs.jsonl", "-o", "r.py",
                                       "--min-discrimination", "2.0"])
        assert res.exit_code == 1 and "min-discrimination" in res.output

    def test_cli_bad_tolerance(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}])
        res = CliRunner().invoke(app, ["synth", "refs.jsonl", "-o", "r.py",
                                       "--tolerance", "-1"])
        assert res.exit_code == 1 and "tolerance" in res.output


class TestPerKindPipeline:
    """Every kind runs the FULL write->load->calibrate pipeline (not just numeric)."""

    def _runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    def _synth(self, tmp_path, monkeypatch, rows, extra=()):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), rows)
        return self._runner().invoke(
            app, ["synth", "refs.jsonl", "-o", "reward.py", *extra])

    def test_json_schema_end_to_end(self, tmp_path, monkeypatch):
        rows = [{"answer": '{"city": "Paris", "pop": 2}'},
                {"answer": '{"city": "Rome", "pop": 3}'}]
        res = self._synth(tmp_path, monkeypatch, rows)
        assert res.exit_code == 0, (res.output, repr(res.exception))
        fn = _load_reward_fn(Path("reward.py").read_text(encoding="utf-8"), tmp_path)
        assert fn(_completions('{"city": "X", "pop": 9}', "nope"),
                  answer=["", ""]) == [1.0, 0.0]

    def test_tool_call_end_to_end(self, tmp_path, monkeypatch):
        rows = [{"answer": '{"name": "search", "arguments": {"q": "a"}}'},
                {"answer": '{"name": "search", "arguments": {"q": "b"}}'}]
        res = self._synth(tmp_path, monkeypatch, rows)
        assert res.exit_code == 0, (res.output, repr(res.exception))
        fn = _load_reward_fn(Path("reward.py").read_text(encoding="utf-8"), tmp_path)
        assert fn(_completions('{"name": "search", "arguments": {"q": "z"}}',
                               '{"name": "search", "arguments": {}}'),
                  answer=["", ""]) == [1.0, 0.0]

    def test_regex_end_to_end(self, tmp_path, monkeypatch):
        rows = [{"answer": "2031-01-02"}, {"answer": "1999-12-31"},
                {"answer": "2000-05-06"}]
        res = self._synth(tmp_path, monkeypatch, rows, extra=["--kind", "regex"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        fn = _load_reward_fn(Path("reward.py").read_text(encoding="utf-8"), tmp_path)
        assert fn(_completions("2020-11-11", "not a date"), answer=["", ""]) == [1.0, 0.0]


class TestPerturbNegativesPerKind:
    def test_tool_call_generates_valid_json_hacks(self):
        golds = ['{"name": "search", "arguments": {"q": "x"}}']
        negs = rs.perturb_negatives(golds, "tool_call")
        # Must include a VALID-JSON wrong-name negative and a foreign-arg negative,
        # not just the syntactically-invalid string.
        assert any('"__nonexistent_tool__"' in n for n in negs)
        assert any("__foreign_arg__" in n for n in negs)

    def test_json_schema_negatives(self):
        negs = rs.perturb_negatives(['{"a": 1}'], "json_schema")
        assert negs and all(n for n in negs[:1])  # non-empty invalid-json negative

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown kind"):
            rs.perturb_negatives(["x"], "bogus")


class TestOutputReport:
    def _runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    def test_report_written_and_parseable(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--output-report", "rep.json"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        import json as _json
        rep = _json.loads(Path("rep.json").read_text(encoding="utf-8"))
        assert rep["kind"] == "numeric" and rep["refused"] is False
        assert set(rep) >= {"pos_accept", "neg_accept", "discrimination", "precision"}

    def test_bad_report_path_fails_before_writing_verifier(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--output-report", "../escape.json"])
        assert res.exit_code == 1, (res.output, repr(res.exception))
        # The verifier is never written when the report path is rejected up front.
        assert not Path("reward.py").exists()


class TestCliRobustness:
    def _runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    def test_missing_references_file(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        res = self._runner().invoke(app, ["synth", "nope.jsonl", "-o", "reward.py"])
        assert res.exit_code == 1, (res.output, repr(res.exception))

    def test_empty_references_file(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        Path("refs.jsonl").write_text("\n\n", encoding="utf-8")
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py"])
        assert res.exit_code == 1 and "no JSON-object rows" in res.output

    def test_force_overwrites(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        Path("reward.py").write_text("# stale\n", encoding="utf-8")
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--force"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "# stale" not in Path("reward.py").read_text(encoding="utf-8")

    def test_custom_field(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"gold": "4"}, {"gold": "6"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "reward.py",
                                          "--field", "gold"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert Path("reward.py").exists()

    def test_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}])
        res = self._runner().invoke(app, ["synth", "refs.jsonl", "-o", "../evil.py"])
        assert res.exit_code == 1 and "cwd" in res.output.lower()

    def test_references_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app
        monkeypatch.chdir(tmp_path)
        res = self._runner().invoke(app, ["synth", "../secret.jsonl", "-o", "reward.py"])
        assert res.exit_code == 1 and "cwd" in res.output.lower()

    def test_registered_on_top_level_cli(self):
        from soup_cli.cli import app as root
        res = self._runner().invoke(root, ["reward", "synth", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))


class TestBoundaries:
    def test_self_accept_boundary_9_of_10(self):
        # 9/10 == 0.9 self-accept floor → NOT refused; 8/9 < 0.9 → refused.
        pos9 = ["p"] * 10
        seq = {"i": 0}

        def accept_first_9(completions, **kw):
            out = []
            for _ in completions:
                out.append(1.0 if seq["i"] < 9 else 0.0)
                seq["i"] += 1
            return out
        rep = rs.calibrate(accept_first_9, pos9, ["n"], min_discrimination=0.1)
        assert abs(rep.pos_accept - 0.9) < 1e-9 and rep.refused is False

    def test_self_accept_below_floor_refused(self):
        seq = {"i": 0}

        def accept_first_7(completions, **kw):  # 7/9 ~= 0.78 < 0.9
            out = []
            for _ in completions:
                out.append(1.0 if seq["i"] < 7 else 0.0)
                seq["i"] += 1
            return out
        rep = rs.calibrate(accept_first_7, ["p"] * 9, ["n"], min_discrimination=0.1)
        assert rep.pos_accept < rs._MIN_SELF_ACCEPT and rep.refused is True
        assert "own references" in rep.reason

    def test_numeric_tolerance_inclusive(self, tmp_path):
        src = rs.render_verifier_py("numeric", rs.NumericSpec(True, 0.1), meta={"n_refs": 1})
        fn = _load_reward_fn(src, tmp_path)
        # |5.1 - 5.0| == 0.1 == tol → inclusive match.
        assert fn(_completions("5.1"), answer=["5.0"]) == [1.0]
        assert fn(_completions("5.2"), answer=["5.0"]) == [0.0]

    def test_precision_value(self):
        def accept_all(completions, **kw):
            return [1.0] * len(completions)
        # 2 refs accepted (tp=2), 3 negatives all accepted (fp=3) → precision 2/5.
        rep = rs.calibrate(accept_all, ["a", "b"], ["x", "y", "z"])
        assert abs(rep.precision - 0.4) < 1e-9

    def test_detect_kind_numeric_confidence(self):
        # 9/10 numeric → numeric; 5/10 → not numeric (falls to regex/None).
        assert rs.detect_kind(["1", "2", "3", "4", "5", "6", "7", "8", "9", "x"]) == "numeric"

    def test_reward_fn_512_boundary(self):
        from soup_cli.config.schema import TrainingConfig
        assert TrainingConfig(reward_fn="a" * 512).reward_fn == "a" * 512
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="512"):
            TrainingConfig(reward_fn="a" * 513)


class TestVerifiableComboGate:
    def test_comma_verifiable_without_domain_rejected(self):
        # #8: "accuracy,verifiable" must fail at config parse (like bare
        # "verifiable"), not silently defer to a runtime crash.
        from soup_cli.config.loader import load_config_from_string
        with pytest.raises(ValueError, match="verifiable_domain"):
            load_config_from_string(
                "base: sshleifer/tiny-gpt2\ntask: grpo\ndata:\n  train: d.jsonl\n"
                "training:\n  reward_fn: accuracy,verifiable\n"
            )

    def test_comma_verifiable_with_domain_ok(self):
        from soup_cli.config.loader import load_config_from_string
        cfg = load_config_from_string(
            "base: sshleifer/tiny-gpt2\ntask: grpo\ndata:\n  train: d.jsonl\n"
            "training:\n  reward_fn: accuracy,verifiable\n  verifiable_domain: math\n"
        )
        assert cfg.training.reward_fn == "accuracy,verifiable"


class TestRewardFnValidatorBlank:
    def test_blank_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig
        with pytest.raises(ValidationError, match="blank"):
            TrainingConfig(reward_fn="   ")

    def test_bool_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig
        with pytest.raises(ValidationError, match="string"):
            TrainingConfig(reward_fn=True)


class TestEnvsDocstringFix:
    def test_calculator_docstring(self):
        from soup_cli.envs import calculator
        assert "reward_fn='math'" not in (calculator.__doc__ or "")
        assert "verifiable" in (calculator.__doc__ or "")

    def test_guess_number_docstring(self):
        from soup_cli.envs import guess_number
        assert "reward_fn='math'" not in (guess_number.__doc__ or "")
        assert "verifiable" in (guess_number.__doc__ or "")


class TestNoTopLevelTorch:
    def test_reward_synth_has_no_top_level_torch(self):
        src = Path(rs.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:  # module-level only
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(("torch", "transformers", "peft"))
                           for n in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("torch", "transformers", "peft"))
