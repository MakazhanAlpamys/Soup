"""Contract tests for soup compile and compile-tools optimizer integration (#1810)."""

import json
import re
import sys
import types

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


@pytest.fixture
def recording_dspy(monkeypatch):
    calls = []

    class _Recorder:
        def __init__(self, **kwargs):
            calls.append((type(self).__name__, kwargs))

        def compile(self, program, **kwargs):
            calls.append(("compile", sorted(kwargs)))
            return program

    class _MIPROv2:
        def __init__(self, metric, auto=None, num_candidates=None, **kwargs):
            if "num_trials" in kwargs:
                raise TypeError(
                    "MIPROv2.__init__() got an unexpected keyword argument 'num_trials'"
                )
            self.metric = metric
            self.auto = auto
            self.num_candidates = num_candidates
            calls.append(("MIPROv2", {
                "metric": metric, "auto": auto, "num_candidates": num_candidates,
            }))

        def compile(self, student, trainset=None, num_trials=None, **kwargs):
            calls.append(("compile", {"num_trials": num_trials, "trainset": trainset}))
            return types.SimpleNamespace(
                program=student, score=0.85, total_calls=num_trials, converged=True
            )

    class _Example:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def with_inputs(self, *keys):
            return self

    fake_dspy = types.SimpleNamespace(
        MIPROv2=_MIPROv2,
        COPRO=type("COPRO", (_Recorder,), {}),
        BootstrapFewShot=type("BootstrapFewShot", (_Recorder,), {}),
        Example=_Example,
    )
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    return calls


@pytest.fixture
def recording_textgrad(monkeypatch):
    calls = []

    class _Variable:
        def __init__(self, value, *, role_description: str, predecessors=None, **kwargs):
            self.value = value
            self.role_description = role_description
            self.predecessors = list(predecessors or [])
            self.gradients = set()
            calls.append(("Variable", value))

        def backward(self, *args, **kwargs):
            for predecessor in self.predecessors:
                predecessor.gradients.add(f"grad_from_{self.value}")
            calls.append(("backward",))

    class _TGD:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters
            calls.append(("TGD", sorted(kwargs)))

        def zero_grad(self):
            for parameter in self.parameters:
                parameter.gradients.clear()
            calls.append(("zero_grad",))

        def step(self):
            calls.append(("step", [len(v.gradients) for v in self.parameters]))
            for parameter in self.parameters:
                if len(parameter.gradients) > 0:
                    parameter.value += " improved"

    fake_tg = types.SimpleNamespace(Variable=_Variable, TGD=_TGD)
    monkeypatch.setitem(sys.modules, "textgrad", fake_tg)
    return calls


@pytest.fixture
def recording_gepa(monkeypatch):
    calls = []

    def fake_optimize(*, seed_candidate, trainset, **kwargs):
        assert isinstance(seed_candidate, dict), (
            f"seed_candidate must be dict, got {type(seed_candidate)}"
        )
        assert "task_lm" in kwargs or "adapter" in kwargs, "GEPA requires task_lm or adapter"
        if kwargs.get("adapter") is not None and kwargs.get("task_lm") is not None:
            raise ValueError("GEPA requires task_lm=None in adapter mode")
        calls.append(("optimize", {"seed_candidate": seed_candidate, "kwargs": kwargs}))
        return types.SimpleNamespace(
            best_candidate={
                "prompt": "optimised prompt text",
                "description": "optimised tool desc",
            },
            best_score=0.95,
        )

    fake_gepa = types.SimpleNamespace(optimize=fake_optimize)
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa)
    return calls


def test_dspy_mipro_max_iters_and_trials_contract(tmp_path, monkeypatch, recording_dspy):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "2+2", "answer": "4"}\n', encoding="utf-8")

    runner = CliRunner()
    res1 = runner.invoke(
        app,
        [
            "compile",
            "prog.py",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "mipro",
            "--max-iters",
            "1",
            "-o",
            "out1.py",
        ],
    )
    assert res1.exit_code == 0, res1.output
    calls1 = list(recording_dspy)

    recording_dspy.clear()
    res2 = runner.invoke(
        app,
        [
            "compile",
            "prog.py",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "mipro",
            "--max-iters",
            "1000",
            "-o",
            "out2.py",
        ],
    )
    assert res2.exit_code == 0, res2.output
    calls2 = list(recording_dspy)

    # 1. auto is None; num_trials belongs to compile, not the constructor
    mipro_calls1 = [c for c in calls1 if c[0] == "MIPROv2"]
    mipro_calls2 = [c for c in calls2 if c[0] == "MIPROv2"]
    assert len(mipro_calls1) == 1
    assert len(mipro_calls2) == 1

    assert mipro_calls1[0][1].get("auto") is None
    assert mipro_calls2[0][1].get("auto") is None
    assert "num_trials" not in mipro_calls1[0][1]
    assert "num_trials" not in mipro_calls2[0][1]
    assert mipro_calls1[0][1]["num_candidates"] == 1
    assert mipro_calls2[0][1]["num_candidates"] == 10
    assert next(c[1] for c in calls1 if c[0] == "compile")["num_trials"] == 1
    assert next(c[1] for c in calls2 if c[0] == "compile")["num_trials"] == 1000

    # 2. --max-iters 1 and --max-iters 1000 produce different arguments
    assert mipro_calls1[0][1] != mipro_calls2[0][1]


def test_dspy_metric_never_none(tmp_path, monkeypatch, recording_dspy):
    monkeypatch.chdir(tmp_path)
    # Program module has no metric attribute
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "2+2", "answer": "4"}\n', encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        app, ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "mipro", "-o", "out.py"]
    )
    assert res.exit_code == 0, res.output

    for name, kwargs in recording_dspy:
        if name in ("MIPROv2", "COPRO", "BootstrapFewShot"):
            metric = kwargs.get("metric")
            assert metric is not None, f"{name} received metric=None"
            assert callable(metric), f"{name} metric must be callable"


def test_textgrad_backward_loss_and_engine_contract(tmp_path, monkeypatch, recording_textgrad):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "math.jsonl").write_text(
        '{"question": "1+1", "answer": "2"}\n{"question": "2+2", "answer": "4"}\n', encoding="utf-8"
    )
    (tmp_path / "geo.jsonl").write_text(
        '{"question": "capital?", "answer": "Paris"}\n{"question": "currency?", "answer": "EUR"}\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    res1 = runner.invoke(
        app,
        [
            "compile",
            "prog.py",
            "--eval",
            "math.jsonl",
            "--optimizer",
            "textgrad",
            "--max-iters",
            "2",
            "-o",
            "out1.py",
        ],
    )
    assert res1.exit_code == 0, res1.output
    calls1 = list(recording_textgrad)

    recording_textgrad.clear()
    res2 = runner.invoke(
        app,
        [
            "compile",
            "prog.py",
            "--eval",
            "geo.jsonl",
            "--optimizer",
            "textgrad",
            "--max-iters",
            "2",
            "-o",
            "out2.py",
        ],
    )
    assert res2.exit_code == 0, res2.output
    calls2 = list(recording_textgrad)

    # 1. Optimizer receives an engine
    tgd_calls = [c for c in calls1 if c[0] == "TGD"]
    assert len(tgd_calls) == 1
    assert "engine" in tgd_calls[0][1], "TGD did not receive engine"

    # 2. Each step is preceded by a backward()
    for i, call in enumerate(calls1):
        if call[0] == "step":
            assert i > 0 and calls1[i - 1][0] == "backward", (
                f"step at {i} not preceded by backward: {calls1}"
            )

    # 3. Two different suites produce different call sequences
    assert calls1 != calls2


def test_result_panel_reports_actual_convergence_and_score(
    tmp_path, monkeypatch, recording_textgrad
):
    monkeypatch.chdir(tmp_path)
    # Text edits alone must not imply convergence.
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "1+1", "answer": "2"}\n', encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compile",
            "prog.py",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "textgrad",
            "--max-iters",
            "3",
            "-o",
            "out.py",
        ],
    )
    assert res.exit_code == 0, res.output
    stripped = _strip_ansi(res.output)
    panel = dict(re.findall(r"(Iterations|Converged|Score):\s+(\S+)", stripped))
    assert panel.get("Converged") == "False"
    assert panel.get("Iterations") == "3"
    assert panel.get("Score") == "0.000"


def test_gepa_seed_candidate_dict_and_string_output(tmp_path, monkeypatch, recording_gepa):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "1+1", "answer": "2"}\n', encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compile",
            "prog.py",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "gepa",
            "--max-iters",
            "5",
            "-o",
            "out.py",
        ],
    )
    assert res.exit_code == 0, res.output

    assert len(recording_gepa) == 1
    call_name, call_dict = recording_gepa[0]
    seed = call_dict["seed_candidate"]
    assert isinstance(seed, dict)
    assert not isinstance(seed, str)

    # Output file holds the string text, not a dict repr "{'prompt': ...}"
    written = (tmp_path / "out.py").read_text(encoding="utf-8")
    assert written == "optimised prompt text"
    assert not written.startswith("{")


def test_compile_tools_textgrad_backward_contract(tmp_path, monkeypatch, recording_textgrad):
    monkeypatch.chdir(tmp_path)
    spec_json = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0"},
        "paths": {
            "/search": {
                "get": {
                    "operationId": "search_items",
                    "description": "Search item catalog",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec_json), encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(
        '{"query": "shoes", "target": "search_items"}\n', encoding="utf-8"
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compile-tools",
            "spec.json",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "textgrad",
            "-o",
            "out.json",
        ],
    )
    assert res.exit_code == 0, res.output

    backward_calls = [c for c in recording_textgrad if c[0] == "backward"]
    assert len(backward_calls) > 0, "compile-tools textgrad did not invoke backward()"
    assert all(c[1] == [1] for c in recording_textgrad if c[0] == "step")
    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert written["tools"][0]["description"] == "Search item catalog improved"


def test_dspy_missing_metric_without_suite_refuses_cleanly(tmp_path, monkeypatch, recording_dspy):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        app, ["compile", "prog.py", "--eval", "empty.jsonl", "--optimizer", "mipro", "-o", "out.py"]
    )
    assert res.exit_code == 2
    assert "metric" in res.output


def test_textgrad_bare_exception_exits_2_with_clean_message(tmp_path, monkeypatch):
    class _FailingTGD:
        def __init__(self, *args, **kwargs):
            raise Exception("No engine provided. Either pass an engine or set backward engine.")

    class _Variable:
        def __init__(self, value, **kwargs):
            self.value = value

    fake_tg = types.SimpleNamespace(Variable=_Variable, TGD=_FailingTGD)
    monkeypatch.setitem(sys.modules, "textgrad", fake_tg)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "1+1", "answer": "2"}\n', encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "textgrad", "-o", "out.py"],
    )
    assert res.exit_code == 2
    assert "No engine provided" in res.output

    spec_json = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0"},
        "paths": {"/search": {"get": {"operationId": "search", "description": "Search"}}},
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec_json), encoding="utf-8")
    res_tools = runner.invoke(
        app,
        [
            "compile-tools",
            "spec.json",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "textgrad",
            "-o",
            "out.json",
        ],
    )
    assert res_tools.exit_code == 2
    assert "No engine provided" in res_tools.output


def test_gepa_adapter_mode_passes_task_lm_none(tmp_path, monkeypatch, recording_gepa):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nadapter = 'my_lora_adapter'\ntask_lm = 'ignored_lm'\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    res = CliRunner().invoke(
        app, ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "gepa", "-o", "out.py"]
    )
    assert res.exit_code == 0, res.output
    kwargs = recording_gepa[0][1]["kwargs"]
    assert kwargs["adapter"] == "my_lora_adapter"
    assert kwargs["task_lm"] is None


def test_textgrad_edit_without_score_improvement_does_not_claim_converged(
    tmp_path, monkeypatch, recording_textgrad
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\ndef metric(gold, pred):\n    return 0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    res = CliRunner().invoke(app, [
        "compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "textgrad",
        "--max-iters", "2", "-o", "out.py",
    ])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out.py").read_text(encoding="utf-8") == (
        "Answer briefly. improved improved"
    )
    panel = dict(re.findall(r"(Converged|Score):\s+(\S+)", _strip_ansi(res.output)))
    assert panel == {"Converged": "False", "Score": "0.000"}


def test_textgrad_loss_predecessor_graph_populates_gradients(
    tmp_path, monkeypatch, recording_textgrad
):
    from soup_cli.utils import prompt_compile

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    variable = sys.modules["textgrad"].Variable("Answer briefly.", role_description="prompt")
    monkeypatch.setattr(prompt_compile, "_load_program_module", lambda path: types.SimpleNamespace(
        program="Answer briefly.", variable=variable,
    ))
    plan = prompt_compile.build_compile_plan(
        program_path="prog.py", eval_suite_path="eval.jsonl", optimizer="textgrad", max_iters=1,
        output_path="out.py",
    )
    result = prompt_compile.run_compile(plan)
    assert variable.gradients
    assert any(c == ("step", [1]) for c in recording_textgrad)
    assert result.program_text == "Answer briefly. improved"


@pytest.mark.parametrize("gold", [{"answer": "4"}, types.SimpleNamespace(answer="4")])
@pytest.mark.parametrize("pred", [" 4 ", {"answer": "4"}, types.SimpleNamespace(output="4")])
def test_suite_metric_handles_dict_and_object_targets(gold, pred):
    from soup_cli.utils.prompt_compile import _resolve_metric

    metric = _resolve_metric(types.SimpleNamespace(), [{"answer": "4"}])
    assert metric(gold, pred) == 1.0
    assert metric(gold, "5") == 0.0


@pytest.mark.parametrize(
    "metric_code, examples, expected_score, expected_converged, expected_iterations",
    [
        (
            "def metric(gold, pred):\n    return float(pred.endswith('improved'))\n",
            [{"answer": "unused"}], 1.0, True, 1,
        ),
        (
            "def metric(pred):\n    return float(pred.endswith('improved'))\n",
            [{"answer": "unused"}], 1.0, True, 1,
        ),
        (
            "",
            [{"answer": "Answer briefly. improved"}], 1.0, True, 1,
        ),
        (
            "",
            [{"answer": "Answer briefly. improved"}, {"answer": "other"}],
            0.5, False, 3,
        ),
        (
            "def metric(gold, pred):\n    return 1.0\n",
            [{"answer": "unused"}], 1.0, False, 3,
        ),
        (
            "def metric(gold, pred):\n    raise ValueError('evaluation failed')\n",
            [{"answer": "unused"}], 0.0, False, 3,
        ),
    ],
)
def test_textgrad_measures_score_and_convergence(
    tmp_path, monkeypatch, recording_textgrad,
    metric_code, examples, expected_score, expected_converged, expected_iterations,
):
    from soup_cli.utils.prompt_compile import build_compile_plan, run_compile

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\n" + metric_code, encoding="utf-8"
    )
    (tmp_path / "eval.jsonl").write_text(
        "\n".join(json.dumps(ex) for ex in examples), encoding="utf-8"
    )
    result = run_compile(build_compile_plan(
        program_path="prog.py", eval_suite_path="eval.jsonl", optimizer="textgrad",
        max_iters=3, output_path="out.py",
    ))
    assert result.score == expected_score
    assert result.converged is expected_converged
    assert result.iterations == expected_iterations
    assert len([call for call in recording_textgrad if call[0] == "step"]) == expected_iterations
