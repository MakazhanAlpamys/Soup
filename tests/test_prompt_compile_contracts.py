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


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text))


def _panel(output: str) -> dict:
    return dict(re.findall(r"(Score|Iterations|Converged):\s+(\S+)", _plain(output)))


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
    singleton_engine = [None]

    class _Variable:
        def __init__(self, value, *, role_description: str, predecessors=None, **kwargs):
            self.value = value
            self.role_description = role_description
            self.predecessors = list(predecessors or [])
            self.gradients = set()
            self.grad_fn = None
            calls.append(("Variable", value))

        def backward(self, *args, **kwargs):
            engine = kwargs.get("engine") or (args[0] if args else None)
            backward_engine = kwargs.get("backward_engine")
            passed_engine = engine or backward_engine
            if singleton_engine[0] is not None and passed_engine is not None:
                raise Exception(
                    "Cannot pass backward_engine to backward() "
                    "if it was already set via set_backward_engine()"
                )
            used_engine = passed_engine or singleton_engine[0]
            if used_engine is None:
                raise Exception("No backward engine provided.")
            calls.append(("backward", used_engine))
            if self.grad_fn is not None:
                self.grad_fn(backward_engine=used_engine)

    class _TextLoss:
        def __init__(self, eval_system_prompt, engine=None):
            if engine == "default":
                raise ValueError("Engine default not supported")
            self.eval_system_prompt = eval_system_prompt
            self.engine = engine
            calls.append(("TextLoss", eval_system_prompt, engine))

        def __call__(self, variable):
            loss_var = _Variable(
                f"Loss on {variable.value}",
                role_description="evaluation loss",
                predecessors=[variable],
            )

            def grad_fn(backward_engine=None):
                used_engine = backward_engine or self.engine
                calls.append(("engine_call", self.eval_system_prompt, used_engine))
                variable.gradients.add(f"grad_from_{self.eval_system_prompt}")

            loss_var.grad_fn = grad_fn
            return loss_var

    class _TGD:
        def __init__(self, parameters, engine=None, **kwargs):
            if engine == "default":
                raise ValueError("Engine default not supported")
            self.parameters = parameters
            self.engine = engine
            kw = dict(kwargs)
            if engine is not None:
                kw["engine"] = engine
            calls.append(("TGD", sorted(kw)))

        def zero_grad(self):
            for parameter in self.parameters:
                parameter.gradients.clear()
            calls.append(("zero_grad",))

        def step(self):
            calls.append(("step", [len(v.gradients) for v in self.parameters]))
            for parameter in self.parameters:
                if len(parameter.gradients) > 0:
                    parameter.value += " improved"

    def _set_backward_engine(e):
        singleton_engine[0] = e
        calls.append(("set_backward_engine", e))

    class _SingletonBackwardEngine:
        def __init__(self):
            pass

        def get_engine(self):
            return singleton_engine[0]

    fake_tg = types.SimpleNamespace(
        Variable=_Variable,
        TextLoss=_TextLoss,
        TGD=_TGD,
        SingletonBackwardEngine=_SingletonBackwardEngine,
        set_backward_engine=_set_backward_engine,
    )
    monkeypatch.setitem(sys.modules, "textgrad", fake_tg)
    return calls


@pytest.fixture
def recording_gepa(monkeypatch):
    calls = []

    def fake_optimize(*, seed_candidate, trainset, **kwargs):
        assert isinstance(seed_candidate, dict), (
            f"seed_candidate must be dict, got {type(seed_candidate)}"
        )
        assert kwargs.get("task_lm") != "default_lm", "default_lm placeholder not permitted"
        assert kwargs.get("reflection_lm") != "default_lm", "default_lm placeholder not permitted"
        assert "task_lm" in kwargs or "adapter" in kwargs, "GEPA requires task_lm or adapter"
        if kwargs.get("adapter") is not None and kwargs.get("task_lm") is not None:
            raise ValueError("GEPA requires task_lm=None in adapter mode")

        active_adapter = kwargs.get("adapter")
        if active_adapter is not None:
            assert hasattr(active_adapter, "evaluate") and callable(
                getattr(active_adapter, "evaluate")
            ), "adapter must implement evaluate()"
        adapter_has_propose = (
            active_adapter is not None
            and hasattr(active_adapter, "propose_new_texts")
            and getattr(active_adapter, "propose_new_texts") is not None
        )
        custom_proposer = kwargs.get("custom_candidate_proposer")
        if not adapter_has_propose and custom_proposer is None:
            has_reflection = (
                kwargs.get("reflection_lm") is not None
                or kwargs.get("reflection_strategy") is not None
            )
            assert has_reflection, (
                "reflection_lm was not provided. The adapter does not provide propose_new_texts, "
                "and custom_candidate_proposer was not provided. GEPA will use default proposer, "
                "which requires reflection_lm (or reflection_strategy) to be specified."
            )

        calls.append(("optimize", {"seed_candidate": seed_candidate, "kwargs": kwargs}))
        return types.SimpleNamespace(
            best_candidate={
                "prompt": "optimised prompt text",
                "description": "optimised tool desc",
            },
            candidates=[
                {
                    "prompt": "optimised prompt text",
                    "description": "optimised tool desc",
                }
            ],
            val_aggregate_scores=[0.95],
            best_idx=0,
            total_metric_calls=kwargs.get(
                "max_metric_calls", len(trainset) if hasattr(trainset, "__len__") else 5
            ),
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
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nengine = 'gpt-4o'\n", encoding="utf-8"
    )
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

    # 2. TextLoss constructed and engine invoked with eval example text
    loss_calls = [c for c in calls1 if c[0] == "TextLoss"]
    assert len(loss_calls) == 2, f"Expected 2 TextLoss calls, got {loss_calls}"
    assert "1+1" in loss_calls[0][1] or "2+2" in loss_calls[0][1]

    engine_calls = [c for c in calls1 if c[0] == "engine_call"]
    assert len(engine_calls) == 2
    assert engine_calls[0][2] is not None, "backward did not receive or use an engine"

    # 3. Each step is preceded by a backward() and has positive parameter gradients
    for i, call in enumerate(calls1):
        if call[0] == "step":
            assert i >= 2 and calls1[i - 2][0] == "backward", (
                f"step at {i} not preceded by backward: {calls1}"
            )
            assert calls1[i - 1][0] == "engine_call", (
                f"step at {i} not preceded by engine_call: {calls1}"
            )
            assert call[1] == [1], f"Parameter gradients were 0: {call}"

    # 4. Two different suites produce different call sequences
    assert calls1 != calls2


def test_result_panel_reports_actual_convergence_and_score(
    tmp_path, monkeypatch, recording_textgrad
):
    monkeypatch.chdir(tmp_path)
    # Text edits alone must not imply convergence.
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nengine = 'gpt-4o'\n", encoding="utf-8"
    )
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
    assert panel.get("Score") == "unmeasured"


def test_gepa_seed_candidate_dict_and_string_output(tmp_path, monkeypatch, recording_gepa):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\ntask_lm = 'my_model'\n", encoding="utf-8"
    )
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
            "--engine",
            "gpt-4o",
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
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nengine = 'dummy_engine'\n", encoding="utf-8"
    )
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
            "--engine",
            "dummy_engine",
            "-o",
            "out.json",
        ],
    )
    assert res_tools.exit_code == 2
    assert "No engine provided" in res_tools.output


def test_gepa_adapter_mode_passes_task_lm_none(tmp_path, monkeypatch, recording_gepa):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\n"
        "class MockAdapter:\n"
        "    def evaluate(self, candidate):\n"
        "        return 1.0\n"
        "adapter = MockAdapter()\n"
        "task_lm = 'ignored_lm'\n"
        "reflection_lm = 'my_reflection_lm'\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    res = CliRunner().invoke(
        app, ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "gepa", "-o", "out.py"]
    )
    assert res.exit_code == 0, res.output
    kwargs = recording_gepa[0][1]["kwargs"]
    assert hasattr(kwargs["adapter"], "evaluate")
    assert kwargs["task_lm"] is None


def test_textgrad_edit_without_score_improvement_does_not_claim_converged(
    tmp_path, monkeypatch, recording_textgrad
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nengine = 'gpt-4o'\ndef metric(gold, pred):\n    return 0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    res = CliRunner().invoke(app, [
        "compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "textgrad",
        "--max-iters", "2", "-o", "out.py",
    ])
    assert res.exit_code == 0, res.output
    written = (tmp_path / "out.py").read_text(encoding="utf-8")
    assert "improved" in written
    panel = dict(re.findall(r"(Converged|Score):\s+(\S+)", _strip_ansi(res.output)))
    assert panel == {"Converged": "False", "Score": "unmeasured"}


def test_textgrad_loss_predecessor_graph_populates_gradients(
    tmp_path, monkeypatch, recording_textgrad
):
    from soup_cli.utils import prompt_compile

    monkeypatch.chdir(tmp_path)
    metric_code = (
        "program = 'Answer briefly.'\n"
        "engine = 'gpt-4o'\n"
        "def metric(gold, pred):\n"
        "    return 1.0 if 'improved' in str(pred) else 0.0\n"
    )
    (tmp_path / "prog.py").write_text(metric_code, encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    variable = sys.modules["textgrad"].Variable("Answer briefly.", role_description="prompt")
    monkeypatch.setattr(
        prompt_compile,
        "_load_program_module",
        lambda path: types.SimpleNamespace(
            program="Answer briefly.",
            engine="gpt-4o",
            variable=variable,
            metric=lambda g, p: 1.0 if "improved" in str(p) else 0.0,
        ),
    )
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
        "program = 'Answer briefly.'\nengine = 'gpt-4o'\n" + metric_code, encoding="utf-8"
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


def test_textgrad_preserves_best_candidate_on_regression(tmp_path, monkeypatch):
    """If score improves then regresses, run_compile preserves best text with best_score."""
    from soup_cli.utils.prompt_compile import build_compile_plan, run_compile

    step_idx = 0

    class _Variable:
        def __init__(self, value, *, role_description: str, predecessors=None, **kwargs):
            self.value = value
            self.role_description = role_description
            self.predecessors = list(predecessors or [])
            self.gradients = set()
            self.grad_fn = None

        def backward(self, *args, **kwargs):
            if self.grad_fn is not None:
                self.grad_fn()

    class _TextLoss:
        def __init__(self, eval_system_prompt, engine=None):
            pass

        def __call__(self, variable):
            loss_var = _Variable(
                "loss", role_description="evaluation loss", predecessors=[variable]
            )

            def grad_fn(*a, **kw):
                variable.gradients.add("grad")

            loss_var.grad_fn = grad_fn
            return loss_var

    class _TGD:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters

        def zero_grad(self):
            for p in self.parameters:
                p.gradients.clear()

        def step(self):
            nonlocal step_idx
            step_idx += 1
            for p in self.parameters:
                p.value = f"Candidate_{step_idx}"

    fake_tg = types.SimpleNamespace(
        Variable=_Variable,
        TextLoss=_TextLoss,
        TGD=_TGD,
        SingletonBackwardEngine=type("S", (), {"get_engine": lambda self: "engine"}),
    )
    monkeypatch.setitem(sys.modules, "textgrad", fake_tg)

    prog_code = """
program = 'Initial prompt'
engine = 'gpt-4o'
def metric(gold, pred):
    if 'Candidate_1' in pred:
        return 0.5
    return 0.0
"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(prog_code, encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "q"}\n', encoding="utf-8")

    plan = build_compile_plan(
        program_path="prog.py",
        eval_suite_path="eval.jsonl",
        optimizer="textgrad",
        max_iters=2,
        output_path="out.py",
    )
    result = run_compile(plan)
    assert result.score == 0.5
    assert result.program_text == "Candidate_1"


def test_gepa_forwards_reflection_lm_and_proposer_configuration(
    tmp_path, monkeypatch, recording_gepa
):
    """User module exporting reflection_lm is forwarded to gepa.optimize."""
    from soup_cli.utils.prompt_compile import build_compile_plan, run_compile

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Prompt'\ntask_lm = 'my_task_lm'\nreflection_lm = 'my_reflection_lm'\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text('{"q": "1"}\n', encoding="utf-8")

    plan = build_compile_plan(
        program_path="prog.py",
        eval_suite_path="eval.jsonl",
        optimizer="gepa",
        max_iters=2,
        output_path="out.py",
    )
    res = run_compile(plan)
    assert res.score == 0.95
    assert len(recording_gepa) == 1
    kwargs = recording_gepa[0][1]["kwargs"]
    assert kwargs.get("task_lm") == "my_task_lm"
    assert kwargs.get("reflection_lm") == "my_reflection_lm"


def test_compile_tools_gepa_passes_reflection_lm(tmp_path, monkeypatch, recording_gepa):
    """compile_tools runs gepa.optimize with task_lm and reflection_lm configured."""
    from soup_cli.utils.compile_tools import ToolCompilePlan, run_tool_compile

    monkeypatch.chdir(tmp_path)
    spec_json = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0"},
        "paths": {
            "/items": {
                "get": {
                    "operationId": "get_items",
                    "description": "Fetch items.",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }
    (tmp_path / "tools.json").write_text(json.dumps(spec_json), encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(
        '{"query": "fetch", "tool": "get_items"}\n', encoding="utf-8"
    )

    plan = ToolCompilePlan(
        spec_path="tools.json",
        eval_suite_path="eval.jsonl",
        optimizer="gepa",
        output_path="tools_out.json",
        task_lm="custom_task_lm",
        reflection_lm="custom_refl_lm",
    )
    count = run_tool_compile(plan)
    assert count == 1
    assert len(recording_gepa) == 1
    kwargs = recording_gepa[0][1]["kwargs"]
    assert kwargs.get("task_lm") == "custom_task_lm"
    assert kwargs.get("reflection_lm") == "custom_refl_lm"


def test_compile_tools_gepa_cli_options(tmp_path, monkeypatch, recording_gepa):
    """compile-tools CLI forwards --task-lm and --reflection-lm to gepa.optimize."""
    monkeypatch.chdir(tmp_path)
    spec_json = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0"},
        "paths": {
            "/items": {
                "get": {
                    "operationId": "get_items",
                    "description": "Fetch items.",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }
    (tmp_path / "tools.json").write_text(json.dumps(spec_json), encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(
        '{"query": "fetch", "tool": "get_items"}\n', encoding="utf-8"
    )

    runner = CliRunner()
    # 1. Custom task and reflection models
    res = runner.invoke(
        app,
        [
            "compile-tools",
            "tools.json",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "gepa",
            "--task-lm",
            "my_task_model",
            "--reflection-lm",
            "my_reflection_model",
            "-o",
            "out1.json",
        ],
    )
    assert res.exit_code == 0, res.output
    assert recording_gepa[-1][1]["kwargs"]["task_lm"] == "my_task_model"
    assert recording_gepa[-1][1]["kwargs"]["reflection_lm"] == "my_reflection_model"

    # 2. Omitting --reflection-lm defaults reflection_lm to task_lm
    res_default_refl = runner.invoke(
        app,
        [
            "compile-tools",
            "tools.json",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "gepa",
            "--task-lm",
            "my_task_model_only",
            "-o",
            "out2.json",
        ],
    )
    assert res_default_refl.exit_code == 0, res_default_refl.output
    assert recording_gepa[-1][1]["kwargs"]["task_lm"] == "my_task_model_only"
    assert recording_gepa[-1][1]["kwargs"]["reflection_lm"] == "my_task_model_only"

    # 3. Omitting --task-lm fails with exit code 2
    res_no_task = runner.invoke(
        app,
        [
            "compile-tools",
            "tools.json",
            "--eval",
            "eval.jsonl",
            "--optimizer",
            "gepa",
            "-o",
            "out3.json",
        ],
    )
    assert res_no_task.exit_code == 2
    assert "GEPA optimizer requires --task-lm" in _plain(res_no_task.output)


def test_gepa_reads_val_aggregate_scores_without_best_score(tmp_path, monkeypatch):
    """_run_gepa reads val_aggregate_scores[best_idx] when best_score is absent."""
    from soup_cli.utils.prompt_compile import CompilePlan, _run_gepa

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Hello'\ntask_lm = 'my_task_lm'\n", encoding="utf-8"
    )
    (tmp_path / "eval.jsonl").write_text('{"input": "a", "output": "b"}\n', encoding="utf-8")

    plan = CompilePlan(
        program_path="prog.py",
        eval_suite_path="eval.jsonl",
        optimizer="gepa",
        max_iters=2,
        output_path="out.py",
    )

    # 1. Real GEPAResult structure (val_aggregate_scores indexed by best_idx,
    # no best_score attribute)
    class FakeGEPAResultReal:
        best_candidate = {"prompt": "Improved prompt"}
        val_aggregate_scores = [0.25, 0.88, 0.45]
        best_idx = 1
        iterations = 2
        converged = True

    fake_gepa_mod = types.SimpleNamespace(optimize=lambda **kw: FakeGEPAResultReal())
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa_mod)

    res = _run_gepa(plan)
    assert res.score == 0.88

    # 2. Legacy structure with only best_score
    class FakeGEPAResultLegacy:
        best_candidate = {"prompt": "Improved prompt"}
        best_score = 0.73
        iterations = 1
        converged = True

    fake_gepa_legacy = types.SimpleNamespace(optimize=lambda **kw: FakeGEPAResultLegacy())
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa_legacy)
    res_legacy = _run_gepa(plan)
    assert res_legacy.score == 0.73


def test_textgrad_refuses_when_no_engine_configured(tmp_path, monkeypatch, recording_textgrad):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"question": "1+1", "answer": "2"}\n', encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "textgrad", "-o", "out.py"],
    )
    assert res.exit_code == 2
    assert "No backward engine configured" in _plain(res.output)

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
    assert "No backward engine configured" in _plain(res_tools.output)


def test_textgrad_preserves_prompt_when_no_score_signal(tmp_path, monkeypatch, recording_textgrad):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nengine = 'gpt-4o'\n", encoding="utf-8"
    )
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
            "2",
            "-o",
            "out.py",
        ],
    )
    assert res.exit_code == 0, res.output
    written = (tmp_path / "out.py").read_text(encoding="utf-8")
    assert "improved" in written
    stripped = _strip_ansi(res.output)
    panel = dict(re.findall(r"(Iterations|Converged|Score):\s+(\S+)", stripped))
    assert panel.get("Score") == "unmeasured"
    assert panel.get("Converged") == "False"


def test_gepa_invalid_adapter_raises_cleanly(tmp_path, monkeypatch, recording_gepa):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\nadapter = 'string_not_adapter_object'\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text('{"answer": "4"}\n', encoding="utf-8")
    res = CliRunner().invoke(
        app, ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "gepa", "-o", "out.py"]
    )
    assert res.exit_code == 2
    assert "does not implement the required 'evaluate' method" in _plain(res.output)


@pytest.fixture
def published_textgrad(monkeypatch):
    """textgrad 0.1.8 guards (config.py / variable.py / loss.py / optimizer.py): no
    get_backward_engine export, Variable has no score, backward() refuses 'no engine' and 'both'."""
    state = {"engine": None}

    def get_engine(name, **kwargs):
        if "gpt-4" in name:
            return types.SimpleNamespace(name=name)
        raise ValueError(f"Engine {name} not supported")

    class Singleton:
        def get_engine(self):
            return state["engine"]

    def set_backward_engine(engine, override=False, **kwargs):
        if isinstance(engine, str):
            engine = get_engine(engine)
        if state["engine"] is not None and not override:
            raise Exception("Engine already set. Use override=True to override cautiously.")
        state["engine"] = engine

    def validate(engine):
        if engine is None and state["engine"] is None:
            raise Exception("No engine provided.")
        if engine is None:
            engine = state["engine"]
        return get_engine(engine) if isinstance(engine, str) else engine

    class Variable:
        def __init__(self, value="", predecessors=None, requires_grad=True, role_description=""):
            self.value = value
            self.predecessors = list(predecessors or [])
            self.requires_grad = requires_grad
            self.gradients = set()

        def backward(self, engine=None):
            if engine is None and state["engine"] is None:
                raise Exception("No backward engine provided.")
            if engine is not None and state["engine"] is not None:
                raise Exception("Both an engine is provided and the global engine is set.")
            for pred in self.predecessors:
                pred.gradients.add("critique")

    class TextLoss:
        def __init__(self, eval_system_prompt, engine=None):
            validate(engine)

        def __call__(self, instance):
            return Variable("loss", predecessors=[instance])

    class TGD:
        def __init__(self, parameters, engine=None, **kwargs):
            self.parameters = parameters
            validate(engine)

        def zero_grad(self):
            for parameter in self.parameters:
                parameter.gradients.clear()

        def step(self):
            for parameter in self.parameters:
                if parameter.gradients:
                    parameter.value = parameter.value + " [edited]"

    fake = types.SimpleNamespace(
        Variable=Variable, TextLoss=TextLoss, TGD=TGD, get_engine=get_engine,
        set_backward_engine=set_backward_engine, SingletonBackwardEngine=Singleton,
    )
    monkeypatch.setitem(sys.modules, "textgrad", fake)
    return types.SimpleNamespace(module=fake, state=state)


def _suite(tmp_path):
    (tmp_path / "eval.jsonl").write_text('{"question": "1+1", "answer": "2"}\n', encoding="utf-8")


def _compile_textgrad(tmp_path, program_source, *extra):
    (tmp_path / "prog.py").write_text(program_source, encoding="utf-8")
    return CliRunner().invoke(
        app,
        ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "textgrad",
         "--max-iters", "2", "-o", "out.py", *extra],
    )


def test_textgrad_runs_when_the_program_calls_set_backward_engine(
    tmp_path, monkeypatch, published_textgrad
):
    """The documented route: no `engine` name in the program, only the singleton."""
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    res = _compile_textgrad(
        tmp_path,
        "import textgrad\ntextgrad.set_backward_engine('gpt-4o')\nprogram = 'Answer briefly.'\n",
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert (tmp_path / "out.py").read_text(encoding="utf-8").endswith("[edited] [edited]")
    plain = _plain(res.output)
    assert _panel(plain).get("Score") == "unmeasured"


@pytest.mark.parametrize(
    "engine_source", ["engine = 'gpt-4o'", "engine = textgrad.get_engine('gpt-4o')"]
)
def test_textgrad_registers_an_exported_engine_for_backward(
    tmp_path, monkeypatch, published_textgrad, engine_source
):
    """loss.backward() takes no argument, so the exported engine must be the singleton by then."""
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    res = _compile_textgrad(
        tmp_path, f"import textgrad\n{engine_source}\nprogram = 'Answer briefly.'\n"
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert published_textgrad.state["engine"] is not None
    assert (tmp_path / "out.py").read_text(encoding="utf-8").endswith("[edited] [edited]")


def test_textgrad_keeps_the_initial_text_when_every_edit_scores_worse(
    tmp_path, monkeypatch, published_textgrad
):
    """A metric that rates the initial text 0.5 and every edit 0.0 is a real signal: keep the
    initial text and print its score, not 'unmeasured'."""
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    res = _compile_textgrad(
        tmp_path,
        "engine = 'gpt-4o'\nprogram = 'Answer briefly.'\n"
        "def metric(gold, pred):\n    return 0.5 if pred == 'Answer briefly.' else 0.0\n",
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert (tmp_path / "out.py").read_text(encoding="utf-8") == "Answer briefly."
    plain = _plain(res.output)
    assert _panel(plain).get("Score") == "0.500"


def _tools_spec(tmp_path):
    spec = {
        "openapi": "3.0.0", "info": {"title": "T", "version": "1"},
        "paths": {
            "/s": {"get": {"operationId": "search", "description": "Search catalog"}},
            "/t": {"get": {"operationId": "tag", "description": "Tag items"}},
        },
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "tools_eval.jsonl").write_text(
        '{"query": "shoes", "tool": "search"}\n', encoding="utf-8"
    )


def _descriptions(tmp_path):
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    return [tool["description"] for tool in data["tools"]]


def test_compile_tools_textgrad_engine_flag_reaches_backward(
    tmp_path, monkeypatch, published_textgrad
):
    monkeypatch.chdir(tmp_path)
    _tools_spec(tmp_path)
    res = CliRunner().invoke(
        app,
        ["compile-tools", "spec.json", "--eval", "tools_eval.jsonl", "--optimizer", "textgrad",
         "--engine", "gpt-4o", "-o", "out.json"],
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert _descriptions(tmp_path) == ["Search catalog [edited]", "Tag items [edited]"]


def test_compile_tools_textgrad_uses_an_engine_the_caller_already_set(
    tmp_path, monkeypatch, published_textgrad
):
    monkeypatch.chdir(tmp_path)
    _tools_spec(tmp_path)
    published_textgrad.module.set_backward_engine("gpt-4o")
    res = CliRunner().invoke(
        app,
        ["compile-tools", "spec.json", "--eval", "tools_eval.jsonl", "--optimizer", "textgrad",
         "-o", "out.json"],
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert _descriptions(tmp_path) == ["Search catalog [edited]", "Tag items [edited]"]


def test_compile_gepa_refuses_a_program_without_task_lm_or_adapter(
    tmp_path, monkeypatch, recording_gepa
):
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    res = CliRunner().invoke(
        app, ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "gepa", "-o", "out.py"]
    )
    assert res.exit_code == 2
    assert "define 'task_lm' or 'adapter'" in _plain(res.output)
    assert recording_gepa == []


def test_compile_gepa_names_the_missing_evaluate_method(tmp_path, monkeypatch, recording_gepa):
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'A'\nadapter = 'string_not_adapter_object'\n", encoding="utf-8"
    )
    res = CliRunner().invoke(
        app, ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "gepa", "-o", "out.py"]
    )
    assert res.exit_code == 2
    assert "does not implement the required 'evaluate' method" in _plain(res.output)
    assert recording_gepa == []


def _fake_dspy(monkeypatch, compiled):
    class _Optimizer:
        def __init__(self, **kwargs):
            pass

        def compile(self, program, **kwargs):
            return compiled

    monkeypatch.setitem(sys.modules, "dspy", types.SimpleNamespace(
        MIPROv2=_Optimizer, COPRO=_Optimizer, BootstrapFewShot=_Optimizer,
        Example=lambda **kw: types.SimpleNamespace(with_inputs=lambda *keys: None),
    ))


@pytest.mark.parametrize("converged_attr, expected", [({"converged": True}, "True"), ({}, "False")])
def test_dspy_panel_reports_what_the_compiled_program_says(
    tmp_path, monkeypatch, converged_attr, expected
):
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    (tmp_path / "prog.py").write_text("program = 'Answer briefly.'\n", encoding="utf-8")
    _fake_dspy(monkeypatch, types.SimpleNamespace(score=0.85, total_calls=11, **converged_attr))
    res = CliRunner().invoke(
        app,
        ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "mipro",
         "--max-iters", "4", "-o", "out.py"],
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    plain = _plain(res.output)
    assert _panel(plain) == {"Score": "0.850", "Iterations": "11", "Converged": expected}


@pytest.mark.parametrize("extra, expected", [({"converged": True}, "True"), ({}, "False")])
def test_gepa_panel_reports_metric_calls_and_convergence(tmp_path, monkeypatch, extra, expected):
    monkeypatch.chdir(tmp_path)
    _suite(tmp_path)
    (tmp_path / "prog.py").write_text(
        "program = 'Answer briefly.'\ntask_lm = 'gpt-4o-mini'\n", encoding="utf-8"
    )
    result = types.SimpleNamespace(
        best_candidate={"prompt": "NEW"}, val_aggregate_scores=[0.4, 0.9], best_idx=1,
        total_metric_calls=37, **extra,
    )
    monkeypatch.setitem(sys.modules, "gepa", types.SimpleNamespace(optimize=lambda **kw: result))
    res = CliRunner().invoke(
        app,
        ["compile", "prog.py", "--eval", "eval.jsonl", "--optimizer", "gepa",
         "--max-iters", "5", "-o", "out.py"],
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    plain = _plain(res.output)
    assert _panel(plain) == {"Score": "0.900", "Iterations": "37", "Converged": expected}


def test_compile_tools_gepa_writes_the_description_component_not_the_dict(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _tools_spec(tmp_path)
    result = types.SimpleNamespace(best_candidate={"other": "X", "description": "NEW"})
    monkeypatch.setitem(sys.modules, "gepa", types.SimpleNamespace(optimize=lambda **kw: result))
    res = CliRunner().invoke(
        app,
        ["compile-tools", "spec.json", "--eval", "tools_eval.jsonl", "--optimizer", "gepa",
         "--task-lm", "gpt-4o-mini", "-o", "out.json"],
    )
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert _descriptions(tmp_path) == ["NEW", "NEW"]


def test_compile_result_score_unmeasured_type_check():
    from soup_cli.utils.prompt_compile import CompileResult

    with pytest.raises(TypeError, match="score_unmeasured must be bool"):
        CompileResult("txt", 0.5, 1, True, score_unmeasured="not_bool")
