"""v0.71.7 "Eval live runners" — closes #161, #162, #208, #211, #212, #165.

Live model-loading is mocked at the ``soup_cli.utils.live_eval`` boundary (and
``lm_eval`` is faked via ``sys.modules``) so every orchestration path is
exercised on CPU without a GPU or a model download. The real model load is
covered by the release-step-6 smoke on SmolLM2-135M.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


# ===========================================================================
# Shared utils/live_eval.py
# ===========================================================================
class TestLiveEvalCore:
    def test_resolve_device_explicit(self) -> None:
        from soup_cli.utils import live_eval

        assert live_eval.resolve_device("cpu") == "cpu"
        assert live_eval.resolve_device("cuda") == "cuda"

    def test_resolve_device_rejects_empty(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.resolve_device("")

    def test_resolve_device_auto_returns_str(self) -> None:
        from soup_cli.utils import live_eval

        assert live_eval.resolve_device(None) in {"cpu", "cuda"}

    def test_apply_prompt_template_no_template(self) -> None:
        from soup_cli.utils import live_eval

        class _Tok:
            chat_template = None

        assert live_eval._apply_prompt_template(_Tok(), "hi") == "hi"

    def test_apply_prompt_template_uses_chat_template(self) -> None:
        from soup_cli.utils import live_eval

        class _Tok:
            chat_template = "x"

            def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
                return "TEMPLATED:" + msgs[0]["content"]

        assert live_eval._apply_prompt_template(_Tok(), "hi") == "TEMPLATED:hi"

    def test_make_generator_rejects_bad_max_new_tokens(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.make_generator("m", max_new_tokens=0)
        with pytest.raises(ValueError):
            live_eval.make_generator("m", max_new_tokens=True)

    def test_make_generator_rejects_bad_loaded(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.make_generator("m", loaded=("only", "two"))

    def test_make_multi_generator_rejects_bad_args(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.make_multi_generator("m", max_new_tokens=0)
        with pytest.raises(ValueError):
            live_eval.make_multi_generator("m", max_new_tokens=True)
        with pytest.raises(ValueError):
            live_eval.make_multi_generator("m", temperature=0)
        with pytest.raises(ValueError):
            live_eval.make_multi_generator("m", temperature=True)

    def test_load_model_rejects_empty_id(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.load_model_and_tokenizer("")

    def test_build_pairs_skips_missing_sides(self) -> None:
        from soup_cli.utils import live_eval

        rows = [
            {"prompt": "p1", "response": "t1"},
            {"prompt": "", "response": "t2"},  # no prompt
            {"prompt": "p3", "response": ""},  # no target
            "not-a-mapping",
        ]
        pairs = live_eval._build_pairs(
            rows,
            input_extractor=lambda r: str(r.get("prompt", "")),
            output_extractor=lambda r: str(r.get("response", "")),
        )
        assert pairs == [("p1", "t1")]

    def test_lora_probe_rejects_bad_steps(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.lora_probe(
                "m", [], input_extractor=str, output_extractor=str, n_steps=0
            )

    def test_measure_logit_agreement_rejects_bad_max_pairs(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.measure_logit_agreement(
                "m", [], input_extractor=str, output_extractor=str, max_pairs=0
            )


# ===========================================================================
# live_eval primitives — real torch tensors on CPU, fake model/tokenizer
# (covers the masking / loss / logit-agreement math without a download).
# ===========================================================================
class _FakeTok:
    """Word-length tokeniser: each whitespace word -> an int id == its length."""

    chat_template = None

    def __init__(self, eos: int | None = 99) -> None:
        self.eos_token_id = eos
        self.pad_token_id = eos

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [len(w) for w in text.split()]}


class TestLiveEvalPrimitives:
    def test_tokenize_pair_masks_prompt(self) -> None:
        from soup_cli.utils import live_eval

        ids, labels = live_eval._tokenize_pair(
            _FakeTok(eos=99), "a bb", "ccc dddd", max_length=256
        )
        assert ids[0].tolist() == [1, 2, 3, 4, 99]
        assert labels[0].tolist() == [-100, -100, 3, 4, 99]

    def test_tokenize_pair_truncates(self) -> None:
        from soup_cli.utils import live_eval

        ids, _ = live_eval._tokenize_pair(
            _FakeTok(eos=99), "a bb ccc", "dddd", max_length=2
        )
        assert ids.shape[1] == 2

    def test_compute_eval_loss_mean(self) -> None:
        import types

        import torch

        from soup_cli.utils import live_eval

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, input_ids, labels):
                return types.SimpleNamespace(loss=torch.tensor(1.5))

        loss = live_eval.compute_eval_loss(
            FakeModel(), _FakeTok(eos=99), [("a", "bb"), ("ccc", "dddd")],
            device="cpu",
        )
        assert loss == pytest.approx(1.5)

    def test_compute_eval_loss_all_empty_targets_nan(self) -> None:
        import types

        import torch

        from soup_cli.utils import live_eval

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, input_ids, labels):
                return types.SimpleNamespace(loss=torch.tensor(0.0))

        # eos=None + empty targets → labels are all -100 → skipped → NaN.
        loss = live_eval.compute_eval_loss(
            FakeModel(), _FakeTok(eos=None), [("a", ""), ("bb", "")], device="cpu",
        )
        assert loss != loss  # NaN

    def test_measure_logit_agreement_perfect(self, monkeypatch) -> None:
        import types

        import torch

        from soup_cli.utils import live_eval

        class FakeLogitModel:
            vocab = 200

            def eval(self):
                return self

            def __call__(self, input_ids):
                seq = input_ids.shape[1]
                logits = torch.zeros((1, seq, self.vocab))
                ids = input_ids[0].tolist()
                for t in range(seq - 1):
                    logits[0, t, ids[t + 1]] = 10.0  # predict the real next token
                return types.SimpleNamespace(logits=logits)

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (FakeLogitModel(), _FakeTok(eos=99), "cpu"),
        )
        score = live_eval.measure_logit_agreement(
            "m", [{"p": "a bb", "t": "ccc dddd"}],
            input_extractor=lambda r: r["p"], output_extractor=lambda r: r["t"],
        )
        assert score == pytest.approx(1.0)

    def test_measure_logit_agreement_wrong(self, monkeypatch) -> None:
        import types

        import torch

        from soup_cli.utils import live_eval

        class WrongModel:
            def eval(self):
                return self

            def __call__(self, input_ids):
                seq = input_ids.shape[1]
                logits = torch.zeros((1, seq, 200))
                logits[0, :, 0] = 10.0  # always predict token 0
                return types.SimpleNamespace(logits=logits)

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (WrongModel(), _FakeTok(eos=99), "cpu"),
        )
        score = live_eval.measure_logit_agreement(
            "m", [{"p": "a bb", "t": "ccc dddd"}],
            input_extractor=lambda r: r["p"], output_extractor=lambda r: r["t"],
        )
        assert score == pytest.approx(0.0)

    def test_lora_probe_rejects_bad_lr_and_max_length(self) -> None:
        from soup_cli.utils import live_eval

        with pytest.raises(ValueError):
            live_eval.lora_probe(
                "m", [], input_extractor=str, output_extractor=str,
                n_steps=1, lr=0,
            )
        with pytest.raises(ValueError):
            live_eval.lora_probe(
                "m", [], input_extractor=str, output_extractor=str,
                n_steps=1, max_length=0,
            )


# ===========================================================================
# #161 — advise live probe (synth_probe_baselines / synth_probe_lora_delta)
# ===========================================================================
class TestAdviseLiveProbe:
    def test_baselines_heuristic_when_no_model(self) -> None:
        from soup_cli.utils.advise import synth_probe_baselines

        rows = [{"prompt": "q", "response": "a short answer"} for _ in range(5)]
        out = synth_probe_baselines(rows)
        assert set(out) == {"zero_shot", "few_shot", "rag"}

    def test_token_f1(self) -> None:
        from soup_cli.utils.live_eval import token_f1

        assert token_f1("hello world", "hello world") == pytest.approx(1.0)
        assert token_f1("foo", "bar") == 0.0
        assert token_f1("", "x") == 0.0

    def test_baselines_live_when_model(self, monkeypatch) -> None:
        from soup_cli.utils import live_eval
        from soup_cli.utils.advise import synth_probe_baselines

        def fake_make_generator(model_id, **kwargs):
            return lambda prompt: "the capital is paris"

        monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)
        rows = [
            {"prompt": "capital of france?", "response": "the capital is paris"}
            for _ in range(4)
        ]
        out = synth_probe_baselines(rows, model="HuggingFaceTB/SmolLM2-135M")
        # Perfect match → high F1.
        assert out["zero_shot"] > 0.8
        assert out["few_shot"] >= out["zero_shot"]

    def test_baselines_falls_back_when_live_raises(self, monkeypatch) -> None:
        from soup_cli.utils import live_eval
        from soup_cli.utils.advise import synth_probe_baselines

        def boom(*a, **k):
            raise RuntimeError("no gpu")

        monkeypatch.setattr(live_eval, "make_generator", boom)
        rows = [{"prompt": "q", "response": "a"} for _ in range(5)]
        out = synth_probe_baselines(rows, model="m")  # falls back, no raise
        assert set(out) == {"zero_shot", "few_shot", "rag"}

    def test_lora_delta_heuristic_when_no_model(self) -> None:
        from soup_cli.utils.advise import synth_probe_lora_delta

        rows = [{"prompt": "q", "response": "a"} for _ in range(200)]
        delta, wall = synth_probe_lora_delta(rows)
        assert -0.2 <= delta <= 0.7
        assert wall > 0

    def test_lora_delta_live(self, monkeypatch) -> None:
        from soup_cli.utils import live_eval
        from soup_cli.utils.advise import synth_probe_lora_delta

        monkeypatch.setattr(
            live_eval, "lora_probe", lambda *a, **k: (2.0, 1.0, 12.5)
        )
        rows = [{"prompt": "q", "response": "a"} for _ in range(10)]
        delta, wall = synth_probe_lora_delta(rows, model="m")
        assert delta == pytest.approx(0.5)  # (2-1)/2
        assert wall == pytest.approx(12.5)

    def test_lora_delta_live_nan_falls_back(self, monkeypatch) -> None:
        from soup_cli.utils import live_eval
        from soup_cli.utils.advise import synth_probe_lora_delta

        monkeypatch.setattr(
            live_eval, "lora_probe", lambda *a, **k: (float("nan"), 1.0, 5.0)
        )
        rows = [{"prompt": "q", "response": "a"} for _ in range(200)]
        delta, wall = synth_probe_lora_delta(rows, model="m")
        # NaN base_loss → fallback heuristic path.
        assert -0.2 <= delta <= 0.7


# ===========================================================================
# #162 — base_model_proximity via logit agreement
# ===========================================================================
class TestBaseModelProximity:
    def test_measure_proximity_live(self, monkeypatch) -> None:
        from soup_cli.utils import live_eval
        from soup_cli.utils.advise import measure_base_model_proximity

        monkeypatch.setattr(
            live_eval, "measure_logit_agreement", lambda *a, **k: 0.42
        )
        rows = [{"prompt": "q", "response": "a"}]
        assert measure_base_model_proximity(rows, model="m") == pytest.approx(0.42)

    def test_measure_proximity_nan_returns_none(self, monkeypatch) -> None:
        from soup_cli.utils import live_eval
        from soup_cli.utils.advise import measure_base_model_proximity

        monkeypatch.setattr(
            live_eval, "measure_logit_agreement", lambda *a, **k: float("nan")
        )
        assert measure_base_model_proximity([{"prompt": "q", "response": "a"}], model="m") is None

    def test_measure_proximity_rejects_empty_model(self) -> None:
        from soup_cli.utils.advise import measure_base_model_proximity

        with pytest.raises(ValueError):
            measure_base_model_proximity([], model="")

    def test_profile_stores_proximity(self) -> None:
        from soup_cli.utils.advise import compute_dataset_profile

        rows = [{"prompt": "q", "response": "a"}]
        prof = compute_dataset_profile(rows, base_model_proximity=0.7)
        assert prof.base_model_proximity == pytest.approx(0.7)

    def test_advise_cli_probe_no_model_still_heuristic(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": "q", "response": "a"} for _ in range(60)])
        result = runner.invoke(app, ["advise", "run", str(data), "--probe"])
        assert result.exit_code == 0, (result.output, result.exception)

    def test_advise_cli_probe_model_live(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils import live_eval

        # Mock the live boundary: proximity + generator + lora probe.
        monkeypatch.setattr(
            live_eval, "measure_logit_agreement", lambda *a, **k: 0.55
        )
        monkeypatch.setattr(
            live_eval, "make_generator", lambda *a, **k: (lambda p: "a")
        )
        monkeypatch.setattr(
            live_eval, "lora_probe", lambda *a, **k: (2.0, 1.0, 9.0)
        )
        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": f"q{i}", "response": "a"} for i in range(60)])
        result = runner.invoke(
            app, ["advise", "run", str(data), "--probe-model", "m"]
        )
        assert result.exit_code == 0, (result.output, result.exception)


# ===========================================================================
# #208 — tunability live LoRA probe
# ===========================================================================
class TestTunabilityLiveProbe:
    def test_live_lora_probe(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils import live_eval, tunability

        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": "q", "response": "a"} for _ in range(10)])
        monkeypatch.setattr(
            live_eval, "lora_probe", lambda *a, **k: (3.0, 2.0, 30.0)
        )
        cand = tunability.CandidateBase(
            name="tiny", repo_id="HuggingFaceTB/SmolLM2-135M",
            params_b=0.135, license_id="apache-2.0",
        )
        res = tunability.live_lora_probe(
            cand, str(data), probe_steps=5, holdout_size=2
        )
        assert isinstance(res, tunability.TunabilityResult)
        assert res.base_loss == pytest.approx(3.0)
        assert res.delta == pytest.approx(1.0)  # base - probe
        assert res.wall_clock_seconds == pytest.approx(30.0)

    def test_live_lora_probe_nan_neutral(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils import live_eval, tunability

        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": "q", "response": "a"} for _ in range(4)])
        monkeypatch.setattr(
            live_eval, "lora_probe", lambda *a, **k: (float("nan"), float("nan"), 1.0)
        )
        cand = tunability.CandidateBase(
            name="t", repo_id="r", params_b=0.1, license_id="mit",
        )
        res = tunability.live_lora_probe(cand, str(data), probe_steps=1, holdout_size=1)
        assert res.delta == 0.0

    def test_live_lora_probe_one_nan(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils import live_eval, tunability

        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": "q", "response": "a"} for _ in range(4)])
        # base_loss NaN but probe_loss finite → still neutral 0 delta.
        monkeypatch.setattr(
            live_eval, "lora_probe", lambda *a, **k: (float("nan"), 1.0, 2.0)
        )
        cand = tunability.CandidateBase(
            name="t", repo_id="r", params_b=0.1, license_id="mit",
        )
        res = tunability.live_lora_probe(cand, str(data), probe_steps=1, holdout_size=1)
        assert res.delta == 0.0

    def test_load_jsonl_rows_outside_cwd_rejected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils import tunability

        outside = tmp_path.parent / "outside.jsonl"
        _write_jsonl(outside, [{"prompt": "q", "response": "a"}])
        with pytest.raises((ValueError, OSError)):
            tunability._load_jsonl_rows(str(outside))

    def test_run_tunability_uses_injected_probe(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils import tunability

        cand = tunability.CandidateBase(
            name="t", repo_id="r", params_b=0.1, license_id="mit",
        )
        calls = []

        def fake_probe(c, ds, *, probe_steps, holdout_size):
            calls.append(c.name)
            return tunability.TunabilityResult(
                candidate=c, base_loss=2.0, probe_loss=1.0, delta=1.0,
                wall_clock_seconds=10.0, estimated_cost_usd=0.5,
            )

        report = tunability.run_tunability(
            candidates=[cand], dataset_path="d.jsonl", probe_fn=fake_probe
        )
        assert calls == ["t"]
        assert report.results[0].delta == pytest.approx(1.0)

    def test_cli_plan_only_live_text(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": "q", "response": "a"}])
        result = runner.invoke(
            app, ["tunability", "--dataset", str(data), "--plan-only", "--live"]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        assert "LIVE LoRA probe" in result.output

    def test_cli_live_flag_in_help(self) -> None:
        result = runner.invoke(app, ["tunability", "--help"])
        assert "--live" in result.output


# ===========================================================================
# #211 — capability live lm-eval runner
# ===========================================================================
def _install_fake_lm_eval(monkeypatch, *, results: dict, raise_for: set | None = None):
    raise_for = raise_for or set()

    def simple_evaluate(*, model, tasks, limit=None):
        task = tasks[0]
        if task in raise_for:
            raise RuntimeError(f"task {task} unavailable")
        return {"results": {task: results.get(task, {"acc,none": 0.5})}}

    class HFLM:
        def __init__(self, *, pretrained, device, batch_size):
            self.pretrained = pretrained

    fake = types.ModuleType("lm_eval")
    fake.simple_evaluate = simple_evaluate
    fake_models = types.ModuleType("lm_eval.models")
    fake_hf = types.ModuleType("lm_eval.models.huggingface")
    fake_hf.HFLM = HFLM
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    monkeypatch.setitem(sys.modules, "lm_eval.models", fake_models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", fake_hf)


class TestCapabilityLiveRunner:
    def test_primary_metric_prefers_acc_norm(self) -> None:
        from soup_cli.utils.capability_suite import _primary_metric

        key, val = _primary_metric({"acc,none": 0.4, "acc_norm,none": 0.6})
        assert key.startswith("acc_norm")
        assert val == pytest.approx(0.6)

    def test_primary_metric_skips_stderr(self) -> None:
        from soup_cli.utils.capability_suite import _primary_metric

        key, val = _primary_metric({"exact_match,none": 0.3, "exact_match_stderr,none": 0.1})
        assert "stderr" not in key
        assert val == pytest.approx(0.3)

    def test_run_capability_suite_tasks(self, monkeypatch) -> None:
        from soup_cli.utils.capability_suite import run_capability_suite

        _install_fake_lm_eval(monkeypatch, results={"arc_easy": {"acc,none": 0.71}})
        out = run_capability_suite(
            run_id="r1", model_id="m", tasks=["arc_easy"], limit=2
        )
        assert out["run_id"] == "r1"
        assert out["results"][0]["benchmark"] == "arc_easy"
        assert out["results"][0]["score"] == pytest.approx(0.71)

    def test_run_capability_suite_per_task_error_isolation(self, monkeypatch) -> None:
        from soup_cli.utils.capability_suite import run_capability_suite

        _install_fake_lm_eval(
            monkeypatch,
            results={"arc_easy": {"acc,none": 0.7}},
            raise_for={"bbeh_unregistered"},
        )
        out = run_capability_suite(
            run_id="r", model_id="m", tasks=["arc_easy", "bbeh_unregistered"]
        )
        assert "score" in out["results"][0]
        assert "error" in out["results"][1]

    def test_run_capability_suite_resolves_profile(self, monkeypatch) -> None:
        from soup_cli.utils.capability_suite import run_capability_suite

        _install_fake_lm_eval(monkeypatch, results={})
        out = run_capability_suite(run_id="r", model_id="m", suite="fast")
        names = {r["benchmark"] for r in out["results"]}
        assert names == {"mmlu-pro", "humaneval-plus"}

    def test_run_capability_empty_result_is_error(self, monkeypatch) -> None:
        from soup_cli.utils.capability_suite import run_capability_suite

        _install_fake_lm_eval(monkeypatch, results={"arc_easy": {}})
        out = run_capability_suite(run_id="r", model_id="m", tasks=["arc_easy"])
        assert out["results"][0]["error"] == "no scalar metric in task result"

    def test_run_capability_missing_lm_eval(self, monkeypatch) -> None:
        from soup_cli.utils.capability_suite import run_capability_suite

        # Force the import to fail.
        monkeypatch.setitem(sys.modules, "lm_eval", None)
        with pytest.raises(RuntimeError, match="lm-eval"):
            run_capability_suite(run_id="r", model_id="m", tasks=["arc_easy"])

    def test_run_capability_validation(self) -> None:
        from soup_cli.utils.capability_suite import run_capability_suite

        with pytest.raises(ValueError):
            run_capability_suite(run_id="", model_id="m", tasks=["x"])
        with pytest.raises(ValueError):
            run_capability_suite(run_id="r", model_id="", tasks=["x"])
        with pytest.raises(ValueError):
            run_capability_suite(run_id="r", model_id="m", tasks=["x"], limit=0)
        with pytest.raises(ValueError):
            run_capability_suite(run_id="r", model_id="m")  # neither suite nor tasks

    def test_cli_live_requires_model(self) -> None:
        result = runner.invoke(app, ["eval", "capability", "r", "--live"])
        assert result.exit_code == 2
        assert "--model" in result.output

    def test_cli_live_runs(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        _install_fake_lm_eval(monkeypatch, results={"arc_easy": {"acc,none": 0.66}})
        result = runner.invoke(
            app,
            ["eval", "capability", "r", "--live", "--model", "m",
             "--tasks", "arc_easy", "--limit", "1"],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        assert "arc_easy" in result.output

    def test_cli_manifest_default(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["eval", "capability", "r", "--suite", "fast", "--output", "out.json"]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        payload = json.loads(Path("out.json").read_text(encoding="utf-8"))
        assert "Manifest only" in payload["note"]


# ===========================================================================
# #212 — behavior live diff
# ===========================================================================
class TestBehaviorLive:
    def _patch_generators(self, monkeypatch, base_text: str, post_text: str):
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: ("model", "tok", "cpu"),
        )

        def fake_make_generator(model_id, *, adapter=None, **kwargs):
            text = post_text if adapter else base_text
            return lambda prompt: text

        monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)

    def test_run_behavior_live_returns_report(self, monkeypatch) -> None:
        from soup_cli.utils.behavior_battery import BehaviorDiffReport, run_behavior_live

        self._patch_generators(monkeypatch, "safe answer", "safe answer")
        report = run_behavior_live(
            run_id="r", battery="xstest", base_model="m", adapter="adp",
        )
        assert isinstance(report, BehaviorDiffReport)
        assert report.battery == "xstest"

    def test_run_behavior_live_rejects_empty_model(self) -> None:
        from soup_cli.utils.behavior_battery import run_behavior_live

        with pytest.raises(ValueError):
            run_behavior_live(run_id="r", battery="xstest", base_model="")

    def test_run_behavior_live_rejects_bad_max_probes(self, monkeypatch) -> None:
        from soup_cli.utils.behavior_battery import run_behavior_live

        self._patch_generators(monkeypatch, "x", "x")
        with pytest.raises(ValueError):
            run_behavior_live(
                run_id="r", battery="xstest", base_model="m", max_probes=0
            )

    def test_cli_live_diff(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        self._patch_generators(monkeypatch, "ok", "ok")
        result = runner.invoke(
            app, ["eval", "behavior", "r", "--battery", "xstest", "--base-model", "m"]
        )
        # Identical pre/post "ok" responses fail the xstest "safe" oracle →
        # MAJOR → exit 2 (the live diff ran and rendered before the gate).
        assert result.exit_code == 2, (result.output, result.exception)
        assert "live" in result.output.lower()

    def test_cli_live_diff_ok(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        # Responses containing the "safe" oracle word → agreement 1.0 → OK.
        self._patch_generators(monkeypatch, "this is safe", "this is safe")
        result = runner.invoke(
            app, ["eval", "behavior", "r", "--battery", "xstest", "--base-model", "m"]
        )
        assert result.exit_code == 0, (result.output, result.exception)

    def test_cli_base_model_in_help(self) -> None:
        result = runner.invoke(app, ["eval", "behavior", "--help"])
        assert "--base-model" in result.output


# ===========================================================================
# #165 — diagnose live runners
# ===========================================================================
class TestDiagnoseLive:
    def _patch_live_eval(self, monkeypatch, base_text="hello", post_text="hello"):
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: ("model", "tok", "cpu"),
        )

        def fake_make_generator(model_id, *, adapter=None, **kwargs):
            text = post_text if adapter else base_text
            return lambda prompt: text

        def fake_make_multi(model_id, *, adapter=None, **kwargs):
            return lambda prompt, k: [f"sample {i}" for i in range(k)]

        monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)
        monkeypatch.setattr(live_eval, "make_multi_generator", fake_make_multi)

    def test_token_f1_helper(self) -> None:
        from soup_cli.utils.live_eval import token_f1

        assert token_f1("a b c", "a b c") == pytest.approx(1.0)
        assert token_f1("x", "y") == 0.0

    def test_looks_like_json_dataset(self) -> None:
        from soup_cli.utils.diagnose.live import _looks_like_json_dataset

        json_rows = [{"response": '{"k": 1}'} for _ in range(5)]
        text_rows = [{"response": "plain text"} for _ in range(5)]
        assert _looks_like_json_dataset(json_rows) is True
        assert _looks_like_json_dataset(text_rows) is False

    def test_load_dataset_rows_outside_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.diagnose.live import _load_dataset_rows

        outside = tmp_path.parent / "x.jsonl"
        _write_jsonl(outside, [{"prompt": "q", "response": "a"}])
        with pytest.raises((ValueError, OSError)):
            _load_dataset_rows(str(outside))

    def test_load_adapter_pair_aliases_when_no_adapter(self, monkeypatch) -> None:
        self._patch_live_eval(monkeypatch)
        from soup_cli.utils.diagnose.live import load_adapter_pair

        closures = load_adapter_pair("m", None)
        assert set(closures) == {"base_gen", "adapter_gen", "base_multi", "adapter_multi"}
        assert closures["base_gen"] is closures["adapter_gen"]

    def test_run_live_diagnose_returns_report(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._patch_live_eval(monkeypatch)
        from soup_cli.utils.diagnose.live import run_live_diagnose
        from soup_cli.utils.diagnose.report import FAILURE_MODES

        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(8)])
        report = run_live_diagnose(
            run_id="r", base="m", adapter=None, dataset_path=str(data)
        )
        # All 6 modes present.
        assert set(report.scores) >= set(FAILURE_MODES)
        assert report.overall in {"OK", "MINOR", "MAJOR"}

    def test_run_live_diagnose_format_neutral_when_not_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._patch_live_eval(monkeypatch)
        from soup_cli.utils.diagnose.live import run_live_diagnose

        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": "q", "response": "plain text"} for _ in range(6)])
        report = run_live_diagnose(run_id="r", base="m", dataset_path=str(data))
        assert "not JSON" in report.scores["format"].evidence

    def test_run_live_diagnose_no_dataset(self, monkeypatch) -> None:
        self._patch_live_eval(monkeypatch)
        from soup_cli.utils.diagnose.live import run_live_diagnose
        from soup_cli.utils.diagnose.report import FAILURE_MODES

        report = run_live_diagnose(run_id="r", base="m")
        # refusal still runs; dataset-driven probes neutral but present.
        assert set(report.scores) >= set(FAILURE_MODES)

    def test_run_live_diagnose_rejects_empty_base(self) -> None:
        from soup_cli.utils.diagnose.live import run_live_diagnose

        with pytest.raises(ValueError):
            run_live_diagnose(run_id="r", base="")

    def test_cli_live_diagnose(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._patch_live_eval(monkeypatch)
        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(8)])
        result = runner.invoke(
            app, ["diagnose", "r1", "--base-model", "m", "--dataset", str(data)]
        )
        assert result.exit_code in (0, 2), (result.output, result.exception)
        assert "overall" in result.output.lower()

    def test_cli_base_model_in_help(self) -> None:
        result = runner.invoke(app, ["diagnose", "--help"])
        assert "--base-model" in result.output
        assert "--tokenizer" in result.output


# ===========================================================================
# Patch invariants
# ===========================================================================
class TestPatchInvariants:
    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        assert __version__ == "0.71.7"

    def test_no_top_level_heavy_imports(self) -> None:
        for mod in (
            "live_eval.py",
            "diagnose/live.py",
        ):
            src = (
                Path(__file__).resolve().parent.parent
                / "src" / "soup_cli" / "utils" / mod
            ).read_text(encoding="utf-8")
            assert "\nimport torch" not in src, mod
            assert "\nimport transformers" not in src, mod
            assert "\nimport peft" not in src, mod
