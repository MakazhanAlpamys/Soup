"""Offline native harness oracles; local qualification needs MLX and the eval extra."""

import math
from contextlib import closing
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("lm_eval")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402


@pytest.fixture
def fixed_model(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_generation

    vocab = {"<unk>": 0, "<eos>": 1, "A": 2, "Ġ": 3, "B": 4, "ĠB": 5, "C": 6, "ĠC": 7, "AB": 8}
    tokenizer = Tokenizer(
        models.BPE(
            vocab,
            [("Ġ", "B"), ("Ġ", "C"), ("A", "B")],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.name_or_path = "offline-fixed-bpe"
    table = np.linspace(-2, 1, 81).reshape(9, 9)
    table[2, 5] = 4
    table[5, 1] = 4
    bias = np.linspace(-0.1, 0.1, 9)

    class FixedLogitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array(table, dtype=mx.float32)
            self.batches = []

        def __call__(self, tokens):
            self.batches.append(tokens.tolist())
            return self.weight[tokens] + (
                mx.arange(tokens.shape[1])[None, :, None] * mx.array(bias)[None, None, :]
            )

    model = FixedLogitModel()
    monkeypatch.setattr(mlx_generation, "load_native_model", lambda *a, **k: (model, tokenizer))
    return model, tokenizer, table, bias


def oracle(context, continuation, table, bias, max_length):
    """Independent scalar log-sum-exp oracle on the exact causal input window."""
    window = (context + continuation)[-(max_length + 1) :]
    inputs = window[:-1]
    start = len(inputs) - len(continuation)
    total, greedy = 0.0, True
    for position, target in enumerate(continuation, start):
        logits = [float(v + position * b) for v, b in zip(table[inputs[position]], bias)]
        maximum = max(logits)
        normalizer = maximum + math.log(sum(math.exp(v - maximum) for v in logits))
        total += logits[target] - normalizer
        greedy &= logits.index(maximum) == target
    return total, greedy


def test_joint_boundary_empty_context_eos_padding_and_original_order(fixed_model):
    from soup_cli.eval.mlx_benchmark import make_native_harness

    model, tokenizer, table, bias = fixed_model
    lm = make_native_harness("unused", batch_size=3, max_length=6, max_tokens=2)
    requests = [
        SimpleNamespace(args=("A ", "B<eos>")),
        SimpleNamespace(args=("", "B")),
        SimpleNamespace(args=("A", " C")),
        SimpleNamespace(args=("A A A A A A ", "B")),
    ]
    # Real BPE merges the context's trailing space with the continuation.
    assert tokenizer.encode("A ", add_special_tokens=False) == [2, 3]
    assert tokenizer.encode("A B", add_special_tokens=False) == [2, 5]
    expected_tokens = [
        ([2], [5, 1]),
        ([1], [4]),
        ([2], [7]),
        ([2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2], [5]),
    ]
    expected = [oracle(ctx, cont, table, bias, 6) for ctx, cont in expected_tokens]
    measured = lm.loglikelihood(requests)
    assert [row[0] for row in measured] == pytest.approx([row[0] for row in expected], abs=2e-6)
    assert [row[1] for row in measured] == [row[1] for row in expected]
    assert [row[1] for row in measured[:3]] == [True, False, False]
    assert lm.scored_token_counts == [2, 1, 1, 1]
    assert len(model.batches[0]) == 3 and len(model.batches[1]) == 1
    assert model.batches[1][0] == [3, 2, 3, 2, 3, 2]


def test_continuation_at_cap_and_adjacent_refusals(fixed_model):
    from soup_cli.eval.mlx_benchmark import make_native_harness

    _, _, table, bias = fixed_model
    lm = make_native_harness("unused", batch_size=2, max_length=3, max_tokens=2)
    result = lm.loglikelihood([SimpleNamespace(args=("A A A A ", "B B B"))])
    expected = oracle([2, 3, 2, 3, 2, 3, 2], [5, 5, 5], table, bias, 3)
    assert result[0][0] == pytest.approx(expected[0], abs=2e-6)
    assert result[0][1] == expected[1]
    assert lm.scored_token_counts == [3]
    with pytest.raises(ValueError, match="continuation.*max-length"):
        lm.loglikelihood([SimpleNamespace(args=("A ", "B B B B"))])
    with pytest.raises(ValueError, match="boundary"):
        lm.loglikelihood([SimpleNamespace(args=("A", "B"))])
    with pytest.raises(ValueError, match="continuation"):
        lm.loglikelihood([SimpleNamespace(args=("", ""))])
    with pytest.raises(ValueError, match="rolling"):
        lm.loglikelihood_rolling([SimpleNamespace(args=("A B",))])


@pytest.fixture
def offline_catalog(monkeypatch):
    """Real harness tasks; only dataset/catalog I/O is replaced with local rows."""
    import lm_eval.tasks
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    def configure(name, rows, *, generation=False, filters=None, until=None):
        config = {
            "task": name,
            "dataset_path": "offline-fixture",
            "custom_dataset": lambda **kwargs: DatasetDict(validation=Dataset.from_list(rows)),
            "validation_split": "validation",
            "num_fewshot": 0,
            "doc_to_text": "prompt",
            "doc_to_target": "answer" if generation else "label",
            "output_type": "generate_until" if generation else "multiple_choice",
            "metric_list": [
                {
                    "metric": "exact_match" if generation else "acc",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
            "metadata": {"version": "offline-1"},
        }
        if generation:
            config["generation_kwargs"] = {
                "until": until or ["STOP"],
                "do_sample": False,
                "temperature": 0,
                "max_gen_toks": 3,
            }
        else:
            config["doc_to_choice"] = "choices"
        if filters:
            config["filter_list"] = filters
        return ConfigurableTask(config=config)

    manager = lm_eval.tasks.TaskManager
    monkeypatch.setattr(
        lm_eval.tasks,
        "TaskManager",
        lambda **kwargs: manager(
            include_defaults=False,
            **kwargs,
        ),
    )
    catalog = {}
    monkeypatch.setattr(
        lm_eval.tasks, "get_task_dict", lambda names, **kw: {name: catalog[name] for name in names}
    )
    return catalog, configure


def test_real_harness_mcq_public_tracker_and_native_oracle(
    fixed_model,
    offline_catalog,
    tmp_path,
    monkeypatch,
):
    import json

    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.experiment.tracker import ExperimentTracker

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "scores.db"))
    model, _, table, bias = fixed_model
    catalog, configure = offline_catalog
    catalog["hellaswag"] = configure(
        "hellaswag",
        [
            {"prompt": "A", "choices": ["B", "C"], "label": 0},
            {"prompt": "A", "choices": ["C", "B"], "label": 1},
            {"prompt": "A", "choices": ["B", "C"], "label": 1},
        ],
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "benchmark",
            "--model",
            "offline-model",
            "--backend",
            "mlx",
            "--benchmarks",
            "hellaswag",
            "--batch-size",
            "3",
            "--limit",
            "2",
            "--max-length",
            "6",
            "--max-tokens",
            "2",
            "--seed",
            "11",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    from tests.conftest import strip_ansi

    plain = strip_ansi(result.output)
    assert "1.0000" in plain and "1 of 1" in plain
    with closing(ExperimentTracker()) as tracker:
        rows = tracker.get_eval_results()
    assert len(rows) == 1 and rows[0]["score"] == 1.0
    evidence = json.loads(rows[0]["details_json"])
    assert evidence["n_samples"] == {"original": 3, "effective": 2}
    assert evidence["task_version"] == "offline-1"
    assert evidence["task_config"]["num_fewshot"] == 0
    assert evidence["native"]["harness_version"] in {"0.4.12", "0.4.13"}
    assert evidence["native"]["seed"] == 11
    assert evidence["native"]["limit"] == 2
    assert evidence["native"]["scored_token_counts"] == [1, 1, 1, 1]
    assert evidence["native"]["process_peak_memory_bytes"] > 0
    samples = sorted(evidence["samples"], key=lambda row: row["doc_id"])
    expected_b = oracle([2], [5], table, bias, 6)
    expected_c = oracle([2], [7], table, bias, 6)
    assert [item[0][0] for item in samples[0]["resps"]] == pytest.approx(
        [expected_b[0], expected_c[0]],
        abs=2e-6,
    )
    assert [item[0][0] for item in samples[1]["resps"]] == pytest.approx(
        [expected_c[0], expected_b[0]],
        abs=2e-6,
    )
    assert model.batches  # Forward really ran on the fixed native table.


@pytest.mark.parametrize("failure", ["empty", "nan", "missing", "no-samples", "runtime"])
def test_harness_failure_never_saves_partial_success(
    fixed_model,
    offline_catalog,
    tmp_path,
    monkeypatch,
    failure,
):
    import lm_eval
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.experiment.tracker import ExperimentTracker

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "failure.db"))
    catalog, configure = offline_catalog
    catalog["hellaswag"] = configure(
        "hellaswag",
        [
            {"prompt": "A", "choices": ["B", "C"], "label": 0},
            {"prompt": "A", "choices": ["C", "B"], "label": 1},
        ],
    )
    real = lm_eval.simple_evaluate

    def corrupt(*args, **kwargs):
        # Run real task/request/filter/aggregation first, then inject a broken
        # external harness result or runtime failure at that boundary.
        measured = real(*args, **kwargs)
        if failure == "runtime":
            raise RuntimeError("fixture harness failure")
        if failure == "empty":
            return {}
        if failure == "nan":
            measured["results"]["hellaswag"]["acc,none"] = float("nan")
        if failure == "missing":
            del measured["results"]["hellaswag"]
        if failure == "no-samples":
            measured["samples"]["hellaswag"] = []
        return measured

    monkeypatch.setattr(lm_eval, "simple_evaluate", corrupt)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "benchmark",
            "--model",
            "offline-model",
            "--backend",
            "mlx",
            "--benchmarks",
            "hellaswag",
            "--max-length",
            "6",
            "--max-tokens",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert (
        "measurement evidence" in result.output
        if failure != "runtime"
        else "fixture harness failure" in result.output
    )
    assert "Results saved" not in result.output
    with closing(ExperimentTracker()) as tracker:
        assert tracker.get_eval_results() == []


@pytest.mark.parametrize(
    "corruption",
    [None, "minimal", "arguments", "resps", "filtered_resps", "metric", "nan", "boolean",
     "disjoint", "identity", "null-arguments", "null-resps", "null-filtered_resps"],
)
def test_sample_evidence_is_complete_and_matches_documents_across_filters(
    fixed_model, offline_catalog, tmp_path, monkeypatch, corruption,
):
    import json

    import lm_eval
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.experiment.tracker import ExperimentTracker

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "evidence.db"))
    catalog, configure = offline_catalog
    catalog["hellaswag"] = configure(
        "hellaswag",
        [{"prompt": "A", "choices": ["B", "C"], "label": 0},
         {"prompt": "A", "choices": ["C", "B"], "label": 1}],
        filters=[{"name": name, "filter": [{"function": "take_first"}]}
                 for name in ("none", "second")],
    )
    real = lm_eval.simple_evaluate

    def measured_then_corrupted(*args, **kwargs):
        result = real(*args, **kwargs)
        samples = result["samples"]["hellaswag"]
        if corruption == "minimal":
            result["samples"]["hellaswag"] = [
                {"doc_id": row["doc_id"], "filter": row["filter"]} for row in samples
            ]
        elif corruption in {"arguments", "resps", "filtered_resps"}:
            samples[0][corruption] = []
        elif corruption == "metric":
            del samples[0]["acc"]
        elif corruption == "nan":
            samples[0]["acc"] = float("nan")
        elif corruption == "boolean":
            samples[0]["acc"] = True
        elif corruption == "disjoint":
            for row in samples:
                if row["filter"] == "second":
                    row["doc_id"] += 10
        elif corruption == "identity":
            next(row for row in samples if row["filter"] == "second")["doc_hash"] = "other"
        elif corruption and corruption.startswith("null-"):
            field = corruption.removeprefix("null-")
            samples[0][field] = [None] * len(samples[0][field])
        return result

    monkeypatch.setattr(lm_eval, "simple_evaluate", measured_then_corrupted)
    result = CliRunner().invoke(
        app,
        ["eval", "benchmark", "--backend", "mlx", "--model", "offline-model",
         "--benchmarks", "hellaswag", "--limit", "2", "--max-length", "6",
         "--max-tokens", "2"],
    )
    with closing(ExperimentTracker()) as tracker:
        persisted = tracker.get_eval_results()
    if corruption is None:
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(persisted) == 1 and persisted[0]["score"] == 1.0
        samples = json.loads(persisted[0]["details_json"])["samples"]
        assert {(row["filter"], row["doc_id"]) for row in samples} == {
            (name, doc) for name in ("none", "second") for doc in (0, 1)
        }
        assert all(row["acc"] == 1.0 and row["resps"] for row in samples)
    else:
        assert result.exit_code == 1
        assert "sample" in result.output and "evidence" in result.output
        assert "Results saved" not in result.output
        assert persisted == []


@pytest.mark.parametrize("quantized", [False, True])
def test_trained_adapter_harness_reload_generation_and_base_control(
    tmp_path,
    monkeypatch,
    quantized,
    offline_catalog,
):
    """Tiny real LoRA: direct oracle, actual batch generation, harness and reload."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from mlx.utils import tree_flatten
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from soup_cli.eval.mlx_benchmark import make_native_harness, run_native_benchmark
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from soup_cli.utils.mlx_generation import load_native_model
    from tests.test_mlx_dora_native import _checkpoint, _config, _rows

    monkeypatch.chdir(tmp_path)
    base = _checkpoint(tmp_path / "base", quantized)
    adapter = tmp_path / "adapter"
    cfg = _config(base, adapter, dora=False, quantized=quantized, targets=["lm_head"])
    cfg.training.lr, cfg.training.epochs = 0.01, 4
    trainer = MLXSFTTrainerWrapper(cfg)
    trainer.setup(_rows())
    frozen = dict(tree_flatten(trainer.model.parameters()))
    mx.eval(frozen)
    trainer.train()
    trained = dict(tree_flatten(trainer.model.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        assert mx.array_equal(
            value, trained.get(f"{parent}.linear.{leaf}", trained.get(name))
        ).item()
    saved = mx.load(str(adapter / "adapters.safetensors"))
    assert any(
        mx.any(value != 0).item() for name, value in saved.items() if name.endswith("lora_b")
    )
    native, tokenizer = load_native_model(str(adapter))
    control, _ = load_native_model(str(base))
    prompt = [5, 6, 7, 5]
    expected = native(mx.array([prompt]))[0, -1].astype(mx.float32)
    expected = expected - mx.logsumexp(expected)
    base_score = control(mx.array([prompt]))[0, -1].astype(mx.float32)
    base_score = base_score - mx.logsumexp(base_score)
    target = 6
    assert abs(expected[target].item() - base_score[target].item()) > 0.1
    mx.save_safetensors(str(tmp_path / "expected.safetensors"), {"scores": expected})
    lm = make_native_harness(str(adapter), batch_size=2, max_length=12, max_tokens=3)
    value = lm.loglikelihood([SimpleNamespace(args=("alpha beta gamma alpha", " beta"))])[0]
    assert value[0] == pytest.approx(expected[target].item(), abs=2e-5)
    assert value[1] is (expected.argmax().item() == target)
    assert lm.scored_token_counts == [1]
    oracle_text = "".join(
        response.text
        for response in stream_generate(
            native,
            tokenizer,
            prompt=prompt,
            max_tokens=3,
            sampler=make_sampler(temp=0),
        )
    )
    generated = lm.generate_until(
        [
            SimpleNamespace(
                args=(
                    "alpha beta gamma alpha",
                    {"until": ["STOP"], "max_gen_toks": 3},
                )
            )
        ]
    )
    assert generated == [oracle_text]
    varied = [
        ("alpha beta " * 10, {"until": [], "max_gen_toks": 1}),
        ("alpha beta gamma alpha", {"until": [], "max_gen_toks": 9}),
        ("", {"until": [], "max_gen_toks": 2}),
    ]
    reference = []
    for text, options in varied:
        cap = min(options["max_gen_toks"], 3)
        ids = tokenizer.encode(text) or [tokenizer.eos_token_id]
        ids = ids[-(12 - cap) :]
        reference.append(
            "".join(
                response.text
                for response in stream_generate(
                    native,
                    tokenizer,
                    prompt=ids,
                    max_tokens=cap,
                    sampler=make_sampler(temp=0),
                )
            )
        )
    assert lm.generate_until([SimpleNamespace(args=row) for row in varied]) == reference
    assert [row["max_tokens"] for row in lm.generation_limits[-3:]] == [1, 3, 2]
    assert lm.generation_limits[-3]["retained_prompt_tokens"] == 11
    assert lm.generation_limits[-3]["prompt_tokens"] == 20
    stop = oracle_text[:1]
    assert stop
    assert lm.generate_until(
        [
            SimpleNamespace(
                args=(
                    "alpha beta gamma alpha",
                    {"until": [stop], "max_gen_toks": 3},
                )
            )
        ]
    ) == [""]

    # True EOS in the engine is handled separately from post-decoding substring stops.
    class EosOnly(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        @property
        def layers(self):
            return self.wrapped.layers

        def __call__(self, tokens, cache=None):
            logits = self.wrapped(tokens, cache=cache)
            return mx.full(logits.shape, -100.0).at[..., tokenizer.eos_token_id].add(200)

    from soup_cli.utils import mlx_generation

    with monkeypatch.context() as patch:
        patch.setattr(
            mlx_generation, "load_native_model", lambda *a, **kw: (EosOnly(native), tokenizer)
        )
        eos_lm = make_native_harness("unused", max_length=12, max_tokens=3)
        assert eos_lm.generate_until([SimpleNamespace(args=("alpha", {"until": []}))]) == [""]

    catalog, configure = offline_catalog
    catalog["hellaswag"] = configure(
        "hellaswag",
        [
            {"prompt": "alpha beta gamma alpha", "choices": ["beta", "gamma"], "label": 0},
            {"prompt": "alpha beta gamma alpha", "choices": ["gamma", "beta"], "label": 1},
        ],
    )
    catalog["gsm8k"] = configure(
        "gsm8k",
        [
            {"prompt": "alpha beta gamma alpha", "answer": oracle_text},
            {"prompt": "alpha beta gamma alpha", "answer": oracle_text},
        ],
        generation=True,
        filters=[
            {"name": "strict-match", "filter": [{"function": "take_first"}]},
            {"name": "flexible-extract", "filter": [{"function": "take_first"}]},
        ],
    )
    options = dict(
        base_model=None,
        num_fewshot=0,
        batch_size=2,
        limit=2,
        max_length=12,
        max_tokens=3,
        seed=7,
        trust_remote_code=False,
    )
    measured = run_native_benchmark(str(adapter), ["hellaswag", "gsm8k"], **options)
    baseline = run_native_benchmark(str(base), ["hellaswag"], **options)
    assert measured["results"]["gsm8k"]["exact_match,strict-match"] == 1
    assert measured["results"]["gsm8k"]["exact_match,flexible-extract"] == 1
    assert measured["n-samples"]["gsm8k"]["effective"] == 2
    assert len(measured["samples"]["gsm8k"]) == 4
    assert measured["_soup_native"]["selected_metrics"]["gsm8k"] == "exact_match,strict-match"
    adapted_ll = measured["samples"]["hellaswag"][0]["resps"][0][0][0]
    baseline_ll = baseline["samples"]["hellaswag"][0]["resps"][0][0][0]
    assert abs(adapted_ll - baseline_ll) > 0.1
    assert measured["_soup_native"]["adapter"] == str(adapter)
    assert measured["_soup_native"]["artifact_sha256"]["adapters.safetensors"]
    (tmp_path / "harness-evidence.json").write_text(json.dumps(measured, indent=2))
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.experiment.tracker import ExperimentTracker

    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "adapter-benchmark.db"))
    public = CliRunner().invoke(
        app,
        [
            "eval",
            "benchmark",
            "--backend",
            "mlx",
            "--model",
            str(adapter),
            "--benchmarks",
            "hellaswag,gsm8k",
            "--fewshot",
            "0",
            "--limit",
            "2",
            "--max-length",
            "12",
            "--max-tokens",
            "3",
            "--batch-size",
            "2",
        ],
    )
    assert public.exit_code == 0, (public.output, repr(public.exception))
    with closing(ExperimentTracker()) as tracker:
        persisted = tracker.get_eval_results()
    assert {row["benchmark"] for row in persisted} == {"hellaswag", "gsm8k"}
    assert all(
        json.loads(row["details_json"])["native"]["adapter"] == str(adapter) for row in persisted
    )
    script = """
import json
from types import SimpleNamespace
import mlx.core as mx
from soup_cli.eval.mlx_benchmark import make_native_harness
lm=make_native_harness('adapter', max_length=12, max_tokens=3)
value=lm.loglikelihood([SimpleNamespace(args=('alpha beta gamma alpha',' beta'))])[0]
expected=mx.load('expected.safetensors')['scores'][6].item()
assert abs(value[0]-expected)<2e-5, (value,expected)
print(json.dumps({'score':value[0]}))
"""
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert abs(json.loads(completed.stdout)["score"] - value[0]) < 2e-5


@pytest.mark.parametrize(
    "options,diagnostic",
    [
        ({"do_sample": True}, "greedy"),
        ({"temperature": 0.5}, "greedy"),
        ({"top_p": 0.9}, "options"),
        ({"max_gen_tokens": 3}, "options"),
        ({"max_gen_toks": 0}, "positive integer"),
        ({"max_gen_toks": True}, "positive integer"),
        ({"until": [""]}, "until"),
    ],
)
def test_generation_unsupported_controls_are_reached_before_engine(
    fixed_model,
    options,
    diagnostic,
):
    from soup_cli.eval.mlx_benchmark import make_native_harness

    lm = make_native_harness("unused", max_length=6, max_tokens=2)
    with pytest.raises(ValueError, match=diagnostic):
        lm.generate_until([SimpleNamespace(args=("A", options))])


@pytest.mark.parametrize("version", ["0.4.11", "0.4.14"])
def test_native_harness_version_refusal_has_adjacent_supported_controls(
    fixed_model,
    monkeypatch,
    version,
):
    from soup_cli.eval import mlx_benchmark

    original = mlx_benchmark.metadata.version
    monkeypatch.setattr(
        mlx_benchmark.metadata,
        "version",
        lambda name: version if name == "lm-eval" else original(name),
    )
    with pytest.raises(ValueError, match="lm-eval>=0.4.12,<0.4.14"):
        mlx_benchmark.make_native_harness("unused", max_length=6, max_tokens=2)
    monkeypatch.setattr(
        mlx_benchmark.metadata,
        "version",
        lambda name: "0.4.12" if name == "lm-eval" else original(name),
    )
    lm = mlx_benchmark.make_native_harness("unused", max_length=6, max_tokens=2)
    score = lm.loglikelihood([SimpleNamespace(args=("A ", "B"))])[0]
    assert math.isfinite(score[0]) and score[1] is True
