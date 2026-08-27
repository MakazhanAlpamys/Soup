"""Regression tests for #549: durable, exactly-once Best-of-N resume."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


def _args(tmp_path, *extra: str) -> list[str]:
    return [
        "best-of-n",
        "--provider",
        "ollama",
        "--model",
        "sampler",
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        "--n",
        "3",
        "--judge",
        "ollama://judge",
        "--output",
        str(tmp_path / "sft.jsonl"),
        "--emit-pairs",
        str(tmp_path / "dpo.jsonl"),
        *extra,
    ]


def _local_args(tmp_path, *extra: str) -> list[str]:
    return [
        "best-of-n",
        "--base",
        "local-model",
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        "--n",
        "2",
        "--judge",
        "ollama://judge",
        "--output",
        str(tmp_path / "sft.jsonl"),
        "--seed",
        "17",
        *extra,
    ]


def test_late_failure_resumes_without_replaying_completed_prefix(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompts = ["first-private-prompt", "second-private-prompt", "third-private-prompt"]
    (tmp_path / "prompts.jsonl").write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )

    sampling_calls: list[str] = []
    judge_calls: list[str] = []
    fail_second = {"value": True}

    def make_generate(*_args, **_kwargs):
        def generate(prompt: str) -> str:
            sampling_calls.append(prompt)
            return "x" * len(sampling_calls)

        return generate

    class Judge:
        def evaluate(self, prompt: str, response: str):
            judge_calls.append(prompt)
            if fail_second["value"] and prompt == prompts[1]:
                raise RuntimeError("simulated late judge outage containing private data")
            return SimpleNamespace(weighted_score=float(len(response)))

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", make_generate)
    monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", lambda **_kwargs: Judge())

    first = CliRunner().invoke(app, _args(tmp_path))
    assert first.exit_code == 1, (first.output, repr(first.exception))
    assert "1/3 prompts" in first.output
    assert "--resume" in first.output
    assert "simulated late" not in first.output
    assert not (tmp_path / "sft.jsonl").exists()
    assert not (tmp_path / "dpo.jsonl").exists()
    checkpoint = tmp_path / "sft.jsonl.checkpoint.jsonl"
    assert checkpoint.exists()
    if os.name != "nt":
        assert checkpoint.stat().st_mode & 0o777 == 0o600

    sample_split = len(sampling_calls)
    judge_split = len(judge_calls)
    fail_second["value"] = False
    resumed = CliRunner().invoke(app, _args(tmp_path, "--resume"))
    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    assert prompts[0] not in sampling_calls[sample_split:]
    assert prompts[0] not in judge_calls[judge_split:]
    assert sampling_calls[sample_split:] == [prompts[1]] * 3 + [prompts[2]] * 3
    assert judge_calls[judge_split:] == [prompts[1]] * 3 + [prompts[2]] * 3

    sft_rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert [row["messages"][0]["content"] for row in sft_rows] == prompts
    assert len((tmp_path / "dpo.jsonl").read_text().splitlines()) == 3

    manifest = json.loads((tmp_path / "sft.jsonl.manifest.json").read_text())
    assert manifest["schema"] == "soup.best_of_n.manifest.v1"
    assert manifest["sft"]["rows"] == 3
    assert manifest["dpo"]["rows"] == 3
    assert manifest["sft"]["sha256"] == hashlib.sha256(
        (tmp_path / "sft.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["dpo"]["sha256"] == hashlib.sha256(
        (tmp_path / "dpo.jsonl").read_bytes()
    ).hexdigest()

    stable_sft = (tmp_path / "sft.jsonl").read_bytes()
    stable_dpo = (tmp_path / "dpo.jsonl").read_bytes()
    sample_split = len(sampling_calls)
    judge_split = len(judge_calls)
    replay = CliRunner().invoke(app, _args(tmp_path, "--resume"))
    assert replay.exit_code == 0, (replay.output, repr(replay.exception))
    assert sampling_calls[sample_split:] == []
    assert judge_calls[judge_split:] == []
    assert (tmp_path / "sft.jsonl").read_bytes() == stable_sft
    assert (tmp_path / "dpo.jsonl").read_bytes() == stable_dpo


@pytest.mark.parametrize("failure_stage", ["sampler", "judge"])
def test_late_value_error_reports_checkpoint_recovery(
    tmp_path, monkeypatch, failure_stage
):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompts = ["completed prompt", "failing prompt"]
    (tmp_path / "prompts.jsonl").write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )

    def make_generate(*_args, **_kwargs):
        def generate(prompt: str) -> str:
            if failure_stage == "sampler" and prompt == prompts[1]:
                raise ValueError("private sampler failure")
            return "candidate"

        return generate

    class Judge:
        def evaluate(self, prompt: str, _response: str):
            if failure_stage == "judge" and prompt == prompts[1]:
                raise ValueError("private judge failure")
            return SimpleNamespace(weighted_score=1.0)

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", make_generate)
    monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", lambda **_kwargs: Judge())

    result = CliRunner().invoke(app, _args(tmp_path))

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "1/2 prompts" in result.output
    assert "--resume" in result.output
    assert "sft.jsonl.checkpoint.jsonl" in result.output
    assert "private" not in result.output
    checkpoint_lines = (tmp_path / "sft.jsonl.checkpoint.jsonl").read_text().splitlines()
    assert len(checkpoint_lines) == 2


def test_resume_rejects_changed_run_before_sampling(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"one"}\n', encoding="utf-8")
    generate_calls: list[str] = []

    def make_generate(*_args, **_kwargs):
        def generate(prompt: str) -> str:
            generate_calls.append(prompt)
            return "candidate"

        return generate

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", make_generate)
    monkeypatch.setattr(
        "soup_cli.eval.judge.JudgeEvaluator",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda _prompt, response: SimpleNamespace(
                weighted_score=float(len(response))
            )
        ),
    )

    first = CliRunner().invoke(app, _args(tmp_path))
    assert first.exit_code == 0, (first.output, repr(first.exception))
    generate_calls.clear()
    changed = _args(tmp_path, "--resume")
    changed[changed.index("3")] = "4"
    result = CliRunner().invoke(app, changed)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "does not match prompts or run configuration" in result.output
    assert generate_calls == []


def test_final_datasets_use_exact_utf8_bytes_for_manifest_hashes(tmp_path, monkeypatch):
    from soup_cli.commands.data import app
    from soup_cli.utils.paths import atomic_write_text as real_text_write

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text(
        '{"prompt":"one"}\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        "soup_cli.utils.magpie.make_magpie_generate_fn",
        lambda *_args, **_kwargs: lambda _prompt: "candidate",
    )
    monkeypatch.setattr(
        "soup_cli.eval.judge.JudgeEvaluator",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda _prompt, response: SimpleNamespace(
                weighted_score=float(len(response))
            )
        ),
    )

    def reject_dataset_text_writes(text, path, *, field):
        if field in {"output", "emit-pairs"}:
            raise AssertionError("dataset JSONL must use exact UTF-8 byte writes")
        return real_text_write(text, path, field=field)

    monkeypatch.setattr(
        "soup_cli.utils.paths.atomic_write_text", reject_dataset_text_writes
    )
    result = CliRunner().invoke(app, _args(tmp_path))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    manifest = json.loads((tmp_path / "sft.jsonl.manifest.json").read_text())
    assert manifest["sft"]["sha256"] == hashlib.sha256(
        (tmp_path / "sft.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["dpo"]["sha256"] == hashlib.sha256(
        (tmp_path / "dpo.jsonl").read_bytes()
    ).hexdigest()


def test_checkpoint_rejects_duplicate_or_reordered_indexes(tmp_path, monkeypatch):
    from soup_cli.utils.best_of_n_checkpoint import load_checkpoint

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        '{"_best_of_n_checkpoint":{"version":1,"run_digest":"d","total":2}}\n'
        '{"index":1,"sft":{},"dpo":null}\n',
        encoding="utf-8",
    )
    try:
        load_checkpoint(str(checkpoint), digest="d", total=2)
    except ValueError as exc:
        assert "sequential and exactly once" in str(exc)
    else:
        raise AssertionError("non-sequential checkpoint must fail closed")


def test_checkpoint_symlink_is_rejected(tmp_path, monkeypatch):
    import pytest

    from soup_cli.utils.best_of_n_checkpoint import load_checkpoint

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "checkpoint.jsonl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        load_checkpoint(str(link), digest="d", total=1)


def test_checkpoint_entry_digest_detects_parseable_corruption(tmp_path, monkeypatch):
    from soup_cli.utils.best_of_n_checkpoint import (
        append_checkpoint,
        initialise_checkpoint,
        load_checkpoint,
    )

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    initialise_checkpoint(str(checkpoint), digest="d", total=1)
    append_checkpoint(
        str(checkpoint),
        index=0,
        sft={"messages": [{"role": "user", "content": "private"}]},
        dpo=None,
    )
    text = checkpoint.read_text().replace("private", "changed")
    checkpoint.write_text(text, encoding="utf-8")
    try:
        load_checkpoint(str(checkpoint), digest="d", total=1)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("parseable checkpoint corruption must fail closed")


def test_resume_discards_only_an_uncommitted_final_fragment(tmp_path, monkeypatch):
    from soup_cli.utils.best_of_n_checkpoint import (
        append_checkpoint,
        initialise_checkpoint,
        load_checkpoint,
    )

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    initialise_checkpoint(str(checkpoint), digest="d", total=2)
    append_checkpoint(
        str(checkpoint), index=0, sft={"messages": []}, dpo=None
    )
    with checkpoint.open("ab") as handle:
        handle.write(b'{"index":1,"sft":')

    entries = load_checkpoint(str(checkpoint), digest="d", total=2)

    assert entries == [({"messages": []}, None)]
    assert checkpoint.read_bytes().endswith(b"\n")
    assert b'"index":1' not in checkpoint.read_bytes()


def test_local_sampling_matches_uninterrupted_run_across_prompt_indices(
    tmp_path, monkeypatch
):
    import torch

    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompts = ["first", "second", "third"]
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    for run_dir in (uninterrupted_dir, resumed_dir):
        run_dir.mkdir()
        (run_dir / "prompts.jsonl").write_text(
            "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "soup_cli.commands.data._load_bon_model", lambda *_args: (object(), object())
    )
    monkeypatch.setattr(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        lambda _model: False,
    )

    phase = {"name": "uninterrupted"}
    sampled: dict[str, dict[str, list[list[str]]]] = {
        "uninterrupted": {prompt: [] for prompt in prompts},
        "resumed": {prompt: [] for prompt in prompts},
    }

    def sample_candidates(_model, _tokenizer, prompt, *, n, **_kwargs):
        candidates = [f"{torch.rand(1).item():.12f}" for _ in range(n)]
        sampled[phase["name"]][prompt].append(candidates)
        return candidates

    fail_second = {"value": False}

    class Judge:
        def evaluate(self, prompt: str, response: str):
            if prompt == "second" and fail_second["value"]:
                fail_second["value"] = False
                raise RuntimeError("simulated interruption")
            return SimpleNamespace(weighted_score=float(response))

    monkeypatch.setattr(
        "soup_cli.utils.best_of_n.sample_candidates", sample_candidates
    )
    monkeypatch.setattr(
        "soup_cli.eval.judge.JudgeEvaluator", lambda **_kwargs: Judge()
    )

    uninterrupted_args = _local_args(
        uninterrupted_dir,
        "--emit-pairs",
        str(uninterrupted_dir / "dpo.jsonl"),
    )
    uninterrupted = CliRunner().invoke(app, uninterrupted_args)
    assert uninterrupted.exit_code == 0, (
        uninterrupted.output,
        repr(uninterrupted.exception),
    )

    phase["name"] = "resumed"
    fail_second["value"] = True
    resumed_args = _local_args(
        resumed_dir,
        "--emit-pairs",
        str(resumed_dir / "dpo.jsonl"),
    )
    first = CliRunner().invoke(app, resumed_args)
    assert first.exit_code == 1, (first.output, repr(first.exception))

    resumed = CliRunner().invoke(app, [*resumed_args, "--resume"])

    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    uninterrupted_final = {
        prompt: calls[-1] for prompt, calls in sampled["uninterrupted"].items()
    }
    resumed_final = {
        prompt: calls[-1] for prompt, calls in sampled["resumed"].items()
    }
    assert resumed_final == uninterrupted_final
    assert len({tuple(candidates) for candidates in uninterrupted_final.values()}) == 3
    assert [len(sampled["resumed"][prompt]) for prompt in prompts] == [1, 2, 1]

    for artifact in ("sft.jsonl", "dpo.jsonl", "sft.jsonl.manifest.json"):
        assert (resumed_dir / artifact).read_bytes() == (
            uninterrupted_dir / artifact
        ).read_bytes()
