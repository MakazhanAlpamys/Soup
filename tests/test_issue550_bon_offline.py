"""Regression tests for #550: artifact-mediated offline Best-of-N."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner


def _export_candidates(tmp_path, monkeypatch, prompts=("question one", "question two")):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    calls = []

    def factory(*_args, **_kwargs):
        def generate(prompt):
            calls.append(prompt)
            return f"candidate-{len(calls)}"

        return generate

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", factory)
    monkeypatch.setattr(
        "soup_cli.commands.data._load_bon_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider export must not load a local model")
        ),
    )
    artifact = tmp_path / "candidates.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--provider",
            "ollama",
            "--model",
            "sampler-model",
            "--base-url",
            "http://localhost:11434/private-route",
            "--prompts",
            str(prompt_path),
            "--n",
            "2",
            "--export-candidates",
            str(artifact),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return artifact, calls


def _artifact_groups(artifact):
    return [json.loads(line) for line in artifact.read_text().splitlines()][1:]


def _write_judgments(path, groups):
    rows = [
        {
            "prompt_id": group["prompt_id"],
            "group_digest": group["group_digest"],
            "winner_idx": 1,
            "scores": [0.25, 0.75],
            "verifier": {"name": "Codex", "version": "offline-v1"},
        }
        for group in groups
    ]
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def test_candidate_export_needs_no_judge_and_preserves_order_and_digests(
    tmp_path, monkeypatch
):
    artifact, calls = _export_candidates(tmp_path, monkeypatch)
    assert calls == ["question one", "question one", "question two", "question two"]
    records = [json.loads(line) for line in artifact.read_text().splitlines()]
    header = records[0]["_best_of_n_candidates"]
    assert header["schema"] == "soup.best_of_n.candidates.v1"
    assert header["sampler"] == {
        "kind": "provider",
        "provider": "ollama",
        "model": "sampler-model",
        "n": 2,
        "temperature": 1.0,
        "max_new_tokens": 256,
    }
    assert [candidate["index"] for candidate in records[1]["candidates"]] == [0, 1]
    assert [candidate["text"] for candidate in records[1]["candidates"]] == [
        "candidate-1",
        "candidate-2",
    ]
    assert [group["source_line"] for group in records[1:]] == [1, 2]
    text = artifact.read_text()
    assert "base_url" not in text
    assert str(tmp_path) not in text


def test_sampler_temperature_integer_overflow_is_a_controlled_validation_error(
    tmp_path, monkeypatch
):
    from soup_cli.utils.best_of_n_artifact import load_candidate_artifact

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    records = [json.loads(line) for line in artifact.read_text().splitlines()]
    records[0]["_best_of_n_candidates"]["sampler"]["temperature"] = 10**4000
    artifact.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate sampler temperature is invalid"):
        load_candidate_artifact(str(artifact))


def test_offline_materialization_never_constructs_sampler_or_judge_and_is_stable(
    tmp_path, monkeypatch
):
    from soup_cli.commands.data import app

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    groups = _artifact_groups(artifact)
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, groups)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline materialization attempted model or network setup")

    monkeypatch.setattr("soup_cli.commands.data._load_bon_model", forbidden)
    monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", forbidden)
    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", forbidden)

    outputs = []
    for suffix in ("a", "b"):
        sft = tmp_path / f"sft-{suffix}.jsonl"
        dpo = tmp_path / f"dpo-{suffix}.jsonl"
        result = CliRunner().invoke(
            app,
            [
                "best-of-n",
                "--candidate-artifact",
                str(artifact),
                "--judgments",
                str(judgments),
                "--output",
                str(sft),
                "--emit-pairs",
                str(dpo),
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        outputs.append((sft.read_bytes(), dpo.read_bytes()))
    assert outputs[0] == outputs[1]

    sft_row = json.loads(outputs[0][0].splitlines()[0])
    dpo_row = json.loads(outputs[0][1].splitlines()[0])
    provenance = sft_row["_best_of_n"]
    assert provenance["mode"] == "offline"
    assert provenance["source_line"] == 1
    assert provenance["sampler"]["model"] == "sampler-model"
    assert provenance["verifier"] == {"name": "Codex", "version": "offline-v1"}
    assert len(provenance["candidate_artifact_sha256"]) == 64
    assert len(provenance["judgments_sha256"]) == 64
    assert dpo_row["_best_of_n"] == provenance
    assert str(tmp_path) not in outputs[0][0].decode()


def test_offline_manifest_commits_and_verifies_exact_output_set(tmp_path, monkeypatch):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import verify_offline_manifest

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    groups = _artifact_groups(artifact)
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, groups)
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(sft),
            "--emit-pairs",
            str(dpo),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    manifest_path = tmp_path / "sft.jsonl.manifest.json"
    manifest = verify_offline_manifest(
        str(manifest_path), sft_path=str(sft), dpo_path=str(dpo)
    )
    assert manifest["schema"] == "soup.best_of_n.offline_manifest.v1"
    assert manifest["dpo_requested"] is True
    assert manifest["sft"]["rows"] == 2
    assert manifest["dpo"]["rows"] == 2
    assert len(manifest["candidate_artifact_sha256"]) == 64
    assert len(manifest["judgments_sha256"]) == 64


def test_sft_only_manifest_records_that_dpo_was_not_requested(tmp_path, monkeypatch):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import verify_offline_manifest

    artifact, _calls = _export_candidates(tmp_path, monkeypatch, prompts=("question",))
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    sft = tmp_path / "sft.jsonl"
    manifest_path = tmp_path / "commit.json"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(sft),
            "--manifest",
            str(manifest_path),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    manifest = verify_offline_manifest(str(manifest_path), sft_path=str(sft))
    assert manifest["dpo_requested"] is False
    assert manifest["dpo"] is None


def test_later_sft_only_generation_removes_prior_manifest_bound_dpo(
    tmp_path, monkeypatch
):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import verify_offline_manifest

    artifact, _calls = _export_candidates(tmp_path, monkeypatch, prompts=("question",))
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    manifest_path = tmp_path / "commit.json"
    common = [
        "best-of-n",
        "--candidate-artifact",
        str(artifact),
        "--judgments",
        str(judgments),
        "--output",
        str(sft),
        "--manifest",
        str(manifest_path),
    ]
    first = CliRunner().invoke(app, [*common, "--emit-pairs", str(dpo)])
    assert first.exit_code == 0, (first.output, repr(first.exception))
    assert dpo.exists()

    second = CliRunner().invoke(app, common)

    assert second.exit_code == 0, (second.output, repr(second.exception))
    assert not dpo.exists()
    manifest = verify_offline_manifest(str(manifest_path), sft_path=str(sft))
    assert manifest["dpo_requested"] is False
    assert manifest["dpo"] is None


def test_failed_dpo_replacement_restores_previous_generation(tmp_path, monkeypatch):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import verify_offline_manifest

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    args = [
        "best-of-n",
        "--candidate-artifact",
        str(artifact),
        "--judgments",
        str(judgments),
        "--output",
        str(sft),
        "--emit-pairs",
        str(dpo),
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, (first.output, repr(first.exception))
    manifest_path = tmp_path / "sft.jsonl.manifest.json"
    assert manifest_path.exists()
    previous = (sft.read_bytes(), dpo.read_bytes(), manifest_path.read_bytes())

    real_replace = __import__("os").replace
    failed_once = False

    def fail_dpo(source, destination):
        nonlocal failed_once
        if not failed_once and destination == str(dpo):
            failed_once = True
            raise OSError("simulated DPO publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr("os.replace", fail_dpo)
    failed = CliRunner().invoke(app, args)
    assert failed.exit_code == 1
    assert (sft.read_bytes(), dpo.read_bytes(), manifest_path.read_bytes()) == previous
    verify_offline_manifest(
        str(manifest_path), sft_path=str(sft), dpo_path=str(dpo)
    )


def test_failed_sft_only_replacement_restores_manifest_bound_dpo(
    tmp_path, monkeypatch
):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import verify_offline_manifest

    artifact, _calls = _export_candidates(tmp_path, monkeypatch, prompts=("question",))
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    manifest_path = tmp_path / "commit.json"
    common = [
        "best-of-n",
        "--candidate-artifact",
        str(artifact),
        "--judgments",
        str(judgments),
        "--output",
        str(sft),
        "--manifest",
        str(manifest_path),
    ]
    first = CliRunner().invoke(app, [*common, "--emit-pairs", str(dpo)])
    assert first.exit_code == 0, (first.output, repr(first.exception))
    previous = (sft.read_bytes(), dpo.read_bytes(), manifest_path.read_bytes())
    changed = _write_judgments(judgments, _artifact_groups(artifact))
    changed[0]["winner_idx"] = 0
    changed[0]["scores"] = [0.9, 0.1]
    judgments.write_text(
        "".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8"
    )

    real_replace = __import__("os").replace
    failed_once = False

    def fail_sft(source, destination):
        nonlocal failed_once
        if (
            not failed_once
            and destination == str(sft)
            and ".soup.group." in str(source)
        ):
            failed_once = True
            raise OSError("simulated SFT publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr("os.replace", fail_sft)
    failed = CliRunner().invoke(app, common)

    assert failed.exit_code == 1
    assert (sft.read_bytes(), dpo.read_bytes(), manifest_path.read_bytes()) == previous
    verify_offline_manifest(
        str(manifest_path), sft_path=str(sft), dpo_path=str(dpo)
    )


def test_manifest_verifier_rejects_replaced_output(tmp_path, monkeypatch):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import verify_offline_manifest

    artifact, _calls = _export_candidates(tmp_path, monkeypatch, prompts=("question",))
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    sft = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(sft),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    sft.write_bytes(sft.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="SFT content does not match"):
        verify_offline_manifest(
            str(tmp_path / "sft.jsonl.manifest.json"), sft_path=str(sft)
        )


def test_offline_manifest_path_must_be_distinct(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    artifact, _calls = _export_candidates(tmp_path, monkeypatch, prompts=("question",))
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    sft = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(sft),
            "--manifest",
            str(sft),
        ],
    )
    assert result.exit_code == 2
    assert "must be distinct" in result.output
    assert not sft.exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "cover every candidate group"),
        ("duplicate", "missing or duplicate"),
        ("out-of-range", "out of range"),
        ("non-finite", "finite"),
        ("integer-overflow", "finite"),
        ("digest", "candidate digest mismatch"),
    ],
)
def test_invalid_offline_judgments_fail_before_publication(
    tmp_path, monkeypatch, mutation, match
):
    from soup_cli.commands.data import app

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    groups = _artifact_groups(artifact)
    judgment_path = tmp_path / "judgments.jsonl"
    rows = _write_judgments(judgment_path, groups)
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "out-of-range":
        rows[0]["winner_idx"] = 2
    elif mutation == "non-finite":
        rows[0]["scores"][0] = float("nan")
    elif mutation == "integer-overflow":
        rows[0]["scores"][0] = 10**4000
    elif mutation == "digest":
        rows[0]["group_digest"] = "0" * 64
    judgment_path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    output = tmp_path / "must-not-exist.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgment_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert match in result.output
    assert not output.exists()


@pytest.mark.parametrize("preexisting", [False, True])
def test_dpo_write_failure_rolls_back_the_complete_publication(
    tmp_path, monkeypatch, preexisting
):
    from soup_cli.commands.data import app

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    groups = _artifact_groups(artifact)
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, groups)
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    if preexisting:
        sft.write_bytes(b"previous-sft\n")
        dpo.write_bytes(b"previous-dpo\n")

    real_replace = __import__("os").replace
    failed = False

    def fail_first_dpo_commit(source, destination):
        nonlocal failed
        if not failed and destination == str(dpo):
            failed = True
            raise OSError("injected DPO publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr("os.replace", fail_first_dpo_commit)
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(sft),
            "--emit-pairs",
            str(dpo),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Failed to write output" in result.output
    if preexisting:
        assert sft.read_bytes() == b"previous-sft\n"
        assert dpo.read_bytes() == b"previous-dpo\n"
    else:
        assert not sft.exists()
        assert not dpo.exists()


def test_local_export_records_revision_and_seed_without_exposing_local_path(
    tmp_path, monkeypatch
):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "private-user-model"
    model_dir.mkdir()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"q"}\n', encoding="utf-8")
    load_calls = []
    monkeypatch.setattr(
        "soup_cli.commands.data._load_bon_model",
        lambda *args, **kwargs: (load_calls.append((args, kwargs)) or (object(), object())),
    )
    monkeypatch.setattr(
        "soup_cli.utils.best_of_n.sample_candidates",
        lambda *_args, **_kwargs: ["a", "b"],
    )
    artifact = tmp_path / "local-candidates.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--base",
            str(model_dir),
            "--revision",
            "abc123",
            "--seed",
            "42",
            "--prompts",
            str(prompts),
            "--n",
            "2",
            "--export-candidates",
            str(artifact),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert load_calls[0][1] == {"revision": "abc123"}
    sampler = json.loads(artifact.read_text().splitlines()[0])[
        "_best_of_n_candidates"
    ]["sampler"]
    assert sampler["model"] == "<local-model>"
    assert sampler["revision"] == "abc123"
    assert sampler["seed"] == 42
    assert str(tmp_path) not in artifact.read_text()


def test_online_workflow_still_requires_judge(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"q"}\n', encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["best-of-n", "--base", "model", "--prompts", str(prompts), "--plan-only"],
    )
    assert result.exit_code == 2
    assert "--judge is required" in result.output


def test_offline_mode_rejects_irrelevant_sampling_overrides(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(tmp_path / "candidates.jsonl"),
            "--judgments",
            str(tmp_path / "judgments.jsonl"),
            "--output",
            str(tmp_path / "sft.jsonl"),
            "--n",
            "9",
        ],
    )
    assert result.exit_code == 2
    assert "does not accept sampling" in result.output
    assert "Invalid offline artifact" not in result.output


@pytest.mark.parametrize("recovery_option", ["resume", "checkpoint"])
def test_offline_mode_rejects_online_recovery_options_before_publication(
    tmp_path, monkeypatch, recovery_option
):
    from soup_cli.commands.data import app

    artifact, _calls = _export_candidates(tmp_path, monkeypatch, prompts=("question",))
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, _artifact_groups(artifact))
    output = tmp_path / "sft.jsonl"
    manifest = tmp_path / "manifest.json"
    args = [
        "best-of-n",
        "--candidate-artifact",
        str(artifact),
        "--judgments",
        str(judgments),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    if recovery_option == "resume":
        args.append("--resume")
    else:
        args.extend(("--checkpoint", str(tmp_path / "checkpoint.jsonl")))

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert "does not support --resume or --checkpoint" in result.output
    assert not output.exists()
    assert not manifest.exists()


def test_two_phase_artifacts_use_exact_utf8_byte_writes(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.setattr(
        "soup_cli.utils.paths.atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Best-of-N JSONL must use exact UTF-8 byte writes")
        ),
    )
    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    groups = _artifact_groups(artifact)
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, groups)
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(tmp_path / "sft.jsonl"),
            "--emit-pairs",
            str(tmp_path / "dpo.jsonl"),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_candidate_text_tampering_is_detected_before_publication(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    artifact, _calls = _export_candidates(tmp_path, monkeypatch)
    groups = _artifact_groups(artifact)
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, groups)
    artifact.write_text(
        artifact.read_text().replace("candidate-1", "tampered-candidate"),
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "candidate digest mismatch" in result.output
    assert not output.exists()
