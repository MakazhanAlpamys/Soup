"""Issue #937 — reward stress must pass JSON-schema references as ``schema``."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.trainer.rewards import load_reward_fn
from soup_cli.utils import reward_stress
from tests.conftest import strip_ansi

runner = CliRunner()


def _schema(*, required: bool) -> str:
    payload = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"] if required else [],
    }
    return json.dumps(payload)


def test_real_json_schema_reward_receives_schema_references() -> None:
    reward_fn = load_reward_fn("verifiable", verifiable_domain="json_schema")

    report = reward_stress.run_stress(reward_fn, [_schema(required=True)])

    # A schema configures validation; it is not itself a valid completion, so
    # self-acceptance is intentionally non-diagnostic for this domain.
    assert report.reference_accept == 0.0
    assert report.gameable is False
    assert {attack.kind for attack in report.attacks} == set(reward_stress.ATTACKS)
    assert all(attack.n == 1 and attack.accepted == 0 for attack in report.attacks)


def test_schema_reference_routing_can_distinguish_gameable_and_strict() -> None:
    def metadata_schema_verifier(completions, **kwargs):
        schemas = kwargs.get("schema", [])
        return [
            1.0 if json.loads(schema)["allows_junk"] else 0.0
            for _completion, schema in zip(completions, schemas)
        ]

    gameable = reward_stress.run_stress(
        metadata_schema_verifier,
        [json.dumps({"allows_junk": True})],
        reference_key="schema",
        attacks=("empty",),
    )
    strict = reward_stress.run_stress(
        metadata_schema_verifier,
        [json.dumps({"allows_junk": False})],
        reference_key="schema",
        attacks=("empty",),
    )

    assert gameable.attacks[0].accepted == 1
    assert gameable.gameable is True
    assert strict.attacks[0].accepted == 0
    assert strict.gameable is False


def test_short_return_with_references_does_not_blame_missing_references() -> None:
    def broken_schema_verifier(completions, **kwargs):
        assert kwargs["schema"]
        return []

    with pytest.raises(ValueError) as exc_info:
        reward_stress.run_stress(
            broken_schema_verifier,
            [_schema(required=True)],
            reference_key="schema",
            attacks=("empty",),
        )

    message = str(exc_info.value)
    assert "despite receiving 1 reference(s) via 'schema'" in message
    assert "--references" not in message


def test_no_references_still_names_references_option() -> None:
    def gold_requiring(completions, **kwargs):
        return []

    with pytest.raises(ValueError, match="--references"):
        reward_stress.run_stress(gold_requiring, [], attacks=("empty",))


def test_json_schema_cli_runs_with_schema_field(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    references = tmp_path / "schemas.jsonl"
    references.write_text(
        json.dumps({"schema": json.loads(_schema(required=True))}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        soup_app,
        [
            "reward",
            "stress",
            "verifiable",
            "--verifiable-domain",
            "json_schema",
            "--references",
            references.name,
            "--field",
            "schema",
            "--attacks",
            "empty",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "robust" in strip_ansi(result.output).lower()
