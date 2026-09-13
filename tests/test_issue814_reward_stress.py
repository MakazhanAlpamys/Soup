"""Tests for Issue #814: structure-preserving attack families in soup reward stress.

Verifies:
1. `soup reward stress format` reports gameable and exits 2.
2. `soup reward stress accuracy --references g.jsonl` flags answer_spray at default threshold.
3. `soup reward stress verifiable --verifiable-domain math --references g.jsonl`
   reports robust (exit 0).
4. Each attack family contributes more than one distinct string and n reflects distinct attempts.
5. `--attacks` accepts new family names and rejects unknown ones with valid options listed.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import soup_cli.utils.reward_stress as rst
from soup_cli.cli import app as soup_app

runner = CliRunner()


def _write_jsonl(path: Path, filename: str, rows: list[dict]) -> Path:
    target = path / filename
    with target.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return target


class TestIssue814AttackVariants:
    def test_all_attack_families_registered(self):
        assert set(rst.CLASSIC_ATTACKS) == {"empty", "length", "repetition", "sentinel"}
        assert set(rst.STRUCTURE_ATTACKS) == {
            "wrapped_junk",
            "answer_spray",
            "structure_without_content",
        }
        assert rst.ATTACKS == rst.CLASSIC_ATTACKS + rst.STRUCTURE_ATTACKS

    def test_each_family_has_multiple_distinct_variants(self):
        for kind in rst.ATTACKS:
            variants = rst.generate_attack_variants(kind, gold="42")
            assert len(variants) > 1, f"{kind} must have > 1 variant"
            assert len(set(variants)) == len(variants), f"{kind} variants must be distinct"

    def test_answer_spray_embeds_gold(self):
        gold = "998877"
        variants = rst.generate_attack_variants("answer_spray", gold=gold)
        for v in variants:
            assert gold in v


class TestIssue814ScoringAndCli:
    def test_format_reward_gameable_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rep_file = tmp_path / "rep.json"
        r = runner.invoke(
            soup_app,
            ["reward", "stress", "format", "--output-report", str(rep_file)],
        )
        assert r.exit_code == 2, (r.output, repr(r.exception))
        assert "gameable" in r.output.lower()

        data = json.loads(rep_file.read_text(encoding="utf-8"))
        assert data["gameable"] is True
        per_attack = {a["kind"]: a for a in data["attacks"]}
        assert per_attack["wrapped_junk"]["accepted"] > 0
        assert per_attack["wrapped_junk"]["accept_rate"] > 0.0
        # n reflects distinct attempts (> 1 per family)
        for a in data["attacks"]:
            assert a["n"] > 1

    def test_accuracy_reward_flags_answer_spray_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        refs = _write_jsonl(tmp_path, "refs.jsonl", [{"answer": "42"}, {"answer": "17"}])
        rep_file = tmp_path / "rep.json"
        r = runner.invoke(
            soup_app,
            [
                "reward", "stress", "accuracy",
                "--references", str(refs),
                "--output-report", str(rep_file),
            ],
        )
        assert r.exit_code == 2, (r.output, repr(r.exception))
        assert "gameable" in r.output.lower()

        data = json.loads(rep_file.read_text(encoding="utf-8"))
        assert data["gameable"] is True
        per_attack = {a["kind"]: a for a in data["attacks"]}
        # answer_spray achieves 100% acceptance on accuracy (0.5 score >= 0.5 threshold)
        assert per_attack["answer_spray"]["accept_rate"] == 1.0
        assert per_attack["answer_spray"]["accepted"] == per_attack["answer_spray"]["n"]

    def test_verifiable_math_remains_robust_exit_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        refs = _write_jsonl(tmp_path, "refs.jsonl", [{"answer": "42"}, {"answer": "17"}])
        rep_file = tmp_path / "rep.json"
        r = runner.invoke(
            soup_app,
            [
                "reward", "stress", "verifiable",
                "--verifiable-domain", "math",
                "--references", str(refs),
                "--output-report", str(rep_file),
            ],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "robust" in r.output.lower()

        data = json.loads(rep_file.read_text(encoding="utf-8"))
        assert data["gameable"] is False
        assert data["gameability"] == 0.0
        assert data["reference_accept"] == 1.0
        for a in data["attacks"]:
            assert a["accepted"] == 0
            assert a["accept_rate"] == 0.0

    def test_attacks_flag_accepts_new_family_names(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        refs = _write_jsonl(tmp_path, "refs.jsonl", [{"answer": "42"}])
        rep_file = tmp_path / "rep.json"
        r = runner.invoke(
            soup_app,
            [
                "reward", "stress", "verifiable",
                "--verifiable-domain", "math",
                "--references", str(refs),
                "--attacks", "wrapped_junk,answer_spray,structure_without_content",
                "--output-report", str(rep_file),
            ],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        data = json.loads(rep_file.read_text(encoding="utf-8"))
        kinds = [a["kind"] for a in data["attacks"]]
        assert kinds == ["wrapped_junk", "answer_spray", "structure_without_content"]

    def test_attacks_flag_rejects_unknown_name_with_options(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", "format", "--attacks", "unknown_family"],
        )
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "unknown attack kind" in r.output.lower()
        for expected in rst.ATTACKS:
            assert expected in r.output
