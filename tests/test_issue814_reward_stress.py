"""Tests for Issue #814: structure-preserving attack families in soup reward stress.

Verifies:
1. `soup reward stress format` reports gameable and exits 2.
2. `soup reward stress <verifier.py> --references g.jsonl` flags answer_spray at default threshold
   for a verifier that pays partial credit when the gold appears anywhere (the built-in
   `accuracy` paid that credit until #1226 removed it).
3. `soup reward stress verifiable --verifiable-domain math --references g.jsonl`
   reports robust (exit 0).
4. Each attack family contributes more than one distinct string and n reflects distinct attempts.
5. `--attacks` accepts new family names and rejects unknown ones with valid options listed.
6. A gold equal to a distractor is not a false positive against math verifier across numbers/floats.
7. `answer_spray` float distractors do not collapse into target or scientific notation.
8. `run_stress` eagerly validates attack kinds before `reward_fn` and uses per-family verdict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.reward_stress as rst
from soup_cli.cli import app as soup_app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


_SUBSTRING_CREDIT_VERIFIER = '''
def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer") or []
    out = []
    for completion, gold in zip(completions, answers):
        text = completion[-1]["content"] if completion else ""
        out.append(0.5 if str(gold).strip().lower() in text.lower() else 0.0)
    return out
'''


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
        }
        assert rst.ATTACKS == rst.CLASSIC_ATTACKS + rst.STRUCTURE_ATTACKS

    def test_each_family_has_multiple_distinct_variants(self):
        for kind in rst.ATTACKS:
            variants = rst.generate_attack_variants(kind, gold="42")
            assert len(variants) > 1, f"{kind} must have > 1 variant"
            assert len(set(variants)) == len(variants), f"{kind} variants must be distinct"

    def test_generate_attacks_multi_variant(self):
        attacks = rst.generate_attacks()
        for kind in rst.ATTACKS:
            kind_texts = [text for k, text in attacks if k == kind]
            assert len(kind_texts) > 1, (
                f"generate_attacks must yield multiple variants for {kind}"
            )
            assert len(set(kind_texts)) == len(kind_texts), (
                f"generate_attacks variants must be distinct for {kind}"
            )

    def test_run_stress_rejects_str_attacks(self):
        with pytest.raises(
            TypeError, match="attacks must be a sequence of attack-kind strings, not a str"
        ):
            rst.run_stress(
                lambda completions, **kw: [0.0] * len(completions),
                golds=[],
                attacks="wrapped_junk",
            )

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
        clean_out = _plain(r.output)
        assert "GAMEABLE" in clean_out
        assert "robust (not gameable)" not in clean_out.lower()

        data = json.loads(rep_file.read_text(encoding="utf-8"))
        assert data["gameable"] is True
        per_attack = {a["kind"]: a for a in data["attacks"]}
        assert per_attack["wrapped_junk"]["accepted"] > 0
        assert per_attack["wrapped_junk"]["accept_rate"] > 0.0
        # n reflects distinct attempts (> 1 per family)
        for a in data["attacks"]:
            assert a["n"] > 1

    def test_substring_crediting_verifier_flags_answer_spray_exit_2(
        self, tmp_path, monkeypatch
    ):
        # The built-in `accuracy` reward paid this 0.5 substring credit until #1226 removed it;
        # the verifier below reproduces it so the stress CLI's answer_spray detection stays
        # pinned. (`accuracy` itself is now robust: tests/test_issue1226_reward_gold_parsing.py.)
        monkeypatch.chdir(tmp_path)
        refs = _write_jsonl(tmp_path, "refs.jsonl", [{"answer": "42"}, {"answer": "17"}])
        verifier = tmp_path / "substring_reward.py"
        verifier.write_text(_SUBSTRING_CREDIT_VERIFIER, encoding="utf-8")
        rep_file = tmp_path / "rep.json"
        r = runner.invoke(
            soup_app,
            [
                "reward", "stress", str(verifier),
                "--references", str(refs),
                "--output-report", str(rep_file),
            ],
        )
        assert r.exit_code == 2, (r.output, repr(r.exception))
        clean_out = _plain(r.output)
        assert "GAMEABLE" in clean_out
        assert "robust (not gameable)" not in clean_out.lower()

        data = json.loads(rep_file.read_text(encoding="utf-8"))
        assert data["gameable"] is True
        per_attack = {a["kind"]: a for a in data["attacks"]}
        # answer_spray achieves 100% acceptance (0.5 score >= 0.5 threshold)
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
        clean_out = _plain(r.output)
        assert "robust (not gameable)" in clean_out.lower()
        assert "GAMEABLE" not in clean_out

        data = json.loads(rep_file.read_text(encoding="utf-8"))
        assert data["gameable"] is False
        assert data["gameability"] == 0.0
        assert data["reference_accept"] == 1.0
        for a in data["attacks"]:
            assert a["accepted"] == 0
            assert a["accept_rate"] == 0.0

    @pytest.mark.parametrize("gold", ["333", "999", "100000.0", "42.5", "-17"])
    def test_a_gold_equal_to_a_distractor_is_not_a_false_positive(self, gold):
        from soup_cli.trainer.rewards import load_reward_fn

        math_verifier = load_reward_fn("verifiable", verifiable_domain="math")
        rep = rst.run_stress(math_verifier, [gold])
        assert rep.gameable is False
        assert rep.gameability == 0.0

    def test_answer_spray_float_distractors_do_not_collapse(self):
        for gold in ["100000.0", "1e5"]:
            variants = rst.generate_attack_variants("answer_spray", gold=gold)
            for v in variants:
                assert gold in v
                # Distractors must not collapse into scientific notation identical to gold
                assert "1e+05" not in v

    def test_run_stress_validates_attacks_eagerly_before_reward_fn(self):
        called = False

        def bad_reward(completions, **kw):
            nonlocal called
            called = True
            return [0.0] * len(completions)

        with pytest.raises(ValueError, match="unknown attack kind"):
            rst.run_stress(bad_reward, ["42"], attacks=["wrapped_junk", "bad_typo"])
        assert called is False, "run_stress must validate attack kinds before reward_fn"

    def test_per_family_verdict_prevents_dilution_and_flips(self):
        # A family with 100% acceptance must flag gameable even if combined with
        # robust families (0% acceptance). Under pooled aggregate denominator this would
        # average to 50%, whereas per-family max is 1.0 (100%).
        def selective_reward(completions, **kw):
            out = []
            for c in completions:
                text = c[0]["content"] if isinstance(c, list) else str(c)
                out.append(1.0 if len(text.split()) >= 30 else 0.0)
            return out

        rep = rst.run_stress(
            selective_reward, ["x"], attacks=["length", "empty"], max_gameable=0.60
        )
        assert rep.gameable is True
        assert rep.gameability == 1.0

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
                "--attacks", "wrapped_junk,answer_spray",
                "--output-report", str(rep_file),
            ],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        data = json.loads(rep_file.read_text(encoding="utf-8"))
        kinds = [a["kind"] for a in data["attacks"]]
        assert kinds == ["wrapped_junk", "answer_spray"]

    def test_attacks_flag_rejects_unknown_name_with_options(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", "format", "--attacks", "unknown_family"],
        )
        assert r.exit_code == 1, (r.output, repr(r.exception))
        clean_out = _plain(r.output)
        assert "unknown attack kind" in clean_out.lower()
        for expected in rst.ATTACKS:
            assert expected in clean_out
