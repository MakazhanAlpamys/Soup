"""v0.71.41 — `soup reward stress` adversarial verifier probe + telemetry doc-fix.

Turns the v0.71.26 reward-hacking expertise on a reward VERIFIER itself: does it
pay out for degenerate completions (empty / length-padded / repetition /
sentinel-spam)? Pure, offline, CPU-only.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils import reward_stress as rst

runner = CliRunner()


# ---------------------------------------------------------------------------
# Task 1 — attack generation
# ---------------------------------------------------------------------------
class TestGenerateAttacks:
    def test_all_kinds_present(self):
        attacks = rst.generate_attacks()
        kinds = {k for k, _ in attacks}
        assert kinds == set(rst.ATTACKS)

    def test_deterministic(self):
        assert rst.generate_attacks() == rst.generate_attacks()

    def test_empty_is_empty_string(self):
        text = dict(rst.generate_attacks())["empty"]
        assert text == ""

    def test_length_attack_exceeds_saturation(self):
        text = dict(rst.generate_attacks())["length"]
        assert len(text.split()) > 32  # _SHAPING_LENGTH_SAT

    def test_sentinel_attack_contains_custom_sentinel(self):
        text = dict(rst.generate_attacks(sentinel="ZZZTOKEN"))["sentinel"]
        assert "ZZZTOKEN" in text
        assert "GOLD" not in text

    def test_default_sentinel_is_gold(self):
        text = dict(rst.generate_attacks())["sentinel"]
        assert "GOLD" in text

    def test_repetition_attack_repeats(self):
        text = dict(rst.generate_attacks())["repetition"]
        words = text.split()
        assert len(words) > 5 and len(set(words)) < len(words)

    def test_kinds_subset_respected(self):
        attacks = rst.generate_attacks(kinds=("empty", "sentinel"))
        assert {k for k, _ in attacks} == {"empty", "sentinel"}

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            rst.generate_attacks(kinds=("bogus",))

    def test_bare_str_kinds_rejected(self):
        # A bare string is a Sequence[char] — must fail with a type error, not
        # silently probe per-letter.
        with pytest.raises(TypeError, match="sequence"):
            rst.generate_attacks(kinds="sentinel")


# ---------------------------------------------------------------------------
# Task 2 — run_stress scoring + verdict
# ---------------------------------------------------------------------------
class TestRunStress:
    def _numeric_verifier(self):
        import re

        num = re.compile(r"[+-]?\d+(?:\.\d+)?")

        def reward_fn(completions, **kwargs):
            answers = kwargs.get("answer") or []
            out = []
            for i, comp in enumerate(completions):
                text = comp[0]["content"] if isinstance(comp, list) else str(comp)
                gold = str(answers[i]) if i < len(answers) else ""
                got = num.findall(text)
                out.append(1.0 if got and got[-1] == gold else 0.0)
            return out

        return reward_fn

    def test_degenerate_always_one_is_gameable(self):
        def reward_fn(completions, **kwargs):
            return [1.0] * len(completions)

        rep = rst.run_stress(reward_fn, ["42", "7"])
        assert rep.gameable is True
        assert rep.gameability == 1.0
        assert rep.reference_accept == 1.0

    def test_strict_numeric_is_robust(self):
        rep = rst.run_stress(self._numeric_verifier(), ["42", "7", "100"])
        assert rep.gameable is False
        assert rep.gameability == 0.0
        assert rep.reference_accept == 1.0

    def test_per_attack_breakdown_distinguishes(self):
        # A length-based verifier accepts the two long attacks (length=60 words,
        # repetition=40 words) but rejects the two short ones — the per-attack
        # breakdown must separate exactly which junk slipped through.
        def reward_fn(completions, **kwargs):
            out = []
            for comp in completions:
                text = comp[0]["content"] if isinstance(comp, list) else str(comp)
                out.append(1.0 if len(text.split()) >= 32 else 0.0)
            return out

        rep = rst.run_stress(reward_fn, ["42"])
        per = {a.kind: a.accept_rate for a in rep.attacks}
        assert per["length"] == 1.0
        assert per["repetition"] == 1.0
        assert per["empty"] == 0.0
        assert per["sentinel"] == 0.0  # 20 words < 32 -> rejected
        assert rep.gameable is True

    def test_no_gold_fallback(self):
        def reward_fn(completions, **kwargs):
            return [0.0] * len(completions)

        rep = rst.run_stress(reward_fn, [])
        assert rep.reference_accept is None
        assert rep.gameable is False

    def test_max_gameable_boundary_inclusive(self):
        # Accepts exactly the 'sentinel' attack -> accept-rate = 1/4 across 4 kinds.
        def reward_fn(completions, **kwargs):
            out = []
            for comp in completions:
                text = comp[0]["content"] if isinstance(comp, list) else str(comp)
                out.append(1.0 if "GOLD" in text else 0.0)
            return out

        rep_at = rst.run_stress(reward_fn, ["x"], max_gameable=0.25)
        assert rep_at.gameability == 0.25
        assert rep_at.gameable is False  # 0.25 > 0.25 is False -> inclusive
        rep_below = rst.run_stress(reward_fn, ["x"], max_gameable=0.2)
        assert rep_below.gameable is True

    def test_threshold_applied(self):
        # Verifier returns 0.4 everywhere: accepted at threshold 0.3, rejected at 0.5.
        def reward_fn(completions, **kwargs):
            return [0.4] * len(completions)

        assert rst.run_stress(reward_fn, ["x"], threshold=0.3).gameable is True
        assert rst.run_stress(reward_fn, ["x"], threshold=0.5).gameable is False
        # Exactly on the boundary: 0.4 >= 0.4 accepts (proves >= is inclusive,
        # mutation-kills a `>` implementation).
        assert rst.run_stress(reward_fn, ["x"], threshold=0.4).gameable is True

    def test_golds_capped(self):
        seen = {"n": 0}

        def reward_fn(completions, **kwargs):
            seen["n"] = max(seen["n"], len(completions))
            return [0.0] * len(completions)

        rst.run_stress(reward_fn, [str(i) for i in range(500)])
        assert seen["n"] == rst._MAX_STRESS_GOLDS == 200  # pins the exact cap value

    def test_per_attack_counts(self):
        rep = rst.run_stress(self._numeric_verifier(), ["42", "7"])
        assert {a.kind for a in rep.attacks} == set(rst.ATTACKS)
        for a in rep.attacks:
            assert a.n == 2 and a.accepted == 0

    def test_short_return_raises_not_false_robust(self):
        # A gold-requiring builtin scored with no answer returns [] (its
        # zip short-circuits) — must be a hard error, never a silent 0/1=robust.
        def gold_requiring(completions, **kwargs):
            answers = kwargs.get("answer", [])
            return [1.0 for _c, _a in zip(completions, answers)]

        with pytest.raises(ValueError, match="--references"):
            rst.run_stress(gold_requiring, [])

    def test_non_finite_score_raises(self):
        def nan_fn(completions, **kwargs):
            return [float("nan")] * len(completions)

        with pytest.raises(ValueError, match="non-finite"):
            rst.run_stress(nan_fn, ["42"])

    def test_real_accuracy_builtin_gold_path_robust(self):
        from soup_cli.trainer.rewards import load_reward_fn

        rep = rst.run_stress(load_reward_fn("accuracy"), ["42", "7"])
        # accuracy compares the completion tail against the gold; junk never matches,
        # and a gold scored as its own completion is a perfect match.
        assert rep.gameable is False
        assert rep.reference_accept == 1.0


# ---------------------------------------------------------------------------
# Task 3 — CLI
# ---------------------------------------------------------------------------
_ROBUST_VERIFIER = '''
import re
_NUM = re.compile(r"[+-]?\\d+(?:\\.\\d+)?")
def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer") or []
    out = []
    for i, c in enumerate(completions):
        text = c[0]["content"] if isinstance(c, list) else str(c)
        gold = str(answers[i]) if i < len(answers) else ""
        got = _NUM.findall(text)
        out.append(1.0 if got and got[-1] == gold else 0.0)
    return out
'''

_DEGENERATE_VERIFIER = '''
def reward_fn(completions, **kwargs):
    return [1.0] * len(completions)
'''


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _refs(tmp_path):
    p = tmp_path / "refs.jsonl"
    p.write_text('{"answer": "42"}\n{"answer": "7"}\n', encoding="utf-8")
    return p


class TestStressCli:
    def test_help(self):
        r = runner.invoke(soup_app, ["reward", "stress", "--help"])
        assert r.exit_code == 0, (r.output, repr(r.exception))

    def test_robust_verifier_exit_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", v.name, "--references", _refs(tmp_path).name],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "robust" in r.output.lower() or "not gameable" in r.output.lower()

    def test_degenerate_verifier_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "d.py", _DEGENERATE_VERIFIER)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", v.name, "--references", _refs(tmp_path).name],
        )
        assert r.exit_code == 2, (r.output, repr(r.exception))
        # Discriminate the verdict text — "gameable" also occurs in "not gameable".
        assert "GAMEABLE" in r.output
        assert "robust" not in r.output.lower()

    def test_bad_attacks_exit_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(soup_app, ["reward", "stress", v.name, "--attacks", "bogus"])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "bogus" in r.output

    def test_bad_threshold_exit_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(soup_app, ["reward", "stress", v.name, "--threshold", "9"])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "threshold" in r.output.lower()

    def test_bad_max_gameable_exit_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(soup_app, ["reward", "stress", v.name, "--max-gameable", "2"])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "max-gameable" in r.output.lower()

    def test_empty_attacks_list_exit_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(soup_app, ["reward", "stress", v.name, "--attacks", ","])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "at least one" in r.output.lower()

    def test_verifiable_target_with_references(self, tmp_path, monkeypatch):
        # The 'verifiable' builtin path + --verifiable-domain must load and probe.
        monkeypatch.chdir(tmp_path)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", "verifiable", "--verifiable-domain", "math",
             "--references", _refs(tmp_path).name],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "robust" in r.output.lower()

    def test_report_path_validated_before_code_runs(self, tmp_path, monkeypatch):
        # A target that writes a marker on import + a bad (outside-cwd) report path:
        # the report-path check must fire FIRST, so the marker is never written.
        monkeypatch.chdir(tmp_path)
        marker = tmp_path / "IMPORTED"
        target = _write(
            tmp_path, "marks.py",
            f"open(r{str(marker)!r}, 'w').close()\n"
            "def reward_fn(completions, **kwargs):\n    return [0.0]*len(completions)\n",
        )
        r = runner.invoke(
            soup_app,
            ["reward", "stress", target.name, "--output-report", "../escape.json"],
        )
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert not marker.exists(), "target code ran before the report path was rejected"

    def test_target_raises_at_import_exit_1(self, tmp_path, monkeypatch):
        # load_reward_fn exec's the target's module code; a raise-at-import must
        # be caught and mapped to exit 1 with a clear message (not leak the
        # traceback / rely on the top-level cli.py safety net).
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "boom.py", 'raise RuntimeError("boom-at-import")\n')
        r = runner.invoke(soup_app, ["reward", "stress", v.name])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "could not load reward target" in r.output

    def test_oversized_sentinel_exit_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", v.name, "--sentinel", "z" * 300],
        )
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "sentinel" in r.output.lower()

    def test_builtin_no_references_exit_1(self, tmp_path, monkeypatch):
        # `soup reward stress accuracy` with no --references cannot probe a
        # gold-requiring builtin — must exit 1 with a helpful message, NOT a
        # false "robust" exit 0.
        monkeypatch.chdir(tmp_path)
        r = runner.invoke(soup_app, ["reward", "stress", "accuracy"])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "--references" in r.output

    def test_duplicate_attacks_deduped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "v.py", _ROBUST_VERIFIER)
        r = runner.invoke(
            soup_app,
            ["reward", "stress", v.name, "--references", _refs(tmp_path).name,
             "--attacks", "empty,empty,sentinel", "--output-report", "rep.json"],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        data = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
        kinds = [a["kind"] for a in data["attacks"]]
        assert kinds == ["empty", "sentinel"]

    def test_output_report_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v = _write(tmp_path, "d.py", _DEGENERATE_VERIFIER)
        r = runner.invoke(
            soup_app,
            [
                "reward", "stress", v.name,
                "--references", _refs(tmp_path).name,
                "--output-report", "rep.json",
            ],
        )
        assert r.exit_code == 2, (r.output, repr(r.exception))
        data = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
        assert data["gameable"] is True
        assert "attacks" in data


# ---------------------------------------------------------------------------
# Task 4 — hardening + registration
# ---------------------------------------------------------------------------
class TestHardening:
    def test_no_top_level_torch(self):
        src = Path(rst.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:  # module-level only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = mod + " " + " ".join(a.name for a in getattr(node, "names", []))
                assert "torch" not in names and "transformers" not in names, names

    def test_stress_registered(self):
        r = runner.invoke(soup_app, ["reward", "--help"])
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "stress" in r.output

    def test_target_outside_cwd_rejected(self, tmp_path, monkeypatch):
        # Use a REAL file outside cwd so the ONLY possible failure is the
        # containment check (a nonexistent path would exit 1 via "not found" even
        # if containment were deleted — a vacuous security test).
        sub = tmp_path / "work"
        sub.mkdir()
        (tmp_path / "evil.py").write_text(_ROBUST_VERIFIER, encoding="utf-8")
        monkeypatch.chdir(sub)
        r = runner.invoke(soup_app, ["reward", "stress", "../evil.py"])
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "under cwd" in r.output.lower()

    def test_symlinked_target_rejected(self, tmp_path, monkeypatch):
        # A symlinked .py target must be refused by enforce_under_cwd_and_no_symlink
        # (a symlink could point outside cwd). POSIX-only — Windows symlink creation
        # needs privilege.
        monkeypatch.chdir(tmp_path)
        real = _write(tmp_path, "real.py", _ROBUST_VERIFIER)
        link = tmp_path / "link.py"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform")
        r = runner.invoke(soup_app, ["reward", "stress", link.name])
        assert r.exit_code == 1, (r.output, repr(r.exception))
