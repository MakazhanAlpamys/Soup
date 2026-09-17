"""#843: ``soup draft measure`` never said whether a pair pays, or at which k.

The acceptance band (STRONG >= 0.70) labelled Llama-3.1-8B <- Llama-3.2-1B STRONG at
0.813, while the measured assisted arm ran at 0.481x of plain. Whether a pair pays
depends on the draft/target latency ratio, which the command never measured.

The model here is the standard expected-tokens one, and it is a CEILING: i.i.d.
per-position acceptance, verification of k+1 tokens costed as one target step,
no framework overhead. The published figures below are the ones
``benchmarks/gate-h100-validation.md`` STEP 26 and #303's closing comment derived
from the same inputs, so pinning them pins the formula, not a fresh measurement.
"""

import json
import math
import re

import pytest

# benchmarks/gate-h100-validation.md STEP 26: target and draft tok/s on one H100.
_TARGET_TOK_S = 39.28
_DRAFT_TOK_S = 65.90
# #303's closing comment: the unrounded measured acceptance. 0.813 (the rounded
# figure) gives 0.9556x at k=5, which rounds to 0.956 rather than the published 0.955.
_ALPHA = 0.81293


def _c():
    from soup_cli.utils.draft import latency_ratio

    return latency_ratio(_TARGET_TOK_S, _DRAFT_TOK_S)


class TestThePublishedFigures:
    def test_latency_ratio(self):
        assert _c() == pytest.approx(0.596, abs=5e-4)

    def test_breakeven_at_the_shipped_k(self):
        from soup_cli.utils.draft import breakeven_acceptance

        assert breakeven_acceptance(5, _c()) == pytest.approx(0.832, abs=5e-4)

    def test_a_perfect_draft_caps_at_1_507x_at_k5(self):
        from soup_cli.utils.draft import modelled_speedup

        assert modelled_speedup(1.0, 5, _c()) == pytest.approx(1.507, abs=5e-4)

    def test_the_measured_alpha_is_below_breakeven_at_k5(self):
        from soup_cli.utils.draft import modelled_speedup

        assert modelled_speedup(_ALPHA, 5, _c()) == pytest.approx(0.955, abs=5e-4)

    def test_k1_and_k2(self):
        """#843 derived these from the ROUNDED 0.813 (at 0.81293, k=2 gives 1.1285,
        which rounds the other way). Each figure is pinned to its source's input."""
        from soup_cli.utils.draft import modelled_speedup

        assert modelled_speedup(0.813, 1, _c()) == pytest.approx(1.136, abs=5e-4)
        assert modelled_speedup(0.813, 2, _c()) == pytest.approx(1.129, abs=5e-4)

    def test_best_k_is_one_or_two(self):
        from soup_cli.utils.draft import modelled_best_k

        k, speedup = modelled_best_k(_ALPHA, _c())
        assert k in {1, 2}
        assert speedup == pytest.approx(1.136, abs=5e-4)

    @pytest.mark.parametrize("k, expected", [(1, 0.596), (2, 0.701), (3, 0.763), (4, 0.803)])
    def test_breakeven_rises_with_k(self, k, expected):
        """Derived in #843 from the same inputs; each is where S crosses 1."""
        from soup_cli.utils.draft import breakeven_acceptance, modelled_speedup

        a = breakeven_acceptance(k, _c())
        assert a == pytest.approx(expected, abs=5e-4)
        assert modelled_speedup(a, k, _c()) == pytest.approx(1.0, abs=1e-6)


class TestExpectedTokens:
    def test_a_perfect_draft_yields_k_plus_one(self):
        from soup_cli.utils.draft import expected_tokens_per_step

        for k in (1, 5, 64):
            assert expected_tokens_per_step(1.0, k) == k + 1

    def test_a_useless_draft_yields_the_target_token_only(self):
        from soup_cli.utils.draft import expected_tokens_per_step

        assert expected_tokens_per_step(0.0, 5) == 1.0

    def test_continuous_approaching_one(self):
        """The closed form divides by (1 - a); it must not jump at a -> 1."""
        from soup_cli.utils.draft import expected_tokens_per_step

        assert expected_tokens_per_step(1 - 1e-12, 5) == pytest.approx(6.0, abs=1e-6)

    def test_matches_the_series(self):
        from soup_cli.utils.draft import expected_tokens_per_step

        a, k = 0.6, 7
        assert expected_tokens_per_step(a, k) == pytest.approx(sum(a**i for i in range(k + 1)))


class TestNoRatePays:
    @pytest.mark.parametrize("c", [1.0, 1.5])
    def test_a_draft_no_faster_than_its_target_has_no_breakeven(self, c):
        """At c >= 1 even a = 1 gives (k+1)/(k*c+1) <= 1: the answer is "none",
        not a number the operator might chase."""
        from soup_cli.utils.draft import breakeven_acceptance

        for k in (1, 5, 64):
            assert breakeven_acceptance(k, c) is None

    def test_just_faster_than_the_target_still_has_one(self):
        from soup_cli.utils.draft import breakeven_acceptance

        a = breakeven_acceptance(1, 0.99)
        assert a is not None and 0.98 < a < 1.0


class TestLatencyRatio:
    @pytest.mark.parametrize(
        "plain, draft",
        [(0.0, 10.0), (10.0, 0.0), (-1.0, 10.0), (float("nan"), 10.0),
         (10.0, float("inf")), (None, 10.0), (10.0, None)],
    )
    def test_unusable_throughput_gives_no_ratio(self, plain, draft):
        from soup_cli.utils.draft import latency_ratio

        assert latency_ratio(plain, draft) is None


class TestInputValidation:
    @pytest.mark.parametrize("a", [-0.1, 1.1, float("nan"), True, "0.5"])
    def test_acceptance_must_be_a_probability(self, a):
        from soup_cli.utils.draft import modelled_speedup

        with pytest.raises((TypeError, ValueError)):
            modelled_speedup(a, 5, 0.5)

    @pytest.mark.parametrize("k", [0, 65, 2.0, True])
    def test_k_is_the_cli_range(self, k):
        from soup_cli.utils.draft import modelled_speedup

        with pytest.raises((TypeError, ValueError)):
            modelled_speedup(0.5, k, 0.5)

    @pytest.mark.parametrize("c", [0.0, -0.5, float("inf"), float("nan"), False])
    def test_latency_ratio_must_be_positive_and_finite(self, c):
        from soup_cli.utils.draft import breakeven_acceptance

        with pytest.raises((TypeError, ValueError)):
            breakeven_acceptance(5, c)


class TestTheModelIsTorchFree:
    def test_no_torch_import(self):
        """The four functions live in a module that imports torch lazily; calling
        them must not pull it in."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from soup_cli.utils.draft import (breakeven_acceptance, latency_ratio,"
            " modelled_best_k, modelled_speedup)\n"
            "c = latency_ratio(39.28, 65.9)\n"
            "breakeven_acceptance(5, c); modelled_best_k(0.8, c); modelled_speedup(0.8, 5, c)\n"
            "print('torch' in sys.modules)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False"


class TestTheBandNoLongerClaimsToPay:
    def test_docs_drop_the_paying_claim(self):
        """#843: 0.70 is an acceptance band, not where speculative decoding pays --
        the one pair measured at scale was a 0.481x slowdown at 0.813."""
        from pathlib import Path

        docs = (
            Path(__file__).resolve().parents[1] / "docs" / "serving-and-export.md"
        ).read_text(encoding="utf-8")
        assert not re.search(r"starts?\s+paying", docs), "0.70 does not mark payback"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _plain(text):
    return " ".join(_ANSI.sub("", text).split())


class _FakeTok:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % self.vocab_size for ch in text]


@pytest.fixture()
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(monkeypatch, tmp_path, throughput, extra=(), acceptance=(81, 100)):
    from typer.testing import CliRunner

    from soup_cli.commands import draft as draft_cmd
    from soup_cli.commands.draft import app

    (tmp_path / "p.jsonl").write_text(
        json.dumps({"prompt": "What is 2+2?"}) + "\n" + json.dumps({"prompt": "Hi"}) + "\n",
        encoding="utf-8",
    )
    target_model, draft_model = object(), object()
    monkeypatch.setattr(draft_cmd, "_vocab_size_of", lambda mid, trc=False: 49152)
    monkeypatch.setattr(
        draft_cmd,
        "_load_pair_member",
        lambda model_id, **kw: (
            target_model if "target" in model_id else draft_model,
            _FakeTok(49152),
            "cpu",
        ),
    )
    monkeypatch.setattr(draft_cmd, "measure_acceptance", lambda *a, **k: acceptance)

    def _throughput(model, tok, prompts, *, assistant_model=None, num_assistant_tokens=5,
                    **kw):
        role = "draft" if model is draft_model else (
            "assisted" if assistant_model is not None else "plain"
        )
        return throughput(role, num_assistant_tokens)

    monkeypatch.setattr(draft_cmd, "measure_throughput", _throughput)
    result = CliRunner().invoke(
        app,
        ["measure", "--target", "org/target", "--draft", "org/tiny",
         "--prompts", "p.jsonl", "-o", "report.json", *extra],
    )
    report = tmp_path / "report.json"
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
    return result, data


def _h100(role, k):
    return {"plain": _TARGET_TOK_S, "draft": _DRAFT_TOK_S, "assisted": 18.90}[role]


class TestMeasureReportsTheModel:
    def test_json_carries_the_modelled_fields(self, in_tmp_cwd, monkeypatch):
        result, data = _run(monkeypatch, in_tmp_cwd, _h100)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert data["tok_s_draft"] == _DRAFT_TOK_S
        assert data["draft_status"] == "complete"
        assert data["latency_ratio"] == pytest.approx(0.596, abs=5e-4)
        assert data["breakeven_acceptance"] == pytest.approx(0.832, abs=5e-4)
        assert data["modelled_best_k"] in {1, 2}
        assert data["modelled_speedup_best_k"] > 1.0
        # Existing fields keep their names and meaning.
        assert data["acceptance_rate"] == 0.81
        assert data["tok_s_plain"] == _TARGET_TOK_S
        assert data["tok_s_assisted"] == 18.90
        assert data["num_assistant_tokens"] == 5

    def test_breakeven_is_for_the_k_in_use(self, in_tmp_cwd, monkeypatch):
        """Break-even is reported at ``--num-assistant-tokens``, not at the default:
        k=2 on the H100 inputs is 0.701, not k=5's 0.832."""
        result, data = _run(
            monkeypatch, in_tmp_cwd, _h100, extra=["--num-assistant-tokens", "2"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert data["num_assistant_tokens"] == 2
        assert data["breakeven_acceptance"] == pytest.approx(0.701, abs=5e-4)
        assert "70.1% acceptance at k=2" in _plain(result.output)

    def test_panel_labels_the_assumption(self, in_tmp_cwd, monkeypatch):
        result, _ = _run(monkeypatch, in_tmp_cwd, _h100)
        out = _plain(result.output)
        assert "83.2%" in out, out
        assert "modelled" in out.lower()
        assert "i.i.d." in out
        assert "framework overhead" in out

    def test_a_draft_no_faster_says_no_rate_pays(self, in_tmp_cwd, monkeypatch):
        def _same_speed(role, k):
            return 20.0

        result, data = _run(monkeypatch, in_tmp_cwd, _same_speed)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert data["latency_ratio"] == 1.0
        assert data["breakeven_acceptance"] is None
        assert "no acceptance rate pays" in _plain(result.output).lower()

    def test_a_best_k_that_is_still_a_slowdown_is_not_recommended(
        self, in_tmp_cwd, monkeypatch
    ):
        """Seen live (gpt2 <- tiny-gpt2 on CPU, 0% acceptance): the panel printed
        "k=1 -> 0.98x" as the best k. Below 1x no k pays, and the panel says so."""
        result, data = _run(monkeypatch, in_tmp_cwd, _h100, acceptance=(0, 100))
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert data["modelled_speedup_best_k"] < 1.0
        out = _plain(result.output).lower()
        assert "no k pays at the measured acceptance" in out

    def test_a_best_k_that_pays_is_recommended(self, in_tmp_cwd, monkeypatch):
        result, data = _run(monkeypatch, in_tmp_cwd, _h100)
        assert data["modelled_speedup_best_k"] > 1.0
        out = _plain(result.output).lower()
        assert "no k pays" not in out
        assert f"k={data['modelled_best_k']} -> " in out

    def test_a_crashed_draft_arm_keeps_everything_else(self, in_tmp_cwd, monkeypatch):
        """Best-effort, #344 pattern: acceptance, plain and assisted all survive,
        and the report says the draft arm crashed rather than looking un-run."""
        def _draft_crashes(role, k):
            if role == "draft":
                raise RuntimeError("draft OOM")
            return _h100(role, k)

        result, data = _run(monkeypatch, in_tmp_cwd, _draft_crashes)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert data["draft_status"] == "crash"
        assert data["tok_s_draft"] is None
        assert data["latency_ratio"] is None
        assert data["breakeven_acceptance"] is None
        assert data["acceptance_rate"] == 0.81
        assert data["tok_s_assisted"] == 18.90
        assert "draft-alone throughput could not be measured" in _plain(result.output)

    def test_exit_two_below_min_acceptance_is_unchanged(self, in_tmp_cwd, monkeypatch):
        result, _ = _run(monkeypatch, in_tmp_cwd, _h100, extra=["--min-acceptance", "0.9"])
        assert result.exit_code == 2


class TestSweepK:
    def test_measures_each_k_and_names_the_best(self, in_tmp_cwd, monkeypatch):
        measured = {1: 44.0, 2: 41.0, 5: 18.9}

        def _by_k(role, k):
            return measured[k] if role == "assisted" else _h100(role, k)

        calls = []
        result, data = _run(
            monkeypatch, in_tmp_cwd,
            lambda role, k: (calls.append((role, k)), _by_k(role, k))[1],
            extra=["--sweep-k", "1,2,5"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        swept = {row["k"]: row for row in data["k_sweep"]}
        assert set(swept) == {1, 2, 5}
        assert swept[1]["tok_s_assisted"] == 44.0
        assert swept[1]["speedup"] == pytest.approx(44.0 / _TARGET_TOK_S)
        assert data["measured_best_k"] == 1
        assert {k for role, k in calls if role == "assisted"} >= {1, 2, 5}
        out = _plain(result.output)
        assert "measured best k" in out.lower()

    def test_one_crashed_k_does_not_discard_the_others(self, in_tmp_cwd, monkeypatch):
        def _k2_crashes(role, k):
            if role == "assisted" and k == 2:
                raise RuntimeError("boom")
            return {1: 44.0, 5: 18.9}.get(k, 18.9) if role == "assisted" else _h100(role, k)

        result, data = _run(monkeypatch, in_tmp_cwd, _k2_crashes, extra=["--sweep-k", "1,2,5"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        swept = {row["k"]: row for row in data["k_sweep"]}
        assert swept[2]["status"] == "crash" and swept[2]["tok_s_assisted"] is None
        assert swept[1]["status"] == "complete"
        assert data["measured_best_k"] == 1

    @pytest.mark.parametrize(
        "value", ["0", "65", "1,x", "", "1,,2", "1,2,3,4,5,6,7,8,9", "2,2"]
    )
    def test_bad_values_refused_before_any_model_loads(
        self, in_tmp_cwd, monkeypatch, value
    ):
        from typer.testing import CliRunner

        from soup_cli.commands import draft as draft_cmd
        from soup_cli.commands.draft import app

        (in_tmp_cwd / "p.jsonl").write_text(json.dumps({"prompt": "Hi"}) + "\n")
        loaded = []
        monkeypatch.setattr(draft_cmd, "_vocab_size_of", lambda mid, trc=False: 49152)
        monkeypatch.setattr(
            draft_cmd, "_load_pair_member",
            lambda model_id, **kw: loaded.append(model_id) or (object(), _FakeTok(49152), "cpu"),
        )
        result = CliRunner().invoke(
            app,
            ["measure", "--target", "org/target", "--draft", "org/tiny",
             "--prompts", "p.jsonl", "--sweep-k", value],
        )
        out = _plain(result.output).lower()
        assert "no such option" not in out, "the option must exist and refuse the value"
        assert result.exit_code == 1, (result.exit_code, out)
        assert loaded == [], "refused before loading either model"
        assert "--sweep-k" in out

    def test_without_the_flag_no_sweep_runs(self, in_tmp_cwd, monkeypatch):
        calls = []
        result, data = _run(
            monkeypatch, in_tmp_cwd, lambda role, k: (calls.append((role, k)), _h100(role, k))[1]
        )
        assert result.exit_code == 0
        assert [c for c in calls if c[0] == "assisted"] == [("assisted", 5)]
        assert data["k_sweep"] is None and data["measured_best_k"] is None


def test_isfinite_guard_used():
    """Sanity for the fixture numbers: all finite."""
    assert all(math.isfinite(x) for x in (_TARGET_TOK_S, _DRAFT_TOK_S, _ALPHA))
