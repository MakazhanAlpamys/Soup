"""``soup apply`` makes no Hub request; ``soup plan`` says before it makes one.

``soup apply`` compares the plan it builds with the one in the state file by hash
(``detect_drift``), so the peak-VRAM estimate in that plan is never read; pricing it
cost a request to the Hub (and a wait) on every apply. ``soup plan`` does need the
number, and prints one dim line before it reaches the Hub.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import model_shape
from soup_cli.utils.terraform_plan import build_plan
from tests._tiny_checkpoint import TINY_LLAMA, write_tiny_model
from tests.conftest import strip_ansi

LINE = "Reading the model's config.json from the Hub"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _document(base: str, **training) -> dict:
    return {
        "base": base,
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "quantization": "none", **training},
    }


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


@pytest.fixture
def hub(monkeypatch):
    """A Hub that knows ``someone/model``, recording every lookup made of it."""
    calls: list[str] = []

    def lookup(repo, **kw):
        calls.append("config")
        return dict(TINY_LLAMA), None

    def count(repo, **kw):
        calls.append("count")
        return 1_234_567

    monkeypatch.setattr(model_shape, "_hub_lookup", lookup)
    monkeypatch.setattr(model_shape, "_hub_param_count", count)
    monkeypatch.setattr(model_shape, "_config_cached", lambda repo: False)
    return calls


def _invoke(*args: str):
    result = CliRunner().invoke(app, list(args))
    return result, _plain(result.output)


def _plan(workdir, base: str = "someone/model"):
    (workdir / "soup.yaml").write_text(yaml.safe_dump(_document(base)), encoding="utf-8")
    result, text = _invoke("plan", "--config", "soup.yaml")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return text


class TestApplyMakesNoHubRequest:
    def test_plan_is_the_control_and_does_reach_the_lookups(self, workdir, hub):
        _plan(workdir)
        assert hub == ["config", "count"]

    @pytest.mark.parametrize("args", [["apply", "--dry-run"], ["apply"]])
    def test_apply_never_looks_the_model_up(self, workdir, hub, args):
        _plan(workdir)
        hub.clear()
        result, text = _invoke(*args, "--config", "soup.yaml")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub == []
        assert LINE not in text

    def test_apply_still_prints_its_dry_run_numbers(self, workdir, hub):
        _plan(workdir)
        result, text = _invoke("apply", "--dry-run", "--config", "soup.yaml")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "No drift detected" in text
        assert "Expected cost:" in text
        assert "ETA:" in text

    def test_apply_still_refuses_on_drift(self, workdir, hub):
        _plan(workdir)
        document = _document("someone/model", epochs=3)
        (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        hub.clear()
        result, text = _invoke("apply", "--dry-run", "--config", "soup.yaml")
        assert result.exit_code == 3
        assert "config_sha" in text
        assert hub == []

    def test_apply_does_not_need_the_model_to_be_readable_at_all(
        self, workdir, hub, monkeypatch
    ):
        _plan(workdir)

        def boom(*args, **kwargs):
            raise AssertionError("apply read the model's shape")

        monkeypatch.setattr(model_shape, "read_model_shape", boom)
        result, _ = _invoke("apply", "--dry-run", "--config", "soup.yaml")
        assert result.exit_code == 0, (result.output, repr(result.exception))


class TestBuildPlan:
    def test_the_estimate_can_be_left_out(self, workdir, hub):
        plan = build_plan(_document("someone/model"), estimate_vram=False)
        assert hub == []
        assert plan.peak_vram_gb is None
        assert "not estimated" in plan.peak_vram_reason

    def test_the_default_still_estimates(self, workdir, hub):
        plan = build_plan(_document("someone/model"))
        assert hub == ["config", "count"]
        assert plan.peak_vram_gb is not None

    def test_the_hashes_do_not_depend_on_the_estimate(self, workdir, hub):
        with_estimate = build_plan(_document("someone/model"))
        without = build_plan(_document("someone/model"), estimate_vram=False)
        assert (with_estimate.config_sha, with_estimate.dataset_sha) == (
            without.config_sha,
            without.dataset_sha,
        )
        assert with_estimate.estimated_minutes == without.estimated_minutes


class TestPlanSaysBeforeItReachesTheHub:
    def test_one_dim_line_before_the_lookup(self, workdir, hub):
        text = _plan(workdir)
        assert text.count(LINE) == 1
        assert text.index(LINE) < text.index("Training plan")

    def test_the_line_names_the_way_to_skip_it(self, workdir, hub):
        assert "HF_HUB_OFFLINE=1" in _plan(workdir)

    def test_the_line_is_printed_before_the_lookup_not_after(self, workdir, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(model_shape, "_config_cached", lambda repo: False)
        monkeypatch.setattr(
            model_shape, "_hub_lookup", lambda repo, **kw: order.append("lookup") or (None, None)
        )
        from soup_cli.commands import plan as plan_module

        real_print = plan_module.console.print

        def recording(*args, **kwargs):
            if args and isinstance(args[0], str) and LINE in args[0]:
                order.append("line")
            return real_print(*args, **kwargs)

        monkeypatch.setattr(plan_module.console, "print", recording)
        _plan(workdir)
        assert order == ["line", "lookup"]

    def test_a_cached_config_still_says_it_when_the_size_comes_from_the_hub(
        self, workdir, hub, monkeypatch
    ):
        monkeypatch.setattr(model_shape, "_config_cached", lambda repo: True)
        assert _plan(workdir).count(LINE) == 1

    def test_a_cached_model_with_its_weights_says_nothing(self, workdir, hub, monkeypatch):
        # The size comes from the cached snapshot's headers: no request is made.
        snapshot = str(workdir / "tiny")
        monkeypatch.setattr(model_shape, "_config_cached", lambda repo: True)
        monkeypatch.setattr(
            model_shape, "_hub_lookup", lambda repo, **kw: (dict(TINY_LLAMA), snapshot)
        )
        text = _plan(workdir)
        assert LINE not in text
        assert hub == []

    @pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_says_nothing(self, workdir, hub, monkeypatch, variable):
        monkeypatch.setenv(variable, "1")
        assert LINE not in _plan(workdir)

    def test_a_local_model_says_nothing(self, workdir, hub):
        text = _plan(workdir, base=str(workdir / "tiny"))
        assert LINE not in text
        assert hub == []

    @pytest.mark.parametrize(
        "extra",
        [{"stream_layers": True}, {"batch_size": "auto"}, {"quantization": "hqq:4bit"}],
    )
    def test_a_run_the_estimate_skips_says_nothing(self, workdir, hub, extra):
        document = _document("someone/model", **extra)
        (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        result, text = _invoke("plan", "--config", "soup.yaml")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert LINE not in text
        assert hub == []

    def test_building_a_plan_needs_no_callback(self):
        # The callback is optional: nothing else that builds a plan has to pass one.
        assert build_plan(_document("someone/model")).task == "sft"
