"""The estimate names the field that stops it, and says before it reaches the Hub (#2306).

``hardware_fit_input_from_config`` returned ``None`` for four different causes, so
``soup plan``, ``soup profile`` and the MCP ``profile`` tool printed one combined line
that listed every cause. ``hardware_fit_input_or_reason`` returns a ``FitUnavailable``
naming the one field. ``soup plan`` validated the config twice, the second time after
a YAML round trip; it now prices the config it validated. And only ``soup plan``
printed a line before the estimate's Hub request: ``soup profile`` (``--config`` and
``--model``) now print the same line; the train pre-flight leaves out the offline hint.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import profile as profile_mod
from soup_cli.commands import train as train_mod
from soup_cli.config import loader
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import model_shape
from soup_cli.utils.hardware_fit import (
    FitUnavailable,
    HardwareFitInput,
    hardware_fit_input_from_config,
    hardware_fit_input_or_reason,
)
from soup_cli.utils.model_shape import HUB_REQUEST_LINE, HUB_REQUEST_LINE_BEFORE_DOWNLOAD
from soup_cli.utils.terraform_plan import build_plan
from tests._tiny_checkpoint import TINY_LLAMA, write_tiny_model
from tests.conftest import make_cli_runner, strip_ansi

# The real readers, before the suite's autouse fixture swaps them for an empty Hub.
_REAL_HUB_LOOKUP = model_shape._hub_lookup
_REAL_HUB_PARAM_COUNT = model_shape._hub_param_count

HUB_MODEL = "someone/model"
LARGE_CARD = {"memory_total_bytes": int(96e9)}


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _document(base: str = HUB_MODEL, *, max_length: int = 512, **training) -> dict:
    return {
        "base": base,
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": max_length},
        "training": {"batch_size": 2, "quantization": "none", **training},
    }


def _cfg(base: str = HUB_MODEL, **training):
    return load_config_from_string(yaml.safe_dump(_document(base, **training)))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


@pytest.fixture
def huge_model(monkeypatch):
    """A Hub model whose safetensors headers count 2,000B parameters."""
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(TINY_LLAMA), None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 2 * 10**12)


# --- which cause stopped the estimate ---------------------------------------------

#: (training overrides, field, words the reason says). The model is a local directory.
_CAUSES = {
    "batch_size auto": (
        {"batch_size": "auto"},
        "training.batch_size",
        "training.batch_size is auto, which is chosen after the model loads",
    ),
    "batch_size above the range": (
        {"batch_size": 2048},
        "training.batch_size",
        "training.batch_size is outside 1 to 1,024, the range the estimate prices",
    ),
    "a quantization with no memory model": (
        {"quantization": "hqq:4bit"},
        "training.quantization",
        "training.quantization has no memory model in the estimate",
    ),
}


class TestTheReason:
    @pytest.mark.parametrize("cause", list(_CAUSES))
    def test_each_cause_names_its_field(self, workdir, cause):
        training, fld, words = _CAUSES[cause]
        result = hardware_fit_input_or_reason(_cfg(str(workdir / "tiny"), **training))
        assert result == FitUnavailable(field=fld, reason=words)

    def test_a_model_over_1000b_parameters_names_base(self, workdir, huge_model):
        result = hardware_fit_input_or_reason(_cfg())
        assert isinstance(result, FitUnavailable)
        assert result.field == "base"
        assert result.reason == (
            "the model in base has over 1,000B parameters, the most the estimate prices"
        )

    def test_max_length_out_of_range_is_named_too(self, workdir):
        # The schema bounds data.max_length to the estimate's range, so only a config
        # built without validation reaches this guard.
        cfg = _cfg(str(workdir / "tiny"))
        cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"max_length": 10**7})})
        result = hardware_fit_input_or_reason(cfg)
        assert isinstance(result, FitUnavailable)
        assert result.field == "data.max_length"
        assert result.reason == (
            "data.max_length is outside 64 to 1,048,576, the range the estimate prices"
        )

    def test_a_run_it_can_price_gets_its_input(self, workdir):
        # Control: the same local model with nothing out of range.
        assert isinstance(
            hardware_fit_input_or_reason(_cfg(str(workdir / "tiny"))), HardwareFitInput
        )

    @pytest.mark.parametrize("cause", list(_CAUSES))
    def test_the_old_function_still_returns_none(self, workdir, cause):
        training, _, _ = _CAUSES[cause]
        assert hardware_fit_input_from_config(_cfg(str(workdir / "tiny"), **training)) is None

    def test_the_reason_holds_no_value_from_the_config(self, workdir, monkeypatch):
        monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(TINY_LLAMA), None))
        monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 2 * 10**12)
        huge = hardware_fit_input_or_reason(_cfg("someone/huge-secret-name"))
        quant = hardware_fit_input_or_reason(
            _cfg(str(workdir / "tiny"), quantization="hqq:4bit")
        )
        assert "huge-secret-name" not in huge.reason
        assert "hqq" not in quant.reason
        assert str(workdir) not in quant.reason

    def test_profile_mode_still_prices_auto_and_an_unmodelled_quantization(self, workdir):
        for training in ({"batch_size": "auto"}, {"quantization": "hqq:4bit"}):
            cfg = _cfg(str(workdir / "tiny"), **training)
            assert isinstance(
                hardware_fit_input_or_reason(cfg, profile_defaults=True), HardwareFitInput
            )

    def test_any_other_refusal_gets_the_generic_reason(self, workdir, monkeypatch):
        # No validated config is known to reach this branch; the dataclass stands in.
        from soup_cli.utils import hardware_fit

        def refuse(**kwargs):
            raise ValueError("refused")

        monkeypatch.setattr(hardware_fit, "HardwareFitInput", refuse)
        result = hardware_fit_input_or_reason(_cfg(str(workdir / "tiny")))
        assert result == FitUnavailable(
            field=None, reason="the run's inputs are outside the range the estimate prices"
        )


class TestWhereTheReasonIsPrinted:
    @pytest.mark.parametrize("cause", list(_CAUSES))
    def test_plan_records_it(self, workdir, cause):
        training, _, words = _CAUSES[cause]
        document = _document(str(workdir / "tiny"), **training)
        (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        result = make_cli_runner().invoke(app, ["plan", "--config", "soup.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "unknown:" in _plain(result.output)
        state = json.loads((workdir / "soup.tfstate").read_text(encoding="utf-8"))
        assert state["plan"]["peak_vram_gb"] is None
        assert state["plan"]["peak_vram_reason"] == words

    def test_plan_records_a_model_over_1000b(self, workdir, huge_model):
        plan = build_plan(_document())
        assert plan.peak_vram_gb is None
        assert plan.peak_vram_reason.startswith("the model in base has over 1,000B")

    def test_profile_prints_it(self, workdir):
        document = _document(str(workdir / "tiny"), batch_size=2048)
        (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        result = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert _plain(result.output) == (
            "Cannot estimate this run: training.batch_size is outside 1 to 1,024, the "
            "range the estimate prices."
        )

    def test_the_mcp_tool_raises_it(self, workdir):
        from soup_cli.mcp_server.registry import McpToolError, tool_profile

        document = _document(str(workdir / "tiny"), batch_size=2048)
        (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(McpToolError) as error:
            tool_profile({"config": "soup.yaml"})
        assert str(error.value) == (
            "training.batch_size is outside 1 to 1,024, the range the estimate prices"
        )


# --- soup plan validates once -----------------------------------------------------


@pytest.fixture
def builds(monkeypatch):
    """How many times a ``SoupConfig`` is built from a mapping."""
    calls: list[int] = []
    real = loader._build_config

    def counting(raw):
        calls.append(1)
        return real(raw)

    monkeypatch.setattr(loader, "_build_config", counting)
    return calls


class TestPlanValidatesOnce:
    def test_soup_plan_builds_the_config_once(self, workdir, builds):
        document = _document(str(workdir / "tiny"))
        (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        result = make_cli_runner().invoke(app, ["plan", "--config", "soup.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(builds) == 1
        state = json.loads((workdir / "soup.tfstate").read_text(encoding="utf-8"))
        assert state["plan"]["peak_vram_gb"] is not None

    def test_a_validated_config_is_not_built_again(self, workdir, builds):
        document = _document(str(workdir / "tiny"))
        soup_config = load_config_from_string(yaml.safe_dump(document))
        builds.clear()
        plan = build_plan(document, soup_config=soup_config)
        assert builds == []
        assert plan.peak_vram_gb is not None

    def test_a_dict_alone_is_built_once(self, workdir, builds):
        plan = build_plan(_document(str(workdir / "tiny")))
        assert len(builds) == 1
        assert plan.peak_vram_gb is not None

    def test_the_hashes_come_from_the_mapping(self, workdir):
        document = _document(str(workdir / "tiny"))
        soup_config = load_config_from_string(yaml.safe_dump(document))
        assert build_plan(document, soup_config=soup_config).config_sha == build_plan(
            document
        ).config_sha

    def test_a_dict_that_does_not_validate_is_still_recorded_not_raised(self, workdir):
        plan = build_plan({**_document(str(workdir / "tiny")), "no_such_key": 1})
        assert plan.peak_vram_gb is None
        assert plan.peak_vram_reason == "the config does not validate"


# --- the line before the Hub request ------------------------------------------------


class _FakeApi:
    def __init__(self, events: list[str]):
        self._events = events

    def get_safetensors_metadata(self, repo):
        self._events.append("count")
        return SimpleNamespace(parameter_count={"BF16": 1_234_567})


@pytest.fixture
def hub(workdir, monkeypatch):
    """The real cache and Hub readers, over a cache in ``tmp_path`` and a fake network.

    Every request the readers send lands in the returned event list, beside each
    line a command prints before one.
    """
    import huggingface_hub
    from huggingface_hub import constants

    from soup_cli.utils import hf

    events: list[str] = []
    monkeypatch.setattr(model_shape, "_hub_lookup", _REAL_HUB_LOOKUP)
    monkeypatch.setattr(model_shape, "_hub_param_count", _REAL_HUB_PARAM_COUNT)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(workdir / "hf-cache"))

    downloaded = workdir / "downloaded"
    downloaded.mkdir()
    (downloaded / "config.json").write_text(json.dumps(TINY_LLAMA), encoding="utf-8")

    def head(url, **kw):
        events.append("head")
        return SimpleNamespace(size=(downloaded / "config.json").stat().st_size)

    def download(repo, filename, **kw):
        events.append("download")
        return str(downloaded / filename)

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", head)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda repo, name: f"hub:{repo}/{name}")
    monkeypatch.setattr(hf, "get_hf_api", lambda *a, **kw: _FakeApi(events))
    return events


def _cache(workdir, repo: str, *, weights: bool) -> None:
    """``repo`` in the Hugging Face cache under ``tmp_path``, with or without weights."""
    folder = workdir / "hf-cache" / ("models--" + repo.replace("/", "--"))
    commit = "0" * 40
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(commit, encoding="utf-8")
    snapshot = folder / "snapshots" / commit
    if weights:
        write_tiny_model(snapshot)
    else:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps(TINY_LLAMA), encoding="utf-8")


def _record_line(monkeypatch, console, events: list[str], line: str = HUB_REQUEST_LINE) -> None:
    real_print = console.print

    def recording(*args, **kwargs):
        if args and isinstance(args[0], str) and line in args[0]:
            events.append("line")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(console, "print", recording)


def _profile_config(workdir, *extra: str):
    (workdir / "soup.yaml").write_text(yaml.safe_dump(_document()), encoding="utf-8")
    return make_cli_runner().invoke(
        app, ["profile", "--config", "soup.yaml", "--gpu", "a100", *extra]
    )


def _profile_model(workdir, *extra: str):
    return make_cli_runner().invoke(
        app, ["profile", "--model", HUB_MODEL, "--gpu", "a100", *extra]
    )


def _preflight(workdir, monkeypatch, events: list[str]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=400, force_terminal=False)
    monkeypatch.setattr(train_mod, "console", console)
    _record_line(monkeypatch, console, events, HUB_REQUEST_LINE_BEFORE_DOWNLOAD)
    train_mod._hardware_fit_preflight(_cfg(), LARGE_CARD, allow_oom_attempt=False)
    return _plain(buffer.getvalue())


_COMMANDS = ["profile --config", "profile --model"]


def _run(command: str, workdir, monkeypatch, events: list[str]) -> str:
    _record_line(monkeypatch, profile_mod.console, events)
    run = _profile_config if command == "profile --config" else _profile_model
    result = run(workdir)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output)


class TestTheLineBeforeTheHub:
    @pytest.mark.parametrize("command", _COMMANDS)
    def test_profile_says_it_once_before_the_first_request(
        self, workdir, hub, monkeypatch, command
    ):
        text = _run(command, workdir, monkeypatch, hub)
        assert hub == ["line", "head", "download", "count"]
        assert text.count(HUB_REQUEST_LINE) == 1

    def test_the_pre_flight_says_it_once_before_the_first_request(
        self, workdir, hub, monkeypatch
    ):
        text = _preflight(workdir, monkeypatch, hub)
        assert hub == ["line", "head", "download", "count"]
        assert text == HUB_REQUEST_LINE_BEFORE_DOWNLOAD

    def test_the_pre_flight_line_has_no_offline_hint(self, workdir, hub, monkeypatch):
        text = _preflight(workdir, monkeypatch, hub)
        assert "HF_HUB_OFFLINE" not in text
        assert text == "Reading the model's details from the Hub to size the run..."

    def test_the_line_says_details_not_config_json(self):
        assert HUB_REQUEST_LINE.startswith("Reading the model's details from the Hub")

    def test_a_batch_it_cannot_price_sends_no_request(self, workdir, hub):
        # A Hub id with nothing cached: reading its shape would send requests.
        result = hardware_fit_input_or_reason(
            _cfg(batch_size=2048), on_hub_request=lambda: hub.append("line")
        )
        assert result == FitUnavailable(
            field="training.batch_size",
            reason="training.batch_size is outside 1 to 1,024, the range the estimate prices",
        )
        assert hub == []

    @pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_profile_config_neither_says_it_nor_sends_a_request(
        self, workdir, hub, monkeypatch, variable
    ):
        # Nothing cached: online this would read the Hub, offline the shape is guessed.
        monkeypatch.setenv(variable, "1")
        text = _run("profile --config", workdir, monkeypatch, hub)
        assert hub == []
        assert HUB_REQUEST_LINE not in text

    @pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_profile_model_neither_says_it_nor_sends_a_request(
        self, workdir, hub, monkeypatch, variable
    ):
        # Nothing cached: --model does not guess a shape, so it stops and says why.
        monkeypatch.setenv(variable, "1")
        _record_line(monkeypatch, profile_mod.console, hub)
        result = _profile_model(workdir)
        assert result.exit_code == 1
        assert f"Cannot read {HUB_MODEL}" in _plain(result.output)
        assert hub == []

    def test_offline_pre_flight_neither_says_it_nor_sends_a_request(
        self, workdir, hub, monkeypatch
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert _preflight(workdir, monkeypatch, hub) == ""
        assert hub == []

    @pytest.mark.parametrize("command", _COMMANDS)
    def test_a_cached_model_with_its_weights_says_nothing(
        self, workdir, hub, monkeypatch, command
    ):
        _cache(workdir, HUB_MODEL, weights=True)
        text = _run(command, workdir, monkeypatch, hub)
        assert hub == []
        assert HUB_REQUEST_LINE not in text

    def test_the_pre_flight_says_nothing_for_a_cached_model_with_its_weights(
        self, workdir, hub, monkeypatch
    ):
        _cache(workdir, HUB_MODEL, weights=True)
        assert _preflight(workdir, monkeypatch, hub) == ""
        assert hub == []

    @pytest.mark.parametrize("command", _COMMANDS)
    def test_a_cached_config_still_says_it_when_the_size_comes_from_the_hub(
        self, workdir, hub, monkeypatch, command
    ):
        _cache(workdir, HUB_MODEL, weights=False)
        text = _run(command, workdir, monkeypatch, hub)
        assert hub == ["line", "count"]
        assert text.count(HUB_REQUEST_LINE) == 1

    @pytest.mark.parametrize("command", _COMMANDS)
    def test_with_json_the_line_is_on_stderr_and_stdout_is_the_document(
        self, workdir, hub, command
    ):
        run = _profile_config if command == "profile --config" else _profile_model
        result = run(workdir, "--json")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert json.loads(result.stdout)["total_memory_gb"] > 0
        assert _plain(result.stderr) == HUB_REQUEST_LINE
        assert hub == ["head", "download", "count"]

    def test_the_mcp_tool_sends_no_request(self, workdir, hub):
        from soup_cli.mcp_server.registry import McpToolError, tool_profile

        (workdir / "soup.yaml").write_text(yaml.safe_dump(_document()), encoding="utf-8")
        try:
            tool_profile({"config": "soup.yaml"})
        except McpToolError:
            pass
        assert hub == []

    def test_plan_prints_the_same_line(self, workdir, hub, monkeypatch):
        from soup_cli.commands import plan as plan_mod

        _record_line(monkeypatch, plan_mod.console, hub)
        (workdir / "soup.yaml").write_text(yaml.safe_dump(_document()), encoding="utf-8")
        result = make_cli_runner().invoke(app, ["plan", "--config", "soup.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert hub == ["line", "head", "download", "count"]
