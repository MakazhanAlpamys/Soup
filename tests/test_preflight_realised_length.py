"""The ``soup train`` pre-flight prices the longest row of the data a padded SFT run trains on.

The VRAM estimate prices rows that FILL ``data.max_length``. A run on short rows needs a
fraction of that, and was refused on a card it fits (21.7 GB predicted, 1.2 GB measured, for a
recipe whose rows were at most 79 tokens). For a run that pads each micro-batch to its
longest row (``utils/profiler._not_padded_to_the_longest_row``) the pre-flight now holds a
REFUSAL back until the data is loaded, measures the longest row with the trainer's own
tokenisation (``utils/realised_length``), and re-decides at ``min(longest row, max_length)``.

What is pinned here:

* the measurement: exact, over both splits, truncated at ``data.max_length``, the same tokens
  the trainer builds; and every case where it is not made (the run is priced at
  ``data.max_length`` exactly as before and the panel says why);
* the hold-back: only a refusal of a run the measurement applies to, only when the run would fit
  at some length; a run that fits reads nothing, a warning and ``--allow-oom-attempt`` print
  what they always did;
* the verdict: one line with both numbers when the run now fits, the refusal with both numbers
  when it still does not, and the invariant behind it all: over a grid of runs the new decision
  refuses only what the old one refused;
* ``soup train`` itself (``--dry-run`` and a real start stopped at the run registry): the
  refusal comes before the confirmation prompt and before a run is registered.

No GPU: the card is a fake ``gpu_info``; the tokenizer is a tiny byte-level BPE trained in the
test and saved next to a tiny checkpoint's ``config.json``.
"""

from __future__ import annotations

import io
import itertools
import json
import types

import pytest
import typer
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _hardware_fit_preflight, _realised_length_verdict
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, model_shape, realised_length
from soup_cli.utils.hardware_fit import VRAM_SAFETY_MARGIN, decide_hardware_fit
from soup_cli.utils.profiler import _not_padded_to_the_longest_row, profile_from_config
from soup_cli.utils.realised_length import (
    LongestRow,
    RealisedLengthUnavailable,
    longest_row_tokens,
    unpriceable_reason,
)
from tests._tiny_checkpoint import write_tiny_model
from tests._tiny_hf_models import build_tokenizer
from tests.conftest import strip_ansi

MAX_LENGTH = 4096
BATCH = 8
_WORDS = "the quick brown fox jumps over a lazy dog and then hello world again"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _messages(words: int, reply: str = "ok then") -> list[dict]:
    prompt = " ".join(itertools.islice(itertools.cycle(_WORDS.split()), words))
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
    ]


def _dataset(*word_counts: int, val: tuple[int, ...] = ()) -> dict:
    out = {"train": [{"messages": _messages(n)} for n in word_counts]}
    if val:
        out["val"] = [{"messages": _messages(n)} for n in val]
    return out


def _yaml(
    base: str,
    *,
    task: str = "sft",
    max_length: int = MAX_LENGTH,
    batch: int = BATCH,
    quantization: str = "none",
    training: dict | None = None,
    data: dict | None = None,
) -> str:
    fmt = {"sft": "alpaca", "dpo": "dpo"}[task]
    return yaml.safe_dump({
        "base": base,
        "task": task,
        "data": {
            "train": "train.jsonl", "format": fmt, "max_length": max_length, "val_split": 0,
            "chat_template": "chatml", **(data or {}),
        },
        "training": {
            "epochs": 1, "batch_size": batch, "quantization": quantization,
            "lora": {"r": 8, "alpha": 16}, **(training or {}),
        },
    })


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> str:
    """A tiny llama checkpoint (config + safetensors header) with a tokenizer beside it."""
    folder = tmp_path_factory.mktemp("realised-length-base")
    write_tiny_model(folder)
    build_tokenizer((_WORDS, "ok then", "hello world")).save_pretrained(folder)
    return str(folder)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def screen(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
    )
    return lambda: _plain(buffer.getvalue())


def _config(base: str, **kwargs):
    return load_config_from_string(_yaml(base, **kwargs))


def _input(cfg, card_gb: float = 8.0):
    """The estimate's input as the pre-flight builds it, and nothing else."""
    inp = train_mod._hardware_fit_input_or_reason(cfg, gpu_memory_gb=card_gb)
    assert isinstance(inp, hardware_fit.HardwareFitInput), inp
    return inp


def _card_between(cfg, tokens: int) -> dict:
    """A card that fits ``cfg`` at ``tokens`` per row and does not fit it at data.max_length."""
    import dataclasses

    inp = _input(cfg)
    small = decide_hardware_fit(dataclasses.replace(inp, seq_len=tokens), available_vram_gb=1e6)
    big = decide_hardware_fit(inp, available_vram_gb=1e6)
    low, high = small.required_with_margin_gb, big.required_with_margin_gb
    assert high > 1.05 * low, "the fixture's run does not grow with the length"
    return {"memory_total_bytes": int((low + high) / 2 * 1e9), "memory_total": "fake card"}


def _tokens_the_trainer_builds(base: str, messages: list[dict]) -> int:
    """The length of one row, computed here with no help from the code under test."""
    from transformers import AutoTokenizer

    from soup_cli.data.chat_templates import resolve_chat_template

    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.chat_template = resolve_chat_template("chatml")
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# --- the measurement ------------------------------------------------------------------


class TestLongestRow:
    def test_it_is_the_exact_longest_row_of_the_data(self, tiny_base):
        cfg = _config(tiny_base)
        got = longest_row_tokens(cfg, _dataset(3, 40, 7))
        assert isinstance(got, LongestRow), got
        assert got.tokens == _tokens_the_trainer_builds(tiny_base, _messages(40))
        assert got.rows == 3

    def test_the_template_counts_not_only_the_words(self, tiny_base):
        cfg = _config(tiny_base)
        got = longest_row_tokens(cfg, _dataset(10))
        assert isinstance(got, LongestRow), got
        words_only = len(
            _tokenizer(tiny_base)(" ".join(m["content"] for m in _messages(10)))["input_ids"]
        )
        assert got.tokens > words_only + 4

    def test_a_longer_validation_row_is_the_longest(self, tiny_base):
        cfg = _config(tiny_base)
        got = longest_row_tokens(cfg, _dataset(5, 6, val=(60,)))
        assert isinstance(got, LongestRow), got
        assert got.tokens == _tokens_the_trainer_builds(tiny_base, _messages(60))
        assert got.rows == 3

    def test_a_row_longer_than_max_length_is_cut_at_max_length(self, tiny_base):
        cfg = _config(tiny_base, max_length=64)
        got = longest_row_tokens(cfg, _dataset(5, 400))
        assert isinstance(got, LongestRow), got
        assert got.tokens == 64

    def test_a_multi_turn_row_counts_every_turn(self, tiny_base):
        cfg = _config(tiny_base)
        turns = [*_messages(4), *_messages(30)]
        got = longest_row_tokens(cfg, {"train": [{"messages": turns}]})
        assert isinstance(got, LongestRow), got
        assert got.tokens == _tokens_the_trainer_builds(tiny_base, turns)

    def test_the_tokenizer_is_loaded_with_the_runs_own_trust_flag_only(
        self, tiny_base, monkeypatch
    ):
        import transformers

        seen = []
        real = transformers.AutoTokenizer.from_pretrained

        def recording(*args, **kwargs):
            seen.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", recording)
        cfg = _config(tiny_base)
        longest_row_tokens(cfg, _dataset(3))
        longest_row_tokens(cfg, _dataset(3), trust_remote_code=True)
        assert [call["trust_remote_code"] for call in seen] == [False, True]

    @pytest.mark.parametrize("flags", [dict(remote_code=True)])
    def test_a_model_that_ships_its_own_code_is_not_run_to_read_a_tokenizer(
        self, tiny_base, flags
    ):
        def must_not_load(cfg, **kwargs):
            raise AssertionError("the tokenizer was loaded")

        got = longest_row_tokens(
            _config(tiny_base), _dataset(3), tokenizer_loader=must_not_load, **flags
        )
        assert isinstance(got, RealisedLengthUnavailable)
        assert "ships its own code" in got.reason

    def test_no_rows_is_unavailable_not_a_64_token_run(self, tiny_base):
        got = longest_row_tokens(_config(tiny_base), {"train": [], "val": []})
        assert got == RealisedLengthUnavailable("the data has no rows")

    def test_more_rows_than_the_cap_are_not_sampled(self, tiny_base, monkeypatch):
        monkeypatch.setattr(realised_length, "MAX_ROWS", 4)

        def must_not_load(cfg, **kwargs):
            raise AssertionError("rows past the cap were read")

        got = longest_row_tokens(
            _config(tiny_base), _dataset(1, 2, 3, val=(4, 5)), tokenizer_loader=must_not_load
        )
        assert isinstance(got, RealisedLengthUnavailable)
        assert "5 rows, more than the 4 the pre-flight reads" in got.reason
        assert "a sample is not a maximum" in got.reason

    def test_exactly_the_cap_is_still_read(self, tiny_base, monkeypatch):
        monkeypatch.setattr(realised_length, "MAX_ROWS", 3)
        assert isinstance(longest_row_tokens(_config(tiny_base), _dataset(1, 2, 3)), LongestRow)

    def test_reading_past_the_time_budget_gives_up(self, tiny_base):
        ticks = itertools.count(0, 1000)
        got = longest_row_tokens(
            _config(tiny_base), _dataset(*([3] * 200)), clock=lambda: float(next(ticks))
        )
        assert isinstance(got, RealisedLengthUnavailable)
        assert "took more than 90 seconds" in got.reason

    def test_a_row_without_messages_is_unavailable_and_named(self, tiny_base):
        got = longest_row_tokens(
            _config(tiny_base), {"train": [{"messages": _messages(3)}, {"text": "raw"}]}
        )
        assert got == RealisedLengthUnavailable("row 2 has no list of chat messages to tokenise")

    def test_a_tokenizer_that_cannot_load_is_unavailable_and_the_message_is_not_echoed(
        self, tiny_base
    ):
        def broken(cfg, **kwargs):
            raise OSError("C:\\secret\\path is not a file")

        got = longest_row_tokens(_config(tiny_base), _dataset(3), tokenizer_loader=broken)
        assert isinstance(got, RealisedLengthUnavailable)
        assert "OSError" in got.reason
        assert "secret" not in got.reason

    def test_a_missing_model_is_unavailable_offline(self, tmp_path):
        got = longest_row_tokens(_config(str(tmp_path / "nowhere")), _dataset(3))
        assert isinstance(got, RealisedLengthUnavailable)
        assert "could not be tokenised" in got.reason


def _tokenizer(base: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base)


# --- the revision the run binds ------------------------------------------------------------


_COMMIT = "a" * 40


def _bound_identity(base: str):
    """A ``BaseRevision`` already resolved to a Hub commit, as the run's is after its first load."""
    from soup_cli.utils.run_provenance import BaseRevision

    identity = BaseRevision(base)
    identity.revision, identity.source, identity._pending = _COMMIT, "hub_commit", False
    return identity


class TestTheRevisionTheRunBinds:
    def test_the_tokenizer_is_loaded_at_the_active_base_revision(self, tiny_base, monkeypatch):
        import transformers

        from soup_cli.utils.run_provenance import base_revision_scope

        tokenizer = _tokenizer(tiny_base)
        seen = []

        def recording(*args, **kwargs):
            seen.append(kwargs)
            return tokenizer

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", recording)
        cfg = types.SimpleNamespace(base="owner/model")
        with base_revision_scope(_bound_identity("owner/model")):
            realised_length._load_tokenizer(cfg, trust_remote_code=False)
        realised_length._load_tokenizer(cfg, trust_remote_code=False)
        assert seen[0]["revision"] == _COMMIT
        assert "revision" not in seen[1]  # no scope: the repository's current revision

    def test_a_scope_for_another_model_pins_nothing(self, tiny_base, monkeypatch):
        import transformers

        from soup_cli.utils.run_provenance import base_revision_scope

        tokenizer = _tokenizer(tiny_base)
        seen = []
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: seen.append(kwargs) or tokenizer,
        )
        with base_revision_scope(_bound_identity("owner/model")):
            realised_length._load_tokenizer(
                types.SimpleNamespace(base="someone/else"), trust_remote_code=False
            )
        assert "revision" not in seen[0]

    def test_the_verdict_measures_inside_the_runs_identity_scope(self, tiny_base, monkeypatch):
        from soup_cli.utils.run_provenance import base_revision_kwargs

        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        seen = []

        def fake(cfg, dataset, **kwargs):
            seen.append(base_revision_kwargs(cfg.base))
            return LongestRow(tokens=79, rows=3)

        monkeypatch.setattr(realised_length, "longest_row_tokens", fake)
        _realised_length_verdict(cfg, decision, _dataset(3), identity=_bound_identity(cfg.base))
        _realised_length_verdict(cfg, decision, _dataset(3))
        assert seen == [{"revision": _COMMIT}, {}]

    def test_the_run_keeps_the_identity_the_measurement_used(self):
        early = _bound_identity("owner/model")
        assert train_mod._base_identity_for_run("owner/model", True, early) is early

    @pytest.mark.parametrize(
        ("base", "supported"), [("someone/else", True), ("owner/model", False)]
    )
    def test_another_base_or_binding_gets_a_fresh_identity(self, base, supported):
        early = _bound_identity("owner/model")
        fresh = train_mod._base_identity_for_run(base, supported, early)
        assert fresh is not early
        assert (fresh.base, fresh.binding_supported) == (base, supported)
        assert train_mod._base_identity_for_run("owner/model", True, None).revision is None

    def test_a_single_process_run_gets_an_identity_to_measure_with(self, tiny_base, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        identity = train_mod._held_back_base_identity(_config(tiny_base))
        assert identity is not None and identity.base == tiny_base
        assert identity.binding_supported is True

    def test_a_distributed_launch_leaves_the_identity_to_rank_zero(self, tiny_base, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        assert train_mod._held_back_base_identity(_config(tiny_base)) is None
        monkeypatch.setenv("WORLD_SIZE", "not a number")
        assert train_mod._held_back_base_identity(_config(tiny_base)) is not None

    def test_soup_train_measures_inside_the_identity_the_run_then_binds(
        self, train_cli, monkeypatch
    ):
        from unittest import mock

        import soup_cli.experiment.tracker as tracker_mod
        import soup_cli.trainer.dispatch as dispatch
        from soup_cli.utils.run_provenance import _ACTIVE_BASE

        train_cli.write(3, 12, 5)
        made, measured_in, run_in = [], [], []
        real = train_mod._held_back_base_identity

        def recording(cfg):
            made.append(real(cfg))
            return made[-1]

        def fake_measure(cfg, dataset, **kwargs):
            measured_in.append(_ACTIVE_BASE.get())
            return LongestRow(tokens=79, rows=3)

        def fake_build(cfg, **kwargs):
            run_in.append(_ACTIVE_BASE.get())
            raise _RegisteredError

        monkeypatch.setattr(train_mod, "_held_back_base_identity", recording)
        monkeypatch.setattr(realised_length, "longest_row_tokens", fake_measure)
        monkeypatch.setattr(tracker_mod, "ExperimentTracker", mock.MagicMock)
        monkeypatch.setattr(dispatch, "build_trainer", fake_build)
        train_cli("--yes")
        # Identity, not equality: a fresh unresolved BaseRevision compares equal to this one.
        assert len(made) == 1
        assert len(measured_in) == 1 and measured_in[0] is made[0]
        assert len(run_in) == 1 and run_in[0] is made[0], "the run created its own identity"


# --- which runs the measurement applies to ----------------------------------------------


def _duck(**overrides):
    """A config-shaped object with every field ``unpriceable_reason`` reads (padded SFT)."""
    parts = dict(
        task="sft",
        modality="text",
        backend="transformers",
        data=types.SimpleNamespace(format="alpaca", streaming=False),
        training=types.SimpleNamespace(
            packing=False, multipack=False, stream_layers=False, hub="hf"
        ),
    )
    for key, value in overrides.items():
        if key in ("task", "modality", "backend"):
            parts[key] = value
        else:
            target = parts["data" if key == "format" or key == "streaming" else "training"]
            setattr(target, key, value)
    return types.SimpleNamespace(**parts)


class TestWhichRunsArePricedAtTheirLongestRow:
    def test_padded_text_sft_is_the_one_case(self):
        assert unpriceable_reason(_duck()) is None

    @pytest.mark.parametrize(
        "change",
        [
            dict(packing=True),
            dict(multipack=True),
            dict(task="dpo"),
            dict(task="grpo"),
            dict(task="pretrain"),
            dict(modality="vision"),
            dict(modality="audio"),
            dict(backend="unsloth"),
            dict(backend="mlx"),
            dict(streaming=True),
            dict(stream_layers=True),
        ],
        ids=lambda change: next(iter(change)) + "=" + str(next(iter(change.values()))),
    )
    def test_every_run_the_profile_does_not_price_short_is_not_priced_short_here(self, change):
        cfg = _duck(**change)
        assert _not_padded_to_the_longest_row(cfg) is not None
        assert unpriceable_reason(cfg) == _not_padded_to_the_longest_row(cfg)

    @pytest.mark.parametrize("fmt", ["raft", "pre_tokenized"])
    def test_rows_that_are_not_chat_messages_are_not_measured(self, fmt):
        assert f"data.format {fmt} does not hand the trainer chat messages" in unpriceable_reason(
            _duck(format=fmt)
        )

    def test_a_model_fetched_from_another_hub_after_the_preflight_is_not_measured(self):
        assert "training.hub is modelscope" in unpriceable_reason(_duck(hub="modelscope"))

    def test_the_schema_still_has_the_fields_the_rule_reads(self):
        from soup_cli.config.schema import DataConfig, TrainingConfig

        assert "format" in DataConfig.model_fields
        assert "hub" in TrainingConfig.model_fields

    def test_a_real_padded_config_is_priceable_and_a_packed_one_is_not(self, tiny_base):
        assert unpriceable_reason(_config(tiny_base)) is None
        assert unpriceable_reason(_config(tiny_base, training={"packing": True})) is not None


# --- the hold-back --------------------------------------------------------------------


@pytest.fixture
def must_not_read(monkeypatch):
    """Any data read or tokenizer load by the pre-flight itself is a failure."""

    def boom(*args, **kwargs):
        raise AssertionError("the pre-flight read the data or loaded a tokenizer")

    monkeypatch.setattr(realised_length, "longest_row_tokens", boom)
    monkeypatch.setattr(realised_length, "_load_tokenizer", boom)


class TestTheHoldBack:
    def test_a_refusal_the_longest_row_could_lift_is_returned_and_prints_nothing(
        self, tiny_base, screen, must_not_read
    ):
        cfg = _config(tiny_base)
        decision = _hardware_fit_preflight(
            cfg, _card_between(cfg, 100), allow_oom_attempt=False, defer_refusal_to_data=True
        )
        assert decision is not None and decision.refuse
        assert screen() == ""

    def test_without_the_flag_the_same_run_is_refused_at_once_as_before(
        self, tiny_base, screen
    ):
        cfg = _config(tiny_base)
        with pytest.raises(typer.Exit) as raised:
            _hardware_fit_preflight(cfg, _card_between(cfg, 100), allow_oom_attempt=False)
        assert raised.value.exit_code == 1
        assert "Hardware-fit gate" in screen()
        assert "exceeds" in screen()

    def test_a_run_that_fits_at_max_length_reads_nothing_and_returns_none(
        self, tiny_base, screen, must_not_read
    ):
        cfg = _config(tiny_base)
        assert _hardware_fit_preflight(
            cfg, {"memory_total_bytes": int(1e12)}, allow_oom_attempt=False,
            defer_refusal_to_data=True,
        ) is None
        assert screen() == ""

    def test_a_run_no_row_length_can_fit_is_refused_at_once_without_reading_the_data(
        self, tiny_base, screen, must_not_read
    ):
        cfg = _config(tiny_base)
        with pytest.raises(typer.Exit) as raised:
            _hardware_fit_preflight(
                cfg, {"memory_total_bytes": int(1e8), "memory_total": "tiny"},
                allow_oom_attempt=False, defer_refusal_to_data=True,
            )
        assert raised.value.exit_code == 1
        assert "Hardware-fit gate" in screen()

    def test_a_run_already_at_the_shortest_length_has_nothing_to_gain(
        self, tiny_base, screen, must_not_read
    ):
        cfg = _config(tiny_base, max_length=64)
        with pytest.raises(typer.Exit):
            _hardware_fit_preflight(
                cfg, {"memory_total_bytes": int(1e8), "memory_total": "tiny"},
                allow_oom_attempt=False, defer_refusal_to_data=True,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(training={"packing": True}),
            dict(training={"multipack": True}),
            dict(task="dpo"),
        ],
        ids=["packing", "multipack", "dpo"],
    )
    def test_a_run_the_measurement_does_not_apply_to_is_refused_at_once(
        self, tiny_base, screen, must_not_read, kwargs
    ):
        cfg = _config(tiny_base, **kwargs)
        card = _card_between(cfg, 100)
        with pytest.raises(typer.Exit) as raised:
            _hardware_fit_preflight(
                cfg, card, allow_oom_attempt=False, defer_refusal_to_data=True
            )
        assert raised.value.exit_code == 1
        assert "Hardware-fit gate" in screen()

    def test_a_model_that_ships_its_own_code_is_refused_at_once_and_its_tokenizer_is_not_run(
        self, tmp_path, screen, must_not_read
    ):
        write_tiny_model(tmp_path / "custom", auto_map={"AutoConfig": "configuration.Custom"})
        cfg = _config(str(tmp_path / "custom"))
        card = _card_between(cfg, 100)
        with pytest.raises(typer.Exit) as raised:
            _hardware_fit_preflight(
                cfg, card, allow_oom_attempt=False, defer_refusal_to_data=True
            )
        assert raised.value.exit_code == 1
        assert "Hardware-fit gate" in screen()

    def test_allow_oom_attempt_prints_the_panel_it_always_printed_and_reads_nothing(
        self, tiny_base, screen, must_not_read
    ):
        cfg = _config(tiny_base)
        card = _card_between(cfg, 100)
        assert _hardware_fit_preflight(
            cfg, card, allow_oom_attempt=True, defer_refusal_to_data=True
        ) is None
        assert "--allow-oom-attempt set: launching anyway." in screen()
        assert "Hardware-fit warning" in screen()

    def test_a_warning_outside_the_calibrated_range_is_printed_not_held_back(
        self, tiny_base, screen, must_not_read
    ):
        cfg = _config(tiny_base, training={"use_liger": True})
        card = _card_between(cfg, 100)
        assert _hardware_fit_preflight(
            cfg, card, allow_oom_attempt=False, defer_refusal_to_data=True
        ) is None
        assert "Warning only: launching anyway." in screen()
        assert "fused-loss flag use_liger" in screen()

    def test_the_default_leaves_the_decision_call_count_alone(self, tiny_base):
        cfg = _config(tiny_base)
        seen = []
        real = hardware_fit.decide_hardware_fit

        def counting(inp, **kwargs):
            seen.append(inp.seq_len)
            return real(inp, **kwargs)

        card = _card_between(cfg, 100)
        hardware_fit.decide_hardware_fit = counting
        try:
            with pytest.raises(typer.Exit):
                _hardware_fit_preflight(cfg, card, allow_oom_attempt=False)
        finally:
            hardware_fit.decide_hardware_fit = real
        assert seen == [MAX_LENGTH]


# --- the verdict ----------------------------------------------------------------------


def _measure(monkeypatch, result):
    """Make the measurement return ``result`` (a LongestRow or RealisedLengthUnavailable)."""
    calls = []

    def fake(cfg, dataset, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(realised_length, "longest_row_tokens", fake)
    return calls


def _held_back(cfg, tokens: int):
    decision = _hardware_fit_preflight(
        cfg, _card_between(cfg, tokens), allow_oom_attempt=False, defer_refusal_to_data=True
    )
    assert decision is not None
    return decision


class TestTheVerdict:
    def test_a_run_that_fits_at_its_longest_row_continues_with_one_line_and_both_numbers(
        self, tiny_base, screen, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        calls = _measure(monkeypatch, LongestRow(tokens=79, rows=72))
        _realised_length_verdict(cfg, decision, _dataset(3), trust_remote_code=True)
        text = screen()
        assert calls == [{"trust_remote_code": True, "remote_code": False}]
        assert "priced at 79 tokens, the longest row; data.max_length is 4096" in text
        assert "(72 rows read)" in text
        # What is about to happen, and where the tokenizer comes from, is said before the wait.
        assert (
            "the run does not fit the card at data.max_length (4096); reading the longest row "
            "of the data, with the model's tokenizer from the cache or the Hub, to price it"
        ) in text
        assert "Hardware-fit" not in text
        at_row = decide_hardware_fit(
            _replace_seq(decision.inp, 79), available_vram_gb=decision.report.available_vram_gb
        )
        assert f"Predicted peak VRAM {at_row.peak_vram_gb:.1f} GB" in text
        assert f"(+10% margin = {at_row.required_with_margin_gb:.1f} GB)" in text
        assert f"at data.max_length it would be {decision.report.peak_vram_gb:.1f} GB" in text
        assert (
            f"({decision.report.required_with_margin_gb:.1f} GB with the margin)" in text
        )

    def test_the_trust_flag_defaults_to_off_and_the_models_own_code_flag_is_passed_on(
        self, tiny_base, screen, monkeypatch
    ):
        import dataclasses

        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        calls = _measure(monkeypatch, LongestRow(tokens=79, rows=72))
        _realised_length_verdict(cfg, decision, _dataset(3))
        shape = dataclasses.replace(decision.inp.shape, remote_code=True)
        custom = dataclasses.replace(decision, inp=dataclasses.replace(decision.inp, shape=shape))
        _realised_length_verdict(cfg, custom, _dataset(3))
        assert calls == [
            {"trust_remote_code": False, "remote_code": False},
            {"trust_remote_code": False, "remote_code": True},
        ]

    def test_the_number_is_the_one_profile_from_config_gives_at_that_length(
        self, tiny_base, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        priced = profile_from_config(cfg, allow_network=False, seq_len=79)
        at_row = decide_hardware_fit(
            _replace_seq(decision.inp, 79), available_vram_gb=decision.report.available_vram_gb
        )
        assert at_row.breakdown.estimate.total_bytes == priced["total_memory_bytes"]

    def test_a_run_that_still_does_not_fit_is_refused_with_both_numbers(
        self, tiny_base, screen, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        _measure(monkeypatch, LongestRow(tokens=MAX_LENGTH - 1, rows=10))
        with pytest.raises(typer.Exit) as raised:
            _realised_length_verdict(cfg, decision, _dataset(3))
        assert raised.value.exit_code == 1
        text = screen()
        assert "Hardware-fit gate" in text
        assert "priced at 4095 tokens, the longest row; data.max_length is 4096" in text
        assert f"at data.max_length it would be {decision.report.peak_vram_gb:.1f} GB" in text
        assert "--allow-oom-attempt" in text
        assert "Reduce batch_size / max_length" in text

    def test_the_panel_of_a_run_that_still_does_not_fit_shows_the_row_length_numbers(
        self, tiny_base, screen, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        _measure(monkeypatch, LongestRow(tokens=MAX_LENGTH - 1, rows=10))
        with pytest.raises(typer.Exit):
            _realised_length_verdict(cfg, decision, _dataset(3))
        at_row = decide_hardware_fit(
            _replace_seq(decision.inp, MAX_LENGTH - 1),
            available_vram_gb=decision.report.available_vram_gb,
        )
        assert f"Predicted peak VRAM {at_row.peak_vram_gb:.1f} GB" in screen()

    def test_a_run_whose_peak_fits_the_card_but_not_with_the_margin_is_still_refused(
        self, tiny_base, screen, monkeypatch
    ):
        # The lift needs the 10% margin, not the bare peak. 2,000 tokens, not 79: at 79 the tiny
        # fixture is priced almost as at 64, so a card in the margin band there is refused at
        # once by the hold-back and never reaches the verdict.
        cfg = _config(tiny_base)
        at_row = decide_hardware_fit(
            _replace_seq(_input(cfg), 2000), available_vram_gb=1e6
        )
        card_gb = (at_row.peak_vram_gb + at_row.required_with_margin_gb) / 2
        assert at_row.peak_vram_gb < card_gb < at_row.required_with_margin_gb
        card = {"memory_total_bytes": int(card_gb * 1e9), "memory_total": "fake card"}
        decision = _hardware_fit_preflight(
            cfg, card, allow_oom_attempt=False, defer_refusal_to_data=True
        )
        assert decision is not None
        _measure(monkeypatch, LongestRow(tokens=2000, rows=10))
        with pytest.raises(typer.Exit) as raised:
            _realised_length_verdict(cfg, decision, _dataset(3))
        assert raised.value.exit_code == 1
        assert "Hardware-fit gate" in screen()

    def test_a_longest_row_below_the_estimates_floor_is_priced_at_the_floor_and_says_so(
        self, tiny_base, screen, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 64)
        _measure(monkeypatch, LongestRow(tokens=10, rows=3))
        _realised_length_verdict(cfg, decision, _dataset(3))
        text = screen()
        assert "priced at 64 tokens, the shortest length the estimate prices" in text
        assert "(the longest row is 10)" in text

    def test_a_longest_row_that_reaches_max_length_gets_the_old_panel_and_the_reason(
        self, tiny_base, screen, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        _measure(monkeypatch, LongestRow(tokens=MAX_LENGTH, rows=10))
        with pytest.raises(typer.Exit) as raised:
            _realised_length_verdict(cfg, decision, _dataset(3))
        assert raised.value.exit_code == 1
        text = screen()
        assert "Hardware-fit gate" in text
        assert "the longest row reaches data.max_length (4096 tokens)" in text
        assert f"Predicted peak VRAM {decision.report.peak_vram_gb:.1f} GB" in text

    def test_an_unreadable_longest_row_gets_the_original_panel_and_the_reason(
        self, tiny_base, screen, monkeypatch
    ):
        cfg = _config(tiny_base)
        decision = _held_back(cfg, 100)
        _measure(monkeypatch, RealisedLengthUnavailable("the data has no rows"))
        with pytest.raises(typer.Exit) as raised:
            _realised_length_verdict(cfg, decision, _dataset(3))
        assert raised.value.exit_code == 1
        text = screen()
        assert "not priced at the longest row of the data: the data has no rows." in text
        assert f"Predicted peak VRAM {decision.report.peak_vram_gb:.1f} GB" in text

    def test_that_panel_is_the_one_an_undeferred_run_gets_plus_the_reason(
        self, tiny_base, monkeypatch
    ):
        """Line for line the old panel; only the reason line and the reading line are new."""
        cfg = _config(tiny_base)
        card = _card_between(cfg, 100)
        old = io.StringIO()
        monkeypatch.setattr(
            train_mod, "console", Console(file=old, width=400, force_terminal=False)
        )
        with pytest.raises(typer.Exit):
            _hardware_fit_preflight(cfg, card, allow_oom_attempt=False)
        new = io.StringIO()
        monkeypatch.setattr(
            train_mod, "console", Console(file=new, width=400, force_terminal=False)
        )
        decision = _hardware_fit_preflight(
            cfg, card, allow_oom_attempt=False, defer_refusal_to_data=True
        )
        _measure(monkeypatch, RealisedLengthUnavailable("the data has no rows"))
        with pytest.raises(typer.Exit):
            _realised_length_verdict(cfg, decision, _dataset(3))
        new_lines = strip_ansi(new.getvalue()).splitlines()
        assert sum("not priced at the longest row of the data" in line for line in new_lines) == 1
        kept = [
            line
            for line in new_lines
            if "reading the longest row" not in line
            and "not priced at the longest row of the data" not in line
        ]
        assert kept == strip_ansi(old.getvalue()).splitlines()


def _replace_seq(inp, seq_len):
    import dataclasses

    return dataclasses.replace(inp, seq_len=seq_len)


# --- the invariant: this can lift a refusal and never add one -------------------------------


_EVIDENCE = None


def _evidence_models() -> dict:
    global _EVIDENCE
    if _EVIDENCE is None:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "benchmarks/evidence"
        _EVIDENCE = json.loads(
            (path / "vram-qwen3-0.6b-realised-length.json").read_text(encoding="utf-8")
        )["models"]
    return _EVIDENCE


def _outcome(action) -> str:
    try:
        action()
    except typer.Exit as exit_:
        return "refused" if exit_.exit_code == 1 else f"exit {exit_.exit_code}"
    return "passed"


GRID_MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")
GRID_CARDS_GB = (4, 6, 8, 12, 24)
GRID_MEASURES = {
    "10 tokens": LongestRow(tokens=10, rows=5),
    "300 tokens": LongestRow(tokens=300, rows=5),
    "at the cap": LongestRow(tokens=1024, rows=5),
    "unreadable": RealisedLengthUnavailable("no tokenizer"),
}
GRID_VARIANTS = {
    "padded sft": dict(),
    "packing": dict(training={"packing": True}),
    "dpo": dict(task="dpo"),
}


def test_over_a_grid_of_runs_the_new_decision_refuses_only_what_the_old_one_refused(
    monkeypatch, tmp_path
):
    known = _evidence_models()
    chosen = {}
    monkeypatch.setattr(
        model_shape, "_hub_lookup", lambda repo, **kw: (dict(known[repo]["config"]), None)
    )
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: known[repo]["params"])
    monkeypatch.setattr(train_mod, "console", Console(file=io.StringIO(), width=400))

    def fake_measure(cfg, dataset, **kwargs):
        return chosen["measure"]

    monkeypatch.setattr(realised_length, "longest_row_tokens", fake_measure)

    cells = lifted = kept_refused = 0
    for model, quant, max_length, batch, (variant, extra) in itertools.product(
        GRID_MODELS, ("none", "4bit"), (1024, 4096), (1, 4), GRID_VARIANTS.items()
    ):
        cfg = load_config_from_string(
            _yaml(model, max_length=max_length, batch=batch, quantization=quant, **extra)
        )
        for card_gb, (label, measure) in itertools.product(GRID_CARDS_GB, GRID_MEASURES.items()):
            chosen["measure"] = measure
            card = {"memory_total_bytes": int(card_gb * 1e9), "memory_total": f"{card_gb} GB"}
            old = _outcome(lambda: _hardware_fit_preflight(cfg, card, allow_oom_attempt=False))

            def new_run():
                held = _hardware_fit_preflight(
                    cfg, card, allow_oom_attempt=False, defer_refusal_to_data=True
                )
                if held is not None:
                    _realised_length_verdict(cfg, held, _dataset(3))

            new = _outcome(new_run)
            where = (model, quant, max_length, batch, variant, card_gb, label)
            cells += 1
            # No new refusal, ever.
            assert not (new == "refused" and old != "refused"), where
            # What passed before passes now.
            assert not (old == "passed" and new != "passed"), where
            if variant != "padded sft":
                assert new == old, where  # a run the measurement does not apply to is untouched
            if old == "refused" and new == "passed":
                lifted += 1
            if old == "refused" and new == "refused":
                kept_refused += 1
    assert cells == 3 * 2 * 2 * 2 * 3 * 5 * 4
    assert lifted > 20, "the grid never exercised a lifted refusal"
    assert kept_refused > 20, "the grid never exercised a refusal that stays"


# --- soup train, stopped where the pre-flight and the data meet --------------------------


class _RegisteredError(Exception):
    """Raised by the fake run registry: the run got past the pre-flight and the data."""


@pytest.fixture
def train_cli(tmp_path, monkeypatch, tiny_base, screen):
    monkeypatch.chdir(tmp_path)

    def write(*word_counts: int) -> None:
        rows = [
            {"instruction": " ".join(itertools.islice(itertools.cycle(_WORDS.split()), n)),
             "output": "ok then"}
            for n in word_counts
        ]
        (tmp_path / "train.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    cfg = _config(tiny_base)
    card = _card_between(cfg, 100)
    monkeypatch.setattr(train_mod, "detect_device", lambda **kw: ("cuda", "fake card"))
    monkeypatch.setattr(train_mod, "get_gpu_info", lambda **kw: dict(card))
    registered = []

    class FakeTracker:
        def start_run(self, **kwargs):
            registered.append(kwargs)
            raise _RegisteredError

    import soup_cli.experiment.tracker as tracker_mod

    monkeypatch.setattr(tracker_mod, "ExperimentTracker", FakeTracker)
    (tmp_path / "soup.yaml").write_text(_yaml(tiny_base), encoding="utf-8")

    def invoke(*args: str):
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", *args])
        return result, screen() + " " + _plain(result.output), registered

    invoke.write = write
    return invoke


class TestSoupTrain:
    def test_a_dry_run_on_short_rows_passes_a_card_the_max_length_estimate_refuses(
        self, train_cli
    ):
        train_cli.write(3, 12, 5)
        result, text, _ = train_cli("--dry-run", "--yes")
        assert result.exit_code == 0, (text, repr(result.exception))
        assert "the longest row; data.max_length is 4096" in text
        assert "(3 rows read)" in text
        assert "Hardware-fit gate" not in text
        assert "Data OK: 3 train samples" in text

    def test_a_real_start_on_short_rows_reaches_the_run_registry(self, train_cli):
        train_cli.write(3, 12, 5)
        result, text, registered = train_cli("--yes")
        assert isinstance(result.exception, _RegisteredError), (text, repr(result.exception))
        assert len(registered) == 1
        assert "the longest row; data.max_length is 4096" in text

    def test_rows_that_fill_the_cap_are_refused_before_a_run_is_registered(self, train_cli):
        train_cli.write(3, 3000, 5)
        result, text, registered = train_cli("--yes")
        assert result.exit_code == 1, (text, repr(result.exception))
        assert registered == []
        assert "Hardware-fit gate" in text
        assert "the longest row reaches data.max_length (4096 tokens)" in text

    def test_the_refusal_comes_before_the_start_training_prompt(self, train_cli):
        train_cli.write(3, 3000, 5)
        result, text, registered = train_cli()  # no --yes, and stdin is not a terminal
        assert result.exit_code == 1, (text, repr(result.exception))
        assert "Hardware-fit gate" in text
        assert "stdin is not a TTY" not in text

    def test_a_run_that_now_fits_is_asked_to_start_after_the_line_not_before(self, train_cli):
        train_cli.write(3, 12, 5)
        result, text, registered = train_cli()  # no --yes, and stdin is not a terminal
        assert result.exit_code == 3, (text, repr(result.exception))
        assert "stdin is not a TTY" in text
        assert text.index("the longest row; data.max_length is 4096") < text.index(
            "stdin is not a TTY"
        )
        assert registered == []

    def test_allow_oom_attempt_launches_as_before_with_the_old_panel(self, train_cli):
        train_cli.write(3, 3000, 5)
        result, text, registered = train_cli("--yes", "--allow-oom-attempt")
        assert isinstance(result.exception, _RegisteredError), (text, repr(result.exception))
        assert "--allow-oom-attempt set: launching anyway." in text
        assert "the longest row" not in text

    def test_a_run_that_fits_at_max_length_never_reads_the_longest_row(
        self, train_cli, monkeypatch
    ):
        train_cli.write(3, 12, 5)
        monkeypatch.setattr(
            train_mod, "get_gpu_info",
            lambda **kw: {"memory_total_bytes": int(1e12), "memory_total": "huge"},
        )

        def boom(*args, **kwargs):
            raise AssertionError("the longest row was read for a run that fits")

        monkeypatch.setattr(realised_length, "longest_row_tokens", boom)
        result, text, registered = train_cli("--yes")
        assert isinstance(result.exception, _RegisteredError), (text, repr(result.exception))
        assert "Pre-flight:" not in text
        assert len(registered) == 1

    def test_packed_runs_are_refused_at_once_with_no_data_read(self, train_cli, tmp_path):
        train_cli.write(3, 12, 5)
        text = (tmp_path / "soup.yaml").read_text(encoding="utf-8").replace(
            "epochs: 1", "epochs: 1\n  packing: true"
        )
        (tmp_path / "soup.yaml").write_text(text, encoding="utf-8")
        result, out, registered = train_cli("--yes")
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "Hardware-fit gate" in out
        assert "Loading dataset" not in out
        assert registered == []


def test_the_margin_is_the_one_the_pre_flight_always_used():
    assert VRAM_SAFETY_MARGIN == 0.10
