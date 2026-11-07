"""The DeepSpeed config's precision matches the trainer's (#2266).

Every preset declares ``"bf16": {"enabled": True}``. When the trainer runs fp16 (a
card without bf16 hardware, ``grpo_fp16``), accelerate refused the pair while the
trainer was being built: "`--mixed_precision` arg cannot be set to `fp16` when
`bf16` is set in the DeepSpeed config file". ``resolve_deepspeed_config`` now takes
the trainer's precision and makes the file agree with it, and ``soup train`` passes
it in. The card check is stubbed throughout, so this runs on a CPU runner.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

import soup_cli.utils.deepspeed as ds
from soup_cli.utils.deepspeed import (
    CONFIGS,
    get_deepspeed_config,
    resolve_deepspeed_config,
    resolve_user_deepspeed_file,
    trainer_precision,
    write_deepspeed_config,
)

PRESETS = sorted(CONFIGS)
REFUSAL = "cannot be set to `fp16` when `bf16` is set in the DeepSpeed config file"

# Rich wraps and colours console.print() output in CI even when capsys captures it
# (width defaults narrower than a developer's terminal, and colour can be forced by
# the runner's env). A substring check on the raw text then depends on where a word
# happened to land relative to the wrap, same issue test_auto_tuning.py works around.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escapes and collapse whitespace, so a wrapped line still matches."""
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", text))


def _cfg(task="sft", grpo_fp16=False, auto_mixed_precision=False):
    return SimpleNamespace(
        task=task,
        training=SimpleNamespace(grpo_fp16=grpo_fp16, auto_mixed_precision=auto_mixed_precision),
    )


@pytest.fixture
def card(monkeypatch):
    """``card("t4")`` / ``card("ampere")``: stub the check every trainer uses."""

    def set_card(kind):
        flags = {"t4": (False, True), "ampere": (True, False)}[kind]
        monkeypatch.setattr("soup_cli.utils.gpu.bf16_fp16_flags", lambda device, **_: flags)

    return set_card


def _plugin(config_or_path):
    accelerate_utils = pytest.importorskip("accelerate.utils")
    return accelerate_utils.DeepSpeedPlugin(hf_ds_config=config_or_path)


class TestTrainerPrecision:
    def test_a_card_without_bf16_runs_fp16(self, card):
        card("t4")
        assert trainer_precision(_cfg(), "cuda") == "fp16"

    def test_a_card_with_bf16_runs_bf16(self, card):
        card("ampere")
        assert trainer_precision(_cfg(), "cuda:0") == "bf16"

    def test_grpo_fp16_is_fp16_even_on_a_bf16_card(self, card):
        card("ampere")
        assert trainer_precision(_cfg(task="grpo", grpo_fp16=True), "cuda") == "fp16"

    def test_auto_mixed_precision_leaves_it_to_the_trainer(self, card):
        card("ampere")
        assert trainer_precision(_cfg(auto_mixed_precision=True), "cuda") == "auto"
        assert trainer_precision(_cfg(task="tts", auto_mixed_precision=True), "cuda") == "auto"

    def test_off_cuda_there_is_nothing_to_match(self, card):
        card("ampere")
        assert trainer_precision(_cfg(), "cpu") is None
        assert trainer_precision(_cfg(), "mps") is None


class TestPresetsMatchTheTrainer:
    @pytest.mark.parametrize("name", PRESETS)
    def test_without_bf16_hardware_fp16_is_on_and_bf16_is_not(self, name):
        cfg, notes = resolve_deepspeed_config(
            get_deepspeed_config(name), gpu_count=1, precision="fp16"
        )
        assert cfg["fp16"]["enabled"] is True
        assert cfg["bf16"]["enabled"] is False
        assert any("#2266" in note and "fp16" in note for note in notes)

    @pytest.mark.parametrize("name", PRESETS)
    def test_with_bf16_hardware_it_keeps_what_it_has_today(self, name):
        today, today_notes = resolve_deepspeed_config(get_deepspeed_config(name), gpu_count=1)
        cfg, notes = resolve_deepspeed_config(
            get_deepspeed_config(name), gpu_count=1, precision="bf16"
        )
        assert cfg == today
        assert notes == today_notes
        assert cfg["bf16"]["enabled"] is True

    @pytest.mark.parametrize("name", PRESETS)
    def test_auto_hands_both_to_the_trainer(self, name):
        cfg, notes = resolve_deepspeed_config(
            get_deepspeed_config(name), gpu_count=1, precision="auto"
        )
        assert cfg["bf16"]["enabled"] == "auto"
        assert "fp16" not in cfg or cfg["fp16"]["enabled"] == "auto"
        assert any("auto_mixed_precision" in note for note in notes)

    def test_without_a_precision_the_preset_is_left_alone(self):
        cfg, _ = resolve_deepspeed_config(get_deepspeed_config("zero2"), gpu_count=1)
        assert cfg == get_deepspeed_config("zero2")

    def test_an_unknown_precision_is_refused(self):
        with pytest.raises(ValueError, match="precision"):
            resolve_deepspeed_config(get_deepspeed_config("zero2"), precision="fp8")


class TestAccelerateAcceptsTheWrittenFile:
    """accelerate's own check (``DeepSpeedPlugin.set_mixed_precision``), on the file
    Soup writes. It needs neither DeepSpeed nor a GPU."""

    @pytest.mark.parametrize("name", PRESETS)
    @pytest.mark.parametrize("precision", ["fp16", "bf16"])
    def test_the_trainers_precision_passes(self, name, precision):
        plugin = _plugin(write_deepspeed_config(name, gpu_count=1, precision=precision))
        plugin.set_mixed_precision(precision)
        config = plugin.deepspeed_config
        assert config["fp16"]["enabled"] is (precision == "fp16")
        assert config["bf16"]["enabled"] is (precision == "bf16")

    @pytest.mark.parametrize("precision", ["fp16", "bf16"])
    def test_an_auto_file_passes_whatever_the_trainer_picks(self, precision):
        _plugin(write_deepspeed_config("zero2", gpu_count=1, precision="auto")) \
            .set_mixed_precision(precision)

    def test_control_the_unmatched_preset_refuses_an_fp16_trainer(self):
        """What #2266 hit on a T4: the preset as written today, an fp16 trainer."""
        with pytest.raises(ValueError, match=REFUSAL):
            _plugin(write_deepspeed_config("zero2", gpu_count=1)).set_mixed_precision("fp16")


class TestTheNoteIsPrinted:
    def test_a_precision_change_is_printed(self, capsys):
        write_deepspeed_config("zero2", gpu_count=1, precision="fp16")
        out = _plain(capsys.readouterr().out)
        assert "bf16 true -> false" in out and "fp16 unset -> true" in out

    def test_no_change_prints_nothing(self, capsys):
        write_deepspeed_config("zero2", gpu_count=1, precision="bf16")
        assert capsys.readouterr().out == ""


class TestZeroPlusPlusQuantiser:
    def test_an_fp16_run_keeps_the_quantiser(self):
        cfg, notes = resolve_deepspeed_config(
            get_deepspeed_config("zero++"), gpu_count=4, precision="fp16"
        )
        zero = cfg["zero_optimization"]
        assert zero["zero_quantized_weights"] is True
        assert zero["zero_quantized_gradients"] is True
        assert not any("quantis" in note for note in notes)

    def test_a_bf16_run_turns_it_off_with_the_note(self):
        cfg, notes = resolve_deepspeed_config(
            get_deepspeed_config("zero++"), gpu_count=4, precision="bf16"
        )
        assert cfg["zero_optimization"]["zero_quantized_weights"] is False
        assert any("this run is bf16" in note for note in notes)

    def test_an_auto_run_turns_it_off_because_it_may_be_bf16(self):
        cfg, notes = resolve_deepspeed_config(
            get_deepspeed_config("zero++"), gpu_count=4, precision="auto"
        )
        assert cfg["zero_optimization"]["zero_quantized_weights"] is False
        assert any("may be bf16" in note for note in notes)

    def test_a_user_file_set_to_auto_keeps_the_quantiser_off_even_on_a_known_fp16_run(self):
        """Pinned per review: a user file that names `bf16: auto, fp16: auto` passes
        `_match_precision` unchanged (an `"auto"` entry already agrees with any
        trainer precision, #2266), so the quantiser decision still reads the FILE's
        `bf16: "auto"`, not the `precision="fp16"` the caller knows. Intended: the
        file as written genuinely does not rule out DeepSpeed resolving to bf16 (it
        decides from the trainer's `mixed_precision` at Trainer init, which this
        resolver does not control), so "may be bf16" stays the honest answer. A file
        that wants the quantiser on an fp16 run should say `fp16: true` explicitly,
        which matches precision and leaves the quantiser alone (see
        TestAUsersOwnFile)."""
        config = {
            "bf16": {"enabled": "auto"}, "fp16": {"enabled": "auto"},
            "zero_optimization": {
                "stage": 3, "zero_quantized_weights": True, "zero_quantized_gradients": True,
            },
        }
        cfg, notes = resolve_deepspeed_config(config, gpu_count=4, precision="fp16")
        assert cfg["bf16"]["enabled"] == "auto"
        assert cfg["fp16"]["enabled"] == "auto"
        assert cfg["zero_optimization"]["zero_quantized_weights"] is False
        assert any("may be bf16" in note for note in notes)


class TestUnsupportedShapesArePassedThrough:
    """Two shapes the resolver leaves alone rather than repairs (#2266 review)."""

    def test_a_file_with_neither_key_is_not_touched(self):
        """An absent key does not enable the wrong dtype, so it needs no repair:
        accelerate fills both bf16 and fp16 in from the trainer's own flags when it
        builds the DeepSpeedPlugin, if the file names neither (#359's promise that an
        untouched config reaches DeepSpeed byte-identical)."""
        config = {"zero_optimization": {"stage": 2}}
        cfg, notes = resolve_deepspeed_config(config, gpu_count=1, precision="fp16")
        assert cfg == config
        assert notes == []

    def test_a_non_mapping_value_does_not_raise_and_is_left_alone(self):
        """{"fp16": true} instead of {"fp16": {"enabled": true}}: Soup cannot read what
        `enabled` would mean there, and spreading it ({**True}) used to raise
        TypeError. Passed through for DeepSpeed to accept or reject on its own terms."""
        for config in ({"fp16": True}, {"bf16": True}):
            for precision in ("fp16", "bf16", "auto"):
                cfg, notes = resolve_deepspeed_config(
                    dict(config), gpu_count=1, precision=precision
                )
                assert cfg == config, (config, precision)
                assert notes == []

    def test_one_key_a_real_dict_the_other_a_non_mapping(self):
        """The non-mapping key is left alone even when its sibling is a genuine
        conflict that does get repaired."""
        config = {"bf16": {"enabled": True}, "fp16": True}
        cfg, notes = resolve_deepspeed_config(config, gpu_count=1, precision="fp16")
        assert cfg["bf16"]["enabled"] is False
        assert cfg["fp16"] is True  # untouched, not turned into a dict
        assert any("bf16" in note for note in notes)

    def test_a_user_file_with_neither_key_resolves_to_the_same_path(self, tmp_path):
        path = tmp_path / "ds.json"
        path.write_text(json.dumps({"zero_optimization": {"stage": 2}}), encoding="utf-8")
        assert resolve_user_deepspeed_file(str(path), gpu_count=1, precision="fp16") == str(path)
        assert resolve_user_deepspeed_file(str(path), gpu_count=1, precision="bf16") == str(path)


class TestEnabledReadsTheStringShape:
    """`{"enabled": "true"}` / `"false"` -- DeepSpeed's own JSON is case- and
    quote-insensitive about these strings in practice; `_enabled` reads them the
    same as the literal booleans. The surviving mutation from the first review
    (`return lowered == "true"` -> `return False`) is caught here."""

    def test_the_string_true_is_read_as_true_and_gets_repaired(self):
        config = {"bf16": {"enabled": "true"}}
        cfg, notes = resolve_deepspeed_config(config, gpu_count=1, precision="fp16")
        assert cfg["bf16"]["enabled"] is False
        assert any("bf16 true -> false" in note for note in notes)

    def test_the_string_false_is_read_as_false_and_is_left_alone(self):
        config = {"bf16": {"enabled": "false"}, "fp16": {"enabled": True}}
        cfg, notes = resolve_deepspeed_config(config, gpu_count=1, precision="fp16")
        assert cfg == config
        assert notes == []


class TestAUsersOwnFile:
    def _write(self, tmp_path, config):
        path = tmp_path / "ds.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        "precision, section",
        [
            ("bf16", {"bf16": {"enabled": True}}),
            ("fp16", {"fp16": {"enabled": True}, "bf16": {"enabled": False}}),
            ("fp16", {"bf16": {"enabled": "auto"}, "fp16": {"enabled": "auto"}}),
        ],
        ids=["bf16-on-bf16", "fp16-on-fp16", "auto"],
    )
    def test_a_matching_file_comes_back_by_the_same_path(self, tmp_path, precision, section):
        path = self._write(tmp_path, {"zero_optimization": {"stage": 2}, **section})
        assert resolve_user_deepspeed_file(str(path), gpu_count=1, precision=precision) == str(path)

    def test_a_conflicting_file_is_repaired_into_a_copy(self, tmp_path, capsys):
        path = self._write(tmp_path, {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}})
        before = path.read_bytes()
        out_path = resolve_user_deepspeed_file(str(path), gpu_count=1, precision="fp16")
        assert out_path != str(path)
        repaired = json.loads(open(out_path, encoding="utf-8").read())
        assert repaired["bf16"]["enabled"] is False
        assert repaired["fp16"]["enabled"] is True
        assert path.read_bytes() == before
        assert "#2266" in _plain(capsys.readouterr().out)


class TestSoupTrainPassesThePrecision:
    """The wiring: ``soup train --deepspeed`` on a card without bf16 asks for fp16,
    and a bad name still fails before any hardware work."""

    def _project(self, tmp_path):
        (tmp_path / "train.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "hi"},
                                     {"role": "assistant", "content": "hello"}]}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "soup.yaml").write_text(
            "base: HuggingFaceTB/SmolLM2-135M-Instruct\ntask: sft\n"
            "data:\n  train: ./train.jsonl\n  max_length: 64\n"
            "training:\n  epochs: 1\n  batch_size: 1\noutput: ./out\n",
            encoding="utf-8",
        )

    def test_a_t4_run_resolves_the_preset_for_fp16(self, tmp_path, monkeypatch, card):
        from typer.testing import CliRunner

        import soup_cli.commands.train as train_cmd
        from soup_cli.cli import app

        self._project(tmp_path)
        monkeypatch.chdir(tmp_path)
        card("t4")
        monkeypatch.setattr(train_cmd, "detect_device", lambda backend=None: ("cuda", "Tesla T4"))
        monkeypatch.setattr(train_cmd, "get_gpu_info", lambda backend=None: {})
        seen = {}

        class ResolvedError(Exception):
            pass

        def spy(deepspeed, *, precision=None):
            seen.update(deepspeed=deepspeed, precision=precision)
            raise ResolvedError

        monkeypatch.setattr(train_cmd, "_resolve_deepspeed", spy)
        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--deepspeed", "zero2", "--yes"]
        )
        assert isinstance(result.exception, ResolvedError), result.output
        assert seen == {"deepspeed": "zero2", "precision": "fp16"}

    def test_a_bad_name_fails_before_the_device_is_detected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import soup_cli.commands.train as train_cmd
        from soup_cli.cli import app

        self._project(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            train_cmd, "detect_device", lambda backend=None: pytest.fail("device detected")
        )
        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--deepspeed", "zero9", "--yes"]
        )
        assert result.exit_code == 1
        assert "Invalid DeepSpeed config" in result.output


def test_the_precision_values_are_the_documented_three():
    assert ds.PRECISIONS == ("bf16", "fp16", "auto")
