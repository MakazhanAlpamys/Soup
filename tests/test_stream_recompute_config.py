"""Layer streaming: the ``training.stream_recompute`` config field.

'on' is the shipped checkpoint recompute; 'off' keeps every streamed layer's
activations; 'auto' keeps them for as many of the last layers as fit, decided at
setup from one synthetic step at the configured batch x max_length (RAM tier
only). The runtime and the setup wiring are in
``tests/test_stream_recompute.py``.
"""

import pytest

from soup_cli.config.loader import load_config_from_string


def _cfg(**training):
    body = "\n".join(f"  {k}: {v}" for k, v in training.items())
    return (
        "base: HuggingFaceTB/SmolLM2-135M\n"
        "task: sft\n"
        "data:\n"
        "  train: d.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 1\n"
        "  lora:\n"
        "    r: 8\n" + body + "\n"
    )


def _schema(task, **training):
    """A config built from Python, as tests/test_stream_loss_chunk.py does; kto
    refuses a batch of one before anything else is looked at."""
    from soup_cli.config.schema import SoupConfig

    data = {"train": "tests/fixtures/sample_train.jsonl"}
    if task != "sft":
        data["format"] = "dpo"
    return SoupConfig(
        base="test/model", task=task, data=data, training={"batch_size": 2, **training}
    )


class TestAccepted:
    def test_default_is_on(self):
        cfg = load_config_from_string(_cfg(stream_layers="true"))
        assert cfg.training.stream_recompute == "on"

    def test_default_is_on_without_streaming(self):
        cfg = load_config_from_string(_cfg())
        assert cfg.training.stream_recompute == "on"

    @pytest.mark.parametrize("mode", ["on", "off", "auto"])
    def test_every_mode_parses_quoted(self, mode):
        cfg = load_config_from_string(_cfg(stream_layers="true", stream_recompute=f"'{mode}'"))
        assert cfg.training.stream_recompute == mode

    def test_auto_parses_unquoted(self):
        cfg = load_config_from_string(_cfg(stream_layers="true", stream_recompute="auto"))
        assert cfg.training.stream_recompute == "auto"

    @pytest.mark.parametrize(
        ("spelling", "mode"),
        [("on", "on"), ("off", "off"), ("true", "on"), ("false", "off"), ("no", "off")],
    )
    def test_yaml_booleans_map_to_the_mode_they_spell(self, spelling, mode):
        """YAML 1.1 reads an unquoted on/off (and yes/no/true/false) as a bool.
        For this field the bool reading agrees with the mode ('recompute: off'
        = no recompute), so it is mapped rather than refused."""
        cfg = load_config_from_string(_cfg(stream_layers="true", stream_recompute=spelling))
        assert cfg.training.stream_recompute == mode

    def test_the_explicit_default_is_accepted_without_streaming(self):
        """A default is not a decision: spelling it out must not trip the gate."""
        cfg = load_config_from_string(_cfg(stream_layers="false", stream_recompute="'on'"))
        assert cfg.training.stream_recompute == "on"


class TestRefused:
    @pytest.mark.parametrize("mode", ["'off'", "'auto'", "off"])
    def test_refused_while_streaming_is_off(self, mode):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="false", stream_recompute=mode))
        message = str(exc.value)
        assert "training.stream_recompute" in message
        assert "set but stream_layers is false" in message
        assert "set stream_layers=true" in message

    def test_refused_while_streaming_is_unset(self):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_recompute="auto"))
        assert "training.stream_recompute" in str(exc.value)
        assert "set but stream_layers is false" in str(exc.value)

    def test_the_refusal_names_only_the_field_that_is_set(self):
        """It listed all twelve stream_* fields (v0.77.0 release smoke), leaving the
        user to work out which one the config sets."""
        with pytest.raises(ValueError) as exc:
            load_config_from_string(_cfg(stream_layers="false", stream_recompute="'off'"))
        assert str(exc.value) == (
            ": Value error, training.stream_recompute set but stream_layers is false; "
            "set stream_layers=true to stream the base layer-by-layer."
        )

    def test_two_set_fields_are_both_named_in_field_order(self):
        with pytest.raises(ValueError) as exc:
            load_config_from_string(
                _cfg(stream_layers="false", stream_buffers=3, stream_recompute="auto")
            )
        assert str(exc.value) == (
            ": Value error, training.stream_buffers / training.stream_recompute set but "
            "stream_layers is false; set stream_layers=true to stream the base "
            "layer-by-layer."
        )

    @pytest.mark.parametrize("value", ["1", "0", "2"])
    def test_an_int_is_refused(self, value):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="true", stream_recompute=value))
        message = str(exc.value)
        assert "training.stream_recompute must be one of 'on', 'off', 'auto'" in message
        assert "got int" in message

    @pytest.mark.parametrize(("value", "kind"), [("[off]", "list"), ("1.5", "float")])
    def test_other_non_strings_are_refused(self, value, kind):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="true", stream_recompute=value))
        message = str(exc.value)
        assert "training.stream_recompute must be one of 'on', 'off', 'auto'" in message
        assert f"got {kind}" in message

    @pytest.mark.parametrize("value", ["'maybe'", "'ON'", "'recompute'", "''"])
    def test_an_unknown_mode_is_refused(self, value):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="true", stream_recompute=value))
        message = str(exc.value)
        assert "stream_recompute" in message
        assert "'on', 'off' or 'auto'" in message

    @pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto"])
    @pytest.mark.parametrize("mode", ["off", "auto"])
    def test_a_task_other_than_sft_is_refused(self, task, mode):
        """Only the SFT step is measured; a loss with two grad-enabled forwards
        per backward would stop at the slot-owner check mid-step."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc:
            _schema(task, stream_layers=True, stream_recompute=mode)
        message = exc.value.errors()[0]["msg"]
        assert (
            f"training.stream_recompute='{mode}' supports task='sft' only; got task='{task}'"
            in message
        )
        assert "set it to 'on'" in message

    @pytest.mark.parametrize("task", ["dpo", "kto"])
    def test_on_is_accepted_for_a_task_other_than_sft(self, task):
        cfg = _schema(task, stream_layers=True, stream_recompute="on")
        assert cfg.training.stream_recompute == "on"

    @pytest.mark.parametrize("mode", ["off", "auto"])
    def test_sft_is_accepted(self, mode):
        cfg = _schema("sft", stream_layers=True, stream_recompute=mode)
        assert cfg.training.stream_recompute == mode

    @pytest.mark.parametrize("offload", ["cpu", "disk"])
    @pytest.mark.parametrize("mode", ["off", "auto"])
    def test_activation_offloading_is_refused(self, mode, offload):
        """A retained layer's own saved-tensor hooks are the innermost, so the
        offloading hooks would never see its activations."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc:
            _schema(
                "sft", stream_layers=True, stream_recompute=mode, activation_offloading=offload
            )
        message = exc.value.errors()[0]["msg"]
        assert (
            f"training.stream_recompute='{mode}' cannot be combined with "
            f"training.activation_offloading='{offload}'" in message
        )
        assert "unset activation_offloading" in message

    def test_on_with_activation_offloading_is_accepted(self):
        cfg = _schema("sft", stream_layers=True, activation_offloading="cpu")
        assert cfg.training.stream_recompute == "on"
        assert cfg.training.activation_offloading == "cpu"

    def test_a_misspelt_key_is_refused(self):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="true", stream_recomputes="'off'"))
        assert "stream_recomputes" in str(exc.value)
