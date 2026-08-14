"""#392 — the MLX adapter_config.json shipped the UNRESOLVED target_modules.

Reported by @armanbot-jpg with a root cause and a control: hand-editing `keys`
in the saved file makes the very same `adapters.safetensors` produce the tuned
behaviour, which is what proves the weights were fine and the config was not.

`_apply_lora` resolved `target_modules: auto` into a LOCAL variable and trained
on the resolved list; the writer serialised `lora_cfg.target_modules`, i.e. the
raw `"auto"`. On load, `linear_to_lora_layers(..., {"keys": ["auto"]})` matches
no module and `load_weights(strict=False)` drops every LoRA tensor without a
word — so the adapter is a no-op and generation is bit-identical to the base.

`"auto"` is the schema DEFAULT, so this was every MLX run that did not name its
modules by hand. Same failure class as v0.72.0's `.inner.` keys and #362's
full-fine-tune adapter, and the third time in this project a healthy loss curve
has shipped a dead artifact: the loss curve cannot see it, only the artifact can.

The repair is one resolver both callers share, rather than the writer repeating
the resolution — two copies of "which modules did we actually train?" is how
they drifted in the first place.
"""

import pytest


def _lora(target_modules):
    class _L:
        r = 16
        alpha = 32
        dropout = 0.0

    _L.target_modules = target_modules
    return _L()


class TestTheResolverIsOneAnswer:
    @pytest.mark.parametrize("raw", ["auto", ["auto"], None, []])
    def test_auto_and_empty_resolve_to_the_trained_default(self, raw):
        from soup_cli.trainer.mlx_sft import resolve_mlx_target_keys

        assert resolve_mlx_target_keys(_lora(raw)) == [
            "self_attn.q_proj",
            "self_attn.v_proj",
        ]

    def test_an_explicit_list_is_passed_through(self):
        from soup_cli.trainer.mlx_sft import resolve_mlx_target_keys

        assert resolve_mlx_target_keys(_lora(["self_attn.k_proj"])) == ["self_attn.k_proj"]

    def test_an_explicit_string_becomes_a_one_item_list(self):
        from soup_cli.trainer.mlx_sft import resolve_mlx_target_keys

        assert resolve_mlx_target_keys(_lora("mlp.down_proj")) == ["mlp.down_proj"]

    def test_the_resolved_value_never_contains_auto(self):
        """The whole defect in one assertion: whatever comes back is a module
        list mlx_lm can match, never the placeholder."""
        from soup_cli.trainer.mlx_sft import resolve_mlx_target_keys

        for raw in ("auto", ["auto"], None, []):
            assert "auto" not in resolve_mlx_target_keys(_lora(raw))


class TestTheWrittenConfigMatchesWhatWasTrained:
    def test_the_default_config_does_not_ship_auto_as_a_key(self):
        """CONTROL for the reported failure: with the schema default
        (`target_modules: auto`) the file must carry the real modules, because
        `{"keys": ["auto"]}` matches nothing and silently drops every tensor."""
        from soup_cli.trainer.mlx_sft import build_mlx_adapter_config

        written = build_mlx_adapter_config(_lora("auto"), adapter_path="/tmp/out")
        keys = written["lora_parameters"]["keys"]
        assert keys == ["self_attn.q_proj", "self_attn.v_proj"]
        assert "auto" not in keys

    def test_the_written_keys_are_exactly_what_apply_lora_would_train(self):
        """The two must not be able to disagree — that disagreement IS #392."""
        from soup_cli.trainer.mlx_sft import build_mlx_adapter_config, resolve_mlx_target_keys

        for raw in ("auto", ["auto"], ["self_attn.k_proj"], "mlp.down_proj", None):
            cfg = _lora(raw)
            assert (
                build_mlx_adapter_config(cfg, adapter_path="/tmp/out")["lora_parameters"]["keys"]
                == resolve_mlx_target_keys(cfg)
            )

    def test_rank_scale_and_dropout_still_come_from_the_config(self):
        """CONTROL. A repair that hardcoded the whole block would also make the
        assertions above pass."""
        from soup_cli.trainer.mlx_sft import build_mlx_adapter_config

        params = build_mlx_adapter_config(_lora("auto"), adapter_path="/tmp/out")[
            "lora_parameters"
        ]
        assert params["rank"] == 16
        assert params["scale"] == pytest.approx(32 / 16)
        assert params["dropout"] == 0.0
