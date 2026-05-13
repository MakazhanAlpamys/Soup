"""Tests for v0.53.4 — Long Context + Architecture.

Covers:
  * #11   — CUDA OOM friendly message extension (batch_size / grad_accum hint)
  * #122  — ``is_flash_attn_v3_available`` + SoupConfig LongLoRA+FA3 reject
  * #120  — ``is_mistral_model`` / ``is_qwen_model`` / ``is_phi_model`` and
            ``validate_longlora_compat`` acceptance of the new families
  * #121  — ``apply_long_context_config`` auto-detects ``llama3`` when
            ``rope_scaling_type is None`` and the model config carries a
            Llama 3.1 rope block
  * #83   — ``expand_model_blocks`` live implementation + ``apply_llama_pro_freeze``
"""

from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from rich.console import Console

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.errors import format_friendly_error

_BASE_DATA_LINE = "data:\n  train: ./train.jsonl\n"


def _load(yaml: str):
    if "data:" not in yaml:
        yaml = yaml + "\n" + _BASE_DATA_LINE
    return load_config_from_string(yaml)


# ---------------------------------------------------------------------------
# #11 — CUDA OOM friendly message
# ---------------------------------------------------------------------------


class TestOomFriendlyMessage:
    def test_cuda_oom_names_batch_size_and_grad_accum(self):
        buf = StringIO()
        test_console = Console(file=buf, stderr=False)
        with patch("soup_cli.utils.errors.console", test_console):
            format_friendly_error(
                RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
                verbose=False,
            )
        output = buf.getvalue()
        assert "--batch-size" in output
        assert "--grad-accum" in output
        assert "4bit" in output  # legacy hint still present

    def test_out_of_memory_error_type_also_hinted(self):
        buf = StringIO()
        test_console = Console(file=buf, stderr=False)

        class OutOfMemoryError(RuntimeError):
            pass

        with patch("soup_cli.utils.errors.console", test_console):
            format_friendly_error(OutOfMemoryError("oom"), verbose=False)
        out = buf.getvalue()
        assert "--batch-size" in out
        assert "--grad-accum" in out


# ---------------------------------------------------------------------------
# #122 — FlashAttention v3 detection + cross-validator
# ---------------------------------------------------------------------------


class TestFlashAttnV3Available:
    def test_returns_false_when_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flash_attn", None)
        from soup_cli.utils.flash_attn import is_flash_attn_v3_available

        # ``None`` in sys.modules makes ``import flash_attn`` raise ImportError.
        assert is_flash_attn_v3_available() is False

    def test_returns_true_for_v3(self, monkeypatch):
        fake = SimpleNamespace(__version__="3.0.0")
        monkeypatch.setitem(sys.modules, "flash_attn", fake)
        from soup_cli.utils.flash_attn import is_flash_attn_v3_available

        assert is_flash_attn_v3_available() is True

    def test_returns_false_for_v2(self, monkeypatch):
        fake = SimpleNamespace(__version__="2.5.7")
        monkeypatch.setitem(sys.modules, "flash_attn", fake)
        from soup_cli.utils.flash_attn import is_flash_attn_v3_available

        assert is_flash_attn_v3_available() is False

    def test_returns_false_on_unparseable_version(self, monkeypatch):
        fake = SimpleNamespace(__version__="abc")
        monkeypatch.setitem(sys.modules, "flash_attn", fake)
        from soup_cli.utils.flash_attn import is_flash_attn_v3_available

        assert is_flash_attn_v3_available() is False

    def test_returns_false_on_non_string_version(self, monkeypatch):
        fake = SimpleNamespace(__version__=3)  # not a string
        monkeypatch.setitem(sys.modules, "flash_attn", fake)
        from soup_cli.utils.flash_attn import is_flash_attn_v3_available

        assert is_flash_attn_v3_available() is False


class TestLongLoraRejectsFlashAttnV3:
    def test_schema_rejects_longlora_when_fa3_present(self, monkeypatch):
        # Force FA3 detected -> SoupConfig schema gate must reject.
        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available",
            lambda: True,
        )
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  use_longlora: true
"""
        with pytest.raises(
            (ValidationError, ValueError), match="FlashAttention v3"
        ):
            _load(yaml_in)

    def test_schema_passes_when_fa3_absent(self, monkeypatch):
        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available",
            lambda: False,
        )
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  use_longlora: true
"""
        cfg = _load(yaml_in)
        assert cfg.training.use_longlora is True


# ---------------------------------------------------------------------------
# #120 — LongLoRA arch allowlist expansion
# ---------------------------------------------------------------------------


class TestIsMistralModel:
    def test_basic(self):
        from soup_cli.utils.longlora import is_mistral_model

        assert is_mistral_model("mistralai/Mistral-7B-v0.1") is True
        assert is_mistral_model("mistralai/Mixtral-8x7B-v0.1") is False  # Mixtral != Mistral name
        assert is_mistral_model("mistralai/Mistral-Nemo-Base-2407") is True
        assert is_mistral_model("Mistral-7B-Instruct-v0.3") is True

    def test_word_boundary_rejects_substring(self):
        from soup_cli.utils.longlora import is_mistral_model

        # Substring inside an unrelated identifier must NOT match.
        assert is_mistral_model("my-mistralish-finetune") is False
        # ``unmistral`` should also reject — char before ``mistral`` is alphanumeric.
        assert is_mistral_model("unmistral-7b") is False

    def test_input_guards(self):
        from soup_cli.utils.longlora import is_mistral_model

        assert is_mistral_model("") is False
        with pytest.raises(TypeError):
            is_mistral_model(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            is_mistral_model("abc\x00def")
        # Oversized → returns False rather than raising.
        assert is_mistral_model("a" * 1024) is False


class TestIsQwenModel:
    def test_basic(self):
        from soup_cli.utils.longlora import is_qwen_model

        assert is_qwen_model("Qwen/Qwen2-7B") is True
        assert is_qwen_model("Qwen/Qwen2.5-7B-Instruct") is True
        assert is_qwen_model("Qwen/Qwen3-7B") is True
        # Reject substring.
        assert is_qwen_model("my-qwenish") is False
        # Not a Qwen.
        assert is_qwen_model("meta-llama/Llama-3.1-8B") is False


class TestIsPhiModel:
    def test_basic(self):
        from soup_cli.utils.longlora import is_phi_model

        assert is_phi_model("microsoft/Phi-3-mini") is True
        assert is_phi_model("microsoft/phi-4") is True
        # Reject substring.
        assert is_phi_model("alphabetagamma") is False
        assert is_phi_model("amphiphilic") is False
        # Not a Phi.
        assert is_phi_model("meta-llama/Llama-3") is False


class TestIsSupportedLongloraArch:
    def test_all_families_accepted(self):
        from soup_cli.utils.longlora import is_supported_longlora_arch

        for name in (
            "meta-llama/Llama-3.1-8B",
            "codellama/CodeLlama-7B",
            "mistralai/Mistral-7B-v0.1",
            "Qwen/Qwen2.5-7B",
            "microsoft/Phi-3-mini",
        ):
            assert is_supported_longlora_arch(name) is True, name

    def test_unsupported_arch_rejected(self):
        from soup_cli.utils.longlora import is_supported_longlora_arch

        assert is_supported_longlora_arch("google/gemma-2-9b") is False
        assert is_supported_longlora_arch("databricks/dbrx-base") is False


class TestLongloraSchemaAcceptsNewArches:
    """v0.53.4 #120 — Mistral / Qwen / Phi base models must now be accepted
    by the LongLoRA schema gate (formerly Llama-only)."""

    @pytest.mark.parametrize(
        "base",
        [
            "mistralai/Mistral-7B-v0.1",
            "Qwen/Qwen2.5-7B",
            "microsoft/Phi-3-mini",
        ],
    )
    def test_accepts(self, monkeypatch, base):
        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available",
            lambda: False,
        )
        yaml_in = f"""
base: {base}
task: sft
training:
  use_longlora: true
"""
        cfg = _load(yaml_in)
        assert cfg.training.use_longlora is True


# ---------------------------------------------------------------------------
# #121 — apply_long_context_config llama3 auto-detect
# ---------------------------------------------------------------------------


class TestApplyLongContextLlama3Autodetect:
    def test_autodetect_picks_llama3_when_rope_block_present(self):
        from soup_cli.utils.long_context import apply_long_context_config

        model_config = SimpleNamespace(
            max_position_embeddings=8192,
            rope_scaling={
                "type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
        )
        result = apply_long_context_config(
            model_config,
            target_length=32768,
            rope_scaling_type=None,  # auto-detect
            model_name="meta-llama/Llama-3.1-8B",
        )
        assert result is not None
        assert result["type"] == "llama3"
        assert model_config.max_position_embeddings == 32768

    def test_autodetect_falls_back_to_dynamic(self):
        from soup_cli.utils.long_context import apply_long_context_config

        # No rope_scaling on config → fall back to ``dynamic``.
        model_config = SimpleNamespace(max_position_embeddings=4096)
        result = apply_long_context_config(
            model_config,
            target_length=32768,
            rope_scaling_type=None,
            model_name="mistralai/Mistral-7B-v0.1",
        )
        assert result is not None
        assert result["type"] == "dynamic"

    def test_explicit_caller_pick_wins_over_autodetect(self):
        from soup_cli.utils.long_context import apply_long_context_config

        model_config = SimpleNamespace(
            max_position_embeddings=8192,
            rope_scaling={"type": "llama3", "factor": 8.0,
                          "low_freq_factor": 1.0, "high_freq_factor": 4.0},
        )
        # Caller explicitly asks for linear — must not silently switch to llama3.
        result = apply_long_context_config(
            model_config,
            target_length=16384,
            rope_scaling_type="linear",
            model_name="meta-llama/Llama-3.1-8B",
        )
        assert result is not None
        assert result["type"] == "linear"

    def test_rope_scaling_with_rope_type_alias(self):
        """v0.49.0 detect helper also accepts the newer ``rope_type`` key."""
        from soup_cli.utils.long_context import apply_long_context_config

        model_config = SimpleNamespace(
            max_position_embeddings=8192,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
        )
        result = apply_long_context_config(
            model_config,
            target_length=32768,
            rope_scaling_type=None,
            model_name="meta-llama/Llama-3.1-8B",
        )
        assert result is not None
        assert result["type"] == "llama3"


# ---------------------------------------------------------------------------
# #83 — expand_model_blocks live + apply_llama_pro_freeze
# ---------------------------------------------------------------------------


class _TensorStub:
    def __init__(self, values):
        self._values = list(values)

    def zero_(self):
        self._values = [0.0 for _ in self._values]


class _ParamStub:
    """Minimal nn.Parameter stand-in (has data / requires_grad / numel)."""

    def __init__(self, value: float = 1.0):
        self.data = _TensorStub([value])
        self.requires_grad = True

    def numel(self) -> int:
        return len(self.data._values)


class FakeLinear:
    """Minimal stand-in for nn.Linear used by zero-init smoke tests."""

    def __init__(self, value: float = 1.0):
        self.weight = _ParamStub(value)
        self.bias = _ParamStub(value)


class FakeBlock:
    """Minimal HF-causal-LM decoder block stand-in."""

    def __init__(self):
        self.mlp = SimpleNamespace(down_proj=FakeLinear(1.0))
        self.self_attn = SimpleNamespace(o_proj=FakeLinear(1.0))

    def parameters(self):
        yield self.mlp.down_proj.weight
        yield self.mlp.down_proj.bias
        yield self.self_attn.o_proj.weight
        yield self.self_attn.o_proj.bias


class FakeLayerList(list):
    """List subclass so ``deepcopy`` of layers still mutates via ``.append``."""


class FakeModel:
    def __init__(self, n_layers: int = 4):
        layers = FakeLayerList(FakeBlock() for _ in range(n_layers))
        self.model = SimpleNamespace(layers=layers)
        self.config = SimpleNamespace(num_hidden_layers=n_layers)
        self._top_param = _ParamStub(1.0)

    def parameters(self):
        for blk in self.model.layers:
            yield from blk.parameters()
        yield self._top_param


class TestExpandModelBlocks:
    def test_zero_short_circuit(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        m = FakeModel(n_layers=4)
        assert expand_model_blocks(m, 0) == 4
        assert len(m.model.layers) == 4  # unchanged

    def test_none_short_circuit(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        m = FakeModel(n_layers=4)
        assert expand_model_blocks(m, None) == 4

    def test_appends_n_blocks_and_updates_config(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        m = FakeModel(n_layers=4)
        total = expand_model_blocks(m, 2)
        assert total == 6
        assert len(m.model.layers) == 6
        assert m.config.num_hidden_layers == 6

    def test_zero_inits_residual_projections(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        m = FakeModel(n_layers=2)
        expand_model_blocks(m, 1)
        new_block = m.model.layers[-1]
        # ``zero_()`` ran on both projections.
        assert all(v == 0.0 for v in new_block.mlp.down_proj.weight.data._values)
        assert all(v == 0.0 for v in new_block.self_attn.o_proj.weight.data._values)
        # The OLD block must still be intact (deepcopy isolated).
        old_block = m.model.layers[0]
        assert any(v != 0.0 for v in old_block.mlp.down_proj.weight.data._values)

    def test_over_expansion_clamps_to_available_blocks(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        m = FakeModel(n_layers=2)
        # Request 5 new blocks on a 2-layer model — should clone only 2.
        total = expand_model_blocks(m, 5)
        assert total == 4  # 2 original + 2 clamped clones

    def test_missing_layers_raises(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        bad = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=0))
        with pytest.raises(ValueError, match="decoder layers"):
            expand_model_blocks(bad, 2)

    def test_validates_input(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        m = FakeModel(n_layers=2)
        with pytest.raises(ValueError):
            expand_model_blocks(m, -1)
        with pytest.raises(ValueError):
            expand_model_blocks(m, True)  # bool-as-int rejected


class TestApplyLlamaProFreezeNegativeInputs:
    """Review-fix: cover the input-validation surface explicitly."""

    def test_rejects_negative_block_count(self):
        from soup_cli.utils.block_expansion import apply_llama_pro_freeze

        m = FakeModel(n_layers=3)
        with pytest.raises(ValueError):
            apply_llama_pro_freeze(m, -1)

    def test_rejects_bool_block_count(self):
        from soup_cli.utils.block_expansion import apply_llama_pro_freeze

        m = FakeModel(n_layers=3)
        with pytest.raises(ValueError):
            apply_llama_pro_freeze(m, True)

    def test_silent_zero_when_layers_missing(self):
        from soup_cli.utils.block_expansion import apply_llama_pro_freeze

        bad = SimpleNamespace()
        # No ``model.layers`` discoverable → returns 0 without raising.
        assert apply_llama_pro_freeze(bad, 2) == 0


class TestExpandSharedHelper:
    """v0.53.4 review fix — centralised ``apply_block_expansion_if_configured``."""

    def test_no_op_when_expand_layers_unset(self):
        from soup_cli.utils.block_expansion import (
            apply_block_expansion_if_configured,
        )

        m = FakeModel(n_layers=4)
        tcfg = SimpleNamespace(expand_layers=None, freeze_trainable_layers=None)
        result = apply_block_expansion_if_configured(m, tcfg, console=None)
        assert result == 4
        assert len(m.model.layers) == 4  # unchanged

    def test_expands_and_optionally_freezes(self):
        from soup_cli.utils.block_expansion import (
            apply_block_expansion_if_configured,
        )

        m = FakeModel(n_layers=3)
        tcfg = SimpleNamespace(expand_layers=2, freeze_trainable_layers=2)
        result = apply_block_expansion_if_configured(m, tcfg, console=None)
        assert result == 5
        # Old blocks frozen.
        for blk in m.model.layers[:3]:
            for p in blk.parameters():
                assert p.requires_grad is False
        # New blocks trainable.
        for blk in m.model.layers[3:]:
            for p in blk.parameters():
                assert p.requires_grad is True

    def test_freeze_skipped_when_freeze_field_none(self):
        from soup_cli.utils.block_expansion import (
            apply_block_expansion_if_configured,
        )

        m = FakeModel(n_layers=3)
        tcfg = SimpleNamespace(expand_layers=1, freeze_trainable_layers=None)
        apply_block_expansion_if_configured(m, tcfg, console=None)
        # No freeze applied — original blocks still trainable.
        for blk in m.model.layers[:3]:
            for p in blk.parameters():
                assert p.requires_grad is True


class TestZeroInitWarningOnUnknownArch:
    """v0.53.4 security review LOW — warn when residual proj not found."""

    def test_warning_emitted_when_no_projection(self):
        import warnings

        from soup_cli.utils.block_expansion import expand_model_blocks

        # Block without standard mlp.down_proj / self_attn.o_proj attributes.
        class StubBlock:
            def __init__(self):
                self.something_else = 1

            def parameters(self):
                return iter(())

        class StubLayers(list):
            pass

        class StubModel:
            def __init__(self):
                self.model = SimpleNamespace(
                    layers=StubLayers([StubBlock(), StubBlock()])
                )
                self.config = SimpleNamespace(num_hidden_layers=2)

            def parameters(self):
                return iter(())

        m = StubModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expand_model_blocks(m, 1)
            assert any(
                "could not locate standard residual projections" in str(w.message)
                for w in caught
            )


class TestQwenPhiInputGuards:
    """Review-fix coverage parity with TestIsMistralModel.test_input_guards."""

    def test_qwen_null_byte_rejected(self):
        from soup_cli.utils.longlora import is_qwen_model

        with pytest.raises(ValueError):
            is_qwen_model("Qwen/\x00Qwen2")

    def test_phi_null_byte_rejected(self):
        from soup_cli.utils.longlora import is_phi_model

        with pytest.raises(ValueError):
            is_phi_model("microsoft/\x00phi-3")

    def test_qwen_oversize_returns_false(self):
        from soup_cli.utils.longlora import is_qwen_model

        assert is_qwen_model("Q" * 1024) is False

    def test_phi_none_typeerror(self):
        from soup_cli.utils.longlora import is_phi_model

        with pytest.raises(TypeError):
            is_phi_model(None)  # type: ignore[arg-type]

    def test_supported_arch_returns_false_on_non_string(self):
        from soup_cli.utils.longlora import is_supported_longlora_arch

        # Defensive surface — never raises on bad input.
        assert is_supported_longlora_arch(None) is False
        assert is_supported_longlora_arch(123) is False
        assert is_supported_longlora_arch(True) is False


class TestValidateLongloraCompatInputGuards:
    """Review-fix: ensure task / backend null-byte rejection."""

    def test_null_byte_in_task_rejected(self):
        from soup_cli.utils.longlora import validate_longlora_compat

        with pytest.raises(ValueError, match="null byte"):
            validate_longlora_compat(
                model_name="meta-llama/Llama-3.1-8B",
                task="sft\x00",
                backend="transformers",
                use_ring_attention=False,
            )

    def test_null_byte_in_backend_rejected(self):
        from soup_cli.utils.longlora import validate_longlora_compat

        with pytest.raises(ValueError, match="null byte"):
            validate_longlora_compat(
                model_name="meta-llama/Llama-3.1-8B",
                task="sft",
                backend="transformers\x00",
                use_ring_attention=False,
            )

    def test_oversize_model_name_truncated_in_error(self):
        from soup_cli.utils.longlora import validate_longlora_compat

        long_name = "google/gemma-2-9b-" + "x" * 600
        with pytest.raises(ValueError) as exc:
            validate_longlora_compat(
                model_name=long_name,
                task="sft",
                backend="transformers",
                use_ring_attention=False,
            )
        # The full long_name must NOT appear verbatim in the message.
        assert long_name not in str(exc.value)
        assert "..." in str(exc.value)


class TestApplyLlamaProFreeze:
    def test_freezes_old_unfreezes_new(self):
        from soup_cli.utils.block_expansion import (
            apply_llama_pro_freeze,
            expand_model_blocks,
        )

        m = FakeModel(n_layers=3)
        expand_model_blocks(m, 2)  # total = 5
        trainable = apply_llama_pro_freeze(m, 2)
        # Trainable count = params in last 2 blocks only.
        assert trainable > 0
        for idx, blk in enumerate(m.model.layers):
            for p in blk.parameters():
                if idx < 3:
                    assert p.requires_grad is False, f"block {idx} should be frozen"
                else:
                    assert p.requires_grad is True, f"block {idx} should be trainable"
        # Top-level param frozen by the global pass.
        assert m._top_param.requires_grad is False

    def test_zero_is_noop(self):
        from soup_cli.utils.block_expansion import apply_llama_pro_freeze

        m = FakeModel(n_layers=3)
        assert apply_llama_pro_freeze(m, 0) == 0


class TestExpandLayersSchema:
    """Verify the schema cross-validator still fires under v0.53.4."""

    def test_expand_layers_requires_freeze_trainable_layers(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  expand_layers: 4
"""
        with pytest.raises(
            (ValidationError, ValueError), match="freeze_trainable_layers"
        ):
            _load(yaml_in)

    def test_expand_layers_with_freeze_accepted(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  expand_layers: 4
  freeze_trainable_layers: 4
"""
        cfg = _load(yaml_in)
        assert cfg.training.expand_layers == 4
        assert cfg.training.freeze_trainable_layers == 4
