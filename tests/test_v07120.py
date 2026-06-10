"""v0.71.20 — Modality II trainers (BETA, hw-gated): #131 TTS, #134 BitNet,
#136 MoE expert quant + router-only training.

Lifts the v0.52.0 schema-only stubs:
- #131 build_tts_trainer → live TTSTrainerWrapper (SFT-CE + per-family
  emotion templating + codec special-token registration). Pre-encoded chat
  mode is the live/smokeable path; the live-codec (audio→tokens at train time)
  path is hardware/dependency-gated with a friendly per-family RuntimeError.
- #134 build_bitnet_trainer + export_bitnet_gguf → real plumbing behind
  friendly onebitllms / llama.cpp gates.
- #136 apply_moe_expert_quant + freeze_experts_train_router + SFT wiring.

torch-needing tests use ``pytest.importorskip``; the pure validators/templating
+ source-wiring guards run everywhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli import __version__

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"


# --------------------------------------------------------------------------- #
# #131 TTS — pure helpers
# --------------------------------------------------------------------------- #
class TestTtsCodecPackage:
    def test_per_family_packages(self):
        from soup_cli.utils.tts import TTS_CODEC_PACKAGES, tts_codec_package

        for fam in ("orpheus", "sesame_csm", "llasa", "spark", "oute"):
            assert tts_codec_package(fam) == TTS_CODEC_PACKAGES[fam]
        assert tts_codec_package("orpheus") == "snac"
        assert tts_codec_package("spark") == "sparktts"

    def test_case_insensitive(self):
        from soup_cli.utils.tts import tts_codec_package

        assert tts_codec_package("OrPheus") == "snac"

    def test_unknown_rejected(self):
        from soup_cli.utils.tts import tts_codec_package

        with pytest.raises(ValueError, match="not supported"):
            tts_codec_package("bark")

    def test_packages_immutable(self):
        from soup_cli.utils.tts import TTS_CODEC_PACKAGES

        with pytest.raises(TypeError):
            TTS_CODEC_PACKAGES["x"] = "y"  # type: ignore[index]


class TestFormatTtsMessages:
    def _msgs(self):
        return [
            {"role": "system", "content": "Speak."},
            {"role": "user", "content": "Hello there."},
            {"role": "assistant", "content": "<codec>123 456</codec>"},
        ]

    def test_no_emotion_passthrough(self):
        from soup_cli.utils.tts import format_tts_messages

        out = format_tts_messages(self._msgs(), "spark", emotion=None)
        assert out[1]["content"] == "Hello there."

    def test_orpheus_emotion_prefixes_user(self):
        from soup_cli.utils.tts import format_tts_messages

        out = format_tts_messages(self._msgs(), "orpheus", emotion="happy")
        assert out[1]["content"].startswith("<|emotion|>happy<|/emotion|>")
        assert "Hello there." in out[1]["content"]

    def test_oute_emotion_template(self):
        from soup_cli.utils.tts import format_tts_messages

        out = format_tts_messages(self._msgs(), "oute", emotion="calm")
        assert out[1]["content"].startswith("[emotion: calm]")

    def test_caller_list_not_mutated(self):
        from soup_cli.utils.tts import format_tts_messages

        msgs = self._msgs()
        original = msgs[1]["content"]
        _ = format_tts_messages(msgs, "orpheus", emotion="happy")
        assert msgs[1]["content"] == original

    def test_emotion_on_unsupported_family_rejected(self):
        from soup_cli.utils.tts import format_tts_messages

        with pytest.raises(ValueError, match="does not support emotion"):
            format_tts_messages(self._msgs(), "spark", emotion="happy")

    def test_non_list_rejected(self):
        from soup_cli.utils.tts import format_tts_messages

        with pytest.raises(TypeError, match="messages must be a list"):
            format_tts_messages("nope", "spark")

    def test_non_dict_message_rejected(self):
        from soup_cli.utils.tts import format_tts_messages

        with pytest.raises(TypeError, match="must be a dict"):
            format_tts_messages(["nope"], "spark")

    def test_only_first_user_turn_prefixed(self):
        from soup_cli.utils.tts import format_tts_messages

        msgs = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        out = format_tts_messages(msgs, "orpheus", emotion="sad")
        assert out[0]["content"].startswith("<|emotion|>sad")
        assert out[1]["content"] == "two"

    def test_non_str_content_left_unchanged(self):
        from soup_cli.utils.tts import format_tts_messages

        parts = [{"type": "text", "text": "hi"}]
        msgs = [{"role": "user", "content": parts}]
        out = format_tts_messages(msgs, "orpheus", emotion="happy")
        # multimodal content list not string-prefixed; deep-copied (not same obj)
        assert out[0]["content"] == parts
        assert out[0]["content"] is not parts

    def test_no_user_turn_with_emotion(self):
        from soup_cli.utils.tts import format_tts_messages

        msgs = [
            {"role": "system", "content": "Speak."},
            {"role": "assistant", "content": "<codec>1</codec>"},
        ]
        out = format_tts_messages(msgs, "orpheus", emotion="happy")
        assert out[0]["content"] == "Speak."
        assert out[1]["content"] == "<codec>1</codec>"

    def test_nested_value_not_shared(self):
        from soup_cli.utils.tts import format_tts_messages

        msgs = [{"role": "user", "content": "hi", "tool_calls": [{"x": 1}]}]
        out = format_tts_messages(msgs, "orpheus", emotion="happy")
        out[0]["tool_calls"][0]["x"] = 99
        assert msgs[0]["tool_calls"][0]["x"] == 1


class TestBuildTtsTrainer:
    def test_no_arg_typeerror(self):
        from soup_cli.utils.tts import build_tts_trainer

        with pytest.raises(TypeError):
            build_tts_trainer()

    def test_returns_wrapper(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.tts import TTSTrainerWrapper
        from soup_cli.utils.tts import build_tts_trainer

        yaml_str = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: tts\n"
            "modality: audio_out\n"
            "data:\n"
            "  train: ./x.jsonl\n"
            "  format: auto\n"
            "training:\n"
            "  tts_family: spark\n"
            "  epochs: 1\n"
            "  lr: 0.0001\n"
            "  batch_size: 1\n"
        )
        cfg = load_config_from_string(yaml_str)
        wrapper = build_tts_trainer(cfg, device="cpu")
        assert isinstance(wrapper, TTSTrainerWrapper)


class TestTtsTrainerSetup:
    def _wrapper(self, *, data_format, family="spark", emotion=None):
        from soup_cli.trainer.tts import TTSTrainerWrapper

        w = object.__new__(TTSTrainerWrapper)
        w.config = SimpleNamespace(
            training=SimpleNamespace(tts_family=family, tts_emotion=emotion),
            data=SimpleNamespace(format=data_format, new_special_tokens=None),
        )
        w._tts_family = None
        return w

    def test_live_codec_mode_gates_when_codec_absent(self):
        # data.format='audio' + family whose codec is not installed.
        w = self._wrapper(data_format="audio", family="spark")
        with pytest.raises(RuntimeError, match="sparktts"):
            w.setup({"train": []})

    def test_live_codec_mode_orpheus_names_snac(self):
        w = self._wrapper(data_format="audio", family="orpheus")
        with pytest.raises(RuntimeError, match="snac"):
            w.setup({"train": []})

    def test_chat_mode_applies_templating_and_delegates(self):
        # In pre-encoded chat mode, setup() emotion-templates the dataset then
        # calls super().setup(). Stub super().setup to capture the dataset.
        from soup_cli.trainer.sft import SFTTrainerWrapper

        w = self._wrapper(data_format="auto", family="orpheus", emotion="happy")
        captured = {}

        def _fake_super_setup(self, dataset):  # noqa: ANN001
            captured["dataset"] = dataset

        orig = SFTTrainerWrapper.setup
        try:
            SFTTrainerWrapper.setup = _fake_super_setup  # type: ignore[assignment]
            w.setup({
                "train": [
                    {"messages": [{"role": "user", "content": "hi"}]},
                ]
            })
        finally:
            SFTTrainerWrapper.setup = orig  # type: ignore[assignment]

        msg = captured["dataset"]["train"][0]["messages"][0]
        assert msg["content"].startswith("<|emotion|>happy")

    def test_apply_tts_templating_non_message_rows_passthrough(self):
        w = self._wrapper(data_format="auto", family="spark")
        ds = w._apply_tts_templating(
            {"train": [{"text": "raw"}], "val": []}, "spark", None
        )
        assert ds["train"][0] == {"text": "raw"}

    def test_apply_tts_templating_val_split_templated(self):
        w = self._wrapper(data_format="auto", family="orpheus", emotion="happy")
        ds = w._apply_tts_templating(
            {
                "train": [],
                "val": [{"messages": [{"role": "user", "content": "v"}]}],
            },
            "orpheus",
            "happy",
        )
        assert ds["val"][0]["messages"][0]["content"].startswith("<|emotion|>happy")

    def test_apply_tts_templating_no_train_key(self):
        w = self._wrapper(data_format="auto", family="spark")
        ds = w._apply_tts_templating({}, "spark", None)
        assert ds == {}

    def test_register_special_tokens_adds_and_resizes(self):
        w = self._wrapper(data_format="auto")
        w.config.data.new_special_tokens = ["<aud_0>", "<aud_1>"]
        calls = {"resize": None, "added": None}

        class _Tok:
            def get_vocab(self):
                return {"<aud_0>": 1}

            def add_special_tokens(self, d):
                calls["added"] = d["additional_special_tokens"]
                return len(d["additional_special_tokens"])

            def __len__(self):
                return 100

        class _Model:
            def resize_token_embeddings(self, n):
                calls["resize"] = n

        w.tokenizer = _Tok()
        w.model = _Model()
        w._register_tts_special_tokens(w.config)
        # <aud_0> already present → only <aud_1> added.
        assert calls["added"] == ["<aud_1>"]
        assert calls["resize"] == 100

    def test_register_special_tokens_all_present_no_resize(self):
        w = self._wrapper(data_format="auto")
        w.config.data.new_special_tokens = ["<aud_0>", "<aud_0>"]
        calls = {"resize": False, "added": False}

        class _Tok:
            def get_vocab(self):
                return {"<aud_0>": 1}

            def add_special_tokens(self, d):
                calls["added"] = True
                return 0

        class _Model:
            def resize_token_embeddings(self, n):
                calls["resize"] = True

        w.tokenizer = _Tok()
        w.model = _Model()
        w._register_tts_special_tokens(w.config)
        # all tokens already present (and deduped) → never add or resize.
        assert calls["added"] is False
        assert calls["resize"] is False

    def test_register_special_tokens_noop_when_none(self):
        w = self._wrapper(data_format="auto")
        w.config.data.new_special_tokens = None
        w.tokenizer = object()
        w.model = object()
        # Should not raise (no tokenizer access).
        w._register_tts_special_tokens(w.config)


class TestTtsSchemaAndRouting:
    def test_tts_config_loads(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: tts\n"
            "modality: audio_out\n"
            "data:\n"
            "  train: ./x.jsonl\n"
            "  format: auto\n"
            "training:\n"
            "  tts_family: orpheus\n"
            "  tts_emotion: happy\n"
        )
        cfg = load_config_from_string(yaml_str)
        assert cfg.task == "tts"
        assert cfg.training.tts_family == "orpheus"

    def test_train_routes_tts(self):
        src = (_SRC / "commands" / "train.py").read_text(encoding="utf-8")
        assert 'cfg.task == "tts"' in src
        assert "TTSTrainerWrapper" in src

    def test_utils_tts_no_top_level_torch(self):
        src = (_SRC / "utils" / "tts.py").read_text(encoding="utf-8")
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


# --------------------------------------------------------------------------- #
# #134 BitNet
# --------------------------------------------------------------------------- #
class TestBuildBitnetTrainer:
    def test_no_arg_typeerror(self):
        from soup_cli.utils.bitnet import build_bitnet_trainer

        with pytest.raises(TypeError):
            build_bitnet_trainer()

    def test_returns_wrapper(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.bitnet import BitNetTrainerWrapper
        from soup_cli.utils.bitnet import build_bitnet_trainer

        yaml_str = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: sft\n"
            "data:\n"
            "  train: ./x.jsonl\n"
            "  format: auto\n"
            "training:\n"
            "  quantization: bitnet_1.58\n"
        )
        cfg = load_config_from_string(yaml_str)
        wrapper = build_bitnet_trainer(cfg, device="cpu")
        assert isinstance(wrapper, BitNetTrainerWrapper)

    def test_setup_gates_on_onebitllms(self):
        from soup_cli.trainer.bitnet import BitNetTrainerWrapper

        # onebitllms is not installed on the maintainer's box → friendly gate.
        if "onebitllms" in sys.modules or _spec("onebitllms"):
            pytest.skip("onebitllms installed; gate path not exercised")
        w = object.__new__(BitNetTrainerWrapper)
        with pytest.raises(RuntimeError, match="onebitllms"):
            w.setup({"train": []})


class TestBitnetGgufQuantArg:
    def test_maps_to_tq1_0(self):
        from soup_cli.utils.bitnet import _BITNET_GGUF_QUANT_ARG

        assert _BITNET_GGUF_QUANT_ARG["bitnet"] == "TQ1_0"
        assert _BITNET_GGUF_QUANT_ARG["tq1_0"] == "TQ1_0"


class TestExportBitnetGguf:
    def test_no_arg_typeerror(self):
        from soup_cli.utils.bitnet import export_bitnet_gguf

        with pytest.raises(TypeError):
            export_bitnet_gguf()

    def test_unknown_format_rejected(self, tmp_path):
        from soup_cli.utils.bitnet import export_bitnet_gguf

        with pytest.raises(ValueError, match="not supported"):
            export_bitnet_gguf(
                model_dir="model",
                output_path="out.gguf",
                export_format="weird",
                llama_cpp_dir="llama",
            )

    def test_outside_cwd_model_rejected(self, tmp_path):
        from soup_cli.utils.bitnet import export_bitnet_gguf

        # tmp_path is outside cwd → containment rejection.
        with pytest.raises(ValueError, match="under cwd"):
            export_bitnet_gguf(
                model_dir=str(tmp_path / "model"),
                output_path="out.gguf",
                export_format="bitnet",
                llama_cpp_dir="llama",
            )

    def test_missing_dir_in_cwd(self, monkeypatch, tmp_path):
        from soup_cli.utils.bitnet import export_bitnet_gguf

        monkeypatch.chdir(tmp_path)
        # model dir under cwd but does not exist → FileNotFoundError.
        with pytest.raises(FileNotFoundError, match="model_dir"):
            export_bitnet_gguf(
                model_dir="nope",
                output_path="out.gguf",
                export_format="bitnet",
                llama_cpp_dir="llama",
            )

    def test_no_llama_cpp_dir(self, monkeypatch, tmp_path):
        from soup_cli.utils.bitnet import export_bitnet_gguf

        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()
        with pytest.raises(FileNotFoundError, match="llama_cpp_dir"):
            export_bitnet_gguf(
                model_dir="model",
                output_path="out.gguf",
                export_format="bitnet",
                llama_cpp_dir="missing_llama",
            )

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlink_model_rejected(self, monkeypatch, tmp_path):
        import os

        from soup_cli.utils.bitnet import export_bitnet_gguf

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "outside"
        target.mkdir()
        link = tmp_path / "model_link"
        os.symlink(str(target), str(link))
        with pytest.raises(ValueError, match="symlink"):
            export_bitnet_gguf(
                model_dir="model_link",
                output_path="out.gguf",
                export_format="bitnet",
                llama_cpp_dir="llama",
            )


class TestBitnetExportCli:
    def _runner(self):
        import typer
        from typer.testing import CliRunner

        from soup_cli.commands.export import export

        app = typer.Typer()
        app.command()(export)
        return CliRunner(), app

    def test_help_lists_bitnet(self):
        from soup_cli.commands.export import SUPPORTED_FORMATS

        assert "bitnet" in SUPPORTED_FORMATS
        assert "tq1_0" in SUPPORTED_FORMATS

    def test_bitnet_export_no_llama_exits_nonzero(self, monkeypatch, tmp_path):
        runner, app = self._runner()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()
        result = runner.invoke(
            app,
            ["--model", "model", "--format", "bitnet", "--llama-cpp-path", "nope"],
        )
        assert result.exit_code != 0, result.output
        assert "llama" in result.output.lower() or "failed" in result.output.lower()

    def test_utils_bitnet_no_top_level_torch(self):
        src = (_SRC / "utils" / "bitnet.py").read_text(encoding="utf-8")
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


# --------------------------------------------------------------------------- #
# #136 MoE expert quant + router-only
# --------------------------------------------------------------------------- #
class TestMoeRouterDetection:
    @pytest.mark.parametrize(
        "name,is_router",
        [
            ("model.layers.0.block_sparse_moe.gate.weight", True),
            ("model.layers.0.block_sparse_moe.gate.bias", True),
            ("model.layers.0.mlp.gate.weight", True),
            ("model.layers.0.mlp.router.weight", True),
            ("model.layers.0.block_sparse_moe.experts.0.gate_proj.weight", False),
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", False),
            ("model.layers.0.self_attn.q_proj.weight", False),
            ("model.layers.0.mlp.gate_proj.weight", False),
            # an expert sub-projection named 'router' must NOT win (expert wins)
            ("model.layers.0.block_sparse_moe.experts.0.router_proj.weight", False),
        ],
    )
    def test_is_router_param(self, name, is_router):
        from soup_cli.utils.moe_quant import _is_router_param

        assert _is_router_param(name) is is_router


class TestMoeExpertHelpers:
    def _fake_moe(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(4, 4)})
                experts = nn.ModuleList([
                    nn.ModuleDict({
                        "w1": nn.Linear(4, 8),
                        "w2": nn.Linear(8, 4),
                    })
                    for _ in range(2)
                ])
                self.block_sparse_moe = nn.ModuleDict({
                    "gate": nn.Linear(4, 2),
                    "experts": experts,
                })

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block()])

            def get_submodule(self, target):
                return dict(self.named_modules())[target]

        return torch, Model()

    def test_find_expert_linears(self):
        from soup_cli.utils.moe_quant import _find_expert_linears

        _torch, model = self._fake_moe()
        found = _find_expert_linears(model)
        names = {n for n, _ in found}
        # 2 experts × 2 linears = 4 expert Linears.
        assert len(found) == 4
        assert all(".experts." in n for n in names)
        # the gate (router) Linear is NOT in the found set.
        assert not any("gate" in n for n in names)

    def test_find_expert_linears_none(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from soup_cli.utils.moe_quant import _find_expert_linears

        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

        assert _find_expert_linears(Plain()) == []

    def test_freeze_experts_train_router(self):
        from soup_cli.utils.moe_quant import freeze_experts_train_router

        _torch, model = self._fake_moe()
        frozen, trainable = freeze_experts_train_router(model)
        assert frozen > 0
        assert trainable > 0
        for name, param in model.named_parameters():
            if ".experts." in name:
                assert param.requires_grad is False
            elif ".gate." in name:
                assert param.requires_grad is True


class TestApplyMoeExpertQuant:
    def test_unknown_format_rejected(self):
        from soup_cli.utils.moe_quant import apply_moe_expert_quant

        with pytest.raises(ValueError, match="not supported"):
            apply_moe_expert_quant(object(), "weird")

    def test_no_experts_returns_zero(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        from soup_cli.utils.moe_quant import apply_moe_expert_quant

        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

        assert apply_moe_expert_quant(Plain(), "nf4") == 0

    def _moe_with_experts(self):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                experts = nn.ModuleList([
                    nn.ModuleDict({"w1": nn.Linear(8, 16), "w2": nn.Linear(16, 8)})
                    for _ in range(2)
                ])
                self.block_sparse_moe = nn.ModuleDict({
                    "gate": nn.Linear(8, 2),
                    "experts": experts,
                })

            def get_submodule(self, target):
                return dict(self.named_modules())[target]

        return Model()

    def test_gates_without_bnb_or_cuda(self):
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available() and _spec("bitsandbytes"):
            pytest.skip("CUDA + bitsandbytes present; real quant path used")
        from soup_cli.utils.moe_quant import apply_moe_expert_quant

        with pytest.raises(RuntimeError, match="bitsandbytes|CUDA"):
            apply_moe_expert_quant(self._moe_with_experts(), "nf4")

    def test_real_nf4_quant_swaps_experts(self):
        """When CUDA + bitsandbytes are present (dev box; skipped on CI),
        the real per-expert quant replaces each expert Linear with a bnb
        Linear4bit. Validated live on an RTX 3050 (Step 6)."""
        torch = pytest.importorskip("torch")
        if not (torch.cuda.is_available() and _spec("bitsandbytes")):
            pytest.skip("requires CUDA + bitsandbytes")
        import bitsandbytes as bnb  # noqa: PLC0415

        from soup_cli.utils.moe_quant import apply_moe_expert_quant

        model = self._moe_with_experts()
        n = apply_moe_expert_quant(model, "nf4")
        assert n == 4  # 2 experts x 2 linears
        # Every expert Linear is now a bnb Linear4bit (a subclass of nn.Linear).
        expert_mods = [
            m for nm, m in model.named_modules() if ".experts." in nm and ".weight" not in nm
        ]
        quantized = [m for m in expert_mods if isinstance(m, bnb.nn.Linear4bit)]
        assert len(quantized) == 4
        # The router (gate) is left in full precision.
        gate = model.get_submodule("block_sparse_moe.gate")
        assert not isinstance(gate, bnb.nn.Linear4bit)


class TestApplyMoeFeaturesIfConfigured:
    def _model(self):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                experts = nn.ModuleList([nn.ModuleDict({"w1": nn.Linear(4, 4)})])
                self.block_sparse_moe = nn.ModuleDict({
                    "gate": nn.Linear(4, 1),
                    "experts": experts,
                })

            def get_submodule(self, target):
                return dict(self.named_modules())[target]

        return Model()

    def test_quant_noop_when_unset(self):
        from soup_cli.utils.moe_quant import apply_moe_expert_quant_if_configured

        tcfg = SimpleNamespace(moe_expert_quant=None)
        # Should not touch the model (object() would error on any access).
        apply_moe_expert_quant_if_configured(object(), tcfg)

    def test_freeze_noop_when_unset(self):
        from soup_cli.utils.moe_quant import apply_router_only_freeze_if_configured

        tcfg = SimpleNamespace(train_router_only=False)
        apply_router_only_freeze_if_configured(object(), tcfg)

    def test_router_only_freezes_and_prints(self):
        pytest.importorskip("torch")

        from soup_cli.utils.moe_quant import apply_router_only_freeze_if_configured

        model = self._model()
        printed = []
        fake_console = SimpleNamespace(print=lambda m: printed.append(m))
        tcfg = SimpleNamespace(train_router_only=True)
        apply_router_only_freeze_if_configured(model, tcfg, fake_console)
        for name, param in model.named_parameters():
            if ".experts." in name:
                assert param.requires_grad is False
            elif ".gate." in name:
                assert param.requires_grad is True
        assert any("router-only" in m for m in printed)

    def test_quant_prints_when_set(self, monkeypatch):
        pytest.importorskip("torch")

        from soup_cli.utils import moe_quant

        model = self._model()
        printed = []
        fake_console = SimpleNamespace(print=lambda m: printed.append(m))
        # Stub the real quant so this runs without CUDA/bnb.
        monkeypatch.setattr(moe_quant, "apply_moe_expert_quant", lambda m, f: 1)
        tcfg = SimpleNamespace(moe_expert_quant="nf4")
        moe_quant.apply_moe_expert_quant_if_configured(model, tcfg, fake_console)
        assert any("expert quant" in m for m in printed)


class TestMoeSchemaAndWiring:
    def test_moe_features_config_loads(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: sft\n"
            "data:\n"
            "  train: ./x.jsonl\n"
            "  format: auto\n"
            "training:\n"
            "  moe_lora: true\n"
            "  moe_expert_quant: nf4\n"
            "  train_router_only: true\n"
        )
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.moe_expert_quant == "nf4"
        assert cfg.training.train_router_only is True

    def test_moe_expert_quant_requires_moe_lora(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: sft\n"
            "data:\n"
            "  train: ./x.jsonl\n"
            "  format: auto\n"
            "training:\n"
            "  moe_expert_quant: nf4\n"
        )
        with pytest.raises(ValueError, match="moe_lora"):
            load_config_from_string(yaml_str)

    def test_sft_wires_moe_features(self):
        src = (_SRC / "trainer" / "sft.py").read_text(encoding="utf-8")
        # Quant runs pre-LoRA, freeze post-LoRA. Assert both wired + ordering.
        assert "apply_moe_expert_quant_if_configured" in src
        assert "apply_router_only_freeze_if_configured" in src
        quant_at = src.index("apply_moe_expert_quant_if_configured(self.model")
        peft_at = src.index("get_peft_model(self.model")
        freeze_at = src.index("apply_router_only_freeze_if_configured(self.model")
        assert quant_at < peft_at < freeze_at

    def test_validate_moe_expert_quant_compat_matrix(self):
        from soup_cli.utils.moe_quant import validate_moe_expert_quant_compat

        # happy
        validate_moe_expert_quant_compat(backend="transformers", moe_lora=True)
        with pytest.raises(ValueError, match="mlx"):
            validate_moe_expert_quant_compat(backend="mlx", moe_lora=True)
        with pytest.raises(ValueError, match="moe_lora"):
            validate_moe_expert_quant_compat(backend="transformers", moe_lora=False)
        with pytest.raises(TypeError, match="must not be bool"):
            validate_moe_expert_quant_compat(backend=True, moe_lora=True)
        with pytest.raises(TypeError, match="must be bool"):
            validate_moe_expert_quant_compat(backend="transformers", moe_lora="yes")

    def test_validate_train_router_only_compat_matrix(self):
        from soup_cli.utils.moe_quant import validate_train_router_only_compat

        validate_train_router_only_compat(backend="transformers", moe_lora=True)
        with pytest.raises(ValueError, match="mlx"):
            validate_train_router_only_compat(backend="mlx", moe_lora=True)
        with pytest.raises(ValueError, match="moe_lora"):
            validate_train_router_only_compat(backend="transformers", moe_lora=False)

    def test_utils_moe_quant_no_top_level_torch(self):
        src = (_SRC / "utils" / "moe_quant.py").read_text(encoding="utf-8")
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


# --------------------------------------------------------------------------- #
# Patch invariants
# --------------------------------------------------------------------------- #
class TestPatchInvariants:
    def test_version_bumped(self):
        parts = tuple(int(p) for p in re.findall(r"\d+", __version__)[:3])
        assert parts >= (0, 71, 20)


def _spec(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
