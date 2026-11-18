"""An auto fp16 pick on a bf16 card is priced as fp32 weights (#3044).

``trainer/sft.py`` gives such a run fp32 non-quantised embeddings, norms and head.
The train pre-flight resolves the pick on the card it knows and prices those
parameters; ``soup profile --config`` and the MCP ``profile`` tool know no card,
so they keep their previous, card-independent answer. No GPU here: the card,
the picker and the model shape are patched.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import yaml

from soup_cli.commands import train as train_mod
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.hardware_fit import (
    HardwareFitInput,
    estimate_peak_vram_gb,
    hardware_fit_input_from_config,
    hardware_fit_input_or_reason,
)
from tests._tiny_checkpoint import write_tiny_model

_TINY_QWEN = {
    "model_type": "qwen2",
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "vocab_size": 32000,
}
_PARAMS = 7_000_000_000


def _cfg(*, auto: bool = True, base: str = "Qwen/Qwen2.5-7B-Instruct", quant: str = "4bit"):
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "data": {"train": "train.jsonl", "max_length": 512},
                "training": {
                    "batch_size": 2,
                    "quantization": quant,
                    "auto_mixed_precision": auto,
                },
            }
        )
    )


@pytest.fixture
def bf16_card_picking_fp16(monkeypatch):
    """A bf16 card (cc 8.6) whose Qwen2.5 pick is fp16; no Hub lookup."""
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))
    monkeypatch.setattr(
        model_shape, "_hub_lookup", lambda repo, **kw: (dict(_TINY_QWEN), None)
    )
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: _PARAMS)


class TestThePreflightPick:
    def test_an_fp16_pick_on_a_bf16_card_keeps_fp32_weights(self, bf16_card_picking_fp16):
        cfg = _cfg()
        assert train_mod._auto_fp16_pick_keeps_fp32(cfg) is True

    def test_a_bf16_pick_does_not(self, monkeypatch, bf16_card_picking_fp16):
        from soup_cli.utils import mixed_precision

        monkeypatch.setattr(mixed_precision, "pick_mixed_precision", lambda name, cc: "bf16")
        assert train_mod._auto_fp16_pick_keeps_fp32(_cfg()) is False

    def test_without_auto_mixed_precision_it_does_not(self, bf16_card_picking_fp16):
        assert train_mod._auto_fp16_pick_keeps_fp32(_cfg(auto=False)) is False

    def test_a_card_without_a_readable_pick_keeps_the_old_answer(
        self, monkeypatch, bf16_card_picking_fp16
    ):
        import torch

        def broken(device=None):
            raise RuntimeError("no card")

        monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
        assert train_mod._auto_fp16_pick_keeps_fp32(_cfg()) is False

    def test_the_preflight_passes_the_flag_to_the_estimate(self, monkeypatch):
        """The pick is resolved where the card is known, not in the shared builder."""
        captured = {}

        def spy(cfg, **kwargs):
            captured.update(kwargs)
            return hardware_fit.FitUnavailable(None, "stop before the estimate")

        monkeypatch.setattr(train_mod, "_auto_fp16_pick_keeps_fp32", lambda cfg: True)
        monkeypatch.setattr(train_mod, "_hardware_fit_input_or_reason", spy)
        train_mod._hardware_fit_preflight(
            _cfg(), {"memory_total_bytes": int(24e9)}, allow_oom_attempt=False
        )
        assert captured["kbit_fp32"] is True


class TestTheEstimate:
    def test_the_weights_term_grows_by_two_bytes_per_embedding_and_head(
        self, bf16_card_picking_fp16
    ):
        cfg = _cfg()
        bf16_inp = hardware_fit_input_from_config(cfg, allow_network=False, kbit_fp32=False)
        fp32_inp = hardware_fit_input_from_config(cfg, allow_network=False, kbit_fp32=True)
        assert isinstance(bf16_inp, HardwareFitInput) and isinstance(fp32_inp, HardwareFitInput)
        bf16 = estimate_peak_vram_gb(bf16_inp)
        fp32 = estimate_peak_vram_gb(fp32_inp)
        # A Qwen2.5 shape stores the embedding and the (untied) head; eight bytes
        # per pair of parameters across the two prices.
        expected = 2 * 2 * _TINY_QWEN["vocab_size"] * _TINY_QWEN["hidden_size"] / 1e9
        assert fp32.weights_gb > bf16.weights_gb
        assert fp32.weights_gb - bf16.weights_gb == pytest.approx(expected, rel=0.05)

    def test_a_dense_run_is_unaffected(self, bf16_card_picking_fp16):
        cfg = _cfg(quant="none")
        plain = hardware_fit_input_from_config(cfg, allow_network=False, kbit_fp32=False)
        flagged = hardware_fit_input_from_config(cfg, allow_network=False, kbit_fp32=True)
        assert isinstance(plain, HardwareFitInput) and isinstance(flagged, HardwareFitInput)
        assert estimate_peak_vram_gb(plain).weights_gb == estimate_peak_vram_gb(
            flagged
        ).weights_gb

    def test_the_shared_builder_only_uses_an_explicit_flag(self, bf16_card_picking_fp16):
        cfg = _cfg()
        inp = hardware_fit_input_or_reason(cfg, allow_network=False)
        assert isinstance(inp, HardwareFitInput)
        assert inp.kbit_fp32 is False


class TestProfileStaysCardIndependent:
    def test_the_profile_input_does_not_read_the_card(self, monkeypatch, bf16_card_picking_fp16):
        cfg = _cfg()
        with_card = hardware_fit_input_from_config(cfg, allow_network=False)
        import torch

        def broken(device=None):
            raise AssertionError("the profile path must not ask the card")

        monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
        without_card = hardware_fit_input_from_config(cfg, allow_network=False)
        assert with_card == without_card
        assert without_card.kbit_fp32 is False

    def test_profile_from_config_does_not_import_torch(self, tmp_path):
        """A fresh interpreter: the profile config path stays on the light stack."""
        write_tiny_model(tmp_path / "tiny")
        cfg_text = yaml.safe_dump(
            {
                "base": str(tmp_path / "tiny"),
                "task": "sft",
                "data": {"train": "train.jsonl", "max_length": 512},
                "training": {
                    "batch_size": 2,
                    "quantization": "4bit",
                    "auto_mixed_precision": True,
                },
            }
        )
        script = tmp_path / "probe.py"
        script.write_text(
            textwrap.dedent(
                '''
                import sys
                from pathlib import Path
                from soup_cli.config.loader import load_config_from_string
                from soup_cli.utils.profiler import profile_from_config

                cfg = load_config_from_string(Path(sys.argv[1]).read_text())
                profile_from_config(cfg, allow_network=False, gpu_memory_gb=8.0)
                print("torch" in sys.modules)
                '''
            )
        )
        config = tmp_path / "soup.yaml"
        config.write_text(cfg_text)
        proc = subprocess.run(
            [sys.executable, str(script), str(config)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert proc.stdout.strip().endswith("False")
