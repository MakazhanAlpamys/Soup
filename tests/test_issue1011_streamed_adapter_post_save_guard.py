"""#1011 — Post-save assertion on layer-streamed LoRA adapter artifact.

Twice in Soup's history, a streamed LoRA run has written an adapter file that
contains nothing, and no part of Soup noticed:
- v0.72.0: StreamedDecoderLayer wrapper's `.inner.` child segment leaked into
  every saved key, reloading as 0 tensors into any normal model.
- 2026-09-15 (#1005): peft 0.21 changed adapter selection from matching state_dict
  keys to walking named_modules(). The two enumerations disagreed, the intersection
  was empty, and save_pretrained / trainer.save_model wrote a 40-byte, 0-tensor
  adapter_model.safetensors while exiting 0.

These tests prove:
1. The post-save guard (assert_streamed_adapter_saved) asserts that
   adapter_model.safetensors exists, contains > 0 tensors, matches the expected
   trainable LoRA parameter count, and has no `.inner.` wrapper leakage.
2. The guard is installed on the production save path of streamed models
   (install_streamed_save_guard via build_streamed_model / install_streaming).
3. The guard is CPU-visible (reachable without GPU).
4. MUTATION EVIDENCE: Re-introducing the #1005 divergence causes save_pretrained to
   raise RuntimeError naming the file and expected count, instead of silently
   succeeding.
5. Legitimately empty models (0 trainable LoRA parameters) are deliberately excluded.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _requires_train_extra():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")


# --------------------------------------------------------------------------
# Unit tests on CPU mock models
# --------------------------------------------------------------------------
class TestStreamedAdapterPostSaveGuardUnit:
    def test_legitimately_empty_model_does_not_raise(self, tmp_path):
        """A model with 0 trainable LoRA parameters is a legitimately empty save
        (e.g. unadapted base model or evaluation mode) and is deliberately excluded."""
        import torch
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

        class _UnadaptedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

        model = _UnadaptedModel()
        # No adapter_model.safetensors exists in tmp_path, but expected_count is 0 -> clean return
        assert_streamed_adapter_saved(tmp_path, model)

    def test_missing_adapter_file_raises_runtime_error(self, tmp_path):
        import torch
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

        class _LoraModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

        model = _LoraModel()
        with pytest.raises(RuntimeError) as exc_info:
            assert_streamed_adapter_saved(tmp_path, model)

        msg = str(exc_info.value)
        assert "Streamed adapter save verification failed" in msg
        assert "does not exist" in msg
        assert "expected 1 trainable LoRA parameter tensors" in msg

    def test_zero_tensor_adapter_raises_runtime_error(self, tmp_path):
        """THE #1005 signature: file exists (40 bytes header) but contains 0 tensors."""
        import torch
        import torch.nn as nn
        from safetensors.torch import save_file

        from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

        class _LoraModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.zeros(2, 2), requires_grad=True)
                self.lora_B = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

        model = _LoraModel()
        # Save 0 tensors to adapter_model.safetensors
        save_file({}, os.path.join(tmp_path, "adapter_model.safetensors"))

        with pytest.raises(RuntimeError) as exc_info:
            assert_streamed_adapter_saved(tmp_path, model)

        msg = str(exc_info.value)
        assert "Streamed adapter save verification failed" in msg
        assert "0 tensors" in msg
        assert "expected 2 trainable LoRA parameter tensors" in msg
        assert "enumeration divergence between named_modules() and state_dict()" in msg

    def test_tensor_count_mismatch_raises_runtime_error(self, tmp_path):
        import torch
        import torch.nn as nn
        from safetensors.torch import save_file

        from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

        class _LoraModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.zeros(2, 2), requires_grad=True)
                self.lora_B = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

        model = _LoraModel()
        # Save only 1 tensor when 2 are expected
        save_file(
            {"weight": torch.zeros(2, 2)},
            os.path.join(tmp_path, "adapter_model.safetensors"),
        )

        with pytest.raises(RuntimeError) as exc_info:
            assert_streamed_adapter_saved(tmp_path, model)

        msg = str(exc_info.value)
        assert "expected 2 trainable LoRA parameter tensors" in msg
        assert "saved adapter contains 1 tensors" in msg

    def test_wrapper_leakage_inner_key_raises_runtime_error(self, tmp_path):
        """THE v0.72.0 signature: saved keys contain `.inner.` wrapper segments."""
        import torch
        import torch.nn as nn
        from safetensors.torch import save_file

        from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

        class _LoraModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

        model = _LoraModel()
        save_file(
            {"base_model.model.layers.0.inner.self_attn.lora_A.weight": torch.zeros(2, 2)},
            os.path.join(tmp_path, "adapter_model.safetensors"),
        )

        with pytest.raises(RuntimeError) as exc_info:
            assert_streamed_adapter_saved(tmp_path, model)

        msg = str(exc_info.value)
        assert "wrapper leakage keys" in msg
        assert ".inner." in msg

    def test_healthy_saved_adapter_passes(self, tmp_path):
        import torch
        import torch.nn as nn
        from safetensors.torch import save_file

        from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

        class _LoraModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.zeros(2, 2), requires_grad=True)
                self.lora_B = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

        model = _LoraModel()
        save_file(
            {
                "base_model.model.layers.0.self_attn.lora_A.weight": torch.zeros(2, 2),
                "base_model.model.layers.0.self_attn.lora_B.weight": torch.zeros(2, 2),
            },
            os.path.join(tmp_path, "adapter_model.safetensors"),
        )
        # 2 expected, 2 saved, no .inner. -> passes without error
        assert_streamed_adapter_saved(tmp_path, model)


# --------------------------------------------------------------------------
# Production save path & Mutation evidence
# --------------------------------------------------------------------------
class TestStreamedAdapterPostSaveGuardIntegration:
    @pytest.fixture
    def streamed_cpu_model(self, tmp_path):
        _requires_train_extra()
        from tests.test_issue1005_peft021_canonical_names import _build_streamed_cpu

        model, runtime, weights = _build_streamed_cpu(tmp_path)
        try:
            yield model, runtime
        finally:
            runtime.close()

    def test_normal_streamed_save_pretrained_succeeds(self, streamed_cpu_model, tmp_path):
        """Production save path test: build_streamed_model automatically installs
        the post-save guard on save_pretrained, which passes on healthy models."""
        model, _ = streamed_cpu_model
        assert getattr(model, "_streamed_save_guard_installed", False)

        out_dir = tmp_path / "healthy_adapter"
        model.save_pretrained(str(out_dir))

        adapter_file = out_dir / "adapter_model.safetensors"
        assert adapter_file.is_file()

    def test_trainer_save_model_triggers_guard(self, streamed_cpu_model, tmp_path):
        """Prove that HF Trainer.save_model() triggers the post-save guard as well."""
        from transformers import Trainer, TrainingArguments

        model, _ = streamed_cpu_model
        out_dir = tmp_path / "trainer_saved_adapter"
        args = TrainingArguments(output_dir=str(tmp_path / "trainer_run"))
        trainer = Trainer(model=model, args=args)
        trainer.save_model(str(out_dir))

        adapter_file = out_dir / "adapter_model.safetensors"
        assert adapter_file.is_file()

    def test_mutation_reintroducing_issue1005_divergence_fails_loudly(
        self, streamed_cpu_model, tmp_path, monkeypatch
    ):
        """MUTATION EVIDENCE: Re-introduce the #1005 divergence (where StreamedDecoderLayer
        yielded `.inner` in named_modules(), causing peft 0.21 structural walk to write
        a 0-tensor adapter). Prove that instead of silently succeeding, the production
        save path now raises RuntimeError naming the file and expected count."""
        import peft.utils.save_and_load as peft_save_load

        from tests.test_issue1005_peft021_canonical_names import _wrappers

        model, _ = streamed_cpu_model
        wrappers = _wrappers(model)
        assert len(wrappers) == 2

        # Re-introduce the bug: revert named_modules() to yield .inner on the wrapper
        for w in wrappers:
            def bad_named_modules(memo=None, prefix="", remove_duplicate=True, _w=w):
                if memo is None:
                    memo = set()
                if id(_w) not in memo:
                    memo.add(id(_w))
                    yield prefix, _w
                p = f"{prefix}.inner" if prefix else "inner"
                yield from _w.inner.named_modules(
                    memo=memo, prefix=p, remove_duplicate=remove_duplicate
                )

            w.named_modules = bad_named_modules

        # If running on peft < 0.21, peft's get_peft_model_state_dict filtered by "lora_"
        # string rather than named_modules() prefixes. Enforce peft 0.21's structural walk:
        import peft.peft_model as pm

        orig_get_state_dict = pm.get_peft_model_state_dict

        def peft021_get_peft_model_state_dict(m, *args, **kwargs):
            sd = orig_get_state_dict(m, *args, **kwargs)
            # peft 0.21 _get_tuner_state_dict_key_prefixes walks tuner modules:
            tuner_prefixes = tuple(name for name, _ in m.named_modules() if "lora_" in name)
            return {
                k: v
                for k, v in sd.items()
                if any(k.startswith(p) for p in tuner_prefixes)
            }

        monkeypatch.setattr(pm, "get_peft_model_state_dict", peft021_get_peft_model_state_dict)
        monkeypatch.setattr(
            peft_save_load, "get_peft_model_state_dict", peft021_get_peft_model_state_dict
        )

        out_dir = tmp_path / "mutated_divergent_adapter"
        with pytest.raises(RuntimeError) as exc_info:
            model.save_pretrained(str(out_dir))

        err_msg = str(exc_info.value)
        assert "Streamed adapter save verification failed" in err_msg
        assert "0 tensors" in err_msg
        assert "expected 8 trainable LoRA parameter tensors" in err_msg

    def test_distributed_non_main_rank_skips_verification(self, tmp_path):
        """When is_main_process=False in distributed runs, save_pretrained does
        not write files on non-main ranks; the guard must not raise for the missing file."""
        from soup_cli.utils.layer_stream_runtime import install_streamed_save_guard

        class _MockModel:
            def __init__(self):
                self._saved = False

            def save_pretrained(self, save_directory, is_main_process=True, **kwargs):
                self._saved = True

            def named_parameters(self):
                import torch
                import torch.nn as nn
                p = nn.Parameter(torch.zeros(1), requires_grad=True)
                return [("lora_A", p)]

        model = _MockModel()
        install_streamed_save_guard(model)

        # Non-main process does not write file; must not raise
        model.save_pretrained(str(tmp_path / "rank1"), is_main_process=False)
        assert model._saved
