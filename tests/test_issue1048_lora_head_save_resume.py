"""#1048 (section 1): resuming a LoRA adapter targeting a streamed large layer
(lm_head/embed_tokens) silently dropped the head's own tensors.

``StreamedLargeLayer._redirect_canonical_weight`` is a ``_load_state_dict_pre_hook``
that exists to route a canonical key (what ``state_dict()`` reports, e.g.
``lm_head.weight``) into the wrapper's real child ``inner`` (what torch's own
recursive ``load_state_dict`` walks), the same job
``StreamedDecoderLayer._redirect_canonical_keys`` already does for decoder-layer
projections. Before this fix it only redirected the literal key ``"weight"``. But
``get_peft_model()`` runs before ``install_streaming()`` (#1012), so a LoRA target on
one of these boundary modules makes ``inner`` a peft tuner layer whose own state dict
carries ``lora_A.default.weight``/``lora_B.default.weight`` (or
``lora_embedding_A``/``lora_embedding_B`` for an embedding) instead of a bare
``"weight"``. None of those match the old literal check, so on
``set_peft_model_state_dict``/``load_adapter`` they were never redirected and torch's
recursion into the real ``inner`` child never saw them: measured 8 of 10 LoRA tensors
landing, with the head's own two silently staying at their freshly-initialised value
(the production restore path in ``rl_checkpoint.py`` swallows the mismatch and reports
success).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _requires_train_extra():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")


def _tiny_llama_dir(tmp_path, tie):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _lora(target_modules):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed(tmp_path, *, tie, target_modules, seed):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(tmp_path, tie=tie)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=_lora(target_modules),
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=seed,
    )
    return model, runtime


def _perturb_lora(model, seed):
    """Move every LoRA tensor off its init value so a dropped-on-load tensor
    is distinguishable from one that loaded correctly. ``lora_B`` starts at
    all-zero by LoRA's own init convention, so an untrained model's ``lora_B``
    is identical across two independently-seeded builds regardless of whether
    resume actually loaded it, a false pass this guards against."""
    import torch

    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name and not param.is_meta:
                param.copy_(torch.randn_like(param) + 5.0)


class TestSaveResumeLandsEveryLoraTensor:
    @pytest.mark.parametrize("head_target", ["lm_head", "embed_tokens"])
    def test_full_round_trip(self, tmp_path, head_target):
        import torch
        from peft import load_peft_weights, set_peft_model_state_dict

        _requires_train_extra()
        targets = ["q_proj", "v_proj", head_target]

        model, runtime = _build_streamed(
            tmp_path / "a", tie=False, target_modules=targets, seed=3
        )
        try:
            _perturb_lora(model, seed=1234)
            saved = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if "lora_" in name
            }
            assert saved, "fixture: no LoRA parameters produced"

            ckpt_dir = tmp_path / "ckpt"
            ckpt_dir.mkdir()
            # save_embedding_layers=False sidesteps the separate
            # save_pretrained meta-tensor crash (#1048 section 1, part 1),
            # which is out of scope for this fix.
            model.save_pretrained(str(ckpt_dir), save_embedding_layers=False)
        finally:
            runtime.close()

        # A fresh, independently-seeded build: what a resumed run starts from.
        model2, runtime2 = _build_streamed(
            tmp_path / "b", tie=False, target_modules=targets, seed=99
        )
        try:
            state = load_peft_weights(str(ckpt_dir))
            assert len(state) == len(saved)
            set_peft_model_state_dict(model2, state)

            loaded = dict(model2.named_parameters())
            for name, original in saved.items():
                assert torch.equal(loaded[name].detach(), original), (
                    f"{name!r} did not land on resume"
                )
        finally:
            runtime2.close()


class TestDecoderOnlyTargetsStillWork:
    def test_no_boundary_target_save_resume_unaffected(self, tmp_path):
        """Negative control: a LoRA config with no target on the streamed
        large-layer boundary never touches StreamedLargeLayer at all, so the
        redirect change here must not affect it."""
        import torch
        from peft import load_peft_weights, set_peft_model_state_dict

        _requires_train_extra()
        targets = ["q_proj", "v_proj"]

        model, runtime = _build_streamed(
            tmp_path / "a", tie=False, target_modules=targets, seed=3
        )
        try:
            _perturb_lora(model, seed=1234)
            saved = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if "lora_" in name
            }
            ckpt_dir = tmp_path / "ckpt"
            ckpt_dir.mkdir()
            model.save_pretrained(str(ckpt_dir))
        finally:
            runtime.close()

        model2, runtime2 = _build_streamed(
            tmp_path / "b", tie=False, target_modules=targets, seed=99
        )
        try:
            set_peft_model_state_dict(model2, load_peft_weights(str(ckpt_dir)))
            loaded = dict(model2.named_parameters())
            for name, original in saved.items():
                assert torch.equal(loaded[name].detach(), original)
        finally:
            runtime2.close()
