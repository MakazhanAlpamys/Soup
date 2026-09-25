"""#1263: GRPO variants log the `kl` metric matching trl's stock loss.

In trl 0.29.1 stock loss records `mean_kl = masked_batch_mean(per_token_kl)` into
`self._metrics[mode]["kl"]` via `self.accelerator.gather(mean_kl).nanmean().item()`.
Soup's GRPO variants replace trl's `_compute_loss` but omitted recording the metric,
leaving the `kl` metric missing in the console, tracker, and W&B.

Acceptance:
1. `compute_variant_mean_kl` correctly reduces per-token KL for standard variants and rft.
2. On a tiny CPU model through the real trainer, a variant run with `grpo_beta: 0.1` logs `kl`.
3. For `dapo`, logged `kl` equals trl's stock `kl` on the same batch.
4. `rft` averages over accepted tokens only.
5. When `beta == 0.0`, nothing is logged to the metric.
"""

from collections import defaultdict

import pytest

VARIANTS = ("gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft")


def _torch():
    return pytest.importorskip("torch")


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _batch(torch):
    gen = torch.Generator().manual_seed(1263)
    dtype = torch.float64
    logp_new = (-2.0 * torch.rand(3, 5, generator=gen, dtype=dtype)).requires_grad_(True)
    offset = torch.empty(3, 5, dtype=dtype).uniform_(-1.5, 1.5, generator=gen)
    return {
        "logp_new": logp_new,
        "reference": logp_new.detach() + offset,
        "advantages": torch.tensor([1.0, -0.5, 0.75], dtype=dtype),
        "mask": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=dtype
        ),
    }


class TestComputeVariantMeanKl:
    @pytest.mark.parametrize("variant", ("gspo", "dapo", "dr_grpo", "bnpo", "two_sided"))
    def test_standard_variants_average_over_mask(self, variant):
        torch = _torch()
        b = _batch(torch)
        from soup_cli.utils.grpo_variants import compute_variant_mean_kl

        kl_mean = compute_variant_mean_kl(
            variant,
            logp_new=b["logp_new"],
            reference_logp=b["reference"],
            completion_mask=b["mask"],
            advantages=b["advantages"],
        )
        ref_minus_logp = b["reference"].detach() - b["logp_new"].detach()
        per_token_kl = torch.exp(ref_minus_logp) - ref_minus_logp - 1
        expected = (per_token_kl * b["mask"]).sum() / b["mask"].sum()
        assert kl_mean.item() == pytest.approx(expected.item(), rel=1e-9)

    def test_rft_averages_over_accepted_tokens_only(self):
        torch = _torch()
        b = _batch(torch)
        from soup_cli.utils.grpo_variants import compute_variant_mean_kl

        kl_mean = compute_variant_mean_kl(
            "rft",
            logp_new=b["logp_new"],
            reference_logp=b["reference"],
            completion_mask=b["mask"],
            advantages=b["advantages"],
        )
        ref_minus_logp = b["reference"].detach() - b["logp_new"].detach()
        per_token_kl = torch.exp(ref_minus_logp) - ref_minus_logp - 1
        accepted_mask = (b["advantages"].unsqueeze(-1) > 0).to(dtype=b["mask"].dtype) * b["mask"]
        expected = (per_token_kl * accepted_mask).sum() / accepted_mask.sum()
        assert kl_mean.item() == pytest.approx(expected.item(), rel=1e-9)

    def test_rft_zero_accepted_tokens_returns_zero(self):
        torch = _torch()
        b = _batch(torch)
        all_negative_advantages = torch.tensor([-1.0, -0.5, -2.0], dtype=torch.float64)
        from soup_cli.utils.grpo_variants import compute_variant_mean_kl

        kl_mean = compute_variant_mean_kl(
            "rft",
            logp_new=b["logp_new"],
            reference_logp=b["reference"],
            completion_mask=b["mask"],
            advantages=all_negative_advantages,
        )
        assert kl_mean.item() == 0.0

    def test_no_mask_averages_all_tokens(self):
        torch = _torch()
        b = _batch(torch)
        from soup_cli.utils.grpo_variants import compute_variant_mean_kl

        kl_mean = compute_variant_mean_kl(
            "dapo",
            logp_new=b["logp_new"],
            reference_logp=b["reference"],
            completion_mask=None,
        )
        ref_minus_logp = b["reference"].detach() - b["logp_new"].detach()
        per_token_kl = torch.exp(ref_minus_logp) - ref_minus_logp - 1
        expected = per_token_kl.mean()
        assert kl_mean.item() == pytest.approx(expected.item(), rel=1e-9)


def _tiny_llama_dir(tmp_path, vocab=64, hidden=64):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _write_tiny_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    fast.save_pretrained(str(directory))


def _build_real(tmp_path, monkeypatch, variant, **training_over):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.grpo import GRPOTrainerWrapper

    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    training = {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "reward_fn": "format",
        "quantization": "none",
        "epochs": 1,
        "grpo_variant": variant,
        "grpo_beta": 0.1,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    if variant == "two_sided":
        training["grpo_delta"] = 0.2
    training.update(training_over)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "grpo",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = GRPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"prompt": "hi"} for _ in range(4)]})
    return wrapper


def _generated_batch_off_reference(trainer):
    import torch

    torch.manual_seed(0)
    batch = next(iter(trainer.get_train_dataloader()))
    inputs = trainer._prepare_inputs(batch)
    assert inputs.get("ref_per_token_logps") is not None
    rows = inputs["advantages"].shape[0]
    inputs["advantages"] = torch.tensor([1.0, -1.0] * (rows // 2))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in trainer.model.named_parameters():
            if "lora_B" in name:
                param.add_(0.5 * torch.randn(param.shape, generator=generator))
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = 1
    return inputs


class TestRealTrainerLogKl:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_variant_logs_kl_when_beta_positive(self, tmp_path, monkeypatch, variant):
        _requires_train_extra()
        wrapper = _build_real(tmp_path, monkeypatch, variant)
        trainer = wrapper.trainer
        inputs = _generated_batch_off_reference(trainer)

        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        loss = trainer._compute_loss(trainer.model, inputs)
        assert loss is not None
        assert "kl" in trainer._metrics["train"]
        assert len(trainer._metrics["train"]["kl"]) == 1
        kl_val = trainer._metrics["train"]["kl"][0]
        assert kl_val > 0.0

    def test_dapo_logged_kl_equals_trl_stock_kl_on_same_batch(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper = _build_real(tmp_path, monkeypatch, "dapo")
        trainer = wrapper.trainer
        inputs = _generated_batch_off_reference(trainer)

        # 1. Compute variant loss and extract variant's logged KL
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        variant_loss = trainer._compute_loss(trainer.model, inputs)
        assert variant_loss is not None
        variant_kl = trainer._metrics["train"]["kl"][0]

        # 2. Compute trl stock loss on the same batch and extract stock logged KL
        stock_cls = type(trainer).__mro__[1]
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        stock_loss = stock_cls._compute_loss(trainer, trainer.model, inputs)
        assert stock_loss is not None
        stock_kl = trainer._metrics["train"]["kl"][0]

        assert variant_kl == pytest.approx(stock_kl, rel=1e-7, abs=1e-7)

    def test_rft_averages_only_over_accepted_tokens_in_trainer(self, tmp_path, monkeypatch):
        _requires_train_extra()
        import torch

        wrapper = _build_real(tmp_path, monkeypatch, "rft")
        trainer = wrapper.trainer
        inputs = _generated_batch_off_reference(trainer)

        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._compute_loss(trainer.model, inputs)
        logged_kl = trainer._metrics["train"]["kl"][0]

        # Independent calculation
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logp, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, inputs["completion_ids"].size(1)
        )
        ref = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref - logp) - (ref - logp) - 1
        pos_mask = (inputs["advantages"].unsqueeze(-1) > 0).to(logp.dtype)
        mask = pos_mask * inputs["completion_mask"]
        expected_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1.0)

        assert logged_kl == pytest.approx(expected_kl.item(), rel=1e-5, abs=1e-7)

    def test_nothing_logged_when_beta_is_zero(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper = _build_real(tmp_path, monkeypatch, "dapo")
        trainer = wrapper.trainer
        inputs = _generated_batch_off_reference(trainer)

        trainer.beta = 0.0
        trainer.args.beta = 0.0
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        loss = trainer._compute_loss(trainer.model, inputs)
        assert loss is not None
        assert len(trainer._metrics["train"]["kl"]) == 0

    def test_eval_mode_records_under_eval_metrics(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper = _build_real(tmp_path, monkeypatch, "dapo")
        trainer = wrapper.trainer
        inputs = _generated_batch_off_reference(trainer)

        trainer.model.eval()
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        loss = trainer._compute_loss(trainer.model, inputs)
        assert loss is not None
        assert "kl" in trainer._metrics["eval"]
        assert len(trainer._metrics["eval"]["kl"]) == 1
        assert len(trainer._metrics["train"]["kl"]) == 0
