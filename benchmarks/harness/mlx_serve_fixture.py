"""Offline trained Llama fixture shared by native server tests and qualification."""

import json
from pathlib import Path


def save_fixture(root: Path) -> dict:
    """Create and actually train a small reloadable Llama LoRA, entirely offline."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import save_config, save_model
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    root.mkdir(parents=True, exist_ok=True)
    base, adapter = root / "base", root / "adapter"
    base.mkdir()
    adapter.mkdir()
    mx.random.seed(13)
    config = dict(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
        tie_word_embeddings=True,
        eos_token_id=2,
    )
    model = Model(ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    save_model(base, model)
    save_config(config, base / "config.json")
    vocabulary = {
        "[UNK]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "red": 3,
        "green": 4,
        "blue": 5,
        **{f"t{i}": i for i in range(6, 32)},
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(base)
    frozen = {name: np.array(value) for name, value in tree_flatten(model.parameters())}
    model.freeze()
    parameters = dict(
        rank=4,
        scale=2.0,
        dropout=0.0,
        keys=[
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.down_proj",
        ],
    )
    linear_to_lora_layers(model, 1, parameters)
    optimizer = optim.Adam(learning_rate=0.03)
    inputs = mx.array([[3, 4], [6, 3]])

    def objective(current):
        return nn.losses.cross_entropy(current(inputs)[:, -1, :], mx.array([5, 5])).mean()

    value_and_grad = nn.value_and_grad(model, objective)
    losses = []
    for _ in range(16):
        loss, gradients = value_and_grad(model)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
    after = dict(tree_flatten(model.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        mapped = name if name in after else f"{parent}.linear.{leaf}"
        np.testing.assert_array_equal(np.array(after[mapped]), value)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert all(bool(mx.all(mx.isfinite(value)).item()) for value in trainable.values())
    delta = max(
        float(mx.max(mx.abs(value)).item())
        for name, value in trainable.items()
        if name.endswith("lora_b")
    )
    assert delta > 1e-5 and losses[-1] < losses[0]
    mx.save_safetensors(str(adapter / "adapters.safetensors"), trainable)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            dict(
                model=str(base),
                fine_tune_type="lora",
                num_layers=1,
                lora_parameters=parameters,
            )
        )
    )
    return dict(
        base=base,
        adapter=adapter,
        losses=losses,
        max_lora_b=delta,
        frozen_tensors_unchanged=len(frozen),
        trainable_tensors=len(trainable),
    )
