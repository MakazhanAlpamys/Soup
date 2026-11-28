"""A real Trainer must keep all updates while recording a cache profile."""

import pytest


@pytest.mark.gpu(reason="Profile callbacks and Trainer BF16 updates need real CUDA")
def test_profile_recording_preserves_real_trainer_adapters_and_adamw(tmp_path, record_property):
    import json
    import time
    from types import SimpleNamespace

    import torch
    from datasets import Dataset
    from transformers import PretrainedConfig, Trainer, TrainingArguments

    from soup_cli.config.schema import SoupConfig
    from soup_cli.utils.stream_cache_profile import (
        build_stream_cache_callback,
        load_measurements,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig(vocab_size=8)
            self.lora_A = torch.nn.Parameter(torch.randn(8, 8, device="cuda") * .02)
            self.lora_B = torch.nn.Parameter(torch.randn(8, 8, device="cuda") * .02)

        def forward(self, input_ids, **_kwargs):
            onehot = torch.nn.functional.one_hot(input_ids, 8).float()
            logits = onehot @ self.lora_A @ self.lora_B
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(),
                                                     input_ids.flatten())
            return {"loss": loss, "logits": logits}

    dataset = Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [5, 6, 7, 0]] * 8})
    states = []
    for record in (False, True):
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        model = Model()
        initial = {name: param.detach().clone() for name, param in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, foreach=False, fused=False)
        args = TrainingArguments(
            output_dir=str(tmp_path / ("record" if record else "plain")), max_steps=8,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            bf16=True, report_to=[], save_strategy="no", logging_strategy="no",
            disable_tqdm=True, seed=31, data_seed=31, learning_rate=.002,
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset,
                          optimizers=(optimizer, None))
        if record:
            path = tmp_path / "profile.json"
            cfg = SoupConfig.model_validate({
                "base": "toy-callback-control", "task": "sft", "data": {"train": "x"},
                "training": {"stream_layers": True, "stream_source": "disk", "batch_size": 1,
                             "gradient_accumulation_steps": 2,
                             "stream_cache_profile": str(path),
                             "stream_cache_profile_mode": "record"},
            })
            index = tmp_path / "index.json"
            index.write_text('{"control":"Trainer callbacks, not model throughput"}')
            owner = SimpleNamespace(trainer=trainer, _stream_runtime=SimpleNamespace(
                tier="disk", pool=SimpleNamespace(is_cuda=True, device="cuda:0", resident_slots={}),
                source=SimpleNamespace(ram_cache_stats=lambda: {"indices": []}),
            ), _stream_step_shape=lambda *_: (1, 4, 8))
            callback = build_stream_cache_callback(owner, cfg, index_path=index,
                setup_started=time.perf_counter(), notify=lambda _: None)
            trainer.add_callback(callback)
        result = trainer.train()
        assert result.global_step == 8
        assert all(not torch.equal(param, initial[name])
                   for name, param in model.named_parameters())
        if record:
            profile = load_measurements(path, callback.identity)
            assert len(profile) == 1 and profile[0].ordinary
            assert len(callback.samples) == 8 and not callback.recording
        states.append({name: {
            "parameter": param.detach().cpu().clone(),
            **{key: value.detach().cpu().clone() for key, value in optimizer.state[param].items()
               if torch.is_tensor(value)},
        } for name, param in model.named_parameters()})
    assert states[0].keys() == states[1].keys()
    for name, expected in states[0].items():
        assert expected.keys() == states[1][name].keys()
        assert all(torch.equal(value, states[1][name][key]) for key, value in expected.items())
    record_property("profile_callback_proof", json.dumps({
        "updates_per_arm": 8, "all_updates_retained": True,
        "parameters_and_adamw_bitwise": True, "profile_samples": 8,
    }))
