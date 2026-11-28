"""Exercise the streamed cache with the same pools and L2L walk used by SFT."""

from contextlib import closing
from io import StringIO

import pytest

from tests.conftest import strip_ansi
from tests.test_l2l_probe_harness import _load, _micro_batches, build_streamed, hermetic_ram
from tests.test_l2l_step import _store


def _freeze(model, cutoff):
    from soup_cli.utils.layer_select import (
        adapter_params_by_layer,
        apply_layer_mask,
        cut_input_grads,
    )

    layers = adapter_params_by_layer(model)
    apply_layer_mask(layers, [idx for idx in layers if idx >= cutoff])
    cut_input_grads(model)


def _record_loads(runtime, monkeypatch):
    loads = []
    load = runtime.pool.load_async

    def watched(index, *args, **kwargs):
        loads.append(index)
        return load(index, *args, **kwargs)

    monkeypatch.setattr(runtime.pool, "load_async", watched)
    return loads


def _stream_forward(model, ids, **kwargs):
    out = model(input_ids=ids, labels=ids, **kwargs)
    out.loss.backward()
    model.zero_grad(set_to_none=True)
    return out.logits.detach()


def test_streamed_hits_skip_prefix_loads_but_eval_runs_the_bottom(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        model.train()
        _freeze(model, 2)
        ids = _micro_batches(1)[0]
        plain = _stream_forward(model, ids)
        loads = _record_loads(runtime, monkeypatch)
        with closing(StreamedPrefixCache(model, 2, 1 << 20).install()) as cache:
            assert torch.equal(_stream_forward(model, ids), plain)
            assert any(index < 2 for index in loads)
            loads.clear()
            assert torch.equal(_stream_forward(model, ids), plain)
            assert all(index >= 2 for index in loads)
            assert (cache.stats.hits, cache.stats.misses) == (1, 1)
            model.eval()
            with torch.no_grad():
                assert torch.equal(model(input_ids=ids).logits, plain)
            assert 0 in loads
            assert runtime.prefetcher.prime_start == 0
            assert cache.stats.hits == 1
        assert runtime.prefetcher.backward_floor == 0


def test_streamed_keys_track_padding_positions_and_embedded_values(tmp_path):
    import torch

    from soup_cli.utils.layer_stream_runtime import decoder_owner
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        model.train()
        _freeze(model, 1)
        ids = torch.cat(_micro_batches(2))
        mask = torch.ones_like(ids)
        mask[0, -1] = 0
        positions = torch.arange(ids.shape[1]).expand_as(ids).clone()
        expected = model(ids, attention_mask=mask, position_ids=positions).logits.detach()
        with closing(StreamedPrefixCache(model, 1, 1 << 20).install()) as cache:
            for order in ([0, 1], [1, 0]):
                got = model(
                    ids[order], attention_mask=mask[order], position_ids=positions[order]
                ).logits.detach()
                torch.testing.assert_close(got, expected[order], rtol=0, atol=0)
            model(ids, attention_mask=mask, position_ids=positions.flip(-1))
            model(ids, attention_mask=1 - mask, position_ids=positions)

            # No preceding embedding hook supplies IDs to these decoder calls.
            decoder = decoder_owner(model)
            embeds = torch.randn(*ids.shape, 64)
            first = decoder(inputs_embeds=embeds).last_hidden_state.detach()
            again = decoder(inputs_embeds=embeds.clone()).last_hidden_state.detach()
            changed = decoder(inputs_embeds=embeds + 1).last_hidden_state.detach()
            assert torch.equal(first, again)
            assert not torch.equal(first, changed)
            assert (cache.stats.hits, cache.stats.misses) == (2, 5)


def test_streamed_prefix_cache_bypasses_calls_with_kv_state(tmp_path):
    from transformers import DynamicCache

    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        model.train()
        _freeze(model, 1)
        ids = _micro_batches(1)[0]
        with closing(StreamedPrefixCache(model, 1, 1 << 20).install()) as cache:
            _stream_forward(model, ids)
            model(input_ids=ids, past_key_values=DynamicCache(), use_cache=False)
            assert (cache.stats.hits, cache.stats.misses, cache.stats.rows_stored) == (0, 1, 1)


def test_streamed_qwen_hybrid_prefix_matches_the_full_forward(tmp_path, monkeypatch):
    import torch
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    torch.manual_seed(7)
    cfg = AutoConfig.for_model(
        "qwen3_5_text",
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        full_attention_interval=2,
        tie_word_embeddings=False,
        max_position_embeddings=64,
    )
    weights = str(tmp_path / "weights")
    AutoModelForCausalLM.from_config(cfg).save_pretrained(weights)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="qwen3")
    config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0,
        target_modules=["q_proj", "v_proj", "in_proj_qkv", "out_proj"],
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=config,
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    with closing(runtime):
        model.train()
        _freeze(model, 2)
        ids = torch.cat(_micro_batches(2))
        mask = torch.ones_like(ids)
        mask[0, -2:] = 0
        expected = _stream_forward(model, ids, attention_mask=mask)
        with closing(StreamedPrefixCache(model, 2, 1 << 20).install()) as cache:
            assert torch.equal(_stream_forward(model, ids, attention_mask=mask), expected)
            loads = _record_loads(runtime, monkeypatch)
            assert torch.equal(_stream_forward(model, ids, attention_mask=mask), expected)
            assert loads and min(loads) >= 2
            assert (cache.stats.hits, cache.stats.misses) == (1, 1)


def test_progressive_cache_raises_cutoff_and_preserves_old_rows(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path, n_layers=4)
    with closing(runtime):
        model.train()
        _freeze(model, 1)
        first, second = _micro_batches(2)
        with closing(StreamedPrefixCache(model, 1, first.numel() * 64 * 4).install()) as cache:
            expected = _stream_forward(model, first)
            _stream_forward(model, second)
            assert cache.stats.rows_stored == 1
            assert cache.stats.rows_not_stored == 1
            assert cache.stats.bytes_stored == first.numel() * 64 * 4
            _freeze(model, 2)
            cache.raise_cutoff(2)
            loads = _record_loads(runtime, monkeypatch)
            assert torch.equal(_stream_forward(model, first), expected)
            assert loads and all(index >= 1 for index in loads)
            assert cache.stats.partial == 1
            assert torch.equal(_stream_forward(model, first), expected)
            assert cache.stats.hits == 1
            assert cache.stats.rows_stored == 2


def test_l2l_uses_partial_rows_and_capture_does_not_populate_the_cache(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.layer_stream_runtime import decoder_owner
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        model.train()
        _load("l2l_schedule").make_non_vacuous(model)
        _freeze(model, 2)
        rows = _micro_batches(3)
        owner = decoder_owner(model)
        visits = []
        loads = _record_loads(runtime, monkeypatch)

        def record(_layer, args):
            visits.append(args[0].shape[1])

        with closing(StreamedPrefixCache(model, 2, 1 << 20).install()) as cache:
            step = L2LStep(model, runtime, capture_layer_call(model, rows[2], draws=False))
            assert cache.stats.rows_stored == 0
            handle = owner.layers[0].register_forward_pre_hook(record)
            try:
                for window, expected_visits in (
                    (rows[:2], 2),
                    (rows[1:], 1),
                    (rows[:2], 0),
                ):
                    visits.clear()
                    loads.clear()
                    model.zero_grad(set_to_none=True)
                    with closing(_store(2)) as store:
                        cached = step.run(window, store)
                    assert len(visits) == expected_visits
                    if expected_visits == 0:
                        assert all(index >= 2 for index in loads)
                    else:
                        assert any(index < 2 for index in loads)
                    cached_grads = {
                        name: param.grad.clone()
                        for name, param in model.named_parameters() if param.grad is not None
                    }
                    model.zero_grad(set_to_none=True)
                    with cache.bypass(), closing(_store(2)) as store:
                        plain = step.run(window, store)
                    assert all(torch.equal(a, b) for a, b in zip(cached, plain, strict=True))
                    for name, param in model.named_parameters():
                        if name in cached_grads:
                            assert torch.equal(param.grad, cached_grads[name])
                assert (cache.stats.l2l_hits, cache.stats.l2l_misses) == (1, 2)
                assert cache.stats.rows_stored == 3
            finally:
                handle.remove()


def test_l2l_reuses_an_older_cutoff_after_progressive_freeze(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    # The resumed suffix must span more layers than the pool has slots.
    model, runtime = build_streamed(tmp_path, n_layers=4)
    with closing(runtime):
        model.train()
        _freeze(model, 1)
        rows = _micro_batches(2)
        step = L2LStep(model, runtime, capture_layer_call(model, rows[0], draws=False))
        with closing(StreamedPrefixCache(model, 1, 1 << 20).install()) as cache:
            with closing(_store(2)) as store:
                plain = step.run(rows, store)
            model.zero_grad(set_to_none=True)
            _freeze(model, 2)
            cache.raise_cutoff(2)
            loads = _record_loads(runtime, monkeypatch)
            with closing(_store(2)) as store:
                promoted = step.run(rows, store)
            assert loads and min(loads) >= 1
            assert all(torch.equal(a, b) for a, b in zip(plain, promoted, strict=True))
            assert cache.stats.l2l_partial == 1
            assert cache.stats.rows_stored == 4


@pytest.mark.parametrize(
    "mode, message",
    [
        (
            "reference",
            "training.frozen_prefix_cache cannot be combined with an L2L reference pass.",
        ),
        ("mbr", "training.frozen_prefix_cache cannot be combined with route-preserving MBR."),
    ],
)
def test_incompatible_l2l_walks_refuse_the_installed_cache(tmp_path, mode, message):
    from soup_cli.utils.l2l_step import L2LStep, ReferencePass, capture_layer_call
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        model.train()
        _freeze(model, 2)
        rows = _micro_batches(1)
        step = L2LStep(model, runtime, capture_layer_call(model, rows[0], draws=False))
        with (
            closing(StreamedPrefixCache(model, 2, 1 << 20).install()),
            closing(_store(1)) as store,
            pytest.raises(ValueError) as caught,
        ):
            options = (
                {"reference": ReferencePass(head=lambda *_: None)}
                if mode == "reference" else {"route_preserving_mbr": True}
            )
            step.run(rows, store, **options)
        assert str(caught.value) == message


def test_a_frozen_weight_change_refuses_reuse(tmp_path):
    import torch

    from soup_cli.utils.layer_stream_runtime import decoder_owner
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        model.train()
        _freeze(model, 1)
        ids = _micro_batches(1)[0]
        owner = decoder_owner(model)
        with closing(StreamedPrefixCache(model, 1, 1 << 20).install()):
            _stream_forward(model, ids)
            frozen = next(param for param in owner.layers[0].parameters() if not param.is_meta)
            with torch.no_grad():
                frozen.add_(1)
            with pytest.raises(RuntimeError) as caught:
                _stream_forward(model, ids)
            assert str(caught.value) == (
                "training.frozen_prefix_cache detected a changed frozen parameter or adapter "
                "state. Restart training to rebuild the streamed cache."
            )
            assert runtime.prefetcher.prime_start == 0


def test_a_trainable_embedding_refuses_install(tmp_path):
    from soup_cli.utils.layer_stream_runtime import decoder_owner
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        _freeze(model, 1)
        embedding = decoder_owner(model).embed_tokens
        next(embedding.parameters()).requires_grad_(True)
        with pytest.raises(ValueError) as caught:
            StreamedPrefixCache(model, 1, 1 << 20)
        assert str(caught.value) == (
            "training.frozen_prefix_cache requires frozen input embeddings and "
            "all prefix parameters frozen."
        )


def test_attention_dropout_refuses_install(tmp_path):
    from soup_cli.utils.layer_stream_runtime import decoder_owner
    from soup_cli.utils.streamed_prefix_cache import StreamedPrefixCache

    model, runtime = build_streamed(tmp_path)
    with closing(runtime):
        _freeze(model, 1)
        decoder_owner(model).config.attention_dropout = 0.1
        with pytest.raises(ValueError) as caught:
            StreamedPrefixCache(model, 1, 1 << 20)
        assert str(caught.value) == (
            "training.frozen_prefix_cache requires zero dropout in the frozen prefix "
            "(including attention_dropout and training.lora.dropout)."
        )


def test_host_limit_preserves_headroom(monkeypatch):
    from soup_cli.utils import layer_stream
    from soup_cli.utils.streamed_prefix_cache import resolve_cache_bytes

    gib = 1 << 30
    monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: 4 * gib)
    assert resolve_cache_bytes(None) == gib
    assert resolve_cache_bytes(0.25) == gib // 4
    monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: None)
    assert resolve_cache_bytes(None) == 8 * gib


@pytest.mark.parametrize("l2l", [False, True])
def test_streamed_trainer_activates_the_cache_at_progressive_freeze(
    tmp_path, monkeypatch, l2l
):
    import torch
    from rich.console import Console
    from transformers import Trainer, TrainingArguments

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class
    from soup_cli.utils.layer_select import attach_layer_select_callback
    from tests.test_frozen_prefix_config import streamed_config

    hermetic_ram(monkeypatch, tmp_path)
    model, runtime = build_streamed(tmp_path)
    stream = StringIO()
    with closing(runtime):
        ids = _micro_batches(2)
        rows = [{"input_ids": row[0], "labels": row[0]} for row in ids]
        cls = make_l2l_trainer_class(Trainer) if l2l else Trainer
        trainer = cls(
            model=model,
            args=TrainingArguments(
                output_dir=str(tmp_path / "train"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                num_train_epochs=3,
                use_cpu=True,
                report_to=[],
                save_strategy="no",
                disable_tqdm=True,
            ),
            train_dataset=rows,
        )
        if l2l:
            attach_l2l_state(trainer, runtime, dtype=torch.float32)
        cfg = SoupConfig.model_validate(
            streamed_config(
                l2l=l2l,
                frozen_prefix_cache_gib=1 / 1024,
                lora_layers={"method": "progressive", "freeze_step": 1, "freeze_below": 2},
            )
        )
        cb = attach_layer_select_callback(
            trainer,
            cfg.training,
            trainer.args.output_dir,
            Console(file=stream, force_terminal=True, width=200),
        )
        trainer.add_callback(cb)
        try:
            trainer.train()
            assert "Frozen-prefix cache enabled: 2 frozen layers" in strip_ansi(stream.getvalue())
            assert "Frozen-prefix cache: hits=" in strip_ansi(stream.getvalue())
            assert cb.selection.cache["l2l_hits" if l2l else "hits"] >= 1
        finally:
            cb.close_prefix_cache()
            if l2l:
                trainer._soup_l2l.close()
