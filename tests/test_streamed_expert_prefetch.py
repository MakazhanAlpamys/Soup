"""Same-layer forward and backward hints and numerical parity at each depth."""

import threading

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def streamed_model(tmp_path):
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_expert_stream import _lora, _shard, _tiny_deepseek_v3

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")

    def build(**options):
        return build_streamed_model(
            model_id=weights,
            shard_dir=str(tmp_path / "shards"),
            index=index,
            lora_config=_lora(),
            device="cpu",
            dtype="float32",
            buffers=2,
            pin=False,
            seed=3,
            expert_pool=True,
            grouped_expert_routes=True,
            **options,
        )

    return build


@pytest.mark.parametrize("options,enabled", [({}, False), ({"expert_prefetch": True}, True)])
def test_streamed_model_defaults_to_one_slot_when_enabled(streamed_model, options, enabled):
    _, runtime = streamed_model(**options)
    try:
        pool = runtime.expert_pool
        assert pool.prefetch_staging == 1
        assert pool.prefetch_enabled is enabled
        assert pool.stats()["prefetch"]["staging"] == 1
        assert (pool.stats()["prefetch"]["host_staging_bytes"] > 0) is enabled
        assert pool._prefetch_executor is None
    finally:
        runtime.close()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("two_reads", [False, True])
def test_streamed_model_prefetch_keeps_logits_loss_gradients_and_read_counts(
    streamed_model, depth, two_reads,
):
    model, runtime = streamed_model(
        expert_prefetch=True, expert_prefetch_staging=depth, two_read_experts=two_reads,
    )
    try:
        ids = (torch.arange(36).reshape(3, 12) * 5 + 1) % 128
        model.train()
        pool = runtime.expert_pool
        assert pool.prefetch_staging == depth

        def step(enabled):
            model.zero_grad(set_to_none=True)
            pool.prefetch_enabled = enabled
            start = pool.fetches, pool.logical_bytes_read
            output = model(input_ids=ids, labels=ids)
            output.loss.backward()
            counts = pool.fetches - start[0], pool.logical_bytes_read - start[1]
            grads = {
                name: param.grad.clone()
                for name, param in model.named_parameters() if param.grad is not None
            }
            return output.logits.detach(), output.loss.detach(), counts, grads

        want_logits, want_loss, want_counts, want_grads = step(False)
        assert pool._prefetch_executor is None and pool.prefetch_hits == 0
        got_logits, got_loss, got_counts, got_grads = step(True)
        assert pool.prefetch_hits > 0
        assert got_logits.equal(want_logits)
        assert got_loss.equal(want_loss)
        assert got_counts == want_counts and all(count > 0 for count in got_counts)
        assert got_grads.keys() == want_grads.keys()
        assert any(grad.count_nonzero().item() for grad in want_grads.values())
        assert all(got_grads[key].equal(want_grads[key]) for key in want_grads)
    finally:
        runtime.close()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("gemm", ["native", "segmented"])
def test_hints_follow_each_layer_forward_and_backward(tmp_path, monkeypatch, depth, gemm):
    from soup_cli.utils.expert_shard import ExpertShardReader
    from tests.test_streamed_experts_pool import _experts, _inputs, _store, _streamed

    weights = _experts()
    reader, pool = _store(tmp_path, weights, prefetch=True, prefetch_staging=depth)
    second = ExpertShardReader(
        reader.path, count=reader.count, shapes=reader.shapes, quant_specs={}, dtype=reader.dtype,
        direct=False,
    )
    try:
        modules = [
            _streamed(
                weights, source, pool, layer_idx=layer, two_reads=True,
                grouped_expert_routes=True, expert_gemm=gemm,
            )
            for layer, source in enumerate((reader, second))
        ]
        visits = ([0, 2, 3, 5], [1, 3, 4, 6])
        hidden, _, scales, grad = _inputs(tokens=8, top_k=1)
        indices = [torch.tensor(visit * 2).reshape(-1, 1) for visit in visits]
        original = pool.prefetch
        hints = []
        reads = []

        def record(source, expert):
            hints.append((source, expert))
            original(source, expert)

        monkeypatch.setattr(pool, "prefetch", record)

        def record_reads(source):
            read_host = source.read_host

            def read(expert, staging):
                worker = threading.current_thread().name.startswith("expert-prefetch")
                reads.append((source, expert, worker))
                return read_host(expert, staging)

            monkeypatch.setattr(source, "read_host", read)

        record_reads(reader)
        record_reads(second)

        def run(enabled):
            pool.prefetch_enabled = enabled
            state = hidden.clone().requires_grad_()
            routing = [scales.clone().requires_grad_() for _ in modules]
            result = state
            for module, index, scale in zip(modules, indices, routing):
                result = module(result, index, scale, group_sizes=[4, 4])
            forward = hints.copy()
            hints.clear()
            grads = torch.autograd.grad(result, (state, *routing), grad)
            backward = hints.copy()
            hints.clear()
            return result, grads, forward, backward

        want, want_grads, no_forward, no_backward = run(False)
        sweeps = (
            (reader, visits[0]), (second, visits[1]),
            (second, visits[1][::-1]), (reader, visits[0][::-1]),
        )
        assert reads == [(source, expert, False) for source, visit in sweeps for expert in visit]
        reads.clear()
        got, got_grads, forward, backward = run(True)
        assert no_forward == no_backward == []
        assert got.equal(want)
        assert all(actual.equal(expected) for actual, expected in zip(got_grads, want_grads))
        assert all(grad.count_nonzero().item() for grad in want_grads)
        assert reads == [
            (source, expert, position > 0)
            for source, visit in sweeps
            for position, expert in enumerate(visit)
        ]

        def ahead(source, order):
            return [
                (source, expert)
                for i in range(len(order))
                for expert in order[i + 1:i + 1 + depth]
            ]

        assert forward == ahead(reader, visits[0]) + ahead(second, visits[1])
        assert backward == ahead(second, visits[1][::-1]) + ahead(reader, visits[0][::-1])
    finally:
        pool.close()
        second.close()
        reader.close()
