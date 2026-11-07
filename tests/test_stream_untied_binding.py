"""Layer streaming: the decoder ``functional_call`` skips tied-name discovery.

``torch.func.functional_call`` defaults to ``tie_weights=True``, which walks every
parameter and buffer of the module on EVERY call to find tensors registered under
more than one name. A streamed decoder layer is called twice per layer per step
(forward + checkpoint recompute), and on a PEFT-wrapped Mistral layer (53 modules)
the walk costs ~104 us per call on the dev box (CPU micro-benchmark, 2026-10-14):
~6.7 ms per 32-layer step.

The walk only changes the result when a SUBSTITUTED name shares its tensor with
another registered name (``torch.nn.utils.stateless._untie_named_tensors_map``
returns a plain copy of the input otherwise). So the wrapper decides once, at
construction, and keeps ``tie_weights=True`` only for a layer that needs it.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


class _FakePool:
    def __init__(self, buffers):
        self._buffers = buffers

    def wait(self, idx):
        return self._buffers

    def slot_for(self, idx):
        return 0


class _FakePrefetcher:
    def advance(self, idx):
        return None


def _tied_inner():
    import torch.nn as nn

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)
            # One tensor, two registered names.
            self.second.weight = self.first.weight

        def forward(self, x):
            return self.second(self.first(x))

    return _Inner()


def _wrap(inner, name_map, buffers):
    from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

    return StreamedDecoderLayer(
        inner,
        0,
        _FakePool(buffers),
        _FakePrefetcher(),
        name_map=name_map,
        use_checkpoint=False,
    )


def _count_untie_calls(monkeypatch):
    import torch.nn.utils.stateless as stateless

    real = stateless._untie_named_tensors_map
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(stateless, "_untie_named_tensors_map", counting)
    return calls


class TestSubstitutedNamesAreTied:
    def test_a_substituted_name_sharing_its_tensor_is_tied(self):
        from soup_cli.utils.layer_stream_runtime import _substituted_names_are_tied

        assert _substituted_names_are_tied(_tied_inner(), ["first.weight"]) is True
        assert _substituted_names_are_tied(_tied_inner(), ["second.weight"]) is True

    def test_a_tie_outside_the_substituted_names_does_not_count(self):
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import _substituted_names_are_tied

        inner = _tied_inner()
        inner.other = nn.Linear(2, 2, bias=False)
        # torch only rewrites the mapping for GIVEN names that are tied, so a tie
        # among names the stream never substitutes leaves the result unchanged.
        assert _substituted_names_are_tied(inner, ["other.weight"]) is False

    def test_untied_module_and_empty_name_list(self):
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import _substituted_names_are_tied

        assert _substituted_names_are_tied(nn.Linear(2, 2), ["weight", "bias"]) is False
        assert _substituted_names_are_tied(_tied_inner(), []) is False

    def test_a_buffer_aliasing_a_parameter_is_a_tie(self):
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import _substituted_names_are_tied

        inner = nn.Linear(2, 2, bias=False)
        inner.register_buffer("alias", inner.weight)
        assert _substituted_names_are_tied(inner, ["weight"]) is True


class TestTiedLayerKeepsTorchSemantics:
    def test_tied_substitution_reaches_both_names(self):
        inner = _tied_inner()
        value = torch.eye(2) * 3.0
        layer = _wrap(inner, {"first.weight": "ckpt.w"}, {"ckpt.w": value})

        assert layer._tie_weights is True
        x = torch.ones(1, 2)
        with torch.no_grad():
            got = layer(x)
        # Both `first` and `second` must see the substituted weight: 3 * 3 * x.
        # With tie_weights=False, `second` would keep the original random weight.
        assert torch.equal(got, x @ value.T @ value.T)


class TestStreamedDecoderSkipsTieDiscovery:
    def test_real_streamed_layers_are_untied(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import decoder_owner
        from tests.test_v07200 import _build_streamed_cpu

        model, _, _, _ = _build_streamed_cpu(tmp_path)
        layers = decoder_owner(model).layers
        assert len(layers) == 2
        for layer in layers:
            assert layer.name_map, "the test needs a layer with substituted weights"
            assert layer._tie_weights is False

    def test_forward_and_backward_never_walk_for_ties(self, tmp_path, monkeypatch):
        from tests.test_v07200 import _build_streamed_cpu

        model, _, _, _ = _build_streamed_cpu(tmp_path)
        calls = _count_untie_calls(monkeypatch)

        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        assert calls["n"] == 0

    def test_untied_path_is_bit_identical_to_the_tie_walking_path(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import decoder_owner
        from tests.test_v07200 import _build_streamed_cpu

        model, _, _, _ = _build_streamed_cpu(tmp_path)
        layers = list(decoder_owner(model).layers)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable
        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(1))

        def run():
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            return (
                out.logits.detach().clone(),
                out.loss.detach().clone(),
                [p.grad.detach().clone() for p in trainable],
            )

        fast_logits, fast_loss, fast_grads = run()
        for layer in layers:
            layer._tie_weights = True
        slow_logits, slow_loss, slow_grads = run()

        assert torch.equal(fast_logits, slow_logits)
        assert torch.equal(fast_loss, slow_loss)
        assert len(fast_grads) == len(slow_grads)
        for fast, slow in zip(fast_grads, slow_grads):
            assert torch.equal(fast, slow)


def _one_tie_among_untied():
    import torch.nn as nn

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4, bias=False)
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight  # the one tie
            self.post = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.post(self.second(self.first(self.pre(x))))

    torch.manual_seed(0)
    resident = _Inner()
    with torch.device("meta"):
        meta = _Inner()
    return resident, meta


class TestOneTieAmongUntiedWeights:
    """Every other fixture has a single tensor group, so a detector that asked
    whether ALL groups were tied would pass them; a real layer mixes one tied pair
    with untied weights."""

    def test_the_detector_sees_the_tie(self):
        from soup_cli.utils.layer_stream_runtime import _substituted_names_are_tied

        resident, _meta = _one_tie_among_untied()
        assert (
            _substituted_names_are_tied(resident, ["pre.weight", "first.weight", "post.weight"])
            is True
        )

    def test_the_streamed_layer_matches_the_resident_one(self):
        from soup_cli.utils.layer_stream_runtime import _layer_name_map

        resident, meta = _one_tie_among_untied()
        name_map = _layer_name_map(meta)
        # named_parameters() de-duplicates, so the tied twin is not in name_map.
        assert sorted(name_map) == ["first.weight", "post.weight", "pre.weight"]
        params = dict(resident.named_parameters())
        buffers = {name: params[name].detach().clone() for name in name_map}
        layer = _wrap(meta, name_map, buffers)

        assert layer._tie_weights is True
        x = torch.randn(2, 4, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            assert torch.equal(layer(x), resident(x))
