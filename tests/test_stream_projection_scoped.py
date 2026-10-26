"""Narrowing dispatch must preserve cache mutation checks and every update."""
from unittest.mock import patch

import pytest

from tests.test_stream_projection_retention import _helper, _mistral_stream


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_projection_scope_matches_native_and_layer_scope_with_fewer_dispatches(tmp_path, quant):
    import torch
    import torch.utils.checkpoint as checkpoint

    from soup_cli.utils.layer_stream_runtime import set_stream_recompute, streamed_decoder_layers

    helper = _helper()
    results, dispatches = {}, {}
    for mode in ("native", "layer", "projection"):
        model, runtime = _mistral_stream(tmp_path / mode, quant)
        controller = None
        handles = []
        seen = {"mlp": []}
        calls = []
        original_contexts = checkpoint.create_selective_checkpoint_contexts

        def counted_contexts(policy, **kwargs):
            def counted(ctx, op, *args, **options):
                calls.append(str(op))
                return policy(ctx, op, *args, **options)
            return original_contexts(counted, **kwargs)

        def scope_probe(name):
            def hook(module, args):
                from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
                seen[name].append(any(type(m).__name__ in (
                    "_CachingTorchDispatchMode", "_CachedTorchDispatchMode",
                ) for m in _get_current_dispatch_mode_stack()))
            return hook

        try:
            set_stream_recompute(model, False, retain=1)
            layers = streamed_decoder_layers(model)
            modules = []
            model.apply(modules.append)
            original_forwards = [module.forward for module in modules]
            with patch.object(checkpoint, "create_selective_checkpoint_contexts", counted_contexts):
                if mode != "native":
                    controller = helper.install_subset(model, [1,2], cache_scope=mode)
                for layer in layers[1:3]:
                    # MLP operations must remain outside the narrow SAC scope.
                    handles.append(layer.inner.mlp.register_forward_pre_hook(scope_probe("mlp")))
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=1e-4,
                )
                records = []
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = []
                    for length in (8,5):
                        ids = torch.randint(
                            0,64,(1,length),generator=torch.Generator().manual_seed(5),
                        )
                        output = model(input_ids=ids, labels=ids)
                        outputs.extend((output.logits.detach().clone(),output.loss.detach().clone()))
                        if controller:
                            assert controller.cache_bytes() == sum(helper.projection_bytes(
                                layers[i],batch=1,seq=length,element_bytes=4) for i in (1,2))
                            assert len(controller.cached_tensors()) == 8
                        (output.loss/2).backward()
                        del output
                    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
                    gradients = {n:p.grad.clone() for n,p in parameters.items()}
                    assert all(g.count_nonzero().item() for g in gradients.values())
                    optimizer.step()
                    records.append((outputs,gradients,{
                        n:p.detach().clone() for n,p in parameters.items()
                    }))
                results[mode] = records
                dispatches[mode] = len(calls)
                if mode == "projection":
                    assert seen['mlp'] and not any(seen['mlp'])
                    assert controller._mode.get() is None
                if mode == "layer":
                    assert seen['mlp'] and all(seen['mlp'])
                if controller:
                    assert controller.counts['saved_forward'] == 32
                    controller.remove()
                    assert [module.forward for module in modules] == original_forwards
        finally:
            for handle in handles:
                handle.remove()
            if controller:
                controller.remove()
            runtime.close()
    assert 0 < dispatches['projection'] < dispatches['layer']
    for mode in ('layer','projection'):
        for baseline,actual in zip(results['native'],results[mode]):
            assert all(torch.equal(a,b) for a,b in zip(baseline[0],actual[0]))
            for expected,obtained in zip(baseline[1:],actual[1:]):
                assert expected.keys() == obtained.keys()
                assert all(torch.equal(expected[name],obtained[name]) for name in expected)


def test_scoped_cache_mutation_is_rejected_and_context_restored(tmp_path):
    import torch

    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_recompute import _ids, _stream

    model,runtime = _stream(tmp_path,'none')
    controller = None
    try:
        set_stream_recompute(model,False,retain=1)
        controller = _helper().install_subset(model,[1,2],cache_scope='projection')
        output = model(input_ids=_ids(),labels=_ids())
        with torch.no_grad():
            controller.cached_tensors()[0].add_(1)
        with pytest.raises(RuntimeError,match='mutat'):
            output.loss.backward()
        assert controller._mode.get() is None
    finally:
        if controller:
            controller.remove()
        runtime.close()


@pytest.mark.parametrize('change', ['recompute','checkpoint'])
def test_scoped_mode_change_is_refused_after_installation(tmp_path,change):
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute, streamed_decoder_layers
    from tests.test_stream_recompute import _ids, _stream

    model,runtime = _stream(tmp_path,'none')
    controller = None
    try:
        controller = _helper().install_subset(model,[1,2],cache_scope='projection')
        if change == 'recompute':
            set_stream_recompute(model,False,retain=2)
        else:
            streamed_decoder_layers(model)[1].use_checkpoint = False
        with pytest.raises(RuntimeError,match='mode changed'):
            model(input_ids=_ids(),labels=_ids())
    finally:
        if controller:
            controller.remove()
        runtime.close()


def test_unknown_scope_refused_before_patching(tmp_path):
    from tests.test_stream_recompute import _stream

    model,runtime = _stream(tmp_path,'none')
    try:
        with pytest.raises(ValueError,match='scope'):
            _helper().install_subset(model,[1],cache_scope='typo')
    finally:
        runtime.close()
