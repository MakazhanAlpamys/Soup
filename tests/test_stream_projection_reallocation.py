"""Reallocating one retained layer is a separate, explicitly budgeted experiment."""
import copy

import pytest

from tests.test_stream_projection_retention import _helper, _layers, _mistral_stream, _plan


@pytest.mark.parametrize('residual,count', [(0, 2), (14, 2), (15, 3)])
def test_reallocation_charges_freed_layer_once_and_obeys_byte_boundary(residual, count):
    layers = _layers(2)
    native = _plan(60+residual, 2) | {'full_layer_charge': 30}
    original = copy.deepcopy(native)
    plan = _helper().reallocated_projection_plan(
        layers, native, batch=1, seq=1, element_bytes=1,
    )
    assert plan['native_k'] == 2 and plan['effective_k'] == 1
    assert plan['retained_indices'] == [3]
    assert plan['selected_indices'] == list(range(3-count, 3))
    assert plan['freed_full_layer_charge'] == 30
    assert plan['residual_bytes'] == 30+residual
    assert plan['charged_bytes'] == 15*count
    assert plan['charged_bytes']+30 <= native['budget_bytes']
    assert native == original
    assert [layer.recompute for layer in layers] == [True, True, False, False]


@pytest.mark.parametrize('native,layers', [
    ({'measured': False}, _layers()), (_plan(29, 0), _layers(0)),
    (_plan(1000, 4), _layers(4)), (_plan(), _layers(2)),
])
def test_reallocation_refuses_unavailable_or_changed_partial_auto(native, layers):
    with pytest.raises(ValueError):
        _helper().reallocated_projection_plan(layers, native, batch=1, seq=1, element_bytes=1)


@pytest.mark.parametrize('quant', ['none', 'nf4'])
def test_fewer_retained_layers_plus_qkvo_matches_all_native_tensors(tmp_path, quant):
    import torch

    from soup_cli.utils.layer_stream_runtime import set_stream_recompute, streamed_decoder_layers

    helper = _helper()
    results = []
    for mode in ('native', 'fewer', 'fewer_qkvo'):
        model, runtime = _mistral_stream(tmp_path/mode, quant)
        controller = None
        try:
            set_stream_recompute(model, False, retain=2)
            if mode != 'native':
                set_stream_recompute(model, False, retain=1)
            layers = streamed_decoder_layers(model)
            assert [layer.idx for layer in layers if not layer.recompute] == (
                [2, 3] if mode == 'native' else [3]
            )
            if mode == 'fewer_qkvo':
                controller = helper.install_subset(model, [0, 1, 2])
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=1e-4)
            records = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(5))
                output = model(input_ids=ids, labels=ids)
                record = [output.logits.detach().clone(), output.loss.detach().clone()]
                output.loss.backward()
                parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
                record.append({n: p.grad.clone() for n, p in parameters.items()})
                optimizer.step()
                record.append({n: p.detach().clone() for n, p in parameters.items()})
                records.append(record)
            results.append(records)
        finally:
            if controller:
                controller.remove()
            runtime.close()
    for candidate in results[1:]:
        for baseline, actual in zip(results[0], candidate):
            assert all(torch.equal(a, b) for a, b in zip(baseline[:2], actual[:2]))
            for expected, obtained in zip(baseline[2:], actual[2:]):
                assert expected.keys() == obtained.keys()
                assert all(torch.equal(expected[n], obtained[n]) for n in expected)
