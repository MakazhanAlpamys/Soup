"""Independent CPU oracles for a proposed AutoDistill student loss slice."""

from __future__ import annotations

import math

import pytest

from soup_cli.autodistill.capture import build_teacher_expert_capture_token
from soup_cli.autodistill.contract import ProbabilityPolicy


def _capture(logits=(0.2, -0.5, 1.4, 0.8), *, top_k=1, temperature=1.0, target=0):
    policy = ProbabilityPolicy(
        name="topk_union_forced_tail.v1", top_k=top_k,
        forced_token_sources=("target", "student_sample"),
        token_id_bytes=4, log_probability_bytes=8, tail_mass_bytes=8, entropy_bytes=8,
        temperature=temperature, renormalize_selected=False,
    )
    return build_teacher_expert_capture_token(
        example_id="example-1", position=0, context_token_ids=(1,),
        target_token_id=target, teacher_logits=logits,
        vocab_size=len(logits), probability_policy=policy,
    )


def _loss(logits, captures, *, chunk=2, checkpoint=False, ce=0.3, kd=0.7):
    from soup_cli.autodistill.student_loss import captured_token_loss

    return captured_token_loss(
        logits, captures, ce_weight=ce, kd_weight=kd,
        token_chunk_size=chunk, checkpoint_chunks=checkpoint,
    )


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_sparse_loss_and_gradients_match_an_independent_grouped_probability_oracle(
    temperature, device,
):
    import torch

    row = _capture(temperature=temperature)
    logits = torch.tensor([[1.1, -0.8, 0.1, 0.3]], dtype=torch.float64,
                          device=device, requires_grad=True)
    result = _loss(logits, [row])

    # The oracle sums student probability over omitted IDs, rather than using
    # the implementation's logsumexp calculation for a coarse tail bucket.
    q = torch.softmax(logits.detach()[0] / temperature, dim=0).tolist()
    p = dict(zip(row.selected_token_ids, map(math.exp, row.teacher_log_probabilities)))
    omitted = set(range(4)) - set(p)
    q_tail = sum(q[index] for index in omitted)
    expected_kd = sum(prob * math.log(prob / q[index]) for index, prob in p.items())
    expected_kd += row.tail_mass * math.log(row.tail_mass / q_tail)
    q_ce = torch.softmax(logits.detach()[0], dim=0).tolist()
    expected_ce = -math.log(q_ce[row.target_token_id])
    expected_gradient = [
        0.3 * (q_ce[index] - (index == row.target_token_id))
        + 0.7 / temperature * (
            q[index] - (p[index] if index in p else row.tail_mass * q[index] / q_tail)
        )
        for index in range(4)
    ]
    assert result.token_count == 1
    assert result.ce.item() == pytest.approx(expected_ce, abs=1e-12)
    assert result.kd.item() == pytest.approx(expected_kd, abs=1e-12)
    assert result.total.item() == pytest.approx(0.3 * expected_ce + 0.7 * expected_kd, abs=1e-12)
    result.total.backward()
    assert logits.grad[0].tolist() == pytest.approx(expected_gradient, abs=1e-12)


@pytest.mark.parametrize("chunk", [1, 2, 8])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_full_vocabulary_matches_dense_loss_and_gradients_with_an_uneven_tail(chunk, checkpoint):
    import torch
    import torch.nn.functional as functional

    teacher = torch.tensor([
        [1.0, -0.4, 0.7], [-0.1, 1.4, -0.7], [0.5, -0.8, 0.1],
        [0.2, 0.4, 0.9], [0.6, -0.3, -0.8],
    ], dtype=torch.float64)
    rows = [_capture(values.tolist(), top_k=3, temperature=2.0, target=index % 3)
            for index, values in enumerate(teacher)]
    logits = torch.linspace(-1.0, 1.5, 15, dtype=torch.float64).reshape(5, 3).requires_grad_()
    dense_logits = logits.detach().clone().requires_grad_()
    result = _loss(logits, rows, chunk=chunk, checkpoint=checkpoint)
    log_p = functional.log_softmax(teacher / 2.0, dim=-1)
    log_q = functional.log_softmax(dense_logits / 2.0, dim=-1)
    dense_kd = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
    targets = torch.tensor([index % 3 for index in range(5)])
    dense_ce = functional.cross_entropy(dense_logits, targets)
    expected = 0.3 * dense_ce + 0.7 * dense_kd
    torch.testing.assert_close(result.total, expected, rtol=1e-12, atol=1e-12)
    result.total.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, dense_logits.grad, rtol=1e-12, atol=1e-12)


def test_sparse_objective_has_finite_difference_gradients():
    import torch

    logits = torch.tensor([[0.4, -0.1, 0.8, -0.6]], dtype=torch.float64, requires_grad=True)
    row = _capture(temperature=1.7)
    assert torch.autograd.gradcheck(lambda values: _loss(values, [row]).total, (logits,))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
def test_tiny_student_tail_is_computed_without_subtracting_rounded_selected_mass(dtype_name):
    import torch

    row = _capture((1.0, 0.0, -0.5, -1.0), target=0)
    logits = torch.tensor([[1000.0, -1000.0, -1001.0, -1002.0]],
                          dtype=getattr(torch, dtype_name), requires_grad=True)
    result = _loss(logits, [row], ce=0.0, kd=1.0)
    assert torch.isfinite(result.total)
    assert result.total.item() > 100.0
    result.total.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0].item() > 0.0
    assert logits.grad[0, 1].item() < 0.0


def test_identical_sparse_distributions_have_zero_kd_and_stationary_gradients():
    import torch

    values = (0.2, -0.5, 1.4, 0.8)
    row = _capture(values, temperature=1.5)
    logits = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    result = _loss(logits, [row], ce=0.0, kd=1.0)
    assert result.kd.item() == pytest.approx(0.0, abs=1e-14)
    result.total.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("dtype_name,offset", [("float32", 2**27), ("float64", 2**55)])
@pytest.mark.parametrize("values", [(0.0, 0.0, 0.0, 0.0), (0.0, 32.0, -16.0, 48.0)])
def test_large_common_logit_offsets_preserve_loss_and_gradients(dtype_name, offset, values):
    import torch

    base = torch.tensor([values], dtype=getattr(torch, dtype_name), requires_grad=True)
    shifted = (base.detach() + offset).requires_grad_()
    assert torch.equal(shifted.detach() - offset, base.detach())
    row = _capture(temperature=1.3)
    expected = _loss(base, [row], checkpoint=True)
    actual = _loss(shifted, [row], checkpoint=True)
    tolerance = 1e-5 if dtype_name == "float32" else 1e-12
    torch.testing.assert_close(actual.total, expected.total, rtol=0.0, atol=tolerance)
    if values == (0.0, 0.0, 0.0, 0.0):
        assert actual.ce.item() == pytest.approx(math.log(4), abs=tolerance)
    expected.total.backward()
    actual.total.backward()
    torch.testing.assert_close(shifted.grad, base.grad, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("ce,kd", [(1.0, 0.0), (0.0, 1.0), (0.4, 0.6)])
def test_microbatch_sums_use_the_global_token_denominator(ce, kd):
    import torch

    rows = [_capture(target=index % 4, temperature=1.3) for index in range(5)]
    logits = torch.linspace(-0.5, 1.2, 20, dtype=torch.float64).reshape(5, 4).requires_grad_()
    whole = _loss(logits, rows, ce=ce, kd=kd)
    expected = torch.autograd.grad(whole.total, logits)[0]
    split = logits.detach().clone().requires_grad_()
    # Unequal microbatches: averaging these means without weighting must fail.
    for start, stop in [(0, 2), (2, 5)]:
        part = _loss(split[start:stop], rows[start:stop], ce=ce, kd=kd)
        (part.total * part.token_count / len(rows)).backward()
    torch.testing.assert_close(split.grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"ce": -1.0}, {"kd": float("nan")}, {"ce": True}, {"kd": float("inf")},
    {"ce": 0.0, "kd": 0.0}, {"chunk": 0}, {"chunk": True}, {"chunk": 1.5},
    {"checkpoint": "yes"},
])
def test_invalid_objective_options_are_refused(kwargs):
    import torch

    with pytest.raises((TypeError, ValueError)):
        _loss(torch.zeros(1, 4), [_capture()], **kwargs)


@pytest.mark.parametrize("case", ["empty", "count", "vocab", "rank", "integer", "nan"])
def test_invalid_logits_are_refused(case):
    import torch

    shapes = {"empty": (0, 4), "count": (2, 4), "vocab": (1, 5), "rank": (4,)}
    logits = torch.zeros(shapes.get(case, (1, 4)),
                         dtype=torch.int64 if case == "integer" else torch.float32)
    if case == "nan":
        logits[0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        _loss(logits, [] if case == "empty" else [_capture()])


def test_rollouts_and_unvalidated_copied_rows_are_refused():
    import torch

    row = _capture()
    rollout = row.model_copy(update={"trajectory_kind": "student_rollout",
                                     "student_sampled_token_id": row.target_token_id})
    with pytest.raises(ValueError, match="teacher_expert"):
        _loss(torch.zeros(1, 4), [rollout])
    corrupt = row.model_copy(update={"tail_mass": 0.99})
    with pytest.raises(ValueError, match="mass"):
        _loss(torch.zeros(1, 4), [corrupt])


def test_a_full_vocabulary_row_cannot_assign_mass_to_an_empty_tail():
    import torch

    # This residual is within the artifact's probability-sum tolerance, but
    # cannot have any corresponding student probability when every ID is present.
    row = _capture(top_k=4).model_copy(update={"tail_mass": 1e-12})
    with pytest.raises(ValueError, match="zero tail"):
        _loss(torch.zeros(1, 4), [row])


@pytest.mark.parametrize("case", ["temperature", "difference", "weight"])
def test_nonrepresentable_objectives_raise_instead_of_returning_nan_or_infinity(case):
    import torch

    # A uniform teacher remains representable at the tiny temperature; this
    # exercises overflow in the student objective rather than capture creation.
    row = (_capture((0.0, 0.0, 0.0, 0.0), temperature=1e-50)
           if case == "temperature" else _capture())
    logits = torch.tensor([[1.0, -1.0, 0.0, 0.5]])
    if case == "difference":
        logits[0, 2] = 3e38
        logits[0, 0] = -3e38
    with pytest.raises(ValueError, match="not finite"):
        _loss(logits, [row], ce=1e300 if case == "weight" else 0.3)


def test_coercible_model_copy_fields_cannot_bypass_the_typed_row_contract():
    import torch

    row = _capture().model_copy(update={"target_token_id": True})
    with pytest.raises(ValueError, match="integer"):
        _loss(torch.zeros(1, 4), [row])


def test_module_import_does_not_load_a_runtime_in_a_fresh_process():
    import os
    import subprocess
    import sys

    script = (
        "import sys; import soup_cli.autodistill.student_loss; "
        "assert not {'torch','transformers','mlx','trl','peft'}.intersection(sys.modules)"
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    process = subprocess.run([sys.executable, "-c", script], env=environment,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr


def test_checkpointing_reduces_saved_objective_storage_excluding_input_logits():
    import weakref

    import torch

    values = tuple(index / 64 for index in range(64))
    rows = [_capture(values, top_k=2) for _ in range(8)]

    def retained_bytes(checkpoint):
        logits = torch.linspace(-1, 1, 512).reshape(8, 64).requires_grad_()
        input_pointer = logits.untyped_storage().data_ptr()
        packed_references = []

        class PackedTensor:
            def __init__(self, tensor):
                self.tensor = tensor.detach()

        def pack(tensor):
            packed = PackedTensor(tensor)
            packed_references.append(weakref.ref(packed))
            return packed

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda packed: packed.tensor):
            result = _loss(logits, rows, chunk=2, checkpoint=checkpoint)
        # Hooks also see transient saves whose graph is discarded (for example
        # during finite-value validation). Only live holders count as retained;
        # recording every pointer ever seen would make allocator reuse look like
        # a memory saving even if checkpointing were disabled on both arms.
        storages = {}
        for reference in packed_references:
            packed = reference()
            if packed is not None:
                storage = packed.tensor.untyped_storage()
                if storage.data_ptr() != input_pointer:
                    storages[storage.data_ptr()] = storage.nbytes()
        result.total.backward()
        assert torch.isfinite(logits.grad).all()
        return sum(storages.values())

    assert retained_bytes(True) < retained_bytes(False)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_real_causal_student_optimizer_update_matches_dense_teacher_objective(lora, device):
    import copy

    import torch
    import torch.nn.functional as functional
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(19)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        attention_dropout=0.0,
    )).double().eval().to(device)
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ))
    reference = copy.deepcopy(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    contexts = [(1, 3), (2, 5, 6), (1, 4)]
    targets = [7, 4, 2]
    teacher_logits = torch.linspace(-0.8, 1.2, 24, dtype=torch.float64).reshape(3, 8)
    rows = [_capture(values.tolist(), top_k=8, temperature=1.7, target=target)
            .model_copy(update={"context_token_ids": context})
            for context, target, values in zip(contexts, targets, teacher_logits)]
    teacher_logits = teacher_logits.to(device)

    def predict(student):
        return torch.stack([
            student(input_ids=torch.tensor([context], device=device)).logits[0, -1]
            for context in contexts
        ])

    with torch.no_grad():
        initial_predictions = predict(model).clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.03)
    dense_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.03)
    actual = _loss(predict(model), rows, chunk=2, checkpoint=True)
    dense_logits = predict(reference)
    log_p = functional.log_softmax(teacher_logits / 1.7, dim=-1)
    log_q = functional.log_softmax(dense_logits / 1.7, dim=-1)
    expected = 0.3 * functional.cross_entropy(dense_logits, torch.tensor(targets, device=device))
    expected = expected + 0.7 * (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
    torch.testing.assert_close(actual.total, expected, rtol=1e-6, atol=1e-8)
    actual.total.backward()
    expected.backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad,
                                       rtol=2e-5, atol=1e-8)
        else:
            assert parameter.grad is None, name
    optimizer.step()
    dense_optimizer.step()
    changed = []
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, reference_parameters[name], rtol=2e-5, atol=1e-8)
        if not torch.equal(parameter, before[name]):
            changed.append(name)
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[name]), name
    assert changed
    if lora:
        assert all("lora_" in name for name in changed), changed
    with torch.no_grad():
        updated = predict(model)
        assert not torch.equal(updated, initial_predictions)
        torch.testing.assert_close(updated, predict(reference), rtol=2e-5, atol=1e-8)
