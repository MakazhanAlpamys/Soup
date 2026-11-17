"""Regression coverage for issue #2895's DAPO global-token normalization."""

import logging

import pytest


def _torch():
    return pytest.importorskip("torch")


def _real_trainer(tmp_path, *, accumulation=2, beta=0.0, trainer_base=None):
    torch = _torch()
    transformers = pytest.importorskip("transformers")
    from soup_cli.trainer.grpo import make_grpo_trainer_variant

    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, **kwargs):
            return self.scale

    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        disable_tqdm=True,
        gradient_accumulation_steps=accumulation,
        per_device_train_batch_size=2,
    )
    trainer_class = make_grpo_trainer_variant(
        trainer_base or transformers.Trainer, "dapo"
    )
    trainer = trainer_class(model=ScalarModel(), args=args)
    # TRL explicitly disables this so Trainer performs exactly one GA division.
    trainer.model_accepts_loss_kwargs = False
    trainer.beta = beta
    trainer.current_gradient_accumulation_steps = accumulation
    return trainer


def _mask(torch, active_counts):
    width = max(10, max(active_counts))
    return torch.tensor(
        [[1.0] * count + [0.0] * (width - count) for count in active_counts]
    )


def _inputs(
    torch,
    model,
    active_counts,
    *,
    global_count,
    tool_mask=None,
    reference=None,
    advantages=None,
):
    mask = _mask(torch, active_counts)
    inputs = {
        "per_token_logps": model.scale.expand_as(mask),
        "old_per_token_logps": torch.zeros_like(mask),
        "advantages": advantages if advantages is not None else torch.tensor([1.0, -1.0]),
        "completion_mask": mask,
    }
    if global_count is not _MISSING:
        inputs["num_items_in_batch"] = global_count
    if tool_mask is not None:
        inputs["tool_mask"] = tool_mask
    if reference is not None:
        inputs["ref_per_token_logps"] = reference
    return inputs


_MISSING = object()


def _dapo_sum(torch, logp_new, logp_old, advantages, mask):
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, min=0.8, max=1.28)
    per_token_loss = -torch.minimum(
        ratio * advantages.unsqueeze(-1),
        clipped * advantages.unsqueeze(-1),
    )
    return (per_token_loss * mask).sum()


def _training_step(trainer, inputs, *, item_count):
    return trainer.training_step(
        trainer.model,
        inputs,
        num_items_in_batch=item_count,
    )


def test_unequal_microbatches_match_global_dapo_gradient_via_trainer(
    tmp_path,
):
    """The actual Transformers training_step must accumulate a global-token mean."""
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2)
    model = trainer.model
    masks = [_mask(torch, [2, 1]), _mask(torch, [10, 1])]
    advantages = torch.tensor([1.0, -1.0])

    model.zero_grad(set_to_none=True)
    step_losses = []
    for mask in masks:
        inputs = {
            "per_token_logps": model.scale.expand_as(mask),
            "old_per_token_logps": torch.zeros_like(mask),
            "advantages": advantages,
            "completion_mask": mask,
            "num_items_in_batch": torch.tensor(14),
        }
        step_losses.append(_training_step(trainer, inputs, item_count=14))
    actual_gradient = model.scale.grad.detach().clone()
    expected_step_losses = [
        _dapo_sum(
            torch,
            model.scale.detach().expand_as(mask),
            torch.zeros_like(mask),
            advantages,
            mask,
        )
        / 14
        / 2
        for mask in masks
    ]
    torch.testing.assert_close(torch.stack(step_losses), torch.stack(expected_step_losses))

    reference_scale = torch.nn.Parameter(model.scale.detach().clone())
    full_mask = torch.cat(masks)
    reference_logp = reference_scale.expand_as(full_mask)
    full_advantages = advantages.repeat(2)
    reference_loss = (
        _dapo_sum(
            torch,
            reference_logp,
            torch.zeros_like(full_mask),
            full_advantages,
            full_mask,
        )
        / 14
        / 2
    )
    expected_gradient = torch.autograd.grad(reference_loss, reference_scale)[0]

    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("active_counts", "accumulation"),
    [([2, 1], 1), ([3, 3], 2)],
    ids=["one-accumulation-step", "equal-token-counts"],
)
def test_global_dapo_preserves_single_step_and_equal_count_gradients(
    tmp_path, active_counts, accumulation
):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=accumulation)
    model = trainer.model
    mask = _mask(torch, active_counts)
    generation_count = int(mask.sum().item()) * accumulation
    advantages = torch.tensor([1.0, 2.0])
    model.zero_grad(set_to_none=True)

    step_losses = []
    for _ in range(accumulation):
        inputs = {
            "per_token_logps": model.scale.expand_as(mask),
            "old_per_token_logps": torch.zeros_like(mask),
            "advantages": advantages,
            "completion_mask": mask,
            "num_items_in_batch": torch.tensor(generation_count),
        }
        step_losses.append(
            _training_step(trainer, inputs, item_count=generation_count)
        )
    actual_gradient = model.scale.grad.detach().clone()
    expected_step_loss = (
        _dapo_sum(
            torch,
            model.scale.detach().expand_as(mask),
            torch.zeros_like(mask),
            advantages,
            mask,
        )
        / generation_count
        / accumulation
    )
    torch.testing.assert_close(torch.stack(step_losses), expected_step_loss.expand(accumulation))

    reference_scale = torch.nn.Parameter(model.scale.detach().clone())
    reference_logp = reference_scale.expand_as(mask).repeat(accumulation, 1)
    reference_mask = mask.repeat(accumulation, 1)
    reference_advantages = advantages.repeat(accumulation)
    reference_loss = (
        _dapo_sum(
            torch,
            reference_logp,
            torch.zeros_like(reference_mask),
            reference_advantages,
            reference_mask,
        )
        / generation_count
        / accumulation
    )
    expected_gradient = torch.autograd.grad(reference_loss, reference_scale)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("global_count", "expected_count"),
    [(14, 14), (14.0, 14.0), ("scalar_tensor", 14)],
)
def test_numeric_counts_use_global_normalizer(tmp_path, global_count, expected_count):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2)
    if global_count == "scalar_tensor":
        global_count = torch.tensor(14)
    inputs = _inputs(
        torch,
        trainer.model,
        [2, 1],
        global_count=global_count,
    )

    result = trainer._compute_variant_loss(trainer.model, inputs)
    expected = _dapo_sum(
        torch,
        inputs["per_token_logps"],
        inputs["old_per_token_logps"],
        inputs["advantages"],
        inputs["completion_mask"],
    ) / expected_count
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing", "num_items_in_batch is missing"),
        ("none", "num_items_in_batch is None"),
        ("boolean", "num_items_in_batch is boolean"),
        ("boolean_tensor", "tensor must have a real numeric dtype"),
        ("nonnumeric", "must be an int, float"),
        ("vector_tensor", "must be zero-dimensional"),
        ("nan_float", "num_items_in_batch is not finite"),
        ("infinite_float", "num_items_in_batch is not finite"),
        ("zero_int", "num_items_in_batch is not positive"),
        ("negative_float", "num_items_in_batch is not positive"),
        ("nan_tensor", "num_items_in_batch is not finite"),
        ("infinite_tensor", "num_items_in_batch is not finite"),
        ("zero_tensor", "num_items_in_batch is not positive"),
        ("negative_tensor", "num_items_in_batch is not positive"),
    ],
)
def test_invalid_count_warns_once_and_uses_stock_loss(
    tmp_path, caplog, case, reason
):
    torch = _torch()
    transformers = pytest.importorskip("transformers")

    class StockLossTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            self.stock_loss_calls += 1
            loss = model.scale * inputs["stock_loss_multiplier"]
            return (loss, None) if return_outputs else loss

    trainer = _real_trainer(
        tmp_path, accumulation=2, trainer_base=StockLossTrainer
    )
    trainer.stock_loss_calls = 0
    count_values = {
        "none": None,
        "boolean": True,
        "boolean_tensor": torch.tensor(True),
        "nonnumeric": "fourteen",
        "vector_tensor": torch.tensor([14]),
        "nan_float": float("nan"),
        "infinite_float": float("inf"),
        "zero_int": 0,
        "negative_float": -1.0,
        "nan_tensor": torch.tensor(float("nan")),
        "infinite_tensor": torch.tensor(float("inf")),
        "zero_tensor": torch.tensor(0),
        "negative_tensor": torch.tensor(-1),
    }
    global_count = _MISSING if case == "missing" else count_values[case]
    inputs = _inputs(
        torch,
        trainer.model,
        [2, 1],
        global_count=global_count,
    )
    inputs["stock_loss_multiplier"] = 3.0
    expected_stock_loss = trainer.model.scale * 3.0

    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.grpo"):
        first_result = trainer.compute_loss(trainer.model, inputs)
        second_result = trainer.compute_loss(trainer.model, inputs)

    torch.testing.assert_close(first_result, expected_stock_loss)
    torch.testing.assert_close(second_result, expected_stock_loss)
    assert trainer.stock_loss_calls == 2
    warnings = [
        record for record in caplog.records if "fell back to the stock TRL loss" in record.message
    ]
    assert len(warnings) == 1
    assert "num_items_in_batch" in warnings[0].message
    assert reason in warnings[0].message


def test_zero_dimensional_tensor_uses_global_normalizer_through_training_step(
    tmp_path, caplog
):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2)
    model = trainer.model
    masks = [_mask(torch, [2, 1]), _mask(torch, [10, 1])]
    advantages = torch.tensor([1.0, 1.0])
    model.zero_grad(set_to_none=True)

    with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.grpo"):
        for mask in masks:
            inputs = {
                "per_token_logps": model.scale.expand_as(mask),
                "old_per_token_logps": torch.zeros_like(mask),
                "advantages": advantages,
                "completion_mask": mask,
                "num_items_in_batch": torch.tensor(14),
            }
            _training_step(trainer, inputs, item_count=14)

    actual_gradient = model.scale.grad.detach().clone()
    reference_scale = torch.nn.Parameter(model.scale.detach().clone())
    full_mask = torch.cat(masks)
    reference_loss = (
        _dapo_sum(
            torch,
            reference_scale.expand_as(full_mask),
            torch.zeros_like(full_mask),
            advantages.repeat(2),
            full_mask,
        )
        / 14
        / 2
    )
    expected_gradient = torch.autograd.grad(reference_loss, reference_scale)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert not any("fell back to the stock TRL loss" in r.message for r in caplog.records)


def test_evaluation_uses_global_count_without_accumulation_division(tmp_path):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2)
    trainer.model.eval()
    inputs = _inputs(
        torch,
        trainer.model,
        [2, 1],
        global_count=torch.tensor(14),
    )

    result = trainer._compute_variant_loss(trainer.model, inputs)
    expected = _dapo_sum(
        torch,
        inputs["per_token_logps"],
        inputs["old_per_token_logps"],
        inputs["advantages"],
        inputs["completion_mask"],
    ) / 14
    torch.testing.assert_close(result, expected)


def test_global_count_is_divided_by_accelerator_process_count(tmp_path):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2)
    trainer.accelerator = type("Accelerator", (), {"num_processes": 2})()
    inputs = _inputs(
        torch,
        trainer.model,
        [2, 1],
        global_count=torch.tensor(14),
    )

    result = trainer._compute_variant_loss(trainer.model, inputs)
    numerator = _dapo_sum(
        torch,
        inputs["per_token_logps"],
        inputs["old_per_token_logps"],
        inputs["advantages"],
        inputs["completion_mask"],
    )
    torch.testing.assert_close(result, numerator / 7)


def test_tool_mask_and_kl_remain_inside_global_dapo_numerator(tmp_path):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2, beta=0.5)
    mask = torch.ones(2, 2)
    tool_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    reference = torch.full_like(mask, 0.4)
    inputs = {
        "per_token_logps": trainer.model.scale.expand_as(mask),
        "old_per_token_logps": torch.zeros_like(mask),
        "advantages": torch.tensor([1.0, -1.0]),
        "completion_mask": mask,
        "tool_mask": tool_mask,
        "ref_per_token_logps": reference,
        "num_items_in_batch": torch.tensor(2),
    }

    result = trainer._compute_variant_loss(trainer.model, inputs)
    combined_mask = mask * tool_mask
    policy_loss = -torch.minimum(
        torch.exp(inputs["per_token_logps"]) * inputs["advantages"].unsqueeze(-1),
        torch.clamp(torch.exp(inputs["per_token_logps"]), 0.8, 1.28)
        * inputs["advantages"].unsqueeze(-1),
    )
    ref_minus_logp = reference - inputs["per_token_logps"]
    kl = torch.exp(ref_minus_logp) - ref_minus_logp - 1
    expected = ((policy_loss + 0.5 * kl) * combined_mask).sum() / 2
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("variant", ["bnpo", "two_sided", "dr_grpo", "gspo", "rft"])
def test_other_variants_ignore_global_count_and_keep_existing_reductions(
    tmp_path, variant
):
    torch = _torch()
    from transformers import Trainer, TrainingArguments

    from soup_cli.trainer.grpo import make_grpo_trainer_variant
    from soup_cli.utils.grpo_variants import apply_variant_loss

    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, **kwargs):
            return self.scale

    model = ScalarModel()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    trainer = make_grpo_trainer_variant(Trainer, variant)(model=model, args=args)
    trainer.current_gradient_accumulation_steps = 2
    if variant == "two_sided":
        trainer._soup_grpo_delta = 0.2
    inputs = _inputs(torch, model, [2, 1], global_count=torch.tensor(14))
    expected = apply_variant_loss(
        variant,
        logp_new=inputs["per_token_logps"],
        logp_old=inputs["old_per_token_logps"],
        advantages=inputs["advantages"],
        delta=0.2 if variant == "two_sided" else None,
        completion_mask=inputs["completion_mask"],
    ) / 2

    torch.testing.assert_close(
        trainer._compute_variant_loss(model, inputs),
        expected,
    )


def test_counts_are_per_call_and_partial_cycle_uses_current_cycle_size(tmp_path):
    torch = _torch()
    trainer = _real_trainer(tmp_path, accumulation=2)
    model = trainer.model
    inputs = _inputs(torch, model, [2, 1], global_count=torch.tensor(14))
    first = trainer._compute_variant_loss(model, inputs)
    inputs["num_items_in_batch"] = torch.tensor(7)
    second = trainer._compute_variant_loss(model, inputs)
    torch.testing.assert_close(first, second * 0.5)

    # Trainer sets this to the actual number of microbatches in an optimizer
    # cycle; if the final cycle is short, its own training_step division uses 1.
    trainer.current_gradient_accumulation_steps = 1
    trainer.model.train()
    trainer.model.zero_grad(set_to_none=True)
    partial_inputs = _inputs(torch, model, [2, 1], global_count=torch.tensor(3))
    _training_step(trainer, partial_inputs, item_count=3)
    reference_logp = model.scale.expand_as(partial_inputs["completion_mask"])
    expected_gradient = torch.autograd.grad(
        _dapo_sum(
            torch,
            reference_logp,
            partial_inputs["old_per_token_logps"],
            partial_inputs["advantages"],
            partial_inputs["completion_mask"],
        )
        / 3,
        model.scale,
    )[0]
    torch.testing.assert_close(model.scale.grad, expected_gradient)
