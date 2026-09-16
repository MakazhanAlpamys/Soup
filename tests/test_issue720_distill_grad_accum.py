"""Issues #720/#932 — distillation gradient-accumulation loss scaling.

The trainer must consume Transformers' full-window ``num_items_in_batch`` and
weight each microbatch mean by its share of trained causal targets. A fixed
``1 / gradient_accumulation_steps`` factor works only for equal token counts;
unequal chunks otherwise change both the norm and direction of the gradient.

``_DistillTrainer`` is defined inside ``setup()``, so the measurement below
compiles the class body straight out of ``distill.py`` and runs it through the
real ``Trainer.training_step`` on a one-layer ``LlamaForCausalLM``. With
``_sequence_mode`` set, the real ``compute_loss`` is plain mean-reduced CE and
needs no teacher. Nothing about the fix is restated in the test, so a
respelled opt-out still passes and a removed one fails.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

_DISTILL_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "soup_cli"
    / "trainer"
    / "distill.py"
).read_text(encoding="utf-8")


def _distill_trainer_class_node() -> ast.ClassDef:
    classes = [
        node
        for node in ast.walk(ast.parse(_DISTILL_SOURCE))
        if isinstance(node, ast.ClassDef) and node.name == "_DistillTrainer"
    ]
    assert classes, "_DistillTrainer is gone; this test needs rewriting"
    return classes[0]


def _compile_distill_trainer(*, without_token_weighting: bool = False) -> type:
    """The real ``_DistillTrainer`` body, bound to the real ``Trainer``.

    ``_sequence_mode=True`` makes ``compute_loss`` return its CE term before any
    teacher, ULD or MiniLLM name is looked up. ``without_token_weighting``
    replaces the real normaliser with an identity function as a mutation
    control: the equal-length check must then recover #720's scaling bug.
    """
    node = ast.parse(ast.unparse(_distill_trainer_class_node())).body[0]
    if without_token_weighting:
        compute_loss = next(
            stmt
            for stmt in node.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "compute_loss"
        )
        normalizer = next(
            stmt
            for stmt in compute_loss.body
            if isinstance(stmt, ast.FunctionDef)
            and stmt.name == "_token_weighted_accumulation"
        )
        normalizer.body = [ast.Return(value=ast.Name(id="loss", ctx=ast.Load()))]
        ast.fix_missing_locations(node)
    namespace = {
        "Trainer": transformers.Trainer,
        "_sequence_mode": True,
        "_minillm_on_policy": False,
    }
    exec(compile(ast.Module([node], []), "distill.py", "exec"), namespace)
    return namespace["_DistillTrainer"]


def _accumulated_gradient(trainer_cls: type, steps: int, tmp_path):
    """One optimizer window of ``steps`` equal microbatches over the same 8 rows."""
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = transformers.LlamaForCausalLM(config)
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / f"ga{steps}"),
        gradient_accumulation_steps=steps,
        per_device_train_batch_size=8 // steps,
        report_to=[],
        use_cpu=True,
    )
    trainer = trainer_cls(model=model, args=args)
    # Set by the training loop per window; training_step divides by it.
    trainer.current_gradient_accumulation_steps = steps

    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (8, 6))
    model.train()
    batches = [
        {
            "input_ids": chunk,
            "labels": chunk,
            "attention_mask": torch.ones_like(chunk),
        }
        for chunk in torch.chunk(input_ids, steps)
    ]
    num_items_in_batch = trainer._get_num_items_in_batch(batches, trainer.args.device)
    assert int(num_items_in_batch) == 40  # 8 rows * 5 shifted causal targets
    for batch in batches:
        trainer.training_step(model, batch, num_items_in_batch=num_items_in_batch)

    return torch.cat(
        [
            parameter.grad.detach().flatten()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    )


def _accumulated_grad_norm(trainer_cls: type, steps: int, tmp_path) -> float:
    return float(_accumulated_gradient(trainer_cls, steps, tmp_path).norm())


def _unequal_length_gradient(
    trainer_cls: type,
    steps: int,
    tmp_path,
    *,
    token_distill: bool = False,
):
    """Gradient for the same 8 rows, padded together or split into 4/60-token chunks."""
    if steps not in (1, 2):
        raise ValueError("unequal-length measurement supports only GA=1 or GA=2")

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = transformers.LlamaForCausalLM(config)
    if token_distill:
        from soup_cli.trainer.distill import _compute_distill_term

        torch.manual_seed(2)
        teacher = transformers.LlamaForCausalLM(config)
        teacher.eval()
        teacher.requires_grad_(False)
        trainer_globals = trainer_cls.compute_loss.__globals__
        trainer_globals.update({
            "_sequence_mode": False,
            "_minillm_on_policy": False,
            "_uld_aligned": False,
            "_uld_teacher_tokenizer": None,
            "_student_tokenizer": None,
            "teacher_ref": teacher,
            "_uld_projection": None,
            "_minillm_cb": None,
            "_CE_WEIGHT": 0.5,
            "_DISTILL_WEIGHT": 0.5,
            "_compute_distill_term": _compute_distill_term,
            "divergence": "forward_kl",
            "temperature": 2.0,
            "_distill_chunk_size": None,
            "_distill_checkpoint": False,
        })
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / f"unequal-ga{steps}"),
        gradient_accumulation_steps=steps,
        per_device_train_batch_size=8 // steps,
        report_to=[],
        use_cpu=True,
    )
    trainer = trainer_cls(model=model, args=args)
    trainer.current_gradient_accumulation_steps = steps

    lengths = [2, 2, 2, 2, 16, 16, 16, 16]
    torch.manual_seed(1)
    input_ids = torch.randint(1, config.vocab_size, (8, 16))
    labels = input_ids.clone()
    attention_mask = torch.zeros_like(input_ids)
    for row, length in enumerate(lengths):
        labels[row, length:] = -100
        attention_mask[row, :length] = 1

    if steps == 1:
        padded_ids = input_ids.clone()
        padded_ids[attention_mask == 0] = 0
        batches = [{
            "input_ids": padded_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }]
    else:
        batches = [
            {
                "input_ids": input_ids[:4, :2],
                "labels": labels[:4, :2],
                "attention_mask": attention_mask[:4, :2],
            },
            {
                "input_ids": input_ids[4:],
                "labels": labels[4:],
                "attention_mask": attention_mask[4:],
            },
        ]

    # Causal shifting leaves 4 targets in the short chunk and 60 in the long
    # one. Exercise Trainer's real window counter, not a test-supplied stand-in.
    num_items_in_batch = trainer._get_num_items_in_batch(batches, trainer.args.device)
    assert int(num_items_in_batch) == 64
    model.train()
    for batch in batches:
        trainer.training_step(model, batch, num_items_in_batch=num_items_in_batch)

    return torch.cat(
        [
            parameter.grad.detach().flatten()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    )


@pytest.mark.parametrize("steps", [4, 8])
def test_distill_trainer_gradient_does_not_scale_with_accumulation(steps, tmp_path):
    """Acceptance: the same 8 rows give the same gradient at GA=1, 4 and 8."""
    trainer_cls = _compile_distill_trainer()

    reference = _accumulated_grad_norm(trainer_cls, 1, tmp_path)
    accumulated = _accumulated_grad_norm(trainer_cls, steps, tmp_path)

    assert accumulated / reference == pytest.approx(1.0, rel=1e-5)


def test_unequal_microbatch_lengths_match_full_batch_token_mean(tmp_path):
    """Issue #932: chunking 4 vs 60 targets must not change the gradient."""
    trainer_cls = _compile_distill_trainer()

    full_batch = _unequal_length_gradient(trainer_cls, 1, tmp_path)
    accumulated = _unequal_length_gradient(trainer_cls, 2, tmp_path)

    torch.testing.assert_close(accumulated, full_batch, rtol=1e-5, atol=1e-6)


def test_unequal_microbatch_lengths_match_for_live_token_distillation(tmp_path):
    """The CE+teacher-KL branch obeys the same window-wide token contract."""
    trainer_cls = _compile_distill_trainer()

    full_batch = _unequal_length_gradient(
        trainer_cls, 1, tmp_path, token_distill=True
    )
    accumulated = _unequal_length_gradient(
        trainer_cls, 2, tmp_path, token_distill=True
    )

    torch.testing.assert_close(accumulated, full_batch, rtol=1e-5, atol=1e-6)


def test_without_token_weighting_the_gradient_scales_with_the_step_count(tmp_path):
    """The #720 bug, measured with the real normaliser replaced by identity.

    This keeps the test above honest: if Transformers ever compensated on its
    own, both would pass and this one would say why.
    """
    trainer_cls = _compile_distill_trainer(without_token_weighting=True)

    reference = _accumulated_grad_norm(trainer_cls, 1, tmp_path)
    accumulated = _accumulated_grad_norm(trainer_cls, 4, tmp_path)

    assert accumulated / reference == pytest.approx(4.0, rel=1e-5)


def test_distill_trainer_sets_loss_kwargs_contract_after_trainer_init(tmp_path):
    """The contract must be set after ``super().__init__``, which assigns the flag.

    Kept alongside the measurement because moving the assignment above the
    super() call is overwritten at construction, and this pins the order
    without pinning how the assignment is spelled.
    """
    inits = [
        node
        for node in _distill_trainer_class_node().body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert inits, "_DistillTrainer defines no __init__, so it cannot set the loss contract"

    body = inits[0].body
    super_at = next(
        i
        for i, stmt in enumerate(body)
        if "super().__init__" in ast.unparse(stmt)
    )
    flag_at = [
        i
        for i, stmt in enumerate(body)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        if isinstance(target, ast.Attribute)
        and target.attr == "model_accepts_loss_kwargs"
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ]
    assert flag_at, "_DistillTrainer never assigns self.model_accepts_loss_kwargs"
    assert min(flag_at) > super_at, "the flag is set before super().__init__ overwrites it"

    trainer_cls = _compile_distill_trainer()
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    trainer = trainer_cls(
        model=transformers.LlamaForCausalLM(config),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path / "loss-contract"),
            report_to=[],
            use_cpu=True,
        ),
    )
    assert trainer.model_accepts_loss_kwargs is True
