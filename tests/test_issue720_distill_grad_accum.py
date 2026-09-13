"""Issue #720 - distillation must not let gradient accumulation change gradient scale.

``_DistillTrainer.compute_loss`` accepts ``num_items_in_batch`` and ignores it:
it returns a mean over its own microbatch. Transformers skips its own
gradient-accumulation compensation whenever it classifies the model as
consuming loss kwargs, so with that classification left in place the
accumulated gradient scales with ``gradient_accumulation_steps`` - the same
effective batch split four ways produces four times the gradient.

The fix is the minimal one the issue names: mark the trainer as not consuming
loss kwargs, which restores the division Transformers would otherwise apply.

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


def _compile_distill_trainer(*, without_init: bool = False) -> type:
    """The real ``_DistillTrainer`` body, bound to the real ``Trainer``.

    ``_sequence_mode=True`` makes ``compute_loss`` return its CE term before any
    teacher, ULD or MiniLLM name is looked up. ``without_init`` drops the
    class's own ``__init__``, which is the state of ``main`` before #720.
    """
    node = ast.parse(ast.unparse(_distill_trainer_class_node())).body[0]
    if without_init:
        node.body = [
            stmt
            for stmt in node.body
            if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__")
        ]
    namespace = {
        "Trainer": transformers.Trainer,
        "_sequence_mode": True,
        "_minillm_on_policy": False,
    }
    exec(compile(ast.Module([node], []), "distill.py", "exec"), namespace)
    return namespace["_DistillTrainer"]


def _accumulated_grad_norm(trainer_cls: type, steps: int, tmp_path) -> float:
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
    for chunk in torch.chunk(input_ids, steps):
        batch = {
            "input_ids": chunk,
            "labels": chunk,
            "attention_mask": torch.ones_like(chunk),
        }
        # The loop always passes a token count here; None would take the
        # compensating branch regardless of the flag and hide the bug.
        trainer.training_step(model, batch, num_items_in_batch=torch.tensor(input_ids.numel()))

    squares = [p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt())


@pytest.mark.parametrize("steps", [4, 8])
def test_distill_trainer_gradient_does_not_scale_with_accumulation(steps, tmp_path):
    """Acceptance: the same 8 rows give the same gradient at GA=1, 4 and 8."""
    trainer_cls = _compile_distill_trainer()

    reference = _accumulated_grad_norm(trainer_cls, 1, tmp_path)
    accumulated = _accumulated_grad_norm(trainer_cls, steps, tmp_path)

    assert accumulated / reference == pytest.approx(1.0, rel=1e-5)


def test_without_the_opt_out_the_gradient_scales_with_the_step_count(tmp_path):
    """The bug, measured on the same class with its ``__init__`` removed.

    This keeps the test above honest: if Transformers ever compensated on its
    own, both would pass and this one would say why.
    """
    trainer_cls = _compile_distill_trainer(without_init=True)

    reference = _accumulated_grad_norm(trainer_cls, 1, tmp_path)
    accumulated = _accumulated_grad_norm(trainer_cls, 4, tmp_path)

    assert accumulated / reference == pytest.approx(4.0, rel=1e-5)


def test_distill_trainer_opts_out_after_trainer_init():
    """The opt-out must come after ``super().__init__``, which assigns the flag.

    Kept alongside the measurement because moving the assignment above the
    super() call is overwritten at construction, and this pins the order
    without pinning how the assignment is spelled.
    """
    inits = [
        node
        for node in _distill_trainer_class_node().body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert inits, "_DistillTrainer defines no __init__, so it cannot opt out"

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
