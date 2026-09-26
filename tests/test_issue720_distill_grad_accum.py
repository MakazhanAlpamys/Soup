"""Issues #720/#932/#990 — distillation gradient-accumulation loss scaling.

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

The class is nested in a factory, so there is no importable symbol. The
harness walks ``compute_loss`` for free names and binds them from the factory
and schema defaults. A new unresolvable name fails with that name listed —
not as a ``NameError`` stamped ``distill.py``. The compile filename is
``<compiled _DistillTrainer>`` so a traceback cannot be mistaken for the
real module (#990).
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys
from typing import Any

import pytest

from soup_cli.config.schema import TrainingConfig

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

_COMPILE_FILENAME = "<compiled _DistillTrainer>"
_BUILTIN_NAMES = frozenset(dir(builtins))


class _UnresolvableFactoryError(Exception):
    """Raised when a factory assignment cannot be evaluated for the harness."""


def _distill_trainer_class_node() -> ast.ClassDef:
    classes = [
        node
        for node in ast.walk(ast.parse(_DISTILL_SOURCE))
        if isinstance(node, ast.ClassDef) and node.name == "_DistillTrainer"
    ]
    assert classes, "_DistillTrainer is gone; this test needs rewriting"
    return classes[0]


def _compute_loss_node(class_node: ast.ClassDef) -> ast.FunctionDef:
    return next(
        stmt
        for stmt in class_node.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "compute_loss"
    )


def _training_config_default(field: str) -> object:
    info = TrainingConfig.model_fields[field]
    factory = info.default_factory
    if callable(factory):
        return factory()
    return info.default


def _scope_parameters(fn: ast.AST) -> set[str]:
    bound: set[str] = set()
    args = fn.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        bound.add(arg.arg)
    if args.vararg is not None:
        bound.add(args.vararg.arg)
    if args.kwarg is not None:
        bound.add(args.kwarg.arg)
    return bound


class _ScopeBindingVisitor(ast.NodeVisitor):
    """Collect names this function binds without entering nested scopes."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.nested: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self.nested.append(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bound.add(node.name)
        self.nested.append(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.nested.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)


class _ScopeLoadVisitor(ast.NodeVisitor):
    """Collect Load names in this function, leaving nested scopes alone."""

    def __init__(self) -> None:
        self.loads: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)


def _compute_loss_free_names(fn: ast.AST) -> set[str]:
    """Names ``fn`` loads but does not bind, including uses inside nested defs.

    Nested ``FunctionDef`` names are bound here, so
    ``_token_weighted_accumulation`` is not reported as missing. A store
    inside a nested function does not bind the outer function — that is the
    naive-walk bug in the other direction.
    """
    bound = _scope_parameters(fn)
    binder = _ScopeBindingVisitor()
    loads = _ScopeLoadVisitor()
    body = fn.body if not isinstance(fn, ast.Lambda) else [fn.body]
    for default in (*fn.args.defaults, *fn.args.kw_defaults):
        if default is not None:
            loads.visit(default)
    for stmt in body:
        binder.visit(stmt)
        loads.visit(stmt)
    bound |= binder.bound
    free = {name for name in loads.loads if name not in bound}
    for nested in binder.nested:
        free |= _compute_loss_free_names(nested) - bound
    return free


def _eval_factory_expr(
    node: ast.AST,
    env: dict[str, object],
    sentinels: set[str] | frozenset[str] | None = None,
) -> object:
    if sentinels is None:
        sentinels = set()
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in sentinels or node.id not in env:
            raise _UnresolvableFactoryError(node.id)
        return env[node.id]
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "tcfg":
            if node.attr not in TrainingConfig.model_fields:
                raise _UnresolvableFactoryError(node.attr)
            return _training_config_default(node.attr)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            raise _UnresolvableFactoryError(f"self.{node.attr}")
        base = _eval_factory_expr(node.value, env, sentinels)
        if base is None:
            raise _UnresolvableFactoryError(ast.dump(node))
        return getattr(base, node.attr)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_factory_expr(node.operand, env, sentinels)
    if isinstance(node, ast.BoolOp):
        values = [
            _eval_factory_expr(value, env, sentinels) for value in node.values
        ]
        result: object = isinstance(node.op, ast.And)
        for value in values:
            if isinstance(node.op, ast.And):
                result = result and value
            else:
                result = result or value
        return result
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _eval_factory_expr(node.left, env, sentinels)
        right = _eval_factory_expr(node.comparators[0], env, sentinels)
        operator = node.ops[0]
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Is):
            return left is right
        if isinstance(operator, ast.IsNot):
            return left is not right
        raise _UnresolvableFactoryError(ast.dump(node))
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "float", "int"}
            and len(node.args) == 1
        ):
            caster = {"bool": bool, "float": float, "int": int}[func.id]
            try:
                return caster(_eval_factory_expr(node.args[0], env, sentinels))
            except (TypeError, ValueError) as exc:
                raise _UnresolvableFactoryError(str(exc)) from exc
        if (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and 2 <= len(node.args) <= 3
        ):
            name = _eval_factory_expr(node.args[1], env, sentinels)
            if (
                isinstance(node.args[0], ast.Name)
                and node.args[0].id == "tcfg"
                and isinstance(name, str)
                and name in TrainingConfig.model_fields
            ):
                return _training_config_default(name)
            raise _UnresolvableFactoryError(ast.dump(node))
        if node.args:
            # Validators wrap a value; bind the wrapped expression.
            return _eval_factory_expr(node.args[0], env, sentinels)
        raise _UnresolvableFactoryError(ast.dump(node))
    raise _UnresolvableFactoryError(ast.dump(node))


def _is_identity_runtime_ref(node: ast.AST, env: dict[str, object]) -> bool:
    """``self.attr`` or an unbound factory local, assigned as-is."""
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "self"
    if isinstance(node, ast.Name):
        return node.id not in env and node.id not in _BUILTIN_NAMES
    return False


def _factory_derived_bindings() -> dict[str, object]:
    """Evaluate ``setup()`` assignments that precede ``_DistillTrainer``."""
    tree = ast.parse(_DISTILL_SOURCE)
    setups = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    ]
    assert len(setups) == 1, (
        f"distill.py has {len(setups)} setup() functions; "
        "the harness needs exactly one to derive closure bindings"
    )
    setup = setups[0]
    env: dict[str, object] = {}
    sentinels: set[str] = set()
    for stmt in setup.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == "_DistillTrainer":
            break
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if _is_identity_runtime_ref(stmt.value, env):
            env[target.id] = None
            sentinels.add(target.id)
            continue
        try:
            env[target.id] = _eval_factory_expr(stmt.value, env, sentinels)
        except _UnresolvableFactoryError:
            continue
    return env


def _closure_namespace(compute_loss: ast.FunctionDef) -> dict[str, Any]:
    """Bind every free name ``compute_loss`` needs, or fail naming the rest.

    Schema fields come from ``TrainingConfig`` defaults, not literals, so the
    #720 tests keep the unchunked path when ``distill_chunk_size``'s default
    is ``None``. Harness overrides keep ``_sequence_mode`` on so CE-only
    measurements still skip the teacher. A name the walker cannot resolve
    fails here instead of as a ``NameError`` inside the recompiled body.
    """
    from soup_cli.trainer import distill as distill_mod

    namespace: dict[str, Any] = {}
    namespace.update(_factory_derived_bindings())
    namespace.update({
        "_sequence_mode": True,
        "_minillm_on_policy": False,
    })
    free = _compute_loss_free_names(compute_loss)
    for name in free:
        if name not in namespace and hasattr(distill_mod, name):
            namespace[name] = getattr(distill_mod, name)
    missing = sorted(
        name
        for name in free
        if name not in namespace and name not in _BUILTIN_NAMES
    )
    assert not missing, (
        "_compile_distill_trainer() must bind "
        f"{missing} — the factory gained them and this harness was not updated."
    )
    return namespace


def _exec_distill_trainer_class(
    node: ast.ClassDef,
    *,
    trainer_base: type | None = None,
) -> type:
    namespace = _closure_namespace(_compute_loss_node(node))
    namespace["Trainer"] = (
        transformers.Trainer if trainer_base is None else trainer_base
    )
    compiled = compile(ast.Module([node], []), _COMPILE_FILENAME, "exec")
    exec(compiled, namespace)
    return namespace["_DistillTrainer"]


def _compile_distill_trainer(*, without_token_weighting: bool = False) -> type:
    """The real ``_DistillTrainer`` body, bound to the real ``Trainer``.

    ``_sequence_mode=True`` makes ``compute_loss`` return its CE term before any
    teacher, ULD or MiniLLM name is looked up. ``without_token_weighting``
    replaces the real normaliser with an identity function as a mutation
    control: the equal-length check must then recover #720's scaling bug.
    """
    node = ast.parse(ast.unparse(_distill_trainer_class_node())).body[0]
    if without_token_weighting:
        compute_loss = _compute_loss_node(node)
        normalizer = next(
            stmt
            for stmt in compute_loss.body
            if isinstance(stmt, ast.FunctionDef)
            and stmt.name == "_token_weighted_accumulation"
        )
        normalizer.body = [ast.Return(value=ast.Name(id="loss", ctx=ast.Load()))]
        ast.fix_missing_locations(node)
    return _exec_distill_trainer_class(node)


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
            "_distill_chunk_size": _training_config_default("distill_chunk_size"),
            "_distill_checkpoint": bool(
                _training_config_default("distill_checkpoint")
            ),
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


def test_nested_function_definitions_are_not_missing_bindings() -> None:
    """``_token_weighted_accumulation`` is defined inside ``compute_loss``."""
    compute_loss = _compute_loss_node(_distill_trainer_class_node())
    free = _compute_loss_free_names(compute_loss)
    assert "_token_weighted_accumulation" not in free
    assert "_sequence_mode" in free
    assert "_distill_chunk_size" in free


def test_nested_store_does_not_bind_an_outer_free_name() -> None:
    """A naive whole-tree Store walk would hide ``_factory_only``."""
    fn = ast.parse(
        "def compute_loss(self):\n"
        "    def _inner():\n"
        "        _factory_only = 1\n"
        "        return _factory_only\n"
        "    return _factory_only\n"
    ).body[0]
    free = _compute_loss_free_names(fn)
    assert "_factory_only" in free
    assert "_inner" not in free


def test_unresolved_free_name_fails_with_the_missing_binding() -> None:
    """Adding a closure variable without a resolvable binding is a named fail."""
    node = ast.parse(ast.unparse(_distill_trainer_class_node())).body[0]
    compute_loss = _compute_loss_node(node)
    compute_loss.body.insert(
        0,
        ast.Expr(value=ast.Name(id="_missing_harness_binding", ctx=ast.Load())),
    )
    ast.fix_missing_locations(node)
    with pytest.raises(AssertionError, match="_missing_harness_binding"):
        _exec_distill_trainer_class(node)


def test_compiled_trainer_binds_schema_defaults_for_chunked_distill() -> None:
    compiled_globals = _closure_namespace(
        _compute_loss_node(_distill_trainer_class_node())
    )
    assert compiled_globals["_distill_chunk_size"] == _training_config_default(
        "distill_chunk_size"
    )
    assert compiled_globals["_distill_checkpoint"] == _training_config_default(
        "distill_checkpoint"
    )


def test_tcfg_attribute_binds_the_schema_default() -> None:
    node = ast.parse("tcfg.distill_chunk_size", mode="eval").body
    assert _eval_factory_expr(node, {}) == _training_config_default(
        "distill_chunk_size"
    )


def test_recompiled_compute_loss_filename_is_not_the_real_module() -> None:
    node = ast.parse(ast.unparse(_distill_trainer_class_node())).body[0]
    trainer_cls = _exec_distill_trainer_class(
        node, trainer_base=type("Trainer", (), {})
    )
    assert trainer_cls.compute_loss.__code__.co_filename == _COMPILE_FILENAME


def test_distill_module_has_exactly_one_setup_function() -> None:
    setups = [
        node
        for node in ast.walk(ast.parse(_DISTILL_SOURCE))
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    ]
    assert len(setups) == 1


def test_runtime_identity_bindings_stay_none_sentinels() -> None:
    compiled_globals = _closure_namespace(
        _compute_loss_node(_distill_trainer_class_node())
    )
    assert compiled_globals["teacher_ref"] is None
    assert compiled_globals["_student_tokenizer"] is None


def test_comparison_over_a_runtime_sentinel_is_unresolvable() -> None:
    node = ast.parse("teacher_ref is not None", mode="eval").body
    with pytest.raises(_UnresolvableFactoryError):
        _eval_factory_expr(node, {"teacher_ref": None}, {"teacher_ref"})


def test_float_of_runtime_attribute_is_named_not_typeerror() -> None:
    node = ast.parse("float(teacher_ref.scale)", mode="eval").body
    with pytest.raises(_UnresolvableFactoryError):
        _eval_factory_expr(node, {"teacher_ref": None}, {"teacher_ref"})


def test_derived_runtime_flag_fails_with_the_missing_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#743 shape: a flag derived from a runtime object must not bind False."""
    tree = ast.parse(_DISTILL_SOURCE)
    setup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    insert_at = next(
        index
        for index, stmt in enumerate(setup.body)
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "_distill_checkpoint"
    )
    setup.body.insert(
        insert_at + 1,
        ast.Assign(
            targets=[ast.Name(id="_brand_new_runtime_flag", ctx=ast.Store())],
            value=ast.Compare(
                left=ast.Name(id="teacher_ref", ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
        ),
    )
    compute_loss = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    compute_loss.body.insert(
        0,
        ast.Expr(value=ast.Name(id="_brand_new_runtime_flag", ctx=ast.Load())),
    )
    ast.fix_missing_locations(tree)
    monkeypatch.setattr(sys.modules[__name__], "_DISTILL_SOURCE", ast.unparse(tree))
    with pytest.raises(AssertionError, match="_brand_new_runtime_flag"):
        _closure_namespace(_compute_loss_node(_distill_trainer_class_node()))
