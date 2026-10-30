"""#2368: a PRM forward keeps the last position's logits only.

``_PRMTrainer.compute_loss`` and the two forwards of ``PRMScorer`` read only
``hidden_states[-1]``, yet the causal-LM forward applied the LM head at every
position, building a ``[batch, seq_len, vocab]`` tensor nothing read. Where the
unwrapped model's ``forward`` takes ``logits_to_keep``, they now pass 1: the logits
shrink to ``[batch, 1, vocab]`` and ``hidden_states`` does not change. A forward
without the argument (or with only ``**kwargs``) keeps today's call. The check is
one ``inspect.signature`` per trainer or scorer, cached as a plain attribute.

CPU, fp32, a tiny random-init Llama; nothing is downloaded.
"""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

_VOCAB = 64
_HIDDEN = 16


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
    )
    model.reward_head = torch.nn.Linear(_HIDDEN, 1)
    return model.float()


def _batch():
    from soup_cli.trainer.prm import _build_collator

    rows = [
        {"input_ids": [5, 6, 7, 8, 9, 10], "attention_mask": [1] * 6,
         "step_positions": [1, 3, 5], "labels": [1.0, 0.0, 1.0]},
        {"input_ids": [11, 12, 13, 14], "attention_mask": [1] * 4,
         "step_positions": [2, 3], "labels": [0.0, 1.0]},
    ]
    return _build_collator(SimpleNamespace(pad_token_id=0))(rows)


def _compute_loss():
    from transformers import Trainer

    from soup_cli.trainer.prm import make_prm_trainer_class

    return make_prm_trainer_class(Trainer).compute_loss


def _logits_shapes(model):
    """Record the logits shape of every forward ``model`` runs."""
    shapes: list[tuple[int, ...]] = []
    model.register_forward_hook(lambda _m, _a, out: shapes.append(tuple(out.logits.shape)))
    return shapes


def _step(model, trainer_self, batch):
    """One PRM step: the loss and every trainable parameter's gradient."""
    model.zero_grad(set_to_none=True)
    loss = _compute_loss()(trainer_self, model, batch)
    loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    return loss.detach().clone(), grads


# --- the training step --------------------------------------------------------------


def test_the_training_step_forward_keeps_one_position(tmp_path):
    """The acceptance item: the step's forward returns ``[batch, 1, vocab]`` logits."""
    from transformers import Trainer, TrainingArguments

    from soup_cli.trainer.prm import make_prm_trainer_class

    model = _tiny_llama().train()
    shapes = _logits_shapes(model)
    batch = _batch()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    trainer = make_prm_trainer_class(Trainer)(model=model, args=args)

    loss = trainer.compute_loss(model, batch)

    assert torch.isfinite(loss)
    assert shapes == [(2, 1, _VOCAB)]
    assert trainer._prm_logits_to_keep is True


def test_loss_and_every_gradient_match_the_full_logits_step():
    """fp32: the loss and each trainable gradient are ``torch.equal`` to the step that
    builds the logits at every position (the forward called without the argument)."""
    model = _tiny_llama().train()
    shapes = _logits_shapes(model)
    batch = _batch()

    kept_loss, kept_grads = _step(model, SimpleNamespace(model=model), batch)
    full_loss, full_grads = _step(
        model, SimpleNamespace(model=model, _prm_logits_to_keep=False), batch
    )

    assert shapes == [(2, 1, _VOCAB), (2, 6, _VOCAB)]
    assert torch.equal(kept_loss, full_loss)
    assert set(kept_grads) == set(full_grads)
    # Every trainable group but lm_head, which the reward never reads (#1235's note).
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(kept_grads) == trainable - {"lm_head.weight"}
    unequal = sorted(name for name in kept_grads if not torch.equal(kept_grads[name],
                                                                    full_grads[name]))
    assert not unequal, unequal


def test_the_signature_is_checked_once_per_trainer(monkeypatch):
    import soup_cli.utils.prm as prm_utils

    calls: list[object] = []

    def counting(obj):
        calls.append(obj)
        return inspect.signature(obj)

    monkeypatch.setattr(
        prm_utils, "inspect", SimpleNamespace(signature=counting, Parameter=inspect.Parameter)
    )
    model = _tiny_llama()
    trainer_self = SimpleNamespace(model=model)
    batch = _batch()
    for _ in range(3):
        _compute_loss()(trainer_self, model, batch)
    assert len(calls) == 1
    assert trainer_self._prm_logits_to_keep is True


def test_the_accelerate_wrapped_forward_still_reads_as_taking_the_argument():
    """Accelerate's mixed-precision wrapper replaces ``model.forward`` with
    ``convert_outputs_to_fp32(autocast(forward))``; the check sees through it, and the
    step still calls that top-level forward."""
    from accelerate.utils import convert_outputs_to_fp32

    from soup_cli.utils.prm import forward_takes_logits_to_keep

    model = _tiny_llama()
    autocast = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    model.forward = MethodType(convert_outputs_to_fp32(autocast(model.forward.__func__)), model)
    assert forward_takes_logits_to_keep(model) is True

    shapes = _logits_shapes(model)
    loss = _compute_loss()(SimpleNamespace(model=model), model, _batch())
    assert torch.isfinite(loss)
    assert shapes == [(2, 1, _VOCAB)]


# --- a forward without the argument ---------------------------------------------------


class _Recorder(torch.nn.Module):
    """A causal-LM stand-in that records the keyword arguments of every call."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
        self.reward_head = torch.nn.Linear(_HIDDEN, 1)
        self.calls: list[dict] = []

    def _out(self, input_ids):
        hidden = self.embed(input_ids)
        return SimpleNamespace(hidden_states=(hidden,), logits=None)


class _NoArgument(_Recorder):
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        self.calls.append({"attention_mask": attention_mask,
                           "output_hidden_states": output_hidden_states})
        return self._out(input_ids)


class _KwargsOnly(_Recorder):
    """A remote-code style forward: whatever it takes arrives in ``**kwargs``."""

    def forward(self, input_ids, **kwargs):
        self.calls.append(dict(kwargs))
        return self._out(input_ids)


@pytest.mark.parametrize("cls", [_NoArgument, _KwargsOnly], ids=["no-argument", "kwargs-only"])
def test_a_forward_without_the_argument_trains_and_is_called_without_it(cls):
    from soup_cli.utils.prm import forward_takes_logits_to_keep

    model = cls()
    assert forward_takes_logits_to_keep(model) is False
    trainer_self = SimpleNamespace(model=model)

    loss = _compute_loss()(trainer_self, model, _batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert model.calls and all("logits_to_keep" not in call for call in model.calls)
    assert float(model.embed.weight.grad.abs().sum()) > 0.0
    assert model.reward_head.weight.grad is not None
    assert trainer_self._prm_logits_to_keep is False


def test_a_model_with_no_forward_attribute_reads_as_not_taking_it():
    from soup_cli.utils.prm import forward_takes_logits_to_keep

    assert forward_takes_logits_to_keep(SimpleNamespace()) is False
    assert forward_takes_logits_to_keep(SimpleNamespace(forward=len)) is False


# --- the scorer -------------------------------------------------------------------------


class _WordTokenizer:
    """One id per whitespace word, inside the tiny vocabulary."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [4 + (sum(map(ord, w)) % (_VOCAB - 4)) for w in text.split()]}


_PROMPTS = ["two plus two", "three plus one", "what is five"]
_COMPLETIONS = [
    "we add two and two\nso the answer is four",
    "start with three\nadd one\nthe answer is four",
    "five is five",
]


def _scorer(*, keep=None):
    from soup_cli.utils.prm_reward import PRMScorer

    scorer = PRMScorer("./prm", aggregate="min", device="cpu")
    model = _tiny_llama().eval()
    model.requires_grad_(False)
    scorer._model = model
    scorer._tokenizer = _WordTokenizer()
    if keep is not None:
        scorer._logits_to_keep = keep  # today's call when False
    return scorer, _logits_shapes(model)


def test_the_batched_scorer_returns_todays_scores_on_one_position():
    kept, kept_shapes = _scorer()
    full, full_shapes = _scorer(keep=False)

    scores = kept(_COMPLETIONS, prompts=_PROMPTS)

    assert scores == full(_COMPLETIONS, prompts=_PROMPTS)
    assert len(set(scores)) == 3, scores  # three different rows, not one constant
    assert [s[1] for s in kept_shapes] == [1] and [s[1] for s in full_shapes] != [1]
    assert kept_shapes[0][0] == 3 and kept_shapes[0][2] == _VOCAB


def test_the_per_completion_scorer_returns_todays_score_on_one_position():
    from soup_cli.utils.prm_reward import split_steps

    kept, kept_shapes = _scorer()
    full, full_shapes = _scorer(keep=False)
    steps = split_steps(_COMPLETIONS[1])

    score = kept._score_one(_PROMPTS[1], steps)

    assert score == full._score_one(_PROMPTS[1], steps)
    assert kept_shapes == [(1, 1, _VOCAB)]
    assert full_shapes[0][1] > 1


def test_the_scorer_checks_the_signature_once():
    scorer, _ = _scorer()
    assert scorer._logits_to_keep is None
    scorer(_COMPLETIONS, prompts=_PROMPTS)
    scorer._score_one(_PROMPTS[0], ["two plus two"])
    assert scorer._logits_to_keep is True
    assert scorer._forward_kwargs() == {"logits_to_keep": 1}


def test_a_scorer_model_without_the_argument_is_called_without_it():
    from soup_cli.utils.prm_reward import PRMScorer

    scorer = PRMScorer("./prm", aggregate="min", device="cpu")
    scorer._model = _NoArgument().eval()
    scorer._tokenizer = _WordTokenizer()

    scores = scorer(_COMPLETIONS, prompts=_PROMPTS)

    assert len(scores) == 3
    assert scorer._model.calls and all("logits_to_keep" not in c for c in scorer._model.calls)
    assert scorer._forward_kwargs() == {}
