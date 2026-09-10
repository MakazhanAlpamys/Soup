"""`training.max_grad_norm` is dropped on the MLX backend.

The field is declared `max_grad_norm: float = Field(default=1.0, gt=0)`
(`config/schema.py:1098`) and forwarded into `TrainingArguments` by sixteen
transformers trainers, so the documented default clips every transformers run
at 1.0. On `backend: mlx` it reaches nothing:

* no MLX trainer file mentions it, and
* `mlx_lm 0.31.3`'s `TrainingArgs` has no such field and its `trainer.py`
  contains no `clip`/`norm` call at all, so upstream never clips either, and
* `_check_unsupported()` does not list it, so nothing is printed.

The same config therefore trains with clipping on transformers and without it
on MLX, silently. Measured on an M1 with the real library, one SGD step on a
deliberately ill-conditioned batch:

    no clipping (today's MLX)      |w - w0| = 388.025909
    clip_grad_norm(max_norm=1.0)   |w - w0| =   0.008490
    clip_grad_norm(max_norm=0.01)  |w - w0| =   0.000085

`mlx.optimizers.clip_grad_norm(grads, max_norm)` exists, and mlx-lm applies
gradients through exactly one call -- `optimizer.update(model, grad)` at
`trainer.py:259` -- so clipping is reachable by handing `train()` an optimizer
that clips before delegating, without forking upstream's loop. That call site
sits inside a function compiled with
`@partial(mx.compile, inputs=state, outputs=state)` (`trainer.py:246-248`), and
the measurements above were taken through that same compiled shape, so the
proxy is known to survive tracing rather than assumed to.

Worth stating because it bounds the impact: AdamW normalises by its second
moment, so clipping changes an AdamW step far less than an SGD one. Under the
same probe with AdamW the movement was 4.848 unclipped and 4.852 clipped. The
runs this actually rescues are `optimizer: sgd` / `lion`, and any run whose
gradients spike.
"""

import sys
import types

import pytest

from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx


class _RecordingClip:
    """Stands in for `mlx.optimizers.clip_grad_norm`, recording its max_norm."""

    def __init__(self):
        self.calls = []

    def __call__(self, grads, max_norm):
        self.calls.append(max_norm)
        # Upstream returns (clipped_grads, total_norm); the scale factor is
        # irrelevant here -- what is under test is that the call happens with
        # the configured norm and that the result is what the real optimizer
        # is then handed.
        return {"clipped": True, "from": grads}, 1.0


class _RecordingOptimizer:
    """Stands in for `mlx.optimizers.AdamW`, recording what update() receives."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = []
        self.state = {}
        self.learning_rate = kwargs.get("learning_rate", 0.0)

    def update(self, model, gradients):
        self.updates.append(gradients)


def _install(monkeypatch):
    """The shared fake MLX harness plus a recording optimizer and clipper."""
    _install_fake_mlx(monkeypatch)
    optim = sys.modules["mlx.optimizers"]
    clip = _RecordingClip()
    built = []

    def _adamw(**kwargs):
        opt = _RecordingOptimizer(**kwargs)
        built.append(opt)
        return opt

    monkeypatch.setattr(optim, "AdamW", _adamw, raising=False)
    monkeypatch.setattr(optim, "clip_grad_norm", clip, raising=False)
    return clip, built


def _run(tmp_path, monkeypatch, **training):
    """Drive the real `train()` body and return (adapter metadata, train kwargs)."""
    import json

    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    seen: dict = {}

    def _recording_train(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(sys.modules["mlx_lm.tuner.trainer"], "train", _recording_train)

    training.setdefault("epochs", 1)
    training.setdefault("batch_size", 1)
    cfg = SoupConfig(
        base="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="plaintext"),
        training=TrainingConfig(**training),
        output=str(tmp_path),
    )
    w = MLXSFTTrainerWrapper(cfg)
    w.model = _FakeMlxModel()
    w.tokenizer = object()
    w._dataset = {"train": [{"text": "hi"}] * 8, "val": []}
    w.train()
    return json.loads((tmp_path / "adapter_config.json").read_text()), seen


class TestTheConfiguredNormReachesTheOptimizer:
    """The wiring, asserted at the optimizer rather than at the plan.

    A metadata-only assertion could not observe this: the value would be
    recorded whether or not anything clipped.
    """

    def test_the_optimizer_train_receives_clips_at_the_configured_norm(
        self, tmp_path, monkeypatch
    ):
        clip, _ = _install(monkeypatch)
        _, seen = _run(tmp_path, monkeypatch, max_grad_norm=0.5)

        optimizer = seen.get("optimizer")
        assert optimizer is not None, "train() was called without an optimizer"
        optimizer.update(object(), {"w": "grads"})

        assert clip.calls == [pytest.approx(0.5)], (
            f"clip_grad_norm was called with {clip.calls}; the configured "
            "training.max_grad_norm never reached the gradients, so an MLX run "
            "trains unclipped while the same config clips on transformers"
        )

    def test_a_different_norm_is_a_different_call(self, tmp_path, monkeypatch):
        """Control: the value must be read from the config, not hardcoded to
        the schema default that the test above would also accept."""
        clip, _ = _install(monkeypatch)
        _, seen = _run(tmp_path, monkeypatch, max_grad_norm=2.5)
        seen["optimizer"].update(object(), {"w": "grads"})

        assert clip.calls == [pytest.approx(2.5)]

    def test_the_real_optimizer_is_handed_the_clipped_gradients(
        self, tmp_path, monkeypatch
    ):
        """Clipping is worthless if the unclipped gradients are applied anyway."""
        clip, built = _install(monkeypatch)
        _, seen = _run(tmp_path, monkeypatch, max_grad_norm=1.0)
        seen["optimizer"].update(object(), {"w": "raw"})

        assert built, "no underlying optimizer was constructed"
        assert built[0].updates == [{"clipped": True, "from": {"w": "raw"}}], (
            f"the underlying optimizer received {built[0].updates}; it must "
            "receive what clip_grad_norm returned, not the raw gradients"
        )

    def test_the_underlying_optimizer_is_still_reachable_through_the_wrapper(
        self, tmp_path, monkeypatch
    ):
        """mlx-lm reads `optimizer.state` (trainer.py:246) and
        `optimizer.learning_rate` (trainer.py:337) off the object it is given,
        so the wrapper must delegate rather than shadow them."""
        _, built = _install(monkeypatch)
        # `scheduler: constant` with no warmup so #686's builder returns a plain
        # float rather than a callable schedule -- the value is then checkable.
        # Delegation is asserted by identity as well, which holds either way.
        _, seen = _run(
            tmp_path, monkeypatch, lr=3e-4, max_grad_norm=1.0,
            scheduler="constant", warmup_ratio=0.0,
        )

        optimizer = seen["optimizer"]
        assert optimizer.state is built[0].state
        assert optimizer.learning_rate is built[0].learning_rate
        assert optimizer.learning_rate == pytest.approx(3e-4)

    def test_delegation_holds_when_the_learning_rate_is_a_schedule(
        self, tmp_path, monkeypatch
    ):
        """#686 hands the optimizer a CALLABLE learning rate for any real
        schedule, and mlx-lm reads `.learning_rate` off the object to report
        it (trainer.py:337). An earlier version of the test above asserted a
        float and broke when #686 merged -- the delegation was fine, the
        assumption was not."""
        _, built = _install(monkeypatch)
        _, seen = _run(
            tmp_path, monkeypatch, max_grad_norm=1.0,
            scheduler="cosine", warmup_ratio=0.25,
        )

        optimizer = seen["optimizer"]
        assert callable(optimizer.learning_rate), (
            "a cosine schedule with warmup must reach the optimizer as a "
            "callable, not collapse to a scalar"
        )
        assert optimizer.learning_rate is built[0].learning_rate


class TestTheAdapterRecordsTheNormThatRan:
    def test_the_effective_norm_is_recorded(self, tmp_path, monkeypatch):
        _install(monkeypatch)
        meta, _ = _run(tmp_path, monkeypatch, max_grad_norm=0.75)
        assert meta["max_grad_norm"] == pytest.approx(0.75)


class TestTheHarnessCanObserveTheAbsence:
    """Reject-everything control for the whole file.

    If `_install` did not actually place a recording clipper, every assertion
    above would be measuring the fake rather than the code, so the absence of
    a call is checked to be observable.
    """

    def test_no_clip_call_happens_until_the_optimizer_updates(
        self, tmp_path, monkeypatch
    ):
        clip, _ = _install(monkeypatch)
        _run(tmp_path, monkeypatch, max_grad_norm=1.0)
        assert clip.calls == [], (
            "clipping must happen per optimizer update, not once at "
            "construction time"
        )


class TestUpstreamStillHasNoClippingOfItsOwn:
    """If mlx-lm ever grows its own clipping, this module should be deleted
    rather than double-clipping. Gated, because it reads the real package."""

    def test_mlx_lm_trainingargs_has_no_grad_norm_field(self):
        trainer = pytest.importorskip("mlx_lm.tuner.trainer")
        fields = getattr(trainer.TrainingArgs, "__dataclass_fields__", {})
        assert not [f for f in fields if "norm" in f or "clip" in f], (
            "mlx-lm now exposes gradient clipping of its own; prefer it over "
            "Soup's wrapper"
        )

    def test_mlx_optimizers_still_exposes_clip_grad_norm(self):
        optim = pytest.importorskip("mlx.optimizers")
        assert hasattr(optim, "clip_grad_norm")


def test_the_fake_harness_is_not_masking_a_missing_symbol(monkeypatch):
    """`_install` adds `clip_grad_norm` to the fake with `raising=False`, which
    would also succeed if the production code called something that does not
    exist upstream. This pins that the name is real."""
    optim = pytest.importorskip("mlx.optimizers")
    assert callable(optim.clip_grad_norm)
    assert isinstance(sys.modules.get("mlx.optimizers"), types.ModuleType)
