"""Does ``--resume`` continue a layer-streamed SFT run EXACTLY?

The question: N optimizer steps in one go, against N/2 steps, a saved checkpoint,
a fresh trainer built from the same config that resumes from it, and N/2 more
steps. "Exactly" is checked field by field, with ``torch.equal`` and ``==``, never
``allclose``: the LoRA adapter tensors, every tensor of the optimizer state and its
step counters, the LR scheduler (through the learning rate each step trains at),
the per-microbatch losses, the micro-batches themselves (the data order, so a wrong
skip shows up as different token ids), the global python / numpy / torch RNG streams
and the bytes of the saved ``adapter_model.safetensors``.

``tests/test_layer_stream_boundary_save.py::test_trainer_checkpoint_and_resume`` proves
the restore is not a no-op (the adapter and the optimizer step counters come back).
It does not compare a resumed run with an uninterrupted one, which is what a user who
resumes after a crash needs.

Each variant runs through the real ``setup()`` path of ``SFTTrainerWrapper``:

* ``streamed-*`` stream the decoder layers (and, untied, the embedding and the head as
  large boundaries) from a tiny Llama's shards.
* ``resident-*`` is the control: the same model, data and trainer with
  ``stream_layers: false``. A divergence the control shares is a general ``Trainer``
  property, one only the streamed variants show is streaming-specific.
* ``*-accumulating`` takes two micro-batches per optimizer step, so a resume has to
  skip whole accumulation groups. ``*-dropout`` trains with LoRA dropout, which makes
  the restored RNG stream decide the losses.

The first leg is cut by a callback and not by a smaller ``max_steps``: the LR
schedule is built from ``max_steps``, so a leg with ``max_steps=2`` would train on a
different schedule from the uninterrupted run and could never be equal to it.

Every run also draws once from each global RNG at the end of each step. With dropout 0
nothing else consumes randomness, so without the draw a skipped RNG restore would be
invisible: reseeding at ``setup()`` already puts every stream where the checkpoint had it.

Float32 on the CPU: on an accelerator the streamed dtype is bf16 and the bit-exact
claim is a different one (the GPU tier measures it).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.test_issue1048_lora_head_save_resume import _requires_train_extra

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

TOTAL_STEPS = 4
CUT_AT = 2
LEARNING_RATE = 5e-3

#: (id, stream_layers, tie_word_embeddings, gradient_accumulation_steps, lora dropout)
VARIANTS = (
    ("streamed-untied", True, False, 1, 0.0),
    ("streamed-tied", True, True, 1, 0.0),
    ("streamed-untied-accumulating", True, False, 2, 0.0),
    ("streamed-untied-dropout", True, False, 1, 0.1),
    ("resident", False, True, 1, 0.0),
    ("resident-accumulating", False, True, 2, 0.0),
    ("resident-dropout", False, True, 1, 0.1),
)

#: Eight different rows. The shared fixture rows are eight copies of one, which would
#: make every batch identical and the data order unobservable.
_CONVERSATIONS = (
    ("hi", "hello world"),
    ("the cat", "sat on the mat"),
    ("question", "good answer"),
    ("yo soup", "bad"),
    ("hello", "the cat sat on the mat"),
    ("good question", "hi yo"),
    ("soup", "hello hello world world"),
    ("cat mat", "answer"),
)


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
        for prompt, answer in _CONVERSATIONS
    ]


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def _detached(value: Any) -> Any:
    """A deep copy of ``value`` with every tensor cloned onto the CPU."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detached(item) for item in value)
    return value


def _adapters(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def _optimizer_state(optimizer: Any, model: Any) -> dict[str, Any]:
    """The optimizer's state keyed by parameter NAME, hyper-parameters by group.

    ``state_dict()`` indexes parameters by position, which says nothing about which
    tensor differs. Positions are assigned in ``param_groups`` order.
    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    raw = optimizer.state_dict()
    ordered = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    by_name = {
        names.get(id(ordered[index]), f"param#{index}"): _detached(state)
        for index, state in raw["state"].items()
    }
    groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in raw["param_groups"]
    ]
    return {"state": by_name, "groups": _detached(groups)}


def _rng_state() -> dict[str, Any]:
    import numpy
    import torch

    numpy_state = numpy.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        "torch": bytes(torch.get_rng_state().numpy().tobytes()),
    }


def _draw_from_every_stream() -> None:
    import numpy
    import torch

    random.random()
    numpy.random.random()
    torch.rand(1)


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Every place ``left`` and ``right`` are not bit-identical, as readable lines."""
    import torch

    where = path or "<root>"
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            return [f"{where}: a tensor against a {type(left).__name__}/{type(right).__name__}"]
        if left.dtype != right.dtype or left.shape != right.shape:
            return [
                f"{where}: {left.dtype}{tuple(left.shape)} against "
                f"{right.dtype}{tuple(right.shape)}"
            ]
        if torch.equal(left, right):
            return []
        gap = (left.double() - right.double()).abs().max().item()
        return [f"{where}: tensors differ, max |delta| {gap:.3e}"]
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                found.append(f"{path}/{key}: present on one side only")
            else:
                found.extend(_differences(left[key], right[key], f"{path}/{key}"))
        return found
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{where}: length {len(left)} against {len(right)}"]
        found = []
        for index, (first, second) in enumerate(zip(left, right)):
            found.extend(_differences(first, second, f"{path}[{index}]"))
        return found
    return [] if left == right else [f"{where}: {left!r} against {right!r}"]


def _assert_identical(left: Any, right: Any, what: str) -> None:
    found = _differences(left, right)
    assert not found, f"{what} is not bit-identical ({len(found)} differences):\n" + "\n".join(
        found[:12]
    )


# ---------------------------------------------------------------------------
# one leg of a run
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    """Everything one ``wrapper.train()`` call left behind."""

    checkpoints: Path
    start_adapters: dict[str, Any]
    streamed: bool = False
    final_adapters: dict[str, Any] = field(default_factory=dict)
    adapters_after: dict[int, dict[str, Any]] = field(default_factory=dict)
    optimizer_after: dict[int, dict[str, Any]] = field(default_factory=dict)
    rng_before: dict[int, dict[str, Any]] = field(default_factory=dict)
    rng_at_end: dict[str, Any] = field(default_factory=dict)
    losses: list[float] = field(default_factory=list)
    batches: list[Any] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    global_step: int = -1
    optimizer_class: str = ""
    scheduler_class: str = ""
    at_train_begin: dict[str, Any] = field(default_factory=dict)

    def adapter_digest(self, step: int) -> str:
        path = self.checkpoints / f"checkpoint-{step}" / "adapter_model.safetensors"
        return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Scenario:
    continuous: Leg
    first: Leg
    resumed: Leg
    stream: bool


def _build(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool,
    tie: bool,
    accumulate: int,
    dropout: float,
) -> Any:
    from tests.conftest import accelerator_device
    from tests.test_v07204 import (
        _stream_cfg,
        _tiny_llama_dir,
        _wrapper_for,
        _write_tiny_tokenizer,
    )

    weights, _, _ = _tiny_llama_dir(root, n_layers=2, tie=tie, hidden=64, vocab=64)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(root / "cache"))
    monkeypatch.chdir(root)
    cfg = _stream_cfg(
        weights,
        root / "out",
        task="sft",
        stream_layers=stream,
        batch_size=1,
        gradient_accumulation_steps=accumulate,
        lr=LEARNING_RATE,
        save_steps=CUT_AT,
        lora={"r": 4, "alpha": 8, "dropout": dropout, "target_modules": ["q_proj", "v_proj"]},
    )
    wrapper = _wrapper_for("sft")(cfg, device=accelerator_device())
    wrapper.setup({"train": _rows()})
    wrapper.trainer.args.max_steps = TOTAL_STEPS
    return wrapper


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_after: int | None = None,
    resume_from: Path | None = None,
    **settings: Any,
) -> Leg:
    """A fresh wrapper from a freshly built config, trained once."""
    from transformers import TrainerCallback

    wrapper = _build(root, monkeypatch, **settings)
    trainer = wrapper.trainer
    leg = Leg(checkpoints=root / "out", start_adapters=_adapters(wrapper.model))
    leg.streamed = getattr(wrapper, "_stream_runtime", None) is not None

    real_training_step = trainer.training_step

    def spy(model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        leg.batches.append(_detached(inputs["input_ids"]))
        loss = real_training_step(model, inputs, *args, **kwargs)
        leg.losses.append(float(loss.detach().cpu()))
        return loss

    trainer.training_step = spy

    class Recorder(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            leg.at_train_begin = {
                "global_step": state.global_step,
                "adapters": _adapters(wrapper.model),
            }

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            leg.rng_before[state.global_step] = _rng_state()

        def on_step_end(
            self, args: Any, state: Any, control: Any, optimizer: Any = None, **kwargs: Any
        ) -> None:
            leg.adapters_after[state.global_step] = _adapters(wrapper.model)
            leg.optimizer_after[state.global_step] = _optimizer_state(optimizer, wrapper.model)
            leg.optimizer_class = type(getattr(optimizer, "optimizer", optimizer)).__name__
            _draw_from_every_stream()
            if stop_after is not None and state.global_step >= stop_after:
                control.should_training_stop = True

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            leg.rng_at_end = _rng_state()

    trainer.add_callback(Recorder())
    try:
        wrapper.train(resume_from_checkpoint=None if resume_from is None else str(resume_from))
        leg.final_adapters = _adapters(wrapper.model)
        leg.global_step = trainer.state.global_step
        leg.log = [
            {key: entry.get(key) for key in ("step", "loss", "grad_norm", "learning_rate", "epoch")}
            for entry in trainer.state.log_history
            if "loss" in entry
        ]
        leg.scheduler_class = type(trainer.lr_scheduler).__name__
    finally:
        wrapper._close_stream_runtime()
    return leg


@pytest.fixture(scope="module", params=VARIANTS, ids=[variant[0] for variant in VARIANTS])
def scenario(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    _requires_train_extra()
    _, stream, tie, accumulate, dropout = request.param
    root = tmp_path_factory.mktemp("stream-resume-exact")
    settings = {"stream": stream, "tie": tie, "accumulate": accumulate, "dropout": dropout}
    with pytest.MonkeyPatch.context() as monkeypatch:
        continuous = _run(root / "continuous", monkeypatch, **settings)
        first = _run(root / "first", monkeypatch, stop_after=CUT_AT, **settings)
        cut = root / "first" / "out" / f"checkpoint-{CUT_AT}"
        assert cut.is_dir(), sorted(path.name for path in (root / "first" / "out").iterdir())
        resumed = _run(root / "resumed", monkeypatch, resume_from=cut, **settings)
    return Scenario(continuous=continuous, first=first, resumed=resumed, stream=stream)


# ---------------------------------------------------------------------------
# the instrument must be able to see a difference
# ---------------------------------------------------------------------------


class TestTheComparisonSeesADifference:
    def test_identical_trees_have_no_difference(self) -> None:
        import torch

        tree = {"a": torch.arange(4.0), "b": [torch.ones(2), 3, "x"], "c": {"d": 1.5}}
        assert _differences(tree, _detached(tree)) == []

    def test_one_ulp_in_one_element(self) -> None:
        import torch

        left = {"w": torch.full((3, 3), 0.25)}
        right = {"w": left["w"].clone()}
        right["w"][1, 2] = torch.nextafter(right["w"][1, 2], torch.tensor(1.0))
        found = _differences(left, right)
        assert len(found) == 1 and found[0].startswith("/w: tensors differ")

    def test_a_dtype_a_shape_a_missing_key_and_a_scalar(self) -> None:
        import torch

        assert _differences(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
        assert _differences(torch.zeros(2), torch.zeros(3))
        assert _differences({"a": 1}, {"a": 1, "b": 2})
        assert _differences([1, 2], [1, 2, 3])
        assert _differences({"step": 3}, {"step": 4})

    def test_the_rng_snapshot_moves_when_a_stream_is_drawn_from(self) -> None:
        before = _rng_state()
        assert _differences(before, _rng_state()) == []
        _draw_from_every_stream()
        found = _differences(before, _rng_state())
        assert {re.split(r"[:\[]", line)[0] for line in found} == {"/python", "/numpy", "/torch"}


class TestTheRunsAreComparable:
    def test_streaming_is_on_exactly_where_the_variant_says(self, scenario: Scenario) -> None:
        # The control is only a control if it is resident, and the streamed variants only
        # test streaming if the setup path really built a streamed model.
        for leg in (scenario.continuous, scenario.first, scenario.resumed):
            assert leg.streamed is scenario.stream

    def test_all_three_runs_start_from_the_same_adapter(self, scenario: Scenario) -> None:
        start = scenario.continuous.start_adapters
        assert start
        _assert_identical(start, scenario.first.start_adapters, "the first leg's start")
        _assert_identical(start, scenario.resumed.start_adapters, "the resumed run's start")

    def test_the_legs_are_the_length_the_scenario_says(self, scenario: Scenario) -> None:
        assert scenario.continuous.global_step == TOTAL_STEPS
        assert scenario.first.global_step == CUT_AT
        assert scenario.resumed.global_step == TOTAL_STEPS
        assert scenario.resumed.at_train_begin["global_step"] == CUT_AT
        assert scenario.continuous.at_train_begin["global_step"] == 0

    def test_training_moves_the_adapter_and_the_batches_differ(self, scenario: Scenario) -> None:
        # Without these the equality tests below could pass on a run that learns nothing
        # or sees one row repeated.
        leg = scenario.continuous
        assert _differences(leg.start_adapters, leg.adapters_after[CUT_AT])
        assert _differences(leg.adapters_after[CUT_AT], leg.adapters_after[TOTAL_STEPS])
        state = leg.optimizer_after[TOTAL_STEPS]["state"]
        assert state
        assert all(float(entry["step"]) == TOTAL_STEPS for entry in state.values())
        assert len({tuple(batch.flatten().tolist()) for batch in leg.batches}) > 1
        assert len(set(leg.losses)) > 1
        assert all(loss == loss and abs(loss) < float("inf") for loss in leg.losses)


# ---------------------------------------------------------------------------
# the state a resumed run starts from
# ---------------------------------------------------------------------------


class TestWhereTheResumedRunStarts:
    def test_the_adapter_is_the_one_the_uninterrupted_run_had_at_the_cut(
        self, scenario: Scenario
    ) -> None:
        _assert_identical(
            scenario.continuous.adapters_after[CUT_AT],
            scenario.resumed.at_train_begin["adapters"],
            f"the adapter a resumed run starts from (step {CUT_AT})",
        )

    def test_the_first_leg_is_the_start_of_the_uninterrupted_run(self, scenario: Scenario) -> None:
        for step in range(1, CUT_AT + 1):
            _assert_identical(
                scenario.continuous.adapters_after[step],
                scenario.first.adapters_after[step],
                f"the first leg's adapter after step {step}",
            )
            _assert_identical(
                scenario.continuous.optimizer_after[step],
                scenario.first.optimizer_after[step],
                f"the first leg's optimizer after step {step}",
            )

    def test_the_saved_cut_checkpoint_is_byte_identical(self, scenario: Scenario) -> None:
        assert scenario.continuous.adapter_digest(CUT_AT) == scenario.first.adapter_digest(CUT_AT)


# ---------------------------------------------------------------------------
# the continuation
# ---------------------------------------------------------------------------


class TestAResumedRunContinuesExactly:
    def test_the_adapter_after_every_resumed_step(self, scenario: Scenario) -> None:
        for step in range(CUT_AT + 1, TOTAL_STEPS + 1):
            _assert_identical(
                scenario.continuous.adapters_after[step],
                scenario.resumed.adapters_after[step],
                f"the adapter after step {step}",
            )

    def test_the_final_adapter(self, scenario: Scenario) -> None:
        _assert_identical(
            scenario.continuous.final_adapters, scenario.resumed.final_adapters, "the final adapter"
        )

    def test_the_optimizer_state_after_every_resumed_step(self, scenario: Scenario) -> None:
        # Every tensor of every parameter (exp_avg, exp_avg_sq, step) and the group
        # hyper-parameters, including the learning rate the scheduler set for that step.
        for step in range(CUT_AT + 1, TOTAL_STEPS + 1):
            _assert_identical(
                scenario.continuous.optimizer_after[step],
                scenario.resumed.optimizer_after[step],
                f"the optimizer state after step {step}",
            )

    def test_the_step_counters(self, scenario: Scenario) -> None:
        assert scenario.resumed.global_step == scenario.continuous.global_step == TOTAL_STEPS
        for entry in scenario.resumed.optimizer_after[TOTAL_STEPS]["state"].values():
            assert float(entry["step"]) == TOTAL_STEPS

    def test_the_learning_rate_schedule(self, scenario: Scenario) -> None:
        assert scenario.resumed.scheduler_class == scenario.continuous.scheduler_class
        assert scenario.resumed.optimizer_class == scenario.continuous.optimizer_class
        rates = [entry["learning_rate"] for entry in scenario.continuous.log]
        assert len(set(rates)) > 1, "a constant schedule would not test the scheduler restore"
        _assert_identical(
            [(entry["step"], entry["learning_rate"]) for entry in scenario.continuous.log],
            [(entry["step"], entry["learning_rate"]) for entry in scenario.resumed.log],
            "the learning rate logged at each step",
        )

    def test_the_per_microbatch_losses(self, scenario: Scenario) -> None:
        both_legs = scenario.first.losses + scenario.resumed.losses
        assert len(both_legs) == len(scenario.continuous.losses)
        assert both_legs == scenario.continuous.losses, (
            f"losses differ: continuous {scenario.continuous.losses} "
            f"against first leg + resumed {both_legs}"
        )

    def test_the_logged_loss_and_gradient_norm(self, scenario: Scenario) -> None:
        _assert_identical(
            scenario.continuous.log, scenario.resumed.log, "the log history (loss, grad_norm, lr)"
        )

    def test_the_micro_batches_are_the_ones_the_uninterrupted_run_saw(
        self, scenario: Scenario
    ) -> None:
        seen = scenario.first.batches + scenario.resumed.batches
        _assert_identical(
            scenario.continuous.batches, seen, "the token ids fed to the model, in order"
        )

    def test_the_saved_final_checkpoint_is_byte_identical(self, scenario: Scenario) -> None:
        assert scenario.continuous.adapter_digest(TOTAL_STEPS) == scenario.resumed.adapter_digest(
            TOTAL_STEPS
        )


class TestTheRandomStreams:
    def test_the_rng_a_resumed_step_starts_with(self, scenario: Scenario) -> None:
        # HF restores the saved RNG after skipping the consumed batches and before the
        # first resumed step, so the comparison point is on_step_begin, not on_train_begin.
        for step in range(CUT_AT, TOTAL_STEPS):
            _assert_identical(
                scenario.continuous.rng_before[step],
                scenario.resumed.rng_before[step],
                f"the python / numpy / torch RNG before step {step + 1}",
            )

    def test_the_rng_at_the_end(self, scenario: Scenario) -> None:
        _assert_identical(
            scenario.continuous.rng_at_end, scenario.resumed.rng_at_end, "the RNG at the end"
        )
