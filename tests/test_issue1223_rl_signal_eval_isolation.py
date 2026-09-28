"""#1223 — an evaluation pass must not reach the reward-hack / echo-trap signal.

With ``training.eval_steps`` set, GRPO evaluates held-out prompts by generating
completions and scoring them with the SAME wrapped reward functions training
uses. Those functions also feed the shared ``RLSignalBuffer`` that the
reward-hack detector, the mitigation controller and the echo-trap detector read
at every step end. An evaluation's rewards describe held-out prompts, not a
training step, so they must never reach those readers.

When is the leak real? TRL regenerates every ``steps_per_generation *
num_iterations`` micro-steps. At the defaults Soup uses today, every optimizer
step starts with a fresh generation that overwrites the buffer before the next
reader looks. With generation reuse (``num_iterations > 1``, which TRL
supports), a step reads the buffer with no generation in between, and without
isolation it reads the evaluation's rewards. The invariant must not depend on
the cadence, so the evaluation pass is kept out of the buffer altogether.

The acceptance property: a run that evaluates leaves the detector and
controller state bit-identical to the same run without evaluation. It is shown
twice. Once on the callbacks themselves, driven in Hugging Face's event order
with generation reuse, which is exact and fast. Once on a real tiny GRPO run on
CPU, where a prompt-length reward keeps the detector's input independent of
what the model samples.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
from pathlib import Path

import pytest

# ==========================================================================
# the buffer and the wrapper
# ==========================================================================


def _record(buffer, rewards, completions=("a", "b")):
    buffer.record(func_name="rm", completions=list(completions), rewards=list(rewards))


class TestTheBufferPause:
    def test_a_paused_record_is_dropped(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buffer = RLSignalBuffer()
        _record(buffer, [1.0, 0.0])
        before = buffer.snapshot()
        with buffer.paused():
            _record(buffer, [9.0, 9.0], completions=("eval", "eval"))
        assert buffer.snapshot() == before

    def test_recording_resumes_after_the_block(self):
        """The control: the pause ends, it does not blind the buffer for good."""
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buffer = RLSignalBuffer()
        with buffer.paused():
            _record(buffer, [9.0, 9.0])
        _record(buffer, [1.0, 0.0])
        assert buffer.snapshot()["rewards"] == [1.0, 0.0]

    def test_the_pause_nests(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buffer = RLSignalBuffer()
        with buffer.paused():
            with buffer.paused():
                pass
            _record(buffer, [9.0, 9.0])  # still inside the outer block
        assert buffer.snapshot()["rewards"] == []

    def test_an_exception_still_ends_the_pause(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buffer = RLSignalBuffer()
        with pytest.raises(RuntimeError, match="evaluation failed"):
            with buffer.paused():
                raise RuntimeError("evaluation failed")
        _record(buffer, [1.0, 0.0])
        assert buffer.snapshot()["rewards"] == [1.0, 0.0]


class _EvalOnlyTrainer:
    """The smallest trainer shape `exclude_evaluation` wraps."""

    def __init__(self, reward_fn, fail=False):
        self.reward_fn = reward_fn
        self.fail = fail
        self.calls = []

    def evaluate(self, *args, **kwargs):
        """Score held-out prompts the way GRPO's evaluation does."""
        self.calls.append((args, kwargs))
        self.reward_fn(prompts=["p", "p"], completions=["eval", "eval"])
        if self.fail:
            raise RuntimeError("evaluation failed")
        return {"eval_loss": 0.5}


class TestExcludeEvaluation:
    def _setup(self, fail=False):
        from soup_cli.utils.rl_signal_buffer import (
            RLSignalBuffer,
            exclude_evaluation,
            wrap_reward_funcs,
        )

        buffer = RLSignalBuffer()

        def rm(prompts=None, completions=None, **kwargs):
            return [7.0 if c == "eval" else 1.0 for c in completions]

        trainer = _EvalOnlyTrainer(wrap_reward_funcs(rm, buffer), fail=fail)
        exclude_evaluation(trainer, buffer)
        return buffer, trainer

    def test_an_evaluation_leaves_the_buffer_untouched(self):
        buffer, trainer = self._setup()
        trainer.reward_fn(prompts=["p", "p"], completions=["train", "train"])
        before = buffer.snapshot()
        assert trainer.evaluate("keys", metric_key_prefix="eval") == {"eval_loss": 0.5}
        assert buffer.snapshot() == before
        assert trainer.calls == [(("keys",), {"metric_key_prefix": "eval"})]

    def test_training_calls_still_record(self):
        buffer, trainer = self._setup()
        trainer.evaluate()
        trainer.reward_fn(prompts=["p", "p"], completions=["train", "train"])
        assert buffer.snapshot()["rewards"] == [1.0, 1.0]

    def test_a_failed_evaluation_does_not_leave_the_buffer_paused(self):
        buffer, trainer = self._setup(fail=True)
        with pytest.raises(RuntimeError, match="evaluation failed"):
            trainer.evaluate()
        trainer.reward_fn(prompts=["p", "p"], completions=["train", "train"])
        assert buffer.snapshot()["rewards"] == [1.0, 1.0]

    def test_a_trainer_without_evaluate_is_left_alone(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer, exclude_evaluation

        trainer = types.SimpleNamespace(add_callback=None)
        exclude_evaluation(trainer, RLSignalBuffer())
        assert not hasattr(trainer, "evaluate")


class TestTheDetectorLogFallback:
    """Without a buffer (PPO's shape) the detector reads ``reward`` from logs.
    An evaluation record is skipped whole: its metrics describe held-out
    prompts, whatever keys it happens to carry."""

    def _detector(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        return build_reward_hack_callback(detector="info_rm", halt_on_hack=False)

    def test_a_training_record_is_observed(self):
        cb = self._detector()
        state = types.SimpleNamespace(global_step=1, log_history=[])
        cb.on_log(None, state, None, logs={"reward": 1.0, "reward_std": 0.5})
        assert cb.last_report() is not None

    def test_an_evaluation_record_is_skipped(self):
        cb = self._detector()
        state = types.SimpleNamespace(global_step=1, log_history=[])
        logs = {"eval_loss": 0.1, "eval_reward": 1.0, "reward": 1.0, "reward_std": 0.5}
        cb.on_log(None, state, None, logs=logs)
        assert cb.last_report() is None
        assert state.log_history == []


# ==========================================================================
# bit-identical state, on the callbacks, in Hugging Face's event order
# ==========================================================================
def _text_value(text: str) -> float:
    """Train completions carry their reward: 'v=<float> ...'."""
    head = text.split()[0]
    return float(head[2:]) if head.startswith("v=") else 0.0


def _rm_a(prompts=None, completions=None, **kwargs):
    return [_text_value(str(c)) for c in completions]


def _rm_b(prompts=None, completions=None, **kwargs):
    return [0.9 * _text_value(str(c)) + 0.05 * i for i, c in enumerate(completions)]


def _train_completions(generation: int) -> list[str]:
    # Well separated, varying by generation, a few words long.
    values = [0.0, 0.1 * generation, 1.0, 1.0 - 0.05 * generation]
    return [f"v={v} answer step {generation} item {i}" for i, v in enumerate(values)]


#: What an evaluation would feed a leaking buffer: bunched rewards and a
#: repetitive, much longer completion -- a HACK verdict, an echo trap and a
#: length-trend jump if any of it got through.
_EVAL_COMPLETIONS = ["v=5.0 " + "echo " * 40] * 4


class _FakeGrpoTrainer:
    """Calls the reward functions the way TRL does, and evaluates the way
    ``GRPOTrainer.evaluate`` does: score held-out completions, then fire
    ``on_log`` and ``on_evaluate`` on every callback."""

    def __init__(self, reward_funcs):
        self.reward_funcs = reward_funcs
        self.callbacks = []
        self.beta = 0.04
        self.args = types.SimpleNamespace(beta=0.04)
        self.state = types.SimpleNamespace(global_step=0, log_history=[])
        self.control = types.SimpleNamespace(should_training_stop=False)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def score(self, completions):
        for fn in self.reward_funcs:
            fn(prompts=["p"] * len(completions), completions=completions, trainer_state=None)

    def evaluate(self, *args, **kwargs):
        self.score(_EVAL_COMPLETIONS)
        metrics = {"eval_loss": 0.25, "eval_reward": 5.0, "eval_reward_std": 0.0, "epoch": 1.0}
        for cb in self.callbacks:
            cb.on_log(self.args, self.state, self.control, logs=dict(metrics))
            cb.on_evaluate(self.args, self.state, self.control, metrics=dict(metrics))
        return metrics


def _tcfg(case: str):
    from soup_cli.config.schema import TrainingConfig

    return {
        "detector+echo_trap": TrainingConfig(
            reward_hack_detector="info_rm", echo_trap_enabled=True
        ),
        "rm_ensemble": TrainingConfig(reward_hack_detector="rm_ensemble"),
        "kl_control+shaping": TrainingConfig(
            reward_hack_detector="info_rm",
            reward_hack_mitigation="kl_control",
            reward_hack_signals=["info_rm", "length_trend", "repetition"],
            reward_hack_signal_smoothing="ema",
            reward_hack_reward_shaping=True,
            reward_hack_shaping_kind="length",
            reward_hack_shaping_strength=0.5,
        ),
        "pid_lagrangian": TrainingConfig(
            reward_hack_detector="info_rm", reward_hack_mitigation="pid_lagrangian"
        ),
    }[case]


_REFERENCES = frozenset(
    {"buffer", "tokenizer", "_trainer", "log_writer", "rl_checkpoint_cb", "_lock"}
)


def _fingerprint(obj):
    """Every value a callback carries between steps, floats as exact bits."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj.hex() if math.isfinite(obj) else repr(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _fingerprint(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _fingerprint(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_fingerprint(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {
            type(obj).__name__: {
                k: _fingerprint(v) for k, v in vars(obj).items() if k not in _REFERENCES
            }
        }
    return repr(obj)


def _simulate(root: Path, case: str, *, evaluate: bool, bypass_isolation: bool = False):
    """Six optimizer steps, a training generation every second step (TRL's
    generation reuse), and an evaluation after every step when asked."""
    from soup_cli.utils.peft_wiring import attach_rl_callbacks
    from soup_cli.utils.reward_hack_control import apply_reward_shaping
    from soup_cli.utils.rl_signal_buffer import RLSignalBuffer, wrap_reward_funcs

    tcfg = _tcfg(case)
    buffer = RLSignalBuffer()
    # The order grpo.py uses: shaping first, then the capture shim.
    reward_funcs = wrap_reward_funcs(apply_reward_shaping([_rm_a, _rm_b], tcfg), buffer)
    trainer = _FakeGrpoTrainer(reward_funcs)
    root.mkdir()
    attached = attach_rl_callbacks(
        trainer, tcfg, buffer=buffer, tokenizer=None, output_dir=str(root), task="grpo"
    )
    assert attached == len(trainer.callbacks) >= 1
    evaluate_fn = trainer.evaluate.__wrapped__ if bypass_isolation else trainer.evaluate
    for step in range(1, 7):
        if step % 2 == 1:
            trainer.score(_train_completions(step))
        trainer.state.global_step = step
        for cb in trainer.callbacks:
            cb.on_step_end(trainer.args, trainer.state, trainer.control, model=None, optimizer=None)
        if evaluate:
            evaluate_fn()
    mitigation_log = root / "mitigation_log.jsonl"
    log_lines = []
    if mitigation_log.exists():
        for line in mitigation_log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            entry.pop("ts", None)
            log_lines.append(entry)
    return {
        "callbacks": [_fingerprint(cb) for cb in trainer.callbacks],
        "log_history": _fingerprint(trainer.state.log_history),
        "stop": trainer.control.should_training_stop,
        "beta": (_fingerprint(trainer.beta), _fingerprint(trainer.args.beta)),
        "mitigation_log": _fingerprint(log_lines),
    }


_CASES = ("detector+echo_trap", "rm_ensemble", "kl_control+shaping", "pid_lagrangian")


class TestEvaluationLeavesTheSignalUntouched:
    @pytest.mark.parametrize("case", _CASES)
    def test_state_is_bit_identical_with_and_without_evaluation(
        self, tmp_path, monkeypatch, case
    ):
        pytest.importorskip("transformers")
        monkeypatch.chdir(tmp_path)
        with_eval = _simulate(tmp_path / "with", case, evaluate=True)
        without = _simulate(tmp_path / "without", case, evaluate=False)
        assert with_eval == without

    @pytest.mark.parametrize("case", _CASES)
    def test_the_comparison_can_see_a_leak(self, tmp_path, monkeypatch, case):
        """Non-vacuity: the same evaluations with the isolation bypassed DO
        change the state, so the equality above is evidence, not luck."""
        pytest.importorskip("transformers")
        monkeypatch.chdir(tmp_path)
        leaked = _simulate(tmp_path / "leak", case, evaluate=True, bypass_isolation=True)
        without = _simulate(tmp_path / "without", case, evaluate=False)
        assert leaked != without


# ==========================================================================
# the same property on a real tiny GRPO run
# ==========================================================================
_PROMPT_LENGTH_REWARD = '''
def reward_fn(prompts=None, completions=None, **kwargs):
    """The prompt's word count: deterministic whatever the model samples."""
    out = []
    for prompt in prompts:
        if isinstance(prompt, list):
            prompt = " ".join(str(m.get("content", "")) for m in prompt)
        out.append(float(len(str(prompt).split())))
    return out
'''


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _real_grpo_run(root: Path, monkeypatch, *, eval_steps):
    """Four steps, generation reused for two (``num_iterations=2``), and when
    ``eval_steps`` is set an evaluation after every step. Returns what the
    detector read at each step end and the state it ended with."""
    from soup_cli.trainer.grpo import GRPOTrainerWrapper
    from tests.test_issue1223_evaluate_val_split import _cfg, _tiny_llama_dir

    root.mkdir()
    monkeypatch.chdir(root)
    (root / "reward.py").write_text(_PROMPT_LENGTH_REWARD, encoding="utf-8")
    weights = _tiny_llama_dir(root)
    over = {
        "batch_size": 4,
        "num_generations": 2,
        "reward_fn": "reward.py",
        "reward_hack_detector": "info_rm",
    }
    if eval_steps:
        over["eval_steps"] = eval_steps
    cfg = _cfg("grpo", weights, root / "out", **over)
    train = [{"prompt": " ".join(["hi"] * n)} for n in range(1, 9)]
    val = [{"prompt": " ".join(["hello"] * 40)} for _ in range(2)]
    wrapper = GRPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": train, "val": val})
    trainer = wrapper.trainer
    trainer.num_iterations = 2
    trainer.args.num_iterations = 2
    trainer.args.max_steps = 4
    detector = next(
        cb for cb in trainer.callback_handler.callbacks
        if type(cb).__name__ == "RewardHackCallback"
    )
    seen = []
    on_step_end = detector.on_step_end

    def spy(*args, **kwargs):
        seen.append(tuple(wrapper._rl_buffer.snapshot()["rewards"]))
        return on_step_end(*args, **kwargs)

    detector.on_step_end = spy
    trainer.train()
    history = trainer.state.log_history
    return {
        "evals": [e["step"] for e in history if "eval_loss" in e],
        "seen": seen,
        "signals": [
            (e["reward_hack_signal"], e["reward_hack_verdict"])
            for e in history
            if "reward_hack_signal" in e
        ],
        "detector": _fingerprint(detector),
    }


class TestARealGrpoRun:
    def test_evaluate_is_wrapped_and_leaves_the_buffer_untouched(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        from tests.test_issue1223_evaluate_val_split import _cfg, _tiny_llama_dir

        monkeypatch.chdir(tmp_path)
        (tmp_path / "reward.py").write_text(_PROMPT_LENGTH_REWARD, encoding="utf-8")
        cfg = _cfg(
            "grpo",
            _tiny_llama_dir(tmp_path),
            tmp_path / "out",
            batch_size=4,
            reward_fn="reward.py",
            reward_hack_detector="info_rm",
            eval_steps=1,
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup(
            {
                "train": [{"prompt": "hi " * n} for n in range(1, 9)],
                "val": [{"prompt": "hello " * 40} for _ in range(2)],
            }
        )
        assert hasattr(wrapper.trainer.evaluate, "__wrapped__")
        before = wrapper._rl_buffer.snapshot()
        metrics = wrapper.trainer.evaluate()
        assert "eval_loss" in metrics
        assert wrapper._rl_buffer.snapshot() == before

    def test_the_detector_state_is_bit_identical_with_and_without_evaluation(
        self, tmp_path, monkeypatch
    ):
        _requires_train_extra()
        with_eval = _real_grpo_run(tmp_path / "with", monkeypatch, eval_steps=1)
        without = _real_grpo_run(tmp_path / "without", monkeypatch, eval_steps=None)
        assert with_eval["evals"] == [1, 2, 3, 4]
        assert without["evals"] == []
        # Non-vacuity: the detector read four rewards per step and observed a
        # signal, so there was something for an evaluation to disturb.
        assert all(len(rewards) == 4 for rewards in without["seen"])
        assert without["signals"]
        for key in ("seen", "signals", "detector"):
            assert with_eval[key] == without[key], key
