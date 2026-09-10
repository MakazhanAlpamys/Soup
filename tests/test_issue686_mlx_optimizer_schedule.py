"""Tests for Issue #686: MLX SFT ignores warmup, scheduler and weight decay.

The MLX backend built `optim.AdamW(learning_rate=<scalar>)` and nothing else,
so `warmup_ratio`, `scheduler`, `weight_decay` and `optimizer` were validated,
accepted and dropped without a warning.

The load-bearing fact is the step unit. MLX calls a callable `learning_rate`
with `optimizer.step`, which advances once per `optimizer.update()`, and mlx-lm
calls that only every `grad_accumulation_steps` iterations -- so the schedule
counts optimizer UPDATES. A schedule built against `iters` would stretch its
warmup by that factor and never reach the cosine floor, while still producing a
perfectly plausible-looking curve. `TestTheStepUnitIsOptimizerUpdates` is the
reason this file exists.
"""

import inspect
import sys
from typing import Any, Dict, List, Tuple

import pytest

from soup_cli.trainer.mlx_optim import (
    _OPTIMIZER_MAP,
    MlxOptimizerError,
    OptimizerPlan,
    build_lr_schedule,
    plan_optimizer,
    resolve_optimizer_name,
)

BASE = dict(
    lr=2e-4,
    optimizer="adamw_torch",
    scheduler="cosine",
    warmup_ratio=0.2,
    weight_decay=0.3,
    total_updates=50,
)


def lr_at(plan, step):
    """Observed learning rate at a given optimizer-update index."""
    mx = pytest.importorskip("mlx.core")
    sched = build_lr_schedule(plan)
    return float(sched(mx.array(step))) if callable(sched) else float(sched)


class TestTheObservedLearningRateCurve:
    """The issue's explicit criterion: first, warmup boundary, final update."""

    def test_warmup_starts_at_zero_and_peaks_at_the_boundary(self):
        plan = plan_optimizer(**BASE)          # warmup = 0.2 * 50 = 10
        assert plan.warmup_updates == 10
        assert lr_at(plan, 0) == pytest.approx(0.0, abs=1e-9)
        assert lr_at(plan, 10) == pytest.approx(2e-4, rel=1e-6)

    def test_the_rate_rises_monotonically_through_warmup(self):
        plan = plan_optimizer(**BASE)
        seen = [lr_at(plan, s) for s in range(0, 11)]
        assert seen == sorted(seen)
        assert seen[0] < seen[5] < seen[10], "warmup is not actually ramping"

    def test_cosine_decays_to_zero_by_the_final_update(self):
        plan = plan_optimizer(**BASE)
        assert lr_at(plan, 50) == pytest.approx(0.0, abs=1e-9)
        mid = lr_at(plan, 30)
        assert 0.0 < mid < 2e-4, "no decay phase between the peak and the end"

    def test_linear_decays_to_zero_too_but_differently(self):
        """Control: `linear` must not be silently the same curve as `cosine`.

        Compared at step 20 and 40, deliberately not at 30: the two curves
        cross exactly at the midpoint of the decay phase (both 1.0e-4 there),
        so a control sampled at 30 passes for an implementation that ignores
        `scheduler` entirely. This test asserted step 30 first and failed for
        that reason.
        """
        cos = plan_optimizer(**{**BASE, "scheduler": "cosine"})
        lin = plan_optimizer(**{**BASE, "scheduler": "linear"})
        assert lr_at(lin, 50) == pytest.approx(0.0, abs=1e-9)
        assert lr_at(cos, 20) > lr_at(lin, 20), "cosine decays slower early"
        assert lr_at(cos, 40) < lr_at(lin, 40), "and faster late"

    def test_the_decay_phase_is_measured_from_the_warmup_boundary(self):
        """`join_schedules` hands the second schedule a step relative to the
        boundary, not an absolute one. If it were absolute the cosine would be
        40/50 of the way down at its own peak."""
        plan = plan_optimizer(**BASE)
        assert lr_at(plan, 10) == pytest.approx(2e-4, rel=1e-6)

    def test_constant_holds_the_peak_after_warmup(self):
        plan = plan_optimizer(**{**BASE, "scheduler": "constant_with_warmup"})
        assert lr_at(plan, 0) == pytest.approx(0.0, abs=1e-9)
        assert lr_at(plan, 10) == pytest.approx(2e-4, rel=1e-6)
        assert lr_at(plan, 49) == pytest.approx(2e-4, rel=1e-6)

    def test_constant_with_no_warmup_is_a_plain_float(self):
        """The old behaviour, preserved exactly for the case that asked for it.

        Returning a float rather than a trivial callable keeps the change
        provably confined to configurations that requested something else.
        """
        plan = plan_optimizer(
            **{**BASE, "scheduler": "constant", "warmup_ratio": 0.0}
        )
        assert build_lr_schedule(plan) == pytest.approx(2e-4)


class TestTheStepUnitIsOptimizerUpdates:
    """The arithmetic that a plausible-looking curve would hide."""

    def test_mlx_advances_the_schedule_once_per_optimizer_update(self):
        """Pinned against MLX itself, since the whole design rests on it."""
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        optim = pytest.importorskip("mlx.optimizers")

        seen = []

        def sched(step):
            seen.append(int(step))
            return mx.array(1e-3)

        model = nn.Linear(2, 2)
        opt = optim.AdamW(learning_rate=sched)
        grads = {k: mx.zeros_like(v) for k, v in model.parameters().items()}
        for _ in range(4):
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)

        assert int(opt.step) == 4, "opt.step must count update() calls"
        assert max(seen) == 3, (
            "MLX no longer drives the schedule from opt.step; the update-unit "
            "assumption in trainer/mlx_optim.py needs rechecking"
        )

    def test_warmup_is_computed_from_updates_not_iterations(self):
        """48 iterations at accum=4 is 12 updates, so 25% warmup is 3, not 12."""
        plan = plan_optimizer(
            **{**BASE, "warmup_ratio": 0.25, "total_updates": 48 // 4}
        )
        assert plan.total_updates == 12
        assert plan.warmup_updates == 3, (
            "warmup was computed against iterations; at accum=4 that stretches "
            "the ramp by 4x and the decay never reaches its floor"
        )

    def test_the_iteration_based_answer_is_a_different_number(self):
        """Control: the two units must actually disagree here, or the test
        above would pass for a wrapper that used either."""
        by_updates = plan_optimizer(**{**BASE, "warmup_ratio": 0.25, "total_updates": 12})
        by_iters = plan_optimizer(**{**BASE, "warmup_ratio": 0.25, "total_updates": 48})
        assert by_updates.warmup_updates != by_iters.warmup_updates


class TestWeightDecayIsActuallyPassed:
    def test_the_configured_decay_reaches_the_optimizer(self):
        pytest.importorskip("mlx.optimizers")
        from soup_cli.trainer.mlx_optim import build_optimizer

        opt = build_optimizer(plan_optimizer(**BASE))
        assert float(opt.weight_decay) == pytest.approx(0.3), (
            "the MLX default of 0.01 was used instead of the configured value"
        )

    def test_the_weight_decay_table_matches_the_real_signatures(self):
        """Drift guard: the table is hardcoded so refusals work without MLX."""
        optim = pytest.importorskip("mlx.optimizers")
        from soup_cli.trainer.mlx_optim import _NO_WEIGHT_DECAY, _OPTIMIZER_MAP

        wrong = []
        for name in sorted(set(_OPTIMIZER_MAP.values())):
            cls = getattr(optim, name, None)
            assert cls is not None, f"MLX no longer ships {name}"
            takes = "weight_decay" in inspect.signature(cls.__init__).parameters
            if takes == (name in _NO_WEIGHT_DECAY):
                wrong.append(name)
        assert not wrong, f"_NO_WEIGHT_DECAY disagrees with MLX for: {wrong}"

    def test_decay_on_an_optimizer_that_cannot_take_it_is_refused(self):
        with pytest.raises(MlxOptimizerError, match="takes no weight_decay"):
            plan_optimizer(**{**BASE, "optimizer": "adam", "weight_decay": 0.3})

    def test_but_zero_decay_on_such_an_optimizer_is_fine(self):
        """Control: the refusal must be about the value, not the optimizer."""
        plan = plan_optimizer(**{**BASE, "optimizer": "adam", "weight_decay": 0.0})
        assert plan.optimizer_name == "Adam"


class TestUnsupportedNamesAreRefusedNotSilentlyAdamW:
    def test_an_optimizer_with_no_mlx_equivalent_is_refused_by_name(self):
        """`adam_mini` is a *valid* Soup optimizer, which is the point.

        A name the schema already rejects would prove nothing about this layer.
        """
        with pytest.raises(MlxOptimizerError, match="adam_mini"):
            resolve_optimizer_name("adam_mini")

    def test_the_refusal_names_what_is_available(self):
        with pytest.raises(MlxOptimizerError, match="adamw"):
            resolve_optimizer_name("adalomo")

    def test_most_of_soups_allowlist_has_no_mlx_equivalent(self):
        """Scale of the silent substitution this replaces.

        Every one of these validated cleanly and then ran as AdamW.
        """
        from soup_cli.trainer.mlx_optim import _OPTIMIZER_MAP
        from soup_cli.utils.optimizer_zoo import SUPPORTED_OPTIMIZERS

        supported = set(SUPPORTED_OPTIMIZERS)
        mappable = supported & set(_OPTIMIZER_MAP)
        assert len(mappable) < len(supported) / 2, (
            "if most names now map, the refusal path matters less and this "
            "test should be revisited rather than deleted"
        )
        for name in sorted(supported - set(_OPTIMIZER_MAP))[:5]:
            with pytest.raises(MlxOptimizerError):
                resolve_optimizer_name(name)

    @pytest.mark.parametrize(
        "name, expected",
        [("adamw_torch", "AdamW"), ("ADAMW_TORCH", "AdamW"), ("  sgd  ", "SGD"),
         ("muon", "Muon"), ("adafactor", "Adafactor")],
    )
    def test_supported_names_map_case_and_space_insensitively(self, name, expected):
        assert resolve_optimizer_name(name) == expected

    def test_an_unsupported_scheduler_is_refused(self):
        with pytest.raises(MlxOptimizerError, match="polynomial"):
            plan_optimizer(**{**BASE, "scheduler": "polynomial"})

    def test_the_scheduler_refusal_says_what_the_old_behaviour_was(self):
        """So a reader upgrading understands why their run now stops."""
        with pytest.raises(MlxOptimizerError, match="constant learning rate"):
            plan_optimizer(**{**BASE, "scheduler": "inverse_sqrt"})


class TestDegenerateWarmupIsWarnedAboutNotSwallowed:
    def test_a_warmup_that_rounds_to_zero_warns(self):
        plan = plan_optimizer(**{**BASE, "warmup_ratio": 0.03, "total_updates": 12})
        assert plan.warmup_updates == 0
        assert any("rounds to 0 warmup" in w for w in plan.warnings)

    def test_a_warmup_covering_the_whole_run_warns_and_leaves_a_decay_step(self):
        """Reachable only by calling this function directly.

        `schema.warmup_ratio` is `le=0.5`, so no valid config can produce
        `warmup >= total`. The guard is kept because `plan_optimizer` takes
        plain floats rather than a validated model, and a caller that grows a
        second entry point should not silently get a run with no decay phase.
        Asserted here with ratio 1.0 so the branch is covered honestly rather
        than being dead code nobody notices.
        """
        plan = plan_optimizer(**{**BASE, "warmup_ratio": 1.0, "total_updates": 4})
        assert any("never leaves warmup" in w for w in plan.warnings)
        assert plan.warmup_updates < plan.total_updates

    def test_the_schema_cap_means_a_real_config_cannot_reach_that_branch(self):
        """Pins the reason above, so the claim is checked rather than asserted."""
        from soup_cli.config.schema import TrainingConfig

        field = TrainingConfig.model_fields["warmup_ratio"]
        caps = [m for m in field.metadata if getattr(m, "le", None) is not None]
        assert caps and caps[0].le <= 0.5

    def test_a_normal_configuration_warns_about_nothing(self):
        """Control: the warnings must discriminate, not always fire."""
        assert plan_optimizer(**BASE).warnings == []

    def test_zero_warmup_ratio_is_not_warned_about(self):
        """Asking for no warmup and getting none is not a surprise."""
        plan = plan_optimizer(**{**BASE, "warmup_ratio": 0.0, "total_updates": 12})
        assert plan.warnings == []


class TestAdapterMetadataRecordsWhatRan:
    def test_the_effective_configuration_is_recorded(self):
        meta = plan_optimizer(**BASE).as_metadata()
        assert meta["optimizer"] == "AdamW"
        assert meta["scheduler"] == "cosine"
        assert meta["warmup_updates"] == 10
        assert meta["total_updates"] == 50
        assert meta["weight_decay"] == pytest.approx(0.3)
        assert meta["peak_lr"] == pytest.approx(2e-4)

    def test_metadata_reports_the_resolved_mlx_name_not_the_soup_name(self):
        """`adamw_torch` is not an MLX optimizer; the adapter must say AdamW."""
        assert plan_optimizer(**BASE).as_metadata()["optimizer"] == "AdamW"

    def test_the_plan_is_frozen_so_metadata_cannot_drift_from_what_ran(self):
        plan = plan_optimizer(**BASE)
        with pytest.raises(Exception):
            plan.warmup_updates = 999
        assert isinstance(plan, OptimizerPlan)


# --------------------------------------------------------------------------
# Wiring. The unit tests above pin the arithmetic; this pins that the WRAPPER
# hands it the right number. Mutating `iters // grad_accumulation_steps` to
# `iters` in mlx_sft.py survived everything above -- the whole "step unit"
# design is that one expression, and nothing reached it. Same fake-MLX harness
# as test_issue684_mlx_grad_accumulation.py, extended with the schedule
# builders that module's fake does not need.
# --------------------------------------------------------------------------

import json  # noqa: E402

# The shared fake-MLX harness, imported rather than re-implemented -- the same
# way tests/test_issue23_mlx_display_bridge.py consumes it. A third private
# copy would drift from the real module the moment either of the others is
# updated, and this suite is precisely about MLX surfaces going unnoticed.
from tests.test_issue634_mlx_resume import (  # noqa: E402
    _FakeMlxModel,
    _install_fake_mlx,
)


def _run_wrapper(tmp_path, rows, **training):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(
        base="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="chatml"),
        training=TrainingConfig(**training),
        output=str(tmp_path),
    )
    w = MLXSFTTrainerWrapper(cfg)
    w.model = _FakeMlxModel()
    w.tokenizer = object()
    w._dataset = {"train": [{"text": "hi"}] * rows, "val": []}
    w.train()
    return json.loads((tmp_path / "adapter_config.json").read_text())


class TestTheWrapperPassesUpdatesNotIterations:
    def test_total_updates_is_iters_divided_by_accumulation(
        self, tmp_path, monkeypatch
    ):
        """48 rows at accum=4 is 48 iterations but 12 optimizer updates."""
        _install_fake_mlx(monkeypatch)
        meta = _run_wrapper(
            tmp_path, rows=48, epochs=1, batch_size=1,
            gradient_accumulation_steps=4, scheduler="cosine", warmup_ratio=0.25,
        )
        assert meta["total_updates"] == 12, (
            "the wrapper passed iterations where the schedule counts optimizer "
            "updates; the warmup would be stretched by gradient_accumulation_"
            "steps and the decay would never reach its floor"
        )
        assert meta["warmup_updates"] == 3, "0.25 of 12 updates, not of 48 iters"

    def test_without_accumulation_the_two_units_coincide(self, tmp_path, monkeypatch):
        """Control: the division must be real, not a constant that happens to
        match. At accum=1 iterations and updates are the same number, so this
        passing while the test above fails localises the bug to the divisor."""
        _install_fake_mlx(monkeypatch)
        meta = _run_wrapper(
            tmp_path, rows=48, epochs=1, batch_size=1,
            gradient_accumulation_steps=1, scheduler="cosine", warmup_ratio=0.25,
        )
        assert meta["total_updates"] == 48
        assert meta["warmup_updates"] == 12

    def test_the_configured_weight_decay_reaches_the_metadata(
        self, tmp_path, monkeypatch
    ):
        _install_fake_mlx(monkeypatch)
        meta = _run_wrapper(
            tmp_path, rows=8, epochs=1, batch_size=1, weight_decay=0.3,
            gradient_accumulation_steps=1,
        )
        assert meta["weight_decay"] == pytest.approx(0.3)

    def test_an_unsupported_optimizer_stops_the_run_instead_of_becoming_adamw(
        self, tmp_path, monkeypatch
    ):
        _install_fake_mlx(monkeypatch)
        with pytest.raises(MlxOptimizerError, match="no MLX equivalent"):
            _run_wrapper(
                tmp_path, rows=8, epochs=1, batch_size=1, optimizer="adam_mini",
                gradient_accumulation_steps=1,
            )


# --------------------------------------------------------------------------
# Construction wiring. @MakazhanAlpamys's review of #734 found three mutations
# that survive everything above, the last of which is #686 verbatim:
#
#   kwargs["weight_decay"] = plan.weight_decay  ->  pass          SURVIVED
#   getattr(optim, plan.optimizer_name)         ->  always AdamW  SURVIVED
#   build_optimizer(plan)  ->  optim.AdamW(learning_rate=lr)      SURVIVED
#
# The cause is structural: everything above asserts the PLAN, and
# `as_metadata()` is derived from that same plan, so no metadata assertion can
# ever observe a broken `build_optimizer`. The one test that did reach the
# constructor was `importorskip`-gated and therefore ran on no CI job.
#
# These assert what the constructor is actually called with, through the same
# fake-MLX harness, so they run on Linux and Windows too.
# --------------------------------------------------------------------------


def _record_schedule_calls(monkeypatch):
    """Record which SCHEDULE BUILDER each config reaches, with its arguments.

    @MakazhanAlpamys'"'"'s second review of #734: `callable(learning_rate)` proves
    a schedule exists, not *which* one. Swapping the bodies of the `cosine` and
    `linear` branches -- so every user asking for cosine gets a linear decay and
    every user asking for linear gets a cosine -- survived all 116 tests, while
    `as_metadata()` kept writing the requested name into `adapter_config.json`.
    Dropping the warmup ramp survived too.

    The only assertion that distinguished the two curves sampled them through
    `lr_at()`, behind `importorskip("mlx.core")` -- so on every runner this
    project has, the schedule'"'"'s identity was unverified.

    `_install_fake_mlx` already defines these three builders; this records them
    rather than redefining them, for the reason the module docstring gives.
    """
    optim = sys.modules["mlx.optimizers"]
    calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def _make(name, ret):
        def _builder(*args):
            calls.append((name, args))
            return ret(*args)

        return _builder

    monkeypatch.setattr(
        optim, "cosine_decay",
        _make("cosine_decay", lambda init, steps: (lambda step: init)),
    )
    monkeypatch.setattr(
        optim, "linear_schedule",
        _make("linear_schedule", lambda init, end, steps: (lambda step: end)),
    )
    monkeypatch.setattr(
        optim, "join_schedules",
        _make("join_schedules", lambda scheds, bounds: (lambda step: scheds[-1](step))),
    )
    return calls


def _record_optimizer_calls(monkeypatch):
    """Make every MLX optimizer class record `(name, kwargs)` when constructed.

    Layered on top of `_install_fake_mlx`, which defines only `AdamW` and
    discards its kwargs. Every class `_OPTIMIZER_MAP` can resolve to is
    installed, so "the wrong class was constructed" is observable as the wrong
    name rather than as an AttributeError that any missing class would produce.
    """
    optim = sys.modules["mlx.optimizers"]
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def _make(name):
        def _ctor(**kwargs):
            calls.append((name, kwargs))
            return object()

        return _ctor

    for cls_name in sorted(set(_OPTIMIZER_MAP.values())):
        monkeypatch.setattr(optim, cls_name, _make(cls_name), raising=False)
    return calls


class TestThePlannedOptimizerIsTheOneConstructed:
    def test_the_configured_decay_reaches_the_constructor(self, tmp_path, monkeypatch):
        """Not the metadata -- the constructor. `as_metadata()` reads the same
        plan the metadata tests assert, so it reports 0.3 whether or not the
        kwarg is ever passed."""
        _install_fake_mlx(monkeypatch)
        calls = _record_optimizer_calls(monkeypatch)
        _run_wrapper(
            tmp_path, rows=8, epochs=1, batch_size=1, weight_decay=0.3,
            gradient_accumulation_steps=1,
        )
        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == "AdamW"
        assert kwargs["weight_decay"] == pytest.approx(0.3), (
            "the configured weight decay never reached the optimizer; the run "
            "trained at MLX's default decay while the metadata reported 0.3"
        )

    def test_an_optimizer_that_takes_no_decay_is_constructed_without_it(
        self, tmp_path, monkeypatch
    ):
        """Control for the test above: `weight_decay` must be passed because
        the plan says to, not unconditionally. Adam's MLX constructor has no
        such parameter, so passing it would TypeError on real MLX."""
        _install_fake_mlx(monkeypatch)
        calls = _record_optimizer_calls(monkeypatch)
        _run_wrapper(
            tmp_path, rows=8, epochs=1, batch_size=1, optimizer="adagrad",
            weight_decay=0.0, gradient_accumulation_steps=1,
        )
        assert calls[0][0] == "Adagrad"
        assert "weight_decay" not in calls[0][1]

    @pytest.mark.parametrize(
        "soup_name,mlx_name",
        # Only names `TrainingConfig` will actually accept: `optimizer` is
        # validated against `utils.optimizer_zoo`, and five keys in
        # `_OPTIMIZER_MAP` (adam, adamw, lion, adamax, adadelta) are not in
        # that allowlist, so no valid config can reach them.
        [
            ("sgd", "SGD"),
            ("adagrad", "Adagrad"),
            ("rmsprop", "RMSprop"),
            ("adamw_torch", "AdamW"),
        ],
    )
    def test_the_configured_optimizer_class_is_the_one_constructed(
        self, tmp_path, monkeypatch, soup_name, mlx_name
    ):
        """The defect #686 reports is that every name silently became AdamW.
        Three of these four cases are indistinguishable from that behaviour in
        the metadata, which is resolved from the plan."""
        _install_fake_mlx(monkeypatch)
        calls = _record_optimizer_calls(monkeypatch)
        _run_wrapper(
            tmp_path, rows=8, epochs=1, batch_size=1, optimizer=soup_name,
            weight_decay=0.0, gradient_accumulation_steps=1,
        )
        assert calls[0][0] == mlx_name, (
            f"training.optimizer={soup_name!r} was planned as {mlx_name} and "
            f"constructed as {calls[0][0]} -- a silent substitution is exactly "
            "what this issue is about"
        )

    def test_the_learning_rate_passed_is_the_schedule_not_a_bare_scalar(
        self, tmp_path, monkeypatch
    ):
        """#686 verbatim: `optim.AdamW(learning_rate=lr)`.

        That mutation keeps the class right and the metadata right, so it is
        caught only here. A configuration with warmup must hand the optimizer
        a callable; the old code handed it a float.
        """
        _install_fake_mlx(monkeypatch)
        calls = _record_optimizer_calls(monkeypatch)
        _run_wrapper(
            tmp_path, rows=48, epochs=1, batch_size=1, scheduler="cosine",
            warmup_ratio=0.25, weight_decay=0.1, gradient_accumulation_steps=1,
        )
        lr = calls[0][1]["learning_rate"]
        assert callable(lr), (
            "the optimizer was given a constant learning rate on a config that "
            "asked for cosine decay with warmup -- the reverted #686 behaviour"
        )

    def test_but_a_constant_schedule_still_passes_a_plain_float(
        self, tmp_path, monkeypatch
    ):
        """Control: `callable` above must distinguish configurations, not
        merely hold for everything. The no-warmup constant path is the one
        that must construct exactly what the old code did."""
        _install_fake_mlx(monkeypatch)
        calls = _record_optimizer_calls(monkeypatch)
        _run_wrapper(
            tmp_path, rows=8, epochs=1, batch_size=1, scheduler="constant",
            warmup_ratio=0.0, lr=3e-4, gradient_accumulation_steps=1,
        )
        assert calls[0][1]["learning_rate"] == pytest.approx(3e-4)


class TestTheScheduleBuiltIsTheScheduleRequested:
    """`callable(lr)` proves a schedule exists; it does not say which one.

    Three mutations survived the whole suite before this class existed, and
    all three keep `as_metadata()` writing the requested name into
    `adapter_config.json` while the optimizer receives something else:

      * swap the `cosine` and `linear` branch bodies -- every user asking for
        cosine gets a linear decay, and vice versa;
      * `if warmup <= 0: return body` -> `if True:` -- the warmup ramp is
        never built, while `warmup_updates` still lands in the metadata.

    The only test that distinguished the two curves sampled them through
    `lr_at()`, behind `importorskip("mlx.core")`, so on every runner this
    project has the schedule's identity was unverified. These assert the
    recorded builder call instead, which needs no MLX.
    """

    def _builders(self, tmp_path, monkeypatch, **training):
        _install_fake_mlx(monkeypatch)
        calls = _record_schedule_calls(monkeypatch)
        _run_wrapper(
            tmp_path, rows=48, epochs=1, batch_size=1,
            gradient_accumulation_steps=1, **training
        )
        return calls, [name for name, _ in calls]

    def test_cosine_builds_a_cosine_decay_and_not_a_linear_one(
        self, tmp_path, monkeypatch
    ):
        _, names = self._builders(
            tmp_path, monkeypatch, scheduler="cosine", warmup_ratio=0.0
        )
        assert names == ["cosine_decay"], (
            f"scheduler='cosine' built {names}; the plan and the adapter "
            "metadata would still say 'cosine' either way"
        )

    def test_linear_builds_a_linear_schedule_and_not_a_cosine_one(
        self, tmp_path, monkeypatch
    ):
        """The other half. Passing only one of these two is what a swapped
        pair of branch bodies looks like."""
        _, names = self._builders(
            tmp_path, monkeypatch, scheduler="linear", warmup_ratio=0.0
        )
        assert names == ["linear_schedule"], (
            f"scheduler='linear' built {names}"
        )

    def test_a_constant_schedule_with_no_warmup_builds_nothing_at_all(
        self, tmp_path, monkeypatch
    ):
        """Reject-everything control for the two assertions above.

        Narrowly named on purpose. `warmup_ratio=0.0` returns at the early
        `return peak` *before* the MLX import, so `names == []` holds for every
        possible implementation of the branch below it. This controls the
        cosine/linear assertions; it says nothing about how `constant` is
        built, and the test below is what covers that.
        """
        _, names = self._builders(
            tmp_path, monkeypatch, scheduler="constant", warmup_ratio=0.0
        )
        assert names == []

    @pytest.mark.parametrize("sched", ["constant", "constant_with_warmup"])
    def test_constant_with_warmup_builds_a_ramp_but_no_decay_curve(
        self, tmp_path, monkeypatch, sched
    ):
        """The fourth scheduler, which the other three tests here left open.

        @MakazhanAlpamys found that `if plan.scheduler in ("constant",
        "constant_with_warmup")` at the post-import branch could be changed to
        `if False:` and the whole suite stayed green -- both constant modes
        fell through to the `linear` body, so a user asking for `constant`
        with warmup got a **linear decay to 0.0** while `as_metadata()` wrote
        `scheduler: constant` into `adapter_config.json`.

        The assertion that kills it, `test_constant_holds_the_peak_after_
        warmup`, is behind `importorskip("mlx.core")` and runs on no CI job --
        the same reason this whole class exists. Both spellings are checked
        because both reach that one branch.
        """
        _, names = self._builders(
            tmp_path, monkeypatch, scheduler=sched, warmup_ratio=0.25
        )
        assert "cosine_decay" not in names
        assert names.count("linear_schedule") == 1, (
            f"{sched} built {names}; exactly one linear_schedule is the warmup "
            "ramp -- a second one is a decay curve nobody asked for"
        )

    def test_warmup_builds_a_ramp_joined_at_the_warmup_boundary(
        self, tmp_path, monkeypatch
    ):
        """`warmup_updates` reaching the metadata does not mean a ramp was
        built. 0.25 of 48 updates is 12."""
        calls, names = self._builders(
            tmp_path, monkeypatch, scheduler="cosine", warmup_ratio=0.25
        )
        assert "join_schedules" in names, (
            "no ramp was joined; the run starts at the peak learning rate "
            "while adapter_config.json reports a warmup"
        )
        joined = dict((n, a) for n, a in calls)["join_schedules"]
        assert joined[1] == [12], f"joined at boundary {joined[1]}, expected [12]"

        ramp = [a for n, a in calls if n == "linear_schedule"]
        assert ramp and ramp[0][0] == 0.0 and ramp[0][2] == 12, (
            f"the ramp is {ramp}; it must rise from 0.0 over 12 updates"
        )

    def test_without_warmup_nothing_is_joined(self, tmp_path, monkeypatch):
        """Control for the test above."""
        _, names = self._builders(
            tmp_path, monkeypatch, scheduler="cosine", warmup_ratio=0.0
        )
        assert "join_schedules" not in names

    def test_the_decay_is_measured_from_the_warmup_boundary(
        self, tmp_path, monkeypatch
    ):
        """`decay_steps = total - warmup`, asserted at the builder rather than
        by sampling: 48 updates with 12 of warmup decays over 36."""
        calls, _ = self._builders(
            tmp_path, monkeypatch, scheduler="cosine", warmup_ratio=0.25
        )
        cosine = [a for n, a in calls if n == "cosine_decay"]
        assert cosine and cosine[0][1] == 36, (
            f"cosine_decay got decay_steps={cosine[0][1] if cosine else None}, "
            "expected 36 (48 total - 12 warmup)"
        )
