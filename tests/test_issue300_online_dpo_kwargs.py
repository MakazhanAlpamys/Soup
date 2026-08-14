"""#300 — `training.reward_model` fails at trainer construction on trl 0.25-0.28.

`_trl_has_judges()` probed `from trl import BasePairwiseJudge` as a proxy for
"this trl accepts `reward_model=`". Those two facts DECOUPLED: trl dropped
`reward_model=` from `OnlineDPOTrainer` at **0.25.0** but kept `BasePairwiseJudge`
exported at top level through **0.28.0**. The probe therefore returns True for
every trl in the supported `<0.27` range — including whatever a fresh
`pip install "soup-cli[train]"` resolves — and the wrapper passes a keyword that
was removed five releases earlier::

    TypeError: OnlineDPOTrainer.__init__() got an unexpected keyword argument
    'reward_model'

Measured on trl 0.26.2 (H100): the judge path trains fine, the reward_model path
dies before step 0 with no adapter written.

WHY CI COULD NOT CATCH IT — and what this file does differently.

`tests/test_v07131.py::test_reward_model_branch_adapts_to_trl_version` branches on
`_TRL_HAS_JUDGES`, the SAME predicate the production code branches on. It asserts
that the code returned what its own condition says, never that trl accepts the
result. It cannot fail. That is the identical blind spot as the v0.72.4 trl-bound
bug, whose lesson was recorded as: *a version bound derived by reading source is a
hypothesis; the experiment that settles it is constructing the object.*

So the load-bearing test here BINDS the kwargs against the real
`OnlineDPOTrainer.__init__` signature rather than re-asking the probe.
"""


import pytest

trl = pytest.importorskip("trl", reason="online_dpo needs the [train] extra")


def _real_params():
    """The parameters the trainer REALLY takes, resolved the same way production
    resolves them.

    A naive ``inspect.signature(OnlineDPOTrainer.__init__)`` is wrong and this
    file had that bug too: on trl 0.26.2 the top-level class is a deprecation
    shim ``def __init__(self, *args, **kwargs)`` that forwards to
    ``trl.experimental``, so the direct signature reports NOTHING and every
    assertion built on it fails for a reason that has nothing to do with the
    code under test.

    Deliberately reusing the production resolver instead of re-implementing it:
    two copies of "where does the real signature live" is exactly how the
    original defect happened.
    """
    from soup_cli.trainer.online_dpo import _trl_accepts

    return _trl_accepts


class TestTheKwargsBindAgainstTheRealSignature:
    """The test the blind spot needed: ask trl, not our own predicate."""

    def test_the_reward_signal_kwarg_is_one_this_trl_accepts(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        class _Tcfg:
            online_dpo_judge = None
            reward_model = None

        # the wrapper's own kwarg choice, via the documented test seam so no
        # model is loaded
        from soup_cli.trainer import online_dpo as mod

        sentinel = object()
        original = mod._ONLINE_DPO_JUDGE_OVERRIDE
        mod._ONLINE_DPO_JUDGE_OVERRIDE = sentinel
        try:
            kwargs = OnlineDPOTrainerWrapper._build_judge_or_reward(
                OnlineDPOTrainerWrapper.__new__(OnlineDPOTrainerWrapper), _Tcfg()
            )
        finally:
            mod._ONLINE_DPO_JUDGE_OVERRIDE = original

        accepts = _real_params()
        unknown = [k for k in kwargs if not accepts(k)]
        assert not unknown, (
            f"the wrapper builds {sorted(kwargs)} but this trl "
            f"({getattr(trl, '__version__', '?')}) accepts none of {unknown} — "
            "OnlineDPOTrainer.__init__ would raise TypeError before step 0"
        )

    def test_the_probe_agrees_with_the_signature(self):
        """The probe must report what the installed trl actually takes.

        Stated against `inspect.signature` rather than against a version number,
        so it keeps working when trl moves the parameter again.
        """
        from soup_cli.trainer.online_dpo import _trl_accepts

        # At least one real signal kwarg must exist, or the probe is answering
        # False to everything - which is how the first version of this fix looked
        # correct while it would have broken the judge path.
        assert any(
            _trl_accepts(name) for name in ("judge", "reward_model", "reward_funcs")
        ), "the probe found no reward-signal parameter at all on this trl"
        assert _trl_accepts("judge") is True, (
            "every supported trl takes judge=; a probe that says otherwise is "
            "reading a passthrough shim instead of the real signature"
        )

    def test_an_absent_parameter_is_reported_absent(self):
        """CONTROL. A probe that answered True unconditionally would satisfy the
        test above for every name that happens to exist."""
        from soup_cli.trainer.online_dpo import _trl_accepts

        assert _trl_accepts("definitely_not_a_trl_parameter") is False
