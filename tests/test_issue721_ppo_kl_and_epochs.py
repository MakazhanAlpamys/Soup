"""Issue #721 - PPO must forward the configured KL strength and epoch budget.

Two values were resolved by name against the installed trl and lost when the
name did not match:

* ``training.ppo_kl_penalty`` was passed only as ``init_kl_coef``. trl 0.29
  renamed that to ``kl_coef``, and an unrecognised keyword here was never an
  error - it was dropped, so PPO ran at trl's own default KL strength with
  nothing said.
* ``training.epochs`` reached only the legacy manual loop, never
  ``num_train_epochs`` on the built-in ``PPOConfig`` path. ``ppo_epochs`` is a
  different setting (optimization passes per batch) and was always correct.

trl is not a test dependency, so these ask the question without it: stand in
synthetic config classes with each signature and read what comes back. The
kwarg-selection half lives in ``_trl_compat.kl_penalty_kwargs`` so it can be
called directly, next to ``prompt_length_kwargs``, which exists for the same
kind of rename.
"""

from __future__ import annotations

import ast
import pathlib

from soup_cli.trainer._trl_compat import kl_penalty_kwargs

_PPO_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "soup_cli"
    / "trainer"
    / "ppo.py"
).read_text(encoding="utf-8")


class _ModernConfig:
    """trl >= 0.29: the coefficient is ``kl_coef``."""

    def __init__(self, kl_coef: float = 0.05, num_train_epochs: float = 3.0):
        pass


class _LegacyConfig:
    """Older trl: the coefficient is ``init_kl_coef``."""

    def __init__(self, init_kl_coef: float = 0.2):
        pass


class _NeitherConfig:
    """A hypothetical trl that renamed it again."""

    def __init__(self, cliprange: float = 0.2):
        pass


class TestKlPenaltyKwargs:
    def test_modern_trl_gets_kl_coef(self):
        assert kl_penalty_kwargs(_ModernConfig, 0.7) == {"kl_coef": 0.7}

    def test_legacy_trl_still_gets_init_kl_coef(self):
        assert kl_penalty_kwargs(_LegacyConfig, 0.7) == {"init_kl_coef": 0.7}

    def test_a_config_with_neither_name_gets_nothing(self):
        # Empty rather than a guess, so the next rename fails visibly at the
        # call site instead of being silently dropped the way this one was.
        assert kl_penalty_kwargs(_NeitherConfig, 0.7) == {}

    def test_the_new_name_wins_when_a_config_carries_both(self):
        class _Both:
            def __init__(self, kl_coef: float = 0.05, init_kl_coef: float = 0.2):
                pass

        assert kl_penalty_kwargs(_Both, 0.7) == {"kl_coef": 0.7}


class TestPpoForwardsTheEpochBudget:
    def test_setup_passes_num_train_epochs(self):
        """``training.epochs`` must reach the built-in PPOConfig path.

        Read statically: reaching ``setup()`` needs a model and a dataset, and
        trl is not installed in the test environment. What is checked is that
        the assignment exists at all - it did not before, which is the bug.
        """
        tree = ast.parse(_PPO_SOURCE)
        assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "num_train_epochs"
        ]
        assert assigns, "ppo.py never puts num_train_epochs into ppo_kwargs"
        assert any(
            ast.unparse(node.value) == "tcfg.epochs" for node in assigns
        ), "num_train_epochs is set from something other than tcfg.epochs"

    def test_ppo_epochs_is_still_a_separate_setting(self):
        """The fix must not conflate the two. ppo_epochs is passes per batch."""
        assert "num_ppo_epochs" in _PPO_SOURCE
        assert "tcfg.ppo_epochs" in _PPO_SOURCE

    def test_init_kl_coef_is_no_longer_resolved_by_hand(self):
        """The old single-name lookup is gone, not merely supplemented."""
        assert 'ppo_kwargs["init_kl_coef"]' not in _PPO_SOURCE
        assert "kl_penalty_kwargs(ppo_config_cls" in _PPO_SOURCE
