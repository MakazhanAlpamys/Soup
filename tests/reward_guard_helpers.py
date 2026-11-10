"""Runtime checks that an LM consumer refuses reward bundles before doing work."""

from __future__ import annotations

import inspect
import sys

import pytest

from soup_cli.utils.mlx_reward import SCORING_ONLY


def assert_reward_guard(function, kwargs, first_work_line, *, refused=True):
    """Stop at the first pre-existing statement, so later guards cannot mask removal.

    This calls the real command/helper. The trace is only a side-effect sentinel;
    recognition and the user-visible refusal are not mocked.
    """
    source, first = inspect.getsourcelines(function)
    boundary = first + next(i for i, line in enumerate(source)
                            if line.strip() == first_work_line)
    previous = sys.gettrace()

    def before_work(frame, event, _arg):
        if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno == boundary:
            raise AssertionError('LM consumer started work before refusing a scoring bundle')
        return before_work

    try:
        sys.settrace(before_work)
        with pytest.raises((ValueError, SystemExit, Exception)) as caught:
            function(**kwargs)
    finally:
        sys.settrace(previous)
    error = caught.value
    if not refused:
        assert isinstance(error, AssertionError), str(error)
        assert str(error) == 'LM consumer started work before refusing a scoring bundle'
        return
    assert not isinstance(error, AssertionError), str(error)
    # CLI wrappers turn the ValueError into typer.Exit, keeping its cause.
    assert str(error.__cause__ or error) == SCORING_ONLY
