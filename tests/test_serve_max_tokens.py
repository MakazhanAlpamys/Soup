"""``soup serve --max-tokens`` refuses a count below 1 before any backend loads (#2730).

Every backend uses ``--max-tokens`` for a request that omits ``max_tokens``. With 0 or
a negative value the server used to start and then answer each such request with a
500 (``max_new_tokens must be greater than 0``). ``serve()`` now refuses the value
next to the ``--max-model-len`` check, before any backend branch, availability check
or model load, with exit 2 and one message on every backend.

The upper end (above 16384) is not refused here: such values work on v0.77.2 for a
request that omits ``max_tokens``, so refusing them waits for v0.78.0.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

runner = CliRunner()

BACKENDS = ["transformers", "vllm", "sglang", "mii"]

# Each is reached only after the check, so none may run on a refused value.
_GUARDED = [
    "soup_cli.commands.serve._load_model",
    "soup_cli.utils.vllm.is_vllm_available",
    "soup_cli.utils.sglang.check_sglang_available",
    "soup_cli.utils.mii.is_mii_available",
]


class _ReachedTheLoadError(Exception):
    """Raised by the patched loader so a passing value stops there."""


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def model_dir(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _serve(model_dir, *args):
    return runner.invoke(app, ["serve", "--model", str(model_dir), *args])


class TestRefused:
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("value", ["0", "-5"])
    def test_a_count_below_1_exits_2_before_anything_loads(
        self, model_dir, backend, value, monkeypatch
    ):
        monkeypatch.setenv("COLUMNS", "400")
        with ExitStack() as stack:
            guards = [
                stack.enter_context(
                    patch(target, side_effect=AssertionError(f"{target} ran before the check"))
                )
                for target in _GUARDED
            ]
            result = _serve(model_dir, "--backend", backend, "--max-tokens", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert _plain(result.output) == f"--max-tokens: must be at least 1 (got {value}).", (
            result.output
        )
        for guard in guards:
            guard.assert_not_called()


class TestAccepted:
    @pytest.mark.parametrize("value", ["1", "16384", "16385"])
    def test_a_count_of_1_or_more_reaches_the_load(self, model_dir, value, monkeypatch):
        monkeypatch.setenv("COLUMNS", "400")
        with patch(
            "soup_cli.commands.serve._load_model", side_effect=_ReachedTheLoadError
        ) as loader:
            result = _serve(model_dir, "--max-tokens", value)

        assert isinstance(result.exception, _ReachedTheLoadError), (
            result.output,
            repr(result.exception),
        )
        assert loader.call_count == 1
        assert "--max-tokens:" not in _plain(result.output), result.output
