"""The SGLang backend must obey the v0.36.0 trust_remote_code gate.

``serve.py`` resolves ``--trust-remote-code`` **once for every backend** and its
own comment states the intent: "so no backend silently executes an untrusted
repo's code". vLLM receives that resolved value. SGLang did not: it was called
without the argument at all, and ``create_sglang_runtime`` hardcoded
``trust_remote_code=True`` at both of its ``sgl.Runtime`` call sites.

The result is that on ``soup serve --backend sglang`` the default-deny gate did
not apply. A user who never passed ``--trust-remote-code`` still had a model's
custom repo code executed, which is the exact behaviour v0.36.0 removed from the
vLLM path.

The panel printed above the load said so honestly -- "SGLang loads models with
trust_remote_code enabled" -- but a notice is not a gate, and it told the user
what was happening rather than letting them decide.

Each surface is pinned independently: the runtime's non-adapter call site, its
adapter call site, and the tokenizer load. A fix that threads the flag through
one and forgets another must fail here rather than pass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest


class TestCreateSglangRuntimeHonoursTheFlag:
    """The two ``sgl.Runtime`` call sites, pinned separately."""

    def _capture(self, monkeypatch, **kwargs):
        calls = {}
        fake_sgl = MagicMock()

        def _runtime(**runtime_kwargs):
            calls.update(runtime_kwargs)
            return MagicMock()

        fake_sgl.Runtime = _runtime
        with mock_patch.dict("sys.modules", {"sglang": fake_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            create_sglang_runtime(**kwargs)
        return calls

    def test_default_is_deny_on_the_plain_path(self, monkeypatch) -> None:
        calls = self._capture(
            monkeypatch,
            model_path="/models/plain",
            base_model=None,
            is_adapter=False,
            tensor_parallel_size=1,
            mem_fraction_static=0.9,
        )
        assert calls["trust_remote_code"] is False

    def test_default_is_deny_on_the_adapter_path(self, monkeypatch) -> None:
        """Second call site. A fix that only patches the first must fail here."""
        calls = self._capture(
            monkeypatch,
            model_path="/adapters/lora",
            base_model="/models/base",
            is_adapter=True,
            tensor_parallel_size=1,
            mem_fraction_static=0.9,
        )
        assert calls["trust_remote_code"] is False

    def test_opt_in_is_honoured_on_the_plain_path(self, monkeypatch) -> None:
        calls = self._capture(
            monkeypatch,
            model_path="/models/plain",
            base_model=None,
            is_adapter=False,
            tensor_parallel_size=1,
            mem_fraction_static=0.9,
            trust_remote_code=True,
        )
        assert calls["trust_remote_code"] is True

    def test_opt_in_is_honoured_on_the_adapter_path(self, monkeypatch) -> None:
        calls = self._capture(
            monkeypatch,
            model_path="/adapters/lora",
            base_model="/models/base",
            is_adapter=True,
            tensor_parallel_size=1,
            mem_fraction_static=0.9,
            trust_remote_code=True,
        )
        assert calls["trust_remote_code"] is True


class TestServeSglangThreadsTheResolvedValue:
    """``_serve_sglang`` must pass what it was given to *both* loaders."""

    def _serve(self, monkeypatch, trust):
        import soup_cli.commands.serve as serve_mod
        import soup_cli.utils.sglang as sglang_mod

        seen = {}

        def _runtime(**kwargs):
            seen["runtime"] = kwargs
            return MagicMock(), "runtime-model"

        def _tokenizer(**kwargs):
            seen["tokenizer"] = kwargs
            tok = MagicMock()
            tok.chat_template = "<TPL>"
            return tok

        monkeypatch.setattr(sglang_mod, "create_sglang_runtime", _runtime, raising=False)
        monkeypatch.setattr(
            sglang_mod, "create_sglang_app", lambda **kwargs: kwargs, raising=False
        )
        monkeypatch.setattr(serve_mod, "_load_serve_tokenizer", _tokenizer)
        printed = []
        monkeypatch.setattr(
            serve_mod.console, "print", lambda *a, **k: printed.append(str(a[0]))
        )
        seen["printed"] = printed

        serve_mod._serve_sglang(
            model_path=Path("/models/custom-code-model"),
            base_model=None,
            is_adapter=False,
            max_tokens_default=256,
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=trust,
        )
        return seen

    @pytest.mark.parametrize("trust", [True, False])
    def test_runtime_receives_the_resolved_value(self, monkeypatch, trust) -> None:
        seen = self._serve(monkeypatch, trust)
        assert seen["runtime"]["trust_remote_code"] is trust

    @pytest.mark.parametrize("trust", [True, False])
    def test_tokenizer_receives_the_resolved_value(self, monkeypatch, trust) -> None:
        """Independent surface: #581 pinned this to a literal True on purpose.

        That was right while the runtime was also an unconditional True -- a
        mismatch there meant the tokenizer failed to load on custom-code models
        and the #360 prompt fix silently degraded. Now that the runtime follows
        the gate, the tokenizer has to follow it too, or the same mismatch comes
        back with the operands swapped.
        """
        seen = self._serve(monkeypatch, trust)
        assert seen["tokenizer"]["trust_remote_code"] is trust

    def test_panel_does_not_promise_trust_when_it_was_denied(self, monkeypatch) -> None:
        seen = self._serve(monkeypatch, False)
        panel_text = " ".join(seen["printed"])
        assert "loads models with trust_remote_code enabled" not in panel_text


class TestServeCommandPassesTheGateToSglang:
    """The end-to-end wiring: the gate's result must reach the backend."""

    def test_sglang_branch_forwards_resolved_trust(self) -> None:
        """A literal ``True`` here would re-open the hole with the gate intact."""
        import inspect

        import soup_cli.commands.serve as serve_mod

        source = inspect.getsource(serve_mod.serve)
        sglang_call = source[source.index('elif backend == "sglang":') :]
        sglang_call = sglang_call[: sglang_call.index("else:")]
        assert "trust_remote_code=resolved_trust" in sglang_call
        assert "trust_remote_code=True" not in sglang_call
