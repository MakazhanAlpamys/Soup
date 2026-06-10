"""v0.71.22 "Perf & measure polish" tests.

Closes #263 (MiniLLM on-policy KV-cache), #262 (`serve --mole` KV-cache),
#143 (first-party generator factories for `deploy autopilot --measure`),
#265-partial (live-codec TTS — Orpheus via SNAC; the other 4 families stay
dep-gated with the encoder tracked in #265).

All four are pure code, fully validatable on the RTX 3050 / CPU budget. The
KV-cache paths are exercised with cache-capable fake LMs (prefix-sum caches
so cached logits provably equal full-re-forward logits); models without an
explicit ``past_key_values`` forward param transparently keep the legacy
full-re-forward path (capability probe — old fakes / exotic models stay
correct).
"""

from __future__ import annotations

import math
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

runner = CliRunner()


def _module_head(rel_path: str) -> str:
    """Return a module's source text up to the first ``def``/``class``."""
    src = (_SRC / rel_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    # Split on the first top-level def or class so we only inspect the head.
    for marker in ("\ndef ", "\nclass "):
        src = src.split(marker, 1)[0]
    return src


def _assert_no_top_level_import(rel_path: str, mod: str) -> None:
    """Reject BOTH ``import {mod}...`` and ``from {mod} ...`` at module top.

    Stronger than the ``\\nimport {mod}\\n`` idiom — catches ``import numpy as
    np``, ``import torch, snac``, and ``from numpy import ...`` (L11).
    """
    head = _module_head(rel_path)
    assert f"\nimport {mod}" not in head, (
        f"top-level `import {mod}` in {rel_path}"
    )
    assert f"\nfrom {mod} " not in head, (
        f"top-level `from {mod} ` in {rel_path}"
    )


def _torch_or_skip():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# #263 — MiniLLM on-policy rollout KV-cache
# ---------------------------------------------------------------------------


def _make_cache_fake_lm(vocab: int = 11, hidden: int = 6, seed: int = 0):
    """Prefix-sum LM with an explicit past_key_values cache.

    The last-token logits depend on the FULL prefix (cumulative embedding
    sum), so a correct cache implementation produces bit-identical logits to
    a full re-forward — the property the equality tests rely on. The cache is
    the running prefix-sum ``[B, 1, H]`` (grad-carrying, like real KV caches).
    """
    torch = _torch_or_skip()
    from torch import nn

    torch.manual_seed(seed)

    class _CacheFakeLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)
            self.config = types.SimpleNamespace(vocab_size=vocab)
            self.calls: list[int] = []  # fed token counts per forward

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
        ):
            self.calls.append(int(input_ids.shape[1]))
            h = self.emb(input_ids)
            csum = torch.cumsum(h, dim=1)
            if past_key_values is not None:
                csum = csum + past_key_values[0]
            logits = self.head(csum)
            out = SimpleNamespace(logits=logits)
            if use_cache:
                out.past_key_values = (csum[:, -1:, :],)
            return out

    return _CacheFakeLM()


def _make_legacy_fake_lm(vocab: int = 11, hidden: int = 6, seed: int = 0):
    """The pre-#263 fake — no explicit past_key_values param (``**kw`` only)."""
    torch = _torch_or_skip()
    from torch import nn

    torch.manual_seed(seed)

    class _LegacyFakeLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)
            self.calls: list[int] = []

        def forward(self, input_ids=None, attention_mask=None, **kw):
            self.calls.append(int(input_ids.shape[1]))
            h = self.emb(input_ids)
            logits = self.head(torch.cumsum(h, dim=1))
            return SimpleNamespace(logits=logits)

    return _LegacyFakeLM()


def _make_no_past_fake_lm(vocab: int = 11, hidden: int = 6, seed: int = 0):
    """Declares past_key_values/use_cache but never returns a cache."""
    torch = _torch_or_skip()
    from torch import nn

    torch.manual_seed(seed)

    class _NoPastFakeLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)
            self.calls: list[int] = []

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
        ):
            self.calls.append(int(input_ids.shape[1]))
            h = self.emb(input_ids)
            logits = self.head(torch.cumsum(h, dim=1))
            return SimpleNamespace(logits=logits)  # no past_key_values attr

    return _NoPastFakeLM()


def _greedy_sampler(probs):
    return probs.argmax(dim=-1, keepdim=True)


class TestSupportsKvCache:
    def test_explicit_params_true(self):
        from soup_cli.utils.minillm import _supports_kv_cache

        assert _supports_kv_cache(_make_cache_fake_lm())

    def test_var_keyword_only_false(self):
        """``**kw`` does NOT count — a swallowed kwarg is not cache support."""
        from soup_cli.utils.minillm import _supports_kv_cache

        assert not _supports_kv_cache(_make_legacy_fake_lm())

    def test_no_params_false(self):
        from soup_cli.utils.minillm import _supports_kv_cache

        def plain(input_ids=None, attention_mask=None):
            return None

        assert not _supports_kv_cache(SimpleNamespace(forward=plain))

    def test_non_model_false(self):
        from soup_cli.utils.minillm import _supports_kv_cache

        assert not _supports_kv_cache(object())

    def test_peft_wrapper_probes_base_model(self):
        """A ``*args, **kwargs`` wrapper (PeftModel-style LoRA student — the
        common live distill case) is probed through ``get_base_model()`` so the
        base's explicit cache params are seen and KV-cache activates (#263)."""
        from soup_cli.utils.minillm import _supports_kv_cache

        inner = _make_cache_fake_lm()

        class _Wrapper:
            def forward(self, *args, **kwargs):
                return inner(*args, **kwargs)

            def get_base_model(self):
                return inner

        assert _supports_kv_cache(_Wrapper())

    def test_peft_wrapper_get_base_model_raising_is_safe(self):
        """If ``get_base_model()`` raises, the probe swallows it and falls back
        to inspecting the wrapper itself (never raises)."""
        from soup_cli.utils.minillm import _supports_kv_cache

        class _Bad:
            def forward(self, *args, **kwargs):
                return None

            def get_base_model(self):
                raise RuntimeError("boom")

        # Wrapper forward is *args/**kwargs only — falls back to it → False.
        assert not _supports_kv_cache(_Bad())


class TestOnPolicyKvCache:
    def _rollout(self, student, teacher, *, use_cache, steps=4):
        torch = _torch_or_skip()
        from soup_cli.utils.minillm import (
            MiniLLMConfig,
            minillm_on_policy_rollout,
        )

        cfg = MiniLLMConfig(teacher_mix_ratio=0.5, on_policy=True)
        ids = torch.tensor([[1, 2, 3]])
        loss, n = minillm_on_policy_rollout(
            student,
            teacher,
            ids,
            None,
            config=cfg,
            max_new_tokens=steps,
            sample_fn=_greedy_sampler,
            use_cache=use_cache,
        )
        return loss, n

    def test_cached_loss_matches_uncached(self):
        torch = _torch_or_skip()
        s1, t1 = _make_cache_fake_lm(seed=0), _make_cache_fake_lm(seed=1)
        s2, t2 = _make_cache_fake_lm(seed=0), _make_cache_fake_lm(seed=1)
        loss_cached, n1 = self._rollout(s1, t1, use_cache=True)
        loss_full, n2 = self._rollout(s2, t2, use_cache=False)
        assert n1 == n2 == 4
        assert torch.allclose(loss_cached, loss_full, rtol=1e-5, atol=1e-6)

    def test_cached_path_feeds_single_tokens(self):
        student, teacher = _make_cache_fake_lm(seed=0), _make_cache_fake_lm(seed=1)
        self._rollout(student, teacher, use_cache=True, steps=3)
        # prompt (3 tokens) once, then 1 token per remaining step.
        assert student.calls == [3, 1, 1]
        assert teacher.calls == [3, 1, 1]

    def test_uncached_path_refeeds_full_prefix(self):
        student, teacher = _make_cache_fake_lm(seed=0), _make_cache_fake_lm(seed=1)
        self._rollout(student, teacher, use_cache=False, steps=3)
        assert student.calls == [3, 4, 5]

    def test_legacy_kwargs_model_routes_uncached(self):
        """A model without explicit cache params keeps the legacy path even
        with use_cache=True (capability probe)."""
        student, teacher = _make_legacy_fake_lm(seed=0), _make_legacy_fake_lm(seed=1)
        loss, _ = self._rollout(student, teacher, use_cache=True, steps=3)
        assert student.calls == [3, 4, 5]
        assert math.isfinite(float(loss))

    def test_missing_past_in_output_falls_back(self):
        """A model that declares but ignores use_cache degrades to full
        re-forwards after the first step (no crash, finite loss)."""
        student, teacher = _make_no_past_fake_lm(seed=0), _make_no_past_fake_lm(seed=1)
        loss, n = self._rollout(student, teacher, use_cache=True, steps=3)
        assert n == 3
        assert math.isfinite(float(loss))
        # step 0 fed the prompt; the fallback re-feeds full prefixes after.
        assert student.calls == [3, 4, 5]

    def test_grad_flows_through_cached_rollout(self):
        student, teacher = _make_cache_fake_lm(seed=0), _make_cache_fake_lm(seed=1)
        loss, _ = self._rollout(student, teacher, use_cache=True)
        loss.backward()
        assert student.emb.weight.grad is not None
        assert float(student.emb.weight.grad.abs().sum()) > 0.0
        # Teacher is no_grad throughout.
        assert teacher.emb.weight.grad is None

    def test_use_cache_bool_rejected(self):
        student, teacher = _make_cache_fake_lm(), _make_cache_fake_lm(seed=1)
        with pytest.raises(TypeError, match="use_cache"):
            self._rollout(student, teacher, use_cache="yes")

    def test_mixed_capability_routes_uncached(self):
        """Cache requires BOTH models capable — a legacy teacher disables it."""
        student = _make_cache_fake_lm(seed=0)
        teacher = _make_legacy_fake_lm(seed=1)
        self._rollout(student, teacher, use_cache=True, steps=3)
        assert student.calls == [3, 4, 5]

    def test_on_policy_term_threads_use_cache(self, monkeypatch):
        _torch_or_skip()
        from soup_cli.utils import minillm as m

        captured = {}

        def fake_rollout(*args, **kwargs):
            captured.update(kwargs)
            import torch

            return torch.tensor(0.0), 1

        monkeypatch.setattr(m, "minillm_on_policy_rollout", fake_rollout)
        cb = m.build_minillm_callback(m.MiniLLMConfig(on_policy=True))
        import torch

        cb.on_policy_term(object(), object(), torch.tensor([[1]]), use_cache=False)
        assert captured["use_cache"] is False
        cb.on_policy_term(object(), object(), torch.tensor([[1]]))
        assert captured["use_cache"] is True


# ---------------------------------------------------------------------------
# #262 — serve --mole per-adapter KV-cache
# ---------------------------------------------------------------------------


def _make_cached_mole_model(hidden: int = 4, vocab: int = 6):
    """Adapter-switching fake with an explicit per-call KV cache.

    PREFIX-DEPENDENT logits (mirrors the minillm ``_CacheFakeLM`` prefix-sum
    design): the active row's logit encodes the TOTAL prefix length the cache
    claims to have seen (``past + t``) plus the adapter id. A correct cache
    therefore yields bit-identical logits to a full re-forward, while a broken
    cache (wrong ``state.lens`` slice, corrupted ``past_key_values``, off-by-one
    catch-up bookkeeping) would corrupt the encoded prefix length and the
    equality assertions would FAIL — the property M4 needs. ``calls`` records
    ``(mode, fed_tokens)`` so the delta-feeding / catch-up tests can assert
    exact feed lengths.
    """
    torch = _torch_or_skip()
    from torch import nn

    class _CachedMoleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.vocab = vocab
            self._adapter = "task_0"
            self._disabled = False
            self.calls: list[tuple[str, int]] = []

        @property
        def device(self):
            return torch.device("cpu")

        def disable_adapter(self):
            import contextlib

            @contextlib.contextmanager
            def _cm():
                self._disabled = True
                try:
                    yield
                finally:
                    self._disabled = False

            return _cm()

        def set_adapter(self, name):
            self._adapter = name

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            output_hidden_states=False,
            past_key_values=None,
            use_cache=False,
        ):
            mode = "base" if self._disabled else self._adapter
            self.calls.append((mode, int(input_ids.shape[1])))
            b, t = input_ids.shape
            hs = torch.ones(b, t, self.hidden)
            idx = 0 if self._disabled else int(self._adapter.split("_")[1])
            past = int(past_key_values[0]) if past_key_values is not None else 0
            # Total prefix length this stream has now seen — bakes the cache
            # bookkeeping into the output so a wrong cache produces wrong logits.
            seen = past + t
            logits = torch.zeros(b, t, self.vocab)
            logits[..., idx % self.vocab] = float(idx + 1) + float(seen)
            out = SimpleNamespace(logits=logits)
            if output_hidden_states:
                out.hidden_states = (hs,)
            if use_cache:
                out.past_key_values = (seen,)
            return out

    return _CachedMoleModel()


def _make_legacy_mole_model(hidden: int = 4, vocab: int = 6):
    """The pre-#262 fake — no cache params on forward."""
    torch = _torch_or_skip()
    from torch import nn

    class _LegacyMoleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.vocab = vocab
            self._adapter = "task_0"
            self._disabled = False
            self.calls: list[tuple[str, int]] = []

        @property
        def device(self):
            return torch.device("cpu")

        def disable_adapter(self):
            import contextlib

            @contextlib.contextmanager
            def _cm():
                self._disabled = True
                try:
                    yield
                finally:
                    self._disabled = False

            return _cm()

        def set_adapter(self, name):
            self._adapter = name

        def forward(
            self, input_ids=None, attention_mask=None, output_hidden_states=False
        ):
            mode = "base" if self._disabled else self._adapter
            self.calls.append((mode, int(input_ids.shape[1])))
            b, t = input_ids.shape
            hs = torch.ones(b, t, self.hidden)
            idx = 0 if self._disabled else int(self._adapter.split("_")[1])
            # Re-feeds the full prefix each step, so t == sequence length so far
            # — encode it identically to the cached model's (past + t) so the
            # legacy and cached blends match exactly (M4 equality property).
            logits = torch.zeros(b, t, self.vocab)
            logits[..., idx % self.vocab] = float(idx + 1) + float(t)
            out = SimpleNamespace(logits=logits)
            if output_hidden_states:
                out.hidden_states = (hs,)
            return out

    return _LegacyMoleModel()


def _make_uniform_gate(hidden: int = 4, n: int = 2):
    torch = _torch_or_skip()
    from torch import nn

    class _UniformGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden, n, bias=False)

        def forward(self, h):
            b = h.shape[0]
            return torch.full((b, n), 1.0 / n)

    return _UniformGate()


def _make_scripted_gate(script, hidden: int = 4, n: int = 2):
    """Gate returning a scripted weight row per call (cache-catch-up test)."""
    torch = _torch_or_skip()
    from torch import nn

    class _ScriptedGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden, n, bias=False)
            self._i = 0

        def forward(self, h):
            row = script[min(self._i, len(script) - 1)]
            self._i += 1
            return torch.tensor([row])

    return _ScriptedGate()


class TestModelSupportsCache:
    def test_explicit_params_true(self):
        from soup_cli.utils.mole_routing import _model_supports_cache

        assert _model_supports_cache(_make_cached_mole_model())

    def test_legacy_fake_false(self):
        from soup_cli.utils.mole_routing import _model_supports_cache

        assert not _model_supports_cache(_make_legacy_mole_model())

    def test_peft_wrapper_probes_base_model(self):
        """A ``*args, **kwargs`` wrapper (PeftModel-style) is probed through
        ``get_base_model()`` so real PEFT models get the cached path."""
        from soup_cli.utils.mole_routing import _model_supports_cache

        inner = _make_cached_mole_model()

        class _Wrapper:
            def forward(self, *args, **kwargs):
                return inner(*args, **kwargs)

            def get_base_model(self):
                return inner

        assert _model_supports_cache(_Wrapper())

    def test_non_model_false(self):
        from soup_cli.utils.mole_routing import _model_supports_cache

        assert not _model_supports_cache(object())


class TestMoleKvCache:
    def _generate(self, model, gate, *, steps=3):
        torch = _torch_or_skip()
        from soup_cli.utils.mole_routing import LoadedMole

        mole = LoadedMole(model, object(), gate, ["task_0", "task_1"])
        ids = torch.tensor([[1, 2]])
        attn = torch.ones_like(ids)
        return mole.generate(ids, attn, max_new_tokens=steps)

    def test_cached_output_matches_legacy(self):
        torch = _torch_or_skip()
        out_cached = self._generate(_make_cached_mole_model(), _make_uniform_gate())
        out_legacy = self._generate(_make_legacy_mole_model(), _make_uniform_gate())
        assert torch.equal(out_cached, out_legacy)

    def test_cached_path_feeds_deltas(self):
        model = _make_cached_mole_model()
        self._generate(model, _make_uniform_gate(), steps=3)
        base_feeds = [t for mode, t in model.calls if mode == "base"]
        a0_feeds = [t for mode, t in model.calls if mode == "task_0"]
        # prompt (2 tokens) once, then 1 token per subsequent step.
        assert base_feeds == [2, 1, 1]
        assert a0_feeds == [2, 1, 1]

    def test_legacy_model_refeeds_full_prefix(self):
        model = _make_legacy_mole_model()
        self._generate(model, _make_uniform_gate(), steps=3)
        base_feeds = [t for mode, t in model.calls if mode == "base"]
        assert base_feeds == [2, 3, 4]

    def test_topk_skip_catches_up_cache(self):
        """An adapter skipped by top-k catches up its cache (fed the missed
        tokens) when it becomes active again — caches stay in lockstep."""
        model = _make_cached_mole_model()
        # step 1: adapter 0 only; step 2: adapter 1 only; step 3: adapter 0.
        gate = _make_scripted_gate([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self._generate(model, gate, steps=3)
        a0_feeds = [t for mode, t in model.calls if mode == "task_0"]
        a1_feeds = [t for mode, t in model.calls if mode == "task_1"]
        # adapter 0: prompt (2) at step 1, skipped at step 2, catch-up of the
        # 2 missed tokens at step 3.
        assert a0_feeds == [2, 2]
        # adapter 1: skipped at step 1, catch-up (prompt + 1 new = 3) at step 2.
        assert a1_feeds == [3]

    def test_topk_skip_catchup_output_matches_legacy(self):
        """The cached top-k skip/catch-up path produces bit-identical output to
        a full-re-forward legacy run driven by the SAME scripted gate. The fake
        bakes the per-adapter prefix length into its logits, so an off-by-one in
        ``state.lens`` bookkeeping or a corrupted catch-up cache would diverge
        here (M4 — the equality must hold THROUGH the skip/catch-up sequence)."""
        torch = _torch_or_skip()
        script = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        out_cached = self._generate(
            _make_cached_mole_model(), _make_scripted_gate(script), steps=4
        )
        out_legacy = self._generate(
            _make_legacy_mole_model(), _make_scripted_gate(script), steps=4
        )
        assert torch.equal(out_cached, out_legacy)

    def test_no_past_in_output_falls_back(self):
        """A capable-signature model that ignores use_cache degrades to the
        legacy full-re-forward path mid-call without crashing."""
        _torch_or_skip()
        from torch import nn

        base = _make_cached_mole_model()

        class _NoPast(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = base
                self.calls = base.calls

            @property
            def device(self):
                return base.device

            def disable_adapter(self):
                return base.disable_adapter()

            def set_adapter(self, name):
                base.set_adapter(name)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                past_key_values=None,
                use_cache=False,
            ):
                out = base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                )
                return out  # never returns past_key_values

        out = self._generate(_NoPast(), _make_uniform_gate(), steps=2)
        assert out.shape[1] == 4  # 2 prompt + 2 generated

    def test_state_is_per_call(self):
        """Two sequential generate() calls don't share cache state."""
        torch = _torch_or_skip()
        from soup_cli.utils.mole_routing import LoadedMole

        model = _make_cached_mole_model()
        mole = LoadedMole(model, object(), _make_uniform_gate(), ["task_0", "task_1"])
        ids = torch.tensor([[1, 2]])
        attn = torch.ones_like(ids)
        mole.generate(ids, attn, max_new_tokens=2)
        model.calls.clear()
        mole.generate(ids, attn, max_new_tokens=2)
        base_feeds = [t for mode, t in model.calls if mode == "base"]
        # Second call starts fresh: full prompt again, then a delta.
        assert base_feeds == [2, 1]


# ---------------------------------------------------------------------------
# #143 — first-party deploy-measure generator factories
# ---------------------------------------------------------------------------


class TestValidateMeasureCandidate:
    @pytest.mark.parametrize(
        "candidate",
        ["none", "4bit", "8bit", "gptq", "awq", "aqlm", "eetq", "mxfp4", "fp8",
         "hqq:4bit", "hqq:8bit", "hqq:1bit"],
    )
    def test_known_candidates_accepted(self, candidate):
        from soup_cli.utils.deploy_measure import validate_measure_candidate

        assert validate_measure_candidate(candidate) == candidate

    def test_case_insensitive(self):
        from soup_cli.utils.deploy_measure import validate_measure_candidate

        assert validate_measure_candidate("GPTQ") == "gptq"

    @pytest.mark.parametrize("candidate", ["evil", "", "hqq:5bit", "4 bit"])
    def test_unknown_rejected(self, candidate):
        from soup_cli.utils.deploy_measure import validate_measure_candidate

        with pytest.raises(ValueError):
            validate_measure_candidate(candidate)

    def test_bool_rejected(self):
        from soup_cli.utils.deploy_measure import validate_measure_candidate

        with pytest.raises(TypeError):
            validate_measure_candidate(True)

    def test_null_byte_rejected(self):
        from soup_cli.utils.deploy_measure import validate_measure_candidate

        with pytest.raises(ValueError):
            validate_measure_candidate("4bit\x00")


class TestMeasureGeneratorFactories:
    def _patch_loader(self, monkeypatch):
        """Stub the model loader + generator builder; record invocations."""
        from soup_cli.utils import deploy_measure as dm

        loads: list[dict] = []

        def fake_load(base, *, quantization, device=None, trust_remote_code=False):
            loads.append({"base": base, "quantization": quantization})
            return ("model", "tok", "cpu")

        def fake_make_generator(model_id, *, loaded=None, max_new_tokens=64, **kw):
            assert loaded == ("model", "tok", "cpu")
            return lambda prompt: f"gen:{prompt}"

        monkeypatch.setattr(dm, "_load_measure_model", fake_load)
        import soup_cli.utils.live_eval as live_eval

        monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)
        return loads

    def test_before_generator_is_lazy(self, monkeypatch):
        from soup_cli.utils.deploy_measure import build_before_generator

        loads = self._patch_loader(monkeypatch)
        gen = build_before_generator("org/tiny")
        assert loads == []  # nothing loaded at build time (cache-hit safety)
        assert gen("hello") == "gen:hello"
        assert loads == [{"base": "org/tiny", "quantization": "none"}]
        gen("again")
        assert len(loads) == 1  # loaded once, reused

    def test_after_factory_threads_candidate_quant(self, monkeypatch):
        from soup_cli.utils.deploy_measure import build_after_generator_factory

        loads = self._patch_loader(monkeypatch)
        factory = build_after_generator_factory("org/tiny")
        gen = factory("4bit")
        assert loads == []
        assert gen("p") == "gen:p"
        assert loads == [{"base": "org/tiny", "quantization": "4bit"}]

    def test_after_factory_rejects_unknown_candidate_eagerly(self, monkeypatch):
        from soup_cli.utils.deploy_measure import build_after_generator_factory

        self._patch_loader(monkeypatch)
        factory = build_after_generator_factory("org/tiny")
        with pytest.raises(ValueError, match="evil"):
            factory("evil")

    def test_builders_validate_base(self):
        from soup_cli.utils.deploy_measure import (
            build_after_generator_factory,
            build_before_generator,
        )

        for builder in (build_before_generator, build_after_generator_factory):
            with pytest.raises(ValueError):
                builder("")
            with pytest.raises(ValueError):
                builder("a\x00b")

    def test_builders_validate_max_new_tokens(self):
        from soup_cli.utils.deploy_measure import build_before_generator

        with pytest.raises(ValueError):
            build_before_generator("org/tiny", max_new_tokens=0)
        with pytest.raises(ValueError):
            build_before_generator("org/tiny", max_new_tokens=True)

    def test_builders_base_len_boundary(self, monkeypatch):
        """``_MAX_BASE_LEN`` is 512: exactly 512 accepted, 513 rejected."""
        from soup_cli.utils.deploy_measure import (
            build_after_generator_factory,
            build_before_generator,
        )

        # Build is lazy (no load) — stub the loader so the 512-char case can't
        # accidentally touch the network even if a later .call happened.
        self._patch_loader(monkeypatch)
        for builder in (build_before_generator, build_after_generator_factory):
            # 512 chars: accepted (returns a callable / factory, no raise).
            assert builder("a" * 512) is not None
            with pytest.raises(ValueError, match="512"):
                builder("a" * 513)

    def test_load_measure_model_builds_quant_config(self, monkeypatch):
        """_load_measure_model routes the candidate through the Quant Menu
        loader (capture tcfg.quantization) without loading a real model."""
        import soup_cli.utils.deploy_measure as dm

        captured = {}

        import soup_cli.utils.quant_menu as quant_menu

        def fake_build(*, tcfg, base, console=None):
            captured["quantization"] = tcfg.quantization
            captured["base"] = base
            return None  # behave like quantization='none'

        monkeypatch.setattr(
            quant_menu, "build_quantization_config_for_loader", fake_build
        )

        class _FakeTok:
            pad_token = "x"
            eos_token = "x"

        class _FakeModel:
            def to(self, dev):
                return self

            def eval(self):
                return self

        fake_tf = types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: _FakeModel()
            ),
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: _FakeTok()
            ),
        )
        monkeypatch.setattr(dm, "_import_transformers", lambda: fake_tf)
        model, tok, dev = dm._load_measure_model(
            "org/tiny", quantization="4bit", device="cpu"
        )
        assert captured == {"quantization": "4bit", "base": "org/tiny"}
        assert dev == "cpu"

    def test_deploy_cli_wires_first_party_factories(self):
        """Source-grep: the CLI uses the first-party builders; the empty-string
        placeholders are gone. The injected seams still win when set."""
        src = (_SRC / "commands" / "deploy.py").read_text(encoding="utf-8")
        assert "build_before_generator" in src
        assert "build_after_generator_factory" in src
        assert "_placeholder_before" not in src
        assert "_DEPLOY_MEASURE_BEFORE_GEN" in src  # seam retained


def _write_measure_tasks(tmp_path: Path) -> Path:
    """Write a tiny 2-row JSONL eval task file (mirrors test_v0531_109)."""
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        '{"prompt": "say hello", "expected": "hello", "scoring": "exact"}\n'
        '{"prompt": "say world", "expected": "world", "scoring": "exact"}\n',
        encoding="utf-8",
    )
    return f


class TestDeployMeasureLiveFailure:
    def test_live_measure_failure_exits_1_no_traceback(self, tmp_path, monkeypatch):
        """#143 — a model-load failure from the live factory (missing quant
        kernel, OOM) surfaces as a friendly exit 1 with no traceback leak."""
        import typer

        from soup_cli.commands.deploy import autopilot
        from soup_cli.utils import deploy_measure as _dm

        monkeypatch.chdir(tmp_path)
        tasks = _write_measure_tasks(tmp_path)

        def boom(prompt):
            raise RuntimeError("no awq kernel")

        # Injected before-gen raises mid-eval; surfaces through run_measure to
        # the CLI's (RuntimeError, ImportError, OSError) -> exit 1 branch.
        monkeypatch.setattr(
            _dm, "_DEPLOY_MEASURE_BEFORE_GEN", boom, raising=False
        )
        monkeypatch.setattr(
            _dm,
            "_DEPLOY_MEASURE_AFTER_FACTORY",
            lambda candidate: (lambda p: "x"),
            raising=False,
        )
        monkeypatch.setenv(
            "SOUP_DEPLOY_AUTOPILOT_CACHE", str(tmp_path / "cache.json")
        )

        app = typer.Typer()
        app.command()(autopilot)
        result = runner.invoke(
            app,
            [
                "--target", "rtx-4090-24gb",
                "--base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--recipe-out", str(tmp_path / "recipe.yaml"),
                "--script-out", str(tmp_path / "deploy.sh"),
                "--measure",
                "--tasks", str(tasks),
                "--measure-candidates", "awq",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "Live measure failed" in result.output
        # No raw traceback text leaks into the output.
        assert "Traceback" not in result.output
        assert "no awq kernel" in result.output


# ---------------------------------------------------------------------------
# #265-partial — live-codec TTS (Orpheus via SNAC)
# ---------------------------------------------------------------------------


def _write_wav(path: Path, *, sr: int = 24_000, seconds: float = 0.05,
               channels: int = 1) -> None:
    """Write a tiny PCM16 sine WAV using only the stdlib."""
    import struct

    n = int(sr * seconds)
    frames = bytearray()
    for i in range(n):
        val = int(12_000 * math.sin(2 * math.pi * 220.0 * i / sr))
        for _ in range(channels):
            frames += struct.pack("<h", val)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(bytes(frames))


class TestInterleaveOrpheusCodes:
    def test_known_interleave(self):
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        # 1 frame: c0=[5], c1=[7, 8], c2=[1, 2, 3, 4]
        out = interleave_orpheus_codes([5], [7, 8], [1, 2, 3, 4])
        # slot order: c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2],
        # c2[4i+3]; index = code + 10 + slot*4096.
        assert out == [
            5 + 10,
            7 + 10 + 4096,
            1 + 10 + 2 * 4096,
            2 + 10 + 3 * 4096,
            8 + 10 + 4 * 4096,
            3 + 10 + 5 * 4096,
            4 + 10 + 6 * 4096,
        ]

    def test_two_frames_length(self):
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        out = interleave_orpheus_codes([0, 1], [0, 1, 2, 3], list(range(8)))
        assert len(out) == 14  # 7 slots per frame

    def test_length_mismatch_rejected(self):
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        with pytest.raises(ValueError, match="medium"):
            interleave_orpheus_codes([0], [0], [0, 1, 2, 3])
        with pytest.raises(ValueError, match="fine"):
            interleave_orpheus_codes([0], [0, 1], [0, 1, 2])

    def test_empty_rejected(self):
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        with pytest.raises(ValueError):
            interleave_orpheus_codes([], [], [])

    def test_out_of_range_code_rejected(self):
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        with pytest.raises(ValueError, match="4096"):
            interleave_orpheus_codes([4096], [0, 1], [0, 1, 2, 3])
        with pytest.raises(ValueError):
            interleave_orpheus_codes([-1], [0, 1], [0, 1, 2, 3])

    def test_bool_code_rejected(self):
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        with pytest.raises(TypeError):
            interleave_orpheus_codes([True], [0, 1], [0, 1, 2, 3])

    def test_max_boundary_code_accepted(self):
        """4095 (one below the 4096 codebook size) is in-bucket and accepted."""
        from soup_cli.utils.tts_codec import interleave_orpheus_codes

        out = interleave_orpheus_codes([4095], [4095, 4095], [4095, 4095, 4095, 4095])
        assert len(out) == 7
        # First slot: code + offset (no codebook multiple).
        assert out[0] == 4095 + 10


class TestOrpheusTokenString:
    def test_token_string_format(self):
        from soup_cli.utils.tts_codec import orpheus_tokens_to_string

        assert (
            orpheus_tokens_to_string([15, 4106])
            == "<custom_token_15><custom_token_4106>"
        )

    def test_empty_rejected(self):
        from soup_cli.utils.tts_codec import orpheus_tokens_to_string

        with pytest.raises(ValueError):
            orpheus_tokens_to_string([])

    def test_bool_index_rejected(self):
        from soup_cli.utils.tts_codec import orpheus_tokens_to_string

        with pytest.raises(TypeError):
            orpheus_tokens_to_string([True])

    def test_float_index_rejected(self):
        from soup_cli.utils.tts_codec import orpheus_tokens_to_string

        with pytest.raises(TypeError):
            orpheus_tokens_to_string([1.5])


class TestLoadAudioMono:
    def test_loads_mono_24k(self, tmp_path):
        np = pytest.importorskip("numpy")
        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import load_audio_mono

        wav = tmp_path / "a.wav"
        _write_wav(wav, sr=24_000, seconds=0.05)
        audio = load_audio_mono(str(wav))
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert abs(len(audio) - 1200) <= 2

    def test_resamples_other_rates(self, tmp_path):
        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import load_audio_mono

        wav = tmp_path / "b.wav"
        _write_wav(wav, sr=8_000, seconds=0.05)
        audio = load_audio_mono(str(wav))
        # 0.05s at 24k target ≈ 1200 samples after resample.
        assert abs(len(audio) - 1200) <= 8

    def test_stereo_mixdown(self, tmp_path):
        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import load_audio_mono

        wav = tmp_path / "c.wav"
        _write_wav(wav, sr=24_000, seconds=0.05, channels=2)
        audio = load_audio_mono(str(wav))
        assert audio.ndim == 1

    def test_missing_file_rejected(self, tmp_path):
        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import load_audio_mono

        with pytest.raises(FileNotFoundError):
            load_audio_mono(str(tmp_path / "nope.wav"))

    def test_null_byte_rejected(self):
        from soup_cli.utils.tts_codec import load_audio_mono

        with pytest.raises(ValueError):
            load_audio_mono("a\x00b.wav")

    def test_non_string_path_rejected(self):
        from soup_cli.utils.tts_codec import load_audio_mono

        with pytest.raises((ValueError, TypeError)):
            load_audio_mono(123)

    def test_empty_path_rejected(self):
        from soup_cli.utils.tts_codec import load_audio_mono

        with pytest.raises(ValueError):
            load_audio_mono("")

    @pytest.mark.skipif(
        not hasattr(__import__("os"), "symlink"),
        reason="symlink rejection needs os.symlink",
    )
    def test_symlink_rejected(self, tmp_path):
        import os

        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import load_audio_mono

        real = tmp_path / "real.wav"
        _write_wav(real, sr=24_000, seconds=0.05)
        link = tmp_path / "link.wav"
        try:
            os.symlink(real, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation unavailable (needs privilege)")
        with pytest.raises(ValueError, match="symlink"):
            load_audio_mono(str(link))

    def test_byte_size_cap(self, tmp_path, monkeypatch):
        pytest.importorskip("soundfile")
        import soup_cli.utils.tts_codec as tc

        monkeypatch.setattr(tc, "_MAX_AUDIO_BYTES", 16)
        wav = tmp_path / "big.wav"
        _write_wav(wav, sr=24_000, seconds=0.05)
        with pytest.raises(ValueError, match="byte"):
            tc.load_audio_mono(str(wav))

    def test_duration_cap(self, tmp_path, monkeypatch):
        pytest.importorskip("soundfile")
        import soup_cli.utils.tts_codec as tc

        monkeypatch.setattr(tc, "_MAX_AUDIO_SECONDS", 0.01)
        wav = tmp_path / "long.wav"
        _write_wav(wav, sr=24_000, seconds=0.05)
        with pytest.raises(ValueError, match="seconds"):
            tc.load_audio_mono(str(wav))


class _FakeSnac:
    """SNAC stand-in: .encode(wav) -> 3 code tensors of the 1/2/4 ratio."""

    def __init__(self, frames: int = 2):
        self.frames = frames

    def encode(self, wav):
        import torch

        t = self.frames
        return [
            torch.arange(t).unsqueeze(0),
            torch.arange(2 * t).unsqueeze(0),
            torch.arange(4 * t).unsqueeze(0),
        ]


class TestEncodeAudioOrpheus:
    def test_encodes_with_injected_model(self, tmp_path):
        _torch_or_skip()
        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import encode_audio_orpheus

        wav = tmp_path / "a.wav"
        _write_wav(wav)
        out = encode_audio_orpheus(str(wav), snac_model=_FakeSnac(frames=2))
        assert out.startswith("<custom_token_")
        assert out.count("<custom_token_") == 14  # 2 frames * 7 slots

    def test_missing_snac_friendly_error(self, tmp_path, monkeypatch):
        """When the snac package is absent the error names the pip install."""
        _torch_or_skip()
        pytest.importorskip("soundfile")
        import sys

        import soup_cli.utils.tts_codec as tc

        monkeypatch.setitem(sys.modules, "snac", None)
        monkeypatch.setattr(tc, "_SNAC_CACHE", {})
        wav = tmp_path / "a.wav"
        _write_wav(wav)
        with pytest.raises(ImportError, match="pip install snac"):
            tc.encode_audio_orpheus(str(wav))


class TestTtsEncoderDispatch:
    def test_orpheus_returns_callable(self):
        from soup_cli.utils.tts_codec import tts_encoder_for_family

        enc = tts_encoder_for_family("orpheus")
        assert callable(enc)

    @pytest.mark.parametrize("family", ["sesame_csm", "llasa", "spark", "oute"])
    def test_other_families_tracked_in_265(self, family):
        from soup_cli.utils.tts_codec import tts_encoder_for_family

        with pytest.raises(RuntimeError, match="#265"):
            tts_encoder_for_family(family)

    def test_unknown_family_rejected(self):
        from soup_cli.utils.tts_codec import tts_encoder_for_family

        with pytest.raises(ValueError):
            tts_encoder_for_family("klingon")

    def test_live_codec_families_constant(self):
        from soup_cli.utils.tts_codec import LIVE_CODEC_FAMILIES

        assert LIVE_CODEC_FAMILIES == frozenset({"orpheus"})


class TestEncodeTtsRow:
    def _encoder(self, path):
        return "<custom_token_10>"

    def test_appends_assistant_turn(self):
        from soup_cli.utils.tts_codec import encode_tts_row

        row = {
            "audio": "a.wav",
            "messages": [{"role": "user", "content": "Say hi"}],
        }
        out = encode_tts_row(row, self._encoder)
        assert "audio" not in out
        assert out["messages"][-1] == {
            "role": "assistant",
            "content": "<custom_token_10>",
        }

    def test_replaces_existing_assistant_turn(self):
        from soup_cli.utils.tts_codec import encode_tts_row

        original_assistant = {"role": "assistant", "content": "placeholder"}
        row = {
            "audio": "a.wav",
            "messages": [
                {"role": "user", "content": "Say hi"},
                original_assistant,
            ],
        }
        out = encode_tts_row(row, self._encoder)
        assert out["messages"][-1]["content"] == "<custom_token_10>"
        assert len(out["messages"]) == 2
        # L8 — the deep-copy guarantee: the ORIGINAL assistant dict is NOT
        # mutated (no nested aliasing between the input row and the output).
        assert original_assistant["content"] == "placeholder"
        assert out["messages"][-1] is not original_assistant

    def test_caller_row_not_mutated(self):
        from soup_cli.utils.tts_codec import encode_tts_row

        messages = [{"role": "user", "content": "Say hi"}]
        row = {"audio": "a.wav", "messages": messages}
        encode_tts_row(row, self._encoder)
        assert row["messages"] is messages
        assert len(messages) == 1
        assert "audio" in row

    def test_missing_audio_rejected(self):
        from soup_cli.utils.tts_codec import encode_tts_row

        with pytest.raises(ValueError, match="audio"):
            encode_tts_row({"messages": []}, self._encoder)

    def test_missing_messages_rejected(self):
        from soup_cli.utils.tts_codec import encode_tts_row

        with pytest.raises(ValueError, match="messages"):
            encode_tts_row({"audio": "a.wav"}, self._encoder)


class TestEncodeTtsDataset:
    def test_maps_train_and_val(self):
        from soup_cli.utils.tts_codec import encode_tts_dataset

        rows = [
            {"audio": "a.wav", "messages": [{"role": "user", "content": "x"}]},
        ]
        ds = {"train": list(rows), "val": list(rows)}
        out = encode_tts_dataset(
            ds, "orpheus", encoder=lambda p: "<custom_token_10>"
        )
        assert out["train"][0]["messages"][-1]["role"] == "assistant"
        assert out["val"][0]["messages"][-1]["role"] == "assistant"
        # input dataset untouched
        assert "audio" in ds["train"][0]

    def test_family_validated(self):
        from soup_cli.utils.tts_codec import encode_tts_dataset

        with pytest.raises(ValueError):
            encode_tts_dataset({"train": []}, "klingon", encoder=lambda p: "x")

    def test_non_dict_rejected(self):
        from soup_cli.utils.tts_codec import encode_tts_dataset

        with pytest.raises(TypeError):
            encode_tts_dataset([], "orpheus", encoder=lambda p: "x")


class TestTtsTrainerLiveCodecWiring:
    def test_live_codec_setup_encodes_then_delegates(self, monkeypatch):
        """Orpheus + data.format='audio' + snac present: setup() encodes the
        rows then falls through to the pre-encoded SFT path (no gate raise)."""
        _torch_or_skip()
        pytest.importorskip("snac")
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.trainer.tts import TTSTrainerWrapper
        from soup_cli.utils import tts_codec

        monkeypatch.setattr(
            tts_codec,
            "tts_encoder_for_family",
            lambda family, device=None: (lambda path: "<custom_token_10>"),
        )
        captured = {}
        monkeypatch.setattr(
            SFTTrainerWrapper,
            "setup",
            lambda self, dataset: captured.update(dataset=dataset),
        )
        w = object.__new__(TTSTrainerWrapper)
        w.config = SimpleNamespace(
            training=SimpleNamespace(tts_family="orpheus", tts_emotion=None),
            data=SimpleNamespace(format="audio", new_special_tokens=None),
        )
        w._tts_family = None
        w.setup(
            {
                "train": [
                    {
                        "audio": "a.wav",
                        "messages": [{"role": "user", "content": "Say hi"}],
                    }
                ]
            }
        )
        row = captured["dataset"]["train"][0]
        assert "audio" not in row
        assert row["messages"][-1] == {
            "role": "assistant",
            "content": "<custom_token_10>",
        }

    def test_trainer_calls_encode_dataset(self):
        src = (_SRC / "trainer" / "tts.py").read_text(encoding="utf-8")
        assert "encode_tts_dataset" in src
        # The unconditional not-yet-validated raise is gone.
        assert "not yet validated" not in src

    def test_dep_gate_still_present(self):
        src = (_SRC / "trainer" / "tts.py").read_text(encoding="utf-8")
        assert "_require_tts_codec" in src

    def test_no_top_level_heavy_imports_in_tts_codec(self):
        for mod in ("torch", "transformers", "snac", "soundfile", "numpy"):
            _assert_no_top_level_import("utils/tts_codec.py", mod)


# ---------------------------------------------------------------------------
# Cross-cutting patch invariants
# ---------------------------------------------------------------------------


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(p) for p in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 22)

    def test_no_top_level_heavy_imports_in_minillm(self):
        for mod in ("torch", "numpy", "transformers", "peft"):
            _assert_no_top_level_import("utils/minillm.py", mod)

    def test_no_top_level_heavy_imports_in_mole_routing(self):
        for mod in ("torch", "numpy", "safetensors", "transformers", "peft"):
            _assert_no_top_level_import("utils/mole_routing.py", mod)

    def test_no_top_level_heavy_imports_in_deploy_measure(self):
        for mod in ("torch", "numpy", "transformers", "peft", "safetensors"):
            _assert_no_top_level_import("utils/deploy_measure.py", mod)
