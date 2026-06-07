"""v0.71.16 — Knowledge edit depth.

Closes:
  * #251 — edit kernels gain GPT-2 ``transformer.h`` / ``mlp.c_proj`` support
           (transpose-aware rank-1 update for the Conv1D weight layout).
  * #252 — EditGovernor edit-count increment is now atomic (baseline-delta
           merge under the cross-process lock) so concurrent ``soup edit set``
           runs cannot lose an increment.
  * #250 — covariance-preconditioned ROME via ``--cov-corpus`` (estimate the
           key covariance C from a stats corpus and use C^{-1} k* instead of
           k*; falls back to C=I when no corpus).
  * #147 — Mixtral joins the LongLoRA architecture allowlist (dedicated
           ``is_mixtral_model`` helper + ``MixtralAttention`` forward override).

Kernel maths are exercised with real torch (the [dev] extra) on tiny CPU
fakes. Full real-model apply paths are covered by the release step-6 smoke
(tiny-gpt2 + SmolLM2-135M).
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn  # assignment (not an import) so it works after importorskip


# ===========================================================================
# Shared fakes — a faithful tiny GPT-2 / Llama LM (forward + loss + tokenizer)
# ===========================================================================


class _Conv1D(nn.Module):
    """Faithful mini ``transformers.pytorch_utils.Conv1D``.

    Weight is ``[nx, nf]`` (transposed relative to nn.Linear's ``[out, in]``),
    ``nf`` is the output feature count, and ``forward`` is ``x @ weight + bias``.
    """

    def __init__(self, nx: int, nf: int):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.randn(nx, nf) * 0.05)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):  # noqa: D401
        return x @ self.weight + self.bias


class _FakeEnc(dict):
    """BatchEncoding-like: dict for ``**`` unpacking + a no-op ``.to``."""

    def to(self, _device):
        return self


class _FakeTok:
    vocab = 24

    def _ids(self, text: str) -> list[int]:
        out = [(sum(ord(c) for c in t) % (self.vocab - 4)) + 1 for t in text.split()]
        return out or [1]

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        ids = self._ids(text)
        if add_special_tokens:
            ids = [0] + ids
        if max_length is not None:
            ids = ids[:max_length]
        if not ids:
            ids = [1]
        if return_tensors == "pt":
            return _FakeEnc(input_ids=torch.tensor([ids], dtype=torch.long))
        return {"input_ids": ids}


def _lm_forward(hidden, lm_head, input_ids, labels):
    logits = lm_head(hidden)
    out = SimpleNamespace(logits=logits)
    if labels is not None:
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        out.loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100
        )
    return out


class _LlamaMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)))


class _LlamaBlock(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.mlp = _LlamaMLP(hidden, inter)

    def forward(self, x):
        return x + self.mlp(x)


class _LlamaInner(nn.Module):
    def __init__(self, vocab, hidden, inter, layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_LlamaBlock(hidden, inter) for _ in range(layers)]
        )

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for blk in self.layers:
            x = blk(x)
        return x


class _FakeLlamaLM(nn.Module):
    def __init__(self, vocab=24, hidden=8, inter=16, layers=3):
        super().__init__()
        self.model = _LlamaInner(vocab, hidden, inter, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, labels=None, **_kw):
        return _lm_forward(self.model(input_ids), self.lm_head, input_ids, labels)


class _GPT2MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.c_fc = _Conv1D(hidden, inter)
        self.c_proj = _Conv1D(inter, hidden)

    def forward(self, x):
        return self.c_proj(torch.relu(self.c_fc(x)))


class _GPT2Block(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.mlp = _GPT2MLP(hidden, inter)

    def forward(self, x):
        return x + self.mlp(x)


class _GPT2Inner(nn.Module):
    def __init__(self, vocab, hidden, inter, layers):
        super().__init__()
        self.wte = nn.Embedding(vocab, hidden)
        self.h = nn.ModuleList([_GPT2Block(hidden, inter) for _ in range(layers)])

    def forward(self, input_ids):
        x = self.wte(input_ids)
        for blk in self.h:
            x = blk(x)
        return x


class _FakeGPT2LM(nn.Module):
    def __init__(self, vocab=24, hidden=8, inter=16, layers=3):
        super().__init__()
        self.transformer = _GPT2Inner(vocab, hidden, inter, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, labels=None, **_kw):
        return _lm_forward(
            self.transformer(input_ids), self.lm_head, input_ids, labels
        )


def _peft_wrap(base):
    """Minimal PEFT-style wrapper exposing ``get_base_model``."""

    class _Peft:
        def __init__(self, b):
            self._b = b

        def get_base_model(self):
            return self._b

    return _Peft(base)


class _PeftCallable:
    """Callable PEFT-style wrapper: delegates forward + train/eval to base.

    Unlike ``_peft_wrap`` (locate-only), this supports the full kernel path —
    ``model(**inputs)`` / ``model.training`` / ``eval()`` / ``train()`` — so a
    GPT-2 base can be edited end-to-end while wrapped (L1).
    """

    def __init__(self, base):
        self._b = base

    def get_base_model(self):
        return self._b

    def __call__(self, *a, **k):
        return self._b(*a, **k)

    @property
    def training(self):
        return self._b.training

    def eval(self):
        self._b.eval()
        return self

    def train(self, mode=True):
        self._b.train(mode)
        return self


_SUBJECT = "Paris is the capital of"
_TARGET = "Lyon"
_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "lorem ipsum dolor sit amet consectetur",
    "machine learning models edit facts surgically",
]


# ===========================================================================
# #251 — GPT-2 transformer.h / mlp.c_proj support
# ===========================================================================


class TestLocateDecoderLayersGpt2:
    def test_locates_transformer_h(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        model = _FakeGPT2LM(layers=4)
        layers = _locate_decoder_layers(model)
        assert len(layers) == 4

    def test_peft_wrapped_gpt2(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        model = _peft_wrap(_FakeGPT2LM(layers=3))
        layers = _locate_decoder_layers(model)
        assert len(layers) == 3

    def test_peft_wrapped_llama_still_works(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        model = _peft_wrap(_FakeLlamaLM(layers=2))
        layers = _locate_decoder_layers(model)
        assert len(layers) == 2

    def test_unknown_arch_raises(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        with pytest.raises(ValueError, match="decoder layers"):
            _locate_decoder_layers(nn.Linear(2, 2))

    def test_get_base_model_raises_swallowed(self):
        """A PEFT wrapper whose get_base_model() blows up must fall through to
        the clear ValueError (DEBUG-logged, not masked) — review L2."""
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        class _BadPeft:
            def get_base_model(self):
                raise RuntimeError("boom")

        with pytest.raises(ValueError, match="decoder layers"):
            _locate_decoder_layers(_BadPeft())


class TestDownProjGpt2:
    def test_returns_c_proj(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers

        model = _FakeGPT2LM()
        down = _down_proj(_locate_decoder_layers(model), 1)
        assert hasattr(down, "weight")
        # Conv1D weight is [in, out] = [inter, hidden] = [16, 8].
        assert tuple(down.weight.shape) == (16, 8)
        assert down.nf == 8

    def test_llama_down_proj_unchanged(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers

        model = _FakeLlamaLM()
        down = _down_proj(_locate_decoder_layers(model), 0)
        # nn.Linear weight is [out, in] = [hidden, inter] = [8, 16].
        assert tuple(down.weight.shape) == (8, 16)

    def test_out_of_range(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers

        layers = _locate_decoder_layers(_FakeGPT2LM())
        with pytest.raises(ValueError, match="out of range"):
            _down_proj(layers, 99)


class TestProjHelpers:
    def test_is_transposed_proj_conv1d(self):
        from soup_cli.utils.edit_kernels import _is_transposed_proj

        assert _is_transposed_proj(_Conv1D(16, 8)) is True

    def test_is_transposed_proj_linear(self):
        from soup_cli.utils.edit_kernels import _is_transposed_proj

        assert _is_transposed_proj(nn.Linear(16, 8, bias=False)) is False

    def test_proj_out_dim_conv1d_uses_nf(self):
        from soup_cli.utils.edit_kernels import _proj_out_dim

        # Conv1D nf = output (hidden) dim, NOT weight.shape[0] (= in dim).
        assert _proj_out_dim(_Conv1D(16, 8)) == 8

    def test_proj_out_dim_linear_uses_shape0(self):
        from soup_cli.utils.edit_kernels import _proj_out_dim

        assert _proj_out_dim(nn.Linear(16, 8, bias=False)) == 8

    def test_is_transposed_proj_rejects_bool_nf(self):
        """``nf=True`` (bool, a subclass of int) must NOT be treated as Conv1D
        — review L3."""
        from soup_cli.utils.edit_kernels import _is_transposed_proj

        assert _is_transposed_proj(SimpleNamespace(nf=True)) is False


class TestRank1UpdateTransposed:
    def test_conv1d_post_condition(self):
        """Conv1D: ``key @ W`` must gain exactly ``delta`` after the update."""
        from soup_cli.utils.edit_kernels import _rank1_update

        conv = _Conv1D(16, 8)
        key = torch.ones(16)
        delta = torch.full((8,), 0.5)
        before = key @ conv.weight  # [out] = [8]
        norm = _rank1_update(conv, key, delta)
        after = key @ conv.weight
        assert norm > 0
        assert torch.allclose(after - before, delta, atol=1e-4)

    def test_linear_post_condition_regression(self):
        from soup_cli.utils.edit_kernels import _rank1_update

        lin = nn.Linear(16, 8, bias=False)
        key = torch.ones(16)
        delta = torch.full((8,), 0.5)
        before = lin.weight @ key
        norm = _rank1_update(lin, key, delta)
        after = lin.weight @ key
        assert norm > 0
        assert torch.allclose(after - before, delta, atol=1e-4)

    def test_conv1d_zero_key_rejected(self):
        from soup_cli.utils.edit_kernels import _rank1_update

        with pytest.raises(ValueError, match="zero norm"):
            _rank1_update(_Conv1D(16, 8), torch.zeros(16), torch.ones(8))


class TestAlphaEditProjectTransposed:
    def test_conv1d_shape_and_determinism(self):
        from soup_cli.utils.edit_kernels import _alphaedit_project

        conv = _Conv1D(16, 8)
        # Logical [out, in] update.
        upd = torch.full((8, 16), 0.3)
        p1 = _alphaedit_project(conv, upd)
        p2 = _alphaedit_project(conv, upd)
        assert tuple(p1.shape) == (8, 16)
        assert torch.allclose(p1, p2)

    def test_conv1d_projection_idempotent(self):
        from soup_cli.utils.edit_kernels import _alphaedit_project

        conv = _Conv1D(16, 8)
        upd = torch.randn(8, 16)
        once = _alphaedit_project(conv, upd)
        twice = _alphaedit_project(conv, once)
        # P is a projection: P(P(u)) == P(u).
        assert torch.allclose(once, twice, atol=1e-4)


class TestApplyKernelsGpt2EndToEnd:
    def _run(self, method, model):
        from soup_cli.utils.edit_kernels import measure_target_prob, run_edit_kernel

        tok = _FakeTok()
        before = measure_target_prob(
            model, tok, subject=_SUBJECT, target=_TARGET, device="cpu"
        )
        result = run_edit_kernel(
            model, tok, method=method, subject=_SUBJECT, target=_TARGET,
            layer=1, device="cpu",
        )
        after = measure_target_prob(
            model, tok, subject=_SUBJECT, target=_TARGET, device="cpu"
        )
        return before, result, after

    def test_rome_gpt2_changes_target(self):
        torch.manual_seed(0)
        before, result, after = self._run("rome", _FakeGPT2LM(layers=3))
        assert result.method == "rome"
        assert result.layers_edited == (1,)
        assert result.norm_delta > 0
        assert 0.0 <= after <= 1.0
        assert after > before  # the fact was edited

    def test_rome_llama_regression(self):
        torch.manual_seed(0)
        before, result, after = self._run("rome", _FakeLlamaLM(layers=3))
        assert result.norm_delta > 0
        assert after > before

    def test_memit_gpt2_runs(self):
        torch.manual_seed(1)
        _before, result, after = self._run("memit", _FakeGPT2LM(layers=3))
        assert result.method == "memit"
        assert len(result.layers_edited) >= 1
        assert result.norm_delta > 0
        assert 0.0 <= after <= 1.0

    def test_alphaedit_gpt2_runs(self):
        torch.manual_seed(2)
        _before, result, after = self._run("alphaedit", _FakeGPT2LM(layers=3))
        assert result.method == "alphaedit"
        assert result.layers_edited == (1,)
        assert result.norm_delta > 0
        assert 0.0 <= after <= 1.0

    def test_memit_gpt2_edits_full_band(self):
        """The #251 ``_proj_out_dim`` Conv1D fix makes the MEMIT band dim-check
        MATCH across uniform-width GPT-2 layers, so the whole band is edited."""
        from soup_cli.utils.edit_kernels import apply_memit_edit

        torch.manual_seed(1)
        result = apply_memit_edit(
            _FakeGPT2LM(layers=3), _FakeTok(),
            subject=_SUBJECT, target=_TARGET, layer=2, device="cpu",
        )
        # _MEMIT_BAND=3, layer=2 → band [0, 1, 2]; all same width → all edited.
        assert result.layers_edited == (0, 1, 2)

    def test_memit_raises_when_no_layer_editable(self, monkeypatch):
        """Defensive band-skip → raise branch (#251 Conv1D dim-check).

        A call-counted ``_proj_out_dim`` returns the real dim for the residual
        sizing (first call) then a mismatching dim for every band check, so the
        sole band layer is skipped and the empty-edit guard fires.
        """
        import soup_cli.utils.edit_kernels as ek

        real = ek._proj_out_dim
        state = {"n": 0}

        def fake(module):
            state["n"] += 1
            return real(module) if state["n"] == 1 else real(module) + 1

        monkeypatch.setattr(ek, "_proj_out_dim", fake)
        with pytest.raises(ValueError, match="could not edit any layer"):
            ek.apply_memit_edit(
                _FakeGPT2LM(layers=1), _FakeTok(),
                subject=_SUBJECT, target=_TARGET, layer=0, device="cpu",
            )

    def test_alphaedit_conv1d_weight_orientation(self):
        """AlphaEdit's ``.t()`` apply keeps the Conv1D weight in [in, out]
        layout — an un-transposed [out, in] apply would shape-error in add_
        (review H3)."""
        from soup_cli.utils.edit_kernels import (
            _down_proj,
            _locate_decoder_layers,
            apply_alphaedit_edit,
        )

        torch.manual_seed(3)
        model = _FakeGPT2LM(layers=3)
        down = _down_proj(_locate_decoder_layers(model), 1)
        w0 = down.weight.detach().clone()  # [in, out] = [16, 8]
        apply_alphaedit_edit(
            model, _FakeTok(),
            subject=_SUBJECT, target=_TARGET, layer=1, device="cpu",
        )
        delta_w = down.weight.detach() - w0
        assert tuple(delta_w.shape) == (16, 8)
        assert torch.isfinite(delta_w).all()
        assert float(torch.linalg.norm(delta_w)) > 0

    def test_rome_peft_wrapped_gpt2(self):
        """A PEFT-wrapped GPT-2 base is editable end-to-end through the kernel
        (review L1 — PEFT fallback at the kernel level, not just locate)."""
        from soup_cli.utils.edit_kernels import run_edit_kernel

        torch.manual_seed(0)
        model = _PeftCallable(_FakeGPT2LM(layers=3))
        result = run_edit_kernel(
            model, _FakeTok(), method="rome",
            subject=_SUBJECT, target=_TARGET, layer=1, device="cpu",
        )
        assert result.method == "rome"
        assert result.norm_delta > 0


# ===========================================================================
# #252 — atomic EditGovernor edit-count increment
# ===========================================================================


class TestGovernorAtomicIncrement:
    def test_baseline_set_on_fresh(self):
        from soup_cli.utils.edit_governor import EditGovernor

        gov = EditGovernor(base_model="m")
        assert gov._persisted_edit_count == 0

    def test_baseline_set_on_loaded(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import (
            EditGovernor,
            EditGovernorStore,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "g.db")
        with EditGovernorStore(db) as store:
            gov = EditGovernor(base_model="m")
            gov.record_edit(method="rome", norm_delta=0.1)
            gov.record_edit(method="rome", norm_delta=0.1)
            save_governor(store, gov)
        with EditGovernorStore(db) as store2:
            restored = load_governor(store2, "m")
        # Baseline mirrors the loaded count so a further edit merges as +1.
        assert restored.edit_count == 2
        assert restored._persisted_edit_count == 2

    def test_concurrent_save_merges_increments(self, tmp_path, monkeypatch):
        """Two governors loaded from the same state both record + save.

        With the pre-#252 absolute-write behaviour the second save would
        clobber the first to 1. The baseline-delta merge keeps both → 2.
        """
        from soup_cli.utils.edit_governor import (
            EditGovernorStore,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "g.db")
        with EditGovernorStore(db) as store:
            gov_a = load_governor(store, "m")
            gov_b = load_governor(store, "m")
            gov_a.record_edit(method="rome", norm_delta=0.1)
            gov_b.record_edit(method="rome", norm_delta=0.2)
            save_governor(store, gov_a)  # persists 1
            save_governor(store, gov_b)  # MERGES → persists 2 (not clobber)
            final = load_governor(store, "m")
        assert final.edit_count == 2

    def test_concurrent_save_merges_multi_increments(self, tmp_path, monkeypatch):
        """Two governors record MULTIPLE edits each → merged is the sum of the
        deltas (3 + 2 = 5), proving the baseline-delta merge (not a naive +1
        per save) — review M4."""
        from soup_cli.utils.edit_governor import (
            EditGovernorStore,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "g.db")
        with EditGovernorStore(db) as store:
            a = load_governor(store, "m")
            b = load_governor(store, "m")
            for _ in range(3):
                a.record_edit(method="rome", norm_delta=0.1)
            for _ in range(2):
                b.record_edit(method="rome", norm_delta=0.1)
            save_governor(store, a)  # persists 3
            save_governor(store, b)  # merges +2 → 5 (NOT clobber to 2, NOT +1)
            assert load_governor(store, "m").edit_count == 5

    def test_merge_onto_existing_row(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import (
            EditGovernorStore,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "g.db")
        with EditGovernorStore(db) as store:
            # Seed a persisted count of 5.
            g0 = load_governor(store, "m")
            for _ in range(5):
                g0.record_edit(method="rome", norm_delta=0.1)
            save_governor(store, g0)
        with EditGovernorStore(db) as store2:
            g1 = load_governor(store2, "m")  # baseline 5
            g1.record_edit(method="rome", norm_delta=0.2)  # → 6
            save_governor(store2, g1)
            final = load_governor(store2, "m")
        assert final.edit_count == 6

    def test_save_updates_in_memory_count(self, tmp_path, monkeypatch):
        """After an atomic merge, ``governor.edit_count`` reflects the merged
        value so the CLI summary shows the real count."""
        from soup_cli.utils.edit_governor import (
            EditGovernorStore,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "g.db")
        with EditGovernorStore(db) as store:
            a = load_governor(store, "m")
            b = load_governor(store, "m")
            a.record_edit(method="rome", norm_delta=0.1)
            b.record_edit(method="rome", norm_delta=0.2)
            save_governor(store, a)
            save_governor(store, b)
            # b merged onto a's persisted 1 → 2; b.edit_count must reflect it.
            assert b.edit_count == 2
            # Re-saving b must NOT double-count (baseline now 2).
            save_governor(store, b)
            assert load_governor(store, "m").edit_count == 2

    def test_save_state_atomic_uses_lock(self):
        """Regression: the get+insert in save_state runs under the lock."""
        import inspect

        from soup_cli.utils.edit_governor import EditGovernorStore

        src = inspect.getsource(EditGovernorStore.save_state)
        assert "_cross_process_lock" in src
        assert "get_state" in src  # read inside the lock


# ===========================================================================
# #250 — covariance-preconditioned ROME
# ===========================================================================


class TestEstimateKeyCovariance:
    def test_shape_and_spd(self):
        from soup_cli.utils.edit_kernels import (
            _down_proj,
            _locate_decoder_layers,
            estimate_key_covariance,
        )

        model = _FakeGPT2LM()
        down = _down_proj(_locate_decoder_layers(model), 1)
        cov = estimate_key_covariance(
            model, _FakeTok(), down, _CORPUS, device="cpu",
        )
        # Covariance dim = down-proj INPUT dim (intermediate) = 16.
        assert tuple(cov.shape) == (16, 16)
        # Symmetric.
        assert torch.allclose(cov, cov.t(), atol=1e-5)
        # SPD (ridge guarantees positive eigenvalues).
        eigvals = torch.linalg.eigvalsh(cov)
        assert float(eigvals.min()) > 0.0

    def test_empty_corpus_rejected(self):
        from soup_cli.utils.edit_kernels import (
            _down_proj,
            _locate_decoder_layers,
            estimate_key_covariance,
        )

        model = _FakeGPT2LM()
        down = _down_proj(_locate_decoder_layers(model), 0)
        with pytest.raises(ValueError, match="corpus"):
            estimate_key_covariance(model, _FakeTok(), down, [], device="cpu")

    def test_bad_caps_rejected(self):
        from soup_cli.utils.edit_kernels import (
            _down_proj,
            _locate_decoder_layers,
            estimate_key_covariance,
        )

        model = _FakeGPT2LM()
        down = _down_proj(_locate_decoder_layers(model), 0)
        with pytest.raises(ValueError, match="max_prompts"):
            estimate_key_covariance(
                model, _FakeTok(), down, _CORPUS, device="cpu", max_prompts=0
            )
        with pytest.raises(ValueError, match="max_tokens"):
            estimate_key_covariance(
                model, _FakeTok(), down, _CORPUS, device="cpu", max_tokens=-1
            )
        with pytest.raises(ValueError, match="ridge"):
            estimate_key_covariance(
                model, _FakeTok(), down, _CORPUS, device="cpu", ridge=-1.0
            )

    def test_all_blank_corpus_rejected(self):
        """A corpus where every entry is blank / non-str captures no keys →
        the runtime ``count == 0`` branch raises (review H4)."""
        from soup_cli.utils.edit_kernels import (
            _down_proj,
            _locate_decoder_layers,
            estimate_key_covariance,
        )

        model = _FakeGPT2LM()
        down = _down_proj(_locate_decoder_layers(model), 0)
        with pytest.raises(ValueError, match="no key vectors"):
            estimate_key_covariance(
                model, _FakeTok(), down, ["", "   ", 123], device="cpu"
            )


class TestRank1UpdatePreconditioned:
    def test_post_condition_preserved_linear(self):
        """With C != I the ROME post-condition ``down(key*) += delta`` still
        holds exactly — the covariance only redistributes the update mass."""
        from soup_cli.utils.edit_kernels import _rank1_update

        lin = nn.Linear(16, 8, bias=False)
        key = torch.randn(16)
        delta = torch.randn(8)
        # Arbitrary SPD covariance.
        a = torch.randn(16, 16)
        cov = a @ a.t() + torch.eye(16)
        before = lin.weight @ key
        norm = _rank1_update(lin, key, delta, cov=cov)
        after = lin.weight @ key
        assert norm > 0
        assert torch.allclose(after - before, delta, atol=1e-3)

    def test_post_condition_preserved_conv1d(self):
        from soup_cli.utils.edit_kernels import _rank1_update

        conv = _Conv1D(16, 8)
        key = torch.randn(16)
        delta = torch.randn(8)
        a = torch.randn(16, 16)
        cov = a @ a.t() + torch.eye(16)
        before = key @ conv.weight
        norm = _rank1_update(conv, key, delta, cov=cov)
        after = key @ conv.weight
        assert norm > 0
        assert torch.allclose(after - before, delta, atol=1e-3)

    def test_non_finite_cov_rejected(self):
        """A non-finite covariance must raise (not silently corrupt weights).

        Review-fix: ``denom = NaN`` would slip past the bare ``<= 0`` guard.
        """
        from soup_cli.utils.edit_kernels import _rank1_update

        lin = nn.Linear(16, 8, bias=False)
        bad_cov = torch.full((16, 16), float("nan"))
        with pytest.raises(ValueError):
            _rank1_update(lin, torch.ones(16), torch.ones(8), cov=bad_cov)

    def test_singular_cov_rejected(self):
        """A singular (rank-deficient) covariance makes the solve fail /
        produce a non-finite denom → clean ValueError (review M3)."""
        from soup_cli.utils.edit_kernels import _rank1_update

        lin = nn.Linear(16, 8, bias=False)
        with pytest.raises(ValueError, match="covariance solve failed|degenerate"):
            _rank1_update(
                lin, torch.ones(16), torch.ones(8), cov=torch.zeros(16, 16)
            )

    def test_cov_changes_update_direction(self):
        """A non-identity covariance produces a different update than C=I."""
        from soup_cli.utils.edit_kernels import _rank1_update

        key = torch.randn(16)
        delta = torch.randn(8)
        a = torch.randn(16, 16)
        cov = a @ a.t() + torch.eye(16)

        lin_iden = nn.Linear(16, 8, bias=False)
        w0 = lin_iden.weight.detach().clone()
        _rank1_update(lin_iden, key, delta)
        upd_iden = lin_iden.weight.detach() - w0

        lin_cov = nn.Linear(16, 8, bias=False)
        with torch.no_grad():
            lin_cov.weight.copy_(w0)
        _rank1_update(lin_cov, key, delta, cov=cov)
        upd_cov = lin_cov.weight.detach() - w0

        assert not torch.allclose(upd_iden, upd_cov, atol=1e-3)


class TestApplyRomeWithCovCorpus:
    def test_apply_edit_rome_cov(self, monkeypatch):
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        torch.manual_seed(0)
        model = _FakeGPT2LM(layers=3)
        tok = _FakeTok()
        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (model, tok, "cpu"),
        )
        plan = build_edit_plan(
            base="b", method="rome", subject=_SUBJECT, target=_TARGET, layer=1,
        )
        result = apply_edit(plan, cov_corpus=_CORPUS)
        assert result.method == "rome"
        assert result.norm_delta > 0
        assert result.target_prob_after > result.target_prob_before

    def test_cov_corpus_rejected_for_memit(self, monkeypatch):
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        # load_model_and_tokenizer must NOT be reached — the reject is before it.
        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")),
        )
        plan = build_edit_plan(
            base="b", method="memit", subject="s", target="t",
        )
        with pytest.raises(ValueError, match="cov-corpus"):
            apply_edit(plan, cov_corpus=_CORPUS)

    def test_cov_corpus_rejected_for_alphaedit(self, monkeypatch):
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")),
        )
        plan = build_edit_plan(
            base="b", method="alphaedit", subject="s", target="t",
        )
        with pytest.raises(ValueError, match="cov-corpus"):
            apply_edit(plan, cov_corpus=_CORPUS)

    def test_cov_corpus_rejected_for_grace(self):
        """grace takes a different code path (codebook sidecar) but the cov
        reject still fires first — before any import / model load (review H1)."""
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        plan = build_edit_plan(base="b", method="grace", subject="s", target="t")
        with pytest.raises(ValueError, match="cov-corpus"):
            apply_edit(plan, cov_corpus=_CORPUS)

    def test_cov_reject_runs_before_governor(self, monkeypatch):
        """Order matters: a non-ROME method + cov_corpus + a governor that
        WOULD refuse must report the cov error, not the governance refusal
        (the cov check precedes governor.check_can_edit) — review M5."""
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.edit_governor import EditGovernor
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")),
        )
        gov = EditGovernor(base_model="b")
        gov.record_edit(method="rome", norm_delta=100.0)  # → BLOWUP: would refuse
        plan = build_edit_plan(base="b", method="memit", subject="s", target="t")
        with pytest.raises(ValueError, match="cov-corpus"):
            apply_edit(plan, governor=gov, cov_corpus=_CORPUS)


class TestLoadCovCorpus:
    def test_parses_jsonl_text_field(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        f = tmp_path / "corpus.jsonl"
        f.write_text(
            json.dumps({"text": "row one"}) + "\n"
            + json.dumps({"prompt": "row two"}) + "\n"
            + json.dumps({"content": "row three"}) + "\n",
            encoding="utf-8",
        )
        rows = _load_cov_corpus("corpus.jsonl")
        assert rows == ["row one", "row two", "row three"]

    def test_parses_raw_text_lines(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        f = tmp_path / "corpus.txt"
        f.write_text("plain line one\nplain line two\n", encoding="utf-8")
        rows = _load_cov_corpus("corpus.txt")
        assert rows == ["plain line one", "plain line two"]

    def test_outside_cwd_rejected(self):
        from soup_cli.commands.edit import _load_cov_corpus

        with pytest.raises(ValueError, match="cwd"):
            _load_cov_corpus("/etc/passwd")

    def test_null_byte_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="null"):
            _load_cov_corpus("a\x00b.jsonl")

    def test_missing_file(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _load_cov_corpus("nope.jsonl")

    def test_directory_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        (tmp_path / "adir").mkdir()
        # POSIX: os.open succeeds, fstat → not S_ISREG → "regular file".
        # Windows: os.open on a dir raises PermissionError → "not readable".
        with pytest.raises(ValueError, match="regular file|not readable"):
            _load_cov_corpus("adir")

    def test_empty_corpus_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        f = tmp_path / "empty.jsonl"
        f.write_text("\n   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="no usable"):
            _load_cov_corpus("empty.jsonl")

    def test_jsonl_dict_without_usable_field_skipped(self, tmp_path, monkeypatch):
        """A JSONL object with no text/prompt/content field is silently dropped
        (NOT appended as raw JSON) — review L5."""
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        f = tmp_path / "c.jsonl"
        f.write_text(
            json.dumps({"other": "x"}) + "\n" + json.dumps({"text": "y"}) + "\n",
            encoding="utf-8",
        )
        assert _load_cov_corpus("c.jsonl") == ["y"]

    def test_oversize_rejected(self, tmp_path, monkeypatch):
        """File larger than the byte cap is rejected before any read — L4."""
        import soup_cli.commands.edit as edit_mod
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(edit_mod, "_MAX_COV_CORPUS_BYTES", 8)
        f = tmp_path / "big.txt"
        f.write_text("this is definitely more than eight bytes\n", encoding="utf-8")
        with pytest.raises(ValueError, match="too large"):
            _load_cov_corpus("big.txt")

    def test_line_cap_truncates(self, tmp_path, monkeypatch):
        """Reading stops at the line cap rather than consuming the whole file — L4."""
        import soup_cli.commands.edit as edit_mod
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(edit_mod, "_MAX_COV_CORPUS_LINES", 2)
        f = tmp_path / "many.txt"
        f.write_text("a\nb\nc\nd\n", encoding="utf-8")
        assert _load_cov_corpus("many.txt") == ["a", "b"]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlink_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.edit import _load_cov_corpus

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.jsonl"
        real.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        os.symlink(real, tmp_path / "link.jsonl")
        with pytest.raises(ValueError, match="symlink"):
            _load_cov_corpus("link.jsonl")


# ===========================================================================
# #147 — Mixtral in the LongLoRA allowlist
# ===========================================================================


class TestIsMixtralModel:
    def test_basic(self):
        from soup_cli.utils.longlora import is_mixtral_model

        assert is_mixtral_model("mistralai/Mixtral-8x7B-v0.1") is True
        assert is_mixtral_model("mistralai/Mixtral-8x22B-Instruct-v0.1") is True

    def test_bare_token(self):
        """The start-of-string anchor matches a lone ``mixtral-...`` id (M1)."""
        from soup_cli.utils.longlora import is_mixtral_model

        assert is_mixtral_model("Mixtral-8x7B-v0.1") is True

    def test_not_plain_mistral(self):
        from soup_cli.utils.longlora import is_mixtral_model

        assert is_mixtral_model("mistralai/Mistral-7B-v0.1") is False

    def test_word_boundary(self):
        from soup_cli.utils.longlora import is_mixtral_model

        assert is_mixtral_model("my-mixtralish-finetune") is False
        assert is_mixtral_model("unmixtral-7b") is False

    def test_input_guards(self):
        from soup_cli.utils.longlora import is_mixtral_model

        assert is_mixtral_model("") is False
        with pytest.raises(TypeError):
            is_mixtral_model(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            is_mixtral_model("a\x00b")
        assert is_mixtral_model("a" * 1024) is False


class TestMixtralStillNotMistral:
    def test_is_mistral_model_excludes_mixtral(self):
        from soup_cli.utils.longlora import is_mistral_model

        # Regression — is_mistral_model stays narrow; Mixtral is detected by
        # the dedicated is_mixtral_model helper.
        assert is_mistral_model("mistralai/Mixtral-8x7B-v0.1") is False


class TestMixtralInAllowlist:
    def test_supported(self):
        from soup_cli.utils.longlora import is_supported_longlora_arch

        assert is_supported_longlora_arch("mistralai/Mixtral-8x7B-v0.1") is True
        assert (
            is_supported_longlora_arch("mistralai/Mixtral-8x22B-Instruct-v0.1")
            is True
        )

    def test_unsupported_unchanged(self):
        from soup_cli.utils.longlora import is_supported_longlora_arch

        assert is_supported_longlora_arch("google/gemma-2-9b") is False
        assert is_supported_longlora_arch("databricks/dbrx-base") is False

    def test_separate_qkv_families_includes_mixtral(self):
        from soup_cli.utils.longlora import _SEPARATE_QKV_FAMILIES

        assert "Mixtral" in _SEPARATE_QKV_FAMILIES

    def test_defensive_surface(self):
        """The rewired ``or``-chain must still swallow non-str / null-byte input
        (returns False, never raises) — review M2."""
        from soup_cli.utils.longlora import is_supported_longlora_arch

        assert is_supported_longlora_arch(None) is False
        assert is_supported_longlora_arch(123) is False
        assert is_supported_longlora_arch("a\x00b") is False


class TestValidateLongloraCompatMixtral:
    def test_accepts_mixtral(self, monkeypatch):
        from soup_cli.utils import longlora

        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available", lambda: False
        )
        # Should not raise.
        longlora.validate_longlora_compat(
            model_name="mistralai/Mixtral-8x7B-v0.1",
            task="sft",
            backend="transformers",
            use_ring_attention=False,
        )

    def test_error_message_lists_mixtral(self, monkeypatch):
        from soup_cli.utils import longlora

        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available", lambda: False
        )
        with pytest.raises(ValueError) as exc:
            longlora.validate_longlora_compat(
                model_name="google/gemma-2-9b",
                task="sft",
                backend="transformers",
                use_ring_attention=False,
            )
        assert "Mixtral" in str(exc.value)

    def test_mixtral_ring_attention_rejected(self, monkeypatch):
        """Now that Mixtral passes the arch gate, the downstream ring-attention
        exclusivity becomes reachable for it — confirm it still fires (M6)."""
        from soup_cli.utils import longlora

        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available", lambda: False
        )
        with pytest.raises(ValueError, match="ring"):
            longlora.validate_longlora_compat(
                model_name="mistralai/Mixtral-8x7B-v0.1",
                task="sft",
                backend="transformers",
                use_ring_attention=True,
            )


class TestMixtralForwardOverride:
    def _fake_attn_model(self, cls_name):
        # The override matches on the attention module's CLASS NAME, so the
        # instance's class must literally be named e.g. ``MixtralAttention``.
        def _init(self):
            nn.Module.__init__(self)
            self.head_dim = 4
            self.num_heads = 2
            self.num_key_value_heads = 2
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)

        attn_cls = type(cls_name, (nn.Module,), {"__init__": _init})
        model = nn.Module()
        model.attn = attn_cls()  # registers as a submodule
        return model

    def test_patches_mixtral_attention(self):
        from soup_cli.utils.longlora import LongLoRAForwardOverride

        model = self._fake_attn_model("MixtralAttention")
        with LongLoRAForwardOverride(model, group_size=4):
            assert getattr(
                model.attn.q_proj.forward, "_soup_longlora_patched", False
            )
            assert getattr(
                model.attn.k_proj.forward, "_soup_longlora_patched", False
            )
        # Restored on exit.
        assert not getattr(
            model.attn.q_proj.forward, "_soup_longlora_patched", False
        )

    def test_llama_attention_still_patched(self):
        from soup_cli.utils.longlora import LongLoRAForwardOverride

        model = self._fake_attn_model("LlamaAttention")
        with LongLoRAForwardOverride(model, group_size=4):
            assert getattr(
                model.attn.q_proj.forward, "_soup_longlora_patched", False
            )


class TestMixtralSchemaGate:
    def test_schema_accepts_mixtral(self, monkeypatch):
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.setattr(
            "soup_cli.utils.flash_attn.is_flash_attn_v3_available", lambda: False
        )
        cfg = load_config_from_string(
            "base: mistralai/Mixtral-8x7B-v0.1\n"
            "task: sft\n"
            "data:\n"
            "  train: data.jsonl\n"
            "training:\n"
            "  use_longlora: true\n"
        )
        assert cfg.training.use_longlora is True


# ===========================================================================
# CLI plumbing
# ===========================================================================


class TestCliCovCorpus:
    def test_edit_set_help_has_cov_corpus(self):
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["edit", "set", "--help"])
        assert result.exit_code == 0, result.output
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "cov-corpus" in clean


# ===========================================================================
# Patch invariants
# ===========================================================================


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = soup_cli.__version__.split(".")
        assert (int(parts[0]), int(parts[1]), int(parts[2])) >= (0, 71, 16)

    @pytest.mark.parametrize(
        "module",
        [
            "soup_cli.utils.edit_kernels",
            "soup_cli.utils.edit_governor",
            "soup_cli.utils.longlora",
            "soup_cli.commands.edit",
        ],
    )
    def test_no_top_level_torch(self, module):
        import importlib

        mod = importlib.import_module(module)
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for line in src.splitlines():
            assert not line.startswith("import torch"), module
            assert not line.startswith("from torch"), module

    def test_edit_kernels_no_top_level_safetensors(self):
        import importlib

        mod = importlib.import_module("soup_cli.utils.edit_kernels")
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert "estimate_key_covariance" in src
