"""v0.71.17 — RAG & serve finish (#253, #254, #259, #260).

* #253 — RAFT distractor shuffle is epoch-aware (a per-epoch salt re-permutes
  the documents each training epoch instead of baking one fixed order).
* #254 — ``citation_style`` (+ ``shuffle_seed``) thread through the live
  diagnose citation probe + the ``soup diagnose`` CLI.
* #259 — serve-time MoLE: a training run writes ``mole_manifest.json`` next to
  ``mole_gate.pt``; ``soup serve --mole <dir>`` loads the base + frozen task
  LoRAs + the gate and blends per-token at decode time.
* #260 — ``soup serve --bank`` resolves the active VeRA/VB-LoRA user per
  request via a ``contextvars.ContextVar`` so concurrent requests on a
  threaded server never race on shared instance state.
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest

from soup_cli.config.loader import load_config_from_string

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _raft_row(n_distractors: int = 2) -> dict:
    return {
        "query": "What is the capital of France?",
        "golden_doc": "Paris has been the capital of France since 987 AD.",
        "distractor_docs": [f"Distractor document number {i}." for i in range(n_distractors)],
        "answer": "The capital of France is Paris [doc-id].",
    }


class _WordTok:
    """Tiny whitespace tokenizer with offset mapping (fast-tokenizer-like)."""

    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = []
        offsets = []
        pos = 0
        for tok in text.split(" "):
            if tok:
                start = text.index(tok, pos)
                end = start + len(tok)
                ids.append((hash(tok) % 1000) + 1)
                offsets.append((start, end))
                pos = end
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# ===========================================================================
# #253 — RAFT epoch-aware shuffle
# ===========================================================================


class TestRaftEpochSalt:
    def test_epoch_zero_is_backward_compatible(self):
        from soup_cli.utils.raft import build_raft_prompt

        a = build_raft_prompt(_raft_row(5), shuffle_seed=7, row_index=3)
        b = build_raft_prompt(_raft_row(5), shuffle_seed=7, row_index=3, epoch=0)
        # epoch=0 reproduces the legacy permutation exactly.
        assert a.doc_ids == b.doc_ids
        assert a.golden_doc_id == b.golden_doc_id
        assert a.prompt == b.prompt

    def test_different_epochs_repermute(self):
        from soup_cli.utils.raft import build_raft_prompt

        # With enough distractors the golden doc lands in a different slot for
        # at least one of several epochs (deterministic per epoch).
        golden_slots = {
            build_raft_prompt(
                _raft_row(6), shuffle_seed=1, row_index=2, epoch=e
            ).golden_doc_id
            for e in range(8)
        }
        assert len(golden_slots) > 1

    def test_same_epoch_deterministic(self):
        from soup_cli.utils.raft import build_raft_prompt

        a = build_raft_prompt(_raft_row(5), shuffle_seed=3, row_index=1, epoch=4)
        b = build_raft_prompt(_raft_row(5), shuffle_seed=3, row_index=1, epoch=4)
        assert a.golden_doc_id == b.golden_doc_id

    def test_epoch_bool_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        # Mirrors the row_index policy: bool-as-int → ValueError naming the field.
        with pytest.raises(ValueError, match="epoch"):
            build_raft_prompt(_raft_row(), epoch=True)

    def test_epoch_negative_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises(ValueError, match="epoch"):
            build_raft_prompt(_raft_row(), epoch=-1)

    def test_epoch_non_int_rejected(self):
        from soup_cli.utils.raft import build_raft_prompt

        with pytest.raises((TypeError, ValueError)):
            build_raft_prompt(_raft_row(), epoch="2")


class TestRaftEpochSchema:
    def _yaml(self, extra: str = "") -> str:
        return (
            "base: HuggingFaceTB/SmolLM2-135M\n"
            "task: sft\n"
            "data:\n"
            "  train: raft.jsonl\n"
            "  format: raft\n"
            f"{extra}"
            "training:\n"
            "  epochs: 2\n"
        )

    def test_default_false(self):
        cfg = load_config_from_string(self._yaml())
        assert cfg.data.raft_epoch_shuffle is False

    def test_accept_true(self):
        cfg = load_config_from_string(self._yaml("  raft_epoch_shuffle: true\n"))
        assert cfg.data.raft_epoch_shuffle is True

    def test_non_bool_rejected(self):
        # loader.load_config_from_string re-raises pydantic ValidationError as a
        # friendly ValueError naming the field — assert that field surfaces.
        with pytest.raises(ValueError, match="raft_epoch_shuffle"):
            load_config_from_string(self._yaml("  raft_epoch_shuffle: 3\n"))


class TestRaftEpochCollator:
    def test_state_holds_epoch(self):
        from soup_cli.trainer.raft import RaftEpochState

        st = RaftEpochState()
        assert st.epoch == 0
        st.epoch = 5
        assert st.epoch == 5

    def test_collator_repermutes_per_epoch(self):
        import torch

        from soup_cli.trainer.raft import RaftEpochShuffleCollator, RaftEpochState

        tok = _WordTok()
        state = RaftEpochState()
        collator = RaftEpochShuffleCollator(
            tok, max_length=128, shuffle_seed=1, epoch_state=state
        )
        row = dict(_raft_row(6))
        row["_raft_row_index"] = 0
        state.epoch = 0
        batch0 = collator([row])
        state.epoch = 3
        batch3 = collator([row])
        assert isinstance(batch0["input_ids"], torch.Tensor)
        # Different epoch -> different document order -> different token ids
        # (the prompt embeds the docs in shuffled order).
        same = batch0["input_ids"].shape == batch3["input_ids"].shape and bool(
            torch.equal(batch0["input_ids"], batch3["input_ids"])
        )
        assert not same

    def test_collator_pads_four_columns(self):
        from soup_cli.trainer.raft import RaftEpochShuffleCollator, RaftEpochState

        tok = _WordTok()
        collator = RaftEpochShuffleCollator(
            tok, max_length=128, epoch_state=RaftEpochState()
        )
        r1 = dict(_raft_row(1))
        r1["_raft_row_index"] = 0
        r2 = dict(_raft_row(4))
        r2["_raft_row_index"] = 1
        batch = collator([r1, r2])
        for key in ("input_ids", "attention_mask", "labels", "loss_weights"):
            assert batch[key].shape[0] == 2

    def test_collator_bad_max_length(self):
        from soup_cli.trainer.raft import RaftEpochShuffleCollator, RaftEpochState

        with pytest.raises(ValueError):
            RaftEpochShuffleCollator(_WordTok(), max_length=4, epoch_state=RaftEpochState())


class TestRaftEpochCallback:
    def test_callback_advances_epoch(self):
        from soup_cli.trainer.raft import RaftEpochState, make_raft_epoch_callback

        state = RaftEpochState()
        cb = make_raft_epoch_callback(state)

        class _State:
            epoch = 2.0

        cb.on_epoch_begin(args=None, state=_State(), control=None)
        assert state.epoch == 2

    def test_callback_none_state_noop(self):
        from soup_cli.trainer.raft import RaftEpochState, make_raft_epoch_callback

        state = RaftEpochState()
        cb = make_raft_epoch_callback(state)
        cb.on_epoch_begin(args=None, state=None, control=None)
        assert state.epoch == 0


class TestRaftSetupWiring:
    def test_sft_branches_on_epoch_shuffle(self):
        from soup_cli.trainer import sft

        src = inspect.getsource(sft)
        assert "raft_epoch_shuffle" in src
        assert "RaftEpochShuffleCollator" in src
        assert "make_raft_epoch_callback" in src


# ===========================================================================
# #254 — citation_style threads through the diagnose probe
# ===========================================================================


class TestDiagnoseCitationStyle:
    def test_run_live_diagnose_threads_style(self):
        from soup_cli.utils.diagnose import live as live_mod

        sig = inspect.signature(live_mod.run_live_diagnose)
        assert "citation_style" in sig.parameters
        assert "shuffle_seed" in sig.parameters

    def test_score_citation_called_with_style(self, monkeypatch):
        from soup_cli.utils.diagnose import live as live_mod

        captured = {}

        def _fake_score_citation(rows, generator, **kwargs):
            captured.update(kwargs)
            from soup_cli.utils.diagnose.report import FailureScore

            return FailureScore(
                mode="citation", score=1.0, verdict="OK", evidence="fake"
            )

        # Patch the module-level imports used by run_live_diagnose.
        import soup_cli.utils.diagnose.citation as cit_mod

        monkeypatch.setattr(cit_mod, "score_citation", _fake_score_citation)

        # Make load_adapter_pair return cheap closures, no model load.
        def _fake_pair(base, adapter=None, **kwargs):
            def _gen(prompt):
                return "answer [doc-0]"

            def _multi(prompt, k):
                return ["a"] * k

            return {
                "base_gen": _gen,
                "adapter_gen": _gen,
                "base_multi": _multi,
                "adapter_multi": _multi,
            }

        monkeypatch.setattr(live_mod, "load_adapter_pair", _fake_pair)

        # Supply a RAFT-shaped dataset so the citation probe runs.
        rows = [_raft_row(2)]
        monkeypatch.setattr(live_mod, "_load_dataset_rows", lambda p: rows)

        live_mod.run_live_diagnose(
            run_id="r1",
            base="fake-base",
            dataset_path="data.jsonl",
            citation_style="inline",
            shuffle_seed=7,
        )
        assert captured.get("citation_style") == "inline"
        assert captured.get("shuffle_seed") == 7

    def test_cli_has_citation_style_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["diagnose", "--help"])
        assert result.exit_code == 0
        assert "--citation-style" in result.output

    def test_cli_rejects_bad_style(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app,
            ["diagnose", "r1", "--base-model", "x", "--citation-style", "nope"],
        )
        assert result.exit_code == 2
        assert "Invalid --citation-style" in result.output


# ===========================================================================
# #260 — serve --bank per-request X-User-Id (ContextVar, thread-safe)
# ===========================================================================


def _make_bank(dim: int = 8):
    from soup_cli.utils.vector_bank import BankEntry, VectorBank

    return VectorBank(
        name="demo",
        base_model="HuggingFaceTB/SmolLM2-135M",
        projection_seed=42,
        vector_dim=dim,
        entries=(
            BankEntry(user_id="alice", scaling=tuple([0.5] * dim)),
            BankEntry(user_id="bob", scaling=tuple([0.25] * dim)),
        ),
    )


class TestBankActiveUserContextVar:
    def test_uses_contextvar(self):
        from soup_cli.utils import vector_bank

        src = inspect.getsource(vector_bank)
        assert "ContextVar" in src
        assert "import contextvars" in src or "from contextvars" in src

    def test_set_active_user_contract_preserved(self):
        from soup_cli.utils.vector_bank import apply_bank_to_serve

        loaded = apply_bank_to_serve(_make_bank())
        assert loaded.set_active_user("alice") is True
        assert loaded.set_active_user("zoe") is False
        assert loaded._active_user is None

    def test_active_user_reads_back(self):
        from soup_cli.utils.vector_bank import apply_bank_to_serve

        loaded = apply_bank_to_serve(_make_bank())
        loaded.set_active_user("alice")
        assert loaded._active_user == "alice"

    def test_active_user_is_thread_isolated(self):
        import threading

        from soup_cli.utils.vector_bank import apply_bank_to_serve

        loaded = apply_bank_to_serve(_make_bank())
        results: dict[str, object] = {}
        barrier = threading.Barrier(2)

        def _worker(user: str):
            loaded.set_active_user(user)
            barrier.wait()  # ensure both threads have set before either reads
            results[user] = loaded._active_user

        t1 = threading.Thread(target=_worker, args=("alice",))
        t2 = threading.Thread(target=_worker, args=("bob",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Each thread must observe its OWN user, not the other thread's.
        assert results["alice"] == "alice"
        assert results["bob"] == "bob"

    def test_stream_reselects_bank_user(self):
        # v0.71.17 #260 — the streaming generator runs in a different
        # contextvars.Context than the sync endpoint, so it must re-set the
        # active user itself. Verify _stream_response calls set_active_user
        # before generating, and the call site threads loaded_bank + x_user_id.
        from soup_cli.commands import serve

        src = inspect.getsource(serve._stream_response)
        assert "loaded_bank=None" in src and "x_user_id=None" in src
        assert "loaded_bank.set_active_user(x_user_id)" in src
        call_src = inspect.getsource(serve._create_app)
        assert "loaded_bank=_loaded_bank" in call_src
        assert "x_user_id=x_user_id" in call_src

    def test_hook_still_shifts_residual(self):
        import torch
        from torch import nn

        from soup_cli.utils.vector_bank import apply_bank_to_serve

        class _Layer(nn.Module):
            def forward(self, x):
                return (x,)

        inner = nn.Module()
        inner.layers = nn.ModuleList([_Layer(), _Layer()])
        model = nn.Module()
        model.model = inner

        loaded = apply_bank_to_serve(_make_bank(dim=8))
        handle = loaded.install_serve_hook(model, layer=-1, strength=1.0)
        block = model.model.layers[-1]
        x = torch.ones(1, 3, 8)
        assert torch.allclose(block(x)[0], x)  # no active user → no-op
        loaded.set_active_user("alice")
        out = block(x)[0]
        expected = x + loaded.delta_for_user("alice", x)
        assert torch.allclose(out, expected, atol=1e-5)
        handle.remove()


# ===========================================================================
# #259 — serve-time MoLE
# ===========================================================================


class TestMoleServeManifest:
    def _manifest(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        return MoleServeManifest(
            base="HuggingFaceTB/SmolLM2-135M",
            adapters=("./a", "./b"),
            num_task_adapters=2,
            hidden_dim=576,
            top_k=2,
            temperature=1.0,
        )

    def test_happy(self):
        m = self._manifest()
        assert m.num_task_adapters == 2
        assert m.adapters == ("./a", "./b")

    def test_frozen(self):
        import dataclasses

        m = self._manifest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.top_k = 1

    def test_count_must_match_adapters(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        with pytest.raises(ValueError):
            MoleServeManifest(
                base="b",
                adapters=("./a", "./b"),
                num_task_adapters=3,
                hidden_dim=8,
                top_k=2,
                temperature=1.0,
            )

    def test_topk_above_adapters_rejected(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        with pytest.raises(ValueError, match="top_k"):
            MoleServeManifest(
                base="b",
                adapters=("./a", "./b"),
                num_task_adapters=2,
                hidden_dim=8,
                top_k=3,
                temperature=1.0,
            )

    def test_bad_base_rejected(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        with pytest.raises((TypeError, ValueError), match="base"):
            MoleServeManifest(
                base="",
                adapters=("./a", "./b"),
                num_task_adapters=2,
                hidden_dim=8,
                top_k=2,
                temperature=1.0,
            )

    def test_adapters_must_be_tuple(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        with pytest.raises(TypeError, match="tuple"):
            MoleServeManifest(
                base="b",
                adapters=["./a", "./b"],  # list, not tuple
                num_task_adapters=2,
                hidden_dim=8,
                top_k=2,
                temperature=1.0,
            )

    def test_bad_temperature_rejected(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        for bad in (float("nan"), float("inf"), 0.0, -1.0):
            with pytest.raises(ValueError, match="temperature"):
                MoleServeManifest(
                    base="b",
                    adapters=("./a", "./b"),
                    num_task_adapters=2,
                    hidden_dim=8,
                    top_k=2,
                    temperature=bad,
                )

    def test_base_too_long_rejected(self):
        from soup_cli.utils.mole_routing import MoleServeManifest

        with pytest.raises(ValueError, match="base too long"):
            MoleServeManifest(
                base="x" * 100_000,
                adapters=("./a", "./b"),
                num_task_adapters=2,
                hidden_dim=8,
                top_k=2,
                temperature=1.0,
            )

    def test_write_load_roundtrip(self, tmp_path, monkeypatch):
        from soup_cli.utils.mole_routing import (
            load_mole_manifest,
            write_mole_manifest,
        )

        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "mole_out"
        out_dir.mkdir()
        m = self._manifest()
        write_mole_manifest(m, str(out_dir))
        loaded = load_mole_manifest(str(out_dir))
        assert loaded == m

    def test_load_missing_dir(self, tmp_path, monkeypatch):
        from soup_cli.utils.mole_routing import load_mole_manifest

        monkeypatch.chdir(tmp_path)
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            load_mole_manifest(str(tmp_path / "nope"))

    def test_load_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.mole_routing import load_mole_manifest

        work = tmp_path / "work"
        work.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises((ValueError, OSError)):
            load_mole_manifest(str(outside))

    def test_load_tampered_count_rejected(self, tmp_path, monkeypatch):
        # A manifest whose num_task_adapters != len(adapters) fails loud.
        import json as _json

        from soup_cli.utils.mole_routing import load_mole_manifest

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "mole_out"
        d.mkdir()
        (d / "mole_manifest.json").write_text(
            _json.dumps(
                {
                    "base": "b",
                    "adapters": ["./a", "./b"],
                    "num_task_adapters": 5,
                    "hidden_dim": 8,
                    "top_k": 2,
                    "temperature": 1.0,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_mole_manifest("mole_out")

    def test_load_missing_key_rejected(self, tmp_path, monkeypatch):
        import json as _json

        from soup_cli.utils.mole_routing import load_mole_manifest

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "mole_out"
        d.mkdir()
        # Missing 'temperature' -> fail loud (machine-written, no default).
        (d / "mole_manifest.json").write_text(
            _json.dumps(
                {
                    "base": "b",
                    "adapters": ["./a", "./b"],
                    "num_task_adapters": 2,
                    "hidden_dim": 8,
                    "top_k": 2,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_mole_manifest("mole_out")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink only")
    def test_load_symlink_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.mole_routing import (
            load_mole_manifest,
            write_mole_manifest,
        )

        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        write_mole_manifest(self._manifest(), str(real_dir))
        link_dir = tmp_path / "link"
        link_dir.mkdir()
        os.symlink(real_dir / "mole_manifest.json", link_dir / "mole_manifest.json")
        with pytest.raises((ValueError, OSError)):
            load_mole_manifest("link")


class _MoleOut:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


def _make_fake_mole_model(hidden=4, vocab=6):
    import torch
    from torch import nn

    class _FakeMoleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.vocab = vocab
            self._adapter = "task_0"
            self._disabled = False

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

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            b, t = input_ids.shape
            hs = torch.ones(b, t, self.hidden)
            idx = 0 if self._disabled else int(self._adapter.split("_")[1])
            logits = torch.zeros(b, t, self.vocab)
            logits[..., idx % self.vocab] = float(idx + 1)
            return _MoleOut(logits, [hs] if output_hidden_states else None)

    return _FakeMoleModel()


class TestLoadedMole:
    def _loaded(self, n_adapters=2):
        from soup_cli.utils.mole_routing import (
            LoadedMole,
            MoleGatingConfig,
            build_gating_kernel,
        )

        model = _make_fake_mole_model(hidden=4, vocab=6)
        gate = build_gating_kernel(
            MoleGatingConfig(
                num_task_adapters=n_adapters,
                hidden_dim=4,
                temperature=1.0,
                top_k=n_adapters,
            )
        )
        # Zero the gate so softmax over hidden=ones is uniform (controllable).
        gate.gate.weight.data.zero_()
        adapter_names = [f"task_{i}" for i in range(n_adapters)]
        return LoadedMole(model, None, gate, adapter_names)

    def test_blended_last_logits_uniform(self):
        import torch

        loaded = self._loaded(n_adapters=2)
        ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones_like(ids)
        blended = loaded._blended_last_logits(ids, attn)
        assert blended.shape == (1, 6)
        # uniform gate weights 0.5/0.5 -> blend[i] = 0.5 * adapter_i_logit
        # adapter 0 -> value 1 at idx0; adapter 1 -> value 2 at idx1.
        expected = torch.zeros(1, 6)
        expected[0, 0] = 0.5 * 1.0
        expected[0, 1] = 0.5 * 2.0
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_generate_greedy_length(self):
        import torch

        loaded = self._loaded(n_adapters=2)
        ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones_like(ids)
        out = loaded.generate(ids, attn, max_new_tokens=4, temperature=0.0)
        assert out.shape[1] == 3 + 4

    def test_generate_stops_on_eos(self):
        import torch

        loaded = self._loaded(n_adapters=2)
        ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones_like(ids)
        # argmax of uniform blend is idx1 (value 2 > value 1). Set eos=1.
        out = loaded.generate(ids, attn, max_new_tokens=10, eos_token_id=1, temperature=0.0)
        assert out.shape[1] == 4  # one new token then eos

    def test_generate_temperature_sampling(self):
        # The temperature>0 + top_p<1 nucleus path runs without crashing and
        # appends max_new_tokens (the only stop here is the length budget).
        import torch

        torch.manual_seed(0)
        loaded = self._loaded(n_adapters=2)
        ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones_like(ids)
        out = loaded.generate(
            ids, attn, max_new_tokens=3, temperature=0.7, top_p=0.9
        )
        assert out.shape[1] == 3 + 3

    def test_requires_two_adapters(self):
        from soup_cli.utils.mole_routing import (
            LoadedMole,
            MoleGatingConfig,
            build_gating_kernel,
        )

        model = _make_fake_mole_model()
        gate = build_gating_kernel(
            MoleGatingConfig(num_task_adapters=2, hidden_dim=4, temperature=1.0, top_k=2)
        )
        with pytest.raises(ValueError):
            LoadedMole(model, None, gate, ["task_0"])

    def test_generate_bad_max_new_tokens(self):
        import torch

        loaded = self._loaded(n_adapters=2)
        ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones_like(ids)
        with pytest.raises(ValueError):
            loaded.generate(ids, attn, max_new_tokens=0)

    def test_generate_rejects_batch(self):
        import torch

        loaded = self._loaded(n_adapters=2)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # B == 2
        attn = torch.ones_like(ids)
        with pytest.raises(ValueError, match="single sequence"):
            loaded.generate(ids, attn, max_new_tokens=2)

    def test_blend_skips_zero_weight_adapter(self):
        # top_k=1 over 3 adapters: only the top-weight adapter is forwarded.
        import torch

        from soup_cli.utils.mole_routing import (
            LoadedMole,
            MoleGatingConfig,
            build_gating_kernel,
        )

        model = _make_fake_mole_model(hidden=4, vocab=6)
        gate = build_gating_kernel(
            MoleGatingConfig(num_task_adapters=3, hidden_dim=4, temperature=1.0, top_k=1)
        )
        # Bias the gate so adapter 2 wins the top-1 (weight 1.0, others 0.0).
        gate.gate.weight.data.zero_()
        gate.gate.weight.data[2, :] = 10.0
        loaded = LoadedMole(model, None, gate, ["task_0", "task_1", "task_2"])
        ids = torch.tensor([[1, 2, 3]])
        blended = loaded._blended_last_logits(ids, torch.ones_like(ids))
        # Only adapter 2 contributes: value 3 at idx2, weight 1.0.
        expected = torch.zeros(1, 6)
        expected[0, 2] = 3.0
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_generate_text_with_fake_tokenizer(self):
        import torch

        class _Tok:
            eos_token_id = 1
            pad_token = "<pad>"
            chat_template = None

            def __call__(self, text, return_tensors="pt"):
                return {
                    "input_ids": torch.tensor([[2, 3]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

            def decode(self, ids, skip_special_tokens=True):
                return "out"

        from soup_cli.utils.mole_routing import (
            LoadedMole,
            MoleGatingConfig,
            build_gating_kernel,
        )

        model = _make_fake_mole_model(hidden=4, vocab=6)
        gate = build_gating_kernel(
            MoleGatingConfig(num_task_adapters=2, hidden_dim=4, temperature=1.0, top_k=2)
        )
        gate.gate.weight.data.zero_()
        loaded = LoadedMole(model, _Tok(), gate, ["task_0", "task_1"])
        text, p_tok, c_tok = loaded.generate_text(
            [{"role": "user", "content": "hi"}], max_tokens=3, temperature=0.0
        )
        assert text == "out"
        assert p_tok == 2
        assert c_tok >= 1


class TestMoleTrainWritesManifest:
    def test_train_writes_manifest_source(self):
        from soup_cli.trainer import mole_routing as trainer_mole

        src = inspect.getsource(trainer_mole)
        assert "write_mole_manifest" in src
        assert "MoleServeManifest" in src


class TestServeMoleWiring:
    def test_serve_has_mole_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--mole" in result.output

    def test_mole_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            app, ["serve", "--model", "somemodel", "--mole", "/etc/passwd"]
        )
        assert result.exit_code == 2
        assert "Invalid --mole path" in result.output

    def test_mole_requires_transformers_backend(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "mole_out"
        d.mkdir()
        result = CliRunner().invoke(
            app, ["serve", "--model", "somemodel", "--mole", "mole_out", "--backend", "vllm"]
        )
        assert result.exit_code == 2
        assert "requires --backend transformers" in result.output

    def test_mole_rejects_bank_combo(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "mole_out").mkdir()
        (tmp_path / "bank.json").write_text("{}", encoding="utf-8")
        result = CliRunner().invoke(
            app,
            ["serve", "--model", "somemodel", "--mole", "mole_out", "--bank", "bank.json"],
        )
        assert result.exit_code == 2
        assert "cannot be combined" in result.output
        assert "--bank" in result.output

    def test_create_app_accepts_mole_runtime(self):
        from soup_cli.commands.serve import _create_app

        sig = inspect.signature(_create_app)
        assert "mole_runtime" in sig.parameters

    def test_generate_response_branch_present(self):
        from soup_cli.commands import serve

        src = inspect.getsource(serve)
        assert "_mole_runtime" in src
        assert "generate_text" in src

    def test_mole_load_uses_base_model_not_model_path(self):
        # Live-smoke regression: --model is the gate/manifest dir, so the base
        # must come from --base (base_model) / the manifest — NOT from
        # str(model_path) (which is the adapter dir with no base config.json).
        from soup_cli.commands import serve

        src = inspect.getsource(serve)
        assert "load_mole_for_serve(" in src
        assert "base=base_model" in src
        assert "base=str(model_path)" not in src

    def test_mole_train_return_shape_is_generic_compatible(self):
        # Live-smoke regression: the MoLE train() result must carry the keys the
        # generic commands/train.py handler reads (initial_loss / final_loss /
        # total_steps / duration_secs / duration) so the run completes cleanly.
        from soup_cli.trainer import mole_routing as trainer_mole

        src = inspect.getsource(trainer_mole)
        for key in (
            '"initial_loss"',
            '"final_loss"',
            '"total_steps"',
            '"duration_secs"',
            '"duration"',
        ):
            assert key in src, f"MoLE train() return missing {key}"


# ===========================================================================
# Cross-cutting invariants
# ===========================================================================


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(x) for x in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 17), soup_cli.__version__

    @pytest.mark.parametrize(
        "module",
        ["soup_cli.utils.raft", "soup_cli.utils.mole_routing"],
    )
    def test_no_top_level_torch_import(self, module):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src",
            "soup_cli",
            "utils",
            module.rsplit(".", 1)[-1] + ".py",
        )
        with open(src_path, encoding="utf-8") as fh:
            text = fh.read()
        assert "\nimport torch" not in text
        assert "\nfrom torch" not in text


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
