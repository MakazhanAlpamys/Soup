"""v0.71.9 — Knowledge edit + unlearn live wiring.

Closes #193 (NPO/SimNPO/RMU loss kernels + UnlearnTrainerWrapper.setup),
#194 (ROME/MEMIT/AlphaEdit kernels + edit-diff generation), #196 (EditGovernor
SQLite persistence + cross-process locking), #197 (apply_edit consults the
governor automatically), #203 (live GRACE codebook lookup/write + Registry
persistence).

Loss / kernel maths are exercised with real torch (available in the [dev]
extra) on tiny CPU tensors; the full model-load paths (NPO train, ROME edit,
GRACE edit) are covered by the release-step-6 smoke on SmolLM2-135M.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")


# ===========================================================================
# #196 — EditGovernor SQLite persistence + cross-process locking
# ===========================================================================


class TestEditGovernorStore:
    def test_path_validation_null_byte(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernorStore

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            EditGovernorStore("evil\x00.db")

    def test_path_validation_outside_cwd(self):
        from soup_cli.utils.edit_governor import EditGovernorStore

        with pytest.raises(ValueError, match="HOME"):
            EditGovernorStore("/etc/evil_governor.db")

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import (
            EditGovernor,
            EditGovernorStore,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "gov.db")
        with EditGovernorStore(db) as store:
            gov = EditGovernor(base_model="tiny/model")
            gov.record_edit(method="rome", norm_delta=0.3)
            gov.record_edit(method="rome", norm_delta=0.4)
            save_governor(store, gov)
        # Reopen — a fresh process would see the same DB.
        with EditGovernorStore(db) as store2:
            restored = load_governor(store2, "tiny/model")
        assert restored.edit_count == 2
        assert restored.last_method == "rome"
        assert restored.last_verdict == "OK"
        assert abs(restored.last_norm_delta - 0.4) < 1e-9

    def test_load_fresh_when_absent(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernorStore, load_governor

        monkeypatch.chdir(tmp_path)
        with EditGovernorStore(str(tmp_path / "gov.db")) as store:
            gov = load_governor(store, "unseen/model")
        assert gov.edit_count == 0
        assert gov.base_model == "unseen/model"

    def test_get_state_none_for_unknown(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernorStore

        monkeypatch.chdir(tmp_path)
        with EditGovernorStore(str(tmp_path / "g.db")) as store:
            assert store.get_state("nope") is None

    def test_save_state_rejects_non_governor(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernorStore

        monkeypatch.chdir(tmp_path)
        with EditGovernorStore(str(tmp_path / "g.db")) as store:
            with pytest.raises(TypeError):
                store.save_state("not-a-governor")

    def test_persisted_blowup_blocks_next(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import (
            EditGovernor,
            EditGovernorStore,
            GovernedEditError,
            load_governor,
            save_governor,
        )

        monkeypatch.chdir(tmp_path)
        db = str(tmp_path / "g.db")
        with EditGovernorStore(db) as store:
            gov = EditGovernor(base_model="m")
            gov.record_edit(method="rome", norm_delta=99.0)  # BLOWUP
            save_governor(store, gov)
        with EditGovernorStore(db) as store2:
            restored = load_governor(store2, "m")
        assert restored.last_verdict == "BLOWUP"
        with pytest.raises(GovernedEditError):
            restored.check_can_edit()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlink_db_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernorStore

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.db"
        target.write_text("", encoding="utf-8")
        link = tmp_path / "link.db"
        os.symlink(target, link)
        with pytest.raises(ValueError, match="symlink"):
            EditGovernorStore(str(link))


class TestDefaultGovernorDbPath:
    def test_default_under_home(self, monkeypatch):
        from soup_cli.utils.edit_governor import default_governor_db_path

        monkeypatch.delenv("SOUP_EDIT_GOVERNOR_DB", raising=False)
        path = default_governor_db_path()
        assert path.endswith("edit_governor.db")
        assert ".soup" in path

    def test_env_override_in_tmp(self, tmp_path, monkeypatch):
        from soup_cli.utils.edit_governor import default_governor_db_path

        target = str(tmp_path / "custom.db")
        monkeypatch.setenv("SOUP_EDIT_GOVERNOR_DB", target)
        assert default_governor_db_path() == target

    def test_env_override_out_of_bounds_falls_back(self, monkeypatch):
        from soup_cli.utils.edit_governor import default_governor_db_path

        monkeypatch.setenv("SOUP_EDIT_GOVERNOR_DB", "/etc/cron.d/x.db")
        path = default_governor_db_path()
        assert path.endswith("edit_governor.db")
        assert ".soup" in path

    def test_env_override_null_byte_falls_back(self, monkeypatch):
        from soup_cli.utils.edit_governor import default_governor_db_path

        monkeypatch.setenv("SOUP_EDIT_GOVERNOR_DB", "x\x00.db")
        assert default_governor_db_path().endswith("edit_governor.db")


# ===========================================================================
# #197 — apply_edit consults the EditGovernor automatically
# ===========================================================================


class TestApplyEditGovernor:
    def _patch_kernel(self, monkeypatch, *, norm_delta=0.3):
        import soup_cli.utils.edit_kernels as ek
        import soup_cli.utils.live_eval as live_eval

        loaded = {"called": 0}

        def _load(*a, **k):
            loaded["called"] += 1
            return ("M", "T", "cpu")

        monkeypatch.setattr(live_eval, "load_model_and_tokenizer", _load)
        monkeypatch.setattr(ek, "measure_target_prob", lambda *a, **k: 0.0)
        monkeypatch.setattr(
            ek, "run_edit_kernel",
            lambda *a, **k: ek.EditKernelResult(
                method="rome", layer=5, norm_delta=norm_delta, layers_edited=(5,),
            ),
        )
        return loaded

    def test_governor_records_after_edit(self, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernor
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        self._patch_kernel(monkeypatch, norm_delta=0.42)
        gov = EditGovernor(base_model="b")
        plan = build_edit_plan(base="b", method="rome", subject="s", target="t")
        result = apply_edit(plan, governor=gov)
        assert result.governed is True
        assert gov.edit_count == 1
        assert abs(gov.last_norm_delta - 0.42) < 1e-9

    def test_governor_refuses_before_load(self, monkeypatch):
        from soup_cli.utils.edit_governor import EditGovernor, GovernedEditError
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        loaded = self._patch_kernel(monkeypatch)
        gov = EditGovernor(base_model="b", max_sequential_edits=1)
        gov.record_edit(method="rome", norm_delta=0.1)  # now at cap
        plan = build_edit_plan(base="b", method="rome", subject="s", target="t")
        with pytest.raises(GovernedEditError):
            apply_edit(plan, governor=gov)
        # The refusal must short-circuit BEFORE the model load.
        assert loaded["called"] == 0

    def test_no_governor_skips(self, monkeypatch):
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        self._patch_kernel(monkeypatch)
        plan = build_edit_plan(base="b", method="rome", subject="s", target="t")
        result = apply_edit(plan)
        assert result.governed is False


# ===========================================================================
# #193 — unlearn loss kernels
# ===========================================================================


class TestUnlearnKernels:
    def test_sequence_logprob_masks_prompt(self):
        from soup_cli.utils.unlearn_kernels import sequence_logprob

        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[-100, -100, 3, 5]])
        lp = sequence_logprob(logits, labels)
        assert lp.shape == (1,)
        # Only the 2 supervised positions contribute (finite negative).
        assert lp.item() < 0

    def test_sequence_lengths(self):
        from soup_cli.utils.unlearn_kernels import sequence_lengths

        labels = torch.tensor([[-100, 1, 2, -100]])
        # shift drops position 0; positions 1,2,3 -> labels 1,2,-100 -> 2 supervised
        assert int(sequence_lengths(labels).item()) == 2

    def test_npo_loss_finite_and_lower_when_policy_below_ref(self):
        from soup_cli.utils.unlearn_kernels import npo_loss

        policy = torch.tensor([-10.0])
        ref = torch.tensor([-5.0])
        loss_low = npo_loss(policy, ref, beta=0.1)
        loss_high = npo_loss(torch.tensor([-1.0]), ref, beta=0.1)
        assert torch.isfinite(loss_low)
        # Policy well below ref ⇒ smaller loss (more forgotten).
        assert loss_low.item() < loss_high.item()

    def test_npo_loss_bool_beta_rejected(self):
        from soup_cli.utils.unlearn_kernels import npo_loss

        with pytest.raises(TypeError):
            npo_loss(torch.tensor([0.0]), torch.tensor([0.0]), beta=True)

    def test_npo_loss_nonpositive_beta_rejected(self):
        from soup_cli.utils.unlearn_kernels import npo_loss

        with pytest.raises(ValueError):
            npo_loss(torch.tensor([0.0]), torch.tensor([0.0]), beta=0.0)

    def test_simnpo_loss_finite(self):
        from soup_cli.utils.unlearn_kernels import simnpo_loss

        policy = torch.tensor([-8.0])
        lengths = torch.tensor([4])
        loss = simnpo_loss(policy, lengths, beta=0.1, gamma=0.0)
        assert torch.isfinite(loss)

    def test_simnpo_gamma_bool_rejected(self):
        from soup_cli.utils.unlearn_kernels import simnpo_loss

        with pytest.raises(TypeError):
            simnpo_loss(torch.tensor([-1.0]), torch.tensor([2]), gamma=True)

    def test_rmu_loss_finite(self):
        from soup_cli.utils.unlearn_kernels import rmu_loss

        fa = torch.zeros(8)
        cv = torch.ones(8)
        ra = torch.zeros(8)
        rf = torch.zeros(8)
        loss = rmu_loss(fa, cv, ra, rf, alpha=1.0)
        # forget term = mean((0-1)^2) = 1.0; retain term = 0.
        assert abs(loss.item() - 1.0) < 1e-5

    def test_rmu_loss_lower_when_acts_approach_control(self):
        # Review H2 — the RMU forget term must shrink as forget_acts steer
        # toward the control vector (lower loss = better steering).
        from soup_cli.utils.unlearn_kernels import rmu_loss

        cv = torch.ones(8)
        rf = torch.zeros(8)
        far = rmu_loss(torch.zeros(8), cv, torch.zeros(8), rf, alpha=1.0)
        close = rmu_loss(cv.clone(), cv, torch.zeros(8), rf, alpha=1.0)
        assert close.item() < far.item()

    def test_rmu_alpha_zero_accepted(self):
        from soup_cli.utils.unlearn_kernels import rmu_loss

        loss = rmu_loss(torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4), alpha=0.0)
        # alpha=0 drops the retain term entirely → forget term only (mean(1)=1).
        assert abs(loss.item() - 1.0) < 1e-5

    def test_rmu_alpha_negative_rejected(self):
        from soup_cli.utils.unlearn_kernels import rmu_loss

        with pytest.raises(ValueError):
            rmu_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2), alpha=-1.0)

    def test_npo_loss_batch_gt_one(self):
        from soup_cli.utils.unlearn_kernels import npo_loss

        loss = npo_loss(torch.tensor([-3.0, -8.0]), torch.tensor([-2.0, -2.0]), beta=0.1)
        assert torch.isfinite(loss)

    def test_simnpo_zero_length_clamped(self):
        from soup_cli.utils.unlearn_kernels import simnpo_loss

        # lengths=0 must NOT produce inf/nan (kernel clamps to 1).
        loss = simnpo_loss(torch.tensor([-4.0]), torch.tensor([0]), beta=0.1)
        assert torch.isfinite(loss)

    def test_simnpo_gamma_inf_rejected(self):
        from soup_cli.utils.unlearn_kernels import simnpo_loss

        with pytest.raises(ValueError):
            simnpo_loss(torch.tensor([-1.0]), torch.tensor([2]), gamma=float("inf"))

    def test_apply_unlearn_loss_returns_kernel(self):
        from soup_cli.utils import unlearn_kernels
        from soup_cli.utils.unlearning import apply_unlearn_loss

        assert apply_unlearn_loss("npo") is unlearn_kernels.npo_loss
        assert apply_unlearn_loss("SIMNPO") is unlearn_kernels.simnpo_loss
        assert apply_unlearn_loss("rmu") is unlearn_kernels.rmu_loss

    def test_apply_unlearn_loss_unknown_rejected(self):
        from soup_cli.utils.unlearning import apply_unlearn_loss

        with pytest.raises(ValueError):
            apply_unlearn_loss("zzz")


class TestUnlearnRowLoader:
    def test_extract_messages(self):
        from soup_cli.trainer.unlearn import _extract_pair

        row = {"messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]}
        assert _extract_pair(row) == ("Q", "A")

    def test_extract_prompt_completion(self):
        from soup_cli.trainer.unlearn import _extract_pair

        assert _extract_pair({"prompt": "p", "completion": "c"}) == ("p", "c")

    def test_extract_text(self):
        from soup_cli.trainer.unlearn import _extract_pair

        assert _extract_pair({"text": "hi"}) == ("", "hi")

    def test_extract_malformed_returns_none(self):
        from soup_cli.trainer.unlearn import _extract_pair

        assert _extract_pair("not-a-dict") is None
        assert _extract_pair({"prompt": "p"}) is None  # no target

    def test_load_rows_jsonl(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _load_unlearn_rows

        monkeypatch.chdir(tmp_path)
        f = tmp_path / "forget.jsonl"
        f.write_text(
            json.dumps({"prompt": "p1", "completion": "c1"}) + "\n"
            + "garbage\n"
            + json.dumps({"text": "t2"}) + "\n",
            encoding="utf-8",
        )
        rows = _load_unlearn_rows("forget.jsonl")
        assert rows == [("p1", "c1"), ("", "t2")]

    def test_load_rows_outside_cwd_rejected(self):
        from soup_cli.trainer.unlearn import _load_unlearn_rows

        with pytest.raises(ValueError, match="cwd"):
            _load_unlearn_rows("/etc/passwd")


class TestUnlearnTrainerWrapper:
    def _cfg(self, method="npo"):
        from soup_cli.config.schema import SoupConfig

        return SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="unlearn",
            data={"train": "t.jsonl", "forget_set": "f.jsonl"},
            training={"unlearn_method": method},
        )

    def test_train_before_setup_raises(self):
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        wrapper = UnlearnTrainerWrapper(self._cfg())
        with pytest.raises(RuntimeError, match="setup"):
            wrapper.train()

    def test_method_attribute(self):
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        assert UnlearnTrainerWrapper(self._cfg("simnpo")).method == "simnpo"

    def test_setup_loads_model(self, monkeypatch):
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        wrapper = UnlearnTrainerWrapper(self._cfg())
        import soup_cli.utils.live_eval as live_eval

        def _boom(*a, **k):
            raise RuntimeError("LOAD")

        monkeypatch.setattr(live_eval, "load_model_and_tokenizer", _boom)
        with pytest.raises(RuntimeError, match="LOAD"):
            wrapper.setup()


# ===========================================================================
# #194 — ROME / MEMIT / AlphaEdit edit kernels
# ===========================================================================


def _fake_llama(hidden=8, inter=16, layers=4):
    """Build a tiny Llama-shaped nn.Module with mlp.down_proj per layer."""
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(inter, hidden, bias=False)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(layers)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    return _Model()


class TestEditKernelHelpers:
    def test_locate_layers(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        model = _fake_llama()
        layers = _locate_decoder_layers(model)
        assert len(layers) == 4

    def test_locate_layers_missing_raises(self):
        import torch.nn as nn

        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        with pytest.raises(ValueError, match="decoder layers"):
            _locate_decoder_layers(nn.Linear(2, 2))

    def test_down_proj(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers

        model = _fake_llama()
        down = _down_proj(_locate_decoder_layers(model), 2)
        assert down.weight.shape == (8, 16)

    def test_down_proj_out_of_range(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers

        layers = _locate_decoder_layers(_fake_llama())
        with pytest.raises(ValueError, match="out of range"):
            _down_proj(layers, 99)

    def test_rank1_update_makes_key_map_to_value(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers, _rank1_update

        model = _fake_llama()
        down = _down_proj(_locate_decoder_layers(model), 0)
        key = torch.ones(16)
        delta = torch.full((8,), 0.5)
        before = down.weight @ key
        norm = _rank1_update(down, key, delta)
        after = down.weight @ key
        assert norm > 0
        assert torch.allclose(after - before, delta, atol=1e-4)

    def test_rank1_update_zero_key_rejected(self):
        from soup_cli.utils.edit_kernels import _down_proj, _locate_decoder_layers, _rank1_update

        down = _down_proj(_locate_decoder_layers(_fake_llama()), 0)
        with pytest.raises(ValueError, match="zero norm"):
            _rank1_update(down, torch.zeros(16), torch.ones(8))

    def test_run_edit_kernel_unknown_method(self):
        from soup_cli.utils.edit_kernels import run_edit_kernel

        with pytest.raises(ValueError, match="does not handle"):
            run_edit_kernel(
                _fake_llama(), None, method="zzz", subject="s", target="t",
                layer=0, device="cpu",
            )

    def test_edit_kernel_result_frozen(self):
        from soup_cli.utils.edit_kernels import EditKernelResult

        r = EditKernelResult(method="rome", layer=1, norm_delta=0.1, layers_edited=(1,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.method = "memit"  # type: ignore


class TestEditResult:
    def test_frozen(self):
        from soup_cli.utils.knowledge_edit import EditResult

        r = EditResult(
            method="rome", layer=5, norm_delta=0.1, layers_edited=(5,),
            output_dir=None, target_prob_before=0.0, target_prob_after=0.5,
            governed=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.method = "x"  # type: ignore

    def test_apply_edit_saves_when_output_dir(self, tmp_path, monkeypatch):
        import soup_cli.utils.edit_kernels as ek
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        monkeypatch.chdir(tmp_path)
        saved = {}

        class _Model:
            def save_pretrained(self, d):
                saved["model"] = d

        class _Tok:
            def save_pretrained(self, d):
                saved["tok"] = d

        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (_Model(), _Tok(), "cpu"),
        )
        monkeypatch.setattr(ek, "measure_target_prob", lambda *a, **k: 0.0)
        monkeypatch.setattr(
            ek, "run_edit_kernel",
            lambda *a, **k: ek.EditKernelResult(
                method="rome", layer=5, norm_delta=0.2, layers_edited=(5,),
            ),
        )
        plan = build_edit_plan(base="b", method="rome", subject="s", target="t")
        result = apply_edit(plan, output_dir="edited")
        assert result.output_dir == "edited"
        assert saved["model"] == "edited"

    def test_save_outside_cwd_rejected(self, tmp_path, monkeypatch):
        import soup_cli.utils.edit_kernels as ek
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (object(), object(), "cpu"),
        )
        monkeypatch.setattr(ek, "measure_target_prob", lambda *a, **k: 0.0)
        monkeypatch.setattr(
            ek, "run_edit_kernel",
            lambda *a, **k: ek.EditKernelResult(
                method="rome", layer=5, norm_delta=0.2, layers_edited=(5,),
            ),
        )
        plan = build_edit_plan(base="b", method="rome", subject="s", target="t")
        with pytest.raises(ValueError, match="cwd"):
            apply_edit(plan, output_dir="/etc/evil")


class TestEditDiffLive:
    def test_live_generation_detects_change(self, monkeypatch, tmp_path):
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.edit_diff import build_diff_report

        monkeypatch.chdir(tmp_path)
        probes = tmp_path / "p.jsonl"
        probes.write_text(
            json.dumps({"prompt": "The capital of France is"}) + "\n"
            + json.dumps({"prompt": "2 + 2 ="}) + "\n",
            encoding="utf-8",
        )

        def _fake_make_gen(model_id, **kwargs):
            if model_id == "before":
                return lambda p: "Paris" if "France" in p else "4"
            return lambda p: "Lyon" if "France" in p else "4"

        monkeypatch.setattr(live_eval, "make_generator", _fake_make_gen)
        report = build_diff_report(
            before_run_id="r1",
            after_run_id="r2",
            probe_file="p.jsonl",
            before_model="before",
            after_model="after",
        )
        # Changed row (France) first.
        assert report.changes[0].changed is True
        assert report.changes[0].before == "Paris"
        assert report.changes[0].after == "Lyon"
        n_changed = sum(1 for c in report.changes if c.changed)
        assert n_changed == 1

    def test_placeholder_without_models(self, monkeypatch, tmp_path):
        from soup_cli.utils.edit_diff import build_diff_report

        monkeypatch.chdir(tmp_path)
        probes = tmp_path / "p.jsonl"
        probes.write_text(json.dumps({"prompt": "x"}) + "\n", encoding="utf-8")
        report = build_diff_report(
            before_run_id="r1", after_run_id="r2", probe_file="p.jsonl",
        )
        assert report.changes[0].changed is False
        assert "supply" in report.changes[0].before.lower()


# ===========================================================================
# #203 — GRACE codebook live (lookup / write + Registry persistence)
# ===========================================================================


class TestGraceCodebook:
    def _cfg(self, dim=4, size=10):
        from soup_cli.utils.grace_codebook import GraceCodebookConfig

        return GraceCodebookConfig(size=size, dim=dim)

    def test_add_and_lookup_hit(self):
        from soup_cli.utils.grace_codebook import GraceCodebook

        cb = GraceCodebook(self._cfg(), epsilon=0.5)
        cb.add([1.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0], "fact")
        assert len(cb) == 1
        hit = cb.lookup([1.0, 0.0, 0.0, 0.0])
        assert hit == [9.0, 9.0, 9.0, 9.0]

    def test_lookup_miss_outside_epsilon(self):
        from soup_cli.utils.grace_codebook import GraceCodebook

        cb = GraceCodebook(self._cfg(), epsilon=0.1)
        cb.add([1.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0], "fact")
        assert cb.lookup([5.0, 5.0, 5.0, 5.0]) is None

    def test_lookup_empty_returns_none(self):
        from soup_cli.utils.grace_codebook import GraceCodebook

        cb = GraceCodebook(self._cfg())
        assert cb.lookup([0.0, 0.0, 0.0, 0.0]) is None

    def test_add_dim_mismatch(self):
        from soup_cli.utils.grace_codebook import GraceCodebook

        cb = GraceCodebook(self._cfg())
        with pytest.raises(ValueError, match="dim"):
            cb.add([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], "x")

    def test_add_full_rejected(self):
        from soup_cli.utils.grace_codebook import GraceCodebook

        cb = GraceCodebook(self._cfg(size=1))
        cb.add([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "a")
        with pytest.raises(ValueError, match="full"):
            cb.add([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "b")

    def test_epsilon_validation(self):
        from soup_cli.utils.grace_codebook import GraceCodebook

        with pytest.raises(ValueError):
            GraceCodebook(self._cfg(), epsilon=0.0)
        with pytest.raises(TypeError):
            GraceCodebook(self._cfg(), epsilon=True)

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        from soup_cli.utils.grace_codebook import (
            GraceCodebook,
            load_codebook,
            save_codebook,
        )

        monkeypatch.chdir(tmp_path)
        cb = GraceCodebook(self._cfg(), epsilon=0.7, layer=3, base_model="m")
        cb.add([1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0], "fact")
        out = tmp_path / "ckpt"
        out.mkdir()
        save_codebook(cb, "ckpt")
        restored = load_codebook("ckpt")
        assert len(restored) == 1
        assert restored.layer == 3
        assert restored.base_model == "m"
        assert restored.lookup([1.0, 0.0, 0.0, 0.0]) == [2.0, 2.0, 2.0, 2.0]

    def test_save_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.grace_codebook import GraceCodebook, save_codebook

        monkeypatch.chdir(tmp_path)
        cb = GraceCodebook(self._cfg())
        with pytest.raises(ValueError, match="cwd"):
            save_codebook(cb, "/etc/evil")

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        from soup_cli.utils.grace_codebook import load_codebook

        monkeypatch.chdir(tmp_path)
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            load_codebook("empty")

    def test_apply_grace_codebook_returns_empty(self):
        from soup_cli.utils.grace_codebook import (
            GraceCodebook,
            apply_grace_codebook,
            build_grace_codebook_config,
        )

        cb = apply_grace_codebook(build_grace_codebook_config(size=10, dim=4))
        assert isinstance(cb, GraceCodebook)
        assert len(cb) == 0


class TestRegistryKinds:
    def test_edit_kinds_registered(self):
        from soup_cli.registry.store import _VALID_KINDS

        assert "edited_model" in _VALID_KINDS
        assert "grace_codebook" in _VALID_KINDS


# ===========================================================================
# CLI plumbing
# ===========================================================================


class TestCli:
    def test_edit_set_help_has_new_flags(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["edit", "set", "--help"])
        assert result.exit_code == 0, result.output
        out = result.output
        # Rich may wrap; strip ANSI + check tokens.
        import re

        clean = re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert "--output" in clean
        assert "--governor" in clean

    def test_edit_diff_help_has_model_flags(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["edit", "diff", "--help"])
        assert result.exit_code == 0, result.output
        import re

        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "before-model" in clean
        assert "after-model" in clean


# ===========================================================================
# Review follow-ups
# ===========================================================================


class TestReviewFollowups:
    def test_validated_output_dir_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _validated_output_dir

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="cwd"):
            _validated_output_dir("/etc/evil")

    def test_validated_output_dir_null_byte(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _validated_output_dir

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="null"):
            _validated_output_dir("x\x00y")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_validated_output_dir_symlink(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _validated_output_dir

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real").mkdir()
        os.symlink(tmp_path / "real", tmp_path / "link")
        with pytest.raises(ValueError, match="symlink"):
            _validated_output_dir("link")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_load_unlearn_rows_symlink_rejected(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _load_unlearn_rows

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.jsonl"
        target.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        os.symlink(target, tmp_path / "link.jsonl")
        with pytest.raises(ValueError, match="symlink"):
            _load_unlearn_rows("link.jsonl")

    def test_load_unlearn_rows_null_byte(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _load_unlearn_rows

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="null"):
            _load_unlearn_rows("a\x00b.jsonl")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_save_edited_model_symlink_rejected(self, tmp_path, monkeypatch):
        import soup_cli.utils.edit_kernels as ek
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real").mkdir()
        os.symlink(tmp_path / "real", tmp_path / "out")
        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: (object(), object(), "cpu"),
        )
        monkeypatch.setattr(ek, "measure_target_prob", lambda *a, **k: 0.0)
        monkeypatch.setattr(
            ek, "run_edit_kernel",
            lambda *a, **k: ek.EditKernelResult(
                method="rome", layer=5, norm_delta=0.1, layers_edited=(5,),
            ),
        )
        plan = build_edit_plan(base="b", method="rome", subject="s", target="t")
        with pytest.raises(ValueError, match="symlink"):
            apply_edit(plan, output_dir="out")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_load_codebook_symlink_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.grace_codebook import load_codebook

        monkeypatch.chdir(tmp_path)
        (tmp_path / "ckpt").mkdir()
        real = tmp_path / "real.json"
        real.write_text("{}", encoding="utf-8")
        os.symlink(real, tmp_path / "ckpt" / "grace_codebook.json")
        with pytest.raises(ValueError, match="symlink"):
            load_codebook("ckpt")

    def test_alphaedit_projection_deterministic(self):
        import soup_cli.utils.edit_kernels as ek

        model = _fake_llama()
        down = ek._down_proj(ek._locate_decoder_layers(model), 0)
        upd = torch.full((8, 16), 0.3)
        p1 = ek._alphaedit_project(down, upd)
        p2 = ek._alphaedit_project(down, upd)
        assert torch.allclose(p1, p2)

    def test_classify_norm_blowup_boundaries(self):
        from soup_cli.utils.edit_governor import NormBlowupPolicy, classify_norm_blowup

        pol = NormBlowupPolicy(warn_threshold=1.0, blowup_threshold=5.0)
        assert classify_norm_blowup(0.99, pol) == "OK"
        assert classify_norm_blowup(1.0, pol) == "WARN"  # >= warn
        assert classify_norm_blowup(4.99, pol) == "WARN"
        assert classify_norm_blowup(5.0, pol) == "BLOWUP"  # >= blowup

    def test_apply_grace_edit_governed_refusal(self, monkeypatch):
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.edit_governor import EditGovernor, GovernedEditError
        from soup_cli.utils.grace_codebook import apply_grace_edit
        from soup_cli.utils.knowledge_edit import build_edit_plan

        loaded = {"n": 0}

        def _load(*a, **k):
            loaded["n"] += 1
            return ("M", "T", "cpu")

        monkeypatch.setattr(live_eval, "load_model_and_tokenizer", _load)
        gov = EditGovernor(base_model="b", max_sequential_edits=1)
        gov.record_edit(method="grace", norm_delta=0.1)
        plan = build_edit_plan(base="b", method="grace", subject="s", target="t")
        with pytest.raises(GovernedEditError):
            apply_grace_edit(plan, governor=gov)
        assert loaded["n"] == 0  # refused before any model load


# ===========================================================================
# Patch invariants
# ===========================================================================


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = soup_cli.__version__.split(".")
        assert (int(parts[0]), int(parts[1]), int(parts[2])) >= (0, 71, 9)

    @pytest.mark.parametrize(
        "module",
        [
            "soup_cli.utils.edit_kernels",
            "soup_cli.utils.unlearn_kernels",
            "soup_cli.utils.edit_governor",
            "soup_cli.utils.grace_codebook",
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
