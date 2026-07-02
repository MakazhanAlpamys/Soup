"""Regression tests for the HIGH findings in CODE_REVIEW.md.

Grouped by the review's own subsections. Each asserts the specific broken
behavior is now correct.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import soup_cli

# ═══════════════════════ HIGH — training correctness ═══════════════════════


def test_ppo_refuses_silent_random_reward_head():
    """PPO with only a reward_fn (no reward_model) must fail loudly, not build
    a randomly-initialised reward head and train the policy against noise."""
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    wrapper = object.__new__(PPOTrainerWrapper)
    wrapper.reward_model_instance = None
    wrapper.device = "cpu"
    wrapper.trust_remote_code = False
    wrapper._trust_remote_code = False
    cfg = types.SimpleNamespace(base="hf-internal-testing/tiny-random-gpt2")
    tcfg = types.SimpleNamespace(reward_model=None)
    with pytest.raises(RuntimeError, match="reward_model"):
        wrapper._get_or_create_reward_model(cfg, tcfg)


def test_alphaedit_rejects_nonfinite_denominator(monkeypatch):
    """A NaN key-norm must be rejected (NaN <= 0.0 is False, so the old bare
    `denom <= 0.0` guard let it corrupt weights in place)."""
    torch = pytest.importorskip("torch")
    import soup_cli.utils.edit_kernels as ek

    class _Down:
        def __init__(self):
            self.weight = torch.zeros(2, 2)

    monkeypatch.setattr(ek, "_locate_decoder_layers", lambda m: ["layer0"])
    monkeypatch.setattr(ek, "_down_proj", lambda layers, layer: _Down())
    monkeypatch.setattr(
        ek, "_capture_key", lambda *a, **k: torch.tensor([float("nan"), float("nan")])
    )
    monkeypatch.setattr(ek, "_optimise_residual", lambda *a, **k: torch.tensor([1.0, 1.0]))

    with pytest.raises(ValueError, match="zero norm"):
        ek.apply_alphaedit_edit(
            object(), object(), subject="s", target="t", layer=0, device="cpu"
        )


def test_orpo_length_normalization_restores_odds_ratio():
    """With summed sequence log-probs exp() underflows and the odds-ratio
    correction collapses; passing lengths length-normalises and restores it."""
    torch = pytest.importorskip("torch")
    from soup_cli.utils.preference_combine import compute_orpo_term

    pol_chosen = torch.tensor([-40.0])
    pol_rejected = torch.tensor([-50.0])
    lens = torch.tensor([20.0])

    degenerate = float(compute_orpo_term(pol_chosen, pol_rejected, 1.0))
    normalized = float(
        compute_orpo_term(
            pol_chosen, pol_rejected, 1.0, chosen_lens=lens, rejected_lens=lens
        )
    )
    import math as _math

    assert _math.isfinite(normalized)
    # The summed-logp version underflows to ~0 loss; the length-normalised
    # version has a meaningful (larger) odds-ratio loss.
    assert normalized - degenerate > 1e-2


def test_ipo_beta_schedule_uses_ipo_tau_not_dpo_beta():
    src = (Path(soup_cli.__file__).parent / "trainer" / "ipo.py").read_text(
        encoding="utf-8"
    )
    assert "beta_start=tcfg.ipo_tau" in src
    assert "beta_start=tcfg.dpo_beta" not in src


def test_beta_schedule_callback_sets_beta_from_start():
    """At step 0 the schedule must apply beta_start — so passing ipo_tau (not
    the DPO default) is what keeps the user's τ intact."""
    from soup_cli.utils.dpo_variants import BetaScheduleCallback

    trainer = types.SimpleNamespace(beta=0.5)  # e.g. the user's ipo_tau
    cb = BetaScheduleCallback(beta_start=0.5, beta_end=0.1, total_steps=10, schedule="linear")
    cb.attach(trainer)
    state = types.SimpleNamespace(max_steps=10, global_step=0)
    cb.on_train_begin(None, state, None)
    cb.on_step_begin(None, state, None)
    assert abs(trainer.beta - 0.5) < 1e-9


def test_distill_term_masks_padding_and_prompt():
    """The default distillation divergence must be measured only over trained
    tokens (labels != -100), not padding/prompt."""
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import _compute_distill_term

    student = torch.zeros(1, 3, 4)
    teacher = torch.zeros(1, 3, 4)
    # Positions 0,1 diverge wildly; position 2 is identical (KL 0).
    student[0, 0, 0] = 10.0
    teacher[0, 0, 3] = 10.0
    student[0, 1, 1] = 10.0
    teacher[0, 1, 2] = 10.0
    labels = torch.tensor([[-100, -100, 5]])  # only the last token is trained

    masked = float(_compute_distill_term(student, teacher, "forward_kl", 1.0, labels=labels))
    unmasked = float(_compute_distill_term(student, teacher, "forward_kl", 1.0))
    assert masked < 1e-5, masked  # only the identical (trained) position counts
    assert unmasked > 0.1, unmasked  # divergent prompt positions dominate unmasked


def test_kto_negative_one_label_is_undesirable():
    """`bool(-1)` is True; a -1 label in the ±1 convention must map to False."""
    from soup_cli.data.formats import _convert_kto

    base = {"prompt": "p", "completion": "c"}
    assert _convert_kto({**base, "label": -1})["label"] is False
    assert _convert_kto({**base, "label": 1})["label"] is True
    assert _convert_kto({**base, "label": 0})["label"] is False
    assert _convert_kto({**base, "label": True})["label"] is True
    assert _convert_kto({**base, "label": False})["label"] is False
    assert _convert_kto({**base, "label": "false"})["label"] is False


def test_apply_llama_pro_freeze_freezes_all_but_new_blocks():
    pytest.importorskip("torch")
    import torch.nn as nn

    from soup_cli.utils.block_expansion import apply_llama_pro_freeze

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])

    model = _M()
    apply_llama_pro_freeze(model, 2)  # keep only the last 2 blocks trainable
    trainable = [
        all(p.requires_grad for p in layer.parameters()) for layer in model.model.layers
    ]
    assert trainable == [False, False, True, True]


def test_block_expansion_freeze_uses_actual_added_not_requested(monkeypatch):
    """When expand_layers over-requests, the freeze must target the ACTUAL
    appended count (clamped), else original layers stay trainable."""
    import soup_cli.utils.block_expansion as be

    class _Inner:
        def __init__(self, n):
            self.layers = list(range(n))

    class _Model:
        def __init__(self, n):
            self.model = _Inner(n)

    model = _Model(4)  # 4 base layers

    def fake_expand(m, n):
        added = min(n, 4)  # expand_model_blocks clamps to available base blocks
        m.model.layers = list(range(4 + added))
        return 4 + added

    captured = {}

    def fake_freeze(m, count):
        captured["count"] = count
        return 123

    monkeypatch.setattr(be, "expand_model_blocks", fake_expand)
    monkeypatch.setattr(be, "apply_llama_pro_freeze", fake_freeze)

    tcfg = types.SimpleNamespace(expand_layers=10, freeze_trainable_layers=1)
    be.apply_block_expansion_if_configured(model, tcfg)
    assert captured["count"] == 4  # actual added, NOT the over-requested 10


def _src(rel: str) -> str:
    return (Path(soup_cli.__file__).parent / rel).read_text(encoding="utf-8")


# ═══════════════════════ HIGH — features that silently do nothing ═══════════


def test_longlora_wired_into_sft_train():
    src = _src("trainer/sft.py")
    assert "apply_longlora_forward_override" in src, "LongLoRA override never installed"


def test_gpus_reexec_passes_run_shaping_flags():
    src = _src("commands/train.py")
    for flag in ('"--gate"', '"--push-as"', '"--trust-remote-code"', '"--tracker"',
                 '"--diagnose-gate"', '"--annex-xi"', '"--repro-receipt"'):
        assert "script_args" in src and flag in src, f"re-exec drops {flag}"


def test_pre_push_hook_enforces_gate_suite():
    from soup_cli.utils.eval_gate_hook import render_pre_push_hook

    hook = render_pre_push_hook(baseline_run_id="run-abc123", suite_path="evals/locked.json")
    assert '--suite "$GATE_SUITE"' in hook


def test_eval_against_blocks_on_unloadable_locked_suite(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.commands.eval import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["against", "base-run", "--candidate", "cand-run", "--suite", "missing.json"],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))


def test_deploy_measure_cache_key_includes_candidates():
    from soup_cli.utils.deploy_measure import compute_cache_key

    common = dict(base_sha="a" * 16, profile_name="p", tasks_sha="b" * 64)
    k_one = compute_cache_key(**common, candidates=["4bit"])
    k_two = compute_cache_key(**common, candidates=["4bit", "8bit"])
    k_none = compute_cache_key(**common)  # back-compat: same as pre-fix key
    assert k_one != k_two, "different candidate sets must not share a cache key"
    assert k_none not in (k_one, k_two)


# ═══════════════════════ HIGH — security ═══════════════════════


def test_sglang_cors_is_loopback_only():
    src = _src("utils/sglang.py")
    assert 'allow_origins=["*"]' not in src
    assert "localhost|127" in src  # loopback-only regex


def test_fetch_lstats_original_path_not_realpath():
    src = _src("commands/fetch.py")
    assert "os.lstat(target_path)" in src
    assert "os.lstat(real_target)" not in src


def test_ui_inspect_uses_commonpath_containment():
    src = _src("ui/app.py")
    assert "is_under_cwd(req.path)" in src


def test_is_under_cwd_rejects_sibling_prefix(tmp_path, monkeypatch):
    from soup_cli.utils.paths import is_under_cwd

    proj = tmp_path / "project"
    proj.mkdir()
    (tmp_path / "project-secrets").mkdir()
    monkeypatch.chdir(proj)
    assert is_under_cwd(str(proj / "d.jsonl")) is True
    # The sibling shares the "project" prefix but is NOT under cwd.
    assert is_under_cwd(str(tmp_path / "project-secrets" / "d.jsonl")) is False


def test_registry_lineage_cycle_detected_beyond_depth_10(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    from soup_cli.registry.store import RegistryStore

    with RegistryStore() as store:
        ids = [
            store.push(
                name=f"m{i}", tag="v1", base_model="b", task="sft",
                run_id=None, config={},
            )
            for i in range(15)
        ]
        # Chain child->parent 14 deep: ids[0] -> ids[1] -> ... -> ids[14].
        for i in range(14):
            store.add_lineage(
                child_id=ids[i], parent_id=ids[i + 1], relation="forked_from"
            )
        # ids[14] is an ancestor of ids[0] 14 hops away — past the old depth-10
        # cap. The unbounded walk must still catch the cycle.
        with pytest.raises(ValueError, match="cycle"):
            store.add_lineage(
                child_id=ids[14], parent_id=ids[0], relation="forked_from"
            )


def test_gguf_calib_reads_from_nofollow_fd_no_reopen():
    src = _src("utils/gguf_quant.py")
    assert "os.fdopen(fd" in src
    assert "os.close(fd)" not in src  # the close+reopen TOCTOU window is gone


def test_namespace_created_at_differs_both_directions():
    from soup_cli.utils.namespace_pin import _created_at_differs

    assert _created_at_differs("2026-01-02T00:00:00", "2026-01-01T00:00:00") is True
    assert _created_at_differs("2026-01-01T00:00:00", "2026-01-02T00:00:00") is True
    assert _created_at_differs("2026-01-01T00:00:00", "2026-01-01T00:00:00") is False


def test_namespace_pin_flags_forward_created_at_drift(tmp_path):
    from soup_cli.utils.namespace_pin import NamespacePinStore, verify_namespace

    store = NamespacePinStore(str(tmp_path / "pins.db"))
    first = verify_namespace(
        store, repo_id="org/model", current_author="alice",
        current_created_at="2026-01-01T00:00:00",
    )
    assert first.ok is True  # trust on first use
    # Same author, LATER created_at (repo re-created / AI-Jacking) must be flagged.
    recreated = verify_namespace(
        store, repo_id="org/model", current_author="alice",
        current_created_at="2026-06-01T00:00:00",
    )
    assert recreated.ok is False


# ═══════════════════════ HIGH — robustness / cross-platform ═══════════════════


def test_typer_exit_is_a_runtimeerror():
    import typer

    # This is WHY the bench.py `except RuntimeError` swallowed it and the
    # eval.py `except SystemExit` missed it.
    assert issubclass(typer.Exit, RuntimeError)
    assert not issubclass(typer.Exit, SystemExit)


def test_bench_reraises_typer_exit():
    assert "except typer.Exit:" in _src("commands/bench.py")


def test_eval_auto_catches_typer_exit():
    assert "except (typer.Exit, SystemExit):" in _src("commands/eval.py")


def test_data_split_rejects_negative_val(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "x"}]}\n' * 20, encoding="utf-8"
    )
    result = CliRunner().invoke(
        app, ["split", "d.jsonl", "--val", "-10", "--absolute"]
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "non-negative" in result.output


def test_trace_parser_reads_bom_first_record(tmp_path):
    from soup_cli.data.traces.parsers import parse_soup_serve

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    # utf-8-sig writes a BOM before line 1.
    (trace_dir / "t.jsonl").write_text(
        '{"prompt": "p1", "response": "r1"}\n{"prompt": "p2", "response": "r2"}\n',
        encoding="utf-8-sig",
    )
    traces = list(parse_soup_serve(str(trace_dir)))
    assert len(traces) == 2, "BOM dropped the first record"
    assert traces[0].prompt == "p1"


def test_plan_estimate_handles_batch_size_auto():
    from soup_cli.utils.terraform_plan import _estimate_runtime_minutes

    minutes = _estimate_runtime_minutes(
        {"training": {"epochs": 2, "batch_size": "auto"}}
    )
    assert minutes > 0  # float("auto") no longer crashes


def test_rl_checkpoint_only_main_process_writes(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    from soup_cli.utils.rl_checkpoint import (
        RLCheckpointConfig,
        build_rl_checkpoint_callback,
    )

    cb = build_rl_checkpoint_callback(
        RLCheckpointConfig(save_every_steps=1), output_dir="run", task="grpo"
    )

    class _M:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)

    monkeypatch.setenv("RANK", "1")  # non-main rank
    path = cb.save_checkpoint(step=1, model=_M(), optimizer=None)
    assert not os.path.exists(path), "non-main rank must not write the checkpoint"

    monkeypatch.setenv("RANK", "0")  # main rank
    path2 = cb.save_checkpoint(step=2, model=_M(), optimizer=None)
    assert os.path.exists(os.path.join(path2, "manifest.json"))
