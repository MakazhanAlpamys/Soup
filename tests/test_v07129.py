"""v0.71.29 — `soup shrink`: depth-prune + distill-heal (arXiv:2403.17887).

Tests the pure verdict half, the torch-lazy prune/importance half, the CLI
orchestration, the subprocess distill-heal wiring, and registry attach.
"""
import ast
import math
import pathlib
from io import StringIO

import pytest
from rich.console import Console

from soup_cli.utils.shrink import (
    DECISION_DONT_SHIP,
    DECISION_SHIP,
    LayerImportance,
    decide_shrink,
    render_shrink_panel,
    shrink_verdict_to_dict,
)


# ---------------------------------------------------------------------------
# Task 1 — pure verdict half
# ---------------------------------------------------------------------------
class TestDecideShrink:
    def test_within_tolerance_ships(self):
        v = decide_shrink(10.0, 10.5, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_SHIP
        assert math.isclose(v.ppl_ratio, 1.05)

    def test_exceeds_tolerance_dont_ship(self):
        v = decide_shrink(10.0, 12.0, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_DONT_SHIP

    def test_boundary_exactly_at_tolerance_ships(self):
        v = decide_shrink(10.0, 11.0, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_SHIP  # ratio-1 == tolerance -> <=, SHIP

    def test_improved_ppl_ships(self):
        v = decide_shrink(10.0, 9.5, tolerance=0.10, layers_before=30, layers_after=24)
        assert v.decision == DECISION_SHIP

    def test_rejects_nonpositive_ppl(self):
        with pytest.raises(ValueError):
            decide_shrink(0.0, 5.0, layers_before=30, layers_after=24)
        with pytest.raises(ValueError):
            decide_shrink(5.0, -1.0, layers_before=30, layers_after=24)

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            decide_shrink(10.0, float("inf"), layers_before=30, layers_after=24)
        with pytest.raises(ValueError):
            decide_shrink(float("nan"), 5.0, layers_before=30, layers_after=24)

    def test_rejects_bool_ppl(self):
        with pytest.raises(ValueError):
            decide_shrink(True, 5.0, layers_before=30, layers_after=24)

    def test_rejects_bad_tolerance(self):
        with pytest.raises(ValueError):
            decide_shrink(10.0, 10.0, tolerance=-0.1, layers_before=30, layers_after=24)
        with pytest.raises(ValueError):
            decide_shrink(10.0, 10.0, tolerance=6.0, layers_before=30, layers_after=24)
        with pytest.raises(ValueError):
            decide_shrink(10.0, 10.0, tolerance=True, layers_before=30, layers_after=24)

    def test_frozen(self):
        v = decide_shrink(10.0, 10.5, layers_before=30, layers_after=24)
        with pytest.raises(Exception):
            v.decision = "x"  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        v = decide_shrink(
            10.0, 10.5, layers_before=30, layers_after=24, params_saved_pct=20.0, healed=True
        )
        d = shrink_verdict_to_dict(v)
        assert d["decision"] == v.decision
        assert d["healed"] is True
        assert set(d) >= {
            "decision", "ppl_original", "ppl_final", "ppl_ratio", "tolerance",
            "layers_before", "layers_after", "params_saved_pct", "healed", "soup_version",
        }

    def test_render_panel_names_decision(self):
        v = decide_shrink(10.0, 12.0, layers_before=30, layers_after=24)
        buf = StringIO()
        Console(file=buf, width=100).print(render_shrink_panel(v))
        assert "DON'T SHIP" in buf.getvalue()

    def test_render_panel_ship(self):
        v = decide_shrink(10.0, 10.2, layers_before=30, layers_after=24)
        buf = StringIO()
        Console(file=buf, width=100).print(render_shrink_panel(v))
        out = buf.getvalue()
        assert "SHIP" in out and "DON'T SHIP" not in out

    def test_layer_importance_frozen(self):
        li = LayerImportance(start=5, block_size=8, angular_distance=0.12)
        assert (li.start, li.block_size) == (5, 8)
        with pytest.raises(Exception):
            li.start = 1  # type: ignore[misc]


class TestNoTopLevelTorch:
    def test_shrink_module_has_no_top_level_heavy_import(self):
        src = pathlib.Path("src/soup_cli/utils/shrink.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert not any(
            m.split(".")[0] in {"torch", "transformers", "peft"} for m in names
        ), names


# ---------------------------------------------------------------------------
# Task 2 — arch allowlist + prune_model_layers (torch, tiny CPU model)
# ---------------------------------------------------------------------------
def _tiny_llama(layers: int = 6, vocab_size: int = 128):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab_size,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(cfg)


class TestPrune:
    def test_arch_detected(self):
        from soup_cli.utils.shrink import shrink_arch_of

        assert shrink_arch_of(_tiny_llama()) == "llama"

    def test_arch_rejects_unsupported(self):
        from soup_cli.utils.shrink import shrink_arch_of

        class _Cfg:
            model_type = "gpt_neox"
            architectures = ["GPTNeoXForCausalLM"]

        class _M:
            config = _Cfg()

        with pytest.raises(ValueError, match="supports"):
            shrink_arch_of(_M())

    def test_prune_removes_block_and_patches_config(self):
        from soup_cli.utils.shrink import prune_model_layers

        m = _tiny_llama(6)
        prune_model_layers(m, start=2, block_size=2)  # drop layers 2,3
        assert len(m.model.layers) == 4
        assert m.config.num_hidden_layers == 4

    def test_prune_rejects_touching_last_layer(self):
        from soup_cli.utils.shrink import prune_model_layers

        m = _tiny_llama(6)
        with pytest.raises(ValueError, match="protected"):
            prune_model_layers(m, start=4, block_size=2)  # would include last (idx 5)

    def test_prune_rejects_touching_first_layer(self):
        from soup_cli.utils.shrink import prune_model_layers

        m = _tiny_llama(6)
        with pytest.raises(ValueError, match="protected"):
            prune_model_layers(m, start=0, block_size=2)

    def test_prune_rejects_block_too_large(self):
        from soup_cli.utils.shrink import prune_model_layers

        m = _tiny_llama(6)
        with pytest.raises(ValueError, match="block_size"):
            prune_model_layers(m, start=1, block_size=6)

    def test_layer_list_arch_guarded(self):
        from soup_cli.utils.shrink import layer_list

        m = _tiny_llama(4)
        assert len(layer_list(m)) == 4


# ---------------------------------------------------------------------------
# Task 3 — importance scan (off-by-one pinned) + selection + drop count
# ---------------------------------------------------------------------------
class TestImportance:
    def test_off_by_one_boundary_indices(self):
        """Pin: block [L, L+n) uses hidden_states[L] and hidden_states[L+n];
        hidden_states has num_layers+1 entries. A fake model returns per-layer
        constant hidden states so the boundary maths is asserted, not just that
        the pass runs."""
        import torch

        from soup_cli.utils import shrink

        num_layers = 4
        # len must be num_layers+1; index 0 = embeddings, index k = layer k-1 out.
        hs = tuple(torch.ones(1, 3, 8) * (k + 1) for k in range(num_layers + 1))

        class _Cfg:
            model_type = "llama"
            architectures = ["LlamaForCausalLM"]
            num_hidden_layers = num_layers

        class _Out:
            hidden_states = hs

        class _Model:
            config = _Cfg()

            def eval(self):
                return self

            def __call__(self, **kw):
                assert kw.get("output_hidden_states") is True
                return _Out()

        class _Tok:
            def __call__(self, text, **kw):
                return {
                    "input_ids": torch.ones(1, 3, dtype=torch.long),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                }

        imps = shrink.compute_layer_importance(
            _Model(), _Tok(), ["hi"], block_size=1, device="cpu"
        )
        # valid starts for n=1, num_layers=4: L in [1, 4-1-1=2] -> {1, 2}
        starts = sorted(i.start for i in imps)
        assert starts == [1, 2]
        # constant (colinear) vectors -> cos == 1 -> angular distance 0.
        assert all(abs(i.angular_distance) < 1e-6 for i in imps)
        assert all(i.block_size == 1 for i in imps)

    def test_hidden_states_length_mismatch_raises(self):
        import torch

        from soup_cli.utils import shrink

        class _Cfg:
            model_type = "llama"
            architectures = ["LlamaForCausalLM"]
            num_hidden_layers = 4

        class _Out:
            hidden_states = tuple(torch.ones(1, 3, 8) for _ in range(3))  # wrong len

        class _Model:
            config = _Cfg()

            def eval(self):
                return self

            def __call__(self, **kw):
                return _Out()

        class _Tok:
            def __call__(self, text, **kw):
                return {
                    "input_ids": torch.ones(1, 3, dtype=torch.long),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                }

        with pytest.raises(ValueError, match="hidden states"):
            shrink.compute_layer_importance(
                _Model(), _Tok(), ["hi"], block_size=1, device="cpu"
            )

    def test_select_drop_block_min(self):
        from soup_cli.utils.shrink import LayerImportance, select_drop_block

        imps = [
            LayerImportance(1, 2, 0.9),
            LayerImportance(3, 2, 0.1),
            LayerImportance(5, 2, 0.5),
        ]
        chosen = select_drop_block(imps)
        assert chosen.start == 3 and chosen.angular_distance == 0.1

    def test_select_drop_block_empty_raises(self):
        from soup_cli.utils.shrink import select_drop_block

        with pytest.raises(ValueError):
            select_drop_block([])

    def test_resolve_drop_count_ratio(self):
        from soup_cli.utils.shrink import resolve_drop_count

        assert resolve_drop_count(30, drop_ratio=0.25, drop_layers=None) == 8  # round(7.5)
        assert resolve_drop_count(30, drop_ratio=None, drop_layers=6) == 6

    def test_resolve_drop_count_rejects_both_or_neither(self):
        from soup_cli.utils.shrink import resolve_drop_count

        with pytest.raises(ValueError, match="exactly one"):
            resolve_drop_count(30, drop_ratio=0.25, drop_layers=6)
        with pytest.raises(ValueError, match="exactly one"):
            resolve_drop_count(30, drop_ratio=None, drop_layers=None)

    def test_resolve_drop_count_position_bound(self):
        from soup_cli.utils.shrink import resolve_drop_count

        with pytest.raises(ValueError, match="range"):
            resolve_drop_count(4, drop_ratio=None, drop_layers=3)  # > num_layers-2

    def test_resolve_drop_count_ratio_bounds(self):
        from soup_cli.utils.shrink import resolve_drop_count

        with pytest.raises(ValueError):
            resolve_drop_count(30, drop_ratio=1.5, drop_layers=None)
        with pytest.raises(ValueError):
            resolve_drop_count(30, drop_ratio=0.0, drop_layers=None)

    def test_compute_importance_no_valid_starts_raises(self):
        import torch

        from soup_cli.utils import shrink

        class _Cfg:
            model_type = "llama"
            architectures = ["LlamaForCausalLM"]
            num_hidden_layers = 4

        class _Out:
            hidden_states = tuple(torch.ones(1, 3, 8) for _ in range(5))

        class _Model:
            config = _Cfg()

            def eval(self):
                return self

            def __call__(self, **kw):
                return _Out()

        class _Tok:
            def __call__(self, text, **kw):
                return {
                    "input_ids": torch.ones(1, 3, dtype=torch.long),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                }

        with pytest.raises(ValueError, match="position-valid"):
            shrink.compute_layer_importance(
                _Model(), _Tok(), ["hi"], block_size=3, device="cpu"
            )


# ---------------------------------------------------------------------------
# Task 4 — commands/shrink.py prune orchestration + CLI registration
# ---------------------------------------------------------------------------
def _write_tiny_model(dir_path, layers: int = 6):
    """Save a tiny CPU Llama + tokenizer to ``dir_path`` for CLI smoke."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    m = _tiny_llama(layers, vocab_size=len(tok))
    m.save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    return str(dir_path)


class TestShrinkCli:
    def test_registered_and_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        r = CliRunner().invoke(app, ["shrink", "--help"])
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "drop-ratio" in r.output
        assert "calib" in r.output

    def test_rejects_both_drop_flags(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        calib = tmp_path / "c.jsonl"
        calib.write_text('{"text":"hello world"}\n', encoding="utf-8")
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", "x", "--drop-ratio", "0.25", "--drop-layers",
             "2", "--calib", "c.jsonl"],
        )
        assert r.exit_code != 0
        assert "exactly one" in r.output.lower() or "exactly one" in str(r.exception).lower()

    def test_rejects_calib_outside_cwd(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        work = tmp_path / "work"
        work.mkdir()
        outside = tmp_path / "outside.jsonl"
        outside.write_text('{"text":"hi"}\n', encoding="utf-8")
        monkeypatch.chdir(work)
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", "x", "--drop-layers", "2", "--calib", str(outside)],
        )
        assert r.exit_code != 0

    def test_rejects_bad_tolerance(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        calib = tmp_path / "c.jsonl"
        calib.write_text('{"text":"hi"}\n', encoding="utf-8")
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", "x", "--drop-layers", "2", "--calib", "c.jsonl",
             "--tolerance", "9.0"],
        )
        assert r.exit_code != 0

    def test_prune_happy_path_cpu(self, tmp_path, monkeypatch):
        """End-to-end prune (no heal) on a tiny CPU Llama: pruned config has
        fewer layers, report JSON written, exit 0 (SHIP) — tolerance wide."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        model_dir = _write_tiny_model(tmp_path / "src_model", layers=6)
        calib = tmp_path / "calib.jsonl"
        calib.write_text(
            "\n".join('{"text":"the quick brown fox jumps over the lazy dog"}'
                      for _ in range(4)),
            encoding="utf-8",
        )
        out_dir = tmp_path / "shrunk"
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--device", "cpu",
             "--output-dir", str(out_dir), "--tolerance", "5.0"],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        cfg = json.loads((out_dir / "model" / "config.json").read_text(encoding="utf-8"))
        assert cfg["num_hidden_layers"] == 4
        report = json.loads((out_dir / "shrink_report.json").read_text(encoding="utf-8"))
        assert report["layers_before"] == 6 and report["layers_after"] == 4
        assert report["healed"] is False
        assert "ppl_original" in report

    def test_plan_only_writes_nothing(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        model_dir = _write_tiny_model(tmp_path / "src_model2", layers=6)
        calib = tmp_path / "calib.jsonl"
        calib.write_text('{"text":"the quick brown fox jumps"}\n', encoding="utf-8")
        out_dir = tmp_path / "shrunk2"
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--device", "cpu",
             "--output-dir", str(out_dir), "--plan-only"],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert not (out_dir / "model").exists()

    def test_reject_unsupported_arch(self, tmp_path, monkeypatch):
        """A GPT-NeoX-family tiny model is a friendly reject (arch allowlist)."""
        from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        cfg = GPTNeoXConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=6,
            num_attention_heads=4, vocab_size=len(tok), max_position_embeddings=512,
        )
        mdir = tmp_path / "neox"
        GPTNeoXForCausalLM(cfg).save_pretrained(str(mdir))
        tok.save_pretrained(str(mdir))
        calib = tmp_path / "calib.jsonl"
        calib.write_text('{"text":"hi there"}\n', encoding="utf-8")
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", str(mdir), "--drop-layers", "2",
             "--calib", "calib.jsonl", "--device", "cpu"],
        )
        assert r.exit_code != 0
        assert "support" in r.output.lower() or "support" in str(r.exception).lower()


# ---------------------------------------------------------------------------
# Task 5 — subprocess distill-heal + fuse
# ---------------------------------------------------------------------------
class TestHeal:
    def test_build_heal_config_parses(self):
        from soup_cli.commands.shrink import _build_heal_config_yaml
        from soup_cli.config.loader import load_config_from_string

        y = _build_heal_config_yaml(
            pruned_dir="./out/model",
            teacher="orig/model",
            heal_data="./heal.jsonl",
            steps=200,
            out_dir="./out/heal_adapter",
            heal_rows=50,
        )
        cfg = load_config_from_string(y)
        assert cfg.task == "distill"
        assert cfg.training.teacher_model == "orig/model"
        assert cfg.base == "./out/model"
        assert cfg.output == "./out/heal_adapter"
        assert cfg.training.epochs >= 1

    def test_build_heal_config_epochs_scale_with_steps(self):
        from soup_cli.commands.shrink import _build_heal_config_yaml
        from soup_cli.config.loader import load_config_from_string

        few = load_config_from_string(
            _build_heal_config_yaml(
                pruned_dir="./m", teacher="t", heal_data="./h.jsonl",
                steps=10, out_dir="./o", heal_rows=100,
            )
        )
        many = load_config_from_string(
            _build_heal_config_yaml(
                pruned_dir="./m", teacher="t", heal_data="./h.jsonl",
                steps=800, out_dir="./o", heal_rows=100,
            )
        )
        assert many.training.epochs > few.training.epochs

    def test_heal_path_sets_healed(self, tmp_path, monkeypatch):
        """With --heal, the report records healed=True (subprocess + fuse are
        stubbed so no real training runs)."""
        import json

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import shrink as shrink_cmd

        # Stub the heavy heal step: no subprocess, no fuse (pruned model stays).
        monkeypatch.setattr(shrink_cmd, "_run_heal", lambda *a, **k: None)

        monkeypatch.chdir(tmp_path)
        model_dir = _write_tiny_model(tmp_path / "src_model_h", layers=6)
        calib = tmp_path / "calib.jsonl"
        calib.write_text('{"text":"the quick brown fox jumps over"}\n', encoding="utf-8")
        heal = tmp_path / "heal.jsonl"
        heal.write_text(
            '{"messages":[{"role":"user","content":"hi"},'
            '{"role":"assistant","content":"hello"}]}\n',
            encoding="utf-8",
        )
        out_dir = tmp_path / "shrunk_h"
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--heal", "heal.jsonl", "--heal-steps", "5",
             "--device", "cpu", "--output-dir", str(out_dir), "--tolerance", "5.0"],
        )
        assert r.exit_code == 0, (r.output, repr(r.exception))
        report = json.loads((out_dir / "shrink_report.json").read_text(encoding="utf-8"))
        assert report["healed"] is True

    def test_heal_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        work = tmp_path / "work"
        work.mkdir()
        outside = tmp_path / "outside_heal.jsonl"
        outside.write_text('{"text":"hi"}\n', encoding="utf-8")
        monkeypatch.chdir(work)
        model_dir = _write_tiny_model(work / "m", layers=6)
        calib = work / "calib.jsonl"
        calib.write_text('{"text":"hi there friend"}\n', encoding="utf-8")
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--heal", str(outside), "--device", "cpu"],
        )
        assert r.exit_code != 0


# ---------------------------------------------------------------------------
# Task 6 — registry attach (real round-trip)
# ---------------------------------------------------------------------------
class TestRegistryAttach:
    def test_attach_round_trip(self, tmp_path, monkeypatch):
        """_attach_to_registry attaches a shrink_report artifact to a real
        registry entry (keyword-correct call; not the diagnose positional bug)."""
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.chdir(tmp_path)

        from soup_cli.commands.shrink import _attach_to_registry
        from soup_cli.registry.store import RegistryStore

        with RegistryStore() as store:
            entry_id = store.push(
                name="tiny-shrunk",
                tag="test",
                base_model="HuggingFaceTB/SmolLM2-135M",
                task="sft",
                run_id=None,
                config={"task": "sft"},
            )

        report = tmp_path / "shrink_report.json"
        report.write_text('{"decision":"SHIP"}', encoding="utf-8")

        _attach_to_registry(entry_id, str(report))

        with RegistryStore() as store:
            artifacts = store.get_artifacts(entry_id)
        assert any(a.get("kind") == "shrink_report" for a in artifacts), artifacts

    def test_attach_unknown_entry_warns_no_raise(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg2.db"))
        monkeypatch.chdir(tmp_path)

        from soup_cli.commands.shrink import _attach_to_registry

        report = tmp_path / "shrink_report.json"
        report.write_text('{"decision":"SHIP"}', encoding="utf-8")
        # Must not raise even for a nonexistent entry (best-effort warn).
        _attach_to_registry("nonexistent-id", str(report))


# ---------------------------------------------------------------------------
# Review-fix regression guards (python-review CRITICAL + MEDIUM-2)
# ---------------------------------------------------------------------------
class TestReviewFixes:
    def test_output_dir_outside_cwd_rejected(self, tmp_path, monkeypatch):
        """--output-dir must be cwd-contained (arbitrary-write guard)."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        calib = work / "calib.jsonl"
        calib.write_text('{"text":"hi there friend"}\n', encoding="utf-8")
        model_dir = _write_tiny_model(work / "m", layers=6)
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--device", "cpu",
             "--output-dir", str(tmp_path / "escape")],
        )
        assert r.exit_code != 0

    def test_heal_epochs_clamp_rejects_absurd_combo(self):
        """Huge --heal-steps over a tiny heal set is refused, not silently run."""
        import typer

        from soup_cli.commands.shrink import _build_heal_config_yaml

        with pytest.raises(typer.BadParameter):
            _build_heal_config_yaml(
                pruned_dir="./m", teacher="t", heal_data="./h.jsonl",
                steps=1_000_000, out_dir="./o", heal_rows=1,
            )

    def test_perplexity_no_top_level_math_import_uses_isnan(self):
        """The NaN filter uses math.isnan, not the x == x self-compare idiom."""
        src = pathlib.Path("src/soup_cli/commands/shrink.py").read_text(encoding="utf-8")
        assert "loss == loss" not in src
        assert "math.isnan(loss)" in src
