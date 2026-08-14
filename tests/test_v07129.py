"""v0.71.29 — `soup shrink`: depth-prune + distill-heal (arXiv:2403.17887).

Tests the pure verdict half, the torch-lazy prune/importance half, the CLI
orchestration, the subprocess distill-heal wiring, and registry attach.
"""
import ast
import inspect
import math
import re
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


def _strip_ansi(text: str) -> str:
    """Drop ANSI SGR codes so Rich-split flag names (``--drop`` + color +
    ``-ratio``) become contiguous substrings on colour-enabled CI terminals."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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
        import soup_cli.utils.shrink as _mod

        src = inspect.getsource(_mod)  # cwd-independent (CI runs from a temp cwd)
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
        """Pin the EXACT indices: block [L, L+n) uses hidden_states[L] and
        hidden_states[L+n]; hidden_states has num_layers+1 entries. Per-layer
        directions are mutually non-colinear with a NON-linear angle schedule
        (theta_k = 0.1*k^2) so the hand-computed distance for each start is
        unique — an off-by-one (hs[L-1]/hs[L+n-1]) or a block_size-ignoring bug
        would produce a different number and FAIL."""
        import math as _math

        import torch

        from soup_cli.utils import shrink

        num_layers = 5  # -> valid starts for block_size=2: L in [1, 2]

        def _vec(k: int):
            theta = 0.1 * k * k
            row = torch.tensor([_math.cos(theta), _math.sin(theta)])
            return row.repeat(1, 3, 1)  # (1, 3, 2): 3 identical seq rows

        hs = tuple(_vec(k) for k in range(num_layers + 1))

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
            _Model(), _Tok(), ["hi"], block_size=2, device="cpu"
        )
        by_start = {i.start: i.angular_distance for i in imps}
        assert sorted(by_start) == [1, 2]
        # start=1 -> hs[1] vs hs[3]: |theta_1 - theta_3| = |0.1 - 0.9| = 0.8 rad.
        assert by_start[1] == pytest.approx(0.8 / _math.pi, abs=1e-6)
        # start=2 -> hs[2] vs hs[4]: |theta_2 - theta_4| = |0.4 - 1.6| = 1.2 rad.
        assert by_start[2] == pytest.approx(1.2 / _math.pi, abs=1e-6)
        assert all(i.block_size == 2 for i in imps)

    def test_importance_averages_over_all_tokens_across_prompts(self):
        """The distance is a mean over EVERY token across the whole calib set,
        not a mean-of-per-prompt-means. Two prompts of different token counts
        with different per-token distances must weight by token count."""
        import math as _math

        import torch

        from soup_cli.utils import shrink

        num_layers = 3  # valid starts for block_size=1: L in [1, 1]

        class _Cfg:
            model_type = "llama"
            architectures = ["LlamaForCausalLM"]
            num_hidden_layers = num_layers

        # Two calls: prompt A has 1 token at angle 0 (distance 0), prompt B has
        # 3 tokens at 90deg (distance 0.5). Token-weighted mean = 3*0.5/4 = 0.375;
        # mean-of-means would be (0 + 0.5)/2 = 0.25.
        calls = {"n": 0}

        def _hs_for(seq_len: int, theta_out: float):
            in_row = torch.tensor([1.0, 0.0]).repeat(1, seq_len, 1)
            out_row = torch.tensor(
                [_math.cos(theta_out), _math.sin(theta_out)]
            ).repeat(1, seq_len, 1)
            # layer 0 (emb) + layer1 out (in) + layer2 out (out) — index1 vs index2
            return (in_row, in_row, out_row, out_row)

        class _Out:
            def __init__(self, hs):
                self.hidden_states = hs

        class _Model:
            config = _Cfg()

            def eval(self):
                return self

            def __call__(self, **kw):
                if calls["n"] == 0:
                    calls["n"] = 1
                    return _Out(_hs_for(1, 0.0))       # 1 token, dist 0
                return _Out(_hs_for(3, _math.pi / 2))  # 3 tokens, dist 0.5

        class _Tok:
            def __call__(self, text, **kw):
                n = 1 if calls["n"] == 0 else 3
                return {
                    "input_ids": torch.ones(1, n, dtype=torch.long),
                    "attention_mask": torch.ones(1, n, dtype=torch.long),
                }

        imps = shrink.compute_layer_importance(
            _Model(), _Tok(), ["a", "b"], block_size=1, device="cpu"
        )
        assert imps[0].angular_distance == pytest.approx(0.375, abs=1e-6)

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
        clean = _strip_ansi(r.output)  # Rich splits flag names with colour codes
        assert "drop-ratio" in clean
        assert "calib" in clean

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
        assert r.exit_code == 1, (r.output, repr(r.exception))
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
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "cwd" in r.output.lower()

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
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "tolerance" in r.output.lower()

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
        assert r.exit_code == 1, (r.output, repr(r.exception))
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
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "cwd" in r.output.lower()


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
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "cwd" in r.output.lower()

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
        import soup_cli.commands.shrink as _mod

        src = inspect.getsource(_mod)
        assert "loss == loss" not in src
        assert "math.isnan(loss)" in src

    def test_fuse_adapter_produces_dense_model(self, tmp_path, monkeypatch):
        """_fuse_adapter merges a LoRA adapter back into the base (atomic swap),
        so the shipped dir is a single dense model (no adapter_config.json)."""
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from soup_cli.commands.shrink import _fuse_adapter

        monkeypatch.chdir(tmp_path)  # base_dir must stay under cwd (containment)
        base_dir = tmp_path / "base"
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        _tiny_llama(4, vocab_size=len(tok)).save_pretrained(str(base_dir))
        tok.save_pretrained(str(base_dir))

        adapter_dir = tmp_path / "adapter"
        base = AutoModelForCausalLM.from_pretrained(str(base_dir))
        peft_model = get_peft_model(
            base,
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                       task_type=TaskType.CAUSAL_LM),
        )
        peft_model.save_pretrained(str(adapter_dir))

        _fuse_adapter(base_dir=str(base_dir), adapter_dir=str(adapter_dir))

        # In-place overwrite yields a dense model — no adapter marker survives.
        assert not (base_dir / "adapter_config.json").exists()
        fused = AutoModelForCausalLM.from_pretrained(str(base_dir))
        assert fused.config.num_hidden_layers == 4

    def test_output_model_symlink_rejected(self, tmp_path, monkeypatch):
        """A symlink planted at <output_dir>/model is rejected before any write
        (derived-path TOCTOU guard). POSIX-only (needs os.symlink)."""
        import os as _os

        if not hasattr(_os, "symlink"):
            pytest.skip("no os.symlink")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        model_dir = _write_tiny_model(tmp_path / "src_sym", layers=6)
        calib = tmp_path / "calib.jsonl"
        calib.write_text('{"text":"the quick brown fox"}\n', encoding="utf-8")
        out_dir = tmp_path / "shrunk_sym"
        out_dir.mkdir()
        escape_target = tmp_path / "escape_target"
        escape_target.mkdir()
        try:
            _os.symlink(str(escape_target), str(out_dir / "model"),
                        target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform")
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--device", "cpu",
             "--output-dir", str(out_dir), "--tolerance", "5.0"],
        )
        assert r.exit_code == 1, (r.output, repr(r.exception))
        assert "symlink" in r.output.lower()

    def test_for_terminal_strips_control_bytes(self):
        from soup_cli.commands.shrink import _for_terminal

        assert _for_terminal("a\x1b]0;evilbc") == "a]0;evilbc"
        # tab / LF / CR preserved.
        assert _for_terminal("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_dont_ship_exit_code_2(self, tmp_path, monkeypatch):
        """A genuine perplexity regression past tolerance exits 2 (DON'T SHIP)."""
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import shrink as shrink_cmd

        # Force a regression: original 10.0 -> pruned 20.0 (ratio 2.0 >> tol).
        seq = iter([10.0, 20.0])
        monkeypatch.setattr(shrink_cmd, "_perplexity", lambda *a, **k: next(seq))

        monkeypatch.chdir(tmp_path)
        model_dir = _write_tiny_model(tmp_path / "m_ds", layers=6)
        calib = tmp_path / "calib.jsonl"
        calib.write_text('{"text":"the quick brown fox"}\n', encoding="utf-8")
        out_dir = tmp_path / "shrunk_ds"
        r = CliRunner().invoke(
            app,
            ["shrink", "--model", model_dir, "--drop-layers", "2",
             "--calib", "calib.jsonl", "--device", "cpu",
             "--output-dir", str(out_dir), "--tolerance", "0.10"],
        )
        assert r.exit_code == 2, (r.output, repr(r.exception))


# ---------------------------------------------------------------------------
# TDD-review gap closure (tdd agent findings #3-#17)
# ---------------------------------------------------------------------------
class _StubProc:
    def __init__(self, returncode=0, stderr=b"", stdout=b""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class TestRunHeal:
    def test_success_calls_fuse_with_right_dirs(self, tmp_path, monkeypatch):
        import soup_cli.commands.shrink as sc

        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()
        seen = {}
        monkeypatch.setattr("subprocess.run", lambda *a, **k: _StubProc(returncode=0))
        monkeypatch.setattr(sc, "_fuse_adapter",
                            lambda **kw: seen.update(kw))
        sc._run_heal(pruned_dir="./model", teacher="t", heal_data="./h.jsonl",
                     steps=5, out_dir="./adapter", heal_rows=10, trc=False)
        assert seen["base_dir"] == "./model"
        assert seen["adapter_dir"] == "./adapter"
        assert (tmp_path / "heal_config.yaml").exists()

    def test_nonzero_returncode_raises_with_tail(self, tmp_path, monkeypatch):
        import soup_cli.commands.shrink as sc

        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: _StubProc(returncode=1, stderr=b"boom")
        )
        monkeypatch.setattr(sc, "_fuse_adapter", lambda **kw: None)
        with pytest.raises(RuntimeError, match="heal distill failed"):
            sc._run_heal(pruned_dir="./model", teacher="t", heal_data="./h.jsonl",
                         steps=5, out_dir="./adapter", heal_rows=10)

    def test_nonzero_tail_control_bytes_stripped(self, tmp_path, monkeypatch):
        import soup_cli.commands.shrink as sc

        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()
        monkeypatch.setattr("subprocess.run",
                            lambda *a, **k: _StubProc(returncode=1, stderr=b"a\x1bb"))
        with pytest.raises(RuntimeError) as exc:
            sc._run_heal(pruned_dir="./model", teacher="t", heal_data="./h.jsonl",
                         steps=5, out_dir="./adapter", heal_rows=10)
        assert "\x1b" not in str(exc.value)

    def test_timeout_raises(self, tmp_path, monkeypatch):
        import subprocess as _sp

        import soup_cli.commands.shrink as sc

        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()

        def _boom(*a, **k):
            raise _sp.TimeoutExpired(cmd="soup train", timeout=1)

        monkeypatch.setattr("subprocess.run", _boom)
        with pytest.raises(RuntimeError, match="timeout"):
            sc._run_heal(pruned_dir="./model", teacher="t", heal_data="./h.jsonl",
                         steps=5, out_dir="./adapter", heal_rows=10)

    def test_device_cpu_hides_gpu_in_subprocess_env(self, tmp_path, monkeypatch):
        """device='cpu' must run the heal with CUDA_VISIBLE_DEVICES=-1 so the
        distill honours CPU (and dodges the GPU hardware-fit gate)."""
        import soup_cli.commands.shrink as sc

        monkeypatch.chdir(tmp_path)
        (tmp_path / "model").mkdir()
        seen = {}

        def _capture(*a, **k):
            seen["env"] = k.get("env")
            return _StubProc(returncode=0)

        monkeypatch.setattr("subprocess.run", _capture)
        monkeypatch.setattr(sc, "_fuse_adapter", lambda **kw: None)
        sc._run_heal(pruned_dir="./model", teacher="t", heal_data="./h.jsonl",
                     steps=5, out_dir="./adapter", heal_rows=10, device="cpu")
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "-1"

    def test_config_has_gradient_checkpointing_and_batch1(self):
        from soup_cli.commands.shrink import _build_heal_config_yaml
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            _build_heal_config_yaml(pruned_dir="./m", teacher="t",
                                    heal_data="./h.jsonl", steps=8, out_dir="./o",
                                    heal_rows=8)
        )
        assert cfg.training.batch_size == 1
        assert cfg.training.gradient_checkpointing is True


class TestDropCountEdges:
    def test_ratio_rounds_to_zero_rejected(self):
        from soup_cli.utils.shrink import resolve_drop_count

        with pytest.raises(ValueError, match="range"):
            resolve_drop_count(10, drop_ratio=0.01, drop_layers=None)  # round(0.1)=0

    def test_accepted_max_boundary(self):
        from soup_cli.utils.shrink import resolve_drop_count

        assert resolve_drop_count(6, drop_ratio=None, drop_layers=4) == 4  # n-2

    def test_rejects_bool_drop_layers(self):
        from soup_cli.utils.shrink import resolve_drop_count

        with pytest.raises(ValueError):
            resolve_drop_count(10, drop_ratio=None, drop_layers=True)

    def test_rejects_bool_drop_ratio(self):
        from soup_cli.utils.shrink import resolve_drop_count

        with pytest.raises(ValueError):
            resolve_drop_count(10, drop_ratio=True, drop_layers=None)

    def test_max_count_agrees_across_three_functions(self):
        from soup_cli.utils.shrink import prune_model_layers, resolve_drop_count

        assert resolve_drop_count(6, drop_ratio=None, drop_layers=4) == 4
        m = _tiny_llama(6)
        prune_model_layers(m, start=1, block_size=4)  # leaves layers 0 and 5
        assert len(m.model.layers) == 2
        # one count higher is rejected by prune (would touch the last layer)
        m2 = _tiny_llama(6)
        with pytest.raises(ValueError, match="protected"):
            prune_model_layers(m2, start=1, block_size=5)


class TestPruneBool:
    def test_rejects_bool_start_and_block(self):
        from soup_cli.utils.shrink import prune_model_layers

        m = _tiny_llama(6)
        with pytest.raises(ValueError):
            prune_model_layers(m, start=True, block_size=2)
        m2 = _tiny_llama(6)
        with pytest.raises(ValueError):
            prune_model_layers(m2, start=1, block_size=True)

    def test_layer_list_missing_modulelist(self):
        from soup_cli.utils.shrink import layer_list

        class _Cfg:
            model_type = "llama"
            architectures = ["LlamaForCausalLM"]

        class _M:
            config = _Cfg()

        with pytest.raises(ValueError, match="ModuleList"):
            layer_list(_M())


class TestReloadFixesLayerIdx:
    def test_prune_leaves_stale_idx_reload_fixes(self, tmp_path):
        from transformers import AutoModelForCausalLM

        from soup_cli.utils.shrink import prune_model_layers

        m = _tiny_llama(6)
        attn = m.model.layers[4].self_attn
        if not hasattr(attn, "layer_idx"):
            pytest.skip("transformers version has no self_attn.layer_idx")
        prune_model_layers(m, start=2, block_size=2)  # drop 2,3 -> old-4 now at pos 2
        # In-memory slice leaves the stale original index on the moved layer.
        assert m.model.layers[2].self_attn.layer_idx == 4
        out = tmp_path / "pruned"
        m.save_pretrained(str(out))
        reloaded = AutoModelForCausalLM.from_pretrained(str(out))
        # from_pretrained rebuilds contiguous indices 0..3.
        assert [reloaded.model.layers[i].self_attn.layer_idx for i in range(4)] == [0, 1, 2, 3]


class TestArchQwen:
    def test_qwen_detected(self):
        from soup_cli.utils.shrink import arch_family_of_config

        class _Cfg:
            model_type = "qwen2"
            architectures = ["Qwen2ForCausalLM"]

        assert arch_family_of_config(_Cfg()) == "qwen"


class TestExtractText:
    def test_plain_string(self):
        from soup_cli.commands.shrink import _extract_text

        assert _extract_text("hello") == "hello"

    def test_prompt_and_content_and_instruction_keys(self):
        from soup_cli.commands.shrink import _extract_text

        assert _extract_text({"prompt": "p"}) == "p"
        assert _extract_text({"content": "c"}) == "c"
        assert _extract_text({"instruction": "i"}) == "i"

    def test_messages_join(self):
        from soup_cli.commands.shrink import _extract_text

        row = {"messages": [{"role": "user", "content": "a"},
                            {"role": "assistant", "content": "b"}]}
        assert _extract_text(row) == "a\nb"

    def test_text_precedence_over_messages(self):
        from soup_cli.commands.shrink import _extract_text

        row = {"text": "T", "messages": [{"role": "user", "content": "M"}]}
        assert _extract_text(row) == "T"

    def test_no_usable_field_returns_empty(self):
        from soup_cli.commands.shrink import _extract_text

        assert _extract_text({"other": 1}) == ""


class TestLoadCalibEdges:
    def _run(self, tmp_path, monkeypatch, content):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "c.jsonl"
        p.write_text(content, encoding="utf-8")
        from soup_cli.commands.shrink import _load_calib

        return _load_calib("c.jsonl")

    def test_empty_file_rejected(self, tmp_path, monkeypatch):
        import typer

        with pytest.raises(typer.BadParameter, match="no usable prompt"):
            self._run(tmp_path, monkeypatch, "")

    def test_whitespace_only_rejected(self, tmp_path, monkeypatch):
        import typer

        with pytest.raises(typer.BadParameter, match="no usable prompt"):
            self._run(tmp_path, monkeypatch, "   \n\n\t\n")

    def test_rows_with_no_usable_field_rejected(self, tmp_path, monkeypatch):
        import typer

        with pytest.raises(typer.BadParameter, match="no usable prompt"):
            self._run(tmp_path, monkeypatch, '{"foo":"bar"}\n')

    def test_raw_text_line_tolerated(self, tmp_path, monkeypatch):
        prompts = self._run(tmp_path, monkeypatch, "the quick brown fox jumps\n")
        assert prompts == ["the quick brown fox jumps"]

    def test_row_cap_truncates(self, tmp_path, monkeypatch):
        from soup_cli.commands import shrink as sc

        monkeypatch.setattr(sc, "_MAX_CALIB_ROWS", 3)
        prompts = self._run(
            tmp_path, monkeypatch,
            "\n".join('{"text":"row %d"}' % i for i in range(10)),
        )
        assert len(prompts) == 3

    def test_size_cap_rejected(self, tmp_path, monkeypatch):
        import typer

        from soup_cli.commands import shrink as sc

        monkeypatch.setattr(sc, "_MAX_INPUT_BYTES", 10)
        with pytest.raises(typer.BadParameter, match="exceeds"):
            self._run(tmp_path, monkeypatch, '{"text":"a long enough line to exceed"}\n')


class TestPerplexityInf:
    def test_returns_inf_when_all_single_token(self):
        import torch

        from soup_cli.commands.shrink import _perplexity

        class _Tok:
            def __call__(self, text, **kw):
                return {"input_ids": torch.ones(1, 1, dtype=torch.long)}

        class _M:
            def eval(self):
                return self

        # input_ids has < 2 tokens for every prompt -> skipped -> inf.
        assert _perplexity(_M(), _Tok(), ["a", "b"], "cpu") == float("inf")


class TestImportanceCaps:
    def _model_tok(self, num_layers=4):
        import torch

        class _Cfg:
            model_type = "llama"
            architectures = ["LlamaForCausalLM"]
            num_hidden_layers = num_layers

        class _Out:
            hidden_states = tuple(torch.ones(1, 2, 4) for _ in range(num_layers + 1))

        counter = {"n": 0}

        class _Model:
            config = _Cfg()

            def eval(self):
                return self

            def __call__(self, **kw):
                counter["n"] += 1
                return _Out()

        class _Tok:
            def __call__(self, text, **kw):
                return {"input_ids": torch.ones(1, 2, dtype=torch.long)}  # no mask

        return _Model(), _Tok(), counter

    def test_mask_none_branch(self):
        from soup_cli.utils.shrink import compute_layer_importance

        model, tok, _ = self._model_tok()
        imps = compute_layer_importance(model, tok, ["hi"], block_size=1, device="cpu")
        assert imps  # no attention_mask key -> mask None branch, still scores

    def test_max_prompts_truncates_forward_calls(self):
        from soup_cli.utils.shrink import compute_layer_importance

        model, tok, counter = self._model_tok()
        compute_layer_importance(
            model, tok, ["a", "b", "c", "d"], block_size=1, device="cpu", max_prompts=2
        )
        assert counter["n"] == 2


class TestDecideShrinkBoundaries:
    def test_just_past_tolerance_dont_ship(self):
        from soup_cli.utils.shrink import DECISION_DONT_SHIP, decide_shrink

        v = decide_shrink(10.0, 10.0 * (1.10 + 5e-9), tolerance=0.10,
                          layers_before=30, layers_after=24)
        assert v.decision == DECISION_DONT_SHIP

    def test_match_keywords_on_validation(self):
        from soup_cli.utils.shrink import decide_shrink

        with pytest.raises(ValueError, match="ppl_original must be"):
            decide_shrink(0.0, 5.0, layers_before=30, layers_after=24)
        with pytest.raises(ValueError, match="ppl_final must be"):
            decide_shrink(5.0, 0.0, layers_before=30, layers_after=24)
        with pytest.raises(ValueError, match="tolerance must be a number"):
            decide_shrink(5.0, 5.0, tolerance="x", layers_before=30, layers_after=24)
        with pytest.raises(ValueError, match="tolerance must be in"):
            decide_shrink(5.0, 5.0, tolerance=9.0, layers_before=30, layers_after=24)


class TestCommandsNoTopLevelTorch:
    def test_commands_shrink_has_no_top_level_heavy_import(self):
        import soup_cli.commands.shrink as _mod

        src = inspect.getsource(_mod)  # cwd-independent (CI runs from a temp cwd)
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


class TestFuseAdapterSymlinkGuard:
    def test_symlinked_base_dir_rejected(self, tmp_path, monkeypatch):
        import os as _os

        if not hasattr(_os, "symlink"):
            pytest.skip("no os.symlink")
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "base"
        try:
            _os.symlink(str(target), str(link), target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted")
        from soup_cli.commands.shrink import _fuse_adapter

        with pytest.raises(ValueError, match="symlink"):
            _fuse_adapter(base_dir="base", adapter_dir="adapter")
