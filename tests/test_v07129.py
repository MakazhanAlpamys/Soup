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
def _tiny_llama(layers: int = 6):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
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
