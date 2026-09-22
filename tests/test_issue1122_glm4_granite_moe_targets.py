"""#1122: two MoE architectures the #1102 ratchet could not see.

``MOE_TEXT_LORA_TARGETS`` (#1070) is sized by a ratchet over every base a shipped
recipe names, but a base whose ``config.json`` cannot be fetched is recorded as
``unresolvable`` and skipped. Two were: ``THUDM/glm-4.6`` (404) and
``ibm-granite/granite-4.0-tiny-base`` (404). #1132 repoints them at
``zai-org/GLM-4.6`` and ``ibm-granite/granite-4.0-tiny-base-preview``, which
resolve -- and pull two uncovered MoE ``model_type`` values into the catalogue.

Measured with ``AutoConfig.from_pretrained`` + ``AutoModelForCausalLM.from_config``
on the meta device (no weights downloaded), 2026-09-22, transformers 5.17.0:

    zai-org/GLM-4.6                             glm4_moe           92 layers
        self_attn leaves: q_proj, k_proj, v_proj, o_proj (+ q_norm / k_norm,
        which are RMSNorms). Ordinary shape, qwen3_moe's list.

    ibm-granite/granite-4.0-tiny-base-preview   granitemoehybrid   40 layers
        layer_types: 36 linear_attention + 4 full_attention, and only those 4
        layers (5, 15, 25, 35) define a self_attn at all. Linear suffix counts
        across the whole model: input_linear 40, output_linear 40, in_proj 36,
        out_proj 36, q_proj/k_proj/v_proj/o_proj 4 each, lm_head 1.

The stand-ins below are tiny configs of the same classes rather than the Hub
repos, because the suite does not touch the network. The granite stand-in
reproduces the real model's suffix counts EXACTLY (40/40/36/36/4/4/4/4/1), which
is the check that makes it a stand-in rather than a different model.
"""

from __future__ import annotations

import collections
import re
import types
from io import StringIO

import pytest
from rich.console import Console

from soup_cli.utils.peft_wiring import (
    MOE_TEXT_LORA_TARGETS,
    PARTIAL_COVERAGE_NOTES,
    resolve_lora_target_modules,
)

pytest.importorskip("torch")
pytest.importorskip("transformers")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def _plain(text: str) -> str:
    """Rich colours AND wraps; both have turned this project's CI red before."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _recording_console():
    buffer = StringIO()
    return Console(file=buffer, width=400, no_color=True), buffer


def _glm4_moe_config(num_hidden_layers: int = 4):
    from transformers import Glm4MoeConfig

    return Glm4MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
    )


def _granite_config(num_hidden_layers: int = 40, attention_every: int = 10):
    """A granitemoehybrid whose attention layers sit where the real one's do.

    ``attention_every`` is a knob, not a constant, so a test can build a DIFFERENT
    depth and prove the printed fraction is counted rather than hard-coded.
    """
    from transformers import GraniteMoeHybridConfig

    layer_types = [
        "full_attention"
        if (index - (attention_every // 2)) % attention_every == 0
        else "linear_attention"
        for index in range(num_hidden_layers)
    ]
    return GraniteMoeHybridConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        shared_intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=layer_types,
        mamba_n_heads=2,
        mamba_n_groups=1,
        mamba_d_state=8,
        mamba_d_head=8,
        mamba_d_conv=4,
        mamba_expand=1,
    )


def _build(config):
    import torch
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def _linear_keys(model):
    import torch

    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _matched(keys, targets):
    """The keys peft's suffix matching would adapt for a list target."""
    return [key for key in keys if key.split(".")[-1] in set(targets)]


def _layer_indices(model, *, only_self_attn=False):
    return {
        name.split("layers.")[1].split(".")[0]
        for name, _module in model.named_modules()
        if "layers." in name and (name.endswith(".self_attn") or not only_self_attn)
    }


class TestTheTwoEntries:
    """Pinned literally, like the rest of the table: each list came from a probe,
    and a plausible-looking edit is a guess unless the probe is re-run."""

    def test_glm4_moe_is_the_measured_list(self):
        assert MOE_TEXT_LORA_TARGETS["glm4_moe"] == _ATTENTION

    def test_granitemoehybrid_is_the_measured_list(self):
        assert MOE_TEXT_LORA_TARGETS["granitemoehybrid"] == _ATTENTION

    def test_glm4_moe_is_not_the_glm_5_entry(self):
        """GLM-4.6 and GLM-5/5.1 are different attention shapes; collapsing them
        onto one row would target names GLM-4.6 does not define."""
        assert MOE_TEXT_LORA_TARGETS["glm4_moe"] != MOE_TEXT_LORA_TARGETS["glm_moe_dsa"]

    @pytest.mark.parametrize("model_type", ["glm4_moe", "granitemoehybrid"])
    def test_the_resolver_returns_them_for_a_real_config(self, model_type):
        """Through a REAL transformers config, not a namespace double: the
        resolver reads ``model_type`` off whatever it is handed, and these two
        values come from transformers' own config classes rather than from me."""
        config = _glm4_moe_config() if model_type == "glm4_moe" else _granite_config()

        assert config.model_type == model_type
        assert resolve_lora_target_modules(config, "auto") == list(_ATTENTION)


class TestBuiltOnTheMetaDevice:
    def test_glm4_moe_targets_hit_every_attention_projection_and_nothing_else(self):
        model = _build(_glm4_moe_config())
        keys = _linear_keys(model)
        matched = _matched(keys, MOE_TEXT_LORA_TARGETS["glm4_moe"])

        attention = [key for key in keys if ".self_attn." in key]
        assert matched, "sanity: the list matched nothing at all"
        assert sorted(matched) == sorted(attention)
        assert not [key for key in matched if "mlp" in key]

    def test_the_glm4_moe_mlp_projections_exist_for_a_careless_list_to_hit(self):
        """The premise of the assertion above, read off the model: GLM-4.6 has
        ``gate_proj`` / ``up_proj`` / ``down_proj`` Linears, so leaving them out
        of the entry is a visible choice rather than an absence."""
        keys = _linear_keys(_build(_glm4_moe_config()))

        assert [
            key
            for key in keys
            if key.split(".")[-1] in {"gate_proj", "up_proj", "down_proj"}
        ]

    def test_granite_is_a_hybrid_and_only_four_of_forty_layers_attend(self):
        """The 4-of-40 fact, COUNTED off the built model rather than asserted as a
        literal: if transformers ever gives every granitemoehybrid layer a
        ``self_attn``, this fails instead of quietly making the notice wrong."""
        config = _granite_config()
        model = _build(config)

        layers = _layer_indices(model)
        attentive = _layer_indices(model, only_self_attn=True)

        assert len(layers) == 40
        assert len(attentive) == 4
        assert len(attentive) == config.layer_types.count("full_attention")
        assert sorted(int(index) for index in attentive) == [5, 15, 25, 35]

    def test_the_granite_stand_in_has_the_real_repos_suffix_counts(self):
        """What makes this a stand-in for ``granite-4.0-tiny-base-preview`` rather
        than merely some granitemoehybrid: the measured counts in the docstring."""
        counts = collections.Counter(
            key.split(".")[-1] for key in _linear_keys(_build(_granite_config()))
        )

        assert counts == {
            "input_linear": 40,
            "output_linear": 40,
            "in_proj": 36,
            "out_proj": 36,
            "q_proj": 4,
            "k_proj": 4,
            "v_proj": 4,
            "o_proj": 4,
            "lm_head": 1,
        }

    def test_granite_targets_reach_the_attention_layers_and_no_mamba_block(self):
        keys = _linear_keys(_build(_granite_config()))
        matched = _matched(keys, MOE_TEXT_LORA_TARGETS["granitemoehybrid"])

        assert len(matched) == 16, matched
        assert all(".self_attn." in key for key in matched)
        assert not [key for key in matched if ".mamba." in key or "shared_mlp" in key]

    def test_the_mamba_and_shared_expert_projections_are_there_to_be_skipped(self):
        """``out_proj`` is a Mamba name here and an ATTENTION name elsewhere, so
        "deliberately not adapted" has to be visible rather than assumed."""
        keys = _linear_keys(_build(_granite_config()))
        skipped = [key for key in keys if ".mamba." in key or "shared_mlp" in key]

        assert len([key for key in keys if key.endswith(".mamba.in_proj")]) == 36
        assert len([key for key in keys if key.endswith(".mamba.out_proj")]) == 36
        assert len([key for key in keys if "shared_mlp" in key]) == 80
        assert not _matched(skipped, MOE_TEXT_LORA_TARGETS["granitemoehybrid"])


class TestThePartialCoverageNotice:
    def test_granite_is_announced_with_its_type_its_fraction_and_the_remainder(self):
        console, buffer = _recording_console()

        resolve_lora_target_modules(_build(_granite_config()), "auto", console)

        out = _plain(buffer.getvalue())
        assert "Partial LoRA coverage" in out
        assert "granitemoehybrid" in out
        assert "4 of 40 decoder layers" in out
        assert "mamba.in_proj" in out and "mamba.out_proj" in out
        assert "shared_mlp.input_linear" in out
        assert "training.lora.target_modules" in out

    def test_the_fraction_is_counted_not_a_hard_coded_four_of_forty(self):
        """A shallower granitemoehybrid reports ITS numbers. A literal "4 of 40"
        in the message would pass every other assertion in this class."""
        console, buffer = _recording_console()

        resolve_lora_target_modules(
            _build(_granite_config(num_hidden_layers=8, attention_every=4)),
            "auto",
            console,
        )

        out = _plain(buffer.getvalue())
        assert "2 of 8 decoder layers" in out
        assert "4 of 40" not in out

    def test_glm4_moe_is_not_announced(self):
        """Full coverage: every GLM-4.6 layer has a self_attn, so a notice here
        would be noise -- and noise is how a real advisory stops being read."""
        console, buffer = _recording_console()

        resolve_lora_target_modules(_build(_glm4_moe_config()), "auto", console)

        assert buffer.getvalue() == ""

    @pytest.mark.parametrize(
        "model_type", sorted(set(MOE_TEXT_LORA_TARGETS) - {"granitemoehybrid"})
    )
    def test_no_other_table_entry_announces_anything(self, model_type):
        console, buffer = _recording_console()
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type=model_type, text_config=None)
        )

        resolve_lora_target_modules(model, "auto", console)

        assert buffer.getvalue() == ""

    def test_a_bare_config_is_announced_from_layer_types(self):
        """``stream_setup`` resolves from a model config, before any model exists;
        the fraction then comes from ``layer_types`` instead of a module count."""
        console, buffer = _recording_console()

        resolve_lora_target_modules(_granite_config(), "auto", console)

        assert "4 of 40 decoder layers" in _plain(buffer.getvalue())

    def test_without_a_console_the_resolution_is_unchanged_and_silent(self, capsys):
        """The default is no console, and the notice must not become a print()."""
        resolved = resolve_lora_target_modules(_build(_granite_config()), "auto")

        assert resolved == list(_ATTENTION)
        assert capsys.readouterr().out == ""

    def test_a_console_that_raises_does_not_fail_the_resolution(self):
        class Exploding:
            def print(self, *_args, **_kwargs):
                raise RuntimeError("no terminal")

        resolved = resolve_lora_target_modules(
            _build(_granite_config()), "auto", Exploding()
        )

        assert resolved == list(_ATTENTION)

    def test_an_explicit_target_list_is_returned_without_a_notice(self):
        """The advisory is about ``auto``. Someone who wrote the list already
        knows what they targeted."""
        console, buffer = _recording_console()

        resolved = resolve_lora_target_modules(
            _build(_granite_config()), ["q_proj"], console
        )

        assert resolved == ["q_proj"]
        assert buffer.getvalue() == ""


class TestEveryCallSiteSuppliesTheConsole:
    """#1102's review found that nothing drove a trainer, so reverting its call
    sites was invisible (F4). The notice has the same shape: the resolver is
    tested directly above, and a trainer that quietly stops passing ``console``
    would silence the advisory with every behaviour test still green.

    A source ratchet rather than nineteen trainer drives -- the property is
    syntactic ("the argument is there"), and the behaviour it enables is already
    pinned against a real Console above.
    """

    @staticmethod
    def _call_sites():
        import ast
        from pathlib import Path

        source_root = Path(__file__).resolve().parent.parent / "src" / "soup_cli"
        for path in sorted(source_root.rglob("*.py")):
            if path.name == "peft_wiring.py":
                continue  # the definition, not a call site
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "resolve_lora_target_modules"
                ):
                    yield f"{path.name}:{node.lineno}", node

    def test_there_are_call_sites_to_check(self):
        """The ratchet's own premise: a rename that made the walk find nothing
        would otherwise pass it silently."""
        assert len(list(self._call_sites())) >= 15

    def test_every_call_site_passes_a_console(self):
        silent = [
            where
            for where, node in self._call_sites()
            if len(node.args) < 3
            and not any(kw.arg == "console" for kw in node.keywords)
        ]

        assert silent == [], (
            "These resolve_lora_target_modules() calls drop the partial-coverage "
            "advisory on the floor:\n  " + "\n  ".join(silent)
        )


class TestTheNotesTableIsConsistent:
    def test_every_note_names_an_entry_that_exists(self):
        assert set(PARTIAL_COVERAGE_NOTES) <= set(MOE_TEXT_LORA_TARGETS)

    def test_the_notes_are_pinned(self):
        """A new partial-coverage row is a reviewed edit, like EXCLUDED_MOE_BASES:
        growing one silently is how an architecture gets half-adapted in quiet."""
        assert sorted(PARTIAL_COVERAGE_NOTES) == ["granitemoehybrid"]
        for model_type, note in PARTIAL_COVERAGE_NOTES.items():
            assert len(note) > 40, model_type
