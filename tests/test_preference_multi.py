"""Tests for v0.40.0 Part D — multi-objective preference loss.

Adds ``training.preference_loss_weights: dict[str, float]`` for blending
preference losses (e.g. ``{"dpo": 0.7, "bco": 0.3}``). The schema-level
surface ships in v0.40.0; the live runtime weighted combination is
deferred to v0.40.1 (mirrors the project's stub-then-live pattern from
v0.27.0 MII / v0.37.0 multipack / v0.38.0 quant menu / v0.39.0 ReLoRA).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

# ─── Schema bounds ──────────────────────────────────────────────────────────


def _base(**training):
    return SoupConfig(
        base="some-model",
        task="preference",
        data={"train": "./data.jsonl", "format": "dpo"},
        training=training,
    )


class TestPreferenceLossWeightsConfig:
    def test_two_loss_blend_accepted(self):
        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        assert cfg.training.preference_loss_weights == {"dpo": 0.7, "bco": 0.3}

    def test_three_loss_blend_accepted(self):
        cfg = _base(
            preference_loss_weights={"dpo": 0.5, "bco": 0.3, "simpo": 0.2},
        )
        assert sum(cfg.training.preference_loss_weights.values()) == pytest.approx(1.0)

    def test_single_entry_rejected_use_scalar_form_instead(self):
        """Single-entry blends are equivalent to scalar preference_loss; reject."""
        with pytest.raises(ValidationError, match="2 and 5"):
            _base(preference_loss_weights={"dpo": 1.0})

    def test_more_than_five_entries_rejected(self):
        with pytest.raises(ValidationError, match="2 and 5"):
            _base(
                preference_loss_weights={
                    "dpo": 0.2, "bco": 0.2, "simpo": 0.2,
                    "orpo": 0.2, "ipo": 0.1, "extra": 0.1,
                },
            )

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError, match="unknown"):
            _base(preference_loss_weights={"dpo": 0.5, "garbage": 0.5})

    def test_null_byte_in_key_rejected(self):
        with pytest.raises(ValidationError, match="null byte"):
            _base(preference_loss_weights={"dpo": 0.5, "bco\x00": 0.5})

    def test_empty_dict_rejected(self):
        with pytest.raises(ValidationError, match="2 and 5"):
            _base(preference_loss_weights={})

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum"):
            _base(preference_loss_weights={"dpo": 0.5, "bco": 0.4})

    def test_weight_zero_rejected(self):
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            _base(preference_loss_weights={"dpo": 1.0, "bco": 0.0})

    def test_weight_negative_rejected(self):
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            _base(preference_loss_weights={"dpo": 1.5, "bco": -0.5})

    def test_weight_above_one_rejected(self):
        # Two values both > 1 — the per-value bound (0,1] fires before the
        # sum gate.
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            _base(preference_loss_weights={"dpo": 1.1, "bco": 1.1})

    def test_weight_bool_coerced_to_zero_then_rejected(self):
        """Pydantic coerces True/False → 1.0/0.0 before model_validator sees
        them, so a False weight is rejected by the (0, 1] gate (not the bool
        guard). Either rejection path is acceptable; this test pins the
        observed behaviour rather than the rejection mechanism."""
        with pytest.raises(ValidationError, match=r"\(0, 1\]|must be a number"):
            _base(preference_loss_weights={"dpo": True, "bco": False})

    def test_requires_preference_task(self):
        with pytest.raises(ValidationError, match="preference"):
            SoupConfig(
                base="some-model",
                task="dpo",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss_weights": {"dpo": 1.0}},
            )

    def test_mutually_exclusive_with_scalar_loss(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            _base(
                preference_loss="dpo",
                preference_loss_weights={"dpo": 0.7, "bco": 0.3},
            )

    def test_rejected_on_mlx(self):
        with pytest.raises(ValidationError, match="mlx"):
            SoupConfig(
                base="some-model",
                task="preference",
                backend="mlx",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss_weights": {"dpo": 0.7, "bco": 0.3}},
            )


# ─── Helper API ─────────────────────────────────────────────────────────────


class TestMultiObjectiveHelpers:
    def test_is_multi_objective_true(self):
        from soup_cli.trainer.preference import is_multi_objective_preference

        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        assert is_multi_objective_preference(cfg) is True

    def test_is_multi_objective_false_scalar(self):
        from soup_cli.trainer.preference import is_multi_objective_preference

        cfg = _base(preference_loss="dpo")
        assert is_multi_objective_preference(cfg) is False

    def test_is_multi_objective_false_legacy(self):
        from soup_cli.trainer.preference import is_multi_objective_preference

        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert is_multi_objective_preference(cfg) is False

    def test_get_loss_weights_returns_copy(self):
        """Defensive copy so caller mutation cannot affect cfg."""
        from soup_cli.trainer.preference import get_loss_weights

        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        weights = get_loss_weights(cfg)
        assert weights == {"dpo": 0.7, "bco": 0.3}
        weights["dpo"] = 999.0
        # Re-fetch — original cfg unchanged.
        assert get_loss_weights(cfg) == {"dpo": 0.7, "bco": 0.3}

    def test_get_loss_weights_none_when_not_set(self):
        from soup_cli.trainer.preference import get_loss_weights

        cfg = _base(preference_loss="dpo")
        assert get_loss_weights(cfg) is None


# ─── Live wiring stub-then-live ─────────────────────────────────────────────


class TestMultiObjectiveDeferred:
    def test_bco_paired_blend_rejected_at_runtime(self):
        """v0.40.1 Part B — live runtime ships, but BCO mixed with paired
        losses (DPO/SimPO/ORPO/IPO) is data-format-incompatible and must
        be rejected at setup() with a ValueError naming 'bco'.
        """
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
        with pytest.raises(ValueError, match="bco"):
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})


# ─── Docs / example contract (#2222) ─────────────────────────────────────────

_DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
_LOSS_KEYS = ("dpo", "simpo", "orpo", "ipo", "bco")
# A weights mapping in Python or YAML flow style, keys quoted or not:
# {'dpo': 0.7, 'simpo': 0.3} or {dpo: 0.7, bco: 0.3}.
_KEY = r"""['"]?(?:dpo|simpo|orpo|ipo|bco)['"]?"""
_BLEND_RE = re.compile(
    r"\{\s*" + _KEY + r"\s*:\s*[0-9.]+(?:\s*,\s*" + _KEY + r"\s*:\s*[0-9.]+)+\s*\}"
)
_PAIR_RE = re.compile(r"""['"]?(dpo|simpo|orpo|ipo|bco)['"]?\s*:\s*([0-9.]+)""")
_CONFIG_LOAD_RE = re.compile(r"config[\s-]+load")
_REFUSAL_VERB_RE = re.compile(r"\b(?:reject|refus|fail|rais)\w*")


def _parse_blend(text: str) -> dict[str, float]:
    return {key: float(value) for key, value in _PAIR_RE.findall(text)}


def _training_md() -> str:
    return (_DOCS_DIR / "training.md").read_text(encoding="utf-8")


def _sentences(markdown: str) -> list[str]:
    """Each table row on its own; prose collapsed per paragraph, then split
    into sentences, so a claim wrapped over two lines is still one sentence."""
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", markdown):
        prose = []
        for line in paragraph.splitlines():
            if line.lstrip().startswith("|"):
                out.append(line.strip())
            else:
                prose.append(line)
        joined = " ".join(" ".join(prose).split())
        out.extend(part for part in re.split(r"(?<=[.!?])\s+", joined) if part)
    return out


def _heading_slugs(markdown: str) -> set[str]:
    slugs = set()
    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*$", markdown, re.MULTILINE):
        title = re.sub(r"[^\w\s-]", "", match.group(1).strip().lower())
        slugs.add(re.sub(r"\s", "-", title))
    return slugs


def _section(markdown: str, heading: str) -> str:
    start = markdown.index(heading)
    level = heading.split(" ", 1)[0]
    rest = markdown[start + len(heading):]
    end = re.search(r"^#{1," + str(len(level)) + r"}\s", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


class TestPreferenceLossWeightsDocsAndExamples:
    """Keep preference_loss_weights examples and docs aligned with setup-time
    BCO refusal (loads at schema time; validate_weight_compat refuses later).
    """

    def test_blend_pattern_reads_quoted_and_unquoted_keys(self):
        for text in ("{'dpo': 0.7, 'bco': 0.3}", "{dpo: 0.7, bco: 0.3}"):
            match = _BLEND_RE.search(f"e.g. {text}.")
            assert match is not None
            assert _parse_blend(match.group(0)) == {"dpo": 0.7, "bco": 0.3}

    def test_schema_examples_pass_validate_weight_compat(self):
        import inspect

        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.preference_combine import validate_weight_compat

        field = TrainingConfig.model_fields["preference_loss_weights"]
        validator_src = inspect.getsource(
            SoupConfig._validate_preference_loss_weights,
        )
        examples = [
            _parse_blend(match.group(0))
            for text in (field.description or "", validator_src)
            for match in _BLEND_RE.finditer(text)
        ]
        assert examples, "expected at least one dict example in schema strings"
        for weights in examples:
            validate_weight_compat(weights)

    def test_docs_yaml_examples_pass_validate_weight_compat(self):
        import yaml

        from soup_cli.utils.preference_combine import validate_weight_compat

        def blends(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "preference_loss_weights" and isinstance(value, dict):
                        yield value
                    else:
                        yield from blends(value)
            elif isinstance(node, list):
                for item in node:
                    yield from blends(item)

        found = 0
        # config-reference.md is regenerated from the schema at release time.
        for path in sorted(_DOCS_DIR.glob("*.md")):
            if path.name == "config-reference.md":
                continue
            text = path.read_text(encoding="utf-8")
            for block in re.findall(r"```ya?ml\n(.*?)```", text, re.DOTALL):
                try:
                    data = yaml.safe_load(block)
                except yaml.YAMLError:
                    continue
                for weights in blends(data):
                    found += 1
                    validate_weight_compat(weights)
        assert found, "expected at least one preference_loss_weights YAML example"

    def test_training_md_does_not_say_bco_rejected_at_config_load(self):
        for sentence in _sentences(_training_md()):
            # Drop code spans so field names like `prompt+chosen+rejected`
            # do not count as a refusal verb.
            prose = re.sub(r"`[^`]*`", "", sentence).lower()
            if "bco" not in sentence.lower():
                continue
            if _CONFIG_LOAD_RE.search(prose) and _REFUSAL_VERB_RE.search(prose):
                raise AssertionError(
                    "docs/training.md still ties a bco blend to config-load "
                    f"refusal: {sentence!r}"
                )

    def test_sentence_split_catches_a_wrapped_claim(self):
        wrapped = (
            "Schema validation rejects `bco` mixed with a\n"
            "paired loss at config load. Next sentence.\n"
        )
        assert any(
            "bco" in s and _CONFIG_LOAD_RE.search(s) for s in _sentences(wrapped)
        )

    def test_training_md_bco_row_states_setup_time_refusal(self):
        rows = [
            line
            for line in _training_md().splitlines()
            if line.startswith("| `bco` mixed with anything |")
        ]
        assert len(rows) == 1
        row = rows[0]
        assert "refused at the start of trainer setup" in row
        assert "after the data has loaded and before the model does" in row
        assert "`validate_weight_compat`" in row
        assert "unpaired completion and label rows" in row

    def test_multi_objective_section_links_resolve(self):
        text = _training_md()
        section = _section(text, "### Multi-objective preference loss")
        anchors = re.findall(r"\]\(#([^)]+)\)", section)
        assert anchors
        slugs = _heading_slugs(text)
        for anchor in anchors:
            assert anchor in slugs, f"broken anchor #{anchor}"

    def test_field_description_says_bco_blend_refused_at_setup(self):
        from soup_cli.config.schema import TrainingConfig

        description = (
            TrainingConfig.model_fields["preference_loss_weights"].description or ""
        )
        assert "contains bco is refused when the trainer is set up" in description
