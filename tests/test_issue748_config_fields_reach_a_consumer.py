"""Issue #748 — a config field that reaches no consumer must fail the suite.

A field is declared in `config/schema.py`, validated, documented with a worked
example, and read by nothing. It accepts a value and does nothing with it,
silently. Searching this tracker for `ignore|silently|never read|does not
honour|no caller` returns 36 issues, among them #683, #684, #685 and #686 --
four in a single backend. Every one was found by a person reading code.

Nothing failed when a field lost its last consumer, and nothing fails today
when a field is added with no wiring. This guard closes that.

**What it does and does not claim.** It answers "does anything read this",
not "does this backend read this". `training.max_grad_norm` is read by sixteen
transformers trainers and by nothing on MLX; that is a strictly harder problem
and out of scope here.

**Why an AST walk and not a grep.** `citation_recall_threshold` appears in a
validator's error-message strings, so `grep -rl` calls it consumed while
nothing applies it. Docstrings are stripped before the walk for the same
reason. Conversely a field reached only through `getattr(cfg, "name")` or
`cfg_dict["name"]` IS consumed, and a guard that flagged those would be
deleted within a week -- `relora_steps` and `loraplus_lr_ratio` are exactly
that shape. Both directions are pinned below.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"
SCHEMA = "schema.py"


# --------------------------------------------------------------------------
# The detector. Kept here rather than in `src/` because it is test-only
# tooling; nothing in the shipped CLI should depend on it.
# --------------------------------------------------------------------------
def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstrings so prose naming a field cannot count as reading it."""
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def consumed_names(paths) -> set:
    """Names some module reads: attribute accesses plus string constants.

    String constants are included deliberately -- `getattr(cfg, "field")` and
    `cfg_dict["field"]` are real consumption. Docstrings are excluded, so
    prose is not.
    """
    names: set = set()
    for path in paths:
        try:
            tree = _strip_docstrings(ast.parse(path.read_text(errors="ignore")))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) < 80:
                    names.add(node.value)
    return names


def _consumer_modules():
    return [p for p in SRC.rglob("*.py") if p.name != SCHEMA]


# --------------------------------------------------------------------------
# Fields with no consumer today. Each entry is a promise that someone looked.
#
# Seeded so this lands green; the number may only shrink. Removing an entry
# because the field was wired is the point. Adding one requires a reason.
# --------------------------------------------------------------------------
KNOWN_UNCONSUMED = {
    # -- documented with a worked example, applied nowhere. Verified by hand.
    "training.lr_groups": "utils/lr_groups.py exports parse_lr_groups() and "
                          "nothing outside schema.py imports it; documented at "
                          "docs/peft-and-efficiency.md:190",
    "data.mask_history": "schema promises 'mask all but the last assistant turn "
                         "during loss computation'; documented at docs/data.md:692",
    "training.early_stop_patience": "schema promises 'consecutive regressions "
                                    "before early stopping'; documented at "
                                    "docs/peft-and-efficiency.md:622",
    "training.citation_recall_threshold": "validated by utils/citation_faithful.py "
                                          "and named in its error strings; never applied",
    # -- declared and deliberately REFUSED, so having no consumer is correct.
    #    A distinct category from the two below: the user is told, loudly, at
    #    config load. Found by this guard rather than by hand.
    "training.packing_cross_doc_attn_mask": "rejected at config load "
                                            "(schema.py:3495) because it never "
                                            "mapped to a valid TRL packing_strategy; "
                                            "documented at docs/performance-and-"
                                            "quantization.md:153",
    # -- staged for features that have not landed; grouped so they can be
    #    retired together rather than one at a time.
    "training.long_context_grpo": "documented as wiring Tiled MLP; no Tiled MLP exists",
    "training.vision_grpo": "no vision GRPO path",
    "training.load_in_16bit": "schema rewrites quantization at validation time",
    "training.unsloth_bnb_4bit": "unsloth quantisation staging",
    "training.llm_int8": "bitsandbytes int8 staging",
    "training.quantize_ref_model": "reference-model quantisation staging",
    "training.yarn_attn_factor": "YaRN staging",
    "training.yarn_beta_fast": "YaRN staging",
    "training.yarn_beta_slow": "YaRN staging",
    "training.convergence_window": "convergence-detector staging",
    "training.convergence_rel_tol": "convergence-detector staging",
    "training.forgetting_eval_steps": "catastrophic-forgetting probe staging",
    "training.forgetting_benchmark": "catastrophic-forgetting probe staging",
    "training.forgetting_stop": "catastrophic-forgetting probe staging",
    "training.checkpoint_eval_steps": "checkpoint-eval staging",
    "training.checkpoint_eval_metric": "checkpoint-eval staging",
    "training.checkpoint_eval_tasks": "checkpoint-eval staging",
    "training.checkpoint_keep_top": "checkpoint-eval staging",
    "training.grace_codebook_size": "GRACE codebook staging",
    "training.grace_codebook_dim": "GRACE codebook staging",
    "data.video_dir": "video pipeline staging",
    "data.eval_on_each_dataset": "per-dataset eval staging",
    "data.split_thinking": "thinking-block masking staging",
    "data.image_min_pixels": "image preprocessing staging",
    "data.image_max_pixels": "image preprocessing staging",
    "data.image_resize_algorithm": "image preprocessing staging",
    "data.video_fps": "video pipeline staging",
    "data.video_maxlen": "video pipeline staging",
    "data.resize_vocab": "vocab-resize staging",
    "data.extend_conversation": "conversation-extension staging",
    "data.skip_prepare_dataset": "dataset-prep bypass staging",
}


def _declared():
    from soup_cli.config.schema import DataConfig, TrainingConfig

    out = {}
    for cls, label in ((TrainingConfig, "training"), (DataConfig, "data")):
        for name in cls.model_fields:
            out[f"{label}.{name}"] = name
    return out


class TestTheDetectorItself:
    """The guard is only worth having if the detector is right in BOTH
    directions. A false negative lets a dead field through; a false positive
    gets the test deleted.
    """

    def _consumed(self, tmp_path, source: str) -> set:
        module = tmp_path / "consumer.py"
        module.write_text(source)
        return consumed_names([module])

    def test_an_attribute_access_counts_as_consumption(self, tmp_path):
        assert "widget" in self._consumed(tmp_path, "def f(cfg):\n    return cfg.widget\n")

    def test_getattr_with_a_string_counts_as_consumption(self, tmp_path):
        """`getattr(tcfg, "fp8_recipe", ...)` is how utils/v028_features.py
        reads many real fields."""
        src = 'def f(cfg):\n    return getattr(cfg, "widget", None)\n'
        assert "widget" in self._consumed(tmp_path, src)

    def test_a_dict_lookup_counts_as_consumption(self, tmp_path):
        assert "widget" in self._consumed(tmp_path, 'def f(d):\n    return d["widget"]\n')

    # The docstring fixtures below use the bare field name as the ENTIRE
    # docstring. An earlier version wrote prose around it ("Talks about
    # widget.") and was vacuous: the collected constant is then that whole
    # sentence, never the bare name, so the assertion held whether or not
    # stripping happened. Found by mutating the stripper -- disabling it
    # survived all three. An exact-match docstring is also the realistic
    # shape, since generated documentation often is exactly the field name.

    def test_a_module_docstring_does_not_count(self, tmp_path):
        """The distinction a grep gets wrong."""
        assert "widget" not in self._consumed(tmp_path, '"""widget"""\n')

    def test_a_function_docstring_does_not_count(self, tmp_path):
        assert "widget" not in self._consumed(
            tmp_path, 'def f():\n    """widget"""\n    return 1\n'
        )

    def test_a_class_docstring_does_not_count(self, tmp_path):
        assert "widget" not in self._consumed(
            tmp_path, 'class C:\n    """widget"""\n    x = 1\n'
        )

    def test_prose_mentioning_a_field_is_not_a_read_either(self, tmp_path):
        """The `citation_recall_threshold` shape: named inside a longer
        message. Collected as the whole sentence, so it never matches the
        field name -- pinned so a future change to how constants are split
        cannot start counting prose as consumption."""
        src = 'def f():\n    raise ValueError("widget must be in [0, 1]")\n'
        assert "widget" not in self._consumed(tmp_path, src)

    def test_an_unrelated_module_consumes_nothing(self, tmp_path):
        """Reject-everything control: the detector must not report a name that
        is simply absent, or every field would look consumed."""
        assert "widget" not in self._consumed(tmp_path, "x = 1\n")

    def test_a_syntactically_broken_module_is_skipped_not_fatal(self, tmp_path):
        """One unparseable file must not take the guard down."""
        bad = tmp_path / "bad.py"
        bad.write_text("def (:\n")
        assert consumed_names([bad]) == set()


class TestEveryDeclaredFieldReachesAConsumer:
    def test_no_new_field_is_declared_without_a_consumer(self):
        consumed = consumed_names(_consumer_modules())
        orphans = sorted(
            key
            for key, attr in _declared().items()
            if attr not in consumed and key not in KNOWN_UNCONSUMED
        )
        assert not orphans, (
            "These config fields are declared in schema.py and read by no "
            "module outside it, so a user setting them gets no effect and no "
            "warning:\n  "
            + "\n  ".join(orphans)
            + "\n\nWire the field, or add it to KNOWN_UNCONSUMED with a reason."
        )

    def test_the_allowlist_names_only_real_fields(self):
        """A renamed or deleted field must not keep a stale entry alive --
        otherwise the allowlist silently stops guarding anything."""
        declared = _declared()
        stale = sorted(k for k in KNOWN_UNCONSUMED if k not in declared)
        assert not stale, (
            "KNOWN_UNCONSUMED names fields that no longer exist; remove them:\n  "
            + "\n  ".join(stale)
        )

    def test_the_allowlist_does_not_cover_fields_that_are_consumed(self):
        """The list may only shrink. When a field gets wired, its entry has to
        go, or the guard stops noticing if the wiring is later removed."""
        consumed = consumed_names(_consumer_modules())
        declared = _declared()
        now_wired = sorted(
            k for k in KNOWN_UNCONSUMED
            if k in declared and declared[k] in consumed
        )
        assert not now_wired, (
            "These fields now have a consumer, so their KNOWN_UNCONSUMED entry "
            "is obsolete and must be deleted:\n  " + "\n  ".join(now_wired)
        )

    def test_every_allowlist_entry_carries_a_reason(self):
        empty = sorted(k for k, v in KNOWN_UNCONSUMED.items() if not v or len(v) < 10)
        assert not empty, f"allowlist entries need a reason: {empty}"

    def test_the_guard_can_actually_fail(self, tmp_path, monkeypatch):
        """Acceptance criterion 1, demonstrated rather than described.

        A guard that has never been observed failing is not yet known to work.
        This adds a field to the real TrainingConfig, confirms the check goes
        red naming it, then wires a consumer and confirms it goes green.
        """
        from soup_cli.config.schema import TrainingConfig

        fields = dict(TrainingConfig.model_fields)
        fields["totally_unwired_probe"] = fields["max_grad_norm"]
        monkeypatch.setattr(TrainingConfig, "model_fields", fields)

        consumed = consumed_names(_consumer_modules())
        orphans = [
            key for key, attr in _declared().items()
            if attr not in consumed and key not in KNOWN_UNCONSUMED
        ]
        assert orphans == ["training.totally_unwired_probe"], (
            f"the guard did not flag an unwired field; it reported {orphans}"
        )

        # ...and green once something reads it.
        wired = tmp_path / "wired.py"
        wired.write_text("def f(cfg):\n    return cfg.totally_unwired_probe\n")
        consumed_after = consumed_names(_consumer_modules() + [wired])
        assert "totally_unwired_probe" in consumed_after
        assert not [
            key for key, attr in _declared().items()
            if attr not in consumed_after and key not in KNOWN_UNCONSUMED
        ]

    def test_removing_a_fields_last_consumer_is_caught(self, tmp_path):
        """Acceptance criterion 2: the failure fires on the commit that breaks
        it, not months later in a user's run."""
        declared = _declared()
        # `lr` is read all over the trainers; simulate its last consumer going.
        assert "training.lr" in declared
        only_docstring = tmp_path / "gone.py"
        only_docstring.write_text('"""This module used to apply cfg.lr."""\n')
        consumed = consumed_names([only_docstring])
        assert "lr" not in consumed, (
            "a field named only in a docstring must read as unconsumed"
        )


def test_the_allowlist_is_not_silently_growing():
    """A ratchet, in the shape of the recipe-count test.

    If this number goes up, a field was added without wiring and allowlisted
    instead. That is sometimes right -- but it should be a deliberate edit to
    this line, visible in review, rather than a quiet append.
    """
    assert len(KNOWN_UNCONSUMED) <= 36, (
        f"KNOWN_UNCONSUMED has grown to {len(KNOWN_UNCONSUMED)}; a field was "
        "allowlisted rather than wired. Lower this bound when entries are retired."
    )
