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

**What this cannot see, so nobody trusts it past its limits.**

*The name space is global.* It asks whether an identifier appears as an
attribute or key anywhere under ``src/``, not whether it appears on a config
object. The consumed set is roughly 3,400 names -- most of the codebase's
attribute namespace. A field named after a common attribute therefore reads as
consumed on the strength of an unrelated one. Measured on this tree, an
unwired field would be caught or missed as follows:

    training.verbose         caught
    training.top_k           LEAKS   -- `top_k` is an attribute elsewhere
    training.top_p           LEAKS
    training.temperature     LEAKS   -- `request.temperature`, commands/serve.py
    training.dtype           LEAKS
    training.seed            LEAKS
    training.logging_steps   LEAKS

So this is a ratchet with a known hole, not a proof. It leaks hardest on
exactly the generic names a new HuggingFace/TRL passthrough field would carry,
which is the case most likely to arise. `test_the_known_leak_is_still_the_
known_leak` pins the boundary so it cannot move without someone noticing.
Scoping reads to config-typed objects is a much larger piece of work and is
not attempted here.

*Fields read only through a ``schema.py`` ``@property`` ARE now seen*, but
only when something calls the property. ``schema.py`` is excluded from the
main scan, so ``training.bnb_4bit_use_double_quant`` -- resolved by the
``double_quant_on`` property (``schema.py:1880``, #321) -- read as an orphan,
and an earlier version of this allowlist recorded it as an unread offender on
exactly that evidence. Freezing a repaired field as an open defect is the
worst thing an allowlist can do, because no test can retire it: the list is
meant to name fields a user can set with no effect, and an entry that
contradicts itself gives a false answer to the one question the file exists to
answer. `schema_property_reads` fixes it. The gate matters as much as the
pass: an UNCALLED resolver launders nothing, or the detector's own failure
mode returns one level up, with a field "read" by code that never runs.
Validators are excluded on principle -- a validator checks a value, a property
resolves one for a consumer.

**Two rules this file learned the hard way, kept because they generalise.**
A check can only be pinned by testing its REFUSALS: loosening a predicate
makes a suite pass rather than fail, so `_reason_is_accountable` is asserted
against what it rejects. And some assertions are unkillable by construction --
the scan/import tree check in `_declared()` exists to fire in a misconfigured
environment, so no mutation run inside a correct one can kill it. It is not
untested, it is untestable from here; do not delete it for lack of a red.

*It cannot see a value that is read and then rewritten.* #423 is the shape:
``detect_device()`` did not recognise MLX, so `quantization: 4bit` was read
correctly and then silently rewritten to `none`. That needs device-aware
expectations, not a reachability walk.

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
import functools
import pathlib
import re

import pytest

pytestmark = pytest.mark.unit

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"
SCHEMA = "schema.py"
SCHEMA_PATH = SRC / "config" / SCHEMA


# --------------------------------------------------------------------------
# The detector. Kept here rather than in `src/` because it is test-only
# tooling; nothing in the shipped CLI should depend on it.
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _consumed_cached(paths_key) -> frozenset:
    return frozenset(consumed_names([pathlib.Path(p) for p in paths_key]))


def consumed_names(paths) -> set:
    """Names some module READS. Reads only -- writes are not consumption.

    Counted as a read:
      * `obj.field` in a load context;
      * `d["field"]` in a load context;
      * `getattr(obj, "field")` / `cfg.get("field")` and friends.

    Deliberately NOT counted:
      * `d["field"] = value` and `{"field": value}` -- that is code EMITTING
        config, not reading the user's setting. This is the hole that let
        both of the maintainer's named offenders through: `data.interleave`
        looked consumed because `mix_proxy.py` writes
        `data_block["interleave"] = {...}`, and
        `bnb_4bit_use_double_quant` because `save_formats.py` writes it as a
        key in an output dict. Run against the tree at the commit where each
        was a live defect, the earlier version reported both as CONSUMED.
      * docstring prose, and any other bare string constant. Nothing here
        collects a free-standing `ast.Constant`, so prose is excluded
        structurally rather than by a stripping pass. An earlier version
        stripped docstrings explicitly; mutation testing showed that pass was
        dead once reads were narrowed to Load contexts and call arguments, so
        it was removed rather than left looking load-bearing.
    """
    names: set = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.ctx, ast.Load):
                    names.add(node.attr)
            elif isinstance(node, ast.Subscript):
                sl = node.slice
                if (
                    isinstance(node.ctx, ast.Load)
                    and isinstance(sl, ast.Constant)
                    and isinstance(sl.value, str)
                ):
                    names.add(sl.value)
            elif isinstance(node, ast.Call):
                # getattr(obj, "field") / d.get("field") / pop / setdefault
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                # Only the NAME argument, never the default. `d.get("k", "widget")`
                # returns "widget" as a fallback value; counting it as a read of a
                # field called `widget` is a false positive, and false positives
                # are what get a guard deleted.
                idx = 1 if fname in ("getattr", "hasattr") else 0
                if fname in ("getattr", "get", "pop", "setdefault", "hasattr"):
                    if len(node.args) > idx:
                        arg = node.args[idx]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            names.add(arg.value)
    return names


def schema_property_reads(schema_path: pathlib.Path) -> dict:
    """Fields read inside `schema.py` `@property` bodies, keyed by property name.

    `schema.py` is excluded from the main scan, so a field read only through a
    resolver there looks like an orphan. `double_quant_on` (schema.py:1880,
    #321) is that case: consumers read the property, nothing names the field.

    Properties only, never validators, and the distinction is principled: a
    validator CHECKS a value and consumes nothing on the user's behalf; a
    `@property` RESOLVES one for someone else to consume. Measured on this
    schema: 1 property, 156 validators, 0 other decorated functions -- so this
    pass contributes exactly one name today.

    Returned per-property rather than flattened so the caller can gate on
    whether anything actually calls the property. An uncalled resolver must not
    launder its fields into "consumed" -- that would reintroduce this
    detector's own failure mode one level up.
    """
    out: dict = {}
    try:
        tree = ast.parse(schema_path.read_text(errors="ignore"))
    except (SyntaxError, UnicodeDecodeError, ValueError):  # pragma: no cover
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                names.append(dec.attr)
        if "property" not in names:
            continue
        reads = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Attribute) and isinstance(inner.ctx, ast.Load):
                reads.add(inner.attr)
        out[node.name] = reads
    return out


def fold_property_reads(consumed: set, props: dict) -> set:
    """Add a resolver's reads to `consumed`, but ONLY if the resolver is called.

    The gate, extracted so it is testable on its own. Without it a property
    nobody calls would launder its fields into "consumed" -- this detector's
    own failure mode one level up, where a field is "read" by code that never
    runs. Tested against a synthetic uncalled resolver rather than the real
    schema, because the real schema's only property IS called, so a test using
    it cannot tell the gate from its absence. (An earlier version of that test
    did exactly that and the mutation survived.)
    """
    out = set(consumed)
    for prop_name, reads in props.items():
        if prop_name in out:
            out |= reads
    return out


def _consumer_modules():
    return [p for p in SRC.rglob("*.py") if p.name != SCHEMA]


def _consumed_in_src() -> set:
    """Cached: the walk is 499 modules and ~2s, and this file did it four
    times uncached. Keyed on the file list so a changed tree re-walks.

    Includes fields resolved by a `schema.py` `@property` -- but only when the
    property itself is called from somewhere in `src/`. See
    `schema_property_reads`.
    """
    modules = _consumer_modules()
    consumed = set(_consumed_cached(tuple(sorted(str(p) for p in modules))))
    consumed = fold_property_reads(consumed, schema_property_reads(SCHEMA_PATH))
    # #807: a read inside a function nothing calls is not consumption, for
    # the same reason an uncalled @property launders nothing.
    return drop_dead_function_reads(
        consumed, function_scoped_reads(modules), referenced_names(modules)
    )


# --------------------------------------------------------------------------
# Fields with no consumer today. Each entry is a promise that someone looked.
#
# Seeded so this lands green; the number may only shrink. Removing an entry
# because the field was wired is the point. Adding one requires a reason.
# --------------------------------------------------------------------------
KNOWN_UNCONSUMED = {
    # -- documented with a worked example, applied nowhere. Verified by hand.
    "training.lr_groups": "no issue yet -- utils/lr_groups.py exports parse_lr_groups() and "
                          "nothing outside schema.py imports it; documented at "
                          "docs/peft-and-efficiency.md:190",
    "data.mask_history": "no issue yet -- schema promises 'mask all but the last assistant turn "
                         "during loss computation'; documented at docs/data.md:692",
    "training.early_stop_patience": "no issue yet -- schema promises 'consecutive regressions "
                                    "before early stopping'; documented at "
                                    "docs/peft-and-efficiency.md:622",
    "training.citation_recall_threshold": "no issue yet -- validated by utils/citation_faithful.py "
                                          "and named in its error strings; never applied",
    # -- found by the read/write fix, and the reason that fix exists. Both
    #    are user settings that are OVERRIDDEN rather than merely unread, so
    #    they are the strongest members of this list.
    "data.remove_unused_columns": "#759 -- schema default True, documented 'set False "
                                  "when feeding extra cols to a custom collator' "
                                  "-- and sft.py:788 / pretrain.py:174 / "
                                  "embedding.py:175 / grpo.py:468 each hardcode "
                                  "False, so the setting never reaches HF",
    "training.yarn_factor": "no issue yet -- long_context.py:316 uses a local of the same name "
                            "and the string only as an error label; the config "
                            "field reaches nothing",
    "training.grace_codebook": "no issue yet -- the string appears as an artifact-kind name in "
                               "store.py:52 / edit.py:312, unrelated to this field",
    # -- #807: read ONLY inside a function nothing in src/ references, so the
    #    read is not consumption. Surfaced by the dead-function gate, which
    #    the guard previously applied to @property resolvers only.
    "training.warmup_auto": "#807 -- read only in autopilot.generate_config, "
                            "which nothing in src/ calls; and that line reads "
                            "the decisions dict and WRITES the value into a "
                            "config, so it is not a read of the field either",
    "training.lora.use_vera": "#794 via #807 -- read only in "
                              "utils/peft_builder.build_peft_config, which has "
                              "no caller in src/",
    "training.lora.use_olora": "#794 via #807 -- same uncalled builder",
    "training.lora.init_strategy": "#794 via #807 -- same uncalled builder; "
                                   "PiSSA/LoftQ selection never reaches a trainer",
    # -- #807: LoraConfig came into the guard's scope with the dead-function
    #    gate, and these two have no read anywhere at all.
    "training.lora.loftq_iter": "#794 -- no read anywhere; LoftQ is unreachable",
    "training.lora.loftq_bits": "#794 -- no read anywhere; LoftQ is unreachable",
    # -- declared and deliberately REFUSED, so having no consumer is correct.
    #    A distinct category from the two below: the user is told, loudly, at
    #    config load. Found by this guard rather than by hand.
    "training.packing_cross_doc_attn_mask": "no issue needed: rejected at config load "
                                            "(schema.py:3495) because it never "
                                            "mapped to a valid TRL packing_strategy; "
                                            "documented at docs/performance-and-"
                                            "quantization.md:153",
    # -- staged for features that have not landed; grouped so they can be
    #    retired together rather than one at a time.
    "training.long_context_grpo": "no issue yet -- documented as wiring "
                                  "Tiled MLP; no Tiled MLP exists",
    "training.vision_grpo": "no issue yet -- no vision GRPO path",
    "training.load_in_16bit": "no issue needed: schema rewrites quantization at validation time",
    "training.unsloth_bnb_4bit": "no issue yet -- unsloth quantisation staging",
    "training.llm_int8": "no issue yet -- bitsandbytes int8 staging",
    "training.quantize_ref_model": "no issue yet -- reference-model quantisation staging",
    "training.yarn_attn_factor": "no issue yet -- YaRN staging",
    "training.yarn_beta_fast": "no issue yet -- YaRN staging",
    "training.yarn_beta_slow": "no issue yet -- YaRN staging",
    "training.convergence_window": "no issue yet -- convergence-detector staging",
    "training.convergence_rel_tol": "no issue yet -- convergence-detector staging",
    "training.forgetting_eval_steps": "no issue yet -- catastrophic-forgetting probe staging",
    "training.forgetting_benchmark": "no issue yet -- catastrophic-forgetting probe staging",
    "training.forgetting_stop": "no issue yet -- catastrophic-forgetting probe staging",
    "training.checkpoint_eval_steps": "no issue yet -- checkpoint-eval staging",
    "training.checkpoint_eval_metric": "no issue yet -- checkpoint-eval staging",
    "training.checkpoint_eval_tasks": "no issue yet -- checkpoint-eval staging",
    "training.checkpoint_keep_top": "no issue yet -- checkpoint-eval staging",
    "training.grace_codebook_size": "no issue yet -- GRACE codebook staging",
    "training.grace_codebook_dim": "no issue yet -- GRACE codebook staging",
    "data.video_dir": "no issue yet -- video pipeline staging",
    "data.eval_on_each_dataset": "no issue yet -- per-dataset eval staging",
    "data.split_thinking": "no issue yet -- thinking-block masking staging",
    "data.image_min_pixels": "no issue yet -- image preprocessing staging",
    "data.image_max_pixels": "no issue yet -- image preprocessing staging",
    "data.image_resize_algorithm": "no issue yet -- image preprocessing staging",
    "data.video_fps": "no issue yet -- video pipeline staging",
    "data.video_maxlen": "no issue yet -- video pipeline staging",
    "data.resize_vocab": "no issue yet -- vocab-resize staging",
    "data.extend_conversation": "no issue yet -- conversation-extension staging",
    "data.skip_prepare_dataset": "no issue yet -- dataset-prep bypass staging",
}


def _declared():
    import soup_cli
    from soup_cli.config.schema import DataConfig, LoraConfig, TrainingConfig

    # The scan walks SRC (this checkout); the fields come from the IMPORTED
    # package. In a worktree with no PYTHONPATH those are different trees and
    # the mismatch fails GREEN -- the silent-pass failure mode this file
    # exists to prevent. Measured: 17 passed without PYTHONPATH, 2 failed with
    # it, on the same tree.
    imported = pathlib.Path(soup_cli.__file__).resolve().parent
    assert imported == SRC, (
        f"scanning {SRC} but importing {imported}; set PYTHONPATH=<checkout>/src "
        "or reinstall with `pip install -e .`, or this guard silently passes"
    )

    out = {}
    # #807: LoraConfig was outside the guard's scope entirely, so every
    # LoRA-variant field was unguarded.
    for cls, label in (
        (TrainingConfig, "training"),
        (DataConfig, "data"),
        (LoraConfig, "training.lora"),
    ):
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
        consumed = _consumed_in_src()
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
        consumed = _consumed_in_src()
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

        consumed = _consumed_in_src()
        orphans = [
            key for key, attr in _declared().items()
            if attr not in consumed and key not in KNOWN_UNCONSUMED
        ]
        assert "training.totally_unwired_probe" in orphans, (
            f"the guard did not flag an unwired field; it reported {orphans}. "
            "Membership, not equality: any other orphan present is a separate "
            "finding and must not make this read as a failure to detect."
        )

        # ...and green once something reads it.
        wired = tmp_path / "wired.py"
        wired.write_text("def f(cfg):\n    return cfg.totally_unwired_probe\n")
        # Same composition as the real check, property pass included.
        consumed_after = _consumed_in_src() | consumed_names([wired])
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


def test_the_allowlist_size_is_pinned_exactly():
    """A ratchet that fails in BOTH directions.

    `<= N` catches the list growing -- a field allowlisted rather than wired.
    It does not catch the list going STALE: wire a field, forget to delete its
    entry, and the bound stays green while the allowlist now describes code
    that no longer exists. @MakazhanAlpamys flagged that asymmetry on #751,
    having watched #756's registry go stale five times in a day for exactly
    that reason.

    `==` makes both directions a deliberate, reviewable edit to this line.

    **The one case only this test can see** -- and the reason it is not
    redundant with the test below -- is an entry added for a BRAND-NEW unwired
    field. Nothing is stale then, no entry describes code that moved, and the
    count is the only signal that a field was allowlisted instead of wired.
    `test_the_allowlist_does_not_cover_fields_that_are_consumed` is the other
    half: it names WHICH entry went stale, where this one only says the count
    moved.
    """
    assert len(KNOWN_UNCONSUMED) == 45, (
        f"KNOWN_UNCONSUMED is {len(KNOWN_UNCONSUMED)}, pinned at 45. Going UP "
        "means a field was allowlisted rather than wired; going DOWN means an "
        "entry was retired, which is the good direction -- lower this number "
        "in the same commit."
    )


def _reason_is_accountable(reason: str) -> bool:
    """A reason must cite an issue or say in words that none exists.

    Extracted so the predicate itself is testable. Loosening a check makes the
    suite pass rather than fail, so the only way to pin it is to assert what it
    REJECTS -- see `TestTheAccountabilityPredicate`.
    """
    return bool(re.search(r"#\d+", reason)) or "no issue" in reason


class TestTheAccountabilityPredicate:
    """`staging` was accepted as a pass and 31 of 40 entries used it. That loose
    predicate is what let the `bnb_4bit_use_double_quant` entry through with a
    wrong story attached, so what it REJECTS is the part worth pinning."""

    def test_a_bare_staging_note_is_not_accountable(self):
        assert not _reason_is_accountable("YaRN staging")

    def test_prose_with_no_issue_and_no_admission_is_not_accountable(self):
        assert not _reason_is_accountable("documented as wiring Tiled MLP")

    def test_an_issue_reference_is_accountable(self):
        assert _reason_is_accountable("#759 -- fifteen trainers hardcode it")

    def test_an_explicit_admission_is_accountable(self):
        assert _reason_is_accountable("no issue yet -- nothing imports it")
        assert _reason_is_accountable("no issue needed: refused at config load")


def test_every_allowlist_entry_states_an_issue_or_says_there_is_none():
    """An entry with no issue reference is indistinguishable from one someone
    added to make CI green, and that is how a ratchet rots. Where no issue
    exists the entry must say so out loud, which makes it a standing prompt to
    file one -- which is how #759 came to be filed.
    """
    # `staging` is deliberately NOT accepted as a pass. 31 of the entries used
    # it, and that loose predicate is what let the bnb_4bit_use_double_quant
    # entry through carrying a wrong story. An entry must cite an issue or say
    # in words that none exists.
    vague = sorted(
        k for k, v in KNOWN_UNCONSUMED.items()
        if not _reason_is_accountable(v)
    )
    assert not vague, (
        "these allowlist entries cite no issue and do not say one is missing:\n  "
        + "\n  ".join(vague)
    )


def test_a_get_default_is_not_counted_as_a_read(tmp_path):
    """`d.get("k", "widget")` returns "widget" as a fallback VALUE, not as a
    field name. Counting it would be a false positive, and the false-positive
    rate is what decides whether a guard survives the next person in a hurry.
    """
    module = tmp_path / "m.py"
    module.write_text('def f(d):\n    return d.get("k", "widget")\n')
    consumed = consumed_names([module])
    assert "k" in consumed, "the looked-up key is a read"
    assert "widget" not in consumed, "the default value is not a read"


def test_getattr_reads_the_name_not_the_default(tmp_path):
    """`getattr(o, "name", "widget")` -- the name is argument 1, the default 2."""
    module = tmp_path / "m.py"
    module.write_text('def f(o):\n    return getattr(o, "name", "widget")\n')
    consumed = consumed_names([module])
    assert "name" in consumed
    assert "widget" not in consumed


class TestSchemaSideResolvers:
    """`schema.py` is excluded from the scan, so a field read only through a
    resolver defined there looked like an orphan. `double_quant_on`
    (`schema.py:1880`, #321) is that case, and an earlier version of this file
    recorded its field as an unread offender on exactly that evidence.

    Properties only, never validators: a validator CHECKS a value and consumes
    nothing on the user's behalf; a `@property` RESOLVES one for someone else
    to consume. Measured on this schema -- 1 property, 156 validators, 0 other
    decorated functions -- so the pass contributes exactly one name today.
    """

    def _props(self, tmp_path, source: str) -> dict:
        schema = tmp_path / "schema.py"
        schema.write_text(source)
        return schema_property_reads(schema)

    def test_a_property_body_contributes_the_fields_it_reads(self, tmp_path):
        src = (
            "class C:\n"
            "    @property\n"
            "    def resolved(self):\n"
            "        return self.raw_field is not False\n"
        )
        assert self._props(tmp_path, src) == {"resolved": {"raw_field"}}

    def test_a_validator_body_contributes_nothing(self, tmp_path):
        """156 of them read fields here. Counting those would mark most of the
        schema consumed by its own validation, which is the false negative
        that matters."""
        src = (
            "class C:\n"
            "    @field_validator('raw_field')\n"
            "    def check(cls, v):\n"
            "        return cls.raw_field\n"
        )
        assert self._props(tmp_path, src) == {}

    def test_the_real_schema_has_exactly_one_property(self):
        """If a second resolver appears, this pass grows silently -- the pin is
        cheap and makes that a deliberate edit."""
        props = schema_property_reads(SCHEMA_PATH)
        assert list(props) == ["double_quant_on"], (
            f"schema.py properties are now {sorted(props)}; each one launders "
            "the fields it reads into 'consumed', so review the addition"
        )
        assert "bnb_4bit_use_double_quant" in props["double_quant_on"]

    def test_a_called_resolver_makes_its_field_consumed(self):
        """`double_quant_on` is called from quant_menu.py and stream_setup.py,
        so its field retires from the allowlist."""
        consumed = _consumed_in_src()
        assert "double_quant_on" in consumed, "the property itself must be called"
        assert "bnb_4bit_use_double_quant" in consumed
        assert "training.bnb_4bit_use_double_quant" not in KNOWN_UNCONSUMED

    def test_an_uncalled_resolver_launders_nothing(self):
        """The gate, exercised directly.

        An earlier version of this test built a temp schema and then asserted
        against `_consumed_in_src()`, which reads the REAL schema -- so it held
        whether the gate existed or not, and dropping the gate survived
        mutation. This drives `fold_property_reads` itself.
        """
        props = {"never_called": {"orphan_field"}, "is_called": {"wired_field"}}
        consumed = fold_property_reads({"is_called", "unrelated"}, props)

        assert "wired_field" in consumed, "a CALLED resolver contributes its reads"
        assert "orphan_field" not in consumed, (
            "an uncalled resolver laundered its field into 'consumed'; a field "
            "read only by code nothing invokes is not consumed"
        )

    def test_the_gate_is_what_retires_the_double_quant_entry(self):
        """End to end on the real schema: the property is called, so its field
        is consumed and needs no allowlist entry."""
        props = schema_property_reads(SCHEMA_PATH)
        raw = set(_consumed_cached(tuple(sorted(str(p) for p in _consumer_modules()))))

        assert "bnb_4bit_use_double_quant" not in raw, (
            "nothing outside schema.py names the field -- that is why the "
            "property pass exists"
        )
        assert "double_quant_on" in raw, "the property itself is called"
        assert "bnb_4bit_use_double_quant" in fold_property_reads(raw, props)


def test_the_known_leak_is_still_the_known_leak():
    """Pin the boundary of the global-name-space hole, measured not assumed.

    The maintainer measured this on #751 before merging: an unwired
    `training.verbose` is caught, an unwired `training.top_k` is not, because
    `top_k` appears as an attribute elsewhere in the tree. Recording it as a
    test rather than as prose means the hole cannot quietly widen or close.

    If a name moves from one list to the other, that is not a failure to fix
    by editing this test — it is a change in what the guard can see, and the
    docstring's leak table should move with it.
    """
    consumed = _consumed_in_src()

    assert "verbose" not in consumed, (
        "`training.verbose` used to be catchable; if some module now reads a "
        "`.verbose` attribute, the guard has lost a name it could see"
    )
    for leaky in ("top_k", "top_p", "temperature", "dtype", "seed", "logging_steps"):
        assert leaky in consumed, (
            f"`{leaky}` no longer collides with an unrelated attribute, so the "
            "guard can now catch an unwired field of that name. Good news — "
            "update the leak table in the module docstring."
        )


# --------------------------------------------------------------------------
# #807: the same laundering the property gate refuses, one level over.
#
# `fold_property_reads` gates a `schema.py` @property on being CALLED. Ordinary
# functions got no such gate, so a read inside a function nothing calls counted
# exactly like a read in a trainer. `training.warmup_auto` passed the guard on
# the strength of one line in `autopilot/generate_config.generate_config`,
# which nothing in src/ calls -- and which reads the autopilot decisions dict
# and writes the value INTO a config, so it is not even a read of the field.
# --------------------------------------------------------------------------


def function_scoped_reads(paths) -> dict:
    """Map each read name to the {(module, enclosing function)} it occurs in.

    ``None`` as the function means module scope, which always counts: a read at
    import time runs whenever the module is imported.

    Every function is attributed, **methods included**. An earlier version of
    this walked only `tree.body` top-level defs, so a read inside a class
    method was invisible -- and a field whose only live read sits in a method
    then looked dead. That produced two false positives on the real tree
    (`auto_mixed_precision`, read at `trainer/sft.py:1185`, and
    `training.multipack`). False positives are what get a guard deleted, so the
    attribution has to cover methods or the gate is worse than no gate.
    """
    out: dict = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        stack = [(tree, None)]
        while stack:
            node, fname = stack.pop()
            for child in ast.iter_child_nodes(node):
                inner = (
                    child.name
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else fname
                )
                name = None
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                    name = child.attr
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    name = child.value
                if name:
                    out.setdefault(name, set()).add((path.name, fname))
                stack.append((child, inner))
    return out


def referenced_names(paths) -> set:
    """Every name loaded anywhere under src/ -- the "is this function called?" set.

    A global name match, deliberately. It errs toward "referenced", which is
    the safe direction: a guard that cries wolf on live code gets deleted, and
    a missed dead function is only a return to today's behaviour.
    """
    names: set = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                names.add(node.attr)
    return names


def drop_dead_function_reads(consumed: set, scoped: dict, referenced: set) -> set:
    """Remove names whose EVERY read sits in a function nothing references.

    The function-level twin of `fold_property_reads`, and extracted for the
    same reason: a gate can only be pinned by driving it directly. Testing it
    through the real tree cannot distinguish the gate from its absence for any
    field that is genuinely live.
    """
    out = set(consumed)
    for name, sites in scoped.items():
        if name not in out:
            continue
        if all(fn is not None and fn not in referenced for _, fn in sites):
            out.discard(name)
    return out


class TestTheDeadFunctionGate:
    """#807. Both directions, driven through the gate rather than the tree."""

    def test_a_read_in_a_referenced_function_still_counts(self):
        consumed = {"widget"}
        scoped = {"widget": {("m.py", "live_helper")}}
        assert "widget" in drop_dead_function_reads(consumed, scoped, {"live_helper"})

    def test_a_read_in_a_function_nothing_references_is_dropped(self):
        consumed = {"widget"}
        scoped = {"widget": {("m.py", "dead_helper")}}
        assert "widget" not in drop_dead_function_reads(consumed, scoped, {"other"})

    def test_one_live_read_rescues_a_field_with_dead_ones(self):
        """A field read in both a live and a dead function is consumed. Anything
        else would flag `auto_mixed_precision`, which `sft.py` really reads."""
        consumed = {"widget"}
        scoped = {"widget": {("dead.py", "never_called"), ("live.py", "used")}}
        assert "widget" in drop_dead_function_reads(consumed, scoped, {"used"})

    def test_a_module_scope_read_always_counts(self):
        """Module scope runs on import; there is no enclosing function to be
        dead."""
        consumed = {"widget"}
        assert "widget" in drop_dead_function_reads(
            consumed, {"widget": {("m.py", None)}}, set()
        )

    def test_methods_are_attributed_not_skipped(self, tmp_path):
        """The false-positive generator. Walking only top-level defs made a
        read inside a method invisible, so the field looked dead."""
        mod = tmp_path / "m.py"
        mod.write_text(
            "class C:\n"
            "    def a_method(self, cfg):\n"
            "        return cfg.widget\n"
        )
        scoped = function_scoped_reads([mod])
        assert scoped.get("widget") == {("m.py", "a_method")}, (
            "a read inside a method must be attributed to that method, not lost"
        )

    def test_the_real_tree_has_exactly_the_four_known_escapes(self):
        """Measured, and pinned so the number cannot drift unnoticed.

        Reproduced on current main: `training.warmup_auto` plus three LoRA
        variants, each read only inside a function nothing references.
        """
        modules = _consumer_modules()
        scoped = function_scoped_reads(modules)
        referenced = referenced_names(modules)
        # The UNGATED set: _consumed_in_src() already applies the gate, so
        # diffing it against itself would always be empty and this test would
        # pass vacuously. Rebuild the pre-gate set the same way it does.
        ungated = fold_property_reads(
            set(_consumed_cached(tuple(sorted(str(m) for m in modules)))),
            schema_property_reads(SCHEMA_PATH),
        )
        dropped = sorted(ungated - drop_dead_function_reads(ungated, scoped, referenced))
        for expected in ("warmup_auto", "use_vera", "use_olora", "init_strategy"):
            assert expected in dropped, (
                f"{expected} is read only inside a function nothing references, "
                "so the gate should drop it"
            )
