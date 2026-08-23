"""#478 — Transformers load kwargs must stay compatible with the declared floor.

``pyproject.toml`` declares ``transformers>=4.36.0,<5.0.0``, but CI's normal
``pip install -e ".[dev]"`` resolves to the newest 4.x. A kwarg that only exists
on >=4.56 (``dtype=`` on ``from_pretrained`` / ``from_config``, the rename of
``torch_dtype=``) therefore passes the 12-cell matrix and TypeErrors on older
installs inside the declared range — exactly what #471 nearly shipped.

``transformers==4.36.0`` cannot resolve against Soup's declared ``trl`` range
(even ``trl==0.14.0`` requires ``transformers>=4.46.0``; see the floor CI job
comments and ``.github/constraints/transformers-floor.txt``). Raising Soup's
declared floor is a dependency-policy call, not something this suite does.
Until that decision lands, the accepted #478 fallback is a static guard: no
production model ``from_pretrained`` / ``from_config`` call site governed by
the Transformers dtype contract may pass ``dtype=``.

Unsloth's ``FastLanguageModel.from_pretrained(..., dtype=)`` and Soup wrapper
kwargs that are not Transformers load APIs are out of scope.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "soup_cli"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS = REPO_ROOT / ".github" / "constraints" / "transformers-floor.txt"

_LOAD_METHODS = frozenset({"from_pretrained", "from_config"})
_MODEL_LOAD_CLASS_SUFFIXES = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "ForImageTextToText",
    "ForMaskedLM",
    "ForSequenceClassification",
    "ForSpeechSeq2Seq",
    "ForTokenClassification",
)


def _attr_parts(node: ast.AST) -> list[str]:
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return list(reversed(parts))


def _is_dtype_guarded_model_load(call: ast.Call) -> bool:
    """True for model loads governed by the Transformers dtype contract."""
    parts = _attr_parts(call.func)
    if len(parts) < 2:
        return False
    if parts[-1] not in _LOAD_METHODS:
        return False
    model_class = parts[-2]
    return (
        model_class.startswith("AutoModel")
        or model_class == "AutoProcessor"
        or model_class.endswith(_MODEL_LOAD_CLASS_SUFFIXES)
    )


def find_post_floor_dtype_kwargs(source: str, *, filename: str = "<string>") -> list[str]:
    """Return ``file:line`` hits for ``dtype=`` on guarded model load/config calls."""
    tree = ast.parse(source, filename=filename)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_dtype_guarded_model_load(node):
            continue
        for kw in node.keywords:
            if kw.arg == "dtype":
                hits.append(f"{filename}:{node.lineno}")
    return hits


def iter_production_hits() -> list[str]:
    hits: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits.extend(
            find_post_floor_dtype_kwargs(
                path.read_text(encoding="utf-8"),
                filename=rel,
            )
        )
    return hits


def _transformers_floor_job_block(text: str) -> str:
    """Return the ``transformers-floor:`` job body through the next top-level job."""
    start = text.find("\n  transformers-floor:")
    assert start != -1, "ci.yml missing transformers-floor job"
    rest = text[start + 1 :]
    # Next job at the same indent is a line starting with two spaces + name + ':'
    # after the transformers-floor block; find "\n  test:" which follows.
    end_rel = rest.find("\n  test:")
    assert end_rel != -1, "ci.yml transformers-floor job not followed by test job"
    return rest[:end_rel]


class TestTransformersFloorDtypeGuard:
    def test_no_production_transformers_model_load_uses_dtype_kwarg(self):
        hits = iter_production_hits()
        assert hits == [], (
            "Transformers model from_pretrained / from_config calls must use torch_dtype= "
            "(works across transformers>=4.36,<5). dtype= is the >=4.56-only "
            "rename and TypeErrors on older installs inside our declared range "
            f"(#478). Offending sites: {hits}"
        )

    def test_scanner_flags_a_deliberate_dtype_mutation(self):
        """CONTROL: a broken scanner that always returns [] would greenwash #478."""
        bad = (
            "from transformers import AutoModelForCausalLM\n"
            "AutoModelForCausalLM.from_pretrained('x', dtype='auto')\n"
        )
        hits = find_post_floor_dtype_kwargs(bad, filename="mutation.py")
        assert hits == ["mutation.py:2"]

    def test_scanner_flags_concrete_conditional_generation_class(self):
        bad = (
            "from transformers import WhisperForConditionalGeneration\n"
            "WhisperForConditionalGeneration.from_pretrained('x', dtype='auto')\n"
        )
        hits = find_post_floor_dtype_kwargs(bad, filename="whisper.py")
        assert hits == ["whisper.py:2"]

    def test_scanner_flags_concrete_speech_seq2seq_class(self):
        bad = (
            "from transformers import WhisperForSpeechSeq2Seq\n"
            "WhisperForSpeechSeq2Seq.from_pretrained('x', dtype='auto')\n"
        )
        hits = find_post_floor_dtype_kwargs(bad, filename="whisper.py")
        assert hits == ["whisper.py:2"]

    def test_scanner_flags_concrete_image_text_to_text_class(self):
        bad = (
            "from transformers import Qwen2VLForImageTextToText\n"
            "Qwen2VLForImageTextToText.from_config(cfg, dtype='auto')\n"
        )
        hits = find_post_floor_dtype_kwargs(bad, filename="vision.py")
        assert hits == ["vision.py:2"]

    def test_scanner_flags_concrete_token_classification_class(self):
        bad = (
            "from transformers import BertForTokenClassification\n"
            "BertForTokenClassification.from_pretrained('x', dtype='auto')\n"
        )
        hits = find_post_floor_dtype_kwargs(bad, filename="tokens.py")
        assert hits == ["tokens.py:2"]

    def test_scanner_flags_concrete_masked_lm_class(self):
        bad = (
            "from transformers import BertForMaskedLM\n"
            "BertForMaskedLM.from_config(cfg, dtype='auto')\n"
        )
        hits = find_post_floor_dtype_kwargs(bad, filename="masked_lm.py")
        assert hits == ["masked_lm.py:2"]

    def test_scanner_allows_torch_dtype(self):
        good = (
            "from transformers import AutoModelForCausalLM\n"
            "AutoModelForCausalLM.from_pretrained('x', torch_dtype='auto')\n"
        )
        assert find_post_floor_dtype_kwargs(good, filename="ok.py") == []

    def test_scanner_ignores_unsloth_and_non_auto_model_apis(self):
        noise = (
            "FastLanguageModel.from_pretrained('x', dtype=None)\n"
            "PeftModel.from_pretrained(base, 'adapter', dtype=None)\n"
            "torch.zeros(3, dtype=torch.float16)\n"
            "load_model_and_tokenizer('x', dtype='auto')\n"
        )
        assert find_post_floor_dtype_kwargs(noise, filename="noise.py") == []

    def test_scanner_sees_from_config_and_qualified_names(self):
        src = (
            "import transformers\n"
            "transformers.AutoModelForCausalLM.from_config(cfg, dtype=x)\n"
            "AutoModelForSequenceClassification.from_pretrained('m', dtype=y)\n"
        )
        hits = find_post_floor_dtype_kwargs(src, filename="q.py")
        assert hits == ["q.py:2", "q.py:3"]


class TestFloorConstraintAndWorkflowPins:
    def test_constraints_pin_lowest_resolvable_pair(self):
        """4.36.0 is documented as unresolvable; pins are 4.46.1 + trl 0.14.0."""
        text = CONSTRAINTS.read_text(encoding="utf-8")
        assert "4.36.0" in text and "not resolvable" in text.lower()
        assert "transformers==4.46.1" in text
        assert "trl==0.14.0" in text
        assert "trl==0.17.0" not in text
        # Never claim 4.55.4 (or any pre-dtype probe) is the declared floor.
        assert "declared" in text.lower() and "floor" in text.lower()

    def test_workflow_runs_pip_check_and_asserts_both_exact_versions(self):
        job = _transformers_floor_job_block(CI_WORKFLOW.read_text(encoding="utf-8"))
        assert "python -m pip check" in job
        assert 'transformers": "4.46.1"' in job or "transformers': '4.46.1'" in job
        # Hard-coded exact pins in the assertion step (not a soft parse-only check).
        assert '"4.46.1"' in job and '"0.14.0"' in job
        assert "trl" in job
        assert "-c .github/constraints/transformers-floor.txt" in job


@pytest.mark.parametrize(
    "snippet,expect_hit",
    [
        ("AutoModel.from_pretrained(m, dtype=t)", True),
        ("AutoModelForVision2Seq.from_pretrained(m, dtype=t)", True),
        ("AutoModelForCausalLM.from_config(c, dtype=t)", True),
        ("AutoModelForCausalLM.from_pretrained(m, torch_dtype=t)", False),
    ],
)
def test_scanner_parametrized_shapes(snippet: str, expect_hit: bool):
    hits = find_post_floor_dtype_kwargs(snippet + "\n", filename="p.py")
    assert bool(hits) is expect_hit
