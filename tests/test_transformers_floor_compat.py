"""Transformers floor policy and model-load keyword compatibility.

#502 moves the declared floor to Transformers 5.12.1 and #503 moves TRL to 0.29.
The dedicated CI cell installs those exact versions so the normal newest-version
matrix cannot hide a floor regression. The historical #478 static guard remains:
Soup continues using the backward-compatible ``torch_dtype=`` spelling at model
load sites throughout the supported Transformers 5.x range.

Unsloth's ``FastLanguageModel.from_pretrained(..., dtype=)`` and Soup wrapper
kwargs that are not Transformers load APIs are out of scope.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "soup_cli"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS = REPO_ROOT / ".github" / "constraints" / "transformers-floor.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

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
_VERSION = re.compile(r"\d+(?:\.\d+){1,2}")


def _declared_transformers_floor(pyproject: str) -> str:
    """Return the Transformers lower bound from the ``train`` extra."""
    train = re.search(r"^train\s*=\s*\[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    assert train, "pyproject.toml has no train dependency list"
    requirement = re.search(
        r'"transformers\s*>=\s*(?P<floor>\d+(?:\.\d+){1,2})[^\"]*"',
        train.group(1),
    )
    assert requirement, "train dependencies have no transformers >= lower bound"
    return requirement.group("floor")


def _constraint_pin(constraints: str, package: str) -> str:
    """Return one exact package pin from the floor constraints file."""
    pins = re.findall(rf"^{re.escape(package)}==([^\s#]+)\s*$", constraints, re.MULTILINE)
    assert len(pins) == 1, f"expected exactly one {package} constraint pin, found {pins}"
    return pins[0]


def _version_key(version: str) -> tuple[int, int, int]:
    assert _VERSION.fullmatch(version), f"floor version must be numeric, got {version!r}"
    parts = [int(part) for part in version.split(".")]
    parts.extend([0] * (3 - len(parts)))
    return parts[0], parts[1], parts[2]


def _validate_transformers_floor_policy(
    pyproject: str,
    constraints: str,
) -> tuple[str, str]:
    """Require the tested floor to follow, or explain its offset from, metadata."""
    declared = _declared_transformers_floor(pyproject)
    pinned = _constraint_pin(constraints, "transformers")
    assert _version_key(pinned) >= _version_key(declared), (
        f"transformers constraint {pinned} is below declared floor {declared}"
    )

    if _version_key(pinned) > _version_key(declared):
        comments = "\n".join(
            line.lstrip()[1:].strip()
            for line in constraints.splitlines()
            if line.lstrip().startswith("#")
        )
        explains_offset = re.search(
            rf"\b{re.escape(declared)}\b[^\n]*"
            r"\b(?:not resolvable|unresolvable|yanked)\b",
            comments,
            re.IGNORECASE,
        )
        assert pinned in comments and explains_offset, (
            f"transformers constraint {pinned} exceeds declared floor {declared} without "
            "a documented reason naming both versions and the resolver/yank exception"
        )
    return declared, pinned


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


def _transformers_floor_version_check(text: str) -> str:
    """Return the Python body that asserts the floor job's installed versions."""
    workflow = yaml.safe_load(text)
    steps = workflow["jobs"]["transformers-floor"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Assert installed versions match floor constraints"
    )
    start_marker = "python - <<'PY'\n"
    start = run.find(start_marker)
    assert start != -1, "floor-version step is missing its Python heredoc"
    start += len(start_marker)
    end = run.find("\nPY", start)
    assert end != -1, "floor-version step has an unterminated Python heredoc"
    return run[start:end]


def _run_transformers_floor_version_check(
    installed: dict[str, str],
    constraints: str,
) -> None:
    script = _transformers_floor_version_check(CI_WORKFLOW.read_text(encoding="utf-8"))
    with (
        patch("pathlib.Path.read_text", return_value=constraints),
        patch("importlib.metadata.version", side_effect=installed.__getitem__),
    ):
        exec(compile(script, str(CI_WORKFLOW), "exec"), {"__name__": "__main__"})


class TestTransformersFloorDtypeGuard:
    def test_no_production_transformers_model_load_uses_dtype_kwarg(self):
        hits = iter_production_hits()
        assert hits == [], (
            "Transformers model from_pretrained / from_config calls must use torch_dtype= "
            "(kept across Soup's Transformers 5.x range). Use of dtype= at these "
            "sites must be an explicit whole-policy migration "
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
    def test_constraints_pin_a_documented_resolvable_floor(self):
        text = CONSTRAINTS.read_text(encoding="utf-8")
        declared, pinned = _validate_transformers_floor_policy(
            PYPROJECT.read_text(encoding="utf-8"),
            text,
        )
        assert _version_key(pinned) >= _version_key(declared)
        _constraint_pin(text, "trl")
        _constraint_pin(text, "peft")
        _constraint_pin(text, "plotext")
        assert "declared" in text.lower() and "floor" in text.lower()

    def test_workflow_runs_pip_check_and_asserts_floor_versions(self):
        job = _transformers_floor_job_block(CI_WORKFLOW.read_text(encoding="utf-8"))
        assert "python -m pip check" in job
        assert "-c .github/constraints/transformers-floor.txt" in job
        assert "tests/test_plotext_compat.py" in job

    @pytest.mark.parametrize("package", ["transformers", "trl", "peft", "plotext"])
    def test_workflow_version_check_accepts_pins_and_rejects_mismatch(self, package: str):
        constraints = (
            "transformers==8.8.8\ntrl==9.9.9\npeft==7.7.7\nplotext==6.6.6\n"
        )
        installed = {
            name: _constraint_pin(constraints, name)
            for name in ("transformers", "trl", "peft", "plotext")
        }
        _run_transformers_floor_version_check(installed, constraints)

        installed[package] = "0.0.0"
        with pytest.raises(AssertionError, match=f"expected {package}=="):
            _run_transformers_floor_version_check(installed, constraints)


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


class TestTransformersFloorTracksDeclaredBound:
    def test_current_higher_floor_keeps_its_unresolvable_exception_documented(self):
        _validate_transformers_floor_policy(
            PYPROJECT.read_text(encoding="utf-8"),
            CONSTRAINTS.read_text(encoding="utf-8"),
        )

    def test_raising_declared_floor_without_updating_constraint_fails(self):
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        constraints = CONSTRAINTS.read_text(encoding="utf-8")
        declared = _declared_transformers_floor(pyproject)
        mutated = pyproject.replace(
            f'transformers>={declared}',
            "transformers>=99.0.0",
            1,
        )

        with pytest.raises(AssertionError, match="below declared floor"):
            _validate_transformers_floor_policy(mutated, constraints)

    def test_declared_floor_may_catch_up_to_the_tested_constraint(self):
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        constraints = CONSTRAINTS.read_text(encoding="utf-8")
        declared = _declared_transformers_floor(pyproject)
        pinned = _constraint_pin(constraints, "transformers")
        aligned = pyproject.replace(
            f"transformers>={declared}",
            f"transformers>={pinned}",
            1,
        )

        assert _validate_transformers_floor_policy(aligned, constraints) == (
            pinned,
            pinned,
        )

    def test_lowering_constraint_below_declared_floor_fails(self):
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        constraints = CONSTRAINTS.read_text(encoding="utf-8")
        pinned = _constraint_pin(constraints, "transformers")
        mutated = constraints.replace(
            f"\ntransformers=={pinned}\n",
            "\ntransformers==0.0.1\n",
            1,
        )

        with pytest.raises(AssertionError, match="below declared floor"):
            _validate_transformers_floor_policy(pyproject, mutated)

    def test_offset_requires_declared_floor_exception_explanation(self):
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        constraints = CONSTRAINTS.read_text(encoding="utf-8")
        declared = _declared_transformers_floor(pyproject)
        pinned = _constraint_pin(constraints, "transformers")
        if _version_key(pinned) == _version_key(declared):
            _validate_transformers_floor_policy(pyproject, constraints)
            return

        without_reason = re.sub(
            rf"^#.*{re.escape(declared)}.*"
            r"(?:not resolvable|unresolvable|yanked).*\n?",
            "",
            constraints,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        assert without_reason != constraints, (
            "current floor offset fixture does not contain its documented reason"
        )

        with pytest.raises(AssertionError, match="documented reason"):
            _validate_transformers_floor_policy(pyproject, without_reason)
