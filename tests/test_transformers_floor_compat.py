"""Transformers floor policy and model-load keyword compatibility.

The declared floor is Transformers 5.16.1 for Qwen4-Exp; the dedicated CI cell
also pins Torch 2.5.1 and TRL 0.29 so the normal newest-version matrix cannot
hide a floor regression. Transformers model loaders use ``dtype=`` throughout
the supported 5.x range; the historical ``torch_dtype=`` spelling is deprecated.

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


def _is_dtype_guarded_model_load(call: ast.Call, aliases: set[str] | None = None) -> bool:
    """True for model loads governed by the Transformers dtype contract."""
    parts = _attr_parts(call.func)
    if len(parts) < 2:
        return False
    if parts[-1] not in _LOAD_METHODS:
        return False
    model_class = parts[-2]
    return (
        model_class in (aliases or ())
        or model_class.startswith("AutoModel")
        or model_class == "AutoProcessor"
        or model_class.endswith(_MODEL_LOAD_CLASS_SUFFIXES)
    )


def _scope_nodes(scope: ast.AST):
    """Walk one lexical scope without mixing identically named kwargs in nested scopes."""
    yield scope
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from _scope_nodes(child)


def find_deprecated_torch_dtype_kwargs(source: str, *, filename: str = "<string>") -> list[str]:
    """Find direct and locally constructed deprecated model-load keyword arguments."""
    tree = ast.parse(source, filename=filename)
    hits: set[int] = set()
    scopes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
    ]
    for scope in scopes:
        nodes = list(_scope_nodes(scope))
        bindings: dict[str, list[ast.AST]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, []).append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    bindings.setdefault(node.target.id, []).append(node.value)

        aliases: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name, values in bindings.items():
                if name in aliases:
                    continue
                for value in values:
                    probe = ast.Call(
                        func=ast.Attribute(value=value, attr="from_pretrained"),
                        args=[],
                        keywords=[],
                    )
                    if _is_dtype_guarded_model_load(probe, aliases):
                        aliases.add(name)
                        changed = True
                        break

        def inspect_kwargs(value: ast.AST, seen: frozenset[str] = frozenset()) -> None:
            if isinstance(value, ast.Dict):
                for key, item in zip(value.keys, value.values):
                    if isinstance(key, ast.Constant) and key.value == "torch_dtype":
                        hits.add(key.lineno)
                    elif key is None:
                        inspect_kwargs(item, seen)
            elif isinstance(value, ast.Name) and value.id not in seen:
                seen = seen | {value.id}
                for bound in bindings.get(value.id, []):
                    inspect_kwargs(bound, seen)
                for node in nodes:
                    if (
                        isinstance(node, ast.Subscript)
                        and isinstance(node.ctx, ast.Store)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == value.id
                        and isinstance(node.slice, ast.Constant)
                        and node.slice.value == "torch_dtype"
                    ):
                        hits.add(node.lineno)
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "update"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == value.id
                    ):
                        for item in node.args:
                            inspect_kwargs(item, seen)
                        for kw in node.keywords:
                            if kw.arg == "torch_dtype":
                                hits.add(kw.lineno)
                            elif kw.arg is None:
                                inspect_kwargs(kw.value, seen)
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == "dict":
                    for kw in value.keywords:
                        if kw.arg == "torch_dtype":
                            hits.add(kw.lineno)
                        elif kw.arg is None:
                            inspect_kwargs(kw.value, seen)

        for node in nodes:
            if not isinstance(node, ast.Call) or not _is_dtype_guarded_model_load(node, aliases):
                continue
            for kw in node.keywords:
                if kw.arg == "torch_dtype":
                    hits.add(node.lineno)
                elif kw.arg is None:
                    inspect_kwargs(kw.value)
    return [f"{filename}:{line}" for line in sorted(hits)]


def iter_production_hits() -> list[str]:
    hits: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits.extend(
            find_deprecated_torch_dtype_kwargs(
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
    def test_no_production_transformers_model_load_uses_torch_dtype_kwarg(self):
        hits = iter_production_hits()
        assert hits == [], (
            "Transformers model from_pretrained / from_config calls must use dtype=, "
            f"not deprecated torch_dtype=. Offending sites: {hits}"
        )

    def test_scanner_flags_a_deliberate_torch_dtype_mutation(self):
        """A broken scanner must not hide reintroduced deprecated spelling."""
        bad = (
            "from transformers import AutoModelForCausalLM\n"
            "AutoModelForCausalLM.from_pretrained('x', torch_dtype='auto')\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(bad, filename="mutation.py")
        assert hits == ["mutation.py:2"]

    def test_scanner_flags_concrete_conditional_generation_class(self):
        bad = (
            "from transformers import WhisperForConditionalGeneration\n"
            "WhisperForConditionalGeneration.from_pretrained('x', torch_dtype='auto')\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(bad, filename="whisper.py")
        assert hits == ["whisper.py:2"]

    def test_scanner_flags_concrete_speech_seq2seq_class(self):
        bad = (
            "from transformers import WhisperForSpeechSeq2Seq\n"
            "WhisperForSpeechSeq2Seq.from_pretrained('x', torch_dtype='auto')\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(bad, filename="whisper.py")
        assert hits == ["whisper.py:2"]

    def test_scanner_flags_concrete_image_text_to_text_class(self):
        bad = (
            "from transformers import Qwen2VLForImageTextToText\n"
            "Qwen2VLForImageTextToText.from_config(cfg, torch_dtype='auto')\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(bad, filename="vision.py")
        assert hits == ["vision.py:2"]

    def test_scanner_flags_concrete_token_classification_class(self):
        bad = (
            "from transformers import BertForTokenClassification\n"
            "BertForTokenClassification.from_pretrained('x', torch_dtype='auto')\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(bad, filename="tokens.py")
        assert hits == ["tokens.py:2"]

    def test_scanner_flags_concrete_masked_lm_class(self):
        bad = (
            "from transformers import BertForMaskedLM\n"
            "BertForMaskedLM.from_config(cfg, torch_dtype='auto')\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(bad, filename="masked_lm.py")
        assert hits == ["masked_lm.py:2"]

    def test_scanner_allows_dtype(self):
        good = (
            "from transformers import AutoModelForCausalLM\n"
            "AutoModelForCausalLM.from_pretrained('x', dtype='auto')\n"
        )
        assert find_deprecated_torch_dtype_kwargs(good, filename="ok.py") == []

    def test_scanner_ignores_unsloth_and_non_auto_model_apis(self):
        noise = (
            "FastLanguageModel.from_pretrained('x', dtype=None)\n"
            "PeftModel.from_pretrained(base, 'adapter', dtype=None)\n"
            "torch.zeros(3, dtype=torch.float16)\n"
            "load_model_and_tokenizer('x', dtype='auto')\n"
        )
        assert find_deprecated_torch_dtype_kwargs(noise, filename="noise.py") == []

    def test_scanner_sees_from_config_and_qualified_names(self):
        src = (
            "import transformers\n"
            "transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=x)\n"
            "AutoModelForSequenceClassification.from_pretrained('m', torch_dtype=y)\n"
        )
        hits = find_deprecated_torch_dtype_kwargs(src, filename="q.py")
        assert hits == ["q.py:2", "q.py:3"]

    @pytest.mark.parametrize(
        "setup",
        [
            "kwargs = {'torch_dtype': 'auto'}",
            "kwargs = dict(torch_dtype='auto')",
            "kwargs = {}; kwargs['torch_dtype'] = 'auto'",
            "kwargs = {}; kwargs.update(torch_dtype='auto')",
            "kwargs = {}; kwargs.update({'torch_dtype': 'auto'})",
            "base = {'torch_dtype': 'auto'}; kwargs = {**base}",
        ],
    )
    def test_scanner_follows_local_load_dictionary(self, setup):
        source = setup + "\nAutoModel.from_pretrained('m', **kwargs)\n"
        assert find_deprecated_torch_dtype_kwargs(source, filename="kwargs.py")

    def test_scanner_follows_a_selected_auto_model_loader(self):
        source = (
            "loader = AutoModelForCausalLM\n"
            "chosen = loader\n"
            "kwargs = {'torch_dtype': 'auto'}\n"
            "chosen.from_pretrained('m', **kwargs)\n"
        )
        assert find_deprecated_torch_dtype_kwargs(source) == ["<string>:3"]

    def test_scanner_flags_inline_unpack(self):
        source = "AutoModel.from_config(cfg, **{'torch_dtype': 'auto'})\n"
        assert find_deprecated_torch_dtype_kwargs(source) == ["<string>:1"]

    def test_scanner_preserves_checkpoint_fields_and_other_apis(self):
        source = (
            "config = {'torch_dtype': 'bfloat16'}\n"
            "kwargs = {'dtype': 'auto'}\n"
            "AutoModel.from_pretrained('m', config=config, **kwargs)\n"
            "PeftModel.from_pretrained(base, 'adapter', torch_dtype='auto')\n"
            "FastLanguageModel.from_pretrained('m', torch_dtype='auto')\n"
        )
        assert find_deprecated_torch_dtype_kwargs(source) == []

    def test_scanner_does_not_mix_identical_names_in_different_functions(self):
        source = (
            "def checkpoint():\n    kwargs = {'torch_dtype': 'bfloat16'}\n    return kwargs\n"
            "def load():\n    kwargs = {'dtype': 'auto'}\n"
            "    return AutoModel.from_pretrained('m', **kwargs)\n"
        )
        assert find_deprecated_torch_dtype_kwargs(source) == []


class TestFloorConstraintAndWorkflowPins:
    def test_constraints_pin_a_documented_resolvable_floor(self):
        text = CONSTRAINTS.read_text(encoding="utf-8")
        declared, pinned = _validate_transformers_floor_policy(
            PYPROJECT.read_text(encoding="utf-8"),
            text,
        )
        assert _version_key(pinned) >= _version_key(declared)
        _constraint_pin(text, "torch")
        _constraint_pin(text, "trl")
        _constraint_pin(text, "peft")
        _constraint_pin(text, "plotext")
        assert "declared" in text.lower() and "floor" in text.lower()

    def test_workflow_runs_pip_check_and_asserts_floor_versions(self):
        job = _transformers_floor_job_block(CI_WORKFLOW.read_text(encoding="utf-8"))
        assert 'pip install -e ".[dev,audio]"' in job
        assert "python -m pip check" in job
        assert "-c .github/constraints/transformers-floor.txt" in job
        assert "tests/test_issue265_xcodec2_audio_contract.py" in job
        assert "tests/test_issue571_qwen4_exp.py" in job
        assert "tests/test_plotext_compat.py" in job

    @pytest.mark.parametrize(
        "package",
        ["torch", "torchaudio", "transformers", "trl", "peft", "plotext", "typer", "click"],
    )
    def test_workflow_version_check_accepts_pins_and_rejects_mismatch(self, package: str):
        # typer and click (#1802): this step is what turns "the file pins the declared
        # floor" into "the cell runs on it" -- see tests/test_cli_dependency_floors.py.
        constraints = (
            "torch==2.5.1\ntorchaudio==2.5.1\n"
            "transformers==8.8.8\ntrl==9.9.9\n"
            "peft==7.7.7\nplotext==6.6.6\n"
            "typer==5.5.5\nclick==4.4.4\n"
        )
        installed = {
            name: _constraint_pin(constraints, name)
            for name in (
                "torch",
                "torchaudio",
                "transformers",
                "trl",
                "peft",
                "plotext",
                "typer",
                "click",
            )
        }
        _run_transformers_floor_version_check(installed, constraints)

        installed[package] = "0.0.0"
        with pytest.raises(AssertionError, match=f"expected {package}=="):
            _run_transformers_floor_version_check(installed, constraints)


@pytest.mark.parametrize(
    "snippet,expect_hit",
    [
        ("AutoModel.from_pretrained(m, torch_dtype=t)", True),
        ("AutoModelForVision2Seq.from_pretrained(m, torch_dtype=t)", True),
        ("AutoModelForCausalLM.from_config(c, torch_dtype=t)", True),
        ("AutoModelForCausalLM.from_pretrained(m, dtype=t)", False),
    ],
)
def test_scanner_parametrized_shapes(snippet: str, expect_hit: bool):
    hits = find_deprecated_torch_dtype_kwargs(snippet + "\n", filename="p.py")
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
